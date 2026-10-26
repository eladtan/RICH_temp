#include "GreyCalculation.hpp"

#ifdef RICH_MPI
#include "source/monte/manager/communication/RDMACommunicationEngine.hpp"
#include "PostProcessCommunication.hpp"
#endif // RICH_MPI
#include "PostProcessResultWriter.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef __linux__
#include <malloc.h>
#endif

#ifdef RICH_MPI
#include <mpi.h>
#include "source/mpi/mpi_commands.hpp"
#endif

#include "AdaptiveStatistics.hpp"
#include "FluxSourceCalculation.hpp"
#include "WeightSplit.hpp"
#include "CoolingLimit.hpp"
#include "PostProcessRestart.hpp"
#include "source/3D/radiation/IMCMeasuredLoadBalance.hpp"
#include "source/3D/radiation/IMCStepCounterCostCalculator.hpp"


namespace imc_postprocess_tde {

ForwardPostprocessResult RunGreyPostprocess(
    Config const& cfg,
    PostprocessRuntime& runtime,
    ForwardPostprocessResult const& forwardResult)
{
    ForwardPostprocessResult result;
    int const rank = runtime.rank;
    int const mpiSize = runtime.mpiSize;
    auto& tess = runtime.tess;
    auto& cells = runtime.cells;
    auto& extensives = runtime.extensives;
    auto& eos = runtime.eos;
    auto& opacity = runtime.opacity;
    auto& greyOpacity = runtime.greyOpacity;
    auto& observer = runtime.observer;
    auto& boundary = runtime.boundary;
    auto& popControl = runtime.popControl;
    auto& physics = runtime.physics;
    auto& manager = runtime.manager;
    size_t Ncells = runtime.nCells;

            using Particle3D = MonteCarloParticle<Vector3D>;

            SphericalObserver::PhotosphereData greyPhotosphere;
            bool hasGreyPhotosphere = false;
            if (observer && observer->hasPhotosphereData()) {
                greyPhotosphere = observer->getPhotosphereData();
                greyPhotosphere.clearMG();
                hasGreyPhotosphere = greyPhotosphere.hasGrey() || greyPhotosphere.hasEmissionSurface();
            }

            // ============================================================
            // Release MG objects before grey run to avoid OOM
            // ============================================================
            manager.reset();
            physics.reset();
            popControl.reset();
            observer.reset();
            boundary.reset();
            opacity.reset();
            if (rank == 0)
                std::cout << "MG resources released." << std::endl;

            // ============================================================
            // Grey IMC run (half generations)
            // ============================================================
            size_t nGreyGens = GreyGenerationCount(cfg);
            size_t greyBudgetDivisor = std::max<size_t>(
                1, cfg.fluxSourceCompare ? nGreyGens : 2 * nGreyGens);
            size_t greyPhotonsPerCell = std::max<size_t>(1, cfg.photonsPerCell / greyBudgetDivisor);

            if (rank == 0)
                std::cout << "\n=== Starting Grey IMC run (" << nGreyGens << " generations) ===" << std::endl;

            // The grey pass's own cooling limit, with the grey Planck mean,
            // when the multigroup one used the group opacities; the
            // escape-origin classes follow its temperatures.
            ApplyGreyPassCoolingLimit(cfg, runtime);
            if (runtime.greyCoolingLimit.applied)
                BuildEscapeOriginHotCells(cfg, runtime);

            auto greyObserver = std::make_shared<SphericalObserver>(
                cfg.center, cfg.radius, cfg.nObservers, std::vector<double>());
            if (hasGreyPhotosphere)
                greyObserver->setPhotosphereData(std::move(greyPhotosphere));
            if (cfg.fluxSourceCompare && cfg.volumeEmissionEnabled)
                greyObserver->setEscapeOriginClassification(
                    &runtime.fluxSourceCellIDs, &runtime.escapeOriginHotCells);

            auto greyBoundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(tess);

            RadiationIMCParameters greyParams;
            greyParams.newPhotonsPerCell = greyPhotonsPerCell;
            greyParams.withHydro = false;
            greyParams.noHydroFeedback = true;
            greyParams.withRandomWalk = cfg.randomWalk && !cfg.fluxSourceCompare;
            greyParams.rwMinCellOpticalDepth = 15;
            greyParams.withDDMC = cfg.ddmc;
            greyParams.ddmcMinCellOpticalDepth = cfg.ddmcMinCellOpticalDepth;
            greyParams.weightCutoffFraction = cfg.weightCutoffFraction;
            greyParams.ddmcUseMovingInterfaceCorrection = cfg.ddmcMovingInterfaceCorrection;
            greyParams.ddmcAdvection = cfg.ddmcAdvection;
            greyParams.ddmcMaxCellAdvectionPeclet = cfg.ddmcMaxCellAdvectionPeclet;
            greyParams.ddmcExternalSourceMinFaceOpticalDepth =
                cfg.fluxSourceDDMCFaceOpticalDepth;
            greyParams.withMultigroupDDMC = false;
            greyParams.MMC = false;
            greyParams.diffusionPressureGradient = false;
            greyParams.withMultigroupOpacity = false;
            greyParams.withCompton = false;
            greyParams.postProcess.enabled = true;
            greyParams.postProcess.sourceDt = cfg.sourceDt;
            greyParams.postProcess.transportTime = cfg.transportTime;
            greyParams.postProcess.useCellVelocities = cfg.useCellVelocities;
            greyParams.postProcess.compressionMaxDynamicalTimes = cfg.ddmcCompressionMaxDynamicalTimes;
            greyParams.postProcess.compressionCenter = {cfg.center.x, cfg.center.y, cfg.center.z};
            greyParams.postProcess.polarization.enabled = cfg.polarization;
            greyParams.postProcess.polarization.manualScatteringsAfterAcceleration = cfg.polarizationManualScatterings;
            greyParams.postProcess.polarization.depolarizationScatterings = cfg.polarizationDepolarizationScatterings;
            greyParams.postProcess.polarization.acceleratedClosure = cfg.polarizationClosure;

            auto greyPhysics = std::make_shared<RadiationIMC>(
                tess, greyBoundary, cells, extensives, eos, greyOpacity, greyParams);
            greyPhysics->setObserver(greyObserver);
            if(cfg.fluxSourceCompare)
                ConfigureFluxSourceForCurrentDecomposition(
                    cfg, runtime, *greyPhysics, *greyOpacity, false);

            auto greyPopControl = std::make_shared<STORM::NoPopulationControl<Vector3D, Tessellation3D>>(tess);

            std::shared_ptr<MonteCarloManager3D> greyManager;
    #ifdef RICH_MPI
            MonteCarloConfig monteCarloConfig;
            std::unique_ptr<STORM::CommunicationEngine<Vector3D>> engine =
                MakeCommunicationEngine(
                tess, monteCarloConfig, cfg.communication, MPI_COMM_WORLD);
            greyManager = std::make_shared<MonteCarloManager3D>(
                tess, greyPhysics, greyPopControl, greyBoundary, monteCarloConfig, std::move(engine));
    #else
            greyManager = std::make_shared<MonteCarloManager3D>(
                tess, greyPhysics, greyPopControl, greyBoundary);
    #endif

            bool const greyMeasuredLBActive =
                cfg.measuredLoadBalance && (cfg.adaptiveSourceCells || nGreyGens > 1);

            if (rank == 0)
                std::cout << "Grey measured LB active: " << (greyMeasuredLBActive ? "yes" : "no") << std::endl;

            imc_measured_lb::Parameters greyLBParams;
            greyLBParams.floorCost = 1.0;
            greyLBParams.stepWeight = 1.0;
            greyLBParams.particleWeight = cfg.adaptiveSourceCells ? 1.0 : 0.0;
            greyLBParams.medianClampFactor = 30.0;
            greyLBParams.missingCellCost = 2.0;
            greyLBParams.grayZeroStepInflation = 2.0;
            greyLBParams.multigroupZeroStepInflation = 5.0;
            greyLBParams.maxCellImbalance = MEASURED_LB_MAX_CELL_IMBALANCE;
            greyLBParams.useMedianClamp = true;
            greyLBParams.clampToActiveMean = cfg.measuredLBClampToActiveMean;
            if (cfg.measuredLBClampFactor > 0.0)
                greyLBParams.medianClampFactor = cfg.measuredLBClampFactor;
            double const greyLBWeightCompression = EffectiveMeasuredLBWeightCompression(cfg);

            AdaptiveSourceState greyAdaptive;
            WeightSplitState greySplit;
            bool greyBurninMeasuredLBDone = false;
            bool greyFirstNonBurninMeasuredLBDone = false;
            greyObserver->clearGenerationStatistics();
            size_t greyIncludedFinalGenerations = 0;
            size_t greyDiscardedBurninGenerations = 0;

            // A restart continues the statistics and learned state of the file
            // and never reruns the burn-in or the learned-only probe.
            bool const restarting = !cfg.restartFrom.empty();
            size_t greyCompletedFinalGenerations = 0;
            LoadBalanceMeasurements greyRestartLoadBalance;   // rank 0
            if (restarting) {
                PassRestartState greyLoaded = BeginPassRestart(
                    cfg, RestartPass::Grey, *greyObserver, nGreyGens, rank);
                greyAdaptive = std::move(greyLoaded.adaptive);
                runtime.lastEscapingLuminosity = greyLoaded.lastEscapingLuminosity;
                greyCompletedFinalGenerations = greyLoaded.completedFinalGenerations;
                greyRestartLoadBalance = std::move(greyLoaded.loadBalance);
                // The restart balance below replaces the first-final-step one.
                greyFirstNonBurninMeasuredLBDone = true;
            }
            // Touched cells of this segment's last final generation (local).
            LoadBalanceMeasurements greyLastFinalMeasurements;
            size_t const greyInitialBurninGenerations = cfg.adaptiveSourceCells ? 1 : 0;
            size_t const greyUniformBurninGenerations = cfg.adaptiveSourceCells ? 19 : 0;
            size_t const greyBurninGenerations = greyInitialBurninGenerations + greyUniformBurninGenerations;
            size_t const greyLearnedProbeGenerations = cfg.adaptiveSourceCells ? 1 : 0;
            size_t const greyProbePhotons = cfg.volumeEmissionEnabled ? 10 : 75;
            size_t const greyProbeEndGeneration = greyBurninGenerations + greyLearnedProbeGenerations;
            // The grey pass may use its own learned-cell packet budget.
            Config greyAllocationCfg = cfg;
            if (cfg.volumeEmissionLearnedPhotonsPerCellBudgetGrey > 0)
                greyAllocationCfg.volumeEmissionLearnedPhotonsPerCellBudget =
                    cfg.volumeEmissionLearnedPhotonsPerCellBudgetGrey;
            // Final-budget ramp between the probe and the final generations
            // (a restart starts at the final generations and skips it).
            double const greyRampFullBudget = static_cast<double>(std::max(
                greyAllocationCfg.adaptiveSourceLearnedPhotonsPerCellBudget,
                cfg.volumeEmissionEnabled ? greyAllocationCfg.volumeEmissionLearnedPhotonsPerCellBudget
                                          : size_t(0)));
            std::vector<double> const greyRampBudgets = cfg.adaptiveSourceCells
                ? FinalRampStageBudgets(cfg, greyRampFullBudget) : std::vector<double>();
            size_t const greyRampGenerations = greyRampBudgets.size();
            size_t const greyFinalStartGeneration = greyProbeEndGeneration + greyRampGenerations;
            size_t const greyTotalGenerations = cfg.adaptiveSourceCells
                ? greyFinalStartGeneration + nGreyGens
                : nGreyGens;
    #ifdef RICH_MPI
            // Repartitions on one generation's per-cell steps and source
            // packets (predicted packets before the probe): the rebalances in
            // the loop, and on a restart the initial one from the stored
            // last-generation measurements.
            auto greyRebalanceOnMeasurements = [&](std::vector<size_t> const& greyLocalSteps,
                                                   std::vector<size_t> const& greyLocalParticles,
                                                   std::string const& greyLBLabel,
                                                   bool predictedProbe)
            {
                if (!greyParams.noHydroFeedback) {
                    throw UniversalError("Grey measured load balance repartition requires noHydroFeedback=true");
                }

                PrintVmRSS("grey_before_measured_lb", rank);

                std::vector<double> greyWeightsForExchange;
                // Set on every rank once the mesh was rebuilt; a rank that received
                // no cells has an empty weight vector but must still rebuild its
                // physics and join the collectives below.
                bool meshRepartitioned = false;
                imc_measured_lb::Parameters greyLBParamsThisPass =
                    greyLBParams;

                {
                    std::vector<size_t> greyCellIDs(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        greyCellIDs[i] = cells[i].ID;

                    auto greyLocalMeas = imc_measured_lb::BuildLocalMeasurements(
                        greyCellIDs, greyLocalSteps, greyLocalParticles);

                    uint64_t greyLocalTotalSteps = 0;
                    uint64_t greyLocalTotalSourceParticles = 0;
                    for (auto const& m : greyLocalMeas) {
                        greyLocalTotalSteps += static_cast<uint64_t>(m.stepCount);
                        greyLocalTotalSourceParticles +=
                            static_cast<uint64_t>(m.particleCount);
                    }

                    uint64_t greyGlobalTotalSteps = 0;
                    uint64_t greyGlobalTotalSourceParticles = 0;
                    MPI_Allreduce(&greyLocalTotalSteps, &greyGlobalTotalSteps, 1,
                                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(&greyLocalTotalSourceParticles,
                                  &greyGlobalTotalSourceParticles, 1,
                                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

                    if (cfg.fluxSourceCompare &&
                        greyGlobalTotalSteps > 0 &&
                        greyGlobalTotalSourceParticles > 0) {
                        greyLBParamsThisPass.particleWeight = std::max(
                            1.0,
                            static_cast<double>(greyGlobalTotalSteps) /
                                static_cast<double>(greyGlobalTotalSourceParticles));
                        greyLBParamsThisPass.particleCostAfterClamp = true;
                        if (rank == 0) {
                            std::cerr
                                << "MEASURED_LB_GREY_CER_SOURCE_COST"
                                << " global_steps=" << greyGlobalTotalSteps
                                << " global_source_particles="
                                << greyGlobalTotalSourceParticles
                                << " steps_per_source_particle="
                                << greyLBParamsThisPass.particleWeight
                                << " post_clamp=yes\n";
                        }
                    }
                    if (predictedProbe) {
                        // Exploration packets of thick cells are absorbed in about
                        // one step; their cost is creation and memory, so weight
                        // each predicted packet as one step, after the clamp.
                        greyLBParamsThisPass.particleWeight = 1.0;
                        greyLBParamsThisPass.particleCostAfterClamp = true;
                    }

                    if (greyGlobalTotalSteps == 0) {
                        if (rank == 0)
                            std::cerr << greyLBLabel
                                      << ": measured generation had zero total steps, skipping repartition\n";
                    } else {
                        auto greyCostByCellID = imc_measured_lb::BuildMeasuredCosts(
                            greyLocalMeas, greyLBParamsThisPass,
                            false, MPI_COMM_WORLD);

                        imc_measured_lb::PrintMeasuredLBDiagnosticsDistributed(
                            greyLocalMeas, greyCostByCellID, false, MPI_COMM_WORLD);

                        IMCStepCounterCostCalculator::Parameters greyCostCalcParams;
                        greyCostCalcParams.floorCost =
                            greyLBParamsThisPass.floorCost;
                        greyCostCalcParams.missingCellCost =
                            greyLBParamsThisPass.missingCellCost;
                        IMCStepCounterCostCalculator greyCostCalc(std::move(greyCostByCellID), greyCostCalcParams);

                        std::vector<Vector3D> greyCurrentPoints(Ncells);
                        for (size_t i = 0; i < Ncells; ++i)
                            greyCurrentPoints[i] = tess.GetMeshPoint(i);

                        auto greyLBWeights = greyCostCalc.CalculateCost(tess, cells);

                        if (greyLBWeights.size() != Ncells) {
                            throw UniversalError("Grey measured LB weight count mismatch before BuildParallel");
                        }

                        greyWeightsForExchange = greyLBWeights;

                        for (auto& w : greyLBWeights)
                            w = std::pow(w, greyLBWeightCompression);

                        // Release the pre-repartition transport objects before the
                        // rebuild (the memory peak). The boundary's escaped energy is
                        // folded into the observer first, since the boundary goes too.
                        greyObserver->addBoxEscapeEnergy(greyBoundary->getEscapedEnergy());
                        greyManager.reset();
                        greyPhysics.reset();
                        greyPopControl.reset();
                        greyBoundary.reset();
#ifdef __linux__
                        malloc_trim(0);
#endif
                        PrintVmRSS("grey_after_release_old_physics", rank);

                        tess.BuildParallel(greyCurrentPoints, greyLBWeights);
                        meshRepartitioned = true;
                        tess.ShrinkToFit();
                    }
                }

                if (meshRepartitioned) {
                    MPI_exchange_data(tess, cells, false, 1, &runtime.dummyCell);
                    MigrateWeightSplitState(runtime, greySplit);

                    double greyDummyWeight =
                        greyLBParamsThisPass.missingCellCost;
                    MPI_exchange_data(tess, greyWeightsForExchange, false, 1, &greyDummyWeight);

                    Ncells = tess.GetPointNo();

                    if (cells.size() != Ncells) {
                        UniversalError eo("Cell count mismatch after grey measured LB repartition");
                        eo.addEntry("cells.size()", static_cast<double>(cells.size()));
                        eo.addEntry("Ncells", static_cast<double>(Ncells));
                        throw eo;
                    }

                    if (greyWeightsForExchange.size() != Ncells) {
                        UniversalError eo("Grey measured weight count mismatch after repartition");
                        eo.addEntry("weights.size()", static_cast<double>(greyWeightsForExchange.size()));
                        eo.addEntry("Ncells", static_cast<double>(Ncells));
                        throw eo;
                    }

                    imc_measured_lb::PrintPostRepartitionDiagnosticsFromWeights(
                        greyWeightsForExchange, greyLBWeightCompression,
                        false, MPI_COMM_WORLD);

                    cells.shrink_to_fit();
                    extensives.resize(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);
                    extensives.shrink_to_fit();
#ifdef __linux__
                    malloc_trim(0);
#endif

                    // Escaped energy was folded into the observer before the release.
                    greyBoundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(tess);

                    greyPhysics = std::make_shared<RadiationIMC>(
                        tess, greyBoundary, cells, extensives, eos, greyOpacity, greyParams);
                    greyPhysics->setObserver(greyObserver);
                    if(cfg.fluxSourceCompare)
                        ConfigureFluxSourceForCurrentDecomposition(
                            cfg, runtime, *greyPhysics, *greyOpacity, false);

                    greyPopControl = std::make_shared<STORM::NoPopulationControl<Vector3D, Tessellation3D>>(tess);

                    std::unique_ptr<STORM::CommunicationEngine<Vector3D>> rebuiltEngine =
                        MakeCommunicationEngine(
            tess, monteCarloConfig, cfg.communication, MPI_COMM_WORLD);
                    greyManager = std::make_shared<MonteCarloManager3D>(
                        tess, greyPhysics, greyPopControl, greyBoundary,
                        monteCarloConfig, std::move(rebuiltEngine));

                    if (rank == 0)
                        std::cout << greyLBLabel
                                  << ": repartitioned, new local cells=" << Ncells << std::endl;

                    PrintVmRSS("grey_after_rebuild_physics", rank);
                }
            };
    #endif // RICH_MPI

    #ifdef RICH_MPI
            if (restarting && greyMeasuredLBActive && nGreyGens > 0) {
                RebalanceFromStoredMeasurements(
                    greyRestartLoadBalance, cells, Ncells, "Grey",
                    [&](std::vector<size_t> const& steps, std::vector<size_t> const& particles) {
                        greyRebalanceOnMeasurements(steps, particles, "MEASURED_LB_GREY_RESTART", false);
                    }, rank);
            }
    #endif // RICH_MPI
            size_t const greyStartGeneration = restarting ? greyFinalStartGeneration : 0;
#ifdef RICH_MPI
            ResetPeakRSS();   // MEMORY_GEN peak_* = one whole generation each
#endif
            for (size_t gen = greyStartGeneration; gen < greyTotalGenerations; ++gen) {
                bool const greyFirstBurninThisGen =
                    cfg.adaptiveSourceCells && gen < greyInitialBurninGenerations;
                bool const greyUniformBurninThisGen =
                    cfg.adaptiveSourceCells &&
                    gen >= greyInitialBurninGenerations &&
                    gen < greyBurninGenerations;
                bool const greyBurninThisGen =
                    greyFirstBurninThisGen || greyUniformBurninThisGen;
                bool const greyLearnedProbeThisGen =
                    cfg.adaptiveSourceCells &&
                    gen >= greyBurninGenerations &&
                    gen < greyProbeEndGeneration;
                bool const greyRampThisGen =
                    cfg.adaptiveSourceCells &&
                    gen >= greyProbeEndGeneration &&
                    gen < greyFinalStartGeneration;
                size_t const greyRampStage = greyRampThisGen ? gen - greyProbeEndGeneration : 0;
                // The extra packets (learned-weight cap, exploration splitting)
                // follow the ramp through their weight bound.
                double const greyRampWeightScale = greyRampThisGen
                    ? greyRampFullBudget / greyRampBudgets[greyRampStage] : 1.0;
                bool const greyFinalThisGen =
                    !cfg.adaptiveSourceCells || gen >= greyFinalStartGeneration;
                size_t const greyFinalGenerationIndex = greyFinalThisGen
                    ? (cfg.adaptiveSourceCells ? gen - greyFinalStartGeneration : gen)
                    : 0;
                bool const greyAdaptiveActiveThisGen =
                    cfg.adaptiveSourceCells &&
                    (greyLearnedProbeThisGen || greyRampThisGen || greyFinalThisGen) &&
                    !greyAdaptive.scoreByCellID.empty();
                size_t const greyPhotonsThisGen = greyFirstBurninThisGen ? 1
                    : (greyUniformBurninThisGen ? 3
                       : (greyLearnedProbeThisGen ? greyProbePhotons
                          : (cfg.adaptiveSourceCells ? 1 : greyPhotonsPerCell)));
                std::string greyPhase = "final";
                if (greyFirstBurninThisGen)
                    greyPhase = "burnin_exact1";
                else if (greyUniformBurninThisGen)
                    greyPhase = "burnin_exact3";
                else if (greyLearnedProbeThisGen)
                    greyPhase = "learned_only_probe_exact" + std::to_string(greyProbePhotons);
                else if (greyRampThisGen)
                    greyPhase = "final_budget_ramp";
                else if (cfg.adaptiveSourceCells)
                    greyPhase = "learned_only_final";
                greyPhysics->setNewPhotonsPerCell(greyPhotonsThisGen);
                if (rank == 0)
                    std::cout << "Grey generation " << (gen + 1) << "/" << greyTotalGenerations
                              << " phase=" << greyPhase
                              << " photons_per_cell_this_gen=" << greyPhotonsThisGen;
                if (rank == 0 && greyRampThisGen)
                    std::cout << " ramp_stage=" << (greyRampStage + 1) << "/" << greyRampGenerations
                              << " packets_per_learned_cell=" << greyRampBudgets[greyRampStage]
                              << " of " << greyRampFullBudget
                              << " weight_bound_scale=" << greyRampWeightScale;
                if (rank == 0 && greyFinalThisGen)
                    std::cout << " final_step=" << (greyFinalGenerationIndex + 1)
                              << "/" << nGreyGens;
                if (rank == 0)
                    std::cout << std::endl;
                PrintAdaptiveGenerationStart("Grey", cfg, greyAdaptive, gen, greyTotalGenerations,
                                             greyBurninGenerations,
                                             greyAdaptiveActiveThisGen, rank);

                IMCPostProcessControl postProcessControl;
                if (greyAdaptiveActiveThisGen && (greyFinalThisGen || greyRampThisGen)) {
                    // Final and ramp generations: Neyman allocation at the
                    // grey budgets, or at the ramp stage's.
                    Config const allocationCfg = greyRampThisGen
                        ? RampStageAllocationConfig(greyAllocationCfg, greyRampBudgets[greyRampStage])
                        : greyAllocationCfg;
                    SourceAllocationPlan const plan = BuildNeymanSourceAllocation(
                        greyAdaptive.scoreByCellID, allocationCfg, &runtime.fluxSourceCellIDs);
                    ApplyNeymanAllocationToControl(postProcessControl, plan, allocationCfg);
                    PrintSourceAllocationPlan("Grey", plan, gen, rank);
                } else if (greyAdaptiveActiveThisGen) {
                    postProcessControl.adaptiveCells.enabled = true;
                    postProcessControl.adaptiveCells.scores =
                        greyAdaptive.scoreByCellID;
                    postProcessControl.adaptiveCells.strength =
                        cfg.adaptiveSourceStrength;
                    postProcessControl.adaptiveCells.maxFactor =
                        cfg.adaptiveSourceMaxFactor;
                    postProcessControl.adaptiveCells.learnedReserveFraction =
                        cfg.adaptiveSourceLearnedReserveFrac;
                    postProcessControl.adaptiveCells.learnedMinFactor = 1.0;
                    postProcessControl.adaptiveCells.observerBudgetMultiplier =
                        greyAdaptive.observerBudgetMultiplier;
                    postProcessControl.adaptiveCells.learnedMinPhotons = 0;
                    postProcessControl.adaptiveCells.learnedMaxPhotons = 0;
                    postProcessControl.adaptiveCells.scorePower = 1.0;
                }
                if (greyFirstBurninThisGen) {
                    postProcessControl.emission.enabled = true;
                    postProcessControl.emission.includeUniformBase = true;
                    postProcessControl.emission.baseMultiplier = 1;
                } else if (greyUniformBurninThisGen) {
                    postProcessControl.emission.enabled = true;
                    postProcessControl.emission.includeUniformBase = true;
                    postProcessControl.emission.baseMultiplier = 3;
                } else if (greyLearnedProbeThisGen) {
                    postProcessControl.emission.enabled = true;
                    postProcessControl.emission.useLearnedScores = true;
                    postProcessControl.emission.includeUniformBase = false;
                    postProcessControl.emission.learnedBoostFactor = 1;
                } else if (cfg.adaptiveSourceCells && (greyFinalThisGen || greyRampThisGen) &&
                           !greyAdaptiveActiveThisGen) {
                    postProcessControl.emission.enabled = true;
                    postProcessControl.emission.useLearnedScores = true;
                    postProcessControl.emission.includeUniformBase = false;
                    postProcessControl.emission.learnedBoostFactor = 1;
                    postProcessControl.emission.learnedExtraBudget = 0;
                }
                greyPhysics->configurePostProcessControl(
                    std::move(postProcessControl));
                if (cfg.fluxSourceCompare)
                    ConfigureFluxSourceForCurrentDecomposition(
                        cfg, runtime, *greyPhysics, *greyOpacity, false);
                greyObserver->resetGenerationSourceCellEscapeStats();

                if (cfg.volumeEmissionEnabled && greyPhysics->hasPostProcessVolumeEmission()) {
                    double fraction = 1.0;
                    if (greyBurninThisGen && runtime.volumeEmissionCells > 0) {
                        double const wanted = static_cast<double>(cfg.volumeEmissionBurninPacketsTarget);
                        double const full = static_cast<double>(greyPhotonsThisGen) *
                                            static_cast<double>(runtime.volumeEmissionCells);
                        fraction = std::clamp(wanted / std::max(1.0, full), 1e-6, 1.0);
                    }
                    greyPhysics->setPostProcessVolumeEmissionSubsample(fraction, static_cast<uint64_t>(gen) + 1);
                    double const escapingReference = runtime.lastEscapingLuminosity > 0.0
                        ? runtime.lastEscapingLuminosity : runtime.fluxSourceInjectedLuminosity;
                    greyPhysics->setPostProcessExplorationMaxWeight(
                        cfg.volumeEmissionExplorationWeightFractionGrey * escapingReference * cfg.sourceDt *
                        greyRampWeightScale);
                    // Probe, ramp and final generations bound the learned
                    // cells' packets the same way; the burn-in keeps its flat
                    // counts.
                    greyPhysics->setPostProcessLearnedMaxWeight(
                        cfg.volumeEmissionCapLearnedWeight && !greyBurninThisGen
                            ? cfg.volumeEmissionExplorationWeightFractionGrey * escapingReference * cfg.sourceDt *
                              greyRampWeightScale
                            : 0.0);
                }

                if (restarting)
                    greyPhysics->reseedRNG(RestartGenerationSeed(
                        RestartPass::Grey, rank,
                        greyCompletedFinalGenerations + greyFinalGenerationIndex));
                else
                    greyPhysics->reseedRNG(static_cast<uint64_t>(rank + 87654321) * greyTotalGenerations + gen);

                greyManager->getParticles().clear();
                InstallWeightSplit(cfg, runtime, *greyPhysics, greySplit);
                greyManager->step(cells, cfg.transportTime);
                if (cfg.volumeEmissionEnabled && greyPhysics->hasPostProcessVolumeEmission()) {
                    double volumeEnergy = greyPhysics->getLastVolumeEmissionEnergy();
                    unsigned long long counts[2] = {
                        static_cast<unsigned long long>(greyPhysics->getLastVolumeEmissionCells()),
                        static_cast<unsigned long long>(greyPhysics->getLastVolumeEmissionSelectedCells())};
#ifdef RICH_MPI
                    MPI_Allreduce(MPI_IN_PLACE, &volumeEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
                    if (rank == 0)
                        std::cout << "Grey VOLUME_EMISSION gen=" << gen + 1
                                  << " emitting_cells=" << counts[0]
                                  << " selected_cells=" << counts[1]
                                  << " volume_luminosity=" << volumeEnergy / cfg.sourceDt
                                  << " face_luminosity=" << runtime.fluxSourceInjectedLuminosity
                                  << " erg/s" << std::endl;
                }
#ifdef RICH_MPI
                ReportGenerationMemory("Grey", gen + 1, rank, runtime.mpiSize, tess, greyManager->getParticles().size());
#endif
#ifdef __linux__
                malloc_trim(0);
#endif
                IMCPostProcessGenerationDiagnostics const generationDiagnostics =
                    greyPhysics->getPostProcessGenerationDiagnostics();
                if (greyFinalThisGen && gen + 1 == greyTotalGenerations)
                    greyLastFinalMeasurements = CollectLocalLoadBalanceMeasurements(
                        cells, Ncells, greyManager->GetCellsStepsCounters(),
                        generationDiagnostics.sourcePhotonsPerCell);

                auto greyAllocation = ReduceSourceAllocationSummary(
                    generationDiagnostics.sourceAllocation);
                result.weightCapExtraPhotons += greyAllocation.weightCapExtraPhotons;
                result.weightCappedCells += greyAllocation.weightCappedCells;
                result.allocatedCells += greyAllocation.sourceCells;
                if (greyLearnedProbeThisGen)
                    result.probePackets += greyAllocation.totalPhotons;
                size_t const photonHistMin = greyFinalThisGen
                    ? cfg.adaptiveSourceLearnedMinPhotons
                    : 1;
                size_t const photonHistMax = greyFinalThisGen
                    ? cfg.adaptiveSourceLearnedMaxPhotons
                    : std::max(photonHistMin, greyPhotonsThisGen * 20);
                auto greyPhotonDistribution = ReduceSourcePhotonDistribution(
                    generationDiagnostics.sourcePhotonsPerCell,
                    photonHistMin,
                    photonHistMax,
                    rank,
                    mpiSize);
                auto greySourceStats = greyObserver->getGenerationSourceCellEscapeStats();
                greyObserver->resetGenerationSourceCellEscapeStats();
                ObserverQualityDiagnostics greyObserverQuality;
                if (cfg.adaptiveSourceCells && cfg.adaptiveObserverEquity) {
                    greyObserverQuality = BuildObserverQualityDiagnostics(
                        CollectGlobalObserverQuality(greyObserver->getObserverQualitySnapshot()),
                        cfg, greyAdaptive, greyFinalThisGen);
                }
                auto greyUpdate = UpdateAdaptiveSourceScoresDistributed(
                    greySourceStats,
                    BuildLocalSourceCellEmission(
                        cells, generationDiagnostics.sourcePhotonsPerCell),
                    cfg, greyAdaptive, greyObserverQuality,
                    !greyBurninThisGen, rank, mpiSize);
                if (cfg.sourceDt > 0.0 && greyUpdate.totalEscapedEnergy > 0.0)
                    runtime.lastEscapingLuminosity = greyUpdate.totalEscapedEnergy / cfg.sourceDt;
                if (!greyBurninThisGen)
                    AccumulateWeightSplitStatistics(cfg, runtime, greyUpdate, greySplit);
                std::vector<SphericalObserver::SourceCellEscapeStat>().swap(greySourceStats);
                bool const greyIncludeGenerationInFinal = greyFinalThisGen;
                PrintAdaptiveGenerationStats(
                    "Grey", cfg, greyAdaptive, greyUpdate, greyAllocation,
                    greyPhotonDistribution,
                    greyObserverQuality, gen, greyTotalGenerations,
                    greyBurninGenerations, greyAdaptiveActiveThisGen, rank);
                PrintAdaptiveIterationSummary(
                    "Grey", greyAdaptive, greyUpdate, greyAllocation,
                    greyObserverQuality, gen, greyTotalGenerations, greyPhase,
                    greyPhotonsThisGen, greyFinalThisGen,
                    greyFinalGenerationIndex, nGreyGens,
                    greyAdaptiveActiveThisGen, greyIncludeGenerationInFinal,
                    rank);
                if (greyFinalThisGen && cfg.adaptiveSourceCells && cfg.adaptiveObserverEquity) {
                    bool const printBlock =
                        (greyFinalGenerationIndex + 1) % 5 == 0 ||
                        greyFinalGenerationIndex + 1 == nGreyGens;
                    PrintAdaptiveConvergenceReport(
                        "Grey", cfg, greyObserverQuality, greyFinalGenerationIndex,
                        nGreyGens, printBlock, rank);
                }
                if (rank == 0 && cfg.adaptiveSourceCells)
                    std::cout << "Grey learned cells after iteration " << (gen + 1)
                              << ": " << greyAdaptive.scoreByCellID.size() << std::endl;
                greyObserver->addBoxEscapeEnergy(greyBoundary->getEscapedEnergy());
                greyBoundary->resetEscapedEnergy();
                greyObserver->printHeavyPacketReport("grey", gen + 1, 10);
                greyObserver->printEscapeOriginReport("grey", gen + 1);
                result.splitCopies += PrintWeightSplitReport(cfg, *greyPhysics, greySplit, "grey", gen + 1, rank);
                PrintCompressionReport(cfg, *greyPhysics, "grey", gen + 1, rank);
                greyObserver->mpiReduceToRank0();
                if (greyIncludeGenerationInFinal) {
                    if (rank == 0)
                        greyObserver->accumulateCurrentTalliesForStatistics(cfg.sourceDt);
                    ++greyIncludedFinalGenerations;
                } else {
                    ++greyDiscardedBurninGenerations;
                }
                greyObserver->resetTallies();
                if (cfg.adaptiveSourceCells && !greyAdaptive.burninCompletePrinted &&
                    greyBurninGenerations > 0 &&
                    gen + 1 == greyBurninGenerations)
                {
                    if (rank == 0)
                        std::cout << "Grey adaptive source burn-in complete" << std::endl;
                    greyAdaptive.burninCompletePrinted = true;
                }

    #ifdef RICH_MPI
                RankStepImbalance const greyStepImbalance =
                    ComputeRankStepImbalance("Grey", gen, greyManager->GetCellsStepsCounters(), rank);
                bool const greyDoBurninMeasuredLB =
                    !cfg.fluxSourceCompare &&
                    greyMeasuredLBActive &&
                    cfg.adaptiveSourceCells &&
                    greyFirstBurninThisGen &&
                    !greyBurninMeasuredLBDone;
                bool const greyDoInitialMeasuredLB =
                    gen == 0 && greyMeasuredLBActive && !restarting &&
                    (cfg.fluxSourceCompare || !cfg.adaptiveSourceCells);
                bool const greyDoPostAdaptiveMeasuredLB =
                    greyMeasuredLBActive &&
                    cfg.adaptiveSourceCells &&
                    !greyBurninThisGen &&
                    !greyFirstNonBurninMeasuredLBDone;
                bool const greyDoAdaptivePeriodicMeasuredLB =
                    greyMeasuredLBActive &&
                    cfg.adaptiveSourceCells &&
                    greyFinalThisGen &&
                    greyFinalGenerationIndex + 1 < nGreyGens &&
                    // Rebalance after final steps 1 and 2 too: the probe LB
                    // balances the probe's allocation, not the final one. A
                    // restart balanced on a final generation already, so it
                    // skips the one after step 1.
                    ((greyFinalGenerationIndex == 0 && !restarting) ||
                     greyFinalGenerationIndex == 1 ||
                     (greyFinalGenerationIndex + 1) %
                         (cfg.measuredLBFinalPeriod > 0 ? cfg.measuredLBFinalPeriod
                                                        : (cfg.fluxSourceCompare ? 10 : 50)) == 0);
                // With exploration splitting, the probe puts most packets in a
                // few thick cells (ceil(E/cap) each), which cost almost nothing
                // in the burn-in (1-3 packets per cell). At a small cap that
                // crowds one rank past its memory (64M packets on one rank at
                // fraction 0.005), so before the probe, balance on the packet
                // counts the probe will create.
                bool const greyDoPredictedProbeLB =
                    greyMeasuredLBActive &&
                    cfg.adaptiveSourceCells &&
                    cfg.volumeEmissionEnabled &&
                    cfg.volumeEmissionExplorationWeightFractionGrey > 0.0 &&
                    greyLearnedProbeGenerations > 0 &&
                    gen + 1 == greyBurninGenerations;
                // With the final-budget ramp, the LB after the probe and after
                // every ramp generation balances on the packets the next
                // generation's plan will create (measured steps of this one).
                bool const greyDoRampLB =
                    greyMeasuredLBActive && greyRampGenerations > 0 &&
                    ((greyDoPostAdaptiveMeasuredLB && greyLearnedProbeThisGen) || greyRampThisGen);
                std::string const greyLBLabel = greyDoPredictedProbeLB
                    ? "MEASURED_LB_GREY_PREDICTED_PROBE"
                    : greyDoRampLB
                    ? "MEASURED_LB_GREY_RAMP_PLAN"
                    : greyDoBurninMeasuredLB
                    ? "MEASURED_LB_GREY_BURNIN"
                    : (greyDoPostAdaptiveMeasuredLB
                        ? "MEASURED_LB_GREY_FIRST_NON_BURNIN"
                        : (greyDoAdaptivePeriodicMeasuredLB
                            ? "MEASURED_LB_GREY_ADAPTIVE_PERIODIC"
                            : "MEASURED_LB_GREY"));
                if (rank == 0 && greyDoBurninMeasuredLB)
                    std::cout << "Grey first burn-in step complete; running measured LB" << std::endl;
                if (rank == 0 && greyDoPostAdaptiveMeasuredLB)
                    std::cout << "Grey first non-burn-in step complete; running measured LB" << std::endl;
                if (rank == 0 && greyDoAdaptivePeriodicMeasuredLB)
                    std::cout << "Grey periodic final measured LB after final step "
                              << (greyFinalGenerationIndex + 1)
                              << ": rank_step_imbalance="
                              << greyStepImbalance.maxOverMean
                              << std::endl;
                if (rank == 0 && greyDoPredictedProbeLB)
                    std::cout << "Grey burn-in complete; running measured LB on predicted probe packets" << std::endl;
                if (rank == 0 && greyDoRampLB)
                    std::cout << "Grey " << (greyRampThisGen ? "ramp stage complete" : "learned-only probe complete")
                              << "; the measured LB uses the next generation's planned packets" << std::endl;
                if (greyDoInitialMeasuredLB || greyDoBurninMeasuredLB ||
                    greyDoPostAdaptiveMeasuredLB || greyDoAdaptivePeriodicMeasuredLB ||
                    greyDoPredictedProbeLB || greyDoRampLB) {
                    {
                        auto const& greyLocalSteps = greyManager->GetCellsStepsCounters();

                        PredictedProbePackets greyPredictedProbePhotons;
                        if (greyDoPredictedProbeLB)
                            greyPredictedProbePhotons = PredictProbeSourcePackets(
                                cells, Ncells, runtime, greyAdaptive.scoreByCellID,
                                generationDiagnostics.sourcePhotonsPerCell, greyProbePhotons,
                                cfg.volumeEmissionExplorationWeightFractionGrey,
                                cfg.volumeEmissionCapLearnedWeight,
                                "MEASURED_LB_GREY_PREDICTED_PROBE", rank, mpiSize);
                        if (greyDoPredictedProbeLB)
                            result.probePredictedPackets = greyPredictedProbePhotons.total;
                        if (greyDoRampLB) {
                            // The plan and weight bound the next generation
                            // will use: the next ramp stage, or the full one.
                            size_t const nextGen = gen + 1;
                            bool const nextIsRamp = nextGen < greyFinalStartGeneration;
                            double const nextBudget = nextIsRamp
                                ? greyRampBudgets[nextGen - greyProbeEndGeneration] : greyRampFullBudget;
                            Config const nextCfg = nextIsRamp
                                ? RampStageAllocationConfig(greyAllocationCfg, nextBudget) : greyAllocationCfg;
                            SourceAllocationPlan const nextPlan = BuildNeymanSourceAllocation(
                                greyAdaptive.scoreByCellID, nextCfg, &runtime.fluxSourceCellIDs);
                            double const escapingReference = runtime.lastEscapingLuminosity > 0.0
                                ? runtime.lastEscapingLuminosity : runtime.fluxSourceInjectedLuminosity;
                            double const capLuminosity = cfg.volumeEmissionEnabled
                                ? cfg.volumeEmissionExplorationWeightFractionGrey * escapingReference *
                                  (greyRampFullBudget / nextBudget)
                                : 0.0;
                            greyPredictedProbePhotons = PredictPlannedSourcePackets(
                                cells, Ncells, runtime, nextPlan, capLuminosity,
                                cfg.volumeEmissionCapLearnedWeight, "MEASURED_LB_GREY_RAMP_PLAN", rank, mpiSize);
                        }
                        bool const predicted = greyDoPredictedProbeLB || greyDoRampLB;

                        greyRebalanceOnMeasurements(
                            greyLocalSteps,
                            predicted ? greyPredictedProbePhotons.perCell
                                      : generationDiagnostics.sourcePhotonsPerCell,
                            greyLBLabel, predicted);
                    }
                    if (greyDoPredictedProbeLB && rank == 0)
                        std::cout << "Grey predicted-probe measured load balance complete" << std::endl;
                    if (greyDoBurninMeasuredLB) {
                        greyBurninMeasuredLBDone = true;
                        if (rank == 0)
                            std::cout << "Grey burn-in measured load balance complete" << std::endl;
                    }
                    if (greyDoPostAdaptiveMeasuredLB) {
                        greyFirstNonBurninMeasuredLBDone = true;
                        greyAdaptive.postAdaptiveMeasuredLBDone = true;
                        greyAdaptive.adaptiveMeasuredLBCount += 1;
                        greyAdaptive.lastAdaptiveMeasuredLBGeneration = gen;
                        if (rank == 0)
                            std::cout << "Grey post-adaptive measured load balance complete" << std::endl;
                    }
                    if (greyDoAdaptivePeriodicMeasuredLB) {
                        greyAdaptive.adaptiveMeasuredLBCount += 1;
                        greyAdaptive.lastAdaptiveMeasuredLBGeneration = gen;
                        if (rank == 0)
                            std::cout << "Grey periodic measured load balance complete" << std::endl;
                    }
                }
    #endif // RICH_MPI
            }
            // Restart state for a later continuation. With no generation run
            // this segment the stored measurements are carried over.
            LoadBalanceMeasurements greySavedLoadBalance = FinishPassLoadBalance(
                greyLastFinalMeasurements, std::move(greyRestartLoadBalance),
                greyIncludedFinalGenerations + greyDiscardedBurninGenerations > 0, rank);

            if (rank == 0) {
                greyObserver->loadStatisticalMeanTallies();
                if (cfg.polarization) {
                    PrintPolarizationSummary(
                        "Grey", greyObserver->getObserverQualitySnapshot(), rank);
                    greyObserver->printGenerationPolarizationSummary("Grey");
                }
            }

            if (rank == 0) {
                std::string const greyVtk = GreyVtkOutputPath(cfg);
                std::vector<double> const& fldLuminosity = runtime.fldLuminosity;
                double const totalFldLum = runtime.totalFldLuminosity;

                if (!greyVtk.empty()) {
                    greyObserver->writeVTK(greyVtk, cfg.sourceDt);

                    size_t const nObs = greyObserver->getNumObservers();
                    if (fldLuminosity.size() != nObs)
                        throw UniversalError("FLD luminosity size mismatch in grey VTK output");
                    std::vector<double> const& greySolidAngles = greyObserver->getObserverSolidAngles();
                    std::ofstream vtkAppend(greyVtk, std::ios::app);
                    if (vtkAppend.is_open()) {
                        vtkAppend << std::scientific << std::setprecision(12);

                        double fourPi = 4.0 * M_PI;

                        vtkAppend << "SCALARS fld_surface_luminosity double 1\n"
                                  << "LOOKUP_TABLE default\n";
                        for (size_t p = 0; p < nObs; ++p)
                            vtkAppend << fldLuminosity[p] << "\n";

                        vtkAppend << "SCALARS fld_surface_isotropic_equivalent_luminosity double 1\n"
                                  << "LOOKUP_TABLE default\n";
                        for (size_t p = 0; p < nObs; ++p) {
                            double isoEquiv = (greySolidAngles[p] > 0.0)
                                ? fldLuminosity[p] * fourPi / greySolidAngles[p] : 0.0;
                            vtkAppend << isoEquiv << "\n";
                        }

                        vtkAppend << "SCALARS log10_fld_surface_luminosity double 1\n"
                                  << "LOOKUP_TABLE default\n";
                        for (size_t p = 0; p < nObs; ++p) {
                            double val = (fldLuminosity[p] > 0.0) ? std::log10(fldLuminosity[p]) : -99.0;
                            vtkAppend << val << "\n";
                        }

                        vtkAppend << "SCALARS fld_surface_flux double 1\n"
                                  << "LOOKUP_TABLE default\n";
                        for (size_t p = 0; p < nObs; ++p) {
                            double patchArea_p = greySolidAngles[p] * cfg.radius * cfg.radius;
                            double flux = (patchArea_p > 0.0) ? fldLuminosity[p] / patchArea_p : 0.0;
                            vtkAppend << flux << "\n";
                        }
                        AppendZeroVtkScalar(vtkAppend, "fld_surface_luminosity_stderr_gen", nObs);
                        AppendZeroVtkScalar(vtkAppend, "fld_surface_luminosity_relerr_gen", nObs);
                        AppendZeroVtkScalar(vtkAppend, "fld_surface_luminosity_stderr_packet", nObs);
                        AppendZeroVtkScalar(vtkAppend, "fld_surface_luminosity_relerr_packet", nObs);
                        AppendZeroVtkScalar(vtkAppend, "fld_surface_luminosity_neff", nObs);
                        AppendZeroVtkScalar(vtkAppend, "fld_surface_isotropic_equivalent_luminosity_stderr_gen", nObs);
                        AppendZeroVtkScalar(vtkAppend, "fld_surface_isotropic_equivalent_luminosity_relerr_gen", nObs);
                        AppendZeroVtkScalar(vtkAppend, "fld_surface_flux_stderr_gen", nObs);
                        AppendZeroVtkScalar(vtkAppend, "fld_surface_flux_relerr_gen", nObs);
                        AppendZeroVtkScalar(vtkAppend, "log10_fld_surface_luminosity_stderr_gen", nObs);
                        AppendZeroVtkScalar(vtkAppend, "log10_fld_surface_luminosity_relerr_gen", nObs);
                    }
                    std::cout << "Grey VTK output: " << greyVtk << std::endl;
                }

                double greyTotalLum = greyObserver->getTotalCrossingEnergy() / cfg.sourceDt;
                double greyEmitted = greyObserver->getEmittedEnergy();
                double greyAbsorbed = greyObserver->getAbsorbedEnergy();
                double greyBoxEscape = greyObserver->getBoxEscapeEnergy();
                double greyTimedOut = greyObserver->getTimedOutEnergy();
                double greyCutoff = greyObserver->getCutoffEnergy();
                double greyCompressionWork = greyObserver->getCompressionWork();
                double greyResidual = greyEmitted + greyCompressionWork - greyAbsorbed - greyBoxEscape
                                    - greyTimedOut - greyCutoff;
                double greyTimedOutFrac = (greyEmitted > 0.0) ? greyTimedOut / greyEmitted : 0.0;
                FluxSourcePolarizationSummary const greyPol =
                    ComputeFluxSourcePolarizationSummary(
                        greyObserver->getObserverQualitySnapshot());
                result.ran = true;
                result.finalGenerationsAdded = greyIncludedFinalGenerations;
                result.finalGenerationsTotal =
                    greyCompletedFinalGenerations + greyIncludedFinalGenerations;
                result.usesVelocity = false;
                result.usesDDMC = cfg.ddmc;
                result.usesPolarization = cfg.polarization;
                result.usesCompton = false;
                result.sourceLuminosity = runtime.fluxSourceInjectedLuminosity;
                result.emittedLuminosity = greyEmitted / cfg.sourceDt;
                result.crossingLuminosity = greyTotalLum;
                result.crossingLuminosityStderr =
                    greyObserver->getTotalLuminosityStderrGen(cfg.sourceDt);
                result.emittedEnergy = greyEmitted;
                result.timedOutFraction = greyTimedOutFrac;
                result.luminosityWeightedPolarizationDegree =
                    greyPol.luminosityWeightedDegree;
                result.polarizedObserverCount = greyPol.observerCount;
                SphericalObserver::Diagnostics greyDiagnostics;
                greyDiagnostics.sourceDt = cfg.sourceDt;
                greyDiagnostics.transportTime = cfg.transportTime;
                greyDiagnostics.mpiRanks = mpiSize;
                greyDiagnostics.comptonEnabled = 0;
                greyDiagnostics.emittedEnergy = greyEmitted;
                greyDiagnostics.absorbedEnergy = greyAbsorbed;
                greyDiagnostics.boxEscapeEnergy = greyBoxEscape;
                greyDiagnostics.timedOutEnergy = greyTimedOut;
                greyDiagnostics.cutoffEnergy = greyCutoff;
                greyDiagnostics.compressionWork = greyCompressionWork;
                greyDiagnostics.snapshotTime = runtime.snapshotTime;
                greyDiagnostics.snapshotCycle = runtime.snapshotCycle;
                // Final generations in the result, like included_final_generations.
                greyDiagnostics.nGenerations = static_cast<int>(greyCompletedFinalGenerations + nGreyGens);
                greyDiagnostics.includedFinalGenerations = static_cast<int>(
                    greyCompletedFinalGenerations + greyIncludedFinalGenerations);
                greyDiagnostics.discardedBurninGenerations =
                    static_cast<int>(greyDiscardedBurninGenerations);
                greyDiagnostics.adaptiveOnlyFinalOutput =
                    cfg.adaptiveSourceCells ? 1 : 0;
                PassRestartState greySaved;
                greySaved.adaptive = greyAdaptive;
                greySaved.lastEscapingLuminosity = runtime.lastEscapingLuminosity;
                greySaved.completedFinalGenerations =
                    greyCompletedFinalGenerations + greyIncludedFinalGenerations;
                greySaved.loadBalance = std::move(greySavedLoadBalance);
                WriteGreyPassScratch(cfg, *greyObserver, greyDiagnostics, greySaved);

                if (cfg.fluxSourceCompare && !forwardResult.ran)
                    throw UniversalError(
                        "Flux-source comparison is missing the MG result");

                std::cout << "\n=== Grey IMC Results ===\n"
                          << "Generations:              " << nGreyGens << "\n"
                          << "Final included generations: " << greyIncludedFinalGenerations << "\n"
                          << "Total final generations:  "
                          << (greyCompletedFinalGenerations + greyIncludedFinalGenerations)
                          << (restarting ? " (restart)" : "") << "\n"
                          << "Discarded burn-in generations: " << greyDiscardedBurninGenerations << "\n";
                if (cfg.adaptiveSourceCells)
                    std::cout << "Schedule burn-in/probe/final: " << greyBurninGenerations
                              << "/" << greyLearnedProbeGenerations
                              << "/" << nGreyGens << "\n";
                std::cout << "Final average policy:     " << (cfg.adaptiveSourceCells ? "adaptive_only" : "all_generations") << "\n"
                          << "Photons/cell/gen:         " << greyPhotonsPerCell << "\n"
                          << "Total crossing luminosity: " << greyTotalLum << " +/- "
                          << greyObserver->getTotalLuminosityStderrGen(cfg.sourceDt)
                          << " erg/s (rel=" << greyObserver->getTotalLuminosityRelErrGen(cfg.sourceDt) << ")\n"
                          << "Polarization degree:      " << greyPol.luminosityWeightedDegree
                          << " (L-weighted, noise-debiased; " << greyPol.observerCount << " observers, "
                          << greyPol.undebiasedObserverCount << " with < 10 effective packets kept raw)\n"
                          << "Total FLD luminosity:     " << totalFldLum << " erg/s\n"
                          << "Emitted energy:           " << greyEmitted << " erg\n"
                          << "Absorbed energy:          " << greyAbsorbed << " erg\n"
                          << "Box escape energy:        " << greyBoxEscape << " erg\n"
                          << "Timed-out energy:         " << greyTimedOut << " erg\n"
                          << "Cutoff energy:            " << greyCutoff << " erg\n"
                          << "Compression work:         " << greyCompressionWork << " erg\n"
                          << "Sink residual:            " << greyResidual << " erg\n"
                          << "Timed-out fraction:       " << greyTimedOutFrac << "\n"
                          << "Grey VTK written to:      " << greyVtk << "\n"
                          << std::endl;
            }
            greyObserver->printEscapeWeightSummary("grey");

    runtime.nCells = Ncells;
    return result;
}

} // namespace imc_postprocess_tde
