#include "GreyCalculation.hpp"

#ifdef RICH_MPI
#include "source/monte/manager/communication/RDMACommunicationEngine.hpp"
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

#ifdef RICH_MPI
#include <mpi.h>
#include "source/mpi/mpi_commands.hpp"
#endif

#include "AdaptiveStatistics.hpp"
#include "FluxSourceCalculation.hpp"
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
                hasGreyPhotosphere = greyPhotosphere.hasGrey();
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
            size_t nGreyGens = cfg.fluxSourceCompare
                ? cfg.nGenerations : std::max<size_t>(1, cfg.nGenerations / 2);
            size_t greyBudgetDivisor = cfg.fluxSourceCompare ? nGreyGens : 2 * nGreyGens;
            size_t greyPhotonsPerCell = std::max<size_t>(1, cfg.photonsPerCell / greyBudgetDivisor);

            if (rank == 0)
                std::cout << "\n=== Starting Grey IMC run (" << nGreyGens << " generations) ===" << std::endl;

            auto greyObserver = std::make_shared<SphericalObserver>(
                cfg.center, cfg.radius, cfg.nObservers, std::vector<double>());
            if (hasGreyPhotosphere)
                greyObserver->setPhotosphereData(std::move(greyPhotosphere));

            auto greyBoundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(tess);

            RadiationIMCParameters greyParams;
            greyParams.newPhotonsPerCell = greyPhotonsPerCell;
            greyParams.withHydro = false;
            greyParams.noHydroFeedback = true;
            greyParams.withRandomWalk = cfg.randomWalk && !cfg.fluxSourceCompare;
            greyParams.rwMinCellOpticalDepth = 15;
            greyParams.withDDMC = cfg.ddmc;
            greyParams.ddmcMinCellOpticalDepth = 15;
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
            greyParams.postProcess.polarization.enabled = cfg.polarization;
            greyParams.postProcess.polarization.manualScatteringsAfterAcceleration = cfg.polarizationManualScatterings;
            greyParams.postProcess.polarization.depolarizationScatterings = cfg.polarizationDepolarizationScatterings;
            greyParams.postProcess.polarization.acceleratedClosure = cfg.polarizationClosure;

            auto greyPhysics = std::make_shared<RadiationIMC>(
                tess, greyBoundary, cells, extensives, eos, greyOpacity, greyParams);
            greyPhysics->setObserver(greyObserver);
            if(cfg.fluxSourceCompare)
                ConfigureFluxSourceForCurrentDecomposition(
                    cfg, runtime, *greyPhysics);

            auto greyPopControl = std::make_shared<STORM::NoPopulationControl<Vector3D, Tessellation3D>>(tess);

            std::shared_ptr<MonteCarloManager3D> greyManager;
    #ifdef RICH_MPI
            MonteCarloConfig monteCarloConfig;
            std::unique_ptr<STORM::CommunicationEngine<Vector3D>> engine =
                std::make_unique<STORM::RDMACommunicationEngine<Vector3D, Tessellation3D>>(
                    tess, monteCarloConfig, MPI_COMM_WORLD, RDMA_Type::AUTO_RDMA);
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
            double const greyLBWeightCompression = EffectiveMeasuredLBWeightCompression(cfg);

            AdaptiveSourceState greyAdaptive;
            bool greyBurninMeasuredLBDone = false;
            bool greyFirstNonBurninMeasuredLBDone = false;
            greyObserver->clearGenerationStatistics();
            size_t greyIncludedFinalGenerations = 0;
            size_t greyDiscardedBurninGenerations = 0;
            size_t const greyInitialBurninGenerations = cfg.adaptiveSourceCells ? 1 : 0;
            size_t const greyUniformBurninGenerations = cfg.adaptiveSourceCells ? 19 : 0;
            size_t const greyBurninGenerations = greyInitialBurninGenerations + greyUniformBurninGenerations;
            size_t const greyLearnedProbeGenerations = cfg.adaptiveSourceCells ? 1 : 0;
            size_t const greyFinalStartGeneration = greyBurninGenerations + greyLearnedProbeGenerations;
            size_t const greyTotalGenerations = cfg.adaptiveSourceCells
                ? greyFinalStartGeneration + nGreyGens
                : nGreyGens;
            for (size_t gen = 0; gen < greyTotalGenerations; ++gen) {
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
                    gen < greyFinalStartGeneration;
                bool const greyFinalThisGen =
                    !cfg.adaptiveSourceCells || gen >= greyFinalStartGeneration;
                size_t const greyFinalGenerationIndex = greyFinalThisGen
                    ? (cfg.adaptiveSourceCells ? gen - greyFinalStartGeneration : gen)
                    : 0;
                bool const greyAdaptiveActiveThisGen =
                    cfg.adaptiveSourceCells &&
                    (greyLearnedProbeThisGen || greyFinalThisGen) &&
                    !greyAdaptive.scoreByCellID.empty();
                size_t const greyPhotonsThisGen = greyFirstBurninThisGen ? 1
                    : (greyUniformBurninThisGen ? 3
                       : (greyLearnedProbeThisGen ? 75
                          : (cfg.adaptiveSourceCells ? 1 : greyPhotonsPerCell)));
                std::string greyPhase = "final";
                if (greyFirstBurninThisGen)
                    greyPhase = "burnin_exact1";
                else if (greyUniformBurninThisGen)
                    greyPhase = "burnin_exact3";
                else if (greyLearnedProbeThisGen)
                    greyPhase = "learned_only_probe_exact75";
                else if (cfg.adaptiveSourceCells)
                    greyPhase = "learned_only_final";
                greyPhysics->setNewPhotonsPerCell(greyPhotonsThisGen);
                if (rank == 0)
                    std::cout << "Grey generation " << (gen + 1) << "/" << greyTotalGenerations
                              << " phase=" << greyPhase
                              << " photons_per_cell_this_gen=" << greyPhotonsThisGen;
                if (rank == 0 && greyFinalThisGen)
                    std::cout << " final_step=" << (greyFinalGenerationIndex + 1)
                              << "/" << nGreyGens;
                if (rank == 0)
                    std::cout << std::endl;
                PrintAdaptiveGenerationStart("Grey", cfg, greyAdaptive, gen, greyTotalGenerations,
                                             greyBurninGenerations,
                                             greyAdaptiveActiveThisGen, rank);

                IMCPostProcessControl postProcessControl;
                if (greyAdaptiveActiveThisGen) {
                    postProcessControl.adaptiveCells.enabled = true;
                    postProcessControl.adaptiveCells.scores =
                        greyAdaptive.scoreByCellID;
                    double const learnedMinFactorThisGen =
                        greyLearnedProbeThisGen ? 1.0 : cfg.adaptiveSourceLearnedMinFactor;
                    size_t const learnedMinPhotonsThisGen =
                        greyFinalThisGen ? cfg.adaptiveSourceLearnedMinPhotons : 0;
                    size_t const learnedMaxPhotonsThisGen =
                        greyFinalThisGen ? cfg.adaptiveSourceLearnedMaxPhotons : 0;
                    double const scorePowerThisGen =
                        greyFinalThisGen ? cfg.adaptiveSourceScorePower : 1.0;
                    postProcessControl.adaptiveCells.strength =
                        cfg.adaptiveSourceStrength;
                    postProcessControl.adaptiveCells.maxFactor =
                        cfg.adaptiveSourceMaxFactor;
                    postProcessControl.adaptiveCells.learnedReserveFraction =
                        cfg.adaptiveSourceLearnedReserveFrac;
                    postProcessControl.adaptiveCells.learnedMinFactor =
                        learnedMinFactorThisGen;
                    postProcessControl.adaptiveCells.observerBudgetMultiplier =
                        greyAdaptive.observerBudgetMultiplier;
                    postProcessControl.adaptiveCells.learnedMinPhotons =
                        learnedMinPhotonsThisGen;
                    postProcessControl.adaptiveCells.learnedMaxPhotons =
                        learnedMaxPhotonsThisGen;
                    postProcessControl.adaptiveCells.scorePower =
                        scorePowerThisGen;
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
                } else if (cfg.adaptiveSourceCells && greyFinalThisGen) {
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
                        cfg, runtime, *greyPhysics);
                greyObserver->resetGenerationSourceCellEscapeStats();

                greyPhysics->reseedRNG(static_cast<uint64_t>(rank + 87654321) * greyTotalGenerations + gen);

                greyManager->getParticles().clear();
                greyManager->step(cells, cfg.transportTime);
                IMCPostProcessGenerationDiagnostics const generationDiagnostics =
                    greyPhysics->getPostProcessGenerationDiagnostics();

                auto greyAllocation = ReduceSourceAllocationSummary(
                    generationDiagnostics.sourceAllocation);
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
                    greySourceStats, cfg, greyAdaptive, greyObserverQuality,
                    !greyBurninThisGen, rank, mpiSize);
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
                if (rank == 0 && cfg.adaptiveSourceCells)
                    std::cout << "Grey learned cells after iteration " << (gen + 1)
                              << ": " << greyAdaptive.scoreByCellID.size() << std::endl;
                greyObserver->addBoxEscapeEnergy(greyBoundary->getEscapedEnergy());
                greyBoundary->resetEscapedEnergy();
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
                    gen == 0 && greyMeasuredLBActive &&
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
                    (greyFinalGenerationIndex + 1) %
                        (cfg.fluxSourceCompare ? 10 : 50) == 0;
                std::string const greyLBLabel = greyDoBurninMeasuredLB
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
                if (greyDoInitialMeasuredLB || greyDoBurninMeasuredLB ||
                    greyDoPostAdaptiveMeasuredLB || greyDoAdaptivePeriodicMeasuredLB) {
                    if (!greyParams.noHydroFeedback) {
                        throw UniversalError("Grey measured load balance repartition requires noHydroFeedback=true");
                    }

                    PrintVmRSS("grey_before_measured_lb", rank);

                    std::vector<double> greyWeightsForExchange;
                    imc_measured_lb::Parameters greyLBParamsThisPass =
                        greyLBParams;

                    {
                        auto const& greyLocalSteps = greyManager->GetCellsStepsCounters();

                        std::vector<size_t> greyCellIDs(Ncells);
                        for (size_t i = 0; i < Ncells; ++i)
                            greyCellIDs[i] = cells[i].ID;

                        auto greyLocalMeas = imc_measured_lb::BuildLocalMeasurements(
                            greyCellIDs, greyLocalSteps,
                            generationDiagnostics.sourcePhotonsPerCell);

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

                            tess.BuildParallel(greyCurrentPoints, greyLBWeights);
                        }
                    }

                    if (!greyWeightsForExchange.empty()) {
                        MPI_exchange_data(tess, cells, false, 1, &runtime.dummyCell);

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

                        extensives.resize(Ncells);
                        for (size_t i = 0; i < Ncells; ++i)
                            PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

                        greyObserver->addBoxEscapeEnergy(greyBoundary->getEscapedEnergy());
                        greyBoundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(tess);

                        greyPhysics = std::make_shared<RadiationIMC>(
                            tess, greyBoundary, cells, extensives, eos, greyOpacity, greyParams);
                        greyPhysics->setObserver(greyObserver);
                        if(cfg.fluxSourceCompare)
                            ConfigureFluxSourceForCurrentDecomposition(
                                cfg, runtime, *greyPhysics);

                        greyPopControl = std::make_shared<STORM::NoPopulationControl<Vector3D, Tessellation3D>>(tess);

                        std::unique_ptr<STORM::CommunicationEngine<Vector3D>> rebuiltEngine =
                            std::make_unique<STORM::RDMACommunicationEngine<Vector3D, Tessellation3D>>(
                                tess, monteCarloConfig, MPI_COMM_WORLD, RDMA_Type::AUTO_RDMA);
                        greyManager = std::make_shared<MonteCarloManager3D>(
                            tess, greyPhysics, greyPopControl, greyBoundary,
                            monteCarloConfig, std::move(rebuiltEngine));

                        if (rank == 0)
                            std::cout << greyLBLabel
                                      << ": repartitioned, new local cells=" << Ncells << std::endl;

                        PrintVmRSS("grey_after_rebuild_physics", rank);
                    }
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
            if (rank == 0) {
                greyObserver->loadStatisticalMeanTallies();
                if (cfg.polarization)
                    PrintPolarizationSummary(
                        "Grey", greyObserver->getObserverQualitySnapshot(), rank);
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
                double greyResidual = greyEmitted - greyAbsorbed - greyBoxEscape - greyTimedOut - greyCutoff;
                double greyTimedOutFrac = (greyEmitted > 0.0) ? greyTimedOut / greyEmitted : 0.0;
                FluxSourcePolarizationSummary const greyPol =
                    ComputeFluxSourcePolarizationSummary(
                        greyObserver->getObserverQualitySnapshot());
                result.ran = true;
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
                greyDiagnostics.snapshotTime = runtime.snapshotTime;
                greyDiagnostics.snapshotCycle = runtime.snapshotCycle;
                greyDiagnostics.nGenerations = static_cast<int>(nGreyGens);
                greyDiagnostics.includedFinalGenerations =
                    static_cast<int>(greyIncludedFinalGenerations);
                greyDiagnostics.discardedBurninGenerations =
                    static_cast<int>(greyDiscardedBurninGenerations);
                greyDiagnostics.adaptiveOnlyFinalOutput =
                    cfg.adaptiveSourceCells ? 1 : 0;
                WriteGreyPassScratch(cfg, *greyObserver, greyDiagnostics);

                if (cfg.fluxSourceCompare && !forwardResult.ran)
                    throw UniversalError(
                        "Flux-source comparison is missing the MG result");

                std::cout << "\n=== Grey IMC Results ===\n"
                          << "Generations:              " << nGreyGens << "\n"
                          << "Final included generations: " << greyIncludedFinalGenerations << "\n"
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
                          << "Total FLD luminosity:     " << totalFldLum << " erg/s\n"
                          << "Emitted energy:           " << greyEmitted << " erg\n"
                          << "Absorbed energy:          " << greyAbsorbed << " erg\n"
                          << "Box escape energy:        " << greyBoxEscape << " erg\n"
                          << "Timed-out energy:         " << greyTimedOut << " erg\n"
                          << "Cutoff energy:            " << greyCutoff << " erg\n"
                          << "Sink residual:            " << greyResidual << " erg\n"
                          << "Timed-out fraction:       " << greyTimedOutFrac << "\n"
                          << "Grey VTK written to:      " << greyVtk << "\n"
                          << std::endl;
            }

    runtime.nCells = Ncells;
    return result;
}

} // namespace imc_postprocess_tde
