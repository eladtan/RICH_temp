#include "ForwardCalculation.hpp"

#ifdef RICH_MPI
#include "source/monte/manager/communication/RDMACommunicationEngine.hpp"
#include "PostProcessCommunication.hpp"
#endif // RICH_MPI

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

ForwardPostprocessResult RunForwardPostprocess(Config const& cfg, PostprocessRuntime& runtime)
{
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
    auto& params = runtime.params;
    size_t Ncells = runtime.nCells;

    ForwardPostprocessResult result;
            using Particle3D = MonteCarloParticle<Vector3D>;

            bool const measuredLBActive =
                cfg.measuredLoadBalance && (cfg.adaptiveSourceCells || cfg.nGenerations > 1);
    #if ENERGY_GROUPS_NUM > 1
            bool const isMultigroup = true;
    #else
            bool const isMultigroup = false;
    #endif
            size_t const genPhotonsPerCell = std::max<size_t>(
                1, cfg.photonsPerCell / std::max<size_t>(1, cfg.nGenerations));

            if (rank == 0)
                std::cout << "Measured LB active: " << (measuredLBActive ? "yes" : "no") << std::endl;

            imc_measured_lb::Parameters measuredLBParams;
            measuredLBParams.floorCost = 1.0;
            measuredLBParams.stepWeight = 1.0;
            measuredLBParams.particleWeight = cfg.adaptiveSourceCells ? 1.0 : 0.0;
            measuredLBParams.medianClampFactor = isMultigroup ? 30.0 : 20.0;
            measuredLBParams.missingCellCost = isMultigroup ? 5.0 : 2.0;
            measuredLBParams.grayZeroStepInflation = 2.0;
            measuredLBParams.multigroupZeroStepInflation = 5.0;
            measuredLBParams.maxCellImbalance = MEASURED_LB_MAX_CELL_IMBALANCE;
            measuredLBParams.useMedianClamp = true;
            measuredLBParams.clampToActiveMean = cfg.measuredLBClampToActiveMean;
            if (cfg.measuredLBClampFactor > 0.0)
                measuredLBParams.medianClampFactor = cfg.measuredLBClampFactor;
            double const measuredLBWeightCompression = EffectiveMeasuredLBWeightCompression(cfg);

            AdaptiveSourceState mgAdaptive;
            WeightSplitState mgSplit;
            AdaptiveGroupHistory mgGroupHistory;
            AdaptiveGroupSourceState mgGroupSourceState;
            RadiationIMC::GroupSamplingDiagnostics mgLastGroupSamplingDiag;
            RadiationIMC::GroupSamplingDiagnostics mgFinalGroupSamplingDiag;
            AdaptiveGroupSourceUpdateSummary mgFinalGroupSourceSummary;
            bool mgGroupFallbackToIntegrated = false;
            std::string mgGroupFallbackReason = "none";
            observer->clearGenerationStatistics();
            size_t mgIncludedFinalGenerations = 0;
            size_t mgDiscardedBurninGenerations = 0;

            // A restart continues the statistics and learned state of the file
            // and never reruns the burn-in or the learned-only probe.
            bool const restarting = !cfg.restartFrom.empty();
            size_t mgCompletedFinalGenerations = 0;
            LoadBalanceMeasurements mgRestartLoadBalance;   // rank 0
            if (restarting) {
                PassRestartState loaded = BeginPassRestart(
                    cfg, RestartPass::Forward, *observer, cfg.nGenerations, rank);
                mgAdaptive = std::move(loaded.adaptive);
                mgGroupHistory = std::move(loaded.groupHistory);
                mgGroupSourceState = std::move(loaded.groupSourceState);
                mgFinalGroupSamplingDiag = loaded.finalGroupSamplingDiag;
                mgFinalGroupSourceSummary = loaded.finalGroupSourceSummary;
                mgGroupFallbackToIntegrated = loaded.groupFallbackToIntegrated;
                mgGroupFallbackReason = loaded.groupFallbackReason;
                runtime.lastEscapingLuminosity = loaded.lastEscapingLuminosity;
                mgCompletedFinalGenerations = loaded.completedFinalGenerations;
                mgRestartLoadBalance = std::move(loaded.loadBalance);
            }
            // Touched cells of this segment's last final generation (local).
            LoadBalanceMeasurements mgLastFinalMeasurements;

            size_t const mgInitialBurninGenerations = cfg.adaptiveSourceCells ? 1 : 0;
            size_t const mgUniformBurninGenerations = cfg.adaptiveSourceCells ? 19 : 0;
            size_t const mgBurninGenerations = mgInitialBurninGenerations + mgUniformBurninGenerations;
            size_t const mgLearnedProbeGenerations = cfg.adaptiveSourceCells ? 1 : 0;
            size_t const mgProbePhotons = cfg.volumeEmissionEnabled ? 10 : 75;
            size_t const mgProbeEndGeneration = mgBurninGenerations + mgLearnedProbeGenerations;
            // Final-budget ramp between the probe and the final generations
            // (a restart starts at the final generations and skips it).
            double const mgRampFullBudget = static_cast<double>(std::max(
                cfg.adaptiveSourceLearnedPhotonsPerCellBudget,
                cfg.volumeEmissionEnabled ? cfg.volumeEmissionLearnedPhotonsPerCellBudget : size_t(0)));
            std::vector<double> const mgRampBudgets = cfg.adaptiveSourceCells
                ? FinalRampStageBudgets(cfg, mgRampFullBudget) : std::vector<double>();
            size_t const mgRampGenerations = mgRampBudgets.size();
            size_t const mgFinalStartGeneration = mgProbeEndGeneration + mgRampGenerations;
            size_t const mgFinalGenerations = cfg.nGenerations;
            size_t const mgTotalGenerations = cfg.adaptiveSourceCells
                ? mgFinalStartGeneration + mgFinalGenerations
                : cfg.nGenerations;
    #ifdef RICH_MPI
            // Repartitions on one generation's measured per-cell steps and
            // source packets (predicted packets before the probe): the
            // rebalances in the loop, and on a restart the initial one from
            // the stored last-generation measurements.
            auto rebalanceOnMeasurements = [&](std::vector<size_t> const& localSteps,
                                               std::vector<size_t> const& localParticles,
                                               std::string const& mgLBLabel,
                                               bool predictedProbe)
            {
                if (!params.noHydroFeedback) {
                    throw UniversalError("Measured load balance repartition requires noHydroFeedback=true");
                }

                PrintVmRSS("before_measured_lb", rank);

                std::vector<double> measuredWeightsForExchange;
                // Set on every rank once the mesh was rebuilt; a rank that received
                // no cells has an empty weight vector but must still rebuild its
                // physics and join the collectives below.
                bool meshRepartitioned = false;
                imc_measured_lb::Parameters measuredLBParamsThisPass =
                    measuredLBParams;

                {
                    std::vector<size_t> cellIDs(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        cellIDs[i] = cells[i].ID;

                    auto localMeasurements = imc_measured_lb::BuildLocalMeasurements(
                        cellIDs, localSteps,
                        localParticles);

                    uint64_t localTotalSteps = 0;
                    uint64_t localTotalSourceParticles = 0;
                    for (auto const& m : localMeasurements) {
                        localTotalSteps += static_cast<uint64_t>(m.stepCount);
                        localTotalSourceParticles +=
                            static_cast<uint64_t>(m.particleCount);
                    }

                    uint64_t globalTotalSteps = 0;
                    uint64_t globalTotalSourceParticles = 0;
                    MPI_Allreduce(&localTotalSteps, &globalTotalSteps, 1,
                                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(&localTotalSourceParticles,
                                  &globalTotalSourceParticles, 1,
                                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

                    // In CER mode the measured step counter is charged to
                    // transport cells, while all new packets are created by
                    // a thin set of source cells.  The generic global-mean
                    // clamp otherwise reduces those source cells to almost
                    // zero partition weight and can place millions of
                    // particles on one rank.  Charge source origins the
                    // measured average downstream work per emitted packet,
                    // and add that term after the generic cell-cost clamp.
                    if (cfg.fluxSourceCompare &&
                        globalTotalSteps > 0 &&
                        globalTotalSourceParticles > 0) {
                        measuredLBParamsThisPass.particleWeight = std::max(
                            1.0,
                            static_cast<double>(globalTotalSteps) /
                                static_cast<double>(globalTotalSourceParticles));
                        measuredLBParamsThisPass.particleCostAfterClamp = true;
                        if (rank == 0) {
                            std::cerr
                                << "MEASURED_LB_CER_SOURCE_COST"
                                << " global_steps=" << globalTotalSteps
                                << " global_source_particles="
                                << globalTotalSourceParticles
                                << " steps_per_source_particle="
                                << measuredLBParamsThisPass.particleWeight
                                << " post_clamp=yes\n";
                        }
                    }
                    if (predictedProbe) {
                        // Exploration packets of thick cells are absorbed in about
                        // one step; their cost is creation and memory, so weight
                        // each predicted packet as one step, after the clamp.
                        measuredLBParamsThisPass.particleWeight = 1.0;
                        measuredLBParamsThisPass.particleCostAfterClamp = true;
                    }

                    if (globalTotalSteps == 0) {
                        if (rank == 0)
                            std::cerr << mgLBLabel
                                      << ": measured generation had zero total steps, skipping repartition\n";
                    } else {
                        // Global-mean-based clamp via MPI_Allreduce (no per-cell allgather).
                        auto localCostByCellID = imc_measured_lb::BuildMeasuredCosts(
                            localMeasurements, measuredLBParamsThisPass,
                            isMultigroup, MPI_COMM_WORLD);

                        imc_measured_lb::PrintMeasuredLBDiagnosticsDistributed(
                            localMeasurements, localCostByCellID, isMultigroup, MPI_COMM_WORLD);

                        PrintVmRSS("after_local_costs", rank);

                        size_t localMissingCosts = 0;
                        for (size_t i = 0; i < Ncells; ++i) {
                            if (localCostByCellID.find(cells[i].ID) == localCostByCellID.end())
                                ++localMissingCosts;
                        }
                        uint64_t localMissing64 = static_cast<uint64_t>(localMissingCosts);
                        uint64_t globalMissingCosts = 0;
                        MPI_Allreduce(&localMissing64, &globalMissingCosts, 1,
                                      MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
                        if (rank == 0 && globalMissingCosts != 0) {
                            std::cerr << "MEASURED_LB_WARNING missing local measured costs before repartition: "
                                      << globalMissingCosts << "\n";
                        }

                        IMCStepCounterCostCalculator::Parameters costCalcParams;
                        costCalcParams.floorCost =
                            measuredLBParamsThisPass.floorCost;
                        costCalcParams.missingCellCost =
                            measuredLBParamsThisPass.missingCellCost;
                        IMCStepCounterCostCalculator measuredCostCalc(std::move(localCostByCellID), costCalcParams);

                        std::vector<Vector3D> currentPoints(Ncells);
                        for (size_t i = 0; i < Ncells; ++i)
                            currentPoints[i] = tess.GetMeshPoint(i);

                        auto lbWeightsNew = measuredCostCalc.CalculateCost(tess, cells);

                        if (lbWeightsNew.size() != Ncells) {
                            throw UniversalError("Measured LB weight count mismatch before BuildParallel");
                        }

                        measuredWeightsForExchange = lbWeightsNew;

                        for (auto& w : lbWeightsNew)
                            w = std::pow(w, measuredLBWeightCompression);

                        // Release the pre-repartition transport objects before
                        // the rebuild: the parallel Voronoi build is the memory
                        // peak of the whole run, and every rank on a node hits
                        // it at the same moment. Their measured costs were
                        // copied above; nothing below reads them until they
                        // are rebuilt on the new decomposition.
                        manager.reset();
                        physics.reset();
                        popControl.reset();
                        boundary.reset();
#ifdef __linux__
                        malloc_trim(0);
#endif
                        PrintVmRSS("after_release_old_physics", rank);

                        tess.BuildParallel(currentPoints, lbWeightsNew);
                        meshRepartitioned = true;
                        // The rebuild leaves every mesh container with the capacity of
                        // the largest mesh this rank ever held; hand it back.
                        tess.ShrinkToFit();
                    }
                }

                PrintVmRSS("after_build_parallel", rank);

                if (meshRepartitioned) {
                    MPI_exchange_data(tess, cells, false, 1, &runtime.dummyCell);
                    MigrateWeightSplitState(runtime, mgSplit);

                    double dummyWeight =
                        measuredLBParamsThisPass.missingCellCost;
                    MPI_exchange_data(tess, measuredWeightsForExchange, false, 1, &dummyWeight);

                    Ncells = tess.GetPointNo();

                    if (cells.size() != Ncells) {
                        UniversalError eo("Cell count mismatch after measured LB repartition");
                        eo.addEntry("cells.size()", static_cast<double>(cells.size()));
                        eo.addEntry("Ncells", static_cast<double>(Ncells));
                        throw eo;
                    }

                    if (measuredWeightsForExchange.size() != Ncells) {
                        UniversalError eo("Measured weight count mismatch after repartition");
                        eo.addEntry("weights.size()", static_cast<double>(measuredWeightsForExchange.size()));
                        eo.addEntry("Ncells", static_cast<double>(Ncells));
                        throw eo;
                    }

                    imc_measured_lb::PrintPostRepartitionDiagnosticsFromWeights(
                        measuredWeightsForExchange, measuredLBWeightCompression,
                        isMultigroup, MPI_COMM_WORLD);

                    PrintVmRSS("after_exchange", rank);

                    cells.shrink_to_fit();
                    extensives.resize(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);
                    extensives.shrink_to_fit();
#ifdef __linux__
                    malloc_trim(0);
#endif

#if ENERGY_GROUPS_NUM > 1
                    if (cfg.opacityScaleMode != OpacityScaleMode::None) {
                        // Measured LB changes which cell IDs are local to each MPI rank.
                        // The alpha table is local and ID-keyed, so refresh it
                        // before rebuilding physics and before the next transport step.
                        RecomputeOpacityScaleFactors(
                            *opacity, *greyOpacity, cells, Ncells, rank,
                            cfg.opacityScaleMode,
                            cfg.opacityScaleAlphaMin, cfg.opacityScaleAlphaMax,
                            runtime.applyOpacityScaleFactors,
                            "after measured LB repartition");
                    }
#endif

                    // Old transport objects were released before the rebuild.
                    boundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(tess);

                    physics = std::make_shared<RadiationIMC>(
                        tess, boundary, cells, extensives, eos, opacity, params);
                    physics->setObserver(observer);
                    if(cfg.fluxSourceCompare)
                        ConfigureFluxSourceForCurrentDecomposition(
                            cfg, runtime, *physics, *opacity, true);

                    popControl = std::make_shared<STORM::NoPopulationControl<Vector3D, Tessellation3D>>(tess);

                    MonteCarloConfig monteCarloConfig;
                    std::unique_ptr<STORM::CommunicationEngine<Vector3D>> rebuiltEngine =
                        MakeCommunicationEngine(
            tess, monteCarloConfig, cfg.communication, MPI_COMM_WORLD);
                    manager = std::make_shared<MonteCarloManager3D>(
                        tess, physics, popControl, boundary,
                        monteCarloConfig, std::move(rebuiltEngine));

                    if (rank == 0)
                        std::cout << mgLBLabel
                                  << ": repartitioned, new local cells=" << Ncells << std::endl;

                    PrintVmRSS("after_rebuild_physics", rank);
                }
            };
    #endif // RICH_MPI

    #ifdef RICH_MPI
            if (restarting && measuredLBActive && mgFinalGenerations > 0) {
                // Balance on the last generation of the run being continued,
                // which used the final allocation, before the first generation.
                RebalanceFromStoredMeasurements(
                    mgRestartLoadBalance, cells, Ncells, "MG",
                    [&](std::vector<size_t> const& steps, std::vector<size_t> const& particles) {
                        rebalanceOnMeasurements(steps, particles, "MEASURED_LB_RESTART", false);
                    }, rank);
            }
    #endif // RICH_MPI
            size_t const mgStartGeneration = restarting ? mgFinalStartGeneration : 0;
#ifdef RICH_MPI
            ResetPeakRSS();   // MEMORY_GEN peak_* = one whole generation each
#endif
            for (size_t gen = mgStartGeneration; gen < mgTotalGenerations; ++gen)
            {
                bool const firstBurninThisGen =
                    cfg.adaptiveSourceCells && gen < mgInitialBurninGenerations;
                bool const uniformBurninThisGen =
                    cfg.adaptiveSourceCells &&
                    gen >= mgInitialBurninGenerations &&
                    gen < mgBurninGenerations;
                bool const burninThisGen = firstBurninThisGen || uniformBurninThisGen;
                bool const learnedProbeThisGen =
                    cfg.adaptiveSourceCells &&
                    gen >= mgBurninGenerations &&
                    gen < mgProbeEndGeneration;
                bool const rampThisGen =
                    cfg.adaptiveSourceCells &&
                    gen >= mgProbeEndGeneration &&
                    gen < mgFinalStartGeneration;
                size_t const rampStage = rampThisGen ? gen - mgProbeEndGeneration : 0;
                // The extra packets (learned-weight cap, exploration splitting)
                // follow the ramp through their weight bound.
                double const rampWeightScale = rampThisGen
                    ? mgRampFullBudget / mgRampBudgets[rampStage] : 1.0;
                bool const finalThisGen =
                    !cfg.adaptiveSourceCells || gen >= mgFinalStartGeneration;
                size_t const finalGenerationIndex = finalThisGen
                    ? (cfg.adaptiveSourceCells ? gen - mgFinalStartGeneration : gen)
                    : 0;
                bool const adaptiveActiveThisGen =
                    cfg.adaptiveSourceCells &&
                    (learnedProbeThisGen || rampThisGen || finalThisGen) &&
                    !mgAdaptive.scoreByCellID.empty();
                size_t const photonsThisGen = firstBurninThisGen ? 1
                    : (uniformBurninThisGen ? 3
                       : (learnedProbeThisGen ? mgProbePhotons
                          : (cfg.adaptiveSourceCells ? 1 : genPhotonsPerCell)));
                std::string phase = "final";
                if (firstBurninThisGen)
                    phase = "burnin_exact1";
                else if (uniformBurninThisGen)
                    phase = "burnin_exact3";
                else if (learnedProbeThisGen)
                    phase = "learned_only_probe_exact" + std::to_string(mgProbePhotons);
                else if (rampThisGen)
                    phase = "final_budget_ramp";
                else if (cfg.adaptiveSourceCells)
                    phase = "learned_only_final";
                physics->setNewPhotonsPerCell(photonsThisGen);
                if (rank == 0)
                    std::cout << "Generation " << (gen + 1) << "/" << mgTotalGenerations
                              << " phase=" << phase
                              << " photons_per_cell_this_gen=" << photonsThisGen;
                if (rank == 0 && rampThisGen)
                    std::cout << " ramp_stage=" << (rampStage + 1) << "/" << mgRampGenerations
                              << " packets_per_learned_cell=" << mgRampBudgets[rampStage]
                              << " of " << mgRampFullBudget
                              << " weight_bound_scale=" << rampWeightScale;
                if (rank == 0 && finalThisGen)
                    std::cout << " final_step=" << (finalGenerationIndex + 1)
                              << "/" << mgFinalGenerations;
                if (rank == 0)
                    std::cout << std::endl;
                PrintAdaptiveGenerationStart("MG", cfg, mgAdaptive, gen, mgTotalGenerations,
                                             mgBurninGenerations,
                                             adaptiveActiveThisGen, rank);

                IMCPostProcessControl postProcessControl;
                if (adaptiveActiveThisGen && (finalThisGen || rampThisGen)) {
                    // Final and ramp generations: explicit Neyman allocation
                    // from the learned variance shares (see
                    // BuildNeymanSourceAllocation), at the ramp stage's budget.
                    Config const allocationCfg = rampThisGen
                        ? RampStageAllocationConfig(cfg, mgRampBudgets[rampStage]) : cfg;
                    SourceAllocationPlan const plan = BuildNeymanSourceAllocation(
                        BuildCombinedSourceScoresForIMC(mgAdaptive, mgGroupSourceState),
                        allocationCfg, &runtime.fluxSourceCellIDs);
                    ApplyNeymanAllocationToControl(postProcessControl, plan, allocationCfg);
                    PrintSourceAllocationPlan("MG", plan, gen, rank);
                } else if (adaptiveActiveThisGen) {
                    // Learned-only probe: flat 75 packets per learned cell,
                    // used to measure costs for the load balance.
                    postProcessControl.adaptiveCells.enabled = true;
                    postProcessControl.adaptiveCells.scores =
                        BuildCombinedSourceScoresForIMC(mgAdaptive, mgGroupSourceState);
                    postProcessControl.adaptiveCells.strength =
                        cfg.adaptiveSourceStrength;
                    postProcessControl.adaptiveCells.maxFactor =
                        cfg.adaptiveSourceMaxFactor;
                    postProcessControl.adaptiveCells.learnedReserveFraction =
                        cfg.adaptiveSourceLearnedReserveFrac;
                    postProcessControl.adaptiveCells.learnedMinFactor = 1.0;
                    postProcessControl.adaptiveCells.observerBudgetMultiplier =
                        mgAdaptive.observerBudgetMultiplier;
                    postProcessControl.adaptiveCells.learnedMinPhotons = 0;
                    postProcessControl.adaptiveCells.learnedMaxPhotons = 0;
                    postProcessControl.adaptiveCells.scorePower = 1.0;
                }
                if (firstBurninThisGen) {
                    postProcessControl.emission.enabled = true;
                    postProcessControl.emission.includeUniformBase = true;
                    postProcessControl.emission.baseMultiplier = 1;
                } else if (uniformBurninThisGen) {
                    postProcessControl.emission.enabled = true;
                    postProcessControl.emission.includeUniformBase = true;
                    postProcessControl.emission.baseMultiplier = 3;
                } else if (learnedProbeThisGen) {
                    postProcessControl.emission.enabled = true;
                    postProcessControl.emission.useLearnedScores = true;
                    postProcessControl.emission.includeUniformBase = false;
                    postProcessControl.emission.learnedBoostFactor = 1;
                } else if (cfg.adaptiveSourceCells && (finalThisGen || rampThisGen) &&
                           !adaptiveActiveThisGen) {
                    // No learned cells at all: keep the legacy behaviour so the
                    // run fails loudly (zero emission) rather than silently
                    // switching to uniform sampling.
                    postProcessControl.emission.enabled = true;
                    postProcessControl.emission.useLearnedScores = true;
                    postProcessControl.emission.includeUniformBase = false;
                    postProcessControl.emission.learnedBoostFactor = 1;
                    postProcessControl.emission.learnedExtraBudget = 0;
                }
                observer->resetGenerationSourceCellEscapeStats();
                observer->resetGenerationSourceCellGroupEscapeStats();
                observer->setGenerationSourceCellGroupStatsEnabled(
                    cfg.adaptiveGroupSourceCells && cfg.adaptiveGroupQuality &&
                    ENERGY_GROUPS_NUM > 1 && !burninThisGen);

                if (adaptiveActiveThisGen && cfg.adaptiveGroupFrequencySampling &&
                    !mgGroupSourceState.scoreByCellGroup.empty()) {
                    postProcessControl.adaptiveGroups.enabled = true;
                    postProcessControl.adaptiveGroups.scores = BuildGroupScoresForIMC(
                        mgGroupSourceState, cells, static_cast<size_t>(ENERGY_GROUPS_NUM));
                    postProcessControl.adaptiveGroups.strength =
                        cfg.adaptiveGroupStrength;
                    postProcessControl.adaptiveGroups.pdfFloor =
                        cfg.adaptiveGroupPdfFloor;
                    postProcessControl.adaptiveGroups.maxBias =
                        cfg.adaptiveGroupMaxBias;
                    postProcessControl.adaptiveGroups.maxWeightCorrection =
                        cfg.adaptiveGroupMaxWeightCorrection;
                }
                physics->configurePostProcessControl(
                    std::move(postProcessControl));
                if (cfg.fluxSourceCompare)
                    ConfigureFluxSourceForCurrentDecomposition(
                        cfg, runtime, *physics, *opacity, true);

                if (cfg.volumeEmissionEnabled && physics->hasPostProcessVolumeEmission()) {
                    // Burn-in generations sample a random subset of the emitting
                    // volume cells sized to the packet target; learned phases
                    // emit from every cell the allocation names.
                    double fraction = 1.0;
                    if (burninThisGen && runtime.volumeEmissionCells > 0) {
                        double const wanted = static_cast<double>(cfg.volumeEmissionBurninPacketsTarget);
                        double const full = static_cast<double>(photonsThisGen) *
                                            static_cast<double>(runtime.volumeEmissionCells);
                        fraction = std::clamp(wanted / std::max(1.0, full), 1e-6, 1.0);
                    }
                    physics->setPostProcessVolumeEmissionSubsample(fraction, static_cast<uint64_t>(gen) + 1);
                    // Bound the weight of exploration packets by a fraction of the
                    // escaping energy of the previous generation (face flux before
                    // the first one).
                    double const escapingReference = runtime.lastEscapingLuminosity > 0.0
                        ? runtime.lastEscapingLuminosity : runtime.fluxSourceInjectedLuminosity;
                    physics->setPostProcessExplorationMaxWeight(
                        cfg.volumeEmissionExplorationWeightFraction * escapingReference * cfg.sourceDt *
                        rampWeightScale);
                    // Probe, ramp and final generations bound the learned
                    // cells' packets the same way; the burn-in keeps its flat
                    // counts.
                    physics->setPostProcessLearnedMaxWeight(
                        cfg.volumeEmissionCapLearnedWeight && !burninThisGen
                            ? cfg.volumeEmissionExplorationWeightFraction * escapingReference * cfg.sourceDt *
                              rampWeightScale
                            : 0.0);
                }

                if (restarting)
                    physics->reseedRNG(RestartGenerationSeed(
                        RestartPass::Forward, rank,
                        mgCompletedFinalGenerations + finalGenerationIndex));
                else
                    physics->reseedRNG(static_cast<uint64_t>(rank+12345678) * mgTotalGenerations + gen);

                manager->getParticles().clear();
                InstallWeightSplit(cfg, runtime, *physics, mgSplit);
                manager->step(cells, cfg.transportTime);
                if (cfg.volumeEmissionEnabled && physics->hasPostProcessVolumeEmission()) {
                    double volumeEnergy = physics->getLastVolumeEmissionEnergy();
                    unsigned long long counts[2] = {
                        static_cast<unsigned long long>(physics->getLastVolumeEmissionCells()),
                        static_cast<unsigned long long>(physics->getLastVolumeEmissionSelectedCells())};
#ifdef RICH_MPI
                    MPI_Allreduce(MPI_IN_PLACE, &volumeEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
                    if (rank == 0)
                        std::cout << "VOLUME_EMISSION gen=" << gen + 1
                                  << " emitting_cells=" << counts[0]
                                  << " selected_cells=" << counts[1]
                                  << " volume_luminosity=" << volumeEnergy / cfg.sourceDt
                                  << " face_luminosity=" << runtime.fluxSourceInjectedLuminosity
                                  << " erg/s" << std::endl;
                }
#ifdef RICH_MPI
                ReportGenerationMemory("MG", gen + 1, rank, runtime.mpiSize, tess, manager->getParticles().size());
#endif
#ifdef __linux__
                // Return freed heap to the OS between generations; per-step
                // particle and communication buffers otherwise leave the RSS
                // high through fragmentation.
                malloc_trim(0);
#endif
                IMCPostProcessGenerationDiagnostics const generationDiagnostics =
                    physics->getPostProcessGenerationDiagnostics();
                if (finalThisGen && gen + 1 == mgTotalGenerations)
                    mgLastFinalMeasurements = CollectLocalLoadBalanceMeasurements(
                        cells, Ncells, manager->GetCellsStepsCounters(),
                        generationDiagnostics.sourcePhotonsPerCell);

                auto mgAllocation = ReduceSourceAllocationSummary(
                    generationDiagnostics.sourceAllocation);
                result.weightCapExtraPhotons += mgAllocation.weightCapExtraPhotons;
                result.weightCappedCells += mgAllocation.weightCappedCells;
                result.allocatedCells += mgAllocation.sourceCells;
                if (learnedProbeThisGen)
                    result.probePackets += mgAllocation.totalPhotons;
                size_t const photonHistMin = finalThisGen
                    ? cfg.adaptiveSourceLearnedMinPhotons
                    : 1;
                size_t const photonHistMax = finalThisGen
                    ? cfg.adaptiveSourceLearnedMaxPhotons
                    : std::max(photonHistMin, photonsThisGen * 20);
                auto mgPhotonDistribution = ReduceSourcePhotonDistribution(
                    generationDiagnostics.sourcePhotonsPerCell,
                    photonHistMin,
                    photonHistMax,
                    rank,
                    mpiSize);
                auto mgGroupSamplingDiag = ReduceGroupSamplingDiagnostics(
                    generationDiagnostics.groupSampling);
                mgLastGroupSamplingDiag = mgGroupSamplingDiag;
                auto mgSourceStats = observer->getGenerationSourceCellEscapeStats();
                observer->resetGenerationSourceCellEscapeStats();
                auto mgGroupSourceStats = observer->getGenerationSourceCellGroupEscapeStats();
                observer->resetGenerationSourceCellGroupEscapeStats();
                ObserverQualityDiagnostics mgObserverQuality;
                if (cfg.adaptiveSourceCells && cfg.adaptiveObserverEquity) {
                    mgObserverQuality = BuildObserverQualityDiagnostics(
                        CollectGlobalObserverQuality(observer->getObserverQualitySnapshot()),
                        cfg, mgAdaptive, finalThisGen);
                }

                ObserverGroupQualityDiagnostics mgGroupQuality;
                if (cfg.adaptiveGroupQuality && ENERGY_GROUPS_NUM > 1) {
                    auto groupSnap = observer->getObserverGroupQualitySnapshot();
                    CollectGlobalObserverGroupQuality(groupSnap);
                    if (cfg.adaptiveGroupFrequencySampling &&
                        groupSnap.groupCount != static_cast<size_t>(ENERGY_GROUPS_NUM)) {
                        throw UniversalError(
                            "--adaptive-group-frequency-sampling requires observer group count to match ENERGY_GROUPS_NUM");
                    }
                    mgGroupQuality = BuildObserverGroupQualityDiagnosticsFromSnapshot(
                        groupSnap, cfg, mgGroupHistory, cfg.sourceDt, finalThisGen);
                }

                auto mgUpdate = UpdateAdaptiveSourceScoresDistributed(
                    mgSourceStats,
                    BuildLocalSourceCellEmission(
                        cells, generationDiagnostics.sourcePhotonsPerCell),
                    cfg, mgAdaptive, mgObserverQuality,
                    !burninThisGen, rank, mpiSize);
                if (cfg.sourceDt > 0.0 && mgUpdate.totalEscapedEnergy > 0.0)
                    runtime.lastEscapingLuminosity = mgUpdate.totalEscapedEnergy / cfg.sourceDt;
                if (!burninThisGen)
                    AccumulateWeightSplitStatistics(cfg, runtime, mgUpdate, mgSplit);
                std::vector<SphericalObserver::SourceCellEscapeStat>().swap(mgSourceStats);

                AdaptiveGroupSourceUpdateSummary mgGroupUpdate;
                if (cfg.adaptiveGroupSourceCells && mgGroupQuality.enabled && !burninThisGen) {
                    mgGroupUpdate = UpdateAdaptiveSourceGroupScores(
                        mgGroupSourceStats, mgGroupQuality, cfg,
                        mgGroupSourceState, rank, mpiSize);
                    if (mgGroupUpdate.fallbackToIntegratedPath) {
                        mgGroupFallbackToIntegrated = true;
                        mgGroupFallbackReason = mgGroupUpdate.fallbackReason;
                    }
                }
                std::vector<SphericalObserver::SourceCellGroupEscapeStat>().swap(mgGroupSourceStats);

                bool const includeGenerationInFinal = finalThisGen;
                PrintAdaptiveGenerationStats(
                    "MG", cfg, mgAdaptive, mgUpdate, mgAllocation,
                    mgPhotonDistribution,
                    mgObserverQuality, gen, mgTotalGenerations,
                    mgBurninGenerations, adaptiveActiveThisGen, rank);
                PrintAdaptiveIterationSummary(
                    "MG", mgAdaptive, mgUpdate, mgAllocation,
                    mgObserverQuality, gen, mgTotalGenerations, phase,
                    photonsThisGen, finalThisGen, finalGenerationIndex,
                    mgFinalGenerations, adaptiveActiveThisGen,
                    includeGenerationInFinal, rank);
                if (finalThisGen && cfg.adaptiveSourceCells && cfg.adaptiveObserverEquity) {
                    bool const printBlock =
                        (finalGenerationIndex + 1) % 5 == 0 ||
                        finalGenerationIndex + 1 == mgFinalGenerations;
                    PrintAdaptiveConvergenceReport(
                        "MG", cfg, mgObserverQuality, finalGenerationIndex,
                        mgFinalGenerations, printBlock, rank);
                }
                PrintAdaptiveGroupGenerationStats(mgGroupQuality, mgGroupSamplingDiag, gen, rank);
                if (rank == 0 && cfg.adaptiveSourceCells)
                    std::cout << "MG learned cells after iteration " << (gen + 1)
                              << ": " << mgAdaptive.scoreByCellID.size() << std::endl;
                observer->addBoxEscapeEnergy(boundary->getEscapedEnergy());
                boundary->resetEscapedEnergy();
                observer->printHeavyPacketReport("MG", gen + 1, 10);
                observer->printEscapeOriginReport("MG", gen + 1);
                result.splitCopies += PrintWeightSplitReport(cfg, *physics, mgSplit, "MG", gen + 1, rank);
                PrintCompressionReport(cfg, *physics, "MG", gen + 1, rank);
                observer->mpiReduceToRank0();
                if (includeGenerationInFinal) {
                    if (rank == 0)
                        observer->accumulateCurrentTalliesForStatistics(cfg.sourceDt);
                    AccumulateGroupSamplingDiagnostics(
                        mgFinalGroupSamplingDiag, mgGroupSamplingDiag);
                    AccumulateAdaptiveGroupSourceSummary(
                        mgFinalGroupSourceSummary, mgGroupUpdate);
                    ++mgIncludedFinalGenerations;
                } else {
                    ++mgDiscardedBurninGenerations;
                }
                observer->resetTallies();
                if (cfg.adaptiveSourceCells && !mgAdaptive.burninCompletePrinted &&
                    gen + 1 == mgBurninGenerations)
                {
                    if (rank == 0)
                        std::cout << "MG adaptive source burn-in complete" << std::endl;
                    mgAdaptive.burninCompletePrinted = true;
                }

    #ifdef RICH_MPI
                RankStepImbalance const mgStepImbalance =
                    ComputeRankStepImbalance("MG", gen, manager->GetCellsStepsCounters(), rank);
                bool const doInitialMeasuredLB =
                    gen == 0 && measuredLBActive && !restarting &&
                    (cfg.fluxSourceCompare || !cfg.adaptiveSourceCells);
                bool const doPostAdaptiveMeasuredLB =
                    measuredLBActive &&
                    cfg.adaptiveSourceCells &&
                    learnedProbeThisGen &&
                    !mgAdaptive.postAdaptiveMeasuredLBDone;
                bool const doAdaptivePeriodicMeasuredLB =
                    measuredLBActive &&
                    cfg.adaptiveSourceCells &&
                    finalThisGen &&
                    finalGenerationIndex + 1 < mgFinalGenerations &&
                    // The probe LB balances the probe's allocation, not the
                    // final one (240x rank imbalance until final step 10 with
                    // exploration splitting), so also rebalance after final
                    // steps 1 and 2. A restart balanced on a final generation
                    // already, so it skips the one after step 1.
                    ((finalGenerationIndex == 0 && !restarting) ||
                     finalGenerationIndex == 1 ||
                     (finalGenerationIndex + 1) %
                         (cfg.measuredLBFinalPeriod > 0 ? cfg.measuredLBFinalPeriod : 10) == 0);
                // With exploration splitting, the probe puts most packets in a
                // few thick cells (ceil(E/cap) each), which cost almost nothing
                // in the burn-in (1-3 packets per cell). At a small cap that
                // crowds one rank past its memory (snapshot 109 at fraction
                // 0.005: 147M packets on one rank), so before the probe,
                // balance on the packet counts the probe will create, as the
                // grey pass does.
                bool const doPredictedProbeLB =
                    measuredLBActive &&
                    cfg.adaptiveSourceCells &&
                    cfg.volumeEmissionEnabled &&
                    cfg.volumeEmissionExplorationWeightFraction > 0.0 &&
                    mgLearnedProbeGenerations > 0 &&
                    gen + 1 == mgBurninGenerations;
                // With the final-budget ramp, the LB after the probe and after
                // every ramp generation balances on the packets the next
                // generation's plan will create (measured steps of this one),
                // so each budget step lands on a decomposition made for it.
                bool const doRampLB =
                    measuredLBActive && mgRampGenerations > 0 &&
                    (doPostAdaptiveMeasuredLB || rampThisGen);
                std::string const mgLBLabel = doPredictedProbeLB
                    ? "MEASURED_LB_PREDICTED_PROBE"
                    : doRampLB
                    ? "MEASURED_LB_RAMP_PLAN"
                    : doPostAdaptiveMeasuredLB
                    ? "MEASURED_LB_ADAPTIVE"
                    : (doAdaptivePeriodicMeasuredLB
                        ? "MEASURED_LB_ADAPTIVE_PERIODIC"
                        : "MEASURED_LB");
                if (rank == 0 && doPredictedProbeLB)
                    std::cout << "MG burn-in complete; running measured LB on predicted probe packets" << std::endl;
                if (rank == 0 && doPostAdaptiveMeasuredLB)
                    std::cout << "MG learned-only probe complete; running measured LB before final calculation" << std::endl;
                if (rank == 0 && doAdaptivePeriodicMeasuredLB)
                    std::cout << "MG periodic final measured LB after final step "
                              << (finalGenerationIndex + 1)
                              << ": rank_step_imbalance="
                              << mgStepImbalance.maxOverMean
                              << std::endl;
                // The just-finished generation has already been recorded into the
                // final-output statistics accumulator when it is eligible. Its step
                // counters are used here only to repartition later generations.
                // Repartition assumptions:
                //   - Each generation is independent (no census particles carried between them).
                //   - noHydroFeedback is true: gas state is not modified by MC generations.
                //   - Extensives can be rebuilt from cell primitives after repartition.
                //   - The observer statistics accumulator and adaptive scores are preserved.
                if (rank == 0 && doRampLB)
                    std::cout << "MG " << (rampThisGen ? "ramp stage complete" : "learned-only probe complete")
                              << "; the measured LB uses the next generation's planned packets" << std::endl;
                if (doInitialMeasuredLB || doPredictedProbeLB || doRampLB ||
                    doPostAdaptiveMeasuredLB || doAdaptivePeriodicMeasuredLB) {
                    PredictedProbePackets predictedProbePhotons;
                    if (doPredictedProbeLB)
                        predictedProbePhotons = PredictProbeSourcePackets(
                            cells, Ncells, runtime,
                            BuildCombinedSourceScoresForIMC(mgAdaptive, mgGroupSourceState),
                            generationDiagnostics.sourcePhotonsPerCell, mgProbePhotons,
                            cfg.volumeEmissionExplorationWeightFraction,
                            cfg.volumeEmissionCapLearnedWeight,
                            "MEASURED_LB_PREDICTED_PROBE", rank, mpiSize);
                    if (doPredictedProbeLB)
                        result.probePredictedPackets = predictedProbePhotons.total;
                    if (doRampLB) {
                        // The plan and weight bound the next generation will
                        // use: the next ramp stage, or the full final budget.
                        size_t const nextGen = gen + 1;
                        bool const nextIsRamp = nextGen < mgFinalStartGeneration;
                        double const nextBudget = nextIsRamp
                            ? mgRampBudgets[nextGen - mgProbeEndGeneration] : mgRampFullBudget;
                        Config const nextCfg = nextIsRamp ? RampStageAllocationConfig(cfg, nextBudget) : cfg;
                        SourceAllocationPlan const nextPlan = BuildNeymanSourceAllocation(
                            BuildCombinedSourceScoresForIMC(mgAdaptive, mgGroupSourceState),
                            nextCfg, &runtime.fluxSourceCellIDs);
                        double const escapingReference = runtime.lastEscapingLuminosity > 0.0
                            ? runtime.lastEscapingLuminosity : runtime.fluxSourceInjectedLuminosity;
                        double const capLuminosity = cfg.volumeEmissionEnabled
                            ? cfg.volumeEmissionExplorationWeightFraction * escapingReference *
                              (mgRampFullBudget / nextBudget)
                            : 0.0;
                        predictedProbePhotons = PredictPlannedSourcePackets(
                            cells, Ncells, runtime, nextPlan, capLuminosity,
                            cfg.volumeEmissionCapLearnedWeight, "MEASURED_LB_RAMP_PLAN", rank, mpiSize);
                    }
                    bool const predicted = doPredictedProbeLB || doRampLB;
                    rebalanceOnMeasurements(
                        manager->GetCellsStepsCounters(),
                        predicted ? predictedProbePhotons.perCell
                                  : generationDiagnostics.sourcePhotonsPerCell,
                        mgLBLabel, predicted);
                    if (doPredictedProbeLB && rank == 0)
                        std::cout << "MG predicted-probe measured load balance complete" << std::endl;
                    if (doPostAdaptiveMeasuredLB) {
                        mgAdaptive.postAdaptiveMeasuredLBDone = true;
                        mgAdaptive.adaptiveMeasuredLBCount += 1;
                        mgAdaptive.lastAdaptiveMeasuredLBGeneration = gen;
                        if (rank == 0)
                            std::cout << "MG post-adaptive measured load balance complete" << std::endl;
                    }
                    if (doAdaptivePeriodicMeasuredLB) {
                        mgAdaptive.adaptiveMeasuredLBCount += 1;
                        mgAdaptive.lastAdaptiveMeasuredLBGeneration = gen;
                        if (rank == 0)
                            std::cout << "MG periodic measured load balance complete" << std::endl;
                    }
                }
    #endif // RICH_MPI
            }

            // Restart state for a later continuation. With no generation run
            // this segment the stored measurements are carried over.
            LoadBalanceMeasurements mgSavedLoadBalance = FinishPassLoadBalance(
                mgLastFinalMeasurements, std::move(mgRestartLoadBalance),
                mgIncludedFinalGenerations + mgDiscardedBurninGenerations > 0, rank);

            // ============================================================
            // Finish diagnostics
            // ============================================================
            if (rank == 0) {
                observer->loadStatisticalMeanTallies();
                if (cfg.polarization) {
                    PrintPolarizationSummary(
                        "MG", observer->getObserverQualitySnapshot(), rank);
                    observer->printGenerationPolarizationSummary("MG");
                }
            }

            if (rank == 0) {
                SphericalObserver::Diagnostics diag;
                diag.sourceDt = cfg.sourceDt;
                diag.transportTime = cfg.transportTime;
                diag.mpiRanks = mpiSize;
                diag.comptonEnabled = cfg.compton ? 1 : 0;
                diag.emittedEnergy = observer->getEmittedEnergy();
                diag.absorbedEnergy = observer->getAbsorbedEnergy();
                diag.boxEscapeEnergy = observer->getBoxEscapeEnergy();
                diag.timedOutEnergy = observer->getTimedOutEnergy();
                diag.cutoffEnergy = observer->getCutoffEnergy();
                diag.compressionWork = observer->getCompressionWork();
                diag.snapshotTime = runtime.snapshotTime;
                diag.snapshotCycle = runtime.snapshotCycle;
                // Final generations in the result, like included_final_generations.
                diag.nGenerations = static_cast<int>(mgCompletedFinalGenerations + mgFinalGenerations);
                diag.includedFinalGenerations = static_cast<int>(
                    mgCompletedFinalGenerations + mgIncludedFinalGenerations);
                diag.discardedBurninGenerations = static_cast<int>(mgDiscardedBurninGenerations);
                diag.adaptiveOnlyFinalOutput = cfg.adaptiveSourceCells ? 1 : 0;
                diag.adaptiveGroupQualityEnabled = cfg.adaptiveGroupQuality ? 1 : 0;
                diag.adaptiveGroupSourceCellsEnabled = cfg.adaptiveGroupSourceCells ? 1 : 0;
                diag.adaptiveGroupFrequencySamplingEnabled = cfg.adaptiveGroupFrequencySampling ? 1 : 0;
                diag.adaptiveGroupHistoryEnabled = cfg.adaptiveGroupHistory ? 1 : 0;
                diag.adaptiveGroupLuminosityNormalization = cfg.adaptiveGroupLuminosityNormalization;
                diag.adaptiveGroupTargetNeff = cfg.adaptiveGroupTargetNeff;
                diag.adaptiveGroupTargetPolSnr = cfg.adaptiveGroupTargetPolSnr;
                diag.adaptiveGroupDeficitMax = cfg.adaptiveGroupDeficitMax;
                diag.adaptiveGroupMinCrossings = static_cast<int>(cfg.adaptiveGroupMinCrossings);
                diag.adaptiveGroupMinLuminosity = cfg.adaptiveGroupMinLuminosity;
                diag.adaptiveGroupMinLuminosityFracOfGroupMax =
                    cfg.adaptiveGroupMinLuminosityFracOfGroupMax;
                diag.adaptiveGroupLatestWeight = cfg.adaptiveGroupLatestWeight;
                diag.adaptiveGroupCumulativeWeight = cfg.adaptiveGroupCumulativeWeight;
                diag.adaptiveGroupEmaWeight = cfg.adaptiveGroupEmaWeight;
                diag.adaptiveGroupSamplingStrength = cfg.adaptiveGroupStrength;
                diag.adaptiveGroupSamplingPdfFloor = cfg.adaptiveGroupPdfFloor;
                diag.adaptiveGroupSamplingMaxBias = cfg.adaptiveGroupMaxBias;
                diag.adaptiveGroupSamplingMaxWeightCorrection = cfg.adaptiveGroupMaxWeightCorrection;
                diag.adaptiveGroupSamplingTotalSampled =
                    static_cast<unsigned long long>(mgFinalGroupSamplingDiag.totalSampled);
                diag.adaptiveGroupWeightCorrectionMin =
                    mgFinalGroupSamplingDiag.weightCorrectionMin;
                diag.adaptiveGroupWeightCorrectionMean =
                    (mgFinalGroupSamplingDiag.weightCorrectionCount > 0)
                        ? mgFinalGroupSamplingDiag.weightCorrectionSum /
                          static_cast<double>(mgFinalGroupSamplingDiag.weightCorrectionCount)
                        : 1.0;
                diag.adaptiveGroupWeightCorrectionMax =
                    mgFinalGroupSamplingDiag.weightCorrectionMax;
                diag.adaptiveGroupWeightCorrectionCappedFraction =
                    (mgFinalGroupSamplingDiag.weightCorrectionCount > 0)
                        ? static_cast<double>(mgFinalGroupSamplingDiag.weightCorrectionCapped) /
                          static_cast<double>(mgFinalGroupSamplingDiag.weightCorrectionCount)
                        : 0.0;
                diag.adaptiveGroupWeightCorrectionFallbackCount =
                    static_cast<unsigned long long>(mgFinalGroupSamplingDiag.weightCorrectionFallback);
                diag.adaptiveGroupInvalidPdfFallbackCount =
                    static_cast<unsigned long long>(mgFinalGroupSamplingDiag.invalidPdfFallback);
                diag.adaptiveGroupInvalidPdfFallbackPacketCount =
                    static_cast<unsigned long long>(mgFinalGroupSamplingDiag.invalidPdfFallbackPackets);
                diag.adaptiveGroupCappedEnergyFraction =
                    mgFinalGroupSamplingDiag.cappedEnergyFraction;
                diag.adaptiveGroupEstimatorPotentiallyBiased =
                    mgFinalGroupSamplingDiag.estimatorPotentiallyBiased ? 1 : 0;
                diag.adaptiveGroupFallbackToIntegratedPath =
                    mgGroupFallbackToIntegrated ? 1 : 0;
                diag.adaptiveGroupFallbackReason = mgGroupFallbackReason;
                diag.adaptiveGroupSourceLocalStatsAfterPrune =
                    mgFinalGroupSourceSummary.localStatsAfterPrune;
                diag.adaptiveGroupSourceLocalStatsDropped =
                    mgFinalGroupSourceSummary.localStatsDropped;
                diag.adaptiveGroupSourceMpiStatsExchanged =
                    mgFinalGroupSourceSummary.mpiStatsExchanged;
                diag.adaptiveGroupSourceMpiPackedBytes =
                    mgFinalGroupSourceSummary.maxPackedBytes;

                observer->writeHDF5(cfg.outputPath, diag);
                {
                    PassRestartState saved;
                    saved.adaptive = mgAdaptive;
                    saved.groupHistory = mgGroupHistory;
                    saved.groupSourceState = mgGroupSourceState;
                    saved.finalGroupSamplingDiag = mgFinalGroupSamplingDiag;
                    saved.finalGroupSourceSummary = mgFinalGroupSourceSummary;
                    saved.groupFallbackToIntegrated = mgGroupFallbackToIntegrated;
                    saved.groupFallbackReason = mgGroupFallbackReason;
                    saved.lastEscapingLuminosity = runtime.lastEscapingLuminosity;
                    saved.completedFinalGenerations =
                        mgCompletedFinalGenerations + mgIncludedFinalGenerations;
                    saved.loadBalance = std::move(mgSavedLoadBalance);
                    WritePassRestartState(cfg.outputPath, saved, *observer);
                }

                std::string const forwardVtk = BaseVtkOutputPath(cfg);
                if (!forwardVtk.empty()) {
                    observer->writeVTK(forwardVtk, cfg.sourceDt);

                    // Append FLD luminosity scalars to the VTK file
                    size_t const nObs = observer->getNumObservers();
                    std::vector<double> const& obsSolidAngles =
                        observer->getObserverSolidAngles();
                    std::vector<double> const& fldLuminosity =
                        runtime.fldLuminosity;
                    if (fldLuminosity.size() != nObs)
                        throw UniversalError("FLD luminosity size mismatch in forward VTK output");
                    std::ofstream vtkAppend(forwardVtk, std::ios::app);
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
                            double isoEquiv = (obsSolidAngles[p] > 0.0)
                                ? fldLuminosity[p] * fourPi / obsSolidAngles[p] : 0.0;
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
                            double patchArea_p = obsSolidAngles[p] * cfg.radius * cfg.radius;
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
                }

                double totalLum = observer->getTotalCrossingEnergy() / cfg.sourceDt;
                double residual = diag.emittedEnergy + diag.compressionWork - diag.absorbedEnergy
                                - diag.boxEscapeEnergy - diag.timedOutEnergy - diag.cutoffEnergy;
                double timedOutFrac = (diag.emittedEnergy > 0.0)
                    ? diag.timedOutEnergy / diag.emittedEnergy : 0.0;
                result.sourceLuminosity = runtime.fluxSourceInjectedLuminosity;
                result.emittedLuminosity = diag.emittedEnergy / cfg.sourceDt;
                result.crossingLuminosity = totalLum;
                result.crossingLuminosityStderr =
                    observer->getTotalLuminosityStderrGen(cfg.sourceDt);
                result.emittedEnergy = diag.emittedEnergy;
                result.timedOutFraction = timedOutFrac;
                FluxSourcePolarizationSummary const polSummary =
                    ComputeFluxSourcePolarizationSummary(
                        observer->getObserverQualitySnapshot());
                result.luminosityWeightedPolarizationDegree =
                    polSummary.luminosityWeightedDegree;
                result.polarizedObserverCount = polSummary.observerCount;

                std::cout << "\n=== TDE Post-Processing Results ===\n"
                          << "Generations:              " << mgFinalGenerations << "\n"
                          << "Final included generations: " << mgIncludedFinalGenerations << "\n"
                          << "Total final generations:  "
                          << (mgCompletedFinalGenerations + mgIncludedFinalGenerations)
                          << (restarting ? " (restart)" : "") << "\n"
                          << "Discarded burn-in generations: " << mgDiscardedBurninGenerations << "\n"
                          << "Final average policy:     " << (cfg.adaptiveSourceCells ? "adaptive_only" : "all_generations") << "\n"
                          << "Photons/cell/gen:         " << genPhotonsPerCell << "\n"
                          << "Total crossing luminosity: " << totalLum << " +/- "
                          << observer->getTotalLuminosityStderrGen(cfg.sourceDt)
                          << " erg/s (rel=" << observer->getTotalLuminosityRelErrGen(cfg.sourceDt) << ")\n"
                          << "Total FLD luminosity:     " << runtime.totalFldLuminosity << " erg/s\n"
                          << "Emitted energy:           " << diag.emittedEnergy << " erg\n"
                          << "Absorbed energy:          " << diag.absorbedEnergy << " erg\n"
                          << "Box escape energy:        " << diag.boxEscapeEnergy << " erg\n"
                          << "Timed-out energy:         " << diag.timedOutEnergy << " erg\n"
                          << "Cutoff energy:            " << diag.cutoffEnergy << " erg\n"
                          << "Compression work:         " << diag.compressionWork << " erg\n"
                          << "Sink residual:            " << residual << " erg\n"
                          << "Timed-out fraction:       " << timedOutFrac << "\n"
                          << "Output written to:        " << cfg.outputPath << "\n"
                          << std::endl;
            }
    result.ran = true;
    result.finalGenerationsAdded = mgIncludedFinalGenerations;
    result.finalGenerationsTotal = mgCompletedFinalGenerations + mgIncludedFinalGenerations;
    result.usesVelocity = cfg.useCellVelocities;
    result.usesDDMC = params.withDDMC;
    result.usesPolarization = cfg.polarization;
    result.usesCompton = cfg.compton;
    if (physics && !physics->getFactorFleck().empty())
        result.fleckFactors = physics->getFactorFleck();
    runtime.nCells = Ncells;
    return result;
}

} // namespace imc_postprocess_tde
