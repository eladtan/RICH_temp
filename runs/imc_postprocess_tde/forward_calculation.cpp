#include "forward_calculation.hpp"

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

#ifdef RICH_MPI
#include <mpi.h>
#include "source/mpi/mpi_commands.hpp"
#endif

#include "adaptive_statistics.hpp"
#include "flux_source_calculation.hpp"
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
            using Particle3D = MonteCarloParticle<Vector3D, Tessellation3D>;

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
            double const measuredLBWeightCompression = EffectiveMeasuredLBWeightCompression(cfg);

            AdaptiveSourceState mgAdaptive;
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

            size_t const mgInitialBurninGenerations = cfg.adaptiveSourceCells ? 1 : 0;
            size_t const mgUniformBurninGenerations = cfg.adaptiveSourceCells ? 19 : 0;
            size_t const mgBurninGenerations = mgInitialBurninGenerations + mgUniformBurninGenerations;
            size_t const mgLearnedProbeGenerations = cfg.adaptiveSourceCells ? 1 : 0;
            size_t const mgFinalStartGeneration = mgBurninGenerations + mgLearnedProbeGenerations;
            size_t const mgFinalGenerations = cfg.nGenerations;
            size_t const mgTotalGenerations = cfg.adaptiveSourceCells
                ? mgFinalStartGeneration + mgFinalGenerations
                : cfg.nGenerations;
            for (size_t gen = 0; gen < mgTotalGenerations; ++gen)
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
                    gen < mgFinalStartGeneration;
                bool const finalThisGen =
                    !cfg.adaptiveSourceCells || gen >= mgFinalStartGeneration;
                size_t const finalGenerationIndex = finalThisGen
                    ? (cfg.adaptiveSourceCells ? gen - mgFinalStartGeneration : gen)
                    : 0;
                bool const adaptiveActiveThisGen =
                    cfg.adaptiveSourceCells &&
                    (learnedProbeThisGen || finalThisGen) &&
                    !mgAdaptive.scoreByCellID.empty();
                size_t const photonsThisGen = firstBurninThisGen ? 1
                    : (uniformBurninThisGen ? 3
                       : (learnedProbeThisGen ? 75
                          : (cfg.adaptiveSourceCells ? 1 : genPhotonsPerCell)));
                std::string phase = "final";
                if (firstBurninThisGen)
                    phase = "burnin_exact1";
                else if (uniformBurninThisGen)
                    phase = "burnin_exact3";
                else if (learnedProbeThisGen)
                    phase = "learned_only_probe_exact75";
                else if (cfg.adaptiveSourceCells)
                    phase = "learned_only_final";
                physics->setNewPhotonsPerCell(photonsThisGen);
                if (rank == 0)
                    std::cout << "Generation " << (gen + 1) << "/" << mgTotalGenerations
                              << " phase=" << phase
                              << " photons_per_cell_this_gen=" << photonsThisGen;
                if (rank == 0 && finalThisGen)
                    std::cout << " final_step=" << (finalGenerationIndex + 1)
                              << "/" << mgFinalGenerations;
                if (rank == 0)
                    std::cout << std::endl;
                PrintAdaptiveGenerationStart("MG", cfg, mgAdaptive, gen, mgTotalGenerations,
                                             mgBurninGenerations,
                                             adaptiveActiveThisGen, rank);

                if (adaptiveActiveThisGen) {
                    auto combinedSourceScores =
                        BuildCombinedSourceScoresForIMC(mgAdaptive, mgGroupSourceState);
                    double const learnedMinFactorThisGen =
                        learnedProbeThisGen ? 1.0 : cfg.adaptiveSourceLearnedMinFactor;
                    size_t const learnedMinPhotonsThisGen =
                        finalThisGen ? cfg.adaptiveSourceLearnedMinPhotons : 0;
                    size_t const learnedMaxPhotonsThisGen =
                        finalThisGen ? cfg.adaptiveSourceLearnedMaxPhotons : 0;
                    double const scorePowerThisGen =
                        finalThisGen ? cfg.adaptiveSourceScorePower : 1.0;
                    physics->setAdaptiveSourceCellScores(
                        std::move(combinedSourceScores),
                        cfg.adaptiveSourceStrength,
                        cfg.adaptiveSourceMaxFactor,
                        cfg.adaptiveSourceLearnedReserveFrac,
                        learnedMinFactorThisGen,
                        mgAdaptive.observerBudgetMultiplier,
                        learnedMinPhotonsThisGen,
                        learnedMaxPhotonsThisGen,
                        scorePowerThisGen);
                } else {
                    physics->clearAdaptiveSourceCellScores();
                }
                if (firstBurninThisGen)
                    physics->setSourceEmissionControl(false, true, 1);
                else if (uniformBurninThisGen)
                    physics->setSourceEmissionControl(false, true, 3);
                else if (learnedProbeThisGen)
                    physics->setSourceEmissionControl(true, false, 1, 1);
                else if (cfg.adaptiveSourceCells && finalThisGen)
                    physics->setSourceEmissionControl(true, false, 1, 1, 0);
                else
                    physics->clearSourceEmissionControl();
                observer->resetGenerationSourceCellEscapeStats();
                observer->resetGenerationSourceCellGroupEscapeStats();
                observer->setGenerationSourceCellGroupStatsEnabled(
                    cfg.adaptiveGroupSourceCells && cfg.adaptiveGroupQuality &&
                    ENERGY_GROUPS_NUM > 1 && !burninThisGen);

                if (adaptiveActiveThisGen && cfg.adaptiveGroupFrequencySampling &&
                    !mgGroupSourceState.scoreByCellGroup.empty()) {
                    auto groupScoresForIMC = BuildGroupScoresForIMC(
                        mgGroupSourceState, cells, static_cast<size_t>(ENERGY_GROUPS_NUM));
                    physics->setAdaptiveSourceCellGroupScores(
                        std::move(groupScoresForIMC),
                        cfg.adaptiveGroupStrength,
                        cfg.adaptiveGroupPdfFloor,
                        cfg.adaptiveGroupMaxBias,
                        cfg.adaptiveGroupMaxWeightCorrection);
                } else {
                    physics->clearAdaptiveSourceCellGroupScores();
                }

                physics->reseedRNG(static_cast<uint64_t>(rank+12345678) * mgTotalGenerations + gen);

                std::vector<Particle3D> empty;
                auto remaining = manager->step(std::move(empty), cells, cfg.transportTime);
                (void)remaining;

                auto mgAllocation = ReduceSourceAllocationSummary(
                    physics->getLastSourceAllocationSummary());
                size_t const photonHistMin = finalThisGen
                    ? cfg.adaptiveSourceLearnedMinPhotons
                    : 1;
                size_t const photonHistMax = finalThisGen
                    ? cfg.adaptiveSourceLearnedMaxPhotons
                    : std::max(photonHistMin, photonsThisGen * 20);
                auto mgPhotonDistribution = ReduceSourcePhotonDistribution(
                    physics->getLastSourcePhotonsPerCell(),
                    photonHistMin,
                    photonHistMax,
                    rank,
                    mpiSize);
                auto mgGroupSamplingDiag = ReduceGroupSamplingDiagnostics(
                    physics->getLastGroupSamplingDiagnostics());
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
                    mgSourceStats, cfg, mgAdaptive, mgObserverQuality,
                    !burninThisGen, rank, mpiSize);
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
                PrintAdaptiveGroupGenerationStats(mgGroupQuality, mgGroupSamplingDiag, gen, rank);
                if (rank == 0 && cfg.adaptiveSourceCells)
                    std::cout << "MG learned cells after iteration " << (gen + 1)
                              << ": " << mgAdaptive.scoreByCellID.size() << std::endl;
                observer->addBoxEscapeEnergy(boundary->getEscapedEnergy());
                boundary->resetEscapedEnergy();
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
                    (gen == 0 && measuredLBActive && !cfg.adaptiveSourceCells);
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
                    (finalGenerationIndex + 1) % 10 == 0;
                std::string const mgLBLabel = doPostAdaptiveMeasuredLB
                    ? "MEASURED_LB_ADAPTIVE"
                    : (doAdaptivePeriodicMeasuredLB
                        ? "MEASURED_LB_ADAPTIVE_PERIODIC"
                        : "MEASURED_LB");
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
                if (doInitialMeasuredLB ||
                    doPostAdaptiveMeasuredLB || doAdaptivePeriodicMeasuredLB) {
                    if (!params.noHydroFeedback) {
                        throw UniversalError("Measured load balance repartition requires noHydroFeedback=true");
                    }

                    PrintVmRSS("before_measured_lb", rank);

                    std::vector<double> measuredWeightsForExchange;

                    {
                        auto const& localSteps = manager->GetCellsStepsCounters();

                        std::vector<size_t> cellIDs(Ncells);
                        for (size_t i = 0; i < Ncells; ++i)
                            cellIDs[i] = cells[i].ID;

                        auto localMeasurements = imc_measured_lb::BuildLocalMeasurements(
                            cellIDs, localSteps, physics->getLastSourcePhotonsPerCell());

                        uint64_t localTotalSteps = 0;
                        for (auto const& m : localMeasurements)
                            localTotalSteps += static_cast<uint64_t>(m.stepCount);

                        uint64_t globalTotalSteps = 0;
                        MPI_Allreduce(&localTotalSteps, &globalTotalSteps, 1,
                                      MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

                        if (globalTotalSteps == 0) {
                            if (rank == 0)
                                std::cerr << mgLBLabel
                                          << ": measured generation had zero total steps, skipping repartition\n";
                        } else {
                            // Global-mean-based clamp via MPI_Allreduce (no per-cell allgather).
                            auto localCostByCellID = imc_measured_lb::BuildMeasuredCosts(
                                localMeasurements, measuredLBParams, isMultigroup, MPI_COMM_WORLD);

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
                            costCalcParams.floorCost = measuredLBParams.floorCost;
                            costCalcParams.missingCellCost = measuredLBParams.missingCellCost;
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

                            tess.BuildParallel(currentPoints, lbWeightsNew);
                        }
                    }

                    PrintVmRSS("after_build_parallel", rank);

                    if (!measuredWeightsForExchange.empty()) {
                        MPI_exchange_data(tess, cells, false, 1, &runtime.dummyCell);

                        double dummyWeight = measuredLBParams.missingCellCost;
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

                        extensives.resize(Ncells);
                        for (size_t i = 0; i < Ncells; ++i)
                            PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

    #if ENERGY_GROUPS_NUM > 1
                        if (cfg.opacityScaleMode != OpacityScaleMode::None) {
                            // Measured LB changes which cell IDs are local to each MPI rank.
                            // The alpha table is local and ID-keyed, so refresh it
                            // before rebuilding physics and before the next transport step.
                            RecomputeOpacityScaleFactors(
                                *opacity, *greyOpacity, cells, Ncells, rank, cfg.opacityScaleMode, "after measured LB repartition");
                        }
    #endif

                        boundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(tess);

                        physics = std::make_shared<RadiationIMC>(
                            tess, boundary, cells, extensives, eos, opacity, params);
                        physics->setObserver(observer);
                        if(cfg.fluxSourceCompare)
                            ConfigureFluxSourceForCurrentDecomposition(
                                cfg, runtime, *physics);

                        popControl = std::make_shared<NoPopulationControl<Vector3D, Tessellation3D>>(tess);

                        manager = std::make_shared<RDMAMonteCarloManager3D>(
                            tess, physics, popControl, boundary);

                        if (rank == 0)
                            std::cout << mgLBLabel
                                      << ": repartitioned, new local cells=" << Ncells << std::endl;

                        PrintVmRSS("after_rebuild_physics", rank);
                    }
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

            // ============================================================
            // Finish diagnostics
            // ============================================================
            if (rank == 0) {
                observer->loadStatisticalMeanTallies();
                if (cfg.polarization)
                    PrintPolarizationSummary(
                        "MG", observer->getObserverQualitySnapshot(), rank);
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
                diag.snapshotTime = runtime.snapshotTime;
                diag.snapshotCycle = runtime.snapshotCycle;
                diag.nGenerations = static_cast<int>(mgFinalGenerations);
                diag.includedFinalGenerations = static_cast<int>(mgIncludedFinalGenerations);
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
                double residual = diag.emittedEnergy - diag.absorbedEnergy
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
                          << "Sink residual:            " << residual << " erg\n"
                          << "Timed-out fraction:       " << timedOutFrac << "\n"
                          << "Output written to:        " << cfg.outputPath << "\n"
                          << std::endl;
            }
    result.ran = true;
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
