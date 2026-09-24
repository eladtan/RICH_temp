#ifndef IMC_POSTPROCESS_TDE_ADAPTIVE_STATISTICS_HPP
#define IMC_POSTPROCESS_TDE_ADAPTIVE_STATISTICS_HPP

#include <array>
#include <cstddef>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "IMCPostProcessControl.hpp"
#include "PostProcessConfig.hpp"
#include "PostProcessRuntime.hpp"

namespace imc_postprocess_tde {

struct AdaptiveSourceState
{
    std::unordered_map<size_t, double> scoreByCellID;
    std::vector<double> observerDeficitByIndex;
    std::vector<double> cumulativeObserverEnergy;
    std::vector<double> cumulativeObserverEnergyWeightSq;
    std::vector<double> cumulativeObserverPolarizationWeightSq;
    std::vector<double> cumulativeObserverStokesQ;
    std::vector<double> cumulativeObserverStokesU;
    std::vector<double> cumulativeObserverSumWQ2;
    std::vector<double> cumulativeObserverSumWU2;
    std::vector<unsigned long long> cumulativeObserverCrossings;
    double observerBudgetMultiplier = 1.0;
    bool burninCompletePrinted = false;
    bool postAdaptiveMeasuredLBDone = false;
    size_t adaptiveMeasuredLBCount = 0;
    size_t lastAdaptiveMeasuredLBGeneration = std::numeric_limits<size_t>::max();
};

struct ObserverQualityDiagnostics
{
    bool enabled = false;
    bool polarizationMode = false;
    size_t observerCount = 0;
    size_t weakObservers = 0;
    size_t zeroStatObservers = 0;
    double budgetMultiplier = 1.0;
    double deficitMin = 1.0;
    double deficitAvg = 1.0;
    double deficitMax = 1.0;
    double neffP05 = 0.0;
    double neffMedian = 0.0;
    double neffP95 = 0.0;
    double snrP05 = 0.0;
    double snrMedian = 0.0;
    double snrP95 = 0.0;
    // Absolute polarization-degree uncertainty per observer (0 when unknown).
    double sigmaP05 = 0.0;
    double sigmaPMedian = 0.0;
    double sigmaP95 = 0.0;
    // Median of the raw deficits before normalization; 1 when not normalizing.
    double deficitNormalization = 1.0;
    // Observers whose uncertainty could not be estimated yet (deficit = cap).
    size_t sentinelObservers = 0;
    std::vector<double> deficitByObserver;
    std::vector<double> neffByObserver;
    std::vector<double> snrByObserver;
    std::vector<double> sigmaPByObserver;
    std::vector<unsigned long long> crossingsByObserver;
};

// Packets emitted by one local source cell in the generation just finished.
// The adaptive score needs this to remove the 1/n dependence of the measured
// weight-squared contributions, so the learned quantity does not change when
// the allocation changes.
struct SourceCellEmission
{
    size_t cellID = 0;
    size_t photons = 0;
};

// Explicit per-cell packet counts for one final generation.
struct SourceAllocationPlan
{
    std::unordered_map<size_t, double> photonsByCell;
    unsigned long long totalPhotons = 0;
    size_t cells = 0;
    size_t flooredCells = 0;
    size_t cappedCells = 0;
    double budgetPhotons = 0.0;
    double minPhotons = 0.0;
    double medianPhotons = 0.0;
    double maxPhotons = 0.0;
    // (sum n_i)^2 / (N sum n_i^2): 1 for a flat allocation, smaller the more
    // the budget is concentrated.
    double concentration = 1.0;
    size_t faceCells = 0;
    size_t volumeCells = 0;
    unsigned long long facePhotons = 0;
    unsigned long long volumePhotons = 0;
};

constexpr double MEASURED_LB_MAX_CELL_IMBALANCE = 2.5;

struct RankStepImbalance
{
    unsigned long long localSteps = 0;
    unsigned long long globalSteps = 0;
    double meanRankSteps = 0.0;
    double maxRankSteps = 0.0;
    double maxOverMean = 0.0;
};

struct AdaptiveGroupHistory
{
    bool initialized = false;
    size_t observerCount = 0;
    size_t groupCount = 0;
    size_t updateCount = 0;
    size_t integratedUpdateCount = 0;

    std::vector<std::vector<double>> emaPriority;
    std::vector<std::vector<double>> emaDeficit;

    std::vector<std::vector<double>> cumulativeEnergy;
    std::vector<std::vector<double>> cumulativeWeightSq;
    std::vector<std::vector<double>> cumulativeStokesQ;
    std::vector<std::vector<double>> cumulativeStokesU;
    std::vector<std::vector<double>> cumulativeSumWQ2;
    std::vector<std::vector<double>> cumulativeSumWU2;
    std::vector<std::vector<size_t>> cumulativeCrossings;
};

struct ObserverGroupQualityDiagnostics
{
    bool enabled = false;
    bool polarizationMode = false;
    size_t observerCount = 0;
    size_t groupCount = 0;

    std::vector<std::vector<double>> luminosity;
    std::vector<std::vector<double>> neff;
    std::vector<std::vector<double>> polarizationDegree;
    std::vector<std::vector<double>> polarizationSnr;
    std::vector<std::vector<int>> polarizationSnrValid;
    std::vector<std::vector<double>> latestPriority;
    std::vector<std::vector<double>> cumulativePriority;
    std::vector<std::vector<double>> predictedPriority;
    std::vector<std::vector<double>> deficit;
    std::vector<std::vector<size_t>> crossings;
    size_t activeBins = 0;
    size_t highPriorityBins = 0;
    double neffP05 = 0.0;
    double neffMedian = 0.0;
    double neffP95 = 0.0;
    double polSnrP05 = 0.0;
    double polSnrMedian = 0.0;
    double polSnrP95 = 0.0;
};

struct AdaptiveGroupSourceState
{
    std::unordered_map<size_t, std::vector<double>> scoreByCellGroup;
    std::unordered_map<size_t, double> cellScoreFromGroups;
};

struct AdaptiveSourceUpdateSummary
{
    double totalEscapedEnergy = 0.0;
    unsigned long long totalCrossings = 0;
    size_t sourceCellObserverPairs = 0;
    size_t observersWithCrossings = 0;
    size_t passedCells = 0;
    size_t newCells = 0;
    size_t retainedCells = 0;
    size_t decayedCells = 0;
    unsigned long long maxLocalSourcePairs = 0;
    unsigned long long maxReceivedShardPairs = 0;
    unsigned long long maxPackedBytes = 0;
    size_t scoreDeltaCells = 0;
    size_t scoreMapCells = 0;
    std::vector<SphericalObserver::SourceCellEscapeStat> topStats;
};

struct AdaptiveGroupSourceUpdateSummary
{
    bool fallbackToIntegratedPath = false;
    std::string fallbackReason = "none";
    unsigned long long localStatsInput = 0;
    unsigned long long localStatsAfterPrune = 0;
    unsigned long long localStatsDropped = 0;
    unsigned long long maxLocalSourceGroupStats = 0;
    unsigned long long maxReceivedShardStats = 0;
    unsigned long long mpiStatsExchanged = 0;
    unsigned long long maxPackedBytes = 0;
    size_t passedStats = 0;
    size_t scoreDeltaCells = 0;
    size_t scoreMapCells = 0;
};



double EffectiveMeasuredLBWeightCompression(Config const& cfg);
RankStepImbalance ComputeRankStepImbalance(
    std::string const& label,
    size_t gen,
    std::vector<size_t> const& localSteps,
    int rank);
bool AdaptiveLBCooldownSatisfied(AdaptiveSourceState const& state, Config const& cfg, size_t gen);
void AppendZeroVtkScalar(std::ofstream& file, std::string const& name, size_t n);
void AccumulateAdaptiveGroupSourceSummary(
    AdaptiveGroupSourceUpdateSummary& total,
    AdaptiveGroupSourceUpdateSummary const& gen);
SphericalObserver::ObserverQualitySnapshot CollectGlobalObserverQuality(
    SphericalObserver::ObserverQualitySnapshot local);
ObserverQualityDiagnostics BuildObserverQualityDiagnostics(
    SphericalObserver::ObserverQualitySnapshot const& snap,
    Config const& cfg,
    AdaptiveSourceState& state,
    bool includeInIntegratedStats);
RadiationIMC::SourceAllocationSummary ReduceSourceAllocationSummary(
    RadiationIMC::SourceAllocationSummary local);
struct SourcePhotonDistribution
{
    struct Bin
    {
        size_t lowerInclusive = 0;
        size_t upperInclusive = 0;
        unsigned long long cellCount = 0;
        unsigned long long photonCount = 0;
    };

    std::vector<Bin> bins;
    unsigned long long emittingCells = 0;
    unsigned long long totalPhotons = 0;
    double p05 = 0.0;
    double p25 = 0.0;
    double p50 = 0.0;
    double p75 = 0.0;
    double p95 = 0.0;
};
SourcePhotonDistribution ReduceSourcePhotonDistribution(
    std::vector<size_t> const& localPhotonsPerCell,
    size_t binMinPhotons,
    size_t binMaxPhotons,
    int rank,
    int mpiSize);
void PrintSourcePhotonDistribution(
    std::string const& label,
    SourcePhotonDistribution const& dist,
    int rank);
RadiationIMC::GroupSamplingDiagnostics ReduceGroupSamplingDiagnostics(
    RadiationIMC::GroupSamplingDiagnostics local);
void AccumulateGroupSamplingDiagnostics(
    RadiationIMC::GroupSamplingDiagnostics& total,
    RadiationIMC::GroupSamplingDiagnostics const& gen);
std::vector<SourceCellEmission> BuildLocalSourceCellEmission(
    std::vector<ComputationalCell3D> const& cells,
    std::vector<size_t> const& photonsPerCell);
// faceCellIDs: cells owning emitting faces; every other learned cell is a
// volume emitter and uses the volume-emission budget and bounds. nullptr or
// empty treats all cells as face cells.
SourceAllocationPlan BuildNeymanSourceAllocation(
    std::unordered_map<size_t, double> const& scores,
    Config const& cfg,
    std::unordered_set<size_t> const* faceCellIDs = nullptr);
void ApplyNeymanAllocationToControl(
    IMCPostProcessControl& control,
    SourceAllocationPlan const& plan,
    Config const& cfg);
void PrintSourceAllocationPlan(
    std::string const& label,
    SourceAllocationPlan const& plan,
    size_t gen,
    int rank);
// Convergence of the observer map toward the polarization target: one
// key=value ADAPTIVE_REPORT line every final generation and, when
// printBlock is set, a human-readable block with percentiles, a histogram of
// sigma_P relative to the target and the projected generations still needed.
void PrintAdaptiveConvergenceReport(
    std::string const& label,
    Config const& cfg,
    ObserverQualityDiagnostics const& observerQuality,
    size_t finalGenerationIndex,
    size_t finalGenerations,
    bool printBlock,
    int rank);
AdaptiveSourceUpdateSummary UpdateAdaptiveSourceScoresDistributed(
    std::vector<SphericalObserver::SourceCellEscapeStat> const& localStats,
    std::vector<SourceCellEmission> const& localEmission,
    Config const& cfg,
    AdaptiveSourceState& state,
    ObserverQualityDiagnostics const& observerQuality,
    bool adaptiveActive,
    int rank,
    int mpiSize);
void PrintAdaptiveGenerationStart(
    std::string const& label,
    Config const& cfg,
    AdaptiveSourceState const& state,
    size_t gen,
    size_t totalGenerations,
    size_t burninGenerations,
    bool adaptiveActive,
    int rank);
void PrintAdaptiveGenerationStats(
    std::string const& label,
    Config const& cfg,
    AdaptiveSourceState const& state,
    AdaptiveSourceUpdateSummary const& update,
    RadiationIMC::SourceAllocationSummary allocation,
    SourcePhotonDistribution const& photonDistribution,
    ObserverQualityDiagnostics const& observerQuality,
    size_t gen,
    size_t totalGenerations,
    size_t burninGenerations,
    bool adaptiveActive,
    int rank);
void CollectGlobalObserverGroupQuality(SphericalObserver::ObserverGroupQualitySnapshot& snap);
ObserverGroupQualityDiagnostics BuildObserverGroupQualityDiagnosticsFromSnapshot(
    SphericalObserver::ObserverGroupQualitySnapshot const& snap,
    Config const& cfg,
    AdaptiveGroupHistory& history,
    double sourceDt,
    bool includeInIntegratedStats);
AdaptiveGroupSourceUpdateSummary UpdateAdaptiveSourceGroupScores(
    std::vector<SphericalObserver::SourceCellGroupEscapeStat> const& localGroupStats,
    ObserverGroupQualityDiagnostics const& groupQuality,
    Config const& cfg,
    AdaptiveGroupSourceState& groupState,
    int rank,
    int mpiSize);
void PrintAdaptiveIterationSummary(
    std::string const& label,
    AdaptiveSourceState const& state,
    AdaptiveSourceUpdateSummary const& update,
    RadiationIMC::SourceAllocationSummary const& allocation,
    ObserverQualityDiagnostics const& observerQuality,
    size_t gen,
    size_t totalGenerations,
    std::string const& phase,
    size_t photonsThisGen,
    bool finalThisGen,
    size_t finalGenerationIndex,
    size_t finalGenerations,
    bool adaptiveActive,
    bool includeInFinal,
    int rank);
void PrintAdaptiveGroupGenerationStats(
    ObserverGroupQualityDiagnostics const& gq,
    RadiationIMC::GroupSamplingDiagnostics const& gsd,
    size_t gen,
    int rank);
void PrintPolarizationSummary(
    std::string const& label,
    SphericalObserver::ObserverQualitySnapshot const& snap,
    int rank);
std::unordered_map<size_t, std::array<double, ENERGY_GROUPS_NUM>> BuildGroupScoresForIMC(
    AdaptiveGroupSourceState const& groupState,
    std::vector<ComputationalCell3D> const& localCells,
    size_t nGroups);
std::unordered_map<size_t, double> BuildCombinedSourceScoresForIMC(
    AdaptiveSourceState const& integratedState,
    AdaptiveGroupSourceState const& groupState);
void RecomputeOpacityScaleFactors(
    OpacityCalculator& opacity,
    OpacityCalculator const& greyOpacity,
    std::vector<ComputationalCell3D> const& cells,
    size_t const nCells,
    int const rank,
    OpacityScaleMode const mode,
    double const alphaMin,  // 0 = unbounded
    double const alphaMax,  // 0 = unbounded
    std::function<void(std::unordered_map<size_t, double>)> const&
        applyScaleFactors,
    std::string const& phaseLabel);
void PrintVmRSS(std::string const& label, int rank);
// Resident set size of this process in kB (0 when unavailable).
size_t CurrentVmRSSkB();
// Memory breakdown of this process, all in kB (0 when unavailable): RSS split
// into anonymous, file-backed and shared-memory pages, plus glibc heap in use
// and heap free-but-retained (the latter is what malloc_trim can return).
struct ProcessMemoryKB
{
    size_t rss = 0, anon = 0, file = 0, shmem = 0, heapInUse = 0, heapFree = 0, heapMmap = 0;
};
ProcessMemoryKB CurrentProcessMemoryKB();
// Prints MEMORY_GEN (rank 0: means, worst node sums via a shared-memory
// communicator) and MEMORY_GEN_MAXRANK (heaviest rank: what it holds) to
// stderr. Collective on MPI_COMM_WORLD.
void ReportGenerationMemory(char const* pass, size_t generation, int rank, int mpiSize,
                            Tessellation3D const& tess, size_t censusParticles);

} // namespace imc_postprocess_tde

#endif // IMC_POSTPROCESS_TDE_ADAPTIVE_STATISTICS_HPP
