#ifndef RICH_IMC_POSTPROCESS_RESTART_HPP
#define RICH_IMC_POSTPROCESS_RESTART_HPP

// Continuing a post-process result with more final generations.
//
// Each pass of a result bundle carries, besides its tallies:
//   /passes/<pass>/statistics_state  running sums of every final generation
//   /passes/<pass>/learned_state     adaptive allocation state after the last one
//   /passes/<pass>/load_balance      step/packet counts of the last generation
// A restart loads all three, skips the burn-in and probe generations, balances
// on the stored counts, and adds generations to the loaded sums. The burn-in is
// never rerun: a file without this state cannot be continued.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "AdaptiveStatistics.hpp"
#include "IMCPostProcess.hpp"
#include "IMCPostProcessControl.hpp"
#include "PostProcessConfig.hpp"
#include "PostProcessRuntime.hpp"
#include "WeightSplit.hpp"

namespace imc_postprocess_tde {

// 2: compression clock and work, split-aware polarization second moments and
// generation cross sums.
// 3: the multigroup pass's weight-split state when transport.split-per-group
// is on (learned_state/weight_split); schema-2 files are still read with the
// per-group window off and seed offset 0 (the must-match legacy values).
constexpr int kRestartSchemaVersion = 3;
constexpr int kOldestReadableRestartSchema = 2;

enum class RestartPass { Forward, Grey };

// "/passes/forward" or "/passes/grey".
std::string RestartPassGroup(RestartPass pass);

// Cells with nonzero steps or source packets in the last final generation of
// a pass, keyed by snapshot cell ID. Every other cell is a zero measurement.
struct LoadBalanceMeasurements
{
    std::vector<size_t> cellIDs;
    std::vector<size_t> steps;
    std::vector<size_t> particles;
};

// Weight-split state of the multigroup pass (schema 3, written only with
// transport.split-per-group): the per-cell escaped and emitted energies of
// every cell that has either, keyed by snapshot cell ID (rank 0), the
// accumulated flag, and the lagged escaping spectrum with its validity and
// the absolute generation that produced it.
struct WeightSplitRestartState
{
    bool present = false;
    bool accumulated = false;
    std::vector<size_t> cellIDs;
    std::vector<double> escaped;
    std::vector<double> emitted;
    std::vector<double> spectrum;
    bool spectrumValid = false;
    long spectrumGeneration = -1;
    // Group boundaries the spectrum and the observer statistics refer to.
    std::vector<double> groupBoundaries;
};

// The state a pass needs to continue. Learned state is identical on every
// rank; loadBalance and split are held by rank 0 only.
struct PassRestartState
{
    AdaptiveSourceState adaptive;
    AdaptiveGroupHistory groupHistory;
    AdaptiveGroupSourceState groupSourceState;
    IMCGroupSamplingDiagnostics finalGroupSamplingDiag;
    AdaptiveGroupSourceUpdateSummary finalGroupSourceSummary;
    bool groupFallbackToIntegrated = false;
    std::string groupFallbackReason = "none";
    // Escaping luminosity of this pass's last generation: the reference for
    // the exploration-weight cap and the volume-emission cutoff.
    double lastEscapingLuminosity = 0.0;
    // Final generations already in the statistics (all earlier segments).
    size_t completedFinalGenerations = 0;
    LoadBalanceMeasurements loadBalance;
    WeightSplitRestartState split;
    // Absolute index the next final generation of this pass takes (-1 when
    // the file predates it): generation records and the lagged spectrum
    // continue from it, whatever the ramp settings of the continuation.
    long nextGenerationID = -1;
};

// Gathers the multigroup weight-split state to rank 0 for the restart file.
// Collective; call on all ranks before the rank-0 write.
WeightSplitRestartState GatherWeightSplitRestartState(
    WeightSplitState const& state,
    std::vector<ComputationalCell3D> const& cells,
    size_t nCells,
    std::vector<double> const& groupBoundaries,
    int rank);

// Hands each rank the stored weight-split state of its cells (by ID) and the
// spectrum. Throws on every rank when the state is missing (the per-group
// window cannot continue without it) or names cells that are not in this
// mesh. Collective; call before the initial restart rebalance.
void RestoreWeightSplitState(
    WeightSplitRestartState const& stored,
    std::vector<ComputationalCell3D> const& cells,
    size_t nCells,
    std::vector<double> const& groupBoundaries,
    WeightSplitState& state,
    int rank);

// Settings compatibility between the run and the file it continues. Throws on
// every rank, listing all differences, when a setting that changes the
// estimate differs. Collective.
void CheckRestartCompatibility(
    std::string const& path,
    PostProcessIMC::PostProcessConfig const& effectiveConfig,
    double snapshotTime,
    int snapshotCycle,
    int rank);

// Compares the recomputed source surface and FLD luminosities with the ones
// the file stores. Collective; call after InitializeFluxSourceSurface.
void CheckRestartSourceSurface(
    std::string const& path, PostprocessRuntime const& runtime, int rank);

// Start of a restarted pass: loads its learned state, refuses a file without
// learned cells when the adaptive allocation is on, and loads the running
// statistics into rank 0's observer. Collective.
PassRestartState BeginPassRestart(
    Config const& cfg, RestartPass pass, SphericalObserver& observer,
    size_t generationsToAdd, int rank);

// Balances the mesh on the stored last-generation measurements by handing
// each rank its cells' counts to `rebalance(steps, particles)`. Collective.
void RebalanceFromStoredMeasurements(
    LoadBalanceMeasurements const& stored,
    std::vector<ComputationalCell3D> const& cells,
    size_t nCells,
    std::string const& passName,
    std::function<void(std::vector<size_t> const&, std::vector<size_t> const&)> const& rebalance,
    int rank);

// Load-balance measurements to save at the end of a pass: the gathered last
// final generation, or the loaded ones when this segment ran no generation.
// Collective.
LoadBalanceMeasurements FinishPassLoadBalance(
    LoadBalanceMeasurements const& localLastFinal,
    LoadBalanceMeasurements stored,
    bool ranGenerations,
    int rank);

// Rank 0: appends statistics_state, learned_state and load_balance at the root
// of a pass output file (MergeHdf5 moves them under /passes/<pass>).
void WritePassRestartState(
    std::string const& filename,
    PassRestartState const& state,
    SphericalObserver const& observer);

// Touched cells of the local mesh. `steps` and `particles` are per local cell.
LoadBalanceMeasurements CollectLocalLoadBalanceMeasurements(
    std::vector<ComputationalCell3D> const& cells,
    size_t nCells,
    std::vector<size_t> const& steps,
    std::vector<size_t> const& particles);

// RNG seed of a restart generation. It depends only on the pass, the rank and
// the generation's index among all final generations of the result, so seeds
// never repeat across segments and a chain of restarts reproduces one longer
// restart on the same decomposition. Fresh runs keep their original seeds.
uint64_t RestartGenerationSeed(RestartPass pass, int rank, size_t absoluteFinalGeneration);

// Replicate seeds (transport.seed-offset): offset 0 returns legacySeed
// unchanged; any other offset R returns
// SplitMix64(legacySeed XOR SplitMix64(R XOR kSeedOffsetDomain)), applied to
// both the fresh and the restart seed of every pass, rank and generation.
uint64_t ApplySeedOffset(uint64_t legacySeed, uint64_t offset);

// Final generations a pass runs this segment.
size_t GreyGenerationCount(Config const& cfg);

} // namespace imc_postprocess_tde

#endif // RICH_IMC_POSTPROCESS_RESTART_HPP
