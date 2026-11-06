#ifndef IMC_POSTPROCESS_TDE_WEIGHT_SPLIT_HPP
#define IMC_POSTPROCESS_TDE_WEIGHT_SPLIT_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "AdaptiveStatistics.hpp"
#include "PostProcessConfig.hpp"
#include "PostProcessRuntime.hpp"

namespace imc_postprocess_tde {

// Weight window (transport.split-observer-fraction): a packet that starts a
// step in cell c while heavier than w_max / I_c is split into equal copies
// (RadiationIMC::setPostProcessWeightSplit), where
//   w_max = fraction x (last generation's escaping energy) / (observer count)
//   I_c   = min(1, p_c / split-escape-reference),
// and p_c is the cell's escaped energy over its emitted energy (volume plus
// face emission), accumulated over the generations in which every cell emits
// its full energy. Deep cells (p_c ~ 0) never split, so copies are made only
// along paths that lead out, where a single heavy packet would otherwise
// carry many times an observer's share.
//
// Each packet carries the product k of its splits (Particle::splitFactor),
// and the adaptive source statistics record k weight^2, the weight-squared
// of the unsplit packet. Without it a split cell's packets would show 1/k of
// their variance, the allocation would move packets away from the cell, its
// fewer and heavier packets would split more, and the cell would drift to its
// packet floor.
//
// Multigroup pass with transport.split-per-group: the window of a packet in
// lab-frame group g is w_max x split-group-scale x max(split-group-floor, f_g),
// f_g the share of g in the escaping energy of the last completed
// full-emission generation (docs/radiation/postprocess-band-weight-window.md).
struct WeightSplitState
{
    // Per local cell of the current decomposition; they migrate with the
    // cells on a repartition (MigrateWeightSplitState).
    std::vector<double> escapedEnergy;
    std::vector<double> emittedEnergy;
    // Set on every rank by the first accumulation, so all ranks agree on
    // whether there is anything to migrate (a rank may own no cells).
    bool accumulated = false;
    // Window of the generation being transported (for the report).
    double maxWeight = 0.0;
    unsigned long long importantCells = 0;

    // Per-group window (multigroup pass, transport.split-per-group) and
    // packet provenance: the escaping spectrum of the last completed
    // full-emission generation (raw outward-crossing energy per group, the
    // same on every rank), whether it is valid, and the absolute generation
    // that produced it (-1 = none).
    std::vector<double> groupSpectrum;
    bool spectrumValid = false;
    long long spectrumGeneration = -1;
    // Windows installed for the generation being transported (reports):
    // the active group windows (empty when the per-group mode is off or the
    // spectrum is not valid yet) and the provenance candidates.
    std::vector<double> groupMaxWeight;
    std::array<std::vector<double>, 2> candidateMaxWeight;
    double importanceFloor = 0.0;
    // The spectrum those windows came from (shares; -1 = none installed).
    std::vector<double> installedShares;
    long long installedSpectrumGeneration = -1;
};

// Scales of the two provenance candidate windows (split-group-scale values
// the diagnostic classifies against).
constexpr std::array<double, 2> kProvenanceCandidateScales = {1.0, 0.2};

// Group windows W_g = scalarWindow x scale x max(floor, f_g), f_g the share
// of group g in `spectrum`. Throws UniversalError on an invalid input or an
// underflowed window. Pure (unit tested).
std::vector<double> ComputeGroupWindows(double scalarWindow, double scale, double floor,
                                        std::vector<double> const& spectrum);

// Importance floor installed for a generation: the configured floor once the
// full-emission importance statistics have started (never during burn-in).
inline double EffectiveImportanceFloor(double configured, bool importanceAccumulated)
{
    return importanceAccumulated ? configured : 0.0;
}

// Records the escaping spectrum of a completed full-emission generation
// (raw energy per group, identical on all ranks). A generation with no
// escaping energy keeps the previous spectrum; a negative or non-finite
// entry throws. Not collective.
void CaptureWeightSplitSpectrum(std::vector<double> const& groupEnergy, long long generation,
                                WeightSplitState& state);

// Adds one generation's escaped energy (summary.ownedEscapedEnergyByCellID,
// fetched from the ranks that hold each cell's statistics) and emitted energy
// (runtime.volumeEmissionCellLuminosity and runtime.fluxSourceCellLuminosity
// times sourceDt) per local cell. Collective.
void AccumulateWeightSplitStatistics(
    Config const& cfg,
    PostprocessRuntime const& runtime,
    AdaptiveSourceUpdateSummary const& summary,
    WeightSplitState& state);

// Moves the per-cell statistics with the cells; call with the same
// tessellation state as the MPI_exchange_data of the cells. Collective.
void MigrateWeightSplitState(PostprocessRuntime const& runtime, WeightSplitState& state);

// Sets the window of the next generation on physics: w_max from
// runtime.lastEscapingLuminosity and the importance of every local cell.
// For the multigroup pass (multigroupPass) it also installs the per-group
// windows (cfg.splitPerGroup) and the provenance candidates
// (cfg.packetProvenance) from state.groupSpectrum. No-op when the window is
// off. Collective.
void InstallWeightSplit(
    Config const& cfg,
    PostprocessRuntime const& runtime,
    RadiationIMC& physics,
    WeightSplitState& state,
    bool multigroupPass);

// Rank 0 prints WEIGHT_SPLIT with the counters of the last transport and
// every rank gets the global number of copies made. Collective; returns 0
// without communicating when the window is off.
// With `totals`, every rank also adds the global refusals and per-group
// copies of the generation to it.
unsigned long long PrintWeightSplitReport(
    Config const& cfg,
    RadiationIMC const& physics,
    WeightSplitState const& state,
    std::string const& label,
    size_t generation,
    int rank,
    ForwardPostprocessResult* totals = nullptr);

} // namespace imc_postprocess_tde
#endif
