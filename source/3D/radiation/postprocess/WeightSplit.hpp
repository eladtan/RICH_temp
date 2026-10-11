#ifndef IMC_POSTPROCESS_TDE_WEIGHT_SPLIT_HPP
#define IMC_POSTPROCESS_TDE_WEIGHT_SPLIT_HPP

#include <cstddef>
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
};

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
// No-op when the window is off. Collective.
void InstallWeightSplit(
    Config const& cfg,
    PostprocessRuntime const& runtime,
    RadiationIMC& physics,
    WeightSplitState& state);

// Rank 0 prints WEIGHT_SPLIT with the counters of the last transport and
// every rank gets the global number of copies made. Collective; returns 0
// without communicating when the window is off.
unsigned long long PrintWeightSplitReport(
    Config const& cfg,
    RadiationIMC const& physics,
    WeightSplitState const& state,
    std::string const& label,
    size_t generation,
    int rank);

} // namespace imc_postprocess_tde
#endif
