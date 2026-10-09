#include "PostProcessRestart.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <unordered_map>

#ifdef RICH_MPI
#include <mpi.h>
#endif

#include <H5Cpp.h>
#include <mpi_utils/serialize/Serializer.hpp>

#include "PostProcessCommandLine.hpp"
#include "source/3D/radiation/StateHdf5IO.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "utils/hdf5/HDF5Writer.hpp"

namespace imc_postprocess_tde {
namespace {

// ------------------------------------------------------------------------
// Rank-0 work with errors reported on every rank
// ------------------------------------------------------------------------

// Runs `work` on rank 0 and makes every rank throw if it failed, so a bad
// restart file stops all ranks instead of leaving the others in a collective.
void RunOnRankZero(int rank, std::function<void()> const& work)
{
    std::string message;
    if (rank == 0) {
        try {
            work();
        } catch (UniversalError const& eo) {
            // The entries are printed here; the message travels to every rank.
            reportError(eo, std::cerr);
            message = eo.getErrorMessage();
            if (message.empty())
                message = "restart error";
        } catch (H5::Exception const& e) {
            // HDF5's C++ exceptions do not derive from std::exception.
            message = "HDF5 error in " + e.getFuncName() + ": " + e.getDetailMsg();
        } catch (std::exception const& e) {
            message = e.what();
            if (message.empty())
                message = "restart error";
        } catch (...) {
            message = "unknown exception while reading the restart file";
        }
    }
#ifdef RICH_MPI
    unsigned long long length = message.size();
    MPI_Bcast(&length, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    message.resize(static_cast<size_t>(length));
    if (length > 0)
        MPI_Bcast(&message[0], static_cast<int>(length), MPI_CHAR, 0, MPI_COMM_WORLD);
#endif
    if (!message.empty())
        throw UniversalError(message);
}

#ifdef RICH_MPI
void BroadcastSerializer(Serializer& buffer, int rank)
{
    unsigned long long size = rank == 0 ? buffer.size() : 0;
    MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        // Serializer::resize appends, so start from an empty buffer.
        buffer.reset();
        buffer.resize(static_cast<size_t>(size));
    }
    size_t offset = 0;
    while (offset < static_cast<size_t>(size)) {
        size_t const chunk = std::min<size_t>(static_cast<size_t>(size) - offset, INT_MAX / 2);
        MPI_Bcast(buffer.getData() + offset, static_cast<int>(chunk), MPI_CHAR, 0, MPI_COMM_WORLD);
        offset += chunk;
    }
}
#endif

// Sorted keys so files and the maps rebuilt from them do not depend on the
// insertion history of the run that wrote them.
template <typename Map>
std::vector<size_t> SortedKeys(Map const& map)
{
    std::vector<size_t> keys;
    keys.reserve(map.size());
    for (auto const& kv : map)
        keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());
    return keys;
}

// Learned state (everything but the load-balance measurements), in a fixed
// field order shared by the two functions below.
void SerializeLearnedState(PassRestartState const& s, Serializer& out)
{
    std::vector<size_t> const scoreKeys = SortedKeys(s.adaptive.scoreByCellID);
    std::vector<double> scoreValues;
    scoreValues.reserve(scoreKeys.size());
    for (size_t key : scoreKeys)
        scoreValues.push_back(s.adaptive.scoreByCellID.at(key));
    out.insert(scoreKeys);
    out.insert(scoreValues);
    out.insert(s.adaptive.observerDeficitByIndex);
    out.insert(s.adaptive.cumulativeObserverEnergy);
    out.insert(s.adaptive.cumulativeObserverEnergyWeightSq);
    out.insert(s.adaptive.cumulativeObserverPolarizationWeightSq);
    out.insert(s.adaptive.cumulativeObserverStokesQ);
    out.insert(s.adaptive.cumulativeObserverStokesU);
    out.insert(s.adaptive.cumulativeObserverSumWQ2);
    out.insert(s.adaptive.cumulativeObserverSumWU2);
    out.insert(s.adaptive.cumulativeObserverCrossings);
    out.insert(s.adaptive.observerBudgetMultiplier);
    out.insert(s.adaptive.adaptiveMeasuredLBCount);

    AdaptiveGroupHistory const& h = s.groupHistory;
    out.insert(static_cast<int>(h.initialized ? 1 : 0));
    out.insert(h.observerCount);
    out.insert(h.groupCount);
    out.insert(h.updateCount);
    out.insert(h.integratedUpdateCount);
    out.insert(h.emaPriority);
    out.insert(h.emaDeficit);
    out.insert(h.cumulativeEnergy);
    out.insert(h.cumulativeWeightSq);
    out.insert(h.cumulativeStokesQ);
    out.insert(h.cumulativeStokesU);
    out.insert(h.cumulativeSumWQ2);
    out.insert(h.cumulativeSumWU2);
    out.insert(h.cumulativeCrossings);

    std::vector<size_t> const groupKeys = SortedKeys(s.groupSourceState.scoreByCellGroup);
    std::vector<std::vector<double>> groupRows;
    groupRows.reserve(groupKeys.size());
    for (size_t key : groupKeys)
        groupRows.push_back(s.groupSourceState.scoreByCellGroup.at(key));
    out.insert(groupKeys);
    out.insert(groupRows);
    std::vector<size_t> const cellKeys = SortedKeys(s.groupSourceState.cellScoreFromGroups);
    std::vector<double> cellValues;
    cellValues.reserve(cellKeys.size());
    for (size_t key : cellKeys)
        cellValues.push_back(s.groupSourceState.cellScoreFromGroups.at(key));
    out.insert(cellKeys);
    out.insert(cellValues);

    IMCGroupSamplingDiagnostics d = s.finalGroupSamplingDiag;
    int const biased = d.estimatorPotentiallyBiased ? 1 : 0;
    out.insert(d.totalSampled);
    out.insert(d.cellsWithGroupScores);
    out.insert(d.weightCorrectionMin);
    out.insert(d.weightCorrectionMax);
    out.insert(d.weightCorrectionSum);
    out.insert(d.weightCorrectionCount);
    out.insert(d.weightCorrectionCapped);
    out.insert(d.weightCorrectionFallback);
    out.insert(d.invalidPdfFallback);
    out.insert(d.invalidPdfFallbackPackets);
    out.insert(d.sampledEnergy);
    out.insert(d.cappedEnergy);
    out.insert(d.cappedEnergyFraction);
    out.insert(biased);

    AdaptiveGroupSourceUpdateSummary const& g = s.finalGroupSourceSummary;
    out.insert(static_cast<int>(g.fallbackToIntegratedPath ? 1 : 0));
    out.insert(g.fallbackReason);
    out.insert(g.localStatsInput);
    out.insert(g.localStatsAfterPrune);
    out.insert(g.localStatsDropped);
    out.insert(g.maxLocalSourceGroupStats);
    out.insert(g.maxReceivedShardStats);
    out.insert(g.mpiStatsExchanged);
    out.insert(g.maxPackedBytes);
    out.insert(g.passedStats);
    out.insert(g.scoreDeltaCells);
    out.insert(g.scoreMapCells);

    out.insert(static_cast<int>(s.groupFallbackToIntegrated ? 1 : 0));
    out.insert(s.groupFallbackReason);
    out.insert(s.lastEscapingLuminosity);
    out.insert(s.completedFinalGenerations);
}

template <typename T>
struct IsStdVector : std::false_type {};
template <typename U, typename A>
struct IsStdVector<std::vector<U, A>> : std::true_type {};

void DeserializeLearnedState(Serializer const& in, PassRestartState& s)
{
    size_t at = 0;
    // Serializer stops a vector at the end of the buffer instead of failing,
    // so compare the bytes read with the byte count stored in front of it.
    auto get = [&](auto& value) {
        using T = std::decay_t<decltype(value)>;
        size_t const before = at;
        if constexpr (IsStdVector<T>::value) {
            size_t declared = 0;
            in.extract(declared, at);
            at += in.extract(value, at);
            if (at - before != sizeof(size_t) + declared)
                throw UniversalError("Restart-state broadcast is truncated");
        } else {
            at += in.extract(value, at);
        }
    };
    auto getFlag = [&]() { int flag = 0; get(flag); return flag != 0; };

    std::vector<size_t> scoreKeys;
    std::vector<double> scoreValues;
    get(scoreKeys);
    get(scoreValues);
    if (scoreValues.size() != scoreKeys.size())
        throw UniversalError("Restart-state broadcast has mismatched learned score ids and values");
    s.adaptive.scoreByCellID.clear();
    s.adaptive.scoreByCellID.reserve(scoreKeys.size());
    for (size_t i = 0; i < scoreKeys.size(); ++i)
        s.adaptive.scoreByCellID.emplace(scoreKeys[i], scoreValues[i]);
    get(s.adaptive.observerDeficitByIndex);
    get(s.adaptive.cumulativeObserverEnergy);
    get(s.adaptive.cumulativeObserverEnergyWeightSq);
    get(s.adaptive.cumulativeObserverPolarizationWeightSq);
    get(s.adaptive.cumulativeObserverStokesQ);
    get(s.adaptive.cumulativeObserverStokesU);
    get(s.adaptive.cumulativeObserverSumWQ2);
    get(s.adaptive.cumulativeObserverSumWU2);
    get(s.adaptive.cumulativeObserverCrossings);
    get(s.adaptive.observerBudgetMultiplier);
    get(s.adaptive.adaptiveMeasuredLBCount);

    AdaptiveGroupHistory& h = s.groupHistory;
    h.initialized = getFlag();
    get(h.observerCount);
    get(h.groupCount);
    get(h.updateCount);
    get(h.integratedUpdateCount);
    get(h.emaPriority);
    get(h.emaDeficit);
    get(h.cumulativeEnergy);
    get(h.cumulativeWeightSq);
    get(h.cumulativeStokesQ);
    get(h.cumulativeStokesU);
    get(h.cumulativeSumWQ2);
    get(h.cumulativeSumWU2);
    get(h.cumulativeCrossings);

    std::vector<size_t> groupKeys;
    std::vector<std::vector<double>> groupRows;
    get(groupKeys);
    get(groupRows);
    if (groupRows.size() != groupKeys.size())
        throw UniversalError("Restart-state broadcast has mismatched group score ids and rows");
    s.groupSourceState.scoreByCellGroup.clear();
    s.groupSourceState.scoreByCellGroup.reserve(groupKeys.size());
    for (size_t i = 0; i < groupKeys.size(); ++i)
        s.groupSourceState.scoreByCellGroup.emplace(groupKeys[i], std::move(groupRows[i]));
    std::vector<size_t> cellKeys;
    std::vector<double> cellValues;
    get(cellKeys);
    get(cellValues);
    if (cellValues.size() != cellKeys.size())
        throw UniversalError("Restart-state broadcast has mismatched group cell ids and scores");
    s.groupSourceState.cellScoreFromGroups.clear();
    s.groupSourceState.cellScoreFromGroups.reserve(cellKeys.size());
    for (size_t i = 0; i < cellKeys.size(); ++i)
        s.groupSourceState.cellScoreFromGroups.emplace(cellKeys[i], cellValues[i]);

    IMCGroupSamplingDiagnostics& d = s.finalGroupSamplingDiag;
    get(d.totalSampled);
    get(d.cellsWithGroupScores);
    get(d.weightCorrectionMin);
    get(d.weightCorrectionMax);
    get(d.weightCorrectionSum);
    get(d.weightCorrectionCount);
    get(d.weightCorrectionCapped);
    get(d.weightCorrectionFallback);
    get(d.invalidPdfFallback);
    get(d.invalidPdfFallbackPackets);
    get(d.sampledEnergy);
    get(d.cappedEnergy);
    get(d.cappedEnergyFraction);
    d.estimatorPotentiallyBiased = getFlag();

    AdaptiveGroupSourceUpdateSummary& g = s.finalGroupSourceSummary;
    g.fallbackToIntegratedPath = getFlag();
    get(g.fallbackReason);
    get(g.localStatsInput);
    get(g.localStatsAfterPrune);
    get(g.localStatsDropped);
    get(g.maxLocalSourceGroupStats);
    get(g.maxReceivedShardStats);
    get(g.mpiStatsExchanged);
    get(g.maxPackedBytes);
    get(g.passedStats);
    get(g.scoreDeltaCells);
    get(g.scoreMapCells);

    s.groupFallbackToIntegrated = getFlag();
    get(s.groupFallbackReason);
    get(s.lastEscapingLuminosity);
    get(s.completedFinalGenerations);
    if (at != in.size())
        throw UniversalError("Restart-state broadcast has trailing bytes");
}

// ------------------------------------------------------------------------
// HDF5 layout of the learned state
// ------------------------------------------------------------------------

void WriteLearnedState(HDF5Writer& w, std::string const& p, PassRestartState const& s)
{
    using state_hdf5::WriteMatrix;
    using state_hdf5::WriteVector;

    std::vector<size_t> const scoreKeys = SortedKeys(s.adaptive.scoreByCellID);
    std::vector<double> scoreValues;
    scoreValues.reserve(scoreKeys.size());
    for (size_t key : scoreKeys)
        scoreValues.push_back(s.adaptive.scoreByCellID.at(key));
    WriteVector(w, p + "/adaptive/score_cell_ids", scoreKeys);
    WriteVector(w, p + "/adaptive/score_values", scoreValues);
    WriteVector(w, p + "/adaptive/observer_deficit", s.adaptive.observerDeficitByIndex);
    WriteVector(w, p + "/adaptive/cumulative_observer_energy", s.adaptive.cumulativeObserverEnergy);
    WriteVector(w, p + "/adaptive/cumulative_observer_energy_weight_sq", s.adaptive.cumulativeObserverEnergyWeightSq);
    WriteVector(w, p + "/adaptive/cumulative_observer_polarization_weight_sq", s.adaptive.cumulativeObserverPolarizationWeightSq);
    WriteVector(w, p + "/adaptive/cumulative_observer_stokes_q", s.adaptive.cumulativeObserverStokesQ);
    WriteVector(w, p + "/adaptive/cumulative_observer_stokes_u", s.adaptive.cumulativeObserverStokesU);
    WriteVector(w, p + "/adaptive/cumulative_observer_sum_wq2", s.adaptive.cumulativeObserverSumWQ2);
    WriteVector(w, p + "/adaptive/cumulative_observer_sum_wu2", s.adaptive.cumulativeObserverSumWU2);
    WriteVector(w, p + "/adaptive/cumulative_observer_crossings", s.adaptive.cumulativeObserverCrossings);
    w.WriteElement(p + "/adaptive/observer_budget_multiplier", s.adaptive.observerBudgetMultiplier);
    w.WriteElement(p + "/adaptive/measured_lb_count", s.adaptive.adaptiveMeasuredLBCount);

    AdaptiveGroupHistory const& h = s.groupHistory;
    w.WriteElement(p + "/group_history/initialized", h.initialized ? 1 : 0);
    w.WriteElement(p + "/group_history/observer_count", h.observerCount);
    w.WriteElement(p + "/group_history/group_count", h.groupCount);
    w.WriteElement(p + "/group_history/update_count", h.updateCount);
    w.WriteElement(p + "/group_history/integrated_update_count", h.integratedUpdateCount);
    WriteMatrix(w, p + "/group_history/ema_priority", h.emaPriority);
    WriteMatrix(w, p + "/group_history/ema_deficit", h.emaDeficit);
    WriteMatrix(w, p + "/group_history/cumulative_energy", h.cumulativeEnergy);
    WriteMatrix(w, p + "/group_history/cumulative_weight_sq", h.cumulativeWeightSq);
    WriteMatrix(w, p + "/group_history/cumulative_stokes_q", h.cumulativeStokesQ);
    WriteMatrix(w, p + "/group_history/cumulative_stokes_u", h.cumulativeStokesU);
    WriteMatrix(w, p + "/group_history/cumulative_sum_wq2", h.cumulativeSumWQ2);
    WriteMatrix(w, p + "/group_history/cumulative_sum_wu2", h.cumulativeSumWU2);
    WriteMatrix(w, p + "/group_history/cumulative_crossings", h.cumulativeCrossings);

    std::vector<size_t> const groupKeys = SortedKeys(s.groupSourceState.scoreByCellGroup);
    std::vector<size_t> groupLengths;
    std::vector<double> groupValues;
    groupLengths.reserve(groupKeys.size());
    for (size_t key : groupKeys) {
        std::vector<double> const& row = s.groupSourceState.scoreByCellGroup.at(key);
        groupLengths.push_back(row.size());
        groupValues.insert(groupValues.end(), row.begin(), row.end());
    }
    WriteVector(w, p + "/group_source/score_cell_ids", groupKeys);
    WriteVector(w, p + "/group_source/score_lengths", groupLengths);
    WriteVector(w, p + "/group_source/score_values", groupValues);
    std::vector<size_t> const cellKeys = SortedKeys(s.groupSourceState.cellScoreFromGroups);
    std::vector<double> cellValues;
    cellValues.reserve(cellKeys.size());
    for (size_t key : cellKeys)
        cellValues.push_back(s.groupSourceState.cellScoreFromGroups.at(key));
    WriteVector(w, p + "/group_source/cell_score_ids", cellKeys);
    WriteVector(w, p + "/group_source/cell_score_values", cellValues);

    IMCGroupSamplingDiagnostics const& d = s.finalGroupSamplingDiag;
    std::string const ds = p + "/group_sampling_diagnostics";
    w.WriteElement(ds + "/total_sampled", d.totalSampled);
    w.WriteElement(ds + "/cells_with_group_scores", d.cellsWithGroupScores);
    w.WriteElement(ds + "/weight_correction_min", d.weightCorrectionMin);
    w.WriteElement(ds + "/weight_correction_max", d.weightCorrectionMax);
    w.WriteElement(ds + "/weight_correction_sum", d.weightCorrectionSum);
    w.WriteElement(ds + "/weight_correction_count", d.weightCorrectionCount);
    w.WriteElement(ds + "/weight_correction_capped", d.weightCorrectionCapped);
    w.WriteElement(ds + "/weight_correction_fallback", d.weightCorrectionFallback);
    w.WriteElement(ds + "/invalid_pdf_fallback", d.invalidPdfFallback);
    w.WriteElement(ds + "/invalid_pdf_fallback_packets", d.invalidPdfFallbackPackets);
    w.WriteElement(ds + "/sampled_energy", d.sampledEnergy);
    w.WriteElement(ds + "/capped_energy", d.cappedEnergy);
    w.WriteElement(ds + "/capped_energy_fraction", d.cappedEnergyFraction);
    w.WriteElement(ds + "/estimator_potentially_biased", d.estimatorPotentiallyBiased ? 1 : 0);

    AdaptiveGroupSourceUpdateSummary const& g = s.finalGroupSourceSummary;
    std::string const gs = p + "/group_source_summary";
    w.WriteElement(gs + "/fallback_to_integrated_path", g.fallbackToIntegratedPath ? 1 : 0);
    w.WriteElement(gs + "/fallback_reason", g.fallbackReason);
    w.WriteElement(gs + "/local_stats_input", g.localStatsInput);
    w.WriteElement(gs + "/local_stats_after_prune", g.localStatsAfterPrune);
    w.WriteElement(gs + "/local_stats_dropped", g.localStatsDropped);
    w.WriteElement(gs + "/max_local_source_group_stats", g.maxLocalSourceGroupStats);
    w.WriteElement(gs + "/max_received_shard_stats", g.maxReceivedShardStats);
    w.WriteElement(gs + "/mpi_stats_exchanged", g.mpiStatsExchanged);
    w.WriteElement(gs + "/max_packed_bytes", g.maxPackedBytes);
    w.WriteElement(gs + "/passed_stats", g.passedStats);
    w.WriteElement(gs + "/score_delta_cells", g.scoreDeltaCells);
    w.WriteElement(gs + "/score_map_cells", g.scoreMapCells);

    w.WriteElement(p + "/group_fallback_to_integrated", s.groupFallbackToIntegrated ? 1 : 0);
    w.WriteElement(p + "/group_fallback_reason", s.groupFallbackReason);
    w.WriteElement(p + "/last_escaping_luminosity", s.lastEscapingLuminosity);
    w.WriteElement(p + "/completed_final_generations", s.completedFinalGenerations);
}

void ReadLearnedState(HDF5Reader const& r, std::string const& p, PassRestartState& s)
{
    using state_hdf5::ReadMatrix;
    using state_hdf5::ReadScalar;
    using state_hdf5::ReadVector;

    std::vector<size_t> const scoreKeys = ReadVector<size_t>(r, p + "/adaptive/score_cell_ids");
    std::vector<double> const scoreValues =
        ReadVector<double>(r, p + "/adaptive/score_values", scoreKeys.size());
    if (scoreValues.size() != scoreKeys.size())
        throw UniversalError("Restart learned scores have mismatched ids and values at " + p);
    s.adaptive.scoreByCellID.clear();
    s.adaptive.scoreByCellID.reserve(scoreKeys.size());
    for (size_t i = 0; i < scoreKeys.size(); ++i)
        s.adaptive.scoreByCellID.emplace(scoreKeys[i], scoreValues[i]);
    s.adaptive.observerDeficitByIndex = ReadVector<double>(r, p + "/adaptive/observer_deficit");
    s.adaptive.cumulativeObserverEnergy = ReadVector<double>(r, p + "/adaptive/cumulative_observer_energy");
    s.adaptive.cumulativeObserverEnergyWeightSq = ReadVector<double>(r, p + "/adaptive/cumulative_observer_energy_weight_sq");
    s.adaptive.cumulativeObserverPolarizationWeightSq = ReadVector<double>(r, p + "/adaptive/cumulative_observer_polarization_weight_sq");
    s.adaptive.cumulativeObserverStokesQ = ReadVector<double>(r, p + "/adaptive/cumulative_observer_stokes_q");
    s.adaptive.cumulativeObserverStokesU = ReadVector<double>(r, p + "/adaptive/cumulative_observer_stokes_u");
    s.adaptive.cumulativeObserverSumWQ2 = ReadVector<double>(r, p + "/adaptive/cumulative_observer_sum_wq2");
    s.adaptive.cumulativeObserverSumWU2 = ReadVector<double>(r, p + "/adaptive/cumulative_observer_sum_wu2");
    s.adaptive.cumulativeObserverCrossings = ReadVector<unsigned long long>(r, p + "/adaptive/cumulative_observer_crossings");
    s.adaptive.observerBudgetMultiplier = ReadScalar<double>(r, p + "/adaptive/observer_budget_multiplier");
    s.adaptive.adaptiveMeasuredLBCount = ReadScalar<size_t>(r, p + "/adaptive/measured_lb_count");

    AdaptiveGroupHistory& h = s.groupHistory;
    h.initialized = ReadScalar<int>(r, p + "/group_history/initialized") != 0;
    h.observerCount = ReadScalar<size_t>(r, p + "/group_history/observer_count");
    h.groupCount = ReadScalar<size_t>(r, p + "/group_history/group_count");
    h.updateCount = ReadScalar<size_t>(r, p + "/group_history/update_count");
    h.integratedUpdateCount = ReadScalar<size_t>(r, p + "/group_history/integrated_update_count");
    h.emaPriority = ReadMatrix<double>(r, p + "/group_history/ema_priority");
    h.emaDeficit = ReadMatrix<double>(r, p + "/group_history/ema_deficit");
    h.cumulativeEnergy = ReadMatrix<double>(r, p + "/group_history/cumulative_energy");
    h.cumulativeWeightSq = ReadMatrix<double>(r, p + "/group_history/cumulative_weight_sq");
    h.cumulativeStokesQ = ReadMatrix<double>(r, p + "/group_history/cumulative_stokes_q");
    h.cumulativeStokesU = ReadMatrix<double>(r, p + "/group_history/cumulative_stokes_u");
    h.cumulativeSumWQ2 = ReadMatrix<double>(r, p + "/group_history/cumulative_sum_wq2");
    h.cumulativeSumWU2 = ReadMatrix<double>(r, p + "/group_history/cumulative_sum_wu2");
    h.cumulativeCrossings = ReadMatrix<size_t>(r, p + "/group_history/cumulative_crossings");

    std::vector<size_t> const groupKeys = ReadVector<size_t>(r, p + "/group_source/score_cell_ids");
    std::vector<size_t> const groupLengths =
        ReadVector<size_t>(r, p + "/group_source/score_lengths", groupKeys.size());
    std::vector<double> const groupValues = ReadVector<double>(r, p + "/group_source/score_values");
    if (groupLengths.size() != groupKeys.size())
        throw UniversalError("Restart group scores have mismatched ids and lengths at " + p);
    s.groupSourceState.scoreByCellGroup.clear();
    s.groupSourceState.scoreByCellGroup.reserve(groupKeys.size());
    size_t offset = 0;
    for (size_t i = 0; i < groupKeys.size(); ++i) {
        if (offset + groupLengths[i] > groupValues.size())
            throw UniversalError("Restart group scores are truncated at " + p);
        s.groupSourceState.scoreByCellGroup.emplace(
            groupKeys[i],
            std::vector<double>(groupValues.begin() + static_cast<std::ptrdiff_t>(offset),
                                groupValues.begin() + static_cast<std::ptrdiff_t>(offset + groupLengths[i])));
        offset += groupLengths[i];
    }
    if (offset != groupValues.size())
        throw UniversalError("Restart group scores have trailing values at " + p);
    std::vector<size_t> const cellKeys = ReadVector<size_t>(r, p + "/group_source/cell_score_ids");
    std::vector<double> const cellValues =
        ReadVector<double>(r, p + "/group_source/cell_score_values", cellKeys.size());
    if (cellValues.size() != cellKeys.size())
        throw UniversalError("Restart group cell scores have mismatched ids and values at " + p);
    s.groupSourceState.cellScoreFromGroups.clear();
    s.groupSourceState.cellScoreFromGroups.reserve(cellKeys.size());
    for (size_t i = 0; i < cellKeys.size(); ++i)
        s.groupSourceState.cellScoreFromGroups.emplace(cellKeys[i], cellValues[i]);

    IMCGroupSamplingDiagnostics& d = s.finalGroupSamplingDiag;
    std::string const ds = p + "/group_sampling_diagnostics";
    d.totalSampled = ReadScalar<size_t>(r, ds + "/total_sampled");
    d.cellsWithGroupScores = ReadScalar<size_t>(r, ds + "/cells_with_group_scores");
    d.weightCorrectionMin = ReadScalar<double>(r, ds + "/weight_correction_min");
    d.weightCorrectionMax = ReadScalar<double>(r, ds + "/weight_correction_max");
    d.weightCorrectionSum = ReadScalar<double>(r, ds + "/weight_correction_sum");
    d.weightCorrectionCount = ReadScalar<size_t>(r, ds + "/weight_correction_count");
    d.weightCorrectionCapped = ReadScalar<size_t>(r, ds + "/weight_correction_capped");
    d.weightCorrectionFallback = ReadScalar<size_t>(r, ds + "/weight_correction_fallback");
    d.invalidPdfFallback = ReadScalar<size_t>(r, ds + "/invalid_pdf_fallback");
    d.invalidPdfFallbackPackets = ReadScalar<size_t>(r, ds + "/invalid_pdf_fallback_packets");
    d.sampledEnergy = ReadScalar<double>(r, ds + "/sampled_energy");
    d.cappedEnergy = ReadScalar<double>(r, ds + "/capped_energy");
    d.cappedEnergyFraction = ReadScalar<double>(r, ds + "/capped_energy_fraction");
    d.estimatorPotentiallyBiased = ReadScalar<int>(r, ds + "/estimator_potentially_biased") != 0;

    AdaptiveGroupSourceUpdateSummary& g = s.finalGroupSourceSummary;
    std::string const gs = p + "/group_source_summary";
    g.fallbackToIntegratedPath = ReadScalar<int>(r, gs + "/fallback_to_integrated_path") != 0;
    g.fallbackReason = ReadScalar<std::string>(r, gs + "/fallback_reason");
    g.localStatsInput = ReadScalar<unsigned long long>(r, gs + "/local_stats_input");
    g.localStatsAfterPrune = ReadScalar<unsigned long long>(r, gs + "/local_stats_after_prune");
    g.localStatsDropped = ReadScalar<unsigned long long>(r, gs + "/local_stats_dropped");
    g.maxLocalSourceGroupStats = ReadScalar<unsigned long long>(r, gs + "/max_local_source_group_stats");
    g.maxReceivedShardStats = ReadScalar<unsigned long long>(r, gs + "/max_received_shard_stats");
    g.mpiStatsExchanged = ReadScalar<unsigned long long>(r, gs + "/mpi_stats_exchanged");
    g.maxPackedBytes = ReadScalar<unsigned long long>(r, gs + "/max_packed_bytes");
    g.passedStats = ReadScalar<size_t>(r, gs + "/passed_stats");
    g.scoreDeltaCells = ReadScalar<size_t>(r, gs + "/score_delta_cells");
    g.scoreMapCells = ReadScalar<size_t>(r, gs + "/score_map_cells");

    s.groupFallbackToIntegrated = ReadScalar<int>(r, p + "/group_fallback_to_integrated") != 0;
    s.groupFallbackReason = ReadScalar<std::string>(r, p + "/group_fallback_reason");
    s.lastEscapingLuminosity = ReadScalar<double>(r, p + "/last_escaping_luminosity");
    s.completedFinalGenerations = ReadScalar<size_t>(r, p + "/completed_final_generations");
}

// Shape checks of the learned state against the run, so a damaged or
// mismatched file fails here instead of indexing past an array later.
void ValidateLearnedState(PassRestartState const& s, size_t observers, size_t groups,
                          std::string const& group)
{
    auto fail = [&](std::string const& what, size_t stored, size_t expected) {
        UniversalError eo("Restart learned state of " + group + " does not match this run: " + what);
        eo.addEntry("stored", static_cast<double>(stored));
        eo.addEntry("expected", static_cast<double>(expected));
        throw eo;
    };
    auto checkObserverVector = [&](auto const& values, char const* name) {
        if (!values.empty() && values.size() != observers)
            fail(std::string(name) + " length", values.size(), observers);
    };
    AdaptiveSourceState const& a = s.adaptive;
    checkObserverVector(a.observerDeficitByIndex, "observer_deficit");
    checkObserverVector(a.cumulativeObserverEnergy, "cumulative_observer_energy");
    checkObserverVector(a.cumulativeObserverEnergyWeightSq, "cumulative_observer_energy_weight_sq");
    checkObserverVector(a.cumulativeObserverPolarizationWeightSq, "cumulative_observer_polarization_weight_sq");
    checkObserverVector(a.cumulativeObserverStokesQ, "cumulative_observer_stokes_q");
    checkObserverVector(a.cumulativeObserverStokesU, "cumulative_observer_stokes_u");
    checkObserverVector(a.cumulativeObserverSumWQ2, "cumulative_observer_sum_wq2");
    checkObserverVector(a.cumulativeObserverSumWU2, "cumulative_observer_sum_wu2");
    checkObserverVector(a.cumulativeObserverCrossings, "cumulative_observer_crossings");

    AdaptiveGroupHistory const& h = s.groupHistory;
    auto checkHistoryMatrix = [&](auto const& matrix, char const* name) {
        if (!h.initialized) {
            if (!matrix.empty())
                fail(std::string("group_history/") + name + " rows (history not initialized)", matrix.size(), 0);
            return;
        }
        if (matrix.size() != h.observerCount)
            fail(std::string("group_history/") + name + " rows", matrix.size(), h.observerCount);
        for (auto const& row : matrix)
            if (row.size() != h.groupCount)
                fail(std::string("group_history/") + name + " columns", row.size(), h.groupCount);
    };
    if (h.initialized) {
        if (h.observerCount != observers)
            fail("group_history/observer_count", h.observerCount, observers);
        if (h.groupCount != groups)
            fail("group_history/group_count", h.groupCount, groups);
    }
    checkHistoryMatrix(h.emaPriority, "ema_priority");
    checkHistoryMatrix(h.emaDeficit, "ema_deficit");
    checkHistoryMatrix(h.cumulativeEnergy, "cumulative_energy");
    checkHistoryMatrix(h.cumulativeWeightSq, "cumulative_weight_sq");
    checkHistoryMatrix(h.cumulativeStokesQ, "cumulative_stokes_q");
    checkHistoryMatrix(h.cumulativeStokesU, "cumulative_stokes_u");
    checkHistoryMatrix(h.cumulativeSumWQ2, "cumulative_sum_wq2");
    checkHistoryMatrix(h.cumulativeSumWU2, "cumulative_sum_wu2");
    checkHistoryMatrix(h.cumulativeCrossings, "cumulative_crossings");
    for (auto const& kv : s.groupSourceState.scoreByCellGroup)
        if (kv.second.size() != groups)
            fail("group_source/score_values per cell", kv.second.size(), groups);
}

// Settings that change what a final generation estimates. Everything else
// (generation count, packet budgets, exploration fractions, adaptive targets,
// load balancing, communication, output) only changes how the packets are
// allocated, which leaves every generation's estimate unbiased.
std::vector<std::string> const& MustMatchOptions()
{
    static std::vector<std::string> const names = {
        "input.snapshot",
        "input.multigroup-opacity-directory",
        "input.grey-opacity-directory",
        "input.eos-directory",
        "observer.center-x",
        "observer.center-y",
        "observer.center-z",
        "observer.radius",
        "observer.count",
        "transport.source-dt",
        "transport.duration",
        "transport.ddmc",
        "transport.random-walk",
        "transport.ddmc-min-cell-optical-depth",
        "transport.ddmc-moving-interface-correction",
        "transport.ddmc-advection",
        "transport.ddmc-max-cell-advection-peclet",
        "transport.use-cell-velocities",
        "transport.compton.enabled",
        "transport.compton.matrix-samples",
        "transport.compton.angle-dependent",
        "polarization.enabled",
        "polarization.manual-scatterings-after-acceleration",
        "polarization.depolarization-scatterings",
        "polarization.accelerated-closure",
        "flux-source.enabled",
        "flux-source.thermalization-tau",
        "flux-source.construction-rays",
        "flux-source.ddmc-face-optical-depth",
        "flux-source.surface-mode",
        "flux-source.remove-enclosed-outside",
        "flux-source.smoothing-floor",
        "volume-emission.enabled",
        "volume-emission.cutoff-fraction",
        "volume-emission.gate-groups",
        "opacity-scaling.mode",
        "opacity-scaling.alpha-min",
        "opacity-scaling.alpha-max",
        // The learned state's structure depends on these.
        "adaptive.source.enabled",
        "adaptive.group.quality",
        "adaptive.group.source-cells",
        "adaptive.group.frequency-sampling",
    };
    return names;
}

std::string ConfigDatasetPath(std::string name)
{
    for (char& c : name)
        if (c == '.')
            c = '/';
    return "/config/" + name;
}

bool RelativelyEqual(double a, double b, double tolerance)
{
    double const scale = std::max(std::abs(a), std::abs(b));
    return std::abs(a - b) <= tolerance * scale;
}

uint64_t SplitMix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

} // namespace

std::string RestartPassGroup(RestartPass pass)
{
    return pass == RestartPass::Forward ? "/passes/forward" : "/passes/grey";
}

void CheckRestartCompatibility(
    std::string const& path,
    PostProcessIMC::PostProcessConfig const& effectiveConfig,
    double snapshotTime,
    int snapshotCycle,
    int rank)
{
    RunOnRankZero(rank, [&]() {
        HDF5Reader reader(path);
        if (!reader.Exists("/restart/schema_version"))
            throw UniversalError(
                "Restart file " + path + " has no restart state (written before restarts were supported, "
                "or not a post-process result); it cannot be continued");
        int const schema = state_hdf5::ReadScalar<int>(reader, "/restart/schema_version");
        if (schema != kRestartSchemaVersion) {
            UniversalError eo("Restart file " + path + " has an unsupported restart schema");
            eo.addEntry("schema", schema);
            eo.addEntry("supported", kRestartSchemaVersion);
            throw eo;
        }

        std::unordered_map<std::string, std::string> current;
        for (auto const& entry : PostProcessIMC::EffectiveConfigEntries(effectiveConfig))
            current.emplace(entry.first, entry.second);

        std::ostringstream differences;
        size_t count = 0;
        for (std::string const& name : MustMatchOptions()) {
            auto const it = current.find(name);
            if (it == current.end())
                throw UniversalError("Restart compatibility list names unknown option " + name);
            std::string const dataset = ConfigDatasetPath(name);
            if (!reader.Exists(dataset)) {
                differences << "\n  " << name << ": missing from the restart file";
                ++count;
                continue;
            }
            std::string stored;
            reader.ReadElement(dataset, stored);
            if (stored != it->second) {
                differences << "\n  " << name << ": file '" << stored
                            << "' vs run '" << it->second << "'";
                ++count;
            }
        }

        double const storedTime = state_hdf5::ReadScalar<double>(reader, "/metadata/snapshot_time");
        int const storedCycle = state_hdf5::ReadScalar<int>(reader, "/metadata/snapshot_cycle");
        if (!RelativelyEqual(storedTime, snapshotTime, 1e-12)) {
            differences << "\n  snapshot time: file " << storedTime << " vs run " << snapshotTime;
            ++count;
        }
        if (storedCycle != snapshotCycle) {
            differences << "\n  snapshot cycle: file " << storedCycle << " vs run " << snapshotCycle;
            ++count;
        }
#ifdef MONTECARLO_POLARIZATION
        int const buildPolarization = 1;
#else
        int const buildPolarization = 0;
#endif
#ifdef RICH_IMC_DDMC_ENABLED
        int const buildDdmc = 1;
#else
        int const buildDdmc = 0;
#endif
        int const storedPolarization = state_hdf5::ReadScalar<int>(reader, "/metadata/build/polarization");
        int const storedDdmc = state_hdf5::ReadScalar<int>(reader, "/metadata/build/ddmc");
        int const storedGroups = state_hdf5::ReadScalar<int>(reader, "/metadata/build/energy_groups");
        if (storedPolarization != buildPolarization) {
            differences << "\n  build MONTECARLO_POLARIZATION: file " << storedPolarization
                        << " vs run " << buildPolarization;
            ++count;
        }
        if (storedDdmc != buildDdmc) {
            differences << "\n  build DDMC: file " << storedDdmc << " vs run " << buildDdmc;
            ++count;
        }
        if (storedGroups != ENERGY_GROUPS_NUM) {
            differences << "\n  build ENERGY_GROUPS_NUM: file " << storedGroups
                        << " vs run " << ENERGY_GROUPS_NUM;
            ++count;
        }

        if (count > 0)
            throw UniversalError(
                "Cannot continue " + path + ": " + std::to_string(count) +
                " setting(s) that change the estimate differ from the file:" + differences.str());
    });
}

void CheckRestartSourceSurface(
    std::string const& path, PostprocessRuntime const& runtime, int rank)
{
    // Recomputed from the snapshot, so they agree up to the summation order
    // of the domain decomposition. A changed snapshot, opacity table or
    // surface setting moves them by far more.
    double const tolerance = 1e-6;
    RunOnRankZero(rank, [&]() {
        HDF5Reader reader(path);
        double const storedInjected = state_hdf5::ReadScalar<double>(
            reader, "/comparison/source_surface/injected_luminosity");
        double const storedFld = state_hdf5::ReadScalar<double>(
            reader, "/comparison/fld_total_luminosity");
        std::ostringstream differences;
        if (!RelativelyEqual(storedInjected, runtime.fluxSourceInjectedLuminosity, tolerance))
            differences << "\n  source-surface injected luminosity: file " << storedInjected
                        << " vs run " << runtime.fluxSourceInjectedLuminosity << " erg/s";
        if (!RelativelyEqual(storedFld, runtime.totalFldLuminosity, tolerance))
            differences << "\n  FLD total luminosity: file " << storedFld
                        << " vs run " << runtime.totalFldLuminosity << " erg/s";
        if (!differences.str().empty())
            throw UniversalError(
                "Cannot continue " + path + ": the recomputed source differs from the file "
                "(relative tolerance 1e-6):" + differences.str());
    });
}

namespace {

PassRestartState LoadPassRestartState(
    std::string const& path, RestartPass pass,
    size_t observerCount, size_t groupCount, int rank)
{
    PassRestartState state;
    Serializer bytes;
    std::string const group = RestartPassGroup(pass);
    RunOnRankZero(rank, [&]() {
        HDF5Reader reader(path);
        if (!reader.Exists(group + "/learned_state/completed_final_generations"))
            throw UniversalError("Restart file " + path + " has no learned state for " + group);
        if (!reader.Exists(group + "/load_balance/count"))
            throw UniversalError("Restart file " + path + " has no load-balance measurements for " + group);
        ReadLearnedState(reader, group + "/learned_state", state);
        size_t const count = state_hdf5::ReadScalar<size_t>(reader, group + "/load_balance/count");
        state.loadBalance.cellIDs = state_hdf5::ReadVector<size_t>(reader, group + "/load_balance/cell_ids", count);
        state.loadBalance.steps = state_hdf5::ReadVector<size_t>(reader, group + "/load_balance/steps", count);
        state.loadBalance.particles = state_hdf5::ReadVector<size_t>(reader, group + "/load_balance/particles", count);
        if (state.loadBalance.cellIDs.size() != count ||
            state.loadBalance.steps.size() != count ||
            state.loadBalance.particles.size() != count)
            throw UniversalError("Restart load-balance measurements are truncated for " + group);
        ValidateLearnedState(state, observerCount, groupCount, group);
        SerializeLearnedState(state, bytes);
    });
#ifdef RICH_MPI
    BroadcastSerializer(bytes, rank);
#endif
    // Every rank, rank 0 included, rebuilds the maps from the same sorted
    // sequence, so their iteration order is identical across ranks.
    LoadBalanceMeasurements loadBalance = std::move(state.loadBalance);
    state = PassRestartState{};
    DeserializeLearnedState(bytes, state);
    if (rank == 0)
        state.loadBalance = std::move(loadBalance);
    state.adaptive.burninCompletePrinted = true;
    state.adaptive.postAdaptiveMeasuredLBDone = true;
    return state;
}

void LoadObserverStatistics(
    std::string const& path, RestartPass pass, SphericalObserver& observer,
    size_t expectedSamples, int rank)
{
    RunOnRankZero(rank, [&]() {
        HDF5Reader reader(path);
        observer.readStatisticsState(reader, RestartPassGroup(pass) + "/statistics_state");
        if (observer.getStatisticsSamples() != expectedSamples) {
            UniversalError eo("Restart statistics and learned state of " + RestartPassGroup(pass) +
                              " disagree on the number of final generations");
            eo.addEntry("statistics samples", static_cast<double>(observer.getStatisticsSamples()));
            eo.addEntry("completed final generations", static_cast<double>(expectedSamples));
            throw eo;
        }
    });
}

} // namespace

void WritePassRestartState(
    std::string const& filename,
    PassRestartState const& state,
    SphericalObserver const& observer)
{
    HDF5Writer writer(filename, false);
    observer.writeStatisticsState(writer, "/statistics_state");
    WriteLearnedState(writer, "/learned_state", state);
    writer.WriteElement("/load_balance/count", state.loadBalance.cellIDs.size());
    state_hdf5::WriteVector(writer, "/load_balance/cell_ids", state.loadBalance.cellIDs);
    state_hdf5::WriteVector(writer, "/load_balance/steps", state.loadBalance.steps);
    state_hdf5::WriteVector(writer, "/load_balance/particles", state.loadBalance.particles);
    writer.Close();
}

LoadBalanceMeasurements CollectLocalLoadBalanceMeasurements(
    std::vector<ComputationalCell3D> const& cells,
    size_t nCells,
    std::vector<size_t> const& steps,
    std::vector<size_t> const& particles)
{
    LoadBalanceMeasurements local;
    for (size_t i = 0; i < nCells; ++i) {
        size_t const s = i < steps.size() ? steps[i] : 0;
        size_t const p = i < particles.size() ? particles[i] : 0;
        if (s == 0 && p == 0)
            continue;
        local.cellIDs.push_back(cells[i].ID);
        local.steps.push_back(s);
        local.particles.push_back(p);
    }
    return local;
}

namespace {

LoadBalanceMeasurements GatherLoadBalanceMeasurements(
    LoadBalanceMeasurements const& local, int rank)
{
#ifdef RICH_MPI
    int mpiSize = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    unsigned long long const localCount = local.cellIDs.size();
    std::vector<unsigned long long> counts(rank == 0 ? static_cast<size_t>(mpiSize) : 0);
    MPI_Gather(&localCount, 1, MPI_UNSIGNED_LONG_LONG,
               counts.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    std::vector<int> recvCounts, displacements;
    unsigned long long total = 0;
    if (rank == 0) {
        recvCounts.resize(counts.size());
        displacements.resize(counts.size());
        for (size_t r = 0; r < counts.size(); ++r) {
            if (total > static_cast<unsigned long long>(INT_MAX) ||
                counts[r] > static_cast<unsigned long long>(INT_MAX) - total)
                throw UniversalError("Too many load-balance measurements to gather for the restart state");
            displacements[r] = static_cast<int>(total);
            recvCounts[r] = static_cast<int>(counts[r]);
            total += counts[r];
        }
    }
    LoadBalanceMeasurements gathered;
    auto gatherField = [&](std::vector<size_t> const& field, std::vector<size_t>& out) {
        std::vector<unsigned long long> send(field.begin(), field.end());
        std::vector<unsigned long long> receive(rank == 0 ? static_cast<size_t>(total) : 0);
        MPI_Gatherv(send.data(), static_cast<int>(send.size()), MPI_UNSIGNED_LONG_LONG,
                    receive.data(), recvCounts.data(), displacements.data(),
                    MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        out.assign(receive.begin(), receive.end());
    };
    gatherField(local.cellIDs, gathered.cellIDs);
    gatherField(local.steps, gathered.steps);
    gatherField(local.particles, gathered.particles);
    return gathered;
#else
    (void)rank;
    return local;
#endif
}

void DistributeLoadBalanceMeasurements(
    LoadBalanceMeasurements const& stored,
    std::vector<ComputationalCell3D> const& cells,
    size_t nCells,
    std::vector<size_t>& steps,
    std::vector<size_t>& particles,
    int rank)
{
    steps.assign(nCells, 0);
    particles.assign(nCells, 0);
    std::unordered_map<size_t, size_t> localIndexByID;
    localIndexByID.reserve(nCells);
    for (size_t i = 0; i < nCells; ++i)
        localIndexByID.emplace(cells[i].ID, i);

    unsigned long long localMatched = 0;
    auto apply = [&](unsigned long long const* triplets, size_t count) {
        for (size_t k = 0; k < count; ++k) {
            auto const it = localIndexByID.find(static_cast<size_t>(triplets[3 * k]));
            if (it == localIndexByID.end())
                continue;
            steps[it->second] = static_cast<size_t>(triplets[3 * k + 1]);
            particles[it->second] = static_cast<size_t>(triplets[3 * k + 2]);
            ++localMatched;
        }
    };

    unsigned long long total = rank == 0 ? stored.cellIDs.size() : 0;
#ifdef RICH_MPI
    MPI_Bcast(&total, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
#endif
    // Chunks bound the memory every rank holds while scanning the whole set.
    size_t const chunk = size_t(1) << 20;
    std::vector<unsigned long long> buffer;
    for (size_t begin = 0; begin < total; begin += chunk) {
        size_t const count = std::min<size_t>(chunk, static_cast<size_t>(total) - begin);
        buffer.assign(3 * count, 0);
        if (rank == 0) {
            for (size_t k = 0; k < count; ++k) {
                buffer[3 * k] = stored.cellIDs[begin + k];
                buffer[3 * k + 1] = stored.steps[begin + k];
                buffer[3 * k + 2] = stored.particles[begin + k];
            }
        }
#ifdef RICH_MPI
        MPI_Bcast(buffer.data(), static_cast<int>(buffer.size()), MPI_UNSIGNED_LONG_LONG,
                  0, MPI_COMM_WORLD);
#endif
        apply(buffer.data(), count);
    }
    // Every stored cell is a local cell of exactly one rank on the same mesh.
    unsigned long long matched = localMatched;
#ifdef RICH_MPI
    MPI_Allreduce(&localMatched, &matched, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (matched != total) {
        UniversalError eo("Restart load-balance measurements name cells that are not in this mesh");
        eo.addEntry("stored cells", static_cast<double>(total));
        eo.addEntry("cells found", static_cast<double>(matched));
        throw eo;
    }
}

} // namespace

PassRestartState BeginPassRestart(
    Config const& cfg, RestartPass pass, SphericalObserver& observer,
    size_t generationsToAdd, int rank)
{
    char const* const name = pass == RestartPass::Forward ? "MG" : "Grey";
    PassRestartState state = LoadPassRestartState(
        cfg.restartFrom, pass, observer.getNumObservers(), observer.getNumGroups(), rank);
    if (cfg.adaptiveSourceCells && state.adaptive.scoreByCellID.empty())
        throw UniversalError(
            std::string("Restart file has no learned ") + name +
            " source cells; the burn-in is not rerun on a restart");
    LoadObserverStatistics(cfg.restartFrom, pass, observer,
                           state.completedFinalGenerations, rank);
    if (rank == 0)
        std::cout << name << " restart: continuing " << state.completedFinalGenerations
                  << " final generations with " << state.adaptive.scoreByCellID.size()
                  << " learned cells; adding " << generationsToAdd
                  << " (burn-in and probe skipped)" << std::endl;
    return state;
}

void RebalanceFromStoredMeasurements(
    LoadBalanceMeasurements const& stored,
    std::vector<ComputationalCell3D> const& cells,
    size_t nCells,
    std::string const& passName,
    std::function<void(std::vector<size_t> const&, std::vector<size_t> const&)> const& rebalance,
    int rank)
{
    std::vector<size_t> steps, particles;
    DistributeLoadBalanceMeasurements(stored, cells, nCells, steps, particles, rank);
    if (rank == 0)
        std::cout << passName << " restart: measured load balance from the stored last generation" << std::endl;
    rebalance(steps, particles);
}

LoadBalanceMeasurements FinishPassLoadBalance(
    LoadBalanceMeasurements const& localLastFinal,
    LoadBalanceMeasurements stored,
    bool ranGenerations,
    int rank)
{
    LoadBalanceMeasurements gathered = GatherLoadBalanceMeasurements(localLastFinal, rank);
    if (ranGenerations)
        return gathered;
    return stored;
}

uint64_t RestartGenerationSeed(RestartPass pass, int rank, size_t absoluteFinalGeneration)
{
    // A tag outside the range of the fresh-run seeds
    // ((rank + 12345678) * total + gen), mixed so neighbouring ranks and
    // generations get unrelated streams.
    uint64_t x = SplitMix64(pass == RestartPass::Forward ? 0x52455354415254ULL : 0x5245535447524559ULL);
    x = SplitMix64(x ^ static_cast<uint64_t>(rank));
    x = SplitMix64(x ^ static_cast<uint64_t>(absoluteFinalGeneration));
    return x;
}

size_t GreyGenerationCount(Config const& cfg)
{
    if (cfg.nGenerations == 0)
        return 0;
    return cfg.fluxSourceCompare
        ? cfg.nGenerations : std::max<size_t>(1, cfg.nGenerations / 2);
}

} // namespace imc_postprocess_tde
