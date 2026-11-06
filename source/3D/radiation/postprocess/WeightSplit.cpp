#include "WeightSplit.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "source/misc/universal_error.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#include "source/mpi/mpi_commands_3d.hpp"
#endif

namespace imc_postprocess_tde {

namespace {

// Escaped energy of each cell in ids, looked up in the ownedEscapedEnergyByCellID
// of the rank that holds the cell's adaptive statistics (AdaptiveCellOwner),
// which in general is not the rank that holds the cell. Collective.
std::vector<double> FetchEscapedEnergy(
    std::vector<size_t> const& ids,
    std::unordered_map<size_t, double> const& ownedEscaped)
{
    std::vector<double> result(ids.size(), 0.0);
    auto lookup = [&](size_t id) {
        auto const it = ownedEscaped.find(id);
        return (it != ownedEscaped.end() && it->second > 0.0 && std::isfinite(it->second)) ? it->second : 0.0;
    };
#ifdef RICH_MPI
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    std::vector<int> sendCounts(static_cast<size_t>(ranks), 0);
    std::vector<int> recvCounts(static_cast<size_t>(ranks), 0);
    for (size_t const id : ids)
        ++sendCounts[static_cast<size_t>(AdaptiveCellOwner(id, ranks))];
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> sendDispls(static_cast<size_t>(ranks), 0);
    std::vector<int> recvDispls(static_cast<size_t>(ranks), 0);
    for (int r = 1; r < ranks; ++r) {
        sendDispls[static_cast<size_t>(r)] = sendDispls[static_cast<size_t>(r - 1)] + sendCounts[static_cast<size_t>(r - 1)];
        recvDispls[static_cast<size_t>(r)] = recvDispls[static_cast<size_t>(r - 1)] + recvCounts[static_cast<size_t>(r - 1)];
    }
    size_t const totalRecv = static_cast<size_t>(recvDispls.back() + recvCounts.back());
    // Requests grouped by owner; position[k] is the index in ids of request k.
    std::vector<unsigned long long> request(ids.size());
    std::vector<size_t> position(ids.size());
    std::vector<int> next(sendDispls);
    for (size_t i = 0; i < ids.size(); ++i) {
        size_t const slot = static_cast<size_t>(next[static_cast<size_t>(AdaptiveCellOwner(ids[i], ranks))]++);
        request[slot] = static_cast<unsigned long long>(ids[i]);
        position[slot] = i;
    }
    std::vector<unsigned long long> received(totalRecv);
    MPI_Alltoallv(request.data(), sendCounts.data(), sendDispls.data(), MPI_UNSIGNED_LONG_LONG,
                  received.data(), recvCounts.data(), recvDispls.data(), MPI_UNSIGNED_LONG_LONG,
                  MPI_COMM_WORLD);
    std::vector<double> reply(totalRecv);
    for (size_t k = 0; k < totalRecv; ++k)
        reply[k] = lookup(static_cast<size_t>(received[k]));
    std::vector<double> answer(ids.size());
    MPI_Alltoallv(reply.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                  answer.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                  MPI_COMM_WORLD);
    for (size_t k = 0; k < ids.size(); ++k)
        result[position[k]] = answer[k];
#else
    for (size_t i = 0; i < ids.size(); ++i)
        result[i] = lookup(ids[i]);
#endif
    return result;
}

} // namespace

std::vector<double> ComputeGroupWindows(double scalarWindow, double scale, double floor,
                                        std::vector<double> const& spectrum)
{
    if (!(scalarWindow > 0.0) || !std::isfinite(scalarWindow) || !(scale > 0.0) || !(scale <= 1.0) ||
        !(floor > 0.0) || !(floor <= 1.0) || spectrum.empty()) {
        UniversalError eo("ComputeGroupWindows: invalid window parameters");
        eo.addEntry("scalarWindow", scalarWindow);
        eo.addEntry("scale", scale);
        eo.addEntry("floor", floor);
        eo.addEntry("groups", spectrum.size());
        throw eo;
    }
    double total = 0.0;
    for (double const e : spectrum) {
        if (!(e >= 0.0) || !std::isfinite(e))
            throw UniversalError("ComputeGroupWindows: negative or non-finite escaping spectrum");
        total += e;
    }
    if (!(total > 0.0) || !std::isfinite(total))
        throw UniversalError("ComputeGroupWindows: escaping spectrum with no energy");
    std::vector<double> windows(spectrum.size(), 0.0);
    for (size_t g = 0; g < spectrum.size(); ++g) {
        windows[g] = scalarWindow * scale * std::max(floor, spectrum[g] / total);
        if (!(windows[g] > 0.0) || !std::isfinite(windows[g]) || windows[g] > scalarWindow) {
            UniversalError eo("ComputeGroupWindows: group window underflowed or exceeds the scalar window");
            eo.addEntry("group", g);
            eo.addEntry("window", windows[g]);
            eo.addEntry("scalarWindow", scalarWindow);
            throw eo;
        }
    }
    return windows;
}

void CaptureWeightSplitSpectrum(std::vector<double> const& groupEnergy, long long generation,
                                WeightSplitState& state)
{
    double total = 0.0;
    for (double const e : groupEnergy) {
        if (!(e >= 0.0) || !std::isfinite(e)) {
            UniversalError eo("Weight window: negative or non-finite escaping group energy");
            eo.addEntry("generation", generation);
            eo.addEntry("energy", e);
            throw eo;
        }
        total += e;
    }
    if (!(total > 0.0))
        return;  // keeps the previous spectrum (or none)
    state.groupSpectrum = groupEnergy;
    state.spectrumValid = true;
    state.spectrumGeneration = generation;
}

void AccumulateWeightSplitStatistics(
    Config const& cfg,
    PostprocessRuntime const& runtime,
    AdaptiveSourceUpdateSummary const& summary,
    WeightSplitState& state)
{
    if(!(cfg.splitObserverFraction > 0.0))
        return;
    size_t const nCells = runtime.tess.GetPointNo();
    state.escapedEnergy.resize(nCells, 0.0);
    state.emittedEnergy.resize(nCells, 0.0);
    state.accumulated = true;
    std::vector<size_t> emitting;
    std::vector<size_t> ids;
    for(size_t i = 0; i < nCells; ++i)
    {
        double luminosity = 0.0;
        if(i < runtime.volumeEmissionCellLuminosity.size())
            luminosity += runtime.volumeEmissionCellLuminosity[i];
        if(i < runtime.fluxSourceCellLuminosity.size())
            luminosity += runtime.fluxSourceCellLuminosity[i];
        if(!(luminosity > 0.0) || !std::isfinite(luminosity))
            continue;
        state.emittedEnergy[i] += luminosity * cfg.sourceDt;
        emitting.push_back(i);
        ids.push_back(runtime.cells[i].ID);
    }
    std::vector<double> const escaped = FetchEscapedEnergy(ids, summary.ownedEscapedEnergyByCellID);
    for(size_t k = 0; k < emitting.size(); ++k)
        state.escapedEnergy[emitting[k]] += escaped[k];
}

void MigrateWeightSplitState(PostprocessRuntime const& runtime, WeightSplitState& state)
{
#ifdef RICH_MPI
    if(!state.accumulated)
        return;
    // The vectors index the cells of the decomposition being left; the
    // exchange keeps the staying cells and appends the arriving ones in the
    // same order as the cells' own exchange.
    MPI_exchange_data(runtime.tess, state.escapedEnergy, false);
    MPI_exchange_data(runtime.tess, state.emittedEnergy, false);
#else
    (void)runtime;
    (void)state;
#endif
}

void InstallWeightSplit(
    Config const& cfg,
    PostprocessRuntime const& runtime,
    RadiationIMC& physics,
    WeightSplitState& state,
    bool multigroupPass)
{
    if(!(cfg.splitObserverFraction > 0.0))
        return;
    size_t const nCells = runtime.tess.GetPointNo();
    if(state.escapedEnergy.size() != nCells || state.emittedEnergy.size() != nCells)
    {
        state.escapedEnergy.resize(nCells, 0.0);
        state.emittedEnergy.resize(nCells, 0.0);
    }
    state.maxWeight = cfg.nObservers > 0 && runtime.lastEscapingLuminosity > 0.0
        ? cfg.splitObserverFraction * runtime.lastEscapingLuminosity * cfg.sourceDt /
              static_cast<double>(cfg.nObservers)
        : 0.0;
    std::vector<double> importance(nCells, 0.0);
    unsigned long long important = 0;
    for(size_t i = 0; i < nCells; ++i)
    {
        if(!(state.emittedEnergy[i] > 0.0) || !(state.escapedEnergy[i] > 0.0))
            continue;
        double const escapeProbability = state.escapedEnergy[i] / state.emittedEnergy[i];
        importance[i] = std::min(1.0, escapeProbability / cfg.splitEscapeReference);
        ++important;
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &important, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    state.importantCells = important;
    physics.setPostProcessWeightSplit(state.maxWeight, std::move(importance), cfg.splitMaxCopies, cfg.splitCopyBudget);

    state.groupMaxWeight.clear();
    for (auto& candidate : state.candidateMaxWeight)
        candidate.clear();
    state.importanceFloor = 0.0;
    state.installedShares.clear();
    state.installedSpectrumGeneration = -1;
    if (!multigroupPass || !(cfg.splitPerGroup || cfg.packetProvenance) || !(state.maxWeight > 0.0))
        return;
    RadiationIMC::PostProcessGroupWindow window;
    if (state.spectrumValid) {
        if (state.groupSpectrum.size() != static_cast<size_t>(ENERGY_GROUPS_NUM))
            throw UniversalError("Weight window: escaping spectrum does not have ENERGY_GROUPS_NUM groups");
        double total = 0.0;
        for (double const e : state.groupSpectrum)
            total += e;
        state.installedShares.assign(state.groupSpectrum.size(), 0.0);
        for (size_t g = 0; g < state.groupSpectrum.size(); ++g)
            state.installedShares[g] = state.groupSpectrum[g] / total;
        state.installedSpectrumGeneration = state.spectrumGeneration;
        if (cfg.splitPerGroup) {
            state.groupMaxWeight = ComputeGroupWindows(
                state.maxWeight, cfg.splitGroupScale, cfg.splitGroupFloor, state.groupSpectrum);
            window.active = true;
            for (size_t g = 0; g < state.groupMaxWeight.size(); ++g)
                window.maxWeight[g] = state.groupMaxWeight[g];
            // The floor starts with the full-emission importance statistics.
            state.importanceFloor = EffectiveImportanceFloor(cfg.splitImportanceFloor, state.accumulated);
            window.importanceFloor = state.importanceFloor;
        }
        if (cfg.packetProvenance) {
            window.provenance = true;
            window.shadowValid = true;
            for (size_t c = 0; c < kProvenanceCandidateScales.size(); ++c) {
                state.candidateMaxWeight[c] = ComputeGroupWindows(
                    state.maxWeight, kProvenanceCandidateScales[c], cfg.splitGroupFloor, state.groupSpectrum);
                for (size_t g = 0; g < state.candidateMaxWeight[c].size(); ++g)
                    window.shadowMaxWeight[c][g] = state.candidateMaxWeight[c][g];
            }
        }
    } else if (cfg.packetProvenance) {
        // No spectrum yet: representation history only.
        window.provenance = true;
    }
    physics.setPostProcessGroupWindow(window);
}

unsigned long long PrintWeightSplitReport(
    Config const& cfg,
    RadiationIMC const& physics,
    WeightSplitState const& state,
    std::string const& label,
    size_t generation,
    int rank,
    ForwardPostprocessResult* totals)
{
    if(!(cfg.splitObserverFraction > 0.0))
        return 0;
    unsigned long long counts[6] = {
        static_cast<unsigned long long>(physics.getPostProcessSplitEvents()),
        static_cast<unsigned long long>(physics.getPostProcessSplitCopies()),
        static_cast<unsigned long long>(physics.getPostProcessSplitCapped()),
        static_cast<unsigned long long>(physics.getPostProcessSplitRefused()),
        static_cast<unsigned long long>(physics.getPostProcessSplitSkippedResidentSteps()),
        static_cast<unsigned long long>(physics.getPostProcessSplitSkippedNoImportanceSteps())};
    // Rank maxima of the copies made and of the refusals, and the number of
    // ranks whose budget reduced or refused a split.
    unsigned long long rankExtremes[3] = {counts[1], counts[3], counts[3] > 0 ? 1ULL : 0ULL};
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, counts, 6, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, rankExtremes, 2, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, rankExtremes + 2, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    bool const groupReport = !state.groupMaxWeight.empty() || !state.candidateMaxWeight[0].empty() ||
        (cfg.packetProvenance && label == "MG");
    constexpr size_t G = static_cast<size_t>(ENERGY_GROUPS_NUM);
    std::vector<unsigned long long> groupCounts;
    if (groupReport) {
        auto const& c = physics.getPostProcessGroupSplitCounters();
        groupCounts.resize(5 * G);
        for (size_t g = 0; g < G; ++g) {
            groupCounts[0 * G + g] = c.events[g];
            groupCounts[1 * G + g] = c.copies[g];
            groupCounts[2 * G + g] = c.capped[g];
            groupCounts[3 * G + g] = c.refused[g];
            groupCounts[4 * G + g] = c.skippedNoImportance[g];
        }
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, groupCounts.data(), static_cast<int>(groupCounts.size()),
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    }
    if (totals) {
        totals->splitRefusals += counts[3];
        if (!groupCounts.empty()) {
            totals->groupSplitCopies.resize(G, 0);
            for (size_t g = 0; g < G; ++g)
                totals->groupSplitCopies[g] += groupCounts[1 * G + g];
        }
    }
    if(rank != 0)
        return counts[1];
    std::ostringstream line;
    line << std::setprecision(4) << "WEIGHT_SPLIT " << label << " gen=" << generation
         << " max_weight=" << state.maxWeight << " erg important_cells=" << state.importantCells
         << " split_events=" << counts[0] << " copies=" << counts[1] << " capped_events=" << counts[2] << " budget_refusals=" << counts[3]
         << " overweight_resident_steps=" << counts[4] << " overweight_no_importance_steps=" << counts[5]
         << " rank_max_copies=" << rankExtremes[0] << " rank_max_refusals=" << rankExtremes[1]
         << " ranks_with_refusals=" << rankExtremes[2];
    std::cout << line.str() << std::endl;
    if (groupReport) {
        auto list = [](std::ostringstream& out, auto const& values, size_t begin, size_t n) {
            out << "[";
            for (size_t g = 0; g < n; ++g)
                out << (g ? "," : "") << values[begin + g];
            out << "]";
        };
        std::ostringstream window;
        // The windows this generation was transported with and the
        // spectrum they came from (not the spectrum just captured).
        window << std::setprecision(4) << "WEIGHT_WINDOW_GROUP " << label << " gen=" << generation
               << " installed_spectrum_generation=" << state.installedSpectrumGeneration
               << " scale=" << cfg.splitGroupScale << " floor=" << cfg.splitGroupFloor
               << " importance_floor=" << state.importanceFloor;
        if (!state.installedShares.empty()) {
            window << " share=";
            list(window, state.installedShares, 0, state.installedShares.size());
        }
        if (!state.groupMaxWeight.empty()) {
            window << " active_window=";
            list(window, state.groupMaxWeight, 0, state.groupMaxWeight.size());
        }
        for (size_t c = 0; c < state.candidateMaxWeight.size(); ++c) {
            if (state.candidateMaxWeight[c].empty())
                continue;
            window << " candidate" << c << "_scale" << kProvenanceCandidateScales[c] << "_window=";
            list(window, state.candidateMaxWeight[c], 0, state.candidateMaxWeight[c].size());
        }
        std::cout << window.str() << std::endl;
        std::ostringstream groups;
        groups << "WEIGHT_SPLIT_GROUP " << label << " gen=" << generation << " events=";
        list(groups, groupCounts, 0 * G, G);
        groups << " copies=";
        list(groups, groupCounts, 1 * G, G);
        groups << " capped=";
        list(groups, groupCounts, 2 * G, G);
        groups << " refused=";
        list(groups, groupCounts, 3 * G, G);
        groups << " overweight_no_importance_steps=";
        list(groups, groupCounts, 4 * G, G);
        std::cout << groups.str() << std::endl;
    }
    return counts[1];
}

} // namespace imc_postprocess_tde
