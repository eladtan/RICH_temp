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
    WeightSplitState& state)
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
}

unsigned long long PrintWeightSplitReport(
    Config const& cfg,
    RadiationIMC const& physics,
    WeightSplitState const& state,
    std::string const& label,
    size_t generation,
    int rank)
{
    if(!(cfg.splitObserverFraction > 0.0))
        return 0;
    unsigned long long counts[4] = {
        static_cast<unsigned long long>(physics.getPostProcessSplitEvents()),
        static_cast<unsigned long long>(physics.getPostProcessSplitCopies()),
        static_cast<unsigned long long>(physics.getPostProcessSplitCapped()),
        static_cast<unsigned long long>(physics.getPostProcessSplitRefused())};
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, counts, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    if(rank != 0)
        return counts[1];
    std::ostringstream line;
    line << std::setprecision(4) << "WEIGHT_SPLIT " << label << " gen=" << generation
         << " max_weight=" << state.maxWeight << " erg important_cells=" << state.importantCells
         << " split_events=" << counts[0] << " copies=" << counts[1] << " capped_events=" << counts[2] << " budget_refusals=" << counts[3];
    std::cout << line.str() << std::endl;
    return counts[1];
}

} // namespace imc_postprocess_tde
