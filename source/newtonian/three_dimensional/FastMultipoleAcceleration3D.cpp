#include "newtonian/three_dimensional/FastMultipoleAcceleration3D.hpp"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <sstream>

#include "misc/memory_profile.hpp"
#include "misc/universal_error.hpp"

namespace
{
FmmGravityOptions validateAccelerationOptions(FmmGravityOptions options)
{
#ifdef RICH_MPI
    int localSupported = options.computePotential ? 0 : 1;
    int globallySupported = 0;
    MPI_Allreduce(&localSupported, &globallySupported, 1, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);
    if(globallySupported == 0)
        throw UniversalError(
            "FastMultipoleAcceleration3D: computePotential is unsupported by the acceleration adapter");
#else
    if(options.computePotential)
        throw UniversalError(
            "FastMultipoleAcceleration3D: computePotential is unsupported by the acceleration adapter");
#endif
    return options;
}

#ifdef RICH_MPI
double validateDistributedGravityConstant(double value)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if(initialized == 0)
        throw UniversalError(
            "FastMultipoleAcceleration3D: MPI must be initialized before construction");
    int localFinite = std::isfinite(value) ? 1 : 0;
    int globalFinite = 0;
    MPI_Allreduce(&localFinite, &globalFinite, 1, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);
    if(globalFinite == 0)
        throw UniversalError(
            "FastMultipoleAcceleration3D: G must be finite on every MPI rank");
    double minimum = 0.0;
    double maximum = 0.0;
    MPI_Allreduce(&value, &minimum, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&value, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(minimum != maximum)
        throw UniversalError(
            "FastMultipoleAcceleration3D: G differs across MPI ranks");
    return value;
}

void requireOnEveryRank(bool localCondition, const char* message)
{
    int local = localCondition ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if(global == 0)
        throw UniversalError(message);
}
#endif

bool fmmTraceEnabled()
{
#ifdef RICH_MPI
    static int enabled = -1;
    if(enabled < 0)
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int rootEnabled = 0;
        if(rank == 0)
        {
            const char* value = std::getenv("RICH_FMM_TRACE");
            rootEnabled = value != nullptr && value[0] != '\0' &&
                          !(value[0] == '0' && value[1] == '\0') ? 1 : 0;
        }
        MPI_Bcast(&rootEnabled, 1, MPI_INT, 0, MPI_COMM_WORLD);
        enabled = rootEnabled;
    }
    return enabled != 0;
#else
    const char* value = std::getenv("RICH_FMM_TRACE");
    return value != nullptr && value[0] != '\0' &&
           !(value[0] == '0' && value[1] == '\0');
#endif
}

void traceFmmSolve(const FmmSolveStats& stats)
{
    if(!fmmTraceEnabled())
        return;

    static std::uint64_t call = 0;
    ++call;
    const double localTimes[8] = {
        stats.totalSeconds, stats.buildSeconds, stats.topologyRebuildSeconds,
        stats.rootDescriptorExchangeSeconds, stats.processTopologySeconds,
        stats.letPlanSeconds, stats.localTraversalSeconds,
        stats.letExecuteSeconds};
    double maximumTimes[8] = {};
    unsigned long long reusedActiveRanks =
        stats.localInteractionPlanReused ? 1ull : 0ull;
    unsigned long long globalReusedActiveRanks = reusedActiveRanks;
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Reduce(localTimes, maximumTimes, 8, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&reusedActiveRanks, &globalReusedActiveRanks, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    for(int i = 0; i < 8; ++i)
        maximumTimes[i] = localTimes[i];
#endif
    if(rank != 0)
        return;

    std::ostringstream line;
    line.setf(std::ios::scientific);
    line.precision(8);
    line << "fmm_solve_trace call=" << call
         << " total_max=" << maximumTimes[0]
         << " build_max=" << maximumTimes[1]
         << " topology_max=" << maximumTimes[2]
         << " descriptor_max=" << maximumTimes[3]
         << " process_topology_max=" << maximumTimes[4]
         << " let_plan_max=" << maximumTimes[5]
         << " local_traversal_max=" << maximumTimes[6]
         << " let_execute_max=" << maximumTimes[7]
         << " epoch=" << stats.topologyEpoch
         << " rebuilds=" << stats.topologyRebuildCount
         << " process_rebuilds=" << stats.processTopologyRebuildCount
         << " let_rebuilds=" << stats.letTopologyRebuildCount
         << " root_change_ranks=" << stats.ranksWithRootGeometryChange
         << " leaf_change_ranks=" << stats.ranksWithLeafTopologyChange
         << " process_rebuilt=" << (stats.processTopologyRebuilt ? 1 : 0)
         << " let_rebuilt=" << (stats.letTopologyRebuilt ? 1 : 0)
         << " process_comm_reused="
         << (stats.processCommunicatorsReused ? 1 : 0)
         << " let_comm_reused=" << (stats.letCommunicatorReused ? 1 : 0)
         << " forced_rebuild=" << (stats.topologyRebuildForced ? 1 : 0)
         << " active_ranks=" << stats.activeRankCount
         << " local_plan_reused_ranks=" << globalReusedActiveRanks
         << " local_plan_reused_all="
         << (globalReusedActiveRanks ==
                 static_cast<unsigned long long>(stats.activeRankCount) ? 1 : 0);
    std::cout << line.str() << std::endl;
}
}

FastMultipoleAcceleration3D::FastMultipoleAcceleration3D(FmmGravityOptions options,
                                                         double G):
#ifdef RICH_MPI
    G_(validateDistributedGravityConstant(G)),
#else
    G_(G),
#endif
    calculator_(validateAccelerationOptions(options))
{
#ifndef RICH_MPI
    if(!std::isfinite(G_))
        throw UniversalError("FastMultipoleAcceleration3D: G must be finite");
#endif
}

#ifdef RICH_MPI
FastMultipoleAcceleration3D::FastMultipoleAcceleration3D(
    FmmGravityOptions options,
    FmmDistributedOptions distributedOptions,
    double G):
    G_(validateDistributedGravityConstant(G)),
    calculator_(validateAccelerationOptions(options), distributedOptions)
{}
#endif

void FastMultipoleAcceleration3D::operator()(const Tessellation3D& tess,
                                             const vector<ComputationalCell3D>& cells,
                                             const vector<Conserved3D>& fluxes,
                                             const double time,
                                             vector<Vector3D>& acc) const
{
    (void) fluxes;
    (void) time;
    MEMORY_PROFILE_SCOPE("fmm gravity source");

    const std::size_t N = tess.GetPointNo();
#ifdef RICH_MPI
    requireOnEveryRank(cells.size() >= N,
        "FastMultipoleAcceleration3D: cell array is smaller than owned tessellation on an MPI rank");
#else
    if(cells.size() < N)
        throw UniversalError("FastMultipoleAcceleration3D: cell array is smaller than owned tessellation");
#endif
    points_.resize(N);
    masses_.resize(N);
#ifdef RICH_MPI
    cellIds_.resize(N);
#endif
    for(std::size_t cellIdx = 0; cellIdx < N; ++cellIdx)
    {
        points_[cellIdx] = tess.GetCellCM(cellIdx);
        masses_[cellIdx] = cells[cellIdx].density * tess.GetVolume(cellIdx);
#ifdef RICH_MPI
        cellIds_[cellIdx] = static_cast<std::uint64_t>(cells[cellIdx].ID);
#else
        if(!std::isfinite(masses_[cellIdx]))
        {
            UniversalError error("FastMultipoleAcceleration3D: non-finite cell mass");
            error.addEntry("cell", cellIdx);
            throw error;
        }
#endif
    }

    const std::pair<Vector3D, Vector3D> boundaries = tess.GetBoxCoordinates();
#ifdef RICH_MPI
    calculator_.solve(points_, masses_, cellIds_, boundaries.first,
                      boundaries.second, acc);
#else
    calculator_.solve(points_, masses_, boundaries.first, boundaries.second, acc);
#endif
    traceFmmSolve(calculator_.stats());

    bool finiteAcceleration = true;
#ifndef RICH_MPI
    std::size_t firstInvalid = 0;
#endif
    for(std::size_t i = 0; i < acc.size(); ++i)
    {
        acc[i] *= G_;
        if(finiteAcceleration &&
           (!std::isfinite(acc[i].x) || !std::isfinite(acc[i].y) ||
            !std::isfinite(acc[i].z)))
        {
            finiteAcceleration = false;
#ifndef RICH_MPI
            firstInvalid = i;
#endif
        }
    }
#ifdef RICH_MPI
    requireOnEveryRank(finiteAcceleration,
        "FastMultipoleAcceleration3D: non-finite acceleration after G scaling on an MPI rank");
#else
    if(!finiteAcceleration)
    {
        UniversalError error("FastMultipoleAcceleration3D: non-finite acceleration after G scaling");
        error.addEntry("cell", firstInvalid);
        throw error;
    }
#endif
}

const FmmSolveStats& FastMultipoleAcceleration3D::getLastStats() const noexcept
{
    return calculator_.stats();
}
