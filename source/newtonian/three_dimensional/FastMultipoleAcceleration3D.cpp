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
    const double localTimes[21] = {
        stats.totalSeconds, stats.buildSeconds, stats.topologyRebuildSeconds,
        stats.rootDescriptorExchangeSeconds, stats.processTopologySeconds,
        stats.letPlanSeconds, stats.letBuildResetSeconds,
        stats.letDescriptorTraversalSeconds, stats.letFinalizeSeconds,
        stats.letSubscriptionSeconds, stats.letPruneCompactSeconds,
        stats.localTraversalSeconds, stats.letExecuteSeconds,
        stats.upwardSeconds, stats.processUpwardSeconds,
        stats.processInteractionSeconds, stats.processDownwardSeconds,
        stats.letExchangeSeconds, stats.letM2LSeconds,
        stats.letP2PSeconds, stats.downwardSeconds};
    double maximumTimes[21] = {};
    unsigned long long reusedActiveRanks =
        stats.localInteractionPlanReused ? 1ull : 0ull;
    unsigned long long globalReusedActiveRanks = reusedActiveRanks;
    const unsigned long long localLetPlanBytes =
        static_cast<unsigned long long>(stats.letPlanBytes);
    unsigned long long maximumLetPlanBytes = localLetPlanBytes;
    const unsigned long long localInactiveCounts[9] = {
        static_cast<unsigned long long>(stats.localInactiveM2LCount),
        static_cast<unsigned long long>(stats.localInactiveP2PBlockCount),
        static_cast<unsigned long long>(stats.letInactiveM2LCount),
        static_cast<unsigned long long>(stats.letInactiveP2PBlockCount),
        static_cast<unsigned long long>(stats.letZeroMultipolePayloadCount),
        static_cast<unsigned long long>(stats.letOmittedMultipolePayloadCount),
        static_cast<unsigned long long>(stats.letOmittedParticlePayloadCount),
        static_cast<unsigned long long>(stats.bytesSent),
        static_cast<unsigned long long>(stats.bytesReceived)};
    const unsigned long long localPlanCounts[4] = {
        static_cast<unsigned long long>(stats.localPlannedM2LCount),
        static_cast<unsigned long long>(stats.localPlannedP2PBlockCount),
        static_cast<unsigned long long>(stats.letPlannedM2LCount),
        static_cast<unsigned long long>(stats.letPlannedP2PBlockCount)};
    unsigned long long globalInactiveCounts[9] = {
        localInactiveCounts[0], localInactiveCounts[1], localInactiveCounts[2],
        localInactiveCounts[3], localInactiveCounts[4], localInactiveCounts[5],
        localInactiveCounts[6], localInactiveCounts[7], localInactiveCounts[8]};
    unsigned long long globalPlanCounts[4] = {
        localPlanCounts[0], localPlanCounts[1], localPlanCounts[2],
        localPlanCounts[3]};
    const unsigned long long localActiveCounts[2] = {
        static_cast<unsigned long long>(stats.letM2LCount),
        static_cast<unsigned long long>(stats.letP2PBlockCount)};
    unsigned long long globalActiveCounts[2] = {
        localActiveCounts[0], localActiveCounts[1]};
    const unsigned long long localBytesOwned =
        static_cast<unsigned long long>(stats.bytesOwned);
    const unsigned long long localPeakRemoteBytes =
        static_cast<unsigned long long>(stats.peakRemoteBytes);
    unsigned long long maximumBytesOwned = localBytesOwned;
    unsigned long long maximumPeakRemoteBytes = localPeakRemoteBytes;
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Reduce(localTimes, maximumTimes, 21, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&reusedActiveRanks, &globalReusedActiveRanks, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localLetPlanBytes, &maximumLetPlanBytes, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(localInactiveCounts, globalInactiveCounts, 9,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(localPlanCounts, globalPlanCounts, 4,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(localActiveCounts, globalActiveCounts, 2,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localBytesOwned, &maximumBytesOwned, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localPeakRemoteBytes, &maximumPeakRemoteBytes, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
#else
    for(int i = 0; i < 21; ++i)
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
         << " let_reset_max=" << maximumTimes[6]
         << " let_descriptor_traversal_max=" << maximumTimes[7]
         << " let_finalize_max=" << maximumTimes[8]
         << " let_subscription_max=" << maximumTimes[9]
         << " let_prune_compact_max=" << maximumTimes[10]
         << " local_traversal_max=" << maximumTimes[11]
         << " let_execute_max=" << maximumTimes[12]
         << " upward_max=" << maximumTimes[13]
         << " process_upward_max=" << maximumTimes[14]
         << " process_interaction_max=" << maximumTimes[15]
         << " process_downward_max=" << maximumTimes[16]
         << " let_exchange_max=" << maximumTimes[17]
         << " let_m2l_max=" << maximumTimes[18]
         << " let_p2p_max=" << maximumTimes[19]
         << " downward_max=" << maximumTimes[20]
         << " epoch=" << stats.topologyEpoch
         << " rebuilds=" << stats.topologyRebuildCount
         << " process_rebuilds=" << stats.processTopologyRebuildCount
         << " let_rebuilds=" << stats.letTopologyRebuildCount
         << " root_change_ranks=" << stats.ranksWithRootGeometryChange
         << " leaf_change_ranks=" << stats.ranksWithLeafTopologyChange
         << " occupancy_change_ranks="
         << stats.ranksWithLeafOccupancyChange
         << " count_only_change_ranks="
         << stats.ranksWithCountOnlyLeafChange
         << " persistent_refit_ranks="
         << stats.persistentTreeRefitRankCount
         << " persistent_leaf_splits="
         << stats.persistentLeafSplitCount
         << " persistent_subtree_merges="
         << stats.persistentSubtreeMergeCount
         << " persistent_empty_leaves="
         << stats.persistentEmptyLeafCount
         << " count_only_reused=" << (stats.countOnlyTopologyReused ? 1 : 0)
         << " process_rebuilt=" << (stats.processTopologyRebuilt ? 1 : 0)
         << " let_rebuilt=" << (stats.letTopologyRebuilt ? 1 : 0)
         << " process_comm_reused="
         << (stats.processCommunicatorsReused ? 1 : 0)
         << " let_comm_reused=" << (stats.letCommunicatorReused ? 1 : 0)
         << " let_storage_reused="
         << (stats.letBuildStorageReused ? 1 : 0)
         << " forced_rebuild=" << (stats.topologyRebuildForced ? 1 : 0)
         << " active_ranks=" << stats.activeRankCount
         << " local_plan_reused_ranks=" << globalReusedActiveRanks
         << " local_plan_reused_all="
         << (globalReusedActiveRanks ==
                 static_cast<unsigned long long>(stats.activeRankCount) ? 1 : 0)
         << " let_plan_bytes_max=" << maximumLetPlanBytes
         << " bytes_owned_max=" << maximumBytesOwned
         << " peak_remote_bytes_max=" << maximumPeakRemoteBytes
         << " local_planned_m2l_sum=" << globalPlanCounts[0]
         << " local_planned_p2p_blocks_sum=" << globalPlanCounts[1]
         << " let_planned_m2l_sum=" << globalPlanCounts[2]
         << " let_planned_p2p_blocks_sum=" << globalPlanCounts[3]
         << " let_active_m2l_sum=" << globalActiveCounts[0]
         << " let_active_p2p_blocks_sum=" << globalActiveCounts[1]
         << " local_inactive_m2l_sum=" << globalInactiveCounts[0]
         << " local_inactive_p2p_blocks_sum=" << globalInactiveCounts[1]
         << " let_inactive_m2l_sum=" << globalInactiveCounts[2]
         << " let_inactive_p2p_blocks_sum=" << globalInactiveCounts[3]
         << " let_zero_multipole_payloads_sum=" << globalInactiveCounts[4]
         << " let_omitted_multipole_payloads_sum=" << globalInactiveCounts[5]
         << " let_omitted_particle_payloads_sum=" << globalInactiveCounts[6]
         << " bytes_sent_sum=" << globalInactiveCounts[7]
         << " bytes_received_sum=" << globalInactiveCounts[8];
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
