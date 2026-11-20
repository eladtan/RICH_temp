#include "newtonian/three_dimensional/FastMultipoleAcceleration3D.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "misc/memory_profile.hpp"
#include "misc/universal_error.hpp"

namespace
{
// Exact position lookup for the individual-target adapter.  Two positions
// match when every coordinate compares equal (so -0.0 matches 0.0 and a NaN
// matches nothing), as with the node-based hash map this replaces; the table
// is a flat open-addressing array reused across calls.
bool SamePosition(Vector3D const& First, Vector3D const& Second)
{
    return First.x == Second.x && First.y == Second.y && First.z == Second.z;
}

// Bit pattern of a coordinate with -0.0 folded onto 0.0, so that coordinates
// comparing equal hash equal.
std::uint64_t CoordinateBits(double Value)
{
    const double Canonical = Value == 0.0 ? 0.0 : Value;
    std::uint64_t Bits = 0;
    std::memcpy(&Bits, &Canonical, sizeof(Bits));
    return Bits;
}

std::uint64_t MixBits(std::uint64_t Value)
{
    Value ^= Value >> 33;
    Value *= UINT64_C(0xff51afd7ed558ccd);
    Value ^= Value >> 33;
    Value *= UINT64_C(0xc4ceb9fe1a85ec53);
    Value ^= Value >> 33;
    return Value;
}

std::size_t PositionSlot(Vector3D const& Position, std::size_t Mask)
{
    const std::uint64_t Hash =
        MixBits(CoordinateBits(Position.x) ^ MixBits(CoordinateBits(Position.y) ^ MixBits(CoordinateBits(Position.z))));
    return static_cast<std::size_t>(Hash) & Mask;
}

// Fills Slots (power-of-two size, at most half full; entry = source index + 1,
// 0 = empty) with every source.  Returns false if two sources match; the first
// of them stays indexed.
bool IndexSourcePositions(std::vector<Vector3D> const& Sources, std::vector<std::size_t>& Slots)
{
    if(Sources.size() > std::numeric_limits<std::size_t>::max() / 4)
        throw UniversalError("FastMultipoleAcceleration3D: too many individual gravity sources");
    std::size_t Capacity = 16;
    while(Capacity < 2 * Sources.size())
        Capacity *= 2;
    Slots.assign(Capacity, 0);
    const std::size_t Mask = Capacity - 1;
    bool Unique = true;
    for(std::size_t Source = 0; Source < Sources.size(); ++Source) {
        std::size_t Slot = PositionSlot(Sources[Source], Mask);
        while(Slots[Slot] != 0 && !SamePosition(Sources[Slots[Slot] - 1], Sources[Source]))
            Slot = (Slot + 1) & Mask;
        if(Slots[Slot] == 0)
            Slots[Slot] = Source + 1;
        else
            Unique = false;
    }
    return Unique;
}

// Index of the source matching Position, or Sources.size() if there is none.
std::size_t FindSourcePosition(std::vector<Vector3D> const& Sources, std::vector<std::size_t> const& Slots,
                               Vector3D const& Position)
{
    const std::size_t Mask = Slots.size() - 1;
    std::size_t Slot = PositionSlot(Position, Mask);
    while(Slots[Slot] != 0) {
        if(SamePosition(Sources[Slots[Slot] - 1], Position))
            return Slots[Slot] - 1;
        Slot = (Slot + 1) & Mask;
    }
    return Sources.size();
}

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

#ifndef RICH_MPI
// Serial builds read the switch directly.  MPI builds take it from the
// distributed calculator, which reads it once, collectively on its own
// communicator, at construction (see FastMultipoleAcceleration3D below).
bool fmmTraceEnabled()
{
    const char* value = std::getenv("RICH_FMM_TRACE");
    return value != nullptr && value[0] != '\0' &&
           !(value[0] == '0' && value[1] == '\0');
}
#endif

// Print-only load-balance record for one solve (RICH_FMM_TRACE).  Collective:
// every rank contributes one fixed row to a Gather; rank 0 returns the
// aggregate line (and, behind RICH_FMM_TRACE_RANKS, one row per rank on a few
// solves).  Other ranks return an empty string.
std::string traceFmmBalance(const FmmSolveStats& stats, std::uint64_t call)
{
#ifdef RICH_MPI
    enum Column
    {
        cOwned, cInput, cLocalTraversal, cLetExecute, cLocalPairs, cLocalM2L,
        cLocalBlocks, cLetPairs, cLetM2L, cLeaves, cMaxLeaf, cMaxDepth,
        cOverCapacity, cOverSplit, cDepthCap, cMaxLeafDepthCap, cLeafSquared,
        cRootHalf, cExtentX, cExtentY, cExtentZ, cCenterX, cCenterY, cCenterZ,
        cUpward, cDownward, cLocalBypass, cLetBypass, cTotal, cLetP2PSeconds,
        cLetM2LSeconds, cBytesOwned, cLocalPlanBytes, cLetM2P, cDiagSeconds,
        cLetP2PBlocks, cColumns
    };
    double row[cColumns] = {};
    row[cOwned] = static_cast<double>(stats.particleCount);
    row[cInput] = static_cast<double>(stats.diagInputParticleCount);
    row[cLocalTraversal] = stats.localTraversalSeconds;
    row[cLetExecute] = stats.letExecuteSeconds;
    row[cLocalPairs] = static_cast<double>(stats.diagLocalP2PPairCount);
    row[cLocalM2L] = static_cast<double>(stats.diagLocalM2LCount);
    row[cLocalBlocks] = static_cast<double>(stats.diagLocalP2PBlockCount);
    row[cLetPairs] = static_cast<double>(stats.diagLetP2PPairCount);
    row[cLetM2L] = static_cast<double>(stats.letM2LCount);
    row[cLeaves] = static_cast<double>(stats.leafCount);
    row[cMaxLeaf] = static_cast<double>(stats.maxLeafOccupancy);
    row[cMaxDepth] = static_cast<double>(stats.maxDepth);
    row[cOverCapacity] = static_cast<double>(stats.diagLeavesOverCapacity);
    row[cOverSplit] = static_cast<double>(stats.diagLeavesOverSplitCapacity);
    row[cDepthCap] = static_cast<double>(stats.diagLeavesAtDepthCap);
    row[cMaxLeafDepthCap] = static_cast<double>(stats.diagMaxLeafAtDepthCap);
    row[cLeafSquared] = stats.diagLeafSquaredOccupancy;
    row[cRootHalf] = stats.diagRootHalfSize;
    row[cExtentX] = stats.diagParticleExtent[0];
    row[cExtentY] = stats.diagParticleExtent[1];
    row[cExtentZ] = stats.diagParticleExtent[2];
    row[cCenterX] = stats.diagRootCenter[0];
    row[cCenterY] = stats.diagRootCenter[1];
    row[cCenterZ] = stats.diagRootCenter[2];
    row[cUpward] = stats.upwardSeconds;
    row[cDownward] = stats.downwardSeconds;
    row[cLocalBypass] = static_cast<double>(stats.localOperatorCacheBypasses);
    row[cLetBypass] = static_cast<double>(stats.letOperatorCacheBypasses);
    row[cTotal] = stats.totalSeconds;
    row[cLetP2PSeconds] = stats.letP2PSeconds;
    row[cLetM2LSeconds] = stats.letM2LSeconds;
    row[cBytesOwned] = static_cast<double>(stats.bytesOwned);
    row[cLocalPlanBytes] = static_cast<double>(stats.localInteractionPlanBytes);
    row[cLetM2P] = static_cast<double>(stats.letM2PCount);
    row[cDiagSeconds] = stats.diagSeconds;
    row[cLetP2PBlocks] = static_cast<double>(stats.letP2PBlockCount);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<double> rows;
    if(rank == 0)
        rows.resize(static_cast<std::size_t>(size) * cColumns);
    MPI_Gather(row, cColumns, MPI_DOUBLE, rank == 0 ? rows.data() : nullptr,
               cColumns, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank != 0)
        return std::string();

    const auto value = [&](int r, int c) {
        return rows[static_cast<std::size_t>(r) * cColumns +
                    static_cast<std::size_t>(c)];
    };
    const auto argmax = [&](int c) {
        int best = 0;
        for(int r = 1; r < size; ++r)
            if(value(r, c) > value(best, c))
                best = r;
        return best;
    };
    const auto mean = [&](int c) {
        double sum = 0.0;
        for(int r = 0; r < size; ++r)
            sum += value(r, c);
        return sum / static_cast<double>(size);
    };
    const auto maximum = [&](int c) { return value(argmax(c), c); };
    const auto minimum = [&](int c) {
        double best = value(0, c);
        for(int r = 1; r < size; ++r)
            best = std::min(best, value(r, c));
        return best;
    };
    const auto sum = [&](int c) { return mean(c) * static_cast<double>(size); };
    const auto correlation = [&](int a, int b) {
        const double ma = mean(a);
        const double mb = mean(b);
        double sab = 0.0, saa = 0.0, sbb = 0.0;
        for(int r = 0; r < size; ++r)
        {
            const double da = value(r, a) - ma;
            const double db = value(r, b) - mb;
            sab += da * db;
            saa += da * da;
            sbb += db * db;
        }
        return saa > 0.0 && sbb > 0.0 ? sab / std::sqrt(saa * sbb) : 0.0;
    };
    const auto describe = [&](std::ostringstream& out, const char* prefix,
                              int r) {
        out << ' ' << prefix << "_rank=" << r
            << ' ' << prefix << "_owned=" << value(r, cOwned)
            << ' ' << prefix << "_input=" << value(r, cInput)
            << ' ' << prefix << "_local_traversal=" << value(r, cLocalTraversal)
            << ' ' << prefix << "_let_execute=" << value(r, cLetExecute)
            << ' ' << prefix << "_total=" << value(r, cTotal)
            << ' ' << prefix << "_upward=" << value(r, cUpward)
            << ' ' << prefix << "_downward=" << value(r, cDownward)
            << ' ' << prefix << "_local_p2p_pairs=" << value(r, cLocalPairs)
            << ' ' << prefix << "_local_p2p_blocks=" << value(r, cLocalBlocks)
            << ' ' << prefix << "_local_m2l=" << value(r, cLocalM2L)
            << ' ' << prefix << "_let_p2p_pairs=" << value(r, cLetPairs)
            << ' ' << prefix << "_let_p2p_blocks=" << value(r, cLetP2PBlocks)
            << ' ' << prefix << "_let_m2l=" << value(r, cLetM2L)
            << ' ' << prefix << "_let_m2p=" << value(r, cLetM2P)
            << ' ' << prefix << "_let_p2p_s=" << value(r, cLetP2PSeconds)
            << ' ' << prefix << "_let_m2l_s=" << value(r, cLetM2LSeconds)
            << ' ' << prefix << "_leaves=" << value(r, cLeaves)
            << ' ' << prefix << "_max_leaf=" << value(r, cMaxLeaf)
            << ' ' << prefix << "_max_depth=" << value(r, cMaxDepth)
            << ' ' << prefix << "_leaves_over_cap=" << value(r, cOverCapacity)
            << ' ' << prefix << "_leaves_over_split=" << value(r, cOverSplit)
            << ' ' << prefix << "_leaves_depth_cap=" << value(r, cDepthCap)
            << ' ' << prefix << "_max_leaf_depth_cap="
            << value(r, cMaxLeafDepthCap)
            << ' ' << prefix << "_leaf_sq=" << value(r, cLeafSquared)
            << ' ' << prefix << "_root_half=" << value(r, cRootHalf)
            << ' ' << prefix << "_extent=" << value(r, cExtentX) << ','
            << value(r, cExtentY) << ',' << value(r, cExtentZ)
            << ' ' << prefix << "_root_center=" << value(r, cCenterX) << ','
            << value(r, cCenterY) << ',' << value(r, cCenterZ)
            << ' ' << prefix << "_cache_bypass_local=" << value(r, cLocalBypass)
            << ' ' << prefix << "_cache_bypass_let=" << value(r, cLetBypass)
            << ' ' << prefix << "_bytes_owned=" << value(r, cBytesOwned)
            << ' ' << prefix << "_local_plan_bytes=" << value(r, cLocalPlanBytes);
    };

    std::ostringstream line;
    line.setf(std::ios::scientific);
    line.precision(4);
    const int slow = argmax(cLocalTraversal);
    const int big = argmax(cOwned);
    line << "fmm_balance_trace call=" << call
         << " solve=" << stats.diagSolveIndex
         << " splitter_solve=" << stats.diagSplitterSolve
         << " domain_lower=" << stats.diagDomainLower[0] << ','
         << stats.diagDomainLower[1] << ',' << stats.diagDomainLower[2]
         << " domain_upper=" << stats.diagDomainUpper[0] << ','
         << stats.diagDomainUpper[1] << ',' << stats.diagDomainUpper[2]
         << " owned_min=" << minimum(cOwned)
         << " owned_mean=" << mean(cOwned)
         << " owned_max=" << maximum(cOwned)
         << " owned_max_rank=" << big
         << " fresh_split_owned_min=" << stats.diagFreshOwnedCountMin
         << " fresh_split_owned_max=" << stats.diagFreshOwnedCountMax
         << " input_mean=" << mean(cInput)
         << " input_max=" << maximum(cInput)
         << " local_traversal_mean=" << mean(cLocalTraversal)
         << " local_traversal_max=" << maximum(cLocalTraversal)
         << " corr_lt_owned=" << correlation(cLocalTraversal, cOwned)
         << " corr_lt_local_pairs=" << correlation(cLocalTraversal, cLocalPairs)
         << " corr_lt_local_m2l=" << correlation(cLocalTraversal, cLocalM2L)
         << " corr_lt_leaf_sq=" << correlation(cLocalTraversal, cLeafSquared)
         << " corr_let_execute_let_pairs="
         << correlation(cLetExecute, cLetPairs)
         << " local_p2p_pairs_sum=" << sum(cLocalPairs)
         << " local_p2p_pairs_max=" << maximum(cLocalPairs)
         << " local_m2l_sum=" << sum(cLocalM2L)
         << " local_m2l_max=" << maximum(cLocalM2L)
         << " let_p2p_pairs_sum=" << sum(cLetPairs)
         << " let_p2p_pairs_max=" << maximum(cLetPairs)
         << " let_m2l_sum=" << sum(cLetM2L)
         << " let_m2l_max=" << maximum(cLetM2L)
         << " leaf_sq_sum=" << sum(cLeafSquared)
         << " leaf_sq_max=" << maximum(cLeafSquared)
         << " max_leaf_max=" << maximum(cMaxLeaf)
         << " max_depth_max=" << maximum(cMaxDepth)
         << " leaves_over_cap_sum=" << sum(cOverCapacity)
         << " leaves_over_split_sum=" << sum(cOverSplit)
         << " leaves_depth_cap_sum=" << sum(cDepthCap)
         << " root_half_mean=" << mean(cRootHalf)
         << " root_half_max=" << maximum(cRootHalf)
         << " diag_seconds_max=" << maximum(cDiagSeconds);
    describe(line, "slow", slow);
    if(big != slow)
        describe(line, "big", big);
    // Five slowest local traversals: rank:seconds:owned:local_pairs:local_m2l.
    std::vector<int> order(static_cast<std::size_t>(size));
    for(int r = 0; r < size; ++r)
        order[static_cast<std::size_t>(r)] = r;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return value(a, cLocalTraversal) > value(b, cLocalTraversal); });
    line << " top_local_traversal=";
    for(int i = 0; i < std::min(size, 5); ++i)
    {
        const int r = order[static_cast<std::size_t>(i)];
        line << (i == 0 ? "" : ";") << r << ':' << value(r, cLocalTraversal)
             << ':' << value(r, cOwned) << ':' << value(r, cLocalPairs) << ':'
             << value(r, cLocalM2L);
    }
    line << '\n';

    // Detail behind a flag: one row per rank on the first solve, every tenth
    // solve, every solve with a root change, and the solve after one.
    static std::uint64_t lastRootChangeCall = 0;
    const bool rootChange = stats.ranksWithRootGeometryChange != 0;
    const bool dumpSolve = call == 1 || call % 10 == 0 || rootChange ||
        (lastRootChangeCall != 0 && call == lastRootChangeCall + 1);
    if(rootChange)
        lastRootChangeCall = call;
    const char* const detail = std::getenv("RICH_FMM_TRACE_RANKS");
    if(dumpSolve && detail != nullptr && detail[0] != '\0' &&
       !(detail[0] == '0' && detail[1] == '\0'))
    {
        for(int r = 0; r < size; ++r)
        {
            line << "fmm_rank_trace call=" << call;
            describe(line, "r", r);
            line << '\n';
        }
    }
    return line.str();
#else
    (void) stats;
    (void) call;
    return std::string();
#endif
}

// individualPrepareSeconds / individualFinishSeconds: the target-evaluation
// caller's own work before and after the solve (negative: not such a solve).
void traceFmmSolve(const FmmSolveStats& stats, bool explicitlyEnabled,
                   double individualPrepareSeconds = -1.0,
                   double individualFinishSeconds = -1.0)
{
#ifdef RICH_MPI
    if(!explicitlyEnabled)
        return;
#else
    if(!explicitlyEnabled && !fmmTraceEnabled())
        return;
#endif

    static std::uint64_t call = 0;
    ++call;
    // The two redistribution parts come last, so the earlier fields keep their order.
    constexpr int PhaseCount = 24;
    const double localTimes[PhaseCount] = {
        stats.totalSeconds, stats.gravityRedistributionSeconds,
        stats.buildSeconds, stats.topologyRebuildSeconds,
        stats.rootDescriptorExchangeSeconds, stats.processTopologySeconds,
        stats.letPlanSeconds, stats.letBuildResetSeconds,
        stats.letDescriptorTraversalSeconds, stats.letFinalizeSeconds,
        stats.letSubscriptionSeconds, stats.letPruneCompactSeconds,
        stats.localTraversalSeconds, stats.letExecuteSeconds,
        stats.upwardSeconds, stats.processUpwardSeconds,
        stats.processInteractionSeconds, stats.processDownwardSeconds,
        stats.letExchangeSeconds, stats.letM2LSeconds,
        stats.letP2PSeconds, stats.downwardSeconds,
        stats.gravityRedistributionForwardSeconds, stats.gravityRedistributionReturnSeconds};
    double minimumTimes[PhaseCount] = {};
    double meanTimes[PhaseCount] = {};
    double maximumTimes[PhaseCount] = {};
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
    const unsigned long long localM2PCount =
        static_cast<unsigned long long>(stats.letM2PCount);
    unsigned long long globalM2PCount = localM2PCount;
    const double localM2PSeconds = stats.letM2PSeconds;
    double maximumM2PSeconds = localM2PSeconds;
    const unsigned long long localPeakRemoteBytes =
        static_cast<unsigned long long>(stats.peakRemoteBytes);
    unsigned long long maximumBytesOwned = localBytesOwned;
    unsigned long long maximumPeakRemoteBytes = localPeakRemoteBytes;
    const unsigned long long localCacheCounts[8] = {
        static_cast<unsigned long long>(stats.localOperatorCacheHits),
        static_cast<unsigned long long>(stats.localOperatorCacheMisses),
        static_cast<unsigned long long>(stats.localOperatorCacheBypasses),
        static_cast<unsigned long long>(stats.letOperatorCacheHits),
        static_cast<unsigned long long>(stats.letOperatorCacheMisses),
        static_cast<unsigned long long>(stats.letOperatorCacheBypasses),
        static_cast<unsigned long long>(stats.processOperatorCacheMisses),
        static_cast<unsigned long long>(stats.processOperatorCacheBypasses)};
    unsigned long long globalCacheCounts[8] = {};
    const unsigned long long localPatchSums[22] = {
        static_cast<unsigned long long>(stats.localPatchCount),
        static_cast<unsigned long long>(stats.reusedPatchCount),
        static_cast<unsigned long long>(stats.reusedLocalPatchPlanCount),
        static_cast<unsigned long long>(stats.rebuiltLocalPatchPlanCount),
        static_cast<unsigned long long>(stats.patchNodeGeometryExpansionCount),
        static_cast<unsigned long long>(stats.patchRetainedBytes),
        static_cast<unsigned long long>(stats.patchReleasedBytes),
        static_cast<unsigned long long>(stats.letTargetSubplansReused),
        static_cast<unsigned long long>(stats.letTargetSubplansRebuilt),
        static_cast<unsigned long long>(stats.letSourceTriggeredInvalidations),
        static_cast<unsigned long long>(stats.letWavePlanRebuildCount),
        static_cast<unsigned long long>(stats.letDescriptorTraversalSkippedCount),
        stats.letPayloadShapeTriggeredRebuild ? 1ull : 0ull,
        stats.letPayloadCapacityRefreshRequired ? 1ull : 0ull,
        stats.letPayloadLayoutRefreshed ? 1ull : 0ull,
        static_cast<unsigned long long>(stats.letPayloadCapacityUpdateCount),
        static_cast<unsigned long long>(stats.letPayloadSourceRepackCount),
        static_cast<unsigned long long>(stats.letSourceGenerationCheckCount),
        static_cast<unsigned long long>(stats.letChangedSourcePatchCount),
        static_cast<unsigned long long>(stats.letReverseDependencyLookupCount),
        static_cast<unsigned long long>(stats.letReverseDependencyTargetCount),
        static_cast<unsigned long long>(stats.letReverseDependencyEdgeCount)};
    unsigned long long globalPatchSums[22] = {};
    const unsigned long long localPatchMaxima[7] = {
        static_cast<unsigned long long>(stats.globalPatchCount),
        static_cast<unsigned long long>(stats.replicatedDescriptorBytes),
        static_cast<unsigned long long>(stats.processTreeBytes),
        static_cast<unsigned long long>(stats.processPlanBytes),
        static_cast<unsigned long long>(stats.processOwnedNodeCountMax),
        static_cast<unsigned long long>(stats.processOwnedM2LCountMax),
        static_cast<unsigned long long>(stats.letWaveCount)};
    unsigned long long globalPatchMaxima[7] = {};
    const double localProcessOwnerImbalance =
        stats.processOwnedNodeImbalance;
    double maximumProcessOwnerImbalance = localProcessOwnerImbalance;
    const double localResidualWait = stats.letResidualWaitSeconds;
    const double localPayloadLifetime = stats.letPayloadLifetimeSeconds;
    double maximumResidualWait = localResidualWait;
    double maximumPayloadLifetime = localPayloadLifetime;
    const double localDetailedExchangeTimes[6] = {
        stats.letPreparationSeconds, stats.letPayloadPlanningSeconds,
        stats.letPayloadPackingSeconds, stats.letValidationSeconds,
        stats.letDecodeSeconds, stats.letPayloadReleaseSeconds};
    double maximumDetailedExchangeTimes[6] = {};
    const unsigned long long localExchangeBytes[3] = {
        static_cast<unsigned long long>(stats.letMaxOutgoingBytes),
        static_cast<unsigned long long>(stats.letMaxIncomingBytes),
        static_cast<unsigned long long>(stats.letMaxSendCapacityBytes)};
    unsigned long long maximumExchangeBytes[3] = {};
    const unsigned long long localPruneCounts[10] = {
        stats.targetPruneActive ? 1ull : 0ull,
        static_cast<unsigned long long>(stats.targetParticleCount),
        static_cast<unsigned long long>(stats.targetNodeCount),
        static_cast<unsigned long long>(stats.localTargetPrunedM2LCount),
        static_cast<unsigned long long>(stats.localTargetPrunedP2PBlockCount),
        static_cast<unsigned long long>(stats.letTargetPrunedM2LCount),
        static_cast<unsigned long long>(stats.letTargetPrunedP2PBlockCount),
        static_cast<unsigned long long>(stats.letTargetPrunedM2PCount),
        static_cast<unsigned long long>(stats.downwardTargetPrunedLeafCount),
        static_cast<unsigned long long>(stats.gravityReturnedResultCount)};
    unsigned long long globalPruneCounts[10] = {};
    const double localIndividualTimes[3] = {
        stats.targetMaskSeconds, individualPrepareSeconds,
        individualFinishSeconds};
    double maximumIndividualTimes[3] = {};
    int rank = 0;
    // Print-only load-balance record; collective, so gathered on every rank.
    const std::string balanceLine = stats.diagFilled ?
        traceFmmBalance(stats, call) : std::string();
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Reduce(localTimes, minimumTimes, PhaseCount, MPI_DOUBLE, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(localTimes, meanTimes, PhaseCount, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(localTimes, maximumTimes, PhaseCount, MPI_DOUBLE, MPI_MAX, 0,
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
    MPI_Reduce(&localM2PCount, &globalM2PCount, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localM2PSeconds, &maximumM2PSeconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&localPeakRemoteBytes, &maximumPeakRemoteBytes, 1,
               MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(localCacheCounts, globalCacheCounts, 8,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(localPatchSums, globalPatchSums, 22,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(localPatchMaxima, globalPatchMaxima, 7,
               MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localProcessOwnerImbalance, &maximumProcessOwnerImbalance, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localResidualWait, &maximumResidualWait, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localPayloadLifetime, &maximumPayloadLifetime, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(localDetailedExchangeTimes, maximumDetailedExchangeTimes, 6,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(localExchangeBytes, maximumExchangeBytes, 3,
               MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(localPruneCounts, globalPruneCounts, 10,
               MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(localIndividualTimes, maximumIndividualTimes, 3, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);
    for(int i = 0; i < PhaseCount; ++i)
        meanTimes[i] /= static_cast<double>(stats.mpiRankCount);
#else
    for(int i = 0; i < PhaseCount; ++i) {
        minimumTimes[i] = localTimes[i];
        meanTimes[i] = localTimes[i];
        maximumTimes[i] = localTimes[i];
    }
    for(int i = 0; i < 8; ++i)
        globalCacheCounts[i] = localCacheCounts[i];
    for(int i = 0; i < 22; ++i)
        globalPatchSums[i] = localPatchSums[i];
    for(int i = 0; i < 7; ++i)
        globalPatchMaxima[i] = localPatchMaxima[i];
    globalM2PCount = localM2PCount;
    maximumM2PSeconds = localM2PSeconds;
    for(int i = 0; i < 6; ++i)
        maximumDetailedExchangeTimes[i] = localDetailedExchangeTimes[i];
    for(int i = 0; i < 3; ++i)
        maximumExchangeBytes[i] = localExchangeBytes[i];
    for(int i = 0; i < 10; ++i)
        globalPruneCounts[i] = localPruneCounts[i];
    for(int i = 0; i < 3; ++i)
        maximumIndividualTimes[i] = localIndividualTimes[i];
#endif
    // RICH_D2_TIMERS: forward-redistribution stages of this solve (rank maximum, mean, rank of the maximum); with
    // RICH_FMM_FORWARD_DEDUP the solves with unordered destinations summed over ranks. Both flags are the same on every
    // rank, so these reductions run on all ranks or on none.
    constexpr int ForwardStageCount = 6;
    const double LocalForwardStages[ForwardStageCount] = {stats.D2ForwardKeysSeconds, stats.D2ForwardSortSeconds,
        stats.D2ForwardDecideSeconds, stats.D2ForwardPackSeconds, stats.D2ForwardWireSeconds,
        stats.D2ForwardUnpackSeconds};
    struct ForwardStageRank {
        double Value;
        int Rank;
    };
    ForwardStageRank LocalForwardMax[ForwardStageCount];
    ForwardStageRank ForwardMax[ForwardStageCount];
    double ForwardMean[ForwardStageCount] = {};
    for(int Stage = 0; Stage < ForwardStageCount; ++Stage) {
        LocalForwardMax[Stage] = {LocalForwardStages[Stage], rank};
        ForwardMax[Stage] = LocalForwardMax[Stage];
        ForwardMean[Stage] = LocalForwardStages[Stage];
    }
    unsigned long long ForwardUnordered = static_cast<unsigned long long>(stats.ForwardDedupUnorderedSolves);
#ifdef RICH_MPI
    if(stats.D2ForwardTimersActive) {
        MPI_Reduce(LocalForwardMax, ForwardMax, ForwardStageCount, MPI_DOUBLE_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
        MPI_Reduce(LocalForwardStages, ForwardMean, ForwardStageCount, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if(stats.ForwardDedupMode != 0) {
        unsigned long long const LocalUnordered = ForwardUnordered;
        MPI_Reduce(&LocalUnordered, &ForwardUnordered, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
#endif
    for(int Stage = 0; Stage < ForwardStageCount; ++Stage)
        ForwardMean[Stage] /= static_cast<double>(stats.mpiRankCount);
    if(rank != 0)
        return;

    std::ostringstream line;
    line.setf(std::ios::scientific);
    line.precision(8);
    const char* const phaseNames[PhaseCount] = {
        "total", "redistribution", "build", "topology", "descriptor", "process_topology",
        "let_plan", "let_reset", "let_descriptor_traversal",
        "let_finalize", "let_subscription", "let_prune_compact",
        "local_traversal", "let_execute", "upward", "process_upward",
        "process_interaction", "process_downward", "let_exchange",
        "let_m2l", "let_p2p", "downward", "redistribution_forward", "redistribution_return"};
    line << "fmm_solve_trace call=" << call;
    for(int i = 0; i < PhaseCount; ++i)
        line << ' ' << phaseNames[i] << "_min=" << minimumTimes[i]
             << ' ' << phaseNames[i] << "_mean=" << meanTimes[i]
             << ' ' << phaseNames[i] << "_max=" << maximumTimes[i];
    line << " let_residual_wait_max=" << maximumResidualWait
         << " let_payload_lifetime_max=" << maximumPayloadLifetime
         << " let_preparation_max=" << maximumDetailedExchangeTimes[0]
         << " let_payload_planning_max=" << maximumDetailedExchangeTimes[1]
         << " let_payload_packing_max=" << maximumDetailedExchangeTimes[2]
         << " let_validation_max=" << maximumDetailedExchangeTimes[3]
         << " let_decode_max=" << maximumDetailedExchangeTimes[4]
         << " let_payload_release_max=" << maximumDetailedExchangeTimes[5]
         << " epoch=" << stats.topologyEpoch
         << " rebuilds=" << stats.topologyRebuildCount
         << " process_rebuilds=" << stats.processTopologyRebuildCount
         << " let_rebuilds=" << stats.letTopologyRebuildCount
         << " gravity_resplit=" << (stats.gravityResampleEnabled ? 1 : 0)
         << " gravity_resample_reason=" << stats.gravityResampleReason
         << " gravity_resample_count=" << stats.gravityResampleCount
         << " straggler_excess=" << stats.gravityStragglerExcessSeconds
         << " straggler_baseline=" << stats.gravityStragglerBaselineSeconds
         << " straggler_baseline_pending="
         << (stats.gravityStragglerBaselinePending ? 1 : 0)
         << " straggler_debt=" << stats.gravityStragglerDebtSeconds
         << " resample_threshold=" << stats.gravityResampleThresholdSeconds
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
         << " local_patch_count_sum=" << globalPatchSums[0]
         << " global_patch_count=" << globalPatchMaxima[0]
         << " reused_patch_count_sum=" << globalPatchSums[1]
         << " reused_patch_plan_count_sum=" << globalPatchSums[2]
         << " rebuilt_patch_plan_count_sum=" << globalPatchSums[3]
         << " patch_geometry_expansions_sum=" << globalPatchSums[4]
         << " patch_retained_bytes_sum=" << globalPatchSums[5]
         << " patch_released_bytes_sum=" << globalPatchSums[6]
         << " replicated_descriptor_bytes_max=" << globalPatchMaxima[1]
         << " process_tree_bytes_max=" << globalPatchMaxima[2]
         << " process_plan_bytes_max=" << globalPatchMaxima[3]
         << " process_owned_nodes_max=" << globalPatchMaxima[4]
         << " process_owned_m2l_max=" << globalPatchMaxima[5]
         << " process_owner_imbalance_max="
         << maximumProcessOwnerImbalance
         << " process_m2l_imbalance="
         << stats.processOwnedM2LImbalance
         << " process_global_owners="
         << (stats.processOwnershipGloballyBalanced ? 1 : 0)
         << " let_target_subplans_reused_sum=" << globalPatchSums[7]
         << " let_target_subplans_rebuilt_sum=" << globalPatchSums[8]
         << " let_source_invalidations_sum=" << globalPatchSums[9]
         << " let_wave_plan_rebuilds_sum=" << globalPatchSums[10]
         << " let_descriptor_traversal_skipped_sum=" << globalPatchSums[11]
         << " let_payload_shape_rebuild_ranks=" << globalPatchSums[12]
         << " let_payload_refresh_required_ranks=" << globalPatchSums[13]
         << " let_payload_layout_refreshed_ranks=" << globalPatchSums[14]
         << " let_particle_payload_compact="
         << (stats.letParticlePayloadCompacted ? 1 : 0)
         << " let_particle_payload_quantized="
         << (stats.letParticlePayloadQuantized ? 1 : 0)
         << " let_multipole_payload_compact="
         << (stats.letMultipolePayloadCompacted ? 1 : 0)
         << " let_payload_capacity_updates_sum=" << globalPatchSums[15]
         << " let_payload_sources_repacked_sum=" << globalPatchSums[16]
         << " let_source_generation_checks_sum=" << globalPatchSums[17]
         << " let_changed_source_patches_sum=" << globalPatchSums[18]
         << " let_reverse_dependency_lookups_sum=" << globalPatchSums[19]
         << " let_reverse_dependency_targets_sum=" << globalPatchSums[20]
         << " let_reverse_dependency_edges_sum=" << globalPatchSums[21]
         << " let_payload_layout_refresh_local="
         << stats.letPayloadLayoutRefreshSeconds
         << " let_invalidation_local="
         << stats.letInvalidationSeconds
         << " let_wave_count_max=" << globalPatchMaxima[6]
         << " bytes_owned_max=" << maximumBytesOwned
         << " peak_remote_bytes_max=" << maximumPeakRemoteBytes
         << " let_outgoing_bytes_max=" << maximumExchangeBytes[0]
         << " let_incoming_bytes_max=" << maximumExchangeBytes[1]
         << " let_send_capacity_bytes_max=" << maximumExchangeBytes[2]
         << " local_planned_m2l_sum=" << globalPlanCounts[0]
         << " local_planned_p2p_blocks_sum=" << globalPlanCounts[1]
         << " let_planned_m2l_sum=" << globalPlanCounts[2]
         << " let_planned_p2p_blocks_sum=" << globalPlanCounts[3]
         << " let_active_m2l_sum=" << globalActiveCounts[0]
         << " let_active_p2p_blocks_sum=" << globalActiveCounts[1]
         << " let_active_m2p_sum=" << globalM2PCount
         << " let_m2p_max=" << maximumM2PSeconds
         << " local_inactive_m2l_sum=" << globalInactiveCounts[0]
         << " local_inactive_p2p_blocks_sum=" << globalInactiveCounts[1]
         << " let_inactive_m2l_sum=" << globalInactiveCounts[2]
         << " let_inactive_p2p_blocks_sum=" << globalInactiveCounts[3]
         << " let_zero_multipole_payloads_sum=" << globalInactiveCounts[4]
         << " let_omitted_multipole_payloads_sum=" << globalInactiveCounts[5]
         << " let_omitted_particle_payloads_sum=" << globalInactiveCounts[6]
         << " bytes_sent_sum=" << globalInactiveCounts[7]
         << " bytes_received_sum=" << globalInactiveCounts[8]
         << " local_cache_hits_sum=" << globalCacheCounts[0]
         << " local_cache_misses_sum=" << globalCacheCounts[1]
         << " local_cache_bypasses_sum=" << globalCacheCounts[2]
         << " let_cache_hits_sum=" << globalCacheCounts[3]
         << " let_cache_misses_sum=" << globalCacheCounts[4]
         << " let_cache_bypasses_sum=" << globalCacheCounts[5]
         << " process_cache_misses_sum=" << globalCacheCounts[6]
         << " process_cache_bypasses_sum=" << globalCacheCounts[7]
         << " target_prune_ranks=" << globalPruneCounts[0]
         << " target_particles_sum=" << globalPruneCounts[1]
         << " target_nodes_sum=" << globalPruneCounts[2]
         << " local_target_pruned_m2l_sum=" << globalPruneCounts[3]
         << " local_target_pruned_p2p_blocks_sum=" << globalPruneCounts[4]
         << " let_target_pruned_m2l_sum=" << globalPruneCounts[5]
         << " let_target_pruned_p2p_blocks_sum=" << globalPruneCounts[6]
         << " let_target_pruned_m2p_sum=" << globalPruneCounts[7]
         << " downward_target_pruned_leaves_sum=" << globalPruneCounts[8]
         << " redistribution_returned_results_sum=" << globalPruneCounts[9]
         << " target_mask_max=" << maximumIndividualTimes[0]
         << " individual_prepare_max=" << maximumIndividualTimes[1]
         << " individual_finish_max=" << maximumIndividualTimes[2];
    if(stats.D2ForwardTimersActive) {
        // Per call site (0 full solve, 1 individual target evaluation): solves and running sums of the stage maxima
        // and means, so the summed pool of each site is read from its last line.
        static std::uint64_t SiteSolves[2] = {0, 0};
        static double SiteMaxSums[2][ForwardStageCount] = {};
        static double SiteMeanSums[2][ForwardStageCount] = {};
        const char* const ForwardStageNames[ForwardStageCount] = {"keys", "sort", "decide", "pack", "wire", "unpack"};
        const int Site = individualPrepareSeconds >= 0.0 ? 1 : 0;
        ++SiteSolves[Site];
        line << " d2_fwd_site=" << (Site == 1 ? "individual" : "full");
        for(int Stage = 0; Stage < ForwardStageCount; ++Stage) {
            SiteMaxSums[Site][Stage] += ForwardMax[Stage].Value;
            SiteMeanSums[Site][Stage] += ForwardMean[Stage];
            line << " d2_fwd_" << ForwardStageNames[Stage] << "_max=" << ForwardMax[Stage].Value << " d2_fwd_"
                 << ForwardStageNames[Stage] << "_mean=" << ForwardMean[Stage] << " d2_fwd_"
                 << ForwardStageNames[Stage] << "_max_rank=" << ForwardMax[Stage].Rank;
        }
        line << " d2_fwd_site_solves=" << SiteSolves[Site];
        for(int Stage = 0; Stage < ForwardStageCount; ++Stage)
            line << " d2_fwd_site_" << ForwardStageNames[Stage] << "_max_sum=" << SiteMaxSums[Site][Stage]
                 << " d2_fwd_site_" << ForwardStageNames[Stage] << "_mean_sum=" << SiteMeanSums[Site][Stage];
    }
    if(stats.ForwardDedupMode != 0)
        line << " fwd_dedup=" << stats.ForwardDedupMode << " fwd_dedup_verified_solves="
             << stats.ForwardDedupVerifiedSolves << " fwd_dedup_unordered_sum=" << ForwardUnordered;
    std::cout << line.str() << std::endl;
    if(!balanceLine.empty())
        std::cout << balanceLine << std::flush;
}
}

FastMultipoleAcceleration3D::FastMultipoleAcceleration3D(FmmGravityOptions options,
                                                         double G):
#ifdef RICH_MPI
    G_(validateDistributedGravityConstant(G)),
#else
    G_(G),
#endif
    traceEnabled_(false),
    calculator_(validateAccelerationOptions(options))
{
#ifdef RICH_MPI
    traceEnabled_ = calculator_.solveTraceRequested();
#else
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
    traceEnabled_(distributedOptions.emitSolveTrace),
    calculator_(validateAccelerationOptions(options), distributedOptions)
{
    traceEnabled_ = traceEnabled_ || calculator_.solveTraceRequested();
}
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
    traceFmmSolve(calculator_.stats(), traceEnabled_);

#ifdef RICH_MPI
    requireOnEveryRank(acc.size() == N,
        "FastMultipoleAcceleration3D: FMM returned the wrong cell count");
#else
    if(acc.size() != N)
        throw UniversalError(
            "FastMultipoleAcceleration3D: FMM returned the wrong cell count");
#endif

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

void FastMultipoleAcceleration3D::EvaluateIndividualTargets(
    std::pair<Vector3D, Vector3D> const& bounds,
    vector<Vector3D> const& source_points,
    vector<double> const& source_masses,
    vector<std::uint64_t> const& source_ids,
    vector<Vector3D> const& target_points,
    vector<ComputationalCell3D> const& /*target_cells*/,
    double /*time*/,
    vector<Vector3D>& acc) const
{
#ifdef RICH_MPI
    requireOnEveryRank(source_points.size() == source_masses.size() &&
        source_points.size() == source_ids.size(),
        "FastMultipoleAcceleration3D: individual source point/mass/ID count mismatch");
#else
    if(source_points.size() != source_masses.size() ||
       source_points.size() != source_ids.size())
        throw UniversalError(
            "FastMultipoleAcceleration3D: individual source point/mass/ID count mismatch");
#endif

    // The FMM backend evaluates its source set.  Individual gravity targets
    // are exact canonical source centroids, so recover their source slots and
    // extract only those accelerations after the collective solve.
    const auto prepareStart = std::chrono::steady_clock::now();
    bool valid_target_mapping = IndexSourcePositions(source_points, SourceSlots_);

    std::vector<std::size_t> target_source_indices(target_points.size(), 0);
    for(std::size_t target = 0; target < target_points.size(); ++target)
    {
        const std::size_t Source = FindSourcePosition(source_points, SourceSlots_, target_points[target]);
        if(Source == source_points.size())
            valid_target_mapping = false;
        else
            target_source_indices[target] = Source;
    }
#ifdef RICH_MPI
    requireOnEveryRank(valid_target_mapping,
        "FastMultipoleAcceleration3D: an individual target does not match a unique source centroid");
#else
    if(!valid_target_mapping)
        throw UniversalError(
            "FastMultipoleAcceleration3D: an individual target does not match a unique source centroid");
#endif

    // The solve reads the caller's source arrays in place (it keeps no
    // reference past the call); the points are copied only if acc is the same
    // vector, since the solve overwrites acc while it reads them.
    const bool AccIsSourcePoints = &acc == &source_points;
    if(AccIsSourcePoints)
        points_ = source_points;
    vector<Vector3D> const& SolvePoints = AccIsSourcePoints ? points_ : source_points;
#ifdef RICH_MPI
    // Only the target slots are read back, so the solve may skip work that
    // reaches no target (DistributedFmmGravityCalculator::solve).
    targetMask_.assign(SolvePoints.size(), 0u);
    for(std::size_t source : target_source_indices)
        targetMask_[source] = 1u;
    const double prepareSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prepareStart).count();
    calculator_.solve(SolvePoints, source_masses, source_ids, bounds.first, bounds.second, acc, nullptr, &targetMask_);
#else
    const double prepareSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prepareStart).count();
    calculator_.solve(SolvePoints, source_masses, bounds.first, bounds.second, acc);
#endif
    const auto finishStart = std::chrono::steady_clock::now();

#ifdef RICH_MPI
    requireOnEveryRank(acc.size() == SolvePoints.size(),
        "FastMultipoleAcceleration3D: individual FMM returned the wrong source count");
#else
    if(acc.size() != SolvePoints.size())
        throw UniversalError(
            "FastMultipoleAcceleration3D: individual FMM returned the wrong source count");
#endif

    bool finite_acceleration = true;
#ifndef RICH_MPI
    std::size_t first_invalid = 0;
#endif
    for(std::size_t source = 0; source < acc.size(); ++source)
    {
        acc[source] *= G_;
        if(finite_acceleration &&
           (!std::isfinite(acc[source].x) || !std::isfinite(acc[source].y) ||
            !std::isfinite(acc[source].z)))
        {
            finite_acceleration = false;
#ifndef RICH_MPI
            first_invalid = source;
#endif
        }
    }
#ifdef RICH_MPI
    requireOnEveryRank(finite_acceleration,
        "FastMultipoleAcceleration3D: non-finite individual acceleration after G scaling on an MPI rank");
#else
    if(!finite_acceleration)
    {
        UniversalError error(
            "FastMultipoleAcceleration3D: non-finite individual acceleration after G scaling");
        error.addEntry("source", first_invalid);
        throw error;
    }
#endif

    vector<Vector3D> target_accelerations;
    target_accelerations.reserve(target_source_indices.size());
    for(std::size_t source : target_source_indices)
        target_accelerations.push_back(acc[source]);
    acc.swap(target_accelerations);
    traceFmmSolve(calculator_.stats(), traceEnabled_, prepareSeconds,
                  std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - finishStart).count());
}

const FmmSolveStats& FastMultipoleAcceleration3D::getLastStats() const noexcept
{
    return calculator_.stats();
}
