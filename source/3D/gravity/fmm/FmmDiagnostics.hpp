#ifndef FMM_DIAGNOSTICS_HPP
#define FMM_DIAGNOSTICS_HPP

#include <cstddef>
#include <cstdint>

struct FmmSolveStats
{
    std::size_t particleCount = 0;
    std::size_t nodeCount = 0;
    std::size_t leafCount = 0;
    std::size_t maxDepth = 0;
    std::size_t maxLeafOccupancy = 0;
    double averageLeafOccupancy = 0;
    std::uint64_t m2lCount = 0;
    std::uint64_t p2pBlockCount = 0;
    std::uint64_t p2pPairCount = 0;
    std::uint64_t rejectedSameNode = 0;
    std::uint64_t rejectedOverlap = 0;
    std::uint64_t rejectedRatio = 0;
    std::size_t maxTraversalStack = 0;
    double buildSeconds = 0;
    double upwardSeconds = 0;
    double interactionSeconds = 0;
    double downwardSeconds = 0;
    double totalSeconds = 0;
    std::size_t bytesOwned = 0;
    double totalMass = 0;
    double rootMass = 0;
    double maxRadiusRatio = 0;

    // Distributed fields are zero in serial builds.
    std::size_t mpiRankCount = 1;
    std::size_t activeRankCount = 1;
    std::size_t processNodeCount = 0;
    std::uint64_t processM2LCount = 0;
    std::uint64_t letM2LCount = 0;
    std::uint64_t letP2PBlockCount = 0;
    std::uint64_t letP2PPairCount = 0;
    std::uint64_t topologyEpoch = 0;
    std::uint64_t topologyRebuildCount = 0;
    std::uint64_t processTopologyRebuildCount = 0;
    std::uint64_t letTopologyRebuildCount = 0;
    std::size_t ranksWithRootGeometryChange = 0;
    std::size_t ranksWithLeafTopologyChange = 0;
    bool localRootGeometryChanged = false;
    bool localLeafTopologyChanged = false;
    bool processTopologyRebuilt = false;
    bool letTopologyRebuilt = false;
    bool topologyRebuildForced = false;
    bool processCommunicatorsReused = false;
    bool letCommunicatorReused = false;
    std::uint64_t bytesSent = 0;
    std::uint64_t bytesReceived = 0;
    std::size_t peakRemoteBytes = 0;
    std::size_t peakProcessBytes = 0;

    // Topology construction is split into a rank-root process topology and a
    // local-leaf-dependent LET topology.  These timings expose the cost paid
    // when a moving mesh changes only the latter.
    double topologyRebuildSeconds = 0;
    double rootDescriptorExchangeSeconds = 0;
    double processTopologySeconds = 0;

    // Persistent distributed-memory attribution.  bytesOwned remains the
    // aggregate value; these fields expose the dominant components.
    std::size_t localTreeBytes = 0;
    std::size_t localMultipoleBytes = 0;
    std::size_t localLocalBytes = 0;
    std::size_t letPlanBytes = 0;
    std::size_t localInteractionPlanBytes = 0;
    bool localInteractionPlanReused = false;
    std::size_t operatorCacheBytes = 0;
    std::size_t operatorCacheBudgetBytes = 0;
    std::size_t operatorCacheEntries = 0;
    std::size_t operatorCacheMaxEntries = 0;
    std::size_t operatorCacheBytesAtSolveStart = 0;
    std::size_t operatorCacheEntriesAtSolveStart = 0;

    // The distributed solver owns one persistent cache shared by LET and local
    // traversal.  The phase byte/entry fields below are snapshots taken after
    // each phase; hit/miss/bypass counters are phase local.  The final shared
    // cache is reported by operatorCacheBytes/operatorCacheEntries.  The
    // process-tree M2L phase still uses reusable scratch only and therefore
    // reports bypasses but no cache bytes.
    std::size_t localOperatorCacheBytes = 0;
    std::size_t localOperatorCacheEntries = 0;
    std::size_t localOperatorCacheMaxEntries = 0;
    std::uint64_t localOperatorCacheHits = 0;
    std::uint64_t localOperatorCacheMisses = 0;
    std::uint64_t localOperatorCacheBypasses = 0;
    std::uint64_t localOperatorIntegerKeyHits = 0;
    std::uint64_t localOperatorIntegerKeyMisses = 0;
    std::size_t letOperatorCacheBytes = 0;
    std::size_t letOperatorCacheEntries = 0;
    std::size_t letOperatorCacheMaxEntries = 0;
    std::uint64_t letOperatorCacheHits = 0;
    std::uint64_t letOperatorCacheMisses = 0;
    std::uint64_t letOperatorCacheBypasses = 0;
    std::uint64_t letOperatorIntegerKeyHits = 0;
    std::uint64_t letOperatorIntegerKeyMisses = 0;
    std::uint64_t processOperatorCacheMisses = 0;
    std::uint64_t processOperatorCacheBypasses = 0;

    double processUpwardSeconds = 0;
    double processInteractionSeconds = 0;
    double processDownwardSeconds = 0;
    double letPlanSeconds = 0;
    double letExecuteSeconds = 0;
    double letExchangeSeconds = 0;

    // Detailed LET exchange instrumentation.  letPayloadLifetimeSeconds is an
    // overlapped interval and must not be added to critical-path phase totals.
    // letExchangeSeconds remains the backward-compatible exposed total:
    // preparation + residual wait + validation + decode.
    double letPreparationSeconds = 0;
    double letPayloadPlanningSeconds = 0;
    double letPayloadPackingSeconds = 0;
    double letPayloadFlattenSeconds = 0;
    double letCountExchangeSeconds = 0;
    double letReceiveSetupSeconds = 0;
    double letPayloadLaunchSeconds = 0;
    double letPayloadReleaseSeconds = 0;
    double letPayloadLifetimeSeconds = 0;
    double letResidualWaitSeconds = 0;
    double letValidationSeconds = 0;
    double letDecodeSeconds = 0;
    std::uint64_t letProgressCallCount = 0;
    std::uint64_t letProgressIncompleteCount = 0;
    std::uint64_t letCompletionProgressCall = 0;
    std::uint64_t letCompletedBeforeFinishCount = 0;

    double letM2LSeconds = 0;
    double letP2PSeconds = 0;
    double localTraversalSeconds = 0;
};

#endif // FMM_DIAGNOSTICS_HPP
