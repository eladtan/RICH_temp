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
    std::uint64_t bytesSent = 0;
    std::uint64_t bytesReceived = 0;
    std::size_t peakRemoteBytes = 0;
    std::size_t peakProcessBytes = 0;
    double processUpwardSeconds = 0;
    double processInteractionSeconds = 0;
    double processDownwardSeconds = 0;
    double letPlanSeconds = 0;
    double letExchangeSeconds = 0;
};

#endif // FMM_DIAGNOSTICS_HPP
