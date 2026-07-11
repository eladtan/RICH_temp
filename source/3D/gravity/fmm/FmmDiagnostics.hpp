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
};

#endif // FMM_DIAGNOSTICS_HPP
