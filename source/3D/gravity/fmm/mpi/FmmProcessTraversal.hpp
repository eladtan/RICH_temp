#ifndef FMM_PROCESS_TRAVERSAL_HPP
#define FMM_PROCESS_TRAVERSAL_HPP

#ifdef RICH_MPI

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <mpi.h>

#include "3D/gravity/fmm/mpi/FmmProcessTree.hpp"

struct FmmProcessM2LPair
{
    std::size_t targetNode = 0;
    std::size_t sourceNode = 0;
};

struct FmmPatchPair
{
    FmmPatchKey target;
    FmmPatchKey source;
};

inline bool operator==(const FmmPatchPair& first, const FmmPatchPair& second)
{
    return first.target == second.target && first.source == second.source;
}

inline bool operator<(const FmmPatchPair& first, const FmmPatchPair& second)
{
    return std::tie(first.target, first.source) <
           std::tie(second.target, second.source);
}

struct FmmProcessPairPlan
{
    std::vector<FmmProcessM2LPair> localM2LPairs;
    std::unordered_map<int, std::vector<std::size_t>> processSendNodesByRank;
    std::vector<FmmPatchKey> localSelfPatches;
    std::vector<FmmPatchPair> localCrossPatchPairs;
    std::vector<FmmPatchPair> remoteLetPairs;
    std::vector<int> letSourceRanks;
    std::vector<int> letTargetRanks;
    std::uint64_t acceptedPairCount = 0;
    std::uint64_t localSelfRankCount = 0;
    std::uint64_t localCrossPatchPairCount = 0;
    std::uint64_t letRankPairCount = 0;

    std::size_t bytesOwned() const;
};

class FmmProcessTraversal
{
public:
    static FmmProcessPairPlan build(const FmmProcessTree& tree,
                                    double thetaCritical,
                                    std::uint64_t topologyEpoch,
                                    int rank,
                                    const MPI_Comm& comm);
};

#endif // RICH_MPI

#endif // FMM_PROCESS_TRAVERSAL_HPP
