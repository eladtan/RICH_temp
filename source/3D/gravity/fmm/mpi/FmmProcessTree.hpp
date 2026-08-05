#ifndef FMM_PROCESS_TREE_HPP
#define FMM_PROCESS_TREE_HPP

#ifdef RICH_MPI

#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "3D/elementary/Vector3D.hpp"
#include "3D/gravity/fmm/mpi/FmmPackets.hpp"
#include "3D/gravity/fmm/mpi/FmmPatchKey.hpp"

struct FmmProcessNode
{
    Vector3D center;
    double halfSize = 0.0;
    double radius = 0.0;
    std::size_t left = std::numeric_limits<std::size_t>::max();
    std::size_t right = std::numeric_limits<std::size_t>::max();
    std::size_t parent = std::numeric_limits<std::size_t>::max();
    std::size_t depth = 0;
    std::size_t begin = 0;
    std::size_t end = 0;
    int owner = -1;
    int leafOwnerRank = -1;
    std::uint64_t leafPatchId = 0;
    std::size_t leafDescriptorIndex = std::numeric_limits<std::size_t>::max();

    bool isLeaf() const
    {
        return leafDescriptorIndex != std::numeric_limits<std::size_t>::max();
    }

    FmmPatchKey leafKey() const
    {
        FmmPatchKey key;
        key.ownerRank = leafOwnerRank;
        key.patchId = leafPatchId;
        return key;
    }
};

class FmmProcessTree
{
public:
    void build(const std::vector<FmmPatchRootDescriptor>& descriptors,
               bool balanceInternalOwners = false,
               bool allowAnyActiveOwner = false);

    const std::vector<FmmProcessNode>& nodes() const { return nodes_; }
    const std::vector<int>& activeRanks() const { return activeRanks_; }
    const std::vector<std::size_t>& activeDescriptorIndices() const
    {
        return activeDescriptorIndices_;
    }
    const std::vector<std::vector<std::size_t>>& levels() const { return levels_; }
    std::size_t root() const { return nodes_.empty() ? invalidIndex() : 0; }
    std::size_t leafForPatch(const FmmPatchKey& patch) const;
    std::size_t leafForRank(int rank) const;
    std::size_t maxDepth() const { return levels_.empty() ? 0 : levels_.size() - 1; }
    std::uint64_t topologyHash() const { return topologyHash_; }
    std::size_t bytesOwned() const;

    static std::size_t invalidIndex() { return std::numeric_limits<std::size_t>::max(); }

private:
    std::size_t buildRange(std::size_t begin, std::size_t end, std::size_t depth);
    void addOwnerWork(int owner);
    void buildLevels();
    void computeHash();

    std::vector<FmmPatchRootDescriptor> descriptorsByIndex_;
    std::vector<int> activeRanks_;
    std::vector<std::size_t> activeDescriptorIndices_;
    std::vector<FmmProcessNode> nodes_;
    std::vector<std::vector<std::size_t>> levels_;
    std::unordered_map<FmmPatchKey, std::size_t, FmmPatchKeyHash> leafByPatch_;
    std::unordered_map<int, std::size_t> compatLeafByRank_;
    std::unordered_map<int, std::size_t> ownerWork_;
    std::set<std::pair<std::size_t, int>> ownerQueue_;
    bool balanceInternalOwners_ = false;
    bool allowAnyActiveOwner_ = false;
    std::uint64_t topologyHash_ = 0;
};

#endif // RICH_MPI

#endif // FMM_PROCESS_TREE_HPP
