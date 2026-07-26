#include "3D/gravity/fmm/mpi/FmmProcessTree.hpp"

#ifdef RICH_MPI

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <limits>
#include <set>
#include <unordered_map>

#include "3D/gravity/fmm/mpi/FmmGlobalDyadicLattice.hpp"
#include "misc/universal_error.hpp"

namespace
{
bool finiteVector(const Vector3D& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

std::uint64_t doubleBits(double value)
{
    std::uint64_t result = 0;
    std::memcpy(&result, &value, sizeof(double));
    return result;
}

bool finiteCube(const Vector3D& center, double halfSize)
{
    return finiteVector(center) && halfSize > 0.0 && std::isfinite(halfSize) &&
        std::isfinite(center.x - halfSize) &&
        std::isfinite(center.x + halfSize) &&
        std::isfinite(center.y - halfSize) &&
        std::isfinite(center.y + halfSize) &&
        std::isfinite(center.z - halfSize) &&
        std::isfinite(center.z + halfSize) &&
        halfSize <= DBL_MAX / std::sqrt(3.0);
}

std::size_t saturatingAdd(std::size_t first, std::size_t second)
{
    return second > std::numeric_limits<std::size_t>::max() - first ?
        std::numeric_limits<std::size_t>::max() : first + second;
}

std::size_t saturatingMultiply(std::size_t first, std::size_t second)
{
    return first != 0 && second > std::numeric_limits<std::size_t>::max() / first ?
        std::numeric_limits<std::size_t>::max() : first * second;
}

double centerDistance(const FmmProcessNode& first,
                      const FmmProcessNode& second)
{
    const Vector3D delta = first.center - second.center;
    return std::sqrt(delta.x * delta.x + delta.y * delta.y +
                     delta.z * delta.z);
}
}

void FmmProcessTree::build(const std::vector<FmmPatchRootDescriptor>& descriptors)
{
    descriptorsByIndex_ = descriptors;
    activeRanks_.clear();
    activeDescriptorIndices_.clear();
    nodes_.clear();
    levels_.clear();
    leafByPatch_.clear();
    compatLeafByRank_.clear();

    bool haveEpoch = false;
    std::uint64_t epoch = 0;
    std::set<int> activeRankSet;
    FmmPatchKey previousKey;
    bool havePreviousKey = false;
    for(std::size_t i = 0; i < descriptorsByIndex_.size(); ++i)
    {
        const FmmPatchRootDescriptor& descriptor = descriptorsByIndex_[i];
        if(descriptor.magic != FMM_MPI_PACKET_MAGIC ||
           descriptor.version != FMM_MPI_PACKET_VERSION)
            throw UniversalError("FmmProcessTree::build: incompatible root descriptor protocol");
        if(!haveEpoch)
        {
            epoch = descriptor.epoch;
            haveEpoch = true;
        }
        else if(descriptor.epoch != epoch)
        {
            throw UniversalError("FmmProcessTree::build: inconsistent topology epochs");
        }

        if(descriptor.active == 0)
        {
            if(descriptor.particleCount != 0 || descriptor.ownerRank < 0)
                throw UniversalError(
                    "FmmProcessTree::build: invalid inactive patch descriptor");
            continue;
        }
        if(descriptor.active != 1 || descriptor.ownerRank < 0 ||
           descriptor.particleCount == 0 ||
           !FmmGlobalDyadicLattice::isValidPatchId(descriptor.patchId) ||
           !finiteCube(descriptor.centerVector(), descriptor.halfSize))
            throw UniversalError(
                "FmmProcessTree::build: invalid active patch descriptor");
        if((descriptor.rootLeaf != 0 && descriptor.rootLeaf != 1) ||
           descriptor.childMask < 0 || descriptor.childMask > 255 ||
           (descriptor.rootLeaf != 0 && descriptor.childMask != 0) ||
           (descriptor.rootLeaf == 0 && descriptor.childMask == 0))
            throw UniversalError("FmmProcessTree::build: inconsistent root topology descriptor");
        const double cubeRadius = std::sqrt(3.0) * descriptor.halfSize;
        const double radiusTolerance =
            64.0 * std::numeric_limits<double>::epsilon() *
            std::max(1.0, cubeRadius);
        if(!std::isfinite(descriptor.radius) || descriptor.radius < 0.0 ||
           descriptor.radius > cubeRadius + radiusTolerance)
            throw UniversalError(
                "FmmProcessTree::build: invalid tight patch radius");

        const FmmPatchKey patchKey{
            descriptor.ownerRank, descriptor.patchId};
        if(havePreviousKey && !(previousKey < patchKey))
            throw UniversalError(
                "FmmProcessTree::build: descriptors must be sorted and unique");
        previousKey = patchKey;
        havePreviousKey = true;
        activeDescriptorIndices_.push_back(i);
        activeRankSet.insert(descriptor.ownerRank);
    }
    activeRanks_.assign(activeRankSet.begin(), activeRankSet.end());
    if(activeDescriptorIndices_.empty())
    {
        topologyHash_ = 0;
        std::vector<FmmPatchRootDescriptor>().swap(descriptorsByIndex_);
        return;
    }

    buildRange(0, activeDescriptorIndices_.size(), 0);
    buildLevels();
    computeHash();

    std::unordered_map<int, std::size_t> patchCountByRank;
    for(const auto& leaf : leafByPatch_)
        ++patchCountByRank[leaf.first.ownerRank];
    for(const auto& leaf : leafByPatch_)
    {
        if(patchCountByRank[leaf.first.ownerRank] == 1 &&
           leaf.first.patchId == FMM_COMPAT_PATCH_ID)
            compatLeafByRank_[leaf.first.ownerRank] = leaf.second;
    }
    std::vector<FmmPatchRootDescriptor>().swap(descriptorsByIndex_);
}

std::size_t FmmProcessTree::buildRange(std::size_t begin,
                                       std::size_t end,
                                       std::size_t depth)
{
    if(begin >= end)
        throw UniversalError("FmmProcessTree::buildRange: empty range");

    Vector3D lower;
    Vector3D upper;
    bool first = true;
    for(std::size_t i = begin; i < end; ++i)
    {
        const FmmPatchRootDescriptor& descriptor =
            descriptorsByIndex_[activeDescriptorIndices_[i]];
        const Vector3D center = descriptor.centerVector();
        const Vector3D lo(center.x - descriptor.halfSize,
                          center.y - descriptor.halfSize,
                          center.z - descriptor.halfSize);
        const Vector3D hi(center.x + descriptor.halfSize,
                          center.y + descriptor.halfSize,
                          center.z + descriptor.halfSize);
        if(first)
        {
            lower = lo;
            upper = hi;
            first = false;
        }
        else
        {
            lower.x = std::min(lower.x, lo.x);
            lower.y = std::min(lower.y, lo.y);
            lower.z = std::min(lower.z, lo.z);
            upper.x = std::max(upper.x, hi.x);
            upper.y = std::max(upper.y, hi.y);
            upper.z = std::max(upper.z, hi.z);
        }
    }

    if(end - begin > 1)
    {
        const Vector3D extent = upper - lower;
        const int axis = extent.x >= extent.y && extent.x >= extent.z ? 0 :
            (extent.y >= extent.z ? 1 : 2);
        std::stable_sort(activeDescriptorIndices_.begin() +
                             static_cast<std::ptrdiff_t>(begin),
                         activeDescriptorIndices_.begin() +
                             static_cast<std::ptrdiff_t>(end),
            [&](std::size_t lhs, std::size_t rhs)
            {
                const Vector3D a =
                    descriptorsByIndex_[lhs].centerVector();
                const Vector3D b =
                    descriptorsByIndex_[rhs].centerVector();
                const double av = axis == 0 ? a.x : (axis == 1 ? a.y : a.z);
                const double bv = axis == 0 ? b.x : (axis == 1 ? b.y : b.z);
                const int ownerA = descriptorsByIndex_[lhs].ownerRank;
                const int ownerB = descriptorsByIndex_[rhs].ownerRank;
                return av < bv || (av == bv && ownerA < ownerB) ||
                       (av == bv && ownerA == ownerB &&
                        descriptorsByIndex_[lhs].patchId <
                            descriptorsByIndex_[rhs].patchId);
            });
    }

    const Vector3D extent = upper - lower;
    if(!finiteVector(extent) || extent.x < 0.0 || extent.y < 0.0 || extent.z < 0.0)
        throw UniversalError("FmmProcessTree::buildRange: process extent overflow");
    const Vector3D center(lower.x + 0.5 * extent.x,
                          lower.y + 0.5 * extent.y,
                          lower.z + 0.5 * extent.z);
    const double halfSize = 0.5 *
        std::max(extent.x, std::max(extent.y, extent.z));
    if(!finiteCube(center, halfSize))
        throw UniversalError("FmmProcessTree::buildRange: invalid process-node cube");

    const std::size_t nodeIndex = nodes_.size();
    nodes_.push_back(FmmProcessNode());
    FmmProcessNode& node = nodes_.back();
    node.begin = begin;
    node.end = end;
    node.depth = depth;
    node.center = center;
    node.halfSize = halfSize;
    node.radius = std::sqrt(3.0) * node.halfSize;

    if(end - begin == 1)
    {
        const std::size_t descriptorIndex = activeDescriptorIndices_[begin];
        const FmmPatchRootDescriptor& descriptor =
            descriptorsByIndex_[descriptorIndex];
        node.leafOwnerRank = descriptor.ownerRank;
        node.leafPatchId = descriptor.patchId;
        node.leafDescriptorIndex = descriptorIndex;
        node.owner = node.leafOwnerRank;
        node.center = descriptor.centerVector();
        node.halfSize = descriptor.halfSize;
        if(!finiteCube(node.center, node.halfSize))
            throw UniversalError("FmmProcessTree::buildRange: invalid process leaf cube");
        const double cubeRadius = std::sqrt(3.0) * node.halfSize;
        const double radiusTolerance =
            32.0 * std::numeric_limits<double>::epsilon() *
            std::max(1.0, cubeRadius);
        if(!std::isfinite(descriptor.radius) || descriptor.radius < 0.0 ||
           descriptor.radius > cubeRadius + radiusTolerance)
            throw UniversalError("FmmProcessTree::buildRange: invalid tight patch radius");
        node.radius = descriptor.radius;
        const FmmPatchKey patchKey = node.leafKey();
        if(!patchKey.valid())
            throw UniversalError("FmmProcessTree::buildRange: invalid leaf patch key");
        if(!leafByPatch_.emplace(patchKey, nodeIndex).second)
            throw UniversalError("FmmProcessTree::buildRange: duplicate patch leaf");
        return nodeIndex;
    }

    const std::size_t mid = begin + (end - begin) / 2;
    const int owner = descriptorsByIndex_[activeDescriptorIndices_[mid]].ownerRank;
    nodes_[nodeIndex].owner = owner;
    const std::size_t left = buildRange(begin, mid, depth + 1);
    const std::size_t right = buildRange(mid, end, depth + 1);
    nodes_[nodeIndex].left = left;
    nodes_[nodeIndex].right = right;
    nodes_[left].parent = nodeIndex;
    nodes_[right].parent = nodeIndex;

    FmmProcessNode& completed = nodes_[nodeIndex];
    completed.radius = std::max(
        centerDistance(completed, nodes_[left]) + nodes_[left].radius,
        centerDistance(completed, nodes_[right]) + nodes_[right].radius);
    if(!std::isfinite(completed.radius) || completed.radius < 0.0)
        throw UniversalError(
            "FmmProcessTree::buildRange: invalid internal tight radius");
    return nodeIndex;
}

void FmmProcessTree::buildLevels()
{
    std::size_t maximum = 0;
    for(const FmmProcessNode& node : nodes_)
        maximum = std::max(maximum, node.depth);
    levels_.assign(maximum + 1, std::vector<std::size_t>());
    for(std::size_t i = 0; i < nodes_.size(); ++i)
        levels_[nodes_[i].depth].push_back(i);
}

void FmmProcessTree::computeHash()
{
    std::uint64_t hash = 1469598103934665603ull;
    const std::uint64_t prime = 1099511628211ull;
    for(const FmmProcessNode& node : nodes_)
    {
        const std::uint64_t fields[] = {
            static_cast<std::uint64_t>(node.owner + 1),
            static_cast<std::uint64_t>(node.leafOwnerRank + 1),
            node.leafPatchId,
            static_cast<std::uint64_t>(node.begin),
            static_cast<std::uint64_t>(node.end),
            static_cast<std::uint64_t>(node.depth),
            doubleBits(node.center.x), doubleBits(node.center.y),
            doubleBits(node.center.z), doubleBits(node.halfSize),
            doubleBits(node.radius)};
        for(std::uint64_t value : fields)
        {
            for(int byte = 0; byte < 8; ++byte)
            {
                hash ^= (value >> (8 * byte)) & 0xffu;
                hash *= prime;
            }
        }
    }
    topologyHash_ = hash;
}

std::size_t FmmProcessTree::bytesOwned() const
{
    std::size_t result = 0;
    result = saturatingAdd(result, saturatingMultiply(
        descriptorsByIndex_.capacity(), sizeof(FmmPatchRootDescriptor)));
    result = saturatingAdd(result, saturatingMultiply(
        activeRanks_.capacity(), sizeof(int)));
    result = saturatingAdd(result, saturatingMultiply(
        activeDescriptorIndices_.capacity(), sizeof(std::size_t)));
    result = saturatingAdd(result, saturatingMultiply(
        nodes_.capacity(), sizeof(FmmProcessNode)));
    result = saturatingAdd(result, saturatingMultiply(
        levels_.capacity(), sizeof(std::vector<std::size_t>)));
    for(const std::vector<std::size_t>& level : levels_)
        result = saturatingAdd(result, saturatingMultiply(
            level.capacity(), sizeof(std::size_t)));
    const std::size_t mapEntryBytes =
        sizeof(std::pair<const FmmPatchKey, std::size_t>) + 2 * sizeof(void*);
    result = saturatingAdd(result, saturatingMultiply(
        leafByPatch_.size(), mapEntryBytes));
    const std::size_t rankMapEntryBytes =
        sizeof(std::pair<const int, std::size_t>) + 2 * sizeof(void*);
    result = saturatingAdd(result, saturatingMultiply(
        compatLeafByRank_.size(), rankMapEntryBytes));
    return result;
}

std::size_t FmmProcessTree::leafForPatch(const FmmPatchKey& patch) const
{
    const auto it = leafByPatch_.find(patch);
    return it == leafByPatch_.end() ? invalidIndex() : it->second;
}

std::size_t FmmProcessTree::leafForRank(int rank) const
{
    const auto found = compatLeafByRank_.find(rank);
    return found == compatLeafByRank_.end() ? invalidIndex() : found->second;
}

#endif // RICH_MPI
