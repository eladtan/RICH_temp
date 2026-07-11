#include "3D/gravity/fmm/mpi/FmmProcessTree.hpp"

#ifdef RICH_MPI

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

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
        halfSize <= std::numeric_limits<double>::max() / std::sqrt(3.0);
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
}

void FmmProcessTree::build(const std::vector<FmmRankRootDescriptor>& descriptors)
{
    descriptorsByRank_ = descriptors;
    activeRanks_.clear();
    nodes_.clear();
    levels_.clear();
    leafByRank_.clear();

    bool haveEpoch = false;
    std::uint64_t epoch = 0;
    for(std::size_t i = 0; i < descriptorsByRank_.size(); ++i)
    {
        const FmmRankRootDescriptor& descriptor = descriptorsByRank_[i];
        if(descriptor.magic != FMM_MPI_PACKET_MAGIC ||
           descriptor.version != FMM_MPI_PACKET_VERSION)
            throw UniversalError("FmmProcessTree::build: incompatible root descriptor protocol");
        if(descriptor.rank != static_cast<int>(i))
            throw UniversalError("FmmProcessTree::build: descriptor rank/index mismatch");
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
            if(descriptor.particleCount != 0)
                throw UniversalError("FmmProcessTree::build: inactive rank reports particles");
            continue;
        }
        if(descriptor.active != 1 || descriptor.particleCount == 0 ||
           !finiteCube(descriptor.centerVector(), descriptor.halfSize))
            throw UniversalError("FmmProcessTree::build: invalid active root descriptor");
        if((descriptor.rootLeaf != 0 && descriptor.rootLeaf != 1) ||
           descriptor.childMask < 0 || descriptor.childMask > 255 ||
           (descriptor.rootLeaf != 0 && descriptor.childMask != 0) ||
           (descriptor.rootLeaf == 0 && descriptor.childMask == 0))
            throw UniversalError("FmmProcessTree::build: inconsistent root topology descriptor");
        activeRanks_.push_back(descriptor.rank);
    }
    if(activeRanks_.empty())
    {
        topologyHash_ = 0;
        std::vector<FmmRankRootDescriptor>().swap(descriptorsByRank_);
        return;
    }

    buildRange(0, activeRanks_.size(), 0);
    buildLevels();
    computeHash();
    std::vector<FmmRankRootDescriptor>().swap(descriptorsByRank_);
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
        const FmmRankRootDescriptor& descriptor =
            descriptorsByRank_[static_cast<std::size_t>(activeRanks_[i])];
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
        std::stable_sort(activeRanks_.begin() + static_cast<std::ptrdiff_t>(begin),
                         activeRanks_.begin() + static_cast<std::ptrdiff_t>(end),
            [&](int lhs, int rhs)
            {
                const Vector3D a = descriptorsByRank_[static_cast<std::size_t>(lhs)].centerVector();
                const Vector3D b = descriptorsByRank_[static_cast<std::size_t>(rhs)].centerVector();
                const double av = axis == 0 ? a.x : (axis == 1 ? a.y : a.z);
                const double bv = axis == 0 ? b.x : (axis == 1 ? b.y : b.z);
                return av < bv || (av == bv && lhs < rhs);
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
        node.leafRank = activeRanks_[begin];
        node.owner = node.leafRank;
        const FmmRankRootDescriptor& descriptor =
            descriptorsByRank_[static_cast<std::size_t>(node.leafRank)];
        node.center = descriptor.centerVector();
        node.halfSize = descriptor.halfSize;
        if(!finiteCube(node.center, node.halfSize))
            throw UniversalError("FmmProcessTree::buildRange: invalid process leaf cube");
        node.radius = std::sqrt(3.0) * node.halfSize;
        leafByRank_[node.leafRank] = nodeIndex;
        return nodeIndex;
    }

    const std::size_t mid = begin + (end - begin) / 2;
    const int owner = activeRanks_[mid];
    nodes_[nodeIndex].owner = owner;
    const std::size_t left = buildRange(begin, mid, depth + 1);
    const std::size_t right = buildRange(mid, end, depth + 1);
    nodes_[nodeIndex].left = left;
    nodes_[nodeIndex].right = right;
    nodes_[left].parent = nodeIndex;
    nodes_[right].parent = nodeIndex;
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
            static_cast<std::uint64_t>(node.leafRank + 1),
            static_cast<std::uint64_t>(node.begin),
            static_cast<std::uint64_t>(node.end),
            static_cast<std::uint64_t>(node.depth),
            doubleBits(node.center.x), doubleBits(node.center.y),
            doubleBits(node.center.z), doubleBits(node.halfSize)};
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
        descriptorsByRank_.capacity(), sizeof(FmmRankRootDescriptor)));
    result = saturatingAdd(result, saturatingMultiply(
        activeRanks_.capacity(), sizeof(int)));
    result = saturatingAdd(result, saturatingMultiply(
        nodes_.capacity(), sizeof(FmmProcessNode)));
    result = saturatingAdd(result, saturatingMultiply(
        levels_.capacity(), sizeof(std::vector<std::size_t>)));
    for(const std::vector<std::size_t>& level : levels_)
        result = saturatingAdd(result, saturatingMultiply(
            level.capacity(), sizeof(std::size_t)));
    const std::size_t mapEntryBytes =
        sizeof(std::pair<const int, std::size_t>) + 2 * sizeof(void*);
    result = saturatingAdd(result, saturatingMultiply(
        leafByRank_.size(), mapEntryBytes));
    return result;
}

std::size_t FmmProcessTree::leafForRank(int rank) const
{
    const auto it = leafByRank_.find(rank);
    return it == leafByRank_.end() ? invalidIndex() : it->second;
}

#endif // RICH_MPI
