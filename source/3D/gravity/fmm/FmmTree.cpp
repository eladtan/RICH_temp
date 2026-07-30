#include "3D/gravity/fmm/FmmTree.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

#include "misc/universal_error.hpp"

namespace
{
double sqr(double x)
{
    return x * x;
}

double nodeDistance(const Vector3D& a, const Vector3D& b)
{
    return std::sqrt(sqr(a.x - b.x) + sqr(a.y - b.y) + sqr(a.z - b.z));
}

bool finiteVector(const Vector3D& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

bool sameRootGeometry(const FmmRootGeometry& first,
                      const FmmRootGeometry& second)
{
    return first.active == second.active &&
        (!first.active || (first.center.x == second.center.x &&
                           first.center.y == second.center.y &&
                           first.center.z == second.center.z &&
                           first.halfSize == second.halfSize &&
                           first.latticeId == second.latticeId &&
                           first.latticeCenterX == second.latticeCenterX &&
                           first.latticeCenterY == second.latticeCenterY &&
                           first.latticeCenterZ == second.latticeCenterZ &&
                           first.latticeHalfUnits == second.latticeHalfUnits &&
                           first.latticeAligned == second.latticeAligned));
}

// Match RICH's existing octree convention: x is bit 2, y bit 1, z bit 0.
int octantOf(const Vector3D& point, const Vector3D& center)
{
    return (point.x >= center.x ? 4 : 0) |
           (point.y >= center.y ? 2 : 0) |
           (point.z >= center.z ? 1 : 0);
}

Vector3D childCenter(const Vector3D& center, double halfSize, int octant)
{
    const double quarter = 0.5 * halfSize;
    return Vector3D(center.x + ((octant & 4) ? quarter : -quarter),
                    center.y + ((octant & 2) ? quarter : -quarter),
                    center.z + ((octant & 1) ? quarter : -quarter));
}

int bitCount(unsigned int value)
{
    int result = 0;
    while(value != 0)
    {
        result += static_cast<int>(value & 1u);
        value >>= 1u;
    }
    return result;
}

std::int64_t childLatticeCoordinate(std::int64_t parent,
                                    std::uint64_t childHalfUnits,
                                    bool upper)
{
    if(childHalfUnits > static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max()))
        throw UniversalError("FmmTree::build: lattice half-size overflow");
    const std::int64_t offset = static_cast<std::int64_t>(childHalfUnits);
    if((upper && parent > std::numeric_limits<std::int64_t>::max() - offset) ||
       (!upper && parent < std::numeric_limits<std::int64_t>::min() + offset))
        throw UniversalError("FmmTree::build: lattice coordinate overflow");
    return upper ? parent + offset : parent - offset;
}
}

void FmmTree::build(const std::vector<Vector3D>& positions,
                    const Vector3D& domainLower,
                    const Vector3D& domainUpper,
                    const FmmGravityOptions& options)
{
    const FmmRootGeometry root =
        FmmRootGeometry::fromDomain(domainLower, domainUpper, true);
    buildWithRoot(positions, root, options);
}

void FmmTree::build(const std::vector<Vector3D>& positions,
                    const FmmRootGeometry& rootGeometry,
                    const FmmGravityOptions& options)
{
    buildWithRoot(positions, rootGeometry, options);
}

void FmmTree::initializeRoot(const std::vector<Vector3D>& positions,
                             const FmmRootGeometry& rootGeometry,
                             const FmmGravityOptions& options)
{
    nodes_.clear();
    particleOrder_.resize(positions.size());
    scratchOrder_.resize(positions.size());
    preOrder_.clear();
    postOrder_.clear();

    rootGeometry.validate();
    rootGeometry_ = rootGeometry;
    if(options.leafCapacity == 0 || options.maxDepth <= 0 ||
       options.maxDepth > FMM_MAX_TREE_DEPTH)
        throw UniversalError("FmmTree::build: invalid tree options");
    if(positions.empty())
    {
        if(rootGeometry.active)
            rootGeometry_ = FmmRootGeometry();
        return;
    }
    if(!rootGeometry.active)
        throw UniversalError("FmmTree::build: non-empty tree requires an active root");

    for(std::size_t i = 0; i < positions.size(); ++i)
    {
        const Vector3D& point = positions[i];
        if(!finiteVector(point))
        {
            UniversalError eo("FmmTree::build: non-finite particle position");
            eo.addEntry("particle", i);
            throw eo;
        }
        if(!rootGeometry.contains(point))
        {
            UniversalError eo("FmmTree::build: particle lies outside supplied domain");
            eo.addEntry("particle", i);
            throw eo;
        }
        particleOrder_[i] = i;
    }

    FmmNode root;
    root.center = rootGeometry.center;
    root.halfSize = rootGeometry.halfSize;
    root.particleEnd = positions.size();
    root.latticeId = rootGeometry.latticeId;
    root.latticeCenterX = rootGeometry.latticeCenterX;
    root.latticeCenterY = rootGeometry.latticeCenterY;
    root.latticeCenterZ = rootGeometry.latticeCenterZ;
    root.latticeHalfUnits = rootGeometry.latticeHalfUnits;
    root.latticeAligned = rootGeometry.latticeAligned;
    nodes_.push_back(root);
}

void FmmTree::buildWithRoot(const std::vector<Vector3D>& positions,
                            const FmmRootGeometry& rootGeometry,
                            const FmmGravityOptions& options)
{
    initializeRoot(positions, rootGeometry, options);
    if(nodes_.empty())
        return;

    buildNode(0, positions, options);
    finishMetadata(positions);
    collectOrders(0);
    validateInvariants(positions.size());
}

void FmmTree::buildPersistent(
    const std::vector<Vector3D>& positions,
    const FmmRootGeometry& rootGeometry,
    const FmmGravityOptions& options,
    std::size_t splitCapacity,
    std::size_t mergeCapacity,
    bool initializeFromScratch,
    FmmPersistentTreeStats& stats)
{
    if(!(options.persistentRadiusSlackFactor >= 1.0) ||
       !std::isfinite(options.persistentRadiusSlackFactor))
        throw UniversalError(
            "FmmTree::buildPersistent: invalid persistent radius slack factor");

    if(splitCapacity <= options.leafCapacity ||
       mergeCapacity >= options.leafCapacity ||
       mergeCapacity >= splitCapacity)
        throw UniversalError("FmmTree::buildPersistent: invalid hysteresis capacities");
    if(!initializeFromScratch &&
       (!rootGeometry_.active || nodes_.empty() ||
        !sameRootGeometry(rootGeometry_, rootGeometry)))
        throw UniversalError(
            "FmmTree::buildPersistent: retained topology has a different root");
    if(!initializeFromScratch)
    {
        for(const FmmNode& node : nodes_)
        {
            if(!node.isLeaf() && node.childMask != 0xffu)
                throw UniversalError(
                    "FmmTree::buildPersistent: retained topology is not full-octant");
        }
    }

    // The planning radius is part of the conservative interaction contract,
    // but not of the dyadic node identity. Keep the previous envelope by
    // spatial key while rebuilding particle ranges. currentRadius remains the
    // tight geometry of this solve, while radius becomes the retained planning
    // envelope used by local and remote traversal.
    std::vector<std::pair<std::uint64_t, double>> previousRadiusByKey;
    if(!initializeFromScratch)
    {
        previousRadiusByKey.reserve(nodes_.size());
        for(const FmmNode& node : nodes_)
            previousRadiusByKey.emplace_back(node.spatialKey, node.radius);
        std::sort(previousRadiusByKey.begin(), previousRadiusByKey.end());
    }

    std::vector<std::uint64_t> previousInternalKeys;
    if(!initializeFromScratch)
    {
        previousInternalKeys.reserve(nodes_.size());
        for(const FmmNode& node : nodes_)
            if(!node.isLeaf())
                previousInternalKeys.push_back(node.spatialKey);
        std::sort(previousInternalKeys.begin(), previousInternalKeys.end());
    }

    stats = FmmPersistentTreeStats();
    stats.initializedFromScratch = initializeFromScratch;
    initializeRoot(positions, rootGeometry, options);
    if(nodes_.empty())
        return;

    buildPersistentNode(0, positions, options, splitCapacity, mergeCapacity,
                        initializeFromScratch, previousInternalKeys, stats);
    finishMetadata(positions);

    // Grow an envelope only when the actual radius breaches the previous one.
    // Contraction keeps the previous radius, which is conservative.  When a
    // breach occurs, reserve fresh multiplicative headroom.  The node's cube
    // radius is an absolute geometric upper bound required by the wire checks.
    for(FmmNode& node : nodes_)
    {
        const double actualRadius = node.currentRadius;
        const double cubeRadius = std::sqrt(3.0) * node.halfSize;
        double retainedRadius = std::min(
            cubeRadius, actualRadius * options.persistentRadiusSlackFactor);

        if(!initializeFromScratch)
        {
            const auto previous = std::lower_bound(
                previousRadiusByKey.begin(), previousRadiusByKey.end(),
                node.spatialKey,
                [](const std::pair<std::uint64_t, double>& value,
                   std::uint64_t key) { return value.first < key; });
            if(previous != previousRadiusByKey.end() &&
               previous->first == node.spatialKey)
            {
                const double tolerance =
                    64.0 * std::numeric_limits<double>::epsilon() *
                    std::max(1.0, std::max(actualRadius, previous->second));
                if(actualRadius <= previous->second + tolerance)
                    retainedRadius = std::max(actualRadius, previous->second);
            }
        }

        if(!std::isfinite(retainedRadius) || retainedRadius < actualRadius ||
           retainedRadius > cubeRadius)
            throw UniversalError(
                "FmmTree::buildPersistent: invalid retained radius envelope");
        node.radius = retainedRadius;
    }

    collectOrders(0);
    validateInvariants(positions.size());

    for(const FmmNode& node : nodes_)
        if(node.isLeaf() && node.particleCount() == 0)
            ++stats.emptyLeaves;
}

void FmmTree::partitionNode(std::size_t nodeIndex,
                            const std::vector<Vector3D>& positions,
                            std::array<std::size_t, 8>& counts)
{
    const FmmNode node = nodes_[nodeIndex];
    counts.fill(0);
    for(std::size_t k = node.particleBegin; k < node.particleEnd; ++k)
        ++counts[static_cast<std::size_t>(
            octantOf(positions[particleOrder_[k]], node.center))];

    std::array<std::size_t, 8> offsets{};
    offsets[0] = node.particleBegin;
    for(int i = 1; i < 8; ++i)
        offsets[static_cast<std::size_t>(i)] =
            offsets[static_cast<std::size_t>(i - 1)] +
            counts[static_cast<std::size_t>(i - 1)];
    std::array<std::size_t, 8> cursor = offsets;

    for(std::size_t k = node.particleBegin; k < node.particleEnd; ++k)
    {
        const std::size_t original = particleOrder_[k];
        const int octant = octantOf(positions[original], node.center);
        scratchOrder_[cursor[static_cast<std::size_t>(octant)]++] = original;
    }
    for(std::size_t k = node.particleBegin; k < node.particleEnd; ++k)
        particleOrder_[k] = scratchOrder_[k];
}

void FmmTree::appendChildren(std::size_t nodeIndex,
                             const std::array<std::size_t, 8>& counts,
                             bool materializeEmptyChildren)
{
    const FmmNode node = nodes_[nodeIndex];
    const std::size_t firstChild = nodes_.size();
    const double childHalfSize = 0.5 * node.halfSize;
    std::uint8_t childMask = 0;
    std::size_t begin = node.particleBegin;
    for(int octant = 0; octant < 8; ++octant)
    {
        const std::size_t count = counts[static_cast<std::size_t>(octant)];
        if(count == 0 && !materializeEmptyChildren)
            continue;
        childMask |= static_cast<std::uint8_t>(1u << octant);

        FmmNode child;
        child.center = childCenter(node.center, node.halfSize, octant);
        child.halfSize = childHalfSize;
        child.particleBegin = begin;
        child.particleEnd = begin + count;
        child.parent = nodeIndex;
        child.depth = node.depth + 1;
        child.spatialKey = (node.spatialKey << 3u) |
            static_cast<std::uint64_t>(octant);
        if(node.latticeAligned != 0)
        {
            if(node.latticeHalfUnits < 2 || (node.latticeHalfUnits & 1u) != 0)
                throw UniversalError("FmmTree::build: indivisible lattice half size");
            const std::uint64_t childHalfUnits = node.latticeHalfUnits / 2;
            child.latticeId = node.latticeId;
            child.latticeCenterX = childLatticeCoordinate(
                node.latticeCenterX, childHalfUnits, (octant & 4) != 0);
            child.latticeCenterY = childLatticeCoordinate(
                node.latticeCenterY, childHalfUnits, (octant & 2) != 0);
            child.latticeCenterZ = childLatticeCoordinate(
                node.latticeCenterZ, childHalfUnits, (octant & 1) != 0);
            child.latticeHalfUnits = childHalfUnits;
            child.latticeAligned = 1;
        }
        nodes_.push_back(child);
        begin += count;
    }

    nodes_[nodeIndex].firstChild = firstChild;
    nodes_[nodeIndex].childMask = childMask;
}

void FmmTree::buildNode(std::size_t nodeIndex,
                        const std::vector<Vector3D>& positions,
                        const FmmGravityOptions& options)
{
    const FmmNode node = nodes_[nodeIndex];
    const bool sizeRequiresSplit = node.particleCount() != 0 &&
        options.maxLeafHalfSize > 0.0 &&
        node.halfSize > options.maxLeafHalfSize;
    if((node.particleCount() <= options.leafCapacity && !sizeRequiresSplit) ||
       node.depth >= static_cast<std::size_t>(options.maxDepth) ||
       node.halfSize <= std::numeric_limits<double>::min())
        return;

    std::array<std::size_t, 8> counts{};
    partitionNode(nodeIndex, positions, counts);
    appendChildren(nodeIndex, counts, false);

    for(int octant = 0; octant < 8; ++octant)
    {
        const std::size_t child = childIndex(nodes_[nodeIndex], octant);
        if(child != std::numeric_limits<std::size_t>::max())
            buildNode(child, positions, options);
    }
}

void FmmTree::buildPersistentNode(
    std::size_t nodeIndex,
    const std::vector<Vector3D>& positions,
    const FmmGravityOptions& options,
    std::size_t splitCapacity,
    std::size_t mergeCapacity,
    bool initializeFromScratch,
    const std::vector<std::uint64_t>& previousInternalKeys,
    FmmPersistentTreeStats& stats)
{
    const FmmNode node = nodes_[nodeIndex];
    const bool wasInternal = !initializeFromScratch &&
        std::binary_search(previousInternalKeys.begin(),
                           previousInternalKeys.end(), node.spatialKey);
    const bool canSplit =
        node.depth < static_cast<std::size_t>(options.maxDepth) &&
        node.halfSize > std::numeric_limits<double>::min();
    const bool sizeRequiresSplit = node.particleCount() != 0 &&
        options.maxLeafHalfSize > 0.0 &&
        node.halfSize > options.maxLeafHalfSize;
    const bool shouldSplit = canSplit &&
        (sizeRequiresSplit ||
         (initializeFromScratch ? node.particleCount() > options.leafCapacity :
         (wasInternal ? node.particleCount() > mergeCapacity :
                        node.particleCount() > splitCapacity)));

    if(!shouldSplit)
    {
        if(wasInternal)
            ++stats.subtreeMerges;
        return;
    }
    if(!initializeFromScratch && !wasInternal)
        ++stats.leafSplits;

    std::array<std::size_t, 8> counts{};
    partitionNode(nodeIndex, positions, counts);
    appendChildren(nodeIndex, counts, true);

    for(int octant = 0; octant < 8; ++octant)
    {
        const std::size_t child = childIndex(nodes_[nodeIndex], octant);
        if(child == std::numeric_limits<std::size_t>::max())
            throw UniversalError(
                "FmmTree::buildPersistent: full child materialization failed");
        buildPersistentNode(child, positions, options, splitCapacity,
                            mergeCapacity, initializeFromScratch,
                            previousInternalKeys, stats);
    }
}

void FmmTree::finishMetadata(const std::vector<Vector3D>& positions)
{
    for(std::size_t reverse = nodes_.size(); reverse > 0; --reverse)
    {
        FmmNode& node = nodes_[reverse - 1];
        double radius = 0.0;
        if(node.isLeaf())
        {
            for(std::size_t k = node.particleBegin; k < node.particleEnd; ++k)
                radius = std::max(radius,
                    nodeDistance(positions[particleOrder_[k]], node.center));
        }
        else
        {
            for(int octant = 0; octant < 8; ++octant)
            {
                const std::size_t child = childIndex(node, octant);
                if(child == std::numeric_limits<std::size_t>::max())
                    continue;
                const FmmNode& childNode = nodes_[child];
                if(childNode.particleCount() == 0)
                    continue;
                radius = std::max(radius,
                    nodeDistance(childNode.center, node.center) +
                    childNode.currentRadius);
            }
        }
        node.currentRadius = radius;
        node.radius = radius;
    }
}

void FmmTree::collectOrders(std::size_t nodeIndex)
{
    preOrder_.push_back(nodeIndex);
    const FmmNode node = nodes_[nodeIndex];
    for(int octant = 0; octant < 8; ++octant)
    {
        const std::size_t child = childIndex(node, octant);
        if(child != std::numeric_limits<std::size_t>::max())
            collectOrders(child);
    }
    postOrder_.push_back(nodeIndex);
}

std::size_t FmmTree::childIndex(const FmmNode& node, int octant) const
{
    if(octant < 0 || octant >= 8 || node.isLeaf() ||
       (node.childMask & (1u << octant)) == 0)
        return std::numeric_limits<std::size_t>::max();
    const unsigned int preceding = static_cast<unsigned int>(node.childMask) &
        ((1u << static_cast<unsigned int>(octant)) - 1u);
    return node.firstChild + static_cast<std::size_t>(bitCount(preceding));
}

void FmmTree::assignExpansionOffsets(std::size_t coefficientCount)
{
    if(coefficientCount == 0 ||
       nodes_.size() > std::numeric_limits<std::size_t>::max() / coefficientCount)
        throw UniversalError(
            "FmmTree::assignExpansionOffsets: coefficient storage overflow");
    for(std::size_t i = 0; i < nodes_.size(); ++i)
    {
        nodes_[i].multipoleOffset = i * coefficientCount;
        nodes_[i].localOffset = i * coefficientCount;
    }
}

void FmmTree::validateInvariants(std::size_t particleCount) const
{
    if(nodes_.empty() || nodes_[0].particleBegin != 0 ||
       nodes_[0].particleEnd != particleCount ||
       nodes_[0].parent != std::numeric_limits<std::size_t>::max() ||
       nodes_[0].latticeId != rootGeometry_.latticeId ||
       nodes_[0].latticeCenterX != rootGeometry_.latticeCenterX ||
       nodes_[0].latticeCenterY != rootGeometry_.latticeCenterY ||
       nodes_[0].latticeCenterZ != rootGeometry_.latticeCenterZ ||
       nodes_[0].latticeHalfUnits != rootGeometry_.latticeHalfUnits ||
       nodes_[0].latticeAligned != rootGeometry_.latticeAligned)
        throw UniversalError("FmmTree::build: invalid root invariants");
    if(preOrder_.size() != nodes_.size() || postOrder_.size() != nodes_.size())
        throw UniversalError("FmmTree::build: incomplete traversal orders");

    for(std::size_t i = 0; i < nodes_.size(); ++i)
    {
        const FmmNode& node = nodes_[i];
        if(node.particleBegin > node.particleEnd ||
           node.particleEnd > particleCount ||
           !finiteVector(node.center) || !std::isfinite(node.radius) ||
           node.radius < 0 || !std::isfinite(node.currentRadius) ||
           node.currentRadius < 0 || node.currentRadius > node.radius)
            throw UniversalError("FmmTree::build: invalid node metadata");
        if(node.isLeaf())
        {
            if(node.childMask != 0)
                throw UniversalError("FmmTree::build: leaf has child mask");
            continue;
        }
        if(node.childMask == 0 || node.firstChild >= nodes_.size())
            throw UniversalError(
                "FmmTree::build: internal node has no valid children");

        std::size_t cursor = node.particleBegin;
        for(int octant = 0; octant < 8; ++octant)
        {
            const std::size_t child = childIndex(node, octant);
            if(child == std::numeric_limits<std::size_t>::max())
                continue;
            if(child >= nodes_.size() || nodes_[child].parent != i ||
               nodes_[child].particleBegin != cursor)
                throw UniversalError(
                    "FmmTree::build: invalid parent/child partition");
            if(node.latticeAligned != 0)
            {
                const FmmNode& childNode = nodes_[child];
                if(node.latticeHalfUnits < 2 ||
                   (node.latticeHalfUnits & 1u) != 0)
                    throw UniversalError(
                        "FmmTree::build: invalid parent lattice half size");
                const std::uint64_t childHalfUnits =
                    node.latticeHalfUnits / 2;
                if(childNode.latticeAligned == 0 ||
                   childNode.latticeId != node.latticeId ||
                   childNode.latticeHalfUnits != childHalfUnits ||
                   childNode.latticeCenterX != childLatticeCoordinate(
                       node.latticeCenterX, childHalfUnits,
                       (octant & 4) != 0) ||
                   childNode.latticeCenterY != childLatticeCoordinate(
                       node.latticeCenterY, childHalfUnits,
                       (octant & 2) != 0) ||
                   childNode.latticeCenterZ != childLatticeCoordinate(
                       node.latticeCenterZ, childHalfUnits,
                       (octant & 1) != 0))
                    throw UniversalError(
                        "FmmTree::build: invalid child lattice metadata");
            }
            cursor = nodes_[child].particleEnd;
        }
        if(cursor != node.particleEnd)
            throw UniversalError(
                "FmmTree::build: child ranges do not cover parent");
    }
}

std::size_t FmmTree::leafCount() const
{
    std::size_t result = 0;
    for(const FmmNode& node : nodes_)
        result += node.isLeaf() ? 1 : 0;
    return result;
}

std::size_t FmmTree::maxDepth() const
{
    std::size_t result = 0;
    for(const FmmNode& node : nodes_)
        result = std::max(result, node.depth);
    return result;
}

std::uint64_t FmmTree::topologyHash() const
{
    const std::uint64_t offset = 1469598103934665603ull;
    const std::uint64_t prime = 1099511628211ull;
    std::uint64_t hash = offset;
    for(const FmmNode& node : nodes_)
    {
        const std::uint64_t values[4] = {
            node.spatialKey,
            static_cast<std::uint64_t>(node.childMask),
            static_cast<std::uint64_t>(node.depth),
            static_cast<std::uint64_t>(node.particleCount())};
        for(std::uint64_t value : values)
        {
            for(int byte = 0; byte < 8; ++byte)
            {
                hash ^= (value >> (8 * byte)) & 0xffu;
                hash *= prime;
            }
        }
    }
    return hash;
}

std::size_t FmmTree::bytesOwned() const
{
    return nodes_.capacity() * sizeof(FmmNode) +
           particleOrder_.capacity() * sizeof(std::size_t) +
           scratchOrder_.capacity() * sizeof(std::size_t) +
           preOrder_.capacity() * sizeof(std::size_t) +
           postOrder_.capacity() * sizeof(std::size_t);
}
