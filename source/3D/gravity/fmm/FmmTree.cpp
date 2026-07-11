#include "3D/gravity/fmm/FmmTree.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

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
}

void FmmTree::build(const std::vector<Vector3D>& positions,
                    const Vector3D& domainLower,
                    const Vector3D& domainUpper,
                    const FmmGravityOptions& options)
{
    nodes_.clear();
    particleOrder_.resize(positions.size());
    scratchOrder_.resize(positions.size());
    preOrder_.clear();
    postOrder_.clear();

    if(!finiteVector(domainLower) || !finiteVector(domainUpper) ||
       !(domainLower.x < domainUpper.x) ||
       !(domainLower.y < domainUpper.y) ||
       !(domainLower.z < domainUpper.z))
        throw UniversalError("FmmTree::build: domain bounds must be finite and strictly ordered");
    if(options.leafCapacity == 0 || options.maxDepth <= 0 ||
       options.maxDepth > FMM_MAX_TREE_DEPTH)
        throw UniversalError("FmmTree::build: invalid tree options");

    for(std::size_t i = 0; i < positions.size(); ++i)
    {
        const Vector3D& point = positions[i];
        if(!finiteVector(point))
        {
            UniversalError eo("FmmTree::build: non-finite particle position");
            eo.addEntry("particle", i);
            throw eo;
        }
        if(point.x < domainLower.x || point.x > domainUpper.x ||
           point.y < domainLower.y || point.y > domainUpper.y ||
           point.z < domainLower.z || point.z > domainUpper.z)
        {
            UniversalError eo("FmmTree::build: particle lies outside supplied domain");
            eo.addEntry("particle", i);
            throw eo;
        }
        particleOrder_[i] = i;
    }

    if(positions.empty())
        return;

    const Vector3D extent = domainUpper - domainLower;
    double halfSize = 0.5 * std::max(extent.x, std::max(extent.y, extent.z));
    halfSize += 16.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, halfSize);
    if(!(halfSize > 0.0) || !std::isfinite(halfSize))
        throw UniversalError("FmmTree::build: invalid root half-size");

    FmmNode root;
    root.center = 0.5 * (domainLower + domainUpper);
    root.halfSize = halfSize;
    root.particleEnd = positions.size();
    nodes_.push_back(root);

    buildNode(0, positions, options);
    finishMetadata(positions);
    collectOrders(0);
    validateInvariants(positions.size());
}

void FmmTree::buildNode(std::size_t nodeIndex,
                        const std::vector<Vector3D>& positions,
                        const FmmGravityOptions& options)
{
    const FmmNode node = nodes_[nodeIndex];
    if(node.particleCount() <= options.leafCapacity ||
       node.depth >= static_cast<std::size_t>(options.maxDepth) ||
       node.halfSize <= std::numeric_limits<double>::min())
        return;

    std::array<std::size_t, 8> counts{};
    for(std::size_t k = node.particleBegin; k < node.particleEnd; ++k)
        ++counts[static_cast<std::size_t>(octantOf(positions[particleOrder_[k]], node.center))];

    std::array<std::size_t, 8> offsets{};
    offsets[0] = node.particleBegin;
    for(int i = 1; i < 8; ++i)
        offsets[static_cast<std::size_t>(i)] =
            offsets[static_cast<std::size_t>(i - 1)] + counts[static_cast<std::size_t>(i - 1)];
    std::array<std::size_t, 8> cursor = offsets;

    for(std::size_t k = node.particleBegin; k < node.particleEnd; ++k)
    {
        const std::size_t original = particleOrder_[k];
        const int octant = octantOf(positions[original], node.center);
        scratchOrder_[cursor[static_cast<std::size_t>(octant)]++] = original;
    }
    for(std::size_t k = node.particleBegin; k < node.particleEnd; ++k)
        particleOrder_[k] = scratchOrder_[k];

    const std::size_t firstChild = nodes_.size();
    const double childHalfSize = 0.5 * node.halfSize;
    std::uint8_t childMask = 0;
    std::size_t begin = node.particleBegin;
    for(int octant = 0; octant < 8; ++octant)
    {
        const std::size_t count = counts[static_cast<std::size_t>(octant)];
        if(count == 0)
            continue;
        childMask |= static_cast<std::uint8_t>(1u << octant);

        FmmNode child;
        child.center = childCenter(node.center, node.halfSize, octant);
        child.halfSize = childHalfSize;
        child.particleBegin = begin;
        child.particleEnd = begin + count;
        child.parent = nodeIndex;
        child.depth = node.depth + 1;
        child.spatialKey = (node.spatialKey << 3u) | static_cast<std::uint64_t>(octant);
        nodes_.push_back(child);
        begin += count;
    }

    nodes_[nodeIndex].firstChild = firstChild;
    nodes_[nodeIndex].childMask = childMask;

    for(int octant = 0; octant < 8; ++octant)
    {
        const std::size_t child = childIndex(nodes_[nodeIndex], octant);
        if(child != std::numeric_limits<std::size_t>::max())
            buildNode(child, positions, options);
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
                radius = std::max(radius, nodeDistance(positions[particleOrder_[k]], node.center));
        }
        else
        {
            for(int octant = 0; octant < 8; ++octant)
            {
                const std::size_t child = childIndex(node, octant);
                if(child == std::numeric_limits<std::size_t>::max())
                    continue;
                const FmmNode& childNode = nodes_[child];
                radius = std::max(radius,
                                  nodeDistance(childNode.center, node.center) + childNode.radius);
            }
        }
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
        throw UniversalError("FmmTree::assignExpansionOffsets: coefficient storage overflow");
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
       nodes_[0].parent != std::numeric_limits<std::size_t>::max())
        throw UniversalError("FmmTree::build: invalid root invariants");
    if(preOrder_.size() != nodes_.size() || postOrder_.size() != nodes_.size())
        throw UniversalError("FmmTree::build: incomplete traversal orders");

    for(std::size_t i = 0; i < nodes_.size(); ++i)
    {
        const FmmNode& node = nodes_[i];
        if(node.particleBegin > node.particleEnd || node.particleEnd > particleCount ||
           !finiteVector(node.center) || !std::isfinite(node.radius) || node.radius < 0)
            throw UniversalError("FmmTree::build: invalid node metadata");
        if(node.isLeaf())
        {
            if(node.childMask != 0)
                throw UniversalError("FmmTree::build: leaf has child mask");
            continue;
        }
        if(node.childMask == 0 || node.firstChild >= nodes_.size())
            throw UniversalError("FmmTree::build: internal node has no valid children");

        std::size_t cursor = node.particleBegin;
        for(int octant = 0; octant < 8; ++octant)
        {
            const std::size_t child = childIndex(node, octant);
            if(child == std::numeric_limits<std::size_t>::max())
                continue;
            if(child >= nodes_.size() || nodes_[child].parent != i ||
               nodes_[child].particleBegin != cursor)
                throw UniversalError("FmmTree::build: invalid parent/child partition");
            cursor = nodes_[child].particleEnd;
        }
        if(cursor != node.particleEnd)
            throw UniversalError("FmmTree::build: child ranges do not cover parent");
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

std::size_t FmmTree::bytesOwned() const
{
    return nodes_.capacity() * sizeof(FmmNode) +
           particleOrder_.capacity() * sizeof(std::size_t) +
           scratchOrder_.capacity() * sizeof(std::size_t) +
           preOrder_.capacity() * sizeof(std::size_t) +
           postOrder_.capacity() * sizeof(std::size_t);
}
