#ifndef FMM_TREE_HPP
#define FMM_TREE_HPP

#include <cstdint>
#include <cstddef>
#include <limits>
#include <vector>

#include "3D/elementary/Vector3D.hpp"
#include "3D/gravity/fmm/FmmConfig.hpp"
#include "3D/gravity/fmm/FmmRootGeometry.hpp"

struct FmmNode
{
    Vector3D center;
    double halfSize = 0;
    double radius = 0;
    std::size_t particleBegin = 0;
    std::size_t particleEnd = 0;
    std::size_t firstChild = std::numeric_limits<std::size_t>::max();
    std::size_t parent = std::numeric_limits<std::size_t>::max();
    std::uint8_t childMask = 0;
    std::size_t depth = 0;
    std::uint64_t spatialKey = 1;
    std::size_t multipoleOffset = 0;
    std::size_t localOffset = 0;

    bool isLeaf() const { return firstChild == std::numeric_limits<std::size_t>::max(); }
    std::size_t particleCount() const { return particleEnd - particleBegin; }
};

class FmmTree
{
public:
    void build(const std::vector<Vector3D>& positions,
               const Vector3D& domainLower,
               const Vector3D& domainUpper,
               const FmmGravityOptions& options);

    void build(const std::vector<Vector3D>& positions,
               const FmmRootGeometry& rootGeometry,
               const FmmGravityOptions& options);

    const std::vector<FmmNode>& nodes() const { return nodes_; }
    const std::vector<std::size_t>& particleOrder() const { return particleOrder_; }
    const std::vector<std::size_t>& preOrder() const { return preOrder_; }
    const std::vector<std::size_t>& postOrder() const { return postOrder_; }
    const FmmRootGeometry& rootGeometry() const { return rootGeometry_; }

    std::size_t childIndex(const FmmNode& node, int octant) const;
    void assignExpansionOffsets(std::size_t coefficientCount);

    std::size_t leafCount() const;
    std::size_t maxDepth() const;
    std::size_t bytesOwned() const;
    std::uint64_t topologyHash() const;

private:
    void buildWithRoot(const std::vector<Vector3D>& positions,
                       const FmmRootGeometry& rootGeometry,
                       const FmmGravityOptions& options);
    void buildNode(std::size_t nodeIndex,
                   const std::vector<Vector3D>& positions,
                   const FmmGravityOptions& options);
    void finishMetadata(const std::vector<Vector3D>& positions);
    void collectOrders(std::size_t nodeIndex);
    void validateInvariants(std::size_t particleCount) const;

    FmmRootGeometry rootGeometry_;
    std::vector<FmmNode> nodes_;
    std::vector<std::size_t> particleOrder_;
    std::vector<std::size_t> scratchOrder_;
    std::vector<std::size_t> preOrder_;
    std::vector<std::size_t> postOrder_;
};

#endif // FMM_TREE_HPP
