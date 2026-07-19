#ifndef FMM_TREE_HPP
#define FMM_TREE_HPP

#include <array>
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

    std::uint64_t latticeId = 0;
    std::int64_t latticeCenterX = 0;
    std::int64_t latticeCenterY = 0;
    std::int64_t latticeCenterZ = 0;
    std::uint64_t latticeHalfUnits = 0;
    int latticeAligned = 0;

    bool isLeaf() const { return firstChild == std::numeric_limits<std::size_t>::max(); }
    std::size_t particleCount() const { return particleEnd - particleBegin; }
};

struct FmmPersistentTreeStats
{
    std::size_t leafSplits = 0;
    std::size_t subtreeMerges = 0;
    std::size_t emptyLeaves = 0;
    bool initializedFromScratch = false;
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

    // Rebuild particle ranges and metadata while using the previous tree as a
    // hysteresis template. Existing leaves split only above splitCapacity and
    // existing internal nodes merge only at or below mergeCapacity. Every
    // retained internal node materializes all eight children, including empty
    // leaves, so ordinary particles crossing octant boundaries do not change
    // structural topology.
    void buildPersistent(const std::vector<Vector3D>& positions,
                         const FmmRootGeometry& rootGeometry,
                         const FmmGravityOptions& options,
                         std::size_t splitCapacity,
                         std::size_t mergeCapacity,
                         bool initializeFromScratch,
                         FmmPersistentTreeStats& stats);

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
    void initializeRoot(const std::vector<Vector3D>& positions,
                        const FmmRootGeometry& rootGeometry,
                        const FmmGravityOptions& options);
    void buildNode(std::size_t nodeIndex,
                   const std::vector<Vector3D>& positions,
                   const FmmGravityOptions& options);
    void buildPersistentNode(
        std::size_t nodeIndex,
        const std::vector<Vector3D>& positions,
        const FmmGravityOptions& options,
        std::size_t splitCapacity,
        std::size_t mergeCapacity,
        bool initializeFromScratch,
        const std::vector<std::uint64_t>& previousInternalKeys,
        FmmPersistentTreeStats& stats);
    void partitionNode(std::size_t nodeIndex,
                       const std::vector<Vector3D>& positions,
                       std::array<std::size_t, 8>& counts);
    void appendChildren(std::size_t nodeIndex,
                        const std::array<std::size_t, 8>& counts,
                        bool materializeEmptyChildren);
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
