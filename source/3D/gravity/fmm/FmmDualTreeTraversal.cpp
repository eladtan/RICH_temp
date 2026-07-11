#include "3D/gravity/fmm/FmmDualTreeTraversal.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <unordered_map>
#include <utility>

#include "3D/gravity/fmm/FmmKernels.hpp"

namespace
{
enum class RejectionReason
{
    None,
    SameNode,
    Overlap,
    Ratio
};

struct DisplacementKey
{
    double x;
    double y;
    double z;

    bool operator==(const DisplacementKey& other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct DisplacementKeyHash
{
    std::size_t operator()(const DisplacementKey& key) const
    {
        std::size_t result = std::hash<double>()(key.x);
        result ^= std::hash<double>()(key.y) + 0x9e3779b9u +
            (result << 6u) + (result >> 2u);
        result ^= std::hash<double>()(key.z) + 0x9e3779b9u +
            (result << 6u) + (result >> 2u);
        return result;
    }
};

double centerDistance(const FmmNode& first, const FmmNode& second)
{
    const Vector3D delta = first.center - second.center;
    return std::sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
}

bool cubesOverlap(const FmmNode& first, const FmmNode& second)
{
    return std::abs(first.center.x - second.center.x) <= first.halfSize + second.halfSize &&
           std::abs(first.center.y - second.center.y) <= first.halfSize + second.halfSize &&
           std::abs(first.center.z - second.center.z) <= first.halfSize + second.halfSize;
}

RejectionReason admissibility(const FmmNode& target,
                              const FmmNode& source,
                              bool identicalNode,
                              double thetaCritical)
{
    if(identicalNode)
        return RejectionReason::SameNode;
    if(cubesOverlap(target, source))
        return RejectionReason::Overlap;
    const double separation = centerDistance(target, source);
    if(!(separation > 0.0) ||
       target.radius + source.radius > thetaCritical * separation)
        return RejectionReason::Ratio;
    return RejectionReason::None;
}

void recordRejection(RejectionReason reason, FmmSolveStats& stats)
{
    if(reason == RejectionReason::SameNode)
        ++stats.rejectedSameNode;
    else if(reason == RejectionReason::Overlap)
        ++stats.rejectedOverlap;
    else if(reason == RejectionReason::Ratio)
        ++stats.rejectedRatio;
}
}

void FmmDualTreeTraversal::run(const FmmTree& targetTree,
                               const FmmTree& sourceTree,
                               const std::vector<Vector3D>& targetPositions,
                               const std::vector<Vector3D>& sourcePositions,
                               const std::vector<double>& sourceMasses,
                               const FmmTaylorExpansion& layout,
                               const std::vector<double>& sourceMultipoles,
                               std::vector<double>& targetLocals,
                               bool sameParticleSet,
                               double thetaCritical,
                               std::vector<Vector3D>& acceleration,
                               std::vector<double>* positiveKernelPotential,
                               FmmSolveStats& stats)
{
    const std::vector<FmmNode>& targetNodes = targetTree.nodes();
    const std::vector<FmmNode>& sourceNodes = sourceTree.nodes();
    if(targetNodes.empty() || sourceNodes.empty())
        return;

    typedef std::pair<std::size_t, std::size_t> NodePair;
    std::vector<NodePair> stack;
    stack.push_back(NodePair(0, 0));
    std::unordered_map<DisplacementKey, std::vector<double>, DisplacementKeyHash>
        translationCache;
    if(targetNodes.size() <= std::numeric_limits<std::size_t>::max() / 8)
        translationCache.reserve(targetNodes.size() * 8);
    std::vector<double> derivativeScratch;
    derivativeScratch.reserve(layout.coefficientCount());

    while(!stack.empty())
    {
        stats.maxTraversalStack = std::max(stats.maxTraversalStack, stack.size());
        const NodePair pair = stack.back();
        stack.pop_back();
        const FmmNode& target = targetNodes[pair.first];
        const FmmNode& source = sourceNodes[pair.second];

        const bool identicalNode = sameParticleSet &&
            &targetTree == &sourceTree && pair.first == pair.second;
        const RejectionReason reason =
            admissibility(target, source, identicalNode, thetaCritical);
        if(reason == RejectionReason::None)
        {
            const Vector3D displacement = target.center - source.center;
            const DisplacementKey key =
                {displacement.x, displacement.y, displacement.z};
            const auto inserted =
                translationCache.emplace(key, std::vector<double>());
            if(inserted.second)
            {
                FmmKernels::computeM2LOperator(
                    displacement, layout, derivativeScratch, inserted.first->second);
            }
            FmmKernels::translateM2L(source, target, layout, sourceMultipoles,
                                     targetLocals, inserted.first->second);
            ++stats.m2lCount;
            continue;
        }
        recordRejection(reason, stats);

        if(target.isLeaf() && source.isLeaf())
        {
            FmmKernels::accumulateP2P(targetPositions, sourcePositions, sourceMasses,
                                      targetTree.particleOrder(), sourceTree.particleOrder(),
                                      target.particleBegin, target.particleEnd,
                                      source.particleBegin, source.particleEnd,
                                      sameParticleSet, acceleration,
                                      positiveKernelPotential, stats.p2pPairCount);
            ++stats.p2pBlockCount;
            continue;
        }

        const bool splitTarget = !target.isLeaf() &&
            (source.isLeaf() || target.radius >= source.radius);
        if(splitTarget)
        {
            for(int octant = 7; octant >= 0; --octant)
            {
                const std::size_t child = targetTree.childIndex(target, octant);
                if(child != std::numeric_limits<std::size_t>::max())
                    stack.push_back(NodePair(child, pair.second));
            }
        }
        else
        {
            for(int octant = 7; octant >= 0; --octant)
            {
                const std::size_t child = sourceTree.childIndex(source, octant);
                if(child != std::numeric_limits<std::size_t>::max())
                    stack.push_back(NodePair(pair.first, child));
            }
        }
    }
}
