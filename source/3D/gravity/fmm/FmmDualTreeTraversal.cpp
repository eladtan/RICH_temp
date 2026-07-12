#include "3D/gravity/fmm/FmmDualTreeTraversal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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
                               FmmM2LOperatorCache& operatorCache,
                               std::size_t maxOperatorCacheBytes,
                               FmmSolveStats& stats)
{
    const std::vector<FmmNode>& targetNodes = targetTree.nodes();
    const std::vector<FmmNode>& sourceNodes = sourceTree.nodes();
    if(targetNodes.empty() || sourceNodes.empty())
        return;

    typedef std::pair<std::size_t, std::size_t> NodePair;
    std::vector<NodePair> stack;
    stack.push_back(NodePair(0, 0));
    const std::size_t entryHint =
        targetNodes.size() <= std::numeric_limits<std::size_t>::max() / 8 ?
        targetNodes.size() * 8 : std::numeric_limits<std::size_t>::max();
    operatorCache.configure(maxOperatorCacheBytes, layout.m2lTerms().size(),
                            entryHint);
    operatorCache.beginPhase();
    std::vector<double> derivativeScratch;
    derivativeScratch.reserve(layout.coefficientCount());
    std::vector<double> uncachedOperator;
    uncachedOperator.reserve(layout.m2lTerms().size());

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
            const std::vector<double>& translationOperator =
                operatorCache.get(displacement, layout, derivativeScratch,
                                  uncachedOperator);
            FmmKernels::translateM2L(source, target, layout, sourceMultipoles,
                                     targetLocals, translationOperator);
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

    stats.localOperatorCacheBytes = operatorCache.bytesOwned();
    stats.localOperatorCacheEntries = operatorCache.entries();
    stats.localOperatorCacheMaxEntries = operatorCache.maxEntries();
    stats.localOperatorCacheHits = operatorCache.hits();
    stats.localOperatorCacheMisses = operatorCache.misses();
    stats.localOperatorCacheBypasses = operatorCache.bypasses();
}
