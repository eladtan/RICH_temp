#include "3D/gravity/fmm/FmmDualTreeTraversal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <tuple>
#include <utility>

#include "3D/gravity/fmm/FmmKernels.hpp"
#include "misc/universal_error.hpp"

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

std::size_t progressInterval()
{
    static const std::size_t interval = [] {
        const char* value = std::getenv("RICH_FMM_PROGRESS_INTERVAL");
        if(value == nullptr || value[0] == '\0')
            return static_cast<std::size_t>(4096);
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if(end == value || *end != '\0' || parsed == 0 ||
           parsed > static_cast<unsigned long long>(
               std::numeric_limits<std::size_t>::max()))
            return static_cast<std::size_t>(4096);
        return static_cast<std::size_t>(parsed);
    }();
    return interval;
}
}

void FmmLocalInteractionPlan::clear()
{
    std::vector<M2LPair>().swap(m2lPairs);
    std::vector<P2PPair>().swap(p2pPairs);
    std::vector<FmmM2LOperatorCache::PreparedGeometry>().swap(geometries);
    std::vector<std::uint64_t>().swap(geometryUseCounts);
    std::vector<NodeGeometry>().swap(nodeGeometry);
    rejectedSameNode = 0;
    rejectedOverlap = 0;
    rejectedRatio = 0;
    maxTraversalStack = 0;
    initialized = false;
}

std::size_t FmmLocalInteractionPlan::bytesOwned() const
{
    return m2lPairs.capacity() * sizeof(M2LPair) +
           p2pPairs.capacity() * sizeof(P2PPair) +
           geometries.capacity() *
               sizeof(FmmM2LOperatorCache::PreparedGeometry) +
           geometryUseCounts.capacity() * sizeof(std::uint64_t) +
           nodeGeometry.capacity() * sizeof(NodeGeometry);
}

void FmmDualTreeTraversal::buildLocalPlan(
    const FmmTree& tree,
    double thetaCritical,
    FmmLocalInteractionPlan& plan)
{
    plan.clear();
    const std::vector<FmmNode>& nodes = tree.nodes();
    if(nodes.empty())
    {
        plan.initialized = true;
        return;
    }
    if(nodes.size() > static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()))
        throw UniversalError(
            "FmmDualTreeTraversal::buildLocalPlan: tree exceeds compact index range");

    plan.nodeGeometry.reserve(nodes.size());
    for(const FmmNode& node : nodes)
    {
        plan.nodeGeometry.push_back(FmmLocalInteractionPlan::NodeGeometry{
            node.center, node.halfSize, node.radius, node.isLeaf()});
    }
    typedef std::pair<std::size_t, std::size_t> NodePair;
    typedef std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                       std::uint64_t, std::uint64_t> GeometryKey;
    std::map<GeometryKey, std::uint32_t> geometryIndexByKey;
    std::vector<NodePair> stack;
    stack.push_back(NodePair(0, 0));
    while(!stack.empty())
    {
        plan.maxTraversalStack = std::max(plan.maxTraversalStack, stack.size());
        const NodePair pair = stack.back();
        stack.pop_back();
        const FmmNode& target = nodes[pair.first];
        const FmmNode& source = nodes[pair.second];
        const RejectionReason reason = admissibility(
            target, source, pair.first == pair.second, thetaCritical);
        if(reason == RejectionReason::None)
        {
            const FmmM2LOperatorCache::PreparedGeometry geometry =
                FmmM2LOperatorCache::prepare(source, target);
            std::uint64_t inverseScaleBits = 0;
            static_assert(sizeof(inverseScaleBits) == sizeof(geometry.inverseScale),
                          "prepared inverse scale must be 64-bit");
            std::memcpy(&inverseScaleBits, &geometry.inverseScale,
                        sizeof(inverseScaleBits));
            const GeometryKey key = std::make_tuple(
                geometry.keyX, geometry.keyY, geometry.keyZ,
                geometry.keyKind, inverseScaleBits);
            auto inserted = geometryIndexByKey.emplace(
                key, static_cast<std::uint32_t>(plan.geometries.size()));
            if(inserted.second)
            {
                if(plan.geometries.size() >= static_cast<std::size_t>(
                        std::numeric_limits<std::uint32_t>::max()))
                    throw UniversalError(
                        "FmmDualTreeTraversal::buildLocalPlan: too many geometries");
                inserted.first->second =
                    static_cast<std::uint32_t>(plan.geometries.size());
                plan.geometries.push_back(geometry);
                plan.geometryUseCounts.push_back(0);
            }

            const Vector3D delta = target.center - source.center;
            const double separation = std::sqrt(delta.x * delta.x +
                delta.y * delta.y + delta.z * delta.z);
            const double exactLimit = thetaCritical * separation;
            float conservativeLimit = static_cast<float>(exactLimit);
            if(static_cast<double>(conservativeLimit) > exactLimit)
                conservativeLimit = std::nextafter(conservativeLimit, 0.0f);
            plan.m2lPairs.push_back(FmmLocalInteractionPlan::M2LPair{
                static_cast<std::uint32_t>(pair.first),
                static_cast<std::uint32_t>(pair.second),
                inserted.first->second, conservativeLimit});
            if(plan.geometryUseCounts[inserted.first->second] ==
               std::numeric_limits<std::uint64_t>::max())
                throw UniversalError(
                    "FmmDualTreeTraversal::buildLocalPlan: geometry use count overflow");
            ++plan.geometryUseCounts[inserted.first->second];
            continue;
        }

        if(reason == RejectionReason::SameNode)
            ++plan.rejectedSameNode;
        else if(reason == RejectionReason::Overlap)
            ++plan.rejectedOverlap;
        else if(reason == RejectionReason::Ratio)
            ++plan.rejectedRatio;

        if(target.isLeaf() && source.isLeaf())
        {
            plan.p2pPairs.push_back(FmmLocalInteractionPlan::P2PPair{
                static_cast<std::uint32_t>(pair.first),
                static_cast<std::uint32_t>(pair.second)});
            continue;
        }

        const bool splitTarget = !target.isLeaf() &&
            (source.isLeaf() || target.radius >= source.radius);
        if(splitTarget)
        {
            for(int octant = 7; octant >= 0; --octant)
            {
                const std::size_t child = tree.childIndex(target, octant);
                if(child != std::numeric_limits<std::size_t>::max())
                    stack.push_back(NodePair(child, pair.second));
            }
        }
        else
        {
            for(int octant = 7; octant >= 0; --octant)
            {
                const std::size_t child = tree.childIndex(source, octant);
                if(child != std::numeric_limits<std::size_t>::max())
                    stack.push_back(NodePair(pair.first, child));
            }
        }
    }

    plan.m2lPairs.shrink_to_fit();
    plan.p2pPairs.shrink_to_fit();
    plan.geometries.shrink_to_fit();
    plan.geometryUseCounts.shrink_to_fit();
    plan.nodeGeometry.shrink_to_fit();
    plan.initialized = true;
}

bool FmmDualTreeTraversal::localPlanReusable(
    const FmmTree& tree,
    const FmmLocalInteractionPlan& plan)
{
    if(!plan.initialized)
        return false;
    if(plan.geometryUseCounts.size() != plan.geometries.size())
        return false;

    std::uint64_t plannedM2LCount = 0;
    for(std::uint64_t count : plan.geometryUseCounts)
    {
        if(count == 0 || count >
           std::numeric_limits<std::uint64_t>::max() - plannedM2LCount)
            return false;
        plannedM2LCount += count;
    }
    if(plannedM2LCount != static_cast<std::uint64_t>(plan.m2lPairs.size()))
        return false;

    const std::vector<FmmNode>& nodes = tree.nodes();
    if(plan.nodeGeometry.size() != nodes.size())
        return false;

    bool radiiUnchanged = true;
    for(std::size_t i = 0; i < nodes.size(); ++i)
    {
        const FmmNode& node = nodes[i];
        const FmmLocalInteractionPlan::NodeGeometry& saved =
            plan.nodeGeometry[i];
        // Prepared operators depend on node centers, while overlap decisions
        // depend on half sizes and P2P terminals depend on leaf status.  A
        // change in any of those requires rebuilding the plan.
        if(node.center.x != saved.center.x ||
           node.center.y != saved.center.y ||
           node.center.z != saved.center.z ||
           node.halfSize != saved.halfSize ||
           node.isLeaf() != saved.leaf)
            return false;
        if(node.radius != saved.radius)
            radiiUnchanged = false;
    }
    if(radiiUnchanged)
        return true;

    // Centers/topology are unchanged, but particle motion changed bounding
    // radii. Preserve the old conservative reuse test in that less-common
    // case rather than forcing a topology-plan rebuild.
    for(const FmmLocalInteractionPlan::M2LPair& pair : plan.m2lPairs)
    {
        if(pair.targetNode >= nodes.size() || pair.sourceNode >= nodes.size() ||
           pair.geometryIndex >= plan.geometries.size())
            return false;
        if(nodes[pair.targetNode].radius + nodes[pair.sourceNode].radius >
           static_cast<double>(pair.admissibleRadiusSum))
            return false;
    }
    for(const FmmLocalInteractionPlan::P2PPair& pair : plan.p2pPairs)
    {
        if(pair.targetNode >= nodes.size() || pair.sourceNode >= nodes.size() ||
           !nodes[pair.targetNode].isLeaf() || !nodes[pair.sourceNode].isLeaf())
            return false;
    }
    return true;
}

void FmmDualTreeTraversal::runLocalPlan(
    const FmmTree& tree,
    const FmmLocalInteractionPlan& plan,
    const std::vector<Vector3D>& positions,
    const std::vector<double>& masses,
    const FmmTaylorExpansion& layout,
    const std::vector<double>& multipoles,
    std::vector<double>& locals,
    std::vector<Vector3D>& acceleration,
    std::vector<double>* positiveKernelPotential,
    FmmM2LOperatorCache& operatorCache,
    std::size_t maxOperatorCacheBytes,
    FmmSolveStats& stats,
    FmmTraversalProgress progress,
    void* progressContext)
{
    if(!plan.initialized)
        throw UniversalError(
            "FmmDualTreeTraversal::runLocalPlan: uninitialized plan");
    const std::vector<FmmNode>& nodes = tree.nodes();

    // The persistent full-octant tree is a stable superset of all interactions
    // that may become active during this topology epoch. Build a cheap dynamic
    // occupancy mask so empty source/target subtrees do not resolve operators
    // or enter the hot M2L loop.
    std::vector<std::uint64_t> activeGeometryUseCounts(
        plan.geometries.size(), 0);
    std::vector<std::uint32_t> activeM2LIndices;
    std::vector<std::uint32_t> activeP2PIndices;
    const std::size_t progressEvery = progressInterval();
    std::size_t progressCountdown = progressEvery;
    if(plan.m2lPairs.size() > std::numeric_limits<std::uint32_t>::max() ||
       plan.p2pPairs.size() > std::numeric_limits<std::uint32_t>::max())
        throw UniversalError(
            "FmmDualTreeTraversal::runLocalPlan: interaction plan is too large");
    std::size_t activeM2LCount = 0;
    for(std::size_t pairIndex = 0; pairIndex < plan.m2lPairs.size(); ++pairIndex)
    {
        const FmmLocalInteractionPlan::M2LPair& pair = plan.m2lPairs[pairIndex];
        if(pair.targetNode >= nodes.size() || pair.sourceNode >= nodes.size() ||
           pair.geometryIndex >= plan.geometries.size())
            throw UniversalError(
                "FmmDualTreeTraversal::runLocalPlan: invalid retained M2L pair");
        if(nodes[pair.targetNode].particleCount() == 0 ||
           nodes[pair.sourceNode].particleCount() == 0)
        {
            ++stats.localInactiveM2LCount;
            if(progress != nullptr && --progressCountdown == 0)
            {
                progress(progressContext);
                progressCountdown = progressEvery;
            }
            continue;
        }
        activeM2LIndices.push_back(static_cast<std::uint32_t>(pairIndex));
        if(activeGeometryUseCounts[pair.geometryIndex] ==
           std::numeric_limits<std::uint64_t>::max())
            throw UniversalError(
                "FmmDualTreeTraversal::runLocalPlan: active geometry count overflow");
        ++activeGeometryUseCounts[pair.geometryIndex];
        ++activeM2LCount;
    }
    operatorCache.configure(maxOperatorCacheBytes, layout.m2lTerms().size(),
                            activeM2LCount);
    operatorCache.beginPhase();
    std::vector<double> derivativeScratch;
    derivativeScratch.reserve(layout.coefficientCount());
    std::vector<double> uncachedOperator;
    uncachedOperator.reserve(layout.m2lTerms().size());

    std::vector<const std::vector<double>*> resolvedOperators;
    const bool operatorsResolved = operatorCache.resolvePreparedBatch(
        plan.geometries, activeGeometryUseCounts, layout,
        derivativeScratch, uncachedOperator, resolvedOperators);

    stats.rejectedSameNode += plan.rejectedSameNode;
    stats.rejectedOverlap += plan.rejectedOverlap;
    stats.rejectedRatio += plan.rejectedRatio;
    stats.maxTraversalStack = std::max(stats.maxTraversalStack,
                                      plan.maxTraversalStack);
    for(std::uint32_t pairIndex : activeM2LIndices)
    {
        const FmmLocalInteractionPlan::M2LPair& pair = plan.m2lPairs[pairIndex];
        const FmmNode& target = nodes[pair.targetNode];
        const FmmNode& source = nodes[pair.sourceNode];
        const std::vector<double>* coefficients = nullptr;
        double inverseScale = plan.geometries[pair.geometryIndex].inverseScale;
        if(operatorsResolved)
        {
            coefficients = resolvedOperators[pair.geometryIndex];
        }
        else
        {
            const FmmM2LOperatorCache::Lookup translationOperator =
                operatorCache.getPrepared(plan.geometries[pair.geometryIndex],
                                          layout, derivativeScratch,
                                          uncachedOperator);
            coefficients = translationOperator.coefficients;
            inverseScale = translationOperator.inverseScale;
        }
        if(coefficients == nullptr)
            throw UniversalError(
                "FmmDualTreeTraversal::runLocalPlan: missing active M2L operator");
        FmmKernels::translateM2L(source, target, layout, multipoles, locals,
                                 *coefficients, inverseScale);
        ++stats.m2lCount;
        if(progress != nullptr && --progressCountdown == 0)
        {
            progress(progressContext);
            progressCountdown = progressEvery;
        }
    }
    for(std::size_t pairIndex = 0; pairIndex < plan.p2pPairs.size(); ++pairIndex)
    {
        const FmmLocalInteractionPlan::P2PPair& pair = plan.p2pPairs[pairIndex];
        const FmmNode& target = nodes[pair.targetNode];
        const FmmNode& source = nodes[pair.sourceNode];
        if(target.particleCount() == 0 || source.particleCount() == 0)
        {
            ++stats.localInactiveP2PBlockCount;
            if(progress != nullptr && --progressCountdown == 0)
            {
                progress(progressContext);
                progressCountdown = progressEvery;
            }
            continue;
        }
        activeP2PIndices.push_back(static_cast<std::uint32_t>(pairIndex));
    }
    for(std::uint32_t pairIndex : activeP2PIndices)
    {
        const FmmLocalInteractionPlan::P2PPair& pair = plan.p2pPairs[pairIndex];
        const FmmNode& target = nodes[pair.targetNode];
        const FmmNode& source = nodes[pair.sourceNode];
        FmmKernels::accumulateP2P(
            positions, positions, masses,
            tree.particleOrder(), tree.particleOrder(),
            target.particleBegin, target.particleEnd,
            source.particleBegin, source.particleEnd,
            true, acceleration, positiveKernelPotential, stats.p2pPairCount);
        ++stats.p2pBlockCount;
        if(progress != nullptr && --progressCountdown == 0)
        {
            progress(progressContext);
            progressCountdown = progressEvery;
        }
    }
    if(progress != nullptr)
        progress(progressContext);

    stats.localOperatorCacheBytes = operatorCache.bytesOwned();
    stats.localOperatorCacheEntries = operatorCache.entries();
    stats.localOperatorCacheMaxEntries = operatorCache.maxEntries();
    stats.localOperatorCacheHits = operatorCache.hits();
    stats.localOperatorCacheMisses = operatorCache.misses();
    stats.localOperatorCacheBypasses = operatorCache.bypasses();
    stats.localOperatorIntegerKeyHits = operatorCache.integerKeyHits();
    stats.localOperatorIntegerKeyMisses = operatorCache.integerKeyMisses();
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

        if(target.particleCount() == 0 || source.particleCount() == 0)
            continue;

        const bool identicalNode = sameParticleSet &&
            &targetTree == &sourceTree && pair.first == pair.second;
        const RejectionReason reason =
            admissibility(target, source, identicalNode, thetaCritical);
        if(reason == RejectionReason::None)
        {
            const FmmM2LOperatorCache::Lookup translationOperator =
                operatorCache.get(source, target, layout, derivativeScratch,
                                  uncachedOperator);
            FmmKernels::translateM2L(source, target, layout, sourceMultipoles,
                                     targetLocals,
                                     *translationOperator.coefficients,
                                     translationOperator.inverseScale);
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
    stats.localOperatorIntegerKeyHits = operatorCache.integerKeyHits();
    stats.localOperatorIntegerKeyMisses = operatorCache.integerKeyMisses();
}
