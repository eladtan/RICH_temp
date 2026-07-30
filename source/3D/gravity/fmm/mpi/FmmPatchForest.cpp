#include "3D/gravity/fmm/mpi/FmmPatchForest.hpp"

#ifdef RICH_MPI

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "3D/gravity/fmm/FmmPasses.hpp"
#include "misc/universal_error.hpp"

namespace
{
std::uint64_t hashCombine(std::uint64_t seed, std::uint64_t value)
{
    return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
}

std::uint64_t hashDouble(double value)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(double));
    return bits;
}

std::uint64_t structuralSignatureHash(
    const std::vector<std::uint64_t>& signature)
{
    std::uint64_t hash = 1469598103934665603ull;
    for(std::uint64_t value : signature)
        hash = hashCombine(hash, value);
    return hash;
}

std::uint64_t nodeGeometryHash(const FmmTree& tree)
{
    std::uint64_t hash = 1469598103934665603ull;
    for(const FmmNode& node : tree.nodes())
    {
        hash = hashCombine(hash, node.spatialKey);
        hash = hashCombine(hash, hashDouble(node.center.x));
        hash = hashCombine(hash, hashDouble(node.center.y));
        hash = hashCombine(hash, hashDouble(node.center.z));
        hash = hashCombine(hash, hashDouble(node.halfSize));
        hash = hashCombine(hash, hashDouble(node.currentRadius));
    }
    return hash;
}

std::uint64_t geometryEnvelopeHash(const FmmTree& tree)
{
    std::uint64_t hash = 1469598103934665603ull;
    for(const FmmNode& node : tree.nodes())
    {
        hash = hashCombine(hash, node.spatialKey);
        hash = hashCombine(hash, hashDouble(node.center.x));
        hash = hashCombine(hash, hashDouble(node.center.y));
        hash = hashCombine(hash, hashDouble(node.center.z));
        hash = hashCombine(hash, hashDouble(node.halfSize));
        hash = hashCombine(hash, hashDouble(node.radius));
    }
    return hash;
}

std::vector<std::uint64_t> structuralTopologySignature(const FmmTree& tree)
{
    std::vector<std::uint64_t> signature;
    signature.reserve(2 * tree.nodes().size());
    for(const FmmNode& node : tree.nodes())
    {
        signature.push_back(node.spatialKey);
        const std::uint64_t metadata =
            static_cast<std::uint64_t>(node.childMask) |
            (static_cast<std::uint64_t>(node.isLeaf() ? 1u : 0u) << 8u) |
            (static_cast<std::uint64_t>(node.depth) << 9u);
        signature.push_back(metadata);
    }
    return signature;
}

std::vector<std::uint64_t> leafOccupancySignature(const FmmTree& tree)
{
    std::vector<std::uint64_t> signature;
    signature.reserve(2 * tree.leafCount());
    for(const FmmNode& node : tree.nodes())
    {
        if(!node.isLeaf())
            continue;
        signature.push_back(node.spatialKey);
        signature.push_back(static_cast<std::uint64_t>(node.particleCount()));
    }
    return signature;
}

std::uint64_t saturatingAddU64(std::uint64_t first, std::uint64_t second)
{
    if(first > std::numeric_limits<std::uint64_t>::max() - second)
        return std::numeric_limits<std::uint64_t>::max();
    return first + second;
}

std::uint64_t saturatingMultiplyU64(std::uint64_t first, std::uint64_t second)
{
    if(first != 0 && second > std::numeric_limits<std::uint64_t>::max() / first)
        return std::numeric_limits<std::uint64_t>::max();
    return first * second;
}

void accumulateDirectCrossPatch(const std::vector<Vector3D>& targetPositions,
                                std::vector<Vector3D>& targetAcceleration,
                                std::vector<double>* targetPotential,
                                const std::vector<Vector3D>& sourcePositions,
                                const std::vector<double>& sourceMasses)
{
    for(std::size_t targetIndex = 0; targetIndex < targetPositions.size();
        ++targetIndex)
    {
        long double ax = 0.0L;
        long double ay = 0.0L;
        long double az = 0.0L;
        long double potential = 0.0L;
        const Vector3D& target = targetPositions[targetIndex];
        for(std::size_t sourceIndex = 0; sourceIndex < sourcePositions.size();
            ++sourceIndex)
        {
            const Vector3D delta = target - sourcePositions[sourceIndex];
            const long double r2 = static_cast<long double>(delta.x) * delta.x +
                                   static_cast<long double>(delta.y) * delta.y +
                                   static_cast<long double>(delta.z) * delta.z;
            if(r2 == 0.0L)
                continue;
            const long double invR = 1.0L / std::sqrt(r2);
            const long double factor =
                static_cast<long double>(sourceMasses[sourceIndex]) * invR * invR * invR;
            ax -= factor * delta.x;
            ay -= factor * delta.y;
            az -= factor * delta.z;
            potential += static_cast<long double>(sourceMasses[sourceIndex]) * invR;
        }
        targetAcceleration[targetIndex].x += static_cast<double>(ax);
        targetAcceleration[targetIndex].y += static_cast<double>(ay);
        targetAcceleration[targetIndex].z += static_cast<double>(az);
        if(targetPotential != nullptr)
            (*targetPotential)[targetIndex] += static_cast<double>(potential);
    }
}
}

FmmPatchForest::FmmPatchForest() = default;

void FmmPatchForest::validatePrepareInputs(
    const std::vector<Vector3D>& positions,
    const std::vector<double>& masses,
    const std::vector<std::uint64_t>& cellIds,
    const Vector3D& domainLower,
    const Vector3D& domainUpper,
    const FmmGravityOptions& gravityOptions,
    const FmmDistributedOptions& distributedOptions,
    int ownerRank) const
{
    if(positions.size() != masses.size() ||
       positions.size() != cellIds.size())
        throw UniversalError("FmmPatchForest::prepare: input size mismatch");
    if(ownerRank < 0)
        throw UniversalError("FmmPatchForest::prepare: invalid owner rank");
    if(distributedOptions.maxLocalPatchCount == 0)
        throw UniversalError("FmmPatchForest::prepare: maxLocalPatchCount must be positive");
    if(!std::isfinite(domainLower.x) || !std::isfinite(domainLower.y) ||
       !std::isfinite(domainLower.z) || !std::isfinite(domainUpper.x) ||
       !std::isfinite(domainUpper.y) || !std::isfinite(domainUpper.z) ||
       !(domainLower.x < domainUpper.x) ||
       !(domainLower.y < domainUpper.y) ||
       !(domainLower.z < domainUpper.z))
        throw UniversalError("FmmPatchForest::prepare: invalid domain bounds");
    if(gravityOptions.expansionOrder < 1 ||
       gravityOptions.expansionOrder > FMM_MAX_ORDER)
        throw UniversalError("FmmPatchForest::prepare: invalid expansion order");
    if(!(gravityOptions.thetaCritical > 0.0) ||
       gravityOptions.thetaCritical > 1.0 ||
       !std::isfinite(gravityOptions.thetaCritical))
        throw UniversalError("FmmPatchForest::prepare: invalid theta");
    if(gravityOptions.leafCapacity == 0)
        throw UniversalError("FmmPatchForest::prepare: invalid leaf capacity");
    if(gravityOptions.maxDepth <= 0 ||
       gravityOptions.maxDepth > FMM_MAX_TREE_DEPTH)
        throw UniversalError("FmmPatchForest::prepare: invalid maximum tree depth");
    if(gravityOptions.maxLeafHalfSize < 0.0 ||
       !std::isfinite(gravityOptions.maxLeafHalfSize))
        throw UniversalError("FmmPatchForest::prepare: invalid maximum leaf half size");
    if(distributedOptions.persistentLocalTreeTopology &&
       (!(distributedOptions.persistentLeafSplitFactor > 1.0) ||
        !std::isfinite(distributedOptions.persistentLeafSplitFactor)))
        throw UniversalError("FmmPatchForest::prepare: invalid persistent split factor");
    if(distributedOptions.persistentLocalTreeTopology &&
       (!(distributedOptions.persistentLeafMergeFactor >= 0.0) ||
        !(distributedOptions.persistentLeafMergeFactor < 1.0) ||
        !std::isfinite(distributedOptions.persistentLeafMergeFactor)))
        throw UniversalError("FmmPatchForest::prepare: invalid persistent merge factor");
    if(distributedOptions.persistentLocalTreeTopology &&
       (!(gravityOptions.persistentRadiusSlackFactor >= 1.0) ||
        !std::isfinite(gravityOptions.persistentRadiusSlackFactor)))
        throw UniversalError(
            "FmmPatchForest::prepare: invalid persistent radius slack factor");
    if(distributedOptions.minimumPatchLevel < 0 ||
       distributedOptions.minimumPatchLevel > FMM_MAX_TREE_DEPTH ||
       distributedOptions.maximumPatchLevel <
           distributedOptions.minimumPatchLevel ||
       distributedOptions.maximumPatchLevel > FMM_MAX_TREE_DEPTH)
        throw UniversalError("FmmPatchForest::prepare: invalid patch levels");

    for(std::size_t index = 0; index < positions.size(); ++index)
    {
        const Vector3D& point = positions[index];
        if(!std::isfinite(point.x) || !std::isfinite(point.y) ||
           !std::isfinite(point.z) || !std::isfinite(masses[index]))
        {
            UniversalError error("FmmPatchForest::prepare: non-finite input");
            error.addEntry("particle", index);
            throw error;
        }
        if(point.x < domainLower.x || point.x > domainUpper.x ||
           point.y < domainLower.y || point.y > domainUpper.y ||
           point.z < domainLower.z || point.z > domainUpper.z)
        {
            UniversalError error("FmmPatchForest::prepare: point outside domain");
            error.addEntry("particle", index);
            error.addEntry("x", point.x);
            error.addEntry("y", point.y);
            error.addEntry("z", point.z);
            throw error;
        }
    }
}

FmmPatchForestChange FmmPatchForest::prepare(
    const std::vector<Vector3D>& positions,
    const std::vector<double>& masses,
    const std::vector<std::uint64_t>& cellIds,
    const Vector3D& domainLower,
    const Vector3D& domainUpper,
    const FmmGravityOptions& gravityOptions,
    const FmmDistributedOptions& distributedOptions,
    int ownerRank)
{
    validatePrepareInputs(positions, masses, cellIds, domainLower, domainUpper,
                          gravityOptions, distributedOptions, ownerRank);

    gravityOptions_ = gravityOptions;
    distributedOptions_ = distributedOptions;
    lattice_ = FmmGlobalDyadicLattice::fromDomain(domainLower, domainUpper);
    inputCount_ = positions.size();

    previousPatches_.swap(patches_);
    patches_.clear();

    std::size_t overfullPatches = 0;
    std::size_t fixedLevelPatchCount = 0;
    const std::vector<PatchBucket> buckets =
        partitionParticles(positions, distributedOptions, overfullPatches,
                           fixedLevelPatchCount);
    buildPatchObjects(buckets, positions, masses, cellIds, ownerRank);

    fixedLevelPatchCount_ = fixedLevelPatchCount;
    updateHashes();
    FmmPatchForestChange change = compareWithPrevious();
    change.overfullPatches = overfullPatches;
    updateDiagnostics(change, fixedLevelPatchCount);
    // The previous forest is needed only for the change summary. Retaining its
    // particle arrays and trees would double persistent patch-forest memory.
    std::vector<FmmLocalPatch>().swap(previousPatches_);
    return change;
}

std::vector<FmmPatchForest::PatchBucket> FmmPatchForest::partitionParticles(
    const std::vector<Vector3D>& positions,
    const FmmDistributedOptions& distributedOptions,
    std::size_t& overfullPatches,
    std::size_t& fixedLevelPatchCount) const
{
    if(distributedOptions.minimumPatchLevel < 0 ||
       distributedOptions.minimumPatchLevel > FMM_MAX_TREE_DEPTH ||
       distributedOptions.maximumPatchLevel < distributedOptions.minimumPatchLevel ||
       distributedOptions.maximumPatchLevel > FMM_MAX_TREE_DEPTH)
        throw UniversalError("FmmPatchForest::partitionParticles: invalid patch levels");
    if(lattice_.wouldOverflowLevel(distributedOptions.minimumPatchLevel))
        throw UniversalError("FmmPatchForest::partitionParticles: minimum patch level overflows");

    std::map<std::uint64_t, std::vector<std::size_t>> fixedBuckets;
    for(std::size_t index = 0; index < positions.size(); ++index)
    {
        const std::uint64_t patchId = lattice_.patchIdAtLevel(
            positions[index], distributedOptions.minimumPatchLevel);
        fixedBuckets[patchId].push_back(index);
    }

    fixedLevelPatchCount = fixedBuckets.size();

    std::vector<PatchBucket> refined;
    refined.reserve(fixedBuckets.size());
    std::size_t projectedPatchCount = 0;
    for(const auto& entry : fixedBuckets)
    {
        std::vector<PatchBucket> split = adaptiveRefine(
            entry.first,
            distributedOptions.minimumPatchLevel,
            entry.second,
            positions,
            distributedOptions,
            overfullPatches,
            projectedPatchCount);
        refined.insert(refined.end(),
                       std::make_move_iterator(split.begin()),
                       std::make_move_iterator(split.end()));
    }

    sortPatchBuckets(refined);
    if(refined.size() > distributedOptions.maxLocalPatchCount)
    {
        UniversalError error("FmmPatchForest::partitionParticles: maxLocalPatchCount exceeded");
        error.addEntry("projected_patch_count", refined.size());
        error.addEntry("max_local_patch_count", distributedOptions.maxLocalPatchCount);
        throw error;
    }
    return refined;
}

std::vector<FmmPatchForest::PatchBucket> FmmPatchForest::adaptiveRefine(
    std::uint64_t patchId,
    int level,
    std::vector<std::size_t> inputIndices,
    const std::vector<Vector3D>& positions,
    const FmmDistributedOptions& distributedOptions,
    std::size_t& overfullPatches,
    std::size_t& projectedPatchCount) const
{
    const std::size_t target = distributedOptions.targetParticlesPerPatch;
    if(target == 0 || inputIndices.size() <= target ||
       level >= distributedOptions.maximumPatchLevel)
    {
        if(target > 0 && inputIndices.size() > target &&
           level >= distributedOptions.maximumPatchLevel)
            ++overfullPatches;
        ++projectedPatchCount;
        if(projectedPatchCount > distributedOptions.maxLocalPatchCount)
        {
            UniversalError error("FmmPatchForest::adaptiveRefine: maxLocalPatchCount exceeded");
            error.addEntry("parent_patch", patchId);
            error.addEntry("level", level);
            error.addEntry("count", inputIndices.size());
            error.addEntry("target", target);
            error.addEntry("projected_patch_count", projectedPatchCount);
            error.addEntry("max_local_patch_count",
                           distributedOptions.maxLocalPatchCount);
            throw error;
        }
        PatchBucket bucket;
        bucket.patchId = patchId;
        bucket.inputIndices = std::move(inputIndices);
        return {bucket};
    }

    std::array<std::vector<std::size_t>, 8> octantBuckets;
    for(std::size_t index : inputIndices)
    {
        const int octant = lattice_.octantForPoint(patchId, positions[index]);
        octantBuckets[static_cast<std::size_t>(octant)].push_back(index);
    }

    std::vector<PatchBucket> result;
    for(int octant = 0; octant < 8; ++octant)
    {
        if(octantBuckets[static_cast<std::size_t>(octant)].empty())
            continue;
        const std::uint64_t childPatchId =
            lattice_.childPatchId(patchId, octant);
        std::vector<PatchBucket> children = adaptiveRefine(
            childPatchId,
            level + 1,
            std::move(octantBuckets[static_cast<std::size_t>(octant)]),
            positions,
            distributedOptions,
            overfullPatches,
            projectedPatchCount);
        result.insert(result.end(),
                      std::make_move_iterator(children.begin()),
                      std::make_move_iterator(children.end()));
    }
    return result;
}

void FmmPatchForest::sortPatchBuckets(std::vector<PatchBucket>& buckets) const
{
    std::sort(buckets.begin(), buckets.end(),
              [](const PatchBucket& first, const PatchBucket& second) {
                  return first.patchId < second.patchId;
              });
}

void FmmPatchForest::sortIndicesWithinPatch(FmmLocalPatch& patch) const
{
    const std::size_t count = patch.inputIndices.size();
    std::vector<std::size_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t first, std::size_t second) {
                  if(patch.cellIds[first] != patch.cellIds[second])
                      return patch.cellIds[first] < patch.cellIds[second];
                  return patch.inputIndices[first] < patch.inputIndices[second];
              });

    const auto reorder = [&](auto& values) {
        using Value = typename std::decay<decltype(values[0])>::type;
        std::vector<Value> copy(count);
        for(std::size_t i = 0; i < count; ++i)
            copy[i] = values[order[i]];
        values.swap(copy);
    };

    reorder(patch.inputIndices);
    reorder(patch.positions);
    reorder(patch.masses);
    reorder(patch.cellIds);
}

void FmmPatchForest::buildPatchObjects(
    const std::vector<PatchBucket>& buckets,
    const std::vector<Vector3D>& positions,
    const std::vector<double>& masses,
    const std::vector<std::uint64_t>& cellIds,
    int ownerRank)
{
    patches_.reserve(buckets.size());

    for(const PatchBucket& bucket : buckets)
    {
        FmmLocalPatch patch;
        patch.key.ownerRank = ownerRank;
        patch.key.patchId = bucket.patchId;
        patch.root = lattice_.patchRootGeometry(bucket.patchId);
        patch.inputIndices = bucket.inputIndices;
        patch.positions.reserve(bucket.inputIndices.size());
        patch.masses.reserve(bucket.inputIndices.size());
        patch.cellIds.reserve(bucket.inputIndices.size());
        for(std::size_t inputIndex : bucket.inputIndices)
        {
            patch.positions.push_back(positions[inputIndex]);
            patch.masses.push_back(masses[inputIndex]);
            patch.cellIds.push_back(cellIds[inputIndex]);
            if(!patch.root.contains(positions[inputIndex]))
                throw UniversalError(
                    "FmmPatchForest::buildPatchObjects: particle outside patch root");
        }
        sortIndicesWithinPatch(patch);

        patch.tree.build(patch.positions, patch.root, gravityOptions_);
        patch.structuralSignature = structuralTopologySignature(patch.tree);
        patch.structuralTreeHash =
            structuralSignatureHash(patch.structuralSignature);
        patch.occupancySignature = leafOccupancySignature(patch.tree);
        patch.nodeGeometryHash = nodeGeometryHash(patch.tree);
        patch.geometryEnvelopeHash = geometryEnvelopeHash(patch.tree);
        patch.geometryEnvelopeGeneration = 1;
        // The nonpersistent path has no retained geometry envelope.
        patch.topologyHash = patch.geometryEnvelopeHash;
        FmmDualTreeTraversal::buildLocalPlan(
            patch.tree, gravityOptions_.thetaCritical, patch.localPlan);

        patches_.push_back(std::move(patch));
    }
}

void FmmPatchForest::buildUpward(const FmmTaylorExpansion& layout)
{
    for(FmmLocalPatch& patch : patches_)
    {
        FmmPasses::allocate(patch.tree, layout, patch.multipoles, patch.locals);
        FmmPasses::upward(patch.tree, patch.positions, patch.masses, layout,
                          patch.multipoles);
    }
}

void FmmPatchForest::clearLocals(const FmmTaylorExpansion& layout)
{
    const std::size_t coeffsPerNode = layout.coefficientCount();
    for(FmmLocalPatch& patch : patches_)
    {
        patch.locals.assign(patch.tree.nodes().size() * coeffsPerNode, 0.0);
        patch.acceleration.assign(patch.positions.size(), Vector3D());
        if(gravityOptions_.computePotential)
            patch.potential.assign(patch.positions.size(), 0.0);
        else
            patch.potential.clear();
    }
}

void FmmPatchForest::executeLocalSelf(const FmmTaylorExpansion& layout,
                                      FmmM2LOperatorCache& operatorCache,
                                      std::size_t maxOperatorCacheBytes,
                                      FmmSolveStats& stats)
{
    for(FmmLocalPatch& patch : patches_)
    {
        FmmSolveStats patchStats;
        std::vector<double>* potential =
            gravityOptions_.computePotential ? &patch.potential : nullptr;
        FmmDualTreeTraversal::runLocalPlan(
            patch.tree, patch.localPlan, patch.positions, patch.masses, layout,
            patch.multipoles, patch.locals, patch.acceleration, potential,
            operatorCache, maxOperatorCacheBytes, patchStats);
        stats.m2lCount += patchStats.m2lCount;
        stats.p2pBlockCount += patchStats.p2pBlockCount;
        stats.p2pPairCount += patchStats.p2pPairCount;
        stats.rejectedSameNode += patchStats.rejectedSameNode;
        stats.rejectedOverlap += patchStats.rejectedOverlap;
        stats.rejectedRatio += patchStats.rejectedRatio;
        stats.maxTraversalStack =
            std::max(stats.maxTraversalStack, patchStats.maxTraversalStack);
    }
}

void FmmPatchForest::executeDirectCrossPatch(const FmmTaylorExpansion& layout,
                                             FmmSolveStats& stats)
{
    (void)layout;
    for(std::size_t targetIndex = 0; targetIndex < patches_.size(); ++targetIndex)
    {
        for(std::size_t sourceIndex = 0; sourceIndex < patches_.size();
            ++sourceIndex)
        {
            if(targetIndex == sourceIndex)
                continue;
            FmmLocalPatch& target = patches_[targetIndex];
            const FmmLocalPatch& source = patches_[sourceIndex];
            std::vector<double>* potential =
                gravityOptions_.computePotential ? &target.potential : nullptr;
            accumulateDirectCrossPatch(target.positions, target.acceleration,
                                       potential, source.positions,
                                       source.masses);
            const std::uint64_t pairCount = saturatingMultiplyU64(
                static_cast<std::uint64_t>(target.positions.size()),
                static_cast<std::uint64_t>(source.positions.size()));
            stats.p2pPairCount = saturatingAddU64(stats.p2pPairCount, pairCount);
            ++stats.p2pBlockCount;
        }
    }
}

void FmmPatchForest::applyDownward(const FmmTaylorExpansion& layout)
{
    for(FmmLocalPatch& patch : patches_)
    {
        std::vector<double>* potential =
            gravityOptions_.computePotential ? &patch.potential : nullptr;
        FmmPasses::downward(patch.tree, patch.positions, layout, patch.locals,
                            patch.acceleration, potential);
    }
}

void FmmPatchForest::scatterAcceleration(
    std::vector<Vector3D>& acceleration) const
{
    if(acceleration.size() != inputCount_)
        throw UniversalError("FmmPatchForest::scatterAcceleration: output size mismatch");
    for(const FmmLocalPatch& patch : patches_)
    {
        for(std::size_t localIndex = 0; localIndex < patch.inputIndices.size();
            ++localIndex)
        {
            const std::size_t globalIndex = patch.inputIndices[localIndex];
            if(globalIndex >= acceleration.size())
                throw UniversalError(
                    "FmmPatchForest::scatterAcceleration: input index out of range");
            acceleration[globalIndex] = patch.acceleration[localIndex];
        }
    }
}

void FmmPatchForest::scatterPotential(std::vector<double>& potential) const
{
    if(potential.size() != inputCount_)
        throw UniversalError("FmmPatchForest::scatterPotential: output size mismatch");
    for(const FmmLocalPatch& patch : patches_)
    {
        for(std::size_t localIndex = 0; localIndex < patch.inputIndices.size();
            ++localIndex)
        {
            const std::size_t globalIndex = patch.inputIndices[localIndex];
            if(globalIndex >= potential.size())
                throw UniversalError(
                    "FmmPatchForest::scatterPotential: input index out of range");
            potential[globalIndex] = patch.potential[localIndex];
        }
    }
}

std::vector<FmmPatchRootDescriptor> FmmPatchForest::descriptors(
    int ownerRank,
    std::uint64_t topologyEpoch) const
{
    std::vector<FmmPatchRootDescriptor> result;
    result.reserve(patches_.size());
    for(const FmmLocalPatch& patch : patches_)
    {
        FmmPatchRootDescriptor descriptor;
        descriptor.ownerRank = ownerRank;
        descriptor.patchId = patch.key.patchId;
        descriptor.active = patch.tree.nodes().empty() ? 0 : 1;
        descriptor.topologyHash = patch.topologyHash;
        descriptor.epoch = topologyEpoch;
        if(descriptor.active != 0)
        {
            const FmmNode& root = patch.tree.nodes()[0];
            descriptor.center[0] = root.center.x;
            descriptor.center[1] = root.center.y;
            descriptor.center[2] = root.center.z;
            descriptor.halfSize = root.halfSize;
            descriptor.radius = root.radius;
            descriptor.particleCount =
                static_cast<std::uint64_t>(root.particleCount());
            descriptor.latticeId = root.latticeId;
            descriptor.latticeCenter[0] = root.latticeCenterX;
            descriptor.latticeCenter[1] = root.latticeCenterY;
            descriptor.latticeCenter[2] = root.latticeCenterZ;
            descriptor.latticeHalfUnits = root.latticeHalfUnits;
            descriptor.rootLeaf = root.isLeaf() ? 1 : 0;
            descriptor.childMask = static_cast<int>(root.childMask);
        }
        result.push_back(descriptor);
    }
    std::sort(result.begin(), result.end(),
              [](const FmmPatchRootDescriptor& first,
                 const FmmPatchRootDescriptor& second) {
                  if(first.patchId != second.patchId)
                      return first.patchId < second.patchId;
                  return first.ownerRank < second.ownerRank;
              });
    return result;
}

std::size_t FmmPatchForest::findPatch(std::uint64_t patchId) const
{
    const auto it = std::lower_bound(
        patches_.begin(), patches_.end(), patchId,
        [](const FmmLocalPatch& patch, std::uint64_t id) {
            return patch.key.patchId < id;
        });
    if(it == patches_.end() || it->key.patchId != patchId)
        return std::numeric_limits<std::size_t>::max();
    return static_cast<std::size_t>(std::distance(patches_.begin(), it));
}

void FmmPatchForest::updateHashes()
{
    geometryHash_ = 1469598103934665603ull;
    structuralHash_ = 1469598103934665603ull;
    occupancyHash_ = 1469598103934665603ull;
    for(const FmmLocalPatch& patch : patches_)
    {
        geometryHash_ = hashCombine(geometryHash_, patch.key.patchId);
        geometryHash_ = hashCombine(geometryHash_, hashDouble(patch.root.center.x));
        geometryHash_ = hashCombine(geometryHash_, hashDouble(patch.root.center.y));
        geometryHash_ = hashCombine(geometryHash_, hashDouble(patch.root.center.z));
        geometryHash_ = hashCombine(geometryHash_, hashDouble(patch.root.halfSize));
        geometryHash_ = hashCombine(geometryHash_, patch.nodeGeometryHash);

        structuralHash_ = hashCombine(structuralHash_, patch.key.patchId);
        structuralHash_ = hashCombine(structuralHash_, patch.structuralTreeHash);
        for(std::uint64_t value : patch.structuralSignature)
            structuralHash_ = hashCombine(structuralHash_, value);

        occupancyHash_ = hashCombine(occupancyHash_, patch.key.patchId);
        for(std::uint64_t value : patch.occupancySignature)
            occupancyHash_ = hashCombine(occupancyHash_, value);
    }
}

FmmPatchForestChange FmmPatchForest::compareWithPrevious() const
{
    FmmPatchForestChange change;
    std::unordered_map<std::uint64_t, const FmmLocalPatch*> previousById;
    previousById.reserve(previousPatches_.size());
    for(const FmmLocalPatch& patch : previousPatches_)
        previousById.emplace(patch.key.patchId, &patch);

    std::unordered_map<std::uint64_t, const FmmLocalPatch*> currentById;
    currentById.reserve(patches_.size());
    for(const FmmLocalPatch& patch : patches_)
        currentById.emplace(patch.key.patchId, &patch);

    for(const auto& entry : currentById)
    {
        const auto previous = previousById.find(entry.first);
        if(previous == previousById.end())
        {
            ++change.createdPatches;
            change.patchSetChanged = true;
            continue;
        }
        ++change.matchedPatchIds;
        const FmmLocalPatch& current = *entry.second;
        const FmmLocalPatch& prior = *previous->second;
        const bool structureSame =
            current.structuralTreeHash == prior.structuralTreeHash &&
            current.structuralSignature == prior.structuralSignature;
        const bool nodeGeometrySame =
            current.nodeGeometryHash == prior.nodeGeometryHash;
        const bool geometryEnvelopeSame =
            current.geometryEnvelopeHash == prior.geometryEnvelopeHash;
        const bool occupancyDiff =
            current.occupancySignature != prior.occupancySignature;

        if(current.root.center.x != prior.root.center.x ||
           current.root.center.y != prior.root.center.y ||
           current.root.center.z != prior.root.center.z ||
           current.root.halfSize != prior.root.halfSize)
            change.patchGeometryChanged = true;
        if(!structureSame)
            change.structuralTopologyChanged = true;
        if(!nodeGeometrySame)
            change.nodeGeometryChanged = true;
        if(!geometryEnvelopeSame)
            change.geometryEnvelopeChanged = true;
        if(occupancyDiff)
            change.occupancyChanged = true;
    }

    for(const auto& entry : previousById)
    {
        if(currentById.find(entry.first) == currentById.end())
        {
            ++change.removedPatches;
            change.patchSetChanged = true;
        }
    }

    if(change.createdPatches > 0 || change.removedPatches > 0)
        change.patchSetChanged = true;
    change.countOnlyChanged = change.occupancyChanged &&
        !change.patchSetChanged && !change.patchGeometryChanged &&
        !change.structuralTopologyChanged && !change.nodeGeometryChanged &&
        !change.geometryEnvelopeChanged;
    return change;
}

void FmmPatchForest::updateDiagnostics(const FmmPatchForestChange& change,
                                       std::size_t fixedLevelPatchCount)
{
    diagnostics_ = FmmPatchForestDiagnostics();
    diagnostics_.patchCount = patches_.size();
    diagnostics_.fixedLevelPatchCount = fixedLevelPatchCount;
    diagnostics_.createdPatches = change.createdPatches;
    diagnostics_.removedPatches = change.removedPatches;
    diagnostics_.matchedPatchIds = change.matchedPatchIds;
    diagnostics_.overfullPatches = change.overfullPatches;
    diagnostics_.levelHistogram.assign(static_cast<std::size_t>(FMM_MAX_TREE_DEPTH) + 1, 0);

    std::vector<std::size_t> counts;
    counts.reserve(patches_.size());
    diagnostics_.copiedInputBytes = 0;
    diagnostics_.patchTreeBytes = 0;
    for(const FmmLocalPatch& patch : patches_)
    {
        const std::size_t particleCount = patch.positions.size();
        counts.push_back(particleCount);
        const int level = lattice_.patchLevel(patch.key.patchId);
        if(level >= 0 && level <= FMM_MAX_TREE_DEPTH)
            ++diagnostics_.levelHistogram[static_cast<std::size_t>(level)];
        diagnostics_.copiedInputBytes +=
            particleCount * (sizeof(Vector3D) + sizeof(double) + sizeof(std::uint64_t));
        diagnostics_.patchTreeBytes += patch.tree.bytesOwned();
        diagnostics_.largestPatchRadius =
            std::max(diagnostics_.largestPatchRadius,
                     patch.tree.nodes().empty() ? 0.0 : patch.tree.nodes()[0].radius);
    }

    if(!counts.empty())
    {
        std::sort(counts.begin(), counts.end());
        diagnostics_.particlesPerPatchMin = counts.front();
        diagnostics_.particlesPerPatchMax = counts.back();
        diagnostics_.particlesPerPatchMedian = counts[counts.size() / 2];
        const std::size_t p95Index =
            std::min(counts.size() - 1, (counts.size() * 95) / 100);
        diagnostics_.particlesPerPatchP95 = counts[p95Index];
    }
}

#endif // RICH_MPI
