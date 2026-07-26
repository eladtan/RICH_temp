#include "3D/gravity/fmm/mpi/FmmPatchForest.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
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
    if(positions.size() != masses.size() ||
       positions.size() != cellIds.size())
        throw UniversalError("FmmPatchForest::prepare: input size mismatch");

    gravityOptions_ = gravityOptions;
    distributedOptions_ = distributedOptions;
    lattice_ = FmmGlobalDyadicLattice::fromDomain(domainLower, domainUpper);

    previousPatches_.swap(patches_);
    patches_.clear();
    patchIdToIndex_.clear();

  std::size_t overfullPatches = 0;
    const std::vector<PatchBucket> buckets =
        partitionParticles(positions, distributedOptions, overfullPatches);
    buildPatchObjects(buckets, positions, masses, cellIds, ownerRank);

    updateHashes();
    updateDiagnostics();
    diagnostics_.overfullPatches = overfullPatches;

    FmmPatchForestChange change = compareWithPrevious();
    change.overfullPatches = overfullPatches;
    return change;
}

std::vector<FmmPatchForest::PatchBucket> FmmPatchForest::partitionParticles(
    const std::vector<Vector3D>& positions,
    const FmmDistributedOptions& distributedOptions,
    std::size_t& overfullPatches) const
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
    patchIdToIndex_.clear();

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
        patch.topologyHash = patch.tree.topologyHash();
        patch.structuralSignature = structuralTopologySignature(patch.tree);
        patch.occupancySignature = leafOccupancySignature(patch.tree);
        FmmDualTreeTraversal::buildLocalPlan(
            patch.tree, gravityOptions_.thetaCritical, patch.localPlan);

        patches_.push_back(std::move(patch));
    }

    std::uint64_t maxPatchId = 0;
    for(const FmmLocalPatch& patch : patches_)
        maxPatchId = std::max(maxPatchId, patch.key.patchId);
    patchIdToIndex_.assign(static_cast<std::size_t>(maxPatchId) + 1,
                           std::numeric_limits<std::size_t>::max());
    for(std::size_t index = 0; index < patches_.size(); ++index)
        patchIdToIndex_[patches_[index].key.patchId] = index;
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
            stats.p2pPairCount += target.positions.size() * source.positions.size();
            ++stats.p2pBlockCount;
        }
    }
}

void FmmPatchForest::applyDownward(const FmmTaylorExpansion& layout,
                                   bool computePotential)
{
    for(FmmLocalPatch& patch : patches_)
    {
        std::vector<double>* potential =
            computePotential ? &patch.potential : nullptr;
        FmmPasses::downward(patch.tree, patch.positions, layout, patch.locals,
                            patch.acceleration, potential);
    }
}

void FmmPatchForest::scatterAcceleration(
    std::vector<Vector3D>& acceleration) const
{
    for(const FmmLocalPatch& patch : patches_)
    {
        for(std::size_t localIndex = 0; localIndex < patch.inputIndices.size();
            ++localIndex)
        {
            const std::size_t globalIndex = patch.inputIndices[localIndex];
            acceleration[globalIndex] = patch.acceleration[localIndex];
        }
    }
}

void FmmPatchForest::scatterPotential(std::vector<double>& potential) const
{
    for(const FmmLocalPatch& patch : patches_)
    {
        for(std::size_t localIndex = 0; localIndex < patch.inputIndices.size();
            ++localIndex)
        {
            const std::size_t globalIndex = patch.inputIndices[localIndex];
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
    if(patchId >= patchIdToIndex_.size())
        return std::numeric_limits<std::size_t>::max();
    return patchIdToIndex_[patchId];
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

        structuralHash_ = hashCombine(structuralHash_, patch.key.patchId);
        structuralHash_ = hashCombine(structuralHash_, patch.topologyHash);
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
        ++change.reusedPatches;
        const FmmLocalPatch& current = *entry.second;
        const FmmLocalPatch& prior = *previous->second;
        if(current.root.center.x != prior.root.center.x ||
           current.root.center.y != prior.root.center.y ||
           current.root.center.z != prior.root.center.z ||
           current.root.halfSize != prior.root.halfSize)
        {
            change.patchGeometryChanged = true;
        }
        if(current.topologyHash != prior.topologyHash ||
           current.structuralSignature != prior.structuralSignature)
            change.structuralTopologyChanged = true;
        if(current.occupancySignature != prior.occupancySignature)
            change.occupancyChanged = true;
        else if(current.inputIndices.size() != prior.inputIndices.size())
            change.countOnlyChanged = true;
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
    return change;
}

void FmmPatchForest::updateDiagnostics()
{
    diagnostics_ = FmmPatchForestDiagnostics();
    diagnostics_.patchCount = patches_.size();
    diagnostics_.fixedLevelPatchCount = patches_.size();
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
