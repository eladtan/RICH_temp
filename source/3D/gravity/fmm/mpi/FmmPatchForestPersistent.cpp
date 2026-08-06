#include "3D/gravity/fmm/mpi/FmmPatchForest.hpp"

#ifdef RICH_MPI

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <utility>

#include "misc/universal_error.hpp"

namespace
{
std::size_t saturatingAdd(std::size_t first, std::size_t second)
{
    return second > std::numeric_limits<std::size_t>::max() - first ?
        std::numeric_limits<std::size_t>::max() : first + second;
}

std::size_t saturatingMultiply(std::size_t first, std::size_t second)
{
    return first != 0 && second >
        std::numeric_limits<std::size_t>::max() / first ?
        std::numeric_limits<std::size_t>::max() : first * second;
}


std::uint64_t hashCombine(std::uint64_t seed, std::uint64_t value)
{
    return seed ^ (value + 0x9e3779b97f4a7c15ull +
                   (seed << 6) + (seed >> 2));
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

std::uint64_t nextTopologyGeneration(std::uint64_t previous)
{
    if(previous == std::numeric_limits<std::uint64_t>::max())
        throw UniversalError(
            "FmmPatchForest::preparePersistent: topology generation overflow");
    return previous + 1;
}

std::size_t patchPersistentBytes(const FmmLocalPatch& patch)
{
    std::size_t bytes = patch.tree.bytesOwned();
    bytes = saturatingAdd(bytes, patch.localPlan.bytesOwned());
    bytes = saturatingAdd(bytes, saturatingMultiply(
        patch.inputIndices.capacity(), sizeof(std::size_t)));
    bytes = saturatingAdd(bytes, saturatingMultiply(
        patch.positions.capacity(), sizeof(Vector3D)));
    bytes = saturatingAdd(bytes, saturatingMultiply(
        patch.masses.capacity(), sizeof(double)));
    bytes = saturatingAdd(bytes, saturatingMultiply(
        patch.cellIds.capacity(), sizeof(std::uint64_t)));
    bytes = saturatingAdd(bytes, saturatingMultiply(
        patch.multipoles.capacity(), sizeof(double)));
    bytes = saturatingAdd(bytes, saturatingMultiply(
        patch.locals.capacity(), sizeof(double)));
    bytes = saturatingAdd(bytes, saturatingMultiply(
        patch.acceleration.capacity(), sizeof(Vector3D)));
    bytes = saturatingAdd(bytes, saturatingMultiply(
        patch.potential.capacity(), sizeof(double)));
    bytes = saturatingAdd(bytes, saturatingMultiply(
        patch.structuralSignature.capacity(), sizeof(std::uint64_t)));
    bytes = saturatingAdd(bytes, saturatingMultiply(
        patch.occupancySignature.capacity(), sizeof(std::uint64_t)));
    return bytes;
}

std::vector<std::uint64_t> structuralTopologySignature(const FmmTree& tree)
{
    std::vector<std::uint64_t> signature;
    if(tree.nodes().size() > std::numeric_limits<std::size_t>::max() / 2)
        throw UniversalError(
            "FmmPatchForest::preparePersistent: structural signature overflow");
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
    if(tree.leafCount() > std::numeric_limits<std::size_t>::max() / 2)
        throw UniversalError(
            "FmmPatchForest::preparePersistent: occupancy signature overflow");
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

bool sameRoot(const FmmRootGeometry& first, const FmmRootGeometry& second)
{
    return first.active == second.active &&
           first.center.x == second.center.x &&
           first.center.y == second.center.y &&
           first.center.z == second.center.z &&
           first.halfSize == second.halfSize &&
           first.latticeId == second.latticeId &&
           first.latticeCenterX == second.latticeCenterX &&
           first.latticeCenterY == second.latticeCenterY &&
           first.latticeCenterZ == second.latticeCenterZ &&
           first.latticeHalfUnits == second.latticeHalfUnits &&
           first.latticeAligned == second.latticeAligned;
}

bool radiusExpanded(double current, double previous)
{
    const double tolerance =
        64.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::max(std::abs(current), std::abs(previous)));
    return current > previous + tolerance;
}

bool anyGeometryEnvelopeExpanded(
    const FmmTree& tree,
    const FmmLocalInteractionPlan& previousPlan)
{
    if(previousPlan.nodeGeometry.size() != tree.nodes().size())
        return true;
    for(std::size_t index = 0; index < tree.nodes().size(); ++index)
    {
        const FmmNode& current = tree.nodes()[index];
        const FmmLocalInteractionPlan::NodeGeometry& previous =
            previousPlan.nodeGeometry[index];
        if(current.center.x != previous.center.x ||
           current.center.y != previous.center.y ||
           current.center.z != previous.center.z ||
           current.halfSize != previous.halfSize ||
           current.isLeaf() != previous.leaf ||
           radiusExpanded(current.radius, previous.radius))
            return true;
    }
    return false;
}

std::size_t persistentSplitCapacity(std::size_t leafCapacity, double factor)
{
    const long double scaled = std::ceil(
        static_cast<long double>(leafCapacity) *
        static_cast<long double>(factor));
    if(!std::isfinite(static_cast<double>(scaled)) ||
       scaled > static_cast<long double>(
           std::numeric_limits<std::size_t>::max()))
        throw UniversalError(
            "FmmPatchForest::preparePersistent: split capacity overflow");
    std::size_t result = static_cast<std::size_t>(scaled);
    if(result <= leafCapacity)
    {
        if(leafCapacity == std::numeric_limits<std::size_t>::max())
            throw UniversalError(
                "FmmPatchForest::preparePersistent: split capacity overflow");
        result = leafCapacity + 1;
    }
    return result;
}

std::size_t persistentMergeCapacity(std::size_t leafCapacity, double factor)
{
    const long double scaled = std::floor(
        static_cast<long double>(leafCapacity) *
        static_cast<long double>(factor));
    if(scaled < 0.0L || scaled > static_cast<long double>(
           std::numeric_limits<std::size_t>::max()))
        throw UniversalError(
            "FmmPatchForest::preparePersistent: merge capacity overflow");
    return static_cast<std::size_t>(scaled);
}
}

FmmPatchForestChange FmmPatchForest::preparePersistent(
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
    const std::vector<PatchBucket> buckets = partitionParticles(
        positions, distributedOptions, overfullPatches, fixedLevelPatchCount);

    std::unordered_map<std::uint64_t, std::size_t> previousIndexById;
    previousIndexById.reserve(previousPatches_.size());
    for(std::size_t index = 0; index < previousPatches_.size(); ++index)
    {
        if(!previousIndexById.emplace(
                previousPatches_[index].key.patchId, index).second)
            throw UniversalError(
                "FmmPatchForest::preparePersistent: duplicate previous patch ID");
    }
    std::vector<unsigned char> previousMatched(previousPatches_.size(), 0);

    const std::size_t splitCapacity =
        distributedOptions.persistentLocalTreeTopology ?
        persistentSplitCapacity(gravityOptions.leafCapacity,
                                distributedOptions.persistentLeafSplitFactor) :
        gravityOptions.leafCapacity;
    const std::size_t mergeCapacity =
        distributedOptions.persistentLocalTreeTopology ?
        persistentMergeCapacity(gravityOptions.leafCapacity,
                                distributedOptions.persistentLeafMergeFactor) :
        0;

    FmmPatchForestChange change;
    change.overfullPatches = overfullPatches;
    patches_.reserve(buckets.size());
    for(const PatchBucket& bucket : buckets)
    {
        const auto previousIt = previousIndexById.find(bucket.patchId);
        const bool matched = previousIt != previousIndexById.end();
        FmmLocalPatch patch;
        FmmRootGeometry previousRoot;
        double previousRootRadius = 0.0;
        std::uint64_t previousStructuralTreeHash = 0;
        std::uint64_t previousNodeGeometryHash = 0;
        std::uint64_t previousGeometryEnvelopeHash = 0;
        std::uint64_t previousGeometryEnvelopeGeneration = 0;
        std::uint64_t previousTopologyGeneration = 0;
        if(matched)
        {
            const std::size_t previousIndex = previousIt->second;
            previousMatched[previousIndex] = 1;
            patch = std::move(previousPatches_[previousIndex]);
            previousRoot = patch.root;
            previousStructuralTreeHash = patch.structuralTreeHash;
            previousNodeGeometryHash = patch.nodeGeometryHash;
            previousGeometryEnvelopeHash = patch.geometryEnvelopeHash;
            previousGeometryEnvelopeGeneration =
                patch.geometryEnvelopeGeneration;
            previousTopologyGeneration = patch.topologyHash;
            if(!patch.tree.nodes().empty())
                previousRootRadius = patch.tree.nodes()[0].radius;
            ++change.matchedPatchIds;
            ++change.reusedPatches;
            change.retainedBytes = saturatingAdd(
                change.retainedBytes, patchPersistentBytes(patch));
        }
        else
        {
            ++change.createdPatches;
        }

        patch.key.ownerRank = ownerRank;
        patch.key.patchId = bucket.patchId;
        patch.root = lattice_.patchRootGeometry(bucket.patchId);
        patch.inputIndices = bucket.inputIndices;
        patch.positions.clear();
        patch.masses.clear();
        patch.cellIds.clear();
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
                    "FmmPatchForest::preparePersistent: particle outside patch root");
        }
        sortIndicesWithinPatch(patch);

        const bool rootUnchanged = matched && sameRoot(previousRoot, patch.root);
        FmmPersistentTreeStats persistentStats;
        if(distributedOptions.persistentLocalTreeTopology && rootUnchanged)
        {
            patch.tree.buildPersistent(
                patch.positions, patch.root, gravityOptions, splitCapacity,
                mergeCapacity, false, persistentStats);
            patch.persistentTreeRefit = true;
        }
        else if(distributedOptions.persistentLocalTreeTopology)
        {
            patch.tree.buildPersistent(
                patch.positions, patch.root, gravityOptions, splitCapacity,
                mergeCapacity, true, persistentStats);
            patch.persistentTreeRefit = false;
        }
        else
        {
            patch.tree.build(patch.positions, patch.root, gravityOptions);
            patch.persistentTreeRefit = false;
        }
        change.persistentLeafSplits += persistentStats.leafSplits;
        change.persistentSubtreeMerges += persistentStats.subtreeMerges;
        change.persistentEmptyLeaves += persistentStats.emptyLeaves;

        std::vector<std::uint64_t> currentStructural =
            structuralTopologySignature(patch.tree);
        const std::uint64_t currentStructuralTreeHash =
            structuralSignatureHash(currentStructural);
        std::vector<std::uint64_t> currentOccupancy =
            leafOccupancySignature(patch.tree);
        const bool structureSame = matched &&
            previousStructuralTreeHash == currentStructuralTreeHash &&
            patch.structuralSignature == currentStructural;
        const bool occupancySame = matched &&
            patch.occupancySignature == currentOccupancy;
        const std::uint64_t currentNodeGeometryHash =
            nodeGeometryHash(patch.tree);
        const std::uint64_t currentGeometryEnvelopeHash =
            geometryEnvelopeHash(patch.tree);
        const bool nodeGeometryChanged = matched && structureSame &&
            previousNodeGeometryHash != currentNodeGeometryHash;
        const bool geometryEnvelopeChanged = matched && structureSame &&
            anyGeometryEnvelopeExpanded(patch.tree, patch.localPlan);
        const bool rootRadiusExpanded = matched && rootUnchanged &&
            !patch.tree.nodes().empty() &&
            radiusExpanded(patch.tree.nodes()[0].radius, previousRootRadius);

        patch.rootGeometryChanged = !rootUnchanged || rootRadiusExpanded;
        patch.leafTopologyChanged = !structureSame;
        patch.nodeGeometryChanged = nodeGeometryChanged;
        patch.geometryEnvelopeChanged = geometryEnvelopeChanged;
        patch.leafOccupancyChanged = !occupancySame;
        patch.nodeGeometryHash = currentNodeGeometryHash;
        // In nonpersistent mode a contraction can reuse the previous, larger
        // local/remote planning geometry even though the freshly built tree is
        // tighter. Preserve the retained envelope identity until an expansion
        // or structural/root change replaces it.
        patch.geometryEnvelopeHash = matched && structureSame &&
            !geometryEnvelopeChanged ? previousGeometryEnvelopeHash :
                                       currentGeometryEnvelopeHash;
        if(geometryEnvelopeChanged)
            ++change.nodeGeometryExpansionPatches;

        patch.structuralTreeHash = currentStructuralTreeHash;
        // The wire-visible value is a collision-free generation counter within
        // one patch lifetime. Radius contraction is conservative and keeps the
        // previous generation; any structural/root change or radius expansion
        // advances it and invalidates dependent cached target subplans.
        const bool descriptorTopologyChanged = !rootUnchanged ||
            !structureSame || geometryEnvelopeChanged;
        if(!matched)
            patch.geometryEnvelopeGeneration = 1;
        else if(descriptorTopologyChanged)
            patch.geometryEnvelopeGeneration = nextTopologyGeneration(
                previousGeometryEnvelopeGeneration);
        else
            patch.geometryEnvelopeGeneration =
                previousGeometryEnvelopeGeneration;
        if(!matched)
            patch.topologyHash = 1;
        else if(descriptorTopologyChanged)
            patch.topologyHash = nextTopologyGeneration(
                previousTopologyGeneration);
        else
            patch.topologyHash = previousTopologyGeneration;
        patch.structuralSignature.swap(currentStructural);
        patch.occupancySignature.swap(currentOccupancy);

        patch.localPlanReused = matched && structureSame &&
            FmmDualTreeTraversal::localPlanReusable(
                patch.tree, patch.localPlan);
        if(patch.localPlanReused)
        {
            ++change.reusedLocalPlans;
        }
        else
        {
            FmmDualTreeTraversal::buildLocalPlan(
                patch.tree, gravityOptions.thetaCritical, patch.localPlan);
            ++change.rebuiltLocalPlans;
        }

        patch.multipoles.clear();
        patch.locals.clear();
        patch.acceleration.clear();
        patch.potential.clear();

        change.patchGeometryChanged = change.patchGeometryChanged ||
            patch.rootGeometryChanged;
        change.structuralTopologyChanged =
            change.structuralTopologyChanged || patch.leafTopologyChanged;
        change.nodeGeometryChanged =
            change.nodeGeometryChanged || patch.nodeGeometryChanged;
        change.geometryEnvelopeChanged =
            change.geometryEnvelopeChanged || patch.geometryEnvelopeChanged;
        change.occupancyChanged =
            change.occupancyChanged || patch.leafOccupancyChanged;
        patches_.push_back(std::move(patch));
    }

    for(std::size_t index = 0; index < previousPatches_.size(); ++index)
    {
        if(previousMatched[index] != 0)
            continue;
        ++change.removedPatches;
        change.releasedBytes = saturatingAdd(
            change.releasedBytes, patchPersistentBytes(previousPatches_[index]));
    }
    change.patchSetChanged =
        change.createdPatches != 0 || change.removedPatches != 0;
    if(change.patchSetChanged)
    {
        change.structuralTopologyChanged = true;
        change.occupancyChanged = true;
    }
    change.countOnlyChanged = change.occupancyChanged &&
        !change.patchSetChanged && !change.patchGeometryChanged &&
        !change.structuralTopologyChanged && !change.geometryEnvelopeChanged;
    // Tight current geometry may move while remaining inside the retained
    // planning envelope. Such motion is deliberately compatible with
    // count-only topology reuse.

    std::vector<FmmLocalPatch>().swap(previousPatches_);
    fixedLevelPatchCount_ = fixedLevelPatchCount;
    updateHashes();
    updateDiagnostics(change, fixedLevelPatchCount);
    diagnostics_.reusedPatches = change.reusedPatches;
    diagnostics_.reusedLocalPlans = change.reusedLocalPlans;
    diagnostics_.rebuiltLocalPlans = change.rebuiltLocalPlans;
    diagnostics_.nodeGeometryExpansionPatches =
        change.nodeGeometryExpansionPatches;
    diagnostics_.retainedBytes = change.retainedBytes;
    diagnostics_.releasedBytes = change.releasedBytes;
    return change;
}

#endif // RICH_MPI
