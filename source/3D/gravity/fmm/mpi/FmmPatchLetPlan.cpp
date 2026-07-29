#include "3D/gravity/fmm/mpi/FmmPatchLetPlan.hpp"

#ifdef RICH_MPI

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <string>

#include "3D/gravity/fmm/FmmKernels.hpp"
#include "misc/universal_error.hpp"

namespace
{
typedef std::chrono::steady_clock Clock;

double elapsed(const Clock::time_point& start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double nodeRadius(const FmmNode& node)
{
    return node.radius > 0.0 ? node.radius :
                               std::sqrt(3.0) * node.halfSize;
}

double descriptorRadius(const FmmRemoteNodeDescriptor& descriptor)
{
    return descriptor.radius > 0.0 ? descriptor.radius :
                                     std::sqrt(3.0) * descriptor.halfSize;
}

bool finiteVector(const Vector3D& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

int bitCount(unsigned int value)
{
    int count = 0;
    while(value != 0)
    {
        count += static_cast<int>(value & 1u);
        value >>= 1u;
    }
    return count;
}

bool validRemoteDescriptor(const FmmRemoteNodeDescriptor& descriptor)
{
    const Vector3D center = descriptor.centerVector();
    const double cubeRadius = std::sqrt(3.0) * descriptor.halfSize;
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, cubeRadius);
    return finiteVector(center) && descriptor.halfSize > 0.0 &&
        std::isfinite(descriptor.halfSize) && descriptor.radius >= 0.0 &&
        std::isfinite(descriptor.radius) &&
        descriptor.radius <= cubeRadius + tolerance &&
        descriptor.spatialKey != 0 && descriptor.patchId != 0 &&
        descriptor.sourceRank >= 0 &&
        // Persistent full-octant trees deliberately expose empty leaf nodes so
        // a later occupancy-only solve can reuse the same interaction plan.
        // Empty internal nodes are never a valid retained topology state.
        (descriptor.particleCount != 0 || descriptor.isLeaf != 0) &&
        (descriptor.isLeaf == 0 || descriptor.isLeaf == 1) &&
        descriptor.childMask >= 0 && descriptor.childMask <= 255 &&
        ((descriptor.isLeaf != 0 && descriptor.childMask == 0) ||
         (descriptor.isLeaf == 0 && descriptor.childMask != 0));
}

std::int64_t shiftCoordinate(std::int64_t center,
                             std::uint64_t offset,
                             bool positive)
{
    if(offset > static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max()))
        throw UniversalError("FmmPatchLetPlan: lattice offset overflow");
    const std::int64_t delta = static_cast<std::int64_t>(offset);
    if(positive)
    {
        if(center > std::numeric_limits<std::int64_t>::max() - delta)
            throw UniversalError("FmmPatchLetPlan: lattice center overflow");
        return center + delta;
    }
    if(center < std::numeric_limits<std::int64_t>::min() + delta)
        throw UniversalError("FmmPatchLetPlan: lattice center overflow");
    return center - delta;
}

void applyRemoteLatticeMetadata(std::uint64_t latticeId,
                                const std::int64_t rootCenter[3],
                                std::uint64_t rootHalfUnits,
                                std::uint64_t spatialKey,
                                FmmNode& node)
{
    if(latticeId == 0 || rootHalfUnits == 0 || spatialKey == 0)
        throw UniversalError("FmmPatchLetPlan: invalid remote lattice metadata");

    std::array<unsigned int, FMM_MAX_TREE_DEPTH> reversed{};
    std::size_t depth = 0;
    std::uint64_t cursor = spatialKey;
    while(cursor != 1)
    {
        if(cursor == 0 || depth >= reversed.size())
            throw UniversalError("FmmPatchLetPlan: malformed remote spatial key");
        reversed[depth++] = static_cast<unsigned int>(cursor & 7u);
        cursor >>= 3u;
    }

    std::int64_t coordinates[3] = {
        rootCenter[0], rootCenter[1], rootCenter[2]};
    std::uint64_t halfUnits = rootHalfUnits;
    for(std::size_t reverse = depth; reverse > 0; --reverse)
    {
        if(halfUnits < 2 || (halfUnits & 1u) != 0)
            throw UniversalError(
                "FmmPatchLetPlan: indivisible remote lattice root");
        halfUnits /= 2;
        const unsigned int octant = reversed[reverse - 1];
        coordinates[0] = shiftCoordinate(coordinates[0], halfUnits,
                                         (octant & 4u) != 0);
        coordinates[1] = shiftCoordinate(coordinates[1], halfUnits,
                                         (octant & 2u) != 0);
        coordinates[2] = shiftCoordinate(coordinates[2], halfUnits,
                                         (octant & 1u) != 0);
    }

    node.latticeId = latticeId;
    node.latticeCenterX = coordinates[0];
    node.latticeCenterY = coordinates[1];
    node.latticeCenterZ = coordinates[2];
    node.latticeHalfUnits = halfUnits;
    node.latticeAligned = 1;
}

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

std::size_t checkedAdd(std::size_t first,
                       std::size_t second,
                       const char* message)
{
    if(second > std::numeric_limits<std::size_t>::max() - first)
        throw UniversalError(message);
    return first + second;
}

std::size_t checkedMultiply(std::size_t first,
                            std::size_t second,
                            const char* message)
{
    if(first != 0 && second > std::numeric_limits<std::size_t>::max() / first)
        throw UniversalError(message);
    return first * second;
}

void collectiveRequire(bool localOk,
                       const std::string& localMessage,
                       const char* context,
                       const MPI_Comm& comm)
{
    int local = localOk ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_LAND, comm);
    if(global == 0)
    {
        if(localOk)
            throw UniversalError(std::string(context) +
                                 ": failed on another MPI rank");
        throw UniversalError(localMessage.empty() ? std::string(context) :
                                                   localMessage);
    }
}

void addDirectRemoteParticles(FmmLocalPatch& targetPatch,
                              const FmmNode& targetNode,
                              const char* payload,
                              std::size_t sourceCount,
                              FmmSolveStats& stats)
{
    const std::vector<std::size_t>& targetOrder =
        targetPatch.tree.particleOrder();
    for(std::size_t k = targetNode.particleBegin;
        k < targetNode.particleEnd; ++k)
    {
        const std::size_t targetBody = targetOrder[k];
        const Vector3D target = targetPatch.positions[targetBody];
        long double ax = 0.0L;
        long double ay = 0.0L;
        long double az = 0.0L;
        long double potential = 0.0L;
        for(std::size_t sourceIndex = 0; sourceIndex < sourceCount;
            ++sourceIndex)
        {
            FmmPatchWireParticle source;
            std::memcpy(&source,
                        payload + sourceIndex * sizeof(FmmPatchWireParticle),
                        sizeof(FmmPatchWireParticle));
            if(!finiteVector(source.positionVector()) ||
               !std::isfinite(source.mass))
                throw UniversalError(
                    "FmmPatchLetPlan::execute: malformed remote particle");
            const Vector3D delta = target - source.positionVector();
            const long double r2 =
                static_cast<long double>(delta.x) * delta.x +
                static_cast<long double>(delta.y) * delta.y +
                static_cast<long double>(delta.z) * delta.z;
            if(r2 == 0.0L)
                continue;
            const long double invR = 1.0L / std::sqrt(r2);
            const long double factor =
                static_cast<long double>(source.mass) * invR * invR * invR;
            ax -= factor * delta.x;
            ay -= factor * delta.y;
            az -= factor * delta.z;
            potential += static_cast<long double>(source.mass) * invR;
        }
        targetPatch.acceleration[targetBody].x += static_cast<double>(ax);
        targetPatch.acceleration[targetBody].y += static_cast<double>(ay);
        targetPatch.acceleration[targetBody].z += static_cast<double>(az);
        if(!targetPatch.potential.empty())
            targetPatch.potential[targetBody] += static_cast<double>(potential);
    }
    ++stats.p2pBlockCount;
    ++stats.letP2PBlockCount;
    const std::uint64_t targetCount = static_cast<std::uint64_t>(
        targetNode.particleCount());
    const std::uint64_t sourceCount64 =
        static_cast<std::uint64_t>(sourceCount);
    if(targetCount != 0 && sourceCount64 >
       std::numeric_limits<std::uint64_t>::max() / targetCount)
        throw UniversalError("FmmPatchLetPlan: P2P pair count overflow");
    const std::uint64_t pairs = targetCount * sourceCount64;
    if(pairs > std::numeric_limits<std::uint64_t>::max() -
                   stats.p2pPairCount ||
       pairs > std::numeric_limits<std::uint64_t>::max() -
                   stats.letP2PPairCount)
        throw UniversalError("FmmPatchLetPlan: accumulated P2P count overflow");
    stats.p2pPairCount += pairs;
    stats.letP2PPairCount += pairs;
}
}

FmmPatchLetPlan::FmmPatchLetPlan():
    waveCount_(1), localWaveCount_(1), maxLetWaveBytes_(0),
    thetaCritical_(0.0), multipoleCoefficientCount_(0),
    maxParticlePayloadCount_(0), particlePayloadSlackFactor_(1.0),
    particlePayloadSlackCount_(0), topologyEpoch_(0), comm_(MPI_COMM_NULL),
    rank_(0), initialized_(false)
{
}

FmmRemoteNodeDescriptor FmmPatchLetPlan::descriptorForNode(
    const FmmNode& node,
    const FmmPatchKey& patch,
    std::uint64_t topologyEpoch)
{
    FmmRemoteNodeDescriptor descriptor;
    descriptor.center[0] = node.center.x;
    descriptor.center[1] = node.center.y;
    descriptor.center[2] = node.center.z;
    descriptor.halfSize = node.halfSize;
    descriptor.radius = node.radius;
    descriptor.spatialKey = node.spatialKey;
    descriptor.patchId = patch.patchId;
    descriptor.particleCount =
        static_cast<std::uint64_t>(node.particleCount());
    descriptor.topologyEpoch = topologyEpoch;
    descriptor.sourceRank = patch.ownerRank;
    descriptor.isLeaf = node.isLeaf() ? 1 : 0;
    descriptor.childMask = static_cast<int>(node.childMask);
    return descriptor;
}

bool FmmPatchLetPlan::admissible(const FmmNode& target,
                                 const FmmRemoteNodeDescriptor& source,
                                 double thetaCritical)
{
    const Vector3D delta = target.center - source.centerVector();
    const double distanceSquared = delta.x * delta.x + delta.y * delta.y +
                                   delta.z * delta.z;
    const double radiusSum = nodeRadius(target) + descriptorRadius(source);
    if(distanceSquared <= radiusSum * radiusSum)
        return false;
    const double distance = std::sqrt(distanceSquared);
    return distance > 0.0 && radiusSum <= thetaCritical * distance;
}

bool FmmPatchLetPlan::m2pAdmissible(
    const FmmNode& target,
    const FmmRemoteNodeDescriptor& source,
    const std::vector<std::size_t>& particleOrder,
    const std::vector<Vector3D>& positions,
    double thetaCritical)
{
    if(!target.isLeaf() || target.particleCount() == 0)
        return false;
    const Vector3D sourceCenter = source.centerVector();
    const double sourceRadius = descriptorRadius(source);
    for(std::size_t k = target.particleBegin; k < target.particleEnd; ++k)
    {
        const Vector3D delta = positions[particleOrder[k]] - sourceCenter;
        const double distanceSquared =
            delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        if(!(distanceSquared > 0.0) ||
           sourceRadius > thetaCritical * std::sqrt(distanceSquared))
            return false;
    }
    return true;
}

FmmNode FmmPatchLetPlan::sourceNodeFromDescriptor(
    const FmmRemoteNodeDescriptor& descriptor,
    const RemoteRootGeometry& root)
{
    FmmNode node;
    node.center = descriptor.centerVector();
    node.halfSize = descriptor.halfSize;
    node.radius = descriptorRadius(descriptor);
    node.spatialKey = descriptor.spatialKey;
    applyRemoteLatticeMetadata(root.latticeId, root.center, root.halfUnits,
                               descriptor.spatialKey, node);
    return node;
}

std::size_t FmmPatchLetPlan::particlePayloadCapacity(
    std::size_t currentCount) const
{
    if(!(particlePayloadSlackFactor_ >= 1.0) ||
       !std::isfinite(particlePayloadSlackFactor_))
        throw UniversalError(
            "FmmPatchLetPlan: invalid particle payload slack factor");

    const long double scaled = std::ceil(
        static_cast<long double>(currentCount) *
        static_cast<long double>(particlePayloadSlackFactor_));
    if(!std::isfinite(static_cast<double>(scaled)) ||
       scaled > static_cast<long double>(
           std::numeric_limits<std::size_t>::max()))
        throw UniversalError(
            "FmmPatchLetPlan: particle payload capacity overflow");

    std::size_t result = std::max(
        maxParticlePayloadCount_, static_cast<std::size_t>(scaled));
    result = std::max(result, currentCount);
    if(particlePayloadSlackCount_ >
       std::numeric_limits<std::size_t>::max() - currentCount)
        throw UniversalError(
            "FmmPatchLetPlan: particle payload additive slack overflow");
    result = std::max(result, currentCount + particlePayloadSlackCount_);
    return result;
}

std::size_t FmmPatchLetPlan::sourceRecordBytes(
    const SourceIdentity& source) const
{
    const FmmPatchKey patch = std::get<0>(source);
    const std::uint64_t spatialKey = std::get<1>(source);
    const int kind = std::get<2>(source);
    const auto patchIt = remoteDescriptors_.find(patch);
    if(patchIt == remoteDescriptors_.end())
        throw UniversalError(
            "FmmPatchLetPlan::sourceRecordBytes: missing remote patch");
    const auto nodeIt = patchIt->second.find(spatialKey);
    if(nodeIt == patchIt->second.end())
        throw UniversalError(
            "FmmPatchLetPlan::sourceRecordBytes: missing remote node");

    std::size_t payload = 0;
    if(kind == static_cast<int>(FmmSubscriptionKind::Multipole))
    {
        payload = checkedMultiply(multipoleCoefficientCount_, sizeof(double),
                                  "FmmPatchLetPlan: multipole byte overflow");
    }
    else if(kind == static_cast<int>(FmmSubscriptionKind::Particles))
    {
        if(nodeIt->second.particleCount > static_cast<std::uint64_t>(
               std::numeric_limits<std::size_t>::max()))
            throw UniversalError(
                "FmmPatchLetPlan: particle count exceeds size_t");
        const std::size_t currentCount =
            static_cast<std::size_t>(nodeIt->second.particleCount);
        const std::size_t plannedCount = particlePayloadCapacity(currentCount);
        payload = checkedMultiply(
            plannedCount, sizeof(FmmPatchWireParticle),
            "FmmPatchLetPlan: particle byte overflow");
    }
    else
    {
        throw UniversalError("FmmPatchLetPlan: invalid source kind");
    }
    return checkedAdd(sizeof(FmmPayloadRecordHeader), payload,
                      "FmmPatchLetPlan: source record byte overflow");
}

std::size_t FmmPatchLetPlan::ensureSourceRecord(
    std::size_t wave,
    const SourceIdentity& identity,
    std::map<WaveSourceIdentity, std::uint32_t>& sourceIndexByWave)
{
    const WaveSourceIdentity waveIdentity(wave, identity);
    const auto found = sourceIndexByWave.find(waveIdentity);
    if(found != sourceIndexByWave.end())
        return found->second;
    if(sources_.size() >= static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()))
        throw UniversalError("FmmPatchLetPlan: too many wave source records");

    const FmmPatchKey patch = std::get<0>(identity);
    const std::uint64_t spatialKey = std::get<1>(identity);
    const int kind = std::get<2>(identity);
    const auto patchIt = remoteDescriptors_.find(patch);
    if(patchIt == remoteDescriptors_.end())
        throw UniversalError("FmmPatchLetPlan: missing source patch");
    const auto descriptorIt = patchIt->second.find(spatialKey);
    if(descriptorIt == patchIt->second.end())
        throw UniversalError("FmmPatchLetPlan: missing source descriptor");
    const auto rootIt = remoteRoots_.find(patch);
    if(rootIt == remoteRoots_.end())
        throw UniversalError("FmmPatchLetPlan: missing source lattice root");

    SourceRecord record;
    record.key.patch = patch;
    record.key.spatialKey = spatialKey;
    record.kind = kind;
    record.wave = wave;
    record.descriptor = descriptorIt->second;
    record.sourceNode = sourceNodeFromDescriptor(record.descriptor,
                                                 rootIt->second);
    const std::uint32_t index = static_cast<std::uint32_t>(sources_.size());
    sources_.push_back(record);
    sourceIndexByWave.emplace(waveIdentity, index);
    return index;
}

void FmmPatchLetPlan::buildWaveRanges()
{
    m2lWaveRanges_.assign(waveCount_, std::make_pair(std::size_t(0),
                                                     std::size_t(0)));
    p2pWaveRanges_.assign(waveCount_, std::make_pair(std::size_t(0),
                                                     std::size_t(0)));
    m2pWaveRanges_.assign(waveCount_, std::make_pair(std::size_t(0),
                                                     std::size_t(0)));

    std::stable_sort(m2lInteractions_.begin(), m2lInteractions_.end(),
        [&](const FmmPatchLetM2LInteraction& first,
            const FmmPatchLetM2LInteraction& second) {
            const std::size_t firstWave = sources_[first.sourceIndex].wave;
            const std::size_t secondWave = sources_[second.sourceIndex].wave;
            return std::tie(firstWave, first.targetPatchIndex,
                            first.targetNode, first.sourceIndex) <
                   std::tie(secondWave, second.targetPatchIndex,
                            second.targetNode, second.sourceIndex);
        });
    std::stable_sort(p2pInteractions_.begin(), p2pInteractions_.end(),
        [&](const FmmPatchLetP2PInteraction& first,
            const FmmPatchLetP2PInteraction& second) {
            const std::size_t firstWave = sources_[first.sourceIndex].wave;
            const std::size_t secondWave = sources_[second.sourceIndex].wave;
            return std::tie(firstWave, first.targetPatchIndex,
                            first.targetNode, first.sourceIndex) <
                   std::tie(secondWave, second.targetPatchIndex,
                            second.targetNode, second.sourceIndex);
        });
    std::stable_sort(m2pInteractions_.begin(), m2pInteractions_.end(),
        [&](const FmmPatchLetM2PInteraction& first,
            const FmmPatchLetM2PInteraction& second) {
            const std::size_t firstWave = sources_[first.sourceIndex].wave;
            const std::size_t secondWave = sources_[second.sourceIndex].wave;
            return std::tie(firstWave, first.sourceIndex,
                            first.targetPatchIndex, first.targetNode) <
                   std::tie(secondWave, second.sourceIndex,
                            second.targetPatchIndex, second.targetNode);
        });

    std::size_t cursor = 0;
    for(std::size_t wave = 0; wave < waveCount_; ++wave)
    {
        const std::size_t begin = cursor;
        while(cursor < m2lInteractions_.size() &&
              sources_[m2lInteractions_[cursor].sourceIndex].wave == wave)
            ++cursor;
        m2lWaveRanges_[wave] = std::make_pair(begin, cursor);
    }
    if(cursor != m2lInteractions_.size())
        throw UniversalError("FmmPatchLetPlan: invalid M2L wave ranges");

    cursor = 0;
    for(std::size_t wave = 0; wave < waveCount_; ++wave)
    {
        const std::size_t begin = cursor;
        while(cursor < p2pInteractions_.size() &&
              sources_[p2pInteractions_[cursor].sourceIndex].wave == wave)
            ++cursor;
        p2pWaveRanges_[wave] = std::make_pair(begin, cursor);
    }
    if(cursor != p2pInteractions_.size())
        throw UniversalError("FmmPatchLetPlan: invalid P2P wave ranges");

    cursor = 0;
    for(std::size_t wave = 0; wave < waveCount_; ++wave)
    {
        const std::size_t begin = cursor;
        while(cursor < m2pInteractions_.size() &&
              sources_[m2pInteractions_[cursor].sourceIndex].wave == wave)
            ++cursor;
        m2pWaveRanges_[wave] = std::make_pair(begin, cursor);
    }
    if(cursor != m2pInteractions_.size())
        throw UniversalError("FmmPatchLetPlan: invalid M2P wave ranges");
}

void FmmPatchLetPlan::build(
    const FmmPatchForest& forest,
    const std::vector<FmmPatchRootDescriptor>& globalDescriptors,
    const FmmProcessPairPlan& processPlan,
    double thetaCritical,
    std::uint64_t topologyEpoch,
    std::size_t maxLetWaveBytes,
    std::size_t maxTargetPatchesPerWave,
    std::size_t multipoleCoefficientCount,
    std::size_t maxParticlePayloadCount,
    double particlePayloadSlackFactor,
    std::size_t particlePayloadSlackCount,
    bool enableLeafM2P,
    const MPI_Comm& comm,
    FmmSolveStats& stats,
    bool reuseUnaffectedTargetSubplans)
{
    const Clock::time_point buildStart = Clock::now();
    if(!(thetaCritical > 0.0) || thetaCritical > 1.0 ||
       !std::isfinite(thetaCritical))
        throw UniversalError("FmmPatchLetPlan::build: invalid theta");
    if(maxTargetPatchesPerWave == 0 || multipoleCoefficientCount == 0 ||
       maxParticlePayloadCount == 0 ||
       !(particlePayloadSlackFactor >= 1.0) ||
       !std::isfinite(particlePayloadSlackFactor))
        throw UniversalError("FmmPatchLetPlan::build: invalid wave configuration");

    // Kept in the API for configuration compatibility. Patch waves are now
    // source-centric: a remote payload is assigned to exactly one wave and may
    // feed any number of target patches without retransmission.
    (void) maxTargetPatchesPerWave;

    std::map<FmmPatchKey, CachedTargetSubplan> previousTargetSubplans;
    std::map<FmmPatchKey, std::uint64_t> previousSourceTopologyHashes;
    if(reuseUnaffectedTargetSubplans && initialized_)
    {
        previousTargetSubplans.swap(targetSubplans_);
        previousSourceTopologyHashes.swap(sourceTopologyHashes_);
        stats.letBuildStorageReused = true;
    }
    else
    {
        targetSubplans_.clear();
        sourceTopologyHashes_.clear();
        reuseUnaffectedTargetSubplans = false;
    }

    comm_ = comm;
    topologyEpoch_ = topologyEpoch;
    maxLetWaveBytes_ = maxLetWaveBytes;
    thetaCritical_ = thetaCritical;
    multipoleCoefficientCount_ = multipoleCoefficientCount;
    maxParticlePayloadCount_ = maxParticlePayloadCount;
    particlePayloadSlackFactor_ = particlePayloadSlackFactor;
    particlePayloadSlackCount_ = particlePayloadSlackCount;
    MPI_Comm_rank(comm_, &rank_);

    localNodeByPatch_.clear();
    remoteRoots_.clear();
    remoteDescriptors_.clear();
    sources_.clear();
    m2lInteractions_.clear();
    p2pInteractions_.clear();
    m2pInteractions_.clear();
    m2lGeometries_.clear();
    m2lWaveRanges_.clear();
    p2pWaveRanges_.clear();
    m2pWaveRanges_.clear();
    subscriptionsReceived_.clear();
    subscriptionsByWave_.clear();
    localParticlePayloadCaps_.clear();
    waveCount_ = 1;
    localWaveCount_ = 1;

    localNodeByPatch_.resize(forest.patches().size());
    for(std::size_t patchIndex = 0; patchIndex < forest.patches().size();
        ++patchIndex)
    {
        const FmmLocalPatch& patch = forest.patches()[patchIndex];
        auto& lookup = localNodeByPatch_[patchIndex];
        lookup.reserve(patch.tree.nodes().size());
        for(std::size_t nodeIndex = 0; nodeIndex < patch.tree.nodes().size();
            ++nodeIndex)
        {
            if(!lookup.emplace(patch.tree.nodes()[nodeIndex].spatialKey,
                               nodeIndex).second)
                throw UniversalError(
                    "FmmPatchLetPlan::build: duplicate local spatial key");
        }
    }

    std::map<FmmPatchKey, std::uint64_t> currentSourceTopologyHashes;
    FmmPatchKey previousGlobalPatch;
    bool havePreviousGlobalPatch = false;
    for(const FmmPatchRootDescriptor& descriptor : globalDescriptors)
    {
        const FmmPatchKey key{descriptor.ownerRank, descriptor.patchId};
        if(!key.valid() || descriptor.active != 1 ||
           descriptor.epoch != topologyEpoch_ ||
           descriptor.magic != FMM_MPI_PACKET_MAGIC ||
           descriptor.version != FMM_MPI_PACKET_VERSION)
            throw UniversalError(
                "FmmPatchLetPlan::build: invalid global patch descriptor");
        if(havePreviousGlobalPatch && !(previousGlobalPatch < key))
            throw UniversalError(
                "FmmPatchLetPlan::build: global descriptors are not sorted and unique");
        previousGlobalPatch = key;
        havePreviousGlobalPatch = true;
        currentSourceTopologyHashes.emplace(key, descriptor.topologyHash);
    }
    const auto findGlobalDescriptor = [&](const FmmPatchKey& key)
        -> const FmmPatchRootDescriptor* {
        const auto found = std::lower_bound(
            globalDescriptors.begin(), globalDescriptors.end(), key,
            [](const FmmPatchRootDescriptor& descriptor,
               const FmmPatchKey& search) {
                return FmmPatchKey{descriptor.ownerRank, descriptor.patchId} <
                       search;
            });
        if(found == globalDescriptors.end() ||
           FmmPatchKey{found->ownerRank, found->patchId} != key)
            return nullptr;
        return &*found;
    };

    std::vector<int> peers = processPlan.letSourceRanks;
    peers.insert(peers.end(), processPlan.letTargetRanks.begin(),
                 processPlan.letTargetRanks.end());
    std::sort(peers.begin(), peers.end());
    peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
    stats.letCommunicatorReused = !exchange_.resetIfChanged(comm_, peers);

    std::map<FmmPatchKey, std::set<FmmPatchKey>> currentSourcesByTarget;
    for(const FmmPatchPair& patchPair : processPlan.remoteLetPairs)
    {
        if(patchPair.target.ownerRank == rank_)
            currentSourcesByTarget[patchPair.target].insert(patchPair.source);
    }

    std::set<FmmPatchKey> reusableTargets;
    std::vector<PendingInteraction> terminalM2L;
    std::vector<PendingInteraction> terminalP2P;
    std::vector<PendingInteraction> terminalM2P;
    for(std::size_t targetPatchIndex = 0;
        targetPatchIndex < forest.patches().size(); ++targetPatchIndex)
    {
        const FmmLocalPatch& targetPatch = forest.patches()[targetPatchIndex];
        const auto previous = previousTargetSubplans.find(targetPatch.key);
        const auto sources = currentSourcesByTarget.find(targetPatch.key);
        const std::set<FmmPatchKey> emptySources;
        const std::set<FmmPatchKey>& currentSources =
            sources == currentSourcesByTarget.end() ? emptySources :
                                                     sources->second;
        const bool sourceSetChanged =
            previous != previousTargetSubplans.end() &&
            previous->second.sourcePatches != currentSources;
        bool sourceChanged = sourceSetChanged;
        bool reusable = reuseUnaffectedTargetSubplans &&
            previous != previousTargetSubplans.end() &&
            !targetPatch.rootGeometryChanged &&
            !targetPatch.leafTopologyChanged &&
            previous->second.targetTopologyHash == targetPatch.topologyHash &&
            !sourceSetChanged;
        if(reusable)
        {
            for(const FmmPatchKey& sourcePatch : currentSources)
            {
                const auto currentHash =
                    currentSourceTopologyHashes.find(sourcePatch);
                const auto previousHash =
                    previousSourceTopologyHashes.find(sourcePatch);
                if(currentHash == currentSourceTopologyHashes.end() ||
                   previousHash == previousSourceTopologyHashes.end() ||
                   currentHash->second != previousHash->second)
                {
                    sourceChanged = true;
                    reusable = false;
                    break;
                }
            }
        }
        if(!reusable)
        {
            ++stats.letTargetSubplansRebuilt;
            if(sourceChanged)
                ++stats.letSourceTriggeredInvalidations;
            continue;
        }

        reusableTargets.insert(targetPatch.key);
        ++stats.letTargetSubplansReused;
        ++stats.letDescriptorTraversalSkippedCount;
        for(const auto& source : previous->second.sources)
        {
            CachedSource cached = source.second;
            cached.descriptor.topologyEpoch = topologyEpoch_;
            remoteRoots_[std::get<0>(source.first)] = cached.root;
            remoteDescriptors_[std::get<0>(source.first)][
                std::get<1>(source.first)] = cached.descriptor;
        }
        const auto appendCached = [&](const CachedTerminal& cached,
                                      std::vector<PendingInteraction>& output) {
            const auto targetNode = localNodeByPatch_[targetPatchIndex].find(
                cached.targetSpatialKey);
            if(targetNode == localNodeByPatch_[targetPatchIndex].end())
                throw UniversalError(
                    "FmmPatchLetPlan::build: cached target node disappeared");
            output.push_back(PendingInteraction{
                targetPatchIndex, targetNode->second,
                std::get<0>(cached.source), std::get<1>(cached.source),
                std::get<2>(cached.source)});
        };
        for(const CachedTerminal& cached : previous->second.m2l)
            appendCached(cached, terminalM2L);
        for(const CachedTerminal& cached : previous->second.p2p)
            appendCached(cached, terminalP2P);
        for(const CachedTerminal& cached : previous->second.m2p)
            appendCached(cached, terminalM2P);
    }
    ++stats.letWavePlanRebuildCount;

    std::vector<PendingPair> pending;
    for(const FmmPatchPair& patchPair : processPlan.remoteLetPairs)
    {
        if(patchPair.target.ownerRank != rank_ ||
           reusableTargets.count(patchPair.target) != 0)
            continue;
        const std::size_t targetPatchIndex =
            forest.findPatch(patchPair.target.patchId);
        if(targetPatchIndex == std::numeric_limits<std::size_t>::max())
            throw UniversalError(
                "FmmPatchLetPlan::build: missing local target patch");
        const FmmLocalPatch& targetPatch =
            forest.patches()[targetPatchIndex];
        if(targetPatch.tree.nodes().empty())
            throw UniversalError(
                "FmmPatchLetPlan::build: target patch has no tree");

        const FmmPatchRootDescriptor* rootPointer =
            findGlobalDescriptor(patchPair.source);
        if(rootPointer == nullptr)
            throw UniversalError(
                "FmmPatchLetPlan::build: missing remote source patch descriptor");
        const FmmPatchRootDescriptor& root = *rootPointer;
        RemoteRootGeometry remoteRoot;
        remoteRoot.latticeId = root.latticeId;
        remoteRoot.center[0] = root.latticeCenter[0];
        remoteRoot.center[1] = root.latticeCenter[1];
        remoteRoot.center[2] = root.latticeCenter[2];
        remoteRoot.halfUnits = root.latticeHalfUnits;
        remoteRoots_[patchPair.source] = remoteRoot;

        FmmRemoteNodeDescriptor descriptor;
        descriptor.center[0] = root.center[0];
        descriptor.center[1] = root.center[1];
        descriptor.center[2] = root.center[2];
        descriptor.halfSize = root.halfSize;
        descriptor.radius = root.radius;
        descriptor.spatialKey = 1;
        descriptor.patchId = root.patchId;
        descriptor.particleCount = root.particleCount;
        descriptor.topologyEpoch = topologyEpoch_;
        descriptor.sourceRank = root.ownerRank;
        descriptor.isLeaf = root.rootLeaf;
        descriptor.childMask = root.childMask;
        if(!validRemoteDescriptor(descriptor))
            throw UniversalError(
                "FmmPatchLetPlan::build: invalid remote root descriptor");
        remoteDescriptors_[patchPair.source][1] = descriptor;
        pending.push_back(PendingPair{
            targetPatchIndex, 0, patchPair.source, 1});
    }

    const Clock::time_point descriptorStart = Clock::now();
    int descriptorRound = 0;
    while(true)
    {
        if(++descriptorRound > FMM_MAX_TREE_DEPTH + 2)
            throw UniversalError(
                "FmmPatchLetPlan::build: descriptor traversal exceeded maximum depth");

        std::vector<PendingPair> work;
        work.swap(pending);
        std::vector<PendingPair> blocked;
        std::unordered_map<int,
            std::set<std::pair<std::uint64_t, std::uint64_t>>> requestSets;

        while(!work.empty())
        {
            const PendingPair pair = work.back();
            work.pop_back();
            if(pair.targetPatchIndex >= forest.patches().size())
                throw UniversalError(
                    "FmmPatchLetPlan::build: target patch index out of range");
            const FmmLocalPatch& targetPatch =
                forest.patches()[pair.targetPatchIndex];
            if(pair.targetNode >= targetPatch.tree.nodes().size())
                throw UniversalError(
                    "FmmPatchLetPlan::build: target node index out of range");
            const FmmNode& target = targetPatch.tree.nodes()[pair.targetNode];

            const auto patchIt = remoteDescriptors_.find(pair.sourcePatch);
            if(patchIt == remoteDescriptors_.end())
                throw UniversalError(
                    "FmmPatchLetPlan::build: missing remote patch cache");
            const auto sourceIt = patchIt->second.find(pair.sourceKey);
            if(sourceIt == patchIt->second.end())
                throw UniversalError(
                    "FmmPatchLetPlan::build: missing remote descriptor");
            const FmmRemoteNodeDescriptor& source = sourceIt->second;
            if(source.topologyEpoch != topologyEpoch_ ||
               source.sourceRank != pair.sourcePatch.ownerRank ||
               source.patchId != pair.sourcePatch.patchId)
                throw UniversalError(
                    "FmmPatchLetPlan::build: stale remote descriptor");

            if(admissible(target, source, thetaCritical))
            {
                terminalM2L.push_back(PendingInteraction{
                    pair.targetPatchIndex, pair.targetNode, pair.sourcePatch,
                    pair.sourceKey,
                    static_cast<int>(FmmSubscriptionKind::Multipole)});
                continue;
            }
            if(enableLeafM2P && target.isLeaf() &&
               m2pAdmissible(target, source,
                             targetPatch.tree.particleOrder(),
                             targetPatch.positions, thetaCritical))
            {
                terminalM2P.push_back(PendingInteraction{
                    pair.targetPatchIndex, pair.targetNode, pair.sourcePatch,
                    pair.sourceKey,
                    static_cast<int>(FmmSubscriptionKind::Multipole)});
                continue;
            }
            if(target.isLeaf() && source.isLeaf != 0)
            {
                terminalP2P.push_back(PendingInteraction{
                    pair.targetPatchIndex, pair.targetNode, pair.sourcePatch,
                    pair.sourceKey,
                    static_cast<int>(FmmSubscriptionKind::Particles)});
                continue;
            }

            const bool splitTarget = !target.isLeaf() &&
                (source.isLeaf != 0 ||
                 nodeRadius(target) >= descriptorRadius(source));
            if(splitTarget)
            {
                for(int octant = 7; octant >= 0; --octant)
                {
                    const std::size_t child =
                        targetPatch.tree.childIndex(target, octant);
                    if(child != std::numeric_limits<std::size_t>::max())
                        work.push_back(PendingPair{
                            pair.targetPatchIndex, child, pair.sourcePatch,
                            pair.sourceKey});
                }
                continue;
            }

            if(source.isLeaf != 0 || source.childMask == 0)
                throw UniversalError(
                    "FmmPatchLetPlan::build: cannot split remote source");
            bool haveAllChildren = true;
            for(int octant = 0; octant < 8; ++octant)
            {
                if((source.childMask & (1 << octant)) == 0)
                    continue;
                const std::uint64_t childKey =
                    (source.spatialKey << 3u) |
                    static_cast<std::uint64_t>(octant);
                if(patchIt->second.find(childKey) == patchIt->second.end())
                    haveAllChildren = false;
            }
            if(!haveAllChildren)
            {
                requestSets[pair.sourcePatch.ownerRank].insert(
                    std::make_pair(pair.sourcePatch.patchId,
                                   pair.sourceKey));
                blocked.push_back(pair);
                continue;
            }
            for(int octant = 7; octant >= 0; --octant)
            {
                if((source.childMask & (1 << octant)) == 0)
                    continue;
                const std::uint64_t childKey =
                    (source.spatialKey << 3u) |
                    static_cast<std::uint64_t>(octant);
                work.push_back(PendingPair{
                    pair.targetPatchIndex, pair.targetNode,
                    pair.sourcePatch, childKey});
            }
        }

        unsigned long long localRequestCount = 0;
        std::unordered_map<int, std::vector<char>> requestBuffers;
        for(const auto& rankEntry : requestSets)
        {
            for(const auto& requestKey : rankEntry.second)
            {
                FmmDescriptorRequest request;
                request.stamp = fmmPacketStamp(
                    FmmPacketKind::DescriptorRequest, topologyEpoch_);
                request.patchId = requestKey.first;
                request.spatialKey = requestKey.second;
                FmmPacketIO::appendPod(requestBuffers[rankEntry.first], request);
                ++localRequestCount;
            }
        }
        unsigned long long globalRequestCount = 0;
        MPI_Allreduce(&localRequestCount, &globalRequestCount, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_);
        if(globalRequestCount == 0)
        {
            if(!blocked.empty())
                throw UniversalError(
                    "FmmPatchLetPlan::build: blocked descriptor pairs without requests");
            break;
        }

        FmmPeerExchangeResult receivedRequests = exchange_.exchangeBytes(
            requestBuffers, &stats.bytesSent, &stats.bytesReceived);
        std::unordered_map<int, std::vector<char>> replyBuffers;
        for(const FmmReceivedMessage& message : receivedRequests.messages())
        {
            const FmmByteView view = receivedRequests.view(message);
            std::size_t offset = 0;
            while(offset < view.size)
            {
                const FmmDescriptorRequest request =
                    FmmPacketIO::readPod<FmmDescriptorRequest>(view, offset);
                validateFmmPacketStamp(
                    request.stamp, FmmPacketKind::DescriptorRequest,
                    topologyEpoch_,
                    "FmmPatchLetPlan::build descriptor request");
                const std::size_t localPatchIndex =
                    forest.findPatch(request.patchId);
                if(localPatchIndex == std::numeric_limits<std::size_t>::max())
                    throw UniversalError(
                        "FmmPatchLetPlan::build: request references missing local patch");
                const FmmLocalPatch& patch =
                    forest.patches()[localPatchIndex];
                if(patch.key.ownerRank != rank_)
                    throw UniversalError(
                        "FmmPatchLetPlan::build: requested patch has wrong owner");
                const auto nodeIt =
                    localNodeByPatch_[localPatchIndex].find(request.spatialKey);
                if(nodeIt == localNodeByPatch_[localPatchIndex].end())
                    throw UniversalError(
                        "FmmPatchLetPlan::build: request references missing local node");
                const FmmNode& node = patch.tree.nodes()[nodeIt->second];
                if(node.isLeaf())
                    throw UniversalError(
                        "FmmPatchLetPlan::build: received child request for leaf");

                int childCount = 0;
                for(int octant = 0; octant < 8; ++octant)
                {
                    if(patch.tree.childIndex(node, octant) !=
                       std::numeric_limits<std::size_t>::max())
                        ++childCount;
                }
                int ordinal = 0;
                for(int octant = 0; octant < 8; ++octant)
                {
                    const std::size_t child =
                        patch.tree.childIndex(node, octant);
                    if(child == std::numeric_limits<std::size_t>::max())
                        continue;
                    FmmDescriptorReply reply;
                    reply.stamp = fmmPacketStamp(
                        FmmPacketKind::DescriptorReply, topologyEpoch_);
                    reply.requestedParentKey = request.spatialKey;
                    reply.child = descriptorForNode(
                        patch.tree.nodes()[child], patch.key,
                        topologyEpoch_);
                    reply.childCount = childCount;
                    reply.childOrdinal = ordinal++;
                    FmmPacketIO::appendPod(replyBuffers[message.source], reply);
                }
            }
        }
        receivedRequests.releaseStorage();

        FmmPeerExchangeResult receivedReplies = exchange_.exchangeBytes(
            replyBuffers, &stats.bytesSent, &stats.bytesReceived);
        std::map<std::tuple<int, std::uint64_t, std::uint64_t>,
                 std::pair<int, std::set<int>>> coverage;
        for(const FmmReceivedMessage& message : receivedReplies.messages())
        {
            const FmmByteView view = receivedReplies.view(message);
            std::size_t offset = 0;
            while(offset < view.size)
            {
                const FmmDescriptorReply reply =
                    FmmPacketIO::readPod<FmmDescriptorReply>(view, offset);
                validateFmmPacketStamp(
                    reply.stamp, FmmPacketKind::DescriptorReply,
                    topologyEpoch_,
                    "FmmPatchLetPlan::build descriptor reply");
                if(reply.childCount <= 0 || reply.childCount > 8 ||
                   reply.childOrdinal < 0 ||
                   reply.childOrdinal >= reply.childCount ||
                   reply.child.sourceRank != message.source ||
                   reply.child.topologyEpoch != topologyEpoch_ ||
                   !validRemoteDescriptor(reply.child) ||
                   (reply.child.spatialKey >> 3u) !=
                       reply.requestedParentKey)
                    throw UniversalError(
                        "FmmPatchLetPlan::build: malformed descriptor reply");
                const auto requestRank = requestSets.find(message.source);
                if(requestRank == requestSets.end() ||
                   requestRank->second.count(std::make_pair(
                       reply.child.patchId,
                       reply.requestedParentKey)) == 0)
                    throw UniversalError(
                        "FmmPatchLetPlan::build: unsolicited descriptor reply");
                const FmmPatchKey sourcePatch{
                    message.source, reply.child.patchId};
                const auto parentPatch =
                    remoteDescriptors_.find(sourcePatch);
                if(parentPatch == remoteDescriptors_.end())
                    throw UniversalError(
                        "FmmPatchLetPlan::build: missing descriptor parent patch");
                const auto parent = parentPatch->second.find(
                    reply.requestedParentKey);
                if(parent == parentPatch->second.end() ||
                   parent->second.isLeaf != 0)
                    throw UniversalError(
                        "FmmPatchLetPlan::build: descriptor reply has invalid parent");
                const unsigned int childBit = 1u <<
                    static_cast<unsigned int>(reply.child.spatialKey & 7u);
                if((static_cast<unsigned int>(parent->second.childMask) &
                    childBit) == 0 ||
                   bitCount(static_cast<unsigned int>(
                       parent->second.childMask)) != reply.childCount)
                    throw UniversalError(
                        "FmmPatchLetPlan::build: descriptor reply contradicts parent");

                auto& item = coverage[std::make_tuple(
                    message.source, reply.child.patchId,
                    reply.requestedParentKey)];
                if(item.first == 0)
                    item.first = reply.childCount;
                if(item.first != reply.childCount ||
                   !item.second.insert(reply.childOrdinal).second)
                    throw UniversalError(
                        "FmmPatchLetPlan::build: inconsistent descriptor reply set");
                const auto inserted =
                    remoteDescriptors_[sourcePatch].emplace(
                        reply.child.spatialKey, reply.child);
                if(!inserted.second)
                {
                    const FmmRemoteNodeDescriptor& old =
                        inserted.first->second;
                    if(old.center[0] != reply.child.center[0] ||
                       old.center[1] != reply.child.center[1] ||
                       old.center[2] != reply.child.center[2] ||
                       old.halfSize != reply.child.halfSize ||
                       old.radius != reply.child.radius ||
                       old.spatialKey != reply.child.spatialKey ||
                       old.patchId != reply.child.patchId ||
                       old.particleCount != reply.child.particleCount ||
                       old.topologyEpoch != reply.child.topologyEpoch ||
                       old.sourceRank != reply.child.sourceRank ||
                       old.isLeaf != reply.child.isLeaf ||
                       old.childMask != reply.child.childMask)
                        throw UniversalError(
                            "FmmPatchLetPlan::build: inconsistent duplicate descriptor");
                }
            }
        }
        receivedReplies.releaseStorage();
        for(const auto& requestRank : requestSets)
        {
            for(const auto& requestKey : requestRank.second)
            {
                const auto found = coverage.find(std::make_tuple(
                    requestRank.first, requestKey.first,
                    requestKey.second));
                if(found == coverage.end() ||
                   found->second.first !=
                       static_cast<int>(found->second.second.size()))
                    throw UniversalError(
                        "FmmPatchLetPlan::build: incomplete descriptor reply set");
            }
        }
        pending.swap(blocked);
    }
    stats.letDescriptorTraversalSeconds += elapsed(descriptorStart);

    const auto interactionLess = [](const PendingInteraction& first,
                                    const PendingInteraction& second) {
        return std::tie(first.targetPatchIndex, first.targetNode,
                        first.sourcePatch, first.sourceKey, first.kind) <
               std::tie(second.targetPatchIndex, second.targetNode,
                        second.sourcePatch, second.sourceKey, second.kind);
    };
    std::sort(terminalM2L.begin(), terminalM2L.end(), interactionLess);
    std::sort(terminalP2P.begin(), terminalP2P.end(), interactionLess);
    std::sort(terminalM2P.begin(), terminalM2P.end(), interactionLess);
    if(std::adjacent_find(terminalM2L.begin(), terminalM2L.end(),
        [&](const PendingInteraction& first,
            const PendingInteraction& second) {
            return !interactionLess(first, second) &&
                   !interactionLess(second, first);
        }) != terminalM2L.end())
        throw UniversalError(
            "FmmPatchLetPlan::build: duplicate M2L interaction");
    if(std::adjacent_find(terminalP2P.begin(), terminalP2P.end(),
        [&](const PendingInteraction& first,
            const PendingInteraction& second) {
            return !interactionLess(first, second) &&
                   !interactionLess(second, first);
        }) != terminalP2P.end())
        throw UniversalError(
            "FmmPatchLetPlan::build: duplicate P2P interaction");
    if(std::adjacent_find(terminalM2P.begin(), terminalM2P.end(),
        [&](const PendingInteraction& first,
            const PendingInteraction& second) {
            return !interactionLess(first, second) &&
                   !interactionLess(second, first);
        }) != terminalM2P.end())
        throw UniversalError(
            "FmmPatchLetPlan::build: duplicate M2P interaction");

    const Clock::time_point compactionStart = Clock::now();
    std::set<FmmRemoteNodeKey> retainedDescriptors;
    for(const PendingInteraction& interaction : terminalM2L)
        retainedDescriptors.insert(FmmRemoteNodeKey{
            interaction.sourcePatch, interaction.sourceKey});
    for(const PendingInteraction& interaction : terminalP2P)
        retainedDescriptors.insert(FmmRemoteNodeKey{
            interaction.sourcePatch, interaction.sourceKey});
    for(const PendingInteraction& interaction : terminalM2P)
        retainedDescriptors.insert(FmmRemoteNodeKey{
            interaction.sourcePatch, interaction.sourceKey});
    for(auto patchIt = remoteDescriptors_.begin();
        patchIt != remoteDescriptors_.end();)
    {
        auto& descriptors = patchIt->second;
        for(auto nodeIt = descriptors.begin(); nodeIt != descriptors.end();)
        {
            if(retainedDescriptors.count(FmmRemoteNodeKey{
                   patchIt->first, nodeIt->first}) == 0)
                nodeIt = descriptors.erase(nodeIt);
            else
                ++nodeIt;
        }
        descriptors.rehash(0);
        if(descriptors.empty())
            patchIt = remoteDescriptors_.erase(patchIt);
        else
            ++patchIt;
    }
    for(auto rootIt = remoteRoots_.begin(); rootIt != remoteRoots_.end();)
    {
        if(remoteDescriptors_.find(rootIt->first) == remoteDescriptors_.end())
            rootIt = remoteRoots_.erase(rootIt);
        else
            ++rootIt;
    }
    stats.letPruneCompactSeconds += elapsed(compactionStart);

    targetSubplans_.clear();
    for(const FmmLocalPatch& patch : forest.patches())
    {
        CachedTargetSubplan subplan;
        subplan.targetPatch = patch.key;
        subplan.targetTopologyHash = patch.topologyHash;
        const auto currentSources = currentSourcesByTarget.find(patch.key);
        if(currentSources != currentSourcesByTarget.end())
            subplan.sourcePatches = currentSources->second;
        targetSubplans_.emplace(patch.key, std::move(subplan));
    }
    enum class CachedInteractionKind { M2L, P2P, M2P };
    const auto cacheTerminal = [&](const PendingInteraction& interaction,
                                   CachedInteractionKind kind) {
        if(interaction.targetPatchIndex >= forest.patches().size())
            throw UniversalError(
                "FmmPatchLetPlan::build: cached target index out of range");
        const FmmLocalPatch& targetPatch =
            forest.patches()[interaction.targetPatchIndex];
        if(interaction.targetNode >= targetPatch.tree.nodes().size())
            throw UniversalError(
                "FmmPatchLetPlan::build: cached target node out of range");
        const SourceIdentity identity(
            interaction.sourcePatch, interaction.sourceKey, interaction.kind);
        const auto descriptorPatch =
            remoteDescriptors_.find(interaction.sourcePatch);
        const auto root = remoteRoots_.find(interaction.sourcePatch);
        if(descriptorPatch == remoteDescriptors_.end() ||
           root == remoteRoots_.end())
            throw UniversalError(
                "FmmPatchLetPlan::build: missing cached source geometry");
        const auto descriptor =
            descriptorPatch->second.find(interaction.sourceKey);
        if(descriptor == descriptorPatch->second.end())
            throw UniversalError(
                "FmmPatchLetPlan::build: missing cached source descriptor");
        CachedTargetSubplan& subplan = targetSubplans_.at(targetPatch.key);
        subplan.sources.emplace(
            identity, CachedSource{descriptor->second, root->second});
        CachedTerminal terminal;
        terminal.targetSpatialKey =
            targetPatch.tree.nodes()[interaction.targetNode].spatialKey;
        terminal.source = identity;
        if(kind == CachedInteractionKind::M2L)
            subplan.m2l.push_back(terminal);
        else if(kind == CachedInteractionKind::P2P)
            subplan.p2p.push_back(terminal);
        else
            subplan.m2p.push_back(terminal);
    };
    for(const PendingInteraction& interaction : terminalM2L)
        cacheTerminal(interaction, CachedInteractionKind::M2L);
    for(const PendingInteraction& interaction : terminalP2P)
        cacheTerminal(interaction, CachedInteractionKind::P2P);
    for(const PendingInteraction& interaction : terminalM2P)
        cacheTerminal(interaction, CachedInteractionKind::M2P);
    sourceTopologyHashes_.swap(currentSourceTopologyHashes);
    std::map<FmmPatchKey, CachedTargetSubplan>().swap(
        previousTargetSubplans);
    std::map<FmmPatchKey, std::uint64_t>().swap(
        previousSourceTopologyHashes);

    std::set<SourceIdentity> uniqueSources;
    for(const PendingInteraction& interaction : terminalM2L)
        uniqueSources.insert(SourceIdentity(interaction.sourcePatch,
                                            interaction.sourceKey,
                                            interaction.kind));
    for(const PendingInteraction& interaction : terminalP2P)
        uniqueSources.insert(SourceIdentity(interaction.sourcePatch,
                                            interaction.sourceKey,
                                            interaction.kind));
    for(const PendingInteraction& interaction : terminalM2P)
        uniqueSources.insert(SourceIdentity(interaction.sourcePatch,
                                            interaction.sourceKey,
                                            interaction.kind));

    std::map<SourceIdentity, std::size_t> waveBySource;
    std::size_t wave = 0;
    std::size_t waveBytes = 0;

    bool localWavePlanOk = true;
    std::string localWavePlanError;
    try
    {
        for(const SourceIdentity& source : uniqueSources)
        {
            const std::size_t recordBytes = sourceRecordBytes(source);
            if(maxLetWaveBytes_ != 0 && recordBytes > maxLetWaveBytes_)
                throw UniversalError(
                    "FmmPatchLetPlan::build: one payload record exceeds maxLetWaveBytes");
            if(maxLetWaveBytes_ != 0 && waveBytes != 0 &&
               recordBytes > maxLetWaveBytes_ - waveBytes)
            {
                ++wave;
                waveBytes = 0;
            }
            waveBySource.emplace(source, wave);
            waveBytes = checkedAdd(
                waveBytes, recordBytes,
                "FmmPatchLetPlan::build: wave byte overflow");
        }
    }
    catch(const UniversalError& error)
    {
        localWavePlanOk = false;
        localWavePlanError = error.getErrorMessage();
    }
    collectiveRequire(localWavePlanOk, localWavePlanError,
                      "FmmPatchLetPlan::build wave planning", comm_);
    localWaveCount_ = uniqueSources.empty() ? 1 : wave + 1;
    unsigned long long localWaves =
        static_cast<unsigned long long>(localWaveCount_);
    unsigned long long globalWaves = 0;
    MPI_Allreduce(&localWaves, &globalWaves, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm_);
    if(globalWaves == 0 || globalWaves > static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max()) ||
       globalWaves > static_cast<unsigned long long>(
            std::numeric_limits<int>::max()))
        throw UniversalError(
            "FmmPatchLetPlan::build: invalid global wave count");
    waveCount_ = static_cast<std::size_t>(globalWaves);

    std::map<WaveSourceIdentity, std::uint32_t> sourceIndexByWave;
    typedef std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                       std::uint64_t, std::uint64_t> GeometryKey;
    std::map<GeometryKey, std::uint32_t> geometryIndexByKey;
    for(const PendingInteraction& interaction : terminalM2L)
    {
        const SourceIdentity source(interaction.sourcePatch,
                                    interaction.sourceKey,
                                    interaction.kind);
        const auto waveIt = waveBySource.find(source);
        if(waveIt == waveBySource.end())
            throw UniversalError(
                "FmmPatchLetPlan::build: M2L dependency has no wave");
        const std::size_t sourceIndex = ensureSourceRecord(
            waveIt->second, source, sourceIndexByWave);
        if(interaction.targetPatchIndex >
               std::numeric_limits<std::uint32_t>::max() ||
           interaction.targetNode >
               std::numeric_limits<std::uint32_t>::max())
            throw UniversalError(
                "FmmPatchLetPlan::build: M2L compact index overflow");
        const FmmNode& targetNode =
            forest.patches()[interaction.targetPatchIndex].tree.nodes()[
                interaction.targetNode];
        const FmmM2LOperatorCache::PreparedGeometry geometry =
            FmmM2LOperatorCache::prepare(
                sources_[sourceIndex].sourceNode, targetNode);
        std::uint64_t inverseScaleBits = 0;
        static_assert(sizeof(inverseScaleBits) ==
                          sizeof(geometry.inverseScale),
                      "M2L inverse scale must be 64-bit");
        std::memcpy(&inverseScaleBits, &geometry.inverseScale,
                    sizeof(inverseScaleBits));
        const GeometryKey geometryKey = std::make_tuple(
            geometry.keyX, geometry.keyY, geometry.keyZ,
            geometry.keyKind, inverseScaleBits);
        auto inserted = geometryIndexByKey.emplace(
            geometryKey, static_cast<std::uint32_t>(
                             m2lGeometries_.size()));
        if(inserted.second)
        {
            if(m2lGeometries_.size() >= static_cast<std::size_t>(
                   std::numeric_limits<std::uint32_t>::max()))
                throw UniversalError(
                    "FmmPatchLetPlan::build: too many M2L geometries");
            inserted.first->second = static_cast<std::uint32_t>(
                m2lGeometries_.size());
            m2lGeometries_.push_back(geometry);
        }
        m2lInteractions_.push_back(FmmPatchLetM2LInteraction{
            static_cast<std::uint32_t>(interaction.targetPatchIndex),
            static_cast<std::uint32_t>(interaction.targetNode),
            static_cast<std::uint32_t>(sourceIndex),
            inserted.first->second});
    }
    for(const PendingInteraction& interaction : terminalM2P)
    {
        const SourceIdentity source(interaction.sourcePatch,
                                    interaction.sourceKey,
                                    interaction.kind);
        const auto waveIt = waveBySource.find(source);
        if(waveIt == waveBySource.end())
            throw UniversalError(
                "FmmPatchLetPlan::build: M2P dependency has no wave");
        const std::size_t sourceIndex = ensureSourceRecord(
            waveIt->second, source, sourceIndexByWave);
        if(interaction.targetPatchIndex >
               std::numeric_limits<std::uint32_t>::max() ||
           interaction.targetNode >
               std::numeric_limits<std::uint32_t>::max())
            throw UniversalError(
                "FmmPatchLetPlan::build: M2P compact index overflow");
        m2pInteractions_.push_back(FmmPatchLetM2PInteraction{
            static_cast<std::uint32_t>(interaction.targetPatchIndex),
            static_cast<std::uint32_t>(interaction.targetNode),
            static_cast<std::uint32_t>(sourceIndex)});
    }
    for(const PendingInteraction& interaction : terminalP2P)
    {
        const SourceIdentity source(interaction.sourcePatch,
                                    interaction.sourceKey,
                                    interaction.kind);
        const auto waveIt = waveBySource.find(source);
        if(waveIt == waveBySource.end())
            throw UniversalError(
                "FmmPatchLetPlan::build: P2P dependency has no wave");
        const std::size_t sourceIndex = ensureSourceRecord(
            waveIt->second, source, sourceIndexByWave);
        if(interaction.targetPatchIndex >
               std::numeric_limits<std::uint32_t>::max() ||
           interaction.targetNode >
               std::numeric_limits<std::uint32_t>::max())
            throw UniversalError(
                "FmmPatchLetPlan::build: P2P compact index overflow");
        p2pInteractions_.push_back(FmmPatchLetP2PInteraction{
            static_cast<std::uint32_t>(interaction.targetPatchIndex),
            static_cast<std::uint32_t>(interaction.targetNode),
            static_cast<std::uint32_t>(sourceIndex)});
    }
    // SourceRecord now contains every terminal descriptor and reconstructed
    // source node needed by warm execution. Release descriptor-pull state before
    // subscriptions are exchanged so it cannot inflate steady LET memory.
    decltype(remoteDescriptors_)().swap(remoteDescriptors_);
    decltype(remoteRoots_)().swap(remoteRoots_);
    buildWaveRanges();

    const Clock::time_point subscriptionStart = Clock::now();
    std::unordered_map<int, std::vector<char>> subscriptionBuffers;
    for(const SourceRecord& source : sources_)
    {
        FmmSubscription subscription;
        subscription.stamp = fmmPacketStamp(
            FmmPacketKind::Subscription, topologyEpoch_);
        subscription.patchId = source.key.patch.patchId;
        subscription.spatialKey = source.key.spatialKey;
        subscription.kind = source.kind;
        subscription.waveIndex = static_cast<int>(source.wave);
        FmmPacketIO::appendPod(
            subscriptionBuffers[source.key.patch.ownerRank], subscription);
    }
    FmmPeerExchangeResult receivedSubscriptions = exchange_.exchangeBytes(
        subscriptionBuffers, &stats.bytesSent, &stats.bytesReceived);
    for(const FmmReceivedMessage& message : receivedSubscriptions.messages())
    {
        const FmmByteView view = receivedSubscriptions.view(message);
        std::size_t offset = 0;
        std::set<std::tuple<std::uint64_t, std::uint64_t, int, int>> unique;
        while(offset < view.size)
        {
            const FmmSubscription subscription =
                FmmPacketIO::readPod<FmmSubscription>(view, offset);
            validateFmmPacketStamp(
                subscription.stamp, FmmPacketKind::Subscription,
                topologyEpoch_,
                "FmmPatchLetPlan::build subscription");
            if(subscription.waveIndex < 0 ||
               static_cast<std::size_t>(subscription.waveIndex) >= waveCount_)
                throw UniversalError(
                    "FmmPatchLetPlan::build: subscription wave out of range");
            const std::size_t localPatchIndex =
                forest.findPatch(subscription.patchId);
            if(localPatchIndex == std::numeric_limits<std::size_t>::max())
                throw UniversalError(
                    "FmmPatchLetPlan::build: subscription references missing local patch");
            const auto nodeIt = localNodeByPatch_[localPatchIndex].find(
                subscription.spatialKey);
            if(nodeIt == localNodeByPatch_[localPatchIndex].end())
                throw UniversalError(
                    "FmmPatchLetPlan::build: subscription references missing local node");
            const FmmNode& node = forest.patches()[localPatchIndex].tree.nodes()[
                nodeIt->second];
            if(subscription.kind ==
                   static_cast<int>(FmmSubscriptionKind::Particles))
            {
                if(!node.isLeaf())
                    throw UniversalError(
                        "FmmPatchLetPlan::build: particle subscription references non-leaf");
                const auto payloadKey = std::make_pair(
                    subscription.patchId, subscription.spatialKey);
                const std::size_t plannedCount = particlePayloadCapacity(node.particleCount());
                const auto inserted = localParticlePayloadCaps_.emplace(
                    payloadKey, plannedCount);
                if(!inserted.second && inserted.first->second != plannedCount)
                    throw UniversalError(
                        "FmmPatchLetPlan::build: inconsistent particle payload capacity");
            }
            if(subscription.kind !=
                   static_cast<int>(FmmSubscriptionKind::Particles) &&
               subscription.kind !=
                   static_cast<int>(FmmSubscriptionKind::Multipole))
                throw UniversalError(
                    "FmmPatchLetPlan::build: invalid subscription kind");
            if(!unique.insert(std::make_tuple(
                    subscription.patchId, subscription.spatialKey,
                    subscription.kind, subscription.waveIndex)).second)
                throw UniversalError(
                    "FmmPatchLetPlan::build: duplicate subscription");
            subscriptionsReceived_[message.source].push_back(subscription);
        }
    }
    receivedSubscriptions.releaseStorage();

    // Resolve subscription ownership and local node indices once at build time.
    // Warm execution then touches only the subscriptions belonging to one wave,
    // avoiding a full scan plus patch/hash lookups for every payload wave.
    subscriptionsByWave_.assign(waveCount_,
        std::vector<SubscriptionReference>());
    std::vector<std::size_t> subscriptionsPerWave(waveCount_, 0);
    for(const auto& peerEntry : subscriptionsReceived_)
        for(const FmmSubscription& subscription : peerEntry.second)
            ++subscriptionsPerWave[static_cast<std::size_t>(
                subscription.waveIndex)];
    for(std::size_t waveIndex = 0; waveIndex < waveCount_; ++waveIndex)
        subscriptionsByWave_[waveIndex].reserve(
            subscriptionsPerWave[waveIndex]);

    for(const auto& peerEntry : subscriptionsReceived_)
    {
        for(std::size_t subscriptionIndex = 0;
            subscriptionIndex < peerEntry.second.size(); ++subscriptionIndex)
        {
            const FmmSubscription& subscription =
                peerEntry.second[subscriptionIndex];
            const std::size_t patchIndex = forest.findPatch(
                subscription.patchId);
            if(patchIndex == std::numeric_limits<std::size_t>::max() ||
               patchIndex >= localNodeByPatch_.size())
                throw UniversalError(
                    "FmmPatchLetPlan::build: indexed subscription patch disappeared");
            const auto nodeIt = localNodeByPatch_[patchIndex].find(
                subscription.spatialKey);
            if(nodeIt == localNodeByPatch_[patchIndex].end())
                throw UniversalError(
                    "FmmPatchLetPlan::build: indexed subscription node disappeared");
            subscriptionsByWave_[static_cast<std::size_t>(
                subscription.waveIndex)].push_back(SubscriptionReference{
                    peerEntry.first, subscriptionIndex, patchIndex,
                    nodeIt->second});
        }
    }

    stats.letSubscriptionSeconds += elapsed(subscriptionStart);
    stats.letWaveCount = waveCount_;
    stats.letLocalWaveCount = localWaveCount_;
    stats.letPlannedM2LCount =
        static_cast<std::uint64_t>(m2lInteractions_.size());
    stats.letPlannedP2PBlockCount =
        static_cast<std::uint64_t>(p2pInteractions_.size());
    stats.letPlanSeconds += elapsed(buildStart);
    initialized_ = true;
}

bool FmmPatchLetPlan::localPayloadShapeReusable(
    const FmmPatchForest& forest) const
{
    if(!initialized_ || localNodeByPatch_.size() != forest.patches().size())
        return false;
    for(const auto& peer : subscriptionsReceived_)
    {
        for(const FmmSubscription& subscription : peer.second)
        {
            if(subscription.kind !=
               static_cast<int>(FmmSubscriptionKind::Particles))
                continue;
            const std::size_t patchIndex =
                forest.findPatch(subscription.patchId);
            if(patchIndex == std::numeric_limits<std::size_t>::max() ||
               patchIndex >= localNodeByPatch_.size())
                return false;
            const auto nodeIt = localNodeByPatch_[patchIndex].find(
                subscription.spatialKey);
            if(nodeIt == localNodeByPatch_[patchIndex].end() ||
               nodeIt->second >= forest.patches()[patchIndex].tree.nodes().size())
                return false;
            const FmmNode& node = forest.patches()[patchIndex].tree.nodes()[
                nodeIt->second];
            const auto capacity = localParticlePayloadCaps_.find(
                std::make_pair(subscription.patchId,
                               subscription.spatialKey));
            if(!node.isLeaf() || capacity == localParticlePayloadCaps_.end() ||
               node.particleCount() > capacity->second)
                return false;
        }
    }

    // M2P admissibility is pointwise rather than node-radius based.  A retained
    // radius envelope makes M2L/P2P topology conservative, but target particles
    // may still move closer to an M2P source.  Revalidate those interactions
    // before deciding to reuse the plan.
    for(const FmmPatchLetM2PInteraction& interaction : m2pInteractions_)
    {
        if(interaction.targetPatchIndex >= forest.patches().size() ||
           interaction.sourceIndex >= sources_.size())
            return false;
        const FmmLocalPatch& targetPatch =
            forest.patches()[interaction.targetPatchIndex];
        if(interaction.targetNode >= targetPatch.tree.nodes().size())
            return false;
        const FmmNode& targetNode =
            targetPatch.tree.nodes()[interaction.targetNode];
        if(targetNode.particleCount() != 0 &&
           !m2pAdmissible(targetNode,
                          sources_[interaction.sourceIndex].descriptor,
                          targetPatch.tree.particleOrder(),
                          targetPatch.positions, thetaCritical_))
            return false;
    }
    return true;
}

void FmmPatchLetPlan::reuse(
    const FmmPatchForest& forest,
    std::uint64_t topologyEpoch,
    FmmSolveStats& stats)
{
    if(!initialized_)
        throw UniversalError("FmmPatchLetPlan::reuse: plan is not initialized");
    if(topologyEpoch != topologyEpoch_)
        throw UniversalError("FmmPatchLetPlan::reuse: topology epoch changed");
    if(localNodeByPatch_.size() != forest.patches().size())
        throw UniversalError("FmmPatchLetPlan::reuse: local patch count changed");
    if(!localPayloadShapeReusable(forest))
        throw UniversalError(
            "FmmPatchLetPlan::reuse: retained payload capacity is stale");

    for(std::size_t patchIndex = 0; patchIndex < forest.patches().size();
        ++patchIndex)
    {
        const FmmLocalPatch& patch = forest.patches()[patchIndex];
        const auto& lookup = localNodeByPatch_[patchIndex];
        if(lookup.size() != patch.tree.nodes().size())
            throw UniversalError(
                "FmmPatchLetPlan::reuse: local node count changed");
        for(std::size_t nodeIndex = 0; nodeIndex < patch.tree.nodes().size();
            ++nodeIndex)
        {
            const auto found = lookup.find(
                patch.tree.nodes()[nodeIndex].spatialKey);
            if(found == lookup.end() || found->second != nodeIndex)
                throw UniversalError(
                    "FmmPatchLetPlan::reuse: local node identity changed");
        }
    }

    for(const FmmPatchLetM2LInteraction& interaction : m2lInteractions_)
    {
        if(interaction.targetPatchIndex >= forest.patches().size() ||
           interaction.targetNode >= forest.patches()[
               interaction.targetPatchIndex].tree.nodes().size() ||
           interaction.sourceIndex >= sources_.size() ||
           interaction.geometryIndex >= m2lGeometries_.size())
            throw UniversalError(
                "FmmPatchLetPlan::reuse: invalid retained M2L interaction");
    }
    for(const FmmPatchLetP2PInteraction& interaction : p2pInteractions_)
    {
        if(interaction.targetPatchIndex >= forest.patches().size() ||
           interaction.targetNode >= forest.patches()[
               interaction.targetPatchIndex].tree.nodes().size() ||
           interaction.sourceIndex >= sources_.size())
            throw UniversalError(
                "FmmPatchLetPlan::reuse: invalid retained P2P interaction");
    }
    for(const FmmPatchLetM2PInteraction& interaction : m2pInteractions_)
    {
        if(interaction.targetPatchIndex >= forest.patches().size() ||
           interaction.targetNode >= forest.patches()[
               interaction.targetPatchIndex].tree.nodes().size() ||
           interaction.sourceIndex >= sources_.size())
            throw UniversalError(
                "FmmPatchLetPlan::reuse: invalid retained M2P interaction");
    }
    for(const auto& peer : subscriptionsReceived_)
    {
        for(const FmmSubscription& subscription : peer.second)
        {
            const std::size_t patchIndex =
                forest.findPatch(subscription.patchId);
            if(patchIndex == std::numeric_limits<std::size_t>::max() ||
               localNodeByPatch_[patchIndex].find(
                   subscription.spatialKey) ==
                   localNodeByPatch_[patchIndex].end())
                throw UniversalError(
                    "FmmPatchLetPlan::reuse: retained subscription is stale");
        }
    }

    stats.letCommunicatorReused = true;
    stats.letBuildStorageReused = true;
    stats.letWaveCount = waveCount_;
    stats.letLocalWaveCount = localWaveCount_;
    stats.letPlannedM2LCount =
        static_cast<std::uint64_t>(m2lInteractions_.size());
    stats.letPlannedP2PBlockCount =
        static_cast<std::uint64_t>(p2pInteractions_.size());
    stats.letTargetSubplansReused =
        static_cast<std::uint64_t>(forest.patches().size());
    stats.letDescriptorTraversalSkippedCount =
        static_cast<std::uint64_t>(forest.patches().size());
}

void FmmPatchLetPlan::executeWave(
    std::size_t wave,
    FmmPatchForest& forest,
    const FmmTaylorExpansion& layout,
    FmmM2LOperatorCache& operatorCache,
    std::size_t maxRemoteBytes,
    FmmSolveStats& stats)
{
    if(wave >= waveCount_)
        throw UniversalError("FmmPatchLetPlan::executeWave: wave out of range");
    const Clock::time_point exchangeStart = Clock::now();

    if(wave >= subscriptionsByWave_.size())
        throw UniversalError(
            "FmmPatchLetPlan::executeWave: missing indexed subscriptions");
    const std::vector<SubscriptionReference>& waveSubscriptions =
        subscriptionsByWave_[wave];

    // Exact pre-reservation avoids repeated growth/copying of large peer
    // buffers.  The solver already performed one collective payload-shape check
    // before entering execution, so repeating it for every wave was redundant.
    std::unordered_map<int, std::size_t> peerReserveBytes;
    for(const SubscriptionReference& reference : waveSubscriptions)
    {
        const auto peer = subscriptionsReceived_.find(reference.peer);
        if(peer == subscriptionsReceived_.end() ||
           reference.subscriptionIndex >= peer->second.size() ||
           reference.patchIndex >= forest.patches().size())
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: invalid indexed subscription");
        const FmmSubscription& subscription =
            peer->second[reference.subscriptionIndex];
        const FmmLocalPatch& patch = forest.patches()[reference.patchIndex];
        if(reference.nodeIndex >= patch.tree.nodes().size())
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: indexed source node disappeared");
        const FmmNode& node = patch.tree.nodes()[reference.nodeIndex];
        std::size_t recordBytes = sizeof(FmmPayloadRecordHeader);
        if(subscription.kind ==
           static_cast<int>(FmmSubscriptionKind::Multipole))
            recordBytes = checkedAdd(recordBytes,
                checkedMultiply(layout.coefficientCount(), sizeof(double),
                    "FmmPatchLetPlan::executeWave: multipole reserve overflow"),
                "FmmPatchLetPlan::executeWave: multipole record overflow");
        else if(subscription.kind ==
                static_cast<int>(FmmSubscriptionKind::Particles))
            recordBytes = checkedAdd(recordBytes,
                checkedMultiply(node.particleCount(),
                    sizeof(FmmPatchWireParticle),
                    "FmmPatchLetPlan::executeWave: particle reserve overflow"),
                "FmmPatchLetPlan::executeWave: particle record overflow");
        else
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: invalid indexed subscription kind");
        peerReserveBytes[reference.peer] = checkedAdd(
            peerReserveBytes[reference.peer], recordBytes,
            "FmmPatchLetPlan::executeWave: peer reserve overflow");
    }

    std::unordered_map<int, std::vector<char>> sendBuffers;
    for(const auto& peerBytes : peerReserveBytes)
        sendBuffers[peerBytes.first].reserve(peerBytes.second);

    for(const SubscriptionReference& reference : waveSubscriptions)
    {
        const auto peer = subscriptionsReceived_.find(reference.peer);
        const FmmSubscription& subscription =
            peer->second[reference.subscriptionIndex];
        std::vector<char>& buffer = sendBuffers[reference.peer];
        FmmLocalPatch& patch = forest.patches()[reference.patchIndex];
        if(patch.key.ownerRank != rank_ ||
           patch.key.patchId != subscription.patchId)
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: source patch identity mismatch");
        const FmmNode& node = patch.tree.nodes()[reference.nodeIndex];

        FmmPayloadRecordHeader header;
        header.stamp = fmmPacketStamp(FmmPacketKind::LetPayload,
                                      topologyEpoch_);
        header.patchId = subscription.patchId;
        header.spatialKey = subscription.spatialKey;
        header.kind = subscription.kind;
        header.waveIndex = subscription.waveIndex;
        if(subscription.kind ==
           static_cast<int>(FmmSubscriptionKind::Multipole))
        {
            header.count = static_cast<std::uint64_t>(
                layout.coefficientCount());
            FmmPacketIO::appendPod(buffer, header);
            FmmPacketIO::appendDoubles(
                buffer,
                patch.multipoles.data() + node.multipoleOffset,
                layout.coefficientCount());
        }
        else if(subscription.kind ==
                static_cast<int>(FmmSubscriptionKind::Particles))
        {
            if(!node.isLeaf())
                throw UniversalError(
                    "FmmPatchLetPlan::executeWave: particle source is not a leaf");
            header.count = static_cast<std::uint64_t>(node.particleCount());
            FmmPacketIO::appendPod(buffer, header);
            for(std::size_t k = node.particleBegin;
                k < node.particleEnd; ++k)
            {
                const std::size_t body = patch.tree.particleOrder()[k];
                FmmPatchWireParticle particle;
                particle.position[0] = patch.positions[body].x;
                particle.position[1] = patch.positions[body].y;
                particle.position[2] = patch.positions[body].z;
                particle.mass = patch.masses[body];
                FmmPacketIO::appendPod(buffer, particle);
            }
        }
        else
        {
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: invalid subscription kind");
        }
    }

    std::size_t totalSendBytes = 0;
    for(const auto& peerEntry : sendBuffers)
        totalSendBytes = checkedAdd(totalSendBytes, peerEntry.second.size(),
            "FmmPatchLetPlan::executeWave: total send byte overflow");

    std::size_t sendCapacityBytes = 0;
    for(const auto& peerEntry : sendBuffers)
    {
        sendCapacityBytes = checkedAdd(
            sendCapacityBytes, peerEntry.second.capacity(),
            "FmmPatchLetPlan::executeWave: send capacity overflow");
    }
    const bool localOutgoingOk =
        totalSendBytes <= maxRemoteBytes / 2 &&
        sendCapacityBytes <= maxRemoteBytes - totalSendBytes;
    collectiveRequire(
        localOutgoingOk,
        localOutgoingOk ? std::string() :
            "FmmPatchLetPlan::executeWave: outgoing wave exceeds memory budget",
        "FmmPatchLetPlan::executeWave outgoing preflight", comm_);

    const std::size_t receiveLimit = std::min(
        maxRemoteBytes - sendCapacityBytes - totalSendBytes,
        maxRemoteBytes / 2);
    FmmPeerExchangeRequest request;
    request.clear();
    exchange_.beginExchangeBytes(
        sendBuffers, request, receiveLimit,
        maxRemoteBytes - sendCapacityBytes);
    const std::size_t launchBytes =
        request.bytesOwned() >
            std::numeric_limits<std::size_t>::max() - sendCapacityBytes ?
        std::numeric_limits<std::size_t>::max() :
        sendCapacityBytes + request.bytesOwned();
    stats.peakRemoteBytes = std::max(stats.peakRemoteBytes, launchBytes);
    std::unordered_map<int, std::vector<char>>().swap(sendBuffers);

    FmmPeerExchangeResult received = request.wait(
        &stats.bytesSent, &stats.bytesReceived);
    const bool wireOverflow = received.bytesOwned() >
        std::numeric_limits<std::size_t>::max() - request.bytesOwned();
    const std::size_t wireBytes = wireOverflow ?
        std::numeric_limits<std::size_t>::max() :
        request.bytesOwned() + received.bytesOwned();
    const bool localReceiveOk = !wireOverflow && wireBytes <= maxRemoteBytes &&
        (maxLetWaveBytes_ == 0 ||
         received.totalBytes() <= maxLetWaveBytes_);
    collectiveRequire(
        localReceiveOk,
        localReceiveOk ? std::string() :
            "FmmPatchLetPlan::executeWave: received wave exceeds its memory budget",
        "FmmPatchLetPlan::executeWave receive preflight", comm_);
    stats.peakRemoteBytes = std::max(stats.peakRemoteBytes, wireBytes);
    stats.letMaxWavePayloadBytes = std::max(
        stats.letMaxWavePayloadBytes, received.totalBytes());
    stats.letExchangeSeconds += elapsed(exchangeStart);

    typedef std::tuple<FmmPatchKey, std::uint64_t, int> PayloadKey;
    struct ExpectedPayload
    {
        PayloadKey key;
        const SourceRecord* source = nullptr;
    };
    struct ReceivedPayload
    {
        PayloadKey key;
        PayloadView view;
    };

    std::size_t expectedCount = 0;
    for(const SourceRecord& source : sources_)
    {
        if(source.wave == wave)
            ++expectedCount;
    }
    const std::size_t estimatedTableBytes = saturatingAdd(
        saturatingMultiply(expectedCount, sizeof(ExpectedPayload)),
        saturatingMultiply(expectedCount, sizeof(ReceivedPayload)));
    const bool localEstimatedTablesOk =
        estimatedTableBytes != std::numeric_limits<std::size_t>::max() &&
        estimatedTableBytes <= maxRemoteBytes - wireBytes;
    collectiveRequire(
        localEstimatedTablesOk,
        localEstimatedTablesOk ? std::string() :
            "FmmPatchLetPlan::executeWave: payload tables exceed memory budget",
        "FmmPatchLetPlan::executeWave table preflight", comm_);

    std::vector<ExpectedPayload> expected;
    std::vector<ReceivedPayload> payloads;
    expected.reserve(expectedCount);
    payloads.reserve(expectedCount);
    const std::size_t tableBytes = saturatingAdd(
        saturatingMultiply(expected.capacity(), sizeof(ExpectedPayload)),
        saturatingMultiply(payloads.capacity(), sizeof(ReceivedPayload)));
    const bool localAllocatedTablesOk =
        tableBytes != std::numeric_limits<std::size_t>::max() &&
        tableBytes <= maxRemoteBytes - wireBytes;
    collectiveRequire(
        localAllocatedTablesOk,
        localAllocatedTablesOk ? std::string() :
            "FmmPatchLetPlan::executeWave: allocated payload tables exceed memory budget",
        "FmmPatchLetPlan::executeWave allocated table preflight", comm_);
    for(const SourceRecord& source : sources_)
    {
        if(source.wave != wave)
            continue;
        expected.push_back(ExpectedPayload{
            PayloadKey(source.key.patch, source.key.spatialKey, source.kind),
            &source});
    }
    std::sort(expected.begin(), expected.end(),
        [](const ExpectedPayload& first, const ExpectedPayload& second) {
            return first.key < second.key;
        });
    if(std::adjacent_find(expected.begin(), expected.end(),
        [](const ExpectedPayload& first, const ExpectedPayload& second) {
            return first.key == second.key;
        }) != expected.end())
        throw UniversalError(
            "FmmPatchLetPlan::executeWave: duplicate expected payload");

    const auto findExpected = [&](const PayloadKey& key)
        -> const ExpectedPayload* {
        const auto found = std::lower_bound(
            expected.begin(), expected.end(), key,
            [](const ExpectedPayload& value, const PayloadKey& search) {
                return value.key < search;
            });
        return found == expected.end() || found->key != key ? nullptr :
                                                               &*found;
    };

    for(const FmmReceivedMessage& message : received.messages())
    {
        const FmmByteView view = received.view(message);
        std::size_t offset = 0;
        while(offset < view.size)
        {
            const FmmPayloadRecordHeader header =
                FmmPacketIO::readPod<FmmPayloadRecordHeader>(view, offset);
            validateFmmPacketStamp(
                header.stamp, FmmPacketKind::LetPayload,
                topologyEpoch_,
                "FmmPatchLetPlan::executeWave payload");
            if(header.waveIndex < 0 ||
               static_cast<std::size_t>(header.waveIndex) != wave ||
               header.patchId == 0 || header.spatialKey == 0)
                throw UniversalError(
                    "FmmPatchLetPlan::executeWave: invalid payload identity");
            const PayloadKey key(
                FmmPatchKey{message.source, header.patchId},
                header.spatialKey, header.kind);
            const ExpectedPayload* expectedPayload = findExpected(key);
            if(expectedPayload == nullptr || expectedPayload->source == nullptr)
                throw UniversalError(
                    "FmmPatchLetPlan::executeWave: unsolicited payload record");

            std::size_t elementBytes = 0;
            if(header.kind ==
               static_cast<int>(FmmSubscriptionKind::Multipole))
            {
                if(header.count != static_cast<std::uint64_t>(
                       layout.coefficientCount()))
                    throw UniversalError(
                        "FmmPatchLetPlan::executeWave: multipole count mismatch");
                elementBytes = sizeof(double);
            }
            else if(header.kind ==
                    static_cast<int>(FmmSubscriptionKind::Particles))
            {
                if(expectedPayload->source->descriptor.isLeaf == 0)
                    throw UniversalError(
                        "FmmPatchLetPlan::executeWave: particle payload for non-leaf");
                // Particle occupancy is payload state, not topology.  A reused
                // LET subscription is keyed by (patch, spatialKey); the current
                // count is carried by this header and may legitimately change.
                elementBytes = sizeof(FmmPatchWireParticle);
            }
            else
            {
                throw UniversalError(
                    "FmmPatchLetPlan::executeWave: invalid payload kind");
            }
            if(header.count > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max()))
                throw UniversalError(
                    "FmmPatchLetPlan::executeWave: payload count overflow");
            const std::size_t count = static_cast<std::size_t>(header.count);
            const std::size_t bytes = checkedMultiply(
                count, elementBytes,
                "FmmPatchLetPlan::executeWave: payload byte overflow");
            if(bytes > FmmPacketIO::remaining(view, offset))
                throw UniversalError(
                    "FmmPatchLetPlan::executeWave: truncated payload");
            if(payloads.size() >= expectedCount)
                throw UniversalError(
                    "FmmPatchLetPlan::executeWave: excessive payload records");
            payloads.push_back(ReceivedPayload{
                key, PayloadView{view.data + offset, count}});
            offset += bytes;
        }
    }
    std::sort(payloads.begin(), payloads.end(),
        [](const ReceivedPayload& first, const ReceivedPayload& second) {
            return first.key < second.key;
        });
    if(payloads.size() != expected.size())
        throw UniversalError(
            "FmmPatchLetPlan::executeWave: missing or excessive payload records");
    for(std::size_t index = 0; index < expected.size(); ++index)
    {
        if(payloads[index].key != expected[index].key ||
           (index != 0 && payloads[index - 1].key == payloads[index].key))
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: missing or duplicate payload record");
    }
    stats.peakRemoteBytes = std::max(
        stats.peakRemoteBytes, saturatingAdd(wireBytes, tableBytes));

    const auto findPayload = [&](const PayloadKey& key)
        -> const ReceivedPayload* {
        const auto found = std::lower_bound(
            payloads.begin(), payloads.end(), key,
            [](const ReceivedPayload& value, const PayloadKey& search) {
                return value.key < search;
            });
        return found == payloads.end() || found->key != key ? nullptr :
                                                               &*found;
    };

    std::vector<double> derivativeScratch;
    std::vector<double> operatorScratch;
    std::vector<double> coefficientScratch(layout.coefficientCount(), 0.0);

    const Clock::time_point m2lStart = Clock::now();
    const std::pair<std::size_t, std::size_t> m2lRange =
        m2lWaveRanges_[wave];
    for(std::size_t interactionIndex = m2lRange.first;
        interactionIndex < m2lRange.second; ++interactionIndex)
    {
        const FmmPatchLetM2LInteraction& interaction =
            m2lInteractions_[interactionIndex];
        if(interaction.targetPatchIndex >= forest.patches().size() ||
           interaction.sourceIndex >= sources_.size() ||
           interaction.geometryIndex >= m2lGeometries_.size())
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: invalid M2L interaction");
        FmmLocalPatch& targetPatch =
            forest.patches()[interaction.targetPatchIndex];
        if(interaction.targetNode >= targetPatch.tree.nodes().size())
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: invalid M2L target node");
        const SourceRecord& source = sources_[interaction.sourceIndex];
        if(source.wave != wave || source.kind !=
           static_cast<int>(FmmSubscriptionKind::Multipole))
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: M2L source assigned to wrong wave");
        const PayloadKey key(source.key.patch, source.key.spatialKey,
                             source.kind);
        const ReceivedPayload* payload = findPayload(key);
        if(payload == nullptr ||
           payload->view.count != layout.coefficientCount())
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: missing M2L payload");
        std::memcpy(coefficientScratch.data(), payload->view.data,
                    layout.coefficientCount() * sizeof(double));
        const FmmNode& targetNode =
            targetPatch.tree.nodes()[interaction.targetNode];
        const FmmM2LOperatorCache::Lookup lookup =
            operatorCache.getPrepared(
                m2lGeometries_[interaction.geometryIndex], layout,
                derivativeScratch, operatorScratch);
        if(lookup.coefficients == nullptr)
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: missing M2L operator");
        FmmKernels::translateM2LRaw(
            source.sourceNode, targetNode, layout,
            coefficientScratch.data(), targetPatch.locals,
            *lookup.coefficients, lookup.inverseScale);
        ++stats.m2lCount;
        ++stats.letM2LCount;
    }
    stats.letM2LSeconds += elapsed(m2lStart);

    const Clock::time_point m2pStart = Clock::now();
    const std::pair<std::size_t, std::size_t> m2pRange =
        m2pWaveRanges_[wave];
    std::uint32_t currentM2PSource =
        std::numeric_limits<std::uint32_t>::max();
    const ReceivedPayload* currentM2PPayload = nullptr;
    std::vector<double> m2pLocals(layout.coefficientCount(), 0.0);
    for(std::size_t interactionIndex = m2pRange.first;
        interactionIndex < m2pRange.second; ++interactionIndex)
    {
        const FmmPatchLetM2PInteraction& interaction =
            m2pInteractions_[interactionIndex];
        if(interaction.targetPatchIndex >= forest.patches().size() ||
           interaction.sourceIndex >= sources_.size())
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: invalid M2P interaction");
        FmmLocalPatch& targetPatch =
            forest.patches()[interaction.targetPatchIndex];
        if(interaction.targetNode >= targetPatch.tree.nodes().size())
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: invalid M2P target node");
        const FmmNode& targetNode =
            targetPatch.tree.nodes()[interaction.targetNode];
        if(!targetNode.isLeaf())
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: M2P target is not a leaf");
        const SourceRecord& source = sources_[interaction.sourceIndex];
        if(source.wave != wave || source.kind !=
           static_cast<int>(FmmSubscriptionKind::Multipole))
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: M2P source assigned to wrong wave");
        if(currentM2PSource != interaction.sourceIndex)
        {
            currentM2PSource = interaction.sourceIndex;
            currentM2PPayload = findPayload(PayloadKey(
                source.key.patch, source.key.spatialKey, source.kind));
            if(currentM2PPayload == nullptr ||
               currentM2PPayload->view.count != layout.coefficientCount())
                throw UniversalError(
                    "FmmPatchLetPlan::executeWave: missing M2P payload");
            std::memcpy(coefficientScratch.data(),
                        currentM2PPayload->view.data,
                        layout.coefficientCount() * sizeof(double));
        }
        std::vector<double>* potential =
            targetPatch.potential.empty() ? nullptr : &targetPatch.potential;
        for(std::size_t k = targetNode.particleBegin;
            k < targetNode.particleEnd; ++k)
        {
            const std::size_t body = targetPatch.tree.particleOrder()[k];
            FmmNode pointTarget;
            pointTarget.center = targetPatch.positions[body];
            pointTarget.particleBegin = k;
            pointTarget.particleEnd = k + 1;
            pointTarget.localOffset = 0;
            const Vector3D displacement =
                pointTarget.center - source.sourceNode.center;
            FmmKernels::computeM2LOperator(
                displacement, layout, derivativeScratch, operatorScratch);
            std::fill(m2pLocals.begin(), m2pLocals.end(), 0.0);
            FmmKernels::translateM2LRaw(
                source.sourceNode, pointTarget, layout,
                coefficientScratch.data(), m2pLocals, operatorScratch, 1.0);
            FmmKernels::evaluateL2P(
                pointTarget, targetPatch.positions,
                targetPatch.tree.particleOrder(), layout, m2pLocals,
                targetPatch.acceleration, potential);
            ++stats.letM2PCount;
        }
    }
    stats.letM2PSeconds += elapsed(m2pStart);

    const Clock::time_point p2pStart = Clock::now();
    const std::pair<std::size_t, std::size_t> p2pRange =
        p2pWaveRanges_[wave];
    for(std::size_t interactionIndex = p2pRange.first;
        interactionIndex < p2pRange.second; ++interactionIndex)
    {
        const FmmPatchLetP2PInteraction& interaction =
            p2pInteractions_[interactionIndex];
        if(interaction.targetPatchIndex >= forest.patches().size() ||
           interaction.sourceIndex >= sources_.size())
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: invalid P2P interaction");
        FmmLocalPatch& targetPatch =
            forest.patches()[interaction.targetPatchIndex];
        if(interaction.targetNode >= targetPatch.tree.nodes().size())
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: invalid P2P target node");
        const FmmNode& targetNode =
            targetPatch.tree.nodes()[interaction.targetNode];
        if(!targetNode.isLeaf())
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: P2P target is not a leaf");
        const SourceRecord& source = sources_[interaction.sourceIndex];
        if(source.wave != wave || source.kind !=
           static_cast<int>(FmmSubscriptionKind::Particles))
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: P2P source assigned to wrong wave");
        const PayloadKey key(source.key.patch, source.key.spatialKey,
                             source.kind);
        const ReceivedPayload* payload = findPayload(key);
        if(payload == nullptr)
            throw UniversalError(
                "FmmPatchLetPlan::executeWave: missing P2P payload");
        addDirectRemoteParticles(targetPatch, targetNode,
                                 payload->view.data,
                                 payload->view.count,
                                 stats);
    }
    stats.letP2PSeconds += elapsed(p2pStart);
    received.releaseStorage();
    request.clear();
}

void FmmPatchLetPlan::execute(
    FmmPatchForest& forest,
    const FmmTaylorExpansion& layout,
    FmmM2LOperatorCache& operatorCache,
    std::size_t maxRemoteBytes,
    std::size_t maxOperatorCacheBytes,
    FmmSolveStats& stats)
{
    const Clock::time_point start = Clock::now();
    if(maxRemoteBytes < 2)
        throw UniversalError("FmmPatchLetPlan::execute: remote budget too small");
    operatorCache.configure(maxOperatorCacheBytes, layout.m2lTerms().size(),
                            m2lGeometries_.size());
    operatorCache.beginPhase();
    for(std::size_t wave = 0; wave < waveCount_; ++wave)
        executeWave(wave, forest, layout, operatorCache, maxRemoteBytes, stats);
    stats.letOperatorCacheBytes = operatorCache.bytesOwned();
    stats.letOperatorCacheEntries = operatorCache.entries();
    stats.letOperatorCacheMaxEntries = operatorCache.maxEntries();
    stats.letOperatorCacheHits = operatorCache.hits();
    stats.letOperatorCacheMisses = operatorCache.misses();
    stats.letOperatorCacheBypasses = operatorCache.bypasses();
    stats.letOperatorIntegerKeyHits = operatorCache.integerKeyHits();
    stats.letOperatorIntegerKeyMisses = operatorCache.integerKeyMisses();
    stats.letExecuteSeconds += elapsed(start);
}

std::size_t FmmPatchLetPlan::bytesOwned() const
{
    std::size_t result = exchange_.bytesOwned();
    const auto add = [&](std::size_t bytes) {
        if(bytes > std::numeric_limits<std::size_t>::max() - result)
            result = std::numeric_limits<std::size_t>::max();
        else
            result += bytes;
    };
    const auto multiply = [](std::size_t count, std::size_t elementSize) {
        return count != 0 && elementSize >
            std::numeric_limits<std::size_t>::max() / count ?
            std::numeric_limits<std::size_t>::max() : count * elementSize;
    };

    add(multiply(sources_.capacity(), sizeof(SourceRecord)));
    add(multiply(m2lInteractions_.capacity(),
                 sizeof(FmmPatchLetM2LInteraction)));
    add(multiply(p2pInteractions_.capacity(),
                 sizeof(FmmPatchLetP2PInteraction)));
    add(multiply(m2pInteractions_.capacity(),
                 sizeof(FmmPatchLetM2PInteraction)));
    add(multiply(m2lGeometries_.capacity(),
                 sizeof(FmmM2LOperatorCache::PreparedGeometry)));
    add(multiply(m2lWaveRanges_.capacity(),
                 sizeof(std::pair<std::size_t, std::size_t>)));
    add(multiply(p2pWaveRanges_.capacity(),
                 sizeof(std::pair<std::size_t, std::size_t>)));
    add(multiply(m2pWaveRanges_.capacity(),
                 sizeof(std::pair<std::size_t, std::size_t>)));

    add(multiply(localNodeByPatch_.capacity(),
                 sizeof(std::unordered_map<std::uint64_t, std::size_t>)));
    const std::size_t localEntryBytes =
        sizeof(std::pair<const std::uint64_t, std::size_t>) +
        2 * sizeof(void*);
    for(const auto& lookup : localNodeByPatch_)
    {
        add(multiply(lookup.bucket_count(), sizeof(void*)));
        add(multiply(lookup.size(), localEntryBytes));
    }

    const std::size_t rootEntryBytes =
        sizeof(std::pair<const FmmPatchKey, RemoteRootGeometry>) +
        2 * sizeof(void*);
    add(multiply(remoteRoots_.size(), rootEntryBytes));

    const std::size_t remoteOuterEntryBytes =
        sizeof(std::pair<const FmmPatchKey,
            std::unordered_map<std::uint64_t, FmmRemoteNodeDescriptor>>) +
        2 * sizeof(void*);
    const std::size_t remoteInnerEntryBytes =
        sizeof(std::pair<const std::uint64_t, FmmRemoteNodeDescriptor>) +
        2 * sizeof(void*);
    add(multiply(remoteDescriptors_.size(), remoteOuterEntryBytes));
    for(const auto& patch : remoteDescriptors_)
    {
        add(multiply(patch.second.bucket_count(), sizeof(void*)));
        add(multiply(patch.second.size(), remoteInnerEntryBytes));
    }

    const std::size_t targetSubplanEntryBytes =
        sizeof(std::pair<const FmmPatchKey, CachedTargetSubplan>) +
        2 * sizeof(void*);
    const std::size_t cachedSourceEntryBytes =
        sizeof(std::pair<const SourceIdentity, CachedSource>) +
        2 * sizeof(void*);
    const std::size_t patchSetEntryBytes = sizeof(FmmPatchKey) +
        3 * sizeof(void*);
    add(multiply(targetSubplans_.size(), targetSubplanEntryBytes));
    for(const auto& target : targetSubplans_)
    {
        add(multiply(target.second.sourcePatches.size(), patchSetEntryBytes));
        add(multiply(target.second.sources.size(), cachedSourceEntryBytes));
        add(multiply(target.second.m2l.capacity(), sizeof(CachedTerminal)));
        add(multiply(target.second.p2p.capacity(), sizeof(CachedTerminal)));
        add(multiply(target.second.m2p.capacity(), sizeof(CachedTerminal)));
    }
    const std::size_t sourceHashEntryBytes =
        sizeof(std::pair<const FmmPatchKey, std::uint64_t>) +
        2 * sizeof(void*);
    add(multiply(sourceTopologyHashes_.size(), sourceHashEntryBytes));

    const std::size_t payloadCapEntryBytes =
        sizeof(std::pair<const std::pair<std::uint64_t, std::uint64_t>,
                         std::size_t>) + 2 * sizeof(void*);
    add(multiply(localParticlePayloadCaps_.size(), payloadCapEntryBytes));

    const std::size_t subscriptionMapEntryBytes =
        sizeof(std::pair<const int, std::vector<FmmSubscription>>) +
        2 * sizeof(void*);
    add(multiply(subscriptionsReceived_.bucket_count(), sizeof(void*)));
    add(multiply(subscriptionsReceived_.size(), subscriptionMapEntryBytes));
    for(const auto& subscriptions : subscriptionsReceived_)
        add(multiply(subscriptions.second.capacity(),
                     sizeof(FmmSubscription)));

    add(multiply(subscriptionsByWave_.capacity(),
                 sizeof(std::vector<SubscriptionReference>)));
    for(const auto& wave : subscriptionsByWave_)
        add(multiply(wave.capacity(), sizeof(SubscriptionReference)));

    return result;
}

#endif // RICH_MPI
