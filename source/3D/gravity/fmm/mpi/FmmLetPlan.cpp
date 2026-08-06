#include "3D/gravity/fmm/mpi/FmmLetPlan.hpp"

#ifdef RICH_MPI

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>

#include "3D/gravity/fmm/FmmKernels.hpp"
#include "misc/universal_error.hpp"

namespace
{
typedef std::chrono::steady_clock Clock;

double elapsed(const Clock::time_point& start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

bool finiteVector(const Vector3D& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

double geometricRadius(const FmmNode& node);

bool boundingSpheresOverlap(const FmmNode& target,
                            const FmmRemoteNodeDescriptor& source)
{
    const Vector3D delta = target.center - source.centerVector();
    const double radius = geometricRadius(target) + source.geometricRadius();
    return delta.x * delta.x + delta.y * delta.y + delta.z * delta.z <=
           radius * radius;
}

double geometricRadius(const FmmNode& node)
{
    return node.radius > 0.0 ? node.radius :
        std::sqrt(3.0) * node.halfSize;
}

bool validRemoteDescriptor(const FmmRemoteNodeDescriptor& descriptor)
{
    const Vector3D center = descriptor.centerVector();
    return finiteVector(center) &&
        descriptor.halfSize > 0.0 && std::isfinite(descriptor.halfSize) &&
        descriptor.radius >= 0.0 && std::isfinite(descriptor.radius) &&
        descriptor.halfSize <= std::numeric_limits<double>::max() / std::sqrt(3.0) &&
        std::isfinite(center.x - descriptor.halfSize) &&
        std::isfinite(center.x + descriptor.halfSize) &&
        std::isfinite(center.y - descriptor.halfSize) &&
        std::isfinite(center.y + descriptor.halfSize) &&
        std::isfinite(center.z - descriptor.halfSize) &&
        std::isfinite(center.z + descriptor.halfSize) &&
        descriptor.spatialKey != 0 &&
        (descriptor.particleCount != 0 || descriptor.isLeaf != 0) &&
        (descriptor.isLeaf == 0 || descriptor.isLeaf == 1) &&
        descriptor.childMask >= 0 && descriptor.childMask <= 255 &&
        ((descriptor.isLeaf != 0 && descriptor.childMask == 0) ||
         (descriptor.isLeaf == 0 && descriptor.childMask != 0));
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

template<typename T>
std::vector<T> readVector(FmmByteView buffer)
{
    if(buffer.size % sizeof(T) != 0)
        throw UniversalError("FmmLetPlan: malformed fixed-size packet vector");
    std::vector<T> result(buffer.size / sizeof(T));
    if(buffer.size != 0)
        std::memcpy(result.data(), buffer.data, buffer.size);
    return result;
}

template<typename T>
const T* packetArray(FmmByteView buffer, std::size_t offset,
                     std::size_t count)
{
    if(count > FmmPacketIO::remaining(buffer, offset) / sizeof(T))
        throw UniversalError("FmmLetPlan: malformed packet array");
    const char* data = buffer.data + offset;
    if(reinterpret_cast<std::uintptr_t>(data) % alignof(T) != 0)
        throw UniversalError("FmmLetPlan: misaligned packet array");
    return reinterpret_cast<const T*>(data);
}

struct RemoteMultipolePayload
{
    FmmPatchKey sourcePatch;
    std::uint64_t spatialKey = 0;
    std::uint32_t sourceIndex = std::numeric_limits<std::uint32_t>::max();
    const double* coefficients = nullptr;
    bool active = false;
};

struct RemoteParticlePayload
{
    FmmPatchKey sourcePatch;
    std::uint64_t spatialKey = 0;
    std::uint32_t sourceIndex = std::numeric_limits<std::uint32_t>::max();
    const FmmWireParticle* particles = nullptr;
    const FmmCompactPatchWireParticle* compactParticles = nullptr;
    const FmmQuantizedPatchWireParticle* quantizedParticles = nullptr;
    Vector3D sourceCenter;
    double sourceHalfSize = 0.0;
    double massScale = 0.0;
    std::size_t particleCount = 0;
};

template<typename Payload>
bool payloadLess(const Payload& first, const Payload& second)
{
    return std::tie(first.sourcePatch, first.spatialKey) <
           std::tie(second.sourcePatch, second.spatialKey);
}

template<typename Payload>
const Payload* findPayload(const std::vector<Payload>& payloads,
                           const FmmPatchKey& sourcePatch,
                           std::uint64_t spatialKey)
{
    const auto found = std::lower_bound(payloads.begin(), payloads.end(),
        std::make_pair(sourcePatch, spatialKey),
        [](const Payload& payload,
           const std::pair<FmmPatchKey, std::uint64_t>& key)
        {
            return std::make_pair(payload.sourcePatch, payload.spatialKey) < key;
        });
    if(found == payloads.end() || found->sourcePatch != sourcePatch ||
       found->spatialKey != spatialKey)
        return nullptr;
    return &*found;
}

[[noreturn]] void abortLetInvariant(const MPI_Comm& comm, const char* message,
                                    const char* detail = nullptr)
{
    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    std::fprintf(stderr, "FMM LET abort on MPI rank %d: %s\n", rank, message);
    if(detail != nullptr && detail[0] != '\0')
        std::fprintf(stderr, "%s\n", detail);
    std::fflush(stderr);
    MPI_Abort(comm, 92);
    throw UniversalError(message);
}

std::size_t saturatingAdd(std::size_t first, std::size_t second)
{
    return second > std::numeric_limits<std::size_t>::max() - first ?
        std::numeric_limits<std::size_t>::max() : first + second;
}

std::size_t saturatingMultiply(std::size_t first, std::size_t second)
{
    return first != 0 && second > std::numeric_limits<std::size_t>::max() / first ?
        std::numeric_limits<std::size_t>::max() : first * second;
}

std::size_t particleWireBytes(bool compact, bool quantized)
{
    if(quantized)
        return sizeof(FmmQuantizedPatchWireParticle);
    return compact ? sizeof(FmmCompactPatchWireParticle) :
                     sizeof(FmmWireParticle);
}

std::size_t particleRecordScaleBytes(bool quantized)
{
    return quantized ? sizeof(double) : 0;
}

std::size_t payloadHeaderBytes(bool compactMultipoles)
{
    return compactMultipoles ? sizeof(FmmCompactPayloadRecordHeader) :
                               sizeof(FmmPayloadRecordHeader);
}

std::size_t multipoleWireElementBytes(bool compactMultipoles)
{
    return compactMultipoles ? sizeof(float) : sizeof(double);
}

void appendFloatMultipoles(std::vector<char>& buffer, const double* values,
                           std::size_t count)
{
    if(count > (std::numeric_limits<std::size_t>::max() - buffer.size()) /
               sizeof(float))
        throw UniversalError(
            "FmmLetPlan::execute: compact multipole payload overflow");
    const std::size_t oldSize = buffer.size();
    buffer.resize(oldSize + count * sizeof(float));
    char* destination = buffer.data() + oldSize;
    for(std::size_t i = 0; i < count; ++i)
    {
        const float converted = static_cast<float>(values[i]);
        if(!std::isfinite(converted))
            throw UniversalError(
                "FmmLetPlan::execute: compact multipole conversion overflow");
        std::memcpy(destination + i * sizeof(float), &converted,
                    sizeof(converted));
    }
}

std::int16_t quantizeSignedUnit(double value)
{
    if(!std::isfinite(value))
        throw UniversalError(
            "FmmLetPlan::execute: non-finite quantized payload value");
    const double bounded = std::max(-1.0, std::min(1.0, value));
    return static_cast<std::int16_t>(std::llround(bounded * 32767.0));
}

std::int64_t shiftedLatticeCoordinate(std::int64_t center,
                                      std::uint64_t offset,
                                      bool upper)
{
    if(offset > static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max()))
        throw UniversalError("FmmLetPlan: lattice offset overflow");
    const std::int64_t signedOffset = static_cast<std::int64_t>(offset);
    if((upper && center > std::numeric_limits<std::int64_t>::max() - signedOffset) ||
       (!upper && center < std::numeric_limits<std::int64_t>::min() + signedOffset))
        throw UniversalError("FmmLetPlan: lattice center overflow");
    return upper ? center + signedOffset : center - signedOffset;
}

void applyRemoteLatticeMetadata(std::uint64_t latticeId,
                                const std::int64_t rootCenter[3],
                                std::uint64_t rootHalfUnits,
                                std::uint64_t spatialKey,
                                FmmNode& node)
{
    if(latticeId == 0 || rootHalfUnits == 0 || spatialKey == 0)
        throw UniversalError("FmmLetPlan: invalid remote lattice metadata");
    std::array<unsigned int, FMM_MAX_TREE_DEPTH> reversedOctants{};
    std::size_t depth = 0;
    std::uint64_t cursor = spatialKey;
    while(cursor != 1)
    {
        if(cursor == 0 || depth >= reversedOctants.size())
            throw UniversalError("FmmLetPlan: malformed remote spatial key");
        reversedOctants[depth++] = static_cast<unsigned int>(cursor & 7u);
        cursor >>= 3u;
    }

    std::int64_t coordinates[3] = {rootCenter[0], rootCenter[1], rootCenter[2]};
    std::uint64_t halfUnits = rootHalfUnits;
    for(std::size_t reverse = depth; reverse > 0; --reverse)
    {
        if(halfUnits < 2 || (halfUnits & 1u) != 0)
            throw UniversalError("FmmLetPlan: indivisible remote lattice root");
        halfUnits /= 2;
        const unsigned int octant = reversedOctants[reverse - 1];
        coordinates[0] = shiftedLatticeCoordinate(
            coordinates[0], halfUnits, (octant & 4u) != 0);
        coordinates[1] = shiftedLatticeCoordinate(
            coordinates[1], halfUnits, (octant & 2u) != 0);
        coordinates[2] = shiftedLatticeCoordinate(
            coordinates[2], halfUnits, (octant & 1u) != 0);
    }

    node.latticeId = latticeId;
    node.latticeCenterX = coordinates[0];
    node.latticeCenterY = coordinates[1];
    node.latticeCenterZ = coordinates[2];
    node.latticeHalfUnits = halfUnits;
    node.latticeAligned = 1;
}

}

FmmLetPlan::FmmLetPlan():
    waveCount_(1),
    executePending_(false), pendingMaxRemoteBytes_(0),
    pendingExchangePreparationSeconds_(0.0), pendingProgressCallCount_(0),
    pendingProgressIncompleteCount_(0), pendingCompletionProgressCall_(0),
    comm_(MPI_COMM_NULL), rank_(0), topologyEpoch_(0),
    compactParticlePayload_(false), quantizedParticlePayload_(false),
    compactMultipolePayload_(false) {}

FmmRemoteNodeDescriptor FmmLetPlan::descriptorForNode(
    const FmmNode& node,
    const FmmPatchKey& patch,
    std::uint64_t topologyEpoch)
{
    FmmRemoteNodeDescriptor result;
    result.center[0] = node.center.x;
    result.center[1] = node.center.y;
    result.center[2] = node.center.z;
    result.halfSize = node.halfSize;
    result.radius = node.radius;
    result.spatialKey = node.spatialKey;
    result.patchId = patch.patchId;
    result.particleCount = static_cast<std::uint64_t>(node.particleCount());
    result.topologyEpoch = topologyEpoch;
    result.sourceRank = patch.ownerRank;
    result.isLeaf = node.isLeaf() ? 1 : 0;
    result.childMask = static_cast<int>(node.childMask);
    return result;
}

bool FmmLetPlan::admissible(const FmmNode& target,
                            const FmmRemoteNodeDescriptor& source,
                            double thetaCritical)
{
    if(boundingSpheresOverlap(target, source))
        return false;
    const Vector3D delta = target.center - source.centerVector();
    const double distance = std::sqrt(delta.x * delta.x + delta.y * delta.y +
                                      delta.z * delta.z);
    return distance > 0.0 &&
        geometricRadius(target) + source.geometricRadius() <= thetaCritical * distance;
}

bool FmmLetPlan::m2pAdmissible(const FmmNode& target,
                               const FmmRemoteNodeDescriptor& source,
                               const std::vector<std::size_t>& particleOrder,
                               const std::vector<Vector3D>& positions,
                               double thetaCritical)
{
    if(target.particleCount() == 0)
        return false;
    const Vector3D sourceCenter = source.centerVector();
    const double sourceRadius = source.geometricRadius();
    // The expansion is evaluated at each particle, so the binding distance is
    // that of the closest particle rather than of the enclosing cell.
    for(std::size_t k = target.particleBegin; k < target.particleEnd; ++k)
    {
        const Vector3D delta = positions[particleOrder[k]] - sourceCenter;
        const double distanceSquared =
            delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
        const double limit = thetaCritical * std::sqrt(distanceSquared);
        if(!(distanceSquared > 0.0) || sourceRadius > limit)
            return false;
    }
    return true;
}

void FmmLetPlan::build(const FmmTree& localTree,
                       const std::vector<Vector3D>& positions,
                       const std::vector<FmmPatchRootDescriptor>& rootDescriptors,
                       const FmmProcessPairPlan& processPlan,
                       double thetaCritical,
                       std::uint64_t topologyEpoch,
                       const MPI_Comm& comm,
                       bool reuseBuildStorage,
                       bool enableLeafM2P,
                       bool compactParticlePayload,
                       bool quantizedParticlePayload,
                       bool compactMultipolePayload,
                       std::size_t maxLetWaveBytes,
                       std::size_t multipoleCoefficientCount,
                       FmmSolveStats& stats)
{
    const Clock::time_point start = Clock::now();
    if(executePending_ || pendingExchange_.active())
        throw UniversalError(
            "FmmLetPlan::build: cannot rebuild with an active LET exchange");

    const Clock::time_point resetStart = Clock::now();
    pendingExchange_.clear();
    comm_ = comm;
    topologyEpoch_ = topologyEpoch;
    compactParticlePayload_ = compactParticlePayload;
    quantizedParticlePayload_ = quantizedParticlePayload;
    compactMultipolePayload_ = compactMultipolePayload;
    MPI_Comm_rank(comm_, &rank_);
    localNodeByKey_.clear();
    if(reuseBuildStorage)
    {
        for(auto& entry : remoteDescriptors_)
            entry.second.clear();
        for(auto& entry : subscriptionsToSend_)
            entry.second.clear();
        for(auto& entry : subscriptionsReceived_)
            entry.second.clear();
    }
    else
    {
        remoteDescriptors_.clear();
        subscriptionsToSend_.clear();
        subscriptionsReceived_.clear();
    }
    remoteLatticeRoots_.clear();
    remoteLatticeRoots_.resize(rootDescriptors.size());
    m2lInteractions_.clear();
    m2lSources_.clear();
    m2lOperatorGeometries_.clear();
    m2lOperatorGeometryUseCounts_.clear();
    p2pInteractions_.clear();
    p2pSources_.clear();
    m2pInteractions_.clear();
    m2lWaveRanges_.clear();
    p2pWaveRanges_.clear();
    m2pWaveRanges_.clear();
    m2lTargetRanges_.clear();
    p2pTargetRanges_.clear();
    m2pTargetRanges_.clear();
    m2lTargetWaveRanges_.clear();
    p2pTargetWaveRanges_.clear();
    m2pTargetWaveRanges_.clear();
    waveCount_ = 1;
    activeM2LInteractionIndices_.clear();
    activeP2PInteractionIndices_.clear();
    activeM2PInteractionIndices_.clear();
    activeM2LSourceFlags_.clear();
    activeP2PSourceFlags_.clear();
    pendingScratch_.clear();
    workScratch_.clear();
    blockedScratch_.clear();
    stats.letBuildStorageReused = reuseBuildStorage;
    stats.letBuildResetSeconds += elapsed(resetStart);

    if(localTree.nodes().size() >
       static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        throw UniversalError("FmmLetPlan::build: local tree exceeds compact index range");

    if(static_cast<double>(localTree.nodes().size()) >
       static_cast<double>(localNodeByKey_.bucket_count()) *
           localNodeByKey_.max_load_factor())
        localNodeByKey_.reserve(localTree.nodes().size());

    for(std::size_t i = 0; i < localTree.nodes().size(); ++i)
    {
        const auto inserted = localNodeByKey_.emplace(localTree.nodes()[i].spatialKey, i);
        if(!inserted.second)
            throw UniversalError("FmmLetPlan::build: duplicate local spatial key");
    }

    // The one-tree-per-rank compatibility LET is nearly complete on the TDE
    // geometry. Normalize it to a complete graph so every rank can select the
    // same optimized MPI_Alltoallv path; absent relationships carry zero
    // counts. This is pure MPI and retains one independently computing rank
    // per core.
    int communicatorSize = 0;
    MPI_Comm_size(comm_, &communicatorSize);
    std::vector<int> peers;
    peers.reserve(static_cast<std::size_t>(std::max(0, communicatorSize - 1)));
    for(int peer = 0; peer < communicatorSize; ++peer)
    {
        if(peer != rank_)
            peers.push_back(peer);
    }
    stats.letCommunicatorReused =
        !exchange_.resetIfChanged(comm_, peers);

    const Clock::time_point descriptorStart = Clock::now();
    std::vector<PendingPair>& pending = pendingScratch_;
    std::vector<PendingInteraction> pendingM2LInteractions;
    std::vector<PendingInteraction> pendingP2PInteractions;
    std::vector<PendingInteraction> pendingM2PInteractions;
    for(int sourceRank : processPlan.letSourceRanks)
    {
        if(sourceRank < 0 ||
           static_cast<std::size_t>(sourceRank) >= rootDescriptors.size())
            throw UniversalError("FmmLetPlan::build: invalid source rank");
        const FmmPatchRootDescriptor& root =
            rootDescriptors[static_cast<std::size_t>(sourceRank)];
        if(root.active == 0 || root.epoch != topologyEpoch_ ||
           root.magic != FMM_MPI_PACKET_MAGIC ||
           root.version != FMM_MPI_PACKET_VERSION ||
           root.patchId == 0 ||
           root.latticeId == 0 || root.latticeHalfUnits == 0 ||
           root.latticeHalfUnits > static_cast<std::uint64_t>(
               std::numeric_limits<std::int64_t>::max()) ||
           (!localTree.nodes().empty() &&
            root.latticeId != localTree.nodes()[0].latticeId))
            throw UniversalError("FmmLetPlan::build: invalid or stale LET source root");
        const FmmPatchKey sourcePatch = root.ownerRank == sourceRank ?
            FmmPatchKey{root.ownerRank, root.patchId} :
            FmmPatchKey{};
        if(!sourcePatch.valid())
            throw UniversalError("FmmLetPlan::build: source patch identity mismatch");
        RemoteLatticeRoot& latticeRoot =
            remoteLatticeRoots_[static_cast<std::size_t>(sourceRank)];
        latticeRoot.latticeId = root.latticeId;
        latticeRoot.center[0] = root.latticeCenter[0];
        latticeRoot.center[1] = root.latticeCenter[1];
        latticeRoot.center[2] = root.latticeCenter[2];
        latticeRoot.halfUnits = root.latticeHalfUnits;
        FmmRemoteNodeDescriptor descriptor;
        descriptor.center[0] = root.center[0];
        descriptor.center[1] = root.center[1];
        descriptor.center[2] = root.center[2];
        descriptor.halfSize = root.halfSize;
        descriptor.radius = root.radius;
        descriptor.spatialKey = 1;
        descriptor.patchId = sourcePatch.patchId;
        descriptor.particleCount = root.particleCount;
        descriptor.topologyEpoch = topologyEpoch_;
        descriptor.sourceRank = sourcePatch.ownerRank;
        descriptor.isLeaf = root.rootLeaf;
        descriptor.childMask = root.childMask;
        remoteDescriptors_[sourcePatch][1] = descriptor;
        if(!localTree.nodes().empty())
            pending.push_back(PendingPair{0, sourcePatch, 1});
    }

    int descriptorRound = 0;
    while(true)
    {
        if(++descriptorRound > FMM_MAX_TREE_DEPTH + 2)
            throw UniversalError("FmmLetPlan::build: descriptor pull exceeded tree depth");
        std::unordered_map<int, std::set<std::pair<std::uint64_t, std::uint64_t>>> requestSets;
        std::vector<PendingPair>& blocked = blockedScratch_;
        std::vector<PendingPair>& work = workScratch_;
        blocked.clear();
        work.clear();
        work.swap(pending);

        while(!work.empty())
        {
            const PendingPair pair = work.back();
            work.pop_back();
            if(pair.targetNode >= localTree.nodes().size())
                throw UniversalError("FmmLetPlan::build: invalid target node");
            const FmmNode& target = localTree.nodes()[pair.targetNode];
            const auto patchIt = remoteDescriptors_.find(pair.sourcePatch);
            if(patchIt == remoteDescriptors_.end())
                throw UniversalError("FmmLetPlan::build: missing remote patch cache");
            const auto nodeIt = patchIt->second.find(pair.sourceKey);
            if(nodeIt == patchIt->second.end())
                throw UniversalError("FmmLetPlan::build: missing remote node descriptor");
            const FmmRemoteNodeDescriptor& source = nodeIt->second;
            if(source.topologyEpoch != topologyEpoch_ ||
               source.sourceRank != pair.sourcePatch.ownerRank ||
               source.patchId != pair.sourcePatch.patchId)
                throw UniversalError("FmmLetPlan::build: stale remote descriptor");

            if(admissible(target, source, thetaCritical))
            {
                pendingM2LInteractions.push_back(PendingInteraction{
                    pair.targetNode, pair.sourcePatch, pair.sourceKey});
                continue;
            }
            // A leaf target cannot shrink its own radius, so without this the
            // pair descends the source all the way to leaves and pulls
            // particles even when a coarse multipole would be accurate.
            if(enableLeafM2P && target.isLeaf() &&
               m2pAdmissible(target, source, localTree.particleOrder(),
                             positions, thetaCritical))
            {
                pendingM2PInteractions.push_back(PendingInteraction{
                    pair.targetNode, pair.sourcePatch, pair.sourceKey});
                continue;
            }
            if(target.isLeaf() && source.isLeaf != 0)
            {
                pendingP2PInteractions.push_back(PendingInteraction{
                    pair.targetNode, pair.sourcePatch, pair.sourceKey});
                continue;
            }

            const bool splitTarget = !target.isLeaf() &&
                (source.isLeaf != 0 || geometricRadius(target) >= source.geometricRadius());
            if(splitTarget)
            {
                for(int octant = 7; octant >= 0; --octant)
                {
                    const std::size_t child = localTree.childIndex(target, octant);
                    if(child != std::numeric_limits<std::size_t>::max())
                        work.push_back(PendingPair{child, pair.sourcePatch, pair.sourceKey});
                }
                continue;
            }

            if(source.isLeaf != 0 || source.childMask == 0)
                throw UniversalError("FmmLetPlan::build: cannot split malformed remote source");
            bool haveAllChildren = true;
            for(int octant = 0; octant < 8; ++octant)
            {
                if((source.childMask & (1 << octant)) == 0)
                    continue;
                const std::uint64_t childKey =
                    (source.spatialKey << 3u) | static_cast<std::uint64_t>(octant);
                if(patchIt->second.find(childKey) == patchIt->second.end())
                    haveAllChildren = false;
            }
            if(!haveAllChildren)
            {
                requestSets[pair.sourcePatch.ownerRank].insert(
                    std::make_pair(pair.sourcePatch.patchId, pair.sourceKey));
                blocked.push_back(pair);
                continue;
            }
            for(int octant = 7; octant >= 0; --octant)
            {
                if((source.childMask & (1 << octant)) == 0)
                    continue;
                const std::uint64_t childKey =
                    (source.spatialKey << 3u) | static_cast<std::uint64_t>(octant);
                work.push_back(PendingPair{pair.targetNode, pair.sourcePatch, childKey});
            }
        }

        std::unordered_map<int, std::vector<char>> requestBuffers;
        unsigned long long localRequestCount = 0;
        for(const auto& entry : requestSets)
        {
            for(const auto& key : entry.second)
            {
                FmmDescriptorRequest request;
                request.stamp = fmmPacketStamp(FmmPacketKind::DescriptorRequest,
                                               topologyEpoch_);
                request.patchId = key.first;
                request.spatialKey = key.second;
                FmmPacketIO::appendPod(requestBuffers[entry.first], request);
                ++localRequestCount;
            }
        }

        unsigned long long globalRequestCount = 0;
        MPI_Allreduce(&localRequestCount, &globalRequestCount, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_);
        if(globalRequestCount == 0)
        {
            if(!blocked.empty())
                throw UniversalError("FmmLetPlan::build: unresolved pairs without requests");
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
                validateFmmPacketStamp(request.stamp,
                    FmmPacketKind::DescriptorRequest, topologyEpoch_,
                    "FmmLetPlan::build descriptor request");
                if(request.patchId == 0)
                    throw UniversalError("FmmLetPlan::build: descriptor request missing patch ID");
                if(request.patchId != FMM_COMPAT_PATCH_ID)
                    throw UniversalError(
                        "FmmLetPlan::build: non-compat patch ID in descriptor request");
                const auto localIt = localNodeByKey_.find(request.spatialKey);
                if(localIt == localNodeByKey_.end())
                    throw UniversalError("FmmLetPlan::build: requested local node does not exist");
                const FmmNode& node = localTree.nodes()[localIt->second];
                if(node.isLeaf())
                    throw UniversalError("FmmLetPlan::build: received child request for a leaf");
                int childCount = 0;
                for(int octant = 0; octant < 8; ++octant)
                    if(localTree.childIndex(node, octant) !=
                       std::numeric_limits<std::size_t>::max())
                        ++childCount;
                const FmmPatchKey localPatch = fmmCompatPatchKey(rank_);
                int ordinal = 0;
                for(int octant = 0; octant < 8; ++octant)
                {
                    const std::size_t child = localTree.childIndex(node, octant);
                    if(child == std::numeric_limits<std::size_t>::max())
                        continue;
                    FmmDescriptorReply reply;
                    reply.stamp = fmmPacketStamp(FmmPacketKind::DescriptorReply,
                                                 topologyEpoch_);
                    reply.requestedParentKey = request.spatialKey;
                    reply.child = descriptorForNode(localTree.nodes()[child],
                                                    localPatch, topologyEpoch_);
                    reply.childCount = childCount;
                    reply.childOrdinal = ordinal++;
                    FmmPacketIO::appendPod(replyBuffers[message.source], reply);
                }
            }
        }

        FmmPeerExchangeResult receivedReplies = exchange_.exchangeBytes(
            replyBuffers, &stats.bytesSent, &stats.bytesReceived);
        std::map<std::tuple<int, std::uint64_t, std::uint64_t>,
                 std::pair<int, std::set<int>>> replyCoverage;
        for(const FmmReceivedMessage& message : receivedReplies.messages())
        {
            const FmmByteView view = receivedReplies.view(message);
            std::size_t offset = 0;
            while(offset < view.size)
            {
                const FmmDescriptorReply reply =
                    FmmPacketIO::readPod<FmmDescriptorReply>(view, offset);
                validateFmmPacketStamp(reply.stamp,
                    FmmPacketKind::DescriptorReply, topologyEpoch_,
                    "FmmLetPlan::build descriptor reply");
                if(reply.childCount <= 0 || reply.childCount > 8 ||
                   reply.childOrdinal < 0 || reply.childOrdinal >= reply.childCount ||
                   reply.child.sourceRank != message.source ||
                   reply.child.patchId == 0 ||
                   reply.child.topologyEpoch != topologyEpoch_ ||
                   !validRemoteDescriptor(reply.child) ||
                   (reply.child.spatialKey >> 3u) != reply.requestedParentKey)
                    throw UniversalError("FmmLetPlan::build: malformed descriptor reply");
                const auto requestRank = requestSets.find(message.source);
                if(requestRank == requestSets.end())
                    throw UniversalError("FmmLetPlan::build: unsolicited descriptor reply");
                const FmmPatchKey sourcePatch{message.source, reply.child.patchId};
                if(requestRank->second.count(
                       std::make_pair(reply.child.patchId,
                                      reply.requestedParentKey)) == 0)
                    throw UniversalError("FmmLetPlan::build: unsolicited descriptor reply");
                const auto parentIt = remoteDescriptors_[sourcePatch].find(
                    reply.requestedParentKey);
                const unsigned int childBit = 1u <<
                    static_cast<unsigned int>(reply.child.spatialKey & 7u);
                if(parentIt == remoteDescriptors_[sourcePatch].end() ||
                   parentIt->second.isLeaf != 0 ||
                   (static_cast<unsigned int>(parentIt->second.childMask) & childBit) == 0 ||
                   bitCount(static_cast<unsigned int>(parentIt->second.childMask)) !=
                       reply.childCount)
                    throw UniversalError("FmmLetPlan::build: reply contradicts parent descriptor");
                auto& coverage = replyCoverage[std::make_tuple(
                    message.source, reply.child.patchId, reply.requestedParentKey)];
                if(coverage.first == 0)
                    coverage.first = reply.childCount;
                if(coverage.first != reply.childCount ||
                   !coverage.second.insert(reply.childOrdinal).second)
                    throw UniversalError("FmmLetPlan::build: inconsistent descriptor reply set");
                const auto inserted = remoteDescriptors_[sourcePatch].emplace(
                    reply.child.spatialKey, reply.child);
                if(!inserted.second)
                {
                    const FmmRemoteNodeDescriptor& old = inserted.first->second;
                    if(old.childMask != reply.child.childMask ||
                       old.isLeaf != reply.child.isLeaf ||
                       old.particleCount != reply.child.particleCount ||
                       old.sourceRank != reply.child.sourceRank ||
                       old.patchId != reply.child.patchId ||
                       old.topologyEpoch != reply.child.topologyEpoch ||
                       old.halfSize != reply.child.halfSize ||
                       old.center[0] != reply.child.center[0] ||
                       old.center[1] != reply.child.center[1] ||
                       old.center[2] != reply.child.center[2])
                        throw UniversalError("FmmLetPlan::build: inconsistent duplicate descriptor");
                }
            }
        }
        for(const auto& requestEntry : requestSets)
        {
            for(const auto& requestKey : requestEntry.second)
            {
                const auto coverage = replyCoverage.find(std::make_tuple(
                    requestEntry.first, requestKey.first, requestKey.second));
                if(coverage == replyCoverage.end() ||
                   coverage->second.first !=
                       static_cast<int>(coverage->second.second.size()))
                    throw UniversalError(
                        "FmmLetPlan::build: incomplete descriptor reply set");
            }
        }
        pending.swap(blocked);
    }

    stats.letDescriptorTraversalSeconds += elapsed(descriptorStart);
    const Clock::time_point finalizeStart = Clock::now();

    std::sort(pendingM2LInteractions.begin(), pendingM2LInteractions.end(),
        [](const PendingInteraction& a, const PendingInteraction& b)
        {
            return std::tie(a.targetNode, a.sourcePatch, a.sourceKey) <
                   std::tie(b.targetNode, b.sourcePatch, b.sourceKey);
        });
    const auto duplicateM2L = std::adjacent_find(
        pendingM2LInteractions.begin(), pendingM2LInteractions.end(),
        [](const PendingInteraction& a, const PendingInteraction& b)
        {
            return a.targetNode == b.targetNode && a.sourcePatch == b.sourcePatch &&
                   a.sourceKey == b.sourceKey;
        });
    if(duplicateM2L != pendingM2LInteractions.end())
        throw UniversalError("FmmLetPlan::build: duplicate LET M2L interaction");

    std::sort(pendingM2PInteractions.begin(), pendingM2PInteractions.end(),
        [](const PendingInteraction& a, const PendingInteraction& b)
        {
            return std::tie(a.targetNode, a.sourcePatch, a.sourceKey) <
                   std::tie(b.targetNode, b.sourcePatch, b.sourceKey);
        });
    const auto duplicateM2P = std::adjacent_find(
        pendingM2PInteractions.begin(), pendingM2PInteractions.end(),
        [](const PendingInteraction& a, const PendingInteraction& b)
        {
            return a.targetNode == b.targetNode && a.sourcePatch == b.sourcePatch &&
                   a.sourceKey == b.sourceKey;
        });
    if(duplicateM2P != pendingM2PInteractions.end())
        throw UniversalError("FmmLetPlan::build: duplicate LET M2P interaction");

    // Resolve each unique remote multipole source once while the topology is
    // built. Warm solves keep target-major interaction order for local-locality,
    // but use a compact direct source index instead of repeating descriptor hash
    // lookups and lattice-key reconstruction for every interaction. M2P shares
    // this table because it consumes the same multipole payload.
    std::map<FmmRemoteNodeKey, std::uint32_t> sourceIndexByKey;
    for(const PendingInteraction& interaction : pendingM2LInteractions)
        sourceIndexByKey.emplace(
            FmmRemoteNodeKey{interaction.sourcePatch, interaction.sourceKey}, 0u);
    for(const PendingInteraction& interaction : pendingM2PInteractions)
        sourceIndexByKey.emplace(
            FmmRemoteNodeKey{interaction.sourcePatch, interaction.sourceKey}, 0u);
    if(sourceIndexByKey.size() > static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()))
        throw UniversalError("FmmLetPlan::build: too many unique LET M2L sources");

    m2lSources_.reserve(sourceIndexByKey.size());
    std::uint32_t nextSourceIndex = 0;
    for(auto& sourceEntry : sourceIndexByKey)
    {
        sourceEntry.second = nextSourceIndex++;
        const FmmPatchKey& sourcePatch = sourceEntry.first.patch;
        const std::uint64_t sourceKey = sourceEntry.first.spatialKey;
        const auto descriptorPatch = remoteDescriptors_.find(sourcePatch);
        if(descriptorPatch == remoteDescriptors_.end())
            throw UniversalError(
                "FmmLetPlan::build: missing resolved M2L source patch");
        const auto descriptorIt = descriptorPatch->second.find(sourceKey);
        if(descriptorIt == descriptorPatch->second.end())
            throw UniversalError(
                "FmmLetPlan::build: missing resolved M2L source descriptor");
        if(sourcePatch.ownerRank < 0 ||
           static_cast<std::size_t>(sourcePatch.ownerRank) >=
               remoteLatticeRoots_.size())
            throw UniversalError(
                "FmmLetPlan::build: missing resolved M2L lattice root");

        FmmNode source;
        source.center = descriptorIt->second.centerVector();
        source.halfSize = descriptorIt->second.halfSize;
        source.radius = descriptorIt->second.geometricRadius();
        const RemoteLatticeRoot& latticeRoot =
            remoteLatticeRoots_[static_cast<std::size_t>(sourcePatch.ownerRank)];
        applyRemoteLatticeMetadata(latticeRoot.latticeId, latticeRoot.center,
                                   latticeRoot.halfUnits, sourceKey, source);

        M2LSource resolved;
        resolved.sourcePatch = sourcePatch;
        resolved.spatialKey = sourceKey;
        resolved.node = source;
        m2lSources_.push_back(resolved);
    }

    m2lInteractions_.reserve(pendingM2LInteractions.size());
    for(const PendingInteraction& interaction : pendingM2LInteractions)
    {
        const auto found = sourceIndexByKey.find(
            FmmRemoteNodeKey{interaction.sourcePatch, interaction.sourceKey});
        if(found == sourceIndexByKey.end())
            throw UniversalError(
                "FmmLetPlan::build: unresolved LET M2L source index");
        m2lInteractions_.push_back(FmmLetM2LInteraction{
            static_cast<std::uint32_t>(interaction.targetNode),
            found->second, 0u});
    }

    // M2P needs no translation operator, so it carries no geometry index and is
    // deliberately excluded from the operator-cache grouping below.
    m2pInteractions_.reserve(pendingM2PInteractions.size());
    for(const PendingInteraction& interaction : pendingM2PInteractions)
    {
        const auto found = sourceIndexByKey.find(
            FmmRemoteNodeKey{interaction.sourcePatch, interaction.sourceKey});
        if(found == sourceIndexByKey.end())
            throw UniversalError(
                "FmmLetPlan::build: unresolved LET M2P source index");
        m2pInteractions_.push_back(FmmLetM2PInteraction{
            static_cast<std::uint32_t>(interaction.targetNode),
            found->second});
    }

    typedef std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                       std::uint64_t, std::uint64_t> GeometryKey;
    std::map<GeometryKey, std::uint32_t> geometryIndexByKey;
    for(std::size_t interactionIndex = 0;
        interactionIndex < m2lInteractions_.size(); ++interactionIndex)
    {
        const FmmLetM2LInteraction& interaction =
            m2lInteractions_[interactionIndex];
        const std::uint32_t sourceIndex = interaction.sourceIndex;
        const FmmM2LOperatorCache::PreparedGeometry geometry =
            FmmM2LOperatorCache::prepare(
                m2lSources_[sourceIndex].node,
                localTree.nodes()[interaction.targetNode]);
        std::uint64_t inverseScaleBits = 0;
        static_assert(sizeof(inverseScaleBits) == sizeof(geometry.inverseScale),
                      "prepared inverse scale must be 64-bit");
        std::memcpy(&inverseScaleBits, &geometry.inverseScale,
                    sizeof(inverseScaleBits));
        const GeometryKey key = std::make_tuple(
            geometry.keyX, geometry.keyY, geometry.keyZ,
            geometry.keyKind, inverseScaleBits);
        auto inserted = geometryIndexByKey.emplace(
            key, static_cast<std::uint32_t>(m2lOperatorGeometries_.size()));
        if(inserted.second)
        {
            if(m2lOperatorGeometries_.size() >= static_cast<std::size_t>(
                    std::numeric_limits<std::uint32_t>::max()))
                throw UniversalError(
                    "FmmLetPlan::build: too many prepared M2L geometries");
            inserted.first->second = static_cast<std::uint32_t>(
                m2lOperatorGeometries_.size());
            m2lOperatorGeometries_.push_back(geometry);
            m2lOperatorGeometryUseCounts_.push_back(0);
        }
        m2lInteractions_[interactionIndex].geometryIndex =
            inserted.first->second;
        if(m2lOperatorGeometryUseCounts_[inserted.first->second] ==
           std::numeric_limits<std::uint64_t>::max())
            throw UniversalError(
                "FmmLetPlan::build: M2L geometry use count overflow");
        ++m2lOperatorGeometryUseCounts_[inserted.first->second];
    }
    // The persistent operator cache is intentionally byte bounded.  When the
    // complete LET operator set does not fit, resolvePreparedBatch() falls back
    // to scratch storage.  Keep all interactions that use the same scale-free
    // operator key contiguous so one scratch operator can serve the whole
    // group instead of being regenerated for every interaction.  Geometry
    // variants with different inverse scales remain separate inside a key
    // group, preserving the per-interaction physical scaling.
    std::vector<std::uint32_t> geometryOrder(
        m2lOperatorGeometries_.size());
    std::iota(geometryOrder.begin(), geometryOrder.end(), 0u);
    std::sort(geometryOrder.begin(), geometryOrder.end(),
        [&](std::uint32_t first, std::uint32_t second)
        {
            const FmmM2LOperatorCache::PreparedGeometry& a =
                m2lOperatorGeometries_[first];
            const FmmM2LOperatorCache::PreparedGeometry& b =
                m2lOperatorGeometries_[second];
            return std::tie(a.keyKind, a.keyX, a.keyY, a.keyZ,
                            a.inverseScale) <
                   std::tie(b.keyKind, b.keyX, b.keyY, b.keyZ,
                            b.inverseScale);
        });

    std::vector<std::size_t> nextSlot(m2lOperatorGeometries_.size(), 0);
    std::size_t orderedCount = 0;
    for(std::uint32_t geometryIndex : geometryOrder)
    {
        nextSlot[geometryIndex] = orderedCount;
        const std::uint64_t count =
            m2lOperatorGeometryUseCounts_[geometryIndex];
        if(count > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max() - orderedCount))
            throw UniversalError(
                "FmmLetPlan::build: grouped M2L interaction count overflow");
        orderedCount += static_cast<std::size_t>(count);
    }
    if(orderedCount != m2lInteractions_.size())
        throw UniversalError(
            "FmmLetPlan::build: grouped M2L interaction count mismatch");

    std::vector<FmmLetM2LInteraction> orderedInteractions(orderedCount);
    for(std::size_t interactionIndex = 0;
        interactionIndex < m2lInteractions_.size(); ++interactionIndex)
    {
        const std::uint32_t geometryIndex =
            m2lInteractions_[interactionIndex].geometryIndex;
        const std::size_t destination = nextSlot[geometryIndex]++;
        orderedInteractions[destination] = m2lInteractions_[interactionIndex];
    }
    m2lInteractions_.swap(orderedInteractions);

    std::sort(pendingP2PInteractions.begin(), pendingP2PInteractions.end(),
        [](const PendingInteraction& a, const PendingInteraction& b)
        {
            return std::tie(a.targetNode, a.sourcePatch, a.sourceKey) <
                   std::tie(b.targetNode, b.sourcePatch, b.sourceKey);
        });
    const auto duplicateP2P = std::adjacent_find(
        pendingP2PInteractions.begin(), pendingP2PInteractions.end(),
        [](const PendingInteraction& a, const PendingInteraction& b)
        {
            return a.targetNode == b.targetNode && a.sourcePatch == b.sourcePatch &&
                   a.sourceKey == b.sourceKey;
        });
    if(duplicateP2P != pendingP2PInteractions.end())
        throw UniversalError("FmmLetPlan::build: duplicate LET P2P interaction");

    std::map<FmmRemoteNodeKey, std::uint32_t> p2pSourceIndexByKey;
    for(const PendingInteraction& interaction : pendingP2PInteractions)
        p2pSourceIndexByKey.emplace(
            FmmRemoteNodeKey{interaction.sourcePatch, interaction.sourceKey}, 0u);
    if(p2pSourceIndexByKey.size() > static_cast<std::size_t>(
           std::numeric_limits<std::uint32_t>::max()))
        throw UniversalError("FmmLetPlan::build: too many unique LET P2P sources");
    p2pSources_.reserve(p2pSourceIndexByKey.size());
    std::uint32_t nextP2PSourceIndex = 0;
    for(auto& sourceEntry : p2pSourceIndexByKey)
    {
        sourceEntry.second = nextP2PSourceIndex++;
        p2pSources_.push_back(RemoteSource{
            sourceEntry.first.patch, sourceEntry.first.spatialKey});
    }
    p2pInteractions_.reserve(pendingP2PInteractions.size());
    for(const PendingInteraction& interaction : pendingP2PInteractions)
    {
        const auto found = p2pSourceIndexByKey.find(
            FmmRemoteNodeKey{interaction.sourcePatch, interaction.sourceKey});
        if(found == p2pSourceIndexByKey.end())
            throw UniversalError("FmmLetPlan::build: unresolved LET P2P source index");
        p2pInteractions_.push_back(FmmLetP2PInteraction{
            static_cast<std::uint32_t>(interaction.targetNode), found->second});
    }
    std::sort(p2pInteractions_.begin(), p2pInteractions_.end(),
        [](const FmmLetP2PInteraction& first,
           const FmmLetP2PInteraction& second)
        {
            return std::tie(first.sourceIndex, first.targetNode) <
                   std::tie(second.sourceIndex, second.targetNode);
        });
    const char* geometryLog = std::getenv("RICH_FMM_GEOM_LOG");
    if(geometryLog != nullptr && geometryLog[0] != '\0' &&
       !(geometryLog[0] == '0' && geometryLog[1] == '\0'))
    {
        std::set<int> m2lRanks;
        for(const M2LSource& source : m2lSources_)
            m2lRanks.insert(source.sourcePatch.ownerRank);
        std::set<int> p2pRanks;
        for(const RemoteSource& source : p2pSources_)
            p2pRanks.insert(source.sourcePatch.ownerRank);
        std::set<std::uint32_t> m2pSourceSet;
        std::set<int> m2pRanks;
        for(const FmmLetM2PInteraction& interaction : m2pInteractions_)
        {
            m2pSourceSet.insert(interaction.sourceIndex);
            m2pRanks.insert(m2lSources_[interaction.sourceIndex].sourcePatch.ownerRank);
        }
        // Empty leaves are valid subscription targets but carry no payload, so
        // source counts alone overstate the request. Price it from the
        // descriptor particle counts instead.
        std::size_t p2pActiveSources = 0;
        std::size_t p2pParticles = 0;
        std::set<int> p2pActiveRanks;
        for(const RemoteSource& source : p2pSources_)
        {
            const auto patchIt = remoteDescriptors_.find(source.sourcePatch);
            if(patchIt == remoteDescriptors_.end())
                continue;
            const auto nodeIt = patchIt->second.find(source.spatialKey);
            if(nodeIt == patchIt->second.end() ||
               nodeIt->second.particleCount == 0)
                continue;
            ++p2pActiveSources;
            p2pParticles += static_cast<std::size_t>(
                nodeIt->second.particleCount);
            p2pActiveRanks.insert(source.sourcePatch.ownerRank);
        }
        const std::size_t p2pPayloadBytes =
            p2pActiveSources * (payloadHeaderBytes(
                compactMultipolePayload_) + particleRecordScaleBytes(
                quantizedParticlePayload_)) +
            p2pParticles * particleWireBytes(compactParticlePayload_,
                                             quantizedParticlePayload_);
        std::fprintf(stderr,
            "FMMLET rank=%d localNodes=%zu m2lInteractions=%zu "
            "m2lSources=%zu m2lRanks=%zu p2pInteractions=%zu "
            "p2pSources=%zu p2pRanks=%zu m2pInteractions=%zu "
            "m2pSources=%zu m2pRanks=%zu p2pActiveSources=%zu "
            "p2pActiveRanks=%zu p2pParticles=%zu p2pPayloadBytes=%zu\n",
            rank_, localTree.nodes().size(), m2lInteractions_.size(),
            m2lSources_.size(), m2lRanks.size(), p2pInteractions_.size(),
            p2pSources_.size(), p2pRanks.size(), m2pInteractions_.size(),
            m2pSourceSet.size(), m2pRanks.size(), p2pActiveSources,
            p2pActiveRanks.size(), p2pParticles, p2pPayloadBytes);
        std::fflush(stderr);
    }

    std::set<std::tuple<std::size_t, FmmPatchKey, std::uint64_t>> terminalKeys;
    for(const PendingInteraction& interaction : pendingM2LInteractions)
        terminalKeys.insert(std::make_tuple(interaction.targetNode,
                                            interaction.sourcePatch,
                                            interaction.sourceKey));
    for(const PendingInteraction& interaction : pendingP2PInteractions)
    {
        if(!terminalKeys.insert(std::make_tuple(interaction.targetNode,
                                                interaction.sourcePatch,
                                                interaction.sourceKey)).second)
            throw UniversalError("FmmLetPlan::build: interaction classified as both M2L and P2P");
    }
    for(const PendingInteraction& interaction : pendingM2PInteractions)
    {
        if(!terminalKeys.insert(std::make_tuple(interaction.targetNode,
                                                interaction.sourcePatch,
                                                interaction.sourceKey)).second)
            throw UniversalError(
                "FmmLetPlan::build: interaction classified as both M2P and another kind");
    }

    stats.letFinalizeSeconds += elapsed(finalizeStart);
    const Clock::time_point subscriptionStart = Clock::now();

    std::unordered_map<int,
        std::set<std::tuple<std::uint64_t, std::uint64_t, int>>> subscriptionSets;
    for(const FmmLetM2LInteraction& interaction : m2lInteractions_)
    {
        const M2LSource& source = m2lSources_[interaction.sourceIndex];
        subscriptionSets[source.sourcePatch.ownerRank].insert(
            std::make_tuple(source.sourcePatch.patchId, source.spatialKey,
                            static_cast<int>(FmmSubscriptionKind::Multipole)));
    }
    for(const FmmLetP2PInteraction& interaction : p2pInteractions_)
    {
        const RemoteSource& source = p2pSources_[interaction.sourceIndex];
        subscriptionSets[source.sourcePatch.ownerRank].insert(
            std::make_tuple(source.sourcePatch.patchId, source.spatialKey,
                            static_cast<int>(FmmSubscriptionKind::Particles)));
    }
    for(const FmmLetM2PInteraction& interaction : m2pInteractions_)
    {
        const M2LSource& source = m2lSources_[interaction.sourceIndex];
        subscriptionSets[source.sourcePatch.ownerRank].insert(
            std::make_tuple(source.sourcePatch.patchId, source.spatialKey,
                            static_cast<int>(FmmSubscriptionKind::Multipole)));
    }

    // Split this rank's request into waves no larger than maxLetWaveBytes, so
    // peak LET memory is set by configuration rather than by the union of all
    // local near fields. The subscriber owns the decision and ships the wave
    // index inside the subscription, so owners need no extra communication to
    // know which wave a record belongs to.
    typedef std::tuple<FmmPatchKey, std::uint64_t, int> SourceIdentity;
    std::map<SourceIdentity, std::size_t> waveBySource;
    std::size_t localWaveCount = 1;
    {
        std::vector<std::pair<SourceIdentity, std::size_t>> costs;
        for(const auto& entry : subscriptionSets)
        {
            for(const auto& item : entry.second)
            {
                const FmmPatchKey sourcePatch{entry.first, std::get<0>(item)};
                std::size_t bytes = payloadHeaderBytes(
                    compactMultipolePayload_);
                if(std::get<2>(item) == static_cast<int>(FmmSubscriptionKind::Multipole))
                {
                    bytes += multipoleCoefficientCount *
                        multipoleWireElementBytes(compactMultipolePayload_);
                }
                else
                {
                    const auto patchIt = remoteDescriptors_.find(sourcePatch);
                    if(patchIt == remoteDescriptors_.end())
                        throw UniversalError(
                            "FmmLetPlan::build: missing descriptor patch for wave sizing");
                    const auto nodeIt = patchIt->second.find(std::get<1>(item));
                    if(nodeIt == patchIt->second.end())
                        throw UniversalError(
                            "FmmLetPlan::build: missing descriptor for wave sizing");
                    bytes += static_cast<std::size_t>(
                        nodeIt->second.particleCount) *
                        particleWireBytes(compactParticlePayload_,
                                          quantizedParticlePayload_);
                    bytes += particleRecordScaleBytes(
                        quantizedParticlePayload_);
                }
                costs.push_back(std::make_pair(
                    SourceIdentity(sourcePatch, std::get<1>(item), std::get<2>(item)),
                    bytes));
            }
        }
        std::sort(costs.begin(), costs.end(),
            [](const std::pair<SourceIdentity, std::size_t>& a,
               const std::pair<SourceIdentity, std::size_t>& b)
            {
                return a.first < b.first;
            });

        std::size_t wave = 0;
        std::size_t waveBytes = 0;
        for(const auto& cost : costs)
        {
            if(maxLetWaveBytes != 0 && cost.second > maxLetWaveBytes)
            {
                char detail[512];
                std::snprintf(detail, sizeof(detail),
                    "ownerRank=%d patchId=%llu spatialKey=%llu kind=%d "
                    "recordBytes=%zu maxLetWaveBytes=%zu",
                    std::get<0>(cost.first).ownerRank,
                    static_cast<unsigned long long>(
                        std::get<0>(cost.first).patchId),
                    static_cast<unsigned long long>(std::get<1>(cost.first)),
                    std::get<2>(cost.first), cost.second, maxLetWaveBytes);
                abortLetInvariant(comm_,
                    "FmmLetPlan::build: one LET payload record exceeds the wave budget",
                    detail);
            }
            if(maxLetWaveBytes != 0 && waveBytes != 0 &&
               cost.second > maxLetWaveBytes - waveBytes)
            {
                ++wave;
                waveBytes = 0;
            }
            waveBySource[cost.first] = wave;
            waveBytes += cost.second;
        }
        localWaveCount = wave + 1;
    }

    // Every rank must call the neighborhood collective the same number of
    // times, so the wave count is a global maximum. Ranks needing fewer waves
    // contribute empty groups for the remainder.
    unsigned long long localWaves =
        static_cast<unsigned long long>(localWaveCount);
    unsigned long long globalWaves = 0;
    MPI_Allreduce(&localWaves, &globalWaves, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, comm_);
    waveCount_ = static_cast<std::size_t>(globalWaves);
    if(waveCount_ == 0)
        waveCount_ = 1;
    stats.letWaveCount = waveCount_;
    stats.letLocalWaveCount = localWaveCount;
    {
        const char* waveLog = std::getenv("RICH_FMM_GEOM_LOG");
        if(waveLog != nullptr && waveLog[0] != '\0' &&
           !(waveLog[0] == '0' && waveLog[1] == '\0'))
        {
            std::size_t requestedBytes = 0;
            for(const auto& entry : subscriptionSets)
            {
                for(const auto& item : entry.second)
                {
                    requestedBytes += payloadHeaderBytes(
                        compactMultipolePayload_);
                    if(std::get<2>(item) ==
                       static_cast<int>(FmmSubscriptionKind::Multipole))
                    {
                        requestedBytes +=
                            multipoleCoefficientCount *
                            multipoleWireElementBytes(
                                compactMultipolePayload_);
                        continue;
                    }
                    const FmmPatchKey sourcePatch{entry.first, std::get<0>(item)};
                    const auto patchIt = remoteDescriptors_.find(sourcePatch);
                    if(patchIt == remoteDescriptors_.end())
                        continue;
                    const auto nodeIt = patchIt->second.find(std::get<1>(item));
                    if(nodeIt == patchIt->second.end())
                        continue;
                    requestedBytes += static_cast<std::size_t>(
                        nodeIt->second.particleCount) *
                        particleWireBytes(compactParticlePayload_,
                                          quantizedParticlePayload_);
                    requestedBytes += particleRecordScaleBytes(
                        quantizedParticlePayload_);
                }
            }
            std::fprintf(stderr,
                "FMMWAVE rank=%d localWaves=%zu globalWaves=%zu "
                "requestedBytes=%zu maxLetWaveBytes=%zu\n",
                rank_, localWaveCount, waveCount_, requestedBytes,
                maxLetWaveBytes);
            std::fflush(stderr);
        }
    }

    std::unordered_map<int, std::vector<char>> subscriptionBuffers;
    for(const auto& entry : subscriptionSets)
    {
        std::vector<FmmSubscription>& subscriptions = subscriptionsToSend_[entry.first];
        for(const auto& item : entry.second)
        {
            if(subscriptions.size() >= static_cast<std::size_t>(
                   std::numeric_limits<std::uint32_t>::max()))
                throw UniversalError(
                    "FmmLetPlan::build: too many subscriptions for one owner");
            FmmSubscription subscription;
            subscription.stamp = fmmPacketStamp(FmmPacketKind::Subscription,
                                                topologyEpoch_);
            subscription.patchId = std::get<0>(item);
            subscription.spatialKey = std::get<1>(item);
            subscription.kind = std::get<2>(item);
            subscription.ownerSubscriptionIndex =
                static_cast<std::uint32_t>(subscriptions.size());
            const FmmPatchKey sourcePatch{entry.first, subscription.patchId};
            const FmmRemoteNodeKey sourceKey{
                sourcePatch, subscription.spatialKey};
            if(subscription.kind ==
               static_cast<int>(FmmSubscriptionKind::Multipole))
            {
                const auto sourceIt = sourceIndexByKey.find(sourceKey);
                if(sourceIt == sourceIndexByKey.end())
                    throw UniversalError(
                        "FmmLetPlan::build: subscription missing multipole source index");
                subscription.receiverSourceIndex = sourceIt->second;
                m2lSources_[sourceIt->second].ownerSubscriptionIndex =
                    subscription.ownerSubscriptionIndex;
            }
            else
            {
                const auto sourceIt = p2pSourceIndexByKey.find(sourceKey);
                if(sourceIt == p2pSourceIndexByKey.end())
                    throw UniversalError(
                        "FmmLetPlan::build: subscription missing particle source index");
                subscription.receiverSourceIndex = sourceIt->second;
                p2pSources_[sourceIt->second].ownerSubscriptionIndex =
                    subscription.ownerSubscriptionIndex;
            }
            const auto waveIt = waveBySource.find(
                SourceIdentity(sourcePatch, subscription.spatialKey,
                               subscription.kind));
            if(waveIt == waveBySource.end())
                throw UniversalError(
                    "FmmLetPlan::build: subscription without an assigned wave");
            subscription.waveIndex = static_cast<int>(waveIt->second);
            subscriptions.push_back(subscription);
            FmmPacketIO::appendPod(subscriptionBuffers[entry.first], subscription);
        }
    }
    // Reorder interactions by wave and target. Warm persistent trees contain
    // many empty leaves; target-contiguous ranges let execution skip every
    // interaction of one empty target with a single particle-count test.
    {
        std::vector<std::size_t> m2lSourceWave(m2lSources_.size(), 0);
        for(std::size_t i = 0; i < m2lSources_.size(); ++i)
        {
            const auto waveIt = waveBySource.find(SourceIdentity(
                m2lSources_[i].sourcePatch, m2lSources_[i].spatialKey,
                static_cast<int>(FmmSubscriptionKind::Multipole)));
            if(waveIt == waveBySource.end())
                throw UniversalError(
                    "FmmLetPlan::build: multipole source without an assigned wave");
            m2lSourceWave[i] = waveIt->second;
        }
        std::vector<std::size_t> p2pSourceWave(p2pSources_.size(), 0);
        for(std::size_t i = 0; i < p2pSources_.size(); ++i)
        {
            const auto waveIt = waveBySource.find(SourceIdentity(
                p2pSources_[i].sourcePatch, p2pSources_[i].spatialKey,
                static_cast<int>(FmmSubscriptionKind::Particles)));
            if(waveIt == waveBySource.end())
                throw UniversalError(
                    "FmmLetPlan::build: particle source without an assigned wave");
            p2pSourceWave[i] = waveIt->second;
        }

        std::stable_sort(m2lInteractions_.begin(), m2lInteractions_.end(),
            [&m2lSourceWave](const FmmLetM2LInteraction& a,
                             const FmmLetM2LInteraction& b)
            {
                return std::make_pair(m2lSourceWave[a.sourceIndex],
                                      a.targetNode) <
                       std::make_pair(m2lSourceWave[b.sourceIndex],
                                      b.targetNode);
            });
        std::stable_sort(m2pInteractions_.begin(), m2pInteractions_.end(),
            [&m2lSourceWave](const FmmLetM2PInteraction& a,
                             const FmmLetM2PInteraction& b)
            {
                return std::make_pair(m2lSourceWave[a.sourceIndex],
                                      a.targetNode) <
                       std::make_pair(m2lSourceWave[b.sourceIndex],
                                      b.targetNode);
            });
        std::stable_sort(p2pInteractions_.begin(), p2pInteractions_.end(),
            [&p2pSourceWave](const FmmLetP2PInteraction& a,
                             const FmmLetP2PInteraction& b)
            {
                return std::make_pair(p2pSourceWave[a.sourceIndex],
                                      a.targetNode) <
                       std::make_pair(p2pSourceWave[b.sourceIndex],
                                      b.targetNode);
            });

        const auto buildRanges = [](std::size_t count,
                                    const std::vector<std::size_t>& waveOf,
                                    std::vector<std::pair<std::size_t,
                                                          std::size_t>>& ranges)
        {
            ranges.assign(count, std::make_pair(std::size_t(0), std::size_t(0)));
            std::size_t cursor = 0;
            for(std::size_t wave = 0; wave < count; ++wave)
            {
                const std::size_t begin = cursor;
                while(cursor < waveOf.size() && waveOf[cursor] == wave)
                    ++cursor;
                ranges[wave] = std::make_pair(begin, cursor);
            }
            if(cursor != waveOf.size())
                throw UniversalError(
                    "FmmLetPlan::build: interaction wave partition is not contiguous");
        };

        std::vector<std::size_t> sortedWaves(m2lInteractions_.size());
        for(std::size_t i = 0; i < m2lInteractions_.size(); ++i)
            sortedWaves[i] = m2lSourceWave[m2lInteractions_[i].sourceIndex];
        buildRanges(waveCount_, sortedWaves, m2lWaveRanges_);

        sortedWaves.assign(m2pInteractions_.size(), 0);
        for(std::size_t i = 0; i < m2pInteractions_.size(); ++i)
            sortedWaves[i] = m2lSourceWave[m2pInteractions_[i].sourceIndex];
        buildRanges(waveCount_, sortedWaves, m2pWaveRanges_);

        sortedWaves.assign(p2pInteractions_.size(), 0);
        for(std::size_t i = 0; i < p2pInteractions_.size(); ++i)
            sortedWaves[i] = p2pSourceWave[p2pInteractions_[i].sourceIndex];
        buildRanges(waveCount_, sortedWaves, p2pWaveRanges_);

        const auto buildTargetRanges = [](
            const auto& interactions,
            const std::vector<std::pair<std::size_t, std::size_t>>& waveRanges,
            std::vector<InteractionTargetRange>& targetRanges,
            std::vector<std::pair<std::size_t, std::size_t>>& targetWaveRanges)
        {
            if(interactions.size() > static_cast<std::size_t>(
                   std::numeric_limits<std::uint32_t>::max()))
                throw UniversalError(
                    "FmmLetPlan::build: target-grouped interaction index overflow");
            targetRanges.clear();
            targetWaveRanges.assign(
                waveRanges.size(), std::make_pair(std::size_t(0),
                                                  std::size_t(0)));
            for(std::size_t wave = 0; wave < waveRanges.size(); ++wave)
            {
                const std::size_t targetRangeBegin = targetRanges.size();
                std::size_t cursor = waveRanges[wave].first;
                while(cursor < waveRanges[wave].second)
                {
                    const std::size_t begin = cursor;
                    const std::uint32_t target = interactions[cursor].targetNode;
                    do
                    {
                        ++cursor;
                    }
                    while(cursor < waveRanges[wave].second &&
                          interactions[cursor].targetNode == target);
                    targetRanges.push_back(InteractionTargetRange{
                        target, static_cast<std::uint32_t>(begin),
                        static_cast<std::uint32_t>(cursor)});
                }
                targetWaveRanges[wave] = std::make_pair(
                    targetRangeBegin, targetRanges.size());
            }
        };
        buildTargetRanges(m2lInteractions_, m2lWaveRanges_,
                          m2lTargetRanges_, m2lTargetWaveRanges_);
        buildTargetRanges(m2pInteractions_, m2pWaveRanges_,
                          m2pTargetRanges_, m2pTargetWaveRanges_);
        buildTargetRanges(p2pInteractions_, p2pWaveRanges_,
                          p2pTargetRanges_, p2pTargetWaveRanges_);
    }

    FmmPeerExchangeResult receivedSubscriptions = exchange_.exchangeBytes(
        subscriptionBuffers, &stats.bytesSent, &stats.bytesReceived);
    for(const FmmReceivedMessage& message : receivedSubscriptions.messages())
    {
        const FmmByteView view = receivedSubscriptions.view(message);
        std::size_t offset = 0;
        std::set<std::tuple<std::uint64_t, std::uint64_t, int>> unique;
        while(offset < view.size)
        {
            const FmmSubscription subscription =
                FmmPacketIO::readPod<FmmSubscription>(view, offset);
            validateFmmPacketStamp(subscription.stamp,
                FmmPacketKind::Subscription, topologyEpoch_,
                "FmmLetPlan::build subscription");
            if(subscription.patchId == 0)
                throw UniversalError("FmmLetPlan::build: subscription missing patch ID");
            if(subscription.patchId != FMM_COMPAT_PATCH_ID)
                throw UniversalError(
                    "FmmLetPlan::build: non-compat patch ID in subscription");
            if(subscription.waveIndex < 0 ||
               static_cast<std::size_t>(subscription.waveIndex) >= waveCount_)
                throw UniversalError(
                    "FmmLetPlan::build: subscription wave index out of range");
            const auto nodeIt = localNodeByKey_.find(subscription.spatialKey);
            if(nodeIt == localNodeByKey_.end())
                throw UniversalError("FmmLetPlan::build: subscription references missing local node");
            if(subscription.kind == static_cast<int>(FmmSubscriptionKind::Particles) &&
               !localTree.nodes()[nodeIt->second].isLeaf())
                throw UniversalError("FmmLetPlan::build: particle subscription references non-leaf");
            if(subscription.kind != static_cast<int>(FmmSubscriptionKind::Particles) &&
               subscription.kind != static_cast<int>(FmmSubscriptionKind::Multipole))
                throw UniversalError("FmmLetPlan::build: invalid subscription kind");
            if(!unique.insert(std::make_tuple(subscription.patchId,
                                              subscription.spatialKey,
                                              subscription.kind)).second)
                throw UniversalError("FmmLetPlan::build: duplicate subscription");
            std::vector<ResolvedSubscription>& receivedSubscriptions =
                subscriptionsReceived_[message.source];
            if(subscription.ownerSubscriptionIndex !=
               receivedSubscriptions.size())
                throw UniversalError(
                    "FmmLetPlan::build: subscription owner index mismatch");
            if(nodeIt->second > static_cast<std::size_t>(
                   std::numeric_limits<std::uint32_t>::max()))
                throw UniversalError(
                    "FmmLetPlan::build: subscribed local node index overflow");
            receivedSubscriptions.push_back(ResolvedSubscription{
                subscription, static_cast<std::uint32_t>(nodeIt->second)});
        }
    }

    stats.letSubscriptionSeconds += elapsed(subscriptionStart);
    const Clock::time_point compactionStart = Clock::now();

    // Descriptor pulling may visit many intermediate remote nodes.  Only
    // descriptors referenced by terminal M2L/P2P interactions are needed by
    // warm solves; discard the rest before the persistent plan is measured.
    std::set<FmmRemoteNodeKey> retainedDescriptorKeys;
    for(const auto& terminal : terminalKeys)
        retainedDescriptorKeys.insert(
            FmmRemoteNodeKey{std::get<1>(terminal), std::get<2>(terminal)});
    for(const FmmRemoteNodeKey& key : retainedDescriptorKeys)
    {
        const auto patchIt = remoteDescriptors_.find(key.patch);
        if(patchIt == remoteDescriptors_.end())
            throw UniversalError(
                "FmmLetPlan::build: terminal descriptor patch disappeared");
        if(patchIt->second.find(key.spatialKey) == patchIt->second.end())
            throw UniversalError(
                "FmmLetPlan::build: terminal descriptor disappeared");
    }
    for(auto patchIt = remoteDescriptors_.begin();
        patchIt != remoteDescriptors_.end();)
    {
        auto& descriptors = patchIt->second;
        for(auto nodeIt = descriptors.begin(); nodeIt != descriptors.end();)
        {
            if(retainedDescriptorKeys.count(
                   FmmRemoteNodeKey{patchIt->first, nodeIt->first}) == 0)
                nodeIt = descriptors.erase(nodeIt);
            else
                ++nodeIt;
        }
        if(descriptors.empty() && !reuseBuildStorage)
            patchIt = remoteDescriptors_.erase(patchIt);
        else
            ++patchIt;
    }

    // Leaf-only moving-mesh rebuilds are frequent and similar in size.  Keep
    // their buckets/vector capacity so the next rebuild amortizes allocation.
    // A full rank-root/process-topology rebuild is rare and is a natural point
    // to compact a transient high-water mark.
    if(!reuseBuildStorage)
    {
        localNodeByKey_.rehash(0);
        remoteDescriptors_.rehash(0);
        for(auto& entry : remoteDescriptors_)
            entry.second.rehash(0);
        m2lInteractions_.shrink_to_fit();
        m2lSources_.shrink_to_fit();
        m2lOperatorGeometries_.shrink_to_fit();
        m2lOperatorGeometryUseCounts_.shrink_to_fit();
        p2pInteractions_.shrink_to_fit();
        p2pSources_.shrink_to_fit();
        m2pInteractions_.shrink_to_fit();
        m2lTargetRanges_.shrink_to_fit();
        p2pTargetRanges_.shrink_to_fit();
        m2pTargetRanges_.shrink_to_fit();
        m2lTargetWaveRanges_.shrink_to_fit();
        p2pTargetWaveRanges_.shrink_to_fit();
        m2pTargetWaveRanges_.shrink_to_fit();
        activeM2LSourceFlags_.shrink_to_fit();
        activeP2PSourceFlags_.shrink_to_fit();
        pendingScratch_.shrink_to_fit();
        workScratch_.shrink_to_fit();
        blockedScratch_.shrink_to_fit();
        subscriptionsToSend_.rehash(0);
        for(auto& entry : subscriptionsToSend_)
            entry.second.shrink_to_fit();
        subscriptionsReceived_.rehash(0);
        for(auto& entry : subscriptionsReceived_)
        {
            entry.second.shrink_to_fit();
        }
    }

    stats.letPruneCompactSeconds += elapsed(compactionStart);
    stats.letPlanSeconds += elapsed(start);
}

std::size_t FmmLetPlan::bytesOwned() const
{
    std::size_t result = saturatingAdd(
        exchange_.bytesOwned(), pendingExchange_.bytesOwned());
    result = saturatingAdd(result, saturatingMultiply(
        remoteLatticeRoots_.capacity(), sizeof(RemoteLatticeRoot)));
    const std::size_t localMapEntry =
        sizeof(std::pair<const std::uint64_t, std::size_t>) + 2 * sizeof(void*);
    result = saturatingAdd(result, saturatingMultiply(
        localNodeByKey_.bucket_count(), sizeof(void*)));
    result = saturatingAdd(result, saturatingMultiply(
        localNodeByKey_.size(), localMapEntry));

    const std::size_t remoteOuterEntry = sizeof(std::pair<const FmmPatchKey,
        std::unordered_map<std::uint64_t, FmmRemoteNodeDescriptor>>) +
        2 * sizeof(void*);
    const std::size_t remoteInnerEntry = sizeof(std::pair<
        const std::uint64_t, FmmRemoteNodeDescriptor>) + 2 * sizeof(void*);
    result = saturatingAdd(result, saturatingMultiply(
        remoteDescriptors_.bucket_count(), sizeof(void*)));
    result = saturatingAdd(result, saturatingMultiply(
        remoteDescriptors_.size(), remoteOuterEntry));
    for(const auto& entry : remoteDescriptors_)
    {
        result = saturatingAdd(result, saturatingMultiply(
            entry.second.bucket_count(), sizeof(void*)));
        result = saturatingAdd(result, saturatingMultiply(
            entry.second.size(), remoteInnerEntry));
    }

    result = saturatingAdd(result, saturatingMultiply(
        m2lInteractions_.capacity(), sizeof(FmmLetM2LInteraction)));
    result = saturatingAdd(result, saturatingMultiply(
        m2lSources_.capacity(), sizeof(M2LSource)));
    result = saturatingAdd(result, saturatingMultiply(
        m2lOperatorGeometries_.capacity(), sizeof(FmmM2LOperatorCache::PreparedGeometry)));
    result = saturatingAdd(result, saturatingMultiply(
        m2lOperatorGeometryUseCounts_.capacity(), sizeof(std::uint64_t)));
    result = saturatingAdd(result, saturatingMultiply(
        p2pInteractions_.capacity(), sizeof(FmmLetP2PInteraction)));
    result = saturatingAdd(result, saturatingMultiply(
        p2pSources_.capacity(), sizeof(RemoteSource)));
    result = saturatingAdd(result, saturatingMultiply(
        m2pInteractions_.capacity(), sizeof(FmmLetM2PInteraction)));
    const std::size_t waveRangeBytes = sizeof(std::pair<std::size_t, std::size_t>);
    result = saturatingAdd(result, saturatingMultiply(
        m2lWaveRanges_.capacity(), waveRangeBytes));
    result = saturatingAdd(result, saturatingMultiply(
        p2pWaveRanges_.capacity(), waveRangeBytes));
    result = saturatingAdd(result, saturatingMultiply(
        m2pWaveRanges_.capacity(), waveRangeBytes));
    result = saturatingAdd(result, saturatingMultiply(
        m2lTargetRanges_.capacity(), sizeof(InteractionTargetRange)));
    result = saturatingAdd(result, saturatingMultiply(
        p2pTargetRanges_.capacity(), sizeof(InteractionTargetRange)));
    result = saturatingAdd(result, saturatingMultiply(
        m2pTargetRanges_.capacity(), sizeof(InteractionTargetRange)));
    result = saturatingAdd(result, saturatingMultiply(
        m2lTargetWaveRanges_.capacity(), waveRangeBytes));
    result = saturatingAdd(result, saturatingMultiply(
        p2pTargetWaveRanges_.capacity(), waveRangeBytes));
    result = saturatingAdd(result, saturatingMultiply(
        m2pTargetWaveRanges_.capacity(), waveRangeBytes));
    result = saturatingAdd(result, saturatingMultiply(
        activeM2LInteractionIndices_.capacity(), sizeof(std::uint32_t)));
    result = saturatingAdd(result, saturatingMultiply(
        activeP2PInteractionIndices_.capacity(), sizeof(std::uint32_t)));
    result = saturatingAdd(result, saturatingMultiply(
        activeM2PInteractionIndices_.capacity(), sizeof(std::uint32_t)));
    result = saturatingAdd(result, saturatingMultiply(
        activeM2LSourceFlags_.capacity(), sizeof(unsigned char)));
    result = saturatingAdd(result, saturatingMultiply(
        activeP2PSourceFlags_.capacity(), sizeof(unsigned char)));
    result = saturatingAdd(result, saturatingMultiply(
        pendingScratch_.capacity(), sizeof(PendingPair)));
    result = saturatingAdd(result, saturatingMultiply(
        workScratch_.capacity(), sizeof(PendingPair)));
    result = saturatingAdd(result, saturatingMultiply(
        blockedScratch_.capacity(), sizeof(PendingPair)));

    const std::size_t sentSubscriptionMapEntry =
        sizeof(std::pair<const int, std::vector<FmmSubscription>>) +
        2 * sizeof(void*);
    result = saturatingAdd(result, saturatingMultiply(
        subscriptionsToSend_.size(), sentSubscriptionMapEntry));
    for(const auto& entry : subscriptionsToSend_)
        result = saturatingAdd(result, saturatingMultiply(
            entry.second.capacity(), sizeof(FmmSubscription)));
    const std::size_t receivedSubscriptionMapEntry = sizeof(std::pair<
        const int, std::vector<ResolvedSubscription>>) + 2 * sizeof(void*);
    result = saturatingAdd(result, saturatingMultiply(
        subscriptionsReceived_.size(), receivedSubscriptionMapEntry));
    for(const auto& entry : subscriptionsReceived_)
        result = saturatingAdd(result, saturatingMultiply(
            entry.second.capacity(), sizeof(ResolvedSubscription)));

    return result;
}

void FmmLetPlan::beginExecute(
    std::size_t wave,
    const FmmTree& localTree,
    const std::vector<Vector3D>& positions,
    const std::vector<double>& masses,
    const std::vector<std::uint64_t>& cellIds,
    const FmmTaylorExpansion& layout,
    const std::vector<double>& localMultipoles,
    const std::vector<double>& localLocals,
    const std::vector<Vector3D>& acceleration,
    const std::vector<double>* positiveKernelPotential,
    std::size_t maxRemoteBytes,
    FmmSolveStats& stats)
{
    const Clock::time_point start = Clock::now();
    if(executePending_ || pendingExchange_.active())
        throw UniversalError("FmmLetPlan::beginExecute: exchange already active");
    if(comm_ == MPI_COMM_NULL || topologyEpoch_ == 0)
        throw UniversalError("FmmLetPlan::beginExecute: LET plan is not initialized");
    if(positions.size() != masses.size() || positions.size() != cellIds.size() ||
       acceleration.size() != positions.size() ||
       localTree.particleOrder().size() != positions.size() ||
       (positiveKernelPotential != nullptr &&
        positiveKernelPotential->size() != positions.size()))
        throw UniversalError("FmmLetPlan::beginExecute: inconsistent particle or output storage");
    if(layout.coefficientCount() == 0 ||
       localTree.nodes().size() > std::numeric_limits<std::size_t>::max() /
                                  layout.coefficientCount())
        throw UniversalError("FmmLetPlan::beginExecute: expansion storage overflow");
    const std::size_t expectedExpansionSize =
        localTree.nodes().size() * layout.coefficientCount();
    if(localMultipoles.size() != expectedExpansionSize ||
       localLocals.size() != expectedExpansionSize)
        throw UniversalError("FmmLetPlan::beginExecute: inconsistent expansion storage");
    if(maxRemoteBytes < 2)
        throw UniversalError("FmmLetPlan::beginExecute: remote memory budget is too small");
    if(m2lInteractions_.size() > std::numeric_limits<std::uint32_t>::max() ||
       p2pInteractions_.size() > std::numeric_limits<std::uint32_t>::max() ||
       m2pInteractions_.size() > std::numeric_limits<std::uint32_t>::max())
        throw UniversalError("FmmLetPlan::beginExecute: interaction plan is too large");
    if(wave >= waveCount_)
        throw UniversalError("FmmLetPlan::beginExecute: wave index out of range");
    // stats is reset per solve, but the plan survives warm solves that skip
    // build(), so restate the wave count here rather than only at build time.
    stats.letWaveCount = waveCount_;
    stats.letParticlePayloadCompacted = compactParticlePayload_;
    stats.letParticlePayloadQuantized = quantizedParticlePayload_;
    stats.letMultipolePayloadCompacted = compactMultipolePayload_;
    // Only this wave's interactions are active; the rest belong to other waves
    // and must not run here, or they would be applied more than once.
    activeM2LInteractionIndices_.clear();
    activeP2PInteractionIndices_.clear();
    activeM2PInteractionIndices_.clear();
    activeM2LSourceFlags_.assign(m2lSources_.size(), 0u);
    activeP2PSourceFlags_.assign(p2pSources_.size(), 0u);
    const std::pair<std::size_t, std::size_t> m2lTargetRange =
        m2lTargetWaveRanges_[wave];
    for(std::size_t r = m2lTargetRange.first;
        r < m2lTargetRange.second; ++r)
    {
        const InteractionTargetRange& range = m2lTargetRanges_[r];
        if(range.targetNode >= localTree.nodes().size() ||
           range.begin > range.end || range.end > m2lInteractions_.size())
            throw UniversalError("FmmLetPlan::beginExecute: invalid M2L target");
        if(localTree.nodes()[range.targetNode].particleCount() == 0)
        {
            stats.letInactiveM2LCount +=
                static_cast<std::uint64_t>(range.end - range.begin);
            continue;
        }
        for(std::uint32_t i = range.begin; i < range.end; ++i)
        {
            activeM2LInteractionIndices_.push_back(i);
            const std::uint32_t sourceIndex = m2lInteractions_[i].sourceIndex;
            if(sourceIndex >= activeM2LSourceFlags_.size())
                throw UniversalError(
                    "FmmLetPlan::beginExecute: invalid active M2L source");
            activeM2LSourceFlags_[sourceIndex] = 1u;
        }
    }
    const std::pair<std::size_t, std::size_t> p2pTargetRange =
        p2pTargetWaveRanges_[wave];
    for(std::size_t r = p2pTargetRange.first;
        r < p2pTargetRange.second; ++r)
    {
        const InteractionTargetRange& range = p2pTargetRanges_[r];
        if(range.targetNode >= localTree.nodes().size() ||
           range.begin > range.end || range.end > p2pInteractions_.size())
            throw UniversalError("FmmLetPlan::beginExecute: invalid P2P target");
        if(localTree.nodes()[range.targetNode].particleCount() == 0)
        {
            stats.letInactiveP2PBlockCount +=
                static_cast<std::uint64_t>(range.end - range.begin);
            continue;
        }
        for(std::uint32_t i = range.begin; i < range.end; ++i)
        {
            activeP2PInteractionIndices_.push_back(i);
            const std::uint32_t sourceIndex = p2pInteractions_[i].sourceIndex;
            if(sourceIndex >= activeP2PSourceFlags_.size())
                throw UniversalError(
                    "FmmLetPlan::beginExecute: invalid active P2P source");
            activeP2PSourceFlags_[sourceIndex] = 1u;
        }
    }
    const std::pair<std::size_t, std::size_t> m2pTargetRange =
        m2pTargetWaveRanges_[wave];
    for(std::size_t r = m2pTargetRange.first;
        r < m2pTargetRange.second; ++r)
    {
        const InteractionTargetRange& range = m2pTargetRanges_[r];
        if(range.targetNode >= localTree.nodes().size() ||
           range.begin > range.end || range.end > m2pInteractions_.size())
            throw UniversalError("FmmLetPlan::beginExecute: invalid M2P target");
        if(localTree.nodes()[range.targetNode].particleCount() == 0)
        {
            stats.letInactiveM2PCount +=
                static_cast<std::uint64_t>(range.end - range.begin);
            continue;
        }
        for(std::uint32_t i = range.begin; i < range.end; ++i)
        {
            activeM2PInteractionIndices_.push_back(i);
            const std::uint32_t sourceIndex = m2pInteractions_[i].sourceIndex;
            if(sourceIndex >= activeM2LSourceFlags_.size())
                throw UniversalError(
                    "FmmLetPlan::beginExecute: invalid active M2P source");
            activeM2LSourceFlags_[sourceIndex] = 1u;
        }
    }
    pendingProgressCallCount_ = 0;
    pendingProgressIncompleteCount_ = 0;
    pendingCompletionProgressCall_ = 0;

    // The retained topology is a safe superset: after particles move, most of
    // its target leaves are empty. Tell each source owner exactly which of its
    // retained subscription slots still feed an occupied target. Slot IDs are
    // four bytes and are exchanged before the much larger particle payload,
    // avoiding wire, decode, and packing work for inactive target branches.
    std::unordered_map<int, std::vector<char>> activeRequestBuffers;
    const auto appendActiveRequest = [&](int ownerRank,
                                         std::uint32_t ownerSubscriptionIndex)
    {
        if(ownerRank < 0 || ownerRank == rank_ ||
           ownerSubscriptionIndex ==
               std::numeric_limits<std::uint32_t>::max())
            throw UniversalError(
                "FmmLetPlan::beginExecute: invalid active subscription slot");
        FmmPacketIO::appendPod(activeRequestBuffers[ownerRank],
                               ownerSubscriptionIndex);
    };
    for(std::size_t i = 0; i < activeM2LSourceFlags_.size(); ++i)
    {
        if(activeM2LSourceFlags_[i] == 0)
            continue;
        const M2LSource& source = m2lSources_[i];
        appendActiveRequest(source.sourcePatch.ownerRank,
                            source.ownerSubscriptionIndex);
    }
    for(std::size_t i = 0; i < activeP2PSourceFlags_.size(); ++i)
    {
        if(activeP2PSourceFlags_[i] == 0)
            continue;
        const RemoteSource& source = p2pSources_[i];
        appendActiveRequest(source.sourcePatch.ownerRank,
                            source.ownerSubscriptionIndex);
    }

    FmmPeerExchangeResult receivedActiveRequests = exchange_.exchangeBytes(
        activeRequestBuffers, &stats.bytesSent, &stats.bytesReceived,
        maxRemoteBytes);
    std::unordered_map<int, std::vector<std::uint32_t>>
        activeSubscriptionsByRank;
    for(const FmmReceivedMessage& message :
        receivedActiveRequests.messages())
    {
        const FmmByteView view = receivedActiveRequests.view(message);
        if(view.size % sizeof(std::uint32_t) != 0)
            throw UniversalError(
                "FmmLetPlan::beginExecute: malformed active subscription request");
        const auto retainedIt = subscriptionsReceived_.find(message.source);
        if(retainedIt == subscriptionsReceived_.end())
            throw UniversalError(
                "FmmLetPlan::beginExecute: request from unknown subscriber");
        std::vector<std::uint32_t>& activeSlots =
            activeSubscriptionsByRank[message.source];
        activeSlots.reserve(view.size / sizeof(std::uint32_t));
        std::vector<unsigned char> seen(retainedIt->second.size(), 0u);
        std::size_t offset = 0;
        while(offset < view.size)
        {
            const std::uint32_t slot =
                FmmPacketIO::readPod<std::uint32_t>(view, offset);
            if(slot >= retainedIt->second.size() || seen[slot] != 0)
                throw UniversalError(
                    "FmmLetPlan::beginExecute: invalid or duplicate active subscription slot");
            const ResolvedSubscription& resolved = retainedIt->second[slot];
            if(resolved.subscription.ownerSubscriptionIndex != slot ||
               resolved.subscription.waveIndex < 0 ||
               static_cast<std::size_t>(resolved.subscription.waveIndex) != wave)
                throw UniversalError(
                    "FmmLetPlan::beginExecute: active subscription slot mismatch");
            seen[slot] = 1u;
            activeSlots.push_back(slot);
        }
    }
    std::unordered_map<int, std::vector<char>>().swap(activeRequestBuffers);
    receivedActiveRequests.releaseStorage();

    std::unordered_map<int, std::size_t> plannedBytesByRank;
    std::size_t outgoingBytes = 0;
    for(const auto& entry : activeSubscriptionsByRank)
    {
        const auto retainedIt = subscriptionsReceived_.find(entry.first);
        if(retainedIt == subscriptionsReceived_.end())
            throw UniversalError(
                "FmmLetPlan::execute: active subscriber disappeared");
        std::size_t peerBytes = 0;
        for(std::uint32_t slot : entry.second)
        {
            if(slot >= retainedIt->second.size())
                throw UniversalError(
                    "FmmLetPlan::execute: active subscription slot vanished");
            const ResolvedSubscription& resolved = retainedIt->second[slot];
            const FmmSubscription& subscription = resolved.subscription;
            if(resolved.localNodeIndex >= localTree.nodes().size())
                throw UniversalError(
                    "FmmLetPlan::execute: subscribed source node vanished");
            const FmmNode& node = localTree.nodes()[resolved.localNodeIndex];
            std::size_t payload = 0;
            if(subscription.kind == static_cast<int>(FmmSubscriptionKind::Multipole))
            {
                if(node.particleCount() == 0)
                {
                    ++stats.letOmittedMultipolePayloadCount;
                    continue;
                }
                const std::size_t wireBytes = multipoleWireElementBytes(
                    compactMultipolePayload_);
                if(layout.coefficientCount() >
                   std::numeric_limits<std::size_t>::max() / wireBytes)
                    throw UniversalError("FmmLetPlan::execute: multipole payload overflow");
                payload = layout.coefficientCount() * wireBytes;
            }
            else if(subscription.kind == static_cast<int>(FmmSubscriptionKind::Particles))
            {
                if(!node.isLeaf())
                    throw UniversalError("FmmLetPlan::execute: particle subscription targets non-leaf");
                if(node.particleCount() == 0)
                {
                    ++stats.letOmittedParticlePayloadCount;
                    continue;
                }
                const std::size_t wireBytes = particleWireBytes(
                    compactParticlePayload_, quantizedParticlePayload_);
                if(node.particleCount() >
                   std::numeric_limits<std::size_t>::max() / wireBytes)
                    throw UniversalError("FmmLetPlan::execute: particle payload overflow");
                payload = node.particleCount() * wireBytes;
                const std::size_t scaleBytes = particleRecordScaleBytes(
                    quantizedParticlePayload_);
                if(payload > std::numeric_limits<std::size_t>::max() -
                              scaleBytes)
                    throw UniversalError(
                        "FmmLetPlan::execute: particle scale payload overflow");
                payload += scaleBytes;
            }
            else
            {
                throw UniversalError("FmmLetPlan::execute: unknown subscription kind");
            }
            const std::size_t headerBytes = payloadHeaderBytes(
                compactMultipolePayload_);
            if(headerBytes >
                   std::numeric_limits<std::size_t>::max() - payload ||
               peerBytes > std::numeric_limits<std::size_t>::max() -
                           headerBytes - payload)
                throw UniversalError("FmmLetPlan::execute: outgoing payload overflow");
            peerBytes += headerBytes + payload;
        }
        if(outgoingBytes > std::numeric_limits<std::size_t>::max() - peerBytes)
            throw UniversalError("FmmLetPlan::execute: total outgoing payload overflow");
        plannedBytesByRank[entry.first] = peerBytes;
        outgoingBytes += peerBytes;
    }
    if(outgoingBytes > maxRemoteBytes / 2)
    {
        char detail[512];
        std::snprintf(detail, sizeof(detail),
            "outgoingBytes=%zu maxRemoteBytes=%zu outgoingLimit=%zu",
            outgoingBytes, maxRemoteBytes, maxRemoteBytes / 2);
        abortLetInvariant(comm_,
            "FmmLetPlan::execute: outgoing LET payload exceeds memory budget",
            detail);
    }
    stats.letMaxOutgoingBytes = std::max(
        stats.letMaxOutgoingBytes, outgoingBytes);
    stats.letPayloadPlanningSeconds += elapsed(start);

    const Clock::time_point packingStart = Clock::now();
    std::unordered_map<int, std::vector<char>> sendBuffers;
    for(const auto& entry : plannedBytesByRank)
        sendBuffers[entry.first].reserve(entry.second);
    for(const auto& entry : activeSubscriptionsByRank)
    {
        const auto retainedIt = subscriptionsReceived_.find(entry.first);
        if(retainedIt == subscriptionsReceived_.end())
            throw UniversalError(
                "FmmLetPlan::execute: active packing subscriber disappeared");
        std::vector<char>& buffer = sendBuffers[entry.first];
        for(std::uint32_t slot : entry.second)
        {
            if(slot >= retainedIt->second.size())
                throw UniversalError(
                    "FmmLetPlan::execute: active packing slot vanished");
            const ResolvedSubscription& resolved = retainedIt->second[slot];
            const FmmSubscription& subscription = resolved.subscription;
            if(resolved.localNodeIndex >= localTree.nodes().size())
                throw UniversalError(
                    "FmmLetPlan::execute: active packing node vanished");
            const FmmNode& node = localTree.nodes()[resolved.localNodeIndex];
            if(node.particleCount() == 0)
                continue;
            const auto appendHeader = [&](std::size_t count)
            {
                if(compactMultipolePayload_)
                {
                    if(count > static_cast<std::size_t>(
                           std::numeric_limits<std::uint32_t>::max()))
                        throw UniversalError(
                            "FmmLetPlan::execute: compact payload count overflow");
                    FmmCompactPayloadRecordHeader header;
                    header.receiverSourceIndex =
                        subscription.receiverSourceIndex;
                    header.count = static_cast<std::uint32_t>(count);
                    header.kind = static_cast<std::uint32_t>(
                        subscription.kind);
                    FmmPacketIO::appendPod(buffer, header);
                }
                else
                {
                    FmmPayloadRecordHeader header;
                    header.stamp = fmmPacketStamp(FmmPacketKind::LetPayload,
                                                  topologyEpoch_);
                    header.patchId = subscription.patchId;
                    header.spatialKey = subscription.spatialKey;
                    header.count = static_cast<std::uint64_t>(count);
                    header.kind = subscription.kind;
                    header.waveIndex = subscription.waveIndex;
                    FmmPacketIO::appendPod(buffer, header);
                }
            };
            if(subscription.kind == static_cast<int>(FmmSubscriptionKind::Multipole))
            {
                appendHeader(layout.coefficientCount());
                if(compactMultipolePayload_)
                    appendFloatMultipoles(buffer,
                        localMultipoles.data() + node.multipoleOffset,
                        layout.coefficientCount());
                else
                    FmmPacketIO::appendDoubles(buffer,
                        localMultipoles.data() + node.multipoleOffset,
                        layout.coefficientCount());
            }
            else
            {
                appendHeader(node.particleCount());
                if(quantizedParticlePayload_)
                {
                    if(!(node.halfSize > 0.0) ||
                       !std::isfinite(node.halfSize))
                        throw UniversalError(
                            "FmmLetPlan::execute: invalid quantized source half size");
                    double massScale = 0.0;
                    for(std::size_t k = node.particleBegin;
                        k < node.particleEnd; ++k)
                    {
                        const std::size_t body = localTree.particleOrder()[k];
                        if(!std::isfinite(masses[body]))
                            throw UniversalError(
                                "FmmLetPlan::execute: non-finite quantized source mass");
                        massScale = std::max(massScale,
                                             std::abs(masses[body]));
                    }
                    FmmPacketIO::appendPod(buffer, massScale);
                    const double inverseHalfSize = 1.0 / node.halfSize;
                    const double inverseMassScale = massScale > 0.0 ?
                        1.0 / massScale : 0.0;
                    for(std::size_t k = node.particleBegin;
                        k < node.particleEnd; ++k)
                    {
                        const std::size_t body = localTree.particleOrder()[k];
                        FmmQuantizedPatchWireParticle packet;
                        packet.offset[0] = quantizeSignedUnit(
                            (positions[body].x - node.center.x) *
                            inverseHalfSize);
                        packet.offset[1] = quantizeSignedUnit(
                            (positions[body].y - node.center.y) *
                            inverseHalfSize);
                        packet.offset[2] = quantizeSignedUnit(
                            (positions[body].z - node.center.z) *
                            inverseHalfSize);
                        packet.mass = quantizeSignedUnit(
                            masses[body] * inverseMassScale);
                        FmmPacketIO::appendPod(buffer, packet);
                    }
                }
                else if(compactParticlePayload_)
                {
                    for(std::size_t k = node.particleBegin;
                        k < node.particleEnd; ++k)
                    {
                        const std::size_t body = localTree.particleOrder()[k];
                        FmmCompactPatchWireParticle packet;
                        packet.offset[0] = static_cast<float>(
                            positions[body].x - node.center.x);
                        packet.offset[1] = static_cast<float>(
                            positions[body].y - node.center.y);
                        packet.offset[2] = static_cast<float>(
                            positions[body].z - node.center.z);
                        packet.mass = static_cast<float>(masses[body]);
                        if(!std::isfinite(packet.offset[0]) ||
                           !std::isfinite(packet.offset[1]) ||
                           !std::isfinite(packet.offset[2]) ||
                           !std::isfinite(packet.mass))
                            throw UniversalError(
                                "FmmLetPlan::execute: compact particle conversion overflow");
                        FmmPacketIO::appendPod(buffer, packet);
                    }
                }
                else
                {
                    for(std::size_t k = node.particleBegin;
                        k < node.particleEnd; ++k)
                    {
                        const std::size_t body = localTree.particleOrder()[k];
                        FmmWireParticle packet;
                        packet.position[0] = positions[body].x;
                        packet.position[1] = positions[body].y;
                        packet.position[2] = positions[body].z;
                        packet.mass = masses[body];
                        packet.cellId = cellIds[body];
                        packet.ownerLocalIndex = static_cast<std::uint64_t>(body);
                        packet.ownerRank = rank_;
                        FmmPacketIO::appendPod(buffer, packet);
                    }
                }
            }
        }
        if(buffer.size() != plannedBytesByRank[entry.first])
            throw UniversalError("FmmLetPlan::execute: payload planning mismatch");
    }
    stats.letPayloadPackingSeconds += elapsed(packingStart);
    std::unordered_map<int, std::vector<std::uint32_t>>().swap(
        activeSubscriptionsByRank);

    std::size_t sendCapacityBytes = 0;
    for(const auto& entry : sendBuffers)
    {
        if(entry.second.capacity() >
           std::numeric_limits<std::size_t>::max() - sendCapacityBytes)
            throw UniversalError("FmmLetPlan::execute: send capacity overflow");
        sendCapacityBytes += entry.second.capacity();
    }
    stats.letMaxSendCapacityBytes = std::max(
        stats.letMaxSendCapacityBytes, sendCapacityBytes);
    if(sendCapacityBytes > maxRemoteBytes - outgoingBytes)
    {
        char detail[512];
        std::snprintf(detail, sizeof(detail),
            "sendCapacityBytes=%zu outgoingBytes=%zu maxRemoteBytes=%zu "
            "remainingBudget=%zu",
            sendCapacityBytes, outgoingBytes, maxRemoteBytes,
            maxRemoteBytes - outgoingBytes);
        abortLetInvariant(comm_,
            "FmmLetPlan::execute: MPI send scratch exceeds memory budget",
            detail);
    }

    // The per-peer payload and contiguous MPI send scratch coexist during
    // the neighborhood collective. Bound both, then release the per-peer
    // payload before decoding so wire and decoded payload are the only copies.
    const std::size_t receiveLimit = std::min(
        maxRemoteBytes - sendCapacityBytes - outgoingBytes,
        maxRemoteBytes / 2);
    FmmPeerExchangeTimings exchangeTimings;
    exchange_.beginExchangeBytes(
        sendBuffers, pendingExchange_, receiveLimit,
        maxRemoteBytes - sendCapacityBytes, &exchangeTimings);
    stats.letPayloadFlattenSeconds += exchangeTimings.flattenSeconds;
    stats.letCountExchangeSeconds += exchangeTimings.countExchangeSeconds;
    stats.letReceiveSetupSeconds += exchangeTimings.receiveSetupSeconds;
    stats.letPayloadLaunchSeconds += exchangeTimings.payloadLaunchSeconds;
    stats.peakRemoteBytes = std::max(stats.peakRemoteBytes,
        sendCapacityBytes + pendingExchange_.bytesOwned());
    const Clock::time_point releaseStart = Clock::now();
    std::unordered_map<int, std::vector<char>>().swap(sendBuffers);
    stats.letPayloadReleaseSeconds += elapsed(releaseStart);

    pendingMaxRemoteBytes_ = maxRemoteBytes;
    pendingExchangePreparationSeconds_ = elapsed(start);
    stats.letPreparationSeconds += pendingExchangePreparationSeconds_;
    executePending_ = true;
}

void FmmLetPlan::progressExecute()
{
    if(!executePending_)
        return;
    ++pendingProgressCallCount_;
    if(!pendingExchange_.progress())
    {
        ++pendingProgressIncompleteCount_;
        return;
    }
    if(pendingCompletionProgressCall_ == 0)
        pendingCompletionProgressCall_ = pendingProgressCallCount_;
}

void FmmLetPlan::finishExecute(
    std::size_t wave,
    const FmmTree& localTree,
    const std::vector<Vector3D>& positions,
    const FmmTaylorExpansion& layout,
    std::vector<double>& localLocals,
    std::vector<Vector3D>& acceleration,
    std::vector<double>* positiveKernelPotential,
    FmmM2LOperatorCache& operatorCache,
    std::size_t maxRemoteBytes,
    std::size_t maxOperatorCacheBytes,
    FmmSolveStats& stats)
{
    if(!executePending_ || maxRemoteBytes != pendingMaxRemoteBytes_)
        throw UniversalError(
            "FmmLetPlan::finishExecute: no matching active exchange");
    const Clock::time_point finishStart = Clock::now();
    const bool completedBeforeFinish = pendingExchange_.completedByProgress();
    FmmPeerExchangeResult received = pendingExchange_.wait(
        &stats.bytesSent, &stats.bytesReceived);
    stats.letPayloadLifetimeSeconds +=
        pendingExchange_.payloadLifetimeSeconds();
    stats.letResidualWaitSeconds += pendingExchange_.residualWaitSeconds();
    stats.letProgressCallCount += pendingProgressCallCount_;
    stats.letProgressIncompleteCount += pendingProgressIncompleteCount_;
    stats.letCompletionProgressCall += pendingCompletionProgressCall_;
    stats.letCompletedBeforeFinishCount += completedBeforeFinish ? 1u : 0u;
    executePending_ = false;
    const std::size_t exchangeWorkspaceBytes = pendingExchange_.bytesOwned();
    const std::size_t receivedWorkspaceBytes = received.bytesOwned();
    stats.letMaxIncomingBytes = std::max(
        stats.letMaxIncomingBytes, received.totalBytes());
    const std::size_t wireWorkspaceBytes = saturatingAdd(
        exchangeWorkspaceBytes, receivedWorkspaceBytes);
    if(wireWorkspaceBytes == std::numeric_limits<std::size_t>::max() ||
       wireWorkspaceBytes > maxRemoteBytes)
    {
        char detail[512];
        std::snprintf(detail, sizeof(detail),
            "wireWorkspaceBytes=%zu maxRemoteBytes=%zu "
            "exchangeWorkspaceBytes=%zu receivedWorkspaceBytes=%zu",
            wireWorkspaceBytes, maxRemoteBytes, exchangeWorkspaceBytes,
            receivedWorkspaceBytes);
        abortLetInvariant(comm_,
            "FmmLetPlan::finishExecute: exchange workspace exceeds memory budget",
            detail);
    }
    stats.peakRemoteBytes = std::max(
        stats.peakRemoteBytes, wireWorkspaceBytes);
    stats.letMaxWavePayloadBytes = std::max(
        stats.letMaxWavePayloadBytes, receivedWorkspaceBytes);

    // Owners answer only the active slot request sent at beginExecute(). Use
    // that exact upper bound for decoded storage instead of reserving for the
    // much larger persistent topology superset.
    const std::size_t expectedMultipoleRecords =
        static_cast<std::size_t>(std::count(
            activeM2LSourceFlags_.begin(), activeM2LSourceFlags_.end(), 1u));
    const std::size_t expectedParticleRecords =
        static_cast<std::size_t>(std::count(
            activeP2PSourceFlags_.begin(), activeP2PSourceFlags_.end(), 1u));
    std::size_t decodedRequestedBytes = 0;
    const auto addRequestedBytes = [&](std::size_t count, std::size_t elementSize)
    {
        const std::size_t bytes = saturatingMultiply(count, elementSize);
        decodedRequestedBytes = saturatingAdd(decodedRequestedBytes, bytes);
    };
    addRequestedBytes(expectedMultipoleRecords, sizeof(RemoteMultipolePayload));
    addRequestedBytes(expectedParticleRecords, sizeof(RemoteParticlePayload));
    if(compactMultipolePayload_)
    {
        const std::size_t coefficientSlots = saturatingMultiply(
            expectedMultipoleRecords, layout.coefficientCount());
        addRequestedBytes(coefficientSlots, sizeof(double));
    }
    addRequestedBytes(m2lSources_.size(), sizeof(FmmNode));
    addRequestedBytes(m2lSources_.size(), sizeof(unsigned char));
    addRequestedBytes(m2lSources_.size(), sizeof(const double*));
    addRequestedBytes(p2pSources_.size(),
                      sizeof(const RemoteParticlePayload*));
    if(decodedRequestedBytes == std::numeric_limits<std::size_t>::max() ||
       decodedRequestedBytes > maxRemoteBytes - wireWorkspaceBytes)
    {
        char detail[512];
        std::snprintf(detail, sizeof(detail),
            "decodedRequestedBytes=%zu maxRemoteBytes=%zu "
            "wireWorkspaceBytes=%zu remainingBudget=%zu",
            decodedRequestedBytes, maxRemoteBytes, wireWorkspaceBytes,
            maxRemoteBytes - wireWorkspaceBytes);
        abortLetInvariant(comm_,
            "FmmLetPlan::execute: decoded LET payload exceeds memory budget",
            detail);
    }

    std::vector<RemoteMultipolePayload> remoteMultipoles;
    std::vector<RemoteParticlePayload> remoteParticles;
    std::vector<double> decodedMultipoleCoefficients;
    std::vector<FmmNode> resolvedM2LSources;
    std::vector<unsigned char> resolvedM2LSourceActive;
    std::vector<const double*> remoteMultipoleCoefficients(
        m2lSources_.size(), nullptr);
    std::vector<const RemoteParticlePayload*> remoteParticlePayloads(
        p2pSources_.size(), nullptr);
    remoteMultipoles.reserve(expectedMultipoleRecords);
    remoteParticles.reserve(expectedParticleRecords);
    if(compactMultipolePayload_)
    {
        const std::size_t coefficientSlots = saturatingMultiply(
            expectedMultipoleRecords, layout.coefficientCount());
        if(coefficientSlots == std::numeric_limits<std::size_t>::max())
            throw UniversalError(
                "FmmLetPlan::execute: decoded multipole count overflow");
        decodedMultipoleCoefficients.reserve(coefficientSlots);
    }
    resolvedM2LSources.reserve(m2lSources_.size());
    resolvedM2LSourceActive.reserve(m2lSources_.size());

    std::size_t decodedBytes = 0;
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        remoteMultipoles.capacity(), sizeof(RemoteMultipolePayload)));
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        remoteParticles.capacity(), sizeof(RemoteParticlePayload)));
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        decodedMultipoleCoefficients.capacity(), sizeof(double)));
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        resolvedM2LSources.capacity(), sizeof(FmmNode)));
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        resolvedM2LSourceActive.capacity(), sizeof(unsigned char)));
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        remoteMultipoleCoefficients.capacity(), sizeof(const double*)));
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        remoteParticlePayloads.capacity(),
        sizeof(const RemoteParticlePayload*)));
    if(decodedBytes == std::numeric_limits<std::size_t>::max() ||
       decodedBytes > maxRemoteBytes - wireWorkspaceBytes)
    {
        char detail[512];
        std::snprintf(detail, sizeof(detail),
            "decodedBytes=%zu maxRemoteBytes=%zu wireWorkspaceBytes=%zu "
            "remainingBudget=%zu",
            decodedBytes, maxRemoteBytes, wireWorkspaceBytes,
            maxRemoteBytes - wireWorkspaceBytes);
        abortLetInvariant(comm_,
            "FmmLetPlan::execute: allocated LET payload exceeds memory budget",
            detail);
    }

    const Clock::time_point decodeStart = Clock::now();
    for(const FmmReceivedMessage& message : received.messages())
    {
        const FmmByteView view = received.view(message);
        std::size_t offset = 0;
        while(offset < view.size)
        {
            FmmPayloadRecordHeader header;
            std::uint32_t receiverSourceIndex =
                std::numeric_limits<std::uint32_t>::max();
            if(compactMultipolePayload_)
            {
                const FmmCompactPayloadRecordHeader compactHeader =
                    FmmPacketIO::readPod<FmmCompactPayloadRecordHeader>(
                        view, offset);
                if(compactHeader.kind > static_cast<std::uint32_t>(
                       std::numeric_limits<int>::max()))
                    throw UniversalError(
                        "FmmLetPlan::execute: compact payload kind overflow");
                header.patchId = FMM_COMPAT_PATCH_ID;
                header.count = compactHeader.count;
                header.kind = static_cast<int>(compactHeader.kind);
                header.waveIndex = static_cast<int>(wave);
                receiverSourceIndex = compactHeader.receiverSourceIndex;
                if(header.kind ==
                   static_cast<int>(FmmSubscriptionKind::Multipole))
                {
                    if(receiverSourceIndex >= m2lSources_.size())
                        throw UniversalError(
                            "FmmLetPlan::execute: compact multipole source index overflow");
                    const M2LSource& source =
                        m2lSources_[receiverSourceIndex];
                    if(source.sourcePatch.ownerRank != message.source)
                        throw UniversalError(
                            "FmmLetPlan::execute: compact multipole source rank mismatch");
                    header.spatialKey = source.spatialKey;
                }
                else if(header.kind ==
                        static_cast<int>(FmmSubscriptionKind::Particles))
                {
                    if(receiverSourceIndex >= p2pSources_.size())
                        throw UniversalError(
                            "FmmLetPlan::execute: compact particle source index overflow");
                    const RemoteSource& source =
                        p2pSources_[receiverSourceIndex];
                    if(source.sourcePatch.ownerRank != message.source)
                        throw UniversalError(
                            "FmmLetPlan::execute: compact particle source rank mismatch");
                    header.spatialKey = source.spatialKey;
                }
            }
            else
            {
                header = FmmPacketIO::readPod<FmmPayloadRecordHeader>(
                    view, offset);
                validateFmmPacketStamp(header.stamp,
                    FmmPacketKind::LetPayload, topologyEpoch_,
                    "FmmLetPlan::execute LET payload decode");
                if(header.patchId != FMM_COMPAT_PATCH_ID)
                    throw UniversalError(
                        "FmmLetPlan::execute: non-compat patch ID in payload");
                if(header.waveIndex < 0 ||
                   static_cast<std::size_t>(header.waveIndex) != wave)
                    throw UniversalError(
                        "FmmLetPlan::execute: payload arrived in wrong LET wave");
            }
            const FmmPatchKey patch{message.source, header.patchId};
            const auto patchIt = remoteDescriptors_.find(patch);
            if(patchIt == remoteDescriptors_.end())
                throw UniversalError(
                    "FmmLetPlan::execute: decoded payload patch vanished");
            const auto descriptorIt = patchIt->second.find(
                header.spatialKey);
            if(descriptorIt == patchIt->second.end())
                throw UniversalError(
                    "FmmLetPlan::execute: decoded payload descriptor vanished");
            if(header.kind == static_cast<int>(FmmSubscriptionKind::Multipole))
            {
                if(header.count != 0 &&
                   header.count != static_cast<std::uint64_t>(
                       layout.coefficientCount()))
                    throw UniversalError(
                        "FmmLetPlan::execute: multipole order mismatch");
                const bool active = header.count != 0;
                const double* coefficients = nullptr;
                if(active)
                {
                    const std::size_t count =
                        static_cast<std::size_t>(header.count);
                    if(remoteMultipoles.size() >= expectedMultipoleRecords)
                        throw UniversalError(
                            "FmmLetPlan::execute: excess multipole payload");
                    if(compactMultipolePayload_)
                    {
                        if(count > FmmPacketIO::remaining(view, offset) /
                                   sizeof(float))
                            throw UniversalError(
                                "FmmLetPlan::execute: truncated compact multipole payload");
                        const float* compactCoefficients = packetArray<float>(
                            view, offset, count);
                        const std::size_t coefficientOffset =
                            decodedMultipoleCoefficients.size();
                        if(count > decodedMultipoleCoefficients.capacity() -
                                   coefficientOffset)
                            throw UniversalError(
                                "FmmLetPlan::execute: excess decoded multipole payload");
                        decodedMultipoleCoefficients.resize(
                            coefficientOffset + count);
                        for(std::size_t i = 0; i < count; ++i)
                        {
                            const double converted = compactCoefficients[i];
                            if(!std::isfinite(converted))
                                throw UniversalError(
                                    "FmmLetPlan::execute: malformed compact multipole");
                            decodedMultipoleCoefficients[
                                coefficientOffset + i] = converted;
                        }
                        coefficients = decodedMultipoleCoefficients.data() +
                            coefficientOffset;
                        offset += count * sizeof(float);
                    }
                    else
                    {
                        if(count > FmmPacketIO::remaining(view, offset) /
                                   sizeof(double))
                            throw UniversalError(
                                "FmmLetPlan::execute: truncated multipole payload");
                        coefficients = packetArray<double>(view, offset, count);
                        offset += count * sizeof(double);
                    }
                }
                remoteMultipoles.push_back(RemoteMultipolePayload{
                    patch, header.spatialKey, receiverSourceIndex,
                    coefficients, active});
            }
            else if(header.kind ==
                    static_cast<int>(FmmSubscriptionKind::Particles))
            {
                if(remoteParticles.size() >= expectedParticleRecords)
                    throw UniversalError(
                        "FmmLetPlan::execute: excess particle payload");
                if(header.count > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max()))
                    throw UniversalError(
                        "FmmLetPlan::execute: particle count overflow");
                const std::size_t count = static_cast<std::size_t>(header.count);
                double massScale = 0.0;
                if(quantizedParticlePayload_)
                {
                    massScale = FmmPacketIO::readPod<double>(view, offset);
                    if(!(massScale >= 0.0) || !std::isfinite(massScale))
                        throw UniversalError(
                            "FmmLetPlan::execute: malformed quantized mass scale");
                }
                const std::size_t wireBytes = particleWireBytes(
                    compactParticlePayload_, quantizedParticlePayload_);
                if(count > FmmPacketIO::remaining(view, offset) / wireBytes)
                    throw UniversalError(
                        "FmmLetPlan::execute: truncated particle payload");
                if(descriptorIt->second.isLeaf == 0)
                    throw UniversalError(
                        "FmmLetPlan::execute: particle payload references non-leaf descriptor");
                descriptorIt->second.particleCount = header.count;
                if(quantizedParticlePayload_)
                {
                    const FmmQuantizedPatchWireParticle* particles =
                        packetArray<FmmQuantizedPatchWireParticle>(
                            view, offset, count);
                    remoteParticles.push_back(RemoteParticlePayload{
                        patch, header.spatialKey, receiverSourceIndex,
                        nullptr, nullptr, particles,
                        descriptorIt->second.centerVector(),
                        descriptorIt->second.halfSize, massScale, count});
                }
                else if(compactParticlePayload_)
                {
                    const FmmCompactPatchWireParticle* particles =
                        packetArray<FmmCompactPatchWireParticle>(
                            view, offset, count);
                    remoteParticles.push_back(RemoteParticlePayload{
                        patch, header.spatialKey, receiverSourceIndex,
                        nullptr, particles, nullptr,
                        descriptorIt->second.centerVector(),
                        descriptorIt->second.halfSize, 0.0, count});
                    for(std::size_t i = 0; i < count; ++i)
                    {
                        const FmmCompactPatchWireParticle& particle =
                            particles[i];
                        if(!std::isfinite(particle.offset[0]) ||
                           !std::isfinite(particle.offset[1]) ||
                           !std::isfinite(particle.offset[2]) ||
                           !std::isfinite(particle.mass))
                            throw UniversalError(
                                "FmmLetPlan::execute: malformed compact remote particle");
                    }
                }
                else
                {
                    const FmmWireParticle* particles =
                        packetArray<FmmWireParticle>(view, offset, count);
                    remoteParticles.push_back(RemoteParticlePayload{
                        patch, header.spatialKey, receiverSourceIndex,
                        particles, nullptr, nullptr,
                        descriptorIt->second.centerVector(),
                        descriptorIt->second.halfSize, 0.0, count});
                    for(std::size_t i = 0; i < count; ++i)
                    {
                        const FmmWireParticle& particle = particles[i];
                        if(particle.ownerRank != message.source ||
                           !finiteVector(particle.positionVector()) ||
                           !std::isfinite(particle.mass))
                            throw UniversalError(
                                "FmmLetPlan::execute: malformed remote particle");
                    }
                }
                offset += count * wireBytes;
            }
            else
                throw UniversalError(
                    "FmmLetPlan::execute: unknown payload kind");
        }
    }
    if(remoteMultipoles.size() > expectedMultipoleRecords ||
       remoteParticles.size() > expectedParticleRecords)
        throw UniversalError(
            "FmmLetPlan::execute: decoded LET payload size mismatch");

    if(!compactMultipolePayload_)
    {
        if(!std::is_sorted(remoteMultipoles.begin(), remoteMultipoles.end(),
                           payloadLess<RemoteMultipolePayload>))
            std::sort(remoteMultipoles.begin(), remoteMultipoles.end(),
                      payloadLess<RemoteMultipolePayload>);
        if(!std::is_sorted(remoteParticles.begin(), remoteParticles.end(),
                           payloadLess<RemoteParticlePayload>))
            std::sort(remoteParticles.begin(), remoteParticles.end(),
                      payloadLess<RemoteParticlePayload>);
    }
    stats.peakRemoteBytes = std::max(stats.peakRemoteBytes,
        wireWorkspaceBytes + decodedBytes);
    stats.letDecodeSeconds += elapsed(decodeStart);
    // Report only exchange work exposed on the critical path: preparation,
    // residual wait, and decode. Transfer time hidden by local traversal is
    // intentionally excluded from this phase timer.
    stats.letExchangeSeconds += pendingExchangePreparationSeconds_ +
        elapsed(finishStart);

    std::vector<double> derivativeScratch;
    derivativeScratch.reserve(layout.coefficientCount());
    std::vector<double> uncachedOperator;
    uncachedOperator.reserve(layout.m2lTerms().size());
    const Clock::time_point m2lStart = Clock::now();
    if(activeM2LInteractionIndices_.size() > m2lInteractions_.size())
        throw UniversalError(
            "FmmLetPlan::execute: resolved LET M2L source table mismatch");
    for(const RemoteMultipolePayload& payload : remoteMultipoles)
    {
        std::size_t sourceIndex = payload.sourceIndex;
        if(sourceIndex == static_cast<std::size_t>(
               std::numeric_limits<std::uint32_t>::max()))
        {
            const auto found = std::lower_bound(
                m2lSources_.begin(), m2lSources_.end(),
                std::make_pair(payload.sourcePatch, payload.spatialKey),
                [](const M2LSource& source,
                   const std::pair<FmmPatchKey, std::uint64_t>& key)
                {
                    return std::make_pair(source.sourcePatch,
                                          source.spatialKey) < key;
                });
            if(found == m2lSources_.end() ||
               found->sourcePatch != payload.sourcePatch ||
               found->spatialKey != payload.spatialKey)
                throw UniversalError(
                    "FmmLetPlan::execute: multipole payload references unknown source");
            sourceIndex = static_cast<std::size_t>(
                found - m2lSources_.begin());
        }
        if(sourceIndex >= m2lSources_.size() ||
           m2lSources_[sourceIndex].sourcePatch != payload.sourcePatch ||
           m2lSources_[sourceIndex].spatialKey != payload.spatialKey)
            throw UniversalError(
                "FmmLetPlan::execute: multipole payload source index mismatch");
        if(remoteMultipoleCoefficients[sourceIndex] != nullptr)
            throw UniversalError(
                "FmmLetPlan::execute: duplicate multipole source payload");
        remoteMultipoleCoefficients[sourceIndex] = payload.coefficients;
    }
    for(const RemoteParticlePayload& payload : remoteParticles)
    {
        std::size_t sourceIndex = payload.sourceIndex;
        if(sourceIndex == static_cast<std::size_t>(
               std::numeric_limits<std::uint32_t>::max()))
        {
            const auto found = std::lower_bound(
                p2pSources_.begin(), p2pSources_.end(),
                std::make_pair(payload.sourcePatch, payload.spatialKey),
                [](const RemoteSource& source,
                   const std::pair<FmmPatchKey, std::uint64_t>& key)
                {
                    return std::make_pair(source.sourcePatch,
                                          source.spatialKey) < key;
                });
            if(found == p2pSources_.end() ||
               found->sourcePatch != payload.sourcePatch ||
               found->spatialKey != payload.spatialKey)
                throw UniversalError(
                    "FmmLetPlan::execute: particle payload references unknown source");
            sourceIndex = static_cast<std::size_t>(
                found - p2pSources_.begin());
        }
        if(sourceIndex >= p2pSources_.size() ||
           p2pSources_[sourceIndex].sourcePatch != payload.sourcePatch ||
           p2pSources_[sourceIndex].spatialKey != payload.spatialKey)
            throw UniversalError(
                "FmmLetPlan::execute: particle payload source index mismatch");
        if(remoteParticlePayloads[sourceIndex] != nullptr)
            throw UniversalError(
                "FmmLetPlan::execute: duplicate particle source payload");
        remoteParticlePayloads[sourceIndex] = &payload;
    }
    for(std::size_t i = 0; i < m2lSources_.size(); ++i)
    {
        FmmNode source = m2lSources_[i].node;
        resolvedM2LSources.push_back(source);
        resolvedM2LSourceActive.push_back(
            remoteMultipoleCoefficients[i] != nullptr ? 1u : 0u);
    }

    std::vector<std::uint64_t> activeGeometryUseCounts(
        m2lOperatorGeometries_.size(), 0);
    std::size_t activeM2LCount = 0;
    for(std::uint32_t interactionIndex : activeM2LInteractionIndices_)
    {
        const FmmLetM2LInteraction& interaction =
            m2lInteractions_[interactionIndex];
        const std::uint32_t sourceIndex = interaction.sourceIndex;
        const std::uint32_t geometryIndex = interaction.geometryIndex;
        if(static_cast<std::size_t>(sourceIndex) >=
               resolvedM2LSourceActive.size() ||
           static_cast<std::size_t>(geometryIndex) >=
               activeGeometryUseCounts.size() ||
           interaction.targetNode >= localTree.nodes().size())
            throw UniversalError(
                "FmmLetPlan::execute: invalid retained M2L interaction");
        if(resolvedM2LSourceActive[sourceIndex] == 0 ||
           remoteMultipoleCoefficients[sourceIndex] == nullptr)
        {
            ++stats.letInactiveM2LCount;
            continue;
        }
        if(activeGeometryUseCounts[geometryIndex] ==
           std::numeric_limits<std::uint64_t>::max())
            throw UniversalError(
                "FmmLetPlan::execute: active geometry count overflow");
        ++activeGeometryUseCounts[geometryIndex];
        ++activeM2LCount;
    }

    operatorCache.configure(maxOperatorCacheBytes, layout.m2lTerms().size(),
                            activeM2LCount);
    operatorCache.beginPhase();
    std::vector<const std::vector<double>*> resolvedOperators;
    const bool operatorsResolved = operatorCache.resolvePreparedBatch(
        m2lOperatorGeometries_, activeGeometryUseCounts, layout,
        derivativeScratch, uncachedOperator, resolvedOperators);
    if(!operatorsResolved)
    {
        // Target-major retained order accelerates the ordinary fully-resolved
        // path. Preserve grouped scratch-operator reuse when a deliberately
        // small cache forces the legacy fallback.
        std::sort(activeM2LInteractionIndices_.begin(),
                  activeM2LInteractionIndices_.end(),
            [this](std::uint32_t first, std::uint32_t second)
            {
                return m2lInteractions_[first].geometryIndex <
                       m2lInteractions_[second].geometryIndex;
            });
    }

    const std::vector<double>* groupedOperator = nullptr;
    std::uint64_t groupedKeyX = 0;
    std::uint64_t groupedKeyY = 0;
    std::uint64_t groupedKeyZ = 0;
    std::uint64_t groupedKeyKind = 0;
    bool haveGroupedOperator = false;
    for(std::uint32_t interactionIndex : activeM2LInteractionIndices_)
    {
        const FmmLetM2LInteraction& interaction =
            m2lInteractions_[interactionIndex];
        const std::uint32_t sourceIndex = interaction.sourceIndex;
        if(static_cast<std::size_t>(sourceIndex) >= resolvedM2LSources.size())
            throw UniversalError(
                "FmmLetPlan::execute: invalid resolved LET M2L source index");
        if(resolvedM2LSourceActive[sourceIndex] == 0 ||
           remoteMultipoleCoefficients[sourceIndex] == nullptr)
            continue;
        const std::uint32_t geometryIndex = interaction.geometryIndex;
        if(static_cast<std::size_t>(geometryIndex) >=
           m2lOperatorGeometries_.size())
            throw UniversalError(
                "FmmLetPlan::execute: invalid prepared LET M2L geometry index");
        const FmmM2LOperatorCache::PreparedGeometry& geometry =
            m2lOperatorGeometries_[geometryIndex];
        const FmmNode& source = resolvedM2LSources[sourceIndex];
        const FmmNode& target = localTree.nodes()[interaction.targetNode];
        const std::vector<double>* coefficients = nullptr;
        const double inverseScale = geometry.inverseScale;
        if(operatorsResolved)
        {
            coefficients = resolvedOperators[geometryIndex];
        }
        else
        {
            const bool sameOperator = haveGroupedOperator &&
                geometry.keyX == groupedKeyX &&
                geometry.keyY == groupedKeyY &&
                geometry.keyZ == groupedKeyZ &&
                geometry.keyKind == groupedKeyKind;
            if(!sameOperator)
            {
                const FmmM2LOperatorCache::Lookup translationOperator =
                    operatorCache.getPrepared(geometry, layout,
                                              derivativeScratch,
                                              uncachedOperator);
                groupedOperator = translationOperator.coefficients;
                groupedKeyX = geometry.keyX;
                groupedKeyY = geometry.keyY;
                groupedKeyZ = geometry.keyZ;
                groupedKeyKind = geometry.keyKind;
                haveGroupedOperator = true;
            }
            coefficients = groupedOperator;
        }
        if(coefficients == nullptr)
            throw UniversalError(
                "FmmLetPlan::execute: missing grouped M2L operator");

        FmmKernels::translateM2LRaw(
            source, target, layout, remoteMultipoleCoefficients[sourceIndex],
            localLocals, *coefficients, inverseScale);
        ++stats.m2lCount;
        ++stats.letM2LCount;
    }
    stats.letOperatorCacheEntries = operatorCache.entries();
    stats.letOperatorCacheMaxEntries = operatorCache.maxEntries();
    stats.letOperatorCacheBytes = operatorCache.bytesOwned();
    stats.letOperatorCacheHits = operatorCache.hits();
    stats.letOperatorCacheMisses = operatorCache.misses();
    stats.letOperatorCacheBypasses = operatorCache.bypasses();
    stats.letOperatorIntegerKeyHits = operatorCache.integerKeyHits();
    stats.letOperatorIntegerKeyMisses = operatorCache.integerKeyMisses();
    stats.letM2LSeconds += elapsed(m2lStart);

    // M2P evaluates the remote multipole at each target particle instead of
    // translating it into a local expansion, so the target extent never enters
    // the accuracy bound. Runs before the multipole payloads are released.
    const Clock::time_point m2pStart = Clock::now();
    const bool directOrder2M2P = layout.order() == 2;
    std::vector<double> m2pLocals;
    if(!directOrder2M2P)
        m2pLocals.resize(layout.coefficientCount(), 0.0);
    for(std::uint32_t interactionIndex : activeM2PInteractionIndices_)
    {
        const FmmLetM2PInteraction& interaction =
            m2pInteractions_[interactionIndex];
        const std::uint32_t sourceIndex = interaction.sourceIndex;
        if(static_cast<std::size_t>(sourceIndex) >= resolvedM2LSources.size())
            throw UniversalError(
                "FmmLetPlan::execute: invalid resolved LET M2P source index");
        if(resolvedM2LSourceActive[sourceIndex] == 0 ||
           remoteMultipoleCoefficients[sourceIndex] == nullptr)
        {
            ++stats.letInactiveM2PCount;
            continue;
        }
        const FmmNode& source = resolvedM2LSources[sourceIndex];
        const FmmNode& targetNode = localTree.nodes()[interaction.targetNode];
        if(!targetNode.isLeaf())
            throw UniversalError("FmmLetPlan::execute: M2P target is not a leaf");
        for(std::size_t k = targetNode.particleBegin;
            k < targetNode.particleEnd; ++k)
        {
            const std::size_t body = localTree.particleOrder()[k];
            const Vector3D displacement = positions[body] - source.center;
            if(directOrder2M2P)
            {
                double* bodyPotential = positiveKernelPotential == nullptr ?
                    nullptr : &(*positiveKernelPotential)[body];
                FmmKernels::accumulateM2POrder2(
                    displacement, remoteMultipoleCoefficients[sourceIndex],
                    acceleration[body], bodyPotential);
                ++stats.letM2PCount;
                continue;
            }

            // A zero-radius single-particle target centred on the body turns
            // the existing M2L plus L2P pair into a direct evaluation.
            FmmNode pointTarget;
            pointTarget.center = positions[body];
            pointTarget.halfSize = 0.0;
            pointTarget.radius = 0.0;
            pointTarget.particleBegin = k;
            pointTarget.particleEnd = k + 1;
            pointTarget.localOffset = 0;
            FmmKernels::computeM2LOperator(displacement, layout,
                                           derivativeScratch, uncachedOperator);
            std::fill(m2pLocals.begin(), m2pLocals.end(), 0.0);
            FmmKernels::translateM2LRaw(
                source, pointTarget, layout,
                remoteMultipoleCoefficients[sourceIndex], m2pLocals,
                uncachedOperator, 1.0);
            FmmKernels::evaluateL2P(pointTarget, positions,
                                    localTree.particleOrder(), layout,
                                    m2pLocals, acceleration,
                                    positiveKernelPotential);
            ++stats.letM2PCount;
        }
    }
    stats.letM2PSeconds += elapsed(m2pStart);

    std::vector<RemoteMultipolePayload>().swap(remoteMultipoles);

    const Clock::time_point p2pStart = Clock::now();
    for(std::uint32_t interactionIndex : activeP2PInteractionIndices_)
    {
        const FmmLetP2PInteraction& interaction =
            p2pInteractions_[interactionIndex];
        if(interaction.sourceIndex >= p2pSources_.size())
            throw UniversalError("FmmLetPlan::execute: invalid P2P source index");
        const RemoteParticlePayload* particlePayload =
            remoteParticlePayloads[interaction.sourceIndex];
        if(particlePayload == nullptr)
        {
            ++stats.letInactiveP2PBlockCount;
            continue;
        }
        const FmmNode& targetNode = localTree.nodes()[interaction.targetNode];
        if(!targetNode.isLeaf())
            throw UniversalError("FmmLetPlan::execute: P2P target is not a leaf");
        if(particlePayload->particleCount == 0)
        {
            ++stats.letInactiveP2PBlockCount;
            continue;
        }
        if(particlePayload->quantizedParticles != nullptr)
        {
            const double positionScale =
                particlePayload->sourceHalfSize / 32767.0;
            const double massScale = particlePayload->massScale / 32767.0;
            for(std::size_t ti = targetNode.particleBegin;
                ti < targetNode.particleEnd; ++ti)
            {
                const std::size_t target = localTree.particleOrder()[ti];
                const Vector3D relativeTarget =
                    positions[target] - particlePayload->sourceCenter;
                for(std::size_t sourceIndex = 0;
                    sourceIndex < particlePayload->particleCount; ++sourceIndex)
                {
                    const FmmQuantizedPatchWireParticle& source =
                        particlePayload->quantizedParticles[sourceIndex];
                    const Vector3D delta(
                        relativeTarget.x -
                            static_cast<double>(source.offset[0]) * positionScale,
                        relativeTarget.y -
                            static_cast<double>(source.offset[1]) * positionScale,
                        relativeTarget.z -
                            static_cast<double>(source.offset[2]) * positionScale);
                    const double r2 = delta.x * delta.x + delta.y * delta.y +
                                      delta.z * delta.z;
                    if(r2 == 0.0)
                    {
                        abortLetInvariant(comm_,
                            "FmmLetPlan::execute: coincident distinct MPI bodies");
                    }
                    const double mass =
                        static_cast<double>(source.mass) * massScale;
                    const double invR = 1.0 / std::sqrt(r2);
                    const double invR3 = invR * invR * invR;
                    acceleration[target] -= mass * delta * invR3;
                    if(positiveKernelPotential != nullptr)
                        (*positiveKernelPotential)[target] += mass * invR;
                    ++stats.p2pPairCount;
                    ++stats.letP2PPairCount;
                }
            }
        }
        else if(particlePayload->compactParticles != nullptr)
        {
            for(std::size_t ti = targetNode.particleBegin;
                ti < targetNode.particleEnd; ++ti)
            {
                const std::size_t target = localTree.particleOrder()[ti];
                const Vector3D relativeTarget =
                    positions[target] - particlePayload->sourceCenter;
                for(std::size_t sourceIndex = 0;
                    sourceIndex < particlePayload->particleCount; ++sourceIndex)
                {
                    const FmmCompactPatchWireParticle& source =
                        particlePayload->compactParticles[sourceIndex];
                    const Vector3D delta(
                        relativeTarget.x - static_cast<double>(source.offset[0]),
                        relativeTarget.y - static_cast<double>(source.offset[1]),
                        relativeTarget.z - static_cast<double>(source.offset[2]));
                    const double r2 = delta.x * delta.x + delta.y * delta.y +
                                      delta.z * delta.z;
                    if(r2 == 0.0)
                    {
                        abortLetInvariant(comm_,
                            "FmmLetPlan::execute: coincident distinct MPI bodies");
                    }
                    const double mass = static_cast<double>(source.mass);
                    const double invR = 1.0 / std::sqrt(r2);
                    const double invR3 = invR * invR * invR;
                    acceleration[target] -= mass * delta * invR3;
                    if(positiveKernelPotential != nullptr)
                        (*positiveKernelPotential)[target] += mass * invR;
                    ++stats.p2pPairCount;
                    ++stats.letP2PPairCount;
                }
            }
        }
        else
        {
            for(std::size_t ti = targetNode.particleBegin;
                ti < targetNode.particleEnd; ++ti)
            {
                const std::size_t target = localTree.particleOrder()[ti];
                for(std::size_t sourceIndex = 0;
                    sourceIndex < particlePayload->particleCount; ++sourceIndex)
                {
                    const FmmWireParticle& source =
                        particlePayload->particles[sourceIndex];
                    if(source.ownerRank == rank_ &&
                       source.ownerLocalIndex == static_cast<std::uint64_t>(target))
                        continue;
                    if(source.ownerRank == rank_)
                        throw UniversalError(
                            "FmmLetPlan::execute: remote payload carries local body token");
                    const Vector3D delta =
                        positions[target] - source.positionVector();
                    const double r2 = delta.x * delta.x + delta.y * delta.y +
                                      delta.z * delta.z;
                    if(r2 == 0.0)
                    {
                        abortLetInvariant(comm_,
                            "FmmLetPlan::execute: coincident distinct MPI bodies");
                    }
                    const double invR = 1.0 / std::sqrt(r2);
                    const double invR3 = invR * invR * invR;
                    acceleration[target] -= source.mass * delta * invR3;
                    if(positiveKernelPotential != nullptr)
                        (*positiveKernelPotential)[target] += source.mass * invR;
                    ++stats.p2pPairCount;
                    ++stats.letP2PPairCount;
                }
            }
        }
        ++stats.p2pBlockCount;
        ++stats.letP2PBlockCount;
    }
    stats.letP2PSeconds += elapsed(p2pStart);
}

void FmmLetPlan::execute(const FmmTree& localTree,
                         const std::vector<Vector3D>& positions,
                         const std::vector<double>& masses,
                         const std::vector<std::uint64_t>& cellIds,
                         const FmmTaylorExpansion& layout,
                         const std::vector<double>& localMultipoles,
                         std::vector<double>& localLocals,
                         std::vector<Vector3D>& acceleration,
                         std::vector<double>* positiveKernelPotential,
                         FmmM2LOperatorCache& operatorCache,
                         std::size_t maxRemoteBytes,
                         std::size_t maxOperatorCacheBytes,
                         FmmSolveStats& stats)
{
    for(std::size_t wave = 0; wave < waveCount_; ++wave)
    {
        beginExecute(wave, localTree, positions, masses, cellIds, layout,
                     localMultipoles, localLocals, acceleration,
                     positiveKernelPotential, maxRemoteBytes, stats);
        finishExecute(wave, localTree, positions, layout, localLocals,
                      acceleration, positiveKernelPotential, operatorCache,
                      maxRemoteBytes, maxOperatorCacheBytes, stats);
    }
}

#endif // RICH_MPI
