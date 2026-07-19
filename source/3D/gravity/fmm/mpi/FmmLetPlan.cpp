#include "3D/gravity/fmm/mpi/FmmLetPlan.hpp"

#ifdef RICH_MPI

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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

bool cubesOverlap(const FmmNode& target, const FmmRemoteNodeDescriptor& source)
{
    const Vector3D center = source.centerVector();
    return std::abs(target.center.x - center.x) <= target.halfSize + source.halfSize &&
           std::abs(target.center.y - center.y) <= target.halfSize + source.halfSize &&
           std::abs(target.center.z - center.z) <= target.halfSize + source.halfSize;
}

double geometricRadius(const FmmNode& node)
{
    return std::sqrt(3.0) * node.halfSize;
}

bool validRemoteDescriptor(const FmmRemoteNodeDescriptor& descriptor)
{
    const Vector3D center = descriptor.centerVector();
    return finiteVector(center) &&
        descriptor.halfSize > 0.0 && std::isfinite(descriptor.halfSize) &&
        descriptor.halfSize <= std::numeric_limits<double>::max() / std::sqrt(3.0) &&
        std::isfinite(center.x - descriptor.halfSize) &&
        std::isfinite(center.x + descriptor.halfSize) &&
        std::isfinite(center.y - descriptor.halfSize) &&
        std::isfinite(center.y + descriptor.halfSize) &&
        std::isfinite(center.z - descriptor.halfSize) &&
        std::isfinite(center.z + descriptor.halfSize) &&
        descriptor.spatialKey != 0 && descriptor.particleCount != 0 &&
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

struct RemoteMultipolePayload
{
    int sourceRank = -1;
    std::uint64_t spatialKey = 0;
    std::size_t coefficientOffset = 0;
};

struct RemoteParticlePayload
{
    int sourceRank = -1;
    std::uint64_t spatialKey = 0;
    std::size_t particleOffset = 0;
    std::size_t particleCount = 0;
};

template<typename Payload>
bool payloadLess(const Payload& first, const Payload& second)
{
    return std::tie(first.sourceRank, first.spatialKey) <
           std::tie(second.sourceRank, second.spatialKey);
}

template<typename Payload>
const Payload* findPayload(const std::vector<Payload>& payloads,
                           int sourceRank,
                           std::uint64_t spatialKey)
{
    const auto found = std::lower_bound(payloads.begin(), payloads.end(),
        std::make_pair(sourceRank, spatialKey),
        [](const Payload& payload, const std::pair<int, std::uint64_t>& key)
        {
            return std::make_pair(payload.sourceRank, payload.spatialKey) < key;
        });
    if(found == payloads.end() || found->sourceRank != sourceRank ||
       found->spatialKey != spatialKey)
        return nullptr;
    return &*found;
}

[[noreturn]] void abortLetInvariant(const MPI_Comm& comm, const char* message)
{
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
    executePending_(false), pendingMaxRemoteBytes_(0),
    pendingExchangePreparationSeconds_(0.0), pendingProgressCallCount_(0),
    pendingProgressIncompleteCount_(0), pendingCompletionProgressCall_(0),
    comm_(MPI_COMM_NULL), rank_(0), topologyEpoch_(0) {}

FmmRemoteNodeDescriptor FmmLetPlan::descriptorForNode(
    const FmmNode& node,
    int sourceRank,
    std::uint64_t topologyEpoch)
{
    FmmRemoteNodeDescriptor result;
    result.center[0] = node.center.x;
    result.center[1] = node.center.y;
    result.center[2] = node.center.z;
    result.halfSize = node.halfSize;
    result.spatialKey = node.spatialKey;
    result.particleCount = static_cast<std::uint64_t>(node.particleCount());
    result.topologyEpoch = topologyEpoch;
    result.sourceRank = sourceRank;
    result.isLeaf = node.isLeaf() ? 1 : 0;
    result.childMask = static_cast<int>(node.childMask);
    return result;
}

bool FmmLetPlan::admissible(const FmmNode& target,
                            const FmmRemoteNodeDescriptor& source,
                            double thetaCritical)
{
    if(cubesOverlap(target, source))
        return false;
    const Vector3D delta = target.center - source.centerVector();
    const double distance = std::sqrt(delta.x * delta.x + delta.y * delta.y +
                                      delta.z * delta.z);
    return distance > 0.0 &&
        geometricRadius(target) + source.geometricRadius() <= thetaCritical * distance;
}

void FmmLetPlan::build(const FmmTree& localTree,
                       const std::vector<FmmRankRootDescriptor>& rootDescriptors,
                       const FmmProcessPairPlan& processPlan,
                       double thetaCritical,
                       std::uint64_t topologyEpoch,
                       const MPI_Comm& comm,
                       bool reuseBuildStorage,
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
    m2lSourceIndices_.clear();
    m2lOperatorGeometries_.clear();
    m2lOperatorGeometryIndices_.clear();
    m2lOperatorGeometryUseCounts_.clear();
    p2pInteractions_.clear();
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

    std::vector<int> peers = processPlan.letSourceRanks;
    peers.insert(peers.end(), processPlan.letTargetRanks.begin(),
                 processPlan.letTargetRanks.end());
    std::sort(peers.begin(), peers.end());
    peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
    stats.letCommunicatorReused =
        !exchange_.resetIfChanged(comm_, peers);

    const Clock::time_point descriptorStart = Clock::now();
    std::vector<PendingPair>& pending = pendingScratch_;
    for(int sourceRank : processPlan.letSourceRanks)
    {
        if(sourceRank < 0 ||
           static_cast<std::size_t>(sourceRank) >= rootDescriptors.size())
            throw UniversalError("FmmLetPlan::build: invalid source rank");
        const FmmRankRootDescriptor& root =
            rootDescriptors[static_cast<std::size_t>(sourceRank)];
        if(root.active == 0 || root.epoch != topologyEpoch_ ||
           root.magic != FMM_MPI_PACKET_MAGIC ||
           root.version != FMM_MPI_PACKET_VERSION ||
           root.latticeId == 0 || root.latticeHalfUnits == 0 ||
           root.latticeHalfUnits > static_cast<std::uint64_t>(
               std::numeric_limits<std::int64_t>::max()) ||
           (!localTree.nodes().empty() &&
            root.latticeId != localTree.nodes()[0].latticeId))
            throw UniversalError("FmmLetPlan::build: invalid or stale LET source root");
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
        descriptor.spatialKey = 1;
        descriptor.particleCount = root.particleCount;
        descriptor.topologyEpoch = topologyEpoch_;
        descriptor.sourceRank = sourceRank;
        descriptor.isLeaf = root.rootLeaf;
        descriptor.childMask = root.childMask;
        remoteDescriptors_[sourceRank][1] = descriptor;
        if(!localTree.nodes().empty())
            pending.push_back(PendingPair{0, sourceRank, 1});
    }

    int descriptorRound = 0;
    while(true)
    {
        if(++descriptorRound > FMM_MAX_TREE_DEPTH + 2)
            throw UniversalError("FmmLetPlan::build: descriptor pull exceeded tree depth");
        std::unordered_map<int, std::set<std::uint64_t>> requestSets;
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
            const auto rankIt = remoteDescriptors_.find(pair.sourceRank);
            if(rankIt == remoteDescriptors_.end())
                throw UniversalError("FmmLetPlan::build: missing remote rank cache");
            const auto nodeIt = rankIt->second.find(pair.sourceKey);
            if(nodeIt == rankIt->second.end())
                throw UniversalError("FmmLetPlan::build: missing remote node descriptor");
            const FmmRemoteNodeDescriptor& source = nodeIt->second;
            if(source.topologyEpoch != topologyEpoch_ ||
               source.sourceRank != pair.sourceRank)
                throw UniversalError("FmmLetPlan::build: stale remote descriptor");

            if(admissible(target, source, thetaCritical))
            {
                m2lInteractions_.push_back(FmmLetM2LInteraction{
                    static_cast<std::uint32_t>(pair.targetNode),
                    pair.sourceRank, pair.sourceKey});
                continue;
            }
            if(target.isLeaf() && source.isLeaf != 0)
            {
                p2pInteractions_.push_back(FmmLetP2PInteraction{
                    static_cast<std::uint32_t>(pair.targetNode),
                    pair.sourceRank, pair.sourceKey});
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
                        work.push_back(PendingPair{child, pair.sourceRank, pair.sourceKey});
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
                if(rankIt->second.find(childKey) == rankIt->second.end())
                    haveAllChildren = false;
            }
            if(!haveAllChildren)
            {
                requestSets[pair.sourceRank].insert(pair.sourceKey);
                blocked.push_back(pair);
                continue;
            }
            for(int octant = 7; octant >= 0; --octant)
            {
                if((source.childMask & (1 << octant)) == 0)
                    continue;
                const std::uint64_t childKey =
                    (source.spatialKey << 3u) | static_cast<std::uint64_t>(octant);
                work.push_back(PendingPair{pair.targetNode, pair.sourceRank, childKey});
            }
        }

        std::unordered_map<int, std::vector<char>> requestBuffers;
        unsigned long long localRequestCount = 0;
        for(const auto& entry : requestSets)
        {
            for(std::uint64_t key : entry.second)
            {
                FmmDescriptorRequest request;
                request.stamp = fmmPacketStamp(FmmPacketKind::DescriptorRequest,
                                               topologyEpoch_);
                request.spatialKey = key;
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
                    reply.child = descriptorForNode(localTree.nodes()[child], rank_,
                                                    topologyEpoch_);
                    reply.childCount = childCount;
                    reply.childOrdinal = ordinal++;
                    FmmPacketIO::appendPod(replyBuffers[message.source], reply);
                }
            }
        }

        FmmPeerExchangeResult receivedReplies = exchange_.exchangeBytes(
            replyBuffers, &stats.bytesSent, &stats.bytesReceived);
        std::map<std::pair<int, std::uint64_t>,
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
                   reply.child.topologyEpoch != topologyEpoch_ ||
                   !validRemoteDescriptor(reply.child) ||
                   (reply.child.spatialKey >> 3u) != reply.requestedParentKey)
                    throw UniversalError("FmmLetPlan::build: malformed descriptor reply");
                const auto requestRank = requestSets.find(message.source);
                if(requestRank == requestSets.end() ||
                   requestRank->second.count(reply.requestedParentKey) == 0)
                    throw UniversalError("FmmLetPlan::build: unsolicited descriptor reply");
                const auto parentIt = remoteDescriptors_[message.source].find(
                    reply.requestedParentKey);
                const unsigned int childBit = 1u <<
                    static_cast<unsigned int>(reply.child.spatialKey & 7u);
                if(parentIt == remoteDescriptors_[message.source].end() ||
                   parentIt->second.isLeaf != 0 ||
                   (static_cast<unsigned int>(parentIt->second.childMask) & childBit) == 0 ||
                   bitCount(static_cast<unsigned int>(parentIt->second.childMask)) !=
                       reply.childCount)
                    throw UniversalError("FmmLetPlan::build: reply contradicts parent descriptor");
                auto& coverage = replyCoverage[std::make_pair(
                    message.source, reply.requestedParentKey)];
                if(coverage.first == 0)
                    coverage.first = reply.childCount;
                if(coverage.first != reply.childCount ||
                   !coverage.second.insert(reply.childOrdinal).second)
                    throw UniversalError("FmmLetPlan::build: inconsistent descriptor reply set");
                const auto inserted = remoteDescriptors_[message.source].emplace(
                    reply.child.spatialKey, reply.child);
                if(!inserted.second)
                {
                    const FmmRemoteNodeDescriptor& old = inserted.first->second;
                    if(old.childMask != reply.child.childMask ||
                       old.isLeaf != reply.child.isLeaf ||
                       old.particleCount != reply.child.particleCount ||
                       old.sourceRank != reply.child.sourceRank ||
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
            for(std::uint64_t parentKey : requestEntry.second)
            {
                const auto coverage = replyCoverage.find(
                    std::make_pair(requestEntry.first, parentKey));
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

    std::sort(m2lInteractions_.begin(), m2lInteractions_.end(),
        [](const FmmLetM2LInteraction& a, const FmmLetM2LInteraction& b)
        {
            return std::tie(a.targetNode, a.sourceRank, a.sourceKey) <
                   std::tie(b.targetNode, b.sourceRank, b.sourceKey);
        });
    const auto duplicateM2L = std::adjacent_find(
        m2lInteractions_.begin(), m2lInteractions_.end(),
        [](const FmmLetM2LInteraction& a, const FmmLetM2LInteraction& b)
        {
            return a.targetNode == b.targetNode && a.sourceRank == b.sourceRank &&
                   a.sourceKey == b.sourceKey;
        });
    if(duplicateM2L != m2lInteractions_.end())
        throw UniversalError("FmmLetPlan::build: duplicate LET M2L interaction");

    // Resolve each unique remote M2L source once while the topology is built.
    // Warm solves keep target-major interaction order for local-locality, but
    // use a compact direct source index instead of repeating descriptor hash
    // lookups and lattice-key reconstruction for every interaction.
    std::map<std::pair<int, std::uint64_t>, std::uint32_t> sourceIndexByKey;
    for(const FmmLetM2LInteraction& interaction : m2lInteractions_)
        sourceIndexByKey.emplace(
            std::make_pair(interaction.sourceRank, interaction.sourceKey), 0u);
    if(sourceIndexByKey.size() > static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()))
        throw UniversalError("FmmLetPlan::build: too many unique LET M2L sources");

    m2lSources_.reserve(sourceIndexByKey.size());
    std::uint32_t nextSourceIndex = 0;
    for(auto& sourceEntry : sourceIndexByKey)
    {
        sourceEntry.second = nextSourceIndex++;
        const int sourceRank = sourceEntry.first.first;
        const std::uint64_t sourceKey = sourceEntry.first.second;
        const auto descriptorRank = remoteDescriptors_.find(sourceRank);
        if(descriptorRank == remoteDescriptors_.end())
            throw UniversalError(
                "FmmLetPlan::build: missing resolved M2L source rank");
        const auto descriptorIt = descriptorRank->second.find(sourceKey);
        if(descriptorIt == descriptorRank->second.end())
            throw UniversalError(
                "FmmLetPlan::build: missing resolved M2L source descriptor");
        if(sourceRank < 0 || static_cast<std::size_t>(sourceRank) >=
                              remoteLatticeRoots_.size())
            throw UniversalError(
                "FmmLetPlan::build: missing resolved M2L lattice root");

        FmmNode source;
        source.center = descriptorIt->second.centerVector();
        source.halfSize = descriptorIt->second.halfSize;
        source.radius = descriptorIt->second.geometricRadius();
        const RemoteLatticeRoot& latticeRoot =
            remoteLatticeRoots_[static_cast<std::size_t>(sourceRank)];
        applyRemoteLatticeMetadata(latticeRoot.latticeId, latticeRoot.center,
                                   latticeRoot.halfUnits, sourceKey, source);

        M2LSource resolved;
        resolved.sourceRank = sourceRank;
        resolved.spatialKey = sourceKey;
        resolved.node = source;
        m2lSources_.push_back(resolved);
    }

    m2lSourceIndices_.reserve(m2lInteractions_.size());
    for(const FmmLetM2LInteraction& interaction : m2lInteractions_)
    {
        const auto found = sourceIndexByKey.find(
            std::make_pair(interaction.sourceRank, interaction.sourceKey));
        if(found == sourceIndexByKey.end())
            throw UniversalError(
                "FmmLetPlan::build: unresolved LET M2L source index");
        m2lSourceIndices_.push_back(found->second);
    }
    if(m2lSourceIndices_.size() != m2lInteractions_.size())
        throw UniversalError(
            "FmmLetPlan::build: incomplete LET M2L source index table");

    typedef std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                       std::uint64_t, std::uint64_t> GeometryKey;
    std::map<GeometryKey, std::uint32_t> geometryIndexByKey;
    m2lOperatorGeometryIndices_.reserve(m2lInteractions_.size());
    for(std::size_t interactionIndex = 0;
        interactionIndex < m2lInteractions_.size(); ++interactionIndex)
    {
        const FmmLetM2LInteraction& interaction =
            m2lInteractions_[interactionIndex];
        const std::uint32_t sourceIndex = m2lSourceIndices_[interactionIndex];
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
        m2lOperatorGeometryIndices_.push_back(inserted.first->second);
        if(m2lOperatorGeometryUseCounts_[inserted.first->second] ==
           std::numeric_limits<std::uint64_t>::max())
            throw UniversalError(
                "FmmLetPlan::build: M2L geometry use count overflow");
        ++m2lOperatorGeometryUseCounts_[inserted.first->second];
    }
    if(m2lOperatorGeometryIndices_.size() != m2lInteractions_.size())
        throw UniversalError(
            "FmmLetPlan::build: incomplete prepared M2L geometry table");

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
    std::vector<std::uint32_t> orderedSourceIndices(orderedCount);
    std::vector<std::uint32_t> orderedGeometryIndices(orderedCount);
    for(std::size_t interactionIndex = 0;
        interactionIndex < m2lInteractions_.size(); ++interactionIndex)
    {
        const std::uint32_t geometryIndex =
            m2lOperatorGeometryIndices_[interactionIndex];
        const std::size_t destination = nextSlot[geometryIndex]++;
        orderedInteractions[destination] = m2lInteractions_[interactionIndex];
        orderedSourceIndices[destination] = m2lSourceIndices_[interactionIndex];
        orderedGeometryIndices[destination] = geometryIndex;
    }
    m2lInteractions_.swap(orderedInteractions);
    m2lSourceIndices_.swap(orderedSourceIndices);
    m2lOperatorGeometryIndices_.swap(orderedGeometryIndices);

    std::sort(p2pInteractions_.begin(), p2pInteractions_.end(),
        [](const FmmLetP2PInteraction& a, const FmmLetP2PInteraction& b)
        {
            return std::tie(a.targetNode, a.sourceRank, a.sourceKey) <
                   std::tie(b.targetNode, b.sourceRank, b.sourceKey);
        });
    const auto duplicateP2P = std::adjacent_find(
        p2pInteractions_.begin(), p2pInteractions_.end(),
        [](const FmmLetP2PInteraction& a, const FmmLetP2PInteraction& b)
        {
            return a.targetNode == b.targetNode && a.sourceRank == b.sourceRank &&
                   a.sourceKey == b.sourceKey;
        });
    if(duplicateP2P != p2pInteractions_.end())
        throw UniversalError("FmmLetPlan::build: duplicate LET P2P interaction");

    std::set<std::tuple<std::size_t, int, std::uint64_t>> terminalKeys;
    for(const FmmLetM2LInteraction& interaction : m2lInteractions_)
        terminalKeys.insert(std::make_tuple(interaction.targetNode,
                                            interaction.sourceRank,
                                            interaction.sourceKey));
    for(const FmmLetP2PInteraction& interaction : p2pInteractions_)
    {
        if(!terminalKeys.insert(std::make_tuple(interaction.targetNode,
                                                interaction.sourceRank,
                                                interaction.sourceKey)).second)
            throw UniversalError("FmmLetPlan::build: interaction classified as both M2L and P2P");
    }

    stats.letFinalizeSeconds += elapsed(finalizeStart);
    const Clock::time_point subscriptionStart = Clock::now();

    std::unordered_map<int, std::set<std::pair<std::uint64_t, int>>> subscriptionSets;
    for(const FmmLetM2LInteraction& interaction : m2lInteractions_)
        subscriptionSets[interaction.sourceRank].insert(
            std::make_pair(interaction.sourceKey,
                           static_cast<int>(FmmSubscriptionKind::Multipole)));
    for(const FmmLetP2PInteraction& interaction : p2pInteractions_)
        subscriptionSets[interaction.sourceRank].insert(
            std::make_pair(interaction.sourceKey,
                           static_cast<int>(FmmSubscriptionKind::Particles)));

    std::unordered_map<int, std::vector<char>> subscriptionBuffers;
    for(const auto& entry : subscriptionSets)
    {
        std::vector<FmmSubscription>& subscriptions = subscriptionsToSend_[entry.first];
        for(const auto& item : entry.second)
        {
            FmmSubscription subscription;
            subscription.stamp = fmmPacketStamp(FmmPacketKind::Subscription,
                                                topologyEpoch_);
            subscription.spatialKey = item.first;
            subscription.kind = item.second;
            subscriptions.push_back(subscription);
            FmmPacketIO::appendPod(subscriptionBuffers[entry.first], subscription);
        }
    }
    FmmPeerExchangeResult receivedSubscriptions = exchange_.exchangeBytes(
        subscriptionBuffers, &stats.bytesSent, &stats.bytesReceived);
    for(const FmmReceivedMessage& message : receivedSubscriptions.messages())
    {
        const FmmByteView view = receivedSubscriptions.view(message);
        std::size_t offset = 0;
        std::set<std::pair<std::uint64_t, int>> unique;
        while(offset < view.size)
        {
            const FmmSubscription subscription =
                FmmPacketIO::readPod<FmmSubscription>(view, offset);
            validateFmmPacketStamp(subscription.stamp,
                FmmPacketKind::Subscription, topologyEpoch_,
                "FmmLetPlan::build subscription");
            const auto nodeIt = localNodeByKey_.find(subscription.spatialKey);
            if(nodeIt == localNodeByKey_.end())
                throw UniversalError("FmmLetPlan::build: subscription references missing local node");
            if(subscription.kind == static_cast<int>(FmmSubscriptionKind::Particles) &&
               !localTree.nodes()[nodeIt->second].isLeaf())
                throw UniversalError("FmmLetPlan::build: particle subscription references non-leaf");
            if(subscription.kind != static_cast<int>(FmmSubscriptionKind::Particles) &&
               subscription.kind != static_cast<int>(FmmSubscriptionKind::Multipole))
                throw UniversalError("FmmLetPlan::build: invalid subscription kind");
            if(!unique.insert(std::make_pair(subscription.spatialKey,
                                             subscription.kind)).second)
                throw UniversalError("FmmLetPlan::build: duplicate subscription");
            subscriptionsReceived_[message.source].push_back(subscription);
        }
    }

    stats.letSubscriptionSeconds += elapsed(subscriptionStart);
    const Clock::time_point compactionStart = Clock::now();

    // Descriptor pulling may visit many intermediate remote nodes.  Only
    // descriptors referenced by terminal M2L/P2P interactions are needed by
    // warm solves; discard the rest before the persistent plan is measured.
    std::set<std::pair<int, std::uint64_t>> retainedDescriptorKeys;
    for(const auto& terminal : terminalKeys)
        retainedDescriptorKeys.insert(std::make_pair(
            std::get<1>(terminal), std::get<2>(terminal)));
    for(const auto& key : retainedDescriptorKeys)
    {
        const auto rankIt = remoteDescriptors_.find(key.first);
        if(rankIt == remoteDescriptors_.end())
            throw UniversalError(
                "FmmLetPlan::build: terminal descriptor rank disappeared");
        if(rankIt->second.find(key.second) == rankIt->second.end())
            throw UniversalError(
                "FmmLetPlan::build: terminal descriptor disappeared");
    }
    for(auto rankIt = remoteDescriptors_.begin();
        rankIt != remoteDescriptors_.end();)
    {
        auto& descriptors = rankIt->second;
        for(auto nodeIt = descriptors.begin(); nodeIt != descriptors.end();)
        {
            if(retainedDescriptorKeys.count(std::make_pair(
                   rankIt->first, nodeIt->first)) == 0)
                nodeIt = descriptors.erase(nodeIt);
            else
                ++nodeIt;
        }
        if(descriptors.empty() && !reuseBuildStorage)
            rankIt = remoteDescriptors_.erase(rankIt);
        else
            ++rankIt;
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
        m2lSourceIndices_.shrink_to_fit();
        m2lOperatorGeometries_.shrink_to_fit();
        m2lOperatorGeometryIndices_.shrink_to_fit();
        m2lOperatorGeometryUseCounts_.shrink_to_fit();
        p2pInteractions_.shrink_to_fit();
        pendingScratch_.shrink_to_fit();
        workScratch_.shrink_to_fit();
        blockedScratch_.shrink_to_fit();
        for(auto* map : {&subscriptionsToSend_, &subscriptionsReceived_})
        {
            map->rehash(0);
            for(auto& entry : *map)
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

    const std::size_t remoteOuterEntry = sizeof(std::pair<const int,
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
        m2lSourceIndices_.capacity(), sizeof(std::uint32_t)));
    result = saturatingAdd(result, saturatingMultiply(
        m2lOperatorGeometries_.capacity(), sizeof(FmmM2LOperatorCache::PreparedGeometry)));
    result = saturatingAdd(result, saturatingMultiply(
        m2lOperatorGeometryIndices_.capacity(), sizeof(std::uint32_t)));
    result = saturatingAdd(result, saturatingMultiply(
        m2lOperatorGeometryUseCounts_.capacity(), sizeof(std::uint64_t)));
    result = saturatingAdd(result, saturatingMultiply(
        p2pInteractions_.capacity(), sizeof(FmmLetP2PInteraction)));
    result = saturatingAdd(result, saturatingMultiply(
        pendingScratch_.capacity(), sizeof(PendingPair)));
    result = saturatingAdd(result, saturatingMultiply(
        workScratch_.capacity(), sizeof(PendingPair)));
    result = saturatingAdd(result, saturatingMultiply(
        blockedScratch_.capacity(), sizeof(PendingPair)));

    const std::size_t subscriptionMapEntry =
        sizeof(std::pair<const int, std::vector<FmmSubscription>>) +
        2 * sizeof(void*);
    for(const auto* map : {&subscriptionsToSend_, &subscriptionsReceived_})
    {
        result = saturatingAdd(result, saturatingMultiply(
            map->size(), subscriptionMapEntry));
        for(const auto& entry : *map)
            result = saturatingAdd(result, saturatingMultiply(
                entry.second.capacity(), sizeof(FmmSubscription)));
    }

    return result;
}

void FmmLetPlan::beginExecute(
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
    pendingProgressCallCount_ = 0;
    pendingProgressIncompleteCount_ = 0;
    pendingCompletionProgressCall_ = 0;
    std::unordered_map<int, std::size_t> plannedBytesByRank;
    std::size_t outgoingBytes = 0;
    for(const auto& entry : subscriptionsReceived_)
    {
        std::size_t peerBytes = 0;
        for(const FmmSubscription& subscription : entry.second)
        {
            validateFmmPacketStamp(subscription.stamp,
                FmmPacketKind::Subscription, topologyEpoch_,
                "FmmLetPlan::execute retained subscription");
            const auto nodeIt = localNodeByKey_.find(subscription.spatialKey);
            if(nodeIt == localNodeByKey_.end())
                throw UniversalError("FmmLetPlan::execute: subscribed source node vanished");
            const FmmNode& node = localTree.nodes()[nodeIt->second];
            std::size_t payload = 0;
            if(subscription.kind == static_cast<int>(FmmSubscriptionKind::Multipole))
            {
                if(layout.coefficientCount() >
                   std::numeric_limits<std::size_t>::max() / sizeof(double))
                    throw UniversalError("FmmLetPlan::execute: multipole payload overflow");
                payload = layout.coefficientCount() * sizeof(double);
            }
            else if(subscription.kind == static_cast<int>(FmmSubscriptionKind::Particles))
            {
                if(!node.isLeaf())
                    throw UniversalError("FmmLetPlan::execute: particle subscription targets non-leaf");
                if(node.particleCount() >
                   std::numeric_limits<std::size_t>::max() / sizeof(FmmWireParticle))
                    throw UniversalError("FmmLetPlan::execute: particle payload overflow");
                payload = node.particleCount() * sizeof(FmmWireParticle);
            }
            else
            {
                throw UniversalError("FmmLetPlan::execute: unknown subscription kind");
            }
            if(sizeof(FmmPayloadRecordHeader) >
                   std::numeric_limits<std::size_t>::max() - payload ||
               peerBytes > std::numeric_limits<std::size_t>::max() -
                           sizeof(FmmPayloadRecordHeader) - payload)
                throw UniversalError("FmmLetPlan::execute: outgoing payload overflow");
            peerBytes += sizeof(FmmPayloadRecordHeader) + payload;
        }
        if(outgoingBytes > std::numeric_limits<std::size_t>::max() - peerBytes)
            throw UniversalError("FmmLetPlan::execute: total outgoing payload overflow");
        plannedBytesByRank[entry.first] = peerBytes;
        outgoingBytes += peerBytes;
    }
    if(outgoingBytes > maxRemoteBytes / 2)
        abortLetInvariant(comm_,
            "FmmLetPlan::execute: outgoing LET payload exceeds memory budget");
    stats.letPayloadPlanningSeconds += elapsed(start);

    const Clock::time_point packingStart = Clock::now();
    std::unordered_map<int, std::vector<char>> sendBuffers;
    for(const auto& entry : plannedBytesByRank)
        sendBuffers[entry.first].reserve(entry.second);
    for(const auto& entry : subscriptionsReceived_)
    {
        std::vector<char>& buffer = sendBuffers[entry.first];
        for(const FmmSubscription& subscription : entry.second)
        {
            const std::size_t nodeIndex = localNodeByKey_.find(
                subscription.spatialKey)->second;
            const FmmNode& node = localTree.nodes()[nodeIndex];
            FmmPayloadRecordHeader header;
            header.stamp = fmmPacketStamp(FmmPacketKind::LetPayload,
                                          topologyEpoch_);
            header.spatialKey = subscription.spatialKey;
            header.kind = subscription.kind;
            if(subscription.kind == static_cast<int>(FmmSubscriptionKind::Multipole))
            {
                header.count = static_cast<std::uint64_t>(layout.coefficientCount());
                FmmPacketIO::appendPod(buffer, header);
                FmmPacketIO::appendDoubles(buffer,
                    localMultipoles.data() + node.multipoleOffset,
                    layout.coefficientCount());
            }
            else
            {
                header.count = static_cast<std::uint64_t>(node.particleCount());
                FmmPacketIO::appendPod(buffer, header);
                for(std::size_t k = node.particleBegin; k < node.particleEnd; ++k)
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
        if(buffer.size() != plannedBytesByRank[entry.first])
            throw UniversalError("FmmLetPlan::execute: payload planning mismatch");
    }
    stats.letPayloadPackingSeconds += elapsed(packingStart);

    std::size_t sendCapacityBytes = 0;
    for(const auto& entry : sendBuffers)
    {
        if(entry.second.capacity() >
           std::numeric_limits<std::size_t>::max() - sendCapacityBytes)
            throw UniversalError("FmmLetPlan::execute: send capacity overflow");
        sendCapacityBytes += entry.second.capacity();
    }
    if(sendCapacityBytes > maxRemoteBytes - outgoingBytes)
        abortLetInvariant(comm_,
            "FmmLetPlan::execute: outgoing LET buffers exceed memory budget");

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
    const Clock::time_point validationStart = Clock::now();
    const std::size_t exchangeWorkspaceBytes = pendingExchange_.bytesOwned();
    const std::size_t receivedWorkspaceBytes = received.bytesOwned();
    const std::size_t wireWorkspaceBytes = saturatingAdd(
        exchangeWorkspaceBytes, receivedWorkspaceBytes);
    if(wireWorkspaceBytes == std::numeric_limits<std::size_t>::max() ||
       wireWorkspaceBytes > maxRemoteBytes)
        abortLetInvariant(comm_,
            "FmmLetPlan::finishExecute: exchange workspace exceeds memory budget");
    stats.peakRemoteBytes = std::max(
        stats.peakRemoteBytes, wireWorkspaceBytes);

    typedef std::tuple<int, std::uint64_t, int> PayloadKey;
    std::size_t expectedRecordCount = 0;
    for(const auto& entry : subscriptionsToSend_)
    {
        if(entry.second.size() >
           std::numeric_limits<std::size_t>::max() - expectedRecordCount)
            abortLetInvariant(comm_,
                "FmmLetPlan::execute: expected payload count overflow");
        expectedRecordCount += entry.second.size();
    }
    std::size_t expectedMultipoleRecords = 0;
    std::size_t expectedParticleRecords = 0;
    std::vector<PayloadKey> expectedPayloadKeys;
    expectedPayloadKeys.reserve(expectedRecordCount);
    for(const auto& entry : subscriptionsToSend_)
    {
        for(const FmmSubscription& subscription : entry.second)
        {
            expectedPayloadKeys.push_back(std::make_tuple(
                entry.first, subscription.spatialKey, subscription.kind));
            if(subscription.kind ==
               static_cast<int>(FmmSubscriptionKind::Multipole))
                ++expectedMultipoleRecords;
            else if(subscription.kind ==
                    static_cast<int>(FmmSubscriptionKind::Particles))
                ++expectedParticleRecords;
            else
                throw UniversalError(
                    "FmmLetPlan::execute: retained subscription has invalid kind");
        }
    }
    std::sort(expectedPayloadKeys.begin(), expectedPayloadKeys.end());
    if(std::adjacent_find(expectedPayloadKeys.begin(),
                          expectedPayloadKeys.end()) !=
       expectedPayloadKeys.end())
        throw UniversalError(
            "FmmLetPlan::execute: duplicate retained payload subscription");
    std::vector<PayloadKey> seenPayloadKeys;
    seenPayloadKeys.reserve(expectedRecordCount);
    const std::size_t keyTableCount = saturatingAdd(
        expectedPayloadKeys.capacity(), seenPayloadKeys.capacity());
    std::size_t keyTableBytes = saturatingMultiply(
        keyTableCount, sizeof(PayloadKey));
    if(keyTableBytes == std::numeric_limits<std::size_t>::max() ||
       keyTableBytes > maxRemoteBytes - wireWorkspaceBytes)
        abortLetInvariant(comm_,
            "FmmLetPlan::execute: payload key tables exceed memory budget");
    stats.peakRemoteBytes = std::max(stats.peakRemoteBytes,
        wireWorkspaceBytes + keyTableBytes);
    std::size_t totalCoefficientCount = 0;
    std::size_t totalParticleCount = 0;
    std::size_t actualMultipoleRecords = 0;
    std::size_t actualParticleRecords = 0;
    for(const FmmReceivedMessage& message : received.messages())
    {
        const FmmByteView view = received.view(message);
        std::size_t offset = 0;
        while(offset < view.size)
        {
            const FmmPayloadRecordHeader header =
                FmmPacketIO::readPod<FmmPayloadRecordHeader>(view, offset);
            validateFmmPacketStamp(header.stamp, FmmPacketKind::LetPayload,
                                   topologyEpoch_,
                                   "FmmLetPlan::execute LET payload preflight");
            const PayloadKey payloadKey =
                std::make_tuple(message.source, header.spatialKey, header.kind);
            if(!std::binary_search(expectedPayloadKeys.begin(),
                                   expectedPayloadKeys.end(), payloadKey))
                throw UniversalError(
                    "FmmLetPlan::execute: unsolicited LET payload record");
            seenPayloadKeys.push_back(payloadKey);
            const auto descriptorRank = remoteDescriptors_.find(message.source);
            if(descriptorRank == remoteDescriptors_.end())
                throw UniversalError(
                    "FmmLetPlan::execute: payload references unknown source rank");
            const auto descriptor = descriptorRank->second.find(header.spatialKey);
            if(descriptor == descriptorRank->second.end())
                throw UniversalError(
                    "FmmLetPlan::execute: payload references unknown descriptor");

            if(header.kind == static_cast<int>(FmmSubscriptionKind::Multipole))
            {
                if(header.count != layout.coefficientCount())
                    throw UniversalError(
                        "FmmLetPlan::execute: multipole order mismatch");
                if(layout.coefficientCount() >
                   FmmPacketIO::remaining(view, offset) / sizeof(double))
                    throw UniversalError(
                        "FmmLetPlan::execute: truncated multipole payload");
                if(totalCoefficientCount >
                   std::numeric_limits<std::size_t>::max() -
                       layout.coefficientCount())
                    abortLetInvariant(comm_,
                        "FmmLetPlan::execute: decoded coefficient count overflow");
                totalCoefficientCount += layout.coefficientCount();
                offset += layout.coefficientCount() * sizeof(double);
                ++actualMultipoleRecords;
            }
            else if(header.kind == static_cast<int>(FmmSubscriptionKind::Particles))
            {
                if(header.count > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max()))
                    throw UniversalError(
                        "FmmLetPlan::execute: particle count overflow");
                const std::size_t count = static_cast<std::size_t>(header.count);
                if(count > FmmPacketIO::remaining(view, offset) /
                           sizeof(FmmWireParticle))
                    throw UniversalError(
                        "FmmLetPlan::execute: truncated particle payload");
                if(descriptor->second.isLeaf == 0 ||
                   header.count != descriptor->second.particleCount)
                    throw UniversalError(
                        "FmmLetPlan::execute: particle payload does not match leaf descriptor");
                if(totalParticleCount >
                   std::numeric_limits<std::size_t>::max() - count)
                    abortLetInvariant(comm_,
                        "FmmLetPlan::execute: decoded particle count overflow");
                totalParticleCount += count;
                offset += count * sizeof(FmmWireParticle);
                ++actualParticleRecords;
            }
            else
            {
                throw UniversalError(
                    "FmmLetPlan::execute: unknown payload kind");
            }
        }
    }
    std::sort(seenPayloadKeys.begin(), seenPayloadKeys.end());
    if(seenPayloadKeys != expectedPayloadKeys ||
       actualMultipoleRecords != expectedMultipoleRecords ||
       actualParticleRecords != expectedParticleRecords)
        throw UniversalError(
            "FmmLetPlan::execute: missing or duplicate subscribed LET payload");
    std::vector<PayloadKey>().swap(expectedPayloadKeys);
    std::vector<PayloadKey>().swap(seenPayloadKeys);
    stats.letValidationSeconds += elapsed(validationStart);

    const Clock::time_point decodeStart = Clock::now();
    std::size_t decodedRequestedBytes = 0;
    const auto addRequestedBytes = [&](std::size_t count, std::size_t elementSize)
    {
        const std::size_t bytes = saturatingMultiply(count, elementSize);
        decodedRequestedBytes = saturatingAdd(decodedRequestedBytes, bytes);
    };
    addRequestedBytes(expectedMultipoleRecords, sizeof(RemoteMultipolePayload));
    addRequestedBytes(expectedParticleRecords, sizeof(RemoteParticlePayload));
    addRequestedBytes(totalCoefficientCount, sizeof(double));
    addRequestedBytes(totalParticleCount, sizeof(FmmWireParticle));
    addRequestedBytes(m2lSources_.size(), sizeof(FmmNode));
    if(decodedRequestedBytes == std::numeric_limits<std::size_t>::max() ||
       decodedRequestedBytes > maxRemoteBytes - wireWorkspaceBytes)
        abortLetInvariant(comm_,
            "FmmLetPlan::execute: decoded LET payload exceeds memory budget");

    std::vector<RemoteMultipolePayload> remoteMultipoles;
    std::vector<RemoteParticlePayload> remoteParticles;
    std::vector<double> remoteCoefficients(totalCoefficientCount);
    std::vector<FmmWireParticle> remoteParticleStorage(totalParticleCount);
    std::vector<FmmNode> resolvedM2LSources;
    remoteMultipoles.reserve(expectedMultipoleRecords);
    remoteParticles.reserve(expectedParticleRecords);
    resolvedM2LSources.reserve(m2lSources_.size());

    std::size_t decodedBytes = 0;
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        remoteMultipoles.capacity(), sizeof(RemoteMultipolePayload)));
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        remoteParticles.capacity(), sizeof(RemoteParticlePayload)));
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        remoteCoefficients.capacity(), sizeof(double)));
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        remoteParticleStorage.capacity(), sizeof(FmmWireParticle)));
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        resolvedM2LSources.capacity(), sizeof(FmmNode)));
    if(decodedBytes == std::numeric_limits<std::size_t>::max() ||
       decodedBytes > maxRemoteBytes - wireWorkspaceBytes)
        abortLetInvariant(comm_,
            "FmmLetPlan::execute: allocated LET payload exceeds memory budget");

    std::size_t coefficientCursor = 0;
    std::size_t particleCursor = 0;
    for(const FmmReceivedMessage& message : received.messages())
    {
        const FmmByteView view = received.view(message);
        std::size_t offset = 0;
        while(offset < view.size)
        {
            const FmmPayloadRecordHeader header =
                FmmPacketIO::readPod<FmmPayloadRecordHeader>(view, offset);
            validateFmmPacketStamp(header.stamp, FmmPacketKind::LetPayload,
                                   topologyEpoch_,
                                   "FmmLetPlan::execute LET payload decode");
            if(header.kind == static_cast<int>(FmmSubscriptionKind::Multipole))
            {
                remoteMultipoles.push_back(RemoteMultipolePayload{
                    message.source, header.spatialKey, coefficientCursor});
                FmmPacketIO::readDoubles(view, offset,
                    remoteCoefficients.data() + coefficientCursor,
                    layout.coefficientCount());
                coefficientCursor += layout.coefficientCount();
            }
            else
            {
                const std::size_t count = static_cast<std::size_t>(header.count);
                remoteParticles.push_back(RemoteParticlePayload{
                    message.source, header.spatialKey, particleCursor, count});
                for(std::size_t i = 0; i < count; ++i)
                {
                    FmmWireParticle& particle =
                        remoteParticleStorage[particleCursor + i];
                    particle = FmmPacketIO::readPod<FmmWireParticle>(view, offset);
                    if(particle.ownerRank != message.source ||
                       !finiteVector(particle.positionVector()) ||
                       !std::isfinite(particle.mass))
                        throw UniversalError(
                            "FmmLetPlan::execute: malformed remote particle");
                }
                particleCursor += count;
            }
        }
    }
    if(coefficientCursor != totalCoefficientCount ||
       particleCursor != totalParticleCount ||
       remoteMultipoles.size() != expectedMultipoleRecords ||
       remoteParticles.size() != expectedParticleRecords)
        throw UniversalError(
            "FmmLetPlan::execute: decoded LET payload size mismatch");

    std::sort(remoteMultipoles.begin(), remoteMultipoles.end(),
        payloadLess<RemoteMultipolePayload>);
    std::sort(remoteParticles.begin(), remoteParticles.end(),
        payloadLess<RemoteParticlePayload>);
    stats.peakRemoteBytes = std::max(stats.peakRemoteBytes,
        wireWorkspaceBytes + decodedBytes);
    received.releaseStorage();
    stats.letDecodeSeconds += elapsed(decodeStart);
    // Report only exchange work exposed on the critical path: preparation,
    // residual wait, and decode. Transfer time hidden by local traversal is
    // intentionally excluded from this phase timer.
    stats.letExchangeSeconds += pendingExchangePreparationSeconds_ +
        elapsed(finishStart);

    operatorCache.configure(maxOperatorCacheBytes, layout.m2lTerms().size(),
                            m2lInteractions_.size());
    operatorCache.beginPhase();
    std::vector<double> derivativeScratch;
    derivativeScratch.reserve(layout.coefficientCount());
    std::vector<double> uncachedOperator;
    uncachedOperator.reserve(layout.m2lTerms().size());
    const Clock::time_point m2lStart = Clock::now();
    if(remoteMultipoles.size() != m2lSources_.size() ||
       m2lSourceIndices_.size() != m2lInteractions_.size() ||
       m2lOperatorGeometryIndices_.size() != m2lInteractions_.size())
        throw UniversalError(
            "FmmLetPlan::execute: resolved LET M2L source table mismatch");
    for(std::size_t i = 0; i < m2lSources_.size(); ++i)
    {
        if(remoteMultipoles[i].sourceRank != m2lSources_[i].sourceRank ||
           remoteMultipoles[i].spatialKey != m2lSources_[i].spatialKey)
            throw UniversalError(
                "FmmLetPlan::execute: resolved LET M2L source order mismatch");
        FmmNode source = m2lSources_[i].node;
        source.multipoleOffset = remoteMultipoles[i].coefficientOffset;
        resolvedM2LSources.push_back(source);
    }
    std::vector<RemoteMultipolePayload>().swap(remoteMultipoles);

    std::vector<const std::vector<double>*> resolvedOperators;
    const bool operatorsResolved = operatorCache.resolvePreparedBatch(
        m2lOperatorGeometries_, m2lOperatorGeometryUseCounts_, layout,
        derivativeScratch, uncachedOperator, resolvedOperators);

    const std::vector<double>* groupedOperator = nullptr;
    std::uint64_t groupedKeyX = 0;
    std::uint64_t groupedKeyY = 0;
    std::uint64_t groupedKeyZ = 0;
    std::uint64_t groupedKeyKind = 0;
    bool haveGroupedOperator = false;
    for(std::size_t interactionIndex = 0;
        interactionIndex < m2lInteractions_.size(); ++interactionIndex)
    {
        const FmmLetM2LInteraction& interaction =
            m2lInteractions_[interactionIndex];
        const std::uint32_t sourceIndex = m2lSourceIndices_[interactionIndex];
        if(static_cast<std::size_t>(sourceIndex) >= resolvedM2LSources.size())
            throw UniversalError(
                "FmmLetPlan::execute: invalid resolved LET M2L source index");
        const std::uint32_t geometryIndex =
            m2lOperatorGeometryIndices_[interactionIndex];
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

        FmmKernels::translateM2L(source, target, layout,
                                 remoteCoefficients, localLocals,
                                 *coefficients, inverseScale);
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

    const Clock::time_point p2pStart = Clock::now();
    for(const FmmLetP2PInteraction& interaction : p2pInteractions_)
    {
        const RemoteParticlePayload* particlePayload = findPayload(
            remoteParticles, interaction.sourceRank, interaction.sourceKey);
        if(particlePayload == nullptr)
            throw UniversalError("FmmLetPlan::execute: missing remote P2P payload");
        const FmmNode& targetNode = localTree.nodes()[interaction.targetNode];
        if(!targetNode.isLeaf())
            throw UniversalError("FmmLetPlan::execute: P2P target is not a leaf");
        for(std::size_t ti = targetNode.particleBegin;
            ti < targetNode.particleEnd; ++ti)
        {
            const std::size_t target = localTree.particleOrder()[ti];
            const std::size_t particleEnd = particlePayload->particleOffset +
                particlePayload->particleCount;
            for(std::size_t sourceIndex = particlePayload->particleOffset;
                sourceIndex < particleEnd; ++sourceIndex)
            {
                const FmmWireParticle& source =
                    remoteParticleStorage[sourceIndex];
                if(source.ownerRank == rank_ &&
                   source.ownerLocalIndex == static_cast<std::uint64_t>(target))
                    continue;
                if(source.ownerRank == rank_)
                    throw UniversalError("FmmLetPlan::execute: remote payload carries local body token");
                const Vector3D delta = positions[target] - source.positionVector();
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
    beginExecute(localTree, positions, masses, cellIds, layout,
                 localMultipoles, localLocals, acceleration,
                 positiveKernelPotential, maxRemoteBytes, stats);
    finishExecute(localTree, positions, layout, localLocals, acceleration,
                  positiveKernelPotential, operatorCache, maxRemoteBytes,
                  maxOperatorCacheBytes, stats);
}

#endif // RICH_MPI
