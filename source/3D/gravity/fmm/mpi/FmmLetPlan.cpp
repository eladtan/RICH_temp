#include "3D/gravity/fmm/mpi/FmmLetPlan.hpp"

#ifdef RICH_MPI

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
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
                       FmmSolveStats& stats)
{
    const Clock::time_point start = Clock::now();
    comm_ = comm;
    topologyEpoch_ = topologyEpoch;
    MPI_Comm_rank(comm_, &rank_);
    localNodeByKey_.clear();
    remoteDescriptors_.clear();
    remoteLatticeRoots_.clear();
    remoteLatticeRoots_.resize(rootDescriptors.size());
    m2lInteractions_.clear();
    p2pInteractions_.clear();
    subscriptionsToSend_.clear();
    subscriptionsReceived_.clear();

    if(localTree.nodes().size() >
       static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
        throw UniversalError("FmmLetPlan::build: local tree exceeds compact index range");

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
    exchange_.reset(comm_, peers);

    std::vector<PendingPair> pending;
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
        std::vector<PendingPair> blocked;
        std::vector<PendingPair> work;
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

    // Descriptor pulling may visit many intermediate remote nodes.  Only
    // descriptors referenced by terminal M2L/P2P interactions are needed by
    // warm solves; discard the rest before the persistent plan is measured.
    std::set<std::pair<int, std::uint64_t>> retainedDescriptorKeys;
    for(const auto& terminal : terminalKeys)
        retainedDescriptorKeys.insert(std::make_pair(
            std::get<1>(terminal), std::get<2>(terminal)));
    std::unordered_map<int,
        std::unordered_map<std::uint64_t, FmmRemoteNodeDescriptor>>
        retainedDescriptors;
    for(const auto& key : retainedDescriptorKeys)
    {
        const auto rankIt = remoteDescriptors_.find(key.first);
        if(rankIt == remoteDescriptors_.end())
            throw UniversalError(
                "FmmLetPlan::build: terminal descriptor rank disappeared");
        const auto descriptorIt = rankIt->second.find(key.second);
        if(descriptorIt == rankIt->second.end())
            throw UniversalError(
                "FmmLetPlan::build: terminal descriptor disappeared");
        retainedDescriptors[key.first].emplace(key.second,
                                               descriptorIt->second);
    }
    remoteDescriptors_.swap(retainedDescriptors);
    std::unordered_map<int,
        std::unordered_map<std::uint64_t, FmmRemoteNodeDescriptor>>().swap(
            retainedDescriptors);

    // The LET plan persists across warm solves. Release construction slack and
    // keep the compact 16-byte interaction arrays at their final sizes.
    localNodeByKey_.rehash(0);
    remoteDescriptors_.rehash(0);
    for(auto& entry : remoteDescriptors_)
        entry.second.rehash(0);
    m2lInteractions_.shrink_to_fit();
    p2pInteractions_.shrink_to_fit();
    for(auto* map : {&subscriptionsToSend_, &subscriptionsReceived_})
    {
        map->rehash(0);
        for(auto& entry : *map)
            entry.second.shrink_to_fit();
    }

    stats.letPlanSeconds += elapsed(start);
}

std::size_t FmmLetPlan::bytesOwned() const
{
    std::size_t result = exchange_.bytesOwned();
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
        p2pInteractions_.capacity(), sizeof(FmmLetP2PInteraction)));

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
                         FmmSolveStats& stats) const
{
    const Clock::time_point start = Clock::now();
    if(comm_ == MPI_COMM_NULL || topologyEpoch_ == 0)
        throw UniversalError("FmmLetPlan::execute: LET plan is not initialized");
    if(positions.size() != masses.size() || positions.size() != cellIds.size() ||
       acceleration.size() != positions.size() ||
       localTree.particleOrder().size() != positions.size() ||
       (positiveKernelPotential != nullptr &&
        positiveKernelPotential->size() != positions.size()))
        throw UniversalError("FmmLetPlan::execute: inconsistent particle or output storage");
    if(layout.coefficientCount() == 0 ||
       localTree.nodes().size() > std::numeric_limits<std::size_t>::max() /
                                  layout.coefficientCount())
        throw UniversalError("FmmLetPlan::execute: expansion storage overflow");
    const std::size_t expectedExpansionSize =
        localTree.nodes().size() * layout.coefficientCount();
    if(localMultipoles.size() != expectedExpansionSize ||
       localLocals.size() != expectedExpansionSize)
        throw UniversalError("FmmLetPlan::execute: inconsistent expansion storage");
    if(maxRemoteBytes < 2)
        throw UniversalError("FmmLetPlan::execute: remote memory budget is too small");
    operatorCache.configure(maxOperatorCacheBytes, layout.m2lTerms().size(),
                            m2lInteractions_.size());
    operatorCache.beginPhase();
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
    FmmPeerExchangeResult received = exchange_.exchangeBytes(
        sendBuffers, &stats.bytesSent, &stats.bytesReceived, receiveLimit);
    stats.peakRemoteBytes = std::max(stats.peakRemoteBytes,
        sendCapacityBytes + outgoingBytes + received.totalBytes());
    std::unordered_map<int, std::vector<char>>().swap(sendBuffers);

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
       keyTableBytes > maxRemoteBytes - received.totalBytes())
        abortLetInvariant(comm_,
            "FmmLetPlan::execute: payload key tables exceed memory budget");
    stats.peakRemoteBytes = std::max(stats.peakRemoteBytes,
                                    received.totalBytes() + keyTableBytes);
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
    if(decodedRequestedBytes == std::numeric_limits<std::size_t>::max() ||
       decodedRequestedBytes > maxRemoteBytes - received.totalBytes())
        abortLetInvariant(comm_,
            "FmmLetPlan::execute: decoded LET payload exceeds memory budget");

    std::vector<RemoteMultipolePayload> remoteMultipoles;
    std::vector<RemoteParticlePayload> remoteParticles;
    std::vector<double> remoteCoefficients(totalCoefficientCount);
    std::vector<FmmWireParticle> remoteParticleStorage(totalParticleCount);
    remoteMultipoles.reserve(expectedMultipoleRecords);
    remoteParticles.reserve(expectedParticleRecords);

    std::size_t decodedBytes = 0;
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        remoteMultipoles.capacity(), sizeof(RemoteMultipolePayload)));
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        remoteParticles.capacity(), sizeof(RemoteParticlePayload)));
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        remoteCoefficients.capacity(), sizeof(double)));
    decodedBytes = saturatingAdd(decodedBytes, saturatingMultiply(
        remoteParticleStorage.capacity(), sizeof(FmmWireParticle)));
    if(decodedBytes == std::numeric_limits<std::size_t>::max() ||
       decodedBytes > maxRemoteBytes - received.totalBytes())
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
                                    received.totalBytes() + decodedBytes);
    received.releaseStorage();
    stats.letExchangeSeconds += elapsed(start);

    std::vector<double> derivativeScratch;
    derivativeScratch.reserve(layout.coefficientCount());
    std::vector<double> uncachedOperator;
    uncachedOperator.reserve(layout.m2lTerms().size());
    const Clock::time_point m2lStart = Clock::now();
    for(const FmmLetM2LInteraction& interaction : m2lInteractions_)
    {
        const auto descriptorRank = remoteDescriptors_.find(interaction.sourceRank);
        const RemoteMultipolePayload* coefficientPayload = findPayload(
            remoteMultipoles, interaction.sourceRank, interaction.sourceKey);
        if(descriptorRank == remoteDescriptors_.end() ||
           coefficientPayload == nullptr)
            throw UniversalError("FmmLetPlan::execute: missing remote M2L payload");
        const auto descriptorIt = descriptorRank->second.find(interaction.sourceKey);
        if(descriptorIt == descriptorRank->second.end())
            throw UniversalError("FmmLetPlan::execute: missing remote M2L descriptor");

        FmmNode source;
        source.center = descriptorIt->second.centerVector();
        source.halfSize = descriptorIt->second.halfSize;
        source.radius = descriptorIt->second.geometricRadius();
        source.multipoleOffset = coefficientPayload->coefficientOffset;
        if(interaction.sourceRank < 0 ||
           static_cast<std::size_t>(interaction.sourceRank) >=
               remoteLatticeRoots_.size())
            throw UniversalError(
                "FmmLetPlan::execute: missing remote lattice root");
        const RemoteLatticeRoot& latticeRoot =
            remoteLatticeRoots_[static_cast<std::size_t>(interaction.sourceRank)];
        applyRemoteLatticeMetadata(latticeRoot.latticeId, latticeRoot.center,
                                   latticeRoot.halfUnits, interaction.sourceKey,
                                   source);
        const FmmNode& target = localTree.nodes()[interaction.targetNode];
        const FmmM2LOperatorCache::Lookup translationOperator =
            operatorCache.get(source, target, layout, derivativeScratch,
                              uncachedOperator);

        FmmKernels::translateM2L(source, target, layout,
                                 remoteCoefficients, localLocals,
                                 *translationOperator.coefficients,
                                 translationOperator.inverseScale);
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

#endif // RICH_MPI
