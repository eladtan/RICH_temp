#include "3D/gravity/fmm/mpi/FmmProcessTraversal.hpp"

#ifdef RICH_MPI

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>
#include <utility>

#include "3D/gravity/fmm/mpi/FmmPackets.hpp"
#include "3D/gravity/fmm/mpi/FmmPeerExchange.hpp"
#include "misc/universal_error.hpp"

namespace
{
bool overlap(const FmmProcessNode& first, const FmmProcessNode& second)
{
    return std::abs(first.center.x - second.center.x) <= first.halfSize + second.halfSize &&
           std::abs(first.center.y - second.center.y) <= first.halfSize + second.halfSize &&
           std::abs(first.center.z - second.center.z) <= first.halfSize + second.halfSize;
}

double separation(const FmmProcessNode& first, const FmmProcessNode& second)
{
    const Vector3D delta = first.center - second.center;
    return std::sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
}

bool admissible(const FmmProcessNode& target,
                const FmmProcessNode& source,
                bool identical,
                double theta)
{
    if(identical || overlap(target, source))
        return false;
    const double distance = separation(target, source);
    return distance > 0.0 && target.radius + source.radius <= theta * distance;
}
}

std::size_t FmmProcessPairPlan::bytesOwned() const
{
    const std::size_t mapEntryBytes =
        sizeof(std::pair<const int, std::vector<std::size_t>>) +
        2 * sizeof(void*);
    std::size_t result = localM2LPairs.capacity() * sizeof(FmmProcessM2LPair) +
        localSelfPatches.capacity() * sizeof(FmmPatchKey) +
        localCrossPatchPairs.capacity() * sizeof(FmmPatchPair) +
        remoteLetPairs.capacity() * sizeof(FmmPatchPair) +
        letSourceRanks.capacity() * sizeof(int) +
        letTargetRanks.capacity() * sizeof(int) +
        processSendNodesByRank.size() * mapEntryBytes;
    for(const auto& entry : processSendNodesByRank)
        result += entry.second.capacity() * sizeof(std::size_t);
    return result;
}

FmmProcessPairPlan FmmProcessTraversal::build(const FmmProcessTree& tree,
                                              double thetaCritical,
                                              std::uint64_t topologyEpoch,
                                              int rank,
                                              const MPI_Comm& comm)
{
    if(!(thetaCritical > 0.0) || thetaCritical > 1.0 ||
       !std::isfinite(thetaCritical))
        throw UniversalError("FmmProcessTraversal::build: invalid theta");

    FmmProcessPairPlan plan;
    if(tree.nodes().empty())
        return plan;

    std::set<int> routePeerSet;
    for(const FmmProcessNode& node : tree.nodes())
    {
        if(node.owner != rank || node.isLeaf())
            continue;
        const int childOwners[2] = {
            tree.nodes()[node.left].owner, tree.nodes()[node.right].owner};
        for(int owner : childOwners)
            if(owner != rank)
                routePeerSet.insert(owner);
    }
    FmmPeerExchange routeExchange(
        comm, std::vector<int>(routePeerSet.begin(), routePeerSet.end()));

    std::vector<FmmProcessPairTask> queue;
    if(tree.nodes()[tree.root()].owner == rank)
    {
        FmmProcessPairTask task;
        task.stamp = fmmPacketStamp(FmmPacketKind::ProcessPairTask,
                                    topologyEpoch);
        task.target = static_cast<std::uint64_t>(tree.root());
        task.source = static_cast<std::uint64_t>(tree.root());
        queue.push_back(task);
    }
    std::set<int> letSources;
    std::unordered_map<int, std::set<std::pair<std::uint64_t, int>>> dependencies;

    while(true)
    {
        std::unordered_map<int, std::vector<char>> outgoing;
        unsigned long long localOutgoing = 0;
        while(!queue.empty())
        {
            const FmmProcessPairTask raw = queue.back();
            queue.pop_back();
            validateFmmPacketStamp(raw.stamp, FmmPacketKind::ProcessPairTask,
                                   topologyEpoch,
                                   "FmmProcessTraversal::build pair task");
            if(raw.target >= tree.nodes().size() || raw.source >= tree.nodes().size())
                throw UniversalError("FmmProcessTraversal::build: invalid routed task");
            const std::size_t targetIndex = static_cast<std::size_t>(raw.target);
            const std::size_t sourceIndex = static_cast<std::size_t>(raw.source);
            const FmmProcessNode& target = tree.nodes()[targetIndex];
            const FmmProcessNode& source = tree.nodes()[sourceIndex];
            if(target.owner != rank)
                throw UniversalError("FmmProcessTraversal::build: task delivered to wrong owner");

            if(admissible(target, source, targetIndex == sourceIndex,
                          thetaCritical))
            {
                ++plan.acceptedPairCount;
                plan.localM2LPairs.push_back(
                    FmmProcessM2LPair{targetIndex, sourceIndex});
                if(source.owner != rank)
                    dependencies[source.owner].insert(
                        std::make_pair(static_cast<std::uint64_t>(sourceIndex), 1));
                continue;
            }

            if(target.isLeaf() && source.isLeaf())
            {
                const FmmPatchKey targetPatch = target.leafKey();
                const FmmPatchKey sourcePatch = source.leafKey();
                if(!targetPatch.valid() || !sourcePatch.valid())
                    throw UniversalError(
                        "FmmProcessTraversal::build: invalid patch leaf identity");
                if(targetPatch == sourcePatch)
                {
                    ++plan.localSelfRankCount;
                    plan.localSelfPatches.push_back(targetPatch);
                }
                else if(targetPatch.ownerRank == sourcePatch.ownerRank)
                {
                    ++plan.localCrossPatchPairCount;
                    plan.localCrossPatchPairs.push_back(
                        FmmPatchPair{targetPatch, sourcePatch});
                }
                else
                {
                    ++plan.letRankPairCount;
                    plan.remoteLetPairs.push_back(
                        FmmPatchPair{targetPatch, sourcePatch});
                    letSources.insert(sourcePatch.ownerRank);
                    dependencies[sourcePatch.ownerRank].insert(
                        std::make_pair(0u, 2));
                }
                continue;
            }

            const bool splitTarget = !target.isLeaf() &&
                (source.isLeaf() || target.radius >= source.radius);
            if(splitTarget)
            {
                const std::size_t children[2] = {target.left, target.right};
                for(std::size_t child : children)
                {
                    FmmProcessPairTask task;
                    task.stamp = fmmPacketStamp(FmmPacketKind::ProcessPairTask,
                                                topologyEpoch);
                    task.target = static_cast<std::uint64_t>(child);
                    task.source = raw.source;
                    const int owner = tree.nodes()[child].owner;
                    if(owner == rank)
                        queue.push_back(task);
                    else
                    {
                        FmmPacketIO::appendPod(outgoing[owner], task);
                        ++localOutgoing;
                    }
                }
            }
            else
            {
                FmmProcessPairTask right;
                right.stamp = fmmPacketStamp(FmmPacketKind::ProcessPairTask,
                                             topologyEpoch);
                right.target = raw.target;
                right.source = static_cast<std::uint64_t>(source.right);
                FmmProcessPairTask left = right;
                left.source = static_cast<std::uint64_t>(source.left);
                queue.push_back(right);
                queue.push_back(left);
            }
        }

        unsigned long long globalOutgoing = 0;
        MPI_Allreduce(&localOutgoing, &globalOutgoing, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
        if(globalOutgoing == 0)
            break;
        FmmPeerExchangeResult received = routeExchange.exchangeBytes(outgoing);
        for(const FmmReceivedMessage& message : received.messages())
        {
            const FmmByteView view = received.view(message);
            std::size_t offset = 0;
            while(offset < view.size)
            {
                FmmProcessPairTask task =
                    FmmPacketIO::readPod<FmmProcessPairTask>(view, offset);
                validateFmmPacketStamp(task.stamp,
                    FmmPacketKind::ProcessPairTask, topologyEpoch,
                    "FmmProcessTraversal::build routed pair task");
                queue.push_back(task);
            }
        }
        received.releaseStorage();
    }

    std::vector<int> dependencyPeers;
    std::unordered_map<int, std::vector<char>> dependencyBuffers;
    for(const auto& entry : dependencies)
    {
        dependencyPeers.push_back(entry.first);
        for(const auto& dependency : entry.second)
        {
            FmmProcessDependency packet;
            packet.stamp = fmmPacketStamp(FmmPacketKind::ProcessDependency,
                                          topologyEpoch);
            packet.sourceNode = dependency.first;
            packet.kind = dependency.second;
            FmmPacketIO::appendPod(dependencyBuffers[entry.first], packet);
        }
    }
    FmmPeerExchange dependencyExchange(comm, dependencyPeers);
    FmmPeerExchangeResult receivedDependencies =
        dependencyExchange.exchangeBytes(dependencyBuffers);
    std::set<int> letTargets;
    std::unordered_map<int, std::set<std::size_t>> sendNodeSets;
    for(const FmmReceivedMessage& message : receivedDependencies.messages())
    {
        const FmmByteView view = receivedDependencies.view(message);
        std::size_t offset = 0;
        while(offset < view.size)
        {
            const FmmProcessDependency dependency =
                FmmPacketIO::readPod<FmmProcessDependency>(view, offset);
            validateFmmPacketStamp(dependency.stamp,
                FmmPacketKind::ProcessDependency, topologyEpoch,
                "FmmProcessTraversal::build dependency");
            if(dependency.kind == 1)
            {
                if(dependency.sourceNode >= tree.nodes().size())
                    throw UniversalError("FmmProcessTraversal::build: invalid source subscription");
                if(tree.nodes()[static_cast<std::size_t>(dependency.sourceNode)].owner != rank)
                    throw UniversalError("FmmProcessTraversal::build: source subscription sent to wrong owner");
                sendNodeSets[message.source].insert(
                    static_cast<std::size_t>(dependency.sourceNode));
            }
            else if(dependency.kind == 2)
            {
                letTargets.insert(message.source);
            }
            else
            {
                throw UniversalError("FmmProcessTraversal::build: invalid dependency kind");
            }
        }
    }
    receivedDependencies.releaseStorage();

    for(const auto& entry : sendNodeSets)
        plan.processSendNodesByRank[entry.first] =
            std::vector<std::size_t>(entry.second.begin(), entry.second.end());
    plan.letSourceRanks.assign(letSources.begin(), letSources.end());
    plan.letTargetRanks.assign(letTargets.begin(), letTargets.end());
    std::sort(plan.localM2LPairs.begin(), plan.localM2LPairs.end(),
        [](const FmmProcessM2LPair& first, const FmmProcessM2LPair& second)
        {
            return std::tie(first.targetNode, first.sourceNode) <
                   std::tie(second.targetNode, second.sourceNode);
        });
    const auto duplicate = std::adjacent_find(
        plan.localM2LPairs.begin(), plan.localM2LPairs.end(),
        [](const FmmProcessM2LPair& first, const FmmProcessM2LPair& second)
        {
            return first.targetNode == second.targetNode &&
                   first.sourceNode == second.sourceNode;
        });
    if(duplicate != plan.localM2LPairs.end())
        throw UniversalError("FmmProcessTraversal::build: duplicate accepted process pair");

    std::sort(plan.localSelfPatches.begin(), plan.localSelfPatches.end());
    if(std::adjacent_find(plan.localSelfPatches.begin(),
                          plan.localSelfPatches.end()) !=
       plan.localSelfPatches.end())
        throw UniversalError(
            "FmmProcessTraversal::build: duplicate local self patch");
    std::sort(plan.localCrossPatchPairs.begin(),
              plan.localCrossPatchPairs.end());
    if(std::adjacent_find(plan.localCrossPatchPairs.begin(),
                          plan.localCrossPatchPairs.end()) !=
       plan.localCrossPatchPairs.end())
        throw UniversalError(
            "FmmProcessTraversal::build: duplicate local cross-patch pair");
    std::sort(plan.remoteLetPairs.begin(), plan.remoteLetPairs.end());
    if(std::adjacent_find(plan.remoteLetPairs.begin(),
                          plan.remoteLetPairs.end()) !=
       plan.remoteLetPairs.end())
        throw UniversalError(
            "FmmProcessTraversal::build: duplicate remote LET patch pair");
    return plan;
}

#endif // RICH_MPI
