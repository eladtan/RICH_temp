#include "3D/gravity/fmm/mpi/FmmPatchDistributedSolver.hpp"

#ifdef RICH_MPI

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "3D/gravity/fmm/FmmDualTreeTraversal.hpp"
#include "3D/gravity/fmm/FmmKernels.hpp"
#include "3D/gravity/fmm/FmmPasses.hpp"
#include "3D/gravity/fmm/FmmTaylorExpansion.hpp"
#include "3D/gravity/fmm/mpi/FmmDescriptorGather.hpp"
#include "misc/universal_error.hpp"

namespace
{
typedef std::chrono::steady_clock Clock;

double elapsed(const Clock::time_point& start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
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

std::size_t maximumStableLeafOccupancy(
    const FmmGravityOptions& options,
    const FmmDistributedOptions& distributedOptions)
{
    if(!distributedOptions.persistentLocalTreeTopology)
        return options.leafCapacity;
    const long double scaled = std::ceil(
        static_cast<long double>(options.leafCapacity) *
        static_cast<long double>(
            distributedOptions.persistentLeafSplitFactor));
    if(!std::isfinite(static_cast<double>(scaled)) ||
       scaled > static_cast<long double>(
           std::numeric_limits<std::size_t>::max()))
        throw UniversalError(
            "FmmPatchDistributedSolver: stable leaf occupancy overflow");
    std::size_t result = static_cast<std::size_t>(scaled);
    if(result <= options.leafCapacity)
    {
        if(options.leafCapacity ==
           std::numeric_limits<std::size_t>::max())
            throw UniversalError(
                "FmmPatchDistributedSolver: stable leaf occupancy overflow");
        result = options.leafCapacity + 1;
    }
    return result;
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

struct ProcessCoefficientStore
{
    explicit ProcessCoefficientStore(std::size_t count): coefficientCount(count) {}

    std::size_t ensure(std::size_t node)
    {
        const auto found = slotByNode.find(node);
        if(found != slotByNode.end())
            return found->second * coefficientCount;
        const std::size_t slot = slotByNode.size();
        if(coefficientCount == 0 ||
           slot == std::numeric_limits<std::size_t>::max() ||
           slot + 1 > std::numeric_limits<std::size_t>::max() / coefficientCount)
            throw UniversalError(
                "FmmPatchDistributedSolver: process coefficient storage overflow");
        slotByNode[node] = slot;
        values.resize((slot + 1) * coefficientCount, 0.0);
        return slot * coefficientCount;
    }

    std::size_t offset(std::size_t node) const
    {
        const auto found = slotByNode.find(node);
        if(found == slotByNode.end())
            throw UniversalError(
                "FmmPatchDistributedSolver: missing process coefficient node");
        return found->second * coefficientCount;
    }

    void zero(std::size_t node)
    {
        const std::size_t begin = ensure(node);
        std::fill(values.begin() + static_cast<std::ptrdiff_t>(begin),
                  values.begin() + static_cast<std::ptrdiff_t>(
                      begin + coefficientCount), 0.0);
    }

    void add(std::size_t node, const double* coefficients)
    {
        const std::size_t begin = ensure(node);
        for(std::size_t i = 0; i < coefficientCount; ++i)
            values[begin + i] += coefficients[i];
    }

    std::size_t bytesOwned() const
    {
        std::size_t result = saturatingMultiply(
            values.capacity(), sizeof(double));
        result = saturatingAdd(result, saturatingMultiply(
            slotByNode.bucket_count(), sizeof(void*)));
        result = saturatingAdd(result, saturatingMultiply(
            slotByNode.size(),
            sizeof(std::pair<const std::size_t, std::size_t>) +
                2 * sizeof(void*)));
        return result;
    }

    void release()
    {
        std::vector<double>().swap(values);
        std::unordered_map<std::size_t, std::size_t>().swap(slotByNode);
    }

    std::size_t coefficientCount;
    std::unordered_map<std::size_t, std::size_t> slotByNode;
    std::vector<double> values;
};

FmmNode processView(const FmmProcessNode& processNode, std::size_t offset)
{
    FmmNode node;
    node.center = processNode.center;
    node.halfSize = processNode.halfSize;
    node.radius = processNode.radius;
    node.multipoleOffset = offset;
    node.localOffset = offset;
    return node;
}

void appendProcessCoefficients(std::vector<char>& buffer,
                               std::size_t node,
                               const ProcessCoefficientStore& store,
                               std::uint64_t topologyEpoch)
{
    FmmProcessCoefficientHeader header;
    header.stamp = fmmPacketStamp(FmmPacketKind::ProcessCoefficient,
                                  topologyEpoch);
    header.nodeIndex = static_cast<std::uint64_t>(node);
    FmmPacketIO::appendPod(buffer, header);
    FmmPacketIO::appendDoubles(buffer,
        store.values.data() + store.offset(node), store.coefficientCount);
}

void parseProcessCoefficients(const FmmPeerExchangeResult& received,
                              ProcessCoefficientStore& store,
                              const FmmProcessTree& tree,
                              std::uint64_t topologyEpoch)
{
    std::unordered_set<std::size_t> receivedNodes;
    for(const FmmReceivedMessage& message : received.messages())
    {
        const FmmByteView view = received.view(message);
        std::size_t offset = 0;
        while(offset < view.size)
        {
            const FmmProcessCoefficientHeader header =
                FmmPacketIO::readPod<FmmProcessCoefficientHeader>(view, offset);
            validateFmmPacketStamp(
                header.stamp, FmmPacketKind::ProcessCoefficient,
                topologyEpoch,
                "FmmPatchDistributedSolver process coefficient");
            if(header.nodeIndex >= tree.nodes().size())
                throw UniversalError(
                    "FmmPatchDistributedSolver: process node out of range");
            const std::size_t node =
                static_cast<std::size_t>(header.nodeIndex);
            if(tree.nodes()[node].owner != message.source)
                throw UniversalError(
                    "FmmPatchDistributedSolver: coefficient sent by non-owner");
            if(!receivedNodes.insert(node).second)
                throw UniversalError(
                    "FmmPatchDistributedSolver: duplicate process coefficient");
            const std::size_t begin = store.ensure(node);
            FmmPacketIO::readDoubles(
                view, offset, store.values.data() + begin,
                store.coefficientCount);
        }
    }
}

void parseAndAddProcessCoefficients(
    const FmmPeerExchangeResult& received,
    ProcessCoefficientStore& store,
    const FmmProcessTree& tree,
    std::uint64_t topologyEpoch,
    int rank)
{
    std::unordered_set<std::size_t> receivedNodes;
    std::vector<double> coefficients(store.coefficientCount, 0.0);
    for(const FmmReceivedMessage& message : received.messages())
    {
        const FmmByteView view = received.view(message);
        std::size_t offset = 0;
        while(offset < view.size)
        {
            const FmmProcessCoefficientHeader header =
                FmmPacketIO::readPod<FmmProcessCoefficientHeader>(view, offset);
            validateFmmPacketStamp(
                header.stamp, FmmPacketKind::ProcessCoefficient,
                topologyEpoch,
                "FmmPatchDistributedSolver downward coefficient");
            if(header.nodeIndex >= tree.nodes().size())
                throw UniversalError(
                    "FmmPatchDistributedSolver: downward node out of range");
            const std::size_t node =
                static_cast<std::size_t>(header.nodeIndex);
            const FmmProcessNode& target = tree.nodes()[node];
            if(target.owner != rank ||
               target.parent == FmmProcessTree::invalidIndex() ||
               tree.nodes()[target.parent].owner != message.source)
                throw UniversalError(
                    "FmmPatchDistributedSolver: invalid downward sender");
            if(!receivedNodes.insert(node).second)
                throw UniversalError(
                    "FmmPatchDistributedSolver: duplicate downward coefficient");
            FmmPacketIO::readDoubles(view, offset, coefficients.data(),
                                     coefficients.size());
            store.add(node, coefficients.data());
        }
    }
}

void executeDirectPatchPair(FmmLocalPatch& target,
                            const FmmLocalPatch& source,
                            FmmSolveStats& stats)
{
    std::vector<double>* potential =
        target.potential.empty() ? nullptr : &target.potential;
    FmmKernels::accumulateP2P(
        target.positions, source.positions, source.masses,
        target.tree.particleOrder(), source.tree.particleOrder(),
        0, target.tree.particleOrder().size(),
        0, source.tree.particleOrder().size(), false,
        target.acceleration, potential, stats.p2pPairCount);
    ++stats.p2pBlockCount;
}

bool finiteVector(const Vector3D& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

struct PatchForestMemory
{
    std::size_t total = 0;
    std::size_t treeAndInputs = 0;
    std::size_t multipoles = 0;
    std::size_t locals = 0;
};

PatchForestMemory patchForestMemory(const FmmPatchForest& forest)
{
    PatchForestMemory memory;
    const std::size_t patchObjects = saturatingMultiply(
        forest.patches().capacity(), sizeof(FmmLocalPatch));
    memory.total = patchObjects;
    memory.treeAndInputs = patchObjects;
    for(const FmmLocalPatch& patch : forest.patches())
    {
        std::size_t treeBytes = patch.tree.bytesOwned();
        treeBytes = saturatingAdd(treeBytes, saturatingMultiply(
            patch.inputIndices.capacity(), sizeof(std::size_t)));
        treeBytes = saturatingAdd(treeBytes, saturatingMultiply(
            patch.positions.capacity(), sizeof(Vector3D)));
        treeBytes = saturatingAdd(treeBytes, saturatingMultiply(
            patch.masses.capacity(), sizeof(double)));
        treeBytes = saturatingAdd(treeBytes, saturatingMultiply(
            patch.cellIds.capacity(), sizeof(std::uint64_t)));
        treeBytes = saturatingAdd(treeBytes, saturatingMultiply(
            patch.structuralSignature.capacity(), sizeof(std::uint64_t)));
        treeBytes = saturatingAdd(treeBytes, saturatingMultiply(
            patch.occupancySignature.capacity(), sizeof(std::uint64_t)));
        memory.treeAndInputs = saturatingAdd(
            memory.treeAndInputs, treeBytes);
        memory.total = saturatingAdd(memory.total, treeBytes);

        memory.total = saturatingAdd(
            memory.total, patch.localPlan.bytesOwned());
        const std::size_t multipoleBytes = saturatingMultiply(
            patch.multipoles.capacity(), sizeof(double));
        const std::size_t localBytes = saturatingMultiply(
            patch.locals.capacity(), sizeof(double));
        memory.multipoles = saturatingAdd(
            memory.multipoles, multipoleBytes);
        memory.locals = saturatingAdd(memory.locals, localBytes);
        memory.total = saturatingAdd(memory.total, multipoleBytes);
        memory.total = saturatingAdd(memory.total, localBytes);
        memory.total = saturatingAdd(memory.total, saturatingMultiply(
            patch.acceleration.capacity(), sizeof(Vector3D)));
        memory.total = saturatingAdd(memory.total, saturatingMultiply(
            patch.potential.capacity(), sizeof(double)));
    }
    return memory;
}
}

FmmPatchDistributedSolver::FmmPatchDistributedSolver(
    const FmmGravityOptions& options,
    const FmmDistributedOptions& distributedOptions,
    const MPI_Comm& comm):
    options_(options), distributedOptions_(distributedOptions), comm_(comm),
    rank_(0), size_(1), topologyEpoch_(0), topologyRebuildCount_(0),
    processTopologyRebuildCount_(0), letTopologyRebuildCount_(0),
    topologyInitialized_(false)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
}

void FmmPatchDistributedSolver::solve(
    const std::vector<Vector3D>& positions,
    const std::vector<double>& masses,
    const std::vector<std::uint64_t>& cellIds,
    const Vector3D& domainLower,
    const Vector3D& domainUpper,
    std::vector<Vector3D>& acceleration,
    std::vector<double>* positiveKernelPotential,
    FmmSolveStats& stats)
{
    const Clock::time_point totalStart = Clock::now();
    stats = FmmSolveStats();
    stats.particleCount = positions.size();
    stats.mpiRankCount = static_cast<std::size_t>(size_);
    stats.operatorCacheBudgetBytes = options_.maxOperatorCacheBytes;
    stats.operatorCacheBytesAtSolveStart = operatorCache_.bytesOwned();
    stats.operatorCacheEntriesAtSolveStart = operatorCache_.entries();

    const Clock::time_point buildStart = Clock::now();
    FmmPatchForestChange forestChange;
    bool localPrepareOk = true;
    std::string localPrepareError;
    try
    {
        forestChange = forest_.preparePersistent(
            positions, masses, cellIds, domainLower, domainUpper,
            options_, distributedOptions_, rank_);
    }
    catch(const UniversalError& error)
    {
        localPrepareOk = false;
        localPrepareError = error.getErrorMessage();
    }
    collectiveRequire(localPrepareOk, localPrepareError,
                      "FmmPatchDistributedSolver::solve patch preparation",
                      comm_);
    stats.buildSeconds = elapsed(buildStart);
    const FmmTaylorExpansion layout(options_.expansionOrder);
    const Clock::time_point upwardStart = Clock::now();
    forest_.buildUpward(layout);
    stats.upwardSeconds = elapsed(upwardStart);
    forest_.clearLocals(layout);
    stats.localRootGeometryChanged =
        forestChange.patchSetChanged || forestChange.patchGeometryChanged;
    stats.localLeafTopologyChanged = forestChange.structuralTopologyChanged;
    stats.localLeafOccupancyChanged = forestChange.occupancyChanged;
    stats.localCountOnlyLeafChange = forestChange.countOnlyChanged;
    stats.localPatchCount = forest_.patches().size();
    stats.reusedPatchCount = forestChange.reusedPatches;
    stats.reusedLocalPatchPlanCount = forestChange.reusedLocalPlans;
    stats.rebuiltLocalPatchPlanCount = forestChange.rebuiltLocalPlans;
    stats.patchNodeGeometryExpansionCount =
        forestChange.nodeGeometryExpansionPatches;
    stats.patchRetainedBytes = forestChange.retainedBytes;
    stats.patchReleasedBytes = forestChange.releasedBytes;
    stats.persistentLeafSplitCount = forestChange.persistentLeafSplits;
    stats.persistentSubtreeMergeCount = forestChange.persistentSubtreeMerges;
    stats.persistentEmptyLeafCount = forestChange.persistentEmptyLeaves;
    stats.localInteractionPlanReused = !forest_.patches().empty() &&
        forestChange.reusedLocalPlans == forest_.patches().size();

    bool localPersistentRefit = false;
    for(const FmmLocalPatch& patch : forest_.patches())
        localPersistentRefit = localPersistentRefit || patch.persistentTreeRefit;

    const unsigned long long localTopologyTerms[9] = {
        stats.localRootGeometryChanged ? 1ull : 0ull,
        stats.localLeafTopologyChanged ? 1ull : 0ull,
        stats.localLeafOccupancyChanged ? 1ull : 0ull,
        stats.localCountOnlyLeafChange ? 1ull : 0ull,
        localPersistentRefit ? 1ull : 0ull,
        static_cast<unsigned long long>(forestChange.persistentLeafSplits),
        static_cast<unsigned long long>(forestChange.persistentSubtreeMerges),
        static_cast<unsigned long long>(forestChange.persistentEmptyLeaves),
        static_cast<unsigned long long>(forest_.patches().size())};
    unsigned long long globalTopologyTerms[9] = {};
    MPI_Allreduce(localTopologyTerms, globalTopologyTerms, 9,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_);
    stats.ranksWithRootGeometryChange =
        static_cast<std::size_t>(globalTopologyTerms[0]);
    stats.ranksWithLeafTopologyChange =
        static_cast<std::size_t>(globalTopologyTerms[1]);
    stats.ranksWithLeafOccupancyChange =
        static_cast<std::size_t>(globalTopologyTerms[2]);
    stats.ranksWithCountOnlyLeafChange =
        static_cast<std::size_t>(globalTopologyTerms[3]);
    stats.persistentTreeRefitRankCount =
        static_cast<std::size_t>(globalTopologyTerms[4]);
    stats.persistentLeafSplitCount = globalTopologyTerms[5];
    stats.persistentSubtreeMergeCount = globalTopologyTerms[6];
    stats.persistentEmptyLeafCount = globalTopologyTerms[7];
    if(globalTopologyTerms[8] > static_cast<unsigned long long>(
           std::numeric_limits<std::size_t>::max()))
        throw UniversalError(
            "FmmPatchDistributedSolver::solve: global patch count exceeds size_t");
    stats.globalPatchCount = static_cast<std::size_t>(globalTopologyTerms[8]);
    stats.replicatedDescriptorBytes = saturatingMultiply(
        stats.globalPatchCount, sizeof(FmmPatchRootDescriptor));

    long double localMassExtended = 0.0L;
    long double localAbsoluteMassExtended = 0.0L;
    for(double mass : masses)
    {
        localMassExtended += static_cast<long double>(mass);
        localAbsoluteMassExtended += std::abs(static_cast<long double>(mass));
    }
    const double localMassTerms[2] = {
        static_cast<double>(localMassExtended),
        static_cast<double>(localAbsoluteMassExtended)};
    double globalMassTerms[2] = {0.0, 0.0};
    MPI_Allreduce(localMassTerms, globalMassTerms, 2,
                  MPI_DOUBLE, MPI_SUM, comm_);
    stats.totalMass = globalMassTerms[0];
    if(!std::isfinite(stats.totalMass) ||
       !std::isfinite(globalMassTerms[1]))
        throw UniversalError(
            "FmmPatchDistributedSolver::solve: non-finite global mass sum");

    int localPayloadShapeReusable =
        topologyInitialized_ && letPlan_.localPayloadShapeReusable(forest_) ?
        1 : 0;
    int globalPayloadShapeReusable = 0;
    MPI_Allreduce(&localPayloadShapeReusable, &globalPayloadShapeReusable, 1,
                  MPI_INT, MPI_LAND, comm_);
    const bool payloadShapeRequiresRebuild = topologyInitialized_ &&
        globalPayloadShapeReusable == 0;

    const bool forcedRebuild = distributedOptions_.rebuildTopologyEverySolve;
    const bool rebuildProcessTopology = !topologyInitialized_ ||
        forcedRebuild || globalTopologyTerms[0] != 0;
    const bool rebuildLetTopology = rebuildProcessTopology ||
        globalTopologyTerms[1] != 0 || payloadShapeRequiresRebuild;
    stats.topologyRebuildForced = forcedRebuild;
    stats.letPayloadShapeTriggeredRebuild = payloadShapeRequiresRebuild;
    stats.processTopologyRebuilt = rebuildProcessTopology;
    stats.letTopologyRebuilt = rebuildLetTopology;
    stats.countOnlyTopologyReused = globalTopologyTerms[3] != 0 &&
        !rebuildLetTopology;

    const Clock::time_point topologyStart = Clock::now();
    if(rebuildLetTopology)
    {
        if(topologyEpoch_ == std::numeric_limits<std::uint64_t>::max() ||
           topologyRebuildCount_ == std::numeric_limits<std::uint64_t>::max() ||
           (rebuildProcessTopology && processTopologyRebuildCount_ ==
                std::numeric_limits<std::uint64_t>::max()) ||
           letTopologyRebuildCount_ ==
                std::numeric_limits<std::uint64_t>::max())
            throw UniversalError(
                "FmmPatchDistributedSolver::solve: topology epoch overflow");
        ++topologyEpoch_;
        ++topologyRebuildCount_;
        ++letTopologyRebuildCount_;
        if(rebuildProcessTopology)
            ++processTopologyRebuildCount_;

        const std::vector<FmmPatchRootDescriptor> localDescriptors =
            forest_.descriptors(rank_, topologyEpoch_);
        const Clock::time_point gatherStart = Clock::now();
        rootDescriptors_ = FmmDescriptorGather::gather(
            localDescriptors, static_cast<std::uint64_t>(positions.size()),
            topologyEpoch_, distributedOptions_.maxReplicatedDescriptorBytes,
            comm_);
        stats.rootDescriptorExchangeSeconds = elapsed(gatherStart);

        if(rebuildProcessTopology)
        {
            const Clock::time_point processTopologyStart = Clock::now();
            processTree_.build(rootDescriptors_, true);
            processPlan_ = FmmProcessTraversal::build(
                processTree_, options_.thetaCritical, topologyEpoch_, rank_,
                comm_);

            std::set<int> upPeers;
            std::set<int> downPeers;
            for(std::size_t nodeIndex = 0;
                nodeIndex < processTree_.nodes().size(); ++nodeIndex)
            {
                const FmmProcessNode& node = processTree_.nodes()[nodeIndex];
                if(node.owner == rank_ &&
                   node.parent != FmmProcessTree::invalidIndex())
                {
                    const int parentOwner =
                        processTree_.nodes()[node.parent].owner;
                    if(parentOwner != rank_)
                        upPeers.insert(parentOwner);
                }
                if(node.owner == rank_ && !node.isLeaf())
                {
                    const int leftOwner = processTree_.nodes()[node.left].owner;
                    const int rightOwner = processTree_.nodes()[node.right].owner;
                    if(leftOwner != rank_)
                        downPeers.insert(leftOwner);
                    if(rightOwner != rank_)
                        downPeers.insert(rightOwner);
                }
            }
            std::vector<int> m2lPeers;
            for(const auto& entry : processPlan_.processSendNodesByRank)
                m2lPeers.push_back(entry.first);
            const bool upReset = processUpExchange_.resetIfChanged(
                comm_, std::vector<int>(upPeers.begin(), upPeers.end()));
            const bool m2lReset = processM2LExchange_.resetIfChanged(
                comm_, m2lPeers);
            const bool downReset = processDownExchange_.resetIfChanged(
                comm_, std::vector<int>(downPeers.begin(), downPeers.end()));
            stats.processCommunicatorsReused =
                !upReset && !m2lReset && !downReset;
            stats.processTopologySeconds = elapsed(processTopologyStart);
        }
        else
        {
            stats.processCommunicatorsReused = true;
        }

        letPlan_.build(
            forest_, rootDescriptors_, processPlan_, options_.thetaCritical,
            topologyEpoch_, distributedOptions_.maxLetWaveBytes,
            distributedOptions_.maxTargetPatchesPerWave,
            layout.coefficientCount(),
            maximumStableLeafOccupancy(options_, distributedOptions_),
            distributedOptions_.letParticlePayloadSlackFactor,
            distributedOptions_.letParticlePayloadSlackCount,
            distributedOptions_.enableLeafM2P,
            comm_, stats,
            topologyInitialized_ && !rebuildProcessTopology &&
                !payloadShapeRequiresRebuild);
        std::vector<FmmPatchRootDescriptor>().swap(rootDescriptors_);
    }
    else
    {
        stats.processCommunicatorsReused = true;
        letPlan_.reuse(forest_, topologyEpoch_, stats);
    }
    topologyInitialized_ = true;
    stats.topologyRebuildSeconds = elapsed(topologyStart);

    std::vector<FmmPatchKey> expectedLocalSelf;
    expectedLocalSelf.reserve(forest_.patches().size());
    for(const FmmLocalPatch& patch : forest_.patches())
        expectedLocalSelf.push_back(patch.key);
    std::sort(expectedLocalSelf.begin(), expectedLocalSelf.end());
    if(processPlan_.localSelfPatches != expectedLocalSelf ||
       processPlan_.localSelfRankCount != expectedLocalSelf.size())
        throw UniversalError(
            "FmmPatchDistributedSolver::solve: local self patch coverage mismatch");

    stats.topologyEpoch = topologyEpoch_;
    stats.topologyRebuildCount = topologyRebuildCount_;
    stats.processTopologyRebuildCount = processTopologyRebuildCount_;
    stats.letTopologyRebuildCount = letTopologyRebuildCount_;
    stats.activeRankCount = processTree_.activeRanks().size();
    stats.processNodeCount = processTree_.nodes().size();
    stats.processTreeBytes = processTree_.bytesOwned();
    stats.processPlanBytes = processPlan_.bytesOwned();

    unsigned long long localOwnedNodes = 0;
    for(const FmmProcessNode& node : processTree_.nodes())
        localOwnedNodes += node.owner == rank_ ? 1ull : 0ull;
    const unsigned long long localOwnedM2L =
        static_cast<unsigned long long>(processPlan_.localM2LPairs.size());
    unsigned long long maxOwnedNodes = 0;
    unsigned long long maxOwnedM2L = 0;
    MPI_Allreduce(&localOwnedNodes, &maxOwnedNodes, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm_);
    MPI_Allreduce(&localOwnedM2L, &maxOwnedM2L, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm_);
    stats.processOwnedNodeCount = static_cast<std::size_t>(localOwnedNodes);
    stats.processOwnedNodeCountMax = static_cast<std::size_t>(maxOwnedNodes);
    stats.processOwnedM2LCount = localOwnedM2L;
    stats.processOwnedM2LCountMax = maxOwnedM2L;
    const double meanOwnedNodes = stats.activeRankCount == 0 ? 0.0 :
        static_cast<double>(stats.processNodeCount) /
        static_cast<double>(stats.activeRankCount);
    stats.processOwnedNodeImbalance = meanOwnedNodes == 0.0 ? 0.0 :
        static_cast<double>(maxOwnedNodes) / meanOwnedNodes;

    ProcessCoefficientStore processMultipoles(layout.coefficientCount());
    ProcessCoefficientStore processLocals(layout.coefficientCount());
    const Clock::time_point processUpStart = Clock::now();
    for(const FmmLocalPatch& patch : forest_.patches())
    {
        const std::size_t leaf = processTree_.leafForPatch(patch.key);
        if(leaf == FmmProcessTree::invalidIndex() || patch.tree.nodes().empty())
            throw UniversalError(
                "FmmPatchDistributedSolver::solve: missing process leaf for local patch");
        const std::size_t destination = processMultipoles.ensure(leaf);
        const std::size_t source = patch.tree.nodes()[0].multipoleOffset;
        std::copy(patch.multipoles.begin() +
                      static_cast<std::ptrdiff_t>(source),
                  patch.multipoles.begin() +
                      static_cast<std::ptrdiff_t>(source +
                                                  layout.coefficientCount()),
                  processMultipoles.values.begin() +
                      static_cast<std::ptrdiff_t>(destination));
    }

    for(std::size_t depth = processTree_.maxDepth(); depth > 0; --depth)
    {
        std::unordered_map<int, std::vector<char>> sendBuffers;
        for(std::size_t nodeIndex : processTree_.levels()[depth])
        {
            const FmmProcessNode& node = processTree_.nodes()[nodeIndex];
            if(node.owner != rank_)
                continue;
            const int parentOwner =
                processTree_.nodes()[node.parent].owner;
            if(parentOwner != rank_)
                appendProcessCoefficients(sendBuffers[parentOwner], nodeIndex,
                                          processMultipoles, topologyEpoch_);
        }
        FmmPeerExchangeResult received = processUpExchange_.exchangeBytes(
            sendBuffers, &stats.bytesSent, &stats.bytesReceived);
        parseProcessCoefficients(received, processMultipoles,
                                 processTree_, topologyEpoch_);
        received.releaseStorage();

        for(std::size_t parentIndex : processTree_.levels()[depth - 1])
        {
            const FmmProcessNode& parent =
                processTree_.nodes()[parentIndex];
            if(parent.owner != rank_ || parent.isLeaf())
                continue;
            processMultipoles.zero(parentIndex);
            const std::size_t parentOffset =
                processMultipoles.offset(parentIndex);
            FmmNode parentNode = processView(parent, parentOffset);
            const std::size_t children[2] = {parent.left, parent.right};
            for(std::size_t childIndex : children)
            {
                const std::size_t childOffset =
                    processMultipoles.offset(childIndex);
                FmmNode childNode = processView(
                    processTree_.nodes()[childIndex], childOffset);
                FmmKernels::translateM2M(childNode, parentNode, layout,
                                         processMultipoles.values);
            }
        }
        stats.peakProcessBytes = std::max(
            stats.peakProcessBytes,
            processMultipoles.bytesOwned() + processLocals.bytesOwned());
    }
    stats.processUpwardSeconds = elapsed(processUpStart);

    double localRootMass = 0.0;
    if(!processTree_.nodes().empty() &&
       processTree_.nodes()[processTree_.root()].owner == rank_)
    {
        localRootMass = processMultipoles.values[
            processMultipoles.offset(processTree_.root()) +
            layout.index(0, 0, 0)];
    }
    MPI_Allreduce(&localRootMass, &stats.rootMass, 1,
                  MPI_DOUBLE, MPI_SUM, comm_);
    const unsigned long long localParticleCount =
        static_cast<unsigned long long>(positions.size());
    unsigned long long globalParticleCount = 0;
    MPI_Allreduce(&localParticleCount, &globalParticleCount, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_);
    const long double massTolerance =
        256.0L * std::numeric_limits<double>::epsilon() *
        static_cast<long double>(std::max<unsigned long long>(
            1, globalParticleCount)) *
        std::max(1.0L, static_cast<long double>(globalMassTerms[1]));
    if(std::abs(stats.rootMass - stats.totalMass) >
       static_cast<double>(massTolerance))
        throw UniversalError(
            "FmmPatchDistributedSolver::solve: process root mass mismatch");

    const Clock::time_point processInteractionStart = Clock::now();
    std::unordered_map<int, std::vector<char>> processM2LSends;
    for(const auto& entry : processPlan_.processSendNodesByRank)
    {
        for(std::size_t nodeIndex : entry.second)
            appendProcessCoefficients(processM2LSends[entry.first], nodeIndex,
                                      processMultipoles, topologyEpoch_);
    }
    FmmPeerExchangeResult processM2LReceived =
        processM2LExchange_.exchangeBytes(
            processM2LSends, &stats.bytesSent, &stats.bytesReceived);
    parseProcessCoefficients(processM2LReceived, processMultipoles,
                             processTree_, topologyEpoch_);
    processM2LReceived.releaseStorage();

    std::vector<double> derivativeScratch;
    std::vector<double> processOperatorScratch;
    for(const FmmProcessM2LPair& pair : processPlan_.localM2LPairs)
    {
        const std::size_t sourceOffset =
            processMultipoles.offset(pair.sourceNode);
        const std::size_t targetOffset =
            processLocals.ensure(pair.targetNode);
        FmmNode source = processView(
            processTree_.nodes()[pair.sourceNode], sourceOffset);
        FmmNode target = processView(
            processTree_.nodes()[pair.targetNode], targetOffset);
        FmmKernels::computeM2LOperator(target.center - source.center,
                                       layout, derivativeScratch,
                                       processOperatorScratch);
        FmmKernels::translateM2L(
            source, target, layout, processMultipoles.values,
            processLocals.values, processOperatorScratch);
        ++stats.m2lCount;
        ++stats.processM2LCount;
        ++stats.processOperatorCacheMisses;
        ++stats.processOperatorCacheBypasses;
    }
    std::size_t processInteractionBytes = saturatingAdd(
        processMultipoles.bytesOwned(), processLocals.bytesOwned());
    processInteractionBytes = saturatingAdd(
        processInteractionBytes,
        saturatingMultiply(derivativeScratch.capacity(), sizeof(double)));
    processInteractionBytes = saturatingAdd(
        processInteractionBytes,
        saturatingMultiply(processOperatorScratch.capacity(), sizeof(double)));
    for(const auto& peer : processM2LSends)
    {
        processInteractionBytes = saturatingAdd(
            processInteractionBytes, peer.second.capacity());
    }
    stats.peakProcessBytes = std::max(
        stats.peakProcessBytes, processInteractionBytes);
    stats.processInteractionSeconds = elapsed(processInteractionStart);

    const Clock::time_point processDownStart = Clock::now();
    std::vector<double> translated(2 * layout.coefficientCount(), 0.0);
    if(!processTree_.nodes().empty())
    {
        if(processTree_.nodes()[processTree_.root()].owner == rank_)
            processLocals.ensure(processTree_.root());
        for(std::size_t depth = 0; depth < processTree_.maxDepth(); ++depth)
        {
            std::unordered_map<int, std::vector<char>> sendBuffers;
            for(std::size_t parentIndex : processTree_.levels()[depth])
            {
                const FmmProcessNode& parent =
                    processTree_.nodes()[parentIndex];
                if(parent.owner != rank_ || parent.isLeaf())
                    continue;
                const std::size_t parentOffset =
                    processLocals.ensure(parentIndex);
                const std::size_t children[2] = {parent.left, parent.right};
                for(std::size_t childIndex : children)
                {
                    const FmmProcessNode& child =
                        processTree_.nodes()[childIndex];
                    std::fill(translated.begin(), translated.end(), 0.0);
                    std::copy(processLocals.values.begin() +
                                  static_cast<std::ptrdiff_t>(parentOffset),
                              processLocals.values.begin() +
                                  static_cast<std::ptrdiff_t>(
                                      parentOffset + layout.coefficientCount()),
                              translated.begin());
                    FmmNode parentNode = processView(parent, 0);
                    FmmNode childNode = processView(
                        child, layout.coefficientCount());
                    FmmKernels::translateL2L(parentNode, childNode,
                                             layout, translated);
                    const double* coefficients = translated.data() +
                        layout.coefficientCount();
                    if(child.owner == rank_)
                    {
                        processLocals.add(childIndex, coefficients);
                    }
                    else
                    {
                        FmmProcessCoefficientHeader header;
                        header.stamp = fmmPacketStamp(
                            FmmPacketKind::ProcessCoefficient,
                            topologyEpoch_);
                        header.nodeIndex =
                            static_cast<std::uint64_t>(childIndex);
                        FmmPacketIO::appendPod(
                            sendBuffers[child.owner], header);
                        FmmPacketIO::appendDoubles(
                            sendBuffers[child.owner], coefficients,
                            layout.coefficientCount());
                    }
                }
            }
            FmmPeerExchangeResult received =
                processDownExchange_.exchangeBytes(
                    sendBuffers, &stats.bytesSent, &stats.bytesReceived);
            parseAndAddProcessCoefficients(
                received, processLocals, processTree_, topologyEpoch_, rank_);
            received.releaseStorage();
            std::size_t processDownBytes = saturatingAdd(
                processMultipoles.bytesOwned(), processLocals.bytesOwned());
            processDownBytes = saturatingAdd(
                processDownBytes,
                saturatingMultiply(translated.capacity(), sizeof(double)));
            stats.peakProcessBytes = std::max(
                stats.peakProcessBytes, processDownBytes);
        }

        for(FmmLocalPatch& patch : forest_.patches())
        {
            const std::size_t leaf = processTree_.leafForPatch(patch.key);
            if(leaf == FmmProcessTree::invalidIndex())
                throw UniversalError(
                    "FmmPatchDistributedSolver::solve: missing local process leaf");
            const std::size_t processOffset = processLocals.ensure(leaf);
            const std::size_t patchOffset = patch.tree.nodes()[0].localOffset;
            for(std::size_t i = 0; i < layout.coefficientCount(); ++i)
                patch.locals[patchOffset + i] +=
                    processLocals.values[processOffset + i];
        }
    }
    stats.processDownwardSeconds = elapsed(processDownStart);

    // Process coefficients and translation scratch are dead after their local
    // expansions have been injected into patch roots. Release them before the
    // LET waves so process storage cannot overlap the largest remote payload.
    processMultipoles.release();
    processLocals.release();
    std::unordered_map<int, std::vector<char>>().swap(processM2LSends);
    std::vector<double>().swap(derivativeScratch);
    std::vector<double>().swap(processOperatorScratch);
    std::vector<double>().swap(translated);

    const Clock::time_point interactionStart = Clock::now();
    const Clock::time_point localStart = Clock::now();
    forest_.executeLocalSelf(layout, operatorCache_,
                             options_.maxOperatorCacheBytes, stats);
    for(const FmmPatchPair& pair : processPlan_.localCrossPatchPairs)
    {
        if(pair.target.ownerRank != rank_ || pair.source.ownerRank != rank_ ||
           pair.target == pair.source)
            throw UniversalError(
                "FmmPatchDistributedSolver::solve: invalid local cross-patch pair");
        const std::size_t targetIndex =
            forest_.findPatch(pair.target.patchId);
        const std::size_t sourceIndex =
            forest_.findPatch(pair.source.patchId);
        if(targetIndex == std::numeric_limits<std::size_t>::max() ||
           sourceIndex == std::numeric_limits<std::size_t>::max())
            throw UniversalError(
                "FmmPatchDistributedSolver::solve: local cross-patch endpoint missing");
        FmmLocalPatch& target = forest_.patches()[targetIndex];
        const FmmLocalPatch& source = forest_.patches()[sourceIndex];
        if(distributedOptions_.useLocalPatchLet)
        {
            std::vector<double>* potential =
                target.potential.empty() ? nullptr : &target.potential;
            FmmDualTreeTraversal::run(
                target.tree, source.tree, target.positions, source.positions,
                source.masses, layout, source.multipoles, target.locals,
                false, options_.thetaCritical, target.acceleration, potential,
                operatorCache_, options_.maxOperatorCacheBytes, stats);
        }
        else
        {
            executeDirectPatchPair(target, source, stats);
        }
    }
    stats.localPlannedM2LCount = 0;
    stats.localPlannedP2PBlockCount = 0;
    stats.localInteractionPlanBytes = 0;
    for(const FmmLocalPatch& patch : forest_.patches())
    {
        stats.localPlannedM2LCount += static_cast<std::uint64_t>(
            patch.localPlan.m2lPairs.size());
        stats.localPlannedP2PBlockCount += static_cast<std::uint64_t>(
            patch.localPlan.p2pPairs.size());
        stats.localInteractionPlanBytes += patch.localPlan.bytesOwned();
    }
    stats.localTraversalSeconds = elapsed(localStart);

    const std::size_t persistentLetBytes = letPlan_.bytesOwned();
    const bool localLetBudgetOk = persistentLetBytes <=
        distributedOptions_.maxRemoteBytes -
            std::min<std::size_t>(2, distributedOptions_.maxRemoteBytes);
    collectiveRequire(
        localLetBudgetOk,
        localLetBudgetOk ? std::string() :
            "FmmPatchDistributedSolver::solve: persistent LET plan exhausts remote memory budget",
        "FmmPatchDistributedSolver::solve LET plan budget", comm_);
    const std::size_t transientRemoteBytes =
        distributedOptions_.maxRemoteBytes - persistentLetBytes;
    letPlan_.execute(forest_, layout, operatorCache_,
                     transientRemoteBytes,
                     options_.maxOperatorCacheBytes, stats);
    stats.peakRemoteBytes = saturatingAdd(
        persistentLetBytes, stats.peakRemoteBytes);
    stats.interactionSeconds = elapsed(interactionStart);

    const Clock::time_point downwardStart = Clock::now();
    forest_.applyDownward(layout);
    acceleration.assign(positions.size(), Vector3D());
    forest_.scatterAcceleration(acceleration);
    if(positiveKernelPotential != nullptr)
    {
        positiveKernelPotential->assign(positions.size(), 0.0);
        forest_.scatterPotential(*positiveKernelPotential);
    }
    stats.downwardSeconds = elapsed(downwardStart);

    if(options_.validateFinite)
    {
        int localFinite = 1;
        for(std::size_t index = 0; index < acceleration.size(); ++index)
        {
            if(!finiteVector(acceleration[index]) ||
               (positiveKernelPotential != nullptr &&
                !std::isfinite((*positiveKernelPotential)[index])))
            {
                localFinite = 0;
                break;
            }
        }
        int globalFinite = 0;
        MPI_Allreduce(&localFinite, &globalFinite, 1, MPI_INT, MPI_LAND, comm_);
        if(globalFinite == 0)
            throw UniversalError(
                "FmmPatchDistributedSolver::solve: non-finite output");
    }

    stats.nodeCount = 0;
    stats.leafCount = 0;
    stats.maxDepth = 0;
    stats.maxLeafOccupancy = 0;
    std::size_t occupiedLeaves = 0;
    std::size_t occupiedParticles = 0;
    for(const FmmLocalPatch& patch : forest_.patches())
    {
        stats.nodeCount += patch.tree.nodes().size();
        stats.leafCount += patch.tree.leafCount();
        for(const FmmNode& node : patch.tree.nodes())
        {
            stats.maxDepth = std::max(stats.maxDepth,
                                      static_cast<std::size_t>(node.depth));
            if(node.isLeaf())
            {
                stats.maxLeafOccupancy = std::max(
                    stats.maxLeafOccupancy, node.particleCount());
                if(node.particleCount() != 0)
                {
                    ++occupiedLeaves;
                    occupiedParticles += node.particleCount();
                }
            }
        }
    }
    stats.averageLeafOccupancy = occupiedLeaves == 0 ? 0.0 :
        static_cast<double>(occupiedParticles) /
        static_cast<double>(occupiedLeaves);
    const PatchForestMemory forestMemory = patchForestMemory(forest_);
    stats.localTreeBytes = forestMemory.treeAndInputs;
    stats.localMultipoleBytes = forestMemory.multipoles;
    stats.localLocalBytes = forestMemory.locals;
    stats.letPlanBytes = letPlan_.bytesOwned();
    stats.operatorCacheBytes = operatorCache_.bytesOwned();
    stats.operatorCacheEntries = operatorCache_.entries();
    stats.operatorCacheMaxEntries = operatorCache_.maxEntries();
    stats.bytesOwned = forestMemory.total;
    stats.bytesOwned = saturatingAdd(stats.bytesOwned, stats.letPlanBytes);
    stats.bytesOwned = saturatingAdd(stats.bytesOwned, saturatingMultiply(
        rootDescriptors_.capacity(), sizeof(FmmPatchRootDescriptor)));
    stats.bytesOwned = saturatingAdd(stats.bytesOwned,
                                     processTree_.bytesOwned());
    stats.bytesOwned = saturatingAdd(stats.bytesOwned,
                                     processPlan_.bytesOwned());
    stats.bytesOwned = saturatingAdd(stats.bytesOwned,
                                     operatorCache_.bytesOwned());
    stats.bytesOwned = saturatingAdd(stats.bytesOwned,
                                     processUpExchange_.bytesOwned());
    stats.bytesOwned = saturatingAdd(stats.bytesOwned,
                                     processM2LExchange_.bytesOwned());
    stats.bytesOwned = saturatingAdd(stats.bytesOwned,
                                     processDownExchange_.bytesOwned());
    stats.totalSeconds = elapsed(totalStart);
}

#endif // RICH_MPI
