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

std::size_t patchForestBytes(const FmmPatchForest& forest)
{
    std::size_t bytes = saturatingMultiply(
        forest.patches().capacity(), sizeof(FmmLocalPatch));
    for(const FmmLocalPatch& patch : forest.patches())
    {
        bytes = saturatingAdd(bytes, saturatingMultiply(
            patch.inputIndices.capacity(), sizeof(std::size_t)));
        bytes = saturatingAdd(bytes, saturatingMultiply(
            patch.positions.capacity(), sizeof(Vector3D)));
        bytes = saturatingAdd(bytes, saturatingMultiply(
            patch.masses.capacity(), sizeof(double)));
        bytes = saturatingAdd(bytes, saturatingMultiply(
            patch.cellIds.capacity(), sizeof(std::uint64_t)));
        bytes = saturatingAdd(bytes, patch.tree.bytesOwned());
        bytes = saturatingAdd(bytes, patch.localPlan.bytesOwned());
        bytes = saturatingAdd(bytes, saturatingMultiply(
            patch.multipoles.capacity(), sizeof(double)));
        bytes = saturatingAdd(bytes, saturatingMultiply(
            patch.locals.capacity(), sizeof(double)));
        bytes = saturatingAdd(bytes, saturatingMultiply(
            patch.acceleration.capacity(), sizeof(Vector3D)));
        bytes = saturatingAdd(bytes, saturatingMultiply(
            patch.potential.capacity(), sizeof(double)));
    }
    return bytes;
}
}

FmmPatchDistributedSolver::FmmPatchDistributedSolver(
    const FmmGravityOptions& options,
    const FmmDistributedOptions& distributedOptions,
    const MPI_Comm& comm):
    options_(options), distributedOptions_(distributedOptions), comm_(comm),
    rank_(0), size_(1), topologyEpoch_(0), topologyRebuildCount_(0)
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

    if(topologyEpoch_ == std::numeric_limits<std::uint64_t>::max() ||
       topologyRebuildCount_ == std::numeric_limits<std::uint64_t>::max())
        throw UniversalError(
            "FmmPatchDistributedSolver::solve: topology epoch overflow");
    ++topologyEpoch_;
    ++topologyRebuildCount_;

    const Clock::time_point buildStart = Clock::now();
    FmmPatchForestChange forestChange;
    bool localPrepareOk = true;
    std::string localPrepareError;
    try
    {
        forestChange = forest_.prepare(
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
    const unsigned long long localTopologyTerms[4] = {
        stats.localRootGeometryChanged ? 1ull : 0ull,
        stats.localLeafTopologyChanged ? 1ull : 0ull,
        stats.localLeafOccupancyChanged ? 1ull : 0ull,
        stats.localCountOnlyLeafChange ? 1ull : 0ull};
    unsigned long long globalTopologyTerms[4] = {};
    MPI_Allreduce(localTopologyTerms, globalTopologyTerms, 4,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_);
    stats.ranksWithRootGeometryChange =
        static_cast<std::size_t>(globalTopologyTerms[0]);
    stats.ranksWithLeafTopologyChange =
        static_cast<std::size_t>(globalTopologyTerms[1]);
    stats.ranksWithLeafOccupancyChange =
        static_cast<std::size_t>(globalTopologyTerms[2]);
    stats.ranksWithCountOnlyLeafChange =
        static_cast<std::size_t>(globalTopologyTerms[3]);
    stats.topologyRebuildForced = true;

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

    const Clock::time_point topologyStart = Clock::now();
    const std::vector<FmmPatchRootDescriptor> localDescriptors =
        forest_.descriptors(rank_, topologyEpoch_);
    const Clock::time_point gatherStart = Clock::now();
    rootDescriptors_ = FmmDescriptorGather::gather(
        localDescriptors, static_cast<std::uint64_t>(positions.size()),
        topologyEpoch_, distributedOptions_.maxReplicatedDescriptorBytes,
        comm_);
    stats.rootDescriptorExchangeSeconds = elapsed(gatherStart);

    const Clock::time_point processTopologyStart = Clock::now();
    processTree_.build(rootDescriptors_);
    processPlan_ = FmmProcessTraversal::build(
        processTree_, options_.thetaCritical, topologyEpoch_, rank_, comm_);

    std::vector<FmmPatchKey> expectedLocalSelf;
    expectedLocalSelf.reserve(forest_.patches().size());
    for(const FmmLocalPatch& patch : forest_.patches())
        expectedLocalSelf.push_back(patch.key);
    std::sort(expectedLocalSelf.begin(), expectedLocalSelf.end());
    if(processPlan_.localSelfPatches != expectedLocalSelf ||
       processPlan_.localSelfRankCount != expectedLocalSelf.size())
        throw UniversalError(
            "FmmPatchDistributedSolver::solve: local self patch coverage mismatch");

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
    const bool m2lReset = processM2LExchange_.resetIfChanged(comm_, m2lPeers);
    const bool downReset = processDownExchange_.resetIfChanged(
        comm_, std::vector<int>(downPeers.begin(), downPeers.end()));
    stats.processCommunicatorsReused = !upReset && !m2lReset && !downReset;
    stats.processTopologySeconds = elapsed(processTopologyStart);

    letPlan_.build(forest_, rootDescriptors_, processPlan_,
                   options_.thetaCritical, topologyEpoch_,
                   distributedOptions_.maxLetWaveBytes,
                   distributedOptions_.maxTargetPatchesPerWave,
                   layout.coefficientCount(), comm_, stats);
    // The process tree and LET plan now own the compact state they need. The
    // replicated root directory is rebuilt every solve in Phases 3-4, so do
    // not retain it through the expensive interaction passes.
    std::vector<FmmPatchRootDescriptor>().swap(rootDescriptors_);
    stats.topologyRebuildSeconds = elapsed(topologyStart);
    stats.processTopologyRebuilt = true;
    stats.letTopologyRebuilt = true;
    stats.topologyEpoch = topologyEpoch_;
    stats.topologyRebuildCount = topologyRebuildCount_;
    stats.processTopologyRebuildCount = topologyRebuildCount_;
    stats.letTopologyRebuildCount = topologyRebuildCount_;
    stats.activeRankCount = processTree_.activeRanks().size();
    stats.processNodeCount = processTree_.nodes().size();

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

    letPlan_.execute(forest_, layout, operatorCache_,
                     distributedOptions_.maxRemoteBytes,
                     options_.maxOperatorCacheBytes, stats);
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
    stats.localTreeBytes = patchForestBytes(forest_);
    stats.letPlanBytes = letPlan_.bytesOwned();
    stats.operatorCacheBytes = operatorCache_.bytesOwned();
    stats.operatorCacheEntries = operatorCache_.entries();
    stats.operatorCacheMaxEntries = operatorCache_.maxEntries();
    stats.bytesOwned = stats.localTreeBytes;
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
