#include "3D/gravity/fmm/mpi/DistributedFmmGravityCalculator.hpp"

#ifdef RICH_MPI

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "3D/gravity/fmm/FmmDualTreeTraversal.hpp"
#include "3D/gravity/fmm/FmmKernels.hpp"
#include "3D/gravity/fmm/FmmPasses.hpp"
#include "3D/gravity/fmm/FmmTaylorExpansion.hpp"
#include "3D/gravity/fmm/mpi/FmmPackets.hpp"
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

bool sameRoot(const FmmRootGeometry& first, const FmmRootGeometry& second)
{
    return first.active == second.active &&
        (!first.active || (first.center.x == second.center.x &&
                           first.center.y == second.center.y &&
                           first.center.z == second.center.z &&
                           first.halfSize == second.halfSize));
}

std::vector<std::uint64_t> leafTopologySignature(const FmmTree& tree)
{
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
                  "Distributed FMM topology signatures require <=64-bit size_t");
    std::vector<std::uint64_t> signature;
    if(tree.leafCount() > std::numeric_limits<std::size_t>::max() / 2)
        throw UniversalError(
            "DistributedFmmGravityCalculator: topology signature size overflow");
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

struct ProcessCoefficientStore
{
    explicit ProcessCoefficientStore(std::size_t count):
        coefficientCount(count) {}

    std::size_t ensure(std::size_t node)
    {
        const auto found = slotByNode.find(node);
        if(found != slotByNode.end())
            return found->second * coefficientCount;
        const std::size_t slot = slotByNode.size();
        if(coefficientCount == 0 ||
           slot == std::numeric_limits<std::size_t>::max() ||
           slot + 1 > std::numeric_limits<std::size_t>::max() / coefficientCount)
            throw UniversalError("ProcessCoefficientStore: storage size overflow");
        slotByNode[node] = slot;
        values.resize((slot + 1) * coefficientCount, 0.0);
        return slot * coefficientCount;
    }

    std::size_t offset(std::size_t node) const
    {
        const auto found = slotByNode.find(node);
        if(found == slotByNode.end())
            throw UniversalError("ProcessCoefficientStore: missing node");
        return found->second * coefficientCount;
    }

    void zero(std::size_t node)
    {
        const std::size_t begin = ensure(node);
        std::fill(values.begin() + static_cast<std::ptrdiff_t>(begin),
                  values.begin() + static_cast<std::ptrdiff_t>(begin + coefficientCount),
                  0.0);
    }

    void add(std::size_t node, const double* coefficients)
    {
        const std::size_t begin = ensure(node);
        for(std::size_t i = 0; i < coefficientCount; ++i)
            values[begin + i] += coefficients[i];
    }

    std::size_t bytesOwned() const
    {
        const std::size_t mapEntryBytes =
            sizeof(std::pair<const std::size_t, std::size_t>) +
            2 * sizeof(void*);
        return values.capacity() * sizeof(double) +
               slotByNode.size() * mapEntryBytes;
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
    FmmNode result;
    result.center = processNode.center;
    result.halfSize = processNode.halfSize;
    result.radius = processNode.radius;
    result.multipoleOffset = offset;
    result.localOffset = offset;
    return result;
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
    const std::size_t begin = store.offset(node);
    FmmPacketIO::appendDoubles(buffer, store.values.data() + begin,
                               store.coefficientCount);
}

void parseProcessCoefficients(
    const FmmPeerExchangeResult& received,
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
            validateFmmPacketStamp(header.stamp,
                FmmPacketKind::ProcessCoefficient, topologyEpoch,
                "parseProcessCoefficients");
            if(header.nodeIndex >= tree.nodes().size())
                throw UniversalError("parseProcessCoefficients: node index out of range");
            const std::size_t node = static_cast<std::size_t>(header.nodeIndex);
            if(tree.nodes()[node].owner != message.source)
                throw UniversalError("parseProcessCoefficients: coefficient sent by non-owner");
            if(!receivedNodes.insert(node).second)
                throw UniversalError("parseProcessCoefficients: duplicate node coefficient");
            const std::size_t begin = store.ensure(node);
            FmmPacketIO::readDoubles(view, offset,
                                     store.values.data() + begin,
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
    std::vector<double> coefficients(store.coefficientCount);
    for(const FmmReceivedMessage& message : received.messages())
    {
        const FmmByteView view = received.view(message);
        std::size_t offset = 0;
        while(offset < view.size)
        {
            const FmmProcessCoefficientHeader header =
                FmmPacketIO::readPod<FmmProcessCoefficientHeader>(view, offset);
            validateFmmPacketStamp(header.stamp,
                FmmPacketKind::ProcessCoefficient, topologyEpoch,
                "parseAndAddProcessCoefficients");
            if(header.nodeIndex >= tree.nodes().size())
                throw UniversalError("parseAndAddProcessCoefficients: node index out of range");
            const std::size_t node = static_cast<std::size_t>(header.nodeIndex);
            const FmmProcessNode& target = tree.nodes()[node];
            if(target.owner != rank || target.parent == FmmProcessTree::invalidIndex() ||
               tree.nodes()[target.parent].owner != message.source)
                throw UniversalError("parseAndAddProcessCoefficients: invalid downward sender");
            if(!receivedNodes.insert(node).second)
                throw UniversalError("parseAndAddProcessCoefficients: duplicate node coefficient");
            FmmPacketIO::readDoubles(view, offset, coefficients.data(),
                                     coefficients.size());
            store.add(node, coefficients.data());
        }
    }
}


struct DisplacementKey
{
    std::uint64_t x;
    std::uint64_t y;
    std::uint64_t z;
    bool operator==(const DisplacementKey& other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct DisplacementKeyHash
{
    std::size_t operator()(const DisplacementKey& key) const
    {
        std::size_t result = std::hash<std::uint64_t>()(key.x);
        result ^= std::hash<std::uint64_t>()(key.y) + 0x9e3779b9u +
            (result << 6u) + (result >> 2u);
        result ^= std::hash<std::uint64_t>()(key.z) + 0x9e3779b9u +
            (result << 6u) + (result >> 2u);
        return result;
    }
};

DisplacementKey displacementKey(const Vector3D& displacement)
{
    DisplacementKey key;
    std::memcpy(&key.x, &displacement.x, sizeof(double));
    std::memcpy(&key.y, &displacement.y, sizeof(double));
    std::memcpy(&key.z, &displacement.z, sizeof(double));
    return key;
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
            throw UniversalError(std::string(context) + ": failed on another MPI rank");
        throw UniversalError(localMessage.empty() ? std::string(context) : localMessage);
    }
}
}

DistributedFmmGravityCalculator::DistributedFmmGravityCalculator(
    FmmGravityOptions options,
    FmmDistributedOptions distributedOptions,
    const MPI_Comm& comm):
    options_(options),
    distributedOptions_(distributedOptions),
    comm_(MPI_COMM_NULL),
    rank_(0),
    size_(1),
    rootInitialized_(false),
    lastLocalTopologyHash_(0),
    topologyEpoch_(0),
    topologyRebuildCount_(0)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if(initialized == 0)
        throw UniversalError("DistributedFmmGravityCalculator: MPI must be initialized before construction");
    if(comm == MPI_COMM_NULL)
        throw UniversalError("DistributedFmmGravityCalculator: communicator is null");
    MPI_Comm_dup(comm, &comm_);
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);

    bool localOptionsOk = true;
    std::string localOptionsError;
    if(options_.expansionOrder < 1 || options_.expansionOrder > FMM_MAX_ORDER)
    {
        localOptionsOk = false;
        localOptionsError = "DistributedFmmGravityCalculator: invalid expansion order";
    }
    else if(!(options_.thetaCritical > 0.0) || options_.thetaCritical > 1.0 ||
            !std::isfinite(options_.thetaCritical))
    {
        localOptionsOk = false;
        localOptionsError = "DistributedFmmGravityCalculator: invalid theta";
    }
    else if(options_.leafCapacity == 0)
    {
        localOptionsOk = false;
        localOptionsError = "DistributedFmmGravityCalculator: leaf capacity is zero";
    }
    else if(options_.maxDepth <= 0 || options_.maxDepth > FMM_MAX_TREE_DEPTH)
    {
        localOptionsOk = false;
        localOptionsError = "DistributedFmmGravityCalculator: invalid maximum tree depth";
    }
    else if(!(distributedOptions_.rootSlackFactor >= 1.0) ||
            !std::isfinite(distributedOptions_.rootSlackFactor))
    {
        localOptionsOk = false;
        localOptionsError = "DistributedFmmGravityCalculator: invalid root slack factor";
    }
    else if(distributedOptions_.maxRemoteBytes < 2)
    {
        localOptionsOk = false;
        localOptionsError = "DistributedFmmGravityCalculator: remote memory budget is too small";
    }

    int localValid = localOptionsOk ? 1 : 0;
    int globalValid = 0;
    MPI_Allreduce(&localValid, &globalValid, 1, MPI_INT, MPI_LAND, comm_);
    if(globalValid == 0)
    {
        MPI_Comm doomed = comm_;
        comm_ = MPI_COMM_NULL;
        MPI_Comm_free(&doomed);
        throw UniversalError(localOptionsOk ?
            "DistributedFmmGravityCalculator: invalid options on another MPI rank" :
            localOptionsError);
    }

    const double localDoubleOptions[2] = {
        options_.thetaCritical, distributedOptions_.rootSlackFactor};
    double minimumDoubleOptions[2] = {0.0, 0.0};
    double maximumDoubleOptions[2] = {0.0, 0.0};
    MPI_Allreduce(localDoubleOptions, minimumDoubleOptions, 2,
                  MPI_DOUBLE, MPI_MIN, comm_);
    MPI_Allreduce(localDoubleOptions, maximumDoubleOptions, 2,
                  MPI_DOUBLE, MPI_MAX, comm_);

    const unsigned long long localIntegerOptions[7] = {
        static_cast<unsigned long long>(options_.expansionOrder),
        static_cast<unsigned long long>(options_.leafCapacity),
        static_cast<unsigned long long>(options_.maxDepth),
        options_.computePotential ? 1ull : 0ull,
        options_.validateFinite ? 1ull : 0ull,
        static_cast<unsigned long long>(distributedOptions_.maxRemoteBytes),
        distributedOptions_.rebuildTopologyEverySolve ? 1ull : 0ull};
    unsigned long long minimumIntegerOptions[7] = {};
    unsigned long long maximumIntegerOptions[7] = {};
    MPI_Allreduce(localIntegerOptions, minimumIntegerOptions, 7,
                  MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm_);
    MPI_Allreduce(localIntegerOptions, maximumIntegerOptions, 7,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm_);

    bool optionsMatch = true;
    for(int i = 0; i < 2; ++i)
        optionsMatch = optionsMatch &&
            minimumDoubleOptions[i] == maximumDoubleOptions[i];
    for(int i = 0; i < 7; ++i)
        optionsMatch = optionsMatch &&
            minimumIntegerOptions[i] == maximumIntegerOptions[i];
    if(!optionsMatch)
    {
        MPI_Comm doomed = comm_;
        comm_ = MPI_COMM_NULL;
        MPI_Comm_free(&doomed);
        throw UniversalError(
            "DistributedFmmGravityCalculator: inconsistent options across MPI ranks");
    }
}

DistributedFmmGravityCalculator::~DistributedFmmGravityCalculator()
{
    processUpExchange_.clear();
    processM2LExchange_.clear();
    processDownExchange_.clear();
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if(initialized != 0)
        MPI_Finalized(&finalized);
    if(comm_ != MPI_COMM_NULL && initialized != 0 && finalized == 0)
        MPI_Comm_free(&comm_);
    comm_ = MPI_COMM_NULL;
}

void DistributedFmmGravityCalculator::validateInputs(
    const std::vector<Vector3D>& positions,
    const std::vector<double>& masses,
    const std::vector<std::uint64_t>& cellIds,
    const Vector3D& domainLower,
    const Vector3D& domainUpper,
    std::vector<double>* positiveKernelPotential) const
{
    if(positions.size() != masses.size() || positions.size() != cellIds.size())
        throw UniversalError("DistributedFmmGravityCalculator::solve: input size mismatch");
    if(options_.computePotential != (positiveKernelPotential != nullptr))
        throw UniversalError("DistributedFmmGravityCalculator::solve: potential option mismatch");
    if(!finiteVector(domainLower) || !finiteVector(domainUpper) ||
       !(domainLower.x < domainUpper.x) ||
       !(domainLower.y < domainUpper.y) ||
       !(domainLower.z < domainUpper.z))
        throw UniversalError("DistributedFmmGravityCalculator::solve: invalid domain");
    for(std::size_t i = 0; i < positions.size(); ++i)
    {
        if(!finiteVector(positions[i]) || !std::isfinite(masses[i]))
        {
            UniversalError error("DistributedFmmGravityCalculator::solve: non-finite input");
            error.addEntry("particle", i);
            throw error;
        }
        if(positions[i].x < domainLower.x || positions[i].x > domainUpper.x ||
           positions[i].y < domainLower.y || positions[i].y > domainUpper.y ||
           positions[i].z < domainLower.z || positions[i].z > domainUpper.z)
        {
            UniversalError error("DistributedFmmGravityCalculator::solve: particle lies outside domain");
            error.addEntry("particle", i);
            throw error;
        }
    }
}

bool DistributedFmmGravityCalculator::prepareLocalTree(
    const std::vector<Vector3D>& positions,
    const Vector3D& domainLower,
    const Vector3D& domainUpper)
{
    FmmRootGeometry nextRoot;
    if(!positions.empty())
    {
        bool contained = rootInitialized_ && localRoot_.active;
        if(contained)
        {
            for(const Vector3D& point : positions)
            {
                if(!localRoot_.contains(point))
                {
                    contained = false;
                    break;
                }
            }
        }
        nextRoot = contained ? localRoot_ :
            FmmRootGeometry::containingPoints(positions, domainLower, domainUpper,
                                               distributedOptions_.rootSlackFactor);
    }

    localTree_.build(positions, nextRoot, options_);
    const std::uint64_t hash = localTree_.topologyHash();
    std::vector<std::uint64_t> signature = leafTopologySignature(localTree_);
    const bool changed = !rootInitialized_ || !sameRoot(localRoot_, nextRoot) ||
                         signature != lastLocalTopologySignature_ ||
                         distributedOptions_.rebuildTopologyEverySolve;
    localRoot_ = nextRoot;
    lastLocalTopologyHash_ = hash;
    lastLocalTopologySignature_.swap(signature);
    rootInitialized_ = true;
    return changed;
}

FmmRankRootDescriptor DistributedFmmGravityCalculator::localRootDescriptor() const
{
    FmmRankRootDescriptor result;
    result.rank = rank_;
    result.active = localTree_.nodes().empty() ? 0 : 1;
    result.topologyHash = lastLocalTopologyHash_;
    result.epoch = topologyEpoch_;
    if(result.active != 0)
    {
        const FmmNode& root = localTree_.nodes()[0];
        result.center[0] = root.center.x;
        result.center[1] = root.center.y;
        result.center[2] = root.center.z;
        result.halfSize = root.halfSize;
        result.particleCount = static_cast<std::uint64_t>(root.particleCount());
        result.rootLeaf = root.isLeaf() ? 1 : 0;
        result.childMask = static_cast<int>(root.childMask);
    }
    return result;
}

void DistributedFmmGravityCalculator::rebuildTopology()
{
    if(topologyEpoch_ == std::numeric_limits<std::uint64_t>::max() ||
       topologyRebuildCount_ == std::numeric_limits<std::uint64_t>::max())
        throw UniversalError(
            "DistributedFmmGravityCalculator::rebuildTopology: topology epoch overflow");
    ++topologyEpoch_;
    ++topologyRebuildCount_;
    const FmmRankRootDescriptor local = localRootDescriptor();
    rootDescriptors_.resize(static_cast<std::size_t>(size_));
    MPI_Allgather(&local, static_cast<int>(sizeof(FmmRankRootDescriptor)), MPI_BYTE,
                  rootDescriptors_.data(),
                  static_cast<int>(sizeof(FmmRankRootDescriptor)), MPI_BYTE,
                  comm_);

    processTree_.build(rootDescriptors_);
    processPlan_ = FmmProcessTraversal::build(processTree_,
                                               options_.thetaCritical, topologyEpoch_,
                                               rank_, comm_);

    std::set<int> upPeers;
    std::set<int> downPeers;
    for(std::size_t i = 0; i < processTree_.nodes().size(); ++i)
    {
        const FmmProcessNode& node = processTree_.nodes()[i];
        if(node.owner == rank_ && node.parent != FmmProcessTree::invalidIndex())
        {
            const int parentOwner = processTree_.nodes()[node.parent].owner;
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
    processUpExchange_.reset(comm_, std::vector<int>(upPeers.begin(), upPeers.end()));
    processM2LExchange_.reset(comm_, m2lPeers);
    processDownExchange_.reset(comm_,
        std::vector<int>(downPeers.begin(), downPeers.end()));
    letPlan_.build(localTree_, rootDescriptors_, processPlan_,
                   options_.thetaCritical, topologyEpoch_, comm_, stats_);
}

void DistributedFmmGravityCalculator::solve(
    const std::vector<Vector3D>& positions,
    const std::vector<double>& masses,
    const std::vector<std::uint64_t>& cellIds,
    const Vector3D& domainLower,
    const Vector3D& domainUpper,
    std::vector<Vector3D>& acceleration,
    std::vector<double>* positiveKernelPotential)
{
    bool localInputsValid = true;
    std::string localInputError;
    try
    {
        validateInputs(positions, masses, cellIds, domainLower, domainUpper,
                       positiveKernelPotential);
    }
    catch(const UniversalError& error)
    {
        localInputsValid = false;
        localInputError = error.getErrorMessage();
    }
    collectiveRequire(localInputsValid, localInputError,
                      "DistributedFmmGravityCalculator::solve input validation",
                      comm_);

    const double localDomain[6] = {
        domainLower.x, domainLower.y, domainLower.z,
        domainUpper.x, domainUpper.y, domainUpper.z};
    double minimumDomain[6] = {};
    double maximumDomain[6] = {};
    MPI_Allreduce(localDomain, minimumDomain, 6, MPI_DOUBLE, MPI_MIN, comm_);
    MPI_Allreduce(localDomain, maximumDomain, 6, MPI_DOUBLE, MPI_MAX, comm_);
    bool commonDomain = true;
    for(int i = 0; i < 6; ++i)
        commonDomain = commonDomain && minimumDomain[i] == maximumDomain[i];
    collectiveRequire(commonDomain,
        "DistributedFmmGravityCalculator::solve: domain bounds differ across MPI ranks",
        "DistributedFmmGravityCalculator::solve domain validation", comm_);

    const Clock::time_point totalStart = Clock::now();
    stats_ = FmmSolveStats();
    stats_.particleCount = positions.size();
    stats_.mpiRankCount = static_cast<std::size_t>(size_);

    const Clock::time_point buildStart = Clock::now();
    const bool localChanged = prepareLocalTree(positions, domainLower, domainUpper);
    FmmPasses::updateTreeStats(localTree_, stats_);
    stats_.buildSeconds = elapsed(buildStart);

    acceleration.assign(positions.size(), Vector3D());
    if(positiveKernelPotential != nullptr)
        positiveKernelPotential->assign(positions.size(), 0.0);

    const FmmTaylorExpansion layout(options_.expansionOrder);
    if(!localTree_.nodes().empty())
    {
        FmmPasses::allocate(localTree_, layout, localMultipoles_, localLocals_);
        const Clock::time_point upwardStart = Clock::now();
        FmmPasses::upward(localTree_, positions, masses, layout, localMultipoles_);
        stats_.upwardSeconds = elapsed(upwardStart);
    }
    else
    {
        localMultipoles_.clear();
        localLocals_.clear();
    }

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
    MPI_Allreduce(localMassTerms, globalMassTerms, 2, MPI_DOUBLE, MPI_SUM, comm_);
    stats_.totalMass = globalMassTerms[0];
    const double totalAbsoluteMass = globalMassTerms[1];
    if(!std::isfinite(stats_.totalMass) || !std::isfinite(totalAbsoluteMass))
        throw UniversalError(
            "DistributedFmmGravityCalculator::solve: non-finite global mass sum");

    int localChangedInt = localChanged ? 1 : 0;
    int globalChangedInt = 0;
    MPI_Allreduce(&localChangedInt, &globalChangedInt, 1, MPI_INT, MPI_LOR, comm_);
    if(globalChangedInt != 0)
        rebuildTopology();

    stats_.topologyEpoch = topologyEpoch_;
    stats_.topologyRebuildCount = topologyRebuildCount_;
    stats_.activeRankCount = processTree_.activeRanks().size();
    stats_.processNodeCount = processTree_.nodes().size();

    ProcessCoefficientStore processMultipoles(layout.coefficientCount());
    ProcessCoefficientStore processLocals(layout.coefficientCount());

    const Clock::time_point processUpStart = Clock::now();
    const std::size_t localLeaf = processTree_.leafForRank(rank_);
    if(localLeaf != FmmProcessTree::invalidIndex())
    {
        if(localTree_.nodes().empty())
            throw UniversalError("DistributedFmmGravityCalculator::solve: active process leaf has no local tree");
        const std::size_t begin = processMultipoles.ensure(localLeaf);
        std::copy(localMultipoles_.begin() +
                  static_cast<std::ptrdiff_t>(localTree_.nodes()[0].multipoleOffset),
                  localMultipoles_.begin() +
                  static_cast<std::ptrdiff_t>(localTree_.nodes()[0].multipoleOffset +
                                               layout.coefficientCount()),
                  processMultipoles.values.begin() + static_cast<std::ptrdiff_t>(begin));
    }

    stats_.peakProcessBytes = std::max(stats_.peakProcessBytes,
        processMultipoles.bytesOwned() + processLocals.bytesOwned());
    if(!processTree_.nodes().empty())
    {
        for(std::size_t depth = processTree_.maxDepth(); depth > 0; --depth)
        {
            std::unordered_map<int, std::vector<char>> sendBuffers;
            for(std::size_t nodeIndex : processTree_.levels()[depth])
            {
                const FmmProcessNode& node = processTree_.nodes()[nodeIndex];
                if(node.owner != rank_)
                    continue;
                const int parentOwner = processTree_.nodes()[node.parent].owner;
                if(parentOwner != rank_)
                    appendProcessCoefficients(sendBuffers[parentOwner], nodeIndex,
                                              processMultipoles, topologyEpoch_);
            }
            const auto received = processUpExchange_.exchangeBytes(
                sendBuffers, &stats_.bytesSent, &stats_.bytesReceived);
            parseProcessCoefficients(received, processMultipoles, processTree_,
                                     topologyEpoch_);

            for(std::size_t parentIndex : processTree_.levels()[depth - 1])
            {
                const FmmProcessNode& parent = processTree_.nodes()[parentIndex];
                if(parent.owner != rank_ || parent.isLeaf())
                    continue;
                processMultipoles.zero(parentIndex);
                const std::size_t parentOffset = processMultipoles.offset(parentIndex);
                FmmNode parentView = processView(parent, parentOffset);
                const std::size_t children[2] = {parent.left, parent.right};
                for(std::size_t childIndex : children)
                {
                    const std::size_t childOffset = processMultipoles.offset(childIndex);
                    FmmNode childView = processView(processTree_.nodes()[childIndex],
                                                    childOffset);
                    FmmKernels::translateM2M(childView, parentView, layout,
                                             processMultipoles.values);
                }
            }
            stats_.peakProcessBytes = std::max(stats_.peakProcessBytes,
                processMultipoles.bytesOwned() + processLocals.bytesOwned());
        }
    }
    stats_.processUpwardSeconds = elapsed(processUpStart);

    double localRootMass = 0.0;
    if(!processTree_.nodes().empty() &&
       processTree_.nodes()[processTree_.root()].owner == rank_)
    {
        const std::size_t rootOffset = processMultipoles.offset(processTree_.root());
        localRootMass = processMultipoles.values[rootOffset + layout.index(0, 0, 0)];
    }
    MPI_Allreduce(&localRootMass, &stats_.rootMass, 1, MPI_DOUBLE, MPI_SUM, comm_);
    if(!std::isfinite(stats_.rootMass))
        throw UniversalError(
            "DistributedFmmGravityCalculator::solve: non-finite global root mass");
    std::uint64_t globalParticleCount = 0;
    for(const FmmRankRootDescriptor& descriptor : rootDescriptors_)
    {
        if(descriptor.particleCount >
           std::numeric_limits<std::uint64_t>::max() - globalParticleCount)
            throw UniversalError(
                "DistributedFmmGravityCalculator::solve: global particle count overflow");
        globalParticleCount += descriptor.particleCount;
    }
    const long double accumulationFactor = static_cast<long double>(
        std::max<std::uint64_t>(1, globalParticleCount));
    const double massTolerance = static_cast<double>(
        256.0L * std::numeric_limits<double>::epsilon() * accumulationFactor *
        std::max(1.0L, static_cast<long double>(totalAbsoluteMass)));
    if(std::abs(stats_.rootMass - stats_.totalMass) > massTolerance)
        throw UniversalError("DistributedFmmGravityCalculator::solve: global root mass mismatch");

    const Clock::time_point processInteractionStart = Clock::now();
    std::unordered_map<int, std::vector<char>> processM2LSends;
    for(const auto& entry : processPlan_.processSendNodesByRank)
    {
        for(std::size_t nodeIndex : entry.second)
            appendProcessCoefficients(processM2LSends[entry.first], nodeIndex,
                                      processMultipoles, topologyEpoch_);
    }
    FmmPeerExchangeResult processM2LReceived = processM2LExchange_.exchangeBytes(
        processM2LSends, &stats_.bytesSent, &stats_.bytesReceived);
    parseProcessCoefficients(processM2LReceived, processMultipoles, processTree_,
                             topologyEpoch_);
    processM2LReceived.releaseStorage();
    std::unordered_map<int, std::vector<char>>().swap(processM2LSends);

    std::unordered_map<DisplacementKey, std::vector<double>, DisplacementKeyHash>
        processOperatorCache;
    processOperatorCache.reserve(processPlan_.localM2LPairs.size());
    std::vector<double> derivativeScratch;
    derivativeScratch.reserve(layout.coefficientCount());
    for(const FmmProcessM2LPair& pair : processPlan_.localM2LPairs)
    {
        const std::size_t sourceOffset = processMultipoles.offset(pair.sourceNode);
        const std::size_t targetOffset = processLocals.ensure(pair.targetNode);
        FmmNode source = processView(processTree_.nodes()[pair.sourceNode], sourceOffset);
        FmmNode target = processView(processTree_.nodes()[pair.targetNode], targetOffset);
        const Vector3D displacement = target.center - source.center;
        const DisplacementKey key = displacementKey(displacement);
        auto inserted = processOperatorCache.emplace(key, std::vector<double>());
        if(inserted.second)
            FmmKernels::computeM2LOperator(displacement, layout,
                                           derivativeScratch, inserted.first->second);
        FmmKernels::translateM2L(source, target, layout,
                                 processMultipoles.values, processLocals.values,
                                 inserted.first->second);
        ++stats_.m2lCount;
        ++stats_.processM2LCount;
    }
    stats_.peakProcessBytes = std::max(stats_.peakProcessBytes,
        processMultipoles.bytesOwned() + processLocals.bytesOwned());
    stats_.processInteractionSeconds = elapsed(processInteractionStart);
    std::unordered_map<DisplacementKey, std::vector<double>,
        DisplacementKeyHash>().swap(processOperatorCache);
    std::vector<double>().swap(derivativeScratch);

    const Clock::time_point processDownStart = Clock::now();
    std::vector<double> translatedProcessLocal(
        2 * layout.coefficientCount(), 0.0);
    if(!processTree_.nodes().empty())
    {
        if(processTree_.nodes()[processTree_.root()].owner == rank_)
            processLocals.ensure(processTree_.root());
        for(std::size_t depth = 0; depth < processTree_.maxDepth(); ++depth)
        {
            std::unordered_map<int, std::vector<char>> sendBuffers;
            for(std::size_t parentIndex : processTree_.levels()[depth])
            {
                const FmmProcessNode& parent = processTree_.nodes()[parentIndex];
                if(parent.owner != rank_ || parent.isLeaf())
                    continue;
                const std::size_t parentOffset = processLocals.ensure(parentIndex);
                const std::size_t children[2] = {parent.left, parent.right};
                for(std::size_t childIndex : children)
                {
                    const FmmProcessNode& childNode = processTree_.nodes()[childIndex];
                    std::fill(translatedProcessLocal.begin(),
                              translatedProcessLocal.end(), 0.0);
                    std::copy(processLocals.values.begin() +
                              static_cast<std::ptrdiff_t>(parentOffset),
                              processLocals.values.begin() +
                              static_cast<std::ptrdiff_t>(parentOffset +
                                                           layout.coefficientCount()),
                              translatedProcessLocal.begin());
                    FmmNode parentView = processView(parent, 0);
                    FmmNode childView = processView(childNode,
                                                    layout.coefficientCount());
                    FmmKernels::translateL2L(parentView, childView, layout,
                                             translatedProcessLocal);
                    if(childNode.owner == rank_)
                    {
                        processLocals.add(childIndex,
                            translatedProcessLocal.data() + layout.coefficientCount());
                    }
                    else
                    {
                        FmmProcessCoefficientHeader header;
                        header.stamp = fmmPacketStamp(
                            FmmPacketKind::ProcessCoefficient, topologyEpoch_);
                        header.nodeIndex = static_cast<std::uint64_t>(childIndex);
                        std::vector<char>& buffer = sendBuffers[childNode.owner];
                        FmmPacketIO::appendPod(buffer, header);
                        FmmPacketIO::appendDoubles(buffer,
                            translatedProcessLocal.data() + layout.coefficientCount(),
                            layout.coefficientCount());
                    }
                }
            }
            const FmmPeerExchangeResult received =
                processDownExchange_.exchangeBytes(
                    sendBuffers, &stats_.bytesSent, &stats_.bytesReceived);
            parseAndAddProcessCoefficients(received, processLocals, processTree_,
                                           topologyEpoch_, rank_);
            stats_.peakProcessBytes = std::max(stats_.peakProcessBytes,
                processMultipoles.bytesOwned() + processLocals.bytesOwned() +
                translatedProcessLocal.capacity() * sizeof(double));
        }

        if(localLeaf != FmmProcessTree::invalidIndex())
        {
            const std::size_t processOffset = processLocals.ensure(localLeaf);
            const std::size_t localOffset = localTree_.nodes()[0].localOffset;
            for(std::size_t i = 0; i < layout.coefficientCount(); ++i)
                localLocals_[localOffset + i] +=
                    processLocals.values[processOffset + i];
        }
    }
    stats_.processDownwardSeconds = elapsed(processDownStart);
    processMultipoles.release();
    processLocals.release();
    std::vector<double>().swap(translatedProcessLocal);

    const Clock::time_point interactionStart = Clock::now();
    letPlan_.execute(localTree_, positions, masses, cellIds, layout,
                     localMultipoles_, localLocals_, acceleration,
                     positiveKernelPotential, distributedOptions_.maxRemoteBytes,
                     stats_);
    if(stats_.peakRemoteBytes > distributedOptions_.maxRemoteBytes)
        throw UniversalError("DistributedFmmGravityCalculator::solve: LET memory budget exceeded");

    if(!localTree_.nodes().empty())
    {
        FmmDualTreeTraversal::run(localTree_, localTree_, positions, positions,
                                  masses, layout, localMultipoles_, localLocals_,
                                  true, options_.thetaCritical, acceleration,
                                  positiveKernelPotential, stats_);
    }
    stats_.interactionSeconds = elapsed(interactionStart);

    const Clock::time_point downwardStart = Clock::now();
    if(!localTree_.nodes().empty())
        FmmPasses::downward(localTree_, positions, layout, localLocals_,
                            acceleration, positiveKernelPotential);
    stats_.downwardSeconds = elapsed(downwardStart);

    stats_.bytesOwned = localTree_.bytesOwned() +
        localMultipoles_.capacity() * sizeof(double) +
        localLocals_.capacity() * sizeof(double) +
        rootDescriptors_.capacity() * sizeof(FmmRankRootDescriptor) +
        lastLocalTopologySignature_.capacity() * sizeof(std::uint64_t) +
        processTree_.bytesOwned() + processPlan_.bytesOwned() +
        letPlan_.bytesOwned() + processUpExchange_.bytesOwned() +
        processM2LExchange_.bytesOwned() + processDownExchange_.bytesOwned();

    if(options_.validateFinite)
    {
        bool localOutputFinite = true;
        std::size_t firstInvalid = 0;
        for(std::size_t i = 0; i < acceleration.size(); ++i)
        {
            if(!finiteVector(acceleration[i]) ||
               (positiveKernelPotential != nullptr &&
                !std::isfinite((*positiveKernelPotential)[i])))
            {
                localOutputFinite = false;
                firstInvalid = i;
                break;
            }
        }
        std::string message;
        if(!localOutputFinite)
            message = "DistributedFmmGravityCalculator::solve: non-finite output at local particle " +
                std::to_string(firstInvalid);
        collectiveRequire(localOutputFinite, message,
            "DistributedFmmGravityCalculator::solve output validation", comm_);
    }
    stats_.totalSeconds = elapsed(totalStart);
}

#endif // RICH_MPI
