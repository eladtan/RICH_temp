#include "3D/gravity/fmm/mpi/DistributedFmmGravityCalculator.hpp"

#ifdef RICH_MPI

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#ifdef __GLIBC__
#include <malloc.h>
#endif
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "3D/gravity/fmm/FmmDualTreeTraversal.hpp"
#include "3D/gravity/fmm/FmmKernels.hpp"
#include "3D/gravity/fmm/FmmPasses.hpp"
#include "3D/gravity/fmm/FmmTaylorExpansion.hpp"
#include "3D/gravity/fmm/mpi/FmmPatchDistributedSolver.hpp"
#include "3D/gravity/fmm/mpi/FmmGlobalDyadicLattice.hpp"
#include "3D/gravity/fmm/mpi/FmmPackets.hpp"
#include <MeshDecomposer3D/hilbert/HilbertOrder3D.hpp>
#include "misc/universal_error.hpp"

namespace
{
typedef std::chrono::steady_clock Clock;

// Each extra forced level costs roughly eight nodes per particle once leaves
// hold a single body. This only catches runaway depth; the real signal is the
// requested LET payload reported by the FMMLET diagnostic.
constexpr std::size_t kMaxNodesPerParticle = 64;

struct GravityOwnerParticle
{
    double position[3] = {0.0, 0.0, 0.0};
    double mass = 0.0;
    std::uint64_t cellId = 0;
    std::uint64_t originIndex = 0;
    std::uint64_t mortonKey = 0;
    int originRank = -1;
    int reserved = 0;
};

struct GravityOwnerResult
{
    double acceleration[3] = {0.0, 0.0, 0.0};
    double potential = 0.0;
    std::uint64_t originIndex = 0;
    int originRank = -1;
    int reserved = 0;
};

static_assert(std::is_trivially_copyable<GravityOwnerParticle>::value,
              "Gravity redistribution particles must be wire-copyable");
static_assert(std::is_trivially_copyable<GravityOwnerResult>::value,
              "Gravity redistribution results must be wire-copyable");

template<typename T>
std::vector<T> exchangeRecordsByRank(
    const std::vector<std::vector<T>>& sendByRank,
    const MPI_Comm& comm,
    const char* context)
{
    int size = 1;
    MPI_Comm_size(comm, &size);
    if(sendByRank.size() != static_cast<std::size_t>(size))
        throw UniversalError(std::string(context) + ": invalid peer table");
    std::vector<int> sendCounts(static_cast<std::size_t>(size), 0);
    std::vector<int> receiveCounts(static_cast<std::size_t>(size), 0);
    std::vector<int> sendDisplacements(static_cast<std::size_t>(size), 0);
    std::vector<int> receiveDisplacements(static_cast<std::size_t>(size), 0);
    std::size_t totalSendRecords = 0;
    bool localValid = true;
    for(int peer = 0; peer < size; ++peer)
    {
        const std::size_t count = sendByRank[static_cast<std::size_t>(peer)].size();
        localValid = localValid && count <= static_cast<std::size_t>(
            std::numeric_limits<int>::max()) / sizeof(T) &&
            totalSendRecords <= static_cast<std::size_t>(
                std::numeric_limits<int>::max()) / sizeof(T) - count;
        if(!localValid)
            break;
        sendCounts[static_cast<std::size_t>(peer)] = static_cast<int>(
            count * sizeof(T));
        sendDisplacements[static_cast<std::size_t>(peer)] = static_cast<int>(
            totalSendRecords * sizeof(T));
        totalSendRecords += count;
    }
    int localValidInt = localValid ? 1 : 0;
    int globalValidInt = 0;
    MPI_Allreduce(&localValidInt, &globalValidInt, 1, MPI_INT, MPI_LAND, comm);
    if(globalValidInt == 0)
        throw UniversalError(std::string(context) +
                             ": byte counts exceed MPI int");

    std::vector<T> sendFlat;
    sendFlat.reserve(totalSendRecords);
    for(const std::vector<T>& peer : sendByRank)
        sendFlat.insert(sendFlat.end(), peer.begin(), peer.end());
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT,
                 receiveCounts.data(), 1, MPI_INT, comm);
    std::size_t totalReceiveBytes = 0;
    for(int peer = 0; peer < size; ++peer)
    {
        const int bytes = receiveCounts[static_cast<std::size_t>(peer)];
        if(bytes < 0 || bytes % static_cast<int>(sizeof(T)) != 0 ||
           totalReceiveBytes > static_cast<std::size_t>(
               std::numeric_limits<int>::max()) -
               static_cast<std::size_t>(bytes))
            localValid = false;
        if(localValid)
        {
            receiveDisplacements[static_cast<std::size_t>(peer)] =
                static_cast<int>(totalReceiveBytes);
            totalReceiveBytes += static_cast<std::size_t>(bytes);
        }
    }
    localValidInt = localValid ? 1 : 0;
    MPI_Allreduce(&localValidInt, &globalValidInt, 1, MPI_INT, MPI_LAND, comm);
    if(globalValidInt == 0)
        throw UniversalError(std::string(context) +
                             ": invalid receive byte counts");
    std::vector<T> received(totalReceiveBytes / sizeof(T));
    MPI_Alltoallv(
        sendFlat.empty() ? nullptr : sendFlat.data(),
        sendCounts.data(), sendDisplacements.data(), MPI_BYTE,
        received.empty() ? nullptr : received.data(),
        receiveCounts.data(), receiveDisplacements.data(), MPI_BYTE, comm);
    return received;
}

double elapsed(const Clock::time_point& start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

bool finiteVector(const Vector3D& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

// Compares the node radius the admissibility test actually uses
// (sqrt(3) * halfSize) against the tightest radius that would still enclose
// every local particle.  The ratio is the geometric over-estimate introduced
// by cubification, root slack, and dyadic round-up.
void logRootGeometryDiagnostic(int rank,
                               const std::vector<Vector3D>& positions,
                               const FmmRootGeometry& root)
{
    static const bool enabled = [] {
        const char* value = std::getenv("RICH_FMM_GEOM_LOG");
        return value != nullptr && value[0] != '\0' &&
               !(value[0] == '0' && value[1] == '\0');
    }();
    if(!enabled || positions.empty() || !root.active)
        return;

    Vector3D lower = positions.front();
    Vector3D upper = positions.front();
    double tightRadiusSquared = 0.0;
    for(const Vector3D& point : positions)
    {
        lower.x = std::min(lower.x, point.x);
        lower.y = std::min(lower.y, point.y);
        lower.z = std::min(lower.z, point.z);
        upper.x = std::max(upper.x, point.x);
        upper.y = std::max(upper.y, point.y);
        upper.z = std::max(upper.z, point.z);
        const Vector3D delta = point - root.center;
        tightRadiusSquared = std::max(tightRadiusSquared,
            delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    }
    const double tightRadius = std::sqrt(tightRadiusSquared);
    const double boxRadius = std::sqrt(3.0) * root.halfSize;
    std::fprintf(stderr,
        "FMMGEOM rank=%d points=%zu extent=%.6e,%.6e,%.6e halfSize=%.6e "
        "boxRadius=%.6e tightRadius=%.6e inflation=%.3f\n",
        rank, positions.size(),
        upper.x - lower.x, upper.y - lower.y, upper.z - lower.z,
        root.halfSize, boxRadius, tightRadius,
        tightRadius > 0.0 ? boxRadius / tightRadius : -1.0);
    std::fflush(stderr);
}

void logLeafGeometryDiagnostic(int rank, const FmmTree& tree,
                               std::size_t particleCount,
                               double maxLeafHalfSize)
{
    static const bool enabled = [] {
        const char* value = std::getenv("RICH_FMM_GEOM_LOG");
        return value != nullptr && value[0] != '\0' &&
               !(value[0] == '0' && value[1] == '\0');
    }();
    if(!enabled)
        return;

    std::vector<double> halfSizes;
    std::vector<double> radii;
    std::vector<double> inflation;
    std::size_t radiusOver10 = 0;
    std::size_t radiusOver100 = 0;
    std::size_t radiusOver1000 = 0;
    for(const FmmNode& node : tree.nodes())
    {
        if(!node.isLeaf() || node.particleCount() == 0)
            continue;
        halfSizes.push_back(node.halfSize);
        radii.push_back(node.radius);
        const double boxRadius = std::sqrt(3.0) * node.halfSize;
        inflation.push_back(node.radius > 0.0 ?
            boxRadius / node.radius : std::numeric_limits<double>::infinity());
        radiusOver10 += node.radius > 10.0 ? 1 : 0;
        radiusOver100 += node.radius > 100.0 ? 1 : 0;
        radiusOver1000 += node.radius > 1000.0 ? 1 : 0;
    }
    if(halfSizes.empty())
        return;

    auto median = [](std::vector<double>& values) {
        const auto middle = values.begin() +
            static_cast<std::ptrdiff_t>(values.size() / 2);
        std::nth_element(values.begin(), middle, values.end());
        return *middle;
    };
    const double minHalfSize = *std::min_element(halfSizes.begin(), halfSizes.end());
    const double maxHalfSize = *std::max_element(halfSizes.begin(), halfSizes.end());
    const double minRadius = *std::min_element(radii.begin(), radii.end());
    const double maxRadius = *std::max_element(radii.begin(), radii.end());
    const double maxInflation = *std::max_element(inflation.begin(), inflation.end());
    const double medianHalfSize = median(halfSizes);
    const double medianRadius = median(radii);
    const double medianInflation = median(inflation);
    std::fprintf(stderr,
        "FMMLEAF rank=%d leaves=%zu nodes=%zu particles=%zu "
        "nodesPerParticle=%.2f maxLeafHalfSize=%.6e halfSizeMin=%.6e "
        "halfSizeMedian=%.6e halfSizeMax=%.6e radiusMin=%.6e "
        "radiusMedian=%.6e radiusMax=%.6e inflationMedian=%.3f "
        "inflationMax=%.3f radiusOver10=%zu radiusOver100=%zu "
        "radiusOver1000=%zu\n",
        rank, halfSizes.size(), tree.nodes().size(), particleCount,
        particleCount == 0 ? 0.0 :
            static_cast<double>(tree.nodes().size()) /
                static_cast<double>(particleCount),
        maxLeafHalfSize, minHalfSize, medianHalfSize, maxHalfSize,
        minRadius, medianRadius, maxRadius, medianInflation, maxInflation,
        radiusOver10, radiusOver100, radiusOver1000);
    std::fflush(stderr);
}

bool geometryLogEnabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv("RICH_FMM_GEOM_LOG");
        return value != nullptr && value[0] != '\0' &&
               !(value[0] == '0' && value[1] == '\0');
    }();
    return enabled;
}

// Integer lattice index of a point within the dyadic cell grid at `level`,
// measured from the global root cube. Packed as a Morton-ish key; only
// distinctness matters here, not ordering.
std::uint64_t dyadicCellKey(const Vector3D& point,
                            const FmmRootGeometry& globalRoot,
                            int level)
{
    const double cells = std::ldexp(1.0, level);
    const double cellSize = 2.0 * globalRoot.halfSize / cells;
    const Vector3D lower = globalRoot.lower();
    const auto axisIndex = [&](double value, double origin) {
        const double scaled = (value - origin) / cellSize;
        const double clamped = std::min(std::max(scaled, 0.0), cells - 1.0);
        return static_cast<std::uint64_t>(clamped);
    };
    const std::uint64_t ix = axisIndex(point.x, lower.x);
    const std::uint64_t iy = axisIndex(point.y, lower.y);
    const std::uint64_t iz = axisIndex(point.z, lower.z);
    // level <= 20 keeps each axis inside 21 bits.
    return (ix << 42) | (iy << 21) | iz;
}

bool sameRoot(const FmmRootGeometry& first, const FmmRootGeometry& second)
{
    return first.active == second.active &&
        (!first.active || (first.center.x == second.center.x &&
                           first.center.y == second.center.y &&
                           first.center.z == second.center.z &&
                           first.halfSize == second.halfSize &&
                           first.latticeId == second.latticeId &&
                           first.latticeCenterX == second.latticeCenterX &&
                           first.latticeCenterY == second.latticeCenterY &&
                           first.latticeCenterZ == second.latticeCenterZ &&
                           first.latticeHalfUnits == second.latticeHalfUnits &&
                            first.latticeAligned == second.latticeAligned));
}

std::size_t scaledPersistentCapacity(std::size_t leafCapacity,
                                     double factor,
                                     bool roundUp)
{
    const long double scaled = static_cast<long double>(leafCapacity) *
        static_cast<long double>(factor);
    const long double maximum = static_cast<long double>(
        std::numeric_limits<std::size_t>::max());
    if(!(scaled >= 0.0L) || scaled > maximum)
        throw UniversalError(
            "DistributedFmmGravityCalculator: persistent tree capacity overflow");
    const long double rounded = roundUp ? std::ceil(scaled) : std::floor(scaled);
    return static_cast<std::size_t>(rounded);
}

std::size_t persistentSplitCapacity(std::size_t leafCapacity,
                                    double factor)
{
    if(leafCapacity == std::numeric_limits<std::size_t>::max())
        throw UniversalError(
            "DistributedFmmGravityCalculator: leaf capacity cannot support hysteresis");
    return std::max(leafCapacity + 1,
                    scaledPersistentCapacity(leafCapacity, factor, true));
}

std::size_t persistentMergeCapacity(std::size_t leafCapacity,
                                    double factor)
{
    return std::min(leafCapacity - 1,
                    scaledPersistentCapacity(leafCapacity, factor, false));
}

void progressLetExchange(void* context)
{
    static_cast<FmmLetPlan*>(context)->progressExecute();
}

std::vector<std::uint64_t> structuralTopologySignature(const FmmTree& tree)
{
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
                  "Distributed FMM topology signatures require <=64-bit size_t");
    std::vector<std::uint64_t> signature;
    if(tree.nodes().size() > std::numeric_limits<std::size_t>::max() / 2)
        throw UniversalError(
            "DistributedFmmGravityCalculator: topology signature size overflow");
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
    static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
                  "Distributed FMM occupancy signatures require <=64-bit size_t");
    std::vector<std::uint64_t> signature;
    if(tree.leafCount() > std::numeric_limits<std::size_t>::max() / 2)
        throw UniversalError(
            "DistributedFmmGravityCalculator: occupancy signature size overflow");
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
    lastEffectiveMaxLeafHalfSize_(0.0),
    lastLocalTopologyHash_(0),
    topologyEpoch_(0),
    topologyRebuildCount_(0),
    processTopologyRebuildCount_(0),
    letTopologyRebuildCount_(0),
    solveCount_(0)
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
    else if(!(options_.maxLeafHalfSize >= 0.0) ||
            !std::isfinite(options_.maxLeafHalfSize))
    {
        localOptionsOk = false;
        localOptionsError =
            "DistributedFmmGravityCalculator: invalid maximum leaf half-size";
    }
    else if(distributedOptions_.maxLeafHalfSizeLevel < 0 ||
            distributedOptions_.maxLeafHalfSizeLevel > FMM_MAX_TREE_DEPTH)
    {
        localOptionsOk = false;
        localOptionsError =
            "DistributedFmmGravityCalculator: invalid maximum leaf half-size level";
    }
    else if(!(distributedOptions_.rootSlackFactor >= 1.0) ||
            !std::isfinite(distributedOptions_.rootSlackFactor))
    {
        localOptionsOk = false;
        localOptionsError = "DistributedFmmGravityCalculator: invalid root slack factor";
    }
    else if(!(distributedOptions_.hilbertGravityVolumeWeight >= 0.0) ||
            !(distributedOptions_.hilbertGravityVolumeWeight < 1.0) ||
            !std::isfinite(
                distributedOptions_.hilbertGravityVolumeWeight))
    {
        localOptionsOk = false;
        localOptionsError =
            "DistributedFmmGravityCalculator: invalid Hilbert gravity volume weight";
    }
    else if(distributedOptions_.maxRemoteBytes < 2)
    {
        localOptionsOk = false;
        localOptionsError = "DistributedFmmGravityCalculator: remote memory budget is too small";
    }
    else if(distributedOptions_.shareLetPayloadsWithinNode &&
            distributedOptions_.letPayloadHandlersPerNode == 0)
    {
        localOptionsOk = false;
        localOptionsError =
            "DistributedFmmGravityCalculator: node payload handler count is zero";
    }
    else if(distributedOptions_.quantizedLetParticlePayload &&
            !distributedOptions_.compactLetParticlePayload)
    {
        localOptionsOk = false;
        localOptionsError =
            "DistributedFmmGravityCalculator: quantized LET particles require compact particles";
    }
    else if(distributedOptions_.replicateProcessMultipoles &&
            distributedOptions_.enablePatchForest)
    {
        localOptionsOk = false;
        localOptionsError =
            "DistributedFmmGravityCalculator: replicated process multipoles require one tree per rank";
    }
    else if(distributedOptions_.persistentLocalTreeTopology &&
            (!(distributedOptions_.persistentLeafSplitFactor > 1.0) ||
             !std::isfinite(distributedOptions_.persistentLeafSplitFactor)))
    {
        localOptionsOk = false;
        localOptionsError =
            "DistributedFmmGravityCalculator: invalid persistent split factor";
    }
    else if(distributedOptions_.persistentLocalTreeTopology &&
            (!(distributedOptions_.persistentLeafMergeFactor >= 0.0) ||
             !(distributedOptions_.persistentLeafMergeFactor < 1.0) ||
             !std::isfinite(distributedOptions_.persistentLeafMergeFactor)))
    {
        localOptionsOk = false;
        localOptionsError =
            "DistributedFmmGravityCalculator: invalid persistent merge factor";
    }
    else if(distributedOptions_.minimumPatchLevel < 0 ||
            distributedOptions_.minimumPatchLevel > FMM_MAX_TREE_DEPTH ||
            distributedOptions_.maximumPatchLevel < 0 ||
            distributedOptions_.maximumPatchLevel > FMM_MAX_TREE_DEPTH ||
            distributedOptions_.minimumPatchLevel >
                distributedOptions_.maximumPatchLevel)
    {
        localOptionsOk = false;
        localOptionsError =
            "DistributedFmmGravityCalculator: invalid patch level bounds";
    }
    else if(distributedOptions_.maxLocalPatchCount == 0 ||
            distributedOptions_.maxTargetPatchesPerWave == 0)
    {
        localOptionsOk = false;
        localOptionsError =
            "DistributedFmmGravityCalculator: invalid patch count limits";
    }
    else if(distributedOptions_.enablePatchForest &&
            distributedOptions_.maxReplicatedDescriptorBytes <
                sizeof(FmmPatchRootDescriptor))
    {
        localOptionsOk = false;
        localOptionsError =
            "DistributedFmmGravityCalculator: replicated patch descriptor budget is too small";
    }
    if(localOptionsOk && distributedOptions_.persistentLocalTreeTopology)
    {
        try
        {
            (void) persistentSplitCapacity(options_.leafCapacity,
                distributedOptions_.persistentLeafSplitFactor);
            (void) persistentMergeCapacity(options_.leafCapacity,
                distributedOptions_.persistentLeafMergeFactor);
        }
        catch(const UniversalError&)
        {
            localOptionsOk = false;
            localOptionsError =
                "DistributedFmmGravityCalculator: invalid persistent tree capacities";
        }
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

    const double localDoubleOptions[6] = {
        options_.thetaCritical, distributedOptions_.rootSlackFactor,
        distributedOptions_.persistentLeafSplitFactor,
        distributedOptions_.persistentLeafMergeFactor,
        options_.maxLeafHalfSize,
        distributedOptions_.hilbertGravityVolumeWeight};
    double minimumDoubleOptions[6] = {};
    double maximumDoubleOptions[6] = {};
    MPI_Allreduce(localDoubleOptions, minimumDoubleOptions, 6,
                  MPI_DOUBLE, MPI_MIN, comm_);
    MPI_Allreduce(localDoubleOptions, maximumDoubleOptions, 6,
                  MPI_DOUBLE, MPI_MAX, comm_);

    const unsigned long long localIntegerOptions[32] = {
        static_cast<unsigned long long>(options_.expansionOrder),
        static_cast<unsigned long long>(options_.leafCapacity),
        static_cast<unsigned long long>(options_.maxDepth),
        options_.computePotential ? 1ull : 0ull,
        options_.validateFinite ? 1ull : 0ull,
        static_cast<unsigned long long>(options_.maxOperatorCacheBytes),
        static_cast<unsigned long long>(distributedOptions_.maxRemoteBytes),
        distributedOptions_.rebuildTopologyEverySolve ? 1ull : 0ull,
        distributedOptions_.reuseInteractionPlansAcrossLeafCountChanges ? 1ull : 0ull,
        distributedOptions_.persistentLocalTreeTopology ? 1ull : 0ull,
        static_cast<unsigned long long>(
            distributedOptions_.maxLeafHalfSizeLevel),
        distributedOptions_.enableLeafM2P ? 1ull : 0ull,
        static_cast<unsigned long long>(distributedOptions_.maxLetWaveBytes),
        distributedOptions_.enablePatchForest ? 1ull : 0ull,
        static_cast<unsigned long long>(distributedOptions_.minimumPatchLevel),
        static_cast<unsigned long long>(distributedOptions_.maximumPatchLevel),
        static_cast<unsigned long long>(
            distributedOptions_.targetParticlesPerPatch),
        static_cast<unsigned long long>(distributedOptions_.maxLocalPatchCount),
        static_cast<unsigned long long>(
            distributedOptions_.maxTargetPatchesPerWave),
        distributedOptions_.useLocalPatchLet ? 1ull : 0ull,
        static_cast<unsigned long long>(
            distributedOptions_.maxReplicatedDescriptorBytes),
        distributedOptions_.shareLetPayloadsWithinNode ? 1ull : 0ull,
        static_cast<unsigned long long>(
            distributedOptions_.letPayloadHandlersPerNode),
        distributedOptions_.spatiallyRedistributeForGravity ? 1ull : 0ull,
        distributedOptions_.replicateProcessMultipoles ? 1ull : 0ull,
        distributedOptions_.reuseBoundedLetWavesAcrossLeafCountChanges ?
            1ull : 0ull,
        static_cast<unsigned long long>(
            distributedOptions_.directErrorSampleCount),
        static_cast<unsigned long long>(
            distributedOptions_.directErrorSampleSolve),
        distributedOptions_.useHilbertGravityRedistribution ? 1ull : 0ull,
        distributedOptions_.compactLetParticlePayload ? 1ull : 0ull,
        distributedOptions_.compactLetMultipolePayload ? 1ull : 0ull,
        distributedOptions_.quantizedLetParticlePayload ? 1ull : 0ull};
    unsigned long long minimumIntegerOptions[32] = {};
    unsigned long long maximumIntegerOptions[32] = {};
    MPI_Allreduce(localIntegerOptions, minimumIntegerOptions, 32,
                  MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm_);
    MPI_Allreduce(localIntegerOptions, maximumIntegerOptions, 32,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm_);

    bool optionsMatch = true;
    for(int i = 0; i < 6; ++i)
        optionsMatch = optionsMatch &&
            minimumDoubleOptions[i] == maximumDoubleOptions[i];
    for(int i = 0; i < 32; ++i)
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
    patchSolver_.reset();
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

// Collective on comm_: every rank must call this. Reports, for a range of
// candidate patch levels, how many process-tree leaves a patch forest would
// create and what the replicated descriptor storage would cost. This decides
// whether a replicated patch process tree is viable before any is built.
void DistributedFmmGravityCalculator::logPatchCountSurvey(
    const std::vector<Vector3D>& positions,
    const Vector3D& domainLower,
    const Vector3D& domainUpper) const
{
    const FmmRootGeometry globalRoot =
        FmmRootGeometry::fromDomain(domainLower, domainUpper, true);

    constexpr int kFirstLevel = 5;
    constexpr int kLastLevel = 10;
    constexpr int kLevelCount = kLastLevel - kFirstLevel + 1;

    unsigned long long localCounts[kLevelCount] = {};
    std::unordered_set<std::uint64_t> cells;
    for(int level = kFirstLevel; level <= kLastLevel; ++level)
    {
        cells.clear();
        for(const Vector3D& point : positions)
            cells.insert(dyadicCellKey(point, globalRoot, level));
        localCounts[level - kFirstLevel] =
            static_cast<unsigned long long>(cells.size());
    }

    unsigned long long totalCounts[kLevelCount] = {};
    unsigned long long maxCounts[kLevelCount] = {};
    MPI_Allreduce(localCounts, totalCounts, kLevelCount,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_);
    MPI_Allreduce(localCounts, maxCounts, kLevelCount,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm_);

    if(rank_ != 0)
        return;

    // Ranks per node is not known here; report per-rank bytes and let the
    // reader multiply by the actual packing.
    const std::size_t descriptorBytes = 128;
    for(int level = kFirstLevel; level <= kLastLevel; ++level)
    {
        const int index = level - kFirstLevel;
        const double halfSize = std::ldexp(globalRoot.halfSize, -level);
        std::fprintf(stderr,
            "FMMPATCHSURVEY level=%d patchHalfSize=%.6e totalPatches=%llu "
            "maxLocalPatches=%llu descriptorMiBPerRank=%.2f "
            "globalRootHalfSize=%.6e\n",
            level, halfSize, totalCounts[index], maxCounts[index],
            static_cast<double>(totalCounts[index]) *
                static_cast<double>(descriptorBytes) / 1048576.0,
            globalRoot.halfSize);
    }
    std::fflush(stderr);
}

// The bound has to be one global length rather than a per-rank one, because
// admissibility compares local nodes against descriptors from every other rank.
double DistributedFmmGravityCalculator::effectiveMaxLeafHalfSize(
    const Vector3D& domainLower,
    const Vector3D& domainUpper) const
{
    if(options_.maxLeafHalfSize > 0.0)
        return options_.maxLeafHalfSize;
    if(distributedOptions_.maxLeafHalfSizeLevel <= 0)
        return 0.0;
    const FmmRootGeometry globalRoot =
        FmmRootGeometry::fromDomain(domainLower, domainUpper, true);
    const double bound = std::ldexp(globalRoot.halfSize,
                                    -distributedOptions_.maxLeafHalfSizeLevel);
    return std::isfinite(bound) && bound > 0.0 ? bound : 0.0;
}

DistributedFmmGravityCalculator::LocalTopologyChange
DistributedFmmGravityCalculator::prepareLocalTree(
    const std::vector<Vector3D>& positions,
    const Vector3D& domainLower,
    const Vector3D& domainUpper)
{
    FmmGravityOptions treeOptions = options_;
    treeOptions.maxLeafHalfSize =
        effectiveMaxLeafHalfSize(domainLower, domainUpper);

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
            FmmRootGeometry::containingPointsOnDyadicLattice(
                positions, domainLower, domainUpper,
                distributedOptions_.rootSlackFactor, options_.maxDepth);
    }

    LocalTopologyChange change;
    // A changed bound invalidates any retained topology, since the split
    // decision that produced it no longer holds.
    change.rootGeometryChanged =
        !rootInitialized_ || !sameRoot(localRoot_, nextRoot) ||
        treeOptions.maxLeafHalfSize != lastEffectiveMaxLeafHalfSize_;
    if(distributedOptions_.persistentLocalTreeTopology)
    {
        FmmPersistentTreeStats persistentStats;
        const bool initializeFromScratch =
            change.rootGeometryChanged || localTree_.nodes().empty();
        localTree_.buildPersistent(
            positions, nextRoot, treeOptions,
            persistentSplitCapacity(options_.leafCapacity,
                distributedOptions_.persistentLeafSplitFactor),
            persistentMergeCapacity(options_.leafCapacity,
                distributedOptions_.persistentLeafMergeFactor),
            initializeFromScratch, persistentStats);
        change.persistentTreeRefit =
            !initializeFromScratch && !positions.empty();
        change.persistentLeafSplits = persistentStats.leafSplits;
        change.persistentSubtreeMerges = persistentStats.subtreeMerges;
        change.persistentEmptyLeaves = persistentStats.emptyLeaves;
    }
    else
    {
        localTree_.build(positions, nextRoot, treeOptions);
    }
    // Only the oversized-domain ranks can trip this, so throwing would leave
    // the rest of the communicator waiting in the next collective. Abort with a
    // printed reason instead, matching the convention in FmmPeerExchange.
    if(treeOptions.maxLeafHalfSize > 0.0 && !positions.empty() &&
       localTree_.nodes().size() / positions.size() > kMaxNodesPerParticle)
    {
        std::fprintf(stderr,
            "DistributedFmmGravityCalculator abort on MPI rank %d: leaf "
            "half-size bound produced an excessive local tree; raise "
            "maxLeafHalfSizeLevel\n"
            "nodes=%zu particles=%zu nodesPerParticle=%.2f "
            "maxLeafHalfSize=%.6e limit=%zu\n",
            rank_, localTree_.nodes().size(), positions.size(),
            static_cast<double>(localTree_.nodes().size()) /
                static_cast<double>(positions.size()),
            treeOptions.maxLeafHalfSize, kMaxNodesPerParticle);
        std::fflush(stderr);
        MPI_Abort(comm_, 94);
    }
    const std::uint64_t hash = localTree_.topologyHash();
    std::vector<std::uint64_t> structuralSignature =
        structuralTopologySignature(localTree_);
    std::vector<std::uint64_t> occupancySignature =
        leafOccupancySignature(localTree_);
    change.leafTopologyChanged =
        structuralSignature != lastLocalStructuralSignature_;
    change.leafOccupancyChanged =
        occupancySignature != lastLocalOccupancySignature_;
    change.countOnlyLeafChange = !change.rootGeometryChanged &&
        !change.leafTopologyChanged && change.leafOccupancyChanged;
    localRoot_ = nextRoot;
    lastEffectiveMaxLeafHalfSize_ = treeOptions.maxLeafHalfSize;
    lastLocalTopologyHash_ = hash;
    lastLocalStructuralSignature_.swap(structuralSignature);
    lastLocalOccupancySignature_.swap(occupancySignature);
    rootInitialized_ = true;
    if(change.rootGeometryChanged)
    {
        logRootGeometryDiagnostic(rank_, positions, localRoot_);
        logLeafGeometryDiagnostic(rank_, localTree_, positions.size(),
                                  treeOptions.maxLeafHalfSize);
    }
    return change;
}

FmmPatchRootDescriptor DistributedFmmGravityCalculator::localRootDescriptor() const
{
    FmmPatchRootDescriptor result;
    result.ownerRank = rank_;
    result.patchId = FMM_COMPAT_PATCH_ID;
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
        result.radius = root.radius;
        result.particleCount = static_cast<std::uint64_t>(root.particleCount());
        result.latticeId = root.latticeId;
        result.latticeCenter[0] = root.latticeCenterX;
        result.latticeCenter[1] = root.latticeCenterY;
        result.latticeCenter[2] = root.latticeCenterZ;
        result.latticeHalfUnits = root.latticeHalfUnits;
        result.rootLeaf = root.isLeaf() ? 1 : 0;
        result.childMask = static_cast<int>(root.childMask);
    }
    return result;
}

void DistributedFmmGravityCalculator::rebuildTopology(
    const std::vector<Vector3D>& positions,
    bool rebuildProcessTopology)
{
    const Clock::time_point topologyStart = Clock::now();
    if(topologyEpoch_ == std::numeric_limits<std::uint64_t>::max() ||
       topologyRebuildCount_ == std::numeric_limits<std::uint64_t>::max() ||
       letTopologyRebuildCount_ == std::numeric_limits<std::uint64_t>::max() ||
       (rebuildProcessTopology && processTopologyRebuildCount_ ==
            std::numeric_limits<std::uint64_t>::max()))
        throw UniversalError(
            "DistributedFmmGravityCalculator::rebuildTopology: topology epoch overflow");
    ++topologyEpoch_;
    ++topologyRebuildCount_;
    ++letTopologyRebuildCount_;
    if(rebuildProcessTopology)
        ++processTopologyRebuildCount_;

    const Clock::time_point descriptorStart = Clock::now();
    const FmmPatchRootDescriptor local = localRootDescriptor();
    rootDescriptors_.resize(static_cast<std::size_t>(size_));
    MPI_Allgather(&local, static_cast<int>(sizeof(FmmPatchRootDescriptor)), MPI_BYTE,
                  rootDescriptors_.data(),
                  static_cast<int>(sizeof(FmmPatchRootDescriptor)), MPI_BYTE,
                  comm_);
    stats_.rootDescriptorExchangeSeconds = elapsed(descriptorStart);

    stats_.processTopologyRebuilt = rebuildProcessTopology;
    stats_.letTopologyRebuilt = true;
    if(rebuildProcessTopology)
    {
        const Clock::time_point processStart = Clock::now();
        processTree_.build(rootDescriptors_);
        processPlan_ = FmmProcessTraversal::build(
            processTree_, options_.thetaCritical, topologyEpoch_, rank_, comm_);

        std::set<int> upPeers;
        std::set<int> downPeers;
        for(std::size_t i = 0; i < processTree_.nodes().size(); ++i)
        {
            const FmmProcessNode& node = processTree_.nodes()[i];
            if(node.owner == rank_ &&
               node.parent != FmmProcessTree::invalidIndex())
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
        const bool upReset = processUpExchange_.resetIfChanged(
            comm_, std::vector<int>(upPeers.begin(), upPeers.end()));
        const bool m2lReset = processM2LExchange_.resetIfChanged(comm_, m2lPeers);
        const bool downReset = processDownExchange_.resetIfChanged(
            comm_, std::vector<int>(downPeers.begin(), downPeers.end()));
        stats_.processCommunicatorsReused =
            !upReset && !m2lReset && !downReset;
        stats_.processTopologySeconds = elapsed(processStart);
    }
    else
    {
        stats_.processCommunicatorsReused = true;
    }

    letPlan_.build(localTree_, positions, rootDescriptors_, processPlan_,
                   options_.thetaCritical, topologyEpoch_, comm_,
                   !rebuildProcessTopology,
                   distributedOptions_.enableLeafM2P,
                   distributedOptions_.compactLetParticlePayload,
                   distributedOptions_.quantizedLetParticlePayload,
                   distributedOptions_.compactLetMultipolePayload,
                   distributedOptions_.maxLetWaveBytes,
                   fmmTaylorCoefficientCount(options_.expansionOrder), stats_);
    stats_.topologyRebuildSeconds = elapsed(topologyStart);
}

void DistributedFmmGravityCalculator::solveRedistributed(
    const std::vector<Vector3D>& positions,
    const std::vector<double>& masses,
    const std::vector<std::uint64_t>& cellIds,
    const Vector3D& domainLower,
    const Vector3D& domainUpper,
    std::vector<Vector3D>& acceleration,
    std::vector<double>* positiveKernelPotential)
{
    const Clock::time_point fullStart = Clock::now();
    const Clock::time_point redistributionStart = Clock::now();
    const FmmGlobalDyadicLattice lattice =
        FmmGlobalDyadicLattice::fromDomain(domainLower, domainUpper);
    // A level-20 octant path is a 60-bit Morton key. It is fine enough that
    // equal-key runs are negligible for a moving Voronoi mesh, while retaining
    // a common, deterministic spatial order on every rank.
    const int keyLevel = std::min(20, FMM_MAX_TREE_DEPTH);
    HilbertCurve3D<Vector3D> hilbertCurve;
    const Vector3D domainExtent = domainUpper - domainLower;
    std::vector<GravityOwnerParticle> localParticles;
    localParticles.reserve(positions.size());
    for(std::size_t index = 0; index < positions.size(); ++index)
    {
        GravityOwnerParticle particle;
        particle.position[0] = positions[index].x;
        particle.position[1] = positions[index].y;
        particle.position[2] = positions[index].z;
        particle.mass = masses[index];
        particle.cellId = cellIds[index];
        particle.originIndex = static_cast<std::uint64_t>(index);
        if(distributedOptions_.useHilbertGravityRedistribution)
        {
            const Vector3D unitPoint(
                std::max(0.0, std::min(1.0,
                    (positions[index].x - domainLower.x) / domainExtent.x)),
                std::max(0.0, std::min(1.0,
                    (positions[index].y - domainLower.y) / domainExtent.y)),
                std::max(0.0, std::min(1.0,
                    (positions[index].z - domainLower.z) / domainExtent.z)));
            particle.mortonKey = hilbertCurve.Hilbert3D_xyz2d(
                unitPoint, keyLevel);
        }
        else
        {
            particle.mortonKey = lattice.patchIdAtLevel(
                positions[index], keyLevel);
        }
        particle.originRank = rank_;
        localParticles.push_back(particle);
    }
    std::sort(localParticles.begin(), localParticles.end(),
        [](const GravityOwnerParticle& first,
           const GravityOwnerParticle& second) {
            return std::tie(first.mortonKey, first.cellId,
                            first.originRank, first.originIndex) <
                   std::tie(second.mortonKey, second.cellId,
                            second.originRank, second.originIndex);
        });

    // Keep gravity ownership stable across warm solves. Re-sampling every
    // moving-mesh step shifted a few range boundaries, which changed one rank's
    // retained root and needlessly invalidated the global LET plan. The first
    // solve establishes balanced splitters; later solves preserve them so small
    // particle motion is handled by the persistent local trees.
    if(gravityRedistributionSplitters_.size() !=
       static_cast<std::size_t>(std::max(0, size_ - 1)))
    {
        const std::size_t localSampleCount = std::min(
            localParticles.size(),
            static_cast<std::size_t>(std::max(0, size_ - 1)));
        std::vector<std::uint64_t> localSamples(localSampleCount);
        for(std::size_t sample = 0; sample < localSampleCount; ++sample)
        {
            const std::size_t index =
                ((sample + 1) * localParticles.size()) /
                (localSampleCount + 1);
            localSamples[sample] = localParticles[
                std::min(index, localParticles.size() - 1)].mortonKey;
        }
        const int localSampleCountInt = static_cast<int>(localSampleCount);
        std::vector<int> sampleCounts;
        if(rank_ == 0)
            sampleCounts.resize(static_cast<std::size_t>(size_));
        MPI_Gather(&localSampleCountInt, 1, MPI_INT,
                   rank_ == 0 ? sampleCounts.data() : nullptr,
                   1, MPI_INT, 0, comm_);
        std::vector<int> sampleDisplacements;
        std::vector<std::uint64_t> gatheredSamples;
        if(rank_ == 0)
        {
            sampleDisplacements.resize(static_cast<std::size_t>(size_));
            std::size_t totalSamples = 0;
            for(int peer = 0; peer < size_; ++peer)
            {
                sampleDisplacements[static_cast<std::size_t>(peer)] =
                    static_cast<int>(totalSamples);
                totalSamples += static_cast<std::size_t>(
                    sampleCounts[static_cast<std::size_t>(peer)]);
            }
            if(totalSamples > static_cast<std::size_t>(
                   std::numeric_limits<int>::max()))
                throw UniversalError(
                    "DistributedFmmGravityCalculator: too many Morton samples");
            gatheredSamples.resize(totalSamples);
        }
        MPI_Gatherv(
            localSamples.empty() ? nullptr : localSamples.data(),
            localSampleCountInt, MPI_UNSIGNED_LONG_LONG,
            rank_ == 0 && !gatheredSamples.empty() ?
                gatheredSamples.data() : nullptr,
            rank_ == 0 ? sampleCounts.data() : nullptr,
            rank_ == 0 ? sampleDisplacements.data() : nullptr,
            MPI_UNSIGNED_LONG_LONG, 0, comm_);

        gravityRedistributionSplitters_.assign(
            static_cast<std::size_t>(std::max(0, size_ - 1)),
            std::numeric_limits<std::uint64_t>::max());
        if(rank_ == 0 && !gatheredSamples.empty())
        {
            std::sort(gatheredSamples.begin(), gatheredSamples.end());
            const double volumeWeight =
                distributedOptions_.useHilbertGravityRedistribution ?
                distributedOptions_.hilbertGravityVolumeWeight : 0.0;
            const long double maximumHilbertKey = static_cast<long double>(
                (UINT64_C(1) << (3 * keyLevel)) - UINT64_C(1));
            // Never give a rank fewer than one eighth of the equal-particle
            // sample occupancy. This prevents large empty key gaps from
            // creating idle ranks while still allowing an 8x shift of rank
            // capacity toward the sparse atmosphere.
            const std::size_t minimumSamplesPerRank = std::max<std::size_t>(
                1, gatheredSamples.size() /
                   (static_cast<std::size_t>(size_) * 8));
            std::size_t previousIndex = 0;
            for(int boundary = 1; boundary < size_; ++boundary)
            {
                std::size_t index = 0;
                if(volumeWeight == 0.0)
                {
                    index = std::min(gatheredSamples.size() - 1,
                        static_cast<std::size_t>(boundary) *
                            gatheredSamples.size() /
                            static_cast<std::size_t>(size_));
                }
                else
                {
                    const long double target =
                        static_cast<long double>(boundary) /
                        static_cast<long double>(size_);
                    std::size_t lower = 0;
                    std::size_t upper = gatheredSamples.size();
                    while(lower < upper)
                    {
                        const std::size_t middle = lower + (upper - lower) / 2;
                        const long double particleFraction =
                            static_cast<long double>(middle + 1) /
                            static_cast<long double>(gatheredSamples.size());
                        const long double volumeFraction =
                            static_cast<long double>(gatheredSamples[middle]) /
                            maximumHilbertKey;
                        const long double score =
                            (1.0L - volumeWeight) * particleFraction +
                            volumeWeight * volumeFraction;
                        if(score < target)
                            lower = middle + 1;
                        else
                            upper = middle;
                    }
                    index = std::min(lower, gatheredSamples.size() - 1);
                    const std::size_t minimumIndex = boundary == 1 ?
                        minimumSamplesPerRank - 1 :
                        previousIndex + minimumSamplesPerRank;
                    const std::size_t remainingRanks =
                        static_cast<std::size_t>(size_ - boundary);
                    const std::size_t maximumIndex =
                        gatheredSamples.size() -
                        remainingRanks * minimumSamplesPerRank - 1;
                    index = std::max(minimumIndex,
                                     std::min(index, maximumIndex));
                }
                gravityRedistributionSplitters_[
                    static_cast<std::size_t>(boundary - 1)] =
                    gatheredSamples[index];
                previousIndex = index;
            }
        }
        if(!gravityRedistributionSplitters_.empty())
            MPI_Bcast(gravityRedistributionSplitters_.data(),
                      static_cast<int>(gravityRedistributionSplitters_.size()),
                      MPI_UNSIGNED_LONG_LONG, 0, comm_);
    }
    const std::vector<std::uint64_t>& splitters =
        gravityRedistributionSplitters_;

    std::vector<std::vector<GravityOwnerParticle>> sendParticles(
        static_cast<std::size_t>(size_));
    std::vector<std::size_t> destinationCounts(
        static_cast<std::size_t>(size_), 0);
    for(const GravityOwnerParticle& particle : localParticles)
    {
        const int destination = static_cast<int>(std::upper_bound(
            splitters.begin(), splitters.end(), particle.mortonKey) -
            splitters.begin());
        ++destinationCounts[static_cast<std::size_t>(destination)];
    }
    for(int destination = 0; destination < size_; ++destination)
        sendParticles[static_cast<std::size_t>(destination)].reserve(
            destinationCounts[static_cast<std::size_t>(destination)]);
    for(const GravityOwnerParticle& particle : localParticles)
    {
        const int destination = static_cast<int>(std::upper_bound(
            splitters.begin(), splitters.end(), particle.mortonKey) -
            splitters.begin());
        sendParticles[static_cast<std::size_t>(destination)].push_back(
            particle);
    }
    localParticles = exchangeRecordsByRank(
        sendParticles, comm_,
        "DistributedFmmGravityCalculator gravity-owner exchange");
    std::sort(localParticles.begin(), localParticles.end(),
        [](const GravityOwnerParticle& first,
           const GravityOwnerParticle& second) {
            return std::tie(first.mortonKey, first.cellId,
                            first.originRank, first.originIndex) <
                   std::tie(second.mortonKey, second.cellId,
                            second.originRank, second.originIndex);
        });

    std::vector<Vector3D> ownedPositions;
    std::vector<double> ownedMasses;
    std::vector<std::uint64_t> ownedCellIds;
    ownedPositions.reserve(localParticles.size());
    ownedMasses.reserve(localParticles.size());
    ownedCellIds.reserve(localParticles.size());
    for(const GravityOwnerParticle& particle : localParticles)
    {
        ownedPositions.push_back(Vector3D(
            particle.position[0], particle.position[1], particle.position[2]));
        ownedMasses.push_back(particle.mass);
        ownedCellIds.push_back(particle.cellId);
    }
    const double beforeSolveRedistributionSeconds = elapsed(
        redistributionStart);
    std::vector<Vector3D> ownedAcceleration;
    std::vector<double> ownedPotential;
    solveOwned(ownedPositions, ownedMasses, ownedCellIds,
               domainLower, domainUpper, ownedAcceleration,
               positiveKernelPotential == nullptr ? nullptr : &ownedPotential);

    const Clock::time_point returnStart = Clock::now();
    std::vector<std::vector<GravityOwnerResult>> sendResults(
        static_cast<std::size_t>(size_));
    for(std::size_t index = 0; index < localParticles.size(); ++index)
    {
        const GravityOwnerParticle& particle = localParticles[index];
        GravityOwnerResult result;
        result.acceleration[0] = ownedAcceleration[index].x;
        result.acceleration[1] = ownedAcceleration[index].y;
        result.acceleration[2] = ownedAcceleration[index].z;
        result.potential = positiveKernelPotential == nullptr ? 0.0 :
            ownedPotential[index];
        result.originIndex = particle.originIndex;
        result.originRank = particle.originRank;
        sendResults[static_cast<std::size_t>(particle.originRank)].push_back(
            result);
    }
    const std::vector<GravityOwnerResult> returned = exchangeRecordsByRank(
        sendResults, comm_,
        "DistributedFmmGravityCalculator gravity-result exchange");
    acceleration.assign(positions.size(), Vector3D());
    if(positiveKernelPotential != nullptr)
        positiveKernelPotential->assign(positions.size(), 0.0);
    std::vector<unsigned char> filled(positions.size(), 0);
    for(const GravityOwnerResult& result : returned)
    {
        if(result.originRank != rank_ ||
           result.originIndex >= positions.size() ||
           filled[static_cast<std::size_t>(result.originIndex)] != 0)
            throw UniversalError(
                "DistributedFmmGravityCalculator: invalid returned gravity result");
        const std::size_t index = static_cast<std::size_t>(
            result.originIndex);
        acceleration[index] = Vector3D(
            result.acceleration[0], result.acceleration[1],
            result.acceleration[2]);
        if(positiveKernelPotential != nullptr)
            (*positiveKernelPotential)[index] = result.potential;
        filled[index] = 1;
    }
    if(returned.size() != positions.size() ||
       std::find(filled.begin(), filled.end(), 0) != filled.end())
        throw UniversalError(
            "DistributedFmmGravityCalculator: missing returned gravity result");
    stats_.gravityRedistributionSeconds =
        beforeSolveRedistributionSeconds + elapsed(returnStart);
    stats_.totalSeconds = elapsed(fullStart);
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
    const double localDomain[6] = {
        domainLower.x, domainLower.y, domainLower.z,
        domainUpper.x, domainUpper.y, domainUpper.z};

    // One MIN reduction performs the old validity AND, domain MIN, and domain
    // MAX (encoded as MIN of the negated values). Invalid ranks contribute
    // zero domain values, which are ignored because validity is checked first.
    double localValidation[13] = {};
    localValidation[0] = localInputsValid ? 1.0 : 0.0;
    for(int i = 0; i < 6; ++i)
    {
        const double value = localInputsValid ? localDomain[i] : 0.0;
        localValidation[1 + i] = value;
        localValidation[7 + i] = -value;
    }
    double globalValidation[13] = {};
    MPI_Allreduce(localValidation, globalValidation, 13, MPI_DOUBLE, MPI_MIN,
                  comm_);
    if(globalValidation[0] != 1.0)
    {
        if(!localInputsValid)
            throw UniversalError(localInputError.empty() ?
                "DistributedFmmGravityCalculator::solve input validation" :
                localInputError);
        throw UniversalError(
            "DistributedFmmGravityCalculator::solve input validation failed on another MPI rank");
    }
    bool commonDomain = true;
    for(int i = 0; i < 6; ++i)
        commonDomain = commonDomain &&
            globalValidation[1 + i] == -globalValidation[7 + i];
    if(!commonDomain)
        throw UniversalError(
            "DistributedFmmGravityCalculator::solve: domain bounds differ across MPI ranks");

    ++solveCount_;

    if(distributedOptions_.spatiallyRedistributeForGravity)
        solveRedistributed(positions, masses, cellIds, domainLower,
                           domainUpper, acceleration,
                           positiveKernelPotential);
    else
        solveOwned(positions, masses, cellIds, domainLower, domainUpper,
                   acceleration, positiveKernelPotential);

    if(distributedOptions_.directErrorSampleCount != 0 &&
       solveCount_ == distributedOptions_.directErrorSampleSolve)
        sampleDirectAccelerationError(positions, masses, acceleration);
}

void DistributedFmmGravityCalculator::sampleDirectAccelerationError(
    const std::vector<Vector3D>& positions,
    const std::vector<double>& masses,
    const std::vector<Vector3D>& acceleration)
{
    const Clock::time_point sampleStart = Clock::now();
    const unsigned long long localCount =
        static_cast<unsigned long long>(positions.size());
    unsigned long long globalCount = 0;
    unsigned long long globalOffset = 0;
    MPI_Allreduce(&localCount, &globalCount, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, comm_);
    MPI_Exscan(&localCount, &globalOffset, 1, MPI_UNSIGNED_LONG_LONG,
               MPI_SUM, comm_);
    if(rank_ == 0)
        globalOffset = 0;

    const std::size_t sampleCount = static_cast<std::size_t>(std::min(
        static_cast<unsigned long long>(
            distributedOptions_.directErrorSampleCount), globalCount));
    if(sampleCount == 0)
        return;

    // One sample per equal-population stratum prevents duplicates while the
    // fixed SplitMix64 seed keeps production comparisons reproducible.
    const std::uint64_t sampleSeed = UINT64_C(0x6a09e667f3bcc909);
    const auto splitMix64 = [](std::uint64_t value) {
        value += UINT64_C(0x9e3779b97f4a7c15);
        value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
        return value ^ (value >> 31);
    };
    std::vector<unsigned long long> targetIndices(sampleCount, 0);
    std::vector<double> targetPositions(3 * sampleCount, 0.0);
    std::vector<double> targetAccelerations(3 * sampleCount, 0.0);
    std::vector<int> targetOwnerCounts(sampleCount, 0);
    for(std::size_t sample = 0; sample < sampleCount; ++sample)
    {
        const unsigned long long begin =
            (static_cast<unsigned long long>(sample) * globalCount) /
            static_cast<unsigned long long>(sampleCount);
        const unsigned long long end =
            (static_cast<unsigned long long>(sample + 1) * globalCount) /
            static_cast<unsigned long long>(sampleCount);
        const unsigned long long width = end - begin;
        const unsigned long long globalIndex = begin +
            splitMix64(sampleSeed + static_cast<std::uint64_t>(sample)) % width;
        targetIndices[sample] = globalIndex;
        if(globalIndex >= globalOffset &&
           globalIndex - globalOffset < localCount)
        {
            const std::size_t localIndex = static_cast<std::size_t>(
                globalIndex - globalOffset);
            targetPositions[3 * sample] = positions[localIndex].x;
            targetPositions[3 * sample + 1] = positions[localIndex].y;
            targetPositions[3 * sample + 2] = positions[localIndex].z;
            targetAccelerations[3 * sample] = acceleration[localIndex].x;
            targetAccelerations[3 * sample + 1] = acceleration[localIndex].y;
            targetAccelerations[3 * sample + 2] = acceleration[localIndex].z;
            targetOwnerCounts[sample] = 1;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, targetPositions.data(),
                  static_cast<int>(targetPositions.size()), MPI_DOUBLE,
                  MPI_SUM, comm_);
    MPI_Allreduce(MPI_IN_PLACE, targetAccelerations.data(),
                  static_cast<int>(targetAccelerations.size()), MPI_DOUBLE,
                  MPI_SUM, comm_);
    MPI_Allreduce(MPI_IN_PLACE, targetOwnerCounts.data(),
                  static_cast<int>(targetOwnerCounts.size()), MPI_INT,
                  MPI_SUM, comm_);
    if(std::find_if(targetOwnerCounts.begin(), targetOwnerCounts.end(),
           [](int count) { return count != 1; }) != targetOwnerCounts.end())
        throw UniversalError(
            "DistributedFmmGravityCalculator: sampled target owner mismatch");

    // Compute distances in double (matching the FMM kernels), but use long
    // double accumulators to make summation roundoff negligible compared with
    // the approximation error under test.
    std::vector<long double> direct(3 * sampleCount, 0.0L);
    for(std::size_t sample = 0; sample < sampleCount; ++sample)
    {
        const double targetX = targetPositions[3 * sample];
        const double targetY = targetPositions[3 * sample + 1];
        const double targetZ = targetPositions[3 * sample + 2];
        for(std::size_t source = 0; source < positions.size(); ++source)
        {
            if(targetIndices[sample] ==
               globalOffset + static_cast<unsigned long long>(source))
                continue;
            const double dx = targetX - positions[source].x;
            const double dy = targetY - positions[source].y;
            const double dz = targetZ - positions[source].z;
            const double r2 = dx * dx + dy * dy + dz * dz;
            if(!(r2 > 0.0) || !std::isfinite(r2))
                throw UniversalError(
                    "DistributedFmmGravityCalculator: singular sampled direct pair");
            const double invR = 1.0 / std::sqrt(r2);
            const double factor = masses[source] * invR * invR * invR;
            direct[3 * sample] -=
                static_cast<long double>(factor * dx);
            direct[3 * sample + 1] -=
                static_cast<long double>(factor * dy);
            direct[3 * sample + 2] -=
                static_cast<long double>(factor * dz);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, direct.data(),
                  static_cast<int>(direct.size()), MPI_LONG_DOUBLE,
                  MPI_SUM, comm_);

    std::vector<double> relativeErrors;
    relativeErrors.reserve(sampleCount);
    double relativeSum = 0.0;
    double relativeSquareSum = 0.0;
    double absoluteSum = 0.0;
    double maximumRelative = 0.0;
    for(std::size_t sample = 0; sample < sampleCount; ++sample)
    {
        const long double dx =
            static_cast<long double>(targetAccelerations[3 * sample]) -
            direct[3 * sample];
        const long double dy =
            static_cast<long double>(targetAccelerations[3 * sample + 1]) -
            direct[3 * sample + 1];
        const long double dz =
            static_cast<long double>(targetAccelerations[3 * sample + 2]) -
            direct[3 * sample + 2];
        const long double absolute = std::sqrt(dx * dx + dy * dy + dz * dz);
        const long double reference = std::sqrt(
            direct[3 * sample] * direct[3 * sample] +
            direct[3 * sample + 1] * direct[3 * sample + 1] +
            direct[3 * sample + 2] * direct[3 * sample + 2]);
        const double relative = static_cast<double>(reference > 0.0L ?
            absolute / reference : absolute);
        relativeErrors.push_back(relative);
        relativeSum += relative;
        relativeSquareSum += relative * relative;
        absoluteSum += static_cast<double>(absolute);
        maximumRelative = std::max(maximumRelative, relative);
    }
    std::sort(relativeErrors.begin(), relativeErrors.end());
    const auto percentile = [&relativeErrors](double fraction) {
        const std::size_t index = static_cast<std::size_t>(std::ceil(
            fraction * static_cast<double>(relativeErrors.size()))) - 1;
        return relativeErrors[std::min(index, relativeErrors.size() - 1)];
    };
    stats_.directErrorSampleCount = sampleCount;
    stats_.directErrorMeanRelativeAcceleration =
        relativeSum / static_cast<double>(sampleCount);
    stats_.directErrorRmsRelativeAcceleration = std::sqrt(
        relativeSquareSum / static_cast<double>(sampleCount));
    stats_.directErrorMaxRelativeAcceleration = maximumRelative;
    stats_.directErrorMedianRelativeAcceleration = percentile(0.50);
    stats_.directErrorP90RelativeAcceleration = percentile(0.90);
    stats_.directErrorP99RelativeAcceleration = percentile(0.99);
    stats_.directErrorMeanAbsoluteAcceleration =
        absoluteSum / static_cast<double>(sampleCount);
    stats_.directErrorSampleSeconds = elapsed(sampleStart);
    if(rank_ == 0)
    {
        std::printf(
            "fmm_tde_sample_error solve=%llu samples=%zu seed=%llu "
            "expansion_order=%d theta=%.17g leaf_capacity=%zu "
            "mean_relative_acceleration_error=%.17g "
            "rms_relative_acceleration_error=%.17g "
            "median_relative_acceleration_error=%.17g "
            "p90_relative_acceleration_error=%.17g "
            "p99_relative_acceleration_error=%.17g "
            "max_relative_acceleration_error=%.17g "
            "mean_absolute_acceleration_error=%.17g "
            "sample_seconds=%.17g\n",
            static_cast<unsigned long long>(solveCount_), sampleCount,
            static_cast<unsigned long long>(sampleSeed),
            options_.expansionOrder, options_.thetaCritical,
            options_.leafCapacity,
            stats_.directErrorMeanRelativeAcceleration,
            stats_.directErrorRmsRelativeAcceleration,
            stats_.directErrorMedianRelativeAcceleration,
            stats_.directErrorP90RelativeAcceleration,
            stats_.directErrorP99RelativeAcceleration,
            stats_.directErrorMaxRelativeAcceleration,
            stats_.directErrorMeanAbsoluteAcceleration,
            stats_.directErrorSampleSeconds);
        std::fflush(stdout);
    }
}

void DistributedFmmGravityCalculator::solveOwned(
    const std::vector<Vector3D>& positions,
    const std::vector<double>& masses,
    const std::vector<std::uint64_t>& cellIds,
    const Vector3D& domainLower,
    const Vector3D& domainUpper,
    std::vector<Vector3D>& acceleration,
    std::vector<double>* positiveKernelPotential)
{

    if(distributedOptions_.enablePatchForest)
    {
        if(!patchSolver_)
        {
            patchSolver_.reset(new FmmPatchDistributedSolver(
                options_, distributedOptions_, comm_));
        }
        patchSolver_->solve(positions, masses, cellIds, domainLower,
                            domainUpper, acceleration,
                            positiveKernelPotential, stats_);
        return;
    }

    const Clock::time_point totalStart = Clock::now();
    stats_ = FmmSolveStats();
    stats_.particleCount = positions.size();
    stats_.mpiRankCount = static_cast<std::size_t>(size_);
    stats_.operatorCacheBytesAtSolveStart = operatorCache_.bytesOwned();
    stats_.operatorCacheEntriesAtSolveStart = operatorCache_.entries();

    const Clock::time_point buildStart = Clock::now();
    const LocalTopologyChange localChange =
        prepareLocalTree(positions, domainLower, domainUpper);
    const bool occupancyRequiresRebuild =
        localChange.leafOccupancyChanged &&
        (!distributedOptions_.reuseInteractionPlansAcrossLeafCountChanges ||
         (distributedOptions_.maxLetWaveBytes > 0 &&
          !distributedOptions_.reuseBoundedLetWavesAcrossLeafCountChanges));
    const bool localTreeTopologyChanged =
        localChange.rootGeometryChanged || localChange.leafTopologyChanged ||
        occupancyRequiresRebuild;
    stats_.localRootGeometryChanged = localChange.rootGeometryChanged;
    stats_.localLeafTopologyChanged = localChange.leafTopologyChanged;
    stats_.localLeafOccupancyChanged = localChange.leafOccupancyChanged;
    stats_.localCountOnlyLeafChange = localChange.countOnlyLeafChange;
    stats_.operatorCacheBudgetBytes = options_.maxOperatorCacheBytes;
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

    // Phase 0 sizing for a patch forest. The process tree gets one leaf per
    // (ownerRank, patchId), so the descriptor count is the sum over ranks of
    // each rank's distinct occupied cells -- a plain SUM, no global set needed.
    if(geometryLogEnabled())
        logPatchCountSurvey(positions, domainLower, domainUpper);

    const unsigned long long localTopologyTerms[9] = {
        static_cast<unsigned long long>(positions.size()),
        localChange.rootGeometryChanged ? 1ull : 0ull,
        localChange.leafTopologyChanged ? 1ull : 0ull,
        localChange.leafOccupancyChanged ? 1ull : 0ull,
        localChange.countOnlyLeafChange ? 1ull : 0ull,
        localChange.persistentTreeRefit ? 1ull : 0ull,
        static_cast<unsigned long long>(localChange.persistentLeafSplits),
        static_cast<unsigned long long>(localChange.persistentSubtreeMerges),
        static_cast<unsigned long long>(localChange.persistentEmptyLeaves)};
    unsigned long long globalTopologyTerms[9] = {};
    MPI_Allreduce(localTopologyTerms, globalTopologyTerms, 9,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_);

    stats_.totalMass = globalMassTerms[0];
    const double totalAbsoluteMass = globalMassTerms[1];
    stats_.ranksWithRootGeometryChange =
        static_cast<std::size_t>(globalTopologyTerms[1]);
    stats_.ranksWithLeafTopologyChange =
        static_cast<std::size_t>(globalTopologyTerms[2]);
    stats_.ranksWithLeafOccupancyChange =
        static_cast<std::size_t>(globalTopologyTerms[3]);
    stats_.ranksWithCountOnlyLeafChange =
        static_cast<std::size_t>(globalTopologyTerms[4]);
    stats_.persistentTreeRefitRankCount =
        static_cast<std::size_t>(globalTopologyTerms[5]);
    stats_.persistentLeafSplitCount =
        static_cast<std::uint64_t>(globalTopologyTerms[6]);
    stats_.persistentSubtreeMergeCount =
        static_cast<std::uint64_t>(globalTopologyTerms[7]);
    stats_.persistentEmptyLeafCount =
        static_cast<std::uint64_t>(globalTopologyTerms[8]);
    if(!std::isfinite(stats_.totalMass) || !std::isfinite(totalAbsoluteMass))
        throw UniversalError(
            "DistributedFmmGravityCalculator::solve: non-finite global mass sum");

    stats_.topologyRebuildForced =
        distributedOptions_.rebuildTopologyEverySolve;
    const bool processTopologyChanged =
        stats_.topologyRebuildForced || globalTopologyTerms[1] != 0ull;
    const bool globalOccupancyRequiresRebuild =
        (!distributedOptions_.reuseInteractionPlansAcrossLeafCountChanges ||
         (distributedOptions_.maxLetWaveBytes > 0 &&
          !distributedOptions_.reuseBoundedLetWavesAcrossLeafCountChanges)) &&
        globalTopologyTerms[3] != 0ull;
    const bool letTopologyChanged =
        processTopologyChanged || globalTopologyTerms[2] != 0ull ||
        globalOccupancyRequiresRebuild;
    stats_.countOnlyTopologyReused =
        globalTopologyTerms[4] != 0ull && !letTopologyChanged;
    if(letTopologyChanged)
        rebuildTopology(positions, processTopologyChanged);
    else
    {
        stats_.processCommunicatorsReused = true;
        stats_.letCommunicatorReused = true;
    }

    stats_.topologyEpoch = topologyEpoch_;
    stats_.topologyRebuildCount = topologyRebuildCount_;
    stats_.processTopologyRebuildCount = processTopologyRebuildCount_;
    stats_.letTopologyRebuildCount = letTopologyRebuildCount_;
    stats_.activeRankCount = processTree_.activeRanks().size();
    stats_.processNodeCount = processTree_.nodes().size();
    stats_.letPlannedM2LCount =
        static_cast<std::uint64_t>(letPlan_.m2lInteractions().size());
    stats_.letPlannedP2PBlockCount =
        static_cast<std::uint64_t>(letPlan_.p2pInteractions().size());

    ProcessCoefficientStore processMultipoles(layout.coefficientCount());
    ProcessCoefficientStore processLocals(layout.coefficientCount());

    const Clock::time_point processUpStart = Clock::now();
    const std::size_t localLeaf = processTree_.leafForRank(rank_);
    if(localLeaf != FmmProcessTree::invalidIndex())
    {
        if(localTree_.nodes().empty())
            throw UniversalError("DistributedFmmGravityCalculator::solve: active process leaf has no local tree");
    }

    if(distributedOptions_.replicateProcessMultipoles)
    {
        const std::size_t coefficients = layout.coefficientCount();
        if(coefficients == 0 ||
           coefficients > static_cast<std::size_t>(
               std::numeric_limits<int>::max()) ||
           static_cast<std::size_t>(size_) >
               std::numeric_limits<std::size_t>::max() / coefficients)
            throw UniversalError(
                "DistributedFmmGravityCalculator::solve: replicated process multipoles overflow");
        std::vector<double> localRootMultipole(coefficients, 0.0);
        if(localLeaf != FmmProcessTree::invalidIndex())
        {
            const std::size_t offset = localTree_.nodes()[0].multipoleOffset;
            std::copy(localMultipoles_.begin() +
                      static_cast<std::ptrdiff_t>(offset),
                      localMultipoles_.begin() +
                      static_cast<std::ptrdiff_t>(offset + coefficients),
                      localRootMultipole.begin());
        }
        std::vector<double> gatheredRootMultipoles(
            static_cast<std::size_t>(size_) * coefficients, 0.0);
        MPI_Allgather(localRootMultipole.data(), static_cast<int>(coefficients),
                      MPI_DOUBLE, gatheredRootMultipoles.data(),
                      static_cast<int>(coefficients), MPI_DOUBLE, comm_);
        for(int peer = 0; peer < size_; ++peer)
        {
            const std::size_t leaf = processTree_.leafForRank(peer);
            if(leaf == FmmProcessTree::invalidIndex())
                continue;
            const std::size_t destination = processMultipoles.ensure(leaf);
            const std::size_t source = static_cast<std::size_t>(peer) *
                coefficients;
            std::copy(gatheredRootMultipoles.begin() +
                      static_cast<std::ptrdiff_t>(source),
                      gatheredRootMultipoles.begin() +
                      static_cast<std::ptrdiff_t>(source + coefficients),
                      processMultipoles.values.begin() +
                      static_cast<std::ptrdiff_t>(destination));
        }
        for(std::size_t depth = processTree_.maxDepth(); depth > 0; --depth)
        {
            for(std::size_t parentIndex : processTree_.levels()[depth - 1])
            {
                const FmmProcessNode& parent = processTree_.nodes()[parentIndex];
                if(parent.isLeaf())
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
        }
    }
    else
    {
        if(localLeaf != FmmProcessTree::invalidIndex())
        {
            const std::size_t begin = processMultipoles.ensure(localLeaf);
            std::copy(localMultipoles_.begin() + static_cast<std::ptrdiff_t>(
                          localTree_.nodes()[0].multipoleOffset),
                      localMultipoles_.begin() + static_cast<std::ptrdiff_t>(
                          localTree_.nodes()[0].multipoleOffset +
                          layout.coefficientCount()),
                      processMultipoles.values.begin() +
                          static_cast<std::ptrdiff_t>(begin));
        }
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
                        appendProcessCoefficients(sendBuffers[parentOwner],
                                                  nodeIndex,
                                                  processMultipoles,
                                                  topologyEpoch_);
                }
                const auto received = processUpExchange_.exchangeBytes(
                    sendBuffers, &stats_.bytesSent, &stats_.bytesReceived);
                parseProcessCoefficients(received, processMultipoles,
                                         processTree_, topologyEpoch_);

                for(std::size_t parentIndex :
                    processTree_.levels()[depth - 1])
                {
                    const FmmProcessNode& parent =
                        processTree_.nodes()[parentIndex];
                    if(parent.owner != rank_ || parent.isLeaf())
                        continue;
                    processMultipoles.zero(parentIndex);
                    const std::size_t parentOffset =
                        processMultipoles.offset(parentIndex);
                    FmmNode parentView = processView(parent, parentOffset);
                    const std::size_t children[2] = {
                        parent.left, parent.right};
                    for(std::size_t childIndex : children)
                    {
                        const std::size_t childOffset =
                            processMultipoles.offset(childIndex);
                        FmmNode childView = processView(
                            processTree_.nodes()[childIndex], childOffset);
                        FmmKernels::translateM2M(
                            childView, parentView, layout,
                            processMultipoles.values);
                    }
                }
            }
        }
    }
    stats_.peakProcessBytes = std::max(stats_.peakProcessBytes,
        processMultipoles.bytesOwned() + processLocals.bytesOwned());
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
    const std::uint64_t globalParticleCount =
        static_cast<std::uint64_t>(globalTopologyTerms[0]);
    const long double accumulationFactor = static_cast<long double>(
        std::max<std::uint64_t>(1, globalParticleCount));
    const double massTolerance = static_cast<double>(
        256.0L * std::numeric_limits<double>::epsilon() * accumulationFactor *
        std::max(1.0L, static_cast<long double>(totalAbsoluteMass)));
    if(std::abs(stats_.rootMass - stats_.totalMass) > massTolerance)
        throw UniversalError("DistributedFmmGravityCalculator::solve: global root mass mismatch");

    const Clock::time_point processInteractionStart = Clock::now();
    if(!distributedOptions_.replicateProcessMultipoles)
    {
        std::unordered_map<int, std::vector<char>> processM2LSends;
        for(const auto& entry : processPlan_.processSendNodesByRank)
        {
            for(std::size_t nodeIndex : entry.second)
                appendProcessCoefficients(processM2LSends[entry.first],
                                          nodeIndex, processMultipoles,
                                          topologyEpoch_);
        }
        FmmPeerExchangeResult processM2LReceived =
            processM2LExchange_.exchangeBytes(
                processM2LSends, &stats_.bytesSent, &stats_.bytesReceived);
        parseProcessCoefficients(processM2LReceived, processMultipoles,
                                 processTree_, topologyEpoch_);
        processM2LReceived.releaseStorage();
    }

    std::vector<double> derivativeScratch;
    derivativeScratch.reserve(layout.coefficientCount());
    std::vector<double> processOperatorScratch;
    processOperatorScratch.reserve(layout.m2lTerms().size());
    for(const FmmProcessM2LPair& pair : processPlan_.localM2LPairs)
    {
        const std::size_t sourceOffset = processMultipoles.offset(pair.sourceNode);
        const std::size_t targetOffset = processLocals.ensure(pair.targetNode);
        FmmNode source = processView(processTree_.nodes()[pair.sourceNode], sourceOffset);
        FmmNode target = processView(processTree_.nodes()[pair.targetNode], targetOffset);
        const Vector3D displacement = target.center - source.center;
        FmmKernels::computeM2LOperator(displacement, layout,
                                       derivativeScratch, processOperatorScratch);
        FmmKernels::translateM2L(source, target, layout,
                                 processMultipoles.values, processLocals.values,
                                 processOperatorScratch);
        ++stats_.processOperatorCacheMisses;
        ++stats_.processOperatorCacheBypasses;
        ++stats_.m2lCount;
        ++stats_.processM2LCount;
    }
    stats_.peakProcessBytes = std::max(stats_.peakProcessBytes,
        processMultipoles.bytesOwned() + processLocals.bytesOwned() +
        derivativeScratch.capacity() * sizeof(double) +
        processOperatorScratch.capacity() * sizeof(double));
    stats_.processInteractionSeconds = elapsed(processInteractionStart);
    std::vector<double>().swap(processOperatorScratch);
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
    // Pack and start the large LET payload before local work. The count
    // exchange is already complete, so progress calls advance the payload.
    const Clock::time_point letBeginStart = Clock::now();
    letPlan_.beginExecute(0, localTree_, positions, masses, cellIds, layout,
                          localMultipoles_, localLocals_, acceleration,
                          positiveKernelPotential,
                          distributedOptions_.maxRemoteBytes, stats_);
    const double letBeginSeconds = elapsed(letBeginStart);

    // Populate the shared operator cache with the small, balanced local M2L
    // operator set before rank-dependent LET interactions consume the remaining
    // byte budget.  This ordering is intentional: when the cache is saturated,
    // local misses otherwise become repeated uncached operator generations.
    const Clock::time_point localTraversalStart = Clock::now();
    if(!localTree_.nodes().empty())
    {
        bool planReused = false;
        const bool localPlanMustRebuild = localTreeTopologyChanged ||
            distributedOptions_.rebuildTopologyEverySolve;
        if(!localPlanMustRebuild && FmmDualTreeTraversal::localPlanReusable(
                localTree_, localInteractionPlan_))
        {
            planReused = true;
        }
        else
        {
            FmmDualTreeTraversal::buildLocalPlan(
                localTree_, options_.thetaCritical, localInteractionPlan_);
        }
        FmmDualTreeTraversal::runLocalPlan(
            localTree_, localInteractionPlan_, positions, masses, layout,
            localMultipoles_, localLocals_, acceleration,
            positiveKernelPotential, operatorCache_,
            options_.maxOperatorCacheBytes, stats_,
            progressLetExchange, &letPlan_);
        stats_.localInteractionPlanReused = planReused;
        stats_.localPlannedM2LCount =
            static_cast<std::uint64_t>(localInteractionPlan_.m2lPairs.size());
        stats_.localPlannedP2PBlockCount =
            static_cast<std::uint64_t>(localInteractionPlan_.p2pPairs.size());
    }
    else
    {
        localInteractionPlan_.clear();
        letPlan_.progressExecute();
    }
    stats_.localInteractionPlanBytes = localInteractionPlan_.bytesOwned();
    stats_.localTraversalSeconds = elapsed(localTraversalStart);

    const Clock::time_point letFinishStart = Clock::now();
    letPlan_.finishExecute(0, localTree_, positions, layout, localLocals_,
                           acceleration, positiveKernelPotential,
                           operatorCache_, distributedOptions_.maxRemoteBytes,
                           options_.maxOperatorCacheBytes, stats_);
    // Only wave 0 can be overlapped with local traversal. Remaining waves run
    // back to back; every rank must participate because the payload exchange is
    // a neighborhood collective, even when its own groups are empty.
    for(std::size_t wave = 1; wave < letPlan_.waveCount(); ++wave)
    {
        letPlan_.beginExecute(wave, localTree_, positions, masses, cellIds,
                              layout, localMultipoles_, localLocals_,
                              acceleration, positiveKernelPotential,
                              distributedOptions_.maxRemoteBytes, stats_);
        letPlan_.finishExecute(wave, localTree_, positions, layout,
                               localLocals_, acceleration,
                               positiveKernelPotential, operatorCache_,
                               distributedOptions_.maxRemoteBytes,
                               options_.maxOperatorCacheBytes, stats_);
    }
    stats_.letExecuteSeconds = letBeginSeconds + elapsed(letFinishStart);
    if(stats_.peakRemoteBytes > distributedOptions_.maxRemoteBytes)
        throw UniversalError("DistributedFmmGravityCalculator::solve: LET memory budget exceeded");

    stats_.interactionSeconds = elapsed(interactionStart);

    const Clock::time_point downwardStart = Clock::now();
    if(!localTree_.nodes().empty())
        FmmPasses::downward(localTree_, positions, layout, localLocals_,
                            acceleration, positiveKernelPotential);
    stats_.downwardSeconds = elapsed(downwardStart);

    stats_.localTreeBytes = localTree_.bytesOwned();
    stats_.localMultipoleBytes =
        localMultipoles_.capacity() * sizeof(double);
    stats_.localLocalBytes = localLocals_.capacity() * sizeof(double);
    stats_.letPlanBytes = letPlan_.bytesOwned();
    stats_.operatorCacheBytes = operatorCache_.bytesOwned();
    stats_.operatorCacheEntries = operatorCache_.entries();
    stats_.operatorCacheMaxEntries = operatorCache_.maxEntries();
    stats_.bytesOwned = stats_.localTreeBytes +
        stats_.localMultipoleBytes + stats_.localLocalBytes +
        stats_.localInteractionPlanBytes + operatorCache_.bytesOwned() +
        rootDescriptors_.capacity() * sizeof(FmmPatchRootDescriptor) +
        lastLocalStructuralSignature_.capacity() * sizeof(std::uint64_t) +
        lastLocalOccupancySignature_.capacity() * sizeof(std::uint64_t) +
        processTree_.bytesOwned() + processPlan_.bytesOwned() +
        stats_.letPlanBytes + processUpExchange_.bytesOwned() +
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
#ifdef __GLIBC__
    malloc_trim(0);
#endif
    stats_.totalSeconds = elapsed(totalStart);
}

#endif // RICH_MPI
