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

// Each extra forced level costs roughly eight nodes per particle once leaves
// hold a single body. This only catches runaway depth; the real signal is the
// requested LET payload reported by the FMMLET diagnostic.
constexpr std::size_t kMaxNodesPerParticle = 64;

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
    letTopologyRebuildCount_(0)
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
    else if(distributedOptions_.maxRemoteBytes < 2)
    {
        localOptionsOk = false;
        localOptionsError = "DistributedFmmGravityCalculator: remote memory budget is too small";
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

    const double localDoubleOptions[5] = {
        options_.thetaCritical, distributedOptions_.rootSlackFactor,
        distributedOptions_.persistentLeafSplitFactor,
        distributedOptions_.persistentLeafMergeFactor,
        options_.maxLeafHalfSize};
    double minimumDoubleOptions[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double maximumDoubleOptions[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(localDoubleOptions, minimumDoubleOptions, 5,
                  MPI_DOUBLE, MPI_MIN, comm_);
    MPI_Allreduce(localDoubleOptions, maximumDoubleOptions, 5,
                  MPI_DOUBLE, MPI_MAX, comm_);

    const unsigned long long localIntegerOptions[13] = {
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
        static_cast<unsigned long long>(distributedOptions_.maxLetWaveBytes)};
    unsigned long long minimumIntegerOptions[13] = {};
    unsigned long long maximumIntegerOptions[13] = {};
    MPI_Allreduce(localIntegerOptions, minimumIntegerOptions, 13,
                  MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm_);
    MPI_Allreduce(localIntegerOptions, maximumIntegerOptions, 13,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm_);

    bool optionsMatch = true;
    for(int i = 0; i < 5; ++i)
        optionsMatch = optionsMatch &&
            minimumDoubleOptions[i] == maximumDoubleOptions[i];
    for(int i = 0; i < 13; ++i)
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
    const FmmRankRootDescriptor local = localRootDescriptor();
    rootDescriptors_.resize(static_cast<std::size_t>(size_));
    MPI_Allgather(&local, static_cast<int>(sizeof(FmmRankRootDescriptor)), MPI_BYTE,
                  rootDescriptors_.data(),
                  static_cast<int>(sizeof(FmmRankRootDescriptor)), MPI_BYTE,
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
                   distributedOptions_.maxLetWaveBytes,
                   fmmTaylorCoefficientCount(options_.expansionOrder), stats_);
    stats_.topologyRebuildSeconds = elapsed(topologyStart);
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
        !distributedOptions_.reuseInteractionPlansAcrossLeafCountChanges;
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
        !distributedOptions_.reuseInteractionPlansAcrossLeafCountChanges &&
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
        rootDescriptors_.capacity() * sizeof(FmmRankRootDescriptor) +
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
