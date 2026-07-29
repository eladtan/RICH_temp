#include "3D/gravity/fmm/mpi/FmmPatchDistributedSolver.hpp"

#ifdef RICH_MPI

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "3D/gravity/fmm/FmmDualTreeTraversal.hpp"
#include "3D/gravity/fmm/FmmKernels.hpp"
#include "3D/gravity/fmm/FmmPasses.hpp"
#include "3D/gravity/fmm/FmmTaylorExpansion.hpp"
#include "3D/gravity/fmm/mpi/FmmDescriptorGather.hpp"
#include "3D/gravity/fmm/mpi/FmmGlobalDyadicLattice.hpp"
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

bool nonuniformDiagnosticsEnabled()
{
    const char* value = std::getenv("RICH_FMM_NONUNIFORM_DIAGNOSTICS");
    return value != nullptr && value[0] != '\0' &&
        !(value[0] == '0' && value[1] == '\0');
}

template<typename T>
T diagnosticPercentile(std::vector<T> values, double fraction)
{
    if(values.empty())
        return T();
    std::sort(values.begin(), values.end());
    const std::size_t index = std::min(
        values.size() - 1, static_cast<std::size_t>(
            std::ceil(fraction * static_cast<double>(values.size()))) - 1);
    return values[index];
}

template<typename T>
long double diagnosticMean(const std::vector<T>& values)
{
    if(values.empty())
        return 0.0L;
    long double sum = 0.0L;
    for(const T& value : values)
        sum += static_cast<long double>(value);
    return sum / static_cast<long double>(values.size());
}

void printUnsignedRankMetric(std::uint64_t call,
                             const char* name,
                             const std::vector<unsigned long long>& values)
{
    if(values.empty())
        return;
    const unsigned long long minimum = *std::min_element(
        values.begin(), values.end());
    const unsigned long long maximum = *std::max_element(
        values.begin(), values.end());
    const long double mean = diagnosticMean(values);
    std::fprintf(stdout,
        "fmm_nonuniform_rank_metric call=%llu metric=%s min=%llu "
        "mean=%.9Le p50=%llu p95=%llu p99=%llu max=%llu "
        "max_over_mean=%.9Le\n",
        static_cast<unsigned long long>(call), name, minimum, mean,
        diagnosticPercentile(values, 0.50),
        diagnosticPercentile(values, 0.95),
        diagnosticPercentile(values, 0.99), maximum,
        mean > 0.0L ? static_cast<long double>(maximum) / mean : 0.0L);
}

void printDoubleRankMetric(std::uint64_t call,
                           const char* name,
                           const std::vector<double>& values)
{
    if(values.empty())
        return;
    const double minimum = *std::min_element(values.begin(), values.end());
    const double maximum = *std::max_element(values.begin(), values.end());
    const long double mean = diagnosticMean(values);
    std::fprintf(stdout,
        "fmm_nonuniform_rank_metric call=%llu metric=%s min=%.9e "
        "mean=%.9Le p50=%.9e p95=%.9e p99=%.9e max=%.9e "
        "max_over_mean=%.9Le\n",
        static_cast<unsigned long long>(call), name, minimum, mean,
        diagnosticPercentile(values, 0.50),
        diagnosticPercentile(values, 0.95),
        diagnosticPercentile(values, 0.99), maximum,
        mean > 0.0L ? static_cast<long double>(maximum) / mean : 0.0L);
}

std::uint64_t diagnosticCellKey(int ix, int iy, int iz)
{
    return (static_cast<std::uint64_t>(ix) << 42u) |
           (static_cast<std::uint64_t>(iy) << 21u) |
           static_cast<std::uint64_t>(iz);
}

void emitRankGeometryDiagnostics(const std::vector<Vector3D>& positions,
                                 const FmmGlobalDyadicLattice& lattice,
                                 int level,
                                 std::uint64_t call,
                                 int rank,
                                 int size,
                                 const MPI_Comm& comm)
{
    std::unordered_set<std::uint64_t> occupied;
    int minIndex[3] = {std::numeric_limits<int>::max(),
                       std::numeric_limits<int>::max(),
                       std::numeric_limits<int>::max()};
    int maxIndex[3] = {-1, -1, -1};
    for(const Vector3D& point : positions)
    {
        int ix = 0;
        int iy = 0;
        int iz = 0;
        lattice.pointToCellIndices(point, level, ix, iy, iz);
        occupied.insert(diagnosticCellKey(ix, iy, iz));
        minIndex[0] = std::min(minIndex[0], ix);
        minIndex[1] = std::min(minIndex[1], iy);
        minIndex[2] = std::min(minIndex[2], iz);
        maxIndex[0] = std::max(maxIndex[0], ix);
        maxIndex[1] = std::max(maxIndex[1], iy);
        maxIndex[2] = std::max(maxIndex[2], iz);
    }

    std::unordered_set<std::uint64_t> remaining = occupied;
    std::size_t components = 0;
    const int axisCells = 1 << level;
    std::vector<std::uint64_t> stack;
    while(!remaining.empty())
    {
        ++components;
        stack.clear();
        stack.push_back(*remaining.begin());
        remaining.erase(stack.back());
        while(!stack.empty())
        {
            const std::uint64_t key = stack.back();
            stack.pop_back();
            const int ix = static_cast<int>((key >> 42u) & 0x1fffffu);
            const int iy = static_cast<int>((key >> 21u) & 0x1fffffu);
            const int iz = static_cast<int>(key & 0x1fffffu);
            const int neighbor[6][3] = {
                {ix - 1, iy, iz}, {ix + 1, iy, iz},
                {ix, iy - 1, iz}, {ix, iy + 1, iz},
                {ix, iy, iz - 1}, {ix, iy, iz + 1}};
            for(const auto& item : neighbor)
            {
                if(item[0] < 0 || item[0] >= axisCells ||
                   item[1] < 0 || item[1] >= axisCells ||
                   item[2] < 0 || item[2] >= axisCells)
                    continue;
                const std::uint64_t neighborKey = diagnosticCellKey(
                    item[0], item[1], item[2]);
                const auto found = remaining.find(neighborKey);
                if(found != remaining.end())
                {
                    stack.push_back(neighborKey);
                    remaining.erase(found);
                }
            }
        }
    }

    unsigned long long boundingCells = 0;
    if(!occupied.empty())
    {
        const unsigned long long nx = static_cast<unsigned long long>(
            maxIndex[0] - minIndex[0] + 1);
        const unsigned long long ny = static_cast<unsigned long long>(
            maxIndex[1] - minIndex[1] + 1);
        const unsigned long long nz = static_cast<unsigned long long>(
            maxIndex[2] - minIndex[2] + 1);
        boundingCells = nx * ny * nz;
    }
    const double fill = boundingCells == 0 ? 0.0 :
        static_cast<double>(occupied.size()) /
        static_cast<double>(boundingCells);

    const unsigned long long localUnsigned[4] = {
        static_cast<unsigned long long>(positions.size()),
        static_cast<unsigned long long>(occupied.size()),
        static_cast<unsigned long long>(components), boundingCells};
    const double localDouble[1] = {fill};
    std::vector<unsigned long long> gatheredUnsigned;
    std::vector<double> gatheredDouble;
    if(rank == 0)
    {
        gatheredUnsigned.resize(static_cast<std::size_t>(size) * 4);
        gatheredDouble.resize(static_cast<std::size_t>(size));
    }
    MPI_Gather(localUnsigned, 4, MPI_UNSIGNED_LONG_LONG,
               rank == 0 ? gatheredUnsigned.data() : nullptr, 4,
               MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Gather(localDouble, 1, MPI_DOUBLE,
               rank == 0 ? gatheredDouble.data() : nullptr, 1,
               MPI_DOUBLE, 0, comm);
    if(rank != 0)
        return;

    const char* names[4] = {
        "rank_particles", "rank_occupied_spatial_cells",
        "rank_spatial_components", "rank_spatial_bbox_cells"};
    for(int metric = 0; metric < 4; ++metric)
    {
        std::vector<unsigned long long> values;
        values.reserve(static_cast<std::size_t>(size));
        for(int r = 0; r < size; ++r)
            values.push_back(gatheredUnsigned[
                static_cast<std::size_t>(4 * r + metric)]);
        printUnsignedRankMetric(call, names[metric], values);
    }
    printDoubleRankMetric(call, "rank_spatial_fill_fraction", gatheredDouble);
}

void emitSpatialDiagnostics(const std::vector<Vector3D>& positions,
                            const Vector3D& domainLower,
                            const Vector3D& domainUpper,
                            int diagnosticLevel,
                            std::uint64_t call,
                            int rank,
                            int size,
                            const MPI_Comm& comm)
{
    const FmmGlobalDyadicLattice lattice =
        FmmGlobalDyadicLattice::fromDomain(domainLower, domainUpper);
    constexpr unsigned long long kMaximumGatheredEntries = 5000000ull;
    std::map<int, unsigned long long> uniqueByLevel;

    const int firstLevel = std::max(4, diagnosticLevel - 2);
    const int lastLevel = std::min(FMM_MAX_TREE_DEPTH,
                                   diagnosticLevel + 3);
    for(int level = firstLevel; level <= lastLevel; ++level)
    {
        std::map<std::uint64_t, unsigned long long> localCounts;
        for(const Vector3D& point : positions)
            ++localCounts[lattice.patchIdAtLevel(point, level)];

        const unsigned long long localEntryCount =
            static_cast<unsigned long long>(localCounts.size());
        std::vector<unsigned long long> entryCounts;
        if(rank == 0)
            entryCounts.resize(static_cast<std::size_t>(size));
        MPI_Gather(&localEntryCount, 1, MPI_UNSIGNED_LONG_LONG,
                   rank == 0 ? entryCounts.data() : nullptr, 1,
                   MPI_UNSIGNED_LONG_LONG, 0, comm);

        int exact = 1;
        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        unsigned long long ownerTagged = 0;
        if(rank == 0)
        {
            receiveCounts.resize(static_cast<std::size_t>(size));
            displacements.resize(static_cast<std::size_t>(size));
            unsigned long long packedOffset = 0;
            for(int r = 0; r < size; ++r)
            {
                ownerTagged += entryCounts[static_cast<std::size_t>(r)];
                const unsigned long long packed =
                    2ull * entryCounts[static_cast<std::size_t>(r)];
                if(entryCounts[static_cast<std::size_t>(r)] >
                       kMaximumGatheredEntries ||
                   packed > static_cast<unsigned long long>(
                       std::numeric_limits<int>::max()) ||
                   packedOffset + packed > static_cast<unsigned long long>(
                       std::numeric_limits<int>::max()))
                    exact = 0;
                if(exact != 0)
                {
                    receiveCounts[static_cast<std::size_t>(r)] =
                        static_cast<int>(packed);
                    displacements[static_cast<std::size_t>(r)] =
                        static_cast<int>(packedOffset);
                }
                packedOffset += packed;
            }
            if(ownerTagged > kMaximumGatheredEntries)
                exact = 0;
        }
        MPI_Bcast(&exact, 1, MPI_INT, 0, comm);
        if(exact == 0)
        {
            if(rank == 0)
            {
                std::fprintf(stdout,
                    "fmm_nonuniform_spatial call=%llu level=%d "
                    "owner_tagged=%llu exact=0 reason=gather_cap\n",
                    static_cast<unsigned long long>(call), level, ownerTagged);
                std::fflush(stdout);
            }
            continue;
        }

        std::vector<unsigned long long> localPacked;
        localPacked.reserve(2 * localCounts.size());
        for(const auto& entry : localCounts)
        {
            localPacked.push_back(static_cast<unsigned long long>(entry.first));
            localPacked.push_back(entry.second);
        }
        std::vector<unsigned long long> gathered;
        if(rank == 0)
            gathered.resize(static_cast<std::size_t>(2ull * ownerTagged));
        MPI_Gatherv(localPacked.data(), static_cast<int>(localPacked.size()),
                    MPI_UNSIGNED_LONG_LONG,
                    rank == 0 ? gathered.data() : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_UNSIGNED_LONG_LONG, 0, comm);
        if(rank != 0)
            continue;

        std::vector<std::pair<unsigned long long, unsigned long long>> entries;
        entries.reserve(static_cast<std::size_t>(ownerTagged));
        for(std::size_t i = 0; i < gathered.size(); i += 2)
            entries.emplace_back(gathered[i], gathered[i + 1]);
        std::sort(entries.begin(), entries.end());

        std::vector<unsigned long long> multiplicities;
        std::vector<unsigned long long> particlesPerPatch;
        std::vector<double> dominantFractions;
        unsigned long long totalParticles = 0;
        unsigned long long sharedParticles = 0;
        unsigned long long sharedPatches = 0;
        long double crossOwnerPairs = 0.0L;
        std::size_t cursor = 0;
        while(cursor < entries.size())
        {
            const unsigned long long id = entries[cursor].first;
            unsigned long long total = 0;
            unsigned long long largest = 0;
            unsigned long long localPairs = 0;
            unsigned long long multiplicity = 0;
            while(cursor < entries.size() && entries[cursor].first == id)
            {
                const unsigned long long count = entries[cursor].second;
                total += count;
                largest = std::max(largest, count);
                localPairs += count * (count - 1ull) / 2ull;
                ++multiplicity;
                ++cursor;
            }
            multiplicities.push_back(multiplicity);
            particlesPerPatch.push_back(total);
            dominantFractions.push_back(total == 0 ? 0.0 :
                static_cast<double>(largest) / static_cast<double>(total));
            totalParticles += total;
            if(multiplicity > 1)
            {
                ++sharedPatches;
                sharedParticles += total;
            }
            const long double allPairs = static_cast<long double>(total) *
                static_cast<long double>(total - 1ull) / 2.0L;
            crossOwnerPairs += allPairs - static_cast<long double>(localPairs);
        }

        const unsigned long long unique =
            static_cast<unsigned long long>(multiplicities.size());
        uniqueByLevel[level] = unique;
        const double duplication = unique == 0 ? 0.0 :
            static_cast<double>(ownerTagged) / static_cast<double>(unique);
        const double sharedPatchFraction = unique == 0 ? 0.0 :
            static_cast<double>(sharedPatches) / static_cast<double>(unique);
        const double sharedParticleFraction = totalParticles == 0 ? 0.0 :
            static_cast<double>(sharedParticles) /
            static_cast<double>(totalParticles);

        std::fprintf(stdout,
            "fmm_nonuniform_spatial call=%llu level=%d exact=1 "
            "owner_tagged=%llu unique=%llu duplication=%.9e "
            "shared_patch_fraction=%.9e shared_particle_fraction=%.9e "
            "owner_mult_p50=%llu owner_mult_p95=%llu owner_mult_p99=%llu "
            "owner_mult_max=%llu particles_p50=%llu particles_p95=%llu "
            "particles_p99=%llu particles_max=%llu "
            "dominant_owner_fraction_p05=%.9e "
            "dominant_owner_fraction_p50=%.9e "
            "cross_owner_same_patch_pairs=%.9Le\n",
            static_cast<unsigned long long>(call), level, ownerTagged, unique,
            duplication, sharedPatchFraction, sharedParticleFraction,
            diagnosticPercentile(multiplicities, 0.50),
            diagnosticPercentile(multiplicities, 0.95),
            diagnosticPercentile(multiplicities, 0.99),
            diagnosticPercentile(multiplicities, 1.0),
            diagnosticPercentile(particlesPerPatch, 0.50),
            diagnosticPercentile(particlesPerPatch, 0.95),
            diagnosticPercentile(particlesPerPatch, 0.99),
            diagnosticPercentile(particlesPerPatch, 1.0),
            diagnosticPercentile(dominantFractions, 0.05),
            diagnosticPercentile(dominantFractions, 0.50),
            crossOwnerPairs);
        std::fflush(stdout);
    }

    if(rank == 0)
    {
        for(auto current = uniqueByLevel.begin(); current != uniqueByLevel.end();
            ++current)
        {
            auto next = current;
            ++next;
            if(next == uniqueByLevel.end() || current->second == 0 ||
               next->second == 0)
                continue;
            const double dimension = std::log(
                static_cast<double>(next->second) /
                static_cast<double>(current->second)) / std::log(2.0);
            std::fprintf(stdout,
                "fmm_nonuniform_dimension call=%llu level_lo=%d level_hi=%d "
                "unique_lo=%llu unique_hi=%llu effective_dimension=%.9e\n",
                static_cast<unsigned long long>(call), current->first,
                next->first, current->second, next->second, dimension);
        }
        std::fflush(stdout);
    }

    emitRankGeometryDiagnostics(positions, lattice, diagnosticLevel, call,
                                rank, size, comm);
}

void emitRankWorkDiagnostics(const FmmSolveStats& stats,
                             std::uint64_t call,
                             int rank,
                             int size,
                             const MPI_Comm& comm)
{
    constexpr int kUnsignedCount = 12;
    const unsigned long long localUnsigned[kUnsignedCount] = {
        static_cast<unsigned long long>(stats.particleCount),
        static_cast<unsigned long long>(stats.localPatchCount),
        static_cast<unsigned long long>(stats.processOwnedNodeCount),
        static_cast<unsigned long long>(stats.processOwnedM2LCount),
        static_cast<unsigned long long>(stats.localPlannedP2PBlockCount),
        static_cast<unsigned long long>(stats.letPlannedP2PBlockCount),
        static_cast<unsigned long long>(stats.letP2PBlockCount),
        static_cast<unsigned long long>(stats.letM2PCount),
        static_cast<unsigned long long>(stats.bytesSent),
        static_cast<unsigned long long>(stats.bytesReceived),
        static_cast<unsigned long long>(stats.peakRemoteBytes),
        static_cast<unsigned long long>(stats.letPlanBytes)};
    constexpr int kDoubleCount = 10;
    const double localDouble[kDoubleCount] = {
        stats.totalSeconds, stats.topologyRebuildSeconds, stats.letPlanSeconds,
        stats.letExecuteSeconds, stats.letExchangeSeconds, stats.letP2PSeconds,
        stats.letM2PSeconds, stats.localTraversalSeconds,
        stats.processInteractionSeconds, stats.processDownwardSeconds};

    std::vector<unsigned long long> gatheredUnsigned;
    std::vector<double> gatheredDouble;
    if(rank == 0)
    {
        gatheredUnsigned.resize(
            static_cast<std::size_t>(size) * kUnsignedCount);
        gatheredDouble.resize(static_cast<std::size_t>(size) * kDoubleCount);
    }
    MPI_Gather(localUnsigned, kUnsignedCount, MPI_UNSIGNED_LONG_LONG,
               rank == 0 ? gatheredUnsigned.data() : nullptr,
               kUnsignedCount, MPI_UNSIGNED_LONG_LONG, 0, comm);
    MPI_Gather(localDouble, kDoubleCount, MPI_DOUBLE,
               rank == 0 ? gatheredDouble.data() : nullptr,
               kDoubleCount, MPI_DOUBLE, 0, comm);
    if(rank != 0)
        return;

    const char* unsignedNames[kUnsignedCount] = {
        "particles", "patches", "process_owned_nodes",
        "process_owned_m2l", "local_planned_p2p_blocks",
        "let_planned_p2p_blocks", "let_active_p2p_blocks", "let_m2p",
        "bytes_sent", "bytes_received", "peak_remote_bytes",
        "let_plan_bytes"};
    for(int metric = 0; metric < kUnsignedCount; ++metric)
    {
        std::vector<unsigned long long> values;
        values.reserve(static_cast<std::size_t>(size));
        for(int r = 0; r < size; ++r)
            values.push_back(gatheredUnsigned[static_cast<std::size_t>(
                kUnsignedCount * r + metric)]);
        printUnsignedRankMetric(call, unsignedNames[metric], values);
    }

    const char* doubleNames[kDoubleCount] = {
        "total_seconds", "topology_seconds", "let_plan_seconds",
        "let_execute_seconds", "let_exchange_seconds", "let_p2p_seconds",
        "let_m2p_seconds", "local_traversal_seconds",
        "process_interaction_seconds", "process_downward_seconds"};
    for(int metric = 0; metric < kDoubleCount; ++metric)
    {
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(size));
        for(int r = 0; r < size; ++r)
            values.push_back(gatheredDouble[static_cast<std::size_t>(
                kDoubleCount * r + metric)]);
        printDoubleRankMetric(call, doubleNames[metric], values);
    }
    std::fflush(stdout);
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
            // Rebuilding the process tree changes process-node indices and
            // routing, but patch LET subplans are keyed only by stable patch
            // identities and patch-tree spatial keys. FmmPatchLetPlan::build
            // compares the exact current source-patch set and source topology
            // generation for every target before reusing it, so unaffected
            // target traversals remain valid across a small patch-set change.
            // An explicitly forced rebuild remains a true cold rebuild.
            topologyInitialized_ && !forcedRebuild);
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

    // Keep diagnostic collectives outside the measured solve.  The resulting
    // fmm_nonuniform_* lines can therefore be compared with the ordinary
    // fmm_solve_trace without subtracting diagnostic overhead.
    if(nonuniformDiagnosticsEnabled())
    {
        static std::uint64_t diagnosticCall = 0;
        ++diagnosticCall;
        emitSpatialDiagnostics(
            positions, domainLower, domainUpper,
            distributedOptions_.minimumPatchLevel, diagnosticCall,
            rank_, size_, comm_);
        emitRankWorkDiagnostics(stats, diagnosticCall, rank_, size_, comm_);
        letPlan_.emitNonuniformityDiagnostics(forest_, diagnosticCall);
    }
}

#endif // RICH_MPI
