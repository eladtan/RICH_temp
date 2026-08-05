#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <mpi.h>

#include "source/3D/gravity/DistributedGravityCalculator.hpp"
#include "source/3D/gravity/fmm/mpi/DistributedFmmGravityCalculator.hpp"
#include "source/misc/universal_error.hpp"

namespace
{
constexpr std::uint64_t kVirtualBins = 4096;
constexpr unsigned int kVirtualBinsPerAxis = 16;
constexpr unsigned int kMortonBitsPerAxis = 4;
constexpr std::size_t kProbeCount = 100;
constexpr std::size_t kMebibyte = 1024u * 1024u;

struct Options
{
    std::uint64_t globalParticles = 0;
    int expectedNodes = 0;
    int expectedRanksPerNode = 0;
    int repeats = 2;
    int fmmOrder = 4;
    double fmmTheta = 0.5;
    double quadrupoleTheta = 0.5;
    std::size_t fmmLeafCapacity = 32;
    double fmmMeanErrorTarget = 1e-3;
    double fmmMaxErrorTarget = 5e-3;
    std::size_t fmmMaxRemoteBytes = 512u * kMebibyte;
    std::size_t fmmMaxOperatorCacheBytes = 64u * kMebibyte;
    bool warmOnly = false;
    std::string outputPath;
};

struct LocalParticles
{
    std::vector<Vector3D> positions;
    std::vector<double> masses;
    std::vector<std::uint64_t> ids;
};

struct ProbeReference
{
    std::array<std::uint64_t, kProbeCount> ids{};
    std::array<Vector3D, kProbeCount> acceleration;
    std::array<double, kProbeCount> forceScale{};
};

struct ProbeErrorStats
{
    double maximum = 0.0;
    double mean = 0.0;
};

constexpr std::size_t kFmmTimingMetricCount = 26;
constexpr std::size_t kFmmWorkMetricCount = 22;

const std::array<const char*, kFmmTimingMetricCount> kFmmTimingMetricNames = {{
    "build_seconds",
    "upward_seconds",
    "process_upward_seconds",
    "process_interaction_seconds",
    "process_downward_seconds",
    "let_plan_seconds",
    "let_execute_seconds",
    "let_exchange_seconds",
    "let_preparation_seconds",
    "let_payload_planning_seconds",
    "let_payload_packing_seconds",
    "let_payload_flatten_seconds",
    "let_count_exchange_seconds",
    "let_receive_setup_seconds",
    "let_payload_launch_seconds",
    "let_payload_release_seconds",
    "let_payload_lifetime_seconds",
    "let_residual_wait_seconds",
    "let_validation_seconds",
    "let_decode_seconds",
    "let_m2l_seconds",
    "let_p2p_seconds",
    "local_traversal_seconds",
    "interaction_seconds",
    "downward_seconds",
    "total_seconds"
}};

const std::array<const char*, kFmmWorkMetricCount> kFmmWorkMetricNames = {{
    "particles",
    "nodes",
    "leaves",
    "local_m2l",
    "local_p2p_pairs",
    "process_m2l",
    "let_m2l",
    "let_p2p_pairs",
    "bytes_sent",
    "bytes_received",
    "peak_remote_bytes",
    "peak_process_bytes",
    "local_cache_hits",
    "local_cache_misses",
    "local_cache_bypasses",
    "let_cache_hits",
    "let_cache_misses",
    "let_cache_bypasses",
    "let_progress_calls",
    "let_progress_incomplete_calls",
    "let_completion_progress_call",
    "let_completed_before_finish"
}};

struct RankMetricSummary
{
    double minimum = 0.0;
    double mean = 0.0;
    double maximum = 0.0;

    double imbalance() const
    {
        return mean > 0.0 ? maximum / mean : 0.0;
    }
};

struct RankMetricAccumulator
{
    double minimumSum = 0.0;
    double meanSum = 0.0;
    double maximumSum = 0.0;
    int samples = 0;

    void add(double minimum, double mean, double maximum)
    {
        minimumSum += minimum;
        meanSum += mean;
        maximumSum += maximum;
        ++samples;
    }

    RankMetricSummary average() const
    {
        RankMetricSummary result;
        if(samples == 0)
            return result;
        const double inverseSamples = 1.0 / static_cast<double>(samples);
        result.minimum = minimumSum * inverseSamples;
        result.mean = meanSum * inverseSamples;
        result.maximum = maximumSum * inverseSamples;
        return result;
    }
};

struct FmmProfile
{
    std::array<RankMetricAccumulator, kFmmTimingMetricCount> timings;
    std::array<RankMetricAccumulator, kFmmWorkMetricCount> work;
};

struct SolverResult
{
    double bestMaxSeconds = std::numeric_limits<double>::infinity();
    double meanMaxSeconds = 0.0;
    double warmBestMaxSeconds = std::numeric_limits<double>::infinity();
    double warmMeanMaxSeconds = 0.0;
    double walkMaxSeconds = std::numeric_limits<double>::infinity();
    double probeScaledError = std::numeric_limits<double>::infinity();
    double probeMeanScaledError = std::numeric_limits<double>::infinity();
    double checksum = 0.0;
    std::uint64_t bytesSent = 0;
    std::uint64_t bytesReceived = 0;
    std::uint64_t peakRemoteBytes = 0;
    std::uint64_t peakProcessBytes = 0;
    std::uint64_t persistentBytes = 0;
    std::uint64_t localTreeBytes = 0;
    std::uint64_t localMultipoleBytes = 0;
    std::uint64_t localLocalBytes = 0;
    std::uint64_t letPlanBytes = 0;
    std::uint64_t operatorCacheBytes = 0;
    std::uint64_t operatorCacheBudgetBytes = 0;
    std::uint64_t localOperatorCacheBytes = 0;
    std::uint64_t localOperatorCacheEntries = 0;
    std::uint64_t localOperatorCacheMaxEntries = 0;
    std::uint64_t localOperatorCacheHits = 0;
    std::uint64_t localOperatorCacheMisses = 0;
    std::uint64_t localOperatorCacheBypasses = 0;
    std::uint64_t letOperatorCacheBytes = 0;
    std::uint64_t letOperatorCacheEntries = 0;
    std::uint64_t letOperatorCacheMaxEntries = 0;
    std::uint64_t letOperatorCacheHits = 0;
    std::uint64_t letOperatorCacheMisses = 0;
    std::uint64_t letOperatorCacheBypasses = 0;
    std::uint64_t processOperatorCacheMisses = 0;
    std::uint64_t processOperatorCacheBypasses = 0;
    FmmProfile coldProfile;
    FmmProfile warmProfile;
    bool topologyReused = true;
    bool finite = true;
};

double radicalInverse(std::uint64_t index, unsigned int base)
{
    double result = 0.0;
    double place = 1.0 / static_cast<double>(base);
    while(index != 0)
    {
        result += static_cast<double>(index % base) * place;
        index /= base;
        place /= static_cast<double>(base);
    }
    return result;
}

unsigned int mortonCoordinate(std::uint64_t code, unsigned int axis)
{
    unsigned int result = 0;
    for(unsigned int bit = 0; bit < kMortonBitsPerAxis; ++bit)
    {
        const unsigned int sourceBit = 3u * bit + axis;
        result |= static_cast<unsigned int>((code >> sourceBit) & 1u) << bit;
    }
    return result;
}

Vector3D positionForId(std::uint64_t id)
{
    const std::uint64_t bin = id % kVirtualBins;
    const std::uint64_t ordinal = id / kVirtualBins;
    const unsigned int bx = mortonCoordinate(bin, 0);
    const unsigned int by = mortonCoordinate(bin, 1);
    const unsigned int bz = mortonCoordinate(bin, 2);
    const double ux = 0.1 + 0.8 * radicalInverse(ordinal + 1, 2);
    const double uy = 0.1 + 0.8 * radicalInverse(ordinal + 1, 3);
    const double uz = 0.1 + 0.8 * radicalInverse(ordinal + 1, 5);
    const double scale = 1.9 / static_cast<double>(kVirtualBinsPerAxis);
    return Vector3D(-0.95 + scale * (static_cast<double>(bx) + ux),
                    -0.95 + scale * (static_cast<double>(by) + uy),
                    -0.95 + scale * (static_cast<double>(bz) + uz));
}

std::array<unsigned long long, 2> localMemoryKiB()
{
    std::array<unsigned long long, 2> result{};
    std::ifstream input("/proc/self/status");
    std::string key;
    while(input >> key)
    {
        if(key == "VmRSS:" || key == "VmHWM:")
        {
            unsigned long long value = 0;
            std::string unit;
            input >> value >> unit;
            result[key == "VmRSS:" ? 0 : 1] = value;
        }
        else
        {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }
    return result;
}

void reportStage(const std::string& stage, const MPI_Comm& comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    const std::array<unsigned long long, 2> local = localMemoryKiB();
    std::array<unsigned long long, 2> global{};
    MPI_Allreduce(local.data(), global.data(), 2, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, comm);
    if(rank == 0)
    {
        std::cout << "benchmark_stage " << stage
                  << " max_rss_kib=" << global[0]
                  << " max_hwm_kib=" << global[1] << std::endl;
    }
}

void trimAllocator()
{
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

std::uint64_t massUnitsForId(std::uint64_t id)
{
    // Normalizing integer mass units by their exact global sum makes the
    // particle masses independent of MPI rank count.
    return 101u + 2u * ((37u * id + 11u) % 101u);
}

std::size_t fmmM2LTermCount(int expansionOrder)
{
    std::size_t result = 1;
    for(int factor = 1; factor <= 6; ++factor)
        result = result * static_cast<std::size_t>(expansionOrder + factor) /
                 static_cast<std::size_t>(factor);
    return result;
}

Options parseOptions(int argc, char** argv)
{
    Options result;
    for(int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if(arg == "--particles" && i + 1 < argc)
            result.globalParticles = std::strtoull(argv[++i], nullptr, 10);
        else if(arg == "--expected-nodes" && i + 1 < argc)
            result.expectedNodes = std::atoi(argv[++i]);
        else if(arg == "--expected-ranks-per-node" && i + 1 < argc)
            result.expectedRanksPerNode = std::atoi(argv[++i]);
        else if(arg == "--repeats" && i + 1 < argc)
            result.repeats = std::atoi(argv[++i]);
        else if(arg == "--warm-only")
            result.warmOnly = true;
        else if(arg == "--fmm-order" && i + 1 < argc)
            result.fmmOrder = std::atoi(argv[++i]);
        else if(arg == "--fmm-theta" && i + 1 < argc)
            result.fmmTheta = std::strtod(argv[++i], nullptr);
        else if(arg == "--quadrupole-theta" && i + 1 < argc)
            result.quadrupoleTheta = std::strtod(argv[++i], nullptr);
        else if(arg == "--fmm-leaf-capacity" && i + 1 < argc)
        {
            const char* text = argv[++i];
            if(text[0] == '-')
                throw UniversalError(
                    "fmm_mpi_scaling_benchmark: invalid FMM leaf capacity");
            const unsigned long long value = std::strtoull(text, nullptr, 10);
            if(value == 0 || value > static_cast<unsigned long long>(
                   std::numeric_limits<std::size_t>::max()))
                throw UniversalError(
                    "fmm_mpi_scaling_benchmark: invalid FMM leaf capacity");
            result.fmmLeafCapacity = static_cast<std::size_t>(value);
        }
        else if(arg == "--fmm-mean-error-target" && i + 1 < argc)
            result.fmmMeanErrorTarget = std::strtod(argv[++i], nullptr);
        else if(arg == "--fmm-max-error-target" && i + 1 < argc)
            result.fmmMaxErrorTarget = std::strtod(argv[++i], nullptr);
        else if(arg == "--fmm-max-remote-mib" && i + 1 < argc)
        {
            const unsigned long long value = std::strtoull(argv[++i], nullptr, 10);
            if(value == 0 || value >
               static_cast<unsigned long long>(
                   std::numeric_limits<std::size_t>::max() / kMebibyte))
                throw UniversalError(
                    "fmm_mpi_scaling_benchmark: invalid FMM memory budget");
            result.fmmMaxRemoteBytes = static_cast<std::size_t>(value) *
                                       kMebibyte;
        }
        else if(arg == "--fmm-operator-cache-mib" && i + 1 < argc)
        {
            const unsigned long long value = std::strtoull(argv[++i], nullptr, 10);
            if(value > static_cast<unsigned long long>(
                   std::numeric_limits<std::size_t>::max() / kMebibyte))
                throw UniversalError(
                    "fmm_mpi_scaling_benchmark: invalid operator cache budget");
            result.fmmMaxOperatorCacheBytes =
                static_cast<std::size_t>(value) * kMebibyte;
        }
        else if(arg == "--output" && i + 1 < argc)
            result.outputPath = argv[++i];
        else
            throw UniversalError("fmm_mpi_scaling_benchmark: invalid command line");
    }
    if(result.globalParticles == 0 || result.expectedNodes <= 0 ||
       result.expectedRanksPerNode <= 0 || result.repeats <= 0 ||
       result.outputPath.empty() || result.fmmOrder < 1 ||
       result.fmmOrder > FMM_MAX_ORDER ||
       !(result.fmmTheta > 0.0) || result.fmmTheta > 1.0 ||
       !std::isfinite(result.fmmTheta) ||
       !(result.quadrupoleTheta > 0.0) || result.quadrupoleTheta > 1.0 ||
       !std::isfinite(result.quadrupoleTheta) || result.fmmLeafCapacity == 0 ||
       !(result.fmmMeanErrorTarget > 0.0) ||
       !std::isfinite(result.fmmMeanErrorTarget) ||
       !(result.fmmMaxErrorTarget > 0.0) ||
       !std::isfinite(result.fmmMaxErrorTarget))
        throw UniversalError("fmm_mpi_scaling_benchmark: missing or invalid option");
    return result;
}

LocalParticles makeLocalParticles(std::uint64_t globalParticles,
                                  int rank,
                                  int size,
                                  const MPI_Comm& comm)
{
    LocalParticles result;
    const std::uint64_t firstBin =
        kVirtualBins * static_cast<std::uint64_t>(rank) /
        static_cast<std::uint64_t>(size);
    const std::uint64_t lastBin =
        kVirtualBins * static_cast<std::uint64_t>(rank + 1) /
        static_cast<std::uint64_t>(size);

    const std::uint64_t estimated = globalParticles /
        static_cast<std::uint64_t>(size) + kVirtualBins;
    result.positions.reserve(static_cast<std::size_t>(estimated));
    result.masses.reserve(static_cast<std::size_t>(estimated));
    result.ids.reserve(static_cast<std::size_t>(estimated));

    unsigned long long localMassUnits = 0;
    for(std::uint64_t bin = firstBin; bin < lastBin; ++bin)
    {
        for(std::uint64_t id = bin; id < globalParticles; id += kVirtualBins)
        {
            const std::uint64_t units = massUnitsForId(id);
            result.positions.push_back(positionForId(id));
            result.masses.push_back(static_cast<double>(units));
            result.ids.push_back(id);
            localMassUnits += static_cast<unsigned long long>(units);
        }
    }

    unsigned long long globalMassUnits = 0;
    MPI_Allreduce(&localMassUnits, &globalMassUnits, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    if(globalMassUnits == 0)
        throw UniversalError("fmm_mpi_scaling_benchmark: invalid global mass");
    for(double& mass : result.masses)
        mass /= static_cast<double>(globalMassUnits);

    const unsigned long long localCount =
        static_cast<unsigned long long>(result.positions.size());
    unsigned long long globalCount = 0;
    MPI_Allreduce(&localCount, &globalCount, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, comm);
    if(globalCount != globalParticles)
        throw UniversalError("fmm_mpi_scaling_benchmark: particle count mismatch");
    return result;
}

std::array<std::uint64_t, kProbeCount> makeProbeIds(std::uint64_t count)
{
    std::array<std::uint64_t, kProbeCount> result{};
    for(std::size_t i = 0; i < kProbeCount; ++i)
    {
        result[i] = static_cast<std::uint64_t>(
            (static_cast<long double>(i) *
             static_cast<long double>(count - 1)) /
            static_cast<long double>(kProbeCount - 1));
    }
    return result;
}

ProbeReference computeProbeReference(const LocalParticles& local,
                                     std::uint64_t globalParticles,
                                     const MPI_Comm& comm)
{
    ProbeReference result;
    result.ids = makeProbeIds(globalParticles);
    std::array<double, 3 * kProbeCount> localAcceleration{};
    std::array<double, kProbeCount> localScale{};

    for(std::size_t probe = 0; probe < kProbeCount; ++probe)
    {
        const Vector3D target = positionForId(result.ids[probe]);
        for(std::size_t source = 0; source < local.positions.size(); ++source)
        {
            if(local.ids[source] == result.ids[probe])
                continue;
            const Vector3D delta = target - local.positions[source];
            const double r2 = delta.x * delta.x + delta.y * delta.y +
                              delta.z * delta.z;
            if(!(r2 > 0.0) || !std::isfinite(r2))
                throw UniversalError(
                    "fmm_mpi_scaling_benchmark: coincident or invalid particles");
            const double invR = 1.0 / std::sqrt(r2);
            const double invR3 = invR / r2;
            const double weight = local.masses[source] * invR3;
            localAcceleration[3 * probe] -= weight * delta.x;
            localAcceleration[3 * probe + 1] -= weight * delta.y;
            localAcceleration[3 * probe + 2] -= weight * delta.z;
            localScale[probe] += local.masses[source] / r2;
        }
    }

    std::array<double, 3 * kProbeCount> globalAcceleration{};
    MPI_Allreduce(localAcceleration.data(), globalAcceleration.data(),
                  static_cast<int>(globalAcceleration.size()), MPI_DOUBLE,
                  MPI_SUM, comm);
    MPI_Allreduce(localScale.data(), result.forceScale.data(),
                  static_cast<int>(result.forceScale.size()), MPI_DOUBLE,
                  MPI_SUM, comm);
    for(std::size_t i = 0; i < kProbeCount; ++i)
    {
        result.acceleration[i] = Vector3D(globalAcceleration[3 * i],
                                          globalAcceleration[3 * i + 1],
                                          globalAcceleration[3 * i + 2]);
    }
    return result;
}

std::array<Vector3D, kProbeCount> collectProbeAccelerations(
    const LocalParticles& local,
    const std::vector<Vector3D>& acceleration,
    const ProbeReference& reference,
    const MPI_Comm& comm)
{
    std::unordered_map<std::uint64_t, std::size_t> slotById;
    for(std::size_t i = 0; i < kProbeCount; ++i)
        slotById[reference.ids[i]] = i;

    std::array<double, 3 * kProbeCount> localValues{};
    for(std::size_t i = 0; i < local.ids.size(); ++i)
    {
        const auto found = slotById.find(local.ids[i]);
        if(found == slotById.end())
            continue;
        const std::size_t slot = found->second;
        localValues[3 * slot] = acceleration[i].x;
        localValues[3 * slot + 1] = acceleration[i].y;
        localValues[3 * slot + 2] = acceleration[i].z;
    }

    std::array<double, 3 * kProbeCount> globalValues{};
    MPI_Allreduce(localValues.data(), globalValues.data(),
                  static_cast<int>(globalValues.size()), MPI_DOUBLE,
                  MPI_SUM, comm);
    std::array<Vector3D, kProbeCount> result;
    for(std::size_t i = 0; i < kProbeCount; ++i)
        result[i] = Vector3D(globalValues[3 * i], globalValues[3 * i + 1],
                             globalValues[3 * i + 2]);
    return result;
}

double norm(const Vector3D& value)
{
    return std::sqrt(value.x * value.x + value.y * value.y +
                     value.z * value.z);
}

ProbeErrorStats probeScaledErrors(
    const std::array<Vector3D, kProbeCount>& calculated,
    const ProbeReference& reference)
{
    ProbeErrorStats result;
    long double sum = 0.0L;
    for(std::size_t i = 0; i < kProbeCount; ++i)
    {
        const double scaledError =
            norm(calculated[i] - reference.acceleration[i]) /
            std::max(reference.forceScale[i], 1e-30);
        result.maximum = std::max(result.maximum, scaledError);
        sum += static_cast<long double>(scaledError);
    }
    result.mean = static_cast<double>(
        sum / static_cast<long double>(kProbeCount));
    return result;
}

bool finiteAcceleration(const std::vector<Vector3D>& values)
{
    for(const Vector3D& value : values)
    {
        if(!std::isfinite(value.x) || !std::isfinite(value.y) ||
           !std::isfinite(value.z))
            return false;
    }
    return true;
}

double accelerationChecksum(const std::vector<Vector3D>& values,
                            const std::vector<std::uint64_t>& ids,
                            const MPI_Comm& comm)
{
    long double local = 0.0L;
    for(std::size_t i = 0; i < values.size(); ++i)
    {
        const long double weight =
            static_cast<long double>((ids[i] % 104729u) + 1u);
        local += weight * (static_cast<long double>(values[i].x) +
                           0.5L * static_cast<long double>(values[i].y) +
                           0.25L * static_cast<long double>(values[i].z));
    }
    const double localDouble = static_cast<double>(local);
    double global = 0.0;
    MPI_Allreduce(&localDouble, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global;
}

int uniqueNodeCount(const MPI_Comm& comm)
{
    int size = 1;
    MPI_Comm_size(comm, &size);
    char localName[MPI_MAX_PROCESSOR_NAME] = {};
    int localLength = 0;
    MPI_Get_processor_name(localName, &localLength);
    std::vector<char> names(static_cast<std::size_t>(size) *
                            MPI_MAX_PROCESSOR_NAME, '\0');
    MPI_Allgather(localName, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  names.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, comm);
    std::set<std::string> unique;
    for(int rank = 0; rank < size; ++rank)
    {
        unique.insert(std::string(names.data() +
            static_cast<std::ptrdiff_t>(rank) * MPI_MAX_PROCESSOR_NAME));
    }
    return static_cast<int>(unique.size());
}

void updateTiming(double localSeconds,
                  double& bestMaximum,
                  double& sumMaximum,
                  const MPI_Comm& comm)
{
    double maximum = 0.0;
    MPI_Allreduce(&localSeconds, &maximum, 1, MPI_DOUBLE, MPI_MAX, comm);
    bestMaximum = std::min(bestMaximum, maximum);
    sumMaximum += maximum;
}

void accumulateFmmProfile(const FmmSolveStats& stats,
                          FmmProfile& profile,
                          const MPI_Comm& comm)
{
    int size = 1;
    MPI_Comm_size(comm, &size);

    const std::array<double, kFmmTimingMetricCount> localTimings = {{
        stats.buildSeconds,
        stats.upwardSeconds,
        stats.processUpwardSeconds,
        stats.processInteractionSeconds,
        stats.processDownwardSeconds,
        stats.letPlanSeconds,
        stats.letExecuteSeconds,
        stats.letExchangeSeconds,
        stats.letPreparationSeconds,
        stats.letPayloadPlanningSeconds,
        stats.letPayloadPackingSeconds,
        stats.letPayloadFlattenSeconds,
        stats.letCountExchangeSeconds,
        stats.letReceiveSetupSeconds,
        stats.letPayloadLaunchSeconds,
        stats.letPayloadReleaseSeconds,
        stats.letPayloadLifetimeSeconds,
        stats.letResidualWaitSeconds,
        stats.letValidationSeconds,
        stats.letDecodeSeconds,
        stats.letM2LSeconds,
        stats.letP2PSeconds,
        stats.localTraversalSeconds,
        stats.interactionSeconds,
        stats.downwardSeconds,
        stats.totalSeconds
    }};
    std::array<double, kFmmTimingMetricCount> minimumTimings{};
    std::array<double, kFmmTimingMetricCount> summedTimings{};
    std::array<double, kFmmTimingMetricCount> maximumTimings{};
    MPI_Allreduce(localTimings.data(), minimumTimings.data(),
                  static_cast<int>(localTimings.size()), MPI_DOUBLE, MPI_MIN,
                  comm);
    MPI_Allreduce(localTimings.data(), summedTimings.data(),
                  static_cast<int>(localTimings.size()), MPI_DOUBLE, MPI_SUM,
                  comm);
    MPI_Allreduce(localTimings.data(), maximumTimings.data(),
                  static_cast<int>(localTimings.size()), MPI_DOUBLE, MPI_MAX,
                  comm);
    for(std::size_t i = 0; i < localTimings.size(); ++i)
    {
        profile.timings[i].add(
            minimumTimings[i], summedTimings[i] / static_cast<double>(size),
            maximumTimings[i]);
    }

    const std::uint64_t distributedM2L =
        stats.processM2LCount + stats.letM2LCount;
    const std::uint64_t localM2L = stats.m2lCount >= distributedM2L ?
        stats.m2lCount - distributedM2L : 0;
    const std::uint64_t localP2PPairs =
        stats.p2pPairCount >= stats.letP2PPairCount ?
        stats.p2pPairCount - stats.letP2PPairCount : 0;
    const std::array<unsigned long long, kFmmWorkMetricCount> localWork = {{
        static_cast<unsigned long long>(stats.particleCount),
        static_cast<unsigned long long>(stats.nodeCount),
        static_cast<unsigned long long>(stats.leafCount),
        static_cast<unsigned long long>(localM2L),
        static_cast<unsigned long long>(localP2PPairs),
        static_cast<unsigned long long>(stats.processM2LCount),
        static_cast<unsigned long long>(stats.letM2LCount),
        static_cast<unsigned long long>(stats.letP2PPairCount),
        static_cast<unsigned long long>(stats.bytesSent),
        static_cast<unsigned long long>(stats.bytesReceived),
        static_cast<unsigned long long>(stats.peakRemoteBytes),
        static_cast<unsigned long long>(stats.peakProcessBytes),
        static_cast<unsigned long long>(stats.localOperatorCacheHits),
        static_cast<unsigned long long>(stats.localOperatorCacheMisses),
        static_cast<unsigned long long>(stats.localOperatorCacheBypasses),
        static_cast<unsigned long long>(stats.letOperatorCacheHits),
        static_cast<unsigned long long>(stats.letOperatorCacheMisses),
        static_cast<unsigned long long>(stats.letOperatorCacheBypasses),
        static_cast<unsigned long long>(stats.letProgressCallCount),
        static_cast<unsigned long long>(stats.letProgressIncompleteCount),
        static_cast<unsigned long long>(stats.letCompletionProgressCall),
        static_cast<unsigned long long>(stats.letCompletedBeforeFinishCount)
    }};
    std::array<unsigned long long, kFmmWorkMetricCount> minimumWork{};
    std::array<unsigned long long, kFmmWorkMetricCount> summedWork{};
    std::array<unsigned long long, kFmmWorkMetricCount> maximumWork{};
    MPI_Allreduce(localWork.data(), minimumWork.data(),
                  static_cast<int>(localWork.size()), MPI_UNSIGNED_LONG_LONG,
                  MPI_MIN, comm);
    MPI_Allreduce(localWork.data(), summedWork.data(),
                  static_cast<int>(localWork.size()), MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, comm);
    MPI_Allreduce(localWork.data(), maximumWork.data(),
                  static_cast<int>(localWork.size()), MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, comm);
    for(std::size_t i = 0; i < localWork.size(); ++i)
    {
        profile.work[i].add(
            static_cast<double>(minimumWork[i]),
            static_cast<double>(summedWork[i]) / static_cast<double>(size),
            static_cast<double>(maximumWork[i]));
    }
}

void writeFmmProfile(std::ostream& output,
                     const char* mode,
                     const FmmProfile& profile)
{
    for(std::size_t i = 0; i < profile.timings.size(); ++i)
    {
        const RankMetricSummary summary = profile.timings[i].average();
        output << "profile " << mode << " timing "
               << kFmmTimingMetricNames[i] << " "
               << summary.minimum << " " << summary.mean << " "
               << summary.maximum << " " << summary.imbalance() << "\n";
    }
    for(std::size_t i = 0; i < profile.work.size(); ++i)
    {
        const RankMetricSummary summary = profile.work[i].average();
        output << "profile " << mode << " work "
               << kFmmWorkMetricNames[i] << " "
               << summary.minimum << " " << summary.mean << " "
               << summary.maximum << " " << summary.imbalance() << "\n";
    }
}

void accumulateFmmStats(const FmmSolveStats& stats,
                        SolverResult& result,
                        const MPI_Comm& comm)
{
    const unsigned long long localSums[10] = {
        static_cast<unsigned long long>(stats.bytesSent),
        static_cast<unsigned long long>(stats.bytesReceived),
        static_cast<unsigned long long>(stats.localOperatorCacheHits),
        static_cast<unsigned long long>(stats.localOperatorCacheMisses),
        static_cast<unsigned long long>(stats.localOperatorCacheBypasses),
        static_cast<unsigned long long>(stats.letOperatorCacheHits),
        static_cast<unsigned long long>(stats.letOperatorCacheMisses),
        static_cast<unsigned long long>(stats.letOperatorCacheBypasses),
        static_cast<unsigned long long>(stats.processOperatorCacheMisses),
        static_cast<unsigned long long>(stats.processOperatorCacheBypasses)};
    unsigned long long globalSums[10] = {};
    MPI_Allreduce(localSums, globalSums, 10, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, comm);

    const unsigned long long localMaxima[15] = {
        static_cast<unsigned long long>(stats.peakRemoteBytes),
        static_cast<unsigned long long>(stats.peakProcessBytes),
        static_cast<unsigned long long>(stats.bytesOwned),
        static_cast<unsigned long long>(stats.localTreeBytes),
        static_cast<unsigned long long>(stats.localMultipoleBytes),
        static_cast<unsigned long long>(stats.localLocalBytes),
        static_cast<unsigned long long>(stats.letPlanBytes),
        static_cast<unsigned long long>(stats.operatorCacheBytes),
        static_cast<unsigned long long>(stats.operatorCacheBudgetBytes),
        static_cast<unsigned long long>(stats.localOperatorCacheBytes),
        static_cast<unsigned long long>(stats.localOperatorCacheEntries),
        static_cast<unsigned long long>(stats.localOperatorCacheMaxEntries),
        static_cast<unsigned long long>(stats.letOperatorCacheBytes),
        static_cast<unsigned long long>(stats.letOperatorCacheEntries),
        static_cast<unsigned long long>(stats.letOperatorCacheMaxEntries)};
    unsigned long long globalMaxima[15] = {};
    MPI_Allreduce(localMaxima, globalMaxima, 15, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, comm);

    result.bytesSent = std::max(result.bytesSent,
        static_cast<std::uint64_t>(globalSums[0]));
    result.bytesReceived = std::max(result.bytesReceived,
        static_cast<std::uint64_t>(globalSums[1]));
    result.localOperatorCacheHits = std::max(result.localOperatorCacheHits,
        static_cast<std::uint64_t>(globalSums[2]));
    result.localOperatorCacheMisses = std::max(result.localOperatorCacheMisses,
        static_cast<std::uint64_t>(globalSums[3]));
    result.localOperatorCacheBypasses = std::max(result.localOperatorCacheBypasses,
        static_cast<std::uint64_t>(globalSums[4]));
    result.letOperatorCacheHits = std::max(result.letOperatorCacheHits,
        static_cast<std::uint64_t>(globalSums[5]));
    result.letOperatorCacheMisses = std::max(result.letOperatorCacheMisses,
        static_cast<std::uint64_t>(globalSums[6]));
    result.letOperatorCacheBypasses = std::max(result.letOperatorCacheBypasses,
        static_cast<std::uint64_t>(globalSums[7]));
    result.processOperatorCacheMisses = std::max(result.processOperatorCacheMisses,
        static_cast<std::uint64_t>(globalSums[8]));
    result.processOperatorCacheBypasses = std::max(result.processOperatorCacheBypasses,
        static_cast<std::uint64_t>(globalSums[9]));
    result.peakRemoteBytes = std::max(result.peakRemoteBytes,
        static_cast<std::uint64_t>(globalMaxima[0]));
    result.peakProcessBytes = std::max(result.peakProcessBytes,
        static_cast<std::uint64_t>(globalMaxima[1]));
    result.persistentBytes = std::max(result.persistentBytes,
        static_cast<std::uint64_t>(globalMaxima[2]));
    result.localTreeBytes = std::max(result.localTreeBytes,
        static_cast<std::uint64_t>(globalMaxima[3]));
    result.localMultipoleBytes = std::max(result.localMultipoleBytes,
        static_cast<std::uint64_t>(globalMaxima[4]));
    result.localLocalBytes = std::max(result.localLocalBytes,
        static_cast<std::uint64_t>(globalMaxima[5]));
    result.letPlanBytes = std::max(result.letPlanBytes,
        static_cast<std::uint64_t>(globalMaxima[6]));
    result.operatorCacheBytes = std::max(result.operatorCacheBytes,
        static_cast<std::uint64_t>(globalMaxima[7]));
    result.operatorCacheBudgetBytes = std::max(result.operatorCacheBudgetBytes,
        static_cast<std::uint64_t>(globalMaxima[8]));
    result.localOperatorCacheBytes = std::max(result.localOperatorCacheBytes,
        static_cast<std::uint64_t>(globalMaxima[9]));
    result.localOperatorCacheEntries = std::max(result.localOperatorCacheEntries,
        static_cast<std::uint64_t>(globalMaxima[10]));
    result.localOperatorCacheMaxEntries = std::max(result.localOperatorCacheMaxEntries,
        static_cast<std::uint64_t>(globalMaxima[11]));
    result.letOperatorCacheBytes = std::max(result.letOperatorCacheBytes,
        static_cast<std::uint64_t>(globalMaxima[12]));
    result.letOperatorCacheEntries = std::max(result.letOperatorCacheEntries,
        static_cast<std::uint64_t>(globalMaxima[13]));
    result.letOperatorCacheMaxEntries = std::max(result.letOperatorCacheMaxEntries,
        static_cast<std::uint64_t>(globalMaxima[14]));
}

SolverResult runFmm(const LocalParticles& local,
                    const ProbeReference& reference,
                    int repeats,
                    int expansionOrder,
                    double thetaCritical,
                    std::size_t leafCapacity,
                    bool warmOnly,
                    std::size_t maxRemoteBytes,
                    std::size_t maxOperatorCacheBytes,
                    const MPI_Comm& comm)
{
    FmmGravityOptions options;
    options.expansionOrder = expansionOrder;
    options.thetaCritical = thetaCritical;
    options.leafCapacity = leafCapacity;
    options.computePotential = false;
    options.validateFinite = true;
    FmmDistributedOptions distributed;
    options.maxOperatorCacheBytes = maxOperatorCacheBytes;
    distributed.maxRemoteBytes = maxRemoteBytes;

    SolverResult result;
    double coldSumMaximum = 0.0;
    const int coldRepeats = warmOnly ? 0 : repeats;
    for(int repeat = 0; repeat < coldRepeats; ++repeat)
    {
        reportStage("fmm_cold_repeat_" + std::to_string(repeat) +
                    "_begin", comm);
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        std::vector<Vector3D> acceleration;
        FmmSolveStats stats;
        {
            DistributedFmmGravityCalculator solver(options, distributed, comm);
            solver.solve(local.positions, local.masses, local.ids,
                         Vector3D(-1, -1, -1), Vector3D(1, 1, 1),
                         acceleration);
            stats = solver.stats();
        }
        const double localSeconds = MPI_Wtime() - start;
        updateTiming(localSeconds, result.bestMaxSeconds, coldSumMaximum, comm);
        accumulateFmmStats(stats, result, comm);
        accumulateFmmProfile(stats, result.coldProfile, comm);

        const int localFinite = finiteAcceleration(acceleration) ? 1 : 0;
        int globalFinite = 0;
        MPI_Allreduce(&localFinite, &globalFinite, 1, MPI_INT, MPI_LAND, comm);
        result.finite = result.finite && globalFinite != 0;
        if(repeat == 0)
        {
            const ProbeErrorStats probeErrors = probeScaledErrors(
                collectProbeAccelerations(local, acceleration, reference, comm),
                reference);
            result.probeScaledError = probeErrors.maximum;
            result.probeMeanScaledError = probeErrors.mean;
            result.checksum = accelerationChecksum(acceleration, local.ids, comm);
        }

        reportStage("fmm_cold_repeat_" + std::to_string(repeat) +
                    "_solved", comm);
        std::vector<Vector3D>().swap(acceleration);
        trimAllocator();
        reportStage("fmm_cold_repeat_" + std::to_string(repeat) +
                    "_end", comm);
    }
    if(coldRepeats > 0)
        result.meanMaxSeconds = coldSumMaximum /
                                static_cast<double>(coldRepeats);

    // Measure the production-relevant topology-reuse path separately.  The
    // setup solve populates the LET plan and the bounded operator cache; only
    // subsequent solves are included in the warm timing.
    reportStage("fmm_warm_setup_begin", comm);
    double warmSetupStart = 0.0;
    if(warmOnly)
    {
        MPI_Barrier(comm);
        warmSetupStart = MPI_Wtime();
    }
    DistributedFmmGravityCalculator warmSolver(options, distributed, comm);
    std::vector<Vector3D> warmAcceleration;
    warmSolver.solve(local.positions, local.masses, local.ids,
                     Vector3D(-1, -1, -1), Vector3D(1, 1, 1),
                     warmAcceleration);
    const double warmSetupLocalSeconds = warmOnly ?
        MPI_Wtime() - warmSetupStart : 0.0;
    const std::uint64_t warmEpoch = warmSolver.stats().topologyEpoch;
    const std::uint64_t warmRebuildCount =
        warmSolver.stats().topologyRebuildCount;
    accumulateFmmStats(warmSolver.stats(), result, comm);
    if(warmOnly)
    {
        double warmSetupSumMaximum = 0.0;
        updateTiming(warmSetupLocalSeconds, result.bestMaxSeconds,
                     warmSetupSumMaximum, comm);
        result.meanMaxSeconds = warmSetupSumMaximum;
        accumulateFmmProfile(warmSolver.stats(), result.coldProfile, comm);

        const int localFinite = finiteAcceleration(warmAcceleration) ? 1 : 0;
        int globalFinite = 0;
        MPI_Allreduce(&localFinite, &globalFinite, 1, MPI_INT, MPI_LAND, comm);
        result.finite = result.finite && globalFinite != 0;

        const ProbeErrorStats probeErrors = probeScaledErrors(
            collectProbeAccelerations(local, warmAcceleration, reference, comm),
            reference);
        result.probeScaledError = probeErrors.maximum;
        result.probeMeanScaledError = probeErrors.mean;
        result.checksum = accelerationChecksum(warmAcceleration, local.ids, comm);
    }
    reportStage("fmm_warm_setup_solved", comm);

    double warmSumMaximum = 0.0;
    for(int repeat = 0; repeat < repeats; ++repeat)
    {
        reportStage("fmm_warm_repeat_" + std::to_string(repeat) +
                    "_begin", comm);
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        warmSolver.solve(local.positions, local.masses, local.ids,
                         Vector3D(-1, -1, -1), Vector3D(1, 1, 1),
                         warmAcceleration);
        const double localSeconds = MPI_Wtime() - start;
        updateTiming(localSeconds, result.warmBestMaxSeconds,
                     warmSumMaximum, comm);
        const FmmSolveStats& stats = warmSolver.stats();
        result.topologyReused = result.topologyReused &&
            stats.topologyEpoch == warmEpoch &&
            stats.topologyRebuildCount == warmRebuildCount;
        accumulateFmmStats(stats, result, comm);
        accumulateFmmProfile(stats, result.warmProfile, comm);

        const int localFinite = finiteAcceleration(warmAcceleration) ? 1 : 0;
        int globalFinite = 0;
        MPI_Allreduce(&localFinite, &globalFinite, 1, MPI_INT, MPI_LAND, comm);
        result.finite = result.finite && globalFinite != 0;
        reportStage("fmm_warm_repeat_" + std::to_string(repeat) +
                    "_solved", comm);
    }
    result.warmMeanMaxSeconds =
        warmSumMaximum / static_cast<double>(repeats);
    const int localTopologyReused = result.topologyReused ? 1 : 0;
    int globalTopologyReused = 0;
    MPI_Allreduce(&localTopologyReused, &globalTopologyReused, 1, MPI_INT,
                  MPI_LAND, comm);
    result.topologyReused = globalTopologyReused != 0;
    std::vector<Vector3D>().swap(warmAcceleration);
    trimAllocator();
    reportStage("fmm_warm_complete", comm);
    return result;
}

SolverResult runQuadrupole(const LocalParticles& local,
                           const ProbeReference& reference,
                           int repeats,
                           double theta,
                           const MPI_Comm& comm)
{
    SolverResult result;
    double sumMaximum = 0.0;
    for(int repeat = 0; repeat < repeats; ++repeat)
    {
        reportStage("quadrupole_repeat_" + std::to_string(repeat) +
                    "_begin", comm);
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        std::vector<Vector3D> acceleration;
        double localWalkSeconds = 0.0;
        {
            DistributedGravityCalculator solver(
                local.positions, local.masses,
                Vector3D(-1, -1, -1), Vector3D(1, 1, 1),
                theta, true, comm);
            acceleration = solver.getAcceleration(local.positions);
            localWalkSeconds = solver.getWalkTime();
        }
        const double localSeconds = MPI_Wtime() - start;
        updateTiming(localSeconds, result.bestMaxSeconds, sumMaximum, comm);
        double maximumWalk = 0.0;
        MPI_Allreduce(&localWalkSeconds, &maximumWalk, 1, MPI_DOUBLE,
                      MPI_MAX, comm);
        result.walkMaxSeconds = std::min(result.walkMaxSeconds, maximumWalk);

        const int localFinite = finiteAcceleration(acceleration) ? 1 : 0;
        int globalFinite = 0;
        MPI_Allreduce(&localFinite, &globalFinite, 1, MPI_INT, MPI_LAND, comm);
        result.finite = result.finite && globalFinite != 0;
        if(repeat == 0)
        {
            const ProbeErrorStats probeErrors = probeScaledErrors(
                collectProbeAccelerations(local, acceleration, reference, comm),
                reference);
            result.probeScaledError = probeErrors.maximum;
            result.probeMeanScaledError = probeErrors.mean;
            result.checksum = accelerationChecksum(acceleration, local.ids, comm);
        }
        reportStage("quadrupole_repeat_" + std::to_string(repeat) +
                    "_solved", comm);
        std::vector<Vector3D>().swap(acceleration);
        trimAllocator();
        reportStage("quadrupole_repeat_" + std::to_string(repeat) +
                    "_end", comm);
    }
    result.meanMaxSeconds = sumMaximum / static_cast<double>(repeats);
    return result;
}
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int returnCode = 1;
    try
    {
        const Options options = parseOptions(argc, argv);
        const int nodes = uniqueNodeCount(MPI_COMM_WORLD);
        reportStage("start", MPI_COMM_WORLD);
        const LocalParticles local = makeLocalParticles(
            options.globalParticles, rank, size, MPI_COMM_WORLD);
        reportStage("particles_ready", MPI_COMM_WORLD);
        const ProbeReference reference = computeProbeReference(
            local, options.globalParticles, MPI_COMM_WORLD);
        reportStage("reference_ready", MPI_COMM_WORLD);

        const unsigned long long localCount =
            static_cast<unsigned long long>(local.positions.size());
        unsigned long long minimumLocal = 0;
        unsigned long long maximumLocal = 0;
        MPI_Allreduce(&localCount, &minimumLocal, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localCount, &maximumLocal, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX, MPI_COMM_WORLD);

        const SolverResult fmm = runFmm(
            local, reference, options.repeats, options.fmmOrder,
            options.fmmTheta, options.fmmLeafCapacity, options.warmOnly,
            options.fmmMaxRemoteBytes, options.fmmMaxOperatorCacheBytes,
            MPI_COMM_WORLD);
        reportStage("fmm_complete", MPI_COMM_WORLD);
        trimAllocator();
        reportStage("after_fmm_trim", MPI_COMM_WORLD);
        const SolverResult quadrupole = runQuadrupole(
            local, reference, options.repeats, options.quadrupoleTheta,
            MPI_COMM_WORLD);
        reportStage("quadrupole_complete", MPI_COMM_WORLD);

        const bool timingFinite =
            fmm.bestMaxSeconds > 0.0 && std::isfinite(fmm.bestMaxSeconds) &&
            fmm.meanMaxSeconds > 0.0 && std::isfinite(fmm.meanMaxSeconds) &&
            fmm.warmBestMaxSeconds > 0.0 &&
            std::isfinite(fmm.warmBestMaxSeconds) &&
            fmm.warmMeanMaxSeconds > 0.0 &&
            std::isfinite(fmm.warmMeanMaxSeconds) &&
            quadrupole.bestMaxSeconds > 0.0 &&
            std::isfinite(quadrupole.bestMaxSeconds) &&
            quadrupole.meanMaxSeconds > 0.0 &&
            std::isfinite(quadrupole.meanMaxSeconds) &&
            quadrupole.walkMaxSeconds >= 0.0 &&
            std::isfinite(quadrupole.walkMaxSeconds);
        const bool accuracyPass =
            fmm.probeScaledError <= options.fmmMaxErrorTarget &&
            quadrupole.probeScaledError < 5e-2 &&
            fmm.probeMeanScaledError >= 0.0 &&
            fmm.probeMeanScaledError <= options.fmmMeanErrorTarget &&
            fmm.probeMeanScaledError <= fmm.probeScaledError &&
            quadrupole.probeMeanScaledError >= 0.0 &&
            quadrupole.probeMeanScaledError <= quadrupole.probeScaledError;
        const bool placementPass =
            nodes == options.expectedNodes &&
            size == options.expectedNodes * options.expectedRanksPerNode;
        const bool finite = timingFinite && fmm.finite && quadrupole.finite &&
            std::isfinite(fmm.probeScaledError) &&
            std::isfinite(quadrupole.probeScaledError) &&
            std::isfinite(fmm.probeMeanScaledError) &&
            std::isfinite(quadrupole.probeMeanScaledError) &&
            std::isfinite(fmm.checksum) &&
            std::isfinite(quadrupole.checksum);
        const bool passed = placementPass && accuracyPass && finite &&
                            fmm.topologyReused &&
                            fmm.operatorCacheBytes <=
                                options.fmmMaxOperatorCacheBytes &&
                            fmm.operatorCacheBudgetBytes ==
                                options.fmmMaxOperatorCacheBytes;

        if(rank == 0)
        {
            const double speedup = quadrupole.bestMaxSeconds /
                                   fmm.bestMaxSeconds;
            const double fmmThroughput =
                static_cast<double>(options.globalParticles) /
                fmm.bestMaxSeconds;
            const double quadrupoleThroughput =
                static_cast<double>(options.globalParticles) /
                quadrupole.bestMaxSeconds;

            std::ofstream output(options.outputPath.c_str());
            if(!output)
                throw UniversalError(
                    "fmm_mpi_scaling_benchmark: could not open output file");
            output << std::scientific << std::setprecision(16);
            output << "fmm_config order " << options.fmmOrder
                   << " theta " << options.fmmTheta
                   << " quadrupole_theta " << options.quadrupoleTheta
                   << " leaf_capacity " << options.fmmLeafCapacity
                   << " mean_error_target " << options.fmmMeanErrorTarget
                   << " max_error_target " << options.fmmMaxErrorTarget
                   << " warm_only " << (options.warmOnly ? 1 : 0)
                   << " coefficient_count "
                   << fmmTaylorCoefficientCount(options.fmmOrder)
                   << " m2l_term_count "
                   << fmmM2LTermCount(options.fmmOrder) << "\n";
            output << "columns particles expected_nodes ranks unique_nodes "
                   << "ranks_per_node repeats local_particles_min "
                   << "local_particles_max "
                   << "fmm_best_max_seconds fmm_mean_max_seconds "
                   << "quadrupole_best_max_seconds quadrupole_mean_max_seconds "
                   << "fmm_probe_scaled_error quadrupole_probe_scaled_error "
                   << "quadrupole_over_fmm_speedup fmm_particles_per_second "
                   << "quadrupole_particles_per_second fmm_bytes_sent "
                   << "fmm_bytes_received fmm_peak_remote_bytes "
                   << "fmm_peak_process_bytes quadrupole_walk_max_seconds "
                   << "fmm_checksum quadrupole_checksum finite run_pass "
                   << "fmm_warm_best_max_seconds fmm_warm_mean_max_seconds "
                   << "fmm_cold_over_warm_speedup fmm_persistent_bytes "
                   << "fmm_local_tree_bytes fmm_local_multipole_bytes "
                   << "fmm_local_local_bytes fmm_let_plan_bytes "
                   << "fmm_operator_cache_bytes "
                   << "fmm_operator_cache_budget_bytes "
                   << "fmm_local_operator_cache_bytes "
                   << "fmm_local_operator_cache_entries "
                   << "fmm_local_operator_cache_max_entries "
                   << "fmm_local_operator_cache_hits "
                   << "fmm_local_operator_cache_misses "
                   << "fmm_local_operator_cache_bypasses "
                   << "fmm_let_operator_cache_bytes "
                   << "fmm_let_operator_cache_entries "
                   << "fmm_let_operator_cache_max_entries "
                   << "fmm_let_operator_cache_hits "
                   << "fmm_let_operator_cache_misses "
                   << "fmm_let_operator_cache_bypasses "
                   << "fmm_process_operator_cache_misses "
                   << "fmm_process_operator_cache_bypasses "
                   << "fmm_topology_reused probe_count "
                   << "fmm_probe_mean_scaled_error "
                   << "quadrupole_probe_mean_scaled_error\n";
            output << "row " << options.globalParticles << " "
                   << options.expectedNodes << " " << size << " " << nodes
                   << " " << options.expectedRanksPerNode << " "
                   << options.repeats << " " << minimumLocal << " "
                   << maximumLocal << " " << fmm.bestMaxSeconds << " "
                   << fmm.meanMaxSeconds << " "
                   << quadrupole.bestMaxSeconds << " "
                   << quadrupole.meanMaxSeconds << " "
                   << fmm.probeScaledError << " "
                   << quadrupole.probeScaledError << " " << speedup << " "
                   << fmmThroughput << " " << quadrupoleThroughput << " "
                   << fmm.bytesSent << " " << fmm.bytesReceived << " "
                   << fmm.peakRemoteBytes << " " << fmm.peakProcessBytes
                   << " " << quadrupole.walkMaxSeconds << " "
                   << fmm.checksum << " " << quadrupole.checksum << " "
                   << (finite ? 1 : 0) << " " << (passed ? 1 : 0) << " "
                   << fmm.warmBestMaxSeconds << " "
                   << fmm.warmMeanMaxSeconds << " "
                   << (fmm.bestMaxSeconds / fmm.warmBestMaxSeconds) << " "
                   << fmm.persistentBytes << " " << fmm.localTreeBytes << " "
                   << fmm.localMultipoleBytes << " " << fmm.localLocalBytes
                   << " " << fmm.letPlanBytes << " "
                   << fmm.operatorCacheBytes << " "
                   << fmm.operatorCacheBudgetBytes << " "
                   << fmm.localOperatorCacheBytes << " "
                   << fmm.localOperatorCacheEntries << " "
                   << fmm.localOperatorCacheMaxEntries << " "
                   << fmm.localOperatorCacheHits << " "
                   << fmm.localOperatorCacheMisses << " "
                   << fmm.localOperatorCacheBypasses << " "
                   << fmm.letOperatorCacheBytes << " "
                   << fmm.letOperatorCacheEntries << " "
                   << fmm.letOperatorCacheMaxEntries << " "
                   << fmm.letOperatorCacheHits << " "
                   << fmm.letOperatorCacheMisses << " "
                   << fmm.letOperatorCacheBypasses << " "
                   << fmm.processOperatorCacheMisses << " "
                   << fmm.processOperatorCacheBypasses << " "
                   << (fmm.topologyReused ? 1 : 0) << " "
                   << kProbeCount << " "
                   << fmm.probeMeanScaledError << " "
                   << quadrupole.probeMeanScaledError << "\n";
            output << "profile_columns mode category metric rank_min rank_mean "
                   << "rank_max max_over_mean\n";
            writeFmmProfile(output, "cold", fmm.coldProfile);
            writeFmmProfile(output, "warm", fmm.warmProfile);
            output << "pass " << (passed ? 1 : 0) << "\n";

            std::cout << "fmm_mpi_scaling_benchmark particles="
                      << options.globalParticles << " nodes=" << nodes
                      << " ranks=" << size
                      << " ranks_per_node=" << options.expectedRanksPerNode
                      << " fmm_order=" << options.fmmOrder
                      << " fmm_theta=" << options.fmmTheta
                      << " quadrupole_theta=" << options.quadrupoleTheta
                      << " fmm_leaf_capacity=" << options.fmmLeafCapacity
                      << " fmm_mean_error_target="
                      << options.fmmMeanErrorTarget
                      << " fmm_max_error_target="
                      << options.fmmMaxErrorTarget
                      << " warm_only=" << options.warmOnly
                      << " fmm_seconds=" << fmm.bestMaxSeconds
                      << " quadrupole_seconds="
                      << quadrupole.bestMaxSeconds
                      << " probe_count=" << kProbeCount
                      << " fmm_probe_max_error=" << fmm.probeScaledError
                      << " fmm_probe_mean_error="
                      << fmm.probeMeanScaledError
                      << " quadrupole_probe_max_error="
                      << quadrupole.probeScaledError
                      << " quadrupole_probe_mean_error="
                      << quadrupole.probeMeanScaledError
                      << " speedup=" << speedup
                      << " fmm_warm_seconds=" << fmm.warmBestMaxSeconds
                      << " operator_cache_mib="
                      << (static_cast<double>(fmm.operatorCacheBytes) /
                          static_cast<double>(kMebibyte))
                      << " local_cache_bypasses="
                      << fmm.localOperatorCacheBypasses
                      << " let_cache_bypasses="
                      << fmm.letOperatorCacheBypasses
                      << " pass=" << passed << std::endl;
        }
        returnCode = passed ? 0 : 1;
    }
    catch(const UniversalError& error)
    {
        std::cerr << "rank " << rank << ": ";
        reportError(error);
        MPI_Abort(MPI_COMM_WORLD, 92);
        returnCode = 2;
    }
    catch(const std::exception& error)
    {
        std::cerr << "rank " << rank << ": " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 93);
        returnCode = 3;
    }

    MPI_Finalize();
    return returnCode;
}
