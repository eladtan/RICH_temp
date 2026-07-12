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

#include <mpi.h>

#include "source/3D/gravity/DistributedGravityCalculator.hpp"
#include "source/3D/gravity/fmm/mpi/DistributedFmmGravityCalculator.hpp"
#include "source/misc/universal_error.hpp"

namespace
{
constexpr std::uint64_t kVirtualBins = 4096;
constexpr std::size_t kProbeCount = 8;

struct Options
{
    std::uint64_t globalParticles = 0;
    int expectedNodes = 0;
    int expectedRanksPerNode = 0;
    int repeats = 2;
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
    std::array<Vector3D, kProbeCount> acceleration{};
    std::array<double, kProbeCount> forceScale{};
};

struct SolverResult
{
    double bestMaxSeconds = std::numeric_limits<double>::infinity();
    double meanMaxSeconds = 0.0;
    double walkMaxSeconds = std::numeric_limits<double>::infinity();
    double probeScaledError = std::numeric_limits<double>::infinity();
    double checksum = 0.0;
    std::uint64_t bytesSent = 0;
    std::uint64_t bytesReceived = 0;
    std::uint64_t peakRemoteBytes = 0;
    std::uint64_t peakProcessBytes = 0;
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

Vector3D positionForId(std::uint64_t id)
{
    const std::uint64_t bin = id % kVirtualBins;
    const std::uint64_t ordinal = id / kVirtualBins;
    const double withinBin = 0.1 + 0.8 * radicalInverse(ordinal + 1, 2);
    const double x = -0.95 + 1.9 *
        (static_cast<double>(bin) + withinBin) /
        static_cast<double>(kVirtualBins);
    const double y = 1.9 * radicalInverse(id + 1, 3) - 0.95;
    const double z = 1.9 * radicalInverse(id + 1, 5) - 0.95;
    return Vector3D(x, y, z);
}

std::uint64_t massUnitsForId(std::uint64_t id)
{
    // Normalizing integer mass units by their exact global sum makes the
    // particle masses independent of MPI rank count.
    return 101u + 2u * ((37u * id + 11u) % 101u);
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
        else if(arg == "--output" && i + 1 < argc)
            result.outputPath = argv[++i];
        else
            throw UniversalError("fmm_mpi_scaling_benchmark: invalid command line");
    }
    if(result.globalParticles == 0 || result.expectedNodes <= 0 ||
       result.expectedRanksPerNode <= 0 || result.repeats <= 0 ||
       result.outputPath.empty())
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
    std::array<Vector3D, kProbeCount> result{};
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

double probeScaledError(
    const std::array<Vector3D, kProbeCount>& calculated,
    const ProbeReference& reference)
{
    double maximum = 0.0;
    for(std::size_t i = 0; i < kProbeCount; ++i)
    {
        maximum = std::max(maximum,
            norm(calculated[i] - reference.acceleration[i]) /
            std::max(reference.forceScale[i], 1e-30));
    }
    return maximum;
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

SolverResult runFmm(const LocalParticles& local,
                    const ProbeReference& reference,
                    int repeats,
                    const MPI_Comm& comm)
{
    SolverResult result;
    double sumMaximum = 0.0;
    for(int repeat = 0; repeat < repeats; ++repeat)
    {
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        std::vector<Vector3D> acceleration;
        FmmSolveStats stats;
        {
            FmmGravityOptions options;
            options.expansionOrder = 4;
            options.thetaCritical = 0.5;
            options.leafCapacity = 32;
            options.computePotential = false;
            options.validateFinite = true;
            FmmDistributedOptions distributed;
            distributed.maxRemoteBytes =
                static_cast<std::size_t>(2) * 1024 * 1024 * 1024;
            DistributedFmmGravityCalculator solver(options, distributed, comm);
            solver.solve(local.positions, local.masses, local.ids,
                         Vector3D(-1, -1, -1), Vector3D(1, 1, 1),
                         acceleration);
            stats = solver.stats();
        }
        const double localSeconds = MPI_Wtime() - start;
        updateTiming(localSeconds, result.bestMaxSeconds, sumMaximum, comm);

        const int localFinite = finiteAcceleration(acceleration) ? 1 : 0;
        int globalFinite = 0;
        MPI_Allreduce(&localFinite, &globalFinite, 1, MPI_INT, MPI_LAND, comm);
        result.finite = result.finite && globalFinite != 0;
        if(repeat == 0)
        {
            result.probeScaledError = probeScaledError(
                collectProbeAccelerations(local, acceleration, reference, comm),
                reference);
            result.checksum = accelerationChecksum(acceleration, local.ids, comm);
        }

        unsigned long long localSent =
            static_cast<unsigned long long>(stats.bytesSent);
        unsigned long long localReceived =
            static_cast<unsigned long long>(stats.bytesReceived);
        unsigned long long localPeakRemote =
            static_cast<unsigned long long>(stats.peakRemoteBytes);
        unsigned long long localPeakProcess =
            static_cast<unsigned long long>(stats.peakProcessBytes);
        unsigned long long globalSent = 0;
        unsigned long long globalReceived = 0;
        unsigned long long globalPeakRemote = 0;
        unsigned long long globalPeakProcess = 0;
        MPI_Allreduce(&localSent, &globalSent, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM, comm);
        MPI_Allreduce(&localReceived, &globalReceived, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
        MPI_Allreduce(&localPeakRemote, &globalPeakRemote, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
        MPI_Allreduce(&localPeakProcess, &globalPeakProcess, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
        result.bytesSent = std::max(result.bytesSent,
            static_cast<std::uint64_t>(globalSent));
        result.bytesReceived = std::max(result.bytesReceived,
            static_cast<std::uint64_t>(globalReceived));
        result.peakRemoteBytes = std::max(result.peakRemoteBytes,
            static_cast<std::uint64_t>(globalPeakRemote));
        result.peakProcessBytes = std::max(result.peakProcessBytes,
            static_cast<std::uint64_t>(globalPeakProcess));
    }
    result.meanMaxSeconds = sumMaximum / static_cast<double>(repeats);
    return result;
}

SolverResult runQuadrupole(const LocalParticles& local,
                           const ProbeReference& reference,
                           int repeats,
                           const MPI_Comm& comm)
{
    SolverResult result;
    double sumMaximum = 0.0;
    for(int repeat = 0; repeat < repeats; ++repeat)
    {
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        std::vector<Vector3D> acceleration;
        double localWalkSeconds = 0.0;
        {
            DistributedGravityCalculator solver(
                local.positions, local.masses,
                Vector3D(-1, -1, -1), Vector3D(1, 1, 1),
                0.5, true, comm);
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
            result.probeScaledError = probeScaledError(
                collectProbeAccelerations(local, acceleration, reference, comm),
                reference);
            result.checksum = accelerationChecksum(acceleration, local.ids, comm);
        }
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
        const LocalParticles local = makeLocalParticles(
            options.globalParticles, rank, size, MPI_COMM_WORLD);
        const ProbeReference reference = computeProbeReference(
            local, options.globalParticles, MPI_COMM_WORLD);

        const unsigned long long localCount =
            static_cast<unsigned long long>(local.positions.size());
        unsigned long long minimumLocal = 0;
        unsigned long long maximumLocal = 0;
        MPI_Allreduce(&localCount, &minimumLocal, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localCount, &maximumLocal, 1, MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX, MPI_COMM_WORLD);

        const SolverResult fmm = runFmm(local, reference, options.repeats,
                                        MPI_COMM_WORLD);
        const SolverResult quadrupole = runQuadrupole(
            local, reference, options.repeats, MPI_COMM_WORLD);

        const bool timingFinite =
            fmm.bestMaxSeconds > 0.0 && std::isfinite(fmm.bestMaxSeconds) &&
            fmm.meanMaxSeconds > 0.0 && std::isfinite(fmm.meanMaxSeconds) &&
            quadrupole.bestMaxSeconds > 0.0 &&
            std::isfinite(quadrupole.bestMaxSeconds) &&
            quadrupole.meanMaxSeconds > 0.0 &&
            std::isfinite(quadrupole.meanMaxSeconds) &&
            quadrupole.walkMaxSeconds >= 0.0 &&
            std::isfinite(quadrupole.walkMaxSeconds);
        const bool accuracyPass =
            fmm.probeScaledError < 5e-3 &&
            quadrupole.probeScaledError < 5e-2;
        const bool placementPass =
            nodes == options.expectedNodes &&
            size == options.expectedNodes * options.expectedRanksPerNode;
        const bool finite = timingFinite && fmm.finite && quadrupole.finite &&
            std::isfinite(fmm.probeScaledError) &&
            std::isfinite(quadrupole.probeScaledError) &&
            std::isfinite(fmm.checksum) &&
            std::isfinite(quadrupole.checksum);
        const bool passed = placementPass && accuracyPass && finite;

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
                   << "fmm_checksum quadrupole_checksum finite run_pass\n";
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
                   << (finite ? 1 : 0) << " " << (passed ? 1 : 0) << "\n";
            output << "pass " << (passed ? 1 : 0) << "\n";

            std::cout << "fmm_mpi_scaling_benchmark particles="
                      << options.globalParticles << " nodes=" << nodes
                      << " ranks=" << size
                      << " ranks_per_node=" << options.expectedRanksPerNode
                      << " fmm_seconds=" << fmm.bestMaxSeconds
                      << " quadrupole_seconds="
                      << quadrupole.bestMaxSeconds
                      << " fmm_probe_error=" << fmm.probeScaledError
                      << " quadrupole_probe_error="
                      << quadrupole.probeScaledError
                      << " speedup=" << speedup
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
