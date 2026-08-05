#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <mpi.h>

#include "source/3D/gravity/fmm/mpi/DistributedFmmGravityCalculator.hpp"

namespace
{
struct Body
{
    Vector3D position;
    double mass = 0.0;
    std::uint64_t id = 0;
    int owner = -1;
    std::size_t localIndex = 0;
};

std::vector<Body> bodiesForRank(int rank, int size, int step)
{
    if(size >= 4 && rank == size - 1)
        return std::vector<Body>();
    const int activeRanks = size >= 4 ? size - 1 : size;
    std::vector<Body> bodies;
    const int count = 6 + ((step == 1 && rank == 0) ? 1 : 0);
    for(int i = 0; i < count; ++i)
    {
        const int clump = i % 2;
        const double rankCoordinate =
            -0.82 + 1.64 * (static_cast<double>(rank) + 0.5) /
                std::max(1, activeRanks);
        Body body;
        body.position = Vector3D(
            clump == 0 ? rankCoordinate : -rankCoordinate,
            -0.7 + 1.4 * (static_cast<double>((rank + 2 * i) % 11) / 10.0),
            -0.6 + 1.2 * (static_cast<double>((3 * rank + i) % 13) / 12.0));
        body.position.x += 0.012 * static_cast<double>((i % 3) - 1);
        body.position.y += 0.009 * static_cast<double>((i % 5) - 2);
        if(step == 1 && rank == 1 && i == 0)
            body.position.z += 0.08;
        body.mass = 0.45 + 0.03 * static_cast<double>(rank + 1) +
                    0.02 * static_cast<double>(i + 1);
        body.id = static_cast<std::uint64_t>(1000 * rank + i);
        body.owner = rank;
        body.localIndex = bodies.size();
        bodies.push_back(body);
    }
    return bodies;
}

std::vector<Body> allBodies(int size, int step)
{
    std::vector<Body> result;
    for(int rank = 0; rank < size; ++rank)
    {
        const std::vector<Body> local = bodiesForRank(rank, size, step);
        result.insert(result.end(), local.begin(), local.end());
    }
    return result;
}

Vector3D directAcceleration(const Body& target,
                            const std::vector<Body>& all)
{
    Vector3D acceleration;
    for(const Body& source : all)
    {
        if(source.owner == target.owner &&
           source.localIndex == target.localIndex)
            continue;
        const Vector3D delta = target.position - source.position;
        const double r2 = delta.x * delta.x + delta.y * delta.y +
                          delta.z * delta.z;
        if(r2 == 0.0)
            continue;
        const double invR = 1.0 / std::sqrt(r2);
        acceleration -= source.mass * delta * (invR * invR * invR);
    }
    return acceleration;
}

double directPotential(const Body& target,
                       const std::vector<Body>& all)
{
    double potential = 0.0;
    for(const Body& source : all)
    {
        if(source.owner == target.owner &&
           source.localIndex == target.localIndex)
            continue;
        const Vector3D delta = target.position - source.position;
        const double r2 = delta.x * delta.x + delta.y * delta.y +
                          delta.z * delta.z;
        if(r2 != 0.0)
            potential += source.mass / std::sqrt(r2);
    }
    return potential;
}

double norm(const Vector3D& value)
{
    return std::sqrt(value.x * value.x + value.y * value.y +
                     value.z * value.z);
}

void unpack(const std::vector<Body>& bodies,
            std::vector<Vector3D>& positions,
            std::vector<double>& masses,
            std::vector<std::uint64_t>& ids)
{
    positions.clear();
    masses.clear();
    ids.clear();
    for(const Body& body : bodies)
    {
        positions.push_back(body.position);
        masses.push_back(body.mass);
        ids.push_back(body.id);
    }
}

double solveError(DistributedFmmGravityCalculator& solver,
                  const std::vector<Body>& localBodies,
                  const std::vector<Body>& globalBodies,
                  FmmSolveStats& stats,
                  double& relativeAccelerationErrorSum,
                  unsigned long long& relativeAccelerationErrorCount)
{
    std::vector<Vector3D> positions;
    std::vector<double> masses;
    std::vector<std::uint64_t> ids;
    unpack(localBodies, positions, masses, ids);
    std::vector<Vector3D> acceleration;
    std::vector<double> potential;
    solver.solve(positions, masses, ids,
                 Vector3D(-1.0, -1.0, -1.0),
                 Vector3D(1.0, 1.0, 1.0),
                 acceleration, &potential);
    stats = solver.stats();

    double maximum = 0.0;
    for(std::size_t i = 0; i < localBodies.size(); ++i)
    {
        const Vector3D reference =
            directAcceleration(localBodies[i], globalBodies);
        const double relativeAccelerationError =
            norm(acceleration[i] - reference) /
            std::max(1.0, norm(reference));
        maximum = std::max(maximum, relativeAccelerationError);
        relativeAccelerationErrorSum += relativeAccelerationError;
        ++relativeAccelerationErrorCount;
        const double referencePotential =
            directPotential(localBodies[i], globalBodies);
        maximum = std::max(maximum,
            std::abs(potential[i] - referencePotential) /
            std::max(1.0, std::abs(referencePotential)));
    }
    return maximum;
}
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    FmmGravityOptions options;
    options.expansionOrder = 5;
    options.thetaCritical = 0.35;
    options.leafCapacity = 2;
    options.computePotential = true;
    options.validateFinite = true;

    FmmDistributedOptions distributed;
    distributed.enablePatchForest = true;
    distributed.minimumPatchLevel = 2;
    distributed.maximumPatchLevel = 5;
    distributed.targetParticlesPerPatch = 3;
    distributed.maxLocalPatchCount = 4096;
    distributed.maxTargetPatchesPerWave = 1;
    distributed.maxLetWaveBytes = 512;
    distributed.maxRemoteBytes = 64u * 1024u * 1024u;
    distributed.maxReplicatedDescriptorBytes = 16u * 1024u * 1024u;
    distributed.enableLeafM2P = true;
    distributed.persistentLocalTreeTopology = false;

    DistributedFmmGravityCalculator solver(options, distributed);
    double localMaximumError = 0.0;
    double localRelativeAccelerationErrorSum = 0.0;
    unsigned long long localRelativeAccelerationErrorCount = 0;
    std::size_t maximumWaveCount = 0;
    std::size_t maximumProcessNodes = 0;
    bool rootMassValid = true;
    for(int step = 0; step < 2; ++step)
    {
        const std::vector<Body> local = bodiesForRank(rank, size, step);
        const std::vector<Body> global = allBodies(size, step);
        FmmSolveStats stats;
        localMaximumError = std::max(
            localMaximumError, solveError(
                solver, local, global, stats,
                localRelativeAccelerationErrorSum,
                localRelativeAccelerationErrorCount));
        maximumWaveCount = std::max(maximumWaveCount, stats.letWaveCount);
        maximumProcessNodes = std::max(maximumProcessNodes,
                                       stats.processNodeCount);
        rootMassValid = rootMassValid &&
            std::isfinite(stats.rootMass) &&
            std::abs(stats.rootMass - stats.totalMass) <=
                1.0e-10 * std::max(1.0, std::abs(stats.totalMass));
    }

    double globalMaximumError = 0.0;
    MPI_Allreduce(&localMaximumError, &globalMaximumError, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    double globalRelativeAccelerationErrorSum = 0.0;
    unsigned long long globalRelativeAccelerationErrorCount = 0;
    MPI_Allreduce(&localRelativeAccelerationErrorSum,
                  &globalRelativeAccelerationErrorSum, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localRelativeAccelerationErrorCount,
                  &globalRelativeAccelerationErrorCount, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    const double globalMeanRelativeAccelerationError =
        globalRelativeAccelerationErrorCount == 0 ? 0.0 :
        globalRelativeAccelerationErrorSum /
            static_cast<double>(globalRelativeAccelerationErrorCount);
    unsigned long long localWaves =
        static_cast<unsigned long long>(maximumWaveCount);
    unsigned long long globalWaves = 0;
    MPI_Allreduce(&localWaves, &globalWaves, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    unsigned long long localNodes =
        static_cast<unsigned long long>(maximumProcessNodes);
    unsigned long long globalNodes = 0;
    MPI_Allreduce(&localNodes, &globalNodes, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    int localMassPass = rootMassValid ? 1 : 0;
    int globalMassPass = 0;
    MPI_Allreduce(&localMassPass, &globalMassPass, 1,
                  MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    const bool pass = globalMaximumError < 0.08 && globalMassPass != 0 &&
                      globalMeanRelativeAccelerationError < 0.01 &&
                      globalWaves > 1 &&
                      globalNodes > static_cast<unsigned long long>(2 * size);
    if(rank == 0)
    {
        std::ofstream output("fmm_patch_let_mpi_metrics.txt");
        output << "pass " << (pass ? 1 : 0) << "\n";
        output << "max_relative_error " << globalMaximumError << "\n";
        output << "mean_relative_acceleration_error "
               << globalMeanRelativeAccelerationError << "\n";
        output << "max_wave_count " << globalWaves << "\n";
        output << "max_process_nodes " << globalNodes << "\n";
        output << "root_mass_pass " << globalMassPass << "\n";
        std::cout << "fmm_patch_let_mpi pass=" << pass
                  << " error=" << globalMaximumError
                  << " mean_acceleration_error="
                  << globalMeanRelativeAccelerationError
                  << " waves=" << globalWaves << std::endl;
    }

    MPI_Finalize();
    return pass ? 0 : 1;
}
