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

std::vector<Body> bodiesForRank(int rank)
{
    std::vector<Body> result;
    if(rank < 0 || rank > 1)
        return result;
    const double sign = rank == 0 ? -1.0 : 1.0;
    for(int i = 0; i < 2; ++i)
    {
        Body body;
        const double edge = 0.90 - 0.04 * static_cast<double>(i);
        body.position = Vector3D(sign * edge, sign * 0.10, sign * 0.10);
        body.mass = 0.8 + 0.1 * static_cast<double>(rank + i);
        body.id = static_cast<std::uint64_t>(900000 + 100 * rank + i);
        body.owner = rank;
        body.localIndex = result.size();
        result.push_back(body);
    }
    return result;
}

std::vector<Body> allBodies(int size)
{
    std::vector<Body> result;
    for(int rank = 0; rank < size; ++rank)
    {
        const std::vector<Body> local = bodiesForRank(rank);
        result.insert(result.end(), local.begin(), local.end());
    }
    return result;
}

double norm(const Vector3D& value)
{
    return std::sqrt(value.x * value.x + value.y * value.y +
                     value.z * value.z);
}

Vector3D directAcceleration(const Body& target,
                            const std::vector<Body>& all)
{
    Vector3D result;
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
        result -= source.mass * delta * (invR * invR * invR);
    }
    return result;
}

double directPotential(const Body& target, const std::vector<Body>& all)
{
    double result = 0.0;
    for(const Body& source : all)
    {
        if(source.owner == target.owner &&
           source.localIndex == target.localIndex)
            continue;
        const Vector3D delta = target.position - source.position;
        const double r2 = delta.x * delta.x + delta.y * delta.y +
                          delta.z * delta.z;
        if(r2 != 0.0)
            result += source.mass / std::sqrt(r2);
    }
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

    if(size < 2)
    {
        if(rank == 0)
            std::cerr << "fmm_patch_m2p_mpi requires at least two ranks\n";
        MPI_Finalize();
        return 2;
    }

    FmmGravityOptions options;
    options.expansionOrder = 5;
    options.thetaCritical = 0.5;
    options.leafCapacity = 64;
    options.computePotential = true;
    options.validateFinite = true;

    FmmDistributedOptions distributed;
    distributed.enablePatchForest = true;
    distributed.minimumPatchLevel = 1;
    distributed.maximumPatchLevel = 4;
    distributed.targetParticlesPerPatch = 0;
    distributed.maxLocalPatchCount = 4096;
    distributed.maxTargetPatchesPerWave = 64;
    distributed.maxLetWaveBytes = 1024u * 1024u;
    distributed.maxRemoteBytes = 64u * 1024u * 1024u;
    distributed.maxReplicatedDescriptorBytes = 16u * 1024u * 1024u;
    distributed.enableLeafM2P = true;
    distributed.persistentLocalTreeTopology = false;

    const std::vector<Body> localBodies = bodiesForRank(rank);
    const std::vector<Body> globalBodies = allBodies(size);
    std::vector<Vector3D> positions;
    std::vector<double> masses;
    std::vector<std::uint64_t> ids;
    for(const Body& body : localBodies)
    {
        positions.push_back(body.position);
        masses.push_back(body.mass);
        ids.push_back(body.id);
    }

    DistributedFmmGravityCalculator solver(options, distributed);
    std::vector<Vector3D> acceleration;
    std::vector<double> potential;
    solver.solve(positions, masses, ids,
                 Vector3D(-1.0, -1.0, -1.0),
                 Vector3D(1.0, 1.0, 1.0),
                 acceleration, &potential);
    const FmmSolveStats& stats = solver.stats();

    double localError = 0.0;
    for(std::size_t i = 0; i < localBodies.size(); ++i)
    {
        const Vector3D referenceAcceleration =
            directAcceleration(localBodies[i], globalBodies);
        localError = std::max(localError,
            norm(acceleration[i] - referenceAcceleration) /
            std::max(1.0, norm(referenceAcceleration)));
        const double referencePotential =
            directPotential(localBodies[i], globalBodies);
        localError = std::max(localError,
            std::abs(potential[i] - referencePotential) /
            std::max(1.0, std::abs(referencePotential)));
    }

    double globalError = 0.0;
    MPI_Allreduce(&localError, &globalError, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    const unsigned long long localM2P =
        static_cast<unsigned long long>(stats.letM2PCount);
    unsigned long long globalM2P = 0;
    MPI_Allreduce(&localM2P, &globalM2P, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    const int localMassPass = std::isfinite(stats.rootMass) &&
        std::abs(stats.rootMass - stats.totalMass) <=
            1.0e-10 * std::max(1.0, std::abs(stats.totalMass));
    int globalMassPass = 0;
    MPI_Allreduce(&localMassPass, &globalMassPass, 1, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);

    const bool pass = globalError < 0.08 && globalM2P > 0 &&
                      globalMassPass != 0;
    if(rank == 0)
    {
        std::ofstream output("fmm_patch_m2p_mpi_metrics.txt");
        output << "pass " << (pass ? 1 : 0) << "\n";
        output << "max_relative_error " << globalError << "\n";
        output << "m2p_count " << globalM2P << "\n";
        output << "root_mass_pass " << globalMassPass << "\n";
        std::cout << "fmm_patch_m2p_mpi pass=" << pass
                  << " error=" << globalError
                  << " m2p=" << globalM2P << std::endl;
    }

    MPI_Finalize();
    return pass ? 0 : 1;
}
