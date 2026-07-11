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
    int ownerRank = -1;
    std::uint64_t ownerLocalIndex = 0;
};

std::vector<Body> bodiesForRank(int rank, int size,
                                double massScale,
                                bool moveFirstBody)
{
    if(size >= 3 && rank == size - 1)
        return std::vector<Body>();
    const int active = size >= 3 ? size - 1 : size;
    std::vector<Body> result;
    for(int i = 0; i < 4; ++i)
    {
        const double u = (static_cast<double>(rank) + 0.17 * (i + 1)) /
                         std::max(1, active);
        Body body;
        body.position = Vector3D(-0.9 + 1.8 * u,
            0.21 * std::sin(1.7 * (rank + 1) * (i + 1)),
            0.17 * std::cos(0.9 * (rank + 2) * (i + 1)));
        if(moveFirstBody && rank == 0 && i == 0)
            body.position.x = -0.999;
        body.mass = massScale * (0.5 + 0.07 * (rank + 1) + 0.03 * i);
        // Deliberately duplicate application IDs across ranks and bodies.  The
        // distributed solver must use its owner token, not this field, for
        // physical identity.
        body.id = static_cast<std::uint64_t>(i % 2);
        body.ownerRank = rank;
        body.ownerLocalIndex = static_cast<std::uint64_t>(i);
        result.push_back(body);
    }
    return result;
}

std::vector<Body> allBodies(int size, double massScale, bool moveFirstBody)
{
    std::vector<Body> result;
    for(int rank = 0; rank < size; ++rank)
    {
        const std::vector<Body> local =
            bodiesForRank(rank, size, massScale, moveFirstBody);
        result.insert(result.end(), local.begin(), local.end());
    }
    return result;
}

bool sameBody(const Body& first, const Body& second)
{
    return first.ownerRank == second.ownerRank &&
           first.ownerLocalIndex == second.ownerLocalIndex;
}

Vector3D directAcceleration(const Body& target, const std::vector<Body>& all)
{
    Vector3D result;
    for(const Body& source : all)
    {
        if(sameBody(source, target))
            continue;
        const Vector3D delta = target.position - source.position;
        const double r2 = delta.x * delta.x + delta.y * delta.y +
                          delta.z * delta.z;
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
        if(sameBody(source, target))
            continue;
        const Vector3D delta = target.position - source.position;
        result += source.mass / std::sqrt(delta.x * delta.x +
                                          delta.y * delta.y +
                                          delta.z * delta.z);
    }
    return result;
}

double norm(const Vector3D& value)
{
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
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

double checkSolve(const std::vector<Body>& localBodies,
                  const std::vector<Body>& globalBodies,
                  const std::vector<Vector3D>& acceleration,
                  const std::vector<double>& potential)
{
    double maximum = 0.0;
    for(std::size_t i = 0; i < localBodies.size(); ++i)
    {
        const Vector3D accelerationReference =
            directAcceleration(localBodies[i], globalBodies);
        const double potentialReference = directPotential(localBodies[i], globalBodies);
        maximum = std::max(maximum,
            norm(acceleration[i] - accelerationReference) /
            std::max(1.0, norm(accelerationReference)));
        maximum = std::max(maximum,
            std::abs(potential[i] - potentialReference) /
            std::max(1.0, std::abs(potentialReference)));
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
    distributed.maxRemoteBytes = 64u * 1024u * 1024u;

    double localMaximumError = 0.0;
    std::uint64_t firstEpoch = 0;
    std::uint64_t secondEpoch = 0;
    std::uint64_t thirdEpoch = 0;
    std::uint64_t firstRebuildCount = 0;
    std::uint64_t secondRebuildCount = 0;
    bool finiteStats = false;
    bool mismatchedDomainRejected = size == 1;

    {
        DistributedFmmGravityCalculator solver(options, distributed);
        std::vector<Vector3D> positions;
        std::vector<double> masses;
        std::vector<std::uint64_t> ids;
        std::vector<Vector3D> acceleration;
        std::vector<double> potential;

        std::vector<Body> localBodies = bodiesForRank(rank, size, 1.0, false);
        unpack(localBodies, positions, masses, ids);
        solver.solve(positions, masses, ids, Vector3D(-1, -1, -1),
                     Vector3D(1, 1, 1), acceleration, &potential);
        localMaximumError = std::max(localMaximumError,
            checkSolve(localBodies, allBodies(size, 1.0, false),
                       acceleration, potential));
        firstEpoch = solver.stats().topologyEpoch;
        firstRebuildCount = solver.stats().topologyRebuildCount;

        localBodies = bodiesForRank(rank, size, 1.01, false);
        unpack(localBodies, positions, masses, ids);
        solver.solve(positions, masses, ids, Vector3D(-1, -1, -1),
                     Vector3D(1, 1, 1), acceleration, &potential);
        localMaximumError = std::max(localMaximumError,
            checkSolve(localBodies, allBodies(size, 1.01, false),
                       acceleration, potential));
        secondEpoch = solver.stats().topologyEpoch;
        secondRebuildCount = solver.stats().topologyRebuildCount;

        localBodies = bodiesForRank(rank, size, 1.01, true);
        unpack(localBodies, positions, masses, ids);
        solver.solve(positions, masses, ids, Vector3D(-1, -1, -1),
                     Vector3D(1, 1, 1), acceleration, &potential);
        localMaximumError = std::max(localMaximumError,
            checkSolve(localBodies, allBodies(size, 1.01, true),
                       acceleration, potential));
        thirdEpoch = solver.stats().topologyEpoch;
        finiteStats = std::isfinite(solver.stats().totalSeconds) &&
                      std::isfinite(solver.stats().totalMass) &&
                      std::isfinite(solver.stats().rootMass) &&
                      solver.stats().activeRankCount ==
                          static_cast<std::size_t>(size >= 3 ? size - 1 : size) &&
                      solver.stats().bytesOwned > 0 &&
                      solver.stats().peakRemoteBytes <= distributed.maxRemoteBytes;

        if(size > 1)
        {
            try
            {
                const Vector3D upper = rank == 0 ? Vector3D(1, 1, 1) :
                                                   Vector3D(1.01, 1, 1);
                solver.solve(positions, masses, ids, Vector3D(-1, -1, -1),
                             upper, acceleration, &potential);
            }
            catch(...)
            {
                mismatchedDomainRejected = true;
            }
        }
    }

    double globalMaximumError = 0.0;
    MPI_Allreduce(&localMaximumError, &globalMaximumError, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    const int localPass = globalMaximumError < 2e-4 &&
                          firstEpoch == secondEpoch &&
                          firstRebuildCount == secondRebuildCount &&
                          thirdEpoch > secondEpoch && finiteStats &&
                          mismatchedDomainRejected;
    int globalPass = 0;
    MPI_Allreduce(&localPass, &globalPass, 1, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);

    if(rank == 0)
    {
        std::ofstream output("fmm_gravity_mpi_metrics.txt");
        output.setf(std::ios::scientific);
        output.precision(16);
        output << "ranks " << size << "\n";
        output << "max_scaled_error " << globalMaximumError << "\n";
        output << "first_epoch " << firstEpoch << "\n";
        output << "second_epoch " << secondEpoch << "\n";
        output << "third_epoch " << thirdEpoch << "\n";
        output << "mismatched_domain_rejected "
               << mismatchedDomainRejected << "\n";
        output << "pass " << globalPass << "\n";
        std::cout << "fmm_gravity_mpi ranks=" << size
                  << " max_scaled_error=" << globalMaximumError
                  << " pass=" << globalPass << std::endl;
    }

    MPI_Finalize();
    return globalPass ? 0 : 1;
}
