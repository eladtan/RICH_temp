#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include <mpi.h>

#include "source/3D/gravity/fmm/mpi/DistributedFmmGravityCalculator.hpp"

namespace
{
struct Body
{
    Vector3D position;
    double mass = 0.0;
    std::uint64_t token = 0;
    int owner = -1;
};

Vector3D patchCenter(int rank, int patch)
{
    static const double yCenters[4] = {-0.75, -0.25, 0.25, 0.75};
    const double y = yCenters[rank % 4];
    if(patch == 0)
        return Vector3D(-0.75, y, -0.75);
    if(patch == 1)
        return Vector3D(0.75, y, 0.75);
    return Vector3D(0.25, y, -0.25);
}

void appendPair(std::vector<Body>& result,
                int rank,
                int patch,
                double offset,
                std::uint64_t tokenBase,
                int step)
{
    const Vector3D center = patchCenter(rank, patch);
    for(int side = -1; side <= 1; side += 2)
    {
        Body body;
        body.position = Vector3D(center.x + side * offset,
                                 center.y + 0.25 * side * offset,
                                 center.z - 0.15 * side * offset);
        body.mass = 0.55 + 0.04 * (rank + 1) +
                    0.015 * static_cast<double>(tokenBase + side + 2) +
                    0.01 * step;
        body.token = tokenBase + static_cast<std::uint64_t>(side > 0);
        body.owner = rank;
        result.push_back(body);
    }
}

std::vector<Body> bodiesForRank(int rank, int size, int step)
{
    if(size >= 4 && rank == size - 1)
        return std::vector<Body>();

    std::vector<Body> result;
    const double patchOffset = 0.08;
    appendPair(result, rank, 0, patchOffset, 100, step);
    appendPair(result, rank, 1, patchOffset, 200, step);

    if(rank == 0 && step == 1)
    {
        // Grow a retained leaf from two to three bodies without crossing the
        // persistent split threshold. This must reuse the LET and refresh only
        // the particle payload count.
        Body body;
        body.position = patchCenter(rank, 0);
        body.mass = 0.47;
        body.token = 250;
        body.owner = rank;
        result.push_back(body);
    }

    if(rank == 0 && step == 2)
    {
        const Vector3D center = patchCenter(rank, 0);
        for(int i = 0; i < 2; ++i)
        {
            Body body;
            const double sign = i == 0 ? -1.0 : 1.0;
            body.position = Vector3D(center.x + 0.018 * sign,
                                     center.y - 0.011 * sign,
                                     center.z + 0.007 * sign);
            body.mass = 0.43 + 0.02 * i;
            body.token = 300 + static_cast<std::uint64_t>(i);
            body.owner = rank;
            result.push_back(body);
        }
    }
    else if(rank == 0 && step == 3)
    {
        // Leave one body in patch 0.  The retained subtree is now below the
        // merge threshold but the patch itself remains active.
        result.erase(result.begin() + 1);
    }

    if(rank == 1 && step == 4)
    {
        // Replace patch 1 by a previously empty patch: one patch disappears
        // and another appears, forcing process-tree invalidation.
        result.erase(result.begin() + 2, result.begin() + 4);
        appendPair(result, rank, 2, 0.06, 400, step);
    }
    return result;
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
                            const std::vector<Body>& sources)
{
    Vector3D acceleration;
    for(const Body& source : sources)
    {
        if(source.owner == target.owner && source.token == target.token)
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
                       const std::vector<Body>& sources)
{
    double potential = 0.0;
    for(const Body& source : sources)
    {
        if(source.owner == target.owner && source.token == target.token)
            continue;
        const Vector3D delta = target.position - source.position;
        const double r2 = delta.x * delta.x + delta.y * delta.y +
                          delta.z * delta.z;
        if(r2 != 0.0)
            potential += source.mass / std::sqrt(r2);
    }
    return potential;
}

double vectorNorm(const Vector3D& value)
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
        ids.push_back(body.token);
    }
}

double solveAndCompare(DistributedFmmGravityCalculator& solver,
                       const std::vector<Body>& local,
                       const std::vector<Body>& global,
                       FmmSolveStats& stats)
{
    std::vector<Vector3D> positions;
    std::vector<double> masses;
    std::vector<std::uint64_t> ids;
    unpack(local, positions, masses, ids);
    std::vector<Vector3D> acceleration;
    std::vector<double> potential;
    solver.solve(positions, masses, ids,
                 Vector3D(-1.0, -1.0, -1.0),
                 Vector3D(1.0, 1.0, 1.0),
                 acceleration, &potential);
    stats = solver.stats();

    double maximum = 0.0;
    for(std::size_t index = 0; index < local.size(); ++index)
    {
        const Vector3D exactAcceleration =
            directAcceleration(local[index], global);
        maximum = std::max(maximum,
            vectorNorm(acceleration[index] - exactAcceleration) /
            std::max(1.0, vectorNorm(exactAcceleration)));
        const double exactPotential = directPotential(local[index], global);
        maximum = std::max(maximum,
            std::abs(potential[index] - exactPotential) /
            std::max(1.0, std::abs(exactPotential)));
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

    FmmGravityOptions numerical;
    numerical.expansionOrder = 5;
    numerical.thetaCritical = 0.35;
    numerical.leafCapacity = 2;
    numerical.computePotential = true;
    numerical.validateFinite = true;

    FmmDistributedOptions distributed;
    distributed.enablePatchForest = true;
    distributed.minimumPatchLevel = 2;
    distributed.maximumPatchLevel = 4;
    distributed.targetParticlesPerPatch = 0;
    distributed.maxTargetPatchesPerWave = 1;
    distributed.maxLetWaveBytes = 512;
    distributed.maxRemoteBytes = 64u * 1024u * 1024u;
    distributed.maxReplicatedDescriptorBytes = 16u * 1024u * 1024u;
    distributed.persistentLocalTreeTopology = true;
    distributed.persistentLeafSplitFactor = 1.5;
    distributed.persistentLeafMergeFactor = 0.5;

    DistributedFmmGravityCalculator solver(numerical, distributed);
    FmmSolveStats stepStats[5];
    double localMaximumError = 0.0;
    for(int step = 0; step < 5; ++step)
    {
        const std::vector<Body> local = bodiesForRank(rank, size, step);
        const std::vector<Body> global = allBodies(size, step);
        localMaximumError = std::max(localMaximumError,
            solveAndCompare(solver, local, global, stepStats[step]));
    }

    double maximumError = 0.0;
    MPI_Allreduce(&localMaximumError, &maximumError, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    const unsigned long long localIncremental[4] = {
        stepStats[2].letTargetSubplansReused,
        stepStats[2].letTargetSubplansRebuilt,
        stepStats[2].letSourceTriggeredInvalidations,
        stepStats[2].letWavePlanRebuildCount};
    unsigned long long globalIncremental[4] = {};
    MPI_Allreduce(localIncremental, globalIncremental, 4,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    double localImbalance = 0.0;
    std::size_t maximumWaves = 0;
    std::size_t maximumDescriptorBytes = 0;
    for(const FmmSolveStats& value : stepStats)
    {
        if(std::isfinite(value.processOwnedNodeImbalance))
            localImbalance = std::max(localImbalance,
                                      value.processOwnedNodeImbalance);
        else
            localImbalance = std::numeric_limits<double>::infinity();
        maximumWaves = std::max(maximumWaves, value.letWaveCount);
        maximumDescriptorBytes = std::max(maximumDescriptorBytes,
                                          value.replicatedDescriptorBytes);
    }
    double maximumImbalance = 0.0;
    MPI_Allreduce(&localImbalance, &maximumImbalance, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    unsigned long long localWaves =
        static_cast<unsigned long long>(maximumWaves);
    unsigned long long globalWaves = 0;
    MPI_Allreduce(&localWaves, &globalWaves, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    unsigned long long localDescriptorBytes =
        static_cast<unsigned long long>(maximumDescriptorBytes);
    unsigned long long globalDescriptorBytes = 0;
    MPI_Allreduce(&localDescriptorBytes, &globalDescriptorBytes, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    const bool warmReuse = !stepStats[1].processTopologyRebuilt &&
        !stepStats[1].letTopologyRebuilt &&
        stepStats[1].countOnlyTopologyReused &&
        !stepStats[1].letPayloadShapeTriggeredRebuild &&
        stepStats[1].reusedPatchCount == stepStats[1].localPatchCount &&
        stepStats[1].reusedLocalPatchPlanCount ==
            stepStats[1].localPatchCount;
    const bool splitIncremental =
        !stepStats[2].processTopologyRebuilt &&
        stepStats[2].letTopologyRebuilt &&
        stepStats[2].persistentLeafSplitCount > 0;
    const bool mergeIncremental =
        !stepStats[3].processTopologyRebuilt &&
        stepStats[3].letTopologyRebuilt &&
        stepStats[3].persistentSubtreeMergeCount > 0;
    const bool emptyPersistentLeavesExercised =
        stepStats[2].persistentEmptyLeafCount > 0;
    const bool patchSetRebuild = stepStats[4].processTopologyRebuilt &&
        stepStats[4].letTopologyRebuilt;
    const bool incrementalCoverage = globalIncremental[0] > 0 &&
        globalIncremental[1] > 0 && globalIncremental[2] > 0 &&
        globalIncremental[3] > 0;
    const bool memoryBound = globalDescriptorBytes <=
        distributed.maxReplicatedDescriptorBytes;
    const bool pass = maximumError < 0.1 &&
        stepStats[0].processTopologyRebuilt &&
        stepStats[0].letTopologyRebuilt && warmReuse &&
        splitIncremental && mergeIncremental && patchSetRebuild &&
        incrementalCoverage && memoryBound && globalWaves > 1 &&
        emptyPersistentLeavesExercised &&
        std::isfinite(maximumImbalance) && maximumImbalance < 3.0;

    if(rank == 0)
    {
        std::ofstream output("fmm_patch_moving_mesh_metrics.txt");
        output << "pass " << (pass ? 1 : 0) << "\n";
        output << "max_relative_error " << maximumError << "\n";
        output << "warm_process_reused "
               << (!stepStats[1].processTopologyRebuilt ? 1 : 0) << "\n";
        output << "warm_let_reused "
               << (!stepStats[1].letTopologyRebuilt ? 1 : 0) << "\n";
        output << "warm_count_only_reused "
               << (stepStats[1].countOnlyTopologyReused ? 1 : 0) << "\n";
        output << "warm_payload_shape_rebuilt "
               << (stepStats[1].letPayloadShapeTriggeredRebuild ? 1 : 0)
               << "\n";
        output << "split_process_reused "
               << (!stepStats[2].processTopologyRebuilt ? 1 : 0) << "\n";
        output << "split_let_rebuilt "
               << (stepStats[2].letTopologyRebuilt ? 1 : 0) << "\n";
        output << "merge_process_reused "
               << (!stepStats[3].processTopologyRebuilt ? 1 : 0) << "\n";
        output << "merge_let_rebuilt "
               << (stepStats[3].letTopologyRebuilt ? 1 : 0) << "\n";
        output << "persistent_empty_leaves_exercised "
               << (emptyPersistentLeavesExercised ? 1 : 0) << "\n";
        output << "patch_set_process_rebuilt "
               << (stepStats[4].processTopologyRebuilt ? 1 : 0) << "\n";
        output << "target_subplans_reused " << globalIncremental[0] << "\n";
        output << "target_subplans_rebuilt " << globalIncremental[1] << "\n";
        output << "source_invalidations " << globalIncremental[2] << "\n";
        output << "wave_plan_rebuilds " << globalIncremental[3] << "\n";
        output << "max_wave_count " << globalWaves << "\n";
        output << "max_owner_imbalance " << maximumImbalance << "\n";
        output << "max_descriptor_bytes " << globalDescriptorBytes << "\n";
        std::cout << "fmm_patch_moving_mesh pass=" << pass
                  << " error=" << maximumError
                  << " reused_targets=" << globalIncremental[0]
                  << " rebuilt_targets=" << globalIncremental[1]
                  << std::endl;
    }

    MPI_Finalize();
    return pass ? 0 : 1;
}
