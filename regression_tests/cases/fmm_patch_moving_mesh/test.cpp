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

    auto appendBody = [&](const Vector3D& position,
                          double mass,
                          std::uint64_t token) {
        Body body;
        body.position = position;
        body.mass = mass;
        body.token = token;
        body.owner = rank;
        result.push_back(body);
    };

    if(rank == 0)
    {
        // Keep patch 0's root internal throughout the persistent split/merge
        // sequence.  The unchanged upper branch dominates the patch-root
        // radius, so splitting the lower leaf does not invalidate the process
        // topology merely because the hierarchical radius bound changes.
        const Vector3D rootCenter = patchCenter(rank, 0);
        const Vector3D lowerChildCenter(rootCenter.x - 0.125,
                                        rootCenter.y - 0.125,
                                        rootCenter.z - 0.125);

        appendBody(Vector3D(rootCenter.x + 0.195,
                            rootCenter.y + 0.195,
                            rootCenter.z + 0.195),
                   0.63 + 0.01 * step, 100);

        appendBody(Vector3D(lowerChildCenter.x - 0.060,
                            lowerChildCenter.y - 0.060,
                            lowerChildCenter.z - 0.060),
                   0.57 + 0.005 * step, 101);

        if(step != 3)
        {
            appendBody(Vector3D(lowerChildCenter.x + 0.060,
                                lowerChildCenter.y + 0.060,
                                lowerChildCenter.z + 0.060),
                       0.59 + 0.005 * step, 102);
        }

        if(step == 1 || step == 2)
        {
            // Grow the retained leaf and expand its exact radius slightly.  A
            // 2% conservative radius envelope must absorb this motion without
            // invalidating the process tree or LET.
            appendBody(Vector3D(lowerChildCenter.x - 0.061,
                                lowerChildCenter.y + 0.060,
                                lowerChildCenter.z - 0.060),
                       0.47, 103);
        }

        if(step == 2)
        {
            // The fourth body crosses the persistent split threshold in the
            // lower leaf while the patch root remains structurally internal.
            appendBody(Vector3D(lowerChildCenter.x + 0.060,
                                lowerChildCenter.y - 0.060,
                                lowerChildCenter.z + 0.060),
                       0.45, 104);
        }
    }
    else
    {
        appendPair(result, rank, 0, patchOffset, 100, step);
    }

    appendPair(result, rank, 1, patchOffset, 200, step);

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

std::vector<Body> envelopeBodiesForRank(int rank, int size, bool expanded)
{
    std::vector<Body> result = bodiesForRank(rank, size, 0);
    if(!expanded || rank != 0)
        return result;

    // Expand the lower leaf of rank 0, patch 0 beyond its retained 2% radius
    // envelope.  This patch has an internal root: token 100 in the opposite
    // branch remains the root-radius extremum, so the patch-root descriptor and
    // process topology stay valid while only a non-root source generation
    // changes.  Keeping the two moved bodies in the same leaf also preserves
    // occupancy, child masks, and structural topology.
    const Vector3D rootCenter = patchCenter(rank, 0);
    const Vector3D lowerChildCenter(rootCenter.x - 0.125,
                                    rootCenter.y - 0.125,
                                    rootCenter.z - 0.125);
    const double offset = 0.065;
    for(Body& body : result)
    {
        if(body.token != 101 && body.token != 102)
            continue;
        const int side = body.token == 101 ? -1 : 1;
        body.position = Vector3D(lowerChildCenter.x + side * offset,
                                 lowerChildCenter.y + side * offset,
                                 lowerChildCenter.z + side * offset);
    }
    return result;
}

std::vector<Body> allEnvelopeBodies(int size, bool expanded)
{
    std::vector<Body> result;
    for(int rank = 0; rank < size; ++rank)
    {
        const std::vector<Body> local =
            envelopeBodiesForRank(rank, size, expanded);
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
    numerical.persistentRadiusSlackFactor = 1.02;
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
    FmmSolveStats payloadBaselineStats;
    FmmSolveStats payloadRefreshStats;
    bool payloadRefreshSemantics = false;
    double localPayloadRefreshError = 0.0;
    {
        // Force a stable max-depth leaf to outgrow its retained particle payload
        // capacity.  Interaction topology and the radius envelope stay valid,
        // so only the payload layout may change.
        FmmGravityOptions payloadNumerical = numerical;
        payloadNumerical.maxDepth = 1;
        // This fixture must exercise a remote particle subscription.  With the
        // production opening angle the growing leaf can be represented by a
        // multipole, in which case no particle-payload capacity can overflow.
        // A tiny positive opening angle forces remote traversal to leaf/leaf
        // P2P without relying on the particular rank geometry.
        payloadNumerical.thetaCritical = 1.0e-6;
        FmmDistributedOptions payloadDistributed = distributed;
        payloadDistributed.letParticlePayloadSlackFactor = 1.0;
        payloadDistributed.letParticlePayloadSlackCount = 0;
        payloadDistributed.enableLeafM2P = false;
        DistributedFmmGravityCalculator payloadSolver(
            payloadNumerical, payloadDistributed);

        const std::vector<Body> baselineLocal = bodiesForRank(rank, size, 0);
        localPayloadRefreshError = std::max(localPayloadRefreshError,
            solveAndCompare(payloadSolver, baselineLocal, allBodies(size, 0),
                            payloadBaselineStats));
        const std::uint64_t baselineEpoch = payloadBaselineStats.topologyEpoch;
        const std::uint64_t baselineRebuilds =
            payloadBaselineStats.topologyRebuildCount;
        const std::uint64_t baselineLetRebuilds =
            payloadBaselineStats.letTopologyRebuildCount;

        const std::vector<Body> grownLocal = bodiesForRank(rank, size, 2);
        localPayloadRefreshError = std::max(localPayloadRefreshError,
            solveAndCompare(payloadSolver, grownLocal, allBodies(size, 2),
                            payloadRefreshStats));
        payloadRefreshSemantics =
            payloadRefreshStats.letPayloadCapacityRefreshRequired &&
            payloadRefreshStats.letPayloadLayoutRefreshed &&
            !payloadRefreshStats.letPayloadShapeTriggeredRebuild &&
            !payloadRefreshStats.processTopologyRebuilt &&
            !payloadRefreshStats.letTopologyRebuilt &&
            payloadRefreshStats.countOnlyTopologyReused &&
            payloadRefreshStats.ranksWithLeafTopologyChange == 0 &&
            payloadRefreshStats.ranksWithGeometryEnvelopeChange == 0 &&
            payloadRefreshStats.topologyEpoch == baselineEpoch &&
            payloadRefreshStats.topologyRebuildCount == baselineRebuilds &&
            payloadRefreshStats.letTopologyRebuildCount == baselineLetRebuilds &&
            payloadRefreshStats.rootDescriptorExchangeSeconds == 0.0 &&
            payloadRefreshStats.letDescriptorTraversalSeconds == 0.0;
    }

    FmmSolveStats envelopeBaselineStats;
    FmmSolveStats envelopeExpansionStats;
    bool localReverseDependencySemantics = false;
    double localEnvelopeExpansionError = 0.0;
    {
        // Force the expanded patch to participate in remote LET dependencies
        // instead of being absorbed by a process-level or M2P multipole.
        FmmGravityOptions envelopeNumerical = numerical;
        envelopeNumerical.thetaCritical = 1.0e-6;
        FmmDistributedOptions envelopeDistributed = distributed;
        envelopeDistributed.enableLeafM2P = false;
        DistributedFmmGravityCalculator envelopeSolver(
            envelopeNumerical, envelopeDistributed);
        const std::vector<Body> baselineLocal =
            envelopeBodiesForRank(rank, size, false);
        localEnvelopeExpansionError = std::max(
            localEnvelopeExpansionError,
            solveAndCompare(envelopeSolver, baselineLocal,
                            allEnvelopeBodies(size, false),
                            envelopeBaselineStats));

        const std::vector<Body> expandedLocal =
            envelopeBodiesForRank(rank, size, true);
        localEnvelopeExpansionError = std::max(
            localEnvelopeExpansionError,
            solveAndCompare(envelopeSolver, expandedLocal,
                            allEnvelopeBodies(size, true),
                            envelopeExpansionStats));

        localReverseDependencySemantics =
            envelopeExpansionStats.ranksWithGeometryEnvelopeChange > 0 &&
            !envelopeExpansionStats.processTopologyRebuilt &&
            envelopeExpansionStats.letTopologyRebuilt &&
            envelopeExpansionStats.letSourceGenerationCheckCount >=
                envelopeExpansionStats.letChangedSourcePatchCount &&
            envelopeExpansionStats.letReverseDependencyLookupCount ==
                envelopeExpansionStats.letChangedSourcePatchCount &&
            envelopeExpansionStats.letSourceTriggeredInvalidations ==
                envelopeExpansionStats.letReverseDependencyTargetCount;
    }

    FmmSolveStats stepStats[5];
    double localMaximumError = 0.0;
    for(int step = 0; step < 5; ++step)
    {
        const std::vector<Body> local = bodiesForRank(rank, size, step);
        const std::vector<Body> global = allBodies(size, step);
        localMaximumError = std::max(localMaximumError,
            solveAndCompare(solver, local, global, stepStats[step]));
    }
    localMaximumError = std::max(
        localMaximumError, localPayloadRefreshError);
    localMaximumError = std::max(
        localMaximumError, localEnvelopeExpansionError);

    double maximumError = 0.0;
    MPI_Allreduce(&localMaximumError, &maximumError, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    const int localPayloadRefreshPass = payloadRefreshSemantics ? 1 : 0;
    int globalPayloadRefreshPass = 0;
    MPI_Allreduce(&localPayloadRefreshPass, &globalPayloadRefreshPass, 1,
                  MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    const unsigned long long localPayloadRefreshCounts[3] = {
        payloadRefreshStats.letPayloadCapacityUpdateCount,
        payloadRefreshStats.letPayloadSourceRepackCount,
        payloadRefreshStats.letWavePlanRebuildCount};
    unsigned long long globalPayloadRefreshCounts[3] = {};
    MPI_Allreduce(localPayloadRefreshCounts, globalPayloadRefreshCounts, 3,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    const int localReverseDependencyPass =
        localReverseDependencySemantics ? 1 : 0;
    int globalReverseDependencyPass = 0;
    MPI_Allreduce(&localReverseDependencyPass, &globalReverseDependencyPass, 1,
                  MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    const unsigned long long localReverseDependencyCounts[8] = {
        envelopeExpansionStats.letSourceGenerationCheckCount,
        envelopeExpansionStats.letChangedSourcePatchCount,
        envelopeExpansionStats.letReverseDependencyLookupCount,
        envelopeExpansionStats.letReverseDependencyTargetCount,
        envelopeExpansionStats.letReverseDependencyEdgeCount,
        envelopeExpansionStats.letSourceTriggeredInvalidations,
        envelopeExpansionStats.letTargetSubplansReused,
        envelopeExpansionStats.letTargetSubplansRebuilt};
    unsigned long long globalReverseDependencyCounts[8] = {};
    MPI_Allreduce(localReverseDependencyCounts,
                  globalReverseDependencyCounts, 8,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    const unsigned long long localIncremental[4] = {
        stepStats[2].letTargetSubplansReused,
        stepStats[2].letTargetSubplansRebuilt,
        stepStats[2].letSourceTriggeredInvalidations,
        stepStats[2].letWavePlanRebuildCount};
    unsigned long long globalIncremental[4] = {};
    MPI_Allreduce(localIncremental, globalIncremental, 4,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    const unsigned long long localPatchSetIncremental[4] = {
        stepStats[4].letTargetSubplansReused,
        stepStats[4].letTargetSubplansRebuilt,
        stepStats[4].letSourceTriggeredInvalidations,
        stepStats[4].letBuildStorageReused ? 1ull : 0ull};
    unsigned long long globalPatchSetIncremental[4] = {};
    MPI_Allreduce(localPatchSetIncremental, globalPatchSetIncremental, 4,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    const unsigned long long localSplitForest[4] = {
        static_cast<unsigned long long>(stepStats[2].localPatchCount),
        static_cast<unsigned long long>(stepStats[2].reusedPatchCount),
        static_cast<unsigned long long>(
            stepStats[2].reusedLocalPatchPlanCount),
        static_cast<unsigned long long>(
            stepStats[2].patchNodeGeometryExpansionCount)};
    unsigned long long globalSplitForest[4] = {};
    MPI_Allreduce(localSplitForest, globalSplitForest, 4,
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
            stepStats[1].localPatchCount &&
        stepStats[1].patchNodeGeometryExpansionCount == 0 &&
        stepStats[1].ranksWithNodeGeometryChange > 0 &&
        stepStats[1].ranksWithGeometryEnvelopeChange == 0 &&
        stepStats[1].ranksWithLeafTopologyChange == 0;
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
        stepStats[4].letTopologyRebuilt &&
        globalPatchSetIncremental[0] > 0 &&
        globalPatchSetIncremental[1] > 0 &&
        globalPatchSetIncremental[3] > 0;
    const bool incrementalCoverage = globalIncremental[0] > 0 &&
        globalIncremental[1] > 0 && globalIncremental[2] > 0 &&
        globalIncremental[3] > 0;
    const bool memoryBound = globalDescriptorBytes <=
        distributed.maxReplicatedDescriptorBytes;
    const bool payloadRefresh = globalPayloadRefreshPass != 0 &&
        globalPayloadRefreshCounts[0] > 0 &&
        globalPayloadRefreshCounts[1] > 0 &&
        globalPayloadRefreshCounts[2] > 0;
    const bool reverseDependencyRefresh =
        globalReverseDependencyPass != 0 &&
        globalReverseDependencyCounts[0] >=
            globalReverseDependencyCounts[1] &&
        globalReverseDependencyCounts[1] > 0 &&
        globalReverseDependencyCounts[2] ==
            globalReverseDependencyCounts[1] &&
        globalReverseDependencyCounts[3] > 0 &&
        globalReverseDependencyCounts[3] ==
            globalReverseDependencyCounts[5] &&
        globalReverseDependencyCounts[4] >=
            globalReverseDependencyCounts[3] &&
        globalReverseDependencyCounts[6] > 0 &&
        globalReverseDependencyCounts[7] > 0;
    const bool pass = maximumError < 0.1 &&
        stepStats[0].processTopologyRebuilt &&
        stepStats[0].letTopologyRebuilt && warmReuse &&
        splitIncremental && mergeIncremental && patchSetRebuild &&
        incrementalCoverage && memoryBound && globalWaves > 1 &&
        emptyPersistentLeavesExercised && payloadRefresh &&
        reverseDependencyRefresh &&
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
        output << "warm_radius_envelope_reused "
               << (stepStats[1].patchNodeGeometryExpansionCount == 0 ? 1 : 0) << "\n";
        output << "warm_exact_node_geometry_changed "
               << (stepStats[1].ranksWithNodeGeometryChange > 0 ? 1 : 0)
               << "\n";
        output << "warm_geometry_envelope_stable "
               << (stepStats[1].ranksWithGeometryEnvelopeChange == 0 ? 1 : 0)
               << "\n";
        output << "payload_refresh_semantics "
               << globalPayloadRefreshPass << "\n";
        output << "payload_refresh_required "
               << (payloadRefreshStats.letPayloadCapacityRefreshRequired ? 1 : 0)
               << "\n";
        output << "payload_layout_refreshed "
               << (payloadRefreshStats.letPayloadLayoutRefreshed ? 1 : 0)
               << "\n";
        output << "payload_process_reused "
               << (!payloadRefreshStats.processTopologyRebuilt ? 1 : 0) << "\n";
        output << "payload_let_topology_reused "
               << (!payloadRefreshStats.letTopologyRebuilt ? 1 : 0) << "\n";
        output << "payload_descriptor_exchange_skipped "
               << (payloadRefreshStats.rootDescriptorExchangeSeconds == 0.0 ? 1 : 0)
               << "\n";
        output << "payload_descriptor_traversal_skipped "
               << (payloadRefreshStats.letDescriptorTraversalSeconds == 0.0 ? 1 : 0)
               << "\n";
        output << "payload_capacity_updates "
               << globalPayloadRefreshCounts[0] << "\n";
        output << "payload_sources_repacked "
               << globalPayloadRefreshCounts[1] << "\n";
        output << "payload_wave_plan_rebuilds "
               << globalPayloadRefreshCounts[2] << "\n";
        output << "reverse_dependency_semantics "
               << globalReverseDependencyPass << "\n";
        output << "reverse_dependency_refresh "
               << (reverseDependencyRefresh ? 1 : 0) << "\n";
        output << "reverse_dependency_process_reused "
               << (!envelopeExpansionStats.processTopologyRebuilt ? 1 : 0)
               << "\n";
        output << "reverse_dependency_let_rebuilt "
               << (envelopeExpansionStats.letTopologyRebuilt ? 1 : 0) << "\n";
        output << "reverse_dependency_geometry_expanded "
               << (envelopeExpansionStats.ranksWithGeometryEnvelopeChange > 0 ? 1 : 0)
               << "\n";
        output << "source_generation_checks "
               << globalReverseDependencyCounts[0] << "\n";
        output << "changed_source_patches "
               << globalReverseDependencyCounts[1] << "\n";
        output << "reverse_dependency_lookups "
               << globalReverseDependencyCounts[2] << "\n";
        output << "reverse_dependency_targets "
               << globalReverseDependencyCounts[3] << "\n";
        output << "reverse_dependency_edges "
               << globalReverseDependencyCounts[4] << "\n";
        output << "reverse_dependency_source_invalidations "
               << globalReverseDependencyCounts[5] << "\n";
        output << "reverse_dependency_targets_reused "
               << globalReverseDependencyCounts[6] << "\n";
        output << "reverse_dependency_targets_rebuilt "
               << globalReverseDependencyCounts[7] << "\n";
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
        output << "patch_set_target_subplans_reused "
               << globalPatchSetIncremental[0] << "\n";
        output << "patch_set_target_subplans_rebuilt "
               << globalPatchSetIncremental[1] << "\n";
        output << "patch_set_source_invalidations "
               << globalPatchSetIncremental[2] << "\n";
        output << "patch_set_storage_reused_ranks "
               << globalPatchSetIncremental[3] << "\n";
        output << "target_subplans_reused " << globalIncremental[0] << "\n";
        output << "target_subplans_rebuilt " << globalIncremental[1] << "\n";
        output << "source_invalidations " << globalIncremental[2] << "\n";
        output << "wave_plan_rebuilds " << globalIncremental[3] << "\n";
        output << "max_wave_count " << globalWaves << "\n";
        output << "max_owner_imbalance " << maximumImbalance << "\n";
        output << "max_descriptor_bytes " << globalDescriptorBytes << "\n";
        output << "split_ranks_with_root_geometry_change "
               << stepStats[2].ranksWithRootGeometryChange << "\n";
        output << "split_ranks_with_leaf_topology_change "
               << stepStats[2].ranksWithLeafTopologyChange << "\n";
        output << "split_ranks_with_leaf_occupancy_change "
               << stepStats[2].ranksWithLeafOccupancyChange << "\n";
        output << "split_global_patch_count "
               << stepStats[2].globalPatchCount << "\n";
        output << "split_global_local_patch_count "
               << globalSplitForest[0] << "\n";
        output << "split_global_reused_patch_count "
               << globalSplitForest[1] << "\n";
        output << "split_global_reused_local_plan_count "
               << globalSplitForest[2] << "\n";
        output << "split_global_node_geometry_expansion_count "
               << globalSplitForest[3] << "\n";
        output << "split_payload_shape_rebuilt "
               << (stepStats[2].letPayloadShapeTriggeredRebuild ? 1 : 0)
               << "\n";
        output << "split_topology_forced "
               << (stepStats[2].topologyRebuildForced ? 1 : 0) << "\n";
        std::cout << "fmm_patch_moving_mesh pass=" << pass
                  << " error=" << maximumError
                  << " payload_refresh=" << payloadRefresh
                  << " reverse_dependencies=" << reverseDependencyRefresh
                  << " reused_targets=" << globalIncremental[0]
                  << " rebuilt_targets=" << globalIncremental[1]
                  << std::endl;
    }

    MPI_Finalize();
    return pass ? 0 : 1;
}
