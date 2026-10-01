#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

#include <mpi.h>

#include "source/3D/gravity/fmm/mpi/DistributedFmmGravityCalculator.hpp"
#include "source/3D/gravity/fmm/mpi/FmmPatchForest.hpp"
#include "source/newtonian/three_dimensional/FastMultipoleAcceleration3D.hpp"

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

enum class BodyLayout
{
    Baseline,
    CountOnlyLeafChange,
    LocalLeafChange,
    PersistentSplit,
    RootBreach
};

std::vector<Body> bodiesForRank(int rank, int size,
                                double massScale,
                                BodyLayout layout)
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
        if(layout == BodyLayout::RootBreach && rank == 0 && i == 0)
        {
            // Leave the retained slack root while remaining inside the
            // global [-1,1]^3 domain, guaranteeing a topology rebuild.
            body.position.x = 0.999;
        }
        body.mass = massScale * (0.5 + 0.07 * (rank + 1) + 0.03 * i);
        // Deliberately duplicate application IDs across ranks and bodies.  The
        // distributed solver must use its owner token, not this field, for
        // physical identity.
        body.id = static_cast<std::uint64_t>(i % 2);
        body.ownerRank = rank;
        body.ownerLocalIndex = static_cast<std::uint64_t>(i);
        result.push_back(body);
    }

    if(layout == BodyLayout::CountOnlyLeafChange && rank == 0 && result.size() >= 2)
    {
        Body extra;
        extra.position = Vector3D(
            0.5 * (result[0].position.x + result[1].position.x),
            0.5 * (result[0].position.y + result[1].position.y),
            0.5 * (result[0].position.z + result[1].position.z));
        extra.mass = massScale * 0.91;
        extra.id = 0;
        extra.ownerRank = rank;
        extra.ownerLocalIndex = static_cast<std::uint64_t>(result.size());
        result.push_back(extra);
    }

    if(layout == BodyLayout::LocalLeafChange && rank == 0 && result.size() >= 4)
    {
        // Move one body close to another body while staying inside the original
        // local coordinate range.  The retained root cube therefore remains
        // valid, but leaf occupancy/spatial keys change for leafCapacity=2.
        const Vector3D& a = result[2].position;
        const Vector3D& b = result[3].position;
        result[0].position = Vector3D(
            0.9 * a.x + 0.1 * b.x,
            0.9 * a.y + 0.1 * b.y,
            0.9 * a.z + 0.1 * b.z);
    }

    if(layout == BodyLayout::PersistentSplit && rank == 0 && !result.empty())
    {
        const Vector3D anchor = result.front().position;
        for(int i = 0; i < 5; ++i)
        {
            const double offset = 1e-4 * static_cast<double>(i + 1);
            Body extra;
            extra.position = Vector3D(anchor.x + offset,
                                      anchor.y + 0.37 * offset,
                                      anchor.z + 0.19 * offset);
            extra.mass = massScale * (0.31 + 0.02 * i);
            extra.id = static_cast<std::uint64_t>(i % 2);
            extra.ownerRank = rank;
            extra.ownerLocalIndex =
                static_cast<std::uint64_t>(result.size());
            result.push_back(extra);
        }
    }
    return result;
}

std::vector<Body> allBodies(int size, double massScale, BodyLayout layout)
{
    std::vector<Body> result;
    for(int rank = 0; rank < size; ++rank)
    {
        const std::vector<Body> local =
            bodiesForRank(rank, size, massScale, layout);
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

double compareSolutions(const std::vector<Vector3D>& firstAcceleration,
                        const std::vector<double>& firstPotential,
                        const std::vector<Vector3D>& secondAcceleration,
                        const std::vector<double>& secondPotential)
{
    double maximum = 0.0;
    for(std::size_t i = 0; i < firstAcceleration.size(); ++i)
    {
        maximum = std::max(maximum,
            norm(firstAcceleration[i] - secondAcceleration[i]) /
            std::max(1.0, norm(secondAcceleration[i])));
        maximum = std::max(maximum,
            std::abs(firstPotential[i] - secondPotential[i]) /
            std::max(1.0, std::abs(secondPotential[i])));
    }
    return maximum;
}

struct PatchForestLifecycleObservation
{
    bool initialPatchCreated = false;
    bool identicalStateStable = false;
    bool countOnlyClassified = false;
    bool motionPatchSetStable = false;
    bool motionPatchGeometryStable = false;
    bool motionStructuralIdentityStable = false;
    bool motionNodeGeometryChanged = false;
    bool patchCreationClassified = false;
    bool patchRemovalClassified = false;
};

PatchForestLifecycleObservation exercisePatchForestLifecycle(int rank)
{
    FmmGravityOptions gravity;
    gravity.expansionOrder = 3;
    gravity.thetaCritical = 0.5;
    gravity.leafCapacity = 8;
    gravity.computePotential = false;
    gravity.validateFinite = true;
    gravity.persistentRadiusSlackFactor = 1.25;

    FmmDistributedOptions distributed;
    distributed.enablePatchForest = true;
    distributed.minimumPatchLevel = 2;
    distributed.maximumPatchLevel = 2;
    distributed.targetParticlesPerPatch = 0;
    distributed.maxLocalPatchCount = 64;
    distributed.persistentLocalTreeTopology = true;
    distributed.persistentLeafSplitFactor = 1.5;
    distributed.persistentLeafMergeFactor = 0.5;

    const Vector3D lower(-1.0, -1.0, -1.0);
    const Vector3D upper(1.0, 1.0, 1.0);

    std::vector<Vector3D> positions = {
        Vector3D(-0.94, -0.82, -0.81),
        Vector3D(-0.79, -0.80, -0.78),
        Vector3D(-0.73, -0.76, -0.74)};
    std::vector<double> masses = {0.7, 0.8, 0.9};
    std::vector<std::uint64_t> ids = {101, 102, 103};

    FmmPatchForest forest;
    PatchForestLifecycleObservation result;

    FmmPatchForestChange change = forest.prepare(
        positions, masses, ids, lower, upper, gravity, distributed, rank);
    result.initialPatchCreated =
        change.patchSetChanged && change.createdPatches == 1 &&
        change.removedPatches == 0 && change.matchedPatchIds == 0;

    std::vector<double> scaledMasses = masses;
    for(double& mass : scaledMasses)
        mass *= 1.01;
    change = forest.prepare(
        positions, scaledMasses, ids, lower, upper, gravity, distributed, rank);
    result.identicalStateStable =
        !change.patchSetChanged && !change.patchGeometryChanged &&
        !change.structuralTopologyChanged && !change.occupancyChanged &&
        !change.countOnlyChanged && change.createdPatches == 0 &&
        change.removedPatches == 0 && change.matchedPatchIds == 1;

    std::vector<Vector3D> countPositions = positions;
    std::vector<double> countMasses = scaledMasses;
    std::vector<std::uint64_t> countIds = ids;
    countPositions.push_back(Vector3D(-0.80, -0.79, -0.77));
    countMasses.push_back(0.65);
    countIds.push_back(104);
    change = forest.prepare(
        countPositions, countMasses, countIds, lower, upper,
        gravity, distributed, rank);
    result.countOnlyClassified =
        !change.patchSetChanged && !change.patchGeometryChanged &&
        !change.structuralTopologyChanged && change.occupancyChanged &&
        change.countOnlyChanged && change.createdPatches == 0 &&
        change.removedPatches == 0 && change.matchedPatchIds == 1;

    std::vector<Vector3D> movedPositions = countPositions;
    // Keep every particle in the same level-2 patch and preserve the leaf
    // occupancy, but change the tight node radius. Structural identity must
    // remain stable while the independent node-geometry signal records the
    // motion for conservative LET invalidation.
    movedPositions[0].x = -0.98;
    change = forest.prepare(
        movedPositions, countMasses, countIds, lower, upper,
        gravity, distributed, rank);
    result.motionPatchSetStable =
        !change.patchSetChanged && change.createdPatches == 0 &&
        change.removedPatches == 0 && change.matchedPatchIds == 1;
    result.motionPatchGeometryStable = !change.patchGeometryChanged;
    result.motionStructuralIdentityStable =
        !change.structuralTopologyChanged;
    result.motionNodeGeometryChanged = change.nodeGeometryChanged;

    std::vector<Vector3D> createdPositions = movedPositions;
    std::vector<double> createdMasses = countMasses;
    std::vector<std::uint64_t> createdIds = countIds;
    createdPositions.push_back(Vector3D(0.76, 0.77, 0.78));
    createdMasses.push_back(0.55);
    createdIds.push_back(105);
    change = forest.prepare(
        createdPositions, createdMasses, createdIds, lower, upper,
        gravity, distributed, rank);
    result.patchCreationClassified =
        change.patchSetChanged && change.createdPatches == 1 &&
        change.removedPatches == 0 && change.matchedPatchIds == 1;

    std::vector<Vector3D> removedPositions = {
        Vector3D(0.70, 0.71, 0.72),
        Vector3D(0.73, 0.74, 0.75),
        Vector3D(0.76, 0.77, 0.78),
        Vector3D(0.79, 0.80, 0.81),
        Vector3D(0.82, 0.83, 0.84)};
    change = forest.prepare(
        removedPositions, createdMasses, createdIds, lower, upper,
        gravity, distributed, rank);
    result.patchRemovalClassified =
        change.patchSetChanged && change.createdPatches == 0 &&
        change.removedPatches == 1 && change.matchedPatchIds == 1;

    return result;
}

bool individualTargetEvaluationPasses(
    int rank,
    int size,
    FmmGravityOptions options,
    const FmmDistributedOptions& distributed)
{
    options.computePotential = false;
    const std::vector<Body> localBodies =
        bodiesForRank(rank, size, 1.0, BodyLayout::Baseline);
    const std::vector<Body> globalBodies =
        allBodies(size, 1.0, BodyLayout::Baseline);

    std::vector<Vector3D> sourcePoints;
    std::vector<double> sourceMasses;
    std::vector<std::uint64_t> sourceIds;
    sourcePoints.reserve(localBodies.size());
    sourceMasses.reserve(localBodies.size());
    sourceIds.reserve(localBodies.size());
    for(const Body& body : localBodies)
    {
        sourcePoints.push_back(body.position);
        sourceMasses.push_back(body.mass);
        sourceIds.push_back(body.id);
    }

    std::vector<std::size_t> targetIndices;
    // With multiple ranks, rank zero owns sources but no active targets.  The
    // final rank in this fixture owns neither, so both collective edge cases
    // participate in the same solve.
    if(!localBodies.empty() && (size == 1 || rank != 0))
    {
        targetIndices.push_back(localBodies.size() - 1);
        if(localBodies.size() > 1)
            targetIndices.push_back(0);
    }
    std::vector<Vector3D> targetPoints;
    targetPoints.reserve(targetIndices.size());
    for(std::size_t index : targetIndices)
        targetPoints.push_back(localBodies[index].position);
    const std::vector<ComputationalCell3D> targetCells(targetPoints.size());

    const double gravitationalConstant = 0.75;
    FastMultipoleAcceleration3D gravity(
        options, distributed, gravitationalConstant);
    std::vector<Vector3D> acceleration;
    gravity.EvaluateIndividualTargets(
        std::make_pair(Vector3D(-2, -2, -2), Vector3D(2, 2, 2)),
        sourcePoints, sourceMasses, sourceIds, targetPoints, targetCells, 0.0,
        acceleration);
    bool parity = acceleration.size() == targetIndices.size();
    for(std::size_t target = 0;
        parity && target < targetIndices.size(); ++target)
    {
        const Vector3D reference = gravitationalConstant *
            directAcceleration(localBodies[targetIndices[target]], globalBodies);
        if(norm(acceleration[target] - reference) /
           std::max(1.0, norm(reference)) >= 1e-3)
            parity = false;
    }

    bool mismatchedSourcesRejected = false;
    std::vector<double> mismatchedMasses = sourceMasses;
    if(rank == 0)
        mismatchedMasses.pop_back();
    try
    {
        gravity.EvaluateIndividualTargets(
            std::make_pair(Vector3D(-2, -2, -2), Vector3D(2, 2, 2)),
            sourcePoints, mismatchedMasses, sourceIds, targetPoints,
            targetCells, 0.0, acceleration);
    }
    catch(UniversalError const&)
    {
        mismatchedSourcesRejected = true;
    }

    bool unknownTargetRejected = false;
    std::vector<Vector3D> unknownTargetPoints = targetPoints;
    std::vector<ComputationalCell3D> unknownTargetCells = targetCells;
    if(rank == 0)
    {
        unknownTargetPoints.push_back(Vector3D(0.125, 0.25, 0.5));
        unknownTargetCells.push_back(ComputationalCell3D());
    }
    try
    {
        gravity.EvaluateIndividualTargets(
            std::make_pair(Vector3D(-2, -2, -2), Vector3D(2, 2, 2)),
            sourcePoints, sourceMasses, sourceIds, unknownTargetPoints,
            unknownTargetCells, 0.0, acceleration);
    }
    catch(UniversalError const&)
    {
        unknownTargetRejected = true;
    }
    bool mismatchedIdsRejected = false;
    std::vector<std::uint64_t> mismatchedIds = sourceIds;
    if(rank == 0)
        mismatchedIds.pop_back();
    try
    {
        gravity.EvaluateIndividualTargets(
            std::make_pair(Vector3D(-2, -2, -2), Vector3D(2, 2, 2)),
            sourcePoints, sourceMasses, mismatchedIds, targetPoints,
            targetCells, 0.0, acceleration);
    }
    catch(UniversalError const&)
    {
        mismatchedIdsRejected = true;
    }
    return parity && mismatchedSourcesRejected && mismatchedIdsRejected &&
        unknownTargetRejected;
}
}

// ---------------------------------------------------------------------------
// Gravity-owner re-sampling (spatiallyRedistributeForGravity).  The gravity
// owners are Hilbert-key intervals whose keys are normalized by the domain;
// the splitters must be re-sampled when the domain grows and when the measured
// straggler time has paid for a re-sampling.

std::uint64_t splitMix64(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

double unitRandom(std::uint64_t key)
{
    return static_cast<double>(splitMix64(key) >> 11) * 0x1.0p-53;
}

// Uniform bodies in [-0.95,0.95]^3, or (clustered) all inside a small cube
// around (0.31,-0.23,0.11).  The last rank of >= 3 owns none, so an empty
// hydro rank takes part in the sampling collectives.
std::vector<Body> resampleBodiesForRank(int rank, int size, bool clustered)
{
    constexpr int perRank = 1500;
    std::vector<Body> result;
    if(size >= 3 && rank == size - 1)
        return result;
    const int active = size >= 3 ? size - 1 : size;
    for(int i = 0; i < perRank; ++i)
    {
        const std::uint64_t key =
            (static_cast<std::uint64_t>(rank) << 32) |
            static_cast<std::uint64_t>(i);
        const double u[3] = {unitRandom(3 * key), unitRandom(3 * key + 1),
                             unitRandom(3 * key + 2)};
        Body body;
        if(clustered)
            body.position = Vector3D(0.31 + 0.02 * (u[0] - 0.5),
                                     -0.23 + 0.02 * (u[1] - 0.5),
                                     0.11 + 0.02 * (u[2] - 0.5));
        else
            body.position = Vector3D(-0.95 + 1.9 * u[0], -0.95 + 1.9 * u[1],
                                     -0.95 + 1.9 * u[2]);
        body.mass = 1.0 / static_cast<double>(perRank * active);
        body.id = static_cast<std::uint64_t>(rank) * 1000000ull +
                  static_cast<std::uint64_t>(i);
        body.ownerRank = rank;
        body.ownerLocalIndex = static_cast<std::uint64_t>(i);
        result.push_back(body);
    }
    return result;
}

std::vector<Body> allResampleBodies(int size, bool clustered)
{
    std::vector<Body> result;
    for(int rank = 0; rank < size; ++rank)
    {
        const std::vector<Body> local =
            resampleBodiesForRank(rank, size, clustered);
        result.insert(result.end(), local.begin(), local.end());
    }
    return result;
}

// Largest gravity-owned particle count over the mean (collective).
double ownedImbalance(const DistributedFmmGravityCalculator& solver, int size)
{
    const double owned = static_cast<double>(solver.stats().particleCount);
    double maximum = 0.0;
    double sum = 0.0;
    MPI_Allreduce(&owned, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&owned, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sum > 0.0 ? maximum * static_cast<double>(size) / sum : 0.0;
}

struct GravityResampleObservation
{
    // Domain growth.
    bool domainReasonReported = false;
    bool frozenKeptSplitters = false;
    bool ownershipMatchesFresh = false;
    double resampledVsFresh = 0.0;
    double resampledDirectError = 0.0;
    double initialImbalance = 0.0;
    double frozenImbalance = 0.0;
    double resampledImbalance = 0.0;
    double freshImbalance = 0.0;
    // Straggler debt.
    bool baselineFromWarmSolve = false;
    bool debtResampled = false;
    int debtResampleSolve = -1;
    double debtImbalanceBefore = 0.0;
    double debtImbalanceAfter = 0.0;
    double debtAtTrigger = 0.0;
    double thresholdAtTrigger = 0.0;
    // Largest leaf over all ranks on the first clustered solve (retained
    // leaves absorb the moved bodies) and on the re-sampling solve, whose
    // changed ownership must rebuild every local tree afresh.
    double maxLeafBeforeTrigger = 0.0;
    double maxLeafAtTrigger = 0.0;
    std::size_t leafCapacity = 0;
    bool domainRebuiltPlans = false;
    bool debtRebuiltPlans = false;
    bool debtResetOnTrigger = false;
    bool baselineRenewed = false;
    bool debtClusteredOwnershipMatchesFresh = false;
    double debtClusteredVsFresh = 0.0;
    double debtClusteredDirectError = 0.0;
    double debtClusteredDirectErrorBefore = 0.0;
};

// Collective: true when the condition holds on every rank.
bool onEveryRank(bool condition)
{
    int value = condition ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    return value != 0;
}

// Collective: true when every rank's owned particle set (count and
// order-independent fingerprint) equals the other calculator's.
bool sameGravityOwnership(const DistributedFmmGravityCalculator& first,
                          const DistributedFmmGravityCalculator& second)
{
    int same = first.stats().particleCount == second.stats().particleCount &&
        first.stats().gravityOwnershipChecksum ==
            second.stats().gravityOwnershipChecksum ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &same, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    return same != 0;
}

double maximumOverRanks(double value, const MPI_Comm& comm = MPI_COMM_WORLD)
{
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MAX, comm);
    return value;
}

GravityResampleObservation exerciseGravityResampling(
    int rank, int size, const FmmDistributedOptions& baseDistributed)
{
    GravityResampleObservation result;
    FmmGravityOptions options;
    options.expansionOrder = 3;
    options.thetaCritical = 0.5;
    options.leafCapacity = 16;
    result.leafCapacity = options.leafCapacity;
    options.computePotential = true;
    options.validateFinite = true;
    FmmDistributedOptions distributed = baseDistributed;
    distributed.spatiallyRedistributeForGravity = true;
    distributed.useHilbertGravityRedistribution = true;
    distributed.persistentLocalTreeTopology = true;
    distributed.resampleGravitySplitters = true;
    FmmDistributedOptions frozenDistributed = distributed;
    frozenDistributed.resampleGravitySplitters = false;

    const Vector3D lower(-1.0, -1.0, -1.0);
    const Vector3D upper(1.0, 1.0, 1.0);
    // Only the upper z bound grows, as in the TDE driver's UpdateBox: every
    // normalized z shrinks by one half and crosses the level-1 key boundary.
    const Vector3D grownUpper(1.0, 1.0, 3.0);

    std::vector<Vector3D> positions;
    std::vector<double> masses;
    std::vector<std::uint64_t> ids;
    const std::vector<Body> uniformBodies =
        resampleBodiesForRank(rank, size, false);
    unpack(uniformBodies, positions, masses, ids);

    {
        DistributedFmmGravityCalculator resampled(options, distributed);
        DistributedFmmGravityCalculator frozen(options, frozenDistributed);
        std::vector<Vector3D> acceleration;
        std::vector<double> potential;
        resampled.solve(positions, masses, ids, lower, upper, acceleration,
                        &potential);
        frozen.solve(positions, masses, ids, lower, upper, acceleration,
                     &potential);
        result.initialImbalance = ownedImbalance(resampled, size);

        resampled.solve(positions, masses, ids, lower, grownUpper,
                        acceleration, &potential);
        const std::vector<Vector3D> resampledAcceleration = acceleration;
        const std::vector<double> resampledPotential = potential;
        result.resampledImbalance = ownedImbalance(resampled, size);
        result.domainReasonReported =
            (resampled.stats().gravityResampleReason &
             FmmSolveStats::gravityResampleDomain) != 0 &&
            resampled.stats().gravityResampleCount == 2 &&
            resampled.stats().gravityResampleEnabled;
        result.domainRebuiltPlans = onEveryRank(
            resampled.stats().processTopologyRebuilt &&
            resampled.stats().letTopologyRebuilt &&
            !resampled.stats().localInteractionPlanReused);
        result.resampledDirectError = checkSolve(
            uniformBodies, allResampleBodies(size, false),
            resampledAcceleration, resampledPotential);

        frozen.solve(positions, masses, ids, lower, grownUpper,
                     acceleration, &potential);
        result.frozenImbalance = ownedImbalance(frozen, size);
        result.frozenKeptSplitters =
            frozen.stats().gravityResampleReason == 0 &&
            frozen.stats().gravityResampleCount == 1 &&
            !frozen.stats().gravityResampleEnabled;

        DistributedFmmGravityCalculator fresh(options, distributed);
        fresh.solve(positions, masses, ids, lower, grownUpper, acceleration,
                    &potential);
        result.freshImbalance = ownedImbalance(fresh, size);
        result.ownershipMatchesFresh = sameGravityOwnership(fresh, resampled);
        result.resampledVsFresh = compareSolutions(
            resampledAcceleration, resampledPotential, acceleration,
            potential);
        MPI_Allreduce(MPI_IN_PLACE, &result.resampledVsFresh, 1, MPI_DOUBLE,
                      MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &result.resampledDirectError, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }

    // Forced imbalance: sample on the uniform state, then move every body into
    // one small cube that the retained key intervals give to one or two ranks.
    // The domain is unchanged, so only the measured straggler debt can trigger
    // the re-sampling.  Timing-based, so the trigger is given many solves.
    {
        DistributedFmmGravityCalculator solver(options, distributed);
        std::vector<Vector3D> acceleration;
        std::vector<double> potential;
        solver.solve(positions, masses, ids, lower, upper, acceleration,
                     &potential);
        const bool pendingAfterSampling =
            solver.stats().gravityStragglerBaselinePending;
        solver.solve(positions, masses, ids, lower, upper, acceleration,
                     &potential);
        result.baselineFromWarmSolve = pendingAfterSampling &&
            solver.stats().letTopologyRebuilt == false &&
            !solver.stats().gravityStragglerBaselinePending;

        // The clustered phase starts right after the baseline solve.  A
        // re-sampling decision uses the debt and threshold the preceding
        // solve left, so they are captured from the baseline solve on.
        const std::vector<Body> clusteredBodies =
            resampleBodiesForRank(rank, size, true);
        unpack(clusteredBodies, positions, masses, ids);
        constexpr int maximumClusteredSolves = 60;
        double previousDebt = solver.stats().gravityStragglerDebtSeconds;
        double previousThreshold =
            solver.stats().gravityResampleThresholdSeconds;
        for(int solve = 0; solve < maximumClusteredSolves; ++solve)
        {
            solver.solve(positions, masses, ids, lower, upper, acceleration,
                         &potential);
            const double imbalance = ownedImbalance(solver, size);
            double maxLeaf = static_cast<double>(
                solver.stats().maxLeafOccupancy);
            MPI_Allreduce(MPI_IN_PLACE, &maxLeaf, 1, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD);
            if(solve == 0)
            {
                result.debtImbalanceBefore = imbalance;
                result.maxLeafBeforeTrigger = maxLeaf;
                result.debtClusteredDirectErrorBefore =
                    maximumOverRanks(checkSolve(
                        clusteredBodies, allResampleBodies(size, true),
                        acceleration, potential));
            }
            if((solver.stats().gravityResampleReason &
                FmmSolveStats::gravityResampleImbalance) != 0)
            {
                result.debtResampled = true;
                result.debtResampleSolve = solve;
                result.debtImbalanceAfter = imbalance;
                result.maxLeafAtTrigger = maxLeaf;
                result.debtAtTrigger = previousDebt;
                result.thresholdAtTrigger = previousThreshold;
                result.debtRebuiltPlans = onEveryRank(
                    solver.stats().processTopologyRebuilt &&
                    solver.stats().letTopologyRebuilt &&
                    !solver.stats().localInteractionPlanReused);
                result.debtResetOnTrigger =
                    solver.stats().gravityStragglerDebtSeconds == 0.0 &&
                    solver.stats().gravityStragglerBaselinePending;
                // The next (warm) solve renews the baseline from its own
                // straggler time.
                solver.solve(positions, masses, ids, lower, upper,
                             acceleration, &potential);
                result.baselineRenewed =
                    !solver.stats().letTopologyRebuilt &&
                    !solver.stats().gravityStragglerBaselinePending &&
                    solver.stats().gravityStragglerBaselineSeconds ==
                        solver.stats().gravityStragglerExcessSeconds &&
                    solver.stats().gravityStragglerDebtSeconds == 0.0;
                result.debtClusteredDirectError = maximumOverRanks(checkSolve(
                    clusteredBodies, allResampleBodies(size, true),
                    acceleration, potential));
                DistributedFmmGravityCalculator fresh(options, distributed);
                std::vector<Vector3D> freshAcceleration;
                std::vector<double> freshPotential;
                fresh.solve(positions, masses, ids, lower, upper,
                            freshAcceleration, &freshPotential);
                result.debtClusteredOwnershipMatchesFresh =
                    sameGravityOwnership(fresh, solver);
                result.debtClusteredVsFresh = maximumOverRanks(
                    compareSolutions(acceleration, potential,
                                     freshAcceleration, freshPotential));
                break;
            }
            previousDebt = solver.stats().gravityStragglerDebtSeconds;
            previousThreshold = solver.stats().gravityResampleThresholdSeconds;
        }
    }
    return result;
}

// A domain change that keeps the global lattice (only the y bounds grow, both
// by the same amount, and y stays shorter than x): every re-sampling must
// rebuild the local, process and LET plans although the lattice, and possibly
// some rank root cubes, are unchanged.
struct LatticePreservingObservation
{
    bool domainReasonReported = false;
    bool rebuiltPlans = false;
    bool ownershipMatchesFresh = false;
    double vsFresh = 0.0;
    double directError = 0.0;
    int ranksWithOwnershipChange = 0;
    int ranksWithRootUnchanged = 0;
    int ranksWithRootUnchangedAndOwnershipChange = 0;
};

LatticePreservingObservation exerciseLatticePreservingResample(
    int rank, int size, const FmmDistributedOptions& baseDistributed)
{
    LatticePreservingObservation result;
    FmmGravityOptions options;
    options.expansionOrder = 3;
    options.thetaCritical = 0.5;
    options.leafCapacity = 16;
    options.computePotential = true;
    FmmDistributedOptions distributed = baseDistributed;
    distributed.spatiallyRedistributeForGravity = true;
    distributed.useHilbertGravityRedistribution = true;
    distributed.persistentLocalTreeTopology = true;
    distributed.resampleGravitySplitters = true;

    std::vector<Body> bodies = resampleBodiesForRank(rank, size, false);
    for(Body& body : bodies)
        body.position.y *= 0.75 / 0.95;
    std::vector<Body> all = allResampleBodies(size, false);
    for(Body& body : all)
        body.position.y *= 0.75 / 0.95;
    std::vector<Vector3D> positions;
    std::vector<double> masses;
    std::vector<std::uint64_t> ids;
    unpack(bodies, positions, masses, ids);
    const Vector3D lower(-1.0, -0.8, -1.0);
    const Vector3D upper(1.0, 0.8, 1.0);
    const Vector3D grownLower(-1.0, -0.81, -1.0);
    const Vector3D grownUpper(1.0, 0.81, 1.0);

    DistributedFmmGravityCalculator solver(options, distributed);
    std::vector<Vector3D> acceleration;
    std::vector<double> potential;
    solver.solve(positions, masses, ids, lower, upper, acceleration,
                 &potential);
    solver.solve(positions, masses, ids, lower, upper, acceleration,
                 &potential);
    const std::uint64_t checksumBefore =
        solver.stats().gravityOwnershipChecksum;
    const std::size_t countBefore = solver.stats().particleCount;
    solver.solve(positions, masses, ids, grownLower, grownUpper,
                 acceleration, &potential);
    result.domainReasonReported =
        (solver.stats().gravityResampleReason &
         FmmSolveStats::gravityResampleDomain) != 0;
    // Checked on every rank: without the forced rebuild a rank whose root
    // cube and leaf structure repeat reuses its local plan (seen with a
    // variant build that dropped the rebuild).
    result.rebuiltPlans = onEveryRank(solver.stats().processTopologyRebuilt &&
        solver.stats().letTopologyRebuilt &&
        !solver.stats().localInteractionPlanReused);
    const bool ownershipChanged =
        solver.stats().gravityOwnershipChecksum != checksumBefore ||
        solver.stats().particleCount != countBefore;
    const bool rootUnchanged = !solver.stats().localRootGeometryChanged;
    int counts[3] = {ownershipChanged ? 1 : 0, rootUnchanged ? 1 : 0,
                     ownershipChanged && rootUnchanged ? 1 : 0};
    MPI_Allreduce(MPI_IN_PLACE, counts, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    result.ranksWithOwnershipChange = counts[0];
    result.ranksWithRootUnchanged = counts[1];
    result.ranksWithRootUnchangedAndOwnershipChange = counts[2];
    result.directError = maximumOverRanks(
        checkSolve(bodies, all, acceleration, potential));

    DistributedFmmGravityCalculator fresh(options, distributed);
    std::vector<Vector3D> freshAcceleration;
    std::vector<double> freshPotential;
    fresh.solve(positions, masses, ids, grownLower, grownUpper,
                freshAcceleration, &freshPotential);
    result.ownershipMatchesFresh = sameGravityOwnership(fresh, solver);
    result.vsFresh = maximumOverRanks(compareSolutions(
        acceleration, potential, freshAcceleration, freshPotential));
    (void) size;
    return result;
}

// Positive Hilbert-volume weight: with fewer particles than ranks (fewer
// pooled samples than boundaries) and with an ordinary population.
double weightedSamplerDirectError(int rank, int size, bool sparse,
                                  const FmmDistributedOptions& baseDistributed)
{
    FmmGravityOptions options;
    // The sparse case has one dominant interaction per body, so use a strict
    // acceptance angle: its error then measures ownership, not truncation.
    options.expansionOrder = sparse ? 5 : 3;
    options.thetaCritical = sparse ? 0.15 : 0.5;
    options.leafCapacity = 16;
    options.computePotential = true;
    FmmDistributedOptions distributed = baseDistributed;
    distributed.spatiallyRedistributeForGravity = true;
    distributed.useHilbertGravityRedistribution = true;
    distributed.hilbertGravityVolumeWeight = 0.5;
    std::vector<Body> bodies;
    std::vector<Body> all;
    if(sparse)
    {
        for(int owner = 0; owner < std::min(size, 3); ++owner)
        {
            Body body;
            body.position = Vector3D(-0.6 + 0.55 * owner, 0.1 * owner,
                                     -0.2 * owner);
            body.mass = 1.0 / 3.0;
            body.id = static_cast<std::uint64_t>(owner);
            body.ownerRank = owner;
            body.ownerLocalIndex = 0;
            all.push_back(body);
            if(owner == rank)
                bodies.push_back(body);
        }
    }
    else
    {
        bodies = resampleBodiesForRank(rank, size, false);
        all = allResampleBodies(size, false);
    }
    std::vector<Vector3D> positions;
    std::vector<double> masses;
    std::vector<std::uint64_t> ids;
    unpack(bodies, positions, masses, ids);
    DistributedFmmGravityCalculator solver(options, distributed);
    std::vector<Vector3D> acceleration;
    std::vector<double> potential;
    const Vector3D lower(-1.0, -1.0, -1.0);
    const Vector3D upper(1.0, 1.0, 1.0);
    solver.solve(positions, masses, ids, lower, upper, acceleration,
                 &potential);
    // A second solve on a grown domain re-samples with the same weight.
    solver.solve(positions, masses, ids, lower, Vector3D(1.0, 1.0, 2.0),
                 acceleration, &potential);
    return maximumOverRanks(checkSolve(bodies, all, acceleration, potential));
}

// Two calculators on disjoint communicators (even and odd world ranks) solve
// concurrently; each reads its environment on its own root, so the two groups
// receive different RICH_FMM_GRAVITY_RESPLIT values.  Any collective on the
// wrong communicator would deadlock or mix the groups.
struct IsolationObservation
{
    bool settingsPerCommunicator = false;
    double directError = 0.0;
};

IsolationObservation exerciseCommunicatorIsolation(
    int rank, int size, const FmmDistributedOptions& baseDistributed)
{
    IsolationObservation result;
    FmmGravityOptions options;
    options.expansionOrder = 3;
    options.thetaCritical = 0.5;
    options.leafCapacity = 16;
    options.computePotential = true;
    FmmDistributedOptions distributed = baseDistributed;
    distributed.spatiallyRedistributeForGravity = true;
    distributed.useHilbertGravityRedistribution = true;
    MPI_Comm group = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &group);
    int groupRank = 0;
    MPI_Comm_rank(group, &groupRank);
    if(groupRank == 0)
        setenv("RICH_FMM_GRAVITY_RESPLIT", rank % 2 == 0 ? "0" : "1", 1);
    bool enabled = false;
    int reason = -1;
    {
        DistributedFmmGravityCalculator solver(options, distributed, group);
        unsetenv("RICH_FMM_GRAVITY_RESPLIT");
        const std::vector<Body> bodies =
            resampleBodiesForRank(rank, size, false);
        std::vector<Body> all;
        for(const Body& body : allResampleBodies(size, false))
            if(body.ownerRank % 2 == rank % 2)
                all.push_back(body);
        std::vector<Vector3D> positions;
        std::vector<double> masses;
        std::vector<std::uint64_t> ids;
        unpack(bodies, positions, masses, ids);
        std::vector<Vector3D> acceleration;
        std::vector<double> potential;
        solver.solve(positions, masses, ids, Vector3D(-1.0, -1.0, -1.0),
                     Vector3D(1.0, 1.0, 1.0), acceleration, &potential);
        solver.solve(positions, masses, ids, Vector3D(-1.0, -1.0, -1.0),
                     Vector3D(1.0, 1.0, 2.0), acceleration, &potential);
        enabled = solver.stats().gravityResampleEnabled;
        reason = solver.stats().gravityResampleReason;
        result.directError = maximumOverRanks(
            checkSolve(bodies, all, acceleration, potential), group);
    }
    MPI_Comm_free(&group);
    const bool expected = rank % 2 == 0 ?
        (!enabled && reason == 0) :
        (enabled && reason == FmmSolveStats::gravityResampleDomain);
    int ok = expected ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    result.settingsPerCommunicator = ok != 0;
    result.directError = maximumOverRanks(result.directError);
    return result;
}

// Collective: true when constructing the calculator threw on every rank.
bool constructionRejected(const FmmGravityOptions& options,
                          const FmmDistributedOptions& distributed)
{
    int rejected = 0;
    try
    {
        DistributedFmmGravityCalculator solver(options, distributed);
    }
    catch(const UniversalError&)
    {
        rejected = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, &rejected, 1, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);
    return rejected != 0;
}

bool constructionAccepted(const FmmGravityOptions& options,
                          const FmmDistributedOptions& distributed)
{
    int accepted = 1;
    try
    {
        DistributedFmmGravityCalculator solver(options, distributed);
    }
    catch(const UniversalError&)
    {
        accepted = 0;
    }
    MPI_Allreduce(MPI_IN_PLACE, &accepted, 1, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);
    return accepted != 0;
}

// Environment and option rules: only rank 0 of the communicator reads the
// environment (garbage elsewhere is ignored), an invalid value on rank 0 is
// rejected on every rank, and a rank-dependent emitSolveTrace is rejected.
struct EnvironmentObservation
{
    bool nonRootGarbageIgnored = false;
    bool rootGarbageRejected = false;
    bool samplesWithoutEveryRejected = false;
    bool negativeIntervalRejected = false;
    bool mismatchedTraceRejected = false;
};

EnvironmentObservation exerciseEnvironmentRules(
    int rank, int size, const FmmDistributedOptions& baseDistributed)
{
    EnvironmentObservation result;
    FmmGravityOptions options;
    options.expansionOrder = 2;
    options.thetaCritical = 0.5;
    options.leafCapacity = 16;
    FmmDistributedOptions distributed = baseDistributed;

    if(rank != 0)
        setenv("RICH_FMM_GRAVITY_RESPLIT", "garbage", 1);
    result.nonRootGarbageIgnored = constructionAccepted(options, distributed);
    unsetenv("RICH_FMM_GRAVITY_RESPLIT");

    if(rank == 0)
        setenv("RICH_FMM_GRAVITY_RESPLIT", "garbage", 1);
    result.rootGarbageRejected = constructionRejected(options, distributed);
    unsetenv("RICH_FMM_GRAVITY_RESPLIT");

    if(rank == 0)
        setenv("RICH_FMM_DIRECT_ERROR_SAMPLES", "5", 1);
    result.samplesWithoutEveryRejected =
        constructionRejected(options, distributed);
    unsetenv("RICH_FMM_DIRECT_ERROR_SAMPLES");

    if(rank == 0)
        setenv("RICH_FMM_STRUCTURAL_INTERVAL", "-3", 1);
    result.negativeIntervalRejected =
        constructionRejected(options, distributed);
    unsetenv("RICH_FMM_STRUCTURAL_INTERVAL");

    FmmDistributedOptions mismatched = distributed;
    mismatched.emitSolveTrace = rank == 0;
    result.mismatchedTraceRejected = size < 2 ||
        constructionRejected(options, mismatched);
    return result;
}

// Repeated emptying inside a nonzero structural window.  Each solve empties
// one more internal subtree (it merges: a forced rebuild while the window is
// closed) and grows one leaf past the split capacity.  Forced rebuilds must
// not restart the window clock, so the capacity split happens once the window
// opens (solve 3 for a window of 3) instead of being deferred while the
// emptying continues.  Rank-local trees (no redistribution).
struct WindowObservation
{
    bool mergesForcedRebuilds = false;
    int firstSplitSolve = -1;
    double directError = 0.0;
};

std::vector<Body> windowBodiesForRank(int rank, int movedClusters)
{
    // Clusters 0..4 in five octants, cluster "B" in the (+,+,+) octant.
    const double centers[5][3] = {{-0.5, -0.5, -0.5}, {0.5, -0.5, -0.5},
                                  {-0.5, 0.5, -0.5}, {-0.5, -0.5, 0.5},
                                  {0.5, 0.5, -0.5}};
    const Vector3D offset(0.011 * rank, 0.007 * rank, 0.003 * rank);
    std::vector<Body> result;
    std::uint64_t index = 0;
    const auto add = [&](const Vector3D& center, std::uint64_t seed) {
        Body body;
        body.position = center + offset + Vector3D(
            0.04 * (unitRandom(seed) - 0.5),
            0.04 * (unitRandom(seed + 1) - 0.5),
            0.04 * (unitRandom(seed + 2) - 0.5));
        body.mass = 1.0 / 400.0;
        body.id = index;
        body.ownerRank = rank;
        body.ownerLocalIndex = index;
        result.push_back(body);
        ++index;
    };
    const std::uint64_t base = 100000ull * static_cast<std::uint64_t>(rank);
    for(int i = 0; i < 3; ++i)
        add(Vector3D(0.5, 0.5, 0.5), base + 3 * static_cast<std::uint64_t>(i));
    for(int cluster = 0; cluster < 5; ++cluster)
        for(int i = 0; i < 8; ++i)
        {
            const std::uint64_t seed = base + 1000 +
                100 * static_cast<std::uint64_t>(cluster) +
                3 * static_cast<std::uint64_t>(i);
            const Vector3D center = cluster < movedClusters ?
                Vector3D(0.45, 0.45, 0.45) :
                Vector3D(centers[cluster][0], centers[cluster][1],
                         centers[cluster][2]);
            add(center, seed);
        }
    return result;
}

WindowObservation exerciseStructuralWindow(
    int rank, int size, const FmmDistributedOptions& baseDistributed)
{
    WindowObservation result;
    FmmGravityOptions options;
    options.expansionOrder = 4;
    options.thetaCritical = 0.3;
    options.leafCapacity = 4;
    options.computePotential = true;
    FmmDistributedOptions distributed = baseDistributed;
    distributed.spatiallyRedistributeForGravity = false;
    distributed.persistentLocalTreeTopology = true;
    distributed.minSolvesBetweenStructuralChanges = 3;
    DistributedFmmGravityCalculator solver(options, distributed);
    std::vector<Vector3D> positions;
    std::vector<double> masses;
    std::vector<std::uint64_t> ids;
    std::vector<Vector3D> acceleration;
    std::vector<double> potential;
    bool merges = true;
    std::vector<Body> bodies;
    for(int solve = 1; solve <= 6; ++solve)
    {
        bodies = windowBodiesForRank(rank, solve - 1);
        unpack(bodies, positions, masses, ids);
        solver.solve(positions, masses, ids, Vector3D(-1.0, -1.0, -1.0),
                     Vector3D(1.0, 1.0, 1.0), acceleration, &potential);
        if(solve == 2 || solve == 3)
            merges = merges &&
                solver.stats().persistentSubtreeMergeCount > 0 &&
                solver.stats().letTopologyRebuilt &&
                solver.stats().ranksWithRootGeometryChange == 0;
        if(solve == 2)
            merges = merges && solver.stats().persistentLeafSplitCount == 0;
        if(result.firstSplitSolve < 0 &&
           solver.stats().persistentLeafSplitCount > 0)
            result.firstSplitSolve = solve;
    }
    result.mergesForcedRebuilds = merges;
    std::vector<Body> all;
    for(int owner = 0; owner < size; ++owner)
    {
        const std::vector<Body> local = windowBodiesForRank(owner, 5);
        all.insert(all.end(), local.begin(), local.end());
    }
    result.directError = maximumOverRanks(
        checkSolve(bodies, all, acceleration, potential));
    return result;
}

// Print the rank and message of an uncaught exception before aborting, so a
// failure inside a collective case is diagnosable from the run log.
[[noreturn]] void reportUncaughtException()
{
    int rank = -1;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if(initialized != 0)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try
    {
        const std::exception_ptr current = std::current_exception();
        if(current)
            std::rethrow_exception(current);
    }
    catch(const UniversalError& error)
    {
        std::cerr << "fmm_gravity_mpi uncaught UniversalError on rank " << rank
                  << ":\n";
        reportError(error, std::cerr);
    }
    catch(const std::exception& error)
    {
        std::cerr << "fmm_gravity_mpi uncaught exception on rank " << rank
                  << ": " << error.what() << std::endl;
    }
    catch(...)
    {
        std::cerr << "fmm_gravity_mpi uncaught unknown exception on rank "
                  << rank << std::endl;
    }
    std::cerr.flush();
    std::abort();
}

int main(int argc, char** argv)
{
    std::set_terminate(reportUncaughtException);
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    FmmGravityOptions options;
    options.expansionOrder = 5;
    // Use a strict acceptance angle so the direct-force comparisons isolate
    // lifecycle/rebuild behavior rather than ordinary multipole truncation.
    // The data set is intentionally tiny, so the extra direct work is cheap.
    options.thetaCritical = 0.15;
    options.leafCapacity = 2;
    options.computePotential = true;
    options.validateFinite = true;

    FmmDistributedOptions distributed;
    // This regression intentionally exercises the one-tree-per-rank fallback
    // and its historical process/LET invalidation counters.
    distributed.enablePatchForest = false;
    distributed.maxRemoteBytes = 64u * 1024u * 1024u;
    // This test compares deliberately near-coincident particles against a
    // direct-force reference.  Keep that accuracy contract independent of the
    // optional lossy LET wire encodings; their production-scale accuracy is
    // exercised by the dedicated TDE and Lane-Emden regressions.
    distributed.compactLetParticlePayload = false;
    distributed.quantizedLetParticlePayload = false;
    // The lifecycle assertions below deliberately control which original rank
    // is empty and where a leaf splits.  Keep that contract independent of the
    // optional temporary gravity repartitioning; dedicated production-scale
    // regressions exercise the redistributed path.
    distributed.spatiallyRedistributeForGravity = false;
    // This regression validates the legacy count-only interaction-plan reuse
    // path. Bounded LET waves intentionally rebuild when leaf occupancy changes
    // because wave membership and payload sizing depend on current counts.
    // Dedicated wave regressions cover that execution mode.
    distributed.maxLetWaveBytes = 0;

    // Preserve the legacy sparse-tree rebuild checks below. Persistent-tree
    // execution, plan reuse, splitting, and automatic merging are exercised
    // independently in the dedicated block below.
    distributed.persistentLocalTreeTopology = false;

    const bool individualTargetEvaluation =
        individualTargetEvaluationPasses(rank, size, options, distributed);

    double localMaximumError = 0.0;
    constexpr std::size_t scenarioCount = 11;
    const char* scenarioNames[scenarioCount] = {
        "count_baseline",
        "count_only_change",
        "persistent_baseline",
        "persistent_refit",
        "persistent_split",
        "persistent_merge",
        "legacy_baseline",
        "legacy_mass_update",
        "legacy_leaf_change",
        "legacy_root_breach",
        "domain_growth"};
    constexpr std::size_t persistentSplitScenario = 4;
    std::array<double, scenarioCount> localScenarioErrors = {};
    double localFreshPersistentSplitError = 0.0;
    double localPersistentSplitVsFresh = 0.0;

    std::uint64_t firstEpoch = 0;
    std::uint64_t secondEpoch = 0;
    std::uint64_t thirdEpoch = 0;
    std::uint64_t firstRebuildCount = 0;
    std::uint64_t secondRebuildCount = 0;
    std::uint64_t secondProcessRebuildCount = 0;
    std::uint64_t secondLetRebuildCount = 0;
    std::uint64_t leafEpoch = 0;
    bool leafOnlyRebuild = false;
    bool rootProcessRebuild = false;
    bool countOnlyTopologyReused = false;
    bool countOnlyLocalPlanReused = false;
    bool leafStorageReused = false;
    bool rootStorageReset = false;
    bool finiteStats = false;
    bool mismatchedDomainRejected = size == 1;
    bool persistentEmptyLeavesExercised = false;
    bool persistentTopologyReused = false;
    bool persistentSplitRebuilt = false;
    bool persistentMergeRebuilt = false;

    PatchForestLifecycleObservation patchForestLifecycle;

    {
        FmmGravityOptions countOptions = options;
        countOptions.leafCapacity = 8;
        DistributedFmmGravityCalculator countSolver(countOptions, distributed);
        std::vector<Vector3D> positions;
        std::vector<double> masses;
        std::vector<std::uint64_t> ids;
        std::vector<Vector3D> acceleration;
        std::vector<double> potential;

        std::vector<Body> localBodies = bodiesForRank(
            rank, size, 1.0, BodyLayout::Baseline);
        unpack(localBodies, positions, masses, ids);
        countSolver.solve(positions, masses, ids, Vector3D(-1, -1, -1),
                          Vector3D(1, 1, 1), acceleration, &potential);
        localScenarioErrors[0] = checkSolve(
            localBodies, allBodies(size, 1.0, BodyLayout::Baseline),
            acceleration, potential);
        localMaximumError = std::max(localMaximumError, localScenarioErrors[0]);
        const std::uint64_t countEpoch = countSolver.stats().topologyEpoch;
        const std::uint64_t countRebuilds =
            countSolver.stats().topologyRebuildCount;
        const std::uint64_t countLetRebuilds =
            countSolver.stats().letTopologyRebuildCount;

        localBodies = bodiesForRank(
            rank, size, 1.0, BodyLayout::CountOnlyLeafChange);
        unpack(localBodies, positions, masses, ids);
        countSolver.solve(positions, masses, ids, Vector3D(-1, -1, -1),
                          Vector3D(1, 1, 1), acceleration, &potential);
        localScenarioErrors[1] = checkSolve(
            localBodies,
            allBodies(size, 1.0, BodyLayout::CountOnlyLeafChange),
            acceleration, potential);
        localMaximumError = std::max(localMaximumError, localScenarioErrors[1]);
        countOnlyTopologyReused =
            countSolver.stats().ranksWithRootGeometryChange == 0 &&
            countSolver.stats().ranksWithLeafTopologyChange == 0 &&
            countSolver.stats().ranksWithLeafOccupancyChange > 0 &&
            countSolver.stats().ranksWithCountOnlyLeafChange > 0 &&
            countSolver.stats().countOnlyTopologyReused &&
            !countSolver.stats().processTopologyRebuilt &&
            !countSolver.stats().letTopologyRebuilt &&
            countSolver.stats().topologyEpoch == countEpoch &&
            countSolver.stats().topologyRebuildCount == countRebuilds &&
            countSolver.stats().letTopologyRebuildCount == countLetRebuilds;
        countOnlyLocalPlanReused =
            localBodies.empty() || countSolver.stats().localInteractionPlanReused;
    }

    {
        FmmDistributedOptions persistentDistributed = distributed;
        persistentDistributed.persistentLocalTreeTopology = true;
        persistentDistributed.persistentLeafSplitFactor = 1.5;
        persistentDistributed.persistentLeafMergeFactor = 0.5;
        // This lifecycle expects a split on solve 3 and a merge on solve 4, so
        // capacity-driven structural changes must not be batched here.
        persistentDistributed.minSolvesBetweenStructuralChanges = 0;
        DistributedFmmGravityCalculator persistentSolver(
            options, persistentDistributed);
        std::vector<Vector3D> positions;
        std::vector<double> masses;
        std::vector<std::uint64_t> ids;
        std::vector<Vector3D> acceleration;
        std::vector<double> potential;

        std::vector<Body> localBodies = bodiesForRank(
            rank, size, 1.0, BodyLayout::Baseline);
        unpack(localBodies, positions, masses, ids);
        persistentSolver.solve(
            positions, masses, ids, Vector3D(-1, -1, -1),
            Vector3D(1, 1, 1), acceleration, &potential);
        localScenarioErrors[2] = checkSolve(
            localBodies, allBodies(size, 1.0, BodyLayout::Baseline),
            acceleration, potential);
        localMaximumError = std::max(localMaximumError, localScenarioErrors[2]);
        const std::uint64_t persistentEpoch =
            persistentSolver.stats().topologyEpoch;
        const std::uint64_t persistentRebuilds =
            persistentSolver.stats().topologyRebuildCount;
        const std::uint64_t persistentProcessRebuilds =
            persistentSolver.stats().processTopologyRebuildCount;
        const std::uint64_t persistentLetRebuilds =
            persistentSolver.stats().letTopologyRebuildCount;
        persistentEmptyLeavesExercised =
            persistentSolver.stats().persistentEmptyLeafCount > 0 &&
            persistentSolver.stats().persistentLeafSplitCount == 0 &&
            persistentSolver.stats().persistentSubtreeMergeCount == 0;

        localBodies = bodiesForRank(
            rank, size, 1.02, BodyLayout::Baseline);
        unpack(localBodies, positions, masses, ids);
        persistentSolver.solve(
            positions, masses, ids, Vector3D(-1, -1, -1),
            Vector3D(1, 1, 1), acceleration, &potential);
        localScenarioErrors[3] = checkSolve(
            localBodies,
            allBodies(size, 1.02, BodyLayout::Baseline),
            acceleration, potential);
        localMaximumError = std::max(localMaximumError, localScenarioErrors[3]);
        const std::size_t expectedActiveRanks =
            static_cast<std::size_t>(size >= 3 ? size - 1 : size);
        persistentTopologyReused =
            persistentSolver.stats().persistentTreeRefitRankCount ==
                expectedActiveRanks &&
            persistentSolver.stats().persistentLeafSplitCount == 0 &&
            persistentSolver.stats().persistentSubtreeMergeCount == 0 &&
            persistentSolver.stats().ranksWithLeafTopologyChange == 0 &&
            !persistentSolver.stats().processTopologyRebuilt &&
            !persistentSolver.stats().letTopologyRebuilt &&
            persistentSolver.stats().topologyEpoch == persistentEpoch &&
            persistentSolver.stats().topologyRebuildCount ==
                persistentRebuilds &&
            persistentSolver.stats().letTopologyRebuildCount ==
                persistentLetRebuilds &&
            (localBodies.empty() ||
             persistentSolver.stats().localInteractionPlanReused);

        localBodies = bodiesForRank(
            rank, size, 1.02, BodyLayout::PersistentSplit);
        unpack(localBodies, positions, masses, ids);
        persistentSolver.solve(
            positions, masses, ids, Vector3D(-1, -1, -1),
            Vector3D(1, 1, 1), acceleration, &potential);
        localScenarioErrors[4] = checkSolve(
            localBodies,
            allBodies(size, 1.02, BodyLayout::PersistentSplit),
            acceleration, potential);
        localMaximumError = std::max(localMaximumError, localScenarioErrors[4]);
        const std::uint64_t splitEpoch =
            persistentSolver.stats().topologyEpoch;
        const std::uint64_t splitRebuilds =
            persistentSolver.stats().topologyRebuildCount;
        const std::uint64_t splitLetRebuilds =
            persistentSolver.stats().letTopologyRebuildCount;
        persistentSplitRebuilt =
            persistentSolver.stats().persistentLeafSplitCount > 0 &&
            persistentSolver.stats().persistentSubtreeMergeCount == 0 &&
            persistentSolver.stats().ranksWithLeafTopologyChange > 0 &&
            !persistentSolver.stats().processTopologyRebuilt &&
            persistentSolver.stats().letTopologyRebuilt &&
            persistentSolver.stats().processTopologyRebuildCount ==
                persistentProcessRebuilds &&
            splitEpoch > persistentEpoch &&
            splitRebuilds == persistentRebuilds + 1 &&
            splitLetRebuilds == persistentLetRebuilds + 1;

        const std::vector<Vector3D> persistentSplitAcceleration = acceleration;
        const std::vector<double> persistentSplitPotential = potential;
        FmmDistributedOptions freshDistributed = distributed;
        freshDistributed.persistentLocalTreeTopology = false;
        DistributedFmmGravityCalculator freshSplitSolver(
            options, freshDistributed);
        std::vector<Vector3D> freshSplitAcceleration;
        std::vector<double> freshSplitPotential;
        freshSplitSolver.solve(
            positions, masses, ids, Vector3D(-1, -1, -1),
            Vector3D(1, 1, 1), freshSplitAcceleration,
            &freshSplitPotential);
        localFreshPersistentSplitError = checkSolve(
            localBodies,
            allBodies(size, 1.02, BodyLayout::PersistentSplit),
            freshSplitAcceleration, freshSplitPotential);
        localPersistentSplitVsFresh = compareSolutions(
            persistentSplitAcceleration, persistentSplitPotential,
            freshSplitAcceleration, freshSplitPotential);

        localBodies = bodiesForRank(
            rank, size, 1.02, BodyLayout::Baseline);
        unpack(localBodies, positions, masses, ids);
        persistentSolver.solve(
            positions, masses, ids, Vector3D(-1, -1, -1),
            Vector3D(1, 1, 1), acceleration, &potential);
        localScenarioErrors[5] = checkSolve(
            localBodies,
            allBodies(size, 1.02, BodyLayout::Baseline),
            acceleration, potential);
        localMaximumError = std::max(localMaximumError, localScenarioErrors[5]);
        persistentMergeRebuilt =
            persistentSolver.stats().persistentSubtreeMergeCount > 0 &&
            persistentSolver.stats().persistentLeafSplitCount == 0 &&
            persistentSolver.stats().ranksWithLeafTopologyChange > 0 &&
            !persistentSolver.stats().processTopologyRebuilt &&
            persistentSolver.stats().letTopologyRebuilt &&
            persistentSolver.stats().processTopologyRebuildCount ==
                persistentProcessRebuilds &&
            persistentSolver.stats().topologyEpoch > splitEpoch &&
            persistentSolver.stats().topologyRebuildCount ==
                splitRebuilds + 1 &&
            persistentSolver.stats().letTopologyRebuildCount ==
                splitLetRebuilds + 1;
    }

    {
        DistributedFmmGravityCalculator solver(options, distributed);
        std::vector<Vector3D> positions;
        std::vector<double> masses;
        std::vector<std::uint64_t> ids;
        std::vector<Vector3D> acceleration;
        std::vector<double> potential;

        std::vector<Body> localBodies = bodiesForRank(
            rank, size, 1.0, BodyLayout::Baseline);
        unpack(localBodies, positions, masses, ids);
        solver.solve(positions, masses, ids, Vector3D(-1, -1, -1),
                     Vector3D(1, 1, 1), acceleration, &potential);
        localScenarioErrors[6] = checkSolve(
            localBodies, allBodies(size, 1.0, BodyLayout::Baseline),
            acceleration, potential);
        localMaximumError = std::max(localMaximumError, localScenarioErrors[6]);
        firstEpoch = solver.stats().topologyEpoch;
        firstRebuildCount = solver.stats().topologyRebuildCount;

        localBodies = bodiesForRank(
            rank, size, 1.01, BodyLayout::Baseline);
        unpack(localBodies, positions, masses, ids);
        solver.solve(positions, masses, ids, Vector3D(-1, -1, -1),
                     Vector3D(1, 1, 1), acceleration, &potential);
        localScenarioErrors[7] = checkSolve(
            localBodies,
            allBodies(size, 1.01, BodyLayout::Baseline),
            acceleration, potential);
        localMaximumError = std::max(localMaximumError, localScenarioErrors[7]);
        secondEpoch = solver.stats().topologyEpoch;
        secondRebuildCount = solver.stats().topologyRebuildCount;
        secondProcessRebuildCount =
            solver.stats().processTopologyRebuildCount;
        secondLetRebuildCount = solver.stats().letTopologyRebuildCount;

        localBodies = bodiesForRank(
            rank, size, 1.01, BodyLayout::LocalLeafChange);
        unpack(localBodies, positions, masses, ids);
        solver.solve(positions, masses, ids, Vector3D(-1, -1, -1),
                     Vector3D(1, 1, 1), acceleration, &potential);
        localScenarioErrors[8] = checkSolve(
            localBodies,
            allBodies(size, 1.01, BodyLayout::LocalLeafChange),
            acceleration, potential);
        localMaximumError = std::max(localMaximumError, localScenarioErrors[8]);
        leafEpoch = solver.stats().topologyEpoch;
        leafStorageReused = solver.stats().letBuildStorageReused;
        leafOnlyRebuild =
            solver.stats().ranksWithRootGeometryChange == 0 &&
            solver.stats().ranksWithLeafTopologyChange > 0 &&
            !solver.stats().processTopologyRebuilt &&
            solver.stats().letTopologyRebuilt &&
            solver.stats().processTopologyRebuildCount ==
                secondProcessRebuildCount &&
            solver.stats().letTopologyRebuildCount == secondLetRebuildCount + 1 &&
            solver.stats().topologyRebuildCount == secondRebuildCount + 1 &&
            solver.stats().processCommunicatorsReused &&
            solver.stats().letCommunicatorReused &&
            !solver.stats().topologyRebuildForced;

        localBodies = bodiesForRank(
            rank, size, 1.01, BodyLayout::RootBreach);
        unpack(localBodies, positions, masses, ids);
        solver.solve(positions, masses, ids, Vector3D(-1, -1, -1),
                     Vector3D(1, 1, 1), acceleration, &potential);
        localScenarioErrors[9] = checkSolve(
            localBodies,
            allBodies(size, 1.01, BodyLayout::RootBreach),
            acceleration, potential);
        localMaximumError = std::max(localMaximumError, localScenarioErrors[9]);
        thirdEpoch = solver.stats().topologyEpoch;
        rootStorageReset = !solver.stats().letBuildStorageReused;
        rootProcessRebuild =
            solver.stats().ranksWithRootGeometryChange > 0 &&
            solver.stats().processTopologyRebuilt &&
            solver.stats().letTopologyRebuilt &&
            solver.stats().processTopologyRebuildCount ==
                secondProcessRebuildCount + 1 &&
            solver.stats().letTopologyRebuildCount == secondLetRebuildCount + 2 &&
            !solver.stats().topologyRebuildForced;
        finiteStats = std::isfinite(solver.stats().totalSeconds) &&
                      std::isfinite(solver.stats().topologyRebuildSeconds) &&
                      std::isfinite(solver.stats().rootDescriptorExchangeSeconds) &&
                      std::isfinite(solver.stats().processTopologySeconds) &&
                      std::isfinite(solver.stats().letBuildResetSeconds) &&
                      std::isfinite(solver.stats().letDescriptorTraversalSeconds) &&
                      std::isfinite(solver.stats().letFinalizeSeconds) &&
                      std::isfinite(solver.stats().letSubscriptionSeconds) &&
                      std::isfinite(solver.stats().letPruneCompactSeconds) &&
                      std::isfinite(solver.stats().totalMass) &&
                      std::isfinite(solver.stats().rootMass) &&
                      solver.stats().letPlannedM2LCount >=
                          solver.stats().letM2LCount &&
                      solver.stats().letPlannedP2PBlockCount >=
                          solver.stats().letP2PBlockCount &&
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

    // Domain growth (a growing simulation box, e.g. the TDE driver's UpdateBox):
    // the same bodies, still inside every rank's retained root, solved again in a
    // larger domain.  This growth changes the global lattice, so every rank
    // that owns bodies must rebuild its root on the new lattice; a rank that kept
    // its old root would publish a stale lattice id to the LET build.  Uses the
    // production defaults for the local tree (persistent topology).
    bool domainGrowthRebuilt = false;
    {
        FmmDistributedOptions growthDistributed = distributed;
        growthDistributed.persistentLocalTreeTopology = true;
        DistributedFmmGravityCalculator growthSolver(options, growthDistributed);
        std::vector<Vector3D> positions;
        std::vector<double> masses;
        std::vector<std::uint64_t> ids;
        std::vector<Vector3D> acceleration;
        std::vector<double> potential;
        std::vector<Body> localBodies = bodiesForRank(
            rank, size, 1.0, BodyLayout::Baseline);
        unpack(localBodies, positions, masses, ids);
        growthSolver.solve(positions, masses, ids, Vector3D(-1, -1, -1),
                           Vector3D(1, 1, 1), acceleration, &potential);
        const std::uint64_t rebuildsBefore =
            growthSolver.stats().processTopologyRebuildCount;
        growthSolver.solve(positions, masses, ids, Vector3D(-2, -2, -2),
                           Vector3D(2, 2, 2), acceleration, &potential);
        localScenarioErrors[10] = checkSolve(
            localBodies, allBodies(size, 1.0, BodyLayout::Baseline),
            acceleration, potential);
        localMaximumError = std::max(localMaximumError, localScenarioErrors[10]);
        int ranksWithBodies = positions.empty() ? 0 : 1;
        MPI_Allreduce(MPI_IN_PLACE, &ranksWithBodies, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        domainGrowthRebuilt =
            static_cast<int>(growthSolver.stats().ranksWithRootGeometryChange) ==
                ranksWithBodies &&
            growthSolver.stats().processTopologyRebuilt &&
            growthSolver.stats().letTopologyRebuilt &&
            growthSolver.stats().processTopologyRebuildCount == rebuildsBefore + 1;
    }

    patchForestLifecycle = exercisePatchForestLifecycle(rank);

    const GravityResampleObservation resample =
        exerciseGravityResampling(rank, size, distributed);
    // Resampling contract: the grown domain triggers a re-sampling whose
    // ownership and forces equal a fresh calculator's; the frozen splitters
    // are measurably worse there; the forced cluster triggers the debt rule
    // and re-balances; the baseline is taken on a warm solve.
    const bool resampleDomainPass = resample.domainReasonReported &&
        resample.frozenKeptSplitters && resample.ownershipMatchesFresh &&
        resample.domainRebuiltPlans &&
        resample.resampledVsFresh < 1e-12 &&
        resample.resampledDirectError < 1e-2 &&
        resample.resampledImbalance == resample.freshImbalance &&
        resample.frozenImbalance > 1.25 * resample.resampledImbalance;
    const bool resampleDebtPass = resample.baselineFromWarmSolve &&
        resample.debtResampled &&
        resample.thresholdAtTrigger > 0.0 &&
        resample.debtAtTrigger >= resample.thresholdAtTrigger &&
        resample.debtRebuiltPlans && resample.debtResetOnTrigger &&
        resample.baselineRenewed &&
        resample.debtImbalanceBefore > 2.5 &&
        resample.debtImbalanceAfter < 0.5 * resample.debtImbalanceBefore &&
        resample.maxLeafAtTrigger <=
            static_cast<double>(resample.leafCapacity) &&
        resample.debtClusteredOwnershipMatchesFresh &&
        resample.debtClusteredVsFresh < 1e-12 &&
        // 0.1 is an empirical tolerance for this fixture: truncation at
        // theta 0.5 / order 3 on the 9000-body cluster reaches a few percent.
        // theta^4/(1-theta) = 0.125 estimates one well-separated interaction
        // relative to its own source field; it does not bound this error,
        // which is normalized by the net acceleration and potential and so
        // can grow where forces cancel.  Before the re-sampling one retained
        // leaf held every body, so that solve was an exact direct sum and is
        // reported only.
        resample.debtClusteredDirectError < 0.1;
    const LatticePreservingObservation lattice =
        exerciseLatticePreservingResample(rank, size, distributed);
    // The fixture is deterministic: with seven ranks five ranks keep their
    // root cube while their owned particles change (job 20260924_214350).
    const bool latticeResamplePass = lattice.domainReasonReported &&
        lattice.rebuiltPlans && lattice.ownershipMatchesFresh &&
        lattice.ranksWithOwnershipChange > 0 &&
        (size < 3 || lattice.ranksWithRootUnchangedAndOwnershipChange > 0) &&
        lattice.vsFresh < 1e-12 && lattice.directError < 1e-2;
    const double sparseWeightedError =
        weightedSamplerDirectError(rank, size, true, distributed);
    const double weightedError =
        weightedSamplerDirectError(rank, size, false, distributed);
    const bool weightedSamplerPass =
        sparseWeightedError < 1e-3 && weightedError < 1e-2;
    const IsolationObservation isolation =
        exerciseCommunicatorIsolation(rank, size, distributed);
    const bool isolationPass = size < 2 ||
        (isolation.settingsPerCommunicator && isolation.directError < 1e-2);
    const EnvironmentObservation environment =
        exerciseEnvironmentRules(rank, size, distributed);
    const bool environmentPass = environment.nonRootGarbageIgnored &&
        environment.rootGarbageRejected &&
        environment.samplesWithoutEveryRejected &&
        environment.negativeIntervalRejected &&
        environment.mismatchedTraceRejected;
    const WindowObservation window =
        exerciseStructuralWindow(rank, size, distributed);
    const bool windowPass = window.mergesForcedRebuilds &&
        window.firstSplitSolve > 0 && window.firstSplitSolve <= 4 &&
        window.directError < 1e-2;

    double globalMaximumError = 0.0;
    MPI_Allreduce(&localMaximumError, &globalMaximumError, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    std::array<double, scenarioCount> globalScenarioErrors = {};
    MPI_Allreduce(localScenarioErrors.data(), globalScenarioErrors.data(),
                  static_cast<int>(scenarioCount), MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    double globalFreshPersistentSplitError = 0.0;
    double globalPersistentSplitVsFresh = 0.0;
    MPI_Allreduce(&localFreshPersistentSplitError,
                  &globalFreshPersistentSplitError, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&localPersistentSplitVsFresh,
                  &globalPersistentSplitVsFresh, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    // The persistent-split distribution deliberately places several particles
    // only O(1e-4) apart.  It is therefore a much harder direct-summation
    // accuracy case than the ordinary regression scenarios.  The ordinary
    // Use the strict acceptance angle above to bound the ordinary cases at
    // 1e-3, require a coarse direct bound for the clustered case, and directly
    // verify that persistent-tree splitting agrees with a fresh nonpersistent
    // rebuild.
    double ordinaryMaximumError = 0.0;
    for(std::size_t i = 0; i < scenarioCount; ++i)
        if(i != persistentSplitScenario)
            ordinaryMaximumError = std::max(
                ordinaryMaximumError, globalScenarioErrors[i]);
    const int ordinaryErrorsWithinTolerance =
        ordinaryMaximumError < 1e-3 ? 1 : 0;
    const int persistentSplitDirectWithinTolerance =
        globalScenarioErrors[persistentSplitScenario] < 1e-2 ? 1 : 0;
    const int persistentSplitFreshDirectWithinTolerance =
        globalFreshPersistentSplitError < 1e-2 ? 1 : 0;
    const int persistentSplitMatchesFresh =
        globalPersistentSplitVsFresh < 2e-4 ? 1 : 0;
    const int errorWithinTolerance = ordinaryErrorsWithinTolerance &&
        persistentSplitDirectWithinTolerance &&
        persistentSplitFreshDirectWithinTolerance &&
        persistentSplitMatchesFresh;

    const int localPatchForestChecks[9] = {
        patchForestLifecycle.initialPatchCreated ? 1 : 0,
        patchForestLifecycle.identicalStateStable ? 1 : 0,
        patchForestLifecycle.countOnlyClassified ? 1 : 0,
        patchForestLifecycle.motionPatchSetStable ? 1 : 0,
        patchForestLifecycle.motionPatchGeometryStable ? 1 : 0,
        patchForestLifecycle.motionStructuralIdentityStable ? 1 : 0,
        patchForestLifecycle.motionNodeGeometryChanged ? 1 : 0,
        patchForestLifecycle.patchCreationClassified ? 1 : 0,
        patchForestLifecycle.patchRemovalClassified ? 1 : 0};
    int globalPatchForestChecks[9] = {};
    MPI_Allreduce(localPatchForestChecks, globalPatchForestChecks, 9,
                  MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    const int patchForestLifecyclePass =
        std::all_of(globalPatchForestChecks, globalPatchForestChecks + 9,
                    [](int value) { return value != 0; }) ? 1 : 0;

    const int localChecks[25] = {
        firstEpoch == secondEpoch ? 1 : 0,
        firstRebuildCount == secondRebuildCount ? 1 : 0,
        leafEpoch > secondEpoch ? 1 : 0,
        thirdEpoch > leafEpoch ? 1 : 0,
        leafOnlyRebuild ? 1 : 0,
        rootProcessRebuild ? 1 : 0,
        countOnlyTopologyReused ? 1 : 0,
        countOnlyLocalPlanReused ? 1 : 0,
        finiteStats ? 1 : 0,
        mismatchedDomainRejected ? 1 : 0,
        leafStorageReused ? 1 : 0,
        rootStorageReset ? 1 : 0,
        persistentEmptyLeavesExercised ? 1 : 0,
        persistentTopologyReused ? 1 : 0,
        persistentSplitRebuilt ? 1 : 0,
        persistentMergeRebuilt ? 1 : 0,
        individualTargetEvaluation ? 1 : 0,
        domainGrowthRebuilt ? 1 : 0,
        resampleDomainPass ? 1 : 0,
        resampleDebtPass ? 1 : 0,
        latticeResamplePass ? 1 : 0,
        weightedSamplerPass ? 1 : 0,
        isolationPass ? 1 : 0,
        environmentPass ? 1 : 0,
        windowPass ? 1 : 0};
    int globalChecks[25] = {};
    MPI_Allreduce(localChecks, globalChecks, 25, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);
    const int globalPass = errorWithinTolerance &&
                           globalChecks[0] && globalChecks[1] &&
                           globalChecks[2] && globalChecks[3] &&
                           globalChecks[4] && globalChecks[5] &&
                           globalChecks[6] && globalChecks[7] &&
                           globalChecks[8] && globalChecks[9] &&
                           globalChecks[10] && globalChecks[11] &&
                           globalChecks[12] && globalChecks[13] &&
                           globalChecks[14] && globalChecks[15] &&
                           globalChecks[16] && globalChecks[17] &&
                           globalChecks[18] && globalChecks[19] &&
                           globalChecks[20] && globalChecks[21] &&
                           globalChecks[22] && globalChecks[23] &&
                           globalChecks[24] &&
                           patchForestLifecyclePass;

    if(rank == 0)
    {
        std::ofstream output("fmm_gravity_mpi_metrics.txt");
        output.setf(std::ios::scientific);
        output.precision(16);
        output << "ranks " << size << "\n";
        output << "max_scaled_error " << globalMaximumError << "\n";
        output << "ordinary_max_scaled_error " << ordinaryMaximumError << "\n";
        for(std::size_t i = 0; i < scenarioCount; ++i)
            output << "scenario_error_" << scenarioNames[i] << " "
                   << globalScenarioErrors[i] << "\n";
        output << "persistent_split_fresh_error "
               << globalFreshPersistentSplitError << "\n";
        output << "persistent_split_vs_fresh "
               << globalPersistentSplitVsFresh << "\n";
        output << "ordinary_errors_within_tolerance "
               << ordinaryErrorsWithinTolerance << "\n";
        output << "persistent_split_direct_within_tolerance "
               << persistentSplitDirectWithinTolerance << "\n";
        output << "persistent_split_fresh_direct_within_tolerance "
               << persistentSplitFreshDirectWithinTolerance << "\n";
        output << "persistent_split_matches_fresh "
               << persistentSplitMatchesFresh << "\n";
        output << "error_within_tolerance " << errorWithinTolerance << "\n";
        output << "first_epoch " << firstEpoch << "\n";
        output << "second_epoch " << secondEpoch << "\n";
        output << "third_epoch " << thirdEpoch << "\n";
        output << "leaf_epoch " << leafEpoch << "\n";
        output << "first_rebuild_count " << firstRebuildCount << "\n";
        output << "second_rebuild_count " << secondRebuildCount << "\n";
        output << "topology_reused " << globalChecks[0] << "\n";
        output << "rebuild_count_reused " << globalChecks[1] << "\n";
        output << "leaf_topology_rebuilt " << globalChecks[2] << "\n";
        output << "topology_rebuilt " << globalChecks[3] << "\n";
        output << "leaf_only_rebuild " << globalChecks[4] << "\n";
        output << "root_process_rebuild " << globalChecks[5] << "\n";
        output << "count_only_topology_reused " << globalChecks[6] << "\n";
        output << "count_only_local_plan_reused " << globalChecks[7] << "\n";
        output << "finite_stats " << globalChecks[8] << "\n";
        output << "mismatched_domain_rejected " << globalChecks[9] << "\n";
        output << "leaf_storage_reused " << globalChecks[10] << "\n";
        output << "root_storage_reset " << globalChecks[11] << "\n";
        output << "persistent_empty_leaves " << globalChecks[12] << "\n";
        output << "persistent_topology_reused " << globalChecks[13] << "\n";
        output << "persistent_split_rebuilt " << globalChecks[14] << "\n";
        output << "persistent_merge_rebuilt " << globalChecks[15] << "\n";
        output << "individual_target_evaluation " << globalChecks[16] << "\n";
        output << "domain_growth_rebuilt " << globalChecks[17] << "\n";
        output << "resample_domain_pass " << globalChecks[18] << "\n";
        output << "resample_debt_pass " << globalChecks[19] << "\n";
        output << "resample_domain_reason_reported "
               << (resample.domainReasonReported ? 1 : 0) << "\n";
        output << "resample_frozen_kept_splitters "
               << (resample.frozenKeptSplitters ? 1 : 0) << "\n";
        output << "resample_ownership_matches_fresh "
               << (resample.ownershipMatchesFresh ? 1 : 0) << "\n";
        output << "resample_vs_fresh " << resample.resampledVsFresh << "\n";
        output << "resample_direct_error "
               << resample.resampledDirectError << "\n";
        output << "resample_initial_imbalance "
               << resample.initialImbalance << "\n";
        output << "resample_frozen_imbalance "
               << resample.frozenImbalance << "\n";
        output << "resample_resampled_imbalance "
               << resample.resampledImbalance << "\n";
        output << "resample_fresh_imbalance "
               << resample.freshImbalance << "\n";
        output << "resample_baseline_from_warm_solve "
               << (resample.baselineFromWarmSolve ? 1 : 0) << "\n";
        output << "resample_debt_triggered "
               << (resample.debtResampled ? 1 : 0) << "\n";
        output << "resample_debt_trigger_solve "
               << resample.debtResampleSolve << "\n";
        output << "resample_debt_imbalance_before "
               << resample.debtImbalanceBefore << "\n";
        output << "resample_debt_imbalance_after "
               << resample.debtImbalanceAfter << "\n";
        output << "resample_debt_at_trigger "
               << resample.debtAtTrigger << "\n";
        output << "resample_threshold_at_trigger "
               << resample.thresholdAtTrigger << "\n";
        output << "resample_max_leaf_before_trigger "
               << resample.maxLeafBeforeTrigger << "\n";
        output << "resample_max_leaf_at_trigger "
               << resample.maxLeafAtTrigger << "\n";
        output << "resample_domain_rebuilt_plans "
               << (resample.domainRebuiltPlans ? 1 : 0) << "\n";
        output << "resample_debt_rebuilt_plans "
               << (resample.debtRebuiltPlans ? 1 : 0) << "\n";
        output << "resample_debt_reset_on_trigger "
               << (resample.debtResetOnTrigger ? 1 : 0) << "\n";
        output << "resample_baseline_renewed "
               << (resample.baselineRenewed ? 1 : 0) << "\n";
        output << "resample_debt_clustered_ownership_matches_fresh "
               << (resample.debtClusteredOwnershipMatchesFresh ? 1 : 0)
               << "\n";
        output << "resample_debt_clustered_vs_fresh "
               << resample.debtClusteredVsFresh << "\n";
        output << "resample_debt_clustered_direct_error "
               << resample.debtClusteredDirectError << "\n";
        output << "resample_debt_clustered_direct_error_before "
               << resample.debtClusteredDirectErrorBefore << "\n";
        output << "lattice_resample_pass " << globalChecks[20] << "\n";
        output << "lattice_rebuilt_plans "
               << (lattice.rebuiltPlans ? 1 : 0) << "\n";
        output << "lattice_ownership_matches_fresh "
               << (lattice.ownershipMatchesFresh ? 1 : 0) << "\n";
        output << "lattice_ranks_with_ownership_change "
               << lattice.ranksWithOwnershipChange << "\n";
        output << "lattice_ranks_with_root_unchanged "
               << lattice.ranksWithRootUnchanged << "\n";
        output << "lattice_ranks_root_unchanged_ownership_changed "
               << lattice.ranksWithRootUnchangedAndOwnershipChange << "\n";
        output << "lattice_vs_fresh " << lattice.vsFresh << "\n";
        output << "lattice_direct_error " << lattice.directError << "\n";
        output << "weighted_sampler_pass " << globalChecks[21] << "\n";
        output << "weighted_sparse_direct_error "
               << sparseWeightedError << "\n";
        output << "weighted_direct_error " << weightedError << "\n";
        output << "communicator_isolation_pass " << globalChecks[22] << "\n";
        output << "communicator_isolation_direct_error "
               << isolation.directError << "\n";
        output << "environment_rules_pass " << globalChecks[23] << "\n";
        output << "environment_non_root_garbage_ignored "
               << (environment.nonRootGarbageIgnored ? 1 : 0) << "\n";
        output << "environment_root_garbage_rejected "
               << (environment.rootGarbageRejected ? 1 : 0) << "\n";
        output << "environment_samples_without_every_rejected "
               << (environment.samplesWithoutEveryRejected ? 1 : 0) << "\n";
        output << "environment_negative_interval_rejected "
               << (environment.negativeIntervalRejected ? 1 : 0) << "\n";
        output << "environment_mismatched_trace_rejected "
               << (environment.mismatchedTraceRejected ? 1 : 0) << "\n";
        output << "structural_window_pass " << globalChecks[24] << "\n";
        output << "structural_window_merges_forced_rebuilds "
               << (window.mergesForcedRebuilds ? 1 : 0) << "\n";
        output << "structural_window_first_split_solve "
               << window.firstSplitSolve << "\n";
        output << "structural_window_direct_error "
               << window.directError << "\n";
        output << "patch_forest_initial_created "
               << globalPatchForestChecks[0] << "\n";
        output << "patch_forest_identical_stable "
               << globalPatchForestChecks[1] << "\n";
        output << "patch_forest_count_only_classified "
               << globalPatchForestChecks[2] << "\n";
        output << "patch_forest_motion_patch_set_stable "
               << globalPatchForestChecks[3] << "\n";
        output << "patch_forest_motion_patch_geometry_stable "
               << globalPatchForestChecks[4] << "\n";
        output << "patch_forest_motion_structural_identity_stable "
               << globalPatchForestChecks[5] << "\n";
        output << "patch_forest_motion_node_geometry_changed "
               << globalPatchForestChecks[6] << "\n";
        output << "patch_forest_patch_creation_classified "
               << globalPatchForestChecks[7] << "\n";
        output << "patch_forest_patch_removal_classified "
               << globalPatchForestChecks[8] << "\n";
        output << "patch_forest_motion_reported_structural_change "
               << (globalPatchForestChecks[5] ? 0 : 1) << "\n";
        output << "patch_forest_lifecycle_pass "
               << patchForestLifecyclePass << "\n";
        output << "pass " << globalPass << "\n";
        const std::size_t worstScenario = static_cast<std::size_t>(
            std::max_element(globalScenarioErrors.begin(),
                             globalScenarioErrors.end()) -
            globalScenarioErrors.begin());
        std::cout << "fmm_gravity_mpi ranks=" << size
                  << " max_scaled_error=" << globalMaximumError
                  << " ordinary_max_scaled_error=" << ordinaryMaximumError
                  << " worst_scenario=" << scenarioNames[worstScenario]
                  << " worst_scenario_error=" << globalScenarioErrors[worstScenario]
                  << " fresh_split_error="
                  << globalFreshPersistentSplitError
                  << " split_vs_fresh="
                  << globalPersistentSplitVsFresh
                  << " topology_reused=" << globalChecks[0]
                  << " leaf_only_rebuild=" << globalChecks[4]
                  << " root_process_rebuild=" << globalChecks[5]
                  << " count_only_reused=" << globalChecks[6]
                  << " finite_stats=" << globalChecks[8]
                  << " domain_rejected=" << globalChecks[9]
                  << " persistent_reused=" << globalChecks[13]
                  << " persistent_merge=" << globalChecks[15]
                  << " patch_forest_lifecycle=" << patchForestLifecyclePass
                  << " motion_structural="
                  << (globalPatchForestChecks[5] ? 0 : 1)
                  << " resample_domain=" << globalChecks[18]
                  << " resample_debt=" << globalChecks[19]
                  << " frozen_imbalance=" << resample.frozenImbalance
                  << " resampled_imbalance=" << resample.resampledImbalance
                  << " debt_trigger_solve=" << resample.debtResampleSolve
                  << " lattice=" << globalChecks[20]
                  << " weighted=" << globalChecks[21]
                  << " isolation=" << globalChecks[22]
                  << " environment=" << globalChecks[23]
                  << " window=" << globalChecks[24]
                  << " window_split_solve=" << window.firstSplitSolve
                  << " pass=" << globalPass << std::endl;
    }

    MPI_Finalize();
    return globalPass ? 0 : 1;
}
