#include <algorithm>
#include <array>
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
    // This regression intentionally exercises the one-tree-per-rank fallback
    // and its historical process/LET invalidation counters. Patch-forest
    // production behavior is covered by fmm_patch_moving_mesh.
    distributed.enablePatchForest = false;
    distributed.maxRemoteBytes = 64u * 1024u * 1024u;
    // This regression validates the legacy count-only interaction-plan reuse
    // path. Bounded LET waves intentionally rebuild when leaf occupancy changes
    // because wave membership and payload sizing depend on current counts.
    // Dedicated wave regressions cover that execution mode.
    distributed.maxLetWaveBytes = 0;

    // Preserve the legacy sparse-tree rebuild checks below. Persistent-tree
    // execution, plan reuse, splitting, and automatic merging are exercised
    // independently in the dedicated block below.
    distributed.persistentLocalTreeTopology = false;

    double localMaximumError = 0.0;
    constexpr std::size_t scenarioCount = 10;
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
        "legacy_root_breach"};
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
    const int errorWithinTolerance = globalMaximumError < 2e-4 ? 1 : 0;
    const int localChecks[16] = {
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
        persistentMergeRebuilt ? 1 : 0};
    int globalChecks[16] = {};
    MPI_Allreduce(localChecks, globalChecks, 16, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);
    const int globalPass = errorWithinTolerance &&
                           globalChecks[0] && globalChecks[1] &&
                           globalChecks[2] && globalChecks[3] &&
                           globalChecks[4] && globalChecks[5] &&
                           globalChecks[6] && globalChecks[7] &&
                           globalChecks[8] && globalChecks[9] &&
                           globalChecks[10] && globalChecks[11] &&
                           globalChecks[12] && globalChecks[13] &&
                           globalChecks[14] && globalChecks[15];

    if(rank == 0)
    {
        std::ofstream output("fmm_gravity_mpi_metrics.txt");
        output.setf(std::ios::scientific);
        output.precision(16);
        output << "ranks " << size << "\n";
        output << "max_scaled_error " << globalMaximumError << "\n";
        for(std::size_t i = 0; i < scenarioCount; ++i)
            output << "scenario_error_" << scenarioNames[i] << " "
                   << globalScenarioErrors[i] << "\n";
        output << "persistent_split_fresh_error "
               << globalFreshPersistentSplitError << "\n";
        output << "persistent_split_vs_fresh "
               << globalPersistentSplitVsFresh << "\n";
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
        output << "pass " << globalPass << "\n";
        const std::size_t worstScenario = static_cast<std::size_t>(
            std::max_element(globalScenarioErrors.begin(),
                             globalScenarioErrors.end()) -
            globalScenarioErrors.begin());
        std::cout << "fmm_gravity_mpi ranks=" << size
                  << " max_scaled_error=" << globalMaximumError
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
                  << " pass=" << globalPass << std::endl;
    }

    MPI_Finalize();
    return globalPass ? 0 : 1;
}
