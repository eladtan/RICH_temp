#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#ifdef RICH_MPI
#include <mpi.h>
#endif

#include "source/3D/gravity/fmm/DirectGravityReference.hpp"
#include "source/3D/gravity/fmm/FmmTaylorExpansion.hpp"
#include "source/3D/gravity/fmm/mpi/FmmGlobalDyadicLattice.hpp"
#include "source/3D/gravity/fmm/mpi/FmmPatchForest.hpp"
#include "source/misc/universal_error.hpp"

namespace
{
const Vector3D kDomainLower(-1.0, -1.0, -1.0);
const Vector3D kDomainUpper(1.0, 1.0, 1.0);

double norm(const Vector3D& value)
{
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

bool close(double first, double second, double tolerance)
{
    return std::abs(first - second) <= tolerance *
        std::max(1.0, std::max(std::abs(first), std::abs(second)));
}

bool closeVector(const Vector3D& first, const Vector3D& second, double tolerance)
{
    return close(first.x, second.x, tolerance) &&
           close(first.y, second.y, tolerance) &&
           close(first.z, second.z, tolerance);
}

std::vector<std::uint64_t> cellIdsForCount(std::size_t count)
{
    std::vector<std::uint64_t> result(count);
    for(std::size_t i = 0; i < count; ++i)
        result[i] = static_cast<std::uint64_t>(i);
    return result;
}

struct SolveOutcome
{
    std::vector<Vector3D> acceleration;
    std::vector<double> potential;
    DirectGravityErrorStats error;
    std::size_t patchCount = 0;
};

SolveOutcome solveForest(const std::vector<Vector3D>& positions,
                         const std::vector<double>& masses,
                         const FmmGravityOptions& options,
                         const FmmDistributedOptions& distributedOptions)
{
    FmmPatchForest forest;
    forest.prepare(positions, masses, cellIdsForCount(positions.size()),
                   kDomainLower, kDomainUpper, options, distributedOptions, 0);

    const FmmTaylorExpansion layout(options.expansionOrder);
    FmmM2LOperatorCache operatorCache;
    FmmSolveStats stats;
    forest.buildUpward(layout);
    forest.clearLocals(layout);
    forest.executeLocalSelf(layout, operatorCache, options.maxOperatorCacheBytes,
                            stats);
    forest.executeDirectCrossPatch(layout, stats);
    forest.applyDownward(layout);

    SolveOutcome result;
    result.acceleration.assign(positions.size(), Vector3D());
    forest.scatterAcceleration(result.acceleration);
    if(options.computePotential)
    {
        result.potential.assign(positions.size(), 0.0);
        forest.scatterPotential(result.potential);
    }
    result.patchCount = forest.patches().size();

    std::vector<Vector3D> reference;
    std::vector<double> referencePotential;
    std::vector<double>* potentialPtr =
        options.computePotential ? &referencePotential : nullptr;
    DirectGravityReference::computeAcceleration(
        positions, masses, reference, potentialPtr);
    std::vector<double> forceScale;
    DirectGravityReference::computeForceScale(positions, masses, forceScale);
    result.error = DirectGravityReference::compareAcceleration(
        reference, result.acceleration, forceScale, 1.0e-12);
    if(options.computePotential && !referencePotential.empty())
    {
        double maxPotentialError = 0.0;
        for(std::size_t i = 0; i < positions.size(); ++i)
        {
            maxPotentialError = std::max(
                maxPotentialError,
                std::abs(referencePotential[i] - result.potential[i]));
        }
        if(maxPotentialError > 0.15)
            result.error.maxRelativeError = 1.0;
    }
    return result;
}

bool testLatticeBoundaries()
{
    const FmmGlobalDyadicLattice lattice =
        FmmGlobalDyadicLattice::fromDomain(kDomainLower, kDomainUpper);
    const Vector3D lower = lattice.globalRoot().lower();
    const Vector3D upper = lattice.globalRoot().upper();

    const std::uint64_t lowerPatch = lattice.patchIdAtLevel(lower, 3);
    const std::uint64_t upperPatch = lattice.patchIdAtLevel(upper, 3);
    if(lowerPatch == 0 || upperPatch == 0)
        return false;

    const FmmRootGeometry lowerRoot = lattice.patchRootGeometry(lowerPatch);
    const FmmRootGeometry upperRoot = lattice.patchRootGeometry(upperPatch);
    if(!lowerRoot.contains(lower) || !upperRoot.contains(upper))
        return false;

    for(int level = 1; level <= 4; ++level)
    {
        const std::uint64_t parent = lattice.patchIdAtLevel(Vector3D(), level - 1);
        for(int octant = 0; octant < 8; ++octant)
        {
            const std::uint64_t child = lattice.childPatchId(parent, octant);
            if(!lattice.validateParentChild(parent, child, octant))
                return false;
        }
    }
    return !lattice.wouldOverflowLevel(FMM_MAX_TREE_DEPTH);
}

bool testPartitionCoverage(const std::vector<Vector3D>& positions,
                           const FmmDistributedOptions& distributedOptions)
{
    FmmPatchForest forest;
    forest.prepare(positions, std::vector<double>(positions.size(), 1.0),
                   cellIdsForCount(positions.size()), kDomainLower, kDomainUpper,
                   FmmGravityOptions(), distributedOptions, 0);

    std::vector<int> coverage(positions.size(), 0);
    for(const FmmLocalPatch& patch : forest.patches())
    {
        for(std::size_t inputIndex : patch.inputIndices)
            ++coverage[inputIndex];
        for(std::size_t localIndex = 0; localIndex < patch.positions.size();
            ++localIndex)
        {
            if(!patch.root.contains(patch.positions[localIndex]))
                return false;
        }
    }
    return std::all_of(coverage.begin(), coverage.end(),
                       [](int count) { return count == 1; });
}

bool testAccuracyCase(const std::vector<Vector3D>& positions,
                      const std::vector<double>& masses,
                      const FmmDistributedOptions& distributedOptions,
                      double maxRelativeError,
                      bool computePotential)
{
    FmmGravityOptions options;
    options.expansionOrder = 4;
    options.thetaCritical = 0.5;
    options.leafCapacity = 16;
    options.computePotential = computePotential;
    options.validateFinite = true;

    const SolveOutcome outcome =
        solveForest(positions, masses, options, distributedOptions);
    if(outcome.patchCount == 0 && !positions.empty())
        return false;
    if(outcome.error.maxRelativeError > maxRelativeError)
        return false;
    if(computePotential)
    {
        for(std::size_t i = 0; i < positions.size(); ++i)
        {
            if(!std::isfinite(outcome.potential[i]))
                return false;
        }
    }
    return true;
}

std::vector<Vector3D> uniformCube(std::size_t count)
{
    std::vector<Vector3D> result;
    result.reserve(count);
    for(std::size_t i = 0; i < count; ++i)
    {
        const double t = static_cast<double>(i) / static_cast<double>(count);
        result.emplace_back(-0.8 + 1.6 * (t - std::floor(t * 5.0) / 5.0),
                            0.7 * std::sin(0.31 * static_cast<double>(i)),
                            0.6 * std::cos(0.17 * static_cast<double>(i)));
    }
    return result;
}

std::vector<Vector3D> twoClumps()
{
    std::vector<Vector3D> result;
    for(int clump = 0; clump < 2; ++clump)
    {
        const double shift = clump == 0 ? -0.55 : 0.55;
        for(int i = 0; i < 18; ++i)
        {
            result.emplace_back(shift + 0.03 * static_cast<double>(i % 3 - 1),
                                0.02 * static_cast<double>((i / 3) % 3 - 1),
                                0.015 * static_cast<double>(i % 5 - 2));
        }
    }
    return result;
}

std::vector<Vector3D> thinShell()
{
    const double pi = std::acos(-1.0);
    std::vector<Vector3D> result;
    for(int i = 0; i < 48; ++i)
    {
        const double angle = 2.0 * pi * static_cast<double>(i) / 48.0;
        result.emplace_back(0.75 * std::cos(angle),
                            0.75 * std::sin(angle),
                            0.04 * static_cast<double>((i % 5) - 2));
    }
    return result;
}

std::vector<Vector3D> boundaryPoints()
{
    const FmmGlobalDyadicLattice lattice =
        FmmGlobalDyadicLattice::fromDomain(kDomainLower, kDomainUpper);
    std::vector<Vector3D> result;
    result.reserve(9);
    for(int level = 2; level <= 4; ++level)
    {
        const std::uint64_t patchId =
            lattice.patchIdAtLevel(Vector3D(0.0, 0.0, 0.0), level);
        const FmmRootGeometry root = lattice.patchRootGeometry(patchId);
        const Vector3D lower = root.lower();
        const Vector3D upper = root.upper();

        // Exercise exact dyadic faces without placing distinct particles at
        // coincident positions. The nested patches all share their lower
        // corner, while the face centres below remain unique across levels.
        result.push_back(Vector3D(lower.x, root.center.y, root.center.z));
        result.push_back(Vector3D(root.center.x, upper.y, root.center.z));
        result.push_back(Vector3D(root.center.x, root.center.y, lower.z));
    }
    return result;
}

bool testAnalyticalChildGeometry()
{
    const FmmGlobalDyadicLattice lattice =
        FmmGlobalDyadicLattice::fromDomain(kDomainLower, kDomainUpper);
    const double parentHalf = lattice.globalRoot().halfSize;
    for(int octant = 0; octant < 8; ++octant)
    {
        const std::uint64_t childId = lattice.childPatchId(1, octant);
        const FmmRootGeometry child = lattice.patchRootGeometry(childId);
        if(!close(child.halfSize, 0.5 * parentHalf, 1.0e-12))
            return false;
        const double offset = 0.5 * parentHalf;
        const Vector3D expected(
            lattice.globalRoot().center.x +
                ((octant & 4) ? offset : -offset),
            lattice.globalRoot().center.y +
                ((octant & 2) ? offset : -offset),
            lattice.globalRoot().center.z +
                ((octant & 1) ? offset : -offset));
        if(!closeVector(child.center, expected, 1.0e-10))
            return false;
    }
    return true;
}

bool testDeepPatchLookup()
{
    const FmmGlobalDyadicLattice lattice =
        FmmGlobalDyadicLattice::fromDomain(kDomainLower, kDomainUpper);
    std::uint64_t patchId = 1;
    for(int level = 0; level < 10; ++level)
        patchId = lattice.childPatchId(patchId, level % 8);

    std::vector<Vector3D> positions;
    std::vector<double> masses;
    const FmmRootGeometry root = lattice.patchRootGeometry(patchId);
    positions.push_back(root.center);
    masses.push_back(1.0);

    FmmDistributedOptions options;
    options.minimumPatchLevel = 10;
    options.targetParticlesPerPatch = 0;

    FmmPatchForest forest;
    forest.prepare(positions, masses, cellIdsForCount(1), kDomainLower,
                   kDomainUpper, FmmGravityOptions(), options, 0);
    return forest.findPatch(patchId) != std::numeric_limits<std::size_t>::max() &&
           forest.findPatch(patchId + 1) == std::numeric_limits<std::size_t>::max();
}

bool testPermutationDeterminism()
{
    std::vector<Vector3D> positions = twoClumps();
    std::vector<double> masses(positions.size(), 1.0);
    std::vector<std::uint64_t> cellIds = cellIdsForCount(positions.size());

    FmmDistributedOptions options;
    options.minimumPatchLevel = 2;
    options.maximumPatchLevel = 5;
    options.targetParticlesPerPatch = 8;

    FmmPatchForest first;
    first.prepare(positions, masses, cellIds, kDomainLower, kDomainUpper,
                  FmmGravityOptions(), options, 0);
    std::vector<std::uint64_t> firstIds;
    for(const FmmLocalPatch& patch : first.patches())
        firstIds.push_back(patch.key.patchId);

    std::vector<std::size_t> permutation(positions.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::shuffle(permutation.begin(), permutation.end(), std::mt19937(12345));

    std::vector<Vector3D> shuffledPositions(positions.size());
    std::vector<double> shuffledMasses(positions.size());
    std::vector<std::uint64_t> shuffledCellIds(positions.size());
    for(std::size_t i = 0; i < permutation.size(); ++i)
    {
        shuffledPositions[i] = positions[permutation[i]];
        shuffledMasses[i] = masses[permutation[i]];
        shuffledCellIds[i] = cellIds[permutation[i]];
    }

    FmmPatchForest second;
    second.prepare(shuffledPositions, shuffledMasses, shuffledCellIds,
                   kDomainLower, kDomainUpper, FmmGravityOptions(), options, 0);
    std::vector<std::uint64_t> secondIds;
    for(const FmmLocalPatch& patch : second.patches())
        secondIds.push_back(patch.key.patchId);

    return firstIds == secondIds && first.geometryHash() == second.geometryHash();
}

bool testAdaptiveSplitAndCap()
{
    std::vector<Vector3D> dense;
    for(int i = 0; i < 40; ++i)
        dense.emplace_back(0.01 * static_cast<double>(i % 5 - 2),
                           0.01 * static_cast<double>((i / 5) % 5 - 2),
                           0.01 * static_cast<double>(i % 7 - 3));

    FmmDistributedOptions splitOptions;
    splitOptions.minimumPatchLevel = 1;
    splitOptions.maximumPatchLevel = 4;
    splitOptions.targetParticlesPerPatch = 6;
    if(!testPartitionCoverage(dense, splitOptions))
        return false;

    FmmPatchForest forest;
    forest.prepare(dense, std::vector<double>(dense.size(), 1.0),
                   cellIdsForCount(dense.size()), kDomainLower, kDomainUpper,
                   FmmGravityOptions(), splitOptions, 0);
    if(forest.patches().size() <= 1)
        return false;

    FmmDistributedOptions capOptions = splitOptions;
    capOptions.maxLocalPatchCount = 2;
    bool capFailed = false;
    try
    {
        FmmPatchForest capped;
        capped.prepare(dense, std::vector<double>(dense.size(), 1.0),
                       cellIdsForCount(dense.size()), kDomainLower, kDomainUpper,
                       FmmGravityOptions(), capOptions, 0);
    }
    catch(const UniversalError&)
    {
        capFailed = true;
    }
    return capFailed;
}

bool testScatterAccumulation()
{
    std::vector<Vector3D> positions = uniformCube(12);
    std::vector<double> masses(positions.size(), 1.0);
    FmmDistributedOptions options;
    options.minimumPatchLevel = 1;
    options.targetParticlesPerPatch = 0;

    FmmPatchForest forest;
    forest.prepare(positions, masses, cellIdsForCount(positions.size()),
                   kDomainLower, kDomainUpper, FmmGravityOptions(), options, 0);
    for(FmmLocalPatch& patch : forest.patches())
    {
        patch.acceleration.assign(patch.positions.size(),
                                  Vector3D(1.0, 2.0, 3.0));
    }

    std::vector<Vector3D> scattered(positions.size(), Vector3D());
    forest.scatterAcceleration(scattered);
    return std::all_of(scattered.begin(), scattered.end(),
                       [](const Vector3D& value) {
                           return closeVector(value, Vector3D(1.0, 2.0, 3.0), 1.0e-12);
                       });
}

bool testCountOnlyChangeClassification()
{
    FmmGravityOptions gravity;
    gravity.leafCapacity = 16;
    FmmDistributedOptions options;
    options.minimumPatchLevel = 1;
    options.maximumPatchLevel = 1;
    options.targetParticlesPerPatch = 0;

    std::vector<Vector3D> firstPositions{
        Vector3D(-0.75, -0.75, -0.75),
        Vector3D(0.75, 0.75, 0.75)};
    std::vector<double> firstMasses(firstPositions.size(), 1.0);
    FmmPatchForest forest;
    forest.prepare(firstPositions, firstMasses,
                   cellIdsForCount(firstPositions.size()),
                   kDomainLower, kDomainUpper, gravity, options, 0);

    // Change the occupancy of only one patch while leaving its geometry and
    // tree address space unchanged. An unchanged second patch must not prevent
    // the aggregate change from being classified as count-only.
    std::vector<Vector3D> secondPositions = firstPositions;
    secondPositions.push_back(firstPositions.front());
    std::vector<double> secondMasses(secondPositions.size(), 1.0);
    const FmmPatchForestChange change = forest.prepare(
        secondPositions, secondMasses,
        cellIdsForCount(secondPositions.size()),
        kDomainLower, kDomainUpper, gravity, options, 0);
    return change.occupancyChanged && change.countOnlyChanged &&
           !change.patchSetChanged && !change.patchGeometryChanged &&
           !change.structuralTopologyChanged;
}

bool testPhysicalDomainValidation()
{
    const Vector3D lower(-1.0, -0.25, -0.25);
    const Vector3D upper(1.0, 0.25, 0.25);
    // This point lies inside the padded cubic lattice root but outside the
    // caller's physical rectangular domain.
    const std::vector<Vector3D> positions{Vector3D(0.0, 0.75, 0.0)};
    bool rejected = false;
    try
    {
        FmmPatchForest forest;
        forest.prepare(positions, std::vector<double>(1, 1.0),
                       cellIdsForCount(1), lower, upper,
                       FmmGravityOptions(), FmmDistributedOptions(), 0);
    }
    catch(const UniversalError&)
    {
        rejected = true;
    }
    return rejected;
}

bool testPersistentOptionValidation()
{
    FmmDistributedOptions options;
    options.minimumPatchLevel = 1;
    options.maximumPatchLevel = 1;
    options.persistentLocalTreeTopology = true;
    options.persistentLeafMergeFactor =
        std::numeric_limits<double>::quiet_NaN();
    bool rejected = false;
    try
    {
        FmmPatchForest forest;
        const std::vector<Vector3D> positions{Vector3D(-0.5, -0.5, -0.5)};
        forest.preparePersistent(
            positions, std::vector<double>(1, 1.0), cellIdsForCount(1),
            kDomainLower, kDomainUpper, FmmGravityOptions(), options, 0);
    }
    catch(const UniversalError&)
    {
        rejected = true;
    }
    return rejected;
}
}

int main(int argc, char** argv)
{
#ifdef RICH_MPI
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);
    const bool ownsMpi = mpiInitialized == 0;
    if(ownsMpi)
    {
        const int initResult = MPI_Init(&argc, &argv);
        if(initResult != MPI_SUCCESS)
        {
            std::cerr << "fmm_patch_forest_local: MPI_Init failed with code "
                      << initResult << std::endl;
            return 2;
        }
    }

    int worldSize = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if(worldSize != 1)
    {
        std::cerr << "fmm_patch_forest_local requires exactly one MPI rank; got "
                  << worldSize << std::endl;
        if(ownsMpi)
            MPI_Finalize();
        return 2;
    }
#endif

    FmmDistributedOptions distributedOptions;
    distributedOptions.minimumPatchLevel = 2;
    distributedOptions.maximumPatchLevel = 6;
    distributedOptions.targetParticlesPerPatch = 12;

    bool pass = true;
    try
    {
        const std::vector<Vector3D> uniform = uniformCube(64);
        const std::vector<Vector3D> clumps = twoClumps();
        const std::vector<Vector3D> shell = thinShell();
        const std::vector<Vector3D> boundaries = boundaryPoints();

        std::vector<double> uniformMasses(uniform.size());
        for(std::size_t i = 0; i < uniform.size(); ++i)
            uniformMasses[i] = 0.75 + 0.05 * static_cast<double>(i % 9);

        auto runCase = [&pass](const char* name, const auto& function) {
            std::cerr << "[ RUN      ] " << name << std::endl;
            try
            {
                const bool result = function();
                std::cerr << (result ? "[       OK ] " : "[  FAILED  ] ")
                          << name << std::endl;
                pass = result && pass;
            }
            catch(const UniversalError& error)
            {
                std::cerr << "[ EXCEPTION ] " << name << std::endl;
                reportError(error, std::cerr);
                pass = false;
            }
            catch(const std::exception& error)
            {
                std::cerr << "[ STD EXC   ] " << name << ": "
                          << error.what() << std::endl;
                pass = false;
            }
            catch(...)
            {
                std::cerr << "[ UNKNOWN   ] " << name << std::endl;
                pass = false;
            }
        };

        runCase("testLatticeBoundaries",
                [&] { return testLatticeBoundaries(); });
        runCase("testAnalyticalChildGeometry",
                [&] { return testAnalyticalChildGeometry(); });
        runCase("testDeepPatchLookup",
                [&] { return testDeepPatchLookup(); });
        runCase("testPartitionCoverage_uniform",
                [&] { return testPartitionCoverage(uniform, distributedOptions); });
        runCase("testPartitionCoverage_clumps",
                [&] { return testPartitionCoverage(clumps, distributedOptions); });
        runCase("testAccuracyCase_uniform",
                [&] {
                    return testAccuracyCase(uniform, uniformMasses,
                                            distributedOptions, 0.08, true);
                });
        runCase("testAccuracyCase_clumps",
                [&] {
                    return testAccuracyCase(
                        clumps, std::vector<double>(clumps.size(), 1.0),
                        distributedOptions, 0.08, false);
                });
        runCase("testAccuracyCase_shell",
                [&] {
                    return testAccuracyCase(
                        shell, std::vector<double>(shell.size(), 1.0),
                        distributedOptions, 0.10, false);
                });
        runCase("testAccuracyCase_boundaries",
                [&] {
                    return testAccuracyCase(
                        boundaries, std::vector<double>(boundaries.size(), 1.0),
                        distributedOptions, 0.12, false);
                });
        runCase("testPermutationDeterminism",
                [&] { return testPermutationDeterminism(); });
        runCase("testAdaptiveSplitAndCap",
                [&] { return testAdaptiveSplitAndCap(); });
        runCase("testScatterAccumulation",
                [&] { return testScatterAccumulation(); });
        runCase("testCountOnlyChangeClassification",
                [&] { return testCountOnlyChangeClassification(); });
        runCase("testPhysicalDomainValidation",
                [&] { return testPhysicalDomainValidation(); });
        runCase("testPersistentOptionValidation",
                [&] { return testPersistentOptionValidation(); });
    }
    catch(const UniversalError& error)
    {
        std::cerr << "[ EXCEPTION ] test setup" << std::endl;
        reportError(error, std::cerr);
        pass = false;
    }
    catch(const std::exception& error)
    {
        std::cerr << "[ STD EXC   ] test setup: " << error.what() << std::endl;
        pass = false;
    }
    catch(...)
    {
        std::cerr << "[ UNKNOWN   ] test setup" << std::endl;
        pass = false;
    }

    std::ofstream output("fmm_patch_forest_local_metrics.txt");
    output << "pass " << (pass ? 1 : 0) << "\n";
    output << "cases 15\n";
    std::cout << "fmm_patch_forest_local pass=" << pass << std::endl;

#ifdef RICH_MPI
    if(ownsMpi)
        MPI_Finalize();
#endif

    return pass ? 0 : 1;
}
