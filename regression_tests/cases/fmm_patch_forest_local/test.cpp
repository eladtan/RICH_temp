#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

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
    forest.applyDownward(layout, options.computePotential);

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
                            -0.7 + 1.4 * std::sin(0.31 * static_cast<double>(i)),
                            -0.6 + 1.2 * std::cos(0.17 * static_cast<double>(i)));
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
    std::vector<Vector3D> result;
    for(int i = 0; i < 48; ++i)
    {
        const double angle = 2.0 * M_PI * static_cast<double>(i) / 48.0;
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
    for(int level = 2; level <= 4; ++level)
    {
        const std::uint64_t patchId =
            lattice.patchIdAtLevel(Vector3D(0.0, 0.0, 0.0), level);
        const FmmRootGeometry root = lattice.patchRootGeometry(patchId);
        result.push_back(root.lower());
        result.push_back(root.upper());
        result.push_back(Vector3D(root.center.x, root.lower().y, root.center.z));
    }
    return result;
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

    std::shuffle(positions.begin(), positions.end(), std::mt19937(12345));
    std::shuffle(masses.begin(), masses.end(), std::mt19937(54321));
    std::shuffle(cellIds.begin(), cellIds.end(), std::mt19937(98765));

    FmmPatchForest second;
    second.prepare(positions, masses, cellIds, kDomainLower, kDomainUpper,
                   FmmGravityOptions(), options, 0);
    std::vector<std::uint64_t> secondIds;
    for(const FmmLocalPatch& patch : second.patches())
        secondIds.push_back(patch.key.patchId);

    return firstIds == secondIds;
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
}

int main()
{
    FmmDistributedOptions distributedOptions;
    distributedOptions.minimumPatchLevel = 2;
    distributedOptions.maximumPatchLevel = 6;
    distributedOptions.targetParticlesPerPatch = 12;

    const std::vector<Vector3D> uniform = uniformCube(64);
    const std::vector<Vector3D> clumps = twoClumps();
    const std::vector<Vector3D> shell = thinShell();
    const std::vector<Vector3D> boundaries = boundaryPoints();

    std::vector<double> uniformMasses(uniform.size());
    for(std::size_t i = 0; i < uniform.size(); ++i)
        uniformMasses[i] = 0.75 + 0.05 * static_cast<double>(i % 9);

    const bool pass =
        testLatticeBoundaries() &&
        testPartitionCoverage(uniform, distributedOptions) &&
        testPartitionCoverage(clumps, distributedOptions) &&
        testAccuracyCase(uniform, uniformMasses, distributedOptions, 0.08, true) &&
        testAccuracyCase(clumps, std::vector<double>(clumps.size(), 1.0),
                         distributedOptions, 0.08, false) &&
        testAccuracyCase(shell, std::vector<double>(shell.size(), 1.0),
                         distributedOptions, 0.10, false) &&
        testAccuracyCase(boundaries, std::vector<double>(boundaries.size(), 1.0),
                         distributedOptions, 0.12, false) &&
        testPermutationDeterminism() &&
        testAdaptiveSplitAndCap() &&
        testScatterAccumulation();

    std::ofstream output("fmm_patch_forest_local_metrics.txt");
    output << "pass " << (pass ? 1 : 0) << "\n";
    output << "cases 9\n";
    std::cout << "fmm_patch_forest_local pass=" << pass << std::endl;
    return pass ? 0 : 1;
}
