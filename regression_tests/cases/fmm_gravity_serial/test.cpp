#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "source/3D/gravity/fmm/DirectGravityReference.hpp"
#include "source/3D/gravity/fmm/FmmExpansionLayout.hpp"
#include "source/3D/gravity/fmm/FmmTree.hpp"
#include "source/3D/gravity/fmm/LaplaceSolidHarmonics.hpp"
#include "source/3D/gravity/fmm/SerialFmmGravityCalculator.hpp"
#include "source/misc/universal_error.hpp"

namespace
{
double norm(const Vector3D& value)
{
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

bool close(double first, double second, double tolerance)
{
    return std::abs(first - second) <= tolerance *
        std::max(1.0, std::max(std::abs(first), std::abs(second)));
}

std::vector<Vector3D> clusteredPoints()
{
    std::vector<Vector3D> result;
    for(int side = -1; side <= 1; side += 2)
    {
        for(int i = 0; i < 12; ++i)
        {
            const double x = 0.65 * side + 0.014 * static_cast<double>(i % 3 - 1);
            const double y = 0.021 * static_cast<double>(2 * ((i / 3) % 2) - 1) +
                             0.003 * static_cast<double>(i);
            const double z = 0.017 * (static_cast<double>(i % 4) - 1.5);
            result.push_back(Vector3D(x, y, z));
        }
    }
    return result;
}

std::vector<double> clusteredMasses(std::size_t count)
{
    std::vector<double> result(count);
    for(std::size_t i = 0; i < count; ++i)
        result[i] = 0.5 + 0.07 * static_cast<double>(i % 7);
    return result;
}

struct SolveResult
{
    std::vector<Vector3D> acceleration;
    std::vector<double> potential;
    FmmSolveStats stats;
    DirectGravityErrorStats error;
    double maxPotentialError = 0.0;
    double maxRelativePotentialError = 0.0;
};

SolveResult solveAndCompare(const std::vector<Vector3D>& points,
                            const std::vector<double>& masses,
                            int order,
                            double theta,
                            std::size_t leafCapacity)
{
    FmmGravityOptions options;
    options.expansionOrder = order;
    options.thetaCritical = theta;
    options.leafCapacity = leafCapacity;
    options.computePotential = true;
    options.validateFinite = true;

    SerialFmmGravityCalculator solver(options);
    SolveResult result;
    solver.solve(points, masses, Vector3D(-1, -1, -1), Vector3D(1, 1, 1),
                 result.acceleration, &result.potential);
    result.stats = solver.stats();

    std::vector<Vector3D> referenceAcceleration;
    std::vector<double> referencePotential;
    std::vector<double> forceScale;
    DirectGravityReference::computeAcceleration(points, masses,
                                                referenceAcceleration,
                                                &referencePotential);
    DirectGravityReference::computeForceScale(points, masses, forceScale);
    result.error = DirectGravityReference::compareAcceleration(
        referenceAcceleration, result.acceleration, forceScale, 1e-30);
    for(std::size_t i = 0; i < points.size(); ++i)
    {
        const double absoluteError =
            std::abs(referencePotential[i] - result.potential[i]);
        result.maxPotentialError = std::max(result.maxPotentialError, absoluteError);
        result.maxRelativePotentialError = std::max(result.maxRelativePotentialError,
            absoluteError / std::max(std::abs(referencePotential[i]), 1e-30));
    }
    return result;
}

bool analyticCasesPass()
{
    FmmGravityOptions options;
    options.leafCapacity = 2;
    options.computePotential = true;
    SerialFmmGravityCalculator solver(options);
    std::vector<Vector3D> acceleration;
    std::vector<double> potential;

    solver.solve(std::vector<Vector3D>(), std::vector<double>(),
                 Vector3D(-1, -1, -1), Vector3D(1, 1, 1),
                 acceleration, &potential);
    if(!acceleration.empty() || !potential.empty() ||
       solver.stats().particleCount != 0 || solver.stats().nodeCount != 0)
        return false;

    const std::vector<Vector3D> onePoint(1, Vector3D(0.1, -0.2, 0.3));
    const std::vector<double> oneMass(1, 2.0);
    solver.solve(onePoint, oneMass, Vector3D(-1, -1, -1), Vector3D(1, 1, 1),
                 acceleration, &potential);
    if(norm(acceleration[0]) != 0.0 || potential[0] != 0.0 ||
       !close(solver.stats().rootMass, 2.0, 1e-15))
        return false;

    std::vector<Vector3D> twoPoints;
    twoPoints.push_back(Vector3D(-0.25, 0, 0));
    twoPoints.push_back(Vector3D(0.25, 0, 0));
    std::vector<double> twoMasses;
    twoMasses.push_back(2.0);
    twoMasses.push_back(3.0);
    solver.solve(twoPoints, twoMasses, Vector3D(-1, -1, -1), Vector3D(1, 1, 1),
                 acceleration, &potential);
    return close(acceleration[0].x, 12.0, 1e-14) &&
           close(acceleration[1].x, -8.0, 1e-14) &&
           close(potential[0], 6.0, 1e-14) &&
           close(potential[1], 4.0, 1e-14) &&
           solver.stats().p2pPairCount == 2;
}

bool harmonicCasesPass()
{
    const FmmExpansionLayout layout(FMM_MAX_ORDER);
    std::vector<bool> used(layout.coefficientCount(), false);
    for(int n = 0; n <= layout.order(); ++n)
    {
        const std::size_t zero = layout.index(n, 0);
        if(used[zero])
            return false;
        used[zero] = true;
        for(int m = 1; m <= n; ++m)
        {
            const std::size_t realIndex = layout.indexReal(n, m);
            const std::size_t imagIndex = layout.indexImag(n, m);
            if(used[realIndex] || used[imagIndex] ||
               layout.index(n, m) != realIndex || layout.index(n, -m) != imagIndex)
                return false;
            used[realIndex] = true;
            used[imagIndex] = true;
        }
    }
    if(std::find(used.begin(), used.end(), false) != used.end())
        return false;

    const Vector3D displacement(0.2, -0.3, 0.4);
    std::vector<double> regular;
    std::vector<double> singular;
    LaplaceSolidHarmonics::fillRegular(displacement, layout, regular);
    LaplaceSolidHarmonics::fillSingular(displacement, layout, singular);
    const double r2 = displacement.x * displacement.x +
                      displacement.y * displacement.y +
                      displacement.z * displacement.z;
    const double invR3 = 1.0 / (std::sqrt(r2) * r2);
    if(!close(regular[layout.index(0, 0)], 1.0, 1e-15) ||
       !close(regular[layout.index(1, 0)], displacement.z, 1e-15) ||
       !close(regular[layout.indexReal(1, 1)], displacement.x, 1e-15) ||
       !close(regular[layout.indexImag(1, 1)], displacement.y, 1e-15) ||
       !close(regular[layout.index(2, 0)], 0.5 * (3.0 * displacement.z * displacement.z - r2), 1e-15) ||
       !close(singular[layout.indexReal(1, 1)], -displacement.x * invR3, 1e-14) ||
       !close(singular[layout.indexImag(1, 1)], -displacement.y * invR3, 1e-14))
        return false;
    for(double value : regular)
        if(!std::isfinite(value))
            return false;
    for(double value : singular)
        if(!std::isfinite(value))
            return false;
    return regular[layout.indexReal(FMM_MAX_ORDER, FMM_MAX_ORDER)] != 0.0;
}

bool clusteredTreePasses()
{
    std::vector<Vector3D> points;
    std::vector<double> masses;
    for(int i = 0; i < 16; ++i)
    {
        const double offset = 1e-6 * static_cast<double>(i);
        points.push_back(Vector3D(0.9 + offset, 0.9 + 0.7 * offset,
                                  0.9 + 0.3 * offset));
        masses.push_back(1.0);
    }
    FmmGravityOptions options;
    options.leafCapacity = 2;
    options.thetaCritical = 0.4;
    SerialFmmGravityCalculator solver(options);
    std::vector<Vector3D> acceleration;
    solver.solve(points, masses, Vector3D(-1, -1, -1), Vector3D(1, 1, 1),
                 acceleration);
    return solver.stats().maxLeafOccupancy <= options.leafCapacity &&
           solver.stats().maxDepth > 1 && solver.stats().nodeCount > 1;
}

std::vector<std::uint64_t> persistentTreeStructure(const FmmTree& tree)
{
    std::vector<std::uint64_t> result;
    result.reserve(3 * tree.nodes().size());
    for(const FmmNode& node : tree.nodes())
    {
        result.push_back(node.spatialKey);
        result.push_back(static_cast<std::uint64_t>(node.childMask));
        result.push_back(static_cast<std::uint64_t>(node.depth));
    }
    return result;
}

bool persistentTreeHysteresisPasses()
{
    FmmGravityOptions options;
    options.leafCapacity = 2;
    options.maxDepth = 8;
    const FmmRootGeometry root = FmmRootGeometry::fromDomain(
        Vector3D(-1, -1, -1), Vector3D(1, 1, 1), true);
    const std::size_t splitCapacity = 4;
    const std::size_t mergeCapacity = 1;

    FmmTree tree;
    FmmPersistentTreeStats stats;
    std::vector<Vector3D> initial = {
        Vector3D(-0.75, -0.75, -0.75),
        Vector3D(-0.25, -0.75, -0.75),
        Vector3D(0.25, -0.25, -0.25),
        Vector3D(-0.25, 0.25, -0.25),
        Vector3D(-0.25, -0.25, 0.25),
        Vector3D(0.25, 0.25, 0.25)};
    tree.buildPersistent(initial, root, options, splitCapacity,
                         mergeCapacity, true, stats);
    if(tree.nodes().empty() || tree.nodes()[0].isLeaf() ||
       tree.nodes()[0].childMask != 0xffu ||
       !stats.initializedFromScratch || stats.emptyLeaves == 0)
        return false;
    const std::vector<std::uint64_t> stableTopology =
        persistentTreeStructure(tree);

    // Move a particle into an octant that was empty. Full child
    // materialization must turn this into an occupancy-only change.
    std::vector<Vector3D> refit = initial;
    refit[4] = Vector3D(-0.25, 0.25, 0.25);
    tree.buildPersistent(refit, root, options, splitCapacity,
                         mergeCapacity, false, stats);
    if(stats.initializedFromScratch || stats.leafSplits != 0 ||
       stats.subtreeMerges != 0 ||
       persistentTreeStructure(tree) != stableTopology)
        return false;

    // Put five particles in one existing root leaf. The high threshold is
    // crossed, so exactly that leaf is refined.
    std::vector<Vector3D> split = {
        Vector3D(0.25, 0.25, 0.25),
        Vector3D(0.75, 0.25, 0.25),
        Vector3D(0.25, 0.75, 0.25),
        Vector3D(0.25, 0.25, 0.75),
        Vector3D(0.75, 0.75, 0.75),
        Vector3D(-0.75, -0.75, -0.75)};
    tree.buildPersistent(split, root, options, splitCapacity,
                         mergeCapacity, false, stats);
    if(stats.leafSplits == 0 || stats.subtreeMerges != 0 ||
       persistentTreeStructure(tree) == stableTopology)
        return false;

    // Return the refined child to one particle. Automatic merging must
    // restore the original full-octant topology.
    tree.buildPersistent(refit, root, options, splitCapacity,
                         mergeCapacity, false, stats);
    return stats.subtreeMerges > 0 &&
           persistentTreeStructure(tree) == stableTopology;
}

bool expectedFailuresPass()
{
    FmmGravityOptions options;
    options.leafCapacity = 1;
    SerialFmmGravityCalculator solver(options);
    std::vector<Vector3D> acceleration;
    bool coincidentFailed = false;
    try
    {
        std::vector<Vector3D> points(2, Vector3D(0, 0, 0));
        std::vector<double> masses(2, 1.0);
        solver.solve(points, masses, Vector3D(-1, -1, -1), Vector3D(1, 1, 1),
                     acceleration);
    }
    catch(UniversalError const&)
    {
        coincidentFailed = true;
    }

    bool containmentFailed = false;
    try
    {
        std::vector<Vector3D> points(1, Vector3D(2, 0, 0));
        std::vector<double> masses(1, 1.0);
        solver.solve(points, masses, Vector3D(-1, -1, -1), Vector3D(1, 1, 1),
                     acceleration);
    }
    catch(UniversalError const&)
    {
        containmentFailed = true;
    }
    return coincidentFailed && containmentFailed;
}
}

int main()
{
    try
    {
        const std::vector<Vector3D> points = clusteredPoints();
        const std::vector<double> masses = clusteredMasses(points.size());
        const SolveResult production = solveAndCompare(points, masses, 4, 0.5, 2);
        const SolveResult order2 = solveAndCompare(points, masses, 2, 0.5, 2);
        const SolveResult order6 = solveAndCompare(points, masses, 6, 0.5, 2);
        const SolveResult loose = solveAndCompare(points, masses, 3, 0.8, 2);
        const SolveResult tight = solveAndCompare(points, masses, 3, 0.08, 2);

        const double totalMass = [&masses]() {
            double result = 0.0;
            for(double mass : masses)
                result += mass;
            return result;
        }();
        const bool persistentHysteresis = persistentTreeHysteresisPasses();

        const bool passed =
            analyticCasesPass() && harmonicCasesPass() && clusteredTreePasses() &&
            persistentHysteresis &&
            expectedFailuresPass() &&
            production.stats.m2lCount > 0 &&
            production.stats.p2pPairCount < points.size() * (points.size() - 1) &&
            production.error.maxScaledError < 2e-5 &&
            production.maxRelativePotentialError < 5e-5 &&
            order6.error.maxScaledError < order2.error.maxScaledError &&
            tight.error.maxScaledError <= loose.error.maxScaledError + 1e-14 &&
            tight.stats.p2pPairCount >= loose.stats.p2pPairCount &&
            close(production.stats.rootMass, totalMass, 1e-13) &&
            production.stats.bytesOwned > 0 &&
            production.stats.maxTraversalStack > 0;

        std::ofstream out("fmm_gravity_serial_metrics.txt");
        out.setf(std::ios::scientific);
        out.precision(16);
        out << "particles " << points.size() << "\n";
        out << "nodes " << production.stats.nodeCount << "\n";
        out << "leaves " << production.stats.leafCount << "\n";
        out << "m2l_count " << production.stats.m2lCount << "\n";
        out << "p2p_pairs " << production.stats.p2pPairCount << "\n";
        out << "max_scaled_error " << production.error.maxScaledError << "\n";
        out << "max_potential_error " << production.maxPotentialError << "\n";
        out << "max_relative_potential_error "
            << production.maxRelativePotentialError << "\n";
        out << "order2_scaled_error " << order2.error.maxScaledError << "\n";
        out << "order6_scaled_error " << order6.error.maxScaledError << "\n";
        out << "loose_scaled_error " << loose.error.maxScaledError << "\n";
        out << "tight_scaled_error " << tight.error.maxScaledError << "\n";
        out << "persistent_hysteresis " << (persistentHysteresis ? 1 : 0) << "\n";
        out << "pass " << (passed ? 1 : 0) << "\n";

        std::cout << "fmm_gravity_serial scaled_error="
                  << production.error.maxScaledError
                  << " m2l=" << production.stats.m2lCount
                  << " p2p_pairs=" << production.stats.p2pPairCount
                  << " pass=" << passed << std::endl;
        return passed ? 0 : 1;
    }
    catch(UniversalError const& error)
    {
        reportError(error);
        return 2;
    }
}
