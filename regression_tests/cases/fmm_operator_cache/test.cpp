#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "source/3D/gravity/fmm/FmmKernels.hpp"
#include "source/3D/gravity/fmm/FmmM2LOperatorCache.hpp"
#include "source/3D/gravity/fmm/FmmTaylorExpansion.hpp"
#include "source/3D/gravity/fmm/SerialFmmGravityCalculator.hpp"
#include "source/misc/universal_error.hpp"

namespace
{
std::vector<Vector3D> makePoints()
{
    std::vector<Vector3D> result;
    const int side = 12;
    result.reserve(static_cast<std::size_t>(side * side * side));
    for(int i = 0; i < side; ++i)
    {
        for(int j = 0; j < side; ++j)
        {
            for(int k = 0; k < side; ++k)
            {
                const std::size_t id = result.size();
                const double jitter = 1e-5 * static_cast<double>((37 * id) % 19);
                result.push_back(Vector3D(
                    -0.9 + 1.8 * (static_cast<double>(i) + 0.31) / side + jitter,
                    -0.9 + 1.8 * (static_cast<double>(j) + 0.47) / side - 0.5 * jitter,
                    -0.9 + 1.8 * (static_cast<double>(k) + 0.63) / side + 0.25 * jitter));
            }
        }
    }
    return result;
}

std::vector<double> makeMasses(std::size_t count)
{
    std::vector<double> result(count);
    for(std::size_t i = 0; i < count; ++i)
        result[i] = 0.5 + 0.01 * static_cast<double>((29 * i + 7) % 53);
    return result;
}

double maxDifference(const std::vector<Vector3D>& first,
                     const std::vector<Vector3D>& second)
{
    if(first.size() != second.size())
        return std::numeric_limits<double>::infinity();
    double result = 0.0;
    for(std::size_t i = 0; i < first.size(); ++i)
    {
        const Vector3D delta = first[i] - second[i];
        result = std::max(result, std::sqrt(delta.x * delta.x +
                                            delta.y * delta.y +
                                            delta.z * delta.z));
    }
    return result;
}

double maxRelativeDifference(const std::vector<double>& first,
                             const std::vector<double>& second)
{
    if(first.size() != second.size())
        return std::numeric_limits<double>::infinity();
    double result = 0.0;
    for(std::size_t i = 0; i < first.size(); ++i)
    {
        const double scale = std::max(1.0,
            std::max(std::abs(first[i]), std::abs(second[i])));
        result = std::max(result, std::abs(first[i] - second[i]) / scale);
    }
    return result;
}

std::pair<double, double> checkScaleFreeKernel()
{
    const FmmTaylorExpansion layout(4);
    const Vector3D primitiveDirection(5.0, -3.0, 2.0);
    const double physicalScale = 0.03125;
    const Vector3D physicalDirection = physicalScale * primitiveDirection;

    std::vector<double> primitiveDerivativeScratch;
    std::vector<double> physicalDerivativeScratch;
    std::vector<double> primitiveOperator;
    std::vector<double> physicalOperator;
    FmmKernels::computeM2LOperator(primitiveDirection, layout,
        primitiveDerivativeScratch, primitiveOperator);
    FmmKernels::computeM2LOperator(physicalDirection, layout,
        physicalDerivativeScratch, physicalOperator);

    std::vector<double> scaledPrimitiveOperator(primitiveOperator.size());
    const double inverseScale = 1.0 / physicalScale;
    std::vector<double> inversePowers(
        static_cast<std::size_t>(layout.order() + 2), 1.0);
    for(std::size_t degree = 1; degree < inversePowers.size(); ++degree)
        inversePowers[degree] = inversePowers[degree - 1] * inverseScale;
    for(std::size_t i = 0; i < primitiveOperator.size(); ++i)
    {
        scaledPrimitiveOperator[i] = primitiveOperator[i] *
            inversePowers[layout.m2lTerms()[i].inverseScalePower];
    }

    FmmNode source;
    FmmNode target;
    source.multipoleOffset = 0;
    target.localOffset = 0;
    std::vector<double> multipoles(layout.coefficientCount());
    for(std::size_t i = 0; i < multipoles.size(); ++i)
        multipoles[i] = 0.25 + 0.03125 * static_cast<double>(i);
    std::vector<double> physicalLocals(layout.coefficientCount(), 0.0);
    std::vector<double> scaleFreeLocals(layout.coefficientCount(), 0.0);
    FmmKernels::translateM2L(source, target, layout, multipoles,
        physicalLocals, physicalOperator, 1.0);
    FmmKernels::translateM2L(source, target, layout, multipoles,
        scaleFreeLocals, primitiveOperator, inverseScale);

    return std::make_pair(
        maxRelativeDifference(physicalOperator, scaledPrimitiveOperator),
        maxRelativeDifference(physicalLocals, scaleFreeLocals));
}

bool checkReconfigurePreservesEntries()
{
    const FmmTaylorExpansion layout(4);
    FmmM2LOperatorCache cache;
    const std::size_t budget = static_cast<std::size_t>(1) * 1024 * 1024;

    FmmNode source;
    FmmNode target;
    source.center = Vector3D(0.0, 0.0, 0.0);
    target.center = Vector3D(0.375, -0.25, 0.125);
    source.latticeAligned = 1;
    target.latticeAligned = 1;
    source.latticeId = 17;
    target.latticeId = 17;
    source.latticeCenterX = 0;
    source.latticeCenterY = 0;
    source.latticeCenterZ = 0;
    target.latticeCenterX = 6;
    target.latticeCenterY = -4;
    target.latticeCenterZ = 2;

    std::vector<double> derivativeScratch;
    std::vector<double> uncachedOperator;
    cache.configure(budget, layout.m2lTerms().size(), 1);
    cache.beginPhase();
    const FmmM2LOperatorCache::Lookup first = cache.get(
        source, target, layout, derivativeScratch, uncachedOperator);
    const std::size_t retainedEntries = cache.entries();
    if(first.coefficients == nullptr || !first.integerKey ||
       cache.misses() != 1 || cache.hits() != 0 || retainedEntries != 1)
        return false;

    // A different traversal/topology may provide a very different entry hint.
    // Reconfiguration with the same actual cache identity must retain entries.
    cache.configure(budget, layout.m2lTerms().size(), 4096);
    cache.beginPhase();
    const FmmM2LOperatorCache::Lookup second = cache.get(
        source, target, layout, derivativeScratch, uncachedOperator);
    return second.coefficients != nullptr && second.integerKey &&
           cache.entries() == retainedEntries && cache.hits() == 1 &&
           cache.misses() == 0 && cache.bypasses() == 0;
}

bool checkLruReplacement()
{
    const FmmTaylorExpansion layout(4);
    FmmM2LOperatorCache cache;
    const std::size_t budget = static_cast<std::size_t>(64) * 1024;
    cache.configure(budget, layout.m2lTerms().size(), 1024);
    const std::size_t capacity = cache.maxEntries();
    if(capacity < 3 || capacity > 1024)
        return false;

    FmmNode source;
    source.center = Vector3D(0.0, 0.0, 0.0);
    source.latticeAligned = 1;
    source.latticeId = 23;
    source.latticeCenterX = 0;
    source.latticeCenterY = 0;
    source.latticeCenterZ = 0;

    std::vector<FmmNode> targets(capacity + 1);
    for(std::size_t i = 0; i < targets.size(); ++i)
    {
        const std::int64_t x = static_cast<std::int64_t>(2 * i + 1);
        const std::int64_t y = static_cast<std::int64_t>(2 * i + 2);
        const std::int64_t z = static_cast<std::int64_t>(2 * i + 4);
        FmmNode& target = targets[i];
        target.center = Vector3D(0.125 * static_cast<double>(x),
                                 0.125 * static_cast<double>(y),
                                 0.125 * static_cast<double>(z));
        target.latticeAligned = 1;
        target.latticeId = source.latticeId;
        target.latticeCenterX = x;
        target.latticeCenterY = y;
        target.latticeCenterZ = z;
    }

    std::vector<double> derivativeScratch;
    std::vector<double> uncachedOperator;
    cache.beginPhase();
    for(std::size_t i = 0; i < capacity; ++i)
        cache.get(source, targets[i], layout, derivativeScratch,
                  uncachedOperator);
    if(cache.entries() != capacity || cache.bypasses() != 0 ||
       cache.bytesOwned() > budget)
        return false;

    // Make the original oldest entry most recent, then force one replacement.
    cache.get(source, targets[0], layout, derivativeScratch, uncachedOperator);
    cache.get(source, targets[capacity], layout, derivativeScratch,
              uncachedOperator);
    if(cache.entries() != capacity || cache.bypasses() != 0 ||
       cache.bytesOwned() > budget)
        return false;

    cache.beginPhase();
    cache.get(source, targets[0], layout, derivativeScratch, uncachedOperator);
    cache.get(source, targets[1], layout, derivativeScratch, uncachedOperator);
    return cache.hits() == 1 && cache.misses() == 1 &&
           cache.bypasses() == 0 && cache.entries() == capacity &&
           cache.bytesOwned() <= budget;
}
}

int main()
{
    try
    {
        const std::vector<Vector3D> points = makePoints();
        const std::vector<double> masses = makeMasses(points.size());
        const std::size_t cacheBudget = 4096;

        FmmGravityOptions boundedOptions;
        boundedOptions.expansionOrder = 4;
        boundedOptions.thetaCritical = 0.5;
        boundedOptions.leafCapacity = 16;
        boundedOptions.maxOperatorCacheBytes = cacheBudget;
        SerialFmmGravityCalculator bounded(boundedOptions);

        std::vector<Vector3D> firstAcceleration;
        std::vector<Vector3D> secondAcceleration;
        bounded.solve(points, masses, Vector3D(-1, -1, -1),
                      Vector3D(1, 1, 1), firstAcceleration);
        const FmmSolveStats firstStats = bounded.stats();
        bounded.solve(points, masses, Vector3D(-1, -1, -1),
                      Vector3D(1, 1, 1), secondAcceleration);
        const FmmSolveStats secondStats = bounded.stats();

        FmmGravityOptions zeroOptions = boundedOptions;
        zeroOptions.maxOperatorCacheBytes = 0;
        SerialFmmGravityCalculator zeroCache(zeroOptions);
        std::vector<Vector3D> zeroAcceleration;
        zeroCache.solve(points, masses, Vector3D(-1, -1, -1),
                        Vector3D(1, 1, 1), zeroAcceleration);
        const FmmSolveStats zeroStats = zeroCache.stats();

        FmmGravityOptions canonicalOptions = boundedOptions;
        canonicalOptions.maxOperatorCacheBytes =
            static_cast<std::size_t>(64) * 1024 * 1024;
        SerialFmmGravityCalculator canonicalCache(canonicalOptions);
        std::vector<Vector3D> canonicalAcceleration;
        canonicalCache.solve(points, masses, Vector3D(-1, -1, -1),
                             Vector3D(1, 1, 1), canonicalAcceleration);
        const FmmSolveStats canonicalStats = canonicalCache.stats();

        const double repeatedDifference =
            maxDifference(firstAcceleration, secondAcceleration);
        const double fallbackDifference =
            maxDifference(firstAcceleration, zeroAcceleration);
        const double canonicalDifference =
            maxDifference(firstAcceleration, canonicalAcceleration);
        const std::pair<double, double> kernelDifferences =
            checkScaleFreeKernel();
        const bool reconfigurePreserved =
            checkReconfigurePreservesEntries();
        const bool lruReplacement = checkLruReplacement();
        const bool boundedPass =
            firstStats.m2lCount > 0 &&
            firstStats.localOperatorCacheBytes <= cacheBudget &&
            firstStats.localOperatorCacheEntries <=
                firstStats.localOperatorCacheMaxEntries &&
            firstStats.localOperatorCacheMaxEntries > 0 &&
            firstStats.localOperatorCacheMisses > 0 &&
            firstStats.localOperatorCacheBypasses == 0 &&
            secondStats.localOperatorCacheBytes <= cacheBudget &&
            secondStats.localOperatorCacheHits > 0 &&
            secondStats.localOperatorCacheBypasses == 0;
        const bool zeroPass =
            zeroStats.localOperatorCacheBytes == 0 &&
            zeroStats.localOperatorCacheEntries == 0 &&
            zeroStats.localOperatorCacheMisses > 0 &&
            zeroStats.localOperatorCacheBypasses ==
                zeroStats.localOperatorCacheMisses;
        const FmmRootGeometry tightRoot = FmmRootGeometry::containingPoints(
            points, Vector3D(-1, -1, -1), Vector3D(1, 1, 1), 1.25);
        const FmmRootGeometry dyadicRoot =
            FmmRootGeometry::containingPointsOnDyadicLattice(
                points, Vector3D(-1, -1, -1), Vector3D(1, 1, 1),
                1.25, FMM_MAX_TREE_DEPTH);
        FmmTree dyadicTree;
        dyadicTree.build(points, dyadicRoot, canonicalOptions);
        const std::uint64_t depthAlignment =
            std::uint64_t(1) << FMM_MAX_TREE_DEPTH;
        bool dyadicRootPass = dyadicRoot.latticeAligned != 0 &&
                               dyadicRoot.latticeId != 0 &&
                               dyadicRoot.latticeHalfUnits != 0 &&
                               dyadicRoot.latticeHalfUnits % depthAlignment == 0 &&
                               dyadicRoot.halfSize <=
                                   tightRoot.halfSize * (1.0 + 1e-6);
        for(const FmmNode& node : dyadicTree.nodes())
        {
            dyadicRootPass = dyadicRootPass &&
                node.latticeAligned != 0 &&
                node.latticeId == dyadicRoot.latticeId &&
                node.latticeHalfUnits != 0;
        }

        const bool canonicalPass =
            canonicalStats.localOperatorCacheBypasses == 0 &&
            canonicalStats.localOperatorCacheHits > 0 &&
            canonicalStats.localOperatorIntegerKeyMisses ==
                canonicalStats.localOperatorCacheMisses &&
            canonicalStats.localOperatorIntegerKeyHits ==
                canonicalStats.localOperatorCacheHits;
        const bool numericalPass = repeatedDifference <= 5e-12 &&
                                   fallbackDifference <= 5e-12 &&
                                   canonicalDifference <= 5e-12 &&
                                   kernelDifferences.first <= 5e-12 &&
                                   kernelDifferences.second <= 5e-12;
        const bool passed = boundedPass && zeroPass && canonicalPass &&
                            reconfigurePreserved && lruReplacement &&
                            dyadicRootPass && numericalPass;

        std::ofstream output("fmm_operator_cache_metrics.txt");
        output.setf(std::ios::scientific);
        output.precision(16);
        output << "particles " << points.size() << "\n";
        output << "cache_budget_bytes " << cacheBudget << "\n";
        output << "first_cache_bytes "
               << firstStats.localOperatorCacheBytes << "\n";
        output << "first_cache_entries "
               << firstStats.localOperatorCacheEntries << "\n";
        output << "first_cache_max_entries "
               << firstStats.localOperatorCacheMaxEntries << "\n";
        output << "first_cache_misses "
               << firstStats.localOperatorCacheMisses << "\n";
        output << "first_cache_bypasses "
               << firstStats.localOperatorCacheBypasses << "\n";
        output << "second_cache_hits "
               << secondStats.localOperatorCacheHits << "\n";
        output << "zero_cache_bytes "
               << zeroStats.localOperatorCacheBytes << "\n";
        output << "zero_cache_entries "
               << zeroStats.localOperatorCacheEntries << "\n";
        output << "zero_cache_misses "
               << zeroStats.localOperatorCacheMisses << "\n";
        output << "zero_cache_bypasses "
               << zeroStats.localOperatorCacheBypasses << "\n";
        output << "repeated_max_difference " << repeatedDifference << "\n";
        output << "fallback_max_difference " << fallbackDifference << "\n";
        output << "canonical_max_difference " << canonicalDifference << "\n";
        output << "kernel_operator_relative_difference "
               << kernelDifferences.first << "\n";
        output << "kernel_translation_relative_difference "
               << kernelDifferences.second << "\n";
        output << "canonical_cache_entries "
               << canonicalStats.localOperatorCacheEntries << "\n";
        output << "canonical_cache_hits "
               << canonicalStats.localOperatorCacheHits << "\n";
        output << "canonical_cache_misses "
               << canonicalStats.localOperatorCacheMisses << "\n";
        output << "canonical_cache_bypasses "
               << canonicalStats.localOperatorCacheBypasses << "\n";
        output << "canonical_integer_hits "
               << canonicalStats.localOperatorIntegerKeyHits << "\n";
        output << "canonical_integer_misses "
               << canonicalStats.localOperatorIntegerKeyMisses << "\n";
        output << "reconfigure_preserved "
               << (reconfigurePreserved ? 1 : 0) << "\n";
        output << "lru_replacement " << (lruReplacement ? 1 : 0) << "\n";
        output << "dyadic_root_aligned " << (dyadicRootPass ? 1 : 0) << "\n";
        output << "pass " << (passed ? 1 : 0) << "\n";

        std::cout << "fmm_operator_cache particles=" << points.size()
                  << " cache_bytes=" << firstStats.localOperatorCacheBytes
                  << " warm_hits=" << secondStats.localOperatorCacheHits
                  << " bypasses=" << firstStats.localOperatorCacheBypasses
                  << " canonical_entries="
                  << canonicalStats.localOperatorCacheEntries
                  << " pass=" << passed << std::endl;
        return passed ? 0 : 1;
    }
    catch(const UniversalError& error)
    {
        reportError(error);
        return 2;
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        return 3;
    }
}
