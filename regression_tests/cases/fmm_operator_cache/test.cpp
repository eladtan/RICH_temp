#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

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

        const double repeatedDifference =
            maxDifference(firstAcceleration, secondAcceleration);
        const double fallbackDifference =
            maxDifference(firstAcceleration, zeroAcceleration);
        const bool boundedPass =
            firstStats.m2lCount > 0 &&
            firstStats.localOperatorCacheBytes <= cacheBudget &&
            firstStats.localOperatorCacheEntries <=
                firstStats.localOperatorCacheMaxEntries &&
            firstStats.localOperatorCacheMisses > 0 &&
            firstStats.localOperatorCacheBypasses > 0 &&
            secondStats.localOperatorCacheBytes <= cacheBudget &&
            secondStats.localOperatorCacheHits > 0;
        const bool zeroPass =
            zeroStats.localOperatorCacheBytes == 0 &&
            zeroStats.localOperatorCacheEntries == 0 &&
            zeroStats.localOperatorCacheMisses > 0 &&
            zeroStats.localOperatorCacheBypasses ==
                zeroStats.localOperatorCacheMisses;
        const bool numericalPass = repeatedDifference <= 1e-13 &&
                                   fallbackDifference <= 1e-13;
        const bool passed = boundedPass && zeroPass && numericalPass;

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
        output << "pass " << (passed ? 1 : 0) << "\n";

        std::cout << "fmm_operator_cache particles=" << points.size()
                  << " cache_bytes=" << firstStats.localOperatorCacheBytes
                  << " warm_hits=" << secondStats.localOperatorCacheHits
                  << " bypasses=" << firstStats.localOperatorCacheBypasses
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
