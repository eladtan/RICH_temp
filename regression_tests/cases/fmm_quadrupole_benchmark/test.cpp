#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "source/3D/gravity/GravityTree.hpp"
#include "source/3D/gravity/fmm/DirectGravityReference.hpp"
#include "source/3D/gravity/fmm/SerialFmmGravityCalculator.hpp"
#include "source/misc/universal_error.hpp"

namespace
{
typedef std::chrono::steady_clock Clock;

struct BenchmarkRow
{
    std::size_t resolution = 0;
    double directSeconds = 0.0;
    double fmmSeconds = 0.0;
    double quadrupoleSeconds = 0.0;
    double fmmScaledError = 0.0;
    double quadrupoleScaledError = 0.0;
    std::uint64_t fmmM2lCount = 0;
    std::uint64_t fmmP2pPairCount = 0;
};

double elapsedSeconds(const Clock::time_point& start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double radicalInverse(std::size_t index, unsigned int base)
{
    double result = 0.0;
    double place = 1.0 / static_cast<double>(base);
    while(index != 0)
    {
        result += static_cast<double>(index % base) * place;
        index /= base;
        place /= static_cast<double>(base);
    }
    return result;
}

std::vector<Vector3D> makePoints(std::size_t count)
{
    std::vector<Vector3D> points;
    points.reserve(count);
    for(std::size_t i = 0; i < count; ++i)
    {
        const std::size_t sequenceIndex = i + 1;
        points.push_back(Vector3D(
            1.8 * radicalInverse(sequenceIndex, 2) - 0.9,
            1.8 * radicalInverse(sequenceIndex, 3) - 0.9,
            1.8 * radicalInverse(sequenceIndex, 5) - 0.9));
    }
    return points;
}

std::vector<double> makeMasses(std::size_t count)
{
    std::vector<double> masses(count);
    double total = 0.0;
    for(std::size_t i = 0; i < count; ++i)
    {
        masses[i] = 0.5 + static_cast<double>((37 * i + 11) % 101) / 101.0;
        total += masses[i];
    }
    for(double& mass : masses)
        mass /= total;
    return masses;
}

BenchmarkRow runResolution(std::size_t resolution)
{
    const Vector3D lower(-1.0, -1.0, -1.0);
    const Vector3D upper(1.0, 1.0, 1.0);
    const std::vector<Vector3D> points = makePoints(resolution);
    const std::vector<double> masses = makeMasses(resolution);

    std::vector<Vector3D> directAcceleration;
    std::vector<double> forceScale;
    const Clock::time_point directStart = Clock::now();
    DirectGravityReference::computeAcceleration(points, masses, directAcceleration);
    DirectGravityReference::computeForceScale(points, masses, forceScale);
    const double directSeconds = elapsedSeconds(directStart);

    FmmGravityOptions fmmOptions;
    fmmOptions.expansionOrder = 4;
    fmmOptions.thetaCritical = 0.5;
    fmmOptions.leafCapacity = 32;
    fmmOptions.validateFinite = true;
    std::vector<Vector3D> fmmAcceleration;
    const Clock::time_point fmmStart = Clock::now();
    SerialFmmGravityCalculator fmm(fmmOptions);
    fmm.solve(points, masses, lower, upper, fmmAcceleration);
    const double fmmSeconds = elapsedSeconds(fmmStart);

    std::vector<MassedPoint<Vector3D>> massedPoints;
    massedPoints.reserve(resolution);
    for(std::size_t i = 0; i < resolution; ++i)
        massedPoints.push_back(MassedPoint<Vector3D>(points[i], masses[i]));
    std::vector<Vector3D> quadrupoleAcceleration(resolution);
    const Clock::time_point quadrupoleStart = Clock::now();
    GravityTree<Vector3D> quadrupoleTree(lower, upper, 0.5, true);
    quadrupoleTree.build(massedPoints);
    for(std::size_t i = 0; i < resolution; ++i)
        quadrupoleAcceleration[i] = quadrupoleTree.gravity(points[i]);
    const double quadrupoleSeconds = elapsedSeconds(quadrupoleStart);

    const DirectGravityErrorStats fmmError =
        DirectGravityReference::compareAcceleration(
            directAcceleration, fmmAcceleration, forceScale, 1e-30);
    const DirectGravityErrorStats quadrupoleError =
        DirectGravityReference::compareAcceleration(
            directAcceleration, quadrupoleAcceleration, forceScale, 1e-30);

    BenchmarkRow row;
    row.resolution = resolution;
    row.directSeconds = directSeconds;
    row.fmmSeconds = fmmSeconds;
    row.quadrupoleSeconds = quadrupoleSeconds;
    row.fmmScaledError = fmmError.maxScaledError;
    row.quadrupoleScaledError = quadrupoleError.maxScaledError;
    row.fmmM2lCount = fmm.stats().m2lCount;
    row.fmmP2pPairCount = fmm.stats().p2pPairCount;
    return row;
}

bool finiteRow(const BenchmarkRow& row)
{
    return row.directSeconds > 0.0 && std::isfinite(row.directSeconds) &&
           row.fmmSeconds > 0.0 && std::isfinite(row.fmmSeconds) &&
           row.quadrupoleSeconds > 0.0 && std::isfinite(row.quadrupoleSeconds) &&
           row.fmmScaledError >= 0.0 && std::isfinite(row.fmmScaledError) &&
           row.quadrupoleScaledError >= 0.0 &&
           std::isfinite(row.quadrupoleScaledError);
}
}

int main()
{
    try
    {
        const std::size_t resolutions[] = {256, 512, 1024, 2048, 16384};
        std::vector<BenchmarkRow> rows;
        rows.reserve(5);
        for(std::size_t resolution : resolutions)
            rows.push_back(runResolution(resolution));

        std::size_t fmmFasterRows = 0;
        std::size_t fmmMoreAccurateRows = 0;
        double maxFmmError = 0.0;
        double maxQuadrupoleError = 0.0;
        bool finite = true;
        for(const BenchmarkRow& row : rows)
        {
            finite = finite && finiteRow(row);
            fmmFasterRows += row.fmmSeconds < row.quadrupoleSeconds ? 1 : 0;
            fmmMoreAccurateRows +=
                row.fmmScaledError < row.quadrupoleScaledError ? 1 : 0;
            maxFmmError = std::max(maxFmmError, row.fmmScaledError);
            maxQuadrupoleError =
                std::max(maxQuadrupoleError, row.quadrupoleScaledError);
        }

        const BenchmarkRow& first = rows.front();
        const BenchmarkRow& last = rows.back();
        const std::uint64_t allPairs = static_cast<std::uint64_t>(last.resolution) *
            static_cast<std::uint64_t>(last.resolution - 1);
        const bool largestFmmIsFaster =
            last.fmmSeconds < last.quadrupoleSeconds;
        const bool largestAccuracyIsComparable =
            last.fmmScaledError <= 1.25 * last.quadrupoleScaledError;
        const bool passed = finite && maxFmmError < 5e-3 &&
            maxQuadrupoleError < 5e-2 && last.fmmM2lCount > 0 &&
            last.fmmP2pPairCount < allPairs && largestFmmIsFaster &&
            largestAccuracyIsComparable;

        std::ofstream output("fmm_quadrupole_benchmark_metrics.txt");
        output << std::scientific << std::setprecision(16);
        output << "columns resolution direct_seconds fmm_seconds quadrupole_seconds "
               << "fmm_scaled_error quadrupole_scaled_error fmm_m2l fmm_p2p_pairs\n";
        for(const BenchmarkRow& row : rows)
        {
            output << "row " << row.resolution << " " << row.directSeconds << " "
                   << row.fmmSeconds << " " << row.quadrupoleSeconds << " "
                   << row.fmmScaledError << " " << row.quadrupoleScaledError << " "
                   << row.fmmM2lCount << " " << row.fmmP2pPairCount << "\n";
        }
        output << "row_count " << rows.size() << "\n";
        output << "largest_resolution " << last.resolution << "\n";
        output << "largest_direct_seconds " << last.directSeconds << "\n";
        output << "largest_fmm_seconds " << last.fmmSeconds << "\n";
        output << "largest_quadrupole_seconds " << last.quadrupoleSeconds << "\n";
        output << "largest_fmm_scaled_error " << last.fmmScaledError << "\n";
        output << "largest_quadrupole_scaled_error "
               << last.quadrupoleScaledError << "\n";
        output << "max_fmm_scaled_error " << maxFmmError << "\n";
        output << "max_quadrupole_scaled_error " << maxQuadrupoleError << "\n";
        output << "fmm_runtime_growth " << last.fmmSeconds / first.fmmSeconds << "\n";
        output << "quadrupole_runtime_growth "
               << last.quadrupoleSeconds / first.quadrupoleSeconds << "\n";
        output << "fmm_faster_rows " << fmmFasterRows << "\n";
        output << "fmm_more_accurate_rows " << fmmMoreAccurateRows << "\n";
        output << "largest_fmm_m2l " << last.fmmM2lCount << "\n";
        output << "largest_fmm_p2p_pairs " << last.fmmP2pPairCount << "\n";
        output << "largest_fmm_speedup "
               << last.quadrupoleSeconds / last.fmmSeconds << "\n";
        output << "largest_fmm_to_quadrupole_error_ratio "
               << last.fmmScaledError / last.quadrupoleScaledError << "\n";
        output << "pass " << (passed ? 1 : 0) << "\n";

        std::cout << "fmm_quadrupole_benchmark N=" << last.resolution
                  << " fmm_seconds=" << last.fmmSeconds
                  << " quadrupole_seconds=" << last.quadrupoleSeconds
                  << " fmm_scaled_error=" << last.fmmScaledError
                  << " quadrupole_scaled_error=" << last.quadrupoleScaledError
                  << " pass=" << passed << std::endl;
        return passed ? 0 : 1;
    }
    catch(UniversalError const& error)
    {
        reportError(error);
        return 2;
    }
}
