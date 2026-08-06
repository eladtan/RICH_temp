#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "source/3D/radiation/SphericalObserver.hpp"
#include "source/Radiation/OpacityCalculator.hpp"
#include "source/misc/mesh_generator3D.hpp"

int main()
{
    constexpr size_t sampleCount = 500000;
    constexpr size_t muBins = 10;
    constexpr size_t phiBins = 20;
    constexpr size_t binCount = muBins * phiBins;
    double const pi = std::acos(-1.0);

    ComputationalCell3D cell;
    OpacityCalculator opacity;
    std::array<unsigned long long, binCount> counts{};

    Vector3D firstMoment(0.0, 0.0, 0.0);
    Vector3D secondMoment(0.0, 0.0, 0.0);
    double sumPureFourth = 0.0;
    double sumX2Y2 = 0.0;
    double sumY2Z2 = 0.0;
    double sumZ2X2 = 0.0;
    double maxSpeedRelativeError = 0.0;

    for (size_t i = 0; i < sampleCount; ++i) {
        Vector3D direction = opacity.getRandomVelocity(cell);
        double const speed = abs(direction);
        if (!(speed > 0.0) || !std::isfinite(speed)) {
            std::cerr << "Invalid sampled radiation direction\n";
            return 2;
        }
        maxSpeedRelativeError = std::max(
            maxSpeedRelativeError,
            std::abs(speed / units::clight - 1.0));
        direction *= 1.0 / speed;

        double const x2 = direction.x * direction.x;
        double const y2 = direction.y * direction.y;
        double const z2 = direction.z * direction.z;
        firstMoment += direction;
        secondMoment.x += x2;
        secondMoment.y += y2;
        secondMoment.z += z2;
        sumPureFourth += x2 * x2 + y2 * y2 + z2 * z2;
        sumX2Y2 += x2 * y2;
        sumY2Z2 += y2 * z2;
        sumZ2X2 += z2 * x2;

        double const mu = std::clamp(direction.z, -1.0, 1.0);
        double phi = std::atan2(direction.y, direction.x);
        if (phi < 0.0)
            phi += 2.0 * pi;
        size_t const muBin = std::min(
            muBins - 1,
            static_cast<size_t>(0.5 * (mu + 1.0) * muBins));
        size_t const phiBin = std::min(
            phiBins - 1,
            static_cast<size_t>(phi * phiBins / (2.0 * pi)));
        ++counts[muBin * phiBins + phiBin];
    }

    double const invSamples = 1.0 / static_cast<double>(sampleCount);
    firstMoment *= invSamples;
    secondMoment *= invSamples;
    double const pureFourthMoment = sumPureFourth * invSamples;
    double const mixedFourthMoments[3] = {
        sumX2Y2 * invSamples,
        sumY2Z2 * invSamples,
        sumZ2X2 * invSamples};

    double const maxAbsMean = std::max({
        std::abs(firstMoment.x), std::abs(firstMoment.y),
        std::abs(firstMoment.z)});
    double const maxSecondMomentError = std::max({
        std::abs(secondMoment.x - 1.0 / 3.0),
        std::abs(secondMoment.y - 1.0 / 3.0),
        std::abs(secondMoment.z - 1.0 / 3.0)});
    double const pureFourthMomentError =
        std::abs(pureFourthMoment - 3.0 / 5.0);
    double maxMixedFourthMomentError = 0.0;
    for (double const moment : mixedFourthMoments) {
        maxMixedFourthMomentError = std::max(
            maxMixedFourthMomentError,
            std::abs(moment - 1.0 / 15.0));
    }

    double const expectedPerBin =
        static_cast<double>(sampleCount) / static_cast<double>(binCount);
    double maxBinRelativeError = 0.0;
    double chiSquare = 0.0;
    for (unsigned long long const count : counts) {
        double const residual = static_cast<double>(count) - expectedPerBin;
        chiSquare += residual * residual / expectedPerBin;
        maxBinRelativeError = std::max(
            maxBinRelativeError, std::abs(residual) / expectedPerBin);
    }
    double const reducedChiSquare =
        chiSquare / static_cast<double>(binCount - 1);

    // Validate the observer weights against the actual nearest-direction bins.
    // A dense independent Fibonacci quadrature has equal-area samples, so its
    // counts converge directly to the spherical Voronoi cell solid angles.
    constexpr size_t observerCount = 32;
    constexpr size_t areaQuadratureCount = 50000;
    SphericalObserver observer(
        Vector3D(0.0, 0.0, 0.0), 1.0, observerCount);
    std::vector<Vector3D> const& observerDirections = observer.getDirections();
    std::vector<double> const& solidAngles = observer.getObserverSolidAngles();
    std::array<unsigned long long, observerCount> areaCounts{};
    for (Vector3D const& direction :
         fibonacci_sphere_directions(areaQuadratureCount)) {
        size_t nearest = 0;
        double best = -std::numeric_limits<double>::infinity();
        for (size_t p = 0; p < observerCount; ++p) {
            double const alignment = ScalarProd(direction, observerDirections[p]);
            if (alignment > best) {
                best = alignment;
                nearest = p;
            }
        }
        ++areaCounts[nearest];
    }

    double solidAngleSum = 0.0;
    double minSolidAngle = std::numeric_limits<double>::max();
    double maxSolidAngle = 0.0;
    double maxVoronoiAreaRelativeError = 0.0;
    bool solidAnglesValid = solidAngles.size() == observerCount;
    for (size_t p = 0; p < observerCount; ++p) {
        double const omega = solidAngles[p];
        solidAnglesValid = solidAnglesValid && omega > 0.0
                         && std::isfinite(omega);
        solidAngleSum += omega;
        minSolidAngle = std::min(minSolidAngle, omega);
        maxSolidAngle = std::max(maxSolidAngle, omega);
        double const empiricalOmega = 4.0 * pi
            * static_cast<double>(areaCounts[p])
            / static_cast<double>(areaQuadratureCount);
        maxVoronoiAreaRelativeError = std::max(
            maxVoronoiAreaRelativeError,
            std::abs(empiricalOmega - omega) / omega);
    }
    double const solidAngleRelativeError =
        std::abs(solidAngleSum - 4.0 * pi) / (4.0 * pi);
    double const solidAngleRatio = maxSolidAngle / minSolidAngle;

    bool const passed = maxSpeedRelativeError < 1e-12
                     && maxAbsMean < 5e-3
                     && maxSecondMomentError < 4e-3
                     && pureFourthMomentError < 4e-3
                     && maxMixedFourthMomentError < 4e-3
                     && maxBinRelativeError < 0.12
                     && reducedChiSquare < 1.5
                     && solidAnglesValid
                     && solidAngleRelativeError < 1e-12
                     && solidAngleRatio < 1.2
                     && maxVoronoiAreaRelativeError < 0.02;

    std::ofstream out("radiation_direction_sampling_metrics.txt");
    out << std::scientific << std::setprecision(16)
        << "samples " << sampleCount << "\n"
        << "max_speed_rel_error " << maxSpeedRelativeError << "\n"
        << "max_abs_mean " << maxAbsMean << "\n"
        << "max_second_moment_error " << maxSecondMomentError << "\n"
        << "pure_fourth_moment " << pureFourthMoment << "\n"
        << "pure_fourth_moment_error " << pureFourthMomentError << "\n"
        << "max_mixed_fourth_moment_error " << maxMixedFourthMomentError << "\n"
        << "max_equal_area_bin_rel_error " << maxBinRelativeError << "\n"
        << "reduced_chi_square " << reducedChiSquare << "\n"
        << "solid_angle_sum " << solidAngleSum << "\n"
        << "solid_angle_rel_error " << solidAngleRelativeError << "\n"
        << "solid_angle_min " << minSolidAngle << "\n"
        << "solid_angle_max " << maxSolidAngle << "\n"
        << "solid_angle_ratio " << solidAngleRatio << "\n"
        << "max_voronoi_area_rel_error " << maxVoronoiAreaRelativeError << "\n"
        << "pass " << (passed ? 1 : 0) << "\n";

    std::cout << "pure_fourth_moment_error = " << pureFourthMomentError << "\n"
              << "max_mixed_fourth_moment_error = " << maxMixedFourthMomentError << "\n"
              << "reduced_chi_square = " << reducedChiSquare << "\n"
              << "max_voronoi_area_rel_error = " << maxVoronoiAreaRelativeError << "\n"
              << "pass = " << (passed ? 1 : 0) << std::endl;
    return passed ? 0 : 1;
}
