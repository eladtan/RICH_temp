#include "3D/gravity/fmm/DirectGravityReference.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "misc/universal_error.hpp"

namespace
{
double squaredMagnitude(const Vector3D& value)
{
    return value.x * value.x + value.y * value.y + value.z * value.z;
}

double magnitude(const Vector3D& value)
{
    return std::sqrt(squaredMagnitude(value));
}

void validateInputs(const std::vector<Vector3D>& positions,
                    const std::vector<double>& masses,
                    const char* caller)
{
    if(positions.size() != masses.size())
        throw UniversalError(std::string(caller) + ": positions/masses size mismatch");
    for(std::size_t i = 0; i < positions.size(); ++i)
    {
        if(!std::isfinite(positions[i].x) || !std::isfinite(positions[i].y) ||
           !std::isfinite(positions[i].z) || !std::isfinite(masses[i]))
        {
            UniversalError error(std::string(caller) + ": non-finite input");
            error.addEntry("particle", i);
            throw error;
        }
    }
}
}

void DirectGravityReference::computeAcceleration(const std::vector<Vector3D>& positions,
                                                 const std::vector<double>& masses,
                                                 std::vector<Vector3D>& acceleration,
                                                 std::vector<double>* positiveKernelPotential,
                                                 std::uint64_t* evaluatedPairs)
{
    validateInputs(positions, masses, "DirectGravityReference::computeAcceleration");
    acceleration.assign(positions.size(), Vector3D());
    if(positiveKernelPotential != nullptr)
        positiveKernelPotential->assign(positions.size(), 0.0);
    if(evaluatedPairs != nullptr)
        *evaluatedPairs = 0;

    for(std::size_t i = 0; i < positions.size(); ++i)
    {
        long double ax = 0.0L;
        long double ay = 0.0L;
        long double az = 0.0L;
        long double potential = 0.0L;
        for(std::size_t j = 0; j < positions.size(); ++j)
        {
            if(i == j)
                continue;
            const Vector3D delta = positions[i] - positions[j];
            const long double r2 = static_cast<long double>(delta.x) * delta.x +
                                   static_cast<long double>(delta.y) * delta.y +
                                   static_cast<long double>(delta.z) * delta.z;
            if(r2 == 0.0L)
            {
                UniversalError error("DirectGravityReference::computeAcceleration: coincident positions");
                error.addEntry("target_index", i);
                error.addEntry("source_index", j);
                throw error;
            }
            const long double invR = 1.0L / std::sqrt(r2);
            const long double factor = static_cast<long double>(masses[j]) * invR * invR * invR;
            ax -= factor * delta.x;
            ay -= factor * delta.y;
            az -= factor * delta.z;
            potential += static_cast<long double>(masses[j]) * invR;
            if(evaluatedPairs != nullptr)
                ++(*evaluatedPairs);
        }
        acceleration[i] = Vector3D(static_cast<double>(ax),
                                   static_cast<double>(ay),
                                   static_cast<double>(az));
        if(positiveKernelPotential != nullptr)
            (*positiveKernelPotential)[i] = static_cast<double>(potential);
    }
}

void DirectGravityReference::computeForceScale(const std::vector<Vector3D>& positions,
                                                const std::vector<double>& masses,
                                                std::vector<double>& forceScale)
{
    validateInputs(positions, masses, "DirectGravityReference::computeForceScale");
    forceScale.assign(positions.size(), 0.0);
    for(std::size_t i = 0; i < positions.size(); ++i)
    {
        long double scale = 0.0L;
        for(std::size_t j = 0; j < positions.size(); ++j)
        {
            if(i == j)
                continue;
            const Vector3D delta = positions[i] - positions[j];
            const long double r2 = static_cast<long double>(delta.x) * delta.x +
                                   static_cast<long double>(delta.y) * delta.y +
                                   static_cast<long double>(delta.z) * delta.z;
            if(r2 == 0.0L)
                throw UniversalError("DirectGravityReference::computeForceScale: coincident positions");
            scale += std::abs(static_cast<long double>(masses[j])) / r2;
        }
        forceScale[i] = static_cast<double>(scale);
        if(!std::isfinite(forceScale[i]))
            throw UniversalError("DirectGravityReference::computeForceScale: non-finite result");
    }
}

DirectGravityErrorStats DirectGravityReference::compareAcceleration(
    const std::vector<Vector3D>& reference,
    const std::vector<Vector3D>& candidate,
    const std::vector<double>& forceScale,
    double relativeFloor)
{
    if(reference.size() != candidate.size() || reference.size() != forceScale.size())
        throw UniversalError("DirectGravityReference::compareAcceleration: vector size mismatch");
    if(!(relativeFloor > 0.0) || !std::isfinite(relativeFloor))
        throw UniversalError("DirectGravityReference::compareAcceleration: invalid relative floor");

    DirectGravityErrorStats stats;
    for(std::size_t i = 0; i < reference.size(); ++i)
    {
        const double absoluteError = magnitude(candidate[i] - reference[i]);
        const double relativeDenominator = std::max(magnitude(reference[i]), relativeFloor);
        const double scaledDenominator = std::max(forceScale[i], relativeFloor);
        stats.maxAbsoluteError = std::max(stats.maxAbsoluteError, absoluteError);
        stats.maxRelativeError = std::max(stats.maxRelativeError,
                                          absoluteError / relativeDenominator);
        stats.maxScaledError = std::max(stats.maxScaledError,
                                        absoluteError / scaledDenominator);
    }
    return stats;
}
