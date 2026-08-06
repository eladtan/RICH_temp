#ifndef DIRECT_GRAVITY_REFERENCE_HPP
#define DIRECT_GRAVITY_REFERENCE_HPP

#include <cstdint>
#include <vector>

#include "3D/elementary/Vector3D.hpp"

struct DirectGravityErrorStats
{
    double maxAbsoluteError = 0;
    double maxRelativeError = 0;
    double maxScaledError = 0;
};

namespace DirectGravityReference
{
void computeAcceleration(const std::vector<Vector3D>& positions,
                         const std::vector<double>& masses,
                         std::vector<Vector3D>& acceleration,
                         std::vector<double>* positiveKernelPotential = nullptr,
                         std::uint64_t* evaluatedPairs = nullptr);

void computeForceScale(const std::vector<Vector3D>& positions,
                       const std::vector<double>& masses,
                       std::vector<double>& forceScale);

DirectGravityErrorStats compareAcceleration(const std::vector<Vector3D>& reference,
                                            const std::vector<Vector3D>& candidate,
                                            const std::vector<double>& forceScale,
                                            double relativeFloor);
}

#endif // DIRECT_GRAVITY_REFERENCE_HPP
