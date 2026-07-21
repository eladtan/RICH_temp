#include "LorentzTransformation.hpp"

void LorentzTransformation(Particle3D &particle, const Vector3D &velocity)
{
    double v2 = ScalarProd(velocity, velocity);
    double gamma = 1.0 / std::sqrt(1 - units::inv_clight2 * v2);
    double dopplerShift = DopplerShift(particle, velocity);
    particle.energy *= dopplerShift;
    particle.weight *= dopplerShift;
    particle.velocity = particle.velocity + velocity * ((gamma - 1) * ScalarProd(particle.velocity, velocity) / v2 - gamma);
    particle.velocity *= units::clight / abs(particle.velocity); // normalize to speed of light
}

double DopplerShift(const Particle3D &particle, const Vector3D &velocity)
{
    double v2 = ScalarProd(velocity, velocity);
    double gamma = 1.0 / std::sqrt(1 - units::inv_clight2 * v2);
    return gamma * (1 - ScalarProd(velocity, particle.velocity) * units::inv_clight2);
}
