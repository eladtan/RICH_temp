#include "HohlraumOpacity.hpp"
#include <algorithm>
#include <cmath>

HohlraumOpacity::HohlraumOpacity()
{
    this->rng = std::mt19937_64(0);
}

double HohlraumOpacity::getPlanckOpacity(const ComputationalCell3D &cell) const
{
    if(cell.tracers[0] > 0.5)
    {
        // Material region: sigma_a = 300 * (T/keV)^{-3}
        double T_keV = cell.temperature / units::kev_kelvin;
        T_keV = std::max(T_keV, 1e-4);
        return 300.0 * std::pow(T_keV, -3.0);
    }
    // Vacuum: essentially zero opacity
    return 1e-20;
}

double HohlraumOpacity::getScatteringOpacity(const ComputationalCell3D &cell) const
{
    return 0;
}

Vector3D HohlraumOpacity::getRandomVelocity(const ComputationalCell3D &cell) const
{
    static std::uniform_real_distribution<double> dist(-1 + EPSILON, 1 - EPSILON);

    double x = dist(this->rng);
    double y = dist(this->rng);
    double z = dist(this->rng);
    Vector3D direction = normalize(Vector3D(x, y, z));
    return direction * units::clight;
}

Vector3D HohlraumOpacity::getNewScatterVelocity(const ComputationalCell3D &cell, const MCParticle &particle) const
{
    static std::uniform_real_distribution<double> dist(-1 + EPSILON, 1 - EPSILON);

    double x = dist(this->rng);
    double y = dist(this->rng);
    double z = dist(this->rng);
    Vector3D direction = normalize(Vector3D(x, y, z));
    return direction * units::clight;
}


double HohlraumOpacity::getGroupAbsorptionOpacity(const ComputationalCell3D &cell, double energyGroup) const
{
    throw UniversalError("getGroupAbsorptionOpacity is not implemented yet for HohlraumOpacity");
}