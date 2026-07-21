#include "PowerLawOpacity.hpp"

MCPowerLawOpacity::MCPowerLawOpacity(double sigmaA0, double sigmaS0, double sigmaA_rho, double sigmaA_T, double sigmaS_rho, double sigmaS_T)
    : sigmaA0(sigmaA0), sigmaS0(sigmaS0), sigmaA_rho(sigmaA_rho), sigmaA_T(sigmaA_T), sigmaS_rho(sigmaS_rho), sigmaS_T(sigmaS_T)
{
    this->rng = std::mt19937_64(0); // seed 0
}

double MCPowerLawOpacity::getPlanckOpacity(const ComputationalCell3D &cell) const
{
    return this->sigmaA0 * std::pow(cell.density, this->sigmaA_rho) * std::pow(cell.temperature, this->sigmaA_T);
}

double MCPowerLawOpacity::getScatteringOpacity(const ComputationalCell3D &cell) const
{
    return this->sigmaS0 * std::pow(cell.density, this->sigmaS_rho) * std::pow(cell.temperature, this->sigmaS_T);
}

Vector3D MCPowerLawOpacity::getRandomVelocity(const ComputationalCell3D &cell) const
{
    static std::uniform_real_distribution<double> dist(-1 + EPSILON, 1 - EPSILON);
    double x = dist(this->rng);
    double y = dist(this->rng);
    double z = dist(this->rng);
    Vector3D direction = normalize(Vector3D(x, y, z));
    return direction * units::clight;
}

Vector3D MCPowerLawOpacity::getNewScatterVelocity(const ComputationalCell3D &cell, const MCParticle &particle) const
{    
    static std::uniform_real_distribution<double> dist(-1 + EPSILON, 1 - EPSILON);
    double x = dist(this->rng);
    double y = dist(this->rng);
    double z = dist(this->rng);
    Vector3D direction = normalize(Vector3D(x, y, z));
    return direction * units::clight;
}