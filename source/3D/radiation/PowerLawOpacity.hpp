#ifndef POWER_LAW_OPACITY_HPP
#define POWER_LAW_OPACITY_HPP

#include <random>
#include "RadiationOpacity.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include "LorentzTransformation.hpp"

class MCPowerLawOpacity : public RadiationOpacity
{
public:
    MCPowerLawOpacity(double sigmaA0, double sigmaS0, double sigmaA_rho, double sigmaA_T, double sigmaS_rho, double sigmaS_T);

    inline ~MCPowerLawOpacity() override = default;

    double getPlanckOpacity(const ComputationalCell3D &cell) const override;

    double getScatteringOpacity(const ComputationalCell3D &cell) const override;

    Vector3D getRandomVelocity(const ComputationalCell3D &cell) const override;

    Vector3D getNewScatterVelocity(const ComputationalCell3D &cell, const MCParticle &particle) const override;

private:
    double sigmaA0;
    double sigmaS0;
    double sigmaA_rho;
    double sigmaA_T;
    double sigmaS_rho;
    double sigmaS_T;
    mutable std::mt19937_64 rng;
};

#endif // POWER_LAW_OPACITY_HPP