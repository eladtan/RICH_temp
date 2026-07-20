#ifndef POWER_LAW_OPACITY_HPP
#define POWER_LAW_OPACITY_HPP

#include "RadiationOpacity.hpp"
#include "CMMC/src/units/units.hpp"
#include "LorentzTransformation.hpp"

class MCPowerLawOpacity : public OpacityCalculator
{
public:
    MCPowerLawOpacity(double sigmaA0, double sigmaS0, double sigmaA_rho, double sigmaA_T, double sigmaS_rho, double sigmaS_T);

    inline ~MCPowerLawOpacity() override = default;

    double CalcPlanckOpacity(const ComputationalCell3D &cell) const override;

    double CalcScatteringOpacity(const ComputationalCell3D &cell) const override;

    double CalcAbsorptionOpacity(const ComputationalCell3D &cell, double energy) const override;
    
private:
    double sigmaA0;
    double sigmaS0;
    double sigmaA_rho;
    double sigmaA_T;
    double sigmaS_rho;
    double sigmaS_T;
};

#endif // POWER_LAW_OPACITY_HPP
