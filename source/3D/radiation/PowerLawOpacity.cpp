#include "PowerLawOpacity.hpp"
#include "Radiation/CMMC/src/units/units.hpp"

MCPowerLawOpacity::MCPowerLawOpacity(double sigmaA0, double sigmaS0, double sigmaA_rho, double sigmaA_T, double sigmaS_rho, double sigmaS_T)
    : sigmaA0(sigmaA0), sigmaS0(sigmaS0), sigmaA_rho(sigmaA_rho), sigmaA_T(sigmaA_T), sigmaS_rho(sigmaS_rho), sigmaS_T(sigmaS_T)
{
}

double MCPowerLawOpacity::CalcPlanckOpacity(const ComputationalCell3D &cell) const
{
    return this->sigmaA0 * std::pow(cell.density, this->sigmaA_rho) * std::pow(cell.temperature, /*std::min(cell.temperature, 10 * units::kev_kelvin),*/ this->sigmaA_T);
}

double MCPowerLawOpacity::CalcScatteringOpacity(const ComputationalCell3D &cell) const
{
    return this->sigmaS0 * std::pow(cell.density, this->sigmaS_rho) * std::pow(cell.temperature, this->sigmaS_T);
}

double MCPowerLawOpacity::CalcAbsorptionOpacity(const ComputationalCell3D &cell, double energy) const
{
    throw UniversalError("CalcAbsorptionOpacity is not implemented yet for MCPowerLawOpacity");
}
