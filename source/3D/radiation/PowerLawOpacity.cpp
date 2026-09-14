#include "PowerLawOpacity.hpp"
#include "CMMC/src/units/units.hpp"

MCPowerLawOpacity::MCPowerLawOpacity(double sigmaA0, double sigmaS0, double sigmaA_rho, double sigmaA_T, double sigmaS_rho, double sigmaS_T)
    : sigmaA0(sigmaA0), sigmaS0(sigmaS0), sigmaA_rho(sigmaA_rho), sigmaA_T(sigmaA_T), sigmaS_rho(sigmaS_rho), sigmaS_T(sigmaS_T)
{
}

double MCPowerLawOpacity::CalcPlanckOpacity(const ComputationalCell3D &cell) const
{
    return CalcPlanckOpacityAtTemperature(cell, cell.temperature);
}

double MCPowerLawOpacity::CalcPlanckOpacityAtTemperature(const ComputationalCell3D &cell, double temperature) const
{
    double const T = std::max(temperature, 1.0);
    return this->sigmaA0 * std::pow(cell.density, this->sigmaA_rho) * std::pow(T, this->sigmaA_T);
}

double MCPowerLawOpacity::CalcScatteringOpacity(const ComputationalCell3D &cell) const
{
    return CalcScatteringOpacityAtTemperature(cell, cell.temperature);
}

double MCPowerLawOpacity::CalcScatteringOpacityAtTemperature(const ComputationalCell3D &cell, double temperature) const
{
    double const T = std::max(temperature, 1.0);
    return this->sigmaS0 * std::pow(cell.density, this->sigmaS_rho) * std::pow(T, this->sigmaS_T);
}

double MCPowerLawOpacity::CalcAbsorptionOpacity(const ComputationalCell3D &cell, double energy) const
{
    return CalcAbsorptionOpacityAtTemperature(cell, energy, cell.temperature);
}

double MCPowerLawOpacity::CalcAbsorptionOpacityAtTemperature(const ComputationalCell3D &cell, double energy, double /*temperature*/) const
{
    throw UniversalError("CalcAbsorptionOpacity is not implemented yet for MCPowerLawOpacity");
}
