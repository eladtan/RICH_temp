#include "HohlraumOpacity.hpp"
#include <algorithm>
#include <cmath>

HohlraumOpacity::HohlraumOpacity() = default;

double HohlraumOpacity::CalcPlanckOpacity(const ComputationalCell3D &cell) const
{
    return CalcPlanckOpacityAtTemperature(cell, cell.temperature);
}

double HohlraumOpacity::CalcPlanckOpacityAtTemperature(const ComputationalCell3D &cell, double temperature) const
{
    if(cell.tracers[0] > 0.5)
    {
        double T_keV = temperature / units::kev_kelvin;
        T_keV = std::max(T_keV, 1e-4);
        return 300.0 * std::pow(T_keV, -3.0);
    }
    return 1e-20;
}

double HohlraumOpacity::CalcScatteringOpacity(const ComputationalCell3D &cell) const
{
    return CalcScatteringOpacityAtTemperature(cell, cell.temperature);
}

double HohlraumOpacity::CalcScatteringOpacityAtTemperature(const ComputationalCell3D &cell, double /*temperature*/) const
{
    return 0;
}

double HohlraumOpacity::CalcAbsorptionOpacity(const ComputationalCell3D &cell, double energyGroup) const
{
    return CalcAbsorptionOpacityAtTemperature(cell, energyGroup, cell.temperature);
}

double HohlraumOpacity::CalcAbsorptionOpacityAtTemperature(const ComputationalCell3D &cell, double energyGroup, double /*temperature*/) const
{
    throw UniversalError("CalcAbsorptionOpacity is not implemented yet for HohlraumOpacity");
}
