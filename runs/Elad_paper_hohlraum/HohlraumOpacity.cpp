#include "HohlraumOpacity.hpp"
#include <algorithm>
#include <cmath>

HohlraumOpacity::HohlraumOpacity() = default;

double HohlraumOpacity::CalcPlanckOpacity(const ComputationalCell3D &cell) const
{
    if(cell.tracers[0] > 0.5)
    {
        double T_keV = cell.temperature / units::kev_kelvin;
        T_keV = std::max(T_keV, 1e-4);
        return 300.0 * std::pow(T_keV, -3.0);
    }
    return 1e-20;
}

double HohlraumOpacity::CalcScatteringOpacity(const ComputationalCell3D &cell) const
{
    return 0;
}

double HohlraumOpacity::CalcAbsorptionOpacity(const ComputationalCell3D &cell, double energyGroup) const
{
    throw UniversalError("CalcAbsorptionOpacity is not implemented yet for HohlraumOpacity");
}
