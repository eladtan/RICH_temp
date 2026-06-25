#include "MultigroupOpacity.hpp"
#include "misc/universal_error.hpp"
#include <algorithm>
#include <cmath>

MultigroupOpacity::MultigroupOpacity(std::shared_ptr<OpacityCalculator> opacity)
    : opacity(opacity)
{
    this->cummulativeOpacityCellID = std::numeric_limits<size_t>::max();
    this->cummulativeOpacity.fill(0.0);
    this->energyCenters.fill(0.0);
    for (size_t g = 0; g < (ENERGY_GROUPS_NUM); g++)
    {
        if(std::isnan(ComputationalCell3D::energyBoundaries[g]))
        {
            UniversalError eo("Energy boundaries must be initialized - found a NaN value");
            eo.addEntry("Group", g);
            throw eo;
        }
        this->energyCenters[g] = (ComputationalCell3D::energyBoundaries[g] + ComputationalCell3D::energyBoundaries[g+1]) / 2.0;
    }
}

void MultigroupOpacity::GetCummulativeOpacity(const ComputationalCell3D &cell)
{
    double const kT = units::k_boltz * cell.temperature;
    this->cummulativeOpacity[0] = 0.0;
    for (size_t g = 1; g < (ENERGY_GROUPS_NUM + 1); g++)
    {
        double const a = ComputationalCell3D::energyBoundaries[g-1] / kT;
        double const b = ComputationalCell3D::energyBoundaries[g] / kT;
    
        double const bg = planck_integral::planck_integral(a, b);
        this->cummulativeOpacity[g] = opacity->CalcAbsorptionOpacity(cell, this->energyCenters[g-1]) * bg;
        this->cummulativeOpacity[g] += this->cummulativeOpacity[g-1];
    }
    this->cummulativeOpacityCellID = cell.ID;
}

double MultigroupOpacity::GetThermalEnergy(const ComputationalCell3D &cell, double random)
{
    if(this->cummulativeOpacityCellID != cell.ID)
    {
        this->GetCummulativeOpacity(cell);
    }
    double interp = LinearInterpolation(this->cummulativeOpacity, ComputationalCell3D::energyBoundaries, random * this->cummulativeOpacity.back());

    return interp;
}

std::array<double, ENERGY_GROUPS_NUM> MultigroupOpacity::GetThermalGroupPdf(const ComputationalCell3D &cell)
{
    if (this->cummulativeOpacityCellID != cell.ID)
    {
        this->GetCummulativeOpacity(cell);
    }
    std::array<double, ENERGY_GROUPS_NUM> pdf{};
    double const total = this->cummulativeOpacity[ENERGY_GROUPS_NUM];
    if (total <= 0.0 || !std::isfinite(total))
        return pdf;
    for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
    {
        double raw = this->cummulativeOpacity[g + 1] - this->cummulativeOpacity[g];
        pdf[g] = (raw > 0.0) ? raw / total : 0.0;
    }
    return pdf;
}

double MultigroupOpacity::SampleThermalEnergyInGroup(const ComputationalCell3D &cell, size_t group, double random)
{
    if (this->cummulativeOpacityCellID != cell.ID)
    {
        this->GetCummulativeOpacity(cell);
    }
    if (group >= ENERGY_GROUPS_NUM)
        group = ENERGY_GROUPS_NUM - 1;

    double const c0 = this->cummulativeOpacity[group];
    double const c1 = this->cummulativeOpacity[group + 1];
    if (c1 <= c0)
        return this->energyCenters[group];
    double const upperRandom = std::nextafter(1.0, 0.0);
    double const r = std::isfinite(random)
        ? std::clamp(random, 0.0, upperRandom)
        : 0.5;
    double const target = c0 + r * (c1 - c0);
    double interp = LinearInterpolation(this->cummulativeOpacity, ComputationalCell3D::energyBoundaries, target);
    return interp;
}
