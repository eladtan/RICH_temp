#include "DiffusionCoefficientCalculator.hpp"

double PowerLawOpacity::CalcDiffusionCoefficient(ComputationalCell3D const& cell) const
{
    return D0_ * std::pow(cell.density, alpha_) * std::pow(cell.temperature, beta_);
}

double PowerLawOpacity::CalcPlanckOpacity(ComputationalCell3D const& cell) const
{
    return planck0_ * std::pow(cell.density, alpha_planck_) * std::pow(cell.temperature, beta_planck_);
}