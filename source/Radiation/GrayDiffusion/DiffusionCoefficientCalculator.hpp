#ifndef DIFFUSION_COEFFICIENT_CALCULATOR_HPP
#define DIFFUSION_COEFFICIENT_CALCULATOR_HPP

#include "source/newtonian/three_dimensional/computational_cell.hpp"

//! \brief Abstract class for calculating the needed data for diffusion
class DiffusionCoefficientCalculator
{
public:
/*!
    \brief Calculates the diffusion coefficient
    \param cell The primitive variables
    \return The diffusion coefficient (default units are cm^2/sec)
*/
virtual double CalcDiffusionCoefficient(ComputationalCell3D const& cell) const = 0;
/*!
    \brief Calculates the Planck opacity
    \param cell The primitive variables
    \return The planck opacity (default units are 1/cm)
*/
virtual double CalcPlanckOpacity(ComputationalCell3D const& cell) const = 0;

virtual double CalcScatteringOpacity(ComputationalCell3D const& cell) const { return 0.0; }
};

//! D=D0*rho^alpha*T^beta, sigma_planck=sigma_planck0*rho^alpha_planck*T^beta_planck
class PowerLawOpacity: public DiffusionCoefficientCalculator
{
private:
    double const D0_, alpha_, beta_, planck0_, alpha_planck_, beta_planck_;
public:
    PowerLawOpacity(double const D0, double const alpha, double const beta,
        double const planck0, double const alpha_planck, double const beta_planck)
        : D0_(D0), alpha_(alpha), beta_(beta), planck0_(planck0), alpha_planck_(alpha_planck),
        beta_planck_(beta_planck){}

    double CalcDiffusionCoefficient(ComputationalCell3D const& cell) const override;

    double CalcPlanckOpacity(ComputationalCell3D const& cell) const override;
};

#endif // DIFFUSION_COEFFICIENT_CALCULATOR_HPP