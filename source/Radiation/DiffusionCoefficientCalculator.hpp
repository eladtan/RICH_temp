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

#endif // DIFFUSION_COEFFICIENT_CALCULATOR_HPP