#ifndef FLECK_FACTOR_HELPER_HPP
#define FLECK_FACTOR_HELPER_HPP

#include <vector>
#include <cmath>
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "Radiation/OpacityCalculator.hpp"
#include "newtonian/common/equation_of_state.hpp"
#include "Radiation/CMMC/src/units/units.hpp"

namespace fleck_helper {

// Formula: f = 1/(1 + 4*arad*T^3*sigP*c*dt/cv)
// Matches RadiationIMC.cpp line 3613 (gamma=1 for non-Compton).
inline double computeSingleFleckFactor(
    double temperature, double density,
    tvector const &tracers,
    std::vector<std::string> const &tracerNames,
    EquationOfState const &eos,
    OpacityCalculator const &opacity,
    ComputationalCell3D const &cell,
    double dt)
{
    if (!(temperature > 0.0) || !(dt > 0.0))
        return 1.0;
    double cv = eos.dT2cv(density, temperature, tracers, tracerNames);
    double sigP = opacity.CalcPlanckOpacity(cell);
    if (cv <= 0.0 || sigP <= 0.0)
        return 1.0;
    double denom = 1.0 + 4.0 * units::arad * temperature * temperature * temperature
                   * sigP * units::clight * dt / cv;
    return 1.0 / denom;
}

inline std::vector<double> computeFleckFactors(
    std::vector<ComputationalCell3D> const &cells,
    EquationOfState const &eos,
    OpacityCalculator const &opacity,
    double dt)
{
    std::vector<double> f(cells.size(), 1.0);
    for (size_t i = 0; i < cells.size(); ++i)
    {
        f[i] = computeSingleFleckFactor(
            cells[i].temperature, cells[i].density,
            cells[i].tracers, ComputationalCell3D::tracerNames,
            eos, opacity, cells[i], dt);
    }
    return f;
}

} // namespace fleck_helper

#endif // FLECK_FACTOR_HELPER_HPP
