/*! \file Rusanov3D.hpp
  \brief Positivity-oriented local Lax-Friedrichs Riemann solver
 */

#ifndef RUSANOV3D_HPP
#define RUSANOV3D_HPP 1

#include "RiemannSolver3D.hpp"

class Rusanov3D : public RiemannSolver3D
{
public:
	Conserved3D operator()(ComputationalCell3D const& left,
		ComputationalCell3D const& right,
		double face_velocity,
		EquationOfState const& eos,
		Vector3D const& normal) const override;
};

#endif // RUSANOV3D_HPP
