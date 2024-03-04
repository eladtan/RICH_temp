/*! \file HllcStress.hpp
\brief HLLC riemann solver on an eulerian grid in 3D
\details This file is based on a code originally written by Omer Bromberg
\author Elad Steinberg
*/

#ifndef HLLCSTRESS_HPP
#define HLLCSTRESS_HPP 1

#include "RiemannSolver3D.hpp"

//! \brief HLLC Riemann solver for an Eulerian grid
class Hllc3D : public RiemannSolver3D
{
public:
	Conserved3D operator()(ComputationalCell3D const& left,	ComputationalCell3D const& right,double velocity,
		EquationOfState const& eos, Vector3D const& normaldir, Vector3D & ustar_vec, Vector3D & pstar_vec) const override;
};

#endif //HLLCSTRESS_HPP
