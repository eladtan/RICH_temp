/*! \file NoMixMotion.hpp
\brief Tries to move mesh points in a way that minimizes mass flux between different tracers
*/
#ifndef NOMIXMOTION_HPP
#define NOMIXMOTION_HPP 1


#include "point_motion_3d.hpp"
#include "LinearGauss3D.hpp"
#include "../common/equation_of_state.hpp"
#include "RiemannSolver3D.hpp"

class NoMixMotion : public PointMotion3D
{
public:
	NoMixMotion(const PointMotion3D& pm, SpatialReconstruction3D const& interpolation, const RiemannSolver3D & rs, const EquationOfState& eos, const vector<std::string>& no_fix = vector<std::string>());

	void operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
		double time, vector<Vector3D> &res) const override;

	void ApplyFix(Tessellation3D const& tess, vector<ComputationalCell3D> const& cells, double time,
		double dt, vector<Vector3D> &velocities)const override;
private:
	const PointMotion3D& pm_;
	SpatialReconstruction3D const & interpolation_;
	std::vector<size_t> no_fix_indeces;
	mutable std::vector<Vector3D> ustar_, lagrangian_direction_;
	mutable std::vector<char> lagrangian_cell_;
	const RiemannSolver3D& rs_;
	const EquationOfState& eos_;
};
#endif //NOMIXMOTION_HPP
