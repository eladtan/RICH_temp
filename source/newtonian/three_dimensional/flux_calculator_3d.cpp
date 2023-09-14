#include "flux_calculator_3d.hpp"

FluxCalculator3D::~FluxCalculator3D(void) {}

namespace
{
	void AddTracers(ComputationalCell3D const& left, ComputationalCell3D const& right, Conserved3D &res)
	{
		size_t ntracers = left.tracers.size();
		res.Erad = (res.mass > 0 ? left.Erad : right.Erad) * res.mass;
		res.mass_stress = (res.mass > 0 ? left.stress : right.stress) * res.mass;
		res.mass_eps = (res.mass > 0 ? left.strain_pl : right.strain_pl) * res.mass;
		res.mass_eps_dot = (res.mass > 0 ? left.strain_pl_dot : right.strain_pl_dot) * res.mass;
		for (size_t i = 0; i < ntracers; ++i)
			res.tracers[i] = (res.mass>0 ? left.tracers[i] : right.tracers[i])*res.mass;
	}
}

void RotateSolveBack3D(Vector3D const& normal, ComputationalCell3D const& left, ComputationalCell3D const& right,
	Vector3D const& face_velocity,RiemannSolver3D const& rs, Conserved3D &res,EquationOfState const& eos)
{
	res = rs(left, right, ScalarProd(normal, face_velocity),eos,normal);
	// Add tracers
	AddTracers(left, right, res);
}
