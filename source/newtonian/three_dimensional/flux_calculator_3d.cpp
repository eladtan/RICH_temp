#include "flux_calculator_3d.hpp"

FluxCalculator3D::~FluxCalculator3D(void) {}

namespace
{
	void AddTracers(ComputationalCell3D const& left, ComputationalCell3D const& right, Conserved3D &res)
	{
		size_t ntracers = left.tracers.size();
		res.mass_stress *= 0;
		res.Eelast = 0;
		res.Erad = (res.mass > 0 ? left.Erad : right.Erad) * res.mass;
		res.Erad_dt = (res.mass > 0 ? left.Erad_dt : right.Erad_dt) * res.mass;
		res.Erad_dt_dt = (res.mass > 0 ? left.Erad_dt_dt : right.Erad_dt_dt) * res.mass;
		res.mass_eps = (res.mass > 0 ? left.strain_plastic : right.strain_plastic) * res.mass;
		res.mass_eps_dt = (res.mass > 0 ? left.strain_plastic_dt : right.strain_plastic_dt) * res.mass;
		for (size_t i = 0; i < ntracers; ++i)
			res.tracers[i] = (res.mass>0 ? left.tracers[i] : right.tracers[i])*res.mass;
	}
}

void RotateSolveBack3D(Vector3D const& normal, ComputationalCell3D const& left, ComputationalCell3D const& right,
	Vector3D const& face_velocity,RiemannSolver3D const& rs, Conserved3D &res,EquationOfState const& eos, Vector3D & ustar_vec, Vector3D & pstar_vec)
{
	res = rs(left, right, ScalarProd(normal, face_velocity),eos,normal,ustar_vec,pstar_vec);
	// Add tracers
	AddTracers(left, right, res);
}

void RotateSolveBack3D(Vector3D const& normal, ComputationalCell3D const& left, ComputationalCell3D const& right,
	Vector3D const& face_velocity,RiemannSolver3D const& rs, Conserved3D &res,EquationOfState const& eos, Vector3D & ustar_vec)
{
	Vector3D pstar_tmp;
	return RotateSolveBack3D(normal, left, right, face_velocity, rs, res, eos, ustar_vec, pstar_tmp);
}
