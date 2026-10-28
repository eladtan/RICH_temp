#include "flux_calculator_3d.hpp"
#include "HllcKernels.hpp"

FluxCalculator3D::~FluxCalculator3D(void) {}

void AddTracers3D(ComputationalCell3D const& left, ComputationalCell3D const& right, Conserved3D &res)
{
		size_t ntracers = left.tracers.size();
		res.Erad = (res.mass > 0 ? left.Erad : right.Erad) * res.mass;
		res.Erad_dt = (res.mass > 0 ? left.Erad_dt : right.Erad_dt) * res.mass;
		res.Erad_dt_dt = (res.mass > 0 ? left.Erad_dt_dt : right.Erad_dt_dt) * res.mass;
		for (size_t i = 0; i < ntracers; ++i)
			res.tracers[i] = (res.mass > 0 ? left.tracers[i] : right.tracers[i]) * res.mass;
		for(size_t i = 0; i < ENERGY_GROUPS_NUM; ++i)
			res.Eg[i] = (res.mass > 0 ? left.Eg[i] : right.Eg[i]) * res.mass;
}

void RotateSolveBack3D(Vector3D const& normal, ComputationalCell3D const& left, ComputationalCell3D const& right,
	Vector3D const& face_velocity,RiemannSolver3D const& rs, Conserved3D &res,EquationOfState const& eos)
{
	res = rs(left, right, hllc::NormalVelocity(normal.x, normal.y, normal.z, face_velocity.x, face_velocity.y, face_velocity.z), eos, normal);
	// Add tracers
	AddTracers3D(left, right, res);
}
