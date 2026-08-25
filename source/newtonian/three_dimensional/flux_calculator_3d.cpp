#include "flux_calculator_3d.hpp"

FluxCalculator3D::~FluxCalculator3D(void) {}

void FluxCalculator3D::CalculateIndividual(
	vector<Conserved3D>& fluxes,
	const Tessellation3D& tess,
	const vector<Vector3D>& edge_velocities,
	const vector<ComputationalCell3D>& cells,
	const vector<Conserved3D>& extensives,
	const EquationOfState& eos,
	const IndividualStepContext& context,
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >& face_values) const
{
	Calculate(fluxes, tess, edge_velocities, cells, extensives, eos,
		context.event_time, 0, face_values);
	const std::size_t norg = context.active_mask.size();
	for(std::size_t face = 0; face < fluxes.size(); ++face)
	{
		const auto neighbors = tess.GetFaceNeighbors(face);
		const bool active = (neighbors.first < norg && context.isActive(neighbors.first)) ||
			(neighbors.second < norg && context.isActive(neighbors.second));
		if(!active)
			fluxes[face] = Conserved3D();
	}
}

namespace
{
	void AddTracers(ComputationalCell3D const& left, ComputationalCell3D const& right, Conserved3D &res)
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
}

void RotateSolveBack3D(Vector3D const& normal, ComputationalCell3D const& left, ComputationalCell3D const& right,
	Vector3D const& face_velocity,RiemannSolver3D const& rs, Conserved3D &res,EquationOfState const& eos)
{
	res = rs(left, right, ScalarProd(normal, face_velocity),eos,normal);
	// Add tracers
	AddTracers(left, right, res);
}
