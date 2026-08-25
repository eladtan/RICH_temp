#include "SourceTerm3D.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

SourceTerm3D::~SourceTerm3D(void){}

double SourceTerm3D::SuggestInverseTimeStep(void)const
{
	return 100 * std::numeric_limits<double>::min();
}

void SourceTerm3D::ApplyIndividual(const Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells,
	const vector<Conserved3D>& fluxes,
	const vector<Vector3D>& point_velocities,
	double time,
	const IndividualStepContext& context,
	IndividualSourcePhase phase,
	vector<Conserved3D>& extensives) const
{
	const double fraction = phase == IndividualSourcePhase::Full ? 1.0 : 0.5;
	for(std::size_t index : context.active_indices)
	{
		if(index >= extensives.size())
			throw std::out_of_range("Individual source cell index is out of range");
		vector<Conserved3D> candidate = extensives;
		(*this)(tess, cells, fluxes, point_velocities, time,
			fraction * context.cellTimeStep(index), candidate);
		extensives[index] = candidate[index];
	}
}

void SourceTerm3D::SuggestIndividualTimeSteps(const Tessellation3D& /*tess*/,
	const vector<ComputationalCell3D>& /*cells*/,
	const IndividualStepContext& context,
	vector<double>& time_step_limits) const
{
	const double inverse = SuggestInverseTimeStep();
	const double limit = inverse > 0 ? 1.0 / inverse : std::numeric_limits<double>::infinity();
	for(std::size_t index : context.active_indices)
		time_step_limits.at(index) = std::min(time_step_limits.at(index), limit);
}

void ZeroForce3D::operator()(const Tessellation3D& /*tess*/, const vector<ComputationalCell3D>& /*cells*/,
		const vector<Conserved3D>& /*fluxes*/, const vector<Vector3D>& /*point_velocities*/, const double /*t*/, 
		double /*dt*/, vector<Conserved3D> &/*extensives*/) const {} 
