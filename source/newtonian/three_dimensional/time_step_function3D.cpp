#include "time_step_function3D.hpp"

#include <algorithm>

TimeStepFunction3D::~TimeStepFunction3D(void) {}

void TimeStepFunction3D::SuggestIndividualTimeSteps(
	const Tessellation3D& /*tess*/,
	const vector<ComputationalCell3D>& /*cells*/,
	const EquationOfState& /*eos*/,
	const vector<Vector3D>& /*face_velocities*/,
	double /*time*/,
	const IndividualStepContext& context,
	vector<double>& time_step_limits) const
{
	const double limit = SuggestTimeStep();
	for(std::size_t index : context.active_indices)
		time_step_limits.at(index) = std::min(time_step_limits.at(index), limit);
}
