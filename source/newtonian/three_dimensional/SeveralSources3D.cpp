#include "SeveralSources3D.hpp"

void SeveralSources3D::operator()(const Tessellation3D &tess, const vector<ComputationalCell3D> &cells,
                                  const vector<Conserved3D> &fluxes, const vector<Vector3D> &point_velocities, const double t, double dt,
                                  vector<Conserved3D> &extensives) const
{
    size_t const Nforces = sources_.size();
    for(size_t i = 0; i < Nforces; ++i)
        sources_[i]->operator()(tess, cells, fluxes, point_velocities, t, dt, extensives);
}

double SeveralSources3D::SuggestInverseTimeStep(void) const
{
    size_t const Nforces = sources_.size();
    double max_inverse = 0;
    for(size_t i = 0; i < Nforces; ++i)
        max_inverse = std::max(max_inverse, sources_[i]->SuggestInverseTimeStep());
    return max_inverse;
}

bool SeveralSources3D::SupportsIndividualTimeSteps(void) const
{
	std::size_t acceleration_caches = 0;
    for(auto const& source : sources_)
    {
        if(!source->SupportsIndividualTimeSteps())
            return false;
		if(source->UsesIndividualAccelerationCache() && ++acceleration_caches > 1)
			return false;
	}
    return true;
}

void SeveralSources3D::ApplyIndividual(
    const Tessellation3D& tess,
    const vector<ComputationalCell3D>& cells,
    const vector<Conserved3D>& fluxes,
    const vector<Vector3D>& point_velocities,
    double time,
    const IndividualStepContext& context,
    IndividualSourcePhase phase,
    vector<Conserved3D>& extensives) const
{
    for(auto const& source : sources_)
        source->ApplyIndividual(tess, cells, fluxes, point_velocities,
                                time, context, phase, extensives);
}

void SeveralSources3D::SuggestIndividualTimeSteps(
    const Tessellation3D& tess,
    const vector<ComputationalCell3D>& cells,
    const IndividualStepContext& context,
    vector<double>& time_step_limits) const
{
    for(auto const& source : sources_)
        source->SuggestIndividualTimeSteps(tess, cells, context,
                                           time_step_limits);
}

void SeveralSources3D::SynchronizedIndividualLimits(
    const Tessellation3D& tess,
    const vector<ComputationalCell3D>& cells,
    const vector<Conserved3D>& extensives,
    double time,
    vector<double>& limits,
    vector<Vector3D>& accelerations) const
{
    // At most one constituent keeps the acceleration cache
    // (SupportsIndividualTimeSteps), so at most one fills `accelerations`.
    for(auto const& source : sources_)
        source->SynchronizedIndividualLimits(tess, cells, extensives, time,
                                             limits, accelerations);
}

bool SeveralSources3D::SupportsPartialMesh(void) const
{
    for(auto const& source : sources_)
        if(!source->SupportsPartialMesh())
            return false;
    return true;
}

bool SeveralSources3D::SupportsIndividualAccelerationRefresh(void) const
{
	for(auto const& source : sources_)
		if(source->UsesIndividualAccelerationCache())
			return source->SupportsIndividualAccelerationRefresh();
	return false;
}

void SeveralSources3D::RefreshIndividualAccelerations(
	const Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells,
	const vector<Conserved3D>& extensives,
	double time,
	vector<Vector3D>& accelerations) const
{
	for(auto const& source : sources_)
		if(source->UsesIndividualAccelerationCache())
		{
			source->RefreshIndividualAccelerations(tess, cells, extensives,
				time, accelerations);
			return;
		}
}

bool SeveralSources3D::UsesIndividualAccelerationCache(void) const
{
	for(auto const& source : sources_)
		if(source->UsesIndividualAccelerationCache())
			return true;
	return false;
}

bool SeveralSources3D::IndividualFirstHalfNeedsGeometry(
	const IndividualStepContext& context) const
{
	for(auto const& source : sources_)
		if(source->IndividualFirstHalfNeedsGeometry(context))
			return true;
	return false;
}

void SeveralSources3D::ApplyIndividualFirstHalfFromCache(
	const vector<ComputationalCell3D>& cells,
	const vector<Vector3D>& point_velocities,
	double time,
	const IndividualStepContext& context,
	vector<Conserved3D>& extensives) const
{
	for(auto const& source : sources_)
		source->ApplyIndividualFirstHalfFromCache(cells, point_velocities,
			time, context, extensives);
}
