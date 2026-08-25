#include "point_motion_3d.hpp"

PointMotion3D::~PointMotion3D(void) {}

void PointMotion3D::ApplyFix(Tessellation3D const& /*tess*/, vector<ComputationalCell3D> const& /*cells*/, double /*time*/,
	double /*dt*/, vector<Vector3D> &/*velocities*/)const
{
	return;
}

void PointMotion3D::ApplyFixIndividual(
	Tessellation3D const& tess,
	vector<ComputationalCell3D> const& cells,
	vector<ComputationalCell3D> const& /*all_cells*/, double time, double dt,
	vector<Vector3D>& velocities,
	vector<Vector3D>& all_velocities) const
{
	ApplyFix(tess, cells, time, dt, velocities);
	tess.SyncPartialBuildData(velocities, all_velocities);
}
