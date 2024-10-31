#include "TimeStep.hpp"

std::pair<std::vector<Vector3D>, std::vector<Vector3D>> TimeStep::CalculateInitialPointFaceVelocities() const
{
    std::vector<Vector3D> point_vel, face_vel;
	this->pm(this->tess, this->cells, this->currentTime, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(this->tess, point_vel, true);
#endif
	CalcFaceVelocities(this->tess, point_vel, face_vel);
    return {point_vel, face_vel};
}

void TimeStep::FixPointFaceVelocities(std::vector<Vector3D> &point_vel, std::vector<Vector3D> &face_vel, dt_t dt) const
{
	// double dt = tsc_(tess_, cells_, eos_, face_vel, pt_.getTime());
	this->pm.ApplyFix(this->tess, this->cells, this->currentTime, dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(this->tess, point_vel, true);
#endif
	CalcFaceVelocities(this->tess, point_vel, face_vel);
}
