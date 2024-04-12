#include "Global.hpp"

GlobalTimeStep::GlobalTimeStep(HydroTimeAdvance &hydroAdvance, const PointMotion3D &pm, dt_t currentTime, const TimeStepFunction3D &tsf):
    TimeStep(hydroAdvance, pm, currentTime), tsf(tsf)
{}

dt_t GlobalTimeStep::apply()
{
    std::vector<Vector3D> points = this->tess.getMeshPoints();
    points.resize(this->tess.GetPointNo());

    auto [point_velocities, face_velocities] = this->CalculateInitialPointFaceVelocities();
    double dt = this->tsf(this->tess, this->cells, this->hydroAdvance.getEOS(), face_velocities, this->currentTime);
    this->FixPointFaceVelocities(point_velocities, face_velocities, dt);
    dt = this->tsf(this->tess, this->cells, this->hydroAdvance.getEOS(), face_velocities, this->currentTime);    
    this->hydroAdvance.beforeAdvance(this->currentTime, dt, point_velocities, face_velocities);
    this->currentTime += dt;
    this->hydroAdvance.afterAdvance(this->currentTime, dt, point_velocities, face_velocities);
    return dt;
}
