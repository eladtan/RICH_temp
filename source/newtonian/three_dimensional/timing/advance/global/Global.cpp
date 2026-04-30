#include "Global.hpp"
#include "newtonian/three_dimensional/CourantFriedrichsLewy.hpp"

GlobalTimeStep::GlobalTimeStep(HydroTimeAdvance &hydroAdvance, const PointMotion3D &pm, dt_t currentTime, TimeStepFunction3D &tsf):
    TimeStep(hydroAdvance, pm, currentTime), tsf(tsf)
{}

dt_t GlobalTimeStep::apply()
{
    std::vector<Vector3D> points = this->tess.getMeshPoints();
    points.resize(this->tess.GetPointNo());

    auto [point_velocities, face_velocities] = this->CalculateInitialPointFaceVelocities();
    auto* cfl = dynamic_cast<CourantFriedrichsLewy*>(&this->tsf);
    if (cfl)
        cfl->SetPointVelocities(&point_velocities);
    double dt = this->tsf(this->tess, this->cells, this->hydroAdvance.getEOS(), face_velocities, this->currentTime);
    this->FixPointFaceVelocities(point_velocities, face_velocities, dt);
    dt = this->tsf(this->tess, this->cells, this->hydroAdvance.getEOS(), face_velocities, this->currentTime);

    this->allPointsHelperVector.resize(this->tess.GetPointNo());
    std::iota(this->allPointsHelperVector.begin(), this->allPointsHelperVector.end(), 0);
    
    this->hydroAdvance.beforeAdvance(this->allPointsHelperVector, this->currentTime, dt);
    this->currentTime += dt;
    this->allPointsHelperVector.resize(this->tess.GetPointNo());
    std::iota(this->allPointsHelperVector.begin(), this->allPointsHelperVector.end(), 0);
    this->hydroAdvance.afterAdvance(this->allPointsHelperVector, this->currentTime, dt);
    return dt;
}
