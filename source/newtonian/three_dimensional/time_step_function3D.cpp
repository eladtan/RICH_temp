#include "time_step_function3D.hpp"

TimeStepFunction3D::~TimeStepFunction3D(void) {}

UserDeterminedTimeStep::UserDeterminedTimeStep(
    double const initial_dt
) : current_dt(initial_dt) {}

void UserDeterminedTimeStep::SetTimeStep(
    double const user_dt
){
    current_dt = user_dt;
}

double UserDeterminedTimeStep::operator()(
    const Tessellation3D& /* tess */, 
    const vector<ComputationalCell3D>& /* cells */,
    const EquationOfState& /* eos */, 
    const vector<Vector3D>& /* face_velocities */, 
    const double /* time */) const
{
    return current_dt;
}