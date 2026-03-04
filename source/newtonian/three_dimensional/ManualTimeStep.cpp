#include "ManualTimeStep.hpp"

ManualTimeStep::ManualTimeStep(double dt) : dt(dt)
{}

double ManualTimeStep::operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells, const EquationOfState& eos, const vector<Vector3D>& face_velocities, const double time)
{
    return this->dt;
}

void ManualTimeStep::SetTimeStep(double dt)
{
    this->dt = dt;
}

double ManualTimeStep::GetTimeStep(void) const
{
    return this->dt;
}

double ManualTimeStep::SuggestTimeStep(void) const
{
    return this->dt;
}