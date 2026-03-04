#ifndef MANUAL_TIME_STEP_HPP
#define MANUAL_TIME_STEP_HPP

#include "time_step_function3D.hpp"

class ManualTimeStep : public TimeStepFunction3D
{
public:
    ManualTimeStep(double dt = std::numeric_limits<double>::max());

    double operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells, const EquationOfState& eos, const vector<Vector3D>& face_velocities, const double time) override;

    void SetTimeStep(double dt) override;

    double GetTimeStep(void) const override;

    double SuggestTimeStep(void) const override;

private:
    double dt;
};

#endif // MANUAL_TIME_STEP_HPP