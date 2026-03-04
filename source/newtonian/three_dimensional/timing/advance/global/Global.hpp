#ifndef GLOBAL_TIMESTEP_HPP
#define GLOBAL_TIMESTEP_HPP

#ifdef RICH_MPI
    #include <mpi.h>
#endif // RICH_MPI
#include "newtonian/three_dimensional/SourceTerm3D.hpp"
#include "newtonian/three_dimensional/time_step_function3D.hpp"
#include "../TimeStep.hpp"

class GlobalTimeStep: public TimeStep
{
public:    
    GlobalTimeStep(HydroTimeAdvance &hydroAdvance, const PointMotion3D &pm, dt_t currentTime, TimeStepFunction3D &tsf);
    
    dt_t apply() override;
    
private:
    TimeStepFunction3D &tsf;
    mutable std::vector<size_t> allPointsHelperVector;
};

#endif // GLOBAL_TIMESTEP_HPP