#ifndef TIME_STEP_HPP
#define TIME_STEP_HPP

#include <vector>
#include "../timesteps.h"
#include "hydro/HydroTimeAdvance.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/common/equation_of_state.hpp"
#include "hydro/HydroTimeAdvance.hpp"
#include "TimeStepUtils.hpp"

/**
 * This class performs a (big) timestep, including a hydrodynamical timestep
*/
class TimeStep
{
public:
    TimeStep(HydroTimeAdvance &hydroAdvance, const PointMotion3D &pm, dt_t currentTime):
        hydroAdvance(hydroAdvance), tess(hydroAdvance.getTessellation()), cells(hydroAdvance.getCells()), pm(pm), currentTime(currentTime)
    {}

    virtual ~TimeStep() = default;
    
    /**
     * applies a timestep. Returns the 'dt' passed
    */
    virtual dt_t apply() = 0;

protected: 
    /**
     * Can be used to calculate initial face and point velocities (the velocities without knowing the 'dt')
    */
    std::pair<std::vector<Vector3D>, std::vector<Vector3D>> CalculateInitialPointFaceVelocities() const;

    /**
     * After knowing an initial 'dt', this method can help to fix the point and face velocities
    */
    void FixPointFaceVelocities(std::vector<Vector3D> &point_vel, std::vector<Vector3D> &face_vel, dt_t dt) const;

    HydroTimeAdvance &hydroAdvance;
    Tessellation3D &tess;
    std::vector<ComputationalCell3D> &cells;
    const PointMotion3D &pm;
    dt_t currentTime;
};

#endif // TIME_STEP_HPP