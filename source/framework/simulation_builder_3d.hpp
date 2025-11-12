#ifndef SIMULATION_BUILDER_3D_HPP
#define SIMULATION_BUILDER_3D_HPP

#include "problem_config_3d.hpp"
#include "newtonian/three_dimensional/hdsim_3d.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "3D/tesselation/Tessellation3D.hpp"

class SpatialReconstruction3D;
class RiemannSolver3D;
class FluxCalculator3D;
class CellUpdater3D;
class ExtensiveUpdater3D;
class RadiationDriver;
class HDSim3D;

namespace rich3d {


class Simulation3DBuilder {
public:
    static void build_and_run(const Problem3DConfig& config);
};

} // namespace rich3d

#endif
