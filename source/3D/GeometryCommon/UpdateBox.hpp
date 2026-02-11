#ifndef UPDATE_BOX_HPP
#define UPDATE_BOX_HPP 1

#include "../../newtonian/three_dimensional/hdsim_3d.hpp"
#include "3D/tessellation/voronoi/Voronoi3D.hpp"

void UpdateBox(Voronoi3D &tess, HDSim3D &sim, double const min_velocity, double const volume_fraction, ComputationalCell3D const& reference_cell);

#endif // UPDATE_BOX_HPP