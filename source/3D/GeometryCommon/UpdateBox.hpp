#ifndef UPDATE_BOX_HPP
#define UPDATE_BOX_HPP 1

#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "3D/tessellation/Voronoi3D.hpp"

void UpdateBox(Voronoi3D &tess, Simulation &sim, double const min_velocity, double const volume_fraction, ComputationalCell3D const& reference_cell);

#endif // UPDATE_BOX_HPP