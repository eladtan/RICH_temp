#ifndef UPDATE_BOX_HPP
#define UPDATE_BOX_HPP 1

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include "../../newtonian/three_dimensional/hdsim_3d.hpp"
#include "3D/tesselation/Tessellation3D.hpp"

void UpdateBox(Tessellation3D &tess, HDSim3D &sim, double const min_velocity, double const volume_fraction, ComputationalCell3D const& reference_cell);

#endif // UPDATE_BOX_HPP