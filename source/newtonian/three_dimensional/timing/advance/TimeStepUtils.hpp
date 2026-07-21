#ifndef TIMESTEP_UTILS_HPP
#define TIMESTEP_UTILS_HPP

#include "3D/tessellation/Tessellation3D.hpp"

void CalcFaceVelocities(const Tessellation3D &tess, const vector<Vector3D> &point_vel, vector<Vector3D> &res);

#endif // TIMESTEP_UTILS_HPP