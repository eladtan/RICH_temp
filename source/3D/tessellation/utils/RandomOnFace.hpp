#ifndef RANDOM_ON_FACE_HPP
#define RANDOM_ON_FACE_HPP

#include <random>
#include "3D/elementary/Vector3D.hpp"
#include "3D/elementary/Face.hpp"
#include "3D/tessellation/Tessellation3D.hpp"

Vector3D RandomPointOnFace(const Tessellation3D &voronoi, size_t faceIndex);

#endif // RANDOM_ON_FACE_HPP