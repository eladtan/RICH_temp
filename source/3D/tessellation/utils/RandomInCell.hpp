#ifndef RANDOM_IN_CELL_HPP
#define RANDOM_IN_CELL_HPP

#include <random>
#include <array>
#include "3D/elementary/Vector3D.hpp"
#include "3D/tessellation/Tessellation3D.hpp"

Vector3D RandomPointInCell(const Tessellation3D &voronoi, size_t cellIndex);

void ReseedRandomInCell(uint64_t seed);

#endif // RANDOM_IN_CELL_HPP