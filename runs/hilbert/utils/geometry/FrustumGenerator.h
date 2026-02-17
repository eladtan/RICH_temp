#ifndef FRUSTUM_GENERATOR_H
#define FRUSTUM_GENERATOR_H

#include <vector>
#include "3D/elementary/Face.hpp"

std::vector<Face> GenerateFrustum(const Face &base1, const Face &base2);

#endif // FRUSTUM_GENERATOR_H