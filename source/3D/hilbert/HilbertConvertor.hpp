#ifndef HILBERT_CONVERTOR_HPP
#define HILBERT_CONVERTOR_HPP

#include "HilbertOrder3D.hpp"

#define MAX_HILBERT_ORDER 19

namespace Hilbert3DConvertor
{
    hilbert_index_t xyz2d(const Vector3D &vector, size_t iterations);

    hilbert_index_t xyz2d(double x, double y, double z, size_t iterations);
}

#endif // HILBERT_CONVERTOR_HPP