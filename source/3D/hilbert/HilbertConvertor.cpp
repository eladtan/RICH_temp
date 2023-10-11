#include "HilbertConvertor.hpp"

namespace Hilbert3DConvertor
{
    static HilbertCurve3D curve;

    hilbert_index_t xyz2d(const Vector3D &vector, size_t iterations)
    {
        return curve.Hilbert3D_xyz2d(vector, iterations);
    };

    hilbert_index_t xyz2d(double x, double y, double z, size_t iterations)
    {
        return Hilbert3DConvertor::xyz2d(Vector3D(x, y, z), iterations);
    };
}