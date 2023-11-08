#include "HilbertConvertor.hpp"

namespace Hilbert3DConvertor
{
    static HilbertCurve3D curve;

    hilbert_index_t xyz2d(const Vector3D &vector, size_t iterations)
    {
        if(vector[0] < 0 || vector[1] < 0 || vector[2] < 0 || vector[0] > 1 || vector[1] > 1 || vector[2] > 1)
        {
            std::stringstream stream;
            stream << vector;
            throw UniversalError("Hilbert3DConvertor::xyz2d: vector (" + stream.str() + ") must have coordinates in [0, 1] range");
        }
        return curve.Hilbert3D_xyz2d(vector, iterations);
    };

    hilbert_index_t xyz2d(double x, double y, double z, size_t iterations)
    {
        return Hilbert3DConvertor::xyz2d(Vector3D(x, y, z), iterations);
    };
}