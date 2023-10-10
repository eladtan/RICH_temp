#ifndef SHRINK_KERNEL_HPP
#define SHRINK_KERNEL_HPP

#include "HilbertIndexing3D.hpp"

class Shrink : public HilbertIndexing3D
{
public:
    inline Shrink(const Vector3D &ll, const Vector3D &ur, size_t order): ll(ll), ur(ur), order(order)
    {
        Vector3D diff = ur - ll;
        this->dx = std::max(diff[0], std::max(diff[1], diff[2]));
    };

    inline hilbert_index_t xyz2d(const Vector3D &point) const override{return this->xyz2d(point, this->order);};

    inline hilbert_index_t xyz2d(const Vector3D &point, size_t order) const
    {
        Vector3D scaledVector = (point - ll) / dx;
        return this->curve.Hilbert3D_xyz2d(scaledVector, order);
    }

private:
    HilbertCurve3D curve;
    Vector3D ll, ur;
    double dx;
    size_t order;
};

#endif // SHRINK_KERNEL_HPP