#ifndef SCALE_KERNEL_HPP
#define SCALE_KERNEL_HPP

#include "HilbertIndexing3D.hpp"

class Scale : public HilbertIndexing3D
{
public:
    inline Scale(const Vector3D &ll, const Vector3D &ur, size_t order): ll(ll), ur(ur), dx(ur - ll), order(order){};

    inline hilbert_index_t xyz2d(const Vector3D &point) const override{return this->xyz2d(point, this->order);};

    inline hilbert_index_t xyz2d(const Vector3D &point, size_t order) const
    {
        Vector3D scaledVector = Vector3D((point.x - this->ll.x) / this->dx.x,
                                         (point.y - this->ll.y) / this->dx.y, 
                                         (point.z - this->ll.z) / this->dx.z);
        return this->curve.Hilbert3D_xyz2d(scaledVector, order);
    }

private:
    HilbertCurve3D curve;
    Vector3D ll, ur, dx;
    size_t order;
};

#endif // SCALE_KERNEL_HPP