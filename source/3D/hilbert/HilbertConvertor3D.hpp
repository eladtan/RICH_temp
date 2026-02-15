#ifndef HILBERT_CONVERTOR_3D_HPP
#define HILBERT_CONVERTOR_3D_HPP

#ifdef DEBUG_MODE
    #include <iostream>
#endif // DEBUG_MODE
#include "3D/elementary/Vector3D.hpp" // for Vector3D
#include "ds/utils/geometry.hpp" // for BoundingBox<Vector3D>
#include "hilbertTypes.h"
#include <memory>

#define MAX_HILBERT_ORDER 19

class HilbertConvertor3D
{
protected:
    using coord_t = Vector3D::coord_type; // coordinate type

    Vector3D ll, ur;
    size_t order;

public:
    explicit HilbertConvertor3D(const Vector3D &ll, const Vector3D &ur, size_t order);
    
    virtual ~HilbertConvertor3D() = default;
    
    virtual std::shared_ptr<HilbertConvertor3D> clone(void) const = 0;

    virtual void changeOrder(size_t order) = 0;
    
    virtual hilbert_index_t xyz2d(coord_t x, coord_t y, coord_t z) const = 0;
    
    virtual inline hilbert_index_t xyz2d(const Vector3D &point) const{return this->xyz2d(point.x, point.y, point.z);};
    
    virtual Vector3D d2xyz(hilbert_index_t d) const = 0;
    
    inline size_t getOrder() const{return this->order;};
};

inline HilbertConvertor3D::HilbertConvertor3D(const Vector3D &ll, const Vector3D &ur, size_t order)
    : ll(ll), ur(ur), order(order)
{}

#endif // HILBERT_CONVERTOR_3D_HPP