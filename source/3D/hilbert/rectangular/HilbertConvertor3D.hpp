#ifndef HILBERT_CONVERTOR_3D_HPP
#define HILBERT_CONVERTOR_3D_HPP

#include <iostream> // todo: remove
#include "3D/elementary/Vector3D.hpp" // for Vector3D
#include "../hilbertTypes.h"

class HilbertConvertor3D
{
private:
    using coord_t = Vector3D::coord_type; // coordinate type

    struct DirectionVector3D
    {
        int x, y, z;
    };

    Vector3D ll, step;
    DirectionVector3D div;
    hilbert_index_t total_points_num;

public:
    explicit HilbertConvertor3D(const Vector3D &ll, const Vector3D &ur, size_t order);
    
    inline hilbert_index_t getHilbertSize() const{return this->total_points_num;};
    hilbert_index_t xyz2d(coord_t x, coord_t y, coord_t z) const;
    inline hilbert_index_t xyz2d(const Vector3D &point){return this->xyz2d(point.x, point.y, point.z);};
    Vector3D d2xyz(hilbert_index_t d) const;

private:
    bool d2xyz_helper(const DirectionVector3D &startPoint, const DirectionVector3D &a, const DirectionVector3D &b, const DirectionVector3D &c,  hilbert_index_t requested_d, hilbert_index_t &current_d, Vector3D &result) const;
    bool xyz2d_helper_base(const DirectionVector3D &startPoint, int steps, const DirectionVector3D &direction, const DirectionVector3D &requested_point, hilbert_index_t &current_d) const;
    bool xyz2d_helper(const DirectionVector3D &startPoint, const DirectionVector3D &a, const DirectionVector3D &b, const DirectionVector3D &c,  const DirectionVector3D &requested_point, hilbert_index_t &current_d) const;
    Vector3D WidthHeightDepthToXYZ(int width, int height, int depth) const;
};

#endif // HILBERT_CONVERTOR_3D_HPP