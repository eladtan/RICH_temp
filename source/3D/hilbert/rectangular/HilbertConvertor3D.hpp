#ifndef HILBERT_CONVERTOR_3D_HPP
#define HILBERT_CONVERTOR_3D_HPP

#include <iostream> // todo: remove
#include "3D/elementary/Vector3D.hpp" // for Vector3D
#include "../hilbertTypes.h"

#define MAX_HILBERT_ORDER 19

class HilbertConvertor3D
{
private:
    using coord_t = Vector3D::coord_type; // coordinate type
    using direction_t = long int;

    struct DirectionVector3D
    {
        direction_t x, y, z;
    };

    struct RecursionArguments
    {
        DirectionVector3D startPoint;
        DirectionVector3D a;
        DirectionVector3D b;
        DirectionVector3D c;
    };

    Vector3D ll, ur, step;
    DirectionVector3D div;
    hilbert_index_t total_points_num;
    size_t order;

public:
    explicit HilbertConvertor3D(const Vector3D &ll, const Vector3D &ur, size_t order);
    inline hilbert_index_t getHilbertSize() const{return this->total_points_num;};
    void changeOrder(size_t order);
    hilbert_index_t xyz2d(coord_t x, coord_t y, coord_t z) const;
    inline hilbert_index_t xyz2d(const Vector3D &point) const{return this->xyz2d(point.x, point.y, point.z);};
    Vector3D d2xyz(hilbert_index_t d) const;
    inline size_t getOrder() const{return this->order;};
    
private:
    std::vector<RecursionArguments> getRecursionArguments(const RecursionArguments &args) const;
    bool d2xyz_helper(const RecursionArguments &args, hilbert_index_t requested_d, hilbert_index_t &current_d, Vector3D &result) const;
    bool xyz2d_helper_base(const DirectionVector3D &startPoint, size_t steps, const DirectionVector3D &direction, const DirectionVector3D &requested_point, hilbert_index_t &current_d) const;
    bool xyz2d_helper(const RecursionArguments &args, const DirectionVector3D &requested_point, hilbert_index_t &current_d) const;
    Vector3D WidthHeightDepthToXYZ(direction_t width, direction_t height, direction_t depth) const;
};

#endif // HILBERT_CONVERTOR_3D_HPP