#ifndef HILBERT_CONVERTOR_3D_HPP
#define HILBERT_CONVERTOR_3D_HPP

#ifdef DEBUG_MODE
    #include <iostream>
#endif // DEBUG_MODE
#include "3D/elementary/Vector3D.hpp" // for Vector3D
#include "ds/utils/geometry.hpp" // for BoundingBox<Vector3D>
#include "../hilbertTypes.h"

#define MAX_HILBERT_ORDER 19

/**
 * see here the algorithm: https://github.com/jakubcerveny/gilbert
*/
class HilbertConvertor3D
{
    template<int max_leaf_ranks>
    friend class HilbertTree3D;

private:
    using coord_t = Vector3D::coord_type; // coordinate type
    using direction_t = long int;

    struct DirectionVector3D
    {
        direction_t x, y, z;

        #ifdef DEBUG_MODE
        friend std::ostream &operator<<(std::ostream &os, const DirectionVector3D &args)
        {
            return os << "(" << args.x << ", " << args.y << ", " << args.z << ")";
        }
        #endif // DEBUG_MODE
    };

    struct RecursionArguments
    {
        DirectionVector3D startPoint;
        DirectionVector3D a;
        DirectionVector3D b;
        DirectionVector3D c;

        #ifdef DEBUG_MODE
        friend std::ostream &operator<<(std::ostream &os, const RecursionArguments &args)
        {
            return os << "startPoint = " << args.startPoint << ", a = " << args.a << ", b = " << args.b << ", c = " << args.c;
        }
        #endif // DEBUG_MODE
    };

    Vector3D ll, ur, step;
    BoundingBox<Vector3D> spaceBoundingBox;
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
    std::pair<typename HilbertConvertor3D::DirectionVector3D, typename HilbertConvertor3D::DirectionVector3D> getBoundingBox(const RecursionArguments &args) const;
    Vector3D WidthHeightDepthToXYZ(direction_t width, direction_t height, direction_t depth) const;
};

#endif // HILBERT_CONVERTOR_3D_HPP