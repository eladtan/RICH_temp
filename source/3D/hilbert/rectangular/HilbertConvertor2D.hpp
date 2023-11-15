#ifndef HILBERT_CONVERTOR_2D_HPP
#define HILBERT_CONVERTOR_2D_HPP

#include <iostream> // todo: remove
#include "tessellation/geometry.hpp" // for Vector2D
#include "../hilbertTypes.h"

#define MAX_HILBERT_ORDER 28

class HilbertConvertor2D
{
private:
    using coord_t = double; // coordinate type

    struct DirectionVector2D
    {
        int x, y;
    };

    Vector2D ll, ur, step;
    DirectionVector2D div;
    hilbert_index_t total_points_num;
    size_t order;

public:
    explicit HilbertConvertor2D(const Vector2D &ll, const Vector2D &ur, size_t order);
    
    inline hilbert_index_t getHilbertSize() const{return this->total_points_num;};
    void changeOrder(size_t order);
    hilbert_index_t xy2d(coord_t x, coord_t y) const;
    inline hilbert_index_t xy2d(const Vector2D &point) const{return this->xy2d(point.x, point.y);};
    Vector2D d2xy(hilbert_index_t d) const;
    inline size_t getOrder() const{return this->order;};

private:
    bool d2xy_helper(const DirectionVector2D &startPoint, const DirectionVector2D &a, const DirectionVector2D &b, hilbert_index_t requested_d, hilbert_index_t &current_d, Vector2D &result) const;
    bool xy2d_helper_base(const DirectionVector2D &startPoint, size_t steps, const DirectionVector2D &direction, const DirectionVector2D &requested_point, hilbert_index_t &current_d) const;
    bool xy2d_helper(const DirectionVector2D &startPoint, const DirectionVector2D &a, const DirectionVector2D &b, const DirectionVector2D &requested_point, hilbert_index_t &current_d) const;
    Vector2D WidthHeightToXY(int width, int height) const;
};

#endif // HILBERT_CONVERTOR_2D_HPP