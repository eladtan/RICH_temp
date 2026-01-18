#ifndef HILBERT_RECTANGULAR_CONVERTOR_2D_HPP
#define HILBERT_RECTANGULAR_CONVERTOR_2D_HPP

#include "tessellation/geometry.hpp" // for Vector2D
#include "misc/universal_error.hpp"
#include "../hilbertTypes.h"

#define MAX_HILBERT_ORDER 28

/**
 * see here the algorithm: https://github.com/jakubcerveny/gilbert
*/
class HilbertRectangularConvertor2D
{
private:
    using coord_t = double; // coordinate type
    using direction_t = long int;

    struct DirectionVector2D
    {
        direction_t x, y;
    };

    struct RecursionArguments
    {
        DirectionVector2D startPoint;
        DirectionVector2D a;
        DirectionVector2D b;
    };

    Vector2D ll, ur, step;
    DirectionVector2D div;
    hilbert_index_t total_points_num;
    size_t order;

public:
    explicit HilbertRectangularConvertor2D(const Vector2D &ll, const Vector2D &ur, size_t order);
    
    ~HilbertRectangularConvertor2D() = default;
    
    inline hilbert_index_t getHilbertSize() const{return this->total_points_num;};
    
    void changeOrder(size_t order);
    
    hilbert_index_t xy2d(coord_t x, coord_t y) const;
    
    inline hilbert_index_t xy2d(const Vector2D &point) const{return this->xy2d(point.x, point.y);};
    
    Vector2D d2xy(hilbert_index_t d) const;
    
    inline size_t getOrder() const{return this->order;};

private:
    std::vector<RecursionArguments> getRecursionArguments(const RecursionArguments &args) const;
    
    bool d2xy_helper(const RecursionArguments &args, hilbert_index_t requested_d, hilbert_index_t &current_d, Vector2D &result) const;
    
    bool xy2d_helper_base(const DirectionVector2D &startPoint, size_t steps, const DirectionVector2D &direction, const DirectionVector2D &requested_point, hilbert_index_t &current_d) const;
    
    bool xy2d_helper(const RecursionArguments &args, const DirectionVector2D &requested_point, hilbert_index_t &current_d) const;
    
    Vector2D WidthHeightToXY(direction_t width, direction_t height) const;
    
    std::pair<DirectionVector2D, DirectionVector2D> getBoundingBox(const RecursionArguments &args) const;
};

#endif // HILBERT_RECTANGULAR_CONVERTOR_2D_HPP