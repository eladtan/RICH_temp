#include "HilbertConvertor2D.hpp"

HilbertConvertor2D::HilbertConvertor2D(const Vector2D &ll, const Vector2D &ur, size_t order)
{
    this->ll = ll;
    coord_t realWidth = ur.x - ll.x;
    coord_t realHeight = ur.y - ll.y;

    // calculate divisions number in x axis and y axis
    this->div.x = std::ceil(std::sqrt(realWidth / realHeight) * std::pow(2, order));
    this->div.y = std::ceil(std::pow(4, order) / this->div.x);
    
    this->total_points_num = this->div.x * this->div.y;
    this->step = Vector2D(realWidth / this->div.x, realHeight / this->div.y);
}

Vector2D HilbertConvertor2D::WidthHeightToXY(int width, int height) const
{
    coord_t x, y;
    x = this->ll[0] + width * this->step[0];
    y = this->ll[1] + height * this->step[1];
    return Vector2D(x, y);
}

/**
 * see here the algorithm: https://github.com/jakubcerveny/gilbert
*/
bool HilbertConvertor2D::d2xy_helper(const DirectionVector2D &startPoint, const DirectionVector2D &a, const DirectionVector2D &b, hilbert_index_t requested_d, hilbert_index_t &current_d, Vector2D &result) const
{
    int width = std::abs(a.x + a.y);
    int height = std::abs(b.x + b.y);

    int num_points = width * height;

    if(requested_d >= current_d + num_points)
    {
        // the rectangle we are iterating over currently is irrelevent
        current_d += num_points;
        return false;
    }

    int dax = SIGN(a.x), day = SIGN(a.y);
    int dbx = SIGN(b.x), dby = SIGN(b.y);

    size_t diff = requested_d - current_d;

    // base cases
    if(height == 1)
    {
        result = this->WidthHeightToXY(startPoint.x + diff * dax, startPoint.y + diff * day);
        return true;
    }

    if(width == 1)
    {
        result = this->WidthHeightToXY(startPoint.x + diff * dbx, startPoint.y + diff * dby);
        return true;
    }

    DirectionVector2D a2 = {a.x >> 1, a.y >> 1}; /* (a.x//2, a.y//2) */
    DirectionVector2D b2 = {b.x >> 1, b.y >> 1}; /* (b.x//2, b.y//2) */

    int width2 = std::abs(a2.x + a2.y);
    int height2 = std::abs(b2.x + b2.y);

    if(2 * width > 3 * height)
    {
        if((width2 % 2) and (width > 2))
        {
            // prefer even steps
            a2.x = a2.x + dax;
            a2.y = a2.y + day;
        }
        
        if(this->d2xy_helper(startPoint, a2, b, requested_d, current_d, result))
        {
            return true;
        }
        if(this->d2xy_helper({startPoint.x + a2.x, startPoint.y + a2.y}, {a.x - a2.x, a.y - a2.y}, b, requested_d, current_d, result))
        {
            return true;
        }
    }
    else
    {
        if((height2 % 2) and (height > 2))
        {
            // prefer even steps
            b2.x = b2.x + dbx;
            b2.y = b2.y + dby;
        }
        // up
        if(this->d2xy_helper(startPoint, b2, a2, requested_d, current_d, result))
        {
            return true;
        }
        // long horizontal
        if(this->d2xy_helper({startPoint.x + b2.x, startPoint.y + b2.y}, a, {b.x - b2.x, b.y - b2.y}, requested_d, current_d, result))
        {
            return true;
        }
        // down
        if(this->d2xy_helper({startPoint.x + (a.x - dax) + (b2.x - dbx), startPoint.y + (a.y - day) + (b2.y - dby)}, 
                        {-b2.x, -b2.y}, {-(a.x - a2.x), -(a.y - a2.y)}, requested_d, current_d, result))
        {
            return true;
        }
    }

    throw UniversalError("Should not reach here (in 2D d->xy)");
    return false;
}

bool HilbertConvertor2D::xy2d_helper_base(const DirectionVector2D &startPoint, int steps, const DirectionVector2D &direction, const DirectionVector2D &requested_point, hilbert_index_t &current_d) const
{
    int x = startPoint.x, y = startPoint.y;
    for(int i = 0; i < steps; i++)
    {
        if((requested_point.x == x) and (requested_point.y == y))
        {
            return true;
        }
        x += direction.x;
        y += direction.y;
        current_d++;
    }
    return false;
}

/**
 * see here the algorithm: https://github.com/jakubcerveny/gilbert
*/
bool HilbertConvertor2D::xy2d_helper(const DirectionVector2D &startPoint, const DirectionVector2D &a, const DirectionVector2D &b, const DirectionVector2D &requested_point, hilbert_index_t &current_d) const
{
    int width = std::abs(a.x + a.y);
    int height = std::abs(b.x + b.y);

    int num_points = width * height;

    int dax = SIGN(a.x), day = SIGN(a.y);
    int dbx = SIGN(b.x), dby = SIGN(b.y);

    DirectionVector2D boundary = {startPoint.x + a.x + b.x, startPoint.y + a.y + b.y};
    std::pair<DirectionVector2D, DirectionVector2D> bounding_box = {{std::min(startPoint.x, boundary.x), std::min(startPoint.y, boundary.y)},
                                                                    {std::max(startPoint.x, boundary.x), std::max(startPoint.y, boundary.y)}};    
    if((requested_point.x < bounding_box.first.x) or (requested_point.x > bounding_box.second.x) or
        (requested_point.y < bounding_box.first.y) or (requested_point.y > bounding_box.second.y))
    {
        // doesn't have a chance to be here
        current_d += num_points;
        return false;
    }    

    // base cases
    if(height == 1)
    {
        return this->xy2d_helper_base(startPoint, width, {dax, day}, requested_point, current_d);
    }

    if(width == 1)
    {
        return this->xy2d_helper_base(startPoint, height, {dax, day}, requested_point, current_d);
    }

    DirectionVector2D a2 = {a.x >> 1, a.y >> 1}; /* (a.x//2, a.y//2) */
    DirectionVector2D b2 = {b.x >> 1, b.y >> 1}; /* (b.x//2, b.y//2) */

    int width2 = std::abs(a2.x + a2.y);
    int height2 = std::abs(b2.x + b2.y);

    if(2 * width > 3 * height)
    {
        if((width2 % 2) and (width > 2))
        {
            // prefer even steps
            a2.x = a2.x + dax;
            a2.y = a2.y + day;
        }
        
        if(this->xy2d_helper(startPoint, a2, b, requested_point, current_d))
        {
            return true;
        }
        if(this->xy2d_helper({startPoint.x + a2.x, startPoint.y + a2.y}, {a.x - a2.x, a.y - a2.y}, b, requested_point, current_d))
        {
            return true;
        }
    }
    else
    {
        if((height2 % 2) and (height > 2))
        {
            // prefer even steps
            b2.x = b2.x + dbx;
            b2.y = b2.y + dby;
        }
        // up
        if(this->xy2d_helper(startPoint, b2, a2, requested_point, current_d))
        {
            return true;
        }
        // long horizontal
        if(this->xy2d_helper({startPoint.x + b2.x, startPoint.y + b2.y}, a, {b.x - b2.x, b.y - b2.y}, requested_point, current_d))
        {
            return true;
        }
        // down
        if(this->xy2d_helper({startPoint.x + (a.x - dax) + (b2.x - dbx), startPoint.y + (a.y - day) + (b2.y - dby)}, 
                        {-b2.x, -b2.y}, {-(a.x - a2.x), -(a.y - a2.y)}, requested_point, current_d))
        {
            return true;
        }
    }
    return false;
}

Vector2D HilbertConvertor2D::d2xy(hilbert_index_t d) const
{
    Vector2D result;
    hilbert_index_t current_d = 0;
    this->d2xy_helper({0, 0}, {this->div.x, 0}, {0, this->div.y}, d, current_d, result);
    return result;
}

hilbert_index_t HilbertConvertor2D::xy2d(coord_t x, coord_t y) const
{
    // convert (x,y) to the integer pair (width, height)
    int width = std::floor((x - this->ll.x) / this->step.x);
    int height = std::floor((y - this->ll.y) / this->step.y);

    hilbert_index_t result = 0;
    if(not this->xy2d_helper({0, 0}, {this->div.x, 0}, {0, this->div.y}, {width, height}, result))
    {
        throw UniversalError("Should not reach here (in 2D xy->d)");
    }
    return result;
}