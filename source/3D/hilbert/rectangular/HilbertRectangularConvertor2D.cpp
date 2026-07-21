#include "HilbertRectangularConvertor2D.hpp"

HilbertRectangularConvertor2D::HilbertRectangularConvertor2D(const Vector2D &ll, const Vector2D &ur, size_t order)
{
    this->ll = ll;
    this->ur = ur;
    this->changeOrder(order);
}

void HilbertRectangularConvertor2D::changeOrder(size_t order)
{
    this->order = order = std::min<size_t>(MAX_HILBERT_ORDER, order);
    coord_t realWidth = this->ur.x - this->ll.x;
    coord_t realHeight = this->ur.y - this->ll.y;

    // calculate divisions number in x axis and y axis
    this->div.x = std::ceil(std::sqrt(realWidth / realHeight) * std::pow(2, order));
    this->div.y = std::ceil(std::pow(4, order) / this->div.x);
    
    this->total_points_num = this->div.x * this->div.y;
    this->step = Vector2D(realWidth / this->div.x, realHeight / this->div.y);
}

Vector2D HilbertRectangularConvertor2D::WidthHeightToXY(direction_t width, direction_t height) const
{
    coord_t x, y;
    x = this->ll[0] + width * this->step[0];
    y = this->ll[1] + height * this->step[1];
    return Vector2D(x, y);
}

std::vector<HilbertRectangularConvertor2D::RecursionArguments> HilbertRectangularConvertor2D::getRecursionArguments(const RecursionArguments &args) const
{
    const DirectionVector2D &startPoint = args.startPoint;
    const DirectionVector2D &a = args.a;
    const DirectionVector2D &b = args.b;

    direction_t width = std::abs(a.x + a.y);
    direction_t height = std::abs(b.x + b.y);

    direction_t dax = SIGN(a.x), day = SIGN(a.y);
    direction_t dbx = SIGN(b.x), dby = SIGN(b.y);

    DirectionVector2D a2 = {a.x >> 1, a.y >> 1}; /* (a.x//2, a.y//2) */
    DirectionVector2D b2 = {b.x >> 1, b.y >> 1}; /* (b.x//2, b.y//2) */

    direction_t width2 = std::abs(a2.x + a2.y);
    direction_t height2 = std::abs(b2.x + b2.y);

    std::vector<RecursionArguments> toReturn;

    if(2 * width > 3 * height)
    {
        if((width2 % 2) and (width > 2))
        {
            // prefer even steps
            a2.x = a2.x + dax;
            a2.y = a2.y + day;
        }
        toReturn.push_back({startPoint, a2, b});
        toReturn.push_back({{startPoint.x + a2.x, startPoint.y + a2.y}, {a.x - a2.x, a.y - a2.y}, b});
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
        toReturn.push_back({startPoint, b2, a2});
        // long horizontal
        toReturn.push_back({{startPoint.x + b2.x, startPoint.y + b2.y}, a, {b.x - b2.x, b.y - b2.y}});
        // down
    toReturn.push_back({{startPoint.x + (a.x - dax) + (b2.x - dbx), startPoint.y + (a.y - day) + (b2.y - dby)}, 
                        {-b2.x, -b2.y}, {-(a.x - a2.x), -(a.y - a2.y)}});
    }

    return toReturn;
}

bool HilbertRectangularConvertor2D::d2xy_helper(const RecursionArguments &args, hilbert_index_t requested_d, hilbert_index_t &current_d, Vector2D &result) const
{
    const DirectionVector2D &startPoint = args.startPoint;
    const DirectionVector2D &a = args.a;
    const DirectionVector2D &b = args.b;

    direction_t width = std::abs(a.x + a.y);
    direction_t height = std::abs(b.x + b.y);

    size_t num_points = width * height;

    if(requested_d >= current_d + num_points)
    {
        // the rectangle we are iterating over currently is irrelevent
        current_d += num_points;
        return false;
    }

    direction_t dax = SIGN(a.x), day = SIGN(a.y);
    direction_t dbx = SIGN(b.x), dby = SIGN(b.y);

    if(requested_d < current_d)
    {
        throw UniversalError("in HilbertRectangularConvertor2D::d2xy_helper, should not reach here (algorithm failed)");
    }
    hilbert_index_t diff = requested_d - current_d;

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

    for(const RecursionArguments &nextArgs : this->getRecursionArguments(args))
    {
        if(this->d2xy_helper(nextArgs, requested_d, current_d, result))
        {
            return true;
        }
    }

    throw UniversalError("Should not reach here (in 2D d->xy)");
    return false;
}

bool HilbertRectangularConvertor2D::xy2d_helper_base(const DirectionVector2D &startPoint, size_t steps, const DirectionVector2D &direction, const DirectionVector2D &requested_point, hilbert_index_t &current_d) const
{
    direction_t x = startPoint.x, y = startPoint.y;
    for(size_t i = 0; i < steps; i++)
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

std::pair<typename HilbertRectangularConvertor2D::DirectionVector2D, typename HilbertRectangularConvertor2D::DirectionVector2D> HilbertRectangularConvertor2D::getBoundingBox(const RecursionArguments &args) const
{
    const DirectionVector2D &startPoint = args.startPoint;
    const DirectionVector2D &a = args.a;
    const DirectionVector2D &b = args.b;

    direction_t x_advancing = a.x + b.x;
    direction_t y_advancing = a.y + b.y;

    DirectionVector2D boundary = {startPoint.x + x_advancing + ((x_advancing >= 0)? 1 : 0), startPoint.y + y_advancing + ((y_advancing >= 0)? 1 : 0)};
    return {{std::min(startPoint.x, boundary.x), std::min(startPoint.y, boundary.y)},
            {std::max(startPoint.x, boundary.x), std::max(startPoint.y, boundary.y)}};    
}

bool HilbertRectangularConvertor2D::xy2d_helper(const RecursionArguments &args, const DirectionVector2D &requested_point, hilbert_index_t &current_d) const
{
    const DirectionVector2D &startPoint = args.startPoint;
    const DirectionVector2D &a = args.a;
    const DirectionVector2D &b = args.b;

    size_t width = std::abs(a.x + a.y);
    size_t height = std::abs(b.x + b.y);

    size_t num_points = width * height;

    direction_t dax = SIGN(a.x), day = SIGN(a.y);
    direction_t dbx = SIGN(b.x), dby = SIGN(b.y);

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
        return this->xy2d_helper_base(startPoint, height, {dbx, dby}, requested_point, current_d);
    }

    for(const RecursionArguments &nextArgs : this->getRecursionArguments(args))
    {
        if(this->xy2d_helper(nextArgs, requested_point, current_d))
        {
            return true;
        }
    }

    return false;
}

Vector2D HilbertRectangularConvertor2D::d2xy(hilbert_index_t d) const
{
    Vector2D result;
    hilbert_index_t current_d = 0;
    this->d2xy_helper({{0, 0}, {this->div.x, 0}, {0, this->div.y}}, d, current_d, result);
    return result;
}

hilbert_index_t HilbertRectangularConvertor2D::xy2d(coord_t x, coord_t y) const
{
    // convert (x,y) to the integer pair (width, height)
    direction_t width = std::floor((x - this->ll.x) / this->step.x);
    direction_t height = std::floor((y - this->ll.y) / this->step.y);

    if(width < 0 or height < 0 or width > this->div.x or height > this->div.y)
    {
        throw UniversalError("Should not reach here, overflow (in 2D xy->d)");
    }
    hilbert_index_t result = 0;
    if(not this->xy2d_helper({{0, 0}, {this->div.x, 0}, {0, this->div.y}}, {width, height}, result))
    {
        throw UniversalError("Should not reach here (in 2D xy->d)");
    }
    return result;
}