#include "HilbertConvertor3D.hpp"

HilbertConvertor3D::HilbertConvertor3D(const Vector3D &ll, const Vector3D &ur, size_t order)
{
    this->ll = ll;
    this->ur = ur;
    this->changeOrder(order);
}

void HilbertConvertor3D::changeOrder(size_t order)
{
    this->order = order = std::min<size_t>(MAX_HILBERT_ORDER, order);
    coord_t realWidth = this->ur.x - this->ll.x;
    coord_t realHeight = this->ur.y - this->ll.y;
    coord_t realDepth = this->ur.z - this->ll.z;

    // calculate divisions number in x, y, z axises
    this->div.x = std::ceil(std::pow((realWidth * realWidth) / (realHeight * realDepth), 0.333333333) * std::pow(2, order));
    this->div.y = std::ceil(std::pow((realHeight * realHeight) / (realWidth * realDepth), 0.333333333) * std::pow(2, order));
    this->div.z = std::ceil(std::pow(8, order) / (this->div.x * this->div.y));
    
    this->total_points_num = this->div.x * this->div.y * this->div.z;
    this->step = Vector3D(realWidth / div.x, realHeight / div.y, realDepth / div.z);
}

Vector3D HilbertConvertor3D::WidthHeightDepthToXYZ(int width, int height, int depth) const
{
    coord_t x, y, z;
    x = this->ll[0] + width * this->step[0];
    y = this->ll[1] + height * this->step[1];
    z = this->ll[2] + depth * this->step[2];
    // std::cout << "translating (" << width << ", " << height << ") to (" << x << ", " << y << ")" << std::endl;
    return Vector3D(x, y, z);
}

/**
 * see here the algorithm: https://github.com/jakubcerveny/gilbert
*/
bool HilbertConvertor3D::d2xyz_helper(const DirectionVector3D &startPoint, const DirectionVector3D &a, const DirectionVector3D &b, const DirectionVector3D &c, hilbert_index_t requested_d, hilbert_index_t &current_d, Vector3D &result) const
{
    int width = std::abs(a.x + a.y + a.z);
    int height = std::abs(b.x + b.y + b.z);
    int depth = std::abs(c.x + c.y + c.z);

    int num_points = width * height * depth;

    int dax = SIGN(a.x), day = SIGN(a.y), daz = SIGN(a.z);
    int dbx = SIGN(b.x), dby = SIGN(b.y), dbz = SIGN(b.z);
    int dcx = SIGN(c.x), dcy = SIGN(c.y), dcz = SIGN(c.z);

    if(requested_d >= current_d + num_points)
    {
        // the rectangle we are iterating over currently is irrelevent
        current_d += num_points;
        return false;
    }

    size_t diff = requested_d - current_d;

    // base cases
    if(height == 1 and depth == 1)
    {
        result = this->WidthHeightDepthToXYZ(startPoint.x + diff * dax, startPoint.y + diff * day, startPoint.z + diff * daz);
        return true;
    }

    if(width == 1 and depth == 1)
    {
        result = this->WidthHeightDepthToXYZ(startPoint.x + diff * dbx, startPoint.y + diff * dby, startPoint.z + diff * dbz);
        return true;
    }

    if(width == 1 and height == 1)
    {
        result = this->WidthHeightDepthToXYZ(startPoint.x + diff * dcx, startPoint.y + diff * dcy, startPoint.z + diff * dcz);
        return true;
    }

    DirectionVector3D a2 = {a.x >> 1, a.y >> 1, a.z >> 1}; /* (a.x//2, a.y//2, a.z//2) */
    DirectionVector3D b2 = {b.x >> 1, b.y >> 1, b.z >> 1}; /* (b.x//2, b.y//2, b.z//2) */
    DirectionVector3D c2 = {c.x >> 1, c.y >> 1, c.z >> 1}; /* (c.x//2, c.y//2, c.z//2) */

    int width2 = std::abs(a2.x + a2.y + a2.z);
    int height2 = std::abs(b2.x + b2.y + b2.z);
    int depth2 = std::abs(c2.x + c2.y + c2.z);

    // prefer even steps
    if((width2 % 2) and (width > 2))
    {
        a2.x = a2.x + dax;
        a2.y = a2.y + day;
        a2.z = a2.z + daz;
    }

    if((height2 % 2) and (height > 2))
    {
        b2.x = b2.x + dbx;
        b2.y = b2.y + dby;
        b2.z = b2.z + dbz;
    }

    if((depth2 % 2) and (depth > 2))
    {
        c2.x = c2.x + dcx;
        c2.y = c2.y + dcy;
        c2.z = c2.z + dcz;
    }

    const int &x = startPoint.x;
    const int &y = startPoint.y;
    const int &z = startPoint.z;

    if((2 * width > 3 * height) and (2 * width > 3 * depth))
    {
        if(this->d2xyz_helper(startPoint, a2, b, c, requested_d, current_d, result)) return true;
        if(this->d2xyz_helper({x + a2.x, y + a2.y, z + a2.z}, {a.x - a2.x, a.y - a2.y, a.z - a2.z}, b, c, requested_d, current_d, result)) return true;
    }
    else if(3 * height > 4 * depth)
    {
        if(this->d2xyz_helper(startPoint, b2, c, a2, requested_d, current_d, result)) return true;
        if(this->d2xyz_helper({x + b2.x, y + b2.y, z + b2.z}, a, {b.x - b2.x, b.y - b2.y, b.z - b2.z}, c, requested_d, current_d, result)) return true;
        if(this->d2xyz_helper({x + (a.x - dax) + (b2.x - dbx), y + (a.y - day) + (b2.y - dby), z + (a.z - daz) + (b2.z - dbz)}, {-b2.x, -b2.y, -b2.z}, c, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}, requested_d, current_d, result)) return true;
    }
    else if(3 * depth > 4 * height)
    {
        if(this->d2xyz_helper(startPoint, c2, a2, b, requested_d, current_d, result)) return true;
        if(this->d2xyz_helper({x + c2.x, y + c2.y, z + c2.z}, a, b, {c.x - c2.x, c.y - c2.y, c.z - c2.z}, requested_d, current_d, result)) return true;
        if(this->d2xyz_helper({x + (a.x - dax) + (c2.x - dcx), y + (a.y - day) + (c2.y - dcy), z + (a.z - daz) + (c2.z - dcz)}, {-c2.x, -c2.y, -c2.z}, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}, b, requested_d, current_d, result)) return true;
    }
    else
    {
        if(this->d2xyz_helper(startPoint, b2, c2, a2, requested_d, current_d, result)) return true;
        if(this->d2xyz_helper({x + b2.x, y + b2.y, z + b2.z}, c, a2, {b.x - b2.x, b.y - b2.y, b.z - b2.z}, requested_d, current_d, result)) return true;
        if(this->d2xyz_helper({x + (b2.x - dbx) + (c.x - dcx), y + (b2.y - dby) + (c.y - dcy), z + (b2.z - dbz) + (c.z - dcz)}, a, {-b2.x, -b2.y, -b2.z}, {-(c.x - c2.x), -(c.y - c2.y), -(c.z - c2.z)}, requested_d, current_d, result)) return true;
        if(this->d2xyz_helper({x + (a.x - dax) + b2.x + (c.x - dcx), y + (a.y - day) + b2.y + (c.y - dcy), z + (a.z - daz) + b2.z + (c.z - dcz)}, {-c.x, -c.y, -c.z}, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}, {b.x - b2.x, b.y - b2.y, b.z - b2.z}, requested_d, current_d, result)) return true;
        if(this->d2xyz_helper({x + (a.x - dax) + (b2.x - dbx), y + (a.y - day) + (b2.y - dby), z + (a.z - daz) + (b2.z - dbz)}, {-b2.x, -b2.y, -b2.z}, c2, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}, requested_d, current_d, result)) return true;
    }

    return false;
}

bool HilbertConvertor3D::xyz2d_helper_base(const DirectionVector3D &startPoint, int steps, const DirectionVector3D &direction, const DirectionVector3D &requested_point, hilbert_index_t &current_d) const
{
    int x = startPoint.x, y = startPoint.y, z = startPoint.z;
    for(int i = 0; i < steps; i++)
    {
        if((requested_point.x == x) and (requested_point.y == y) and (requested_point.z == z))
        {
            return true;
        }
        x += direction.x;
        y += direction.y;
        z += direction.z;
        current_d++;
    }
    return false;
}

/**
 * see here the algorithm: https://github.com/jakubcerveny/gilbert
*/
bool HilbertConvertor3D::xyz2d_helper(const DirectionVector3D &startPoint, const DirectionVector3D &a, const DirectionVector3D &b, const DirectionVector3D &c, const DirectionVector3D &requested_point, hilbert_index_t &current_d) const
{
    int width = std::abs(a.x + a.y + a.z);
    int height = std::abs(b.x + b.y + b.z);
    int depth = std::abs(c.x + c.y + c.z);

    int num_points = width * height * depth;

    int dax = SIGN(a.x), day = SIGN(a.y), daz = SIGN(a.z);
    int dbx = SIGN(b.x), dby = SIGN(b.y), dbz = SIGN(b.z);
    int dcx = SIGN(c.x), dcy = SIGN(c.y), dcz = SIGN(c.z);

    DirectionVector3D boundary = {startPoint.x + a.x + b.x + c.x, startPoint.y + a.y + b.y + c.y, startPoint.z + a.z + b.z + c.z};
    std::pair<DirectionVector3D, DirectionVector3D> bounding_box = {{std::min(startPoint.x, boundary.x), std::min(startPoint.y, boundary.y), std::min(startPoint.z, boundary.z)},
                                                                    {std::max(startPoint.x, boundary.x), std::max(startPoint.y, boundary.y), std::max(startPoint.z, boundary.z)}};    
    if((requested_point.x < bounding_box.first.x) or (requested_point.x > bounding_box.second.x) or
        (requested_point.y < bounding_box.first.y) or (requested_point.y > bounding_box.second.y) or
        (requested_point.z < bounding_box.first.z) or (requested_point.z > bounding_box.second.z))
    {
        // doesn't have a chance to be here
        current_d += num_points;
        return false;
    }    

    // base cases
    if(height == 1 and depth == 1)
    {
        return this->xyz2d_helper_base(startPoint, width, {dax, day, daz}, requested_point, current_d);
    }

    if(width == 1 and depth == 1)
    {
        return this->xyz2d_helper_base(startPoint, height, {dbx, dby, dbz}, requested_point, current_d);
    }

    if(width == 1 and height == 1)
    {
        return this->xyz2d_helper_base(startPoint, depth, {dcx, dcy, dcz}, requested_point, current_d);
    }

    DirectionVector3D a2 = {a.x >> 1, a.y >> 1, a.z >> 1}; /* (a.x//2, a.y//2, a.z//2) */
    DirectionVector3D b2 = {b.x >> 1, b.y >> 1, b.z >> 1}; /* (b.x//2, b.y//2, b.z//2) */
    DirectionVector3D c2 = {c.x >> 1, c.y >> 1, c.z >> 1}; /* (c.x//2, c.y//2, c.z//2) */

    int width2 = std::abs(a2.x + a2.y + a2.z);
    int height2 = std::abs(b2.x + b2.y + b2.z);
    int depth2 = std::abs(c2.x + c2.y + c2.z);

    // prefer even steps
    if((width2 % 2) and (width > 2))
    {
        a2.x = a2.x + dax;
        a2.y = a2.y + day;
        a2.z = a2.z + daz;
    }

    if((height2 % 2) and (height > 2))
    {
        b2.x = b2.x + dbx;
        b2.y = b2.y + dby;
        b2.z = b2.z + dbz;
    }

    if((depth2 % 2) and (depth > 2))
    {
        c2.x = c2.x + dcx;
        c2.y = c2.y + dcy;
        c2.z = c2.z + dcz;
    }

    const int &x = startPoint.x;
    const int &y = startPoint.y;
    const int &z = startPoint.z;

    if((2 * width > 3 * height) and (2 * width > 3 * depth))
    {
        if(this->xyz2d_helper(startPoint, a2, b, c, requested_point, current_d)) return true;
        if(this->xyz2d_helper({x + a2.x, y + a2.y, z + a2.z}, {a.x - a2.x, a.y - a2.y, a.z - a2.z}, b, c, requested_point, current_d)) return true;
    }
    else if(3 * height > 4 * depth)
    {
        if(this->xyz2d_helper(startPoint, b2, c, a2, requested_point, current_d)) return true;
        if(this->xyz2d_helper({x + b2.x, y + b2.y, z + b2.z}, a, {b.x - b2.x, b.y - b2.y, b.z - b2.z}, c, requested_point, current_d)) return true;
        if(this->xyz2d_helper({x + (a.x - dax) + (b2.x - dbx), y + (a.y - day) + (b2.y - dby), z + (a.z - daz) + (b2.z - dbz)}, {-b2.x, -b2.y, -b2.z}, c, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}, requested_point, current_d)) return true;
    }
    else if(3 * depth > 4 * height)
    {
        if(this->xyz2d_helper(startPoint, c2, a2, b, requested_point, current_d)) return true;
        if(this->xyz2d_helper({x + c2.x, y + c2.y, z + c2.z}, a, b, {c.x - c2.x, c.y - c2.y, c.z - c2.z}, requested_point, current_d)) return true;
        if(this->xyz2d_helper({x + (a.x - dax) + (c2.x - dcx), y + (a.y - day) + (c2.y - dcy), z + (a.z - daz) + (c2.z - dcz)}, {-c2.x, -c2.y, -c2.z}, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}, b, requested_point, current_d)) return true;
    }
    else
    {
        if(this->xyz2d_helper(startPoint, b2, c2, a2, requested_point, current_d)) return true;
        if(this->xyz2d_helper({x + b2.x, y + b2.y, z + b2.z}, c, a2, {b.x - b2.x, b.y - b2.y, b.z - b2.z}, requested_point, current_d)) return true;
        if(this->xyz2d_helper({x + (b2.x - dbx) + (c.x - dcx), y + (b2.y - dby) + (c.y - dcy), z + (b2.z - dbz) + (c.z - dcz)}, a, {-b2.x, -b2.y, -b2.z}, {-(c.x - c2.x), -(c.y - c2.y), -(c.z - c2.z)}, requested_point, current_d)) return true;
        if(this->xyz2d_helper({x + (a.x - dax) + b2.x + (c.x - dcx), y + (a.y - day) + b2.y + (c.y - dcy), z + (a.z - daz) + b2.z + (c.z - dcz)}, {-c.x, -c.y, -c.z}, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}, {b.x - b2.x, b.y - b2.y, b.z - b2.z}, requested_point, current_d)) return true;
        if(this->xyz2d_helper({x + (a.x - dax) + (b2.x - dbx), y + (a.y - day) + (b2.y - dby), z + (a.z - daz) + (b2.z - dbz)}, {-b2.x, -b2.y, -b2.z}, c2, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}, requested_point, current_d)) return true;
    }

    return false;
}


Vector3D HilbertConvertor3D::d2xyz(hilbert_index_t d) const
{
    Vector3D result;
    hilbert_index_t current_d = 0;
    this->d2xyz_helper({0, 0, 0}, {this->div.x, 0, 0}, {0, this->div.y, 0}, {0, 0, this->div.z}, d, current_d, result);
    return result;
}

hilbert_index_t HilbertConvertor3D::xyz2d(coord_t x, coord_t y, coord_t z) const
{
    // convert (x,y, z) to the integer triple (width, height, width)
    int width = std::floor((x - this->ll.x) / this->step.x);
    int height = std::floor((y - this->ll.y) / this->step.y);
    int depth = std::floor((z - this->ll.z) / this->step.z);

    hilbert_index_t result = 0;
    if(not this->xyz2d_helper({0, 0, 0}, {this->div.x, 0, 0}, {0, this->div.y, 0}, {0, 0, this->div.z}, {width, height, depth}, result))
    {
        throw UniversalError("Should not reach here (in 3D xyz->d), point is (" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ") (maybe outside the box?)");
    }
    return result;
}