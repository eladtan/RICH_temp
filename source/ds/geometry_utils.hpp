#ifndef _GEOMETRY_UTILS_RICH_HPP
#define _GEOMETRY_UTILS_RICH_HPP

#include <vectorclass.h>

typedef double coord_t;

#define DIM 3

template<typename T>
class _BoundingBox
{
public:
    T ll; // leftmost point of the box
    T ur; // rightmost point of the box

    inline _BoundingBox(const T &ll, const T &ur): ll(ll), ur(ur){};
    inline _BoundingBox(): _BoundingBox(T(), T()){};
    
    inline bool contains(const T &point) const
    {
        for(int i = 0; i < DIM; i++)
        {
            if(point[i] < ll[i] | point[i] > ur[i])
            {
                return false;
            }
        }
        return true;
    }

    inline T closestPoint(const T &point) const
    {
        Vec8d _point(point[0], point[1], point[2], point[0], point[1], point[2], 0, 0);
        Vec8d _boundaries(this->ll[0], this->ll[1], this->ll[2], this->ur[0], this->ur[1], this->ur[2], 0, 0);
        Vec8db cmp = _point < _boundaries;
        T closestPoint;
        for(int i = 0; i < DIM; i++)
        {
            if(cmp[i])
            {
                // that means point[i] < this->ll[i]
                closestPoint[i] = this->ll[i];
            }
            else
            {
                if(cmp[i + DIM])
                {
                    // that means point[i] < this->ur[i]
                    closestPoint[i] = point[i];
                }
                else
                {
                    closestPoint[i] = this->ur[i];
                }
            }
        }
        return closestPoint;
    }

    inline T furthestPoint(const T &point) const
    {
        Vec4d _point(point[0], point[1], point[2], 0);
        Vec4d _ll(this->ll[0], this->ll[1], this->ll[2], 0);
        Vec4d _ur(this->ur[0], this->ur[1], this->ur[2], 0);
        Vec4db cmp = (2 * _point) > (_ll + _ur);
        T furthestPoint;
        for(int i = 0; i < DIM; i++)
        {
            if(cmp[i])
            {
                // that means point[i] > (this->ll[i] + this->ur[i]) / 2
                furthestPoint[i] = this->ll[i];
            }
            else
            {
                furthestPoint[i] = this->ur[i];
            }
        }
        return furthestPoint;
    }
};

template<typename T>
class _Sphere
{
public:
    T center;
    typename T::coord_type radius;

    _Sphere(const T &center, typename T::coord_type radius): center(center), radius(radius){};
    inline bool contains(const T &point) const
    {
        typename T::coord_type distance = 0;
        for(int i = 0; i < DIM; i++)
        {
            double _distance = (point[i] - this->center[i]);
            distance += _distance * _distance;
        }
        return distance <= (this->radius * this->radius);
    }
};

template<typename T>
bool SphereBoxIntersection(const _BoundingBox<T> &box, const _Sphere<T> &sphere)
{
    T closestPoint;
    typename T::coord_type distance = 0;
    for(int i = 0; i < DIM; i++)
    {
        typename T::coord_type centerCoord = sphere.center[i];
        if(centerCoord < box.ll[i])
        {
            closestPoint[i] = box.ll[i];
        }
        else
        {
            if(centerCoord <= box.ur[i])
            {
                closestPoint[i] = centerCoord;
            }
            else
            {
                closestPoint[i] = box.ur[i];
            }
        }
        // closestPoint[i] = (sphere.center[i] < box.ll[i])? box.ll[i] : ((sphere.center[i] > box.ur[i])? box.ur[i] : sphere.center[i]);
        typename T::coord_type _distance = (closestPoint[i] - sphere.center[i]);
        _distance *= _distance;
        distance += _distance;
    }
    return (distance <= (sphere.radius * sphere.radius));
};

#endif // _GEOMETRY_UTILS_RICH_HPP