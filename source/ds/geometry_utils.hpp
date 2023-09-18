#ifndef _GEOMETRY_UTILS_RICH_HPP
#define _GEOMETRY_UTILS_RICH_HPP

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
        T closestPoint;
        for(int i = 0; i < DIM; i++)
        {
            if(point[i] < this->ll[i])
            {
                closestPoint[i] = this->ll[i];
            }
            else
            {
                if(point[i] <= this->ur[i])
                {
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
};

template<typename T>
class _Sphere
{
public:
    T center;
    coord_t radius;

    _Sphere(const T &center, coord_t radius): center(center), radius(radius){};
    inline bool contains(const T &point) const
    {
        coord_t distance = 0;
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
    coord_t distance = 0;
    for(int i = 0; i < DIM /*sphere.dim*/; i++)
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