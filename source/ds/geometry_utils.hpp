#ifndef _GEOMETRY_UTILS_RICH_HPP
#define _GEOMETRY_UTILS_RICH_HPP

#include <vectorclass.h>

#define DIM 3

template<typename T>
class _BoundingBox
{
private:
    T ll; // leftmost point of the box
    T ur; // rightmost point of the box
    Vec4d llVec, urVec;
    Vec4d llPlusUrVec;
    Vec8d boundariesVec;
    typename T::coord_type widthSquared;

    void recalculateFields()
    {
        typename T::coord_type width = std::max(this->ur[0] - this->ll[0], std::max(this->ur[1] - this->ll[1], this->ur[2] - this->ll[2]));
        this->widthSquared = width * width;
        this->llVec = Vec4d(this->ll[0], this->ll[1], this->ll[2], 0);
        this->urVec = Vec4d(this->ur[0], this->ur[1], this->ur[2], 0);
        this->llPlusUrVec = this->llVec + this->urVec;
        this->boundariesVec = Vec8d(this->ll[0], this->ll[1], this->ll[2], this->ur[0], this->ur[1], this->ur[2], 0, 0);
    }

public:
    inline _BoundingBox(const T &ll, const T &ur):
        ll(ll), ur(ur)
    {
        this->recalculateFields();
    };
    inline _BoundingBox(): _BoundingBox(T(), T()){}

    const T &getLL() const{return this->ll;};
    const T &getUR() const{return this->ur;};

    inline void setLL(const T &newLL)
    {
        this->ll = newLL;
        this->recalculateFields();
    }

    inline void setUR(const T &newUR)
    {
        this->ur = newUR;
        this->recalculateFields();
    }
    
    inline void setBounds(const T &newLL, const T &newUR)
    {
        this->ll = newLL;
        this->ur = newUR;
        this->recalculateFields();
    }

    inline typename T::coord_type getWidthSquared() const
    {
        return this->widthSquared;
    }

    template<typename U>
    inline bool contains(const U &point) const
    {
        for(int i = 0; i < DIM; i++)
        {
            if(point[i] < ll[i] | point[i] > ur[i])
            {
                return false;
            }
        }
        return true;
        /*
        Vec8d _point(point[0], point[1], point[2], -point[0], -point[1], -point[2], 0, 0);
        Vec8d _boundaries(this->ll[0], this->ll[1], this->ll[2], -this->ur[0], -this->ur[1], -this->ur[2], 1, 1);
        Vec8db cmp = _point < _boundaries;

        for(int i = 0; i < DIM; i++)
        {
            if((!cmp[i]) or (!cmp[i + DIM]))
            {
                return false;
            }
        }
        return true;
        */
    }

    inline T closestPoint(const T &point) const
    {
        Vec8d _point(point[0], point[1], point[2], point[0], point[1], point[2], 0, 0);
        Vec8db cmp = _point < this->boundariesVec; // _point < _boundaries;
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
        Vec4db cmp = (2 * _point) > this->llPlusUrVec;
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
    const T &boxLL = box.getLL(), &boxUR = box.getUR();
    for(int i = 0; i < DIM; i++)
    {
        typename T::coord_type centerCoord = sphere.center[i];
        if(centerCoord < boxLL[i])
        {
            closestPoint[i] = boxLL[i];
        }
        else
        {
            if(centerCoord <= boxUR[i])
            {
                closestPoint[i] = centerCoord;
            }
            else
            {
                closestPoint[i] = boxUR[i];
            }
        }
        typename T::coord_type _distance = (closestPoint[i] - sphere.center[i]);
        _distance *= _distance;
        distance += _distance;
    }
    return (distance <= (sphere.radius * sphere.radius));
};

#endif // _GEOMETRY_UTILS_RICH_HPP