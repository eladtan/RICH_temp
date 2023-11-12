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
    template<typename U>
    inline _BoundingBox(const U &_ll, const U &_ur)
    {
        this->ll[0] = _ll[0];
        this->ll[1] = _ll[1];
        this->ll[2] = _ll[2];
        this->ur[0] = _ur[0];
        this->ur[1] = _ur[1];
        this->ur[2] = _ur[2];
        this->recalculateFields();
    };

    inline _BoundingBox(): ll(T()), ur(T())
    {
        this->recalculateFields();
    };

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
    
    template<typename U>
    inline void setBounds(const U &newLL, const U &newUR)
    {
        this->ll[0] = newLL[0];
        this->ll[1] = newLL[1];
        this->ll[2] = newLL[2];
        this->ur[0] = newUR[0];
        this->ur[1] = newUR[1];
        this->ur[2] = newUR[2];
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

    template<typename U>
    inline bool intersects(const _BoundingBox<U> &other) const
    {
        return this->contains(other.getLL()) or this->contains(other.getUR()) or other.contains(this->ll) or other.contains(this->ur);
    }

    template<typename U>
    inline T closestPoint(const U &point) const
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

    template<typename U>
    inline T furthestPoint(const U &point) const
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

    friend std::ostream &operator<<(std::ostream &stream, const _BoundingBox<T> &box)
    {
        return stream << "BoundingBox(" << box.ll << ", " << box.ur << ")";
    }
};

template<typename T>
class _Sphere
{
public:
    T center;
    typename T::coord_type radius;

    template<typename U>
    _Sphere(const U &center, typename T::coord_type radius): radius(radius)
    {
        this->center[0] = center[0];
        this->center[1] = center[1];
        this->center[2] = center[2];
    }

    template<typename U>
    inline bool contains(const U &point) const
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

template<typename T, typename U, typename V = T>
bool SphereBoxIntersection(const _BoundingBox<T> &box, const _Sphere<U> &sphere)
{
    V closestPoint;
    typename T::coord_type distance = 0;
    const T &boxLL = box.getLL(), &boxUR = box.getUR();
    for(int i = 0; i < DIM; i++)
    {
        typename V::coord_type centerCoord = sphere.center[i];
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
        typename V::coord_type _distance = (closestPoint[i] - sphere.center[i]);
        _distance *= _distance;
        distance += _distance;
    }
    return (distance <= (sphere.radius * sphere.radius));
};

#endif // _GEOMETRY_UTILS_RICH_HPP