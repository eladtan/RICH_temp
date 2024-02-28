#ifndef _GEOMETRY_UTILS_RICH_HPP
#define _GEOMETRY_UTILS_RICH_HPP

#ifdef USE_VCL_VECTORIZATION
    #include <vectorclass.h>
#endif // USE_VCL_VECTORIZATION

#include "misc/serializable.hpp"

#define DIM 3

template<typename T>
class _BoundingBox : public Serializable
{
private:
    T ll; // leftmost point of the box
    T ur; // rightmost point of the box

    #ifdef USE_VCL_VECTORIZATION
        Vec4d llVec, urVec;
        Vec4d llPlusUrVec;
        Vec8d boundariesVec;
    #endif // USE_VCL_VECTORIZATION
    typename T::coord_type widthSquared;

    void recalculateFields()
    {
        typename T::coord_type width = std::max(this->ur[0] - this->ll[0], std::max(this->ur[1] - this->ll[1], this->ur[2] - this->ll[2]));
        this->widthSquared = width * width;
        #ifdef USE_VCL_VECTORIZATION
            this->llVec = Vec4d(this->ll[0], this->ll[1], this->ll[2], 0);
            this->urVec = Vec4d(this->ur[0], this->ur[1], this->ur[2], 0);
            this->llPlusUrVec = this->llVec + this->urVec;
            this->boundariesVec = Vec8d(this->ll[0], this->ll[1], this->ll[2], this->ur[0], this->ur[1], this->ur[2], 0, 0);
        #endif // USE_VCL_VECTORIZATION
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
    inline bool contained(const _BoundingBox<U> &other) const
    {
        // need: other.ll[i] >= this->ll[i] and other.ur[i] <= this->ur[i] for all i
        for(int i = 0; i < DIM; i++)
        {
            if((other.ll[i] < this->ll[i]) or (other.ur[i] > this->ur[i]))
            {
                return false;
            }
        }
        return true;
    }

    template<typename U>
    inline bool intersects(const _BoundingBox<U> &other) const
    {
        return this->contains(other.getLL()) or this->contains(other.getUR()) or other.contains(this->ll) or other.contains(this->ur);
    }

    template<typename U>
    inline T closestPoint(const U &point) const
    {
        #ifdef USE_VCL_VECTORIZATION
            Vec8d _point(point[0], point[1], point[2], point[0], point[1], point[2], 0, 0);
            Vec8db cmp = _point < this->boundariesVec; // _point < _boundaries;
        #endif // USE_VCL_VECTORIZATION
        T closestPoint;
        for(int i = 0; i < DIM; i++)
        {
            #ifdef USE_VCL_VECTORIZATION
                if(cmp[i])
            #else
                if(point[i] < this->ll[i])
            #endif // USE_VCL_VECTORIZATION
            {
                // that means point[i] < this->ll[i]
                closestPoint[i] = this->ll[i];
            }
            else
            {
                #ifdef USE_VCL_VECTORIZATION
                    if(cmp[i + DIM])
                #else // USE_VCL_VECTORIZATION
                    if(point[i] < this->ur[i])
                #endif // USE_VCL_VECTORIZATION
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
    inline typename T::coord_type distanceSquared(const U &point) const
    {
        T closestPoint = this->closestPoint(point);
        #ifdef USE_VCL_VECTORIZATION
            Vec4d closestPointVec(closestPoint[0] - point[0], closestPoint[1] - point[1], closestPoint[2] - point[2], 0);
            Vec4d pointSquaredVec = closestPointVec * closestPointVec;
            return pointSquaredVec[0] + pointSquaredVec[1] + pointSquaredVec[2];
        #else
            T::coord_type closestDistance = 0;
            for(int i = 0; i < DIM; i++)
            {
                closestDistance += (point[i] - closestPoint[i]);
            }
            return closestDistance;
        #endif // USE_VCL_VECTORIZATION
    }

    template<typename U>
    inline T furthestPoint(const U &point) const
    {
        #ifdef USE_VCL_VECTORIZATION
            Vec4d _point(point[0], point[1], point[2], 0);
            Vec4db cmp = (2 * _point) > this->llPlusUrVec;
        #endif // USE_VCL_VECTORIZATION

        T furthestPoint;

        for(int i = 0; i < DIM; i++)
        {
            #ifdef USE_VCL_VECTORIZATION
                if(cmp[i])
            #else // USE_VCL_VECTORIZATION
                if(point[i] < ((this->ll[i] + this->ur[i]) * 0.5))
            #endif // USE_VCL_VECTORIZATION
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

    friend inline std::ostream &operator<<(std::ostream &stream, const _BoundingBox<T> &box)
    {
        return stream << "BoundingBox(" << box.ll << ", " << box.ur << ")";
    }

    inline size_t getChunkSize(void) const override
    {
        if constexpr(is_serializable<T>::value)
        {
            return this->ll.getChunkSize() + this->ur.getChunkSize();
        }
        throw UniversalError("BoundingBox: type is not serializable");
    }

    inline std::vector<double> serialize(void) const override
    {
        std::vector<double> data;
        if constexpr(is_serializable<T>::value)
        {
            std::vector<double> llData = this->ll.serialize();
            std::vector<double> urData = this->ur.serialize();
            data.insert(data.end(), llData.begin(), llData.end());
            data.insert(data.end(), urData.begin(), urData.end());
            return data;
        }
        throw UniversalError("BoundingBox: type is not serializable");
    }

    inline void unserialize(const std::vector<double>& data) override
    {
        if constexpr(is_serializable<T>::value)
        {
            std::vector<double> llData(data.begin(), data.begin() + this->ll.getChunkSize());
            std::vector<double> urData(data.begin() + this->ll.getChunkSize(), data.end());
            T ll_;
            ll_.unserialize(llData);
            T ur_;
            ur_.unserialize(urData);
            this->setBounds(ll_, ur_);
        }
        else
        {
            throw UniversalError("BoundingBox: type is not serializable");
        }
    }

    template<typename U>
    inline bool operator==(const _BoundingBox<U> &other)
    {
        return ((this->ll == other.ll) and (this->ur == other.ur));
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

    friend inline std::ostream &operator<<(std::ostream &stream, const _Sphere<T> &sphere)
    {
        return stream << "Sphere(" << sphere.center << ", " << sphere.radius << ")";
    }
};

/**
 * @brief Checks if a sphere intersects a box
*/
template<typename T, typename U>
bool SphereBoxIntersection(const _BoundingBox<T> &box, const _Sphere<U> &sphere)
{
    typename T::coord_type distance = 0;
    const T &boxLL = box.getLL(), &boxUR = box.getUR();
    for(int i = 0; i < DIM; i++)
    {
        typename T::coord_type centerCoord = sphere.center[i];
        typename T::coord_type closestPointCoord;

        if(centerCoord < boxLL[i])
        {
            closestPointCoord = boxLL[i];
        }
        else
        {
            if(centerCoord <= boxUR[i])
            {
                closestPointCoord = centerCoord;
            }
            else
            {
                closestPointCoord = boxUR[i];
            }
        }
        typename T::coord_type _distance = (closestPointCoord - sphere.center[i]);
        _distance *= _distance;
        distance += _distance;
    }
    return (distance <= (sphere.radius * sphere.radius));
};

#endif // _GEOMETRY_UTILS_RICH_HPP