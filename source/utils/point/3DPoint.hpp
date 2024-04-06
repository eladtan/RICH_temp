#ifndef _3DPOINT_HPP
#define _3DPOINT_HPP

#include <iostream>
#include "3D/elementary/Vector3D.hpp"
#ifdef RICH_MPI
    #include "misc/serializable.hpp"
#endif // RICH_MPI

#define EPSILON 1e-12

struct _3DPoint 
    #ifdef RICH_MPI
                : public Serializable
    #endif // RICH_MPI
{
    using coord_type = double;

    coord_type x;
    coord_type y;
    coord_type z;

    inline _3DPoint(coord_type x, coord_type y, coord_type z): x(x), y(y), z(z){};

    inline _3DPoint(): _3DPoint(coord_type(), coord_type(), coord_type()){};

    template<typename VectorType>
    inline _3DPoint(const VectorType &other): _3DPoint(other[0], other[1], other[2]){};

    template<typename VectorType>
    inline _3DPoint &operator=(const VectorType &other)
    {
        this->x = other[0];
        this->y = other[1];
        this->z = other[2];
        return (*this);
    };

    inline const coord_type &operator[](size_t idx) const{if(idx == 0) return x; if(idx == 1) return y; return z;};
    
    inline coord_type &operator[](size_t idx){if(idx == 0) return x; if(idx == 1) return y; return z;};
    
    template<typename VectorType>
    inline _3DPoint operator+(const VectorType &other) const
    {
        return _3DPoint(this->x + other[0], this->y + other[1], this->z + other[2]);
    };
    
    template<typename VectorType>
    inline _3DPoint operator-(const VectorType &other) const
    {
        return _3DPoint(this->x - other[0], this->y - other[1], this->z - other[2]);
    };
    
    inline _3DPoint operator*(coord_type constant) const
    {
        return _3DPoint(this->x * constant, this->y * constant, this->z * constant);
    };

    inline friend _3DPoint operator*(coord_type constant, const _3DPoint &point)
    {
        return _3DPoint(point.x * constant, point.y * constant, point.z * constant);
    };

    inline _3DPoint operator/(coord_type constant) const{return this->operator*(1/constant);};
    
    template<typename VectorType>
    inline _3DPoint &operator+=(const VectorType &other)
    {
        return this->operator=(this->operator+(other));
    };

    template<typename VectorType>
    inline _3DPoint &operator-=(const VectorType &other)
    {
        return this->operator=(this->operator-(other));
    };

    inline _3DPoint &operator*=(coord_type constant)
    {
        return this->operator=(this->operator*(constant));
    };

    inline _3DPoint &operator/=(coord_type constant)
    {
        return this->operator=(this->operator/(constant));
    };

    template<typename VectorType>
    inline bool operator==(const VectorType &other) const
    {
        return (std::abs(x - other[0]) < EPSILON) && (std::abs(y - other[1]) < EPSILON) && (std::abs(z - other[2]) < EPSILON);
    };

    template<typename VectorType>
    inline bool operator!=(const VectorType &other) const
    {
        return !(this->operator==(other));
    };

    friend inline std::ostream &operator<<(std::ostream &stream, const _3DPoint &point)
    {
        return stream << "(" << point.x << ", " << point.y << ", " << point.z << ")";
    }

    friend inline std::istream &operator>>(std::istream &stream, _3DPoint &point)
    {
        std::string str;
        std::getline(stream, str, '(');
        std::getline(stream, str, ',');
        point.x = std::stod(str);
        std::getline(stream, str, ',');
        point.y = std::stod(str);
        std::getline(stream, str, ')');
        point.z = std::stod(str);
        return stream;
    }

    #ifdef RICH_MPI
        inline size_t getChunkSize(void) const override
        {
            return 3;
        }
        
        inline std::vector<double> serialize(void) const override
        {
            return std::vector<double>({this->x, this->y, this->z});
        }

        inline void unserialize(const std::vector<double>& data) override
        {
            this->x = data[0];
            this->y = data[1];
            this->z = data[2];
        }
    #endif // RICH_MPI
};

// specializations

template<>
inline _3DPoint::_3DPoint(const _3DPoint &other): _3DPoint(other.x, other.y, other.z){};

template<>
inline _3DPoint &_3DPoint::operator=(const _3DPoint &other)
{
    this->x = other.x;
    this->y = other.y;
    this->z = other.y;
    return (*this);
};

template<>
inline _3DPoint::_3DPoint(const Vector3D &other): _3DPoint(other.x, other.y, other.z){};

template<>
inline _3DPoint &_3DPoint::operator=(const Vector3D &other)
{
    this->x = other.x;
    this->y = other.y;
    this->z = other.y;
    return (*this);
};

template<>
inline _3DPoint _3DPoint::operator+(const _3DPoint &other) const
{
    return _3DPoint(this->x + other.x, this->y + other.y, this->z + other.z);
};

template<>
inline _3DPoint _3DPoint::operator+(const Vector3D &other) const
{
    return _3DPoint(this->x + other.x, this->y + other.y, this->z + other.z);
};

template<>
inline _3DPoint _3DPoint::operator-(const _3DPoint &other) const
{
    return _3DPoint(this->x - other.x, this->y - other.y, this->z - other.z);
};

template<>
inline _3DPoint _3DPoint::operator-(const Vector3D &other) const
{
    return _3DPoint(this->x - other.x, this->y - other.y, this->z - other.z);
};

template<>
inline bool _3DPoint::operator==(const _3DPoint &other) const
{
    return (std::abs(x - other.x) < EPSILON) && (std::abs(y - other.y) < EPSILON) && (std::abs(z - other.z) < EPSILON);
};

template<>
inline bool _3DPoint::operator==(const Vector3D &other) const
{
    return (std::abs(x - other.x) < EPSILON) && (std::abs(y - other.y) < EPSILON) && (std::abs(z - other.z) < EPSILON);
};

// abs and fastabs methods

inline _3DPoint::coord_type abs(const _3DPoint &v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline _3DPoint::coord_type fastabs(const _3DPoint &v)
{
    return fastsqrt((v.x * v.x) + (v.y * v.y) + (v.z * v.z));
}

#endif // _3DPOINT_HPP