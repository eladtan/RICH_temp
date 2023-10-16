#ifndef _HILBERT_TYPES_H
#define _HILBERT_TYPES_H

#include <iostream>
#include <cmath>
#include "3D/elementary/Vector3D.hpp"

typedef double coord_t;
typedef unsigned long int hilbert_index_t;


#define EPSILON 1e-12

typedef struct _3DPoint
{
    using coord_type = coord_t;

    coord_type x;
    coord_type y;
    coord_type z;

    _3DPoint(coord_type x, coord_type y, coord_type z): x(x), y(y), z(z){};
    _3DPoint(): _3DPoint(coord_type(), coord_type(), coord_type()){};
    _3DPoint(const _3DPoint &other): _3DPoint(other.x, other.y, other.z){};
    _3DPoint(const Vector3D &vector): _3DPoint(vector.x, vector.y, vector.z){};

    inline _3DPoint operator+(const _3DPoint &other) const{return _3DPoint(this->x + other.x, this->y + other.y, this->z + other.z);};
    inline _3DPoint operator-(const _3DPoint &other) const{return _3DPoint(this->x - other.x, this->y - other.y, this->z - other.z);};
    inline _3DPoint operator*(coord_type scalar) const{return _3DPoint(this->x * scalar, this->y * scalar, this->z * scalar);};
    inline _3DPoint operator/(coord_type scalar) const{return this->operator*(1 / scalar);};
    inline _3DPoint &operator=(const _3DPoint &other){this->x = other.x; this->y = other.y; this->z = other.z; return *this;};
    inline _3DPoint &operator+=(const _3DPoint &other){this->x += other.x; this->y += other.y; this->z += other.z; return *this;};
    inline _3DPoint &operator-=(const _3DPoint &other){this->x -= other.x; this->y -= other.y; this->z -= other.z; return *this;};
    inline bool operator==(const _3DPoint &other) const{return (std::abs(this->x - other.x) <= EPSILON) and (std::abs(this->y - other.y) <= EPSILON) and (std::abs(this->z - other.z) <= EPSILON);};
    inline bool operator!=(const _3DPoint &other) const{return !this->operator==(other);};
    inline coord_type &operator[](size_t idx){if(idx == 0) return this->x; if(idx == 1) return this->y; return this->z;};
    inline coord_type operator[](size_t idx) const{if(idx == 0) return this->x; if(idx == 1) return this->y; return this->z;};

    inline coord_type fastabs() const{return fastsqrt((this->x * this->x) + (this->y * this->y) + (this->z * this->z));}
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

} _3DPoint;

inline typename _3DPoint::coord_type abs(_3DPoint const& v)
{
	return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline typename _3DPoint::coord_type fastabs(_3DPoint const& v)
{
	return fastsqrt((v.x * v.x) + (v.y * v.y) + (v.z * v.z));
}

typedef struct _3DPointRadius
{
    _3DPoint point;
    double radius;
} _3DPointRadius;


typedef struct _2DPoint
{
    using coord_type = coord_t;

    coord_type x;
    coord_type y;
} _2DPoint;

#endif // _HILBERT_TYPES_H