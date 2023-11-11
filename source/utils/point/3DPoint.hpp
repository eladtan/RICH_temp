#include <iostream> // todo: remove
#include "3D/elementary/Vector3D.hpp"

struct _3DPoint
{
    using coord_type = double;

    coord_type x, y, z;
    _3DPoint(coord_type x, coord_type y, coord_type z): x(x), y(y), z(z){};
    _3DPoint(): _3DPoint(coord_type(), coord_type(), coord_type()){};
    _3DPoint(const _3DPoint &other): _3DPoint(other.x, other.y, other.z){};
    _3DPoint &operator=(const _3DPoint &other){this->x = other.x; this->y = other.y; this->z = other.z; return (*this);};
    const coord_type &operator[](size_t idx) const{if(idx == 0) return x; if(idx == 1) return y; return z;};
    coord_type &operator[](size_t idx){if(idx == 0) return x; if(idx == 1) return y; return z;};
    inline _3DPoint operator+(const _3DPoint &other) const{return _3DPoint(this->x + other.x, this->y + other.y, this->z + other.z);};
    inline _3DPoint operator-(const _3DPoint &other) const{return _3DPoint(this->x - other.x, this->y - other.y, this->z - other.z);};
    inline _3DPoint operator*(double constant) const{return _3DPoint(this->x * constant, this->y * constant, this->z * constant);};
    inline _3DPoint operator/(double constant) const{return this->operator*(1/constant);};
    inline bool operator==(const _3DPoint &other) const{return this->x == other.x and this->y == other.y and this->z == other.z;};
    inline bool operator!=(const _3DPoint &other) const{return !(this->operator==(other));};
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
};