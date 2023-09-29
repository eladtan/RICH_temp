#ifndef _GRAVITY_TYPES_H
#define _GRAVITY_TYPES_H

#include <iostream> // todo remove

typedef double gravity_result_t;

template<typename T>
struct MassedPoint
{
    T point;
    gravity_result_t mass;

    MassedPoint(const T &point, gravity_result_t mass): point(point), mass(mass){};
    MassedPoint(): MassedPoint(T(), 0){};

    friend inline std::ostream &operator<<(std::ostream &stream, const MassedPoint<T> &massedPoint)
    {
        return stream << massedPoint.point << " (mass " << massedPoint.mass << ")" << std::endl;
    }
};

#endif // _GRAVITY_TYPES_H