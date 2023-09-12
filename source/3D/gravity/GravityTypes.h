#ifndef _GRAVITY_TYPES_H
#define _GRAVITY_TYPES_H

typedef double gravity_result_t;

template<typename T>
struct MassedPoint
{
    T point;
    gravity_result_t mass;

    MassedPoint(const T &point, gravity_result_t mass): point(point), mass(mass){};
};

#endif // _GRAVITY_TYPES_H