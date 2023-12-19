#ifndef RANGE_QUERY_DATA
#define RANGE_QUERY_DATA

#include "utils/point/3DPoint.hpp"

struct RangeQueryData
{
    size_t pointIdx;
    _3DPoint center;
    typename _3DPoint::coord_type radius;
};

#endif // RANGE_QUERY_DATA