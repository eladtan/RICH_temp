#ifndef _RANGE_FINDER_HPP
#define _RANGE_FINDER_HPP

#include "utils/IndexedVector.hpp"
#include "../../elementary/Vector3D.hpp"
#include <vector>

class RangeFinder
{
public:
    virtual std::vector<IndexedVector3D> range(const _3DPoint &center, typename _3DPoint::coord_type radius) const = 0;
    // virtual const Vector3D &getPoint(size_t index) const = 0;
    virtual size_t size() const = 0;
};

#endif // _RANGE_FINDER_HPP