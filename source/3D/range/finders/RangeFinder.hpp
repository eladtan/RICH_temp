#ifndef _RANGE_FINDER_HPP
#define _RANGE_FINDER_HPP

#include "../../elementary/Vector3D.hpp"
#include <vector>

class RangeFinder
{
public:
    template<typename T>
    using _set = boost::container::flat_set<T>;

    virtual std::vector<size_t> range(const Vector3D &center, double radius, size_t maxPointsToGet) const = 0;
    virtual std::vector<size_t> closestPointInSphere(const Vector3D &center, double radius, const Vector3D &point, const _set<size_t> &ignore) const = 0;
    virtual const Vector3D &getPoint(size_t index) const = 0;
    virtual size_t size() const = 0;
};

#endif // _RANGE_FINDER_HPP