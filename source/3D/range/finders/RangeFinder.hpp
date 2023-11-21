#ifndef _RANGE_FINDER_HPP
#define _RANGE_FINDER_HPP

#include <boost/container/flat_set.hpp>
#include "../../elementary/Vector3D.hpp"
#include <vector>
#include <limits>
#include <unordered_set>
#include <boost/unordered_set.hpp>

class RangeFinder
{
public:
    template<typename T>
    using _set = boost::container::flat_set<T>; // boost::unordered_set<T>; // std::unordered_set<T>; // boost::container::flat_set<T>;
    virtual ~RangeFinder() = default;
    virtual std::vector<size_t> range(const Vector3D &center, double radius, size_t N = std::numeric_limits<size_t>::max(), const _set<size_t> &ignore = _set<size_t>()) const = 0;
    virtual std::vector<size_t> closestPointInSphere(const Vector3D &center, double radius, const Vector3D &point, const _set<size_t> &ignore) const = 0;
    virtual const Vector3D &getPoint(size_t index) const = 0;
    virtual size_t size() const = 0;
};

#endif // _RANGE_FINDER_HPP