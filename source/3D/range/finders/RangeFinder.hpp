#ifndef _RANGE_FINDER_HPP
#define _RANGE_FINDER_HPP

#include <boost/container/flat_set.hpp>
#include <boost/unordered_set.hpp>
#include <vector>
#include <limits>
#include <unordered_set>
#include "../../elementary/Vector3D.hpp"

class RangeFinder
{
public:
    // Using unordered_set for O(1) lookup instead of O(log N) for flat_set
    template<typename T>
    using _set = std::unordered_set<T>;
    virtual ~RangeFinder() = default;
    
    virtual std::vector<size_t> range(const Vector3D &center, double radius, size_t N = std::numeric_limits<size_t>::max(), const _set<size_t> &ignore = _set<size_t>()) const = 0;
    
    virtual std::vector<size_t> closestPointInSphere(const Vector3D &center, double radius, const Vector3D &point, const _set<size_t> &ignore) const = 0;
    
    virtual const Vector3D &getPoint(size_t index) const = 0;
    
    virtual size_t size() const = 0;
};

#endif // _RANGE_FINDER_HPP