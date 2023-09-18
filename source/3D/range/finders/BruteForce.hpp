#ifndef _BRUTE_FORCE_RANGE_HPP
#define _BRUTE_FORCE_RANGE_HPP

#include "RangeFinder.hpp"

class BruteForceFinder : public RangeFinder
{
public:
    template<typename RandomAccessIterator>
    BruteForceFinder(const RandomAccessIterator &first, const RandomAccessIterator &last):
                points(std::vector<Vector3D>()){this->points.insert(this->points.begin(), first, last); this->pointsSize = this->points.size();};
    BruteForceFinder(const std::vector<Vector3D> &points): BruteForceFinder(points.begin(), points.end()){};

    std::vector<size_t> closestPointInSphere(const Vector3D &center, double radius, const Vector3D &point, const _set<size_t> &ignore) const override
    {
        return std::vector<size_t>();
    }

    inline std::vector<size_t> range(const Vector3D &center, double radius, size_t maxPointsToGet) const override
    {
        std::vector<size_t> result;
        const Vector3D *_points = this->points.data();
        for(size_t i = 0; i < this->pointsSize; i++)
        {
            //  __builtin_prefetch(&_points[i]);
            const Vector3D &point = _points[i];
            double dx = point.x - center.x;
            double dy = point.y - center.y;
            double dz = point.z - center.z;
            dx *= dx;
            dy *= dy;
            dz *= dz;
            if(std::abs((dx + dy + dz) - (radius * radius)) <= EPSILON)
            {
                result.push_back(i);
            }
        }
        return result;
    }

    inline const Vector3D &getPoint(size_t index) const override{return this->points[index];};

    inline size_t size() const override{return this->pointsSize;};

private:
    std::vector<Vector3D> points;
    size_t pointsSize;
};

#endif // _BRUTE_FORCE_RANGE_HPP
