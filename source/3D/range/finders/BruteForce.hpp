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

    inline std::vector<IndexedVector3D> range(const _3DPoint &center, double radius) const override
    {
        std::vector<IndexedVector3D> result;
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
                result.emplace_back(IndexedVector3D(point.x, point.y, point.z, i));
            }
        }
        return result;
    }

    inline size_t size() const override{return this->pointsSize;};

private:
    std::vector<Vector3D> points;
    size_t pointsSize;
};

#endif // _BRUTE_FORCE_RANGE_HPP
