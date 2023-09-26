#ifndef _GROUP_RANGE_TREE_FINDER_HPP
#define _GROUP_RANGE_TREE_FINDER_HPP

#include "ds/GroupTree/GroupTree.h"
#include "ds/GroupRangeTree/GroupRangeTree.h"
#include "ds/GroupTree/GroupTree.cpp" // todo: not good
#include "ds/GroupRangeTree/GroupRangeTree.cpp" // todo: not good
#include "utils/IndexedVector.hpp"
#include "RangeFinder.hpp"

#define DIMENSIONS 3

template<int N>
class GroupRangeTreeFinder : public RangeFinder
{
public:
    template<typename RandomAccessIterator>
    GroupRangeTreeFinder(RandomAccessIterator first, RandomAccessIterator last);
    inline GroupRangeTreeFinder(std::vector<Vector3D> &myPoints): GroupRangeTreeFinder(myPoints.begin(), myPoints.end()){};
    ~GroupRangeTreeFinder();

    std::vector<size_t> closestPointInSphere(const Vector3D &center, double radius, const Vector3D &point, const _set<size_t> &ignore) const override
    {
        return std::vector<size_t>();
    }

    inline std::vector<size_t> range(const Vector3D &center, double radius) const override{
        std::vector<size_t> toReturn;
        for(const IndexedVector3D &vec : this->groupRangeTree->circularRange(center, radius))
        {
            toReturn.push_back(vec.index);
        }
        return toReturn;
    };
    inline size_t size() const override{return this->myPoints.size();};
    inline const Vector3D &getPoint(size_t index) const override{return this->myPoints[index];};

private:
    GroupRangeTree<IndexedVector3D, N> *groupRangeTree;
    std::vector<Vector3D> myPoints;
};

template<int N>
template<typename RandomAccessIterator>
GroupRangeTreeFinder<N>::GroupRangeTreeFinder(RandomAccessIterator first, RandomAccessIterator last)
{
    std::vector<IndexedVector3D> data;
    size_t index = 0;
    for(RandomAccessIterator it = first; it != last; it++)
    {
        const Vector3D &vec = *it;
        data.push_back(IndexedVector3D(vec.x, vec.y, vec.z, index));
        this->myPoints.push_back(vec);
        index++;
    }
    this->groupRangeTree = new GroupRangeTree<IndexedVector3D, N>(DIMENSIONS);
    this->groupRangeTree->build(data.begin(), data.end());
}

template<int N>
GroupRangeTreeFinder<N>::~GroupRangeTreeFinder()
{
    delete this->groupRangeTree;
}

#endif // _GROUP_RANGE_TREE_FINDER_HPP