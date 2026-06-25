#ifndef _GROUP_RANGE_TREE_FINDER_HPP
#define _GROUP_RANGE_TREE_FINDER_HPP

#include <spatial_ds/GroupRangeTree/GroupRangeTree.hpp>
#include "utils/IndexedVector.hpp"
#include "RangeFinder.hpp"

#define DIMENSIONS 3

template<int GroupSize>
class GroupRangeTreeFinder : public RangeFinder
{
public:
    template<typename RandomAccessIterator>
    GroupRangeTreeFinder(RandomAccessIterator first, RandomAccessIterator last);
    
    inline GroupRangeTreeFinder(std::vector<Vector3D> &myPoints): GroupRangeTreeFinder(myPoints.begin(), myPoints.end()){};
    
    inline ~GroupRangeTreeFinder() override{delete this->groupRangeTree;};

    inline const Vector3D &getPoint(size_t index) const override{return this->myPoints[index];};

    inline std::vector<size_t> range(const Vector3D &center, double radius, size_t N, const _set<size_t> &ignore) const override
    {
        std::vector<size_t> toReturn;
        for(const IndexedVector3D &vec : this->groupRangeTree->circularRange(center, radius, N, [&ignore](const IndexedVector3D &vec){return ignore.find(vec.getIndex()) == ignore.cend();}))
        {
            toReturn.push_back(vec.index);
        }
        return toReturn;
    };
    inline size_t size() const override{return this->myPoints.size();};

private:
    std::vector<Vector3D> myPoints;
    GroupRangeTree<IndexedVector3D, GroupSize> *groupRangeTree;
};

template<int GroupSize>
template<typename RandomAccessIterator>
inline GroupRangeTreeFinder<GroupSize>::GroupRangeTreeFinder(RandomAccessIterator first, RandomAccessIterator last)
{
    size_t index = 0;
    std::vector<IndexedVector3D> data;
    for(RandomAccessIterator it = first; it != last; it++)
    {
        const Vector3D &vec = *it;
        this->myPoints.push_back(vec);
        IndexedVector3D idx_vec = IndexedVector3D(vec.x, vec.y, vec.z, index);
        data.push_back(idx_vec);
        index++;
    }
    this->groupRangeTree = new GroupRangeTree<IndexedVector3D, GroupSize>(DIMENSIONS);
    this->groupRangeTree->build(data.begin(), data.end());
}

#endif // _GROUP_RANGE_TREE_FINDER_HPP