#ifndef _RANGE_TREE_FINDER_HPP
#define _RANGE_TREE_FINDER_HPP

#include <spatial_ds/BinaryTree.hpp>
#include <spatial_ds/RangeTree/RangeTree.hpp>
#include "utils/IndexedVector.hpp"
#include "RangeFinder.hpp"

#define DIMENSIONS 3

class RangeTreeFinder : public RangeFinder
{
public:
    template<typename RandomAccessIterator>
    RangeTreeFinder(RandomAccessIterator first, RandomAccessIterator last);
    
    inline RangeTreeFinder(std::vector<Vector3D> &myPoints): RangeTreeFinder(myPoints.begin(), myPoints.end()){};
    
    inline ~RangeTreeFinder() override{delete this->rangeTree;};

    inline const Vector3D &getPoint(size_t index) const override{return this->myPoints[index];};

    inline std::vector<size_t> range(const Vector3D &center, double radius, size_t N, const _set<size_t> &ignore) const override
    {
        std::vector<size_t> toReturn;
        for(const IndexedVector3D &vec : this->rangeTree->circularRange(center, radius, N, [&ignore](const IndexedVector3D &vec){return ignore.find(vec.getIndex()) == ignore.cend();}))
        {
            toReturn.push_back(vec.index);
        }
        return toReturn;
    };
    inline size_t size() const override{return this->rangeTree->size();};

private:
    std::vector<Vector3D> myPoints;
    RangeTree<IndexedVector3D> *rangeTree;
};

template<typename RandomAccessIterator>
inline RangeTreeFinder::RangeTreeFinder(RandomAccessIterator first, RandomAccessIterator last)
{
    std::vector<IndexedVector3D> data;
    size_t index = 0;

    myPoints.reserve(last - first);
    for(RandomAccessIterator it = first; it != last; it++)
    {
        const Vector3D &vec = *it;
        this->myPoints.push_back(vec);
        IndexedVector3D idx_vec = IndexedVector3D(vec.x, vec.y, vec.z, index);
        data.push_back(idx_vec);
        index++;
    }
    this->rangeTree = new RangeTree<IndexedVector3D>(DIMENSIONS);
    this->rangeTree->build(data.begin(), data.end());
}

#endif // _RANGE_TREE_FINDER_HPP