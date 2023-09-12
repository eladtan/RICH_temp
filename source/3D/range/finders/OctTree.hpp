#ifndef _OCT_TREE_FINDER_HPP
#define _OCT_TREE_FINDER_HPP

#include "ds/OctTree/OctTree.hpp"
#include "utils/IndexedVector.hpp"
#include "RangeFinder.hpp"

#define DIMENSIONS 3

class OctTreeFinder : public RangeFinder
{
public:
    template<typename RandomAccessIterator>
    OctTreeFinder(RandomAccessIterator first, RandomAccessIterator last, const Vector3D &ll ,const Vector3D &ur);
    inline OctTreeFinder(std::vector<Vector3D> &myPoints, const Vector3D &ll ,const Vector3D &ur): OctTreeFinder(myPoints.begin(), myPoints.end(), ll, ur){};
    ~OctTreeFinder();
    inline std::vector<IndexedVector3D> range(const _3DPoint &center, typename _3DPoint::coord_type radius) const override
    {
        return this->octTree->range(_Sphere<IndexedVector3D>(IndexedVector3D(center.x, center.y, center.z, ILLEGAL_IDX), radius));
    };
    inline size_t size() const override{return this->octTree->getSize();};

private:
    OctTree<IndexedVector3D> *octTree;
};

template<typename RandomAccessIterator>
OctTreeFinder::OctTreeFinder(RandomAccessIterator first, RandomAccessIterator last, const Vector3D &ll ,const Vector3D &ur)
{
    size_t index = 0;
    this->octTree = new OctTree<IndexedVector3D>(ll, ur);

    for(RandomAccessIterator it = first; it != last; it++)
    {
        const Vector3D &vec = *it;
        IndexedVector3D idx_vec = IndexedVector3D(vec.x, vec.y, vec.z, index);
        this->octTree->insert(idx_vec);
        index++;
    }
}

OctTreeFinder::~OctTreeFinder()
{
    delete this->octTree;
}

#endif // _OCT_TREE_FINDER_HPP