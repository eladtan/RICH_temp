#ifndef _OCT_TREE_FINDER_HPP
#define _OCT_TREE_FINDER_HPP

#include <spatial_ds/OctTree/OctTree.hpp>
#include "utils/IndexedVector.hpp"
#include "RangeFinder.hpp"

#define DIMENSIONS 3

class OctTreeFinder : public RangeFinder
{
public:
    template<typename RandomAccessIterator>
    OctTreeFinder(RandomAccessIterator first, RandomAccessIterator last, const Vector3D &ll ,const Vector3D &ur);
    
    inline OctTreeFinder(std::vector<Vector3D> &myPoints, const Vector3D &ll ,const Vector3D &ur): OctTreeFinder(myPoints.begin(), myPoints.end(), ll, ur){};
    
    inline OctTreeFinder(OctTree<IndexedVector3D> *tree, const std::vector<Vector3D> &myPoints): octTree(tree), myPoints(myPoints), givenOctTree(true){}

    inline ~OctTreeFinder() override
    {
        if(not this->givenOctTree)
        {
            delete this->octTree;
        }
    };

    std::vector<size_t> closestPointInSphere(const Vector3D &center, double radius, const Vector3D &point, const _set<size_t> &ignore) const override
    {
        std::pair<IndexedVector3D, double> closestPointPair = this->octTree->getClosestPointInSphere(Sphere<Vector3D>(center, radius), point,
                                                                                                    [&ignore](const IndexedVector3D &vec){return ignore.find(vec.getIndex()) == ignore.cend();});
        const IndexedVector3D &closestPoint = closestPointPair.first;
        const double &closestDistance = closestPointPair.second;

        if(closestDistance != std::numeric_limits<typename IndexedVector3D::coord_type>::max())
        {
            return std::vector<size_t>({closestPoint.index});
        }
        return std::vector<size_t>(); // empty
    };

    inline std::vector<size_t> range(const Vector3D &center, double radius, size_t N, const _set<size_t> &ignore) const override
    {
        std::vector<size_t> toReturn;
        for(const IndexedVector3D &vec : this->octTree->range(Sphere<IndexedVector3D>(IndexedVector3D(center.x, center.y, center.z, ILLEGAL_IDX), radius + EPSILON), N,
                                                              [&ignore](const IndexedVector3D &vec){return ignore.find(vec.getIndex()) == ignore.cend();}))
        {
            toReturn.push_back(vec.index);
        }
        return toReturn;
    };

    inline size_t size() const override{return this->octTree->getSize();};

    inline const Vector3D &getPoint(size_t index) const override{return this->myPoints[index];};

private:

    std::vector<Vector3D> myPoints;
    OctTree<IndexedVector3D> *octTree;
    bool givenOctTree;
};

template<typename RandomAccessIterator>
inline OctTreeFinder::OctTreeFinder(RandomAccessIterator first, RandomAccessIterator last, const Vector3D &ll ,const Vector3D &ur)
{
    this->octTree = new OctTree<IndexedVector3D>(ll, ur);
    this->givenOctTree = false;
    this->myPoints.reserve(last - first);
    size_t index = 0;
    for(RandomAccessIterator it = first; it != last; it++)
    {
        const Vector3D &vec = *it;
        this->myPoints.push_back(vec);
        IndexedVector3D idx_vec = IndexedVector3D(vec.x, vec.y, vec.z, index);
        this->octTree->insert(idx_vec);
        index++;
    }
}

#endif // _OCT_TREE_FINDER_HPP