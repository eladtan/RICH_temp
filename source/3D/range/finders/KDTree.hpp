#ifndef _KDTREE_FINDER_HPP
#define _KDTREE_FINDER_HPP

#include <spatial_ds/KDTree/KDTree.hpp>
#include "utils/IndexedVector.hpp"
#include "RangeFinder.hpp"

#define DIMENSIONS 3

class KDTreeFinder : public RangeFinder
{
public:
    template<typename RandomAccessIterator>
    KDTreeFinder(RandomAccessIterator first, RandomAccessIterator last, const Vector3D &ll ,const Vector3D &ur);
    
    inline KDTreeFinder(std::vector<Vector3D> &myPoints, const Vector3D &ll ,const Vector3D &ur): KDTreeFinder(myPoints.begin(), myPoints.end(), ll, ur){};
    
    inline ~KDTreeFinder() override{delete this->kdTree;};
    
    std::vector<size_t> closestPointInSphere(const Vector3D &center, double radius, const Vector3D &point, const _set<size_t> &ignore) const override
    {
        std::pair<IndexedVector3D, double> closestPointPair = this->kdTree->getClosestPointInSphere(Sphere<Vector3D>(center, radius), point,
                                                                                                    [&ignore](const IndexedVector3D &vec){return ignore.find(vec.getIndex()) == ignore.cend();});
        const IndexedVector3D &closestPoint = closestPointPair.first;
        const double &closestDistance = closestPointPair.second;

        if(closestDistance != std::numeric_limits<typename IndexedVector3D::coord_type>::max())
        {
            return std::vector<size_t>({closestPoint.index});
        }
        return std::vector<size_t>(); // empty
    }

    inline const Vector3D &getPoint(size_t index) const override{return this->myPoints[index];};

    inline std::vector<size_t> range(const Vector3D &center, double radius, size_t N, const _set<size_t> &ignore) const override
    {
        std::vector<size_t> toReturn;
        for(const IndexedVector3D &vec : this->kdTree->range(Sphere<Vector3D>(center, radius), N,
                                                            [&ignore](const IndexedVector3D &vec){return ignore.find(vec.getIndex()) == ignore.cend();}))
        {
            toReturn.push_back(vec.index);
        }
        return toReturn;
    };
    inline size_t size() const override{return this->kdTree->getSize();};

private:
    std::vector<Vector3D> myPoints;
    KDTree<IndexedVector3D, DIMENSIONS> *kdTree;
};

template<typename RandomAccessIterator>
inline KDTreeFinder::KDTreeFinder(RandomAccessIterator first, RandomAccessIterator last, const Vector3D &ll ,const Vector3D &ur)
{
    size_t index = 0;
    this->kdTree = new KDTree<IndexedVector3D, DIMENSIONS>(ll, ur);

    myPoints.reserve(last - first);
    for(RandomAccessIterator it = first; it != last; it++)
    {
        const Vector3D &vec = *it;
        IndexedVector3D idx_vec = IndexedVector3D(vec.x, vec.y, vec.z, index);
        this->myPoints.push_back(vec);
        this->kdTree->insert(idx_vec);
        index++;
    }
}

#endif // _KDTREE_FINDER_HPP