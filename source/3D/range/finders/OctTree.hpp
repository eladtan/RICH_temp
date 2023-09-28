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
    inline ~OctTreeFinder()
    {
        delete this->octTree;
    }

    std::vector<size_t> closestPointInSphere(const Vector3D &center, double radius, const Vector3D &point, const _set<size_t> &ignore) const override
    {
        std::vector<size_t> toReturn;
        IndexedVector3D closestPoint;
        typename IndexedVector3D::coord_type closestDistance = std::numeric_limits<typename IndexedVector3D::coord_type>::max();
        this->getClosestPointHelper(_Sphere<IndexedVector3D>(IndexedVector3D(center.x, center.y, center.z, ILLEGAL_IDX), radius),
                                    point, this->octTree->getRoot(), closestPoint, closestDistance, ignore);
        if(closestDistance != std::numeric_limits<typename IndexedVector3D::coord_type>::max())
        {
            toReturn.push_back(closestPoint.index);
        }
        return toReturn;
    };

    inline std::vector<size_t> range(const Vector3D &center, double radius) const override
    {
        std::vector<size_t> toReturn;
        for(const IndexedVector3D &vec : this->octTree->range(_Sphere<IndexedVector3D>(IndexedVector3D(center.x, center.y, center.z, ILLEGAL_IDX), radius)))
        {
            toReturn.push_back(vec.index);
        }
        return toReturn;
    };
    inline size_t size() const override{return this->octTree->getSize();};

    inline const Vector3D &getPoint(size_t index) const override{return this->myPoints[index];};

    inline Vector3D closestPoint(const Vector3D &point) const
    {
        _3DPoint res = this->octTree->closestPoint(IndexedVector3D(point, ILLEGAL_IDX)).getData();
        return Vector3D(res.x, res.y, res.z);
    }

    inline double closestPointDistance(const Vector3D &point) const
    {
        return this->octTree->closestPointDistance(IndexedVector3D(point, ILLEGAL_IDX));
    }

private:
    // void getClosestPointHelper(const _Sphere<IndexedVector3D> &sphere, const IndexedVector3D &point, const typename OctTree<IndexedVector3D>::OctTreeNode *node, IndexedVector3D &closestPoint, typename IndexedVector3D::coord_type &closestDistance, const std::vector<size_t> &ignoreValues) const;
    void getClosestPointHelper(const _Sphere<IndexedVector3D> &sphere, const IndexedVector3D &point, const typename OctTree<IndexedVector3D>::OctTreeNode *node, IndexedVector3D &closestPoint,
                                typename IndexedVector3D::coord_type &closestDistance, const _set<size_t> &ignore) const
    {
        if(node == nullptr)
        {
            return;
        }
        IndexedVector3D closestPointInBox = node->boundingBox.closestPoint(point);
        // calculate distance squared
        typename IndexedVector3D::coord_type dist = 0;
        for(int i = 0; i < DIM; i++)
        {
            dist += (closestPointInBox[i] - point[i]) * (closestPointInBox[i] - point[i]);
        }
        if((dist >= closestDistance) or (!SphereBoxIntersection(node->boundingBox, sphere)))
        {
            return;
        }
        // there might be a closer point in the subtrees
        if(node->isValue)
        {
            //if(std::find(ignore.begin(), ignore.end(), node->value.index) == ignore.end())
            if(ignore.find(node->value.index) == ignore.cend())
            {
                // should not be ignored
                closestPoint = node->value;
                closestDistance = dist;
            }
        }
        else
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                this->getClosestPointHelper(sphere, point, node->children[i], closestPoint, closestDistance, ignore);
            }
        }
    }

    std::vector<Vector3D> myPoints;
    OctTree<IndexedVector3D> *octTree;
};

template<typename RandomAccessIterator>
OctTreeFinder::OctTreeFinder(RandomAccessIterator first, RandomAccessIterator last, const Vector3D &ll ,const Vector3D &ur)
{
    size_t index = 0;
    this->octTree = new OctTree<IndexedVector3D>(ll, ur);
    
    myPoints.reserve(last - first);
    for(RandomAccessIterator it = first; it != last; it++)
    {
        const Vector3D &vec = *it;
        this->myPoints.push_back(vec);
        IndexedVector3D idx_vec = IndexedVector3D(vec.x, vec.y, vec.z, index);
        this->octTree->insert(idx_vec);
        index++;
    }
}

/*
void OctTreeFinder::getClosestPointHelper(const _Sphere<IndexedVector3D> &sphere, const IndexedVector3D &point, const typename OctTree<IndexedVector3D>::OctTreeNode *node, IndexedVector3D &closestPoint, typename IndexedVector3D::coord_type &closestDistance, const std::vector<size_t> &ignore) const
{
    if(node == nullptr)
    {
        return;
    }
    IndexedVector3D closestPointInBox = node->boundingBox.closestPoint(point);
    // calculate distance squared
    typename IndexedVector3D::coord_type dist = 0;
    for(int i = 0; i < DIM; i++)
    {
        dist += (closestPointInBox[i] - point[i]) * (closestPointInBox[i] - point[i]);
    }
    if((dist >= closestDistance) or (!SphereBoxIntersection(node->boundingBox, sphere)))
    {
        return;
    }
    // there might be a closer point in the subtrees
    if(node->isValue)
    {
        if(std::find(ignore.begin(), ignore.end(), node->value.index) == ignore.end())
        {
            // should not be ignored
            closestPoint = node->value;
            closestDistance = dist;
        }
    }
    else
    {
        for(int i = 0; i < CHILDREN; i++)
        {
            this->getClosestPointHelper(sphere, point, node->children[i], closestPoint, closestDistance, ignore);
        }
    }
}
*/

#endif // _OCT_TREE_FINDER_HPP