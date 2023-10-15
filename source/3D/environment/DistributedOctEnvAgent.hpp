#ifndef _DIST_OCT_ENVIRONMENT_AGENT_HPP
#define _DIST_OCT_ENVIRONMENT_AGENT_HPP

#ifdef RICH_MPI

#include "EnvironmentAgent.h"
#include "3D/hilbert/HilbertConvertor.hpp"
#include "ds/DistributedOctTree/DistributedOctTree.hpp"

class DistributedOctEnvironmentAgent : public EnvironmentAgent
{
public:
    inline DistributedOctEnvironmentAgent(const IndexingKernel3D *indexing, const Vector3D &ll, const Vector3D &ur, const std::vector<Vector3D> &points, const std::vector<hilbert_index_t> &ranges, int order, const MPI_Comm &comm = MPI_COMM_WORLD): 
            indexing(indexing), range(ranges), EnvironmentAgent(ll, ur, comm)
    {
        OctTree<Vector3D> myTree(this->ll, this->ur, points);
        this->distributedOctTree = new DistributedOctTree<Vector3D>(&myTree, false, this->comm);
        this->order = order;
    };

    inline ~DistributedOctEnvironmentAgent(){delete this->distributedOctTree;};

    inline EnvironmentAgent::RanksSet getIntersectingRanks(const Vector3D &center, double radius) const override
    {
        return this->distributedOctTree->getIntersectingRanks(center, radius);
    };

    inline int getOwner(const Vector3D &point) const override{return this->getCellOwner(Hilbert3DConvertor::xyz2d((*this->indexing)(point), this->order));};

    inline int getCellOwner(hilbert_index_t d) const
    {
        return std::min<int>(std::distance(this->range.begin(), std::upper_bound(this->range.begin(), this->range.end(), d)), this->size - 1);
    };

    inline void updatePoints(const std::vector<Vector3D> &newPoints)
    {
        delete this->distributedOctTree;
        OctTree<Vector3D> myTree(this->ll, this->ur, newPoints);
        this->distributedOctTree = new DistributedOctTree(&myTree, false, this->comm);
    }

    inline void updateBorders(const std::vector<hilbert_index_t> &newRange, int newOrder)
    {
        this->range = newRange;
        this->order = newOrder;
        return; // nothing else to do
    }

    const DistributedOctTree<Vector3D> *getOctTree() const{return this->distributedOctTree;};

private:
    DistributedOctTree<Vector3D> *distributedOctTree = nullptr;
    const IndexingKernel3D *indexing = nullptr;
    std::vector<hilbert_index_t> range;
    int order;
};

#endif // RICH_MPI

#endif // _DIST_OCT_ENVIRONMENT_AGENT_HPP