#ifndef _DIST_OCT_ENVIRONMENT_AGENT_HPP
#define _DIST_OCT_ENVIRONMENT_AGENT_HPP

#ifdef RICH_MPI

#include "EnvironmentAgent.h"
#include "ds/DistributedOctTree/DistributedOctTree.hpp"

class DistributedOctEnvironmentAgent : public EnvironmentAgent
{
public:
    inline DistributedOctEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, const std::vector<Vector3D> &points, const std::vector<hilbert_index_t> &ranges, int order, const MPI_Comm &comm = MPI_COMM_WORLD): 
            range(ranges), EnvironmentAgent(ll, ur, comm)
    {
        OctTree<Vector3D> myTree(this->ll, this->ur, points);
        this->distributedOctTree = new DistributedOctTree(this->comm, &myTree);
        this->order = order;
    };

    inline ~DistributedOctEnvironmentAgent(){delete this->distributedOctTree;};

    inline EnvironmentAgent::_set<int> getIntersectingRanks(const Vector3D &center, double radius) const override
    {
        return this->distributedOctTree->getIntersectingRanks(center, radius);
    };
    inline int getOwner(const Vector3D &point) const override{return this->getCellOwner(this->xyz2d(point, this->order));};
    inline int getCellOwner(hilbert_index_t d) const
    {
        return std::min<int>(std::distance(this->range.begin(), std::upper_bound(this->range.begin(), this->range.end(), d)), this->size - 1);
    };
    inline void update(const std::vector<Vector3D> &newPoints) override
    {
        delete this->distributedOctTree;
        OctTree<Vector3D> myTree(this->ll, this->ur, newPoints);
        this->distributedOctTree = new DistributedOctTree(this->comm, &myTree);
    }

    inline void updateBorders(const std::vector<hilbert_index_t> &newRange, int newOrder) override
    {
        this->range = newRange;
        this->order = newOrder;
        return; // nothing to do
    }

    const DistributedOctTree<Vector3D> *getOctTree() const{return this->distributedOctTree;};

private:
    DistributedOctTree<Vector3D> *distributedOctTree = nullptr;
    std::vector<hilbert_index_t> range;
    int order;
};

#endif // RICH_MPI

#endif // _DIST_OCT_ENVIRONMENT_AGENT_HPP