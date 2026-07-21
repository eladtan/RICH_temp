#ifndef PLAIN_DIST_OCT_ENVIRONMENT_AGENT_HPP
#define PLAIN_DIST_OCT_ENVIRONMENT_AGENT_HPP

#ifdef RICH_MPI

#include "EnvironmentAgent.h"
#include "ds/DistributedOctTree/DistributedOctTree.hpp"

#define RANKS_IN_LEAF 1

class PlainDistributedOctEnvironmentAgent : public EnvironmentAgent
{
public:
    using DistributedOctTree_Type = DistributedOctTree<Vector3D, RANKS_IN_LEAF>;

    inline PlainDistributedOctEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, const std::vector<Vector3D> &points, const MPI_Comm &comm = MPI_COMM_WORLD): 
            EnvironmentAgent(ll, ur, comm)
    {
        OctTree<Vector3D> myTree(this->ll, this->ur, points);
        this->distributedOctTree = new DistributedOctTree_Type(&myTree, false /* no detailed nodes info */, this->comm);
    };

    inline ~PlainDistributedOctEnvironmentAgent(){delete this->distributedOctTree;};

    inline EnvironmentAgent::RanksSet getIntersectingRanks(const Vector3D &center, double radius) const override
    {
        return this->distributedOctTree->getIntersectingRanks(center, radius);
    };

    inline void updatePoints(const std::vector<Vector3D> &newPoints) override
    {
        delete this->distributedOctTree;
        OctTree<Vector3D> myTree(this->ll, this->ur, newPoints);
        this->distributedOctTree = new DistributedOctTree_Type(&myTree, false, this->comm);
    }

    inline int getOwner(const Vector3D &point) const override
    {
        return this->distributedOctTree->GetRanksOfPoint(point)[0];
    };

    const DistributedOctTree_Type *getOctTree() const{return this->distributedOctTree;};

    inline int getOrder() const{return this->order;};

private:
    DistributedOctTree_Type *distributedOctTree = nullptr;
    int order;
};

#endif // RICH_MPI

#endif // PLAIN_DIST_OCT_ENVIRONMENT_AGENT_HPP