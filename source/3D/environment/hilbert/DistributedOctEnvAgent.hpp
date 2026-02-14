#ifndef _DIST_OCT_ENVIRONMENT_AGENT_HPP
#define _DIST_OCT_ENVIRONMENT_AGENT_HPP

#ifdef RICH_MPI

#include "HilbertCurveEnvAgent.hpp"
#include "ds/DistributedOctTree/DistributedOctTree.hpp"

#define RANKS_IN_LEAF 4

class DistributedOctEnvironmentAgent : public HilbertCurveEnvironmentAgent
{
public:
    using DistributedOctTree_Type = DistributedOctTree<Vector3D, RANKS_IN_LEAF>;

    inline DistributedOctEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, const std::vector<Vector3D> &points, const std::shared_ptr<HilbertLoadBalancer> &loadBalancer, const MPI_Comm &comm = MPI_COMM_WORLD): 
            HilbertCurveEnvironmentAgent(ll, ur, loadBalancer, comm)
    {
        OctTree<Vector3D> myTree(this->ll, this->ur, points);
        this->distributedOctTree = std::make_shared<DistributedOctTree_Type>(&myTree, false /* no detailed nodes info */, this->comm);
    };

    ~DistributedOctEnvironmentAgent() = default;

    inline EnvironmentAgent::RanksSet getIntersectingRanks(const Vector3D &center, double radius) const override
    {
        return this->distributedOctTree->getIntersectingRanks(center, radius);
    };

    inline void onExchange(const std::vector<Vector3D> &newPoints) override
    {
        this->HilbertCurveEnvironmentAgent::onExchange(newPoints);
        OctTree<Vector3D> myTree(this->ll, this->ur, newPoints);
        this->distributedOctTree = std::make_shared<DistributedOctTree_Type>(&myTree, false, this->comm);
    }

    inline int getOwner(const Vector3D &point) const override
    {
        return this->getCellOwner(this->loadBalancer->convertor->xyz2d((*this->loadBalancer->indexing)(point)));
    };

    inline void onRebalance(void) override
    {
        this->HilbertCurveEnvironmentAgent::onRebalance();
    }

    const std::shared_ptr<DistributedOctTree_Type> &getOctTree() const{return this->distributedOctTree;};
    
    template<typename U>
    inline HilbertCurveEnvironmentAgent::DistancesVector getClosestFurthestPointsByRanks(const U &point) const
    {
        return this->distributedOctTree->getClosestFurthestPointsByRanks(point);
    }

private:
    std::shared_ptr<DistributedOctTree_Type> distributedOctTree = nullptr;
};

#endif // RICH_MPI

#endif // _DIST_OCT_ENVIRONMENT_AGENT_HPP