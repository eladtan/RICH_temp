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

    inline DistributedOctEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, const std::vector<Vector3D> &points, const std::vector<hilbert_index_t> &ranges, HilbertConvertor3D *convertor, const Kernelization3D::IndexingKernel3D *indexing, const MPI_Comm &comm = MPI_COMM_WORLD): 
            HilbertCurveEnvironmentAgent(ll, ur, ranges, convertor, comm), indexing(indexing)
    {
        OctTree<Vector3D> myTree(this->ll, this->ur, points);
        this->distributedOctTree = new DistributedOctTree_Type(&myTree, false /* no detailed nodes info */, this->comm);
    };

    inline ~DistributedOctEnvironmentAgent(){delete this->distributedOctTree;};

    inline EnvironmentAgent::RanksSet getIntersectingRanks(const Vector3D &center, double radius) const override
    {
        return this->distributedOctTree->getIntersectingRanks(center, radius);
    };

    inline void updatePoints(const std::vector<Vector3D> &newPoints) override
    {
        this->HilbertCurveEnvironmentAgent::updatePoints(newPoints);
        delete this->distributedOctTree;
        OctTree<Vector3D> myTree(this->ll, this->ur, newPoints);
        this->distributedOctTree = new DistributedOctTree<Vector3D, RANKS_IN_LEAF>(&myTree, false, this->comm);
    }

    inline int getOwner(const Vector3D &point) const override
    {
        return this->getCellOwner(this->convertor->xyz2d((*this->indexing)(point)));
    };

    inline void updateBorders(const std::vector<hilbert_index_t> &newRange, int newOrder) override
    {
        this->HilbertCurveEnvironmentAgent::updateBorders(newRange, newOrder);
    }

    const DistributedOctTree_Type *getOctTree() const{return this->distributedOctTree;};

    inline int getOrder() const{return this->order;};
    
    template<typename U>
    inline HilbertCurveEnvironmentAgent::DistancesVector getClosestFurthestPointsByRanks(const U &point) const
    {
        return this->distributedOctTree->getClosestFurthestPointsByRanks(point);
    }

private:
    DistributedOctTree_Type *distributedOctTree = nullptr;
    HilbertConvertor3D *convertor = nullptr;
    const Kernelization3D::IndexingKernel3D *indexing = nullptr;
    std::vector<hilbert_index_t> range;
    int order;
};

#endif // RICH_MPI

#endif // _DIST_OCT_ENVIRONMENT_AGENT_HPP