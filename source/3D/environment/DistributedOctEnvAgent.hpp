#ifndef _DIST_OCT_ENVIRONMENT_AGENT_HPP
#define _DIST_OCT_ENVIRONMENT_AGENT_HPP

#ifdef RICH_MPI

#include "EnvironmentAgent.h"
#include "3D/hilbert/rectangular/HilbertConvertor3D.hpp"
#include "ds/DistributedOctTree/DistributedOctTree.hpp"

class DistributedOctEnvironmentAgent : public EnvironmentAgent
{
public:
    inline DistributedOctEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, const std::vector<Vector3D> &points, const std::vector<hilbert_index_t> &ranges, HilbertConvertor3D *convertor, const IndexingKernel3D *indexing, const MPI_Comm &comm = MPI_COMM_WORLD): 
            range(ranges), convertor(convertor), indexing(indexing), EnvironmentAgent(ll, ur, comm)
    {
        this->order = this->convertor->getOrder();
        OctTree<Vector3D> myTree(this->ll, this->ur, points);
        this->distributedOctTree = new DistributedOctTree<Vector3D>(&myTree, false /* no detailed nodes info */, this->comm);
    };

    inline ~DistributedOctEnvironmentAgent(){delete this->distributedOctTree;};

    inline EnvironmentAgent::RanksSet getIntersectingRanks(const Vector3D &center, double radius) const override
    {
        return this->distributedOctTree->getIntersectingRanks(center, radius);
    };

    inline int getOwner(const Vector3D &point) const override{return this->getCellOwner(this->convertor->xyz2d((*this->indexing)(point)));};

    inline int getCellOwner(hilbert_index_t d) const
    {
        int index = static_cast<int>(std::distance(this->range.begin(), std::upper_bound(this->range.begin(), this->range.end(), d)));
        return std::min<int>(index, this->size - 1);
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
        if(this->convertor != nullptr)
        {
            this->convertor->changeOrder(newOrder);
        }
        return; // nothing else to do
    }

    const DistributedOctTree<Vector3D> *getOctTree() const{return this->distributedOctTree;};

    inline int getOrder() const{return this->order;};
    
private:
    DistributedOctTree<Vector3D> *distributedOctTree = nullptr;
    HilbertConvertor3D *convertor = nullptr;
    const IndexingKernel3D *indexing = nullptr;
    std::vector<hilbert_index_t> range;
    int order;
};

#endif // RICH_MPI

#endif // _DIST_OCT_ENVIRONMENT_AGENT_HPP