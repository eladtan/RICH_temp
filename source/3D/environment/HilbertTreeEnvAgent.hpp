#ifndef _HILBERT_TREE_ENVIRONMENT_AGENT_HPP
#define _HILBERT_TREE_ENVIRONMENT_AGENT_HPP

#ifdef RICH_MPI

#include "EnvironmentAgent.h"
#include "3D/hilbert/rectangular/HilbertConvertor3D.hpp"
#include "3D/hilbert/rectangular/HilbertTree3D.hpp"

#define MAX_RANKS_PER_LEAF 1

class HilbertTreeEnvironmentAgent : public EnvironmentAgent
{
public:
    inline HilbertTreeEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, const std::vector<Vector3D> &points, const std::vector<hilbert_index_t> &ranges, HilbertConvertor3D *convertor, const IndexingKernel3D *indexing, const MPI_Comm &comm = MPI_COMM_WORLD): 
            range(ranges), convertor(convertor), indexing(indexing), EnvironmentAgent(ll, ur, comm)
    {
        this->order = this->convertor->getOrder();
        this->hilbertTree = new HilbertTree3D<MAX_RANKS_PER_LEAF>(this->convertor, this->range, this->comm);
    };

    inline ~HilbertTreeEnvironmentAgent(){delete this->hilbertTree;};

    inline EnvironmentAgent::RanksSet getIntersectingRanks(const Vector3D &center, double radius) const override
    {
        throw UniversalError("When using kernelization, the results are not accurate (since the tree represents what happens after kernelization, but the sphere does not)");
        return this->hilbertTree->getIntersectingRanks(center, radius);
    };

    inline int getOwner(const Vector3D &point) const override
    {
        return this->getCellOwner(this->convertor->xyz2d((*this->indexing)(point)));
    };

    inline int getCellOwner(hilbert_index_t d) const
    {
        int index = static_cast<int>(std::distance(this->range.begin(), std::upper_bound(this->range.begin(), this->range.end(), d)));
        return std::min<int>(index, this->size - 1);
    };

    inline void updatePoints(const std::vector<Vector3D> &newPoints)
    {
        return; // nothing to do
    }
    
    inline void updateBorders(const std::vector<hilbert_index_t> &newRange, int newOrder)
    {
        this->range = newRange;
        if(this->convertor != nullptr)
        {
            this->convertor->changeOrder(newOrder);
        }

        delete this->hilbertTree;
        this->hilbertTree = new HilbertTree3D<MAX_RANKS_PER_LEAF>(this->convertor, this->range, this->comm);
        
        return; // nothing else to do
    }

    inline int getOrder() const{return this->order;};
    
    template<typename U>
    inline std::vector<std::pair<typename Vector3D::coord_type, typename Vector3D::coord_type>> getClosestFurthestPointsByRanks(const U &point) const
    {
        throw UniversalError("When using kernelization, the results are not accurate (since the tree represents what happens after kernelization, but the sphere does not)");
        return this->hilbertTree->getClosestFurthestPointsByRanks(point);
    }

private:
    const HilbertTree3D<MAX_RANKS_PER_LEAF> *hilbertTree;
    HilbertConvertor3D *convertor = nullptr;
    const IndexingKernel3D *indexing = nullptr;
    std::vector<hilbert_index_t> range;
    int order;
};

#endif // RICH_MPI

#endif // _HILBERT_TREE_ENVIRONMENT_AGENT_HPP