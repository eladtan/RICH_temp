#ifndef _HILBERT_ENVIRONMENT_AGENT_HPP
#define _HILBERT_ENVIRONMENT_AGENT_HPP

#ifdef RICH_MPI

#include "EnvironmentAgent.h"
#include "3D/hilbert/rectangular/HilbertConvertor3D.hpp"
#include "ds/DistributedOctTree/DistributedOctTree.hpp"

#define AVERAGE_INTERSECT 128
#define NULL_ORDER -1

class HilbertEnvironmentAgent : public EnvironmentAgent
{
public:
    using CellsSet = boost::container::flat_set<hilbert_index_t>;

    inline HilbertEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, int order, const MPI_Comm &comm = MPI_COMM_WORLD):
            EnvironmentAgent(ll, ur, comm)
    {
        this->setOrder(order);
    };
    
    inline HilbertEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, const MPI_Comm &comm = MPI_COMM_WORLD): EnvironmentAgent(ll, ur, comm), order(NULL_ORDER){};
    
    RanksSet getIntersectingRanks(const Vector3D &center, double radius) const override;
    
    CellsSet getIntersectingCells(const Vector3D &center, double radius) const;
    
    inline int getOwner(const Vector3D &point) const override{return this->getCellOwner(this->convertor->xyz2d(point));};
    
    inline int getCellOwner(hilbert_index_t d) const
    {
        return std::min<int>(std::distance(this->range.begin(), std::upper_bound(this->range.begin(), this->range.end(), d)), this->size - 1);
    };

    inline int getOrder() const{return this->order;};

    inline void updateBorders(const std::vector<hilbert_index_t> &newRange, int newOrder)
    {
        this->range = newRange;
        this->setOrder(newOrder);
    }

private:
    Vector3D sidesLengths;
    int order;
    int rank, size;
    std::vector<hilbert_index_t> range;
    HilbertConvertor3D *convertor = nullptr;

    inline void setOrder(int order)
    {
        if(order == NULL_ORDER)
        {
            return;
        }
        this->order = std::min<int>(order, MAX_HILBERT_ORDER);
        this->sidesLengths = (this->ur - this->ll) / pow(2, this->order);
        this->convertor->changeOrder(this->order);
    }
};

#endif // RICH_MPI

#endif // _HILBERT_ENVIRONMENT_AGENT_HPP