#ifndef _HILBERT_TREE_ENVIRONMENT_AGENT_HPP
#define _HILBERT_TREE_ENVIRONMENT_AGENT_HPP

#ifdef RICH_MPI

#include "HilbertCurveEnvAgent.hpp"
#include "3D/hilbert/rectangular/HilbertTree3D.hpp"

#define MAX_RANKS_PER_LEAF 1

class HilbertTreeEnvironmentAgent : public HilbertCurveEnvironmentAgent
{
public:
    inline HilbertTreeEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, const std::vector<Vector3D> &points, const std::vector<hilbert_index_t> &ranges, HilbertConvertor3D *convertor, const MPI_Comm &comm = MPI_COMM_WORLD): 
            HilbertCurveEnvironmentAgent(ll, ur, ranges, convertor, comm)
    {
        this->hilbertTree = new HilbertTree3D<MAX_RANKS_PER_LEAF>(this->convertor, this->range, this->comm);
    };

    inline ~HilbertTreeEnvironmentAgent() override
    {
        delete this->hilbertTree;
    };

    inline EnvironmentAgent::RanksSet getIntersectingRanks(const Vector3D &center, double radius) const override
    {
        return this->hilbertTree->getIntersectingRanks(center, radius);
    };

    inline void updatePoints(const std::vector<Vector3D> &newPoints) override
    {
        this->HilbertCurveEnvironmentAgent::updatePoints(newPoints);
    }
    
    inline void updateBorders(const std::vector<hilbert_index_t> &newRange, int newOrder) override
    {
        this->HilbertCurveEnvironmentAgent::updateBorders(newRange, newOrder);
        delete this->hilbertTree;
        this->hilbertTree = new HilbertTree3D<MAX_RANKS_PER_LEAF>(this->convertor, this->range, this->comm);
    }

    inline int getOrder() const{return this->order;};
    
    template<typename U>
    inline HilbertCurveEnvironmentAgent::DistancesVector getClosestFurthestPointsByRanks(const U &point) const
    {
        return this->hilbertTree->getClosestFurthestPointsByRanks(point);
    }

private:
    const HilbertTree3D<MAX_RANKS_PER_LEAF> *hilbertTree;
};

#endif // RICH_MPI

#endif // _HILBERT_TREE_ENVIRONMENT_AGENT_HPP