#ifndef _HILBERT_TREE_ENVIRONMENT_AGENT_HPP
#define _HILBERT_TREE_ENVIRONMENT_AGENT_HPP

#ifdef RICH_MPI

#include "HilbertCurveEnvAgent.hpp"
#include "3D/hilbert/rectangular/HilbertRectangularTree3D.hpp"

class HilbertTreeEnvironmentAgent : public HilbertCurveEnvironmentAgent
{
public:
    using HilbertTree_Type = HilbertTree3D<DEFAULT_RANKS_IN_LEAVES>;

    inline HilbertTreeEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, const std::shared_ptr<HilbertLoadBalancer> loadBalancer, const MPI_Comm &comm = MPI_COMM_WORLD): 
            HilbertCurveEnvironmentAgent(ll, ur, loadBalancer, comm)
    {
        this->rectangularConvertor = dynamic_cast<HilbertRectangularConvertor3D*>(this->convertor);
        if(this->rectangularConvertor == nullptr)
        {
            throw UniversalError("'HilbertTreeEnvironmentAgent' should be initialized with a rectangular hilbert convertor");
        }

        this->hilbertTree = new HilbertTree_Type(this->rectangularConvertor, this->range, this->comm);
    };

    inline ~HilbertTreeEnvironmentAgent() override
    {};

    inline std::shared_ptr<HilbertCurveEnvironmentAgent> clone(const std::shared_ptr<HilbertLoadBalancer> newLoadBalancer) const override
    {
        return std::make_shared<HilbertTreeEnvironmentAgent>(this->ll, this->ur, newLoadBalancer, this->comm);
    }

    inline EnvironmentAgent::RanksSet getIntersectingRanks(const Vector3D &center, double radius) const override
    {
        return this->hilbertTree->getIntersectingRanks(center, radius);
    };

    inline void updatePoints(const std::vector<Vector3D> &newPoints) override
    {
        this->HilbertCurveEnvironmentAgent::updatePoints(newPoints);
    }
    
    template<typename U>
    inline HilbertCurveEnvironmentAgent::DistancesVector getClosestFurthestPointsByRanks(const U &point) const
    {
        return this->hilbertTree->getClosestFurthestPointsByRanks(point);
    }

    inline const HilbertTree_Type *getHilbertTree() const{return this->hilbertTree;};

private:
    const HilbertTree_Type *hilbertTree;
    HilbertRectangularConvertor3D *rectangularConvertor;
};

#endif // RICH_MPI

#endif // _HILBERT_TREE_ENVIRONMENT_AGENT_HPP