#ifndef HILBERT_CURVE_ENVAGENT_HPP
#define HILBERT_CURVE_ENVAGENT_HPP

#include "../CurveEnvAgent.hpp"
#include "3D/hilbert/HilbertConvertor3D.hpp"
#include "3D/tessellation/loadBalancing/HilbertLoadBalancer.hpp"

#ifdef RICH_MPI

class HilbertCurveEnvironmentAgent : public CurveEnvironmentAgent<hilbert_index_t, HilbertLoadBalancer>
{
public:
    using DistancesVector = std::vector<std::pair<typename Vector3D::coord_type, typename Vector3D::coord_type>>;

    inline HilbertCurveEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, const std::shared_ptr<HilbertLoadBalancer> loadBalancer, const MPI_Comm &comm = MPI_COMM_WORLD):
        CurveEnvironmentAgent<hilbert_index_t, HilbertLoadBalancer>(ll, ur, loadBalancer, comm)
    {};

    virtual ~HilbertCurveEnvironmentAgent() = default;

    virtual std::shared_ptr<HilbertCurveEnvironmentAgent> clone(const std::shared_ptr<HilbertLoadBalancer> newLoadBalancer) const = 0;

    virtual inline int getOwner(const Vector3D &point) const override
    {
        // TODO: that's wrong
        return this->getCellOwner(this->loadBalancer->convertor->xyz2d(point));
    };

    virtual void onExchange(const std::vector<Vector3D> &newPoints) override
    {
        this->CurveEnvironmentAgent::onExchange(newPoints);
    }

    virtual void onRebalance(void) override
    {
        this->CurveEnvironmentAgent::onRebalance();
    }

    inline int getOrder() const{return this->loadBalancer->convertor->getOrder();};
};

#endif // RICH_MPI

#endif // HILBERT_CURVE_ENVAGENT_HPP
