#ifndef CURVE_ENVIRONMENT_AGENT
#define CURVE_ENVIRONMENT_AGENT

#include <vector>
#include "3D/tessellation/loadBalancing/CurveLoadBalancer.hpp"
#include "EnvironmentAgent.h"

#ifdef RICH_MPI

template<typename curve_index_t = size_t, typename LoadBalancerType = CurveLoadBalancer>
class CurveEnvironmentAgent : public EnvironmentAgent
{
    // static assert that LoadBalancerType inherits CurveLoadBalancer
    static_assert(std::is_base_of<CurveLoadBalancer, LoadBalancerType>::value, "LoadBalancerType must inherit from CurveLoadBalancer");

public:
    inline CurveEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, const std::shared_ptr<LoadBalancerType> curveLoadBalancer, const MPI_Comm &comm = MPI_COMM_WORLD):
        EnvironmentAgent(ll, ur, comm), loadBalancer(curveLoadBalancer)
    {}

    virtual ~CurveEnvironmentAgent() = default;

    virtual void setLoadBalancer(std::shared_ptr<LoadBalancerType> newLoadBalancer)
    {
        this->loadBalancer = newLoadBalancer;
        this->onRebalance();
    }

    virtual inline int getCellOwner(curve_index_t d) const
    {
        size_t index = static_cast<size_t>(std::distance(this->loadBalancer->boundaries.cbegin(), std::upper_bound(this->loadBalancer->boundaries.cbegin(), this->loadBalancer->boundaries.cend(), d)));
        return std::min<size_t>(index, this->size - 1);
    };

    virtual void onExchange(const std::vector<Vector3D> &newPoints)
    {}

    virtual void onRebalance(void)
    {}

protected:
    std::shared_ptr<LoadBalancerType> loadBalancer;
};

#endif // RICH_MPI

#endif // CURVE_ENVIRONMENT_AGENT