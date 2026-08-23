#ifndef ONE_DIMENSIONAL_LOAD_BALANCER_IO_HANDLER_HPP
#define ONE_DIMENSIONAL_LOAD_BALANCER_IO_HANDLER_HPP

#ifdef RICH_MPI

#include "LoadBalancerIOHandler.hpp"

class OneDimensionalLoadBalancerIOHandler : public LoadBalancerIOHandler
{
public:
    void dump(HDF5Writer &writer, const std::string &group, const LoadBalancer<Vector3D> &lb) const override;

    std::shared_ptr<LoadBalancer<Vector3D>> load(const HDF5Reader &reader, const std::string &group) const override;
};

#endif // RICH_MPI

#endif // ONE_DIMENSIONAL_LOAD_BALANCER_IO_HANDLER_HPP
