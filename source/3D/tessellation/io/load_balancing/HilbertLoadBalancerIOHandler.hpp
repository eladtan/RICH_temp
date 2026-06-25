#ifndef HILBERT_LOAD_BALANCER_IO_HANDLER_HPP
#define HILBERT_LOAD_BALANCER_IO_HANDLER_HPP

#ifdef RICH_MPI

#include "LoadBalancerIOHandler.hpp"

class HilbertLoadBalancerIOHandler : public LoadBalancerIOHandler
{
public:
    void dump(HDF5Writer &writer, const std::string &group, const LoadBalancer<Vector3D> &lb) const override;

    std::shared_ptr<LoadBalancer<Vector3D>> load(const HDF5Reader &reader, const std::string &group) const override;
};

#endif // RICH_MPI

#endif // HILBERT_LOAD_BALANCER_IO_HANDLER_HPP
