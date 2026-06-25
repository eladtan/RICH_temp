#ifndef LOAD_BALANCER_IO_HANDLER_HPP
#define LOAD_BALANCER_IO_HANDLER_HPP

#ifdef RICH_MPI

#include <memory>
#include <string>
#include "3D/elementary/Vector3D.hpp"
#include <MeshDecomposer3D/load_balancing/LoadBalancer.hpp>

class HDF5Writer;
class HDF5Reader;

class LoadBalancerIOHandler
{
public:
    virtual ~LoadBalancerIOHandler() = default;

    virtual void dump(HDF5Writer &writer, const std::string &group, const LoadBalancer<Vector3D> &lb) const = 0;

    virtual std::shared_ptr<LoadBalancer<Vector3D>> load(const HDF5Reader &reader, const std::string &group) const = 0;
};

#endif // RICH_MPI

#endif // LOAD_BALANCER_IO_HANDLER_HPP
