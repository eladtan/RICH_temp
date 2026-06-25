#ifndef LOAD_BALANCER_IO_HANDLER_FACTORY_HPP
#define LOAD_BALANCER_IO_HANDLER_FACTORY_HPP

#ifdef RICH_MPI

#include <memory>
#include <string>
#include <map>
#include "LoadBalancerIOHandler.hpp"
#include "misc/universal_error.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

namespace LoadBalancerIO
{
    inline std::map<std::string, std::unique_ptr<LoadBalancerIOHandler>> &getRegistry()
    {
        static std::map<std::string, std::unique_ptr<LoadBalancerIOHandler>> reg;
        return reg;
    }

    inline void registerHandler(const std::string &name, std::unique_ptr<LoadBalancerIOHandler> handler)
    {
        getRegistry()[name] = std::move(handler);
    }

    inline void writeLoadBalancer(HDF5Writer &writer, const std::string &group, const LoadBalancer<Vector3D> &lb)
    {
        const std::string name = lb.getTypeName();
        auto &reg = getRegistry();
        auto it = reg.find(name);
        if(it == reg.end())
        {
            throw UniversalError("LoadBalancerIO::writeLoadBalancer: no handler for type \"" + name + "\"");
        }

        writer.WriteElement(group + "/type", name);
        it->second->dump(writer, group, lb);
    }

    inline std::shared_ptr<LoadBalancer<Vector3D>> readLoadBalancer(const HDF5Reader &reader, const std::string &group)
    {
        std::string name;
        reader.ReadElement(group + "/type", name);

        auto &reg = getRegistry();
        auto it = reg.find(name);
        if(it == reg.end())
        {
            throw UniversalError("LoadBalancerIO::readLoadBalancer: unknown type \"" + name + "\"");
        }

        return it->second->load(reader, group);
    }
}

#endif // RICH_MPI

#endif // LOAD_BALANCER_IO_HANDLER_FACTORY_HPP
