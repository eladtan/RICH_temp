#ifndef KERNEL_IO_HANDLER_FACTORY_HPP
#define KERNEL_IO_HANDLER_FACTORY_HPP

#include <memory>
#include <string>
#include <map>
#include "KernelIOHandler.hpp"
#include "misc/universal_error.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

namespace KernelIO
{
    inline std::map<std::string, std::unique_ptr<KernelIOHandler>> &getRegistry()
    {
        static std::map<std::string, std::unique_ptr<KernelIOHandler>> reg;
        return reg;
    }

    inline void registerHandler(const std::string &name, std::unique_ptr<KernelIOHandler> handler)
    {
        getRegistry()[name] = std::move(handler);
    }

    inline void writeKernel(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel)
    {
        const std::string &name = kernel.getTypeName();
        auto &reg = getRegistry();
        auto it = reg.find(name);
        if(it == reg.end())
        {
            throw UniversalError("KernelIO::writeKernel: no handler for kernel type \"" + name + "\"");
        }

        writer.WriteElement(group + "/type", name);
        it->second->dump(writer, group, kernel);
    }

    inline std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> readKernel(const HDF5Reader &reader, const std::string &group)
    {
        std::string name;
        reader.ReadElement(group + "/type", name);

        auto &reg = getRegistry();
        auto it = reg.find(name);
        if(it == reg.end())
        {
            throw UniversalError("KernelIO::readKernel: unknown kernel type \"" + name + "\"");
        }

        return it->second->load(reader, group);
    }
}

#endif // KERNEL_IO_HANDLER_FACTORY_HPP
