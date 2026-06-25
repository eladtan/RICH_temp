#ifndef CONVERTOR_IO_HANDLER_FACTORY_HPP
#define CONVERTOR_IO_HANDLER_FACTORY_HPP

#include <memory>
#include <string>
#include <map>
#include "ConvertorIOHandler.hpp"
#include "misc/universal_error.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

namespace ConvertorIO
{
    inline std::map<std::string, std::unique_ptr<ConvertorIOHandler>> registry;

    inline void registerHandler(const std::string &name, std::unique_ptr<ConvertorIOHandler> handler)
    {
        registry[name] = std::move(handler);
    }

    inline void writeConvertor(HDF5Writer &writer, const std::string &group, const HilbertConvertor3D<Vector3D> &convertor)
    {
        const std::string name = convertor.getTypeName();
        auto it = registry.find(name);
        if(it == registry.end())
        {
            throw UniversalError("ConvertorIO::writeConvertor: no handler for type \"" + name + "\"");
        }

        writer.WriteElement(group + "/type", name);
        it->second->dump(writer, group, convertor);
    }

    inline std::shared_ptr<HilbertConvertor3D<Vector3D>> readConvertor(const HDF5Reader &reader, const std::string &group)
    {
        std::string name;
        reader.ReadElement(group + "/type", name);

        auto it = registry.find(name);
        if(it == registry.end())
        {
            throw UniversalError("ConvertorIO::readConvertor: unknown type \"" + name + "\"");
        }

        return it->second->load(reader, group);
    }
}

#endif // CONVERTOR_IO_HANDLER_FACTORY_HPP
