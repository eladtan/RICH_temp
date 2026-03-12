#ifndef PHYSICS_STEP_IO_HANDLER_FACTORY_HPP
#define PHYSICS_STEP_IO_HANDLER_FACTORY_HPP

#include <memory>
#include <string>
#include <map>
#include "PhysicsStepIOHandler.hpp"
#include "misc/universal_error.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

namespace PhysicsStepIO
{
    inline std::map<std::string, std::unique_ptr<PhysicsStepIOHandler>> registry;

    inline void registerHandler(const std::string &name, std::unique_ptr<PhysicsStepIOHandler> handler)
    {
        registry[name] = std::move(handler);
    }

    inline void writeStep(HDF5Writer &writer, const std::string &prefix, const PhysicsStep &step)
    {
        const std::string name = step.getName();
        auto it = registry.find(name);
        if(it == registry.end())
        {
            throw UniversalError("PhysicsStepIO::writeStep: no handler for step \"" + name + "\"");
        }
        it->second->dump(writer, prefix + "/" + name, step);
    }

    inline void readStep(const HDF5Reader &reader, const std::string &prefix, PhysicsStep &step)
    {
        const std::string name = step.getName();
        const std::string group = prefix + "/" + name;
        if(!reader.Exists(group))
        {
            return;
        }
        auto it = registry.find(name);
        if(it == registry.end())
        {
            throw UniversalError("PhysicsStepIO::readStep: no handler for step \"" + name + "\"");
        }
        it->second->load(reader, group, step);
    }
}

#endif // PHYSICS_STEP_IO_HANDLER_FACTORY_HPP
