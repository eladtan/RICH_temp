#ifndef POINTS_MANAGER_IO_HANDLER_FACTORY_HPP
#define POINTS_MANAGER_IO_HANDLER_FACTORY_HPP

#ifdef RICH_MPI

#include <memory>
#include <string>
#include <map>
#include "PointsManagerIOHandler.hpp"
#include "misc/universal_error.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

namespace PointsManagerIO
{
    using PointsManager = ::PointsManager<Vector3D, MadVoro::VoronoiPayload<Vector3D>>;

    inline std::map<std::string, std::unique_ptr<PointsManagerIOHandler>> registry;

    inline void registerHandler(const std::string &name, std::unique_ptr<PointsManagerIOHandler> handler)
    {
        registry[name] = std::move(handler);
    }

    inline void writePointsManager(HDF5Writer &writer, const std::string &group, const PointsManager &pm)
    {
        const std::string name = pm.getTypeName();
        auto it = registry.find(name);
        if(it == registry.end())
        {
            throw UniversalError("PointsManagerIO::writePointsManager: no handler for type \"" + name + "\"");
        }

        writer.WriteElement(group + "/type", name);
        it->second->dump(writer, group, pm);
    }

    inline std::shared_ptr<PointsManager> readPointsManager(const HDF5Reader &reader, const std::string &group, const Vector3D &ll, const Vector3D &ur)
    {
        std::string name;
        reader.ReadElement(group + "/type", name);

        auto it = registry.find(name);
        if(it == registry.end())
        {
            throw UniversalError("PointsManagerIO::readPointsManager: unknown type \"" + name + "\"");
        }

        return it->second->load(reader, group, ll, ur);
    }
}

#endif // RICH_MPI

#endif // POINTS_MANAGER_IO_HANDLER_FACTORY_HPP
