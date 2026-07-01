#ifndef POINTS_MANAGER_IO_HANDLER_HPP
#define POINTS_MANAGER_IO_HANDLER_HPP

#ifdef RICH_MPI

#include <memory>
#include <string>
#include <MeshDecomposer3D/points_manager/PointsManager.hpp>
#include "3D/elementary/Vector3D.hpp"
#include "3D/tessellation/voronoi/VoronoiPayload.hpp"

class HDF5Writer;
class HDF5Reader;

class PointsManagerIOHandler
{
public:
    using PointsManager = PointsManager<Vector3D,MadVoro::VoronoiPayload<Vector3D>>;

    virtual ~PointsManagerIOHandler() = default;

    virtual void dump(HDF5Writer &writer, const std::string &group, const PointsManager &pm) const = 0;

    virtual std::shared_ptr<PointsManager> load(const HDF5Reader &reader, const std::string &group, const Vector3D &ll, const Vector3D &ur) const = 0;
};

#endif // RICH_MPI

#endif // POINTS_MANAGER_IO_HANDLER_HPP
