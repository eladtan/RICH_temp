#ifdef RICH_MPI

#include "HilbertPointsManagerIOHandler.hpp"
#include "PointsManagerIOHandlerFactory.hpp"
#include <MeshDecomposer3D/points_manager/HilbertPointsManager.hpp>
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

void HilbertPointsManagerIOHandler::dump(HDF5Writer &, const std::string &, const PointsManager &) const
{}

std::shared_ptr<HilbertPointsManagerIOHandler::PointsManager> HilbertPointsManagerIOHandler::load(const HDF5Reader &, const std::string &, const Vector3D &ll, const Vector3D &ur) const
{
    return std::make_shared<HilbertPointsManager<Vector3D, MadVoro::VoronoiPayload<Vector3D>>>(ll, ur);
}

namespace
{
    static bool reg = (PointsManagerIO::registerHandler("hilbert",
        std::make_unique<HilbertPointsManagerIOHandler>()), true);
}

#endif // RICH_MPI
