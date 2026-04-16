#ifdef RICH_MPI

#include "HilbertPointsManagerIOHandler.hpp"
#include "PointsManagerIOHandlerFactory.hpp"
#include "3D/tessellation/voronoi/pointsManager/HilbertPointsManager.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

void HilbertPointsManagerIOHandler::dump(HDF5Writer &, const std::string &, const PointsManager &) const
{
}

std::shared_ptr<PointsManager> HilbertPointsManagerIOHandler::load(const HDF5Reader &, const std::string &, const Vector3D &ll, const Vector3D &ur) const
{
    return std::make_shared<HilbertPointsManager>(ll, ur);
}

namespace
{
    static bool reg = (PointsManagerIO::registerHandler("hilbert",
        std::make_unique<HilbertPointsManagerIOHandler>()), true);
}

#endif // RICH_MPI
