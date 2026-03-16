#ifdef RICH_MPI

#include "HilbertPointsManagerIOHandler.hpp"
#include "PointsManagerIOHandlerFactory.hpp"
#include "3D/tessellation/voronoi/pointsManager/HilbertPointsManager.hpp"
#include "3D/hilbert/io/ConvertorIOHandlerFactory.hpp"
#include "3D/hilbert/io/RectangularConvertorIOHandler.hpp"
#include "3D/environment/kernels/io/KernelIOHandlerFactory.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

void HilbertPointsManagerIOHandler::dump(HDF5Writer &writer, const std::string &group, const PointsManager &pm) const
{
    const auto &hpm = static_cast<const HilbertPointsManager &>(pm);

    auto convertor = hpm.getConvertor();
    if(convertor)
    {
        ConvertorIO::writeConvertor(writer, group + "/convertor", *convertor);
    }

    auto indexing = hpm.getIndexing();
    if(indexing)
    {
        KernelIO::writeKernel(writer, group + "/indexing", *indexing);
    }
}

std::shared_ptr<PointsManager> HilbertPointsManagerIOHandler::load(const HDF5Reader &reader, const std::string &group, const Vector3D &ll, const Vector3D &ur) const
{
    std::shared_ptr<const Kernelization3D::IndexingKernel3D> indexing;
    if(reader.Exists(group + "/indexing/type"))
    {
        indexing = KernelIO::readKernel(reader, group + "/indexing");
    }

    auto hpm = std::make_shared<HilbertPointsManager>(ll, ur);
    hpm->setIndexing(indexing);

    if(reader.Exists(group + "/convertor/type"))
    {
        auto convertor = ConvertorIO::readConvertor(reader, group + "/convertor");
        hpm->setConvertor(std::move(convertor));
    }

    return hpm;
}

namespace
{
    static bool reg = (PointsManagerIO::registerHandler("hilbert",
        std::make_unique<HilbertPointsManagerIOHandler>()), true);
}

#endif // RICH_MPI
