#ifdef RICH_MPI

#include "HilbertLoadBalancerIOHandler.hpp"
#include "LoadBalancerIOHandlerFactory.hpp"
#include "3D/tessellation/loadBalancing/HilbertLoadBalancer.hpp"
#include "3D/hilbert/io/ConvertorIOHandlerFactory.hpp"
#include "3D/environment/kernels/io/KernelIOHandlerFactory.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

void HilbertLoadBalancerIOHandler::dump(HDF5Writer &writer, const std::string &group, const LoadBalancer &lb) const
{
    const auto &hlb = static_cast<const HilbertLoadBalancer &>(lb);
    const auto &curve = static_cast<const CurveLoadBalancer &>(lb);
    writer.WriteElement(group + "/information", curve.boundaries);

    auto convertor = hlb.getConvertor();
    if (convertor)
    {
        ConvertorIO::writeConvertor(writer, group + "/convertor", *convertor);
    }

    auto indexing = hlb.getIndexing();
    if (indexing)
    {
        KernelIO::writeKernel(writer, group + "/indexing", *indexing);
    }
}

std::shared_ptr<LoadBalancer> HilbertLoadBalancerIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    std::vector<curve_index_t> boundaries;
    reader.ReadElement(group + "/information", boundaries);

    std::shared_ptr<HilbertConvertor3D> convertor;
    if (reader.Exists(group + "/convertor/type"))
    {
        convertor = ConvertorIO::readConvertor(reader, group + "/convertor");
    }

    std::shared_ptr<const Kernelization3D::IndexingKernel3D> indexing;
    if (reader.Exists(group + "/indexing/type"))
    {
        indexing = KernelIO::readKernel(reader, group + "/indexing");
    }
    else
    {
        indexing = std::make_shared<const Kernelization3D::Identity>();
    }

    return std::make_shared<HilbertLoadBalancer>(convertor, indexing, boundaries);
}

namespace
{
    static bool reg = (LoadBalancerIO::registerHandler("hilbert",
        std::make_unique<HilbertLoadBalancerIOHandler>()), true);
}

#endif // RICH_MPI
