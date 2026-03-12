#ifdef RICH_MPI

#include "HilbertLoadBalancerIOHandler.hpp"
#include "LoadBalancerIOHandlerFactory.hpp"
#include "3D/tessellation/loadBalancing/HilbertLoadBalancer.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

void HilbertLoadBalancerIOHandler::dump(HDF5Writer &writer, const std::string &group, const LoadBalancer &lb) const
{
    const auto &curve = static_cast<const CurveLoadBalancer &>(lb);
    writer.WriteElement(group + "/information", curve.boundaries);
}

std::shared_ptr<LoadBalancer> HilbertLoadBalancerIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    auto lb = std::make_shared<HilbertLoadBalancer>(nullptr, nullptr);
    reader.ReadElement(group + "/information", lb->boundaries);
    return lb;
}

namespace
{
    static bool reg = (LoadBalancerIO::registerHandler("hilbert",
        std::make_unique<HilbertLoadBalancerIOHandler>()), true);
}

#endif // RICH_MPI
