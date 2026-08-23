#ifdef RICH_MPI

#include "OneDimensionalLoadBalancerIOHandler.hpp"
#include "LoadBalancerIOHandlerFactory.hpp"
#include <MeshDecomposer3D/load_balancing/OneDimensionalLoadBalancer.hpp>
#include "3D/output/vectorData.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

void OneDimensionalLoadBalancerIOHandler::dump(HDF5Writer &writer, const std::string &group, const LoadBalancer<Vector3D> &lb) const
{
    const OneDimensionalLoadBalancer<Vector3D> &oned = static_cast<const OneDimensionalLoadBalancer<Vector3D> &>(lb);
    const int axis = static_cast<int>(oned.GetAxis());
    writer.WriteElement(group + "/axis", axis);
    writer.WriteElement(group + "/ll", oned.GetLowerLeft());
    writer.WriteElement(group + "/ur", oned.GetUpperRight());
    writer.WriteElement(group + "/bins", oned.GetBins());
}

std::shared_ptr<LoadBalancer<Vector3D>> OneDimensionalLoadBalancerIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    int axisInt = 0;
    reader.ReadElement(group + "/axis", axisInt);
    Vector3D ll;
    Vector3D ur;
    reader.ReadElement(group + "/ll", ll);
    reader.ReadElement(group + "/ur", ur);
    std::vector<double> bins;
    if(reader.Exists(group + "/bins"))
    {
        reader.ReadElement(group + "/bins", bins);
    }
    Axis axis = static_cast<Axis>(axisInt);
    return std::make_shared<OneDimensionalLoadBalancer<Vector3D>>(ll, ur, axis, bins);
}

namespace
{
    static bool reg = (LoadBalancerIO::registerHandler("1d", std::make_unique<OneDimensionalLoadBalancerIOHandler>()), true);
}

#endif // RICH_MPI
