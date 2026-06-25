#include "FrustrumIOHandler.hpp"
#include <MeshDecomposer3D/kernels/Frustrum.hpp>
#include "KernelIOHandlerFactory.hpp"
#include "3D/output/matrixData.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "utils/hdf5/HDF5Helper.hpp"

struct FrustrumState
{
    Kernelization3D::Mat44<double> P;
};

namespace HDF5Utils
{
    template<>
    struct HasCompType<FrustrumState> : std::true_type {};

    template<>
    struct CompTypeCreator<FrustrumState>
    {
        static H5::CompType get()
        {
            static H5::CompType type = []()
            {
                H5::CompType t(sizeof(FrustrumState));
                t.insertMember("P", HOFFSET(FrustrumState, P),
                               HDF5Utils::CompTypeCreator<Kernelization3D::Mat44<double>>::get());
                return t;
            }();
            return type;
        }
    };
}

void FrustrumIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const
{
    const auto &frustrum = static_cast<const Kernelization3D::Frustrum<Vector3D> &>(kernel);
    FrustrumState state{frustrum.getP()};
    writer.WriteElement(group + "/state", state);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> FrustrumIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    FrustrumState state;
    reader.ReadElement(group + "/state", state);
    return std::shared_ptr<Kernelization3D::Frustrum<Vector3D>>(new Kernelization3D::Frustrum<Vector3D>(state.P));
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Frustrum", std::make_unique<FrustrumIOHandler>()), true);
}
