#include "RotationIOHandler.hpp"
#include "3D/environment/kernels/Rotation.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "3D/output/matrixData.hpp"
#include "utils/hdf5/HDF5Helper.hpp"
#include "KernelIOHandlerFactory.hpp"

struct RotationState
{
    Mat33<double> mat;
};

namespace HDF5Utils
{
    template<>
    struct HasCompType<RotationState> : std::true_type {};

    template<>
    struct CompTypeCreator<RotationState>
    {
        static H5::CompType get()
        {
            static H5::CompType type = []()
            {
                H5::CompType t(sizeof(RotationState));
                t.insertMember("mat", HOFFSET(RotationState, mat),
                               HDF5Utils::CompTypeCreator<Mat33<double>>::get());
                return t;
            }();
            return type;
        }
    };
}

void RotationIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D &kernel) const
{
    const auto &rotation = static_cast<const Kernelization3D::Rotation &>(kernel);
    RotationState state{rotation.mat};
    writer.WriteElement(group + "/state", state);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D> RotationIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    RotationState state;
    reader.ReadElement(group + "/state", state);
    auto kernel = std::make_shared<Kernelization3D::Rotation>(0.0, Vector3D(0, 0, 1));
    kernel->mat = state.mat;
    return kernel;
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Rotation", std::make_unique<RotationIOHandler>()), true);
}
