#include "ShrinkIOHandler.hpp"
#include <MeshDecomposer3D/kernels/Shrink.hpp>
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "utils/hdf5/HDF5Helper.hpp"
#include "KernelIOHandlerFactory.hpp"

struct ShrinkState
{
    double scale;
};

namespace HDF5Utils
{
    template<>
    struct HasCompType<ShrinkState> : std::true_type {};

    template<>
    struct CompTypeCreator<ShrinkState>
    {
        static H5::CompType get()
        {
            static H5::CompType type = []()
            {
                H5::CompType t(sizeof(ShrinkState));
                t.insertMember("scale", HOFFSET(ShrinkState, scale), H5::PredType::NATIVE_DOUBLE);
                return t;
            }();
            return type;
        }
    };
}

void ShrinkIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const
{
    const auto &shrink = static_cast<const Kernelization3D::Shrink<Vector3D> &>(kernel);
    ShrinkState state{shrink.getScale()};
    writer.WriteElement(group + "/state", state);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> ShrinkIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    ShrinkState state;
    reader.ReadElement(group + "/state", state);
    return std::make_shared<Kernelization3D::Shrink<Vector3D>>(
        Kernelization3D::Shrink<Vector3D>::fromStoredScale(state.scale));
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Shrink", std::make_unique<ShrinkIOHandler>()), true);
}
