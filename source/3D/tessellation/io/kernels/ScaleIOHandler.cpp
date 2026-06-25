#include "ScaleIOHandler.hpp"
#include <MeshDecomposer3D/kernels/Scale.hpp>
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "3D/output/vectorData.hpp"
#include "utils/hdf5/HDF5Helper.hpp"
#include "KernelIOHandlerFactory.hpp"

struct ScaleState
{
    Vector3D scale;
};

namespace HDF5Utils
{
    template<>
    struct HasCompType<ScaleState> : std::true_type {};

    template<>
    struct CompTypeCreator<ScaleState>
    {
        static H5::CompType get()
        {
            static H5::CompType type = []()
            {
                H5::CompType t(sizeof(ScaleState));
                t.insertMember("scale", HOFFSET(ScaleState, scale), HDF5Utils::CompTypeCreator<Vector3D>::get());
                return t;
            }();
            return type;
        }
    };
}

void ScaleIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const
{
    const auto &scale = static_cast<const Kernelization3D::Scale<Vector3D> &>(kernel);
    ScaleState state{scale.getScale()};
    writer.WriteElement(group + "/state", state);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> ScaleIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    ScaleState state;
    reader.ReadElement(group + "/state", state);
    return std::make_shared<Kernelization3D::Scale<Vector3D>>(state.scale);
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Scale", std::make_unique<ScaleIOHandler>()), true);
}
