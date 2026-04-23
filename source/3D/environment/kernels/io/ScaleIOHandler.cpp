#include "ScaleIOHandler.hpp"
#include "3D/environment/kernels/Scale.hpp"
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
        static hid_t get()
        {
            static hid_t type = []()
            {
                hid_t t = H5Tcreate(H5T_COMPOUND, sizeof(ScaleState));
                H5Tinsert(t, "scale", HOFFSET(ScaleState, scale), HDF5Utils::CompTypeCreator<Vector3D>::get());
                return t;
            }();
            return type;
        }
    };
}

void ScaleIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D &kernel) const
{
    const auto &scale = static_cast<const Kernelization3D::Scale &>(kernel);
    ScaleState state{scale.scale};
    writer.WriteElement(group + "/state", state);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D> ScaleIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    ScaleState state;
    reader.ReadElement(group + "/state", state);
    return std::make_shared<Kernelization3D::Scale>(state.scale);
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Scale", std::make_unique<ScaleIOHandler>()), true);
}
