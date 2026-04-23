#include "ShrinkIOHandler.hpp"
#include "3D/environment/kernels/Shrink.hpp"
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
        static hid_t get()
        {
            static hid_t type = []()
            {
                hid_t t = H5Tcreate(H5T_COMPOUND, sizeof(ShrinkState));
                H5Tinsert(t, "scale", HOFFSET(ShrinkState, scale), H5T_NATIVE_DOUBLE);
                return t;
            }();
            return type;
        }
    };
}

void ShrinkIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D &kernel) const
{
    const auto &shrink = static_cast<const Kernelization3D::Shrink &>(kernel);
    ShrinkState state{shrink.scale};
    writer.WriteElement(group + "/state", state);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D> ShrinkIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    ShrinkState state;
    reader.ReadElement(group + "/state", state);
    auto kernel = std::make_shared<Kernelization3D::Shrink>(Vector3D(1, 1, 1));
    kernel->scale = state.scale;
    return kernel;
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Shrink", std::make_unique<ShrinkIOHandler>()), true);
}
