#include "ParallelepipedIOHandler.hpp"
#include "3D/environment/kernels/Parallelepiped.hpp"
#include "KernelIOHandlerFactory.hpp"
#include "3D/output/matrixData.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "utils/hdf5/HDF5Helper.hpp"

struct ParallelepipedState
{
    Mat33<double> transformation;
};

namespace HDF5Utils
{
    template<>
    struct HasCompType<ParallelepipedState> : std::true_type {};

    template<>
    struct CompTypeCreator<ParallelepipedState>
    {
        static hid_t get()
        {
            static hid_t type = []()
            {
                hid_t t = H5Tcreate(H5T_COMPOUND, sizeof(ParallelepipedState));
                H5Tinsert(t, "transformation", HOFFSET(ParallelepipedState, transformation),
                            HDF5Utils::CompTypeCreator<Mat33<double>>::get());
                return t;
            }();
            return type;
        }
    };
}

void ParallelepipedIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D &kernel) const
{
    const auto &pp = static_cast<const Kernelization3D::Parallelepiped &>(kernel);
    ParallelepipedState state{pp.transformation};
    writer.WriteElement(group + "/state", state);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D> ParallelepipedIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    ParallelepipedState state;
    reader.ReadElement(group + "/state", state);
    auto kernel = std::make_shared<Kernelization3D::Parallelepiped>(Vector3D(1, 0, 0), Vector3D(0, 1, 0), Vector3D(0, 0, 1));
    kernel->transformation = state.transformation;
    return kernel;
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Parallelepiped", std::make_unique<ParallelepipedIOHandler>()), true);
}
