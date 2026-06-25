#include "ParallelepipedIOHandler.hpp"
#include <MeshDecomposer3D/kernels/Parallelepiped.hpp>
#include "KernelIOHandlerFactory.hpp"
#include "3D/output/matrixData.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "utils/hdf5/HDF5Helper.hpp"

struct ParallelepipedState
{
    Kernelization3D::Mat33<double> transformation;
};

namespace HDF5Utils
{
    template<>
    struct HasCompType<ParallelepipedState> : std::true_type {};

    template<>
    struct CompTypeCreator<ParallelepipedState>
    {
        static H5::CompType get()
        {
            static H5::CompType type = []()
            {
                H5::CompType t(sizeof(ParallelepipedState));
                t.insertMember("transformation", HOFFSET(ParallelepipedState, transformation),
                               HDF5Utils::CompTypeCreator<Kernelization3D::Mat33<double>>::get());
                return t;
            }();
            return type;
        }
    };
}

void ParallelepipedIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const
{
    const auto &pp = static_cast<const Kernelization3D::Parallelepiped<Vector3D> &>(kernel);
    ParallelepipedState state{pp.getTransformation()};
    writer.WriteElement(group + "/state", state);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> ParallelepipedIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    ParallelepipedState state;
    reader.ReadElement(group + "/state", state);
    return std::make_shared<Kernelization3D::Parallelepiped<Vector3D>>(
        Kernelization3D::Parallelepiped<Vector3D>::fromTransformation(state.transformation));
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Parallelepiped", std::make_unique<ParallelepipedIOHandler>()), true);
}
