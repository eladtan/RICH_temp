#include "ReflectionIOHandler.hpp"
#include <MeshDecomposer3D/kernels/Reflection.hpp>
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "3D/output/vectorData.hpp"
#include "utils/hdf5/HDF5Helper.hpp"
#include "KernelIOHandlerFactory.hpp"

struct ReflectionState
{
    Vector3D reflectionVector;
    Vector3D factoredVec;
};

namespace HDF5Utils
{
    template<>
    struct HasCompType<ReflectionState> : std::true_type {};

    template<>
    struct CompTypeCreator<ReflectionState>
    {
        static H5::CompType get()
        {
            static H5::CompType type = []()
            {
                H5::CompType t(sizeof(ReflectionState));
                t.insertMember("reflectionVector", HOFFSET(ReflectionState, reflectionVector),
                               HDF5Utils::CompTypeCreator<Vector3D>::get());
                t.insertMember("factoredVec", HOFFSET(ReflectionState, factoredVec),
                               HDF5Utils::CompTypeCreator<Vector3D>::get());
                return t;
            }();
            return type;
        }
    };
}

void ReflectionIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const
{
    const auto &reflection = static_cast<const Kernelization3D::Reflection<Vector3D> &>(kernel);
    ReflectionState state{reflection.getReflectionVector(), reflection.getFactoredVec()};
    writer.WriteElement(group + "/state", state);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> ReflectionIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    ReflectionState state;
    reader.ReadElement(group + "/state", state);
    return std::make_shared<Kernelization3D::Reflection<Vector3D>>(state.reflectionVector, state.factoredVec);
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Reflection", std::make_unique<ReflectionIOHandler>()), true);
}
