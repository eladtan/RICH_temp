#include "LinearIOHandler.hpp"
#include <MeshDecomposer3D/kernels/Linear.hpp>
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "3D/output/matrixData.hpp"
#include "utils/hdf5/HDF5Helper.hpp"
#include "KernelIOHandlerFactory.hpp"

struct LinearState
{
    Kernelization3D::Mat33<double> transformation;
};

namespace HDF5Utils
{
    template<>
    struct HasCompType<LinearState> : std::true_type {};

    template<>
    struct CompTypeCreator<LinearState>
    {
        static H5::CompType get()
        {
            static H5::CompType type = []()
            {
                H5::CompType t(sizeof(LinearState));
                t.insertMember("transformation", HOFFSET(LinearState, transformation),
                               HDF5Utils::CompTypeCreator<Kernelization3D::Mat33<double>>::get());
                return t;
            }();
            return type;
        }
    };
}

void LinearIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const
{
    const auto &linear = static_cast<const Kernelization3D::Linear<Vector3D> &>(kernel);
    LinearState state{linear.getTransformation()};
    writer.WriteElement(group + "/state", state);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> LinearIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    LinearState state;
    reader.ReadElement(group + "/state", state);
    return std::make_shared<Kernelization3D::Linear<Vector3D>>(state.transformation);
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Linear", std::make_unique<LinearIOHandler>()), true);
}
