#include "LinearIOHandler.hpp"
#include "3D/environment/kernels/Linear.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "3D/output/matrixData.hpp"
#include "utils/hdf5/HDF5Helper.hpp"
#include "KernelIOHandlerFactory.hpp"

struct LinearState
{
    Mat33<double> transformation;
};

namespace HDF5Utils
{
    template<>
    struct HasCompType<LinearState> : std::true_type {};

    template<>
    struct CompTypeCreator<LinearState>
    {
        static hid_t get()
        {
            static hid_t type = []()
            {
                hid_t t = H5Tcreate(H5T_COMPOUND, sizeof(LinearState));
                H5Tinsert(t, "transformation", HOFFSET(LinearState, transformation),
                            HDF5Utils::CompTypeCreator<Mat33<double>>::get());
                return t;
            }();
            return type;
        }
    };
}

void LinearIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D &kernel) const
{
    const auto &linear = static_cast<const Kernelization3D::Linear &>(kernel);
    LinearState state{linear.transformation};
    writer.WriteElement(group + "/state", state);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D> LinearIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    LinearState state;
    reader.ReadElement(group + "/state", state);
    return std::make_shared<Kernelization3D::Linear>(state.transformation);
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Linear", std::make_unique<LinearIOHandler>()), true);
}
