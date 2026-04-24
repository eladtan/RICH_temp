#include "RevedFrustrumIOHandler.hpp"
#include "3D/environment/kernels/RevedFrustrum.hpp"
#include "KernelIOHandlerFactory.hpp"
#include "3D/output/vectorData.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "utils/hdf5/HDF5Helper.hpp"

struct RevedFrustrumState
{
    Vector3D S;
    double h;
    double ratio;
};

namespace HDF5Utils
{
    template<>
    struct HasCompType<RevedFrustrumState> : std::true_type {};

    template<>
    struct CompTypeCreator<RevedFrustrumState>
    {
        static H5::CompType get()
        {
            static H5::CompType type = []()
            {
                H5::CompType t(sizeof(RevedFrustrumState));
                t.insertMember("S", HOFFSET(RevedFrustrumState, S), HDF5Utils::CompTypeCreator<Vector3D>::get());
                t.insertMember("h", HOFFSET(RevedFrustrumState, h), H5::PredType::NATIVE_DOUBLE);
                t.insertMember("ratio", HOFFSET(RevedFrustrumState, ratio), H5::PredType::NATIVE_DOUBLE);
                return t;
            }();
            return type;
        }
    };
}

void RevedFrustrumIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D &kernel) const
{
    const auto &reved = static_cast<const Kernelization3D::RevedFrustrum &>(kernel);
    RevedFrustrumState state{reved.S, reved.h, reved.ratio};
    writer.WriteElement(group + "/state", state);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D> RevedFrustrumIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    RevedFrustrumState state;
    reader.ReadElement(group + "/state", state);
    return std::shared_ptr<Kernelization3D::RevedFrustrum>(
        new Kernelization3D::RevedFrustrum(state.S, state.h, state.ratio));
}

namespace
{
    static bool reg = (KernelIO::registerHandler("RevedFrustrum", std::make_unique<RevedFrustrumIOHandler>()), true);
}
