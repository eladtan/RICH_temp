#include "MoveIOHandler.hpp"
#include "3D/environment/kernels/Move.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "3D/output/vectorData.hpp"
#include "utils/hdf5/HDF5Helper.hpp"
#include "KernelIOHandlerFactory.hpp"

struct MoveState
{
    Vector3D moveVec;
};

namespace HDF5Utils
{
    template<>
    struct HasCompType<MoveState> : std::true_type {};

    template<>
    struct CompTypeCreator<MoveState>
    {
        static hid_t get()
        {
            static hid_t type = []()
            {
                hid_t t = H5Tcreate(H5T_COMPOUND, sizeof(MoveState));
                H5Tinsert(t, "moveVec", HOFFSET(MoveState, moveVec), HDF5Utils::CompTypeCreator<Vector3D>::get());
                return t;
            }();
            return type;
        }
    };
}

void MoveIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D &kernel) const
{
    const auto &move = static_cast<const Kernelization3D::Move &>(kernel);
    MoveState state{move.moveVec};
    writer.WriteElement(group + "/state", state);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D> MoveIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    MoveState state;
    reader.ReadElement(group + "/state", state);
    return std::make_shared<Kernelization3D::Move>(state.moveVec);
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Move", std::make_unique<MoveIOHandler>()), true);
}
