#include "MoveIOHandler.hpp"
#include <MeshDecomposer3D/kernels/Move.hpp>
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
        static H5::CompType get()
        {
            static H5::CompType type = []()
            {
                H5::CompType t(sizeof(MoveState));
                t.insertMember("moveVec", HOFFSET(MoveState, moveVec), HDF5Utils::CompTypeCreator<Vector3D>::get());
                return t;
            }();
            return type;
        }
    };
}

void MoveIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const
{
    const auto &move = static_cast<const Kernelization3D::Move<Vector3D> &>(kernel);
    MoveState state{move.getMoveVec()};
    writer.WriteElement(group + "/state", state);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> MoveIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    MoveState state;
    reader.ReadElement(group + "/state", state);
    return std::make_shared<Kernelization3D::Move<Vector3D>>(state.moveVec);
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Move", std::make_unique<MoveIOHandler>()), true);
}
