#include "AffineIOHandler.hpp"
#include <MeshDecomposer3D/kernels/Affine.hpp>
#include "KernelIOHandlerFactory.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

void AffineIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const
{
    const auto &affine = static_cast<const Kernelization3D::Affine<Vector3D> &>(kernel);
    KernelIO::writeKernel(writer, group + "/linear", affine.getLinear());
    KernelIO::writeKernel(writer, group + "/move", affine.getMove());
}

std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> AffineIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    auto linear = KernelIO::readKernel(reader, group + "/linear");
    auto move = KernelIO::readKernel(reader, group + "/move");
    auto linearKernel = std::dynamic_pointer_cast<Kernelization3D::Linear<Vector3D>>(linear);
    auto moveKernel = std::dynamic_pointer_cast<Kernelization3D::Move<Vector3D>>(move);
    return std::make_shared<Kernelization3D::Affine<Vector3D>>(*linearKernel, moveKernel->getMoveVec());
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Affine", std::make_unique<AffineIOHandler>()), true);
}
