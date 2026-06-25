#include "RectangleShrinkIOHandler.hpp"
#include <MeshDecomposer3D/kernels/RectangleShrink.hpp>
#include "KernelIOHandlerFactory.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

void RectangleShrinkIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const
{
    const auto &rect = static_cast<const Kernelization3D::RectangleShrink<Vector3D> &>(kernel);
    KernelIO::writeKernel(writer, group + "/moveIndexing", rect.getMoveIndexing());
    KernelIO::writeKernel(writer, group + "/shrinkIndexing", rect.getShrinkIndexing());
}

std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> RectangleShrinkIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    auto movePtr = KernelIO::readKernel(reader, group + "/moveIndexing");
    auto shrinkPtr = KernelIO::readKernel(reader, group + "/shrinkIndexing");
    auto moveKernel = std::dynamic_pointer_cast<Kernelization3D::Move<Vector3D>>(movePtr);
    auto shrinkKernel = std::dynamic_pointer_cast<Kernelization3D::Shrink<Vector3D>>(shrinkPtr);
    return std::make_shared<Kernelization3D::RectangleShrink<Vector3D>>(*moveKernel, *shrinkKernel);
}

namespace
{
    static bool reg = (KernelIO::registerHandler("RectangleShrink", std::make_unique<RectangleShrinkIOHandler>()), true);
}
