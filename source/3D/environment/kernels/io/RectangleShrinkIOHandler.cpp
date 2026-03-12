#include "RectangleShrinkIOHandler.hpp"
#include "3D/environment/kernels/RectangleShrink.hpp"
#include "KernelIOHandlerFactory.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

void RectangleShrinkIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D &kernel) const
{
    const auto &rect = static_cast<const Kernelization3D::RectangleShrink &>(kernel);
    KernelIO::writeKernel(writer, group + "/moveIndexing", rect.moveIndexing);
    KernelIO::writeKernel(writer, group + "/shrinkIndexing", rect.shrinkIndexing);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D> RectangleShrinkIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    auto movePtr = KernelIO::readKernel(reader, group + "/moveIndexing");
    auto shrinkPtr = KernelIO::readKernel(reader, group + "/shrinkIndexing");
    auto moveKernel = std::dynamic_pointer_cast<Kernelization3D::Move>(movePtr);
    auto shrinkKernel = std::dynamic_pointer_cast<Kernelization3D::Shrink>(shrinkPtr);
    auto kernel = std::make_shared<Kernelization3D::RectangleShrink>(std::vector<Vector3D>());
    kernel->moveIndexing = *moveKernel;
    kernel->shrinkIndexing = *shrinkKernel;
    return kernel;
}

namespace
{
    static bool reg = (KernelIO::registerHandler("RectangleShrink", std::make_unique<RectangleShrinkIOHandler>()), true);
}
