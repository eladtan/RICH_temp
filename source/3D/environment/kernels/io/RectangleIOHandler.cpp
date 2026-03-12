#include "RectangleIOHandler.hpp"
#include "3D/environment/kernels/Rectangle.hpp"
#include "KernelIOHandlerFactory.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

void RectangleIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D &kernel) const
{
    const auto &rect = static_cast<const Kernelization3D::Rectangle &>(kernel);
    KernelIO::writeKernel(writer, group + "/moveIndexing", rect.moveIndexing);
    KernelIO::writeKernel(writer, group + "/scaleIndexing", rect.scaleIndexing);
}

std::shared_ptr<Kernelization3D::IndexingKernel3D> RectangleIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    auto movePtr = KernelIO::readKernel(reader, group + "/moveIndexing");
    auto scalePtr = KernelIO::readKernel(reader, group + "/scaleIndexing");
    auto moveKernel = std::dynamic_pointer_cast<Kernelization3D::Move>(movePtr);
    auto scaleKernel = std::dynamic_pointer_cast<Kernelization3D::Scale>(scalePtr);
    auto kernel = std::make_shared<Kernelization3D::Rectangle>(std::vector<Vector3D>());
    kernel->moveIndexing = *moveKernel;
    kernel->scaleIndexing = *scaleKernel;
    return kernel;
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Rectangle", std::make_unique<RectangleIOHandler>()), true);
}
