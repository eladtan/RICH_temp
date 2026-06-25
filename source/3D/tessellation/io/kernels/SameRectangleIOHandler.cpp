#include "SameRectangleIOHandler.hpp"
#include <MeshDecomposer3D/kernels/SameRectangle.hpp>
#include "KernelIOHandlerFactory.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

void SameRectangleIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const
{
    const auto &rect = static_cast<const Kernelization3D::SameRectangle<Vector3D> &>(kernel);
    KernelIO::writeKernel(writer, group + "/moveIndexing", rect.getMoveIndexing());
    KernelIO::writeKernel(writer, group + "/scaleIndexing", rect.getScaleIndexing());
}

std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> SameRectangleIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    auto movePtr = KernelIO::readKernel(reader, group + "/moveIndexing");
    auto scalePtr = KernelIO::readKernel(reader, group + "/scaleIndexing");
    auto moveKernel = std::dynamic_pointer_cast<Kernelization3D::Move<Vector3D>>(movePtr);
    auto scaleKernel = std::dynamic_pointer_cast<Kernelization3D::Scale<Vector3D>>(scalePtr);
    return std::make_shared<Kernelization3D::SameRectangle<Vector3D>>(*moveKernel, *scaleKernel);
}

namespace
{
    static bool reg = (KernelIO::registerHandler("SameRectangle", std::make_unique<SameRectangleIOHandler>()), true);
}
