#include "SphereIOHandler.hpp"
#include <MeshDecomposer3D/kernels/Sphere.hpp>
#include "KernelIOHandlerFactory.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

void SphereIOHandler::dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const
{
    const auto &sphere = static_cast<const Kernelization3D::Sphere<Vector3D> &>(kernel);
    KernelIO::writeKernel(writer, group + "/moveIndexing", sphere.getMoveIndexing());
}

std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> SphereIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    auto movePtr = KernelIO::readKernel(reader, group + "/moveIndexing");
    auto moveKernel = std::dynamic_pointer_cast<Kernelization3D::Move<Vector3D>>(movePtr);
    return std::make_shared<Kernelization3D::Sphere<Vector3D>>(moveKernel->getMoveVec());
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Sphere", std::make_unique<SphereIOHandler>()), true);
}
