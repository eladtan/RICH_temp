#include "IdentityIOHandler.hpp"
#include "3D/environment/kernels/Identity.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "KernelIOHandlerFactory.hpp"

void IdentityIOHandler::dump(HDF5Writer &, const std::string &, const Kernelization3D::IndexingKernel3D &) const
{
}

std::shared_ptr<Kernelization3D::IndexingKernel3D> IdentityIOHandler::load(const HDF5Reader &, const std::string &) const
{
    return std::make_shared<Kernelization3D::Identity>();
}

namespace
{
    static bool reg = (KernelIO::registerHandler("Identity", std::make_unique<IdentityIOHandler>()), true);
}
