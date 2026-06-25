#ifndef FRUSTRUM_IO_HANDLER_HPP
#define FRUSTRUM_IO_HANDLER_HPP

#include "KernelIOHandler.hpp"
#include <MeshDecomposer3D/kernels/Frustrum.hpp>

class FrustrumIOHandler : public KernelIOHandler
{
public:
    void dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const override;

    std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> load(const HDF5Reader &reader, const std::string &group) const override;
};

#endif // FRUSTRUM_IO_HANDLER_HPP
