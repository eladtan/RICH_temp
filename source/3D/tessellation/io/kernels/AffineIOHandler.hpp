#ifndef AFFINE_IO_HANDLER_HPP
#define AFFINE_IO_HANDLER_HPP

#include <MeshDecomposer3D/kernels/Affine.hpp>
#include "KernelIOHandlerFactory.hpp"
#include "KernelIOHandler.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

class AffineIOHandler : public KernelIOHandler
{
public:
    void dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const override;

    std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> load(const HDF5Reader &reader, const std::string &group) const override;
};

#endif // AFFINE_IO_HANDLER_HPP
