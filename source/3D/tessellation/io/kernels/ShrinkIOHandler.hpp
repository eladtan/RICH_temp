#ifndef SHRINK_IO_HANDLER_HPP
#define SHRINK_IO_HANDLER_HPP

#include "KernelIOHandler.hpp"

class ShrinkIOHandler : public KernelIOHandler
{
public:
    void dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const override;

    std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> load(const HDF5Reader &reader, const std::string &group) const override;
};

#endif // SHRINK_IO_HANDLER_HPP
