#ifndef SCALE_IO_HANDLER_HPP
#define SCALE_IO_HANDLER_HPP

#include "KernelIOHandler.hpp"

class ScaleIOHandler : public KernelIOHandler
{
public:
    void dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const override;

    std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> load(const HDF5Reader &reader, const std::string &group) const override;
};

#endif // SCALE_IO_HANDLER_HPP
