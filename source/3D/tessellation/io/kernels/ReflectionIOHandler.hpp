#ifndef REFLECTION_IO_HANDLER_HPP
#define REFLECTION_IO_HANDLER_HPP

#include "KernelIOHandler.hpp"

class ReflectionIOHandler : public KernelIOHandler
{
public:
    void dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D<Vector3D> &kernel) const override;

    std::shared_ptr<Kernelization3D::IndexingKernel3D<Vector3D>> load(const HDF5Reader &reader, const std::string &group) const override;
};

#endif // REFLECTION_IO_HANDLER_HPP
