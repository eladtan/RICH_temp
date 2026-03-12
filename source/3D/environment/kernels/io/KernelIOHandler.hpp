#ifndef KERNEL_IO_HANDLER_HPP
#define KERNEL_IO_HANDLER_HPP

#include <memory>
#include <string>
#include "3D/environment/kernels/IndexingKernel3D.hpp"

class HDF5Writer;
class HDF5Reader;

class KernelIOHandler
{
public:
    virtual ~KernelIOHandler() = default;

    virtual void dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D &kernel) const = 0;

    virtual std::shared_ptr<Kernelization3D::IndexingKernel3D> load(const HDF5Reader &reader, const std::string &group) const = 0;
};

#endif // KERNEL_IO_HANDLER_HPP
