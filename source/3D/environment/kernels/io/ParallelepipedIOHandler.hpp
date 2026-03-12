#ifndef PARALLELEPIPED_IO_HANDLER_HPP
#define PARALLELEPIPED_IO_HANDLER_HPP

#include "KernelIOHandler.hpp"
#include "3D/environment/kernels/Parallelepiped.hpp"

class ParallelepipedIOHandler : public KernelIOHandler
{
public:
    void dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D &kernel) const override;

    std::shared_ptr<Kernelization3D::IndexingKernel3D> load(const HDF5Reader &reader, const std::string &group) const override;
};

#endif // PARALLELEPIPED_IO_HANDLER_HPP
