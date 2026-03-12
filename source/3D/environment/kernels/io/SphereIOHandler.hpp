#ifndef SPHERE_IO_HANDLER_HPP
#define SPHERE_IO_HANDLER_HPP

#include "3D/environment/kernels/Sphere.hpp"
#include "3D/output/sim/KernelIOHandlerFactory.hpp"
#include "KernelIOHandler.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

class SphereIOHandler : public KernelIOHandler
{
public:
    void dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D &kernel) const override;

    std::shared_ptr<Kernelization3D::IndexingKernel3D> load(const HDF5Reader &reader, const std::string &group) const override;
};

#endif // SPHERE_IO_HANDLER_HPP
