#ifndef RECTANGLE_SHRINK_IO_HANDLER_HPP
#define RECTANGLE_SHRINK_IO_HANDLER_HPP

#include "3D/environment/kernels/RectangleShrink.hpp"
#include "3D/output/sim/KernelIOHandlerFactory.hpp"
#include "KernelIOHandler.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

class RectangleShrinkIOHandler : public KernelIOHandler
{
public:
    void dump(HDF5Writer &writer, const std::string &group, const Kernelization3D::IndexingKernel3D &kernel) const override;

    std::shared_ptr<Kernelization3D::IndexingKernel3D> load(const HDF5Reader &reader, const std::string &group) const override;
};

#endif // RECTANGLE_SHRINK_IO_HANDLER_HPP
