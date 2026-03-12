#ifndef GENERIC_SCALE_KERNEL_HPP
#define GENERIC_SCALE_KERNEL_HPP

#include <functional> // for std::function

#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    class GenericScale : public IndexingKernel3D
    {
    public:
        using ScaleFunction = std::function<double(double)>;

        GenericScale(const ScaleFunction &x, const ScaleFunction &y, const ScaleFunction &z, const IndexingKernel3D *beforeIndexing = nullptr);

        Vector3D operator()(const Vector3D &vector) const override;

        std::string getTypeName() const override;

    private:
        ScaleFunction x, y, z;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // GENERIC_SCALE_KERNEL_HPP