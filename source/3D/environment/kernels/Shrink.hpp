#ifndef SHRINK_KERNEL_HPP
#define SHRINK_KERNEL_HPP

#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    class Shrink : public IndexingKernel3D
    {
    public:
        Shrink(const Vector3D &scale = Vector3D(), const IndexingKernel3D *beforeIndexing = nullptr);

        Shrink(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *beforeIndexing = nullptr);

        Vector3D operator()(const Vector3D &vector) const override;

        std::string getTypeName() const override;

    private:
        friend class ShrinkIOHandler;
        double scale;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // SHRINK_KERNEL_HPP