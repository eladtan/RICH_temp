#ifndef SCALE_KERNEL_HPP
#define SCALE_KERNEL_HPP

#include "IndexingKernel3D.hpp"

class ScaleIOHandler;

namespace Kernelization3D
{
    class Scale : public IndexingKernel3D
    {
    public:
        Scale(const Vector3D &scale = Vector3D(), const IndexingKernel3D *beforeIndexing = nullptr);

        Scale(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *beforeIndexing = nullptr);

        Vector3D operator()(const Vector3D &vector) const override;

        std::string getTypeName() const override;

    private:
        friend class ::ScaleIOHandler;
        Vector3D scale;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // SCALE_KERNEL_HPP