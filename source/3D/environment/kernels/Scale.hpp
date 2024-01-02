#ifndef SCALE_KERNEL_HPP
#define SCALE_KERNEL_HPP

#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    class Scale : public IndexingKernel3D
    {
    public:
        inline Scale(const Vector3D &scale = Vector3D(), const IndexingKernel3D *beforeIndexing = nullptr): scale(scale), beforeIndexing(beforeIndexing){};

        inline Scale(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *beforeIndexing = nullptr): Scale(ur - ll, beforeIndexing){};

        inline Vector3D operator()(const Vector3D &vector) const override
        {
            Vector3D vec = (this->beforeIndexing == nullptr)? vector : (*this->beforeIndexing)(vector);
            return Vector3D(vec.x / this->scale.x, vec.y / this->scale.y, vec.z / this->scale.z);
        };

    private:
        Vector3D scale;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // SCALE_KERNEL_HPP