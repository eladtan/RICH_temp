#ifndef SHRINK_KERNEL_HPP
#define SHRINK_KERNEL_HPP

#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    class Shrink : public IndexingKernel3D
    {
    public:
        inline Shrink(const Vector3D &scale = Vector3D(), const IndexingKernel3D *beforeIndexing = nullptr): beforeIndexing(beforeIndexing)
        {
            this->scale = 1 / std::max(scale[0], std::max(scale[1], scale[2]));
        };

        inline Shrink(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *beforeIndexing = nullptr): Shrink(ur - ll, beforeIndexing){};

        inline Vector3D operator()(const Vector3D &vector) const override
        {
            Vector3D vec = (this->beforeIndexing == nullptr)? vector : (*this->beforeIndexing)(vector);
            return (vec * this->scale);
        };

    private:
        double scale;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // SHRINK_KERNEL_HPP