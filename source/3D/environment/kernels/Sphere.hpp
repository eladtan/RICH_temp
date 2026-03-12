#ifndef SPHERICAL_KERNEL_HPP
#define SPHERICAL_KERNEL_HPP

#include <vector>
#include <algorithm>

#include "Move.hpp"
#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    class Sphere : public IndexingKernel3D
    {
    public:
        Sphere(const Vector3D &center, const IndexingKernel3D *beforeIndexing = nullptr);

        Vector3D operator()(const Vector3D &vector) const override;

        std::string getTypeName() const override;

    private:
        friend class SphereIOHandler;
        const IndexingKernel3D *beforeIndexing;
        Move moveIndexing;
    };
}

#endif // SPHERICAL_KERNEL_HPP