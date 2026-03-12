#ifndef RECTANGULAR_KERNEL_HPP
#define RECTANGULAR_KERNEL_HPP

#include <vector>
#include <algorithm>

#include "Move.hpp"
#include "Scale.hpp"
#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    class Rectangle : public IndexingKernel3D
    {
    public:
        Rectangle(const std::vector<Vector3D> &vertices = std::vector<Vector3D>(), const IndexingKernel3D *beforeIndexing = nullptr);
        
        Rectangle(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *beforeIndexing = nullptr);

        Vector3D operator()(const Vector3D &vector) const override;

        std::string getTypeName() const override;

    private:
        friend class RectangleIOHandler;
        const IndexingKernel3D *beforeIndexing;
        Move moveIndexing;
        Scale scaleIndexing;
    };
}

#endif // RECTANGULAR_KERNEL_HPP