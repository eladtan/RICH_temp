#ifndef RECTANGULAR_SHRINK_KERNEL_HPP
#define RECTANGULAR_SHRINK_KERNEL_HPP

#include <vector>
#include <algorithm>

#include "Move.hpp"
#include "Shrink.hpp"
#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    class RectangleShrink : public IndexingKernel3D
    {
    public:
        RectangleShrink(const std::vector<Vector3D> &vertices = std::vector<Vector3D>(), const IndexingKernel3D *beforeIndexing = nullptr);
        
        RectangleShrink(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *beforeIndexing = nullptr);

        Vector3D operator()(const Vector3D &vector) const override;

        std::string getTypeName() const override;

    private:
        friend class RectangleShrinkIOHandler;
        const IndexingKernel3D *beforeIndexing;
        Move moveIndexing;
        Shrink shrinkIndexing;
    };
}

#endif // RECTANGULAR_SHRINK_KERNEL_HPP