#ifndef AFFINE_KERNEL_HPP
#define AFFINE_KERNEL_HPP

#include "Linear.hpp"
#include "Move.hpp"

class AffineIOHandler;

namespace Kernelization3D
{
    class Affine : public IndexingKernel3D
    {
    public:
        Affine(const Linear &linear, const Vector3D &b = Vector3D(), const IndexingKernel3D *beforeIndexing = nullptr);
        
        Affine(const Mat33<double> &A = Mat33<double>(), const Vector3D &b = Vector3D(), const IndexingKernel3D *beforeIndexing = nullptr);

        Vector3D operator()(const Vector3D &vector) const override;

        std::string getTypeName() const override;

    private:
        friend class ::AffineIOHandler;
        Linear linear;
        Move move;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // AFFINE_KERNEL_HPP