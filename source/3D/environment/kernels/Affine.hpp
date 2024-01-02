#ifndef AFFINE_KERNEL_HPP
#define AFFINE_KERNEL_HPP

#include "Linear.hpp"
#include "Move.hpp"

namespace Kernelization3D
{
    class Affine : public IndexingKernel3D
    {
    public:
        inline Affine(const Linear &linear, const Vector3D &b = Vector3D(), const IndexingKernel3D *beforeIndexing = nullptr): linear(linear), move(b), beforeIndexing(beforeIndexing){};
        
        inline Affine(const Mat33<double> &A = Mat33<double>(), const Vector3D &b = Vector3D(), const IndexingKernel3D *beforeIndexing = nullptr): Affine(Linear(A), b, beforeIndexing){};

        inline Vector3D operator()(const Vector3D &vector) const override
        {
            Vector3D vec = (this->beforeIndexing == nullptr)? vector : (*this->beforeIndexing)(vector);
            return this->move(this->linear(vec));
        };

    private:
        Linear linear;
        Move move;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // AFFINE_KERNEL_HPP