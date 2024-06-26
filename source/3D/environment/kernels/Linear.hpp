#ifndef LINEAR_KERNEL_HPP
#define LINEAR_KERNEL_HPP

#include "3D/elementary/Mat33.hpp"
#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    class Linear : public IndexingKernel3D
    {
    public:
        inline Linear(const Mat33<double> &transformation = Mat33<double>(), const IndexingKernel3D *beforeIndexing = nullptr): transformation(transformation), beforeIndexing(beforeIndexing){};

        inline Vector3D operator()(const Vector3D &vector) const override
        {
            Vector3D vec = (this->beforeIndexing == nullptr)? vector : (*this->beforeIndexing)(vector);
            return this->transformation * vec;
        };

    private:
        Mat33<double> transformation;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // LINEAR_KERNEL_HPP