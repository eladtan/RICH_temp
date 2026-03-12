#ifndef LINEAR_KERNEL_HPP
#define LINEAR_KERNEL_HPP

#include "3D/elementary/Mat33.hpp"
#include "IndexingKernel3D.hpp"

class LinearIOHandler;

namespace Kernelization3D
{
    class Linear : public IndexingKernel3D
    {
    public:
        Linear(const Mat33<double> &transformation = Mat33<double>(), const IndexingKernel3D *beforeIndexing = nullptr);

        Vector3D operator()(const Vector3D &vector) const override;

        std::string getTypeName() const override;

    private:
        friend class ::LinearIOHandler;
        Mat33<double> transformation;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // LINEAR_KERNEL_HPP