#ifndef LINEAR_KERNEL_HPP
#define LINEAR_KERNEL_HPP

#include "3D/elementary/Mat33.hpp"
#include "IndexingKernel3D.hpp"

class Linear : public IndexingKernel3D
{
public:
    inline Linear(const Mat33<double> &transformation = Mat33<double>(), const IndexingKernel3D *indexing = nullptr): transformation(transformation), indexing(indexing){};

    inline Vector3D operator()(const Vector3D &vector) const override
    {
        Vector3D vec = (this->indexing == nullptr)? vector : (*this->indexing)(vector);
        return this->transformation * vec;
    };

private:
    Mat33<double> transformation;
    const IndexingKernel3D *indexing;
};

#endif // LINEAR_KERNEL_HPP