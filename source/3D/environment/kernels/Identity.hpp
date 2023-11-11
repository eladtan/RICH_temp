#ifndef IDENTITY_KERNEL_HPP
#define IDENTITY_KERNEL_HPP

#include "IndexingKernel3D.hpp"

class Identity : public IndexingKernel3D
{
public:
    inline Identity(const IndexingKernel3D *indexing = nullptr): indexing(indexing){};

    inline Vector3D operator()(const Vector3D &vector) const override
    {
        Vector3D vec = (this->indexing == nullptr)? vector : (*this->indexing)(vector);
        return vec;
    };

private:
    const IndexingKernel3D *indexing;
};

#endif // IDENTITY_KERNEL_HPP