#ifndef SHRINK_KERNEL_HPP
#define SHRINK_KERNEL_HPP

#include "IndexingKernel3D.hpp"

class Shrink : public IndexingKernel3D
{
public:
    inline Shrink(const Vector3D &scale, const IndexingKernel3D *indexing = nullptr): indexing(indexing)
    {
        this->scale = 1 / std::max(scale[0], std::max(scale[1], scale[2]));
    };

    inline Shrink(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *indexing = nullptr): Shrink(ur - ll, indexing){};

    inline Vector3D operator()(const Vector3D &vector) const override
    {
        Vector3D vec = (this->indexing == nullptr)? vector : (*this->indexing)(vector);
        return (vec * this->scale);
    };

private:
    double scale;
    const IndexingKernel3D *indexing;
};

#endif // SHRINK_KERNEL_HPP