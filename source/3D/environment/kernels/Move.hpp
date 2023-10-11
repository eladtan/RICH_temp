#ifndef MOVE_KERNEL_HPP
#define MOVE_KERNEL_HPP

#include "IndexingKernel3D.hpp"

class Move : public IndexingKernel3D
{
public:
    inline Move(const Vector3D &vector, const IndexingKernel3D *indexing = nullptr): moveVec(vector), indexing(indexing){};

    inline Vector3D operator()(const Vector3D &vector) const override
    {
        Vector3D vec = (this->indexing == nullptr)? vector : (*this->indexing)(vector);
        return vec - moveVec;
    };

private:
    Vector3D moveVec;
    const IndexingKernel3D *indexing;
};

#endif // MOVE_KERNEL_HPP