#ifndef MOVE_KERNEL_HPP
#define MOVE_KERNEL_HPP

#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    class Move : public IndexingKernel3D
    {
    public:
        inline Move(const Vector3D &vector = Vector3D(), const IndexingKernel3D *beforeIndexing = nullptr): moveVec(vector), beforeIndexing(beforeIndexing){};

        inline Vector3D operator()(const Vector3D &vector) const override
        {
            Vector3D vec = (this->beforeIndexing == nullptr)? vector : (*this->beforeIndexing)(vector);
            return vec - moveVec;
        };

    private:
        Vector3D moveVec;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // MOVE_KERNEL_HPP