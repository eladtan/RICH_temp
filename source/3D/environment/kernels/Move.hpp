#ifndef MOVE_KERNEL_HPP
#define MOVE_KERNEL_HPP

#include "IndexingKernel3D.hpp"

class MoveIOHandler;

namespace Kernelization3D
{
    class Move : public IndexingKernel3D
    {
    public:
        Move(const Vector3D &vector = Vector3D(), const IndexingKernel3D *beforeIndexing = nullptr);

        Vector3D operator()(const Vector3D &vector) const override;

        std::string getTypeName() const override;

    private:
        friend class ::MoveIOHandler;
        Vector3D moveVec;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // MOVE_KERNEL_HPP