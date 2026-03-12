#ifndef PARALLELEPIPED_KERNEL_HPP
#define PARALLELEPIPED_KERNEL_HPP

#include <vector>
#include <algorithm>

#include "3D/elementary/Face.hpp"
#include "3D/elementary/Mat33.hpp"
#include "Move.hpp" // move kernel
#include "IndexingKernel3D.hpp"

#define NUM_FACES 6 // a parallelepiped has 6 faces
#define FACE_VERTICES_NUM 4 // each face should have 4 vertices

namespace Kernelization3D
{
    class Parallelepiped : public IndexingKernel3D
    {
    public:
        Parallelepiped(const Vector3D &u, const Vector3D &v, const Vector3D &w, const IndexingKernel3D *beforeIndexing = nullptr);
        
        ~Parallelepiped();

        Parallelepiped(const std::vector<Face> &faces, const IndexingKernel3D *beforeIndexing = nullptr);

        Vector3D operator()(const Vector3D &vector) const override;

        std::string getTypeName() const override;

    private:
        friend class ParallelepipedIOHandler;
        void calculateTransformation(const Vector3D &u, const Vector3D &v, const Vector3D &w);

        Mat33<typename Vector3D::coord_type> transformation;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // PARALLELEPIPED_KERNEL_HPP