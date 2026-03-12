#ifndef FRUSTRUM_KERNEL_HPP
#define FRUSTRUM_KERNEL_HPP

#include <vector>
#include <algorithm>

#include "3D/elementary/Face.hpp"
#include "3D/elementary/Mat44.hpp"
#include "Linear.hpp"
#include "Rectangle.hpp"
#include "RectangleShrink.hpp"
#include "IndexingKernel3D.hpp"

#define NUM_FACES 6
#define FACE_EDGES_NUMBER 4
#define VERTICES_NUMBER 4

// see here: https://math.stackexchange.com/questions/2265255/mapping-a-3d-point-inside-a-hexahedron-to-a-unit-cube

namespace Kernelization3D
{
    class Frustrum : public IndexingKernel3D
    {
    public:
        Frustrum(const std::vector<Face> &faces, const IndexingKernel3D *beforeIndexing = nullptr, const IndexingKernel3D *afterIndexing = nullptr);
        
        ~Frustrum();

        Vector3D operator()(const Vector3D &vector) const override;

        std::string getTypeName() const override;

    private:
        friend class FrustrumIOHandler;
        Frustrum(const Mat44<double> &P)
            : P(P), beforeIndexing(nullptr), afterIndexing(nullptr)
        {
        }
        Mat44<double> P;
        const IndexingKernel3D *beforeIndexing;
        const IndexingKernel3D *afterIndexing;

        Vector3D find_S(const std::vector<Face> &faces) const;

        Vector3D beforeTransformation(const Vector3D &vector) const;
    };
}

#endif // FRUSTRUM_KERNEL_HPP