#ifndef FRUSTRUM_KERNEL_HPP
#define FRUSTRUM_KERNEL_HPP

#include <sstream>
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

class Frustrum : public IndexingKernel3D
{
public:
    Frustrum(const std::vector<Face> &faces, const IndexingKernel3D *beforeIndexing = nullptr, const IndexingKernel3D *afterIndexing = nullptr);
    
    inline ~Frustrum(){delete this->beforeIndexing; delete this->afterIndexing;};

    inline Vector3D operator()(const Vector3D &vector) const override
    {
        Vector3D vec = this->beforeTransformation(vector);
        Vector3D result = (this->afterIndexing == nullptr)? vec : (*this->afterIndexing)(vec);
        return result;
    }

private:
    Mat44<double> P;
    const IndexingKernel3D *beforeIndexing;
    const IndexingKernel3D *afterIndexing;

    Vector3D find_S(const std::vector<Face> &faces) const;

    inline Vector3D beforeTransformation(const Vector3D &vector) const
    {
        Vector3D vec = (this->beforeIndexing == nullptr)? vector : (*this->beforeIndexing)(vector);
        // matrix multiplication
        Vector3D almostResult;
        almostResult.x = (this->P(0, 0) * vec[0]) + (this->P(0, 1) * vec[1]) + (this->P(0, 2) * vec[2]) + this->P(0, 3);
        almostResult.y = (this->P(1, 0) * vec[0]) + (this->P(1, 1) * vec[1]) + (this->P(1, 2) * vec[2]) + this->P(1, 3);
        almostResult.z = (this->P(2, 0) * vec[0]) + (this->P(2, 1) * vec[1]) + (this->P(2, 2) * vec[2]) + this->P(2, 3);
        double factor = 1/((this->P(3, 0) * vec[0]) + (this->P(3, 1) * vec[1]) + (this->P(3, 2) * vec[2]) + this->P(3, 3));
        return (almostResult * factor);
    }
};

#endif // FRUSTRUM_KERNEL_HPP