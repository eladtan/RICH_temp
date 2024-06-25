#ifndef SPHERICAL_KERNEL_HPP
#define SPHERICAL_KERNEL_HPP

#include <vector>
#include <algorithm>

#include "Move.hpp"
#include "IndexingKernel3D.hpp"

class Sphere : public IndexingKernel3D
{
public:
    inline Sphere(const Vector3D &center, const IndexingKernel3D *beforeIndexing = nullptr): beforeIndexing(beforeIndexing)
    {
        this->moveIndexing = Move(center);
    }

    inline Vector3D operator()(const Vector3D &vector) const override
    {
        Vector3D vec = this->moveIndexing((this->beforeIndexing == nullptr)? vector : (*this->beforeIndexing)(vector));
        double numerator = abs(vec);
        double denominator = std::max(std::abs(vec.x), std::max(std::abs(vec.y), std::abs(vec.z)));
        return (numerator / denominator) * vec;
    };

private:
    const IndexingKernel3D *beforeIndexing;
    Move moveIndexing;
};

#endif // SPHERICAL_KERNEL_HPP