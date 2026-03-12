#include "Sphere.hpp"

Kernelization3D::Sphere::Sphere(const Vector3D &center, const IndexingKernel3D *beforeIndexing)
    : beforeIndexing(beforeIndexing)
{
    this->moveIndexing = Move(center);
}

Vector3D Kernelization3D::Sphere::operator()(const Vector3D &vector) const
{
    Vector3D vec = this->moveIndexing((this->beforeIndexing == nullptr) ? vector : (*this->beforeIndexing)(vector));
    double numerator = abs(vec);
    double denominator = std::max(std::abs(vec.x), std::max(std::abs(vec.y), std::abs(vec.z)));
    return (numerator / denominator) * vec;
}
