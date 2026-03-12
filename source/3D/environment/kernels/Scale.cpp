#include "Scale.hpp"

Kernelization3D::Scale::Scale(const Vector3D &scale, const IndexingKernel3D *beforeIndexing)
    : scale(scale), beforeIndexing(beforeIndexing) {}

Kernelization3D::Scale::Scale(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *beforeIndexing)
    : Scale(ur - ll, beforeIndexing) {}

Vector3D Kernelization3D::Scale::operator()(const Vector3D &vector) const
{
    Vector3D vec = (this->beforeIndexing == nullptr) ? vector : (*this->beforeIndexing)(vector);
    return Vector3D(vec.x / this->scale.x, vec.y / this->scale.y, vec.z / this->scale.z);
}

std::string Kernelization3D::Scale::getTypeName() const { return "Scale"; }
