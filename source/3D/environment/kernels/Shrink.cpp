#include "Shrink.hpp"

Kernelization3D::Shrink::Shrink(const Vector3D &scale, const IndexingKernel3D *beforeIndexing)
    : beforeIndexing(beforeIndexing)
{
    this->scale = 1 / std::max(scale[0], std::max(scale[1], scale[2]));
}

Kernelization3D::Shrink::Shrink(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *beforeIndexing)
    : Shrink(ur - ll, beforeIndexing) {}

Vector3D Kernelization3D::Shrink::operator()(const Vector3D &vector) const
{
    Vector3D vec = (this->beforeIndexing == nullptr) ? vector : (*this->beforeIndexing)(vector);
    return (vec * this->scale);
}

std::string Kernelization3D::Shrink::getTypeName() const { return "Shrink"; }
