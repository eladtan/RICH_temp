#include "Identity.hpp"

Kernelization3D::Identity::Identity(const IndexingKernel3D *beforeIndexing)
    : beforeIndexing(beforeIndexing) {}

Vector3D Kernelization3D::Identity::operator()(const Vector3D &vector) const
{
    return (this->beforeIndexing == nullptr) ? vector : (*this->beforeIndexing)(vector);
}

std::string Kernelization3D::Identity::getTypeName() const { return "Identity"; }
