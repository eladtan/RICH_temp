#include "GenericScale.hpp"

Kernelization3D::GenericScale::GenericScale(const ScaleFunction &x, const ScaleFunction &y, const ScaleFunction &z, const IndexingKernel3D *beforeIndexing)
    : x(x), y(y), z(z), beforeIndexing(beforeIndexing) {}

Vector3D Kernelization3D::GenericScale::operator()(const Vector3D &vector) const
{
    Vector3D vec = (this->beforeIndexing == nullptr) ? vector : (*this->beforeIndexing)(vector);
    return Vector3D(this->x(vec.x), this->y(vec.y), this->z(vec.z));
}

std::string Kernelization3D::GenericScale::getTypeName() const { return "GenericScale"; }
