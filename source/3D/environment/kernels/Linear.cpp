#include "Linear.hpp"

Kernelization3D::Linear::Linear(const Mat33<double> &transformation, const IndexingKernel3D *beforeIndexing)
    : transformation(transformation), beforeIndexing(beforeIndexing) {}

Vector3D Kernelization3D::Linear::operator()(const Vector3D &vector) const
{
    Vector3D vec = (this->beforeIndexing == nullptr) ? vector : (*this->beforeIndexing)(vector);
    return this->transformation * vec;
}

std::string Kernelization3D::Linear::getTypeName() const { return "Linear"; }
