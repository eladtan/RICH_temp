#include "Affine.hpp"

Kernelization3D::Affine::Affine(const Linear &linear, const Vector3D &b, const IndexingKernel3D *beforeIndexing)
    : linear(linear), move(b), beforeIndexing(beforeIndexing) {}

Kernelization3D::Affine::Affine(const Mat33<double> &A, const Vector3D &b, const IndexingKernel3D *beforeIndexing)
    : Affine(Linear(A), b, beforeIndexing) {}

Vector3D Kernelization3D::Affine::operator()(const Vector3D &vector) const
{
    Vector3D vec = (this->beforeIndexing == nullptr) ? vector : (*this->beforeIndexing)(vector);
    return this->move(this->linear(vec));
}

std::string Kernelization3D::Affine::getTypeName() const { return "Affine"; }
