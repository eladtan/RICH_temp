#include "Move.hpp"

Kernelization3D::Move::Move(const Vector3D &vector, const IndexingKernel3D *beforeIndexing)
    : moveVec(vector), beforeIndexing(beforeIndexing) {}

Vector3D Kernelization3D::Move::operator()(const Vector3D &vector) const
{
    Vector3D vec = (this->beforeIndexing == nullptr) ? vector : (*this->beforeIndexing)(vector);
    return vec - moveVec;
}

std::string Kernelization3D::Move::getTypeName() const { return "Move"; }
