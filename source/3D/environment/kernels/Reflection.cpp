#include "Reflection.hpp"

Kernelization3D::Reflection::Reflection(const Vector3D &reflectionVector, const IndexingKernel3D *beforeIndexing)
    : reflectionVector(reflectionVector), factoredVec(reflectionVector * (2 / abs(reflectionVector))), beforeIndexing(beforeIndexing) {}

Vector3D Kernelization3D::Reflection::operator()(const Vector3D &vector) const
{
    Vector3D vec = (this->beforeIndexing == nullptr) ? vector : (*this->beforeIndexing)(vector);
    return vec - (ScalarProd(vec, reflectionVector)) * this->factoredVec;
}

std::string Kernelization3D::Reflection::getTypeName() const { return "Reflection"; }
