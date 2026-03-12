#include "Rectangle.hpp"

Kernelization3D::Rectangle::Rectangle(const std::vector<Vector3D> &vertices, const IndexingKernel3D *beforeIndexing)
    : beforeIndexing(beforeIndexing)
{
    Vector3D ll(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    Vector3D ur(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());

    for(const Vector3D &vertex : vertices)
    {
        Vector3D kerneledVertex = (this->beforeIndexing == nullptr) ? vertex : (*this->beforeIndexing)(vertex);
        ll.x = std::min(ll.x, kerneledVertex.x);
        ll.y = std::min(ll.y, kerneledVertex.y);
        ll.z = std::min(ll.z, kerneledVertex.z);
        ur.x = std::max(ur.x, kerneledVertex.x);
        ur.y = std::max(ur.y, kerneledVertex.y);
        ur.z = std::max(ur.z, kerneledVertex.z);
    }

    this->moveIndexing = Move(ll);
    this->scaleIndexing = Scale(ur - ll);
}

Kernelization3D::Rectangle::Rectangle(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *beforeIndexing)
    : beforeIndexing(beforeIndexing)
{
    this->moveIndexing = Move(ll);
    this->scaleIndexing = Scale(ur - ll);
}

Vector3D Kernelization3D::Rectangle::operator()(const Vector3D &vector) const
{
    Vector3D vec = (this->beforeIndexing == nullptr) ? vector : (*this->beforeIndexing)(vector);
    return this->scaleIndexing(this->moveIndexing(vec));
}
