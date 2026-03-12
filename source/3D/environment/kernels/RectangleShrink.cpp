#include "RectangleShrink.hpp"

Kernelization3D::RectangleShrink::RectangleShrink(const std::vector<Vector3D> &vertices, const IndexingKernel3D *beforeIndexing)
    : beforeIndexing(beforeIndexing)
{
    Vector3D ll(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    Vector3D ur(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());

    for(const Vector3D &vertex : vertices)
    {
        ll.x = std::min(ll.x, vertex.x);
        ll.y = std::min(ll.y, vertex.y);
        ll.z = std::min(ll.z, vertex.z);
        ur.x = std::max(ur.x, vertex.x);
        ur.y = std::max(ur.y, vertex.y);
        ur.z = std::max(ur.z, vertex.z);
    }
    this->moveIndexing = Move(ll);
    this->shrinkIndexing = Shrink(ur - ll);
}

Kernelization3D::RectangleShrink::RectangleShrink(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *beforeIndexing)
    : beforeIndexing(beforeIndexing)
{
    this->moveIndexing = Move(ll);
    this->shrinkIndexing = Shrink(ur - ll);
}

Vector3D Kernelization3D::RectangleShrink::operator()(const Vector3D &vector) const
{
    Vector3D vec = (this->beforeIndexing == nullptr) ? vector : (*this->beforeIndexing)(vector);
    return this->shrinkIndexing(this->moveIndexing(vec));
}

std::string Kernelization3D::RectangleShrink::getTypeName() const { return "RectangleShrink"; }
