#ifndef SAMERECTANGULAR_KERNEL_HPP
#define SAMERECTANGULAR_KERNEL_HPP

#include <vector>
#include <algorithm>

#include "3D/elementary/Face.hpp"
#include "3D/elementary/Mat33.hpp"
#include "Move.hpp"
#include "Scale.hpp"
#include "IndexingKernel3D.hpp"

class SameRectangle : public Kernelization3D::IndexingKernel3D
{
public:
    inline SameRectangle(const std::vector<Vector3D> &vertices = std::vector<Vector3D>(), const Kernelization3D::IndexingKernel3D *indexing = nullptr): indexing(indexing)
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
        double const max_scale = std::max(ur.z - ll.z, std::max(ur.x - ll.x, ur.y - ll.y));
        this->moveIndexing = Kernelization3D::Move(ll);
        this->scaleIndexing = Kernelization3D::Scale(Vector3D(max_scale, max_scale, max_scale));
    }
    
    SameRectangle(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *indexing = nullptr): indexing(indexing)
    {
        double const max_scale = std::max(ur.z - ll.z, std::max(ur.x - ll.x, ur.y - ll.y));
        this->moveIndexing = Kernelization3D::Move(ll);
        this->scaleIndexing = Kernelization3D::Scale(Vector3D(max_scale, max_scale, max_scale));
    }

    inline Vector3D operator()(const Vector3D &vector) const override
    {
        Vector3D vec = (this->indexing == nullptr)? vector : (*this->indexing)(vector);
        return this->scaleIndexing(this->moveIndexing(vec));
    };

private:
    const Kernelization3D::IndexingKernel3D *indexing;
    Kernelization3D::Move moveIndexing;
    Kernelization3D::Scale scaleIndexing;
};

#endif // RECTANGULAR_KERNEL_HPP