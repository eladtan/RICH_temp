#ifndef RECTANGULAR_KERNEL_HPP
#define RECTANGULAR_KERNEL_HPP

#include <vector>
#include <algorithm>

#include "3D/elementary/Face.hpp"
#include "3D/elementary/Mat33.hpp"
#include "Move.hpp"
#include "Scale.hpp"
#include "IndexingKernel3D.hpp"

class Rectangle : public IndexingKernel3D
{
public:
    inline Rectangle(const std::vector<Vector3D> &vertices = std::vector<Vector3D>(), const IndexingKernel3D *indexing = nullptr): indexing(indexing)
    {
        Vector3D ll(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
        Vector3D ur(std::numeric_limits<double>::min(), std::numeric_limits<double>::min(), std::numeric_limits<double>::min());
        
        for(const Vector3D &vertex : vertices)
        {
            Vector3D kerneledVertex = (this->indexing == nullptr)? vertex : (*this->indexing)(vertex);
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
    
    Rectangle(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *indexing = nullptr): indexing(indexing)
    {
        this->moveIndexing = Move(ll);
        this->scaleIndexing = Scale(ur - ll);
    }

    inline Vector3D operator()(const Vector3D &vector) const override
    {
        Vector3D vec = (this->indexing == nullptr)? vector : (*this->indexing)(vector);
        return this->scaleIndexing(this->moveIndexing(vec));
    };

private:
    const IndexingKernel3D *indexing;
    Move moveIndexing;
    Scale scaleIndexing;
};

#endif // RECTANGULAR_KERNEL_HPP