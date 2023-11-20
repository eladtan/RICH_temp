#ifndef RECTANGULAR_KERNEL_HPP
#define RECTANGULAR_KERNEL_HPP

#include <vector>
#include <algorithm>

#include "Move.hpp"
#include "Scale.hpp"
#include "IndexingKernel3D.hpp"

class Rectangle : public IndexingKernel3D
{
public:
    inline Rectangle(const std::vector<Vector3D> &vertices = std::vector<Vector3D>(), const IndexingKernel3D *beforeIndexing = nullptr): beforeIndexing(beforeIndexing)
    {
        Vector3D ll(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
        Vector3D ur(std::numeric_limits<double>::min(), std::numeric_limits<double>::min(), std::numeric_limits<double>::min());
        
        for(const Vector3D &vertex : vertices)
        {
            Vector3D kerneledVertex = (this->beforeIndexing == nullptr)? vertex : (*this->beforeIndexing)(vertex);
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
    
    Rectangle(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *beforeIndexing = nullptr): beforeIndexing(beforeIndexing)
    {
        this->moveIndexing = Move(ll);
        this->scaleIndexing = Scale(ur - ll);
    }

    inline Vector3D operator()(const Vector3D &vector) const override
    {
        Vector3D vec = (this->beforeIndexing == nullptr)? vector : (*this->beforeIndexing)(vector);
        return this->scaleIndexing(this->moveIndexing(vec));
    };

private:
    const IndexingKernel3D *beforeIndexing;
    Move moveIndexing;
    Scale scaleIndexing;
};

#endif // RECTANGULAR_KERNEL_HPP