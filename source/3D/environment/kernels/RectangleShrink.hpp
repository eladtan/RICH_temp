#ifndef RECTANGULAR_SHRINK_KERNEL_HPP
#define RECTANGULAR_SHRINK_KERNEL_HPP

#include <vector>
#include <algorithm>

#include "Move.hpp"
#include "Shrink.hpp"
#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    class RectangleShrink : public IndexingKernel3D
    {
    public:
        inline RectangleShrink(const std::vector<Vector3D> &vertices = std::vector<Vector3D>(), const IndexingKernel3D *beforeIndexing = nullptr): beforeIndexing(beforeIndexing)
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
        
        RectangleShrink(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *beforeIndexing = nullptr): beforeIndexing(beforeIndexing)
        {
            this->moveIndexing = Move(ll);
            this->shrinkIndexing = Shrink(ur - ll);
        }

        inline Vector3D operator()(const Vector3D &vector) const override
        {
            Vector3D vec = (this->beforeIndexing == nullptr)? vector : (*this->beforeIndexing)(vector);
            return this->shrinkIndexing(this->moveIndexing(vec));
        };

    private:
        const IndexingKernel3D *beforeIndexing;
        Move moveIndexing;
        Shrink shrinkIndexing;
    };
}

#endif // RECTANGULAR_SHRINK_KERNEL_HPP