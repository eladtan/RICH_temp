#ifndef REVED_FRUSTRUM_KERNEL_HPP
#define REVED_FRUSTRUM_KERNEL_HPP

#include <vector>
#include <algorithm>

#include "3D/elementary/Face.hpp"
#include "3D/elementary/Mat33.hpp"
#include "Identity.hpp"

#define NUM_FACES 6
#define FACE_EDGES_NUMBER 4
#define VERTICES_NUMBER 4

namespace Kernelization3D
{
    /*
    A transformation from a frustum to a rectangle.
    Named after Omri Reved (this transformation was his idea).
    */
    class RevedFrustrum : public IndexingKernel3D
    {
    public:
        RevedFrustrum(const std::vector<Face> &faces, const IndexingKernel3D *beforeIndexing = nullptr, const IndexingKernel3D *afterIndexing = nullptr);
        
        inline ~RevedFrustrum(){delete this->beforeIndexing; delete this->afterIndexing;};

        inline Vector3D operator()(const Vector3D &vector) const override
        {
            Vector3D vec = this->beforeTransformation(vector);
            return (this->afterIndexing == nullptr)? vec : (*this->afterIndexing)(vec);
        }

    private:
        Vector3D S;
        double h;
        const IndexingKernel3D *beforeIndexing;
        const IndexingKernel3D *afterIndexing;

        Vector3D find_S(const std::vector<Face> &faces) const;

        inline Vector3D beforeTransformation(const Vector3D &vector) const
        {
            Vector3D vec = (this->beforeIndexing == nullptr)? vector : (*this->beforeIndexing)(vector);
            double slope = (this->h - this->S.z) / (vec.z - this->S.z);
            double new_x = this->S.x + slope * (vec.x - this->S.x);
            double new_y = this->S.y + slope * (vec.y - this->S.y);
            double new_z = std::pow((vec.z - this->S.z), 8);
            return Vector3D(new_x, new_y, new_z);
        }
    };
}

#endif // REVED_FRUSTRUM_KERNEL_HPP