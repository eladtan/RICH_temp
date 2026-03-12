#ifndef REVED_FRUSTRUM_KERNEL_HPP
#define REVED_FRUSTRUM_KERNEL_HPP

#include <vector>
#include <algorithm>

#include "3D/elementary/Face.hpp"
#include "3D/elementary/Mat33.hpp"
#include "Identity.hpp"

class RevedFrustrumIOHandler;

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
        
        ~RevedFrustrum();

        Vector3D operator()(const Vector3D &vector) const override;

        std::string getTypeName() const override;

    private:
        friend class ::RevedFrustrumIOHandler;
        RevedFrustrum(const Vector3D &S, double h, double ratio)
            : S(S), h(h), beforeIndexing(nullptr), afterIndexing(nullptr), ratio(ratio)
        {
        }
        Vector3D S;
        double h;
        const IndexingKernel3D *beforeIndexing;
        const IndexingKernel3D *afterIndexing;
        double ratio;

        Vector3D find_S(const std::vector<Face> &faces) const;

        Vector3D beforeTransformation(const Vector3D &vector) const;
    };
}

#endif // REVED_FRUSTRUM_KERNEL_HPP