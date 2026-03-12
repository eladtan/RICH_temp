#ifndef HILBERT_INDEXING_HPP
#define HILBERT_INDEXING_HPP

#include <vector>
#include <string>
#include "3D/hilbert/HilbertOrder3D.hpp"
#include "3D/elementary/Vector3D.hpp"

namespace Kernelization3D
{
    class IndexingKernel3D
    {
    public:
        virtual ~IndexingKernel3D() = default;

        virtual std::string getTypeName() const = 0;

        virtual Vector3D operator()(const Vector3D &vector) const = 0;

        inline Vector3D operator()(double x, double y, double z) const
        {
            return this->operator()(Vector3D(x, y, z));
        }
    
        inline std::vector<Vector3D> apply(const std::vector<Vector3D> &points) const
        {
            std::vector<Vector3D> kerneledPoints(points.size());
            std::transform(points.begin(), points.end(), kerneledPoints.begin(),
                [this](const Vector3D &point)
                {
                    return this->operator()(point);
                });
            return kerneledPoints;
        }
    };
}

#endif // HILBERT_INDEXING_HPP