#ifndef ROTATION_KERNEL_HPP
#define ROTATION_KERNEL_HPP

#include "IndexingKernel3D.hpp"
#include "3D/elementary/Mat33.hpp"

class RotationIOHandler;

namespace Kernelization3D
{
    class Rotation : public IndexingKernel3D
    {
    public:

        enum Axis
        {
            X, Y, Z
        };

        Rotation(double theta, const Vector3D &rotationVector, const IndexingKernel3D *beforeIndexing = nullptr);

        Rotation(double theta, const Axis &axis, const IndexingKernel3D *indexing = nullptr);

        Vector3D operator()(const Vector3D &vector) const override;

        std::string getTypeName() const override;

    private:
        friend class ::RotationIOHandler;
        Mat33<double> mat;
        const IndexingKernel3D *beforeIndexing;

        void initializeMatrix(const Vector3D &rotationVector, double theta);
    };
}

#endif // ROTATION_KERNEL_HPP