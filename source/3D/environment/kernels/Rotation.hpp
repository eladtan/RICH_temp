#ifndef ROTATION_KERNEL_HPP
#define ROTATION_KERNEL_HPP

#include "IndexingKernel3D.hpp"
#include "3D/elementary/Mat33.hpp"

class Rotation : public IndexingKernel3D
{
public:

    enum Axis
    {
        X, Y, Z
    };

    inline Rotation(double theta, const Vector3D &rotationVector, const IndexingKernel3D *beforeIndexing = nullptr): beforeIndexing(beforeIndexing)
    {
        this->initializeMatrix(normalize(rotationVector), theta);
    };

    inline Rotation(double theta, const Axis &axis, const IndexingKernel3D *indexing = nullptr): beforeIndexing(beforeIndexing)
    {
        Vector3D rotationVector;
        switch(axis)
        {
            case X:
                rotationVector = Vector3D(1, 0, 0);
                break;
            case Y:
                rotationVector = Vector3D(0, 1, 0);
                break;
            case Z:
                rotationVector = Vector3D(0, 0, 1);
                break;
        }
        this->initializeMatrix(rotationVector, theta);
    };

    inline Vector3D operator()(const Vector3D &vector) const override
    {
        Vector3D vec = (this->beforeIndexing == nullptr)? vector : (*this->beforeIndexing)(vector);
        return this->mat * vec;
    };

private:
    Mat33<double> mat;
    const IndexingKernel3D *beforeIndexing;

    inline void initializeMatrix(const Vector3D &rotationVector, double theta)
    {
        double sinTheta = std::sin(theta);
        double cosTheta = std::cos(theta);

        mat.at(0, 0) = cosTheta + rotationVector.x * rotationVector.x * (1 - cosTheta);
        mat.at(0, 1) = rotationVector.x * rotationVector.y * (1 - cosTheta) - rotationVector.z * sinTheta;
        mat.at(0, 2) = rotationVector.x * rotationVector.z * (1 - cosTheta) + rotationVector.y * sinTheta;
        mat.at(1, 0) = rotationVector.y * rotationVector.x * (1 - cosTheta) + rotationVector.z * sinTheta;
        mat.at(1, 1) = cosTheta + rotationVector.y * rotationVector.y * (1 - cosTheta);
        mat.at(1, 2) = rotationVector.y * rotationVector.z * (1 - cosTheta) - rotationVector.x * sinTheta;
        mat.at(2, 0) = rotationVector.z * rotationVector.x * (1 - cosTheta) - rotationVector.y * sinTheta;
        mat.at(2, 1) = rotationVector.z * rotationVector.y * (1 - cosTheta) + rotationVector.x * sinTheta;
        mat.at(2, 2) = cosTheta + rotationVector.z * rotationVector.z * (1 - cosTheta);
    }   
};

#endif // ROTATION_KERNEL_HPP