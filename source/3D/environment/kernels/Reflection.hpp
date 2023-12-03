#ifndef REFLECTION_KERNEL_HPP
#define REFLECTION_KERNEL_HPP

#include "IndexingKernel3D.hpp"
#include "3D/elementary/Mat33.hpp"

namespace Kernelization3D
{
    class Reflection : public IndexingKernel3D
    {
    public:
        inline Reflection(const Vector3D &reflectionVector, const IndexingKernel3D *beforeIndexing = nullptr): beforeIndexing(beforeIndexing), reflectionVector(reflectionVector), factoredVec(reflectionVector * (2 / abs(reflectionVector))){};

        inline Vector3D operator()(const Vector3D &vector) const override
        {
            Vector3D vec = (this->beforeIndexing == nullptr)? vector : (*this->beforeIndexing)(vector);
            Vector3D result = vec - (ScalarProd(vec, reflectionVector)) * this->factoredVec;
            return result;
        };

    private:
        Vector3D reflectionVector;
        Vector3D factoredVec;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // REFLECTION_KERNEL_HPP