#ifndef REFLECTION_KERNEL_HPP
#define REFLECTION_KERNEL_HPP

#include "IndexingKernel3D.hpp"
#include "3D/elementary/Mat33.hpp"

class ReflectionIOHandler;

namespace Kernelization3D
{
    class Reflection : public IndexingKernel3D
    {
    public:
        Reflection(const Vector3D &reflectionVector, const IndexingKernel3D *beforeIndexing = nullptr);

        Vector3D operator()(const Vector3D &vector) const override;

        std::string getTypeName() const override;

    private:
        friend class ::ReflectionIOHandler;
        Vector3D reflectionVector;
        Vector3D factoredVec;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // REFLECTION_KERNEL_HPP