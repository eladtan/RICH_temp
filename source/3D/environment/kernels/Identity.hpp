#ifndef IDENTITY_KERNEL_HPP
#define IDENTITY_KERNEL_HPP

#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    class Identity : public IndexingKernel3D
    {
    public:
        Identity(const IndexingKernel3D *beforeIndexing = nullptr);

        Vector3D operator()(const Vector3D &vector) const override;

        std::string getTypeName() const override;

    private:
        friend class IdentityIOHandler;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // IDENTITY_KERNEL_HPP