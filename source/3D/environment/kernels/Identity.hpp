#ifndef IDENTITY_KERNEL_HPP
#define IDENTITY_KERNEL_HPP

#include "IndexingKernel3D.hpp"

class Identity : public IndexingKernel3D
{
public:
    inline Identity(const IndexingKernel3D *beforeIndexing = nullptr): beforeIndexing(beforeIndexing){};

    inline Vector3D operator()(const Vector3D &vector) const override
    {
        return (this->beforeIndexing == nullptr)? vector : (*this->beforeIndexing)(vector);
    };

private:
    const IndexingKernel3D *beforeIndexing;
};

#endif // IDENTITY_KERNEL_HPP