#ifndef GENERIC_SCALE_KERNEL_HPP
#define GENERIC_SCALE_KERNEL_HPP

#include <functional> // for std::function

#include "IndexingKernel3D.hpp"

namespace Kernelization3D
{
    class GenericScale : public IndexingKernel3D
    {
    public:
        using ScaleFunction = std::function<double(double)>;

        inline GenericScale(const ScaleFunction &x, const ScaleFunction &y, const ScaleFunction &z, const IndexingKernel3D *beforeIndexing = nullptr): x(x), y(y), z(z), beforeIndexing(beforeIndexing){};

        inline Vector3D operator()(const Vector3D &vector) const override
        {
            Vector3D vec = (this->beforeIndexing == nullptr)? vector : (*this->beforeIndexing)(vector);
            Vector3D result = Vector3D(this->x(vec.x), this->y(vec.y), this->z(vec.z));
            // std::cout << "then to " << result << std::endl;
            return result;
        };

    private:
        ScaleFunction x, y, z;
        const IndexingKernel3D *beforeIndexing;
    };
}

#endif // GENERIC_SCALE_KERNEL_HPP