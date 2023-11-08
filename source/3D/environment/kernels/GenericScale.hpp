#ifndef GENERIC_SCALE_KERNEL_HPP
#define GENERIC_SCALE_KERNEL_HPP

#include <functional>

#include "IndexingKernel3D.hpp"

class GenericScale : public IndexingKernel3D
{
public:
    using ScaleFunction = std::function<double(double)>;

    inline GenericScale(const ScaleFunction &x, const ScaleFunction &y, const ScaleFunction &z, const IndexingKernel3D *indexing = nullptr): x(x), y(y), z(z), indexing(indexing){};

    inline Vector3D operator()(const Vector3D &vector) const override
    {
        Vector3D vec = (this->indexing == nullptr)? vector : (*this->indexing)(vector);
        Vector3D result = Vector3D(this->x(vec.x), this->y(vec.y), this->z(vec.z));
        // std::cout << "then to " << result << std::endl;
        return result;
    };

private:
    ScaleFunction x, y, z;
    const IndexingKernel3D *indexing;
};

#endif // GENERIC_SCALE_KERNEL_HPP