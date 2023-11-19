#ifndef HILBERT_INDEXING_HPP
#define HILBERT_INDEXING_HPP

#include "3D/hilbert/HilbertOrder3D.hpp"
#include "3D/elementary/Vector3D.hpp"

class IndexingKernel3D
{
public:
    virtual ~IndexingKernel3D() = default;
    virtual Vector3D operator()(const Vector3D &vector) const = 0;
    inline Vector3D operator()(double x, double y, double z) const{return this->operator()(Vector3D(x, y, z));};
};

#endif // HILBERT_INDEXING_HPP