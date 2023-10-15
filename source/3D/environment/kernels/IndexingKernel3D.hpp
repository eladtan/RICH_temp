#ifndef HILBERT_INDEXING_HPP
#define HILBERT_INDEXING_HPP

#include "3D/hilbert/HilbertOrder3D.hpp"
#include "3D/elementary/Vector3D.hpp"

class IndexingKernel3D
{
public:
    virtual Vector3D operator()(const Vector3D &vector) const = 0;
};

#endif // HILBERT_INDEXING_HPP