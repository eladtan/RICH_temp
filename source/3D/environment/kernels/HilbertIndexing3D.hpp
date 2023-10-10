#ifndef HILBERT_INDEXING_HPP
#define HILBERT_INDEXING_HPP

#include "3D/hilbert/HilbertOrder3D.hpp"
#include "3D/elementary/Vector3D.hpp"

class HilbertIndexing3D
{
public:
    virtual hilbert_index_t xyz2d(const Vector3D &point) const = 0;
};

#endif // HILBERT_INDEXING_HPP