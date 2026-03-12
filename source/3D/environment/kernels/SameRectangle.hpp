#ifndef SAMERECTANGULAR_KERNEL_HPP
#define SAMERECTANGULAR_KERNEL_HPP

#include <vector>
#include <algorithm>

#include "3D/elementary/Face.hpp"
#include "3D/elementary/Mat33.hpp"
#include "Move.hpp"
#include "Scale.hpp"
#include "IndexingKernel3D.hpp"

class SameRectangle : public Kernelization3D::IndexingKernel3D
{
public:
    SameRectangle(const std::vector<Vector3D> &vertices = std::vector<Vector3D>(), const Kernelization3D::IndexingKernel3D *indexing = nullptr);
    
    SameRectangle(const Vector3D &ll, const Vector3D &ur, const IndexingKernel3D *indexing = nullptr);

    Vector3D operator()(const Vector3D &vector) const override;

    std::string getTypeName() const override;

private:
    friend class SameRectangleIOHandler;
    const Kernelization3D::IndexingKernel3D *indexing;
    Kernelization3D::Move moveIndexing;
    Kernelization3D::Scale scaleIndexing;
};

#endif // RECTANGULAR_KERNEL_HPP