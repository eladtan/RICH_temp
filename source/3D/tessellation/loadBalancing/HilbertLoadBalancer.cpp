#ifdef RICH_MPI

#include "HilbertLoadBalancer.hpp"

#include <algorithm>

namespace
{
    // CurveLoadBalancer::getOwner uses std::upper_bound on boundaries, so the
    // boundaries vector must be sorted.  rescale()/changeBox() map every old cut
    // point through d -> xyz -> d after changing the Hilbert convertor box.
    // Hilbert order is not coordinate-wise monotone under that mapping, so two
    // adjacent cuts can swap.  Keeping the swapped order corrupts owner lookup
    // and can route particles/ghosts to the wrong rank.
    inline void SortHilbertBoundaries(std::vector<curve_index_t> &boundaries)
    {
        std::sort(boundaries.begin(), boundaries.end());
    }
}

HilbertLoadBalancer::HilbertLoadBalancer(const Vector3D &ll, const Vector3D &ur, const std::vector<Vector3D> &points, const std::shared_ptr<const Kernelization3D::IndexingKernel3D> indexing, const std::vector<curve_index_t> &boundaries)
    : CurveLoadBalancer(boundaries), indexing(indexing)
{
    SortHilbertBoundaries(this->boundaries);
    this->initializeConvertor(ll, ur, points);
}

HilbertLoadBalancer::HilbertLoadBalancer(std::shared_ptr<HilbertConvertor3D> convertor, std::shared_ptr<const Kernelization3D::IndexingKernel3D> indexing, const std::vector<curve_index_t> &boundaries)
    : CurveLoadBalancer(boundaries), convertor(std::move(convertor)), indexing(std::move(indexing))
{
    SortHilbertBoundaries(this->boundaries);
}

// Duplicates are allowed and are meaningful for empty/zero-width rank regions.

void HilbertLoadBalancer::initializeConvertor(const Vector3D &ll, const Vector3D &ur, const std::vector<Vector3D> &points)
{
    std::vector<Vector3D> kerneledVectors;
    Vector3D kerneledLL(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    Vector3D kerneledUR(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());

    for(const Vector3D &point : points)
    {
        Vector3D kerneledPoint = (*this->indexing)(point);
        kerneledVectors.push_back(kerneledPoint);
        kerneledLL.x = std::min<double>(kerneledLL.x, kerneledPoint.x);
        kerneledLL.y = std::min<double>(kerneledLL.y, kerneledPoint.y);
        kerneledLL.z = std::min<double>(kerneledLL.z, kerneledPoint.z);
        kerneledUR.x = std::max<double>(kerneledUR.x, kerneledPoint.x);
        kerneledUR.y = std::max<double>(kerneledUR.y, kerneledPoint.y);
        kerneledUR.z = std::max<double>(kerneledUR.z, kerneledPoint.z);
    }

    // consider the ll and ur as well
    for(const Vector3D &point : std::vector<Vector3D>({ll, ur}))
    {
        Vector3D kerneledPoint = (*this->indexing)(point);
        kerneledVectors.push_back(kerneledPoint);
        kerneledLL.x = std::min<double>(kerneledLL.x, kerneledPoint.x);
        kerneledLL.y = std::min<double>(kerneledLL.y, kerneledPoint.y);
        kerneledLL.z = std::min<double>(kerneledLL.z, kerneledPoint.z);
        kerneledUR.x = std::max<double>(kerneledUR.x, kerneledPoint.x);
        kerneledUR.y = std::max<double>(kerneledUR.y, kerneledPoint.y);
        kerneledUR.z = std::max<double>(kerneledUR.z, kerneledPoint.z);
    }

    OctTree<Vector3D> tree(kerneledLL, kerneledUR, kerneledVectors);

    MPI_Allreduce(MPI_IN_PLACE, &kerneledLL.x, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &kerneledLL.y, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &kerneledLL.z, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &kerneledUR.x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &kerneledUR.y, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &kerneledUR.z, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // make a little bit space
    double x_length = kerneledUR.x - kerneledLL.x, y_length = kerneledUR.y - kerneledLL.y, z_length = kerneledUR.z - kerneledLL.z;
    kerneledLL.x -= std::abs(SPACE_FACTOR * x_length);
    kerneledLL.y -= std::abs(SPACE_FACTOR * y_length);
    kerneledLL.z -= std::abs(SPACE_FACTOR * z_length);
    kerneledUR.x += std::abs(SPACE_FACTOR * x_length);
    kerneledUR.y += std::abs(SPACE_FACTOR * y_length);
    kerneledUR.z += std::abs(SPACE_FACTOR * z_length);

    size_t hilbertOrder = tree.getDepth();
    MPI_Allreduce(MPI_IN_PLACE, &hilbertOrder, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD); // calculates maximal depth
    hilbertOrder = std::min<size_t>(MAX_HILBERT_ORDER, hilbertOrder);
    this->convertor = std::make_shared<HilbertRectangularConvertor3D>(kerneledLL, kerneledUR, hilbertOrder);
}

void HilbertLoadBalancer::rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights)
{
    if(this->convertor == nullptr)
    {
        throw UniversalError("HilbertLoadBalancer::rebalance: convertor was not initialized yet");
    }

    std::vector<curve_index_t> indices;
    for(const Vector3D &point : points)
    {
        indices.push_back(this->convertor->xyz2d((*this->indexing)(point)));
    }

    if(this->rank == 0)
    {
        std::cout << "Running rebalancing" << std::endl;
    }
        // responsibilityRange = getWeightedBorders(indices, weights);
    this->boundaries = getWeightedBorders2(indices, weights);
    SortHilbertBoundaries(this->boundaries);
}

curve_index_t HilbertLoadBalancer::getCurveIndex(const Vector3D &point) const
{
    return this->convertor->xyz2d((*this->indexing)(point));
}

void HilbertLoadBalancer::rescale(const Vector3D &ll, const Vector3D &ur, const std::vector<Vector3D> &points)
{
    if (boundaries.empty())
    {
        this->initializeConvertor(ll, ur, points);
        return;
    }

    const Vector3D ll_old = convertor->getLL();
    const Vector3D ur_old = convertor->getUR();
    const Vector3D size_old = ur_old - ll_old;
    const Vector3D size_new = ur - ll;

    std::vector<Vector3D> rescaled(boundaries.size());
    for (size_t i = 0; i < boundaries.size(); ++i)
    {
        Vector3D p = convertor->d2xyz(boundaries[i]);
        rescaled[i].x = ll.x + (p.x - ll_old.x) / size_old.x * size_new.x;
        rescaled[i].y = ll.y + (p.y - ll_old.y) / size_old.y * size_new.y;
        rescaled[i].z = ll.z + (p.z - ll_old.z) / size_old.z * size_new.z;
    }

    this->initializeConvertor(ll, ur, points);

    for(size_t i = 0; i < boundaries.size(); ++i)
    {
        boundaries[i] = convertor->xyz2d(rescaled[i]);
    }
    SortHilbertBoundaries(this->boundaries);
}

void HilbertLoadBalancer::changeBox(const std::pair<Vector3D, Vector3D> &newBox)
{
    if (this->convertor == nullptr)
        return;

    const Vector3D &ll_new = newBox.first;
    const Vector3D &ur_new = newBox.second;

    const Vector3D ll_old = convertor->getLL();
    const Vector3D ur_old = convertor->getUR();
    const Vector3D size_old = ur_old - ll_old;

    Vector3D padded_ll = ll_new, padded_ur = ur_new;
    double x_len = ur_new.x - ll_new.x;
    double y_len = ur_new.y - ll_new.y;
    double z_len = ur_new.z - ll_new.z;
    padded_ll.x -= std::abs(SPACE_FACTOR * x_len);
    padded_ll.y -= std::abs(SPACE_FACTOR * y_len);
    padded_ll.z -= std::abs(SPACE_FACTOR * z_len);
    padded_ur.x += std::abs(SPACE_FACTOR * x_len);
    padded_ur.y += std::abs(SPACE_FACTOR * y_len);
    padded_ur.z += std::abs(SPACE_FACTOR * z_len);

    const Vector3D size_new = padded_ur - padded_ll;

    if(boundaries.empty())
    {
        convertor = std::make_shared<HilbertRectangularConvertor3D>(padded_ll, padded_ur, convertor->getOrder());
        return;
    }

    std::vector<Vector3D> rescaled(boundaries.size());
    for (size_t i = 0; i < boundaries.size(); ++i)
    {
        Vector3D p = convertor->d2xyz(boundaries[i]);
        rescaled[i].x = padded_ll.x + (p.x - ll_old.x) / size_old.x * size_new.x;
        rescaled[i].y = padded_ll.y + (p.y - ll_old.y) / size_old.y * size_new.y;
        rescaled[i].z = padded_ll.z + (p.z - ll_old.z) / size_old.z * size_new.z;
    }

    convertor = std::make_shared<HilbertRectangularConvertor3D>(padded_ll, padded_ur, convertor->getOrder());

    for (size_t i = 0; i < boundaries.size(); ++i)
    {
        boundaries[i] = convertor->xyz2d(rescaled[i]);
    }
    SortHilbertBoundaries(this->boundaries);
}

void HilbertLoadBalancer::setIndexing(const std::shared_ptr<const Kernelization3D::IndexingKernel3D>& newIndexing)
{
    this->indexing = newIndexing;
    this->convertor = nullptr;
}

std::shared_ptr<HilbertLoadBalancer> HilbertLoadBalancer::clone(void) const
{
    auto clonedConvertor = this->convertor
        ? std::dynamic_pointer_cast<HilbertConvertor3D>(this->convertor->clone())
        : nullptr;
    return std::make_shared<HilbertLoadBalancer>(clonedConvertor, this->indexing, this->boundaries);
}

void HilbertLoadBalancer::printInfo(void)
{
    std::cout << "HilbertLoadBalancer: boundaries=" << this->boundaries << std::endl;
}

#endif // RICH_MPI