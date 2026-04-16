#ifdef RICH_MPI

#include "CurveLoadBalancer.hpp"

CurveLoadBalancer::CurveLoadBalancer(const std::vector<curve_index_t> &boundaries): LoadBalancer(), boundaries(boundaries)
{}

int CurveLoadBalancer::getOwner(const Vector3D &point) const
{
    curve_index_t d = this->getCurveIndex(point);
    size_t index = std::distance(this->boundaries.cbegin(), std::upper_bound(this->boundaries.cbegin(), this->boundaries.cend(), d));
    return std::min<size_t>(index, (this->size - 1));
}

#endif // RICH_MPI
