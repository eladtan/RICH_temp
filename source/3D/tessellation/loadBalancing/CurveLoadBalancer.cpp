#ifdef RICH_MPI

#include "CurveLoadBalancer.hpp"

CurveLoadBalancer::CurveLoadBalancer(const std::vector<curve_index_t> &boundaries): LoadBalancer(), boundaries(boundaries)
{}

#endif // RICH_MPI