#ifndef PARTITION_LOAD_BALANCER_HPP
#define PARTITION_LOAD_BALANCER_HPP

#include <cstddef>

using curve_index_t = size_t;

#ifdef RICH_MPI

#include <vector>
#include "LoadBalancer.hpp"

class CurveLoadBalancer : public LoadBalancer
{
public:
    CurveLoadBalancer(const std::vector<curve_index_t> &boundaries = std::vector<curve_index_t>());

    virtual ~CurveLoadBalancer() override = default;

    std::vector<curve_index_t> boundaries;
};

#endif // RICH_MPI

#endif // PARTITION_LOAD_BALANCER_HPP