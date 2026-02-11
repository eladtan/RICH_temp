#ifndef PARTITION_LOAD_BALANCER_HPP
#define PARTITION_LOAD_BALANCER_HPP

#include <vector>
#include "LoadBalancer.hpp"

class PartitionLoadBalancer : public LoadBalancer
{
public:
    PartitionLoadBalancer(const std::vector<size_t> &boundaries): boundaries(boundaries){};

    ~PartitionLoadBalancer() override = default;

    std::vector<size_t> boundaries;
};

#endif // PARTITION_LOAD_BALANCER_HPP