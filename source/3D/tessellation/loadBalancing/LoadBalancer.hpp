#ifndef LOAD_BALANCER_HPP
#define LOAD_BALANCER_HPP

#ifdef RICH_MPI

#include "3D/elementary/Vector3D.hpp"
#include <vector>
#include <mpi.h>

class LoadBalancer
{
public:
    LoadBalancer(const MPI_Comm &comm = MPI_COMM_WORLD);

    virtual void rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights = std::vector<double>()) = 0;

    virtual ~LoadBalancer() = default;

protected:
    MPI_Comm comm;
    rank_t rank, size;
};

#endif // RICH_MPI

#endif // LOAD_BALANCER_HPP