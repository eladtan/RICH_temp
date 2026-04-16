#ifndef LOAD_BALANCER_HPP
#define LOAD_BALANCER_HPP

#ifdef RICH_MPI

#include "3D/elementary/Vector3D.hpp"
#include <string>
#include <vector>
#include <mpi.h>

class LoadBalancer
{
public:
    LoadBalancer(const MPI_Comm &comm = MPI_COMM_WORLD);

    virtual void rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights = std::vector<double>()) = 0;

    virtual void changeBox(const std::pair<Vector3D, Vector3D> &newBox) = 0;
    
    virtual void printInfo(void) = 0;
    
    virtual std::string getTypeName() const = 0;

    virtual int getOwner(const Vector3D &point) const = 0;

    virtual ~LoadBalancer() = default;

protected:
    MPI_Comm comm;
    rank_t rank, size;
};

#endif // RICH_MPI

#endif // LOAD_BALANCER_HPP