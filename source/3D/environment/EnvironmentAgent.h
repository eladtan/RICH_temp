#ifndef ENVIRONMENT_RICH_H
#define ENVIRONMENT_RICH_H

#ifdef RICH_MPI

#include <mpi.h>
#include <boost/container/flat_set.hpp>

#include "kernels/IndexingKernel3D.hpp"

class EnvironmentAgent
{
public:
    using RanksSet = boost::container::flat_set<int>;

    inline EnvironmentAgent(const Vector3D &ll, const Vector3D &ur, const MPI_Comm &comm = MPI_COMM_WORLD): ll(ll), ur(ur), comm(comm)
    {
        MPI_Comm_rank(this->comm, &this->rank);
        MPI_Comm_size(this->comm, &this->size);
    };
    virtual RanksSet getIntersectingRanks(const Vector3D &center, double radius) const = 0;
    virtual int getOwner(const Vector3D &point) const = 0;
    virtual void update(const std::vector<Vector3D> &newPoints) = 0;
    virtual void updateBorders(const std::vector<hilbert_index_t> &newRange, int newOrder) = 0;

protected:
    Vector3D ll, ur;
    MPI_Comm comm;
    int rank, size;
};

#endif // RICH_MPI

#endif // ENVIRONMENT_RICH_H