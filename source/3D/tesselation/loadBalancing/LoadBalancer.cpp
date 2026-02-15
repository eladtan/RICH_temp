#ifdef RICH_MPI

#include "LoadBalancer.hpp"

LoadBalancer::LoadBalancer(const MPI_Comm &comm)
    : comm(comm)
{
    MPI_Comm_rank(this->comm, &this->rank);
    MPI_Comm_size(this->comm, &this->size);
}

#endif // RICH_MPI