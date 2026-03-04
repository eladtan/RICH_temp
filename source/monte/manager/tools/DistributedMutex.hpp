#ifndef DISTRIBUTED_MUTEX_HPP
#define DISTRIBUTED_MUTEX_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include <memory>
#include "mpi/mpi_commands.hpp"
#include "utils/rma/RMAFactory.hpp"

class DistributedMutex
{
public:
    DistributedMutex(const MPI_Comm &comm, rank_t rank, RDMA_Type rdma_type);
    
    ~DistributedMutex();

    void Lock(void);

    void Unlock(void);

    void Destroy(void);

private:
    MPI_Comm comm;
    rank_t rank;
    std::unique_ptr<RemoteMemoryAgent<int>> agent;
    bool destroyed;
};

#endif // RICH_MPI

#endif // DISTRIBUTED_MUTEX_HPP