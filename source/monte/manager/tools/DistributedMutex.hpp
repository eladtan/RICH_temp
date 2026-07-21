#ifndef DISTRIBUTED_MUTEX_HPP
#define DISTRIBUTED_MUTEX_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include "mpi/mpi_commands.hpp"

class DistributedMutex
{
public:
    DistributedMutex(const MPI_Comm &comm, rank_t rank);
    
    ~DistributedMutex();

    void Lock(void);

    void Unlock(void);

    void Destroy(void);

    void Sync(void);

private:
    const MPI_Comm &comm;
    rank_t rank;
    int *value;
    MPI_Win win;
    bool destroyed;
};

#endif // RICH_MPI

#endif // DISTRIBUTED_MUTEX_HPP