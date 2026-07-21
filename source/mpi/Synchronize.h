#ifndef SYNCHRONIZE_H
#define SYNCHRONIZE_H

#ifdef RICH_MPI
#include <mpi.h>
#include "types.h"

#define SYNC_TAG 503

template<typename Function>
void MPI_Sync(const MPI_Comm &comm, const Function &func)
{
    rank_t rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    for(int i = 0; i < size; i++)
    {
        if(rank == i)
        {
            func();
        }
        MPI_Barrier(comm);
    }
}

#endif // RICH_MPI

#endif // SYNCHRONIZE_H