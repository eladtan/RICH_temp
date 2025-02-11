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

    if(rank > 0)
    {
        MPI_Recv(NULL, 0, MPI_BYTE, rank - 1, SYNC_TAG, comm, MPI_STATUS_IGNORE);
    }
    func();
    if(rank < size - 1)
    {
        MPI_Send(NULL, 0, MPI_BYTE, rank + 1, SYNC_TAG, comm);
    }
}

#endif // RICH_MPI

#endif // SYNCHRONIZE_H