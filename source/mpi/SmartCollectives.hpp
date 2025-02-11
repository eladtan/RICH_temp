#ifndef SMART_COLLECTIVES_HPP
#define SMART_COLLECTIVES_HPP


#include <vector>
#include <string>
#include <execinfo.h>
#ifdef RICH_MPI
#include <mpi.h>
#include "mpi/serialize/mpi_commands.hpp"

int RMPI_Barrier(const MPI_Comm &comm);

int RMPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, const MPI_Comm &comm);

int RMPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, const MPI_Comm &comm);

#endif // RICH_MPI

#endif // SMART_COLLECTIVES_HPP