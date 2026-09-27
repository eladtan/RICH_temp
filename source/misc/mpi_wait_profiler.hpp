/*! \file mpi_wait_profiler.hpp
\brief Per-rank time spent inside MPI calls, for busy-time diagnostics
\details With RICH_MPI_WAIT_PROFILE=1 the PMPI interposers in
mpi_wait_profiler.cpp add the wall time of every blocking or polling MPI call
this process makes (collectives, waits, tests, probes, blocking point-to-point,
neighbour collectives, RMA synchronisation and window creation) to a counter.
The busy time of a code region is its wall time minus the change of Seconds()
over it; nested MPI calls (for example from a user reduction operator) are
counted once.  Not counted, so reported as busy: the initiation of nonblocking
operations (MPI_Isend, MPI_Irecv, MPI_Iallreduce, ...; their completion in
MPI_Wait/MPI_Test is counted), local queries (MPI_Comm_rank, MPI_Comm_size) and
datatype handling.  "Busy" is time outside counted MPI calls, which includes
local bookkeeping and polling loops around MPI_Test, not only useful work.  Unset or 0: the interposers only forward to PMPI.  The switch
is read per process; callers that act on it collectively must agree on it.
*/
#ifndef RICH_MPI_WAIT_PROFILER_HPP
#define RICH_MPI_WAIT_PROFILER_HPP 1


namespace mpi_wait_profiler
{
	//! Whether this process counts MPI time (RICH_MPI_WAIT_PROFILE=1)
	bool Enabled(void);

	//! Seconds this thread has spent inside counted MPI calls
	double Seconds(void);

}

#endif // RICH_MPI_WAIT_PROFILER_HPP
