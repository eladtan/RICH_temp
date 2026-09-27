#include "mpi_wait_profiler.hpp"

#include <cstdlib>
#include <cstring>

#ifdef RICH_MPI
#include <mpi.h>
#include <time.h>

namespace
{
	// Per thread: the counters describe the calling thread, and a nested MPI
	// call (from a user reduction operator, or MPI calling back into these
	// symbols) is counted once, by the outermost call.
	thread_local double counted_seconds = 0;
	thread_local int call_depth = 0;

	bool ProfileRequested(void)
	{
		static int const state = []()
		{
			char const* const value = std::getenv("RICH_MPI_WAIT_PROFILE");
			return value != nullptr && std::strcmp(value, "1") == 0 ? 1 : 0;
		}();
		return state == 1;
	}

	double MonotonicSeconds(void)
	{
		timespec now;
		clock_gettime(CLOCK_MONOTONIC, &now);
		return static_cast<double>(now.tv_sec) + 1e-9 * static_cast<double>(now.tv_nsec);
	}

	class CountedCall
	{
	public:
		CountedCall(void) : counted_(call_depth == 0 && ProfileRequested()), start_(0)
		{
			++call_depth;
			if(counted_)
				start_ = MonotonicSeconds();
		}

		~CountedCall(void)
		{
			--call_depth;
			if(counted_)
			{
				counted_seconds += MonotonicSeconds() - start_;
			}
		}

		CountedCall(CountedCall const&) = delete;
		CountedCall& operator=(CountedCall const&) = delete;

	private:
		bool const counted_;
		double start_;
	};
}

namespace mpi_wait_profiler
{
	bool Enabled(void)
	{
		return ProfileRequested();
	}

	double Seconds(void)
	{
		return counted_seconds;
	}
}

// PMPI interposers: same prototypes as OpenMPI 4.1.6 mpi.h.  Each one times the
// call and forwards it unchanged.  Non-blocking initiations (Isend, Irecv,
// Iallreduce, ...) are not timed: their waiting happens in Wait/Test.
extern "C"
{
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{ CountedCall counted; return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm); }

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{ CountedCall counted; return PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm); }

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{ CountedCall counted; return PMPI_Bcast(buffer, count, datatype, root, comm); }

int MPI_Barrier(MPI_Comm comm)
{ CountedCall counted; return PMPI_Barrier(comm); }

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{ CountedCall counted; return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm); }

int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm)
{ CountedCall counted; return PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm); }

int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{ CountedCall counted; return PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm); }

int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm)
{ CountedCall counted; return PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm); }

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{ CountedCall counted; return PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm); }

int MPI_Scatterv(const void *sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{ CountedCall counted; return PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm); }

int MPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{ CountedCall counted; return PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm); }

int MPI_Alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm)
{ CountedCall counted; return PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm); }

int MPI_Alltoallw(const void *sendbuf, const int sendcounts[], const int sdispls[], const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[], const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm)
{ CountedCall counted; return PMPI_Alltoallw(sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm); }

int MPI_Scan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{ CountedCall counted; return PMPI_Scan(sendbuf, recvbuf, count, datatype, op, comm); }

int MPI_Exscan(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{ CountedCall counted; return PMPI_Exscan(sendbuf, recvbuf, count, datatype, op, comm); }

int MPI_Reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{ CountedCall counted; return PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm); }

int MPI_Reduce_scatter_block(const void *sendbuf, void *recvbuf, int recvcount, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{ CountedCall counted; return PMPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm); }

int MPI_Neighbor_alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
{ CountedCall counted; return PMPI_Neighbor_alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm); }

int MPI_Neighbor_alltoallv(const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype, void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm)
{ CountedCall counted; return PMPI_Neighbor_alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm); }

int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
{ CountedCall counted; return PMPI_Comm_split(comm, color, key, newcomm); }

int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm)
{ CountedCall counted; return PMPI_Comm_dup(comm, newcomm); }

int MPI_Dist_graph_create(MPI_Comm comm_old, int n, const int nodes[], const int degrees[], const int targets[], const int weights[], MPI_Info info, int reorder, MPI_Comm *newcomm)
{ CountedCall counted; return PMPI_Dist_graph_create(comm_old, n, nodes, degrees, targets, weights, info, reorder, newcomm); }

int MPI_Comm_create_group(MPI_Comm comm, MPI_Group group, int tag, MPI_Comm *newcomm)
{ CountedCall counted; return PMPI_Comm_create_group(comm, group, tag, newcomm); }

int MPI_Comm_split_type(MPI_Comm comm, int split_type, int key, MPI_Info info, MPI_Comm *newcomm)
{ CountedCall counted; return PMPI_Comm_split_type(comm, split_type, key, info, newcomm); }

int MPI_Comm_free(MPI_Comm *comm)
{ CountedCall counted; return PMPI_Comm_free(comm); }

int MPI_Dist_graph_create_adjacent(MPI_Comm comm_old, int indegree, const int sources[], const int sourceweights[], int outdegree, const int destinations[], const int destweights[], MPI_Info info, int reorder, MPI_Comm *comm_dist_graph)
{ CountedCall counted; return PMPI_Dist_graph_create_adjacent(comm_old, indegree, sources, sourceweights, outdegree, destinations, destweights, info, reorder, comm_dist_graph); }

int MPI_Wait(MPI_Request *request, MPI_Status *status)
{ CountedCall counted; return PMPI_Wait(request, status); }

int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses)
{ CountedCall counted; return PMPI_Waitall(count, array_of_requests, array_of_statuses); }

int MPI_Waitany(int count, MPI_Request array_of_requests[], int *index, MPI_Status *status)
{ CountedCall counted; return PMPI_Waitany(count, array_of_requests, index, status); }

int MPI_Waitsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[])
{ CountedCall counted; return PMPI_Waitsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses); }

int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
{ CountedCall counted; return PMPI_Test(request, flag, status); }

int MPI_Testall(int count, MPI_Request array_of_requests[], int *flag, MPI_Status array_of_statuses[])
{ CountedCall counted; return PMPI_Testall(count, array_of_requests, flag, array_of_statuses); }

int MPI_Testany(int count, MPI_Request array_of_requests[], int *index, int *flag, MPI_Status *status)
{ CountedCall counted; return PMPI_Testany(count, array_of_requests, index, flag, status); }

int MPI_Testsome(int incount, MPI_Request array_of_requests[], int *outcount, int array_of_indices[], MPI_Status array_of_statuses[])
{ CountedCall counted; return PMPI_Testsome(incount, array_of_requests, outcount, array_of_indices, array_of_statuses); }

int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status)
{ CountedCall counted; return PMPI_Probe(source, tag, comm, status); }

int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status)
{ CountedCall counted; return PMPI_Iprobe(source, tag, comm, flag, status); }

int MPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message *message, MPI_Status *status)
{ CountedCall counted; return PMPI_Mprobe(source, tag, comm, message, status); }

int MPI_Improbe(int source, int tag, MPI_Comm comm, int *flag, MPI_Message *message, MPI_Status *status)
{ CountedCall counted; return PMPI_Improbe(source, tag, comm, flag, message, status); }

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
{ CountedCall counted; return PMPI_Recv(buf, count, datatype, source, tag, comm, status); }

int MPI_Mrecv(void *buf, int count, MPI_Datatype type, MPI_Message *message, MPI_Status *status)
{ CountedCall counted; return PMPI_Mrecv(buf, count, type, message, status); }

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{ CountedCall counted; return PMPI_Send(buf, count, datatype, dest, tag, comm); }

int MPI_Ssend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{ CountedCall counted; return PMPI_Ssend(buf, count, datatype, dest, tag, comm); }

int MPI_Rsend(const void *ibuf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{ CountedCall counted; return PMPI_Rsend(ibuf, count, datatype, dest, tag, comm); }

int MPI_Bsend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
{ CountedCall counted; return PMPI_Bsend(buf, count, datatype, dest, tag, comm); }

int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status)
{ CountedCall counted; return PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status); }

int MPI_Win_fence(int assertion, MPI_Win win)
{ CountedCall counted; return PMPI_Win_fence(assertion, win); }

int MPI_Win_flush(int rank, MPI_Win win)
{ CountedCall counted; return PMPI_Win_flush(rank, win); }

int MPI_Win_flush_all(MPI_Win win)
{ CountedCall counted; return PMPI_Win_flush_all(win); }

int MPI_Win_lock(int lock_type, int rank, int assertion, MPI_Win win)
{ CountedCall counted; return PMPI_Win_lock(lock_type, rank, assertion, win); }

int MPI_Win_unlock(int rank, MPI_Win win)
{ CountedCall counted; return PMPI_Win_unlock(rank, win); }

int MPI_Win_lock_all(int assertion, MPI_Win win)
{ CountedCall counted; return PMPI_Win_lock_all(assertion, win); }

int MPI_Win_unlock_all(MPI_Win win)
{ CountedCall counted; return PMPI_Win_unlock_all(win); }

int MPI_Win_sync(MPI_Win win)
{ CountedCall counted; return PMPI_Win_sync(win); }

int MPI_Win_create(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win)
{ CountedCall counted; return PMPI_Win_create(base, size, disp_unit, info, comm, win); }

int MPI_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win)
{ CountedCall counted; return PMPI_Win_allocate(size, disp_unit, info, comm, baseptr, win); }

int MPI_Win_allocate_shared(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win)
{ CountedCall counted; return PMPI_Win_allocate_shared(size, disp_unit, info, comm, baseptr, win); }

int MPI_Win_free(MPI_Win *win)
{ CountedCall counted; return PMPI_Win_free(win); }
}

#else // RICH_MPI

namespace mpi_wait_profiler
{
	bool Enabled(void)
	{
		return false;
	}

	double Seconds(void)
	{
		return 0;
	}
}

#endif // RICH_MPI
