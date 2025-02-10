#include "mpi_commands.hpp"

#ifdef RICH_MPI

void MPI_Timed_barrier(const MPI_Comm &comm, double seconds, std::string const &place)
{
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	std::vector<MPI_Request> requests(size, MPI_REQUEST_NULL);
	std::vector<bool> arrived(size, false);
	for(int _rank = 0; _rank < size; _rank++)
	{
		MPI_Isend(MPI_BOTTOM, 0, MPI_INT, _rank, MPI_TIMED_BARRIER_TAG, comm, &requests[_rank]);
	}

	int numArrived = 0;
	int ifNewMessageArrived = 0;
	MPI_Status status;

	std::chrono::_V2::system_clock::time_point start, end;
	start = std::chrono::system_clock::now();

	while(numArrived != size)
	{
		MPI_Iprobe(MPI_ANY_SOURCE, MPI_TIMED_BARRIER_TAG, comm, &ifNewMessageArrived, &status);
		if(ifNewMessageArrived)
		{
			MPI_Recv(MPI_BOTTOM, 0, MPI_INT, status.MPI_SOURCE, MPI_TIMED_BARRIER_TAG, comm, MPI_STATUS_IGNORE);
			arrived[status.MPI_SOURCE] = true;
			numArrived++;
		}
		else
		{
            end = std::chrono::system_clock::now();
			double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
			if(elapsed_seconds > seconds)
			{
				break; // timeout
			}
		}
	}

	for(int _rank = 0; _rank < size; _rank++)
	{
		if(not arrived[_rank])
		{
			UniversalError eo("One of more ranks did not arrive to the timed barrier");
			eo.addEntry("Function", place);
			eo.addEntry("Rank", _rank);
			throw eo;
		}
	}

	if(not requests.empty())
	{
		MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
	}
}

void MPI_Bcast_all(const void *buffer_send, int count, MPI_Datatype datatype, void *buffer_recv, MPI_Comm communicator)
{
	int size;
	MPI_Comm_size(communicator, &size);

	// first know how much data is being sent from each one
	std::vector<int> recvCounts(size, 0);
	MPI_Allgather(&count, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, communicator);
	std::vector<int> recvDisplacements(size, 0);
	for(int _rank = 1; _rank < size; _rank++)
	{
		recvDisplacements[_rank] = recvDisplacements[_rank - 1] + recvCounts[_rank];
	}
	std::vector<int> sendDisplacements(size, 0);
	std::vector<int> sendCounts(size, count);
	MPI_Alltoallv(buffer_send, sendCounts.data(), sendDisplacements.data(), datatype, buffer_recv, recvCounts.data(), recvDisplacements.data(), datatype, communicator);
}

#endif
