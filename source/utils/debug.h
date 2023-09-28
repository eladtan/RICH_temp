#include <iostream>
#include <unistd.h>
#include <execinfo.h>

#ifdef RICH_MPI
#include <mpi.h>
#include <shmem.h>
#define MPI_SYNC_TAG 2003
#define MPI_MASTER_SYNC_TAG 2002
#endif // RICH_MPI

#define MAX_HOSTNAME_LENGTH 50
#define MAX_STACK_LENGTH 120

void printDebugInfo()
{
    char hostname[MAX_HOSTNAME_LENGTH + 1];
    gethostname(hostname, MAX_HOSTNAME_LENGTH + 1);
    std::cout << "host: " << hostname << ", pid: " << getpid() << std::endl;
}

#ifdef RICH_MPI
void printDebugInfoMPI()
{
    char hostname[MPI_MAX_PROCESSOR_NAME + 1];
    int len;
    MPI_Get_processor_name(hostname, &len);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "Rank " << rank << ", hostname: " << hostname << ", pid: " << getpid() << std::endl;
}

void tellWhere()
{

}

void MPI_Synchronoize(int seconds, const MPI_Comm &comm = MPI_COMM_WORLD)
{
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    int arrived;
    int numArrived = 0;
    int dummy;

    MPI_Status status;
    std::vector<MPI_Request> requests;

    MPI_Iprobe(MPI_ANY_SOURCE, MPI_MASTER_SYNC_TAG, comm, &arrived, &status);
    if(arrived == 0)
    {
        requests.reserve(size);
        for(int _rank = 0; _rank < size; _rank++)
        {
            requests.push_back(MPI_REQUEST_NULL);
            MPI_Isend(&dummy, 1, MPI_BYTE, _rank, MPI_MASTER_SYNC_TAG, comm, &requests[requests.size() - 1]);
        }
    }
    MPI_Probe(MPI_ANY_SOURCE, MPI_MASTER_SYNC_TAG, comm, &status);
    int master = status.MPI_SOURCE;

    requests.push_back(MPI_REQUEST_NULL);
    MPI_Isend(&dummy, 1, MPI_BYTE, master, MPI_SYNC_TAG, comm, &requests[requests.size() - 1]);

    if(master == rank)
    {
        sleep(seconds);
        bool somebodyDidntArrived = false;
        for(int _rank = 0; _rank < size; _rank++)
        {
            MPI_Iprobe(_rank, MPI_SYNC_TAG, comm, &arrived, MPI_STATUS_IGNORE);
            if(!arrived)
            {
                somebodyDidntArrived = true;
                std::cout << "[MPI_Sync] rank " << _rank << " didn't arrive." << std::endl;
            }
        }
        if(!somebodyDidntArrived)
        {
            std::cout << "[MPI_Sync] All arrived." << std::endl;
        }
    }

    MPI_Iprobe(MPI_ANY_SOURCE, MPI_MASTER_SYNC_TAG, comm, &arrived, &status);
    while(arrived)
    {
        MPI_Recv(&dummy, 1, MPI_BYTE, status.MPI_SOURCE, MPI_MASTER_SYNC_TAG, comm, MPI_STATUS_IGNORE);
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_MASTER_SYNC_TAG, comm, &arrived, &status);
    }
    
    MPI_Iprobe(MPI_ANY_SOURCE, MPI_SYNC_TAG, comm, &arrived, &status);
    while(arrived)
    {
        MPI_Recv(&dummy, 1, MPI_BYTE, status.MPI_SOURCE, MPI_SYNC_TAG, comm, MPI_STATUS_IGNORE);
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_SYNC_TAG, comm, &arrived, &status);
    }
    if(!requests.empty())
    {    
        MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
    }
}

#endif // RICH_MPI

void printStack(int accuracy = MAX_STACK_LENGTH)
{
    void *stack[MAX_STACK_LENGTH];
    int size = backtrace(stack, accuracy);
    char **symbols = backtrace_symbols(stack, size);
    for(int i = 0; i < size; i++)
    {
        std::cout << symbols[i] << std::endl;
    }
    free(symbols);
}