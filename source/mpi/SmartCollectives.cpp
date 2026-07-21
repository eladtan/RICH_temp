#include "SmartCollectives.hpp"

#ifdef RICH_MPI

#define MAX_STACKTRACE 64

std::vector<std::string> getStack(void)
{
    void *array[MAX_STACKTRACE];
    char **stack_data;
    int size = backtrace(array, MAX_STACKTRACE);
    stack_data = backtrace_symbols(array, size);
    std::vector<std::string> my_stack;
    if(stack_data != NULL)
    {
        for(int i = 0; i < size; i++)
        {
            my_stack.push_back(std::string(stack_data[i]));
        }
    }
    free(stack_data);
    return my_stack;
}

void EnsureSameStack(const MPI_Comm &comm)
{
    rank_t rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    std::vector<std::string> my_stack = getStack();
    std::vector<std::vector<std::string>> stacks = MPI_All_cast_by_ranks(my_stack, comm);
    for(int i = 1; i < size; i++)
    {
        bool equal = false;
        if(stacks[i].size() == stacks[i-1].size())
        {
            equal = std::equal(stacks[i-1].cbegin(), stacks[i-1].cend(), stacks[i].cbegin());
        }
        if(not equal)
        {
            UniversalError eo("Different stacks found before performing a collective call");
            eo.addEntry("Stack of rank " + std::to_string(i-1), stacks[i-1]);
            eo.addEntry("Stack of rank " + std::to_string(i), stacks[i]);
            throw eo;
        }
    }
    // All equal
}

int RMPI_Barrier(const MPI_Comm &comm)
{
    EnsureSameStack(comm);
    return MPI_Barrier(comm);
}

int RMPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, const MPI_Comm &comm)
{
    EnsureSameStack(comm);
    return MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
}

int RMPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, const MPI_Comm &comm)
{
    EnsureSameStack(comm);
    return MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

#endif // RICH_MPI