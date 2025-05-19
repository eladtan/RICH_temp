#ifndef GLOBAL_COUNTER_HPP
#define GLOBAL_COUNTER_HPP

#ifdef RICH_MPI

#include "mpi/mpi_commands.hpp"

class GlobalCounter
{
public:
    GlobalCounter(const MPI_Comm &comm, int globalInitialValue);

    ~GlobalCounter();

    int Increment(int n);

    inline int Decrement(int n = 1){return this->Increment(-n);};

    int GetValue(void) const{return *this->counter;};

private:
    MPI_Comm comm;
    rank_t rank, size, master_rank;
    volatile int *counter;
    MPI_Win counter_win;
};

#endif // RICH_MPI

#endif // GLOBAL_COUNTER_HPP