#ifndef PROGRESS_COUNTER_HPP
#define PROGRESS_COUNTER_HPP

#ifdef RICH_MPI

#include "GlobalCounter.hpp"

class ProgressCounter
{
public:
    ProgressCounter(const MPI_Comm &comm, int myNumParticles);

    ~ProgressCounter();

    int Increment(int n);

    inline int Decrement(int n = 1){return this->Increment(-n);};
    
    void MarkDone(void);

    inline void Sync(void)
    {
        MPI_Win_lock(MPI_LOCK_SHARED, this->rank, MPI_MODE_NOCHECK, this->is_done_win);
        MPI_Win_sync(this->is_done_win);
        MPI_Win_unlock(this->rank, this->is_done_win);
    };

    volatile int *is_done;
    int localDecrementAmount;

private:    
    rank_t rank, size, master_rank;
    std::shared_ptr<GlobalCounter> counter;
    MPI_Win is_done_win;
};

#endif // RICH_MPI

#endif // PROGRESS_COUNTER_HPP