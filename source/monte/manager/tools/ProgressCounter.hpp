#ifndef PROGRESS_COUNTER_HPP
#define PROGRESS_COUNTER_HPP

#ifdef RICH_MPI

#include "GlobalCounter.hpp"

class ProgressCounter
{
public:
    ProgressCounter(const MPI_Comm &comm);

    ~ProgressCounter();

    void Destroy(void);

    void Reset(int myNumParticles);

    int Increment(int n);

    inline int Decrement(int n = 1){return this->Increment(-n);};
    
    void MarkDone(void);

    inline void Sync(void)
    {
        MPI_Win_lock(MPI_LOCK_SHARED, this->rank, MPI_MODE_NOCHECK, this->is_done_win);
        MPI_Win_sync(this->is_done_win);
        MPI_Win_unlock(this->rank, this->is_done_win);
    };

    int GetValue(void) const{return this->counter->GetValue();};
    
    volatile int *is_done;
    int localDecrementAmount;
    
private:    
    std::shared_ptr<GlobalCounter> counter; // TODO: should be private
    rank_t rank, size, master_rank;
    MPI_Win is_done_win;
    MPI_Comm comm;
    bool destroyed;
};

#endif // RICH_MPI

#endif // PROGRESS_COUNTER_HPP