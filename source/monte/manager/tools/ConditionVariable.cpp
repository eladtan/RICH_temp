#include "ConditionVariable.hpp"

#ifdef RICH_MPI

ConditionVariable::ConditionVariable(const MPI_Comm &comm)
    : comm(comm), destroyed(false)
{
    MPI_Comm_rank(comm, &this->internal_rank);
    rank_t size;
    MPI_Comm_size(comm, &size);
    if(size != 2)
    {
        throw UniversalError("ConditionVariable only works with 2 ranks");
    }
    this->other_rank = 1 - this->internal_rank;
    MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm, &this->value, &this->win);
    *this->value = 0;
    MPI_Barrier(this->comm);
}

ConditionVariable::~ConditionVariable(void)
{
    if(not this->destroyed)
    {
        this->Destroy();
    }
}

void ConditionVariable::Destroy(void)
{
    if(this->destroyed)
    {
        return;
    }
    // std::cout << "Destroys cond var" << std::endl;
    MPI_Barrier(this->comm);
    MPI_Win_free(&this->win);
    this->destroyed = true;
}

void ConditionVariable::Sync(void)
{
    MPI_Win_lock(MPI_LOCK_SHARED, this->internal_rank, MPI_MODE_NOCHECK, this->win);
    MPI_Win_sync(this->win);
    MPI_Win_unlock(this->internal_rank, this->win);
}

void ConditionVariable::Wait(DistributedMutex &mutex, const std::function<void(void)> &work_function)
{
    mutex.Unlock();
    int &value = *this->value;
    while(value == 0)
    {
        work_function(); // do work, then try again
        this->Sync();
    }
    // out! reset value
    value = 0;
    MPI_Barrier(this->comm);
    mutex.Lock();
}

void ConditionVariable::Notify(void)
{
    static int one = 1;
    MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->win);
    MPI_Put(&one, 1, MPI_INT, this->other_rank, 0, 1, MPI_INT, this->win);
    MPI_Win_unlock(this->other_rank, this->win);
    MPI_Barrier(this->comm);
}

#endif // RICH_MPI