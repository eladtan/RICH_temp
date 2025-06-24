// #ifdef RICH_MPI

// #include "ReallocationAgent.hpp"

// static const rank_t NO_RANK = -1;

// ReallocationAgent::ReallocationAgent(const MPI_Comm &comm, const std::function<void(rank_t)> &reallocationFunction)
//     : comm(comm), reallocationFunction(reallocationFunction)
// {
//     MPI_Comm_rank(this->comm, &this->rank);
//     MPI_Comm_size(this->comm, &this->size);
//     int err = MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm, &this->who_to_reallocate, &this->win);
//     assert(err == MPI_SUCCESS);
//     err = MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm, &this->wants_for_reallocation, &this->wants_win);
//     assert(err == MPI_SUCCESS);
//     *this->who_to_reallocate = NO_RANK;
//     *this->wants_for_reallocation = NO_RANK;
//     MPI_Barrier(this->comm);
// }

// ReallocationAgent::~ReallocationAgent()
// {
//     MPI_Win_free(&this->win);
//     MPI_Win_free(&this->wants_win);
//     this->who_to_reallocate = nullptr;
//     this->wants_for_reallocation = nullptr;
// }

// rank_t ReallocationAgent::ShouldReallocate(void) const
// {
//     rank_t val;
//     MPI_Win_lock(MPI_LOCK_SHARED, this->rank, MPI_MODE_NOCHECK, this->win);
//     MPI_Get(&val, 1, MPI_INT, this->rank, 0, 1, MPI_INT, this->win);
//     MPI_Win_unlock(this->rank, this->win);
//     return val;
// }

// void ReallocationAgent::LockForChanges(void)
// {
//     bool done = false;
//     do
//     {
//         rank_t previous;
//         // MPI_Win_lock(MPI_LOCK_SHARED, this->rank, MPI_MODE_NOCHECK, this->win);
//         MPI_Win_lock_all(MPI_MODE_NOCHECK, this->win);
//         MPI_Compare_and_swap(&this->rank, &NO_RANK, &previous, MPI_INT, this->rank, 0, this->win);
//         MPI_Win_flush(this->rank, this->win);
//         // MPI_Win_unlock(this->rank, this->win);
//         MPI_Win_unlock_all(this->win);
//         done = previous == NO_RANK;
//         if(not done)
//         {
//             // we can not ask for reallocation since somebody is waiting for us. 
//             std::cout << "Rank " << this->rank << " found out that rank " << *this->who_to_reallocate << " is waiting for him." << std::endl;
//             this->HandleWaitingReallocations();
//         }
//     } while(not done);
//     assert(*this->who_to_reallocate == this->rank);
//     std::cout << "Successfuly set rank " << this->rank << " to reallocate for rank " << this->rank << "(I am rank " << this->rank << ")" << std::endl;
// }

// void ReallocationAgent::UnlockForChanges(void)
// {
//     this->FreeReallocation(this->rank);
// }

// void ReallocationAgent::RequestReallocation(rank_t fromRank)
// {
//     this->SetMyWant(fromRank);    
//     this->LockForChanges();
//     this->RequestReallocation(fromRank, this->rank);
//     std::cout << "Rank " << this->rank << " calls reallocation sync with rank " << fromRank << std::endl;
//     this->reallocationFunction(fromRank); // join to peer with reallocation
//     // the peer frees itself
//     this->UnlockForChanges();
//     this->SetMyWant(NO_RANK);
// }

// void ReallocationAgent::HandleWaitingReallocations(void)
// {
//     rank_t handle = this->ShouldReallocate();
//     if(handle != NO_RANK)
//     {
//         assert(handle != this->rank);
//         this->reallocationFunction(handle);
//         this->FreeReallocation(this->rank);
//     }
// }

// void ReallocationAgent::SetMyWant(rank_t rank)
// {
//     // ask `fromRank` to change to `toRank`.
//     MPI_Win_lock(MPI_LOCK_EXCLUSIVE, this->rank, MPI_MODE_NOCHECK, this->wants_win);
//     MPI_Put(&rank, 1, MPI_INT, this->rank, 0, 1, MPI_INT, this->wants_win);
//     MPI_Win_unlock(this->rank, this->wants_win); 
//     assert(*this->wants_for_reallocation == rank);   
// }

// void ReallocationAgent::RequestReallocation(rank_t fromRank, rank_t toRank)
// {
//     std::cout << "Rank " << this->rank << " asks rank " << fromRank << " to reallocate for rank " << toRank << std::endl;

//     bool done = false;
//     do
//     {
//         rank_t previous;
//         // MPI_Win_lock(MPI_LOCK_SHARED, fromRank, MPI_MODE_NOCHECK, this->win);
//         MPI_Win_lock_all(MPI_MODE_NOCHECK, this->win);
//         MPI_Compare_and_swap(&toRank, &NO_RANK, &previous, MPI_INT, fromRank, 0, this->win);
//         MPI_Win_flush(fromRank, this->win);
//         MPI_Win_unlock_all(this->win);
//         // MPI_Win_unlock(fromRank, this->win);
//         done = previous == NO_RANK;
//         if(not done and previous == fromRank) // self lock. Check who it waits for
//         {
//             // get the one the peer is waiting for. If that's me, everything's great!
//             rank_t who_it_waits;
//             MPI_Win_lock(MPI_LOCK_SHARED, fromRank, MPI_MODE_NOCHECK, this->wants_win);
//             MPI_Get(&who_it_waits, 1, MPI_INT, fromRank, 0, 1, MPI_INT, this->wants_win);
//             MPI_Win_unlock(fromRank, this->wants_win);
//             if(who_it_waits == this->rank)
//             {
//                 break;
//             }
//         }
//     } while(not done);
//     std::cout << "Successfuly set rank " << fromRank << " to reallocate for rank " << toRank << "(I am rank " << this->rank << ")" << std::endl;
// }

// void ReallocationAgent::FreeReallocation(rank_t rank)
// {
//     std::cout << "Rank " << this->rank << " frees rank " << rank << std::endl;
//     MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, MPI_MODE_NOCHECK, this->win);
//     MPI_Put(&NO_RANK, 1, MPI_INT, rank, 0, 1, MPI_INT, this->win);
//     MPI_Win_unlock(rank, this->win);
// }

// #endif // RICH_MPI

#ifdef RICH_MPI

#include "ReallocationAgent.hpp"

#define ASK_REALLOCATION_TAG 553
#define ANSWER_REALLOCATION_TAG 554

static const rank_t NO_RANK = -1;

ReallocationAgent::ReallocationAgent(const MPI_Comm &comm, const std::function<void(rank_t)> &reallocationFunction)
    : comm(comm), reallocationFunction(reallocationFunction)
{
    MPI_Comm_rank(this->comm, &this->rank);
    MPI_Comm_size(this->comm, &this->size);
    MPI_Barrier(this->comm);
}

ReallocationAgent::~ReallocationAgent()
{
}

rank_t ReallocationAgent::ShouldReallocate(void) const
{
    MPI_Status status;
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, ASK_REALLOCATION_TAG, this->comm, &flag, &status);
    if(__glibc_unlikely(flag))
    {
        int dummy;
        MPI_Recv(&dummy, 1, MPI_INT, status.MPI_SOURCE, ASK_REALLOCATION_TAG, this->comm, &status);
        MPI_Request request;
        MPI_Isend(&dummy, 1, MPI_INT, status.MPI_SOURCE, ANSWER_REALLOCATION_TAG, this->comm, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        return status.MPI_SOURCE;
    }
    return NO_RANK;
}

void ReallocationAgent::LockForChanges(void)
{}

void ReallocationAgent::UnlockForChanges(void)
{}

void ReallocationAgent::RequestReallocation(rank_t fromRank)
{
    this->RequestReallocation(fromRank, this->rank);
}

void ReallocationAgent::HandleWaitingReallocations(void)
{
    rank_t handle = this->ShouldReallocate();
    if(handle != NO_RANK)
    {
        assert(handle != this->rank);
        this->reallocationFunction(handle);
        this->FreeReallocation(this->rank);
    }
}

void ReallocationAgent::SetMyWant(rank_t rank)
{
}

void ReallocationAgent::RequestReallocation(rank_t fromRank, rank_t toRank)
{
    MPI_Request request1, request2;
    int dummy;
    MPI_Irecv(&dummy, 1, MPI_INT, fromRank, ANSWER_REALLOCATION_TAG, this->comm, &request1);
    MPI_Isend(&dummy, 1, MPI_INT, fromRank, ASK_REALLOCATION_TAG, this->comm, &request2);
    while(true)
    {
        int flag;
        MPI_Test(&request1, &flag, MPI_STATUS_IGNORE);
        if(flag)
        {
            // finally!
            this->reallocationFunction(fromRank); // join to peer with reallocation
            return;
        }
        else
        {
            this->HandleWaitingReallocations();
        }
    }
}

void ReallocationAgent::FreeReallocation(rank_t rank)
{
}

#endif // RICH_MPI