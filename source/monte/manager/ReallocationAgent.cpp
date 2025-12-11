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

rank_t ReallocationAgent::ShouldReallocate(void)
{
    if(!this->incoming.empty())
    {
        rank_t toHandle = this->incoming.front().first;
        this->incoming.erase(this->incoming.begin());
        return toHandle;
    }
    MPI_Status status;
    int flag;
    while(true)
    {
        MPI_Iprobe(MPI_ANY_SOURCE, ASK_REALLOCATION_TAG, this->comm, &flag, &status);
        if(__glibc_unlikely(flag))
        {
            MPI_Request request;
            double time;
            MPI_Recv(&time, 1, MPI_DOUBLE, status.MPI_SOURCE, ASK_REALLOCATION_TAG, this->comm, &status);
            MPI_Isend(MPI_BOTTOM, 0, MPI_INT, status.MPI_SOURCE, ANSWER_REALLOCATION_TAG, this->comm, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            this->incoming.push_back({status.MPI_SOURCE, time});
        }
        else
        {
            break;
        }
    }
    std::sort(this->incoming.begin(), this->incoming.end(), [](const std::pair<rank_t, double> &a, const std::pair<rank_t, double> &b)
    {
        return a.second < b.second;
    });
    return NO_RANK;
}

void ReallocationAgent::HandleWaitingReallocations(void)
{
    rank_t handle = this->ShouldReallocate();
    if(handle != NO_RANK)
    {
        assert(handle != this->rank);
        this->reallocationFunction(handle);
    }
}

void ReallocationAgent::HandleAllWaitingReallocations(void)
{
    rank_t handle = this->ShouldReallocate();
    while(handle != NO_RANK)
    {
        assert(handle != this->rank);
        this->reallocationFunction(handle);
        handle = this->ShouldReallocate();
    }
}

void ReallocationAgent::RequestReallocation(rank_t fromRank)
{
    MPI_Request request1, request2;
    MPI_Irecv(MPI_BOTTOM, 0, MPI_INT, fromRank, ANSWER_REALLOCATION_TAG, this->comm, &request1);
    double time = MPI_Wtime();
    MPI_Isend(&time, 1, MPI_DOUBLE, fromRank, ASK_REALLOCATION_TAG, this->comm, &request2);
    while(true)
    {
        int flag;
        MPI_Test(&request1, &flag, MPI_STATUS_IGNORE);
        if(flag)
        {
            // finally!
            MPI_Wait(&request2, MPI_STATUS_IGNORE);
            this->reallocationFunction(fromRank); // join to peer with reallocation
            return;
        }
        else
        {
            // make progress
            this->HandleWaitingReallocations();
        }
    }
}

#endif // RICH_MPI