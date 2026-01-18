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
    this->waitingFor = NO_RANK;
    MPI_Irecv(&this->incomingData, 1, MPI_DOUBLE, MPI_ANY_SOURCE, ASK_REALLOCATION_TAG, this->comm, &this->incomingRequest);
}

ReallocationAgent::~ReallocationAgent()
{
    if(this->incomingRequest != MPI_REQUEST_NULL)
    {
        MPI_Cancel(&this->incomingRequest);
    }
}

void ReallocationAgent::GetIncoming(void)
{
    while(true)
    {
        int flag;
        MPI_Status status;
        MPI_Test(&this->incomingRequest, &flag, &status);
        if(__glibc_unlikely(flag))
        {
            this->incoming.push_back({status.MPI_SOURCE, this->incomingData});
            MPI_Irecv(&this->incomingData, 1, MPI_DOUBLE, MPI_ANY_SOURCE, ASK_REALLOCATION_TAG, this->comm, &this->incomingRequest);
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
}

rank_t ReallocationAgent::ShouldReallocate(void)
{
    this->GetIncoming();

    if(not this->incoming.empty())
    {
        rank_t toHandle = NO_RANK;
        std::vector<std::pair<rank_t, double>>::iterator it;

        if(this->waitingFor != NO_RANK)
        {
            it = std::find_if(this->incoming.begin(), this->incoming.end(), [this](const std::pair<rank_t, double> &p)
            {
                return p.first == this->waitingFor;
            });
        }
        if(this->waitingFor == NO_RANK or it == this->incoming.end())
        {
            it = this->incoming.begin();
        }
        
        toHandle = it->first;
        this->incoming.erase(it);
        
        MPI_Send(MPI_BOTTOM, 0, MPI_INT, toHandle, ANSWER_REALLOCATION_TAG, this->comm);
        return toHandle;
    }

    return NO_RANK;
}

rank_t ReallocationAgent::HandleWaitingReallocations(void)
{
    rank_t handle = this->ShouldReallocate();
    if(handle != NO_RANK)
    {
        assert(handle != this->rank);
        this->reallocationFunction(handle);
        this->reallocationsWhileWaiting++;
    }
    return handle;
}

void ReallocationAgent::HandleAllWaitingReallocations(void)
{
    rank_t handle = this->ShouldReallocate();
    while(handle != NO_RANK)
    {
        assert(handle != this->rank);
        this->reallocationFunction(handle);
        handle = this->ShouldReallocate();
        this->reallocationsWhileWaiting++;
    }
}

void ReallocationAgent::RequestReallocation(rank_t fromRank)
{
    rank_t r;
    do
    {
        r = this->HandleWaitingReallocations();
        if(r == fromRank)
        {
            // the peer asked before
            return; // no need to ask again
        }
    } while(r != NO_RANK);
    
    this->waitingFor = fromRank;
    MPI_Request request1, request2;
    MPI_Irecv(MPI_BOTTOM, 0, MPI_INT, fromRank, ANSWER_REALLOCATION_TAG, this->comm, &request1);
    double time = MPI_Wtime();
    MPI_Issend(&time, 1, MPI_DOUBLE, fromRank, ASK_REALLOCATION_TAG, this->comm, &request2);
    this->reallocationsWhileWaiting = 0;
    
    // bool alreadyJoinedWithPeer = false;
    // double alreadyJoinedTime = 0;

    // bool printed = false;

    while(true)
    {
        double elapsed_time = MPI_Wtime() - time;
        int flag;
        MPI_Test(&request1, &flag, MPI_STATUS_IGNORE);
        if(flag)
        {
            // finally!
            MPI_Wait(&request2, MPI_STATUS_IGNORE);
            // if(elapsed_time > 2)
            // {
            //     std::cout << "Warning: rank " << this->rank << " waited " << elapsed_time << " seconds for reallocation from rank " << fromRank << ", meanwhile did " << this->reallocationsWhileWaiting << " reallocations, already joined: " << alreadyJoinedWithPeer << " (since " << (alreadyJoinedWithPeer? std::to_string((MPI_Wtime() - alreadyJoinedTime)) : "-") << ")" << std::endl;
            // }
            
            this->reallocationFunction(fromRank); // join to peer with reallocation
            this->waitingFor = NO_RANK;
            return;
        }
        else
        {
            // make progress
            rank_t handled = this->HandleWaitingReallocations();
            // if(handled == this->waitingFor)
            // {
            //     alreadyJoinedTime = MPI_Wtime();
            //     alreadyJoinedWithPeer = true;
            // }
        }
    }
}

#endif // RICH_MPI