#include "AmountManager.hpp"

#ifdef RICH_MPI

#define INCREASE_TAG 9918
#define DONE_TAG 9919
#define VERIFY_TAG 9920
#define VERIFY_RESULT_TAG 9921

namespace
{
    void CancelAndWait(MPI_Request &request)
    {
        if(request == MPI_REQUEST_NULL)
        {
            return;
        }
        MPI_Cancel(&request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        request = MPI_REQUEST_NULL;
    }
}

AmountManager::AmountManager(MPI_Comm comm, size_t flushInterval)
    : comm(comm), cyclesWaiting(0), flushInterval(flushInterval), globalNum(0), tempNum(0), outgoingNum(0),
      childVerifyResult1(0), childVerifyResult2(0), localVerifyRecorded(false), localVerifyOk(false), verify(false), done(false)
{
    MPI_Comm_rank(this->comm, &this->rank);
    MPI_Comm_size(this->comm, &this->size);

    this->parent = (this->rank - 1) / 2;
    this->child1 = this->rank * 2 + 1;
    this->child2 = this->rank * 2 + 2;

    this->request1 = MPI_REQUEST_NULL;
    this->request2 = MPI_REQUEST_NULL;
    this->parentDoneRequest = MPI_REQUEST_NULL;
    this->parentVerifyRequest = MPI_REQUEST_NULL;
    this->outgoingRequest = MPI_REQUEST_NULL;
    this->childVerifyRequest1 = MPI_REQUEST_NULL;
    this->childVerifyRequest2 = MPI_REQUEST_NULL;

    if(this->child1 < this->size)
    {
        MPI_Irecv(&this->recv1, 1, MPI_LONG_LONG, this->child1, INCREASE_TAG, this->comm, &this->request1);
    }
    else
    {
        this->recv1 = 0;
        this->request1 = MPI_REQUEST_NULL;
    }

    if(this->child2 < this->size)
    {
        MPI_Irecv(&this->recv2, 1, MPI_LONG_LONG, this->child2, INCREASE_TAG, this->comm, &this->request2);
    }
    else
    {
        this->recv2 = 0;
        this->request2 = MPI_REQUEST_NULL;
    }

    if(this->rank != 0)
    {
        MPI_Irecv(MPI_BOTTOM, 0, MPI_INT, this->parent, DONE_TAG, this->comm, &this->parentDoneRequest);
        MPI_Irecv(MPI_BOTTOM, 0, MPI_INT, this->parent, VERIFY_TAG, this->comm, &this->parentVerifyRequest);
    }

    MPI_Barrier(this->comm);
}

void AmountManager::Initialize(counter_t num)
{
    MPI_Reduce(&num, &this->globalNum, 1, MPI_LONG_LONG, MPI_SUM, 0, this->comm);
}

void AmountManager::Increase(counter_t n)
{
    this->tempNum += n;
}

void AmountManager::FlushToParent(void)
{
    if(this->rank == 0 || this->tempNum == 0)
        return;

    if(this->outgoingRequest != MPI_REQUEST_NULL)
        return;

    this->outgoingNum = this->tempNum;
    this->tempNum = 0;
    MPI_Isend(&this->outgoingNum, 1, MPI_LONG_LONG, this->parent, INCREASE_TAG, this->comm, &this->outgoingRequest);
    this->cyclesWaiting = 0;
}

void AmountManager::Progress(void)
{
    this->CheckVerify();
    this->CheckDone();
    MPI_Status status;
    int flag;

    if(this->outgoingRequest != MPI_REQUEST_NULL)
    {
        MPI_Test(&this->outgoingRequest, &flag, MPI_STATUS_IGNORE);
        if(flag)
        {
            this->outgoingNum = 0;
            this->outgoingRequest = MPI_REQUEST_NULL;
        }
    }

    MPI_Test(&this->request1, &flag, &status);
    if(this->child1 < this->size and flag)
    {
        this->tempNum += this->recv1;
        MPI_Irecv(&this->recv1, 1, MPI_LONG_LONG, this->child1, INCREASE_TAG, this->comm, &this->request1);
    }

    MPI_Test(&this->request2, &flag, &status);
    if(this->child2 < this->size and flag)
    {
        this->tempNum += this->recv2;
        MPI_Irecv(&this->recv2, 1, MPI_LONG_LONG, this->child2, INCREASE_TAG, this->comm, &this->request2);
    }

    if(this->rank == 0)
    {
        // no parent
        this->globalNum += this->tempNum;
        this->tempNum = 0;
    }
    else
    {
        this->cyclesWaiting++;
        bool urgent = (this->tempNum < 0);
        bool intervalElapsed = (this->cyclesWaiting >= this->flushInterval) && (this->tempNum != 0);
        if(urgent || intervalElapsed)
        {
            this->FlushToParent();
        }
    }

    if(this->rank == 0 and this->globalNum == 0)
    {
        this->AskChildrenVerify();
        return;
    }
}

void AmountManager::MarkChildrenDone(void)
{
    if(this->done)
    {
        return;
    }
    if(this->child1 < this->size)
    {
        MPI_Send(MPI_BOTTOM, 0, MPI_INT, this->child1, DONE_TAG, this->comm);
    }
    if(this->child2 < this->size)
    {
        MPI_Send(MPI_BOTTOM, 0, MPI_INT, this->child2, DONE_TAG, this->comm);
    }
    this->done = true;
}

void AmountManager::AskChildrenVerify(void)
{
    if(this->verify)
    {
        return;
    }
    this->ResetVerifyAttempt();
    this->verify = true;
    if(this->child1 < this->size)
    {
        MPI_Irecv(&this->childVerifyResult1, 1, MPI_INT, this->child1, VERIFY_RESULT_TAG, this->comm, &this->childVerifyRequest1);
    }
    if(this->child2 < this->size)
    {
        MPI_Irecv(&this->childVerifyResult2, 1, MPI_INT, this->child2, VERIFY_RESULT_TAG, this->comm, &this->childVerifyRequest2);
    }
    if(this->child1 < this->size)
    {
        MPI_Send(MPI_BOTTOM, 0, MPI_INT, this->child1, VERIFY_TAG, this->comm);
    }
    if(this->child2 < this->size)
    {
        MPI_Send(MPI_BOTTOM, 0, MPI_INT, this->child2, VERIFY_TAG, this->comm);
    }
}

void AmountManager::ResetVerifyAttempt(void)
{
    this->childVerifyResult1 = 0;
    this->childVerifyResult2 = 0;
    this->localVerifyRecorded = false;
    this->localVerifyOk = false;
    this->childVerifyRequest1 = MPI_REQUEST_NULL;
    this->childVerifyRequest2 = MPI_REQUEST_NULL;
}

bool AmountManager::ChildVerifyResultsReady(void)
{
    int flag;
    if(this->child1 < this->size && this->childVerifyRequest1 != MPI_REQUEST_NULL)
    {
        MPI_Test(&this->childVerifyRequest1, &flag, MPI_STATUS_IGNORE);
        if(!flag)
        {
            return false;
        }
    }
    if(this->child2 < this->size && this->childVerifyRequest2 != MPI_REQUEST_NULL)
    {
        MPI_Test(&this->childVerifyRequest2, &flag, MPI_STATUS_IGNORE);
        if(!flag)
        {
            return false;
        }
    }
    return true;
}

void AmountManager::FinishVerifyAttempt(bool subtreeOk)
{
    if(this->rank == 0)
    {
        if(subtreeOk)
        {
            this->MarkChildrenDone();
        }
        this->verify = false;
        this->ResetVerifyAttempt();
        return;
    }

    MPI_Irecv(MPI_BOTTOM, 0, MPI_INT, this->parent, VERIFY_TAG, this->comm, &this->parentVerifyRequest);
    int result = subtreeOk ? 1 : 0;
    MPI_Send(&result, 1, MPI_INT, this->parent, VERIFY_RESULT_TAG, this->comm);
    this->verify = false;
    this->ResetVerifyAttempt();
}

void AmountManager::Verify(bool ok)
{
    if(this->verify)
    {
        const bool currentLocalOk = ok && !this->HasPending() && (this->rank != 0 || this->globalNum == 0);
        if(!this->localVerifyRecorded)
        {
            this->localVerifyOk = currentLocalOk;
            this->localVerifyRecorded = true;
        }
        else
        {
            this->localVerifyOk = this->localVerifyOk && currentLocalOk;
        }

        if(!this->ChildVerifyResultsReady())
        {
            return;
        }

        bool subtreeOk = this->localVerifyOk;
        if(this->child1 < this->size)
        {
            subtreeOk = subtreeOk && (this->childVerifyResult1 != 0);
        }
        if(this->child2 < this->size)
        {
            subtreeOk = subtreeOk && (this->childVerifyResult2 != 0);
        }
        this->FinishVerifyAttempt(subtreeOk);
    }
}

void AmountManager::CheckVerify(void)
{    
    if(this->rank == 0)
    {
        return;
    }
    if(this->verify)
    {
        // avoid duplications
        return;
    }

    int flag;
    MPI_Test(&this->parentVerifyRequest, &flag, MPI_STATUS_IGNORE);
    if(flag)
    {
        this->AskChildrenVerify();
    }
}

void AmountManager::CheckDone(void)
{
    if(this->rank == 0)
    {
        return;
    }
    if(this->done)
    {
        // avoid duplications
        return;
    }
    int flag;
    MPI_Test(&this->parentDoneRequest, &flag, MPI_STATUS_IGNORE);
    if(flag)
    {
        this->MarkChildrenDone();
    }
}

AmountManager::~AmountManager()
{
    if(this->outgoingRequest != MPI_REQUEST_NULL)
    {
        MPI_Wait(&this->outgoingRequest, MPI_STATUS_IGNORE);
        this->outgoingRequest = MPI_REQUEST_NULL;
    }
    CancelAndWait(this->request1);
    CancelAndWait(this->request2);
    CancelAndWait(this->parentDoneRequest);
    CancelAndWait(this->parentVerifyRequest);
    CancelAndWait(this->childVerifyRequest1);
    CancelAndWait(this->childVerifyRequest2);
}

#endif // RICH_MPI
