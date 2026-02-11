#include "ParticleAmountManager2.hpp"

#ifdef RICH_MPI

#define INCREASE_TAG 9918
#define DONE_TAG 9919
#define VERIFY_TAG 9920

ParticleAmountManager2::ParticleAmountManager2(MPI_Comm comm)
    : comm(comm), cyclesWaiting(0), globalNum(0), tempNum(0), done(false), verify(false)
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

void ParticleAmountManager2::Initialize(counter_t num)
{
    MPI_Reduce(&num, &this->globalNum, 1, MPI_LONG_LONG, MPI_SUM, 0, this->comm);
}

void ParticleAmountManager2::Increase(counter_t n)
{
    this->tempNum += n;
}

void ParticleAmountManager2::Progress(void)
{
    this->CheckVerify();
    this->CheckDone();
    MPI_Status status;
    int flag;

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
        if(this->cyclesWaiting % 30 == 0 and this->tempNum != 0)
        {
            MPI_Send(&this->tempNum, 1, MPI_LONG_LONG, this->parent, INCREASE_TAG, this->comm);
            this->tempNum = 0;
        }
    }

    if(this->rank == 0 and this->globalNum == 0)
    {
        this->AskChildrenVerify();
        return;
    }
}

void ParticleAmountManager2::MarkChildrenDone(void)
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

void ParticleAmountManager2::AskChildrenVerify(void)
{
    if(this->verify)
    {
        return;
    }
    if(this->child1 < this->size)
    {
        MPI_Send(MPI_BOTTOM, 0, MPI_INT, this->child1, VERIFY_TAG, this->comm);
    }
    if(this->child2 < this->size)
    {
        MPI_Send(MPI_BOTTOM, 0, MPI_INT, this->child2, VERIFY_TAG, this->comm);
    }
    this->verify = true;
}

void ParticleAmountManager2::Verify(bool ok)
{
    if(this->verify)
    {
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_CXX_BOOL, MPI_LAND, this->comm);
        if(ok)
        {
            this->done = true;
        }
        this->verify = false;
        MPI_Irecv(MPI_BOTTOM, 0, MPI_INT, this->parent, VERIFY_TAG, this->comm, &this->parentVerifyRequest);
    }
}

void ParticleAmountManager2::CheckVerify(void)
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

void ParticleAmountManager2::CheckDone(void)
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

ParticleAmountManager2::~ParticleAmountManager2()
{
    if(this->request1 != MPI_REQUEST_NULL)
    {
        MPI_Cancel(&this->request1);
    }
    if(this->request2 != MPI_REQUEST_NULL)
    {
        MPI_Cancel(&this->request2);
    }
    if(this->parentDoneRequest != MPI_REQUEST_NULL)
    {
        MPI_Cancel(&this->parentDoneRequest);
    }
    if(this->parentVerifyRequest != MPI_REQUEST_NULL)
    {
        MPI_Cancel(&this->parentVerifyRequest);
    }
}

#endif // RICH_MPI