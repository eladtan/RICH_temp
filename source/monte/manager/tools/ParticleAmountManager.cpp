#ifdef RICH_MPI

#include "ParticleAmountManager.hpp"

ParticleAmountManager::ParticleAmountManager(MPI_Comm comm)
    : comm(comm)
{
    MPI_Comm_rank(this->comm, &this->rank);
    MPI_Comm_size(this->comm, &this->size);
    this->counter = 0;
    MPI_Win_allocate(1, sizeof(int), MPI_INFO_NULL, this->comm, &this->shouldVerify, &this->shouldVerifyWin);
    *this->shouldVerify = 0; // Initialize to false
    MPI_Win_allocate(1, sizeof(bool), MPI_INFO_NULL, this->comm, &this->done, &this->doneWin);
    *this->done = false; // Initialize to false
    this->request = MPI_REQUEST_NULL;
    this->verifyRequest = MPI_REQUEST_NULL;
    this->destroyed = false;
    MPI_Barrier(this->comm);
}

void ParticleAmountManager::Verify(bool verify)
{
    // std::cout << "Rank " << this->rank << " verifies " << ((verify)? "OK" : "NOT OK") << std::endl;
    assert(*this->shouldVerify);
    // send a commit
    this->dummy = (verify)? 1 : 0;
    if(verify)
    {
        MPI_Isend(&this->dummy, 1, MPI_INT, 0, COMMIT_VERIFY_TAG, this->comm, &this->verifyRequest);
    }
    *this->shouldVerify = 0;
}

void ParticleAmountManager::ReceiveVerifies(void)
{
    assert(this->rank == 0);
    bool can_finish = true;
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        int verify;
        MPI_Recv(&verify, 1, MPI_INT, _rank, COMMIT_VERIFY_TAG, this->comm, MPI_STATUS_IGNORE);
        if(!verify)
        {
            can_finish = false;
        }
    }

    if(can_finish)
    {
        this->MarkAllDone();
    }
}

void ParticleAmountManager::MarkAllDone(void)
{
    static int one = 1;
    assert(this->rank == 0);
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, _rank, MPI_MODE_NOCHECK, this->doneWin);
        MPI_Put(&one, 1, MPI_INT, _rank, 0, 1, MPI_INT, this->doneWin);
        MPI_Win_unlock(_rank, this->doneWin);
    }
}

void ParticleAmountManager::MarkShouldVerify(void)
{
    static int one = 1;
    assert(this->rank == 0);
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        MPI_Win_lock(MPI_LOCK_SHARED, _rank, MPI_MODE_NOCHECK, this->shouldVerifyWin);
        MPI_Put(&one, 1, MPI_INT, _rank, 0, 1, MPI_INT, this->shouldVerifyWin);
        MPI_Win_unlock(_rank, this->shouldVerifyWin);
    }
    // std::cout << "Done sending verify requests" << std::endl;
}

void ParticleAmountManager::CheckToFinish(void)
{
    assert(this->rank == 0);
    int flag = 0;
    do
    {
        MPI_Iprobe(MPI_ANY_SOURCE, INCREASE_TAG, this->comm, &flag, MPI_STATUS_IGNORE);
        if(flag)
        {
            // receive
            int amount;
            MPI_Recv(&amount, 1, MPI_INT, MPI_ANY_SOURCE, INCREASE_TAG, this->comm, MPI_STATUS_IGNORE);
            // std::cout << "Current counter value is " << this->counter << ", increases by " << amount << std::endl;
            this->counter += amount; 
            if(this->counter < 0)
            {
                std::cerr << "Counter cannot be negative after increase (" << amount << "), current value " << this->counter << std::endl;
                MPI_Abort(this->comm, 1);
            }
        }
    } while(flag);

    // std::cout << "Current counter value is " << this->counter << std::endl;
    if(this->counter == 0)
    {
        // first, send verify requests
        this->MarkShouldVerify();
        // std::cout << "Sent verify requests" << std::endl;
    }
}

void ParticleAmountManager::Initialize(int num)
{
    MPI_Reduce(&num, &this->counter, 1, MPI_INT, MPI_SUM, 0, this->comm);
}

void ParticleAmountManager::Increase(int n)
{
    // std::cout << "Rank " << this->rank << " sends an increast message by " << n << std::endl;
    if(this->rank == 0)
    {
        this->counter += n;
        this->CheckToFinish();
        return;
    }
    if(this->request != MPI_REQUEST_NULL)
    {
        // can't send, wait to completion
        MPI_Wait(&this->request, MPI_STATUS_IGNORE);
        this->request = MPI_REQUEST_NULL;
    }
    this->tmpValue = n;
    MPI_Isend(&this->tmpValue, 1, MPI_INT, 0, INCREASE_TAG, this->comm, &this->request);
}

ParticleAmountManager::~ParticleAmountManager()
{
    this->Destroy();
}

void ParticleAmountManager::Destroy(void)
{
    if(this->destroyed)
    {
        return;
    }
    MPI_Win_free(&this->shouldVerifyWin);
    MPI_Win_free(&this->doneWin);
    MPI_Barrier(this->comm);
    this->destroyed = true;
}

#endif // RICH_MPI
