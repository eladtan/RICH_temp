#ifdef RICH_MPI

#include "ParticleAmountManager.hpp"

ParticleAmountManager::ParticleAmountManager(MPI_Comm comm, bool withRDMA)
    : comm(comm), withRDMA(withRDMA)
{
    MPI_Comm_rank(this->comm, &this->rank);
    MPI_Comm_size(this->comm, &this->size);
    this->counter = 0;
    this->timesSentVerifies = 0;
    this->doneVerifyCycle = false;

    if(this->withRDMA)
    {
        MPI_Win_allocate(1, sizeof(int), MPI_INFO_NULL, this->comm, &this->shouldVerify, &this->shouldVerifyWin);
        MPI_Win_allocate(1, sizeof(bool), MPI_INFO_NULL, this->comm, &this->done, &this->doneWin);
    }
    else
    {
        this->requests = std::vector<MPI_Request>(this->size, MPI_REQUEST_NULL);
        this->done = new int(0);
        this->shouldVerify = new int(0);
        MPI_Irecv(MPI_BOTTOM, 0, MPI_INT, 0, ASK_COMMIT_TAG, this->comm, &this->askCommitRequest);
        MPI_Irecv(MPI_BOTTOM, 0, MPI_INT, 0, MARK_DONE_TAG, this->comm, &this->markDoneRequest);
    }

    this->Reset();
}

void ParticleAmountManager::Reset(void)
{
    *this->shouldVerify = 0; // Initialize to false
    *this->done = false; // Initialize to false
    this->verifyRequest = MPI_REQUEST_NULL;
    this->askCommitRequest = MPI_REQUEST_NULL;
    this->markDoneRequest = MPI_REQUEST_NULL;
    this->doneVerifyCycle = false;
    this->destroyed = false;
    this->increaseRequests.clear();
    this->tmpValues.clear();
    this->requests.clear();
}

void ParticleAmountManager::Verify(bool verify)
{
    // std::cout << "Rank " << this->rank << " verifies " << ((verify)? "OK" : "NOT OK") << std::endl;
    assert(*this->shouldVerify);
    // send a commit
    if(this->verifyRequest != MPI_REQUEST_NULL)
    {
        // make sure we can use `this->dummy` safely
        MPI_Wait(&this->verifyRequest, MPI_STATUS_IGNORE);
    }

    this->dummy = (verify)? 1 : 0;
    MPI_Isend(&this->dummy, 1, MPI_INT, 0, COMMIT_VERIFY_TAG, this->comm, &this->verifyRequest);

    *this->shouldVerify = 0;
    if(not this->withRDMA)
    {
        MPI_Irecv(MPI_BOTTOM, 0, MPI_INT, 0, ASK_COMMIT_TAG, this->comm, &this->askCommitRequest);
    }
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
            // std::cout << "Can't finish, got " << verify << " from rank " << _rank << std::endl;
            can_finish = false;
            this->doneVerifyCycle = false;
        }
    }

    if(can_finish)
    {
        // std::cout << "Can finish." << std::endl;
        this->MarkAllDone();
    }
}

void ParticleAmountManager::Progress(void)
{
    if(this->withRDMA)
    {
        return; // function is unused
    }

    int flag;
    MPI_Test(&this->askCommitRequest, &flag, MPI_STATUS_IGNORE);
    if(flag)
    {
        *this->shouldVerify = 1;
    }

    MPI_Test(&this->markDoneRequest, &flag, MPI_STATUS_IGNORE);
    if(flag)
    {
        *this->done = 1;
    }
}

void ParticleAmountManager::MarkAllDone(void)
{
    static int one = 1;
    assert(this->rank == 0);
    if(this->withRDMA)
    {
        for(int _rank = 0; _rank < this->size; _rank++)
        {
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, _rank, MPI_MODE_NOCHECK, this->doneWin);
            MPI_Put(&one, 1, MPI_INT, _rank, 0, 1, MPI_INT, this->doneWin);
            MPI_Win_unlock(_rank, this->doneWin);
        }
    }
    else
    {
        MPI_Waitall(this->requests.size(), this->requests.data(), MPI_STATUSES_IGNORE);
        for(int _rank = 0; _rank < this->size; _rank++)
        {
            MPI_Isend(MPI_BOTTOM, 0, MPI_INT, _rank, MARK_DONE_TAG, this->comm, &this->requests[_rank]);
        }
    }
}

void ParticleAmountManager::MarkShouldVerify(void)
{
    static int one = 1;
    if(this->doneVerifyCycle)
    {
        // don't send again!
        return;
    }
    this->timesSentVerifies++;
    assert(this->rank == 0);
    if(this->withRDMA)
    {
        for(int _rank = 0; _rank < this->size; _rank++)
        {
            MPI_Win_lock(MPI_LOCK_SHARED, _rank, MPI_MODE_NOCHECK, this->shouldVerifyWin);
            MPI_Put(&one, 1, MPI_INT, _rank, 0, 1, MPI_INT, this->shouldVerifyWin);
            MPI_Win_unlock(_rank, this->shouldVerifyWin);
        }
    }
    else
    {
        MPI_Waitall(this->requests.size(), this->requests.data(), MPI_STATUSES_IGNORE);
        for(int _rank = 0; _rank < this->size; _rank++)
        {
            MPI_Isend(MPI_BOTTOM, 0, MPI_INT, _rank, ASK_COMMIT_TAG, this->comm, &this->requests[_rank]);
        }
        // std::cout << "Sent verifies to all ranks, for the " << this->timesSentVerifies << "-th time" << std::endl;
    }
    this->doneVerifyCycle = true;
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
            MPI_Status status;
            MPI_Recv(&amount, 1, MPI_INT, MPI_ANY_SOURCE, INCREASE_TAG, this->comm, &status);
            // std::cout << "Current counter value is " << this->counter << ", increases by " << amount << std::endl;
            this->counter += amount; 
            // std::cout << "Rank " << status.MPI_SOURCE << " increased counter by " << amount << " (current: " << this->counter << ", initial: " << this->initialValue << ")" << std::endl;
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

void ParticleAmountManager::Initialize(int64_t num)
{
    this->Reset();
    MPI_Reduce(&num, &this->counter, 1, MPI_INT64_T, MPI_SUM, 0, this->comm);
    this->initialValue = this->counter;
}

void ParticleAmountManager::Increase(int n)
{
    // std::cout << "Rank " << this->rank << " sends an increast message by " << n << std::endl;
    if(this->rank == 0)
    {
        this->counter += n;
        // std::cout << "Self Increased counter by " << n << " (current: " << this->counter << ", initial: " << this->initialValue << ")" << std::endl;
        this->CheckToFinish();
        return;
    }

    MPI_Request &request = increaseRequests.emplace_back(MPI_REQUEST_NULL);
    this->tmpValues.push_back(std::vector<int>(1, n));

    // std::cout << "Rank " << this->rank << " sends a message to increase by " << this->tmpValues.back() << std::endl;
    MPI_Isend(this->tmpValues.back().data(), 1, MPI_INT, 0, INCREASE_TAG, this->comm, &request);
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
    if(this->withRDMA)
    {
        MPI_Win_free(&this->shouldVerifyWin);
        MPI_Win_free(&this->doneWin);
    }
    else
    {
        MPI_Wait(&this->verifyRequest, MPI_STATUS_IGNORE);
        MPI_Waitall(this->increaseRequests.size(), this->increaseRequests.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(this->requests.size(), this->requests.data(), MPI_STATUSES_IGNORE);
        MPI_Cancel(&this->askCommitRequest);
        delete this->shouldVerify;
        delete this->done;
    }
    MPI_Barrier(this->comm);
    this->destroyed = true;
}

#endif // RICH_MPI
