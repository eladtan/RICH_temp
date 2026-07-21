#ifndef PARTICLES_AMOUNT_MANAGER_HPP
#define PARTICLES_AMOUNT_MANAGER_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>

#define INCREASE_TAG 2018
#define VERIFY_TAG 2019
#define COMMIT_VERIFY_TAG 2020
#define ASK_COMMIT_TAG 2021
#define MARK_DONE_TAG 2022

class ParticleAmountManager
{
public:
    ParticleAmountManager(MPI_Comm comm, bool withRDMA);

    ~ParticleAmountManager();

    void Reset(void);
    
    void Destroy(void);

    void Initialize(int64_t num);

    void Increase(int n);

    inline void Decrease(int n){this->Increase(-n);};

    inline bool ShouldVerify(void) const{return *this->shouldVerify;};

    void Verify(bool done);

    void Progress(void);

    int *shouldVerify;
    int *done;

    void CheckToFinish(void);

    void ReceiveVerifies(void);

    inline int64_t GetCounter(void) const{return this->counter;};

private:
    MPI_Comm comm;
    int rank, size;
    int64_t counter;
    int64_t initialValue;
    MPI_Win shouldVerifyWin;
    MPI_Win doneWin;
    bool withRDMA;
    size_t timesSentVerifies;
    bool doneVerifyCycle;
    std::vector<MPI_Request> increaseRequests;

    std::vector<std::vector<int>> tmpValues;
    int dummy;
    MPI_Request verifyRequest;
    
    int tmpValue;

    bool destroyed;
    
    // for non-RDMA mode
    MPI_Request askCommitRequest, markDoneRequest;
    std::vector<MPI_Request> requests; 
    
    void MarkAllDone(void);

    void MarkShouldVerify(void);
};

#endif // RICH_MPI

#endif // PARTICLES_AMOUNT_MANAGER_HPP