#ifndef AMOUNT_MANAGER_HPP
#define AMOUNT_MANAGER_HPP

#ifdef RICH_MPI

#include <iostream>
#include <cassert>
#include <mpi.h>

class AmountManager
{
public:
    using counter_t = long long int;

    AmountManager(MPI_Comm comm, size_t flushInterval = 5);

    ~AmountManager();
    
    void Initialize(counter_t num);

    void Increase(counter_t n);

    inline void Decrease(counter_t n){this->Increase(-n);};

    inline void SetFlushInterval(size_t interval){this->flushInterval = interval;};

    void Progress(void);

    void Verify(bool ok);
    
    inline const bool &GetDoneRef(void) const{return this->done;};

    inline const bool &GetVerifyRef(void) const{return this->verify;};

    inline const counter_t &GetValue(void) const{return this->globalNum;};

    inline bool HasPending(void) const{return this->tempNum != 0 || this->outgoingRequest != MPI_REQUEST_NULL;};

    inline counter_t GetPendingValue(void) const{return this->HasPending() ? ((this->tempNum != 0) ? this->tempNum : 1) : 0;};
    
private:
    void AskChildrenVerify(void);
    
    void MarkChildrenDone(void);

    void CheckDone(void);

    void CheckVerify(void);

    void FlushToParent(void);

    void ResetVerifyAttempt(void);

    bool ChildVerifyResultsReady(void);

    void FinishVerifyAttempt(bool subtreeOk);

    MPI_Comm comm;
    int rank, size;
    size_t cyclesWaiting;
    size_t flushInterval;
    counter_t globalNum; // current global number
    counter_t tempNum; // number before sent to parent
    counter_t outgoingNum; // number currently being sent to parent
    counter_t recv1, recv2;
    int childVerifyResult1;
    int childVerifyResult2;
    MPI_Request request1;
    MPI_Request request2;
    MPI_Request parentDoneRequest;
    MPI_Request parentVerifyRequest;
    MPI_Request outgoingRequest;
    MPI_Request childVerifyRequest1;
    MPI_Request childVerifyRequest2;
    int parent, child1, child2;
    bool localVerifyRecorded;
    bool localVerifyOk;
    bool verify;
    bool done;
};

#endif // RICH_MPI

#endif // AMOUNT_MANAGER_HPP
