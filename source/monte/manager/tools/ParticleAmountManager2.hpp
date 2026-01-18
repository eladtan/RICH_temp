#ifndef PARTICLE_AMOUNT_MANAGER2_HPP
#define PARTICLE_AMOUNT_MANAGER2_HPP

#ifdef RICH_MPI

#include <iostream>
#include <cassert>
#include <mpi.h>

class ParticleAmountManager2
{
public:
    using counter_t = long long int;

    ParticleAmountManager2(MPI_Comm comm);

    ~ParticleAmountManager2();
    
    void Initialize(counter_t num);

    void Increase(counter_t n);

    inline void Decrease(counter_t n){this->Increase(-n);};

    void Progress(void);

    void Verify(bool verify);
    
    inline const bool &GetDoneRef(void) const{return this->done;};

    inline const bool &GetVerifyRef(void) const{return this->verify;};

    inline const counter_t &GetValue(void) const{return this->globalNum;};

private:
    void AskChildrenVerify(void);
    
    void MarkChildrenDone(void);

    void CheckDone(void);

    void CheckVerify(void);

    MPI_Comm comm;
    int rank, size;
    size_t cyclesWaiting;
    counter_t globalNum; // current global number
    counter_t tempNum; // number before sent to parent
    counter_t recv1, recv2;
    MPI_Request request1;
    MPI_Request request2;
    MPI_Request parentDoneRequest;
    MPI_Request parentVerifyRequest;
    int parent, child1, child2;
    bool verify;
    bool done;
};

#endif // RICH_MPI

#endif // PARTICLE_AMOUNT_MANAGER2_HPP