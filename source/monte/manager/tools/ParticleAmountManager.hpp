#ifndef PARTICLES_AMOUNT_MANAGER_HPP
#define PARTICLES_AMOUNT_MANAGER_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include <iostream>
#include <cassert>

#define INCREASE_TAG 2018
#define VERIFY_TAG 2019
#define COMMIT_VERIFY_TAG 2020

class ParticleAmountManager
{
public:
    ParticleAmountManager(MPI_Comm comm);

    ~ParticleAmountManager();

    void Destroy(void);

    void Initialize(int num);

    void Increase(int n);

    inline void Decrease(int n){this->Increase(-n);};

    inline bool ShouldVerify(void) const{return *this->shouldVerify;};

    void Verify(bool done);

    int *shouldVerify;
    int *done;

    void CheckToFinish(void);

    void ReceiveVerifies(void);

private:
    MPI_Comm comm;
    int rank, size;
    int counter;
    MPI_Win shouldVerifyWin;
    MPI_Win doneWin;
    
    int dummy;
    MPI_Request verifyRequest;

    int tmpValue;
    MPI_Request request;

    bool destroyed;

    void SendVerifies(void);
    
    void MarkAllDone(void);

    void MarkShouldVerify(void);
};

#endif // RICH_MPI

#endif // PARTICLES_AMOUNT_MANAGER_HPP