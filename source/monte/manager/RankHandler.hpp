#ifndef MONTECARLO_RANK_HANDLER_HPP
#define MONTECARLO_RANK_HANDLER_HPP

#ifdef RICH_MPI

#include <vector>
#include <memory>
#include <iostream>
#include <mpi.h>
#include "mpi/mpi_commands.hpp"
#include "tools/DistributedMutex.hpp"
#include "tools/ConditionVariable.hpp"
#include "ReallocationAgent.hpp"

#define BUFFER_REALLOCATION_FACTOR 2 // 1.618033 // golden ratio
#define BUFFER_SHRINK_FACTOR 0.5
#define MPI_INDEX_T MPI_UINT32_T

template<typename T, typename Grid>
class RankHandler
{
public:
    using index_t = uint32_t;
    using MCParticle = MonteCarloParticle<T, Grid>;

    RankHandler(size_t buffsize, const MPI_Comm &comm_world, const MPI_Comm &private_comm, std::shared_ptr<ReallocationAgent> &reallocationAgent);
    
    ~RankHandler();
    
    void TransferParticles(const std::vector<MCParticle> &particles);

    void RemoveParticles(const std::vector<size_t> &indicesInToHandle, size_t num);

    void Sync(void);

    void Reset(void);

    void Destroy(void);

    MPI_Comm comm_world, comm;
    rank_t rank_world, rank_internal;
    rank_t size_world, size_internal;
    rank_t other_rank, peer_rank_world;

    size_t buffsize, peer_buffsize;
    MCParticle *particles;
    index_t *av;
    volatile int *av_length;
    index_t *th;
    volatile int *th_length; 

    std::shared_ptr<ReallocationAgent> &reallocationAgent;
    
    void Reallocate(double factor);

private:
    MPI_Win particles_win;
    MPI_Win av_win;
    MPI_Win th_win;
    MPI_Win av_length_win;
    MPI_Win th_length_win;
    std::shared_ptr<DistributedMutex> localTHMutex;
    std::shared_ptr<DistributedMutex> remoteTHMutex;
    bool destroyed;
};

template<typename T, typename Grid>
RankHandler<T, Grid>::RankHandler(size_t buffsize, const MPI_Comm &comm_world, const MPI_Comm &private_comm, std::shared_ptr<ReallocationAgent> &reallocationAgent):
    comm_world(comm_world), comm(private_comm), buffsize(buffsize), particles_win(MPI_WIN_NULL),
    av_win(MPI_WIN_NULL), th_win(MPI_WIN_NULL), av_length_win(MPI_WIN_NULL), th_length_win(MPI_WIN_NULL),
    destroyed(false), reallocationAgent(reallocationAgent)
{
    assert(private_comm != MPI_COMM_NULL);

    MPI_Comm_rank(this->comm_world, &this->rank_world);
    MPI_Comm_size(this->comm_world, &this->size_world);
    MPI_Comm_rank(this->comm, &this->rank_internal);
    MPI_Comm_size(this->comm, &this->size_internal);
    
    assert(this->size_internal == 2 or this->size_internal == 1);
    assert(this->rank_internal == 0 or this->rank_internal == 1);

    this->other_rank = (this->rank_internal == 0)? 1 : 0;
    
    if(this->size_internal > 1)
    {
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "accumulate_ordering", "none"); // No strict ordering
        MPI_Info_set(info, "accumulate_ops", "same_op");
        MPI_Info_set(info, "same_disp_unit", "true");    
        // initialize windows

        auto reportErrorAndExit = [](const std::string &str, int err)
        {
            if(err == MPI_SUCCESS)
            {
                return;
            }
            char error_string[MPI_MAX_ERROR_STRING];
            int length_of_error_string;
            MPI_Error_string(err, error_string, &length_of_error_string);
            std::cerr << "Error: " << str << ": " << error_string << std::endl;
            exit(1);
        };

        int retval;
        retval = MPI_Win_allocate(this->buffsize * sizeof(MCParticle), sizeof(MCParticle), info, this->comm, &this->particles, &this->particles_win);
        reportErrorAndExit("MPI_Win_allocate for particles", retval);
        assert(this->particles != nullptr);
        retval = MPI_Win_allocate(this->buffsize * sizeof(index_t), sizeof(index_t), info, this->comm, &this->av, &this->av_win);
        reportErrorAndExit("MPI_Win_allocate for av", retval);
        assert(this->av != nullptr);
        retval = MPI_Win_allocate(this->buffsize * sizeof(index_t), sizeof(index_t), info, this->comm, &this->th, &this->th_win);
        reportErrorAndExit("MPI_Win_allocate for th", retval);
        assert(this->th != nullptr);
        retval = MPI_Win_allocate(sizeof(int), sizeof(int), info, this->comm, static_cast<void*>(&this->av_length), &this->av_length_win);
        reportErrorAndExit("MPI_Win_allocate for av length", retval);
        assert(this->av_length != nullptr);
        retval = MPI_Win_allocate(sizeof(int), sizeof(int), info, this->comm, static_cast<void*>(&this->th_length), &this->th_length_win);
        reportErrorAndExit("MPI_Win_allocate for th length", retval);
        assert(this->th_length != nullptr);
        MPI_Info_free(&info);

        MPI_Win windows[5] = {this->particles_win, this->av_win, this->th_win, this->av_length_win, this->th_length_win};
        for(int i = 0; i < 5; i++)
        {
            MPI_Win_set_errhandler(windows[i], MPI_ERRORS_RETURN);
            int *model, flag;
            MPI_Win_get_attr(windows[i], MPI_WIN_MODEL, &model, &flag);
            if(*model == MPI_WIN_SEPARATE)
            {
                std::cout << "MPI is using WIN_SEPARATE (" << MPI_WIN_SEPARATE << "). Can not continue" << std::endl;
                exit(1);
            }
        }
        
        // initialize mutexes
        std::shared_ptr<DistributedMutex> rank0Mutex = std::make_shared<DistributedMutex>(comm, 0);
        std::shared_ptr<DistributedMutex> rank1Mutex = std::make_shared<DistributedMutex>(comm, 1);
        this->localTHMutex = (this->rank_internal == 0)? rank0Mutex : rank1Mutex;
        this->remoteTHMutex = (this->rank_internal == 0)? rank1Mutex : rank0Mutex;

        this->Reset();
        MPI_Barrier(this->comm);
    }
    else
    {
        // initialized from outside
        this->particles = new MCParticle[this->buffsize];
        this->av = new index_t[this->buffsize];
        this->th = new index_t[this->buffsize];
        this->av_length = new int(this->buffsize);
        this->th_length = new int(0);
        std::iota(this->av, this->av + this->buffsize, 0);
    }

    if(this->size_internal > 1)
    {
        MPI_Sendrecv(&this->buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0, &this->peer_buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0, this->comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&this->rank_world, 1, MPI_INT, this->other_rank, 0, &this->peer_rank_world, 1, MPI_INT, this->other_rank, 0, this->comm, MPI_STATUS_IGNORE);
    }
    else
    {
        this->peer_buffsize = this->buffsize;
        this->peer_rank_world = this->rank_world;
    }

    MPI_Barrier(this->comm);
}

template<typename T, typename Grid>
void RankHandler<T, Grid>::Sync(void)
{
    if(this->size_internal == 1)
    {
        return;
    }
    MPI_Win_lock_all(MPI_MODE_NOCHECK, this->particles_win);
    MPI_Win_sync(this->particles_win);
    MPI_Win_unlock_all(this->particles_win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, this->av_win);
    MPI_Win_sync(this->av_win);
    MPI_Win_unlock_all(this->av_win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, this->th_win);
    MPI_Win_sync(this->th_win);
    MPI_Win_unlock_all(this->th_win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, this->av_length_win);
    MPI_Win_sync(this->av_length_win);
    MPI_Win_unlock_all(this->av_length_win);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, this->th_length_win);
    MPI_Win_sync(this->th_length_win);
    MPI_Win_unlock_all(this->th_length_win);
    // sync distributed mutex
    this->localTHMutex->Sync();
    this->remoteTHMutex->Sync();
}

template<typename T, typename Grid>
void RankHandler<T, Grid>::Reset(void)
{
    *this->av_length = static_cast<int>(this->buffsize);
    *this->th_length = 0;
    std::iota(this->av, this->av + this->buffsize, 0);
}

template<typename T, typename Grid>
void RankHandler<T, Grid>::Destroy(void)
{
    if(this->destroyed)
    {
        return;
    }

    #ifdef RICH_MPI
    if(this->size_internal > 1)
    {
        MPI_Win_free(&this->particles_win);
        MPI_Win_free(&this->av_win);
        MPI_Win_free(&this->th_win);
        MPI_Win_free(&this->av_length_win);
        MPI_Win_free(&this->th_length_win);
        DistributedMutex *mutex1 = (this->rank_internal == 0)? this->localTHMutex.get() : this->remoteTHMutex.get();
        DistributedMutex *mutex2 = (this->rank_internal == 1)? this->localTHMutex.get() : this->remoteTHMutex.get();
        mutex1->Destroy();
        mutex2->Destroy();
    }
    else
    {
    #endif // RICH_MPI
        delete[] this->particles;
        delete[] this->av;
        delete[] this->th;
        delete this->av_length;
        delete this->th_length;
    #ifdef RICH_MPI
    }
    #endif // RICH_MPI
    this->destroyed = true;
}

template<typename T, typename Grid>
RankHandler<T, Grid>::~RankHandler()
{
    if(not this->destroyed)
    {
        this->Destroy();
    }
}

template<typename T, typename Grid>
void RankHandler<T, Grid>::RemoveParticles(const std::vector<size_t> &indicesInToHandle, size_t num)
{
    if(indicesInToHandle.empty())
    {
        return;
    }
    if(this->size_internal > 1)
    {
        // lock self mutex
        this->localTHMutex->Lock();
    }
    
    // std::cout << "Rank " << this->rank_world << " removes particles " << indicesInToHandle << " from rank " << this->peer_rank_world << "'s buffer" << std::endl;
    volatile int &th_length = *this->th_length;
    volatile int &av_length = *this->av_length;

    this->Sync();
    for(int i = static_cast<int>(num) - 1; i >= 0; i--)
    {
        const size_t &toHandleIndex = indicesInToHandle[i];
        assert(i == 0 or indicesInToHandle[i] > indicesInToHandle[i-1]); // should be in a descending order
        // std::cout << "Rank " << this->rank_world << " removes particle " << toHandleIndex << " from handler of rank " << this->peer_rank_world << std::endl;
        assert(toHandleIndex < th_length);
        index_t particleIdx = this->th[toHandleIndex];
        assert(av_length < this->buffsize);
        this->av[av_length++] = particleIdx;
        this->th[toHandleIndex] = this->th[--th_length];
        assert(th_length >= 0);
    }    
    this->Sync();

    if(this->size_internal > 1)
    {
        // release self mutex
        this->localTHMutex->Unlock();
    }
}

template<typename T, typename Grid>
void RankHandler<T, Grid>::Reallocate(double factor)
{
    size_t newBuffSize = std::ceil(this->buffsize * factor); // ceil is necessary to avoid 0
    size_t oldBuffSize = this->buffsize;
    if(oldBuffSize > newBuffSize)
    {
        if(*this->av_length != this->buffsize)
        {
            std::cerr << "Can not shrink memory when there are particles" << std::endl;
            exit(1);
        }
    }
    size_t peerNewBuffSize = std::ceil(this->peer_buffsize * factor);
    if(newBuffSize < 10 or peerNewBuffSize < 10)
    {
        return;
    }

    // start reallocation
    this->buffsize = newBuffSize;

    if(this->size_internal > 1)
    {
        assert(this->size_internal == 2);
        
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "accumulate_ordering", "none"); // No strict ordering
        MPI_Info_set(info, "accumulate_ops", "same_op");
        MPI_Info_set(info, "same_disp_unit", "true");    
        // initialize windows

        auto reportErrorAndExit = [](const std::string &str, int err)
        {
            if(err == MPI_SUCCESS)
            {
                return;
            }
            char error_string[MPI_MAX_ERROR_STRING];
            int length_of_error_string;
            MPI_Error_string(err, error_string, &length_of_error_string);
            std::cerr << "Error: " << str << ": " << error_string << std::endl;
            exit(1);
        };

        // std::cout << "Allocating size " << this->buffsize << std::endl;
        
        MPI_Win new_particles_win;
        MCParticle *new_particles;
        int err = MPI_Win_allocate(this->buffsize * sizeof(MCParticle), sizeof(MCParticle), info, this->comm, &new_particles, &new_particles_win);
        reportErrorAndExit("MPI_Win_allocate new_particles with buffsize = " + std::to_string(this->buffsize), err);
        assert(new_particles != nullptr);

        MPI_Win new_av_win;
        index_t *new_av;
        err = MPI_Win_allocate(this->buffsize * sizeof(index_t), sizeof(index_t), info, this->comm, &new_av, &new_av_win);
        reportErrorAndExit("MPI_Win_allocate new_av with buffsize = " + std::to_string(this->buffsize), err);
        assert(new_av != nullptr);

        MPI_Win new_th_win;
        index_t *new_th;
        err = MPI_Win_allocate(this->buffsize * sizeof(index_t), sizeof(index_t), info, this->comm, &new_th, &new_th_win);
        reportErrorAndExit("MPI_Win_allocate new_th with buffsize = " + std::to_string(this->buffsize), err);
        assert(new_th != nullptr);

        if(this->buffsize >= oldBuffSize)
        {
            std::memcpy(new_particles, this->particles, oldBuffSize * sizeof(MCParticle));
            std::memcpy(new_th, this->th, *this->th_length * sizeof(index_t));
            size_t difference = this->buffsize - oldBuffSize;
            std::memcpy(new_av + difference, this->av, oldBuffSize * sizeof(index_t));
            std::iota(new_av, new_av + difference, oldBuffSize);
        }
        else
        {
            std::memcpy(new_particles, this->particles, this->buffsize * sizeof(MCParticle));
            std::memcpy(new_av, this->av, this->buffsize * sizeof(index_t));
            std::memcpy(new_th, this->th, this->buffsize * sizeof(index_t));
            std::iota(new_av, new_av + this->buffsize, 0);
        }

        err = MPI_Win_free(&this->particles_win);
        reportErrorAndExit("MPI_Win_free particles_win", err);
        err = MPI_Win_free(&this->av_win);
        reportErrorAndExit("MPI_Win_free av_win", err);
        err = MPI_Win_free(&this->th_win);
        reportErrorAndExit("MPI_Win_free th_win", err);

        this->particles_win = new_particles_win;
        this->av_win = new_av_win;
        this->th_win = new_th_win;

        MPI_Win windows[3] = {this->particles_win, this->av_win, this->th_win};
        for(int i = 0; i < 3; i++)
        {
            // MPI_Win_set_errhandler(windows[i], MPI_ERRORS_RETURN);
            int *model, flag;
            err = MPI_Win_get_attr(windows[i], MPI_WIN_MODEL, &model, &flag);
            reportErrorAndExit("MPI_Win_get_attr for " + std::to_string(i), err);
            if(*model == MPI_WIN_SEPARATE)
            {
                std::cout << "MPI is using WIN_SEPARATE (" << MPI_WIN_SEPARATE << "). Can not continue" << std::endl;
                exit(1);
            }
        }

        // std::cout << "Rank " << this->rank_world << " reallocated " << oldBuffSize << " to " << this->buffsize << std::endl;
        this->particles = new_particles;
        this->av = new_av;
        this->th = new_th;

        if(this->buffsize >= oldBuffSize)
        {
            int difference_int = static_cast<int>(this->buffsize - oldBuffSize);            
            MPI_Win_lock(MPI_LOCK_SHARED, this->rank_internal, MPI_MODE_NOCHECK, this->av_length_win);
            MPI_Accumulate(&difference_int, 1, MPI_INT, this->rank_internal, 0, 1, MPI_INT, MPI_SUM, this->av_length_win);
            MPI_Win_unlock(this->rank_internal, this->av_length_win);
        }
        
        MPI_Sendrecv(&this->buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0, &this->peer_buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0, this->comm, MPI_STATUS_IGNORE);

        // std::cout << "Rank " << this->rank_world << " AV old window: " << oldwin << ", new window: " << new_av_win << " (peer size: " << this->peer_buffsize << ")" << std::endl;
        MPI_Barrier(this->comm); // unnecessary because of the sendrecv above
    }
    else
    {        
        this->buffsize = newBuffSize;

        MCParticle *new_particles = new MCParticle[this->buffsize];
        index_t *new_av = new typename RankHandler::index_t[this->buffsize];
        index_t *new_th = new typename RankHandler::index_t[this->buffsize];

        if(this->buffsize >= oldBuffSize)
        {
            std::memcpy(new_particles, this->particles, oldBuffSize * sizeof(MCParticle));
            std::memcpy(new_th, this->th, *this->th_length * sizeof(index_t));
            size_t difference = this->buffsize - oldBuffSize;
            std::memcpy(new_av + difference, this->av, oldBuffSize * sizeof(index_t));
            std::iota(new_av, new_av + difference, oldBuffSize);
            int difference_int = difference;
            *this->av_length += difference_int;
        }
        else
        {
            std::memcpy(new_particles, this->particles, this->buffsize * sizeof(MCParticle));
            std::memcpy(new_av, this->av, this->buffsize * sizeof(index_t));
            std::memcpy(new_th, this->th, this->buffsize * sizeof(index_t));
            std::iota(new_av, new_av + this->buffsize, 0);
        }

        delete[] this->particles;
        delete[] this->av;
        delete[] this->th;
        this->particles = new_particles;
        this->av = new_av;
        this->th = new_th;

        this->peer_buffsize = this->buffsize;
    }
}

template<typename T, typename Grid>
void RankHandler<T, Grid>::TransferParticles(const std::vector<MCParticle> &particles)
{
    // static int minus_one = -1;
    // static int plus_one = 1;
    size_t Np = particles.size();
    int decrement = -static_cast<int>(Np);
    int increment = static_cast<int>(Np);
    if(particles.empty())
    {
        return;
    }

    if(this->size_internal > 1)
    {
        // acquire remote mutex
        this->remoteTHMutex->Lock();

        auto getAvailableLength = [&](void)
        {
            int availLength;
            MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->av_length_win);
            // todo: get to fetch_and_op
            int retval = MPI_Fetch_and_op(&decrement, &availLength, MPI_INT, this->other_rank, 0, MPI_SUM, this->av_length_win);
            assert(retval == MPI_SUCCESS);
            MPI_Win_flush(this->other_rank, this->av_length_win);
            assert(availLength <= this->peer_buffsize);
            if(availLength < Np)
            {
                // increment back, a request to reallocate will be made
                MPI_Accumulate(&increment, 1, MPI_INT, this->other_rank, 0, 1, MPI_INT, MPI_SUM, this->av_length_win);
            }
            MPI_Win_unlock(this->other_rank, this->av_length_win);
            assert(availLength >= 0);
            return availLength;
        };

        int availLength = getAvailableLength();
        while(availLength < Np)
        {
            this->remoteTHMutex->Unlock();
            this->reallocationAgent->RequestReallocation(this->peer_rank_world);
            availLength = getAvailableLength();
            this->remoteTHMutex->Lock();
        }
        assert(availLength >= Np);

        // get particle empty index from avail list
        std::vector<index_t> availIndices(Np);
        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->av_win);
        int retval = MPI_Get(availIndices.data(), Np, MPI_INDEX_T, this->other_rank, availLength - Np, Np, MPI_INDEX_T, this->av_win);
        assert(retval == MPI_SUCCESS);
        MPI_Win_unlock(this->other_rank, this->av_win);

        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->particles_win);
        for(size_t i = 0; i < Np; i++)
        {
            size_t availIndex = availIndices[i];
            assert(availIndex < this->peer_buffsize);
            assert(i < Np);
            const MCParticle *particle = &particles[i];
            /* put particle itself */
            retval = MPI_Put(particle, sizeof(MCParticle), MPI_BYTE, this->other_rank, availIndex, sizeof(MCParticle), MPI_BYTE, this->particles_win);
            assert(retval == MPI_SUCCESS);
        }
        MPI_Win_flush(this->other_rank, this->particles_win);
        MPI_Win_unlock(this->other_rank, this->particles_win);

        // MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, 0, this->av_length_win);
        // MPI_Accumulate(&minus_one, 1, MPI_INT, this->other_rank, 0, 1, MPI_INT, MPI_SUM, this->av_length_win);
        // MPI_Win_unlock(this->other_rank, this->av_length_win);

        /* put in to handle list */

        // get length of to handle list
        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->th_length_win);
        int toHandleLength;
        retval = MPI_Get(&toHandleLength, 1, MPI_INT, this->other_rank, 0, 1, MPI_INT, this->th_length_win);
        assert(retval == MPI_SUCCESS);
        MPI_Win_unlock(this->other_rank, this->th_length_win);
        assert(toHandleLength >= 0);
        assert(toHandleLength < this->peer_buffsize);

        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->th_win);
        MPI_Put(availIndices.data(), Np, MPI_INDEX_T, this->other_rank, toHandleLength, Np, MPI_INDEX_T, this->th_win);
        MPI_Win_unlock(this->other_rank, this->th_win);

        // add 1 to to handle index
        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->th_length_win);
        MPI_Accumulate(&increment, 1, MPI_INT, this->other_rank, 0, 1, MPI_INT, MPI_SUM, this->th_length_win);
        MPI_Win_unlock(this->other_rank, this->th_length_win);

        // release remote mutex
        this->remoteTHMutex->Unlock();

        // std::cout << "In sending of from rank " << this->rank_world << " to rank " << this->peer_rank_world << ", avail length of peer is " << availLength << ", and to handle " << toHandleLength << std::endl;

    }
    else
    {
        for(size_t i = 0; i < Np; i++)
        {
            assert(*this->av_length > 0);
            size_t availIndex = this->av[--(*this->av_length)];
            this->particles[availIndex] = particles[i];
            this->th[(*this->th_length)++] = availIndex;
            assert(*this->th_length < this->buffsize);
        }
    }
}

#endif // RICH_MPI

#endif // MONTECARLO_RANK_HANDLER_HPP