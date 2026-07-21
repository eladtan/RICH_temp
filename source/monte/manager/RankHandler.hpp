#ifndef MONTECARLO_RANK_HANDLER_HPP
#define MONTECARLO_RANK_HANDLER_HPP

#ifdef RICH_MPI

#include <vector>
#include <memory>
#include <iostream>
#include <chrono>
#include <mpi.h>
#include "mpi/mpi_commands.hpp"
#include "tools/DistributedMutex.hpp"
#include "tools/ConditionVariable.hpp"
#include "ReallocationAgent.hpp"

#define BUFFER_REALLOCATION_FACTOR 3 // 1.618033 // golden ratio
#define MINIMAL_BUFF_SIZE 50
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

    // todo: necessary?
    inline void LockSelfBuffer(void)
    {
        if(this->size_internal > 1)
        {
            this->localTHMutex->Lock();
        }
    }

    inline void UnlockSelfBuffer(void)
    {
        if(this->size_internal > 1)
        {
            this->localTHMutex->Unlock();
        }
    }

    double reallocationTime;
    size_t reallocationsThisStep;
    size_t reallocationsTotal;

private:
    MPI_Win particles_win;
    MPI_Win av_win;
    MPI_Win th_win;
    MPI_Win av_length_win;
    MPI_Win th_length_win;
    std::shared_ptr<DistributedMutex> localTHMutex;
    std::shared_ptr<DistributedMutex> remoteTHMutex;
    bool destroyed;
    double requestedFactor;
    MPI_Group group_world, group_internal;

    #ifdef ADVANCED_MONTECARLO_DEBUG
    void ValidateArraysContents(void) const;

    void ValidateRemoteArraysContents(void) const;
    #endif // ADVANCED_MONTECARLO_DEBUG
};

template<typename T, typename Grid>
RankHandler<T, Grid>::RankHandler(size_t buffsize, const MPI_Comm &comm_world, const MPI_Comm &private_comm, std::shared_ptr<ReallocationAgent> &reallocationAgent):
    comm_world(comm_world), comm(private_comm), buffsize(buffsize), particles_win(MPI_WIN_NULL),
    av_win(MPI_WIN_NULL), th_win(MPI_WIN_NULL), av_length_win(MPI_WIN_NULL), th_length_win(MPI_WIN_NULL),
    destroyed(false), reallocationAgent(reallocationAgent), reallocationsTotal(0)
{
    assert(private_comm != MPI_COMM_NULL);

    MPI_Comm_rank(this->comm_world, &this->rank_world);
    MPI_Comm_size(this->comm_world, &this->size_world);
    MPI_Comm_rank(this->comm, &this->rank_internal);
    MPI_Comm_size(this->comm, &this->size_internal);
    
    assert(this->size_internal == 2 or this->size_internal == 1);
    assert(this->rank_internal == 0 or this->rank_internal == 1);
    
    this->requestedFactor = 1;
    
    if(this->size_internal > 1)
    {
        this->other_rank = 1 - this->rank_internal;

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
        this->other_rank = 0;
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

        MPI_Comm_group(this->comm_world, &this->group_world);
        MPI_Comm_group(this->comm, &this->group_internal);

        // int ranks_in_group[2] = {this->rank_internal, this->other_rank};
        // int ranks_in_world[2];
        // MPI_Group_translate_ranks(this->group_internal, 2, ranks_in_group, this->group_world, ranks_in_world);
        // if(ranks_in_world[0] != this->rank_world)
        // {
        //     // UniversalError eo("RankHandler constructor: ranks translation failed");
        //     // eo.addEntry("Real rank", ranks_in_world[0]);
        //     // eo.addEntry("Expected rank", this->rank_world);
        //     // eo.addEntry("Peer rank", this->peer_rank_world);
        //     // throw eo;
        //     std::cout << "RankHandler constructor: ranks translation failed" << std::endl;
        //     std::cout << "Real rank " << ranks_in_world[0] << std::endl;
        //     std::cout << "Real other rank " << ranks_in_world[1] << std::endl;
        //     std::cout << "Expected rank " << this->rank_world << std::endl;
        //     std::cout << "Peer rank " << this->peer_rank_world << std::endl;
        // }
        // assert(ranks_in_world[1] == this->peer_rank_world);
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
    MPI_Win_lock(MPI_LOCK_SHARED, this->rank_internal, MPI_MODE_NOCHECK, this->particles_win);
    MPI_Win_sync(this->particles_win);
    MPI_Win_unlock(this->rank_internal,this->particles_win);
    MPI_Win_lock(MPI_LOCK_SHARED, this->rank_internal, MPI_MODE_NOCHECK, this->av_win);
    MPI_Win_sync(this->av_win);
    MPI_Win_unlock(this->rank_internal,this->av_win);
    MPI_Win_lock(MPI_LOCK_SHARED, this->rank_internal, MPI_MODE_NOCHECK, this->th_win);
    MPI_Win_sync(this->th_win);
    MPI_Win_unlock(this->rank_internal,this->th_win);
    MPI_Win_lock(MPI_LOCK_SHARED, this->rank_internal, MPI_MODE_NOCHECK, this->av_length_win);
    MPI_Win_sync(this->av_length_win);
    MPI_Win_unlock(this->rank_internal,this->av_length_win);
    MPI_Win_lock(MPI_LOCK_SHARED, this->rank_internal, MPI_MODE_NOCHECK, this->th_length_win);
    MPI_Win_sync(this->th_length_win);
    MPI_Win_unlock(this->rank_internal,this->th_length_win);
    // sync distributed mutex
    this->localTHMutex->Sync();
    this->remoteTHMutex->Sync();
}

template<typename T, typename Grid>
void RankHandler<T, Grid>::Reset(void)
{
    *this->av_length = static_cast<int>(this->buffsize);
    *this->th_length = 0;
    std::fill(this->th, this->th + this->buffsize, std::numeric_limits<index_t>::max());
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
        MPI_Group_free(&this->group_world);
        MPI_Group_free(&this->group_internal);
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
    if(not std::uncaught_exceptions())
    {
        if(not this->destroyed)
        {
            this->Destroy();
        }
    }
}

#ifdef ADVANCED_MONTECARLO_DEBUG
template<typename T, typename Grid>
void RankHandler<T, Grid>::ValidateArraysContents(void) const
{
    // if(this->localTHMutex != nullptr)
    // {
    //     this->localTHMutex->Lock();
    // }
        
    boost::container::flat_map<index_t, size_t> avMap;
    int av_length = *this->av_length;
    for(int i = 0; i < av_length; i++)
    {
        index_t avValue = this->av[i];
        if(avValue >= this->buffsize)
        {
            UniversalError eo("RankHandler::ValidateArraysContents: AV value is out of bounds");
            eo.addEntry("AV value", avValue);
            eo.addEntry("AV index", i);
            eo.addEntry("AV length", av_length);
            eo.addEntry("Buffer size", this->buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        if(avMap.find(avValue) != avMap.end())
        {
            UniversalError eo("RankHandler::ValidateArraysContents: AV value is duplicated");
            eo.addEntry("AV value", avValue);
            eo.addEntry("AV index", i);
            eo.addEntry("Previously In Index", avMap[avValue]);
            eo.addEntry("AV length", av_length);
            eo.addEntry("Buffer size", this->buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        avMap[avValue] = i;
    }

    boost::container::flat_map<index_t, size_t> thMap;
    int th_length = *this->th_length;
    for(int i = 0; i < th_length; i++)
    {
        index_t thValue = this->th[i];
        if(thValue >= this->buffsize)
        {
            UniversalError eo("RankHandler::ValidateArraysContents: TH value is out of bounds");
            eo.addEntry("TH value", thValue);
            eo.addEntry("TH index", i);
            eo.addEntry("TH length", th_length);
            eo.addEntry("Buffer size", this->buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        if(thMap.find(thValue) != thMap.end())
        {
            UniversalError eo("RankHandler::ValidateArraysContents: TH value is duplicated");
            eo.addEntry("TH value", thValue);
            eo.addEntry("TH index", i);
            eo.addEntry("Previously In Index", thMap[thValue]);
            eo.addEntry("TH length", th_length);
            eo.addEntry("Buffer size", this->buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        if(avMap.find(thValue) != avMap.end())
        {
            UniversalError eo("RankHandler::ValidateArraysContents: TH value is in AV");
            eo.addEntry("Value", thValue);
            eo.addEntry("TH index", i);
            eo.addEntry("AV index", avMap[thValue]);
            eo.addEntry("TH length", th_length);
            eo.addEntry("AV length", av_length);
            eo.addEntry("The Particle Index", this->th[i]);
            eo.addEntry("The Particle", this->particles[this->th[i]]);
            eo.addEntry("Buffer size", this->buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        thMap[thValue] = i;
    }

    // if(this->localTHMutex != nullptr)
    // {
    //     this->localTHMutex->Unlock();
    // }
}

template<typename T, typename Grid>
void RankHandler<T, Grid>::ValidateRemoteArraysContents(void) const
{
    static std::vector<index_t> remoteAV;
    static std::vector<index_t> remoteTH;

    int av_length;
    MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->av_length_win);
    int retval = MPI_Get(&av_length, 1, MPI_INT, this->other_rank, 0, 1, MPI_INT, this->av_length_win);
    assert(retval == MPI_SUCCESS);
    MPI_Win_unlock(this->other_rank, this->av_length_win);

    if(remoteAV.size() < av_length)
    {
        remoteAV.resize(av_length);
    }
    // get AV
    MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->av_win);
    retval = MPI_Get(remoteAV.data(), av_length, MPI_INDEX_T, this->other_rank, 0, av_length, MPI_INDEX_T, this->av_win);
    assert(retval == MPI_SUCCESS);
    MPI_Win_unlock(this->other_rank, this->av_win);

    int th_length;
    MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->th_length_win);
    retval = MPI_Get(&th_length, 1, MPI_INT, this->other_rank, 0, 1, MPI_INT, this->th_length_win);
    assert(retval == MPI_SUCCESS);
    MPI_Win_unlock(this->other_rank, this->th_length_win);

    if(remoteTH.size() < th_length)
    {
        remoteTH.resize(th_length);
    }

    // get TH
    MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->th_win);
    retval = MPI_Get(remoteTH.data(), th_length, MPI_INDEX_T, this->other_rank, 0, th_length, MPI_INDEX_T, this->th_win);
    assert(retval == MPI_SUCCESS);
    MPI_Win_unlock(this->other_rank, this->th_win); 

    // get AV and TH
    boost::container::flat_map<index_t, size_t> avMap;
    for(int i = 0; i < av_length; i++)
    {
        index_t avValue = remoteAV[i];
        if(avValue >= this->peer_buffsize)
        {
            UniversalError eo("RankHandler::ValidateRemoteArraysContents: Remote AV value is out of bounds");
            eo.addEntry("AV value", avValue);
            eo.addEntry("AV index", i);
            eo.addEntry("AV length", av_length);
            eo.addEntry("Remote Buffer size", this->peer_buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        if(avMap.find(avValue) != avMap.end())
        {
            UniversalError eo("RankHandler::ValidateRemoteArraysContents: Remote AV value is duplicated");
            eo.addEntry("AV value", avValue);
            eo.addEntry("AV index", i);
            eo.addEntry("Previously In Index", avMap[avValue]);
            eo.addEntry("AV length", av_length);
            eo.addEntry("Remote Buffer size", this->peer_buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        avMap[avValue] = i;
    }

    boost::container::flat_map<index_t, size_t> thMap;
    for(int i = 0; i < th_length; i++)
    {
        index_t thValue = remoteTH[i];
        if(thValue >= this->peer_buffsize)
        {
            UniversalError eo("RankHandler::ValidateRemoteArraysContents: Remote TH value is out of bounds");
            eo.addEntry("TH value", thValue);
            eo.addEntry("TH index", i);
            eo.addEntry("TH length", th_length);
            eo.addEntry("Remote Buffer size", this->peer_buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        if(thMap.find(thValue) != thMap.end())
        {
            UniversalError eo("RankHandler::ValidateRemoteArraysContents: Remote TH value is duplicated");
            eo.addEntry("TH value", thValue);
            eo.addEntry("TH index", i);
            eo.addEntry("Previously In Index", thMap[thValue]);
            eo.addEntry("TH length", th_length);
            eo.addEntry("Remote Buffer size", this->peer_buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        if(avMap.find(thValue) != avMap.end())
        {
            UniversalError eo("RankHandler::ValidateRemoteArraysContents: Remote TH value is in AV");
            eo.addEntry("Value", thValue);
            eo.addEntry("TH index", i);
            eo.addEntry("AV index", avMap[thValue]);
            eo.addEntry("TH length", th_length);
            eo.addEntry("AV length", av_length);
            eo.addEntry("Remote Buffer size", this->peer_buffsize);
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Peer Rank", this->peer_rank_world);
            throw eo;
        }
        thMap[thValue] = i;
    }
}
#endif // ADVANCED_MONTECARLO_DEBUG

template<typename T, typename Grid>
void RankHandler<T, Grid>::RemoveParticles(const std::vector<size_t> &indicesInToHandle, size_t num)
{
    static constexpr index_t inf = std::numeric_limits<index_t>::max();
    static constexpr int minus_one = -1;

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

    int startThLength = th_length;
    
    this->Sync();
    
    #ifdef ADVANCED_MONTECARLO_DEBUG
    try
    {
        this->ValidateArraysContents();
    }
    catch(UniversalError &e)
    {
        e.addEntry("Where", std::string("RankHandler::RemoveParticles - beginning"));
        e.addEntry("To Remove", num);
        throw e;
    }
    #endif // ADVANCED_MONTECARLO_DEBUG

    #ifdef MONTECARLO_DEBUG
    boost::container::flat_map<size_t, size_t> indicesMap;
    #endif // MONTECARLO_DEBUG
    for(int i = static_cast<int>(num) - 1; i >= 0; i--)
    {
        const size_t &toHandleIndex = indicesInToHandle[i];
        assert(i == 0 or indicesInToHandle[i] > indicesInToHandle[i-1]); // should be in a descending order
        // std::cout << "Rank " << this->rank_world << " removes particle " << toHandleIndex << " from handler of rank " << this->peer_rank_world << std::endl;
        assert(toHandleIndex < th_length);
        index_t particleIdx = this->th[toHandleIndex];
        #ifdef MONTECARLO_DEBUG
        if(indicesMap.find(particleIdx) != indicesMap.end())
        {
            UniversalError eo("RankHandler::RemoveParticles: trying to remove the same particle twice");
            eo.addEntry("Particle index", particleIdx);
            eo.addEntry("TH 1", indicesMap[particleIdx]);
            eo.addEntry("TH 2", toHandleIndex);
            eo.addEntry("Rank", this->rank_world);
            throw eo;
        }
        indicesMap.insert({particleIdx, toHandleIndex});
        #endif // MONTECARLO_DEBUG
        assert(av_length < this->buffsize);
        #ifdef MONTECARLO_DEBUG
        auto it = std::find(this->av, this->av + av_length, particleIdx);
        if(it != this->av + av_length)
        {
            UniversalError eo("RankHandler::RemoveParticles: trying to remove an already available particle");
            eo.addEntry("Particle index", particleIdx);
            eo.addEntry("Already found in index", std::distance(this->av, it));
            eo.addEntry("AV Length", av_length);
            eo.addEntry("Rank", this->rank_world);
            throw eo;
        }
        #endif // MONTECARLO_DEBUG
        this->av[av_length++] = particleIdx;
        this->th[toHandleIndex] = this->th[--th_length];
        this->th[th_length] = inf;
        assert(th_length >= 0);
    }   

    this->Sync();

    #ifdef ADVANCED_MONTECARLO_DEBUG
    try
    {
        this->ValidateArraysContents();
    }
    catch(UniversalError &e)
    {
        e.addEntry("Where", std::string("RankHandler::RemoveParticles - end"));
        e.addEntry("To Remove", num);
        throw e;
    }
    #endif // ADVANCED_MONTECARLO_DEBUG

    if(this->size_internal > 1)
    {
        // release self mutex
        this->localTHMutex->Unlock();
    }
}

template<typename T, typename Grid>
void RankHandler<T, Grid>::Reallocate(double factor)
{
    static constexpr index_t inf = std::numeric_limits<index_t>::max();
    // std::cout << "Rank " << this->rank_world << " reallocates (with rank " << this->peer_rank_world << ", factor " << factor << ", current size " << this->buffsize << ")" << std::endl;
    
    this->reallocationsThisStep++;
    this->reallocationsTotal++;

    double requestedFactorSelf;
    MPI_Sendrecv(&this->requestedFactor, 1, MPI_DOUBLE, this->other_rank, 0, &requestedFactorSelf, 1, MPI_DOUBLE, this->other_rank, 0, this->comm, MPI_STATUS_IGNORE);
    factor = std::max(factor, requestedFactorSelf);

    // if(this->rank_world <= this->peer_rank_world)
    // {
    //     std::cout << "Ranks " << this->rank_world << " and " << this->peer_rank_world << " have reallocation, for " << this->reallocationsThisStep << " times this step and " << this->reallocationsTotal << " times in total. ";
    //     std::cout << "Increasing sizes to " << (this->buffsize * factor) << " (current: " << this->buffsize << ")" << std::endl;
    // }
    
    size_t newBuffSize = std::ceil(this->buffsize * factor); // ceil is necessary to avoid 0
    size_t oldBuffSize = this->buffsize;
    if(oldBuffSize > newBuffSize)
    {
        if(*this->th_length != 0)
        {
            std::cerr << "Can not shrink memory when there are particles (there are " << (*this->th_length) << " particles)" << std::endl;
            exit(1);
        }
    }
    size_t peerNewBuffSize = std::ceil(this->peer_buffsize * factor);
    if(newBuffSize < MINIMAL_BUFF_SIZE or peerNewBuffSize < MINIMAL_BUFF_SIZE)
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
        if(new_particles == nullptr)
        {
            UniversalError eo("Allocation in MPI_Win_allocate (new_particles) returned null");
            eo.addEntry("Buffersize", this->buffsize);
            throw eo;
        }

        MPI_Win new_av_win;
        index_t *new_av;
        err = MPI_Win_allocate(this->buffsize * sizeof(index_t), sizeof(index_t), info, this->comm, &new_av, &new_av_win);
        reportErrorAndExit("MPI_Win_allocate new_av with buffsize = " + std::to_string(this->buffsize), err);
        if(new_av == nullptr)
        {
            UniversalError eo("Allocation in MPI_Win_allocate (new_av) returned null");
            eo.addEntry("Buffersize", this->buffsize);
            throw eo;
        }

        MPI_Win new_th_win;
        index_t *new_th;
        err = MPI_Win_allocate(this->buffsize * sizeof(index_t), sizeof(index_t), info, this->comm, &new_th, &new_th_win);
        reportErrorAndExit("MPI_Win_allocate new_th with buffsize = " + std::to_string(this->buffsize), err);
        if(new_th == nullptr)
        {
            UniversalError eo("Allocation in MPI_Win_allocate (new_th) returned null");
            eo.addEntry("Buffersize", this->buffsize);
            throw eo;
        }
        MPI_Info_free(&info);

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
            std::memcpy(new_th, this->th, this->buffsize * sizeof(index_t));
            std::iota(new_av, new_av + this->buffsize, 0);
            *this->av_length = static_cast<int>(this->buffsize);
        }
        std::fill(new_th + *this->th_length, new_th + this->buffsize, inf);

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

        #ifdef ADVANCED_MONTECARLO_DEBUG
            this->ValidateArraysContents();
        #endif // ADVANCED_MONTECARLO_DEBUG
        // std::cout << "Rank " << this->rank_world << " AV old window: " << oldwin << ", new window: " << new_av_win << " (peer size: " << this->peer_buffsize << ")" << std::endl;
        
        MPI_Barrier(this->comm); // unnecessary because of the sendrecv above
    }
    else
    {        
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
            std::memcpy(new_th, this->th, this->buffsize * sizeof(index_t));
            std::iota(new_av, new_av + this->buffsize, 0);
            *this->av_length = static_cast<int>(this->buffsize);
        }
        std::fill(new_th + *this->th_length, new_th + this->buffsize, inf);

        delete[] this->particles;
        delete[] this->av;
        delete[] this->th;
        this->particles = new_particles;
        this->av = new_av;
        this->th = new_th;

        this->peer_buffsize = this->buffsize;

        #ifdef ADVANCED_MONTECARLO_DEBUG
            this->ValidateArraysContents();
        #endif // ADVANCED_MONTECARLO_DEBUG
    }

    this->requestedFactor = 1;
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

        #ifdef ADVANCED_MONTECARLO_DEBUG
            try
            {
                ValidateRemoteArraysContents();
            }
            catch(UniversalError &eo)
            {
                eo.addEntry("Where", std::string("RankHandler<T, Grid>::TransferParticles - before transfer"));
                throw eo;
            }
        #endif // ADVANCED_MONTECARLO_DEBUG

        size_t reallocationsCounter = 0;
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
                int availLength2;
                retval = MPI_Fetch_and_op(&increment, &availLength2, MPI_INT, this->other_rank, 0, MPI_SUM, this->av_length_win);
                MPI_Win_flush(this->other_rank, this->av_length_win);
                #ifdef MONTECARLO_DEBUG
                if(availLength2 != (availLength + decrement))
                {
                    UniversalError eo("RankHandler<T, Grid>::TransferParticles: Reallocation: Fetch and op returned unexpected value");
                    eo.addEntry("Expected", availLength + decrement);
                    eo.addEntry("Got", availLength2);
                    eo.addEntry("Rank", this->rank_world);
                    eo.addEntry("Peer Rank", this->peer_rank_world);
                    throw eo;
                }
                #endif // MONTECARLO_DEBUG
                reallocationsCounter++;
            }
            MPI_Win_unlock(this->other_rank, this->av_length_win);
            assert(availLength >= 0);
            return availLength;
        };

        int availLength = getAvailableLength();
        auto start = std::chrono::high_resolution_clock::now();
        size_t requestReallocationTimes = 0;
        while(availLength < Np)
        {
            this->requestedFactor = static_cast<double>((Np + availLength)) / this->peer_buffsize * 1.5; // request 50% more than needed
            this->remoteTHMutex->Unlock();
            this->reallocationAgent->RequestReallocation(this->peer_rank_world);
            this->remoteTHMutex->Lock();
            availLength = getAvailableLength();
            requestReallocationTimes++;
        }
        assert(availLength >= Np);
        auto end = std::chrono::high_resolution_clock::now();
        this->reallocationTime += std::chrono::duration<double>(end - start).count();

        // if(requestReallocationTimes > 1)
        // {
        //     std::cout << "Rank " << this->rank_world << " and " << this->peer_rank_world << ", requested allocations " << requestReallocationTimes << " times before being able to transfer " << Np << " particles" << std::endl;
        // }
        // get particle empty index from avail list
        std::vector<index_t> availIndices(Np);

        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->av_win);
        int retval = MPI_Get(availIndices.data(), Np, MPI_INDEX_T, this->other_rank, availLength - Np, Np, MPI_INDEX_T, this->av_win);
        assert(retval == MPI_SUCCESS);
        MPI_Win_unlock(this->other_rank, this->av_win);

        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->particles_win);

        #ifdef MONTECARLO_DEBUG
            boost::container::flat_map<index_t, size_t> availIndicesMap;
        #endif // MONTECARLO_DEBUG

        for(size_t i = 0; i < Np; i++)
        {
            index_t availIndex = availIndices[i];
            #ifdef MONTECARLO_DEBUG
            if(availIndicesMap.find(availIndex) != availIndicesMap.end())
            {
                UniversalError eo("RankHandler<T, Grid>::TransferParticles: duplication in available Index");
                eo.addEntry("Available Index", availIndex);
                eo.addEntry("Already in Index", availIndicesMap[availIndex]);
                eo.addEntry("Index", i);
                eo.addEntry("Rank", this->rank_world);
                eo.addEntry("Peer Rank", this->peer_rank_world);
                throw eo;
            }
            availIndicesMap.insert({availIndex, i});
            #endif // MONTECARLO_DEBUG
            assert(availIndex < this->peer_buffsize);
            assert(i < Np);
            const MCParticle *particle = &particles[i];

            #ifdef MONTECARLO_DEBUG
            if(particle->nextRank != this->peer_rank_world)
            {
                UniversalError eo("RankHandler<T, Grid>::TransferParticles: Particle will not be sent to the expected rank");
                eo.addEntry("Particle", *particle);
                eo.addEntry("Origin", this->rank_world);
                eo.addEntry("Expected Rank", particle->nextRank);
                eo.addEntry("Next Rank", this->peer_rank_world);
                throw eo;
            }
            #endif // MONTECARLO_DEBUG
            /* put particle itself */
            // std::cout << "Rank " << this->rank_world << " transferrs particle " << particle->id << " of rank " << particle->rank << " to rank " << this->peer_rank_world << std::endl;
            
            assert(this->other_rank != this->rank_internal);
            retval = MPI_Put(particle, sizeof(MCParticle), MPI_BYTE, this->other_rank, availIndex, sizeof(MCParticle), MPI_BYTE, this->particles_win);
            #ifdef MONTECARLO_DEBUG
            if(retval != MPI_SUCCESS or particle->nextRank != this->peer_rank_world)
            {
                UniversalError eo("Put failed");
                eo.addEntry("Particle", *particle);
                eo.addEntry("Origin", this->rank_world);
                eo.addEntry("Expected Rank", particle->nextRank);
                eo.addEntry("Next Rank", this->peer_rank_world);
                eo.addEntry("Put Return Value", retval);
                throw eo;
            }
            #endif // MONTECARLO_DEBUG
            assert(retval == MPI_SUCCESS);
        }
        MPI_Win_flush(this->other_rank, this->particles_win);
        MPI_Win_sync(this->particles_win);
        MPI_Win_unlock(this->other_rank, this->particles_win);

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

        #ifdef ADVANCED_MONTECARLO_DEBUG
            try
            {
                ValidateRemoteArraysContents();
            }
            catch(UniversalError &eo)
            {
                eo.addEntry("Where", std::string("RankHandler<T, Grid>::TransferParticles - after transfer"));
                eo.addEntry("Transfer Amount", Np);
                eo.addEntry("AV Indices", availIndices);
                eo.addEntry("Expected TH length", toHandleLength + Np);
                eo.addEntry("Expected AV length", availLength);
                eo.addEntry("Reallocations Counter", reallocationsCounter);
                throw eo;
            }
        #endif // ADVANCED_MONTECARLO_DEBUG

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
            #ifdef MONTECARLO_DEBUG
            if(particles[i].nextRank != this->rank_world)
            {
                UniversalError eo("Particle will not be sent to the expected rank #2");
                eo.addEntry("Particle", particles[i]);
                eo.addEntry("Origin", particles[i].sentByRank);
                eo.addEntry("Next Rank", particles[i].nextRank);
                throw eo;
            }
            #endif // MONTECARLO_DEBUG
            this->th[(*this->th_length)++] = availIndex;
            assert(*this->th_length < this->buffsize);
        }
    }
}

#endif // RICH_MPI

#endif // MONTECARLO_RANK_HANDLER_HPP