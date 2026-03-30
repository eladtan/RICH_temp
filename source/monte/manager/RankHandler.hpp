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
#include "utils/rma/RMAFactory.hpp"

#define BUFFER_REALLOCATION_FACTOR 1.5 // 1.618033 // golden ratio
#define MINIMAL_BUFF_SIZE 50
#define BUFFER_SHRINK_FACTOR 0.1
#define BUFFER_SHRINK_NEIGHBOR_FACTOR 0.5
#define MPI_INDEX_T MPI_UINT32_T

template<typename T, typename Grid>
class RankHandler
{
public:
    using index_t = uint32_t;
    using MCParticle = MonteCarloParticle<T, Grid>;

    RankHandler(size_t buffsize, const MPI_Comm &comm_world, const MPI_Comm &private_comm, std::shared_ptr<ReallocationAgent> &reallocationAgent, RDMA_Type rdma_type = RDMA_Type::AUTO_RDMA);
    
    ~RankHandler();
    
    void TransferParticles(const std::vector<MCParticle> &particles);

    void RemoveParticles(const std::vector<size_t> &indicesInToHandle, size_t num);

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
    double requestedFactor;

private:
    std::unique_ptr<RemoteMemoryAgent<MCParticle>> particles_agent;
    std::unique_ptr<RemoteMemoryAgent<index_t>> av_agent;
    std::unique_ptr<RemoteMemoryAgent<index_t>> th_agent;
    std::unique_ptr<RemoteMemoryAgent<int>> av_length_agent;
    std::unique_ptr<RemoteMemoryAgent<int>> th_length_agent;
    std::shared_ptr<DistributedMutex> localTHMutex;
    std::shared_ptr<DistributedMutex> remoteTHMutex;
    RDMA_Type rdma_type;
    bool destroyed;
    MPI_Group group_world, group_internal;

    #ifdef ADVANCED_MONTECARLO_DEBUG
    void ValidateArraysContents(void) const;

    void ValidateRemoteArraysContents(void);
    #endif // ADVANCED_MONTECARLO_DEBUG
};

template<typename T, typename Grid>
RankHandler<T, Grid>::RankHandler(size_t buffsize, const MPI_Comm &comm_world, const MPI_Comm &private_comm, std::shared_ptr<ReallocationAgent> &reallocationAgent, RDMA_Type rdma_type):
    comm_world(comm_world), comm(private_comm), buffsize(buffsize),
    rdma_type(rdma_type), destroyed(false), reallocationAgent(reallocationAgent), reallocationsTotal(0)
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

        this->particles_agent = RMAFactory::Create<MCParticle>(this->rdma_type, this->buffsize, this->comm);
        this->av_agent = RMAFactory::Create<index_t>(this->rdma_type, this->buffsize, this->comm);
        this->th_agent = RMAFactory::Create<index_t>(this->rdma_type, this->buffsize, this->comm);
        this->av_length_agent = RMAFactory::Create<int>(this->rdma_type, 1, this->comm);
        this->th_length_agent = RMAFactory::Create<int>(this->rdma_type, 1, this->comm);

        this->particles = this->particles_agent->GetLocalPointer();
        this->av = this->av_agent->GetLocalPointer();
        this->th = this->th_agent->GetLocalPointer();
        this->av_length = const_cast<volatile int*>(this->av_length_agent->GetLocalPointer());
        this->th_length = const_cast<volatile int*>(this->th_length_agent->GetLocalPointer());
        
        // initialize mutexes
        std::shared_ptr<DistributedMutex> rank0Mutex = std::make_shared<DistributedMutex>(comm, 0, this->rdma_type);
        std::shared_ptr<DistributedMutex> rank1Mutex = std::make_shared<DistributedMutex>(comm, 1, this->rdma_type);
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
        this->particles_agent->Free();
        this->av_agent->Free();
        this->th_agent->Free();
        this->av_length_agent->Free();
        this->th_length_agent->Free();
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
void RankHandler<T, Grid>::ValidateRemoteArraysContents(void)
{
    static std::vector<index_t> remoteAV;
    static std::vector<index_t> remoteTH;

    int av_length;
    this->av_length_agent->Get(&av_length, 1, this->other_rank, 0);

    if(remoteAV.size() < static_cast<size_t>(av_length))
    {
        remoteAV.resize(av_length);
    }
    this->av_agent->Get(remoteAV.data(), av_length, this->other_rank, 0);

    int th_length;
    this->th_length_agent->Get(&th_length, 1, this->other_rank, 0);

    if(remoteTH.size() < static_cast<size_t>(th_length))
    {
        remoteTH.resize(th_length);
    }
    this->th_agent->Get(remoteTH.data(), th_length, this->other_rank, 0);

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
    
    this->reallocationsThisStep++;
    this->reallocationsTotal++;

    double requestedFactorSelf;
    MPI_Sendrecv(&this->requestedFactor, 1, MPI_DOUBLE, this->other_rank, 0, &requestedFactorSelf, 1, MPI_DOUBLE, this->other_rank, 0, this->comm, MPI_STATUS_IGNORE);
    factor = std::max(factor, requestedFactorSelf);
    
    size_t newBuffSize = std::ceil(this->buffsize * factor);
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
    newBuffSize = std::max<size_t>(newBuffSize, MINIMAL_BUFF_SIZE);
    peerNewBuffSize = std::max<size_t>(peerNewBuffSize, MINIMAL_BUFF_SIZE);

    this->buffsize = newBuffSize;

    if(this->size_internal > 1)
    {
        assert(this->size_internal == 2);

        this->particles_agent->Resize(this->buffsize);
        this->av_agent->Resize(this->buffsize);
        this->th_agent->Resize(this->buffsize);

        this->particles = this->particles_agent->GetLocalPointer();
        this->av = this->av_agent->GetLocalPointer();
        this->th = this->th_agent->GetLocalPointer();

        if(this->buffsize >= oldBuffSize)
        {
            size_t difference = this->buffsize - oldBuffSize;
            std::memmove(this->av + difference, this->av, oldBuffSize * sizeof(index_t));
            std::iota(this->av, this->av + difference, static_cast<index_t>(oldBuffSize));

            std::fill(this->th + oldBuffSize, this->th + this->buffsize, inf);

            int difference_int = static_cast<int>(difference);
            *this->av_length += difference_int;
        }
        else
        {
            std::iota(this->av, this->av + this->buffsize, 0);
            *this->av_length = static_cast<int>(this->buffsize);
        }
        MPI_Sendrecv(&this->buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0, &this->peer_buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0, this->comm, MPI_STATUS_IGNORE);

        #ifdef ADVANCED_MONTECARLO_DEBUG
            this->ValidateArraysContents();
        #endif // ADVANCED_MONTECARLO_DEBUG
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
    size_t Np = particles.size();
    if(particles.empty())
    {
        return;
    }

    if(this->size_internal > 1)
    {
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
            this->av_length_agent->Get(&availLength, 1, this->other_rank, 0);
            assert(availLength <= static_cast<int>(this->peer_buffsize));
            if(availLength >= static_cast<int>(Np))
            {
                int newAvail = availLength - static_cast<int>(Np);
                this->av_length_agent->Put(&newAvail, 1, this->other_rank, 0, false);
            }
            else
            {
                reallocationsCounter++;
            }
            assert(availLength >= 0);
            return availLength;
        };

        int availLength = getAvailableLength();
        auto start = std::chrono::high_resolution_clock::now();
        while(availLength < Np)
        {
            size_t peerUsed = this->peer_buffsize - static_cast<size_t>(availLength);
            this->requestedFactor = static_cast<double>(peerUsed + Np) /
                                    static_cast<double>(this->peer_buffsize) * 1.5;

            this->remoteTHMutex->Unlock();
            this->reallocationAgent->RequestReallocation(this->peer_rank_world);
            this->remoteTHMutex->Lock();
            availLength = getAvailableLength();
        }
        assert(availLength >= Np);
        auto end = std::chrono::high_resolution_clock::now();
        this->reallocationTime += std::chrono::duration<double>(end - start).count();

        std::vector<index_t> availIndices(Np);
        this->av_agent->Get(availIndices.data(), Np, this->other_rank, availLength - Np);

        #ifdef MONTECARLO_DEBUG
            boost::container::flat_map<index_t, size_t> availIndicesMap;
            for(size_t i = 0; i < Np; i++)
            {
                index_t availIndex = availIndices[i];
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
                assert(availIndex < this->peer_buffsize);
                if(particles[i].nextRank != this->peer_rank_world)
                {
                    UniversalError eo("RankHandler<T, Grid>::TransferParticles: Particle will not be sent to the expected rank");
                    eo.addEntry("Particle", particles[i]);
                    eo.addEntry("Origin", this->rank_world);
                    eo.addEntry("Expected Rank", particles[i].nextRank);
                    eo.addEntry("Next Rank", this->peer_rank_world);
                    throw eo;
                }
            }
        #endif // MONTECARLO_DEBUG

        assert(this->other_rank != this->rank_internal);
        this->particles_agent->PutScatter(particles.data(), availIndices.data(), Np, this->other_rank);

        int toHandleLength;
        this->th_length_agent->Get(&toHandleLength, 1, this->other_rank, 0);
        assert(toHandleLength >= 0);
        assert(toHandleLength < static_cast<int>(this->peer_buffsize));

        this->th_agent->Put(availIndices.data(), Np, this->other_rank, toHandleLength, false);

        int newThLength = toHandleLength + static_cast<int>(Np);
        this->th_length_agent->Put(&newThLength, 1, this->other_rank, 0, false);

        // Explicit flush points:
        // 1) commit available-length reservation
        // 2) commit TH payload
        // 3) publish new TH length
        // Keep these before validation/unlock so remote state is consistent.
        this->av_length_agent->Flush(this->other_rank);
        this->th_agent->Flush(this->other_rank);
        this->th_length_agent->Flush(this->other_rank);

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