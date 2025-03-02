#ifndef MONTE_CARLO_MANAGER2_HPP
#define MONTE_CARLO_MANAGER2_HPP

#include "mpi/mpi_commands.hpp"
#include "MonteCarloParticle.hpp"
#include <memory>
#include <mpi.h>

#define MPI_INDEX_T MPI_UINT32_T

template<typename T, typename Grid>
class MonteCarloManager2
{
    using MCParticle = MonteCarloParticle<T, Grid>;

public:
    struct MonteCarloStepData
    {
        std::vector<MCParticle> remaining;
        std::vector<MCParticle> leaving;
    };

    MonteCarloManager2(size_t buffer_size, const Grid &grid, const std::vector<MCParticle> &particleList, const MPI_Comm &comm = MPI_COMM_WORLD);

    void TransferParticle(rank_t rankBuffer, size_t bufferIdx, rank_t rank);

    bool HandleAll(double fullDt, MonteCarloStepData &stepData);

    MonteCarloStepData step(dt_t fullDt);

private:
    class RankHandler
    {
    public:
        using index_t = uint32_t;
        
        RankHandler(size_t buffsize, const MPI_Comm &comm_world, const MPI_Comm &private_comm);
        
        ~RankHandler();
        
        void TransferParticle(const MCParticle *particle);

        void RemoveParticle(index_t toHandleIndex);

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
        
    private:
        class DistributedMutex
        {
        public:
            DistributedMutex(const MPI_Comm &comm, rank_t rank);
            
            ~DistributedMutex();

            void Lock(void);

            void Unlock(void);

        private:
            const MPI_Comm &comm;
            rank_t rank;
            int *value;
            MPI_Win win;
        };

        MPI_Win particles_win;
        MPI_Win av_win;
        MPI_Win th_win;
        MPI_Win av_length_win;
        MPI_Win th_length_win;
        std::shared_ptr<DistributedMutex> localTHMutex;
        std::shared_ptr<DistributedMutex> remoteTHMutex;
    };

    class ProgressCounter
    {
    public:
        ProgressCounter(const MPI_Comm &comm, int myNumParticles);

        ~ProgressCounter();

        int Increment(int n);

        inline int Decrement(int n = 1){return this->Increment(-n);};
        
        void MarkDone(void);

        volatile int *is_done;

    private:
        rank_t size, master_rank;
        volatile int *counter;
        MPI_Win counter_win;
        MPI_Win is_done_win;
    };

    MPI_Comm comm_world;
    rank_t rank_world, size_world;
    const Grid &grid;
    size_t Ncells;
    std::shared_ptr<ProgressCounter> progress;

    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ranks_ghost_map;
    std::vector<RankHandler*> rankHandlers;
    boost::container::flat_map<rank_t, size_t> rank_to_index;
    std::vector<std::tuple<rank_t, size_t, RankHandler*>> rank_to_index_vector;

    void PutSelfParticles(const std::vector<MCParticle> &particles);

    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> GetGhostMap(void);

};

template<typename T, typename Grid>
MonteCarloManager2<T, Grid>::RankHandler::DistributedMutex::DistributedMutex(const MPI_Comm &comm, rank_t rank):
    comm(comm), rank(rank), value(nullptr)
{
    assert(this->comm != MPI_COMM_NULL);
    rank_t my_rank, size;
    MPI_Comm_rank(this->comm, &my_rank);
    MPI_Comm_size(this->comm, &size);
    assert(size > 1);

    MPI_Win_allocate((my_rank == rank)? sizeof(int) : 0, sizeof(int), MPI_INFO_NULL, this->comm, &this->value, &this->win);
    if(my_rank == rank)
    {
        *this->value = 0;
    }
    MPI_Barrier(this->comm);
}

template<typename T, typename Grid>
MonteCarloManager2<T, Grid>::RankHandler::DistributedMutex::~DistributedMutex()
{
    // MPI_Win_free(&this->win);
}

template<typename T, typename Grid>
void MonteCarloManager2<T, Grid>::RankHandler::DistributedMutex::Lock(void)
{
    static int plus_one = 1;
    static int minus_one = -1;

    int val = -1;
    do
    {
        val = -1;
        MPI_Win_lock(MPI_LOCK_SHARED, this->rank, 0, this->win);
        MPI_Fetch_and_op(&plus_one, &val, MPI_INT, this->rank, 0, MPI_SUM, this->win);
        MPI_Win_flush(this->rank, this->win);
        assert(val >= 0);
        if(val > 0)
        {
            // std::cout << "Failed to lock mutex, trying again" << std::endl;
            // failure, decrement
            MPI_Accumulate(&minus_one, 1, MPI_INT, this->rank, 0, 1, MPI_INT, MPI_SUM, this->win);
            usleep(10); // sleep a while and try again
        }
        MPI_Win_unlock(this->rank, this->win);
    } while(val > 0);
    assert(val == 0);
}

template<typename T, typename Grid>
void MonteCarloManager2<T, Grid>::RankHandler::DistributedMutex::Unlock(void)
{
    static int minus_one = -1;
    static int zero = 0;
    MPI_Win_lock(MPI_LOCK_SHARED, this->rank, 0, this->win);
    // MPI_Put(&zero, 1, MPI_INT, this->rank, 0, 1, MPI_INT, this->win);
    MPI_Accumulate(&minus_one, 1, MPI_INT, this->rank, 0, 1, MPI_INT, MPI_SUM, this->win);
    MPI_Win_unlock(this->rank, this->win);
}

template<typename T, typename Grid>
MonteCarloManager2<T, Grid>::ProgressCounter::ProgressCounter(const MPI_Comm &comm, int myNumParticles)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &this->size);

    this->master_rank = 0;
    bool master = (rank == this->master_rank);
    MPI_Win_allocate((master)? sizeof(size_t) : 0, sizeof(size_t), MPI_INFO_NULL, comm, &this->counter, &this->counter_win);
    MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, comm, &this->is_done, &this->is_done_win);

    MPI_Reduce(&myNumParticles, (void*)this->counter, 1, MPI_INT, MPI_SUM, this->master_rank, comm);
    if(master)
    {
        // std::cout << "Initial value is " << *this->counter << std::endl;
    }   
    *this->is_done = 0;

    MPI_Barrier(comm);
}

template<typename T, typename Grid>
MonteCarloManager2<T, Grid>::ProgressCounter::~ProgressCounter()
{
    // MPI_Win_free(&this->is_done_win);
    // MPI_Win_free(&this->counter_win);
}

template<typename T, typename Grid>
int MonteCarloManager2<T, Grid>::ProgressCounter::Increment(int n)
{
    int result;
    MPI_Win_lock(MPI_LOCK_SHARED, this->master_rank, 0, this->counter_win);
    MPI_Fetch_and_op(&n, &result, MPI_INT, this->master_rank, 0, MPI_SUM, this->counter_win);
    MPI_Win_flush(this->master_rank, this->counter_win);
    MPI_Win_unlock(this->master_rank, this->counter_win);
    int currValue = result + n;
    // std::cout << "Incremented by " << n << ", value is now " << currValue << std::endl;
    if(currValue == 0)
    {
        this->MarkDone();
    }
    return currValue;
}

template<typename T, typename Grid>
void MonteCarloManager2<T, Grid>::ProgressCounter::MarkDone(void)
{
    static int plus_one = 1;
    for(rank_t _rank = 0; _rank < this->size; _rank++)
    {
        MPI_Win_lock(MPI_LOCK_SHARED, _rank, 0, this->is_done_win);
        MPI_Put(&plus_one, 1, MPI_INT, _rank, 0, 1, MPI_INT, this->is_done_win);
        MPI_Win_unlock(_rank, this->is_done_win);
    }
}

template<typename T, typename Grid>
boost::container::flat_map<size_t, std::pair<rank_t, size_t>> MonteCarloManager2<T, Grid>::GetGhostMap()
{
    std::vector<std::vector<size_t>> incoming = MPI_exchange_data(grid.GetDuplicatedProcs(), grid.GetDuplicatedPoints());
    const std::vector<std::vector<size_t>> &ghosts = grid.GetGhostIndeces();
    for(size_t i = 0; i < incoming.size(); i++)
    {
        int _rank = grid.GetDuplicatedProcs()[i];
        for(size_t j = 0; j < incoming[i].size(); j++)
        {
            assert(incoming[i].size() == ghosts[i].size());
            ranks_ghost_map[ghosts[i][j]] = {_rank, incoming[i][j]};
        }
    }
    return ranks_ghost_map;
}

template<typename T, typename Grid>
MonteCarloManager2<T, Grid>::MonteCarloManager2(size_t buffer_size, const Grid &grid, const std::vector<MCParticle> &particleList, const MPI_Comm &comm):
    grid(grid), comm_world(comm)
{
    MPI_Comm_rank(this->comm_world, &this->rank_world);
    MPI_Comm_size(this->comm_world, &this->size_world);

    const std::vector<rank_t> &dupProcs = this->grid.GetDuplicatedProcs();
    this->rankHandlers.resize(dupProcs.size() + 1);
    this->rank_to_index = boost::container::flat_map<rank_t, size_t>();
    this->rank_to_index_vector = std::vector<std::tuple<rank_t, size_t, RankHandler*>>();

    // iterate all pairs (rank1, rank2)
    for(int rank1 = 0; rank1 < this->size_world; rank1++)
    {
        for(int rank2 = 0; rank2 <= rank1; rank2++)
        {
            int color = (this->rank_world == rank1 || this->rank_world == rank2) ? 1 : MPI_UNDEFINED;

            MPI_Comm new_comm = MPI_COMM_NULL;
            MPI_Comm_split(this->comm_world, color, this->rank_world, &new_comm);

            if(rank1 == this->rank_world or rank2 == this->rank_world)
            {
                assert(new_comm != MPI_COMM_NULL);
                rank_t other_rank = (rank1 == this->rank_world)? rank2 : rank1;
                size_t index = std::distance(dupProcs.cbegin(), std::find(dupProcs.cbegin(), dupProcs.cend(), other_rank));
                if((rank1 != rank2) and (index == dupProcs.size()))
                {
                    // rank is not a neighbor, allow self
                    continue;
                }
                // std::cout << "rank1 = " << rank1 << ", rank2 = " << rank2 << std::endl;

                // self is the last index
                assert(index < this->rankHandlers.size());
                this->rankHandlers[index] = new RankHandler(buffer_size, this->comm_world, new_comm);
                this->rank_to_index.insert({other_rank, index});
                this->rank_to_index_vector.push_back(std::make_tuple(other_rank, index, this->rankHandlers[index]));
            }
            else
            {
                assert(new_comm == MPI_COMM_NULL);
            }
        }
    }

    this->ranks_ghost_map = this->GetGhostMap();
    this->Ncells = this->grid.GetPointNo();
    this->PutSelfParticles(particleList);
}

template<typename T, typename Grid>
MonteCarloManager2<T, Grid>::RankHandler::RankHandler(size_t buffsize, const MPI_Comm &comm_world, const MPI_Comm &private_comm):
    comm_world(comm_world), comm(private_comm), buffsize(buffsize), particles_win(MPI_WIN_NULL), av_win(MPI_WIN_NULL), th_win(MPI_WIN_NULL), av_length_win(MPI_WIN_NULL), th_length_win(MPI_WIN_NULL)
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
        // initialize windows
        MPI_Win_allocate(this->buffsize * sizeof(MCParticle), sizeof(MCParticle), MPI_INFO_NULL, this->comm, &this->particles, &this->particles_win);
        assert(this->particles != nullptr);
        MPI_Win_allocate(this->buffsize * sizeof(index_t), sizeof(index_t), MPI_INFO_NULL, this->comm, &this->av, &this->av_win);
        assert(this->av != nullptr);
        MPI_Win_allocate(this->buffsize * sizeof(index_t), sizeof(index_t), MPI_INFO_NULL, this->comm, &this->th, &this->th_win);
        assert(this->th != nullptr);
        MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm, static_cast<void*>(&this->av_length), &this->av_length_win);
        assert(this->av_length != nullptr);
        MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm, static_cast<void*>(&this->th_length), &this->th_length_win);
        assert(this->th_length != nullptr);

        // initialize mutexes
        std::shared_ptr<DistributedMutex> rank0Mutex = std::make_shared<DistributedMutex>(comm, 0);
        std::shared_ptr<DistributedMutex> rank1Mutex = std::make_shared<DistributedMutex>(comm, 1);
        this->localTHMutex = (this->rank_internal == 0)? rank0Mutex : rank1Mutex;
        this->remoteTHMutex = (this->rank_internal == 0)? rank1Mutex : rank0Mutex;
    }
    else
    {
        this->particles = new MCParticle[this->buffsize];
        this->av = new index_t[this->buffsize];
        this->th = new index_t[this->buffsize];
        this->av_length = new int(0);
        this->th_length = new int(0);
    }

    *this->av_length = this->buffsize;
    *this->th_length = 0;

    std::iota(this->av, this->av + this->buffsize, 0);
    MPI_Sendrecv(&this->buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0, &this->peer_buffsize, 1, MPI_UNSIGNED_LONG_LONG, this->other_rank, 0, this->comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&this->rank_world, 1, MPI_INT, this->other_rank, 0, &this->peer_rank_world, 1, MPI_INT, this->other_rank, 0, this->comm, MPI_STATUS_IGNORE);
    MPI_Barrier(this->comm);
}

template<typename T, typename Grid>
MonteCarloManager2<T, Grid>::RankHandler::~RankHandler()
{
    if(this->size_internal > 1)
    {
        // MPI_Win_free(&this->particles_win);
        // MPI_Win_free(&this->av_win);
        // MPI_Win_free(&this->th_win);
        // MPI_Win_free(&this->av_length_win);
        // MPI_Win_free(&this->th_length_win);
    }
    else
    {
        delete[] this->particles;
        delete[] this->av;
        delete[] this->th;
        delete this->av_length;
        delete this->th_length;
    }
}

template<typename T, typename Grid>
void MonteCarloManager2<T, Grid>::RankHandler::RemoveParticle(index_t toHandleIndex)
{
    static int plus_one = 1;
    static int minus_one = -1;

    assert(*this->th_length > 0);
    assert(toHandleIndex < static_cast<index_t>(*this->th_length));
    index_t particleIdx = this->th[toHandleIndex];
    // std::cout << "Rank " << this->rank_world  << " (with buff of rank " << this->peer_rank_world << ") is here to remove THI " << toHandleIndex << ", which is particle " << particleIdx << ": " << this->particles[particleIdx] << std::endl;
    
    int th_length = *this->th_length;
    if(this->size_internal == 1)
    {
        assert(*this->av_length < this->buffsize);
        this->av[(*this->av_length)++] = particleIdx;
        this->th[toHandleIndex] = this->th[--(*this->th_length)];
    }
    else
    {
        // acquire self mutex
        this->localTHMutex->Lock();

        assert(*this->av_length < this->buffsize);
        this->av[(*this->av_length)++] = particleIdx;
        this->th[toHandleIndex] = this->th[--(*this->th_length)];

        // // increase available, decrease to handle
        // MPI_Win_lock(MPI_LOCK_SHARED, this->rank_internal, 0, this->av_length_win);
        // int availableLength = -1;
        // MPI_Fetch_and_op(&plus_one, &availableLength, MPI_INT, this->rank_internal, 0, MPI_SUM, this->av_length_win);
        // MPI_Win_unlock(this->rank_internal, this->av_length_win);
        // assert(0 <= availableLength);
        // assert(availableLength < this->buffsize);

        // // write particle index
        // MPI_Win_lock(MPI_LOCK_SHARED, this->rank_internal, 0, this->av_win);
        // MPI_Put(&particleIdx, 1, MPI_INDEX_T, this->rank_internal, availableLength, 1, MPI_INDEX_T, this->av_win);
        // MPI_Win_unlock(this->rank_internal, this->av_win);

        // // decrease to handle length
        // MPI_Win_lock(MPI_LOCK_SHARED, this->rank_internal, 0, this->th_length_win);
        // int th_length = -1;
        // MPI_Get(&th_length, 1, MPI_INT, this->rank_internal, 0, 1, MPI_INT, this->th_length_win);
        // assert(th_length > 0);
        // MPI_Win_flush_local(this->rank_internal, this->th_length_win);

        // MPI_Win_lock(MPI_LOCK_SHARED, this->rank_internal, 0, this->th_win);
        // MPI_Get(&this->th[toHandleIndex], 1, MPI_INDEX_T, this->rank_internal, th_length - 1, 1, MPI_INDEX_T, this->th_win);
        // MPI_Win_flush_local(this->rank_internal, this->th_win);
        // MPI_Win_unlock(this->rank_internal, this->th_win);

        // MPI_Accumulate(&minus_one, 1, MPI_INT, this->rank_internal, 0, 1, MPI_INT, MPI_SUM, this->th_length_win);
        // MPI_Win_unlock(this->rank_internal, this->th_length_win);

        // assert(*this->th_length == th_length - 1);
        // std::cout << "Rank " << this->rank_world << " (with buff of rank " << this->peer_rank_world << "): local available list length is now " << *this->av_length << ", to handle list length is now " << *this->th_length << " (was: " << th_length << ") (particle index in th = " << toHandleIndex << " is now " << this->th[toHandleIndex] << ")" << std::endl;

        // release self mutex
        this->localTHMutex->Unlock();
    }
}

template<typename T, typename Grid>
void MonteCarloManager2<T, Grid>::RankHandler::TransferParticle(const MCParticle *particle)
{
    static int minus_one = -1;
    static int plus_one = 1;

    if(this->size_internal > 1)
    {
        // acquire remote mutex
        this->remoteTHMutex->Lock();

        int availLength;
        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, 0, this->av_length_win);
        // todo: get to fetch_and_op
        MPI_Fetch_and_op(&minus_one, &availLength, MPI_INT, this->other_rank, 0, MPI_SUM, this->av_length_win);
        MPI_Win_flush(this->other_rank, this->av_length_win);
        if(availLength == 0)
        {
            MPI_Accumulate(&plus_one, 1, MPI_INT, this->other_rank, 0, 1, MPI_INT, MPI_SUM, this->av_length_win);
            assert(false); // TODO: handle when transfer is infeasible
        }
        MPI_Win_unlock(this->other_rank, this->av_length_win);
        assert(availLength > 0);
        assert(availLength <= this->peer_buffsize);

        // get particle empty index from avail list
        index_t availIndex;
        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, 0, this->av_win);
        MPI_Get(&availIndex, 1, MPI_INDEX_T, this->other_rank, availLength - 1, 1, MPI_INDEX_T, this->av_win);
        MPI_Win_unlock(this->other_rank, this->av_win);

        /* put particle itself */
        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, 0, this->particles_win);
        MPI_Put(particle, sizeof(MCParticle), MPI_BYTE, this->other_rank, availIndex, sizeof(MCParticle), MPI_BYTE, this->particles_win);
        MPI_Win_unlock(this->other_rank, this->particles_win);

        // MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, 0, this->av_length_win);
        // MPI_Accumulate(&minus_one, 1, MPI_INT, this->other_rank, 0, 1, MPI_INT, MPI_SUM, this->av_length_win);
        // MPI_Win_unlock(this->other_rank, this->av_length_win);

        /* put in to handle list */

        // get length of to handle list
        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, 0, this->th_length_win);
        int toHandleLength;
        MPI_Get(&toHandleLength, 1, MPI_INT, this->other_rank, 0, 1, MPI_INT, this->th_length_win);
        MPI_Win_flush(this->other_rank, this->th_length_win);
        // MPI_Win_unlock(this->other_rank, this->th_length_win);
        assert(toHandleLength >= 0);
        assert(toHandleLength < this->peer_buffsize);

        // put index in to handle list
        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, 0, this->th_win);
        MPI_Put(&availIndex, 1, MPI_INDEX_T, this->other_rank, toHandleLength, 1, MPI_INDEX_T, this->th_win);
        MPI_Win_unlock(this->other_rank, this->th_win);

        // add 1 to to handle index
        // MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, 0, this->th_length_win);
        MPI_Accumulate(&plus_one, 1, MPI_INT, this->other_rank, 0, 1, MPI_INT, MPI_SUM, this->th_length_win);
        MPI_Win_unlock(this->other_rank, this->th_length_win);

        // release remote mutex
        this->remoteTHMutex->Unlock();

        // std::cout << "In sending of particle ID " << particle->id << " from rank " << this->rank_world << " to rank " << this->peer_rank_world << ", avail length of peer is " << availLength << ", and got particle index " << availIndex << " and to handle " << toHandleLength << std::endl;

    }
    else
    {
        assert(*this->av_length > 0);
        index_t availIndex = this->av[--(*this->av_length)];
        this->particles[availIndex] = *particle;
        assert(*this->th_length < this->buffsize);
        this->th[*this->th_length] = availIndex;
        (*this->th_length)++;
    }
}

template<typename T, typename Grid>
void MonteCarloManager2<T, Grid>::MonteCarloManager2::PutSelfParticles(const std::vector<MCParticle> &particles)
{
    size_t selfIndex = this->rank_to_index.at(this->rank_world);
    assert(selfIndex == this->rankHandlers.size() - 1);

    RankHandler *handler = this->rankHandlers[selfIndex];
    assert(particles.size() < handler->buffsize);
    std::memcpy(handler->particles, particles.data(), particles.size() * sizeof(MCParticle));

    // update 'to handle' and 'available' lists accordingly
    size_t particlesNum = particles.size();
    *handler->th_length = particlesNum;
    for(size_t i = 0; i < particlesNum; i++)
    {
        assert(i < handler->buffsize);
        handler->th[i] = i;
    }

    size_t availLength = handler->buffsize - particlesNum;
    *handler->av_length = availLength;
    for(size_t i = 0; i < availLength; i++)
    {
        size_t idx = i + particlesNum;
        assert(idx < handler->buffsize);
        handler->av[i] = idx;
    }
}

template<typename T, typename Grid>
void MonteCarloManager2<T, Grid>::MonteCarloManager2::TransferParticle(rank_t rankBuffer, size_t indexInToHandle, rank_t toRank)
{
    assert(toRank != this->rank_world); // can't send to self

    size_t currentRankIndex = this->rank_to_index.at(rankBuffer);
    RankHandler *currRankHandler = this->rankHandlers[currentRankIndex];
    size_t bufferIdx = currRankHandler->th[indexInToHandle];
    MCParticle *particle = &currRankHandler->particles[bufferIdx];

    size_t remoteRankIndex = this->rank_to_index.at(toRank);
    RankHandler *remoteHandler = this->rankHandlers[remoteRankIndex];

    // std::cout << "Migrating particle " << *particle << ", from rankbuff " << rankBuffer << " to rank " << toRank << std::endl;
    remoteHandler->TransferParticle(particle);

    currRankHandler->RemoveParticle(indexInToHandle);
}

template<typename T, typename Grid>
bool MonteCarloManager2<T, Grid>::MonteCarloManager2::HandleAll(double fullDt, MonteCarloStepData &stepData)
{
    static std::vector<std::tuple<rank_t, size_t, RankHandler*>> active_ranks;
    static std::vector<std::tuple<rank_t, size_t, RankHandler*>> next_active_ranks;

    next_active_ranks.clear();
    if(active_ranks.empty())
    {
        // std::cout << "active_ranks is empty" << std::endl;
        for(const std::tuple<rank_t, size_t, RankHandler*> &buffer : this->rank_to_index_vector)
        {
            RankHandler *handler = std::get<2>(buffer);
            if(*handler->th_length > 0)
            {
                active_ranks.push_back(buffer);
            }
        }
    }
    bool isEmpty = true;
    int decrementAmount = 0;
    size_t activeRanksNum = active_ranks.size();
    for(size_t index = 0; index < activeRanksNum; index++)
    {
        const std::tuple<rank_t, size_t, RankHandler*> &buffer = active_ranks[index];
        rank_t buffRank = std::get<0>(buffer);
        size_t handlerIndex = std::get<1>(buffer);
        RankHandler *handler = std::get<2>(buffer);
        int length = *handler->th_length;
        
        for(int i = 0; i < length; i++)
        {
            isEmpty = false;
            size_t particleIndex = handler->th[i];
            MCParticle &particle = handler->particles[particleIndex];
            // std::cout << "Rank " << this->rank_world << ", iterates particle " << particle << " (to handle: " << i << ", particle index: " << particleIndex << ")" << std::endl;

            auto [faceIntersect, timeIntersect] = particle.distanceToNearestFace(grid);
            timeIntersect *= (1 + EPSILON);
            dt_t timeScattering = std::numeric_limits<dt_t>::max(); // TODO
            dt_t timeLeft = particle.timeLeft;

            dt_t dt = std::min(timeLeft, std::min(timeScattering, timeIntersect));
            assert(dt >= 0);

            if(dt == timeIntersect)
            {
                const std::pair<size_t, size_t> &cellNeighbors = grid.GetFaceNeighbors(faceIntersect);
                assert(particle.cellIndex == cellNeighbors.first or particle.cellIndex == cellNeighbors.second);
                size_t nextCellIndex = (cellNeighbors.first == particle.cellIndex)? cellNeighbors.second : cellNeighbors.first;
                // std::cout << "Rank " << this->rank_world << ", intersection time of paricle ID " << particle.id << " is " << timeIntersect << ", will be moved to " << nextCellIndex << " (generating point: " << grid.GetMeshPoint(nextCellIndex) << ")" << std::endl;
                assert(nextCellIndex != particle.cellIndex);
                T previousLocation = particle.location;
                particle.timeLeft -= dt;
                assert(particle.timeLeft >= 0);
                particle.location += particle.velocity * dt;
                if(nextCellIndex < this->Ncells)
                {
                    particle.cellIndex = nextCellIndex;
                    // assert(realContainingCell == nextCellIndex);
                }
                else
                {
                    // a ghost point, check rank and index in rank
                    auto it = ranks_ghost_map.find(nextCellIndex);
                    if(it == ranks_ghost_map.end())
                    {
                        stepData.leaving.push_back(particle); // leaving domain
                        // remove particle from current list
                        // std::cout << "Rank " << this->rank_world << " removes particle " << particle << " since it leaves the domain" << std::endl;
                        decrementAmount++;
                        handler->RemoveParticle(i);
                        i -= 1; length -= 1;
                        continue;
                    }
                    auto [otherRank, neighborIndexInRank] = it->second;
                    size_t rankIndex = this->rank_to_index.at(otherRank);
                    assert(rankIndex < this->rankHandlers.size());
                    // std::cout << "Rank " << this->rank_world << " sends " << particle << " (from cell " << particle.cellIndex << ") to rank " << otherRank << " at index " << neighborIndexInRank << " there, " << nextCellIndex << " here" << std::endl;
                    particle.cellIndex = neighborIndexInRank;
                    particle.sender = this->rank_world;
                    this->TransferParticle(buffRank, i, otherRank);
                    i -= 1; length -= 1; // repeat the particle in this location, as it is a new one replacing the leaving particle
                    continue;
                }
            }
            else if(dt == timeScattering)
            {
            }
            else if(dt == timeLeft)
            {
                stepData.remaining.push_back(particle);
                // remove particle from current list
                handler->RemoveParticle(i);
                i -= 1; length -= 1;
                // particleChange += 1; // decrement particles num later
                continue;
            }
        }

        if(length > 0)
        {
            next_active_ranks.push_back(buffer);
        }
    }
    active_ranks.swap(next_active_ranks);

    if(decrementAmount > 0)
    {
        this->progress->Decrement(decrementAmount);
    }
    return isEmpty;
}


template<typename T, typename Grid>
typename MonteCarloManager2<T, Grid>::MonteCarloStepData MonteCarloManager2<T, Grid>::MonteCarloManager2::step(dt_t fullDt)
{
    size_t totalParticles = 0;
    for(size_t handlerIndex = 0; handlerIndex < this->rankHandlers.size(); handlerIndex++)
    {
        RankHandler *handler = this->rankHandlers[handlerIndex];
        int length = *handler->th_length;
        totalParticles += length;
        for(int i = 0; i < length; i++)
        {
            size_t particleIndex = handler->th[i];
            MCParticle &p = handler->particles[particleIndex];
            p.cellIndex = this->grid.GetContainingCell(p.location);
            T loc = p.location;
            double distanceToCenter = abs(grid.GetMeshPoint(p.cellIndex) - loc);
            for(size_t j = 0; j < grid.GetPointNo(); j++)
            {
                double distance = abs(grid.GetMeshPoint(j) - loc);
                assert(distanceToCenter <= distance + EPSILON);
            }
        
            // AssertContainingCell(grid, p.cellIndex, p.location);
            // std::cout << "Rank " << this->rank_world << " particle with id " << p.id << ", location is " << p.location << " and containing cell is " << p.cellIndex << std::endl;
        }

    }
    this->progress = std::make_shared<ProgressCounter>(this->comm_world, totalParticles);

    size_t numParticles = *this->rankHandlers.back()->th_length;
    
    MonteCarloStepData stepData;
    volatile int &done = *this->progress->is_done;

    MPI_Barrier(this->comm_world);
    std::cout << "Rank " << this->rank_world << " starts the main loop" << std::endl;

    vtune_start();
    // measure time
    auto start = std::chrono::high_resolution_clock::now();
    while(not done) // todo: while remaining
    {
        this->HandleAll(fullDt, stepData);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Rank " << this->rank_world << " is outside of step() loop, in " << seconds << " seconds (" << numParticles << " particles)" << std::endl;
    vtune_stop();
    return stepData;
}

#endif // MONTE_CARLO_MANAGER2_HPP