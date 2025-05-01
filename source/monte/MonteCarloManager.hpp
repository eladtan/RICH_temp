#ifndef MONTE_CARLO_MANAGER_HPP
#define MONTE_CARLO_MANAGER_HPP

#include "mpi/mpi_commands.hpp"
#include "mpi/serialize/mpi_commands.hpp"
#include "MonteCarloParticle.hpp"
#include "physics/MonteCarloPhysics.hpp"
#include "population/PopulationControl.hpp"
#include "utils/debug/vtune.h" // TODO: remove
#include <memory>
#include <mpi.h>

#define MONTECARLO_EPSILON 1e-8
#define MPI_INDEX_T MPI_UINT32_T

template<typename T, typename Grid>
class MonteCarloManager
{
    using MCParticle = MonteCarloParticle<T, Grid>;

public:
    struct MonteCarloStepFinalData
    {
        std::vector<MCParticle> remaining;
        std::vector<MCParticle> leaving;
    };

    // todo: remove?
    struct MonteCarloStepInternalData
    {
        std::vector<std::vector<T>> normalsOfCells;
        std::vector<std::vector<T>> pointsOnFaces;
        MonteCarloStepFinalData finalData;

        // MonteCarloStepInternalData(const Grid &grid): grid(grid){};
    };

    MonteCarloManager(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics, const std::shared_ptr<PopulationControl<T, Grid>> &populationControl, const MPI_Comm &comm = MPI_COMM_WORLD);

    ~MonteCarloManager();

    void ClearCommunicator(void);

    void SetCommunicator(const MPI_Comm &comm);

    void UpdateHandlers(size_t bufferSizes);

    void TransferParticles(rank_t rankBuffer, const std::vector<size_t> &indicesInToHandle, const std::vector<rank_t> &transferRanks, size_t num);

    // todo: should return that?
    std::vector<MCParticle> step(const std::vector<MCParticle> &particleList, dt_t fullDt, size_t bufferSizes);
    
    class Tracker
    {
    public:
        Tracker(void);

        void Reset(void);

        #ifdef RICH_MPI
            std::vector<MCParticle> GetLocalTrackParticleRoute(size_t id);
        #endif // RICH_MPI

        std::vector<MCParticle> GetTrackParticleRoute(size_t id);

        void ReportParticle(MCParticle &particle);
    
    private:
        boost::container::flat_map<size_t, std::vector<MCParticle>> track;
    };

    inline const Tracker &getTracker(void){return this->tracker;};

    inline void resetTracker(void){this->tracker.Reset();};

private:
    class RankHandler
    {
    public:
        using index_t = uint32_t;
        
        RankHandler(size_t buffsize, const MPI_Comm &comm_world, const MPI_Comm &private_comm);
        
        ~RankHandler();
        
        void TransferParticles(const std::vector<const MCParticle*> &particles);

        void RemoveParticles(const std::vector<size_t> &indicesInToHandle, size_t num);

        void Sync(void);

        void Destroy();

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
        
        bool destroyed;

    private:
        class DistributedMutex
        {
        public:
            DistributedMutex(const MPI_Comm &comm, rank_t rank);
            
            ~DistributedMutex();

            void Lock(void);

            void Unlock(void);

            void Destroy(void);

            void Sync(void);

        private:
            const MPI_Comm &comm;
            rank_t rank;
            int *value;
            MPI_Win win;
            bool destroyed;
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

        inline void Sync(void)
        {
            MPI_Win_lock(MPI_LOCK_SHARED, this->rank, MPI_MODE_NOCHECK, this->is_done_win);
            MPI_Win_sync(this->is_done_win);
            MPI_Win_unlock(this->rank, this->is_done_win);
        };

        volatile int *is_done;
        int localDecrementAmount;

    private:
        rank_t rank, size, master_rank;
        volatile int *counter;
        MPI_Win counter_win;
        MPI_Win is_done_win;
    };

    const Grid &grid;
    MPI_Comm comm_world;
    rank_t rank_world, size_world;
    size_t Ncells;
    std::shared_ptr<ProgressCounter> progress;
    std::vector<MPI_Comm> communicators;
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ranks_ghost_map;
    std::vector<RankHandler*> rankHandlers;
    boost::container::flat_map<rank_t, size_t> rank_to_index;
    std::vector<std::tuple<rank_t, size_t, RankHandler*>> rank_to_index_vector;
    T ll, ur;
    size_t numScatters; // todo: remove?
    std::shared_ptr<MonteCarloPhysics<T, Grid>> physics;
    std::shared_ptr<PopulationControl<T, Grid>> populationControl;
    Tracker tracker;

    bool HandleAll(MonteCarloStepFinalData &cache);

    void PutSelfParticles(const std::vector<MCParticle> &particles);

    void PrepareForStep(size_t bufferSizes);

    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> GetGhostMap(void);

};

template<typename T, typename Grid>
MonteCarloManager<T, Grid>::RankHandler::DistributedMutex::DistributedMutex(const MPI_Comm &comm, rank_t rank):
    comm(comm), rank(rank), value(nullptr), destroyed(false)
{
    assert(this->comm != MPI_COMM_NULL);
    rank_t my_rank, size;
    MPI_Comm_rank(this->comm, &my_rank);
    MPI_Comm_size(this->comm, &size);
    assert(size > 1);

    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "accumulate_ordering", "none"); // No strict ordering
    MPI_Info_set(info, "accumulate_ops", "same_op");
    MPI_Info_set(info, "same_disp_unit", "true");
    MPI_Win_allocate((my_rank == rank)? sizeof(int) : 0, sizeof(int), info, this->comm, &this->value, &this->win);
    MPI_Win_set_errhandler(this->win, MPI_ERRORS_RETURN);
    MPI_Info_free(&info);

    int *model, flag;
    MPI_Win_get_attr(this->win, MPI_WIN_MODEL, &model, &flag);
    if(*model == MPI_WIN_SEPARATE)
    {
        std::cout << "MPI is using WIN_SEPARATE (" << MPI_WIN_SEPARATE << "). Can not continue" << std::endl;
        exit(1);
    }

    if(my_rank == rank)
    {
        *this->value = 0;
    }

    MPI_Barrier(this->comm);
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::RankHandler::DistributedMutex::Destroy()
{
    MPI_Win_free(&this->win);
    this->destroyed = true;
}

template<typename T, typename Grid>
MonteCarloManager<T, Grid>::RankHandler::DistributedMutex::~DistributedMutex()
{
    if(not this->destroyed)
    {
        this->Destroy();
    }
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::RankHandler::DistributedMutex::Sync(void)
{
    MPI_Win_lock(MPI_LOCK_SHARED, this->rank, MPI_MODE_NOCHECK, this->win);
    MPI_Win_sync(this->win);
    MPI_Win_unlock(this->rank, this->win);
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::RankHandler::DistributedMutex::Lock(void)
{
    static int plus_one = 1;
    static int minus_one = -1;

    int val = -1;
    // MPI_Win_lock(MPI_LOCK_SHARED, this->rank, MPI_MODE_NOCHECK, this->win);
    do
    {
        MPI_Win_lock_all(MPI_MODE_NOCHECK, this->win);
        val = -1;
        int retval = MPI_Fetch_and_op(&plus_one, &val, MPI_INT, this->rank, 0, MPI_SUM, this->win);
        assert(retval == MPI_SUCCESS);
        MPI_Win_flush(this->rank, this->win);
        assert(val >= 0);
        if(val > 0)
        {
            // std::cout << "Failed to lock mutex, trying again" << std::endl;
            // failure, decrement
            MPI_Accumulate(&minus_one, 1, MPI_INT, this->rank, 0, 1, MPI_INT, MPI_SUM, this->win);
            MPI_Win_flush(this->rank, this->win);
            usleep(10); // sleep a while and try again
        }
        MPI_Win_unlock_all(this->win);
    } while(val > 0);
    // MPI_Win_unlock(this->rank, this->win);
    assert(val <= 0);
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::RankHandler::DistributedMutex::Unlock(void)
{
    static int minus_one = -1;
    static int zero = 0;
    MPI_Win_lock(MPI_LOCK_SHARED, this->rank, MPI_MODE_NOCHECK, this->win);
    // MPI_Put(&zero, 1, MPI_INT, this->rank, 0, 1, MPI_INT, this->win);
    MPI_Accumulate(&minus_one, 1, MPI_INT, this->rank, 0, 1, MPI_INT, MPI_SUM, this->win);
    // MPI_Win_sync(this->win);
    MPI_Win_flush(this->rank, this->win);
    MPI_Win_unlock(this->rank, this->win);
}

template<typename T, typename Grid>
MonteCarloManager<T, Grid>::ProgressCounter::ProgressCounter(const MPI_Comm &comm, int myNumParticles)
{
    MPI_Comm_rank(comm, &this->rank);
    MPI_Comm_size(comm, &this->size);

    this->master_rank = 0;
    bool master = (this->rank == this->master_rank);
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "accumulate_ordering", "none"); // No strict ordering
    MPI_Info_set(info, "accumulate_ops", "same_op");
    MPI_Info_set(info, "same_disp_unit", "true");
    MPI_Win_allocate((master)? sizeof(size_t) : 0, sizeof(size_t), info, comm, &this->counter, &this->counter_win);
    MPI_Win_set_errhandler(this->counter_win, MPI_ERRORS_RETURN);
    MPI_Win_allocate(sizeof(int), sizeof(int), info, comm, &this->is_done, &this->is_done_win);
    MPI_Win_set_errhandler(this->is_done_win, MPI_ERRORS_RETURN);
    MPI_Info_free(&info);

    MPI_Win windows[2] = {this->counter_win, this->is_done_win};
    for(int i = 0; i < 2; i++)
    {
        int *model, flag;
        MPI_Win_get_attr(windows[i], MPI_WIN_MODEL, &model, &flag);
        if(*model == MPI_WIN_SEPARATE)
        {
            std::cout << "MPI is using WIN_SEPARATE (" << MPI_WIN_SEPARATE << "). Can not continue" << std::endl;
            exit(1);
        }
    }

    MPI_Reduce(&myNumParticles, (void*)this->counter, 1, MPI_INT, MPI_SUM, this->master_rank, comm);
    if(master)
    {
        // std::cout << "Initial value is " << *this->counter << std::endl;
    }   
    *this->is_done = 0;

    MPI_Barrier(comm);
}

template<typename T, typename Grid>
MonteCarloManager<T, Grid>::ProgressCounter::~ProgressCounter()
{
    MPI_Win_free(&this->is_done_win);
    MPI_Win_free(&this->counter_win);
}

template<typename T, typename Grid>
int MonteCarloManager<T, Grid>::ProgressCounter::Increment(int n)
{
    int result;
    MPI_Win_lock(MPI_LOCK_SHARED, this->master_rank, MPI_MODE_NOCHECK, this->counter_win);
    int retval = MPI_Fetch_and_op(&n, &result, MPI_INT, this->master_rank, 0, MPI_SUM, this->counter_win);
    assert(retval == MPI_SUCCESS);
    MPI_Win_unlock(this->master_rank, this->counter_win);
    int currValue = result + n;
    if(currValue == 0)
    {
        this->MarkDone();
    }
    return currValue;
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::ProgressCounter::MarkDone(void)
{
    static int plus_one = 1;
    MPI_Win_lock_all(MPI_MODE_NOCHECK, this->is_done_win);
    for(rank_t _rank = 0; _rank < this->size; _rank++)
    {
        MPI_Put(&plus_one, 1, MPI_INT, _rank, 0, 1, MPI_INT, this->is_done_win);
    }
    MPI_Win_unlock_all(this->is_done_win);
}

template<typename T, typename Grid>
boost::container::flat_map<size_t, std::pair<rank_t, size_t>> MonteCarloManager<T, Grid>::GetGhostMap(void)
{
    std::vector<std::vector<size_t>> incoming = MPI_exchange_data(this->grid.GetDuplicatedProcs(), this->grid.GetDuplicatedPoints());
    const std::vector<std::vector<size_t>> &ghosts = this->grid.GetGhostIndeces();
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
MonteCarloManager<T, Grid>::MonteCarloManager(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics, const std::shared_ptr<PopulationControl<T, Grid>> &populationControl, const MPI_Comm &comm):
    grid(grid), physics(physics), populationControl(populationControl), comm_world(MPI_COMM_NULL)
{
    this->rank_to_index = boost::container::flat_map<rank_t, size_t>();
    this->rank_to_index_vector = std::vector<std::tuple<rank_t, size_t, RankHandler*>>();
    this->SetCommunicator(comm);
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::PrepareForStep(size_t bufferSizes)
{
    this->UpdateHandlers(bufferSizes);
    this->numScatters = 0;
    this->Ncells = this->grid.GetPointNo();
    this->GetGhostMap();
    std::tie(this->ll, this->ur) = this->grid.GetBoxCoordinates();
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::ClearCommunicator()
{
    if(this->comm_world == MPI_COMM_NULL)
    {
        return;
    }

    for(MPI_Comm &comm : this->communicators)
    {
        MPI_Comm_free(&comm);
    }

    this->comm_world = MPI_COMM_NULL;
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::SetCommunicator(const MPI_Comm &comm)
{
    this->ClearCommunicator();

    this->comm_world = comm;
    MPI_Comm_rank(this->comm_world, &this->rank_world);
    MPI_Comm_size(this->comm_world, &this->size_world);

    this->communicators.clear();

    for(int rank1 = 0; rank1 < this->size_world; rank1++)
    {
        for(int rank2 = 0; rank2 <= rank1; rank2++)
        {
            int color = (this->rank_world == rank1 || this->rank_world == rank2) ? 1 : MPI_UNDEFINED;

            MPI_Comm new_comm = MPI_COMM_NULL;
            MPI_Comm_split(this->comm_world, color, this->rank_world, &new_comm);
            this->communicators.push_back(new_comm);
        }
    }

    if(this->rank_world == 0)
    {
        std::cout << "Finished communicators creation" << std::endl;
    }
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::UpdateHandlers(size_t bufferSizes)
{
    const std::vector<rank_t> &dupProcs = this->grid.GetDuplicatedProcs();
    this->rankHandlers.clear();
    this->rankHandlers.resize(dupProcs.size() + 1);
    this->rank_to_index.clear();
    this->rank_to_index_vector.clear();

    size_t i = 0;
    for(int rank1 = 0; rank1 < this->size_world; rank1++)
    {
        for(int rank2 = 0; rank2 <= rank1; rank2++) // self SHOULD be included
        {
            MPI_Barrier(this->comm_world);
            MPI_Comm &communicator = this->communicators[i];
            i++;
            
            if(rank1 == this->rank_world or rank2 == this->rank_world)
            {
                assert(communicator != MPI_COMM_NULL);
                rank_t other_rank = (rank1 == this->rank_world)? rank2 : rank1;
                size_t index = std::distance(dupProcs.cbegin(), std::find(dupProcs.cbegin(), dupProcs.cend(), other_rank));
                if((rank1 != rank2) and (index == dupProcs.size()))
                {
                    // rank is not a neighbor, allow self
                    continue;
                }
                
                // self is the last index
                assert(index < this->rankHandlers.size());
                this->rankHandlers[index] = new RankHandler(bufferSizes, this->comm_world, communicator);
                // std::cout << "My rank is " << this->rank_world << ", remote rank " << other_rank << " is in index " << index << std::endl;
                this->rank_to_index.insert({other_rank, index});
                this->rank_to_index_vector.push_back(std::make_tuple(other_rank, index, this->rankHandlers[index]));
            }
            else
            {
                assert(communicator == MPI_COMM_NULL);
            }
        }
    }
}

template<typename T, typename Grid>
MonteCarloManager<T, Grid>::~MonteCarloManager()
{
    for(rank_t rank1 = 0; rank1 < this->size_world; rank1++)
    {
        for(rank_t rank2 = 0; rank2 <= rank1; rank2++)
        {
            if(this->rank_world == rank1 or this->rank_world == rank2)
            {
                // free handler
                rank_t otherRank = (rank1 == this->rank_world)? rank2 : rank1;
                if(this->rank_to_index.find(otherRank) == this->rank_to_index.end())
                {
                    continue;
                }
                size_t handlerIndex = this->rank_to_index.at(otherRank);
                this->rankHandlers[handlerIndex]->Destroy();
                delete this->rankHandlers[handlerIndex];
            }
        }
    }

    this->ClearCommunicator();
}


template<typename T, typename Grid>
MonteCarloManager<T, Grid>::RankHandler::RankHandler(size_t buffsize, const MPI_Comm &comm_world, const MPI_Comm &private_comm):
    comm_world(comm_world), comm(private_comm), buffsize(buffsize), particles_win(MPI_WIN_NULL), av_win(MPI_WIN_NULL), th_win(MPI_WIN_NULL), av_length_win(MPI_WIN_NULL), th_length_win(MPI_WIN_NULL), destroyed(false)
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
        MPI_Info_set(info, "same_size", "true");
        MPI_Info_set(info, "same_disp_unit", "true");    
        // initialize windows

        MPI_Win_allocate(this->buffsize * sizeof(MCParticle), sizeof(MCParticle), info, this->comm, &this->particles, &this->particles_win);
        assert(this->particles != nullptr);
        MPI_Win_set_errhandler(this->particles_win, MPI_ERRORS_RETURN);
        MPI_Win_allocate(this->buffsize * sizeof(index_t), sizeof(index_t), info, this->comm, &this->av, &this->av_win);
        assert(this->av != nullptr);
        MPI_Win_set_errhandler(this->av_win, MPI_ERRORS_RETURN);
        MPI_Win_allocate(this->buffsize * sizeof(index_t), sizeof(index_t), info, this->comm, &this->th, &this->th_win);
        assert(this->th != nullptr);
        MPI_Win_set_errhandler(this->th_win, MPI_ERRORS_RETURN);
        MPI_Win_allocate(sizeof(int), sizeof(int), info, this->comm, static_cast<void*>(&this->av_length), &this->av_length_win);
        assert(this->av_length != nullptr);
        MPI_Win_set_errhandler(this->av_length_win, MPI_ERRORS_RETURN);
        MPI_Win_allocate(sizeof(int), sizeof(int), info, this->comm, static_cast<void*>(&this->th_length), &this->th_length_win);
        assert(this->th_length != nullptr);
        MPI_Win_set_errhandler(this->th_length_win, MPI_ERRORS_RETURN);
        MPI_Info_free(&info);

        MPI_Win windows[5] = {this->particles_win, this->av_win, this->th_win, this->av_length_win, this->th_length_win};
        for(int i = 0; i < 5; i++)
        {
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

        *this->av_length = this->buffsize;
        *this->th_length = 0;
    
        std::iota(this->av, this->av + this->buffsize, 0);
    }
    else
    {
        this->particles = nullptr;
        this->av = nullptr;
        this->th = nullptr;
        this->av_length = nullptr;
        this->th_length = nullptr;
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
void MonteCarloManager<T, Grid>::RankHandler::Sync(void)
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
void MonteCarloManager<T, Grid>::RankHandler::Destroy()
{
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
        delete[] this->particles;
        delete[] this->av;
        delete[] this->th;
        delete this->av_length;
        delete this->th_length;
    }
    this->destroyed = true;
}

template<typename T, typename Grid>
MonteCarloManager<T, Grid>::RankHandler::~RankHandler()
{
    if(not this->destroyed)
    {
        this->Destroy();
    }
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::RankHandler::RemoveParticles(const std::vector<size_t> &indicesInToHandle, size_t num)
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
void MonteCarloManager<T, Grid>::RankHandler::TransferParticles(const std::vector<const MCParticle*> &particles)
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
        
        int availLength;
        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->av_length_win);
        // todo: get to fetch_and_op
        int retval = MPI_Fetch_and_op(&decrement, &availLength, MPI_INT, this->other_rank, 0, MPI_SUM, this->av_length_win);
        assert(retval == MPI_SUCCESS);
        MPI_Win_flush(this->other_rank, this->av_length_win);
        if(availLength == 0)
        {
            int increment = particles.size();
            MPI_Accumulate(&increment, 1, MPI_INT, this->other_rank, 0, 1, MPI_INT, MPI_SUM, this->av_length_win);
            assert(false); // TODO: handle when transfer is infeasible
        }
        MPI_Win_unlock(this->other_rank, this->av_length_win);
        assert(availLength > 0);
        assert(availLength <= this->peer_buffsize);

        if(availLength < Np)
        {
            std::cout << "Impossible move from rank " << this->rank_world << " to rank " << this->peer_rank_world << " due to lack of space in available list (" << availLength << ")" << std::endl;
            assert(false);
        }
        // get particle empty index from avail list
        std::vector<index_t> availIndices(Np);
        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->av_win);
        MPI_Get(availIndices.data(), Np, MPI_INDEX_T, this->other_rank, availLength - Np, Np, MPI_INDEX_T, this->av_win);
        MPI_Win_unlock(this->other_rank, this->av_win);

        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->particles_win);
        for(size_t i = 0; i < Np; i++)
        {
            size_t availIndex = availIndices[i];
            const MCParticle *particle = particles[i];
            if(particle->id == 52)
            {
                std::cout << "Rank " << this->rank_world << " puts particle " << particle->id << " into rank " << this->peer_rank_world << std::endl;
            }
            /* put particle itself */
            MPI_Put(particle, sizeof(MCParticle), MPI_BYTE, this->other_rank, availIndex, sizeof(MCParticle), MPI_BYTE, this->particles_win);
        }
        MPI_Win_unlock(this->other_rank, this->particles_win);

        // MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, 0, this->av_length_win);
        // MPI_Accumulate(&minus_one, 1, MPI_INT, this->other_rank, 0, 1, MPI_INT, MPI_SUM, this->av_length_win);
        // MPI_Win_unlock(this->other_rank, this->av_length_win);

        /* put in to handle list */

        // get length of to handle list
        MPI_Win_lock(MPI_LOCK_SHARED, this->other_rank, MPI_MODE_NOCHECK, this->th_length_win);
        int toHandleLength;
        MPI_Get(&toHandleLength, 1, MPI_INT, this->other_rank, 0, 1, MPI_INT, this->th_length_win);
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
            this->particles[availIndex] = *particles[i];
            this->th[(*this->th_length)++] = availIndex;
            assert(*this->th_length < this->buffsize);
        }
    }
}

template<typename T, typename Grid>
MonteCarloManager<T, Grid>::Tracker::Tracker(void)
{}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::Tracker::Reset(void)
{
    this->track.clear();
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::Tracker::ReportParticle(MCParticle &particle)
{
    if(this->track.find(particle.id) == this->track.end())
    {
        this->track[particle.id] = std::vector<MCParticle>();
    }
    this->track[particle.id].push_back(particle);
}

#ifdef RICH_MPI
template<typename T, typename Grid>
std::vector<typename MonteCarloManager<T, Grid>::MCParticle> MonteCarloManager<T, Grid>::Tracker::GetLocalTrackParticleRoute(size_t id)
{
    if(this->track.find(id) == this->track.end())
    {
        return std::vector<MCParticle>();
    }
    return this->track[id];
}

template<typename T, typename Grid>
std::vector<typename MonteCarloManager<T, Grid>::MCParticle> MonteCarloManager<T, Grid>::Tracker::GetTrackParticleRoute(size_t id)
{
    std::vector<MCParticle> local = this->GetLocalTrackParticleRoute(id);
    std::vector<MCParticle> global = MPI_All_cast(local, this->comm_world);
    // sort by `particle.steps`
    std::sort(global.begin(), global.end(), [](const MCParticle &a, const MCParticle &b) { return a.steps < b.steps; });
    return global;
}
#else // RICH_MPI

template<typename T, typename Grid>
std::vector<typename MonteCarloManager<T, Grid>::MCParticle> MonteCarloManager<T, Grid>::Tracker::GetTrackParticleRoute(size_t id)
{
    if(this->track.find(id) == this->track.end())
    {
        return std::vector<MCParticle>();
    }
    return this->track[id];
}
#endif // RICH_MPI

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::MonteCarloManager::PutSelfParticles(const std::vector<MCParticle> &particles)
{
    size_t selfIndex = this->rank_to_index.at(this->rank_world);
    assert(selfIndex == this->rankHandlers.size() - 1);

    RankHandler *handler = this->rankHandlers[selfIndex];

    size_t particlesNum = particles.size();
    handler->buffsize = particlesNum;
    handler->particles = new MCParticle[handler->buffsize];
    handler->av = new typename RankHandler::index_t[handler->buffsize];
    handler->th = new typename RankHandler::index_t[handler->buffsize];
    handler->av_length = new int(0);
    handler->th_length = new int(0);

    std::memcpy(handler->particles, particles.data(), particles.size() * sizeof(MCParticle));

    // update 'to handle' and 'available' lists accordingly
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
void MonteCarloManager<T, Grid>::MonteCarloManager::TransferParticles(rank_t rankBuffer, const std::vector<size_t> &indicesInToHandle, const std::vector<rank_t> &transferRanks, size_t num)
{
    if(indicesInToHandle.empty())
    {
        // nothing to transfer
        return;
    }

    boost::container::flat_map<rank_t, std::vector<const MCParticle*>> rankToParticles;

    size_t currentRankIndex = this->rank_to_index.at(rankBuffer);
    RankHandler *currRankHandler = this->rankHandlers[currentRankIndex];

    std::vector<const MCParticle*> particles;
    for(size_t i = 0; i < num; i++)
    {
        const size_t &indexInToHandle = indicesInToHandle[i];
        const size_t &toRank = transferRanks[i];
        assert(toRank != this->rank_world); // can't send to self
        size_t bufferIdx = currRankHandler->th[indexInToHandle];
        auto it = rankToParticles.find(toRank);
        if(it == rankToParticles.end())
        {
            rankToParticles[toRank] = std::vector<const MCParticle*>();
        }
        const MCParticle *particle = &currRankHandler->particles[bufferIdx];
        rankToParticles[toRank].push_back(particle);
    }

    for(const auto &[toRank, particles] : rankToParticles)
    {
        assert(toRank != this->rank_world); // can't send to self
        size_t remoteRankIndex = this->rank_to_index.at(toRank);
        RankHandler *remoteHandler = this->rankHandlers[remoteRankIndex];
        if(remoteHandler->peer_rank_world != toRank) // todo remove
        {
            std::cout << "[" << this->rank_world << "] wants to send to rank " << toRank << ", remote rank index " << remoteRankIndex << ", but real remote rank is " << remoteHandler->rank_world << std::endl;
            for(const auto &keyval : this->rank_to_index)
            {
                std::cout << "[" << this->rank_world << "] rank " << keyval.first << " is in index " << keyval.second << std::endl;
            }
            assert(remoteHandler->peer_rank_world == toRank);
        }

        // std::cout << "Migrating particle " << *particle << ", from rankbuff " << rankBuffer << " to rank " << toRank << std::endl;
        remoteHandler->TransferParticles(particles);

    }

    // currRankHandler->RemoveParticles(indicesInToHandle);
}

template<typename T, typename Grid>
bool MonteCarloManager<T, Grid>::MonteCarloManager::HandleAll(MonteCarloStepFinalData &stepData)
{
    static std::vector<std::tuple<rank_t, size_t, RankHandler*>> active_ranks;
    static std::vector<std::tuple<rank_t, size_t, RankHandler*>> next_active_ranks;
    static std::vector<size_t> removeParticlesVec;
    static std::vector<size_t> transferParticlesVec;
    static std::vector<rank_t> transferRanks;
    // static std::uniform_real_distribution<double> dist(0, 1);
    // static std::mt19937 re(this->rank_world);
        
    next_active_ranks.clear();
    if(active_ranks.empty())
    {
        // std::cout << "active_ranks is empty" << std::endl;
        for(const std::tuple<rank_t, size_t, RankHandler*> &buffer : this->rank_to_index_vector)
        {
            RankHandler *handler = std::get<2>(buffer);
            // std::cout << "Running sync on window of rank " << std::get<0>(buffer) << std::endl;
            handler->Sync();
            // std::cout << "Done!" << std::endl;
            if(*handler->th_length > 0)
            {
                active_ranks.push_back(buffer);
            }
        }
    }

    bool isEmpty = true;
    size_t activeRanksNum = active_ranks.size();
    int removeCounter = 0;
    int transferCounter = 0;

    auto removeParticle = [&](size_t i)
    {
        if(removeCounter >= removeParticlesVec.size())
        {
            removeParticlesVec.push_back(i);
            removeCounter++;
        }
        else
        {
            removeParticlesVec[removeCounter++] = i;
        }
    };

    // for(const std::tuple<rank_t, size_t, RankHandler*> &buffer : this->rank_to_index_vector) // size_t index = 0; index < activeRanksNum; index++)
    for(size_t index = 0; index < activeRanksNum; index++)
    {
        const std::tuple<rank_t, size_t, RankHandler*> &buffer = active_ranks[index];
        rank_t buffRank = std::get<0>(buffer);
        size_t handlerIndex = std::get<1>(buffer);
        RankHandler *handler = std::get<2>(buffer);
        int length = *handler->th_length;
        removeCounter = 0;
        transferCounter = 0;
        distance_t scatteringLength = abs(this->ur - this->ll) / 10;

        for(int i = 0; i < length; i++)
        {
            isEmpty = false;
            size_t particleIndex = handler->th[i];
            MCParticle &particle = handler->particles[particleIndex];
            if(particle.on_track)
            {
                this->tracker.ReportParticle(particle);
            }
            particle.steps++;

            T prevLoc = particle.location;
            MonteCarloFunctionality<T, Grid> functionality = this->physics->step(particle);

            if(functionality.change == MonteCarloParticleStatus::CELL_MOVE)
            {
                size_t nextCellIndex = functionality.nextCellIndex;

                assert(nextCellIndex != particle.cellIndex);
                assert(particle.timeLeft >= 0);

                particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
                rank_t rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);

                // particle.location += 1e-8 * (grid.GetMeshPoint(nextCellIndex) - particle.location);
                if(BOOST_LIKELY(nextCellIndex < this->Ncells))
                {
                    particle.cellIndex = nextCellIndex;
                }
                else
                {
                    // a ghost point, check rank and index in rank
                    auto it = ranks_ghost_map.find(nextCellIndex);
                    if(it == ranks_ghost_map.end())
                    {
                        stepData.leaving.push_back(particle); // leaving domain
                        // remove particle from current list
                        this->progress->localDecrementAmount++;
                        removeParticle(i);
                        continue;
                    }

                    auto [otherRank, neighborIndexInRank] = it->second;
                    size_t rankIndex = this->rank_to_index.at(otherRank);
                    assert(rankIndex < this->rankHandlers.size());
                    particle.cellIndex = neighborIndexInRank;

                    if(transferCounter >= transferParticlesVec.size())
                    {
                        transferParticlesVec.push_back(i);
                        transferRanks.push_back(otherRank);
                        transferCounter++;
                    }
                    else
                    {
                        transferRanks[transferCounter] = otherRank;
                        transferParticlesVec[transferCounter++] = i;
                    }
                    removeParticle(i);
                    continue;
                }
            }
            else if(functionality.change == MonteCarloParticleStatus::REMOVE)
            {
                removeParticle(i);
                continue;
            }
            else if(functionality.change == MonteCarloParticleStatus::DONE)
            {
                stepData.remaining.push_back(particle);
                // remove particle from current list
                removeParticle(i);
                this->progress->localDecrementAmount += 1;
                continue;
            }
        }

        if(transferCounter > 0)
        {
            this->TransferParticles(buffRank, transferParticlesVec, transferRanks, transferCounter);
        }
        if(removeCounter > 0)
        {
            handler->RemoveParticles(removeParticlesVec, removeCounter);
        }
        if(length > 0)
        {
            next_active_ranks.push_back(buffer);
        }
    }
    active_ranks.swap(next_active_ranks);

    if(isEmpty and this->progress->localDecrementAmount > 0)
    {
        // a chance to update the progress counter 
        // std::cout << "Decrementing by " << this->progress->localDecrementAmount << " particles in the progress counter" << std::endl;
        this->progress->Decrement(this->progress->localDecrementAmount);
        this->progress->localDecrementAmount = 0;
    }
    return isEmpty;
}


template<typename T, typename Grid>
std::vector<typename MonteCarloManager<T, Grid>::MCParticle> MonteCarloManager<T, Grid>::MonteCarloManager::step(const std::vector<MCParticle> &particleList, dt_t fullDt, size_t bufferSizes)

{
    this->PrepareForStep(bufferSizes);
    this->PutSelfParticles(particleList);
    this->resetTracker();

    size_t myNparticles = particleList.size();
    size_t particleCounter = 0;
    MPI_Exscan(&myNparticles, &particleCounter, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    size_t totalParticles = 0;
    for(RankHandler *handler : this->rankHandlers)
    {
        int length = *handler->th_length;
        totalParticles += length;
        for(int i = 0; i < length; i++)
        {
            size_t particleIndex = handler->th[i];
            MCParticle &p = handler->particles[particleIndex];
            assert(p.timeLeft > 0);
            // p.timeLeft = fullDt;
            p.id = particleCounter;
            particleCounter++;
        }
    }

    // size_t totalParticles = 0;
    for(RankHandler *handler : this->rankHandlers)
    {
        int length = *handler->th_length;
        // totalParticles += length;
        for(int i = 0; i < length; i++)
        {
            size_t particleIndex = handler->th[i];
            MCParticle &p = handler->particles[particleIndex];
            p.initialWeight = p.weight;
            p.steps = 0;
        }
    }

    this->progress = std::make_shared<ProgressCounter>(this->comm_world, totalParticles);

    size_t Nfaces = grid.GetAllFaceNeighbors().size();
    std::vector<T> normals(Nfaces);
    for(size_t i = 0; i < Nfaces; i++)
    {
        normals[i] = grid.Normal(i);
    }

    size_t numParticles = *this->rankHandlers.back()->th_length;
    
    this->progress->localDecrementAmount = 0;
    
    volatile int &done = *this->progress->is_done;

    MPI_Barrier(this->comm_world);
    std::cout << "Rank " << this->rank_world << " starts the main loop" << std::endl;

    this->physics->updateGridData();
    this->physics->preStep(fullDt);

    MonteCarloStepFinalData data;
    vtune_start();
    // measure time
    auto start = std::chrono::high_resolution_clock::now();
    while(not done)
    {
        this->HandleAll(data);
        this->progress->Sync();
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::vector<MCParticle> newParticles = this->populationControl->activate(particleList);
    this->physics->postStep(newParticles);

    MPI_Barrier(this->comm_world);
    double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Rank " << this->rank_world << " is outside of step() loop, in " << seconds << " seconds (" << numParticles << " particles)" << std::endl;

    size_t leavingNumber = data.leaving.size();
    size_t leavingTotal;
    size_t scattersTotal;
    MPI_Reduce(&leavingNumber, &leavingTotal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce(&this->numScatters, &scattersTotal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    if(this->rank_world == 0)
    {
        std::cout << "Number of leaving particles is " << leavingTotal << ", num scatters " << scattersTotal << std::endl;
    }
    MPI_Barrier(this->comm_world);
    vtune_stop();
    // return data.finalData;
    return newParticles;
}

#endif // MONTE_CARLO_MANAGER_HPP