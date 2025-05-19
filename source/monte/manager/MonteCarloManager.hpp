#ifndef MONTE_CARLO_MANAGER_HPP
#define MONTE_CARLO_MANAGER_HPP

#include "mpi/mpi_commands.hpp"
#include "mpi/serialize/mpi_commands.hpp"
#include "monte/MonteCarloParticle.hpp"
#include "monte/physics/MonteCarloPhysics.hpp"
#include "monte/population/PopulationControl.hpp"
#include "tools/ProgressCounter.hpp"
#include "monte/boundary/BoundaryCondition.hpp"
#include "utils/debug/vtune.h" // TODO: remove
#include "RankHandler.hpp"
#include "ReallocationAgent.hpp"
#include <memory>
#include <random>
#include <mpi.h>

#define MONTECARLO_EPSILON 1e-8
#define DEFAULT_BUFFER_SIZE 10

template<typename T, typename Grid>
class MonteCarloManager
{
    using MCParticle = MonteCarloParticle<T, Grid>;
    using RankHandler = RankHandler<T, Grid>;

public:
    struct MonteCarloStepFinalData
    {
        std::vector<MCParticle> remaining;
        std::vector<MCParticle> leaving;
    };

    MonteCarloManager(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics,
                    const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
                    const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition,
                    size_t bufferSizes = DEFAULT_BUFFER_SIZE,
                    const MPI_Comm &comm = MPI_COMM_WORLD);

    ~MonteCarloManager();

    void ClearCommunicator(void);

    void SetCommunicator(const MPI_Comm &comm);

    void TransferParticles(rank_t rankBuffer, const std::vector<size_t> &indicesInToHandle, const std::vector<rank_t> &transferRanks, size_t num);

    // todo: should return that?
    std::vector<MCParticle> step(const std::vector<MCParticle> &particleList, dt_t fullDt);
    
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
    const Grid &grid;
    MPI_Comm comm_world;
    rank_t rank_world, size_world;
    size_t Ncells;
    std::shared_ptr<ProgressCounter> progress;
    std::vector<MPI_Comm> communicators;
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ranks_ghost_map;
    std::vector<RankHandler*> rankHandlers;
    T ll, ur;
    std::shared_ptr<MonteCarloPhysics<T, Grid>> physics;
    std::shared_ptr<PopulationControl<T, Grid>> populationControl;
    std::shared_ptr<BoundaryCondition<T, Grid>> boundaryCondition;
    Tracker tracker;
    std::shared_ptr<ReallocationAgent> reallocationAgent;
    size_t myIDCounter;
    size_t currentStep;

    bool HandleAll(MonteCarloStepFinalData &cache);

    void PutSelfParticles(const std::vector<MCParticle> &particles);

    void FreeHandlers(void);

    void AddParticles(const std::vector<MCParticle> &particles);

    void ResetAllBuffers(void); 

    void ShrinkAllBuffers(void);
    
    void InitializeHandlers(size_t bufferSizes);

    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> GetGhostMap(void);
};

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
MonteCarloManager<T, Grid>::MonteCarloManager(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics, const std::shared_ptr<PopulationControl<T, Grid>> &populationControl, 
                                            const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition, size_t bufferSizes, const MPI_Comm &comm):
    grid(grid), physics(physics), populationControl(populationControl), boundaryCondition(boundaryCondition), comm_world(MPI_COMM_NULL)
{
    this->myIDCounter = 0;
    this->currentStep = 0;
    this->SetCommunicator(comm);
    this->InitializeHandlers(bufferSizes);
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::InitializeHandlers(size_t bufferSizes)
{
    this->rankHandlers = std::vector<RankHandler*>(this->size_world, nullptr);

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
                this->rankHandlers[other_rank] = new RankHandler(bufferSizes, this->comm_world, communicator, this->reallocationAgent);
            }
            else
            {
                assert(communicator == MPI_COMM_NULL);
            }
        }
    }

}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::ClearCommunicator()
{
    if(this->comm_world == MPI_COMM_NULL)
    {
        return;
    }

    for(size_t i = 0; i < this->communicators.size(); i++)
    {
        MPI_Comm &comm = this->communicators[i];
        if(comm != MPI_COMM_NULL)
        {
            MPI_Comm_free(&comm);
        }
        MPI_Barrier(this->comm_world);
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

    auto reallocationFunction = [this](rank_t rank)
    {
        this->rankHandlers[rank]->Reallocate(BUFFER_REALLOCATION_FACTOR);
    };

    this->reallocationAgent = std::make_shared<ReallocationAgent>(this->comm_world, reallocationFunction);
    if(this->rank_world == 0)
    {
        std::cout << "Finished communicators creation" << std::endl;
    }
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::FreeHandlers(void)
{
    for(rank_t rank1 = 0; rank1 < this->size_world; rank1++)
    {
        for(rank_t rank2 = 0; rank2 <= rank1; rank2++)
        {
            MPI_Barrier(this->comm_world);
            if(this->rank_world == rank1 or this->rank_world == rank2)
            {
                // free handler
                rank_t otherRank = (rank1 == this->rank_world)? rank2 : rank1;
                this->rankHandlers[otherRank]->Destroy();
                delete this->rankHandlers[otherRank];
            }
        }
    }
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::AddParticles(const std::vector<MCParticle> &particles)
{
    using index_t = typename RankHandler::index_t;
    if(particles.empty())
    {
        return;
    }

    RankHandler *myHandler = this->rankHandlers[this->rank_world];

    // std::cout << "In add particles, handler size is " << myHandler->buffsize << ", particles size to add is " << particles.size() << std::endl;

    while(*myHandler->av_length < particles.size())
    {
        myHandler->Reallocate(BUFFER_REALLOCATION_FACTOR);
    }

    // set particles
    // update 'to handle' and 'available' lists accordingly
    index_t particlesNum = particles.size();
    *myHandler->av_length -= particlesNum;
    index_t *avIndices = myHandler->av + (*myHandler->av_length);
    index_t *thIndices = myHandler->th + (*myHandler->th_length);
    *myHandler->th_length += particlesNum;
    size_t firstID = this->myIDCounter;
    this->myIDCounter += particles.size();

    for(size_t i = 0; i < particlesNum; i++)
    {
        index_t idx = avIndices[i];
        // copy particle
        MCParticle *particle = myHandler->particles + idx;
        std::memcpy(particle, &particles[i], sizeof(MCParticle));
        // set to handle
        thIndices[i] = idx;
        // set ID
        particle->rank = this->rank_world;
        particle->id = firstID + i;
    }

    this->progress->Increment(particlesNum);
    // std::cout << "Done add particles" << std::endl;
}

template<typename T, typename Grid>
MonteCarloManager<T, Grid>::~MonteCarloManager()
{
    this->FreeHandlers();
    this->ClearCommunicator();
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
    RankHandler *handler = this->rankHandlers[this->rank_world];

    size_t particlesNum = particles.size();
    while(*handler->av_length < particlesNum)
    {
        handler->Reallocate(BUFFER_REALLOCATION_FACTOR);
    }

    std::memcpy(handler->particles, particles.data(), particlesNum * sizeof(MCParticle));

    // update 'to handle' and 'available' lists accordingly
    *handler->th_length = particlesNum;
    for(size_t i = 0; i < particlesNum; i++)
    {
        assert(i < handler->buffsize);
        handler->th[i] = i;
        handler->particles[i].rank = this->rank_world;
        handler->particles[i].id = this->myIDCounter++;
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
void MonteCarloManager<T, Grid>::MonteCarloManager::TransferParticles(rank_t fromRank, const std::vector<size_t> &indicesInToHandle, const std::vector<rank_t> &transferRanks, size_t num)
{
    if(indicesInToHandle.empty())
    {
        // nothing to transfer
        return;
    }

    boost::container::flat_map<rank_t, std::vector<MCParticle>> rankToParticles;

    RankHandler *currRankHandler = this->rankHandlers[fromRank];

    for(size_t i = 0; i < num; i++)
    {
        const size_t &indexInToHandle = indicesInToHandle[i];
        const size_t &toRank = transferRanks[i];
        assert(toRank != this->rank_world); // can't send to self
        size_t bufferIdx = currRankHandler->th[indexInToHandle];
        auto it = rankToParticles.find(toRank);
        if(it == rankToParticles.end())
        {
            rankToParticles[toRank] = std::vector<MCParticle>();
        }
        const MCParticle &particle = currRankHandler->particles[bufferIdx];
        rankToParticles[toRank].push_back(particle);
    }

    for(const auto &[toRank, particles] : rankToParticles)
    {
        assert(toRank != this->rank_world); // can't send to self
        RankHandler *remoteHandler = this->rankHandlers[toRank];

        // std::cout << "Migrating particle " << *particle << ", from rankbuff " << rankBuffer << " to rank " << toRank << std::endl;
        remoteHandler->TransferParticles(particles);
    }

    // currRankHandler->RemoveParticles(indicesInToHandle);
}

template<typename T, typename Grid>
bool MonteCarloManager<T, Grid>::MonteCarloManager::HandleAll(MonteCarloStepFinalData &stepData)
{
    static std::vector<rank_t> active_ranks;
    static std::vector<rank_t> next_active_ranks;
    static std::vector<size_t> removeParticlesVec;
    static std::vector<size_t> transferParticlesVec;
    static std::vector<rank_t> transferRanks;
    // static std::uniform_real_distribution<double> dist(0, 1);
    // static std::mt19937 re(this->rank_world);
    
    this->reallocationAgent->HandleWaitingReallocations();

    next_active_ranks.clear();
    if(active_ranks.empty())
    {
        // std::cout << "active_ranks is empty" << std::endl;
        for(rank_t _rank = 0; _rank < this->size_world; _rank++)
        {
            RankHandler *handler = this->rankHandlers[_rank];
            // std::cout << "Running sync on window of rank " << std::get<0>(buffer) << std::endl;
            handler->Sync();
            // std::cout << "Done!" << std::endl;
            if(*handler->th_length > 0)
            {
                active_ranks.push_back(_rank);
            }
        }
    }

    bool isEmpty = true;
    size_t activeRanksNum = active_ranks.size();
    size_t removeCounter = 0;
    size_t transferCounter = 0;

    auto eliminateParticle = [&](size_t i)
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

    auto removeParticle = [&](size_t i)
    {
        eliminateParticle(i);
        this->progress->localDecrementAmount += 1;
    };

    for(size_t index = 0; index < activeRanksNum; index++)
    {
        rank_t _rank = active_ranks[index];
        RankHandler *handler = this->rankHandlers[_rank];
        int length = *handler->th_length;
        removeCounter = 0;
        transferCounter = 0;
        distance_t scatteringLength = abs(this->ur - this->ll) / 10;

        for(int i = 0; i < length; i++)
        {
            isEmpty = false;
            assert(i < handler->buffsize);
            size_t particleIndex = handler->th[i];
            assert(particleIndex < handler->buffsize);
            MCParticle &particle = handler->particles[particleIndex];
            if(particle.on_track)
            {
                this->tracker.ReportParticle(particle);
            }
            particle.steps++;

            T prevLoc = particle.location;
            MonteCarloFunctionality<T, Grid> functionality = this->physics->step(particle);
            // std::cout << "Handling particle " << particle << ", functionality is " << functionality.change << std::endl;

            if(functionality.change == MonteCarloParticleStatus::CELL_MOVE)
            {
                size_t nextCellIndex = functionality.nextCellIndex;

                assert(nextCellIndex != particle.cellIndex);
                assert(particle.timeLeft >= 0);

                rank_t rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);

                if(BOOST_LIKELY(nextCellIndex < this->Ncells))
                {
                    // local neighbor
                    particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
                    particle.cellIndex = nextCellIndex;
                }
                else
                {
                    // a ghost point, check rank and index in rank
                    auto it = ranks_ghost_map.find(nextCellIndex);
                    if(it == ranks_ghost_map.end())
                    {
                        // leaving domain
                        MonteCarloParticleStatus status = this->boundaryCondition->apply(particle);
                        if(status == MonteCarloParticleStatus::REFLECT)
                        {}
                        else if(status == MonteCarloParticleStatus::REMOVE)
                        {
                            stepData.leaving.push_back(particle);
                            // remove particle from current list
                            removeParticle(i);
                        }
                        else
                        {
                            std::cout << "Unknown boundary condition for particle " << particle << std::endl;
                            exit(1);
                        }
                        continue;    
                    }

                    particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
                    auto [otherRank, neighborIndexInRank] = it->second;
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
                    eliminateParticle(i); // removed but only from my domain
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
                continue;
            }
        }

        if(transferCounter > 0)
        {
            this->TransferParticles(_rank, transferParticlesVec, transferRanks, transferCounter);
        }
        if(removeCounter > 0)
        {
            handler->RemoveParticles(removeParticlesVec, removeCounter);
        }
        if(length > 0)
        {
            next_active_ranks.push_back(_rank);
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
void MonteCarloManager<T, Grid>::MonteCarloManager::ResetAllBuffers(void)
{
    for(RankHandler *handler : this->rankHandlers)
    {
        handler->Reset();
    }
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::MonteCarloManager::ShrinkAllBuffers(void)
{
    for(rank_t rank1 = 0; rank1 < this->size_world; rank1++)
    {
        for(rank_t rank2 = 0; rank2 <= rank1; rank2++)
        {
            MPI_Barrier(this->comm_world);
            if(this->rank_world == rank1 or this->rank_world == rank2)
            {
                // free handler
                rank_t otherRank = (rank1 == this->rank_world)? rank2 : rank1;
                this->rankHandlers[otherRank]->Reallocate(BUFFER_SHRINK_FACTOR);
            }
        }
    }
}

template<typename T, typename Grid>
std::vector<typename MonteCarloManager<T, Grid>::MCParticle> MonteCarloManager<T, Grid>::MonteCarloManager::step(const std::vector<MCParticle> &particleList, dt_t fullDt)
{
    this->Ncells = this->grid.GetPointNo();
    this->GetGhostMap();
    std::tie(this->ll, this->ur) = this->grid.GetBoxCoordinates();

    if(this->currentStep % 100 == 0)
    {
        this->ShrinkAllBuffers();
    }
    this->ResetAllBuffers();
    this->PutSelfParticles(particleList);
    this->resetTracker();
    this->currentStep++;

    size_t totalParticles = 0;
    for(RankHandler *handler : this->rankHandlers)
    {
        int length = *handler->th_length;
        totalParticles += length;
        for(int i = 0; i < length; i++)
        {
            size_t particleIndex = handler->th[i];
            MCParticle &p = handler->particles[particleIndex];
            p.timeLeft = fullDt;
            p.initialWeight = p.weight;
            p.steps = 0;
        }
    }

    this->progress = std::make_shared<ProgressCounter>(this->comm_world, totalParticles);

    RankHandler *handler = this->rankHandlers[this->rank_world];
    size_t numParticles = *handler->th_length;
    
    this->progress->localDecrementAmount = 0;
    
    volatile int &done = *this->progress->is_done;

    this->physics->updateGridData();
    std::vector<MCParticle> newParticles1 = this->physics->preStep(fullDt);
    this->AddParticles(newParticles1);
    MPI_Barrier(this->comm_world);
    
    MonteCarloStepFinalData data;
    // measure time
    // vtune_start();
    auto start = std::chrono::high_resolution_clock::now();
    while(not done)
    {
        this->HandleAll(data);
        this->progress->Sync();
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::vector<MCParticle> populationControlParticles = this->populationControl->activate(data.remaining);
    this->physics->postStep(populationControlParticles);

    MPI_Barrier(this->comm_world);
    double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    // std::cout << "Rank " << this->rank_world << " is outside of step() loop, in " << seconds << " seconds (" << numParticles << " particles)" << std::endl;

    size_t newParticlesNum = populationControlParticles.size();
    size_t leavingNumber = data.leaving.size();
    // std::cout << "leavingNumber = " << leavingNumber << " and newParticlesNum = " << newParticlesNum << std::endl; 
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &leavingNumber, &leavingNumber, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &newParticlesNum, &newParticlesNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    if(this->rank_world == 0)
    {
        // std::cout << "Number of leaving particles is " << leavingNumber << " and remaining (after population control) " << newParticlesNum << std::endl;
    }
    MPI_Barrier(this->comm_world);
    // vtune_stop();
    // return data.finalData;
    return populationControlParticles;
}

#endif // MONTE_CARLO_MANAGER_HPP