#ifndef MONTE_CARLO_MANAGER_HPP
#define MONTE_CARLO_MANAGER_HPP

#include "mpi/mpi_commands.hpp"
#include "mpi/serialize/mpi_commands.hpp"
#include "monte/MonteCarloParticle.hpp"
#include "monte/physics/MonteCarloPhysics.hpp"
#include "monte/population/PopulationControl.hpp"
// #include "tools/ProgressCounter.hpp"
#include "tools/ParticleAmountManager.hpp"
#include "monte/boundary/BoundaryCondition.hpp"
#include "monte/utils/GhostMap.hpp"
#include "monte/utils/RankSync.hpp"
#include "utils/debug/vtune.h" // TODO: remove
#include "RankHandler.hpp"
#include "ReallocationAgent.hpp"
#include "utils/debug/SmartTimer.hpp"
#include <memory>
#include <random>
#include <mpi.h>

#define MONTECARLO_EPSILON 1e-8
#define DEFAULT_BUFFER_SIZE 1000
#define MONTECARLO_CHANGE_TAG 1280
#define SHRINK_BUFFERS_CYCLE 50


template<typename Grid>
std::vector<rank_t> GetNeighborList(const Grid &tess, const boost::container::flat_map<size_t, std::pair<rank_t, size_t>> &ghostsMap)
{
    size_t N = tess.GetPointNo();
    boost::container::flat_set<rank_t> ranks;

    std::vector<size_t> allNeighboringGhosts;
    for(size_t i = 0; i < N; i++)
    {
        for(size_t ghostIdx : tess.GetNeighbors(i))
        {
            if(ghostIdx >= N)
            {
                auto it = ghostsMap.find(ghostIdx);
                if(it != ghostsMap.end())
                {
                    rank_t ownerRank = (*it).second.first;
                    ranks.insert(ownerRank);
                }
            }
        }
    }

    return std::vector<rank_t>(ranks.cbegin(), ranks.cend());
}

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

    void TransferParticles(const std::vector<rank_t> &rankBuffers, const std::vector<std::vector<size_t>> &indicesInToHandle, const std::vector<std::vector<rank_t>> &transferRanks);

    inline size_t GetStepCounter(void) const{return this->allStepsCounter;};

    inline const std::vector<size_t> &GetCellsStepsCounters(void) const {return this->cellsStepsCounters;}

    std::vector<MCParticle> step(const std::vector<MCParticle> &particleList, dt_t fullDt);
    
    class Tracker
    {
    public:
        Tracker(const MPI_Comm &comm);

        void Reset(void);

        #ifdef RICH_MPI
            std::vector<MCParticle> GetLocalTrackParticleRoute(size_t id);
        #endif // RICH_MPI

        std::vector<MCParticle> GetTrackParticleRoute(size_t id);

        void ReportParticle(MCParticle &particle);
    
    private:
        const MPI_Comm &comm;
        boost::container::flat_map<size_t, std::vector<MCParticle>> track;
    };

    inline const Tracker &getTracker(void){return this->tracker;};

    inline void resetTracker(void){this->tracker.Reset();};

private:
    const Grid &grid;
    MPI_Comm comm_world;
    rank_t rank_world, size_world;
    size_t Ncells;
    // std::shared_ptr<ProgressCounter> progress;
    int localDecrementAmount;
    std::vector<MPI_Comm> communicators;
    std::vector<rank_t> ranksOrder;
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ranks_ghost_map;
    std::vector<RankHandler*> rankHandlers;
    T ll, ur;
    std::shared_ptr<MonteCarloPhysics<T, Grid>> physics;
    std::shared_ptr<PopulationControl<T, Grid>> populationControl;
    std::shared_ptr<BoundaryCondition<T, Grid>> boundaryCondition;
    std::shared_ptr<ParticleAmountManager> amountManager;
    Tracker tracker;
    std::shared_ptr<ReallocationAgent> reallocationAgent;
    size_t myIDCounter;
    size_t currentStep;
    size_t allStepsCounter;
    size_t transfersCounter;
    std::vector<rank_t> neighbors;
    std::vector<size_t> cellsStepsCounters;
    size_t iteration;
    size_t dynamicallyAdded;
    size_t maxConsecutiveSteps;
    double maxConsecutiveStepsTime;
    
    bool HandleAll(MonteCarloStepFinalData &stepData);

    void PutSelfParticles(const std::vector<MCParticle> &particles);

    void FreeHandlers(void);

    void AddParticles(const std::vector<MCParticle> &particles);

    void ResetAllBuffers(void); 

    void ShrinkAllBuffers(void);
    
    void InitializeHandlers(size_t bufferSizes);
};

template<typename T, typename Grid>
MonteCarloManager<T, Grid>::MonteCarloManager(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics, const std::shared_ptr<PopulationControl<T, Grid>> &populationControl, 
                                            const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition, size_t bufferSizes, const MPI_Comm &comm):
    grid(grid), physics(physics), populationControl(populationControl), boundaryCondition(boundaryCondition), comm_world(MPI_COMM_NULL), tracker(comm)
{
    this->myIDCounter = 0;
    this->currentStep = 0;
    // this->progress = std::make_shared<ProgressCounter>(comm);
    this->SetCommunicator(comm);
    this->InitializeHandlers(bufferSizes);
    this->amountManager = std::make_shared<ParticleAmountManager>(this->comm_world, true /* use RDMA */);

    if(this->rank_world == 0)
    {
        std::cout << "Done initializing MonteCarloManager" << std::endl;
    }
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::InitializeHandlers(size_t bufferSizes)
{
    this->rankHandlers = std::vector<RankHandler*>(this->size_world, nullptr);

    auto createHandler = [&](rank_t _rank)
    {
        this->rankHandlers[_rank] = new RankHandler(bufferSizes, this->comm_world, this->communicators[_rank], this->reallocationAgent);
        if(this->rankHandlers[_rank]->peer_rank_world != _rank)
        {
            UniversalError eo("Peer rank world does not match");
            eo.addEntry("Rank", _rank);
            eo.addEntry("Peer Rank World", this->rankHandlers[_rank]->peer_rank_world);
            throw eo;
        }
    };
    
    ForEachRankSync(this->comm_world, this->ranksOrder, createHandler);

    MPI_Barrier(this->comm_world);
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::ClearCommunicator()
{
    if(this->comm_world == MPI_COMM_NULL)
    {
        return;
    }

    if(this->communicators.size() < this->size_world)
    {
        return;
    }

    auto clearRankComm = [this](rank_t _rank)
    {
        MPI_Comm &comm = this->communicators[_rank];
        if(comm == MPI_COMM_NULL)
        {
            return;
        }
        MPI_Comm_free(&comm);
    };

    ForEachRankSync(this->comm_world, this->ranksOrder, clearRankComm);

    this->comm_world = MPI_COMM_NULL;
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::SetCommunicator(const MPI_Comm &comm)
{
    this->ClearCommunicator();

    this->comm_world = comm;
    MPI_Comm_rank(this->comm_world, &this->rank_world);
    MPI_Comm_size(this->comm_world, &this->size_world);

    this->ranksOrder = GetRanksOrder(this->comm_world);

    this->communicators = std::vector<MPI_Comm>(this->size_world, MPI_COMM_NULL);

    MPI_Group worldGroup;
    MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
    auto setComm = [&](rank_t _rank)
    {
        MPI_Group group;
        int ranks[2] = {std::min(this->rank_world, _rank), std::max(this->rank_world, _rank)};
        int tag = ranks[0] * this->size_world + ranks[1];
        MPI_Group_incl(worldGroup, (_rank == this->rank_world)? 1 : 2, ranks, &group);
        MPI_Comm_create_group(this->comm_world, group, tag, &this->communicators[_rank]);
        MPI_Group_free(&group);
    };

    ForEachRankSync(this->comm_world, this->ranksOrder, setComm);
    
    MPI_Group_free(&worldGroup);

    // this->communicators = std::vector<MPI_Comm>(this->size_world, MPI_COMM_NULL);

    // for(int rank1 = 0; rank1 < this->size_world; rank1++)
    // {
    //     for(int rank2 = 0; rank2 <= rank1; rank2++)
    //     {
    //         MPI_Barrier(this->comm_world);
    //         int color = (this->rank_world == rank1 || this->rank_world == rank2) ? 1 : MPI_UNDEFINED;

    //         MPI_Comm new_comm = MPI_COMM_NULL;
    //         MPI_Comm_split(this->comm_world, color, this->rank_world, &new_comm);
    //         this->communicators.push_back(new_comm);

    //         if(this->rank_world == rank1 or this->rank_world == rank2)
    //         {
    //             rank_t otherRank = (rank1 == this->rank_world)? rank2 : rank1;
    //             this->communicators[otherRank] = new_comm;
    //         }
    //     }
    // }


    auto reallocationFunction = [this](rank_t rank)
    {
        this->rankHandlers[rank]->Reallocate(BUFFER_REALLOCATION_FACTOR);
    };

    this->reallocationAgent = std::make_shared<ReallocationAgent>(this->comm_world, reallocationFunction);
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::FreeHandlers(void)
{
    auto freeHandler = [&](rank_t _rank)
    {
        this->rankHandlers[_rank]->Destroy();
        delete this->rankHandlers[_rank];    
    };
    
    ForEachRankSync(this->comm_world, this->ranksOrder, freeHandler);
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

    if(*myHandler->av_length < particles.size())
    {
        double factor = std::max<double>(BUFFER_REALLOCATION_FACTOR, std::ceil(static_cast<double>(particles.size() + myHandler->buffsize) / static_cast<double>(myHandler->buffsize)));
        myHandler->Reallocate(factor);
        assert(*myHandler->av_length >= particles.size());
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

        #ifdef MONTECARLO_DEBUG
            particle->checkedHere = true;
            particle->nextRank = std::numeric_limits<rank_t>::max();
            particle->removedFromRank = false;
            particle->sentByRank = std::numeric_limits<rank_t>::max();
            particle->lastSeen = 0;
        #endif // MONTECARLO_DEBUG

        #ifdef MONTECARLO_DEBUG
        if(not this->grid.IsPointInCell(particle->location, particle->cellIndex))
        {
            const T &declaredCell = this->grid.GetMeshPoint(particle->cellIndex);
            size_t containingIdx = this->grid.GetContainingCell(particle->location);
            const T &containingCell = this->grid.GetMeshPoint(containingIdx);
            UniversalError eo("MonteCarloManager<T, Grid>::AddParticles");
            eo.addEntry("rank", this->rank_world);
            eo.addEntry("Particle", *particle);
            eo.addEntry("Declared Cell Index", particle->cellIndex);
            eo.addEntry("Declared Cell", declaredCell);
            eo.addEntry("Declared Cell - Distance", abs(declaredCell - particle->location));
            eo.addEntry("Real Containing Cell Index", containingIdx);
            eo.addEntry("Real Containing Cell", containingCell);
            eo.addEntry("Real Cell - Distance", abs(containingCell - particle->location));
            throw eo;
        }
        #endif // MONTECARLO_DEBUG
    }

    this->localDecrementAmount -= static_cast<int>(particlesNum);
    // std::cout << "Done add particles" << std::endl;
}

template<typename T, typename Grid>
MonteCarloManager<T, Grid>::~MonteCarloManager()
{
    if(not std::uncaught_exceptions())
    {
        this->FreeHandlers();
        this->ClearCommunicator();
    }
}

template<typename T, typename Grid>
MonteCarloManager<T, Grid>::Tracker::Tracker(const MPI_Comm &comm): comm(comm)
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
    std::vector<MCParticle> global = MPI_All_cast(local, this->comm);
    // sort by `particle.steps`
    std::sort(global.begin(), global.end(), [](const MCParticle &a, const MCParticle &b) { return a.steps < b.steps; });
    return global;
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::MonteCarloManager::PutSelfParticles(const std::vector<MCParticle> &particles)
{
    using index_t = typename RankHandler::index_t;

    RankHandler *handler = this->rankHandlers[this->rank_world];

    #ifdef MONTECARLO_DEBUG
    boost::container::flat_set<std::pair<rank_t, size_t>> particlesSet;
    for(const MCParticle &particle : particles)
    {
        if(particle.id == std::numeric_limits<size_t>::max())
        {
            continue;
        }
        std::pair<rank_t, size_t> particleSetKey = {particle.rank, particle.id};
        if(particlesSet.find(particleSetKey) != particlesSet.end())
        {
            UniversalError eo("Particle with the same ID is being added to the same rank twice");
            eo.addEntry("Particle", particle);
            eo.addEntry("Rank", this->rank_world);
            eo.addEntry("ID", particle.id);
            throw eo;
        }
        particlesSet.insert(particleSetKey);
    }
    #endif // MONTECARLO_DEBUG

    size_t particlesNum = particles.size();
    if(handler->buffsize < particlesNum)
    {
        // reallocate buffer if needed
        double factor = std::ceil(static_cast<double>(particlesNum) / static_cast<double>(handler->buffsize));
        handler->Reallocate(factor);
    }

    *handler->av_length -= static_cast<int>(particlesNum);
    index_t *av_indices = handler->av + *handler->av_length;
    *handler->th_length += particlesNum;

    for(size_t i = 0; i < particlesNum; i++)
    {
        size_t particleIdx = av_indices[i];
        handler->th[i] = particleIdx;
        std::memcpy(handler->particles + particleIdx, &particles[i], sizeof(MCParticle));
        MCParticle &particle = handler->particles[particleIdx];
        if(particle.id == std::numeric_limits<size_t>::max())
        {
            // no ID has been assigned
            particle.rank = this->rank_world;
            particle.id = this->myIDCounter++;
        }
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

    this->transfersCounter++;

    boost::container::flat_map<rank_t, std::vector<MCParticle>> rankToParticles;
    #ifdef MONTECARLO_DEBUG
        boost::container::flat_map<size_t, rank_t> sentAndToWhom;
    #endif // MONTECARLO_DEBUG
    RankHandler *currRankHandler = this->rankHandlers[fromRank];

    for(size_t i = 0; i < num; i++)
    {
        const size_t &indexInToHandle = indicesInToHandle[i];
        const rank_t &toRank = transferRanks[i];
        assert(toRank != this->rank_world); // can't send to self
        size_t particleIdx = currRankHandler->th[indexInToHandle];
        auto it = rankToParticles.find(toRank);
        if(it == rankToParticles.end())
        {
            rankToParticles[toRank] = std::vector<MCParticle>();
        }

        #ifdef MONTECARLO_DEBUG
            if(sentAndToWhom.find(particleIdx) == sentAndToWhom.end())
            {
                sentAndToWhom[particleIdx] = toRank;
            }
            else
            {
                UniversalError eo("Particle is being sent to multiple ranks");
                eo.addEntry("Particle Index", particleIdx);
                eo.addEntry("Particle", currRankHandler->particles[particleIdx]);
                eo.addEntry("I am rank", this->rank_world);
                eo.addEntry("From Rank Buffer", fromRank);
                eo.addEntry("Already Sent To", sentAndToWhom[particleIdx]);
                eo.addEntry("Now Sending To", toRank);
                throw eo;
            }
        #endif // MONTECARLO_DEBUG
        MCParticle &particle = currRankHandler->particles[particleIdx];        
        particle.sent = false; // reset

        // std::cout << "Rank " << this->rank_world << " transfers particle TH = " << indexInToHandle << ", particle index " << particleIdx << " (particle: " << particle << ") to rank " << toRank << std::endl; 
        
        if(toRank == this->rank_world)
        {
            UniversalError eo("Trying to transfer particle to the same rank");
            eo.addEntry("Particle", particle);
            eo.addEntry("From Rank", fromRank);
            eo.addEntry("To Rank", toRank);
            throw eo;
        }
        #ifdef MONTECARLO_DEBUG
        if(std::find_if(rankToParticles[toRank].begin(), rankToParticles[toRank].end(),
                        [&particle](const MCParticle &p) { return p == particle; }) != rankToParticles[toRank].end())
        {
            UniversalError eo("Particle with the same ID is being sent to the same rank twice");
            eo.addEntry("Index In Transfer Queue", i);
            eo.addEntry("Particle Index", particleIdx);
            eo.addEntry("Particle", particle);
            for(size_t j = 0; j < i; j++)
            {
                size_t indexInToHandle2 = indicesInToHandle[j];
                rank_t toRank2 = transferRanks[j];
                size_t particle2Idx = currRankHandler->th[indexInToHandle2];
                const MCParticle &particle2 = currRankHandler->particles[particle2Idx];
                if(toRank2 == toRank and particle2 == particle)
                {
                    eo.addEntry("Already Appeared In Index", j);
                    eo.addEntry("Particle2", particle2);
                    eo.addEntry("Particle 2 Index", particle2Idx);
                    break;
                }
            }
            eo.addEntry("From Rank", fromRank);
            eo.addEntry("To Rank", toRank);
            throw eo;
        }
        #endif // MONTECARLO_DEBUG

        rankToParticles[toRank].push_back(particle);

        #ifdef MONTECARLO_DEBUG
        if(toRank != particle.nextRank)
        {
            UniversalError eo("Particle will not be sent to the expected rank #1");
            eo.addEntry("Particle", particle);
            eo.addEntry("Origin", particle.sentByRank);
            eo.addEntry("Expected Rank", toRank);
            eo.addEntry("Next Rank", particle.nextRank);
            throw eo;
        }
        #endif // MONTECARLO_DEBUG
    }

    for(const auto &[toRank, particles] : rankToParticles)
    {
        assert(toRank != this->rank_world); // can't send to self
        RankHandler *remoteHandler = this->rankHandlers[toRank];
        assert(remoteHandler->peer_rank_world == toRank);
        #ifdef MONTECARLO_DEBUG
        if(remoteHandler->peer_rank_world != toRank)
        {
            UniversalError eo("Remote handler has wrong peer rank world");
            eo.addEntry("Expected", toRank);
            eo.addEntry("Got", remoteHandler->peer_rank_world);
            throw eo;
        }
        for(const MCParticle &particle : particles)
        {
            if(particle.nextRank != toRank)
            {
                UniversalError eo("Particle will not be sent to the expected rank #2");
                eo.addEntry("Particle", particle);
                eo.addEntry("Origin", particle.sentByRank);
                eo.addEntry("Expected Rank", toRank);
                eo.addEntry("Next Rank", particle.nextRank);
                throw eo;
            }
        }
        #endif // MONTECARLO_DEBUG
        // std::cout << "Migrating particle " << *particle << ", from rankbuff " << rankBuffer << " to rank " << toRank << std::endl;
        remoteHandler->TransferParticles(particles);
    }
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::MonteCarloManager::TransferParticles(const std::vector<rank_t> &rankBuffers, const std::vector<std::vector<size_t>> &indicesInToHandle, const std::vector<std::vector<rank_t>> &transferRanks)
{
    if(indicesInToHandle.empty())
    {
        // nothing to transfer
        return;
    }

    this->transfersCounter++;

    boost::container::flat_map<rank_t, std::vector<MCParticle>> rankToParticles;
    #ifdef MONTECARLO_DEBUG
        boost::container::flat_map<std::pair<rank_t, size_t>, rank_t> sentAndToWhom;
    #endif // MONTECARLO_DEBUG
    
    assert(rankBuffers.size() == indicesInToHandle.size());

    size_t numRanks = rankBuffers.size();
    for(size_t i = 0; i < numRanks; i++)
    {
        const rank_t &fromRank = rankBuffers[i];
        RankHandler *currRankHandler = this->rankHandlers[fromRank];
        const std::vector<size_t> &myTHIndices = indicesInToHandle[i];
        size_t numToHandle = myTHIndices.size();
        const std::vector<rank_t> &myTransferRanks = transferRanks[i];

        for(size_t j = 0; j < numToHandle; j++)
        {
            const size_t &indexInToHandle = myTHIndices[j];
            const rank_t &toRank = myTransferRanks[j];

            assert(toRank != this->rank_world); // can't send to self
            size_t particleIdx = currRankHandler->th[indexInToHandle];
            auto it = rankToParticles.find(toRank);
            if(it == rankToParticles.end())
            {
                rankToParticles[toRank] = std::vector<MCParticle>();
            }

            #ifdef MONTECARLO_DEBUG
                std::pair<rank_t, size_t> particleKey = {fromRank, particleIdx};
                if(sentAndToWhom.find(particleKey) == sentAndToWhom.end())
                {
                    sentAndToWhom[particleKey] = toRank;
                }
                else
                {
                    UniversalError eo("Particle is being sent to multiple ranks");
                    eo.addEntry("Particle Index", particleIdx);
                    eo.addEntry("Particle", currRankHandler->particles[particleIdx]);
                    eo.addEntry("I am rank", this->rank_world);
                    eo.addEntry("From Rank Buffer", fromRank);
                    eo.addEntry("Already Sent To", sentAndToWhom[particleKey]);
                    eo.addEntry("Now Sending To", toRank);
                    throw eo;
                }
            #endif // MONTECARLO_DEBUG
            MCParticle &particle = currRankHandler->particles[particleIdx];        
            particle.sent = false; // reset

            // std::cout << "Rank " << this->rank_world << " transfers particle TH = " << indexInToHandle << ", particle index " << particleIdx << " (particle: " << particle << ") to rank " << toRank << std::endl; 
            
            if(toRank == this->rank_world)
            {
                UniversalError eo("Trying to transfer particle to the same rank");
                eo.addEntry("Particle", particle);
                eo.addEntry("From Rank", fromRank);
                eo.addEntry("To Rank", toRank);
                throw eo;
            }
            #ifdef MONTECARLO_DEBUG
            if(std::find_if(rankToParticles[toRank].begin(), rankToParticles[toRank].end(),
                            [&particle](const MCParticle &p) { return p == particle; }) != rankToParticles[toRank].end())
            {
                UniversalError eo("Particle with the same ID is being sent to the same rank twice");
                eo.addEntry("Index In Transfer Queue", i);
                eo.addEntry("Particle Index", particleIdx);
                eo.addEntry("Particle", particle);
                for(size_t j = 0; j < i; j++)
                {
                    size_t indexInToHandle2 = myTHIndices[j];
                    rank_t toRank2 = myTransferRanks[j];
                    size_t particle2Idx = currRankHandler->th[indexInToHandle2];
                    const MCParticle &particle2 = currRankHandler->particles[particle2Idx];
                    if(toRank2 == toRank and particle2 == particle)
                    {
                        eo.addEntry("Already Appeared In Index", j);
                        eo.addEntry("Particle2", particle2);
                        eo.addEntry("Particle 2 Index", particle2Idx);
                        break;
                    }
                }
                eo.addEntry("From Rank", fromRank);
                eo.addEntry("To Rank", toRank);
                throw eo;
            }
            #endif // MONTECARLO_DEBUG

            rankToParticles[toRank].push_back(particle);

            #ifdef MONTECARLO_DEBUG
            if(toRank != particle.nextRank)
            {
                UniversalError eo("Particle will not be sent to the expected rank #1");
                eo.addEntry("Particle", particle);
                eo.addEntry("Origin", particle.sentByRank);
                eo.addEntry("Expected Rank", toRank);
                eo.addEntry("Next Rank", particle.nextRank);
                throw eo;
            }
            #endif // MONTECARLO_DEBUG
        }
    }

    for(const auto &[toRank, particles] : rankToParticles)
    {
        assert(toRank != this->rank_world); // can't send to self
        RankHandler *remoteHandler = this->rankHandlers[toRank];
        assert(remoteHandler->peer_rank_world == toRank);
        #ifdef MONTECARLO_DEBUG
        if(remoteHandler->peer_rank_world != toRank)
        {
            UniversalError eo("Remote handler has wrong peer rank world");
            eo.addEntry("Expected", toRank);
            eo.addEntry("Got", remoteHandler->peer_rank_world);
            throw eo;
        }
        for(const MCParticle &particle : particles)
        {
            if(particle.nextRank != toRank)
            {
                UniversalError eo("Particle will not be sent to the expected rank #2");
                eo.addEntry("Particle", particle);
                eo.addEntry("Origin", particle.sentByRank);
                eo.addEntry("Expected Rank", toRank);
                eo.addEntry("Next Rank", particle.nextRank);
                throw eo;
            }
        }
        #endif // MONTECARLO_DEBUG
        // std::cout << "Migrating particle " << *particle << ", from rankbuff " << rankBuffer << " to rank " << toRank << std::endl;
        remoteHandler->TransferParticles(particles);
    }
}

template<typename T, typename Grid>
bool MonteCarloManager<T, Grid>::MonteCarloManager::HandleAll(MonteCarloStepFinalData &stepData)
{
    static std::vector<rank_t> active_ranks;
    static std::vector<rank_t> next_active_ranks;
    static std::vector<std::vector<size_t>> removeParticlesVec;
    static std::vector<std::vector<rank_t>> transferToRanks;
    static std::vector<std::vector<size_t>> transferParticlesVec;
    static std::vector<MCParticle> particlesToAdd;

    // static std::uniform_real_distribution<double> dist(0, 1);
    // static std::mt19937 re(this->rank_world);
    
    next_active_ranks.clear();
    if(active_ranks.empty())
    {
        // std::cout << "active_ranks is empty" << std::endl;
        const int PREFETCH_DISTANCE = 3;
        size_t N = this->neighbors.size();

        for (size_t i = 0; i < N; ++i)
        {
            // Prefetch future data to hide memory latency
            if (i + PREFETCH_DISTANCE < N)
            {
                rank_t future_rank = this->neighbors[i + PREFETCH_DISTANCE];

                // Prefetch the RankHandler* object (heap-allocated, likely scattered)
                RankHandler* future_handler = this->rankHandlers[future_rank];
                __builtin_prefetch(future_handler, 0, 1);                  // bring RankHandler into cache
                __builtin_prefetch((const void*) (future_handler->th_length), 0, 1);       // bring th_length target into cache
            }

            // Access current handler
            rank_t _rank = this->neighbors[i];
            RankHandler* handler = this->rankHandlers[_rank];

            // Cache the dereferenced value to avoid repeated indirection
            int len = *handler->th_length;

            // Only proceed if there's work to do
            if(len)
            {
                active_ranks.push_back(_rank);
            }
        }
        {
            RankHandler *handler = this->rankHandlers[this->rank_world];
            // std::cout << "Running sync on window of rank " << std::get<0>(buffer) << std::endl;
            // handler->Sync(); // todo: what about that?
            // std::cout << "Done!" << std::endl;
            if(*handler->th_length > 0)
            {
                active_ranks.push_back(this->rank_world);
            }
        }
    }

    bool isEmpty = true;
    size_t activeRanksNum = active_ranks.size();

    auto eliminateParticle = [&](size_t rankIndex, size_t particleTH)
    {
        removeParticlesVec[rankIndex].push_back(particleTH);
    };
    
    auto transferParticle = [&](size_t rankIndex, size_t particleTH, rank_t toRank)
    {
        assert(toRank != this->rank_world); // can't send to self
        transferToRanks[rankIndex].push_back(toRank);
        transferParticlesVec[rankIndex].push_back(particleTH);
        eliminateParticle(rankIndex, particleTH);
    };

    auto removeParticle = [&](size_t rankIndex, size_t particleTH)
    {
        eliminateParticle(rankIndex, particleTH);
        this->localDecrementAmount += 1;
    };

    transferParticlesVec.clear();
    transferToRanks.clear();
    removeParticlesVec.clear();

    for(size_t index = 0; index < activeRanksNum; index++)
    {
        rank_t _rank = active_ranks[index];
        RankHandler *handler = this->rankHandlers[_rank];
        volatile int &length = *handler->th_length;

        std::vector<size_t> &TransferParticlesVecOfRank = transferParticlesVec.emplace_back();
        transferToRanks.emplace_back();
        removeParticlesVec.emplace_back();

        #ifdef ADVANCED_MONTECARLO_DEBUG
            handler->LockSelfBuffer();
        #endif // ADVANCED_MONTECARLO_DEBUG
        
        size_t consecutiveSteps = 0;
        auto stepStartTime = std::chrono::high_resolution_clock::now();

        for(int i = 0; i < length; i++)
        {
            assert(i < handler->buffsize);
            size_t particleIndex = handler->th[i];
            assert(particleIndex < handler->buffsize);
            MCParticle &particle = handler->particles[particleIndex];
            bool debug = false;

            #ifdef MONTECARLO_DEBUG
            if(particle.lastSeen == this->iteration and particle.lastSeenRank == this->rank_world)
            {
                UniversalError eo("Particle was already handled in this iteration");
                eo.addEntry("My Rank", this->rank_world);
                eo.addEntry("Particle", particle);
                eo.addEntry("Iteration", this->iteration);
                eo.addEntry("In Rank Buffer (1)", particle.lastSeenRankBuf);
                eo.addEntry("In TH Index (1)", particle.lastSeenIndex);
                eo.addEntry("In Rank Buffer (2)", _rank);
                eo.addEntry("In TH Index (2)", i);
                throw eo;
            }
            particle.lastSeen = this->iteration;
            particle.lastSeenRankBuf = _rank;
            particle.lastSeenRank = this->rank_world;
            particle.lastSeenIndex = i;
            #endif // MONTECARLO_DEBUG

            isEmpty = false;
            while(true)
            {
                consecutiveSteps++;
                // debug = debug or (particle.id == 6480574 and particle.rank == 21);
                // debug = debug or (particle.id == 6531002 and particle.rank == 9);
                // debug = debug or (particle.id == 6531241 and particle.rank == 27);
                // debug = debug or (particle.id == 6582636 and particle.rank == 10);

                // TODO: shouldn't be, there's a bug
                // if(particle.sent)
                // {
                //     continue;
                // }
                
                // std::cout << "Rank " << this->rank_world << " handles TH = " << i << ", which is index " << particleIndex << ", particle: " << particle << std::endl;

                if(particle.on_track)
                {
                    this->tracker.ReportParticle(particle);
                }
                particle.steps++;
                this->cellsStepsCounters[particle.cellIndex]++;
                
                // std::cout << "Rank " << this->rank_world << " handles particle " << particle.id << " of rank " << particle.rank << ", step " << particle.steps << std::endl;

                #ifdef MONTECARLO_DEBUG
                if(particle.cellIndex >= this->Ncells)
                {
                    UniversalError eo("Particle has invalid cell index (ghost)");
                    eo.addEntry("Particle", particle);
                    eo.addEntry("Cell Index", particle.cellIndex);
                    eo.addEntry("Rank", this->rank_world);
                    eo.addEntry("Buffer of Rank", _rank);
                    throw eo;
                }
                if(particle.removedFromRank)
                {
                    continue; 
                    UniversalError eo("Particle was removed from rank, but still in the list");
                    eo.addEntry("Particle", particle);
                    eo.addEntry("Rank", this->rank_world);
                    eo.addEntry("Buffer of Rank", _rank);
                    throw eo;
                }
                if(not particle.checkedHere)
                {
                    if(particle.nextRank != this->rank_world)
                    {
                        // particle is in the right cell, but not in the right place
                        UniversalError eo("Particle Arrived to a Wrong Rank After Transfer");
                        eo.addEntry("Particle", particle);
                        eo.addEntry("Origin", particle.sentByRank);
                        eo.addEntry("Particle Previous Location", particle.previousLocation);
                        eo.addEntry("Cell Index In Origin (Before Movement)", particle.cellIndexInPrevRank);
                        eo.addEntry("Expected", particle.nextRank);
                        eo.addEntry("Got (me)", this->rank_world);
                        eo.addEntry("The Particle Index In Last Rank", particle.particleIndexInLastRank);
                        eo.addEntry("Particle Index In This Rank", particleIndex);
                        eo.addEntry("The Particle TH In Last Rank", particle.particleTHInLastRank);
                        eo.addEntry("Particle TH In This Rank", i);
                        eo.addEntry("New Cell Index Should Be", particle.cellIndex); 
                        eo.addEntry("New Cell Value Should Be", particle.newCellValue); 
                        throw eo;
                    }
                    particle.checkedHere = true;
                    particle.nextRank = std::numeric_limits<rank_t>::max();
                    particle.removedFromRank = false;
                    particle.sentByRank = std::numeric_limits<rank_t>::max();
                }
                if(not this->grid.IsPointInCell(particle.location, particle.cellIndex))
                {
                    const T &declaredCell = this->grid.GetMeshPoint(particle.cellIndex);
                    size_t containingIdx = this->grid.GetContainingCell(particle.location);
                    const T &containingCell = this->grid.GetMeshPoint(containingIdx);
                    if(containingIdx != particle.cellIndex)
                    {
                        if(not this->grid.IsPointInCell(particle.location, containingIdx))
                        {
                            // particle is in the right cell, but not in the right place
                            UniversalError eo("Particle Arrived to a Wrong Rank After Transfer");
                            eo.addEntry("My Rank", this->rank_world);
                            eo.addEntry("Transferred From Rank", _rank);
                            eo.addEntry("Particle", particle);
                            eo.addEntry("Cell Index Transffered From Previous Rank", particle.cellIndexInPrevRank);
                            eo.addEntry("Ghost Index In Previous Rank", particle.ghostIndex);
                            eo.addEntry("New Cell Value Should Be", particle.newCellValue); 
                            eo.addEntry("Declared Cell Index", particle.cellIndex);
                            eo.addEntry("Declared Cell", declaredCell);
                            eo.addEntry("Declared Cell - Distance", abs(declaredCell - particle.location));
                            eo.addEntry("Real Containing Cell Index", containingIdx);
                            eo.addEntry("Real Containing Cell", containingCell);
                            eo.addEntry("Real Cell - Distance", abs(containingCell - particle.location));
                            eo.addEntry("Particle Previous Location", particle.previousLocation);
                            eo.addEntry("Particle Previous Cell Index", particle.cellIndexInPrevRank);
                            throw eo;
                        }
                    }
                    if(abs(abs(declaredCell - particle.location) - abs(containingCell - particle.location)) >= 1e-12)
                    {
                        UniversalError eo("Particle is in Wrong Location After Transfer");
                        eo.addEntry("My Rank", this->rank_world);
                        eo.addEntry("Transferred From Rank", _rank);
                        eo.addEntry("Particle", particle);
                        eo.addEntry("Cell Index Transffered From Previous Rank", particle.cellIndexInPrevRank);
                        eo.addEntry("Particle Previous Location", particle.previousLocation);
                        eo.addEntry("Ghost Index In Previous Rank", particle.ghostIndex);
                        eo.addEntry("New Cell Value Should Be", particle.newCellValue);                        
                        eo.addEntry("Declared Cell Index", particle.cellIndex);
                        eo.addEntry("Declared Cell", declaredCell);
                        eo.addEntry("Declared Cell - Distance", abs(declaredCell - particle.location));
                        eo.addEntry("Real Containing Cell Index", containingIdx);
                        eo.addEntry("Real Containing Cell", containingCell);
                        eo.addEntry("Real Cell - Distance", abs(containingCell - particle.location));
                        for(const size_t &faceIdx : this->grid.GetCellFaces(particle.cellIndex))
                        {
                            eo.addEntry("Face Index", faceIdx);
                            eo.addEntry("Face normal", this->grid.Normal(faceIdx));
                            eo.addEntry("Face CM", this->grid.FaceCM(faceIdx));
                            eo.addEntry("Eucledian distance to face", std::abs(ScalarProd(particle.location - this->grid.FaceCM(faceIdx), this->grid.Normal(faceIdx))) / abs(this->grid.Normal(faceIdx)));
                        }
                        throw eo;
                    }
                }
                #endif // MONTECARLO_DEBUG
                T prevLoc = particle.location;
                #ifdef MONTECARLO_DEBUG
                    particle.previousLocation = particle.location;
                #endif // MONTECARLO_DEBUG
                MonteCarloFunctionality<T, Grid> functionality = this->physics->step(particle);

                // std::cout << "Handling particle " << particle << ", functionality is " << functionality.change << std::endl;
                if(debug)
                {
                    std::cout << "Particle " << particle << ", functionality is " << functionality.change << std::endl;
                }

                if(not functionality.particlesToAdd.empty())
                {
                    particlesToAdd.insert(particlesToAdd.end(), functionality.particlesToAdd.cbegin(), functionality.particlesToAdd.cend());
                }
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
                        size_t previousCell = particle.cellIndex;
                        particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
                        particle.cellIndex = nextCellIndex;
                        #ifdef MONTECARLO_DEBUG
                        if(not this->grid.IsPointInCell(particle.location, particle.cellIndex))
                        {
                            const T &declaredCell = this->grid.GetMeshPoint(particle.cellIndex);
                            size_t containingIdx = this->grid.GetContainingCell(particle.location);
                            const T &containingCell = this->grid.GetMeshPoint(containingIdx);
                            UniversalError eo("Particle is in Wrong Location");
                            eo.addEntry("rank", this->rank_world);
                            eo.addEntry("Particle", particle);
                            eo.addEntry("Previous Cell Index", previousCell);
                            eo.addEntry("Previous Cell", this->grid.GetMeshPoint(previousCell));
                            eo.addEntry("Previous Location", prevLoc);
                            eo.addEntry("Last location is in previous cell?", this->grid.IsPointInCell(prevLoc, previousCell));
                            eo.addEntry("Declared Cell Index", particle.cellIndex);
                            eo.addEntry("Declared Cell", declaredCell);
                            eo.addEntry("Declared Cell - Distance", abs(declaredCell - particle.location));
                            eo.addEntry("Real Containing Cell Index", containingIdx);
                            eo.addEntry("Real Containing Cell", containingCell);
                            eo.addEntry("Real Cell - Distance", abs(containingCell - particle.location));
                            for(const size_t &faceIdx : this->grid.GetCellFaces(particle.cellIndex))
                            {
                                eo.addEntry("Face Index", faceIdx);
                                eo.addEntry("Face normal", this->grid.Normal(faceIdx));
                                eo.addEntry("Face CM", this->grid.FaceCM(faceIdx));
                                eo.addEntry("Eucledian distance to face", std::abs(ScalarProd(particle.location - this->grid.FaceCM(faceIdx), this->grid.Normal(faceIdx))) / abs(this->grid.Normal(faceIdx)));
                            }
                            throw eo;
                        }
                        #endif // MONTECARLO_DEBUG
                    }
                    else
                    {
                        // a ghost point, check rank and index in rank
                        auto it = ranks_ghost_map.find(nextCellIndex);
                        if(it == ranks_ghost_map.end())
                        {
                            // leaving domain
                            MonteCarloParticleStatus status = this->boundaryCondition->apply(particle);
                            if(debug)
                            {
                                std::cout << "Particle " << particle << ", leaving domain. status from bounday condition: " << status << std::endl;
                            }
                            if(status == MonteCarloParticleStatus::REFLECT)
                            {}
                            else if(status == MonteCarloParticleStatus::REMOVE)
                            {
                                stepData.leaving.push_back(particle);
                                this->allStepsCounter += particle.steps;
                                // remove particle from current list
                                removeParticle(index, i);
                            }
                            else
                            {
                                std::cout << "Unknown boundary condition for particle " << particle << std::endl;
                                exit(1);
                            }
                            break;    
                        }

                        particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
                        auto [otherRank, neighborIndexInRank] = it->second;
                        #ifdef MONTECARLO_DEBUG
                        particle.checkedHere = false; // reset checked here flag
                        if(particle.nextRank != std::numeric_limits<rank_t>::max())
                        {
                            UniversalError eo("Particle was already sent, and not sent again");
                            eo.addEntry("Particle", particle);
                            eo.addEntry("Already Transferred To Rank", particle.nextRank);
                            eo.addEntry("Being Transferred To Rank", otherRank);
                            eo.addEntry("Being Transferred To Index In Rank", neighborIndexInRank);
                            throw eo;
                        }
                        const std::vector<rank_t> &neighbors = this->grid.GetDuplicatedProcs();
                        if(std::find(neighbors.cbegin(), neighbors.cend(), otherRank) == neighbors.cend())
                        {
                            UniversalError eo("Particle is going to be transffered to a non-neighboring rank");
                            eo.addEntry("Particle", particle);
                            eo.addEntry("My Rank", this->rank_world);
                            eo.addEntry("Next Rank", otherRank);
                            eo.addEntry("Index In Remote Rank", neighborIndexInRank);
                            throw eo;
                        }
                        particle.cellIndexInPrevRank = particle.cellIndex;
                        particle.sentByRank = this->rank_world;
                        particle.ghostIndex = nextCellIndex;
                        particle.newCellValue = this->grid.GetMeshPoint(nextCellIndex);
                        particle.particleIndexInLastRank = particleIndex;
                        particle.particleTHInLastRank = i;
                        particle.nextRank = otherRank;
                        particle.sent = true;
                        
                        if(particle.nextRank == this->rank_world)
                        {
                            UniversalError eo("Particle is going to be sent to the same rank");
                            eo.addEntry("Particle", particle);
                            eo.addEntry("My Rank", this->rank_world);
                            eo.addEntry("Next Rank", otherRank);
                            eo.addEntry("Index In Remote Rank", neighborIndexInRank);
                            throw eo;
                        }
                        #endif // MONTECARLO_DEBUG
                        particle.cellIndex = neighborIndexInRank;

                        #ifdef MONTECARLO_DEBUG
                        if(not TransferParticlesVecOfRank.empty())
                        {
                            size_t lastTHIndex = TransferParticlesVecOfRank.back();
                            size_t lastParticleIndex = handler->th[lastTHIndex];
                            const MCParticle &lastParticle = handler->particles[lastParticleIndex];
                            if(lastParticle == particle)
                            {
                                UniversalError eo("Particle is already in the transfer list");
                                eo.addEntry("Iteration", this->iteration);
                                eo.addEntry("Particle", particle);
                                eo.addEntry("My Rank", this->rank_world);
                                eo.addEntry("TH Index 1", lastTHIndex);
                                eo.addEntry("TH Index 2", i);
                                eo.addEntry("Length of Transfer List", TransferParticlesVecOfRank.size());
                                eo.addEntry("In Rank Buffer", _rank);
                                eo.addEntry("Sent to Rank", otherRank);
                                throw eo;
                            }
                        }
                        #endif // MONTECARLO_DEBUG

                        transferParticle(index, i, otherRank);
                        break; // TODO:
                    }
                }
                else if(functionality.change == MonteCarloParticleStatus::REMOVE)
                {
                    this->allStepsCounter += particle.steps;
                    removeParticle(index, i);
                    break;
                }
                else if(functionality.change == MonteCarloParticleStatus::DONE)
                {
                    stepData.remaining.push_back(particle);
                    this->allStepsCounter += particle.steps;
                    // remove particle from current list
                    removeParticle(index, i);
                    break;
                }
            }
        }
        
        if(length > 0)
        {
            next_active_ranks.push_back(_rank);
        }

        double consecutiveStepsTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stepStartTime).count();
        this->maxConsecutiveSteps = std::max(this->maxConsecutiveSteps, consecutiveSteps);
        this->maxConsecutiveStepsTime = std::max(this->maxConsecutiveStepsTime, consecutiveStepsTime);
        
        this->reallocationAgent->HandleAllWaitingReallocations();

        #ifdef ADVANCED_MONTECARLO_DEBUG
            handler->UnlockSelfBuffer();
        #endif // ADVANCED_MONTECARLO_DEBUG
    }

    this->TransferParticles(active_ranks, transferParticlesVec, transferToRanks);

    for(size_t i = 0; i < activeRanksNum; i++)
    {
        rank_t _rank = active_ranks[i];
        const std::vector<size_t> &rankRemoveParticlesVec = removeParticlesVec[i];
        if(rankRemoveParticlesVec.empty())
        {
            continue; // nothing to remove
        }
        RankHandler *handler = this->rankHandlers[_rank];
        handler->RemoveParticles(rankRemoveParticlesVec, rankRemoveParticlesVec.size());
    }
    active_ranks.swap(next_active_ranks);

    if(not particlesToAdd.empty())
    {
        this->dynamicallyAdded += particlesToAdd.size();
        this->AddParticles(particlesToAdd);
    }

    return isEmpty and particlesToAdd.empty();
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::MonteCarloManager::ResetAllBuffers(void)
{
    for(RankHandler *handler : this->rankHandlers)
    {
        if(handler != nullptr)
        {
            handler->Reset();
        }
    }
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::MonteCarloManager::ShrinkAllBuffers(void)
{
    auto shrinkBuffer = [&](rank_t _rank)
    {
        if(_rank == this->rank_world)
        {
            return;
        }
        if(std::find(this->neighbors.cbegin(), this->neighbors.cend(), _rank) != this->neighbors.cend())
        {
            // a neighbor - don't shrink
            return;
        }
        this->rankHandlers[_rank]->Reallocate(BUFFER_SHRINK_FACTOR);
    };
    ForEachRankSync(this->comm_world, this->ranksOrder, shrinkBuffer);
}

template<typename T, typename Grid>
std::vector<typename MonteCarloManager<T, Grid>::MCParticle> MonteCarloManager<T, Grid>::MonteCarloManager::step(const std::vector<MCParticle> &particleList, dt_t fullDt)
{
    // if(this->Ncells != this->grid.GetPointNo())
    // {
    //     std::cout << "Changed grid for rank " << this->rank_world << ": " << this->Ncells << " -> " << this->grid.GetPointNo() <<  std::endl;
    // }
    START_TIMER_PREEMPTIVE("Initialization");

    this->Ncells = this->grid.GetPointNo();
    this->ranks_ghost_map = GetGhostMap(this->grid);
    std::tie(this->ll, this->ur) = this->grid.GetBoxCoordinates();
    
    this->neighbors = GetNeighborList(this->grid, this->ranks_ghost_map);
    this->ResetAllBuffers();
    if(this->currentStep > 0 and this->currentStep % SHRINK_BUFFERS_CYCLE == 0)
    {
        this->ShrinkAllBuffers();
    }
    this->PutSelfParticles(particleList);
    this->resetTracker();
    this->currentStep++;
    this->iteration = 0;
    this->allStepsCounter = 0;
    this->maxConsecutiveSteps = 0;
    this->maxConsecutiveStepsTime = 0;
    this->dynamicallyAdded = 0;
    // this->neighbors = this->grid.GetDuplicatedProcs();    
    this->cellsStepsCounters = std::vector<size_t>(this->Ncells, 0);
    this->transfersCounter = 0;
    MPI_Barrier(this->comm_world);
    
    size_t totalParticles = 0;
    for(RankHandler *handler : this->rankHandlers)
    {
        if(handler == nullptr)
        {
            continue;
        }
        handler->reallocationTime = 0;
        handler->reallocationsThisStep = 0;
        int length = *handler->th_length;
        totalParticles += length;
        for(int i = 0; i < length; i++)
        {
            size_t particleIndex = handler->th[i];
            MCParticle &p = handler->particles[particleIndex];
            #ifdef MONTECARLO_DEBUG
            p.checkedHere = true;
            p.nextRank = std::numeric_limits<rank_t>::max();
            p.removedFromRank = false;
            p.sentByRank = std::numeric_limits<rank_t>::max();
            p.lastSeen = 0;
            #endif // MONTECARLO_DEBUG
            p.timeLeft = fullDt;
            p.initialWeight = p.weight;
            p.steps = 0;
        }
    }

    // this->progress->Reset(totalParticles);
    MPI_Barrier(this->comm_world);

    START_TIMER_PREEMPTIVE("Prestep");

    size_t initialParticlesNum = particleList.size();

    RankHandler *handler = this->rankHandlers[this->rank_world];
    
    this->physics->updateGridData();
    // MPI_Barrier(MPI_COMM_WORLD);
    // measure time for prestep
    std::chrono::high_resolution_clock::time_point preStepStart = std::chrono::high_resolution_clock::now();
    std::vector<MCParticle> newParticles1 = this->physics->preStep(fullDt);
    std::chrono::high_resolution_clock::time_point preStepEnd = std::chrono::high_resolution_clock::now();

    double preStepSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(preStepEnd - preStepStart).count();
    // get maximal
    struct
    {
        double seconds;
        int rank;
    } myPreStepData, maxPreStepData;
    myPreStepData.seconds = preStepSeconds;
    myPreStepData.rank = this->rank_world;
    MPI_Reduce(&myPreStepData, &maxPreStepData, 1, MPI_DOUBLE_INT, MPI_MAXLOC, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &preStepSeconds, &preStepSeconds, 1, MPI_DOUBLE, MPI_SUM, 0, this->comm_world);

    if(this->rank_world == 0)
    {
        // std::cout << "Prestep average time: " << preStepSeconds / this->size_world << " seconds, max is " << maxPreStepData.seconds << " on rank " << maxPreStepData.rank << std::endl;
    }
    // MPI_Barrier(MPI_COMM_WORLD);

    {
        START_TIMER("Adding Particles");
        this->AddParticles(newParticles1);
    }
    // MPI_Barrier(this->comm_world);
    size_t numParticles = *handler->th_length;
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &numParticles, &numParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);

    size_t preStepParticlesNum = newParticles1.size();
    int64_t startingParticleNum = initialParticlesNum + preStepParticlesNum;

    this->localDecrementAmount = 0;
    this->amountManager->Initialize(startingParticleNum);

    MonteCarloStepFinalData data;
    // measure time
    // vtune_start();
    size_t numOfCounterDecrementations = 0;
    auto start = std::chrono::high_resolution_clock::now();
    size_t lastLocalDecrementAmount;
    size_t decrementTryCounter = 0;

    volatile int &verify = *this->amountManager->shouldVerify;
    volatile int &done = *this->amountManager->done;

    START_TIMER_PREEMPTIVE("Main Loop");
    try
    {
        while(not done)
        {
            lastLocalDecrementAmount = this->localDecrementAmount;

            this->reallocationAgent->HandleAllWaitingReallocations();

            bool isEmpty = this->HandleAll(data);
    
            if(isEmpty and (this->localDecrementAmount != 0) and (this->localDecrementAmount == lastLocalDecrementAmount))
            {
                decrementTryCounter++;
            }
            if(decrementTryCounter == 30)
            {
                numOfCounterDecrementations++;
                this->amountManager->Decrease(this->localDecrementAmount);
                this->localDecrementAmount = 0;
                decrementTryCounter = 0;
            }
            
            if(this->rank_world == 0 and this->iteration % 20 == 0)
            {
                this->amountManager->CheckToFinish();
            }
            if(verify)
            {
                // std::cout << "Rank " << this->rank_world << " should verify" << std::endl;
                bool ok = true;
                for(RankHandler *handler : this->rankHandlers)
                {
                    if(handler == nullptr)
                    {
                        continue;
                    }
                    if(*handler->th_length != 0)
                    {
                        ok = false;
                        break;
                    }
                }
                this->amountManager->Verify(ok);
                if(this->rank_world == 0)
                {
                    this->amountManager->ReceiveVerifies();
                }
            }

            // if(this->rank_world == 0)
            // {
            //     double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start).count();
            //     // if(elapsed > 5)
            //     // {
            //     //     std::cout << "Elapsed " << elapsed << " seconds, currently " << this->amountManager->GetCounter() << std::endl;
            //     // }
            // }
            this->iteration++;
        }
    }
    catch(const UniversalError &eo)
    {
        reportError(eo);
        throw;
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    START_TIMER_PREEMPTIVE("Boundary Condition");
    std::vector<MCParticle> populationControlParticles = this->populationControl->activate(data.remaining);
    
    START_TIMER_PREEMPTIVE("Poststep");
    this->physics->postStep(populationControlParticles, fullDt);

    START_TIMER_PREEMPTIVE("Diagnostics");

    double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    // std::cout << "Rank " << this->rank_world << " is outside of step() loop, in " << seconds << " seconds (" << numParticles << " particles)" << std::endl;

    size_t newParticlesNum = populationControlParticles.size();
    size_t leavingNumber = data.leaving.size();

    size_t totalSteps = this->allStepsCounter;
    size_t totalCounterDecrementations = numOfCounterDecrementations;
    size_t callsToTransfer = this->transfersCounter;

    struct
    {
        int x;
        int rank;
    } mySteps, maxSteps, myTransfers, maxTransfers;

    mySteps.x = 0;
    for(size_t counter : this->cellsStepsCounters)
    {
        mySteps.x += static_cast<int>(counter);
    }
    mySteps.rank = this->rank_world;
    
    MPI_Reduce(&mySteps, &maxSteps, 1, MPI_2INT, MPI_MAXLOC, 0, this->comm_world);

    myTransfers.x = static_cast<int>(this->transfersCounter);
    myTransfers.rank = this->rank_world;
    MPI_Reduce(&myTransfers, &maxTransfers, 1, MPI_2INT, MPI_MAXLOC, 0, this->comm_world);

    double reallocationTime = 0, maxReallocationTime = 0;
    for(RankHandler *handler : this->rankHandlers)
    {
        if(handler == nullptr)
        {
            continue;
        }
        reallocationTime += handler->reallocationTime;
    }
    // std::cout << "leavingNumber = " << leavingNumber << " and newParticlesNum = " << newParticlesNum << std::endl; 
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &initialParticlesNum, &initialParticlesNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &preStepParticlesNum, &preStepParticlesNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &leavingNumber, &leavingNumber, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &newParticlesNum, &newParticlesNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &totalSteps, &totalSteps, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &totalCounterDecrementations, &totalCounterDecrementations, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &startingParticleNum, &startingParticleNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &callsToTransfer, &callsToTransfer, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce(&reallocationTime, &maxReallocationTime, 1, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &reallocationTime, &reallocationTime, 1, MPI_DOUBLE, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &this->maxConsecutiveSteps, &this->maxConsecutiveSteps, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &this->maxConsecutiveStepsTime, &this->maxConsecutiveStepsTime, 1, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
    
    if(this->rank_world == 0)
    {
        // double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        // std::cout << "Elapsed: " << elapsed << " seconds, max " << maxReallocationTime << " in reallocation (average: " << reallocationTime / this->size_world << ")" << std::endl;
        // std::cout << "Started with " << startingParticleNum << ". Came with " << initialParticlesNum << ". Generated " << preStepParticlesNum << " particles in preStep. ";
        // std::cout << "Number of leaving particles is " << leavingNumber << " and remaining (after population control) " << newParticlesNum << ". ";
        // std::cout << "Total steps: " << totalSteps << ", total counter decrementations: " << totalCounterDecrementations << std::endl;
        // // std::cout << "Max steps: " << maxSteps.x << " on rank " << maxSteps.rank << ", average is " << totalSteps / this->size_world << std::endl;
        // // std::cout << "Max calls to transfer: " << maxTransfers.x << " on rank " << maxTransfers.rank << ", average is " << callsToTransfer / this->size_world << std::endl;
        // std::cout << "Max consecutive steps: " << this->maxConsecutiveSteps << ", time " << this->maxConsecutiveStepsTime << std::endl;
    }
    MPI_Barrier(this->comm_world);
    // vtune_stop();
    // return data.finalData;

    for(const RankHandler *handler : this->rankHandlers)
    {
        if(handler == nullptr)
        {
            continue;
        }
        if(*handler->th_length != 0)
        {
            UniversalError eo("End of MonteCarloManager::step: th length is not 0");
            eo.addEntry("Rank", this->rank_world);
            eo.addEntry("TH Length", *handler->th_length);
            eo.addEntry("Peer Rank", handler->peer_rank_world);
            throw eo;
        }
    }
    return populationControlParticles;
}

#endif // MONTE_CARLO_MANAGER_HPP