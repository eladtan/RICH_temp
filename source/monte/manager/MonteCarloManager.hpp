#ifndef MONTE_CARLO_MANAGER_HPP
#define MONTE_CARLO_MANAGER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "Radiation/OpacityCalculator.hpp"
#include "mpi/mpi_commands.hpp"
#include "mpi/serialize/mpi_commands.hpp"
#include "monte/MonteCarloParticle.hpp"
#include "monte/physics/MonteCarloPhysics.hpp"
#include "monte/population/PopulationControl.hpp"
#include "utils/amountManager/AmountManager.hpp"
#include "monte/boundary/BoundaryCondition.hpp"
#include "monte/utils/GhostMap.hpp"
#include "monte/utils/RankSync.hpp"
#include "utils/debug/vtune.h" // TODO: remove
#include "RankHandler.hpp"
#include "ReallocationAgent.hpp"
#include "utils/debug/SmartTimer.hpp"
#include "misc/memory_debug.hpp"
#include <memory>
#include <random>
#include <mpi.h>

#define MONTECARLO_EPSILON 1e-8
#define DEFAULT_BUFFER_SIZE 500
#define MONTECARLO_CHANGE_TAG 1280
#define SHRINK_BUFFERS_CYCLE 50
#define SEND_BUFFER_MIN_SIZE 200
#define SEND_BUFFER_MIN_CYCLES 100
#define RW_PROGRESS_TAG 9941
#define MC_PROGRESS_COUNTERS 6

enum MCProgressCounterIndex : size_t
{
    MC_PROGRESS_RW_STEPS = 0,
    MC_PROGRESS_DDMC_STEPS,
    MC_PROGRESS_DDMC_LEAKS,
    MC_PROGRESS_DDMC_CENSUS,
    MC_PROGRESS_DDMC_UPSCATTER,
    MC_PROGRESS_DDMC_FALLBACK,
    MC_PROGRESS_COUNTER_COUNT
};

static_assert(MC_PROGRESS_COUNTER_COUNT == MC_PROGRESS_COUNTERS,
              "Update MC_PROGRESS_COUNTERS when progress fields change");

template<typename T>
double MaxAxisRelativeDrift(const T &drift, const T &boxSize)
{
    double maxRelDrift = 0.0;
    auto update = [&maxRelDrift](double delta, double size)
    {
        if(size > 0.0)
            maxRelDrift = std::max(maxRelDrift, std::abs(delta) / size);
    };

    update(std::abs(drift.x), std::abs(boxSize.x));
    update(std::abs(drift.y), std::abs(boxSize.y));
    update(std::abs(drift.z), std::abs(boxSize.z));
    return maxRelDrift;
}

template<typename T>
void ComputeBoxDriftDiagnostics(const T &location, const T &boxLL, const T &boxUR,
                                double &relativeDrift, double &maxAxisRelativeDrift)
{
    T boxSize = boxUR - boxLL;
    T clamped = location;
    clamped.x = std::max(boxLL.x, std::min(boxUR.x, clamped.x));
    clamped.y = std::max(boxLL.y, std::min(boxUR.y, clamped.y));
    clamped.z = std::max(boxLL.z, std::min(boxUR.z, clamped.z));

    T drift = location - clamped;
    relativeDrift = abs(drift) / abs(boxSize);
    maxAxisRelativeDrift = MaxAxisRelativeDrift(drift, boxSize);
}

template<typename Grid>
std::vector<rank_t> GetNeighborList(const Grid &tess, const boost::container::flat_map<size_t, std::pair<rank_t, size_t>> &ghostsMap)
{
    size_t N = tess.GetPointNo();
    boost::container::flat_set<rank_t> ranks;

    std::vector<size_t> allNeighboringGhosts;
    std::vector<size_t> mc_neigh_buf;
    for(size_t i = 0; i < N; i++)
    {
        tess.GetNeighbors(i, mc_neigh_buf);
        for(size_t ghostIdx : mc_neigh_buf)
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
    using RankHandler_t = ::RankHandler<T, Grid>;

public:
    struct MonteCarloStepFinalData
    {
        std::vector<MCParticle> remaining;
        size_t leavingCount = 0;
    };

    MonteCarloManager(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics,
                    const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
                    const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition,
                    size_t bufferSizes = DEFAULT_BUFFER_SIZE,
                    const MPI_Comm &comm = MPI_COMM_WORLD, RDMA_Type rdma_type = RDMA_Type::AUTO_RDMA);

    ~MonteCarloManager();

    void ClearCommunicator(void);

    void TransferParticles(rank_t rankBuffer, const std::vector<size_t> &indicesInToHandle, const std::vector<rank_t> &transferRanks, size_t num);

    void TransferParticles(const std::vector<rank_t> &rankBuffers, const std::vector<std::vector<size_t>> &indicesInToHandle, const std::vector<std::vector<rank_t>> &transferRanks);

    inline size_t GetStepCounter(void) const{return this->allStepsCounter;};

    inline const std::vector<size_t> &GetCellsStepsCounters(void) const {return this->cellsStepsCounters;}

    inline std::vector<size_t> &GetCellsStepsCounters(void) {return this->cellsStepsCounters;}

    inline size_t GetStartParticleCount(void) const {return this->startParticleCount_;}

    inline size_t GetInitialParticleCount(void) const {return this->initialParticleCount_;}

    inline size_t GetPreStepParticleCount(void) const {return this->preStepParticleCount_;}

    inline size_t GetEndParticleCount(void) const {return this->endParticleCount_;}

    inline size_t GetHandlerMemoryBytes(void) const {return this->handlerMemoryBytes_;}

    std::vector<MCParticle> step(std::vector<MCParticle> &&particleList, dt_t fullDt);
    
    class Tracker
    {
    public:
        Tracker(const MPI_Comm &comm);

        void Reset(void);

        #ifdef RICH_MPI
            std::vector<MCParticle> GetLocalTrackParticleRoute(size_t id) const;
        #endif // RICH_MPI

        std::vector<MCParticle> GetTrackParticleRoute(size_t id) const;

        void ReportParticle(MCParticle &particle);
    
    private:
        MPI_Comm comm;
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
    typename AmountManager::counter_t localDecrementAmount;
    std::vector<MPI_Comm> communicators;
    std::vector<rank_t> ranksOrder;
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ranks_ghost_map;
    std::vector<RankHandler_t*> rankHandlers;
    T ll, ur;
    std::shared_ptr<MonteCarloPhysics<T, Grid>> physics;
    std::shared_ptr<PopulationControl<T, Grid>> populationControl;
    std::shared_ptr<BoundaryCondition<T, Grid>> boundaryCondition;
    Tracker tracker;
    std::shared_ptr<ReallocationAgent> reallocationAgent;
    size_t myIDCounter;
    size_t currentStep;
    size_t allStepsCounter;
    size_t transfersCounter;
    std::chrono::high_resolution_clock::time_point progressStartTime_;
    double lastProgressPrintTime_ = 0.0;
    int64_t progressStartParticles_ = 0;
    size_t progressRemovedCount_ = 0;
public:
    const void* progressCellsPtr_ = nullptr;
    const void* progressOpacityPtr_ = nullptr;
private:
    std::vector<rank_t> neighbors;
    std::vector<size_t> cellsStepsCounters;
    size_t iteration;
    size_t dynamicallyAdded;
    RDMA_Type rdma_type;
    size_t lastBuildGeneration;
    size_t initialParticleCount_ = 0;
    size_t preStepParticleCount_ = 0;
    size_t startParticleCount_ = 0;
    size_t endParticleCount_ = 0;
    size_t handlerMemoryBytes_ = 0;

    boost::container::flat_map<rank_t, std::vector<MCParticle>> sendBuffers;
    size_t sendBufferCycleCounter;
    
    bool HandleAll(MonteCarloStepFinalData &stepData);

    void PutSelfParticles(std::vector<MCParticle> &&particles);

    void PrepareHandlers(void);

    void FreeHandlers(void);

    void AddParticles(const std::vector<MCParticle> &particles);

    void ResetAllBuffers(void); 

    void ShrinkAllBuffers(void);

    void FlushSendBuffers(void);

    void FlushAllSendBuffers(void);

    bool AllSendBuffersEmpty(void) const;

    void PrintMemoryDiagnostics(size_t initialParticlesNum, size_t preStepParticlesNum);
};

template<typename T, typename Grid>
MonteCarloManager<T, Grid>::MonteCarloManager(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics, const std::shared_ptr<PopulationControl<T, Grid>> &populationControl, 
                                            const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition, size_t bufferSizes, const MPI_Comm &comm, RDMA_Type rdma_type):
    grid(grid), physics(physics), populationControl(populationControl), boundaryCondition(boundaryCondition), comm_world(MPI_COMM_NULL), tracker(comm), rdma_type(rdma_type)
{
    this->myIDCounter = 0;
    this->currentStep = 0;
    // this->progress = std::make_shared<ProgressCounter>(comm);
    this->comm_world = comm;
    MPI_Comm_rank(this->comm_world, &this->rank_world);
    MPI_Comm_size(this->comm_world, &this->size_world);

    this->ranksOrder = GetRanksOrder(this->comm_world);
    this->communicators = std::vector<MPI_Comm>(this->size_world, MPI_COMM_NULL);

    this->rankHandlers = std::vector<RankHandler_t*>(this->size_world, nullptr);

    auto reallocationFunction = [this](rank_t rank)
    {
        this->rankHandlers[rank]->Reallocate(BUFFER_REALLOCATION_FACTOR);
    };

    this->reallocationAgent = std::make_shared<ReallocationAgent>(this->comm_world, reallocationFunction);

    if(this->rank_world == 0)
    {
        std::cout << "Done initializing MonteCarloManager" << std::endl;
    }
    this->cellsStepsCounters.assign(this->grid.GetPointNo(), 0);
    this->lastBuildGeneration = std::numeric_limits<size_t>::max();
    this->sendBufferCycleCounter = 0;
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
void MonteCarloManager<T, Grid>::FreeHandlers(void)
{
    auto freeHandler = [&](rank_t _rank)
    {
        RankHandler_t *handler = this->rankHandlers[_rank];
        if(handler != nullptr)
        {
            handler->Destroy();
            delete handler;    
        }
        this->rankHandlers[_rank] = nullptr;
    };
    
    ForEachRankSync(this->comm_world, this->ranksOrder, freeHandler);
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::AddParticles(const std::vector<MCParticle> &particles)
{
    using index_t = typename RankHandler_t::index_t;
    if(particles.empty())
    {
        return;
    }

    RankHandler_t *myHandler = this->rankHandlers[this->rank_world];

    // std::cout << "In add particles, handler size is " << myHandler->buffsize << ", particles size to add is " << particles.size() << std::endl;

    if(myHandler->av_length < particles.size())
    {
        double factor = std::max<double>(BUFFER_REALLOCATION_FACTOR, std::ceil(static_cast<double>(particles.size() + myHandler->buffsize) / static_cast<double>(myHandler->buffsize)));
        myHandler->Reallocate(factor);
        assert(myHandler->av_length >= particles.size());
    }

    // set particles
    // update 'to handle' and 'available' lists accordingly
    index_t particlesNum = particles.size();
    myHandler->av_length -= particlesNum;
    index_t *avIndices = myHandler->av + (myHandler->av_length);
    index_t *thIndices = myHandler->th + (myHandler->th_length);
    myHandler->th_length += particlesNum;
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

    this->localDecrementAmount -= static_cast<typename AmountManager::counter_t>(particlesNum);
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
std::vector<typename MonteCarloManager<T, Grid>::MCParticle> MonteCarloManager<T, Grid>::Tracker::GetLocalTrackParticleRoute(size_t id) const
{
    auto it = this->track.find(id);
    if(it == this->track.end())
    {
        return std::vector<MCParticle>();
    }
    return it->second;
}

template<typename T, typename Grid>
std::vector<typename MonteCarloManager<T, Grid>::MCParticle> MonteCarloManager<T, Grid>::Tracker::GetTrackParticleRoute(size_t id) const
{
    std::vector<MCParticle> local = this->GetLocalTrackParticleRoute(id);
    std::vector<MCParticle> global = MPI_All_cast(local, this->comm);
    // sort by `particle.steps`
    std::sort(global.begin(), global.end(), [](const MCParticle &a, const MCParticle &b) { return a.steps < b.steps; });
    return global;
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::MonteCarloManager::PutSelfParticles(std::vector<MCParticle> &&particles)
{
    using index_t = typename RankHandler_t::index_t;

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
    if(particlesNum == 0)
    {
        return;
    }

    RankHandler_t *handler = this->rankHandlers[this->rank_world];

    if(static_cast<size_t>(handler->av_length) < particlesNum)
    {
        double factor = std::max<double>(BUFFER_REALLOCATION_FACTOR, std::ceil(static_cast<double>(particlesNum) / static_cast<double>(handler->buffsize)));
        handler->Reallocate(factor);
    }

    handler->av_length -= static_cast<int>(particlesNum);
    index_t *av_indices = handler->av + handler->av_length;
    int oldTHLength = handler->th_length;
    handler->th_length += particlesNum;

    for(size_t i = 0; i < particlesNum; i++)
    {
        size_t particleIdx = av_indices[i];
        handler->th[oldTHLength + i] = particleIdx;
        std::memcpy(handler->particles + particleIdx, &particles[i], sizeof(MCParticle));
        MCParticle &particle = handler->particles[particleIdx];
        if(particle.id == std::numeric_limits<size_t>::max())
        {
            // no ID has been assigned
            particle.rank = this->rank_world;
            particle.id = this->myIDCounter++;
        }
    }

    // don't waste memory - remove current particles from the input vector
    std::vector<MCParticle> empty;
    particles.swap(empty);
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
    RankHandler_t *currRankHandler = this->rankHandlers[fromRank];

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
        RankHandler_t *remoteHandler = this->rankHandlers[toRank];
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
        remoteHandler->TransferParticles(particles);
        this->reallocationAgent->HandleAllWaitingReallocations();
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
        RankHandler_t *currRankHandler = this->rankHandlers[fromRank];
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
        RankHandler_t *remoteHandler = this->rankHandlers[toRank];
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
    static size_t progressStepCounter;

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

                // Prefetch the RankHandler *object (heap-allocated, likely scattered)
                RankHandler_t *future_handler = this->rankHandlers[future_rank];
                __builtin_prefetch(future_handler, 0, 1);                  // bring RankHandler into cache
                __builtin_prefetch((const void*) &(future_handler->th_length), 0, 1);      // bring th_length into cache
            }

            // Access current handler
            rank_t _rank = this->neighbors[i];
            RankHandler_t *handler = this->rankHandlers[_rank];

            // Cache the dereferenced value to avoid repeated indirection
            int len = handler->th_length;

            // Only proceed if there's work to do
            if(len)
            {
                active_ranks.push_back(_rank);
            }
        }
        {
            RankHandler_t *handler = this->rankHandlers[this->rank_world];
            if(handler->th_length > 0)
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
        ++this->progressRemovedCount_;
    };

    transferParticlesVec.clear();
    transferToRanks.clear();
    removeParticlesVec.clear();

    for(size_t index = 0; index < activeRanksNum; index++)
    {
        rank_t _rank = active_ranks[index];
        RankHandler_t *handler = this->rankHandlers[_rank];
        volatile int &length = handler->th_length;

        transferParticlesVec.emplace_back();
        transferToRanks.emplace_back();
        removeParticlesVec.emplace_back();

        #ifdef ADVANCED_MONTECARLO_DEBUG
            handler->LockSelfBuffer();
        #endif // ADVANCED_MONTECARLO_DEBUG
                
        for(int i = 0; i < length; i++)
        {
            assert(i < handler->buffsize);
            size_t particleIndex = handler->th[i];
            assert(particleIndex < handler->buffsize);
            MCParticle &particle = handler->particles[particleIndex];
            bool debug = false; // (particle.rank == 5 and particle.id == 518987);

            try
            {
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
                    ++progressStepCounter;
                    if ((progressStepCounter & 0x3FFFF) == 0 && particle.steps > 100000) {
                        std::cerr << "[StuckParticle] rank=" << this->rank_world
                                  << " localPts=" << this->grid.GetPointNo()
                                  << " " << particle
                                  << " freq=" << particle.frequency
                                  << " w/w0=" << (particle.initialWeight > 0 ? particle.weight / particle.initialWeight : 0.0) << std::endl;
                        if (this->progressCellsPtr_ && particle.cellIndex < this->Ncells) {
                            auto const& cc = (*static_cast<const std::vector<ComputationalCell3D>*>(this->progressCellsPtr_))[particle.cellIndex];
                            std::cerr << " cell=" << cc << std::endl;
                            if (this->progressOpacityPtr_) {
                                auto const* opa = static_cast<const OpacityCalculator*>(this->progressOpacityPtr_);
                                double sigP = opa->CalcPlanckOpacity(cc);
                                double sigS = opa->CalcScatteringOpacity(cc);
                                double charLen = std::cbrt(this->grid.GetVolume(particle.cellIndex));
                                std::cerr << " sig_planck=" << sigP
                                          << " sig_scat=" << sigS
                                          << " tau_planck=" << sigP * charLen
                                          << " tau_scat=" << sigS * charLen << std::endl;
                            }
                        }
                        std::string accelInfo = this->physics->getAccelerationDebugInfo(particle.cellIndex, particle.frequency);
                        if(!accelInfo.empty())
                            std::cerr << accelInfo << std::endl;
                        std::cerr << std::endl;
                    }

                    const size_t traceStep = particle.steps;
                    if(particle.on_track)
                    {
                        MCParticle trackedParticle = particle;
                        trackedParticle.steps = traceStep * 2;
                        this->tracker.ReportParticle(trackedParticle);
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
                    
                    prevLoc = particle.location;
                    particle.previousLocation = particle.location;
                    #endif // MONTECARLO_DEBUG

                    if(debug)
                    {
                        std::cout << "Before running particle step, particle is " << particle << std::endl;
                    }

                    const T beforeStepLocation = particle.location;
                    const T beforeStepVelocity = particle.velocity;
                    const dt_t beforeStepTimeLeft = particle.timeLeft;
                    if(BOOST_UNLIKELY(this->grid.IsPointOutsideBox(particle.location)))
                    {
                        auto const [boxLL, boxUR] = this->grid.GetBoxCoordinates();
                        double relDrift = 0.0;
                        double maxAxisRelDrift = 0.0;
                        ComputeBoxDriftDiagnostics(particle.location, boxLL, boxUR, relDrift, maxAxisRelDrift);

                        UniversalError eo("MonteCarloManager: particle outside box before physics step");
                        eo.addEntry("Rank", this->rank_world);
                        eo.addEntry("Particle before step", particle);
                        eo.addEntry("Location before step", beforeStepLocation);
                        eo.addEntry("Velocity before step", beforeStepVelocity);
                        eo.addEntry("Time left before step", beforeStepTimeLeft);
                        eo.addEntry("Box lower", boxLL);
                        eo.addEntry("Box upper", boxUR);
                        eo.addEntry("Relative drift", relDrift);
                        eo.addEntry("Max axis relative drift", maxAxisRelDrift);
                        eo.addEntry("Cell count", this->Ncells);
                        if(particle.cellIndex < this->Ncells)
                        {
                            eo.addEntry("Cell index", particle.cellIndex);
                            eo.addEntry("Cell center", this->grid.GetMeshPoint(particle.cellIndex));
                            eo.addEntry("Inside declared cell before step", this->grid.IsPointInCell(particle.location, particle.cellIndex));
                        }
                        throw eo;
                    }
                    MonteCarloFunctionality<T, Grid> functionality = this->physics->step(particle, particlesToAdd);
                    if(BOOST_UNLIKELY(functionality.change != MonteCarloParticleStatus::REMOVE &&
                                      functionality.change != MonteCarloParticleStatus::CELL_MOVE &&
                                      this->grid.IsPointOutsideBox(particle.location)))
                    {
                        auto const [boxLL, boxUR] = this->grid.GetBoxCoordinates();
                        double relDrift = 0.0;
                        double maxAxisRelDrift = 0.0;
                        ComputeBoxDriftDiagnostics(particle.location, boxLL, boxUR, relDrift, maxAxisRelDrift);

                        UniversalError eo("MonteCarloManager: physics step moved particle outside the box");
                        eo.addEntry("Rank", this->rank_world);
                        eo.addEntry("Particle after step", particle);
                        eo.addEntry("Functionality", MonteCarloParticleStatusToString(functionality.change));
                        eo.addEntry("Next cell index", functionality.nextCellIndex);
                        eo.addEntry("Location before step", beforeStepLocation);
                        eo.addEntry("Velocity before step", beforeStepVelocity);
                        eo.addEntry("Time left before step", beforeStepTimeLeft);
                        eo.addEntry("Box lower", boxLL);
                        eo.addEntry("Box upper", boxUR);
                        eo.addEntry("Relative drift", relDrift);
                        eo.addEntry("Max axis relative drift", maxAxisRelDrift);
                        eo.addEntry("Cell count", this->Ncells);
                        if(particle.cellIndex < this->Ncells)
                        {
                            eo.addEntry("Cell index", particle.cellIndex);
                            eo.addEntry("Cell center", this->grid.GetMeshPoint(particle.cellIndex));
                            eo.addEntry("Inside declared cell after step", this->grid.IsPointInCell(particle.location, particle.cellIndex));
                        }
                        throw eo;
                    }

                    if(particle.on_track)
                    {
                        MCParticle trackedParticle = particle;
                        trackedParticle.steps = traceStep * 2 + 1;
                        this->tracker.ReportParticle(trackedParticle);
                    }

                    #ifdef MC_TRACING_HISTORY
                        particle.recordHistory(particle.cellIndex, static_cast<int>(this->rank_world), static_cast<int>(functionality.change));
                    #endif // MC_TRACING_HISTORY
    
                    // std::cout << "Handling particle " << particle << ", functionality is " << functionality.change << std::endl;
                    if(debug)
                    {
                        std::cout << "Particle " << particle << ", functionality is " << functionality.change << std::endl;
                    }
    
                    if(functionality.change == MonteCarloParticleStatus::CELL_MOVE)
                    {
                        size_t nextCellIndex = functionality.nextCellIndex;
    
                        assert(nextCellIndex != particle.cellIndex);
                        assert(particle.timeLeft >= 0);

                        auto throwCellMoveOutsideBox = [&](const std::string &cellMoveTarget)
                        {
                            auto const [boxLL, boxUR] = this->grid.GetBoxCoordinates();
                            double relDrift = 0.0;
                            double maxAxisRelDrift = 0.0;
                            ComputeBoxDriftDiagnostics(particle.location, boxLL, boxUR, relDrift, maxAxisRelDrift);

                            UniversalError eo("MonteCarloManager: CELL_MOVE moved particle outside box before a non-boundary cell move");
                            eo.addEntry("Rank", this->rank_world);
                            eo.addEntry("Particle after step", particle);
                            eo.addEntry("Cell move target", cellMoveTarget);
                            eo.addEntry("Next cell index", nextCellIndex);
                            eo.addEntry("Location before step", beforeStepLocation);
                            eo.addEntry("Velocity before step", beforeStepVelocity);
                            eo.addEntry("Time left before step", beforeStepTimeLeft);
                            eo.addEntry("Box lower", boxLL);
                            eo.addEntry("Box upper", boxUR);
                            eo.addEntry("Relative drift", relDrift);
                            eo.addEntry("Max axis relative drift", maxAxisRelDrift);
                            eo.addEntry("Cell count", this->Ncells);
                            if(particle.cellIndex < this->Ncells)
                            {
                                eo.addEntry("Cell index", particle.cellIndex);
                                eo.addEntry("Cell center", this->grid.GetMeshPoint(particle.cellIndex));
                                eo.addEntry("Inside declared cell after step", this->grid.IsPointInCell(particle.location, particle.cellIndex));
                            }
                            throw eo;
                        };
        
                        if(BOOST_LIKELY(nextCellIndex < this->Ncells))
                        {
                            if(BOOST_UNLIKELY(this->grid.IsPointOutsideBox(particle.location)))
                                throwCellMoveOutsideBox("local cell move");

                            // local neighbor
                            #ifdef MONTECARLO_DEBUG
                            size_t previousCell = particle.cellIndex;
                            #endif // MONTECARLO_DEBUG
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
                                #ifdef MC_TRACING_HISTORY
                                    T preReflectLoc = particle.location;
                                    T preReflectVel = particle.velocity;
                                #endif // MC_TRACING_HISTORY
                                MonteCarloParticleStatus status = this->boundaryCondition->apply(particle);
                                if(debug)
                                {
                                    std::cout << "Particle " << particle << ", leaving domain. status from bounday condition: " << status << std::endl;
                                }
                                if(status == MonteCarloParticleStatus::REFLECT)
                                {
                                    #ifdef MC_TRACING_HISTORY
                                        particle.markLastHistoryReflected(preReflectLoc, preReflectVel);
                                    #endif // MC_TRACING_HISTORY
                                }
                                else if(status == MonteCarloParticleStatus::REMOVE)
                                {
                                    stepData.leavingCount++;
                                    this->allStepsCounter += particle.steps;
                                    // remove particle from current list
                                    removeParticle(index, i);
                                }
                                else
                                {
                                    UniversalError eo("Unknown boundary condition for particle");
                                    eo.addEntry("Particle", particle);
                                    eo.addEntry("Status", status);
                                    throw eo;
                                }
                                break;    
                            }

                            if(BOOST_UNLIKELY(this->grid.IsPointOutsideBox(particle.location)))
                                throwCellMoveOutsideBox("remote rank transfer");
    
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
                            break; 
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
            catch(UniversalError &eo)
            {
                eo.addEntry("Particle TH index", i);
                eo.addEntry("Particle index", particleIndex);
                throw eo;
            }
            this->reallocationAgent->HandleAllWaitingReallocations();
        }
        
        if(length > 0)
        {
            next_active_ranks.push_back(_rank);
        }
        
        this->reallocationAgent->HandleAllWaitingReallocations();

        #ifdef ADVANCED_MONTECARLO_DEBUG
            handler->UnlockSelfBuffer();
        #endif // ADVANCED_MONTECARLO_DEBUG
    }

    for(size_t i = 0; i < activeRanksNum; i++)
    {
        const rank_t &fromRank = active_ranks[i];
        RankHandler_t *currRankHandler = this->rankHandlers[fromRank];
        const std::vector<size_t> &myTHIndices = transferParticlesVec[i];
        const std::vector<rank_t> &myTransferRanks = transferToRanks[i];
        size_t numToTransfer = myTHIndices.size();

        for(size_t j = 0; j < numToTransfer; j++)
        {
            const size_t &indexInToHandle = myTHIndices[j];
            const rank_t &toRank = myTransferRanks[j];
            assert(toRank != this->rank_world);
            size_t particleIdx = currRankHandler->th[indexInToHandle];
            MCParticle &particle = currRankHandler->particles[particleIdx];
            particle.sent = false;
            this->sendBuffers[toRank].push_back(particle);
        }
    }

    for(size_t i = 0; i < activeRanksNum; i++)
    {
        rank_t _rank = active_ranks[i];
        const std::vector<size_t> &rankRemoveParticlesVec = removeParticlesVec[i];
        if(rankRemoveParticlesVec.empty())
        {
            continue; // nothing to remove
        }
        RankHandler_t *handler = this->rankHandlers[_rank];
        handler->RemoveParticles(rankRemoveParticlesVec, rankRemoveParticlesVec.size());
    }
    active_ranks.swap(next_active_ranks);

    bool toReturn = isEmpty and particlesToAdd.empty();
    if(not particlesToAdd.empty())
    {
        this->dynamicallyAdded += particlesToAdd.size();
        this->AddParticles(particlesToAdd);
        particlesToAdd.clear();
    }

    return toReturn;
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::MonteCarloManager::ResetAllBuffers(void)
{
    for(RankHandler_t *handler : this->rankHandlers)
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
    std::vector<rank_t> shrinkList;
    for(rank_t r = 0; r < static_cast<rank_t>(this->rankHandlers.size()); r++)
    {
        if(r != this->rank_world and this->rankHandlers[r] != nullptr and this->rankHandlers[r]->buffsize > MINIMAL_BUFF_SIZE)
        {
            shrinkList.push_back(r);
        }
    }

    auto shrinkBuffer = [this](rank_t _rank)
    {
        double factor;
        if(std::find(this->neighbors.cbegin(), this->neighbors.cend(), _rank) != this->neighbors.cend())
        {
            factor = BUFFER_SHRINK_NEIGHBOR_FACTOR;
        }    
        else
        {
            factor = BUFFER_SHRINK_FACTOR;
        }
        this->rankHandlers[_rank]->requestedFactor = factor;
        this->rankHandlers[_rank]->Reallocate(factor);
    };
    ForEachRankSyncByList(this->comm_world, shrinkList, shrinkBuffer);
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::FlushSendBuffers(void)
{
    this->sendBufferCycleCounter++;
    bool cycleFull = this->sendBufferCycleCounter >= SEND_BUFFER_MIN_CYCLES;

    for(auto &[toRank, particles] : this->sendBuffers)
    {
        if(particles.empty())
            continue;
        if(particles.size() >= SEND_BUFFER_MIN_SIZE or cycleFull)
        {
            RankHandler_t *remoteHandler = this->rankHandlers[toRank];
            remoteHandler->TransferParticles(particles);
            particles.clear();
        }
    }

    if(cycleFull)
        this->sendBufferCycleCounter = 0;
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::FlushAllSendBuffers(void)
{
    for(auto &[toRank, particles] : this->sendBuffers)
    {
        if(particles.empty())
            continue;
        RankHandler_t *remoteHandler = this->rankHandlers[toRank];
        remoteHandler->TransferParticles(particles);
        particles.clear();
    }
    this->sendBufferCycleCounter = 0;
}

template<typename T, typename Grid>
bool MonteCarloManager<T, Grid>::AllSendBuffersEmpty(void) const
{
    for(const auto &[rank, particles] : this->sendBuffers)
    {
        if(!particles.empty())
            return false;
    }
    return true;
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::PrintMemoryDiagnostics(size_t initialParticlesNum, size_t preStepParticlesNum)
{
    using index_t = typename RankHandler_t::index_t;
    const size_t bytesPerSlot = sizeof(MCParticle) + 2 * sizeof(index_t);
    size_t localHandlerMemory = 0;
    for(const RankHandler_t *h : this->rankHandlers)
    {
        if(h == nullptr) continue;
        localHandlerMemory += h->buffsize * bytesPerSlot;
    }

    struct { double val; int rank; } myMem, maxMem;
    myMem.val = static_cast<double>(localHandlerMemory);
    myMem.rank = this->rank_world;
    MPI_Allreduce(&myMem, &maxMem, 1, MPI_DOUBLE_INT, MPI_MAXLOC, this->comm_world);

    double avgMem = static_cast<double>(localHandlerMemory);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &avgMem, &avgMem, 1, MPI_DOUBLE, MPI_SUM, 0, this->comm_world);

    size_t preLoopInitial = initialParticlesNum;
    size_t preLoopPreStep = preStepParticlesNum;
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &preLoopInitial, &preLoopInitial, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &preLoopPreStep, &preLoopPreStep, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);

    std::string handlerBreakdown;
    if(this->rank_world == maxMem.rank)
    {
        boost::container::flat_set<rank_t> neighborSet(this->neighbors.begin(), this->neighbors.end());
        std::ostringstream ss;
        double selfTotal = 0, neighborTotal = 0, nonNeighborTotal = 0;
        for(rank_t r = 0; r < static_cast<rank_t>(this->rankHandlers.size()); r++)
        {
            const RankHandler_t *h = this->rankHandlers[r];
            if(h == nullptr) continue;
            double handlerMB = h->buffsize * bytesPerSlot / (1024.0 * 1024.0);
            bool isSelf = (h->peer_rank_world == this->rank_world);
            bool isNeighbor = !isSelf && neighborSet.count(r) > 0;
            (isSelf ? selfTotal : isNeighbor ? neighborTotal : nonNeighborTotal) += handlerMB;
            // const char *tag = isSelf ? " [self]" : isNeighbor ? " [neighbor]" : " [non-neighbor]";
            // ss << "  [" << h->peer_rank_world << "]: "
            //    << handlerMB << " MB (buffsize=" << h->buffsize << ")"
            //    << tag << "\n";
        }
        ss << "  Totals: self=" << selfTotal << " MB, neighbors=" << neighborTotal << " MB, non-neighbors=" << nonNeighborTotal << " MB\n";
        handlerBreakdown = ss.str();
    }

    if(maxMem.rank != 0)
    {
        if(this->rank_world == maxMem.rank)
        {
            int strLen = static_cast<int>(handlerBreakdown.size());
            MPI_Send(&strLen, 1, MPI_INT, 0, 999, this->comm_world);
            MPI_Send(handlerBreakdown.data(), strLen, MPI_CHAR, 0, 1000, this->comm_world);
        }
        else if(this->rank_world == 0)
        {
            int strLen = 0;
            MPI_Recv(&strLen, 1, MPI_INT, maxMem.rank, 999, this->comm_world, MPI_STATUS_IGNORE);
            handlerBreakdown.resize(strLen);
            MPI_Recv(&handlerBreakdown[0], strLen, MPI_CHAR, maxMem.rank, 1000, this->comm_world, MPI_STATUS_IGNORE);
        }
    }

    if(this->rank_world == 0)
    {
        avgMem /= this->size_world;
        std::cout << "RankHandler memory: max=" << maxMem.val / (1024.0 * 1024.0) << " MB (rank " << maxMem.rank << "), avg=" << avgMem / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << handlerBreakdown;
        std::cout << "Starting with " << (preLoopInitial + preLoopPreStep) << ". Came with " << preLoopInitial << ". Generated " << preLoopPreStep << " particles in preStep." << std::endl;
    }
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::PrepareHandlers(void)
{
    boost::container::flat_set<rank_t> oldNeighbors(this->neighbors.cbegin(), this->neighbors.cend());
    this->neighbors = GetNeighborList(this->grid, this->ranks_ghost_map);

    // Self handler: 1-process communicator, no coordination needed
    if(this->rankHandlers[this->rank_world] == nullptr)
    {
        MPI_Group worldGroup;
        MPI_Comm_group(this->comm_world, &worldGroup);
        MPI_Group group;
        int ranks[1] = {this->rank_world};
        int tag = this->rank_world * this->size_world + this->rank_world;
        MPI_Group_incl(worldGroup, 1, ranks, &group);
        MPI_Comm_create_group(this->comm_world, group, tag, &this->communicators[this->rank_world]);
        MPI_Group_free(&group);
        MPI_Group_free(&worldGroup);
        this->rankHandlers[this->rank_world] = new RankHandler_t(DEFAULT_BUFFER_SIZE, this->comm_world, this->communicators[this->rank_world], this->reallocationAgent, this->rdma_type);
    }

    std::vector<rank_t> newNeighbors;
    for(rank_t rank : this->neighbors)
    {
        if(oldNeighbors.find(rank) == oldNeighbors.end() and this->rankHandlers[rank] == nullptr)
        {
            newNeighbors.push_back(rank);
        }
    }

    int numNewNeighbors = newNeighbors.size();
    MPI_Allreduce(MPI_IN_PLACE, &numNewNeighbors, 1, MPI_INT, MPI_SUM, this->comm_world);
    if(numNewNeighbors > 0)
    {
        auto createHandler = [this](rank_t rank, MPI_Comm pair_comm)
        {
            if(this->rankHandlers[rank] != nullptr)
            {
                return;
            }

            this->communicators[rank] = pair_comm;
            this->rankHandlers[rank] = new RankHandler_t(DEFAULT_BUFFER_SIZE, this->comm_world, pair_comm, this->reallocationAgent, this->rdma_type);
            if(this->rankHandlers[rank]->peer_rank_world != rank)
            {
                UniversalError eo("Peer rank world does not match");
                eo.addEntry("Rank", rank);
                eo.addEntry("Peer Rank World", this->rankHandlers[rank]->peer_rank_world);
                throw eo;
            }    
        };
        ForEachRankSyncByList(this->comm_world, newNeighbors, createHandler);
    }

    if(this->rank_world == 0)
    {
        std::cout << "Number of new neighbors: " << numNewNeighbors << std::endl;
    }
    this->ResetAllBuffers();
}

template<typename T, typename Grid>
std::vector<typename MonteCarloManager<T, Grid>::MCParticle> MonteCarloManager<T, Grid>::MonteCarloManager::step(std::vector<MCParticle> &&particleList, dt_t fullDt)
{
    // if(this->Ncells != this->grid.GetPointNo())
    // {
    //     std::cout << "Changed grid for rank " << this->rank_world << ": " << this->Ncells << " -> " << this->grid.GetPointNo() <<  std::endl;
    // }
    auto managerStepStart = std::chrono::high_resolution_clock::now();

    START_TIMER_PREEMPTIVE("Initialization");

    auto initStart = std::chrono::high_resolution_clock::now();
    this->Ncells = this->grid.GetPointNo();
    this->ranks_ghost_map = GetGhostMap(this->grid);
    std::tie(this->ll, this->ur) = this->grid.GetBoxCoordinates();
    
    this->PrepareHandlers();
    this->ResetAllBuffers();
    this->sendBuffers.clear();
    this->sendBufferCycleCounter = 0;

    bool didRebalance = this->grid.DidRebalance() and (this->lastBuildGeneration != this->grid.GetBuildGeneration());
    if(didRebalance)
    {
        if(this->rank_world == 0)
        {
            std::cout << "Doing shrink because of rebalance" << std::endl;
        }
        this->ShrinkAllBuffers();
    }
    this->lastBuildGeneration = this->grid.GetBuildGeneration();

    size_t initialParticlesNum = particleList.size();
    this->PutSelfParticles(std::move(particleList));

    START_TIMER_PREEMPTIVE("Prestep");

    this->physics->updateGridData();
    std::chrono::high_resolution_clock::time_point preStepStart = std::chrono::high_resolution_clock::now();
    std::vector<MCParticle> newParticles1 = this->physics->preStep(fullDt);
    std::chrono::high_resolution_clock::time_point preStepEnd = std::chrono::high_resolution_clock::now();

    double preStepSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(preStepEnd - preStepStart).count();
    auto [maxPreStepRank, maxPreStepTime] = MPI_Max_loc(preStepSeconds, this->comm_world);
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &preStepSeconds, &preStepSeconds, 1, MPI_DOUBLE, MPI_SUM, 0, this->comm_world);

    if(this->rank_world == 0)
    {
        std::cout << "Prestep time: avg=" << preStepSeconds / this->size_world << "s, max=" << maxPreStepTime << "s (rank " << maxPreStepRank << ")" << std::endl;
    }

    size_t preStepParticlesNum = newParticles1.size();
    this->initialParticleCount_ = initialParticlesNum;
    this->preStepParticleCount_ = preStepParticlesNum;
    this->startParticleCount_ = initialParticlesNum + preStepParticlesNum;
    unsigned long long globalInitialParticles = static_cast<unsigned long long>(this->initialParticleCount_);
    unsigned long long globalPreStepParticles = static_cast<unsigned long long>(this->preStepParticleCount_);
    unsigned long long globalStartParticles = static_cast<unsigned long long>(this->startParticleCount_);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalInitialParticles, &globalInitialParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalPreStepParticles, &globalPreStepParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &globalStartParticles, &globalStartParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
    if(this->rank_world == 0)
    {
        std::cout << "MC particle counts before transport:"
                  << " initial=" << globalInitialParticles
                  << " prestep_generated=" << globalPreStepParticles
                  << " active_after_prestep=" << globalStartParticles
                  << std::endl;
    }

    this->resetTracker();
    this->currentStep++;
    this->iteration = 0;
    this->allStepsCounter = 0;
    this->dynamicallyAdded = 0;
    // this->neighbors = this->grid.GetDuplicatedProcs();    
    this->cellsStepsCounters.assign(this->Ncells, 0);
    this->transfersCounter = 0;

    for(RankHandler_t *handler : this->rankHandlers)
    {
        if(handler == nullptr)
        {
            continue;
        }
        handler->reallocationTime = 0;
        handler->reallocationsThisStep = 0;
        int length = handler->th_length;
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
            #ifdef MC_TRACING_HISTORY
            p.tracingHistoryIndex = 0;
            p.tracingHistoryCount = 0;
            #endif // MC_TRACING_HISTORY
            p.timeLeft = fullDt;
            p.initialWeight = std::abs(p.weight);
            p.steps = 0;
        }
    }

    auto addParticlesStart = std::chrono::high_resolution_clock::now();
    {
        START_TIMER("Adding Particles");
        this->AddParticles(newParticles1);
        std::vector<MCParticle>().swap(newParticles1);
    }

    MPI_Barrier(this->comm_world);
    double initTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - initStart).count();

    size_t numParticles = initialParticlesNum + preStepParticlesNum;
    MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &numParticles, &numParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);

    int64_t startingParticleNum = initialParticlesNum + preStepParticlesNum;

    this->localDecrementAmount = 0;

    AmountManager amountManager(this->comm_world);
    amountManager.Initialize(startingParticleNum);

    MonteCarloStepFinalData data;
    size_t numOfCounterDecrementations = 0;
    double addParticlesTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - addParticlesStart).count();

    {
        const size_t bytesPerSlot = sizeof(MCParticle) + 2 * sizeof(typename RankHandler_t::index_t);
        this->handlerMemoryBytes_ = 0;
        for (const RankHandler_t *h : this->rankHandlers)
            if (h != nullptr) this->handlerMemoryBytes_ += h->buffsize * bytesPerSlot;
    }

    this->PrintMemoryDiagnostics(initialParticlesNum, preStepParticlesNum);

    auto start = std::chrono::high_resolution_clock::now();
    this->progressStartTime_ = start;
    this->lastProgressPrintTime_ = 0.0;
    this->progressStartParticles_ = (this->rank_world == 0) ? static_cast<int64_t>(numParticles) : startingParticleNum;
    this->progressRemovedCount_ = 0;
    int64_t globalInitialForProgress = (this->rank_world == 0) ? static_cast<int64_t>(numParticles) : startingParticleNum;
#ifdef RICH_MPI
    std::vector<std::array<unsigned long long, MC_PROGRESS_COUNTERS>> progressCountersByRank(this->size_world);
    MPI_Request progressReportSendReq = MPI_REQUEST_NULL;
    std::array<unsigned long long, MC_PROGRESS_COUNTERS> progressReportSendValue{};
    double progressLastReportSendTime = 0.0;
#endif

    auto buildProgressCounters = [this]()
    {
        std::array<unsigned long long, MC_PROGRESS_COUNTERS> counters{};
        counters[MC_PROGRESS_RW_STEPS] =
            static_cast<unsigned long long>(this->physics->getRandomWalkStepCount());
        counters[MC_PROGRESS_DDMC_STEPS] =
            static_cast<unsigned long long>(this->physics->getDDMCStepCount());
        counters[MC_PROGRESS_DDMC_LEAKS] =
            static_cast<unsigned long long>(this->physics->getDDMCLeakCount());
        counters[MC_PROGRESS_DDMC_CENSUS] =
            static_cast<unsigned long long>(this->physics->getDDMCCensusCount());
        counters[MC_PROGRESS_DDMC_UPSCATTER] =
            static_cast<unsigned long long>(this->physics->getDDMCUpscatterCount());
        counters[MC_PROGRESS_DDMC_FALLBACK] =
            static_cast<unsigned long long>(this->physics->getDDMCFallbackCount());
        return counters;
    };

    const bool &verify = amountManager.GetVerifyRef();
    const bool &done = amountManager.GetDoneRef();

    MEMORY_DEBUG_PRINT("Before main loop in MCM");
    START_TIMER_PREEMPTIVE("Main Loop");
    try
    {
        while(not done)
        {
            this->reallocationAgent->HandleAllWaitingReallocations();
            this->HandleAll(data);

            this->FlushSendBuffers();

            amountManager.Decrease(this->localDecrementAmount);
            this->localDecrementAmount = 0;

            amountManager.Progress();

            auto now = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - this->progressStartTime_).count();

#ifdef RICH_MPI
            if(this->rank_world == 0)
            {
                progressCountersByRank[0] = buildProgressCounters();

                int hasMsg = 0;
                MPI_Status status;
                while(true)
                {
                    MPI_Iprobe(MPI_ANY_SOURCE, RW_PROGRESS_TAG, this->comm_world, &hasMsg, &status);
                    if(!hasMsg)
                        break;
                    std::array<unsigned long long, MC_PROGRESS_COUNTERS> recvCounters{};
                    MPI_Recv(recvCounters.data(), MC_PROGRESS_COUNTERS, MPI_UNSIGNED_LONG_LONG, status.MPI_SOURCE,
                             RW_PROGRESS_TAG, this->comm_world, MPI_STATUS_IGNORE);
                    progressCountersByRank[status.MPI_SOURCE] = recvCounters;
                }
            }
            else if(elapsed_s - progressLastReportSendTime >= 5.0)
            {
                if(progressReportSendReq != MPI_REQUEST_NULL)
                {
                    int sendDone = 0;
                    MPI_Test(&progressReportSendReq, &sendDone, MPI_STATUS_IGNORE);
                    if(sendDone)
                        progressReportSendReq = MPI_REQUEST_NULL;
                }
                if(progressReportSendReq == MPI_REQUEST_NULL)
                {
                    progressReportSendValue = buildProgressCounters();
                    MPI_Isend(progressReportSendValue.data(), MC_PROGRESS_COUNTERS, MPI_UNSIGNED_LONG_LONG, 0,
                              RW_PROGRESS_TAG, this->comm_world, &progressReportSendReq);
                    progressLastReportSendTime = elapsed_s;
                }
            }
#endif

            if(this->rank_world == 0 && elapsed_s - this->lastProgressPrintTime_ >= 10.0)
            {
                this->lastProgressPrintTime_ = elapsed_s;
                std::array<unsigned long long, MC_PROGRESS_COUNTERS> globalCounters{};
#ifdef RICH_MPI
                for(const auto &counters : progressCountersByRank)
                {
                    for(size_t i = 0; i < globalCounters.size(); ++i)
                        globalCounters[i] += counters[i];
                }
#else
                globalCounters = buildProgressCounters();
#endif
                int64_t globalRemaining = amountManager.GetValue();
                int64_t globalDone = globalInitialForProgress - globalRemaining;
                double done_frac = (globalInitialForProgress > 0) ? static_cast<double>(globalDone) / static_cast<double>(globalInitialForProgress) : 0.0;
                double rate = (elapsed_s > 0) ? static_cast<double>(globalDone) / elapsed_s : 0.0;
                double eta = (rate > 0) ? static_cast<double>(globalRemaining) / rate : 0.0;
                RankHandler_t *selfHandler = this->rankHandlers[this->rank_world];
                int localRemaining = selfHandler ? selfHandler->th_length : 0;
                std::cerr << "[Progress] ~"
                          << (done_frac * 100.0) << "% done, "
                          << elapsed_s << "s elapsed, "
                          << "~" << eta << "s ETA, "
                          << "global_done=" << globalDone << "/" << globalInitialForProgress
                          << " rank0_local_remaining=" << localRemaining
                          << " rw_steps_total=" << globalCounters[MC_PROGRESS_RW_STEPS]
                          << " ddmc_steps_total=" << globalCounters[MC_PROGRESS_DDMC_STEPS]
                          << " ddmc_leaks=" << globalCounters[MC_PROGRESS_DDMC_LEAKS]
                          << " ddmc_census=" << globalCounters[MC_PROGRESS_DDMC_CENSUS]
                          << " ddmc_upscatter=" << globalCounters[MC_PROGRESS_DDMC_UPSCATTER]
                          << " ddmc_fallback=" << globalCounters[MC_PROGRESS_DDMC_FALLBACK]
                          << " eta_is_count_based=1"
                          << std::endl;
            }

            if(verify)
            {
                this->FlushAllSendBuffers();
                this->reallocationAgent->HandleAllWaitingReallocations();
                bool ok = amountManager.GetPendingValue() == 0
                          and this->AllSendBuffersEmpty();
                amountManager.Verify(ok);
            }

            this->iteration++;
        }
    }
    catch(const UniversalError &eo)
    {
        reportError(eo);
        throw;
    }

#ifdef RICH_MPI
    if(this->rank_world != 0 && progressReportSendReq != MPI_REQUEST_NULL)
        MPI_Wait(&progressReportSendReq, MPI_STATUS_IGNORE);
#endif

    MPI_Barrier(this->comm_world);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
   
    START_TIMER_PREEMPTIVE("Boundary Condition");

    MPI_Barrier(this->comm_world);
    start = std::chrono::high_resolution_clock::now();
    data.remaining = this->populationControl->activate(data.remaining);
    MPI_Barrier(this->comm_world);
    end = std::chrono::high_resolution_clock::now();
    double populationControlTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    START_TIMER_PREEMPTIVE("Poststep");
    MPI_Barrier(this->comm_world);
    start = std::chrono::high_resolution_clock::now();
    this->physics->postStep(data.remaining, fullDt);
    MPI_Barrier(this->comm_world);
    end = std::chrono::high_resolution_clock::now();
    double postStepTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    START_TIMER_PREEMPTIVE("Diagnostics");
    auto diagnosticsStart = std::chrono::high_resolution_clock::now();

    size_t newParticlesNum = data.remaining.size();
    this->endParticleCount_ = newParticlesNum;
    size_t leavingNumber = data.leavingCount;

    size_t totalSteps = this->allStepsCounter;
    size_t totalCounterDecrementations = numOfCounterDecrementations;
    size_t callsToTransfer = this->transfersCounter;

    int myStepsCount = 0;
    for(size_t counter : this->cellsStepsCounters)
    {
        myStepsCount += static_cast<int>(counter);
    }
    auto [maxStepsRank, maxStepsVal] = MPI_Max_loc(myStepsCount, this->comm_world);
    auto [maxTransfersRank, maxTransfersVal] = MPI_Max_loc(static_cast<int>(this->transfersCounter), this->comm_world);

    double reallocationTime = 0, maxReallocationTime = 0;
    size_t totalReallocations = 0;
    for(RankHandler_t *handler : this->rankHandlers)
    {
        if(handler == nullptr)
        {
            continue;
        }
        reallocationTime += handler->reallocationTime;
        totalReallocations += handler->reallocationsThisStep;
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

    if(this->rank_world == 0)
    {
        std::cout << "Elapsed: " << elapsed << " seconds, max " << maxReallocationTime << " in reallocation (average: " << reallocationTime / this->size_world << ")" << std::endl;
        std::cout << "Started with " << startingParticleNum << ". Came with " << initialParticlesNum << ". Generated " << preStepParticlesNum << " particles in preStep. ";
        std::cout << "Number of leaving particles is " << leavingNumber << " and remaining (after population control) " << newParticlesNum << ". ";
        std::cout << "Total steps: " << totalSteps << ", total counter decrementations: " << totalCounterDecrementations << std::endl;
        std::cout << "Population control time: " << populationControlTime << ", post step time: " << postStepTime << std::endl;
    }
    std::cout.flush();
    MPI_Barrier(this->comm_world);

    for(const RankHandler_t *handler : this->rankHandlers)
    {
        if(handler == nullptr)
        {
            continue;
        }
        if(handler->th_length != 0)
        {
            UniversalError eo("End of MonteCarloManager::step: th length is not 0");
            eo.addEntry("Rank", this->rank_world);
            eo.addEntry("TH Length", handler->th_length);
            eo.addEntry("Peer Rank", handler->peer_rank_world);
            throw eo;
        }
    }
    
    if(not didRebalance)
    {
        if(this->currentStep > 0 and this->currentStep % SHRINK_BUFFERS_CYCLE == 0)
        {
            if(this->rank_world == 0)
            {
                std::cout << "Doing shrink becuase of step number (currentStep=" << this->currentStep << ", SHRINK_BUFFERS_CYCLE=" << SHRINK_BUFFERS_CYCLE << ")" << std::endl;
            }
            this->ShrinkAllBuffers();
        }
    }

    double diagnosticsTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - diagnosticsStart).count();
    double managerTotalTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - managerStepStart).count();
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &initTime, &initTime, 1, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &addParticlesTime, &addParticlesTime, 1, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &diagnosticsTime, &diagnosticsTime, 1, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
    MPI_Reduce((this->rank_world == 0) ? MPI_IN_PLACE : &managerTotalTime, &managerTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, this->comm_world);
    if(this->rank_world == 0)
    {
        double accounted = maxPreStepTime + elapsed + populationControlTime + postStepTime;
        std::cout << "Manager breakdown (max): init=" << initTime << "s, addParticles=" << addParticlesTime
                  << "s, prestep=" << maxPreStepTime << "s, mainLoop=" << elapsed
                  << "s, popControl=" << populationControlTime << "s, postStep=" << postStepTime
                  << "s, diagnostics=" << diagnosticsTime << "s, total=" << managerTotalTime
                  << "s, previously accounted=" << accounted << "s" << std::endl;
    }

    return data.remaining;
}

#endif // MONTE_CARLO_MANAGER_HPP
