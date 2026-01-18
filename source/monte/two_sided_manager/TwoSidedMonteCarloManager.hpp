#ifndef TWO_SIDED_MONTE_CARLO_MANAGER_HPP
#define TWO_SIDED_MONTE_CARLO_MANAGER_HPP

#include <memory>
#include <random>
#include <boost/container/flat_set.hpp>
#include <mpi.h>
#include "mpi/mpi_commands.hpp"
#include "mpi/serialize/mpi_commands.hpp"
#include "monte/utils/GhostMap.hpp"
#include "monte/MonteCarloParticle.hpp"
#include "monte/physics/MonteCarloPhysics.hpp"
#include "monte/population/PopulationControl.hpp"
#include "monte/boundary/BoundaryCondition.hpp"
#include "monte/manager/tools/ParticleAmountManager.hpp"
#include "monte/manager/tools/ParticleAmountManager2.hpp"
#include "BuffersManager.hpp"

#define MONTECARLO_EPSILON 1e-8
#define PARTICLES_TAG 8817
#define RECV_BUFFER_MAX_SIZE 1000
#define SEND_BUFFER_DISPATCH_MIN_SIZE 500
#define SEND_BUFFER_DISPATCH_MIN_CYCLES 500

template<typename T, typename Grid>
class TwoSidedMonteCarloManager
{
    using MCParticle = MonteCarloParticle<T, Grid>;

public:
    TwoSidedMonteCarloManager(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics,
                    const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
                    const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition,
                    const MPI_Comm &comm = MPI_COMM_WORLD);

    virtual ~TwoSidedMonteCarloManager() = default;

    inline size_t GetStepCounter(void) const{return this->allStepsCounter;};

    inline const std::vector<size_t> &GetCellsStepsCounters(void) const {return this->cellsStepsCounters;}

    // todo: should return that?
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
    struct MonteCarloStepFinalData
    {
        std::vector<MCParticle> remaining;
        std::vector<MCParticle> leaving;
    };

    const Grid &grid;
    MPI_Comm comm_world;
    rank_t rank_world, size_world;
    size_t Ncells;
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ranks_ghost_map;
    T ll, ur;
    std::shared_ptr<MonteCarloPhysics<T, Grid>> physics;
    std::shared_ptr<PopulationControl<T, Grid>> populationControl;
    std::shared_ptr<BoundaryCondition<T, Grid>> boundaryCondition;
    
    std::vector<MCParticle> particles;
    std::shared_ptr<BuffersManager<MCParticle>> buffersManager;
    typename ParticleAmountManager2::counter_t localDecrementAmount;
    Tracker tracker;

    size_t allStepsCounter;
    std::vector<size_t> cellsStepsCounters;

    size_t iteration;
    size_t myIDCounter;
    size_t currentStep;

    bool HandleAll(MonteCarloStepFinalData &stepData);

    void PutSelfParticles(const MCParticle *particles, size_t particlesNum);

    void RemoveParticles(const std::vector<size_t> &indices);
};

template<typename T, typename Grid>
TwoSidedMonteCarloManager<T, Grid>::TwoSidedMonteCarloManager(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics,
                    const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
                    const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition,
                    const MPI_Comm &comm):
    grid(grid), physics(physics), populationControl(populationControl), boundaryCondition(boundaryCondition), comm_world(comm), tracker(comm)
{
    MPI_Comm_rank(this->comm_world, &this->rank_world);
    MPI_Comm_size(this->comm_world, &this->size_world);

    this->myIDCounter = 0;
    this->currentStep = 0;
}

template<typename T, typename Grid>
TwoSidedMonteCarloManager<T, Grid>::Tracker::Tracker(const MPI_Comm &comm): comm(comm)
{}

template<typename T, typename Grid>
void TwoSidedMonteCarloManager<T, Grid>::Tracker::Reset(void)
{
    this->track.clear();
}

template<typename T, typename Grid>
void TwoSidedMonteCarloManager<T, Grid>::Tracker::ReportParticle(MCParticle &particle)
{
    if(this->track.find(particle.id) == this->track.end())
    {
        this->track[particle.id] = std::vector<MCParticle>();
    }
    this->track[particle.id].push_back(particle);
}

template<typename T, typename Grid>
std::vector<typename TwoSidedMonteCarloManager<T, Grid>::MCParticle> TwoSidedMonteCarloManager<T, Grid>::Tracker::GetLocalTrackParticleRoute(size_t id)
{
    if(this->track.find(id) == this->track.end())
    {
        return std::vector<MCParticle>();
    }
    return this->track[id];
}

template<typename T, typename Grid>
std::vector<typename TwoSidedMonteCarloManager<T, Grid>::MCParticle> TwoSidedMonteCarloManager<T, Grid>::Tracker::GetTrackParticleRoute(size_t id)
{
    std::vector<MCParticle> local = this->GetLocalTrackParticleRoute(id);
    std::vector<MCParticle> global = MPI_All_cast(local, this->comm);
    // sort by `particle.steps`
    std::sort(global.begin(), global.end(), [](const MCParticle &a, const MCParticle &b) { return a.steps < b.steps; });
    return global;
}

template<typename T, typename Grid>
void TwoSidedMonteCarloManager<T, Grid>::PutSelfParticles(const MCParticle *newParticles, size_t particlesNum)
{
    #ifdef MONTECARLO_DEBUG
    boost::container::flat_set<std::pair<rank_t, size_t>> particlesSet;
    for(size_t i = 0; i < particlesNum; i++)
    {
        const MCParticle &particle = newParticles[i];
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

    size_t oldSize = this->particles.size();
    this->particles.insert(this->particles.end(), newParticles, newParticles + particlesNum);
    size_t firstID = this->myIDCounter;
    this->myIDCounter += particlesNum;

    for(size_t i = 0; i < particlesNum; i++)
    {
        size_t idx = oldSize + i;
        MCParticle &particle = this->particles[idx];
        if(particle.id == std::numeric_limits<size_t>::max())
        {
            // no ID has been assigned
            particle.rank = this->rank_world;
            particle.id = firstID + i;
        }
    }
}

template<typename T, typename Grid>
void TwoSidedMonteCarloManager<T, Grid>::RemoveParticles(const std::vector<size_t> &indices)
{
    assert(not indices.empty());
    for(long long int i = indices.size() - 1; i >= 0; i--)
    {
        size_t particleIndex = indices[i];
        if(i > 0)
        {
            assert(particleIndex > indices[i - 1]);
        }
        std::swap(this->particles[particleIndex], this->particles.back());
        this->particles.pop_back();
    }
}

template<typename T, typename Grid>
bool TwoSidedMonteCarloManager<T, Grid>::HandleAll(MonteCarloStepFinalData &stepData)
{
    static std::vector<size_t> removeParticlesVec;
    static std::vector<MCParticle> newParticles;
    removeParticlesVec.clear();
    newParticles.clear();
    
    auto eliminateParticle = [&](size_t particleIndex)
    {
        removeParticlesVec.push_back(particleIndex);
    };

    auto transferParticle = [&](size_t particleIndex, rank_t toRank)
    {
        assert(toRank != this->rank_world); // can't send to self
        this->buffersManager->Add(toRank, this->particles[particleIndex]);
        eliminateParticle(particleIndex);
    };

    auto removeParticle = [&](size_t particleIndex)
    {
        eliminateParticle(particleIndex);
        this->localDecrementAmount += 1;
    };

    this->iteration++;

    size_t length = this->particles.size();
    for(size_t i = 0; i < length; i++)
    {
        assert(i < this->particles.size());
        MCParticle &particle = this->particles[i];
        bool debug = false;

        #ifdef MONTECARLO_DEBUG
        if(particle.lastSeen == this->iteration and particle.lastSeenRank == this->rank_world)
        {
            UniversalError eo("Particle was already handled in this iteration");
            auto it = std::find(this->particles.cbegin() + i, this->particles.cend(), particle);
            if(it != this->particles.cend())
            {
                eo.addEntry("Second Location", std::distance(this->particles.cbegin(), it));
            }
            eo.addEntry("My Rank", this->rank_world);
            eo.addEntry("Particle", particle);
            eo.addEntry("Iteration", this->iteration);
            throw eo;
        }
        particle.lastSeen = this->iteration;
        particle.lastSeenRank = this->rank_world;
        particle.lastSeenIndex = i;
        #endif // MONTECARLO_DEBUG

        while(true)
        {
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
                throw eo;
            }
            if(particle.removedFromRank)
            {
                continue; 
                UniversalError eo("Particle was removed from rank, but still in the list");
                eo.addEntry("Particle", particle);
                eo.addEntry("Rank", this->rank_world);
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
                    eo.addEntry("Particle Index In This Rank", i);
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
                newParticles.insert(newParticles.end(), functionality.particlesToAdd.cbegin(), functionality.particlesToAdd.cend());
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
                            removeParticle(i);
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
                    particle.particleIndexInLastRank = i;
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

                    transferParticle(i, otherRank);
                    break;
                }
            }
            else if(functionality.change == MonteCarloParticleStatus::REMOVE)
            {
                this->allStepsCounter += particle.steps;
                removeParticle(i);
                break;
            }
            else if(functionality.change == MonteCarloParticleStatus::DONE)
            {
                stepData.remaining.push_back(particle);
                this->allStepsCounter += particle.steps;
                // remove particle from current list
                removeParticle(i);
                break;
            }
        }
    }

    if(not removeParticlesVec.empty())
    {
        this->RemoveParticles(removeParticlesVec);
    }
    if(not newParticles.empty())
    {        
        this->PutSelfParticles(newParticles.data(), newParticles.size());
        this->localDecrementAmount -= newParticles.size();
    }
    return (length == 0);
}

template<typename T, typename Grid>
std::vector<typename TwoSidedMonteCarloManager<T, Grid>::MCParticle> TwoSidedMonteCarloManager<T, Grid>::step(const std::vector<MCParticle> &particleList, dt_t fullDt)
{
    try
    {
        this->Ncells = this->grid.GetPointNo();
        this->ranks_ghost_map = GetGhostMap(this->grid);
        std::tie(this->ll, this->ur) = this->grid.GetBoxCoordinates();
        this->particles.clear();

        this->PutSelfParticles(particleList.data(), particleList.size());
        this->resetTracker();
        this->currentStep++;
        this->iteration = 0;
        this->allStepsCounter = 0;
        // this->neighbors = this->grid.GetDuplicatedProcs();    
        this->cellsStepsCounters = std::vector<size_t>(this->Ncells, 0);
        MPI_Barrier(this->comm_world);
        
        size_t length = this->particles.size();
        for(int i = 0; i < length; i++)
        {
            MCParticle &p = this->particles[i];
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

        MPI_Barrier(this->comm_world);

        size_t initialParticlesNum = particleList.size();
        
        this->physics->updateGridData();

        std::chrono::high_resolution_clock::time_point preStepStart = std::chrono::high_resolution_clock::now();
        std::vector<MCParticle> newParticles1 = this->physics->preStep(fullDt);
        std::chrono::high_resolution_clock::time_point preStepEnd = std::chrono::high_resolution_clock::now();

        double preStepSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(preStepEnd - preStepStart).count(); // todo: necessary?

        this->PutSelfParticles(newParticles1.data(), newParticles1.size());
        // MPI_Barrier(this->comm_world);

        size_t numParticles = this->particles.size();
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &numParticles, &numParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);

        size_t preStepParticlesNum = newParticles1.size();
        int64_t startingParticleNum = initialParticlesNum + preStepParticlesNum;
        // std::cout << "Rank " << this->rank_world << ", startingParticleNum is " << startingParticleNum << " = " << initialParticlesNum << " + " << preStepParticlesNum << std::endl;

        this->localDecrementAmount = 0;
        ParticleAmountManager2 amountManager(this->comm_world);
        amountManager.Initialize(startingParticleNum);

        MonteCarloStepFinalData data;
        // measure time
        // vtune_start();
        size_t numOfCounterDecrementations = 0;
        auto start = std::chrono::high_resolution_clock::now();
        size_t lastLocalDecrementAmount;
        size_t decrementTryCounter = 0;

        size_t i = 0;
        // volatile int &verify = *amountManager.shouldVerify;
        const bool &done = amountManager.GetDoneRef();
        const bool &verify = amountManager.GetVerifyRef();

        auto receiveCallback = [this](const MCParticle *newValues, size_t newValuesCount, rank_t fromRank)
                                    {
                                        // std::cout << "Rank " << this->rank_world << " is here, got " << newValuesCount << " new particles from rank " << fromRank << "." << std::endl;
                                        this->PutSelfParticles(newValues, newValuesCount);
                                    };
        this->buffersManager = std::make_shared<BuffersManager<MCParticle>>(this->comm_world, receiveCallback, PARTICLES_TAG, RECV_BUFFER_MAX_SIZE * sizeof(MCParticle), SEND_BUFFER_DISPATCH_MIN_SIZE * sizeof(MCParticle), SEND_BUFFER_DISPATCH_MIN_CYCLES, this->size_world);

        bool printed = false; // todo remove
    
        while(not done)
        {        
            i++;
            
            if(i % 20 == 0)
            {
                this->buffersManager->HandleIncomingOutcoming();
            }

            bool isEmpty = this->HandleAll(data);

            amountManager.Decrease(static_cast<ParticleAmountManager2::counter_t>(this->localDecrementAmount));
            this->localDecrementAmount = 0;

            amountManager.Progress();

            if(verify)
            {
                bool ok = this->particles.empty() and this->buffersManager->CountOutcoming() == 0;
                amountManager.Verify(ok);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        
        std::vector<MCParticle> populationControlParticles = this->populationControl->activate(data.remaining);
        this->physics->postStep(populationControlParticles, fullDt);

        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        // std::cout << "Rank " << this->rank_world << " is outside of step() loop, in " << seconds << " seconds (" << numParticles << " particles)" << std::endl;

        size_t newParticlesNum = populationControlParticles.size();
        size_t leavingNumber = data.leaving.size();

        size_t totalSteps = this->allStepsCounter;
        size_t totalCounterDecrementations = numOfCounterDecrementations;
        size_t callsToTransfer = this->buffersManager->GetSentCounter();

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

        myTransfers.x = static_cast<int>(callsToTransfer);
        myTransfers.rank = this->rank_world;
        MPI_Reduce(&myTransfers, &maxTransfers, 1, MPI_2INT, MPI_MAXLOC, 0, this->comm_world);

        // std::cout << "leavingNumber = " << leavingNumber << " and newParticlesNum = " << newParticlesNum << std::endl; 
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &initialParticlesNum, &initialParticlesNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &preStepParticlesNum, &preStepParticlesNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &leavingNumber, &leavingNumber, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &newParticlesNum, &newParticlesNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &totalSteps, &totalSteps, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &totalCounterDecrementations, &totalCounterDecrementations, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &startingParticleNum, &startingParticleNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &callsToTransfer, &callsToTransfer, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, this->comm_world);

        int sent = this->buffersManager->GetSentCounter();
        int recv = this->buffersManager->GetRecvCounter();

        struct
        {
            int x;
            int rank;
        } recvMax, recvRanked = {recv, this->rank_world};
        MPI_Reduce(&recvRanked, &recvMax, 1, MPI_2INT, MPI_MAXLOC, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &sent, &sent, 1, MPI_INT, MPI_SUM, 0, this->comm_world);
        MPI_Reduce((this->rank_world == 0)? MPI_IN_PLACE : &recv, &recv, 1, MPI_INT, MPI_SUM, 0, this->comm_world);

        if(this->rank_world == 0)
        {
            std::cout << "Started with " << startingParticleNum << ". Came with " << initialParticlesNum << ". Generated " << preStepParticlesNum << " particles in preStep. ";
            std::cout << "Number of leaving particles is " << leavingNumber << " and remaining (after population control) " << newParticlesNum << ". ";
            std::cout << "Total steps: " << totalSteps << ", total counter decrementations: " << totalCounterDecrementations << std::endl;
            std::cout << "Total send communications: " << sent << ", total receive communications: " << recv << " (max: " << recvMax.x << " in rank " << recvMax.rank << ")" << std::endl;
            std::cout << "Max steps: " << maxSteps.x << " on rank " << maxSteps.rank << ", average is " << totalSteps / this->size_world << std::endl;
            std::cout << "Max calls to transfer: " << maxTransfers.x << " on rank " << maxTransfers.rank << ", average is " << callsToTransfer / this->size_world << std::endl;
            assert(sent == recv);
        }


        assert(this->particles.empty());
        MPI_Barrier(this->comm_world);

        this->buffersManager = nullptr; // TODO: good?
        // if(this->rank_world == 0)
        // std::cout << "====================================" << std::endl;
        // MPI_Barrier(this->comm_world);
        return populationControlParticles;
    } 
    catch(const UniversalError &eo)
    {
        reportError(eo);
        throw;
    }
}

#endif // TWO_SIDED_MONTE_CARLO_MANAGER_HPP