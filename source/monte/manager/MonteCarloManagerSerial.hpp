#ifndef MONTE_CARLO_MANAGER_SERIAL_HPP
#define MONTE_CARLO_MANAGER_SERIAL_HPP

#include <memory>
#include <random>
#include <chrono>
#include "monte/MonteCarloParticle.hpp"
#include "monte/physics/MonteCarloPhysics.hpp"
#include "monte/population/PopulationControl.hpp"
#include "tools/ProgressCounter.hpp"
#include "monte/boundary/BoundaryCondition.hpp"
#include "utils/debug/vtune.h" // TODO: remove

#define MONTECARLO_EPSILON 1e-8
#define REALLOCATION_FACTOR 2
#define DEFAULT_BUFFER_SIZE 1000

template<typename T, typename Grid>
class MonteCarloManagerSerial
{
    using index_t = uint32_t;
    using MCParticle = MonteCarloParticle<T, Grid>;

public:
    struct MonteCarloStepFinalData
    {
        std::vector<MCParticle> remaining;
        std::vector<MCParticle> leaving;
    };

    MonteCarloManagerSerial(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics,
                    const std::shared_ptr<PopulationControl<T, Grid>> &populationControl,
                    const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition);

    ~MonteCarloManagerSerial();

    std::vector<MCParticle> step(const std::vector<MCParticle> &particleList, dt_t fullDt);
    
    class Tracker
    {
    public:
        Tracker(void);

        void Reset(void);

        std::vector<MCParticle> GetTrackParticleRoute(size_t id);

        void ReportParticle(MCParticle &particle);
    
    private:
        boost::container::flat_map<size_t, std::vector<MCParticle>> track;
    };

    inline const Tracker &getTracker(void){return this->tracker;};

    inline void resetTracker(void){this->tracker.Reset();};

private:
    const Grid &grid;
    size_t Ncells;
    int progress;
    T ll, ur;
    std::shared_ptr<MonteCarloPhysics<T, Grid>> physics;
    std::shared_ptr<PopulationControl<T, Grid>> populationControl;
    std::shared_ptr<BoundaryCondition<T, Grid>> boundaryCondition;
    Tracker tracker;
    size_t myIDCounter;

    struct
    {
        size_t buffsize;
        MCParticle *particles;
        index_t *av;
        size_t av_length;
        index_t *th;
        size_t th_length;
    } particlesData;

    void RemoveParticles(const std::vector<size_t> &indicesInToHandle, size_t num);

    void HandleAll(MonteCarloStepFinalData &cache);

    void PutSelfParticles(const std::vector<MCParticle> &particles);

    void PrepareForStep(void);

    void AddParticles(const std::vector<MCParticle> &particles);
};

template<typename T, typename Grid>
MonteCarloManagerSerial<T, Grid>::MonteCarloManagerSerial(const Grid &grid, const std::shared_ptr<MonteCarloPhysics<T, Grid>> &physics, const std::shared_ptr<PopulationControl<T, Grid>> &populationControl, 
                                            const std::shared_ptr<BoundaryCondition<T, Grid>> &boundaryCondition):
    grid(grid), physics(physics), populationControl(populationControl), boundaryCondition(boundaryCondition), myIDCounter(0)

{
    this->particlesData.buffsize = 0;
    this->particlesData.particles = nullptr;
    this->particlesData.av = nullptr;
    this->particlesData.th = nullptr;
}

template<typename T, typename Grid>
void MonteCarloManagerSerial<T, Grid>::PrepareForStep(void)
{
    this->Ncells = this->grid.GetPointNo();
    std::tie(this->ll, this->ur) = this->grid.GetBoxCoordinates();
}

template<typename T, typename Grid>
void MonteCarloManagerSerial<T, Grid>::AddParticles(const std::vector<MCParticle> &particles)
{
    if(this->particlesData.av_length < particles.size())
    {
        // allocation is needed
        size_t newBuffSize = std::max(this->particlesData.buffsize * REALLOCATION_FACTOR, this->particlesData.buffsize + particles.size());
        size_t oldBuffSize = this->particlesData.buffsize;
        this->particlesData.buffsize = newBuffSize;
        MCParticle *new_particles = new MCParticle[this->particlesData.buffsize];
        index_t *new_av = new index_t[this->particlesData.buffsize];
        index_t *new_th = new index_t[this->particlesData.buffsize];
        std::memcpy(new_particles, this->particlesData.particles, oldBuffSize * sizeof(MCParticle));
        std::memcpy(new_av, this->particlesData.av, oldBuffSize * sizeof(index_t));
        std::memcpy(new_th, this->particlesData.th, oldBuffSize * sizeof(index_t));
        delete[] this->particlesData.particles;
        delete[] this->particlesData.av;
        delete[] this->particlesData.th;
        this->particlesData.particles = new_particles;
        this->particlesData.av = new_av;
        this->particlesData.th = new_th;
        // set `av`
        assert(oldBuffSize < newBuffSize);
        index_t difference = newBuffSize - oldBuffSize;
        std::iota(new_av, new_av + difference, oldBuffSize);
        this->particlesData.av_length += static_cast<int>(difference);
    }

    // set particles
    // update 'to handle' and 'available' lists accordingly
    index_t particlesNum = particles.size();
    this->particlesData.av_length -= particlesNum;
    index_t *avIndices = this->particlesData.av + this->particlesData.av_length;
    index_t *thIndices = this->particlesData.th + this->particlesData.th_length;
    this->particlesData.th_length += particlesNum;
    size_t firstID = this->myIDCounter;
    this->myIDCounter += particles.size();

    for(size_t i = 0; i < particlesNum; i++)
    {
        index_t idx = avIndices[i];
        // copy particle
        MCParticle *particle = this->particlesData.particles + idx;
        std::memcpy(particle, &particles[i], sizeof(MCParticle));
        // set to handle
        thIndices[i] = idx;
        particle->id = firstID + i;
    }
}

template<typename T, typename Grid>
MonteCarloManagerSerial<T, Grid>::~MonteCarloManagerSerial()
{
    delete[] this->particlesData.av;
    this->particlesData.av = nullptr;
    delete[] this->particlesData.th;
    this->particlesData.th = nullptr;
    delete[] this->particlesData.particles;
    this->particlesData.particles = nullptr;
}

template<typename T, typename Grid>
MonteCarloManagerSerial<T, Grid>::Tracker::Tracker(void)
{}

template<typename T, typename Grid>
void MonteCarloManagerSerial<T, Grid>::Tracker::Reset(void)
{
    this->track.clear();
}

template<typename T, typename Grid>
void MonteCarloManagerSerial<T, Grid>::Tracker::ReportParticle(MCParticle &particle)
{
    if(this->track.find(particle.id) == this->track.end())
    {
        this->track[particle.id] = std::vector<MCParticle>();
    }
    this->track[particle.id].push_back(particle);
}

template<typename T, typename Grid>
std::vector<typename MonteCarloManagerSerial<T, Grid>::MCParticle> MonteCarloManagerSerial<T, Grid>::Tracker::GetTrackParticleRoute(size_t id)
{
    if(this->track.find(id) == this->track.end())
    {
        return std::vector<MCParticle>();
    }
    return this->track[id];
}
template<typename T, typename Grid>
void MonteCarloManagerSerial<T, Grid>::MonteCarloManagerSerial::PutSelfParticles(const std::vector<MCParticle> &particles)
{
    size_t particlesNum = particles.size();
    bool reallocated = false;

    MCParticle *old_particles = this->particlesData.particles;
    index_t *old_av = this->particlesData.av;
    index_t *old_th = this->particlesData.th;

    if(this->particlesData.buffsize < particlesNum)
    {
        this->particlesData.buffsize = particlesNum;
        MCParticle *new_particles = new MCParticle[this->particlesData.buffsize];
        index_t *new_th = new index_t[this->particlesData.buffsize];
        index_t *new_av = new index_t[this->particlesData.buffsize];
        this->particlesData.th = new_th;
        this->particlesData.av = new_av;
        this->particlesData.particles = new_particles;
    }
    this->particlesData.av_length = 0;
    this->particlesData.th_length = 0;

    std::memcpy(this->particlesData.particles, particles.data(), particles.size() * sizeof(MCParticle));

    if(reallocated)
    {
        delete[] old_particles;
        delete[] old_av;
        delete[] old_th;
    }
    // update 'to handle' and 'available' lists accordingly
    this->particlesData.th_length = particlesNum;
    for(size_t i = 0; i < particlesNum; i++)
    {
        assert(i < this->particlesData.buffsize);
        this->particlesData.th[i] = i;
    }

    size_t availLength = this->particlesData.buffsize - particlesNum;
    this->particlesData.av_length = availLength;
    
    for(size_t i = 0; i < availLength; i++)
    {
        size_t idx = i + particlesNum;
        assert(idx < this->particlesData.buffsize);
        this->particlesData.av[i] = idx;
    }

    size_t firstID = this->myIDCounter;
    size_t assignedCounter = 0;
    for(size_t i = 0; i < particlesNum; i++)
    {
        if(this->particlesData.particles[i].id == std::numeric_limits<size_t>::max())
        {
            this->particlesData.particles[i].id = firstID + assignedCounter;
            assignedCounter++;
        }
        // std::cout << "Assigned ID " << firstID + i << std::endl;
    }
    this->myIDCounter += assignedCounter;
}

template<typename T, typename Grid>
void MonteCarloManagerSerial<T, Grid>::RemoveParticles(const std::vector<size_t> &indicesInToHandle, size_t num)
{
    for(int i = static_cast<int>(num) - 1; i >= 0; i--)
    {
        const size_t &toHandleIndex = indicesInToHandle[i];
        assert(i == 0 or indicesInToHandle[i] > indicesInToHandle[i-1]); // should be in a descending order
        assert(toHandleIndex < this->particlesData.th_length);
        index_t particleIdx = this->particlesData.th[toHandleIndex];
        assert(this->particlesData.av_length < this->particlesData.buffsize);
        this->particlesData.av[this->particlesData.av_length++] = particleIdx;
        this->particlesData.th[toHandleIndex] = this->particlesData.th[--this->particlesData.th_length];
        assert(this->particlesData.th_length >= 0);
    }    
}

template<typename T, typename Grid>
void MonteCarloManagerSerial<T, Grid>::MonteCarloManagerSerial::HandleAll(MonteCarloStepFinalData &stepData)
{
    static std::vector<size_t> removeParticlesVec;
        
    size_t removeCounter = 0;

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

    int length = this->particlesData.th_length;
    distance_t scatteringLength = abs(this->ur - this->ll) / 10;

    for(int i = 0; i < length; i++)
    {
        assert(i < this->particlesData.buffsize);
        size_t particleIndex = this->particlesData.th[i];
        assert(particleIndex < this->particlesData.buffsize);
        MCParticle &particle = this->particlesData.particles[particleIndex];
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

            if(BOOST_LIKELY(nextCellIndex < this->Ncells))
            {
                // local neighbor
                particle.location = (1 - MONTECARLO_EPSILON) * particle.location + MONTECARLO_EPSILON * this->grid.GetMeshPoint(nextCellIndex);
                particle.cellIndex = nextCellIndex;
            }
            else
            {
                // leaving domain
                MonteCarloParticleStatus status = this->boundaryCondition->apply(particle);
                if(status == MonteCarloParticleStatus::REFLECT)
                {}
                else if(status == MonteCarloParticleStatus::REMOVE)
                {
                    // std::cout << "Particle " << particle << "(" << &particle << ")" << " is leaving the domain" << std::endl;
                    stepData.leaving.push_back(particle);
                    removeParticle(i);
                }
                else
                {
                    std::cerr << "Unknown boundary condition for particle " << particle << std::endl;
                    exit(1);
                }
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

    if(removeCounter > 0)
    {
        this->RemoveParticles(removeParticlesVec, removeCounter);
    }
}

template<typename T, typename Grid>
std::vector<typename MonteCarloManagerSerial<T, Grid>::MCParticle> MonteCarloManagerSerial<T, Grid>::MonteCarloManagerSerial::step(const std::vector<MCParticle> &particleList, dt_t fullDt)

{
    this->PrepareForStep();
    this->physics->updateGridData();
    std::vector<MCParticle> particlesListCpy = particleList;
    
    this->resetTracker();

    std::vector<MCParticle> newParticles1 = this->physics->preStep(fullDt, particlesListCpy);
    this->PutSelfParticles(particlesListCpy);

    int length = this->particlesData.th_length;
    for(int i = 0; i < length; i++)
    {
        size_t particleIndex = this->particlesData.th[i];
        MCParticle &p = this->particlesData.particles[particleIndex];
        p.timeLeft = fullDt;
        p.initialWeight = p.weight;
        p.steps = 0;
    }
    this->AddParticles(newParticles1);
    
    size_t numParticles = this->particlesData.th_length;
    
    MonteCarloStepFinalData data;
    // measure time
    vtune_start();
    auto start = std::chrono::high_resolution_clock::now();

    try
    {
        while(this->particlesData.th_length != 0)
        {
            this->HandleAll(data);
        }
    }
    catch(const UniversalError &eo)
    {
        reportError(eo);
        throw;
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::vector<MCParticle> populationControlParticles = this->populationControl->activate(data.remaining);
    this->physics->postStep(populationControlParticles, fullDt);

    double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    // std::cout << "I'm outside of step() loop, in " << seconds << " seconds (" << numParticles << " particles after prestep, originally came with " << particleList.size() << " particles)" << std::endl;

    // std::cout << "Number of leaving particles is " << data.leaving.size() << " and remaining (after population control) " << populationControlParticles.size() << std::endl;
    return populationControlParticles;
}

#endif // MONTE_CARLO_MANAGER_SERIAL_HPP