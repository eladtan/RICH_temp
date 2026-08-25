#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <functional>
#include <cstdint>
#include <limits>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include "ProgressTracker.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include <MeshDecomposer3D/load_balancing/LoadBalancer.hpp>
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/simulation/steps/PhysicsStep.hpp"
#include "newtonian/three_dimensional/simulation/IndividualTimeStep.hpp"
#include "newtonian/three_dimensional/time_step_function3D.hpp"
#include "utils/debug/vtune.h"

#ifdef RICH_MPI
    #include <mpi.h>
    #include "mpi/mpi_commands.hpp"
    #include "mpi/mpi_commands.hpp"
    #include "mpi/ExchangeChain.hpp"
#endif // RICH_MPI

class Simulation
{
public:
    using IndividualAMRCallback =
        std::function<IndividualAMRChangeSet(IndividualStepContext const&)>;
    using IndividualPostPhysicsCallback =
        std::function<void(IndividualStepContext const&)>;

    Simulation(Tessellation3D &tess, const std::vector<ComputationalCell3D> &cells, EquationOfState &eos, bool new_start = true);

    inline ProgressTracker &getTracker(void){return this->tracker;};

    size_t& GetMaxID(void);
    const size_t& GetMaxID(void) const;

    void initializeCellIDs(void);
    void recomputeMaxID(void);

    inline void SetTimeStepFunction(std::shared_ptr<TimeStepFunction3D> tsc){this->tsc = tsc;};

    inline Tessellation3D &getTessellation(void){return this->tess;};

    inline const Tessellation3D &getTessellation(void) const{return this->tess;};

    inline std::vector<ComputationalCell3D> &getCells(void){return this->cells;};

    inline std::vector<Conserved3D> &getExtensives(void){return this->extensives;};

    inline const std::vector<ComputationalCell3D> &getCells(void) const{return this->cells;};

    inline const std::vector<Conserved3D> &getExtensives(void) const{return this->extensives;};

    inline const std::vector<std::shared_ptr<PhysicsStep>> &getPhysicsSteps(void) const{return this->physics;};
    
    inline std::vector<std::shared_ptr<PhysicsStep>> &getPhysicsSteps(void){return this->physics;};

    inline const std::map<std::string, double> &getLastPhysicsTimes(void) const{return this->lastPhysicsTimes;};
    inline const std::map<std::string, double> &getLastLocalPhysicsTimes(void) const{return this->lastLocalPhysicsTimes;};

    void SetTimeStep(double dt);

    double GetTimeStep(void) const;

    void step(void);

    void EnableIndividualTimeSteps(
        IndividualTimeStepOptions options = IndividualTimeStepOptions());

    /** Request one all-active individual event.
     *
     * The request is process-local and clears only after the event commits.
     * Callers must not checkpoint while waiting for the requested event.
     */
    void RequestSynchronizedIndividualEvent(void);

    /** True when every owned primitive is committed at the current event tick
     * and the tessellation contains every owned cell. Collective under MPI.
     */
    bool IndividualStateSynchronized(void) const;

    void SetIndividualAMR(IndividualAMRCallback callback)
    {this->individualAMR = std::move(callback);};

    /** Run a collective, run-specific state update after all individual
     * physics steps and before the next timestep limits are evaluated.
    */
    void SetIndividualPostPhysics(IndividualPostPhysicsCallback callback)
    {this->individualPostPhysics = std::move(callback);};

    TimeIntegrationMode GetTimeIntegrationMode(void) const{return this->timeIntegrationMode;};

    const IndividualTimeStepScheduler *GetIndividualTimeStepScheduler(void) const
    {return this->individualScheduler.get();};

    IndividualTimeStepScheduler *GetIndividualTimeStepScheduler(void)
    {return this->individualScheduler.get();};

    bool IndividualEventInProgress(void) const
    {return this->individualEventInProgress;};

    void addPhysics(std::shared_ptr<PhysicsStep> physics);

    double GetTime(void) const;

    size_t GetCycle(void) const;

    void SetCycle(size_t cycle);

    void SetTime(double time);

    double GetWallclockTime(void) const;

    void SetWallclockTime(double t);

    #ifdef RICH_MPI
        void buildDataTransfer(void);

        void buildDataTransfer(const ExchangeChain &chain);

        /**
         * Adds a buffer that should be transferred between mesh movements
         */
        template<typename T>
        void addMigrationBuffer(std::vector<T> &buffer);

        void storeLoadBalance(const std::string &name, std::shared_ptr<LoadBalancer<Vector3D>> lb);

        void setCurrentLoadBalance(const std::string &name);

        void PresetLoadBalance(const std::string &name);

        std::vector<std::pair<std::string, std::shared_ptr<LoadBalancer<Vector3D>>>> GetLoads(void) const;

        inline const std::string &getCurrentLB() const{return this->currentLB;};

        inline void setForceRebalanceSteps(size_t n){this->forceRebalanceSteps = n;};
    #endif // RICH_MPI

private:
    void stepIndividual(void);

#ifdef RICH_MPI
    struct IndividualRebalanceResult
    {
        bool applied = false;
        double weightSkew = 1;
        unsigned long long migratedCells = 0;
        unsigned long long totalOwnedCells = 0;
        double localSeconds = 0;
        double maximumSeconds = 0;
        double currentRssBeforeKiB = -1;
        double currentRssAfterKiB = -1;
    };

    std::shared_ptr<PhysicsStep> findIndividualBalanceStep(void) const;

    IndividualRebalanceResult rebalanceCommittedIndividualState(
        std::shared_ptr<PhysicsStep> const& balanceStep,
        bool forceRebalance, double threshold);

    void maybeRebalanceBeforeFirstIndividualEvent(void);
#endif

    int rank, size;
    Tessellation3D &tess;
    std::vector<std::shared_ptr<PhysicsStep>> physics;
    std::vector<ComputationalCell3D> cells;
    std::vector<Conserved3D> extensives;
    ProgressTracker tracker;
    EquationOfState &eos;
    size_t Max_ID;
    double wallclockTime;
    bool initializedFromRestart;
    std::shared_ptr<TimeStepFunction3D> tsc; // todo: why?
    TimeIntegrationMode timeIntegrationMode = TimeIntegrationMode::Global;
    std::unique_ptr<IndividualTimeStepScheduler> individualScheduler;
    IndividualAMRCallback individualAMR;
    IndividualPostPhysicsCallback individualPostPhysics;
    bool individualEventInProgress = false;
    bool individualSynchronizedEventRequested = false;
    std::map<std::string, double> lastPhysicsTimes;
    std::map<std::string, double> lastLocalPhysicsTimes;

#ifdef RICH_MPI
    std::shared_ptr<LoadBalancer<Vector3D>> currentLoad;

    struct MigrationBuffer
    {
        std::any ref;
        std::function<void(void)> transfer;
        std::function<void(const ExchangeChain &chain)> transferChain;
    };

    std::vector<MigrationBuffer> migrationBuffers; // buffers that need to be moved after each call to 'BuildParallel'

    std::string currentLB;
    std::map<std::string, std::shared_ptr<LoadBalancer<Vector3D>>> loads;
    size_t forceRebalanceSteps = 0;
    std::pair<Vector3D, Vector3D> currentBox;
    size_t lastRebalanceCycle = std::numeric_limits<size_t>::max();
    size_t lastIndividualBalanceCheckCycle =
        std::numeric_limits<size_t>::max();
    std::uint64_t individualOwnershipEpoch = 0;
    double lastIndividualRebalanceSeconds = 0;
    bool preFirstIndividualRebalanceChecked = false;
#endif // RICH_MPI
};

#ifdef RICH_MPI
    template<typename T>
    void Simulation::addMigrationBuffer(std::vector<T> &buffer)
    {
        MigrationBuffer buff;
        buff.ref = std::ref(buffer);
        buff.transfer = [&buffer, this](void){MPI_exchange_data<T>(tess, buffer, false);};
        buff.transferChain = [&buffer, this](const ExchangeChain &chain){MPI_exchange_data<T>(chain, buffer);};
        this->migrationBuffers.push_back(buff);
    }
#endif // RICH_MPI

#endif // SIMULATION_HPP
