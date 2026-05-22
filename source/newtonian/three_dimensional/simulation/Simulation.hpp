#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <functional>
#include <limits>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include "ProgressTracker.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "3D/tessellation/loadBalancing/LoadBalancer.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/simulation/steps/PhysicsStep.hpp"
#include "newtonian/three_dimensional/time_step_function3D.hpp"
#include "utils/debug/vtune.h"

#ifdef RICH_MPI
    #include <mpi.h>
    #include "mpi/mpi_commands.hpp"
    #include "mpi/serialize/mpi_commands.hpp"
    #include "mpi/ExchangeChain.hpp"
#endif // RICH_MPI

class Simulation
{
public:
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

    void SetTimeStep(double dt);

    double GetTimeStep(void) const;

    void step(void);

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

        void storeLoadBalance(const std::string &name, std::shared_ptr<LoadBalancer> lb);

        void setCurrentLoadBalance(const std::string &name);

        void PresetLoadBalance(const std::string &name);

        std::vector<std::pair<std::string, std::shared_ptr<LoadBalancer>>> GetLoads(void) const;

        inline const std::string &getCurrentLB() const{return this->currentLB;};

        inline void setForceRebalanceSteps(size_t n){this->forceRebalanceSteps = n;};
    #endif // RICH_MPI

private:
    int rank, size;
    Tessellation3D &tess;
    std::vector<std::shared_ptr<PhysicsStep>> physics;
    std::vector<ComputationalCell3D> cells;
    std::vector<Conserved3D> extensives;
    ProgressTracker tracker;
    EquationOfState &eos;
    size_t Max_ID;
    double wallclockTime;
    std::shared_ptr<TimeStepFunction3D> tsc; // todo: why?

#ifdef RICH_MPI
    std::shared_ptr<LoadBalancer> currentLoad;

    struct MigrationBuffer
    {
        std::any ref;
        std::function<void(void)> transfer;
        std::function<void(const ExchangeChain &chain)> transferChain;
    };

    std::vector<MigrationBuffer> migrationBuffers; // buffers that need to be moved after each call to 'BuildParallel'

    std::string currentLB;
    std::map<std::string, std::shared_ptr<LoadBalancer>> loads;
    size_t forceRebalanceSteps = 0;
    std::pair<Vector3D, Vector3D> currentBox;
    size_t lastRebalanceCycle = std::numeric_limits<size_t>::max();

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