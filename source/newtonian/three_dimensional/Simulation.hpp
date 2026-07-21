#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <functional>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include "ProgressTracker.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "3D/tessellation/loadBalancing/LoadBalancer.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "PhysicsStep.hpp"
#include "newtonian/three_dimensional/time_step_function3D.hpp"

#ifdef RICH_MPI
    #include <mpi.h>
    #include "mpi/mpi_commands.hpp"
    #include "mpi/serialize/mpi_commands.hpp"
    #include "mpi/ExchangeChain.hpp"
#endif // RICH_MPI

class Simulation
{
public:
    Simulation(Tessellation3D &tess, const std::vector<ComputationalCell3D> &cells, EquationOfState &eos, TimeStepFunction3D &tsc);

    inline ProgressTracker &getTracker(void){return this->tracker;};

    inline Tessellation3D &getTessellation(void){return this->tess;};

    inline std::vector<ComputationalCell3D> &getCells(void){return this->cells;};

    inline std::vector<Conserved3D> &getExtensives(void){return this->extensives;};

    inline const std::vector<ComputationalCell3D> &getCells(void) const{return this->cells;};

    inline const std::vector<Conserved3D> &getExtensives(void) const{return this->extensives;};

    inline void SetTimeStep(double dt){this->tsc.SetTimeStep(dt);};

    inline double GetTimeStep(void) const{return this->tsc.GetTimeStep();};

    void step(void);

    void addPhysics(std::shared_ptr<PhysicsStep> physics);

    double GetTime(void) const;

    size_t GetCycle(void) const;

    void SetCycle(size_t cycle);

    void SetTime(double time);

    #ifdef RICH_MPI
        void buildDataTransfer(void);

        void buildDataTransfer(const ExchangeChain &chain);

        /**
         * Adds a buffer that should be transferred between mesh movements
         */
        template<typename T>
        void addMigrationBuffer(std::vector<T> &buffer);
    #endif // RICH_MPI

private:
    int rank, size;
    Tessellation3D &tess;
    std::vector<std::shared_ptr<PhysicsStep>> physics;
    std::vector<ComputationalCell3D> cells;
    std::vector<Conserved3D> extensives;
    ProgressTracker tracker;
    EquationOfState &eos;
    TimeStepFunction3D &tsc;

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