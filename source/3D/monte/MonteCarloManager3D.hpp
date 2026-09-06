#ifndef MONTECARLO_MANAGER_3D_HPP
#define MONTECARLO_MANAGER_3D_HPP

#include <cassert>
#include <memory>
#include <utility>
#include <vector>

#include "3D/tessellation/Tessellation3D.hpp"
#include "monte/manager/MonteCarloManager.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

using STORM::BoundaryCondition;
using STORM::dt_t;
using STORM::MonteCarloConfig;
using STORM::MonteCarloPhysics;
using STORM::PopulationControl;

class MonteCarloManager3D
{
    using MCParticle = MonteCarloParticle<Vector3D>;

public:
    template<typename Physics>
    MonteCarloManager3D(const Tessellation3D &grid,
                        const std::shared_ptr<Physics> &physics,
                        const std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> &populationControl,
                        const std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> &boundaryCondition,
                        const MonteCarloConfig &config = MonteCarloConfig(),
                        std::unique_ptr<STORM::CommunicationEngine<Vector3D>> engine =
                            std::make_unique<STORM::SerialCommunicationEngine<Vector3D>>());

    const std::vector<size_t> &GetCellsStepsCounters(void) const;
    std::vector<size_t> &GetCellsStepsCounters(void);
    const std::vector<size_t> &GetCellsParticleCounters(void) const;
    size_t GetStartParticleCount(void) const;
    size_t GetInitialParticleCount(void) const;
    size_t GetPreStepParticleCount(void) const;
    size_t GetEndParticleCount(void) const;
    double GetPureComputeTime(void) const;
    const std::vector<size_t> &GetBeginningParticleCount(void) const;
    std::vector<size_t> &GetBeginningParticleCount(void);
    size_t GetHandlerMemoryBytes(void) const;

    /// References returned by getParticles() are invalidated by step().
    const std::vector<MCParticle> &getParticles(void) const;
    std::vector<MCParticle> &getParticles(void);
    void step(const std::vector<ComputationalCell3D> &cells, dt_t fullDt);

private:
    class Implementation
    {
    public:
        virtual ~Implementation() = default;

        virtual const std::vector<size_t> &GetCellsStepsCounters(void) const = 0;
        virtual std::vector<size_t> &GetCellsStepsCounters(void) = 0;
        virtual const std::vector<size_t> &GetCellsParticleCounters(void) const = 0;
        virtual size_t GetStartParticleCount(void) const = 0;
        virtual size_t GetInitialParticleCount(void) const = 0;
        virtual size_t GetPreStepParticleCount(void) const = 0;
        virtual size_t GetEndParticleCount(void) const = 0;
        virtual double GetPureComputeTime(void) const = 0;
        virtual const std::vector<size_t> &GetBeginningParticleCount(void) const = 0;
        virtual std::vector<size_t> &GetBeginningParticleCount(void) = 0;
        virtual size_t GetHandlerMemoryBytes(void) const = 0;
        virtual const std::vector<MCParticle> &GetParticles(void) const = 0;
        virtual std::vector<MCParticle> &GetParticles(void) = 0;
        virtual void Step(const std::vector<ComputationalCell3D> &cells, dt_t fullDt) = 0;
    };

    template<typename Physics>
    class TypedImplementation : public Implementation
    {
        using Manager = STORM::MonteCarloManager<Vector3D, Tessellation3D, Physics>;

    public:
        TypedImplementation(const Tessellation3D &grid,
                            const std::shared_ptr<Physics> &physics,
                            const std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> &populationControl,
                            const std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> &boundaryCondition,
                            const MonteCarloConfig &config,
                            std::unique_ptr<STORM::CommunicationEngine<Vector3D>> engine)
            : manager(grid, physics, populationControl, boundaryCondition, config, std::move(engine)), latestCells(nullptr)
        {
        }

        const std::vector<size_t> &GetCellsStepsCounters(void) const override
        {
            return this->manager.GetCellsStepsCounters();
        }

        std::vector<size_t> &GetCellsStepsCounters(void) override
        {
            return this->manager.GetCellsStepsCounters();
        }

        const std::vector<size_t> &GetCellsParticleCounters(void) const override
        {
            return this->manager.GetCellsParticleCounters();
        }

        size_t GetStartParticleCount(void) const override
        {
            return this->manager.GetStartParticleCount();
        }

        size_t GetInitialParticleCount(void) const override
        {
            return this->manager.GetInitialParticleCount();
        }

        size_t GetPreStepParticleCount(void) const override
        {
            return this->manager.GetPreStepParticleCount();
        }

        size_t GetEndParticleCount(void) const override
        {
            return this->manager.GetEndParticleCount();
        }

        double GetPureComputeTime(void) const override
        {
            return this->manager.GetPureComputeTime();
        }

        const std::vector<size_t> &GetBeginningParticleCount(void) const override
        {
            return this->manager.GetBeginningParticleCount();
        }

        std::vector<size_t> &GetBeginningParticleCount(void) override
        {
            return this->manager.GetBeginningParticleCount();
        }

        size_t GetHandlerMemoryBytes(void) const override
        {
            return this->manager.GetHandlerMemoryBytes();
        }

        const std::vector<MCParticle> &GetParticles(void) const override
        {
            this->SyncCellIDs();
            return this->manager.getParticles();
        }

        std::vector<MCParticle> &GetParticles(void) override
        {
            this->SyncCellIDs();
            return this->manager.getParticles();
        }

        void Step(const std::vector<ComputationalCell3D> &cells, dt_t fullDt) override
        {
            this->latestCells = &cells;
            this->manager.step(fullDt);
        }

    private:
        Manager manager;
        mutable const std::vector<ComputationalCell3D> *latestCells;

        void SyncCellIDs(void) const
        {
            if(this->latestCells == nullptr)
            {
                return;
            }

            const std::vector<MCParticle> &particles = this->manager.getParticles();
            for(MCParticle &particle : const_cast<std::vector<MCParticle> &>(particles))
            {
                assert(particle.cellIndex < this->latestCells->size());
                particle.cellID = (*this->latestCells)[particle.cellIndex].ID;
            }
        }
    };

    std::unique_ptr<Implementation> implementation;

    const Implementation &GetImplementation(void) const
    {
        return *this->implementation;
    }

    Implementation &GetImplementation(void)
    {
        return *this->implementation;
    }
};

template<typename Physics>
MonteCarloManager3D::MonteCarloManager3D(
    const Tessellation3D &grid,
    const std::shared_ptr<Physics> &physics,
    const std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> &populationControl,
    const std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> &boundaryCondition,
    const MonteCarloConfig &config,
    std::unique_ptr<STORM::CommunicationEngine<Vector3D>> engine)
    : implementation(std::make_unique<TypedImplementation<Physics>>(
          grid, physics, populationControl, boundaryCondition, config, std::move(engine)))
{
}

inline const std::vector<size_t> &MonteCarloManager3D::GetCellsStepsCounters(void) const
{
    return this->GetImplementation().GetCellsStepsCounters();
}

inline std::vector<size_t> &MonteCarloManager3D::GetCellsStepsCounters(void)
{
    return this->GetImplementation().GetCellsStepsCounters();
}

inline const std::vector<size_t> &MonteCarloManager3D::GetCellsParticleCounters(void) const
{
    return this->GetImplementation().GetCellsParticleCounters();
}

inline size_t MonteCarloManager3D::GetStartParticleCount(void) const
{
    return this->GetImplementation().GetStartParticleCount();
}

inline size_t MonteCarloManager3D::GetInitialParticleCount(void) const
{
    return this->GetImplementation().GetInitialParticleCount();
}

inline size_t MonteCarloManager3D::GetPreStepParticleCount(void) const
{
    return this->GetImplementation().GetPreStepParticleCount();
}

inline size_t MonteCarloManager3D::GetEndParticleCount(void) const
{
    return this->GetImplementation().GetEndParticleCount();
}

inline double MonteCarloManager3D::GetPureComputeTime(void) const
{
    return this->GetImplementation().GetPureComputeTime();
}

inline const std::vector<size_t> &MonteCarloManager3D::GetBeginningParticleCount(void) const
{
    return this->GetImplementation().GetBeginningParticleCount();
}

inline std::vector<size_t> &MonteCarloManager3D::GetBeginningParticleCount(void)
{
    return this->GetImplementation().GetBeginningParticleCount();
}

inline size_t MonteCarloManager3D::GetHandlerMemoryBytes(void) const
{
    return this->GetImplementation().GetHandlerMemoryBytes();
}

inline const std::vector<MonteCarloManager3D::MCParticle> &MonteCarloManager3D::getParticles(void) const
{
    return this->GetImplementation().GetParticles();
}

inline std::vector<MonteCarloManager3D::MCParticle> &MonteCarloManager3D::getParticles(void)
{
    return this->GetImplementation().GetParticles();
}

inline void MonteCarloManager3D::step(const std::vector<ComputationalCell3D> &cells, dt_t fullDt)
{
    this->GetImplementation().Step(cells, fullDt);
}

#endif
