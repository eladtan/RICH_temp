#ifndef MONTECARLO_MANAGER_3D_HPP
#define MONTECARLO_MANAGER_3D_HPP

#ifdef RICH_MPI
#include "monte/manager/MonteCarloManager.hpp"
#include "monte/two_sided_manager/TwoSidedMonteCarloManager.hpp"
#endif // RICH_MPI
#include "monte/serial_manager/MonteCarloManagerSerial.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

class MonteCarloManager3D
{
    using MCParticle = MonteCarloParticle<Vector3D, Tessellation3D>;

public:
    virtual ~MonteCarloManager3D() = default;

    virtual const std::vector<size_t> &GetCellsStepsCounters(void) const = 0;

    virtual std::vector<size_t> &GetCellsStepsCounters(void) = 0;

    virtual size_t GetStartParticleCount(void) const = 0;

    virtual size_t GetInitialParticleCount(void) const = 0;

    virtual size_t GetPreStepParticleCount(void) const = 0;

    virtual size_t GetEndParticleCount(void) const = 0;

    virtual size_t GetHandlerMemoryBytes(void) const = 0;

    virtual std::vector<MCParticle> step(std::vector<MCParticle> &&particleList, const std::vector<ComputationalCell3D> &cells, dt_t fullDt) = 0;
};

class MonteCarloManagerSerial3D : public MonteCarloManagerSerial<Vector3D, Tessellation3D>, public MonteCarloManager3D
{
    using MCParticle = MonteCarloParticle<Vector3D, Tessellation3D>;

public:
    MonteCarloManagerSerial3D(const Tessellation3D &grid, const std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> &physics,
                        const std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> &populationControl,
                        const std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> &boundaryCondition);

    inline const std::vector<size_t> &GetCellsStepsCounters(void) const override{return MonteCarloManagerSerial<Vector3D, Tessellation3D>::GetCellsStepsCounters();};

    inline std::vector<size_t> &GetCellsStepsCounters(void) override{return MonteCarloManagerSerial<Vector3D, Tessellation3D>::GetCellsStepsCounters();};

    inline size_t GetStartParticleCount(void) const override{return MonteCarloManagerSerial<Vector3D, Tessellation3D>::GetStartParticleCount();};

    inline size_t GetInitialParticleCount(void) const override{return MonteCarloManagerSerial<Vector3D, Tessellation3D>::GetInitialParticleCount();};

    inline size_t GetPreStepParticleCount(void) const override{return MonteCarloManagerSerial<Vector3D, Tessellation3D>::GetPreStepParticleCount();};

    inline size_t GetEndParticleCount(void) const override{return MonteCarloManagerSerial<Vector3D, Tessellation3D>::GetEndParticleCount();};

    inline size_t GetHandlerMemoryBytes(void) const override{return MonteCarloManagerSerial<Vector3D, Tessellation3D>::GetHandlerMemoryBytes();};

    std::vector<MCParticle> step(std::vector<MCParticle> &&particleList, const std::vector<ComputationalCell3D> &cells, dt_t fullDt) override;
};

#ifdef RICH_MPI

class RDMAMonteCarloManager3D : public MonteCarloManager<Vector3D, Tessellation3D>, public MonteCarloManager3D
{
    using MCParticle = MonteCarloParticle<Vector3D, Tessellation3D>;

public:
    RDMAMonteCarloManager3D(const Tessellation3D &grid, const std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> &physics,
                        const std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> &populationControl,
                        const std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> &boundaryCondition,
                        size_t bufferSizes = DEFAULT_BUFFER_SIZE,
                        const MPI_Comm &comm = MPI_COMM_WORLD,
                        RDMA_Type rdma_type = RDMA_Type::AUTO_RDMA);

    inline const std::vector<size_t> &GetCellsStepsCounters(void) const override{return MonteCarloManager<Vector3D, Tessellation3D>::GetCellsStepsCounters();};

    inline std::vector<size_t> &GetCellsStepsCounters(void) override{return MonteCarloManager<Vector3D, Tessellation3D>::GetCellsStepsCounters();};

    inline size_t GetStartParticleCount(void) const override{return MonteCarloManager<Vector3D, Tessellation3D>::GetStartParticleCount();};

    inline size_t GetInitialParticleCount(void) const override{return MonteCarloManager<Vector3D, Tessellation3D>::GetInitialParticleCount();};

    inline size_t GetPreStepParticleCount(void) const override{return MonteCarloManager<Vector3D, Tessellation3D>::GetPreStepParticleCount();};

    inline size_t GetEndParticleCount(void) const override{return MonteCarloManager<Vector3D, Tessellation3D>::GetEndParticleCount();};

    inline size_t GetHandlerMemoryBytes(void) const override{return MonteCarloManager<Vector3D, Tessellation3D>::GetHandlerMemoryBytes();};

    std::vector<MCParticle> step(std::vector<MCParticle> &&particleList, const std::vector<ComputationalCell3D> &cells, dt_t fullDt) override;
};


class TwoSidedMonteCarloManager3D : public TwoSidedMonteCarloManager<Vector3D, Tessellation3D>, public MonteCarloManager3D
{
    using MCParticle = MonteCarloParticle<Vector3D, Tessellation3D>;

public:
    TwoSidedMonteCarloManager3D(const Tessellation3D &grid, const std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> &physics,
                                const std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> &populationControl,
                                const std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> &boundaryCondition,
                                const MPI_Comm &comm = MPI_COMM_WORLD);

    inline const std::vector<size_t> &GetCellsStepsCounters(void) const override{return TwoSidedMonteCarloManager<Vector3D, Tessellation3D>::GetCellsStepsCounters();};

    inline std::vector<size_t> &GetCellsStepsCounters(void) override{return TwoSidedMonteCarloManager<Vector3D, Tessellation3D>::GetCellsStepsCounters();};

    inline size_t GetStartParticleCount(void) const override{return TwoSidedMonteCarloManager<Vector3D, Tessellation3D>::GetStartParticleCount();};

    inline size_t GetInitialParticleCount(void) const override{return TwoSidedMonteCarloManager<Vector3D, Tessellation3D>::GetInitialParticleCount();};

    inline size_t GetPreStepParticleCount(void) const override{return TwoSidedMonteCarloManager<Vector3D, Tessellation3D>::GetPreStepParticleCount();};

    inline size_t GetEndParticleCount(void) const override{return TwoSidedMonteCarloManager<Vector3D, Tessellation3D>::GetEndParticleCount();};

    inline size_t GetHandlerMemoryBytes(void) const override{return TwoSidedMonteCarloManager<Vector3D, Tessellation3D>::GetHandlerMemoryBytes();};

    std::vector<MCParticle> step(std::vector<MCParticle> &&particleList, const std::vector<ComputationalCell3D> &cells, dt_t fullDt) override;
};

#endif // RICH_MPI

#endif // MONTECARLO_MANAGER_3D_HPP
