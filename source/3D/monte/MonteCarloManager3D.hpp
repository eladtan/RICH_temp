#ifndef MONTECARLO_MANAGER_3D_HPP
#define MONTECARLO_MANAGER_3D_HPP

#ifdef RICH_MPI

#include "monte/manager/MonteCarloManager.hpp"
#include "monte/two_sided_manager/TwoSidedMonteCarloManager.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

class MonteCarloManager3D
{
    using MCParticle = MonteCarloParticle<Vector3D, Tessellation3D>;

public:
    virtual ~MonteCarloManager3D() = default;

    virtual const std::vector<size_t> &GetCellsStepsCounters(void) const = 0;

    virtual std::vector<MCParticle> step(const std::vector<MCParticle> &particleList, const std::vector<ComputationalCell3D> &cells, dt_t fullDt) = 0;
};

class RDMAMonteCarloManager3D : public MonteCarloManager<Vector3D, Tessellation3D>, public MonteCarloManager3D
{
    using MCParticle = MonteCarloParticle<Vector3D, Tessellation3D>;

public:
    RDMAMonteCarloManager3D(const Tessellation3D &grid, const std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> &physics,
                        const std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> &populationControl,
                        const std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> &boundaryCondition,
                        size_t bufferSizes = DEFAULT_BUFFER_SIZE,
                        const MPI_Comm &comm = MPI_COMM_WORLD);

    inline const std::vector<size_t> &GetCellsStepsCounters(void) const{return MonteCarloManager<Vector3D, Tessellation3D>::GetCellsStepsCounters();};

    std::vector<MCParticle> step(const std::vector<MCParticle> &particleList, const std::vector<ComputationalCell3D> &cells, dt_t fullDt) override;
};


class TwoSidedMonteCarloManager3D : public TwoSidedMonteCarloManager<Vector3D, Tessellation3D>, public MonteCarloManager3D
{
    using MCParticle = MonteCarloParticle<Vector3D, Tessellation3D>;

public:
    TwoSidedMonteCarloManager3D(const Tessellation3D &grid, const std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> &physics,
                                const std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> &populationControl,
                                const std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> &boundaryCondition,
                                const MPI_Comm &comm = MPI_COMM_WORLD);

    inline const std::vector<size_t> &GetCellsStepsCounters(void) const{return TwoSidedMonteCarloManager<Vector3D, Tessellation3D>::GetCellsStepsCounters();};

    std::vector<MCParticle> step(const std::vector<MCParticle> &particleList, const std::vector<ComputationalCell3D> &cells, dt_t fullDt) override;
};

#endif // RICH_MPI

#endif // MONTECARLO_MANAGER_3D_HPP