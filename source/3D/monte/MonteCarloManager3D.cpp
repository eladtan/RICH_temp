#ifdef RICH_MPI

#include "MonteCarloManager3D.hpp"

RDMAMonteCarloManager3D::RDMAMonteCarloManager3D(const Tessellation3D &grid, const std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> &physics,
                                         const std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> &populationControl,
                                         const std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> &boundaryCondition,
                                         size_t bufferSizes, const MPI_Comm &comm): MonteCarloManager<Vector3D, Tessellation3D>(grid, physics, populationControl, boundaryCondition, bufferSizes, comm)
{}

std::vector<typename RDMAMonteCarloManager3D::MCParticle> RDMAMonteCarloManager3D::step(const std::vector<MCParticle> &particleList, const std::vector<ComputationalCell3D> &cells, dt_t fullDt)
{
    std::vector<MCParticle> newParticles = MonteCarloManager<Vector3D, Tessellation3D>::step(particleList, fullDt);
    for(MCParticle &p : newParticles)
    {
        size_t cellIndex = p.cellIndex;
        assert(cells.size() > cellIndex);
        p.cellID = cells[cellIndex].ID;
    }
    return newParticles;
}

TwoSidedMonteCarloManager3D::TwoSidedMonteCarloManager3D(const Tessellation3D &grid, const std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> &physics,
                                                         const std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> &populationControl,
                                                         const std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> &boundaryCondition,
                                                         const MPI_Comm &comm): TwoSidedMonteCarloManager<Vector3D, Tessellation3D>(grid, physics, populationControl, boundaryCondition, comm)
{}

std::vector<typename TwoSidedMonteCarloManager3D::MCParticle> TwoSidedMonteCarloManager3D::step(const std::vector<MCParticle> &particleList, const std::vector<ComputationalCell3D> &cells, dt_t fullDt)
{
    std::vector<MCParticle> newParticles = TwoSidedMonteCarloManager<Vector3D, Tessellation3D>::step(particleList, fullDt);
    for(MCParticle &p : newParticles)
    {
        size_t cellIndex = p.cellIndex;
        assert(cells.size() > cellIndex);
        p.cellID = cells[cellIndex].ID;
    }
    return newParticles;
}

#endif // RICH_MPI