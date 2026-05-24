#include "MonteCarloManager3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "3D/radiation/MonteCarloPhysics3D.hpp"

MonteCarloManagerSerial3D::MonteCarloManagerSerial3D(const Tessellation3D &grid, const std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> &physics,
    const std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> &populationControl,
    const std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> &boundaryCondition)
    : MonteCarloManagerSerial<Vector3D, Tessellation3D>(grid, physics, populationControl, boundaryCondition)
{}

std::vector<typename MonteCarloManagerSerial3D::MCParticle> MonteCarloManagerSerial3D::step(std::vector<MCParticle> &&particleList, const std::vector<ComputationalCell3D> &cells, dt_t fullDt)
{
    std::vector<MCParticle> newParticles = MonteCarloManagerSerial<Vector3D, Tessellation3D>::step(std::move(particleList), fullDt);
    for(MCParticle &p : newParticles)
    {
        size_t cellIndex = p.cellIndex;
        assert(cells.size() > cellIndex);
        p.cellID = cells[cellIndex].ID;
    }
    return newParticles;
}

#ifdef RICH_MPI

RDMAMonteCarloManager3D::RDMAMonteCarloManager3D(const Tessellation3D &grid, const std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> &physics,
                                         const std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> &populationControl,
                                         const std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> &boundaryCondition,
                                         size_t bufferSizes, const MPI_Comm &comm, RDMA_Type rdma_type): MonteCarloManager<Vector3D, Tessellation3D>(grid, physics, populationControl, boundaryCondition, bufferSizes, comm, rdma_type)
{
    auto* radPhysics = dynamic_cast<MonteCarloRadiationPhysics3D*>(physics.get());
    if (radPhysics)
        this->progressOpacityPtr_ = radPhysics->getOpacity();
}

std::vector<typename RDMAMonteCarloManager3D::MCParticle> RDMAMonteCarloManager3D::step(std::vector<MCParticle> &&particleList, const std::vector<ComputationalCell3D> &cells, dt_t fullDt)
{
    this->progressCellsPtr_ = &cells;
    std::vector<MCParticle> newParticles = MonteCarloManager<Vector3D, Tessellation3D>::step(std::move(particleList), fullDt);
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

std::vector<typename TwoSidedMonteCarloManager3D::MCParticle> TwoSidedMonteCarloManager3D::step(std::vector<MCParticle> &&particleList, const std::vector<ComputationalCell3D> &cells, dt_t fullDt)
{
    std::vector<MCParticle> newParticles = TwoSidedMonteCarloManager<Vector3D, Tessellation3D>::step(std::move(particleList), fullDt);
    for(MCParticle &p : newParticles)
    {
        size_t cellIndex = p.cellIndex;
        assert(cells.size() > cellIndex);
        p.cellID = cells[cellIndex].ID;
    }
    return newParticles;
}

#endif // RICH_MPI
