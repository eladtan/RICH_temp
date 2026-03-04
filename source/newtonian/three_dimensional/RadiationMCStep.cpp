#include "RadiationMCStep.hpp"
#include "3D/radiation/MonteCarloPhysics3D.hpp"
#include "utils/rma/RMAFactory.hpp"

RadiationMCStep::RadiationMCStep(const Tessellation3D &tess,
                                std::vector<ComputationalCell3D> &cells,
                                std::vector<Conserved3D> &extensives,
                                std::shared_ptr<MonteCarloRadiationPhysics3D> physics,
                                std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl,
                                std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond,
                                const std::vector<Particle3D> &particles,
                                bool withHydro
                                #ifdef RICH_MPI
                                    , ManagerType managerType
                                    , std::shared_ptr<CostCalculator3D> cost
                                #endif // RICH_MPI
                            ) : tess(tess), cells(cells), extensives(extensives), particles(particles), popControl(popControl), boundaryCond(boundaryCond), physics(physics), withHydro(withHydro)
                                #ifdef RICH_MPI
                                , managerType(managerType), cost(cost)
                                #endif // RICH_MPI
{
    this->stepCounter = 0;
    #ifdef RICH_MPI
        switch(this->managerType)
        {
            case ManagerType::MPI_RMA:
            case ManagerType::IBV_RDMA:
            {
                RDMA_Type type = (managerType == ManagerType::IBV_RDMA)? RDMA_Type::IBV_RDMA : RDMA_Type::MPI_RMA;
                this->manager = std::make_shared<RDMAMonteCarloManager3D>(tess, physics, popControl, boundaryCond, DEFAULT_BUFFER_SIZE, MPI_COMM_WORLD, type);
                break;
            }
            case ManagerType::P2P:
                this->manager = std::make_shared<TwoSidedMonteCarloManager3D>(tess, physics, popControl, boundaryCond);
                break;
        }
    #else // RICH_MPI
        this->manager = std::make_shared<MonteCarloManagerSerial3D>(tess, physics, popControl, boundaryCond);
    #endif // RICH_MPI
}

std::vector<Particle3D> &RadiationMCStep::getParticles(void)
{
    return this->particles;
}

const std::vector<Particle3D> &RadiationMCStep::getParticles(void) const
{
    return this->particles;
}

double RadiationMCStep::suggestTimeStep(void) const
{
    return std::numeric_limits<double>::max(); // TODO: implement
}

std::string RadiationMCStep::getName(void) const
{
    return "radiation-mc";
}

void RadiationMCStep::step(double dt)
{
    if(this->withHydro)
    {
        // cells location might have changed because of hydro movements
        UpdateNewCells(this->tess, this->particles, this->cells);
    }
    this->stepCounter++;
    this->particles = this->manager->step(this->particles, this->cells, dt);
}

#ifdef RICH_MPI
    ExchangeChain RadiationMCStep::GetExchangeChain(void)
    {
        return ExchangeChain(); // no point movement in radiation MC
    }

    bool RadiationMCStep::allowRebalance(void)
    {
        return (this->stepCounter % 10 == 0) and this->stepCounter != 0;
    }

    std::string RadiationMCStep::getRequiredLB(void) const
    {
        return "radiation-mc";
    }

    std::vector<double> RadiationMCStep::getLoadBalanceWeights(void)
    {
        return (this->cost)? this->cost->CalculateCost(this->tess, this->cells) : std::vector<double>(this->tess.GetPointNo(), 1.0);
    }

    void RadiationMCStep::uponLBChange(void)
    {}

    void RadiationMCStep::beforeLB(void)
    {
        if(this->withHydro)
        {
            // update to current cells first
            UpdateNewCells(this->tess, this->particles, this->cells);
        }
    }

    void RadiationMCStep::afterLB(void)
    {
        UpdateNewCellsAfterExchange(this->tess, this->particles);
    }

#endif // RICH_MPI