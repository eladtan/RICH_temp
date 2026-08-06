#include "RadiationMCStep.hpp"
#include "3D/radiation/MonteCarloPhysics3D.hpp"
#include "3D/radiation/RadiationIMC.hpp"
#include "utils/rma/RMAFactory.hpp"
#include "misc/universal_error.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace
{
    void SyncParticleCellIDs(
        const std::vector<ComputationalCell3D> &cells,
        std::vector<Particle3D> &particles,
        const std::string &where)
    {
        for(Particle3D &p : particles)
        {
            if(p.cellIndex >= cells.size())
            {
                UniversalError eo(where + ": particle cellIndex out of cell range");
                eo.addEntry("Particle", p);
                eo.addEntry("cellIndex", p.cellIndex);
                eo.addEntry("cells.size()", cells.size());
                throw eo;
            }

            p.cellID = cells[p.cellIndex].ID;
        }
    }
}

RadiationMCStep::RadiationMCStep(const Tessellation3D &tess,
                                std::vector<ComputationalCell3D> &cells,
                                std::vector<Conserved3D> &extensives,
                                std::shared_ptr<MonteCarloRadiationPhysics3D> physics,
                                std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl,
                                std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond,
                                const std::vector<Particle3D> &particles,
                                size_t initialParticlesPerCell,
                                bool withHydro
                                #ifdef RICH_MPI
                                    , ManagerType managerType
                                    , std::shared_ptr<CostCalculator3D> cost
                                    , const MonteCarloConfig &monteCarloConfig
                                #endif // RICH_MPI
                            ) : tess(tess), cells(cells), extensives(extensives), particles(particles), popControl(popControl), boundaryCond(boundaryCond), physics(physics), withHydro(withHydro)
                                #ifdef RICH_MPI
                                , managerType(managerType), cost(cost)
                                #endif // RICH_MPI
{
    this->stepCounter = 0;
    this->suggested_dt = std::numeric_limits<double>::max();
    #ifdef RICH_MPI
        switch(this->managerType)
        {
            case ManagerType::LEGACY_MPI_RMA:
            case ManagerType::LEGACY_IBV_RDMA:
            case ManagerType::LEGACY_AUTO_RDMA:
            {
                RDMA_Type type;
                if(managerType == ManagerType::LEGACY_IBV_RDMA)
                    type = RDMA_Type::IBV_RDMA;
                else if(managerType == ManagerType::LEGACY_MPI_RMA)
                    type = RDMA_Type::MPI_RMA;
                else
                    type = RDMA_Type::AUTO_RDMA;
                this->manager = std::make_shared<RDMAMonteCarloManagerLegacy3D>(tess, physics, popControl, boundaryCond, monteCarloConfig, MPI_COMM_WORLD, type);
                break;
            }
            case ManagerType::P2P:
                this->manager = std::make_shared<TwoSidedMonteCarloManager3D>(tess, physics, popControl, boundaryCond);
                break;
            case ManagerType::RDMA:
                this->manager = std::make_shared<RDMAMonteCarloManager3D>(tess, physics, popControl, boundaryCond, monteCarloConfig, MPI_COMM_WORLD, RDMA_Type::AUTO_RDMA);
                break;
            case ManagerType::RDMA_IBV:
                this->manager = std::make_shared<RDMAMonteCarloManager3D>(tess, physics, popControl, boundaryCond, monteCarloConfig, MPI_COMM_WORLD, RDMA_Type::IBV_RDMA);
                break;
        }
    #else // RICH_MPI
        this->manager = std::make_shared<MonteCarloManagerSerial3D>(tess, physics, popControl, boundaryCond);
    #endif // RICH_MPI

    if(this->particles.empty())
    {
        this->particles = this->physics->generateInitialParticles(initialParticlesPerCell);
        int rank = 0;
        #ifdef RICH_MPI
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        #endif
        if(rank == 0)
            std::cout << "RadiationMCStep: generated " << this->particles.size() << " initial photon particles" << std::endl;
    }
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
    return suggested_dt;
}

void RadiationMCStep::step(double dt)
{
    this->stepCounter++;
    auto radiationStepStart = std::chrono::high_resolution_clock::now();

    size_t N = tess.GetPointNo();
    if(cells.size() < N)
    {
        UniversalError eo("RadiationMCStep: cells.size() < tess.GetPointNo()");
        eo.addEntry("cells.size()", cells.size());
        eo.addEntry("tess.GetPointNo()", N);
        throw eo;
    }

    std::vector<double> old_Erad(N), old_temperature(N);
    for(size_t i = 0; i < N; ++i)
    {
        old_Erad[i] = cells[i].Erad * cells[i].density;
        old_temperature[i] = cells[i].temperature;
    }

    auto preManagerStart = std::chrono::high_resolution_clock::now();
    if(this->withHydro)
    {
        UpdateNewCells(this->tess, this->particles, this->cells);
    }
    // it's adjustExistingParticles's reponsibility to call UNC inside, if needed
    this->physics->adjustExistingParticles(this->particles, dt);
    // if(this->withHydro)
    // {
    //     UpdateNewCells(this->tess, this->particles, this->cells);
    // }
    double preManagerTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - preManagerStart).count();

    auto managerStart = std::chrono::high_resolution_clock::now();
    this->particles = this->manager->step(std::move(this->particles), this->cells, dt);
    SyncParticleCellIDs(this->cells, this->particles,
                        "RadiationMCStep::step after manager");
    double managerTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - managerStart).count();

    int rank = 0;
    #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif

    unsigned long long initialParticles = static_cast<unsigned long long>(this->manager->GetInitialParticleCount());
    unsigned long long preStepParticles = static_cast<unsigned long long>(this->manager->GetPreStepParticleCount());
    unsigned long long activeAfterPreStepParticles = static_cast<unsigned long long>(this->manager->GetStartParticleCount());
    unsigned long long censusParticles = static_cast<unsigned long long>(this->manager->GetEndParticleCount());
    #ifdef RICH_MPI
        MPI_Reduce((rank == 0) ? MPI_IN_PLACE : &initialParticles, &initialParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce((rank == 0) ? MPI_IN_PLACE : &preStepParticles, &preStepParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce((rank == 0) ? MPI_IN_PLACE : &activeAfterPreStepParticles, &activeAfterPreStepParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce((rank == 0) ? MPI_IN_PLACE : &censusParticles, &censusParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    #endif
    if(rank == 0)
    {
        std::cout << "MC particle counts:"
                  << " initial=" << initialParticles
                  << " prestep_generated=" << preStepParticles
                  << " active_after_prestep=" << activeAfterPreStepParticles
                  << " census=" << censusParticles
                  << std::endl;
    }

    auto postManagerStart = std::chrono::high_resolution_clock::now();

    double reductionArray[2] = {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()};

    double &max_Erad = reductionArray[0];
    max_Erad = N > 0 ? *std::max_element(old_Erad.begin(), old_Erad.end()) : std::numeric_limits<double>::lowest();
    double &max_temperature = reductionArray[1];
    max_temperature = N > 0 ? *std::max_element(old_temperature.begin(), old_temperature.end()) : std::numeric_limits<double>::lowest();

    #ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, reductionArray, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    #endif

    double max_Erad_diff = 0.0;
    double max_temperature_diff = 0.0;
    int max_Erad_loc = 0, max_temperature_loc = 0;
    for(size_t i = 0; i < N; ++i)
    {
        double const er_new = cells[i].Erad * cells[i].density;
        double Erad_diff = std::abs(er_new - old_Erad[i])
            / (er_new + 0.02 * max_Erad + 1e-30);
        double temperature_diff = std::abs(cells[i].temperature - old_temperature[i])
            / (cells[i].temperature + 0.02 * max_temperature + 1e-30);
        if(Erad_diff > max_Erad_diff)
        {
            max_Erad_diff = Erad_diff;
            max_Erad_loc = i;
        }
        if(temperature_diff > max_temperature_diff)
        {
            max_temperature_diff = temperature_diff;
            max_temperature_loc = i;
        }
    }
    max_Erad_diff *= 0.5;

    double max_diff = max_temperature_diff; // TODO: change later
    // double max_diff = std::max(max_Erad_diff, max_temperature_diff);
    constexpr double min_rel_diff = 1e-12;
    max_diff = std::max(max_diff, min_rel_diff);
    #ifdef RICH_MPI
        rank_t max_diff_rank = rank;
        std::tie(max_diff_rank, max_diff) = MPI_Max_loc(max_diff, MPI_COMM_WORLD);
        max_diff = std::max(max_diff, min_rel_diff);
    #endif // RICH_MPI

    std::cout.flush();
    #ifdef RICH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == max_diff_rank and N > 0)
    #endif // RICH_MPI
    {
        size_t max_loc = max_temperature_loc; // TODO: (max_Erad_diff > max_temperature_diff) ? max_Erad_loc : max_temperature_loc;
        double fleckFactor = std::numeric_limits<double>::quiet_NaN();
        if(const auto *imc = dynamic_cast<const ::RadiationIMC *>(this->physics.get()); imc != nullptr)
        {
            const auto &fleck = imc->getFactorFleck();
            if(max_loc < fleck.size())
                fleckFactor = fleck[max_loc];
        }
        std::cout << "MC Radiation time step ID " << cells[max_loc].ID
            << " old temperature " << old_temperature[max_loc] << " new temperature " << cells[max_loc].temperature
			<< " old Erad " << old_Erad[max_loc] << " new Erad "
			<< cells[max_loc].Erad * cells[max_loc].density
            << " diff " << max_diff << " Tgas " << cells[max_loc].temperature
            << " max_Erad " << max_Erad << " max_temperature " << max_temperature << " rank " << rank
            << " density " << cells[max_loc].density
            << " width " << tess.GetWidth(max_loc)
            << " fleck " << fleckFactor
            << " location " << tess.GetMeshPoint(max_loc) << std::endl;
        std::cout<<cells[max_loc]<<std::endl;
        std::cout << "Next MC time step is " << dt * std::min(1.25, 0.15 / max_diff) << std::endl;
    }
    std::cout.flush();
    #ifdef RICH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
    #endif // RICH_MPI

    double postManagerTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - postManagerStart).count();
    double radiationStepTotal = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - radiationStepStart).count();

    #ifdef RICH_MPI
        double reductionArray2[4] = {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()};
        reductionArray2[0] = preManagerTime;
        reductionArray2 [1] = managerTime;
        reductionArray2[2] = postManagerTime;
        reductionArray2[3] = radiationStepTotal;
        MPI_Reduce((rank == 0) ? MPI_IN_PLACE : reductionArray2, reductionArray2, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    #endif
    if(rank == 0)
    {
        std::cout << "RadiationMCStep breakdown (max): preManager=" << preManagerTime
                  << "s, manager=" << managerTime << "s, postManager=" << postManagerTime
                  << "s, total=" << radiationStepTotal << "s" << std::endl;
    }

    this->suggested_dt = dt * std::min(1.25, 0.15 / max_diff);
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
        SyncParticleCellIDs(this->cells, this->particles,
                            "RadiationMCStep::afterLB");
    }

    void RadiationMCStep::dumpCost(size_t cycle) const
    {
        if(this->cost)
            this->cost->Dump(cycle);
    }

#endif // RICH_MPI
