#include <mpi.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include "mpi/mpi_commands.hpp"
#include "mpi/mpi_commands_3d.hpp"
#include "misc/mesh_generator3D.hpp"
#include "3D/GeometryCommon/RoundGrid3D.hpp"
#include "3D/tessellation/Voronoi3D.hpp"
#include "monte/utils/RandomInCell.hpp"
#include "CMMC/src/units/units.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "3D/radiation/MonteCarloPhysics3D.hpp"
#include "3D/monte/MonteCarloManager3D.hpp"
#include "3D/monte/STORMVoronoi3DMovement.hpp"
#include "monte/manager/communication/P2PCommunicationEngine.hpp"
#include "monte/manager/communication/RDMACommunicationEngine.hpp"
#include "monte/population/PopulationControl.hpp"
#include "monte/boundary/BoundaryCondition.hpp"
#include "monte/boundary/RigidBoundary.hpp"
#include "utils/arguments/ArgumentParser.hpp"
#include "utils/debug/vtune.h"

using Particle3D = MonteCarloParticle<Vector3D>;

class TransparentOpacity : public OpacityCalculator
{
public:
    double CalcPlanckOpacity(const ComputationalCell3D&) const override
    {
        return 0.0;
    }

    double CalcScatteringOpacity(const ComputationalCell3D&) const override
    {
        return 0.0;
    }

    double CalcDiffusionCoefficient(const ComputationalCell3D&) const override
    {
        return std::numeric_limits<double>::infinity();
    }

    double CalcAbsorptionOpacity(const ComputationalCell3D&, double) const override
    {
        return 0.0;
    }

    double CalcScatteringOpacity(const ComputationalCell3D&, double) const override
    {
        return 0.0;
    }

    double CalcDiffusionCoefficient(const ComputationalCell3D&, double) const override
    {
        return std::numeric_limits<double>::infinity();
    }
};

template<typename T, typename Grid>
class IdentityPopulationControl : public PopulationControl<T, Grid>
{
public:
    using Particle = MonteCarloParticle<T>;

    explicit IdentityPopulationControl(const Grid &grid)
        : PopulationControl<T, Grid>(grid)
    {}

    std::vector<Particle> activate(const std::vector<Particle> &particles) override
    {
        return particles;
    }
};

class SpaceEmissionPhysics : public MonteCarloRadiationPhysics3D
{
public:
    SpaceEmissionPhysics(Tessellation3D &grid,
                         const std::shared_ptr<BoundaryCond> &boundary,
                         std::vector<ComputationalCell3D> &cells,
                         std::vector<Conserved3D> &conserved,
                         const std::shared_ptr<EquationOfState> &eos,
                         const std::shared_ptr<OpacityCalculator> &opacity,
                         size_t photonsPerCell,
                         double particleWeight,
                         uint64_t seed)
        : MonteCarloRadiationPhysics3D(grid, boundary, cells, conserved, eos, opacity),
          photonsPerCell(photonsPerCell),
          particleWeight(particleWeight),
          lastEmittedCount(0),
          cycle(0),
          rng(seed),
          dist(0.0, 1.0)
    {}

    std::vector<Particle> preStep(double fullDt) override
    {
        std::vector<Particle> particles;
        const size_t emitterCells = countEmitterCells();
        particles.reserve(emitterCells * photonsPerCell);

        const size_t N = this->grid.GetPointNo();
        for(size_t i = 0; i < N; i++)
        {
            for(size_t j = 0; j < photonsPerCell; j++)
            {
                Particle particle = generateSingleParticle(i, this->cells[i]);
                particle.timeLeft = fullDt;
                particles.push_back(particle);
            }
        }

        lastEmittedCount = particles.size();
        cycle++;
        return particles;
    }

    Functionality step(Particle &particle, std::vector<Particle> &particlesToAdd) override
    {
        (void) particlesToAdd;

        Functionality functionality;
        auto details = this->getIntersectionDetails(particle);
        const dt_t timeIntersect = std::get<1>(details);
        const size_t nextCellIndex = std::get<2>(details);
        const dt_t timeLeftBefore = particle.timeLeft;
        const dt_t dt = std::min(timeIntersect, timeLeftBefore);

        if(dt < 0)
        {
            UniversalError eo("Negative time step in SpaceEmissionPhysics::step");
            eo.addEntry("timeIntersect", timeIntersect);
            eo.addEntry("timeLeft", timeLeftBefore);
            eo.addEntry("Particle", particle);
            throw eo;
        }

        particle.location += particle.velocity * dt;
        particle.timeLeft -= dt;

        if(timeIntersect <= timeLeftBefore)
        {
            functionality.change = MonteCarloParticleStatus::CELL_MOVE;
            functionality.nextCellIndex = nextCellIndex;
        }
        else
        {
            functionality.change = MonteCarloParticleStatus::DONE;
        }

        return functionality;
    }

    void postStep(const std::vector<Particle>&, double) override
    {}

    Particle generateSingleParticle(size_t cellIndex, const ComputationalCell3D &cell) const override
    {
        (void) cell;

        Particle particle;
        particle.id = std::numeric_limits<size_t>::max();
        particle.cellIndex = cellIndex;
        particle.cellID = this->cells[cellIndex].ID;
        particle.frequency = 1.0;
        particle.weight = particleWeight;
        particle.initialWeight = particleWeight;
        particle.location = RandomPointInCell(this->grid, cellIndex);

        static constexpr double nudge = 1e-10;
        particle.location = particle.location * (1.0 - nudge) + nudge * this->grid.GetMeshPoint(cellIndex);

        Vector3D direction = randomUnitVector();

        particle.velocity = direction * units::clight;
        particle.timeLeft = 0.0;
        return particle;
    }

    size_t countEmitterCells() const
    {
        return this->grid.GetPointNo();
    }

    size_t getLastEmittedCount() const
    {
        return lastEmittedCount;
    }

private:
    Vector3D randomUnitVector() const
    {
        const double z = 2.0 * dist(rng) - 1.0;
        const double phi = 2.0 * M_PI * dist(rng);
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        return Vector3D(r * std::cos(phi), r * std::sin(phi), z);
    }

    size_t photonsPerCell;
    double particleWeight;
    size_t lastEmittedCount;
    size_t cycle;
    mutable std::mt19937_64 rng;
    mutable std::uniform_real_distribution<double> dist;
};

static size_t GlobalSum(size_t local)
{
    unsigned long long localValue = static_cast<unsigned long long>(local);
    unsigned long long globalValue = 0;
    MPI_Allreduce(&localValue, &globalValue, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return static_cast<size_t>(globalValue);
}

static double GlobalMax(double local)
{
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return global;
}

static std::vector<double> BuildCellStepWeights(const Tessellation3D &tess,
                                                const std::vector<size_t> &cellSteps)
{
    const size_t N = tess.GetPointNo();
    if(cellSteps.size() != N)
    {
        UniversalError eo("BuildCellStepWeights: cell step counters do not match the mesh");
        eo.addEntry("cellSteps.size()", cellSteps.size());
        eo.addEntry("mesh cells", N);
        throw eo;
    }

    std::vector<double> weights(N);
    for(size_t i = 0; i < N; i++)
        weights[i] = static_cast<double>(cellSteps[i]);
    return weights;
}

int main(int argc, char *argv[])
{
    vtune_stop();
    DISABLE_TIMERS();

    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

    rank_t rank, ws;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

    try
    {
        ArgumentParser arguments("Space emission benchmark");
        arguments.addPositional<size_t>("N_base", 20000, "number of background mesh points");
        arguments.addPositional<size_t>("photons_per_cell", 10, "photons emitted from each cell per cycle");
        arguments.addPositional<size_t>("steps", 20, "number of benchmark cycles");

        arguments.addOption<std::string>("manager", "new-rdma-auto", "Monte Carlo communication manager")
            .choices({"new-rdma-auto", "new-rdma-ibv", "new-rdma-mpi-rma",
                      "old-rdma-auto", "old-rdma-ibv", "old-rdma-mpi-rma", "p2p"})
            .flagAlias("p2p", "p2p")
            .flagAlias("new-rdma", "new-rdma-auto")
            .flagAlias("new-ibv", "new-rdma-ibv")
            .flagAlias("new_ibv", "new-rdma-ibv")
            .flagAlias("new-mpi-rma", "new-rdma-mpi-rma")
            .flagAlias("rdma", "old-rdma-auto")
            .flagAlias("old-rdma", "old-rdma-auto")
            .flagAlias("ibv", "old-rdma-ibv")
            .flagAlias("mpi-rma", "old-rdma-mpi-rma");

        arguments.addOption<double>("dt", 1e-10, "step size in seconds");
        arguments.addOption<double>("domain-size", 10.0, "side length of the cubic domain");
        arguments.addOption<double>("weight", 1.0, "particle weight");
        arguments.addOption<std::string>("profiling-dir", "", "directory for per-rank cell profiling output");

        try
        {
            if(!arguments.parse(argc, argv))
            {
                if(rank == 0)
                    std::cout << arguments.help() << std::endl;
                MPI_Finalize();
                return 0;
            }
        }
        catch(const std::exception &e)
        {
            if(rank == 0)
            {
                std::cerr << e.what() << std::endl;
                std::cerr << arguments.help() << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        size_t N_base = arguments.get<size_t>("N_base");
        size_t photonsPerCell = arguments.get<size_t>("photons_per_cell");
        size_t steps = arguments.get<size_t>("steps");
        double domainSize = arguments.get<double>("domain-size");
        double dt = arguments.get<double>("dt");
        double particleWeight = arguments.get<double>("weight");
        std::string profilingDir = arguments.get<std::string>("profiling-dir");

        enum ManagerKind
        {
            MANAGER_NEW_RDMA,
            MANAGER_OLD_RDMA,
            MANAGER_P2P
        };
        std::string managerName = arguments.get<std::string>("manager");
        ManagerKind managerKind = managerName == "p2p" ? MANAGER_P2P :
                                  managerName.find("old-rdma") == 0 ? MANAGER_OLD_RDMA :
                                  MANAGER_NEW_RDMA;

        #ifdef RICH_MPI
        RDMA_Type rdmaType = RDMA_Type::AUTO_RDMA;
        if(managerName.find("ibv") != std::string::npos)
            rdmaType = RDMA_Type::IBV_RDMA;
        else if(managerName.find("mpi-rma") != std::string::npos)
            rdmaType = RDMA_Type::MPI_RMA;
        #endif

        const double halfDomain = 0.5 * domainSize;
        const Vector3D ll(-halfDomain, -halfDomain, -halfDomain);
        const Vector3D ur(halfDomain, halfDomain, halfDomain);
        const Vector3D center(0.0, 0.0, 0.0);

        std::vector<Vector3D> points;
        if(rank == 0)
        {
            const size_t backgroundPoints = N_base;

            points = RandRectangular(backgroundPoints, ll, ur);

            if(static_cast<rank_t>(points.size()) < ws)
            {
                std::cerr << "ERROR: only " << points.size() << " mesh points for "
                          << ws << " MPI ranks. Increase N_base." << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            std::cout << "Generated " << points.size() << " mesh points"
                      << " (background=" << backgroundPoints << ")"
                      << std::endl;
        }

        points = MPI_Spread(points, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        points = RoundGrid3D(points, ll, ur, 5);

        Voronoi3D tess(ll, ur);
        tess.BuildParallel(points);

        IdealGas eos(5.0 / 3.0, 1.0, 1.0, 0.0);
        ComputationalCell3D initCell;
        initCell.density = 1.0;
        initCell.temperature = 1.0;
        initCell.velocity = Vector3D(0.0, 0.0, 0.0);
        initCell.internal_energy = eos.dT2e(initCell.density, initCell.temperature,
                                            initCell.tracers, ComputationalCell3D::tracerNames);
        initCell.pressure = eos.de2p(initCell.density, initCell.internal_energy,
                                     initCell.tracers, ComputationalCell3D::tracerNames);

        const size_t initialLocalCells = tess.GetPointNo();
        std::vector<ComputationalCell3D> cells(initialLocalCells, initCell);
        std::vector<Conserved3D> extensives(initialLocalCells);
        for(size_t i = 0; i < initialLocalCells; i++)
        {
            cells[i].ID = i;
            PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);
        }

        auto eosPtr = std::make_shared<IdealGas>(eos);
        auto opacity = std::make_shared<TransparentOpacity>();
        auto boundary = std::make_shared<RigidBoundaryCondition<Vector3D, Tessellation3D>>(tess);

        auto physics = std::make_shared<SpaceEmissionPhysics>(
            tess, boundary, cells, extensives, eosPtr, opacity,
            photonsPerCell, particleWeight,
            static_cast<uint64_t>(12345 + 7919 * rank));
        auto popControl = std::make_shared<IdentityPopulationControl<Vector3D, Tessellation3D>>(tess);

        const size_t localEmitterCells = physics->countEmitterCells();
        const size_t globalEmitterCells = GlobalSum(localEmitterCells);
        if(globalEmitterCells == 0)
        {
            if(rank == 0)
                std::cerr << "ERROR: no emitter cells found. Increase N_base." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MonteCarloConfig config;
        config.holdSmallIdleFlushes = true;
        config.transferDiagnosticsLevel = MonteCarloTransferDiagnosticsLevel::Off;
        config.initialBufferSize = std::max<size_t>(5000, globalEmitterCells * photonsPerCell / std::max<rank_t>(1, ws));
        config.minimalBuffSize = std::max<size_t>(50, config.initialBufferSize / 10);

        std::shared_ptr<MonteCarloManager3D> manager;
        #ifdef RICH_MPI
        std::unique_ptr<STORM::CommunicationEngine<Vector3D>> engine;
        if(managerKind == MANAGER_P2P)
        {
            engine = std::make_unique<STORM::P2PCommunicationEngine<Vector3D, Tessellation3D>>(
                tess, config, MPI_COMM_WORLD);
        }
        else
        {
            engine = std::make_unique<STORM::RDMACommunicationEngine<Vector3D, Tessellation3D>>(
                tess, config, MPI_COMM_WORLD, rdmaType);
        }
        manager = std::make_shared<MonteCarloManager3D>(
            tess, physics, popControl, boundary, config, std::move(engine));
        #else
        (void) managerKind;
        managerName = "serial";
        manager = std::make_shared<MonteCarloManager3D>(tess, physics, popControl, boundary, config);
        #endif

        if(rank == 0)
        {
            std::cout << "Space emission benchmark:"
                      << " manager=" << managerName
                      << ", ranks=" << ws
                      << ", local cells(rank0)=" << initialLocalCells
                      << ", emitter cells=" << globalEmitterCells
                      << ", photons/cell/cycle=" << photonsPerCell
                      << ", emitted/cycle=" << globalEmitterCells * photonsPerCell
                      << ", boundary=rigid"
                      << ", direction=isotropic"
                      << ", domain=[" << -halfDomain << "," << halfDomain << "]^3"
                      << ", dt=" << dt
                      << ", steps=" << steps
                      << std::endl;
        }

        manager->getParticles().clear();
        double simTime = 0.0;
        double totalStepWall = 0.0;

        for(size_t cycle = 0; cycle < steps; cycle++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto stepStart = std::chrono::high_resolution_clock::now();
            manager->step(cells, dt);
            auto stepEnd = std::chrono::high_resolution_clock::now();

            const double localStepWall = std::chrono::duration<double>(stepEnd - stepStart).count();
            const double stepWall = GlobalMax(localStepWall);
            totalStepWall += stepWall;
            simTime += dt;

            const size_t emitted = GlobalSum(physics->getLastEmittedCount());
            const size_t remaining = GlobalSum(
                static_cast<const MonteCarloManager3D &>(
                    *manager).getParticles().size());

            if(rank == 0)
            {
                std::cout << "Cycle " << (cycle + 1)
                          << "  t=" << simTime
                          << "  emitted=" << emitted
                          << "  remaining=" << remaining
                          << "  step_wall(max)=" << stepWall << "s"
                          << std::endl;
            }
        }

        if(rank == 0)
        {
            std::cout << "Total benchmark step wall time(max-summed): "
                      << totalStepWall << "s" << std::endl;
        }

        if(!profilingDir.empty())
        {
            if(rank == 0)
            {
                const int mkdirStatus = mkdir(profilingDir.c_str(), 0777);
                if(mkdirStatus != 0 && errno != EEXIST)
                {
                    std::cerr << "ERROR: failed to create profiling directory "
                              << profilingDir << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);

            const char lastChar = profilingDir[profilingDir.size() - 1];
            const std::string profileFileName = profilingDir +
                (lastChar == '/' ? "" : "/") + std::to_string(rank);
            std::ofstream profileFile(profileFileName.c_str());
            if(!profileFile)
            {
                std::cerr << "ERROR: rank " << rank
                          << " failed to open " << profileFileName << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            const std::vector<size_t> &cellSteps = manager->GetCellsStepsCounters();
            profileFile.precision(std::numeric_limits<double>::max_digits10);
            const size_t N = tess.GetPointNo();
            for(size_t i = 0; i < N; i++)
            {
                const double r = abs(tess.GetCellCM(i) - center);
                const size_t calls = i < cellSteps.size() ? cellSteps[i] : 0;
                profileFile << r << ',' << calls << std::endl;
            }

            profileFile.close();
            if(!profileFile)
            {
                std::cerr << "ERROR: rank " << rank
                          << " failed to write " << profileFileName << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
    catch(const UniversalError &e)
    {
        std::cerr << "=== UniversalError on rank " << rank << " ===" << std::endl;
        reportError(e, std::cerr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    catch(const std::exception &e)
    {
        std::cerr << "=== std::exception on rank " << rank << ": " << e.what()
                  << " ===" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
