#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "mpi/mpi_commands.hpp"
#include "misc/mesh_generator3D.hpp"
#include "3D/GeometryCommon/RoundGrid3D.hpp"
#include "3D/tessellation/voronoi/Voronoi3D.hpp"
#include "3D/tessellation/utils/RandomInCell.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "3D/radiation/MonteCarloPhysics3D.hpp"
#include "3D/monte/MonteCarloManager3D.hpp"
#include "monte/population/PopulationControl.hpp"
#include "monte/boundary/BoundaryCondition.hpp"
#include "utils/arguments/ArgumentParser.hpp"
#include "utils/debug/vtune.h"

using Particle3D = MonteCarloParticle<Vector3D, Tessellation3D>;

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
class VacuumBoundary : public BoundaryCondition<T, Grid>
{
public:
    using Particle = MonteCarloParticle<T, Grid>;

    explicit VacuumBoundary(const Grid &grid)
        : BoundaryCondition<T, Grid>(grid)
    {}

    MonteCarloParticleStatus apply(Particle&) override
    {
        return MonteCarloParticleStatus::REMOVE;
    }

    std::vector<Particle> generateNewBoundaryParticles(double) override
    {
        return std::vector<Particle>();
    }
};

template<typename T, typename Grid>
class IdentityPopulationControl : public PopulationControl<T, Grid>
{
public:
    using Particle = MonteCarloParticle<T, Grid>;

    explicit IdentityPopulationControl(const Grid &grid)
        : PopulationControl<T, Grid>(grid)
    {}

    std::vector<Particle> activate(const std::vector<Particle> &particles) override
    {
        return particles;
    }
};

class BallEmissionPhysics : public MonteCarloRadiationPhysics3D
{
public:
    BallEmissionPhysics(Tessellation3D &grid,
                        const std::shared_ptr<BoundaryCond> &boundary,
                        std::vector<ComputationalCell3D> &cells,
                        std::vector<Conserved3D> &conserved,
                        const std::shared_ptr<EquationOfState> &eos,
                        const std::shared_ptr<OpacityCalculator> &opacity,
                        const Vector3D &center,
                        double radius,
                        double emitterFraction,
                        bool isotropicEmission,
                        size_t photonsPerEmitterCell,
                        double particleWeight,
                        uint64_t seed)
        : MonteCarloRadiationPhysics3D(grid, boundary, cells, conserved, eos, opacity),
          center(center),
          radius(radius),
          emitterFraction(emitterFraction),
          isotropicEmission(isotropicEmission),
          photonsPerEmitterCell(photonsPerEmitterCell),
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
        particles.reserve(emitterCells * photonsPerEmitterCell);

        const size_t N = this->grid.GetPointNo();
        for(size_t i = 0; i < N; i++)
        {
            if(!isEmitterCell(i))
                continue;

            for(size_t j = 0; j < photonsPerEmitterCell; j++)
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
            UniversalError eo("Negative time step in BallEmissionPhysics::step");
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
        if(!isotropicEmission)
        {
            const Vector3D radial = particle.location - center;
            if(abs(radial) > radius * 1e-8 && ScalarProd(direction, radial) < 0.0)
                direction *= -1.0;
        }

        particle.velocity = direction * units::clight;
        particle.timeLeft = 0.0;
        return particle;
    }

    size_t countEmitterCells() const
    {
        size_t count = 0;
        const size_t N = this->grid.GetPointNo();
        for(size_t i = 0; i < N; i++)
            if(isEmitterCell(i))
                count++;
        return count;
    }

    size_t getLastEmittedCount() const
    {
        return lastEmittedCount;
    }

private:
    bool isEmitterCell(size_t i) const
    {
        const Vector3D point = this->grid.GetMeshPoint(i);
        if(abs(point - center) > radius)
            return false;
        if(emitterFraction >= 1.0)
            return true;
        return hashPointToUnitInterval(point) < emitterFraction;
    }

    Vector3D randomUnitVector() const
    {
        const double z = 2.0 * dist(rng) - 1.0;
        const double phi = 2.0 * M_PI * dist(rng);
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        return Vector3D(r * std::cos(phi), r * std::sin(phi), z);
    }

    static uint64_t mix64(uint64_t value)
    {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    static uint64_t quantizeCoordinate(double value)
    {
        return static_cast<uint64_t>(
            static_cast<int64_t>(std::llround(value * 1.0e12)));
    }

    double hashPointToUnitInterval(const Vector3D &point) const
    {
        uint64_t hash = 0x6a09e667f3bcc909ULL;
        hash ^= mix64(quantizeCoordinate(point.x));
        hash = mix64(hash ^ quantizeCoordinate(point.y));
        hash = mix64(hash ^ quantizeCoordinate(point.z));

        static constexpr double invMantissa = 1.0 / 9007199254740992.0;
        return static_cast<double>(hash >> 11) * invMantissa;
    }

    Vector3D center;
    double radius;
    double emitterFraction;
    bool isotropicEmission;
    size_t photonsPerEmitterCell;
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

static size_t CountCellsInsideBall(const Tessellation3D &grid,
                                   const Vector3D &center,
                                   double radius)
{
    size_t count = 0;
    const size_t N = grid.GetPointNo();
    for(size_t i = 0; i < N; i++)
        if(abs(grid.GetMeshPoint(i) - center) <= radius)
            count++;
    return count;
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
        ArgumentParser arguments("Ball emission benchmark");
        arguments.addPositional<size_t>("N_base", 20000, "number of background mesh points");
        arguments.addPositional<size_t>("photons_per_emitter_cell", 10, "photons emitted from each emitter cell per cycle");
        arguments.addPositional<size_t>("steps", 20, "number of benchmark cycles");
        arguments.addPositional<size_t>("N_ball", 0, "number of extra mesh points inside the emitting ball")
            .optionAlias("n-ball");

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
        arguments.addOption<double>("radius", 1.5, "emitting ball radius");
        arguments.addOption<size_t>("target-emitters", 0, "target number of emitter cells")
            .alias("emitter-target");
        arguments.addOption<std::string>("direction-mode", "outward", "emission direction mode")
            .choices({"outward", "isotropic"})
            .flagAlias("isotropic", "isotropic");
        arguments.addOption<double>("weight", 1.0, "particle weight");

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
        size_t photonsPerEmitterCell = arguments.get<size_t>("photons_per_emitter_cell");
        size_t steps = arguments.get<size_t>("steps");
        size_t N_ball = arguments.get<size_t>("N_ball");
        size_t targetEmitterCells = arguments.get<size_t>("target-emitters");
        double domainSize = arguments.get<double>("domain-size");
        double ballRadius = arguments.get<double>("radius");
        double dt = arguments.get<double>("dt");
        double particleWeight = arguments.get<double>("weight");

        std::string directionMode = arguments.get<std::string>("direction-mode");
        bool isotropicEmission = directionMode == "isotropic";

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

        if(!arguments.wasSet("N_ball"))
            N_ball = std::max<size_t>(512, N_base / 20);

        const double halfDomain = 0.5 * domainSize;
        const Vector3D ll(-halfDomain, -halfDomain, -halfDomain);
        const Vector3D ur(halfDomain, halfDomain, halfDomain);
        const Vector3D center(0.0, 0.0, 0.0);

        std::vector<Vector3D> points;
        if(rank == 0)
        {
            points = RandRectangular(N_base, ll, ur);
            std::vector<Vector3D> ballPoints = RandSphereR(N_ball, ll, ur, 0.0, ballRadius, center);
            points.insert(points.end(), ballPoints.begin(), ballPoints.end());

            const size_t shellPoints = std::max<size_t>(128, N_ball / 2);
            std::vector<Vector3D> sourceShell = RandSphereR(shellPoints, ll, ur, ballRadius, 2.0 * ballRadius, center);
            points.insert(points.end(), sourceShell.begin(), sourceShell.end());
            points.push_back(center);

            if(static_cast<rank_t>(points.size()) < ws)
            {
                std::cerr << "ERROR: only " << points.size() << " mesh points for "
                          << ws << " MPI ranks. Increase N_base." << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            std::cout << "Generated " << points.size() << " mesh points"
                      << " (background=" << N_base
                      << ", ball=" << N_ball
                      << ", shell=" << shellPoints << ")" << std::endl;
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

        const size_t N = tess.GetPointNo();
        std::vector<ComputationalCell3D> cells(N, initCell);
        std::vector<Conserved3D> extensives(N);
        for(size_t i = 0; i < N; i++)
        {
            cells[i].ID = i;
            PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);
        }

        auto eosPtr = std::make_shared<IdealGas>(eos);
        auto opacity = std::make_shared<TransparentOpacity>();
        auto boundary = std::make_shared<VacuumBoundary<Vector3D, Tessellation3D>>(tess);

        const size_t localCandidateCells = CountCellsInsideBall(tess, center, ballRadius);
        const size_t globalCandidateCells = GlobalSum(localCandidateCells);
        double emitterFraction = 1.0;
        if(targetEmitterCells > 0 && globalCandidateCells > targetEmitterCells)
        {
            emitterFraction = static_cast<double>(targetEmitterCells) /
                              static_cast<double>(globalCandidateCells);
        }

        auto physics = std::make_shared<BallEmissionPhysics>(
            tess, boundary, cells, extensives, eosPtr, opacity,
            center, ballRadius, emitterFraction, isotropicEmission,
            photonsPerEmitterCell, particleWeight,
            static_cast<uint64_t>(12345 + 7919 * rank));
        auto popControl = std::make_shared<IdentityPopulationControl<Vector3D, Tessellation3D>>(tess);

        const size_t localEmitterCells = physics->countEmitterCells();
        const size_t globalEmitterCells = GlobalSum(localEmitterCells);
        if(globalEmitterCells == 0)
        {
            if(rank == 0)
                std::cerr << "ERROR: no local mesh points landed inside the emitting ball."
                          << " Increase --n-ball or --radius." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MonteCarloConfig config;
        config.holdSmallIdleFlushes = true;
        config.transferDiagnosticsLevel = MonteCarloTransferDiagnosticsLevel::Off;
        config.initialBufferSize = std::max<size_t>(5000, globalEmitterCells * photonsPerEmitterCell / std::max<rank_t>(1, ws));
        config.minimalBuffSize = std::max<size_t>(50, config.initialBufferSize / 10);

        std::shared_ptr<MonteCarloManager3D> manager;
        #ifdef RICH_MPI
        if(managerKind == MANAGER_P2P)
        {
            manager = std::make_shared<TwoSidedMonteCarloManager3D>(tess, physics, popControl, boundary, MPI_COMM_WORLD);
        }
        else if(managerKind == MANAGER_OLD_RDMA)
        {
            manager = std::make_shared<RDMAMonteCarloManagerLegacy3D>(tess, physics, popControl, boundary, config, MPI_COMM_WORLD, rdmaType);
        }
        else
        {
            manager = std::make_shared<RDMAMonteCarloManager3D>(tess, physics, popControl, boundary, config, MPI_COMM_WORLD, rdmaType);
        }
        #else
        (void) managerKind;
        managerName = "serial";
        manager = std::make_shared<MonteCarloManagerSerial3D>(tess, physics, popControl, boundary);
        #endif

        if(rank == 0)
        {
            std::cout << "Ball emission benchmark:"
                      << " manager=" << managerName
                      << ", ranks=" << ws
                      << ", local cells(rank0)=" << N
                      << ", emitter cells=" << globalEmitterCells
                      << ", photons/emitter/cycle=" << photonsPerEmitterCell
                      << ", emitted/cycle=" << globalEmitterCells * photonsPerEmitterCell
                      << ", radius=" << ballRadius
                      << ", candidate emitters=" << globalCandidateCells
                      << ", target emitters=" << targetEmitterCells
                      << ", emitter fraction=" << emitterFraction
                      << ", direction=" << directionMode
                      << ", domain=[" << -halfDomain << "," << halfDomain << "]^3"
                      << ", dt=" << dt
                      << ", steps=" << steps
                      << std::endl;
        }

        std::vector<Particle3D> particles;
        double simTime = 0.0;
        double totalStepWall = 0.0;

        for(size_t cycle = 0; cycle < steps; cycle++)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            auto stepStart = std::chrono::high_resolution_clock::now();
            particles = manager->step(std::move(particles), cells, dt);
            auto stepEnd = std::chrono::high_resolution_clock::now();

            const double localStepWall = std::chrono::duration<double>(stepEnd - stepStart).count();
            const double stepWall = GlobalMax(localStepWall);
            totalStepWall += stepWall;
            simTime += dt;

            const size_t emitted = GlobalSum(physics->getLastEmittedCount());
            const size_t remaining = GlobalSum(particles.size());

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
