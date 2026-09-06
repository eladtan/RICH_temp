#include "FluxSourceCalculation.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "source/3D/monte/MonteCarloManager3D.hpp"
#ifdef RICH_MPI
#include "source/monte/manager/communication/RDMACommunicationEngine.hpp"
#endif // RICH_MPI
#include "source/Radiation/Diffusion.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/misc/universal_error.hpp"
#include "source/misc/utils.hpp"
#include "source/monte/particle/StepResult.hpp"
#include "source/monte/particle/ParticleStatus.hpp"
#include "source/monte/boundary/Vacuum.hpp"
#include "source/monte/population/NoPopulationControl.hpp"
#include "source/monte/physics/MonteCarloPhysics.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#include "source/mpi/mpi_commands.hpp"
#include "source/mpi/mpi_commands_3d.hpp"
#endif

namespace imc_postprocess_tde {
namespace {

double EffectiveThermalizationOpacity(
    OpacityCalculator const& opacity,
    ComputationalCell3D const& cell)
{
    double const absorption = opacity.CalcPlanckOpacity(cell);
    double const scattering = opacity.CalcScatteringOpacity(cell);
    if(!std::isfinite(absorption) || !std::isfinite(scattering) ||
       absorption < 0.0 || scattering < 0.0)
        throw UniversalError(
            "Flux-source thermalization probe encountered invalid opacity");
    return std::sqrt(std::max(
        0.0, 3.0 * absorption * (absorption + scattering)));
}

#ifdef RICH_MPI
constexpr int fluxSourceSlowRayMpiTag = 9942;
#endif

struct FluxSourceSlowRayReport
{
    int rank = 0;
    size_t rayId = 0;
    double elapsed_s = 0.0;
    uint64_t steps = 0;
    double location[3] = {0.0, 0.0, 0.0};
    double radial_cm = 0.0;
    size_t cellIndex = 0;
    size_t cellId = 0;
    double rho = 0.0;
    double temperature = 0.0;
    double tauAcc = 0.0;
    double targetTau = 0.0;
    double sigPlanck = 0.0;
    double sigScat = 0.0;
    double sigEff = 0.0;
    double sigDiff = 0.0;
};

class FluxSourceSlowRayMonitor
{
public:
    static constexpr double kInterval_s = 10.0;

    FluxSourceSlowRayMonitor(
        std::vector<ComputationalCell3D> const& cells,
        OpacityCalculator const& opacity,
        Vector3D center,
        double targetTau)
        : cells_(cells), opacity_(opacity), center_(center), targetTau_(targetTau)
    {
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank_);
#else
        mpiRank_ = 0;
#endif
    }

    void touchRay(
        size_t rayId,
        size_t cellIndex,
        Vector3D const& location,
        uint64_t steps,
        double tauAcc)
    {
        auto& tracked = tracked_[rayId];
        if(!tracked.active)
        {
            tracked.active = true;
            tracked.start = SteadyClock::now();
            tracked.lastPrintElapsed_s = -kInterval_s;
        }
        tracked.cellIndex = cellIndex;
        tracked.location = location;
        tracked.steps = steps;
        tracked.tauAcc = tauAcc;
    }

    void unregisterRay(size_t rayId) { tracked_.erase(rayId); }

    // Must run on the main MPI thread only (OpenMPI/UCX is not thread-safe here).
    void poll()
    {
        SteadyClock::time_point const now = SteadyClock::now();
        if(lastPoll_ != SteadyClock::time_point{} &&
           now - lastPoll_ < std::chrono::seconds(1))
            return;
        lastPoll_ = now;

        std::vector<FluxSourceSlowRayReport> reports;
        for(auto& entry : tracked_)
        {
            TrackedRay& ray = entry.second;
            if(!ray.active || ray.cellIndex >= cells_.size())
                continue;

            double const elapsed_s = std::chrono::duration<double>(
                now - ray.start).count();
            if(elapsed_s < kInterval_s)
                continue;
            if(elapsed_s - ray.lastPrintElapsed_s < kInterval_s)
                continue;

            ray.lastPrintElapsed_s = elapsed_s;
            reports.push_back(buildReport(entry.first, ray, elapsed_s));
        }

        for(FluxSourceSlowRayReport const& report : reports)
            publishReport(report);

        if(mpiRank_ == 0)
            drainIncomingReports();
    }

private:
    using SteadyClock = std::chrono::steady_clock;

    struct TrackedRay
    {
        bool active = false;
        SteadyClock::time_point start{};
        double lastPrintElapsed_s = -kInterval_s;
        size_t cellIndex = 0;
        Vector3D location{};
        uint64_t steps = 0;
        double tauAcc = 0.0;
    };

    FluxSourceSlowRayReport buildReport(
        size_t rayId,
        TrackedRay const& ray,
        double elapsed_s) const
    {
        ComputationalCell3D const& cell = cells_[ray.cellIndex];
        FluxSourceSlowRayReport report;
        report.rank = mpiRank_;
        report.rayId = rayId;
        report.elapsed_s = elapsed_s;
        report.steps = ray.steps;
        report.location[0] = ray.location.x;
        report.location[1] = ray.location.y;
        report.location[2] = ray.location.z;
        report.radial_cm = fastabs(ray.location - center_);
        report.cellIndex = ray.cellIndex;
        report.cellId = cell.ID;
        report.rho = cell.density;
        report.temperature = cell.temperature;
        report.tauAcc = ray.tauAcc;
        report.targetTau = targetTau_;
        report.sigPlanck = opacity_.CalcPlanckOpacity(cell);
        report.sigScat = opacity_.CalcScatteringOpacity(cell);
        report.sigEff = EffectiveThermalizationOpacity(opacity_, cell);
        report.sigDiff = opacity_.CalcDiffusionCoefficient(cell);
        return report;
    }

    void publishReport(FluxSourceSlowRayReport const& report)
    {
#ifdef RICH_MPI
        if(mpiRank_ == 0)
            printReport(report);
        else
        {
            FluxSourceSlowRayReport sendBuffer = report;
            MPI_Send(&sendBuffer, sizeof(sendBuffer), MPI_BYTE, 0,
                     fluxSourceSlowRayMpiTag, MPI_COMM_WORLD);
        }
#else
        printReport(report);
#endif
    }

    void drainIncomingReports()
    {
#ifdef RICH_MPI
        int hasMessage = 0;
        MPI_Status status;
        while(true)
        {
            MPI_Iprobe(MPI_ANY_SOURCE, fluxSourceSlowRayMpiTag, MPI_COMM_WORLD,
                         &hasMessage, &status);
            if(!hasMessage)
                break;
            FluxSourceSlowRayReport report{};
            MPI_Recv(&report, sizeof(FluxSourceSlowRayReport), MPI_BYTE, status.MPI_SOURCE,
                     fluxSourceSlowRayMpiTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printReport(report);
        }
#endif
    }

    static void printReport(FluxSourceSlowRayReport const& report)
    {
        std::cerr << "[FluxSourceSlowRay] rank=" << report.rank
                  << " ray_id=" << report.rayId
                  << " elapsed_s=" << report.elapsed_s
                  << " steps=" << report.steps
                  << " location=(" << report.location[0] << ", "
                  << report.location[1] << ", " << report.location[2] << ")"
                  << " radial_cm=" << report.radial_cm
                  << " cell_index=" << report.cellIndex
                  << " cell_id=" << report.cellId
                  << " rho=" << report.rho
                  << " T=" << report.temperature
                  << " tau_acc=" << report.tauAcc
                  << " target_tau=" << report.targetTau
                  << " sig_planck=" << report.sigPlanck
                  << " sig_scat=" << report.sigScat
                  << " sig_eff=" << report.sigEff
                  << " sig_diff=" << report.sigDiff
                  << std::endl;
        std::cerr.flush();
    }

    std::vector<ComputationalCell3D> const& cells_;
    OpacityCalculator const& opacity_;
    Vector3D center_;
    double targetTau_;
    int mpiRank_ = 0;
    SteadyClock::time_point lastPoll_{};
    std::unordered_map<size_t, TrackedRay> tracked_;
};

class GreyThermalizationProbePhysics
    : public MonteCarloPhysics<Vector3D, Tessellation3D>
{
public:
    using Particle = MonteCarloParticle<Vector3D>;
    using Functionality = MonteCarloFunctionality;
    using BoundaryCond = BoundaryCondition<Vector3D, Tessellation3D>;

    GreyThermalizationProbePhysics(
        Tessellation3D const& grid,
        std::shared_ptr<BoundaryCond> const& boundary,
        std::vector<ComputationalCell3D> const& cells,
        OpacityCalculator const& opacity,
        Vector3D center,
        double targetTau,
        size_t observerCount)
        : MonteCarloPhysics<Vector3D, Tessellation3D>(grid, boundary),
          cells_(cells), opacity_(opacity), center_(center),
          targetTau_(targetTau), radius_(observerCount, -1.0),
          valid_(observerCount, 0),
          slowRayMonitor_(std::make_unique<FluxSourceSlowRayMonitor>(
              cells, opacity, center, targetTau))
    {}

    ~GreyThermalizationProbePhysics() override = default;

    std::vector<Particle> preStep(double) override { return {}; }
    void postStep(std::vector<Particle> const&, double) override {}

    Functionality step(Particle& particle, std::vector<Particle>&) override
    {
        slowRayMonitor_->poll();
        Functionality result;
        result.change = MonteCarloParticleStatus::REMOVE;
        if(particle.id >= radius_.size() || particle.cellIndex >= cells_.size())
            return result;

        double const directionNorm = abs(particle.velocity);
        if(!(directionNorm > 0.0) || !std::isfinite(directionNorm))
            return result;
        Vector3D const direction = particle.velocity * (1.0 / directionNorm);
        particle.velocity = direction;

        slowRayMonitor_->touchRay(
            particle.id, particle.cellIndex, particle.location,
            particle.steps, particle.weight);
        slowRayMonitor_->poll();

        auto const intersection = this->getIntersectionDetails(particle);
        double const ds = std::get<1>(intersection);
        size_t const nextCell = std::get<2>(intersection);
        if(!(ds >= 0.0) || !std::isfinite(ds))
        {
            slowRayMonitor_->unregisterRay(particle.id);
            return result;
        }

        double const sigma = EffectiveThermalizationOpacity(
            opacity_, cells_[particle.cellIndex]);
        double const oldTau = particle.weight;
        double const newTau = oldTau + sigma * ds;
        if(oldTau < targetTau_ && newTau >= targetTau_)
        {
            double fraction = (sigma > 0.0 && ds > 0.0)
                ? (targetTau_ - oldTau) / (sigma * ds) : 0.0;
            fraction = std::clamp(fraction, 0.0, 1.0);
            Vector3D const crossing = particle.location
                + direction * (fraction * ds);
            radius_[particle.id] = fastabs(crossing - center_);
            valid_[particle.id] = 1;
            slowRayMonitor_->unregisterRay(particle.id);
            return result;
        }

        particle.weight = newTau;
        particle.location += direction * ds;
        particle.timeLeft -= ds;
        if(this->grid.IsPointOutsideBox(nextCell))
        {
            slowRayMonitor_->unregisterRay(particle.id);
            return result;
        }
        result.change = MonteCarloParticleStatus::CELL_MOVE;
        result.nextCellIndex = nextCell;
        return result;
    }

    std::vector<double> const& radius() const { return radius_; }
    std::vector<int> const& valid() const { return valid_; }

private:
    std::vector<ComputationalCell3D> const& cells_;
    OpacityCalculator const& opacity_;
    Vector3D center_;
    double targetTau_;
    std::vector<double> radius_;
    std::vector<int> valid_;
    std::unique_ptr<FluxSourceSlowRayMonitor> slowRayMonitor_;
};

size_t NearestObserverDirection(
    Vector3D const& point,
    Vector3D const& center,
    std::vector<Vector3D> const& directions)
{
    Vector3D radial = point - center;
    double const radius = abs(radial);
    if(!(radius > 0.0) || directions.empty())
        return 0;
    radial *= 1.0 / radius;
    size_t best = 0;
    double bestDot = -std::numeric_limits<double>::infinity();
    for(size_t i = 0; i < directions.size(); ++i)
    {
        double const dot = ScalarProd(radial, directions[i]);
        if(dot > bestDot)
        {
            bestDot = dot;
            best = i;
        }
    }
    return best;
}

std::vector<unsigned char> BuildOutsideSurfaceMask(
    Config const& cfg,
    PostprocessRuntime const& runtime)
{
    size_t const pointCount = runtime.tess.getMeshPoints().size();
    std::vector<unsigned char> outside(pointCount, 0);
    std::vector<Vector3D> const& directions = runtime.fluxSourceDirections;
    if(directions.empty() || directions.size() != runtime.fluxSourceRadius.size())
        throw UniversalError("Flux-source directions are unavailable");
    for(size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex)
    {
        Vector3D const point = runtime.tess.GetMeshPoint(pointIndex);
        double const radius = fastabs(point - cfg.center);
        if(!(radius > 0.0))
            continue;
        size_t const direction = NearestObserverDirection(
            point, cfg.center, directions);
        if(direction >= runtime.fluxSourceRadius.size() ||
           direction >= runtime.fluxSourceRadiusDirectlyResolved.size())
            throw UniversalError("Flux-source angular surface index is out of range");
        // A direction that never reaches the requested tau_eff has no CER.
        // Treat that angular channel as open instead of inventing a surface
        // radius from a neighboring ray.
        if(runtime.fluxSourceRadiusDirectlyResolved[direction] == 0)
            outside[pointIndex] = 1;
        else
            outside[pointIndex] =
                radius >= runtime.fluxSourceRadius[direction] ? 1 : 0;
    }
    return outside;
}

std::vector<Vector3D> ComputeGreyFldFlux(PostprocessRuntime& runtime)
{
    size_t const nCells = runtime.tess.GetPointNo();
    std::vector<double> radiationEnergy(nCells, 0.0);
    std::vector<double> diffusionCoefficient(nCells, 0.0);
    double localRadiationEnergy = 0.0;
    for(size_t i = 0; i < nCells; ++i)
    {
        radiationEnergy[i] = runtime.cells[i].density * runtime.cells[i].Erad;
        diffusionCoefficient[i] =
            runtime.greyOpacity->CalcDiffusionCoefficient(runtime.cells[i]);
        if(!std::isfinite(radiationEnergy[i]) || radiationEnergy[i] < 0.0 ||
           !std::isfinite(diffusionCoefficient[i]) ||
           diffusionCoefficient[i] < 0.0)
        {
            UniversalError eo("Flux-source FLD reconstruction encountered invalid cell data");
            eo.addEntry("Cell index", i);
            eo.addEntry("Radiation energy density", radiationEnergy[i]);
            eo.addEntry("Diffusion coefficient", diffusionCoefficient[i]);
            throw eo;
        }
        localRadiationEnergy += radiationEnergy[i] * runtime.tess.GetVolume(i);
    }

    double globalRadiationEnergy = localRadiationEnergy;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &globalRadiationEnergy, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_exchange_data(runtime.tess, radiationEnergy, true);
#endif
    if(!(globalRadiationEnergy > 0.0) || !std::isfinite(globalRadiationEnergy))
        throw UniversalError(
            "Flux-source comparison requires positive snapshot radiation energy (Erad)");

    std::vector<Vector3D> gradient(nCells, Vector3D(0.0, 0.0, 0.0));
    std::vector<size_t> neighbors;
    for(size_t i = 0; i < nCells; ++i)
    {
        auto const& faces = runtime.tess.GetCellFaces(i);
        runtime.tess.GetNeighbors(i, neighbors);
        if(faces.size() != neighbors.size())
            throw UniversalError("Flux-source FLD face/neighbor count mismatch");
        Vector3D grad(0.0, 0.0, 0.0);
        for(size_t j = 0; j < neighbors.size(); ++j)
        {
            size_t const neighbor = neighbors[j];
            if(runtime.tess.IsPointOutsideBox(neighbor) ||
               neighbor >= radiationEnergy.size())
                continue;
            Vector3D separation = runtime.tess.GetMeshPoint(neighbor)
                - runtime.tess.GetMeshPoint(i);
            double const distance = fastabs(separation);
            if(!(distance > 0.0) || !std::isfinite(distance))
                continue;
            Vector3D const normal = separation * (1.0 / distance);
            double const faceEnergy =
                0.5 * (radiationEnergy[i] + radiationEnergy[neighbor]);
            double const area = runtime.tess.GetArea(faces[j]);
            if(!(area > 0.0) || !std::isfinite(area))
                continue;
            grad += normal * (area * faceEnergy);
        }
        double const volume = runtime.tess.GetVolume(i);
        if(volume > 0.0 && std::isfinite(volume))
            grad *= 1.0 / volume;
        gradient[i] = grad;
    }

    std::vector<Vector3D> flux(nCells, Vector3D(0.0, 0.0, 0.0));
    for(size_t i = 0; i < nCells; ++i)
    {
        double const limiter = CG::CalcSingleFluxLimiter(
            gradient[i], diffusionCoefficient[i], radiationEnergy[i]);
        flux[i] = gradient[i] * (-limiter * diffusionCoefficient[i]);
        if(!std::isfinite(flux[i].x) || !std::isfinite(flux[i].y) ||
           !std::isfinite(flux[i].z))
            throw UniversalError("Flux-source FLD reconstruction produced non-finite flux");
    }
#ifdef RICH_MPI
    MPI_exchange_data(runtime.tess, flux, true);
#endif
    return flux;
}

} // namespace

void InitializeFluxSourceSurface(
    Config const& cfg,
    PostprocessRuntime& runtime)
{
    size_t const nSourceRays = cfg.fluxSourceRays > 0
        ? cfg.fluxSourceRays
        : runtime.observer->getNumObservers();
    if(nSourceRays == 0 || nSourceRays > static_cast<size_t>(INT_MAX))
        throw UniversalError("Flux-source surface requires a valid source-ray count");

    // The CER angular resolution is independent of the final observer binning.
    // Keep the old behavior when --flux-source-rays is omitted by using the
    // same ray count as --n-observers, but generate a dedicated direction set.
    std::vector<Vector3D> const directions =
        fibonacci_sphere_directions(nSourceRays);
    if(directions.size() != nSourceRays)
        throw UniversalError("Flux-source ray direction count mismatch");
    runtime.fluxSourceDirections = directions;

    auto boundary = std::make_shared<
        VacuumBoundaryCondition<Vector3D, Tessellation3D>>(runtime.tess);
    auto physics = std::make_shared<GreyThermalizationProbePhysics>(
        runtime.tess, boundary, runtime.cells, *runtime.greyOpacity,
        cfg.center, cfg.fluxSourceThermalizationTau, nSourceRays);
    auto population = std::make_shared<
        STORM::NoPopulationControl<Vector3D, Tessellation3D>>(runtime.tess);
    std::shared_ptr<MonteCarloManager3D> manager;
#ifdef RICH_MPI
    MonteCarloConfig monteCarloConfig;
    std::unique_ptr<STORM::CommunicationEngine<Vector3D>> engine =
        std::make_unique<STORM::RDMACommunicationEngine<Vector3D, Tessellation3D>>(
            runtime.tess, monteCarloConfig, MPI_COMM_WORLD, RDMA_Type::AUTO_RDMA);
    manager = std::make_shared<MonteCarloManager3D>(
        runtime.tess, physics, population, boundary, monteCarloConfig, std::move(engine));
#else
    manager = std::make_shared<MonteCarloManager3D>(
        runtime.tess, physics, population, boundary);
#endif

    // Keep the global number of live probe rays below the default 500-slot
    // RDMA peer buffer.  The probe creates no secondary particles, so a
    // 400-ray global batch cannot require remote-handler reallocation.
    size_t constexpr probeBatchSize = 400;
    size_t const batchCount =
        (nSourceRays + probeBatchSize - 1) / probeBatchSize;
    for(size_t batchBegin = 0; batchBegin < nSourceRays;
        batchBegin += probeBatchSize)
    {
        size_t const batchEnd = std::min(
            nSourceRays, batchBegin + probeBatchSize);
        std::vector<GreyThermalizationProbePhysics::Particle> particles;
        particles.reserve(
            (batchEnd - batchBegin) / std::max(1, runtime.mpiSize) + 1);

        for(size_t rayIndex = batchBegin; rayIndex < batchEnd; ++rayIndex)
        {
            Vector3D direction = directions[rayIndex];
            double const directionNorm = abs(direction);
            if(!(directionNorm > 0.0))
                continue;
            direction *= 1.0 / directionNorm;
            Vector3D const spherePoint = cfg.center + cfg.radius * direction;

            bool local = false;
#ifdef RICH_MPI
            if(!runtime.tess.IsPointOutsideBox(spherePoint))
                local = runtime.tess.GetOwner(spherePoint) == runtime.rank;
#else
            local = !runtime.tess.IsPointOutsideBox(spherePoint);
#endif
            if(!local)
                continue;

            size_t const cellIndex = runtime.tess.GetContainingCell(spherePoint);
            if(cellIndex >= runtime.tess.GetPointNo())
                continue;
            GreyThermalizationProbePhysics::Particle particle;
            particle.id = rayIndex;
            particle.location = spherePoint;
            particle.velocity = -1.0 * direction;
            particle.cellIndex = cellIndex;
            particle.cellID = runtime.cells[cellIndex].ID;
            particle.sourceCellID = particle.cellID;
            particle.weight = 0.0;
            particles.push_back(particle);
        }

        unsigned long long localLaunched =
            static_cast<unsigned long long>(particles.size());
        unsigned long long globalLaunched = localLaunched;
#ifdef RICH_MPI
        MPI_Allreduce(&localLaunched, &globalLaunched, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
        size_t const batchNumber = batchBegin / probeBatchSize + 1;
        if(globalLaunched > static_cast<unsigned long long>(probeBatchSize) ||
           globalLaunched > static_cast<unsigned long long>(batchEnd - batchBegin))
        {
            UniversalError eo(
                "Flux-source probe batch exceeded its global ray limit");
            eo.addEntry("Batch", batchNumber);
            eo.addEntry("Batch begin", batchBegin);
            eo.addEntry("Batch end", batchEnd);
            eo.addEntry("Global launched rays", globalLaunched);
            eo.addEntry("Probe batch limit", probeBatchSize);
            throw eo;
        }
        if(runtime.rank == 0)
            std::cout << "FLUX_SOURCE_PROBE_BATCH batch="
                      << batchNumber << "/" << batchCount
                      << " range=[" << batchBegin << "," << batchEnd << ")"
                      << " global_rays=" << globalLaunched
                      << " limit=" << probeBatchSize
                      << std::endl;

        if(globalLaunched > 0)
        {
            manager->getParticles() = std::move(particles);
            manager->step(runtime.cells, 2.01 * cfg.radius);
        }
    }

    runtime.fluxSourceRadius = physics->radius();
    runtime.fluxSourceRadiusDirectlyResolved = physics->valid();
#ifdef RICH_MPI
    if(!runtime.fluxSourceRadius.empty())
    {
        MPI_Allreduce(MPI_IN_PLACE, runtime.fluxSourceRadius.data(),
                      static_cast<int>(runtime.fluxSourceRadius.size()),
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE,
                      runtime.fluxSourceRadiusDirectlyResolved.data(),
                      static_cast<int>(runtime.fluxSourceRadiusDirectlyResolved.size()),
                      MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
#endif

    size_t const directCount = static_cast<size_t>(std::count(
        runtime.fluxSourceRadiusDirectlyResolved.begin(),
        runtime.fluxSourceRadiusDirectlyResolved.end(), 1));
    if(directCount == 0)
        throw UniversalError(
            "No inward grey rays reached the requested flux-source optical depth");

    for(size_t i = 0; i < nSourceRays; ++i)
    {
        if(runtime.fluxSourceRadiusDirectlyResolved[i] == 0)
            continue;
        double const radius = runtime.fluxSourceRadius[i];
        if(!(radius > 0.0) || !(radius < cfg.radius) ||
           !std::isfinite(radius))
            throw UniversalError("Flux-source ray returned an invalid radius");
    }

    runtime.fluxSourceEnabled = true;
    runtime.fluxSourceTau = cfg.fluxSourceThermalizationTau;
    runtime.fluxSourceDirectlyResolvedFraction =
        static_cast<double>(directCount) / static_cast<double>(nSourceRays);
    if(runtime.rank == 0)
        std::cout << "FLUX_SOURCE_SURFACE tau_eff="
                  << runtime.fluxSourceTau
                  << " source_rays=" << nSourceRays
                  << " output_observers=" << runtime.observer->getNumObservers()
                  << " directly_resolved=" << directCount << "/"
                  << nSourceRays
                  << " fraction="
                  << runtime.fluxSourceDirectlyResolvedFraction
                  << std::endl;
}

void ConfigureFluxSourceForCurrentDecomposition(
    Config const& cfg,
    PostprocessRuntime& runtime,
    RadiationIMC& physics)
{
    if(!cfg.fluxSourceCompare)
    {
        physics.clearPostProcessExternalSources();
        return;
    }
    if(!runtime.fluxSourceEnabled || runtime.fluxSourceRadius.empty())
        throw UniversalError("Flux-source surface was not initialized");

    std::vector<Vector3D> const fldFlux = ComputeGreyFldFlux(runtime);
    std::vector<unsigned char> const outside =
        BuildOutsideSurfaceMask(cfg, runtime);
    size_t const nCells = runtime.tess.GetPointNo();
    if(runtime.cells.size() < nCells)
        throw UniversalError(
            "Flux-source setup has fewer owned cells than tessellation points");

    // FLD and surface-mask arrays contain MPI ghost slots.  Build the same
    // point-indexed view of stable cell IDs so a CER face whose interior
    // neighbor is a ghost can be represented without indexing the owned-only
    // runtime.cells array.  Skipping such a face would make the physical CER
    // and its luminosity depend on the MPI partition.
    size_t const invalidCellID = std::numeric_limits<size_t>::max();
    std::vector<size_t> pointCellIDs(nCells, invalidCellID);
    for(size_t i = 0; i < nCells; ++i)
        pointCellIDs[i] = runtime.cells[i].ID;
#ifdef RICH_MPI
    MPI_exchange_data(
        runtime.tess, pointCellIDs, true, 1, &invalidCellID);
#endif

    std::vector<RadiationIMC::PostProcessExternalSource> sources;
    std::vector<size_t> neighbors;
    double localLuminosity = 0.0;
    double localNetLuminosity = 0.0;
    double localInwardLuminosity = 0.0;
    uint64_t localBoundaryFaces = 0;
    uint64_t localEmittingFaces = 0;

    for(size_t outerCell = 0; outerCell < nCells; ++outerCell)
    {
        if(outerCell >= outside.size() || outside[outerCell] == 0)
            continue;

        Vector3D const outerPoint = runtime.tess.GetMeshPoint(outerCell);
        auto const& faces = runtime.tess.GetCellFaces(outerCell);
        runtime.tess.GetNeighbors(outerCell, neighbors);
        if(faces.size() != neighbors.size())
            throw UniversalError("Flux-source face/neighbor count mismatch");
        for(size_t j = 0; j < neighbors.size(); ++j)
        {
            size_t const innerCell = neighbors[j];
            if(runtime.tess.IsPointOutsideBox(innerCell) ||
               innerCell >= fldFlux.size() || innerCell >= outside.size() ||
               innerCell >= pointCellIDs.size() ||
               outside[innerCell] != 0)
                continue;

            Vector3D const innerPoint = runtime.tess.GetMeshPoint(innerCell);
            Vector3D normal = outerPoint - innerPoint;
            double const distance = abs(normal);
            double const area = runtime.tess.GetArea(faces[j]);
            if(!(distance > 0.0) || !std::isfinite(distance) ||
               !(area > 0.0) || !std::isfinite(area))
                throw UniversalError("Flux-source surface has invalid face geometry");
            normal *= 1.0 / distance;

            Vector3D const faceFlux = 0.5 *
                (fldFlux[outerCell] + fldFlux[innerCell]);
            double const signedOutwardFlux = ScalarProd(faceFlux, normal);
            if(!std::isfinite(signedOutwardFlux))
                throw UniversalError("Flux-source surface has non-finite face flux");
            double const signedLuminosity = signedOutwardFlux * area;
            double const luminosity = std::max(0.0, signedLuminosity);
            localNetLuminosity += signedLuminosity;
            localInwardLuminosity += std::max(0.0, -signedLuminosity);

            size_t const interiorCellID = pointCellIDs[innerCell];
            if(interiorCellID == invalidCellID)
            {
                UniversalError eo(
                    "Flux-source CER face is missing its interior ghost cell ID");
                eo.addEntry("Interior point index", innerCell);
                eo.addEntry("Exterior point index", outerCell);
                throw eo;
            }
            RadiationIMC::PostProcessExternalSource source;
            source.faceIndex = faces[j];
            source.cellID = runtime.cells[outerCell].ID;
            source.interiorCellID = interiorCellID;
            source.location = runtime.tess.FaceCM(faces[j]);
            source.outwardNormal = normal;
            source.luminosity = luminosity;
            sources.push_back(source);
            ++localBoundaryFaces;
            if(luminosity > 0.0)
            {
                localLuminosity += luminosity;
                ++localEmittingFaces;
            }
        }
    }

    double rawGlobalLuminosity = localLuminosity;
    double globalNetLuminosity = localNetLuminosity;
    double globalInwardLuminosity = localInwardLuminosity;
    uint64_t globalBoundaryFaces = localBoundaryFaces;
    uint64_t globalEmittingFaces = localEmittingFaces;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &rawGlobalLuminosity, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &globalNetLuminosity, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &globalInwardLuminosity, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &globalBoundaryFaces, 1,
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &globalEmittingFaces, 1,
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
    if(!(rawGlobalLuminosity > 0.0) || globalBoundaryFaces == 0 ||
       globalEmittingFaces == 0)
        throw UniversalError(
            "Flux-source CER produced no outward positive-flux faces");

    double luminosityScale = 1.0;
    if(runtime.fluxSourceInjectedLuminosity > 0.0)
    {
        double const reference = runtime.fluxSourceInjectedLuminosity;
        double const relDifference = std::abs(rawGlobalLuminosity - reference) /
            std::max({rawGlobalLuminosity, reference,
                      std::numeric_limits<double>::min()});
        if(relDifference > 1.0e-8)
        {
            UniversalError eo(
                "Flux-source luminosity changed after MPI repartition/rebuild");
            eo.addEntry("Reference luminosity", reference);
            eo.addEntry("Rebuilt luminosity", rawGlobalLuminosity);
            eo.addEntry("Relative difference", relDifference);
            throw eo;
        }
        if(runtime.fluxSourceBoundaryFaceCount != globalBoundaryFaces ||
           runtime.fluxSourceEmittingFaceCount != globalEmittingFaces)
            throw UniversalError(
                "Flux-source face topology changed after MPI repartition/rebuild");
        // Signed fluxes can be near zero after cancellation.  Normalize their
        // rebuild tolerance to the positive source luminosity as well, rather
        // than amplifying harmless partition-order roundoff by dividing by a
        // nearly zero signed diagnostic.
        double const netScale = std::max({
            reference,
            std::abs(runtime.fluxSourceNetLuminosity),
            std::abs(globalNetLuminosity),
            std::numeric_limits<double>::min()});
        double const inwardScale = std::max({
            reference,
            runtime.fluxSourceInwardLuminosity,
            globalInwardLuminosity,
            std::numeric_limits<double>::min()});
        if(std::abs(runtime.fluxSourceNetLuminosity - globalNetLuminosity) /
               netScale > 1.0e-8 ||
           std::abs(runtime.fluxSourceInwardLuminosity -
                    globalInwardLuminosity) / inwardScale > 1.0e-8)
            throw UniversalError(
                "Flux-source signed flux changed after MPI repartition/rebuild");
        luminosityScale = reference / rawGlobalLuminosity;
    }
    else
    {
        runtime.fluxSourceInjectedLuminosity = rawGlobalLuminosity;
        runtime.fluxSourceNetLuminosity = globalNetLuminosity;
        runtime.fluxSourceInwardLuminosity = globalInwardLuminosity;
        runtime.fluxSourceBoundaryFaceCount = globalBoundaryFaces;
        runtime.fluxSourceEmittingFaceCount = globalEmittingFaces;
    }

    if(luminosityScale != 1.0)
    {
        for(auto& source : sources)
            source.luminosity *= luminosityScale;
    }
    physics.setPostProcessExternalSources(std::move(sources));

    if(runtime.rank == 0)
        std::cout << "FLUX_SOURCE_CONFIG boundary_faces="
                  << globalBoundaryFaces
                  << " emitting_faces=" << globalEmittingFaces
                  << " raw_luminosity=" << rawGlobalLuminosity
                  << " normalized_luminosity="
                  << runtime.fluxSourceInjectedLuminosity
                  << " net_luminosity=" << globalNetLuminosity
                  << " clipped_inward_luminosity="
                  << globalInwardLuminosity
                  << " inward_fraction="
                  << (rawGlobalLuminosity > 0.0
                      ? globalInwardLuminosity / rawGlobalLuminosity : 0.0)
                  << " erg/s" << std::endl;
}

FluxSourcePolarizationSummary ComputeFluxSourcePolarizationSummary(
    SphericalObserver::ObserverQualitySnapshot const& snapshot)
{
    FluxSourcePolarizationSummary result;
    if(!snapshot.polarizationEnabled)
        return result;

    size_t const count = std::min(
        snapshot.energy.size(),
        std::min(snapshot.stokesQ.size(), snapshot.stokesU.size()));
    double totalEnergy = 0.0;
    double polarizedEnergy = 0.0;
    for(size_t i = 0; i < count; ++i)
    {
        double const energy = snapshot.energy[i];
        double const q = snapshot.stokesQ[i];
        double const u = snapshot.stokesU[i];
        if(!(energy > 0.0) || !std::isfinite(energy) ||
           !std::isfinite(q) || !std::isfinite(u))
            continue;
        totalEnergy += energy;
        polarizedEnergy += std::sqrt(q * q + u * u);
        ++result.observerCount;
    }
    if(totalEnergy > 0.0)
        result.luminosityWeightedDegree = std::clamp(
            polarizedEnergy / totalEnergy, 0.0, 1.0);
    return result;
}

} // namespace imc_postprocess_tde
