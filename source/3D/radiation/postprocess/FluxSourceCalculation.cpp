#include "FluxSourceCalculation.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <functional>
#include <string>
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
#include "PostProcessCommunication.hpp"
#endif // RICH_MPI
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/planck_integral/planck_integral.hpp"
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

// Same effective thermalization opacity, evaluated for one energy group of a
// multigroup opacity at the group's representative energy.
double EffectiveThermalizationOpacityAtEnergy(
    OpacityCalculator const& opacity,
    ComputationalCell3D const& cell,
    double energy)
{
    double const absorption = opacity.CalcAbsorptionOpacity(cell, energy);
    double const scattering = opacity.CalcScatteringOpacity(cell, energy);
    if(!std::isfinite(absorption) || !std::isfinite(scattering) ||
       absorption < 0.0 || scattering < 0.0)
    {
        UniversalError eo(
            "Flux-source thermalization probe encountered invalid group opacity");
        eo.addEntry("Energy", energy);
        eo.addEntry("Absorption", absorption);
        eo.addEntry("Scattering", scattering);
        throw eo;
    }
    return std::sqrt(std::max(
        0.0, 3.0 * absorption * (absorption + scattering)));
}

// One optical-depth channel followed by a probe ray: the grey effective
// opacity, or one energy group's effective opacity.
struct ThermalizationProbeChannel
{
    std::string name;
    std::function<double(ComputationalCell3D const&)> effectiveOpacity;
};

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

class ThermalizationProbePhysics
    : public MonteCarloPhysics<Vector3D, Tessellation3D>
{
public:
    using Particle = MonteCarloParticle<Vector3D>;
    using Functionality = MonteCarloFunctionality;
    using BoundaryCond = BoundaryCondition<Vector3D, Tessellation3D>;

    // Every ray integrates all channels at once, inward from the observer
    // sphere, and records for each channel the radius where that channel's
    // effective optical depth first reaches targetTau. The ray ends when all
    // channels have resolved, when it leaves the box, or at its closest
    // approach to the centre (beyond that it would be climbing out the far
    // side and any later crossing would belong to the opposite direction).
    ThermalizationProbePhysics(
        Tessellation3D const& grid,
        std::shared_ptr<BoundaryCond> const& boundary,
        std::vector<ComputationalCell3D> const& cells,
        std::vector<ThermalizationProbeChannel> channels,
        OpacityCalculator const& monitorOpacity,
        Vector3D center,
        double targetTau,
        size_t rayCount)
        : MonteCarloPhysics<Vector3D, Tessellation3D>(grid, boundary),
          cells_(cells), channels_(std::move(channels)), center_(center),
          targetTau_(targetTau), rayCount_(rayCount),
          tau_(rayCount * channels_.size(), 0.0),
          radius_(channels_.size(), std::vector<double>(rayCount, -1.0)),
          valid_(channels_.size(), std::vector<int>(rayCount, 0)),
          slowRayMonitor_(std::make_unique<FluxSourceSlowRayMonitor>(
              cells, monitorOpacity, center, targetTau))
    {
        if(channels_.empty())
            throw UniversalError("Flux-source probe needs at least one opacity channel");
    }

    ~ThermalizationProbePhysics() override = default;

    std::vector<Particle> preStep(double) override { return {}; }
    void postStep(std::vector<Particle> const&, double) override {}

    Functionality step(Particle& particle, std::vector<Particle>&) override
    {
        slowRayMonitor_->poll();
        Functionality result;
        result.change = MonteCarloParticleStatus::REMOVE;
        if(particle.id >= rayCount_ || particle.cellIndex >= cells_.size())
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

        // Distance left before the ray passes the centre; never integrate
        // beyond it.
        double const remainingInward =
            -ScalarProd(particle.location - center_, direction);
        bool const reachesCentre = !(remainingInward > ds);
        double const dsEffective = reachesCentre
            ? std::max(0.0, remainingInward) : ds;

        size_t const channelCount = channels_.size();
        double* tau = &tau_[particle.id * channelCount];
        bool allResolved = true;
        for(size_t channel = 0; channel < channelCount; ++channel)
        {
            if(valid_[channel][particle.id])
                continue;
            double const sigma =
                channels_[channel].effectiveOpacity(cells_[particle.cellIndex]);
            double const oldTau = tau[channel];
            double const newTau = oldTau + sigma * dsEffective;
            if(oldTau < targetTau_ && newTau >= targetTau_)
            {
                double fraction = (sigma > 0.0 && dsEffective > 0.0)
                    ? (targetTau_ - oldTau) / (sigma * dsEffective) : 0.0;
                fraction = std::clamp(fraction, 0.0, 1.0);
                Vector3D const crossing = particle.location
                    + direction * (fraction * dsEffective);
                radius_[channel][particle.id] = fastabs(crossing - center_);
                valid_[channel][particle.id] = 1;
                tau[channel] = targetTau_;
            }
            else
            {
                tau[channel] = newTau;
                allResolved = false;
            }
        }
        particle.weight = tau[0];

        if(allResolved || reachesCentre)
        {
            slowRayMonitor_->unregisterRay(particle.id);
            return result;
        }

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

    size_t channelCount() const { return channels_.size(); }
    std::string const& channelName(size_t channel) const { return channels_[channel].name; }
    std::vector<double> const& radius(size_t channel) const { return radius_[channel]; }
    std::vector<int> const& valid(size_t channel) const { return valid_[channel]; }

private:
    std::vector<ComputationalCell3D> const& cells_;
    std::vector<ThermalizationProbeChannel> channels_;
    Vector3D center_;
    double targetTau_;
    size_t rayCount_;
    std::vector<double> tau_;                 // rayCount * channels, ray-major
    std::vector<std::vector<double>> radius_; // [channel][ray]
    std::vector<std::vector<int>> valid_;     // [channel][ray]
    std::unique_ptr<FluxSourceSlowRayMonitor> slowRayMonitor_;
};

// Percentiles of the resolved radii of one channel, for the surface report.
struct RadiusSummary
{
    size_t resolved = 0;
    double p05 = 0.0, median = 0.0, p95 = 0.0;
};

RadiusSummary SummarizeRadii(std::vector<double> const& radius, std::vector<int> const& valid)
{
    RadiusSummary summary;
    std::vector<double> values;
    values.reserve(radius.size());
    for(size_t i = 0; i < radius.size() && i < valid.size(); ++i)
        if(valid[i]) values.push_back(radius[i]);
    summary.resolved = values.size();
    if(values.empty())
        return summary;
    std::sort(values.begin(), values.end());
    auto at = [&](double q) {
        size_t idx = static_cast<size_t>(q * static_cast<double>(values.size() - 1) + 0.5);
        return values[std::min(idx, values.size() - 1)];
    };
    summary.p05 = at(0.05);
    summary.median = at(0.5);
    summary.p95 = at(0.95);
    return summary;
}

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

// Nearest probe direction for every mesh point (local and ghost), or SIZE_MAX
// for a point at the centre.
std::vector<size_t> BuildNearestDirectionIndex(
    Config const& cfg,
    PostprocessRuntime const& runtime)
{
    size_t const pointCount = runtime.tess.getMeshPoints().size();
    std::vector<size_t> nearest(pointCount, std::numeric_limits<size_t>::max());
    std::vector<Vector3D> const& directions = runtime.fluxSourceDirections;
    if(directions.empty())
        throw UniversalError("Flux-source directions are unavailable");
    for(size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex)
    {
        Vector3D const point = runtime.tess.GetMeshPoint(pointIndex);
        if(!(fastabs(point - cfg.center) > 0.0))
            continue;
        nearest[pointIndex] = NearestObserverDirection(point, cfg.center, directions);
    }
    return nearest;
}

// Points at or beyond the per-direction surface `radius` (1) or inside it (0).
// A direction whose ray never reached the target depth has no surface and is
// treated as open.
std::vector<unsigned char> BuildOutsideMaskForRadii(
    Config const& cfg,
    PostprocessRuntime const& runtime,
    std::vector<size_t> const& nearestDirection,
    std::vector<double> const& surfaceRadius,
    std::vector<int> const& surfaceResolved)
{
    size_t const pointCount = runtime.tess.getMeshPoints().size();
    std::vector<unsigned char> outside(pointCount, 0);
    if(surfaceRadius.size() != runtime.fluxSourceDirections.size() ||
       surfaceResolved.size() != surfaceRadius.size())
        throw UniversalError("Flux-source surface radii do not match the direction set");
    for(size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex)
    {
        size_t const direction = nearestDirection[pointIndex];
        if(direction == std::numeric_limits<size_t>::max())
            continue;
        if(direction >= surfaceRadius.size())
            throw UniversalError("Flux-source angular surface index is out of range");
        if(surfaceResolved[direction] == 0)
            outside[pointIndex] = 1;
        else
        {
            double const radius = fastabs(runtime.tess.GetMeshPoint(pointIndex) - cfg.center);
            outside[pointIndex] = radius >= surfaceRadius[direction] ? 1 : 0;
        }
    }
    return outside;
}

std::vector<unsigned char> BuildOutsideSurfaceMask(
    Config const& cfg,
    PostprocessRuntime const& runtime)
{
    return BuildOutsideMaskForRadii(
        cfg, runtime, BuildNearestDirectionIndex(cfg, runtime),
        runtime.fluxSourceRadius, runtime.fluxSourceRadiusDirectlyResolved);
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
    // Channel 0 is always the grey effective depth (kept for the report). In
    // mg-innermost mode one channel per energy group follows, evaluated with
    // the multigroup opacity at the group's representative energy.
    bool const mgSurface =
        cfg.fluxSourceSurfaceMode == FluxSourceSurfaceMode::MultigroupInnermost;
    std::vector<ThermalizationProbeChannel> channels;
    {
        OpacityCalculator const& grey = *runtime.greyOpacity;
        channels.push_back(ThermalizationProbeChannel{
            "grey",
            [&grey](ComputationalCell3D const& cell) {
                return EffectiveThermalizationOpacity(grey, cell);
            }});
    }
#if ENERGY_GROUPS_NUM > 1
    if(mgSurface)
    {
        if(!runtime.opacity)
            throw UniversalError("mg-innermost flux-source surface needs the multigroup opacity");
        OpacityCalculator const& mg = *runtime.opacity;
        if(mg.energy_groups_center.size() != static_cast<size_t>(ENERGY_GROUPS_NUM))
        {
            UniversalError eo("Multigroup opacity group count does not match ENERGY_GROUPS_NUM");
            eo.addEntry("Opacity groups", mg.energy_groups_center.size());
            eo.addEntry("ENERGY_GROUPS_NUM", static_cast<double>(ENERGY_GROUPS_NUM));
            throw eo;
        }
        for(size_t group = 0; group < static_cast<size_t>(ENERGY_GROUPS_NUM); ++group)
        {
            double const energy = mg.energy_groups_center[group];
            channels.push_back(ThermalizationProbeChannel{
                "group" + std::to_string(group),
                [&mg, energy](ComputationalCell3D const& cell) {
                    return EffectiveThermalizationOpacityAtEnergy(mg, cell, energy);
                }});
        }
    }
#else
    if(mgSurface && runtime.rank == 0)
        std::cout << "FLUX_SOURCE_SURFACE note: single-group build, mg-innermost equals grey"
                  << std::endl;
#endif
    auto physics = std::make_shared<ThermalizationProbePhysics>(
        runtime.tess, boundary, runtime.cells, std::move(channels),
        *runtime.greyOpacity, cfg.center, cfg.fluxSourceThermalizationTau,
        nSourceRays);
    auto population = std::make_shared<
        STORM::NoPopulationControl<Vector3D, Tessellation3D>>(runtime.tess);
    std::shared_ptr<MonteCarloManager3D> manager;
#ifdef RICH_MPI
    MonteCarloConfig monteCarloConfig;
    std::unique_ptr<STORM::CommunicationEngine<Vector3D>> engine =
        MakeCommunicationEngine(
                runtime.tess, monteCarloConfig, cfg.communication, MPI_COMM_WORLD);
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
        std::vector<ThermalizationProbePhysics::Particle> particles;
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
            ThermalizationProbePhysics::Particle particle;
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

    // Each ray ran on exactly one rank; a max-reduction assembles every
    // channel's radii (unresolved entries stay at -1 / 0).
    size_t const channelCount = physics->channelCount();
    std::vector<std::vector<double>> channelRadius(channelCount);
    std::vector<std::vector<int>> channelValid(channelCount);
    for(size_t channel = 0; channel < channelCount; ++channel)
    {
        channelRadius[channel] = physics->radius(channel);
        channelValid[channel] = physics->valid(channel);
#ifdef RICH_MPI
        if(!channelRadius[channel].empty())
        {
            MPI_Allreduce(MPI_IN_PLACE, channelRadius[channel].data(),
                          static_cast<int>(channelRadius[channel].size()),
                          MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, channelValid[channel].data(),
                          static_cast<int>(channelValid[channel].size()),
                          MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        }
#endif
    }
    runtime.fluxSourceGreyRadius = channelRadius[0];
    runtime.fluxSourceGreyResolved = channelValid[0];
    runtime.fluxSourceGroupRadius.clear();
    runtime.fluxSourceGroupResolved.clear();

    size_t directionsMissingAGroup = 0;
    if(!mgSurface || channelCount == 1)
    {
        runtime.fluxSourceRadius = channelRadius[0];
        runtime.fluxSourceRadiusDirectlyResolved = channelValid[0];
    }
    else
    {
        // Innermost resolved group radius per direction. A group that never
        // reaches the target along a ray has no thermalization surface there
        // and is left out of the minimum; count those directions.
        runtime.fluxSourceRadius.assign(nSourceRays, -1.0);
        runtime.fluxSourceRadiusDirectlyResolved.assign(nSourceRays, 0);
        for(size_t i = 0; i < nSourceRays; ++i)
        {
            double innermost = std::numeric_limits<double>::infinity();
            bool anyGroup = false;
            bool allGroups = true;
            for(size_t channel = 1; channel < channelCount; ++channel)
            {
                if(channelValid[channel][i])
                {
                    anyGroup = true;
                    innermost = std::min(innermost, channelRadius[channel][i]);
                }
                else
                    allGroups = false;
            }
            if(anyGroup)
            {
                runtime.fluxSourceRadius[i] = innermost;
                runtime.fluxSourceRadiusDirectlyResolved[i] = 1;
            }
            if(!allGroups)
                ++directionsMissingAGroup;
        }
        runtime.fluxSourceGroupRadius.assign(channelRadius.begin() + 1, channelRadius.end());
        runtime.fluxSourceGroupResolved.assign(channelValid.begin() + 1, channelValid.end());
    }

    if(runtime.rank == 0)
    {
        for(size_t channel = 0; channel < channelCount; ++channel)
        {
            RadiusSummary const summary = SummarizeRadii(channelRadius[channel], channelValid[channel]);
            std::cout << "FLUX_SOURCE_CHANNEL name=" << physics->channelName(channel)
                      << " resolved=" << summary.resolved << "/" << nSourceRays
                      << " radius_p05/med/p95=" << summary.p05 << "/" << summary.median
                      << "/" << summary.p95 << " cm" << std::endl;
        }
        if(mgSurface && channelCount > 1)
        {
            RadiusSummary const grey = SummarizeRadii(channelRadius[0], channelValid[0]);
            RadiusSummary const surface = SummarizeRadii(
                runtime.fluxSourceRadius, runtime.fluxSourceRadiusDirectlyResolved);
            std::cout << "FLUX_SOURCE_SURFACE_MG innermost_radius_p05/med/p95="
                      << surface.p05 << "/" << surface.median << "/" << surface.p95
                      << " cm grey_median=" << grey.median
                      << " median_ratio_to_grey="
                      << (grey.median > 0.0 ? surface.median / grey.median : 0.0)
                      << " directions_missing_a_group=" << directionsMissingAGroup
                      << "/" << nSourceRays << std::endl;
        }
    }

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
                  << " surface_mode=" << (mgSurface ? "mg-innermost" : "grey")
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
    RadiationIMC& physics,
    OpacityCalculator const& emissionOpacity,
    bool multigroupPass)
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
    // Every rank needs the full set of face-cell IDs to tell face cells from
    // volume cells in the learned allocation.
    {
        std::vector<std::uint64_t> localIDs;
        localIDs.reserve(sources.size());
        for(auto const& source : sources)
            localIDs.push_back(static_cast<std::uint64_t>(source.cellID));
        std::sort(localIDs.begin(), localIDs.end());
        localIDs.erase(std::unique(localIDs.begin(), localIDs.end()), localIDs.end());
        std::vector<std::uint64_t> allIDs = localIDs;
#ifdef RICH_MPI
        int const localCount = static_cast<int>(localIDs.size());
        std::vector<int> counts(static_cast<size_t>(runtime.mpiSize), 0);
        MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        std::vector<int> displs(counts.size(), 0);
        size_t total = 0;
        for(size_t r = 0; r < counts.size(); ++r)
        {
            displs[r] = static_cast<int>(total);
            total += static_cast<size_t>(counts[r]);
        }
        allIDs.assign(total, 0);
        MPI_Allgatherv(localIDs.data(), localCount, MPI_UINT64_T,
                       allIDs.data(), counts.data(), displs.data(), MPI_UINT64_T, MPI_COMM_WORLD);
#endif
        runtime.fluxSourceCellIDs.clear();
        runtime.fluxSourceCellIDs.reserve(allIDs.size());
        for(std::uint64_t id : allIDs)
            runtime.fluxSourceCellIDs.insert(static_cast<size_t>(id));
    }

    physics.setPostProcessExternalSources(std::move(sources));

    if(cfg.volumeEmissionEnabled)
    {
        // Per group, a cell outside that group's thermalization surface has
        // Fleck factor 1 (real absorption and emission); inside it has 0
        // (effective scattering). The grey pass, and the grey surface mode,
        // use one surface for every group. Cells inside the deepest surface
        // therefore emit nothing and conserve energy exactly.
        size_t const groupCount = multigroupPass ? static_cast<size_t>(ENERGY_GROUPS_NUM) : 1;
        std::vector<size_t> const nearest = BuildNearestDirectionIndex(cfg, runtime);
        std::vector<std::vector<unsigned char>> outsideGroup(groupCount);
        bool const perGroupRadii = multigroupPass &&
            runtime.fluxSourceGroupRadius.size() == groupCount &&
            runtime.fluxSourceGroupResolved.size() == groupCount;
        for(size_t g = 0; g < groupCount; ++g)
        {
            if(!cfg.volumeEmissionGateGroups)
                outsideGroup[g] = outside;  // full spectrum from every cell outside the deep surface
            else if(perGroupRadii)
                outsideGroup[g] = BuildOutsideMaskForRadii(
                    cfg, runtime, nearest, runtime.fluxSourceGroupRadius[g], runtime.fluxSourceGroupResolved[g]);
            else if(!multigroupPass && !runtime.fluxSourceGreyRadius.empty())
                outsideGroup[g] = BuildOutsideMaskForRadii(
                    cfg, runtime, nearest, runtime.fluxSourceGreyRadius, runtime.fluxSourceGreyResolved);
            else
                outsideGroup[g] = outside;
        }
        std::vector<std::uint16_t> groupBits(nCells, 0);
        for(size_t i = 0; i < nCells; ++i)
        {
            std::uint16_t bits = 0;
            for(size_t g = 0; g < groupCount; ++g)
                if(i < outsideGroup[g].size() && outsideGroup[g][i])
                    bits |= static_cast<std::uint16_t>(1u << g);
            if(!multigroupPass && bits)
                bits = static_cast<std::uint16_t>((1u << ENERGY_GROUPS_NUM) - 1u); // grey: all groups alike
            groupBits[i] = bits;
        }

        // Instantaneous emissivity of each cell at its snapshot temperature,
        // restricted to the groups it can emit: sum_g f_g c kappa_g u_g(T) V.
        double constexpr aRad = 7.565732690980505e-15;
        double constexpr cLight = 2.99792458e10;
        std::vector<double> const& edges = emissionOpacity.energy_groups_boundary;
        bool const groupEdgesOk = multigroupPass && edges.size() == groupCount + 1;
        std::vector<double> cellLuminosity(nCells, 0.0);
        double localVolumeLuminosity = 0.0;
        // Hydro-side budget of the same cells: gross emission in all groups and
        // absorption of the snapshot radiation field with the same Planck mean,
        // c kappa_P (a T^4 - E_rad) V. The escaping luminosity the transport
        // should reproduce is the face flux plus this net emission.
        double localHydroGross = 0.0;
        double localHydroAbsorbed = 0.0;
        uint64_t localOutside = 0;
        for(size_t i = 0; i < nCells; ++i)
        {
            if(groupBits[i] == 0)
                continue;
            ++localOutside;
            ComputationalCell3D const& cell = runtime.cells[i];
            double const T = cell.temperature;
            double const volume = runtime.tess.GetVolume(i);
            double luminosity = 0.0;
            double grossAll = 0.0;
            if(groupEdgesOk)
            {
                for(size_t g = 0; g < groupCount; ++g)
                {
                    double const kappa = emissionOpacity.CalcAbsorptionOpacity(
                        cell, emissionOpacity.energy_groups_center[g]);
                    double const energyDensity = planck_integral::planck_energy_density_group_integral(
                        edges[g], edges[g + 1], T);
                    double const groupLuminosity = cLight * kappa * energyDensity * volume;
                    grossAll += groupLuminosity;
                    if((groupBits[i] >> g) & 1u)
                        luminosity += groupLuminosity;
                }
            }
            else
            {
                luminosity = emissionOpacity.CalcPlanckOpacity(cell) * aRad * cLight * T * T * T * T * volume;
                grossAll = luminosity;
            }
            {
                double const aT4 = aRad * T * T * T * T;
                double const kappaP = aT4 > 0.0 ? grossAll / (cLight * aT4 * volume) : 0.0;
                double const radiationEnergyDensity = cell.density * cell.Erad;
                if(std::isfinite(radiationEnergyDensity) && radiationEnergyDensity > 0.0)
                    localHydroAbsorbed += cLight * kappaP * radiationEnergyDensity * volume;
                localHydroGross += grossAll;
            }
            if(!std::isfinite(luminosity) || luminosity < 0.0)
            {
                UniversalError eo("Volume emission produced an invalid cell luminosity");
                eo.addEntry("Cell index", i);
                eo.addEntry("Temperature", T);
                eo.addEntry("Luminosity", luminosity);
                throw eo;
            }
            cellLuminosity[i] = luminosity;
            localVolumeLuminosity += luminosity;
        }
        double globalVolumeLuminosity = localVolumeLuminosity;
        double hydroBudget[2] = {localHydroGross, localHydroAbsorbed};
        uint64_t globalOutside = localOutside;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &globalVolumeLuminosity, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, hydroBudget, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &globalOutside, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
        double const hydroNetEmission = hydroBudget[0] - hydroBudget[1];
        // A cell cannot add more to the escaping luminosity than it emits, so
        // testing its gross emission against a fraction of the *escaping*
        // luminosity is a conservative cut. Before the first generation the
        // face-source flux stands in for the escaping luminosity.
        double const escapingReference = runtime.lastEscapingLuminosity > 0.0
            ? runtime.lastEscapingLuminosity : runtime.fluxSourceInjectedLuminosity;
        double const cutoff = cfg.volumeEmissionCutoffFraction * escapingReference;
        std::vector<std::uint8_t> mask(nCells, 0);
        uint64_t localKept = 0;
        double localKeptLuminosity = 0.0;
        for(size_t i = 0; i < nCells; ++i)
        {
            if(cellLuminosity[i] > 0.0 && cellLuminosity[i] >= cutoff)
            {
                mask[i] = 1;
                ++localKept;
                localKeptLuminosity += cellLuminosity[i];
            }
        }
        uint64_t globalKept = localKept;
        double globalKeptLuminosity = localKeptLuminosity;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &globalKept, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &globalKeptLuminosity, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        bool const changed = globalKept != runtime.volumeEmissionCells ||
                             globalOutside != runtime.volumeEmissionOutsideCells;
        runtime.volumeEmissionOutsideCells = globalOutside;
        runtime.volumeEmissionCells = globalKept;
        runtime.volumeEmissionLuminosity = globalVolumeLuminosity;
        runtime.volumeEmissionKeptLuminosity = globalKeptLuminosity;
        physics.setPostProcessVolumeEmission(std::move(mask), std::move(groupBits), 1.0, 0);
        physics.setPostProcessVolumeEmissionExactBase(cfg.volumeEmissionBurninExact);
        if(runtime.rank == 0 && (changed || !runtime.volumeEmissionReported))
        {
            runtime.volumeEmissionReported = true;
            std::cout << "VOLUME_EMISSION_CONFIG outside_cells=" << globalOutside
                      << " emitting_cells=" << globalKept
                      << " cutoff_fraction=" << cfg.volumeEmissionCutoffFraction
                      << " cutoff_reference_luminosity=" << escapingReference
                      << " cutoff_luminosity=" << cutoff
                      << " volume_luminosity=" << globalVolumeLuminosity
                      << " kept_luminosity=" << globalKeptLuminosity
                      << " face_luminosity=" << runtime.fluxSourceInjectedLuminosity
                      << " volume_to_face_ratio="
                      << (runtime.fluxSourceInjectedLuminosity > 0.0
                          ? globalVolumeLuminosity / runtime.fluxSourceInjectedLuminosity : 0.0)
                      << " hydro_gross_emission=" << hydroBudget[0]
                      << " hydro_absorbed=" << hydroBudget[1]
                      << " hydro_net_emission=" << hydroNetEmission
                      << " hydro_implied_luminosity=" << runtime.fluxSourceInjectedLuminosity + hydroNetEmission
                      << " erg/s" << std::endl;
        }
    }
    else
    {
        physics.clearPostProcessVolumeEmission();
    }

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
