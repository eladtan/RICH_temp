#include "FluxSourceCalculation.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <functional>
#include <string>
#include <cmath>
#include <iostream>
#include <sstream>
#include <limits>
#include <memory>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
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
#include "source/3D/radiation/StateHdf5IO.hpp"
#include "PostProcessConfig.hpp"
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

// Total optical depth of the photosphere, as in the observer photosphere
// (PhotosphereCalculation.cpp).
constexpr double kPhotosphereTau = 2.0 / 3.0;

// Absorption and scattering opacity of one probe channel in one cell.
struct ChannelOpacity
{
    double absorption = 0.0;
    double scattering = 0.0;
};

ChannelOpacity GreyChannelOpacity(
    OpacityCalculator const& opacity,
    ComputationalCell3D const& cell)
{
    ChannelOpacity const k{opacity.CalcPlanckOpacity(cell), opacity.CalcScatteringOpacity(cell)};
    if(!std::isfinite(k.absorption) || !std::isfinite(k.scattering) ||
       k.absorption < 0.0 || k.scattering < 0.0)
        throw UniversalError(
            "Flux-source thermalization probe encountered invalid opacity");
    return k;
}

// The same for one energy group of a multigroup opacity, at the group's
// representative energy.
ChannelOpacity GroupChannelOpacity(
    OpacityCalculator const& opacity,
    ComputationalCell3D const& cell,
    double energy)
{
    ChannelOpacity const k{opacity.CalcAbsorptionOpacity(cell, energy),
                           opacity.CalcScatteringOpacity(cell, energy)};
    if(!std::isfinite(k.absorption) || !std::isfinite(k.scattering) ||
       k.absorption < 0.0 || k.scattering < 0.0)
    {
        UniversalError eo(
            "Flux-source thermalization probe encountered invalid group opacity");
        eo.addEntry("Energy", energy);
        eo.addEntry("Absorption", k.absorption);
        eo.addEntry("Scattering", k.scattering);
        throw eo;
    }
    return k;
}

// Effective (thermalization) opacity sqrt(3 k_a (k_a + k_s)).
double EffectiveSigma(ChannelOpacity const& k)
{
    return std::sqrt(std::max(0.0, 3.0 * k.absorption * (k.absorption + k.scattering)));
}

// Total opacity k_a + k_s, the photosphere's.
double TotalSigma(ChannelOpacity const& k)
{
    return k.absorption + k.scattering;
}

double EffectiveThermalizationOpacity(
    OpacityCalculator const& opacity,
    ComputationalCell3D const& cell)
{
    return EffectiveSigma(GreyChannelOpacity(opacity, cell));
}

// One opacity channel followed by a probe ray: grey, or one energy group.
struct ThermalizationProbeChannel
{
    std::string name;
    std::function<ChannelOpacity(ComputationalCell3D const&)> opacity;
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

    // One particle per (ray, channel), with id = ray * channelCount + channel,
    // runs inward from the observer sphere and records the radius where the
    // channel's effective optical depth first reaches targetTau, the radius
    // where it reaches depthTau (the neighbour mapping's floor; 0 skips it),
    // and the radius where its total optical depth first reaches
    // kPhotosphereTau (its photosphere along the same ray). The particle
    // carries both depths (weight: effective, initialWeight: total), so they
    // survive its migration to another rank's domain. It ends when all are
    // found, when it leaves the box, or at its closest approach to the centre
    // (beyond that it would be climbing out the far side and any later
    // crossing would belong to the opposite direction).
    ThermalizationProbePhysics(
        Tessellation3D const& grid,
        std::shared_ptr<BoundaryCond> const& boundary,
        std::vector<ComputationalCell3D> const& cells,
        std::vector<ThermalizationProbeChannel> channels,
        OpacityCalculator const& monitorOpacity,
        Vector3D center,
        double targetTau,
        double depthTau,
        size_t rayCount)
        : MonteCarloPhysics<Vector3D, Tessellation3D>(grid, boundary),
          cells_(cells), channels_(std::move(channels)), center_(center),
          targetTau_(targetTau), depthTau_(depthTau), rayCount_(rayCount),
          radius_(channels_.size(), std::vector<double>(rayCount, -1.0)),
          valid_(channels_.size(), std::vector<int>(rayCount, 0)),
          depthRadius_(channels_.size(), std::vector<double>(rayCount, -1.0)),
          depthValid_(channels_.size(), std::vector<int>(rayCount, 0)),
          photosphereRadius_(channels_.size(), std::vector<double>(rayCount, -1.0)),
          photosphereValid_(channels_.size(), std::vector<int>(rayCount, 0)),
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
        size_t const channelCount = channels_.size();
        if(particle.id >= rayCount_ * channelCount || particle.cellIndex >= cells_.size())
            return result;
        size_t const ray = particle.id / channelCount;
        size_t const channel = particle.id % channelCount;

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

        // The depths travel with the particle: a crossing is recorded on the
        // rank where it happens, exactly once.
        ChannelOpacity const k = channels_[channel].opacity(cells_[particle.cellIndex]);
        double const oldTau = particle.weight;
        double const oldTauTotal = particle.initialWeight;
        if(oldTauTotal < kPhotosphereTau)
        {
            double const sigma = TotalSigma(k);
            double const newTauTotal = oldTauTotal + sigma * dsEffective;
            if(newTauTotal >= kPhotosphereTau)
            {
                photosphereRadius_[channel][ray] = crossingRadius(
                    particle.location, direction, dsEffective, sigma, oldTauTotal, kPhotosphereTau);
                photosphereValid_[channel][ray] = 1;
            }
            particle.initialWeight = newTauTotal;
        }
        double const finalTau = std::max(targetTau_, depthTau_);
        if(oldTau < finalTau)
        {
            double const sigma = EffectiveSigma(k);
            double const newTau = oldTau + sigma * dsEffective;
            if(oldTau < targetTau_ && newTau >= targetTau_)
            {
                radius_[channel][ray] = crossingRadius(
                    particle.location, direction, dsEffective, sigma, oldTau, targetTau_);
                valid_[channel][ray] = 1;
            }
            if(depthTau_ > 0.0 && oldTau < depthTau_ && newTau >= depthTau_)
            {
                depthRadius_[channel][ray] = crossingRadius(
                    particle.location, direction, dsEffective, sigma, oldTau, depthTau_);
                depthValid_[channel][ray] = 1;
            }
            particle.weight = newTau;
        }
        bool const allResolved =
            particle.weight >= finalTau && particle.initialWeight >= kPhotosphereTau;

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
    std::vector<double> const& photosphereRadius(size_t channel) const { return photosphereRadius_[channel]; }
    std::vector<int> const& photosphereValid(size_t channel) const { return photosphereValid_[channel]; }
    std::vector<double> const& depthRadius(size_t channel) const { return depthRadius_[channel]; }
    std::vector<int> const& depthValid(size_t channel) const { return depthValid_[channel]; }

private:
    // Radius where the depth accumulated over this step reaches `target`.
    double crossingRadius(Vector3D const& location, Vector3D const& direction,
                          double ds, double sigma, double oldTau, double target) const
    {
        double fraction = (sigma > 0.0 && ds > 0.0) ? (target - oldTau) / (sigma * ds) : 0.0;
        fraction = std::clamp(fraction, 0.0, 1.0);
        return fastabs(location + direction * (fraction * ds) - center_);
    }

    std::vector<ComputationalCell3D> const& cells_;
    std::vector<ThermalizationProbeChannel> channels_;
    Vector3D center_;
    double targetTau_;
    double depthTau_;
    size_t rayCount_;
    std::vector<std::vector<double>> radius_; // [channel][ray]
    std::vector<std::vector<int>> valid_;     // [channel][ray]
    std::vector<std::vector<double>> depthRadius_;       // [channel][ray]
    std::vector<std::vector<int>> depthValid_;           // [channel][ray]
    std::vector<std::vector<double>> photosphereRadius_; // [channel][ray]
    std::vector<std::vector<int>> photosphereValid_;     // [channel][ray]
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

// Point-indexed copies of the owned cells with the MPI ghosts appended, so a
// CER face whose interior neighbour is a ghost sees that cell's state.
std::vector<ComputationalCell3D> PointCellsWithGhosts(PostprocessRuntime const& runtime)
{
    size_t const nCells = runtime.tess.GetPointNo();
    std::vector<ComputationalCell3D> cells(runtime.cells.begin(),
                                           runtime.cells.begin() + static_cast<std::ptrdiff_t>(nCells));
#ifdef RICH_MPI
    MPI_exchange_data(runtime.tess, cells, true);
#endif
    return cells;
}

// The hydro's FLD luminosity from innerCell to outerCell across one face
// (Diffusion.cpp, matrix assembly): two-point gradient between the cells'
// centres of mass, both cells' diffusion coefficients at the face temperature
// ((T_1^4 + T_2^4) / 2)^(1/4), their harmonic mean, and the flux limiter on
// that gradient. The hydro also raises tiny gradients before the limiter
// (grad_factor); that only changes faces whose flux is negligible and is left
// out. Positive outward.
double HydroFldFaceLuminosity(
    PostprocessRuntime const& runtime,
    std::vector<ComputationalCell3D> const& pointCells,
    size_t innerCell,
    size_t outerCell,
    size_t face)
{
    if(innerCell >= pointCells.size() || outerCell >= pointCells.size())
        throw UniversalError("Flux-source hydro face flux is missing a cell (ghost exchange)");
    ComputationalCell3D inner = pointCells[innerCell];
    ComputationalCell3D outer = pointCells[outerCell];
    double const energyInner = inner.density * inner.Erad;
    double const energyOuter = outer.density * outer.Erad;
    Vector3D const centres = runtime.tess.GetCellCM(innerCell) - runtime.tess.GetCellCM(outerCell);
    Vector3D const separation = runtime.tess.GetMeshPoint(innerCell) - runtime.tess.GetMeshPoint(outerCell);
    double const centres2 = ScalarProd(centres, centres);
    double const separationLength = abs(separation);
    double const area = runtime.tess.GetArea(face);
    double const t4 = 0.5 * (std::pow(inner.temperature, 4) + std::pow(outer.temperature, 4));
    if(!(centres2 > 0.0) || !std::isfinite(centres2) || !(separationLength > 0.0) ||
       !(area > 0.0) || !std::isfinite(area) || !(t4 > 0.0) || !std::isfinite(t4))
    {
        UniversalError eo("Flux-source hydro face flux has invalid geometry or temperatures");
        eo.addEntry("Inner cell ID", inner.ID);
        eo.addEntry("Outer cell ID", outer.ID);
        eo.addEntry("Inner temperature", inner.temperature);
        eo.addEntry("Outer temperature", outer.temperature);
        eo.addEntry("Face area", area);
        throw eo;
    }
    double const faceTemperature = std::pow(t4, 0.25);
    inner.temperature = faceTemperature;
    outer.temperature = faceTemperature;
    double const dInner = runtime.greyOpacity->CalcDiffusionCoefficient(inner);
    double const dOuter = runtime.greyOpacity->CalcDiffusionCoefficient(outer);
    if(!(dInner > 0.0) || !std::isfinite(dInner) || !(dOuter > 0.0) || !std::isfinite(dOuter))
    {
        UniversalError eo("Flux-source hydro face flux has an invalid diffusion coefficient");
        eo.addEntry("Inner D", dInner);
        eo.addEntry("Outer D", dOuter);
        eo.addEntry("Face temperature", faceTemperature);
        throw eo;
    }
    double const faceD = 2.0 * dInner * dOuter / (dInner + dOuter);
    double const difference = energyInner - energyOuter;
    Vector3D const gradient = centres * (difference / centres2);
    double const limiter = CG::CalcSingleFluxLimiter(gradient, faceD, 0.5 * (energyInner + energyOuter));
    double const geometry = ScalarProd(centres, separation) / (separationLength * centres2);
    return limiter * faceD * geometry * difference * area;
}

// ---------------------------------------------------------------------------
// Surface smoothing and enclosed-region removal
// ---------------------------------------------------------------------------

constexpr size_t kSmoothingNeighbours = 8;

// For every resolved direction, the (up to) k resolved directions closest in
// angle, nearest first, the direction itself excluded; empty for unresolved
// directions. Shared by the smoothing and the neighbour mapping.
std::vector<std::vector<size_t>> NearestResolvedNeighbours(
    std::vector<Vector3D> const& directions,
    std::vector<int> const& resolved,
    size_t k)
{
    std::vector<std::vector<size_t>> neighbours(directions.size());
    if(k == 0)
        return neighbours;
    std::vector<std::pair<double, size_t>> byAngle;
    byAngle.reserve(directions.size());
    for(size_t d = 0; d < directions.size(); ++d)
    {
        if(!resolved[d])
            continue;
        byAngle.clear();
        for(size_t e = 0; e < directions.size(); ++e)
            if(e != d && resolved[e])
                byAngle.emplace_back(-ScalarProd(directions[d], directions[e]), e);
        size_t const count = std::min(k, byAngle.size());
        std::partial_sort(byAngle.begin(), byAngle.begin() + static_cast<std::ptrdiff_t>(count),
                          byAngle.end());
        neighbours[d].resize(count);
        for(size_t m = 0; m < count; ++m)
            neighbours[d][m] = byAngle[m].second;
    }
    return neighbours;
}

// Raises every resolved direction whose radius is below `floor` times the
// median radius of its kSmoothingNeighbours nearest resolved directions (the
// first ones in `neighbours`) to that value. Only raises: a raised surface is
// shallower than the target thermalization depth in that direction. With a
// non-empty `ceiling` no direction is raised above ceiling[d] (its own
// photosphere), and a direction without one (negative) is not raised at all;
// the directions held below the smoothing value are returned in `capped`.
// Returns the raised directions.
std::vector<size_t> SmoothSurfaceDips(
    std::vector<std::vector<size_t>> const& neighbours,
    std::vector<double>& radius,
    std::vector<int> const& resolved,
    double floor,
    std::vector<double> const& ceiling,
    std::vector<size_t>& capped)
{
    std::vector<size_t> raised;
    capped.clear();
    if(!(floor > 0.0))
        return raised;
    if(!ceiling.empty() && ceiling.size() != radius.size())
        throw UniversalError("Flux-source smoothing ceiling does not match the direction set");
    std::vector<double> const original = radius;
    std::vector<double> neighbourRadii;
    for(size_t d = 0; d < radius.size(); ++d)
    {
        if(!resolved[d])
            continue;
        size_t const k = std::min(kSmoothingNeighbours, neighbours[d].size());
        if(k == 0)
            continue;
        neighbourRadii.assign(k, 0.0);
        for(size_t m = 0; m < k; ++m)
            neighbourRadii[m] = original[neighbours[d][m]];
        std::nth_element(neighbourRadii.begin(),
                         neighbourRadii.begin() + static_cast<std::ptrdiff_t>(k / 2),
                         neighbourRadii.end());
        double const limit = floor * neighbourRadii[k / 2];
        if(!(original[d] < limit))
            continue;
        double target = limit;
        if(!ceiling.empty())
        {
            target = ceiling[d] > 0.0 ? std::min(limit, ceiling[d]) : original[d];
            if(target < limit)
                capped.push_back(d);
        }
        if(target > original[d])
        {
            radius[d] = target;
            raised.push_back(d);
        }
    }
    return raised;
}

// Lowers every resolved direction's radius to the smallest radius among
// itself and its `count` nearest resolved directions (the first ones in
// `neighbours`), but never below floor[d] (empty: no floor). A cell takes the
// radius of its nearest direction, so after this it is inside only if it
// lies below the surface of that direction and of all of its neighbours: a
// cell between directions never inherits a neighbour's larger radius, while
// the floor keeps the mapping out of gas far below the thermalization depth.
// Only lowers. Returns the lowered directions; the directions the floor
// stopped short of their neighbours' radius go to `held`.
std::vector<size_t> ApplyNeighbourMapping(
    std::vector<std::vector<size_t>> const& neighbours,
    std::vector<double>& radius,
    std::vector<int> const& resolved,
    size_t count,
    std::vector<double> const& floor,
    std::vector<size_t>& held)
{
    std::vector<size_t> lowered;
    held.clear();
    if(count == 0)
        return lowered;
    if(!floor.empty() && floor.size() != radius.size())
        throw UniversalError("Flux-source mapping floor does not match the direction set");
    std::vector<double> const original = radius;
    for(size_t d = 0; d < radius.size(); ++d)
    {
        if(!resolved[d])
            continue;
        double smallest = original[d];
        size_t const k = std::min(count, neighbours[d].size());
        for(size_t m = 0; m < k; ++m)
            smallest = std::min(smallest, original[neighbours[d][m]]);
        double target = smallest;
        if(!floor.empty() && floor[d] > smallest)
        {
            target = std::min(original[d], floor[d]);
            held.push_back(d);
        }
        if(target < original[d])
        {
            radius[d] = target;
            lowered.push_back(d);
        }
    }
    return lowered;
}

// Median and largest ratio numerator[d] / denominator[d] over `changed`, for
// the surface reports.
std::pair<double, double> MedianAndMaxRatio(
    std::vector<size_t> const& changed,
    std::vector<double> const& numerator,
    std::vector<double> const& denominator)
{
    std::vector<double> factors;
    factors.reserve(changed.size());
    for(size_t d : changed)
        factors.push_back(numerator[d] / denominator[d]);
    if(factors.empty())
        return {1.0, 1.0};
    std::sort(factors.begin(), factors.end());
    return {factors[factors.size() / 2], factors.back()};
}

#ifdef RICH_MPI
template <typename T>
std::vector<T> GatherToRankZero(std::vector<T> const& local, MPI_Datatype type, int rank)
{
    int mpiSize = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    int const count = static_cast<int>(local.size());
    std::vector<int> counts(rank == 0 ? static_cast<size_t>(mpiSize) : 0);
    MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> displacements(counts.size(), 0);
    int total = 0;
    for(size_t r = 0; r < counts.size(); ++r)
    {
        displacements[r] = total;
        total += counts[r];
    }
    std::vector<T> all(rank == 0 ? static_cast<size_t>(total) : 0);
    MPI_Gatherv(local.data(), count, type, all.data(), counts.data(),
                displacements.data(), type, 0, MPI_COMM_WORLD);
    return all;
}
#endif

// Outside cells are kept only if they connect, through faces between outside
// cells, to the exterior beyond the outermost surface radius. A region of
// outside cells enclosed on every side by inside cells is a trap: each of its
// faces is a thermalizing wall that returns packets into it, so nothing born
// or arriving there can leave before the transport time runs out. Such regions
// form where one direction's surface dips far below its neighbours' within a
// cone narrower than a cell. Clears `outside` for those cells (local and
// ghost) and returns the local ones. Collective.
std::vector<size_t> FindEnclosedOutsideCells(
    Config const& cfg,
    PostprocessRuntime const& runtime,
    std::vector<unsigned char> const& outside)
{
    double seedRadius = 0.0;
    for(size_t d = 0; d < runtime.fluxSourceRadius.size(); ++d)
        if(runtime.fluxSourceRadiusDirectlyResolved[d])
            seedRadius = std::max(seedRadius, runtime.fluxSourceRadius[d]);
    size_t const nLocal = runtime.tess.GetPointNo();
    size_t const nPoints = outside.size();
    std::vector<size_t> reached(nPoints, 0);
    std::vector<size_t> stack;
    unsigned long long seeds = 0;
    for(size_t i = 0; i < nLocal; ++i)
    {
        if(outside[i] && fastabs(runtime.tess.GetMeshPoint(i) - cfg.center) > seedRadius)
        {
            reached[i] = 1;
            stack.push_back(i);
            ++seeds;
        }
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &seeds, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    if(seeds == 0)
    {
        UniversalError eo("Flux-source exterior has no outside cell beyond the outermost surface radius");
        eo.addEntry("Outermost surface radius", seedRadius);
        throw eo;
    }
    std::vector<size_t> neighbors;
    auto connected = [&](size_t j) {
        return j < nPoints && !runtime.tess.IsPointOutsideBox(j) && outside[j];
    };
    auto flood = [&]() {
        while(!stack.empty())
        {
            size_t const i = stack.back();
            stack.pop_back();
            runtime.tess.GetNeighbors(i, neighbors);
            for(size_t j : neighbors)
            {
                if(!connected(j) || reached[j])
                    continue;
                reached[j] = 1;
                if(j < nLocal)
                    stack.push_back(j);
            }
        }
    };
    flood();
#ifdef RICH_MPI
    // Each round hands every rank its neighbours' reached cells (as ghosts)
    // and continues the fill from there, until no rank reaches a new cell.
    auto syncGhosts = [&]() {
        std::vector<size_t> shared(reached.begin(), reached.begin() + static_cast<std::ptrdiff_t>(nLocal));
        MPI_exchange_data(runtime.tess, shared, true);
        for(size_t g = nLocal; g < nPoints && g < shared.size(); ++g)
            reached[g] = shared[g];
    };
    while(true)
    {
        syncGhosts();
        unsigned long long added = 0;
        for(size_t i = 0; i < nLocal; ++i)
        {
            if(!outside[i] || reached[i])
                continue;
            runtime.tess.GetNeighbors(i, neighbors);
            for(size_t j : neighbors)
            {
                if(connected(j) && reached[j])
                {
                    reached[i] = 1;
                    stack.push_back(i);
                    ++added;
                    break;
                }
            }
        }
        flood();
        MPI_Allreduce(MPI_IN_PLACE, &added, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(added == 0)
            break;
    }
#endif
    std::vector<size_t> enclosed;
    for(size_t i = 0; i < nLocal; ++i)
        if(outside[i] && !reached[i])
            enclosed.push_back(i);
    return enclosed;
}

} // namespace

std::vector<unsigned char> BuildOutsideFluxSourceMask(
    Config const& cfg,
    PostprocessRuntime const& runtime)
{
    if(!runtime.fluxSourceEnabled || runtime.fluxSourceRadius.empty())
        throw UniversalError("Flux-source surface was not initialized");
    return BuildOutsideSurfaceMask(cfg, runtime);
}

void InitializeFluxSourceSurface(
    Config const& cfg,
    PostprocessRuntime& runtime)
{
    size_t const nSourceRays = cfg.fluxSourceRays > 0
        ? cfg.fluxSourceRays
        : runtime.observer->getNumObservers();
    if(nSourceRays == 0 || nSourceRays > static_cast<size_t>(INT_MAX))
        throw UniversalError("Flux-source surface requires a valid source-ray count");
    if(cfg.fluxSourceMappingNeighbours >= nSourceRays)
    {
        UniversalError eo("--flux-source.mapping-neighbours must be smaller than the number of "
                          "construction rays, or every ray takes the smallest radius on the sphere");
        eo.addEntry("Mapping neighbours", cfg.fluxSourceMappingNeighbours);
        eo.addEntry("Construction rays", nSourceRays);
        throw eo;
    }

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
    // Channel 0 is always grey (kept for the report). In mg-innermost mode one
    // channel per energy group follows, evaluated with the multigroup opacity
    // at the group's representative energy.
    bool const mgSurface =
        cfg.fluxSourceSurfaceMode == FluxSourceSurfaceMode::MultigroupInnermost;
    std::vector<ThermalizationProbeChannel> channels;
    {
        OpacityCalculator const& grey = *runtime.greyOpacity;
        channels.push_back(ThermalizationProbeChannel{
            "grey",
            [&grey](ComputationalCell3D const& cell) {
                return GreyChannelOpacity(grey, cell);
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
                    return GroupChannelOpacity(mg, cell, energy);
                }});
        }
    }
#else
    if(mgSurface && runtime.rank == 0)
        std::cout << "FLUX_SOURCE_SURFACE note: single-group build, mg-innermost equals grey"
                  << std::endl;
#endif
    // The neighbour mapping's floor: where tau_eff reaches K x the target.
    double const mappingFloorTau = cfg.fluxSourceMappingNeighbours > 0 &&
                                   cfg.fluxSourceMappingMaxDepthFactor > 0.0
        ? cfg.fluxSourceMappingMaxDepthFactor * cfg.fluxSourceThermalizationTau : 0.0;
    auto physics = std::make_shared<ThermalizationProbePhysics>(
        runtime.tess, boundary, runtime.cells, std::move(channels),
        *runtime.greyOpacity, cfg.center, cfg.fluxSourceThermalizationTau,
        mappingFloorTau, nSourceRays);
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

    // Keep the global number of live probe particles below the default
    // 500-slot RDMA peer buffer.  The probe creates no secondary particles,
    // so a 400-particle global batch cannot require remote-handler
    // reallocation.  Each ray launches one particle per channel.
    size_t constexpr probeParticleLimit = 400;
    size_t const channelCount = physics->channelCount();
    size_t const probeBatchSize = std::max<size_t>(1, probeParticleLimit / channelCount);
    size_t const batchCount =
        (nSourceRays + probeBatchSize - 1) / probeBatchSize;
    for(size_t batchBegin = 0; batchBegin < nSourceRays;
        batchBegin += probeBatchSize)
    {
        size_t const batchEnd = std::min(
            nSourceRays, batchBegin + probeBatchSize);
        std::vector<ThermalizationProbePhysics::Particle> particles;
        particles.reserve(
            channelCount * (batchEnd - batchBegin) / std::max(1, runtime.mpiSize) + channelCount);

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
            for(size_t channel = 0; channel < channelCount; ++channel)
            {
                ThermalizationProbePhysics::Particle particle;
                particle.id = rayIndex * channelCount + channel;
                particle.location = spherePoint;
                particle.velocity = -1.0 * direction;
                particle.cellIndex = cellIndex;
                particle.cellID = runtime.cells[cellIndex].ID;
                particle.sourceCellID = particle.cellID;
                particle.weight = 0.0;
                particle.initialWeight = 0.0;
                particles.push_back(particle);
            }
        }

        unsigned long long localLaunched =
            static_cast<unsigned long long>(particles.size());
        unsigned long long globalLaunched = localLaunched;
#ifdef RICH_MPI
        MPI_Allreduce(&localLaunched, &globalLaunched, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
        size_t const batchNumber = batchBegin / probeBatchSize + 1;
        if(globalLaunched > static_cast<unsigned long long>(probeParticleLimit) ||
           globalLaunched > static_cast<unsigned long long>(channelCount * (batchEnd - batchBegin)))
        {
            UniversalError eo(
                "Flux-source probe batch exceeded its global particle limit");
            eo.addEntry("Batch", batchNumber);
            eo.addEntry("Batch begin", batchBegin);
            eo.addEntry("Batch end", batchEnd);
            eo.addEntry("Channels", channelCount);
            eo.addEntry("Global launched particles", globalLaunched);
            eo.addEntry("Probe particle limit", probeParticleLimit);
            throw eo;
        }
        if(runtime.rank == 0)
            std::cout << "FLUX_SOURCE_PROBE_BATCH batch="
                      << batchNumber << "/" << batchCount
                      << " range=[" << batchBegin << "," << batchEnd << ")"
                      << " global_particles=" << globalLaunched
                      << " (" << channelCount << " channels per ray)"
                      << " limit=" << probeParticleLimit
                      << std::endl;

        if(globalLaunched > 0)
        {
            manager->getParticles() = std::move(particles);
            manager->step(runtime.cells, 2.01 * cfg.radius);
        }
    }

    // Each crossing was recorded once, on the rank where it happened; a
    // max-reduction assembles every channel's radii (unresolved entries stay
    // at -1 / 0).
    std::vector<std::vector<double>> channelRadius(channelCount);
    std::vector<std::vector<int>> channelValid(channelCount);
    std::vector<std::vector<double>> channelPhotosphere(channelCount);
    std::vector<std::vector<int>> channelPhotosphereValid(channelCount);
    std::vector<std::vector<double>> channelDepth(channelCount);
    std::vector<std::vector<int>> channelDepthValid(channelCount);
    for(size_t channel = 0; channel < channelCount; ++channel)
    {
        channelRadius[channel] = physics->radius(channel);
        channelValid[channel] = physics->valid(channel);
        channelPhotosphere[channel] = physics->photosphereRadius(channel);
        channelPhotosphereValid[channel] = physics->photosphereValid(channel);
        channelDepth[channel] = physics->depthRadius(channel);
        channelDepthValid[channel] = physics->depthValid(channel);
#ifdef RICH_MPI
        if(!channelRadius[channel].empty())
        {
            int const count = static_cast<int>(channelRadius[channel].size());
            MPI_Allreduce(MPI_IN_PLACE, channelRadius[channel].data(), count,
                          MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, channelValid[channel].data(), count,
                          MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, channelPhotosphere[channel].data(), count,
                          MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, channelPhotosphereValid[channel].data(), count,
                          MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, channelDepth[channel].data(), count,
                          MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, channelDepthValid[channel].data(), count,
                          MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        }
#endif
    }
    runtime.fluxSourceGreyRadius = channelRadius[0];
    runtime.fluxSourceGreyResolved = channelValid[0];
    runtime.fluxSourceGreyPhotosphere = channelPhotosphere[0];
    runtime.fluxSourceGreyPhotosphereResolved = channelPhotosphereValid[0];
    runtime.fluxSourceGroupRadius.clear();
    runtime.fluxSourceGroupResolved.clear();
    runtime.fluxSourceGroupPhotosphere.clear();
    runtime.fluxSourceGroupPhotosphereResolved.clear();

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
        runtime.fluxSourceGroupPhotosphere.assign(channelPhotosphere.begin() + 1, channelPhotosphere.end());
        runtime.fluxSourceGroupPhotosphereResolved.assign(
            channelPhotosphereValid.begin() + 1, channelPhotosphereValid.end());
    }

    // Per ray, the deepest photosphere of the channels the surface is built
    // from: the grey one, or the innermost resolved group photosphere. A group
    // whose total depth never reaches 2/3 along the ray is optically thin
    // there and has no photosphere to compare with.
    size_t const firstSurfaceChannel = (!mgSurface || channelCount == 1) ? 0 : 1;
    runtime.fluxSourceSurfacePhotosphere.assign(nSourceRays, -1.0);
    for(size_t i = 0; i < nSourceRays; ++i)
    {
        double deepest = std::numeric_limits<double>::infinity();
        for(size_t channel = firstSurfaceChannel; channel < channelCount; ++channel)
            if(channelPhotosphereValid[channel][i])
                deepest = std::min(deepest, channelPhotosphere[channel][i]);
        if(std::isfinite(deepest))
            runtime.fluxSourceSurfacePhotosphere[i] = deepest;
    }

    // Per ray, the floor of the neighbour mapping: the radius where every
    // channel the surface is built from (and resolved on this ray) has
    // reached tau_eff = K x target, the deepest of them. No floor (0) when the
    // floor is off or such a channel never gets that deep along the ray.
    runtime.fluxSourceMappingFloorTau = mappingFloorTau;
    runtime.fluxSourceMappingFloor.assign(nSourceRays, 0.0);
    if(mappingFloorTau > 0.0)
    {
        for(size_t i = 0; i < nSourceRays; ++i)
        {
            double floor = std::numeric_limits<double>::infinity();
            for(size_t channel = firstSurfaceChannel; channel < channelCount; ++channel)
            {
                if(!channelValid[channel][i])
                    continue;
                floor = channelDepthValid[channel][i] ? std::min(floor, channelDepth[channel][i]) : 0.0;
                if(!(floor > 0.0))
                    break;
            }
            runtime.fluxSourceMappingFloor[i] = std::isfinite(floor) ? floor : 0.0;
        }
    }

    if(runtime.rank == 0)
    {
        for(size_t channel = 0; channel < channelCount; ++channel)
        {
            RadiusSummary const summary = SummarizeRadii(channelRadius[channel], channelValid[channel]);
            RadiusSummary const photosphere =
                SummarizeRadii(channelPhotosphere[channel], channelPhotosphereValid[channel]);
            std::cout << "FLUX_SOURCE_CHANNEL name=" << physics->channelName(channel)
                      << " resolved=" << summary.resolved << "/" << nSourceRays
                      << " radius_p05/med/p95=" << summary.p05 << "/" << summary.median
                      << "/" << summary.p95 << " cm"
                      << " photosphere_resolved=" << photosphere.resolved << "/" << nSourceRays
                      << " photosphere_p05/med/p95=" << photosphere.p05 << "/" << photosphere.median
                      << "/" << photosphere.p95 << " cm" << std::endl;
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

    // Smoothing: no direction may dip far below its neighbours. A raised
    // direction no longer sits at the target depth, so say so loudly. With
    // the photosphere cap it is never raised above its own photosphere.
    runtime.fluxSourceRawRadius = runtime.fluxSourceRadius;
    // The nearest resolved directions of every direction, once, for the
    // smoothing and the neighbour mapping (the resolved set is the same).
    std::vector<std::vector<size_t>> const neighbours = NearestResolvedNeighbours(
        directions, runtime.fluxSourceRadiusDirectlyResolved,
        std::max(kSmoothingNeighbours, cfg.fluxSourceMappingNeighbours));
    std::vector<double> const noCeiling;
    runtime.fluxSourceSmoothedDirections = SmoothSurfaceDips(
        neighbours, runtime.fluxSourceRadius,
        runtime.fluxSourceRadiusDirectlyResolved, cfg.fluxSourceSmoothingFloor,
        cfg.fluxSourceSmoothingPhotosphereCap ? runtime.fluxSourceSurfacePhotosphere : noCeiling,
        runtime.fluxSourceCappedDirections);
    if(runtime.rank == 0 && runtime.fluxSourceSmoothedDirections.empty() &&
       !runtime.fluxSourceCappedDirections.empty())
        std::cout << "FLUX_SOURCE_SMOOTHING note: no direction raised; "
                  << runtime.fluxSourceCappedDirections.size() << "/" << nSourceRays
                  << " dips held by their own photosphere (or without one)" << std::endl;
    if(runtime.rank == 0 && !runtime.fluxSourceSmoothedDirections.empty())
    {
        std::pair<double, double> const factor = MedianAndMaxRatio(
            runtime.fluxSourceSmoothedDirections, runtime.fluxSourceRadius,
            runtime.fluxSourceRawRadius);
        std::ostringstream warning;
        warning << "FLUX_SOURCE_SMOOTHING WARNING: raised the source surface in "
                << runtime.fluxSourceSmoothedDirections.size() << "/" << nSourceRays
                << " directions toward " << cfg.fluxSourceSmoothingFloor
                << " x the median radius of their " << kSmoothingNeighbours
                << " nearest directions (radius raised by median x" << factor.first
                << ", max x" << factor.second << ")";
        if(cfg.fluxSourceSmoothingPhotosphereCap)
            warning << "; " << runtime.fluxSourceCappedDirections.size()
                    << " directions held below that value by their own photosphere (or not"
                    << " raised, having none)";
        warning << ". In raised directions the surface is shallower than tau_eff="
                << cfg.fluxSourceThermalizationTau
                << ", so the gas between the old and new radius is transported instead of"
                << " treated as thermalized. Set --flux-source.smoothing-floor 0 to disable.";
        std::cout << warning.str() << std::endl;
        std::cerr << warning.str() << std::endl;
    }

    // Neighbour mapping: a cell takes the radius of its nearest direction, so
    // lowering each direction to the smallest radius among itself and its
    // neighbours keeps every cell between directions below all of them.
    runtime.fluxSourceSmoothedRadius = runtime.fluxSourceRadius;
    runtime.fluxSourceMappingNeighbours = cfg.fluxSourceMappingNeighbours;
    std::vector<double> const noFloor;
    runtime.fluxSourceMappedDirections = ApplyNeighbourMapping(
        neighbours, runtime.fluxSourceRadius,
        runtime.fluxSourceRadiusDirectlyResolved, cfg.fluxSourceMappingNeighbours,
        mappingFloorTau > 0.0 ? runtime.fluxSourceMappingFloor : noFloor,
        runtime.fluxSourceFloorHeldDirections);
    if(runtime.rank == 0 && cfg.fluxSourceMappingNeighbours > 0)
    {
        std::pair<double, double> const factor = MedianAndMaxRatio(
            runtime.fluxSourceMappedDirections, runtime.fluxSourceSmoothedRadius,
            runtime.fluxSourceRadius);
        RadiusSummary const mapped = SummarizeRadii(
            runtime.fluxSourceRadius, runtime.fluxSourceRadiusDirectlyResolved);
        std::cout << "FLUX_SOURCE_MAPPING neighbours=" << cfg.fluxSourceMappingNeighbours
                  << " lowered=" << runtime.fluxSourceMappedDirections.size() << "/" << nSourceRays
                  << " radius_lowered_by_median/max=x" << factor.first << "/x" << factor.second
                  << " depth_floor_tau_eff=" << mappingFloorTau
                  << " held_by_depth_floor=" << runtime.fluxSourceFloorHeldDirections.size()
                  << "/" << nSourceRays
                  << " surface_radius_p05/med/p95=" << mapped.p05 << "/" << mapped.median
                  << "/" << mapped.p95 << " cm" << std::endl;
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

    // The surface as the code built it, written before any transport so even
    // a cancelled run leaves it behind.
    runtime.fluxSourceSurfaceFile =
        InsertSuffixBeforeExtension(cfg.outputPath, ".source-surface");
    if(runtime.rank == 0)
    {
        HDF5Writer writer(runtime.fluxSourceSurfaceFile);
        std::vector<double> dx, dy, dz;
        for(Vector3D const& d : directions)
        {
            dx.push_back(d.x);
            dy.push_back(d.y);
            dz.push_back(d.z);
        }
        state_hdf5::WriteVector(writer, "/directions/x", dx);
        state_hdf5::WriteVector(writer, "/directions/y", dy);
        state_hdf5::WriteVector(writer, "/directions/z", dz);
        writer.WriteElement("/thermalization_tau", cfg.fluxSourceThermalizationTau);
        writer.WriteElement("/photosphere_tau", kPhotosphereTau);
        writer.WriteElement("/smoothing_floor", cfg.fluxSourceSmoothingFloor);
        writer.WriteElement("/smoothing_photosphere_cap", cfg.fluxSourceSmoothingPhotosphereCap ? 1 : 0);
        writer.WriteElement("/mapping_neighbours", cfg.fluxSourceMappingNeighbours);
        writer.WriteElement("/mapping_floor_tau", runtime.fluxSourceMappingFloorTau);
        writer.WriteElement("/surface_mode", std::string(mgSurface ? "mg-innermost" : "grey"));
        // radius: the final surface (after smoothing and the neighbour
        // mapping); radius_before_mapping: after smoothing only;
        // radius_before_smoothing: as the probe found it.
        state_hdf5::WriteVector(writer, "/surface/radius", runtime.fluxSourceRadius);
        state_hdf5::WriteVector(writer, "/surface/radius_before_mapping", runtime.fluxSourceSmoothedRadius);
        state_hdf5::WriteVector(writer, "/surface/radius_before_smoothing", runtime.fluxSourceRawRadius);
        state_hdf5::WriteVector(writer, "/surface/resolved", runtime.fluxSourceRadiusDirectlyResolved);
        state_hdf5::WriteVector(writer, "/surface/smoothed_directions", runtime.fluxSourceSmoothedDirections);
        state_hdf5::WriteVector(writer, "/surface/capped_directions", runtime.fluxSourceCappedDirections);
        state_hdf5::WriteVector(writer, "/surface/mapped_directions", runtime.fluxSourceMappedDirections);
        state_hdf5::WriteVector(writer, "/surface/mapping_floor_radius", runtime.fluxSourceMappingFloor);
        state_hdf5::WriteVector(writer, "/surface/floor_held_directions", runtime.fluxSourceFloorHeldDirections);
        state_hdf5::WriteVector(writer, "/surface/photosphere_radius", runtime.fluxSourceSurfacePhotosphere);
        state_hdf5::WriteVector(writer, "/grey/radius", runtime.fluxSourceGreyRadius);
        state_hdf5::WriteVector(writer, "/grey/resolved", runtime.fluxSourceGreyResolved);
        state_hdf5::WriteVector(writer, "/grey/photosphere_radius", runtime.fluxSourceGreyPhotosphere);
        state_hdf5::WriteVector(writer, "/grey/photosphere_resolved", runtime.fluxSourceGreyPhotosphereResolved);
        state_hdf5::WriteMatrix(writer, "/groups/radius", runtime.fluxSourceGroupRadius);
        state_hdf5::WriteMatrix(writer, "/groups/resolved", runtime.fluxSourceGroupResolved);
        state_hdf5::WriteMatrix(writer, "/groups/photosphere_radius", runtime.fluxSourceGroupPhotosphere);
        state_hdf5::WriteMatrix(writer, "/groups/photosphere_resolved", runtime.fluxSourceGroupPhotosphereResolved);
        writer.Close();
        std::cout << "FLUX_SOURCE_SURFACE written to " << runtime.fluxSourceSurfaceFile << std::endl;
    }
}

void AttachEmissionSurfaceToObserver(
    Config const& cfg,
    PostprocessRuntime& runtime)
{
    if(!runtime.fluxSourceEnabled || !runtime.observer)
        throw UniversalError("Emission surface requested before the flux-source surface was built");
    std::vector<Vector3D> const& observerDirections = runtime.observer->getDirections();
    size_t const nObservers = observerDirections.size();
    size_t const groupCount = runtime.fluxSourceGroupRadius.size();
    SphericalObserver::PhotosphereData data = runtime.observer->getPhotosphereData();
    data.emissionSurfaceTau = runtime.fluxSourceTau;
    data.emissionSurfaceRadius.assign(nObservers, -1.0);
    data.emissionSurfaceGreyRadius.assign(nObservers, -1.0);
    data.emissionSurfaceGroupRadius.assign(
        groupCount > 0 ? nObservers : 0, std::vector<double>(groupCount, -1.0));
    for(size_t obs = 0; obs < nObservers; ++obs)
    {
        size_t const ray = NearestObserverDirection(
            cfg.center + observerDirections[obs], cfg.center, runtime.fluxSourceDirections);
        if(runtime.fluxSourceRadiusDirectlyResolved[ray])
            data.emissionSurfaceRadius[obs] = runtime.fluxSourceRadius[ray];
        if(runtime.fluxSourceGreyResolved[ray])
            data.emissionSurfaceGreyRadius[obs] = runtime.fluxSourceGreyRadius[ray];
        for(size_t g = 0; g < groupCount; ++g)
            if(runtime.fluxSourceGroupResolved[g][ray])
                data.emissionSurfaceGroupRadius[obs][g] = runtime.fluxSourceGroupRadius[g][ray];
    }
    runtime.observer->setPhotosphereData(std::move(data));
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
        physics.setPostProcessExternalSourceSpectrumTemperatures({});
        runtime.volumeEmissionCellLuminosity.clear();
        runtime.fluxSourceCellLuminosity.clear();
        return;
    }
    if(!runtime.fluxSourceEnabled || runtime.fluxSourceRadius.empty())
        throw UniversalError("Flux-source surface was not initialized");

    std::vector<Vector3D> const fldFlux = ComputeGreyFldFlux(runtime);
    // The construction ray of every point, once for the surface mask, the
    // photosphere check and the per-group volume-emission masks.
    std::vector<size_t> const nearestDirection = BuildNearestDirectionIndex(cfg, runtime);
    std::vector<unsigned char> outside = BuildOutsideMaskForRadii(
        cfg, runtime, nearestDirection, runtime.fluxSourceRadius,
        runtime.fluxSourceRadiusDirectlyResolved);
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

    // Enclosed outside regions (packet traps) become inside. Found once by a
    // connectivity fill; every later decomposition applies the same cell IDs.
    if(cfg.fluxSourceRemoveEnclosedOutside)
    {
        if(!runtime.fluxSourceEnclosedComputed)
        {
            std::vector<size_t> const enclosed = FindEnclosedOutsideCells(cfg, runtime, outside);
            // Positive face luminosity the walls of these cells would inject.
            double removedLuminosity = 0.0;
            std::vector<size_t> faceNeighbors;
            std::vector<double> localIDs, localX, localY, localZ;
            for(size_t i : enclosed)
            {
                Vector3D const p = runtime.tess.GetMeshPoint(i);
                localIDs.push_back(static_cast<double>(runtime.cells[i].ID));
                localX.push_back(p.x);
                localY.push_back(p.y);
                localZ.push_back(p.z);
                auto const& cellFaces = runtime.tess.GetCellFaces(i);
                runtime.tess.GetNeighbors(i, faceNeighbors);
                for(size_t j = 0; j < faceNeighbors.size() && j < cellFaces.size(); ++j)
                {
                    size_t const n = faceNeighbors[j];
                    if(runtime.tess.IsPointOutsideBox(n) || n >= outside.size() ||
                       n >= fldFlux.size() || outside[n] != 0)
                        continue;
                    Vector3D normal = p - runtime.tess.GetMeshPoint(n);
                    double const distance = abs(normal);
                    if(!(distance > 0.0))
                        continue;
                    normal *= 1.0 / distance;
                    double const flux = ScalarProd(0.5 * (fldFlux[i] + fldFlux[n]), normal) *
                        runtime.tess.GetArea(cellFaces[j]);
                    removedLuminosity += std::max(0.0, flux);
                }
            }
            std::vector<double> allIDs = localIDs, allX = localX, allY = localY, allZ = localZ;
#ifdef RICH_MPI
            MPI_Allreduce(MPI_IN_PLACE, &removedLuminosity, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            {
                // Every rank needs the IDs; rank 0 also the positions.
                int const count = static_cast<int>(localIDs.size());
                std::vector<int> counts(static_cast<size_t>(runtime.mpiSize), 0);
                MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
                std::vector<int> displs(counts.size(), 0);
                int total = 0;
                for(size_t r = 0; r < counts.size(); ++r)
                {
                    displs[r] = total;
                    total += counts[r];
                }
                allIDs.assign(static_cast<size_t>(total), 0.0);
                MPI_Allgatherv(localIDs.data(), count, MPI_DOUBLE, allIDs.data(),
                               counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
                allX = GatherToRankZero(localX, MPI_DOUBLE, runtime.rank);
                allY = GatherToRankZero(localY, MPI_DOUBLE, runtime.rank);
                allZ = GatherToRankZero(localZ, MPI_DOUBLE, runtime.rank);
            }
#endif
            runtime.fluxSourceEnclosedCellIDs.clear();
            for(double id : allIDs)
                runtime.fluxSourceEnclosedCellIDs.insert(static_cast<size_t>(id));
            runtime.fluxSourceEnclosedLuminosity = removedLuminosity;
            runtime.fluxSourceEnclosedComputed = true;
            if(runtime.rank == 0)
            {
                std::cout << "FLUX_SOURCE_ENCLOSED removed_cells=" << allIDs.size()
                          << " wall_luminosity_removed=" << removedLuminosity << " erg/s"
                          << " (outside cells enclosed by the source surface; their walls"
                          << " would return every packet into them)" << std::endl;
                if(!runtime.fluxSourceSurfaceFile.empty())
                {
                    std::vector<size_t> ids;
                    for(double id : allIDs)
                        ids.push_back(static_cast<size_t>(id));
                    HDF5Writer writer(runtime.fluxSourceSurfaceFile, false);
                    writer.WriteElement("/enclosed/removed_cells", ids.size());
                    writer.WriteElement("/enclosed/wall_luminosity_removed", removedLuminosity);
                    state_hdf5::WriteVector(writer, "/enclosed/cell_ids", ids);
                    state_hdf5::WriteVector(writer, "/enclosed/x", allX);
                    state_hdf5::WriteVector(writer, "/enclosed/y", allY);
                    state_hdf5::WriteVector(writer, "/enclosed/z", allZ);
                    writer.Close();
                }
            }
        }
        // Local and ghost points alike, by stable cell ID.
        for(size_t p = 0; p < outside.size() && p < pointCellIDs.size(); ++p)
            if(outside[p] && runtime.fluxSourceEnclosedCellIDs.count(pointCellIDs[p]))
                outside[p] = 0;
    }

    std::vector<RadiationIMC::PostProcessExternalSource> sources;
    runtime.fluxSourceCellLuminosity.assign(nCells, 0.0);
    std::vector<ComputationalCell3D> const pointCells =
        cfg.fluxSourceFaceFlux == FluxSourceFaceFlux::Hydro
            ? PointCellsWithGhosts(runtime) : std::vector<ComputationalCell3D>();
    // Spectrum temperature of each source (outer) cell, by cell ID, and the
    // face-luminosity weighted diagnostics of the FLUX_SOURCE_SPECTRUM line.
    std::unordered_map<size_t, double> spectrumTemperature;
    double localHotLuminosity = 0.0;      // faces with T_gas > 10 T_rad
    double localLogRadiationT = 0.0;      // sum L ln T_rad
    double localLogGasT = 0.0;            // sum L ln T_gas
    std::vector<size_t> neighbors;
    double localLuminosity = 0.0;
    double localNetLuminosity = 0.0;
    double localInwardLuminosity = 0.0;
    uint64_t localBoundaryFaces = 0;
    uint64_t localEmittingFaces = 0;
    // Photosphere check of the first setup (later setups only repartition the
    // same faces): each face against the photosphere of the construction ray
    // its outer cell is mapped to, grey (entry 0), per group, and the deepest
    // one. A face is either above a photosphere that exists, or on a ray
    // where the channel never reaches total depth 2/3 (transparent).
    bool const checkPhotosphere = !runtime.fluxSourcePhotosphereCheck.computed &&
        runtime.fluxSourceSurfacePhotosphere.size() == runtime.fluxSourceDirections.size();
    size_t const checkChannels = 2 + runtime.fluxSourceGroupPhotosphere.size();  // deepest, grey, groups
    std::vector<uint64_t> checkFaces(2 * checkChannels, 0);      // [above | transparent] per channel
    std::vector<double> checkLuminosity(2 * checkChannels, 0.0);
    auto checkFace = [&](size_t channel, double faceRadius, double photosphere, bool resolved,
                         double luminosity) {
        size_t const slot = resolved ? channel : checkChannels + channel;
        if(!resolved || faceRadius > photosphere)
        {
            ++checkFaces[slot];
            checkLuminosity[slot] += luminosity;
        }
    };

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

            double signedLuminosity = 0.0;
            if(cfg.fluxSourceFaceFlux == FluxSourceFaceFlux::Hydro)
                signedLuminosity = HydroFldFaceLuminosity(runtime, pointCells, innerCell, outerCell, faces[j]);
            else
            {
                Vector3D const faceFlux = 0.5 *
                    (fldFlux[outerCell] + fldFlux[innerCell]);
                signedLuminosity = ScalarProd(faceFlux, normal) * area;
            }
            if(!std::isfinite(signedLuminosity))
                throw UniversalError("Flux-source surface has non-finite face flux");
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
            runtime.fluxSourceCellLuminosity[outerCell] += luminosity;
            {
                ComputationalCell3D const& outer = runtime.cells[outerCell];
                double const radiationT = std::pow(
                    outer.density * outer.Erad / CG::radiation_constant, 0.25);
                if(!(radiationT > 0.0) || !std::isfinite(radiationT) ||
                   !(outer.temperature > 0.0) || !std::isfinite(outer.temperature))
                {
                    UniversalError eo("Flux-source cell has no valid radiation or gas temperature");
                    eo.addEntry("Cell ID", outer.ID);
                    eo.addEntry("Radiation energy density", outer.density * outer.Erad);
                    eo.addEntry("Gas temperature", outer.temperature);
                    throw eo;
                }
                spectrumTemperature[outer.ID] = radiationT;
                if(outer.temperature > 10.0 * radiationT)
                    localHotLuminosity += luminosity;
                localLogRadiationT += luminosity * std::log(radiationT);
                localLogGasT += luminosity * std::log(outer.temperature);
            }
            ++localBoundaryFaces;
            if(luminosity > 0.0)
            {
                localLuminosity += luminosity;
                ++localEmittingFaces;
            }
            size_t const ray = outerCell < nearestDirection.size()
                ? nearestDirection[outerCell] : std::numeric_limits<size_t>::max();
            if(checkPhotosphere && ray < runtime.fluxSourceDirections.size())
            {
                double const faceRadius = fastabs(source.location - cfg.center);
                double const deepest = runtime.fluxSourceSurfacePhotosphere[ray];
                checkFace(0, faceRadius, deepest, deepest > 0.0, luminosity);
                checkFace(1, faceRadius, runtime.fluxSourceGreyPhotosphere[ray],
                          runtime.fluxSourceGreyPhotosphereResolved[ray] != 0, luminosity);
                for(size_t g = 0; g + 2 < checkChannels; ++g)
                    checkFace(2 + g, faceRadius, runtime.fluxSourceGroupPhotosphere[g][ray],
                              runtime.fluxSourceGroupPhotosphereResolved[g][ray] != 0, luminosity);
            }
        }
    }

    double spectrumSums[3] = {localHotLuminosity, localLogRadiationT, localLogGasT};
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, spectrumSums, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
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

    if(checkPhotosphere)
    {
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, checkFaces.data(), static_cast<int>(checkFaces.size()),
                      MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, checkLuminosity.data(), static_cast<int>(checkLuminosity.size()),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        auto& check = runtime.fluxSourcePhotosphereCheck;
        check.computed = true;
        check.faces = globalBoundaryFaces;
        check.luminosity = rawGlobalLuminosity;
        check.facesAboveDeepest = checkFaces[0];
        check.luminosityAboveDeepest = checkLuminosity[0];
        check.facesTransparentDeepest = checkFaces[checkChannels];
        check.luminosityTransparentDeepest = checkLuminosity[checkChannels];
        // Per channel without the deepest entry: grey, then the groups.
        check.facesAbove.assign(checkFaces.begin() + 1, checkFaces.begin() + checkChannels);
        check.luminosityAbove.assign(checkLuminosity.begin() + 1, checkLuminosity.begin() + checkChannels);
        check.facesTransparent.assign(checkFaces.begin() + checkChannels + 1, checkFaces.end());
        check.luminosityTransparent.assign(checkLuminosity.begin() + checkChannels + 1, checkLuminosity.end());
        if(runtime.rank == 0)
        {
            auto share = [&](double luminosity) {
                return rawGlobalLuminosity > 0.0 ? luminosity / rawGlobalLuminosity : 0.0;
            };
            std::ostringstream line;
            line << "FLUX_SOURCE_PHOTOSPHERE_CHECK boundary_faces=" << globalBoundaryFaces
                 << " face_luminosity=" << rawGlobalLuminosity << " erg/s"
                 << " (faces above the channel's photosphere on their ray / faces on rays where"
                 << " the channel is transparent; luminosity shares in brackets)"
                 << " deepest: " << check.facesAboveDeepest << " (" << share(check.luminosityAboveDeepest)
                 << ") / " << check.facesTransparentDeepest << " (" << share(check.luminosityTransparentDeepest) << ")";
            for(size_t c = 0; c < check.facesAbove.size(); ++c)
                line << " | " << (c == 0 ? std::string("grey") : "group" + std::to_string(c - 1)) << ": "
                     << check.facesAbove[c] << " (" << share(check.luminosityAbove[c]) << ") / "
                     << check.facesTransparent[c] << " (" << share(check.luminosityTransparent[c]) << ")";
            std::cout << line.str() << std::endl;
            if(!runtime.fluxSourceSurfaceFile.empty())
            {
                HDF5Writer writer(runtime.fluxSourceSurfaceFile, false);
                writer.WriteElement("/photosphere_check/boundary_faces", check.faces);
                writer.WriteElement("/photosphere_check/face_luminosity", check.luminosity);
                writer.WriteElement("/photosphere_check/faces_above_deepest", check.facesAboveDeepest);
                writer.WriteElement("/photosphere_check/luminosity_above_deepest", check.luminosityAboveDeepest);
                writer.WriteElement("/photosphere_check/faces_transparent_deepest", check.facesTransparentDeepest);
                writer.WriteElement("/photosphere_check/luminosity_transparent_deepest",
                                    check.luminosityTransparentDeepest);
                state_hdf5::WriteVector(writer, "/photosphere_check/faces_above", check.facesAbove);
                state_hdf5::WriteVector(writer, "/photosphere_check/luminosity_above", check.luminosityAbove);
                state_hdf5::WriteVector(writer, "/photosphere_check/faces_transparent", check.facesTransparent);
                state_hdf5::WriteVector(writer, "/photosphere_check/luminosity_transparent",
                                        check.luminosityTransparent);
                writer.Close();
            }
        }
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
    // Multigroup face and re-emitted packets take a Planck spectrum at this
    // temperature (an empty map keeps the cell's gas temperature).
    physics.setPostProcessExternalSourceSpectrumTemperatures(
        cfg.fluxSourceSpectrumTemperature == FluxSourceSpectrumTemperature::Radiation
            ? std::move(spectrumTemperature) : std::unordered_map<size_t, double>());

    if(cfg.volumeEmissionEnabled)
    {
        // Per group, a cell outside that group's thermalization surface has
        // Fleck factor 1 (real absorption and emission); inside it has 0
        // (effective scattering). The grey pass, and the grey surface mode,
        // use one surface for every group. Cells inside the deepest surface
        // therefore emit nothing and conserve energy exactly.
        size_t const groupCount = multigroupPass ? static_cast<size_t>(ENERGY_GROUPS_NUM) : 1;
        std::vector<size_t> const& nearest = nearestDirection;
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
            // A group surface is never inside the deepest one, so its mask is
            // a subset of the uncleaned one; drop the enclosed cells from it.
            for(size_t i = 0; i < outsideGroup[g].size() && i < outside.size(); ++i)
                outsideGroup[g][i] = outsideGroup[g][i] && outside[i];
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
        runtime.volumeEmissionCellLuminosity.assign(nCells, 0.0);
        for(size_t i = 0; i < nCells; ++i)
            if(mask[i])
                runtime.volumeEmissionCellLuminosity[i] = cellLuminosity[i];
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
        runtime.volumeEmissionCellLuminosity.clear();
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
                  << " face_flux="
                  << (cfg.fluxSourceFaceFlux == FluxSourceFaceFlux::Hydro ? "hydro" : "cell-average")
                  << " erg/s" << std::endl;
    if(runtime.rank == 0 && rawGlobalLuminosity > 0.0)
        std::cout << "FLUX_SOURCE_SPECTRUM temperature="
                  << (cfg.fluxSourceSpectrumTemperature == FluxSourceSpectrumTemperature::Radiation
                      ? "radiation" : "gas")
                  << " luminosity_share_Tgas_gt_10_Trad=" << spectrumSums[0] / rawGlobalLuminosity
                  << " L_weighted_geomean_Trad=" << std::exp(spectrumSums[1] / rawGlobalLuminosity)
                  << " L_weighted_geomean_Tgas=" << std::exp(spectrumSums[2] / rawGlobalLuminosity)
                  << " K" << std::endl;
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
