#include "PhotosphereCalculation.hpp"

#include "source/monte/physics/MonteCarloPhysics.hpp"
#include "source/monte/particle/StepResult.hpp"
#include "source/monte/particle/ParticleStatus.hpp"
#ifdef RICH_MPI
#include "source/monte/manager/communication/RDMACommunicationEngine.hpp"
#endif // RICH_MPI

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace imc_postprocess_tde {
namespace {

constexpr double kPhotosphereTau = 2.0 / 3.0;
constexpr double kThermalizationTau = 1.0;

enum class ChannelKind
{
    MgGroupTotal,
    MgGroupThermalization,
    MgIntegratedTotal,
    MgIntegratedThermalization,
    GreyTotal,
    GreyThermalization
};

struct ProbeChannel
{
    ChannelKind kind = ChannelKind::MgIntegratedTotal;
    size_t group = 0;
};

struct ProbePassResult
{
    std::vector<double> value;
    std::vector<int> valid;
};

struct SigmaPair
{
    double total = 0.0;
    double thermalization = 0.0;
};

bool IsTotalChannel(ChannelKind kind)
{
    return kind == ChannelKind::MgGroupTotal ||
           kind == ChannelKind::MgIntegratedTotal ||
           kind == ChannelKind::GreyTotal;
}

ChannelKind ThermalizationKind(ChannelKind kind)
{
    switch (kind) {
    case ChannelKind::MgGroupTotal:
        return ChannelKind::MgGroupThermalization;
    case ChannelKind::MgIntegratedTotal:
        return ChannelKind::MgIntegratedThermalization;
    case ChannelKind::GreyTotal:
        return ChannelKind::GreyThermalization;
    case ChannelKind::MgGroupThermalization:
    case ChannelKind::MgIntegratedThermalization:
    case ChannelKind::GreyThermalization:
        return kind;
    }
    return kind;
}

bool SameChannelTarget(ProbeChannel const& a, ProbeChannel const& b)
{
    return a.kind == b.kind && a.group == b.group;
}

std::vector<size_t> BuildThermalPartnerMap(std::vector<ProbeChannel> const& channels)
{
    std::vector<size_t> partners(channels.size(), channels.size());
    for (size_t i = 0; i < channels.size(); ++i) {
        if (!IsTotalChannel(channels[i].kind))
            continue;
        ProbeChannel const want{ThermalizationKind(channels[i].kind), channels[i].group};
        for (size_t j = 0; j < channels.size(); ++j) {
            if (SameChannelTarget(channels[j], want)) {
                partners[i] = j;
                break;
            }
        }
    }
    return partners;
}

std::vector<size_t> BuildRayChannelIndices(std::vector<ProbeChannel> const& channels)
{
    std::vector<size_t> indices;
    indices.reserve(channels.size() / 2 + 1);
    std::vector<size_t> const thermalPartners = BuildThermalPartnerMap(channels);
    for (size_t i = 0; i < channels.size(); ++i) {
        if (IsTotalChannel(channels[i].kind) && thermalPartners[i] < channels.size())
            indices.push_back(i);
    }
    return indices;
}

double SecondsSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

size_t PhotosphereChunkSize(size_t nObs)
{
    if (nObs == 0)
        return 1;
    size_t const targetChunks = std::min<size_t>(32, nObs);
    size_t const byTarget = (nObs + targetChunks - 1) / targetChunks;
    return std::min(nObs, std::max<size_t>(8, byTarget));
}

size_t CountValidRange(std::vector<int> const& valid,
                       size_t obsBegin,
                       size_t obsEnd,
                       size_t channelCount)
{
    size_t count = 0;
    for (size_t obs = obsBegin; obs < obsEnd; ++obs) {
        for (size_t ch = 0; ch < channelCount; ++ch) {
            size_t const flat = obs * channelCount + ch;
            if (flat < valid.size() && valid[flat] != 0)
                ++count;
        }
    }
    return count;
}

size_t CountValid(std::vector<int> const& valid)
{
    return static_cast<size_t>(
        std::count_if(valid.begin(), valid.end(), [](int v) { return v != 0; }));
}

double PlanckGroupWeight(double temperature, double energy)
{
    double const kT = CG::boltzmann_constant * temperature;
    if (!(kT > 0.0) || !(energy > 0.0))
        return 0.0;
    double const x = energy / kT;
    if (!(x > 0.0) || x >= 500.0)
        return 0.0;
    return x * x * x / std::expm1(x);
}

std::pair<double, double> MgIntegratedAbsScat(OpacityCalculator const& opacity,
                                              ComputationalCell3D const& cell)
{
    double weightedAbs = 0.0;
    double weightedScat = 0.0;
    double totalWeight = 0.0;
    size_t const groupCount = opacity.energy_groups_center.size();
    for (size_t g = 0; g < groupCount; ++g) {
        double const nu = opacity.energy_groups_center[g];
        double const w = PlanckGroupWeight(cell.temperature, nu);
        weightedAbs += opacity.CalcAbsorptionOpacity(cell, nu) * w;
        weightedScat += opacity.CalcScatteringOpacity(cell, nu) * w;
        totalWeight += w;
    }
    if (totalWeight > 0.0)
        return {weightedAbs / totalWeight, weightedScat / totalWeight};

    if (groupCount == 0)
        return {0.0, opacity.CalcScatteringOpacity(cell)};

    double absSum = 0.0;
    double scatSum = 0.0;
    for (size_t g = 0; g < groupCount; ++g) {
        double const nu = opacity.energy_groups_center[g];
        absSum += opacity.CalcAbsorptionOpacity(cell, nu);
        scatSum += opacity.CalcScatteringOpacity(cell, nu);
    }
    double const inv = 1.0 / static_cast<double>(groupCount);
    return {absSum * inv, scatSum * inv};
}

double EffectiveSigma(double absorption, double scattering, bool thermalization)
{
    absorption = std::max(0.0, absorption);
    scattering = std::max(0.0, scattering);
    if (!thermalization)
        return absorption + scattering;
    return std::sqrt(std::max(0.0, 3.0 * absorption * (absorption + scattering)));
}

// Observer-inward probes start at the observer sphere surface. Each migrating
// particle carries one total/thermalization pair, records the two threshold
// crossings independently, and stops after both are found or the ray exits.
class PhotosphereProbePhysics : public MonteCarloPhysics<Vector3D, Tessellation3D>
{
public:
    using Particle = MonteCarloParticle<Vector3D>;
    using Functionality = MonteCarloFunctionality;
    using BoundaryCond = BoundaryCondition<Vector3D, Tessellation3D>;

    PhotosphereProbePhysics(Tessellation3D const& grid,
                            std::shared_ptr<BoundaryCond> const& boundary,
                            std::vector<ComputationalCell3D> const& cells,
                            OpacityCalculator const& mgOpacity,
                            OpacityCalculator const& greyOpacity,
                            Vector3D center,
                            double observerRadius,
                            std::vector<ProbeChannel> channels,
                            size_t particleCount)
        : MonteCarloPhysics<Vector3D, Tessellation3D>(grid, boundary),
          cells_(cells),
          mgOpacity_(mgOpacity),
          greyOpacity_(greyOpacity),
          center_(center),
          observerRadius_(observerRadius),
          channels_(std::move(channels)),
          thermalPartners_(BuildThermalPartnerMap(channels_)),
          values_(particleCount, -1.0),
          valid_(particleCount, 0)
    {}

    std::vector<Particle> preStep(double /*fullDt*/) override { return {}; }
    void postStep(std::vector<Particle> const& /*particles*/, double /*fullDt*/) override {}

    Functionality step(Particle& particle,
                       std::vector<Particle>& /*particlesToAdd*/) override
    {
        Functionality functionality;
        functionality.change = MonteCarloParticleStatus::REMOVE;

        if (particle.id >= values_.size() || channels_.empty())
            return functionality;

        size_t const channelIndex = particle.id % channels_.size();
        if (channelIndex >= channels_.size() ||
            channelIndex >= thermalPartners_.size() ||
            !IsTotalChannel(channels_[channelIndex].kind))
            return functionality;
        size_t const thermalChannelIndex = thermalPartners_[channelIndex];
        if (thermalChannelIndex >= channels_.size())
            return functionality;
        size_t const obsOffset = particle.id - channelIndex;
        size_t const totalFlat = particle.id;
        size_t const thermalFlat = obsOffset + thermalChannelIndex;
        if (thermalFlat >= values_.size())
            return functionality;

        double const dirNorm = abs(particle.velocity);
        if (!(dirNorm > 0.0) || !std::isfinite(dirNorm))
            return functionality;
        Vector3D dir = particle.velocity * (1.0 / dirNorm);
        particle.velocity = dir;

        if (particle.cellIndex >= cells_.size())
            return functionality;

        double const radiusNow = fastabs(particle.location - center_);
        if (radiusNow <= 0.0) {
            return functionality;
        }

        auto const [faceIntersect, distanceToFace, nextCellIndex] =
            this->getIntersectionDetails(particle);
        (void)faceIntersect;

        double const ds = distanceToFace;
        if (!(ds >= 0.0) || !std::isfinite(ds))
            return functionality;

        ProbeChannel const& channel = channels_[channelIndex];
        SigmaPair const sigma = sigmasFor(channel, cells_[particle.cellIndex]);
        double const oldTotalTau = particle.weight;
        double const oldThermalTau = particle.initialWeight;
        double const newTotalTau = oldTotalTau + sigma.total * ds;
        double const newThermalTau = oldThermalTau + sigma.thermalization * ds;

        if (oldTotalTau < kPhotosphereTau && newTotalTau >= kPhotosphereTau) {
            double const radius =
                crossingRadius(particle.location, dir, ds, sigma.total, oldTotalTau,
                               kPhotosphereTau);
            record(totalFlat, radius, true);
        }

        if (oldThermalTau < kThermalizationTau && newThermalTau >= kThermalizationTau) {
            double const radius =
                crossingRadius(particle.location, dir, ds, sigma.thermalization,
                               oldThermalTau, kThermalizationTau);
            record(thermalFlat, radius, true);
        }

        particle.weight = newTotalTau;
        particle.initialWeight = newThermalTau;
        if (newTotalTau >= kPhotosphereTau && newThermalTau >= kThermalizationTau)
            return functionality;

        particle.location += dir * ds;
        particle.timeLeft -= ds;

        bool const leavingDomain = this->grid.IsPointOutsideBox(nextCellIndex);
        if (leavingDomain) {
            return functionality;
        }

        functionality.change = MonteCarloParticleStatus::CELL_MOVE;
        functionality.nextCellIndex = nextCellIndex;
        return functionality;
    }

    std::vector<double> const& values() const { return values_; }
    std::vector<int> const& valid() const { return valid_; }

private:
    SigmaPair sigmasFor(ProbeChannel const& channel,
                        ComputationalCell3D const& cell) const
    {
        if (channel.kind == ChannelKind::MgGroupTotal)
        {
            if (channel.group >= mgOpacity_.energy_groups_center.size())
                return {};
            double const nu = mgOpacity_.energy_groups_center[channel.group];
            double const absorption = mgOpacity_.CalcAbsorptionOpacity(cell, nu);
            double const scattering = mgOpacity_.CalcScatteringOpacity(cell, nu);
            return {EffectiveSigma(absorption, scattering, false),
                    EffectiveSigma(absorption, scattering, true)};
        }

        if (channel.kind == ChannelKind::MgIntegratedTotal)
        {
            auto const [absorption, scattering] = MgIntegratedAbsScat(mgOpacity_, cell);
            return {EffectiveSigma(absorption, scattering, false),
                    EffectiveSigma(absorption, scattering, true)};
        }

        double const absorption = greyOpacity_.CalcPlanckOpacity(cell);
        double const scattering = greyOpacity_.CalcScatteringOpacity(cell);
        return {EffectiveSigma(absorption, scattering, false),
                EffectiveSigma(absorption, scattering, true)};
    }

    double crossingRadius(Vector3D const& location,
                          Vector3D const& dir,
                          double ds,
                          double sigma,
                          double oldTau,
                          double targetTau) const
    {
        double frac = (sigma > 0.0 && ds > 0.0) ? (targetTau - oldTau) / (sigma * ds) : 0.0;
        frac = std::clamp(frac, 0.0, 1.0);
        Vector3D crossingPoint = location + dir * (frac * ds);
        return fastabs(crossingPoint - center_);
    }

    void record(size_t flatIndex, double radius, bool isValid)
    {
        if (flatIndex >= values_.size())
            return;
        values_[flatIndex] = isValid ? radius : -1.0;
        valid_[flatIndex] = isValid ? 1 : 0;
    }

    std::vector<ComputationalCell3D> const& cells_;
    OpacityCalculator const& mgOpacity_;
    OpacityCalculator const& greyOpacity_;
    Vector3D center_;
    double observerRadius_ = 0.0;
    std::vector<ProbeChannel> channels_;
    std::vector<size_t> thermalPartners_;
    std::vector<double> values_;
    std::vector<int> valid_;
};

ProbePassResult RunInwardProbePass(Config const& cfg,
                                    PostprocessRuntime& runtime,
                                    std::vector<ProbeChannel> const& channels,
                                    size_t obsBegin,
                                    size_t obsEnd,
                                    size_t chunkIndex,
                                    size_t chunkCount)
{
    size_t const nObs = runtime.observer->getNumObservers();
    size_t const channelCount = channels.size();
    size_t const particleCount = nObs * channelCount;
    std::vector<size_t> const rayChannelIndices = BuildRayChannelIndices(channels);
    size_t const rayChannelCount = rayChannelIndices.size();
    obsEnd = std::min(obsEnd, nObs);

    auto const& directions = runtime.observer->getDirections();

    std::vector<PhotosphereProbePhysics::Particle> particles;
    size_t const chunkRayCount = (obsEnd > obsBegin) ? (obsEnd - obsBegin) * rayChannelCount : 0;
    particles.reserve(chunkRayCount / std::max(1, runtime.mpiSize));

    for (size_t obs = obsBegin; obs < obsEnd; ++obs) {
        Vector3D dir = directions[obs];
        double const norm = abs(dir);
        if (!(norm > 0.0))
            continue;
        dir = dir * (1.0 / norm);

        Vector3D spherePoint = cfg.center + dir * cfg.radius;

        bool isLocal = false;
#ifdef RICH_MPI
        if (!runtime.tess.IsPointOutsideBox(spherePoint)) {
            int owner = runtime.tess.GetOwner(spherePoint);
            isLocal = (owner == runtime.rank);
        }
#else
        isLocal = !runtime.tess.IsPointOutsideBox(spherePoint);
#endif

        if (!isLocal)
            continue;

        size_t startCell = runtime.tess.GetContainingCell(spherePoint);
        if (startCell >= runtime.tess.GetPointNo())
            continue;

        Vector3D inwardDir = dir * (-1.0);

        for (size_t ch : rayChannelIndices) {
            size_t const flat = obs * channelCount + ch;
            PhotosphereProbePhysics::Particle p;
            p.id = flat;
            p.location = spherePoint;
            p.velocity = inwardDir;
            p.cellIndex = startCell;
            p.cellID = runtime.cells[startCell].ID;
            p.sourceCellID = p.cellID;
            p.weight = 0.0;
            p.initialWeight = 0.0;
            p.frequency = 0.0;
            particles.push_back(p);
        }
    }

    unsigned long long localLaunched = static_cast<unsigned long long>(particles.size());
    unsigned long long globalLaunched = localLaunched;
#ifdef RICH_MPI
    MPI_Allreduce(&localLaunched, &globalLaunched, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif

    if (runtime.rank == 0) {
        std::cout << "[photosphere] chunk " << (chunkIndex + 1) << "/" << chunkCount
                  << " observers [" << obsBegin << "," << obsEnd << ")"
                  << " launching " << globalLaunched << "/" << chunkRayCount
                  << " paired rays for " << ((obsEnd > obsBegin) ?
                                             (obsEnd - obsBegin) * channelCount : 0)
                  << " output radii" << std::endl;
    }

    auto probeBoundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(runtime.tess);
    auto probePhysics = std::make_shared<PhotosphereProbePhysics>(
        runtime.tess, probeBoundary, runtime.cells, *runtime.opacity, *runtime.greyOpacity,
        cfg.center, cfg.radius, channels, particleCount);
    auto probePopControl = std::make_shared<STORM::NoPopulationControl<Vector3D, Tessellation3D>>(runtime.tess);

    std::shared_ptr<MonteCarloManager3D> probeManager;
#ifdef RICH_MPI
    MonteCarloConfig monteCarloConfig;
    std::unique_ptr<STORM::CommunicationEngine<Vector3D>> engine =
        std::make_unique<STORM::RDMACommunicationEngine<Vector3D, Tessellation3D>>(
            runtime.tess, monteCarloConfig, MPI_COMM_WORLD, RDMA_Type::AUTO_RDMA);
    probeManager = std::make_shared<MonteCarloManager3D>(
        runtime.tess, probePhysics, probePopControl, probeBoundary, monteCarloConfig, std::move(engine));
#else
    probeManager = std::make_shared<MonteCarloManager3D>(
        runtime.tess, probePhysics, probePopControl, probeBoundary);
#endif

    double const fullDistance = cfg.radius * 2.01;
    auto const chunkStart = std::chrono::steady_clock::now();
    probeManager->getParticles() = std::move(particles);
    probeManager->step(runtime.cells, fullDistance);

    ProbePassResult result;
    result.value = probePhysics->values();
    result.valid = probePhysics->valid();

#ifdef RICH_MPI
    if (!result.value.empty()) {
        MPI_Allreduce(MPI_IN_PLACE, result.value.data(),
                      static_cast<int>(result.value.size()),
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, result.valid.data(),
                      static_cast<int>(result.valid.size()),
                      MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
#endif

    size_t const chunkValid = CountValidRange(result.valid, obsBegin, obsEnd, channelCount);
    if (runtime.rank == 0) {
        std::cout << "[photosphere] chunk " << (chunkIndex + 1) << "/" << chunkCount
                  << " done in " << SecondsSince(chunkStart) << " s"
                  << ", valid radii " << chunkValid << "/"
                  << ((obsEnd > obsBegin) ? (obsEnd - obsBegin) * channelCount : 0)
                  << std::endl;
    }

    return result;
}

void FillOutputs(SphericalObserver::PhotosphereData& data,
                 std::vector<ProbeChannel> const& channels,
                 ProbePassResult const& radii,
                 size_t nObs,
                 size_t groupCount)
{
    if (groupCount > 0) {
        data.mgGroupRadiusTauTotal.assign(nObs, std::vector<double>(groupCount, -1.0));
        data.mgGroupRadiusThermalization.assign(nObs, std::vector<double>(groupCount, -1.0));
        data.mgGroupValidTauTotal.assign(nObs, std::vector<int>(groupCount, 0));
        data.mgGroupValidThermalization.assign(nObs, std::vector<int>(groupCount, 0));
    }

    data.mgIntegratedRadiusTauTotal.assign(nObs, -1.0);
    data.mgIntegratedRadiusThermalization.assign(nObs, -1.0);
    data.mgIntegratedValidTauTotal.assign(nObs, 0);
    data.mgIntegratedValidThermalization.assign(nObs, 0);

    data.greyRadiusTauTotal.assign(nObs, -1.0);
    data.greyRadiusThermalization.assign(nObs, -1.0);
    data.greyValidTauTotal.assign(nObs, 0);
    data.greyValidThermalization.assign(nObs, 0);

    size_t const channelCount = channels.size();
    for (size_t obs = 0; obs < nObs; ++obs) {
        for (size_t ch = 0; ch < channelCount; ++ch) {
            size_t const flat = obs * channelCount + ch;
            if (flat >= radii.value.size())
                continue;
            ProbeChannel const& channel = channels[ch];
            double const radius = radii.value[flat];
            int const valid = radii.valid[flat];

            switch (channel.kind) {
            case ChannelKind::MgGroupTotal:
                data.mgGroupRadiusTauTotal[obs][channel.group] = radius;
                data.mgGroupValidTauTotal[obs][channel.group] = valid;
                break;
            case ChannelKind::MgGroupThermalization:
                data.mgGroupRadiusThermalization[obs][channel.group] = radius;
                data.mgGroupValidThermalization[obs][channel.group] = valid;
                break;
            case ChannelKind::MgIntegratedTotal:
                data.mgIntegratedRadiusTauTotal[obs] = radius;
                data.mgIntegratedValidTauTotal[obs] = valid;
                break;
            case ChannelKind::MgIntegratedThermalization:
                data.mgIntegratedRadiusThermalization[obs] = radius;
                data.mgIntegratedValidThermalization[obs] = valid;
                break;
            case ChannelKind::GreyTotal:
                data.greyRadiusTauTotal[obs] = radius;
                data.greyValidTauTotal[obs] = valid;
                break;
            case ChannelKind::GreyThermalization:
                data.greyRadiusThermalization[obs] = radius;
                data.greyValidThermalization[obs] = valid;
                break;
            }
        }
    }
}

void PrintPhotosphereSummary(SphericalObserver::PhotosphereData const& data,
                             size_t nObs,
                             size_t groupCount)
{
    std::cout << "[photosphere] valid summary:"
              << " mg_integrated_tau_total=" << CountValid(data.mgIntegratedValidTauTotal) << "/" << nObs
              << " mg_integrated_thermalization=" << CountValid(data.mgIntegratedValidThermalization) << "/" << nObs
              << " grey_tau_total=" << CountValid(data.greyValidTauTotal) << "/" << nObs
              << " grey_thermalization=" << CountValid(data.greyValidThermalization) << "/" << nObs;
    if (groupCount > 0) {
        size_t groupTotal = 0;
        size_t groupThermal = 0;
        for (auto const& row : data.mgGroupValidTauTotal)
            groupTotal += CountValid(row);
        for (auto const& row : data.mgGroupValidThermalization)
            groupThermal += CountValid(row);
        size_t const denom = nObs * groupCount;
        std::cout << " mg_group_tau_total=" << groupTotal << "/" << denom
                  << " mg_group_thermalization=" << groupThermal << "/" << denom;
    }
    std::cout << std::endl;
}

} // namespace

SphericalObserver::PhotosphereData ComputeObserverPhotospheres(
    Config const& cfg,
    PostprocessRuntime& runtime)
{
    SphericalObserver::PhotosphereData data;
    data.tauThreshold = kPhotosphereTau;
    data.thermalizationTauThreshold = kThermalizationTau;

    size_t const nObs = runtime.observer->getNumObservers();
    size_t const groupCount = runtime.opacity->energy_groups_center.size();
    std::vector<ProbeChannel> channels;
    channels.reserve(2 * groupCount + 4);
    for (size_t g = 0; g < groupCount; ++g) {
        channels.push_back({ChannelKind::MgGroupTotal, g});
        channels.push_back({ChannelKind::MgGroupThermalization, g});
    }
    channels.push_back({ChannelKind::MgIntegratedTotal, 0});
    channels.push_back({ChannelKind::MgIntegratedThermalization, 0});
    channels.push_back({ChannelKind::GreyTotal, 0});
    channels.push_back({ChannelKind::GreyThermalization, 0});

    auto const totalStart = std::chrono::steady_clock::now();
    if (runtime.rank == 0)
        std::cout << "Computing observer photospheres (inward) for " << nObs
                  << " observers, " << groupCount << " MG groups, "
                  << channels.size() << " output channels." << std::endl;

    ProbePassResult radii;
    size_t const channelCount = channels.size();
    size_t const rayChannelCount = BuildRayChannelIndices(channels).size();
    size_t const particleCount = nObs * channelCount;
    radii.value.assign(particleCount, -1.0);
    radii.valid.assign(particleCount, 0);

    size_t const chunkSize = PhotosphereChunkSize(nObs);
    size_t const chunkCount = (nObs + chunkSize - 1) / chunkSize;
    if (runtime.rank == 0) {
        std::cout << "[photosphere] inward ray pass: chunk_size=" << chunkSize
                  << " observers, chunks=" << chunkCount
                  << ", paired rays=" << nObs * rayChannelCount
                  << ", output radii=" << particleCount
                  << ", tau_total=" << kPhotosphereTau
                  << ", tau_thermalization=" << kThermalizationTau
                  << std::endl;
    }

    size_t cumulativeValid = 0;
    for (size_t chunk = 0; chunk < chunkCount; ++chunk) {
        size_t const obsBegin = chunk * chunkSize;
        size_t const obsEnd = std::min(nObs, obsBegin + chunkSize);
        ProbePassResult chunkRadii =
            RunInwardProbePass(cfg, runtime, channels, obsBegin, obsEnd, chunk, chunkCount);

        for (size_t obs = obsBegin; obs < obsEnd; ++obs) {
            for (size_t ch = 0; ch < channelCount; ++ch) {
                size_t const flat = obs * channelCount + ch;
                if (flat < radii.value.size() && flat < chunkRadii.value.size()) {
                    radii.value[flat] = chunkRadii.value[flat];
                    radii.valid[flat] = chunkRadii.valid[flat];
                }
            }
        }

        cumulativeValid += CountValidRange(chunkRadii.valid, obsBegin, obsEnd, channelCount);
        if (runtime.rank == 0) {
            size_t const processedProbes = obsEnd * channelCount;
            std::cout << "[photosphere] cumulative valid radii "
                      << cumulativeValid << "/" << processedProbes
                      << " after " << SecondsSince(totalStart) << " s"
                      << std::endl;
        }
    }

    FillOutputs(data, channels, radii, nObs, groupCount);

    if (runtime.rank == 0) {
        PrintPhotosphereSummary(data, nObs, groupCount);
        std::cout << "Observer photospheres computed in "
                  << SecondsSince(totalStart) << " s." << std::endl;
    }

    return data;
}

} // namespace imc_postprocess_tde
