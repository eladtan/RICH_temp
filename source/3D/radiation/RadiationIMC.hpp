#ifndef RADIATION_IMC_HPP
#define RADIATION_IMC_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "MonteCarloPhysics3D.hpp"
#include "monte/utils/RandomInCell.hpp"
#include "monte/utils/LinearInterpolation.hpp"
#include "monte/radiation/RadiationIMC.hpp"
#include "SphericalObserver.hpp"
#include "postprocess/IMCPostProcessControl.hpp"
#include "CMMC/src/planck_integral/planck_integral.hpp"

class SphericalObserver;

using RadiationIMCParameters =
    STORM::RadiationIMCParameters<ENERGY_GROUPS_NUM>;
using ComptonInducedMode = STORM::ComptonInducedMode;

class RICHRadiationOpacityAdapter final
    : public STORM::RadiationOpacityModel<Vector3D, Tessellation3D, ComputationalCell3D, ENERGY_GROUPS_NUM>
{
public:
    using Base = STORM::RadiationOpacityModel<Vector3D, Tessellation3D, ComputationalCell3D, ENERGY_GROUPS_NUM>;
    using GroupArray = typename Base::GroupArray;
    using GroupBoundaries = typename Base::GroupBoundaries;
    using GroupCdf = std::array<double, ENERGY_GROUPS_NUM + 1>;

    explicit RICHRadiationOpacityAdapter(std::shared_ptr<OpacityCalculator> opacity):
        opacity_(std::move(opacity))
    {}

    double CalcPlanckOpacity(const ComputationalCell3D &cell) override
    {
        return this->opacity_->CalcPlanckOpacity(cell);
    }

    double CalcAbsorptionOpacity(const ComputationalCell3D &cell, double frequency) override
    {
        return this->opacity_->CalcAbsorptionOpacity(cell, frequency);
    }

    double CalcScatteringOpacity(const ComputationalCell3D &cell) override
    {
        return this->opacity_->CalcScatteringOpacity(cell);
    }

    double CalcScatteringOpacity(const ComputationalCell3D &cell, double frequency) override
    {
        return this->opacity_->CalcScatteringOpacity(cell, frequency);
    }

    Vector3D getRandomVelocity(const ComputationalCell3D &cell,
                               double directionRandom1,
                               double directionRandom2) override
    {
        return this->opacity_->getRandomVelocity(
            cell, directionRandom1, directionRandom2);
    }

    Vector3D getNewScatterVelocity(const ComputationalCell3D &cell,
                                   const Vector3D &oldVelocity,
                                   double frequency,
                                   double directionRandom1,
                                   double directionRandom2) override
    {
        STORM::Particle<Vector3D> particle;
        particle.velocity = oldVelocity;
        particle.frequency = frequency;
        return this->opacity_->getNewScatterVelocity(
            cell, particle, directionRandom1, directionRandom2);
    }

    bool ComptonIncludedInTransport() const override
    {
        return this->opacity_->ComptonIncludedInTransport();
    }

    void reseed(std::uint64_t seed) override
    {
        this->opacity_->rng_.seed(seed);
    }

    std::size_t findGroup(double frequency, const GroupBoundaries &boundaries) const override
    {
        for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
        {
            if(frequency < boundaries[g + 1])
            {
                return g;
            }
        }
        return ENERGY_GROUPS_NUM - 1;
    }

    double GetThermalEnergy(const ComputationalCell3D &cell,
                            double random,
                            const GroupBoundaries &boundaries) const override
    {
        GroupCdf cumulative = this->computeCumulativeOpacity(cell, boundaries);
        double const total = cumulative[ENERGY_GROUPS_NUM];
        if(!(total > 0.0) || !std::isfinite(total))
        {
            return Base::GetThermalEnergy(cell, random, boundaries);
        }

        double const r = this->clampUnitOpen(random);
        return STORM::LinearInterpolation(cumulative, boundaries, r * total);
    }

    double SampleThermalEnergyInGroup(const ComputationalCell3D &cell,
                                      std::size_t group,
                                      double random,
                                      const GroupBoundaries &boundaries) const override
    {
        group = std::min<std::size_t>(group, ENERGY_GROUPS_NUM - 1);
        GroupCdf cumulative = this->computeCumulativeOpacity(cell, boundaries);
        double const c0 = cumulative[group];
        double const c1 = cumulative[group + 1];
        if(c1 <= c0 || !std::isfinite(c1 - c0))
        {
            return 0.5 * (boundaries[group] + boundaries[group + 1]);
        }

        double const r = this->clampUnitOpen(random);
        return STORM::LinearInterpolation(cumulative, boundaries, c0 + r * (c1 - c0));
    }

    GroupArray GetThermalGroupPdf(const ComputationalCell3D &cell,
                                  const GroupBoundaries &boundaries) const override
    {
        GroupArray pdf{};
        GroupCdf cumulative = this->computeCumulativeOpacity(cell, boundaries);
        double const total = cumulative[ENERGY_GROUPS_NUM];
        if(!(total > 0.0) || !std::isfinite(total))
        {
            return pdf;
        }

        for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
        {
            double const weight = cumulative[g + 1] - cumulative[g];
            pdf[g] = (weight > 0.0 && std::isfinite(weight)) ? weight / total : 0.0;
        }
        return pdf;
    }

    GroupArray GetCumulativeOpacity(const ComputationalCell3D &cell,
                                    const GroupBoundaries &boundaries) const override
    {
        GroupArray cumulativeUpper{};
        GroupCdf cumulative = this->computeCumulativeOpacity(cell, boundaries);
        for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
        {
            cumulativeUpper[g] = cumulative[g + 1];
        }
        return cumulativeUpper;
    }

    GroupArray getEnergyCenters(const GroupBoundaries &boundaries) const override
    {
        return this->energyCenters(boundaries);
    }

    const OpacityCalculator &richOpacity() const { return *this->opacity_; }

private:
    static double clampUnitOpen(double random)
    {
        double const upper = std::nextafter(1.0, 0.0);
        return std::isfinite(random) ? std::clamp(random, 0.0, upper) : 0.5;
    }

    static GroupArray energyCenters(const GroupBoundaries &boundaries)
    {
        GroupArray centers{};
        for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
        {
            centers[g] = 0.5 * (boundaries[g] + boundaries[g + 1]);
        }
        return centers;
    }

    GroupCdf computeCumulativeOpacity(const ComputationalCell3D &cell,
                                      const GroupBoundaries &boundaries) const
    {
        if(cumulativeCacheValid_
           && cumulativeCacheCellID_ == cell.ID
           && cumulativeCacheTemperature_ == cell.temperature
           && cumulativeCacheBoundaries_ == boundaries)
        {
            return cumulativeCache_;
        }

        GroupCdf cumulative{};
        GroupArray centers = this->energyCenters(boundaries);
        double const kT = units::k_boltz * cell.temperature;
        if(!(kT > 0.0) || !std::isfinite(kT))
        {
            return cumulative;
        }

        cumulative[0] = 0.0;
        for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
        {
            double const a = boundaries[g] / kT;
            double const b = boundaries[g + 1] / kT;
            double const bg = (a > 0.0 && b > a)
                ? planck_integral::planck_integral(a, b)
                : 0.0;
            double const sigma = this->opacity_->CalcAbsorptionOpacity(cell, centers[g]);
            double const weight = (sigma > 0.0 && std::isfinite(sigma) && std::isfinite(bg))
                ? sigma * bg
                : 0.0;
            cumulative[g + 1] = cumulative[g] + weight;
        }

        cumulativeCacheValid_ = true;
        cumulativeCacheCellID_ = cell.ID;
        cumulativeCacheTemperature_ = cell.temperature;
        cumulativeCacheBoundaries_ = boundaries;
        cumulativeCache_ = cumulative;
        return cumulative;
    }

    std::shared_ptr<OpacityCalculator> opacity_;
    mutable bool cumulativeCacheValid_ = false;
    mutable std::size_t cumulativeCacheCellID_ = std::numeric_limits<std::size_t>::max();
    mutable double cumulativeCacheTemperature_ = std::numeric_limits<double>::quiet_NaN();
    mutable GroupBoundaries cumulativeCacheBoundaries_{};
    mutable GroupCdf cumulativeCache_{};
};

struct RICHRadiationIMCTraits
{
    using GroupArray = std::array<double, ENERGY_GROUPS_NUM>;
    using GroupBoundaries = std::array<double, ENERGY_GROUPS_NUM + 1>;

    GroupBoundaries energyBoundaries(const ComputationalCell3D &cell) const
    {
        (void) cell;
        return ComputationalCell3D::energyBoundaries;
    }

    decltype(auto) tracers(const ComputationalCell3D &cell) const
    {
        return (cell.tracers);
    }

    decltype(auto) tracerNames(const ComputationalCell3D &cell) const
    {
        (void) cell;
        return (ComputationalCell3D::tracerNames);
    }

    double groupEnergyPerMass(const ComputationalCell3D &cell, std::size_t group) const
    {
        return cell.Eg[group];
    }

    double extensiveGroupEnergy(const Conserved3D &conserved, std::size_t group) const
    {
        return conserved.Eg[group];
    }
};

struct RICHRadiationPositionSampler
{
    using Decomposition = CellVolumeDecomposition;
    using Sampler = RandomInCellPositionSampler<Vector3D, Tessellation3D>;

    void BuildDecomposition(const Tessellation3D &grid,
                            std::size_t cellIndex,
                            Decomposition &out) const
    {
        Sampler().BuildDecomposition(grid, cellIndex, out);
    }

    /// Draws from the same thread-local stream as `operator()`, in the same order.
    Vector3D Sample(const Tessellation3D &grid,
                    std::size_t cellIndex,
                    const Decomposition &decomp,
                    std::mt19937_64 &rng,
                    std::uniform_real_distribution<double> &dist) const
    {
        (void) rng;
        (void) dist;
        Vector3D location = Sampler().Sample(grid, cellIndex, decomp,
                                             random_in_cell_detail::GetRNG(),
                                             random_in_cell_detail::GetDist());
        static constexpr double nudge = 1e-10;
        return location * (1.0 - nudge) + nudge * grid.GetMeshPoint(cellIndex);
    }

    Vector3D Sample(const Tessellation3D &grid,
                    std::size_t cellIndex,
                    const Decomposition &decomp,
                    std::uint64_t rngKey,
                    std::uint64_t &rngCounter) const
    {
        Vector3D location =
            Sampler().Sample(grid, cellIndex, decomp, rngKey, rngCounter);
        static constexpr double nudge = 1e-10;
        return location * (1.0 - nudge) + nudge * grid.GetMeshPoint(cellIndex);
    }

    Vector3D operator()(const Tessellation3D &grid,
                        std::size_t cellIndex,
                        std::mt19937_64 &rng,
                        std::uniform_real_distribution<double> &dist) const
    {
        (void) rng;
        (void) dist;
        Vector3D location = RandomPointInCell(grid, cellIndex);
        static constexpr double nudge = 1e-10;
        return location * (1.0 - nudge) + nudge * grid.GetMeshPoint(cellIndex);
    }
};

class RadiationIMC final : public MonteCarloRadiationPhysics3D
{
public:
    using Particle = STORM::Particle<Vector3D>;
    using Functionality = STORM::StepResult;
    using BoundaryCond = STORM::BoundaryCondition<Vector3D, Tessellation3D>;
    using GroupArray = std::array<double, ENERGY_GROUPS_NUM>;
    using Impl = STORM::RadiationIMC<Vector3D,
                                     Tessellation3D,
                                     ComputationalCell3D,
                                     Conserved3D,
                                     EquationOfState,
                                     ENERGY_GROUPS_NUM,
                                     RICHRadiationOpacityAdapter,
                                     RICHRadiationIMCTraits,
                                     RICHRadiationPositionSampler>;
    using ImplSourceAllocationSummary = typename Impl::SourceAllocationSummary;
    using ImplGroupSamplingDiagnostics = typename Impl::GroupSamplingDiagnostics;
    using PostProcessExternalSource = IMCPostProcessExternalSource;
    using SourceAllocationSummary = IMCSourceAllocationSummary;
    using GroupSamplingDiagnostics = IMCGroupSamplingDiagnostics;
    using ComptonCellData = typename Impl::ComptonCellData;
    using Parameters = typename Impl::Parameters;

    class ObserverAdapter final : public STORM::RadiationObserver<Vector3D>
    {
    public:
        explicit ObserverAdapter(std::shared_ptr<SphericalObserver> observer):
            observer_(std::move(observer))
        {}

        Crossing nextOutwardCrossing(const Vector3D &position,
                                     const Vector3D &velocity,
                                     double maxTime) const override
        {
            auto const crossing = observer_->nextOutwardCrossing(position, velocity, maxTime);
            return Crossing{crossing.hit, crossing.time, crossing.point};
        }

        void recordCrossing(const STORM::ObserverCrossingRecord<Vector3D> &record) override
        {
            ::ObserverCrossingRecord oldRecord;
            oldRecord.crossingPoint = record.crossingPoint;
            oldRecord.direction = record.direction;
            oldRecord.weight = record.weight;
            oldRecord.frequency = record.frequency;
            oldRecord.sourceCellID = record.sourceCellID;
#ifdef MONTECARLO_POLARIZATION
            oldRecord.stokesQ = record.stokesQ;
            oldRecord.stokesU = record.stokesU;
            oldRecord.polBasis = record.polarizationBasis;
            oldRecord.polarizationInitialized = record.polarizationInitialized;
#endif
            observer_->recordCrossing(oldRecord);
        }

        void addEmittedEnergy(double energy) override { observer_->addEmittedEnergy(energy); }
        void addAbsorbedEnergy(double energy) override { observer_->addAbsorbedEnergy(energy); }
        void addBoxEscapeEnergy(double energy) override { observer_->addBoxEscapeEnergy(energy); }
        void addTimedOutEnergy(double energy) override { observer_->addTimedOutEnergy(energy); }
        void addCutoffEnergy(double energy) override { observer_->addCutoffEnergy(energy); }
        void resetTallies() override { observer_->resetTallies(); }

    private:
        std::shared_ptr<SphericalObserver> observer_;
    };

    RadiationIMC(Tessellation3D &grid,
                 const std::shared_ptr<BoundaryCond> &boundary,
                 std::vector<ComputationalCell3D> &cells,
                 std::vector<Conserved3D> &conserved,
                 std::shared_ptr<EquationOfState> eos,
                 std::shared_ptr<OpacityCalculator> opacity,
                 Parameters parameters):
        MonteCarloRadiationPhysics3D(grid, boundary, cells, conserved, eos, opacity),
        opacityAdapter_(std::make_shared<RICHRadiationOpacityAdapter>(std::move(opacity))),
        impl_(grid,
              boundary,
              cells,
              conserved,
              std::move(eos),
              opacityAdapter_,
              parameters,
              RICHRadiationIMCTraits{},
              RICHRadiationPositionSampler{})
    {}

    RadiationIMC(Tessellation3D &grid,
                 const std::shared_ptr<BoundaryCond> &boundary,
                 std::vector<ComputationalCell3D> &cells,
                 std::vector<Conserved3D> &conserved,
                 std::shared_ptr<EquationOfState> eos,
                 std::shared_ptr<OpacityCalculator> opacity,
                 std::size_t newPhotonsPerCell):
        RadiationIMC(grid,
                     boundary,
                     cells,
                     conserved,
                     std::move(eos),
                     std::move(opacity),
                     makeParameters(newPhotonsPerCell))
    {}

    std::vector<Particle> preStep(double fullDt) override
    {
        return this->impl_.preStep(fullDt);
    }

    void updateGridData(void) override
    {
        this->impl_.updateGridData();
    }

    Functionality step(Particle &particle, std::vector<Particle> &particlesToAdd) override
    {
        return this->impl_.step(particle, particlesToAdd);
    }

    void postStep(const std::vector<Particle> &particles, double fullDt) override
    {
        this->impl_.postStep(particles, fullDt);
    }

    Particle generateSingleParticle(std::size_t cellIndex, const ComputationalCell3D &cell) const override
    {
        return const_cast<RadiationIMC *>(this)->impl_.generateSingleParticle(cellIndex, cell);
    }

    std::vector<Particle> generateInitialParticles(std::size_t particlesPerCell) override
    {
        return this->impl_.generateInitialParticles(particlesPerCell);
    }

    void adjustExistingParticles(std::vector<Particle> &particles, double fullDt) override
    {
        this->impl_.adjustExistingParticles(particles, fullDt);
    }

    const std::vector<double> &getEradTimeAvg(void) const override { return this->impl_.getEradTimeAvg(); }
    std::vector<double> &getEradTimeAvg(void) override { return this->impl_.getEradTimeAvg(); }
    const std::vector<GroupArray> &getEgTimeAvg(void) const override { return this->impl_.getEgTimeAvg(); }
    std::vector<GroupArray> &getEgTimeAvg(void) override { return this->impl_.getEgTimeAvg(); }

    const std::vector<double> &getFactorFleck() const { return this->impl_.getFactorFleck(); }
    const std::vector<double> &getPlanckOpacities() const { return this->impl_.getPlanckOpacities(); }
    const std::vector<ComptonCellData> &getComptonData() const { return this->impl_.getComptonData(); }
    const GroupArray &getComptonGroupCenters() const { return this->impl_.getComptonGroupCenters(); }
    const GroupArray &getComptonGroupWidths() const { return this->impl_.getComptonGroupWidths(); }
#ifdef STORM_WITH_GPU
    bool UsesDeviceTransport() const
    {
        return this->impl_.UsesDeviceTransport();
    }

    STORM::gpu::GreyIMCViews<STORM::gpu::DeviceVec3> GetDeviceTransportViews() const
    {
        return this->impl_.GetDeviceTransportViews();
    }
#endif

    void setObserver(std::shared_ptr<SphericalObserver> observer)
    {
        observer_ = std::move(observer);
        observerAdapter_ = observer_
            ? std::make_shared<ObserverAdapter>(observer_) : nullptr;
#ifdef MONTECARLO_POLARIZATION
        if(observer_)
        {
            auto const &parameters = this->impl_.getParameters();
            auto const &polarization = parameters.postProcess.polarization;
            observer_->setPolarizationMetadata(
                parameters.withPolarization || polarization.enabled,
                polarization.manualScatteringsAfterAcceleration,
                polarization.depolarizationScatterings,
                polarization.acceleratedClosure);
        }
#endif
        this->impl_.setObserver(observerAdapter_);
    }

    void setNewPhotonsPerCell(std::size_t n) { this->impl_.setNewPhotonsPerCell(n); }

    void reseedRNG(std::uint64_t seed)
    {
        MonteCarloRadiationPhysics3D::reseedRNG(seed);
        this->impl_.reseedRNG(seed);
    }

    std::size_t getRandomWalkStepCount() const override
    {
        return this->impl_.getRandomWalkStepCount();
    }

    std::size_t getDDMCStepCount() const override
    {
        return this->impl_.getDDMCStepCount();
    }

    std::size_t getDDMCLeakCount() const override
    {
        return this->impl_.getDDMCLeakCount();
    }

    std::size_t getDDMCCensusCount() const override
    {
        return this->impl_.getDDMCCensusCount();
    }

    std::size_t getDDMCUpscatterCount() const override
    {
        return this->impl_.getDDMCUpscatterCount();
    }

    std::size_t getDDMCFallbackCount() const override
    {
        return this->impl_.getDDMCFallbackCount();
    }

    std::string getAccelerationDebugInfo(
        std::size_t cellIndex, double frequency) const override
    {
        return this->impl_.getAccelerationDebugInfo(cellIndex, frequency);
    }

    std::string getDDMCFaceDiagnosticsTSV(double xMin, double xMax) const
    {
        return this->impl_.getDDMCFaceDiagnosticsTSV(xMin, xMax);
    }

    std::string getDDMCInterfaceEventDiagnosticsTSV(
        double xMin, double xMax) const
    {
        return this->impl_.getDDMCInterfaceEventDiagnosticsTSV(xMin, xMax);
    }

    void setAdaptiveSourceCellScores(std::unordered_map<std::size_t, double> scores,
                                     double strength,
                                     double maxFactor,
                                     double learnedReserveFrac,
                                     double learnedMinFactor,
                                     double observerBudgetMultiplier,
                                     std::size_t learnedMinPhotons = 0,
                                     std::size_t learnedMaxPhotons = 0,
                                     double scorePower = 1.0)
    {
        this->impl_.setAdaptiveSourceCellScores(std::move(scores),
                                                strength,
                                                maxFactor,
                                                learnedReserveFrac,
                                                learnedMinFactor,
                                                observerBudgetMultiplier,
                                                learnedMinPhotons,
                                                learnedMaxPhotons,
                                                scorePower);
    }

    void clearAdaptiveSourceCellScores()
    {
        this->impl_.clearAdaptiveSourceCellScores();
    }

    void setAdaptiveSourceCellGroupScores(std::unordered_map<std::size_t, GroupArray> scores,
                                          double strength,
                                          double pdfFloor,
                                          double maxBias,
                                          double maxWeightCorrection)
    {
        this->impl_.setAdaptiveSourceCellGroupScores(std::move(scores),
                                                     strength,
                                                     pdfFloor,
                                                     maxBias,
                                                     maxWeightCorrection);
    }

    void clearAdaptiveSourceCellGroupScores()
    {
        this->impl_.clearAdaptiveSourceCellGroupScores();
    }

    GroupSamplingDiagnostics getLastGroupSamplingDiagnostics() const
    {
        return convertGroupSamplingDiagnostics(
            this->impl_.getLastGroupSamplingDiagnostics());
    }

    void setSourceEmissionControl(bool useLearnedScores,
                                  bool includeUniformBase,
                                  std::size_t baseMultiplier,
                                  std::size_t learnedBoostFactor = 20,
                                  std::size_t learnedExtraBudget = 0)
    {
        this->impl_.setSourceEmissionControl(useLearnedScores,
                                             includeUniformBase,
                                             baseMultiplier,
                                             learnedBoostFactor,
                                             learnedExtraBudget);
    }

    void clearSourceEmissionControl()
    {
        this->impl_.clearSourceEmissionControl();
    }

    SourceAllocationSummary getLastSourceAllocationSummary() const
    {
        return convertSourceAllocationSummary(
            this->impl_.getLastSourceAllocationSummary());
    }

    const std::vector<std::size_t> &getLastSourcePhotonsPerCell() const
    {
        return this->impl_.getLastSourcePhotonsPerCell();
    }

    void setPostProcessExternalSources(
        std::vector<PostProcessExternalSource> sources)
    {
        std::vector<typename Impl::PostProcessExternalSource> converted;
        converted.reserve(sources.size());
        for(PostProcessExternalSource const &source : sources)
        {
            typename Impl::PostProcessExternalSource target;
            target.faceIndex = source.faceIndex;
            target.cellID = source.cellID;
            target.interiorCellID = source.interiorCellID;
            target.location = source.location;
            target.outwardNormal = source.outwardNormal;
            target.luminosity = source.luminosity;
            converted.push_back(target);
        }
        this->impl_.setPostProcessExternalSources(std::move(converted));
    }

    void clearPostProcessExternalSources()
    {
        this->impl_.clearPostProcessExternalSources();
    }

    bool hasPostProcessExternalSources() const
    {
        return this->impl_.hasPostProcessExternalSources();
    }

    void configurePostProcessControl(IMCPostProcessControl control)
    {
        if(control.adaptiveCells.enabled)
        {
            this->setAdaptiveSourceCellScores(
                std::move(control.adaptiveCells.scores),
                control.adaptiveCells.strength,
                control.adaptiveCells.maxFactor,
                control.adaptiveCells.learnedReserveFraction,
                control.adaptiveCells.learnedMinFactor,
                control.adaptiveCells.observerBudgetMultiplier,
                control.adaptiveCells.learnedMinPhotons,
                control.adaptiveCells.learnedMaxPhotons,
                control.adaptiveCells.scorePower);
        }
        else
        {
            this->clearAdaptiveSourceCellScores();
        }

        if(control.adaptiveGroups.enabled)
        {
            this->setAdaptiveSourceCellGroupScores(
                std::move(control.adaptiveGroups.scores),
                control.adaptiveGroups.strength,
                control.adaptiveGroups.pdfFloor,
                control.adaptiveGroups.maxBias,
                control.adaptiveGroups.maxWeightCorrection);
        }
        else
        {
            this->clearAdaptiveSourceCellGroupScores();
        }

        if(control.emission.enabled)
        {
            this->setSourceEmissionControl(
                control.emission.useLearnedScores,
                control.emission.includeUniformBase,
                control.emission.baseMultiplier,
                control.emission.learnedBoostFactor,
                control.emission.learnedExtraBudget);
        }
        else
        {
            this->clearSourceEmissionControl();
        }

        if(control.externalSources.empty())
        {
            this->clearPostProcessExternalSources();
        }
        else
        {
            this->setPostProcessExternalSources(
                std::move(control.externalSources));
        }
    }

    IMCPostProcessGenerationDiagnostics
    getPostProcessGenerationDiagnostics() const
    {
        IMCPostProcessGenerationDiagnostics diagnostics;
        diagnostics.sourceAllocation =
            this->getLastSourceAllocationSummary();
        diagnostics.groupSampling =
            this->getLastGroupSamplingDiagnostics();
        diagnostics.sourcePhotonsPerCell =
            this->getLastSourcePhotonsPerCell();
        return diagnostics;
    }

private:
    static SourceAllocationSummary convertSourceAllocationSummary(
        ImplSourceAllocationSummary const &source)
    {
        SourceAllocationSummary target;
        target.adaptiveEnabled = source.adaptiveEnabled;
        target.totalPhotons = source.totalPhotons;
        target.sourceCells = source.sourceCells;
        target.boostedCells = source.boostedCells;
        target.learnedCells = source.learnedCells;
        target.learnedBoostedCells = source.learnedBoostedCells;
        target.learnedPhotons = source.learnedPhotons;
        target.learnedExtraPhotons = source.learnedExtraPhotons;
        target.minPhotons = source.minPhotons;
        target.maxPhotons = source.maxPhotons;
        target.learnedMinPhotons = source.learnedMinPhotons;
        target.learnedMaxPhotons = source.learnedMaxPhotons;
        target.adaptiveScoreSum = source.adaptiveScoreSum;
        target.adaptiveScoreP05 = source.adaptiveScoreP05;
        target.adaptiveScoreP50 = source.adaptiveScoreP50;
        target.adaptiveScoreP95 = source.adaptiveScoreP95;
        target.adaptiveScoreMax = source.adaptiveScoreMax;
        target.adaptiveScoreSpanLow = source.adaptiveScoreSpanLow;
        target.adaptiveScoreSpanHigh = source.adaptiveScoreSpanHigh;
        target.learnedPhotonsAtLeast1000 =
            source.learnedPhotonsAtLeast1000;
        target.learnedPhotonsAtLeast2000 =
            source.learnedPhotonsAtLeast2000;
        return target;
    }

    static GroupSamplingDiagnostics convertGroupSamplingDiagnostics(
        ImplGroupSamplingDiagnostics const &source)
    {
        GroupSamplingDiagnostics target;
        target.totalSampled = source.totalSampled;
        target.cellsWithGroupScores = source.cellsWithGroupScores;
        target.weightCorrectionMin = source.weightCorrectionMin;
        target.weightCorrectionMax = source.weightCorrectionMax;
        target.weightCorrectionSum = source.weightCorrectionSum;
        target.weightCorrectionCount = source.weightCorrectionCount;
        target.weightCorrectionCapped = source.weightCorrectionCapped;
        target.weightCorrectionFallback = source.weightCorrectionFallback;
        target.invalidPdfFallback = source.invalidPdfFallback;
        target.invalidPdfFallbackPackets = source.invalidPdfFallbackPackets;
        target.sampledEnergy = source.sampledEnergy;
        target.cappedEnergy = source.cappedEnergy;
        target.cappedEnergyFraction = source.cappedEnergyFraction;
        target.estimatorPotentiallyBiased =
            source.estimatorPotentiallyBiased;
        return target;
    }

    static Parameters makeParameters(std::size_t newPhotonsPerCell)
    {
        Parameters parameters;
        parameters.newPhotonsPerCell = newPhotonsPerCell;
        return parameters;
    }

    std::shared_ptr<RICHRadiationOpacityAdapter> opacityAdapter_;
    Impl impl_;
    std::shared_ptr<SphericalObserver> observer_;
    std::shared_ptr<ObserverAdapter> observerAdapter_;
};

#endif // RADIATION_IMC_HPP
