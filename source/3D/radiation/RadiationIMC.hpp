#ifndef RADIATION_IMC_HPP
#define RADIATION_IMC_HPP

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "MonteCarloPhysics3D.hpp"
#include "monte/utils/RandomInCell.hpp"
#include "monte/radiation/RadiationIMC.hpp"

class SphericalObserver;

class RICHRadiationOpacityAdapter final
    : public STORM::RadiationOpacityModel<Vector3D, Tessellation3D, ComputationalCell3D, ENERGY_GROUPS_NUM>
{
public:
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
                               std::mt19937_64 &rng,
                               std::uniform_real_distribution<double> &dist) override
    {
        (void) rng;
        (void) dist;
        return this->opacity_->getRandomVelocity(cell);
    }

    Vector3D getNewScatterVelocity(const ComputationalCell3D &cell,
                                   const Vector3D &oldVelocity,
                                   double frequency,
                                   std::mt19937_64 &rng,
                                   std::uniform_real_distribution<double> &dist) override
    {
        (void) rng;
        (void) dist;
        STORM::Particle<Vector3D, Tessellation3D> particle;
        particle.velocity = oldVelocity;
        particle.frequency = frequency;
        return this->opacity_->getNewScatterVelocity(cell, particle);
    }

    bool ComptonIncludedInTransport() const override
    {
        return this->opacity_->ComptonIncludedInTransport();
    }

    void reseed(std::uint64_t seed) override
    {
        this->opacity_->rng_.seed(seed);
    }

    const OpacityCalculator &richOpacity() const { return *this->opacity_; }

private:
    std::shared_ptr<OpacityCalculator> opacity_;
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

class RadiationIMC : public MonteCarloRadiationPhysics3D
{
public:
    using Particle = STORM::Particle<Vector3D, Tessellation3D>;
    using Functionality = STORM::StepResult<Vector3D, Tessellation3D>;
    using BoundaryCond = STORM::BoundaryCondition<Vector3D, Tessellation3D>;
    using GroupArray = std::array<double, ENERGY_GROUPS_NUM>;
    using Impl = STORM::RadiationIMC<Vector3D,
                                     Tessellation3D,
                                     ComputationalCell3D,
                                     Conserved3D,
                                     EquationOfState,
                                     ENERGY_GROUPS_NUM,
                                     RICHRadiationIMCTraits,
                                     RICHRadiationPositionSampler>;
    using SourceAllocationSummary = typename Impl::SourceAllocationSummary;
    using GroupSamplingDiagnostics = typename Impl::GroupSamplingDiagnostics;
    using ComptonCellData = typename Impl::ComptonCellData;
    using Parameters = typename Impl::Parameters;

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
    {
        this->syncTimeAverages();
    }

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
        std::vector<Particle> result = this->impl_.preStep(fullDt);
        this->syncTimeAverages();
        return result;
    }

    Functionality step(Particle &particle, std::vector<Particle> &particlesToAdd) override
    {
        return this->impl_.step(particle, particlesToAdd);
    }

    void postStep(const std::vector<Particle> &particles, double fullDt) override
    {
        this->impl_.postStep(particles, fullDt);
        this->syncTimeAverages();
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

    const std::vector<double> &getFactorFleck() const { return this->impl_.getFactorFleck(); }
    const std::vector<double> &getPlanckOpacities() const { return this->impl_.getPlanckOpacities(); }
    const std::vector<ComptonCellData> &getComptonData() const { return this->impl_.getComptonData(); }
    const GroupArray &getComptonGroupCenters() const { return this->impl_.getComptonGroupCenters(); }
    const GroupArray &getComptonGroupWidths() const { return this->impl_.getComptonGroupWidths(); }

    void setObserver(std::shared_ptr<SphericalObserver> observer)
    {
        observer_ = std::move(observer);
    }

    void setNewPhotonsPerCell(std::size_t n) { this->impl_.setNewPhotonsPerCell(n); }

    void setAdaptiveSourceCellScores(std::unordered_map<std::size_t, double> scores,
                                     double strength,
                                     double maxFactor,
                                     double learnedReserveFrac,
                                     double learnedMinFactor,
                                     double observerBudgetMultiplier)
    {
        this->impl_.setAdaptiveSourceCellScores(std::move(scores),
                                                strength,
                                                maxFactor,
                                                learnedReserveFrac,
                                                learnedMinFactor,
                                                observerBudgetMultiplier);
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
        return this->impl_.getLastGroupSamplingDiagnostics();
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
        return this->impl_.getLastSourceAllocationSummary();
    }

    const std::vector<std::size_t> &getLastSourcePhotonsPerCell() const
    {
        return this->impl_.getLastSourcePhotonsPerCell();
    }

private:
    static Parameters makeParameters(std::size_t newPhotonsPerCell)
    {
        Parameters parameters;
        parameters.newPhotonsPerCell = newPhotonsPerCell;
        return parameters;
    }

    void syncTimeAverages()
    {
        this->Erad_time_avg = this->impl_.getEradTimeAvg();
    }

    std::shared_ptr<RICHRadiationOpacityAdapter> opacityAdapter_;
    Impl impl_;
    std::shared_ptr<SphericalObserver> observer_;
};

#endif // RADIATION_IMC_HPP
