#ifndef RADIATION_IMC_HPP
#define RADIATION_IMC_HPP

#include <array>
#include <limits>
#include <memory>
#include "MonteCarloPhysics3D.hpp"
#include "MultigroupOpacity.hpp"
#include "3D/monte/Voronoi3DMovement.hpp"
#include "RandomWalk.hpp"
#include "Radiation/CMMC/src/compton_matrix_mc.hpp"

enum class ComptonTransportMode
{
    FReducedEvents,
    DeterministicSegment
};

struct RadiationIMCParameters
{
    size_t newPhotonsPerCell;
    bool withHydro = false;
    bool diffusionPressureGradient = false;
    bool MMC = false;
    bool withMultigroupOpacity = false;
    bool withRandomWalk = false;
    double rwMinCellOpticalDepth = 25.0;
    double rwMinParticleOpticalDepth = 5.0;
    bool noHydroFeedback = false;
    bool withEgTimeAvg = false;
    bool withCompton = false;
    bool comptonUseInduced = true;
    bool comptonAllowNZeroFallback = true;
    bool comptonDebugParityCheck = false;
    bool comptonDiagnostics = false;
    size_t comptonMatrixSamples = 200000;
    ComptonTransportMode comptonTransportMode = ComptonTransportMode::DeterministicSegment;

    friend std::ostream &operator<<(std::ostream &os, const RadiationIMCParameters &parameters);
};

class RadiationIMC : public MonteCarloRadiationPhysics3D
{
public:
    using Particle = MonteCarloParticle<Vector3D, Tessellation3D>;
    using Functionality = MonteCarloFunctionality<Vector3D, Tessellation3D>;
    using BoundaryCond = BoundaryCondition<Vector3D, Tessellation3D>;
    using GroupArray = std::array<double, ENERGY_GROUPS_NUM>;
    using GroupCdf = std::array<double, ENERGY_GROUPS_NUM + 1>;
    using GroupMatrix = std::array<GroupArray, ENERGY_GROUPS_NUM>;
    using GroupCdfMatrix = std::array<GroupCdf, ENERGY_GROUPS_NUM>;

    struct ComptonCellData
    {
        GroupArray absorptionOpacity{};
        GroupArray planckFraction{};
        GroupArray baseSourceFraction{};
        GroupCdf planckCdf{};
        GroupCdf baseSourceCdf{};
        bool active = false;
        double planckOpacity = 0.0;
        double volume = 0.0;
        double temperature = 0.0;
        double Um = 0.0;
        double beta = 0.0;
        double cv = 0.0;
        double fleck = 0.0;
        double Upsilon = 0.0;
        double Gamma = 0.0;
        double betaCdtF = 0.0;
        bool useNZero = false;
        GroupArray oldRadiationEnergy{};
        GroupArray occupation{};
        GroupArray D{};
        GroupArray M{};
        GroupArray rowS{};
        GroupArray Lambda{};
        GroupArray Bbase{};
        GroupArray Bcorr{};
        GroupArray Btotal{};
        GroupArray Bpos{};
        GroupArray Bres{};
        GroupArray baseEffectiveOpacity{};
        GroupArray comptonOutRate{};
        GroupCdfMatrix comptonTargetCdf{};
        GroupMatrix tau{};
        GroupMatrix dtau_dUm{};
        GroupMatrix S{};
        GroupMatrix dSdUm{};
        GroupMatrix segmentKernel{};
        GroupMatrix residualKernel{};
    };

    RadiationIMC(Tessellation3D &grid, const std::shared_ptr<BoundaryCond> &boundary, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, std::shared_ptr<EquationOfState> eos, std::shared_ptr<OpacityCalculator> opacity, RadiationIMCParameters parameters);

    std::vector<Particle> preStep(double fullDt) override;

    Functionality step(Particle &particle, std::vector<Particle> &particlesToAdd) override;

    void postStep(const std::vector<Particle> &particles, double fullDt) override;

    Particle generateSingleParticle(size_t cellIndex, const ComputationalCell3D &cell) const override;

    std::vector<Particle> generateInitialParticles(size_t particlesPerCell) override;

    void adjustExistingParticles(std::vector<Particle> &particles, double fullDt) override;

    inline const std::vector<double> &getFactorFleck(void) const{return this->factorFleck;}

    inline const std::vector<double> &getPlanckOpacities(void) const {return this->planckOpacities;}

    inline const std::vector<ComptonCellData> &getComptonData(void) const {return this->comptonData;}

    inline const GroupArray &getComptonGroupCenters(void) const {return this->comptonGroupCenters;}

    inline const GroupArray &getComptonGroupWidths(void) const {return this->comptonGroupWidths;}

private:    
    std::vector<Particle> generateParticles(double fullDt);
    std::vector<Particle> generateComptonParticles(double fullDt);
    void precomputeComptonData(double fullDt);
    void initializeComptonGroups();
    void initializeComptonMatrixGenerator();
    void buildComptonMatricesForCell(const ComputationalCell3D &cell, size_t cellIndex, bool calculateN, ComptonCellData &cd);
    void recomputeComptonContractions(ComptonCellData &cd);
    void buildComptonEventData(size_t cellIndex, ComptonCellData &cd);
    void buildComptonSources(double fullDt, ComptonCellData &cd);
    void applyComptonScatterEvent(size_t cellIndex, const ComputationalCell3D &cell, size_t sourceGroup, const Vector3D &oldVelocity, double oldWeight, double dopplerShift, Particle &particle);
    void applyComptonDeterministicSegment(size_t cellIndex, const ComputationalCell3D &cell, size_t sourceGroup, double dt, double dopplerShift, const Vector3D &oldVelocity, Particle &particle);
    void applyComptonResidualCorrection(double fullDt);
    void reconcileComptonParticles(std::vector<Particle> &particles);
    void resetComptonDiagnostics();
    void printComptonDiagnostics();
    void validateComptonParity(size_t cellIndex, const ComptonCellData &cd) const;
    double frequencyForComptonGroup(size_t group) const;
    size_t sampleComptonCdf(const GroupCdf &cdf, double random) const;
    static GroupCdf buildSafeComptonCdf(const GroupArray &weights);

    std::vector<double> factorFleck;
    std::vector<double> planckOpacities;
    std::shared_ptr<MultigroupOpacity> multigroupOpacity;
    std::vector<ComptonCellData> comptonData;
    GroupArray comptonGroupCenters{};
    GroupArray comptonGroupWidths{};
    bool comptonGroupsInitialized = false;
    std::unique_ptr<ComptonMatrixMC> comptonMatrixGen;

    bool withHydro;
    bool diffusionPressureGradient;
    bool MMC;
    size_t newPhotonsPerCell;
    bool withRandomWalk;
    double rwMinCellOpticalDepth;
    double rwMinParticleOpticalDepth;
    bool noHydroFeedback;
    bool withEgTimeAvg;
    bool withCompton;
    bool comptonUseInduced;
    bool comptonAllowNZeroFallback;
    bool comptonDebugParityCheck;
    bool comptonDiagnostics;
    size_t comptonMatrixSamples;
    ComptonTransportMode comptonTransportMode;
    double comptonSourceMaterialExchange = 0.0;
    double comptonContinuousMaterialExchange = 0.0;
    double comptonImplicitMaterialExchange = 0.0;
    double comptonResidualMaterialExchange = 0.0;
    double comptonRemovalMaterialExchange = 0.0;
    double comptonSourceMaterialExchangeAbs = 0.0;
    double comptonContinuousMaterialExchangeAbs = 0.0;
    double comptonImplicitMaterialExchangeAbs = 0.0;
    double comptonResidualMaterialExchangeAbs = 0.0;
    double comptonRemovalMaterialExchangeAbs = 0.0;
    double comptonSourceBposEnergy = 0.0;
    double comptonSourceBresEnergy = 0.0;
    double comptonSourceBtotalEnergy = 0.0;
    GroupArray comptonSourceBposEnergyByGroup{};
    GroupArray comptonSourceBresEnergyByGroup{};
    GroupArray comptonSourceBtotalEnergyByGroup{};
    double comptonResidualRadiationDelta = 0.0;
    double comptonTotalMaterialEnergy = 0.0;
    double comptonTotalRadiationEnergy = 0.0;
    GroupArray comptonContinuousMaterialExchangeByGroup{};
    GroupArray comptonContinuousMaterialExchangeAbsByGroup{};
    GroupArray comptonImplicitMaterialExchangeBySourceGroup{};
    GroupArray comptonImplicitMaterialExchangeAbsBySourceGroup{};
    GroupArray comptonImplicitMaterialExchangeByTargetGroup{};
    GroupArray comptonImplicitMaterialExchangeAbsByTargetGroup{};
    double comptonMaxCdtComptonOutRate = 0.0;
    double comptonMaxCdtBaseEffectiveOpacity = 0.0;
    double comptonMaxCdtSegmentKernel = 0.0;
    double comptonMaxCdtResidualKernel = 0.0;
    double comptonMaxCdtFleckAbsorptionOpacity = 0.0;
    double comptonMaxCdtPlanckOpacity = 0.0;
    size_t comptonMaxFleckAbsorptionCell = std::numeric_limits<size_t>::max();
    size_t comptonMaxFleckAbsorptionGroup = std::numeric_limits<size_t>::max();
    double comptonMaxFleckAbsorptionOpacity = 0.0;
    double comptonMaxFleckAbsorptionFleck = 0.0;
    double comptonMaxFleckAbsorptionGroupEnergy = 0.0;
    double comptonMaxFleckAbsorptionPlanckEnergy = 0.0;
    double comptonMaxFleckAbsorptionGroupExcess = 0.0;
    double comptonMaxFleckAbsorptionTemperature = 0.0;
    double comptonMaxFleckAbsorptionDensity = 0.0;
    double comptonMaxFleckAbsorptionMaterialEnergy = 0.0;
    size_t comptonCgOpacityLimitCount = 0;
    size_t comptonMaterialOpacityLimitCount = 0;
    double comptonMaxOpacityReduction = 1.0;
    double comptonMaxOpacityReductionRaw = 0.0;
    double comptonMaxOpacityReductionFinal = 0.0;
    size_t comptonMaxOpacityReductionCell = std::numeric_limits<size_t>::max();
    size_t comptonMaxOpacityReductionGroup = std::numeric_limits<size_t>::max();
    double comptonMaxContinuousDepositAbs = 0.0;
    double comptonMaxContinuousDeposit = 0.0;
    double comptonMaxContinuousOldWeight = 0.0;
    double comptonMaxContinuousDt = 0.0;
    double comptonMaxContinuousTau = 0.0;
    double comptonMaxContinuousOpacity = 0.0;
    double comptonMaxContinuousFleck = 0.0;
    double comptonMaxContinuousFrequency = 0.0;
    size_t comptonMaxContinuousCell = std::numeric_limits<size_t>::max();
    size_t comptonMaxContinuousGroup = std::numeric_limits<size_t>::max();
    double comptonMaxEventDepositAbs = 0.0;
    double comptonMaxEventDeposit = 0.0;
    double comptonMaxEventOldWeight = 0.0;
    double comptonMaxEventNewWeight = 0.0;
    double comptonMaxEventRatio = 0.0;
    double comptonMaxEventFleck = 0.0;
    size_t comptonMaxEventCell = std::numeric_limits<size_t>::max();
    size_t comptonMaxEventSourceGroup = std::numeric_limits<size_t>::max();
    size_t comptonMaxEventTargetGroup = std::numeric_limits<size_t>::max();
    size_t comptonSegmentUpdateCount = 0;
    double comptonSegmentNegativeCorrection = 0.0;
    double comptonSegmentNegativeCorrectionAbs = 0.0;
    double comptonMaxSegmentDepositAbs = 0.0;
    double comptonMaxSegmentDeposit = 0.0;
    double comptonMaxSegmentOldWeight = 0.0;
    double comptonMaxSegmentNewWeight = 0.0;
    double comptonMaxSegmentDt = 0.0;
    double comptonMaxSegmentNegativeCorrection = 0.0;
    size_t comptonMaxSegmentCell = std::numeric_limits<size_t>::max();
    size_t comptonMaxSegmentSourceGroup = std::numeric_limits<size_t>::max();
    size_t comptonMaxSegmentTargetGroup = std::numeric_limits<size_t>::max();
    double comptonMinGroupEnergy = std::numeric_limits<double>::infinity();
    double comptonMaxGroupEnergy = -std::numeric_limits<double>::infinity();
    double comptonMinFleck = std::numeric_limits<double>::infinity();
    double comptonMaxFleck = -std::numeric_limits<double>::infinity();
    double comptonMinGamma = std::numeric_limits<double>::infinity();
    double comptonMaxGamma = -std::numeric_limits<double>::infinity();
    double comptonMinUpsilon = std::numeric_limits<double>::infinity();
    double comptonMaxUpsilon = -std::numeric_limits<double>::infinity();
    size_t comptonNZeroFallbackCount = 0;
    size_t comptonImplicitEventCount = 0;

    std::unique_ptr<RandomWalk> randomWalk;
    std::vector<bool> rwCellEligible;
    std::vector<double> rwCellTotalOpacity;
    std::vector<PGRWCellData> rwCellData;
    size_t rwStepCount = 0;

    bool tryRandomWalkStep(Particle &particle, Functionality &functionality, double dopplerShift);
    void precomputeRandomWalkData();
};

#endif // RADIATION_IMC_HPP
