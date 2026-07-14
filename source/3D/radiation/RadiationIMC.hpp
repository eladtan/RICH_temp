#ifndef RADIATION_IMC_HPP
#define RADIATION_IMC_HPP

#include <array>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include "MonteCarloPhysics3D.hpp"
#include "MultigroupOpacity.hpp"
#include "3D/monte/Voronoi3DMovement.hpp"
#include "RandomWalk.hpp"
#include "Radiation/CMMC/src/compton_matrix_mc.hpp"

class SphericalObserver;

enum class ComptonInducedMode
{
    RadiationField,
    AdaptivePlanckFallback
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
    bool withDDMC = false;
    double ddmcMinCellOpticalDepth = 15.0;
    bool ddmcUseMovingInterfaceCorrection = true;
    double ddmcMaxInterfaceVelocityOverC = 0.1;
    double ddmcInterfaceTargetWeightRatio = 2.0;
    size_t ddmcMaxInterfaceSplits = 64;
    bool ddmcUseMultigroupPGRW = false;
    size_t ddmcMaxGroupCutoff = ENERGY_GROUPS_NUM;
    bool ddmcInterfaceDiagnostics = false;
    bool noHydroFeedback = false;
    bool withEgTimeAvg = false;
    bool capAbsorptionOpacity = false;
    bool withCompton = false;
    bool comptonUseInduced = true;
    ComptonInducedMode comptonInducedMode = ComptonInducedMode::AdaptivePlanckFallback;
    bool comptonAllowNZeroFallback = true;
    bool comptonDebugParityCheck = false;
    bool comptonCheckSignedTallies = false;
    bool comptonDiagnostics = false;
    bool comptonAngleDependent = true;
    double comptonSignedTallyTolerance = 1e-10;
    size_t comptonMatrixSamples = 200000;

    struct PostProcessParameters
    {
        bool enabled = false;
        double sourceDt = 0.0;
        double transportTime = 0.0;
        bool forceGreyFleckOne = true;
        bool useCellVelocities = true;
        struct PolarizationParameters
        {
            bool enabled = false;
            int manualScatteringsAfterAcceleration = 4;
            double depolarizationScatterings = 2.0;
            std::string acceleratedClosure = "damped_last_scatterings";
        } polarization;
    } postProcess;

    friend std::ostream &operator<<(std::ostream &os, const RadiationIMCParameters &parameters);
};

enum class DDMCFaceKind
{
    Internal,
    InterfaceToIMC
};

struct DDMCFaceLeak
{
    size_t faceIndex = std::numeric_limits<size_t>::max();
    size_t nextCellIndex = std::numeric_limits<size_t>::max();
    DDMCFaceKind kind = DDMCFaceKind::InterfaceToIMC;
    double rate = 0.0;
    double internalRate = 0.0;
    double boundaryRate = 0.0;
    double ddmcRate = 0.0;
    double transportRate = 0.0;
    double sourceBandMass = 1.0;
    double commonBandMass = 1.0;
    double ddmcFraction = 0.0;
    double area = 0.0;
    double sourceDistanceToFace = 0.0;
    double targetDistanceToFace = 0.0;
    double conductance = 0.0;
    bool targetDDMCEligible = false;
    size_t targetGroupCutoff = 0;
    Vector3D outwardNormal = Vector3D(0.0, 0.0, 0.0);
};

struct DDMCCellData
{
    bool eligible = false;
    bool boundaryExcluded = false;
    bool observerExcluded = false;
    size_t rigidBoundaryFaceCount = 0;
    size_t unsupportedBoundaryFaceCount = 0;
    size_t firstUnsupportedBoundaryFace = std::numeric_limits<size_t>::max();
    size_t groupCutoff = ENERGY_GROUPS_NUM;
    double sigmaA = 0.0;
    double sigmaT = 0.0;
    double sigmaEnergyAbs = 0.0;
    double sigmaMomentum = 0.0;
    double sigmaDiffusion = 0.0;
    double sigmaParticleGate = 0.0;
    double sigmaGroupExit = 0.0;
    double diffusionCoefficient = 0.0;
    double gamma = 1.0;
    double totalLeakRate = 0.0;
    double faceAreaSum = 0.0;
    double velocityDivergence = 0.0;
    double maxFaceVelocityJumpOverC = 0.0;
    std::array<double, 6> fluxMatrix{};
    std::vector<DDMCFaceLeak> faceLeaks;
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

    struct SourceAllocationSummary
    {
        bool adaptiveEnabled = false;
        size_t totalPhotons = 0;
        size_t sourceCells = 0;
        size_t boostedCells = 0;
        size_t learnedCells = 0;
        size_t learnedBoostedCells = 0;
        size_t learnedPhotons = 0;
        size_t learnedExtraPhotons = 0;
        size_t minPhotons = 0;
        size_t maxPhotons = 0;
        size_t learnedMinPhotons = 0;
        size_t learnedMaxPhotons = 0;
        double adaptiveScoreSum = 0.0;
    };

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
        bool usePlanckInduced = false;
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
        GroupMatrix Ktotal{};
        GroupArray comptonMu{};
        GroupArray comptonMh{};
        GroupArray riskScore{};
        std::array<size_t, ENERGY_GROUPS_NUM> riskTargetPackets{};
    };

    enum class ComptonOccupationMode
    {
        Zero,
        RadiationField,
        PlanckFunction
    };

    RadiationIMC(Tessellation3D &grid, const std::shared_ptr<BoundaryCond> &boundary, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, std::shared_ptr<EquationOfState> eos, std::shared_ptr<OpacityCalculator> opacity, RadiationIMCParameters parameters);

    std::vector<Particle> preStep(double fullDt) override;

    Functionality step(Particle &particle, std::vector<Particle> &particlesToAdd) override;

    void postStep(const std::vector<Particle> &particles, double fullDt) override;

    size_t getRandomWalkStepCount() const override { return this->rwStepCount; }
    size_t getDDMCStepCount() const override { return this->ddmcStepCount; }
    size_t getDDMCLeakCount() const override { return this->ddmcLeakCount; }
    size_t getDDMCCensusCount() const override { return this->ddmcCensusCount; }
    size_t getDDMCUpscatterCount() const override { return this->ddmcUpscatterCount; }
    size_t getDDMCFallbackCount() const override { return this->ddmcFallbackCount; }
#ifdef RICH_IMC_DDMC_ENABLED
    std::string getAccelerationDebugInfo(size_t cellIndex, double frequency) const override;
    std::string getDDMCFaceDiagnosticsTSV(double xMin, double xMax) const;
    std::string getDDMCInterfaceEventDiagnosticsTSV(double xMin,
                                                     double xMax) const;
#else
    std::string getAccelerationDebugInfo(size_t, double) const override { return std::string(); }
    std::string getDDMCFaceDiagnosticsTSV(double, double) const
        { return std::string(); }
    std::string getDDMCInterfaceEventDiagnosticsTSV(double, double) const
        { return std::string(); }
#endif

    Particle generateSingleParticle(size_t cellIndex, const ComputationalCell3D &cell) const override;

    std::vector<Particle> generateInitialParticles(size_t particlesPerCell) override;

    void adjustExistingParticles(std::vector<Particle> &particles, double fullDt) override;

    inline const std::vector<double> &getFactorFleck(void) const{return this->factorFleck;}

    inline const std::vector<double> &getPlanckOpacities(void) const {return this->planckOpacities;}

    inline const std::vector<ComptonCellData> &getComptonData(void) const {return this->comptonData;}

    inline const GroupArray &getComptonGroupCenters(void) const {return this->comptonGroupCenters;}

    inline const GroupArray &getComptonGroupWidths(void) const {return this->comptonGroupWidths;}

    void setObserver(std::shared_ptr<SphericalObserver> observer);
    void setNewPhotonsPerCell(size_t n) { newPhotonsPerCell = n; }
    void setAdaptiveSourceCellScores(std::unordered_map<size_t, double> scores,
                                     double strength, double maxFactor,
                                     double learnedReserveFrac,
                                     double learnedMinFactor,
                                     double observerBudgetMultiplier,
                                     size_t learnedMinPhotons = 0,
                                     size_t learnedMaxPhotons = 0,
                                     double scorePower = 1.0);
    void clearAdaptiveSourceCellScores();

    struct GroupSamplingDiagnostics
    {
        size_t totalSampled = 0;
        size_t cellsWithGroupScores = 0;
        double weightCorrectionMin = 1.0;
        double weightCorrectionMax = 1.0;
        double weightCorrectionSum = 0.0;
        size_t weightCorrectionCount = 0;
        size_t weightCorrectionCapped = 0;
        size_t weightCorrectionFallback = 0;
        size_t invalidPdfFallback = 0;
        size_t invalidPdfFallbackPackets = 0;
        double sampledEnergy = 0.0;
        double cappedEnergy = 0.0;
        double cappedEnergyFraction = 0.0;
        bool estimatorPotentiallyBiased = false;
    };

    void setAdaptiveSourceCellGroupScores(
        std::unordered_map<size_t, GroupArray> scores,
        double strength,
        double pdfFloor,
        double maxBias,
        double maxWeightCorrection);
    void clearAdaptiveSourceCellGroupScores();
    GroupSamplingDiagnostics getLastGroupSamplingDiagnostics() const { return lastGroupSamplingDiagnostics_; }
    void setSourceEmissionControl(bool useLearnedScores, bool includeUniformBase,
                                  size_t baseMultiplier,
                                  size_t learnedBoostFactor = 20,
                                  size_t learnedExtraBudget = 0);
    void clearSourceEmissionControl();
    SourceAllocationSummary getLastSourceAllocationSummary() const { return lastSourceAllocationSummary_; }
    std::vector<size_t> const &getLastSourcePhotonsPerCell() const { return lastSourcePhotonsPerCell_; }
private:    
    std::vector<Particle> generateParticles(double fullDt);
    std::vector<Particle> generateComptonParticles(double fullDt);
    void precomputeComptonData(double fullDt);
    void initializeComptonGroups();
    void initializeComptonMatrixGenerator();
    void buildComptonMatricesForCell(const ComputationalCell3D &cell, size_t cellIndex, ComptonOccupationMode occupationMode, ComptonCellData &cd);
    void recomputeComptonContractions(ComptonCellData &cd);
    void buildComptonEventData(size_t cellIndex, ComptonCellData &cd);
    void buildComptonSources(double fullDt, ComptonCellData &cd);
    void applyComptonScatterEvent(size_t cellIndex, const ComputationalCell3D &cell, size_t sourceGroup, const Vector3D &oldVelocity, double oldWeight, double dopplerShift, Particle &particle);
    void computeComptonRiskForCell(size_t cellIndex, double fullDt, ComptonCellData &cd);
    void splitComptonRiskyParticles(std::vector<Particle> &particles, double fullDt);
    void applyComptonEndOfStepCorrection(double fullDt);
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
    std::vector<std::array<size_t, ENERGY_GROUPS_NUM>> lastComptonPacketCounts_;
    std::vector<GroupArray> lastComptonMaxPacketWeight_;
    double comptonRiskPrecomputeDt_ = -1.0;
    bool comptonDataReusableInPreStep_ = false;

    bool withHydro;
    bool diffusionPressureGradient;
    bool MMC;
    size_t newPhotonsPerCell;
    bool withRandomWalk;
    double rwMinCellOpticalDepth;
    double rwMinParticleOpticalDepth;
    bool withDDMC;
    double ddmcMinCellOpticalDepth;
    bool ddmcUseMovingInterfaceCorrection;
    double ddmcMaxInterfaceVelocityOverC;
    double ddmcInterfaceTargetWeightRatio;
    size_t ddmcMaxInterfaceSplits;
    bool ddmcUseMultigroupPGRW;
    size_t ddmcMaxGroupCutoff;
    bool ddmcInterfaceDiagnostics;
    bool noHydroFeedback;
    bool withEgTimeAvg;
    bool capAbsorptionOpacity;
    bool withCompton;
    RadiationIMCParameters::PostProcessParameters postProcess_;
    bool useTransportVelocities_ = true;
    bool comptonUseInduced;
    ComptonInducedMode comptonInducedMode;
    bool comptonAllowNZeroFallback;
    bool comptonAngleDependent;
    bool comptonDebugParityCheck;
    bool comptonCheckSignedTallies;
    bool comptonDiagnostics;
    double comptonSignedTallyTolerance;
    size_t comptonMatrixSamples;
    double comptonSourceMaterialExchange = 0.0;
    double comptonContinuousMaterialExchange = 0.0;
    double comptonImplicitMaterialExchange = 0.0;
    double comptonRemovalMaterialExchange = 0.0;
    double comptonMinGroupEnergy = std::numeric_limits<double>::infinity();
    double comptonMaxGroupEnergy = -std::numeric_limits<double>::infinity();
    double comptonProjectedRadiationEnergy = 0.0;
    double comptonMinFleck = std::numeric_limits<double>::infinity();
    double comptonMaxFleck = -std::numeric_limits<double>::infinity();
    double comptonMinGamma = std::numeric_limits<double>::infinity();
    double comptonMaxGamma = -std::numeric_limits<double>::infinity();
    double comptonMinUpsilon = std::numeric_limits<double>::infinity();
    double comptonMaxUpsilon = -std::numeric_limits<double>::infinity();
    size_t comptonNZeroFallbackCount = 0;
    size_t comptonImplicitEventCount = 0;
    size_t comptonOpacityLimitedGroupCount = 0;
    size_t comptonProjectedNegativeGroupCount = 0;

    std::unique_ptr<RandomWalk> randomWalk;
    std::vector<bool> rwCellEligible;
    std::vector<double> rwCellTotalOpacity;
    std::vector<PGRWCellData> rwCellData;
    size_t rwStepCount = 0;

    std::vector<DDMCCellData> ddmcCellData;
    std::vector<int> ddmcPointEligible;
    std::vector<double> ddmcPointDiffusionCoefficient;
    std::vector<double> ddmcPointSigmaDiffusion;
    std::vector<size_t> ddmcPointGroupCutoff;
    std::vector<Vector3D> ddmcPointVelocity;
    std::vector<size_t> ddmcPointCellID;
    std::vector<Vector3D> ddmcFluxRhsIntegrated;
    size_t ddmcStepCount = 0;
    size_t ddmcLeakCount = 0;
    size_t ddmcResidentLeakCount = 0;
    size_t ddmcTransportLeakCount = 0;
    size_t ddmcRemoteResidentLeakCount = 0;
    size_t ddmcMomentumFeedbackCount = 0;
    size_t ddmcMomentumMatrixFallbackCount = 0;
    size_t ddmcMovingMediumUpdateCount = 0;
    size_t ddmcFaceFrameShiftCount = 0;
    double ddmcMaxMovingMediumLogShift = 0.0;
    double ddmcMaxFaceFrameLogShift = 0.0;
    double ddmcFaceFluxEnergy = 0.0;
    double ddmcFaceFluxMpiEnergy = 0.0;
    double ddmcMaterialEnergyExchangeCo = 0.0;
    double ddmcMaterialEnergyExchangeLab = 0.0;
    Vector3D ddmcMaterialMomentumExchangeLab = Vector3D(0.0, 0.0, 0.0);
    Vector3D ddmcFluxMomentumExchangeLab = Vector3D(0.0, 0.0, 0.0);
    Vector3D ddmcAppliedMomentumExchangeLab = Vector3D(0.0, 0.0, 0.0);
    double ddmcLocalFaceFluxPairResidualMax = 0.0;
    double ddmcWeightRatioMax = 0.0;
    double ddmcWeightRatioSum = 0.0;
    std::vector<double> ddmcWeightRatioSamples;
    size_t ddmcCensusCount = 0;
    size_t ddmcUpscatterCount = 0;
    size_t ddmcFallbackCount = 0;
    size_t ddmcMpiFaceFluxReductionCount = 0;
    size_t ddmcInterfaceFluxTallyCount = 0;
    size_t ddmcBoundaryFluxTallyCount = 0;
    size_t ddmcObserverEnergyOnlyTallyCount = 0;
    size_t ddmcLocalFaceFluxPairCheckCount = 0;
    size_t ddmcWeightRatioCount = 0;
    size_t ddmcWeightRatioSamplesDropped = 0;
    size_t ddmcWeightRatioOutlierCount = 0;
    size_t ddmcInterfaceIncidentCount = 0;
    size_t ddmcInterfaceAdmissionCount = 0;
    size_t ddmcInterfaceReflectionCount = 0;
    size_t ddmcInterfaceMovingFactorCount = 0;
    size_t ddmcInterfaceMovingFallbackCount = 0;
    size_t ddmcInterfaceSplitPacketCount = 0;
    double ddmcInterfaceMinimumMu = std::numeric_limits<double>::infinity();
    double ddmcInterfaceMaximumFactor = 1.0;
    double ddmcLeakReciprocityResidualMax = 0.0;
    size_t ddmcLeakReciprocityCheckCount = 0;
    size_t ddmcLeakInvalidGeometryCount = 0;
    size_t ddmcInterfaceBypassCount = 0;
    size_t ddmcDopplerCutoffExitCount = 0;

#ifdef RICH_IMC_DDMC_ENABLED
    enum class DDMCDiagnosticEventKind : unsigned char
    {
        IMCCandidate,
        IMCFrequencyReject,
        IMCIncident,
        IMCAdmitted,
        IMCReflected,
        IMCBypass,
        DDMCToDDMC,
        DDMCToIMC
    };

    static constexpr size_t DDMC_DIAGNOSTIC_GREY_GROUP =
        std::numeric_limits<size_t>::max();

    struct DDMCDiagnosticEventKey
    {
        DDMCDiagnosticEventKind kind = DDMCDiagnosticEventKind::IMCCandidate;
        size_t faceIndex = std::numeric_limits<size_t>::max();
        size_t sourceCellID = std::numeric_limits<size_t>::max();
        size_t targetCellID = std::numeric_limits<size_t>::max();
        size_t group = DDMC_DIAGNOSTIC_GREY_GROUP;

        bool operator<(DDMCDiagnosticEventKey const &other) const
        {
            return std::tie(kind, faceIndex, sourceCellID, targetCellID, group) <
                   std::tie(other.kind, other.faceIndex, other.sourceCellID,
                            other.targetCellID, other.group);
        }
    };

    struct DDMCDiagnosticEventAccumulator
    {
        size_t faceIndex = std::numeric_limits<size_t>::max();
        size_t sourceCellID = std::numeric_limits<size_t>::max();
        size_t targetCellID = std::numeric_limits<size_t>::max();
        size_t group = DDMC_DIAGNOSTIC_GREY_GROUP;
        size_t sourceGroupCutoff = 0;
        size_t targetGroupCutoff = 0;
        double faceX = std::numeric_limits<double>::quiet_NaN();
        double sourceGeneratorX = std::numeric_limits<double>::quiet_NaN();
        double targetGeneratorX = std::numeric_limits<double>::quiet_NaN();
        size_t count = 0;
        double signedEnergy = 0.0;
        double absoluteEnergy = 0.0;
        double muSum = 0.0;
        size_t muCount = 0;
        double admissionProbabilitySum = 0.0;
        size_t admissionProbabilityCount = 0;
    };

    std::map<DDMCDiagnosticEventKey, DDMCDiagnosticEventAccumulator>
        ddmcDiagnosticEvents;

    void recordDDMCDiagnosticEvent(DDMCDiagnosticEventKind kind,
                                   size_t sourceCellIndex,
                                   size_t targetCellIndex,
                                   size_t faceIndex,
                                   size_t group,
                                   double energy,
                                   size_t sourceGroupCutoff,
                                   size_t targetGroupCutoff,
                                   double mu,
                                   double admissionProbability);
#endif

    std::shared_ptr<SphericalObserver> observer_;
    std::unordered_map<size_t, double> adaptiveSourceScores_;
    bool adaptiveSourceScoresEnabled_ = false;
    double adaptiveSourceStrength_ = 0.0;
    double adaptiveSourceMaxFactor_ = 1.0;
    double adaptiveSourceLearnedReserveFrac_ = 0.0;
    double adaptiveSourceLearnedMinFactor_ = 1.0;
    double adaptiveSourceObserverBudgetMultiplier_ = 1.0;
    size_t adaptiveSourceLearnedMinPhotons_ = 0;
    size_t adaptiveSourceLearnedMaxPhotons_ = 0;
    double adaptiveSourceScorePower_ = 1.0;
    bool sourceEmissionControlEnabled_ = false;
    bool sourceEmissionUseLearnedScores_ = false;
    bool sourceEmissionIncludeUniformBase_ = true;
    size_t sourceEmissionBaseMultiplier_ = 1;
    size_t sourceEmissionLearnedBoostFactor_ = 20;
    size_t sourceEmissionLearnedExtraBudget_ = 0;
    std::vector<size_t> lastSourcePhotonsPerCell_;
    SourceAllocationSummary lastSourceAllocationSummary_;

    std::unordered_map<size_t, GroupArray> adaptiveSourceCellGroupScores_;
    bool adaptiveSourceCellGroupScoresEnabled_ = false;
    double adaptiveGroupStrength_ = 0.0;
    double adaptiveGroupPdfFloor_ = 0.0;
    double adaptiveGroupMaxBias_ = 1.0;
    double adaptiveGroupMaxWeightCorrection_ = 1.0;
    GroupSamplingDiagnostics lastGroupSamplingDiagnostics_;

    bool tryRandomWalkStep(Particle &particle, Functionality &functionality, double dopplerShift);
    void precomputeRandomWalkData();
#ifdef RICH_IMC_DDMC_ENABLED
    void precomputeDDMCData();
    Vector3D sampleDDMCTransportLocation(size_t cellIndex);
    double sampleDDMCPlanckFrequency(size_t cellIndex,
                                     size_t beginGroup,
                                     size_t endGroup);
    void validateDDMCTransportLocation(size_t cellIndex,
                                       Vector3D const &location,
                                       char const *context) const;
    bool tryIMCToDDMCInterface(Particle &particle, Functionality &functionality,
                               std::vector<Particle> &particlesToAdd,
                               size_t sourceCellIndex, size_t targetCellIndex,
                               size_t faceIndex);
    bool tryDDMCStep(Particle &particle, Functionality &functionality, double dopplerShift);
    void reduceDDMCFaceFluxTallies();
    void applyDDMCMomentumFeedback(double fullDt);
    void tallyDDMCFaceFlux(size_t sourceCellIndex, const DDMCFaceLeak &faceLeak,
                           double comovingEnergy, const Vector3D &fluxDirection,
                           bool includeTarget);
    void recordDDMCWeightRatio(double weight, double initialWeight);
    void tallyDDMCMaterialEnergy(size_t cellIndex, double comovingEnergy,
                                 const Vector3D &cellVelocity);
#endif
};

#endif // RADIATION_IMC_HPP
