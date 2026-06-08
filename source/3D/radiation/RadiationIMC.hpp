#ifndef RADIATION_IMC_HPP
#define RADIATION_IMC_HPP

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "MonteCarloPhysics3D.hpp"
#include "MultigroupOpacity.hpp"
#include "3D/monte/Voronoi3DMovement.hpp"
#include "RandomWalk.hpp"
#include "Radiation/CMMC/src/compton_matrix_mc.hpp"
#include "PostProcessIMCHelpers.hpp"
#include "PeelOffTypes.hpp"
#include "FaceExitInfo.hpp"
#ifdef RICH_MPI
#include <boost/container/flat_map.hpp>
#endif

class SphericalObserver;

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
    double ddmcMinParticleOpticalDepth = 5.0;
    bool ddmcUseMultigroupPGRW = true;
    bool noHydroFeedback = false;
    bool withEgTimeAvg = false;
    bool withCompton = false;
    bool comptonUseInduced = true;
    bool comptonAllowNZeroFallback = true;
    bool comptonAngleDependent = true;
    size_t comptonMatrixSamples = 200000;

    RadiationIMCPostProcessConfig postProcess;

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

    struct SourceAllocationSummary
    {
        bool adaptiveEnabled = false;
        unsigned long long totalPhotons = 0;
        unsigned long long sourceCells = 0;
        unsigned long long boostedCells = 0;
        unsigned long long learnedCells = 0;
        unsigned long long learnedBoostedCells = 0;
        unsigned long long learnedPhotons = 0;
        unsigned long long learnedExtraPhotons = 0;
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
    std::string getAccelerationDebugInfo(size_t cellIndex, double frequency) const override;

    Particle generateSingleParticle(size_t cellIndex, const ComputationalCell3D &cell) const override;

    std::vector<Particle> generateInitialParticles(size_t particlesPerCell) override;

    void adjustExistingParticles(std::vector<Particle> &particles, double fullDt) override;

    inline const std::vector<double> &getFactorFleck(void) const{return this->factorFleck;}

    inline const std::vector<double> &getPlanckOpacities(void) const {return this->planckOpacities;}

    inline const std::vector<ComptonCellData> &getComptonData(void) const {return this->comptonData;}

    inline const GroupArray &getComptonGroupCenters(void) const {return this->comptonGroupCenters;}

    inline const GroupArray &getComptonGroupWidths(void) const {return this->comptonGroupWidths;}

    void setObserver(std::shared_ptr<SphericalObserver> observer);
    std::shared_ptr<SphericalObserver> getObserver() const;
    bool isPostProcessMode() const;
    void setAdaptiveSourceCellScores(std::unordered_map<size_t, double> scores,
                                     double strength, double maxFactor,
                                     double learnedReserveFrac = 0.0,
                                     double learnedMinFactor = 1.0);
    void clearAdaptiveSourceCellScores();
    void setNewPhotonsPerCell(size_t newPhotonsPerCell);
    void setSourceEmissionControl(bool learnedCellsOnly,
                                  bool forceUniformPhotons,
                                  size_t uniformPhotons,
                                  size_t learnedMinPhotons = 0,
                                  size_t learnedMaxPhotons = 0);
    void clearSourceEmissionControl();
    SourceAllocationSummary getLastSourceAllocationSummary() const;
    std::vector<size_t> const& getLastSourcePhotonsPerCell() const;
    void queueExternalSourcePeelOffEvents(std::vector<Particle> const& particles,
                                          double eventTimeLeft);
    void drainPendingCollectiveWork() override;
    bool isPeelOffProgressEnabled() const override;
    MonteCarloPeelOffProgressSnapshot getPeelOffProgressSnapshot() const override;

private:    
    struct DDMCFaceLeak
    {
        size_t faceIndex = std::numeric_limits<size_t>::max();
        size_t nextCellIndex = std::numeric_limits<size_t>::max();
        double rate = 0.0;
    };

    struct DDMCCellData
    {
        bool eligible = false;
        bool observerExcluded = false;
        bool boundaryExcluded = false;
        size_t rigidBoundaryFaceCount = 0;
        size_t unsupportedBoundaryFaceCount = 0;
        size_t firstUnsupportedBoundaryFace = std::numeric_limits<size_t>::max();
        double sigmaT = 0.0;
        double sigmaA = 0.0;
        double diffusionCoefficient = 0.0;
        double gamma = 1.0;
        size_t groupCutoff = 0;
        double totalLeakRate = 0.0;
        std::vector<DDMCFaceLeak> faceLeaks;
    };

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
    void applyComptonEndOfStepCorrection(double fullDt);
    void reconcileComptonParticles(std::vector<Particle> &particles);
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
    bool withDDMC;
    double ddmcMinCellOpticalDepth;
    double ddmcMinParticleOpticalDepth;
    bool ddmcUseMultigroupPGRW;
    bool noHydroFeedback;
    bool withEgTimeAvg;
    bool withCompton;
    bool comptonUseInduced;
    bool comptonAllowNZeroFallback;
    bool comptonAngleDependent;
    size_t comptonMatrixSamples;
    bool useTransportVelocities_ = false;
    bool adaptiveSourceCellsEnabled_ = false;
    double adaptiveSourceStrength_ = 0.0;
    double adaptiveSourceMaxFactor_ = 20.0;
    double adaptiveSourceLearnedReserveFrac_ = 0.0;
    double adaptiveSourceLearnedMinFactor_ = 1.0;
    std::unordered_map<size_t, double> adaptiveSourceScoreByCellID_;
    SourceAllocationSummary lastSourceAllocationSummary_;
    std::vector<size_t> lastSourcePhotonsPerCell_;
    bool sourceLearnedCellsOnly_ = false;
    bool sourceForceUniformPhotons_ = false;
    size_t sourceUniformPhotons_ = 1;
    size_t sourceLearnedMinPhotons_ = 0;
    size_t sourceLearnedMaxPhotons_ = 0;

    std::unique_ptr<RandomWalk> randomWalk;
    std::vector<bool> rwCellEligible;
    std::vector<double> rwCellTotalOpacity;
    std::vector<PGRWCellData> rwCellData;
    size_t rwStepCount = 0;

    std::vector<DDMCCellData> ddmcCellData;
    size_t ddmcStepCount = 0;
    size_t ddmcLeakCount = 0;
    size_t ddmcCensusCount = 0;
    size_t ddmcUpscatterCount = 0;
    size_t ddmcFallbackCount = 0;
    size_t ddmcFallbackOutsideCellCount = 0;
    size_t ddmcFallbackLeakFaceDistanceCount = 0;
    size_t ddmcFallbackInvalidLeakFaceDistanceCount = 0;

    RadiationIMCPostProcessConfig postProcess_;
    std::shared_ptr<SphericalObserver> observer_;

    struct PeelOffSource
    {
        enum class StartKind
        {
            LocalCellPoint,
            PhysicalVacuumBoundary,
            RemoteBoundaryFace,
            Invalid
        };

        StartKind startKind = StartKind::LocalCellPoint;
        size_t sourceCellIndex = 0;
        FaceExitInfo startExit;

        Vector3D sourceLocation;
        double labFrequency = 0.0;
        double labWeight = 0.0;
        double eventTimeLeft = 0.0;
        PeelOffEventKind kind = PeelOffEventKind::SOURCE_EMISSION;

        enum class PhaseMode { Isotropic, ElasticScatter, CosineLeak };
        PhaseMode phaseMode = PhaseMode::Isotropic;
        Vector3D incomingDirectionLab;
        Vector3D surfaceNormalLab;
        double stokesQ = 0.0;
        double stokesU = 0.0;
        Vector3D polarizationBasis;
        bool polarizationInitialized = false;
    };

    struct PeelOffRayState
    {
        unsigned long long rayId = 0;
        PeelOffEventKind kind = PeelOffEventKind::SOURCE_EMISSION;
        size_t observerIndex = 0;
        Vector3D nObsLab;
        Vector3D position;
        double remainingDist = 0.0;
        double tau = 0.0;
        double labFrequency = 0.0;
        double contributionPrefactor = 0.0;
        double stokesQ = 0.0;
        double stokesU = 0.0;
        bool polarizationInitialized = false;
        double eventTimeLeft = -1.0;
        size_t currentLocalCell = std::numeric_limits<size_t>::max();
        int originRank = 0;
        int currentRank = 0;
        unsigned int mpiHops = 0;
        unsigned int cellsTraversed = 0;
        bool crossedAnyMpiBoundary = false;
        bool valid = true;

        static constexpr size_t PackedDoubles = 23;
        void packInto(double* buf) const;
        static PeelOffRayState unpack(const double* buf);
    };

    struct LocalTraceOutcome
    {
        enum class Status
        {
            CompletedAtObserver,
            CompletedAfterPhysicalVacuumExit,
            NeedsRemoteContinuation,
            TauClipped,
            TimeRejected,
            NoExitFace,
            MaxCellsExceeded,
            UnsupportedBoundary,
            InvalidState
        } status = Status::InvalidState;

        PeelOffRayState state;
        FaceExitInfo remoteExit;
    };

    FaceExitInfo classifyFaceExit(size_t faceGlobalIdx, size_t fromCell) const;

    void maybeRecordPeelOff(PeelOffSource const& source);
    void traceOrQueuePeelOffRay(PeelOffRayState ray);
    bool recordPeelOffContribution(PeelOffRayState const& ray, double contribution);

    LocalTraceOutcome continuePeelOffRayLocally(PeelOffRayState state) const;

    void processPendingPeelOffRays();

    double evaluatePeelOffPhasePdf(
        PeelOffSource const& source,
        Vector3D const& nObsLab,
        double& qObserver,
        double& uObserver,
        bool& polarizationInitialized) const;

    double computePeelOffRayOpacity(
        ComputationalCell3D const& cell,
        size_t localCellIndex,
        double labFrequency,
        Vector3D const& nLab,
        double& dopplerShiftOut,
        double& shiftedFreqOut) const;

    void resetPeelOffCounters();

    PeelOffCounters peelOffCounters_;
    std::vector<PeelOffRayState> pendingPeelOffRays_;
    std::vector<PeelOffSource> pendingExternalSourcePeelOff_;
    unsigned long long nextPeelOffRayId_ = 0;

#ifdef RICH_MPI
    // Built once in constructor; valid for entire RadiationIMC lifetime.
    // INVARIANT: mesh topology is frozen during the radiation timestep.
    boost::container::flat_map<size_t, std::pair<int, size_t>> peelOffGhostMap_;
#endif

    bool tryRandomWalkStep(Particle &particle, Functionality &functionality, double dopplerShift);
    void precomputeRandomWalkData();
    bool tryDDMCStep(Particle &particle, Functionality &functionality, double dopplerShift);
    void precomputeDDMCData();

    double computeMinSignedDistanceToAllCellFaces(size_t cellIndex,
                                                  Vector3D const &location) const;
    double computeDDMCGeometryTolerance(size_t cellIndex) const;
    double computeMinDistanceToDDMCLeakFaces(size_t cellIndex,
                                             Vector3D const &location,
                                             DDMCCellData const &data) const;
};

#endif // RADIATION_IMC_HPP
