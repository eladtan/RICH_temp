#ifndef SPHERICAL_OBSERVER_HPP
#define SPHERICAL_OBSERVER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
#include "3D/elementary/Vector3D.hpp"

// Stokes convention: weight = Stokes I (packet energy).
// stokesQ/stokesU are fractional (q = Q/I, u = U/I), so
// absolute Stokes Q = weight * stokesQ.
struct ObserverCrossingRecord
{
    Vector3D crossingPoint;
    Vector3D direction;
    double weight = 0.0;
    double frequency = 0.0;
    size_t sourceCellID = std::numeric_limits<size_t>::max();
#ifdef MONTECARLO_POLARIZATION
    double stokesQ = 0.0;
    double stokesU = 0.0;
    Vector3D polBasis;
    bool polarizationInitialized = false;
#endif
};

struct ObserverPolarizationConfig
{
    Vector3D referenceAxis = Vector3D(0.0, 0.0, 1.0);
    Vector3D fallbackAxis = Vector3D(1.0, 0.0, 0.0);
    double poleTolerance = 0.999999;
    double warnMismatchAngle = 0.01;
    double failMismatchAngle = -1.0;
};

class SphericalObserver
{
public:
    struct Crossing
    {
        bool hit = false;
        double time = 0.0;
        Vector3D point;
    };

    struct Diagnostics
    {
        double sourceDt = 0.0;
        double transportTime = 0.0;
        int mpiRanks = 1;
        int comptonEnabled = 0;
        double emittedEnergy = 0.0;
        double absorbedEnergy = 0.0;
        double boxEscapeEnergy = 0.0;
        double timedOutEnergy = 0.0;
        double cutoffEnergy = 0.0;
        double snapshotTime = 0.0;
        int snapshotCycle = 0;
        int nGenerations = 1;
        int includedFinalGenerations = 1;
        int discardedBurninGenerations = 0;
        int adaptiveOnlyFinalOutput = 0;
        int adaptiveGroupQualityEnabled = 0;
        int adaptiveGroupSourceCellsEnabled = 0;
        int adaptiveGroupFrequencySamplingEnabled = 0;
        int adaptiveGroupHistoryEnabled = 0;
        std::string adaptiveGroupLuminosityNormalization = "mixed";
        double adaptiveGroupTargetNeff = 0.0;
        double adaptiveGroupTargetPolSnr = 0.0;
        double adaptiveGroupDeficitMax = 1.0;
        int adaptiveGroupMinCrossings = 0;
        double adaptiveGroupMinLuminosity = 0.0;
        double adaptiveGroupMinLuminosityFracOfGroupMax = 0.0;
        double adaptiveGroupLatestWeight = 0.0;
        double adaptiveGroupCumulativeWeight = 0.0;
        double adaptiveGroupEmaWeight = 0.0;
        double adaptiveGroupSamplingStrength = 0.0;
        double adaptiveGroupSamplingPdfFloor = 0.0;
        double adaptiveGroupSamplingMaxBias = 1.0;
        double adaptiveGroupSamplingMaxWeightCorrection = 1.0;
        unsigned long long adaptiveGroupSamplingTotalSampled = 0;
        double adaptiveGroupWeightCorrectionMin = 1.0;
        double adaptiveGroupWeightCorrectionMean = 1.0;
        double adaptiveGroupWeightCorrectionMax = 1.0;
        double adaptiveGroupWeightCorrectionCappedFraction = 0.0;
        unsigned long long adaptiveGroupWeightCorrectionFallbackCount = 0;
        unsigned long long adaptiveGroupInvalidPdfFallbackCount = 0;
        unsigned long long adaptiveGroupInvalidPdfFallbackPacketCount = 0;
        double adaptiveGroupCappedEnergyFraction = 0.0;
        int adaptiveGroupEstimatorPotentiallyBiased = 0;
        int adaptiveGroupFallbackToIntegratedPath = 0;
        std::string adaptiveGroupFallbackReason = "none";
        unsigned long long adaptiveGroupSourceLocalStatsAfterPrune = 0;
        unsigned long long adaptiveGroupSourceLocalStatsDropped = 0;
        unsigned long long adaptiveGroupSourceMpiStatsExchanged = 0;
        unsigned long long adaptiveGroupSourceMpiPackedBytes = 0;
    };

    struct PhotosphereData
    {
        double tauThreshold = 2.0 / 3.0;
        double thermalizationTauThreshold = 1.0;

        std::vector<std::vector<double>> mgGroupRadiusTauTotal;
        std::vector<std::vector<double>> mgGroupRadiusThermalization;
        std::vector<std::vector<int>> mgGroupValidTauTotal;
        std::vector<std::vector<int>> mgGroupValidThermalization;

        std::vector<double> mgIntegratedRadiusTauTotal;
        std::vector<double> mgIntegratedRadiusThermalization;
        std::vector<int> mgIntegratedValidTauTotal;
        std::vector<int> mgIntegratedValidThermalization;

        std::vector<double> greyRadiusTauTotal;
        std::vector<double> greyRadiusThermalization;
        std::vector<int> greyValidTauTotal;
        std::vector<int> greyValidThermalization;

        bool hasMG() const
        {
            return !mgGroupRadiusTauTotal.empty() ||
                   !mgIntegratedRadiusTauTotal.empty();
        }

        bool hasGrey() const
        {
            return !greyRadiusTauTotal.empty() ||
                   !greyRadiusThermalization.empty();
        }

        bool hasAny() const { return hasMG() || hasGrey(); }

        void clearMG()
        {
            mgGroupRadiusTauTotal.clear();
            mgGroupRadiusThermalization.clear();
            mgGroupValidTauTotal.clear();
            mgGroupValidThermalization.clear();
            mgIntegratedRadiusTauTotal.clear();
            mgIntegratedRadiusThermalization.clear();
            mgIntegratedValidTauTotal.clear();
            mgIntegratedValidThermalization.clear();
        }
    };

    struct SourceCellEscapeStat
    {
        size_t cellID = std::numeric_limits<size_t>::max();
        size_t observerIndex = std::numeric_limits<size_t>::max();
        double energy = 0.0;
        double weightSq = 0.0;
        double maxWeight = 0.0;
        size_t count = 0;
    };

    struct SourceCellGroupEscapeStat
    {
        size_t cellID = std::numeric_limits<size_t>::max();
        size_t observerIndex = std::numeric_limits<size_t>::max();
        size_t groupIndex = std::numeric_limits<size_t>::max();
        double energy = 0.0;
        double weightSq = 0.0;
        double maxWeight = 0.0;
        size_t count = 0;
    };

    struct SourceObserverGroupCellKey
    {
        size_t observerIndex = 0;
        size_t groupIndex = 0;
        size_t cellID = 0;

        bool operator==(SourceObserverGroupCellKey const& other) const
        {
            return observerIndex == other.observerIndex
                && groupIndex == other.groupIndex
                && cellID == other.cellID;
        }
    };

    struct SourceObserverGroupCellKeyHash
    {
        size_t operator()(SourceObserverGroupCellKey const& key) const
        {
            uint64_t x = static_cast<uint64_t>(key.cellID);
            x ^= static_cast<uint64_t>(key.observerIndex) + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
            x ^= static_cast<uint64_t>(key.groupIndex) + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
            x += 0x9e3779b97f4a7c15ULL;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            x ^= (x >> 31);
            return static_cast<size_t>(x);
        }
    };

    struct ObserverQualitySnapshot
    {
        bool polarizationEnabled = false;
        std::vector<double> energy;
        std::vector<double> energyWeightSq;
        std::vector<unsigned long long> crossingCount;
        std::vector<double> stokesQ;
        std::vector<double> stokesU;
        std::vector<double> polarizationWeightSq;
        std::vector<double> sumWQ2;
        std::vector<double> sumWU2;
    };

    struct ObserverGroupQualitySnapshot
    {
        bool polarizationEnabled = false;
        size_t observerCount = 0;
        size_t groupCount = 0;
        std::vector<std::vector<double>> energy;
        std::vector<std::vector<double>> energyWeightSq;
        std::vector<std::vector<size_t>> crossingCount;
        std::vector<std::vector<double>> stokesQ;
        std::vector<std::vector<double>> stokesU;
        std::vector<std::vector<double>> sumWQ2;
        std::vector<std::vector<double>> sumWU2;
    };

    SphericalObserver(Vector3D center, double radius, size_t numObservers,
                      std::vector<double> groupBoundaries = {});

    Crossing nextOutwardCrossing(Vector3D const& position,
                                 Vector3D const& velocity,
                                 double maxTime) const;

    void recordCrossing(Vector3D const& crossingPoint,
                        double weight, double frequency);
    void recordCrossing(::ObserverCrossingRecord const& record);
#ifdef MONTECARLO_POLARIZATION
    void recordCrossing(Vector3D const& crossingPoint,
                        double weight,
                        double frequency,
                        double qObserver,
                        double uObserver);
    void setPolarizationMetadata(bool enabled,
                                 int manualScatteringsAfterAcceleration,
                                 double depolarizationScatterings,
                                 std::string acceleratedClosure);
    void setPolarizationConfig(ObserverPolarizationConfig const& config);
#endif

    void addEmittedEnergy(double energy);
    void addAbsorbedEnergy(double energy);
    void addBoxEscapeEnergy(double energy);
    void addTimedOutEnergy(double energy);
    void addCutoffEnergy(double energy);

    void resetGenerationSourceCellEscapeStats();
    std::vector<SourceCellEscapeStat> getGenerationSourceCellEscapeStats() const;
    ObserverQualitySnapshot getObserverQualitySnapshot() const;

    void resetGenerationSourceCellGroupEscapeStats();
    std::vector<SourceCellGroupEscapeStat> getGenerationSourceCellGroupEscapeStats() const;
    ObserverGroupQualitySnapshot getObserverGroupQualitySnapshot() const;
    void setGenerationSourceCellGroupStatsEnabled(bool enabled);
    void resetTallies();
    void clearGenerationStatistics();
    void accumulateCurrentTalliesForStatistics(double sourceDt);
    void loadStatisticalMeanTallies();
    size_t getStatisticsSamples() const;
    double getTotalLuminosityStderrGen(double sourceDt) const;
    double getTotalLuminosityRelErrGen(double sourceDt) const;

    void scale(double factor);

    void mpiReduceToRank0();

    void writeHDF5(std::string const& filename,
                   Diagnostics const& diagnostics) const;

    void writeVTK(std::string const& filename, double sourceDt) const;

    void writeTXT(std::string const& filename, double sourceDt) const;

    void setPhotosphereData(PhotosphereData data);
    PhotosphereData const& getPhotosphereData() const;
    bool hasPhotosphereData() const;

    std::vector<double> getLuminosity(double sourceDt) const;
    std::vector<std::vector<double>> getGroupLuminosity(double sourceDt) const;

    double getTotalCrossingEnergy() const;
    double getEmittedEnergy() const;
    double getAbsorbedEnergy() const;
    double getBoxEscapeEnergy() const;
    double getTimedOutEnergy() const;
    double getCutoffEnergy() const;

    std::vector<Vector3D> const& getDirections() const;
    Vector3D getCenter() const;
    double getRadius() const;
    size_t getNumObservers() const;
    size_t getNumGroups() const;
    double getSolidAngle() const;
    double getPatchArea() const;
    std::vector<double> const& getObserverSolidAngles() const;
    std::vector<double> const& getMaxPacketEnergy() const;
    std::vector<double> const& getObserverEnergy() const { return observerEnergy_; }
    std::vector<size_t> const& getObserverCrossingCount() const { return observerCrossingCount_; }
#ifdef MONTECARLO_POLARIZATION
    std::vector<double> const& getObserverStokesQ() const { return observerStokesQ_; }
    std::vector<double> const& getObserverStokesU() const { return observerStokesU_; }
    std::vector<std::vector<double>> const& getGroupStokesQ() const { return groupStokesQ_; }
    std::vector<std::vector<double>> const& getGroupStokesU() const { return groupStokesU_; }
    std::vector<double> const& getMismatchWeightedSum() const { return mismatchWeightedSum_; }
    std::vector<double> const& getMismatchWeighted2Sum() const { return mismatchWeighted2Sum_; }
    std::vector<double> const& getMismatchMax() const { return mismatchMax_; }
#endif

private:
    Vector3D center_;
    double radius_ = 0.0;
    double radiusSquared_ = 0.0;
    size_t numObservers_ = 0;
    size_t numGroups_ = 1;
    std::vector<double> groupBoundaries_;
    std::vector<Vector3D> directions_;
    std::vector<double> observerEnergy_;
    std::vector<double> observerEnergyWeightSq_;
    std::vector<double> observerMaxPacketEnergy_;
    std::vector<size_t> observerCrossingCount_;
    std::unordered_map<size_t, std::unordered_map<size_t, SourceCellEscapeStat>> generationSourceCellEscape_;
    std::unordered_map<SourceObserverGroupCellKey, SourceCellGroupEscapeStat, SourceObserverGroupCellKeyHash> generationSourceCellGroupEscape_;
    bool generationSourceCellGroupStatsEnabled_ = false;
    std::vector<double> observerSolidAngle_;
    std::vector<std::vector<double>> groupEnergy_;
    std::vector<std::vector<double>> groupEnergyWeightSq_;
    std::vector<std::vector<size_t>> groupCrossingCount_;
    PhotosphereData photosphereData_;
#ifdef MONTECARLO_POLARIZATION
    bool polarizationOutputEnabled_ = false;
    int polarizationManualScatteringsAfterAcceleration_ = 4;
    double polarizationDepolarizationScatterings_ = 2.0;
    std::string polarizationAcceleratedClosure_ = "damped_last_scatterings";
    ObserverPolarizationConfig polConfig_;
    std::vector<Vector3D> skyE1_;
    std::vector<double> observerStokesQ_;
    std::vector<double> observerStokesU_;
    std::vector<double> observerSumWeightSq_;
    std::vector<double> observerSumWQ2_;
    std::vector<double> observerSumWU2_;
    std::vector<double> mismatchWeightedSum_;
    std::vector<double> mismatchWeighted2Sum_;
    std::vector<double> mismatchMax_;
    size_t mismatchWarningCount_ = 0;
    size_t uninitializedPolarizationCount_ = 0;
    std::vector<std::vector<double>> groupStokesQ_;
    std::vector<std::vector<double>> groupStokesU_;
    std::vector<std::vector<double>> groupSumWQ2_;
    std::vector<std::vector<double>> groupSumWU2_;

    void buildSkyBases();
    void rotateAndAccumulate(::ObserverCrossingRecord const& rec, size_t obs);
    void accumulateMismatch(::ObserverCrossingRecord const& rec, size_t obs,
                            Vector3D const& rhat);
#endif

    double emittedEnergy_ = 0.0;
    double absorbedEnergy_ = 0.0;
    double boxEscapeEnergy_ = 0.0;
    double timedOutEnergy_ = 0.0;
    double cutoffEnergy_ = 0.0;

public:
    struct RunningScalarStats
    {
        double sum = 0.0;
        double sumSq = 0.0;
    };

    struct RunningVectorStats
    {
        std::vector<double> sum;
        std::vector<double> sumSq;
    };

    struct RunningMatrixStats
    {
        std::vector<std::vector<double>> sum;
        std::vector<std::vector<double>> sumSq;
    };

private:
    struct GenerationStatistics
    {
        size_t samples = 0;
        RunningVectorStats energy;
        RunningVectorStats luminosity;
        RunningVectorStats isoLuminosity;
        RunningVectorStats flux;
        RunningMatrixStats groupEnergy;
        RunningMatrixStats groupLuminosity;
#ifdef MONTECARLO_POLARIZATION
        RunningVectorStats stokesQ;
        RunningVectorStats stokesU;
        RunningVectorStats q;
        RunningVectorStats u;
        RunningVectorStats stokesQLuminosity;
        RunningVectorStats stokesULuminosity;
        RunningVectorStats polarizationDegree;
        RunningVectorStats polarizationAngle;
        RunningMatrixStats groupStokesQ;
        RunningMatrixStats groupStokesU;
        RunningMatrixStats groupQ;
        RunningMatrixStats groupU;
        RunningMatrixStats groupQLuminosity;
        RunningMatrixStats groupULuminosity;
        RunningMatrixStats groupPolarizationDegree;
        RunningMatrixStats groupPolarizationAngle;
#endif
        RunningScalarStats totalEnergy;
        RunningScalarStats totalLuminosity;
        RunningScalarStats emittedEnergy;
        RunningScalarStats absorbedEnergy;
        RunningScalarStats boxEscapeEnergy;
        RunningScalarStats timedOutEnergy;
        RunningScalarStats cutoffEnergy;
        RunningScalarStats transportSinkResidual;
        RunningScalarStats timedOutFraction;
        std::vector<double> energyWeightSqSum;
        std::vector<std::vector<double>> groupEnergyWeightSqSum;
        double totalEnergyWeightSqSum = 0.0;
        std::vector<size_t> observerCrossingCountSum;
        std::vector<std::vector<size_t>> groupCrossingCountSum;
        std::vector<double> observerMaxPacketEnergyMax;
#ifdef MONTECARLO_POLARIZATION
        std::vector<double> observerSumWeightSqSum;
        std::vector<double> observerSumWQ2Sum;
        std::vector<double> observerSumWU2Sum;
        std::vector<std::vector<double>> groupSumWQ2Sum;
        std::vector<std::vector<double>> groupSumWU2Sum;
        std::vector<double> mismatchWeightedSumSum;
        std::vector<double> mismatchWeighted2SumSum;
        std::vector<double> mismatchMaxMax;
#endif
    };

    GenerationStatistics generationStats_;

    size_t findNearestObserver(Vector3D const& crossingPoint) const;
    size_t findNearestObserverDirection(Vector3D const& direction) const;
    size_t findGroup(double frequency) const;
    void recordGenerationSourceCellEscape(size_t observerIndex, size_t cellID, double energy);
    void recordGenerationSourceCellGroupEscape(size_t observerIndex, size_t groupIndex, size_t cellID, double weight);
};

#endif // SPHERICAL_OBSERVER_HPP
