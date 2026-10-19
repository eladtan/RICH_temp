#ifndef SPHERICAL_OBSERVER_HPP
#define SPHERICAL_OBSERVER_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "3D/elementary/Vector3D.hpp"

class HDF5Writer;
class HDF5Reader;

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
    // Emission weight of the packet (0 when unknown) and its transport steps,
    // for the heavy-packet diagnostics.
    double initialWeight = 0.0;
    size_t steps = 0;
    // Weight-window split factor k of the packet (1 when never split). The
    // adaptive source statistics record k weight^2, the weight-squared of the
    // unsplit packet; the observer tallies and errors use the weight itself.
    double splitFactor = 1.0;
    // The packet reached its band through the DDMC compression cutoff.
    bool compressionConverted = false;
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
        // Signed pdV work of the comoving-frame shift on DDMC packets.
        double compressionWork = 0.0;
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

        // The flux-source emission surface along each observer direction,
        // from the nearest construction ray: the radius the cells along that
        // line are tested against (after smoothing and the neighbour
        // mapping), and the grey and per-group ([observer][group], in
        // mg-innermost mode) radii where tau_eff reaches emissionSurfaceTau.
        // -1 where the ray never reached it. Empty without a flux source.
        double emissionSurfaceTau = 0.0;
        std::vector<double> emissionSurfaceRadius;
        std::vector<double> emissionSurfaceGreyRadius;
        std::vector<std::vector<double>> emissionSurfaceGroupRadius;

        bool hasEmissionSurface() const { return !emissionSurfaceRadius.empty(); }

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
        // Split-aware second moments (PolarizationStatistics.hpp): sum k w^2
        // (polarizationWeightSq), sum k w^2 q and q^2, sum k w^2 u and u^2.
        std::vector<double> polarizationWeightSq;
        std::vector<double> sumKW2Q;
        std::vector<double> sumKW2QQ;
        std::vector<double> sumKW2U;
        std::vector<double> sumKW2UU;
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
        // energyWeightSq is sum k w^2; the rest as in ObserverQualitySnapshot.
        std::vector<std::vector<double>> sumKW2Q;
        std::vector<std::vector<double>> sumKW2QQ;
        std::vector<std::vector<double>> sumKW2U;
        std::vector<std::vector<double>> sumKW2UU;
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
    // Signed work of the DDMC comoving-frame compression (+) and expansion
    // (-) on packet weights; part of the energy budget.
    void addCompressionWork(double energy);

    void resetGenerationSourceCellEscapeStats();
    std::vector<SourceCellEscapeStat> getGenerationSourceCellEscapeStats() const;
    ObserverQualitySnapshot getObserverQualitySnapshot() const;

    void resetGenerationSourceCellGroupEscapeStats();
    std::vector<SourceCellGroupEscapeStat> getGenerationSourceCellGroupEscapeStats() const;
    ObserverGroupQualitySnapshot getObserverGroupQualitySnapshot() const;
    void setGenerationSourceCellGroupStatsEnabled(bool enabled);
    void resetTallies();
    // Heavy-packet report of the current generation's tallies: how much of
    // the crossing energy came from packets whose weight grew since emission,
    // and the observer/group bins most dominated by a single packet, with that
    // packet's emission weight, source cell and steps. Collective (call before
    // mpiReduceToRank0); rank 0 prints.
    void printHeavyPacketReport(std::string const& label, size_t generation,
                                size_t topBins) const;
    // Escape-origin diagnostic: this generation's crossing energy split by the
    // class of the packet's source cell and by group. Classes: 0 a cell that
    // owns emitting flux-source faces (its face and volume packets share the
    // ID), 1 another volume cell below 1e5 K, 2 [1e5, 1e6) K, 3 [1e6, 1e7) K,
    // 4 >= 1e7 K, 5 unknown source. hotCells is sorted by cell ID and lists
    // the cells of classes 2-4. Both must outlive the observer's use; null
    // pointers disable the tally.
    static constexpr size_t kEscapeOriginClasses = 6;
    void setEscapeOriginClassification(
        std::unordered_set<size_t> const* faceCellIDs,
        std::vector<std::pair<size_t, unsigned char>> const* hotCells);
    // Collective (call before mpiReduceToRank0); rank 0 prints ESCAPE_ORIGIN.
    void printEscapeOriginReport(std::string const& label, size_t generation) const;
    void clearGenerationStatistics();
    void accumulateCurrentTalliesForStatistics(double sourceDt);
    void loadStatisticalMeanTallies();
    size_t getStatisticsSamples() const;
    // Exact copy of the running generation statistics (sample count, sums,
    // sums of squares, weight-squared sums, crossing counts and maxima) under
    // `prefix`, so a later run can continue adding generations to them.
    void writeStatisticsState(HDF5Writer& writer, std::string const& prefix) const;
    // Replaces the running statistics with the ones stored under `prefix`.
    // Throws if the stored shapes do not match this observer.
    void readStatisticsState(HDF5Reader const& reader, std::string const& prefix);
    double getTotalLuminosityStderrGen(double sourceDt) const;
    // Generation error of the reported (luminosity-weighted) polarization
    // q = sum_g Q_g / sum_g I_g of one bin over the accumulated final
    // generations, by the delta method:
    //   var(q) = n/(n-1) sum_g (Q_g - q I_g)^2 / (sum_g I_g)^2  (u alike),
    // with the effective number of generations (sum I_g)^2 / sum I_g^2. A bin
    // with fewer than kMinGenerationNeff effective generations is dominated by
    // a few generations (typically single heavy packets), so its error is not
    // estimable and `measured` is false. Uses the statistics on the rank that
    // accumulates them (rank 0).
    struct GenerationPolarizationError
    {
        double sigmaP = 0.0;
        double snr = 0.0;
        double generationNeff = 0.0;
        bool measured = false;
    };
    static constexpr double kMinGenerationNeff = 5.0;
    // group = numGroups for the observer total.
    GenerationPolarizationError getGenerationPolarizationError(size_t observer, size_t group) const;
    // Rank 0 prints "<label> generation polarization errors: ..." with the
    // number of measured observers and the percentiles of sigma_P and
    // P / sigma_P over them.
    void printGenerationPolarizationSummary(std::string const& label) const;
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
    double getCompressionWork() const;

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
    // Heaviest local crossing of each observer/group bin this generation.
    struct HeavyCrossing
    {
        double weight = 0.0;
        double initialWeight = 0.0;
        double frequency = 0.0;
        size_t sourceCellID = std::numeric_limits<size_t>::max();
        size_t steps = 0;
        bool compressionConverted = false;
    };
    std::vector<HeavyCrossing> heavyCrossingByBin_;
    // Crossings binned by weight growth w/w0: unknown w0, <=1.01, (1.01,10],
    // (10,1e3], >1e3.
    static constexpr size_t kGrowthBins = 5;
    std::array<double, kGrowthBins> growthBinEnergy_{};
    std::array<double, kGrowthBins> growthBinCount_{};
    double maxWeightGrowth_ = 0.0;
    void recordHeavyCrossing(size_t obs, size_t group, ::ObserverCrossingRecord const& rec);
    std::unordered_set<size_t> const* escapeOriginFaceCells_ = nullptr;
    std::vector<std::pair<size_t, unsigned char>> const* escapeOriginHotCells_ = nullptr;
    std::vector<double> escapeOriginEnergy_;  // class * numGroups_ + group
    // Crossing energy per group of packets converted by the DDMC compression
    // cutoff (any class).
    std::vector<double> escapeOriginConvertedEnergy_;
    size_t escapeOriginClass(size_t sourceCellID) const;
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
    // Split-aware second moments of the polarized crossings: sum k w^2,
    // sum k w^2 q, sum k w^2 q^2 and the same for u (PolarizationStatistics.hpp).
    std::vector<double> observerSumWeightSq_;
    std::vector<double> observerSumKW2Q_;
    std::vector<double> observerSumKW2QQ_;
    std::vector<double> observerSumKW2U_;
    std::vector<double> observerSumKW2UU_;
    std::vector<double> mismatchWeightedSum_;
    std::vector<double> mismatchWeighted2Sum_;
    std::vector<double> mismatchMax_;
    size_t mismatchWarningCount_ = 0;
    size_t uninitializedPolarizationCount_ = 0;
    std::vector<std::vector<double>> groupStokesQ_;
    std::vector<std::vector<double>> groupStokesU_;
    std::vector<std::vector<double>> groupSumKW2Q_;
    std::vector<std::vector<double>> groupSumKW2QQ_;
    std::vector<std::vector<double>> groupSumKW2U_;
    std::vector<std::vector<double>> groupSumKW2UU_;

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
    double compressionWork_ = 0.0;

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
        RunningScalarStats compressionWork;
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
        std::vector<double> observerSumKW2QSum;
        std::vector<double> observerSumKW2QQSum;
        std::vector<double> observerSumKW2USum;
        std::vector<double> observerSumKW2UUSum;
        std::vector<std::vector<double>> groupSumKW2QSum;
        std::vector<std::vector<double>> groupSumKW2QQSum;
        std::vector<std::vector<double>> groupSumKW2USum;
        std::vector<std::vector<double>> groupSumKW2UUSum;
        // Generation cross sums sum_g I_g Q_g and sum_g I_g U_g (crossing
        // energy and Stokes energy of each final generation), which with the
        // sums of squares in energy/stokesQ/stokesU give the generation error
        // of the luminosity-weighted q = sum Q / sum I (generationRatioSigma).
        std::vector<double> energyStokesQCrossSum;
        std::vector<double> energyStokesUCrossSum;
        std::vector<std::vector<double>> groupEnergyStokesQCrossSum;
        std::vector<std::vector<double>> groupEnergyStokesUCrossSum;
        std::vector<double> mismatchWeightedSumSum;
        std::vector<double> mismatchWeighted2SumSum;
        std::vector<double> mismatchMaxMax;
#endif
    };

    GenerationStatistics generationStats_;

    size_t findNearestObserver(Vector3D const& crossingPoint) const;
    size_t findNearestObserverDirection(Vector3D const& direction) const;
    size_t findGroup(double frequency) const;
    void recordGenerationSourceCellEscape(size_t observerIndex, size_t cellID, double energy, double splitFactor);
    void recordGenerationSourceCellGroupEscape(size_t observerIndex, size_t groupIndex, size_t cellID, double weight,
                                               double splitFactor);
};

#endif // SPHERICAL_OBSERVER_HPP
