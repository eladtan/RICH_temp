#ifndef SPHERICAL_OBSERVER_HPP
#define SPHERICAL_OBSERVER_HPP

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>
#include "3D/elementary/Vector3D.hpp"
#include "PeelOffTypes.hpp"

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
    };

    struct SourceCellEscapeStat
    {
        size_t cellID = std::numeric_limits<size_t>::max();
        size_t observerIndex = std::numeric_limits<size_t>::max();
        double energy = 0.0;
        size_t count = 0;
    };

    SphericalObserver(Vector3D center, double radius, size_t numObservers,
                      std::vector<double> groupBoundaries = {});

    Crossing nextOutwardCrossing(Vector3D const& position,
                                 Vector3D const& velocity,
                                 double maxTime) const;

    void recordCrossing(Vector3D const& crossingPoint,
                        double weight, double frequency);
    void recordCrossing(ObserverCrossingRecord const& record);
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
    void resetTallies();
    void clearGenerationStatistics();
    void accumulateCurrentTalliesForStatistics(double sourceDt);
    void loadStatisticalMeanTallies();
    size_t getStatisticsSamples() const;
    double getTotalLuminosityStderrGen(double sourceDt) const;
    double getTotalLuminosityRelErrGen(double sourceDt) const;

    struct PeelOffConfigSnapshot
    {
        bool sourceEmission = true;
        bool resolvedElastic = false;
        bool resolvedEffective = false;
        bool rwClosure = false;
        bool rwUpscatter = false;
        bool ddmcLeak = false;
        bool ddmcUpscatter = false;
        double maxTau = 700.0;
        double rayNudgeFraction = 1e-10;
        size_t maxRayCells = 100000;
        size_t maxDistributedExchangeRounds = 64;
        std::string mpiRayPolicy = "DistributedExact";
        int mpiRayPolicyId = 2;  // 0=StrictAbort, 1=LocalConservativeVacuum, 2=DistributedExact
        bool allowApproximateMpiPeelOff = false;
        bool writePerKindTallies = true;
    };

    void setPeelOffMetadata(bool enabled, bool writePerKindTallies = true);
    void setPeelOffConfig(PeelOffConfigSnapshot const& snap);
    void setPeelOffCounters(PeelOffCounters const& counters);
    bool recordPeelOff(size_t observerIndex, double energy, double frequency);
    bool recordPeelOff(size_t observerIndex, double energy, double frequency,
                       PeelOffEventKind kind);
#ifdef MONTECARLO_POLARIZATION
    bool recordPeelOff(size_t observerIndex, double energy, double frequency,
                       double qObserver, double uObserver, PeelOffEventKind kind);
#endif
    std::vector<double> const& getPeelOffEnergy() const { return peelOffEnergy_; }
    std::vector<size_t> const& getPeelOffCount() const { return peelOffCount_; }
    std::vector<std::vector<double>> const& getPeelOffGroupEnergy() const { return peelOffGroupEnergy_; }
#ifdef MONTECARLO_POLARIZATION
    std::vector<double> const& getPeelOffStokesQ() const { return peelOffStokesQ_; }
    std::vector<double> const& getPeelOffStokesU() const { return peelOffStokesU_; }
#endif

    void scale(double factor);

    void mpiReduceToRank0();

    void writeHDF5(std::string const& filename,
                   Diagnostics const& diagnostics) const;

    void writeVTK(std::string const& filename, double sourceDt) const;

    void writeTXT(std::string const& filename, double sourceDt) const;

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
    std::vector<double> observerSolidAngle_;
    std::vector<std::vector<double>> groupEnergy_;
    std::vector<std::vector<double>> groupEnergyWeightSq_;
    std::vector<std::vector<size_t>> groupCrossingCount_;
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

    void buildSkyBases();
    void rotateAndAccumulate(ObserverCrossingRecord const& rec, size_t obs);
    void accumulateMismatch(ObserverCrossingRecord const& rec, size_t obs,
                            Vector3D const& rhat);
#endif

    bool peelOffOutputEnabled_ = false;
    bool peelOffPerKindEnabled_ = false;
    bool peelOffNeedsMpiReduction_ = false;
    PeelOffCounters peelOffCounters_;
    PeelOffConfigSnapshot peelOffConfigSnap_;
    std::vector<double> peelOffEnergy_;
    std::vector<double> peelOffEnergyWeightSq_;
    std::vector<size_t> peelOffCount_;
    std::vector<std::vector<double>> peelOffGroupEnergy_;
    std::vector<std::vector<double>> peelOffGroupEnergyWeightSq_;
#ifdef MONTECARLO_POLARIZATION
    std::vector<double> peelOffStokesQ_;
    std::vector<double> peelOffStokesU_;
#endif

    std::array<std::vector<double>, NumPeelOffKinds> peelOffEnergyByKind_;
    std::array<std::vector<double>, NumPeelOffKinds> peelOffEnergyByKindWeightSq_;
    std::array<std::vector<size_t>, NumPeelOffKinds> peelOffCountByKind_;
    std::array<std::vector<std::vector<double>>, NumPeelOffKinds> peelOffGroupEnergyByKind_;
    std::array<std::vector<std::vector<double>>, NumPeelOffKinds> peelOffGroupEnergyByKindWeightSq_;
#ifdef MONTECARLO_POLARIZATION
    std::array<std::vector<double>, NumPeelOffKinds> peelOffStokesQByKind_;
    std::array<std::vector<double>, NumPeelOffKinds> peelOffStokesUByKind_;
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
        RunningVectorStats peelOffEnergy;
        RunningVectorStats peelOffLuminosity;
        RunningVectorStats peelOffIsoLuminosity;
#ifdef MONTECARLO_POLARIZATION
        RunningVectorStats peelOffStokesQ;
        RunningVectorStats peelOffStokesU;
        RunningVectorStats peelOffQ;
        RunningVectorStats peelOffU;
        RunningVectorStats peelOffQLuminosity;
        RunningVectorStats peelOffULuminosity;
        RunningVectorStats peelOffPolarizationDegree;
        RunningVectorStats peelOffPolarizationAngle;
#endif
        RunningMatrixStats peelOffGroupEnergy;
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
        std::vector<double> peelOffEnergyWeightSqSum;
        std::vector<std::vector<double>> peelOffGroupEnergyWeightSqSum;
        double totalEnergyWeightSqSum = 0.0;
        std::vector<size_t> observerCrossingCountSum;
        std::vector<std::vector<size_t>> groupCrossingCountSum;
        std::vector<size_t> peelOffCountSum;
        std::array<std::vector<double>, NumPeelOffKinds> peelOffEnergyByKindSum;
        std::array<std::vector<std::vector<double>>, NumPeelOffKinds> peelOffGroupEnergyByKindSum;
        std::array<std::vector<size_t>, NumPeelOffKinds> peelOffCountByKindSum;
        std::vector<double> observerMaxPacketEnergyMax;
#ifdef MONTECARLO_POLARIZATION
        std::vector<double> observerSumWeightSqSum;
        std::vector<double> observerSumWQ2Sum;
        std::vector<double> observerSumWU2Sum;
        std::vector<double> mismatchWeightedSumSum;
        std::vector<double> mismatchWeighted2SumSum;
        std::vector<double> mismatchMaxMax;
        std::array<std::vector<double>, NumPeelOffKinds> peelOffStokesQByKindSum;
        std::array<std::vector<double>, NumPeelOffKinds> peelOffStokesUByKindSum;
#endif
    };

    GenerationStatistics generationStats_;

    size_t findNearestObserver(Vector3D const& crossingPoint) const;
    size_t findGroup(double frequency) const;
    void recordGenerationSourceCellEscape(size_t observerIndex, size_t cellID, double energy);
};

#endif // SPHERICAL_OBSERVER_HPP
