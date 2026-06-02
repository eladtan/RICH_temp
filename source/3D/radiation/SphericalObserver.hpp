#ifndef SPHERICAL_OBSERVER_HPP
#define SPHERICAL_OBSERVER_HPP

#include <array>
#include <cstddef>
#include <string>
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
    std::vector<double> const& getPeelOffEnergy() const { return peelOffEnergy_; }
    std::vector<size_t> const& getPeelOffCount() const { return peelOffCount_; }
    std::vector<std::vector<double>> const& getPeelOffGroupEnergy() const { return peelOffGroupEnergy_; }

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
    std::vector<double> observerMaxPacketEnergy_;
    std::vector<size_t> observerCrossingCount_;
    std::vector<double> observerSolidAngle_;
    std::vector<std::vector<double>> groupEnergy_;
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
    std::vector<size_t> peelOffCount_;
    std::vector<std::vector<double>> peelOffGroupEnergy_;

    std::array<std::vector<double>, NumPeelOffKinds> peelOffEnergyByKind_;
    std::array<std::vector<size_t>, NumPeelOffKinds> peelOffCountByKind_;
    std::array<std::vector<std::vector<double>>, NumPeelOffKinds> peelOffGroupEnergyByKind_;

    double emittedEnergy_ = 0.0;
    double absorbedEnergy_ = 0.0;
    double boxEscapeEnergy_ = 0.0;
    double timedOutEnergy_ = 0.0;
    double cutoffEnergy_ = 0.0;

    size_t findNearestObserver(Vector3D const& crossingPoint) const;
    size_t findGroup(double frequency) const;
};

#endif // SPHERICAL_OBSERVER_HPP
