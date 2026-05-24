#ifndef SPHERICAL_OBSERVER_HPP
#define SPHERICAL_OBSERVER_HPP

#include <cstddef>
#include <string>
#include <vector>
#include "3D/elementary/Vector3D.hpp"

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

    void addEmittedEnergy(double energy);
    void addAbsorbedEnergy(double energy);
    void addBoxEscapeEnergy(double energy);
    void addTimedOutEnergy(double energy);
    void addCutoffEnergy(double energy);

    void scale(double factor);

    void mpiReduceToRank0();

    void writeHDF5(std::string const& filename,
                   Diagnostics const& diagnostics) const;

    void writeVTK(std::string const& filename, double sourceDt) const;

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

private:
    Vector3D center_;
    double radius_ = 0.0;
    double radiusSquared_ = 0.0;
    size_t numObservers_ = 0;
    size_t numGroups_ = 1;
    std::vector<double> groupBoundaries_;
    std::vector<Vector3D> directions_;
    std::vector<double> observerEnergy_;
    std::vector<size_t> observerCrossingCount_;
    std::vector<double> observerSolidAngle_;
    std::vector<std::vector<double>> groupEnergy_;

    double emittedEnergy_ = 0.0;
    double absorbedEnergy_ = 0.0;
    double boxEscapeEnergy_ = 0.0;
    double timedOutEnergy_ = 0.0;
    double cutoffEnergy_ = 0.0;

    size_t findNearestObserver(Vector3D const& crossingPoint) const;
    size_t findGroup(double frequency) const;
};

#endif // SPHERICAL_OBSERVER_HPP
