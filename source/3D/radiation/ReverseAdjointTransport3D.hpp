#ifndef REVERSE_ADJOINT_TRANSPORT_3D_HPP
#define REVERSE_ADJOINT_TRANSPORT_3D_HPP

#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "3D/elementary/Vector3D.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "Radiation/OpacityCalculator.hpp"
#include "MultigroupOpacity.hpp"
#include "ReverseEstimatorConfig.hpp"
#include "ReversePacket.hpp"
#include "ReverseObserverTallies.hpp"
#include "ReverseDDMC.hpp"
#include "ReverseDoppler.hpp"
#include "ReversePolarizationMueller.hpp"

class SphericalObserver;
class HDF5Writer;

class ReverseAdjointTransport3D
{
public:
    ReverseAdjointTransport3D(
        Tessellation3D &tess,
        std::vector<ComputationalCell3D> const &cells,
        std::shared_ptr<OpacityCalculator> opacity,
        std::shared_ptr<SphericalObserver> observer,
        ReverseEstimatorConfig const &config,
        double sourceDt,
        double transportTime,
        std::vector<double> const &fleckFactors,
        bool forwardUsesVelocity,
        bool forwardUsesDDMC,
        bool forwardUsesPolarization,
        bool forwardUsesCompton = false);

    void run();
    void setFleckFromForwardVector(bool v) { fleckFromForwardVector_ = v; }

    void writeOutputs(std::string const &filename) const;

    void writeComparisonOutputs(std::string const &filename,
                                std::string const &prefix,
                                std::vector<double> const &fwdLum,
                                std::vector<double> const &revLum,
                                std::vector<double> const &delta,
                                std::vector<double> const &relDelta,
                                std::vector<double> const &fwdQ,
                                std::vector<double> const &fwdU,
                                std::vector<std::vector<double>> const &fwdGroupLum,
                                std::vector<std::vector<double>> const &fwdGroupQ,
                                std::vector<std::vector<double>> const &fwdGroupU) const;

    ReverseObserverTallies const &tallies() const { return tallies_; }
    std::vector<uint64_t> const &localCellStepCounts() const { return localCellStepCounts_; }
    std::vector<uint64_t> const &localCellPacketEntries() const { return localCellPacketEntries_; }

private:
    Tessellation3D &tess_;
    std::vector<ComputationalCell3D> const &cells_;
    std::shared_ptr<OpacityCalculator> opacity_;
    std::shared_ptr<MultigroupOpacity> multigroupOpacity_;
    std::shared_ptr<SphericalObserver> observer_;
    ReverseEstimatorConfig config_;
    double sourceDt_;
    double transportTime_;
    std::vector<double> fleckFactors_;
    bool fleckDefaultedToOne_ = false;
    bool fleckFromForwardVector_ = false;
    mutable bool comparisonWritten_ = false;
    mutable std::string comparisonFile_;
    mutable std::string comparisonGroup_;

    bool useVelocity_;
    bool useDDMC_;
    bool usePolarization_;

    ReverseObserverTallies tallies_;
    ReverseDDMC ddmc_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uniform_;

    size_t numObservers_;
    size_t numGroups_;
    std::vector<Vector3D> observerDirections_;
    std::vector<Vector3D> observerSkyE1_;
    std::vector<double> observerSolidAngles_;
    double observerRadius_;
    Vector3D observerCenter_;
    double patchArea_;
    std::vector<double> groupBoundaries_;
    std::vector<uint64_t> localCellStepCounts_;
    std::vector<uint64_t> localCellPacketEntries_;

    void initializeObserverData();
    void precomputeDDMCData();
    void resetLocalMeasurements();
    void recordCellStep(size_t cellIndex);
    void recordCellPacketEntry(size_t cellIndex);

    ReverseAdjointPacket launchPacket(size_t obsIndex, size_t groupIndex);

    void transportPacket(ReverseAdjointPacket &pkt);
    void ordinaryStep(ReverseAdjointPacket &pkt);
    void handleReverseThomsonScatter(ReverseAdjointPacket &pkt,
                                     ReverseDoppler::FrameState const &fs,
                                     Vector3D const &cellVelocity);
    void handleReverseReset(ReverseAdjointPacket &pkt);
    void crossFace(ReverseAdjointPacket &pkt, size_t newCell);
    void terminatePacket(ReverseAdjointPacket &pkt, std::string const &reason);

    double getSourceLuminosity(size_t cellIndex,
                               size_t groupIndex = std::numeric_limits<size_t>::max()) const;
    Vector3D getCellVelocity(size_t cellIndex) const;
    double getAbsorptionOpacity(size_t cellIndex, double frequency) const;
    double getScatteringOpacity(size_t cellIndex, double frequency) const;
    double getPatchAreaOverN(size_t obsIndex, size_t groupIndex) const;

    size_t findCellForPoint(Vector3D const &point) const;
    double distanceToFace(Vector3D const &pos, Vector3D const &dir,
                          size_t cellIndex, size_t &exitFace,
                          size_t &nextCell) const;
    size_t findGroup(double frequency) const;
};

#endif // REVERSE_ADJOINT_TRANSPORT_3D_HPP
