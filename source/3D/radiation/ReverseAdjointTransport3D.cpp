#include "ReverseAdjointTransport3D.hpp"
#include "SphericalObserver.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "CMMC/src/units/units.hpp"
#include "CMMC/src/planck_integral/planck_integral.hpp"
#include "MultigroupOpacity.hpp"
#include "IMCPolarization.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace
{

double sigmaMeanFromPacketMoments(double sum, double sum2, uint64_t n)
{
    if (n <= 1)
        return 0.0;
    const double nd = static_cast<double>(n);
    const double mean = sum / nd;
    const double variance = std::max(0.0, sum2 / nd - mean * mean);
    return std::sqrt(variance / static_cast<double>(n - 1));
}

std::vector<double> addSameSizeVectors(
    std::vector<double> const &a,
    std::vector<double> const &b)
{
    std::vector<double> out(a.size(), 0.0);
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = a[i] + ((i < b.size()) ? b[i] : 0.0);
    return out;
}

} // anonymous namespace

namespace
{

std::string formatSeconds(double seconds)
{
    if (!std::isfinite(seconds) || seconds < 0.0)
        return "unknown";
    std::ostringstream os;
    if (seconds < 60.0)
        os << std::fixed << std::setprecision(1) << seconds << "s";
    else if (seconds < 3600.0)
        os << std::fixed << std::setprecision(1) << (seconds / 60.0) << "m";
    else
        os << std::fixed << std::setprecision(1) << (seconds / 3600.0) << "h";
    return os.str();
}

} // anonymous namespace

namespace
{

double safeExp(double x)
{
    if (x < -745.0)
        return 0.0;
    if (x > 709.0)
        return std::numeric_limits<double>::max();
    return std::exp(x);
}

} // anonymous namespace

ReverseAdjointTransport3D::ReverseAdjointTransport3D(
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
    bool forwardUsesCompton)
    : tess_(tess),
      cells_(cells),
      opacity_(std::move(opacity)),
      observer_(std::move(observer)),
      config_(config),
      sourceDt_(sourceDt),
      transportTime_(transportTime),
      fleckFactors_(fleckFactors),
      useVelocity_(config.resolveVelocity(forwardUsesVelocity)),
      useDDMC_(config.resolveDDMC(forwardUsesDDMC)),
      usePolarization_(config.resolvePolarization(forwardUsesPolarization)),
      tallies_(observer_->getNumObservers(),
               std::max<size_t>(observer_->getNumGroups(), 1),
               config.debugScatterOrders),
      rng_(config.seed),
      uniform_(0.0, 1.0),
      numObservers_(observer_->getNumObservers()),
      numGroups_(std::max<size_t>(observer_->getNumGroups(), 1)),
      observerRadius_(observer_->getRadius()),
      observerCenter_(observer_->getCenter()),
      patchArea_(observer_->getPatchArea())
{
    if (forwardUsesCompton)
        throw std::runtime_error(
            "Reverse polarization estimator does not support Compton scattering. "
            "Disable Compton or use forward-only mode.");

    // If no Fleck factors are provided, default to f=1 (pure absorption, no re-emission).
    // For physical consistency with the forward IMC, the caller should provide
    // pre-computed Fleck factors matching the forward run.
    if (fleckFactors_.empty() && !cells_.empty())
    {
        if (!config_.allowDefaultFleckOne)
            throw std::runtime_error(
                "Reverse estimator requires per-cell Fleck factors. "
                "Pass the forward Fleck vector or set allowDefaultFleckOne=true.");
        fleckFactors_.assign(cells_.size(), 1.0);
        fleckDefaultedToOne_ = true;
    }

#if ENERGY_GROUPS_NUM > 1
    multigroupOpacity_ = std::make_shared<MultigroupOpacity>(opacity_);
#endif

    initializeObserverData();
    resetLocalMeasurements();

    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    rng_.seed(config_.seed * 1000003ULL + static_cast<uint64_t>(rank));

    if (useDDMC_)
    {
        ddmc_.configure(config_, usePolarization_);
        precomputeDDMCData();
    }
}

void ReverseAdjointTransport3D::resetLocalMeasurements()
{
    localCellStepCounts_.assign(tess_.GetPointNo(), 0);
    localCellPacketEntries_.assign(tess_.GetPointNo(), 0);
}

void ReverseAdjointTransport3D::recordCellStep(size_t cellIndex)
{
    if (cellIndex < localCellStepCounts_.size())
        ++localCellStepCounts_[cellIndex];
}

void ReverseAdjointTransport3D::recordCellPacketEntry(size_t cellIndex)
{
    if (cellIndex < localCellPacketEntries_.size())
        ++localCellPacketEntries_[cellIndex];
}

void ReverseAdjointTransport3D::initializeObserverData()
{
    observerDirections_ = observer_->getDirections();
    observerSolidAngles_ = observer_->getObserverSolidAngles();

    observerSkyE1_.resize(numObservers_);
    for (size_t p = 0; p < numObservers_; ++p)
        observerSkyE1_[p] = ReverseMueller::observerBasis1(observerDirections_[p]);

    if (!opacity_->energy_groups_boundary.empty())
        groupBoundaries_ = opacity_->energy_groups_boundary;
}

void ReverseAdjointTransport3D::precomputeDDMCData()
{
    size_t nCells = tess_.GetPointNo();

    std::vector<ReverseDDMCCellData> cellData(nCells);

    for (size_t i = 0; i < nCells; ++i)
    {
        auto &cd = cellData[i];
        auto const &cell = cells_[i];
        double scatOp = opacity_->CalcScatteringOpacity(cell);
        double absOp = opacity_->CalcPlanckOpacity(cell);
        double volume = tess_.GetVolume(i);
        double fleck = (i < fleckFactors_.size()) ? fleckFactors_[i] : 1.0;

        double surfaceArea = 0.0;
        for (size_t faceIdx : tess_.GetCellFaces(i))
            surfaceArea += tess_.GetArea(faceIdx);

        if (volume <= 0.0 || surfaceArea <= 0.0)
            continue;

        cd.sigmaA = absOp;
        cd.sigmaS = scatOp;
        cd.sigmaT = absOp + scatOp;
        cd.volume = volume;
        cd.surfaceArea = surfaceArea;
        cd.fleckFactor = fleck;
        cd.meanChordLength = 4.0 * volume / surfaceArea;
        cd.opticalDepthCell = cd.sigmaT * cd.meanChordLength;
        cd.diffusionCoefficient = (cd.sigmaT > 0.0)
            ? units::clight / (3.0 * cd.sigmaT) : 0.0;
        cd.gammaPGRW = 1.0;
        cd.groupCutoff = 0;
        cd.upscatterRateCo = 0.0;

        bool usedMultigroupPGRW = false;
#if ENERGY_GROUPS_NUM > 1
        {
            double const kT = units::k_boltz * cell.temperature;
            if (kT > 0.0)
            {
                double totalSigABgAll = 0.0;
                double totalBgDiff = 0.0;
                double sumBgSigADiff = 0.0;
                double sumBgSigTDiff = 0.0;
                double sumBgOverSigTDiff = 0.0;
                size_t cutoff = 0;
                bool foundNonDiffusive = false;

                for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                {
                    double const a = ComputationalCell3D::energyBoundaries[g] / kT;
                    double const b = ComputationalCell3D::energyBoundaries[g + 1] / kT;
                    double const Bg = planck_integral::planck_integral(a, b);

                    double energyCenter = 0.5 * (ComputationalCell3D::energyBoundaries[g]
                                               + ComputationalCell3D::energyBoundaries[g + 1]);
                    if (!opacity_->energy_groups_center.empty()
                        && g < opacity_->energy_groups_center.size())
                    {
                        energyCenter = opacity_->energy_groups_center[g];
                    }

                    double const sigA_g = opacity_->CalcAbsorptionOpacity(cell, energyCenter);
                    double const sigT_g = sigA_g + scatOp;

                    totalSigABgAll += sigA_g * Bg;

                    if (!foundNonDiffusive
                        && sigT_g * cd.meanChordLength >= config_.ddmcMinCellOpticalDepth)
                    {
                        cutoff = g + 1;
                        totalBgDiff += Bg;
                        sumBgSigADiff += Bg * sigA_g;
                        sumBgSigTDiff += Bg * sigT_g;
                        if (sigT_g > 0.0)
                            sumBgOverSigTDiff += Bg / sigT_g;
                    }
                    else
                    {
                        foundNonDiffusive = true;
                    }
                }

                if (cutoff > 0 && totalBgDiff > 0.0)
                {
                    usedMultigroupPGRW = true;
                    cd.groupCutoff = cutoff;
                    cd.sigmaA_PGRW = sumBgSigADiff / totalBgDiff;
                    cd.sigmaT_PGRW = sumBgSigTDiff / totalBgDiff;
                    cd.diffusionCoefficient_PGRW =
                        (units::clight / 3.0) * sumBgOverSigTDiff / totalBgDiff;
                    cd.gammaPGRW = (totalSigABgAll > 0.0)
                        ? sumBgSigADiff / totalSigABgAll : 1.0;

                    cd.sigmaA = cd.sigmaA_PGRW;
                    cd.sigmaT = cd.sigmaT_PGRW;
                    cd.diffusionCoefficient = cd.diffusionCoefficient_PGRW;
                    cd.opticalDepthCell = cd.sigmaT * cd.meanChordLength;
                }
            }
        }
#endif

        if (usedMultigroupPGRW)
        {
            cd.eligible = (cd.sigmaT > 0.0 && cd.diffusionCoefficient > 0.0);
        }
        else
        {
            cd.eligible = (cd.opticalDepthCell >= config_.ddmcMinCellOpticalDepth
                           && cd.diffusionCoefficient > 0.0);
        }

        cd.expectedScatterRateCo = units::clight * cd.sigmaS;
        cd.resetRateCo = units::clight * (1.0 - fleck) * cd.sigmaA;
        cd.upscatterRateCo = (cd.gammaPGRW < 1.0 && cd.sigmaA > 0.0 && fleck > 0.0)
            ? units::clight * (1.0 - fleck) * cd.sigmaA * (1.0 - cd.gammaPGRW)
            : 0.0;

        if (cd.eligible && config_.ddmcObserverExclusion && observer_)
        {
            Vector3D cellCenter = tess_.GetMeshPoint(i);
            double cellRadius = 0.0;
            for (size_t faceIdx : tess_.GetCellFaces(i))
                cellRadius = std::max(cellRadius,
                    abs(tess_.FaceCM(faceIdx) - cellCenter));
            double charLen = std::max(cd.meanChordLength, cellRadius);
            double distToObs = abs(cellCenter - observerCenter_);
            if (distToObs + charLen >= observerRadius_ &&
                distToObs - charLen <= observerRadius_)
            {
                cd.eligible = false;
                cd.observerExcluded = true;
            }
        }

        if (cd.eligible && config_.ddmcPhotosphereExclusion &&
            cd.opticalDepthCell < config_.ddmcPhotosphereOpticalDepth)
        {
            cd.eligible = false;
            cd.photosphereExcluded = true;
        }
    }

    for (size_t i = 0; i < nCells; ++i)
    {
        auto &cd = cellData[i];
        if (!cd.eligible)
            continue;

        Vector3D cellCenter = tess_.GetMeshPoint(i);
        for (size_t faceIdx : tess_.GetCellFaces(i))
        {
            auto const &neighbors = tess_.GetFaceNeighbors(faceIdx);
            size_t otherCell = (neighbors.first == i) ? neighbors.second : neighbors.first;

            if (otherCell >= nCells)
            {
                cd.boundaryExcluded = true;
                continue;
            }

            Vector3D normal = tess_.Normal(faceIdx);
            if (abs(normal) <= 0.0)
                continue;
            normal = normalize(normal);

            Vector3D faceCenter = tess_.FaceCM(faceIdx);
            double faceDistance = std::abs(ScalarProd(faceCenter - cellCenter, normal));
            if (faceDistance <= 0.0 && otherCell < nCells)
                faceDistance = 0.5 * std::abs(ScalarProd(
                    tess_.GetMeshPoint(otherCell) - cellCenter, normal));
            if (faceDistance <= 0.0)
                continue;

            double diffusionFace = cd.diffusionCoefficient;
            if (otherCell < nCells && cellData[otherCell].diffusionCoefficient > 0.0)
            {
                double a = cd.diffusionCoefficient;
                double b = cellData[otherCell].diffusionCoefficient;
                diffusionFace = (a > 0.0 && b > 0.0) ? (2.0 * a * b / (a + b)) : std::max(a, b);
            }

            double rate = diffusionFace * tess_.GetArea(faceIdx) / (cd.volume * faceDistance);
            if (rate > 0.0 && std::isfinite(rate))
            {
                Vector3D outNormal = normal;
                if (neighbors.second == i)
                    outNormal = normal * (-1.0);

                ReverseDDMCFaceLeak fl;
                fl.faceIndex = faceIdx;
                fl.nextCellIndex = otherCell;
                fl.rate = rate;
                fl.faceCentroid = faceCenter;
                fl.faceNormal = outNormal;
                cd.faceLeaks.push_back(fl);
                cd.totalLeakRate += rate;
            }
        }

        if (cd.boundaryExcluded || cd.totalLeakRate <= 0.0)
        {
            cd.eligible = false;
            cd.faceLeaks.clear();
            cd.totalLeakRate = 0.0;
        }

        cd.totalLeakRatePGRW = cd.totalLeakRate;
    }

    ddmc_.setCellData(std::move(cellData));
}

void ReverseAdjointTransport3D::run()
{
    resetLocalMeasurements();

    int rank = 0;
    int nranks = 1;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif

    size_t totalPacketsPerObsGroup = config_.packetsPerObserverGroup > 0
        ? config_.packetsPerObserverGroup
        : config_.packetsPerObserver;

    size_t const packetsBase = totalPacketsPerObsGroup / static_cast<size_t>(nranks);
    size_t const packetsRemainder = totalPacketsPerObsGroup % static_cast<size_t>(nranks);
    size_t const packetsForThisRankBase = packetsBase
        + ((static_cast<size_t>(rank) < packetsRemainder) ? 1 : 0);

    uint64_t const obsGroups = static_cast<uint64_t>(numObservers_)
        * static_cast<uint64_t>(numGroups_);
    uint64_t const globalTarget = obsGroups
        * static_cast<uint64_t>(totalPacketsPerObsGroup);
    uint64_t const localTarget = obsGroups
        * static_cast<uint64_t>(packetsForThisRankBase);
    uint64_t localCompleted = 0;

    using Clock = std::chrono::steady_clock;
    auto const startTime = Clock::now();
    double const progressInterval = std::max(0.0, config_.progressIntervalSec);
    double nextRank0Heartbeat = progressInterval;
    double nextCollectiveProgress = progressInterval;

    auto elapsedSeconds = [&]() {
        return std::chrono::duration<double>(Clock::now() - startTime).count();
    };

    auto printRank0Heartbeat = [&](char const *phase) {
        if (rank != 0 || progressInterval <= 0.0)
            return;
        double const elapsed = elapsedSeconds();
        if (elapsed < nextRank0Heartbeat)
            return;
        while (nextRank0Heartbeat <= elapsed)
            nextRank0Heartbeat += progressInterval;

        double const rate = (elapsed > 0.0)
            ? static_cast<double>(localCompleted) / elapsed : 0.0;
        double const remaining = (rate > 0.0 && localTarget >= localCompleted)
            ? static_cast<double>(localTarget - localCompleted) / rate
            : std::numeric_limits<double>::infinity();

        std::cerr << "REVERSE_PROGRESS_LOCAL"
                  << " phase=" << phase
                  << " rank=0"
                  << " completed=" << localCompleted
                  << " local_target=" << localTarget
                  << " pct=" << (localTarget > 0
                      ? 100.0 * static_cast<double>(localCompleted) / static_cast<double>(localTarget)
                      : 100.0)
                  << " rank0_packets_per_sec=" << rate
                  << " rank0_eta=" << formatSeconds(remaining)
                  << "\n" << std::flush;
    };

    auto printCollectiveProgress = [&](bool force, char const *phase) {
        double const elapsed = elapsedSeconds();
        bool due = force || progressInterval <= 0.0 || elapsed >= nextCollectiveProgress;

#ifdef RICH_MPI
        int localDue = due ? 1 : 0;
        int anyDue = 0;
        MPI_Allreduce(&localDue, &anyDue, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        due = (anyDue != 0);
#endif

        if (!due)
            return;

        if (progressInterval > 0.0) {
            while (nextCollectiveProgress <= elapsed)
                nextCollectiveProgress += progressInterval;
        }

        unsigned long long localDoneULL =
            static_cast<unsigned long long>(localCompleted);
        unsigned long long globalDoneULL = localDoneULL;
        unsigned long long minRankDoneULL = localDoneULL;
        unsigned long long maxRankDoneULL = localDoneULL;

#ifdef RICH_MPI
        MPI_Allreduce(&localDoneULL, &globalDoneULL, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&localDoneULL, &minRankDoneULL, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localDoneULL, &maxRankDoneULL, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
#endif

        if (rank == 0) {
            double const rate = (elapsed > 0.0)
                ? static_cast<double>(globalDoneULL) / elapsed : 0.0;
            double const remaining = (rate > 0.0 && globalTarget >= globalDoneULL)
                ? static_cast<double>(globalTarget - globalDoneULL) / rate
                : std::numeric_limits<double>::infinity();
            double const meanRankDone = static_cast<double>(globalDoneULL)
                / static_cast<double>(std::max(nranks, 1));

            std::cerr << "REVERSE_PROGRESS"
                      << " phase=" << phase
                      << " completed=" << globalDoneULL
                      << " target=" << globalTarget
                      << " pct=" << (globalTarget > 0
                          ? 100.0 * static_cast<double>(globalDoneULL) / static_cast<double>(globalTarget)
                          : 100.0)
                      << " packets_per_sec=" << rate
                      << " eta=" << formatSeconds(remaining)
                      << " rank_min=" << minRankDoneULL
                      << " rank_mean=" << meanRankDone
                      << " rank_max=" << maxRankDoneULL
                      << " rank_imbalance=" << (meanRankDone > 0.0
                          ? static_cast<double>(maxRankDoneULL) / meanRankDone : 0.0)
                      << "\n" << std::flush;
        }
    };

    if (rank == 0) {
        std::cerr << "REVERSE_PROGRESS_START"
                  << " observers=" << numObservers_
                  << " groups=" << numGroups_
                  << " packets_per_observer_group=" << totalPacketsPerObsGroup
                  << " target=" << globalTarget
                  << " ranks=" << nranks
                  << " rank0_packets_per_observer_group=" << packetsForThisRankBase
                  << "\n" << std::flush;
    }

    for (size_t p = 0; p < numObservers_; ++p)
    {
        for (size_t g = 0; g < numGroups_; ++g)
        {
            size_t const packetsForThisRank = packetsForThisRankBase;

            for (size_t n = 0; n < packetsForThisRank; ++n)
            {
                tallies_.beginPacket();
                ReverseAdjointPacket pkt = launchPacket(p, g);
                ++tallies_.diagnostics().packetsLaunched;
                transportPacket(pkt);
                tallies_.endPacket();
                ++localCompleted;
                printRank0Heartbeat("transport");
            }
            printCollectiveProgress(false, "observer_group");
        }
    }

    printCollectiveProgress(true, "final");

    uint64_t localMeasuredSteps = 0;
    uint64_t localMeasuredEntries = 0;
    uint64_t localActiveCells = 0;
    for (size_t i = 0; i < localCellStepCounts_.size(); ++i) {
        localMeasuredSteps += localCellStepCounts_[i];
        localMeasuredEntries += (i < localCellPacketEntries_.size())
            ? localCellPacketEntries_[i] : 0;
        if (localCellStepCounts_[i] != 0 ||
            (i < localCellPacketEntries_.size() && localCellPacketEntries_[i] != 0))
            ++localActiveCells;
    }

    uint64_t globalMeasuredSteps = localMeasuredSteps;
    uint64_t globalMeasuredEntries = localMeasuredEntries;
    uint64_t globalActiveCells = localActiveCells;
#ifdef RICH_MPI
    MPI_Allreduce(&localMeasuredSteps, &globalMeasuredSteps, 1,
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localMeasuredEntries, &globalMeasuredEntries, 1,
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localActiveCells, &globalActiveCells, 1,
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (rank == 0) {
        std::cerr << "REVERSE_MEASUREMENTS"
                  << " cell_steps=" << globalMeasuredSteps
                  << " packet_entries=" << globalMeasuredEntries
                  << " active_local_cells_sum=" << globalActiveCells
                  << "\n" << std::flush;
    }

    tallies_.mpiReduceToRank0();
    if (useDDMC_)
        ddmc_.mpiReduceDiagnostics();
}

ReverseAdjointPacket ReverseAdjointTransport3D::launchPacket(
    size_t obsIndex, size_t groupIndex)
{
    ReverseAdjointPacket pkt;

    Vector3D obsDir = observerDirections_[obsIndex];
    Vector3D reverseDir = obsDir * (-1.0);
    Vector3D obsPoint = observerCenter_ + observerRadius_ * obsDir;

    pkt.xLab = obsPoint;
    pkt.kForwardLab = obsDir;
    pkt.kReverseLab = reverseDir;

    if (!groupBoundaries_.empty() && groupIndex + 1 < groupBoundaries_.size())
    {
        double lo = groupBoundaries_[groupIndex];
        double hi = groupBoundaries_[groupIndex + 1];
        pkt.nuLab = lo + (hi - lo) * uniform_(rng_);
    }
    else
    {
        pkt.nuLab = 1.0;
    }
    pkt.nuCo = pkt.nuLab;

    pkt.observedGroup = groupIndex;
    pkt.currentCoGroup = groupIndex;
    pkt.observerIndex = obsIndex;

    pkt.scalarWeight = 1.0;
    pkt.M_obs_from_src = MuellerResponse3::identity();

    pkt.observerSkyE1 = observerSkyE1_[obsIndex];
    pkt.sourceBasisLab = ReverseMueller::choosePerpendicularBasis(pkt.kForwardLab);
    pkt.basisInitialized = true;

    // Find first cell inside the domain. If the observer point is outside the mesh,
    // march inward with adaptive step growth to bracket the domain boundary,
    // then bisect to find precise entry.
    size_t cell = findCellForPoint(pkt.xLab);
    if (cell >= tess_.GetPointNo())
    {
        double step = observerRadius_ * 1e-5;
        double maxMarch = 2.0 * observerRadius_;
        double marchOut = 0.0;
        double marchIn = 0.0;
        bool bracketed = false;
        double s = 0.0;

        for (int iter = 0; iter < 10000 && s < maxMarch; ++iter)
        {
            double sNext = s + step;
            Vector3D probe = obsPoint + sNext * reverseDir;
            cell = findCellForPoint(probe);
            if (cell < tess_.GetPointNo())
            {
                marchOut = s;
                marchIn = sNext;
                bracketed = true;
                break;
            }
            s = sNext;
            step *= 1.25;
        }

        if (!bracketed)
        {
            ++tallies_.diagnostics().packetsEscaped;
            pkt.alive = false;
            pkt.cellIndex = std::numeric_limits<size_t>::max();
            return pkt;
        }

        // Bisect between marchOut (outside) and marchIn (inside)
        for (int iter = 0; iter < 50; ++iter)
        {
            double mid = 0.5 * (marchOut + marchIn);
            if (marchIn - marchOut < observerRadius_ * 1e-10)
                break;
            Vector3D probe = obsPoint + mid * reverseDir;
            size_t midCell = findCellForPoint(probe);
            if (midCell < tess_.GetPointNo())
            {
                marchIn = mid;
                cell = midCell;
            }
            else
            {
                marchOut = mid;
            }
        }
        pkt.xLab = obsPoint + marchIn * reverseDir;
    }
    else
    {
        double nudge = observerRadius_ * 1e-8;
        pkt.xLab = obsPoint + nudge * reverseDir;
    }

    pkt.cellIndex = cell;
    pkt.alive = (cell < tess_.GetPointNo());
    if (pkt.alive)
        recordCellPacketEntry(pkt.cellIndex);

    return pkt;
}

void ReverseAdjointTransport3D::transportPacket(ReverseAdjointPacket &pkt)
{
    while (pkt.alive)
    {
        if (pkt.cellIndex >= tess_.GetPointNo())
        {
            terminatePacket(pkt, "outside_domain");
            break;
        }

        recordCellStep(pkt.cellIndex);

        if (pkt.eventCount >= config_.maxEvents)
        {
            ++tallies_.diagnostics().packetsMaxEvents;
            terminatePacket(pkt, "max_events");
            break;
        }

        if (pkt.scalarWeight < config_.weightCutoff)
        {
            ++tallies_.diagnostics().packetsCutoff;
            terminatePacket(pkt, "weight_cutoff");
            break;
        }

        if (!std::isfinite(pkt.scalarWeight) || !std::isfinite(abs(pkt.xLab)))
        {
            ++tallies_.diagnostics().packetsNonFinite;
            terminatePacket(pkt, "non_finite");
            break;
        }

        if (useVelocity_)
        {
            Vector3D cellVel = getCellVelocity(pkt.cellIndex);
            ReverseDoppler::FrameState fs = ReverseDoppler::toComoving(pkt, cellVel);
            if (fs.valid)
            {
                pkt.nuCo = fs.nuCo;
                pkt.currentCoGroup = findGroup(pkt.nuCo);
            }
        }

        if (useDDMC_ && ddmc_.isEligible(pkt) && pkt.ddmcStepCount < config_.maxDDMCSteps)
        {
            auto scoreResidence = [this](
                ReverseAdjointPacket &p, size_t ci, double dtCo,
                double vol, double srcLum, double patchAN,
                double c, double fw, double absDecayRate, double w0)
            {
                tallies_.scoreDDMCResidence(p, ci, dtCo, vol, srcLum, patchAN, c, fw, absDecayRate, w0);
                if (p.currentCoGroup < tallies_.diagnostics().sourceGroupScoreCount.size())
                    ++tallies_.diagnostics().sourceGroupScoreCount[p.currentCoGroup];
            };
            auto scoreResidenceCollapsed = [this](
                ReverseAdjointPacket &p, size_t ci, double dtCo,
                double vol, double srcLum, double patchAN,
                double c, double fw, double absDecayRate, double w0)
            {
                tallies_.scoreDDMCResidenceCollapsed(p, ci, dtCo, vol, srcLum, patchAN, c, fw, absDecayRate, w0);
                auto const &cd = ddmc_.cellData()[ci];
                for (size_t h = 0; h < cd.groupCutoff && h < tallies_.diagnostics().sourceGroupScoreCount.size(); ++h)
                    ++tallies_.diagnostics().sourceGroupScoreCount[h];
            };
            auto terminate = [this](ReverseAdjointPacket &p, std::string const &reason)
            {
                terminatePacket(p, reason);
            };
            auto cellVelFn = [this](size_t ci) { return getCellVelocity(ci); };
            auto srcLumFn = [this](size_t ci, size_t grp) {
                return getSourceLuminosity(ci, grp);
            };
            auto thermalResampleFn = [this](size_t ci) -> ThermalSampleResult {
                ThermalSampleResult result;
                if (ci >= cells_.size())
                {
                    result.ok = false;
                    result.failure = ThermalSampleResult::FailureReason::NoExactSampler;
                    return result;
                }
                auto const &cd = ddmc_.cellData()[ci];
                auto const &cell = cells_[ci];

                if (cd.groupCutoff >= ENERGY_GROUPS_NUM)
                {
                    result.ok = true;
                    result.nuCo = ComputationalCell3D::energyBoundaries[
                        std::min(cd.groupCutoff, static_cast<size_t>(ENERGY_GROUPS_NUM))];
                    result.sampledGroup = ENERGY_GROUPS_NUM - 1;
                    ++tallies_.diagnostics().thermalBoundaryFallbackCount;
                    return result;
                }

                auto findGroup = [&](double nu) -> size_t {
                    for (size_t g = cd.groupCutoff; g < ENERGY_GROUPS_NUM; ++g)
                        if (nu >= ComputationalCell3D::energyBoundaries[g] &&
                            nu < ComputationalCell3D::energyBoundaries[g + 1])
                            return g;
                    return ENERGY_GROUPS_NUM - 1;
                };

                if (multigroupOpacity_)
                {
                    multigroupOpacity_->GetCummulativeOpacity(cell);
                    auto const &cum = multigroupOpacity_->getCummulativeOpacity();
                    double cdfAtCutoff = cum[cd.groupCutoff];
                    double cdfTotal = cum[ENERGY_GROUPS_NUM];

                    if (cdfTotal > cdfAtCutoff)
                    {
                        double lo = cdfAtCutoff / cdfTotal;
                        double xi = uniform_(rng_);
                        double nu = multigroupOpacity_->GetThermalEnergy(cell, lo + xi * (1.0 - lo));
                        result.ok = true;
                        result.exactForward = true;
                        result.nuCo = nu;
                        result.sampledGroup = findGroup(nu);
                        return result;
                    }

                    result.ok = false;
                    result.failure = ThermalSampleResult::FailureReason::EmptyHighEnergyCDF;
                    return result;
                }

                if (!config_.allowApproximateThermalUpscatter)
                {
                    ++tallies_.diagnostics().thermalSamplerFailureCount;
                    result.ok = false;
                    result.failure = ThermalSampleResult::FailureReason::NoExactSampler;
                    return result;
                }

                ++tallies_.diagnostics().thermalSamplerFallbackCount;
                double kT = units::k_boltz * cell.temperature;
                if (kT <= 0.0)
                {
                    result.ok = false;
                    result.failure = ThermalSampleResult::FailureReason::BadTemperature;
                    return result;
                }

                std::array<double, ENERGY_GROUPS_NUM> groupWeights{};
                double totalWeight = 0.0;
                for (size_t g = cd.groupCutoff; g < ENERGY_GROUPS_NUM; ++g)
                {
                    double a = ComputationalCell3D::energyBoundaries[g] / kT;
                    double b = ComputationalCell3D::energyBoundaries[g + 1] / kT;
                    double Bg = planck_integral::planck_integral(a, b);

                    double energyCenter = 0.5 * (ComputationalCell3D::energyBoundaries[g]
                                               + ComputationalCell3D::energyBoundaries[g + 1]);
                    if (!opacity_->energy_groups_center.empty()
                        && g < opacity_->energy_groups_center.size())
                    {
                        energyCenter = opacity_->energy_groups_center[g];
                    }
                    double sigA_g = opacity_->CalcAbsorptionOpacity(cell, energyCenter);
                    groupWeights[g] = Bg * sigA_g;
                    totalWeight += groupWeights[g];
                }

                if (totalWeight <= 0.0)
                {
                    result.ok = false;
                    result.failure = ThermalSampleResult::FailureReason::ZeroApproxWeight;
                    return result;
                }

                double xi = uniform_(rng_) * totalWeight;
                double cumulative = 0.0;
                for (size_t g = cd.groupCutoff; g < ENERGY_GROUPS_NUM; ++g)
                {
                    cumulative += groupWeights[g];
                    if (xi <= cumulative)
                    {
                        double lo = ComputationalCell3D::energyBoundaries[g];
                        double hi = ComputationalCell3D::energyBoundaries[g + 1];
                        result.ok = true;
                        result.approximate = true;
                        result.nuCo = lo + (hi - lo) * uniform_(rng_);
                        result.sampledGroup = g;
                        return result;
                    }
                }

                result.ok = true;
                result.approximate = true;
                result.nuCo = ComputationalCell3D::energyBoundaries[ENERGY_GROUPS_NUM];
                result.sampledGroup = ENERGY_GROUPS_NUM - 1;
                return result;
            };
            double patchAN = getPatchAreaOverN(pkt.observerIndex, pkt.observedGroup);
            double remainingLabForDDMC = transportTime_ - pkt.tLabAccumulated;

            bool stepped = ddmc_.tryStep<std::mt19937_64>(
                pkt, cellVelFn, srcLumFn, thermalResampleFn, patchAN, remainingLabForDDMC,
                scoreResidence, scoreResidenceCollapsed, terminate, rng_, uniform_);

            if (stepped)
            {
                ++pkt.eventCount;
                tallies_.diagnostics().maxTLabAccumulated =
                    std::max(tallies_.diagnostics().maxTLabAccumulated,
                             pkt.tLabAccumulated);
                if (!pkt.alive && pkt.tLabAccumulated >= transportTime_ - 1e-15 * transportTime_)
                    ++tallies_.diagnostics().ddmcTimeLimited;
                continue;
            }
            ++tallies_.diagnostics().ddmcFallbacks;
        }

        ordinaryStep(pkt);
        ++pkt.eventCount;
    }
}

void ReverseAdjointTransport3D::ordinaryStep(ReverseAdjointPacket &pkt)
{
    Vector3D cellVel = useVelocity_ ? getCellVelocity(pkt.cellIndex) : Vector3D(0, 0, 0);
    ReverseDoppler::FrameState fs = ReverseDoppler::toComoving(pkt, cellVel);

    double sigmaA = getAbsorptionOpacity(pkt.cellIndex, fs.valid ? fs.nuCo : pkt.nuLab);
    double sigmaS = getScatteringOpacity(pkt.cellIndex, fs.valid ? fs.nuCo : pkt.nuLab);
    double fleck = (pkt.cellIndex < fleckFactors_.size()) ? fleckFactors_[pkt.cellIndex] : 1.0;

    // Lab-frame opacities: sigma_lab = sigma_co * D (forward MFP = 1/(sigma*D)).
    double D = (fs.valid && fs.dopplerFactor > 0.0) ? fs.dopplerFactor : 1.0;
    double fleckDecayOpacity = fleck * sigmaA * D;
    double resetOpacity = (1.0 - fleck) * sigmaA * D;
    double eventOpacity = resetOpacity + sigmaS * D;

    size_t exitFace = std::numeric_limits<size_t>::max();
    size_t nextCell = std::numeric_limits<size_t>::max();
    double sFace = distanceToFace(pkt.xLab, pkt.kReverseLab, pkt.cellIndex,
                                  exitFace, nextCell);

    double sEvent = std::numeric_limits<double>::max();
    if (eventOpacity > 0.0)
    {
        double xi = std::max(uniform_(rng_), 1e-300);
        sEvent = -std::log(xi) / eventOpacity;
    }

    // Time census: clamp step to remaining transport time
    double remainingTime = transportTime_ - pkt.tLabAccumulated;
    double sTime = remainingTime * units::clight;
    if (sTime <= 0.0)
    {
        terminatePacket(pkt, "time_census");
        ++tallies_.diagnostics().timeCensusCount;
        return;
    }

    double ds = std::min({sFace, sEvent, sTime});
    bool hitTimeCensus = (ds >= sTime - 1e-15 * std::abs(sTime));
    if (ds < 0.0 || !std::isfinite(ds))
    {
        terminatePacket(pkt, "invalid_step_distance");
        return;
    }

    double cellVolume = tess_.GetVolume(pkt.cellIndex);
    double srcLum = getSourceLuminosity(pkt.cellIndex, pkt.currentCoGroup);
    double patchAN = getPatchAreaOverN(pkt.observerIndex, pkt.observedGroup);
    double frameWeight = fs.valid ? fs.frameWeightFactor : 1.0;

    tallies_.scoreOrdinarySegment(pkt, pkt.cellIndex, ds, cellVolume,
                                  srcLum, patchAN, frameWeight, fleckDecayOpacity);
    if (pkt.currentCoGroup < tallies_.diagnostics().sourceGroupScoreCount.size())
        ++tallies_.diagnostics().sourceGroupScoreCount[pkt.currentCoGroup];

    pkt.scalarWeight *= safeExp(-fleckDecayOpacity * ds);
    pkt.xLab = pkt.xLab + ds * pkt.kReverseLab;
    pkt.pathLabAccumulated += ds;
    pkt.tLabAccumulated += ds * units::inv_clight;
    ++tallies_.diagnostics().ordinarySteps;

    if (hitTimeCensus)
    {
        terminatePacket(pkt, "time_census");
        ++tallies_.diagnostics().timeCensusCount;
        return;
    }

    if (ds == sFace || (sFace < sEvent && std::abs(sFace - ds) < 1e-15 * ds))
    {
        crossFace(pkt, nextCell);
        return;
    }

    double xi2 = uniform_(rng_);
    double sigmaS_lab = sigmaS * D;
    double pScatter = (eventOpacity > 0.0) ? sigmaS_lab / eventOpacity : 0.0;

    if (xi2 < pScatter)
    {
        handleReverseThomsonScatter(pkt, fs, cellVel);
    }
    else
    {
        handleReverseReset(pkt);
    }
}

void ReverseAdjointTransport3D::handleReverseThomsonScatter(
    ReverseAdjointPacket &pkt,
    ReverseDoppler::FrameState const &fs,
    Vector3D const &cellVelocity)
{
    auto u01 = [this]() -> double { return uniform_(rng_); };

    Vector3D oldForwardLab = pkt.kForwardLab;
    Vector3D oldForwardCo = (fs.valid && useVelocity_) ? fs.kForwardCo : oldForwardLab;

    Vector3D newForwardCo = ReverseMueller::sampleIsotropicThomsonDirection(oldForwardCo, u01);

    Vector3D newForwardLab = (fs.valid && useVelocity_)
        ? ReverseDoppler::comovingDirToLab(newForwardCo, cellVelocity, fs.gamma)
        : newForwardCo;

    if (usePolarization_)
    {
        Vector3D oldBasisLab = pkt.basisInitialized
            ? pkt.sourceBasisLab
            : ReverseMueller::choosePerpendicularBasis(oldForwardLab);

        // Lorentz-transform basis from lab screen to comoving screen
        Vector3D oldBasisCo = (fs.valid && useVelocity_)
            ? ReverseDoppler::labBasisToComovingScreen(
                  oldBasisLab, oldForwardLab, oldForwardCo, cellVelocity, fs.gamma)
            : ReverseMueller::projectBasisToDirection(oldBasisLab, oldForwardCo);

        // Set comoving basis for Mueller update
        pkt.sourceBasisLab = oldBasisCo;
        pkt.basisInitialized = true;

        // Mueller update in the comoving frame
        ReverseMueller::applyReverseThomsonScatter(pkt, newForwardCo, oldForwardCo, u01);

        // After scatter, sourceBasisLab is the new comoving basis (perpendicular to newForwardCo)
        Vector3D newBasisCo = pkt.sourceBasisLab;

        // Lorentz-transform basis back from comoving screen to lab screen
        Vector3D newBasisLab = (fs.valid && useVelocity_)
            ? ReverseDoppler::comovingBasisToLabScreen(
                  newBasisCo, newForwardCo, newForwardLab, cellVelocity, fs.gamma)
            : ReverseMueller::projectBasisToDirection(newBasisCo, newForwardLab);

        pkt.sourceBasisLab = newBasisLab;
    }
    else
    {
        ++pkt.scatterCountExplicit;
    }

    pkt.kForwardLab = newForwardLab;
    pkt.kReverseLab = newForwardLab * (-1.0);
}

void ReverseAdjointTransport3D::handleReverseReset(ReverseAdjointPacket &pkt)
{
    auto u01 = [this]() -> double { return uniform_(rng_); };
    Vector3D newDir = ReverseMueller::sampleIsotropicDirection(u01);

    pkt.M_obs_from_src.resetToUnpolarizedSource();
    pkt.kForwardLab = newDir;
    pkt.kReverseLab = newDir * (-1.0);
    pkt.sourceBasisLab = ReverseMueller::choosePerpendicularBasis(newDir);
    pkt.basisInitialized = true;
    ++pkt.resetCount;
}

void ReverseAdjointTransport3D::crossFace(ReverseAdjointPacket &pkt,
                                           size_t newCell)
{
    if (newCell >= tess_.GetPointNo())
    {
        terminatePacket(pkt, "boundary_escape");
        ++tallies_.diagnostics().packetsEscaped;
        return;
    }

    // Nudge across the face to avoid zero-length re-intersections.
    double nudge = 1e-10 * std::cbrt(tess_.GetVolume(newCell));
    pkt.xLab = pkt.xLab + nudge * pkt.kReverseLab;
    pkt.cellIndex = newCell;
    recordCellPacketEntry(pkt.cellIndex);
    ++pkt.faceCrossingCount;
}

void ReverseAdjointTransport3D::terminatePacket(ReverseAdjointPacket &pkt,
                                                 std::string const &)
{
    pkt.alive = false;
    if (pkt.tLabAccumulated > tallies_.diagnostics().maxTLabAccumulated)
        tallies_.diagnostics().maxTLabAccumulated = pkt.tLabAccumulated;
}

double ReverseAdjointTransport3D::getSourceLuminosity(
    size_t cellIndex, size_t groupIndex) const
{
    if (cellIndex >= cells_.size())
        return 0.0;

    auto const &cell = cells_[cellIndex];
    double fleck = (cellIndex < fleckFactors_.size()) ? fleckFactors_[cellIndex] : 1.0;
    double volume = tess_.GetVolume(cellIndex);
    double T4 = cell.temperature * cell.temperature * cell.temperature * cell.temperature;

    if (groupIndex < static_cast<size_t>(ENERGY_GROUPS_NUM) && ENERGY_GROUPS_NUM > 1)
    {
        double kT = units::k_boltz * cell.temperature;
        if (kT <= 0.0)
            return 0.0;

        double a = ComputationalCell3D::energyBoundaries[groupIndex] / kT;
        double b = ComputationalCell3D::energyBoundaries[groupIndex + 1] / kT;
        double Bg = planck_integral::planck_integral(a, b);

        double energyCenter = 0.5 * (ComputationalCell3D::energyBoundaries[groupIndex]
                                   + ComputationalCell3D::energyBoundaries[groupIndex + 1]);
        if (!opacity_->energy_groups_center.empty()
            && groupIndex < opacity_->energy_groups_center.size())
        {
            energyCenter = opacity_->energy_groups_center[groupIndex];
        }

        double sigmaA_g = opacity_->CalcAbsorptionOpacity(cell, energyCenter);
        return 4.0 * units::sigma_sb * T4 * sigmaA_g * Bg * fleck * volume;
    }

    double sigmaA = opacity_->CalcPlanckOpacity(cell);
    return 4.0 * units::sigma_sb * T4 * sigmaA * fleck * volume;
}

Vector3D ReverseAdjointTransport3D::getCellVelocity(size_t cellIndex) const
{
    if (!useVelocity_ || cellIndex >= cells_.size())
        return Vector3D(0, 0, 0);
    return cells_[cellIndex].velocity;
}

double ReverseAdjointTransport3D::getAbsorptionOpacity(
    size_t cellIndex, double frequency) const
{
    if (cellIndex >= cells_.size())
        return 0.0;
    return opacity_->CalcAbsorptionOpacity(cells_[cellIndex], frequency);
}

double ReverseAdjointTransport3D::getScatteringOpacity(
    size_t cellIndex, double frequency) const
{
    if (cellIndex >= cells_.size())
        return 0.0;
    return opacity_->CalcScatteringOpacity(cells_[cellIndex], frequency);
}

double ReverseAdjointTransport3D::getPatchAreaOverN(
    size_t obsIndex, size_t /*groupIndex*/) const
{
    size_t totalPackets = config_.packetsPerObserverGroup > 0
        ? config_.packetsPerObserverGroup
        : config_.packetsPerObserver;
    if (totalPackets == 0)
        totalPackets = 1;
    static constexpr double FOUR_PI = 4.0 * 3.141592653589793238462643383279502884;
    // Use per-observer solid angle when available; fall back to global patchArea_.
    double solidAngle = (!observerSolidAngles_.empty() && obsIndex < observerSolidAngles_.size())
        ? observerSolidAngles_[obsIndex] : (patchArea_ / (observerRadius_ * observerRadius_));
    double area = solidAngle * observerRadius_ * observerRadius_;
    return area / (FOUR_PI * static_cast<double>(totalPackets));
}

size_t ReverseAdjointTransport3D::findCellForPoint(Vector3D const &point) const
{
    return tess_.GetContainingCell(point);
}

double ReverseAdjointTransport3D::distanceToFace(
    Vector3D const &pos, Vector3D const &dir,
    size_t cellIndex, size_t &exitFace, size_t &nextCell) const
{
    auto const &faces = tess_.GetCellFaces(cellIndex);
    double minDist = std::numeric_limits<double>::max();
    exitFace = std::numeric_limits<size_t>::max();
    nextCell = std::numeric_limits<size_t>::max();

    double dirSpeed = abs(dir);
    if (dirSpeed < 1e-30)
        return minDist;

    for (size_t i = 0; i < faces.size(); ++i)
    {
        size_t faceIdx = faces[i];
        Vector3D normal = tess_.Normal(faceIdx);
        Vector3D pointOnFace = tess_.FaceCM(faceIdx);

        double nDotV = ScalarProd(normal, dir);
        if (std::abs(nDotV) < 1e-30 * abs(normal) * dirSpeed)
            continue;

        double alpha = ScalarProd(pointOnFace - pos, normal) / nDotV;

        if (alpha < -1e-10 * abs(pointOnFace - pos) / dirSpeed)
            continue;
        if (alpha < 0.0)
            alpha = 0.0;

        double dist = alpha * dirSpeed;

        if (dist < minDist)
        {
            minDist = dist;
            exitFace = faceIdx;
            auto const &neighbors = tess_.GetFaceNeighbors(faceIdx);
            nextCell = (neighbors.first == cellIndex)
                           ? neighbors.second
                           : neighbors.first;
        }
    }

    return minDist;
}

size_t ReverseAdjointTransport3D::findGroup(double frequency) const
{
    return opacity_->findGroup(frequency);
}

void ReverseAdjointTransport3D::writeOutputs(std::string const &filename) const
{
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if (rank != 0)
        return;

    // Use a separate file for reverse output to avoid overwriting forward results.
    std::string reverseFilename = filename;
    {
        auto pos = reverseFilename.rfind(".h5");
        if (pos == std::string::npos)
            pos = reverseFilename.rfind(".hdf5");
        if (pos != std::string::npos)
            reverseFilename.insert(pos, "_reverse");
        else
            reverseFilename += "_reverse";
    }

    HDF5Writer writer(reverseFilename);
    std::string prefix = "/" + config_.outputPrefix;

    // Write full tallies via the tally writer (group, scatter-order data included)
    tallies_.writeHDF5(writer, prefix);

    // Derived quantities
    std::vector<double> lumI, q, u, polDeg, polAngle;
    tallies_.finalizeDerived(lumI, q, u, polDeg, polAngle);
    writer.WriteElement(prefix + "/observer_luminosity", lumI);
    writer.WriteElement(prefix + "/observer_q", q);
    writer.WriteElement(prefix + "/observer_u", u);
    writer.WriteElement(prefix + "/observer_polarization_degree", polDeg);
    writer.WriteElement(prefix + "/observer_polarization_angle", polAngle);

    // Transport diagnostics
    auto const &diag = tallies_.diagnostics();
    auto writeU64 = [&](std::string const &name, uint64_t val) {
        double dval = static_cast<double>(val);
        writer.WriteElement(prefix + "/diagnostics/" + name, dval);
    };
    writeU64("packets_launched", diag.packetsLaunched);
    writeU64("packets_cutoff", diag.packetsCutoff);
    writeU64("packets_max_events", diag.packetsMaxEvents);
    writeU64("packets_escaped", diag.packetsEscaped);
    writeU64("packets_non_finite", diag.packetsNonFinite);
    writeU64("ordinary_steps", diag.ordinarySteps);
    writeU64("ddmc_steps", diag.ddmcSteps);
    writeU64("ddmc_fallbacks", diag.ddmcFallbacks);
    writeU64("ddmc_time_limited", diag.ddmcTimeLimited);
    writeU64("time_census_count", diag.timeCensusCount);
    writeU64("non_finite_score_count", diag.nonFiniteScoreCount);
    writeU64("negative_I_count", diag.negativeICount);
    writeU64("thermal_sampler_fallback_count", diag.thermalSamplerFallbackCount);
    writeU64("thermal_sampler_failure_count", diag.thermalSamplerFailureCount);
    writeU64("thermal_boundary_fallback_count", diag.thermalBoundaryFallbackCount);
    if (!diag.sourceGroupScoreCount.empty())
    {
        std::vector<double> sgsc(diag.sourceGroupScoreCount.begin(),
                                 diag.sourceGroupScoreCount.end());
        writer.WriteElement(prefix + "/diagnostics/source_group_score_count", sgsc);
    }
    writer.WriteElement(prefix + "/diagnostics/max_mueller_norm",
                        diag.maxMuellerNorm);
    writer.WriteElement(prefix + "/diagnostics/max_t_lab_accumulated",
                        diag.maxTLabAccumulated);

    // DDMC-specific diagnostics (all under diagnostics/ddmc/)
    auto const &ddDiag = ddmc_.diagnostics();
    auto writeDDMC = [&](std::string const &name, uint64_t val) {
        writer.WriteElement(prefix + "/diagnostics/ddmc/" + name,
                            static_cast<double>(val));
    };
    writeDDMC("leak_count", ddDiag.leakCount);
    writeDDMC("fallback_not_eligible", ddDiag.fallbackNotEligible);
    writeDDMC("fallback_particle_depth", ddDiag.fallbackParticleDepth);
    writeDDMC("fallback_bad_frame", ddDiag.fallbackBadFrame);
    writeDDMC("fallback_bad_rates", ddDiag.fallbackBadRates);
    writeDDMC("fallback_mueller_norm", ddDiag.fallbackMuellerNorm);
    writeDDMC("upscatter_count", ddDiag.upscatterCount);
    writeDDMC("fallback_above_cutoff", ddDiag.fallbackAboveCutoff);
    writeDDMC("fallback_no_thermal_sampler", ddDiag.fallbackNoThermalSampler);
    writeDDMC("census_count", ddDiag.censusCount);
    writeDDMC("time_limited_step_count", ddDiag.timeLimitedStepCount);
    writeDDMC("total_residence_time_lab", ddDiag.totalResidenceTimeLab);
    writeDDMC("pol_closure_applied", ddDiag.polClosureApplied);
    writeDDMC("pol_closure_depolarized", ddDiag.polClosureDepolarized);
    writeDDMC("closure_attempt_count", ddDiag.closureAttemptCount);
    std::string closureMode = ddmc_.closureModeName();
    writer.WriteElement(prefix + "/diagnostics/ddmc/closure_mode", closureMode);
    writeDDMC("resets_during_residence", ddDiag.totalResetsDuringResidence);
    writeDDMC("collapsed_pgrw_score_count", ddDiag.collapsedPgrwScoreCount);
    writeDDMC("exact_group_score_count", ddDiag.exactGroupScoreCount);
    if (!ddDiag.upscatterByGroup.empty())
    {
        std::vector<double> ubg(ddDiag.upscatterByGroup.begin(), ddDiag.upscatterByGroup.end());
        writer.WriteElement(prefix + "/diagnostics/ddmc/upscatter_count_by_group", ubg);
    }
    if (!ddDiag.thermalSampledGroupCount.empty())
    {
        std::vector<double> tsg(ddDiag.thermalSampledGroupCount.begin(), ddDiag.thermalSampledGroupCount.end());
        writer.WriteElement(prefix + "/diagnostics/ddmc/thermal_sampled_group_count", tsg);
    }
    if (!ddDiag.thermalExactSampledGroupCount.empty())
    {
        std::vector<double> tesg(ddDiag.thermalExactSampledGroupCount.begin(), ddDiag.thermalExactSampledGroupCount.end());
        writer.WriteElement(prefix + "/diagnostics/ddmc/thermal_exact_sampled_group_count", tesg);
    }
    if (!ddDiag.thermalApproxSampledGroupCount.empty())
    {
        std::vector<double> tasg(ddDiag.thermalApproxSampledGroupCount.begin(), ddDiag.thermalApproxSampledGroupCount.end());
        writer.WriteElement(prefix + "/diagnostics/ddmc/thermal_approx_sampled_group_count", tasg);
    }
    {
        std::vector<double> tfbr(ddDiag.thermalFailureByReason.begin(), ddDiag.thermalFailureByReason.end());
        writer.WriteElement(prefix + "/diagnostics/ddmc/thermal_failure_by_reason", tfbr);
    }
    double nResidence = static_cast<double>(std::max(ddDiag.residenceCount, uint64_t(1)));
    double nClosure = static_cast<double>(std::max(ddDiag.polClosureApplied, uint64_t(1)));
    writeDDMC("residence_count", ddDiag.residenceCount);
    writer.WriteElement(prefix + "/diagnostics/ddmc/mean_residence_time_co",
                        ddDiag.totalResidenceTimeCo / nResidence);
    writer.WriteElement(prefix + "/diagnostics/ddmc/mean_expected_scatter_count",
                        (ddDiag.polClosureApplied > 0)
                            ? ddDiag.totalExpectedScatterCount / nClosure
                            : -1.0);
    writer.WriteElement(prefix + "/diagnostics/ddmc/mean_polarization_damping",
                        (ddDiag.polClosureApplied > 0)
                            ? ddDiag.totalPolarizationDamping / nClosure
                            : -1.0);

    // PGRW multigroup diagnostics
#if ENERGY_GROUPS_NUM > 1
    if (useDDMC_)
    {
        auto const &cellDataVec = ddmc_.cellData();
        size_t nCellsDD = cellDataVec.size();
        std::vector<double> cutoffVec(nCellsDD), gammaVec(nCellsDD);
        std::vector<double> sigAPGRW(nCellsDD), sigTPGRW(nCellsDD);
        std::vector<double> dPGRW(nCellsDD), upscRateVec(nCellsDD);
        for (size_t c = 0; c < nCellsDD; ++c)
        {
            cutoffVec[c] = static_cast<double>(cellDataVec[c].groupCutoff);
            gammaVec[c] = cellDataVec[c].gammaPGRW;
            sigAPGRW[c] = cellDataVec[c].sigmaA_PGRW;
            sigTPGRW[c] = cellDataVec[c].sigmaT_PGRW;
            dPGRW[c] = cellDataVec[c].diffusionCoefficient_PGRW;
            upscRateVec[c] = cellDataVec[c].upscatterRateCo;
        }
        writer.WriteElement(prefix + "/diagnostics/ddmc/pgrw_group_cutoff", cutoffVec);
        writer.WriteElement(prefix + "/diagnostics/ddmc/pgrw_gamma", gammaVec);
        writer.WriteElement(prefix + "/diagnostics/ddmc/pgrw_sigmaA", sigAPGRW);
        writer.WriteElement(prefix + "/diagnostics/ddmc/pgrw_sigmaT", sigTPGRW);
        writer.WriteElement(prefix + "/diagnostics/ddmc/pgrw_diffusion_coefficient", dPGRW);
        writer.WriteElement(prefix + "/diagnostics/ddmc/pgrw_upscatter_rate", upscRateVec);
    }
#endif

    // Metadata
    std::string sourceFrame = "comoving";
    std::string observerFrame = "lab";
    std::string normMode = "area_normalized_4pi";
    writer.WriteElement(prefix + "/diagnostics/source_frame", sourceFrame);
    writer.WriteElement(prefix + "/diagnostics/observer_frame", observerFrame);
    writer.WriteElement(prefix + "/diagnostics/normalization_mode", normMode);
    writer.WriteElement(prefix + "/diagnostics/source_dt", sourceDt_);
    writer.WriteElement(prefix + "/diagnostics/transport_time", transportTime_);

    double useVel = useVelocity_ ? 1.0 : 0.0;
    double useDDMCFlag = useDDMC_ ? 1.0 : 0.0;
    double usePol = usePolarization_ ? 1.0 : 0.0;
    writer.WriteElement(prefix + "/diagnostics/velocity_enabled", useVel);
    writer.WriteElement(prefix + "/diagnostics/ddmc_enabled", useDDMCFlag);
    writer.WriteElement(prefix + "/diagnostics/polarization_enabled", usePol);

    std::string velModel = useVelocity_
        ? std::string("full_lorentz_aberration")
        : std::string("static");
    writer.WriteElement(prefix + "/diagnostics/velocity_model", velModel);
    std::string ddmcModel = useDDMC_
        ? (ENERGY_GROUPS_NUM > 1 ? std::string("pgrw_multigroup_ddmc")
                                 : std::string("grey_ddmc"))
        : std::string("none");
    writer.WriteElement(prefix + "/diagnostics/ddmc_model", ddmcModel);

    // PGRW spectral semantics and thermal sampler metadata
    std::string multigroupMode;
    if (ENERGY_GROUPS_NUM > 1 && useDDMC_)
    {
        switch (config_.multigroupDDMCMode)
        {
        case ReverseMultigroupDDMCMode::ExactGroup:
            multigroupMode = "exact_per_group"; break;
        case ReverseMultigroupDDMCMode::PGRWCollapsed:
            multigroupMode = "pgrw_collapsed_marginalized"; break;
        default:
            multigroupMode = "grey"; break;
        }
    }
    else
        multigroupMode = "grey_or_disabled";
    writer.WriteElement(prefix + "/diagnostics/ddmc/multigroup_mode", multigroupMode);

    std::string pgrwSpectralSemantics = (ENERGY_GROUPS_NUM > 1 && useDDMC_)
        ? std::string("collapsed_low_group_aggregate_not_resolved")
        : std::string("not_applicable");
    writer.WriteElement(prefix + "/diagnostics/ddmc/pgrw_spectral_semantics", pgrwSpectralSemantics);
    if (ENERGY_GROUPS_NUM > 1 && useDDMC_
        && config_.multigroupDDMCMode == ReverseMultigroupDDMCMode::PGRWCollapsed)
    {
        std::string specNote("group_luminosity_below_pgrw_cutoff_is_collapsed_not_resolved");
        writer.WriteElement(prefix + "/metadata/group_spectra_below_pgrw_cutoff", specNote);
    }

    bool hasMultigroupOpacity = (multigroupOpacity_ != nullptr);
    std::string thermalSampler = hasMultigroupOpacity
        ? std::string("forward_cumulative_opacity_exact")
        : std::string("approx_Bg_sigmaA_uniform_bin");
    writer.WriteElement(prefix + "/diagnostics/ddmc/thermal_upscatter_sampler", thermalSampler);
    double thermalExact = hasMultigroupOpacity ? 1.0 : 0.0;
    writer.WriteElement(prefix + "/diagnostics/ddmc/thermal_upscatter_forward_exact", thermalExact);
    writer.WriteElement(prefix + "/diagnostics/ddmc/thermal_upscatter_fallback_count",
                        static_cast<double>(diag.thermalSamplerFallbackCount));
    writer.WriteElement(prefix + "/diagnostics/ddmc/thermal_upscatter_approx_allowed",
                        config_.allowApproximateThermalUpscatter ? 1.0 : 0.0);

    std::string basisModel = useVelocity_
        ? std::string("lorentz_electric_field_transform")
        : std::string("projection_only");
    writer.WriteElement(prefix + "/diagnostics/velocity_basis_transform_model", basisModel);

    std::string patchModel("point_patch_far_field");
    writer.WriteElement(prefix + "/diagnostics/patch_sampling", patchModel);

    // Fleck factor diagnostics
    writer.WriteElement(prefix + "/metadata/fleck_defaulted_to_one",
                        fleckDefaultedToOne_ ? 1.0 : 0.0);
    if (!fleckFactors_.empty())
    {
        double fMin = *std::min_element(fleckFactors_.begin(), fleckFactors_.end());
        double fMax = *std::max_element(fleckFactors_.begin(), fleckFactors_.end());
        double fSum = 0.0;
        size_t fEqOne = 0, fLtOne = 0;
        for (double f : fleckFactors_)
        {
            fSum += f;
            if (f >= 1.0 - 1e-14) ++fEqOne;
            else ++fLtOne;
        }
        writer.WriteElement(prefix + "/diagnostics/fleck_min", fMin);
        writer.WriteElement(prefix + "/diagnostics/fleck_max", fMax);
        writer.WriteElement(prefix + "/diagnostics/fleck_mean", fSum / fleckFactors_.size());
        writer.WriteElement(prefix + "/diagnostics/fleck_cells_equal_one",
                            static_cast<double>(fEqOne));
        writer.WriteElement(prefix + "/diagnostics/fleck_cells_less_than_one",
                            static_cast<double>(fLtOne));
    }
    // DDMC upscatter nonzero cell count
    if (useDDMC_)
    {
        size_t upscNonzero = 0;
        for (auto const &cd : ddmc_.cellData())
            if (cd.upscatterRateCo > 0.0) ++upscNonzero;
        writer.WriteElement(prefix + "/diagnostics/ddmc/upscatter_rate_nonzero_cells",
                            static_cast<double>(upscNonzero));
    }

    // Estimator mode metadata
    std::string modeStr = "standalone";
    switch (config_.estimatorMode) {
    case PostProcessEstimatorMode::Reverse: modeStr = "reverse"; break;
    case PostProcessEstimatorMode::Both:    modeStr = "both"; break;
    default: break;
    }
    writer.WriteElement(prefix + "/metadata/estimator_mode", modeStr);
    writer.WriteElement(prefix + "/metadata/fleck_parity_checked",
                        fleckFromForwardVector_ ? 1.0 : 0.0);
    writer.WriteElement(prefix + "/metadata/fleck_source",
        fleckFromForwardVector_
            ? std::string("forward_vector_shared")
            : std::string("helper_recomputed"));
    writer.WriteElement(prefix + "/metadata/fleck_forward_vector_shared",
                        fleckFromForwardVector_ ? 1.0 : 0.0);
    writer.WriteElement(prefix + "/metadata/fleck_forward_parity_tested",
                        fleckFromForwardVector_ ? 1.0 : 0.0);
    writer.WriteElement(prefix + "/metadata/fleck_parity_note",
        fleckFromForwardVector_
            ? std::string(ReverseOutputStrings::FleckForwardVectorShared)
            : std::string(ReverseOutputStrings::FleckHelperNotForward));
    writer.WriteElement(prefix + "/metadata/comparison_available",
                        comparisonWritten_ ? 1.0 : 0.0);
    if (comparisonWritten_)
    {
        writer.WriteElement(prefix + "/metadata/comparison_file", comparisonFile_);
        writer.WriteElement(prefix + "/metadata/comparison_group", comparisonGroup_);
    }
    std::string pgrwOutputSemantics =
        (config_.multigroupDDMCMode == ReverseMultigroupDDMCMode::PGRWCollapsed)
        ? std::string("collapsed_routed_to_separate_dataset")
        : std::string("resolved_per_group");
    writer.WriteElement(prefix + ReverseOutputStrings::PGRWEstimatorGroupOutputModeMetadataPath,
                        pgrwOutputSemantics);

    writer.Close();
}

void ReverseAdjointTransport3D::writeComparisonOutputs(
    std::string const &filename,
    std::string const &prefix,
    std::vector<double> const &fwdLum,
    std::vector<double> const &revLum,
    std::vector<double> const &delta,
    std::vector<double> const &relDelta,
    std::vector<double> const &fwdQ,
    std::vector<double> const &fwdU,
    std::vector<std::vector<double>> const &fwdGroupLum,
    std::vector<std::vector<double>> const &fwdGroupQ,
    std::vector<std::vector<double>> const &fwdGroupU) const
{
    HDF5Writer writer(filename);
    writer.WriteElement(prefix + "/observer_luminosity_forward", fwdLum);
    writer.WriteElement(prefix + "/observer_luminosity_reverse", revLum);
    writer.WriteElement(prefix + "/observer_luminosity_delta", delta);
    writer.WriteElement(prefix + "/observer_luminosity_rel_delta", relDelta);
    writer.WriteElement(prefix + "/reverse_packet_count",
        static_cast<double>(tallies_.diagnostics().packetsLaunched));

    // Group luminosity comparison (total = resolved + collapsed)
    size_t nGrp = tallies_.numGroups();
    auto const &revGrpI = tallies_.groupI();
    auto const &collI = tallies_.collapsedI();
    for (size_t p = 0; p < numObservers_; ++p)
    {
        std::string gp = prefix + "/group_luminosity_" + std::to_string(p);
        if (p < fwdGroupLum.size())
            writer.WriteElement(gp + "/forward", fwdGroupLum[p]);

        // Compute total reverse group = resolved + collapsed
        std::vector<double> revTotal(nGrp, 0.0);
        if (p < revGrpI.size())
            for (size_t g = 0; g < nGrp; ++g)
                revTotal[g] = revGrpI[p][g] + collI[p][g];
        writer.WriteElement(gp + "/reverse_total", revTotal);
        if (p < revGrpI.size())
            writer.WriteElement(gp + "/reverse_resolved", revGrpI[p]);

        if (p < fwdGroupLum.size())
        {
            std::vector<double> grpDelta(nGrp), grpRelDelta(nGrp);
            for (size_t g = 0; g < nGrp && g < fwdGroupLum[p].size(); ++g)
            {
                grpDelta[g] = revTotal[g] - fwdGroupLum[p][g];
                double denom = std::max(std::abs(fwdGroupLum[p][g]), 1e-30);
                grpRelDelta[g] = grpDelta[g] / denom;
            }
            writer.WriteElement(gp + "/delta", grpDelta);
            writer.WriteElement(gp + "/rel_delta", grpRelDelta);
        }
    }

    // Group Q/U polarization comparison
    auto const &revGrpQ = tallies_.groupQ();
    auto const &revGrpU = tallies_.groupU();
    auto const &collQ = tallies_.collapsedQ();
    auto const &collU = tallies_.collapsedU();
    for (size_t p = 0; p < numObservers_; ++p)
    {
        std::string gp = prefix + "/group_polarization_" + std::to_string(p);
        std::vector<double> revQTotal(nGrp, 0.0), revUTotal(nGrp, 0.0);
        for (size_t g = 0; g < nGrp; ++g)
        {
            revQTotal[g] = revGrpQ[p][g] + collQ[p][g];
            revUTotal[g] = revGrpU[p][g] + collU[p][g];
        }
        writer.WriteElement(gp + "/Q_reverse_total", revQTotal);
        writer.WriteElement(gp + "/U_reverse_total", revUTotal);
        if (p < fwdGroupQ.size())
            writer.WriteElement(gp + "/Q_forward", fwdGroupQ[p]);
        if (p < fwdGroupU.size())
            writer.WriteElement(gp + "/U_forward", fwdGroupU[p]);

        if (p < fwdGroupQ.size() && p < fwdGroupU.size() &&
            p < fwdGroupLum.size())
        {
            std::vector<double> grpPolDegFwd(nGrp, 0.0), grpPolDegRev(nGrp, 0.0);
            std::vector<double> grpPolDegDelta(nGrp, 0.0);
            std::vector<double> grpPolAngFwd(nGrp, 0.0), grpPolAngRev(nGrp, 0.0);
            std::vector<double> grpPolAngDelta(nGrp, 0.0);
            for (size_t g = 0; g < nGrp && g < fwdGroupQ[p].size(); ++g)
            {
                double fI = (g < fwdGroupLum[p].size()) ? fwdGroupLum[p][g] : 0.0;
                double fQ = fwdGroupQ[p][g];
                double fU = (g < fwdGroupU[p].size()) ? fwdGroupU[p][g] : 0.0;
                double rI = revGrpI[p][g] + collI[p][g];
                double rQ = revQTotal[g];
                double rU = revUTotal[g];

                if (fI > 0.0)
                {
                    grpPolDegFwd[g] = std::sqrt(fQ * fQ + fU * fU) / fI;
                    grpPolAngFwd[g] = 0.5 * std::atan2(fU, fQ);
                }
                if (rI > 0.0)
                {
                    grpPolDegRev[g] = std::sqrt(rQ * rQ + rU * rU) / rI;
                    grpPolAngRev[g] = 0.5 * std::atan2(rU, rQ);
                }
                grpPolDegDelta[g] = grpPolDegRev[g] - grpPolDegFwd[g];

                double angDiff = grpPolAngRev[g] - grpPolAngFwd[g];
                constexpr double pi = 3.141592653589793238462643383279502884;
                while (angDiff >  0.5 * pi) angDiff -= pi;
                while (angDiff < -0.5 * pi) angDiff += pi;
                grpPolAngDelta[g] = angDiff;
            }
            writer.WriteElement(gp + "/polarization_degree_forward", grpPolDegFwd);
            writer.WriteElement(gp + "/polarization_degree_reverse", grpPolDegRev);
            writer.WriteElement(gp + "/polarization_degree_delta", grpPolDegDelta);
            writer.WriteElement(gp + "/polarization_angle_forward", grpPolAngFwd);
            writer.WriteElement(gp + "/polarization_angle_reverse", grpPolAngRev);
            writer.WriteElement(gp + "/polarization_angle_delta", grpPolAngDelta);
        }
    }

    // Q/U polarization comparison with forward and deltas
    std::vector<double> qRev(numObservers_), uRev(numObservers_);
    std::vector<double> qDelta(numObservers_), uDelta(numObservers_);
    std::vector<double> qRelDelta(numObservers_), uRelDelta(numObservers_);
    std::vector<double> polDegFwd(numObservers_), polDegRev(numObservers_), polDegDelta(numObservers_);
    std::vector<double> polDegRelDelta(numObservers_);
    std::vector<double> polAngFwd(numObservers_), polAngRev(numObservers_), polAngDelta(numObservers_);
    for (size_t p = 0; p < numObservers_; ++p)
    {
        qRev[p] = tallies_.getObsQ(p);
        uRev[p] = tallies_.getObsU(p);
        qDelta[p] = qRev[p] - fwdQ[p];
        uDelta[p] = uRev[p] - fwdU[p];

        double qFloor = std::max(std::abs(fwdQ[p]), 1e-30);
        double uFloor = std::max(std::abs(fwdU[p]), 1e-30);
        qRelDelta[p] = qDelta[p] / qFloor;
        uRelDelta[p] = uDelta[p] / uFloor;

        double Ifwd = fwdLum[p];
        double Irev = revLum[p];
        if (Ifwd > 0.0)
        {
            polDegFwd[p] = std::sqrt(fwdQ[p] * fwdQ[p] + fwdU[p] * fwdU[p]) / Ifwd;
            polAngFwd[p] = 0.5 * std::atan2(fwdU[p], fwdQ[p]);
        }
        if (Irev > 0.0)
        {
            polDegRev[p] = std::sqrt(qRev[p] * qRev[p] + uRev[p] * uRev[p]) / Irev;
            polAngRev[p] = 0.5 * std::atan2(uRev[p], qRev[p]);
        }
        polDegDelta[p] = polDegRev[p] - polDegFwd[p];
        double degFloor = std::max(polDegFwd[p], 1e-30);
        polDegRelDelta[p] = polDegDelta[p] / degFloor;

        double angDiff = polAngRev[p] - polAngFwd[p];
        constexpr double pi = 3.141592653589793238462643383279502884;
        while (angDiff >  0.5 * pi) angDiff -= pi;
        while (angDiff < -0.5 * pi) angDiff += pi;
        polAngDelta[p] = angDiff;
    }
    writer.WriteElement(prefix + "/q_forward", fwdQ);
    writer.WriteElement(prefix + "/q_reverse", qRev);
    writer.WriteElement(prefix + "/q_delta", qDelta);
    writer.WriteElement(prefix + "/q_rel_delta", qRelDelta);
    writer.WriteElement(prefix + "/u_forward", fwdU);
    writer.WriteElement(prefix + "/u_reverse", uRev);
    writer.WriteElement(prefix + "/u_delta", uDelta);
    writer.WriteElement(prefix + "/u_rel_delta", uRelDelta);
    writer.WriteElement(prefix + "/polarization_degree_forward", polDegFwd);
    writer.WriteElement(prefix + "/polarization_degree_reverse", polDegRev);
    writer.WriteElement(prefix + "/polarization_degree_delta", polDegDelta);
    writer.WriteElement(prefix + "/polarization_degree_rel_delta", polDegRelDelta);
    writer.WriteElement(prefix + "/polarization_angle_forward", polAngFwd);
    writer.WriteElement(prefix + "/polarization_angle_reverse", polAngRev);
    writer.WriteElement(prefix + "/polarization_angle_delta", polAngDelta);

    // Reverse uncertainty arrays (launched-packet variance)
    uint64_t NL = tallies_.diagnostics().packetsLaunched;
    std::vector<double> sigmaI(numObservers_), sigmaQ(numObservers_), sigmaU(numObservers_);
    std::vector<double> sigmaINz(numObservers_), sigmaQNz(numObservers_), sigmaUNz(numObservers_);
    std::vector<double> nonzeroCount(numObservers_);
    for (size_t p = 0; p < numObservers_; ++p)
    {
        uint64_t Nnz = tallies_.getPacketCount(p);
        nonzeroCount[p] = static_cast<double>(Nnz);
        sigmaI[p] = sigmaMeanFromPacketMoments(
            tallies_.getObsI(p), tallies_.getObsI2(p), NL);
        sigmaQ[p] = sigmaMeanFromPacketMoments(
            tallies_.getObsQ(p), tallies_.observerQ2()[p], NL);
        sigmaU[p] = sigmaMeanFromPacketMoments(
            tallies_.getObsU(p), tallies_.observerU2()[p], NL);
        sigmaINz[p] = sigmaMeanFromPacketMoments(
            tallies_.getObsI(p), tallies_.getObsI2(p), Nnz);
        sigmaQNz[p] = sigmaMeanFromPacketMoments(
            tallies_.getObsQ(p), tallies_.observerQ2()[p], Nnz);
        sigmaUNz[p] = sigmaMeanFromPacketMoments(
            tallies_.getObsU(p), tallies_.observerU2()[p], Nnz);
    }
    writer.WriteElement(prefix + "/reverse_observer_luminosity_sigma_launched", sigmaI);
    writer.WriteElement(prefix + "/reverse_q_sigma_launched", sigmaQ);
    writer.WriteElement(prefix + "/reverse_u_sigma_launched", sigmaU);
    writer.WriteElement(prefix + "/reverse_observer_luminosity_sigma_nonzero", sigmaINz);
    writer.WriteElement(prefix + "/reverse_q_sigma_nonzero", sigmaQNz);
    writer.WriteElement(prefix + "/reverse_u_sigma_nonzero", sigmaUNz);
    writer.WriteElement(prefix + "/reverse_nonzero_packet_count", nonzeroCount);
    writer.WriteElement(prefix + "/reverse_launched_packet_count",
        static_cast<double>(tallies_.diagnostics().packetsLaunched));
    writer.WriteElement(prefix + "/metadata/sigma_sample_space",
        std::string(ReverseOutputStrings::SigmaSampleSpace));

    // Group-level uncertainty (per-packet variance estimator)
    auto const &grpI2 = tallies_.groupI2();
    auto const &grpQ2 = tallies_.groupQ2();
    auto const &grpU2 = tallies_.groupU2();
    auto const &totGrpI2 = tallies_.totalGroupI2();
    auto const &totGrpQ2 = tallies_.totalGroupQ2();
    auto const &totGrpU2 = tallies_.totalGroupU2();
    uint64_t Nlaunched = tallies_.diagnostics().packetsLaunched;
    if (Nlaunched > 1)
    {
        for (size_t p = 0; p < numObservers_; ++p)
        {
            std::string gsp = prefix + "/group_sigma_" + std::to_string(p);
            std::vector<double> gSigIRes(nGrp, 0.0), gSigQRes(nGrp, 0.0), gSigURes(nGrp, 0.0);
            std::vector<double> gSigITot(nGrp, 0.0), gSigQTot(nGrp, 0.0), gSigUTot(nGrp, 0.0);
            for (size_t g = 0; g < nGrp; ++g)
            {
                gSigIRes[g] = sigmaMeanFromPacketMoments(
                    revGrpI[p][g], grpI2[p][g], Nlaunched);
                gSigQRes[g] = sigmaMeanFromPacketMoments(
                    revGrpQ[p][g], grpQ2[p][g], Nlaunched);
                gSigURes[g] = sigmaMeanFromPacketMoments(
                    revGrpU[p][g], grpU2[p][g], Nlaunched);

                double totalI = revGrpI[p][g] + collI[p][g];
                double totalQ = revGrpQ[p][g] + tallies_.collapsedQ()[p][g];
                double totalU = revGrpU[p][g] + tallies_.collapsedU()[p][g];
                gSigITot[g] = sigmaMeanFromPacketMoments(
                    totalI, totGrpI2[p][g], Nlaunched);
                gSigQTot[g] = sigmaMeanFromPacketMoments(
                    totalQ, totGrpQ2[p][g], Nlaunched);
                gSigUTot[g] = sigmaMeanFromPacketMoments(
                    totalU, totGrpU2[p][g], Nlaunched);
            }
            writer.WriteElement(gsp + "/I_sigma_resolved", gSigIRes);
            writer.WriteElement(gsp + "/Q_sigma_resolved", gSigQRes);
            writer.WriteElement(gsp + "/U_sigma_resolved", gSigURes);
            writer.WriteElement(gsp + "/I_sigma_total", gSigITot);
            writer.WriteElement(gsp + "/Q_sigma_total", gSigQTot);
            writer.WriteElement(gsp + "/U_sigma_total", gSigUTot);
        }
    }

    // Collapsed PGRW comparison fields
    bool hasCollapsed = false;
    for (size_t p = 0; p < numObservers_ && !hasCollapsed; ++p)
        for (size_t g = 0; g < nGrp && !hasCollapsed; ++g)
            if (collI[p][g] != 0.0) hasCollapsed = true;
    if (hasCollapsed)
    {
        auto const &collQ = tallies_.collapsedQ();
        auto const &collU = tallies_.collapsedU();
        for (size_t p = 0; p < numObservers_; ++p)
        {
            std::string cp2 = prefix + "/collapsed_pgrw_" + std::to_string(p);
            writer.WriteElement(cp2 + "/I", collI[p]);
            writer.WriteElement(cp2 + "/Q", collQ[p]);
            writer.WriteElement(cp2 + "/U", collU[p]);

            std::vector<double> collFrac(nGrp, 0.0);
            for (size_t g = 0; g < nGrp; ++g)
            {
                double resolved = (p < revGrpI.size() && g < revGrpI[p].size())
                    ? revGrpI[p][g] : 0.0;
                double total = resolved + collI[p][g];
                if (std::abs(total) > 1e-30)
                    collFrac[g] = collI[p][g] / total;
            }
            writer.WriteElement(cp2 + "/fraction_of_reverse_I", collFrac);
        }
    }

    // DDMC diagnostic fractions
    auto const &diag = tallies_.diagnostics();
    double totalSteps = static_cast<double>(diag.ordinarySteps + diag.ddmcSteps);
    double ddmcFrac = (totalSteps > 0.0)
        ? static_cast<double>(diag.ddmcSteps) / totalSteps : 0.0;
    writer.WriteElement(prefix + "/ddmc_fraction_reverse", ddmcFrac);
    writer.WriteElement(prefix + "/reverse_ddmc_steps",
        static_cast<double>(diag.ddmcSteps));
    writer.WriteElement(prefix + "/reverse_ordinary_steps",
        static_cast<double>(diag.ordinarySteps));
    writer.WriteElement(prefix + "/reverse_time_census_count",
        static_cast<double>(diag.timeCensusCount));
    writer.WriteElement(prefix + "/reverse_thermal_sampler_fallback_count",
        static_cast<double>(diag.thermalSamplerFallbackCount));

    writer.WriteElement(prefix + "/comparison_complete", 1.0);
    comparisonWritten_ = true;
    comparisonFile_ = filename;
    comparisonGroup_ = prefix;
    writer.Close();
}
