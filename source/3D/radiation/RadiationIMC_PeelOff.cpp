#include "RadiationIMC.hpp"
#include "SphericalObserver.hpp"
#include "LorentzTransformation.hpp"
#include "IMCPolarization.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#ifdef RICH_MPI
#include "monte/utils/GhostMap.hpp"
#include <mpi.h>
#endif
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr double pi = 3.14159265358979323846;
constexpr double FACE_EPS = 1e-12;

bool findObserverSphereHit(Vector3D const& origin, Vector3D const& direction,
                           Vector3D const& center, double radiusSquared,
                           double &distanceOut)
{
    Vector3D oc = origin - center;
    double a = ScalarProd(direction, direction);
    if (a <= 0.0) return false;
    double b = 2.0 * ScalarProd(oc, direction);
    double c = ScalarProd(oc, oc) - radiusSquared;
    double disc = b * b - 4.0 * a * c;
    if (disc < 0.0) return false;

    double sqrtDisc = std::sqrt(disc);
    double t1 = (-b - sqrtDisc) / (2.0 * a);
    double t2 = (-b + sqrtDisc) / (2.0 * a);

    double tHit = -1.0;
    if (t1 > 1e-12) tHit = t1;
    else if (t2 > 1e-12) tHit = t2;
    else return false;

    distanceOut = tHit;
    return true;
}

double computeCellScale(Vector3D const& meshPoint, Vector3D const& rayPos)
{
    double d = abs(meshPoint - rayPos);
    return std::max(d, 1.0);
}

} // anonymous namespace

// --- MPI ray state packing ---
//
// Wire format: 23 contiguous doubles per ray. Any new integer-like field
// needs pack-side exactness checks (assertExactDouble) and receive-side
// target-type + context validation.
//
//  Slot  Field                Type       Range / notes
//  ----  -------------------  ---------  ----------------------------------
//   0    nObsLab.x            double     finite
//   1    nObsLab.y            double     finite
//   2    nObsLab.z            double     finite, |nObsLab|≈1
//   3    position.x           double     finite
//   4    position.y           double     finite
//   5    position.z           double     finite
//   6    remainingDist        double     finite, >= 0
//   7    tau                  double     finite, >= 0
//   8    labFrequency         double     finite, > 0
//   9    contributionPrefactor double    finite, >= 0
//  10    stokesQ              double     finite
//  11    stokesU              double     finite
//  12    polarizationInit     bool       exactly 0.0 or 1.0
//  13    eventTimeLeft        double     finite
//  14    rayId                uint64     exact-in-double (<= 2^53)
//  15    kind (PeelOffEventKind) int     [0, NumPeelOffKinds)
//  16    observerIndex        size_t     exact-in-double, < numObservers
//  17    currentLocalCell     size_t     exact-in-double, < Nreal at recv
//  18    originRank           int        [0, numRanks)
//  19    currentRank          int        intended destination rank; validated == myRank at recv
//  20    mpiHops              uint32     [0, UINT_MAX]
//  21    cellsTraversed       uint32     [0, UINT_MAX]
//  22    crossedAnyMpiBoundary bool      exactly 0.0 or 1.0

// Pack-side check: integer value must round-trip exactly through double.
static void assertExactDouble(unsigned long long v, const char* field)
{
    constexpr unsigned long long kMaxExact = 1ULL << 53;
    if (v > kMaxExact)
    {
        std::cerr << "PeelOff FATAL: " << field << "=" << v
                  << " exceeds exact double range" << std::endl;
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        throw std::runtime_error(std::string("PeelOff pack overflow: ") + field);
#endif
    }
}

void RadiationIMC::PeelOffRayState::packInto(double* buf) const
{
    static_assert(PackedDoubles == 23, "Update packInto/unpack if PackedDoubles changes");
    assertExactDouble(rayId, "rayId");
    assertExactDouble(static_cast<unsigned long long>(observerIndex), "observerIndex");
    assertExactDouble(static_cast<unsigned long long>(currentLocalCell), "currentLocalCell");

    buf[0]  = nObsLab.x;
    buf[1]  = nObsLab.y;
    buf[2]  = nObsLab.z;
    buf[3]  = position.x;
    buf[4]  = position.y;
    buf[5]  = position.z;
    buf[6]  = remainingDist;
    buf[7]  = tau;
    buf[8]  = labFrequency;
    buf[9]  = contributionPrefactor;
    buf[10] = stokesQ;
    buf[11] = stokesU;
    buf[12] = polarizationInitialized ? 1.0 : 0.0;
    buf[13] = eventTimeLeft;
    buf[14] = static_cast<double>(rayId);
    buf[15] = static_cast<double>(kind);
    buf[16] = static_cast<double>(observerIndex);
    buf[17] = static_cast<double>(currentLocalCell);
    buf[18] = static_cast<double>(originRank);
    buf[19] = static_cast<double>(currentRank);
    buf[20] = static_cast<double>(mpiHops);
    buf[21] = static_cast<double>(cellsTraversed);
    buf[22] = crossedAnyMpiBoundary ? 1.0 : 0.0;
}

// Integer fields packed as doubles are exact up to 2^53; validate integrality
// and range before casting to avoid undefined/implementation-defined behavior.
static bool validPackedUint(double v, double maxVal)
{
    return std::isfinite(v) && v >= 0.0 && v == std::floor(v) && v <= maxVal;
}
static bool validPackedInt(double v, double lo, double hi)
{
    return std::isfinite(v) && v >= lo && v <= hi && v == std::floor(v);
}

RadiationIMC::PeelOffRayState RadiationIMC::PeelOffRayState::unpack(const double* buf)
{
    PeelOffRayState s;
    s.nObsLab.x        = buf[0];
    s.nObsLab.y        = buf[1];
    s.nObsLab.z        = buf[2];
    s.position.x       = buf[3];
    s.position.y       = buf[4];
    s.position.z       = buf[5];
    s.remainingDist    = buf[6];
    s.tau              = buf[7];
    s.labFrequency     = buf[8];
    s.contributionPrefactor = buf[9];
    s.stokesQ          = buf[10];
    s.stokesU          = buf[11];
    s.polarizationInitialized = (buf[12] > 0.5);
    s.eventTimeLeft    = buf[13];

    constexpr double kMaxExactInt = 9007199254740992.0; // 2^53

    if (!std::isfinite(buf[0]) || !std::isfinite(buf[1]) || !std::isfinite(buf[2]) ||
        !std::isfinite(buf[3]) || !std::isfinite(buf[4]) || !std::isfinite(buf[5]) ||
        !std::isfinite(s.remainingDist) || s.remainingDist < 0.0 ||
        !std::isfinite(s.tau) || s.tau < 0.0 ||
        !std::isfinite(s.labFrequency) || s.labFrequency <= 0.0 ||
        !std::isfinite(s.contributionPrefactor) || s.contributionPrefactor < 0.0 ||
        !std::isfinite(s.stokesQ) || !std::isfinite(s.stokesU) ||
        !(buf[12] == 0.0 || buf[12] == 1.0) ||
        !std::isfinite(s.eventTimeLeft))
    {
        s.valid = false;
        return s;
    }

    constexpr double kIntMax = static_cast<double>(std::numeric_limits<int>::max());
    constexpr double kUintMax = static_cast<double>(std::numeric_limits<unsigned int>::max());

    if (!validPackedUint(buf[14], kMaxExactInt) ||
        !validPackedInt(buf[15], 0.0, static_cast<double>(NumPeelOffKinds - 1)) ||
        !validPackedUint(buf[16], kMaxExactInt) ||
        !validPackedUint(buf[17], kMaxExactInt) ||
        !validPackedInt(buf[18], 0.0, kIntMax) ||
        !validPackedInt(buf[19], 0.0, kIntMax) ||
        !validPackedUint(buf[20], kUintMax) ||
        !validPackedUint(buf[21], kUintMax) ||
        !(buf[22] == 0.0 || buf[22] == 1.0))
    {
        s.valid = false;
        return s;
    }

    s.rayId            = static_cast<unsigned long long>(buf[14]);
    s.kind             = static_cast<PeelOffEventKind>(static_cast<int>(buf[15]));
    s.observerIndex    = static_cast<size_t>(buf[16]);
    s.currentLocalCell = static_cast<size_t>(buf[17]);
    s.originRank       = static_cast<int>(buf[18]);
    s.currentRank      = static_cast<int>(buf[19]);
    s.mpiHops          = static_cast<unsigned int>(buf[20]);
    s.cellsTraversed   = static_cast<unsigned int>(buf[21]);
    s.crossedAnyMpiBoundary = (buf[22] > 0.5);
    s.valid            = true;
    return s;
}

// --- Face exit classifier ---
// Contract: peelOffGhostMap_ maps ghost cell indices to (ownerRank, ownerLocalIndex)
// via GetGhostMap(), which is the canonical RICH MPI cell-identity API.  The map is
// built once in the RadiationIMC constructor and remains valid for the object's
// lifetime because mesh topology is frozen during the radiation timestep (mesh
// motion and repartitioning occur between timesteps, outside RadiationIMC).

FaceExitInfo RadiationIMC::classifyFaceExit(size_t faceGlobalIdx, size_t fromCell) const
{
    FaceExitInfo info;
    auto const& nb = this->grid.GetFaceNeighbors(faceGlobalIdx);

    if (nb.first != fromCell && nb.second != fromCell)
    {
        info.kind = FaceExitKind::InvalidOrUnknown;
        return info;
    }

    size_t next = (nb.first == fromCell) ? nb.second : nb.first;
    size_t const Nreal = this->grid.GetPointNo();

    if (next < Nreal)
    {
        info.kind = FaceExitKind::LocalRealCell;
        info.nextLocalCell = next;
        return info;
    }

#ifdef RICH_MPI
    auto it = peelOffGhostMap_.find(next);
    if (it != peelOffGhostMap_.end())
    {
        if (this->grid.BoundaryFace(faceGlobalIdx))
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            std::cerr << "PeelOff FATAL: face " << faceGlobalIdx << " is both BoundaryFace "
                      << "and in ghost map (remote rank=" << it->second.first
                      << ") on rank " << rank << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        info.kind = FaceExitKind::RemoteRankBoundary;
        info.remoteRank = it->second.first;
        info.remoteLocalIndex = it->second.second;
        return info;
    }

    // Under MPI, a non-real, non-ghost neighbor that is not a BoundaryFace is
    // UnsupportedBoundary. Do NOT use IsPointOutsideBox as a fallback because
    // it checks against the local subdomain box, not the global physical domain.
    if (this->grid.BoundaryFace(faceGlobalIdx))
    {
        info.kind = FaceExitKind::PhysicalVacuumBoundary;
        return info;
    }
#else
    if (this->grid.BoundaryFace(faceGlobalIdx) || this->grid.IsPointOutsideBox(next))
    {
        info.kind = FaceExitKind::PhysicalVacuumBoundary;
        return info;
    }
#endif

    info.kind = FaceExitKind::UnsupportedBoundary;
    info.faceId = faceGlobalIdx;
    info.fromCellId = fromCell;
    info.nextCellId = next;
    return info;
}

// --- Centralized opacity helper (M1) ---

double RadiationIMC::computePeelOffRayOpacity(
    ComputationalCell3D const& cell,
    size_t localCellIndex,
    double labFrequency,
    Vector3D const& nLab,
    double& dopplerShiftOut,
    double& shiftedFreqOut) const
{
    dopplerShiftOut = 1.0;
    shiftedFreqOut = labFrequency;

    if (this->useTransportVelocities_)
    {
        double v2 = ScalarProd(cell.velocity, cell.velocity);
        double gamma = 1.0 / std::sqrt(1.0 - units::inv_clight2 * v2);
        dopplerShiftOut = gamma * (1.0 - ScalarProd(cell.velocity, nLab) * units::inv_clight);
        if (!std::isfinite(dopplerShiftOut) || dopplerShiftOut <= 0.0)
            return -1.0;
        shiftedFreqOut = labFrequency * dopplerShiftOut;
    }

    double sigmaAbs = 0.0;
    // Grey scattering: CalcScatteringOpacity is frequency-independent in the
    // current opacity models. If frequency-dependent scattering is added, this
    // call must be updated to pass the shifted frequency, matching transport.
    double sigmaScat = this->opacity->CalcScatteringOpacity(cell);

    if (this->multigroupOpacity)
    {
        shiftedFreqOut = std::clamp(shiftedFreqOut,
            ComputationalCell3D::energyBoundaries[0],
            ComputationalCell3D::energyBoundaries[ENERGY_GROUPS_NUM]);
        sigmaAbs = this->opacity->CalcAbsorptionOpacity(cell, shiftedFreqOut);
    }
    else
    {
        sigmaAbs = this->planckOpacities[localCellIndex];
    }

    return std::max(0.0, sigmaAbs + sigmaScat);
}

// --- Phase PDF helpers (M1) ---

void RadiationIMC::resetPeelOffCounters()
{
    peelOffCounters_.reset();
}

double RadiationIMC::evaluatePeelOffPhasePdf(
    PeelOffSource const& source,
    Vector3D const& nObsLab,
    double& qObserver,
    double& uObserver,
    bool& polarizationInitialized) const
{
    qObserver = 0.0;
    uObserver = 0.0;
    polarizationInitialized = false;

    switch (source.phaseMode)
    {
        case PeelOffSource::PhaseMode::Isotropic:
            // Source emission and effective scatter: isotropic in lab frame.
            // PDF = 1/(4*pi). This matches the transport sampler which draws
            // uniform random directions for source emission and reemission.
            return 1.0 / (4.0 * pi);

        case PeelOffSource::PhaseMode::ElasticScatter:
#ifdef MONTECARLO_POLARIZATION
            if (postProcess_.polarization.enabled)
            {
                struct PeelOffParticle
                {
                    Vector3D velocity;
                    double stokesQ = 0.0;
                    double stokesU = 0.0;
                    Vector3D polarizationBasis;
                    bool polarizationInitialized = false;
                };

                PeelOffParticle p;
                p.velocity = source.incomingDirectionLab;
                p.stokesQ = source.stokesQ;
                p.stokesU = source.stokesU;
                p.polarizationBasis = source.polarizationBasis;
                p.polarizationInitialized = source.polarizationInitialized;

                auto const result = IMCPolarization::EvaluateThomsonPeelOff(
                    p,
                    source.incomingDirectionLab,
                    nObsLab,
                    IMCPolarization::ObserverBasis1(nObsLab));
                if (!result.valid)
                    return 0.0;
                qObserver = result.stokesQ;
                uObserver = result.stokesU;
                polarizationInitialized = true;
                return result.phasePdf;
            }
#endif
            // Non-polarized elastic scattering is isotropic in the current codebase
            // (getNewScatterVelocity samples a uniform random direction).
            // This is a contract: if anisotropic scattering is added, both the
            // transport sampler and this PDF must be updated together.
            return 1.0 / (4.0 * pi);

        case PeelOffSource::PhaseMode::CosineLeak:
        {
            // DDMC Lambertian leak: p(mu) = mu/pi for mu > 0.
            double nNorm = abs(source.surfaceNormalLab);
            if (nNorm <= 0.0) return 0.0;
            Vector3D n = source.surfaceNormalLab * (1.0 / nNorm);
            double mu = ScalarProd(n, nObsLab);
            return (mu > 0.0) ? mu / pi : 0.0;
        }
    }
    return 0.0;
}

// --- Local ray segment tracer ---

RadiationIMC::LocalTraceOutcome RadiationIMC::continuePeelOffRayLocally(
    PeelOffRayState state) const
{
    LocalTraceOutcome outcome;
    outcome.state = state;

    double const maxTau = postProcess_.peelOff.maxTau;
    double const nudgeFraction = postProcess_.peelOff.rayNudgeFraction;
    size_t const maxSteps = postProcess_.peelOff.maxRayCells;
    size_t const Nreal = this->grid.GetPointNo();

    Vector3D const& nObsLab = state.nObsLab;
    Vector3D rayPos = state.position;
    size_t currentCell = state.currentLocalCell;
    double remainingDist = state.remainingDist;
    double tau = state.tau;
    unsigned int cellsTraversed = state.cellsTraversed;

    if (currentCell >= Nreal)
    {
        outcome.status = LocalTraceOutcome::Status::InvalidState;
        return outcome;
    }

    // Initial nudge into the current cell
    {
        double cellScale = computeCellScale(this->grid.GetMeshPoint(currentCell), rayPos);
        double nudge = std::min(nudgeFraction * cellScale, remainingDist);

        double ds, sf;
        double sigmaTotal = computePeelOffRayOpacity(
            this->cells[currentCell], currentCell,
            state.labFrequency, nObsLab, ds, sf);
        if (sigmaTotal < 0.0)
        {
            outcome.status = LocalTraceOutcome::Status::InvalidState;
            return outcome;
        }

        tau += sigmaTotal * ds * nudge;
        if (tau > maxTau)
        {
            outcome.state.tau = tau;
            outcome.status = LocalTraceOutcome::Status::TauClipped;
            return outcome;
        }

        rayPos = rayPos + nObsLab * nudge;
        remainingDist -= nudge;
    }

    bool rayComplete = (remainingDist <= FACE_EPS);

    for (size_t step = 0; step < maxSteps && remainingDist > 0.0; ++step)
    {
        if (currentCell >= Nreal)
        {
            outcome.status = LocalTraceOutcome::Status::InvalidState;
            return outcome;
        }

        cellsTraversed++;

        double ds, sf;
        double sigmaTotal = computePeelOffRayOpacity(
            this->cells[currentCell], currentCell,
            state.labFrequency, nObsLab, ds, sf);
        if (sigmaTotal < 0.0)
        {
            outcome.status = LocalTraceOutcome::Status::InvalidState;
            return outcome;
        }

        auto const& faces = this->grid.GetCellFaces(currentCell);
        auto const& normals = this->gridData.normalsOfCells[currentCell];
        auto const& onFaces = this->gridData.pointsOnFaces[currentCell];
        size_t Nfaces = faces.size();

        double minAlpha = std::numeric_limits<double>::max();
        size_t exitFaceLocalIdx = std::numeric_limits<size_t>::max();

        for (size_t fi = 0; fi < Nfaces; ++fi)
        {
            double denom = ScalarProd(normals[fi], nObsLab);
            if (std::abs(denom) < FACE_EPS)
                continue;
            double alpha = ScalarProd(onFaces[fi] - rayPos, normals[fi]) / denom;
            if (alpha > FACE_EPS && alpha < minAlpha)
            {
                minAlpha = alpha;
                exitFaceLocalIdx = fi;
            }
        }

        double segLength;
        if (exitFaceLocalIdx != std::numeric_limits<size_t>::max() &&
            minAlpha < remainingDist)
        {
            segLength = minAlpha;
        }
        else if (exitFaceLocalIdx == std::numeric_limits<size_t>::max() &&
                 remainingDist > FACE_EPS)
        {
            outcome.state.tau = tau;
            outcome.state.cellsTraversed = cellsTraversed;
            outcome.status = LocalTraceOutcome::Status::NoExitFace;
            return outcome;
        }
        else
        {
            segLength = remainingDist;
            rayComplete = true;
        }

        tau += sigmaTotal * ds * segLength;
        if (tau > maxTau)
        {
            outcome.state.tau = tau;
            outcome.state.cellsTraversed = cellsTraversed;
            outcome.status = LocalTraceOutcome::Status::TauClipped;
            return outcome;
        }

        remainingDist -= segLength;
        rayPos = rayPos + nObsLab * segLength;

        if (rayComplete)
            break;

        if (exitFaceLocalIdx != std::numeric_limits<size_t>::max())
        {
            size_t faceGlobalIdx = faces[exitFaceLocalIdx];
            FaceExitInfo exit = classifyFaceExit(faceGlobalIdx, currentCell);

            switch (exit.kind)
            {
                case FaceExitKind::LocalRealCell:
                {
                    currentCell = exit.nextLocalCell;

                    double cellScale = computeCellScale(
                        this->grid.GetMeshPoint(currentCell), rayPos);
                    double nextNudge = std::min(nudgeFraction * cellScale,
                                                remainingDist);

                    double nds, nsf;
                    double nst = computePeelOffRayOpacity(
                        this->cells[currentCell], currentCell,
                        state.labFrequency, nObsLab, nds, nsf);
                    if (nst < 0.0)
                    {
                        outcome.status = LocalTraceOutcome::Status::InvalidState;
                        return outcome;
                    }
                    tau += nst * nds * nextNudge;
                    if (tau > maxTau)
                    {
                        outcome.state.tau = tau;
                        outcome.state.cellsTraversed = cellsTraversed;
                        outcome.status = LocalTraceOutcome::Status::TauClipped;
                        return outcome;
                    }

                    rayPos = rayPos + nObsLab * nextNudge;
                    remainingDist -= nextNudge;
                    break;
                }

                case FaceExitKind::PhysicalVacuumBoundary:
                {
                    outcome.state.tau = tau;
                    outcome.state.position = rayPos;
                    outcome.state.remainingDist = remainingDist;
                    outcome.state.cellsTraversed = cellsTraversed;
                    outcome.state.currentLocalCell = currentCell;
                    outcome.status = LocalTraceOutcome::Status::CompletedAfterPhysicalVacuumExit;
                    return outcome;
                }

                case FaceExitKind::RemoteRankBoundary:
                {
                    outcome.state.tau = tau;
                    outcome.state.position = rayPos;
                    outcome.state.remainingDist = remainingDist;
                    outcome.state.cellsTraversed = cellsTraversed;
                    outcome.state.currentLocalCell = exit.remoteLocalIndex;
                    outcome.state.currentRank = exit.remoteRank;
                    outcome.remoteExit = exit;
                    outcome.status = LocalTraceOutcome::Status::NeedsRemoteContinuation;
                    return outcome;
                }

                case FaceExitKind::UnsupportedBoundary:
                case FaceExitKind::InvalidOrUnknown:
                default:
                {
                    outcome.state.tau = tau;
                    outcome.state.cellsTraversed = cellsTraversed;
                    outcome.status = LocalTraceOutcome::Status::UnsupportedBoundary;
                    return outcome;
                }
            }

            if (remainingDist <= FACE_EPS)
            {
                rayComplete = true;
                break;
            }
        }
        else
        {
            outcome.state.tau = tau;
            outcome.state.cellsTraversed = cellsTraversed;
            outcome.status = LocalTraceOutcome::Status::NoExitFace;
            return outcome;
        }
    }

    if (!rayComplete && remainingDist > FACE_EPS)
    {
        outcome.state.tau = tau;
        outcome.state.cellsTraversed = cellsTraversed;
        outcome.status = LocalTraceOutcome::Status::MaxCellsExceeded;
        return outcome;
    }

    outcome.state.tau = tau;
    outcome.state.cellsTraversed = cellsTraversed;
    outcome.state.position = rayPos;
    outcome.state.remainingDist = remainingDist;
    outcome.state.currentLocalCell = currentCell;
    outcome.status = LocalTraceOutcome::Status::CompletedAtObserver;
    return outcome;
}

// --- Pending ray queue processing ---

void RadiationIMC::processPendingPeelOffRays()
{
    using MpiPolicy = RadiationIMCPostProcessConfig::PeelOffConfig::MpiRayPolicy;

#ifdef RICH_MPI
    int localHasPending = pendingPeelOffRays_.empty() ? 0 : 1;
    int anyHasPending = 0;
    MPI_Allreduce(&localHasPending, &anyHasPending, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (!anyHasPending)
        return;

    if (postProcess_.peelOff.mpiRayPolicy == MpiPolicy::DistributedExact)
    {
        if (!observer_)
        {
            std::cerr << "PeelOff FATAL: DistributedExact requires a non-null observer on every rank"
                      << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int myRank = 0, numRanks = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

        std::vector<PeelOffRayState> active;
        std::vector<std::vector<PeelOffRayState>> initialOutgoing(
            static_cast<size_t>(numRanks));

        // Partition pending rays: local rays go to active, non-local rays
        // go directly into the outgoing exchange queue for their owning rank.
        for (auto& ray : pendingPeelOffRays_)
        {
            if (ray.currentRank == myRank)
                active.push_back(std::move(ray));
            else if (ray.currentRank >= 0 && ray.currentRank < numRanks)
                initialOutgoing[static_cast<size_t>(ray.currentRank)].push_back(std::move(ray));
            else
            {
                size_t k = static_cast<size_t>(ray.kind);
                if (k < NumPeelOffKinds)
                {
                    peelOffCounters_.lostRemoteCell[k]++;
                    peelOffCounters_.rayFailed[k]++;
                    if (ray.crossedAnyMpiBoundary)
                        peelOffCounters_.raysCrossedMpiBoundary[k]++;
                }
            }
        }
        pendingPeelOffRays_.clear();

        size_t const maxExchangeRounds = postProcess_.peelOff.maxDistributedExchangeRounds;

        // Rays that complete on a remote rank call observer_->recordPeelOff()
        // locally on that rank. The tallies are globally reduced by
        // SphericalObserver::mpiReduceToRank0(), which is called after
        // RadiationIMC::postStep() by the run driver (e.g. main.cpp).
        for (size_t iter = 0; iter < maxExchangeRounds; ++iter)
        {
            std::vector<std::vector<PeelOffRayState>> outgoingByRank(
                static_cast<size_t>(numRanks));

            // On the first iteration, merge any non-local rays that were
            // initially destined for other ranks into the outgoing queue.
            if (iter == 0)
            {
                for (int r = 0; r < numRanks; ++r)
                {
                    auto& dst = outgoingByRank[static_cast<size_t>(r)];
                    auto& src = initialOutgoing[static_cast<size_t>(r)];
                    dst.insert(dst.end(),
                               std::make_move_iterator(src.begin()),
                               std::make_move_iterator(src.end()));
                    src.clear();
                }
            }

            for (auto& ray : active)
            {
                LocalTraceOutcome out = continuePeelOffRayLocally(ray);
                size_t const kind = static_cast<size_t>(out.state.kind);
                if (kind >= NumPeelOffKinds)
                {
                    std::cerr << "PeelOff FATAL: invalid event kind " << kind
                              << " in distributed ray on rank " << myRank
                              << ", rayId=" << out.state.rayId << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                switch (out.status)
                {
                    case LocalTraceOutcome::Status::CompletedAtObserver:
                    case LocalTraceOutcome::Status::CompletedAfterPhysicalVacuumExit:
                    {
                        if (out.status == LocalTraceOutcome::Status::CompletedAfterPhysicalVacuumExit)
                            peelOffCounters_.physicalVacuumExits[kind]++;

                        peelOffCounters_.raysCompleted[kind]++;
                        double contribution = out.state.contributionPrefactor
                                            * std::exp(-out.state.tau);
                        if (recordPeelOffContribution(out.state, contribution))
                            peelOffCounters_.recorded[kind]++;
                        if (out.state.crossedAnyMpiBoundary)
                            peelOffCounters_.raysCrossedMpiBoundary[kind]++;
                        break;
                    }

                    case LocalTraceOutcome::Status::NeedsRemoteContinuation:
                    {
                        out.state.crossedAnyMpiBoundary = true;
                        out.state.mpiHops++;
                        peelOffCounters_.mpiBoundaryCrossings[kind]++;
                        int destRank = out.remoteExit.remoteRank;
                        if (destRank >= 0 && destRank < numRanks)
                        {
                            out.state.currentRank = destRank;
                            outgoingByRank[static_cast<size_t>(destRank)].push_back(out.state);
                        }
                        else
                        {
                            peelOffCounters_.lostRemoteCell[kind]++;
                            peelOffCounters_.rayFailed[kind]++;
                            peelOffCounters_.raysCrossedMpiBoundary[kind]++;
                        }
                        break;
                    }

                    case LocalTraceOutcome::Status::TauClipped:
                        peelOffCounters_.tauClipped[kind]++;
                        peelOffCounters_.rayFailed[kind]++;
                        if (out.state.crossedAnyMpiBoundary)
                            peelOffCounters_.raysCrossedMpiBoundary[kind]++;
                        break;

                    case LocalTraceOutcome::Status::NoExitFace:
                        peelOffCounters_.noExitFace[kind]++;
                        peelOffCounters_.rayFailed[kind]++;
                        if (out.state.crossedAnyMpiBoundary)
                            peelOffCounters_.raysCrossedMpiBoundary[kind]++;
                        break;

                    case LocalTraceOutcome::Status::MaxCellsExceeded:
                        peelOffCounters_.maxCellsExceeded[kind]++;
                        peelOffCounters_.rayFailed[kind]++;
                        if (out.state.crossedAnyMpiBoundary)
                            peelOffCounters_.raysCrossedMpiBoundary[kind]++;
                        break;

                    case LocalTraceOutcome::Status::UnsupportedBoundary:
                        peelOffCounters_.unsupportedBoundary[kind]++;
                        peelOffCounters_.rayFailed[kind]++;
                        if (out.state.crossedAnyMpiBoundary)
                            peelOffCounters_.raysCrossedMpiBoundary[kind]++;
                        break;

                    case LocalTraceOutcome::Status::InvalidState:
                    default:
                        peelOffCounters_.invalidState[kind]++;
                        peelOffCounters_.rayFailed[kind]++;
                        if (out.state.crossedAnyMpiBoundary)
                            peelOffCounters_.raysCrossedMpiBoundary[kind]++;
                        break;
                }
            }

            // Exchange ray states via MPI_Alltoallv
            constexpr size_t P = PeelOffRayState::PackedDoubles;

            std::vector<int> sendCounts(static_cast<size_t>(numRanks), 0);
            std::vector<int> sendDispls(static_cast<size_t>(numRanks), 0);

            size_t totalSend = 0;
            for (int r = 0; r < numRanks; ++r)
            {
                size_t nDoubles = outgoingByRank[static_cast<size_t>(r)].size() * P;
                if (nDoubles > static_cast<size_t>(INT_MAX))
                {
                    std::cerr << "PeelOff FATAL: send count exceeds INT_MAX on rank "
                              << myRank << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                sendCounts[static_cast<size_t>(r)] = static_cast<int>(nDoubles);
                totalSend += outgoingByRank[static_cast<size_t>(r)].size();
            }
            {
                long long cumSendDispl = 0;
                for (int r = 1; r < numRanks; ++r)
                {
                    cumSendDispl += sendCounts[static_cast<size_t>(r - 1)];
                    if (cumSendDispl > INT_MAX)
                    {
                        std::cerr << "PeelOff FATAL: cumulative send displacement exceeds INT_MAX on rank "
                                  << myRank << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    sendDispls[static_cast<size_t>(r)] = static_cast<int>(cumSendDispl);
                }
            }

            std::vector<double> sendBuf(totalSend * P);
            size_t offset = 0;
            for (int r = 0; r < numRanks; ++r)
            {
                for (auto const& ray : outgoingByRank[static_cast<size_t>(r)])
                {
                    ray.packInto(&sendBuf[offset]);
                    offset += P;
                }
            }

            std::vector<int> recvCounts(static_cast<size_t>(numRanks), 0);
            MPI_Alltoall(sendCounts.data(), 1, MPI_INT,
                         recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

            std::vector<int> recvDispls(static_cast<size_t>(numRanks), 0);
            {
                long long cumRecvDispl = 0;
                for (int r = 1; r < numRanks; ++r)
                {
                    cumRecvDispl += recvCounts[static_cast<size_t>(r - 1)];
                    if (cumRecvDispl > INT_MAX)
                    {
                        std::cerr << "PeelOff FATAL: cumulative recv displacement exceeds INT_MAX on rank "
                                  << myRank << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    recvDispls[static_cast<size_t>(r)] = static_cast<int>(cumRecvDispl);
                }
            }

            long long totalRecvDoublesLL = 0;
            for (int r = 0; r < numRanks; ++r)
            {
                if (recvCounts[static_cast<size_t>(r)] % static_cast<int>(P) != 0)
                {
                    std::cerr << "PeelOff FATAL: received count not a multiple of PackedDoubles"
                              << " from rank " << r << " on rank " << myRank << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                totalRecvDoublesLL += recvCounts[static_cast<size_t>(r)];
            }
            if (totalRecvDoublesLL > INT_MAX)
            {
                std::cerr << "PeelOff FATAL: total receive count exceeds INT_MAX on rank "
                          << myRank << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            int totalRecvDoubles = static_cast<int>(totalRecvDoublesLL);

            std::vector<double> recvBuf(static_cast<size_t>(totalRecvDoubles));
            MPI_Alltoallv(sendBuf.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                          recvBuf.data(), recvCounts.data(), recvDispls.data(), MPI_DOUBLE,
                          MPI_COMM_WORLD);

            size_t const Nreal = this->grid.GetPointNo();
            size_t numRecvRays = static_cast<size_t>(totalRecvDoubles) / P;
            active.clear();
            active.reserve(numRecvRays);
            for (size_t i = 0; i < numRecvRays; ++i)
            {
                PeelOffRayState rs = PeelOffRayState::unpack(&recvBuf[i * P]);
                if (!rs.valid ||
                    rs.originRank < 0 || rs.originRank >= numRanks ||
                    rs.observerIndex >= observer_->getNumObservers() ||
                    rs.currentRank != myRank)
                {
                    std::cerr << "PeelOff FATAL: invalid ray packet on rank " << myRank
                              << " round " << iter << " pkt " << i
                              << " valid=" << rs.valid
                              << " originRank=" << rs.originRank
                              << " currentRank=" << rs.currentRank
                              << " observerIdx=" << rs.observerIndex
                              << " numObs=" << observer_->getNumObservers()
                              << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if (rs.currentLocalCell < Nreal)
                    active.push_back(rs);
                else
                {
                    size_t k = static_cast<size_t>(rs.kind);
                    if (k < NumPeelOffKinds)
                    {
                        peelOffCounters_.lostRemoteCell[k]++;
                        peelOffCounters_.rayFailed[k]++;
                        if (rs.crossedAnyMpiBoundary)
                            peelOffCounters_.raysCrossedMpiBoundary[k]++;
                    }
                    std::cerr << "PeelOff: lostRemoteCell on rank " << myRank
                              << " round " << iter
                              << " rayId=" << rs.rayId
                              << " originRank=" << rs.originRank
                              << " cell=" << rs.currentLocalCell
                              << " Nreal=" << Nreal << std::endl;
                }
            }

            unsigned long long localWork = active.size();
            unsigned long long globalWork = 0;
            MPI_Allreduce(&localWork, &globalWork, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            if (globalWork == 0) break;
        }

        if (!active.empty())
        {
            for (auto const& ray : active)
            {
                size_t kind = static_cast<size_t>(ray.kind);
                if (kind < NumPeelOffKinds)
                {
                    peelOffCounters_.distributedExchangeLimitExceeded[kind]++;
                    peelOffCounters_.rayFailed[kind]++;
                    if (ray.crossedAnyMpiBoundary)
                        peelOffCounters_.raysCrossedMpiBoundary[kind]++;
                }
            }
            std::cerr << "PeelOff FATAL: DistributedExact exchange limit reached with "
                      << active.size() << " rays still active on rank " << myRank
                      << " after " << maxExchangeRounds << " exchange rounds.";
            if (!active.empty())
            {
                auto const& r0 = active.front();
                std::cerr << " First ray: id=" << r0.rayId
                          << " originRank=" << r0.originRank
                          << " currentRank=" << r0.currentRank
                          << " mpiHops=" << r0.mpiHops
                          << " kind=" << static_cast<int>(r0.kind)
                          << " currentLocalCell=" << r0.currentLocalCell
                          << " remainingDist=" << r0.remainingDist
                          << " tau=" << r0.tau
                          << " cellsTraversed=" << r0.cellsTraversed;
            }
            std::cerr << " Increase maxDistributedExchangeRounds or investigate mesh/partition."
                      << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        return;
    }
#endif

    // Non-distributed mode: trace all pending rays locally
    if (pendingPeelOffRays_.empty())
        return;
    for (auto& ray : pendingPeelOffRays_)
    {
        size_t const kind = static_cast<size_t>(ray.kind);
        if (kind >= NumPeelOffKinds)
        {
            peelOffCounters_.invalidState[0]++;
            peelOffCounters_.rayFailed[0]++;
            continue;
        }

        LocalTraceOutcome out = continuePeelOffRayLocally(ray);

        switch (out.status)
        {
            case LocalTraceOutcome::Status::CompletedAtObserver:
            case LocalTraceOutcome::Status::CompletedAfterPhysicalVacuumExit:
            {
                if (out.status == LocalTraceOutcome::Status::CompletedAfterPhysicalVacuumExit)
                    peelOffCounters_.physicalVacuumExits[kind]++;

                peelOffCounters_.raysCompleted[kind]++;
                double contribution = out.state.contributionPrefactor
                                    * std::exp(-out.state.tau);
                if (recordPeelOffContribution(out.state, contribution))
                    peelOffCounters_.recorded[kind]++;
                break;
            }

            case LocalTraceOutcome::Status::NeedsRemoteContinuation:
            {
                // Non-distributed modes: apply legacy policy
                peelOffCounters_.mpiBoundaryCrossings[kind]++;
                peelOffCounters_.raysCrossedMpiBoundary[kind]++;

#ifdef RICH_MPI
                if (postProcess_.peelOff.mpiRayPolicy == MpiPolicy::LocalConservativeVacuum)
                {
                    peelOffCounters_.raysCompleted[kind]++;
                    double contribution = out.state.contributionPrefactor
                                        * std::exp(-out.state.tau);
                    if (recordPeelOffContribution(out.state, contribution))
                        peelOffCounters_.recorded[kind]++;
                }
                else
#endif
                {
                    peelOffCounters_.mpiBoundaryRejected[kind]++;
                    peelOffCounters_.rayFailed[kind]++;
                }
                break;
            }

            case LocalTraceOutcome::Status::TauClipped:
                peelOffCounters_.tauClipped[kind]++;
                peelOffCounters_.rayFailed[kind]++;
                break;

            case LocalTraceOutcome::Status::NoExitFace:
                peelOffCounters_.noExitFace[kind]++;
                peelOffCounters_.rayFailed[kind]++;
                break;

            case LocalTraceOutcome::Status::MaxCellsExceeded:
                peelOffCounters_.maxCellsExceeded[kind]++;
                peelOffCounters_.rayFailed[kind]++;
                break;

            case LocalTraceOutcome::Status::UnsupportedBoundary:
                peelOffCounters_.unsupportedBoundary[kind]++;
                peelOffCounters_.rayFailed[kind]++;
                break;

            case LocalTraceOutcome::Status::InvalidState:
            default:
                peelOffCounters_.invalidState[kind]++;
                peelOffCounters_.rayFailed[kind]++;
                break;
        }
    }
    pendingPeelOffRays_.clear();
}

// --- Enqueue + record logic ---

bool RadiationIMC::recordPeelOffContribution(PeelOffRayState const& ray,
                                              double contribution)
{
    if (!observer_ || !(contribution > 0.0) || !std::isfinite(contribution))
        return false;
#ifdef MONTECARLO_POLARIZATION
    if (ray.polarizationInitialized)
    {
        return observer_->recordPeelOff(ray.observerIndex, contribution,
                                        ray.labFrequency, ray.stokesQ,
                                        ray.stokesU, ray.kind);
    }
#endif
    return observer_->recordPeelOff(ray.observerIndex, contribution,
                                    ray.labFrequency, ray.kind);
}

void RadiationIMC::traceOrQueuePeelOffRay(PeelOffRayState ray)
{
    size_t const kind = static_cast<size_t>(ray.kind);
    if (kind >= NumPeelOffKinds)
        return;

    peelOffCounters_.raysStarted[kind]++;

#ifdef RICH_MPI
    int myRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    if (ray.currentRank != myRank)
    {
        pendingPeelOffRays_.push_back(ray);
        return;
    }
#endif

    LocalTraceOutcome out = continuePeelOffRayLocally(ray);
    switch (out.status)
    {
        case LocalTraceOutcome::Status::CompletedAtObserver:
        case LocalTraceOutcome::Status::CompletedAfterPhysicalVacuumExit:
        {
            if (out.status == LocalTraceOutcome::Status::CompletedAfterPhysicalVacuumExit)
                peelOffCounters_.physicalVacuumExits[kind]++;
            peelOffCounters_.raysCompleted[kind]++;
            double const contribution = out.state.contributionPrefactor
                                      * std::exp(-out.state.tau);
            if (recordPeelOffContribution(out.state, contribution))
                peelOffCounters_.recorded[kind]++;
            if (out.state.crossedAnyMpiBoundary)
                peelOffCounters_.raysCrossedMpiBoundary[kind]++;
            break;
        }

        case LocalTraceOutcome::Status::NeedsRemoteContinuation:
#ifdef RICH_MPI
        {
            out.state.crossedAnyMpiBoundary = true;
            out.state.mpiHops++;
            peelOffCounters_.mpiBoundaryCrossings[kind]++;
            out.state.currentRank = out.remoteExit.remoteRank;
            out.state.currentLocalCell = out.remoteExit.remoteLocalIndex;
            pendingPeelOffRays_.push_back(out.state);
            break;
        }
#else
            peelOffCounters_.unsupportedBoundary[kind]++;
            peelOffCounters_.rayFailed[kind]++;
            break;
#endif

        case LocalTraceOutcome::Status::TauClipped:
            peelOffCounters_.tauClipped[kind]++;
            peelOffCounters_.rayFailed[kind]++;
            break;
        case LocalTraceOutcome::Status::TimeRejected:
            peelOffCounters_.timeRejected[kind]++;
            peelOffCounters_.rayFailed[kind]++;
            break;
        case LocalTraceOutcome::Status::NoExitFace:
            peelOffCounters_.noExitFace[kind]++;
            peelOffCounters_.rayFailed[kind]++;
            break;
        case LocalTraceOutcome::Status::MaxCellsExceeded:
            peelOffCounters_.maxCellsExceeded[kind]++;
            peelOffCounters_.rayFailed[kind]++;
            break;
        case LocalTraceOutcome::Status::UnsupportedBoundary:
            peelOffCounters_.unsupportedBoundary[kind]++;
            peelOffCounters_.rayFailed[kind]++;
            break;
        case LocalTraceOutcome::Status::InvalidState:
        default:
            peelOffCounters_.invalidState[kind]++;
            peelOffCounters_.rayFailed[kind]++;
            break;
    }
}

void RadiationIMC::maybeRecordPeelOff(PeelOffSource const& source)
{
    if (!postProcess_.peelOff.enabled || !observer_)
        return;
    if (source.labWeight <= 0.0 || !std::isfinite(source.labWeight))
        return;
    if (source.labFrequency <= 0.0 || !std::isfinite(source.labFrequency))
        return;
    if (!std::isfinite(source.sourceLocation.x) ||
        !std::isfinite(source.sourceLocation.y) ||
        !std::isfinite(source.sourceLocation.z))
        return;

    size_t const kind = static_cast<size_t>(source.kind);
    if (kind >= NumPeelOffKinds)
        return;

    auto const& directions = observer_->getDirections();
    auto const& solidAngles = observer_->getObserverSolidAngles();
    double const obsRadiusSq = observer_->getRadius() * observer_->getRadius();
    Vector3D const obsCenter = observer_->getCenter();

    int myRank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
#endif

    for (size_t j = 0; j < observer_->getNumObservers(); ++j)
    {
        Vector3D const& nObsLab = directions[j];
        peelOffCounters_.directionsConsidered[kind]++;

        double qObserver = 0.0;
        double uObserver = 0.0;
        bool polarizationInitialized = false;
        double phase = evaluatePeelOffPhasePdf(
            source, nObsLab, qObserver, uObserver, polarizationInitialized);
        if (!(phase > 0.0) || !std::isfinite(phase))
        {
            peelOffCounters_.phaseRejected[kind]++;
            continue;
        }
        peelOffCounters_.phaseAccepted[kind]++;

        // Check observer sphere hit
        double distanceToObserver = 0.0;
        if (!findObserverSphereHit(source.sourceLocation, nObsLab,
                                   obsCenter, obsRadiusSq, distanceToObserver))
        {
            peelOffCounters_.observerMissed[kind]++;
            continue;
        }

        // Time rejection check
        if (source.eventTimeLeft >= 0.0)
        {
            double const travelTime = distanceToObserver / units::clight;
            double const tol = 1e-12 * std::max(1.0, source.eventTimeLeft);
            if (travelTime > source.eventTimeLeft + tol)
            {
                peelOffCounters_.timeRejected[kind]++;
                continue;
            }
        }

        // Build ray state
        PeelOffRayState ray;
        ray.rayId = nextPeelOffRayId_++;
        ray.kind = source.kind;
        ray.observerIndex = j;
        ray.nObsLab = nObsLab;
        ray.labFrequency = source.labFrequency;
        ray.contributionPrefactor = source.labWeight * phase * solidAngles[j];
        ray.stokesQ = qObserver;
        ray.stokesU = uObserver;
        ray.polarizationInitialized = polarizationInitialized;
        ray.eventTimeLeft = source.eventTimeLeft;
        ray.originRank = myRank;
        ray.currentRank = myRank;

        bool rayHandledBySource = false;

        switch (source.startKind)
        {
            case PeelOffSource::StartKind::LocalCellPoint:
                ray.position = source.sourceLocation;
                ray.remainingDist = distanceToObserver;
                ray.currentLocalCell = source.sourceCellIndex;
                break;

            case PeelOffSource::StartKind::PhysicalVacuumBoundary:
                // Source on physical boundary: no more mesh opacity.
                // Record directly with tau=0.
                peelOffCounters_.raysStarted[kind]++;
                peelOffCounters_.raysCompleted[kind]++;
                peelOffCounters_.physicalVacuumExits[kind]++;
                {
                    double contribution = ray.contributionPrefactor;
                    if (recordPeelOffContribution(ray, contribution))
                        peelOffCounters_.recorded[kind]++;
                }
                rayHandledBySource = true;
                break;

            case PeelOffSource::StartKind::RemoteBoundaryFace:
            {
#ifdef RICH_MPI
                using MpiPolicy = RadiationIMCPostProcessConfig::PeelOffConfig::MpiRayPolicy;
                if (postProcess_.peelOff.mpiRayPolicy == MpiPolicy::DistributedExact)
                {
                    ray.position = source.sourceLocation;
                    ray.remainingDist = distanceToObserver;
                    ray.currentLocalCell = source.startExit.remoteLocalIndex;
                    ray.currentRank = source.startExit.remoteRank;
                    ray.crossedAnyMpiBoundary = true;
                    ray.mpiHops = 1;
                    peelOffCounters_.mpiBoundaryCrossings[kind]++;
                }
                else if (postProcess_.peelOff.mpiRayPolicy == MpiPolicy::LocalConservativeVacuum)
                {
                    peelOffCounters_.raysStarted[kind]++;
                    peelOffCounters_.raysCompleted[kind]++;
                    peelOffCounters_.mpiBoundaryCrossings[kind]++;
                    peelOffCounters_.raysCrossedMpiBoundary[kind]++;
                    double contribution = ray.contributionPrefactor;
                    if (recordPeelOffContribution(ray, contribution))
                        peelOffCounters_.recorded[kind]++;
                    rayHandledBySource = true;
                }
                else
                {
                    peelOffCounters_.raysStarted[kind]++;
                    peelOffCounters_.mpiBoundaryCrossings[kind]++;
                    peelOffCounters_.raysCrossedMpiBoundary[kind]++;
                    peelOffCounters_.mpiBoundaryRejected[kind]++;
                    peelOffCounters_.rayFailed[kind]++;
                    rayHandledBySource = true;
                }
#else
                ray.position = source.sourceLocation;
                ray.remainingDist = distanceToObserver;
                ray.currentLocalCell = source.sourceCellIndex;
#endif
                break;
            }

            default:
                peelOffCounters_.raysStarted[kind]++;
                peelOffCounters_.invalidState[kind]++;
                peelOffCounters_.rayFailed[kind]++;
                rayHandledBySource = true;
                break;
        }

        if (rayHandledBySource)
            continue;

        traceOrQueuePeelOffRay(ray);
    }
}
