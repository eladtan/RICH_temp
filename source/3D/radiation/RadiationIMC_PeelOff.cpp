#include "RadiationIMC.hpp"
#include "SphericalObserver.hpp"
#include "LorentzTransformation.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

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

void RadiationIMC::maybeRecordPeelOffIsotropic(
    size_t sourceCellIndex,
    Vector3D const& sourceLocation,
    double labFrequency,
    double labWeight,
    double eventTimeLeft,
    PeelOffEventKind /*kind*/)
{
    if (!postProcess_.peelOff.enabled || !observer_)
        return;
    if (!postProcess_.peelOff.sourceEmission)
        return;
    if (labWeight <= 0.0)
        return;

    double const maxTau = postProcess_.peelOff.maxTau;
    double const nudgeFraction = postProcess_.peelOff.rayNudgeFraction;
    size_t const numObs = observer_->getNumObservers();
    auto const& directions = observer_->getDirections();
    auto const& solidAngles = observer_->getObserverSolidAngles();
    Vector3D const obsCenter = observer_->getCenter();
    double const obsRadius = observer_->getRadius();
    double const obsRadiusSq = obsRadius * obsRadius;
    size_t const Nreal = this->grid.GetPointNo();
    double const invFourPi = 1.0 / (4.0 * pi);
    bool const useVelocities = this->useTransportVelocities_;

    for (size_t j = 0; j < numObs; ++j) {
        Vector3D const& nObs = directions[j];

        double distanceToSphere = 0.0;
        if (!findObserverSphereHit(sourceLocation, nObs, obsCenter, obsRadiusSq, distanceToSphere))
            continue;

        peelOffAttemptedCount_++;

        if (eventTimeLeft >= 0.0) {
            double const travelTime = distanceToSphere / units::clight;
            double const tol = 1e-12 * std::max(1.0, eventTimeLeft);
            if (travelTime > eventTimeLeft + tol) {
                peelOffTimeRejectedCount_++;
                continue;
            }
        }

        double tau = 0.0;
        double cellScale = computeCellScale(this->grid.GetMeshPoint(sourceCellIndex), sourceLocation);
        double nudge = nudgeFraction * cellScale;
        Vector3D rayPos = sourceLocation + nObs * nudge;
        size_t currentCell = sourceCellIndex;
        double remainingDist = distanceToSphere - nudge;

        bool rayComplete = (remainingDist <= FACE_EPS);
        size_t maxSteps = Nreal + 100;
        for (size_t step = 0; step < maxSteps && remainingDist > 0.0; ++step) {
            if (currentCell >= Nreal) {
                rayComplete = true;
                break;
            }

            ComputationalCell3D const& cell = this->cells[currentCell];

            double dopplerShift = 1.0;
            double shiftedFrequency = labFrequency;
            if (useVelocities) {
                double v2 = ScalarProd(cell.velocity, cell.velocity);
                double gamma = 1.0 / std::sqrt(1.0 - units::inv_clight2 * v2);
                dopplerShift = gamma * (1.0 - ScalarProd(cell.velocity, nObs) * units::inv_clight);
                if (!std::isfinite(dopplerShift) || dopplerShift <= 0.0) {
                    peelOffRayFailedCount_++;
                    break;
                }
                shiftedFrequency = labFrequency * dopplerShift;
            }

            double sigmaAbs = 0.0;
            double sigmaScat = this->opacity->CalcScatteringOpacity(cell);
            if (this->multigroupOpacity) {
                shiftedFrequency = std::clamp(shiftedFrequency,
                    ComputationalCell3D::energyBoundaries[0],
                    ComputationalCell3D::energyBoundaries[ENERGY_GROUPS_NUM]);
                sigmaAbs = this->opacity->CalcAbsorptionOpacity(cell, shiftedFrequency);
            } else {
                sigmaAbs = this->planckOpacities[currentCell];
            }
            double sigmaTotal = std::max(0.0, sigmaAbs + sigmaScat);

            auto const& faces = this->grid.GetCellFaces(currentCell);
            auto const& normals = this->gridData.normalsOfCells[currentCell];
            auto const& onFaces = this->gridData.pointsOnFaces[currentCell];
            size_t Nfaces = faces.size();

            double minAlpha = std::numeric_limits<double>::max();
            size_t exitFaceIdx = std::numeric_limits<size_t>::max();

            for (size_t fi = 0; fi < Nfaces; ++fi) {
                double denom = ScalarProd(normals[fi], nObs);
                if (std::abs(denom) < FACE_EPS)
                    continue;
                double alpha = ScalarProd(onFaces[fi] - rayPos, normals[fi]) / denom;
                if (alpha > FACE_EPS && alpha < minAlpha) {
                    minAlpha = alpha;
                    exitFaceIdx = fi;
                }
            }

            double segLength;
            if (exitFaceIdx != std::numeric_limits<size_t>::max() && minAlpha < remainingDist) {
                segLength = minAlpha;
            } else if (exitFaceIdx == std::numeric_limits<size_t>::max() && remainingDist > FACE_EPS) {
                peelOffRayFailedCount_++;
                break;
            } else {
                segLength = remainingDist;
                rayComplete = true;
            }

            tau += sigmaTotal * dopplerShift * segLength;
            if (tau > maxTau) {
                peelOffTauClippedCount_++;
                break;
            }

            remainingDist -= segLength;
            rayPos = rayPos + nObs * segLength;

            if (rayComplete)
                break;

            if (exitFaceIdx != std::numeric_limits<size_t>::max()) {
                size_t faceGlobalIdx = faces[exitFaceIdx];
                auto const& neighbors = this->grid.GetFaceNeighbors(faceGlobalIdx);
                size_t nextCell = (neighbors.first == currentCell) ? neighbors.second : neighbors.first;
                currentCell = nextCell;

                double nextCellScale = (currentCell < Nreal)
                    ? computeCellScale(this->grid.GetMeshPoint(currentCell), rayPos)
                    : 1.0;
                double nextNudge = nudgeFraction * nextCellScale;
                rayPos = rayPos + nObs * nextNudge;
                remainingDist -= nextNudge;
                if (remainingDist <= FACE_EPS) {
                    rayComplete = true;
                    break;
                }
            } else {
                peelOffRayFailedCount_++;
                break;
            }
        }

        if (!rayComplete)
            continue;

        if (tau > maxTau)
            continue;

        double contrib = labWeight * solidAngles[j] * invFourPi * std::exp(-tau);
        observer_->recordPeelOff(j, contrib, labFrequency);
        peelOffRecordedCount_++;
    }
}
