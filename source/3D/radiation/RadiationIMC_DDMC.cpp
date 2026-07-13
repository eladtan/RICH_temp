#include "RadiationIMC.hpp"
#include "DDMCWollaegerInterface.hpp"
#include "SphericalObserver.hpp"
#include "IMCPolarization.hpp"
#include "3D/tessellation/utils/RandomInCell.hpp"
#include "Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#ifdef RICH_MPI
#include "mpi/mpi_commands.hpp"
#endif

/*
 * DDMC references
 *
 * Frequency-dependent grey-band DDMC:
 *   J. D. Densmore, K. G. Thompson, and T. J. Urbatsch,
 *   J. Comput. Phys. 231, 6924-6934 (2012),
 *   DOI 10.1016/j.jcp.2012.06.020.
 *
 * Moving-medium IMC-DDMC interface and G_U correction:
 *   R. T. Wollaeger et al., ApJS 209, 36 (2013),
 *   DOI 10.1088/0067-0049/209/2/36, arXiv:1306.5700.
 *
 * RICH retains the Densmore contiguous low-frequency DDMC interval.
 * Compton and other inelastic physical scattering are deliberately not
 * supported while DDMC is active.
 */

namespace {
    inline void ClampFrequencyToBoundsDDMC(double &frequency)
    {
        frequency = std::clamp(frequency,
            ComputationalCell3D::energyBoundaries[0],
            ComputationalCell3D::energyBoundaries[ENERGY_GROUPS_NUM]);
    }

    inline double PositiveRandom(double xi)
    {
        return std::max(xi, std::numeric_limits<double>::min());
    }

    inline double CellGamma(Vector3D const &v)
    {
        double const beta2 = ScalarProd(v, v) * units::inv_clight2;
        if(beta2 <= 0.0)
            return 1.0;
        if(beta2 >= 1.0)
            return std::numeric_limits<double>::infinity();
        return 1.0 / std::sqrt(1.0 - beta2);
    }

    inline bool SolveSymmetric3x3(std::array<double, 6> const &m,
                                  Vector3D const &b,
                                  Vector3D &x)
    {
        double const a00 = m[0];
        double const a01 = m[1];
        double const a02 = m[2];
        double const a11 = m[3];
        double const a12 = m[4];
        double const a22 = m[5];

        double const c00 = a11 * a22 - a12 * a12;
        double const c01 = a02 * a12 - a01 * a22;
        double const c02 = a01 * a12 - a02 * a11;
        double const c11 = a00 * a22 - a02 * a02;
        double const c12 = a01 * a02 - a00 * a12;
        double const c22 = a00 * a11 - a01 * a01;
        double const det = a00 * c00 + a01 * c01 + a02 * c02;

        double scale = std::max(std::abs(a00), std::abs(a01));
        scale = std::max(scale, std::abs(a02));
        scale = std::max(scale, std::abs(a11));
        scale = std::max(scale, std::abs(a12));
        scale = std::max(scale, std::abs(a22));
        scale = std::max(scale, 1.0);
        if(!std::isfinite(det) || std::abs(det) < 1e-24 * scale * scale * scale)
            return false;

        x.x = (c00 * b.x + c01 * b.y + c02 * b.z) / det;
        x.y = (c01 * b.x + c11 * b.y + c12 * b.z) / det;
        x.z = (c02 * b.x + c12 * b.y + c22 * b.z) / det;
        return std::isfinite(x.x) && std::isfinite(x.y) && std::isfinite(x.z);
    }

    inline double Percentile(std::vector<double> values, double q)
    {
        if(values.empty())
            return 0.0;
        q = std::clamp(q, 0.0, 1.0);
        size_t const idx = std::min(values.size() - 1,
            static_cast<size_t>(std::floor(q * static_cast<double>(values.size() - 1))));
        std::nth_element(values.begin(), values.begin() + idx, values.end());
        return values[idx];
    }

    constexpr size_t DDMC_WEIGHT_RATIO_SAMPLE_LIMIT = 200000;

    Vector3D SampleHemisphereDirection(Vector3D normal, double mu,
                                       double phi)
    {
        normal = normalize(normal);
        Vector3D helper = (std::abs(normal.x) < 0.9)
            ? Vector3D(1.0, 0.0, 0.0)
            : Vector3D(0.0, 1.0, 0.0);
        Vector3D e1 = normalize(helper - ScalarProd(helper, normal) * normal);
        Vector3D e2 = normalize(CrossProduct(normal, e1));
        double const sine = std::sqrt(std::max(0.0, 1.0 - mu * mu));
        return mu * normal + sine * std::cos(phi) * e1
             + sine * std::sin(phi) * e2;
    }

    Vector3D SampleIsotropicDirection(double xiMu, double xiPhi)
    {
        constexpr double pi = 3.14159265358979323846;
        double const mu = 2.0 * std::clamp(xiMu, 0.0, 1.0) - 1.0;
        double const phi = 2.0 * pi * std::clamp(xiPhi, 0.0, 1.0);
        double const sine = std::sqrt(std::max(0.0, 1.0 - mu * mu));
        return Vector3D(sine * std::cos(phi),
                        sine * std::sin(phi),
                        mu);
    }

    double PlanckBandMass(ComputationalCell3D const &cell,
                          size_t beginGroup, size_t endGroup)
    {
        beginGroup = std::min(beginGroup,
                              static_cast<size_t>(ENERGY_GROUPS_NUM));
        endGroup = std::min(endGroup,
                            static_cast<size_t>(ENERGY_GROUPS_NUM));
        if(beginGroup >= endGroup)
            return 0.0;
        double const kT = units::k_boltz * cell.temperature;
        if(!(kT > 0.0) || !std::isfinite(kT))
            return 0.0;
        double mass = 0.0;
        for(size_t g = beginGroup; g < endGroup; ++g)
            mass += planck_integral::planck_integral(
                ComputationalCell3D::energyBoundaries[g] / kT,
                ComputationalCell3D::energyBoundaries[g + 1] / kT);
        return mass;
    }

    double SampleCosineMu(double xi)
    {
        return std::sqrt(std::clamp(xi, 0.0, 1.0));
    }

    double SampleAsymptoticDDMCToIMCMu(double xi)
    {
        // Detailed balance with the static Wollaeger admission law gives
        // p(mu)=mu*(1+3*mu/2), whose CDF is mu^2*(1+mu)/2.
        xi = std::clamp(xi, 0.0, 1.0);
        double lo = 0.0;
        double hi = 1.0;
        for(int iteration = 0; iteration < 56; ++iteration)
        {
            double const mu = 0.5 * (lo + hi);
            double const cdf = 0.5 * mu * mu * (1.0 + mu);
            if(cdf < xi)
                lo = mu;
            else
                hi = mu;
        }
        return 0.5 * (lo + hi);
    }

    bool FrequencyFitsDDMCPoint(bool usePGRW, int eligible,
                                size_t cutoff, double frequency)
    {
        if(eligible == 0)
            return false;
        if(!usePGRW)
            return true;
        if(cutoff == 0 || cutoff > ENERGY_GROUPS_NUM)
            return false;
        ClampFrequencyToBoundsDDMC(frequency);
        return frequency < ComputationalCell3D::energyBoundaries[cutoff];
    }

#ifdef RICH_MPI
    std::vector<rank_t> BuildSymmetricDDMCCorrespondents(
        const std::vector<rank_t> &localProcs)
    {
        int rank = 0;
        int size = 1;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int const localCount = static_cast<int>(localProcs.size());
        std::vector<int> counts(static_cast<size_t>(size), 0);
        MPI_Allgather(&localCount, 1, MPI_INT,
                      counts.data(), 1, MPI_INT,
                      MPI_COMM_WORLD);

        std::vector<int> displs(static_cast<size_t>(size), 0);
        int totalCount = 0;
        for(int r = 0; r < size; ++r)
        {
            displs[static_cast<size_t>(r)] = totalCount;
            totalCount += counts[static_cast<size_t>(r)];
        }

        std::vector<int> allProcs(static_cast<size_t>(std::max(totalCount, 1)), 0);
        MPI_Allgatherv(localCount > 0 ? localProcs.data() : nullptr,
                       localCount,
                       MPI_INT,
                       allProcs.data(),
                       counts.data(),
                       displs.data(),
                       MPI_INT,
                       MPI_COMM_WORLD);

        std::set<rank_t> symmetric(localProcs.begin(), localProcs.end());

        for(int sourceRank = 0; sourceRank < size; ++sourceRank)
        {
            int const offset = displs[static_cast<size_t>(sourceRank)];
            int const count = counts[static_cast<size_t>(sourceRank)];
            for(int j = 0; j < count; ++j)
            {
                int const listedTarget = allProcs[static_cast<size_t>(offset + j)];
                if(listedTarget == rank && sourceRank != rank)
                    symmetric.insert(sourceRank);
            }
        }

        symmetric.erase(rank);
        return std::vector<rank_t>(symmetric.begin(), symmetric.end());
    }
#endif
}

void RadiationIMC::precomputeDDMCData()
{
    size_t const Ncells = this->grid.GetPointNo();
    this->ddmcCellData.assign(Ncells, DDMCCellData{});

    this->ddmcInterfaceIncidentCount = 0;
    this->ddmcInterfaceAdmissionCount = 0;
    this->ddmcInterfaceReflectionCount = 0;
    this->ddmcInterfaceMovingFactorCount = 0;
    this->ddmcInterfaceMovingFallbackCount = 0;
    this->ddmcInterfaceSplitPacketCount = 0;
    this->ddmcInterfaceMinimumMu = std::numeric_limits<double>::infinity();
    this->ddmcInterfaceMaximumFactor = 1.0;
    this->ddmcLeakReciprocityResidualMax = 0.0;
    this->ddmcLeakReciprocityCheckCount = 0;
    this->ddmcLeakInvalidGeometryCount = 0;
    this->ddmcInterfaceBypassCount = 0;
    this->ddmcDopplerCutoffExitCount = 0;
    this->ddmcDiagnosticEvents.clear();

    for(size_t i = 0; i < Ncells; ++i)
    {
        DDMCCellData &data = this->ddmcCellData[i];
        const ComputationalCell3D &cell = this->cells[i];
        double const scatOp = this->opacity->CalcScatteringOpacity(cell);
        double const volume = this->grid.GetVolume(i);
        double surfaceArea = 0.0;
        for(size_t faceIdx : this->grid.GetCellFaces(i))
            surfaceArea += this->grid.GetArea(faceIdx);

        if(volume <= 0.0 || surfaceArea <= 0.0)
            continue;

        double const meanChordLength = 4.0 * volume / surfaceArea;
        Vector3D const cellCenter = this->grid.GetMeshPoint(i);
        for(size_t faceIdx : this->grid.GetCellFaces(i))
        {
            const auto &neighbors = this->grid.GetFaceNeighbors(faceIdx);
            size_t const nextCellIndex =
                (neighbors.first == i) ? neighbors.second : neighbors.first;

            Vector3D normal = this->grid.Normal(faceIdx);
            if(abs(normal) <= 0.0)
                continue;
            normal = normalize(normal);

            Vector3D towardNeighbor;
            if(nextCellIndex < this->grid.getMeshPoints().size())
                towardNeighbor = this->grid.GetMeshPoint(nextCellIndex) - cellCenter;
            else
                towardNeighbor = this->grid.FaceCM(faceIdx) - cellCenter;
            if(ScalarProd(normal, towardNeighbor) < 0.0)
                normal = -1.0 * normal;

            Vector3D neighborVelocity = cell.velocity;
            if(nextCellIndex < this->cells.size() &&
               !this->grid.IsPointOutsideBox(nextCellIndex))
            {
                neighborVelocity = this->cells[nextCellIndex].velocity;
            }

            double const area = this->grid.GetArea(faceIdx);
            data.velocityDivergence +=
                0.5 * ScalarProd(cell.velocity + neighborVelocity, normal) * area;
            data.maxFaceVelocityJumpOverC = std::max(
                data.maxFaceVelocityJumpOverC,
                abs(neighborVelocity - cell.velocity) * units::inv_clight);
        }
        data.velocityDivergence /= volume;

        bool const usePGRW = (this->multigroupOpacity != nullptr && this->ddmcUseMultigroupPGRW);

        if(usePGRW)
        {
            const auto &energyCenters = this->multigroupOpacity->getEnergyCenters();
            double const kT = units::k_boltz * cell.temperature;
            if(kT <= 0.0)
                continue;

            double totalSigABgAll = 0.0;
            double totalBgDiff = 0.0;
            double sumBgSigADiff = 0.0;
            double sumBgSigTDiff = 0.0;
            double sumBgOverSigTDiff = 0.0;
            size_t cutoff = 0;
            bool foundNonDiffusive = false;
            size_t const cutoffLimit = std::min(
                this->ddmcMaxGroupCutoff, static_cast<size_t>(ENERGY_GROUPS_NUM));

            for(size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
            {
                double const a = ComputationalCell3D::energyBoundaries[g] / kT;
                double const b = ComputationalCell3D::energyBoundaries[g + 1] / kT;
                double const Bg = planck_integral::planck_integral(a, b);
                double const sigA_g = this->opacity->CalcAbsorptionOpacity(cell, energyCenters[g]);
                double const sigT_g = sigA_g + scatOp;

                totalSigABgAll += sigA_g * Bg;

                if(g >= cutoffLimit)
                {
                    foundNonDiffusive = true;
                    continue;
                }

                if(!foundNonDiffusive && sigT_g * meanChordLength >= this->ddmcMinCellOpticalDepth)
                {
                    cutoff = g + 1;
                    totalBgDiff += Bg;
                    sumBgSigADiff += Bg * sigA_g;
                    sumBgSigTDiff += Bg * sigT_g;
                    if(sigT_g > 0.0)
                        sumBgOverSigTDiff += Bg / sigT_g;
                }
                else
                {
                    foundNonDiffusive = true;
                }
            }

            if(cutoff > 0 && totalBgDiff > 0.0)
            {
                data.groupCutoff = cutoff;
                data.sigmaA = sumBgSigADiff / totalBgDiff;
                data.sigmaT = sumBgSigTDiff / totalBgDiff;
                data.sigmaEnergyAbs = data.sigmaA;
                data.sigmaMomentum = data.sigmaT;
                data.sigmaDiffusion = (sumBgOverSigTDiff > 0.0)
                    ? totalBgDiff / sumBgOverSigTDiff
                    : data.sigmaT;
                data.sigmaParticleGate = data.sigmaT;
                data.sigmaGroupExit = data.sigmaT;
                data.diffusionCoefficient = (units::clight / 3.0) * sumBgOverSigTDiff / totalBgDiff;
                data.gamma = (totalSigABgAll > 0.0) ? sumBgSigADiff / totalSigABgAll : 1.0;
                data.eligible = (data.sigmaParticleGate > 0.0 && data.diffusionCoefficient > 0.0);
            }
        }
        else
        {
            data.sigmaA = this->planckOpacities[i];
            data.sigmaT = data.sigmaA + scatOp;
            data.sigmaEnergyAbs = data.sigmaA;
            data.sigmaMomentum = data.sigmaT;
            data.sigmaDiffusion = data.sigmaT;
            data.sigmaParticleGate = data.sigmaT;
            data.sigmaGroupExit = data.sigmaT;
            data.diffusionCoefficient = (data.sigmaDiffusion > 0.0) ? units::clight / (3.0 * data.sigmaDiffusion) : 0.0;
            data.gamma = 1.0;
            data.eligible = (data.sigmaParticleGate * meanChordLength >= this->ddmcMinCellOpticalDepth
                             && data.diffusionCoefficient > 0.0);
        }

        if(data.eligible && this->postProcess_.enabled && this->observer_)
        {
            Vector3D const cellCenter = this->grid.GetMeshPoint(i);
            double cellRadius = 0.0;
            for(size_t faceIdx : this->grid.GetCellFaces(i))
                cellRadius = std::max(cellRadius, abs(this->grid.FaceCM(faceIdx) - cellCenter));
            double const charLen = std::max(meanChordLength, cellRadius);
            double const distToObserver = abs(cellCenter - this->observer_->getCenter());
            double const obsR = this->observer_->getRadius();
            if(distToObserver + charLen >= obsR && distToObserver - charLen <= obsR)
            {
                data.eligible = false;
                data.observerExcluded = true;
            }
        }
    }

    // Finalize exclusions before exchanging eligibility.  The original patch
    // exchanged the first-pass flag and only afterwards removed unsupported
    // boundary cells, so neighboring ranks could construct an internal
    // leakage edge into a cell that had already been disabled locally.
    for(size_t i = 0; i < Ncells; ++i)
    {
        DDMCCellData &data = this->ddmcCellData[i];
        if(!data.eligible)
            continue;

        bool hasUsableTransportFace = false;
        Vector3D const center = this->grid.GetMeshPoint(i);
        for(size_t faceIdx : this->grid.GetCellFaces(i))
        {
            const auto &neighbors = this->grid.GetFaceNeighbors(faceIdx);
            size_t const nextCellIndex =
                (neighbors.first == i) ? neighbors.second : neighbors.first;
            if(this->grid.IsPointOutsideBox(nextCellIndex))
            {
                DDMCBoundaryFaceBehavior const behavior =
                    this->boundary->getDDMCBoundaryFaceBehavior(
                        faceIdx, i, nextCellIndex);
                if(behavior == DDMCBoundaryFaceBehavior::ReflectingRigid)
                {
                    ++data.rigidBoundaryFaceCount;
                }
                else
                {
                    ++data.unsupportedBoundaryFaceCount;
                    if(data.firstUnsupportedBoundaryFace ==
                       std::numeric_limits<size_t>::max())
                        data.firstUnsupportedBoundaryFace = faceIdx;
                    data.boundaryExcluded = true;
                }
                continue;
            }

            Vector3D normal = this->grid.Normal(faceIdx);
            double const area = this->grid.GetArea(faceIdx);
            if(abs(normal) <= 0.0 || !(area > 0.0))
                continue;
            normal = normalize(normal);
            Vector3D const faceCenter = this->grid.FaceCM(faceIdx);
            double const di = std::abs(ScalarProd(faceCenter - center, normal));
            double const dj = std::abs(ScalarProd(
                this->grid.GetMeshPoint(nextCellIndex) - faceCenter, normal));
            if(di > 0.0 && dj > 0.0 && std::isfinite(di) &&
               std::isfinite(dj) && std::isfinite(area))
                hasUsableTransportFace = true;
        }
        if(data.boundaryExcluded || !hasUsableTransportFace)
            data.eligible = false;
    }

    size_t const pointCount = std::max(this->grid.GetTotalPointNumber(),
                                       this->grid.getMeshPoints().size());
    this->ddmcPointEligible.assign(pointCount, 0);
    this->ddmcPointDiffusionCoefficient.assign(pointCount, 0.0);
    this->ddmcPointSigmaDiffusion.assign(pointCount, 0.0);
    this->ddmcPointGroupCutoff.assign(pointCount, 0);
    this->ddmcPointVelocity.assign(pointCount, Vector3D(0.0, 0.0, 0.0));
    this->ddmcPointCellID.assign(pointCount, std::numeric_limits<size_t>::max());
    for(size_t i = 0; i < Ncells; ++i)
    {
        this->ddmcPointEligible[i] = this->ddmcCellData[i].eligible ? 1 : 0;
        this->ddmcPointDiffusionCoefficient[i] =
            this->ddmcCellData[i].diffusionCoefficient;
        this->ddmcPointSigmaDiffusion[i] = this->ddmcCellData[i].sigmaDiffusion;
        this->ddmcPointGroupCutoff[i] = this->ddmcCellData[i].groupCutoff;
        this->ddmcPointVelocity[i] = this->cells[i].velocity;
        this->ddmcPointCellID[i] = this->cells[i].ID;
    }
#ifdef RICH_MPI
    // Internal leakage needs the target-cell resistance even when the target
    // is an MPI ghost.  Never fall back to the source diffusion coefficient.
    MPI_exchange_data(this->grid, this->ddmcPointEligible, true);
    MPI_exchange_data(this->grid, this->ddmcPointDiffusionCoefficient, true);
    MPI_exchange_data(this->grid, this->ddmcPointSigmaDiffusion, true);
    MPI_exchange_data(this->grid, this->ddmcPointGroupCutoff, true);
    MPI_exchange_data(this->grid, this->ddmcPointVelocity, true);
    MPI_exchange_data(this->grid, this->ddmcPointCellID, true);
#endif

    for(size_t i = 0; i < Ncells; ++i)
    {
        DDMCCellData &data = this->ddmcCellData[i];
        if(!data.eligible)
            continue;

        double const volume = this->grid.GetVolume(i);
        Vector3D const cellCenter = this->grid.GetMeshPoint(i);
        bool const usePGRW =
            this->multigroupOpacity != nullptr && this->ddmcUseMultigroupPGRW;
        double const sourceBandMass = usePGRW
            ? PlanckBandMass(this->cells[i], 0, data.groupCutoff) : 1.0;
        for(size_t faceIdx : this->grid.GetCellFaces(i))
        {
            const auto &neighbors = this->grid.GetFaceNeighbors(faceIdx);
            if(neighbors.first != i && neighbors.second != i)
                continue;
            size_t const nextCellIndex = (neighbors.first == i) ? neighbors.second : neighbors.first;
            if(this->grid.IsPointOutsideBox(nextCellIndex))
            {
                continue;
            }

            Vector3D normal = this->grid.Normal(faceIdx);
            if(abs(normal) <= 0.0)
                continue;
            normal = normalize(normal);

            Vector3D const faceCenter = this->grid.FaceCM(faceIdx);
            double const sourceDistanceToFace =
                std::abs(ScalarProd(faceCenter - cellCenter, normal));
            double targetDistanceToFace = 0.0;
            if(nextCellIndex < this->grid.getMeshPoints().size())
            {
                targetDistanceToFace = std::abs(ScalarProd(
                    this->grid.GetMeshPoint(nextCellIndex) - faceCenter,
                    normal));
            }

            double const area = this->grid.GetArea(faceIdx);
            if(!(sourceDistanceToFace > 0.0) || !(area > 0.0) ||
               !std::isfinite(sourceDistanceToFace) || !std::isfinite(area))
            {
                ++this->ddmcLeakInvalidGeometryCount;
                continue;
            }

            bool const targetEligible =
                nextCellIndex < this->ddmcPointEligible.size() &&
                this->ddmcPointEligible[nextCellIndex] != 0;
            double internalRate = 0.0;
            double conductance = 0.0;
            if(targetEligible && targetDistanceToFace > 0.0 &&
               nextCellIndex < this->ddmcPointDiffusionCoefficient.size())
            {
                double const targetDiffusion =
                    this->ddmcPointDiffusionCoefficient[nextCellIndex];
                if(data.diffusionCoefficient > 0.0 && targetDiffusion > 0.0)
                {
                    // Conservative two-sided finite-volume DDMC conductance.
                    // Both diffusion resistances are required.  On a uniform
                    // Voronoi mesh this gives D*A/(V*d_center), not twice that
                    // value.  See Wollaeger et al. (2013), Eq. (63).
                    double const resistance =
                        sourceDistanceToFace / data.diffusionCoefficient +
                        targetDistanceToFace / targetDiffusion;
                    if(resistance > 0.0 && std::isfinite(resistance))
                    {
                        conductance = area / resistance;
                        internalRate = conductance / volume;
                    }
                }
            }

            // Static asymptotic DDMC-to-IMC boundary leakage.  This rate is
            // also retained on a DDMC-DDMC face because a packet can lie above
            // the target cell's frequency cutoff.
            double const boundaryRate = DDMCWollaeger::BoundaryLeakRate(
                area, volume, data.sigmaDiffusion, sourceDistanceToFace,
                units::clight);
            size_t const targetGroupCutoff =
                (nextCellIndex < this->ddmcPointGroupCutoff.size())
                ? this->ddmcPointGroupCutoff[nextCellIndex] : 0;

            // Densmore et al. (2012), Eq. (49).  A grey DDMC particle has no
            // microscopic frequency.  If the target cutoff is smaller than
            // the source cutoff, split the face reaction into a DDMC channel
            // for the common low-frequency band and a transport channel for
            // the remainder, weighted by the source-cell Planck spectrum.
            double ddmcFraction = 0.0;
            if(targetEligible && internalRate > 0.0)
            {
                if(!usePGRW || targetGroupCutoff >= data.groupCutoff)
                    ddmcFraction = 1.0;
                else if(targetGroupCutoff > 0 && sourceBandMass > 0.0)
                    ddmcFraction = std::clamp(
                        PlanckBandMass(this->cells[i], 0, targetGroupCutoff) /
                            sourceBandMass,
                        0.0, 1.0);
            }

            double const ddmcRate = ddmcFraction * internalRate;
            double const transportRate =
                (1.0 - ddmcFraction) * boundaryRate;
            double const rate = ddmcRate + transportRate;
            if(rate > 0.0 && std::isfinite(rate))
            {
                DDMCFaceLeak faceLeak;
                faceLeak.faceIndex = faceIdx;
                faceLeak.nextCellIndex = nextCellIndex;
                faceLeak.kind = ddmcRate > 0.0 ? DDMCFaceKind::Internal
                                              : DDMCFaceKind::InterfaceToIMC;
                faceLeak.rate = rate;
                faceLeak.internalRate = internalRate;
                faceLeak.boundaryRate = boundaryRate;
                faceLeak.ddmcRate = ddmcRate;
                faceLeak.transportRate = transportRate;
                faceLeak.sourceBandMass = sourceBandMass;
                faceLeak.commonBandMass = ddmcFraction * sourceBandMass;
                faceLeak.ddmcFraction = ddmcFraction;
                faceLeak.area = area;
                faceLeak.sourceDistanceToFace = sourceDistanceToFace;
                faceLeak.targetDistanceToFace = targetDistanceToFace;
                faceLeak.conductance = conductance;
                faceLeak.targetDDMCEligible = ddmcRate > 0.0;
                faceLeak.targetGroupCutoff = targetGroupCutoff;
                faceLeak.outwardNormal = normal;
                data.faceLeaks.push_back(faceLeak);
                data.totalLeakRate += rate;
                data.faceAreaSum += faceLeak.area;
                data.fluxMatrix[0] += faceLeak.area * normal.x * normal.x;
                data.fluxMatrix[1] += faceLeak.area * normal.x * normal.y;
                data.fluxMatrix[2] += faceLeak.area * normal.x * normal.z;
                data.fluxMatrix[3] += faceLeak.area * normal.y * normal.y;
                data.fluxMatrix[4] += faceLeak.area * normal.y * normal.z;
                data.fluxMatrix[5] += faceLeak.area * normal.z * normal.z;
            }
        }

        if(data.boundaryExcluded || data.totalLeakRate <= 0.0)
        {
            data.eligible = false;
            data.faceLeaks.clear();
            data.totalLeakRate = 0.0;
        }
    }

    // Check the finite-volume reciprocity identity V_i*lambda_ij =
    // V_j*lambda_ji for local-local DDMC faces.
    for(size_t i = 0; i < Ncells; ++i)
    {
        double const volumeI = this->grid.GetVolume(i);
        for(DDMCFaceLeak const &forward : this->ddmcCellData[i].faceLeaks)
        {
            size_t const j = forward.nextCellIndex;
            if(!(forward.internalRate > 0.0) || j >= Ncells || j <= i)
                continue;
            DDMCFaceLeak const *reverse = nullptr;
            for(DDMCFaceLeak const &candidate : this->ddmcCellData[j].faceLeaks)
            {
                if(candidate.faceIndex == forward.faceIndex &&
                   candidate.nextCellIndex == i)
                {
                    reverse = &candidate;
                    break;
                }
            }
            if(reverse == nullptr || !(reverse->internalRate > 0.0))
                continue;
            double const lhs = volumeI * forward.internalRate;
            double const rhs = this->grid.GetVolume(j) * reverse->internalRate;
            double const scale = std::max({std::abs(lhs), std::abs(rhs),
                                           std::numeric_limits<double>::min()});
            double const residual = std::abs(lhs - rhs) / scale;
            this->ddmcLeakReciprocityResidualMax = std::max(
                this->ddmcLeakReciprocityResidualMax, residual);
            ++this->ddmcLeakReciprocityCheckCount;
        }
    }
}

void RadiationIMC::recordDDMCDiagnosticEvent(
    DDMCDiagnosticEventKind kind,
    size_t sourceCellIndex,
    size_t targetCellIndex,
    size_t faceIndex,
    size_t group,
    double energy,
    size_t sourceGroupCutoff,
    size_t targetGroupCutoff,
    double mu,
    double admissionProbability)
{
    if(!this->ddmcInterfaceDiagnostics)
        return;

    auto pointID = [this](size_t index) {
        if(index < this->ddmcPointCellID.size())
            return this->ddmcPointCellID[index];
        if(index < this->cells.size())
            return this->cells[index].ID;
        return std::numeric_limits<size_t>::max();
    };
    auto pointX = [this](size_t index) {
        if(index < this->grid.getMeshPoints().size())
            return this->grid.GetMeshPoint(index).x;
        return std::numeric_limits<double>::quiet_NaN();
    };

    size_t const sourceCellID = pointID(sourceCellIndex);
    size_t const targetCellID = pointID(targetCellIndex);
    DDMCDiagnosticEventKey const key{
        kind, faceIndex, sourceCellID, targetCellID, group};
    auto inserted = this->ddmcDiagnosticEvents.emplace(
        key, DDMCDiagnosticEventAccumulator{});
    DDMCDiagnosticEventAccumulator &entry = inserted.first->second;
    if(inserted.second)
    {
        entry.faceIndex = faceIndex;
        entry.sourceCellID = sourceCellID;
        entry.targetCellID = targetCellID;
        entry.group = group;
        entry.sourceGroupCutoff = sourceGroupCutoff;
        entry.targetGroupCutoff = targetGroupCutoff;
        entry.faceX = this->grid.FaceCM(faceIndex).x;
        entry.sourceGeneratorX = pointX(sourceCellIndex);
        entry.targetGeneratorX = pointX(targetCellIndex);
    }

    ++entry.count;
    if(std::isfinite(energy))
    {
        entry.signedEnergy += energy;
        entry.absoluteEnergy += std::abs(energy);
    }
    if(std::isfinite(mu))
    {
        entry.muSum += mu;
        ++entry.muCount;
    }
    if(std::isfinite(admissionProbability))
    {
        entry.admissionProbabilitySum += admissionProbability;
        ++entry.admissionProbabilityCount;
    }
}

std::string RadiationIMC::getDDMCFaceDiagnosticsTSV(double xMin,
                                                     double xMax) const
{
    std::ostringstream os;
    os.precision(17);
    if(!this->withDDMC || !this->ddmcInterfaceDiagnostics)
        return os.str();

    for(size_t i = 0; i < this->ddmcCellData.size(); ++i)
    {
        DDMCCellData const &data = this->ddmcCellData[i];
        size_t const sourceID = i < this->ddmcPointCellID.size()
            ? this->ddmcPointCellID[i] : this->cells[i].ID;
        double const sourceGeneratorX = this->grid.GetMeshPoint(i).x;
        double const sourceCellCMX = this->grid.GetCellCM(i).x;
        double const volume = this->grid.GetVolume(i);
        for(DDMCFaceLeak const &face : data.faceLeaks)
        {
            double const faceX = this->grid.FaceCM(face.faceIndex).x;
            if(faceX < xMin || faceX > xMax)
                continue;

            size_t const target = face.nextCellIndex;
            size_t const targetID = target < this->ddmcPointCellID.size()
                ? this->ddmcPointCellID[target]
                : std::numeric_limits<size_t>::max();
            double const targetGeneratorX =
                target < this->grid.getMeshPoints().size()
                ? this->grid.GetMeshPoint(target).x
                : std::numeric_limits<double>::quiet_NaN();
            int const targetEligible = target < this->ddmcPointEligible.size()
                ? this->ddmcPointEligible[target] : 0;
            double const targetSigma =
                target < this->ddmcPointSigmaDiffusion.size()
                ? this->ddmcPointSigmaDiffusion[target] : 0.0;
            double const targetD =
                target < this->ddmcPointDiffusionCoefficient.size()
                ? this->ddmcPointDiffusionCoefficient[target] : 0.0;

            os << sourceID << '\t' << targetID
               << '\t' << face.faceIndex
               << '\t' << sourceGeneratorX
               << '\t' << sourceCellCMX
               << '\t' << targetGeneratorX
               << '\t' << faceX
               << '\t' << volume
               << '\t' << data.groupCutoff
               << '\t' << face.targetGroupCutoff
               << '\t' << data.eligible
               << '\t' << targetEligible
               << '\t' << data.sigmaDiffusion
               << '\t' << targetSigma
               << '\t' << data.diffusionCoefficient
               << '\t' << targetD
               << '\t' << face.sourceDistanceToFace
               << '\t' << face.targetDistanceToFace
               << '\t' << face.area
               << '\t' << face.conductance
               << '\t' << face.internalRate
               << '\t' << face.boundaryRate
               << '\t' << face.sourceBandMass
               << '\t' << face.commonBandMass
               << '\t' << face.ddmcFraction
               << '\t' << face.ddmcRate
               << '\t' << face.transportRate
               << '\t' << face.rate
               << '\n';
        }
    }
    return os.str();
}

std::string RadiationIMC::getDDMCInterfaceEventDiagnosticsTSV(
    double xMin, double xMax) const
{
    std::ostringstream os;
    os.precision(17);
    if(!this->withDDMC || !this->ddmcInterfaceDiagnostics)
        return os.str();

    auto eventName = [](DDMCDiagnosticEventKind kind) {
        switch(kind)
        {
            case DDMCDiagnosticEventKind::IMCCandidate:
                return "imc_candidate";
            case DDMCDiagnosticEventKind::IMCFrequencyReject:
                return "imc_frequency_reject";
            case DDMCDiagnosticEventKind::IMCIncident:
                return "imc_incident";
            case DDMCDiagnosticEventKind::IMCAdmitted:
                return "imc_admitted";
            case DDMCDiagnosticEventKind::IMCReflected:
                return "imc_reflected";
            case DDMCDiagnosticEventKind::IMCBypass:
                return "imc_bypass";
            case DDMCDiagnosticEventKind::DDMCToDDMC:
                return "ddmc_to_ddmc";
            case DDMCDiagnosticEventKind::DDMCToIMC:
                return "ddmc_to_imc";
        }
        return "unknown";
    };

    for(auto const &item : this->ddmcDiagnosticEvents)
    {
        DDMCDiagnosticEventKey const &key = item.first;
        DDMCDiagnosticEventAccumulator const &entry = item.second;
        if(entry.faceX < xMin || entry.faceX > xMax)
            continue;
        long long const outputGroup =
            key.group == DDMC_DIAGNOSTIC_GREY_GROUP
            ? -1LL : static_cast<long long>(key.group);
        os << eventName(key.kind)
           << '\t' << entry.sourceCellID
           << '\t' << entry.targetCellID
           << '\t' << entry.faceIndex
           << '\t' << entry.sourceGeneratorX
           << '\t' << entry.targetGeneratorX
           << '\t' << entry.faceX
           << '\t' << outputGroup
           << '\t' << entry.sourceGroupCutoff
           << '\t' << entry.targetGroupCutoff
           << '\t' << entry.count
           << '\t' << entry.signedEnergy
           << '\t' << entry.absoluteEnergy
           << '\t' << entry.muSum
           << '\t' << entry.muCount
           << '\t' << entry.admissionProbabilitySum
           << '\t' << entry.admissionProbabilityCount
           << '\n';
    }
    return os.str();
}

bool RadiationIMC::tryIMCToDDMCInterface(
    Particle &particle, Functionality &functionality,
    std::vector<Particle> &particlesToAdd,
    size_t sourceCellIndex, size_t targetCellIndex, size_t faceIndex)
{
    if(targetCellIndex >= this->ddmcPointEligible.size() ||
       this->grid.IsPointOutsideBox(targetCellIndex) ||
       this->ddmcPointEligible[targetCellIndex] == 0 ||
       sourceCellIndex >= this->cells.size())
    {
        return false;
    }

    bool const usePGRW =
        this->multigroupOpacity != nullptr && this->ddmcUseMultigroupPGRW;
    ComputationalCell3D const &sourceCell = this->cells[sourceCellIndex];
    size_t const targetCellID =
        (targetCellIndex < this->ddmcPointCellID.size())
        ? this->ddmcPointCellID[targetCellIndex]
        : std::numeric_limits<size_t>::max();
    Vector3D const targetVelocity =
        (targetCellIndex < this->ddmcPointVelocity.size())
        ? this->ddmcPointVelocity[targetCellIndex] : sourceCell.velocity;

    bool const useVelocityTransport = this->useTransportVelocities_ && !this->MMC;

    Vector3D const faceCenter = this->grid.FaceCM(faceIndex);
    Vector3D const targetCenter = this->grid.GetMeshPoint(targetCellIndex);
    Vector3D const sourceCenter = this->grid.GetMeshPoint(sourceCellIndex);
    Vector3D normalOutOfDDMC = this->grid.Normal(faceIndex);
    if(abs(normalOutOfDDMC) <= 0.0)
        return false;
    normalOutOfDDMC = normalize(normalOutOfDDMC);
    if(ScalarProd(normalOutOfDDMC, sourceCenter - targetCenter) < 0.0)
        normalOutOfDDMC = -1.0 * normalOutOfDDMC;

    double const sourceDistance = std::abs(ScalarProd(
        faceCenter - sourceCenter, normalOutOfDDMC));
    double const targetDistance = std::abs(ScalarProd(
        targetCenter - faceCenter, normalOutOfDDMC));
    double const distanceSum = sourceDistance + targetDistance;
    Vector3D faceMaterialVelocity =
        0.5 * (sourceCell.velocity + targetVelocity);
    if(distanceSum > 0.0 && std::isfinite(distanceSum))
    {
        // Linear interpolation to the face.  This reduces to the arithmetic
        // mean for a Voronoi bisector.
        faceMaterialVelocity =
            (targetDistance * sourceCell.velocity +
             sourceDistance * targetVelocity) / distanceSum;
    }

    Particle faceComoving = particle;
    if(useVelocityTransport)
        LabToComovingPacket(faceComoving, faceMaterialVelocity);
    if(this->multigroupOpacity)
        ClampFrequencyToBoundsDDMC(faceComoving.frequency);

    Particle targetComoving = faceComoving;
    if(useVelocityTransport)
    {
        ComovingToLabPacket(targetComoving, faceMaterialVelocity);
        LabToComovingPacket(targetComoving, targetVelocity);
    }
    if(this->multigroupOpacity)
        ClampFrequencyToBoundsDDMC(targetComoving.frequency);
    size_t const sourceGroupCutoff =
        sourceCellIndex < this->ddmcPointGroupCutoff.size()
        ? this->ddmcPointGroupCutoff[sourceCellIndex] : 0;
    size_t const targetGroupCutoff =
        this->ddmcPointGroupCutoff[targetCellIndex];
    size_t const diagnosticGroup = this->opacity->findGroup(
        targetComoving.frequency);
    this->recordDDMCDiagnosticEvent(
        DDMCDiagnosticEventKind::IMCCandidate,
        sourceCellIndex, targetCellIndex, faceIndex, diagnosticGroup,
        faceComoving.weight, sourceGroupCutoff, targetGroupCutoff,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN());
    if(!FrequencyFitsDDMCPoint(usePGRW,
            this->ddmcPointEligible[targetCellIndex],
            targetGroupCutoff,
            targetComoving.frequency))
    {
        this->recordDDMCDiagnosticEvent(
            DDMCDiagnosticEventKind::IMCFrequencyReject,
            sourceCellIndex, targetCellIndex, faceIndex, diagnosticGroup,
            faceComoving.weight, sourceGroupCutoff, targetGroupCutoff,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
        return false;
    }

    double const speed = abs(faceComoving.velocity);
    if(!(speed > 0.0) || !std::isfinite(speed))
        return false;
    double const mu = -ScalarProd(faceComoving.velocity / speed,
                                  normalOutOfDDMC);
    if(!(mu > 0.0) || !std::isfinite(mu))
        return false;

    ++this->ddmcInterfaceIncidentCount;
    this->ddmcInterfaceMinimumMu = std::min(this->ddmcInterfaceMinimumMu, mu);
    double const ddmcDistance = targetDistance;
    double const sigmaTransport = this->ddmcPointSigmaDiffusion[targetCellIndex];

    auto bypassInterfaceAsIMC = [&]() -> bool {
        this->recordDDMCDiagnosticEvent(
            DDMCDiagnosticEventKind::IMCBypass, sourceCellIndex,
            targetCellIndex, faceIndex, diagnosticGroup, faceComoving.weight,
            sourceGroupCutoff, targetGroupCutoff, mu,
            std::numeric_limits<double>::quiet_NaN());
        particle.ddmcMode = false;
        particle.ddmcCellResident = false;
        particle.ddmcComovingFrame = false;
        particle.ddmcHasPendingFluxContribution = false;
        particle.ddmcPendingFluxContribution = Vector3D(0.0, 0.0, 0.0);
        particle.ddmcBypassCellID = targetCellID;
        if(targetCellID != std::numeric_limits<size_t>::max())
            particle.cellID = targetCellID;
        functionality.change = MonteCarloParticleStatus::CELL_MOVE;
        functionality.nextCellIndex = targetCellIndex;
        ++this->ddmcInterfaceBypassCount;
        return true;
    };

    double movingFactor = 1.0;
    if(useVelocityTransport && this->ddmcUseMovingInterfaceCorrection)
    {
        double const betaNormal =
            ScalarProd(faceMaterialVelocity, normalOutOfDDMC) *
            units::inv_clight;
        if(!std::isfinite(betaNormal) ||
           std::abs(betaNormal) > this->ddmcMaxInterfaceVelocityOverC)
        {
            ++this->ddmcInterfaceMovingFallbackCount;
            return bypassInterfaceAsIMC();
        }
        movingFactor = DDMCWollaeger::MovingFactor(mu, betaNormal);
        if(!(movingFactor > 0.0) || !std::isfinite(movingFactor))
        {
            ++this->ddmcInterfaceMovingFallbackCount;
            return bypassInterfaceAsIMC();
        }
        ++this->ddmcInterfaceMovingFactorCount;
        this->ddmcInterfaceMaximumFactor = std::max(
            this->ddmcInterfaceMaximumFactor, movingFactor);
    }

    double const targetWeight = this->ddmcInterfaceTargetWeightRatio *
        std::max(std::abs(particle.weight),
                 std::numeric_limits<double>::min());
    size_t requiredSplitCount = 1;
    if(targetWeight > 0.0)
    {
        requiredSplitCount = static_cast<size_t>(std::ceil(
            std::abs(faceComoving.weight * movingFactor) / targetWeight));
        requiredSplitCount = std::max<size_t>(1, requiredSplitCount);
    }
    if(requiredSplitCount > std::max<size_t>(1, this->ddmcMaxInterfaceSplits))
        return bypassInterfaceAsIMC();

    double const admissionProbability =
        DDMCWollaeger::StaticAdmissionProbability(
            mu, sigmaTransport, ddmcDistance);

    this->recordDDMCDiagnosticEvent(
        DDMCDiagnosticEventKind::IMCIncident, sourceCellIndex,
        targetCellIndex, faceIndex, diagnosticGroup, faceComoving.weight,
        sourceGroupCutoff, targetGroupCutoff, mu, admissionProbability);

    if(this->dist(this->re) > admissionProbability)
    {
        // Standard diffuse-albedo rejection.  The reflected packet remains
        // IMC and is nudged back into the source cell.
        constexpr double pi = 3.14159265358979323846;
        double const reflectedMu = SampleCosineMu(this->dist(this->re));
        double const phi = 2.0 * pi * this->dist(this->re);
        faceComoving.velocity = units::clight * SampleHemisphereDirection(
            normalOutOfDDMC, reflectedMu, phi);
        if(useVelocityTransport)
            ComovingToLabPacket(faceComoving, faceMaterialVelocity);
        particle.velocity = faceComoving.velocity;
        particle.frequency = faceComoving.frequency;
        particle.weight = faceComoving.weight;
        particle.location = 0.9999999999 * faceCenter
                          + 0.0000000001 * sourceCenter;
        particle.ddmcBypassCellID = std::numeric_limits<size_t>::max();
        functionality.change = MonteCarloParticleStatus::NO_CELL_MOVE;
        ++this->ddmcInterfaceReflectionCount;
        this->recordDDMCDiagnosticEvent(
            DDMCDiagnosticEventKind::IMCReflected, sourceCellIndex,
            targetCellIndex, faceIndex, diagnosticGroup, faceComoving.weight,
            sourceGroupCutoff, targetGroupCutoff, mu,
            admissionProbability);
        return true;
    }

    faceComoving.weight *= movingFactor;
    targetComoving = faceComoving;
    if(useVelocityTransport)
    {
        ComovingToLabPacket(targetComoving, faceMaterialVelocity);
        LabToComovingPacket(targetComoving, targetVelocity);
    }
    if(this->multigroupOpacity)
        ClampFrequencyToBoundsDDMC(targetComoving.frequency);
    double const admittedTargetWeight = targetComoving.weight;

    size_t splitCount = requiredSplitCount;
    // Additional packets can be inserted locally without entering a ghost
    // cell.  A remote interface keeps one unbiased, corrected-weight packet.
    if(targetCellIndex >= this->grid.GetPointNo())
        splitCount = 1;
    targetComoving.weight /= static_cast<double>(splitCount);
    targetComoving.initialWeight = std::abs(targetComoving.weight);
    targetComoving.location = targetCenter;
    targetComoving.ddmcMode = true;
    targetComoving.ddmcCellResident = true;
    targetComoving.ddmcComovingFrame = true;
    targetComoving.ddmcHasPendingFluxContribution = false;
    targetComoving.ddmcPendingFluxContribution = Vector3D(0.0, 0.0, 0.0);
    targetComoving.ddmcBypassCellID = std::numeric_limits<size_t>::max();
    targetComoving.cellID = targetCellID;

    Vector3D entryDirection = targetComoving.velocity;
    double const entrySpeed = abs(entryDirection);
    if(entrySpeed > 0.0 && std::isfinite(entrySpeed))
    {
        entryDirection = entryDirection / entrySpeed;
        Vector3D const contribution = admittedTargetWeight * entryDirection;
        if(targetCellIndex < this->ddmcFluxRhsIntegrated.size())
        {
            this->ddmcFluxRhsIntegrated[targetCellIndex] += contribution;
            ++this->ddmcInterfaceFluxTallyCount;
        }
        else
        {
            targetComoving.ddmcHasPendingFluxContribution = true;
            targetComoving.ddmcPendingFluxContribution = contribution;
        }
    }

    for(size_t copy = 1; copy < splitCount; ++copy)
    {
        Particle extra = targetComoving;
        extra.id = std::numeric_limits<size_t>::max();
        extra.cellIndex = targetCellIndex;
        particlesToAdd.push_back(extra);
        ++this->ddmcInterfaceSplitPacketCount;
    }

    size_t const oldCellIndex = particle.cellIndex;
    particle = targetComoving;
    // The Monte Carlo manager performs the actual local/MPI cell move.  Keep
    // the old index until it consumes the CELL_MOVE result.
    particle.cellIndex = oldCellIndex;
    particle.location = faceCenter;
    functionality.change = MonteCarloParticleStatus::CELL_MOVE;
    functionality.nextCellIndex = targetCellIndex;
    ++this->ddmcInterfaceAdmissionCount;
    this->recordDDMCDiagnosticEvent(
        DDMCDiagnosticEventKind::IMCAdmitted, sourceCellIndex,
        targetCellIndex, faceIndex, diagnosticGroup, admittedTargetWeight,
        sourceGroupCutoff, targetGroupCutoff, mu,
        admissionProbability);
    return true;
}

Vector3D RadiationIMC::sampleDDMCTransportLocation(size_t cellIndex)
{
    if(cellIndex >= this->grid.GetPointNo())
    {
        UniversalError eo("sampleDDMCTransportLocation: invalid local cell index");
        eo.addEntry("Cell index", cellIndex);
        eo.addEntry("Local cell count", this->grid.GetPointNo());
        throw eo;
    }

    Vector3D location = RandomPointInCell(this->grid, cellIndex);

    static constexpr double nudge = 1e-10;
    location = location * (1.0 - nudge) + nudge * this->grid.GetMeshPoint(cellIndex);

    this->validateDDMCTransportLocation(cellIndex,
                                        location,
                                        "sampleDDMCTransportLocation");
    return location;
}

double RadiationIMC::sampleDDMCPlanckFrequency(size_t cellIndex,
                                               size_t beginGroup,
                                               size_t endGroup)
{
    if(cellIndex >= this->cells.size())
    {
        UniversalError eo("sampleDDMCPlanckFrequency: invalid local cell index");
        eo.addEntry("Cell index", cellIndex);
        eo.addEntry("Cell count", this->cells.size());
        throw eo;
    }

    beginGroup = std::min(beginGroup,
                          static_cast<size_t>(ENERGY_GROUPS_NUM));
    endGroup = std::min(endGroup,
                        static_cast<size_t>(ENERGY_GROUPS_NUM));
    if(beginGroup >= endGroup)
        return ComputationalCell3D::energyBoundaries[beginGroup];

    ComputationalCell3D const &cell = this->cells[cellIndex];
    double const kT = units::k_boltz * cell.temperature;
    std::vector<double> cdf(endGroup - beginGroup + 1, 0.0);
    if(kT > 0.0 && std::isfinite(kT))
    {
        for(size_t g = beginGroup; g < endGroup; ++g)
        {
            cdf[g - beginGroup + 1] = cdf[g - beginGroup] +
                planck_integral::planck_integral(
                    ComputationalCell3D::energyBoundaries[g] / kT,
                    ComputationalCell3D::energyBoundaries[g + 1] / kT);
        }
    }

    double const total = cdf.back();
    if(!(total > 0.0) || !std::isfinite(total))
        return 0.5 * (ComputationalCell3D::energyBoundaries[beginGroup] +
                      ComputationalCell3D::energyBoundaries[endGroup]);

    double const unitUpper = std::nextafter(1.0, 0.0);
    double const target = std::clamp(this->dist(this->re), 0.0, unitUpper) * total;
    auto upper = std::upper_bound(cdf.begin(), cdf.end(), target);
    size_t localGroup = upper == cdf.begin()
        ? 0 : static_cast<size_t>(upper - cdf.begin() - 1);
    localGroup = std::min(localGroup, cdf.size() - 2);
    size_t const g = beginGroup + localGroup;
    double const c0 = cdf[localGroup];
    double const c1 = cdf[localGroup + 1];
    double const fraction = c1 > c0 ? (target - c0) / (c1 - c0) : 0.5;
    double frequency = ComputationalCell3D::energyBoundaries[g] +
        std::clamp(fraction, 0.0, 1.0) *
        (ComputationalCell3D::energyBoundaries[g + 1] -
         ComputationalCell3D::energyBoundaries[g]);
    frequency = std::min(frequency, std::nextafter(
        ComputationalCell3D::energyBoundaries[endGroup],
        ComputationalCell3D::energyBoundaries[beginGroup]));
    ClampFrequencyToBoundsDDMC(frequency);
    return frequency;
}

void RadiationIMC::validateDDMCTransportLocation(size_t cellIndex,
                                                 Vector3D const &location,
                                                 char const *context) const
{
    std::string const contextName = context != nullptr ? context : "unknown";

    if(cellIndex >= this->grid.GetPointNo() ||
       this->grid.IsPointOutsideBox(location) ||
       !this->grid.IsPointInCell(location, cellIndex))
    {
        UniversalError eo("Invalid DDMC transport location");
        eo.addEntry("Context", contextName);
        eo.addEntry("Cell index", cellIndex);
        eo.addEntry("Local cell count", this->grid.GetPointNo());
        eo.addEntry("Location", location);
        if(cellIndex < this->grid.GetPointNo())
            eo.addEntry("Cell mesh point", this->grid.GetMeshPoint(cellIndex));
        throw eo;
    }
}

void RadiationIMC::tallyDDMCFaceFlux(size_t sourceCellIndex,
                                     const DDMCFaceLeak &faceLeak,
                                     double comovingEnergy,
                                     const Vector3D &fluxDirection,
                                     bool includeTarget)
{
    if(!(comovingEnergy != 0.0) || !std::isfinite(comovingEnergy))
        return;
    if(sourceCellIndex >= this->ddmcFluxRhsIntegrated.size())
        return;

    Vector3D const contribution = comovingEnergy * fluxDirection;
    this->ddmcFluxRhsIntegrated[sourceCellIndex] += contribution;
    if(includeTarget && faceLeak.nextCellIndex < this->ddmcFluxRhsIntegrated.size())
    {
        this->ddmcFluxRhsIntegrated[faceLeak.nextCellIndex] += contribution;
        if(faceLeak.nextCellIndex < this->ddmcCellData.size())
        {
            double const sourceNormalFlux = ScalarProd(contribution, fluxDirection);
            double const targetNormalFlux = ScalarProd(contribution, -1.0 * fluxDirection);
            double const residual = std::abs(sourceNormalFlux + targetNormalFlux);
            if(std::isfinite(residual))
            {
                this->ddmcLocalFaceFluxPairResidualMax =
                    std::max(this->ddmcLocalFaceFluxPairResidualMax, residual);
                ++this->ddmcLocalFaceFluxPairCheckCount;
            }
        }
    }
    this->ddmcFaceFluxEnergy += std::abs(comovingEnergy);
    ++this->ddmcInterfaceFluxTallyCount;
    if(faceLeak.nextCellIndex >= this->grid.getMeshPoints().size())
        ++this->ddmcBoundaryFluxTallyCount;
}

void RadiationIMC::reduceDDMCFaceFluxTallies()
{
#ifdef RICH_MPI
    size_t const totalPoints = this->grid.GetTotalPointNumber();

    const std::vector<rank_t> rawProcs = this->grid.GetDuplicatedProcs();
    const std::vector<std::vector<size_t>> &rawGhostIndices =
        this->grid.GetGhostIndeces();
    const std::vector<std::vector<size_t>> &rawOwnerIndices =
        this->grid.GetDuplicatedPoints();

    if(rawProcs.size() != rawGhostIndices.size() ||
       rawProcs.size() != rawOwnerIndices.size())
    {
        UniversalError eo("DDMC MPI face-flux reduction has inconsistent tessellation exchange maps");
        eo.addEntry("procs", rawProcs.size());
        eo.addEntry("ghost maps", rawGhostIndices.size());
        eo.addEntry("owner maps", rawOwnerIndices.size());
        eo.addEntry("local cells", this->grid.GetPointNo());
        eo.addEntry("total points", totalPoints);
        throw eo;
    }

    {
        int mpiSize = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

        std::set<rank_t> seenRawProcs;
        for(rank_t peer : rawProcs)
        {
            if(peer < 0 || peer >= mpiSize)
            {
                UniversalError eo("DDMC MPI face-flux reduction has invalid peer rank");
                eo.addEntry("peer rank", peer);
                eo.addEntry("mpi size", mpiSize);
                eo.addEntry("local cells", this->grid.GetPointNo());
                eo.addEntry("total points", totalPoints);
                throw eo;
            }

            if(!seenRawProcs.insert(peer).second)
            {
                UniversalError eo("DDMC MPI face-flux reduction has duplicate peer rank");
                eo.addEntry("peer rank", peer);
                eo.addEntry("local cells", this->grid.GetPointNo());
                eo.addEntry("total points", totalPoints);
                throw eo;
            }
        }
    }

    // MPI_exchange_data_indexed posts one receive per correspondent and finishes
    // with an MPI_COMM_WORLD barrier. If rank A lists rank B but rank B does not
    // list rank A, the exchange hangs. A zero-owned-cell rank is the common way
    // to create that asymmetry.
    std::vector<rank_t> procs = BuildSymmetricDDMCCorrespondents(rawProcs);

    std::vector<std::vector<size_t>> ghostIndices(procs.size());
    std::vector<std::vector<size_t>> ownerIndices(procs.size());

    for(size_t i = 0; i < procs.size(); ++i)
    {
        auto it = std::find(rawProcs.begin(), rawProcs.end(), procs[i]);
        if(it == rawProcs.end())
            continue;

        size_t const rawSlot =
            static_cast<size_t>(std::distance(rawProcs.begin(), it));
        ghostIndices[i] = rawGhostIndices[rawSlot];
        ownerIndices[i] = rawOwnerIndices[rawSlot];
    }

    size_t requiredFluxSize = totalPoints;
    auto includeRequiredIndices = [&requiredFluxSize](
        const std::vector<std::vector<size_t>> &maps)
    {
        for(const auto &indices : maps)
            for(size_t idx : indices)
                requiredFluxSize = std::max(requiredFluxSize, idx + 1);
    };
    includeRequiredIndices(ghostIndices);
    includeRequiredIndices(ownerIndices);

    if(this->ddmcFluxRhsIntegrated.size() < requiredFluxSize)
    {
        this->ddmcFluxRhsIntegrated.resize(
            requiredFluxSize, Vector3D(0.0, 0.0, 0.0));
    }

    std::vector<std::vector<Vector3D>> incoming =
        MPI_exchange_data_indexed(procs, this->ddmcFluxRhsIntegrated, ghostIndices);

    if(incoming.size() != ownerIndices.size())
    {
        UniversalError eo("DDMC MPI face-flux reduction returned wrong rank-slot count");
        eo.addEntry("incoming slots", incoming.size());
        eo.addEntry("owner slots", ownerIndices.size());
        eo.addEntry("local cells", this->grid.GetPointNo());
        eo.addEntry("total points", totalPoints);
        throw eo;
    }

    double receivedEnergy = 0.0;
    size_t reducedValues = 0;
    for(size_t i = 0; i < incoming.size(); ++i)
    {
        if(incoming[i].size() != ownerIndices[i].size())
        {
            UniversalError eo("DDMC MPI face-flux reduction received wrong buffer length");
            eo.addEntry("rank slot", i);
            eo.addEntry("peer rank", procs[i]);
            eo.addEntry("received", incoming[i].size());
            eo.addEntry("expected", ownerIndices[i].size());
            eo.addEntry("local cells", this->grid.GetPointNo());
            eo.addEntry("total points", totalPoints);
            throw eo;
        }

        for(size_t j = 0; j < incoming[i].size(); ++j)
        {
            size_t const ownerIndex = ownerIndices[i][j];
            if(ownerIndex >= this->ddmcFluxRhsIntegrated.size())
                continue;

            this->ddmcFluxRhsIntegrated[ownerIndex] += incoming[i][j];
            receivedEnergy += abs(incoming[i][j]);
            ++reducedValues;
        }
    }

    for(auto const &indices : ghostIndices)
    {
        for(size_t ghostIndex : indices)
        {
            if(ghostIndex < this->ddmcFluxRhsIntegrated.size())
                this->ddmcFluxRhsIntegrated[ghostIndex] = Vector3D(0.0, 0.0, 0.0);
        }
    }

    if(reducedValues > 0)
    {
        ++this->ddmcMpiFaceFluxReductionCount;
        this->ddmcFaceFluxMpiEnergy += receivedEnergy;
    }
#endif
}

void RadiationIMC::recordDDMCWeightRatio(double weight, double initialWeight)
{
    if(!(std::isfinite(weight) && std::isfinite(initialWeight)))
        return;
    double const denom = std::abs(initialWeight);
    if(!(denom > 0.0))
        return;

    double const ratio = std::abs(weight) / denom;
    if(!std::isfinite(ratio))
        return;

    if(this->ddmcWeightRatioSamples.size() < DDMC_WEIGHT_RATIO_SAMPLE_LIMIT)
        this->ddmcWeightRatioSamples.push_back(ratio);
    else
        ++this->ddmcWeightRatioSamplesDropped;
    this->ddmcWeightRatioMax = std::max(this->ddmcWeightRatioMax, ratio);
    this->ddmcWeightRatioSum += ratio;
    ++this->ddmcWeightRatioCount;
    if(ratio > 8.0)
        ++this->ddmcWeightRatioOutlierCount;
}

void RadiationIMC::tallyDDMCMaterialEnergy(size_t cellIndex,
                                           double comovingEnergy,
                                           const Vector3D &cellVelocity)
{
    if(!std::isfinite(comovingEnergy))
        return;

    this->ddmcMaterialEnergyExchangeCo += comovingEnergy;

    double const beta2 = ScalarProd(cellVelocity, cellVelocity) * units::inv_clight2;
    double gamma = 1.0;
    if(beta2 > 0.0 && beta2 < 1.0)
        gamma = 1.0 / std::sqrt(1.0 - beta2);
    if(!std::isfinite(gamma))
        return;

    double const labEnergy = gamma * comovingEnergy;
    this->ddmcMaterialEnergyExchangeLab += labEnergy;
    Vector3D const labMomentum = labEnergy * cellVelocity * units::inv_clight2;
    this->ddmcMaterialMomentumExchangeLab += labMomentum;
    if(!this->noHydroFeedback && this->withHydro && !this->diffusionPressureGradient &&
       cellIndex < this->conserved.size())
    {
        this->conserved[cellIndex].momentum += labMomentum;
        this->ddmcAppliedMomentumExchangeLab += labMomentum;
    }
}

void RadiationIMC::applyDDMCMomentumFeedback(double fullDt)
{
    (void)fullDt;
    if(this->noHydroFeedback || !this->withHydro || !this->withDDMC)
        return;

    size_t const Ncells = this->grid.GetPointNo();
    if(this->ddmcFluxRhsIntegrated.size() < Ncells ||
       this->ddmcCellData.size() < Ncells)
        return;

    for(size_t i = 0; i < Ncells; ++i)
    {
        Vector3D const &rhsIntegrated = this->ddmcFluxRhsIntegrated[i];
        if(!(std::isfinite(rhsIntegrated.x) &&
             std::isfinite(rhsIntegrated.y) &&
             std::isfinite(rhsIntegrated.z)))
            continue;
        if(abs(rhsIntegrated) == 0.0)
            continue;

        DDMCCellData const &data = this->ddmcCellData[i];
        if(!data.eligible || data.sigmaMomentum <= 0.0)
            continue;

        Vector3D fluxDt;
        bool solved = SolveSymmetric3x3(data.fluxMatrix, rhsIntegrated, fluxDt);
        if(!solved)
        {
            if(!(data.faceAreaSum > 0.0))
                continue;
            fluxDt = rhsIntegrated / data.faceAreaSum;
            ++this->ddmcMomentumMatrixFallbackCount;
        }

        Vector3D const deltaP =
            data.sigmaMomentum * this->grid.GetVolume(i) * units::inv_clight * fluxDt;
        if(!(std::isfinite(deltaP.x) &&
             std::isfinite(deltaP.y) &&
             std::isfinite(deltaP.z)))
            continue;

        this->conserved[i].momentum += deltaP;
        this->ddmcFluxMomentumExchangeLab += deltaP;
        this->ddmcAppliedMomentumExchangeLab += deltaP;
        ++this->ddmcMomentumFeedbackCount;
    }
}

bool RadiationIMC::tryDDMCStep(Particle &particle, Functionality &functionality, double dopplerShift)
{
    (void)dopplerShift;

    size_t const cellIndex = particle.cellIndex;
    if(cellIndex >= this->ddmcCellData.size())
    {
        if(particle.ddmcMode)
        {
            UniversalError eo("DDMC resident particle has out-of-range cell index");
            eo.addEntry("Particle", particle);
            eo.addEntry("DDMC cell data size", this->ddmcCellData.size());
            throw eo;
        }
        return false;
    }

    ComputationalCell3D &cell = this->cells[cellIndex];

    bool const useVelocityTransport = this->useTransportVelocities_ && !this->MMC;
    bool const continuingDDMC =
        particle.ddmcMode && particle.ddmcCellResident && particle.ddmcComovingFrame;

    if(particle.ddmcMode && !continuingDDMC)
    {
        UniversalError eo("Inconsistent DDMC particle state");
        eo.addEntry("Particle", particle);
        eo.addEntry("ddmcMode", particle.ddmcMode);
        eo.addEntry("ddmcCellResident", particle.ddmcCellResident);
        eo.addEntry("ddmcComovingFrame", particle.ddmcComovingFrame);
        throw eo;
    }

    auto convertResidentDDMCToTransport = [&]() {
        particle.location = this->sampleDDMCTransportLocation(cellIndex);
        particle.velocity = units::clight * SampleIsotropicDirection(
            this->dist(this->re), this->dist(this->re));
        particle.ddmcMode = false;
        particle.ddmcCellResident = false;
        particle.ddmcComovingFrame = false;
        particle.ddmcHasPendingFluxContribution = false;
        particle.ddmcPendingFluxContribution = Vector3D(0.0, 0.0, 0.0);
        if(useVelocityTransport)
        {
            ComovingToLabPacket(particle, cell.velocity);
            if(this->multigroupOpacity)
                ClampFrequencyToBoundsDDMC(particle.frequency);
        }
        double const transportReferenceWeight = std::abs(particle.weight);
        if(transportReferenceWeight > 0.0)
            particle.initialWeight = transportReferenceWeight;
    };

    DDMCCellData const &data = this->ddmcCellData[cellIndex];
    if(particle.ddmcHasPendingFluxContribution)
    {
        if(data.eligible && cellIndex < this->ddmcFluxRhsIntegrated.size())
        {
            this->ddmcFluxRhsIntegrated[cellIndex] +=
                particle.ddmcPendingFluxContribution;
        }
        particle.ddmcHasPendingFluxContribution = false;
        particle.ddmcPendingFluxContribution = Vector3D(0.0, 0.0, 0.0);
    }

    if(!data.eligible || data.totalLeakRate <= 0.0 || data.faceLeaks.empty()
       || data.sigmaParticleGate <= 0.0 || data.diffusionCoefficient <= 0.0)
    {
        if(continuingDDMC)
            convertResidentDDMCToTransport();
        return false;
    }

    double const gammaCell = useVelocityTransport ? CellGamma(cell.velocity) : 1.0;

    if(!(gammaCell > 0.0) || !std::isfinite(gammaCell))
    {
        UniversalError eo("Invalid DDMC cell gamma");
        eo.addEntry("Particle", particle);
        eo.addEntry("Cell index", cellIndex);
        eo.addEntry("Cell velocity", cell.velocity);
        eo.addEntry("Gamma", gammaCell);
        throw eo;
    }

    Particle materialParticle = particle;
    if(!continuingDDMC && useVelocityTransport)
    {
        LabToComovingPacket(materialParticle, cell.velocity);
        if(this->multigroupOpacity)
            ClampFrequencyToBoundsDDMC(materialParticle.frequency);
    }
    materialParticle.location = this->grid.GetMeshPoint(cellIndex);
    materialParticle.ddmcMode = true;
    materialParticle.ddmcCellResident = true;
    materialParticle.ddmcComovingFrame = true;
    if(!continuingDDMC)
        materialParticle.initialWeight = std::abs(materialParticle.weight);

    bool const usePGRW = (this->multigroupOpacity != nullptr && this->ddmcUseMultigroupPGRW);
    if(usePGRW)
    {
        if(data.groupCutoff == 0 || data.groupCutoff > ENERGY_GROUPS_NUM)
        {
            ++this->ddmcFallbackCount;
            if(continuingDDMC)
                convertResidentDDMCToTransport();
            return false;
        }
        if(!continuingDDMC)
        {
            double coFreq = materialParticle.frequency;
            ClampFrequencyToBoundsDDMC(coFreq);
            if(coFreq >= ComputationalCell3D::energyBoundaries[data.groupCutoff])
            {
                ++this->ddmcFallbackCount;
                return false;
            }
        }
        // Once admitted, the packet represents the frequency-integrated grey
        // band.  Refresh an auxiliary local-equilibrium frequency for Doppler
        // bookkeeping only; leakage and absorption rates must not depend on it.
        materialParticle.frequency = this->sampleDDMCPlanckFrequency(
            cellIndex, 0, data.groupCutoff);
    }

    double const f = this->factorFleck[cellIndex];
    double const upscatterRateCo = (usePGRW && data.gamma < 1.0 && data.sigmaEnergyAbs > 0.0 && f > 0.0)
        ? units::clight * (1.0 - f) * data.sigmaEnergyAbs * (1.0 - data.gamma)
        : 0.0;
    double applicableLeakRate = 0.0;
    for(DDMCFaceLeak const &face : data.faceLeaks)
    {
        if(face.rate > 0.0 && std::isfinite(face.rate))
            applicableLeakRate += face.rate;
    }
    double const eventRateCo = applicableLeakRate + upscatterRateCo;
    if(!(eventRateCo > 0.0) || !std::isfinite(eventRateCo))
    {
        ++this->ddmcFallbackCount;
        if(continuingDDMC)
            convertResidentDDMCToTransport();
        return false;
    }

    if(!continuingDDMC && cellIndex < this->ddmcFluxRhsIntegrated.size())
    {
        Vector3D entryDirection = materialParticle.velocity;
        double const entrySpeed = abs(entryDirection);
        if(entrySpeed > 0.0 && std::isfinite(entrySpeed))
        {
            entryDirection = entryDirection / entrySpeed;
            this->ddmcFluxRhsIntegrated[cellIndex] +=
                materialParticle.weight * entryDirection;
            ++this->ddmcInterfaceFluxTallyCount;
        }
    }

    double const tEventCo = -std::log(PositiveRandom(this->dist(this->re))) / eventRateCo;
    double const tCensusCo = useVelocityTransport ? particle.timeLeft / gammaCell
                                              : particle.timeLeft;

    double tCutoffCo = std::numeric_limits<double>::infinity();
    if(usePGRW && useVelocityTransport && data.velocityDivergence < 0.0 &&
       data.groupCutoff > 0 && data.groupCutoff <= ENERGY_GROUPS_NUM)
    {
        double const cutoffFrequency =
            ComputationalCell3D::energyBoundaries[data.groupCutoff];
        double const logarithmicGrowthRate = -data.velocityDivergence / 3.0;
        if(materialParticle.frequency > 0.0 &&
           materialParticle.frequency < cutoffFrequency &&
           logarithmicGrowthRate > 0.0)
        {
            tCutoffCo = std::log(cutoffFrequency /
                                 materialParticle.frequency) /
                        logarithmicGrowthRate;
        }
    }

    double const dtCo = std::min({tEventCo, tCensusCo, tCutoffCo});

    // Level-1 mixed-frame approximation: dtLab = gammaCell * dtCo.
    // This intentionally ignores the event displacement term
    // gammaCell * dot(cell.velocity, dxCo) / c^2.
    // Use Level 2 before relying on high-v/c DDMC results.
    double dtLab = useVelocityTransport ? gammaCell * dtCo : dtCo;
    bool const cutoffEvent =
        tCutoffCo <= tEventCo && tCutoffCo < tCensusCo;
    bool const censusEvent =
        !cutoffEvent && tCensusCo <= tEventCo;
    if(censusEvent)
        dtLab = particle.timeLeft;

    if(dtLab < 0.0 || !std::isfinite(dtLab))
    {
        ++this->ddmcFallbackCount;
        return false;
    }
    if(dtLab > particle.timeLeft)
        dtLab = particle.timeLeft;

    if(useVelocityTransport && data.velocityDivergence != 0.0)
    {
        double const logShift = -data.velocityDivergence * dtCo / 3.0;
        if(std::isfinite(logShift) && logShift != 0.0)
        {
            double const boundedLogShift = std::clamp(logShift, -50.0, 50.0);
            double const shift = std::exp(boundedLogShift);
            materialParticle.frequency *= shift;
            materialParticle.weight *= shift;
            ++this->ddmcMovingMediumUpdateCount;
            this->ddmcMaxMovingMediumLogShift = std::max(
                this->ddmcMaxMovingMediumLogShift,
                std::abs(logShift));
        }
    }

    double const absRateCo = data.sigmaEnergyAbs * f * units::clight;
    double const oldCoWeight = materialParticle.weight;
    double const expFactorCo = std::expm1(-dtCo * absRateCo);
    double const absorbedCo = -expFactorCo * oldCoWeight;

    this->tallyDDMCMaterialEnergy(cellIndex, absorbedCo, cell.velocity);
    if(!this->noHydroFeedback)
        this->conserved[cellIndex].internal_energy += absorbedCo;

    // TODO(mixed-frame): Erad_time_avg is accumulated as a material-frame
    // diffusion tally here. Ordinary IMC mixed-frame tally semantics should be
    // audited separately before using Erad_time_avg for high-v/c hydro diagnostics.
    double integratedCo;
    if(absRateCo > 0.0)
        integratedCo = oldCoWeight * expFactorCo * (-1.0 / absRateCo);
    else
        integratedCo = oldCoWeight * dtCo;

    this->Erad_time_avg[cellIndex] += integratedCo;

    if(this->withEgTimeAvg && this->multigroupOpacity)
    {
        if(usePGRW && data.groupCutoff > 0)
        {
            double const bandMass = PlanckBandMass(cell, 0, data.groupCutoff);
            double const kT = units::k_boltz * cell.temperature;
            if(bandMass > 0.0 && kT > 0.0)
            {
                for(size_t g = 0; g < data.groupCutoff; ++g)
                {
                    double const Bg = planck_integral::planck_integral(
                        ComputationalCell3D::energyBoundaries[g] / kT,
                        ComputationalCell3D::energyBoundaries[g + 1] / kT);
                    this->Eg_time_avg[cellIndex][g] +=
                        integratedCo * Bg / bandMass;
                }
            }
        }
        else
        {
            double freqForGroup = materialParticle.frequency;
            ClampFrequencyToBoundsDDMC(freqForGroup);
            size_t const g = this->opacity->findGroup(freqForGroup);
            this->Eg_time_avg[cellIndex][g] += integratedCo;
        }
    }

    materialParticle.weight *= 1.0 + expFactorCo;

    if(this->postProcess_.enabled && this->observer_)
    {
        double const absorbed = oldCoWeight - materialParticle.weight;
        if(absorbed > 0.0)
        {
            this->observer_->addAbsorbedEnergy(absorbed);
            ++this->ddmcObserverEnergyOnlyTallyCount;
        }
    }

    materialParticle.timeLeft = particle.timeLeft - dtLab;
    if(materialParticle.timeLeft < 0.0 && materialParticle.timeLeft > -1e-12)
        materialParticle.timeLeft = 0.0;

    ++this->ddmcStepCount;

    auto setParticleCellIdentity = [&](Particle &p, size_t idx)
    {
        p.cellIndex = idx;
        if(idx < this->cells.size())
            p.cellID = this->cells[idx].ID;
    };

    auto storeDDMCResidentParticle = [&](size_t residentCellIndex) {
        particle.location = this->grid.GetMeshPoint(residentCellIndex);
        particle.velocity  = materialParticle.velocity;
        setParticleCellIdentity(particle, residentCellIndex);
        particle.frequency = materialParticle.frequency;
        particle.weight    = materialParticle.weight;
        particle.initialWeight = materialParticle.initialWeight;
        particle.timeLeft  = materialParticle.timeLeft;
        particle.ddmcMode = true;
        particle.ddmcCellResident = true;
        particle.ddmcComovingFrame = true;
        particle.ddmcHasPendingFluxContribution = false;
        particle.ddmcPendingFluxContribution = Vector3D(0.0, 0.0, 0.0);
#ifdef MONTECARLO_POLARIZATION
        particle.stokesQ = materialParticle.stokesQ;
        particle.stokesU = materialParticle.stokesU;
        particle.polarizationBasis = materialParticle.polarizationBasis;
        particle.polarizationInitialized = materialParticle.polarizationInitialized;
#endif
    };

    auto storeTransportParticle = [&](Particle const &transportParticle) {
        particle.location  = transportParticle.location;
        particle.velocity  = transportParticle.velocity;
        particle.frequency = transportParticle.frequency;
        particle.weight    = transportParticle.weight;
        setParticleCellIdentity(particle, transportParticle.cellIndex);
        double const transportReferenceWeight =
            std::abs(transportParticle.weight);
        if(transportReferenceWeight > 0.0)
            particle.initialWeight = transportReferenceWeight;
        particle.timeLeft  = transportParticle.timeLeft;
        particle.ddmcMode = false;
        particle.ddmcCellResident = false;
        particle.ddmcComovingFrame = false;
        particle.ddmcHasPendingFluxContribution = false;
        particle.ddmcPendingFluxContribution = Vector3D(0.0, 0.0, 0.0);
#ifdef MONTECARLO_POLARIZATION
        particle.stokesQ = transportParticle.stokesQ;
        particle.stokesU = transportParticle.stokesU;
        particle.polarizationBasis = transportParticle.polarizationBasis;
        particle.polarizationInitialized = transportParticle.polarizationInitialized;
        if(particle.polarizationInitialized)
            particle.polarizationBasis =
                IMCPolarization::ProjectBasisToDirection(particle.polarizationBasis,
                                                         particle.velocity);
#endif
    };

    auto storeDDMCTransferParticle = [&]() {
        particle.location  = materialParticle.location;
        particle.velocity  = materialParticle.velocity;
        particle.frequency = materialParticle.frequency;
        particle.weight    = materialParticle.weight;
        setParticleCellIdentity(particle, materialParticle.cellIndex);
        particle.initialWeight = materialParticle.initialWeight;
        particle.timeLeft  = materialParticle.timeLeft;
        particle.ddmcMode = true;
        particle.ddmcCellResident = true;
        particle.ddmcComovingFrame = true;
        particle.ddmcHasPendingFluxContribution =
            materialParticle.ddmcHasPendingFluxContribution;
        particle.ddmcPendingFluxContribution =
            materialParticle.ddmcPendingFluxContribution;
#ifdef MONTECARLO_POLARIZATION
        particle.stokesQ = materialParticle.stokesQ;
        particle.stokesU = materialParticle.stokesU;
        particle.polarizationBasis = materialParticle.polarizationBasis;
        particle.polarizationInitialized = materialParticle.polarizationInitialized;
#endif
    };

    Vector3D finalTransportFrameVelocity = cell.velocity;

    auto finalizeAccelerationStep = [&](bool remove,
                                        bool remainInDDMC,
                                        bool preservePhysicalState) -> bool {
        if(remove)
        {
            functionality.change = MonteCarloParticleStatus::REMOVE;
            return true;
        }

        if(remainInDDMC)
        {
            if(preservePhysicalState)
                storeDDMCTransferParticle();
            else
                storeDDMCResidentParticle(materialParticle.cellIndex);
            return true;
        }

        Particle finalLabParticle = materialParticle;

        if(useVelocityTransport)
        {
            ComovingToLabPacket(finalLabParticle, finalTransportFrameVelocity);
            if(this->multigroupOpacity)
                ClampFrequencyToBoundsDDMC(finalLabParticle.frequency);
        }

        finalLabParticle.ddmcMode = false;
        finalLabParticle.ddmcCellResident = false;
        finalLabParticle.ddmcComovingFrame = false;
        storeTransportParticle(finalLabParticle);

        return true;
    };

    bool removeParticle = false;

    double const lowWeightCutoff = this->postProcess_.enabled ? 1e-8 : 1e-3;
    double const initialCoWeight = materialParticle.initialWeight;
    double const absoluteCoWeight = std::abs(materialParticle.weight);
    double const referenceCoWeight = std::abs(initialCoWeight);

    if(absoluteCoWeight == 0.0 ||
       (referenceCoWeight > 0.0 &&
        absoluteCoWeight <= referenceCoWeight * lowWeightCutoff))
    {
        removeParticle = true;

        if(this->postProcess_.enabled && this->observer_)
        {
            this->observer_->addCutoffEnergy(materialParticle.weight);
            ++this->ddmcObserverEnergyOnlyTallyCount;
        }

        this->tallyDDMCMaterialEnergy(cellIndex, materialParticle.weight,
                                      cell.velocity);
        if(!this->noHydroFeedback)
            this->conserved[cellIndex].internal_energy += materialParticle.weight;
    }

    this->recordDDMCWeightRatio(materialParticle.weight,
                                materialParticle.initialWeight);

    if(removeParticle)
        return finalizeAccelerationStep(true, false, false);

    if(cutoffEvent)
    {
        double const cutoffFrequency =
            ComputationalCell3D::energyBoundaries[data.groupCutoff];
        materialParticle.frequency = std::nextafter(
            cutoffFrequency, std::numeric_limits<double>::max());
        materialParticle.velocity = this->opacity->getRandomVelocity(cell);
#ifdef MONTECARLO_POLARIZATION
        if(this->postProcess_.enabled && this->postProcess_.polarization.enabled)
            IMCPolarization::ResetUnpolarized(materialParticle);
#endif
        ++this->ddmcDopplerCutoffExitCount;
        return finalizeAccelerationStep(false, false, false);
    }

    if(censusEvent)
    {
#ifdef MONTECARLO_POLARIZATION
        if(this->postProcess_.enabled && this->postProcess_.polarization.enabled)
        {
            IMCPolarization::InitializeIfNeeded(materialParticle);
            materialParticle.polarizationBasis =
                IMCPolarization::ProjectBasisToDirection(materialParticle.polarizationBasis,
                                                         materialParticle.velocity);

            double const scatOp = this->opacity->CalcScatteringOpacity(cell);
            double const sigmaReset = (1.0 - f) * data.sigmaEnergyAbs;
            IMCPolarization::ApplyAcceleratedPolarizationHistory(
                materialParticle,
                dtCo,
                scatOp,
                sigmaReset,
                materialParticle.velocity,
                this->postProcess_.polarization.manualScatteringsAfterAcceleration,
                this->postProcess_.polarization.depolarizationScatterings,
                this->re,
                this->dist);
        }
#endif
        // Eq. (31) supplies the lost microscopic state of the grey particle.
        // Store it as a transport packet at census so the next time step can
        // classify it against the newly recomputed cutoff.
        materialParticle.location = this->sampleDDMCTransportLocation(cellIndex);
        materialParticle.velocity = units::clight * SampleIsotropicDirection(
            this->dist(this->re), this->dist(this->re));
        if(usePGRW)
            materialParticle.frequency = this->sampleDDMCPlanckFrequency(
                cellIndex, 0, data.groupCutoff);
        functionality.change = MonteCarloParticleStatus::DONE;
        ++this->ddmcCensusCount;
        return finalizeAccelerationStep(false, false, false);
    }

    double eventPick = this->dist(this->re) * eventRateCo;
    if(eventPick <= applicableLeakRate)
    {
        double facePick = this->dist(this->re) * applicableLeakRate;
        DDMCFaceLeak const *chosen = nullptr;
        for(DDMCFaceLeak const &faceLeak : data.faceLeaks)
        {
            facePick -= faceLeak.rate;
            if(facePick <= 0.0)
            {
                chosen = &faceLeak;
                break;
            }
        }
        if(chosen == nullptr && !data.faceLeaks.empty())
            chosen = &data.faceLeaks.back();
        if(chosen == nullptr)
        {
            ++this->ddmcFallbackCount;
            functionality.change = MonteCarloParticleStatus::DONE;
            return finalizeAccelerationStep(false, true, false);
        }

        double const channelPick = this->dist(this->re) * chosen->rate;
        bool const chosenDDMCChannel = chosen->ddmcRate > 0.0 &&
            (chosen->transportRate <= 0.0 || channelPick < chosen->ddmcRate);

        Vector3D const leakFaceCenter = this->grid.FaceCM(chosen->faceIndex);

        Vector3D nOut = chosen->outwardNormal;
        if(abs(nOut) <= 0.0)
            nOut = this->grid.Normal(chosen->faceIndex);
        if(abs(nOut) <= 0.0)
        {
            ++this->ddmcFallbackCount;
            functionality.change = MonteCarloParticleStatus::DONE;
            return finalizeAccelerationStep(false, true, false);
        }
        nOut = normalize(nOut);

        Vector3D const sourceCenter = this->grid.GetMeshPoint(cellIndex);
        Vector3D towardNeighbor;
        if(chosen->nextCellIndex < this->grid.getMeshPoints().size())
            towardNeighbor = this->grid.GetMeshPoint(chosen->nextCellIndex) - sourceCenter;
        else
            towardNeighbor = leakFaceCenter - sourceCenter;

        if(ScalarProd(nOut, towardNeighbor) < 0.0)
            nOut = -1.0 * nOut;

        bool const localDDMCNeighborCandidate =
            chosenDDMCChannel &&
            chosen->targetDDMCEligible &&
            chosen->nextCellIndex < this->ddmcCellData.size() &&
            this->ddmcCellData[chosen->nextCellIndex].totalLeakRate > 0.0;

        Particle targetMaterialParticle = materialParticle;
        if(localDDMCNeighborCandidate)
        {
            targetMaterialParticle.ddmcMode = true;
            targetMaterialParticle.ddmcCellResident = true;
            targetMaterialParticle.ddmcComovingFrame = true;
        }

        bool const localDDMCNeighbor =
            localDDMCNeighborCandidate;

        bool const remoteDDMCNeighbor =
            chosenDDMCChannel && chosen->targetDDMCEligible &&
            !localDDMCNeighbor &&
            chosen->nextCellIndex >= this->ddmcCellData.size() &&
            chosen->nextCellIndex < this->grid.getMeshPoints().size();

        if(localDDMCNeighbor)
        {
            this->recordDDMCDiagnosticEvent(
                DDMCDiagnosticEventKind::DDMCToDDMC, cellIndex,
                chosen->nextCellIndex, chosen->faceIndex,
                DDMC_DIAGNOSTIC_GREY_GROUP, materialParticle.weight,
                data.groupCutoff, chosen->targetGroupCutoff,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN());
            this->tallyDDMCFaceFlux(cellIndex, *chosen, materialParticle.weight,
                                    nOut, true);
            materialParticle = targetMaterialParticle;
            setParticleCellIdentity(materialParticle, chosen->nextCellIndex);
            materialParticle.location = this->grid.GetMeshPoint(chosen->nextCellIndex);
            materialParticle.ddmcMode = true;
            materialParticle.ddmcCellResident = true;
            materialParticle.ddmcComovingFrame = true;
            functionality.change = MonteCarloParticleStatus::NO_CELL_MOVE;
            ++this->ddmcLeakCount;
            ++this->ddmcResidentLeakCount;
            return finalizeAccelerationStep(false, true, false);
        }

        if(remoteDDMCNeighbor)
        {
            this->recordDDMCDiagnosticEvent(
                DDMCDiagnosticEventKind::DDMCToDDMC, cellIndex,
                chosen->nextCellIndex, chosen->faceIndex,
                DDMC_DIAGNOSTIC_GREY_GROUP, materialParticle.weight,
                data.groupCutoff, chosen->targetGroupCutoff,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN());
            this->tallyDDMCFaceFlux(cellIndex, *chosen, materialParticle.weight,
                                    nOut, true);
            setParticleCellIdentity(materialParticle, chosen->nextCellIndex);
            materialParticle.location = this->grid.GetMeshPoint(chosen->nextCellIndex);
            materialParticle.ddmcMode = true;
            materialParticle.ddmcCellResident = true;
            materialParticle.ddmcComovingFrame = true;
            materialParticle.ddmcHasPendingFluxContribution = false;
            materialParticle.ddmcPendingFluxContribution =
                Vector3D(0.0, 0.0, 0.0);
            functionality.change = MonteCarloParticleStatus::CELL_MOVE;
            functionality.nextCellIndex = chosen->nextCellIndex;
            ++this->ddmcLeakCount;
            ++this->ddmcResidentLeakCount;
            ++this->ddmcRemoteResidentLeakCount;
            return finalizeAccelerationStep(false, true, true);
        }

        if(usePGRW)
        {
            size_t lowerGroup = 0;
            if(chosen->targetDDMCEligible &&
               chosen->targetGroupCutoff > 0 &&
               chosen->targetGroupCutoff < data.groupCutoff)
                lowerGroup = chosen->targetGroupCutoff;
            materialParticle.frequency = this->sampleDDMCPlanckFrequency(
                cellIndex, lowerGroup, data.groupCutoff);
        }

        constexpr double DDMC_PI = 3.14159265358979323846;
        double const mu = SampleAsymptoticDDMCToIMCMu(this->dist(this->re));
        double const phiLeak = 2.0 * DDMC_PI * this->dist(this->re);
        Vector3D const dir = SampleHemisphereDirection(nOut, mu, phiLeak);

        materialParticle.location = leakFaceCenter;
        Vector3D const oldVelocityCoForPol = materialParticle.velocity;
        Vector3D const finalVelocityCoForPol = normalize(dir) * units::clight;
        materialParticle.velocity = finalVelocityCoForPol;

#ifdef MONTECARLO_POLARIZATION
        if(this->postProcess_.enabled && this->postProcess_.polarization.enabled)
        {
            materialParticle.velocity = oldVelocityCoForPol;

            double const scatOp = this->opacity->CalcScatteringOpacity(cell);
            double const sigmaReset = (1.0 - f) * data.sigmaEnergyAbs;
            IMCPolarization::ApplyAcceleratedPolarizationHistory(
                materialParticle,
                dtCo,
                scatOp,
                sigmaReset,
                finalVelocityCoForPol,
                this->postProcess_.polarization.manualScatteringsAfterAcceleration,
                this->postProcess_.polarization.depolarizationScatterings,
                this->re,
                this->dist);
        }
#endif

        materialParticle.velocity = finalVelocityCoForPol;

        assert(ScalarProd(materialParticle.velocity, nOut) > 0.0);

        size_t const emittedGroup = this->opacity->findGroup(
            materialParticle.frequency);
        this->recordDDMCDiagnosticEvent(
            DDMCDiagnosticEventKind::DDMCToIMC, cellIndex,
            chosen->nextCellIndex, chosen->faceIndex, emittedGroup,
            materialParticle.weight, data.groupCutoff,
            chosen->targetGroupCutoff,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
        this->tallyDDMCFaceFlux(cellIndex, *chosen, materialParticle.weight,
                                nOut, false);

        // Representation changes use the reconstructed face material frame.
        // Ordinary DDMC-to-DDMC leakage above deliberately has no frame
        // transformation; Doppler evolution remains operator split.
        // A resident DDMC packet has no microscopic direction before this
        // leakage event.  Sample the outgoing direction in the source-cell
        // comoving frame and transform with cell.velocity; transforming its
        // stale stored direction to an averaged face frame would be unphysical.

        functionality.change = MonteCarloParticleStatus::CELL_MOVE;
        functionality.nextCellIndex = chosen->nextCellIndex;
        ++this->ddmcLeakCount;
        ++this->ddmcTransportLeakCount;

    }
    else
    {
        if(!usePGRW)
        {
            functionality.change = MonteCarloParticleStatus::DONE;
            ++this->ddmcCensusCount;
            return finalizeAccelerationStep(false, true, false);
        }

        this->multigroupOpacity->GetCummulativeOpacity(cell);
        const auto &cumOp = this->multigroupOpacity->getCummulativeOpacity();
        double const cdfAtCutoff = cumOp[data.groupCutoff];
        double const cdfTotal = cumOp[ENERGY_GROUPS_NUM];
        if(cdfTotal > cdfAtCutoff)
        {
            double const lo = cdfAtCutoff / cdfTotal;
            double const xi = this->dist(this->re);
            materialParticle.frequency = this->multigroupOpacity->GetThermalEnergy(cell, lo + xi * (1.0 - lo));
        }
        else
        {
            materialParticle.frequency = std::nextafter(
                ComputationalCell3D::energyBoundaries[data.groupCutoff],
                std::numeric_limits<double>::max());
        }
        ClampFrequencyToBoundsDDMC(materialParticle.frequency);
        materialParticle.velocity = this->opacity->getRandomVelocity(cell);
#ifdef MONTECARLO_POLARIZATION
        if(this->postProcess_.enabled && this->postProcess_.polarization.enabled)
            IMCPolarization::ResetUnpolarized(materialParticle);
#endif
        ++this->ddmcUpscatterCount;

    }

    return finalizeAccelerationStep(false, false, false);
}

std::string RadiationIMC::getAccelerationDebugInfo(size_t cellIndex, double frequency) const
{
    std::ostringstream os;
    os.precision(17);
    if(!this->withDDMC)
        return std::string();

    os << " ddmc=on"
       << " ddmc_interface_incident=" << this->ddmcInterfaceIncidentCount
       << " ddmc_interface_admitted=" << this->ddmcInterfaceAdmissionCount
       << " ddmc_interface_reflected=" << this->ddmcInterfaceReflectionCount
       << " ddmc_interface_gu_applied=" << this->ddmcInterfaceMovingFactorCount
       << " ddmc_interface_gu_fallback=" << this->ddmcInterfaceMovingFallbackCount
       << " ddmc_interface_split_packets=" << this->ddmcInterfaceSplitPacketCount
       << " ddmc_interface_min_mu=" << this->ddmcInterfaceMinimumMu
       << " ddmc_interface_max_gu=" << this->ddmcInterfaceMaximumFactor
       << " ddmc_leak_reciprocity_checks=" << this->ddmcLeakReciprocityCheckCount
       << " ddmc_leak_reciprocity_max=" << this->ddmcLeakReciprocityResidualMax
       << " ddmc_leak_invalid_geometry=" << this->ddmcLeakInvalidGeometryCount;
    if(cellIndex >= this->ddmcCellData.size())
    {
        os << " reason=cell_out_of_range";
        return os.str();
    }

    DDMCCellData const &data = this->ddmcCellData[cellIndex];
    Vector3D const &cellVel = this->cells[cellIndex].velocity;
    double internalLeakRateSum = 0.0;
    double internalConductanceSum = 0.0;
    double ddmcChannelRateSum = 0.0;
    double transportChannelRateSum = 0.0;
    double boundaryRateSum = 0.0;
    size_t internalFaceCount = 0;
    size_t mixedFaceCount = 0;
    for(DDMCFaceLeak const &face : data.faceLeaks)
    {
        ddmcChannelRateSum += face.ddmcRate;
        transportChannelRateSum += face.transportRate;
        boundaryRateSum += face.boundaryRate;
        if(face.ddmcRate > 0.0 && face.transportRate > 0.0)
            ++mixedFaceCount;
        if(!face.targetDDMCEligible || !(face.internalRate > 0.0) ||
           !(face.conductance > 0.0))
            continue;
        internalLeakRateSum += face.internalRate;
        internalConductanceSum += face.conductance;
        ++internalFaceCount;
    }

    os << " useTransportVelocities=" << this->useTransportVelocities_
       << " ddmc_interface_bypass=" << this->ddmcInterfaceBypassCount
       << " ddmc_doppler_cutoff_exit=" << this->ddmcDopplerCutoffExitCount
       << " postProcess=" << this->postProcess_.enabled
       << " cell_vel=(" << cellVel.x << "," << cellVel.y << "," << cellVel.z << ")"
       << " freq_assumed_lab=" << frequency
       << " eligible=" << data.eligible
       << " observer_excluded=" << data.observerExcluded
       << " boundary_excluded=" << data.boundaryExcluded
       << " rigid_boundary_faces=" << data.rigidBoundaryFaceCount
       << " unsupported_boundary_faces=" << data.unsupportedBoundaryFaceCount
       << " first_unsupported_boundary_face=" << data.firstUnsupportedBoundaryFace
       << " sigmaT=" << data.sigmaT
       << " sigmaA=" << data.sigmaA
       << " sigmaEnergyAbs=" << data.sigmaEnergyAbs
       << " sigmaMomentum=" << data.sigmaMomentum
       << " sigmaDiffusion=" << data.sigmaDiffusion
       << " sigmaParticleGate=" << data.sigmaParticleGate
       << " sigmaGroupExit=" << data.sigmaGroupExit
       << " ddmc_gamma=" << data.gamma
       << " D=" << data.diffusionCoefficient
       << " leak_rate=" << data.totalLeakRate
       << " ddmc_internal_faces=" << internalFaceCount
       << " ddmc_internal_leak_rate_sum=" << internalLeakRateSum
       << " ddmc_internal_conductance_sum=" << internalConductanceSum
       << " ddmc_channel_rate_sum=" << ddmcChannelRateSum
       << " ddmc_transport_channel_rate_sum=" << transportChannelRateSum
       << " ddmc_boundary_rate_sum=" << boundaryRateSum
       << " ddmc_mixed_face_count=" << mixedFaceCount
       << " div_v=" << data.velocityDivergence
       << " max_face_dv_over_c=" << data.maxFaceVelocityJumpOverC
       << " faces=" << data.faceLeaks.size();

    bool const ddmcMomentumFeedbackEnabled =
        !this->noHydroFeedback && this->withHydro && !this->diffusionPressureGradient;
    Vector3D const expectedMomentum = ddmcMomentumFeedbackEnabled
        ? this->ddmcMaterialMomentumExchangeLab + this->ddmcFluxMomentumExchangeLab
        : this->ddmcAppliedMomentumExchangeLab;
    Vector3D const momentumResidual =
        this->ddmcAppliedMomentumExchangeLab - expectedMomentum;

    os << " ddmc_fallback_total=" << this->ddmcFallbackCount
       << " ddmc_leaks=" << this->ddmcLeakCount
       << " ddmc_resident_leaks=" << this->ddmcResidentLeakCount
       << " ddmc_transport_leaks=" << this->ddmcTransportLeakCount
       << " ddmc_remote_resident_leaks=" << this->ddmcRemoteResidentLeakCount
       << " ddmc_face_flux_energy=" << this->ddmcFaceFluxEnergy
       << " ddmc_face_flux_mpi_energy=" << this->ddmcFaceFluxMpiEnergy
       << " ddmc_mpi_face_flux_reductions=" << this->ddmcMpiFaceFluxReductionCount
       << " ddmc_interface_flux_tallies=" << this->ddmcInterfaceFluxTallyCount
       << " ddmc_boundary_flux_tallies=" << this->ddmcBoundaryFluxTallyCount
       << " ddmc_local_face_pair_checks=" << this->ddmcLocalFaceFluxPairCheckCount
       << " ddmc_local_face_pair_residual_max=" << this->ddmcLocalFaceFluxPairResidualMax
       << " ddmc_momentum_cells=" << this->ddmcMomentumFeedbackCount
       << " ddmc_momentum_matrix_fallbacks=" << this->ddmcMomentumMatrixFallbackCount
       << " ddmc_material_energy_co=" << this->ddmcMaterialEnergyExchangeCo
       << " ddmc_material_energy_lab_est=" << this->ddmcMaterialEnergyExchangeLab
       << " ddmc_material_momentum_lab_est=("
       << this->ddmcMaterialMomentumExchangeLab.x << ","
       << this->ddmcMaterialMomentumExchangeLab.y << ","
       << this->ddmcMaterialMomentumExchangeLab.z << ")"
       << " ddmc_flux_momentum_lab_est=("
       << this->ddmcFluxMomentumExchangeLab.x << ","
       << this->ddmcFluxMomentumExchangeLab.y << ","
       << this->ddmcFluxMomentumExchangeLab.z << ")"
       << " ddmc_applied_momentum_lab=("
       << this->ddmcAppliedMomentumExchangeLab.x << ","
       << this->ddmcAppliedMomentumExchangeLab.y << ","
       << this->ddmcAppliedMomentumExchangeLab.z << ")"
       << " ddmc_momentum_feedback_enabled=" << ddmcMomentumFeedbackEnabled
       << " ddmc_momentum_source_residual="
       << abs(momentumResidual)
       << " ddmc_w_over_w0_count=" << this->ddmcWeightRatioCount
       << " ddmc_w_over_w0_mean="
       << (this->ddmcWeightRatioCount > 0
           ? this->ddmcWeightRatioSum / static_cast<double>(this->ddmcWeightRatioCount)
           : 0.0)
       << " ddmc_w_over_w0_p99=" << Percentile(this->ddmcWeightRatioSamples, 0.99)
       << " ddmc_w_over_w0_max=" << this->ddmcWeightRatioMax
       << " ddmc_w_over_w0_samples_dropped=" << this->ddmcWeightRatioSamplesDropped
       << " ddmc_w_over_w0_gt8=" << this->ddmcWeightRatioOutlierCount
       << " ddmc_observer_energy_only_tallies=" << this->ddmcObserverEnergyOnlyTallyCount
       << " ddmc_moving_medium_updates=" << this->ddmcMovingMediumUpdateCount
       << " ddmc_face_frame_shifts=" << this->ddmcFaceFrameShiftCount
       << " ddmc_max_moving_medium_log_shift=" << this->ddmcMaxMovingMediumLogShift
       << " ddmc_max_face_frame_log_shift=" << this->ddmcMaxFaceFrameLogShift;

    if(this->multigroupOpacity && this->ddmcUseMultigroupPGRW)
    {
        double coFreq = frequency;
        ClampFrequencyToBoundsDDMC(coFreq);
        double cutoff = (data.groupCutoff <= ENERGY_GROUPS_NUM)
            ? ComputationalCell3D::energyBoundaries[data.groupCutoff]
            : ComputationalCell3D::energyBoundaries[ENERGY_GROUPS_NUM];
        os << " group_cutoff=" << data.groupCutoff
           << " freq_diffusive=" << (data.groupCutoff > 0 && coFreq < cutoff);
    }
    return os.str();
}
