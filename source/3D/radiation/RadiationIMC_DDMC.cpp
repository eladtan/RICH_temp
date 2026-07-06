#include "RadiationIMC.hpp"
#include "SphericalObserver.hpp"
#include "IMCPolarization.hpp"
#include "Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <vector>
#ifdef RICH_MPI
#include "mpi/mpi_commands.hpp"
#endif

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

    inline double HarmonicMean(double a, double b)
    {
        return (a > 0.0 && b > 0.0) ? (2.0 * a * b / (a + b)) : std::max(a, b);
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

            for(size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
            {
                double const a = ComputationalCell3D::energyBoundaries[g] / kT;
                double const b = ComputationalCell3D::energyBoundaries[g + 1] / kT;
                double const Bg = planck_integral::planck_integral(a, b);
                double const sigA_g = this->opacity->CalcAbsorptionOpacity(cell, energyCenters[g]);
                double const sigT_g = sigA_g + scatOp;

                totalSigABgAll += sigA_g * Bg;

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

    for(size_t i = 0; i < Ncells; ++i)
    {
        DDMCCellData &data = this->ddmcCellData[i];
        if(!data.eligible)
            continue;

        double const volume = this->grid.GetVolume(i);
        Vector3D const cellCenter = this->grid.GetMeshPoint(i);
        for(size_t faceIdx : this->grid.GetCellFaces(i))
        {
            const auto &neighbors = this->grid.GetFaceNeighbors(faceIdx);
            if(neighbors.first != i && neighbors.second != i)
                continue;
            size_t const nextCellIndex = (neighbors.first == i) ? neighbors.second : neighbors.first;
            if(this->grid.IsPointOutsideBox(nextCellIndex))
            {
                DDMCBoundaryFaceBehavior const faceBehavior =
                    this->boundary->getDDMCBoundaryFaceBehavior(
                        faceIdx, i, nextCellIndex);

                if(faceBehavior == DDMCBoundaryFaceBehavior::ReflectingRigid)
                {
                    ++data.rigidBoundaryFaceCount;
                    continue;
                }

                ++data.unsupportedBoundaryFaceCount;
                if(data.firstUnsupportedBoundaryFace ==
                   std::numeric_limits<size_t>::max())
                {
                    data.firstUnsupportedBoundaryFace = faceIdx;
                }
                data.boundaryExcluded = true;
                continue;
            }

            Vector3D normal = this->grid.Normal(faceIdx);
            if(abs(normal) <= 0.0)
                continue;
            normal = normalize(normal);

            Vector3D const faceCenter = this->grid.FaceCM(faceIdx);
            double faceDistance = std::abs(ScalarProd(faceCenter - cellCenter, normal));
            if(faceDistance <= 0.0 && nextCellIndex < this->grid.getMeshPoints().size())
                faceDistance = 0.5 * std::abs(ScalarProd(this->grid.GetMeshPoint(nextCellIndex) - cellCenter, normal));
            if(faceDistance <= 0.0)
                continue;

            double diffusionFace = data.diffusionCoefficient;
            if(nextCellIndex < Ncells && this->ddmcCellData[nextCellIndex].diffusionCoefficient > 0.0)
                diffusionFace = HarmonicMean(data.diffusionCoefficient, this->ddmcCellData[nextCellIndex].diffusionCoefficient);

            double const rate = diffusionFace * this->grid.GetArea(faceIdx) / (volume * faceDistance);
            if(rate > 0.0 && std::isfinite(rate))
            {
                DDMCFaceLeak faceLeak;
                faceLeak.faceIndex = faceIdx;
                faceLeak.nextCellIndex = nextCellIndex;
                faceLeak.rate = rate;
                faceLeak.area = this->grid.GetArea(faceIdx);
                faceLeak.distance = faceDistance;
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
}

double RadiationIMC::computeMinSignedDistanceToAllCellFaces(
    size_t cellIndex,
    Vector3D const &location) const
{
    if(cellIndex >= this->gridData.normalsOfCells.size() ||
       cellIndex >= this->gridData.pointsOnFaces.size())
        return -std::numeric_limits<double>::infinity();

    const auto &normals = this->gridData.normalsOfCells[cellIndex];
    const auto &facePoints = this->gridData.pointsOnFaces[cellIndex];

    if(normals.size() != facePoints.size() || normals.empty())
        return -std::numeric_limits<double>::infinity();

    double minSignedDistance = std::numeric_limits<double>::max();
    for(size_t f = 0; f < normals.size(); ++f)
    {
        double const d = ScalarProd(location - facePoints[f], normals[f]);
        minSignedDistance = std::min(minSignedDistance, d);
    }

    return minSignedDistance;
}

double RadiationIMC::computeDDMCGeometryTolerance(size_t cellIndex) const
{
    double scale = 0.0;

    if(cellIndex < this->grid.GetPointNo())
    {
        Vector3D const cellCenter = this->grid.GetMeshPoint(cellIndex);
        for(size_t faceIdx : this->grid.GetCellFaces(cellIndex))
            scale = std::max(scale, abs(this->grid.FaceCM(faceIdx) - cellCenter));
    }

    if(!(scale > 0.0) || !std::isfinite(scale))
        scale = 1.0;

    return std::max(1e-12 * scale, 1e-14);
}

double RadiationIMC::computeMinDistanceToDDMCLeakFaces(
    size_t cellIndex,
    Vector3D const &location,
    DDMCCellData const &data) const
{
    (void)cellIndex;

    if(data.faceLeaks.empty())
        return std::numeric_limits<double>::infinity();

    double minDistance = std::numeric_limits<double>::max();

    for(DDMCFaceLeak const &faceLeak : data.faceLeaks)
    {
        if(faceLeak.faceIndex == std::numeric_limits<size_t>::max())
            continue;

        Vector3D normal = this->grid.Normal(faceLeak.faceIndex);
        double const normalMag = abs(normal);
        if(!(normalMag > 0.0) || !std::isfinite(normalMag))
            continue;

        normal = normal / normalMag;

        Vector3D const faceCenter = this->grid.FaceCM(faceLeak.faceIndex);
        double const distance = std::abs(ScalarProd(location - faceCenter, normal));

        if(std::isfinite(distance))
            minDistance = std::min(minDistance, distance);
    }

    return minDistance;
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
        auto sampleCellLocation = [&]() {
            Vector3D const center = this->grid.GetMeshPoint(cellIndex);
            double radius = 0.0;
            for(size_t faceIdx : this->grid.GetCellFaces(cellIndex))
                radius = std::max(radius, abs(this->grid.FaceCM(faceIdx) - center));
            if(!(radius > 0.0) || !std::isfinite(radius))
                radius = std::cbrt(std::max(this->grid.GetVolume(cellIndex), 0.0));
            if(!(radius > 0.0) || !std::isfinite(radius))
                return center;

            double const sampleHalfWidth = 2.0 * radius;
            for(size_t attempt = 0; attempt < 256; ++attempt)
            {
                Vector3D const offset(
                    (2.0 * this->dist(this->re) - 1.0) * sampleHalfWidth,
                    (2.0 * this->dist(this->re) - 1.0) * sampleHalfWidth,
                    (2.0 * this->dist(this->re) - 1.0) * sampleHalfWidth);
                Vector3D const candidate = center + offset;
                if(this->grid.IsPointOutsideBox(candidate))
                    continue;
                size_t const containingCell = this->grid.GetContainingCell(candidate);
                if(containingCell == cellIndex)
                    return candidate;
            }
            return center;
        };

        auto sampleIsotropicComovingVelocity = [&]() {
            constexpr double DDMC_PI = 3.14159265358979323846;
            double const mu = 2.0 * this->dist(this->re) - 1.0;
            double const phi = 2.0 * DDMC_PI * this->dist(this->re);
            double const sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
            return units::clight * Vector3D(
                sinTheta * std::cos(phi),
                sinTheta * std::sin(phi),
                mu);
        };

        particle.location = sampleCellLocation();
        particle.velocity = sampleIsotropicComovingVelocity();
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
        particle.initialWeight = std::abs(particle.weight);
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
    if(continuingDDMC)
    {
        materialParticle.location = this->grid.GetMeshPoint(cellIndex);
    }
    else if(useVelocityTransport)
    {
        LabToComovingPacket(materialParticle, cell.velocity);
        if(this->multigroupOpacity)
            ClampFrequencyToBoundsDDMC(materialParticle.frequency);
    }
    materialParticle.ddmcMode = true;
    materialParticle.ddmcCellResident = true;
    materialParticle.ddmcComovingFrame = true;
    if(!continuingDDMC)
        materialParticle.initialWeight = std::abs(materialParticle.weight);

    if(!continuingDDMC)
    {
        double const insideDistanceAllFaces =
            this->computeMinSignedDistanceToAllCellFaces(cellIndex, particle.location);

        double const insideTolerance =
            this->computeDDMCGeometryTolerance(cellIndex);

        if(!std::isfinite(insideDistanceAllFaces) ||
           insideDistanceAllFaces < -insideTolerance)
        {
            ++this->ddmcFallbackCount;
            ++this->ddmcFallbackOutsideCellCount;
            return false;
        }

        double const leakDistanceActiveFaces =
            this->computeMinDistanceToDDMCLeakFaces(cellIndex, particle.location, data);

        if(!std::isfinite(leakDistanceActiveFaces) ||
           leakDistanceActiveFaces == std::numeric_limits<double>::max())
        {
            ++this->ddmcFallbackCount;
            ++this->ddmcFallbackInvalidLeakFaceDistanceCount;
            return false;
        }

        if(leakDistanceActiveFaces * data.sigmaParticleGate < this->ddmcMinParticleOpticalDepth)
        {
            ++this->ddmcFallbackCount;
            ++this->ddmcFallbackLeakFaceDistanceCount;
            return false;
        }

        materialParticle.location = this->grid.GetMeshPoint(cellIndex);
    }

    bool const usePGRW = (this->multigroupOpacity != nullptr && this->ddmcUseMultigroupPGRW);
    auto frequencyFitsDDMCCellAt = [&](DDMCCellData const &cellData, double frequency) -> bool {
        if(!usePGRW)
            return true;
        if(cellData.groupCutoff == 0 || cellData.groupCutoff > ENERGY_GROUPS_NUM)
            return false;
        double coFreq = frequency;
        ClampFrequencyToBoundsDDMC(coFreq);
        return coFreq < ComputationalCell3D::energyBoundaries[cellData.groupCutoff];
    };
    auto frequencyFitsDDMCCell = [&](DDMCCellData const &cellData) -> bool {
        return frequencyFitsDDMCCellAt(cellData, materialParticle.frequency);
    };
    if(usePGRW)
    {
        if(data.groupCutoff == 0 || data.groupCutoff > ENERGY_GROUPS_NUM)
        {
            ++this->ddmcFallbackCount;
            if(continuingDDMC)
                convertResidentDDMCToTransport();
            return false;
        }
        double coFreq = materialParticle.frequency;
        ClampFrequencyToBoundsDDMC(coFreq);
        if(coFreq >= ComputationalCell3D::energyBoundaries[data.groupCutoff])
        {
            ++this->ddmcFallbackCount;
            if(continuingDDMC)
                convertResidentDDMCToTransport();
            return false;
        }
    }

    double const f = this->factorFleck[cellIndex];
    double const upscatterRateCo = (usePGRW && data.gamma < 1.0 && data.sigmaEnergyAbs > 0.0 && f > 0.0)
        ? units::clight * (1.0 - f) * data.sigmaEnergyAbs * (1.0 - data.gamma)
        : 0.0;
    double const eventRateCo = data.totalLeakRate + upscatterRateCo;
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

    double const dtCo = std::min(tEventCo, tCensusCo);

    // Level-1 mixed-frame approximation: dtLab = gammaCell * dtCo.
    // This intentionally ignores the event displacement term
    // gammaCell * dot(cell.velocity, dxCo) / c^2.
    // Use Level 2 before relying on high-v/c DDMC results.
    double dtLab = useVelocityTransport ? gammaCell * dtCo : dtCo;
    bool const censusEvent = (tCensusCo <= tEventCo);
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
        double freqForGroup = materialParticle.frequency;
        ClampFrequencyToBoundsDDMC(freqForGroup);
        size_t const g = this->opacity->findGroup(freqForGroup);
        this->Eg_time_avg[cellIndex][g] += integratedCo;
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

    auto storeDDMCResidentParticle = [&](size_t residentCellIndex) {
        particle.location  = this->grid.GetMeshPoint(residentCellIndex);
        particle.velocity  = materialParticle.velocity;
        particle.cellIndex = residentCellIndex;
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
        particle.initialWeight = std::abs(transportParticle.weight);
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
            ComovingToLabPacket(finalLabParticle, cell.velocity);
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

    if(std::abs(materialParticle.weight) < std::abs(initialCoWeight) * lowWeightCutoff)
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
        functionality.change = MonteCarloParticleStatus::DONE;
        ++this->ddmcCensusCount;
        return finalizeAccelerationStep(false, true, false);
    }

    double eventPick = this->dist(this->re) * eventRateCo;
    if(eventPick <= data.totalLeakRate)
    {
        double facePick = this->dist(this->re) * data.totalLeakRate;
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
            chosen->nextCellIndex < this->ddmcCellData.size() &&
            this->ddmcCellData[chosen->nextCellIndex].eligible &&
            this->ddmcCellData[chosen->nextCellIndex].totalLeakRate > 0.0;

        Particle targetMaterialParticle = materialParticle;
        if(localDDMCNeighborCandidate)
        {
            targetMaterialParticle.ddmcMode = true;
            targetMaterialParticle.ddmcCellResident = true;
            targetMaterialParticle.ddmcComovingFrame = true;
        }

        bool const localDDMCNeighbor =
            localDDMCNeighborCandidate &&
            frequencyFitsDDMCCellAt(this->ddmcCellData[chosen->nextCellIndex],
                                    targetMaterialParticle.frequency);

        bool const remoteDDMCNeighbor =
            !localDDMCNeighbor &&
            chosen->nextCellIndex >= this->ddmcCellData.size() &&
            chosen->nextCellIndex < this->grid.getMeshPoints().size();

        if(localDDMCNeighbor)
        {
            this->tallyDDMCFaceFlux(cellIndex, *chosen, materialParticle.weight,
                                    nOut, true);
            materialParticle = targetMaterialParticle;
            materialParticle.cellIndex = chosen->nextCellIndex;
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
            this->tallyDDMCFaceFlux(cellIndex, *chosen, materialParticle.weight,
                                    nOut, true);
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

        Vector3D helper = (std::abs(nOut.x) < 0.9)
            ? Vector3D(1.0, 0.0, 0.0)
            : Vector3D(0.0, 1.0, 0.0);
        Vector3D e1 = normalize(helper - ScalarProd(helper, nOut) * nOut);
        Vector3D e2 = CrossProduct(nOut, e1);
        if(abs(e2) > 0.0)
            e2 = normalize(e2);

        constexpr double DDMC_PI = 3.14159265358979323846;
        double const xiMu = std::min(1.0, PositiveRandom(this->dist(this->re)));
        double const mu = std::sqrt(xiMu);
        double const phiLeak = 2.0 * DDMC_PI * this->dist(this->re);
        double const sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));

        Vector3D dir = mu * nOut
            + sinTheta * std::cos(phiLeak) * e1
            + sinTheta * std::sin(phiLeak) * e2;

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

        this->tallyDDMCFaceFlux(cellIndex, *chosen, materialParticle.weight,
                                nOut, false);

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
    if(!this->withDDMC)
        return std::string();

    os << " ddmc=on";
    if(cellIndex >= this->ddmcCellData.size())
    {
        os << " reason=cell_out_of_range";
        return os.str();
    }

    DDMCCellData const &data = this->ddmcCellData[cellIndex];
    Vector3D const &cellVel = this->cells[cellIndex].velocity;
    os << " useTransportVelocities=" << this->useTransportVelocities_
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
       << " D=" << data.diffusionCoefficient
       << " leak_rate=" << data.totalLeakRate
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
       << " ddmc_max_face_frame_log_shift=" << this->ddmcMaxFaceFrameLogShift
       << " ddmc_fallback_outside_cell=" << this->ddmcFallbackOutsideCellCount
       << " ddmc_fallback_leak_distance=" << this->ddmcFallbackLeakFaceDistanceCount
       << " ddmc_fallback_invalid_leak_face=" << this->ddmcFallbackInvalidLeakFaceDistanceCount;

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
