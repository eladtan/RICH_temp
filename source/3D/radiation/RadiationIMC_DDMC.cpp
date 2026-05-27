#include "RadiationIMC.hpp"
#include "SphericalObserver.hpp"
#include "IMCPolarization.hpp"
#include "Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>

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
                data.diffusionCoefficient = (units::clight / 3.0) * sumBgOverSigTDiff / totalBgDiff;
                data.gamma = (totalSigABgAll > 0.0) ? sumBgSigADiff / totalSigABgAll : 1.0;
                data.eligible = (data.sigmaT > 0.0 && data.diffusionCoefficient > 0.0);
            }
        }
        else
        {
            data.sigmaA = this->planckOpacities[i];
            data.sigmaT = data.sigmaA + scatOp;
            data.diffusionCoefficient = (data.sigmaT > 0.0) ? units::clight / (3.0 * data.sigmaT) : 0.0;
            data.gamma = 1.0;
            data.eligible = (data.sigmaT * meanChordLength >= this->ddmcMinCellOpticalDepth
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
                data.faceLeaks.push_back(faceLeak);
                data.totalLeakRate += rate;
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

bool RadiationIMC::tryDDMCStep(Particle &particle, Functionality &functionality, double dopplerShift)
{
    (void)dopplerShift;

    size_t const cellIndex = particle.cellIndex;
    if(cellIndex >= this->ddmcCellData.size())
        return false;

    DDMCCellData const &data = this->ddmcCellData[cellIndex];
    if(!data.eligible || data.totalLeakRate <= 0.0 || data.faceLeaks.empty()
       || data.sigmaT <= 0.0 || data.diffusionCoefficient <= 0.0)
        return false;

    ComputationalCell3D &cell = this->cells[cellIndex];

    bool const useVelocityTransport = this->useTransportVelocities_;
    double const gammaCell = useVelocityTransport ? CellGamma(cell.velocity) : 1.0;

    if(!(gammaCell > 0.0) || !std::isfinite(gammaCell))
    {
        ++this->ddmcFallbackCount;
        return false;
    }

    Vector3D const oldLabVelocity = particle.velocity;
    double const oldLabWeight = particle.weight;

    Particle materialParticle = particle;
    if(useVelocityTransport)
    {
        LorentzTransformation(materialParticle, cell.velocity);
        if(this->multigroupOpacity)
            ClampFrequencyToBoundsDDMC(materialParticle.frequency);
    }

    const auto &normals = this->gridData.normalsOfCells[cellIndex];
    const auto &facePoints = this->gridData.pointsOnFaces[cellIndex];
    double Ro = std::numeric_limits<double>::max();
    for(size_t f = 0; f < normals.size(); ++f)
    {
        double const d = ScalarProd(particle.location - facePoints[f], normals[f]);
        Ro = std::min(Ro, d);
    }
    if(!(Ro > 0.0) || Ro * data.sigmaT < this->ddmcMinParticleOpticalDepth)
    {
        ++this->ddmcFallbackCount;
        return false;
    }

    bool const usePGRW = (this->multigroupOpacity != nullptr && this->ddmcUseMultigroupPGRW);
    if(usePGRW)
    {
        if(data.groupCutoff == 0 || data.groupCutoff > ENERGY_GROUPS_NUM)
        {
            ++this->ddmcFallbackCount;
            return false;
        }
        double coFreq = materialParticle.frequency;
        ClampFrequencyToBoundsDDMC(coFreq);
        if(coFreq >= ComputationalCell3D::energyBoundaries[data.groupCutoff])
        {
            ++this->ddmcFallbackCount;
            return false;
        }
    }

    double const f = this->factorFleck[cellIndex];
    double const upscatterRateCo = (usePGRW && data.gamma < 1.0 && data.sigmaA > 0.0 && f > 0.0)
        ? units::clight * (1.0 - f) * data.sigmaA * (1.0 - data.gamma)
        : 0.0;
    double const eventRateCo = data.totalLeakRate + upscatterRateCo;
    if(!(eventRateCo > 0.0) || !std::isfinite(eventRateCo))
    {
        ++this->ddmcFallbackCount;
        return false;
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

    double const absRateCo = data.sigmaA * f * units::clight;
    double const oldCoWeight = materialParticle.weight;
    double const expFactorCo = std::expm1(-dtCo * absRateCo);
    double const absorbedCo = -expFactorCo * oldCoWeight;

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
            this->observer_->addAbsorbedEnergy(absorbed);
    }

    materialParticle.timeLeft = particle.timeLeft - dtLab;
    if(materialParticle.timeLeft < 0.0 && materialParticle.timeLeft > -1e-12)
        materialParticle.timeLeft = 0.0;

    ++this->ddmcStepCount;

    auto finalizeAccelerationStep = [&](bool remove) -> bool {
        if(remove)
        {
            functionality.change = MonteCarloParticleStatus::REMOVE;

            if(useVelocityTransport && !this->diffusionPressureGradient && !this->noHydroFeedback)
            {
                this->conserved[cellIndex].momentum +=
                    (oldLabWeight * oldLabVelocity) * units::inv_clight2;
            }

            return true;
        }

        Particle finalLabParticle = materialParticle;

        if(useVelocityTransport)
        {
            LorentzTransformation(finalLabParticle, -1 * cell.velocity);
            if(this->multigroupOpacity)
                ClampFrequencyToBoundsDDMC(finalLabParticle.frequency);
        }

        particle.location  = finalLabParticle.location;
        particle.velocity  = finalLabParticle.velocity;
        particle.frequency = finalLabParticle.frequency;
        particle.weight    = finalLabParticle.weight;
        particle.timeLeft  = finalLabParticle.timeLeft;
#ifdef MONTECARLO_POLARIZATION
        particle.stokesQ = finalLabParticle.stokesQ;
        particle.stokesU = finalLabParticle.stokesU;
        particle.polarizationBasis = finalLabParticle.polarizationBasis;
        particle.polarizationInitialized = finalLabParticle.polarizationInitialized;
        if(particle.polarizationInitialized)
            particle.polarizationBasis =
                IMCPolarization::ProjectBasisToDirection(particle.polarizationBasis,
                                                         particle.velocity);
#endif

        if(useVelocityTransport && !this->diffusionPressureGradient && !this->noHydroFeedback)
        {
            this->conserved[cellIndex].momentum +=
                (oldLabWeight * oldLabVelocity - particle.weight * particle.velocity)
                * units::inv_clight2;
        }

        return true;
    };

    bool removeParticle = false;

    double const lowWeightCutoff = this->postProcess_.enabled ? 1e-8 : 1e-3;
    double const initialCoWeightApprox = useVelocityTransport
        ? particle.initialWeight * DopplerShift(particle, cell.velocity)
        : particle.initialWeight;

    if(std::abs(materialParticle.weight) < std::abs(initialCoWeightApprox) * lowWeightCutoff)
    {
        removeParticle = true;

        if(this->postProcess_.enabled && this->observer_)
            this->observer_->addCutoffEnergy(materialParticle.weight);

        if(!this->noHydroFeedback)
            this->conserved[cellIndex].internal_energy += materialParticle.weight;
    }

    if(removeParticle)
        return finalizeAccelerationStep(true);

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
            double const sigmaReset = (1.0 - f) * data.sigmaA;
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
        return finalizeAccelerationStep(false);
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
            return finalizeAccelerationStep(false);
        }

        Vector3D const leakFaceCenter = this->grid.FaceCM(chosen->faceIndex);

        Vector3D nOut = this->grid.Normal(chosen->faceIndex);
        if(abs(nOut) <= 0.0)
        {
            ++this->ddmcFallbackCount;
            functionality.change = MonteCarloParticleStatus::DONE;
            return finalizeAccelerationStep(false);
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
            double const sigmaReset = (1.0 - f) * data.sigmaA;
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

        functionality.change = MonteCarloParticleStatus::CELL_MOVE;
        functionality.nextCellIndex = chosen->nextCellIndex;
        ++this->ddmcLeakCount;
    }
    else
    {
        if(!usePGRW)
        {
            functionality.change = MonteCarloParticleStatus::DONE;
            ++this->ddmcCensusCount;
            return finalizeAccelerationStep(false);
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

    return finalizeAccelerationStep(false);
}

std::string RadiationIMC::getAccelerationDebugInfo(size_t cellIndex, double frequency) const
{
    std::ostringstream os;
    if(!this->withDDMC)
        return std::string();

    os << " ddmc=off";
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
       << " sigmaT=" << data.sigmaT
       << " sigmaA=" << data.sigmaA
       << " D=" << data.diffusionCoefficient
       << " leak_rate=" << data.totalLeakRate
       << " faces=" << data.faceLeaks.size();

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
