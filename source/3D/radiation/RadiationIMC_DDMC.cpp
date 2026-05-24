#include "RadiationIMC.hpp"
#include "SphericalObserver.hpp"
#include "Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include <algorithm>
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
            double const distToObserver = abs(cellCenter - this->observer_->getCenter());
            double const obsR = this->observer_->getRadius();
            if(distToObserver + meanChordLength >= obsR && distToObserver - meanChordLength <= obsR)
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
    if(this->withHydro && !this->MMC)
    {
        ++this->ddmcFallbackCount;
        return false;
    }

    size_t const cellIndex = particle.cellIndex;
    if(cellIndex >= this->ddmcCellData.size())
        return false;

    DDMCCellData const &data = this->ddmcCellData[cellIndex];
    if(!data.eligible || data.totalLeakRate <= 0.0 || data.faceLeaks.empty()
       || data.sigmaT <= 0.0 || data.diffusionCoefficient <= 0.0)
        return false;

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
        if(data.groupCutoff == 0)
        {
            ++this->ddmcFallbackCount;
            return false;
        }
        double coFreq = particle.frequency;
        if(this->withHydro && !this->MMC)
            coFreq *= dopplerShift;
        ClampFrequencyToBoundsDDMC(coFreq);
        if(coFreq >= ComputationalCell3D::energyBoundaries[data.groupCutoff])
        {
            ++this->ddmcFallbackCount;
            return false;
        }
    }

    double const f = this->factorFleck[cellIndex];
    double const upscatterRate = (usePGRW && data.gamma < 1.0 && data.sigmaA > 0.0 && f > 0.0)
        ? units::clight * (1.0 - f) * data.sigmaA * (1.0 - data.gamma)
        : 0.0;
    double const eventRate = data.totalLeakRate + upscatterRate;
    if(!(eventRate > 0.0) || !std::isfinite(eventRate))
    {
        ++this->ddmcFallbackCount;
        return false;
    }

    double const tEvent = -std::log(PositiveRandom(this->dist(this->re))) / eventRate;
    double const tCensus = particle.timeLeft;
    double const dt = std::min(tEvent, tCensus);

    Vector3D oldVelocity = particle.velocity;
    double const oldWeight = particle.weight;
    double const absRate = data.sigmaA * f * units::clight;
    double const expFactor = std::expm1(-dt * absRate);
    if(!this->noHydroFeedback)
        this->conserved[cellIndex].internal_energy += -expFactor * particle.weight;

    if(absRate > 0.0)
    {
        double const integrated = particle.weight * expFactor * (-1.0 / absRate);
        this->Erad_time_avg[cellIndex] += integrated;
        if(this->withEgTimeAvg && this->multigroupOpacity)
        {
            size_t const g = this->opacity->findGroup(particle.frequency);
            this->Eg_time_avg[cellIndex][g] += integrated;
        }
    }
    else
    {
        this->Erad_time_avg[cellIndex] += particle.weight * dt;
        if(this->withEgTimeAvg && this->multigroupOpacity)
        {
            size_t const g = this->opacity->findGroup(particle.frequency);
            this->Eg_time_avg[cellIndex][g] += particle.weight * dt;
        }
    }

    particle.weight *= 1.0 + expFactor;
    if(this->postProcess_.enabled && this->observer_)
    {
        double const absorbed = oldWeight - particle.weight;
        if(absorbed > 0.0)
            this->observer_->addAbsorbedEnergy(absorbed);
    }

    particle.timeLeft -= dt;
    ++this->ddmcStepCount;

    double const lowWeightCutoff = this->postProcess_.enabled ? 1e-4 : 1e-3;
    if(std::abs(particle.weight) < particle.initialWeight * lowWeightCutoff)
    {
        if(this->postProcess_.enabled && this->observer_)
            this->observer_->addCutoffEnergy(particle.weight);
        functionality.change = MonteCarloParticleStatus::REMOVE;
        if(!this->noHydroFeedback)
            this->conserved[cellIndex].internal_energy += particle.weight;
        return true;
    }

    if(tCensus <= tEvent)
    {
        functionality.change = MonteCarloParticleStatus::DONE;
        ++this->ddmcCensusCount;
        return true;
    }

    double eventPick = this->dist(this->re) * eventRate;
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
            functionality.change = MonteCarloParticleStatus::DONE;
            ++this->ddmcCensusCount;
            return true;
        }

        particle.location = this->grid.FaceCM(chosen->faceIndex);
        particle.velocity = this->opacity->getRandomVelocity(this->cells[cellIndex]);
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
            return true;
        }

        ComputationalCell3D &cell = this->cells[cellIndex];
        this->multigroupOpacity->GetCummulativeOpacity(cell);
        const auto &cumOp = this->multigroupOpacity->getCummulativeOpacity();
        double const cdfAtCutoff = cumOp[data.groupCutoff];
        double const cdfTotal = cumOp[ENERGY_GROUPS_NUM];
        if(cdfTotal > cdfAtCutoff)
        {
            double const lo = cdfAtCutoff / cdfTotal;
            double const xi = this->dist(this->re);
            particle.frequency = this->multigroupOpacity->GetThermalEnergy(cell, lo + xi * (1.0 - lo));
        }
        else
        {
            particle.frequency = std::nextafter(
                ComputationalCell3D::energyBoundaries[data.groupCutoff],
                std::numeric_limits<double>::max());
        }
        particle.velocity = this->opacity->getRandomVelocity(cell);
        ++this->ddmcUpscatterCount;
    }

    if(!this->diffusionPressureGradient && !this->noHydroFeedback)
        this->conserved[cellIndex].momentum += (oldWeight * oldVelocity - particle.weight * particle.velocity) * units::inv_clight2;

    return true;
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
    os << " eligible=" << data.eligible
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
