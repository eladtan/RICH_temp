#include "RadiationIMC.hpp"
#include "SphericalObserver.hpp"
#include "Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include <iostream>

namespace {
    inline void ClampFrequencyToBounds(double &frequency)
    {
        frequency = std::clamp(frequency,
            ComputationalCell3D::energyBoundaries[0],
            ComputationalCell3D::energyBoundaries[ENERGY_GROUPS_NUM]);
    }

    constexpr double RW_PI = 3.14159265358979323846;

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
}

void RadiationIMC::precomputeRandomWalkData()
{
    size_t Ncells = this->grid.GetPointNo();
    this->rwCellEligible.assign(Ncells, false);
    this->rwCellTotalOpacity.assign(Ncells, 0.0);
    if(this->multigroupOpacity)
        this->rwCellData.resize(Ncells);

    for(size_t i = 0; i < Ncells; i++)
    {
        const ComputationalCell3D &cell = this->cells[i];
        double scatOp = this->opacity->CalcScatteringOpacity(cell);
        double sigmaT_gray = this->planckOpacities[i] + scatOp;
        this->rwCellTotalOpacity[i] = sigmaT_gray;

        double surfaceArea = 0.0;
        for(size_t faceIdx : this->grid.GetCellFaces(i))
            surfaceArea += this->grid.GetArea(faceIdx);
        double volume = this->grid.GetVolume(i);
        double meanChordLength = (surfaceArea > 0) ? 4.0 * volume / surfaceArea : 0.0;

        if(!this->multigroupOpacity)
        {
            this->rwCellEligible[i] = (sigmaT_gray * meanChordLength >= this->rwMinCellOpticalDepth);
        }
        else
        {
            const auto &energyCenters = this->multigroupOpacity->getEnergyCenters();
            double kT = units::k_boltz * cell.temperature;

            double totalSigABgAll = 0;
            double totalBgDiff = 0, sumBgSigADiff = 0, sumBgSigTDiff = 0, sumBgOverSigTDiff = 0;
            size_t cutoff = 0;
            bool foundNonDiffusive = false;

            for(size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
            {
                double a = ComputationalCell3D::energyBoundaries[g] / kT;
                double b = ComputationalCell3D::energyBoundaries[g + 1] / kT;
                double Bg = planck_integral::planck_integral(a, b);
                double sigA_g = this->opacity->CalcAbsorptionOpacity(cell, energyCenters[g]);
                double sigT_g = sigA_g + scatOp;

                totalSigABgAll += sigA_g * Bg;

                if(!foundNonDiffusive && sigT_g * meanChordLength >= this->rwMinCellOpticalDepth)
                {
                    cutoff = g + 1;
                    totalBgDiff += Bg;
                    sumBgSigADiff += Bg * sigA_g;
                    sumBgSigTDiff += Bg * sigT_g;
                    if(sigT_g > 0)
                        sumBgOverSigTDiff += Bg / sigT_g;
                }
                else
                {
                    foundNonDiffusive = true;
                }
            }

            PGRWCellData &data = this->rwCellData[i];
            if(cutoff > 0 && totalBgDiff > 0)
            {
                data.groupCutoff = cutoff;
                data.sigmaA_bar = sumBgSigADiff / totalBgDiff;
                data.sigmaT_bar = sumBgSigTDiff / totalBgDiff;
                data.D = (units::clight / 3.0) * sumBgOverSigTDiff / totalBgDiff;
                data.gamma = (totalSigABgAll > 0) ? sumBgSigADiff / totalSigABgAll : 1.0;
                this->rwCellTotalOpacity[i] = data.sigmaT_bar;
                this->rwCellEligible[i] = true;
            }
            else
            {
                data = PGRWCellData{};
                this->rwCellEligible[i] = false;
            }
        }

        if (this->rwCellEligible[i] && this->postProcess_.enabled && this->observer_)
        {
            Vector3D cellCenter = this->grid.GetMeshPoint(i);
            double distToObserver = abs(cellCenter - this->observer_->getCenter());
            double charLen = meanChordLength;
            double obsR = this->observer_->getRadius();
            if (distToObserver + charLen >= obsR && distToObserver - charLen <= obsR)
                this->rwCellEligible[i] = false;
        }
    }
}

bool RadiationIMC::tryRandomWalkStep(Particle &particle, Functionality &functionality, double dopplerShift)
{
    (void)dopplerShift;

    size_t cellIndex = particle.cellIndex;
    ComputationalCell3D &cell = this->cells[cellIndex];

    bool const movingHydroNoMMC = this->withHydro && !this->MMC;
    double const gammaCell = movingHydroNoMMC ? CellGamma(cell.velocity) : 1.0;

    if(!(gammaCell > 0.0) || !std::isfinite(gammaCell))
        return false;

    Vector3D const oldLabVelocity = particle.velocity;
    double const oldLabWeight = particle.weight;

    Particle materialParticle = particle;
    if(movingHydroNoMMC)
    {
        LorentzTransformation(materialParticle, cell.velocity);
        if(this->multigroupOpacity)
            ClampFrequencyToBounds(materialParticle.frequency);
    }

    const auto &normals = this->gridData.normalsOfCells[cellIndex];
    const auto &facePoints = this->gridData.pointsOnFaces[cellIndex];
    double Ro = std::numeric_limits<double>::max();
    for(size_t f = 0; f < normals.size(); ++f)
    {
        double d = ScalarProd(particle.location - facePoints[f], normals[f]);
        Ro = std::min(Ro, d);
    }
    if(Ro <= 0.0)
        Ro = 0.0;

    double sigmaT, sigma_a_eff, D_phys, gamma_rw;
    bool isPGRW = (this->multigroupOpacity != nullptr);
    size_t groupCutoff = 0;

    if(isPGRW)
    {
        const PGRWCellData &rwd = this->rwCellData[cellIndex];
        sigmaT = rwd.sigmaT_bar;
        sigma_a_eff = rwd.sigmaA_bar;
        D_phys = rwd.D;
        gamma_rw = rwd.gamma;
        groupCutoff = rwd.groupCutoff;
    }
    else
    {
        sigmaT = this->rwCellTotalOpacity[cellIndex];
        sigma_a_eff = this->planckOpacities[cellIndex];
        D_phys = units::clight / (3.0 * sigmaT);
        gamma_rw = 1.0;
    }

    bool doRW = (Ro > 0.0 && std::isfinite(Ro)
                 && sigmaT > 0.0 && std::isfinite(sigmaT)
                 && D_phys > 0.0 && std::isfinite(D_phys)
                 && Ro * sigmaT >= this->rwMinParticleOpticalDepth);

    if(doRW && isPGRW)
    {
        if(groupCutoff == 0 || groupCutoff > ENERGY_GROUPS_NUM)
        {
            doRW = false;
        }
        else
        {
            double const cutoffEnergy = ComputationalCell3D::energyBoundaries[groupCutoff];
            double coFreq = materialParticle.frequency;
            ClampFrequencyToBounds(coFreq);
            if(coFreq >= cutoffEnergy)
                doRW = false;
        }
    }

    if(!doRW)
        return false;

    Vector3D rwCenter = particle.location;
    double f = this->factorFleck[cellIndex];

    double const tauLeak = this->randomWalk->sampleLeakTime(this->dist(this->re));
    double const tLeakCo = tauLeak * Ro * Ro / D_phys;

    double const tCensusCo = movingHydroNoMMC ? particle.timeLeft / gammaCell
                                              : particle.timeLeft;

    double tUpscatterCo = std::numeric_limits<double>::max();
    if(isPGRW && gamma_rw < 1.0 && sigma_a_eff > 0.0 && f > 0.0)
    {
        double const xiUp = PositiveRandom(this->dist(this->re));
        double const upscatterRateCo =
            units::clight * (1.0 - f) * sigma_a_eff * (1.0 - gamma_rw);
        tUpscatterCo = -std::log(xiUp) / upscatterRateCo;
    }

    enum { RW_LEAK, RW_CENSUS, RW_UPSCATTER };
    int rwEvent;
    double dtCo;
    if(tLeakCo <= tCensusCo && tLeakCo <= tUpscatterCo)
    {
        rwEvent = RW_LEAK;
        dtCo = tLeakCo;
    }
    else if(tCensusCo <= tUpscatterCo)
    {
        rwEvent = RW_CENSUS;
        dtCo = tCensusCo;
    }
    else
    {
        rwEvent = RW_UPSCATTER;
        dtCo = tUpscatterCo;
    }

    // Level-1 mixed-frame approximation: dtLab = gammaCell * dtCo.
    // This intentionally ignores the event displacement term
    // gammaCell * dot(cell.velocity, dxCo) / c^2.
    // Use Level 2 before relying on high-v/c RW results.
    double dtLab = movingHydroNoMMC ? gammaCell * dtCo : dtCo;
    if(rwEvent == RW_CENSUS)
        dtLab = particle.timeLeft;

    if(dtLab < 0.0 || !std::isfinite(dtLab))
        return false;

    if(dtLab > particle.timeLeft)
        dtLab = particle.timeLeft;

    double const absRateCo = sigma_a_eff * f * units::clight;
    double const oldCoWeight = materialParticle.weight;
    double const rwExpCo = std::expm1(-dtCo * absRateCo);
    double const absorbedCo = -rwExpCo * oldCoWeight;

    if(!this->noHydroFeedback)
        this->conserved[cellIndex].internal_energy += absorbedCo;

    // TODO(mixed-frame): Erad_time_avg is accumulated as a material-frame
    // diffusion tally here. Ordinary IMC mixed-frame tally semantics should be
    // audited separately before using Erad_time_avg for high-v/c hydro diagnostics.
    double integratedCo;
    if(absRateCo > 0.0)
        integratedCo = oldCoWeight * rwExpCo * (-1.0 / absRateCo);
    else
        integratedCo = oldCoWeight * dtCo;

    this->Erad_time_avg[cellIndex] += integratedCo;

    if(this->withEgTimeAvg && this->multigroupOpacity)
    {
        double freqForGroup = materialParticle.frequency;
        ClampFrequencyToBounds(freqForGroup);
        size_t const g = this->opacity->findGroup(freqForGroup);
        this->Eg_time_avg[cellIndex][g] += integratedCo;
    }

    materialParticle.weight *= 1.0 + rwExpCo;

    if(this->postProcess_.enabled && this->observer_)
    {
        double absorbed = oldCoWeight - materialParticle.weight;
        if(absorbed > 0.0)
            this->observer_->addAbsorbedEnergy(absorbed);
    }

    materialParticle.timeLeft = particle.timeLeft - dtLab;
    if(materialParticle.timeLeft < 0.0 && materialParticle.timeLeft > -1e-12)
        materialParticle.timeLeft = 0.0;

    auto finalizeAccelerationStep = [&](bool remove) -> bool {
        if(remove)
        {
            functionality.change = MonteCarloParticleStatus::REMOVE;

            if(movingHydroNoMMC && !this->diffusionPressureGradient && !this->noHydroFeedback)
            {
                this->conserved[cellIndex].momentum +=
                    (oldLabWeight * oldLabVelocity) * units::inv_clight2;
            }

            return true;
        }

        Particle finalLabParticle = materialParticle;

        if(movingHydroNoMMC)
        {
            LorentzTransformation(finalLabParticle, -1 * cell.velocity);
            if(this->multigroupOpacity)
                ClampFrequencyToBounds(finalLabParticle.frequency);
        }

        particle.location  = finalLabParticle.location;
        particle.velocity  = finalLabParticle.velocity;
        particle.frequency = finalLabParticle.frequency;
        particle.weight    = finalLabParticle.weight;
        particle.timeLeft  = finalLabParticle.timeLeft;

        if(movingHydroNoMMC && !this->diffusionPressureGradient && !this->noHydroFeedback)
        {
            this->conserved[cellIndex].momentum +=
                (oldLabWeight * oldLabVelocity - particle.weight * particle.velocity)
                * units::inv_clight2;
        }

        return true;
    };

    bool removeParticle = false;

    double const lowWeightCutoff = this->postProcess_.enabled ? 1e-4 : 1e-3;
    double const initialCoWeightApprox = movingHydroNoMMC
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

    double cosTheta = 2.0 * this->dist(this->re) - 1.0;
    double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));
    double phi = 2.0 * RW_PI * this->dist(this->re);
    Vector3D posDir(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);

    double displacement;
    if(rwEvent == RW_LEAK)
    {
        displacement = Ro;
    }
    else
    {
        double const tauPos = D_phys * dtCo / (Ro * Ro);
        displacement = Ro * this->randomWalk->sampleRadius(tauPos, this->dist(this->re));
    }

    if(displacement > Ro * (1.0 + 1e-12))
    {
        std::cerr << "RW BUG: displacement=" << displacement << " > Ro=" << Ro
                  << " ratio=" << displacement / Ro
                  << " event=" << rwEvent << " cell=" << cellIndex
                  << " tauPos=" << (D_phys * dtCo / (Ro * Ro))
                  << " particle=" << particle.id << std::endl;
        displacement = Ro;
    }

    Vector3D newLocation = rwCenter + displacement * posDir;

    static constexpr double nudge = 1e-10;
    newLocation = newLocation * (1.0 - nudge) + nudge * this->grid.GetMeshPoint(cellIndex);

    for(size_t fi = 0; fi < normals.size(); ++fi)
    {
        double d = ScalarProd(newLocation - facePoints[fi], normals[fi]);
        if(d < 0)
        {
            std::cerr << "RW BUG: particle outside cell after RW step!"
                      << " face=" << fi << " d=" << d
                      << " Ro=" << Ro << " displacement=" << displacement
                      << " cell=" << cellIndex
                      << " nFaces=" << normals.size()
                      << " rwCenter=(" << rwCenter.x << "," << rwCenter.y << "," << rwCenter.z << ")"
                      << " newLoc=(" << newLocation.x << "," << newLocation.y << "," << newLocation.z << ")"
                      << " cellCenter=(" << this->grid.GetMeshPoint(cellIndex).x << "," << this->grid.GetMeshPoint(cellIndex).y << "," << this->grid.GetMeshPoint(cellIndex).z << ")"
                      << " facePoint=(" << facePoints[fi].x << "," << facePoints[fi].y << "," << facePoints[fi].z << ")"
                      << " normal=(" << normals[fi].x << "," << normals[fi].y << "," << normals[fi].z << ")"
                      << " event=" << rwEvent
                      << " particle=" << particle.id
                      << std::endl;
            displacement *= 0.99;
            newLocation = rwCenter + displacement * posDir;
            fi = static_cast<size_t>(-1);
        }
    }

    materialParticle.location = newLocation;
    materialParticle.velocity = this->opacity->getRandomVelocity(cell);

    if(rwEvent == RW_UPSCATTER && isPGRW)
    {
        this->multigroupOpacity->GetCummulativeOpacity(cell);
        const auto &cumOp = this->multigroupOpacity->getCummulativeOpacity();
        double cdfAtCutoff = cumOp[groupCutoff];
        double cdfTotal = cumOp[ENERGY_GROUPS_NUM];
        if(cdfTotal > cdfAtCutoff)
        {
            double lo = cdfAtCutoff / cdfTotal;
            double xi = this->dist(this->re);
            materialParticle.frequency = this->multigroupOpacity->GetThermalEnergy(cell, lo + xi * (1.0 - lo));
        }
        else
        {
            materialParticle.frequency = std::nextafter(
                ComputationalCell3D::energyBoundaries[groupCutoff],
                std::numeric_limits<double>::max());
        }
        ClampFrequencyToBounds(materialParticle.frequency);
    }

    if(rwEvent == RW_CENSUS)
        functionality.change = MonteCarloParticleStatus::DONE;

    return finalizeAccelerationStep(false);
}
