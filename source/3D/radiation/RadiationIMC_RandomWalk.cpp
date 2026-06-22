#include "RadiationIMC.hpp"
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
    /*  — diagnostics commented out —
    constexpr double DIAG_WEIGHT_CUTOFF_RW = 1e8;
    constexpr double DIAG_RHO_MAX_RW = 0.01;
    inline bool isDiagFreqRW(double freq, double weight, double rho, const OpacityCalculator &opacity)
    {
        if(std::abs(weight) < DIAG_WEIGHT_CUTOFF_RW) return false;
        if(rho > DIAG_RHO_MAX_RW) return false;
        size_t g = opacity.findGroup(freq);
        return g == 15 || g == 16 || g == 18;
    }
    inline double freqToKeVRW(double freq) { return freq / units::kev; }
    */
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
    }
}

bool RadiationIMC::tryRandomWalkStep(Particle &particle, Functionality &functionality, double dopplerShift)
{
    size_t cellIndex = particle.cellIndex;
    ComputationalCell3D &cell = this->cells[cellIndex];

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

    bool doRW = (Ro > 0 && sigmaT > 0 && D_phys > 0
                 && Ro * sigmaT >= this->rwMinParticleOpticalDepth);

    if(doRW && isPGRW)
    {
        double cutoffEnergy = ComputationalCell3D::energyBoundaries[groupCutoff];
        double coFreq = particle.frequency;
        if(this->withHydro && !this->MMC)
            coFreq *= dopplerShift;
        ClampFrequencyToBounds(coFreq);
        if(coFreq >= cutoffEnergy)
            doRW = false;
    }

    if(!doRW)
        return false;

    /* — diagnostics commented out —
    if(this->multigroupOpacity && isDiagFreqRW(particle.frequency, particle.weight, cell.density, *this->opacity))
    {
        const char *evNames[] = {"LEAK", "CENSUS", "UPSCATTER"};
        (void)evNames;
        std::cerr << "[HF-RW-ENTRY] id=" << particle.id
                  << " freq=" << freqToKeVRW(particle.frequency) << " keV"
                  << " group=" << this->opacity->findGroup(particle.frequency)
                  << " cell=" << cellIndex
                  << " Ro=" << Ro << " sigmaT=" << sigmaT
                  << " T=" << cell.temperature
                  << " rho=" << cell.density
                  << std::endl;
    }
    */

    Vector3D rwCenter = particle.location;
    Vector3D oldVelocity = particle.velocity;
    double oldWeight = particle.weight;
    double f = this->factorFleck[cellIndex];

    double tauLeak = this->randomWalk->sampleLeakTime(this->dist(this->re));
    double tLeak = tauLeak * Ro * Ro / D_phys;

    double tCensus = particle.timeLeft;

    double tUpscatter = std::numeric_limits<double>::max();
    if(isPGRW && gamma_rw < 1.0 && sigma_a_eff > 0 && f > 0)
    {
        double xiUp = this->dist(this->re);
        tUpscatter = -std::log(xiUp) / (units::clight * (1.0 - f) * sigma_a_eff * (1.0 - gamma_rw));
    }

    enum { RW_LEAK, RW_CENSUS, RW_UPSCATTER };
    int rwEvent;
    double dt;
    if(tLeak <= tCensus && tLeak <= tUpscatter)
    {
        rwEvent = RW_LEAK;
        dt = tLeak;
    }
    else if(tCensus <= tUpscatter)
    {
        rwEvent = RW_CENSUS;
        dt = tCensus;
    }
    else
    {
        rwEvent = RW_UPSCATTER;
        dt = tUpscatter;
    }

    double rwAbsRate = sigma_a_eff * f * units::clight;
    double rwExp = std::expm1(-dt * rwAbsRate);
    if(!this->noHydroFeedback)
    {
        this->conserved[cellIndex].internal_energy += -rwExp * particle.weight;
    }
    if(rwAbsRate > 0)
    {
        this->Erad_time_avg[cellIndex] += particle.weight * rwExp * (-1.0 / rwAbsRate);
        if(this->withEgTimeAvg && this->multigroupOpacity)
        {
            size_t g = this->opacity->findGroup(particle.frequency);
            this->Eg_time_avg[cellIndex][g] += particle.weight * rwExp * (-1.0 / rwAbsRate);
        }
    }
    particle.weight *= 1.0 + rwExp;

    particle.timeLeft -= dt;

    if(std::abs(particle.weight) < particle.initialWeight * 1e-4)
    {
        functionality.change = MonteCarloParticleStatus::REMOVE;
        if(!this->noHydroFeedback)
        {
            this->conserved[cellIndex].internal_energy += particle.weight;
        }
        return true;
    }

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
        double tauPos = D_phys * dt / (Ro * Ro);
        displacement = Ro * this->randomWalk->sampleRadius(tauPos, this->dist(this->re));
    }

    if(displacement > Ro * (1.0 + 1e-12))
    {
        std::cerr << "RW BUG: displacement=" << displacement << " > Ro=" << Ro
                  << " ratio=" << displacement / Ro
                  << " event=" << rwEvent << " cell=" << cellIndex
                  << " tauPos=" << (D_phys * dt / (Ro * Ro))
                  << " particle=" << particle.id << std::endl;
        displacement = Ro;
    }

    particle.location = rwCenter + displacement * posDir;

    static constexpr double nudge = 1e-10;
    particle.location = particle.location * (1.0 - nudge) + nudge * this->grid.GetMeshPoint(cellIndex);

    for(size_t fi = 0; fi < normals.size(); ++fi)
    {
        double d = ScalarProd(particle.location - facePoints[fi], normals[fi]);
        if(d < 0)
        {
            std::cerr << "RW BUG: particle outside cell after RW step!"
                      << " face=" << fi << " d=" << d
                      << " Ro=" << Ro << " displacement=" << displacement
                      << " cell=" << cellIndex
                      << " nFaces=" << normals.size()
                      << " rwCenter=(" << rwCenter.x << "," << rwCenter.y << "," << rwCenter.z << ")"
                      << " newLoc=(" << particle.location.x << "," << particle.location.y << "," << particle.location.z << ")"
                      << " cellCenter=(" << this->grid.GetMeshPoint(cellIndex).x << "," << this->grid.GetMeshPoint(cellIndex).y << "," << this->grid.GetMeshPoint(cellIndex).z << ")"
                      << " facePoint=(" << facePoints[fi].x << "," << facePoints[fi].y << "," << facePoints[fi].z << ")"
                      << " normal=(" << normals[fi].x << "," << normals[fi].y << "," << normals[fi].z << ")"
                      << " event=" << rwEvent
                      << " particle=" << particle.id
                      << std::endl;
            displacement *= 0.99;
            particle.location = rwCenter + displacement * posDir;
            fi = static_cast<size_t>(-1);
        }
    }

    particle.velocity = this->opacity->getRandomVelocity(cell);

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
            particle.frequency = this->multigroupOpacity->GetThermalEnergy(cell, lo + xi * (1.0 - lo));
        }
        else
        {
            particle.frequency = std::nextafter(
                ComputationalCell3D::energyBoundaries[groupCutoff],
                std::numeric_limits<double>::max());
        }
        /* — diagnostics commented out —
        if(isDiagFreqRW(particle.frequency, particle.weight, cell.density, *this->opacity))
        {
            std::cerr << "[HF-RW-UPSCATTER] id=" << particle.id
                      << " freq=" << freqToKeVRW(particle.frequency) << " keV"
                      << " group=" << this->opacity->findGroup(particle.frequency)
                      << " cutoff=" << groupCutoff
                      << " cdfCut=" << cdfAtCutoff << " cdfTot=" << cdfTotal
                      << " cell=" << cellIndex
                      << " T=" << cell.temperature
                      << " rho=" << cell.density
                      << std::endl;
        }
        */
    }

    if(this->withHydro && !this->MMC)
    {
        double freqBeforeLT = particle.frequency;
        (void)freqBeforeLT;
        LorentzTransformation(particle, -1 * cell.velocity);
        if(this->multigroupOpacity)
        {
            ClampFrequencyToBounds(particle.frequency);
            /* — diagnostics commented out —
            bool isHighAfterLT = isDiagFreqRW(particle.frequency, particle.weight, cell.density, *this->opacity);
            if(wasHighBefore || isHighAfterLT)
            {
                std::cerr << "[HF-RW-LORENTZ] id=" << particle.id
                          << " freqBefore=" << freqToKeVRW(freqBeforeLT) << " keV"
                          << " freqAfter=" << freqToKeVRW(particle.frequency) << " keV"
                          << " groupAfter=" << this->opacity->findGroup(particle.frequency)
                          << " cell=" << cellIndex
                          << " rwEvent=" << rwEvent
                          << std::endl;
            }
            */
        }
        if(!this->diffusionPressureGradient && !this->noHydroFeedback)
            this->conserved[cellIndex].momentum += (oldWeight * oldVelocity - particle.weight * particle.velocity) * units::inv_clight2;
    }

    if(rwEvent == RW_CENSUS)
        functionality.change = MonteCarloParticleStatus::DONE;
    return true;
}
