#ifndef REVERSE_DDMC_HPP
#define REVERSE_DDMC_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>
#include "3D/elementary/Vector3D.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include "ReversePacket.hpp"
#include "ReverseEstimatorConfig.hpp"
#include "ReverseDoppler.hpp"
#include "ReversePolarizationMueller.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

struct ReverseDDMCFaceLeak
{
    size_t faceIndex = std::numeric_limits<size_t>::max();
    size_t nextCellIndex = std::numeric_limits<size_t>::max();
    double rate = 0.0;
    Vector3D faceNormal;
    Vector3D faceCentroid;
};

struct ReverseDDMCCellData
{
    bool eligible = false;
    bool observerExcluded = false;
    bool photosphereExcluded = false;
    bool boundaryExcluded = false;

    double sigmaA = 0.0;
    double sigmaS = 0.0;
    double sigmaT = 0.0;
    double diffusionCoefficient = 0.0;
    double fleckFactor = 1.0;
    double volume = 0.0;
    double surfaceArea = 0.0;
    double meanChordLength = 0.0;
    double opticalDepthCell = 0.0;

    std::vector<ReverseDDMCFaceLeak> faceLeaks;
    double totalLeakRate = 0.0;

    double expectedScatterRateCo = 0.0;
    double resetRateCo = 0.0;
    double depolRateEstimate = 0.0;

    // PGRW multigroup aggregate (mirrors forward RadiationIMC_DDMC.cpp)
    size_t groupCutoff = 0;
    double sigmaA_PGRW = 0.0;
    double sigmaT_PGRW = 0.0;
    double diffusionCoefficient_PGRW = 0.0;
    double gammaPGRW = 1.0;
    double totalLeakRatePGRW = 0.0;
    double upscatterRateCo = 0.0;
};

struct ReverseDDMCRates
{
    bool valid = false;
    double eventRate = 0.0;
    double absDecayRate = 0.0;
    double scatterRate = 0.0;
    double upscatterRate = 0.0;
};

struct ReverseDDMCDiagnostics
{
    uint64_t fallbackNotEligible = 0;
    uint64_t fallbackBadFrame = 0;
    uint64_t fallbackBadRates = 0;
    uint64_t fallbackParticleDepth = 0;
    uint64_t fallbackMuellerNorm = 0;
    uint64_t upscatterCount = 0;
    uint64_t fallbackAboveCutoff = 0;
    uint64_t fallbackNoThermalSampler = 0;
    uint64_t censusCount = 0;
    uint64_t timeLimitedStepCount = 0;
    uint64_t leakCount = 0;
    uint64_t residenceCount = 0;
    uint64_t closureAttemptCount = 0;
    uint64_t polClosureApplied = 0;
    uint64_t polClosureDepolarized = 0;
    double totalResidenceTimeCo = 0.0;
    double totalResidenceTimeLab = 0.0;
    double totalExpectedScatterCount = 0.0;
    double totalSyntheticScatterCount = 0.0;
    double totalPolarizationDamping = 0.0;
    uint64_t totalResetsDuringResidence = 0;
    uint64_t collapsedPgrwScoreCount = 0;
    uint64_t exactGroupScoreCount = 0;
    std::vector<uint64_t> upscatterByGroup;
    std::vector<uint64_t> thermalSampledGroupCount;
    std::vector<uint64_t> thermalExactSampledGroupCount;
    std::vector<uint64_t> thermalApproxSampledGroupCount;
    static constexpr size_t NumThermalFailureReasons = 5;
    std::array<uint64_t, NumThermalFailureReasons> thermalFailureByReason{};
};

class ReverseDDMC
{
public:
    ReverseDDMC() = default;

    void configure(ReverseEstimatorConfig const &cfg, bool usePolarization);
    void setCellData(std::vector<ReverseDDMCCellData> data);
    bool isEligible(ReverseAdjointPacket const &pkt) const;

    template <class RNG>
    bool tryStep(ReverseAdjointPacket &pkt,
                 std::function<Vector3D(size_t)> cellVelocityFn,
                 std::function<double(size_t, size_t)> sourceLuminosityFn,
                 std::function<ThermalSampleResult(size_t)> thermalResampleAboveCutoffFn,
                 double patchAreaOverN, double remainingLabTime,
                 std::function<void(ReverseAdjointPacket &, size_t, double,
                                    double, double, double, double, double, double, double)> scoreResidenceFn,
                 std::function<void(ReverseAdjointPacket &, size_t, double,
                                    double, double, double, double, double, double, double)> scoreResidenceCollapsedFn,
                 std::function<void(ReverseAdjointPacket &, std::string const &)> terminateFn,
                 RNG &rng, std::uniform_real_distribution<double> &uniform);

    ReverseDDMCDiagnostics const &diagnostics() const { return diag_; }
    std::vector<ReverseDDMCCellData> const &cellData() const { return cellData_; }

    std::string closureModeName() const;

    void mpiReduceDiagnostics();

private:
    double minParticleOpticalDepth_ = 5.0;
    ReverseDDMCPolarizationClosure polClosure_ = ReverseDDMCPolarizationClosure::Synthetic;
    ReverseMultigroupDDMCMode multigroupMode_ = ReverseMultigroupDDMCMode::PGRWCollapsed;
    int manualK_ = 4;
    double depolN_ = 2.0;
    double muellerNormSafetyBound_ = 1e6;
    bool usePolarization_ = true;

    std::vector<ReverseDDMCCellData> cellData_;
    ReverseDDMCDiagnostics diag_;

    ReverseDDMCRates computeRates(ReverseDDMCCellData const &cd) const;

    template <class RNG>
    void applyPolarizationClosure(
        ReverseAdjointPacket &pkt, double dtCo,
        double scatterRate, double resetRate,
        Vector3D const &kLeakForward,
        RNG &rng, std::uniform_real_distribution<double> &uniform);

    template <class RNG>
    uint64_t drawPoissonOrApprox(double mean, RNG &rng,
                                 std::uniform_real_distribution<double> &uniform);

    template <class RNG>
    ReverseDDMCFaceLeak const &chooseLeakFace(
        ReverseDDMCCellData const &cd, RNG &rng,
        std::uniform_real_distribution<double> &uniform);

    template <class RNG>
    Vector3D sampleLeakDirection(Vector3D const &faceNormal,
                                 RNG &rng,
                                 std::uniform_real_distribution<double> &uniform);

    double computeMinDistToLeakFaces(ReverseDDMCCellData const &cd,
                                     Vector3D const &pos) const;
};

// ---------- template implementations ----------

template <class RNG>
bool ReverseDDMC::tryStep(
    ReverseAdjointPacket &pkt,
    std::function<Vector3D(size_t)> cellVelocityFn,
    std::function<double(size_t, size_t)> sourceLuminosityFn,
    std::function<ThermalSampleResult(size_t)> thermalResampleAboveCutoffFn,
    double patchAreaOverN, double remainingLabTime,
    std::function<void(ReverseAdjointPacket &, size_t, double,
                       double, double, double, double, double, double, double)> scoreResidenceFn,
    std::function<void(ReverseAdjointPacket &, size_t, double,
                       double, double, double, double, double, double, double)> scoreResidenceCollapsedFn,
    std::function<void(ReverseAdjointPacket &, std::string const &)> terminateFn,
    RNG &rng, std::uniform_real_distribution<double> &uniform)
{
    auto &cd = cellData_[pkt.cellIndex];

    if (!cd.eligible)
    {
        ++diag_.fallbackNotEligible;
        return false;
    }

    double minFaceDist = computeMinDistToLeakFaces(cd, pkt.xLab);
    if (minFaceDist * cd.sigmaT < minParticleOpticalDepth_)
    {
        ++diag_.fallbackParticleDepth;
        return false;
    }

    Vector3D cellVel = cellVelocityFn(pkt.cellIndex);
    ReverseDoppler::FrameState fs = ReverseDoppler::toComoving(pkt, cellVel);
    if (!fs.valid)
    {
        ++diag_.fallbackBadFrame;
        return false;
    }

    if (cd.groupCutoff > 0 && ENERGY_GROUPS_NUM > 1)
    {
        double nuCo = fs.valid ? fs.nuCo : pkt.nuLab;
        if (nuCo >= ComputationalCell3D::energyBoundaries[cd.groupCutoff])
        {
            ++diag_.fallbackAboveCutoff;
            return false;
        }
    }

    ReverseDDMCRates rates = computeRates(cd);
    if (!rates.valid || rates.eventRate <= 0.0)
    {
        ++diag_.fallbackBadRates;
        return false;
    }

    // Time census: clamp DDMC residence to remaining transport time
    double remainingCo = (fs.valid && fs.gamma > 0.0)
        ? remainingLabTime / fs.gamma : remainingLabTime;
    if (remainingCo <= 0.0)
    {
        terminateFn(pkt, "time_census");
        return true;
    }

    auto u01 = [&]() -> double { return uniform(rng); };

    double tEventCo = -std::log(std::max(u01(), 1e-300)) / rates.eventRate;
    double dtCo = std::min(tEventCo, remainingCo);
    bool hitCensus = (remainingCo <= tEventCo);

    double eventPick = u01() * rates.eventRate;
    bool isLeak = (eventPick <= cd.totalLeakRate);

    ReverseDDMCFaceLeak const *chosenLeak = nullptr;
    Vector3D leakDir;
    Vector3D postEventForward;

    if (isLeak)
    {
        chosenLeak = &chooseLeakFace(cd, rng, uniform);
        leakDir = sampleLeakDirection(chosenLeak->faceNormal, rng, uniform);
        postEventForward = leakDir * (-1.0);
    }
    else
    {
        postEventForward = ReverseMueller::sampleIsotropicDirection(u01);
    }

    // Save pre-decay weight for scoring (score uses the entry weight,
    // attenuation integral accounts for decay within residence).
    double w0 = pkt.scalarWeight;

    // Compute lab-frame time from comoving residence time
    double dtLab = (fs.valid && fs.gamma > 0.0) ? fs.gamma * dtCo : dtCo;
    dtLab = std::min(dtLab, remainingLabTime);  // guard against roundoff overshoot

    pkt.scalarWeight *= std::exp(-rates.absDecayRate * dtCo);
    pkt.tCoAccumulated += dtCo;
    pkt.tLabAccumulated += dtLab;
    pkt.pathLabAccumulated += units::clight * dtLab;

    if (pkt.scalarWeight < 1e-30)
    {
        terminateFn(pkt, "ddmc_weight_cutoff");
        return true;
    }

    if (usePolarization_)
    {
        applyPolarizationClosure(pkt, dtCo, rates.scatterRate,
                                 cd.resetRateCo, postEventForward, rng, uniform);
    }

    // Score DDMC residence: aggregate over all source groups below cutoff
    // when in PGRWCollapsed mode. Route to collapsed dataset.
    if (multigroupMode_ == ReverseMultigroupDDMCMode::PGRWCollapsed
        && cd.groupCutoff > 1 && pkt.currentCoGroup < cd.groupCutoff)
    {
        double totalSrcLum = 0.0;
        for (size_t h = 0; h < cd.groupCutoff; ++h)
            totalSrcLum += sourceLuminosityFn(pkt.cellIndex, h);
        scoreResidenceCollapsedFn(pkt, pkt.cellIndex, dtCo, cd.volume,
                         totalSrcLum,
                         patchAreaOverN, units::clight, fs.frameWeightFactor,
                         rates.absDecayRate, w0);
        ++diag_.collapsedPgrwScoreCount;
    }
    else
    {
        scoreResidenceFn(pkt, pkt.cellIndex, dtCo, cd.volume,
                         sourceLuminosityFn(pkt.cellIndex, pkt.currentCoGroup),
                         patchAreaOverN, units::clight, fs.frameWeightFactor,
                         rates.absDecayRate, w0);
        ++diag_.exactGroupScoreCount;
    }

    ++diag_.residenceCount;
    diag_.totalResidenceTimeCo += dtCo;
    diag_.totalResidenceTimeLab += dtLab;

    if (hitCensus)
    {
        ++diag_.censusCount;
        ++diag_.timeLimitedStepCount;
        terminateFn(pkt, "time_census");
        return true;
    }

    if (isLeak)
    {
        double nudge = 1e-10 * std::cbrt(cd.volume);
        pkt.xLab = chosenLeak->faceCentroid + nudge * leakDir;
        pkt.cellIndex = chosenLeak->nextCellIndex;
        pkt.kForwardLab = postEventForward;
        pkt.kReverseLab = leakDir;
        ++pkt.ddmcLeakCount;
        ++pkt.faceCrossingCount;
        ++diag_.leakCount;
    }
    else
    {
        auto sample = thermalResampleAboveCutoffFn(pkt.cellIndex);
        if (!sample.ok)
        {
            ++diag_.fallbackNoThermalSampler;
            size_t reasonIdx = static_cast<size_t>(sample.failure);
            if (reasonIdx < diag_.thermalFailureByReason.size())
                ++diag_.thermalFailureByReason[reasonIdx];
            return false;
        }
        pkt.nuCo = sample.nuCo;
        pkt.nuLab = ReverseDoppler::toLabFrequency(sample.nuCo, fs.dopplerFactor);
        pkt.currentCoGroup = sample.sampledGroup;

        pkt.M_obs_from_src.dampPolarizationRows(0.0);
        pkt.kForwardLab = postEventForward;
        pkt.kReverseLab = postEventForward * (-1.0);
        pkt.sourceBasisLab = ReverseMueller::choosePerpendicularBasis(postEventForward);
        pkt.basisInitialized = true;
        ++diag_.upscatterCount;
        if (sample.sampledGroup < diag_.upscatterByGroup.size())
            ++diag_.upscatterByGroup[sample.sampledGroup];
        if (sample.sampledGroup < diag_.thermalSampledGroupCount.size())
            ++diag_.thermalSampledGroupCount[sample.sampledGroup];
        if (sample.exactForward && sample.sampledGroup < diag_.thermalExactSampledGroupCount.size())
            ++diag_.thermalExactSampledGroupCount[sample.sampledGroup];
        if (sample.approximate && sample.sampledGroup < diag_.thermalApproxSampledGroupCount.size())
            ++diag_.thermalApproxSampledGroupCount[sample.sampledGroup];
    }

    ++pkt.ddmcStepCount;
    pkt.usedDDMC = true;
    return true;
}

template <class RNG>
ReverseDDMCFaceLeak const &ReverseDDMC::chooseLeakFace(
    ReverseDDMCCellData const &cd, RNG &rng,
    std::uniform_real_distribution<double> &uniform)
{
    double xi = uniform(rng) * cd.totalLeakRate;
    double cumulative = 0.0;
    for (size_t f = 0; f < cd.faceLeaks.size(); ++f)
    {
        cumulative += cd.faceLeaks[f].rate;
        if (xi <= cumulative)
            return cd.faceLeaks[f];
    }
    return cd.faceLeaks.back();
}

template <class RNG>
Vector3D ReverseDDMC::sampleLeakDirection(
    Vector3D const &faceNormal, RNG &rng,
    std::uniform_real_distribution<double> &uniform)
{
    auto u01 = [&]() -> double { return uniform(rng); };
    Vector3D outNormal = ReverseMueller::safeNormalize(faceNormal, Vector3D(0, 0, 1));

    double cosTheta = std::sqrt(std::max(u01(), 1e-30));
    double phi = 2.0 * ReverseMueller::POL_PI * u01();
    double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));

    Vector3D e1, e2;
    ReverseMueller::buildBasisAroundDirection(outNormal, e1, e2);
    Vector3D dir = cosTheta * outNormal
                 + sinTheta * std::cos(phi) * e1
                 + sinTheta * std::sin(phi) * e2;
    return ReverseMueller::safeNormalize(dir, outNormal);
}

template <class RNG>
void ReverseDDMC::applyPolarizationClosure(
    ReverseAdjointPacket &pkt, double dtCo,
    double scatterRate, double resetRate,
    Vector3D const &kLeakForward,
    RNG &rng, std::uniform_real_distribution<double> &uniform)
{
    auto u01 = [&]() -> double { return uniform(rng); };

    ++diag_.closureAttemptCount;

    if (polClosure_ == ReverseDDMCPolarizationClosure::Depolarize)
    {
        pkt.M_obs_from_src.dampPolarizationRows(0.0);
        ++diag_.polClosureDepolarized;
        return;
    }

    bool resetOccurred = false;
    double ageSinceReset = dtCo;
    if (resetRate > 0.0 && dtCo > 0.0)
    {
        double pReset = -std::expm1(-resetRate * dtCo);
        if (u01() < pReset)
        {
            resetOccurred = true;
            double xi = std::clamp(u01(), 0.0,
                1.0 - std::numeric_limits<double>::epsilon());
            double y = xi * pReset;
            ageSinceReset = -std::log1p(-y) / resetRate;
            ageSinceReset = std::min(ageSinceReset, dtCo);

            pkt.M_obs_from_src.resetToUnpolarizedSource();
            ++pkt.resetCount;
            ++diag_.totalResetsDuringResidence;
        }
    }

    double meanScat = scatterRate * ageSinceReset;
    uint64_t N = drawPoissonOrApprox(meanScat, rng, uniform);

    uint64_t K = static_cast<uint64_t>(manualK_);
    K = std::min(K, N);

    uint64_t Ndamped = (N > K) ? (N - K) : 0;
    double damping = 1.0;
    if (Ndamped > 0)
    {
        double exponent = -static_cast<double>(Ndamped) / depolN_;
        damping = (exponent < -745.0) ? 0.0 : std::exp(exponent);
    }

    pkt.M_obs_from_src.dampPolarizationRows(damping);

    // Synthetic scatterings bridge from the pre-residence direction
    // to the sampled leak direction, so the closure sees the actual geometry.
    Vector3D kBefore = pkt.kForwardLab;
    ReverseMueller::applySyntheticScatterings(pkt, kBefore, kLeakForward, K, u01);

    if (pkt.M_obs_from_src.frobeniusNorm() > muellerNormSafetyBound_)
    {
        pkt.M_obs_from_src.dampPolarizationRows(0.0);
        ++diag_.fallbackMuellerNorm;
    }

    diag_.totalExpectedScatterCount += meanScat;
    diag_.totalSyntheticScatterCount += static_cast<double>(K);
    diag_.totalPolarizationDamping += damping;
    ++diag_.polClosureApplied;
}

template <class RNG>
uint64_t ReverseDDMC::drawPoissonOrApprox(
    double mean, RNG &rng, std::uniform_real_distribution<double> &uniform)
{
    if (!(mean > 0.0) || !std::isfinite(mean))
        return 0;

    double zeroDampThreshold = static_cast<double>(manualK_) + 80.0 * depolN_;
    if (mean > zeroDampThreshold + 10.0 * std::sqrt(std::max(1.0, mean)))
        return static_cast<uint64_t>(std::ceil(zeroDampThreshold + 1.0));

    std::poisson_distribution<unsigned long long> pois(mean);
    return static_cast<uint64_t>(pois(rng));
}

#endif // REVERSE_DDMC_HPP
