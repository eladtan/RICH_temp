#include "ReverseDDMC.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#endif

void ReverseDDMC::configure(ReverseEstimatorConfig const &cfg, bool usePolarization)
{
    minParticleOpticalDepth_ = cfg.ddmcMinParticleOpticalDepth;
    polClosure_ = cfg.ddmcPolClosure;
    multigroupMode_ = cfg.multigroupDDMCMode;
    manualK_ = cfg.ddmcManualScatterings;
    depolN_ = cfg.ddmcDepolarizationScatterings;
    muellerNormSafetyBound_ = 1e6;
    usePolarization_ = usePolarization;
    diag_.upscatterByGroup.assign(ENERGY_GROUPS_NUM, 0);
    diag_.thermalSampledGroupCount.assign(ENERGY_GROUPS_NUM, 0);
    diag_.thermalExactSampledGroupCount.assign(ENERGY_GROUPS_NUM, 0);
    diag_.thermalApproxSampledGroupCount.assign(ENERGY_GROUPS_NUM, 0);
    diag_.thermalFailureByReason.fill(0);
}

void ReverseDDMC::setCellData(std::vector<ReverseDDMCCellData> data)
{
    cellData_ = std::move(data);
}

bool ReverseDDMC::isEligible(ReverseAdjointPacket const &pkt) const
{
    if (pkt.cellIndex >= cellData_.size())
        return false;
    return cellData_[pkt.cellIndex].eligible;
}

ReverseDDMCRates ReverseDDMC::computeRates(ReverseDDMCCellData const &cd) const
{
    ReverseDDMCRates r;
    double effectiveSigmaA = cd.fleckFactor * cd.sigmaA;
    r.absDecayRate = units::clight * effectiveSigmaA;
    r.scatterRate = units::clight * cd.sigmaS;
    r.upscatterRate = cd.upscatterRateCo;
    r.eventRate = cd.totalLeakRate + cd.upscatterRateCo;

    if (!std::isfinite(r.eventRate) || r.eventRate <= 0.0)
    {
        r.valid = false;
        return r;
    }
    r.valid = true;
    return r;
}

double ReverseDDMC::computeMinDistToLeakFaces(
    ReverseDDMCCellData const &cd, Vector3D const &pos) const
{
    double minDist = std::numeric_limits<double>::max();
    for (auto const &fl : cd.faceLeaks)
    {
        double dist = std::abs(ScalarProd(fl.faceCentroid - pos, fl.faceNormal));
        minDist = std::min(minDist, dist);
    }
    return minDist;
}

std::string ReverseDDMC::closureModeName() const
{
    switch (polClosure_)
    {
    case ReverseDDMCPolarizationClosure::Synthetic: return "synthetic";
    case ReverseDDMCPolarizationClosure::Depolarize: return "depolarize";
    case ReverseDDMCPolarizationClosure::ExplicitK: return "explicit_k";
    }
    return "unknown";
}

void ReverseDDMC::mpiReduceDiagnostics()
{
#ifdef RICH_MPI
    auto reduceSum64 = [](uint64_t &val) {
        unsigned long long local = static_cast<unsigned long long>(val);
        unsigned long long global = 0;
        MPI_Reduce(&local, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        val = static_cast<uint64_t>(global);
    };
    auto reduceSumDbl = [](double &val) {
        double local = val;
        double global = 0.0;
        MPI_Reduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        val = global;
    };

    reduceSum64(diag_.fallbackNotEligible);
    reduceSum64(diag_.fallbackBadFrame);
    reduceSum64(diag_.fallbackBadRates);
    reduceSum64(diag_.fallbackParticleDepth);
    reduceSum64(diag_.fallbackMuellerNorm);
    reduceSum64(diag_.upscatterCount);
    reduceSum64(diag_.fallbackAboveCutoff);
    reduceSum64(diag_.fallbackNoThermalSampler);
    reduceSum64(diag_.censusCount);
    reduceSum64(diag_.timeLimitedStepCount);
    reduceSum64(diag_.leakCount);
    reduceSum64(diag_.residenceCount);
    reduceSum64(diag_.closureAttemptCount);
    reduceSum64(diag_.polClosureApplied);
    reduceSum64(diag_.polClosureDepolarized);
    reduceSum64(diag_.totalResetsDuringResidence);
    reduceSum64(diag_.collapsedPgrwScoreCount);
    reduceSum64(diag_.exactGroupScoreCount);
    reduceSumDbl(diag_.totalResidenceTimeCo);
    reduceSumDbl(diag_.totalResidenceTimeLab);
    reduceSumDbl(diag_.totalExpectedScatterCount);
    reduceSumDbl(diag_.totalSyntheticScatterCount);
    reduceSumDbl(diag_.totalPolarizationDamping);

    if (!diag_.upscatterByGroup.empty())
    {
        std::vector<unsigned long long> local(diag_.upscatterByGroup.begin(), diag_.upscatterByGroup.end());
        std::vector<unsigned long long> global(local.size(), 0);
        MPI_Reduce(local.data(), global.data(), static_cast<int>(local.size()),
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            for (size_t i = 0; i < global.size(); ++i)
                diag_.upscatterByGroup[i] = static_cast<uint64_t>(global[i]);
    }
    if (!diag_.thermalSampledGroupCount.empty())
    {
        std::vector<unsigned long long> local(diag_.thermalSampledGroupCount.begin(), diag_.thermalSampledGroupCount.end());
        std::vector<unsigned long long> global(local.size(), 0);
        MPI_Reduce(local.data(), global.data(), static_cast<int>(local.size()),
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            for (size_t i = 0; i < global.size(); ++i)
                diag_.thermalSampledGroupCount[i] = static_cast<uint64_t>(global[i]);
    }
    if (!diag_.thermalExactSampledGroupCount.empty())
    {
        std::vector<unsigned long long> local(diag_.thermalExactSampledGroupCount.begin(), diag_.thermalExactSampledGroupCount.end());
        std::vector<unsigned long long> global(local.size(), 0);
        MPI_Reduce(local.data(), global.data(), static_cast<int>(local.size()),
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            for (size_t i = 0; i < global.size(); ++i)
                diag_.thermalExactSampledGroupCount[i] = static_cast<uint64_t>(global[i]);
    }
    if (!diag_.thermalApproxSampledGroupCount.empty())
    {
        std::vector<unsigned long long> local(diag_.thermalApproxSampledGroupCount.begin(), diag_.thermalApproxSampledGroupCount.end());
        std::vector<unsigned long long> global(local.size(), 0);
        MPI_Reduce(local.data(), global.data(), static_cast<int>(local.size()),
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            for (size_t i = 0; i < global.size(); ++i)
                diag_.thermalApproxSampledGroupCount[i] = static_cast<uint64_t>(global[i]);
    }
    {
        std::array<unsigned long long, ReverseDDMCDiagnostics::NumThermalFailureReasons> local{}, global{};
        for (size_t i = 0; i < local.size(); ++i)
            local[i] = static_cast<unsigned long long>(diag_.thermalFailureByReason[i]);
        MPI_Reduce(local.data(), global.data(), static_cast<int>(local.size()),
                   MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            for (size_t i = 0; i < global.size(); ++i)
                diag_.thermalFailureByReason[i] = static_cast<uint64_t>(global[i]);
    }
#endif
}
