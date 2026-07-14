#include "adaptive_statistics.hpp"
#include "source/3D/radiation/PolarizationStatistics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_set>

#ifdef RICH_MPI
#include <mpi.h>
#include "source/mpi/mpi_commands.hpp"
#endif

namespace imc_postprocess_tde {

double EffectiveMeasuredLBWeightCompression(Config const& cfg)
{
    if (cfg.measuredLBWeightCompression > 0.0)
        return cfg.measuredLBWeightCompression;
    return cfg.adaptiveSourceCells ? 1.0 : 0.5;
}

RankStepImbalance ComputeRankStepImbalance(
    std::string const& label,
    size_t gen,
    std::vector<size_t> const& localSteps,
    int rank)
{
    RankStepImbalance out;
    for (size_t s : localSteps)
        out.localSteps += static_cast<unsigned long long>(s);

#ifdef RICH_MPI
    unsigned long long globalSteps = out.localSteps;
    MPI_Allreduce(MPI_IN_PLACE, &globalSteps, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    out.globalSteps = globalSteps;

    double localStepsD = static_cast<double>(out.localSteps);
    MPI_Allreduce(&localStepsD, &out.maxRankSteps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int mpiSize = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    out.meanRankSteps = static_cast<double>(out.globalSteps) / std::max(mpiSize, 1);
#else
    out.globalSteps = out.localSteps;
    out.maxRankSteps = static_cast<double>(out.localSteps);
    out.meanRankSteps = static_cast<double>(out.localSteps);
#endif

    out.maxOverMean = (out.meanRankSteps > 0.0)
        ? out.maxRankSteps / out.meanRankSteps : 0.0;
    if (rank == 0) {
        std::cout << label << " rank_step_imbalance after generation " << (gen + 1)
                  << ": global_steps=" << out.globalSteps
                  << " mean_rank_steps=" << out.meanRankSteps
                  << " max_rank_steps=" << out.maxRankSteps
                  << " max_over_mean=" << out.maxOverMean
                  << std::endl;
    }
    return out;
}

bool AdaptiveLBCooldownSatisfied(AdaptiveSourceState const& state,
                                 Config const& cfg,
                                 size_t gen)
{
    if (state.lastAdaptiveMeasuredLBGeneration == std::numeric_limits<size_t>::max())
        return true;
    return gen >= state.lastAdaptiveMeasuredLBGeneration + cfg.adaptiveLBCooldownGenerations;
}

void AppendZeroVtkScalar(std::ofstream& file, std::string const& name, size_t n)
{
    file << "SCALARS " << name << " double 1\n"
         << "LOOKUP_TABLE default\n";
    for (size_t i = 0; i < n; ++i)
        file << 0.0 << "\n";
}

struct PackedSourceEscapeStat
{
    unsigned long long cellID = 0;
    unsigned long long observerIndex = 0;
    double weightSq = 0.0;
    double maxWeight = 0.0;
    double energy = 0.0;
    unsigned long long count = 0;
};

struct PackedAdaptiveScoreDelta
{
    unsigned long long cellID = 0;
    double delta = 0.0;
};

struct PackedSourceGroupEscapeStat
{
    unsigned long long cellID = 0;
    unsigned long long observerIndex = 0;
    unsigned long long groupIndex = 0;
    double weightSq = 0.0;
    double maxWeight = 0.0;
    double energy = 0.0;
    unsigned long long count = 0;
};

struct PackedAdaptiveCellGroupScoreDelta
{
    unsigned long long cellID = 0;
    unsigned long long groupIndex = 0;
    double delta = 0.0;
};

void AccumulateAdaptiveGroupSourceSummary(
    AdaptiveGroupSourceUpdateSummary& total,
    AdaptiveGroupSourceUpdateSummary const& gen)
{
    total.localStatsAfterPrune += gen.localStatsAfterPrune;
    total.localStatsDropped += gen.localStatsDropped;
    total.mpiStatsExchanged += gen.mpiStatsExchanged;
    total.maxReceivedShardStats =
        std::max(total.maxReceivedShardStats, gen.maxReceivedShardStats);
    total.maxPackedBytes = std::max(total.maxPackedBytes, gen.maxPackedBytes);
}

uint64_t SplitMix64(uint64_t x);

struct AdaptivePairKey
{
    size_t observerIndex = 0;
    size_t cellID = 0;

    bool operator==(AdaptivePairKey const& other) const
    {
        return observerIndex == other.observerIndex && cellID == other.cellID;
    }
};

struct AdaptivePairKeyHash
{
    size_t operator()(AdaptivePairKey const& key) const
    {
        uint64_t x = static_cast<uint64_t>(key.cellID);
        x ^= static_cast<uint64_t>(key.observerIndex) + 0x9e3779b97f4a7c15ULL +
             (x << 6) + (x >> 2);
        return static_cast<size_t>(SplitMix64(x));
    }
};

struct AdaptiveSourceGroupKey
{
    size_t observerIndex = 0;
    size_t groupIndex = 0;
    size_t cellID = 0;

    bool operator==(AdaptiveSourceGroupKey const& other) const
    {
        return observerIndex == other.observerIndex
            && groupIndex == other.groupIndex
            && cellID == other.cellID;
    }
};

struct AdaptiveSourceGroupKeyHash
{
    size_t operator()(AdaptiveSourceGroupKey const& key) const
    {
        uint64_t x = static_cast<uint64_t>(key.cellID);
        x ^= static_cast<uint64_t>(key.observerIndex) + 0x9e3779b97f4a7c15ULL +
             (x << 6) + (x >> 2);
        x ^= static_cast<uint64_t>(key.groupIndex) + 0x9e3779b97f4a7c15ULL +
             (x << 6) + (x >> 2);
        return static_cast<size_t>(SplitMix64(x));
    }
};

uint64_t SplitMix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

int AdaptiveCellOwner(size_t cellID, int ranks)
{
    if (ranks <= 1)
        return 0;
    return static_cast<int>(SplitMix64(static_cast<uint64_t>(cellID)) %
                            static_cast<uint64_t>(ranks));
}

int CheckedByteCount(size_t count, size_t elementSize, std::string const& label)
{
    unsigned long long bytes =
        static_cast<unsigned long long>(count) * static_cast<unsigned long long>(elementSize);
    if (bytes > static_cast<unsigned long long>(std::numeric_limits<int>::max()))
        throw UniversalError(label + " too large for MPI byte count");
    return static_cast<int>(bytes);
}

int CheckedByteTotal(unsigned long long bytes, std::string const& label)
{
    if (bytes > static_cast<unsigned long long>(std::numeric_limits<int>::max()))
        throw UniversalError(label + " total too large for MPI byte displacements");
    return static_cast<int>(bytes);
}

PackedSourceEscapeStat PackSourceEscapeStat(SphericalObserver::SourceCellEscapeStat const& s)
{
    PackedSourceEscapeStat p;
    p.cellID = static_cast<unsigned long long>(s.cellID);
    p.observerIndex = static_cast<unsigned long long>(s.observerIndex);
    p.energy = s.energy;
    p.count = static_cast<unsigned long long>(s.count);
    p.weightSq = s.weightSq;
    p.maxWeight = s.maxWeight;
    return p;
}

SphericalObserver::SourceCellEscapeStat UnpackSourceEscapeStat(PackedSourceEscapeStat const& p)
{
    SphericalObserver::SourceCellEscapeStat s;
    s.cellID = static_cast<size_t>(p.cellID);
    s.observerIndex = static_cast<size_t>(p.observerIndex);
    s.energy = p.energy;
    s.count = static_cast<size_t>(p.count);
    s.weightSq = p.weightSq;
    s.maxWeight = p.maxWeight;
    return s;
}

PackedSourceGroupEscapeStat PackSourceGroupEscapeStat(
    SphericalObserver::SourceCellGroupEscapeStat const& s)
{
    PackedSourceGroupEscapeStat p;
    p.cellID = static_cast<unsigned long long>(s.cellID);
    p.observerIndex = static_cast<unsigned long long>(s.observerIndex);
    p.groupIndex = static_cast<unsigned long long>(s.groupIndex);
    p.energy = s.energy;
    p.count = static_cast<unsigned long long>(s.count);
    p.weightSq = s.weightSq;
    p.maxWeight = s.maxWeight;
    return p;
}

std::vector<PackedSourceGroupEscapeStat>
ExchangeSourceGroupStatsByCellOwner(
    std::vector<SphericalObserver::SourceCellGroupEscapeStat> const& localStats,
    AdaptiveGroupSourceUpdateSummary& summary)
{
    summary.maxLocalSourceGroupStats =
        static_cast<unsigned long long>(localStats.size());

#ifdef RICH_MPI
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    std::vector<size_t> sendElements(static_cast<size_t>(ranks), 0);
    size_t sendTotal = 0;
    for (auto const& s : localStats) {
        if (!(s.energy > 0.0) || s.count == 0 || !std::isfinite(s.energy))
            continue;
        int owner = AdaptiveCellOwner(s.cellID, ranks);
        ++sendElements[static_cast<size_t>(owner)];
        ++sendTotal;
    }

    std::vector<int> sendCounts(static_cast<size_t>(ranks), 0);
    std::vector<int> recvCounts(static_cast<size_t>(ranks), 0);
    for (int r = 0; r < ranks; ++r) {
        sendCounts[static_cast<size_t>(r)] =
            CheckedByteCount(sendElements[static_cast<size_t>(r)],
                             sizeof(PackedSourceGroupEscapeStat),
                             "Adaptive source-group shard");
    }
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    std::vector<int> sendDispls(static_cast<size_t>(ranks), 0);
    std::vector<int> recvDispls(static_cast<size_t>(ranks), 0);
    unsigned long long totalSendBytes64 = 0;
    unsigned long long totalRecvBytes64 = 0;
    for (int r = 0; r < ranks; ++r) {
        sendDispls[static_cast<size_t>(r)] =
            CheckedByteTotal(totalSendBytes64, "Adaptive source-group shard send");
        recvDispls[static_cast<size_t>(r)] =
            CheckedByteTotal(totalRecvBytes64, "Adaptive source-group shard receive");
        totalSendBytes64 += static_cast<unsigned long long>(sendCounts[static_cast<size_t>(r)]);
        totalRecvBytes64 += static_cast<unsigned long long>(recvCounts[static_cast<size_t>(r)]);
    }
    int const totalSendBytes = CheckedByteTotal(totalSendBytes64,
                                                "Adaptive source-group shard send");
    int const totalRecvBytes = CheckedByteTotal(totalRecvBytes64,
                                                "Adaptive source-group shard receive");

    std::vector<PackedSourceGroupEscapeStat> sendData(sendTotal);
    std::vector<size_t> nextSendIndex(static_cast<size_t>(ranks), 0);
    for (int r = 0; r < ranks; ++r)
        nextSendIndex[static_cast<size_t>(r)] =
            static_cast<size_t>(sendDispls[static_cast<size_t>(r)]) /
            sizeof(PackedSourceGroupEscapeStat);
    for (auto const& s : localStats) {
        if (!(s.energy > 0.0) || s.count == 0 || !std::isfinite(s.energy))
            continue;
        int owner = AdaptiveCellOwner(s.cellID, ranks);
        sendData[nextSendIndex[static_cast<size_t>(owner)]++] =
            PackSourceGroupEscapeStat(s);
    }

    std::vector<PackedSourceGroupEscapeStat> recvData(
        static_cast<size_t>(totalRecvBytes) / sizeof(PackedSourceGroupEscapeStat));
    MPI_Alltoallv(sendData.empty() ? nullptr : sendData.data(), sendCounts.data(),
                  sendDispls.data(), MPI_BYTE,
                  recvData.empty() ? nullptr : recvData.data(), recvCounts.data(),
                  recvDispls.data(), MPI_BYTE, MPI_COMM_WORLD);

    unsigned long long localPairs = static_cast<unsigned long long>(localStats.size());
    unsigned long long recvPairs = static_cast<unsigned long long>(recvData.size());
    unsigned long long packedBytes = totalSendBytes64 + totalRecvBytes64;
    MPI_Allreduce(&localPairs, &summary.maxLocalSourceGroupStats, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&recvPairs, &summary.maxReceivedShardStats, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&recvPairs, &summary.mpiStatsExchanged, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&packedBytes, &summary.maxPackedBytes, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    (void)totalSendBytes;
    return recvData;
#else
    std::vector<PackedSourceGroupEscapeStat> result;
    result.reserve(localStats.size());
    for (auto const& s : localStats) {
        if (!(s.energy > 0.0) || s.count == 0 || !std::isfinite(s.energy))
            continue;
        result.push_back(PackSourceGroupEscapeStat(s));
    }
    summary.maxReceivedShardStats = static_cast<unsigned long long>(result.size());
    summary.mpiStatsExchanged = static_cast<unsigned long long>(result.size());
    summary.maxPackedBytes =
        static_cast<unsigned long long>(result.size()) *
        sizeof(PackedSourceGroupEscapeStat);
    return result;
#endif
}

std::vector<PackedAdaptiveCellGroupScoreDelta>
AllgatherAdaptiveCellGroupScoreDeltas(
    std::vector<PackedAdaptiveCellGroupScoreDelta> const& localDeltas)
{
#ifdef RICH_MPI
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int localBytes = CheckedByteCount(localDeltas.size(),
                                      sizeof(PackedAdaptiveCellGroupScoreDelta),
                                      "Adaptive source-group score delta packet");
    std::vector<int> counts(static_cast<size_t>(ranks), 0);
    MPI_Allgather(&localBytes, 1, MPI_INT, counts.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);

    std::vector<int> displs(static_cast<size_t>(ranks), 0);
    unsigned long long totalBytes64 = 0;
    for (int r = 0; r < ranks; ++r) {
        displs[static_cast<size_t>(r)] =
            CheckedByteTotal(totalBytes64, "Adaptive source-group score delta");
        totalBytes64 += static_cast<unsigned long long>(counts[static_cast<size_t>(r)]);
    }
    int const totalBytes =
        CheckedByteTotal(totalBytes64, "Adaptive source-group score delta");

    std::vector<PackedAdaptiveCellGroupScoreDelta> result(
        static_cast<size_t>(totalBytes) / sizeof(PackedAdaptiveCellGroupScoreDelta));
    MPI_Allgatherv(localDeltas.empty() ? nullptr : localDeltas.data(), localBytes,
                   MPI_BYTE, result.empty() ? nullptr : result.data(),
                   counts.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD);
    return result;
#else
    return localDeltas;
#endif
}

std::vector<PackedSourceEscapeStat>
ExchangeSourceStatsByCellOwner(std::vector<SphericalObserver::SourceCellEscapeStat> const& localStats,
                               AdaptiveSourceUpdateSummary& summary)
{
    summary.maxLocalSourcePairs = static_cast<unsigned long long>(localStats.size());

#ifdef RICH_MPI
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    std::vector<size_t> sendElements(static_cast<size_t>(ranks), 0);
    size_t sendTotal = 0;
    for (auto const& s : localStats) {
        if (!(s.energy > 0.0) || s.count == 0 || !std::isfinite(s.energy))
            continue;
        int owner = AdaptiveCellOwner(s.cellID, ranks);
        ++sendElements[static_cast<size_t>(owner)];
        ++sendTotal;
    }

    std::vector<int> sendCounts(static_cast<size_t>(ranks), 0);
    std::vector<int> recvCounts(static_cast<size_t>(ranks), 0);
    for (int r = 0; r < ranks; ++r) {
        sendCounts[static_cast<size_t>(r)] =
            CheckedByteCount(sendElements[static_cast<size_t>(r)],
                             sizeof(PackedSourceEscapeStat),
                             "Adaptive source shard");
    }
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> sendDispls(static_cast<size_t>(ranks), 0);
    std::vector<int> recvDispls(static_cast<size_t>(ranks), 0);
    unsigned long long totalSendBytes64 = 0;
    unsigned long long totalRecvBytes64 = 0;
    for (int r = 0; r < ranks; ++r) {
        sendDispls[static_cast<size_t>(r)] =
            CheckedByteTotal(totalSendBytes64, "Adaptive source shard send");
        recvDispls[static_cast<size_t>(r)] =
            CheckedByteTotal(totalRecvBytes64, "Adaptive source shard receive");
        totalSendBytes64 += static_cast<unsigned long long>(sendCounts[static_cast<size_t>(r)]);
        totalRecvBytes64 += static_cast<unsigned long long>(recvCounts[static_cast<size_t>(r)]);
    }
    int const totalSendBytes = CheckedByteTotal(totalSendBytes64, "Adaptive source shard send");
    int const totalRecvBytes = CheckedByteTotal(totalRecvBytes64, "Adaptive source shard receive");

    std::vector<PackedSourceEscapeStat> sendData(sendTotal);
    std::vector<size_t> nextSendIndex(static_cast<size_t>(ranks), 0);
    for (int r = 0; r < ranks; ++r)
        nextSendIndex[static_cast<size_t>(r)] =
            static_cast<size_t>(sendDispls[static_cast<size_t>(r)]) / sizeof(PackedSourceEscapeStat);
    for (auto const& s : localStats) {
        if (!(s.energy > 0.0) || s.count == 0 || !std::isfinite(s.energy))
            continue;
        int owner = AdaptiveCellOwner(s.cellID, ranks);
        sendData[nextSendIndex[static_cast<size_t>(owner)]++] = PackSourceEscapeStat(s);
    }
    std::vector<PackedSourceEscapeStat> recvData(static_cast<size_t>(totalRecvBytes) /
                                                 sizeof(PackedSourceEscapeStat));

    MPI_Alltoallv(sendData.empty() ? nullptr : sendData.data(), sendCounts.data(),
                  sendDispls.data(), MPI_BYTE,
                  recvData.empty() ? nullptr : recvData.data(), recvCounts.data(),
                  recvDispls.data(), MPI_BYTE, MPI_COMM_WORLD);

    unsigned long long localPairs = static_cast<unsigned long long>(localStats.size());
    unsigned long long recvPairs = static_cast<unsigned long long>(recvData.size());
    unsigned long long packedBytes = totalSendBytes64 + totalRecvBytes64;
    MPI_Allreduce(&localPairs, &summary.maxLocalSourcePairs, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&recvPairs, &summary.maxReceivedShardPairs, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&packedBytes, &summary.maxPackedBytes, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
    return recvData;
#else
    std::vector<PackedSourceEscapeStat> result;
    result.reserve(localStats.size());
    for (auto const& s : localStats) {
        if (!(s.energy > 0.0) || s.count == 0 || !std::isfinite(s.energy))
            continue;
        result.push_back(PackSourceEscapeStat(s));
    }
    summary.maxReceivedShardPairs = static_cast<unsigned long long>(result.size());
    summary.maxPackedBytes =
        static_cast<unsigned long long>(result.size()) * sizeof(PackedSourceEscapeStat);
    return result;
#endif
}

std::vector<PackedAdaptiveScoreDelta>
AllgatherAdaptiveScoreDeltas(std::vector<PackedAdaptiveScoreDelta> const& localDeltas)
{
#ifdef RICH_MPI
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int localBytes = CheckedByteCount(localDeltas.size(), sizeof(PackedAdaptiveScoreDelta),
                                      "Adaptive source score delta packet");
    std::vector<int> counts(static_cast<size_t>(ranks), 0);
    MPI_Allgather(&localBytes, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> displs(static_cast<size_t>(ranks), 0);
    unsigned long long totalBytes64 = 0;
    for (int r = 0; r < ranks; ++r) {
        displs[static_cast<size_t>(r)] =
            CheckedByteTotal(totalBytes64, "Adaptive source score delta");
        totalBytes64 += static_cast<unsigned long long>(counts[static_cast<size_t>(r)]);
    }
    int const totalBytes = CheckedByteTotal(totalBytes64, "Adaptive source score delta");

    std::vector<PackedAdaptiveScoreDelta> result(static_cast<size_t>(totalBytes) /
                                                 sizeof(PackedAdaptiveScoreDelta));
    MPI_Allgatherv(localDeltas.empty() ? nullptr : localDeltas.data(), localBytes, MPI_BYTE,
                   result.empty() ? nullptr : result.data(), counts.data(), displs.data(),
                   MPI_BYTE, MPI_COMM_WORLD);
    return result;
#else
    return localDeltas;
#endif
}

std::vector<SphericalObserver::SourceCellEscapeStat>
GatherTopSourceStats(std::vector<SphericalObserver::SourceCellEscapeStat> const& localStats)
{
    constexpr size_t TOP_N = 10;

    std::vector<SphericalObserver::SourceCellEscapeStat> localTop;
    localTop.reserve(TOP_N);
    for (auto const& s : localStats) {
        if (localTop.size() < TOP_N) {
            localTop.push_back(s);
            continue;
        }

        auto minIt = std::min_element(localTop.begin(), localTop.end(),
                                      [](auto const& a, auto const& b) {
                                          return a.energy < b.energy;
                                      });
        if (minIt != localTop.end() && s.energy > minIt->energy)
            *minIt = s;
    }
    std::sort(localTop.begin(), localTop.end(),
              [](auto const& a, auto const& b) { return a.energy > b.energy; });

#ifdef RICH_MPI
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    std::vector<PackedSourceEscapeStat> send(TOP_N);
    for (size_t i = 0; i < localTop.size(); ++i)
        send[i] = PackSourceEscapeStat(localTop[i]);

    std::vector<PackedSourceEscapeStat> recv;
    if (rank == 0)
        recv.resize(TOP_N * static_cast<size_t>(ranks));
    MPI_Gather(send.data(), static_cast<int>(TOP_N * sizeof(PackedSourceEscapeStat)), MPI_BYTE,
               rank == 0 ? recv.data() : nullptr,
               static_cast<int>(TOP_N * sizeof(PackedSourceEscapeStat)), MPI_BYTE,
               0, MPI_COMM_WORLD);

    std::vector<SphericalObserver::SourceCellEscapeStat> result;
    if (rank == 0) {
        for (auto const& p : recv) {
            if (p.count == 0 || !(p.energy > 0.0) || !std::isfinite(p.energy))
                continue;
            result.push_back(UnpackSourceEscapeStat(p));
        }
        std::sort(result.begin(), result.end(),
                  [](auto const& a, auto const& b) { return a.energy > b.energy; });
        if (result.size() > TOP_N)
            result.resize(TOP_N);
    }
    return result;
#else
    return localTop;
#endif
}

void ReduceDoubleVector(std::vector<double>& values)
{
#ifdef RICH_MPI
    if (!values.empty())
        MPI_Allreduce(MPI_IN_PLACE, values.data(), static_cast<int>(values.size()),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    (void)values;
#endif
}

void ReduceUnsignedLongLongVector(std::vector<unsigned long long>& values)
{
#ifdef RICH_MPI
    if (!values.empty())
        MPI_Allreduce(MPI_IN_PLACE, values.data(), static_cast<int>(values.size()),
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#else
    (void)values;
#endif
}

SphericalObserver::ObserverQualitySnapshot
CollectGlobalObserverQuality(SphericalObserver::ObserverQualitySnapshot local)
{
    ReduceDoubleVector(local.energy);
    ReduceDoubleVector(local.energyWeightSq);
    ReduceUnsignedLongLongVector(local.crossingCount);
    ReduceDoubleVector(local.stokesQ);
    ReduceDoubleVector(local.stokesU);
    ReduceDoubleVector(local.polarizationWeightSq);
    ReduceDoubleVector(local.sumWQ2);
    ReduceDoubleVector(local.sumWU2);
#ifdef RICH_MPI
    int polEnabled = local.polarizationEnabled ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &polEnabled, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    local.polarizationEnabled = (polEnabled != 0);
#endif
    return local;
}

double Percentile(std::vector<double> values, double p)
{
    values.erase(std::remove_if(values.begin(), values.end(),
                 [](double x) { return !std::isfinite(x); }), values.end());
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    double const pos = std::clamp(p, 0.0, 1.0) *
                       static_cast<double>(values.size() - 1);
    size_t const lo = static_cast<size_t>(std::floor(pos));
    size_t const hi = std::min(values.size() - 1, lo + 1);
    double const t = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - t) + values[hi] * t;
}

ObserverQualityDiagnostics BuildObserverQualityDiagnostics(
    SphericalObserver::ObserverQualitySnapshot const& snap,
    Config const& cfg,
    AdaptiveSourceState& state,
    bool includeInIntegratedStats)
{
    ObserverQualityDiagnostics diag;
    diag.enabled = cfg.adaptiveSourceCells && cfg.adaptiveObserverEquity;
    diag.polarizationMode = diag.enabled && cfg.polarization && snap.polarizationEnabled;
    diag.observerCount = snap.energy.size();
    diag.budgetMultiplier = 1.0;

    if (!diag.enabled || diag.observerCount == 0) {
        state.observerBudgetMultiplier = 1.0;
        return diag;
    }

    if (state.cumulativeObserverEnergy.size() != diag.observerCount) {
        state.cumulativeObserverEnergy.assign(diag.observerCount, 0.0);
        state.cumulativeObserverEnergyWeightSq.assign(diag.observerCount, 0.0);
        state.cumulativeObserverPolarizationWeightSq.assign(diag.observerCount, 0.0);
        state.cumulativeObserverStokesQ.assign(diag.observerCount, 0.0);
        state.cumulativeObserverStokesU.assign(diag.observerCount, 0.0);
        state.cumulativeObserverSumWQ2.assign(diag.observerCount, 0.0);
        state.cumulativeObserverSumWU2.assign(diag.observerCount, 0.0);
        state.cumulativeObserverCrossings.assign(diag.observerCount, 0ULL);
    }

    if (includeInIntegratedStats) {
        for (size_t i = 0; i < diag.observerCount; ++i) {
            state.cumulativeObserverEnergy[i] += snap.energy[i];
            if (i < snap.energyWeightSq.size())
                state.cumulativeObserverEnergyWeightSq[i] += snap.energyWeightSq[i];
            if (diag.polarizationMode) {
                if (i < snap.polarizationWeightSq.size())
                    state.cumulativeObserverPolarizationWeightSq[i] += snap.polarizationWeightSq[i];
                if (i < snap.stokesQ.size())
                    state.cumulativeObserverStokesQ[i] += snap.stokesQ[i];
                if (i < snap.stokesU.size())
                    state.cumulativeObserverStokesU[i] += snap.stokesU[i];
                if (i < snap.sumWQ2.size())
                    state.cumulativeObserverSumWQ2[i] += snap.sumWQ2[i];
                if (i < snap.sumWU2.size())
                    state.cumulativeObserverSumWU2[i] += snap.sumWU2[i];
            }
            if (i < snap.crossingCount.size())
                state.cumulativeObserverCrossings[i] += snap.crossingCount[i];
        }
    }

    std::vector<double> rawDeficit(diag.observerCount, 1.0);
    diag.neffByObserver.assign(diag.observerCount, 0.0);
    diag.snrByObserver.assign(diag.observerCount, 0.0);
    diag.crossingsByObserver = state.cumulativeObserverCrossings;

    for (size_t i = 0; i < diag.observerCount; ++i) {
        double const energy = state.cumulativeObserverEnergy[i];
        double const luminosityW2 = state.cumulativeObserverEnergyWeightSq[i];
        double neff = 0.0;
        if (energy > 0.0 && luminosityW2 > 0.0 &&
            std::isfinite(energy) && std::isfinite(luminosityW2))
            neff = energy * energy / luminosityW2;
        diag.neffByObserver[i] = neff;

        double deficit = 1.0;
        if (!includeInIntegratedStats) {
            deficit = 1.0;
        } else if (neff > 0.0) {
            deficit = std::max(deficit, cfg.adaptiveObserverTargetNeff / neff);
        } else {
            deficit = cfg.adaptiveObserverDeficitMax;
        }

        if (diag.polarizationMode && energy > 0.0) {
            auto const quality = polarization_statistics::ComputeQuality(
                energy,
                state.cumulativeObserverStokesQ[i],
                state.cumulativeObserverStokesU[i],
                state.cumulativeObserverPolarizationWeightSq[i],
                state.cumulativeObserverSumWQ2[i],
                state.cumulativeObserverSumWU2[i]);
            diag.snrByObserver[i] = quality.snr;
            if (!includeInIntegratedStats)
                deficit = 1.0;
            else if (quality.uncertaintyValid && quality.snr > 0.0)
                deficit = std::max(
                    deficit, cfg.adaptiveObserverTargetPolSnr / quality.snr);
            else
                deficit = cfg.adaptiveObserverDeficitMax;
        }

        rawDeficit[i] = std::clamp(deficit, 1.0, cfg.adaptiveObserverDeficitMax);
    }

    if (state.observerDeficitByIndex.size() != diag.observerCount)
        state.observerDeficitByIndex.assign(diag.observerCount, 1.0);
    diag.deficitByObserver.resize(diag.observerCount, 1.0);

    double deficitSum = 0.0;
    diag.deficitMin = std::numeric_limits<double>::max();
    diag.deficitMax = 1.0;
    for (size_t i = 0; i < diag.observerCount; ++i) {
        double const oldDeficit = state.observerDeficitByIndex[i];
        double const smooth = oldDeficit * (1.0 - cfg.adaptiveObserverDeficitEma)
                            + rawDeficit[i] * cfg.adaptiveObserverDeficitEma;
        double const finalDeficit = includeInIntegratedStats
            ? std::clamp(smooth, 1.0, cfg.adaptiveObserverDeficitMax)
            : 1.0;
        if (includeInIntegratedStats)
            state.observerDeficitByIndex[i] = finalDeficit;
        diag.deficitByObserver[i] = finalDeficit;
        deficitSum += finalDeficit;
        diag.deficitMin = std::min(diag.deficitMin, finalDeficit);
        diag.deficitMax = std::max(diag.deficitMax, finalDeficit);
        if (finalDeficit > 1.0001)
            ++diag.weakObservers;
        unsigned long long const crossings =
            (i < diag.crossingsByObserver.size()) ? diag.crossingsByObserver[i] : 0ULL;
        if (diag.neffByObserver[i] <= 0.0 || crossings == 0)
            ++diag.zeroStatObservers;
    }
    if (diag.deficitMin == std::numeric_limits<double>::max())
        diag.deficitMin = 1.0;
    diag.deficitAvg = deficitSum / static_cast<double>(diag.observerCount);

    diag.neffP05 = Percentile(diag.neffByObserver, 0.05);
    diag.neffMedian = Percentile(diag.neffByObserver, 0.50);
    diag.neffP95 = Percentile(diag.neffByObserver, 0.95);
    diag.snrP05 = Percentile(diag.snrByObserver, 0.05);
    diag.snrMedian = Percentile(diag.snrByObserver, 0.50);
    diag.snrP95 = Percentile(diag.snrByObserver, 0.95);

    double const weakFrac = static_cast<double>(diag.weakObservers) /
                        static_cast<double>(diag.observerCount);

    // The old weakFrac-only driver barely responds when only a few observers are
    // terrible.  Use deficit severity as well, so low-SNR / low-Neff tails get
    // meaningful extra budget.
    double const deficitTail95 = Percentile(diag.deficitByObserver, 0.95);

    double deficitDriver = 0.0;
    deficitDriver = std::max(deficitDriver, diag.deficitAvg - 1.0);
    deficitDriver = std::max(deficitDriver, 0.25 * (deficitTail95 - 1.0));
    deficitDriver = std::max(deficitDriver, 0.10 * (diag.deficitMax - 1.0));

    // Keep a small weak-fraction term so many mildly weak observers still increase
    // budget, but do not rely on it for a few pathological observers.
    deficitDriver = std::max(deficitDriver, weakFrac);

    diag.budgetMultiplier =
        1.0 + cfg.adaptiveObserverExtraBudgetFrac * std::max(0.0, deficitDriver);

    // Hard safety cap.  Increase this only if you are prepared for the memory/runtime
    // cost.  With --adaptive-observer-extra-budget-frac 2, this can still get large
    // when deficitMax is 100.
    diag.budgetMultiplier = std::min(diag.budgetMultiplier, 10.0);

    state.observerBudgetMultiplier = diag.budgetMultiplier;
    return diag;
}

RadiationIMC::SourceAllocationSummary
ReduceSourceAllocationSummary(RadiationIMC::SourceAllocationSummary local)
{
#ifdef RICH_MPI
    unsigned long long const localSourceCells = local.sourceCells;
    unsigned long long const localLearnedCells = local.learnedCells;
    unsigned long long sums[8] = {
        local.totalPhotons,
        local.sourceCells,
        local.boostedCells,
        local.learnedCells,
        local.learnedBoostedCells,
        local.learnedPhotons,
        local.learnedExtraPhotons,
        0
    };
    MPI_Allreduce(MPI_IN_PLACE, sums, 8, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    local.totalPhotons = sums[0];
    local.sourceCells = sums[1];
    local.boostedCells = sums[2];
    local.learnedCells = sums[3];
    local.learnedBoostedCells = sums[4];
    local.learnedPhotons = sums[5];
    local.learnedExtraPhotons = sums[6];

    unsigned long long minPhotons = localSourceCells > 0
        ? static_cast<unsigned long long>(local.minPhotons)
        : ULLONG_MAX;
    unsigned long long maxPhotons = static_cast<unsigned long long>(local.maxPhotons);
    unsigned long long learnedMinPhotons = localLearnedCells > 0
        ? static_cast<unsigned long long>(local.learnedMinPhotons)
        : ULLONG_MAX;
    unsigned long long learnedMaxPhotons = static_cast<unsigned long long>(local.learnedMaxPhotons);
    MPI_Allreduce(MPI_IN_PLACE, &minPhotons, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxPhotons, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &learnedMinPhotons, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &learnedMaxPhotons, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    local.minPhotons = (minPhotons == ULLONG_MAX) ? 0 : static_cast<size_t>(minPhotons);
    local.maxPhotons = static_cast<size_t>(maxPhotons);
    local.learnedMinPhotons = (learnedMinPhotons == ULLONG_MAX) ? 0 : static_cast<size_t>(learnedMinPhotons);
    local.learnedMaxPhotons = static_cast<size_t>(learnedMaxPhotons);

    MPI_Allreduce(MPI_IN_PLACE, &local.adaptiveScoreSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    int adaptive = local.adaptiveEnabled ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &adaptive, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    local.adaptiveEnabled = adaptive != 0;
#endif
    return local;
}

namespace {

std::vector<std::pair<size_t, size_t>> MakePhotonHistogramEdges(
    size_t binMinPhotons,
    size_t binMaxPhotons)
{
    std::vector<std::pair<size_t, size_t>> edges;
    if (binMinPhotons == 0)
        binMinPhotons = 1;
    if (binMaxPhotons < binMinPhotons)
        binMaxPhotons = binMinPhotons;

    edges.emplace_back(0, 0);
    if (binMinPhotons > 1)
        edges.emplace_back(1, binMinPhotons - 1);

    size_t const width = 200;
    for (size_t lo = binMinPhotons; lo <= binMaxPhotons; lo += width) {
        size_t const hi = std::min(lo + width - 1, binMaxPhotons);
        edges.emplace_back(lo, hi);
    }
    if (edges.empty() || edges.back().second < binMaxPhotons)
        edges.emplace_back(binMaxPhotons, binMaxPhotons);
    edges.emplace_back(binMaxPhotons + 1, std::numeric_limits<size_t>::max() / 4);

    std::vector<std::pair<size_t, size_t>> uniqueEdges;
    uniqueEdges.reserve(edges.size());
    for (auto const& edge : edges) {
        if (!uniqueEdges.empty() &&
            uniqueEdges.back().first == edge.first &&
            uniqueEdges.back().second == edge.second)
            continue;
        uniqueEdges.push_back(edge);
    }
    return uniqueEdges;
}

size_t BinIndexForPhotonCount(
    std::vector<std::pair<size_t, size_t>> const& edges,
    size_t photons)
{
    for (size_t i = 0; i < edges.size(); ++i) {
        if (photons >= edges[i].first && photons <= edges[i].second)
            return i;
    }
    return edges.empty() ? 0 : edges.size() - 1;
}

void ComputePhotonPercentiles(
    std::vector<size_t> const& photonCounts,
    SourcePhotonDistribution& dist)
{
    if (photonCounts.empty())
        return;

    std::vector<double> values;
    values.reserve(photonCounts.size());
    for (size_t const n : photonCounts)
        values.push_back(static_cast<double>(n));

    dist.p05 = Percentile(values, 0.05);
    dist.p25 = Percentile(values, 0.25);
    dist.p50 = Percentile(values, 0.50);
    dist.p75 = Percentile(values, 0.75);
    dist.p95 = Percentile(values, 0.95);
}

} // namespace

SourcePhotonDistribution ReduceSourcePhotonDistribution(
    std::vector<size_t> const& localPhotonsPerCell,
    size_t binMinPhotons,
    size_t binMaxPhotons,
    int rank,
    int mpiSize)
{
    auto const edges = MakePhotonHistogramEdges(binMinPhotons, binMaxPhotons);
    SourcePhotonDistribution dist;
    dist.bins.resize(edges.size());
    for (size_t i = 0; i < edges.size(); ++i) {
        dist.bins[i].lowerInclusive = edges[i].first;
        dist.bins[i].upperInclusive = edges[i].second;
    }

    std::vector<size_t> localEmitting;
    localEmitting.reserve(localPhotonsPerCell.size());
    for (size_t const photons : localPhotonsPerCell) {
        if (photons == 0)
            continue;
        size_t const bin = BinIndexForPhotonCount(edges, photons);
        ++dist.bins[bin].cellCount;
        dist.bins[bin].photonCount += static_cast<unsigned long long>(photons);
        ++dist.emittingCells;
        dist.totalPhotons += static_cast<unsigned long long>(photons);
        localEmitting.push_back(photons);
    }

#ifdef RICH_MPI
    if (!dist.bins.empty()) {
        std::vector<unsigned long long> localCellCounts(dist.bins.size(), 0);
        std::vector<unsigned long long> localPhotonCounts(dist.bins.size(), 0);
        for (size_t i = 0; i < dist.bins.size(); ++i) {
            localCellCounts[i] = dist.bins[i].cellCount;
            localPhotonCounts[i] = dist.bins[i].photonCount;
        }
        MPI_Allreduce(MPI_IN_PLACE, localCellCounts.data(),
                      static_cast<int>(localCellCounts.size()),
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, localPhotonCounts.data(),
                      static_cast<int>(localPhotonCounts.size()),
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        for (size_t i = 0; i < dist.bins.size(); ++i) {
            dist.bins[i].cellCount = localCellCounts[i];
            dist.bins[i].photonCount = localPhotonCounts[i];
        }
    }

    unsigned long long localEmittingCells = dist.emittingCells;
    unsigned long long localTotalPhotons = dist.totalPhotons;
    MPI_Allreduce(MPI_IN_PLACE, &localEmittingCells, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &localTotalPhotons, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    dist.emittingCells = localEmittingCells;
    dist.totalPhotons = localTotalPhotons;

    int const localCount = static_cast<int>(localEmitting.size());
    std::vector<int> counts(static_cast<size_t>(mpiSize), 0);
    std::vector<int> displs(static_cast<size_t>(mpiSize), 0);
    MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);

    int total = 0;
    for (int r = 0; r < mpiSize; ++r) {
        displs[static_cast<size_t>(r)] = total;
        total += counts[static_cast<size_t>(r)];
    }

    std::vector<unsigned long long> gathered;
    if (rank == 0)
        gathered.resize(static_cast<size_t>(total));
    if (total > 0) {
        MPI_Gatherv(
            localEmitting.empty() ? nullptr : localEmitting.data(),
            localCount,
            MPI_UNSIGNED_LONG_LONG,
            rank == 0 ? gathered.data() : nullptr,
            counts.data(),
            displs.data(),
            MPI_UNSIGNED_LONG_LONG,
            0,
            MPI_COMM_WORLD);
    }

    if (rank == 0) {
        std::vector<size_t> globalEmitting;
        globalEmitting.reserve(gathered.size());
        for (unsigned long long const n : gathered)
            globalEmitting.push_back(static_cast<size_t>(n));
        ComputePhotonPercentiles(globalEmitting, dist);
    }

    double percentiles[5] = {
        dist.p05, dist.p25, dist.p50, dist.p75, dist.p95};
    MPI_Bcast(percentiles, 5, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    dist.p05 = percentiles[0];
    dist.p25 = percentiles[1];
    dist.p50 = percentiles[2];
    dist.p75 = percentiles[3];
    dist.p95 = percentiles[4];
#else
    ComputePhotonPercentiles(localEmitting, dist);
#endif

    return dist;
}

void PrintSourcePhotonDistribution(
    std::string const& label,
    SourcePhotonDistribution const& dist,
    int rank)
{
    if (rank != 0 || dist.emittingCells == 0)
        return;

    std::cout << label << " source photons/cell distribution:"
              << " emitting_cells=" << dist.emittingCells
              << " total_photons=" << dist.totalPhotons
              << " p05/p25/p50/p75/p95="
              << dist.p05 << "/" << dist.p25 << "/" << dist.p50
              << "/" << dist.p75 << "/" << dist.p95
              << std::endl;

    std::cout << label << " source photons/cell histogram:" << std::endl;
    for (auto const& bin : dist.bins) {
        if (bin.cellCount == 0)
            continue;
        double const cellFrac = static_cast<double>(bin.cellCount) /
            static_cast<double>(dist.emittingCells);
        double const photonFrac = dist.totalPhotons > 0
            ? static_cast<double>(bin.photonCount) /
                  static_cast<double>(dist.totalPhotons)
            : 0.0;
        std::cout << "  [" << bin.lowerInclusive << "," << bin.upperInclusive
                  << "] cells=" << bin.cellCount
                  << " cell_frac=" << std::fixed << std::setprecision(4)
                  << cellFrac
                  << " photons=" << bin.photonCount
                  << " photon_frac=" << photonFrac
                  << std::endl;
    }
    std::cout << std::defaultfloat;
}

RadiationIMC::GroupSamplingDiagnostics
ReduceGroupSamplingDiagnostics(RadiationIMC::GroupSamplingDiagnostics local)
{
#ifdef RICH_MPI
    size_t const localWeightCorrectionCount = local.weightCorrectionCount;
    unsigned long long sums[7] = {
        static_cast<unsigned long long>(local.totalSampled),
        static_cast<unsigned long long>(local.weightCorrectionCount),
        static_cast<unsigned long long>(local.weightCorrectionCapped),
        static_cast<unsigned long long>(local.weightCorrectionFallback),
        static_cast<unsigned long long>(local.invalidPdfFallback),
        static_cast<unsigned long long>(local.invalidPdfFallbackPackets),
        local.estimatorPotentiallyBiased ? 1ULL : 0ULL
    };
    MPI_Allreduce(MPI_IN_PLACE, sums, 7, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    local.totalSampled = static_cast<size_t>(sums[0]);
    local.weightCorrectionCount = static_cast<size_t>(sums[1]);
    local.weightCorrectionCapped = static_cast<size_t>(sums[2]);
    local.weightCorrectionFallback = static_cast<size_t>(sums[3]);
    local.invalidPdfFallback = static_cast<size_t>(sums[4]);
    local.invalidPdfFallbackPackets = static_cast<size_t>(sums[5]);
    local.estimatorPotentiallyBiased = sums[6] > 0;

    unsigned long long cellsWithGroupScores =
        static_cast<unsigned long long>(local.cellsWithGroupScores);
    MPI_Allreduce(MPI_IN_PLACE, &cellsWithGroupScores, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    local.cellsWithGroupScores = static_cast<size_t>(cellsWithGroupScores);

    double doubleSums[3] = {
        local.weightCorrectionSum,
        local.sampledEnergy,
        local.cappedEnergy
    };
    MPI_Allreduce(MPI_IN_PLACE, doubleSums, 3, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    local.weightCorrectionSum = doubleSums[0];
    local.sampledEnergy = doubleSums[1];
    local.cappedEnergy = doubleSums[2];

    double minCorr = localWeightCorrectionCount > 0
        ? local.weightCorrectionMin
        : std::numeric_limits<double>::infinity();
    double maxCorr = localWeightCorrectionCount > 0
        ? local.weightCorrectionMax
        : 1.0;
    MPI_Allreduce(MPI_IN_PLACE, &minCorr, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxCorr, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    local.weightCorrectionMin = std::isfinite(minCorr) ? minCorr : 1.0;
    local.weightCorrectionMax = local.weightCorrectionCount > 0 ? maxCorr : 1.0;
#endif
    local.cappedEnergyFraction = local.sampledEnergy > 0.0
        ? local.cappedEnergy / local.sampledEnergy
        : 0.0;
    local.estimatorPotentiallyBiased =
        local.estimatorPotentiallyBiased ||
        local.weightCorrectionCapped > 0 ||
        local.cappedEnergy > 0.0;
    return local;
}

void AccumulateGroupSamplingDiagnostics(
    RadiationIMC::GroupSamplingDiagnostics& total,
    RadiationIMC::GroupSamplingDiagnostics const& gen)
{
    if (gen.weightCorrectionCount > 0) {
        if (total.weightCorrectionCount == 0) {
            total.weightCorrectionMin = gen.weightCorrectionMin;
            total.weightCorrectionMax = gen.weightCorrectionMax;
        } else {
            total.weightCorrectionMin =
                std::min(total.weightCorrectionMin, gen.weightCorrectionMin);
            total.weightCorrectionMax =
                std::max(total.weightCorrectionMax, gen.weightCorrectionMax);
        }
    }

    total.totalSampled += gen.totalSampled;
    total.cellsWithGroupScores =
        std::max(total.cellsWithGroupScores, gen.cellsWithGroupScores);
    total.weightCorrectionSum += gen.weightCorrectionSum;
    total.weightCorrectionCount += gen.weightCorrectionCount;
    total.weightCorrectionCapped += gen.weightCorrectionCapped;
    total.weightCorrectionFallback += gen.weightCorrectionFallback;
    total.invalidPdfFallback += gen.invalidPdfFallback;
    total.invalidPdfFallbackPackets += gen.invalidPdfFallbackPackets;
    total.sampledEnergy += gen.sampledEnergy;
    total.cappedEnergy += gen.cappedEnergy;
    total.estimatorPotentiallyBiased =
        total.estimatorPotentiallyBiased || gen.estimatorPotentiallyBiased;
    total.cappedEnergyFraction = total.sampledEnergy > 0.0
        ? total.cappedEnergy / total.sampledEnergy
        : 0.0;
}

AdaptiveSourceUpdateSummary UpdateAdaptiveSourceScoresDistributed(
    std::vector<SphericalObserver::SourceCellEscapeStat> const& localStats,
    Config const& cfg,
    AdaptiveSourceState& state,
    ObserverQualityDiagnostics const& observerQuality,
    bool decayExistingScores,
    int /*rank*/,
    int /*mpiSize*/)
{
    AdaptiveSourceUpdateSummary summary;
    size_t const nObs = cfg.nObservers;
    std::vector<double> energyByObserver(nObs, 0.0);
    std::vector<double> weightSqByObserver(nObs, 0.0);
    std::vector<unsigned long long> crossingsByObserver(nObs, 0ULL);

    for (auto const& s : localStats) {
        if (s.observerIndex >= nObs || !(s.energy > 0.0) || !std::isfinite(s.energy))
            continue;

        energyByObserver[s.observerIndex] += s.energy;
        double const w2 = (s.weightSq > 0.0 && std::isfinite(s.weightSq))
            ? s.weightSq
            : s.energy * s.energy;
        weightSqByObserver[s.observerIndex] += w2;
        crossingsByObserver[s.observerIndex] += static_cast<unsigned long long>(s.count);
    }

#ifdef RICH_MPI
    if (!energyByObserver.empty()) {
        MPI_Allreduce(MPI_IN_PLACE, energyByObserver.data(), static_cast<int>(energyByObserver.size()),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, weightSqByObserver.data(), static_cast<int>(weightSqByObserver.size()),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, crossingsByObserver.data(), static_cast<int>(crossingsByObserver.size()),
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    }
#endif

    for (size_t obs = 0; obs < nObs; ++obs) {
        summary.totalEscapedEnergy += energyByObserver[obs];
        summary.totalCrossings += crossingsByObserver[obs];
        if (crossingsByObserver[obs] > 0)
            ++summary.observersWithCrossings;
    }

    if (!cfg.adaptiveSourceCells) {
        summary.scoreMapCells = state.scoreByCellID.size();
        return summary;
    }

    if (decayExistingScores) {
        double const decay = 1.0 - cfg.adaptiveSourceEma;
        summary.decayedCells = state.scoreByCellID.size();
        for (auto& kv : state.scoreByCellID) {
            kv.second *= decay;
        }
    }

    std::vector<PackedSourceEscapeStat> received = ExchangeSourceStatsByCellOwner(localStats, summary);
    std::unordered_map<AdaptivePairKey, SphericalObserver::SourceCellEscapeStat, AdaptivePairKeyHash> byPair;
    byPair.reserve(received.size());
    for (auto const& p : received) {
        if (p.count == 0 || !(p.energy > 0.0) || !std::isfinite(p.energy))
            continue;
        size_t const observerIndex = static_cast<size_t>(p.observerIndex);
        size_t const cellID = static_cast<size_t>(p.cellID);
        AdaptivePairKey const key{observerIndex, cellID};
        auto& s = byPair[key];
        s.cellID = cellID;
        s.observerIndex = observerIndex;
        s.energy += p.energy;
        s.weightSq += p.weightSq;
        s.maxWeight = std::max(s.maxWeight, p.maxWeight);
        s.count += static_cast<size_t>(p.count);
    }
    std::vector<PackedSourceEscapeStat>().swap(received);

    std::vector<SphericalObserver::SourceCellEscapeStat> ownedStats;
    ownedStats.reserve(byPair.size());
    for (auto const& kv : byPair)
        ownedStats.push_back(kv.second);
    std::unordered_map<AdaptivePairKey, SphericalObserver::SourceCellEscapeStat,
                       AdaptivePairKeyHash>().swap(byPair);
    summary.topStats = GatherTopSourceStats(ownedStats);

    unsigned long long localPairCount = static_cast<unsigned long long>(ownedStats.size());
    unsigned long long globalPairCount = localPairCount;
#ifdef RICH_MPI
    MPI_Allreduce(&localPairCount, &globalPairCount, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
#endif
    summary.sourceCellObserverPairs = static_cast<size_t>(globalPairCount);

    std::unordered_map<size_t, double> deltaByCell;
    std::unordered_set<size_t> touchedNewCells;
    for (auto const& s : ownedStats) {
        if (s.observerIndex >= energyByObserver.size() ||
            !(energyByObserver[s.observerIndex] > 0.0) ||
            !std::isfinite(energyByObserver[s.observerIndex]))
            continue;

        double observerBoost = 1.0;
        if (observerQuality.enabled &&
            s.observerIndex < observerQuality.deficitByObserver.size() &&
            observerQuality.deficitByObserver[s.observerIndex] > 0.0 &&
            std::isfinite(observerQuality.deficitByObserver[s.observerIndex]))
        {
            observerBoost = observerQuality.deficitByObserver[s.observerIndex];
        }

        double const eFrac = s.energy / energyByObserver[s.observerIndex];

        double const w2 = (s.weightSq > 0.0 && std::isfinite(s.weightSq))
            ? s.weightSq
            : s.energy * s.energy;

        double const w2Total = weightSqByObserver[s.observerIndex];

        double const w2Frac = (w2Total > 0.0 && std::isfinite(w2Total))
            ? w2 / w2Total
            : eFrac;

        // In polarization mode, variance matters much more than energy:
        // a cell emitting a few huge packets can dominate Q/U uncertainty.
        double const varianceMix = observerQuality.polarizationMode
            ? std::clamp(cfg.adaptiveSourceWeightScoreFrac, 0.0, 1.0)
            : 0.35;

        double const sourceQualityScore =
            (1.0 - varianceMix) * eFrac + varianceMix * w2Frac;

        double const minScore =
            cfg.adaptiveSourceMinEscapedFrac / std::max(1.0, observerBoost);

        if (!(sourceQualityScore >= minScore) || !std::isfinite(sourceQualityScore))
            continue;

        ++summary.passedCells;

        auto it = state.scoreByCellID.find(s.cellID);
        bool existed = (it != state.scoreByCellID.end() && it->second > 0.0);
        bool alreadyTouched = touchedNewCells.find(s.cellID) != touchedNewCells.end();
        if (existed || alreadyTouched)
            ++summary.retainedCells;
        else {
            ++summary.newCells;
            touchedNewCells.insert(s.cellID);
        }

        deltaByCell[s.cellID] += cfg.adaptiveSourceEma * observerBoost * sourceQualityScore;
    }

    std::vector<PackedAdaptiveScoreDelta> localDeltas;
    localDeltas.reserve(deltaByCell.size());
    for (auto const& kv : deltaByCell) {
        if (!(kv.second > 0.0) || !std::isfinite(kv.second))
            continue;
        PackedAdaptiveScoreDelta p;
        p.cellID = static_cast<unsigned long long>(kv.first);
        p.delta = kv.second;
        localDeltas.push_back(p);
    }
    std::unordered_map<size_t, double>().swap(deltaByCell);
    std::unordered_set<size_t>().swap(touchedNewCells);
    std::vector<SphericalObserver::SourceCellEscapeStat>().swap(ownedStats);

    size_t localDeltaCells = localDeltas.size();
    std::vector<PackedAdaptiveScoreDelta> allDeltas = AllgatherAdaptiveScoreDeltas(localDeltas);
    std::vector<PackedAdaptiveScoreDelta>().swap(localDeltas);
    for (auto const& p : allDeltas) {
        if (!(p.delta > 0.0) || !std::isfinite(p.delta))
            continue;
        state.scoreByCellID[static_cast<size_t>(p.cellID)] += p.delta;
    }
    size_t const scoreDeltaCells = allDeltas.size();
    std::vector<PackedAdaptiveScoreDelta>().swap(allDeltas);

    for (auto it = state.scoreByCellID.begin(); it != state.scoreByCellID.end(); ) {
        if (!(it->second > 0.0) || !std::isfinite(it->second))
            it = state.scoreByCellID.erase(it);
        else
            ++it;
    }

    unsigned long long localPassed = static_cast<unsigned long long>(summary.passedCells);
    unsigned long long localNew = static_cast<unsigned long long>(summary.newCells);
    unsigned long long localRetained = static_cast<unsigned long long>(summary.retainedCells);
    unsigned long long globalPassed = localPassed;
    unsigned long long globalNew = localNew;
    unsigned long long globalRetained = localRetained;
#ifdef RICH_MPI
    MPI_Allreduce(&localPassed, &globalPassed, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localNew, &globalNew, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localRetained, &globalRetained, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    summary.passedCells = static_cast<size_t>(globalPassed);
    summary.newCells = static_cast<size_t>(globalNew);
    summary.retainedCells = static_cast<size_t>(globalRetained);
    summary.scoreDeltaCells = scoreDeltaCells;
    summary.scoreMapCells = state.scoreByCellID.size();

    unsigned long long maxLocalDeltaCells = static_cast<unsigned long long>(localDeltaCells);
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &maxLocalDeltaCells, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
#endif
    summary.maxPackedBytes = std::max(
        summary.maxPackedBytes,
        (maxLocalDeltaCells + static_cast<unsigned long long>(scoreDeltaCells)) *
            static_cast<unsigned long long>(sizeof(PackedAdaptiveScoreDelta)));

    return summary;
}

void PrintAdaptiveGenerationStart(
    std::string const& label,
    Config const& cfg,
    AdaptiveSourceState const& state,
    size_t gen,
    size_t totalGenerations,
    size_t burninGenerations,
    bool adaptiveActive,
    int rank)
{
    if (rank != 0 || !cfg.adaptiveSourceCells)
        return;
    std::string mode = "burn-in";
    if (adaptiveActive)
        mode = (gen == burninGenerations) ? "first adaptive" : "adaptive";
    std::string postAdaptiveLB = "disabled";
    if (cfg.measuredLoadBalance)
        postAdaptiveLB = state.postAdaptiveMeasuredLBDone ? "done" : "pending";
    size_t burninRemaining = (gen < burninGenerations)
        ? burninGenerations - gen : 0;
    std::cout << label << " adaptive generation state: gen " << (gen + 1)
              << "/" << totalGenerations
              << " mode=" << mode
              << " burnin_remaining=" << burninRemaining
              << " learned_cells=" << state.scoreByCellID.size()
              << " post_adaptive_LB="
              << postAdaptiveLB
              << std::endl;
    if (gen == burninGenerations && !state.burninCompletePrinted)
        std::cout << label << " adaptive source weights active for first time" << std::endl;
}

void PrintAdaptiveIterationSummary(
    std::string const& label,
    AdaptiveSourceState const& state,
    AdaptiveSourceUpdateSummary const& update,
    RadiationIMC::SourceAllocationSummary const& allocation,
    ObserverQualityDiagnostics const& observerQuality,
    size_t gen,
    size_t totalGenerations,
    std::string const& phase,
    size_t photonsThisGen,
    bool finalThisGen,
    size_t finalGenerationIndex,
    size_t finalGenerations,
    bool adaptiveActive,
    bool includeInFinal,
    int rank)
{
    if (rank != 0)
        return;

    double const avgPhotons = allocation.sourceCells > 0
        ? static_cast<double>(allocation.totalPhotons) /
              static_cast<double>(allocation.sourceCells)
        : 0.0;
    double const learnedAvgPhotons = allocation.learnedCells > 0
        ? static_cast<double>(allocation.learnedPhotons) /
              static_cast<double>(allocation.learnedCells)
        : 0.0;
    double const learnedPhotonFrac = allocation.totalPhotons > 0
        ? static_cast<double>(allocation.learnedPhotons) /
              static_cast<double>(allocation.totalPhotons)
        : 0.0;
    double const boostedFrac = allocation.sourceCells > 0
        ? static_cast<double>(allocation.boostedCells) /
              static_cast<double>(allocation.sourceCells)
        : 0.0;
    bool const observerQualityHasAccumulatedStats =
        observerQuality.enabled &&
        std::any_of(observerQuality.crossingsByObserver.begin(),
                    observerQuality.crossingsByObserver.end(),
                    [](unsigned long long crossings) { return crossings > 0; });

    std::cout << "ITERATION_SUMMARY type=" << label
              << " iteration=" << (gen + 1) << "/" << totalGenerations
              << " phase=" << phase
              << " photons_per_cell=" << photonsThisGen
              << " final=" << (finalThisGen ? 1 : 0);
    if (finalThisGen)
        std::cout << " final_step=" << (finalGenerationIndex + 1)
                  << "/" << finalGenerations;
    std::cout << " include_in_final=" << (includeInFinal ? 1 : 0)
              << " adaptive_active=" << (adaptiveActive ? 1 : 0)
              << " source_cells=" << allocation.sourceCells
              << " total_photons=" << allocation.totalPhotons
              << " photons_per_cell_min/avg/max=" << allocation.minPhotons
              << "/" << avgPhotons << "/" << allocation.maxPhotons
              << " boosted_cells=" << allocation.boostedCells
              << " boosted_cell_frac=" << boostedFrac
              << " learned_score_cells=" << state.scoreByCellID.size()
              << " learned_cells_allocated=" << allocation.learnedCells
              << " learned_boosted_cells=" << allocation.learnedBoostedCells
              << " learned_photons=" << allocation.learnedPhotons
              << " learned_photon_frac=" << learnedPhotonFrac
              << " learned_avg_photons_per_cell=" << learnedAvgPhotons
              << " learned_extra_photons=" << allocation.learnedExtraPhotons
              << " crossing_energy=" << update.totalEscapedEnergy
              << " crossing_count=" << update.totalCrossings
              << " observers_with_crossings=" << update.observersWithCrossings;
    if (observerQualityHasAccumulatedStats) {
        std::cout << " weak_observers=" << observerQuality.weakObservers
                  << "/" << observerQuality.observerCount
                  << " zero_stat_observers=" << observerQuality.zeroStatObservers
                  << " deficit_min/avg/max=" << observerQuality.deficitMin
                  << "/" << observerQuality.deficitAvg
                  << "/" << observerQuality.deficitMax
                  << " observer_budget_multiplier="
                  << observerQuality.budgetMultiplier;
        if (observerQuality.polarizationMode)
            std::cout << " pol_snr_p05/med/p95=" << observerQuality.snrP05
                      << "/" << observerQuality.snrMedian
                      << "/" << observerQuality.snrP95;
    }
    std::cout << std::endl;
}

void PrintAdaptiveGenerationStats(
    std::string const& label,
    Config const& cfg,
    AdaptiveSourceState const& state,
    AdaptiveSourceUpdateSummary const& update,
    RadiationIMC::SourceAllocationSummary allocation,
    SourcePhotonDistribution const& photonDistribution,
    ObserverQualityDiagnostics const& observerQuality,
    size_t gen,
    size_t /*totalGenerations*/,
    size_t /*burninGenerations*/,
    bool /*adaptiveActive*/,
    int rank)
{
    if (rank != 0 || !cfg.adaptiveSourceCells)
        return;
    double avgPhotons = allocation.sourceCells > 0
        ? static_cast<double>(allocation.totalPhotons) / static_cast<double>(allocation.sourceCells)
        : 0.0;
    double learnedAvgPhotons = allocation.learnedCells > 0
        ? static_cast<double>(allocation.learnedPhotons) / static_cast<double>(allocation.learnedCells)
        : 0.0;
    double learnedPhotonFrac = allocation.totalPhotons > 0
        ? static_cast<double>(allocation.learnedPhotons) / static_cast<double>(allocation.totalPhotons)
        : 0.0;
    bool const observerQualityHasAccumulatedStats =
        observerQuality.enabled &&
        std::any_of(observerQuality.crossingsByObserver.begin(),
                    observerQuality.crossingsByObserver.end(),
                    [](unsigned long long crossings) { return crossings > 0; });
    std::cout << label << " adaptive stats after generation " << (gen + 1)
              << ": crossing_energy=" << update.totalEscapedEnergy
              << " crossing_count=" << update.totalCrossings
              << " source_cell_observer_pairs=" << update.sourceCellObserverPairs
              << " observers_with_crossings=" << update.observersWithCrossings
              << " cells_passing_filter=" << update.passedCells
              << " learned_cells=" << state.scoreByCellID.size()
              << " new=" << update.newCells
              << " retained=" << update.retainedCells
              << " decayed=" << update.decayedCells << "\n"
              << label << " source allocation used: adaptive="
              << (allocation.adaptiveEnabled ? "yes" : "no")
              << " total_photons=" << allocation.totalPhotons
              << " boosted_cells=" << allocation.boostedCells
              << " learned_cells_allocated=" << allocation.learnedCells
              << " learned_boosted_cells=" << allocation.learnedBoostedCells
              << " learned_photons=" << allocation.learnedPhotons
              << " learned_photon_frac=" << learnedPhotonFrac
              << " learned_extra_photons=" << allocation.learnedExtraPhotons
              << " photons/cell min/avg/max=" << allocation.minPhotons
              << "/" << avgPhotons
              << "/" << allocation.maxPhotons
              << " learned photons/cell min/avg/max=" << allocation.learnedMinPhotons
              << "/" << learnedAvgPhotons
              << "/" << allocation.learnedMaxPhotons
              << std::endl;

    std::cout << label << " emission learning effect: learned_score_cells="
              << state.scoreByCellID.size()
              << " learned_cells_allocated=" << allocation.learnedCells
              << " learned_boosted_cells=" << allocation.learnedBoostedCells
              << " boosted_cells=" << allocation.boostedCells
              << " learned_photons=" << allocation.learnedPhotons
              << " learned_photon_frac=" << learnedPhotonFrac
              << " learned_extra_photons=" << allocation.learnedExtraPhotons
              << " learned_avg_photons_per_cell=" << learnedAvgPhotons
              << " all_avg_photons_per_cell=" << avgPhotons
              << std::endl;

    PrintSourcePhotonDistribution(label, photonDistribution, rank);

    std::cout << label << " adaptive tally memory: max_local_pairs="
              << update.maxLocalSourcePairs
              << " max_received_shard_pairs=" << update.maxReceivedShardPairs
              << " score_delta_cells=" << update.scoreDeltaCells
              << " score_map_cells=" << update.scoreMapCells
              << " max_packed_bytes=" << update.maxPackedBytes
              << std::endl;

    if (observerQualityHasAccumulatedStats) {
        std::cout << label << " observer-equity stats: mode="
                  << (observerQuality.polarizationMode ? "polarization" : "luminosity")
                  << " weak_observers=" << observerQuality.weakObservers
                  << "/" << observerQuality.observerCount
                  << " zero_stat_observers=" << observerQuality.zeroStatObservers
                  << " deficit min/avg/max=" << observerQuality.deficitMin
                  << "/" << observerQuality.deficitAvg
                  << "/" << observerQuality.deficitMax
                  << " neff p05/med/p95=" << observerQuality.neffP05
                  << "/" << observerQuality.neffMedian
                  << "/" << observerQuality.neffP95;
        if (observerQuality.polarizationMode)
            std::cout << " pol_snr p05/med/p95=" << observerQuality.snrP05
                      << "/" << observerQuality.snrMedian
                      << "/" << observerQuality.snrP95;
        std::cout << " next_adaptive_budget_multiplier="
                  << observerQuality.budgetMultiplier
                  << std::endl;

        std::vector<size_t> order(observerQuality.deficitByObserver.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) {
                      return observerQuality.deficitByObserver[a] >
                             observerQuality.deficitByObserver[b];
                  });
        size_t const topWeak = std::min<size_t>(10, order.size());
        if (topWeak > 0) {
            std::cout << label << " weakest observers:" << std::endl;
            for (size_t j = 0; j < topWeak; ++j) {
                size_t const obs = order[j];
                double const neff = (obs < observerQuality.neffByObserver.size())
                    ? observerQuality.neffByObserver[obs] : 0.0;
                double const snr = (obs < observerQuality.snrByObserver.size())
                    ? observerQuality.snrByObserver[obs] : 0.0;
                unsigned long long crossings =
                    (obs < observerQuality.crossingsByObserver.size())
                    ? observerQuality.crossingsByObserver[obs] : 0ULL;
                std::cout << "  observer=" << obs
                          << " deficit=" << observerQuality.deficitByObserver[obs]
                          << " crossing_count=" << crossings
                          << " neff=" << neff;
                if (observerQuality.polarizationMode)
                    std::cout << " pol_snr=" << snr;
                std::cout << std::endl;
            }
        }
    }

    auto const& stats = update.topStats;
    size_t const topN = std::min<size_t>(10, stats.size());
    if (topN > 0) {
        std::cout << label << " top escaping source cells:" << std::endl;
        for (size_t i = 0; i < topN; ++i) {
            auto const& s = stats[i];
            double frac = (update.totalEscapedEnergy > 0.0) ? s.energy / update.totalEscapedEnergy : 0.0;
            auto it = state.scoreByCellID.find(s.cellID);
            double score = (it != state.scoreByCellID.end()) ? it->second : 0.0;
            double const sourceNeff = (s.weightSq > 0.0)
                ? s.energy * s.energy / s.weightSq
                : 0.0;
            double const avgWeight = (s.count > 0)
                ? s.energy / static_cast<double>(s.count)
                : 0.0;
            std::cout << "  observer=" << s.observerIndex
                << " cellID=" << s.cellID
                << " escaped_energy=" << s.energy
                << " escaped_frac=" << frac
                << " crossings=" << s.count
                << " source_neff=" << sourceNeff
                << " avg_weight=" << avgWeight
                << " max_weight=" << s.maxWeight
                << " weightSq=" << s.weightSq
                << " adaptive_score=" << score
                << std::endl;
        }
    }
}

// --- GROUP-AWARE ADAPTIVE FUNCTIONS ---

void CollectGlobalObserverGroupQuality(
    SphericalObserver::ObserverGroupQualitySnapshot& snap)
{
#ifdef RICH_MPI
    size_t const nObs = snap.observerCount;
    size_t const nGrp = snap.groupCount;
    size_t const flat = nObs * nGrp;
    if (flat == 0) return;

    auto flattenD = [&](std::vector<std::vector<double>>& mat) {
        std::vector<double> buf(flat, 0.0);
        for (size_t o = 0; o < nObs; ++o)
            for (size_t g = 0; g < nGrp; ++g)
                buf[o * nGrp + g] = mat[o][g];
        MPI_Allreduce(MPI_IN_PLACE, buf.data(), static_cast<int>(flat), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (size_t o = 0; o < nObs; ++o)
            for (size_t g = 0; g < nGrp; ++g)
                mat[o][g] = buf[o * nGrp + g];
    };

    auto flattenSz = [&](std::vector<std::vector<size_t>>& mat) {
        std::vector<unsigned long long> buf(flat, 0ULL);
        for (size_t o = 0; o < nObs; ++o)
            for (size_t g = 0; g < nGrp; ++g)
                buf[o * nGrp + g] = static_cast<unsigned long long>(mat[o][g]);
        MPI_Allreduce(MPI_IN_PLACE, buf.data(), static_cast<int>(flat), MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        for (size_t o = 0; o < nObs; ++o)
            for (size_t g = 0; g < nGrp; ++g)
                mat[o][g] = static_cast<size_t>(buf[o * nGrp + g]);
    };

    flattenD(snap.energy);
    flattenD(snap.energyWeightSq);
    flattenSz(snap.crossingCount);
    if (snap.polarizationEnabled) {
        flattenD(snap.stokesQ);
        flattenD(snap.stokesU);
        flattenD(snap.sumWQ2);
        flattenD(snap.sumWU2);
    }
#else
    (void)snap;
#endif
}

ObserverGroupQualityDiagnostics BuildObserverGroupQualityDiagnosticsFromSnapshot(
    SphericalObserver::ObserverGroupQualitySnapshot const& snap,
    Config const& cfg,
    AdaptiveGroupHistory& history,
    double sourceDt,
    bool includeInIntegratedStats)
{
    ObserverGroupQualityDiagnostics diag;
    diag.enabled = cfg.adaptiveGroupQuality;
    if (!diag.enabled) return diag;

    size_t const nObs = snap.observerCount;
    size_t const nGrp = snap.groupCount;
    diag.observerCount = nObs;
    diag.groupCount = nGrp;
    diag.polarizationMode = snap.polarizationEnabled;

    auto make2d = [&](double val = 0.0) {
        return std::vector<std::vector<double>>(nObs, std::vector<double>(nGrp, val));
    };

    diag.luminosity = make2d();
    diag.neff = make2d();
    diag.polarizationDegree = make2d();
    diag.polarizationSnr = make2d();
    diag.polarizationSnrValid.assign(nObs, std::vector<int>(nGrp, 0));
    diag.latestPriority = make2d();
    diag.cumulativePriority = make2d();
    diag.predictedPriority = make2d(1.0);
    diag.deficit = make2d(1.0);
    diag.crossings.assign(nObs, std::vector<size_t>(nGrp, 0));

    double const eps = 1e-30;
    double invDt = (sourceDt > 0.0) ? 1.0 / sourceDt : 0.0;

    double maxLumGlobal = 0.0;
    std::vector<double> maxLumPerGroup(nGrp, 0.0);
    for (size_t o = 0; o < nObs; ++o) {
        for (size_t g = 0; g < nGrp; ++g) {
            double L = snap.energy[o][g] * invDt;
            diag.luminosity[o][g] = L;
            maxLumGlobal = std::max(maxLumGlobal, L);
            maxLumPerGroup[g] = std::max(maxLumPerGroup[g], L);
            diag.crossings[o][g] = snap.crossingCount[o][g];

            double E = snap.energy[o][g];
            double W2 = snap.energyWeightSq[o][g];
            diag.neff[o][g] = (W2 > eps) ? (E * E) / W2 : 0.0;
        }
    }

    if (snap.polarizationEnabled) {
        for (size_t o = 0; o < nObs; ++o) {
            for (size_t g = 0; g < nGrp; ++g) {
                auto const quality = polarization_statistics::ComputeQuality(
                    snap.energy[o][g],
                    snap.stokesQ[o][g],
                    snap.stokesU[o][g],
                    snap.energyWeightSq[o][g],
                    snap.sumWQ2[o][g],
                    snap.sumWU2[o][g]);
                diag.polarizationDegree[o][g] = quality.degree;
                diag.polarizationSnr[o][g] = quality.snr;
                diag.polarizationSnrValid[o][g] =
                    quality.uncertaintyValid ? 1 : 0;
            }
        }
    }

    double const gw = cfg.adaptiveGroupLuminosityGlobalWeight;
    for (size_t o = 0; o < nObs; ++o) {
        for (size_t g = 0; g < nGrp; ++g) {
            double L = diag.luminosity[o][g];
            double normGlobal = (maxLumGlobal > 0.0) ? L / maxLumGlobal : 0.0;
            double normGroup = (maxLumPerGroup[g] > 0.0) ? L / maxLumPerGroup[g] : 0.0;
            double lumImportance = 0.0;
            if (cfg.adaptiveGroupLuminosityNormalization == "global")
                lumImportance = normGlobal;
            else if (cfg.adaptiveGroupLuminosityNormalization == "per-group")
                lumImportance = normGroup;
            else
                lumImportance = gw * normGlobal + (1.0 - gw) * normGroup;

            double lumComponent = cfg.adaptiveGroupLuminosityWeight
                * std::pow(lumImportance, cfg.adaptiveGroupLuminosityPower);

            double polComponent = 0.0;
            if (snap.polarizationEnabled) {
                double polNorm = std::max(diag.polarizationDegree[o][g], cfg.adaptiveGroupPolarizationFloor);
                polComponent = cfg.adaptiveGroupPolarizationWeight
                    * std::pow(polNorm, cfg.adaptiveGroupPolarizationPower);
            }

            double science = lumComponent + polComponent;

            double defNeff = (diag.neff[o][g] > eps)
                ? cfg.adaptiveGroupTargetNeff / diag.neff[o][g] : cfg.adaptiveGroupDeficitMax;
            double defPol = 1.0;
            if (snap.polarizationEnabled &&
                diag.polarizationDegree[o][g] >= cfg.adaptiveGroupPolarizationFloor &&
                snap.crossingCount[o][g] >= cfg.adaptiveGroupMinCrossings) {
                if (diag.polarizationSnrValid[o][g] != 0 &&
                    diag.polarizationSnr[o][g] > eps)
                    defPol = cfg.adaptiveGroupTargetPolSnr / diag.polarizationSnr[o][g];
                else
                    defPol = cfg.adaptiveGroupDeficitMax;
            }
            double deficitRaw = std::max(defNeff, defPol);
            deficitRaw = std::clamp(deficitRaw, 1.0, cfg.adaptiveGroupDeficitMax);
            diag.deficit[o][g] = deficitRaw;

            double priority = science * deficitRaw;

            bool const luminosityEligible =
                (cfg.adaptiveGroupMinLuminosity > 0.0 &&
                 L >= cfg.adaptiveGroupMinLuminosity) ||
                (cfg.adaptiveGroupMinLuminosityFracOfGroupMax > 0.0 &&
                 maxLumPerGroup[g] > 0.0 &&
                 L >= cfg.adaptiveGroupMinLuminosityFracOfGroupMax * maxLumPerGroup[g]);
            bool const retainedHistoryPriority =
                history.initialized &&
                o < history.emaPriority.size() &&
                g < history.emaPriority[o].size() &&
                history.emaPriority[o][g] >= cfg.adaptiveGroupRetainPriorityFloor;
            bool eligible = (snap.crossingCount[o][g] >= cfg.adaptiveGroupMinCrossings)
                || luminosityEligible
                || retainedHistoryPriority;
            if (!eligible)
                priority = std::min(priority, cfg.adaptiveGroupIneligiblePriorityCap);

            diag.latestPriority[o][g] = priority;
        }
    }

    if (!history.initialized) {
        history.initialized = true;
        history.observerCount = nObs;
        history.groupCount = nGrp;
        history.updateCount = 0;
        history.integratedUpdateCount = 0;
        history.emaPriority = diag.latestPriority;
        history.emaDeficit = diag.deficit;
        history.cumulativeEnergy = std::vector<std::vector<double>>(nObs, std::vector<double>(nGrp, 0.0));
        history.cumulativeWeightSq = std::vector<std::vector<double>>(nObs, std::vector<double>(nGrp, 0.0));
        history.cumulativeStokesQ = std::vector<std::vector<double>>(nObs, std::vector<double>(nGrp, 0.0));
        history.cumulativeStokesU = std::vector<std::vector<double>>(nObs, std::vector<double>(nGrp, 0.0));
        history.cumulativeSumWQ2 = std::vector<std::vector<double>>(nObs, std::vector<double>(nGrp, 0.0));
        history.cumulativeSumWU2 = std::vector<std::vector<double>>(nObs, std::vector<double>(nGrp, 0.0));
        history.cumulativeCrossings.assign(nObs, std::vector<size_t>(nGrp, 0));
    }

    if (includeInIntegratedStats) {
        for (size_t o = 0; o < nObs; ++o) {
            for (size_t g = 0; g < nGrp; ++g) {
                history.cumulativeEnergy[o][g] += snap.energy[o][g];
                history.cumulativeWeightSq[o][g] += snap.energyWeightSq[o][g];
                if (snap.polarizationEnabled) {
                    history.cumulativeStokesQ[o][g] += snap.stokesQ[o][g];
                    history.cumulativeStokesU[o][g] += snap.stokesU[o][g];
                    history.cumulativeSumWQ2[o][g] += snap.sumWQ2[o][g];
                    history.cumulativeSumWU2[o][g] += snap.sumWU2[o][g];
                }
                history.cumulativeCrossings[o][g] += snap.crossingCount[o][g];
            }
        }
        ++history.integratedUpdateCount;
    }

    double maxCumLumGlobal = 0.0;
    std::vector<double> maxCumLumPerGroup(nGrp, 0.0);
    for (size_t o = 0; o < nObs; ++o) {
        for (size_t g = 0; g < nGrp; ++g) {
            double const cumL = history.cumulativeEnergy[o][g];
            maxCumLumGlobal = std::max(maxCumLumGlobal, cumL);
            maxCumLumPerGroup[g] = std::max(maxCumLumPerGroup[g], cumL);
        }
    }

    for (size_t o = 0; o < nObs; ++o) {
        for (size_t g = 0; g < nGrp; ++g) {
            double cumE = history.cumulativeEnergy[o][g];
            double cumW2 = history.cumulativeWeightSq[o][g];
            double cumNeff = (cumW2 > eps) ? (cumE * cumE) / cumW2 : 0.0;
            if (history.integratedUpdateCount > 0)
                diag.neff[o][g] = cumNeff;
            double cumDefNeff = (cumNeff > eps) ? cfg.adaptiveGroupTargetNeff / cumNeff : cfg.adaptiveGroupDeficitMax;
            double cumDefPol = 1.0;
            double cumPolDegree = 0.0;
            double cumPolSnr = 0.0;
            if (snap.polarizationEnabled && cumE > eps) {
                auto const quality = polarization_statistics::ComputeQuality(
                    cumE,
                    history.cumulativeStokesQ[o][g],
                    history.cumulativeStokesU[o][g],
                    cumW2,
                    history.cumulativeSumWQ2[o][g],
                    history.cumulativeSumWU2[o][g]);
                cumPolDegree = quality.degree;
                cumPolSnr = quality.snr;
                if (history.integratedUpdateCount > 0) {
                    diag.polarizationDegree[o][g] = cumPolDegree;
                    diag.polarizationSnr[o][g] = cumPolSnr;
                    diag.polarizationSnrValid[o][g] =
                        quality.uncertaintyValid ? 1 : 0;
                }
                if (cumPolDegree >= cfg.adaptiveGroupPolarizationFloor &&
                    history.cumulativeCrossings[o][g] >= cfg.adaptiveGroupMinCrossings) {
                    cumDefPol = (quality.uncertaintyValid && cumPolSnr > eps)
                        ? cfg.adaptiveGroupTargetPolSnr / cumPolSnr
                        : cfg.adaptiveGroupDeficitMax;
                }
            }
            double cumDef = std::clamp(std::max(cumDefNeff, cumDefPol), 1.0, cfg.adaptiveGroupDeficitMax);
            if (history.integratedUpdateCount > 0) {
                diag.deficit[o][g] = cumDef;
                diag.crossings[o][g] = history.cumulativeCrossings[o][g];
            }
            double normGlobal = (maxCumLumGlobal > 0.0) ? cumE / maxCumLumGlobal : 0.0;
            double normGroup = (maxCumLumPerGroup[g] > 0.0) ? cumE / maxCumLumPerGroup[g] : 0.0;
            double cumLumImportance = 0.0;
            if (cfg.adaptiveGroupLuminosityNormalization == "global")
                cumLumImportance = normGlobal;
            else if (cfg.adaptiveGroupLuminosityNormalization == "per-group")
                cumLumImportance = normGroup;
            else
                cumLumImportance = gw * normGlobal + (1.0 - gw) * normGroup;
            double const cumLumComponent = cfg.adaptiveGroupLuminosityWeight
                * std::pow(cumLumImportance, cfg.adaptiveGroupLuminosityPower);
            double cumPolComponent = 0.0;
            if (snap.polarizationEnabled) {
                double const polNorm = std::max(cumPolDegree, cfg.adaptiveGroupPolarizationFloor);
                cumPolComponent = cfg.adaptiveGroupPolarizationWeight
                    * std::pow(polNorm, cfg.adaptiveGroupPolarizationPower);
            }
            diag.cumulativePriority[o][g] = (cumLumComponent + cumPolComponent) * cumDef;
        }
    }

    if (cfg.adaptiveGroupHistory && history.updateCount > 0) {
        double alpha = cfg.adaptiveGroupHistoryEma;
        for (size_t o = 0; o < nObs; ++o) {
            for (size_t g = 0; g < nGrp; ++g) {
                double combined = (history.integratedUpdateCount > 0)
                    ? diag.cumulativePriority[o][g]
                    : diag.latestPriority[o][g];
                history.emaPriority[o][g] =
                    (history.integratedUpdateCount > 1)
                    ? (1.0 - alpha) * history.emaPriority[o][g] + alpha * combined
                    : combined;
                history.emaDeficit[o][g] = (1.0 - alpha) * history.emaDeficit[o][g] + alpha * diag.deficit[o][g];
                diag.predictedPriority[o][g] = std::clamp(history.emaPriority[o][g], 1.0, cfg.adaptiveGroupDeficitMax);
            }
        }
    } else {
        for (size_t o = 0; o < nObs; ++o)
            for (size_t g = 0; g < nGrp; ++g)
                diag.predictedPriority[o][g] = (history.integratedUpdateCount > 0)
                    ? std::clamp(diag.cumulativePriority[o][g], 1.0, cfg.adaptiveGroupDeficitMax)
                    : diag.latestPriority[o][g];
    }
    ++history.updateCount;

    std::vector<double> allNeff, allPolSnr;
    for (size_t o = 0; o < nObs; ++o) {
        for (size_t g = 0; g < nGrp; ++g) {
            size_t const statCrossings = (history.integratedUpdateCount > 0)
                ? history.cumulativeCrossings[o][g]
                : snap.crossingCount[o][g];
            if (statCrossings > 0) {
                allNeff.push_back(diag.neff[o][g]);
                if (snap.polarizationEnabled)
                    allPolSnr.push_back(diag.polarizationSnr[o][g]);
                ++diag.activeBins;
                if (diag.predictedPriority[o][g] > 5.0)
                    ++diag.highPriorityBins;
            }
        }
    }
    auto percentile = [](std::vector<double>& v, double p) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        size_t idx = static_cast<size_t>(p * static_cast<double>(v.size() - 1));
        return v[std::min(idx, v.size() - 1)];
    };
    diag.neffP05 = percentile(allNeff, 0.05);
    diag.neffMedian = percentile(allNeff, 0.50);
    diag.neffP95 = percentile(allNeff, 0.95);
    diag.polSnrP05 = percentile(allPolSnr, 0.05);
    diag.polSnrMedian = percentile(allPolSnr, 0.50);
    diag.polSnrP95 = percentile(allPolSnr, 0.95);

    return diag;
}

double PredictedGroupPriority(
    ObserverGroupQualityDiagnostics const& groupQuality,
    size_t observerIndex,
    size_t groupIndex)
{
    if (observerIndex < groupQuality.predictedPriority.size() &&
        groupIndex < groupQuality.predictedPriority[observerIndex].size() &&
        std::isfinite(groupQuality.predictedPriority[observerIndex][groupIndex]) &&
        groupQuality.predictedPriority[observerIndex][groupIndex] > 0.0)
        return groupQuality.predictedPriority[observerIndex][groupIndex];
    return 1.0;
}

double PreMpiGroupStatScore(
    SphericalObserver::SourceCellGroupEscapeStat const& s,
    ObserverGroupQualityDiagnostics const& groupQuality)
{
    double const priority =
        PredictedGroupPriority(groupQuality, s.observerIndex, s.groupIndex);
    double const energy = (s.energy > 0.0 && std::isfinite(s.energy)) ? s.energy : 0.0;
    double const weightSq = (s.weightSq > 0.0 && std::isfinite(s.weightSq))
        ? s.weightSq
        : energy * energy;
    return priority * (energy + std::sqrt(std::max(weightSq, 0.0)));
}

std::vector<SphericalObserver::SourceCellGroupEscapeStat>
PruneLocalSourceGroupStats(
    std::vector<SphericalObserver::SourceCellGroupEscapeStat> const& localStats,
    ObserverGroupQualityDiagnostics const& groupQuality,
    Config const& cfg,
    AdaptiveGroupSourceUpdateSummary& summary)
{
    summary.localStatsInput = static_cast<unsigned long long>(localStats.size());
    std::vector<SphericalObserver::SourceCellGroupEscapeStat> pruned;
    pruned.reserve(std::min(localStats.size(), cfg.adaptiveGroupMaxLocalStats));

    for (auto const& s : localStats) {
        if (s.observerIndex >= groupQuality.observerCount ||
            s.groupIndex >= groupQuality.groupCount ||
            !(s.energy > 0.0) ||
            !std::isfinite(s.energy))
            continue;
        double const priority =
            PredictedGroupPriority(groupQuality, s.observerIndex, s.groupIndex);
        if (s.count >= cfg.adaptiveGroupStatMinCount ||
            priority >= cfg.adaptiveGroupStatPriorityKeep)
            pruned.push_back(s);
    }

    if (pruned.size() > cfg.adaptiveGroupMaxLocalStats) {
        if (cfg.adaptiveGroupFallbackToIntegratedOnOverflow) {
            summary.fallbackToIntegratedPath = true;
            summary.fallbackReason = "local_group_stats_overflow";
            summary.localStatsDropped =
                static_cast<unsigned long long>(localStats.size());
            return {};
        }

        std::sort(pruned.begin(), pruned.end(),
            [&](auto const& a, auto const& b) {
                return PreMpiGroupStatScore(a, groupQuality) >
                       PreMpiGroupStatScore(b, groupQuality);
            });
        pruned.resize(cfg.adaptiveGroupMaxLocalStats);
    }

    summary.localStatsAfterPrune = static_cast<unsigned long long>(pruned.size());
    summary.localStatsDropped =
        summary.localStatsInput > summary.localStatsAfterPrune
            ? summary.localStatsInput - summary.localStatsAfterPrune
            : 0ULL;
    return pruned;
}

void DecayAndPruneGroupSourceState(
    AdaptiveGroupSourceState& groupState,
    double decay,
    double pruneThreshold)
{
    for (auto it = groupState.scoreByCellGroup.begin();
         it != groupState.scoreByCellGroup.end(); ) {
        double sum = 0.0;
        double maxAbs = 0.0;
        for (auto& v : it->second) {
            v *= decay;
            if (!std::isfinite(v) || v < 0.0)
                v = 0.0;
            sum += v;
            maxAbs = std::max(maxAbs, std::abs(v));
        }
        if (maxAbs < pruneThreshold) {
            groupState.cellScoreFromGroups.erase(it->first);
            it = groupState.scoreByCellGroup.erase(it);
        } else {
            groupState.cellScoreFromGroups[it->first] = sum;
            ++it;
        }
    }

    for (auto it = groupState.cellScoreFromGroups.begin();
         it != groupState.cellScoreFromGroups.end(); ) {
        if (!(it->second > pruneThreshold) || !std::isfinite(it->second))
            it = groupState.cellScoreFromGroups.erase(it);
        else
            ++it;
    }
}

AdaptiveGroupSourceUpdateSummary UpdateAdaptiveSourceGroupScores(
    std::vector<SphericalObserver::SourceCellGroupEscapeStat> const& localGroupStats,
    ObserverGroupQualityDiagnostics const& groupQuality,
    Config const& cfg,
    AdaptiveGroupSourceState& groupState,
    int rank,
    [[maybe_unused]] int mpiSize)
{
    AdaptiveGroupSourceUpdateSummary summary;
    if (!cfg.adaptiveGroupSourceCells || !groupQuality.enabled)
        return summary;

    size_t const nObs = groupQuality.observerCount;
    size_t const nGrp = groupQuality.groupCount;

    std::vector<std::vector<double>> totalEnergyByOG(nObs, std::vector<double>(nGrp, 0.0));
    std::vector<std::vector<double>> totalW2ByOG(nObs, std::vector<double>(nGrp, 0.0));

    for (auto const& s : localGroupStats) {
        if (s.observerIndex < nObs && s.groupIndex < nGrp) {
            totalEnergyByOG[s.observerIndex][s.groupIndex] += s.energy;
            totalW2ByOG[s.observerIndex][s.groupIndex] +=
                (s.weightSq > 0.0 && std::isfinite(s.weightSq))
                    ? s.weightSq
                    : s.energy * s.energy;
        }
    }

#ifdef RICH_MPI
    size_t flat = nObs * nGrp;
    std::vector<double> flatE(flat, 0.0), flatW2(flat, 0.0);
    for (size_t o = 0; o < nObs; ++o)
        for (size_t g = 0; g < nGrp; ++g) {
            flatE[o * nGrp + g] = totalEnergyByOG[o][g];
            flatW2[o * nGrp + g] = totalW2ByOG[o][g];
        }
    MPI_Allreduce(MPI_IN_PLACE, flatE.data(), static_cast<int>(flat), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, flatW2.data(), static_cast<int>(flat), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (size_t o = 0; o < nObs; ++o)
        for (size_t g = 0; g < nGrp; ++g) {
            totalEnergyByOG[o][g] = flatE[o * nGrp + g];
            totalW2ByOG[o][g] = flatW2[o * nGrp + g];
        }
#endif

    double const eps = 1e-30;
    bool const polMode = groupQuality.polarizationMode;
    double const varianceMix = polMode ? 0.85 : 0.50;
    double const ema = cfg.adaptiveGroupScoreEma;
    double const decay = 1.0 - ema;

    auto prunedLocalStats = PruneLocalSourceGroupStats(
        localGroupStats, groupQuality, cfg, summary);
#ifdef RICH_MPI
    int fallbackLocal = summary.fallbackToIntegratedPath ? 1 : 0;
    int fallbackGlobal = fallbackLocal;
    MPI_Allreduce(&fallbackLocal, &fallbackGlobal, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (fallbackGlobal != 0) {
        summary.fallbackToIntegratedPath = true;
        summary.fallbackReason = "local_group_stats_overflow";
    }
#endif
    if (summary.fallbackToIntegratedPath) {
        groupState.scoreByCellGroup.clear();
        groupState.cellScoreFromGroups.clear();
        return summary;
    }

    DecayAndPruneGroupSourceState(groupState, decay, 1e-20);

    std::vector<PackedSourceGroupEscapeStat> received =
        ExchangeSourceGroupStatsByCellOwner(prunedLocalStats, summary);
    std::vector<SphericalObserver::SourceCellGroupEscapeStat>().swap(prunedLocalStats);

    std::unordered_map<AdaptiveSourceGroupKey,
                       SphericalObserver::SourceCellGroupEscapeStat,
                       AdaptiveSourceGroupKeyHash> byGroupCell;
    byGroupCell.reserve(received.size());
    for (auto const& p : received) {
        if (p.count == 0 || !(p.energy > 0.0) || !std::isfinite(p.energy))
            continue;
        size_t const observerIndex = static_cast<size_t>(p.observerIndex);
        size_t const groupIndex = static_cast<size_t>(p.groupIndex);
        size_t const cellID = static_cast<size_t>(p.cellID);
        AdaptiveSourceGroupKey const key{observerIndex, groupIndex, cellID};
        auto& s = byGroupCell[key];
        s.cellID = cellID;
        s.observerIndex = observerIndex;
        s.groupIndex = groupIndex;
        s.energy += p.energy;
        s.weightSq += p.weightSq;
        s.maxWeight = std::max(s.maxWeight, p.maxWeight);
        s.count += static_cast<size_t>(p.count);
    }
    std::vector<PackedSourceGroupEscapeStat>().swap(received);

    std::unordered_map<size_t, std::vector<double>> deltaByCellGroup;
    for (auto const& kv : byGroupCell) {
        auto const& s = kv.second;
        if (s.observerIndex >= nObs || s.groupIndex >= nGrp)
            continue;
        if (!(s.energy > 0.0) || s.count < cfg.adaptiveGroupStatMinCount)
            continue;

        double const totE = totalEnergyByOG[s.observerIndex][s.groupIndex];
        double const totW2 = totalW2ByOG[s.observerIndex][s.groupIndex];
        double const eFrac = (totE > eps) ? s.energy / totE : 0.0;
        double const statW2 = (s.weightSq > 0.0 && std::isfinite(s.weightSq))
            ? s.weightSq
            : s.energy * s.energy;
        double const w2Frac = (totW2 > eps) ? statW2 / totW2 : eFrac;
        double const sourceQuality = (1.0 - varianceMix) * eFrac + varianceMix * w2Frac;

        double const priority =
            PredictedGroupPriority(groupQuality, s.observerIndex, s.groupIndex);

        double delta = ema * priority * sourceQuality;
        if (!(delta > 0.0) || !std::isfinite(delta))
            continue;

        auto& gvec = deltaByCellGroup[s.cellID];
        if (gvec.empty()) gvec.assign(nGrp, 0.0);
        gvec[s.groupIndex] += delta;
        ++summary.passedStats;
    }
    std::unordered_map<AdaptiveSourceGroupKey,
                       SphericalObserver::SourceCellGroupEscapeStat,
                       AdaptiveSourceGroupKeyHash>().swap(byGroupCell);

    std::vector<PackedAdaptiveCellGroupScoreDelta> localDeltas;
    for (auto const& kv : deltaByCellGroup) {
        size_t const cellID = kv.first;
        for (size_t g = 0; g < kv.second.size(); ++g) {
            double const delta = kv.second[g];
            if (!(delta > 0.0) || !std::isfinite(delta))
                continue;
            PackedAdaptiveCellGroupScoreDelta p;
            p.cellID = static_cast<unsigned long long>(cellID);
            p.groupIndex = static_cast<unsigned long long>(g);
            p.delta = delta;
            localDeltas.push_back(p);
        }
    }
    std::unordered_map<size_t, std::vector<double>>().swap(deltaByCellGroup);

    size_t const localDeltaCells = localDeltas.size();
    std::vector<PackedAdaptiveCellGroupScoreDelta> allDeltas =
        AllgatherAdaptiveCellGroupScoreDeltas(localDeltas);
    std::vector<PackedAdaptiveCellGroupScoreDelta>().swap(localDeltas);

    for (auto const& p : allDeltas) {
        if (!(p.delta > 0.0) || !std::isfinite(p.delta))
            continue;
        size_t const cellID = static_cast<size_t>(p.cellID);
        size_t const groupIndex = static_cast<size_t>(p.groupIndex);
        if (groupIndex >= nGrp)
            continue;
        auto& gvec = groupState.scoreByCellGroup[cellID];
        if (gvec.empty()) gvec.assign(nGrp, 0.0);
        gvec[groupIndex] += p.delta;
        groupState.cellScoreFromGroups[cellID] += p.delta;
    }
    summary.scoreDeltaCells = allDeltas.size();
    std::vector<PackedAdaptiveCellGroupScoreDelta>().swap(allDeltas);

    DecayAndPruneGroupSourceState(groupState, 1.0, 1e-20);
    summary.scoreMapCells = groupState.scoreByCellGroup.size();

    unsigned long long maxLocalDeltaCells =
        static_cast<unsigned long long>(localDeltaCells);
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &maxLocalDeltaCells, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
#endif
    summary.maxPackedBytes = std::max(
        summary.maxPackedBytes,
        (maxLocalDeltaCells + static_cast<unsigned long long>(summary.scoreDeltaCells)) *
            static_cast<unsigned long long>(sizeof(PackedAdaptiveCellGroupScoreDelta)));

    if (rank == 0 && cfg.adaptiveDiagnosticsVerbose) {
        std::cout << "GROUP_ADAPT source_group_state: cells_with_group_scores="
                  << groupState.scoreByCellGroup.size()
                  << " cells_with_cell_score=" << groupState.cellScoreFromGroups.size()
                  << " dropped_stats=" << summary.localStatsDropped
                  << " mpi_bytes_max=" << summary.maxPackedBytes
                  << std::endl;
    }
    return summary;
}

void PrintAdaptiveGroupGenerationStats(
    ObserverGroupQualityDiagnostics const& gq,
    RadiationIMC::GroupSamplingDiagnostics const& gsd,
    size_t gen,
    int rank)
{
    if (rank != 0 || !gq.enabled) return;
    std::cout << "GROUP_ADAPT gen=" << (gen + 1)
              << " groups=" << gq.groupCount
              << " active_bins=" << gq.activeBins
              << " high_priority_bins=" << gq.highPriorityBins
              << " neff_p05/med/p95=" << gq.neffP05 << "/" << gq.neffMedian << "/" << gq.neffP95;
    if (gq.polarizationMode)
        std::cout << " pol_snr_p05/med/p95=" << gq.polSnrP05 << "/" << gq.polSnrMedian << "/" << gq.polSnrP95;
    if (gsd.totalSampled > 0 || gsd.weightCorrectionFallback > 0 || gsd.invalidPdfFallback > 0) {
        double avgCorr = (gsd.weightCorrectionCount > 0) ? gsd.weightCorrectionSum / gsd.weightCorrectionCount : 1.0;
        double cappedFrac = (gsd.weightCorrectionCount > 0)
            ? static_cast<double>(gsd.weightCorrectionCapped) / static_cast<double>(gsd.weightCorrectionCount)
            : 0.0;
        std::cout << " weight_corr_min/avg/max=" << gsd.weightCorrectionMin << "/" << avgCorr << "/" << gsd.weightCorrectionMax
                  << " capped_frac=" << cappedFrac
                  << " capped_energy_frac=" << gsd.cappedEnergyFraction
                  << " fallback=" << gsd.weightCorrectionFallback
                  << " invalid_pdf_cells=" << gsd.invalidPdfFallback
                  << " biased=" << (gsd.estimatorPotentiallyBiased ? 1 : 0);
    }
    std::cout << std::endl;

    struct GroupNeedRow
    {
        size_t observer = 0;
        size_t group = 0;
        double predictedPriority = 0.0;
        double latestPriority = 0.0;
        double deficit = 0.0;
        double luminosity = 0.0;
        double neff = 0.0;
        double polSnr = 0.0;
        size_t crossings = 0;
    };

    std::vector<GroupNeedRow> rows;
    rows.reserve(gq.observerCount * gq.groupCount);
    for (size_t o = 0; o < gq.observerCount; ++o) {
        for (size_t g = 0; g < gq.groupCount; ++g) {
            double predicted = 0.0;
            if (o < gq.predictedPriority.size() &&
                g < gq.predictedPriority[o].size())
                predicted = gq.predictedPriority[o][g];
            double latest = 0.0;
            if (o < gq.latestPriority.size() &&
                g < gq.latestPriority[o].size())
                latest = gq.latestPriority[o][g];
            double deficit = 0.0;
            if (o < gq.deficit.size() && g < gq.deficit[o].size())
                deficit = gq.deficit[o][g];
            if (!(predicted > 0.0) && !(latest > 0.0) && !(deficit > 1.0))
                continue;
            GroupNeedRow row;
            row.observer = o;
            row.group = g;
            row.predictedPriority = predicted;
            row.latestPriority = latest;
            row.deficit = deficit;
            if (o < gq.luminosity.size() && g < gq.luminosity[o].size())
                row.luminosity = gq.luminosity[o][g];
            if (o < gq.neff.size() && g < gq.neff[o].size())
                row.neff = gq.neff[o][g];
            if (o < gq.polarizationSnr.size() &&
                g < gq.polarizationSnr[o].size())
                row.polSnr = gq.polarizationSnr[o][g];
            if (o < gq.crossings.size() && g < gq.crossings[o].size())
                row.crossings = gq.crossings[o][g];
            rows.push_back(row);
        }
    }

    std::sort(rows.begin(), rows.end(),
              [](GroupNeedRow const& a, GroupNeedRow const& b) {
                  if (a.predictedPriority != b.predictedPriority)
                      return a.predictedPriority > b.predictedPriority;
                  return a.deficit > b.deficit;
              });
    size_t const topBins = std::min<size_t>(10, rows.size());
    if (topBins > 0) {
        std::cout << "GROUP_ADAPT top observer/group needs:" << std::endl;
        for (size_t i = 0; i < topBins; ++i) {
            GroupNeedRow const& row = rows[i];
            std::cout << "  observer=" << row.observer
                      << " group=" << row.group
                      << " predicted_priority=" << row.predictedPriority
                      << " latest_priority=" << row.latestPriority
                      << " deficit=" << row.deficit
                      << " luminosity=" << row.luminosity
                      << " neff=" << row.neff
                      << " crossings=" << row.crossings;
            if (gq.polarizationMode)
                std::cout << " pol_snr=" << row.polSnr;
            std::cout << std::endl;
        }
    }
}

std::unordered_map<size_t, std::array<double, ENERGY_GROUPS_NUM>>
BuildGroupScoresForIMC(
    AdaptiveGroupSourceState const& groupState,
    std::vector<ComputationalCell3D> const& localCells,
    size_t nGroups)
{
    std::unordered_set<size_t> localCellIDs;
    localCellIDs.reserve(localCells.size());
    for (auto const& cell : localCells)
        localCellIDs.insert(cell.ID);

    std::unordered_map<size_t, std::array<double, ENERGY_GROUPS_NUM>> result;
    for (auto const& kv : groupState.scoreByCellGroup) {
        if (localCellIDs.find(kv.first) == localCellIDs.end())
            continue;
        std::array<double, ENERGY_GROUPS_NUM> arr{};
        size_t const copyLen = std::min(kv.second.size(), static_cast<size_t>(ENERGY_GROUPS_NUM));
        for (size_t g = 0; g < copyLen; ++g)
            arr[g] = kv.second[g];
        (void)nGroups;
        result[kv.first] = arr;
    }
    return result;
}

std::unordered_map<size_t, double>
BuildCombinedSourceScoresForIMC(
    AdaptiveSourceState const& integratedState,
    AdaptiveGroupSourceState const& groupState)
{
    std::unordered_map<size_t, double> combined = integratedState.scoreByCellID;
    for (auto const& kv : groupState.cellScoreFromGroups) {
        if (kv.second > 0.0 && std::isfinite(kv.second))
            combined[kv.first] += kv.second;
    }
    return combined;
}

// Rosseland weight fraction for a single group with dimensionless boundaries [a, b].
// Uses: integral_a^b x^4 e^x/(e^x-1)^2 dx = a^4/(e^a-1) - b^4/(e^b-1) + 4*(pi^4/15)*planck_integral(a,b)
// Normalized by the full-spectrum integral 4*pi^4/15.
double RosselandWeightFraction(double a, double b)
{
    double const fullIntegral = 4.0 * std::pow(M_PI, 4) / 15.0;
    double boundaryTerm = 0.0;
    if (a > 0.0 && a < 500.0)
        boundaryTerm += std::pow(a, 4) / std::expm1(a);
    if (b > 0.0 && b < 500.0)
        boundaryTerm -= std::pow(b, 4) / std::expm1(b);
    double planckTerm = 4.0 * (std::pow(M_PI, 4) / 15.0) * planck_integral::planck_integral(a, b);
    return (boundaryTerm + planckTerm) / fullIntegral;
}

// Solve for alpha such that:
//   sum_g [ f_g / (alpha * sigmaA[g] + sigmaS[g]) ] = 1/kappaRGrey
// F(alpha) is monotonically decreasing, so bisection converges.
double SolveRosselandAlpha(
    std::vector<double> const& sigmaA,
    std::vector<double> const& sigmaS,
    std::vector<double> const& fRoss,
    double kappaRGrey)
{
    double const target = 1.0 / kappaRGrey;

    auto evalF = [&](double alpha) {
        double sum = 0.0;
        for (size_t g = 0; g < sigmaA.size(); ++g) {
            double denom = alpha * sigmaA[g] + sigmaS[g];
            if (denom > 0.0)
                sum += fRoss[g] / denom;
        }
        return sum - target;
    };

    // F is decreasing: F(0) = sum(f_g/sigmaS_g) - target (large positive if scattering << grey)
    //                   F(inf) → -target (negative)
    // Find bracket
    double lo = 0.0, hi = 1.0;
    while (evalF(hi) > 0.0 && hi < 1e10)
        hi *= 2.0;

    if (evalF(lo) <= 0.0)
        return lo;
    if (evalF(hi) >= 0.0)
        return hi;

    for (int iter = 0; iter < 100; ++iter) {
        double mid = 0.5 * (lo + hi);
        if (evalF(mid) > 0.0)
            lo = mid;
        else
            hi = mid;
        if ((hi - lo) < 1e-12 * (lo + hi + 1e-300))
            break;
    }
    return 0.5 * (lo + hi);
}

// Planck weight fraction for a single group with dimensionless boundaries [a, b].
// Returns the fraction of the Planck spectrum integral_a^b x^3/(e^x-1) dx
// normalised by the full-spectrum integral pi^4/15.
double PlanckWeightFraction(double a, double b)
{
    return planck_integral::planck_integral(a, b);
}

// Solve for alpha such that the Planck-weighted MG absorption matches the grey Planck:
//   sum_g [ f_planck_g * alpha * sigmaA[g] ] = kappaPGrey
// This is a simple ratio (no iteration needed).
double SolvePlanckAlpha(
    std::vector<double> const& sigmaA,
    std::vector<double> const& fPlanck,
    double kappaPGrey)
{
    double mgPlanck = 0.0;
    for (size_t g = 0; g < sigmaA.size(); ++g)
        mgPlanck += fPlanck[g] * sigmaA[g];
    if (mgPlanck <= 0.0)
        return 1.0;
    return kappaPGrey / mgPlanck;
}

void RecomputeOpacityScaleFactors(
    STAMGopacityMC& opacity,
    STAgreyOpacity const& greyOpacity,
    std::vector<ComputationalCell3D> const& cells,
    size_t const Ncells,
    int const rank,
    OpacityScaleMode mode,
    std::string const& label)
{
  // The scale-factor map is rank-local and keyed by cell.ID.  Clear it first so
  // CalcAbsorptionOpacity below samples the unscaled MG opacity, then rebuild it
  // for the cells currently owned by this rank.
  opacity.SetRosselandScaleFactors(std::unordered_map<size_t, double>());

  size_t const Ng = opacity.energy_groups_boundary.size() - 1;
  std::unordered_map<size_t, double> scaleFactors;
  scaleFactors.reserve(Ncells);

  double alphaMin = std::numeric_limits<double>::max();
  double alphaMax = 0.0;
  double alphaSum = 0.0;
  size_t alphaCount = 0;
  size_t alphaOutliers = 0;

  bool const usePlanck = (mode == OpacityScaleMode::Planck);

  for (size_t i = 0; i < Ncells; ++i) {
    double const kT = CG::boltzmann_constant * cells[i].temperature;
    if (kT <= 0.0 || !std::isfinite(kT)) continue;

    std::vector<double> fWeight(Ng);
    double fTotal = 0.0;
    for (size_t g = 0; g < Ng; ++g) {
      double a = opacity.energy_groups_boundary[g] / kT;
      double b = opacity.energy_groups_boundary[g + 1] / kT;
      if (a >= b || a > 500.0) {
        fWeight[g] = 0.0;
        continue;
      }
      b = std::min(b, 500.0);
      fWeight[g] = usePlanck ? PlanckWeightFraction(a, b)
                             : RosselandWeightFraction(a, b);
      fTotal += fWeight[g];
    }
    if (fTotal <= 0.0) continue;
    for (double& f : fWeight) f /= fTotal;

    std::vector<double> sigA(Ng);
    for (size_t g = 0; g < Ng; ++g)
      sigA[g] = opacity.CalcAbsorptionOpacity(cells[i], opacity.energy_groups_center[g]);

    double alpha;
    if (usePlanck) {
      double const kappaPGrey = greyOpacity.CalcPlanckOpacity(cells[i]);
      if (kappaPGrey <= 0.0 || !std::isfinite(kappaPGrey)) continue;
      alpha = 30 * SolvePlanckAlpha(sigA, fWeight, kappaPGrey); // 2 is ad hoc factor to match the gray luminosity
    } else {
      std::vector<double> sigS(Ng);
      for (size_t g = 0; g < Ng; ++g)
        sigS[g] = opacity.CalcScatteringOpacity(cells[i], opacity.energy_groups_center[g]);
      double const D_grey = greyOpacity.CalcDiffusionCoefficient(cells[i]);
      if (D_grey <= 0.0 || !std::isfinite(D_grey)) continue;
      double const kappaRGrey = CG::speed_of_light / (3.0 * D_grey);
      alpha = SolveRosselandAlpha(sigA, sigS, fWeight, kappaRGrey);
    }

    scaleFactors[cells[i].ID] = alpha;

    alphaMin = std::min(alphaMin, alpha);
    alphaMax = std::max(alphaMax, alpha);
    alphaSum += alpha;
    ++alphaCount;
    if (alpha < 0.5 || alpha > 2.0) ++alphaOutliers;
  }

  opacity.SetRosselandScaleFactors(std::move(scaleFactors));

  double globalMin = 0.0, globalMax = 0.0, globalSum = 0.0;
  size_t globalCount = 0, globalOutliers = 0;
#ifdef RICH_MPI
  MPI_Reduce(&alphaMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&alphaMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&alphaSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&alphaCount, &globalCount, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&alphaOutliers, &globalOutliers, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
#else
  globalMin = alphaMin;
  globalMax = alphaMax;
  globalSum = alphaSum;
  globalCount = alphaCount;
  globalOutliers = alphaOutliers;
#endif

  if (rank == 0) {
    char const* modeStr = usePlanck ? "Planck" : "Rosseland";
    double const alphaMean = (globalCount > 0) ? globalSum / static_cast<double>(globalCount) : 1.0;
    std::cout << modeStr << " scale " << label << ": alpha min=" << globalMin
              << " max=" << globalMax << " mean=" << alphaMean
              << " outliers(>2x)=" << globalOutliers << "/" << globalCount
              << std::endl;
  }
}

void PrintPolarizationSummary(
    std::string const& label,
    SphericalObserver::ObserverQualitySnapshot const& snap,
    int rank)
{
    if (rank != 0)
        return;

    if (!snap.polarizationEnabled) {
        std::cout << label
                  << " polarization diagnostics unavailable: observer polarization tracking is disabled"
                  << std::endl;
        return;
    }

    size_t const observerCount = snap.energy.size();
    std::vector<double> degree;
    std::vector<double> snr;
    degree.reserve(observerCount);
    snr.reserve(observerCount);

    size_t activeObservers = 0;
    for (size_t i = 0; i < observerCount; ++i) {
        if (i >= snap.stokesQ.size() || i >= snap.stokesU.size() ||
            i >= snap.polarizationWeightSq.size() ||
            i >= snap.sumWQ2.size() || i >= snap.sumWU2.size())
            continue;

        auto const quality = polarization_statistics::ComputeQuality(
            snap.energy[i], snap.stokesQ[i], snap.stokesU[i],
            snap.polarizationWeightSq[i], snap.sumWQ2[i], snap.sumWU2[i]);
        if (!quality.intensityValid)
            continue;

        ++activeObservers;
        degree.push_back(quality.degree);
        if (quality.uncertaintyValid)
            snr.push_back(quality.snr);
    }

    std::cout << label << " polarization summary: active_observers="
              << activeObservers << "/" << observerCount
              << " valid_snr_observers=" << snr.size()
              << " degree_p05/med/p95=" << Percentile(degree, 0.05)
              << "/" << Percentile(degree, 0.50)
              << "/" << Percentile(degree, 0.95);
    if (snr.empty()) {
        std::cout << " snr_p05/med/p95=n/a";
    } else {
        std::cout << " snr_p05/med/p95=" << Percentile(snr, 0.05)
                  << "/" << Percentile(snr, 0.50)
                  << "/" << Percentile(snr, 0.95);
    }
    std::cout << std::endl;
}

bool MeasuredLBDebugMemory()
{
    static int cached = -1;
    if (cached < 0) {
        char const* val = std::getenv("RICH_MEASURED_LB_DEBUG_MEMORY");
        cached = (val != nullptr && std::string(val) != "0" && std::string(val) != "false") ? 1 : 0;
    }
    return cached != 0;
}

void PrintVmRSS(std::string const& label, int rank) {
    if (!MeasuredLBDebugMemory())
        return;
#ifdef __linux__
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::cerr << "MEMORY_RSS rank=" << rank
                      << " label=" << label
                      << " " << line << "\n";
            break;
        }
    }
#endif
}



} // namespace imc_postprocess_tde
