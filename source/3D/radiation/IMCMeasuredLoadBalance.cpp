#include "IMCMeasuredLoadBalance.hpp"
#include "misc/universal_error.hpp"

namespace imc_measured_lb {

std::vector<LocalCellMeasurement> BuildLocalMeasurements(
    std::vector<size_t> const& cellIDs,
    std::vector<size_t> const& steps,
    std::vector<size_t> const& particlesPerCell)
{
    assert(steps.size() == cellIDs.size());

    std::vector<LocalCellMeasurement> out;
    out.reserve(cellIDs.size());

    for (size_t i = 0; i < cellIDs.size(); ++i) {
        LocalCellMeasurement m;
        m.globalCellID = cellIDs[i];
        m.localCellIndex = i;
        m.stepCount = steps[i];
        m.particleCount = (i < particlesPerCell.size()) ? particlesPerCell[i] : 0;
        m.predictiveCost = 0.0;
        out.push_back(m);
    }

    return out;
}

namespace {

std::unordered_map<size_t, double> BuildRawCosts(
    std::vector<LocalCellMeasurement> const& measurements,
    Parameters const& params,
    bool multigroup)
{
    std::unordered_map<size_t, double> costByID;
    costByID.reserve(measurements.size());

    for (auto const& m : measurements) {
        double c = params.floorCost
                 + params.stepWeight * static_cast<double>(m.stepCount)
                 + params.particleWeight * static_cast<double>(m.particleCount)
                 + params.predictiveWeight * m.predictiveCost;

        if (m.stepCount == 0) {
            double zeroInflation = multigroup ? params.multigroupZeroStepInflation
                                              : params.grayZeroStepInflation;
            c = std::max(c, zeroInflation * params.floorCost);
        }

        costByID[m.globalCellID] += c;
    }

    return costByID;
}

void ClampCosts(std::unordered_map<size_t, double>& costByID,
                double floorCost, double maxCost)
{
    for (auto& kv : costByID)
        kv.second = std::min(std::max(kv.second, floorCost), maxCost);
}

} // anonymous namespace

std::unordered_map<size_t, double> BuildMeasuredCosts(
    std::vector<LocalCellMeasurement> const& measurements,
    Parameters const& params,
    bool multigroup)
{
    auto costByID = BuildRawCosts(measurements, params, multigroup);

    if (params.useMedianClamp && !costByID.empty()) {
        double localSum = 0.0;
        for (auto const& kv : costByID)
            localSum += kv.second;
        double localMean = localSum / static_cast<double>(costByID.size());
        double maxCost = params.medianClampFactor * std::max(localMean, params.floorCost);
        ClampCosts(costByID, params.floorCost, maxCost);
    }

    return costByID;
}

#ifdef RICH_MPI
std::unordered_map<size_t, double> BuildMeasuredCosts(
    std::vector<LocalCellMeasurement> const& measurements,
    Parameters const& params,
    bool multigroup,
    MPI_Comm comm)
{
    auto costByID = BuildRawCosts(measurements, params, multigroup);

    if (params.useMedianClamp) {
        double localSum = 0.0;
        uint64_t localCount = 0;
        for (auto const& kv : costByID) {
            localSum += kv.second;
            ++localCount;
        }

        double globalSum = 0.0;
        uint64_t globalCount = 0;
        MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&localCount, &globalCount, 1, MPI_UINT64_T, MPI_SUM, comm);

        double globalMean = (globalCount > 0)
            ? globalSum / static_cast<double>(globalCount)
            : params.floorCost;

        double maxCost = params.medianClampFactor * std::max(globalMean, params.floorCost);
        double minCost = std::max(globalMean / params.medianClampFactor, params.floorCost);

        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) {
            std::cerr << "MEASURED_LB_CLAMP"
                      << " global_mean_cost=" << globalMean
                      << " clamp_factor=" << params.medianClampFactor
                      << " clamp_ceiling=" << maxCost
                      << " clamp_floor=" << minCost
                      << " effective_ratio=" << (minCost > 0 ? maxCost / minCost : 0)
                      << " global_cells=" << globalCount
                      << "\n";
        }

        ClampCosts(costByID, minCost, maxCost);
    }

    return costByID;
}

std::vector<LocalCellMeasurement> GatherMeasurementsAllRanksDebugOnly(
    std::vector<LocalCellMeasurement> const& localMeasurements,
    MPI_Comm comm,
    uint64_t maxAllowedGlobalCells)
{
    uint64_t localCount64 = static_cast<uint64_t>(localMeasurements.size());
    uint64_t globalCount64 = 0;
    MPI_Allreduce(&localCount64, &globalCount64, 1, MPI_UINT64_T, MPI_SUM, comm);

    if (globalCount64 > maxAllowedGlobalCells) {
        throw UniversalError("Refusing debug all-rank measurement gather for production-size mesh");
    }

    if (localMeasurements.size() > static_cast<size_t>(std::numeric_limits<int>::max() / 3)) {
        throw UniversalError("Too many local IMC measurements for MPI_Allgatherv int counts");
    }

    int localCount = static_cast<int>(localMeasurements.size());
    int mpiSize = 0;
    MPI_Comm_size(comm, &mpiSize);

    std::vector<int> allCounts(mpiSize);
    MPI_Allgather(&localCount, 1, MPI_INT,
                  allCounts.data(), 1, MPI_INT, comm);

    int totalCount = 0;
    std::vector<int> displacements(mpiSize, 0);
    for (int r = 0; r < mpiSize; ++r) {
        displacements[r] = totalCount;
        totalCount += allCounts[r];
    }

    std::vector<uint64_t> localPacked(localMeasurements.size() * 3);
    for (size_t i = 0; i < localMeasurements.size(); ++i) {
        localPacked[i * 3 + 0] = static_cast<uint64_t>(localMeasurements[i].globalCellID);
        localPacked[i * 3 + 1] = static_cast<uint64_t>(localMeasurements[i].stepCount);
        localPacked[i * 3 + 2] = static_cast<uint64_t>(localMeasurements[i].particleCount);
    }

    std::vector<int> sendCounts(mpiSize), recvDisp(mpiSize);
    for (int r = 0; r < mpiSize; ++r) {
        sendCounts[r] = allCounts[r] * 3;
        recvDisp[r] = displacements[r] * 3;
    }

    int localSendCount = localCount * 3;
    std::vector<uint64_t> globalPacked(static_cast<size_t>(totalCount) * 3);

    MPI_Allgatherv(localPacked.data(), localSendCount, MPI_UINT64_T,
                   globalPacked.data(), sendCounts.data(), recvDisp.data(),
                   MPI_UINT64_T, comm);

    std::vector<LocalCellMeasurement> result(static_cast<size_t>(totalCount));
    for (int i = 0; i < totalCount; ++i) {
        result[i].globalCellID = static_cast<size_t>(globalPacked[i * 3 + 0]);
        result[i].localCellIndex = 0;
        result[i].stepCount = static_cast<size_t>(globalPacked[i * 3 + 1]);
        result[i].particleCount = static_cast<size_t>(globalPacked[i * 3 + 2]);
        result[i].predictiveCost = 0.0;
    }

    return result;
}

void PrintMeasuredLBDiagnosticsDistributed(
    std::vector<LocalCellMeasurement> const& localMeasurements,
    std::unordered_map<size_t, double> const& localCostByCellID,
    bool multigroup,
    MPI_Comm comm)
{
    int rank = 0, mpiSize = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpiSize);

    uint64_t localCellCount = static_cast<uint64_t>(localMeasurements.size());
    uint64_t localStepSum = 0;
    uint64_t localStepMax = 0;
    uint64_t localZeroStepCells = 0;

    for (auto const& m : localMeasurements) {
        uint64_t s = static_cast<uint64_t>(m.stepCount);
        localStepSum += s;
        if (s > localStepMax)
            localStepMax = s;
        if (m.stepCount == 0)
            ++localZeroStepCells;
    }

    std::cerr << "MEASURED_LB_BEFORE rank=" << rank
              << " local_cells=" << localCellCount
              << " local_step_sum=" << localStepSum
              << " local_step_max=" << localStepMax
              << " local_zero_step_cells=" << localZeroStepCells
              << "\n";

    double localStepSumD = static_cast<double>(localStepSum);
    double maxRankSteps = 0.0, sumRankSteps = 0.0;
    MPI_Allreduce(&localStepSumD, &maxRankSteps, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&localStepSumD, &sumRankSteps, 1, MPI_DOUBLE, MPI_SUM, comm);
    double meanRankSteps = sumRankSteps / std::max(mpiSize, 1);

    uint64_t globalCellCount = 0;
    uint64_t globalZeroStepCells = 0;
    uint64_t globalStepMax = 0;
    MPI_Allreduce(&localCellCount, &globalCellCount, 1, MPI_UINT64_T, MPI_SUM, comm);
    MPI_Allreduce(&localZeroStepCells, &globalZeroStepCells, 1, MPI_UINT64_T, MPI_SUM, comm);
    MPI_Allreduce(&localStepMax, &globalStepMax, 1, MPI_UINT64_T, MPI_MAX, comm);

    double localCostSum = 0.0;
    double localCostMax = 0.0;
    uint64_t localFloorLikeCells = 0;

    for (auto const& kv : localCostByCellID) {
        localCostSum += kv.second;
        if (kv.second > localCostMax)
            localCostMax = kv.second;
        if (kv.second <= 1.0)
            ++localFloorLikeCells;
    }

    double globalCostSum = 0.0, globalCostMax = 0.0;
    uint64_t globalFloorLikeCells = 0;
    MPI_Allreduce(&localCostSum, &globalCostSum, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&localCostMax, &globalCostMax, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&localFloorLikeCells, &globalFloorLikeCells, 1, MPI_UINT64_T, MPI_SUM, comm);

    double globalMeanCost = (globalCellCount > 0)
        ? globalCostSum / static_cast<double>(globalCellCount) : 0.0;

    if (rank == 0) {
        std::cerr << "MEASURED_LB_GLOBAL"
                  << " total_cells=" << globalCellCount
                  << " total_cost=" << globalCostSum
                  << " mean_cost=" << globalMeanCost
                  << " max_cost=" << globalCostMax
                  << " max_over_mean=" << (globalMeanCost > 0.0 ? globalCostMax / globalMeanCost : 0.0)
                  << " zero_step_cells=" << globalZeroStepCells
                  << " floor_like_cells=" << globalFloorLikeCells
                  << " mode=" << (multigroup ? "multigroup" : "gray")
                  << " clamp_scope=global_mean"
                  << "\n";

        std::cerr << "MEASURED_LB_RANK_IMBALANCE"
                  << " max_rank_steps=" << maxRankSteps
                  << " mean_rank_steps=" << meanRankSteps
                  << " max_over_mean=" << (meanRankSteps > 0.0 ? maxRankSteps / meanRankSteps : 0.0)
                  << "\n";
    }

    std::cerr << std::flush;
}

void PrintPostRepartitionDiagnosticsFromWeights(
    std::vector<double> const& localWeightsAfterRepartition,
    double weightCompression,
    bool multigroup,
    MPI_Comm comm)
{
    int rank = 0, mpiSize = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpiSize);

    double localRawCostSum = 0.0;
    double localPartitionWeightSum = 0.0;
    for (double w : localWeightsAfterRepartition) {
        double const safeWeight = std::max(0.0, w);
        localRawCostSum += safeWeight;
        localPartitionWeightSum += std::pow(safeWeight, weightCompression);
    }

    double maxRankRawCost = 0.0, sumRankRawCost = 0.0;
    double maxRankPartitionWeight = 0.0, sumRankPartitionWeight = 0.0;
    MPI_Allreduce(&localRawCostSum, &maxRankRawCost, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&localRawCostSum, &sumRankRawCost, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&localPartitionWeightSum, &maxRankPartitionWeight, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&localPartitionWeightSum, &sumRankPartitionWeight, 1, MPI_DOUBLE, MPI_SUM, comm);
    double meanRankRawCost = sumRankRawCost / std::max(mpiSize, 1);
    double meanRankPartitionWeight = sumRankPartitionWeight / std::max(mpiSize, 1);

    uint64_t localNewCells = static_cast<uint64_t>(localWeightsAfterRepartition.size());
    uint64_t globalNewCells = 0;
    MPI_Allreduce(&localNewCells, &globalNewCells, 1, MPI_UINT64_T, MPI_SUM, comm);

    std::vector<double> gatheredRawCosts;
    std::vector<double> gatheredPartitionWeights;
    std::vector<uint64_t> gatheredCells;
    if (rank == 0) {
        gatheredRawCosts.resize(static_cast<size_t>(mpiSize), 0.0);
        gatheredPartitionWeights.resize(static_cast<size_t>(mpiSize), 0.0);
        gatheredCells.resize(static_cast<size_t>(mpiSize), 0);
    }
    MPI_Gather(&localRawCostSum, 1, MPI_DOUBLE,
               rank == 0 ? gatheredRawCosts.data() : nullptr, 1, MPI_DOUBLE,
               0, comm);
    MPI_Gather(&localPartitionWeightSum, 1, MPI_DOUBLE,
               rank == 0 ? gatheredPartitionWeights.data() : nullptr, 1, MPI_DOUBLE,
               0, comm);
    MPI_Gather(&localNewCells, 1, MPI_UINT64_T,
               rank == 0 ? gatheredCells.data() : nullptr, 1, MPI_UINT64_T,
               0, comm);

    if (rank == 0) {
        std::cerr << "MEASURED_LB_AFTER_SUMMARY"
                  << " total_cells=" << globalNewCells
                  << " mode=" << (multigroup ? "multigroup" : "gray")
                  << " weight_compression=" << weightCompression
                  << " max_rank_cost=" << maxRankRawCost
                  << " mean_rank_cost=" << meanRankRawCost
                  << " max_over_mean=" << (meanRankRawCost > 0.0 ? maxRankRawCost / meanRankRawCost : 0.0)
                  << " raw_predicted_max_over_mean=" << (meanRankRawCost > 0.0 ? maxRankRawCost / meanRankRawCost : 0.0)
                  << " partition_weight_max_over_mean=" << (meanRankPartitionWeight > 0.0 ? maxRankPartitionWeight / meanRankPartitionWeight : 0.0)
                  << "\n";

        std::vector<int> order(static_cast<size_t>(mpiSize), 0);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return gatheredRawCosts[static_cast<size_t>(a)] >
                   gatheredRawCosts[static_cast<size_t>(b)];
        });
        int const topCount = std::min(10, mpiSize);
        for (int i = 0; i < topCount; ++i) {
            int const r = order[static_cast<size_t>(i)];
            std::cerr << "MEASURED_LB_AFTER_TOP_RAW"
                      << " rank=" << r
                      << " predicted_raw_cost=" << gatheredRawCosts[static_cast<size_t>(r)]
                      << " partition_weight=" << gatheredPartitionWeights[static_cast<size_t>(r)]
                      << " local_cells=" << gatheredCells[static_cast<size_t>(r)]
                      << "\n";
        }
    }

    std::cerr << std::flush;
}
#endif // RICH_MPI

} // namespace imc_measured_lb
