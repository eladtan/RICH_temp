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

std::unordered_map<size_t, double> BuildMeasuredCosts(
    std::vector<LocalCellMeasurement> const& globalMeasurements,
    Parameters const& params,
    bool multigroup)
{
    std::unordered_map<size_t, double> costByID;
    costByID.reserve(globalMeasurements.size());

    for (auto const& m : globalMeasurements) {
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

    if (params.useMedianClamp) {
        std::vector<double> rawCosts;
        rawCosts.reserve(costByID.size());
        for (auto const& kv : costByID)
            rawCosts.push_back(kv.second);

        double med = MedianPositive(rawCosts);
        double maxCost = params.medianClampFactor * med;

        for (auto& kv : costByID)
            kv.second = std::min(std::max(kv.second, params.floorCost), maxCost);
    }

    return costByID;
}

#ifdef RICH_MPI
std::vector<LocalCellMeasurement> GatherMeasurementsAllRanks(
    std::vector<LocalCellMeasurement> const& localMeasurements,
    MPI_Comm comm)
{
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

void PrintMeasuredLBDiagnostics(
    std::vector<LocalCellMeasurement> const& localMeasurements,
    std::unordered_map<size_t, double> const& costByCellID,
    bool multigroup,
    MPI_Comm comm)
{
    int rank = 0, mpiSize = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpiSize);

    size_t localStepSum = 0;
    size_t localStepMax = 0;
    size_t localNonzero = 0;
    for (auto const& m : localMeasurements) {
        localStepSum += m.stepCount;
        if (m.stepCount > localStepMax)
            localStepMax = m.stepCount;
        if (m.stepCount > 0)
            ++localNonzero;
    }

    std::cerr << "MEASURED_LB_BEFORE rank=" << rank
              << " local_cells=" << localMeasurements.size()
              << " local_step_sum=" << localStepSum
              << " local_step_max=" << localStepMax
              << " local_nonzero_cells=" << localNonzero
              << "\n";

    // Gather rank step sums to compute imbalance ratio
    double localStepSumD = static_cast<double>(localStepSum);
    double maxRankSteps = 0.0, sumRankSteps = 0.0;
    MPI_Allreduce(&localStepSumD, &maxRankSteps, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&localStepSumD, &sumRankSteps, 1, MPI_DOUBLE, MPI_SUM, comm);
    double meanRankSteps = sumRankSteps / std::max(mpiSize, 1);

    if (rank == 0) {
        double totalCost = 0.0;
        double maxCost = 0.0;
        size_t zeroStepCells = 0;
        std::vector<double> allCosts;
        allCosts.reserve(costByCellID.size());

        for (auto const& kv : costByCellID) {
            allCosts.push_back(kv.second);
            totalCost += kv.second;
            if (kv.second > maxCost)
                maxCost = kv.second;
        }

        double meanCost = costByCellID.empty() ? 0.0
                        : totalCost / static_cast<double>(costByCellID.size());
        double medianCost = MedianPositive(allCosts);

        for (auto const& kv : costByCellID) {
            if (kv.second <= 1.0)
                ++zeroStepCells;
        }

        std::cerr << "MEASURED_LB_GLOBAL"
                  << " total_cells=" << costByCellID.size()
                  << " total_cost=" << totalCost
                  << " mean_cost=" << meanCost
                  << " median_cost=" << medianCost
                  << " max_cost=" << maxCost
                  << " max_over_mean=" << (meanCost > 0.0 ? maxCost / meanCost : 0.0)
                  << " zero_step_cells=" << zeroStepCells
                  << " mode=" << (multigroup ? "multigroup" : "gray")
                  << "\n";

        std::cerr << "MEASURED_LB_RANK_IMBALANCE"
                  << " max_rank_steps=" << maxRankSteps
                  << " mean_rank_steps=" << meanRankSteps
                  << " max_over_mean=" << (meanRankSteps > 0.0 ? maxRankSteps / meanRankSteps : 0.0)
                  << "\n";
    }

    std::cerr << std::flush;
}

void PrintPostRepartitionDiagnostics(
    std::unordered_map<size_t, double> const& costByCellID,
    std::vector<size_t> const& newCellIDs,
    double missingCellCost,
    bool multigroup,
    MPI_Comm comm)
{
    int rank = 0, mpiSize = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &mpiSize);

    double localCostSum = 0.0;
    for (size_t id : newCellIDs) {
        auto it = costByCellID.find(id);
        localCostSum += (it != costByCellID.end()) ? it->second : missingCellCost;
    }

    double maxRankCost = 0.0, sumRankCost = 0.0;
    MPI_Allreduce(&localCostSum, &maxRankCost, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&localCostSum, &sumRankCost, 1, MPI_DOUBLE, MPI_SUM, comm);
    double meanRankCost = sumRankCost / std::max(mpiSize, 1);

    std::cerr << "MEASURED_LB_AFTER rank=" << rank
              << " new_local_cells=" << newCellIDs.size()
              << " new_predicted_local_cost=" << localCostSum
              << "\n";

    if (rank == 0) {
        std::cerr << "MEASURED_LB_AFTER_SUMMARY"
                  << " mode=" << (multigroup ? "multigroup" : "gray")
                  << " max_rank_cost=" << maxRankCost
                  << " mean_rank_cost=" << meanRankCost
                  << " max_over_mean=" << (meanRankCost > 0.0 ? maxRankCost / meanRankCost : 0.0)
                  << "\n";
    }

    std::cerr << std::flush;
}
#endif // RICH_MPI

} // namespace imc_measured_lb
