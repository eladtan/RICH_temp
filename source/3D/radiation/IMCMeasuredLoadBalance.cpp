#include "IMCMeasuredLoadBalance.hpp"

#include <unordered_set>
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
        double const particleCost =
            params.particleWeight * static_cast<double>(m.particleCount);
        double c = params.floorCost
                 + params.stepWeight * static_cast<double>(m.stepCount)
                 + (params.particleCostAfterClamp ? 0.0 : particleCost)
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

void AddPostClampParticleCosts(
    std::unordered_map<size_t, double>& costByID,
    std::vector<LocalCellMeasurement> const& measurements,
    Parameters const& params)
{
    if (!params.particleCostAfterClamp || !(params.particleWeight > 0.0))
        return;

    for (auto const& m : measurements) {
        costByID[m.globalCellID] +=
            params.particleWeight * static_cast<double>(m.particleCount);
    }
}

double CellCountFloor(double meanCost, Parameters const& params)
{
    if (!(params.maxCellImbalance > 0.0) || !std::isfinite(params.maxCellImbalance))
        return params.floorCost;
    return std::max(params.floorCost, meanCost / params.maxCellImbalance);
}

// The cell-count floor limits a rank to maxCellImbalance times the mean cell
// count only if it is taken against the mean of the final costs. Particle
// costs added after the clamp can dominate them (predicted exploration
// packets: 440M packets against ~5e8 steps), and a floor from the step-only
// mean then lets the Hilbert partition pack the cheap cells onto a run of
// consecutive ranks (33M of 70M cells on one node, 336 GB). Re-apply the
// floor against the mean that includes the particle costs.
//
// The same holds whenever the cap lets costs far above the all-cell mean
// through (clampToActiveMean): the final mean then exceeds the one the floor
// was taken against. The floor is therefore solved to its fixed point,
// floor = mean(max(cost, floor)) / maxCellImbalance, which is what bounds a
// rank to maxCellImbalance times the mean cell count; a single pass leaves it
// below that (2.9x observed on snap_109 with the all-cell cap).
bool NeedsPostParticleCellFloor(Parameters const& params)
{
    bool const finiteImbalance =
        params.maxCellImbalance > 0.0 && std::isfinite(params.maxCellImbalance);
    bool const postClampParticles =
        params.particleCostAfterClamp && params.particleWeight > 0.0;
    return finiteImbalance && (postClampParticles || params.clampToActiveMean);
}

// Raw-cost sum and count over the cells that did work.
void ActiveCostSums(std::vector<LocalCellMeasurement> const& measurements,
                    std::unordered_map<size_t, double> const& costByID,
                    double& sum, uint64_t& count)
{
    std::unordered_set<size_t> active;
    for (auto const& m : measurements)
        if (m.stepCount > 0 || m.particleCount > 0)
            active.insert(m.globalCellID);
    sum = 0.0;
    count = 0;
    for (size_t id : active) {
        auto const it = costByID.find(id);
        if (it == costByID.end())
            continue;
        sum += it->second;
        ++count;
    }
}

// Fixed point of the cell-count floor (see NeedsPostParticleCellFloor).
// `reduce` turns local (sum, count) into global ones. Returns the floor and
// sets the final mean, the iterations and the number of cells raised (local).
template <typename Reduce>
double SolveCellCountFloor(std::unordered_map<size_t, double>& costByID,
                           Parameters const& params, Reduce const& reduce,
                           double& finalMean, int& iterations, uint64_t& raised)
{
    double floor = 0.0;
    finalMean = 0.0;
    iterations = 0;
    raised = 0;
    for (int it = 0; it < 200; ++it) {
        double sum = 0.0;
        uint64_t count = 0;
        for (auto const& kv : costByID) {
            sum += kv.second;
            ++count;
        }
        reduce(sum, count);
        if (count == 0)
            break;
        finalMean = sum / static_cast<double>(count);
        double const next = CellCountFloor(finalMean, params);
        iterations = it + 1;
        if (next <= floor * (1.0 + 1e-6))
            break;
        floor = next;
        for (auto& kv : costByID) {
            if (kv.second < floor) {
                kv.second = floor;
                ++raised;
            }
        }
    }
    return floor;
}

} // anonymous namespace

std::unordered_map<size_t, double> BuildMeasuredCosts(
    std::vector<LocalCellMeasurement> const& measurements,
    Parameters const& params,
    bool multigroup)
{
    auto costByID = BuildRawCosts(measurements, params, multigroup);

    if (!costByID.empty()) {
        double localSum = 0.0;
        for (auto const& kv : costByID)
            localSum += kv.second;
        double localMean = localSum / static_cast<double>(costByID.size());
        double reference = localMean;
        if (params.clampToActiveMean) {
            double activeSum = 0.0;
            uint64_t activeCount = 0;
            ActiveCostSums(measurements, costByID, activeSum, activeCount);
            if (activeCount > 0)
                reference = activeSum / static_cast<double>(activeCount);
        }
        double minCost = CellCountFloor(localMean, params);
        double maxCost = std::numeric_limits<double>::infinity();
        if (params.useMedianClamp) {
            maxCost = params.medianClampFactor * std::max(reference, params.floorCost);
            minCost = std::max(minCost, localMean / params.medianClampFactor);
        }
        ClampCosts(costByID, minCost, maxCost);
    }
    AddPostClampParticleCosts(costByID, measurements, params);
    if (NeedsPostParticleCellFloor(params) && !costByID.empty()) {
        double finalMean = 0.0;
        int iterations = 0;
        uint64_t raised = 0;
        SolveCellCountFloor(costByID, params, [](double&, uint64_t&) {},
                            finalMean, iterations, raised);
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

    {
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

        double activeSum = 0.0;
        uint64_t activeCount = 0;
        ActiveCostSums(measurements, costByID, activeSum, activeCount);
        MPI_Allreduce(MPI_IN_PLACE, &activeSum, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, &activeCount, 1, MPI_UINT64_T, MPI_SUM, comm);
        double const activeMean = activeCount > 0
            ? activeSum / static_cast<double>(activeCount) : globalMean;
        double const reference = params.clampToActiveMean ? activeMean : globalMean;

        double maxCost = std::numeric_limits<double>::infinity();
        double minCost = CellCountFloor(globalMean, params);
        if (params.useMedianClamp) {
            maxCost = params.medianClampFactor * std::max(reference, params.floorCost);
            minCost = std::max(minCost, globalMean / params.medianClampFactor);
        }

        // How much measured cost the ceiling discards: the cells above it
        // are balanced as if they cost the ceiling.
        double rawMax = 0.0;
        double aboveCeilingCount = 0.0;
        double aboveCeilingExcess = 0.0;
        for (auto const& kv : costByID) {
            rawMax = std::max(rawMax, kv.second);
            if (kv.second > maxCost) {
                aboveCeilingCount += 1.0;
                aboveCeilingExcess += kv.second - maxCost;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &rawMax, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(MPI_IN_PLACE, &aboveCeilingCount, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, &aboveCeilingExcess, 1, MPI_DOUBLE, MPI_SUM, comm);

        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) {
            std::cerr << "MEASURED_LB_CEILING_LOSS"
                      << " raw_max_cell_cost=" << rawMax
                      << " raw_max_over_ceiling=" << (maxCost > 0.0 ? rawMax / maxCost : 0.0)
                      << " cells_above_ceiling=" << aboveCeilingCount
                      << " cost_discarded=" << aboveCeilingExcess
                      << " cost_discarded_frac=" << (globalSum > 0.0 ? aboveCeilingExcess / globalSum : 0.0)
                      << "\n";
            std::cerr << "MEASURED_LB_CLAMP"
                      << " global_mean_cost=" << globalMean
                      << " active_mean_cost=" << activeMean
                      << " active_cells=" << activeCount
                      << " clamp_reference=" << (params.clampToActiveMean ? "active" : "all")
                      << " clamp_factor=" << params.medianClampFactor
                      << " clamp_ceiling=" << maxCost
                      << " clamp_floor=" << minCost
                      << " cell_floor=" << CellCountFloor(globalMean, params)
                      << " max_cell_imbalance=" << params.maxCellImbalance
                      << " effective_ratio=" << (minCost > 0 ? maxCost / minCost : 0)
                      << " global_cells=" << globalCount
                      << "\n";
        }

        ClampCosts(costByID, minCost, maxCost);
    }
    AddPostClampParticleCosts(costByID, measurements, params);
    if (NeedsPostParticleCellFloor(params)) {
        double finalMean = 0.0;
        int iterations = 0;
        uint64_t localRaised = 0;
        double sumBeforeFloor = 0.0;
        for (auto const& kv : costByID)
            sumBeforeFloor += kv.second;
        double const cellFloor = SolveCellCountFloor(
            costByID, params,
            [comm](double& sum, uint64_t& count) {
                MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, comm);
                MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_UINT64_T, MPI_SUM, comm);
            },
            finalMean, iterations, localRaised);
        uint64_t globalRaised = 0;
        MPI_Allreduce(&localRaised, &globalRaised, 1, MPI_UINT64_T, MPI_SUM, comm);
        // Share of the balanced weight that is the floor itself rather than
        // measured work.
        double sums[2] = {sumBeforeFloor, 0.0};
        for (auto const& kv : costByID)
            sums[1] += kv.second;
        MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, comm);
        int rank = 0;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) {
            std::cerr << "MEASURED_LB_POST_PARTICLE_FLOOR"
                      << " final_mean_cost=" << finalMean
                      << " cell_floor=" << cellFloor
                      << " iterations=" << iterations
                      << " cell_raises=" << globalRaised
                      << " floor_added_weight_frac=" << (sums[1] > 0.0 ? (sums[1] - sums[0]) / sums[1] : 0.0)
                      << " max_cells_per_rank_bound=" << params.maxCellImbalance
                      << "x_mean\n";
        }
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
        uint64_t minRankCells = gatheredCells.empty() ? 0 : gatheredCells.front();
        uint64_t maxRankCells = 0;
        int minCellRank = 0;
        int maxCellRank = 0;
        for (int r = 0; r < mpiSize; ++r) {
            uint64_t const cells = gatheredCells[static_cast<size_t>(r)];
            if (cells < minRankCells) {
                minRankCells = cells;
                minCellRank = r;
            }
            if (cells > maxRankCells) {
                maxRankCells = cells;
                maxCellRank = r;
            }
        }
        double const meanRankCells =
            static_cast<double>(globalNewCells) / std::max(mpiSize, 1);

        std::cerr << "MEASURED_LB_AFTER_SUMMARY"
                  << " total_cells=" << globalNewCells
                  << " mode=" << (multigroup ? "multigroup" : "gray")
                  << " weight_compression=" << weightCompression
                  << " local_cells_min=" << minRankCells
                  << " local_cells_min_rank=" << minCellRank
                  << " local_cells_mean=" << meanRankCells
                  << " local_cells_max=" << maxRankCells
                  << " local_cells_max_rank=" << maxCellRank
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
