#include "RadiationDriver.hpp"
#include "RadiationDriverTestHooks.hpp"
#include "RadiationMpiFailure.hpp"
#include "SpectralPositivity.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#if defined(__linux__)
#include <malloc.h>
#include <sys/resource.h>
#endif

#include "misc/universal_error.hpp"
#ifdef RICH_MPI
#include "mpi/mpi_commands.hpp"
#endif

namespace {

struct ActivePassiveCorrectionLedger
{
    double radiation_before = 0;
    double correction_sum = 0;
    double most_negative_term = 0;
    std::size_t most_negative_group = std::numeric_limits<std::size_t>::max();
    std::size_t terms = 0;
};

double dot(std::vector<double> const& left, std::vector<double> const& right)
{
    double result = 0;
    for(std::size_t i = 0; i < left.size(); ++i)
        result += left[i] * right[i];
    return result;
}

double verifiedBackwardErrorTolerance(
    double const requested_tolerance,
    std::size_t const maximum_row_nonzeros)
{
    return std::max(requested_tolerance,
        32 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(maximum_row_nonzeros));
}

std::string spectralRepairFailureReason(
    bool const passive,
    RadiationPositivity::ControlledSpectralRepairResult const& controlled,
    double const relative_tolerance,
    double const global_maximum_cell_radiation_extent,
    double const total_radiation_extent,
    double const mass)
{
    RadiationPositivity::SpectralRepairResult const& repair =
        controlled.repair;
    std::ostringstream reason;
    reason << std::setprecision(17)
           << (passive ? "passive" : "active");
    if(repair.repaired_groups > 0)
        reason << " multigroup radiation negativity exceeds "
                  "the controlled spectral positivity repair: group="
               << repair.most_negative_group
               << " extent=" << repair.most_negative_extent;
    else {
        reason << " multigroup radiation state is invalid before "
                  "the controlled spectral positivity repair: reason="
               << RadiationPositivity::SpectralRepairFailureLabel(
                      repair.failure);
        if(repair.failure_group != std::numeric_limits<std::size_t>::max())
            reason << " group=" << repair.failure_group
                   << " extent=" << repair.failure_extent;
        else
            reason << " failure_extent=" << repair.failure_extent;
    }
    reason << " failure="
           << RadiationPositivity::SpectralRepairFailureLabel(repair.failure)
           << " negative_extent=" << repair.negative_extent
           << " positive_extent=" << repair.positive_extent
           << " relative_deficit=" << repair.relative_deficit
           << " tolerance=" << relative_tolerance
           << " negative_extent_over_global_E_max="
           << controlled.global_negative.
                  negative_extent_to_global_max_ratio
           << " global_negative_tolerance="
           << RadiationPositivity::
                  spectral_globally_negligible_negative_fraction
           << " diagnostic_E_cell_over_E_max="
           << controlled.diagnostic_total_to_global_max_ratio
           << " global_E_max="
           << global_maximum_cell_radiation_extent
           << " total_radiation_extent=" << total_radiation_extent
           << " mass=" << mass;
    return reason.str();
}

class AccurateResidualAccumulator
{
public:
    void Add(double const value)
    {
        double const sum = high_ + value;
        double const value_in_sum = sum - high_;
        double const addition_error =
            (high_ - (sum - value_in_sum)) + (value - value_in_sum);
        double const corrected_low = low_ + addition_error;
        high_ = sum + corrected_low;
        low_ = corrected_low - (high_ - sum);
    }

    void AddProduct(double const left, double const right)
    {
        double const product = left * right;
        double const product_error = std::fma(left, right, -product);
        Add(product);
        Add(product_error);
    }

    double Value() const
    {
        return high_ + low_;
    }

private:
    double high_ = 0;
    double low_ = 0;
};

void multiply(CG::mat const& matrix,
              CG::size_t_mat const& columns,
              std::vector<double> const& input,
              std::vector<double>& output)
{
    output.assign(matrix.size(), 0);
    for(std::size_t row = 0; row < matrix.size(); ++row)
        for(std::size_t entry = 0; entry < matrix[row].size(); ++entry)
            if(columns[row][entry] != CG::max_size_t)
                output[row] += matrix[row][entry] * input.at(columns[row][entry]);
}

struct TrueResidualAssessment
{
    double backward_error = std::numeric_limits<double>::infinity();
    std::size_t maximum_row_nonzeros = 0;
    std::size_t representative_unknown = CG::max_size_t;
    int representative_rank = 0;
    double representative_residual =
        std::numeric_limits<double>::quiet_NaN();
    double representative_scale =
        std::numeric_limits<double>::quiet_NaN();
    double maximum_scale = 0;
    double safe_minimum_scale = std::numeric_limits<double>::min();
    bool finite = false;
};

bool computeResidualAccurately(
    CG::mat const& matrix,
    CG::size_t_mat const& columns,
    std::vector<double> const& input,
    std::vector<double> const& rhs,
    std::vector<double> const& fixed_scale,
    std::vector<double> const* const scale_base,
    std::size_t const unknowns_per_cell,
    std::vector<double>& residual,
    TrueResidualAssessment& assessment)
{
    bool valid = matrix.size() == input.size() &&
        columns.size() == input.size() && rhs.size() == input.size() &&
        fixed_scale.size() == input.size() &&
        (scale_base == nullptr || scale_base->size() == input.size()) &&
        unknowns_per_cell > 0;
    assessment.backward_error = 0;
    assessment.maximum_row_nonzeros = 0;
    assessment.representative_unknown = CG::max_size_t;
    assessment.representative_rank = 0;
    assessment.representative_residual =
        std::numeric_limits<double>::quiet_NaN();
    assessment.representative_scale =
        std::numeric_limits<double>::quiet_NaN();
    assessment.maximum_scale = 0;
    assessment.safe_minimum_scale = std::numeric_limits<double>::min();
    assessment.finite = true;
    residual.assign(input.size(), 0);
    std::vector<double> row_scales(input.size(), 0);
    std::vector<double> maximum_group_scales(unknowns_per_cell, 0);
    for(std::size_t row = 0; row < matrix.size() &&
            row < columns.size() && row < rhs.size(); ++row) {
        if(matrix[row].size() != columns[row].size()) {
            valid = false;
            continue;
        }
        AccurateResidualAccumulator row_residual;
        AccurateResidualAccumulator row_scale;
        row_residual.Add(rhs[row]);
        row_scale.Add(fixed_scale[row]);
        std::size_t row_nonzeros = 0;
        for(std::size_t entry = 0; entry < matrix[row].size(); ++entry) {
            std::size_t const column = columns[row][entry];
            if(column == CG::max_size_t)
                continue;
            ++row_nonzeros;
            if(column >= input.size()) {
                valid = false;
                continue;
            }
            row_residual.AddProduct(-matrix[row][entry], input[column]);
            double const scale_value = scale_base == nullptr ?
                input[column] : scale_base->at(column) + input[column];
            row_scale.AddProduct(std::abs(matrix[row][entry]),
                                 std::abs(scale_value));
        }
        residual[row] = row_residual.Value();
        double const scale = row_scale.Value();
        row_scales[row] = scale;
        bool const finite = std::isfinite(residual[row]) &&
            std::isfinite(scale) && scale >= 0;
        assessment.finite = assessment.finite && finite;
        if(finite)
            maximum_group_scales[row % unknowns_per_cell] = std::max(
                maximum_group_scales[row % unknowns_per_cell], scale);
        assessment.maximum_row_nonzeros = std::max(
            assessment.maximum_row_nonzeros, row_nonzeros);
    }
    for(std::size_t row = 0; row < residual.size(); ++row) {
        if(!std::isfinite(residual[row]) ||
           !std::isfinite(row_scales[row]))
            continue;
        double const maximum_scale =
            maximum_group_scales[row % unknowns_per_cell];
        double const safe_minimum_scale = std::max(
            std::numeric_limits<double>::min(),
            32 * std::numeric_limits<double>::epsilon() * maximum_scale);
        double const row_backward_error = std::abs(residual[row]) /
            std::max(row_scales[row], safe_minimum_scale);
        if(row_backward_error > assessment.backward_error ||
           assessment.representative_unknown == CG::max_size_t) {
            assessment.backward_error = row_backward_error;
            assessment.representative_unknown = row;
            assessment.representative_residual = residual[row];
            assessment.representative_scale = row_scales[row];
            assessment.maximum_scale = maximum_scale;
            assessment.safe_minimum_scale = safe_minimum_scale;
        }
    }
    if(!valid)
        assessment.finite = false;
    return valid;
}

unsigned long long BlockCellId(
    std::vector<std::size_t> const& block_cell_ids,
    std::size_t const block)
{
    return block < block_cell_ids.size() ?
        static_cast<unsigned long long>(block_cell_ids[block]) :
        std::numeric_limits<unsigned long long>::max();
}

void ReportSerialPreconditionerSetup(
    CG::CellBlockJacobiPreconditioner const& preconditioner,
    std::vector<std::size_t> const& block_cell_ids,
    bool const detailed)
{
    if(detailed)
        std::clog << "MG_PRECONDITIONER_SETUP kind="
              << CG::PreconditionerKindLabel(preconditioner.Kind())
              << " scope=serial_active"
              << " block_size=" << preconditioner.BlockSize()
              << " blocks=" << preconditioner.BlockCount()
              << " factorized_blocks="
              << preconditioner.FactorizedBlockCount()
              << " fallback_blocks=" << preconditioner.FallbackBlockCount()
              << " min_normalized_pivot="
              << preconditioner.MinimumNormalizedPivot()
              << " row_equilibrated="
              << (preconditioner.Kind() !=
                  CG::PreconditionerKind::ScalarJacobi ? 1 : 0)
              << " block_sweeps="
              << CG::CellBlockJacobiSweepCount(preconditioner.Kind())
              << " neighbor_correction_damping="
              << (CG::UsesCellBlockNeighborCorrection(preconditioner.Kind()) ?
                  CG::cell_block_neighbor_correction_damping : 0)
              << " relative_pivot_threshold="
              << 64 * std::numeric_limits<double>::epsilon()
              << " scalar_fallback_enabled=1 regularization=0"
              << " setup_seconds=" << preconditioner.SetupSeconds()
              << " storage_bytes=" << preconditioner.StorageBytes()
              << std::endl;
    if(preconditioner.FallbackBlockCount() > 0)
        std::clog << "MG_PRECONDITIONER_FALLBACK count="
                  << preconditioner.FallbackBlockCount()
                  << " representative_rank=0 representative_block="
                  << preconditioner.FirstFallbackBlock()
                  << " representative_cell_id=" << BlockCellId(
                      block_cell_ids, preconditioner.FirstFallbackBlock())
                  << " representative_group="
                  << preconditioner.FirstFallbackGroup()
                  << " reason=" << CG::CellBlockFallbackReasonLabel(
                      preconditioner.FirstFallbackReason()) << std::endl;
}

#ifdef RICH_MPI
[[noreturn]] void abortDistributedActiveFailure(
    char const* operation, int mpi_error,
    char const* detail = nullptr) noexcept;
void requireDistributedMpiSuccess(
    int error, char const* operation) noexcept;

void ReportDistributedPreconditionerSetup(
    CG::CellBlockJacobiPreconditioner const& preconditioner,
    std::vector<std::size_t> const& block_cell_ids)
{
    int rank = 0;
    requireDistributedMpiSuccess(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank(distributed preconditioner report)");
    unsigned long long counts[4] = {
        static_cast<unsigned long long>(preconditioner.BlockCount()),
        static_cast<unsigned long long>(preconditioner.FactorizedBlockCount()),
        static_cast<unsigned long long>(preconditioner.FallbackBlockCount()),
        static_cast<unsigned long long>(preconditioner.StorageBytes())};
    unsigned long long storage_max = counts[3];
    double setup_max = preconditioner.SetupSeconds();
    double minimum_pivot = preconditioner.MinimumNormalizedPivot();
    if(!std::isfinite(minimum_pivot))
        minimum_pivot = std::numeric_limits<double>::infinity();
    int representative_rank = preconditioner.FallbackBlockCount() > 0 ?
        rank : std::numeric_limits<int>::max();
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, counts, 4, MPI_UNSIGNED_LONG_LONG,
                      MPI_SUM, MPI_COMM_WORLD),
        "MPI_Allreduce(distributed preconditioner counts)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &storage_max, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(distributed preconditioner storage)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &setup_max, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(distributed preconditioner setup time)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &minimum_pivot, 1, MPI_DOUBLE, MPI_MIN,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(distributed preconditioner pivot)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &representative_rank, 1, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD),
        "MPI_Allreduce(distributed preconditioner representative rank)");
    if(!std::isfinite(minimum_pivot))
        minimum_pivot = std::numeric_limits<double>::quiet_NaN();
    if(rank == 0)
        std::clog << "MG_PRECONDITIONER_SETUP kind="
                  << CG::PreconditionerKindLabel(preconditioner.Kind())
                  << " scope=distributed_active"
                  << " block_size=" << preconditioner.BlockSize()
                  << " blocks=" << counts[0]
                  << " factorized_blocks=" << counts[1]
                  << " fallback_blocks=" << counts[2]
                  << " min_normalized_pivot=" << minimum_pivot
                  << " row_equilibrated="
                  << (preconditioner.Kind() !=
                      CG::PreconditionerKind::ScalarJacobi ? 1 : 0)
                  << " block_sweeps="
                  << CG::CellBlockJacobiSweepCount(preconditioner.Kind())
                  << " neighbor_correction_damping="
                  << (CG::UsesCellBlockNeighborCorrection(
                          preconditioner.Kind()) ?
                      CG::cell_block_neighbor_correction_damping : 0)
                  << " relative_pivot_threshold="
                  << 64 * std::numeric_limits<double>::epsilon()
                  << " scalar_fallback_enabled=1 regularization=0"
                   << " setup_seconds_max=" << setup_max
                   << " storage_bytes_total=" << counts[3]
                   << " storage_bytes_max=" << storage_max
                   << std::endl;
    if(representative_rank != std::numeric_limits<int>::max()) {
        unsigned long long representative[3] = {
            std::numeric_limits<unsigned long long>::max(),
            std::numeric_limits<unsigned long long>::max(),
            std::numeric_limits<unsigned long long>::max()};
        int reason = static_cast<int>(CG::CellBlockFallbackReason::None);
        if(rank == representative_rank) {
            representative[0] = static_cast<unsigned long long>(
                preconditioner.FirstFallbackBlock());
            representative[1] = static_cast<unsigned long long>(
                preconditioner.FirstFallbackGroup());
            representative[2] = BlockCellId(
                block_cell_ids, preconditioner.FirstFallbackBlock());
            reason = static_cast<int>(preconditioner.FirstFallbackReason());
        }
        requireDistributedMpiSuccess(
            MPI_Bcast(representative, 3, MPI_UNSIGNED_LONG_LONG,
                      representative_rank, MPI_COMM_WORLD),
            "MPI_Bcast(distributed preconditioner representative)");
        requireDistributedMpiSuccess(
            MPI_Bcast(&reason, 1, MPI_INT, representative_rank,
                      MPI_COMM_WORLD),
            "MPI_Bcast(distributed preconditioner fallback reason)");
        if(rank == 0)
            std::clog << "MG_PRECONDITIONER_FALLBACK count=" << counts[2]
                      << " representative_rank=" << representative_rank
                      << " representative_block=" << representative[0]
                      << " representative_cell_id=" << representative[2]
                      << " representative_group=" << representative[1]
                      << " reason=" << CG::CellBlockFallbackReasonLabel(
                          static_cast<CG::CellBlockFallbackReason>(reason))
                      << std::endl;
    }
}
#endif

struct ActiveBiCGSTABTiming
{
    double matvec_seconds = 0;
    double local_matvec_seconds = 0;
    double remote_matvec_seconds = 0;
    double exchange_seconds = 0;
    double exchange_pack_seconds = 0;
    double exchange_start_seconds = 0;
    double exchange_wait_seconds = 0;
    double exchange_unpack_seconds = 0;
    double reduction_seconds = 0;
    double preconditioner_setup_seconds = 0;
    double preconditioner_apply_seconds = 0;
    unsigned long long matvec_calls = 0;
    unsigned long long exchange_calls = 0;
    unsigned long long exchange_validity_reductions = 0;
    unsigned long long dot_reductions = 0;
    unsigned long long true_residual_reductions = 0;
    unsigned long long report_identity_reductions = 0;
};

using ActiveBiCGSTABMetrics = CG::HistoricalMGMetrics;

enum class ActiveBreakdownResolution
{
    Converged,
    Restart,
    Rejected
};

struct ActiveFinalCorrectionCandidate
{
    CG::HistoricalMGCorrectionAssessment assessment;
    std::vector<double> corrected_delta;
    std::vector<double> capped_corrected_delta;
    CG::HistoricalMGResidualCorrectionDiagnostics correction_diagnostics;
    std::uint64_t pre_correction_negative_group_count = 0;
    double pre_correction_negative_extent = 0;
    std::uint64_t introduced_negative_group_count = 0;
    double introduced_negative_extent = 0;
    std::uint64_t amplified_negative_group_count = 0;
    double amplified_negative_extent = 0;
    CG::HistoricalMGCorrectedNegativity corrected_negativity;
    CG::HistoricalMGPositiveFloorAssessment positive_floor;
    CG::HistoricalMGCorrectionSpectralFailure spectral_failure;
};

ActiveFinalCorrectionCandidate BuildActiveFinalCorrectionCandidate(
    std::vector<double> const& base_solution,
    std::vector<double> const& solution,
    std::vector<double> const& true_residual,
    std::vector<double> const& correction_volume,
    std::size_t const unknowns_per_cell,
    std::vector<std::size_t> const& block_cell_ids)
{
    ActiveFinalCorrectionCandidate candidate;
    std::size_t const size = solution.size();
    if(base_solution.size() != size || true_residual.size() != size ||
       correction_volume.size() != size || unknowns_per_cell == 0 ||
       size % unknowns_per_cell != 0 ||
       block_cell_ids.size() != size / unknowns_per_cell)
        return candidate;
    candidate.corrected_delta.resize(size);
    CG::ResetHistoricalMGResidualCorrectionDiagnostics(
        candidate.correction_diagnostics, unknowns_per_cell, size);
    candidate.assessment.finite = true;
    for(std::size_t row = 0; row < size; ++row) {
        double const volume = correction_volume[row];
        if(!std::isfinite(volume) || volume <= 0 ||
           !std::isfinite(base_solution[row]) ||
           !std::isfinite(solution[row]) ||
           !std::isfinite(true_residual[row])) {
            candidate.assessment.finite = false;
            break;
        }
        double const pre_correction_physical =
            base_solution[row] + solution[row];
        double const unscaled_corrected_delta = solution[row] +
            true_residual[row] / volume;
        double const unscaled_corrected_physical = base_solution[row] +
            unscaled_corrected_delta;
        std::size_t const group = row % unknowns_per_cell;
        std::size_t const block = row / unknowns_per_cell;
        CG::HistoricalMGResidualCorrectionLimit const limit =
            CG::DetermineHistoricalMGResidualCorrectionLimit(
                pre_correction_physical * volume,
                unscaled_corrected_physical * volume,
                block_cell_ids[block], group);
        CG::RecordHistoricalMGResidualCorrectionLimit(
            limit, candidate.correction_diagnostics);
        candidate.correction_diagnostics.correction_scale[row] = limit.Scale;
        candidate.corrected_delta[row] = solution[row] + limit.Scale *
            true_residual[row] / volume;
        candidate.correction_diagnostics.pre_correction_solution[row] =
            pre_correction_physical;
        double const corrected_physical = base_solution[row] +
            candidate.corrected_delta[row];
        if(!std::isfinite(candidate.corrected_delta[row]) ||
           !std::isfinite(pre_correction_physical) ||
           !std::isfinite(corrected_physical)) {
            candidate.assessment.finite = false;
            break;
        }
        if(pre_correction_physical < 0) {
            ++candidate.pre_correction_negative_group_count;
            candidate.pre_correction_negative_extent -=
                pre_correction_physical * volume;
        }
        if(corrected_physical < 0) {
            ++candidate.assessment.negative_group_count;
            candidate.assessment.negative_extent -=
                corrected_physical * volume;
            if(pre_correction_physical >= 0) {
                ++candidate.introduced_negative_group_count;
                candidate.introduced_negative_extent -=
                    corrected_physical * volume;
            }
            else if(corrected_physical < pre_correction_physical) {
                ++candidate.amplified_negative_group_count;
                candidate.amplified_negative_extent +=
                    (pre_correction_physical - corrected_physical) * volume;
            }
        }
    }
    if(!std::isfinite(candidate.pre_correction_negative_extent) ||
       !std::isfinite(candidate.introduced_negative_extent) ||
       !std::isfinite(candidate.amplified_negative_extent) ||
       !candidate.correction_diagnostics.finite)
        candidate.assessment.finite = false;
    if(!candidate.assessment.finite) {
        candidate.corrected_delta.clear();
        candidate.correction_diagnostics.finite = false;
    }
    return candidate;
}

struct FixedPositiveRadiationScale
{
    bool finite = true;
    double maximum_cell_energy = 0;
    double total_energy = 0;
};

FixedPositiveRadiationScale FixedCellPositiveRadiationScale(
    Tessellation3D const& Tess,
    std::vector<double> const& FullSolution,
    std::vector<std::size_t> const& SolvedUnknowns,
    std::size_t const UnknownsPerCell,
    double const LengthScale)
{
    FixedPositiveRadiationScale Result;
    if(UnknownsPerCell == 0 ||
       Tess.GetPointNo() > FullSolution.size() / UnknownsPerCell) {
        Result.finite = false;
        return Result;
    }
    std::vector<unsigned char> SolvedCells(Tess.GetPointNo(), 0);
    for(std::size_t const Unknown : SolvedUnknowns) {
        std::size_t const Cell = Unknown / UnknownsPerCell;
        if(Cell < SolvedCells.size())
            SolvedCells[Cell] = 1;
    }

    double const LengthScaleCubed = LengthScale * LengthScale * LengthScale;
    for(std::size_t Cell = 0; Cell < SolvedCells.size(); ++Cell) {
        if(SolvedCells[Cell])
            continue;
        double const Volume = Tess.GetVolume(Cell) * LengthScaleCubed;
        if(!std::isfinite(Volume) || Volume <= 0) {
            Result.finite = false;
            return Result;
        }
        long double PositiveCellEnergy = 0;
        for(std::size_t Group = 0; Group < UnknownsPerCell; ++Group) {
            double const Eg = FullSolution[Cell * UnknownsPerCell + Group];
            if(!std::isfinite(Eg)) {
                Result.finite = false;
                return Result;
            }
            if(Eg >= 0)
                PositiveCellEnergy += static_cast<long double>(Eg) * Volume;
        }
        double const PositiveCellEnergyDouble =
            static_cast<double>(PositiveCellEnergy);
        if(!std::isfinite(PositiveCellEnergyDouble)) {
            Result.finite = false;
            return Result;
        }
        Result.maximum_cell_energy = std::max(
            Result.maximum_cell_energy, PositiveCellEnergyDouble);
        Result.total_energy += PositiveCellEnergyDouble;
        if(!std::isfinite(Result.total_energy)) {
            Result.finite = false;
            return Result;
        }
    }
    return Result;
}

double FixedCellRadiationMaximumAbsoluteGroup(
    Tessellation3D const& Tess,
    std::vector<double> const& FullSolution,
    std::vector<std::size_t> const& SolvedUnknowns,
    std::size_t const UnknownsPerCell)
{
    if(UnknownsPerCell == 0 ||
       Tess.GetPointNo() > FullSolution.size() / UnknownsPerCell)
        return std::numeric_limits<double>::quiet_NaN();
    std::vector<unsigned char> SolvedCells(Tess.GetPointNo(), 0);
    for(std::size_t const Unknown : SolvedUnknowns) {
        std::size_t const Cell = Unknown / UnknownsPerCell;
        if(Cell < SolvedCells.size())
            SolvedCells[Cell] = 1;
    }

    double Maximum = 0;
    for(std::size_t Cell = 0; Cell < SolvedCells.size(); ++Cell) {
        if(SolvedCells[Cell])
            continue;
        for(std::size_t Group = 0; Group < UnknownsPerCell; ++Group) {
            double const Eg = FullSolution[Cell * UnknownsPerCell + Group];
            if(!std::isfinite(Eg))
                return std::numeric_limits<double>::quiet_NaN();
            Maximum = std::max(Maximum, std::abs(Eg));
        }
    }
    return Maximum;
}

bool AssessActiveFinalCorrectionSpectralFailure(
    std::vector<double> const& BaseSolution,
    std::vector<double> const& Solution,
    std::vector<double> const& CorrectionVolume,
    std::size_t const UnknownsPerCell,
    std::vector<std::size_t> const& BlockCellIds,
    double const FixedCellMaximumAbsoluteEg,
    double const FixedMaximumPositiveCellEnergy,
    double const FixedPositiveEnergy,
    bool const CollectGlobally,
    ActiveFinalCorrectionCandidate& Candidate)
{
    std::size_t const Size = Solution.size();
    bool Valid = UnknownsPerCell > 0 && Size % UnknownsPerCell == 0 &&
        BaseSolution.size() == Size && CorrectionVolume.size() == Size &&
        Candidate.corrected_delta.size() == Size &&
        BlockCellIds.size() == Size / UnknownsPerCell &&
        std::isfinite(FixedCellMaximumAbsoluteEg) &&
        FixedCellMaximumAbsoluteEg >= 0 &&
        std::isfinite(FixedMaximumPositiveCellEnergy) &&
        FixedMaximumPositiveCellEnergy >= 0 &&
        std::isfinite(FixedPositiveEnergy) && FixedPositiveEnergy >= 0;

    double GlobalExtents[2] = {0, FixedPositiveEnergy};
    std::vector<double> PreCorrectionEg(Size, 0);
    std::vector<double> PostCorrectionEg(Size, 0);
    for(std::size_t Block = 0; Valid && Block < BlockCellIds.size(); ++Block) {
        for(std::size_t Group = 0; Group < UnknownsPerCell; ++Group) {
            std::size_t const Row = Block * UnknownsPerCell + Group;
            double const Volume = CorrectionVolume[Row];
            PreCorrectionEg[Row] = BaseSolution[Row] + Solution[Row];
            PostCorrectionEg[Row] =
                BaseSolution[Row] + Candidate.corrected_delta[Row];
            if(!std::isfinite(Volume) || Volume <= 0 ||
               !std::isfinite(PreCorrectionEg[Row]) ||
               !std::isfinite(PostCorrectionEg[Row])) {
                Valid = false;
                break;
            }
            if(PostCorrectionEg[Row] < 0)
                GlobalExtents[0] -= PostCorrectionEg[Row] * Volume;
            else
                GlobalExtents[1] += PostCorrectionEg[Row] * Volume;
        }
    }
#ifdef RICH_MPI
    if(CollectGlobally) {
        int GloballyValid = Valid ? 1 : 0;
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &GloballyValid, 1, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(corrected candidate validity)");
        if(GloballyValid == 0)
            return false;
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, GlobalExtents, 2, MPI_DOUBLE,
                          MPI_SUM, MPI_COMM_WORLD),
            "MPI_Allreduce(corrected candidate extents)");
    }
    else if(!Valid)
        return false;
#else
    (void)CollectGlobally;
    if(!Valid)
        return false;
#endif

    Candidate.corrected_negativity =
        CG::AssessHistoricalMGCorrectedNegativity(
            PreCorrectionEg, PostCorrectionEg,
            Candidate.correction_diagnostics.correction_scale,
            UnknownsPerCell, BlockCellIds, FixedCellMaximumAbsoluteEg,
            CollectGlobally);
    if(!Candidate.corrected_negativity.Finite)
        return false;
    if(!CG::CollectHistoricalMGCorrectedNegativityGlobalExtents(
           Candidate.corrected_negativity, GlobalExtents[0],
           GlobalExtents[1], false))
        return false;
    Candidate.capped_corrected_delta = Candidate.corrected_delta;
    Candidate.positive_floor = CG::AssessAndApplyHistoricalMGPositiveFloor(
        PreCorrectionEg, PostCorrectionEg,
        Candidate.correction_diagnostics.correction_scale, CorrectionVolume,
        UnknownsPerCell, BlockCellIds, FixedMaximumPositiveCellEnergy,
        FixedPositiveEnergy, CollectGlobally);
    CG::RecordHistoricalMGPositiveFloor(
        Candidate.positive_floor, Candidate.correction_diagnostics);
    if(Candidate.positive_floor.Applied)
        for(std::size_t Row = 0; Row < Size; ++Row)
            Candidate.corrected_delta[Row] =
                PostCorrectionEg[Row] - BaseSolution[Row];
    Candidate.spectral_failure = CG::HistoricalMGPositiveFloorFailure(
        Candidate.positive_floor, Candidate.corrected_negativity);
    return true;
}

bool BuildLimitedActiveFinalCorrectionCandidate(
    std::vector<double> const& BaseSolution,
    std::vector<double> const& Solution,
    std::vector<double> const& TrueResidual,
    std::vector<double> const& CorrectionVolume,
    std::size_t const UnknownsPerCell,
    std::vector<std::size_t> const& BlockCellIds,
    double const FixedCellMaximumAbsoluteEg,
    double const FixedMaximumPositiveCellEnergy,
    double const FixedPositiveEnergy,
    bool const CollectGlobally,
    ActiveFinalCorrectionCandidate& Candidate)
{
    Candidate = BuildActiveFinalCorrectionCandidate(
        BaseSolution, Solution, TrueResidual, CorrectionVolume,
        UnknownsPerCell, BlockCellIds);
#ifdef RICH_MPI
    if(CollectGlobally)
        CG::CollectHistoricalMGResidualCorrectionLimitingDiagnostic(
            Candidate.correction_diagnostics);
#endif
    return Candidate.assessment.finite &&
        AssessActiveFinalCorrectionSpectralFailure(
            BaseSolution, Solution, CorrectionVolume, UnknownsPerCell,
            BlockCellIds, FixedCellMaximumAbsoluteEg,
            FixedMaximumPositiveCellEnergy, FixedPositiveEnergy,
            CollectGlobally, Candidate);
}

char const* HistoricalCorrectionRejectionReason(
    CG::HistoricalMGCorrectionDisposition const Disposition)
{
    return Disposition ==
        CG::HistoricalMGCorrectionDisposition::RejectPostCapNonphysical ?
        "historical_residual_correction_post_cap_nonphysical" :
        "historical_final_correction_nonfinite";
}

struct ActiveUnknownIdentity
{
    unsigned long long cell_id =
        std::numeric_limits<unsigned long long>::max();
    unsigned long long group =
        std::numeric_limits<unsigned long long>::max();
};

ActiveUnknownIdentity ActiveIdentity(
    std::size_t const unknown,
    std::size_t const unknowns_per_cell,
    std::vector<std::size_t> const& block_cell_ids)
{
    ActiveUnknownIdentity identity;
    if(unknown == CG::max_size_t || unknowns_per_cell == 0)
        return identity;
    std::size_t const block = unknown / unknowns_per_cell;
    if(block >= block_cell_ids.size())
        return identity;
    identity.cell_id = static_cast<unsigned long long>(block_cell_ids[block]);
    identity.group = static_cast<unsigned long long>(
        unknown % unknowns_per_cell);
    return identity;
}

void ReportSerialActiveBiCGSTAB(
    char const* const marker,
    char const* const outcome,
    char const* const reason,
    std::size_t const iterations,
    double const error,
    ActiveBiCGSTABMetrics const& metrics,
    std::size_t const unknowns_per_cell,
    std::vector<std::size_t> const& block_cell_ids,
    double const backward_error =
        std::numeric_limits<double>::quiet_NaN(),
    double const backward_tolerance =
        std::numeric_limits<double>::quiet_NaN(),
    std::size_t const maximum_row_nonzeros = 0,
    std::size_t const backward_unknown = CG::max_size_t,
    int const backward_rank = 0,
    double const backward_residual =
        std::numeric_limits<double>::quiet_NaN(),
    double const backward_scale =
        std::numeric_limits<double>::quiet_NaN(),
    double const maximum_scale =
        std::numeric_limits<double>::quiet_NaN(),
    double const safe_minimum_scale =
        std::numeric_limits<double>::quiet_NaN())
{
    ActiveUnknownIdentity const max0 = ActiveIdentity(
        metrics.max0_unknown, unknowns_per_cell, block_cell_ids);
    ActiveUnknownIdentity const max1 = ActiveIdentity(
        metrics.max1_unknown, unknowns_per_cell, block_cell_ids);
    ActiveUnknownIdentity const negative = ActiveIdentity(
        metrics.negative_unknown, unknowns_per_cell, block_cell_ids);
    ActiveUnknownIdentity const backward = ActiveIdentity(
        backward_unknown, unknowns_per_cell, block_cell_ids);
    ActiveUnknownIdentity const representative = metrics.negative ?
        negative : max1;
    std::clog << marker << " scope=serial_active";
    if(outcome != nullptr)
        std::clog << " outcome=" << outcome << " reason=" << reason
                  << " iterations=" << iterations;
    else
        std::clog << " iteration=" << iterations;
    std::clog << " error=" << error
              << " weighted_residual_squared="
              << metrics.weighted_residual_squared
              << " weighted_rhs_squared=" << metrics.weighted_rhs_squared
              << " backward_error=" << backward_error
              << " backward_tolerance=" << backward_tolerance
              << " max_global_row_nnz=" << maximum_row_nonzeros
              << " backward_rank=" << backward_rank
              << " backward_cell_id=" << backward.cell_id
              << " backward_group=" << backward.group
              << " backward_residual=" << backward_residual
              << " backward_scale=" << backward_scale
              << " maximum_scale=" << maximum_scale
              << " safe_minimum_scale=" << safe_minimum_scale
              << " max0=" << metrics.max0
              << " max1=" << metrics.max1
              << " negative=" << metrics.negative
              << " representative_rank=0 representative_cell_id="
              << representative.cell_id
              << " representative_group=" << representative.group
              << " max0_rank=0 max0_cell_id=" << max0.cell_id
              << " max0_group=" << max0.group
              << " max1_rank=0 max1_cell_id=" << max1.cell_id
              << " max1_group=" << max1.group
              << " negative_rank=" << (metrics.negative ? 0 : -1)
              << " negative_cell_id=" << negative.cell_id
              << " negative_group=" << negative.group << std::endl;
}

#ifdef RICH_MPI
std::array<ActiveUnknownIdentity, 4> CollectActiveIdentities(
    std::array<std::size_t, 4> const& local_unknowns,
    std::array<int, 4> const& owner_ranks,
    int const rank,
    std::size_t const unknowns_per_cell,
    std::vector<std::size_t> const& block_cell_ids)
{
    unsigned long long values[8] = {};
    for(std::size_t slot = 0; slot < local_unknowns.size(); ++slot) {
        if(owner_ranks[slot] == std::numeric_limits<int>::max())
            continue;
        if(rank == owner_ranks[slot]) {
            ActiveUnknownIdentity const identity = ActiveIdentity(
                local_unknowns[slot], unknowns_per_cell, block_cell_ids);
            values[2 * slot] = identity.cell_id;
            values[2 * slot + 1] = identity.group;
        }
    }
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, values, 8, MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(distributed active identities)");
    std::array<ActiveUnknownIdentity, 4> identities;
    for(std::size_t slot = 0; slot < identities.size(); ++slot) {
        if(owner_ranks[slot] == std::numeric_limits<int>::max())
            continue;
        identities[slot].cell_id = values[2 * slot];
        identities[slot].group = values[2 * slot + 1];
    }
    return identities;
}

void ReportDistributedActiveBiCGSTAB(
    char const* const marker,
    char const* const outcome,
    char const* const reason,
    std::size_t const iterations,
    double const error,
    ActiveBiCGSTABMetrics const& local_metrics,
    std::size_t const local_size,
    std::size_t const unknowns_per_cell,
    std::vector<std::size_t> const& block_cell_ids,
    double& reduction_seconds,
    double const backward_error =
        std::numeric_limits<double>::quiet_NaN(),
    double const backward_tolerance =
        std::numeric_limits<double>::quiet_NaN(),
    std::size_t const maximum_row_nonzeros = 0,
    std::size_t const backward_unknown = CG::max_size_t,
    int const backward_rank = 0,
    double const backward_residual =
        std::numeric_limits<double>::quiet_NaN(),
    double const backward_scale =
        std::numeric_limits<double>::quiet_NaN(),
    double const maximum_scale =
        std::numeric_limits<double>::quiet_NaN(),
    double const safe_minimum_scale =
        std::numeric_limits<double>::quiet_NaN(),
    unsigned long long* const identity_reductions = nullptr)
{
    auto const reduction_start = std::chrono::steady_clock::now();
    int rank = 0;
    requireDistributedMpiSuccess(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank(distributed solver report)");
    (void)local_size;
    struct DoubleRank
    {
        double value;
        int rank;
    } maxima[2] = {
        {local_metrics.max0, local_metrics.max0_rank},
        {local_metrics.max1, local_metrics.max1_rank}};
    int const negative_rank = local_metrics.negative ?
        local_metrics.negative_rank : std::numeric_limits<int>::max();
    std::array<ActiveUnknownIdentity, 4> const identities =
        CollectActiveIdentities(
            {{local_metrics.max0_unknown, local_metrics.max1_unknown,
              backward_unknown, local_metrics.negative_unknown}},
            {{maxima[0].rank, maxima[1].rank, backward_rank, negative_rank}},
            rank, unknowns_per_cell, block_cell_ids);
    if(identity_reductions != nullptr)
        ++*identity_reductions;
    ActiveUnknownIdentity const& max0 = identities[0];
    ActiveUnknownIdentity const& max1 = identities[1];
    ActiveUnknownIdentity const& backward = identities[2];
    ActiveUnknownIdentity const& negative = identities[3];
    reduction_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - reduction_start).count();

    int const representative_rank =
        negative_rank != std::numeric_limits<int>::max() ?
            negative_rank : maxima[1].rank;
    ActiveUnknownIdentity const& representative =
        negative_rank != std::numeric_limits<int>::max() ? negative : max1;
    if(rank == 0)
    {
        std::clog << marker << " scope=distributed_active";
        if(outcome != nullptr)
            std::clog << " outcome=" << outcome << " reason=" << reason
                      << " iterations=" << iterations;
        else
            std::clog << " iteration=" << iterations;
        std::clog << " error=" << error
                  << " weighted_residual_squared="
                  << local_metrics.weighted_residual_squared
                  << " weighted_rhs_squared="
                  << local_metrics.weighted_rhs_squared
                  << " backward_error=" << backward_error
                  << " backward_tolerance=" << backward_tolerance
                  << " max_global_row_nnz=" << maximum_row_nonzeros
                  << " backward_rank=" << backward_rank
                  << " backward_cell_id=" << backward.cell_id
                  << " backward_group=" << backward.group
                  << " backward_residual=" << backward_residual
                  << " backward_scale=" << backward_scale
                  << " maximum_scale=" << maximum_scale
                  << " safe_minimum_scale=" << safe_minimum_scale
                  << " max0=" << std::max(0.0, maxima[0].value)
                  << " max1=" << std::max(0.0, maxima[1].value)
                  << " negative="
                  << (negative_rank != std::numeric_limits<int>::max() ? 1 : 0)
                  << " representative_rank=" << representative_rank
                  << " representative_cell_id=" << representative.cell_id
                  << " representative_group=" << representative.group
                  << " max0_rank=" << maxima[0].rank
                  << " max0_cell_id=" << max0.cell_id
                  << " max0_group=" << max0.group
                  << " max1_rank=" << maxima[1].rank
                  << " max1_cell_id=" << max1.cell_id
                  << " max1_group=" << max1.group
                  << " negative_rank="
                  << (negative_rank == std::numeric_limits<int>::max() ?
                          -1 : negative_rank)
                  << " negative_cell_id=" << negative.cell_id
                  << " negative_group=" << negative.group << std::endl;
    }
}
#endif

bool solveLocalBiCGSTAB(double tolerance,
                       int& total_iters,
                       CG::mat const& matrix,
                       CG::size_t_mat const& columns,
                       std::vector<double> const& rhs,
                       std::vector<double> const& verification_rhs,
                       std::vector<double> const& verification_scale,
                       std::vector<double> const& base_solution,
                       std::vector<double> const& final_correction_volume,
                       double const fixed_cell_maximum_absolute_Eg,
                       double const fixed_maximum_positive_cell_energy,
                       double const fixed_positive_energy,
                       std::vector<double>& solution,
                       std::size_t const unknowns_per_cell,
                       CG::PreconditionerKind const preconditioner_kind,
                       std::vector<std::size_t> const& block_cell_ids,
                       CG::MatrixBuilder const& matrix_builder,
                       CG::HistoricalMGResidualCorrectionDiagnostics&
                           correction_diagnostics)
{
    auto const total_start = std::chrono::steady_clock::now();
    ActiveBiCGSTABTiming timing;
    std::size_t const size = rhs.size();
    correction_diagnostics =
        CG::HistoricalMGResidualCorrectionDiagnostics{};
    CG::CellBlockJacobiPreconditioner direction_preconditioner;
    bool preconditioner_ready = false;
    std::size_t preconditioner_applications = 0;
    std::size_t neighbor_correction_matvec_calls = 0;
    TrueResidualAssessment last_true_assessment;
    std::size_t last_true_eta_iteration = 0;
    double pre_correction_eta_inf =
        std::numeric_limits<double>::quiet_NaN();
    double backward_tolerance =
        std::numeric_limits<double>::quiet_NaN();
    std::size_t maximum_row_nonzeros = 0;
    CG::HistoricalMGPositivityContinuation positivity_continuation;
    CG::HistoricalMGBranch AcceptedConvergenceBranch =
        CG::HistoricalMGBranch::Continue;
    auto const finish = [&](bool const result,
                            char const* const outcome,
                            char const* const reason,
                            std::size_t const iterations,
                            double const error,
                            ActiveBiCGSTABMetrics const& metrics)
    {
        if(!result && positivity_continuation.Active &&
           correction_diagnostics.failure_reason.empty()) {
            std::string const rescue_reason =
                std::string("positivity_rescue_solver_") +
                (reason != nullptr ? reason : "unknown");
            CG::RecordHistoricalMGPositivityRescueFailure(
                positivity_continuation, correction_diagnostics, iterations,
                rescue_reason.c_str(),
                RadiationPositivity::SpectralRepairFailure::
                    PositivityRescueSolverFailure);
        }
        if(positivity_continuation.Active &&
           !positivity_continuation.Closed) {
            positivity_continuation.AdditionalIterationsUsed = iterations >=
                positivity_continuation.InitialIteration ?
                iterations - positivity_continuation.InitialIteration : 0;
            positivity_continuation.BlocksCompleted = std::min(
                CG::historical_mg_positivity_continuation_maximum_blocks,
                (positivity_continuation.AdditionalIterationsUsed +
                 CG::historical_mg_positivity_continuation_block_iterations -
                 1) /
                    CG::historical_mg_positivity_continuation_block_iterations);
            CG::ReportHistoricalMGPositivityContinuationClose(
                "serial_active", positivity_continuation,
                positivity_continuation.LastNegativity, "solver_failure");
            CG::RecordHistoricalMGPositivityContinuation(
                positivity_continuation, correction_diagnostics);
        }
        total_iters = static_cast<int>(std::min<std::size_t>(
            iterations, static_cast<std::size_t>(
                std::numeric_limits<int>::max())));
        if(!result)
            ReportSerialActiveBiCGSTAB(
                "MG_BICGSTAB_CONVERGENCE", outcome, reason, iterations,
                error, metrics, unknowns_per_cell, block_cell_ids,
                last_true_assessment.backward_error, backward_tolerance,
                maximum_row_nonzeros,
                last_true_assessment.representative_unknown,
                last_true_assessment.representative_rank,
                last_true_assessment.representative_residual,
                last_true_assessment.representative_scale,
                last_true_assessment.maximum_scale,
                last_true_assessment.safe_minimum_scale);
        std::size_t const last_true_eta_age = iterations >=
            last_true_eta_iteration ? iterations - last_true_eta_iteration : 0;
        if(!result)
            std::clog << "MG_BICGSTAB_HISTORICAL_POLICY scope=serial_active"
                  << " squared_scaled_tolerance=" << tolerance
                  << " effective_norm_tolerance=" << std::sqrt(tolerance)
                  << " last_true_eta_inf="
                  << last_true_assessment.backward_error
                  << " last_true_eta_iteration=" << last_true_eta_iteration
                  << " last_true_eta_age=" << last_true_eta_age
                  << " pre_correction_eta_inf=" << pre_correction_eta_inf
                  << " final_eta_inf=not_evaluated"
                  << " eta_inf_role=diagnostic_only" << std::endl;
        if(preconditioner_ready && !result)
        {
            timing.preconditioner_apply_seconds =
                direction_preconditioner.ApplySeconds();
            std::clog << "MG_PRECONDITIONER_APPLY kind="
                      << CG::PreconditionerKindLabel(
                             direction_preconditioner.Kind())
                      << " scope=serial_active calls="
                      << direction_preconditioner.ApplyCalls()
                      << " applications=" << preconditioner_applications
                      << " neighbor_correction_matvec_calls="
                      << neighbor_correction_matvec_calls
                      << " seconds="
                      << direction_preconditioner.ApplySeconds()
                      << std::endl;
        }
        if(!result) {
            double const total_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - total_start).count();
            std::clog << "MG_BICGSTAB_TIMING scope=serial_active outcome="
                      << outcome << " iterations=" << iterations
                      << " matvec_seconds=" << timing.matvec_seconds
                      << " exchange_seconds=0 reduction_seconds=0"
                      << " preconditioner_setup_seconds="
                      << timing.preconditioner_setup_seconds
                      << " preconditioner_apply_seconds="
                      << timing.preconditioner_apply_seconds
                      << " total_seconds=" << total_seconds << std::endl;
        }
        direction_preconditioner.Release();
        return result;
    };
    ActiveBiCGSTABMetrics last_metrics;
    double last_error = std::numeric_limits<double>::quiet_NaN();
    if(matrix.size() != size || columns.size() != size ||
       verification_rhs.size() != size || verification_scale.size() != size ||
       base_solution.size() != size ||
       final_correction_volume.size() != size || solution.size() != size)
        return finish(false, "breakdown", "inconsistent_system_size", 0,
                      last_error, last_metrics);
    if(size == 0) {
        last_true_assessment.backward_error = 0;
        last_true_assessment.finite = true;
        backward_tolerance = tolerance;
        total_iters = 0;
        return finish(true, "converged", "empty_system", 0,
                      0, last_metrics);
    }

    std::vector<double> inverse_diagonal(size, 0);
    std::vector<double> diagonal(size, 0);
    for(std::size_t row = 0; row < size; ++row) {
        double row_diagonal = 0;
        std::size_t row_nonzeros = 0;
        for(std::size_t entry = 0; entry < columns[row].size(); ++entry)
            if(columns[row][entry] != CG::max_size_t) {
                ++row_nonzeros;
                if(columns[row][entry] == row)
                    row_diagonal += matrix[row][entry];
            }
        maximum_row_nonzeros = std::max(maximum_row_nonzeros,
                                        row_nonzeros);
        if(!std::isfinite(row_diagonal) || row_diagonal <= 0)
            return finish(false, "breakdown", "invalid_diagonal", 0,
                          last_error, last_metrics);
        diagonal[row] = row_diagonal;
        inverse_diagonal[row] = 1.0 / row_diagonal;
    }

    if(!direction_preconditioner.Setup(matrix, columns, unknowns_per_cell,
                                       preconditioner_kind,
                                       inverse_diagonal)) {
        timing.preconditioner_setup_seconds =
            direction_preconditioner.SetupSeconds();
        return finish(false, "breakdown", "preconditioner_setup", 0,
                      last_error, last_metrics);
    }
    preconditioner_ready = true;
    timing.preconditioner_setup_seconds =
        direction_preconditioner.SetupSeconds();
    ReportSerialPreconditionerSetup(
        direction_preconditioner, block_cell_ids, false);

    backward_tolerance = verifiedBackwardErrorTolerance(
        tolerance, maximum_row_nonzeros);
    // The Krylov vector is the correction delta.  The physical solution used
    // by the authoritative residual and diagnostics is x0 + delta.
    std::fill(solution.begin(), solution.end(), 0);
    std::vector<double> physical_solution(size, 0);
    std::vector<double> previous_physical_solution = base_solution;
    auto const refresh_physical_solution = [&]()
    {
        for(std::size_t i = 0; i < size; ++i)
            physical_solution[i] = base_solution[i] + solution[i];
    };
    auto const measure_current = [&](std::vector<double> const& values)
    {
        refresh_physical_solution();
        return CG::MeasureHistoricalMG(
            physical_solution, previous_physical_solution, values,
            verification_rhs, diagonal, unknowns_per_cell);
    };
    auto const measure_update = [&](std::vector<double> const& values,
                                    std::vector<double> const& first_update,
                                    double const first_scale,
                                    std::vector<double> const& second_update,
                                    double const second_scale)
    {
        refresh_physical_solution();
        previous_physical_solution = physical_solution;
        for(std::size_t i = 0; i < previous_physical_solution.size(); ++i) {
            if(i < first_update.size())
                previous_physical_solution[i] -=
                    first_scale * first_update[i];
            if(i < second_update.size())
                previous_physical_solution[i] -=
                    second_scale * second_update[i];
        }
        return CG::MeasureHistoricalMG(
            physical_solution, previous_physical_solution, values,
            verification_rhs, diagonal, unknowns_per_cell);
    };

    std::vector<double> product;
    auto matvec_start = std::chrono::steady_clock::now();
    multiply(matrix, columns, solution, product);
    timing.matvec_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - matvec_start).count();
    std::vector<double> residual(size), shadow(size), direction(size, 0),
        preconditioned(size), matrix_direction(size), intermediate(size),
        preconditioned_intermediate(size), matrix_intermediate(size),
        preconditioner_residual(size);
    auto apply_direction_preconditioner =
        [&](std::vector<double> const& input,
            std::vector<double>& output,
            std::vector<double>& matrix_output)
    {
        direction_preconditioner.Apply(input, output);
        ++preconditioner_applications;
        std::size_t const block_sweeps = CG::CellBlockJacobiSweepCount(
            direction_preconditioner.Kind());
        for(std::size_t sweep = 1; sweep < block_sweeps; ++sweep) {
            auto const neighbor_matvec_start =
                std::chrono::steady_clock::now();
            multiply(matrix, columns, output, matrix_output);
            timing.matvec_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() -
                neighbor_matvec_start).count();
            CG::ApplyCellBlockJacobiCorrectionSweep(
                direction_preconditioner, input, matrix_output, output,
                preconditioner_residual);
            ++neighbor_correction_matvec_calls;
        }
    };
    std::vector<double> const no_update;
    for(std::size_t i = 0; i < size; ++i)
        residual[i] = rhs[i] - product[i];
    shadow = residual;

    auto const scaled_norm = [&](std::vector<double> const& values)
    {
        double sum = 0;
        for(std::size_t i = 0; i < size; ++i)
        {
            double const scaled = inverse_diagonal[i] * values[i];
            sum += scaled * scaled;
        }
        return std::sqrt(sum);
    };
    std::vector<double> sampled_true_residual;
    std::size_t requested_true_eta_iteration = 0;
    auto const recompute_true_residual = [&]()
    {
        auto const start = std::chrono::steady_clock::now();
        refresh_physical_solution();
        bool const valid = computeResidualAccurately(
            matrix, columns, physical_solution, verification_rhs,
            verification_scale, nullptr, unknowns_per_cell,
            sampled_true_residual, last_true_assessment);
        timing.matvec_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if(!valid || !last_true_assessment.finite)
            return false;
        last_true_eta_iteration = requested_true_eta_iteration;
        return std::isfinite(scaled_norm(sampled_true_residual));
    };
    if(!recompute_true_residual())
        return finish(false, "diverged", "true_initial_residual_nonfinite",
                      0, last_error, last_metrics);
    // This owner-consistent true residual is the accurately assembled
    // correction RHS for delta=0.  Use its norm and values to seed Krylov;
    // the provisional assembly above may contain stale MPI ghost copies.
    residual = sampled_true_residual;
    shadow = residual;
    last_metrics = measure_current(residual);
    last_error = last_metrics.historical_error;

    auto const attempt_historical_final_correction =
        [&](std::size_t const iterations)
    {
        pre_correction_eta_inf = last_true_assessment.backward_error;
        if(last_true_eta_iteration != iterations ||
           !last_true_assessment.finite ||
           sampled_true_residual.size() != size ||
           final_correction_volume.size() != size)
            return CG::HistoricalMGCorrectionDisposition::RejectNonFinite;
        ActiveFinalCorrectionCandidate candidate;
        if(!BuildLimitedActiveFinalCorrectionCandidate(
               base_solution, solution, sampled_true_residual,
               final_correction_volume, unknowns_per_cell, block_cell_ids,
               fixed_cell_maximum_absolute_Eg,
               fixed_maximum_positive_cell_energy, fixed_positive_energy,
               false, candidate))
            return CG::HistoricalMGCorrectionDisposition::RejectNonFinite;
        correction_diagnostics = candidate.correction_diagnostics;
        CG::HistoricalMGPositivityContinuationDecision const
            continuation_decision =
                CG::EvaluateHistoricalMGPositivityContinuation(
                    candidate.corrected_negativity, candidate.positive_floor,
                    iterations,
                    positivity_continuation);
        if(continuation_decision ==
           CG::HistoricalMGPositivityContinuationDecision::Restart)
            CG::ReportHistoricalMGPositivityContinuationOpen(
                "serial_active", positivity_continuation);
        else if(continuation_decision ==
                CG::HistoricalMGPositivityContinuationDecision::Cleared)
            CG::ReportHistoricalMGPositivityContinuationClose(
                "serial_active", positivity_continuation,
                candidate.corrected_negativity, "cleared");
        else if(continuation_decision ==
                CG::HistoricalMGPositivityContinuationDecision::Exhausted)
            CG::ReportHistoricalMGPositivityContinuationClose(
                "serial_active", positivity_continuation,
                candidate.corrected_negativity, "budget_exhausted");
        CG::RecordHistoricalMGPositivityContinuation(
            positivity_continuation, correction_diagnostics);
        if(continuation_decision ==
           CG::HistoricalMGPositivityContinuationDecision::Restart)
            return CG::HistoricalMGCorrectionDisposition::DeferPositivity;
        if(continuation_decision ==
           CG::HistoricalMGPositivityContinuationDecision::Continue)
            return CG::HistoricalMGCorrectionDisposition::ContinuePositivity;
        if(CG::ShouldCommitHistoricalMGComptonFallback(
               candidate.corrected_negativity, continuation_decision,
               matrix_builder.HistoricalMGComptonFallbackAvailable(
                   candidate.corrected_negativity.CellId))) {
            correction_diagnostics.compton_fallback_candidate = true;
            CG::RecordHistoricalMGPositiveFloor(
                CG::HistoricalMGPositiveFloorAssessment{},
                correction_diagnostics);
            solution = std::move(candidate.capped_corrected_delta);
            return CG::HistoricalMGCorrectionDisposition::Commit;
        }
        if(candidate.spectral_failure.CausedRejection) {
            CG::RecordHistoricalMGResidualCorrectionFailure(
                candidate.spectral_failure, correction_diagnostics,
                iterations, &positivity_continuation);
            return CG::HistoricalMGCorrectionDisposition::
                RejectPostCapNonphysical;
        }
        solution = std::move(candidate.corrected_delta);
        if(candidate.positive_floor.Applied) {
            requested_true_eta_iteration = iterations;
            correction_diagnostics.post_floor_true_residual_evaluated = true;
            if(!recompute_true_residual()) {
                correction_diagnostics.post_floor_true_residual_finite = false;
                correction_diagnostics.failure_reason =
                    "post_floor_true_residual_nonfinite";
                correction_diagnostics.failure_class =
                    RadiationPositivity::SpectralRepairFailure::
                        NonfiniteRepairedExtent;
                return CG::HistoricalMGCorrectionDisposition::RejectNonFinite;
            }
            ActiveBiCGSTABMetrics const post_floor_metrics =
                measure_current(sampled_true_residual);
            correction_diagnostics.post_floor_true_residual_finite =
                post_floor_metrics.finite;
            correction_diagnostics.post_floor_true_residual_error =
                post_floor_metrics.historical_error;
            if(!post_floor_metrics.finite) {
                correction_diagnostics.failure_reason =
                    "post_floor_true_residual_nonfinite";
                correction_diagnostics.failure_class =
                    RadiationPositivity::SpectralRepairFailure::
                        NonfiniteRepairedExtent;
                return CG::HistoricalMGCorrectionDisposition::RejectNonFinite;
            }
        }
        return CG::HistoricalMGCorrectionDisposition::Commit;
    };
    double rho_previous = 1;
    double alpha = 1;
    double omega = 1;
    auto const restart_from_true_residual = [&]()
    {
        residual = sampled_true_residual;
        shadow = residual;
        std::fill(direction.begin(), direction.end(), 0);
        std::fill(matrix_direction.begin(), matrix_direction.end(), 0);
        rho_previous = 1;
        alpha = 1;
        omega = 1;
    };
    auto const finish_breakdown =
        [&](CG::HistoricalMGBreakdown const breakdown,
            std::size_t const zero_based_iteration)
    {
        std::size_t const iterations = zero_based_iteration + 1;
        requested_true_eta_iteration = iterations;
        if(!recompute_true_residual()) {
            finish(false, "rejected", "diagnostic_eta_nonfinite",
                   iterations, last_error, last_metrics);
            return ActiveBreakdownResolution::Rejected;
        }
        last_metrics = measure_current(sampled_true_residual);
        last_error = last_metrics.historical_error;
        if(CG::ShouldRestartHistoricalMGFiniteBreakdown(
               breakdown, last_metrics, iterations)) {
            restart_from_true_residual();
            return ActiveBreakdownResolution::Restart;
        }
        CG::HistoricalMGDecision const decision =
            CG::ClassifyHistoricalMG(last_metrics, zero_based_iteration,
                                     tolerance, breakdown,
                                     breakdown !=
                                         CG::HistoricalMGBreakdown::NonFinite);
        if(decision.accept) {
            CG::HistoricalMGCorrectionDisposition const disposition =
                attempt_historical_final_correction(iterations);
            if(disposition ==
                   CG::HistoricalMGCorrectionDisposition::DeferPositivity ||
               disposition == CG::HistoricalMGCorrectionDisposition::
                   ContinuePositivity) {
                restart_from_true_residual();
                return ActiveBreakdownResolution::Restart;
            }
            if(disposition ==
               CG::HistoricalMGCorrectionDisposition::Commit) {
                finish(true, "converged",
                       CG::HistoricalMGBranchLabel(decision.branch),
                       iterations, last_error, last_metrics);
                return ActiveBreakdownResolution::Converged;
            }
            finish(false, "rejected",
                   HistoricalCorrectionRejectionReason(disposition), iterations,
                   last_error, last_metrics);
            return ActiveBreakdownResolution::Rejected;
        }
        finish(false, "rejected",
               CG::HistoricalMGBranchLabel(decision.branch), iterations,
               last_error, last_metrics);
        return ActiveBreakdownResolution::Rejected;
    };
    std::size_t constexpr maximum_iterations =
        CG::historical_mg_maximum_iterations;
    for(std::size_t iteration = 0;
        iteration < maximum_iterations +
            CG::historical_mg_positivity_continuation_iteration_budget;
        ++iteration) {
        if(iteration >= maximum_iterations &&
           !positivity_continuation.Active)
            break;
        double const rho = dot(shadow, residual);
        if(!std::isfinite(rho)) {
            ActiveBreakdownResolution const resolution = finish_breakdown(
                CG::HistoricalMGBreakdown::NonFinite, iteration);
            if(resolution == ActiveBreakdownResolution::Restart)
                continue;
            return resolution == ActiveBreakdownResolution::Converged;
        }
        if(std::abs(rho) <= std::numeric_limits<double>::min() * 1e100) {
            ActiveBreakdownResolution const resolution = finish_breakdown(
                CG::HistoricalMGBreakdown::TinyRho, iteration);
            if(resolution == ActiveBreakdownResolution::Restart)
                continue;
            return resolution == ActiveBreakdownResolution::Converged;
        }
        double const beta = (rho / rho_previous) * (alpha / omega);
        for(std::size_t i = 0; i < size; ++i)
            direction[i] = residual[i] + beta * (direction[i] - omega * matrix_direction[i]);
        apply_direction_preconditioner(direction, preconditioned,
                                       matrix_direction);
        matvec_start = std::chrono::steady_clock::now();
        multiply(matrix, columns, preconditioned, matrix_direction);
        timing.matvec_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - matvec_start).count();
        double const shadow_product = dot(shadow, matrix_direction);
        if(!std::isfinite(shadow_product)) {
            ActiveBreakdownResolution const resolution = finish_breakdown(
                CG::HistoricalMGBreakdown::NonFinite, iteration);
            if(resolution == ActiveBreakdownResolution::Restart)
                continue;
            return resolution == ActiveBreakdownResolution::Converged;
        }
        alpha = std::abs(shadow_product) <=
            std::numeric_limits<double>::min() * 1e100 ?
            0.0 : rho / shadow_product;
        for(std::size_t i = 0; i < size; ++i) {
            intermediate[i] = residual[i] - alpha * matrix_direction[i];
            solution[i] += alpha * preconditioned[i];
        }
        // Standard BiCGSTAB permits convergence after the alpha update.  Test
        // that candidate before forming omega: an exactly solved
        // intermediate system has a zero omega denominator, which is a happy
        // breakdown rather than a nonfinite physical state.
        last_metrics = measure_update(
            intermediate, preconditioned, alpha, no_update, 0);
        last_error = last_metrics.historical_error;
        if(!last_metrics.finite)
            return finish(false, "rejected",
                          CG::HistoricalMGBranchLabel(
                              CG::HistoricalMGBranch::RejectNonFinite),
                          iteration + 1, last_error, last_metrics);
        CG::HistoricalMGDecision intermediate_decision =
            CG::ClassifyHistoricalMGCorrectionCandidate(
                last_metrics, iteration, tolerance);
        if(intermediate_decision.reject)
            return finish(false, "rejected",
                          CG::HistoricalMGBranchLabel(
                              intermediate_decision.branch),
                          iteration + 1, last_error, last_metrics);
        if(intermediate_decision.accept) {
            requested_true_eta_iteration = iteration + 1;
            if(!recompute_true_residual())
                return finish(false, "rejected", "diagnostic_eta_nonfinite",
                              iteration + 1, last_error, last_metrics);
            last_metrics = measure_current(sampled_true_residual);
            last_error = last_metrics.historical_error;
            intermediate_decision =
                CG::ClassifyHistoricalMGCorrectionCandidate(
                last_metrics, iteration, tolerance);
            if(intermediate_decision.reject)
                return finish(false, "rejected",
                              CG::HistoricalMGBranchLabel(
                                  intermediate_decision.branch),
                              iteration + 1, last_error, last_metrics);
            if(intermediate_decision.accept) {
                CG::HistoricalMGCorrectionDisposition const disposition =
                    attempt_historical_final_correction(iteration + 1);
                if(disposition ==
                   CG::HistoricalMGCorrectionDisposition::DeferPositivity) {
                    restart_from_true_residual();
                    continue;
                }
                if(disposition ==
                       CG::HistoricalMGCorrectionDisposition::RejectNonFinite ||
                   disposition == CG::HistoricalMGCorrectionDisposition::
                       RejectPostCapNonphysical)
                    return finish(false, "rejected",
                                  HistoricalCorrectionRejectionReason(
                                      disposition),
                                  iteration + 1, last_error, last_metrics);
                if(disposition ==
                   CG::HistoricalMGCorrectionDisposition::Commit)
                    return finish(true, "converged",
                                  CG::HistoricalMGBranchLabel(
                                      intermediate_decision.branch),
                                  iteration + 1, last_error, last_metrics);
            }
        }
        apply_direction_preconditioner(intermediate,
                                       preconditioned_intermediate,
                                       matrix_intermediate);
        matvec_start = std::chrono::steady_clock::now();
        multiply(matrix, columns, preconditioned_intermediate, matrix_intermediate);
        timing.matvec_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - matvec_start).count();
        double const denominator = dot(matrix_intermediate, matrix_intermediate);
        if(!std::isfinite(denominator)) {
            ActiveBreakdownResolution const resolution = finish_breakdown(
                CG::HistoricalMGBreakdown::NonFinite, iteration);
            if(resolution == ActiveBreakdownResolution::Restart)
                continue;
            return resolution == ActiveBreakdownResolution::Converged;
        }
        omega = denominator <= std::numeric_limits<double>::min() * 1e100 ?
            0.0 : dot(matrix_intermediate, intermediate) / denominator;
        if(!std::isfinite(omega)) {
            ActiveBreakdownResolution const resolution = finish_breakdown(
                CG::HistoricalMGBreakdown::NonFinite, iteration);
            if(resolution == ActiveBreakdownResolution::Restart)
                continue;
            return resolution == ActiveBreakdownResolution::Converged;
        }
        if(std::abs(alpha) < std::numeric_limits<double>::min() * 1e100 &&
           std::abs(omega) < std::numeric_limits<double>::min() * 1e100) {
            ActiveBreakdownResolution const resolution = finish_breakdown(
                CG::HistoricalMGBreakdown::TinyAlphaOmega, iteration);
            if(resolution == ActiveBreakdownResolution::Restart)
                continue;
            return resolution == ActiveBreakdownResolution::Converged;
        }
        if(std::abs(omega) <= std::numeric_limits<double>::min()) {
            ActiveBreakdownResolution const resolution = finish_breakdown(
                CG::HistoricalMGBreakdown::TinyAlphaOmega, iteration);
            if(resolution == ActiveBreakdownResolution::Restart)
                continue;
            return resolution == ActiveBreakdownResolution::Converged;
        }
        for(std::size_t i = 0; i < size; ++i) {
            solution[i] += omega * preconditioned_intermediate[i];
            residual[i] = intermediate[i] - omega * matrix_intermediate[i];
        }
        last_metrics = measure_update(
            residual, preconditioned, alpha,
            preconditioned_intermediate, omega);
        last_error = last_metrics.historical_error;
        if(!last_metrics.finite)
            return finish(false, "rejected",
                          CG::HistoricalMGBranchLabel(
                              CG::HistoricalMGBranch::RejectNonFinite),
                          iteration + 1, last_error, last_metrics);
        CG::HistoricalMGDecision decision =
            CG::ClassifyHistoricalMGCorrectionCandidate(
                last_metrics, iteration, tolerance);
        if(decision.reject)
            return finish(false, "rejected",
                          CG::HistoricalMGBranchLabel(decision.branch),
                          iteration + 1, last_error, last_metrics);
        bool const positivity_block_boundary =
            CG::HistoricalMGPositivityContinuationBlockBoundaryReached(
                positivity_continuation, iteration + 1);
        bool const positivity_budget_reached =
            CG::HistoricalMGPositivityContinuationBudgetReached(
                positivity_continuation, iteration + 1);
        if((iteration + 1) % 50 == 0 || decision.accept ||
           positivity_block_boundary) {
            requested_true_eta_iteration = iteration + 1;
            if(!recompute_true_residual())
                return finish(false, "rejected",
                              "diagnostic_eta_nonfinite", iteration + 1,
                              last_error, last_metrics);
            last_metrics = measure_current(sampled_true_residual);
            last_error = last_metrics.historical_error;
            decision = CG::ClassifyHistoricalMGCorrectionCandidate(
                last_metrics, iteration, tolerance);
            if(decision.reject)
                return finish(false, "rejected",
                              CG::HistoricalMGBranchLabel(decision.branch),
                              iteration + 1, last_error, last_metrics);
        }
        if(CG::ShouldRestartHistoricalMGPositivityContinuation(
               decision.accept, positivity_block_boundary,
               positivity_budget_reached)) {
            positivity_continuation.AdditionalIterationsUsed = iteration + 1 -
                positivity_continuation.InitialIteration;
            positivity_continuation.BlocksCompleted = std::min(
                CG::historical_mg_positivity_continuation_maximum_blocks,
                positivity_continuation.AdditionalIterationsUsed /
                    CG::historical_mg_positivity_continuation_block_iterations);
            positivity_continuation.BlocksStarted = std::min(
                CG::historical_mg_positivity_continuation_maximum_blocks,
                positivity_continuation.BlocksCompleted + 1);
            restart_from_true_residual();
            continue;
        }
        if(decision.accept && !positivity_continuation.Active)
            AcceptedConvergenceBranch = decision.branch;
        if(CG::ShouldAttemptHistoricalMGPositivityFinalization(
               decision.accept, positivity_block_boundary,
               positivity_budget_reached)) {
            CG::HistoricalMGCorrectionDisposition const disposition =
                attempt_historical_final_correction(iteration + 1);
            if(disposition ==
               CG::HistoricalMGCorrectionDisposition::DeferPositivity) {
                restart_from_true_residual();
                continue;
            }
            if(disposition ==
                   CG::HistoricalMGCorrectionDisposition::RejectNonFinite ||
               disposition == CG::HistoricalMGCorrectionDisposition::
                   RejectPostCapNonphysical)
                return finish(false, "rejected",
                              HistoricalCorrectionRejectionReason(disposition),
                              iteration + 1, last_error, last_metrics);
            if(disposition ==
               CG::HistoricalMGCorrectionDisposition::Commit) {
                CG::HistoricalMGBranch const CompletionBranch =
                    decision.accept ? decision.branch :
                    AcceptedConvergenceBranch;
                return finish(true, "converged",
                              CG::HistoricalMGBranchLabel(CompletionBranch),
                              iteration + 1, last_error, last_metrics);
            }
        }
        rho_previous = rho;
    }
    total_iters = static_cast<int>(maximum_iterations);
    return finish(false, "not_converged", "maximum_iterations",
                  maximum_iterations, last_error, last_metrics);
}

bool parseDistributedActiveToggle(std::string const& setting,
                                  bool& value)
{
    if(setting == "1" || setting == "true" || setting == "TRUE" ||
       setting == "on" || setting == "ON" || setting == "yes" ||
       setting == "YES") {
        value = true;
        return true;
    }
    if(setting == "0" || setting == "false" || setting == "FALSE" ||
       setting == "off" || setting == "OFF" || setting == "no" ||
       setting == "NO") {
        value = false;
        return true;
    }
    return false;
}

bool environmentToggle(char const* const name, bool const fallback,
                       bool& valid)
{
    char const* const value = std::getenv(name);
    if(value == nullptr || value[0] == '\0')
        return fallback;
    bool parsed_value = fallback;
    valid = parseDistributedActiveToggle(value, parsed_value) && valid;
    return parsed_value;
}

enum class IndividualPassiveRadiationPolicy
{
    ImmediateConservativeLegacy = 0,
    ShadowReservoirExperimental = 1,
    FrozenDirichletMeasuredDefect = 2
};

constexpr IndividualPassiveRadiationPolicy
    individual_passive_radiation_default =
        IndividualPassiveRadiationPolicy::FrozenDirichletMeasuredDefect;

struct IndividualRadiationDefectConfiguration
{
    std::uint64_t version = 3;
    double local_withdrawal_limit = 1e-2;
    double local_absolute_limit = 1e-9;
    double event_absolute_target = 1e-6;
    double cumulative_signed_limit = 1e-4;
    double cumulative_absolute_limit = 1e-3;
};

IndividualRadiationDefectConfiguration const&
individualRadiationDefectConfiguration()
{
    static IndividualRadiationDefectConfiguration const configuration;
    return configuration;
}

char const* individualPassiveRadiationPolicyLabel(
    IndividualPassiveRadiationPolicy const policy)
{
    switch(policy) {
    case IndividualPassiveRadiationPolicy::ImmediateConservativeLegacy:
        return "legacy";
    case IndividualPassiveRadiationPolicy::ShadowReservoirExperimental:
        return "shadow";
    case IndividualPassiveRadiationPolicy::FrozenDirichletMeasuredDefect:
        return "dirichlet";
    }
    return "invalid";
}

bool parseIndividualPassiveRadiationPolicy(
    std::string const& setting,
    IndividualPassiveRadiationPolicy& policy)
{
    if(setting == "legacy") {
        policy = IndividualPassiveRadiationPolicy::
            ImmediateConservativeLegacy;
        return true;
    }
    if(setting == "shadow") {
        policy = IndividualPassiveRadiationPolicy::
            ShadowReservoirExperimental;
        return true;
    }
    if(setting == "dirichlet") {
        policy = IndividualPassiveRadiationPolicy::
            FrozenDirichletMeasuredDefect;
        return true;
    }
    return false;
}

struct IndividualPassiveRadiationRuntimeOption
{
    IndividualPassiveRadiationPolicy policy =
        individual_passive_radiation_default;
    bool valid = true;
    bool deprecated_alias_present = false;
};

IndividualPassiveRadiationRuntimeOption const&
individualPassiveRadiationRuntimeOption()
{
    // Both selectors alter the transaction's commit semantics.  Resolve them
    // once, reject an invalid or conflicting launch collectively, and cache
    // the agreed policy for all later candidates.
    static IndividualPassiveRadiationRuntimeOption const option = []()
    {
        IndividualPassiveRadiationRuntimeOption result;
        char const* const selector = std::getenv(
            "RICH_MG_INDIVIDUAL_PASSIVE_POLICY");
        bool const selector_present =
            selector != nullptr && selector[0] != '\0';
        if(selector_present)
            result.valid = parseIndividualPassiveRadiationPolicy(
                selector, result.policy);

        char const* const alias = std::getenv(
            "RICH_MG_INDIVIDUAL_SHADOW_RESERVOIRS");
        result.deprecated_alias_present =
            alias != nullptr && alias[0] != '\0';
        if(result.deprecated_alias_present) {
            bool alias_valid = true;
            bool const alias_enabled = environmentToggle(
                "RICH_MG_INDIVIDUAL_SHADOW_RESERVOIRS", false,
                alias_valid);
            IndividualPassiveRadiationPolicy const alias_policy =
                alias_enabled ?
                IndividualPassiveRadiationPolicy::
                    ShadowReservoirExperimental :
                IndividualPassiveRadiationPolicy::
                    ImmediateConservativeLegacy;
            if(selector_present && result.policy != alias_policy)
                result.valid = false;
            else if(!selector_present)
                result.policy = alias_policy;
            result.valid = alias_valid && result.valid;
        }

        int local_state = result.valid ? static_cast<int>(result.policy) : 3;
#ifdef RICH_MPI
        int minimum_state = local_state;
        int maximum_state = local_state;
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &minimum_state, 1, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(minimum individual passive policy)");
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &maximum_state, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(maximum individual passive policy)");
        result.valid = minimum_state == maximum_state &&
            minimum_state >= 0 && maximum_state <= 2;
        if(result.valid)
            result.policy = static_cast<IndividualPassiveRadiationPolicy>(
                minimum_state);
#else
        result.valid = local_state >= 0 && local_state <= 2;
#endif
        return result;
    }();
    return option;
}

#ifdef RICH_MPI
bool checkedDistributedSizeAdd(std::size_t const left,
                               std::size_t const right,
                               std::size_t& result)
{
    if(right > std::numeric_limits<std::size_t>::max() - left)
        return false;
    result = left + right;
    return true;
}

bool checkedDistributedSizeMultiply(std::size_t const left,
                                    std::size_t const right,
                                    std::size_t& result)
{
    if(left != 0 &&
       right > std::numeric_limits<std::size_t>::max() / left)
        return false;
    result = left * right;
    return true;
}

[[noreturn]] void abortDistributedActiveFailure(
    char const* const operation,
    int const mpi_error,
    char const* const detail) noexcept
{
    RadiationMpi::AbortFailure(
        "MG_DISTRIBUTED_ACTIVE_FATAL", operation, mpi_error, detail);
}

void requireDistributedMpiSuccess(int const error,
                                  char const* const operation) noexcept
{
    RadiationMpi::RequireSuccess(
        error, "MG_DISTRIBUTED_ACTIVE_FATAL", operation);
}

struct IndividualActiveUnknownRequest : public Serializable
{
    std::size_t cell_id = 0;
    std::size_t group = 0;

    force_inline std::size_t dump(Serializer* serializer) const override
    {
        std::size_t bytes = 0;
        bytes += serializer->insert(cell_id);
        bytes += serializer->insert(group);
        return bytes;
    }

    force_inline std::size_t load(
        Serializer const* serializer,
        std::size_t byte_offset) override
    {
        std::size_t bytes = 0;
        bytes += serializer->extract(cell_id, byte_offset);
        bytes += serializer->extract(group, byte_offset + bytes);
        return bytes;
    }
};

struct IndividualRadiationDelta : public Serializable
{
    std::size_t cell_id = 0;
    std::size_t counterpart_cell_id = 0;
    std::size_t group = 0;
    double gain = 0;
    double reference_time_step = 0;

    force_inline std::size_t dump(Serializer* serializer) const override
    {
        std::size_t bytes = 0;
        bytes += serializer->insert(cell_id);
        bytes += serializer->insert(counterpart_cell_id);
        bytes += serializer->insert(group);
        bytes += serializer->insert(gain);
        bytes += serializer->insert(reference_time_step);
        return bytes;
    }

    force_inline std::size_t load(
        Serializer const* serializer,
        std::size_t byte_offset) override
    {
        std::size_t bytes = 0;
        bytes += serializer->extract(cell_id, byte_offset);
        bytes += serializer->extract(counterpart_cell_id, byte_offset + bytes);
        bytes += serializer->extract(group, byte_offset + bytes);
        bytes += serializer->extract(gain, byte_offset + bytes);
        bytes += serializer->extract(reference_time_step, byte_offset + bytes);
        return bytes;
    }
};

// One active/passive diffusion-face contribution used to construct the
// storage-only passive row on the canonical owner of the passive cell.  The
// active endpoint may be absent from that owner's partial tessellation, hence
// both stable IDs and the active base value travel with the coefficient.
struct IndividualShadowFaceRecord : public Serializable
{
    std::size_t passive_cell_id = 0;
    std::size_t active_cell_id = 0;
    std::size_t group = 0;
    int active_owner = -1;
    double coefficient = 0;
    double passive_volume_cgs = 0;
    double active_base = 0;

    force_inline std::size_t dump(Serializer* serializer) const override
    {
        std::size_t bytes = 0;
        bytes += serializer->insert(passive_cell_id);
        bytes += serializer->insert(active_cell_id);
        bytes += serializer->insert(group);
        bytes += serializer->insert(active_owner);
        bytes += serializer->insert(coefficient);
        bytes += serializer->insert(passive_volume_cgs);
        bytes += serializer->insert(active_base);
        return bytes;
    }

    force_inline std::size_t load(
        Serializer const* serializer,
        std::size_t byte_offset) override
    {
        std::size_t bytes = 0;
        bytes += serializer->extract(passive_cell_id, byte_offset + bytes);
        bytes += serializer->extract(active_cell_id, byte_offset + bytes);
        bytes += serializer->extract(group, byte_offset + bytes);
        bytes += serializer->extract(active_owner, byte_offset + bytes);
        bytes += serializer->extract(coefficient, byte_offset + bytes);
        bytes += serializer->extract(passive_volume_cgs, byte_offset + bytes);
        bytes += serializer->extract(active_base, byte_offset + bytes);
        return bytes;
    }
};

struct IndividualShadowInitialValue : public Serializable
{
    std::size_t passive_cell_id = 0;
    std::size_t group = 0;
    double value = 0;

    force_inline std::size_t dump(Serializer* serializer) const override
    {
        std::size_t bytes = 0;
        bytes += serializer->insert(passive_cell_id);
        bytes += serializer->insert(group);
        bytes += serializer->insert(value);
        return bytes;
    }

    force_inline std::size_t load(
        Serializer const* serializer,
        std::size_t byte_offset) override
    {
        std::size_t bytes = 0;
        bytes += serializer->extract(passive_cell_id, byte_offset + bytes);
        bytes += serializer->extract(group, byte_offset + bytes);
        bytes += serializer->extract(value, byte_offset + bytes);
        return bytes;
    }
};

struct IndividualShadowStateRequest : public Serializable
{
    std::size_t passive_cell_id = 0;

    force_inline std::size_t dump(Serializer* serializer) const override
    {
        return serializer->insert(passive_cell_id);
    }

    force_inline std::size_t load(
        Serializer const* serializer,
        std::size_t byte_offset) override
    {
        return serializer->extract(passive_cell_id, byte_offset);
    }
};

struct IndividualShadowPrimitiveValue : public Serializable
{
    std::size_t passive_cell_id = 0;
    std::size_t group = 0;
    double specific_energy = 0;

    force_inline std::size_t dump(Serializer* serializer) const override
    {
        std::size_t bytes = 0;
        bytes += serializer->insert(passive_cell_id);
        bytes += serializer->insert(group);
        bytes += serializer->insert(specific_energy);
        return bytes;
    }

    force_inline std::size_t load(
        Serializer const* serializer,
        std::size_t byte_offset) override
    {
        std::size_t bytes = 0;
        bytes += serializer->extract(passive_cell_id, byte_offset + bytes);
        bytes += serializer->extract(group, byte_offset + bytes);
        bytes += serializer->extract(specific_energy, byte_offset + bytes);
        return bytes;
    }
};

struct RemoteUnknownKey
{
    int owner = -1;
    std::size_t cell_id = 0;
    std::size_t group = 0;

    bool operator<(RemoteUnknownKey const& other) const
    {
        return std::tie(owner, cell_id, group) <
               std::tie(other.owner, other.cell_id, other.group);
    }
};

struct DistributedActiveCSR
{
    std::vector<std::size_t> row_offsets;
    std::vector<std::uint32_t> columns32;
    std::vector<CG::matrix_index_t> columns_wide;
    std::vector<double> values;
    std::vector<std::size_t> local_rows;
    std::vector<std::size_t> remote_rows;
    bool uses_narrow_columns = true;

    std::size_t RowCount() const
    {
        return row_offsets.empty() ? 0 : row_offsets.size() - 1;
    }

    std::size_t ColumnCount() const
    {
        return uses_narrow_columns ? columns32.size() :
            columns_wide.size();
    }

    std::size_t Column(std::size_t const entry) const
    {
        return uses_narrow_columns ?
            static_cast<std::size_t>(columns32[entry]) :
            columns_wide[entry];
    }

    void ReserveColumns(std::size_t const count,
                        bool const narrow_columns)
    {
        uses_narrow_columns = narrow_columns;
        if(uses_narrow_columns)
            columns32.reserve(count);
        else
            columns_wide.reserve(count);
    }

    void ReserveStorageForAppend(std::size_t const required,
                                 std::size_t const maximum)
    {
        if(required <= values.capacity())
            return;
        if(required > maximum)
            throw std::length_error(
                "distributed active CSR capacity exceeds exact nonzero count");

        std::size_t capacity = values.capacity();
        if(capacity == 0)
            capacity = std::min<std::size_t>(
                maximum, std::max<std::size_t>(required, 4096));
        while(capacity < required) {
            std::size_t const growth =
                std::max<std::size_t>(capacity / 2, 1);
            if(capacity > maximum - growth)
                capacity = maximum;
            else
                capacity += growth;
        }
        values.reserve(capacity);
        if(uses_narrow_columns)
            columns32.reserve(capacity);
        else
            columns_wide.reserve(capacity);
    }

    void PushColumn(std::size_t const column)
    {
        if(uses_narrow_columns &&
           column <= static_cast<std::size_t>(
               std::numeric_limits<std::uint32_t>::max())) {
            columns32.push_back(static_cast<std::uint32_t>(column));
            return;
        }
        if(uses_narrow_columns) {
            columns_wide.reserve(columns32.size() + 1);
            columns_wide.insert(columns_wide.end(), columns32.begin(),
                                columns32.end());
            std::vector<std::uint32_t>().swap(columns32);
            uses_narrow_columns = false;
        }
        columns_wide.push_back(column);
    }

    unsigned long long ColumnStorageBytes() const
    {
        return uses_narrow_columns ?
            static_cast<unsigned long long>(columns32.capacity()) *
                sizeof(std::uint32_t) :
            static_cast<unsigned long long>(columns_wide.capacity()) *
                sizeof(CG::matrix_index_t);
    }

    void Release()
    {
        std::vector<std::size_t>().swap(row_offsets);
        std::vector<std::uint32_t>().swap(columns32);
        std::vector<CG::matrix_index_t>().swap(columns_wide);
        std::vector<double>().swap(values);
        std::vector<std::size_t>().swap(local_rows);
        std::vector<std::size_t>().swap(remote_rows);
        uses_narrow_columns = true;
    }
};

struct DistributedActivePeerTransfer
{
    int peer = -1;
    std::size_t offset = 0;
    int count = 0;
};

constexpr bool distributed_active_overlap_default = false;

bool appendDistributedActiveTransferRange(
    int const peer, std::size_t const begin, std::size_t const end,
    std::vector<DistributedActivePeerTransfer>& transfers)
{
    if(begin > end)
        return false;
    std::size_t const maximum_chunk =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    std::size_t const remaining_count = end - begin;
    std::size_t chunk_count = remaining_count / maximum_chunk;
    if(remaining_count % maximum_chunk != 0 &&
       !checkedDistributedSizeAdd(chunk_count, 1, chunk_count))
        return false;
    std::size_t required_size = 0;
    if(!checkedDistributedSizeAdd(
           transfers.size(), chunk_count, required_size))
        return false;
    transfers.reserve(required_size);

    std::size_t offset = begin;
    std::size_t remaining = remaining_count;
    while(remaining > 0) {
        std::size_t const chunk = std::min(remaining, maximum_chunk);
        DistributedActivePeerTransfer transfer;
        transfer.peer = peer;
        transfer.offset = offset;
        transfer.count = static_cast<int>(chunk);
        transfers.push_back(transfer);
        if(!checkedDistributedSizeAdd(offset, chunk, offset))
            return false;
        remaining -= chunk;
    }
    return offset == end;
}

struct DistributedActiveExchange
{
    std::size_t local_size = 0;
    std::size_t global_size = 0;
    std::size_t remote_size = 0;
    std::vector<std::size_t> send_offsets;
    std::vector<std::size_t> receive_offsets;
    std::vector<std::size_t> send_local_indices;
    std::vector<std::size_t> receive_remote_slots;
    std::vector<double> send_values;
    std::vector<double> receive_values;
    std::vector<double> remote_values;
    std::vector<double> auxiliary_remote_values;
    std::vector<DistributedActivePeerTransfer> send_transfers;
    std::vector<DistributedActivePeerTransfer> receive_transfers;
    std::vector<MPI_Request> requests;
    std::size_t request_send_chunks = 0;
    std::size_t request_receive_chunks = 0;
    bool overlap_local_rows = distributed_active_overlap_default;
    bool pair_omega_reduction = true;
    bool rank_profile_enabled = false;
    bool in_flight = false;
};

bool collectiveAllTrue(bool local_value)
{
    int value = local_value ? 1 : 0;
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(collectiveAllTrue)");
    return value != 0;
}

bool collectiveDistributedTagIsValid(int const local_tag)
{
    int* tag_upper_bound = nullptr;
    int attribute_found = 0;
    requireDistributedMpiSuccess(
        MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &tag_upper_bound,
                          &attribute_found),
        "MPI_Comm_get_attr(MPI_TAG_UB)");
    int minimum_tag = local_tag;
    int maximum_tag = local_tag;
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &minimum_tag, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(minimum request tag)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &maximum_tag, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(maximum request tag)");
    bool const locally_valid = attribute_found != 0 &&
        tag_upper_bound != nullptr && minimum_tag == maximum_tag &&
        minimum_tag >= 0 && maximum_tag <= *tag_upper_bound;
    return collectiveAllTrue(locally_valid);
}

struct DistributedActiveRuntimeOptions
{
    bool profile = false;
    // Stage-4 AB timing found no benefit from overlap (off was 0.266% faster,
    // within noise).  Keep it reversible, but prefer the simpler default.
    bool overlap_local_rows = distributed_active_overlap_default;
    bool pair_omega_reduction = true;
    bool fixed_16_remote_slots = false;
    IndividualPassiveRadiationPolicy passive_policy =
        individual_passive_radiation_default;

    bool usesShadowPassiveRows() const
    {
        return passive_policy == IndividualPassiveRadiationPolicy::
            ShadowReservoirExperimental;
    }

    bool usesFrozenDirichlet() const
    {
        return passive_policy == IndividualPassiveRadiationPolicy::
            FrozenDirichletMeasuredDefect;
    }
};

bool loadDistributedActiveRuntimeOptions(
    DistributedActiveRuntimeOptions& options)
{
    struct CachedOptions
    {
        DistributedActiveRuntimeOptions values;
        bool consistent = false;
    };
    // These are launch-time environment options.  Validate them collectively
    // once, then avoid adding two reductions to every radiation event.
    static CachedOptions const cached = []()
    {
        CachedOptions result;
        bool locally_valid = true;
        result.values.profile = environmentToggle(
            "RICH_MG_DISTRIBUTED_ACTIVE_PROFILE", false, locally_valid);
        result.values.overlap_local_rows = environmentToggle(
            "RICH_MG_DISTRIBUTED_ACTIVE_OVERLAP",
            distributed_active_overlap_default, locally_valid);
        result.values.pair_omega_reduction = environmentToggle(
            "RICH_MG_DISTRIBUTED_ACTIVE_PAIRED_OMEGA", true,
            locally_valid);
        result.values.fixed_16_remote_slots = environmentToggle(
            "RICH_MG_DISTRIBUTED_ACTIVE_FIXED16_REMOTE_SLOTS", false,
            locally_valid);
        IndividualPassiveRadiationRuntimeOption const& passive_option =
            individualPassiveRadiationRuntimeOption();
        result.values.passive_policy = passive_option.policy;
        locally_valid = passive_option.valid && locally_valid;
        int const local_mask = (result.values.profile ? 1 : 0) |
            (result.values.overlap_local_rows ? 2 : 0) |
            (result.values.pair_omega_reduction ? 4 : 0) |
            (result.values.fixed_16_remote_slots ? 8 : 0) |
            (static_cast<int>(result.values.passive_policy) << 4);
        int minimum_mask = local_mask;
        int maximum_mask = local_mask;
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &minimum_mask, 1, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(minimum runtime option mask)");
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &maximum_mask, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(maximum runtime option mask)");
        result.consistent = minimum_mask == maximum_mask &&
            collectiveAllTrue(locally_valid);
        return result;
    }();
    options = cached.values;
    return cached.consistent;
}

template<class T>
void appendRankDistribution(
    std::ostream& stream,
    char const* const label,
    std::vector<std::pair<T, int> > samples)
{
    stream << ' ' << label << "_available_ranks=" << samples.size();
    if(samples.empty())
        return;
    std::sort(samples.begin(), samples.end(),
              [](std::pair<T, int> const& left,
                 std::pair<T, int> const& right)
              {
                  return left.first < right.first ||
                      (left.first == right.first &&
                       left.second < right.second);
              });
    std::size_t const median_index = samples.size() / 2;
    std::size_t const p95_index = std::min(
        samples.size() - 1,
        (95 * samples.size() + 99) / 100 - 1);
    std::pair<T, int> const& minimum = samples.front();
    std::pair<T, int> const& median = samples[median_index];
    std::pair<T, int> const& p95 = samples[p95_index];
    std::pair<T, int> const& maximum = samples.back();
    stream << ' ' << label << "_min=" << minimum.first
           << ' ' << label << "_min_rank=" << minimum.second
           << ' ' << label << "_median=" << median.first
           << ' ' << label << "_median_rank=" << median.second
           << ' ' << label << "_p95=" << p95.first
           << ' ' << label << "_p95_rank=" << p95.second
           << ' ' << label << "_max=" << maximum.first
           << ' ' << label << "_max_rank=" << maximum.second;
}

void reportDistributedActivePhaseTiming(
    std::array<double, 11> const& local_seconds)
{
    int rank = 0;
    int rank_count = 1;
    requireDistributedMpiSuccess(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank(distributed phase timing)");
    requireDistributedMpiSuccess(
        MPI_Comm_size(MPI_COMM_WORLD, &rank_count),
        "MPI_Comm_size(distributed phase timing)");
    std::vector<double> gathered;
    if(rank == 0) {
        std::size_t gathered_size = 0;
        if(!checkedDistributedSizeMultiply(
               static_cast<std::size_t>(rank_count), local_seconds.size(),
               gathered_size))
            abortDistributedActiveFailure(
                "distributed phase timing size overflow", MPI_ERR_OTHER);
        try {
            gathered.resize(gathered_size);
        }
        catch(std::exception const& error) {
            abortDistributedActiveFailure(
                "distributed phase timing storage", MPI_ERR_OTHER,
                error.what());
        }
        catch(...) {
            abortDistributedActiveFailure(
                "distributed phase timing storage", MPI_ERR_OTHER,
                "unknown allocation exception");
        }
    }
    requireDistributedMpiSuccess(
        MPI_Gather(local_seconds.data(),
                   static_cast<int>(local_seconds.size()), MPI_DOUBLE,
                   gathered.data(), static_cast<int>(local_seconds.size()),
                   MPI_DOUBLE, 0, MPI_COMM_WORLD),
        "MPI_Gather(distributed phase timing)");
    if(rank != 0)
        return;
    char const* const labels[11] = {
        "snapshot", "candidate", "matrix_build", "active_mapping",
        "csr_extract", "exchange_setup", "solver", "postprocess",
        "profile_overhead", "accounted", "wall"};
    std::clog << std::setprecision(17)
              << "MG_DISTRIBUTED_ACTIVE_PHASE_TIMING"
              << " ranks=" << rank_count;
    for(std::size_t field = 0; field < local_seconds.size(); ++field) {
        std::vector<std::pair<double, int> > samples;
        samples.reserve(static_cast<std::size_t>(rank_count));
        for(int peer = 0; peer < rank_count; ++peer)
            samples.emplace_back(
                gathered[static_cast<std::size_t>(peer) *
                             local_seconds.size() + field],
                peer);
        appendRankDistribution(std::clog, labels[field], samples);
    }
    std::clog << std::endl;
}

bool readProcKilobytes(
    char const* const path,
    char const* const field,
    unsigned long long& value)
{
    std::ifstream input(path);
    if(!input)
        return false;
    std::string line;
    while(std::getline(input, line)) {
        std::istringstream parser(line);
        std::string label;
        unsigned long long candidate = 0;
        if(parser >> label >> candidate && label == field) {
            value = candidate;
            return true;
        }
    }
    return false;
}

bool readUnsignedFile(
    char const* const path,
    unsigned long long& value)
{
    std::ifstream input(path);
    std::string token;
    if(!(input >> token) || token == "max")
        return false;
    try {
        std::size_t consumed = 0;
        unsigned long long const candidate = std::stoull(token, &consumed);
        if(consumed != token.size())
            return false;
        value = candidate;
        return true;
    }
    catch(...) {
        return false;
    }
}

struct DistributedProcessMemorySample
{
    static std::size_t constexpr field_count = 7;
    unsigned long long values[field_count] = {};
    unsigned long long valid_mask = 0;
};

DistributedProcessMemorySample captureDistributedProcessMemory()
{
    DistributedProcessMemorySample sample;
#if defined(__linux__)
    rusage usage{};
    if(getrusage(RUSAGE_SELF, &usage) == 0 && usage.ru_maxrss >= 0) {
        sample.values[0] = static_cast<unsigned long long>(usage.ru_maxrss);
        sample.valid_mask |= 1ULL << 0;
    }
    if(readProcKilobytes("/proc/self/status", "VmRSS:",
                         sample.values[1]))
        sample.valid_mask |= 1ULL << 1;
    if(readProcKilobytes("/proc/self/status", "VmHWM:",
                         sample.values[2]))
        sample.valid_mask |= 1ULL << 2;
    if(readProcKilobytes("/proc/self/smaps_rollup", "Rss:",
                         sample.values[3]))
        sample.valid_mask |= 1ULL << 3;
#if defined(__GLIBC__) && defined(__GLIBC_PREREQ)
#if __GLIBC_PREREQ(2, 33)
    struct mallinfo2 const allocator = ::mallinfo2();
    sample.values[4] = static_cast<unsigned long long>(allocator.uordblks) +
        static_cast<unsigned long long>(allocator.hblkhd);
    sample.values[5] = static_cast<unsigned long long>(allocator.arena) +
        static_cast<unsigned long long>(allocator.hblkhd);
    sample.valid_mask |= (1ULL << 4) | (1ULL << 5);
#endif
#endif
    if(readUnsignedFile("/sys/fs/cgroup/memory.peak", sample.values[6]) ||
       readUnsignedFile(
           "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
           sample.values[6]))
        sample.valid_mask |= 1ULL << 6;
#endif
    return sample;
}

void reportDistributedMemoryPhase(char const* const phase)
{
    int rank = 0;
    int rank_count = 1;
    requireDistributedMpiSuccess(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank(distributed memory report)");
    requireDistributedMpiSuccess(
        MPI_Comm_size(MPI_COMM_WORLD, &rank_count),
        "MPI_Comm_size(distributed memory report)");
    DistributedProcessMemorySample const local =
        captureDistributedProcessMemory();
    unsigned long long packed[DistributedProcessMemorySample::field_count + 1]
        = {};
    for(std::size_t field = 0;
        field < DistributedProcessMemorySample::field_count; ++field)
        packed[field] = local.values[field];
    packed[DistributedProcessMemorySample::field_count] = local.valid_mask;
    std::vector<unsigned long long> gathered;
    if(rank == 0) {
        std::size_t gathered_size = 0;
        if(!checkedDistributedSizeMultiply(
               static_cast<std::size_t>(rank_count),
               DistributedProcessMemorySample::field_count + 1,
               gathered_size))
            abortDistributedActiveFailure(
                "distributed memory profile size overflow", MPI_ERR_OTHER);
        try {
            gathered.resize(gathered_size);
        }
        catch(std::exception const& error) {
            abortDistributedActiveFailure(
                "distributed memory profile storage", MPI_ERR_OTHER,
                error.what());
        }
        catch(...) {
            abortDistributedActiveFailure(
                "distributed memory profile storage", MPI_ERR_OTHER,
                "unknown allocation exception");
        }
    }
    requireDistributedMpiSuccess(
        MPI_Gather(
            packed,
            static_cast<int>(DistributedProcessMemorySample::field_count + 1),
            MPI_UNSIGNED_LONG_LONG, gathered.data(),
            static_cast<int>(DistributedProcessMemorySample::field_count + 1),
            MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD),
        "MPI_Gather(distributed memory report)");
    if(rank != 0)
        return;
    char const* const labels[DistributedProcessMemorySample::field_count] = {
        "ru_maxrss_kib", "vmrss_kib", "vmhwm_kib", "smaps_rss_kib",
        "allocator_live_bytes", "allocator_reserved_bytes",
        "cgroup_peak_bytes"};
    std::clog << std::setprecision(17)
              << "MG_DISTRIBUTED_ACTIVE_MEMORY phase=" << phase
              << " ranks=" << rank_count;
    std::size_t const stride =
        DistributedProcessMemorySample::field_count + 1;
    for(std::size_t field = 0;
        field < DistributedProcessMemorySample::field_count; ++field) {
        std::vector<std::pair<unsigned long long, int> > samples;
        for(int peer = 0; peer < rank_count; ++peer) {
            std::size_t const base = static_cast<std::size_t>(peer) * stride;
            unsigned long long const mask = gathered[base +
                DistributedProcessMemorySample::field_count];
            if((mask & (1ULL << field)) != 0)
                samples.emplace_back(gathered[base + field], peer);
        }
        appendRankDistribution(std::clog, labels[field], std::move(samples));
    }
    std::clog << std::endl;
}

void reportDistributedActiveRankProfile(
    ActiveBiCGSTABTiming const& timing,
    DistributedActiveCSR const& matrix,
    DistributedActiveExchange const& exchange,
    unsigned long long const preconditioner_storage_bytes)
{
    int rank = 0;
    int rank_count = 1;
    requireDistributedMpiSuccess(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank(distributed rank profile)");
    requireDistributedMpiSuccess(
        MPI_Comm_size(MPI_COMM_WORLD, &rank_count),
        "MPI_Comm_size(distributed rank profile)");
    double const local_times[5] = {
        timing.local_matvec_seconds, timing.remote_matvec_seconds,
        timing.exchange_wait_seconds, timing.reduction_seconds,
        timing.preconditioner_apply_seconds};
    unsigned long long const local_counts[8] = {
        static_cast<unsigned long long>(matrix.local_rows.size()),
        static_cast<unsigned long long>(matrix.remote_rows.size()),
        static_cast<unsigned long long>(exchange.send_values.size()) *
            sizeof(double),
        static_cast<unsigned long long>(exchange.receive_values.size()) *
            sizeof(double),
        static_cast<unsigned long long>(matrix.values.size()),
        preconditioner_storage_bytes,
        static_cast<unsigned long long>(exchange.remote_size),
        matrix.uses_narrow_columns ? 32ULL :
            8ULL * sizeof(CG::matrix_index_t)};
    std::vector<double> gathered_times;
    std::vector<unsigned long long> gathered_counts;
    if(rank == 0) {
        std::size_t gathered_time_count = 0;
        std::size_t gathered_value_count = 0;
        if(!checkedDistributedSizeMultiply(
               static_cast<std::size_t>(rank_count), 5,
               gathered_time_count) ||
           !checkedDistributedSizeMultiply(
               static_cast<std::size_t>(rank_count), 8,
               gathered_value_count))
            abortDistributedActiveFailure(
                "distributed rank profile size overflow", MPI_ERR_OTHER);
        try {
            gathered_times.resize(gathered_time_count);
            gathered_counts.resize(gathered_value_count);
        }
        catch(std::exception const& error) {
            abortDistributedActiveFailure(
                "distributed rank profile storage", MPI_ERR_OTHER,
                error.what());
        }
        catch(...) {
            abortDistributedActiveFailure(
                "distributed rank profile storage", MPI_ERR_OTHER,
                "unknown allocation exception");
        }
    }
    requireDistributedMpiSuccess(
        MPI_Gather(local_times, 5, MPI_DOUBLE, gathered_times.data(), 5,
                   MPI_DOUBLE, 0, MPI_COMM_WORLD),
        "MPI_Gather(distributed rank profile times)");
    requireDistributedMpiSuccess(
        MPI_Gather(local_counts, 8, MPI_UNSIGNED_LONG_LONG,
                   gathered_counts.data(), 8, MPI_UNSIGNED_LONG_LONG, 0,
                   MPI_COMM_WORLD),
        "MPI_Gather(distributed rank profile counts)");
    if(rank != 0)
        return;
    char const* const time_labels[5] = {
        "local_matvec_seconds", "remote_matvec_seconds",
        "exchange_wait_seconds", "reduction_seconds",
        "preconditioner_apply_seconds"};
    char const* const count_labels[8] = {
        "local_rows", "remote_rows", "send_payload_bytes",
        "receive_payload_bytes", "csr_nonzeros",
        "preconditioner_storage_bytes", "remote_unknowns",
        "csr_column_slot_bits"};
    std::clog << std::setprecision(17)
              << "MG_DISTRIBUTED_ACTIVE_RANK_PROFILE ranks=" << rank_count
              << " overlap=" << (exchange.overlap_local_rows ? 1 : 0)
              << " paired_omega="
              << (exchange.pair_omega_reduction ? 1 : 0)
              << " request_send_chunks=" << exchange.request_send_chunks
              << " request_receive_chunks="
              << exchange.request_receive_chunks;
    for(std::size_t field = 0; field < 5; ++field) {
        std::vector<std::pair<double, int> > samples;
        samples.reserve(static_cast<std::size_t>(rank_count));
        for(int peer = 0; peer < rank_count; ++peer)
            samples.emplace_back(
                gathered_times[static_cast<std::size_t>(peer) * 5 + field],
                peer);
        appendRankDistribution(std::clog, time_labels[field],
                               std::move(samples));
    }
    for(std::size_t field = 0; field < 8; ++field) {
        std::vector<std::pair<unsigned long long, int> > samples;
        samples.reserve(static_cast<std::size_t>(rank_count));
        for(int peer = 0; peer < rank_count; ++peer)
            samples.emplace_back(
                gathered_counts[static_cast<std::size_t>(peer) * 8 + field],
                peer);
        appendRankDistribution(std::clog, count_labels[field],
                               std::move(samples));
    }
    std::clog << std::endl;
}

struct IndividualActiveRequestExchangeStats
{
    std::size_t send_chunks = 0;
    std::size_t receive_chunks = 0;
};

bool distributedChunkCount(std::size_t const value_count,
                           std::size_t const chunk_limit,
                           std::size_t& chunk_count)
{
    if(chunk_limit == 0)
        return false;
    chunk_count = value_count / chunk_limit;
    if(value_count % chunk_limit != 0)
        return checkedDistributedSizeAdd(chunk_count, 1, chunk_count);
    return true;
}

bool exchangeIndividualActiveUnknownRequests(
    std::vector<std::vector<IndividualActiveUnknownRequest> > const& outgoing,
    std::size_t request_chunk_limit,
    std::vector<std::vector<IndividualActiveUnknownRequest> >& incoming,
    int const request_tag,
    IndividualActiveRequestExchangeStats* const stats = nullptr)
{
    int rank = 0;
    int rank_count = 1;
    requireDistributedMpiSuccess(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank(active unknown requests)");
    requireDistributedMpiSuccess(
        MPI_Comm_size(MPI_COMM_WORLD, &rank_count),
        "MPI_Comm_size(active unknown requests)");
    bool const request_tag_valid =
        collectiveDistributedTagIsValid(request_tag);

    std::size_t const maximum_request_chunk =
        static_cast<std::size_t>(std::numeric_limits<int>::max()) / 2;
    request_chunk_limit = std::min(request_chunk_limit,
                                   maximum_request_chunk);
    bool valid = request_tag_valid &&
        outgoing.size() == static_cast<std::size_t>(rank_count) &&
        request_chunk_limit > 0;

    unsigned long long local_chunk_limit =
        static_cast<unsigned long long>(request_chunk_limit);
    valid = static_cast<std::size_t>(local_chunk_limit) ==
        request_chunk_limit && valid;
    unsigned long long minimum_chunk_limit = local_chunk_limit;
    unsigned long long maximum_chunk_limit = local_chunk_limit;
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &minimum_chunk_limit, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD),
        "MPI_Allreduce(minimum request chunk limit)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &maximum_chunk_limit, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(maximum request chunk limit)");
    valid = minimum_chunk_limit == maximum_chunk_limit && valid;
    if(!collectiveAllTrue(valid))
        return false;

    std::vector<unsigned long long> send_counts(
        static_cast<std::size_t>(rank_count), 0);
    std::vector<unsigned long long> receive_counts(
        static_cast<std::size_t>(rank_count), 0);
    for(int peer = 0; peer < rank_count; ++peer) {
        std::size_t const count = outgoing[static_cast<std::size_t>(peer)].size();
        send_counts[static_cast<std::size_t>(peer)] =
            static_cast<unsigned long long>(count);
        valid = static_cast<std::size_t>(
            send_counts[static_cast<std::size_t>(peer)]) == count && valid;
        for(IndividualActiveUnknownRequest const& request :
            outgoing[static_cast<std::size_t>(peer)]) {
            unsigned long long const cell_id =
                static_cast<unsigned long long>(request.cell_id);
            unsigned long long const group =
                static_cast<unsigned long long>(request.group);
            valid = static_cast<std::size_t>(cell_id) == request.cell_id &&
                static_cast<std::size_t>(group) == request.group && valid;
        }
    }
    if(!collectiveAllTrue(valid))
        return false;

    requireDistributedMpiSuccess(
        MPI_Alltoall(send_counts.data(), 1, MPI_UNSIGNED_LONG_LONG,
                     receive_counts.data(), 1, MPI_UNSIGNED_LONG_LONG,
                     MPI_COMM_WORLD),
        "MPI_Alltoall(active request counts)");

    std::vector<std::size_t> send_offsets(
        static_cast<std::size_t>(rank_count) + 1, 0);
    std::vector<std::size_t> receive_offsets(
        static_cast<std::size_t>(rank_count) + 1, 0);
    for(int peer = 0; peer < rank_count; ++peer) {
        std::size_t const peer_index = static_cast<std::size_t>(peer);
        std::size_t const receive_count =
            static_cast<std::size_t>(receive_counts[peer_index]);
        valid = static_cast<unsigned long long>(receive_count) ==
            receive_counts[peer_index] && valid;
        if(send_counts[peer_index] >
               static_cast<unsigned long long>(CG::max_size_t -
                                                send_offsets[peer_index]) ||
           receive_count > CG::max_size_t - receive_offsets[peer_index]) {
            valid = false;
            continue;
        }
        send_offsets[peer_index + 1] = send_offsets[peer_index] +
            static_cast<std::size_t>(send_counts[peer_index]);
        receive_offsets[peer_index + 1] = receive_offsets[peer_index] +
            receive_count;
    }
    valid = send_offsets.back() <= CG::max_size_t / 2 &&
        receive_offsets.back() <= CG::max_size_t / 2 && valid;
    if(!collectiveAllTrue(valid))
        return false;

    std::vector<unsigned long long> send_wire(2 * send_offsets.back(), 0);
    std::vector<unsigned long long> receive_wire(
        2 * receive_offsets.back(), 0);
    for(int peer = 0; peer < rank_count; ++peer) {
        std::size_t const peer_index = static_cast<std::size_t>(peer);
        std::size_t request_index = send_offsets[peer_index];
        for(IndividualActiveUnknownRequest const& request :
            outgoing[peer_index]) {
            send_wire[2 * request_index] =
                static_cast<unsigned long long>(request.cell_id);
            send_wire[2 * request_index + 1] =
                static_cast<unsigned long long>(request.group);
            ++request_index;
        }
    }

    std::size_t const self = static_cast<std::size_t>(rank);
    if(send_counts[self] != receive_counts[self])
        valid = false;
    else if(send_counts[self] > 0)
        std::copy(send_wire.begin() + 2 * send_offsets[self],
                  send_wire.begin() + 2 * send_offsets[self + 1],
                  receive_wire.begin() + 2 * receive_offsets[self]);
    if(!collectiveAllTrue(valid))
        return false;

    for(int step = 1; step < rank_count; ++step) {
        int const send_to = (rank + step) % rank_count;
        int const receive_from = (rank - step + rank_count) % rank_count;

        std::size_t receive_offset =
            receive_offsets[static_cast<std::size_t>(receive_from)];
        std::size_t receive_remaining =
            receive_offsets[static_cast<std::size_t>(receive_from) + 1] -
            receive_offset;
        std::size_t send_offset =
            send_offsets[static_cast<std::size_t>(send_to)];
        std::size_t send_remaining =
            send_offsets[static_cast<std::size_t>(send_to) + 1] - send_offset;
        std::size_t receive_chunk_count = 0;
        std::size_t send_chunk_count = 0;
        std::size_t request_count = 0;
        if(!distributedChunkCount(receive_remaining, request_chunk_limit,
                                  receive_chunk_count) ||
           !distributedChunkCount(send_remaining, request_chunk_limit,
                                  send_chunk_count) ||
           !checkedDistributedSizeAdd(receive_chunk_count, send_chunk_count,
                                      request_count))
            abortDistributedActiveFailure(
                "active unknown request count overflow", MPI_ERR_OTHER);
        std::vector<MPI_Request> requests;
        try {
            requests.resize(request_count, MPI_REQUEST_NULL);
        }
        catch(std::exception const& error) {
            abortDistributedActiveFailure(
                "active unknown request storage", MPI_ERR_OTHER,
                error.what());
        }
        catch(...) {
            abortDistributedActiveFailure(
                "active unknown request storage", MPI_ERR_OTHER,
                "unknown allocation exception");
        }
        std::size_t request_index = 0;
        while(receive_remaining > 0) {
            std::size_t const chunk = std::min(
                receive_remaining, request_chunk_limit);
            int const error = MPI_Irecv(
                receive_wire.data() + 2 * receive_offset,
                static_cast<int>(2 * chunk), MPI_UNSIGNED_LONG_LONG,
                receive_from, request_tag, MPI_COMM_WORLD,
                &requests[request_index]);
            requireDistributedMpiSuccess(
                error, "MPI_Irecv(active unknown requests)");
            ++request_index;
            receive_offset += chunk;
            receive_remaining -= chunk;
            if(stats != nullptr)
                ++stats->receive_chunks;
        }

        while(send_remaining > 0) {
            std::size_t const chunk = std::min(send_remaining,
                                               request_chunk_limit);
            int const error = MPI_Isend(
                send_wire.data() + 2 * send_offset,
                static_cast<int>(2 * chunk), MPI_UNSIGNED_LONG_LONG,
                send_to, request_tag, MPI_COMM_WORLD,
                &requests[request_index]);
            requireDistributedMpiSuccess(
                error, "MPI_Isend(active unknown requests)");
            ++request_index;
            send_offset += chunk;
            send_remaining -= chunk;
            if(stats != nullptr)
                ++stats->send_chunks;
        }
        if(request_index != requests.size())
            abortDistributedActiveFailure(
                "active unknown request accounting", MPI_ERR_OTHER);

        std::size_t waited = 0;
        while(waited < requests.size()) {
            std::size_t const batch = std::min(
                requests.size() - waited,
                static_cast<std::size_t>(std::numeric_limits<int>::max()));
            int const error = MPI_Waitall(
                static_cast<int>(batch), requests.data() + waited,
                MPI_STATUSES_IGNORE);
            requireDistributedMpiSuccess(
                error, "MPI_Waitall(active unknown requests)");
            waited += batch;
        }
    }
    if(!collectiveAllTrue(valid))
        return false;

    incoming.assign(static_cast<std::size_t>(rank_count), {});
    for(int peer = 0; peer < rank_count; ++peer) {
        std::size_t const peer_index = static_cast<std::size_t>(peer);
        incoming[peer_index].reserve(
            receive_offsets[peer_index + 1] - receive_offsets[peer_index]);
        for(std::size_t request_index = receive_offsets[peer_index];
            request_index < receive_offsets[peer_index + 1];
            ++request_index) {
            unsigned long long const wire_cell_id =
                receive_wire[2 * request_index];
            unsigned long long const wire_group =
                receive_wire[2 * request_index + 1];
            IndividualActiveUnknownRequest request;
            request.cell_id = static_cast<std::size_t>(wire_cell_id);
            request.group = static_cast<std::size_t>(wire_group);
            valid = static_cast<unsigned long long>(request.cell_id) ==
                wire_cell_id &&
                static_cast<unsigned long long>(request.group) == wire_group &&
                valid;
            incoming[peer_index].push_back(request);
        }
    }
    return collectiveAllTrue(valid);
}

std::vector<int> meshPointOwners(
    Tessellation3D const& tess,
    std::size_t point_count,
    bool& valid)
{
    int rank = 0;
    int rank_count = 1;
    requireDistributedMpiSuccess(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank(mesh point owners)");
    requireDistributedMpiSuccess(
        MPI_Comm_size(MPI_COMM_WORLD, &rank_count),
        "MPI_Comm_size(mesh point owners)");
    std::vector<int> owners(point_count, -1);
    valid = point_count <= tess.getMeshPoints().size();
    Tessellation3D::AllPointsMap const& local_to_global =
        tess.GetIndicesInAllPoints();
    valid = valid && local_to_global.size() >= tess.GetPointNo();
    for(auto const& mapping : local_to_global) {
        if(mapping.first >= owners.size()) {
            valid = false;
            continue;
        }
        owners[mapping.first] = rank;
    }

    std::vector<int> const& peers = tess.GetDuplicatedProcs();
    std::vector<std::vector<std::size_t> > const& ghosts =
        tess.GetGhostIndeces();
    valid = valid && peers.size() == ghosts.size();
    for(std::size_t peer = 0; peer < peers.size() && peer < ghosts.size(); ++peer)
    {
        if(peers[peer] < 0 || peers[peer] >= rank_count)
            valid = false;
        for(std::size_t point : ghosts[peer]) {
            if(point >= owners.size()) {
                valid = false;
                continue;
            }
            // Match SyncPartialBuildData when one mesh slot appears in
            // multiple incoming lists: the final peer supplies its value.
            owners[point] = peers[peer];
        }
    }
    for(std::size_t point = 0; point < tess.GetPointNo(); ++point)
        if(point >= owners.size() || owners[point] != rank)
            valid = false;
    return owners;
}

bool initializeDistributedExchange(
    std::vector<RemoteUnknownKey> const& remote_unknowns,
    std::map<std::size_t, std::size_t> const& owned_active_cell_bases,
    std::size_t const unknowns_per_cell,
    std::size_t local_size,
    DistributedActiveExchange& exchange,
    std::size_t const request_chunk_limit =
        static_cast<std::size_t>(std::numeric_limits<int>::max()) / 2,
    IndividualActiveRequestExchangeStats* const request_stats = nullptr,
    int const request_tag = 21059)
{
    int rank_count = 1;
    requireDistributedMpiSuccess(
        MPI_Comm_size(MPI_COMM_WORLD, &rank_count),
        "MPI_Comm_size(distributed exchange initialization)");
    if(!collectiveDistributedTagIsValid(21060))
        return false;
    exchange.local_size = local_size;
    exchange.remote_size = remote_unknowns.size();

    std::vector<std::vector<std::size_t> > send_local_indices(rank_count);
    std::vector<std::vector<std::size_t> > receive_remote_slots(rank_count);

    std::vector<std::vector<IndividualActiveUnknownRequest> > outgoing(
        rank_count);
    bool valid = unknowns_per_cell > 0;
    for(std::size_t slot = 0; slot < remote_unknowns.size(); ++slot) {
        RemoteUnknownKey const& key = remote_unknowns[slot];
        if(key.owner < 0 || key.owner >= rank_count ||
           key.group >= unknowns_per_cell) {
            valid = false;
            continue;
        }
        IndividualActiveUnknownRequest request;
        request.cell_id = key.cell_id;
        request.group = key.group;
        outgoing[key.owner].push_back(request);
        receive_remote_slots[key.owner].push_back(slot);
    }
    if(!collectiveAllTrue(valid))
        return false;

    IndividualActiveRequestExchangeStats local_request_stats;
    IndividualActiveRequestExchangeStats* const effective_request_stats =
        request_stats != nullptr ? request_stats : &local_request_stats;
    std::vector<std::vector<IndividualActiveUnknownRequest> > incoming;
    if(!exchangeIndividualActiveUnknownRequests(
           outgoing, request_chunk_limit, incoming, request_tag,
           effective_request_stats))
        return false;
    exchange.request_send_chunks = effective_request_stats->send_chunks;
    exchange.request_receive_chunks =
        effective_request_stats->receive_chunks;
    valid = incoming.size() == static_cast<std::size_t>(rank_count);
    for(int peer = 0; peer < rank_count && peer < static_cast<int>(incoming.size()); ++peer)
        for(IndividualActiveUnknownRequest const& request : incoming[peer]) {
            auto const found = owned_active_cell_bases.find(request.cell_id);
            if(found == owned_active_cell_bases.end() ||
               request.group >= unknowns_per_cell ||
               found->second > local_size ||
               request.group > local_size - found->second) {
                valid = false;
                continue;
            }
            std::size_t const local = found->second + request.group;
            if(local >= local_size) {
                valid = false;
                continue;
            }
            send_local_indices[peer].push_back(local);
        }
    if(!collectiveAllTrue(valid))
        return false;

    exchange.send_offsets.assign(static_cast<std::size_t>(rank_count) + 1, 0);
    exchange.receive_offsets.assign(
        static_cast<std::size_t>(rank_count) + 1, 0);
    for(int peer = 0; peer < rank_count; ++peer) {
        std::size_t const peer_index = static_cast<std::size_t>(peer);
        if(send_local_indices[peer].size() >
               CG::max_size_t - exchange.send_offsets[peer_index] ||
           receive_remote_slots[peer].size() >
               CG::max_size_t - exchange.receive_offsets[peer_index]) {
            valid = false;
            continue;
        }
        exchange.send_offsets[peer_index + 1] =
            exchange.send_offsets[peer_index] +
            send_local_indices[peer].size();
        exchange.receive_offsets[peer_index + 1] =
            exchange.receive_offsets[peer_index] +
            receive_remote_slots[peer].size();
    }
    if(!collectiveAllTrue(valid))
        return false;

    exchange.send_local_indices.resize(exchange.send_offsets.back());
    exchange.receive_remote_slots.resize(exchange.receive_offsets.back());
    for(int peer = 0; peer < rank_count; ++peer) {
        std::copy(send_local_indices[peer].begin(),
                  send_local_indices[peer].end(),
                  exchange.send_local_indices.begin() +
                      exchange.send_offsets[static_cast<std::size_t>(peer)]);
        std::copy(receive_remote_slots[peer].begin(),
                  receive_remote_slots[peer].end(),
                  exchange.receive_remote_slots.begin() +
                      exchange.receive_offsets[static_cast<std::size_t>(peer)]);
    }
    for(std::size_t const slot : exchange.receive_remote_slots)
        if(slot >= exchange.remote_size)
            valid = false;
    if(!collectiveAllTrue(valid))
        return false;

    for(int peer = 0; peer < rank_count; ++peer) {
        std::size_t const peer_index = static_cast<std::size_t>(peer);
        valid = appendDistributedActiveTransferRange(
                    peer, exchange.send_offsets[peer_index],
                    exchange.send_offsets[peer_index + 1],
                    exchange.send_transfers) && valid;
        valid = appendDistributedActiveTransferRange(
                    peer, exchange.receive_offsets[peer_index],
                    exchange.receive_offsets[peer_index + 1],
                    exchange.receive_transfers) && valid;
    }
    if(exchange.send_transfers.size() >
       CG::max_size_t - exchange.receive_transfers.size())
        valid = false;
    if(!collectiveAllTrue(valid))
        return false;

    exchange.send_values.assign(exchange.send_local_indices.size(), 0);
    exchange.receive_values.assign(exchange.receive_remote_slots.size(), 0);
    exchange.remote_values.assign(exchange.remote_size, 0);
    exchange.auxiliary_remote_values.assign(exchange.remote_size, 0);
    exchange.requests.assign(
        exchange.send_transfers.size() + exchange.receive_transfers.size(),
        MPI_REQUEST_NULL);

    unsigned long long local_count =
        static_cast<unsigned long long>(local_size);
    valid = static_cast<std::size_t>(local_count) == local_size && valid;
    if(!collectiveAllTrue(valid))
        return false;
    unsigned long long global_count = 0;
    requireDistributedMpiSuccess(
        MPI_Allreduce(&local_count, &global_count, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
        "MPI_Allreduce(distributed active global size)");
    exchange.global_size = static_cast<std::size_t>(global_count);
    return static_cast<unsigned long long>(exchange.global_size) == global_count;
}

bool startRemoteActiveValues(
    DistributedActiveExchange& exchange,
    std::vector<double> const& local_values,
    ActiveBiCGSTABTiming& timing)
{
    auto const pack_start = std::chrono::steady_clock::now();
    bool valid = !exchange.in_flight &&
        local_values.size() == exchange.local_size &&
        exchange.send_values.size() == exchange.send_local_indices.size() &&
        exchange.receive_values.size() ==
            exchange.receive_remote_slots.size();
    for(std::size_t index = 0;
        index < exchange.send_local_indices.size(); ++index) {
        std::size_t const local = exchange.send_local_indices[index];
        if(local >= local_values.size()) {
            valid = false;
            exchange.send_values[index] = 0;
            continue;
        }
        exchange.send_values[index] = local_values[local];
    }
    double const pack_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - pack_start).count();
    timing.exchange_pack_seconds += pack_seconds;
    timing.exchange_seconds += pack_seconds;

    auto const start = std::chrono::steady_clock::now();
    std::fill(exchange.requests.begin(), exchange.requests.end(),
              MPI_REQUEST_NULL);
    std::size_t request = 0;
    int constexpr tag = 21060;
    for(DistributedActivePeerTransfer const& transfer :
        exchange.receive_transfers) {
        int const error = MPI_Irecv(
            exchange.receive_values.data() + transfer.offset,
            transfer.count, MPI_DOUBLE, transfer.peer, tag, MPI_COMM_WORLD,
            &exchange.requests[request++]);
        requireDistributedMpiSuccess(
            error, "MPI_Irecv(distributed active values)");
    }
    for(DistributedActivePeerTransfer const& transfer :
        exchange.send_transfers) {
        int const error = MPI_Isend(
            exchange.send_values.data() + transfer.offset,
            transfer.count, MPI_DOUBLE, transfer.peer, tag, MPI_COMM_WORLD,
            &exchange.requests[request++]);
        requireDistributedMpiSuccess(
            error, "MPI_Isend(distributed active values)");
    }
    valid = valid && request == exchange.requests.size();
    exchange.in_flight = true;
    double const start_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    timing.exchange_start_seconds += start_seconds;
    timing.exchange_seconds += start_seconds;
    ++timing.exchange_calls;
    return valid;
}

bool finishRemoteActiveValues(
    DistributedActiveExchange& exchange,
    std::vector<double>& remote_values,
    ActiveBiCGSTABTiming& timing)
{
    bool valid = exchange.in_flight &&
        remote_values.size() == exchange.remote_size;
    auto const wait_start = std::chrono::steady_clock::now();
    std::size_t waited = 0;
    while(waited < exchange.requests.size()) {
        std::size_t const chunk = std::min(
            exchange.requests.size() - waited,
            static_cast<std::size_t>(std::numeric_limits<int>::max()));
        int const error = MPI_Waitall(
            static_cast<int>(chunk), exchange.requests.data() + waited,
            MPI_STATUSES_IGNORE);
        requireDistributedMpiSuccess(
            error, "MPI_Waitall(distributed active values)");
        waited += chunk;
    }
    exchange.in_flight = false;
    double const wait_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wait_start).count();
    timing.exchange_wait_seconds += wait_seconds;
    timing.exchange_seconds += wait_seconds;

    auto const unpack_start = std::chrono::steady_clock::now();
    if(remote_values.size() == exchange.remote_size)
        std::fill(remote_values.begin(), remote_values.end(), 0);
    for(std::size_t index = 0;
        index < exchange.receive_remote_slots.size(); ++index) {
        std::size_t const slot = exchange.receive_remote_slots[index];
        if(index >= exchange.receive_values.size() ||
           slot >= remote_values.size()) {
            valid = false;
            continue;
        }
        remote_values[slot] = exchange.receive_values[index];
    }
    double const unpack_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - unpack_start).count();
    timing.exchange_unpack_seconds += unpack_seconds;
    timing.exchange_seconds += unpack_seconds;
    return valid;
}

bool exchangeRemoteActiveValues(
    DistributedActiveExchange& exchange,
    std::vector<double> const& local_values,
    std::vector<double>& remote_values,
    ActiveBiCGSTABTiming& timing)
{
    bool valid = startRemoteActiveValues(exchange, local_values, timing);
    valid = finishRemoteActiveValues(exchange, remote_values, timing) && valid;
    auto const reduction_start = std::chrono::steady_clock::now();
    valid = collectiveAllTrue(valid);
    timing.reduction_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - reduction_start).count();
    ++timing.exchange_validity_reductions;
    return valid;
}

template<bool accurately, class ColumnIndex>
bool multiplyDistributedRowsWithColumns(
    DistributedActiveCSR const& matrix,
    std::vector<ColumnIndex> const& columns,
    std::vector<std::size_t> const& rows,
    std::vector<double> const& input,
    std::vector<double> const& remote_values,
    std::vector<double>& output)
{
    bool valid = matrix.row_offsets.size() == input.size() + 1 &&
        columns.size() == matrix.values.size();
    for(std::size_t const row : rows) {
        if(row >= input.size() || row + 1 >= matrix.row_offsets.size() ||
           matrix.row_offsets[row] > matrix.row_offsets[row + 1] ||
           matrix.row_offsets[row + 1] > matrix.values.size()) {
            valid = false;
            continue;
        }
        AccurateResidualAccumulator accumulator;
        for(std::size_t entry = matrix.row_offsets[row];
            entry < matrix.row_offsets[row + 1]; ++entry) {
            std::size_t const column =
                static_cast<std::size_t>(columns[entry]);
            double value = 0;
            if(column < input.size())
                value = input[column];
            else {
                std::size_t const remote = column - input.size();
                if(remote >= remote_values.size()) {
                    valid = false;
                    continue;
                }
                value = remote_values[remote];
            }
            if(accurately)
                accumulator.AddProduct(matrix.values[entry], value);
            else
                output[row] += matrix.values[entry] * value;
        }
        if(accurately)
            output[row] = accumulator.Value();
        if(!std::isfinite(output[row]))
            valid = false;
    }
    return valid;
}

bool multiplyDistributedRows(
    DistributedActiveCSR const& matrix,
    std::vector<std::size_t> const& rows,
    std::vector<double> const& input,
    std::vector<double> const& remote_values,
    std::vector<double>& output,
    bool const accurately)
{
    if(matrix.uses_narrow_columns)
        return accurately ?
            multiplyDistributedRowsWithColumns<true>(
                matrix, matrix.columns32, rows, input, remote_values, output) :
            multiplyDistributedRowsWithColumns<false>(
                matrix, matrix.columns32, rows, input, remote_values, output);
    return accurately ?
        multiplyDistributedRowsWithColumns<true>(
            matrix, matrix.columns_wide, rows, input, remote_values, output) :
        multiplyDistributedRowsWithColumns<false>(
            matrix, matrix.columns_wide, rows, input, remote_values, output);
}

bool multiplyDistributed(
    DistributedActiveCSR const& matrix,
    DistributedActiveExchange& exchange,
    std::vector<double> const& input,
    std::vector<double>& output,
    ActiveBiCGSTABTiming& timing,
    bool const accurately = false)
{
    output.assign(input.size(), 0);
    bool valid = startRemoteActiveValues(exchange, input, timing);

    auto const multiply_local_rows = [&]()
    {
        auto const local_start = std::chrono::steady_clock::now();
        valid = multiplyDistributedRows(
            matrix, matrix.local_rows, input, exchange.remote_values, output,
            accurately) && valid;
        double const local_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - local_start).count();
        timing.local_matvec_seconds += local_seconds;
        timing.matvec_seconds += local_seconds;
    };
    if(exchange.overlap_local_rows)
        multiply_local_rows();

    valid = finishRemoteActiveValues(
        exchange, exchange.remote_values, timing) && valid;
    if(!exchange.overlap_local_rows)
        multiply_local_rows();

    auto const remote_start = std::chrono::steady_clock::now();
    valid = multiplyDistributedRows(
        matrix, matrix.remote_rows, input, exchange.remote_values, output,
        accurately) && valid;
    double const remote_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - remote_start).count();
    timing.remote_matvec_seconds += remote_seconds;
    timing.matvec_seconds += remote_seconds;

    auto const reduction_start = std::chrono::steady_clock::now();
    valid = collectiveAllTrue(valid);
    timing.reduction_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - reduction_start).count();
    ++timing.matvec_calls;
    ++timing.exchange_validity_reductions;
    return valid;
}

bool computeDistributedResidualAccurately(
    DistributedActiveCSR const& matrix,
    DistributedActiveExchange& exchange,
    std::vector<double> const& input,
    std::vector<double> const& rhs,
    std::vector<double> const& fixed_scale,
    std::vector<double> const* const scale_base,
    std::size_t const unknowns_per_cell,
    std::vector<double>& residual,
    TrueResidualAssessment& assessment,
    ActiveBiCGSTABTiming& timing)
{
    bool const exchange_valid =
        exchangeRemoteActiveValues(
            exchange, input, exchange.remote_values, timing);
    if(!exchange_valid)
        return false;
    if(scale_base != nullptr) {
        bool const scale_exchange_valid = exchangeRemoteActiveValues(
            exchange, *scale_base, exchange.auxiliary_remote_values, timing);
        if(!scale_exchange_valid)
            return false;
    }

    bool valid = matrix.RowCount() == input.size() &&
        matrix.row_offsets.size() == input.size() + 1 &&
        matrix.ColumnCount() == matrix.values.size() &&
        rhs.size() == input.size() &&
        fixed_scale.size() == input.size() &&
        (scale_base == nullptr || scale_base->size() == input.size()) &&
        unknowns_per_cell > 0;
    assessment.backward_error = 0;
    assessment.maximum_row_nonzeros = 0;
    assessment.representative_unknown = CG::max_size_t;
    assessment.representative_rank = 0;
    assessment.representative_residual =
        std::numeric_limits<double>::quiet_NaN();
    assessment.representative_scale =
        std::numeric_limits<double>::quiet_NaN();
    assessment.maximum_scale = 0;
    assessment.safe_minimum_scale = std::numeric_limits<double>::min();
    assessment.finite = true;
    residual.assign(input.size(), 0);
    std::vector<double> row_scales(input.size(), 0);
    std::vector<double> maximum_group_scales(unknowns_per_cell, 0);
    auto const accumulateRows = [&](std::vector<std::size_t> const& rows)
    {
        bool rows_valid = true;
        for(std::size_t const row : rows) {
            if(row >= input.size() || row >= rhs.size() ||
               row + 1 >= matrix.row_offsets.size() ||
               matrix.row_offsets[row] > matrix.row_offsets[row + 1] ||
               matrix.row_offsets[row + 1] > matrix.values.size()) {
                rows_valid = false;
                continue;
            }
            AccurateResidualAccumulator row_residual;
            AccurateResidualAccumulator row_scale;
            row_residual.Add(rhs[row]);
            row_scale.Add(fixed_scale[row]);
            for(std::size_t entry = matrix.row_offsets[row];
                entry < matrix.row_offsets[row + 1]; ++entry) {
                std::size_t const column = matrix.Column(entry);
                double value = 0;
                double base_value = 0;
                if(column < input.size())
                {
                    value = input[column];
                    if(scale_base != nullptr)
                        base_value = scale_base->at(column);
                }
                else {
                    std::size_t const remote = column - input.size();
                    if(remote >= exchange.remote_values.size()) {
                        rows_valid = false;
                        continue;
                    }
                    value = exchange.remote_values[remote];
                    if(scale_base != nullptr) {
                        if(remote >=
                           exchange.auxiliary_remote_values.size()) {
                            rows_valid = false;
                            continue;
                        }
                        base_value =
                            exchange.auxiliary_remote_values[remote];
                    }
                }
                row_residual.AddProduct(-matrix.values[entry], value);
                row_scale.AddProduct(std::abs(matrix.values[entry]),
                                     std::abs(base_value + value));
            }
            residual[row] = row_residual.Value();
            double const scale = row_scale.Value();
            row_scales[row] = scale;
            bool const finite = std::isfinite(residual[row]) &&
                std::isfinite(scale) && scale >= 0;
            assessment.finite = assessment.finite && finite;
            if(finite)
                maximum_group_scales[row % unknowns_per_cell] = std::max(
                    maximum_group_scales[row % unknowns_per_cell], scale);
            assessment.maximum_row_nonzeros = std::max(
                assessment.maximum_row_nonzeros,
                matrix.row_offsets[row + 1] - matrix.row_offsets[row]);
        }
        return rows_valid;
    };
    auto const local_start = std::chrono::steady_clock::now();
    valid = accumulateRows(matrix.local_rows) && valid;
    double const local_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - local_start).count();
    timing.local_matvec_seconds += local_seconds;
    timing.matvec_seconds += local_seconds;
    auto const remote_start = std::chrono::steady_clock::now();
    valid = accumulateRows(matrix.remote_rows) && valid;
    double const remote_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - remote_start).count();
    timing.remote_matvec_seconds += remote_seconds;
    timing.matvec_seconds += remote_seconds;
    ++timing.matvec_calls;
    auto const reduction_start = std::chrono::steady_clock::now();
    int rank = 0;
    requireDistributedMpiSuccess(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank(distributed true residual)");
    unsigned long long residual_summary[3] = {
        valid ? 0ULL : 1ULL,
        assessment.finite ? 0ULL : 1ULL,
        static_cast<unsigned long long>(assessment.maximum_row_nonzeros)};
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, residual_summary, 3,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(distributed residual summary)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, maximum_group_scales.data(),
                      static_cast<int>(maximum_group_scales.size()),
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(distributed group scales)");
    assessment.maximum_row_nonzeros =
        static_cast<std::size_t>(residual_summary[2]);
    assessment.backward_error = 0;
    assessment.representative_unknown = CG::max_size_t;
    for(std::size_t row = 0; row < residual.size(); ++row) {
        if(!std::isfinite(residual[row]) ||
           !std::isfinite(row_scales[row]))
            continue;
        double const maximum_scale =
            maximum_group_scales[row % unknowns_per_cell];
        double const safe_minimum_scale = std::max(
            std::numeric_limits<double>::min(),
            32 * std::numeric_limits<double>::epsilon() * maximum_scale);
        double const row_backward_error = std::abs(residual[row]) /
            std::max(row_scales[row], safe_minimum_scale);
        if(row_backward_error > assessment.backward_error ||
           assessment.representative_unknown == CG::max_size_t) {
            assessment.backward_error = row_backward_error;
            assessment.representative_unknown = row;
            assessment.representative_residual = residual[row];
            assessment.representative_scale = row_scales[row];
            assessment.maximum_scale = maximum_scale;
            assessment.safe_minimum_scale = safe_minimum_scale;
        }
    }
    struct BackwardErrorRank
    {
        double value;
        int rank;
    } maximum_backward_error = {
        input.empty() ? -1.0 : assessment.backward_error, rank};
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &maximum_backward_error, 1,
                      MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD),
        "MPI_Allreduce(distributed backward error)");
    assessment.backward_error = std::max(0.0, maximum_backward_error.value);
    assessment.representative_rank = maximum_backward_error.rank;
    unsigned long long representative_unknown = static_cast<unsigned long long>(
        rank == maximum_backward_error.rank ?
            assessment.representative_unknown : CG::max_size_t);
    double representative_values[4] = {
        assessment.representative_residual,
        assessment.representative_scale,
        assessment.maximum_scale,
        assessment.safe_minimum_scale};
    std::array<unsigned char,
               sizeof(representative_unknown) + sizeof(representative_values)>
        representative_payload{};
    if(rank == maximum_backward_error.rank) {
        std::memcpy(representative_payload.data(), &representative_unknown,
                    sizeof(representative_unknown));
        std::memcpy(
            representative_payload.data() + sizeof(representative_unknown),
            representative_values, sizeof(representative_values));
    }
    requireDistributedMpiSuccess(
        MPI_Bcast(representative_payload.data(),
                  static_cast<int>(representative_payload.size()), MPI_BYTE,
                  maximum_backward_error.rank, MPI_COMM_WORLD),
        "MPI_Bcast(distributed residual representative)");
    std::memcpy(&representative_unknown, representative_payload.data(),
                sizeof(representative_unknown));
    std::memcpy(
        representative_values,
        representative_payload.data() + sizeof(representative_unknown),
        sizeof(representative_values));
    assessment.representative_unknown =
        static_cast<std::size_t>(representative_unknown);
    assessment.representative_residual = representative_values[0];
    assessment.representative_scale = representative_values[1];
    assessment.maximum_scale = representative_values[2];
    assessment.safe_minimum_scale = representative_values[3];
    assessment.finite = residual_summary[1] == 0;
    timing.true_residual_reductions += 4;
    timing.reduction_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - reduction_start).count();
    return residual_summary[0] == 0;
}

double distributedDot(
    std::vector<double> const& left,
    std::vector<double> const& right,
    ActiveBiCGSTABTiming& timing)
{
    double local = dot(left, right);
    double global = 0;
    auto const reduction_start = std::chrono::steady_clock::now();
    requireDistributedMpiSuccess(
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(distributed dot product)");
    timing.reduction_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - reduction_start).count();
    ++timing.dot_reductions;
    return global;
}

std::pair<double, double> distributedDotPair(
    std::vector<double> const& shared_left,
    std::vector<double> const& first_right,
    std::vector<double> const& second_right,
    ActiveBiCGSTABTiming& timing,
    bool const paired_reduction)
{
    if(!paired_reduction)
        return std::make_pair(
            distributedDot(shared_left, first_right, timing),
            distributedDot(shared_left, second_right, timing));
    double local[2] = {
        dot(shared_left, first_right),
        dot(shared_left, second_right)};
    double global[2] = {0, 0};
    auto const reduction_start = std::chrono::steady_clock::now();
    requireDistributedMpiSuccess(
        MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
        "MPI_Allreduce(paired distributed dot product)");
    timing.reduction_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - reduction_start).count();
    ++timing.dot_reductions;
    return std::make_pair(global[0], global[1]);
}

#if 0
// Retained only as non-building experiment history. Production and regression
// paths use the same-cell block-Jacobi BiCGSTAB solver exclusively.
enum class DistributedGMRESOutcome
{
    Converged,
    Stagnated,
    Breakdown,
    Exhausted
};

struct DistributedGMRESResult
{
    DistributedGMRESOutcome outcome = DistributedGMRESOutcome::Breakdown;
    std::size_t iterations = 0;
    char const* reason = "uninitialized";
};

double distributedLongDoubleNorm(
    std::vector<double> const& values,
    ActiveBiCGSTABTiming& timing)
{
    long double local = 0;
    for(double const value : values) {
        long double const extended = static_cast<long double>(value);
        local += extended * extended;
    }
    long double global = 0;
    auto const reduction_start = std::chrono::steady_clock::now();
    requireDistributedMpiSuccess(
        MPI_Allreduce(&local, &global, 1, MPI_LONG_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(distributed long double norm)");
    timing.reduction_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - reduction_start).count();
    if(!(global >= 0) || !std::isfinite(global))
        return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(static_cast<double>(global));
}

DistributedGMRESResult recoverDistributedActiveWithGMRES(
    double const backward_tolerance,
    std::size_t const maximum_iterations,
    DistributedActiveCSR const& matrix,
    std::vector<double> const& correction_rhs,
    std::vector<double> const& verification_scale,
    std::vector<double> const& base_solution,
    std::vector<double>& solution,
    DistributedActiveExchange& exchange,
    std::size_t const unknowns_per_cell,
    CG::CellBlockJacobiPreconditioner& preconditioner,
    ActiveBiCGSTABTiming& timing)
{
    DistributedGMRESResult result;
    if(maximum_iterations == 0) {
        result.outcome = DistributedGMRESOutcome::Exhausted;
        result.reason = "fixed_iteration_budget";
        return result;
    }

    std::size_t const size = solution.size();
    std::size_t const restart_length = std::min<std::size_t>(
        32, std::max<std::size_t>(8, unknowns_per_cell));
    std::vector<std::vector<double> > basis(
        restart_length + 1, std::vector<double>(size, 0));
    std::vector<double> hessenberg((restart_length + 1) * restart_length, 0);
    std::vector<double> cosines(restart_length, 0);
    std::vector<double> sines(restart_length, 0);
    std::vector<double> projected_rhs(restart_length + 1, 0);
    std::vector<double> coefficients(restart_length, 0);
    std::vector<double> residual(size, 0), preconditioned(size, 0);
    std::vector<double> matrix_product(size, 0), work(size, 0);
    std::vector<double> candidate(size, 0);
    std::vector<double> restart_solution(size, 0);
    TrueResidualAssessment assessment;

    auto const recompute_true_residual = [&]()
    {
        return computeDistributedResidualAccurately(
            matrix, exchange, solution, correction_rhs,
            verification_scale, &base_solution, unknowns_per_cell,
            residual, assessment, timing) && assessment.finite;
    };
    if(!recompute_true_residual()) {
        result.reason = "initial_true_residual";
        return result;
    }
    if(assessment.backward_error <= backward_tolerance) {
        result.outcome = DistributedGMRESOutcome::Converged;
        result.reason = "initial_backward_error";
        return result;
    }

    int rank = 0;
    requireDistributedMpiSuccess(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank(distributed active solver)");
    if(rank == 0)
        std::clog << "MG_GMRES_RECOVERY_START scope=distributed_active"
                  << " side=left"
                  << " restart=" << restart_length
                  << " maximum_iterations=" << maximum_iterations
                  << " backward_error=" << assessment.backward_error
                  << " backward_tolerance=" << backward_tolerance
                  << std::endl;

    double stagnation_window_start = assessment.backward_error;
    std::size_t stagnant_restarts = 0;
    while(result.iterations < maximum_iterations) {
        preconditioner.Apply(residual, preconditioned);
        double const beta = distributedLongDoubleNorm(preconditioned, timing);
        if(!std::isfinite(beta) ||
           beta <= std::numeric_limits<double>::min()) {
            result.reason = "preconditioned_residual_norm";
            return result;
        }
        for(std::size_t i = 0; i < size; ++i)
            basis[0][i] = preconditioned[i] / beta;
        std::fill(hessenberg.begin(), hessenberg.end(), 0);
        std::fill(cosines.begin(), cosines.end(), 0);
        std::fill(sines.begin(), sines.end(), 0);
        std::fill(projected_rhs.begin(), projected_rhs.end(), 0);
        projected_rhs[0] = beta;

        std::size_t used = 0;
        bool arnoldi_valid = true;
        for(std::size_t column = 0;
            column < restart_length && result.iterations < maximum_iterations;
            ++column) {
            if(!multiplyDistributed(matrix, exchange, basis[column],
                                    matrix_product, timing, true)) {
                result.reason = "arnoldi_matvec";
                return result;
            }
            preconditioner.Apply(matrix_product, work);
            for(int pass = 0; pass < 2; ++pass) {
                std::vector<long double> local(column + 1, 0);
                std::vector<long double> global(column + 1, 0);
                for(std::size_t row = 0; row < size; ++row) {
                    long double const value =
                        static_cast<long double>(work[row]);
                    for(std::size_t vector = 0; vector <= column; ++vector)
                        local[vector] +=
                            static_cast<long double>(basis[vector][row]) * value;
                }
                auto const reduction_start = std::chrono::steady_clock::now();
                requireDistributedMpiSuccess(
                    MPI_Allreduce(local.data(), global.data(),
                                  static_cast<int>(column + 1),
                                  MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
                    "MPI_Allreduce(distributed Arnoldi orthogonalization)");
                timing.reduction_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - reduction_start).count();
                for(std::size_t vector = 0; vector <= column; ++vector) {
                    double const projection = static_cast<double>(global[vector]);
                    hessenberg[vector * restart_length + column] += projection;
                    for(std::size_t row = 0; row < size; ++row)
                        work[row] -= projection * basis[vector][row];
                }
            }
            double const next_norm = distributedLongDoubleNorm(work, timing);
            if(!std::isfinite(next_norm)) {
                result.reason = "arnoldi_norm";
                return result;
            }
            hessenberg[(column + 1) * restart_length + column] = next_norm;
            if(next_norm > std::numeric_limits<double>::min()) {
                for(std::size_t row = 0; row < size; ++row)
                    basis[column + 1][row] = work[row] / next_norm;
            }

            for(std::size_t rotation = 0; rotation < column; ++rotation) {
                double const upper =
                    hessenberg[rotation * restart_length + column];
                double const lower =
                    hessenberg[(rotation + 1) * restart_length + column];
                hessenberg[rotation * restart_length + column] =
                    cosines[rotation] * upper + sines[rotation] * lower;
                hessenberg[(rotation + 1) * restart_length + column] =
                    -sines[rotation] * upper + cosines[rotation] * lower;
            }
            double const diagonal =
                hessenberg[column * restart_length + column];
            double const subdiagonal =
                hessenberg[(column + 1) * restart_length + column];
            double const magnitude = std::hypot(diagonal, subdiagonal);
            if(!std::isfinite(magnitude) ||
               magnitude <= std::numeric_limits<double>::min()) {
                arnoldi_valid = false;
                result.reason = "givens_rotation";
                break;
            }
            cosines[column] = diagonal / magnitude;
            sines[column] = subdiagonal / magnitude;
            hessenberg[column * restart_length + column] = magnitude;
            hessenberg[(column + 1) * restart_length + column] = 0;
            double const rhs_value = projected_rhs[column];
            projected_rhs[column] = cosines[column] * rhs_value;
            projected_rhs[column + 1] = -sines[column] * rhs_value;
            used = column + 1;
            ++result.iterations;
            if(next_norm <= std::numeric_limits<double>::epsilon() * magnitude)
                break;
        }
        if(!arnoldi_valid || used == 0)
            return result;

        std::fill(coefficients.begin(), coefficients.end(), 0);
        for(std::size_t reverse = used; reverse > 0; --reverse) {
            std::size_t const row = reverse - 1;
            long double value = projected_rhs[row];
            for(std::size_t column = row + 1; column < used; ++column)
                value -= static_cast<long double>(
                    hessenberg[row * restart_length + column]) *
                    static_cast<long double>(coefficients[column]);
            double const diagonal =
                hessenberg[row * restart_length + row];
            if(!std::isfinite(diagonal) ||
               std::abs(diagonal) <= std::numeric_limits<double>::min()) {
                result.reason = "triangular_solve";
                return result;
            }
            coefficients[row] = static_cast<double>(value / diagonal);
        }
        restart_solution = solution;
        std::fill(candidate.begin(), candidate.end(), 0);
        for(std::size_t vector = 0; vector < used; ++vector)
            for(std::size_t row = 0; row < size; ++row)
                candidate[row] += coefficients[vector] * basis[vector][row];
        double const previous_backward_error = assessment.backward_error;
        double accepted_damping = 0;
        double damping = 1;
        for(std::size_t line_search = 0; line_search < 8; ++line_search) {
            for(std::size_t row = 0; row < size; ++row)
                solution[row] = restart_solution[row] +
                    damping * candidate[row];
            if(!recompute_true_residual()) {
                result.reason = "restart_true_residual";
                return result;
            }
            if(assessment.backward_error < previous_backward_error ||
               assessment.backward_error <= backward_tolerance) {
                accepted_damping = damping;
                break;
            }
            damping *= 0.5;
        }
        if(accepted_damping == 0) {
            solution = restart_solution;
            if(!recompute_true_residual()) {
                result.reason = "rollback_true_residual";
                return result;
            }
        }
        if(rank == 0)
            std::clog << "MG_GMRES_RECOVERY_PROGRESS scope=distributed_active"
                      << " iterations=" << result.iterations
                      << " accepted_damping=" << accepted_damping
                      << " backward_error=" << assessment.backward_error
                      << " backward_tolerance=" << backward_tolerance
                      << std::endl;
        if(assessment.backward_error <= backward_tolerance) {
            result.outcome = DistributedGMRESOutcome::Converged;
            result.reason = "true_backward_error";
            return result;
        }
        if(assessment.backward_error >= 0.9 * stagnation_window_start) {
            ++stagnant_restarts;
            if(stagnant_restarts >= 3) {
                result.outcome = DistributedGMRESOutcome::Stagnated;
                result.reason = "backward_error_stagnation";
                return result;
            }
        }
        else {
            stagnation_window_start = assessment.backward_error;
            stagnant_restarts = 0;
        }
    }
    result.outcome = DistributedGMRESOutcome::Exhausted;
    result.reason = "fixed_iteration_budget";
    return result;
}
#endif

bool solveDistributedActiveBiCGSTAB(
    double tolerance,
    int& total_iters,
    DistributedActiveCSR const& matrix,
    std::vector<double> const& rhs,
    std::vector<double> const& verification_rhs,
    std::vector<double> const& verification_scale,
    std::vector<double> const& base_solution,
    std::vector<double> const& final_correction_volume,
    double const fixed_cell_maximum_absolute_Eg,
    double const fixed_maximum_positive_cell_energy,
    double const fixed_positive_energy,
    std::vector<double>& solution,
    DistributedActiveExchange& exchange,
    std::size_t const unknowns_per_cell,
    CG::PreconditionerKind const preconditioner_kind,
    std::vector<std::size_t> const& block_cell_ids,
    CG::MatrixBuilder const& matrix_builder,
    CG::HistoricalMGResidualCorrectionDiagnostics& correction_diagnostics)
{
    auto const total_start = std::chrono::steady_clock::now();
    ActiveBiCGSTABTiming timing;
    std::size_t const size = rhs.size();
    int rank = 0;
    requireDistributedMpiSuccess(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank(distributed active solver)");
    correction_diagnostics =
        CG::HistoricalMGResidualCorrectionDiagnostics{};
    CG::CellBlockJacobiPreconditioner direction_preconditioner;
    bool preconditioner_ready = false;
    unsigned long long preconditioner_applications = 0;
    unsigned long long neighbor_correction_matvec_calls = 0;
    ActiveBiCGSTABMetrics last_metrics;
    double last_error = std::numeric_limits<double>::quiet_NaN();
    TrueResidualAssessment last_true_assessment;
    std::size_t last_true_eta_iteration = 0;
    double pre_correction_eta_inf =
        std::numeric_limits<double>::quiet_NaN();
    double backward_tolerance =
        std::numeric_limits<double>::quiet_NaN();
    std::size_t maximum_row_nonzeros = 0;
    CG::HistoricalMGPositivityContinuation positivity_continuation;
    CG::HistoricalMGBranch AcceptedConvergenceBranch =
        CG::HistoricalMGBranch::Continue;
    auto const finish = [&](bool const result,
                            char const* const outcome,
                            char const* const reason,
                            std::size_t const iterations,
                            double const error,
                            ActiveBiCGSTABMetrics const& metrics)
    {
        if(!result && positivity_continuation.Active &&
           correction_diagnostics.failure_reason.empty()) {
            std::string const rescue_reason =
                std::string("positivity_rescue_solver_") +
                (reason != nullptr ? reason : "unknown");
            CG::RecordHistoricalMGPositivityRescueFailure(
                positivity_continuation, correction_diagnostics, iterations,
                rescue_reason.c_str(),
                RadiationPositivity::SpectralRepairFailure::
                    PositivityRescueSolverFailure);
        }
        if(positivity_continuation.Active &&
           !positivity_continuation.Closed) {
            positivity_continuation.AdditionalIterationsUsed = iterations >=
                positivity_continuation.InitialIteration ?
                iterations - positivity_continuation.InitialIteration : 0;
            positivity_continuation.BlocksCompleted = std::min(
                CG::historical_mg_positivity_continuation_maximum_blocks,
                (positivity_continuation.AdditionalIterationsUsed +
                 CG::historical_mg_positivity_continuation_block_iterations -
                 1) /
                    CG::historical_mg_positivity_continuation_block_iterations);
            CG::ReportHistoricalMGPositivityContinuationClose(
                "distributed_active", positivity_continuation,
                positivity_continuation.LastNegativity, "solver_failure",
                rank == 0);
            CG::RecordHistoricalMGPositivityContinuation(
                positivity_continuation, correction_diagnostics);
        }
        total_iters = static_cast<int>(std::min<std::size_t>(
            iterations, static_cast<std::size_t>(
                std::numeric_limits<int>::max())));
        bool const detailed_report = exchange.rank_profile_enabled || !result;
        if(detailed_report)
            ReportDistributedActiveBiCGSTAB(
                "MG_BICGSTAB_CONVERGENCE", outcome, reason, iterations,
                error, metrics, size, unknowns_per_cell, block_cell_ids,
                timing.reduction_seconds,
                last_true_assessment.backward_error, backward_tolerance,
                maximum_row_nonzeros,
                last_true_assessment.representative_unknown,
                last_true_assessment.representative_rank,
                last_true_assessment.representative_residual,
                last_true_assessment.representative_scale,
                last_true_assessment.maximum_scale,
                last_true_assessment.safe_minimum_scale,
                &timing.report_identity_reductions);
        if(preconditioner_ready && detailed_report)
        {
            timing.preconditioner_apply_seconds =
                direction_preconditioner.ApplySeconds();
            unsigned long long calls = static_cast<unsigned long long>(
                direction_preconditioner.ApplyCalls());
            unsigned long long applications = preconditioner_applications;
            unsigned long long neighbor_matvec_calls =
                neighbor_correction_matvec_calls;
            double seconds = direction_preconditioner.ApplySeconds();
            auto const reduction_start = std::chrono::steady_clock::now();
            requireDistributedMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &calls, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(preconditioner apply calls)");
            requireDistributedMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &seconds, 1, MPI_DOUBLE,
                              MPI_MAX, MPI_COMM_WORLD),
                "MPI_Allreduce(preconditioner apply time)");
            requireDistributedMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &applications, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(preconditioner applications)");
            requireDistributedMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &neighbor_matvec_calls, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(neighbor correction matvec calls)");
            timing.reduction_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - reduction_start).count();
            if(rank == 0)
                std::clog << "MG_PRECONDITIONER_APPLY kind="
                          << CG::PreconditionerKindLabel(
                                 direction_preconditioner.Kind())
                          << " scope=distributed_active calls_max=" << calls
                          << " applications_max=" << applications
                          << " neighbor_correction_matvec_calls_max="
                          << neighbor_matvec_calls
                          << " seconds_max=" << seconds << std::endl;
        }
        if(detailed_report) {
        double const total_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - total_start).count();
        double values[12] = {
            timing.matvec_seconds,
            timing.local_matvec_seconds,
            timing.remote_matvec_seconds,
            timing.exchange_seconds,
            timing.exchange_pack_seconds,
            timing.exchange_start_seconds,
            timing.exchange_wait_seconds,
            timing.exchange_unpack_seconds,
            timing.reduction_seconds,
            timing.preconditioner_setup_seconds,
            timing.preconditioner_apply_seconds,
            total_seconds};
        unsigned long long const csr_storage_bytes =
            static_cast<unsigned long long>(matrix.row_offsets.capacity()) *
                sizeof(std::size_t) +
            matrix.ColumnStorageBytes() +
            static_cast<unsigned long long>(matrix.values.capacity()) *
                sizeof(double) +
            static_cast<unsigned long long>(matrix.local_rows.capacity()) *
                sizeof(std::size_t) +
            static_cast<unsigned long long>(matrix.remote_rows.capacity()) *
                sizeof(std::size_t);
        unsigned long long const exchange_storage_bytes =
            static_cast<unsigned long long>(exchange.send_offsets.capacity()) * sizeof(std::size_t) +
            static_cast<unsigned long long>(exchange.receive_offsets.capacity()) * sizeof(std::size_t) +
            static_cast<unsigned long long>(exchange.send_local_indices.capacity()) * sizeof(std::size_t) +
            static_cast<unsigned long long>(exchange.receive_remote_slots.capacity()) * sizeof(std::size_t) +
            static_cast<unsigned long long>(exchange.send_values.capacity()) * sizeof(double) +
            static_cast<unsigned long long>(exchange.receive_values.capacity()) * sizeof(double) +
            static_cast<unsigned long long>(exchange.remote_values.capacity()) * sizeof(double) +
            static_cast<unsigned long long>(exchange.auxiliary_remote_values.capacity()) * sizeof(double) +
            static_cast<unsigned long long>(exchange.send_transfers.capacity()) *
                sizeof(DistributedActivePeerTransfer) +
            static_cast<unsigned long long>(exchange.receive_transfers.capacity()) *
                sizeof(DistributedActivePeerTransfer) +
            static_cast<unsigned long long>(exchange.requests.capacity()) *
                sizeof(MPI_Request);
        unsigned long long counts[12] = {
            timing.matvec_calls,
            timing.exchange_calls,
            timing.exchange_validity_reductions,
            timing.dot_reductions,
            static_cast<unsigned long long>(matrix.local_rows.size()),
            static_cast<unsigned long long>(matrix.remote_rows.size()),
            static_cast<unsigned long long>(matrix.values.size()),
            csr_storage_bytes,
            exchange_storage_bytes,
            static_cast<unsigned long long>(exchange.remote_size),
            timing.true_residual_reductions,
            timing.report_identity_reductions};
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, values, 12, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(distributed solver timing)");
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, counts, 12,
                          MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD),
            "MPI_Allreduce(distributed solver counters)");
        if(rank == 0)
            std::clog << "MG_BICGSTAB_HISTORICAL_POLICY"
                      << " scope=distributed_active"
                      << " squared_scaled_tolerance=" << tolerance
                      << " effective_norm_tolerance="
                      << std::sqrt(tolerance)
                      << " last_true_eta_inf="
                      << last_true_assessment.backward_error
                      << " last_true_eta_iteration="
                      << last_true_eta_iteration
                      << " last_true_eta_age="
                      << (iterations >= last_true_eta_iteration ?
                          iterations - last_true_eta_iteration : 0)
                      << " pre_correction_eta_inf="
                      << pre_correction_eta_inf
                      << " final_eta_inf=not_evaluated"
                      << " eta_inf_role=diagnostic_only" << std::endl;
        if(rank == 0)
            std::clog << "MG_BICGSTAB_TIMING scope=distributed_active outcome="
                      << outcome << " iterations=" << iterations
                      << " overlap="
                      << (exchange.overlap_local_rows ? 1 : 0)
                      << " paired_omega="
                      << (exchange.pair_omega_reduction ? 1 : 0)
                      << " matvec_seconds_max=" << values[0]
                      << " local_matvec_seconds_max=" << values[1]
                      << " remote_matvec_seconds_max=" << values[2]
                      << " exchange_seconds_max=" << values[3]
                      << " exchange_pack_seconds_max=" << values[4]
                      << " exchange_start_seconds_max=" << values[5]
                      << " exchange_wait_seconds_max=" << values[6]
                      << " exchange_unpack_seconds_max=" << values[7]
                      << " reduction_seconds_max=" << values[8]
                      << " preconditioner_setup_seconds_max=" << values[9]
                      << " preconditioner_apply_seconds_max=" << values[10]
                      << " total_seconds_max=" << values[11]
                      << " matvec_calls_max=" << counts[0]
                      << " exchange_calls_max=" << counts[1]
                      << " exchange_validity_reductions_max=" << counts[2]
                      << " dot_reductions_max=" << counts[3]
                      << " local_rows_max=" << counts[4]
                      << " remote_rows_max=" << counts[5]
                      << " csr_nonzeros_max=" << counts[6]
                      << " csr_storage_bytes_max=" << counts[7]
                      << " csr_column_slot_bits_rank0="
                      << (matrix.uses_narrow_columns ? 32 :
                          8 * sizeof(CG::matrix_index_t))
                      << " exchange_storage_bytes_max=" << counts[8]
                      << " remote_unknowns_max=" << counts[9]
                      << " true_residual_reductions_max=" << counts[10]
                      << " report_identity_reductions_max=" << counts[11]
                      << std::endl;
        if(exchange.rank_profile_enabled) {
            reportDistributedActiveRankProfile(
                timing, matrix, exchange,
                static_cast<unsigned long long>(
                    direction_preconditioner.StorageBytes()));
            reportDistributedMemoryPhase("solve_complete");
        }
        }
        direction_preconditioner.Release();
        return result;
    };

    bool valid = matrix.RowCount() == size &&
                 matrix.row_offsets.size() == size + 1 &&
                 matrix.ColumnCount() == matrix.values.size() &&
                 verification_rhs.size() == size &&
                  verification_scale.size() == size &&
                  base_solution.size() == size &&
                  final_correction_volume.size() == size &&
                  solution.size() == size &&
                 exchange.local_size == size;
    std::vector<double> inverse_diagonal(size, 0);
    std::vector<double> diagonal(size, 0);
    for(std::size_t row = 0; row < size; ++row) {
        double row_diagonal = 0;
        std::size_t row_nonzeros = 0;
        if(row + 1 >= matrix.row_offsets.size() ||
           matrix.row_offsets[row] > matrix.row_offsets[row + 1] ||
           matrix.row_offsets[row + 1] > matrix.values.size()) {
            valid = false;
            continue;
        }
        for(std::size_t entry = matrix.row_offsets[row];
            entry < matrix.row_offsets[row + 1]; ++entry)
            if(matrix.Column(entry) != CG::max_size_t) {
                ++row_nonzeros;
                if(matrix.Column(entry) == row)
                    row_diagonal += matrix.values[entry];
            }
        maximum_row_nonzeros = std::max(maximum_row_nonzeros,
                                        row_nonzeros);
        if(!std::isfinite(row_diagonal) || row_diagonal <= 0)
            valid = false;
        else {
            diagonal[row] = row_diagonal;
            inverse_diagonal[row] = 1.0 / row_diagonal;
        }
    }
    auto reduction_start = std::chrono::steady_clock::now();
    bool const globally_valid = collectiveAllTrue(valid);
    timing.reduction_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - reduction_start).count();
    if(!globally_valid)
        return finish(false, "breakdown", "invalid_diagonal", 0,
                      last_error, last_metrics);
    unsigned long long global_row_nonzeros =
        static_cast<unsigned long long>(maximum_row_nonzeros);
    reduction_start = std::chrono::steady_clock::now();
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &global_row_nonzeros, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(distributed solver row nonzeros)");
    timing.reduction_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - reduction_start).count();
    maximum_row_nonzeros = static_cast<std::size_t>(global_row_nonzeros);
    backward_tolerance = verifiedBackwardErrorTolerance(
        tolerance, maximum_row_nonzeros);

    valid = matrix.uses_narrow_columns ?
        direction_preconditioner.SetupCSR(
            matrix.row_offsets, matrix.columns32, matrix.values,
            unknowns_per_cell, preconditioner_kind,
            inverse_diagonal) :
        direction_preconditioner.SetupCSR(
            matrix.row_offsets, matrix.columns_wide, matrix.values,
            unknowns_per_cell, preconditioner_kind,
            inverse_diagonal);
    timing.preconditioner_setup_seconds =
        direction_preconditioner.SetupSeconds();
    reduction_start = std::chrono::steady_clock::now();
    int setup_state = (valid ? 0 : 1) |
        (direction_preconditioner.FallbackBlockCount() > 0 ? 2 : 0);
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &setup_state, 1, MPI_INT, MPI_BOR,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(distributed preconditioner setup state)");
    bool const setup_valid = (setup_state & 1) == 0;
    timing.reduction_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - reduction_start).count();
    if(!setup_valid)
        return finish(false, "breakdown", "preconditioner_setup", 0,
                      last_error, last_metrics);
    preconditioner_ready = true;
    if(exchange.rank_profile_enabled || (setup_state & 2) != 0) {
        auto const setup_report_start = std::chrono::steady_clock::now();
        ReportDistributedPreconditionerSetup(direction_preconditioner,
                                             block_cell_ids);
        timing.reduction_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - setup_report_start).count();
    }
    if(exchange.rank_profile_enabled)
        reportDistributedMemoryPhase("preconditioner_setup");
    if(exchange.global_size == 0) {
        last_true_assessment.backward_error = 0;
        last_true_assessment.maximum_row_nonzeros = maximum_row_nonzeros;
        last_true_assessment.finite = true;
        total_iters = 0;
        return finish(true, "converged", "empty_system", 0,
                      0, last_metrics);
    }

    if(exchange.rank_profile_enabled && rank == 0)
        std::clog << "MG_BICGSTAB_TOLERANCE scope=distributed_active"
                  << " requested=" << tolerance
                  << " squared_scaled_tolerance=" << tolerance
                  << " effective_norm_tolerance=" << std::sqrt(tolerance)
                  << " diagnostic_backward_reference="
                  << backward_tolerance
                  << " global_unknowns=" << exchange.global_size
                  << " criterion=historical_diagonal_scaled_squared_residual"
                  << " eta_inf_role=diagnostic_only"
                  << " max_global_row_nnz=" << maximum_row_nonzeros
                  << std::endl;

    std::fill(solution.begin(), solution.end(), 0);
    std::vector<double> physical_solution(size, 0);
    std::vector<double> previous_physical_solution = base_solution;
    auto const refresh_physical_solution = [&]()
    {
        for(std::size_t i = 0; i < size; ++i)
            physical_solution[i] = base_solution[i] + solution[i];
    };
    auto const measure_current = [&](std::vector<double> const& values)
    {
        refresh_physical_solution();
        return CG::MeasureHistoricalMG(
            physical_solution, previous_physical_solution, values,
            verification_rhs, diagonal, unknowns_per_cell,
            &timing.reduction_seconds);
    };
    auto const measure_update = [&](std::vector<double> const& values,
                                    std::vector<double> const& first_update,
                                    double const first_scale,
                                    std::vector<double> const& second_update,
                                    double const second_scale)
    {
        refresh_physical_solution();
        previous_physical_solution = physical_solution;
        for(std::size_t i = 0; i < previous_physical_solution.size(); ++i) {
            if(i < first_update.size())
                previous_physical_solution[i] -=
                    first_scale * first_update[i];
            if(i < second_update.size())
                previous_physical_solution[i] -=
                    second_scale * second_update[i];
        }
        return CG::MeasureHistoricalMG(
            physical_solution, previous_physical_solution, values,
            verification_rhs, diagonal, unknowns_per_cell,
            &timing.reduction_seconds);
    };

    std::vector<double> product;
    if(!multiplyDistributed(matrix, exchange, solution, product,
                            timing))
        return finish(false, "breakdown", "initial_matvec", 0,
                      last_error, last_metrics);
    std::vector<double> residual(size), shadow(size), direction(size, 0),
        preconditioned(size), matrix_direction(size), intermediate(size),
        preconditioned_intermediate(size), matrix_intermediate(size),
        scaled_values(size), preconditioner_residual(size);
    auto apply_direction_preconditioner =
        [&](std::vector<double> const& input,
            std::vector<double>& output,
            std::vector<double>& matrix_output)
    {
        direction_preconditioner.Apply(input, output);
        ++preconditioner_applications;
        std::size_t const block_sweeps = CG::CellBlockJacobiSweepCount(
            direction_preconditioner.Kind());
        for(std::size_t sweep = 1; sweep < block_sweeps; ++sweep) {
            if(!multiplyDistributed(matrix, exchange, output,
                                    matrix_output, timing))
                return false;
            CG::ApplyCellBlockJacobiCorrectionSweep(
                direction_preconditioner, input, matrix_output, output,
                preconditioner_residual);
            ++neighbor_correction_matvec_calls;
        }
        return true;
    };
    std::vector<double> const no_update;
    for(std::size_t i = 0; i < size; ++i)
        residual[i] = rhs[i] - product[i];
    shadow = residual;

    auto const scaled_norm = [&](std::vector<double> const& values)
    {
        for(std::size_t i = 0; i < size; ++i)
            scaled_values[i] = inverse_diagonal[i] * values[i];
        return std::sqrt(distributedDot(scaled_values, scaled_values,
                                        timing));
    };
    std::vector<double> sampled_true_residual;
    std::size_t requested_true_eta_iteration = 0;
    auto const recompute_true_residual = [&]()
    {
        refresh_physical_solution();
        bool const valid = computeDistributedResidualAccurately(
            matrix, exchange, physical_solution,
            verification_rhs, verification_scale, nullptr,
            unknowns_per_cell, sampled_true_residual,
            last_true_assessment, timing);
        if(!valid)
            return false;
        last_true_eta_iteration = requested_true_eta_iteration;
        return last_true_assessment.finite &&
            std::isfinite(scaled_norm(sampled_true_residual));
    };
    if(!recompute_true_residual())
        return finish(false, "breakdown", "true_initial_residual_matvec",
                      0, last_error, last_metrics);
    // The distributed true residual exchanges canonical owner values.  At
    // delta=0 it is the correction RHS and must seed the Krylov recurrence.
    residual = sampled_true_residual;
    shadow = residual;
    last_metrics = measure_current(residual);
    last_error = last_metrics.historical_error;

    auto const attempt_historical_final_correction =
        [&](std::size_t const iterations)
    {
        pre_correction_eta_inf = last_true_assessment.backward_error;
        bool local_valid = last_true_eta_iteration == iterations &&
            last_true_assessment.finite &&
            sampled_true_residual.size() == size &&
            final_correction_volume.size() == size;
        ActiveFinalCorrectionCandidate candidate;
        if(local_valid) {
            candidate = BuildActiveFinalCorrectionCandidate(
                base_solution, solution, sampled_true_residual,
                final_correction_volume, unknowns_per_cell,
                block_cell_ids);
            local_valid = candidate.assessment.finite;
        }
        if(!collectiveAllTrue(local_valid))
            return CG::HistoricalMGCorrectionDisposition::RejectNonFinite;
        CG::CollectHistoricalMGResidualCorrectionLimitingDiagnostic(
            candidate.correction_diagnostics);
        if(!AssessActiveFinalCorrectionSpectralFailure(
               base_solution, solution, final_correction_volume,
               unknowns_per_cell, block_cell_ids,
               fixed_cell_maximum_absolute_Eg,
               fixed_maximum_positive_cell_energy, fixed_positive_energy,
               true, candidate))
            return CG::HistoricalMGCorrectionDisposition::RejectNonFinite;
        if(!collectiveAllTrue(candidate.assessment.finite))
            return CG::HistoricalMGCorrectionDisposition::RejectNonFinite;
        correction_diagnostics = candidate.correction_diagnostics;
        CG::HistoricalMGPositivityContinuationDecision const
            continuation_decision =
                CG::EvaluateHistoricalMGPositivityContinuation(
                    candidate.corrected_negativity, candidate.positive_floor,
                    iterations,
                    positivity_continuation);
        if(continuation_decision ==
           CG::HistoricalMGPositivityContinuationDecision::Restart)
            CG::ReportHistoricalMGPositivityContinuationOpen(
                "distributed_active", positivity_continuation, rank == 0);
        else if(continuation_decision ==
                CG::HistoricalMGPositivityContinuationDecision::Cleared)
            CG::ReportHistoricalMGPositivityContinuationClose(
                "distributed_active", positivity_continuation,
                candidate.corrected_negativity, "cleared", rank == 0);
        else if(continuation_decision ==
                CG::HistoricalMGPositivityContinuationDecision::Exhausted)
            CG::ReportHistoricalMGPositivityContinuationClose(
                "distributed_active", positivity_continuation,
                candidate.corrected_negativity, "budget_exhausted",
                rank == 0);
        CG::RecordHistoricalMGPositivityContinuation(
            positivity_continuation, correction_diagnostics);
        if(continuation_decision ==
           CG::HistoricalMGPositivityContinuationDecision::Restart)
            return CG::HistoricalMGCorrectionDisposition::DeferPositivity;
        if(continuation_decision ==
           CG::HistoricalMGPositivityContinuationDecision::Continue)
            return CG::HistoricalMGCorrectionDisposition::ContinuePositivity;
        if(CG::ShouldCommitHistoricalMGComptonFallback(
               candidate.corrected_negativity, continuation_decision,
               matrix_builder.HistoricalMGComptonFallbackAvailable(
                   candidate.corrected_negativity.CellId))) {
            correction_diagnostics.compton_fallback_candidate = true;
            CG::RecordHistoricalMGPositiveFloor(
                CG::HistoricalMGPositiveFloorAssessment{},
                correction_diagnostics);
            solution = std::move(candidate.capped_corrected_delta);
            return CG::HistoricalMGCorrectionDisposition::Commit;
        }
        if(candidate.spectral_failure.CausedRejection) {
            CG::RecordHistoricalMGResidualCorrectionFailure(
                candidate.spectral_failure, correction_diagnostics,
                iterations, &positivity_continuation);
            return CG::HistoricalMGCorrectionDisposition::
                RejectPostCapNonphysical;
        }
        solution = std::move(candidate.corrected_delta);
        if(candidate.positive_floor.Applied) {
            requested_true_eta_iteration = iterations;
            correction_diagnostics.post_floor_true_residual_evaluated = true;
            if(!recompute_true_residual()) {
                correction_diagnostics.post_floor_true_residual_finite = false;
                correction_diagnostics.failure_reason =
                    "post_floor_true_residual_nonfinite";
                correction_diagnostics.failure_class =
                    RadiationPositivity::SpectralRepairFailure::
                        NonfiniteRepairedExtent;
                return CG::HistoricalMGCorrectionDisposition::RejectNonFinite;
            }
            ActiveBiCGSTABMetrics const post_floor_metrics =
                measure_current(sampled_true_residual);
            correction_diagnostics.post_floor_true_residual_finite =
                post_floor_metrics.finite;
            correction_diagnostics.post_floor_true_residual_error =
                post_floor_metrics.historical_error;
            if(!post_floor_metrics.finite) {
                correction_diagnostics.failure_reason =
                    "post_floor_true_residual_nonfinite";
                correction_diagnostics.failure_class =
                    RadiationPositivity::SpectralRepairFailure::
                        NonfiniteRepairedExtent;
                return CG::HistoricalMGCorrectionDisposition::RejectNonFinite;
            }
        }
        return CG::HistoricalMGCorrectionDisposition::Commit;
    };
    double rho_previous = 1;
    double alpha = 1;
    double omega = 1;
    auto const restart_from_true_residual = [&]()
    {
        residual = sampled_true_residual;
        shadow = residual;
        std::fill(direction.begin(), direction.end(), 0);
        std::fill(matrix_direction.begin(), matrix_direction.end(), 0);
        rho_previous = 1;
        alpha = 1;
        omega = 1;
    };
    auto const finish_breakdown =
        [&](CG::HistoricalMGBreakdown const breakdown,
            std::size_t const zero_based_iteration)
    {
        std::size_t const iterations = zero_based_iteration + 1;
        requested_true_eta_iteration = iterations;
        if(!recompute_true_residual()) {
            finish(false, "rejected", "diagnostic_eta_nonfinite",
                   iterations, last_error, last_metrics);
            return ActiveBreakdownResolution::Rejected;
        }
        last_metrics = measure_current(sampled_true_residual);
        last_error = last_metrics.historical_error;
        if(CG::ShouldRestartHistoricalMGFiniteBreakdown(
               breakdown, last_metrics, iterations)) {
            restart_from_true_residual();
            return ActiveBreakdownResolution::Restart;
        }
        CG::HistoricalMGDecision const decision =
            CG::ClassifyHistoricalMG(last_metrics, zero_based_iteration,
                                     tolerance, breakdown,
                                     breakdown !=
                                         CG::HistoricalMGBreakdown::NonFinite);
        if(decision.accept) {
            CG::HistoricalMGCorrectionDisposition const disposition =
                attempt_historical_final_correction(iterations);
            if(disposition ==
                   CG::HistoricalMGCorrectionDisposition::DeferPositivity ||
               disposition == CG::HistoricalMGCorrectionDisposition::
                   ContinuePositivity) {
                restart_from_true_residual();
                return ActiveBreakdownResolution::Restart;
            }
            if(disposition ==
               CG::HistoricalMGCorrectionDisposition::Commit) {
                finish(true, "converged",
                       CG::HistoricalMGBranchLabel(decision.branch),
                       iterations, last_error, last_metrics);
                return ActiveBreakdownResolution::Converged;
            }
            finish(false, "rejected",
                   HistoricalCorrectionRejectionReason(disposition), iterations,
                   last_error, last_metrics);
            return ActiveBreakdownResolution::Rejected;
        }
        finish(false, "rejected",
               CG::HistoricalMGBranchLabel(decision.branch), iterations,
               last_error, last_metrics);
        return ActiveBreakdownResolution::Rejected;
    };
    std::size_t constexpr maximum_iterations =
        CG::historical_mg_maximum_iterations;
    for(std::size_t iteration = 0;
        iteration < maximum_iterations +
            CG::historical_mg_positivity_continuation_iteration_budget;
        ++iteration) {
        if(iteration >= maximum_iterations &&
           !positivity_continuation.Active)
            break;
        double const rho = distributedDot(shadow, residual, timing);
        if(!std::isfinite(rho)) {
            ActiveBreakdownResolution const resolution = finish_breakdown(
                CG::HistoricalMGBreakdown::NonFinite, iteration);
            if(resolution == ActiveBreakdownResolution::Restart)
                continue;
            return resolution == ActiveBreakdownResolution::Converged;
        }
        if(std::abs(rho) <= std::numeric_limits<double>::min() * 1e100) {
            ActiveBreakdownResolution const resolution = finish_breakdown(
                CG::HistoricalMGBreakdown::TinyRho, iteration);
            if(resolution == ActiveBreakdownResolution::Restart)
                continue;
            return resolution == ActiveBreakdownResolution::Converged;
        }
        double const beta = (rho / rho_previous) * (alpha / omega);
        if(!std::isfinite(beta)) {
            ActiveBreakdownResolution const resolution = finish_breakdown(
                CG::HistoricalMGBreakdown::NonFinite, iteration);
            if(resolution == ActiveBreakdownResolution::Restart)
                continue;
            return resolution == ActiveBreakdownResolution::Converged;
        }
        for(std::size_t i = 0; i < size; ++i)
            direction[i] = residual[i] +
                beta * (direction[i] - omega * matrix_direction[i]);
        if(!apply_direction_preconditioner(direction, preconditioned,
                                           matrix_direction))
            return finish(false, "breakdown",
                          "direction_preconditioner_neighbor_matvec",
                          iteration + 1, last_error, last_metrics);
        if(!multiplyDistributed(matrix, exchange, preconditioned,
                                matrix_direction, timing))
            return finish(false, "breakdown", "direction_matvec",
                          iteration + 1, last_error, last_metrics);
        double const denominator = distributedDot(shadow, matrix_direction,
                                                   timing);
        if(!std::isfinite(denominator)) {
            ActiveBreakdownResolution const resolution = finish_breakdown(
                CG::HistoricalMGBreakdown::NonFinite, iteration);
            if(resolution == ActiveBreakdownResolution::Restart)
                continue;
            return resolution == ActiveBreakdownResolution::Converged;
        }
        alpha = std::abs(denominator) <=
            std::numeric_limits<double>::min() * 1e100 ?
            0.0 : rho / denominator;
        for(std::size_t i = 0; i < size; ++i) {
            intermediate[i] = residual[i] - alpha * matrix_direction[i];
            solution[i] += alpha * preconditioned[i];
        }
        last_metrics = measure_update(
            intermediate, preconditioned, alpha, no_update, 0);
        last_error = last_metrics.historical_error;
        if(!last_metrics.finite)
            return finish(false, "rejected",
                          CG::HistoricalMGBranchLabel(
                              CG::HistoricalMGBranch::RejectNonFinite),
                          iteration + 1, last_error, last_metrics);
        CG::HistoricalMGDecision intermediate_decision =
            CG::ClassifyHistoricalMGCorrectionCandidate(
                last_metrics, iteration, tolerance);
        if(intermediate_decision.reject)
            return finish(false, "rejected",
                          CG::HistoricalMGBranchLabel(
                              intermediate_decision.branch),
                          iteration + 1, last_error, last_metrics);
        if(intermediate_decision.accept) {
            requested_true_eta_iteration = iteration + 1;
            if(!recompute_true_residual())
                return finish(false, "rejected", "diagnostic_eta_nonfinite",
                              iteration + 1, last_error, last_metrics);
            last_metrics = measure_current(sampled_true_residual);
            last_error = last_metrics.historical_error;
            intermediate_decision =
                CG::ClassifyHistoricalMGCorrectionCandidate(
                last_metrics, iteration, tolerance);
            if(intermediate_decision.reject)
                return finish(false, "rejected",
                              CG::HistoricalMGBranchLabel(
                                  intermediate_decision.branch),
                              iteration + 1, last_error, last_metrics);
            if(intermediate_decision.accept) {
                CG::HistoricalMGCorrectionDisposition const disposition =
                    attempt_historical_final_correction(iteration + 1);
                if(disposition ==
                   CG::HistoricalMGCorrectionDisposition::DeferPositivity) {
                    restart_from_true_residual();
                    continue;
                }
                if(disposition ==
                       CG::HistoricalMGCorrectionDisposition::RejectNonFinite ||
                   disposition == CG::HistoricalMGCorrectionDisposition::
                       RejectPostCapNonphysical)
                    return finish(false, "rejected",
                                  HistoricalCorrectionRejectionReason(
                                      disposition),
                                  iteration + 1, last_error, last_metrics);
                if(disposition ==
                   CG::HistoricalMGCorrectionDisposition::Commit)
                    return finish(true, "converged",
                                  CG::HistoricalMGBranchLabel(
                                      intermediate_decision.branch),
                                  iteration + 1, last_error, last_metrics);
            }
        }
        if(!apply_direction_preconditioner(intermediate,
                                           preconditioned_intermediate,
                                           matrix_intermediate))
            return finish(false, "breakdown",
                          "intermediate_preconditioner_neighbor_matvec",
                          iteration + 1, last_error, last_metrics);
        if(!multiplyDistributed(matrix, exchange,
                                preconditioned_intermediate,
                                matrix_intermediate, timing))
            return finish(false, "breakdown", "intermediate_matvec",
                          iteration + 1, last_error, last_metrics);
        std::pair<double, double> const omega_products =
            distributedDotPair(matrix_intermediate, matrix_intermediate,
                               intermediate, timing,
                               exchange.pair_omega_reduction);
        double const omega_denominator = omega_products.first;
        if(!std::isfinite(omega_denominator)) {
            ActiveBreakdownResolution const resolution = finish_breakdown(
                CG::HistoricalMGBreakdown::NonFinite, iteration);
            if(resolution == ActiveBreakdownResolution::Restart)
                continue;
            return resolution == ActiveBreakdownResolution::Converged;
        }
        omega = omega_denominator <=
            std::numeric_limits<double>::min() * 1e100 ? 0.0 :
            omega_products.second / omega_denominator;
        if(!std::isfinite(omega)) {
            ActiveBreakdownResolution const resolution = finish_breakdown(
                CG::HistoricalMGBreakdown::NonFinite, iteration);
            if(resolution == ActiveBreakdownResolution::Restart)
                continue;
            return resolution == ActiveBreakdownResolution::Converged;
        }
        if(std::abs(alpha) < std::numeric_limits<double>::min() * 1e100 &&
           std::abs(omega) < std::numeric_limits<double>::min() * 1e100) {
            ActiveBreakdownResolution const resolution = finish_breakdown(
                CG::HistoricalMGBreakdown::TinyAlphaOmega, iteration);
            if(resolution == ActiveBreakdownResolution::Restart)
                continue;
            return resolution == ActiveBreakdownResolution::Converged;
        }
        if(std::abs(omega) <= std::numeric_limits<double>::min()) {
            ActiveBreakdownResolution const resolution = finish_breakdown(
                CG::HistoricalMGBreakdown::TinyAlphaOmega, iteration);
            if(resolution == ActiveBreakdownResolution::Restart)
                continue;
            return resolution == ActiveBreakdownResolution::Converged;
        }
        for(std::size_t i = 0; i < size; ++i) {
            solution[i] += omega * preconditioned_intermediate[i];
            residual[i] = intermediate[i] - omega * matrix_intermediate[i];
        }
        bool const report_progress = exchange.rank_profile_enabled &&
            (iteration == 0 || (iteration + 1) % 100 == 0);
        last_metrics = measure_update(
            residual, preconditioned, alpha,
            preconditioned_intermediate, omega);
        last_error = last_metrics.historical_error;
        if(!last_metrics.finite)
            return finish(false, "rejected",
                          CG::HistoricalMGBranchLabel(
                              CG::HistoricalMGBranch::RejectNonFinite),
                          iteration + 1, last_error, last_metrics);

        CG::HistoricalMGDecision decision =
            CG::ClassifyHistoricalMGCorrectionCandidate(
                last_metrics, iteration, tolerance);
        if(decision.reject)
            return finish(false, "rejected",
                          CG::HistoricalMGBranchLabel(decision.branch),
                          iteration + 1, last_error, last_metrics);
        bool const positivity_block_boundary =
            CG::HistoricalMGPositivityContinuationBlockBoundaryReached(
                positivity_continuation, iteration + 1);
        bool const positivity_budget_reached =
            CG::HistoricalMGPositivityContinuationBudgetReached(
                positivity_continuation, iteration + 1);
        if((iteration + 1) % 50 == 0 || decision.accept ||
           positivity_block_boundary) {
            requested_true_eta_iteration = iteration + 1;
            if(!recompute_true_residual())
                return finish(false, "rejected",
                              "diagnostic_eta_nonfinite", iteration + 1,
                              last_error, last_metrics);
            last_metrics = measure_current(sampled_true_residual);
            last_error = last_metrics.historical_error;
            decision = CG::ClassifyHistoricalMGCorrectionCandidate(
                last_metrics, iteration, tolerance);
            if(decision.reject)
                return finish(false, "rejected",
                              CG::HistoricalMGBranchLabel(decision.branch),
                              iteration + 1, last_error, last_metrics);
        }
        if(CG::ShouldRestartHistoricalMGPositivityContinuation(
               decision.accept, positivity_block_boundary,
               positivity_budget_reached)) {
            positivity_continuation.AdditionalIterationsUsed = iteration + 1 -
                positivity_continuation.InitialIteration;
            positivity_continuation.BlocksCompleted = std::min(
                CG::historical_mg_positivity_continuation_maximum_blocks,
                positivity_continuation.AdditionalIterationsUsed /
                    CG::historical_mg_positivity_continuation_block_iterations);
            positivity_continuation.BlocksStarted = std::min(
                CG::historical_mg_positivity_continuation_maximum_blocks,
                positivity_continuation.BlocksCompleted + 1);
            restart_from_true_residual();
            continue;
        }
        if(decision.accept && !positivity_continuation.Active)
            AcceptedConvergenceBranch = decision.branch;
        if(CG::ShouldAttemptHistoricalMGPositivityFinalization(
               decision.accept, positivity_block_boundary,
               positivity_budget_reached)) {
            CG::HistoricalMGCorrectionDisposition const disposition =
                attempt_historical_final_correction(iteration + 1);
            if(disposition ==
               CG::HistoricalMGCorrectionDisposition::DeferPositivity) {
                restart_from_true_residual();
                continue;
            }
            if(disposition ==
                   CG::HistoricalMGCorrectionDisposition::RejectNonFinite ||
               disposition == CG::HistoricalMGCorrectionDisposition::
                   RejectPostCapNonphysical)
                return finish(false, "rejected",
                              HistoricalCorrectionRejectionReason(disposition),
                              iteration + 1, last_error, last_metrics);
            if(disposition ==
               CG::HistoricalMGCorrectionDisposition::Commit) {
                CG::HistoricalMGBranch const CompletionBranch =
                    decision.accept ? decision.branch :
                    AcceptedConvergenceBranch;
                return finish(true, "converged",
                              CG::HistoricalMGBranchLabel(CompletionBranch),
                              iteration + 1, last_error, last_metrics);
            }
        }
        if(report_progress)
            ReportDistributedActiveBiCGSTAB(
                "MG_BICGSTAB_PROGRESS", nullptr, nullptr, iteration + 1,
                last_error, last_metrics, size, unknowns_per_cell,
                block_cell_ids, timing.reduction_seconds,
                last_true_assessment.backward_error,
                backward_tolerance, maximum_row_nonzeros,
                last_true_assessment.representative_unknown,
                last_true_assessment.representative_rank,
                last_true_assessment.representative_residual,
                last_true_assessment.representative_scale,
                last_true_assessment.maximum_scale,
                last_true_assessment.safe_minimum_scale,
                &timing.report_identity_reductions);
        rho_previous = rho;
    }
    total_iters = static_cast<int>(maximum_iterations);
    return finish(false, "not_converged", "maximum_iterations",
                  maximum_iterations, last_error, last_metrics);
}
#endif

class IndividualContextGuard
{
public:
    IndividualContextGuard(IndividualStepContext const*& context_target,
                           double& fraction_target,
                           IndividualStepContext const& context,
                           double interval_fraction)
        : context_target_(context_target), fraction_target_(fraction_target)
    {
        context_target_ = &context;
        fraction_target_ = interval_fraction;
    }

    ~IndividualContextGuard()
    {
        context_target_ = nullptr;
        fraction_target_ = 1.0;
    }

private:
    IndividualStepContext const*& context_target_;
    double& fraction_target_;
};

struct OwnedCanonicalMappingAssessment
{
    bool valid = false;
    bool identity = false;
};

OwnedCanonicalMappingAssessment assessOwnedCanonicalMapping(
    std::vector<ComputationalCell3D> const& owned_cells,
    std::size_t const owned_count,
    std::vector<ComputationalCell3D> const* canonical_cells,
    std::vector<std::size_t> const* owned_to_canonical)
{
    OwnedCanonicalMappingAssessment result;
    if(canonical_cells == nullptr || owned_cells.size() < owned_count ||
       canonical_cells->size() != owned_count ||
       (owned_to_canonical != nullptr &&
        owned_to_canonical->size() != owned_count))
        return result;

    result.identity = true;
    for(std::size_t owned = 0; owned < owned_count; ++owned) {
        std::size_t const canonical = owned_to_canonical == nullptr ?
            owned : owned_to_canonical->at(owned);
        if(canonical >= owned_count ||
           owned_cells[owned].ID != canonical_cells->at(canonical).ID) {
            result.identity = false;
            return result;
        }
        result.identity = result.identity && canonical == owned;
    }
    result.valid = true;
    return result;
}

bool hasCompleteOwnedActivity(IndividualStepContext const& context,
                              std::size_t const owned_count)
{
    if(context.active_indices.size() != owned_count ||
       context.active_mask.size() < owned_count)
        return false;
    try {
        std::vector<unsigned char> seen(owned_count, 0);
        for(std::size_t const cell : context.active_indices) {
            if(cell >= owned_count || seen[cell] != 0 ||
               !context.isActive(cell))
                return false;
            seen[cell] = 1;
        }
        for(std::size_t cell = 0; cell < owned_count; ++cell)
            if(seen[cell] == 0 || !context.isActive(cell))
                return false;
    }
    catch(...) {
        return false;
    }
    return true;
}

bool schedulerEventInterval(IndividualStepContext const& context,
                            double& interval)
{
    interval = std::numeric_limits<double>::quiet_NaN();
    if(context.event_tick <= context.previous_event_tick ||
       !std::isfinite(context.time_quantum) || context.time_quantum <= 0)
        return false;
    interval = context.time_quantum * static_cast<double>(
        context.event_tick - context.previous_event_tick);
    return std::isfinite(interval) && interval > 0;
}

struct MappedAllActiveRuntimeOption
{
    bool enabled = true;
    bool valid = true;
};

struct AllActiveInnerSnapshotRuntimeOption
{
    bool enabled = false;
    bool valid = true;
};

AllActiveInnerSnapshotRuntimeOption const&
allActiveInnerSnapshotRuntimeOption()
{
    // This launch-time option controls a transaction boundary, so every rank
    // must agree before any rank may omit its inner rollback snapshot.  Cache
    // the collective result to avoid adding reductions to later events.
    static AllActiveInnerSnapshotRuntimeOption const option = []()
    {
        AllActiveInnerSnapshotRuntimeOption result;
        bool locally_valid = true;
        bool const locally_enabled = environmentToggle(
            "RICH_MG_ELIDE_INNER_ALL_ACTIVE_SNAPSHOT", false,
            locally_valid);
#ifdef RICH_MPI
        int const local_state = !locally_valid ? 2 :
            (locally_enabled ? 1 : 0);
        int minimum_state = local_state;
        int maximum_state = local_state;
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &minimum_state, 1, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(minimum inner snapshot option)");
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &maximum_state, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(maximum inner snapshot option)");
        result.valid = minimum_state == maximum_state &&
            minimum_state >= 0 && maximum_state <= 1;
        result.enabled = result.valid && minimum_state == 1;
#else
        result.valid = locally_valid;
        result.enabled = locally_valid && locally_enabled;
#endif
        return result;
    }();
    return option;
}

MappedAllActiveRuntimeOption const& mappedAllActiveRuntimeOption()
{
    static MappedAllActiveRuntimeOption const option = []()
    {
        MappedAllActiveRuntimeOption result;
        result.enabled = environmentToggle(
            "RICH_INDIVIDUAL_MAPPED_ALL_ACTIVE", true, result.valid);
        return result;
    }();
    return option;
}

} // namespace

RadiationDriver::IndividualRadiationLocalDefectMeasure
RadiationDriver::measureIndividualRadiationLocalDefect(
    double const withdrawal,
    double const passive_extent,
    double const roundoff_floor,
    double const normalization_scale)
{
    IndividualRadiationDefectConfiguration const& configuration =
        individualRadiationDefectConfiguration();
    double const relative_scale =
        std::max(0.0, passive_extent) + roundoff_floor;
    IndividualRadiationLocalDefectMeasure result;
    result.relative_fraction = withdrawal / relative_scale;
    result.allowed_withdrawal =
        configuration.local_withdrawal_limit * relative_scale +
        configuration.local_absolute_limit * normalization_scale;
    result.tolerance_ratio = withdrawal / result.allowed_withdrawal;
    return result;
}

#ifdef RICH_MPI
RadiationDriverTestHooks::OwnedCanonicalMappingProbeResult
RadiationDriverTestHooks::ProbeOwnedCanonicalMapping(
    std::vector<ComputationalCell3D> const& owned_cells,
    std::vector<ComputationalCell3D> const& canonical_cells,
    std::vector<std::size_t> const& owned_to_canonical)
{
    OwnedCanonicalMappingAssessment const assessment =
        assessOwnedCanonicalMapping(owned_cells, owned_cells.size(),
                                    &canonical_cells,
                                    &owned_to_canonical);
    OwnedCanonicalMappingProbeResult result;
    result.valid = assessment.valid;
    result.identity = assessment.identity;
    return result;
}

bool RadiationDriverTestHooks::ProbeCompleteOwnedActivity(
    IndividualStepContext const& context, std::size_t const owned_count)
{
    return hasCompleteOwnedActivity(context, owned_count);
}

RadiationDriverTestHooks::SchedulerEventIntervalProbeResult
RadiationDriverTestHooks::ProbeSchedulerEventInterval(
    IndividualStepContext const& context, double const cell_interval)
{
    SchedulerEventIntervalProbeResult result;
    result.valid = schedulerEventInterval(context, result.interval);
    result.matches_cell_interval = result.valid &&
        result.interval == cell_interval;
    return result;
}

RadiationDriverTestHooks::DistributedActiveMatVecProbeResult
RadiationDriverTestHooks::ProbeDistributedActiveMatVec(
    std::vector<ComputationalCell3D> const& owned_active_cells,
    std::vector<double> const& local_values,
    std::vector<int> const& remote_owners,
    std::vector<std::size_t> const& remote_cell_ids,
    std::vector<std::vector<double> > const& matrix,
    std::vector<std::vector<std::size_t> > const& columns,
    std::size_t request_chunk_limit,
    int request_tag)
{
    DistributedActiveMatVecProbeResult result;
    DistributedActiveRuntimeOptions runtime_options;
    if(!loadDistributedActiveRuntimeOptions(runtime_options))
        return result;
    bool valid = owned_active_cells.size() == local_values.size() &&
        remote_owners.size() == remote_cell_ids.size() &&
        matrix.size() == local_values.size() &&
        columns.size() == local_values.size();
    std::map<std::size_t, std::size_t> owned_active_cell_bases;
    for(std::size_t local = 0; local < owned_active_cells.size(); ++local)
        valid = owned_active_cell_bases.emplace(
            owned_active_cells[local].ID, local).second && valid;

    DistributedActiveCSR csr;
    csr.row_offsets.assign(local_values.size() + 1, 0);
    for(std::size_t row = 0; row < local_values.size(); ++row) {
        csr.row_offsets[row] = csr.values.size();
        bool remote_row = false;
        if(matrix[row].size() != columns[row].size()) {
            valid = false;
            csr.row_offsets[row + 1] = csr.values.size();
            csr.local_rows.push_back(row);
            continue;
        }
        for(std::size_t entry = 0; entry < matrix[row].size(); ++entry) {
            csr.values.push_back(matrix[row][entry]);
            csr.PushColumn(columns[row][entry]);
            remote_row = remote_row || columns[row][entry] >=
                local_values.size();
        }
        csr.row_offsets[row + 1] = csr.values.size();
        if(remote_row)
            csr.remote_rows.push_back(row);
        else
            csr.local_rows.push_back(row);
    }

    std::vector<RemoteUnknownKey> remote_unknowns(remote_cell_ids.size());
    for(std::size_t remote = 0; remote < remote_cell_ids.size(); ++remote) {
        remote_unknowns[remote].owner = remote_owners[remote];
        remote_unknowns[remote].cell_id = remote_cell_ids[remote];
        remote_unknowns[remote].group = 0;
    }
    if(!collectiveAllTrue(valid))
        return result;

    DistributedActiveExchange exchange;
    exchange.overlap_local_rows = runtime_options.overlap_local_rows;
    exchange.pair_omega_reduction = runtime_options.pair_omega_reduction;
    exchange.rank_profile_enabled = runtime_options.profile;
    IndividualActiveRequestExchangeStats request_stats;
    if(request_chunk_limit == 0)
        request_chunk_limit =
            static_cast<std::size_t>(std::numeric_limits<int>::max()) / 2;
    result.exchange_initialized = initializeDistributedExchange(
        remote_unknowns, owned_active_cell_bases, 1,
        local_values.size(), exchange, request_chunk_limit, &request_stats,
        request_tag);
    result.request_send_chunks = request_stats.send_chunks;
    result.request_receive_chunks = request_stats.receive_chunks;
    result.global_size = exchange.global_size;
    result.csr_nonzeros = csr.values.size();
    result.csr_uses_narrow_columns = csr.uses_narrow_columns;
    result.local_row_count = csr.local_rows.size();
    result.remote_row_count = csr.remote_rows.size();
    if(!result.exchange_initialized)
        return result;

    ActiveBiCGSTABTiming timing;
    double const* const send_data = exchange.send_values.data();
    double const* const receive_data = exchange.receive_values.data();
    double const* const remote_data = exchange.remote_values.data();
    result.multiplied = multiplyDistributed(
        csr, exchange, local_values, result.output, timing, true);
    std::vector<double> second_values = local_values;
    for(double& value : second_values)
        value += 1;
    result.second_multiply_succeeded = multiplyDistributed(
        csr, exchange, second_values, result.second_output, timing, true);
    result.exchange_storage_reused =
        send_data == exchange.send_values.data() &&
        receive_data == exchange.receive_values.data() &&
        remote_data == exchange.remote_values.data();
    result.exchange_validity_reductions =
        timing.exchange_validity_reductions;
    if(runtime_options.profile) {
        reportDistributedActiveRankProfile(timing, csr, exchange, 0);
        reportDistributedMemoryPhase("focused_probe_complete");
    }
    return result;
}

RadiationDriverTestHooks::DistributedCSRColumnProbeResult
RadiationDriverTestHooks::ProbeDistributedCSRColumns(
    std::vector<std::size_t> const& columns)
{
    DistributedActiveCSR csr;
    for(std::size_t const column : columns)
        csr.PushColumn(column);
    DistributedCSRColumnProbeResult result;
    result.uses_narrow_columns = csr.uses_narrow_columns;
    result.storage_bytes = csr.ColumnStorageBytes();
    result.columns.reserve(csr.ColumnCount());
    for(std::size_t entry = 0; entry < csr.ColumnCount(); ++entry)
        result.columns.push_back(csr.Column(entry));
    return result;
}

RadiationDriverTestHooks::DistributedActiveTransferChunkProbeResult
RadiationDriverTestHooks::ProbeDistributedActiveTransferChunks(
    std::size_t const total_count)
{
    DistributedActiveTransferChunkProbeResult result;
    std::vector<DistributedActivePeerTransfer> transfers;
    result.valid = appendDistributedActiveTransferRange(
        7, 0, total_count, transfers);
    result.offsets.reserve(transfers.size());
    result.counts.reserve(transfers.size());
    for(DistributedActivePeerTransfer const& transfer : transfers) {
        result.valid = transfer.peer == 7 && transfer.count > 0 &&
            result.valid;
        result.offsets.push_back(transfer.offset);
        result.counts.push_back(static_cast<std::size_t>(transfer.count));
    }
    return result;
}

RadiationDriverTestHooks::DistributedActiveSizeArithmeticProbeResult
RadiationDriverTestHooks::ProbeDistributedActiveSizeArithmetic(
    std::size_t const left, std::size_t const right)
{
    DistributedActiveSizeArithmeticProbeResult result;
    result.add_valid = checkedDistributedSizeAdd(left, right, result.sum);
    result.multiply_valid = checkedDistributedSizeMultiply(
        left, right, result.product);
    return result;
}

RadiationDriverTestHooks::DistributedActiveToggleProbeResult
RadiationDriverTestHooks::ProbeDistributedActiveToggle(
    std::string const& setting, bool const fallback)
{
    DistributedActiveToggleProbeResult result;
    result.value = fallback;
    result.valid = parseDistributedActiveToggle(setting, result.value);
    return result;
}

bool RadiationDriverTestHooks::ProbeDistributedActiveOverlapDefault()
{
    return distributed_active_overlap_default;
}

[[noreturn]] void
RadiationDriverTestHooks::TriggerDistributedActiveFailStopForTest()
{
    abortDistributedActiveFailure(
        "focused injected communication failure", MPI_ERR_OTHER,
        "rank-local fail-stop regression");
}

[[noreturn]] void
RadiationDriverTestHooks::TriggerDistributedActiveReturnedMpiFailureForTest()
{
    requireDistributedMpiSuccess(
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN),
        "MPI_Comm_set_errhandler(focused returned-error regression)");

    int rank = 0;
    int ranks = 1;
    requireDistributedMpiSuccess(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank(focused returned-error regression)");
    requireDistributedMpiSuccess(
        MPI_Comm_size(MPI_COMM_WORLD, &ranks),
        "MPI_Comm_size(focused returned-error regression)");
    if(ranks < 2)
        abortDistributedActiveFailure(
            "focused returned-error regression requires two ranks",
            MPI_ERR_OTHER);

    if(rank == 0) {
        double pending_value = 0;
        MPI_Request pending_request = MPI_REQUEST_NULL;
        requireDistributedMpiSuccess(
            MPI_Irecv(&pending_value, 1, MPI_DOUBLE, 1, 9173,
                      MPI_COMM_WORLD, &pending_request),
            "MPI_Irecv(focused partial post)");

        MPI_Request invalid_request = MPI_REQUEST_NULL;
        int const returned_error = MPI_Irecv(
            &pending_value, -1, MPI_DOUBLE, 1, 9174, MPI_COMM_WORLD,
            &invalid_request);
        if(returned_error == MPI_SUCCESS)
            abortDistributedActiveFailure(
                "MPI_Irecv(focused returned-error injection)", MPI_ERR_OTHER,
                "negative count unexpectedly returned MPI_SUCCESS");
        requireDistributedMpiSuccess(
            returned_error,
            "MPI_Irecv(focused returned-error injection after partial post)");
    }

    requireDistributedMpiSuccess(
        MPI_Barrier(MPI_COMM_WORLD),
        "MPI_Barrier(focused returned-error peer wait)");
    std::abort();
}

[[noreturn]] void RadiationDriverTestHooks::
TriggerDistributedActiveReturnedCollectiveFailureForTest()
{
    requireDistributedMpiSuccess(
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN),
        "MPI_Comm_set_errhandler(focused collective-error regression)");

    int rank = 0;
    requireDistributedMpiSuccess(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank(focused collective-error regression)");
    int value = rank;
    int const returned_error = MPI_Allreduce(
        MPI_IN_PLACE, &value, -1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(rank == 0) {
        if(returned_error == MPI_SUCCESS)
            abortDistributedActiveFailure(
                "MPI_Allreduce(focused returned collective injection)",
                MPI_ERR_OTHER,
                "negative count unexpectedly returned MPI_SUCCESS");
        requireDistributedMpiSuccess(
            returned_error,
            "MPI_Allreduce(focused branch collective returned error)");
    }

    requireDistributedMpiSuccess(
        MPI_Barrier(MPI_COMM_WORLD),
        "MPI_Barrier(focused collective-error peer wait)");
    std::abort();
}
#endif

void RadiationDriver::commitResidualCorrectionAccounting(
    CG::HistoricalMGResidualCorrectionDiagnostics const& local) const
{
    unsigned long long limited_groups =
        static_cast<unsigned long long>(local.limited_group_count);
    unsigned long long group_count =
        static_cast<unsigned long long>(local.group_count);
    double signed_bias = local.signed_energy_bias;
    double absolute_bias = local.absolute_energy_bias;
    double minimum_scale = local.limited_group_count > 0
        ? local.minimum_scale : 1.0;
    int collective_candidate =
        local.positive_floor_collected_globally ? 1 : 0;
    unsigned long long positivity_rescue_events =
        local.positivity_blocks_started > 0 ? 1 : 0;
    unsigned long long positivity_rescue_blocks =
        static_cast<unsigned long long>(local.positivity_blocks_started);
    unsigned long long positivity_rescue_iterations =
        static_cast<unsigned long long>(
            local.positivity_additional_iterations);
    unsigned long long positive_floor_events =
        local.positive_floor_applied ? 1 : 0;
    unsigned long long positive_floor_cells =
        static_cast<unsigned long long>(local.positive_floor_local_cells);
    unsigned long long positive_floor_groups =
        static_cast<unsigned long long>(local.positive_floor_local_groups);
    double positive_floor_injected_energy =
        local.positive_floor_local_injected_energy;
    double positive_floor_global_energy =
        local.positive_floor_global_positive_energy;
    double positive_floor_cell_ratio =
        local.positive_floor_maximum_cell_injection_ratio;
    double positive_floor_global_ratio =
        local.positive_floor_global_injection_ratio;
    double positive_floor_post_residual =
        local.post_floor_true_residual_evaluated &&
        local.post_floor_true_residual_finite ?
        local.post_floor_true_residual_error : 0;
#ifdef RICH_MPI
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &collective_candidate, 1, MPI_INT,
                      MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(repair collective candidate)");
    MPI_Op const event_operation = collective_candidate != 0 ?
        MPI_MAX : MPI_SUM;
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &positivity_rescue_events, 1,
                      MPI_UNSIGNED_LONG_LONG, event_operation,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(positivity rescue events)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &positivity_rescue_blocks, 1,
                      MPI_UNSIGNED_LONG_LONG, event_operation,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(positivity rescue blocks)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &positivity_rescue_iterations, 1,
                      MPI_UNSIGNED_LONG_LONG, event_operation,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(positivity rescue iterations)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &positive_floor_events, 1,
                      MPI_UNSIGNED_LONG_LONG, event_operation,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(positive floor events)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &positive_floor_cells, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
        "MPI_Allreduce(positive floor cells)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &positive_floor_groups, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
        "MPI_Allreduce(positive floor groups)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &positive_floor_injected_energy, 1,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
        "MPI_Allreduce(positive floor injected energy)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &positive_floor_global_energy, 1,
                      MPI_DOUBLE,
                      collective_candidate != 0 ? MPI_MAX : MPI_SUM,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(positive floor global energy)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &positive_floor_cell_ratio, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(positive floor cell ratio)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &positive_floor_global_ratio, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(positive floor global ratio)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &positive_floor_post_residual, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(positive floor post residual)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &limited_groups, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
        "MPI_Allreduce(limited correction groups)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &group_count, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(repair group count)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &signed_bias, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(signed correction bias)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &absolute_bias, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(absolute correction bias)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &minimum_scale, 1, MPI_DOUBLE, MPI_MIN,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(minimum correction scale)");
#endif
    (void)collective_candidate;

    std::vector<double> signed_by_group(
        static_cast<std::size_t>(group_count), 0.0);
    std::vector<double> absolute_by_group(
        static_cast<std::size_t>(group_count), 0.0);
    std::copy_n(local.signed_energy_bias_by_group.begin(),
                std::min(signed_by_group.size(),
                         local.signed_energy_bias_by_group.size()),
                signed_by_group.begin());
    std::copy_n(local.absolute_energy_bias_by_group.begin(),
                std::min(absolute_by_group.size(),
                         local.absolute_energy_bias_by_group.size()),
                absolute_by_group.begin());
#ifdef RICH_MPI
    if(!signed_by_group.empty()) {
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, signed_by_group.data(),
                          static_cast<int>(signed_by_group.size()), MPI_DOUBLE,
                          MPI_SUM, MPI_COMM_WORLD),
            "MPI_Allreduce(signed correction bias by group)");
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, absolute_by_group.data(),
                          static_cast<int>(absolute_by_group.size()),
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
            "MPI_Allreduce(absolute correction bias by group)");
    }
#endif

    RadiationRepairAccounting* accounting = &standalone_repair_accounting_;
    if(individual_context_ != nullptr &&
       individual_context_->radiation_repair_accounting != nullptr)
        accounting = individual_context_->radiation_repair_accounting;
    accounting->residual_correction_limited_groups +=
        static_cast<std::uint64_t>(limited_groups);
    accounting->residual_correction_signed_energy_bias += signed_bias;
    accounting->residual_correction_absolute_energy_bias += absolute_bias;
    if(accounting->residual_correction_signed_bias_by_group.size() <
       signed_by_group.size())
        accounting->residual_correction_signed_bias_by_group.resize(
            signed_by_group.size(), 0.0);
    if(accounting->residual_correction_absolute_bias_by_group.size() <
       absolute_by_group.size())
        accounting->residual_correction_absolute_bias_by_group.resize(
            absolute_by_group.size(), 0.0);
    for(std::size_t group = 0; group < signed_by_group.size(); ++group) {
        accounting->residual_correction_signed_bias_by_group[group] +=
            signed_by_group[group];
        accounting->residual_correction_absolute_bias_by_group[group] +=
            absolute_by_group[group];
    }
    if(limited_groups > 0)
        accounting->residual_correction_minimum_scale = std::min(
            accounting->residual_correction_minimum_scale, minimum_scale);
    accounting->positivity_rescue_events +=
        static_cast<std::uint64_t>(positivity_rescue_events);
    accounting->positivity_rescue_blocks +=
        static_cast<std::uint64_t>(positivity_rescue_blocks);
    accounting->positivity_rescue_additional_iterations +=
        static_cast<std::uint64_t>(positivity_rescue_iterations);
    accounting->residual_positive_floor_events +=
        static_cast<std::uint64_t>(positive_floor_events);
    accounting->residual_positive_floor_cells +=
        static_cast<std::uint64_t>(positive_floor_cells);
    accounting->residual_positive_floor_groups +=
        static_cast<std::uint64_t>(positive_floor_groups);
    accounting->residual_positive_floor_cumulative_injected_energy +=
        positive_floor_injected_energy;
    accounting->residual_positive_floor_maximum_cell_injection_ratio =
        std::max(
            accounting->residual_positive_floor_maximum_cell_injection_ratio,
            positive_floor_cell_ratio);
    accounting->residual_positive_floor_maximum_global_injection_ratio =
        std::max(
            accounting->residual_positive_floor_maximum_global_injection_ratio,
            positive_floor_global_ratio);
    accounting->residual_positive_floor_maximum_post_true_residual_error =
        std::max(
            accounting->
                residual_positive_floor_maximum_post_true_residual_error,
            positive_floor_post_residual);
    if(accounting->residual_positive_floor_initial_global_radiation_energy ==
           0 &&
       std::isfinite(positive_floor_global_energy) &&
       positive_floor_global_energy > 0)
        accounting->residual_positive_floor_initial_global_radiation_energy =
            positive_floor_global_energy;
}

void RadiationDriver::commitSpectralRepairAccounting(
    SpectralRepairEvent const& local_event,
    char const* const scope) const
{
    unsigned long long repaired_cells =
        static_cast<unsigned long long>(local_event.repaired_cells);
    unsigned long long repaired_groups =
        static_cast<unsigned long long>(local_event.repaired_groups);
    double injected_energy = local_event.injected_energy;
    double global_radiation_energy = local_event.owned_radiation_energy;
    double maximum_relative_deficit = local_event.maximum_relative_deficit;
    int rank = 0;
    int representative_rank = 0;
#ifdef RICH_MPI
    requireDistributedMpiSuccess(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank(positive floor accounting)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &repaired_cells, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
        "MPI_Allreduce(positive floor repaired cells)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &repaired_groups, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
        "MPI_Allreduce(positive floor repaired groups)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &injected_energy, 1,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
        "MPI_Allreduce(positive floor injected energy)");
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &global_radiation_energy, 1,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
        "MPI_Allreduce(positive floor global radiation energy)");
    struct {
        double value;
        int rank;
    } local_pick{
        local_event.repaired_cells > 0
            ? local_event.maximum_relative_deficit : -1.0,
        rank}, global_pick{-1.0, 0};
    requireDistributedMpiSuccess(
        MPI_Allreduce(&local_pick, &global_pick, 1, MPI_DOUBLE_INT,
                      MPI_MAXLOC, MPI_COMM_WORLD),
        "MPI_Allreduce(positive floor representative)");
    maximum_relative_deficit = std::max(0.0, global_pick.value);
    representative_rank = global_pick.rank;
#endif

    unsigned long long representative_identity[2] = {
        static_cast<unsigned long long>(local_event.representative_cell_id),
        static_cast<unsigned long long>(local_event.representative_group)};
    double representative_details[3] = {
        local_event.representative_original_extent,
        local_event.representative_floor_extent,
        local_event.representative_injected_extent};
#ifdef RICH_MPI
    if(repaired_cells > 0) {
        requireDistributedMpiSuccess(
            MPI_Bcast(representative_identity, 2, MPI_UNSIGNED_LONG_LONG,
                      representative_rank, MPI_COMM_WORLD),
            "MPI_Bcast(positive floor representative identity)");
        requireDistributedMpiSuccess(
            MPI_Bcast(representative_details, 3, MPI_DOUBLE,
                      representative_rank, MPI_COMM_WORLD),
            "MPI_Bcast(positive floor representative details)");
    }
#endif

    RadiationRepairAccounting* accounting = &standalone_repair_accounting_;
    if(individual_context_ != nullptr &&
       individual_context_->radiation_repair_accounting != nullptr)
        accounting = individual_context_->radiation_repair_accounting;

    if(std::isfinite(global_radiation_energy) &&
       global_radiation_energy >= 0)
        accounting->maximum_global_radiation_energy = std::max(
            accounting->maximum_global_radiation_energy,
            global_radiation_energy);
    if(repaired_cells == 0)
        return;

    accounting->repaired_cells +=
        static_cast<std::uint64_t>(repaired_cells);
    accounting->repaired_groups +=
        static_cast<std::uint64_t>(repaired_groups);
    accounting->cumulative_injected_energy += injected_energy;
    if(maximum_relative_deficit >= accounting->maximum_relative_deficit) {
        accounting->maximum_relative_deficit = maximum_relative_deficit;
        accounting->representative_cell_id =
            static_cast<std::uint64_t>(representative_identity[0]);
        accounting->representative_group =
            static_cast<std::uint64_t>(representative_identity[1]);
        accounting->representative_rank = representative_rank;
        accounting->representative_original_extent =
            representative_details[0];
        accounting->representative_floor_extent =
            representative_details[1];
        accounting->representative_injected_extent =
            representative_details[2];
    }

    double const injection_fraction =
        accounting->maximum_global_radiation_energy > 0
        ? accounting->cumulative_injected_energy /
              accounting->maximum_global_radiation_energy
        : 0;
    bool const warn = AdvanceRadiationRepairWarning(
        *accounting, injection_fraction);

    if(rank == 0) {
        std::clog << std::setprecision(17)
                  << "MG_SPECTRAL_POSITIVITY_REPAIR"
                  << " scope=" << scope
                  << " cells=" << repaired_cells
                  << " groups=" << repaired_groups
                  << " injected_extent_sum=" << injected_energy
                  << " max_relative_deficit="
                  << maximum_relative_deficit
                  << " representative_rank=" << representative_rank
                  << " representative_cell_id="
                  << representative_identity[0]
                  << " representative_negative_group="
                  << representative_identity[1]
                  << " representative_original_extent="
                  << representative_details[0]
                  << " representative_floor_extent="
                  << representative_details[1]
                  << " representative_injected_extent="
                  << representative_details[2]
                  << " cumulative_cells=" << accounting->repaired_cells
                  << " cumulative_groups=" << accounting->repaired_groups
                  << " cumulative_injected_extent="
                  << accounting->cumulative_injected_energy
                  << " maximum_global_radiation_energy="
                  << accounting->maximum_global_radiation_energy
                  << " cumulative_injection_fraction="
                  << injection_fraction << std::endl;
        if(warn)
            std::clog << std::setprecision(17)
                      << "MG_SPECTRAL_POSITIVITY_REPAIR_WARNING"
                      << " scope=" << scope
                      << " cumulative_injection_fraction="
                      << injection_fraction
                      << " next_warning_fraction="
                      << accounting->next_warning_fraction
                      << std::endl;
    }
}

bool RadiationDriver::prestepIndividual(
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D> const& cells,
    IndividualStepContext const&) const
{
    return prestep(tess, cells);
}

void RadiationDriver::releaseIndividualTopologyStorage() const noexcept
{
    std::vector<IndividualFaceCoefficient>().swap(
        individual_face_coefficients_);
    ReleaseDormantGlobalSolverStorage();
}

bool RadiationDriver::stepIndividual(
    double tolerance,
    int& total_iters,
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D>& cells,
    std::vector<Conserved3D>& extensives,
    IndividualStepContext const& context,
    double interval_fraction,
    double time,
    std::vector<ComputationalCell3D> const* canonical_cells,
    std::vector<Conserved3D>* canonical_extensives,
    std::vector<std::size_t> const* owned_to_canonical) const
{
    // Inject a documented floor only for a locally small deficit or when the
    // total negative group extent is globally negligible.  Positive groups
    // and material energy are unchanged; aggregate Erad is synchronized to
    // the repaired group sum and the injection is accounted explicitly.
    double constexpr spectral_positivity_repair_tolerance =
        RadiationPositivity::spectral_repair_relative_limit;

    // A reduced active-row solve is unnecessary when the event is globally
    // synchronized.  In that case the canonical matrix, unknown set, and
    // timestep are exactly the global problem, whose established solver is
    // both cheaper and more robust.  Keep the test collective so empty ranks
    // and partial-mesh mappings cannot select a different path.
    {
        std::size_t const owned_count = tess.GetPointNo();
        std::size_t const mesh_count = tess.getMeshPoints().size();
        OwnedCanonicalMappingAssessment const owned_mapping =
            assessOwnedCanonicalMapping(cells, owned_count, canonical_cells,
                                        owned_to_canonical);
        MappedAllActiveRuntimeOption const& mapped_option =
            mappedAllActiveRuntimeOption();
        bool const mapped_dispatch_allowed = mapped_option.valid &&
            (owned_mapping.identity || mapped_option.enabled);
        bool const complete_owned_activity =
            hasCompleteOwnedActivity(context, owned_count);
        bool const supported = supportsAllActiveIndividualGlobalStep();
        bool const mesh_sizes_valid = owned_count <= mesh_count &&
            cells.size() == mesh_count && extensives.size() == mesh_count;
        bool const canonical_sizes_valid = canonical_cells != nullptr &&
            canonical_extensives != nullptr &&
            canonical_cells->size() == owned_count &&
            canonical_extensives->size() == owned_count;
        bool const context_sizes_valid =
            context.active_mask.size() == mesh_count &&
            context.cell_time_steps.size() == mesh_count;
        bool const interval_fraction_valid =
            std::isfinite(interval_fraction) && interval_fraction > 0 &&
            interval_fraction <= 1;
        double scheduler_event_interval =
            std::numeric_limits<double>::quiet_NaN();
        bool const event_tick_interval_valid =
            schedulerEventInterval(context, scheduler_event_interval);
        bool local_fast_path = supported && mesh_sizes_valid &&
            canonical_sizes_valid && owned_mapping.valid &&
            mapped_dispatch_allowed && complete_owned_activity &&
            context_sizes_valid && interval_fraction_valid &&
            event_tick_interval_valid;
        // One bit per local predicate.  Reduce this mask instead of a separate
        // boolean so a profiled rejection is attributable without adding a
        // collective to the production path.
        unsigned long long local_rejection_mask = 0;
        if(!supported) local_rejection_mask |= 1ULL << 0;
        if(!mesh_sizes_valid) local_rejection_mask |= 1ULL << 1;
        if(!canonical_sizes_valid) local_rejection_mask |= 1ULL << 2;
        if(!owned_mapping.valid) local_rejection_mask |= 1ULL << 3;
        if(!mapped_dispatch_allowed) local_rejection_mask |= 1ULL << 4;
        if(!complete_owned_activity) local_rejection_mask |= 1ULL << 5;
        if(!context_sizes_valid) local_rejection_mask |= 1ULL << 6;
        if(!interval_fraction_valid) local_rejection_mask |= 1ULL << 7;
        if(!event_tick_interval_valid) local_rejection_mask |= 1ULL << 9;
        double local_minimum_dt = std::numeric_limits<double>::infinity();
        double local_maximum_dt = 0;
        bool const owned_mapping_identity = owned_mapping.identity;
        // Scheduler event order is not a canonical-cell order after
        // redistribution.  Complete activity is represented by the owned
        // active mask and count; requiring active_indices[i] == i incorrectly
        // disabled the global path for a valid permutation.
        if(local_fast_path)
            for(std::size_t i = 0; i < owned_count; ++i) {
                double const cell_dt = context.cellTimeStep(i);
                if(!context.isActive(i) || !std::isfinite(cell_dt) ||
                   cell_dt <= 0) {
                    local_fast_path = false;
                    local_rejection_mask |= 1ULL << 8;
                    break;
                }
                local_minimum_dt = std::min(local_minimum_dt, cell_dt);
                local_maximum_dt = std::max(local_maximum_dt, cell_dt);
            }

        unsigned long long global_owned_cells =
            static_cast<unsigned long long>(owned_count);
        unsigned long long global_active_cells =
            static_cast<unsigned long long>(context.active_indices.size());
        double global_minimum_dt = local_minimum_dt;
        double global_maximum_dt = local_maximum_dt;
        // Scheduler cell intervals and event intervals are both exact integer
        // tick differences times the same quantum.  Do not subtract two large
        // accumulated floating-point timestamps: that lost hundreds of ulps by
        // cycle 643 and spuriously disabled the all-active path.
        double minimum_event_interval = scheduler_event_interval;
        double maximum_event_interval = minimum_event_interval;
#ifdef RICH_MPI
        unsigned long long empty_owned_ranks = owned_count == 0 ? 1 : 0;
        unsigned long long nonempty_zero_active_ranks =
            owned_count > 0 && context.active_indices.empty() ? 1 : 0;
        unsigned long long global_rejection_mask = local_rejection_mask;
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &global_rejection_mask, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_BOR, MPI_COMM_WORLD),
            "MPI_Allreduce(individual all-active rejection mask)");
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &global_owned_cells, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
            "MPI_Allreduce(individual owned cells)");
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &global_active_cells, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
            "MPI_Allreduce(individual active cells)");
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &empty_owned_ranks, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
            "MPI_Allreduce(individual empty owned ranks)");
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &nonempty_zero_active_ranks, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
            "MPI_Allreduce(individual zero active ranks)");
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &global_minimum_dt, 1, MPI_DOUBLE,
                          MPI_MIN, MPI_COMM_WORLD),
            "MPI_Allreduce(individual minimum dt)");
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &global_maximum_dt, 1, MPI_DOUBLE,
                          MPI_MAX, MPI_COMM_WORLD),
            "MPI_Allreduce(individual maximum dt)");
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &minimum_event_interval, 1,
                          MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD),
            "MPI_Allreduce(individual minimum event interval)");
        requireDistributedMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &maximum_event_interval, 1,
                          MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
            "MPI_Allreduce(individual maximum event interval)");
        local_fast_path = global_rejection_mask == 0;
        static bool reported_zero_active_ownership = false;
        if(!reported_zero_active_ownership &&
           (empty_owned_ranks > 0 || nonempty_zero_active_ranks > 0)) {
            int rank = 0;
            int ranks = 1;
            requireDistributedMpiSuccess(
                MPI_Comm_rank(MPI_COMM_WORLD, &rank),
                "MPI_Comm_rank(active ownership report)");
            requireDistributedMpiSuccess(
                MPI_Comm_size(MPI_COMM_WORLD, &ranks),
                "MPI_Comm_size(active ownership report)");
            if(rank == 0)
                std::clog << "MG_ACTIVE_OWNERSHIP"
                          << " ranks=" << ranks
                          << " global_owned_cells=" << global_owned_cells
                          << " global_active_cells=" << global_active_cells
                          << " empty_owned_ranks=" << empty_owned_ranks
                          << " nonempty_zero_active_ranks="
                          << nonempty_zero_active_ranks << std::endl;
            reported_zero_active_ownership = true;
        }
#endif
        unsigned long long post_rejection_mask = 0;
        if(global_owned_cells == 0) post_rejection_mask |= 1ULL << 10;
        if(global_active_cells != global_owned_cells)
            post_rejection_mask |= 1ULL << 11;
        if(!std::isfinite(global_minimum_dt) || global_minimum_dt <= 0)
            post_rejection_mask |= 1ULL << 12;
        if(global_minimum_dt != global_maximum_dt)
            post_rejection_mask |= 1ULL << 13;
        if(!std::isfinite(minimum_event_interval) ||
           minimum_event_interval <= 0)
            post_rejection_mask |= 1ULL << 14;
        if(minimum_event_interval != maximum_event_interval)
            post_rejection_mask |= 1ULL << 15;
        if(global_minimum_dt != minimum_event_interval)
            post_rejection_mask |= 1ULL << 16;
#ifdef RICH_MPI
        global_rejection_mask |= post_rejection_mask;
#else
        local_rejection_mask |= post_rejection_mask;
#endif
        bool const use_fast_path = local_fast_path &&
            post_rejection_mask == 0;
        bool report_fast_path_eligibility = false;
        if(char const* const setting =
               std::getenv("RICH_MG_DISTRIBUTED_ACTIVE_PROFILE")) {
            bool parsed = false;
            report_fast_path_eligibility =
                parseDistributedActiveToggle(setting, parsed) && parsed;
        }
        if(report_fast_path_eligibility && !use_fast_path) {
            int diagnostic_rank = 0;
#ifdef RICH_MPI
            requireDistributedMpiSuccess(
                MPI_Comm_rank(MPI_COMM_WORLD, &diagnostic_rank),
                "MPI_Comm_rank(all-active eligibility report)");
#endif
            if(diagnostic_rank == 0)
                std::clog << "MG_INDIVIDUAL_ALL_ACTIVE_ELIGIBILITY"
#ifdef RICH_MPI
                          << " rejection_mask=" << global_rejection_mask
#else
                          << " rejection_mask=" << local_rejection_mask
#endif
                          << " global_owned_cells=" << global_owned_cells
                          << " global_active_cells=" << global_active_cells
                          << " owned_mapping_valid="
                          << (owned_mapping.valid ? 1 : 0)
                          << " owned_mapping_identity="
                          << (owned_mapping_identity ? 1 : 0)
                          << " complete_owned_activity="
                          << (complete_owned_activity ? 1 : 0)
                          << " global_minimum_dt=" << global_minimum_dt
                          << " global_maximum_dt=" << global_maximum_dt
                          << " minimum_event_interval="
                          << minimum_event_interval
                          << " maximum_event_interval="
                          << maximum_event_interval
                          << " interval_fraction=" << interval_fraction
                          << std::endl;
        }
        if(use_fast_path) {
            clearStepFailure();
            AllActiveInnerSnapshotRuntimeOption const& snapshot_option =
                allActiveInnerSnapshotRuntimeOption();
            if(!snapshot_option.valid)
                throw UniversalError(
                    "RICH_MG_ELIDE_INNER_ALL_ACTIVE_SNAPSHOT is invalid or differs across MPI ranks");
            std::vector<ComputationalCell3D> saved_cells = cells;
            std::vector<Conserved3D> saved_extensives = extensives;
            bool prepared = true;
            try {
                prepareIndividualCandidate(tess, cells);
                individual_face_coefficients_.clear();
            }
            catch(UniversalError const& error) {
                setStepFailure(error.getErrorMessage());
                prepared = false;
            }
            catch(std::exception const& error) {
                setStepFailure(error.what());
                prepared = false;
            }
            catch(...) {
                setStepFailure(
                    "unknown exception while preparing all-active radiation");
                prepared = false;
            }
#ifdef RICH_MPI
            int collective_prepared = prepared ? 1 : 0;
            requireDistributedMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &collective_prepared, 1, MPI_INT,
                              MPI_MIN, MPI_COMM_WORLD),
                "MPI_Allreduce(all-active preparation)");
            prepared = collective_prepared != 0;
#endif
            if(!prepared) {
                cells = saved_cells;
                extensives = saved_extensives;
                if(getLastStepFailureReason().empty())
                {
                    setStepFailure(
                        "all-active radiation preparation failed on another rank");
                    markStepFailureRemote();
                }
                return false;
            }

            int diagnostic_rank = 0;
#ifdef RICH_MPI
            requireDistributedMpiSuccess(
                MPI_Comm_rank(MPI_COMM_WORLD, &diagnostic_rank),
                "MPI_Comm_rank(all-active fast path report)");
#endif
            double const candidate_dt =
                interval_fraction * global_minimum_dt;
            if(diagnostic_rank == 0)
                std::clog << "MG_INDIVIDUAL_ALL_ACTIVE_FAST_PATH"
                          << " active_cells=" << global_active_cells
                          << " diagnostic_owned_cells=" << owned_count
                          << " diagnostic_mesh_cells=" << mesh_count
                          << " owned_mapping="
                          << (owned_mapping_identity ? "identity" : "permutation")
                          << " array_scope=mesh_with_ghosts"
                          << " scheduled_dt=" << global_minimum_dt
                          << " interval_fraction=" << interval_fraction
                          << " candidate_dt=" << candidate_dt
                          << " inner_snapshot_elided="
                          << (snapshot_option.enabled ? 1 : 0) << std::endl;
            std::vector<ComputationalCell3D> const*
                const previous_transaction_cells = outer_transaction_cells_;
            std::vector<Conserved3D> const*
                const previous_transaction_extensives =
                    outer_transaction_extensives_;
            if(snapshot_option.enabled) {
                outer_transaction_cells_ = &cells;
                outer_transaction_extensives_ = &extensives;
            }
            bool accepted = false;
            try {
                accepted = step(tolerance, total_iters, tess, cells,
                                extensives, candidate_dt, time);
            }
            catch(...) {
                if(snapshot_option.enabled) {
                    outer_transaction_cells_ = previous_transaction_cells;
                    outer_transaction_extensives_ =
                        previous_transaction_extensives;
                    // Restore without allocating so the original exception is
                    // not replaced by a rollback allocation failure.
                    cells.swap(saved_cells);
                    extensives.swap(saved_extensives);
                }
                throw;
            }
            if(snapshot_option.enabled) {
                outer_transaction_cells_ = previous_transaction_cells;
                outer_transaction_extensives_ =
                    previous_transaction_extensives;
            }
#ifdef RICH_MPI
            int collective_accepted = accepted ? 1 : 0;
            requireDistributedMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &collective_accepted, 1, MPI_INT,
                              MPI_MIN, MPI_COMM_WORLD),
                "MPI_Allreduce(all-active acceptance)");
            accepted = collective_accepted != 0;
#endif
            if(!accepted) {
                // The outer transaction already owns complete rollback
                // vectors.  Swap them back so a rejection cannot require a
                // fresh allocation while the solver is under memory pressure.
                cells.swap(saved_cells);
                extensives.swap(saved_extensives);
                if(getLastStepFailureReason().empty())
                {
                    setStepFailure(
                        "all-active global radiation solve failed on another rank");
                    markStepFailureRemote();
                }
            }
            return accepted;
        }
    }
    ReleaseDormantGlobalSolverStorage();
#ifdef RICH_MPI
    clearStepFailure();
    int rank = 0;
    int rank_count = 1;
    requireDistributedMpiSuccess(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank(distributed active step)");
    requireDistributedMpiSuccess(
        MPI_Comm_size(MPI_COMM_WORLD, &rank_count),
        "MPI_Comm_size(distributed active step)");
    try {
    auto const phase_total_start = std::chrono::steady_clock::now();
    auto phase_start = phase_total_start;
    double snapshot_seconds = 0;
    double candidate_seconds = 0;
    double matrix_build_seconds = 0;
    double active_mapping_seconds = 0;
    double csr_extract_seconds = 0;
    double exchange_setup_seconds = 0;
    double solver_seconds = 0;
    double profile_overhead_seconds = 0;
    std::vector<ComputationalCell3D> saved_cells = cells;
    std::vector<Conserved3D> saved_extensives = extensives;
    std::size_t const saved_extensives_size = saved_extensives.size();
    SpectralRepairEvent spectral_repair_event;
    IndividualRadiationDefectEvent pending_defect_event;
    bool pending_dirichlet_defect = false;
    std::map<std::size_t, double> pending_dirichlet_wakes;
    double local_defect_rhs_magnitude = 0;
    // The rollback snapshot is also the immutable pre-correction state used by
    // the diagnostics below.  Keeping a second deep copy of Conserved3D made
    // the largest AMR rank enter direct reclaim before every active solve.
    std::vector<Conserved3D>& transaction_start_extensives =
        saved_extensives;
    IndividualContextGuard const guard(individual_context_,
                                       individual_interval_fraction_,
                                       context,
                                       interval_fraction);
    auto reject = [&]() {
        cells = std::move(saved_cells);
        saved_extensives.resize(saved_extensives_size);
        extensives = std::move(saved_extensives);
        return false;
    };
    DistributedActiveRuntimeOptions runtime_options;
    if(!loadDistributedActiveRuntimeOptions(runtime_options)) {
        setStepFailure(
            "individual radiation runtime policy/options are invalid, "
            "conflicting, or differ across ranks");
        return reject();
    }
    static bool passive_policy_reported = false;
    if(!passive_policy_reported && rank == 0) {
        IndividualPassiveRadiationRuntimeOption const& passive_option =
            individualPassiveRadiationRuntimeOption();
        std::clog << "MG_INDIVIDUAL_PASSIVE_POLICY policy="
                  << individualPassiveRadiationPolicyLabel(
                         runtime_options.passive_policy)
                  << " deprecated_shadow_alias="
                  << (passive_option.deprecated_alias_present ? 1 : 0)
                  << " library_default="
                  << individualPassiveRadiationPolicyLabel(
                         individual_passive_radiation_default)
                  << std::endl;
        passive_policy_reported = true;
    }
    if(runtime_options.profile && rank == 0)
        std::clog << "MG_DISTRIBUTED_ACTIVE_OPTIONS profile=1 overlap="
                  << (runtime_options.overlap_local_rows ? 1 : 0)
                  << " paired_omega="
                  << (runtime_options.pair_omega_reduction ? 1 : 0)
                  << " fixed16_remote_slots="
                  << (runtime_options.fixed_16_remote_slots ? 1 : 0)
                  << " passive_policy="
                  << individualPassiveRadiationPolicyLabel(
                         runtime_options.passive_policy)
                  << std::endl;

    snapshot_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase_start).count();
    phase_start = std::chrono::steady_clock::now();
    std::size_t const unknowns_per_cell = individualUnknownsPerCell();
    CG::mat full_matrix;
    CG::size_t_mat full_columns;
    std::vector<double> full_rhs;
    std::vector<double> full_initial;
    std::vector<int> shadow_prebuild_point_owner;
    std::vector<ComputationalCell3D> shadow_matrix_cells;
    std::vector<ComputationalCell3D> const* matrix_cells = &cells;
    bool valid = true;
    try {
        if(runtime_options.usesShadowPassiveRows()) {
            valid = unknowns_per_cell > 1 &&
                canonical_cells != nullptr &&
                canonical_extensives != nullptr &&
                canonical_cells->size() ==
                    canonical_extensives->size();
            bool owner_mapping_valid = true;
            shadow_prebuild_point_owner =
                meshPointOwners(
                    tess, cells.size(), owner_mapping_valid);
            valid = valid && owner_mapping_valid;

            std::map<std::size_t, std::size_t>
                canonical_index_by_id;
            if(valid)
                for(std::size_t canonical = 0;
                    canonical < canonical_cells->size(); ++canonical)
                    if(!canonical_index_by_id.emplace(
                           (*canonical_cells)[canonical].ID,
                           canonical).second)
                        valid = false;

            std::vector<std::vector<IndividualShadowStateRequest> >
                outgoing_requests(
                    static_cast<std::size_t>(rank_count));
            std::map<std::size_t, int> requested_owner_by_id;
            std::set<std::pair<int, std::size_t> > unique_requests;
            std::vector<std::size_t> neighbors;
            for(std::size_t const active :
                context.active_indices) {
                if(active >= tess.GetPointNo() ||
                   active >= cells.size() ||
                   !context.isActive(active)) {
                    valid = false;
                    continue;
                }
                tess.GetNeighbors(active, neighbors);
                for(std::size_t const passive : neighbors) {
                    if(passive >= cells.size() ||
                       passive >=
                           shadow_prebuild_point_owner.size() ||
                       passive >= context.active_mask.size() ||
                       tess.IsPointOutsideBox(passive) ||
                       context.isActive(passive))
                        continue;
                    int const owner =
                        shadow_prebuild_point_owner[passive];
                    if(owner < 0 || owner >= rank_count) {
                        valid = false;
                        continue;
                    }
                    std::size_t const passive_id =
                        cells[passive].ID;
                    auto const prior =
                        requested_owner_by_id.emplace(
                            passive_id, owner);
                    if(!prior.second &&
                       prior.first->second != owner) {
                        valid = false;
                        continue;
                    }
                    if(unique_requests.emplace(
                           owner, passive_id).second) {
                        IndividualShadowStateRequest request;
                        request.passive_cell_id = passive_id;
                        outgoing_requests[
                            static_cast<std::size_t>(owner)]
                            .push_back(request);
                    }
                }
            }
            if(!valid)
                setStepFailure(
                    "shadow passive primitive requests are invalid");
            if(!collectiveAllTrue(valid))
                return reject();

            std::vector<std::vector<IndividualShadowStateRequest> >
                const incoming_requests =
                    MPI_Exchange_all_to_all(
                        outgoing_requests, MPI_COMM_WORLD);
            valid = incoming_requests.size() ==
                static_cast<std::size_t>(rank_count);
            std::vector<
                std::vector<IndividualShadowPrimitiveValue> >
                outgoing_values(
                    static_cast<std::size_t>(rank_count));
            for(std::size_t peer = 0;
                peer < incoming_requests.size(); ++peer)
                for(IndividualShadowStateRequest const& request :
                    incoming_requests[peer]) {
                    auto const canonical =
                        canonical_index_by_id.find(
                            request.passive_cell_id);
                    if(canonical ==
                       canonical_index_by_id.end()) {
                        valid = false;
                        continue;
                    }
                    Conserved3D const& extent =
                        (*canonical_extensives)[
                            canonical->second];
                    if(!std::isfinite(extent.mass) ||
                       !(extent.mass > 0) ||
                       extent.Eg.size() <
                           unknowns_per_cell) {
                        valid = false;
                        continue;
                    }
                    for(std::size_t group = 0;
                        group < unknowns_per_cell;
                        ++group) {
                        IndividualShadowPrimitiveValue value;
                        value.passive_cell_id =
                            request.passive_cell_id;
                        value.group = group;
                        value.specific_energy =
                            extent.Eg[group] / extent.mass;
                        if(!std::isfinite(
                               value.specific_energy) ||
                           value.specific_energy < 0)
                            valid = false;
                        outgoing_values[peer].push_back(value);
                    }
                }
            if(!valid)
                setStepFailure(
                    "shadow passive canonical primitive state is invalid");
            if(!collectiveAllTrue(valid))
                return reject();

            std::vector<
                std::vector<IndividualShadowPrimitiveValue> >
                const incoming_values =
                    MPI_Exchange_all_to_all(
                        outgoing_values, MPI_COMM_WORLD);
            valid = incoming_values.size() ==
                static_cast<std::size_t>(rank_count);
            std::map<
                std::pair<std::size_t, std::size_t>, double>
                primitive_by_id_group;
            for(std::size_t peer = 0;
                peer < incoming_values.size(); ++peer)
                for(IndividualShadowPrimitiveValue const& value :
                    incoming_values[peer]) {
                    auto const requested =
                        requested_owner_by_id.find(
                            value.passive_cell_id);
                    if(requested ==
                           requested_owner_by_id.end() ||
                       requested->second !=
                           static_cast<int>(peer) ||
                       value.group >= unknowns_per_cell ||
                       !std::isfinite(value.specific_energy) ||
                       value.specific_energy < 0) {
                        valid = false;
                        continue;
                    }
                    auto const inserted =
                        primitive_by_id_group.emplace(
                            std::make_pair(
                                value.passive_cell_id,
                                value.group),
                            value.specific_energy);
                    if(!inserted.second &&
                       inserted.first->second !=
                           value.specific_energy)
                        valid = false;
                }

            shadow_matrix_cells = cells;
            for(auto const& requested :
                requested_owner_by_id) {
                std::vector<double> spectrum(
                    unknowns_per_cell, 0);
                double total = 0;
                for(std::size_t group = 0;
                    group < unknowns_per_cell; ++group) {
                    auto const value =
                        primitive_by_id_group.find(
                            std::make_pair(
                                requested.first, group));
                    if(value ==
                       primitive_by_id_group.end()) {
                        valid = false;
                        continue;
                    }
                    spectrum[group] = value->second;
                    total += value->second;
                }
                bool found_local_copy = false;
                for(std::size_t cell = 0;
                    cell < shadow_matrix_cells.size();
                    ++cell) {
                    if(shadow_matrix_cells[cell].ID !=
                       requested.first)
                        continue;
                    if(cell < context.active_mask.size() &&
                       context.isActive(cell)) {
                        valid = false;
                        continue;
                    }
                    if(shadow_matrix_cells[cell].Eg.size() <
                       unknowns_per_cell) {
                        valid = false;
                        continue;
                    }
                    found_local_copy = true;
                    for(std::size_t group = 0;
                        group < unknowns_per_cell; ++group)
                        shadow_matrix_cells[cell].Eg[group] =
                            spectrum[group];
                    shadow_matrix_cells[cell].Erad = total;
                }
                if(!found_local_copy)
                    valid = false;
            }
            if(!valid)
                setStepFailure(
                    "shadow passive primitive synchronization failed");
            if(!collectiveAllTrue(valid))
                return reject();
            matrix_cells = &shadow_matrix_cells;
        }

        prepareIndividualCandidate(tess, *matrix_cells);
        individual_face_coefficients_.clear();
        candidate_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - phase_start).count();
        phase_start = std::chrono::steady_clock::now();
        BuildMatrix(tess, full_matrix, full_columns, *matrix_cells, 0,
                    full_rhs, full_initial, time);
        valid = validateIndividualCoefficients(
            context, *matrix_cells);
        if(!valid && getLastStepFailureReason().empty())
            setStepFailure("individual radiation coefficients are invalid");
    }
    catch(std::exception const& error) {
        setStepFailure(error.what());
        valid = false;
    }
    catch(...) {
        setStepFailure("unknown exception while building the active radiation matrix");
        valid = false;
    }
    if(!collectiveAllTrue(valid))
        return reject();
    matrix_build_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase_start).count();
    if(runtime_options.profile) {
        auto const profile_start = std::chrono::steady_clock::now();
        reportDistributedMemoryPhase("full_matrix_built");
        profile_overhead_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - profile_start).count();
    }
    phase_start = std::chrono::steady_clock::now();

    std::size_t owned_row_count = 0;
    valid = unknowns_per_cell > 0 &&
        checkedDistributedSizeMultiply(
            tess.GetPointNo(), unknowns_per_cell, owned_row_count) &&
        full_matrix.size() == owned_row_count &&
        full_columns.size() == owned_row_count &&
        full_rhs.size() == owned_row_count &&
        full_initial.size() >= owned_row_count;
    if(!valid)
        setStepFailure("radiation matrix size does not match the MPI cell/group layout");

    bool owner_mapping_valid = true;
    std::vector<int> point_owner =
        runtime_options.usesShadowPassiveRows() ?
        std::move(shadow_prebuild_point_owner) :
        meshPointOwners(
            tess, cells.size(), owner_mapping_valid);
    valid = valid && owner_mapping_valid;
    if(!owner_mapping_valid)
        setStepFailure("individual radiation MPI point ownership is inconsistent");
    if(!collectiveAllTrue(valid))
        return reject();

    struct OwnedShadowCell
    {
        std::size_t cell_index = CG::max_size_t;
        std::size_t canonical_index = CG::max_size_t;
        double volume_cgs = 0;
        std::vector<IndividualShadowFaceRecord> faces;
    };
    std::map<std::size_t, OwnedShadowCell> owned_shadow_cells;
    std::map<std::size_t, int> solver_owner_by_id;
    std::map<std::pair<int, std::size_t>, std::size_t>
        solver_cell_index_by_owner_id;
    std::map<std::size_t, std::size_t>
        interior_owned_cell_index_by_id;
    for(std::size_t cell = 0; cell < cells.size(); ++cell) {
        if(cell < point_owner.size() && point_owner[cell] >= 0)
            solver_cell_index_by_owner_id.emplace(
                std::make_pair(point_owner[cell], cells[cell].ID), cell);
        if(cell < point_owner.size() && point_owner[cell] == rank &&
           cell < tess.GetPointNo() && !tess.IsPointOutsideBox(cell))
            interior_owned_cell_index_by_id.emplace(
                cells[cell].ID, cell);
    }

    if(runtime_options.usesShadowPassiveRows()) {
        valid = unknowns_per_cell > 1 && canonical_cells != nullptr &&
            canonical_extensives != nullptr &&
            canonical_cells->size() == canonical_extensives->size();
        if(!valid)
            setStepFailure(
                "shadow passive radiation rows require canonical multigroup state");
        if(!collectiveAllTrue(valid))
            return reject();

        std::map<std::size_t, std::size_t> shadow_canonical_index_by_id;
        for(std::size_t canonical = 0;
            canonical < canonical_cells->size(); ++canonical)
            if(!shadow_canonical_index_by_id.emplace(
                   (*canonical_cells)[canonical].ID, canonical).second)
                valid = false;

        std::vector<std::vector<IndividualShadowFaceRecord> >
            outgoing_shadow_faces(static_cast<std::size_t>(rank_count));
        double const length_scale_cubed =
            GetLengthScale() * GetLengthScale() * GetLengthScale();
        for(IndividualFaceCoefficient const& face :
            individual_face_coefficients_) {
            bool const left_active =
                face.left < context.active_mask.size() &&
                context.isActive(face.left);
            bool const right_active =
                face.right < context.active_mask.size() &&
                context.isActive(face.right);
            if(left_active == right_active)
                continue;
            std::size_t const active =
                left_active ? face.left : face.right;
            std::size_t const passive =
                left_active ? face.right : face.left;
            if(active >= cells.size() || passive >= cells.size() ||
               active >= point_owner.size() ||
               passive >= point_owner.size() ||
               tess.IsPointOutsideBox(passive) ||
               point_owner[active] < 0 ||
               point_owner[active] >= rank_count ||
               point_owner[passive] < 0 ||
               point_owner[passive] >= rank_count ||
               face.group >= unknowns_per_cell) {
                valid = false;
                continue;
            }
            std::size_t active_unknown = 0;
            if(!checkedDistributedSizeMultiply(
                   active, unknowns_per_cell, active_unknown) ||
               !checkedDistributedSizeAdd(
                   active_unknown, face.group, active_unknown) ||
               active_unknown >= full_initial.size()) {
                valid = false;
                continue;
            }
            double const passive_volume_cgs =
                tess.GetVolume(passive) * length_scale_cubed;
            if(!std::isfinite(face.coefficient) ||
               face.coefficient < 0 ||
               !std::isfinite(passive_volume_cgs) ||
               !(passive_volume_cgs > 0) ||
               !std::isfinite(full_initial[active_unknown])) {
                valid = false;
                continue;
            }
            IndividualShadowFaceRecord record;
            record.passive_cell_id = cells[passive].ID;
            record.active_cell_id = cells[active].ID;
            record.group = face.group;
            record.active_owner = point_owner[active];
            record.coefficient = face.coefficient;
            record.passive_volume_cgs = passive_volume_cgs;
            record.active_base = full_initial[active_unknown];
            outgoing_shadow_faces[
                static_cast<std::size_t>(point_owner[passive])].push_back(
                    record);
            solver_owner_by_id[record.passive_cell_id] =
                point_owner[passive];
            solver_owner_by_id[record.active_cell_id] =
                record.active_owner;
        }
        if(!valid)
            setStepFailure(
                "shadow passive radiation face metadata is invalid");
        if(!collectiveAllTrue(valid))
            return reject();

        std::vector<std::vector<IndividualShadowFaceRecord> > const
            incoming_shadow_faces =
                MPI_Exchange_all_to_all(
                    outgoing_shadow_faces, MPI_COMM_WORLD);
        valid = incoming_shadow_faces.size() ==
            static_cast<std::size_t>(rank_count);
        std::vector<std::vector<IndividualShadowInitialValue> >
            outgoing_shadow_initials(static_cast<std::size_t>(rank_count));
        double const shadow_extensive_conversion =
            time_scale_ * time_scale_ /
            (length_scale_ * length_scale_ * mass_scale_);
        valid = valid && std::isfinite(shadow_extensive_conversion) &&
            shadow_extensive_conversion > 0;

        for(std::size_t peer = 0;
            peer < incoming_shadow_faces.size(); ++peer)
            for(IndividualShadowFaceRecord const& record :
                incoming_shadow_faces[peer]) {
                auto const canonical = shadow_canonical_index_by_id.find(
                    record.passive_cell_id);
                if(canonical == shadow_canonical_index_by_id.end() ||
                   record.active_owner != static_cast<int>(peer) ||
                   record.group >= unknowns_per_cell ||
                   !std::isfinite(record.coefficient) ||
                   record.coefficient < 0 ||
                   !std::isfinite(record.passive_volume_cgs) ||
                   !(record.passive_volume_cgs > 0) ||
                   !std::isfinite(record.active_base)) {
                    valid = false;
                    continue;
                }
                Conserved3D const& passive_extent =
                    (*canonical_extensives)[canonical->second];
                if(record.group >= passive_extent.Eg.size() ||
                   !std::isfinite(passive_extent.Eg[record.group]) ||
                   passive_extent.Eg[record.group] < 0) {
                    valid = false;
                    continue;
                }

                OwnedShadowCell& shadow =
                    owned_shadow_cells[record.passive_cell_id];
                if(shadow.cell_index == CG::max_size_t) {
                    auto const existing =
                        interior_owned_cell_index_by_id.find(
                            record.passive_cell_id);
                    if(existing != interior_owned_cell_index_by_id.end())
                        shadow.cell_index = existing->second;
                    else {
                        shadow.cell_index = cells.size();
                        cells.push_back(
                            (*canonical_cells)[canonical->second]);
                        extensives.push_back(passive_extent);
                        transaction_start_extensives.push_back(
                            passive_extent);
                        point_owner.push_back(rank);
                    }
                    // Replace any earlier ghost/outside duplicate selected by
                    // the generic endpoint map with the owner-held row cell.
                    solver_cell_index_by_owner_id[
                        std::make_pair(rank, record.passive_cell_id)] =
                        shadow.cell_index;
                    shadow.canonical_index = canonical->second;
                    shadow.volume_cgs = record.passive_volume_cgs;
                }
                double const volume_scale = std::max(
                    std::abs(shadow.volume_cgs),
                    std::abs(record.passive_volume_cgs));
                if(std::abs(
                       shadow.volume_cgs - record.passive_volume_cgs) >
                   64 * std::numeric_limits<double>::epsilon() *
                       std::max(volume_scale, 1.0)) {
                    valid = false;
                    continue;
                }
                shadow.faces.push_back(record);
                solver_owner_by_id[record.passive_cell_id] = rank;
                solver_owner_by_id[record.active_cell_id] =
                    record.active_owner;

                auto active_index = solver_cell_index_by_owner_id.find(
                    std::make_pair(
                        record.active_owner, record.active_cell_id));
                if(active_index ==
                   solver_cell_index_by_owner_id.end()) {
                    ComputationalCell3D proxy;
                    proxy.ID = record.active_cell_id;
                    std::size_t const proxy_index = cells.size();
                    cells.push_back(proxy);
                    extensives.emplace_back();
                    transaction_start_extensives.emplace_back();
                    point_owner.push_back(record.active_owner);
                    active_index =
                        solver_cell_index_by_owner_id.emplace(
                            std::make_pair(
                                record.active_owner,
                                record.active_cell_id),
                            proxy_index).first;
                }

                IndividualShadowInitialValue initial;
                initial.passive_cell_id = record.passive_cell_id;
                initial.group = record.group;
                initial.value =
                    passive_extent.Eg[record.group] /
                    (shadow_extensive_conversion *
                     record.passive_volume_cgs);
                if(!std::isfinite(initial.value) || initial.value < 0)
                    valid = false;
                outgoing_shadow_initials[peer].push_back(initial);
            }
        if(!valid)
            setStepFailure(
                "shadow passive radiation owner state is invalid");
        if(!collectiveAllTrue(valid))
            return reject();

        std::vector<std::vector<IndividualShadowInitialValue> > const
            incoming_shadow_initials =
                MPI_Exchange_all_to_all(
                    outgoing_shadow_initials, MPI_COMM_WORLD);
        valid = incoming_shadow_initials.size() ==
            static_cast<std::size_t>(rank_count);

        std::size_t full_row_count = 0;
        valid = checkedDistributedSizeMultiply(
            cells.size(), unknowns_per_cell, full_row_count) && valid;
        if(!valid || full_row_count < full_initial.size()) {
            setStepFailure(
                "shadow passive radiation full-row count is invalid");
            return reject();
        }
        full_matrix.resize(full_row_count);
        full_columns.resize(full_row_count);
        full_rhs.resize(full_row_count, 0);
        full_initial.resize(full_row_count, 0);

        std::map<std::size_t, std::vector<std::size_t> >
            cell_indices_by_id;
        for(std::size_t cell = 0; cell < cells.size(); ++cell)
            cell_indices_by_id[cells[cell].ID].push_back(cell);
        for(std::size_t peer = 0;
            peer < incoming_shadow_initials.size(); ++peer)
            for(IndividualShadowInitialValue const& initial :
                incoming_shadow_initials[peer]) {
                auto const expected_owner =
                    solver_owner_by_id.find(initial.passive_cell_id);
                auto const indices =
                    cell_indices_by_id.find(initial.passive_cell_id);
                if(expected_owner == solver_owner_by_id.end() ||
                   expected_owner->second != static_cast<int>(peer) ||
                   indices == cell_indices_by_id.end() ||
                   initial.group >= unknowns_per_cell ||
                   !std::isfinite(initial.value) ||
                   initial.value < 0) {
                    valid = false;
                    continue;
                }
                for(std::size_t const cell : indices->second) {
                    std::size_t unknown = 0;
                    if(!checkedDistributedSizeMultiply(
                           cell, unknowns_per_cell, unknown) ||
                       !checkedDistributedSizeAdd(
                           unknown, initial.group, unknown) ||
                       unknown >= full_initial.size()) {
                        valid = false;
                        continue;
                    }
                    full_initial[unknown] = initial.value;
                }
            }

        for(auto& entry : owned_shadow_cells) {
            OwnedShadowCell& shadow = entry.second;
            if(shadow.cell_index >= cells.size() ||
               shadow.canonical_index >= canonical_extensives->size() ||
               !(shadow.volume_cgs > 0)) {
                valid = false;
                continue;
            }
            Conserved3D const& passive_extent =
                (*canonical_extensives)[shadow.canonical_index];
            for(std::size_t group = 0;
                group < unknowns_per_cell; ++group) {
                if(group >= passive_extent.Eg.size()) {
                    valid = false;
                    continue;
                }
                std::size_t row = 0;
                if(!checkedDistributedSizeMultiply(
                       shadow.cell_index, unknowns_per_cell, row) ||
                   !checkedDistributedSizeAdd(row, group, row) ||
                   row >= full_matrix.size()) {
                    valid = false;
                    continue;
                }
                double diagonal = shadow.volume_cgs;
                full_matrix[row].clear();
                full_columns[row].clear();
                full_matrix[row].push_back(diagonal);
                full_columns[row].push_back(row);
                for(IndividualShadowFaceRecord const& face :
                    shadow.faces) {
                    if(face.group != group)
                        continue;
                    auto const active_index =
                        solver_cell_index_by_owner_id.find(
                            std::make_pair(
                                face.active_owner,
                                face.active_cell_id));
                    if(active_index ==
                       solver_cell_index_by_owner_id.end()) {
                        valid = false;
                        continue;
                    }
                    std::size_t column = 0;
                    if(!checkedDistributedSizeMultiply(
                           active_index->second,
                           unknowns_per_cell, column) ||
                       !checkedDistributedSizeAdd(
                           column, group, column) ||
                       column >= full_initial.size()) {
                        valid = false;
                        continue;
                    }
                    diagonal += face.coefficient;
                    full_matrix[row].push_back(-face.coefficient);
                    full_columns[row].push_back(column);
                    full_initial[column] = face.active_base;
                }
                full_matrix[row][0] = diagonal;
                full_rhs[row] =
                    passive_extent.Eg[group] /
                    shadow_extensive_conversion;
                full_initial[row] =
                    full_rhs[row] / shadow.volume_cgs;
            }
        }
        if(!valid)
            setStepFailure(
                "shadow passive radiation rows are inconsistent");
        if(!collectiveAllTrue(valid))
            return reject();
    }

    std::vector<std::size_t> global_to_local(
        full_initial.size(), CG::max_size_t);
    std::vector<std::size_t> local_to_global;
    std::size_t solver_cell_capacity = 0;
    valid = checkedDistributedSizeAdd(
        context.active_indices.size(), owned_shadow_cells.size(),
        solver_cell_capacity) && valid;
    std::size_t active_row_capacity = 0;
    if(!checkedDistributedSizeMultiply(
           solver_cell_capacity, unknowns_per_cell,
           active_row_capacity))
        valid = false;
    else
        local_to_global.reserve(active_row_capacity);
    std::map<std::size_t, std::size_t> owned_solver_cell_bases;
    std::vector<unsigned char> solver_unknown_cell(cells.size(), 0);
    std::vector<int> solver_cell_owner = point_owner;
    solver_cell_owner.resize(cells.size(), -1);
    std::vector<double> solver_cell_volume_cgs(
        cells.size(), std::numeric_limits<double>::quiet_NaN());
    double const solver_length_scale_cubed =
        GetLengthScale() * GetLengthScale() * GetLengthScale();
    // Preserve the legacy active-column semantics, including remote active
    // ghost copies.  Shadow metadata below only adds to this stable-ID map.
    for(std::size_t cell = 0;
        cell < cells.size() &&
        cell < context.active_mask.size(); ++cell) {
        if(!context.isActive(cell))
            continue;
        if(cell >= solver_cell_owner.size() ||
           solver_cell_owner[cell] < 0 ||
           solver_cell_owner[cell] >= rank_count) {
            valid = false;
            continue;
        }
        auto const inserted = solver_owner_by_id.emplace(
            cells[cell].ID, solver_cell_owner[cell]);
        if(!inserted.second &&
           inserted.first->second != solver_cell_owner[cell])
            valid = false;
    }
    auto append_owned_solver_block =
        [&](std::size_t const cell, double const volume_cgs)
    {
        if(cell >= cells.size() || cell >= solver_cell_owner.size() ||
           solver_cell_owner[cell] != rank ||
           !std::isfinite(volume_cgs) || !(volume_cgs > 0)) {
            valid = false;
            return;
        }
        std::size_t const cell_base = local_to_global.size();
        if(!owned_solver_cell_bases.emplace(
               cells[cell].ID, cell_base).second) {
            valid = false;
            return;
        }
        solver_owner_by_id[cells[cell].ID] = rank;
        solver_cell_volume_cgs[cell] = volume_cgs;
        for(std::size_t group = 0;
            group < unknowns_per_cell; ++group) {
            std::size_t global = 0;
            if(!checkedDistributedSizeMultiply(
                   cell, unknowns_per_cell, global) ||
               !checkedDistributedSizeAdd(global, group, global) ||
               global >= global_to_local.size() ||
               global_to_local[global] != CG::max_size_t) {
                valid = false;
                continue;
            }
            global_to_local[global] = local_to_global.size();
            local_to_global.push_back(global);
        }
    };
    for(std::size_t cell : context.active_indices) {
        if(cell >= tess.GetPointNo() || cell >= cells.size() ||
           !context.isActive(cell)) {
            valid = false;
            continue;
        }
        append_owned_solver_block(
            cell, tess.GetVolume(cell) * solver_length_scale_cubed);
    }
    for(auto const& entry : owned_shadow_cells)
        append_owned_solver_block(
            entry.second.cell_index, entry.second.volume_cgs);
    for(std::size_t cell = 0; cell < cells.size(); ++cell) {
        auto const owner = solver_owner_by_id.find(cells[cell].ID);
        if(owner == solver_owner_by_id.end())
            continue;
        solver_unknown_cell[cell] = 1;
        solver_cell_owner[cell] = owner->second;
    }
    if(!valid)
        setStepFailure(
            "individual radiation active/shadow-row mapping is inconsistent");
    if(!collectiveAllTrue(valid))
        return reject();
    active_mapping_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase_start).count();
    phase_start = std::chrono::steady_clock::now();

    DistributedActiveCSR matrix;
    std::size_t row_offset_count = 0;
    valid = checkedDistributedSizeAdd(
        local_to_global.size(), 1, row_offset_count) && valid;
    if(!valid)
        setStepFailure(
            "individual radiation active CSR row count overflow");
    if(!collectiveAllTrue(valid))
        return reject();
    matrix.row_offsets.assign(row_offset_count, 0);
    std::size_t active_nonzero_count = 0;
    for(std::size_t local_row = 0;
        local_row < local_to_global.size(); ++local_row) {
        std::size_t const global_row = local_to_global[local_row];
        if(global_row >= full_matrix.size() ||
           full_matrix[global_row].size() != full_columns[global_row].size()) {
            valid = false;
            matrix.row_offsets[local_row + 1] = active_nonzero_count;
            continue;
        }
        for(std::size_t const global_column : full_columns[global_row]) {
            if(global_column == CG::max_size_t)
                continue;
            if(global_column >= full_initial.size()) {
                valid = false;
                continue;
            }
            std::size_t const column_cell =
                global_column / unknowns_per_cell;
            if(column_cell >= solver_unknown_cell.size() ||
               solver_unknown_cell[column_cell] == 0)
                continue;
            if(active_nonzero_count == CG::max_size_t) {
                valid = false;
                continue;
            }
            ++active_nonzero_count;
        }
        matrix.row_offsets[local_row + 1] = active_nonzero_count;
    }
    if(!valid)
        setStepFailure(
            "individual radiation active CSR nonzero count is inconsistent");
    if(!collectiveAllTrue(valid))
        return reject();

    std::size_t const maximum_narrow_column =
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max());
    bool const narrow_columns =
        active_nonzero_count == 0 ||
        (local_to_global.size() <= maximum_narrow_column &&
         active_nonzero_count - 1 <=
             maximum_narrow_column - local_to_global.size());
    matrix.ReserveColumns(0, narrow_columns);
    matrix.local_rows.reserve(local_to_global.size());
    matrix.remote_rows.reserve(local_to_global.size());
    std::vector<double> rhs(local_to_global.size(), 0);
    std::vector<double> verification_rhs(local_to_global.size(), 0);
    std::vector<double> verification_scale(local_to_global.size(), 0);
    std::vector<double> base_solution(local_to_global.size(), 0);
    std::vector<double> final_correction_volume(local_to_global.size(), 0);
    std::vector<double> solution(local_to_global.size(), 0);
    bool const fixed_group_remote_slots =
        runtime_options.fixed_16_remote_slots &&
        !runtime_options.usesShadowPassiveRows() &&
        unknowns_per_cell == 16;
    std::vector<std::size_t> remote_cell_base_by_point(
        cells.size(), CG::max_size_t);
    std::map<RemoteUnknownKey, std::size_t> remote_cell_base_by_key;
    std::map<RemoteUnknownKey, std::size_t> remote_slot_by_key;
    std::vector<RemoteUnknownKey> remote_unknowns;
    remote_unknowns.reserve(local_to_global.size());

    // Inactive full-system rows can never contribute to the reduced operator.
    // Release them before streaming the active rows into CSR, so the build
    // never retains complete full and reduced matrix representations.
    for(std::size_t global_row = 0;
        global_row < full_matrix.size(); ++global_row)
        if(global_row >= global_to_local.size() ||
           global_to_local[global_row] == CG::max_size_t) {
            CG::vec().swap(full_matrix[global_row]);
            CG::vec_size_t().swap(full_columns[global_row]);
        }
    if(runtime_options.profile) {
        csr_extract_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - phase_start).count();
        auto const profile_start = std::chrono::steady_clock::now();
        reportDistributedMemoryPhase("inactive_rows_released");
        profile_overhead_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - profile_start).count();
        phase_start = std::chrono::steady_clock::now();
    }

    for(std::size_t local_row = 0;
        local_row < local_to_global.size(); ++local_row) {
        std::size_t const global_row = local_to_global[local_row];
        matrix.ReserveStorageForAppend(
            matrix.row_offsets[local_row + 1], active_nonzero_count);
        matrix.row_offsets[local_row] = matrix.values.size();
        bool has_remote_column = false;
        if(global_row >= full_matrix.size() ||
           full_matrix[global_row].size() != full_columns[global_row].size()) {
            valid = false;
            matrix.row_offsets[local_row + 1] = matrix.values.size();
            matrix.local_rows.push_back(local_row);
            continue;
        }
        AccurateResidualAccumulator correction_row;
        AccurateResidualAccumulator verification_row;
        AccurateResidualAccumulator fixed_scale_row;
        correction_row.Add(full_rhs[global_row]);
        verification_row.Add(full_rhs[global_row]);
        fixed_scale_row.Add(std::abs(full_rhs[global_row]));
        base_solution[local_row] = full_initial[global_row];
        std::size_t const row_cell =
            global_row / unknowns_per_cell;
        if(row_cell >= solver_cell_volume_cgs.size() ||
           !std::isfinite(solver_cell_volume_cgs[row_cell]) ||
           !(solver_cell_volume_cgs[row_cell] > 0)) {
            valid = false;
            continue;
        }
        final_correction_volume[local_row] =
            solver_cell_volume_cgs[row_cell];
        for(std::size_t entry = 0;
            entry < full_matrix[global_row].size(); ++entry) {
            std::size_t const global_column =
                full_columns[global_row][entry];
            if(global_column == CG::max_size_t)
                continue;
            if(global_column >= full_initial.size()) {
                valid = false;
                continue;
            }
            double const value = full_matrix[global_row][entry];
            correction_row.AddProduct(
                -value, full_initial[global_column]);
            std::size_t const column_cell =
                global_column / unknowns_per_cell;
            std::size_t const column_group =
                global_column % unknowns_per_cell;
            bool const column_is_unknown =
                column_cell < solver_unknown_cell.size() &&
                solver_unknown_cell[column_cell] != 0;
            if(!column_is_unknown) {
                verification_row.AddProduct(
                    -value, full_initial[global_column]);
                fixed_scale_row.AddProduct(
                    std::abs(value),
                    std::abs(full_initial[global_column]));
                continue;
            }
            if(column_cell >= cells.size() ||
               column_cell >= point_owner.size()) {
                valid = false;
                continue;
            }
            int const owner = solver_cell_owner[column_cell];
            if(owner == rank) {
                std::size_t const local_column =
                    global_to_local[global_column];
                if(local_column == CG::max_size_t) {
                    valid = false;
                    continue;
                }
                matrix.values.push_back(value);
                matrix.PushColumn(local_column);
            }
            else if(owner >= 0 && owner < rank_count) {
                if(!fixed_group_remote_slots) {
                    RemoteUnknownKey key;
                    key.owner = owner;
                    key.cell_id = cells[column_cell].ID;
                    key.group = column_group;
                    auto const inserted = remote_slot_by_key.emplace(
                        key, remote_unknowns.size());
                    if(inserted.second)
                        remote_unknowns.push_back(key);
                    std::size_t remote_column = 0;
                    if(!checkedDistributedSizeAdd(
                           local_to_global.size(), inserted.first->second,
                           remote_column)) {
                        valid = false;
                        continue;
                    }
                    matrix.values.push_back(value);
                    matrix.PushColumn(remote_column);
                    has_remote_column = true;
                    continue;
                }
                std::size_t remote_cell_base =
                    remote_cell_base_by_point[column_cell];
                if(remote_cell_base == CG::max_size_t) {
                    RemoteUnknownKey cell_key;
                    cell_key.owner = owner;
                    cell_key.cell_id = cells[column_cell].ID;
                    cell_key.group = 0;
                    auto const inserted = remote_cell_base_by_key.emplace(
                        cell_key, remote_unknowns.size());
                    remote_cell_base = inserted.first->second;
                    if(inserted.second) {
                        std::size_t remote_unknown_count = 0;
                        if(!checkedDistributedSizeAdd(
                               remote_unknowns.size(), unknowns_per_cell,
                               remote_unknown_count)) {
                            valid = false;
                            continue;
                        }
                        for(std::size_t group = 0;
                            group < unknowns_per_cell; ++group) {
                            RemoteUnknownKey unknown = cell_key;
                            unknown.group = group;
                            remote_unknowns.push_back(unknown);
                        }
                        if(remote_unknowns.size() != remote_unknown_count) {
                            valid = false;
                            continue;
                        }
                    }
                    remote_cell_base_by_point[column_cell] =
                        remote_cell_base;
                }
                std::size_t remote_slot = 0;
                if(!checkedDistributedSizeAdd(
                       remote_cell_base, column_group, remote_slot)) {
                    valid = false;
                    continue;
                }
                std::size_t remote_column = 0;
                if(!checkedDistributedSizeAdd(
                       local_to_global.size(), remote_slot,
                       remote_column)) {
                    valid = false;
                    continue;
                }
                matrix.values.push_back(value);
                matrix.PushColumn(remote_column);
                has_remote_column = true;
            }
            else {
                valid = false;
            }
        }
        rhs[local_row] = correction_row.Value();
        verification_rhs[local_row] = verification_row.Value();
        verification_scale[local_row] = fixed_scale_row.Value();
        matrix.row_offsets[local_row + 1] = matrix.values.size();
        if(has_remote_column)
            matrix.remote_rows.push_back(local_row);
        else
            matrix.local_rows.push_back(local_row);
        CG::vec().swap(full_matrix[global_row]);
        CG::vec_size_t().swap(full_columns[global_row]);
    }
    if(!valid)
        setStepFailure("individual radiation matrix references an unmapped active unknown");
    if(!collectiveAllTrue(valid))
        return reject();

    AccurateResidualAccumulator defect_rhs_accumulator;
    for(double const value : full_rhs) {
        if(!std::isfinite(value)) {
            local_defect_rhs_magnitude =
                std::numeric_limits<double>::quiet_NaN();
            break;
        }
        defect_rhs_accumulator.Add(std::abs(value));
    }
    if(std::isfinite(local_defect_rhs_magnitude))
        local_defect_rhs_magnitude =
            std::abs(defect_rhs_accumulator.Value());

    // The active-row system below is self-contained.  Release the full build
    // workspace before solving so that a cell-local Compton rebuild does not
    // overlap two complete copies of the distributed matrix.
    CG::mat().swap(full_matrix);
    CG::size_t_mat().swap(full_columns);
    std::vector<double>().swap(full_rhs);
    std::vector<std::size_t>().swap(global_to_local);
    std::vector<std::size_t>().swap(remote_cell_base_by_point);
    remote_cell_base_by_key.clear();
    remote_slot_by_key.clear();
    csr_extract_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase_start).count();
    phase_start = std::chrono::steady_clock::now();

    DistributedActiveExchange exchange;
    exchange.overlap_local_rows = runtime_options.overlap_local_rows;
    exchange.pair_omega_reduction = runtime_options.pair_omega_reduction;
    exchange.rank_profile_enabled = runtime_options.profile;
    std::vector<RemoteUnknownKey> const final_remote_unknowns =
        remote_unknowns;
    if(!initializeDistributedExchange(remote_unknowns,
                                      owned_solver_cell_bases,
                                      unknowns_per_cell,
                                      local_to_global.size(),
                                      exchange)) {
        setStepFailure(
            "individual radiation remote active/shadow-row requests are inconsistent");
        return reject();
    }
    owned_solver_cell_bases.clear();
    std::vector<RemoteUnknownKey>().swap(remote_unknowns);
    valid = unknowns_per_cell > 0 &&
        local_to_global.size() % unknowns_per_cell == 0;
    std::vector<std::size_t> block_cell_ids;
    if(valid) {
        block_cell_ids.reserve(local_to_global.size() / unknowns_per_cell);
        for(std::size_t row = 0; row < local_to_global.size();
            row += unknowns_per_cell) {
            std::size_t const cell =
                local_to_global[row] / unknowns_per_cell;
            if(cell >= cells.size()) {
                valid = false;
                break;
            }
            block_cell_ids.push_back(cells[cell].ID);
        }
    }
    if(!valid)
        setStepFailure(
            "distributed active radiation rows are not complete cell blocks");
    if(!collectiveAllTrue(valid))
        return reject();
    if(runtime_options.profile) {
        exchange_setup_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - phase_start).count();
        auto const profile_start = std::chrono::steady_clock::now();
        reportDistributedMemoryPhase("csr_exchange_ready");
        profile_overhead_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - profile_start).count();
        phase_start = std::chrono::steady_clock::now();
    }
    FixedPositiveRadiationScale const fixed_positive_scale =
        FixedCellPositiveRadiationScale(
            tess, full_initial, local_to_global, unknowns_per_cell,
            GetLengthScale());
    double const fixed_cell_maximum_absolute_Eg =
        FixedCellRadiationMaximumAbsoluteGroup(
            tess, full_initial, local_to_global, unknowns_per_cell);
    valid = fixed_positive_scale.finite &&
        std::isfinite(fixed_cell_maximum_absolute_Eg);
    if(!valid)
        setStepFailure(
            "fixed-cell radiation maximum is non-finite");
    if(!collectiveAllTrue(valid))
        return reject();
    exchange_setup_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase_start).count();
    phase_start = std::chrono::steady_clock::now();
    CG::HistoricalMGResidualCorrectionDiagnostics correction_diagnostics;
    bool const solver_converged = solveDistributedActiveBiCGSTAB(
        tolerance, total_iters, matrix, rhs, verification_rhs,
        verification_scale, base_solution, final_correction_volume,
        fixed_cell_maximum_absolute_Eg,
        fixed_positive_scale.maximum_cell_energy,
        fixed_positive_scale.total_energy,
        solution, exchange, unknowns_per_cell,
        GetPreconditionerKind(), block_cell_ids,
        *this, correction_diagnostics);
    solver_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase_start).count();
    phase_start = std::chrono::steady_clock::now();
    if(!solver_converged) {
        std::vector<double> failed_physical_solution = full_initial;
        for(std::size_t local = 0; local < local_to_global.size(); ++local)
            failed_physical_solution[local_to_global[local]] =
                base_solution[local] + solution[local];
        std::vector<double> failed_pre_correction_solution =
            failed_physical_solution;
        if(correction_diagnostics.available &&
           correction_diagnostics.pre_correction_solution.size() ==
               local_to_global.size())
            for(std::size_t local = 0; local < local_to_global.size(); ++local)
                failed_pre_correction_solution[local_to_global[local]] =
                    correction_diagnostics.pre_correction_solution[local];
        if(requestIndividualSolutionRetry(
               failed_pre_correction_solution, failed_physical_solution,
               local_to_global, cells,
               "distributed_active_failed_iterate")) {
            int const completed_iterations = total_iters;
            int retry_iterations = 0;
            // stepIndividual performs the mixed rebuild recursively.  Empty
            // every no-longer-needed solve buffer first; otherwise the retry
            // transiently retains both complete distributed systems and can
            // exhaust one node's memory on a full-variable production mesh.
            matrix.Release();
            std::vector<double>().swap(rhs);
            std::vector<double>().swap(verification_rhs);
            std::vector<double>().swap(verification_scale);
            std::vector<double>().swap(base_solution);
            std::vector<double>().swap(final_correction_volume);
            std::vector<double>().swap(solution);
            std::vector<double>().swap(full_initial);
            std::vector<std::size_t>().swap(local_to_global);
            std::vector<std::size_t>().swap(block_cell_ids);
            exchange = DistributedActiveExchange();
            correction_diagnostics =
                CG::HistoricalMGResidualCorrectionDiagnostics();
            std::vector<double>().swap(failed_pre_correction_solution);
            std::vector<double>().swap(failed_physical_solution);
            saved_extensives.resize(saved_extensives_size);
            cells = std::move(saved_cells);
            extensives = std::move(saved_extensives);
            rich_trim_after_rare_spike();
            bool const accepted = RadiationDriver::stepIndividual(
                tolerance, retry_iterations, tess, cells, extensives, context,
                interval_fraction, time, canonical_cells,
                canonical_extensives, owned_to_canonical);
            total_iters = completed_iterations + retry_iterations;
            return accepted;
        }
        if(!correction_diagnostics.failure_reason.empty()) {
            std::ostringstream reason;
            CG::AppendHistoricalMGResidualCorrectionFailureDiagnostics(
                reason, correction_diagnostics,
                static_cast<std::size_t>(std::max(total_iters, 0)));
            setCellLocalStepFailure(
                reason.str(), correction_diagnostics.failure_cell_id);
        }
        else
            setStepFailure(
                "distributed active-only BiCGSTAB failed or found a non-positive diagonal");
        return reject();
    }

    std::vector<double> full_solution = full_initial;
    valid = true;
    for(std::size_t local = 0; local < local_to_global.size(); ++local) {
        std::size_t const global_unknown = local_to_global[local];
        solution[local] = base_solution[local] + solution[local];
        if(!std::isfinite(solution[local])) {
            std::size_t const cell =
                global_unknown / unknowns_per_cell;
            std::ostringstream reason;
            reason << "active radiation solution is non-finite"
                   << " (value=" << solution[local]
                   << ", initial=" << full_initial[global_unknown]
                   << ", rhs=" << rhs[local] << ')';
            if(cell < cells.size())
                setCellLocalStepFailure(reason.str(), cells[cell].ID);
            else
                setStepFailure(reason.str());
            valid = false;
            continue;
        }
        full_solution[local_to_global[local]] = solution[local];
    }
    if(!collectiveAllTrue(valid))
        return reject();

    std::vector<double> pre_correction_full_solution = full_solution;
    if(correction_diagnostics.available &&
       correction_diagnostics.pre_correction_solution.size() ==
           local_to_global.size())
        for(std::size_t local = 0; local < local_to_global.size(); ++local)
            pre_correction_full_solution[local_to_global[local]] =
                correction_diagnostics.pre_correction_solution[local];
    bool retry_requested = requestIndividualSolutionRetry(
        pre_correction_full_solution, full_solution, local_to_global, cells,
        "distributed_active");
    int collective_retry_requested = retry_requested ? 1 : 0;
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &collective_retry_requested, 1, MPI_INT,
                      MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(distributed solution retry)");
    retry_requested = collective_retry_requested != 0;
    if(retry_requested) {
        int const completed_iterations = total_iters;
        int retry_iterations = 0;
        // The mixed Compton rebuild is recursive.  Release the completed
        // first system before entering it so two full distributed-active
        // matrices, preconditioners, and Krylov workspaces cannot overlap.
        matrix.Release();
        std::vector<double>().swap(rhs);
        std::vector<double>().swap(verification_rhs);
        std::vector<double>().swap(verification_scale);
        std::vector<double>().swap(base_solution);
        std::vector<double>().swap(final_correction_volume);
        std::vector<double>().swap(solution);
        std::vector<double>().swap(full_initial);
        std::vector<std::size_t>().swap(local_to_global);
        std::vector<std::size_t>().swap(block_cell_ids);
        exchange = DistributedActiveExchange();
        correction_diagnostics =
            CG::HistoricalMGResidualCorrectionDiagnostics();
        std::vector<double>().swap(pre_correction_full_solution);
        std::vector<double>().swap(full_solution);
        saved_extensives.resize(saved_extensives_size);
        cells = std::move(saved_cells);
        extensives = std::move(saved_extensives);
        rich_trim_after_rare_spike();
        bool const accepted = RadiationDriver::stepIndividual(
            tolerance, retry_iterations, tess, cells, extensives, context,
            interval_fraction, time, canonical_cells, canonical_extensives,
            owned_to_canonical);
        total_iters = completed_iterations + retry_iterations;
        return accepted;
    }

    // PostCG needs the solved passive endpoint on the active rank for its
    // face-gradient/radiation-force work.  Krylov owns only local rows, so
    // perform one final halo exchange of the physical (base + correction)
    // solution and scatter remote stable-ID values back into full_solution.
    if(runtime_options.usesShadowPassiveRows()) {
        ActiveBiCGSTABTiming final_exchange_timing;
        valid = exchangeRemoteActiveValues(
            exchange, solution, exchange.remote_values,
            final_exchange_timing);
        std::map<std::size_t, std::vector<std::size_t> >
            local_indices_by_stable_id;
        for(std::size_t cell = 0; cell < cells.size(); ++cell)
            local_indices_by_stable_id[cells[cell].ID].push_back(cell);
        if(final_remote_unknowns.size() !=
           exchange.remote_values.size())
            valid = false;
        std::size_t const remote_count = std::min(
            final_remote_unknowns.size(),
            exchange.remote_values.size());
        for(std::size_t slot = 0; slot < remote_count; ++slot) {
            RemoteUnknownKey const& key = final_remote_unknowns[slot];
            auto const indices =
                local_indices_by_stable_id.find(key.cell_id);
            if(indices == local_indices_by_stable_id.end() ||
               key.group >= unknowns_per_cell ||
               !std::isfinite(exchange.remote_values[slot])) {
                valid = false;
                continue;
            }
            for(std::size_t const cell : indices->second) {
                std::size_t unknown = 0;
                if(!checkedDistributedSizeMultiply(
                       cell, unknowns_per_cell, unknown) ||
                   !checkedDistributedSizeAdd(
                       unknown, key.group, unknown) ||
                   unknown >= full_solution.size()) {
                    valid = false;
                    continue;
                }
                full_solution[unknown] =
                    exchange.remote_values[slot];
            }
        }
        if(!valid)
            setStepFailure(
                "shadow passive radiation final halo is inconsistent");
        if(!collectiveAllTrue(valid))
            return reject();
    }

    try {
        PostCG(tess, extensives, 0, cells, full_solution, full_solution);
    }
    catch(UniversalError const& error) {
        setStepFailure(error.getErrorMessage());
        valid = false;
    }
    catch(std::exception const& error) {
        setStepFailure(error.what());
        valid = false;
    }
    catch(...) {
        setStepFailure("unknown exception while committing active radiation rows");
        valid = false;
    }
    if(!collectiveAllTrue(valid))
        return reject();

    std::map<std::size_t, std::size_t> owned_index_by_id;
    for(std::size_t cell = 0; cell < cells.size(); ++cell) {
        if(cell >= point_owner.size() || point_owner[cell] != rank)
            continue;
        if(cell < tess.GetPointNo()) {
            if(tess.IsPointOutsideBox(cell))
                continue;
        }
        else {
            auto const shadow = owned_shadow_cells.find(cells[cell].ID);
            if(shadow == owned_shadow_cells.end() ||
               shadow->second.cell_index != cell)
                continue;
        }
        auto inserted = owned_index_by_id.emplace(cells[cell].ID, cell);
        if(!inserted.second && inserted.first->second >= tess.GetPointNo() &&
           cell < tess.GetPointNo())
            inserted.first->second = cell;
    }

    // An active face may target an inactive owned cell omitted from this
    // rank's partial mesh.  Keep those passive updates in the same candidate
    // transaction, keyed by the canonical stable ID, without refreshing their
    // primitive state.
    std::map<std::size_t, std::size_t> canonical_index_by_id;
    std::vector<std::pair<std::size_t, std::size_t> > canonical_work_indices;
    if((canonical_cells == nullptr) != (canonical_extensives == nullptr))
        valid = false;
    if(canonical_cells != nullptr && canonical_extensives != nullptr) {
        if(canonical_cells->size() != canonical_extensives->size())
            valid = false;
        else {
            for(std::size_t canonical = 0;
                canonical < canonical_cells->size(); ++canonical)
                if(!canonical_index_by_id.emplace(
                       (*canonical_cells)[canonical].ID, canonical).second)
                    valid = false;
            for(auto const& entry : owned_index_by_id) {
                auto const canonical = canonical_index_by_id.find(entry.first);
                if(canonical == canonical_index_by_id.end()) {
                    valid = false;
                    continue;
                }
                canonical_work_indices.emplace_back(
                    entry.second, canonical->second);
            }
        }
    }
    if(!valid)
        setStepFailure(
            "individual radiation canonical stable-cell mapping is inconsistent");
    if(!collectiveAllTrue(valid))
        return reject();

    // Local and remote active faces can both reach a canonical owned passive
    // cell omitted from the partial mesh.  Load it into the same transaction.
    auto const find_or_load_passive_cell = [&](std::size_t const cell_id)
    {
        auto found = owned_index_by_id.find(cell_id);
        if(found == owned_index_by_id.end() &&
           canonical_extensives != nullptr) {
            auto const canonical = canonical_index_by_id.find(cell_id);
            if(canonical != canonical_index_by_id.end() &&
               cells.size() == extensives.size() &&
               extensives.size() == transaction_start_extensives.size()) {
                std::size_t const work_index = cells.size();
                cells.push_back((*canonical_cells)[canonical->second]);
                extensives.push_back(
                    (*canonical_extensives)[canonical->second]);
                transaction_start_extensives.push_back(
                    (*canonical_extensives)[canonical->second]);
                found = owned_index_by_id.emplace(cell_id, work_index).first;
                canonical_work_indices.emplace_back(
                    work_index, canonical->second);
            }
        }
        return found;
    };

    struct PendingPassiveTransfer
    {
        std::size_t passive = 0;
        std::size_t active_cell_id = 0;
        std::size_t group = 0;
        int active_owner = -1;
        double proposed_gain = 0;
        double applied_gain = 0;
        double reference_time_step = 0;
    };
    std::vector<PendingPassiveTransfer> passive_transfers;
    std::vector<std::vector<IndividualRadiationDelta> > outgoing(rank_count);
    double const extensive_conversion =
        time_scale_ * time_scale_ /
        (length_scale_ * length_scale_ * mass_scale_);
    for(IndividualFaceCoefficient const& face : individual_face_coefficients_) {
        if(runtime_options.usesShadowPassiveRows())
            continue;
        if(face.left >= cells.size() || face.right >= cells.size() ||
           face.left >= context.active_mask.size() ||
           face.right >= context.active_mask.size() ||
           face.group >= unknowns_per_cell) {
            valid = false;
            continue;
        }
        bool const left_active = context.isActive(face.left);
        bool const right_active = context.isActive(face.right);
        if(left_active == right_active)
            continue;
        std::size_t const active = left_active ? face.left : face.right;
        std::size_t const passive = left_active ? face.right : face.left;
        if(active >= point_owner.size() || point_owner[active] != rank ||
           passive >= point_owner.size()) {
            valid = false;
            continue;
        }
        std::size_t active_unknown = 0;
        std::size_t passive_unknown = 0;
        if(!checkedDistributedSizeMultiply(
               active, unknowns_per_cell, active_unknown) ||
           !checkedDistributedSizeAdd(
               active_unknown, face.group, active_unknown) ||
           !checkedDistributedSizeMultiply(
               passive, unknowns_per_cell, passive_unknown) ||
           !checkedDistributedSizeAdd(
               passive_unknown, face.group, passive_unknown)) {
            valid = false;
            continue;
        }
        if(active_unknown >= full_solution.size() ||
           passive_unknown >= full_solution.size()) {
            valid = false;
            continue;
        }
        double const passive_gain = face.coefficient *
            (full_solution[active_unknown] -
             full_solution[passive_unknown]) *
            extensive_conversion;
        if(!std::isfinite(passive_gain) ||
           !(face.time_step > 0) || !std::isfinite(face.time_step)) {
            valid = false;
            continue;
        }
        int const owner = point_owner[passive];
        if(owner == rank) {
            auto const found = find_or_load_passive_cell(cells[passive].ID);
            if(found == owned_index_by_id.end()) {
                valid = false;
                continue;
            }
            PendingPassiveTransfer transfer;
            transfer.passive = found->second;
            transfer.active_cell_id = cells[active].ID;
            transfer.group = face.group;
            transfer.active_owner = rank;
            transfer.proposed_gain = passive_gain;
            transfer.applied_gain = passive_gain;
            transfer.reference_time_step = face.time_step;
            passive_transfers.push_back(transfer);
        }
        else if(owner >= 0 && owner < rank_count) {
            IndividualRadiationDelta packet;
            packet.cell_id = cells[passive].ID;
            packet.counterpart_cell_id = cells[active].ID;
            packet.group = face.group;
            packet.gain = passive_gain;
            packet.reference_time_step = face.time_step;
            outgoing[owner].push_back(packet);
        }
        else
            valid = false;
    }
    if(!valid)
        setStepFailure("individual radiation passive-face ownership is inconsistent");
    if(!collectiveAllTrue(valid))
        return reject();

    std::vector<std::vector<IndividualRadiationDelta> > const incoming =
        MPI_Exchange_all_to_all(outgoing, MPI_COMM_WORLD);
    valid = incoming.size() == static_cast<std::size_t>(rank_count);
    for(std::size_t peer = 0; peer < incoming.size(); ++peer)
        for(IndividualRadiationDelta const& packet : incoming[peer]) {
            auto const found = find_or_load_passive_cell(packet.cell_id);
            if(found == owned_index_by_id.end() ||
               packet.group >= unknowns_per_cell ||
               !std::isfinite(packet.gain) ||
               !(packet.reference_time_step > 0) ||
               !std::isfinite(packet.reference_time_step)) {
                valid = false;
                continue;
            }
            PendingPassiveTransfer transfer;
            transfer.passive = found->second;
            transfer.active_cell_id = packet.counterpart_cell_id;
            transfer.group = packet.group;
            transfer.active_owner = static_cast<int>(peer);
            transfer.proposed_gain = packet.gain;
            transfer.applied_gain = packet.gain;
            transfer.reference_time_step = packet.reference_time_step;
            passive_transfers.push_back(transfer);
        }
    for(PendingPassiveTransfer const& transfer : passive_transfers) {
        std::size_t const cell = transfer.passive;
        std::size_t const group = transfer.group;
        if(cell >= extensives.size() ||
           transfer.active_owner < 0 || transfer.active_owner >= rank_count ||
           (unknowns_per_cell > 1 &&
            group >= extensives[cell].Eg.size()))
            valid = false;
    }
    if(!valid)
        setStepFailure("individual radiation received a passive delta for an unknown stable cell ID");
    if(!collectiveAllTrue(valid))
        return reject();

    if(runtime_options.usesFrozenDirichlet()) {
        double local_candidate_start_positive_extent = 0;
        AccurateResidualAccumulator positive_extent_accumulator;
        // The normalization scale is a global physical inventory, not an
        // active-mesh inventory.  Use every canonical owned cell so AMR
        // closure and active/passive topology cannot change an otherwise
        // identical candidate's defect fraction.
        std::vector<Conserved3D> const& scale_extensives =
            canonical_extensives != nullptr ?
            *canonical_extensives : transaction_start_extensives;
        for(Conserved3D const& extensive : scale_extensives) {
            if(unknowns_per_cell == 1) {
                double const extent = extensive.Erad;
                if(!std::isfinite(extent)) {
                    valid = false;
                    continue;
                }
                if(extent > 0)
                    positive_extent_accumulator.Add(extent);
            }
            else {
                for(double const extent : extensive.Eg) {
                    if(!std::isfinite(extent)) {
                        valid = false;
                        continue;
                    }
                    if(extent > 0)
                        positive_extent_accumulator.Add(extent);
                }
            }
        }
        local_candidate_start_positive_extent =
            positive_extent_accumulator.Value();
        double const local_rhs_floor =
            1024 * std::numeric_limits<double>::epsilon() *
            extensive_conversion * local_defect_rhs_magnitude;
        pending_defect_event.normalization_scale =
            collectiveIndividualRadiationDefectScale(
                local_candidate_start_positive_extent, local_rhs_floor,
                pending_defect_event.
                    candidate_start_positive_global_extent,
                pending_defect_event.rhs_derived_global_floor);
        pending_defect_event.valid = valid &&
            std::isfinite(extensive_conversion) && extensive_conversion > 0 &&
            std::isfinite(pending_defect_event.normalization_scale) &&
            pending_defect_event.normalization_scale > 0;

        struct PassiveGroupDefect
        {
            AccurateResidualAccumulator withdrawal;
            AccurateResidualAccumulator deposit;
            std::uint64_t face_group_terms = 0;
            double representative_withdrawal_term = 0;
            std::uint64_t representative_active_id =
                std::numeric_limits<std::uint64_t>::max();
            int representative_active_owner = -1;
        };
        std::map<std::pair<std::size_t, std::size_t>, PassiveGroupDefect>
            grouped_defects;
        AccurateResidualAccumulator signed_defect;
        AccurateResidualAccumulator absolute_defect;
        AccurateResidualAccumulator total_withdrawal;
        AccurateResidualAccumulator total_deposit;
        std::set<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t> >
            stable_face_group_keys;
        bool have_duplicate_face_group = false;
        std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                   std::uint64_t, std::uint64_t> duplicate_face_group;
        for(PendingPassiveTransfer const& transfer : passive_transfers) {
            double const defect = -transfer.proposed_gain;
            if(!std::isfinite(defect) ||
               transfer.passive >= cells.size() ||
               transfer.passive >= transaction_start_extensives.size()) {
                pending_defect_event.valid = false;
                continue;
            }
            std::uint64_t const active_id =
                static_cast<std::uint64_t>(transfer.active_cell_id);
            std::uint64_t const passive_id =
                static_cast<std::uint64_t>(cells[transfer.passive].ID);
            std::tuple<std::uint64_t, std::uint64_t, std::uint64_t> const
                stable_key(std::min(active_id, passive_id),
                           std::max(active_id, passive_id),
                           static_cast<std::uint64_t>(transfer.group));
            if(!stable_face_group_keys.insert(stable_key).second) {
                ++pending_defect_event.duplicate_face_group_terms;
                std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                           std::uint64_t, std::uint64_t> const offender(
                    std::get<0>(stable_key), std::get<1>(stable_key),
                    std::get<2>(stable_key), active_id, passive_id);
                if(!have_duplicate_face_group ||
                   offender < duplicate_face_group) {
                    have_duplicate_face_group = true;
                    duplicate_face_group = offender;
                }
            }
            signed_defect.Add(defect);
            absolute_defect.Add(std::abs(defect));
            PassiveGroupDefect& group = grouped_defects[
                std::make_pair(transfer.passive, transfer.group)];
            if(group.face_group_terms ==
               std::numeric_limits<std::uint64_t>::max())
                pending_defect_event.valid = false;
            else
                ++group.face_group_terms;
            if(defect >= 0) {
                group.withdrawal.Add(defect);
                total_withdrawal.Add(defect);
                std::uint64_t const active_id =
                    static_cast<std::uint64_t>(transfer.active_cell_id);
                std::tuple<double, std::uint64_t, int> const candidate(
                    -defect, active_id, transfer.active_owner);
                std::tuple<double, std::uint64_t, int> const current(
                    -group.representative_withdrawal_term,
                    group.representative_active_id,
                    group.representative_active_owner);
                if(candidate < current) {
                    group.representative_withdrawal_term = defect;
                    group.representative_active_id = active_id;
                    group.representative_active_owner =
                        transfer.active_owner;
                }
            }
            else {
                group.deposit.Add(-defect);
                total_deposit.Add(-defect);
            }
            if(pending_defect_event.face_group_terms ==
               std::numeric_limits<std::uint64_t>::max())
                pending_defect_event.valid = false;
            else
                ++pending_defect_event.face_group_terms;
        }
        pending_defect_event.signed_extent = signed_defect.Value();
        pending_defect_event.absolute_extent = absolute_defect.Value();
        pending_defect_event.passive_withdrawal_extent =
            total_withdrawal.Value();
        pending_defect_event.passive_deposit_extent =
            total_deposit.Value();
        double const local_roundoff_floor =
            1024 * std::numeric_limits<double>::epsilon() *
            pending_defect_event.normalization_scale;
        bool have_representative = false;
        for(auto const& entry : grouped_defects) {
            std::size_t const cell = entry.first.first;
            std::size_t const group = entry.first.second;
            if(cell >= transaction_start_extensives.size() ||
               cell >= cells.size() ||
               (unknowns_per_cell > 1 &&
                group >= transaction_start_extensives[cell].Eg.size())) {
                pending_defect_event.valid = false;
                continue;
            }
            double const passive_extent = unknowns_per_cell == 1 ?
                transaction_start_extensives[cell].Erad :
                transaction_start_extensives[cell].Eg[group];
            double const withdrawal = entry.second.withdrawal.Value();
            if(!std::isfinite(passive_extent) || passive_extent < 0 ||
               !std::isfinite(withdrawal) || withdrawal < 0 ||
               !std::isfinite(local_roundoff_floor) ||
               local_roundoff_floor <= 0) {
                pending_defect_event.valid = false;
                continue;
            }
            IndividualRadiationLocalDefectMeasure const local_measure =
                measureIndividualRadiationLocalDefect(
                    withdrawal, passive_extent, local_roundoff_floor,
                    pending_defect_event.normalization_scale);
            std::uint64_t const active_id =
                entry.second.representative_active_id;
            std::uint64_t const passive_id =
                static_cast<std::uint64_t>(cells[cell].ID);
            std::tuple<double, std::uint64_t, std::uint64_t,
                       std::uint64_t> const candidate(
                -local_measure.tolerance_ratio, active_id, passive_id,
                static_cast<std::uint64_t>(group));
            std::tuple<double, std::uint64_t, std::uint64_t,
                       std::uint64_t> const current(
                -pending_defect_event.maximum_local_tolerance_ratio,
                pending_defect_event.representative_active_id,
                pending_defect_event.representative_passive_id,
                pending_defect_event.representative_group);
            if(!std::isfinite(local_measure.relative_fraction) ||
               !std::isfinite(local_measure.allowed_withdrawal) ||
               local_measure.allowed_withdrawal <= 0 ||
               !std::isfinite(local_measure.tolerance_ratio) ||
               local_measure.tolerance_ratio < 0)
                pending_defect_event.valid = false;
            else if(!have_representative || candidate < current) {
                have_representative = true;
                pending_defect_event.maximum_local_tolerance_ratio =
                    local_measure.tolerance_ratio;
                pending_defect_event.representative_active_id = active_id;
                pending_defect_event.representative_passive_id = passive_id;
                pending_defect_event.representative_group = group;
                pending_defect_event.representative_active_rank =
                    entry.second.representative_active_owner >= 0 ?
                    static_cast<std::uint64_t>(
                        entry.second.representative_active_owner) :
                    std::numeric_limits<std::uint64_t>::max();
            }
            if(std::isfinite(local_measure.relative_fraction))
                pending_defect_event.maximum_local_fraction = std::max(
                    pending_defect_event.maximum_local_fraction,
                    local_measure.relative_fraction);
        }
        if(have_duplicate_face_group) {
            pending_defect_event.valid = false;
            pending_defect_event.representative_active_id =
                std::get<3>(duplicate_face_group);
            pending_defect_event.representative_passive_id =
                std::get<4>(duplicate_face_group);
            pending_defect_event.representative_group =
                std::get<2>(duplicate_face_group);
        }
        if(!validateIndividualRadiationDefect(pending_defect_event))
            return reject();
        IndividualRadiationDefectConfiguration const& defect_configuration =
            individualRadiationDefectConfiguration();
        bool const synchronize_passive_neighbors =
            pending_defect_event.maximum_local_tolerance_ratio > 1 ||
            pending_defect_event.event_absolute_fraction >
                defect_configuration.event_absolute_target;
        if(synchronize_passive_neighbors)
            for(PendingPassiveTransfer const& transfer : passive_transfers) {
                if(transfer.proposed_gain == 0 ||
                   transfer.passive >= cells.size())
                    continue;
                double const reference_time_step =
                    std::isfinite(transfer.reference_time_step) &&
                    transfer.reference_time_step > 0 ?
                    transfer.reference_time_step : context.time_quantum;
                auto const inserted = pending_dirichlet_wakes.emplace(
                    transfer.passive, reference_time_step);
                if(!inserted.second)
                    inserted.first->second = std::min(
                        inserted.first->second, reference_time_step);
            }
        pending_dirichlet_defect = true;
        for(std::size_t cell = 0;
            cell < cells.size() &&
            cell < transaction_start_extensives.size(); ++cell) {
            bool const active = cell < context.active_mask.size() &&
                context.isActive(cell);
            if(active)
                continue;
            extensives[cell] = transaction_start_extensives[cell];
            if(cell < saved_cells.size())
                cells[cell] = saved_cells[cell];
        }
        // Frozen Dirichlet semantics intentionally omit the equal-and-opposite
        // passive commit.  The globally reduced pending record above measures
        // that exact omission; the remainder of the legacy limiter sees no
        // passive transfers and therefore cannot mutate passive state.
        passive_transfers.clear();
    }

    if(unknowns_per_cell == 0 ||
       cells.size() > std::numeric_limits<std::size_t>::max() /
           unknowns_per_cell) {
        setStepFailure("individual radiation passive limiter size overflow");
        valid = false;
    }
    if(!collectiveAllTrue(valid))
        return reject();
    std::size_t const passive_value_count =
        cells.size() * unknowns_per_cell;
    std::vector<double> positive_gain(passive_value_count, 0);
    std::vector<double> negative_loss(passive_value_count, 0);
    std::vector<double> negative_scale(passive_value_count, 1);
    std::vector<unsigned char> touched_value(passive_value_count, 0);
    std::vector<std::size_t> negative_transfer_index(
        passive_value_count, std::numeric_limits<std::size_t>::max());
    unsigned long long local_passive_roundoff_values = 0;
    double local_passive_roundoff_max_relative = 0;
    std::size_t local_passive_roundoff_cell =
        std::numeric_limits<std::size_t>::max();
    std::size_t local_passive_roundoff_group =
        std::numeric_limits<std::size_t>::max();
    double local_passive_roundoff_extent = 0;
    double local_passive_roundoff_scale = 0;
    for(std::size_t transfer_index = 0;
        transfer_index < passive_transfers.size(); ++transfer_index) {
        PendingPassiveTransfer const& transfer =
            passive_transfers[transfer_index];
        std::size_t const key =
            transfer.passive * unknowns_per_cell + transfer.group;
        touched_value[key] = 1;
        if(transfer.proposed_gain >= 0)
            positive_gain[key] += transfer.proposed_gain;
        else {
            negative_loss[key] -= transfer.proposed_gain;
            if(negative_transfer_index[key] ==
               std::numeric_limits<std::size_t>::max())
                negative_transfer_index[key] = transfer_index;
        }
    }
    for(std::size_t cell = 0; cell < cells.size(); ++cell)
        for(std::size_t group = 0; group < unknowns_per_cell; ++group) {
            std::size_t const key = cell * unknowns_per_cell + group;
            if(!touched_value[key])
                continue;
            double const available = unknowns_per_cell == 1
                ? transaction_start_extensives[cell].Erad
                : transaction_start_extensives[cell].Eg[group];
            RadiationPositivity::PassiveRoundoffResult const roundoff =
                RadiationPositivity::ClassifyPassiveRoundoff(
                    available, positive_gain[key], negative_loss[key]);
            if(!roundoff.valid ||
               (available < 0 && !roundoff.repairable)) {
                std::ostringstream reason;
                reason << std::setprecision(17)
                       << "passive radiation extent was invalid before face limiting"
                       << ": group=" << group
                       << " extent=" << available
                       << " positive_gain=" << positive_gain[key]
                       << " negative_loss=" << negative_loss[key]
                       << " roundoff_tolerance=" << roundoff.tolerance
                       << " mass=" << transaction_start_extensives[cell].mass;
                setCellLocalStepFailure(reason.str(), cells[cell].ID);
                valid = false;
                continue;
            }
            if(roundoff.repairable) {
                ++local_passive_roundoff_values;
                if(roundoff.relative_extent >
                   local_passive_roundoff_max_relative) {
                    local_passive_roundoff_max_relative =
                        roundoff.relative_extent;
                    local_passive_roundoff_cell = cells[cell].ID;
                    local_passive_roundoff_group = group;
                    local_passive_roundoff_extent = available;
                    local_passive_roundoff_scale = roundoff.face_scale;
                }
            }
            double const capacity = std::max(0.0, available) +
                positive_gain[key];
            if(negative_loss[key] > capacity) {
                double const ratio = capacity > 0
                    ? capacity / negative_loss[key] : 0;
                negative_scale[key] = ratio > 0
                    ? std::nextafter(std::min(1.0, ratio), 0.0) : 0;
            }
        }
    if(!collectiveAllTrue(valid))
        return reject();

    std::vector<unsigned char> touched_passive(cells.size(), 0);
    std::vector<unsigned char> corrected_active(cells.size(), 0);
    std::vector<std::tuple<std::size_t, std::size_t, double> > local_corrections;
    std::vector<std::vector<IndividualRadiationDelta> >
        correction_outgoing(rank_count);
    auto queue_active_correction = [&](PendingPassiveTransfer const& transfer,
                                       double correction) {
        if(correction == 0)
            return;
        if(transfer.active_owner == rank) {
            auto const found =
                owned_index_by_id.find(transfer.active_cell_id);
            if(found == owned_index_by_id.end()) {
                valid = false;
                return;
            }
            local_corrections.emplace_back(
                found->second, transfer.group, correction);
        }
        else {
            IndividualRadiationDelta packet;
            packet.cell_id = transfer.active_cell_id;
            packet.group = transfer.group;
            packet.gain = correction;
            correction_outgoing[transfer.active_owner].push_back(packet);
        }
    };
    for(PendingPassiveTransfer& transfer : passive_transfers) {
        std::size_t const key =
            transfer.passive * unknowns_per_cell + transfer.group;
        if(transfer.proposed_gain < 0)
            transfer.applied_gain =
                transfer.proposed_gain * negative_scale[key];
        touched_passive[transfer.passive] = 1;
        extensives[transfer.passive].Erad += transfer.applied_gain;
        if(unknowns_per_cell > 1)
            extensives[transfer.passive].Eg[transfer.group] +=
                transfer.applied_gain;

        double const correction =
            transfer.proposed_gain - transfer.applied_gain;
        queue_active_correction(transfer, correction);
    }
    for(std::size_t cell = 0; cell < cells.size(); ++cell)
        for(std::size_t group = 0; group < unknowns_per_cell; ++group) {
            std::size_t const key = cell * unknowns_per_cell + group;
            if(!touched_value[key])
                continue;
            double& passive_energy = unknowns_per_cell == 1
                ? extensives[cell].Erad : extensives[cell].Eg[group];
            if(!std::isfinite(passive_energy)) {
                valid = false;
                continue;
            }
            if(passive_energy >= 0)
                continue;
            std::size_t const transfer_index = negative_transfer_index[key];
            if(transfer_index == std::numeric_limits<std::size_t>::max()) {
                valid = false;
                continue;
            }
            double const residual = -passive_energy;
            passive_energy = 0;
            if(unknowns_per_cell > 1)
                extensives[cell].Erad += residual;
            queue_active_correction(
                passive_transfers[transfer_index], -residual);
    }
    if(!valid)
        setStepFailure(
            "individual radiation could not map a local passive-flux correction");
    if(!collectiveAllTrue(valid))
        return reject();

    std::vector<std::vector<IndividualRadiationDelta> > const
        correction_incoming =
            MPI_Exchange_all_to_all(correction_outgoing, MPI_COMM_WORLD);
    valid = correction_incoming.size() ==
        static_cast<std::size_t>(rank_count);
    for(auto const& peer_packets : correction_incoming)
        for(IndividualRadiationDelta const& packet : peer_packets) {
            auto const found = owned_index_by_id.find(packet.cell_id);
            if(found == owned_index_by_id.end() ||
               packet.group >= unknowns_per_cell ||
               !std::isfinite(packet.gain)) {
                valid = false;
                continue;
            }
            local_corrections.emplace_back(
                found->second, packet.group, packet.gain);
        }
    if(!valid)
        setStepFailure(
            "individual radiation received an invalid passive-flux correction");
    if(!collectiveAllTrue(valid))
        return reject();

    std::map<std::size_t, ActivePassiveCorrectionLedger>
        active_passive_correction_ledger;
    for(auto const& correction : local_corrections) {
        std::size_t const cell = std::get<0>(correction);
        std::size_t const group = std::get<1>(correction);
        double const gain = std::get<2>(correction);
        auto const inserted = active_passive_correction_ledger.emplace(
            cell, ActivePassiveCorrectionLedger());
        ActivePassiveCorrectionLedger& ledger = inserted.first->second;
        if(inserted.second)
            ledger.radiation_before = extensives[cell].Erad;
        ledger.correction_sum += gain;
        ++ledger.terms;
        if(gain < ledger.most_negative_term) {
            ledger.most_negative_term = gain;
            ledger.most_negative_group = group;
        }
        corrected_active[cell] = 1;
        extensives[cell].Erad += gain;
        if(unknowns_per_cell > 1)
            extensives[cell].Eg[group] += gain;
    }
    if(runtime_options.usesShadowPassiveRows()) {
        double const shadow_extensive_conversion =
            time_scale_ * time_scale_ /
            (length_scale_ * length_scale_ * mass_scale_);
        for(auto const& entry : owned_shadow_cells) {
            OwnedShadowCell const& shadow = entry.second;
            if(shadow.cell_index >= extensives.size() ||
               shadow.canonical_index >= canonical_extensives->size() ||
               shadow.cell_index >= touched_passive.size() ||
               extensives[shadow.cell_index].Eg.size() <
                   unknowns_per_cell) {
                valid = false;
                continue;
            }
            Conserved3D const& initial_extent =
                (*canonical_extensives)[shadow.canonical_index];
            if(initial_extent.Eg.size() < unknowns_per_cell) {
                valid = false;
                continue;
            }
            long double total_extent = 0;
            for(std::size_t group = 0;
                group < unknowns_per_cell; ++group) {
                std::size_t passive_unknown = 0;
                if(!checkedDistributedSizeMultiply(
                       shadow.cell_index, unknowns_per_cell,
                       passive_unknown) ||
                   !checkedDistributedSizeAdd(
                       passive_unknown, group,
                       passive_unknown) ||
                   passive_unknown >= full_solution.size()) {
                    valid = false;
                    continue;
                }
                double const passive_value =
                    full_solution[passive_unknown];
                if(!std::isfinite(passive_value) ||
                   passive_value < 0) {
                    std::ostringstream reason;
                    reason << std::setprecision(17)
                           << "shadow passive radiation solution is negative"
                           << " group=" << group
                           << " value=" << passive_value;
                    setCellLocalStepFailure(
                        reason.str(), entry.first);
                    valid = false;
                    continue;
                }
                double const endpoint_extent =
                    shadow_extensive_conversion *
                    shadow.volume_cgs * passive_value;
                long double transfer_extent =
                    static_cast<long double>(
                        initial_extent.Eg[group]);
                for(IndividualShadowFaceRecord const& face :
                    shadow.faces) {
                    if(face.group != group)
                        continue;
                    auto const active_index =
                        solver_cell_index_by_owner_id.find(
                            std::make_pair(
                                face.active_owner,
                                face.active_cell_id));
                    if(active_index ==
                       solver_cell_index_by_owner_id.end()) {
                        valid = false;
                        continue;
                    }
                    std::size_t active_unknown = 0;
                    if(!checkedDistributedSizeMultiply(
                           active_index->second,
                           unknowns_per_cell, active_unknown) ||
                       !checkedDistributedSizeAdd(
                           active_unknown, group,
                           active_unknown) ||
                       active_unknown >= full_solution.size()) {
                        valid = false;
                        continue;
                    }
                    double const active_value =
                        full_solution[active_unknown];
                    if(!std::isfinite(active_value)) {
                        valid = false;
                        continue;
                    }
                    transfer_extent +=
                        static_cast<long double>(
                            shadow_extensive_conversion) *
                        static_cast<long double>(face.coefficient) *
                        static_cast<long double>(
                            active_value - passive_value);
                }
                double const transfer_extent_double =
                    static_cast<double>(transfer_extent);
                double const residual_scale = std::max(
                    {1.0, std::abs(endpoint_extent),
                     std::abs(transfer_extent_double),
                     std::abs(initial_extent.Eg[group])});
                // The historical multigroup tolerance is a squared
                // diagonal-scaled norm.  Compare this linear row residual
                // against its declared effective norm tolerance.
                double const effective_norm_tolerance =
                    tolerance > 0 ? std::sqrt(tolerance) : 0;
                double const residual_tolerance =
                    std::max(
                        1024 *
                            std::numeric_limits<double>::epsilon(),
                        effective_norm_tolerance) *
                    residual_scale;
                if(!std::isfinite(endpoint_extent) ||
                   endpoint_extent < 0 ||
                   !std::isfinite(transfer_extent_double) ||
                   transfer_extent_double < 0 ||
                   std::abs(
                       endpoint_extent -
                       transfer_extent_double) >
                       residual_tolerance) {
                    std::ostringstream reason;
                    reason << std::setprecision(17)
                           << "shadow passive radiation row residual failed"
                           << " group=" << group
                           << " endpoint=" << endpoint_extent
                           << " transfer=" << transfer_extent_double
                           << " tolerance=" << residual_tolerance;
                    setCellLocalStepFailure(
                        reason.str(), entry.first);
                    valid = false;
                    continue;
                }
                // Store the compensated transfer sum, not V*E_p.  The latter
                // differs by the accepted Krylov row residual; the transfer
                // form is exactly paired with the active face contribution.
                extensives[shadow.cell_index].Eg[group] =
                    transfer_extent_double;
                total_extent +=
                    static_cast<long double>(transfer_extent_double);
            }
            double const total_extent_double =
                static_cast<double>(total_extent);
            if(!std::isfinite(total_extent_double) ||
               total_extent_double < 0) {
                valid = false;
                continue;
            }
            extensives[shadow.cell_index].Erad =
                total_extent_double;
            touched_passive[shadow.cell_index] = 1;
        }
        if(!valid && getLastStepFailureReason().empty())
            setStepFailure(
                "shadow passive radiation commit is invalid");
        if(!collectiveAllTrue(valid))
            return reject();
    }

    double global_maximum_cell_radiation_extent = 0;
    if(canonical_extensives != nullptr) {
        std::vector<unsigned char> canonical_overridden(
            canonical_extensives->size(), 0);
        for(auto const& mapping : canonical_work_indices) {
            if(mapping.first >= extensives.size() ||
               mapping.second >= canonical_extensives->size()) {
                valid = false;
                continue;
            }
            canonical_overridden[mapping.second] = 1;
            if(std::isfinite(extensives[mapping.first].Erad))
                global_maximum_cell_radiation_extent = std::max(
                    global_maximum_cell_radiation_extent,
                    extensives[mapping.first].Erad);
        }
        for(std::size_t canonical = 0;
            canonical < canonical_extensives->size(); ++canonical)
            if(!canonical_overridden[canonical] &&
               std::isfinite((*canonical_extensives)[canonical].Erad))
                global_maximum_cell_radiation_extent = std::max(
                    global_maximum_cell_radiation_extent,
                    (*canonical_extensives)[canonical].Erad);
    }
    else {
        std::size_t const owned_extent_count = std::min<std::size_t>(
            tess.GetPointNo(), extensives.size());
        for(std::size_t cell = 0; cell < owned_extent_count; ++cell)
            if(std::isfinite(extensives[cell].Erad))
                global_maximum_cell_radiation_extent = std::max(
                    global_maximum_cell_radiation_extent,
                    extensives[cell].Erad);
    }
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE,
                      &global_maximum_cell_radiation_extent, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
        "MPI_Allreduce(distributed radiation extent)");

    if(unknowns_per_cell > 1) {
        std::vector<unsigned char> repair_target = touched_passive;
        repair_target.resize(extensives.size(), 0);
        for(std::size_t const cell : context.active_indices)
            if(cell < repair_target.size())
                repair_target[cell] = 1;
            else
                valid = false;
        for(std::size_t cell = 0; cell < repair_target.size(); ++cell) {
            if(!repair_target[cell])
                continue;
            if(cell >= extensives.size() || cell >= cells.size()) {
                valid = false;
                continue;
            }
            double const cell_radiation_extent = extensives[cell].Erad;
            auto const controlled =
                RadiationPositivity::RepairControlledNegativeGroupExtents(
                    extensives[cell].Eg, extensives[cell].Erad,
                    spectral_positivity_repair_tolerance,
                    global_maximum_cell_radiation_extent);
            auto const& repair = controlled.repair;
            bool const globally_negligible_negative_floor =
                controlled.used_global_negative_exception;
            if(!repair.valid) {
                std::string const reason = spectralRepairFailureReason(
                    touched_passive[cell] != 0, controlled,
                    spectral_positivity_repair_tolerance,
                    global_maximum_cell_radiation_extent,
                    cell_radiation_extent, extensives[cell].mass);
                if(touched_passive[cell])
                    setStepFailure(reason, cells[cell].ID);
                else
                    setCellLocalStepFailure(reason, cells[cell].ID);
                valid = false;
                continue;
            }
            if(!repair.repaired &&
               controlled.aggregate_sync_correction == 0)
                continue;
            if(cell < corrected_active.size() &&
               (!touched_passive[cell] ||
                globally_negligible_negative_floor ||
                controlled.aggregate_sync_correction != 0))
                corrected_active[cell] = 1;
            if(!repair.repaired)
                continue;
            ++spectral_repair_event.repaired_cells;
            spectral_repair_event.repaired_groups += repair.repaired_groups;
            spectral_repair_event.injected_energy += repair.injected_extent;
            if(repair.relative_deficit >
               spectral_repair_event.maximum_relative_deficit) {
                spectral_repair_event.maximum_relative_deficit =
                    repair.relative_deficit;
                spectral_repair_event.representative_cell_id = cells[cell].ID;
                spectral_repair_event.representative_group =
                    repair.most_negative_group;
                spectral_repair_event.representative_original_extent =
                    repair.most_negative_extent;
                spectral_repair_event.representative_floor_extent =
                    repair.floor_extent;
                spectral_repair_event.representative_injected_extent =
                    repair.injected_extent;
            }
        }
    }
    if(!collectiveAllTrue(valid))
        return reject();

    for(std::size_t cell = 0; cell < corrected_active.size(); ++cell) {
        if(!corrected_active[cell])
            continue;
        if(!(extensives[cell].mass > 0)) {
            setCellLocalStepFailure(
                "active radiation correction has non-positive cell mass",
                cells[cell].ID);
            valid = false;
            continue;
        }
        cells[cell].Erad = extensives[cell].Erad / extensives[cell].mass;
        if(unknowns_per_cell > 1)
            for(std::size_t group = 0;
                group < extensives[cell].Eg.size(); ++group)
                cells[cell].Eg[group] =
                    extensives[cell].Eg[group] / extensives[cell].mass;
    }

    // Diffusion, passive-face conservation, and the controlled spectral repair
    // are now complete.  Only now may a derived driver apply cell-local
    // operator-split physics such as Compton scattering.
    try {
        valid = applyIndividualPostSolvePhysics(
            tess, cells, extensives, 0,
            global_maximum_cell_radiation_extent);
        if(!valid && getLastStepFailureReason().empty())
            setStepFailure("individual post-solve radiation physics failed");
    }
    catch(std::exception const& error) {
        setStepFailure(error.what());
        valid = false;
    }
    catch(...) {
        setStepFailure("unknown exception in individual post-solve radiation physics");
        valid = false;
    }
    if(!collectiveAllTrue(valid))
        return reject();

    for(std::size_t cell : context.active_indices) {
        if(cell >= cells.size() || cell >= extensives.size()) {
            valid = false;
            continue;
        }
        if(!std::isfinite(extensives[cell].internal_energy) ||
           extensives[cell].internal_energy <= 0) {
            std::ostringstream reason;
            reason << std::setprecision(17)
                   << "active internal energy is invalid at final radiation "
                      "validation"
                   << " internal_energy="
                   << extensives[cell].internal_energy
                   << " Erad=" << extensives[cell].Erad
                   << " mass=" << extensives[cell].mass
                   << " origin_rank=" << rank;
            auto const correction =
                active_passive_correction_ledger.find(cell);
            if(correction != active_passive_correction_ledger.end()) {
                reason << " Erad_before_passive_correction="
                       << correction->second.radiation_before
                       << " passive_correction_sum="
                       << correction->second.correction_sum
                       << " passive_correction_terms="
                       << correction->second.terms
                       << " most_negative_passive_correction="
                       << correction->second.most_negative_term;
                if(correction->second.most_negative_group !=
                   std::numeric_limits<std::size_t>::max())
                    reason << " most_negative_correction_group="
                           << correction->second.most_negative_group;
            }
            setCellLocalStepFailure(reason.str(), cells[cell].ID);
            valid = false;
            continue;
        }
        if(!std::isfinite(extensives[cell].Erad) ||
           extensives[cell].Erad < 0) {
            std::ostringstream reason;
            reason << std::setprecision(17)
                   << "active radiation extent is invalid at final radiation "
                      "validation"
                   << " Erad=" << extensives[cell].Erad
                   << " pre_postsolve_global_maximum_cell_Erad="
                   << global_maximum_cell_radiation_extent
                   << " internal_energy="
                   << extensives[cell].internal_energy
                   << " mass=" << extensives[cell].mass
                   << " origin_rank=" << rank;
            if(std::isfinite(extensives[cell].Erad) &&
               global_maximum_cell_radiation_extent > 0)
                reason << " abs_Erad_over_pre_postsolve_global_maximum="
                       << std::abs(extensives[cell].Erad) /
                              global_maximum_cell_radiation_extent;
            auto const correction =
                active_passive_correction_ledger.find(cell);
            if(correction != active_passive_correction_ledger.end()) {
                reason << " Erad_before_passive_correction="
                       << correction->second.radiation_before
                       << " passive_correction_sum="
                       << correction->second.correction_sum
                       << " passive_correction_terms="
                       << correction->second.terms
                       << " most_negative_passive_correction="
                       << correction->second.most_negative_term;
                if(correction->second.most_negative_group !=
                   std::numeric_limits<std::size_t>::max())
                    reason << " most_negative_correction_group="
                           << correction->second.most_negative_group;
            }
            setCellLocalStepFailure(reason.str(), cells[cell].ID);
            valid = false;
            continue;
        }
        if(unknowns_per_cell > 1)
            for(std::size_t group = 0;
                group < extensives[cell].Eg.size(); ++group) {
                double const after = extensives[cell].Eg[group];
                if(!std::isfinite(after) || after < 0) {
                    double const before =
                        transaction_start_extensives[cell].Eg[group];
                    std::ostringstream reason;
                    reason << std::setprecision(17)
                           << "active multigroup radiation extent became "
                              "non-finite or negative: group=" << group
                           << " before=" << before
                           << " candidate_delta=" << (after - before)
                           << " after=" << after
                           << " total_radiation_extent="
                           << extensives[cell].Erad
                           << " mass=" << extensives[cell].mass;
                    setCellLocalStepFailure(
                        reason.str(), cells[cell].ID);
                    valid = false;
                    break;
                }
            }
    }
    for(std::size_t cell = 0; cell < touched_passive.size(); ++cell) {
        if(!touched_passive[cell])
            continue;
        bool cell_valid = std::isfinite(extensives[cell].Erad) &&
                          extensives[cell].Erad >= 0;
        std::size_t failed_group = unknowns_per_cell;
        if(unknowns_per_cell > 1)
            for(std::size_t group = 0;
                group < extensives[cell].Eg.size(); ++group)
                if(!std::isfinite(extensives[cell].Eg[group]) ||
                   extensives[cell].Eg[group] < 0) {
                    cell_valid = false;
                    failed_group = group;
                    break;
                }
        if(!cell_valid) {
            std::ostringstream reason;
            reason << std::setprecision(17);
            if(failed_group < extensives[cell].Eg.size()) {
                double const before =
                    transaction_start_extensives[cell].Eg[failed_group];
                double const after = extensives[cell].Eg[failed_group];
                reason << "passive multigroup radiation extent became "
                       << "non-finite or negative: group=" << failed_group
                       << " before=" << before
                       << " candidate_delta=" << (after - before)
                       << " after=" << after;
            }
            else {
                double const before =
                    transaction_start_extensives[cell].Erad;
                double const after = extensives[cell].Erad;
                reason << "passive total radiation extent became non-finite "
                       << "or negative: before=" << before
                       << " candidate_delta=" << (after - before)
                       << " after=" << after;
            }
            setStepFailure(reason.str(), cells[cell].ID);
            valid = false;
        }
    }
    if(!collectiveAllTrue(valid))
        return reject();

    unsigned long long global_passive_roundoff_values =
        local_passive_roundoff_values;
    double global_passive_roundoff_max_relative =
        local_passive_roundoff_max_relative;
    int passive_representative_rank = rank;
#ifdef RICH_MPI
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, &global_passive_roundoff_values, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
        "MPI_Allreduce(passive roundoff values)");
    struct {
        double value;
        int rank;
    } passive_local_pick{local_passive_roundoff_max_relative, rank},
      passive_global_pick{0, 0};
    requireDistributedMpiSuccess(
        MPI_Allreduce(&passive_local_pick, &passive_global_pick, 1,
                      MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD),
        "MPI_Allreduce(passive roundoff representative)");
    global_passive_roundoff_max_relative = passive_global_pick.value;
    passive_representative_rank = passive_global_pick.rank;
    unsigned long long passive_identity[2] = {
        static_cast<unsigned long long>(local_passive_roundoff_cell),
        static_cast<unsigned long long>(local_passive_roundoff_group)};
    double passive_details[2] = {local_passive_roundoff_extent,
                                 local_passive_roundoff_scale};
    requireDistributedMpiSuccess(
        MPI_Bcast(passive_identity, 2, MPI_UNSIGNED_LONG_LONG,
                  passive_representative_rank, MPI_COMM_WORLD),
        "MPI_Bcast(passive roundoff identity)");
    requireDistributedMpiSuccess(
        MPI_Bcast(passive_details, 2, MPI_DOUBLE,
                  passive_representative_rank, MPI_COMM_WORLD),
        "MPI_Bcast(passive roundoff details)");
    local_passive_roundoff_cell =
        static_cast<std::size_t>(passive_identity[0]);
    local_passive_roundoff_group =
        static_cast<std::size_t>(passive_identity[1]);
    local_passive_roundoff_extent = passive_details[0];
    local_passive_roundoff_scale = passive_details[1];
#endif
    if(global_passive_roundoff_values > 0 && rank == 0)
        std::clog << std::setprecision(17)
                  << "MG_PASSIVE_ROUNDOFF_REPAIR"
                  << " scope="
#ifdef RICH_MPI
                  << "distributed_active"
#else
                  << "serial_active"
#endif
                  << " values=" << global_passive_roundoff_values
                  << " max_relative_to_face_scale="
                  << global_passive_roundoff_max_relative
                  << " representative_rank=" << passive_representative_rank
                  << " representative_cell_id="
                  << local_passive_roundoff_cell
                  << " representative_group="
                  << local_passive_roundoff_group
                  << " representative_extent="
                  << local_passive_roundoff_extent
                  << " representative_face_scale="
                  << local_passive_roundoff_scale << std::endl;
    if(canonical_extensives != nullptr)
        for(auto const& mapping : canonical_work_indices)
            (*canonical_extensives)[mapping.second] =
                extensives[mapping.first];
    if(canonical_extensives != nullptr) {
        for(Conserved3D const& extensive : *canonical_extensives)
            spectral_repair_event.owned_radiation_energy += extensive.Erad;
    }
    else {
        std::size_t const owned_count =
            std::min(tess.GetPointNo(), extensives.size());
        for(std::size_t cell = 0; cell < owned_count; ++cell)
            spectral_repair_event.owned_radiation_energy +=
                extensives[cell].Erad;
    }
    appendPendingSpectralRepairEvent(spectral_repair_event);
    commitResidualCorrectionAccounting(correction_diagnostics);
    commitSpectralRepairAccounting(
        spectral_repair_event, "distributed_active");
    if(pending_dirichlet_defect)
        commitIndividualRadiationDefect(pending_defect_event);
    for(auto const& wake : pending_dirichlet_wakes) {
        if(wake.first >= cells.size())
            continue;
        auto const canonical =
            canonical_index_by_id.find(cells[wake.first].ID);
        if(canonical == canonical_index_by_id.end() ||
           canonical->second >=
               individual_passive_reference_time_steps_.size())
            continue;
        double& reference =
            individual_passive_reference_time_steps_[canonical->second];
        reference = std::min(reference, wake.second);
    }
    for(PendingPassiveTransfer const& transfer : passive_transfers) {
        if(transfer.applied_gain == 0 || transfer.passive >= cells.size())
            continue;
        auto const canonical =
            canonical_index_by_id.find(cells[transfer.passive].ID);
        if(canonical == canonical_index_by_id.end() ||
           canonical->second >=
               individual_passive_reference_time_steps_.size())
            continue;
        double& reference =
            individual_passive_reference_time_steps_[canonical->second];
        reference = std::min(reference, transfer.reference_time_step);
    }
    if(runtime_options.profile) {
        double const postprocess_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - phase_start).count();
        double const accounted_seconds =
            snapshot_seconds + candidate_seconds + matrix_build_seconds +
            active_mapping_seconds + csr_extract_seconds +
            exchange_setup_seconds + solver_seconds + postprocess_seconds;
        double const wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - phase_total_start).count();
        reportDistributedActivePhaseTiming({{
            snapshot_seconds, candidate_seconds, matrix_build_seconds,
            active_mapping_seconds, csr_extract_seconds,
            exchange_setup_seconds, solver_seconds, postprocess_seconds,
            profile_overhead_seconds, accounted_seconds, wall_seconds}});
        reportDistributedMemoryPhase("output_committed");
    }
    cells.resize(saved_cells.size());
    extensives.resize(saved_extensives_size);
    return true;
    }
    catch(std::exception const& error) {
        abortDistributedActiveFailure(
            "unhandled distributed active exception", MPI_ERR_OTHER,
            error.what());
    }
    catch(...) {
        abortDistributedActiveFailure(
            "unhandled distributed active exception", MPI_ERR_OTHER,
            "unknown exception");
    }
#else
    clearStepFailure();
    std::vector<ComputationalCell3D> saved_cells = cells;
    std::vector<Conserved3D> saved_extensives = extensives;
    SpectralRepairEvent spectral_repair_event;
    IndividualRadiationDefectEvent pending_defect_event;
    bool pending_dirichlet_defect = false;
    std::map<std::size_t, double> pending_dirichlet_wakes;
    CG::HistoricalMGResidualCorrectionDiagnostics correction_diagnostics;
    IndividualContextGuard const guard(individual_context_,
                                       individual_interval_fraction_,
                                       context,
                                       interval_fraction);
    IndividualPassiveRadiationRuntimeOption const& passive_option =
        individualPassiveRadiationRuntimeOption();
    if(!passive_option.valid) {
        setStepFailure(
            "individual radiation passive policy is invalid or conflicting");
        cells = saved_cells;
        extensives = saved_extensives;
        return false;
    }
    static bool passive_policy_reported = false;
    if(!passive_policy_reported) {
        std::clog << "MG_INDIVIDUAL_PASSIVE_POLICY policy="
                  << individualPassiveRadiationPolicyLabel(
                         passive_option.policy)
                  << " deprecated_shadow_alias="
                  << (passive_option.deprecated_alias_present ? 1 : 0)
                  << " library_default="
                  << individualPassiveRadiationPolicyLabel(
                         individual_passive_radiation_default)
                  << " serial_shadow_uses_legacy_commit=1" << std::endl;
        passive_policy_reported = true;
    }

    struct SerialPassiveTransfer
    {
        std::size_t active = 0;
        std::size_t passive = 0;
        std::size_t group = 0;
        double proposed_gain = 0;
        double applied_gain = 0;
        double reference_time_step = 0;
    };
    std::vector<SerialPassiveTransfer> passive_transfers;
    try {
        prepareIndividualCandidate(tess, cells);
        CG::mat full_matrix;
        CG::size_t_mat full_columns;
        std::vector<double> full_rhs;
        std::vector<double> full_initial;
        individual_face_coefficients_.clear();
        BuildMatrix(tess, full_matrix, full_columns, cells, 0,
                    full_rhs, full_initial, time);
        if(!validateIndividualCoefficients(context, cells)) {
            cells = saved_cells;
            extensives = saved_extensives;
            return false;
        }

        std::size_t const unknowns_per_cell = individualUnknownsPerCell();
        std::size_t const row_count = tess.GetPointNo() * unknowns_per_cell;
        if(full_matrix.size() != row_count || full_rhs.size() != row_count ||
           full_initial.size() < row_count)
            throw std::runtime_error("radiation matrix size does not match cell/group layout");

        // BuildMatrix may append fixed passive-halo values to full_initial.
        // They are legal matrix columns but never become reduced unknowns.
        std::vector<std::size_t> global_to_local(full_initial.size(), CG::max_size_t);
        std::vector<std::size_t> local_to_global;
        local_to_global.reserve(context.active_indices.size() * unknowns_per_cell);
        for(std::size_t cell : context.active_indices)
            for(std::size_t group = 0; group < unknowns_per_cell; ++group) {
                std::size_t const global = cell * unknowns_per_cell + group;
                global_to_local.at(global) = local_to_global.size();
                local_to_global.push_back(global);
            }

        CG::mat matrix(local_to_global.size());
        CG::size_t_mat columns(local_to_global.size());
        std::vector<double> rhs(local_to_global.size(), 0);
        std::vector<double> verification_rhs(local_to_global.size(), 0);
        std::vector<double> verification_scale(local_to_global.size(), 0);
        std::vector<double> base_solution(local_to_global.size(), 0);
        std::vector<double> final_correction_volume(
            local_to_global.size(), 0);
        std::vector<double> solution(local_to_global.size(), 0);
        for(std::size_t local_row = 0; local_row < local_to_global.size(); ++local_row) {
            std::size_t const global_row = local_to_global[local_row];
            AccurateResidualAccumulator correction_row;
            AccurateResidualAccumulator verification_row;
            AccurateResidualAccumulator fixed_scale_row;
            correction_row.Add(full_rhs[global_row]);
            verification_row.Add(full_rhs[global_row]);
            fixed_scale_row.Add(std::abs(full_rhs[global_row]));
            base_solution[local_row] = full_initial[global_row];
            final_correction_volume[local_row] = tess.GetVolume(
                global_row / unknowns_per_cell) *
                pow<3>(GetLengthScale());
            for(std::size_t entry = 0; entry < full_matrix[global_row].size(); ++entry) {
                std::size_t const global_column = full_columns[global_row][entry];
                if(global_column == CG::max_size_t)
                    continue;
                double const value = full_matrix[global_row][entry];
                if(global_column >= full_initial.size())
                    throw std::runtime_error("serial individual radiation matrix contains a remote column");
                correction_row.AddProduct(
                    -value, full_initial[global_column]);
                std::size_t const local_column = global_to_local[global_column];
                if(local_column == CG::max_size_t)
                {
                    verification_row.AddProduct(
                        -value, full_initial[global_column]);
                    fixed_scale_row.AddProduct(
                        std::abs(value),
                        std::abs(full_initial[global_column]));
                }
                else {
                    matrix[local_row].push_back(value);
                    columns[local_row].push_back(local_column);
                }
            }
            rhs[local_row] = correction_row.Value();
            verification_rhs[local_row] = verification_row.Value();
            verification_scale[local_row] = fixed_scale_row.Value();
        }

        if(unknowns_per_cell == 0 ||
           local_to_global.size() % unknowns_per_cell != 0) {
            setStepFailure(
                "active radiation rows are not complete cell blocks");
            cells = saved_cells;
            extensives = saved_extensives;
            return false;
        }
        std::vector<std::size_t> block_cell_ids;
        block_cell_ids.reserve(local_to_global.size() / unknowns_per_cell);
        for(std::size_t row = 0; row < local_to_global.size();
            row += unknowns_per_cell) {
            std::size_t const cell =
                local_to_global[row] / unknowns_per_cell;
            if(cell >= cells.size()) {
                setStepFailure(
                    "active radiation block references a missing cell");
                cells = saved_cells;
                extensives = saved_extensives;
                return false;
            }
            block_cell_ids.push_back(cells[cell].ID);
        }
        FixedPositiveRadiationScale const fixed_positive_scale =
            FixedCellPositiveRadiationScale(
                tess, full_initial, local_to_global, unknowns_per_cell,
                GetLengthScale());
        double const fixed_cell_maximum_absolute_Eg =
            FixedCellRadiationMaximumAbsoluteGroup(
                tess, full_initial, local_to_global, unknowns_per_cell);
        if(!fixed_positive_scale.finite ||
           !std::isfinite(fixed_cell_maximum_absolute_Eg)) {
            setStepFailure(
                "fixed-cell radiation maximum is non-finite");
            cells = saved_cells;
            extensives = saved_extensives;
            return false;
        }
        bool const solver_converged = solveLocalBiCGSTAB(
            tolerance, total_iters, matrix, columns, rhs, verification_rhs,
            verification_scale, base_solution, final_correction_volume,
            fixed_cell_maximum_absolute_Eg,
            fixed_positive_scale.maximum_cell_energy,
            fixed_positive_scale.total_energy,
            solution, unknowns_per_cell,
            GetPreconditionerKind(), block_cell_ids,
            *this, correction_diagnostics);
        if(!solver_converged) {
            std::vector<double> failed_physical_solution = full_initial;
            for(std::size_t local = 0; local < local_to_global.size(); ++local)
                failed_physical_solution[local_to_global[local]] =
                    base_solution[local] + solution[local];
            std::vector<double> failed_pre_correction_solution =
                failed_physical_solution;
            if(correction_diagnostics.available &&
               correction_diagnostics.pre_correction_solution.size() ==
                   local_to_global.size())
                for(std::size_t local = 0; local < local_to_global.size();
                    ++local)
                    failed_pre_correction_solution[local_to_global[local]] =
                        correction_diagnostics.pre_correction_solution[local];
            if(requestIndividualSolutionRetry(
                   failed_pre_correction_solution, failed_physical_solution,
                   local_to_global, cells,
                   "serial_active_failed_iterate")) {
                int const completed_iterations = total_iters;
                int retry_iterations = 0;
                CG::mat().swap(matrix);
                CG::size_t_mat().swap(columns);
                CG::mat().swap(full_matrix);
                CG::size_t_mat().swap(full_columns);
                std::vector<double>().swap(full_rhs);
                std::vector<double>().swap(full_initial);
                std::vector<double>().swap(rhs);
                std::vector<double>().swap(verification_rhs);
                std::vector<double>().swap(verification_scale);
                std::vector<double>().swap(base_solution);
                std::vector<double>().swap(final_correction_volume);
                std::vector<double>().swap(solution);
                std::vector<std::size_t>().swap(global_to_local);
                std::vector<std::size_t>().swap(local_to_global);
                std::vector<std::size_t>().swap(block_cell_ids);
                correction_diagnostics =
                    CG::HistoricalMGResidualCorrectionDiagnostics();
                std::vector<double>().swap(failed_pre_correction_solution);
                std::vector<double>().swap(failed_physical_solution);
                rich_trim_after_rare_spike();
                cells = std::move(saved_cells);
                extensives = std::move(saved_extensives);
                bool const accepted = RadiationDriver::stepIndividual(
                    tolerance, retry_iterations, tess, cells, extensives,
                    context, interval_fraction, time, canonical_cells,
                    canonical_extensives, owned_to_canonical);
                total_iters = completed_iterations + retry_iterations;
                return accepted;
            }
            if(!correction_diagnostics.failure_reason.empty()) {
                std::ostringstream reason;
                CG::AppendHistoricalMGResidualCorrectionFailureDiagnostics(
                    reason, correction_diagnostics,
                    static_cast<std::size_t>(std::max(total_iters, 0)));
                setCellLocalStepFailure(
                    reason.str(), correction_diagnostics.failure_cell_id);
            }
            else
                setStepFailure(
                    "active-only BiCGSTAB failed or found a non-positive diagonal");
            cells = saved_cells;
            extensives = saved_extensives;
            return false;
        }

        std::vector<double> full_solution = full_initial;
        for(std::size_t local = 0; local < local_to_global.size(); ++local) {
            std::size_t const global_unknown = local_to_global[local];
            solution[local] = base_solution[local] + solution[local];
            if(!std::isfinite(solution[local])) {
                std::size_t const cell = global_unknown / unknowns_per_cell;
                std::ostringstream reason;
                reason << "active radiation solution is non-finite"
                       << " (value=" << solution[local]
                       << ", initial=" << full_initial[global_unknown]
                       << ", rhs=" << rhs[local] << ')';
                setCellLocalStepFailure(reason.str(), cells[cell].ID);
                cells = saved_cells;
                extensives = saved_extensives;
                return false;
            }
            full_solution[local_to_global[local]] = solution[local];
        }

        std::vector<double> pre_correction_full_solution = full_solution;
        if(correction_diagnostics.available &&
           correction_diagnostics.pre_correction_solution.size() ==
               local_to_global.size())
            for(std::size_t local = 0; local < local_to_global.size(); ++local)
                pre_correction_full_solution[local_to_global[local]] =
                    correction_diagnostics.pre_correction_solution[local];
        if(requestIndividualSolutionRetry(
               pre_correction_full_solution, full_solution, local_to_global,
               cells, "serial_active")) {
            int const completed_iterations = total_iters;
            int retry_iterations = 0;
            CG::mat().swap(matrix);
            CG::size_t_mat().swap(columns);
            CG::mat().swap(full_matrix);
            CG::size_t_mat().swap(full_columns);
            std::vector<double>().swap(full_rhs);
            std::vector<double>().swap(full_initial);
            std::vector<double>().swap(rhs);
            std::vector<double>().swap(verification_rhs);
            std::vector<double>().swap(verification_scale);
            std::vector<double>().swap(base_solution);
            std::vector<double>().swap(final_correction_volume);
            std::vector<double>().swap(solution);
            std::vector<std::size_t>().swap(global_to_local);
            std::vector<std::size_t>().swap(local_to_global);
            std::vector<std::size_t>().swap(block_cell_ids);
            correction_diagnostics =
                CG::HistoricalMGResidualCorrectionDiagnostics();
            std::vector<double>().swap(pre_correction_full_solution);
            std::vector<double>().swap(full_solution);
            rich_trim_after_rare_spike();
            cells = std::move(saved_cells);
            extensives = std::move(saved_extensives);
            bool const accepted = RadiationDriver::stepIndividual(
                tolerance, retry_iterations, tess, cells, extensives, context,
                interval_fraction, time, canonical_cells,
                canonical_extensives, owned_to_canonical);
            total_iters = completed_iterations + retry_iterations;
            return accepted;
        }

        PostCG(tess, extensives, 0, cells, full_solution, full_solution);
        double const extensive_conversion =
            time_scale_ * time_scale_ /
            (length_scale_ * length_scale_ * mass_scale_);
        std::vector<unsigned char> touched_passive(cells.size(), 0);
        for(IndividualFaceCoefficient const& face : individual_face_coefficients_) {
            bool const left_active = context.isActive(face.left);
            bool const right_active = context.isActive(face.right);
            if(left_active == right_active)
                continue;
            std::size_t const active = left_active ? face.left : face.right;
            std::size_t const passive = left_active ? face.right : face.left;
            std::size_t const active_unknown = active * unknowns_per_cell + face.group;
            std::size_t const passive_unknown = passive * unknowns_per_cell + face.group;
            double const passive_gain = face.coefficient *
                (full_solution[active_unknown] - full_solution[passive_unknown]) *
                extensive_conversion;
            SerialPassiveTransfer transfer;
            transfer.active = active;
            transfer.passive = passive;
            transfer.group = face.group;
            transfer.proposed_gain = passive_gain;
            transfer.applied_gain = passive_gain;
            transfer.reference_time_step = face.time_step;
            passive_transfers.push_back(transfer);
        }

        if(passive_option.policy == IndividualPassiveRadiationPolicy::
               FrozenDirichletMeasuredDefect) {
            AccurateResidualAccumulator positive_extent_accumulator;
            std::size_t const owned_count =
                std::min(tess.GetPointNo(), saved_extensives.size());
            for(std::size_t cell = 0; cell < owned_count; ++cell) {
                if(unknowns_per_cell == 1) {
                    double const extent = saved_extensives[cell].Erad;
                    if(!std::isfinite(extent))
                        pending_defect_event.valid = false;
                    else if(extent > 0)
                        positive_extent_accumulator.Add(extent);
                }
                else {
                    for(double const extent : saved_extensives[cell].Eg) {
                        if(!std::isfinite(extent))
                            pending_defect_event.valid = false;
                        else if(extent > 0)
                            positive_extent_accumulator.Add(extent);
                    }
                }
            }
            AccurateResidualAccumulator rhs_accumulator;
            for(double const value : full_rhs) {
                if(!std::isfinite(value)) {
                    pending_defect_event.valid = false;
                    continue;
                }
                rhs_accumulator.Add(std::abs(value));
            }
            double const local_rhs_floor =
                1024 * std::numeric_limits<double>::epsilon() *
                extensive_conversion * std::abs(rhs_accumulator.Value());
            pending_defect_event.normalization_scale =
                collectiveIndividualRadiationDefectScale(
                    positive_extent_accumulator.Value(), local_rhs_floor,
                    pending_defect_event.
                        candidate_start_positive_global_extent,
                    pending_defect_event.rhs_derived_global_floor);
            pending_defect_event.valid = pending_defect_event.valid &&
                std::isfinite(extensive_conversion) &&
                extensive_conversion > 0 &&
                std::isfinite(pending_defect_event.normalization_scale) &&
                pending_defect_event.normalization_scale > 0;

            struct PassiveGroupDefect
            {
                AccurateResidualAccumulator withdrawal;
                AccurateResidualAccumulator deposit;
                std::uint64_t face_group_terms = 0;
                double representative_withdrawal_term = 0;
                std::uint64_t representative_active_id =
                    std::numeric_limits<std::uint64_t>::max();
            };
            std::map<std::pair<std::size_t, std::size_t>,
                     PassiveGroupDefect> grouped_defects;
            AccurateResidualAccumulator signed_defect;
            AccurateResidualAccumulator absolute_defect;
            AccurateResidualAccumulator total_withdrawal;
            AccurateResidualAccumulator total_deposit;
            std::set<std::tuple<std::uint64_t, std::uint64_t,
                                std::uint64_t> > stable_face_group_keys;
            bool have_duplicate_face_group = false;
            std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                       std::uint64_t, std::uint64_t> duplicate_face_group;
            for(SerialPassiveTransfer const& transfer : passive_transfers) {
                double const defect = -transfer.proposed_gain;
                if(!std::isfinite(defect) ||
                   transfer.active >= cells.size() ||
                   transfer.passive >= cells.size() ||
                   transfer.passive >= saved_extensives.size()) {
                    pending_defect_event.valid = false;
                    continue;
                }
                std::uint64_t const active_id =
                    static_cast<std::uint64_t>(cells[transfer.active].ID);
                std::uint64_t const passive_id =
                    static_cast<std::uint64_t>(cells[transfer.passive].ID);
                std::tuple<std::uint64_t, std::uint64_t,
                           std::uint64_t> const stable_key(
                    std::min(active_id, passive_id),
                    std::max(active_id, passive_id),
                    static_cast<std::uint64_t>(transfer.group));
                if(!stable_face_group_keys.insert(stable_key).second) {
                    ++pending_defect_event.duplicate_face_group_terms;
                    std::tuple<std::uint64_t, std::uint64_t,
                               std::uint64_t, std::uint64_t,
                               std::uint64_t> const offender(
                        std::get<0>(stable_key), std::get<1>(stable_key),
                        std::get<2>(stable_key), active_id, passive_id);
                    if(!have_duplicate_face_group ||
                       offender < duplicate_face_group) {
                        have_duplicate_face_group = true;
                        duplicate_face_group = offender;
                    }
                }
                signed_defect.Add(defect);
                absolute_defect.Add(std::abs(defect));
                PassiveGroupDefect& group = grouped_defects[
                    std::make_pair(transfer.passive, transfer.group)];
                if(group.face_group_terms ==
                   std::numeric_limits<std::uint64_t>::max())
                    pending_defect_event.valid = false;
                else
                    ++group.face_group_terms;
                if(defect >= 0) {
                    group.withdrawal.Add(defect);
                    total_withdrawal.Add(defect);
                    std::uint64_t const active_id =
                        static_cast<std::uint64_t>(
                            cells[transfer.active].ID);
                    std::tuple<double, std::uint64_t> const candidate(
                        -defect, active_id);
                    std::tuple<double, std::uint64_t> const current(
                        -group.representative_withdrawal_term,
                        group.representative_active_id);
                    if(candidate < current) {
                        group.representative_withdrawal_term = defect;
                        group.representative_active_id = active_id;
                    }
                }
                else {
                    group.deposit.Add(-defect);
                    total_deposit.Add(-defect);
                }
                if(pending_defect_event.face_group_terms ==
                   std::numeric_limits<std::uint64_t>::max())
                    pending_defect_event.valid = false;
                else
                    ++pending_defect_event.face_group_terms;
            }
            pending_defect_event.signed_extent = signed_defect.Value();
            pending_defect_event.absolute_extent = absolute_defect.Value();
            pending_defect_event.passive_withdrawal_extent =
                total_withdrawal.Value();
            pending_defect_event.passive_deposit_extent =
                total_deposit.Value();
            double const local_roundoff_floor =
                1024 * std::numeric_limits<double>::epsilon() *
                pending_defect_event.normalization_scale;
            bool have_representative = false;
            for(auto const& entry : grouped_defects) {
                std::size_t const cell = entry.first.first;
                std::size_t const group = entry.first.second;
                if(cell >= cells.size() || cell >= saved_extensives.size() ||
                   (unknowns_per_cell > 1 &&
                    group >= saved_extensives[cell].Eg.size())) {
                    pending_defect_event.valid = false;
                    continue;
                }
                double const passive_extent = unknowns_per_cell == 1 ?
                    saved_extensives[cell].Erad :
                    saved_extensives[cell].Eg[group];
                double const withdrawal = entry.second.withdrawal.Value();
                if(!std::isfinite(passive_extent) || passive_extent < 0 ||
                   !std::isfinite(withdrawal) || withdrawal < 0 ||
                   !std::isfinite(local_roundoff_floor) ||
                   local_roundoff_floor <= 0) {
                    pending_defect_event.valid = false;
                    continue;
                }
                IndividualRadiationLocalDefectMeasure const local_measure =
                    measureIndividualRadiationLocalDefect(
                        withdrawal, passive_extent, local_roundoff_floor,
                        pending_defect_event.normalization_scale);
                std::uint64_t const active_id =
                    entry.second.representative_active_id;
                std::uint64_t const passive_id =
                    static_cast<std::uint64_t>(cells[cell].ID);
                std::tuple<double, std::uint64_t, std::uint64_t,
                           std::uint64_t> const candidate(
                    -local_measure.tolerance_ratio, active_id, passive_id,
                    static_cast<std::uint64_t>(group));
                std::tuple<double, std::uint64_t, std::uint64_t,
                           std::uint64_t> const current(
                    -pending_defect_event.maximum_local_tolerance_ratio,
                    pending_defect_event.representative_active_id,
                    pending_defect_event.representative_passive_id,
                    pending_defect_event.representative_group);
                if(!std::isfinite(local_measure.relative_fraction) ||
                   !std::isfinite(local_measure.allowed_withdrawal) ||
                   local_measure.allowed_withdrawal <= 0 ||
                   !std::isfinite(local_measure.tolerance_ratio) ||
                   local_measure.tolerance_ratio < 0)
                    pending_defect_event.valid = false;
                else if(!have_representative || candidate < current) {
                    have_representative = true;
                    pending_defect_event.maximum_local_tolerance_ratio =
                        local_measure.tolerance_ratio;
                    pending_defect_event.representative_active_id = active_id;
                    pending_defect_event.representative_passive_id = passive_id;
                    pending_defect_event.representative_group = group;
                    pending_defect_event.representative_active_rank = 0;
                }
                if(std::isfinite(local_measure.relative_fraction))
                    pending_defect_event.maximum_local_fraction = std::max(
                        pending_defect_event.maximum_local_fraction,
                        local_measure.relative_fraction);
            }
            if(have_duplicate_face_group) {
                pending_defect_event.valid = false;
                pending_defect_event.representative_active_id =
                    std::get<3>(duplicate_face_group);
                pending_defect_event.representative_passive_id =
                    std::get<4>(duplicate_face_group);
                pending_defect_event.representative_group =
                    std::get<2>(duplicate_face_group);
            }
            if(!validateIndividualRadiationDefect(pending_defect_event)) {
                cells = saved_cells;
                extensives = saved_extensives;
                return false;
            }
            IndividualRadiationDefectConfiguration const&
                defect_configuration =
                    individualRadiationDefectConfiguration();
            bool const synchronize_passive_neighbors =
                pending_defect_event.maximum_local_tolerance_ratio > 1 ||
                pending_defect_event.event_absolute_fraction >
                    defect_configuration.event_absolute_target;
            if(synchronize_passive_neighbors)
                for(SerialPassiveTransfer const& transfer :
                    passive_transfers) {
                    if(transfer.proposed_gain == 0 ||
                       transfer.passive >= cells.size())
                        continue;
                    double const reference_time_step =
                        std::isfinite(transfer.reference_time_step) &&
                        transfer.reference_time_step > 0 ?
                        transfer.reference_time_step : context.time_quantum;
                    auto const inserted = pending_dirichlet_wakes.emplace(
                        transfer.passive, reference_time_step);
                    if(!inserted.second)
                        inserted.first->second = std::min(
                            inserted.first->second, reference_time_step);
                }
            pending_dirichlet_defect = true;
            for(std::size_t cell = 0;
                cell < cells.size() && cell < saved_extensives.size(); ++cell) {
                bool const active = cell < context.active_mask.size() &&
                    context.isActive(cell);
                if(active)
                    continue;
                extensives[cell] = saved_extensives[cell];
                if(cell < saved_cells.size())
                    cells[cell] = saved_cells[cell];
            }
            passive_transfers.clear();
        }

        if(unknowns_per_cell == 0 ||
           cells.size() > std::numeric_limits<std::size_t>::max() /
               unknowns_per_cell) {
            setStepFailure("individual radiation passive limiter size overflow");
            cells = saved_cells;
            extensives = saved_extensives;
            return false;
        }
        std::size_t const passive_value_count =
            cells.size() * unknowns_per_cell;
        std::vector<double> positive_gain(passive_value_count, 0);
        std::vector<double> negative_loss(passive_value_count, 0);
        std::vector<double> negative_scale(passive_value_count, 1);
        std::vector<unsigned char> touched_value(passive_value_count, 0);
        std::vector<std::size_t> negative_transfer_index(
            passive_value_count, std::numeric_limits<std::size_t>::max());
        for(std::size_t transfer_index = 0;
            transfer_index < passive_transfers.size(); ++transfer_index) {
            SerialPassiveTransfer const& transfer =
                passive_transfers[transfer_index];
            std::size_t const key =
                transfer.passive * unknowns_per_cell + transfer.group;
            touched_value[key] = 1;
            if(transfer.proposed_gain >= 0)
                positive_gain[key] += transfer.proposed_gain;
            else {
                negative_loss[key] -= transfer.proposed_gain;
                if(negative_transfer_index[key] ==
                   std::numeric_limits<std::size_t>::max())
                    negative_transfer_index[key] = transfer_index;
            }
        }
        for(std::size_t cell = 0; cell < cells.size(); ++cell)
            for(std::size_t group = 0; group < unknowns_per_cell; ++group) {
                std::size_t const key = cell * unknowns_per_cell + group;
                if(!touched_value[key])
                    continue;
                double const available = unknowns_per_cell == 1
                    ? saved_extensives[cell].Erad
                    : saved_extensives[cell].Eg[group];
                if(!std::isfinite(available) || available < 0 ||
                   !std::isfinite(positive_gain[key]) ||
                   !std::isfinite(negative_loss[key])) {
                    std::ostringstream reason;
                    reason << std::setprecision(17)
                           << "passive radiation extent was invalid before face limiting"
                           << ": group=" << group
                           << " extent=" << available
                           << " positive_gain=" << positive_gain[key]
                           << " negative_loss=" << negative_loss[key]
                           << " mass=" << saved_extensives[cell].mass;
                    setStepFailure(reason.str(), cells[cell].ID);
                    cells = saved_cells;
                    extensives = saved_extensives;
                    return false;
                }
                double const capacity = available + positive_gain[key];
                if(negative_loss[key] > capacity) {
                    double const ratio = capacity > 0
                        ? capacity / negative_loss[key] : 0;
                    negative_scale[key] = ratio > 0
                        ? std::nextafter(std::min(1.0, ratio), 0.0) : 0;
                }
            }

        std::vector<unsigned char> corrected_active(cells.size(), 0);
        std::map<std::size_t, ActivePassiveCorrectionLedger>
            active_passive_correction_ledger;
        auto apply_active_passive_correction =
            [&](std::size_t const cell, std::size_t const group,
                double const correction)
            {
                auto const inserted = active_passive_correction_ledger.emplace(
                    cell, ActivePassiveCorrectionLedger());
                ActivePassiveCorrectionLedger& ledger = inserted.first->second;
                if(inserted.second)
                    ledger.radiation_before = extensives.at(cell).Erad;
                ledger.correction_sum += correction;
                ++ledger.terms;
                if(correction < ledger.most_negative_term) {
                    ledger.most_negative_term = correction;
                    ledger.most_negative_group = group;
                }
                corrected_active.at(cell) = 1;
                extensives.at(cell).Erad += correction;
                if(unknowns_per_cell > 1)
                    extensives.at(cell).Eg.at(group) += correction;
            };
        for(SerialPassiveTransfer& transfer : passive_transfers) {
            std::size_t const key =
                transfer.passive * unknowns_per_cell + transfer.group;
            if(transfer.proposed_gain < 0)
                transfer.applied_gain =
                    transfer.proposed_gain * negative_scale[key];
            touched_passive.at(transfer.passive) = 1;
            extensives.at(transfer.passive).Erad += transfer.applied_gain;
            if(unknowns_per_cell > 1)
                extensives.at(transfer.passive).Eg.at(transfer.group) +=
                    transfer.applied_gain;
            double const correction =
                transfer.proposed_gain - transfer.applied_gain;
            if(correction == 0)
                continue;
            apply_active_passive_correction(
                transfer.active, transfer.group, correction);
        }
        for(std::size_t cell = 0; cell < cells.size(); ++cell)
            for(std::size_t group = 0; group < unknowns_per_cell; ++group) {
                std::size_t const key = cell * unknowns_per_cell + group;
                if(!touched_value[key])
                    continue;
                double& passive_energy = unknowns_per_cell == 1
                    ? extensives[cell].Erad : extensives[cell].Eg[group];
                if(!std::isfinite(passive_energy)) {
                    setStepFailure(
                        "passive radiation extent became non-finite after limiting",
                        cells[cell].ID);
                    cells = saved_cells;
                    extensives = saved_extensives;
                    return false;
                }
                if(passive_energy >= 0)
                    continue;
                std::size_t const transfer_index = negative_transfer_index[key];
                if(transfer_index == std::numeric_limits<std::size_t>::max()) {
                    setStepFailure(
                        "passive radiation limiter has no outgoing transfer",
                        cells[cell].ID);
                    cells = saved_cells;
                    extensives = saved_extensives;
                    return false;
                }
                SerialPassiveTransfer const& transfer =
                    passive_transfers[transfer_index];
                double const residual = -passive_energy;
                passive_energy = 0;
                if(unknowns_per_cell > 1)
                    extensives[cell].Erad += residual;
                apply_active_passive_correction(
                    transfer.active, transfer.group, -residual);
            }
        double global_maximum_cell_radiation_extent = 0;
        if((canonical_cells == nullptr) !=
           (canonical_extensives == nullptr))
            throw std::runtime_error(
                "individual radiation canonical arrays are incomplete");
        if(canonical_cells != nullptr && canonical_extensives != nullptr) {
            if(canonical_cells->size() != canonical_extensives->size())
                throw std::runtime_error(
                    "individual radiation canonical arrays have different sizes");
            std::map<std::size_t, std::size_t> work_index_by_id;
            for(std::size_t cell = 0;
                cell < cells.size() && cell < extensives.size(); ++cell)
                work_index_by_id.emplace(cells[cell].ID, cell);
            for(std::size_t canonical = 0;
                canonical < canonical_extensives->size(); ++canonical) {
                auto const found = work_index_by_id.find(
                    canonical_cells->at(canonical).ID);
                double const extent = found == work_index_by_id.end()
                    ? canonical_extensives->at(canonical).Erad
                    : extensives[found->second].Erad;
                if(std::isfinite(extent))
                    global_maximum_cell_radiation_extent = std::max(
                        global_maximum_cell_radiation_extent, extent);
            }
        }
        else {
            std::size_t const owned_extent_count = std::min<std::size_t>(
                tess.GetPointNo(), extensives.size());
            for(std::size_t cell = 0; cell < owned_extent_count; ++cell)
                if(std::isfinite(extensives[cell].Erad))
                    global_maximum_cell_radiation_extent = std::max(
                        global_maximum_cell_radiation_extent,
                        extensives[cell].Erad);
        }

        if(unknowns_per_cell > 1) {
            std::vector<unsigned char> repair_target = touched_passive;
            for(std::size_t const cell : context.active_indices)
                repair_target.at(cell) = 1;
            for(std::size_t cell = 0; cell < repair_target.size(); ++cell) {
                if(!repair_target[cell])
                    continue;
                double const cell_radiation_extent = extensives[cell].Erad;
                auto const controlled =
                    RadiationPositivity::RepairControlledNegativeGroupExtents(
                        extensives[cell].Eg, extensives[cell].Erad,
                        spectral_positivity_repair_tolerance,
                        global_maximum_cell_radiation_extent);
                auto const& repair = controlled.repair;
                bool const globally_negligible_negative_floor =
                    controlled.used_global_negative_exception;
                if(!repair.valid) {
                    std::string const reason = spectralRepairFailureReason(
                        touched_passive[cell] != 0, controlled,
                        spectral_positivity_repair_tolerance,
                        global_maximum_cell_radiation_extent,
                        cell_radiation_extent, extensives[cell].mass);
                    if(touched_passive[cell])
                        setStepFailure(reason, cells[cell].ID);
                    else
                        setCellLocalStepFailure(reason, cells[cell].ID);
                    cells = saved_cells;
                    extensives = saved_extensives;
                    return false;
                }
                if(!repair.repaired &&
                   controlled.aggregate_sync_correction == 0)
                    continue;
                if(!touched_passive[cell] ||
                   globally_negligible_negative_floor ||
                   controlled.aggregate_sync_correction != 0)
                    corrected_active[cell] = 1;
                if(!repair.repaired)
                    continue;
                ++spectral_repair_event.repaired_cells;
                spectral_repair_event.repaired_groups +=
                    repair.repaired_groups;
                spectral_repair_event.injected_energy +=
                    repair.injected_extent;
                if(repair.relative_deficit >
                   spectral_repair_event.maximum_relative_deficit) {
                    spectral_repair_event.maximum_relative_deficit =
                        repair.relative_deficit;
                    spectral_repair_event.representative_cell_id =
                        cells[cell].ID;
                    spectral_repair_event.representative_group =
                        repair.most_negative_group;
                    spectral_repair_event.representative_original_extent =
                        repair.most_negative_extent;
                    spectral_repair_event.representative_floor_extent =
                        repair.floor_extent;
                    spectral_repair_event.representative_injected_extent =
                        repair.injected_extent;
                }
            }
        }
        for(std::size_t cell = 0; cell < corrected_active.size(); ++cell) {
            if(!corrected_active[cell])
                continue;
            if(!(extensives[cell].mass > 0)) {
                setCellLocalStepFailure(
                    "active radiation correction has non-positive cell mass",
                    cells[cell].ID);
                cells = saved_cells;
                extensives = saved_extensives;
                return false;
            }
            cells[cell].Erad = extensives[cell].Erad / extensives[cell].mass;
            if(unknowns_per_cell > 1)
                for(std::size_t group = 0;
                    group < extensives[cell].Eg.size(); ++group)
                    cells[cell].Eg[group] =
                        extensives[cell].Eg[group] / extensives[cell].mass;
        }
        // Complete diffusion/passive conservation and its spectral repair
        // before any derived cell-local operator (notably Compton) runs.
        if(!applyIndividualPostSolvePhysics(
               tess, cells, extensives, 0,
               global_maximum_cell_radiation_extent)) {
            if(getLastStepFailureReason().empty())
                setStepFailure(
                    "individual post-solve radiation physics failed");
            cells = saved_cells;
            extensives = saved_extensives;
            return false;
        }
        for(std::size_t cell : context.active_indices) {
            if(!std::isfinite(extensives[cell].internal_energy) ||
               extensives[cell].internal_energy <= 0) {
                std::ostringstream reason;
                reason << std::setprecision(17)
                       << "active internal energy is invalid at final "
                          "radiation validation"
                       << " internal_energy="
                       << extensives[cell].internal_energy
                       << " Erad=" << extensives[cell].Erad
                       << " mass=" << extensives[cell].mass;
                auto const correction =
                    active_passive_correction_ledger.find(cell);
                if(correction != active_passive_correction_ledger.end()) {
                    reason << " Erad_before_passive_correction="
                           << correction->second.radiation_before
                           << " passive_correction_sum="
                           << correction->second.correction_sum
                           << " passive_correction_terms="
                           << correction->second.terms
                           << " most_negative_passive_correction="
                           << correction->second.most_negative_term;
                    if(correction->second.most_negative_group !=
                       std::numeric_limits<std::size_t>::max())
                        reason << " most_negative_correction_group="
                               << correction->second.most_negative_group;
                }
                setCellLocalStepFailure(reason.str(), cells[cell].ID);
                cells = saved_cells;
                extensives = saved_extensives;
                return false;
            }
            if(!std::isfinite(extensives[cell].Erad) ||
               extensives[cell].Erad < 0) {
                std::ostringstream reason;
                reason << std::setprecision(17)
                       << "active radiation extent is invalid at final "
                          "radiation validation"
                       << " Erad=" << extensives[cell].Erad
                       << " pre_postsolve_global_maximum_cell_Erad="
                       << global_maximum_cell_radiation_extent
                       << " internal_energy="
                       << extensives[cell].internal_energy
                       << " mass=" << extensives[cell].mass;
                if(std::isfinite(extensives[cell].Erad) &&
                   global_maximum_cell_radiation_extent > 0)
                    reason << " abs_Erad_over_pre_postsolve_global_maximum="
                           << std::abs(extensives[cell].Erad) /
                                  global_maximum_cell_radiation_extent;
                auto const correction =
                    active_passive_correction_ledger.find(cell);
                if(correction != active_passive_correction_ledger.end()) {
                    reason << " Erad_before_passive_correction="
                           << correction->second.radiation_before
                           << " passive_correction_sum="
                           << correction->second.correction_sum
                           << " passive_correction_terms="
                           << correction->second.terms
                           << " most_negative_passive_correction="
                           << correction->second.most_negative_term;
                    if(correction->second.most_negative_group !=
                       std::numeric_limits<std::size_t>::max())
                        reason << " most_negative_correction_group="
                               << correction->second.most_negative_group;
                }
                setCellLocalStepFailure(reason.str(), cells[cell].ID);
                cells = saved_cells;
                extensives = saved_extensives;
                return false;
            }
            if(unknowns_per_cell > 1)
                for(double energy : extensives[cell].Eg)
                    if(!std::isfinite(energy) || energy < 0) {
                        setCellLocalStepFailure(
                            "active group radiation energy is non-finite or negative",
                            cells[cell].ID);
                        cells = saved_cells;
                        extensives = saved_extensives;
                        return false;
                    }
        }
        for(std::size_t cell = 0; cell < touched_passive.size(); ++cell) {
            if(!touched_passive[cell])
                continue;
            bool valid = std::isfinite(extensives[cell].Erad) &&
                         extensives[cell].Erad >= 0;
            if(unknowns_per_cell > 1)
                for(double energy : extensives[cell].Eg)
                    valid = valid && std::isfinite(energy) && energy >= 0;
            if(!valid) {
                setStepFailure("passive radiation extent became non-finite or negative", cells[cell].ID);
                cells = saved_cells;
                extensives = saved_extensives;
                return false;
            }
        }
    }
    catch(std::exception const& error) {
        setStepFailure(error.what());
        cells = saved_cells;
        extensives = saved_extensives;
        return false;
    }
    catch(...) {
        setStepFailure("unknown exception in active radiation candidate");
        cells = saved_cells;
        extensives = saved_extensives;
        return false;
    }
    for(Conserved3D const& extensive : extensives)
        spectral_repair_event.owned_radiation_energy += extensive.Erad;
    appendPendingSpectralRepairEvent(spectral_repair_event);
    commitResidualCorrectionAccounting(correction_diagnostics);
    commitSpectralRepairAccounting(spectral_repair_event, "serial_active");
    if(pending_dirichlet_defect)
        commitIndividualRadiationDefect(pending_defect_event);
    for(auto const& wake : pending_dirichlet_wakes)
        if(wake.first < individual_passive_reference_time_steps_.size()) {
            double& reference =
                individual_passive_reference_time_steps_[wake.first];
            reference = std::min(reference, wake.second);
        }
    for(SerialPassiveTransfer const& transfer : passive_transfers)
        if(transfer.applied_gain != 0 &&
           transfer.passive <
           individual_passive_reference_time_steps_.size()) {
            double& reference =
                individual_passive_reference_time_steps_[transfer.passive];
            reference = std::min(
                reference, transfer.reference_time_step);
        }
    return true;
#endif
}

void RadiationDriver::calculateIndividualTimeSteps(
    IndividualStepContext const& context,
    Tessellation3D& tess,
    std::vector<ComputationalCell3D>& cells,
    std::vector<double>& time_step_limits,
    std::vector<ComputationalCell3D> const*,
    std::vector<std::size_t> const*) const
{
    double fallback = std::numeric_limits<double>::max();
    if(!context.active_indices.empty()) {
        std::size_t const first = context.active_indices.front();
        fallback = calculate_dt(context.cellTimeStep(first), tess, cells);
    }
    for(std::size_t cell : context.active_indices)
        time_step_limits.at(cell) = std::min(time_step_limits.at(cell), fallback);
}

double RadiationDriver::individualCellTimeStep(std::size_t index,
                                               double fallback) const
{
    if(individual_context_ == nullptr)
        return fallback;
    if(!individual_context_->isActive(index))
        return 0;
    return individualScheduledTimeStep(index, fallback);
}

double RadiationDriver::individualScheduledTimeStep(std::size_t index,
                                                    double fallback) const
{
    if(individual_context_ == nullptr)
        return fallback;
    return individual_interval_fraction_ * individual_context_->cellTimeStep(index);
}

double RadiationDriver::individualFaceTimeStep(std::size_t left,
                                               std::size_t right,
                                               double fallback) const
{
    if(individual_context_ == nullptr)
        return fallback;
    if(!individual_context_->isActive(left) &&
       (right >= individual_context_->active_mask.size() ||
        !individual_context_->isActive(right)))
        return 0;
    if(right >= individual_context_->cell_time_steps.size())
        return individualCellTimeStep(left, fallback);
    return individual_interval_fraction_ *
           individual_context_->faceTimeStep(left, right);
}

bool RadiationDriver::individualCellActive(std::size_t index) const
{
    return individual_context_ == nullptr || individual_context_->isActive(index);
}

void RadiationDriver::recordIndividualFaceCoefficient(
    std::size_t left,
    std::size_t right,
    std::size_t group,
    double coefficient) const
{
    if(individual_context_ == nullptr || coefficient == 0 ||
       left >= individual_context_->active_mask.size() ||
       right >= individual_context_->active_mask.size())
        return;
    if(individual_context_->isActive(left) == individual_context_->isActive(right))
        return;
    IndividualFaceCoefficient face;
    face.left = left;
    face.right = right;
    face.group = group;
    face.coefficient = coefficient;
    face.time_step = individual_context_->faceTimeStep(left, right);
    individual_face_coefficients_.push_back(face);
}

IndividualRadiationDefectAccounting&
RadiationDriver::individualRadiationDefectAccounting() const
{
    if(individual_context_ != nullptr &&
       individual_context_->radiation_defect_accounting != nullptr)
        return *individual_context_->radiation_defect_accounting;
    return standalone_defect_accounting_;
}

double RadiationDriver::collectiveIndividualRadiationDefectScale(
    double local_candidate_start_positive_extent,
    double local_rhs_floor,
    double& candidate_start_positive_global_extent,
    double& rhs_derived_global_floor) const
{
    double global_values[2] = {
        local_candidate_start_positive_extent, local_rhs_floor};
#ifdef RICH_MPI
    requireDistributedMpiSuccess(
        MPI_Allreduce(MPI_IN_PLACE, global_values, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD),
        "MPI_Allreduce(individual radiation defect scale)");
#endif
    IndividualRadiationDefectAccounting& accounting =
        individualRadiationDefectAccounting();
    IndividualRadiationDefectConfiguration const& configuration =
        individualRadiationDefectConfiguration();
    bool const configuration_matches =
        accounting.config_version == configuration.version &&
        accounting.local_withdrawal_limit ==
            configuration.local_withdrawal_limit &&
        accounting.local_absolute_limit ==
            configuration.local_absolute_limit &&
        accounting.event_absolute_target ==
            configuration.event_absolute_target &&
        accounting.cumulative_signed_limit ==
            configuration.cumulative_signed_limit &&
        accounting.cumulative_absolute_limit ==
            configuration.cumulative_absolute_limit;
    if(!configuration_matches ||
       !std::isfinite(global_values[0]) || global_values[0] < 0 ||
       !std::isfinite(global_values[1]) || global_values[1] < 0 ||
       !std::isfinite(accounting.initial_positive_global_extent) ||
       accounting.initial_positive_global_extent < 0)
        return std::numeric_limits<double>::quiet_NaN();
    candidate_start_positive_global_extent = global_values[0];
    rhs_derived_global_floor = global_values[1];
    return std::max(accounting.initial_positive_global_extent,
                    std::max(global_values[0], global_values[1]));
}

bool RadiationDriver::validateIndividualRadiationDefect(
    IndividualRadiationDefectEvent& local_event) const
{
    struct PackedEvent
    {
        long double signed_extent;
        long double absolute_extent;
        long double passive_withdrawal_extent;
        long double passive_deposit_extent;
        double maximum_local_fraction;
        double maximum_local_tolerance_ratio;
        double candidate_start_positive_global_extent;
        double rhs_derived_global_floor;
        double normalization_scale;
        std::uint64_t face_group_terms;
        std::uint64_t duplicate_face_group_terms;
        std::uint64_t representative_active_id;
        std::uint64_t representative_passive_id;
        std::uint64_t representative_group;
        std::uint64_t representative_active_rank;
        std::uint64_t representative_rank;
        int valid;
    };
    PackedEvent local = {
        local_event.signed_extent,
        local_event.absolute_extent,
        local_event.passive_withdrawal_extent,
        local_event.passive_deposit_extent,
        local_event.maximum_local_fraction,
        local_event.maximum_local_tolerance_ratio,
        local_event.candidate_start_positive_global_extent,
        local_event.rhs_derived_global_floor,
        local_event.normalization_scale,
        local_event.face_group_terms,
        local_event.duplicate_face_group_terms,
        local_event.representative_active_id,
        local_event.representative_passive_id,
        local_event.representative_group,
        local_event.representative_active_rank,
        0,
        local_event.valid ? 1 : 0};
    std::vector<PackedEvent> events(1, local);
#ifdef RICH_MPI
    int rank = 0;
    int rank_count = 1;
    requireDistributedMpiSuccess(
        MPI_Comm_rank(MPI_COMM_WORLD, &rank),
        "MPI_Comm_rank(individual radiation defect)");
    requireDistributedMpiSuccess(
        MPI_Comm_size(MPI_COMM_WORLD, &rank_count),
        "MPI_Comm_size(individual radiation defect)");
    local.representative_rank = static_cast<std::uint64_t>(rank);
    if(sizeof(PackedEvent) >
       static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error(
            "individual radiation defect record exceeds MPI count range");
    events.resize(static_cast<std::size_t>(rank_count));
    requireDistributedMpiSuccess(
        MPI_Allgather(&local, static_cast<int>(sizeof(PackedEvent)), MPI_BYTE,
                      events.data(), static_cast<int>(sizeof(PackedEvent)),
                      MPI_BYTE, MPI_COMM_WORLD),
        "MPI_Allgather(individual radiation defect record)");
#endif

    IndividualRadiationDefectEvent event;
    long double signed_correction = 0;
    long double absolute_correction = 0;
    long double withdrawal_correction = 0;
    long double deposit_correction = 0;
    auto add_compensated = [](long double const value,
                              long double& sum,
                              long double& correction) {
        long double const adjusted = value - correction;
        long double const next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    };
    bool have_representative = false;
    bool representative_is_duplicate = false;
    for(PackedEvent const& sample : events) {
        event.valid = event.valid && sample.valid != 0;
        event.valid = event.valid &&
            std::isfinite(sample.signed_extent) &&
            std::isfinite(sample.absolute_extent) &&
            std::isfinite(sample.passive_withdrawal_extent) &&
            std::isfinite(sample.passive_deposit_extent) &&
            std::isfinite(sample.maximum_local_fraction) &&
            sample.maximum_local_fraction >= 0 &&
            std::isfinite(sample.maximum_local_tolerance_ratio) &&
            sample.maximum_local_tolerance_ratio >= 0 &&
            std::isfinite(
                sample.candidate_start_positive_global_extent) &&
            sample.candidate_start_positive_global_extent >= 0 &&
            std::isfinite(sample.rhs_derived_global_floor) &&
            sample.rhs_derived_global_floor >= 0 &&
            std::isfinite(sample.normalization_scale) &&
            sample.normalization_scale > 0;
        if(event.candidate_start_positive_global_extent == 0)
            event.candidate_start_positive_global_extent =
                sample.candidate_start_positive_global_extent;
        else if(event.candidate_start_positive_global_extent !=
                sample.candidate_start_positive_global_extent)
            event.valid = false;
        if(event.rhs_derived_global_floor == 0)
            event.rhs_derived_global_floor =
                sample.rhs_derived_global_floor;
        else if(event.rhs_derived_global_floor !=
                sample.rhs_derived_global_floor)
            event.valid = false;
        if(event.normalization_scale == 0)
            event.normalization_scale = sample.normalization_scale;
        else if(event.normalization_scale != sample.normalization_scale)
            event.valid = false;
        add_compensated(sample.signed_extent, event.signed_extent,
                        signed_correction);
        add_compensated(sample.absolute_extent, event.absolute_extent,
                        absolute_correction);
        add_compensated(sample.passive_withdrawal_extent,
                        event.passive_withdrawal_extent,
                        withdrawal_correction);
        add_compensated(sample.passive_deposit_extent,
                        event.passive_deposit_extent,
                        deposit_correction);
        if(sample.face_group_terms >
           std::numeric_limits<std::uint64_t>::max() -
               event.face_group_terms)
            event.valid = false;
        else
            event.face_group_terms += sample.face_group_terms;
        if(sample.duplicate_face_group_terms >
           std::numeric_limits<std::uint64_t>::max() -
               event.duplicate_face_group_terms)
            event.valid = false;
        else
            event.duplicate_face_group_terms +=
                sample.duplicate_face_group_terms;
        if(sample.duplicate_face_group_terms > 0)
            event.valid = false;
        event.maximum_local_fraction = std::max(
            event.maximum_local_fraction,
            sample.maximum_local_fraction);
        std::tuple<int, double, std::uint64_t, std::uint64_t,
                   std::uint64_t, std::uint64_t> const candidate(
            sample.duplicate_face_group_terms > 0 ? 0 : 1,
            -sample.maximum_local_tolerance_ratio,
            sample.representative_active_id,
            sample.representative_passive_id,
            sample.representative_group,
            sample.representative_rank);
        std::tuple<int, double, std::uint64_t, std::uint64_t,
                   std::uint64_t, std::uint64_t> const current(
            representative_is_duplicate ? 0 : 1,
            -event.maximum_local_tolerance_ratio,
            event.representative_active_id,
            event.representative_passive_id,
            event.representative_group,
            event.representative_rank);
        if(!have_representative || candidate < current) {
            have_representative = true;
            representative_is_duplicate =
                sample.duplicate_face_group_terms > 0;
            event.maximum_local_tolerance_ratio =
                sample.maximum_local_tolerance_ratio;
            event.representative_active_id =
                sample.representative_active_id;
            event.representative_passive_id =
                sample.representative_passive_id;
            event.representative_group = sample.representative_group;
            event.representative_active_rank =
                sample.representative_active_rank;
            event.representative_rank = sample.representative_rank;
        }
    }

    IndividualRadiationDefectAccounting& accounting =
        individualRadiationDefectAccounting();
    IndividualRadiationDefectConfiguration const& configuration =
        individualRadiationDefectConfiguration();
    if(event.valid) {
        event.event_absolute_fraction = static_cast<double>(
            event.absolute_extent / event.normalization_scale);
        long double const projected_signed =
            accounting.cumulative_signed_extent + event.signed_extent;
        long double const projected_absolute =
            accounting.cumulative_absolute_extent + event.absolute_extent;
        event.projected_cumulative_signed_fraction = static_cast<double>(
            std::abs(projected_signed) / event.normalization_scale);
        event.projected_cumulative_absolute_fraction = static_cast<double>(
            projected_absolute / event.normalization_scale);
        event.valid =
            std::isfinite(event.event_absolute_fraction) &&
            std::isfinite(event.projected_cumulative_signed_fraction) &&
            std::isfinite(event.projected_cumulative_absolute_fraction) &&
            event.absolute_extent >= 0 &&
            event.passive_withdrawal_extent >= 0 &&
            event.passive_deposit_extent >= 0 &&
            accounting.cumulative_absolute_extent >= 0;
    }
    local_event = event;
    if(event.valid)
        return true;

    ++accounting.defect_rejections;
    std::string const reason =
        "invalid frozen Dirichlet radiation defect accounting";
    std::ostringstream diagnostics;
    diagnostics << std::setprecision(17)
                << "valid=" << (event.valid ? 1 : 0)
                << " | signed_extent=" << event.signed_extent
                << " | absolute_extent=" << event.absolute_extent
                << " | passive_withdrawal_extent="
                << event.passive_withdrawal_extent
                << " | passive_deposit_extent="
                << event.passive_deposit_extent
                << " | normalization_scale=" << event.normalization_scale
                << " | event_absolute_fraction="
                << event.event_absolute_fraction
                << " | event_absolute_target="
                << configuration.event_absolute_target
                << " | event_limit_ratio="
                << event.event_absolute_fraction /
                       configuration.event_absolute_target
                << " | projected_cumulative_signed_fraction="
                << event.projected_cumulative_signed_fraction
                << " | cumulative_signed_limit="
                << configuration.cumulative_signed_limit
                << " | cumulative_signed_limit_ratio="
                << event.projected_cumulative_signed_fraction /
                       configuration.cumulative_signed_limit
                << " | projected_cumulative_absolute_fraction="
                << event.projected_cumulative_absolute_fraction
                << " | cumulative_absolute_limit="
                << configuration.cumulative_absolute_limit
                << " | cumulative_absolute_limit_ratio="
                << event.projected_cumulative_absolute_fraction /
                       configuration.cumulative_absolute_limit
                << " | maximum_local_fraction="
                << event.maximum_local_fraction
                << " | maximum_local_tolerance_ratio="
                << event.maximum_local_tolerance_ratio
                << " | local_tolerance_limit=1"
                << " | candidate_start_positive_global_extent="
                << event.candidate_start_positive_global_extent
                << " | rhs_derived_global_floor="
                << event.rhs_derived_global_floor
                << " | face_group_terms=" << event.face_group_terms
                << " | duplicate_face_group_terms="
                << event.duplicate_face_group_terms;
    auto append_identifier = [&diagnostics](
        char const* const name, std::uint64_t const value) {
        diagnostics << " | " << name << "=";
        if(value == std::numeric_limits<std::uint64_t>::max())
            diagnostics << "none";
        else
            diagnostics << value;
    };
    append_identifier("representative_active_id",
                      event.representative_active_id);
    append_identifier("representative_passive_id",
                      event.representative_passive_id);
    append_identifier("representative_group", event.representative_group);
    append_identifier("representative_active_rank",
                      event.representative_active_rank);
    append_identifier("representative_rank", event.representative_rank);
    std::string const failure_diagnostics = diagnostics.str();
    setStepFailure(
        reason, std::numeric_limits<size_t>::max(), failure_diagnostics);
    return false;
}

void RadiationDriver::commitIndividualRadiationDefect(
    IndividualRadiationDefectEvent const& event) const
{
    IndividualRadiationDefectAccounting& accounting =
        individualRadiationDefectAccounting();
    if(accounting.initial_positive_global_extent == 0 &&
       event.candidate_start_positive_global_extent > 0)
        accounting.initial_positive_global_extent =
            event.candidate_start_positive_global_extent;
    accounting.cumulative_signed_extent += event.signed_extent;
    accounting.cumulative_absolute_extent += event.absolute_extent;
    accounting.last_normalization_scale = event.normalization_scale;
    accounting.maximum_event_absolute_fraction = std::max(
        accounting.maximum_event_absolute_fraction,
        event.event_absolute_fraction);
    accounting.maximum_local_fraction = std::max(
        accounting.maximum_local_fraction, event.maximum_local_fraction);
    accounting.maximum_local_tolerance_ratio = std::max(
        accounting.maximum_local_tolerance_ratio,
        event.maximum_local_tolerance_ratio);
    ++accounting.accepted_dirichlet_candidates;
}
