#ifndef RICH_SPECTRAL_POSITIVITY_HPP
#define RICH_SPECTRAL_POSITIVITY_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace RadiationPositivity {

double constexpr spectral_repair_relative_limit = 1e-6;
double constexpr spectral_repair_floor_fraction = 1e-11;
// A locally large finite negative spectrum may use the controlled floor only
// when the *total negative group extent* is globally negligible.  The cell's
// net/aggregate radiation extent is diagnostic only and never controls this
// exception.
double constexpr spectral_globally_negligible_negative_fraction = 1e-8;

enum class SpectralRepairFailure
{
    None,
    NonfiniteTotalExtent,
    NegativeTotalExtent,
    NonfiniteRelativeTolerance,
    NegativeRelativeTolerance,
    NonfiniteGroupExtent,
    NonfiniteAccumulatedExtent,
    NoPositiveExtent,
    NonfiniteMaximumNegativeExtent,
    NegativeExtentExceedsTolerance,
    InvalidFloorExtent,
    NonfiniteRepairedExtent,
    NonpositiveInjectedExtent,
    InvalidGlobalComparisonScale,
    SingleCellInjectedEnergyLimit,
    GlobalInjectedEnergyLimit,
    FloorEnergyDiscrepancyLimit,
    PositivityRescueNotConverged,
    PositivityRescueSolverFailure
};

inline char const* SpectralRepairFailureLabel(
    SpectralRepairFailure const failure)
{
    switch(failure) {
    case SpectralRepairFailure::None:
        return "none";
    case SpectralRepairFailure::NonfiniteTotalExtent:
        return "nonfinite_total_extent";
    case SpectralRepairFailure::NegativeTotalExtent:
        return "negative_total_extent";
    case SpectralRepairFailure::NonfiniteRelativeTolerance:
        return "nonfinite_relative_tolerance";
    case SpectralRepairFailure::NegativeRelativeTolerance:
        return "negative_relative_tolerance";
    case SpectralRepairFailure::NonfiniteGroupExtent:
        return "nonfinite_group_extent";
    case SpectralRepairFailure::NonfiniteAccumulatedExtent:
        return "nonfinite_accumulated_extent";
    case SpectralRepairFailure::NoPositiveExtent:
        return "no_positive_extent";
    case SpectralRepairFailure::NonfiniteMaximumNegativeExtent:
        return "nonfinite_maximum_negative_extent";
    case SpectralRepairFailure::NegativeExtentExceedsTolerance:
        return "negative_extent_exceeds_tolerance";
    case SpectralRepairFailure::InvalidFloorExtent:
        return "invalid_floor_extent";
    case SpectralRepairFailure::NonfiniteRepairedExtent:
        return "nonfinite_repaired_extent";
    case SpectralRepairFailure::NonpositiveInjectedExtent:
        return "nonpositive_injected_extent";
    case SpectralRepairFailure::InvalidGlobalComparisonScale:
        return "invalid_global_comparison_scale";
    case SpectralRepairFailure::SingleCellInjectedEnergyLimit:
        return "single_cell_injected_energy_limit";
    case SpectralRepairFailure::GlobalInjectedEnergyLimit:
        return "global_injected_energy_limit";
    case SpectralRepairFailure::FloorEnergyDiscrepancyLimit:
        return "floor_energy_discrepancy_limit";
    case SpectralRepairFailure::PositivityRescueNotConverged:
        return "positivity_rescue_not_converged";
    case SpectralRepairFailure::PositivityRescueSolverFailure:
        return "positivity_rescue_solver_failure";
    }
    return "unknown";
}

struct SpectralRepairResult
{
    bool valid = true;
    bool repaired = false;
    SpectralRepairFailure failure = SpectralRepairFailure::None;
    std::size_t failure_group = std::numeric_limits<std::size_t>::max();
    double failure_extent = 0;
    std::size_t repaired_groups = 0;
    std::size_t most_negative_group = std::numeric_limits<std::size_t>::max();
    double most_negative_extent = 0;
    double positive_extent = 0;
    double negative_extent = 0;
    double injected_extent = 0;
    double floor_extent = 0;
    double relative_deficit = 0;
    double group_sum_change = 0;
    double original_group_sum = 0;
    double repaired_group_sum = 0;
};

struct PassiveRoundoffResult
{
    bool valid = true;
    bool repairable = false;
    double face_scale = 0;
    double tolerance = 0;
    double relative_extent = 0;
};

struct GloballyNegligibleNegativeExtentResult
{
    bool valid = false;
    bool negligible = false;
    double negative_extent_to_global_max_ratio =
        std::numeric_limits<double>::quiet_NaN();
};

inline GloballyNegligibleNegativeExtentResult
ClassifyGloballyNegligibleNegativeExtent(
    double const negative_extent,
    double const global_maximum_cell_extent)
{
    GloballyNegligibleNegativeExtentResult result;
    if(!std::isfinite(negative_extent) || negative_extent < 0 ||
       !std::isfinite(global_maximum_cell_extent) ||
       global_maximum_cell_extent <= 0)
        return result;
    result.negative_extent_to_global_max_ratio =
        negative_extent / global_maximum_cell_extent;
    result.valid =
        std::isfinite(result.negative_extent_to_global_max_ratio);
    result.negligible = result.valid &&
        result.negative_extent_to_global_max_ratio <=
            spectral_globally_negligible_negative_fraction;
    return result;
}

struct ControlledSpectralRepairResult
{
    SpectralRepairResult repair;
    GloballyNegligibleNegativeExtentResult global_negative;
    bool used_global_negative_exception = false;
    double diagnostic_total_to_global_max_ratio =
        std::numeric_limits<double>::quiet_NaN();
    double aggregate_sync_correction = 0;
};

struct ResidualCorrectionPositiveFloorProposal
{
    bool valid = true;
    bool required = false;
    std::size_t repaired_groups = 0;
    std::size_t representative_group = std::numeric_limits<std::size_t>::max();
    double representative_original_extent = 0;
    double positive_extent = 0;
    double floor_extent = 0;
    double injected_extent = 0;
};

/**
 * Propose, but do not apply, the residual-correction positivity floor for one
 * cell. Group values are extensive energies. The ordinary floor is 1e-11 of
 * the cell's positive spectral energy. An all-nonpositive cell uses 1e-11 of
 * the MPI-global maximum positive cell energy divided evenly over all groups.
 */
template<class GroupContainer>
inline ResidualCorrectionPositiveFloorProposal
ProposeResidualCorrectionPositiveFloor(
    GroupContainer const& groups,
    double const global_maximum_positive_cell_energy)
{
    ResidualCorrectionPositiveFloorProposal result;
    if(groups.empty()) {
        result.valid = false;
        return result;
    }

    long double positive_extent = 0;
    for(std::size_t group = 0; group < groups.size(); ++group) {
        double const extent = groups[group];
        if(!std::isfinite(extent)) {
            result.valid = false;
            return result;
        }
        if(extent < 0) {
            result.required = true;
            ++result.repaired_groups;
            if(result.representative_group ==
                   std::numeric_limits<std::size_t>::max() ||
               extent < result.representative_original_extent) {
                result.representative_group = group;
                result.representative_original_extent = extent;
            }
        }
        else
            positive_extent += static_cast<long double>(extent);
    }
    result.positive_extent = static_cast<double>(positive_extent);
    if(!result.required)
        return result;
    if(!std::isfinite(result.positive_extent) || result.positive_extent < 0 ||
       !std::isfinite(global_maximum_positive_cell_energy) ||
       global_maximum_positive_cell_energy <= 0) {
        result.valid = false;
        return result;
    }

    result.floor_extent = spectral_repair_floor_fraction *
        (result.positive_extent > 0 ? result.positive_extent :
         global_maximum_positive_cell_energy /
             static_cast<double>(groups.size()));
    if(!std::isfinite(result.floor_extent) || result.floor_extent <= 0) {
        result.valid = false;
        return result;
    }

    long double injected_extent = 0;
    for(double const extent : groups)
        if(extent < 0)
            injected_extent += static_cast<long double>(result.floor_extent) -
                static_cast<long double>(extent);
    result.injected_extent = static_cast<double>(injected_extent);
    if(!std::isfinite(result.injected_extent) ||
       result.injected_extent <= 0)
        result.valid = false;
    return result;
}

template<class GroupContainer>
inline bool ApplyResidualCorrectionPositiveFloor(
    GroupContainer& groups,
    ResidualCorrectionPositiveFloorProposal const& proposal)
{
    if(!proposal.valid)
        return false;
    if(!proposal.required)
        return true;
    for(double& extent : groups)
        if(extent < 0)
            extent = proposal.floor_extent;
    return true;
}

inline PassiveRoundoffResult ClassifyPassiveRoundoff(
    double const available,
    double const positive_gain,
    double const negative_loss)
{
    PassiveRoundoffResult result;
    if(!std::isfinite(available) || !std::isfinite(positive_gain) ||
       !std::isfinite(negative_loss) || positive_gain < 0 ||
       negative_loss < 0) {
        result.valid = false;
        return result;
    }
    result.face_scale = std::max({positive_gain, negative_loss,
                                  std::numeric_limits<double>::min()});
    result.tolerance = 512 * std::numeric_limits<double>::epsilon() *
        result.face_scale;
    if(available < 0) {
        result.relative_extent = -available / result.face_scale;
        result.repairable = -available <= result.tolerance;
    }
    return result;
}

template<class GroupContainer>
inline SpectralRepairResult RepairSmallNegativeGroupExtents(
    GroupContainer& groups,
    double const total_extent,
    double const relative_tolerance,
    bool const allow_large_finite_deficit = false)
{
    SpectralRepairResult result;
    if(!std::isfinite(total_extent)) {
        result.valid = false;
        result.failure = SpectralRepairFailure::NonfiniteTotalExtent;
        result.failure_extent = total_extent;
        return result;
    }
    if(total_extent < 0) {
        result.valid = false;
        result.failure = SpectralRepairFailure::NegativeTotalExtent;
        result.failure_extent = total_extent;
        return result;
    }
    if(!std::isfinite(relative_tolerance)) {
        result.valid = false;
        result.failure = SpectralRepairFailure::NonfiniteRelativeTolerance;
        result.failure_extent = relative_tolerance;
        return result;
    }
    if(relative_tolerance < 0) {
        result.valid = false;
        result.failure = SpectralRepairFailure::NegativeRelativeTolerance;
        result.failure_extent = relative_tolerance;
        return result;
    }

    long double original_sum = 0;
    double original_sum_double = 0;
    long double positive_extent = 0;
    long double negative_extent = 0;
    for(std::size_t group = 0; group < groups.size(); ++group) {
        double const extent = groups[group];
        if(!std::isfinite(extent)) {
            result.valid = false;
            result.failure = SpectralRepairFailure::NonfiniteGroupExtent;
            result.failure_group = group;
            result.failure_extent = extent;
            return result;
        }
        original_sum += static_cast<long double>(extent);
        original_sum_double += extent;
        if(extent < 0) {
            negative_extent -= static_cast<long double>(extent);
            ++result.repaired_groups;
            if(result.most_negative_group ==
                   std::numeric_limits<std::size_t>::max() ||
               extent < result.most_negative_extent) {
                result.most_negative_group = group;
                result.most_negative_extent = extent;
            }
        }
        else
            positive_extent += static_cast<long double>(extent);
    }
    result.positive_extent = static_cast<double>(positive_extent);
    result.negative_extent = static_cast<double>(negative_extent);
    result.original_group_sum = original_sum_double;
    result.repaired_group_sum = result.original_group_sum;
    if(!std::isfinite(result.positive_extent) ||
       !std::isfinite(result.negative_extent) ||
       !std::isfinite(result.original_group_sum)) {
        result.valid = false;
        result.failure = SpectralRepairFailure::NonfiniteAccumulatedExtent;
        return result;
    }
    if(negative_extent == 0)
        return result;

    if(!(positive_extent > 0)) {
        result.valid = false;
        result.failure = SpectralRepairFailure::NoPositiveExtent;
        return result;
    }
    result.relative_deficit = static_cast<double>(
        negative_extent / positive_extent);
    // The policy is inclusive at exactly tolerance * E_positive.  Compare the
    // reported double-precision extents so a value constructed at that exact
    // runtime boundary is not rejected by a long-double product-rounding
    // mismatch.
    double const maximum_negative_extent =
        relative_tolerance * result.positive_extent;
    if(!std::isfinite(maximum_negative_extent)) {
        result.valid = false;
        result.failure =
            SpectralRepairFailure::NonfiniteMaximumNegativeExtent;
        return result;
    }
    if(result.negative_extent > maximum_negative_extent &&
       !allow_large_finite_deficit) {
        result.valid = false;
        result.failure =
            SpectralRepairFailure::NegativeExtentExceedsTolerance;
        return result;
    }

    result.floor_extent = spectral_repair_floor_fraction *
        result.positive_extent;
    if(!std::isfinite(result.floor_extent) || result.floor_extent <= 0) {
        result.valid = false;
        result.failure = SpectralRepairFailure::InvalidFloorExtent;
        return result;
    }

    long double injected_extent = 0;
    for(double& extent : groups)
        if(extent < 0) {
            injected_extent += static_cast<long double>(result.floor_extent) -
                static_cast<long double>(extent);
            extent = result.floor_extent;
        }

    long double repaired_sum = 0;
    double repaired_sum_double = 0;
    for(double const extent : groups) {
        repaired_sum += static_cast<long double>(extent);
        repaired_sum_double += extent;
    }
    result.injected_extent = static_cast<double>(injected_extent);
    result.group_sum_change = static_cast<double>(
        repaired_sum - original_sum);
    result.repaired_group_sum = repaired_sum_double;
    if(!std::isfinite(result.injected_extent) ||
       !std::isfinite(result.group_sum_change) ||
       !std::isfinite(result.repaired_group_sum)) {
        result.valid = false;
        result.failure = SpectralRepairFailure::NonfiniteRepairedExtent;
        return result;
    }
    if(result.injected_extent <= 0) {
        result.valid = false;
        result.failure = SpectralRepairFailure::NonpositiveInjectedExtent;
        return result;
    }
    result.repaired = true;
    return result;
}

/**
 * Apply the production two-scale positivity policy atomically.
 *
 * The ordinary local test remains negative/positive <= relative_tolerance.
 * Only a finite NegativeExtentExceedsTolerance failure may use the global
 * exception, whose decision variable is negative/global-owned-max.  Erad is
 * required to agree with the pre-repair group sum to roundoff, then is set
 * exactly to the repaired group sum.  No mutation occurs on failure.
 */
template<class GroupContainer>
inline ControlledSpectralRepairResult RepairControlledNegativeGroupExtents(
    GroupContainer& groups,
    double& total_extent,
    double const relative_tolerance,
    double const global_maximum_cell_extent)
{
    ControlledSpectralRepairResult result;
    GroupContainer repaired_groups = groups;
    result.repair = RepairSmallNegativeGroupExtents(
        repaired_groups, total_extent, relative_tolerance);
    result.global_negative = ClassifyGloballyNegligibleNegativeExtent(
        result.repair.negative_extent, global_maximum_cell_extent);
    if(std::isfinite(total_extent) &&
       std::isfinite(global_maximum_cell_extent) &&
       global_maximum_cell_extent > 0)
        result.diagnostic_total_to_global_max_ratio =
            total_extent / global_maximum_cell_extent;

    if(!result.repair.valid &&
       result.repair.failure ==
           SpectralRepairFailure::NegativeExtentExceedsTolerance &&
       result.global_negative.negligible) {
        repaired_groups = groups;
        result.repair = RepairSmallNegativeGroupExtents(
            repaired_groups, total_extent, relative_tolerance, true);
        result.used_global_negative_exception =
            result.repair.valid && result.repair.repaired;
    }
    if(!result.repair.valid)
        return result;

    result.aggregate_sync_correction =
        result.repair.repaired_group_sum - total_extent;
    if(!std::isfinite(result.aggregate_sync_correction)) {
        result.repair.valid = false;
        result.repair.failure =
            SpectralRepairFailure::NonfiniteRepairedExtent;
        result.repair.failure_extent = result.aggregate_sync_correction;
        return result;
    }
    groups = repaired_groups;
    total_extent = result.repair.repaired_group_sum;
    return result;
}

} // namespace RadiationPositivity

#endif // RICH_SPECTRAL_POSITIVITY_HPP
