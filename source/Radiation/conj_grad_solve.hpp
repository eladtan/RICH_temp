#ifndef CG_HPP
#define CG_HPP 1

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <algorithm>
#include <numeric>
#ifdef RICH_MPI
#include <mpi.h>
#include "mpi/mpi_commands.hpp"
#endif
#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "SpectralPositivity.hpp"
#include "misc/utils.hpp"

namespace CG
{
#ifdef RICH_MPI
    // MPI results used by inline solver/positivity helpers must fail-stop
    // before a rank consumes incomplete collective output.
    void RequireCGMpiSuccess(
        int MpiError, char const* Operation) noexcept;
#endif

    // Matrix rows commonly exceed the former inline capacity of 20.  Keeping
    // that inline payload in every row made the full and active matrices use
    // substantial memory even after the rows spilled to heap storage.
    // Dynamically sized rows retain the exact entry order without embedding
    // unused values in every row object.
    using matrix_index_t = std::size_t;
    static_assert(sizeof(matrix_index_t) >= 8,
                  "Radiation matrix indices must support more than 2^32 cells");
    using vec = std::vector<double>;         // matrix/value row
    using vec_size_t = std::vector<matrix_index_t>; // matrix/index row
    using mat = std::vector<vec>;            // matrix (=collection of (row) vectors)
    using size_t_mat = std::vector<vec_size_t>;
    double constexpr speed_of_light = 2.99792458e10;
    size_t constexpr max_size_t = std::numeric_limits<size_t>::max();
    double constexpr stefan_boltzman = 5.670374e-5;
    double constexpr radiation_constant = 4 * stefan_boltzman / speed_of_light;
    double constexpr boltzmann_constant = 1.380649e-16;
    double constexpr electron_mass = 9.1093837015e-28;
    double constexpr max_coupling_strength = 1e2;
    double constexpr compton_optical_depth_turn_off = 100;

    enum class PreconditionerKind
    {
        ScalarJacobi,
        CellBlockJacobi,
        CellBlockGaussSeidel,
        RankLocalILU0,
        CellBlockJacobiTwoSweep,
        CellBlockJacobiFourSweep,
        CellBlockJacobiEightSweep
    };

    //! Fixed damping for every neighbor-aware block-Jacobi correction sweep.
    double constexpr cell_block_neighbor_correction_damping = 0.5;
    //! Compatibility name retained for the original two-sweep mode.
    double constexpr cell_block_two_sweep_damping =
        cell_block_neighbor_correction_damping;

    inline std::size_t CellBlockJacobiSweepCount(
        PreconditionerKind const kind)
    {
        if(kind == PreconditionerKind::CellBlockJacobiTwoSweep)
            return 2;
        if(kind == PreconditionerKind::CellBlockJacobiFourSweep)
            return 4;
        if(kind == PreconditionerKind::CellBlockJacobiEightSweep)
            return 8;
        return 1;
    }

    inline bool UsesCellBlockNeighborCorrection(
        PreconditionerKind const kind)
    {
        return CellBlockJacobiSweepCount(kind) > 1;
    }

    enum class CellBlockFallbackReason
    {
        None,
        NonFiniteEntry,
        ZeroRowScale,
        UnsafePivot,
        NonFiniteFactor
    };

    char const* PreconditionerKindLabel(PreconditionerKind kind);
    char const* CellBlockFallbackReasonLabel(CellBlockFallbackReason reason);

    // Historical MG stopping constants. The tolerance is a squared,
    // diagonal-scaled residual ratio; it is not a componentwise tolerance.
    double constexpr historical_mg_squared_tolerance = 1e-11;
    double constexpr historical_mg_effective_norm_tolerance =
        3.162277660168379e-6;
    double constexpr historical_mg_loose_error = 1e-2;
    double constexpr historical_mg_loose_max0 = 1e-6;
    double constexpr historical_mg_loose_max1 = 1e-6;
    double constexpr historical_mg_normal_max0 = 1e-5;
    double constexpr historical_mg_normal_max1 = 1e-5;
    double constexpr historical_mg_max0_factor = 4e-5;
    std::size_t constexpr historical_mg_minimum_breakdown_iteration = 10;
    double constexpr historical_mg_residual_correction_minimum_retained_fraction = 0.5;
    double constexpr
        historical_mg_positivity_continuation_trigger_fraction = 1e-10;
    double constexpr historical_mg_positive_floor_single_cell_fraction = 1e-7;
    double constexpr historical_mg_positive_floor_global_fraction = 1e-8;
    double constexpr historical_mg_positive_floor_energy_discrepancy_fraction = 1e-6;
    std::size_t constexpr historical_mg_positivity_continuation_block_iterations = 10;
    std::size_t constexpr historical_mg_positivity_continuation_maximum_blocks = 3;
    std::size_t constexpr historical_mg_positivity_continuation_iteration_budget =
        historical_mg_positivity_continuation_block_iterations *
        historical_mg_positivity_continuation_maximum_blocks;
    // Compatibility names retained for diagnostics and focused tests. These
    // are floor/halving limits, not Krylov-continuation triggers.
    double constexpr
        historical_mg_positivity_continuation_single_cell_negative_fraction =
            historical_mg_positive_floor_single_cell_fraction;
    double constexpr
        historical_mg_positivity_continuation_global_negative_fraction =
            historical_mg_positive_floor_global_fraction;
    std::size_t constexpr historical_mg_maximum_iterations = 10000;

    enum class HistoricalMGBreakdown
    {
        None,
        TinyRho,
        TinyAlphaOmega,
        NonFinite
    };

    enum class HistoricalMGBranch
    {
        Continue,
        LooseMaxima,
        NormalTolerance,
        ExtremelySmallError,
        TinyRho,
        TinyAlphaOmega,
        RejectNonFinite,
        RejectEarlyBreakdown,
        RejectBreakdownMetrics
    };

    struct HistoricalMGMetrics
    {
        double weighted_residual_squared = 0;
        double weighted_rhs_squared = 0;
        double historical_error = std::numeric_limits<double>::quiet_NaN();
        double max0 = 0;
        double max1 = 0;
        int negative = 0;
        std::size_t max0_unknown = max_size_t;
        std::size_t max1_unknown = max_size_t;
        std::size_t negative_unknown = max_size_t;
        int max0_rank = 0;
        int max1_rank = 0;
        int negative_rank = -1;
        bool finite = false;
    };

    inline bool ShouldRestartHistoricalMGFiniteBreakdown(
        HistoricalMGBreakdown const breakdown,
        HistoricalMGMetrics const& metrics,
        std::size_t const completed_iterations)
    {
        return metrics.finite && breakdown != HistoricalMGBreakdown::None &&
            breakdown != HistoricalMGBreakdown::NonFinite &&
            completed_iterations < historical_mg_minimum_breakdown_iteration;
    }

    struct HistoricalMGDecision
    {
        bool accept = false;
        bool reject = false;
        HistoricalMGBranch branch = HistoricalMGBranch::Continue;
    };

    struct HistoricalMGCorrectionAssessment
    {
        bool finite = false;
        std::uint64_t negative_group_count = 0;
        double negative_extent = 0;
    };

    enum class HistoricalMGCorrectionDisposition
    {
        Commit,
        ContinuePositivity,
        DeferPositivity,
        RejectPostCapNonphysical,
        RejectNonFinite
    };

    enum class HistoricalMGResidualCorrectionCausality
    {
        CreatedNegativity,
        WorsenedExistingNegativity,
        Neither
    };

    inline char const* HistoricalMGResidualCorrectionCausalityLabel(
        HistoricalMGResidualCorrectionCausality const Causality)
    {
        switch(Causality) {
        case HistoricalMGResidualCorrectionCausality::CreatedNegativity:
            return "created_negativity";
        case HistoricalMGResidualCorrectionCausality::WorsenedExistingNegativity:
            return "worsened_existing_negativity";
        case HistoricalMGResidualCorrectionCausality::Neither:
            return "neither";
        }
        return "unknown";
    }

    inline HistoricalMGResidualCorrectionCausality
    ClassifyHistoricalMGResidualCorrectionCausality(
        double const PreCorrectionEg,
        double const PostCorrectionEg)
    {
        if(PostCorrectionEg < 0 && PreCorrectionEg >= 0)
            return HistoricalMGResidualCorrectionCausality::CreatedNegativity;
        if(PostCorrectionEg < PreCorrectionEg && PreCorrectionEg < 0)
            return HistoricalMGResidualCorrectionCausality::
                WorsenedExistingNegativity;
        return HistoricalMGResidualCorrectionCausality::Neither;
    }

    struct HistoricalMGCorrectedNegativity
    {
        bool Finite = true;
        bool HasNegative = false;
        int Rank = -1;
        std::size_t CellId = max_size_t;
        std::size_t Group = max_size_t;
        double PreCorrectionEg = 0;
        double ResidualCorrection = 0;
        double PostCorrectionEg = 0;
        double GlobalMaximumAbsoluteEg = 0;
        double NegativeToGlobalMaximumRatio = 0;
        double GlobalNegativeExtent = 0;
        double GlobalPositiveExtent = 0;
        double GlobalNegativeToPositiveRatio = 0;
        double CorrectionScale = 1;
        HistoricalMGResidualCorrectionCausality Causality =
            HistoricalMGResidualCorrectionCausality::Neither;
    };

    inline bool CollectHistoricalMGCorrectedNegativityGlobalExtents(
        HistoricalMGCorrectedNegativity& Negativity,
        double const LocalNegativeExtent,
        double const LocalPositiveExtent,
        bool const CollectGlobally)
    {
        double Extents[2] = {LocalNegativeExtent, LocalPositiveExtent};
        int Valid = std::isfinite(Extents[0]) && Extents[0] >= 0 &&
            std::isfinite(Extents[1]) && Extents[1] >= 0 ? 1 : 0;
#ifdef RICH_MPI
        if(CollectGlobally) {
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &Valid, 1, MPI_INT, MPI_MIN,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(corrected negativity validity)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, Extents, 2, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(corrected negativity extents)");
        }
#else
        (void)CollectGlobally;
#endif
        Negativity.Finite = Negativity.Finite && Valid != 0 &&
            std::isfinite(Extents[0]) && Extents[0] >= 0 &&
            std::isfinite(Extents[1]) && Extents[1] >= 0;
        if(!Negativity.Finite)
            return false;
        Negativity.GlobalNegativeExtent = Extents[0];
        Negativity.GlobalPositiveExtent = Extents[1];
        if(Extents[0] == 0)
            Negativity.GlobalNegativeToPositiveRatio = 0;
        else if(Extents[1] > 0)
            Negativity.GlobalNegativeToPositiveRatio =
                Extents[0] / Extents[1];
        else
            Negativity.GlobalNegativeToPositiveRatio =
                std::numeric_limits<double>::infinity();
        return true;
    }

    struct HistoricalMGPositiveFloorAssessment
    {
        bool Finite = true;
        bool HasNegative = false;
        bool Eligible = true;
        bool Applied = false;
        bool CollectedGlobally = false;
        RadiationPositivity::SpectralRepairFailure Failure =
            RadiationPositivity::SpectralRepairFailure::None;
        std::uint64_t LocalFlooredCells = 0;
        std::uint64_t LocalFlooredGroups = 0;
        std::uint64_t FlooredCells = 0;
        std::uint64_t FlooredGroups = 0;
        double LocalInjectedEnergy = 0;
        double GlobalInjectedEnergy = 0;
        double GlobalPositiveEnergy = 0;
        double GlobalMaximumCellEnergy = 0;
        double MaximumCellInjectedEnergy = 0;
        double MaximumCellInjectionRatio = 0;
        double GlobalInjectionRatio = 0;
        double EnergyDiscrepancyRatio = 0;
        int RepresentativeRank = -1;
        std::size_t RepresentativeCellId = max_size_t;
        std::size_t RepresentativeGroup = max_size_t;
        double RepresentativePreCorrectionEg = 0;
        double RepresentativePostCorrectionEg = 0;
        double RepresentativeFloorEg = 0;
        double RepresentativeGroupInjectedEnergy = 0;
        double RepresentativeCellInjectedEnergy = 0;
        double RepresentativeCellVolume = 0;
        double RepresentativeCorrectionScale = 1;
        HistoricalMGResidualCorrectionCausality RepresentativeCausality =
            HistoricalMGResidualCorrectionCausality::Neither;
    };

    inline HistoricalMGPositiveFloorAssessment
    AssessAndApplyHistoricalMGPositiveFloor(
        std::vector<double> const& PreCorrectionEg,
        std::vector<double>& PostCorrectionEg,
        std::vector<double> const& CorrectionScale,
        std::vector<double> const& Volume,
        std::size_t const GroupCount,
        std::vector<std::size_t> const& CellIds,
        double const FixedMaximumPositiveCellEnergy,
        double const FixedPositiveEnergy,
        bool const CollectGlobally)
    {
        HistoricalMGPositiveFloorAssessment Result;
        Result.CollectedGlobally = CollectGlobally;
        std::size_t const Size = PostCorrectionEg.size();
        int Rank = 0;
#ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Comm_rank(MPI_COMM_WORLD, &Rank),
            "MPI_Comm_rank(positive floor assessment)");
#endif
        int Valid = GroupCount > 0 && Size % GroupCount == 0 &&
            PreCorrectionEg.size() == Size && CorrectionScale.size() == Size &&
            Volume.size() == Size && CellIds.size() == Size / GroupCount &&
            std::isfinite(FixedMaximumPositiveCellEnergy) &&
            FixedMaximumPositiveCellEnergy >= 0 &&
            std::isfinite(FixedPositiveEnergy) && FixedPositiveEnergy >= 0;
        int HasNegative = 0;
        double GlobalMaximumCellEnergy = FixedMaximumPositiveCellEnergy;
        double GlobalPositiveEnergy = FixedPositiveEnergy;
        for(std::size_t Cell = 0; Valid && Cell < CellIds.size(); ++Cell) {
            double const CellVolume = Volume[Cell * GroupCount];
            long double PositiveCellEnergy = 0;
            if(!std::isfinite(CellVolume) || CellVolume <= 0) {
                Valid = 0;
                break;
            }
            for(std::size_t Group = 0; Group < GroupCount; ++Group) {
                std::size_t const Row = Cell * GroupCount + Group;
                double const Eg = PostCorrectionEg[Row];
                if(!std::isfinite(Eg) || !std::isfinite(PreCorrectionEg[Row]) ||
                   !std::isfinite(CorrectionScale[Row]) ||
                   !std::isfinite(Volume[Row]) || Volume[Row] != CellVolume) {
                    Valid = 0;
                    break;
                }
                if(Eg < 0)
                    HasNegative = 1;
                else
                    PositiveCellEnergy += static_cast<long double>(Eg) *
                        static_cast<long double>(CellVolume);
            }
            double const PositiveCellEnergyDouble =
                static_cast<double>(PositiveCellEnergy);
            if(!std::isfinite(PositiveCellEnergyDouble) ||
               PositiveCellEnergyDouble < 0) {
                Valid = 0;
                break;
            }
            GlobalMaximumCellEnergy = std::max(
                GlobalMaximumCellEnergy, PositiveCellEnergyDouble);
            GlobalPositiveEnergy += PositiveCellEnergyDouble;
        }
#ifdef RICH_MPI
        if(CollectGlobally) {
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &Valid, 1, MPI_INT, MPI_MIN,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(positive floor validity)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &HasNegative, 1, MPI_INT,
                              MPI_MAX, MPI_COMM_WORLD),
                "MPI_Allreduce(positive floor negative flag)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &GlobalMaximumCellEnergy, 1,
                              MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
                "MPI_Allreduce(positive floor maximum cell energy)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &GlobalPositiveEnergy, 1,
                              MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
                "MPI_Allreduce(positive floor positive energy)");
        }
#else
        (void)CollectGlobally;
#endif
        Result.Finite = Valid != 0 &&
            std::isfinite(GlobalMaximumCellEnergy) &&
            std::isfinite(GlobalPositiveEnergy);
        Result.HasNegative = HasNegative != 0;
        Result.GlobalMaximumCellEnergy = GlobalMaximumCellEnergy;
        Result.GlobalPositiveEnergy = GlobalPositiveEnergy;
        if(!Result.Finite) {
            Result.Eligible = false;
            Result.Failure = RadiationPositivity::SpectralRepairFailure::
                InvalidGlobalComparisonScale;
            return Result;
        }
        if(!Result.HasNegative)
            return Result;
        if(GlobalMaximumCellEnergy <= 0 || GlobalPositiveEnergy <= 0) {
            Result.Finite = false;
            Result.Eligible = false;
            Result.Failure = RadiationPositivity::SpectralRepairFailure::
                InvalidGlobalComparisonScale;
            return Result;
        }

        double LocalMaximumCellInjectedEnergy = -1;
        std::size_t LocalRepresentativeCell = max_size_t;
        std::size_t LocalRepresentativeGroup = max_size_t;
        double LocalRepresentativeDetails[8] = {0, 0, 0, 0, 0, 0, 1, 0};
        for(std::size_t Cell = 0; Cell < CellIds.size(); ++Cell) {
            double const CellVolume = Volume[Cell * GroupCount];
            vec Groups(GroupCount, 0);
            for(std::size_t Group = 0; Group < GroupCount; ++Group)
                Groups[Group] = PostCorrectionEg[Cell * GroupCount + Group] *
                    CellVolume;
            RadiationPositivity::ResidualCorrectionPositiveFloorProposal const
                Proposal =
                    RadiationPositivity::ProposeResidualCorrectionPositiveFloor(
                        Groups, GlobalMaximumCellEnergy);
            if(!Proposal.valid) {
                Result.Finite = false;
                break;
            }
            if(!Proposal.required)
                continue;
            ++Result.LocalFlooredCells;
            Result.LocalFlooredGroups += Proposal.repaired_groups;
            Result.LocalInjectedEnergy += Proposal.injected_extent;
            bool const MoreRepresentative =
                Proposal.injected_extent > LocalMaximumCellInjectedEnergy ||
                (Proposal.injected_extent == LocalMaximumCellInjectedEnergy &&
                 CellIds[Cell] < LocalRepresentativeCell);
            if(MoreRepresentative) {
                LocalMaximumCellInjectedEnergy = Proposal.injected_extent;
                LocalRepresentativeCell = CellIds[Cell];
                LocalRepresentativeGroup = Proposal.representative_group;
                std::size_t const Row = Cell * GroupCount +
                    Proposal.representative_group;
                double const FloorEg = Proposal.floor_extent / CellVolume;
                LocalRepresentativeDetails[0] = PreCorrectionEg[Row];
                LocalRepresentativeDetails[1] = PostCorrectionEg[Row];
                LocalRepresentativeDetails[2] = FloorEg;
                LocalRepresentativeDetails[3] =
                    Proposal.floor_extent - Groups[Proposal.representative_group];
                LocalRepresentativeDetails[4] = Proposal.injected_extent;
                LocalRepresentativeDetails[5] = CellVolume;
                LocalRepresentativeDetails[6] = CorrectionScale[Row];
                LocalRepresentativeDetails[7] = static_cast<double>(
                    ClassifyHistoricalMGResidualCorrectionCausality(
                        PreCorrectionEg[Row], PostCorrectionEg[Row]));
            }
        }
        Valid = Result.Finite && std::isfinite(Result.LocalInjectedEnergy) &&
            Result.LocalInjectedEnergy >= 0 ? 1 : 0;
        unsigned long long Counts[2] = {
            static_cast<unsigned long long>(Result.LocalFlooredCells),
            static_cast<unsigned long long>(Result.LocalFlooredGroups)};
        double GlobalInjectedEnergy = Result.LocalInjectedEnergy;
        struct DoubleRank
        {
            double Value;
            int Rank;
        } LocalPick{LocalMaximumCellInjectedEnergy, Rank},
          GlobalPick{LocalMaximumCellInjectedEnergy, Rank};
#ifdef RICH_MPI
        if(CollectGlobally) {
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &Valid, 1, MPI_INT, MPI_MIN,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(positive floor application validity)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, Counts, 2,
                              MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(positive floor counts)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &GlobalInjectedEnergy, 1,
                              MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
                "MPI_Allreduce(positive floor injected energy)");
            RequireCGMpiSuccess(
                MPI_Allreduce(&LocalPick, &GlobalPick, 1, MPI_DOUBLE_INT,
                              MPI_MAXLOC, MPI_COMM_WORLD),
                "MPI_Allreduce(positive floor representative)");
        }
#endif
        Result.Finite = Valid != 0 && std::isfinite(GlobalInjectedEnergy) &&
            GlobalInjectedEnergy > 0 && GlobalPick.Value >= 0;
        if(!Result.Finite) {
            Result.Eligible = false;
            Result.Failure = RadiationPositivity::SpectralRepairFailure::
                InvalidFloorExtent;
            return Result;
        }
        Result.FlooredCells = static_cast<std::uint64_t>(Counts[0]);
        Result.FlooredGroups = static_cast<std::uint64_t>(Counts[1]);
        Result.GlobalInjectedEnergy = GlobalInjectedEnergy;
        Result.MaximumCellInjectedEnergy = GlobalPick.Value;
        Result.MaximumCellInjectionRatio = GlobalPick.Value /
            GlobalMaximumCellEnergy;
        Result.GlobalInjectionRatio = GlobalInjectedEnergy /
            GlobalPositiveEnergy;
        Result.EnergyDiscrepancyRatio = Result.GlobalInjectionRatio;
        Result.RepresentativeRank = GlobalPick.Rank;
        unsigned long long RepresentativeIdentity[2] = {
            static_cast<unsigned long long>(LocalRepresentativeCell),
            static_cast<unsigned long long>(LocalRepresentativeGroup)};
#ifdef RICH_MPI
        if(CollectGlobally) {
            RequireCGMpiSuccess(
                MPI_Bcast(RepresentativeIdentity, 2,
                          MPI_UNSIGNED_LONG_LONG,
                          Result.RepresentativeRank, MPI_COMM_WORLD),
                "MPI_Bcast(positive floor representative identity)");
            RequireCGMpiSuccess(
                MPI_Bcast(LocalRepresentativeDetails, 8, MPI_DOUBLE,
                          Result.RepresentativeRank, MPI_COMM_WORLD),
                "MPI_Bcast(positive floor representative details)");
        }
#endif
        Result.RepresentativeCellId =
            static_cast<std::size_t>(RepresentativeIdentity[0]);
        Result.RepresentativeGroup =
            static_cast<std::size_t>(RepresentativeIdentity[1]);
        Result.RepresentativePreCorrectionEg = LocalRepresentativeDetails[0];
        Result.RepresentativePostCorrectionEg = LocalRepresentativeDetails[1];
        Result.RepresentativeFloorEg = LocalRepresentativeDetails[2];
        Result.RepresentativeGroupInjectedEnergy = LocalRepresentativeDetails[3];
        Result.RepresentativeCellInjectedEnergy = LocalRepresentativeDetails[4];
        Result.RepresentativeCellVolume = LocalRepresentativeDetails[5];
        Result.RepresentativeCorrectionScale = LocalRepresentativeDetails[6];
        Result.RepresentativeCausality =
            static_cast<HistoricalMGResidualCorrectionCausality>(
                static_cast<int>(LocalRepresentativeDetails[7]));

        Result.Eligible =
            Result.MaximumCellInjectionRatio <=
                historical_mg_positive_floor_single_cell_fraction &&
            Result.GlobalInjectionRatio <=
                historical_mg_positive_floor_global_fraction &&
            Result.EnergyDiscrepancyRatio <=
                historical_mg_positive_floor_energy_discrepancy_fraction;
        if(Result.MaximumCellInjectionRatio >
           historical_mg_positive_floor_single_cell_fraction)
            Result.Failure = RadiationPositivity::SpectralRepairFailure::
                SingleCellInjectedEnergyLimit;
        else if(Result.GlobalInjectionRatio >
                historical_mg_positive_floor_global_fraction)
            Result.Failure = RadiationPositivity::SpectralRepairFailure::
                GlobalInjectedEnergyLimit;
        else if(Result.EnergyDiscrepancyRatio >
                historical_mg_positive_floor_energy_discrepancy_fraction)
            Result.Failure = RadiationPositivity::SpectralRepairFailure::
                FloorEnergyDiscrepancyLimit;
        if(!Result.Eligible)
            return Result;

        for(std::size_t Cell = 0; Cell < CellIds.size(); ++Cell) {
            double const CellVolume = Volume[Cell * GroupCount];
            vec Groups(GroupCount, 0);
            for(std::size_t Group = 0; Group < GroupCount; ++Group)
                Groups[Group] = PostCorrectionEg[Cell * GroupCount + Group] *
                    CellVolume;
            RadiationPositivity::ResidualCorrectionPositiveFloorProposal const
                Proposal =
                    RadiationPositivity::ProposeResidualCorrectionPositiveFloor(
                        Groups, GlobalMaximumCellEnergy);
            if(!RadiationPositivity::ApplyResidualCorrectionPositiveFloor(
                   Groups, Proposal)) {
                Result.Finite = false;
                Result.Eligible = false;
                Result.Failure = RadiationPositivity::SpectralRepairFailure::
                    InvalidFloorExtent;
                return Result;
            }
            for(std::size_t Group = 0; Group < GroupCount; ++Group)
                PostCorrectionEg[Cell * GroupCount + Group] =
                    Groups[Group] / CellVolume;
        }
        Result.Applied = true;
        return Result;
    }

    inline bool
    HistoricalMGCorrectedNegativityExceedsSingleCellContinuationThreshold(
        HistoricalMGCorrectedNegativity const& Negativity)
    {
        return Negativity.Finite && Negativity.HasNegative &&
            std::isfinite(Negativity.NegativeToGlobalMaximumRatio) &&
            Negativity.NegativeToGlobalMaximumRatio >
                historical_mg_positivity_continuation_trigger_fraction;
    }

    inline bool
    HistoricalMGCorrectedNegativityExceedsGlobalContinuationThreshold(
        HistoricalMGCorrectedNegativity const& Negativity)
    {
        return Negativity.Finite && Negativity.HasNegative &&
            Negativity.GlobalNegativeExtent > 0 &&
            (Negativity.GlobalPositiveExtent == 0 ||
             (std::isfinite(Negativity.GlobalNegativeToPositiveRatio) &&
              Negativity.GlobalNegativeToPositiveRatio >
                  historical_mg_positivity_continuation_global_negative_fraction));
    }

    inline bool HistoricalMGCorrectedNegativityExceedsContinuationThreshold(
        HistoricalMGCorrectedNegativity const& Negativity)
    {
        return HistoricalMGCorrectedNegativityExceedsSingleCellContinuationThreshold(
            Negativity);
    }

    enum class HistoricalMGPositivityContinuationDecision
    {
        NoExtension,
        Continue,
        Restart,
        Cleared,
        Exhausted
    };

    inline bool ShouldCommitHistoricalMGComptonFallback(
        HistoricalMGCorrectedNegativity const& Negativity,
        HistoricalMGPositivityContinuationDecision const Decision,
        bool const FallbackAvailable)
    {
        return FallbackAvailable &&
            Decision == HistoricalMGPositivityContinuationDecision::Exhausted &&
            HistoricalMGCorrectedNegativityExceedsContinuationThreshold(
                Negativity);
    }

    struct HistoricalMGPositivityContinuation
    {
        bool Active = false;
        bool OpenDiagnosticEmitted = false;
        bool Closed = false;
        std::size_t InitialIteration = 0;
        std::size_t AdditionalIterationsUsed = 0;
        std::size_t BlocksStarted = 0;
        std::size_t BlocksCompleted = 0;
        HistoricalMGCorrectedNegativity InitialNegativity;
        HistoricalMGCorrectedNegativity LastNegativity;
        HistoricalMGPositiveFloorAssessment InitialFloor;
        HistoricalMGPositiveFloorAssessment LastFloor;
    };

    inline HistoricalMGPositivityContinuationDecision
    EvaluateHistoricalMGPositivityContinuation(
        HistoricalMGCorrectedNegativity const& Negativity,
        HistoricalMGPositiveFloorAssessment const& Floor,
        std::size_t const Iterations,
        HistoricalMGPositivityContinuation& Continuation)
    {
        bool const OldTrigger =
            HistoricalMGCorrectedNegativityExceedsContinuationThreshold(
                Negativity);
        bool const PendingFiniteHalving = Floor.Finite && Floor.HasNegative &&
            !Floor.Eligible;
        bool const RequiresRescue = OldTrigger || PendingFiniteHalving;
        bool const Opening = !Continuation.Active && RequiresRescue;
        if(!Continuation.Active) {
            if(!RequiresRescue)
                return HistoricalMGPositivityContinuationDecision::
                    NoExtension;
            Continuation.Active = true;
            Continuation.InitialIteration = Iterations;
            Continuation.InitialNegativity = Negativity;
            Continuation.InitialFloor = Floor;
            Continuation.BlocksStarted = 1;
        }
        Continuation.AdditionalIterationsUsed = Iterations >=
            Continuation.InitialIteration ?
            Iterations - Continuation.InitialIteration : 0;
        Continuation.BlocksCompleted = std::min(
            historical_mg_positivity_continuation_maximum_blocks,
            Continuation.AdditionalIterationsUsed /
                historical_mg_positivity_continuation_block_iterations);
        Continuation.LastNegativity = Negativity;
        Continuation.LastFloor = Floor;
        if(!RequiresRescue)
            return HistoricalMGPositivityContinuationDecision::Cleared;
        if(Continuation.AdditionalIterationsUsed >=
           historical_mg_positivity_continuation_iteration_budget)
            return HistoricalMGPositivityContinuationDecision::Exhausted;
        if(Opening)
            return HistoricalMGPositivityContinuationDecision::Restart;
        if(Continuation.AdditionalIterationsUsed > 0 &&
           Continuation.AdditionalIterationsUsed %
                   historical_mg_positivity_continuation_block_iterations == 0) {
            Continuation.BlocksStarted = std::min(
                historical_mg_positivity_continuation_maximum_blocks,
                Continuation.BlocksCompleted + 1);
            return HistoricalMGPositivityContinuationDecision::Restart;
        }
        return HistoricalMGPositivityContinuationDecision::Continue;
    }

    inline bool HistoricalMGPositivityContinuationBlockBoundaryReached(
        HistoricalMGPositivityContinuation const& Continuation,
        std::size_t const Iterations)
    {
        if(!Continuation.Active || Iterations <= Continuation.InitialIteration)
            return false;
        std::size_t const Additional = Iterations -
            Continuation.InitialIteration;
        return Additional <=
                   historical_mg_positivity_continuation_iteration_budget &&
            Additional %
                historical_mg_positivity_continuation_block_iterations == 0;
    }

    inline bool HistoricalMGPositivityContinuationBudgetReached(
        HistoricalMGPositivityContinuation const& Continuation,
        std::size_t const Iterations)
    {
        return Continuation.Active &&
            Iterations >= Continuation.InitialIteration &&
            Iterations - Continuation.InitialIteration >=
                historical_mg_positivity_continuation_iteration_budget;
    }

    inline bool ShouldRestartHistoricalMGPositivityContinuation(
        bool const Accepted,
        bool const BlockBoundary,
        bool const BudgetReached)
    {
        return BlockBoundary && !Accepted && !BudgetReached;
    }

    inline bool ShouldAttemptHistoricalMGPositivityFinalization(
        bool const Accepted,
        bool const BlockBoundary,
        bool const BudgetReached)
    {
        return Accepted || (BlockBoundary && BudgetReached);
    }

    inline void ReportHistoricalMGPositivityContinuationOpen(
        char const* const Scope,
        HistoricalMGPositivityContinuation& Continuation,
        bool const Emit = true)
    {
        if(Continuation.OpenDiagnosticEmitted)
            return;
        Continuation.OpenDiagnosticEmitted = true;
        if(!Emit)
            return;
        HistoricalMGCorrectedNegativity const& Worst =
            Continuation.InitialNegativity;
        std::clog << std::setprecision(17)
                  << "MG_BICGSTAB_POSITIVITY_CONTINUATION_OPEN"
                  << " scope=" << Scope
                  << " initial_iteration=" << Continuation.InitialIteration
                  << " additional_iteration_budget="
                  << historical_mg_positivity_continuation_iteration_budget
                  << " single_cell_trigger_fraction="
                  << historical_mg_positivity_continuation_single_cell_negative_fraction
                  << " global_trigger_fraction="
                  << historical_mg_positivity_continuation_global_negative_fraction
                  << " single_cell_triggered="
                  << HistoricalMGCorrectedNegativityExceedsSingleCellContinuationThreshold(
                         Worst)
                  << " global_triggered="
                  << HistoricalMGCorrectedNegativityExceedsGlobalContinuationThreshold(
                         Worst)
                  << " worst_rank=" << Worst.Rank
                  << " worst_cell_id=" << Worst.CellId
                  << " worst_group=" << Worst.Group
                  << " pre_correction_Eg=" << Worst.PreCorrectionEg
                  << " residual_correction=" << Worst.ResidualCorrection
                  << " post_correction_Eg=" << Worst.PostCorrectionEg
                  << " global_maximum_absolute_Eg="
                  << Worst.GlobalMaximumAbsoluteEg
                  << " negative_to_global_maximum_ratio="
                  << Worst.NegativeToGlobalMaximumRatio
                  << " global_negative_extent="
                  << Worst.GlobalNegativeExtent
                  << " global_positive_extent="
                  << Worst.GlobalPositiveExtent
                  << " global_negative_to_positive_ratio="
                  << Worst.GlobalNegativeToPositiveRatio
                  << " correction_lambda=" << Worst.CorrectionScale
                  << " correction_minimum_retained_fraction="
                  << historical_mg_residual_correction_minimum_retained_fraction
                  << " causality="
                  << HistoricalMGResidualCorrectionCausalityLabel(
                         Worst.Causality)
                  << std::endl;
    }

    inline void ReportHistoricalMGPositivityContinuationClose(
        char const* const Scope,
        HistoricalMGPositivityContinuation& Continuation,
        HistoricalMGCorrectedNegativity const& FinalNegativity,
        char const* const Outcome,
        bool const Emit = true)
    {
        if(Continuation.Closed)
            return;
        Continuation.Closed = true;
        Continuation.LastNegativity = FinalNegativity;
        if(!Emit)
            return;
        HistoricalMGCorrectedNegativity const& Worst =
            FinalNegativity.HasNegative ? FinalNegativity :
            Continuation.InitialNegativity;
        std::clog << std::setprecision(17)
                  << "MG_BICGSTAB_POSITIVITY_CONTINUATION_CLOSE"
                  << " scope=" << Scope
                  << " initial_iteration=" << Continuation.InitialIteration
                  << " additional_iterations_used="
                  << Continuation.AdditionalIterationsUsed
                  << " worst_rank=" << Worst.Rank
                  << " worst_cell_id=" << Worst.CellId
                  << " worst_group=" << Worst.Group
                  << " pre_correction_Eg=" << Worst.PreCorrectionEg
                  << " residual_correction=" << Worst.ResidualCorrection
                  << " post_correction_Eg=" << Worst.PostCorrectionEg
                  << " global_maximum_absolute_Eg="
                  << Worst.GlobalMaximumAbsoluteEg
                  << " negative_to_global_maximum_ratio="
                  << Worst.NegativeToGlobalMaximumRatio
                  << " global_negative_extent="
                  << Worst.GlobalNegativeExtent
                  << " global_positive_extent="
                  << Worst.GlobalPositiveExtent
                  << " global_negative_to_positive_ratio="
                  << Worst.GlobalNegativeToPositiveRatio
                  << " correction_lambda=" << Worst.CorrectionScale
                  << " causality="
                  << HistoricalMGResidualCorrectionCausalityLabel(
                         Worst.Causality)
                  << " final_negative_to_global_maximum_ratio="
                  << (FinalNegativity.HasNegative ?
                      FinalNegativity.NegativeToGlobalMaximumRatio : 0)
                  << " final_global_negative_to_positive_ratio="
                  << (FinalNegativity.HasNegative ?
                      FinalNegativity.GlobalNegativeToPositiveRatio : 0)
                  << " outcome=" << Outcome << std::endl;
    }

    struct HistoricalMGResidualCorrectionLimit
    {
        bool Finite = true;
        double Scale = 1;
        std::size_t CellId = max_size_t;
        std::size_t Group = max_size_t;
        double BeforeExtent = 0;
        double UnscaledAfterExtent = 0;
        double MinimumAllowedExtent = 0;
    };

    inline bool HistoricalMGResidualCorrectionLimited(
        HistoricalMGResidualCorrectionLimit const& Limit)
    {
        return Limit.Finite && Limit.Scale < 1;
    }

    inline HistoricalMGResidualCorrectionLimit
    DetermineHistoricalMGResidualCorrectionLimit(
        double const BeforeExtent,
        double const UnscaledAfterExtent,
        std::size_t const CellId,
        std::size_t const Group)
    {
        HistoricalMGResidualCorrectionLimit Limit;
        Limit.CellId = CellId;
        Limit.Group = Group;
        Limit.BeforeExtent = BeforeExtent;
        Limit.UnscaledAfterExtent = UnscaledAfterExtent;
        if(!std::isfinite(BeforeExtent) ||
           !std::isfinite(UnscaledAfterExtent)) {
            Limit.Finite = false;
            return Limit;
        }
        // The user-selected cap is strictly local to one positive cell/group.
        // Exactly zero and already-negative groups do not constrain another
        // component's correction; downstream positivity validation owns them.
        if(BeforeExtent <= 0 || UnscaledAfterExtent >= BeforeExtent)
            return Limit;
        double const MinimumAllowedExtent =
            historical_mg_residual_correction_minimum_retained_fraction *
            BeforeExtent;
        Limit.MinimumAllowedExtent = MinimumAllowedExtent;
        if(UnscaledAfterExtent >= MinimumAllowedExtent)
            return Limit;
        double const FullReduction = BeforeExtent - UnscaledAfterExtent;
        if(!std::isfinite(FullReduction) || FullReduction <= 0) {
            Limit.Finite = false;
            return Limit;
        }
        double Scale = (BeforeExtent - MinimumAllowedExtent) /
            FullReduction;
        Scale = std::max(0.0, std::min(1.0, Scale));
        if(Scale > 0 && Scale < 1)
            Scale = std::nextafter(Scale, 0.0);
        Limit.Scale = Scale;
        return Limit;
    }

    template<class GroupContainer>
    inline std::vector<HistoricalMGResidualCorrectionLimit>
    DetermineHistoricalMGResidualCorrectionLimits(
        GroupContainer const& PreCorrectionGroups,
        GroupContainer const& UnscaledPostCorrectionGroups,
        std::size_t const CellId)
    {
        std::vector<HistoricalMGResidualCorrectionLimit> Limits;
        if(PreCorrectionGroups.size() == 0 ||
           PreCorrectionGroups.size() != UnscaledPostCorrectionGroups.size())
            return Limits;
        Limits.reserve(PreCorrectionGroups.size());
        for(std::size_t Group = 0;
            Group < PreCorrectionGroups.size(); ++Group) {
            Limits.push_back(DetermineHistoricalMGResidualCorrectionLimit(
                PreCorrectionGroups[Group],
                UnscaledPostCorrectionGroups[Group], CellId, Group));
        }
        return Limits;
    }

    inline double AppliedHistoricalMGResidualCorrectionExtent(
        HistoricalMGResidualCorrectionLimit const& Limit)
    {
        return Limit.BeforeExtent + Limit.Scale *
            (Limit.UnscaledAfterExtent - Limit.BeforeExtent);
    }

    inline HistoricalMGCorrectedNegativity
    AssessHistoricalMGCorrectedNegativity(
        std::vector<double> const& PreCorrectionEg,
        std::vector<double> const& PostCorrectionEg,
        std::vector<double> const& CorrectionScale,
        std::size_t const GroupCount,
        std::vector<std::size_t> const& CellIds,
        double const FixedMaximumAbsoluteEg,
        bool const CollectGlobally)
    {
        HistoricalMGCorrectedNegativity Result;
        std::size_t const Size = PostCorrectionEg.size();
        bool LocallyValid = GroupCount > 0 && Size % GroupCount == 0 &&
            PreCorrectionEg.size() == Size && CorrectionScale.size() == Size &&
            CellIds.size() == Size / GroupCount &&
            std::isfinite(FixedMaximumAbsoluteEg) &&
            FixedMaximumAbsoluteEg >= 0;
        double GlobalMaximumAbsoluteEg = FixedMaximumAbsoluteEg;
        for(std::size_t Row = 0; LocallyValid && Row < Size; ++Row) {
            LocallyValid = std::isfinite(PreCorrectionEg[Row]) &&
                std::isfinite(PostCorrectionEg[Row]) &&
                std::isfinite(CorrectionScale[Row]) &&
                CorrectionScale[Row] >= 0 && CorrectionScale[Row] <= 1;
            if(LocallyValid)
                GlobalMaximumAbsoluteEg = std::max(
                    GlobalMaximumAbsoluteEg,
                    std::abs(PostCorrectionEg[Row]));
        }

#ifdef RICH_MPI
        int Rank = 0;
        RequireCGMpiSuccess(
            MPI_Comm_rank(MPI_COMM_WORLD, &Rank),
            "MPI_Comm_rank(corrected negativity assessment)");
        if(CollectGlobally) {
            int GloballyValid = LocallyValid ? 1 : 0;
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &GloballyValid, 1, MPI_INT,
                              MPI_MIN, MPI_COMM_WORLD),
                "MPI_Allreduce(corrected negativity validity)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &GlobalMaximumAbsoluteEg, 1,
                              MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
                "MPI_Allreduce(corrected negativity scale)");
            LocallyValid = GloballyValid != 0;
        }
#else
        int constexpr Rank = 0;
        (void)CollectGlobally;
#endif
        Result.Finite = LocallyValid &&
            std::isfinite(GlobalMaximumAbsoluteEg) &&
            GlobalMaximumAbsoluteEg >= 0;
        Result.GlobalMaximumAbsoluteEg = GlobalMaximumAbsoluteEg;
        if(!Result.Finite)
            return Result;

        double LocalSeverity = -1;
        for(std::size_t Row = 0; Row < Size; ++Row) {
            if(PostCorrectionEg[Row] >= 0 ||
               std::abs(PostCorrectionEg[Row]) <= LocalSeverity)
                continue;
            LocalSeverity = std::abs(PostCorrectionEg[Row]);
            Result.HasNegative = true;
            Result.Rank = Rank;
            Result.CellId = CellIds[Row / GroupCount];
            Result.Group = Row % GroupCount;
            Result.PreCorrectionEg = PreCorrectionEg[Row];
            Result.ResidualCorrection =
                PostCorrectionEg[Row] - PreCorrectionEg[Row];
            Result.PostCorrectionEg = PostCorrectionEg[Row];
            Result.CorrectionScale = CorrectionScale[Row];
            Result.Causality =
                ClassifyHistoricalMGResidualCorrectionCausality(
                    Result.PreCorrectionEg, Result.PostCorrectionEg);
        }
        if(Result.HasNegative && GlobalMaximumAbsoluteEg > 0)
            Result.NegativeToGlobalMaximumRatio =
                std::abs(Result.PostCorrectionEg) /
                GlobalMaximumAbsoluteEg;

#ifdef RICH_MPI
        if(CollectGlobally) {
            struct DoubleRank
            {
                double Value;
                int Rank;
            } LocalPick{LocalSeverity, Rank}, GlobalPick{-1, 0};
            RequireCGMpiSuccess(
                MPI_Allreduce(&LocalPick, &GlobalPick, 1, MPI_DOUBLE_INT,
                              MPI_MAXLOC, MPI_COMM_WORLD),
                "MPI_Allreduce(corrected negativity representative)");
            if(GlobalPick.Value < 0) {
                Result.HasNegative = false;
                Result.Rank = -1;
                return Result;
            }

            unsigned long long Identity[2] = {0, 0};
            int Causality = 0;
            double Details[6] = {0, 0, 0, 0, 0, 1};
            if(Rank == GlobalPick.Rank) {
                Identity[0] = static_cast<unsigned long long>(Result.CellId);
                Identity[1] = static_cast<unsigned long long>(Result.Group);
                Causality = static_cast<int>(Result.Causality);
                Details[0] = Result.PreCorrectionEg;
                Details[1] = Result.ResidualCorrection;
                Details[2] = Result.PostCorrectionEg;
                Details[3] = Result.GlobalMaximumAbsoluteEg;
                Details[4] = Result.NegativeToGlobalMaximumRatio;
                Details[5] = Result.CorrectionScale;
            }
            RequireCGMpiSuccess(
                MPI_Bcast(Identity, 2, MPI_UNSIGNED_LONG_LONG,
                          GlobalPick.Rank, MPI_COMM_WORLD),
                "MPI_Bcast(corrected negativity identity)");
            RequireCGMpiSuccess(
                MPI_Bcast(&Causality, 1, MPI_INT, GlobalPick.Rank,
                          MPI_COMM_WORLD),
                "MPI_Bcast(corrected negativity causality)");
            RequireCGMpiSuccess(
                MPI_Bcast(Details, 6, MPI_DOUBLE, GlobalPick.Rank,
                          MPI_COMM_WORLD),
                "MPI_Bcast(corrected negativity details)");
            Result.HasNegative = true;
            Result.Rank = GlobalPick.Rank;
            Result.CellId = static_cast<std::size_t>(Identity[0]);
            Result.Group = static_cast<std::size_t>(Identity[1]);
            Result.Causality = static_cast<
                HistoricalMGResidualCorrectionCausality>(Causality);
            Result.PreCorrectionEg = Details[0];
            Result.ResidualCorrection = Details[1];
            Result.PostCorrectionEg = Details[2];
            Result.GlobalMaximumAbsoluteEg = Details[3];
            Result.NegativeToGlobalMaximumRatio = Details[4];
            Result.CorrectionScale = Details[5];
        }
#endif
        return Result;
    }

    struct HistoricalMGResidualCorrectionDiagnostics
    {
        bool finite = true;
        bool available = false;
        bool compton_fallback_candidate = false;
        std::size_t group_count = 0;
        std::vector<double> pre_correction_solution;
        std::vector<double> correction_scale;
        std::vector<double> signed_energy_bias_by_group;
        std::vector<double> absolute_energy_bias_by_group;
        std::uint64_t limited_group_count = 0;
        double signed_energy_bias = 0;
        double absolute_energy_bias = 0;
        double minimum_scale = 1;
        std::size_t limiting_cell_id = max_size_t;
        std::size_t limiting_group = max_size_t;
        double limiting_before_extent = 0;
        double limiting_unscaled_after_extent = 0;
        double limiting_applied_after_extent = 0;
        std::string failure_reason;
        RadiationPositivity::SpectralRepairFailure failure_class =
            RadiationPositivity::SpectralRepairFailure::None;
        std::size_t failure_cell_id = max_size_t;
        std::size_t failure_group = max_size_t;
        double failure_signed_group_extent = 0;
        double failure_negative_extent = 0;
        double failure_positive_extent = 0;
        double failure_relative_deficit = 0;
        double failure_global_maximum_cell_extent = 0;
        int failure_rank = -1;
        double failure_pre_correction_Eg = 0;
        double failure_residual_correction = 0;
        double failure_post_correction_Eg = 0;
        double failure_global_maximum_absolute_Eg = 0;
        double failure_negative_to_global_maximum_ratio = 0;
        double failure_global_negative_extent = 0;
        double failure_global_positive_extent = 0;
        double failure_global_negative_to_positive_ratio = 0;
        double failure_correction_scale = 1;
        HistoricalMGResidualCorrectionCausality failure_causality =
            HistoricalMGResidualCorrectionCausality::Neither;
        std::size_t failure_solver_iterations = 0;
        std::size_t positivity_initial_iteration = 0;
        std::size_t positivity_additional_iterations = 0;
        std::size_t positivity_blocks_started = 0;
        std::size_t positivity_blocks_completed = 0;
        bool positive_floor_applied = false;
        bool positive_floor_collected_globally = false;
        std::uint64_t positive_floor_local_cells = 0;
        std::uint64_t positive_floor_local_groups = 0;
        std::uint64_t positive_floor_global_cells = 0;
        std::uint64_t positive_floor_global_groups = 0;
        double positive_floor_local_injected_energy = 0;
        double positive_floor_global_injected_energy = 0;
        double positive_floor_global_positive_energy = 0;
        double positive_floor_global_maximum_cell_energy = 0;
        double positive_floor_maximum_cell_injection_ratio = 0;
        double positive_floor_global_injection_ratio = 0;
        double positive_floor_energy_discrepancy_ratio = 0;
        int positive_floor_representative_rank = -1;
        std::size_t positive_floor_representative_cell_id = max_size_t;
        std::size_t positive_floor_representative_group = max_size_t;
        double positive_floor_representative_pre_correction_Eg = 0;
        double positive_floor_representative_post_correction_Eg = 0;
        double positive_floor_representative_floor_Eg = 0;
        double positive_floor_representative_group_injected_energy = 0;
        double positive_floor_representative_cell_injected_energy = 0;
        double positive_floor_representative_cell_volume = 0;
        double positive_floor_representative_correction_scale = 1;
        HistoricalMGResidualCorrectionCausality
            positive_floor_representative_causality =
                HistoricalMGResidualCorrectionCausality::Neither;
        bool post_floor_true_residual_evaluated = false;
        bool post_floor_true_residual_finite = true;
        double post_floor_true_residual_error = 0;
    };

    inline void ResetHistoricalMGResidualCorrectionDiagnostics(
        HistoricalMGResidualCorrectionDiagnostics& Diagnostics,
        std::size_t const GroupCount,
        std::size_t const SolutionSize)
    {
        Diagnostics = HistoricalMGResidualCorrectionDiagnostics{};
        Diagnostics.group_count = GroupCount;
        Diagnostics.pre_correction_solution.assign(SolutionSize, 0);
        Diagnostics.correction_scale.assign(SolutionSize, 1);
        Diagnostics.signed_energy_bias_by_group.assign(GroupCount, 0);
        Diagnostics.absolute_energy_bias_by_group.assign(GroupCount, 0);
    }

    inline void RecordHistoricalMGPositiveFloor(
        HistoricalMGPositiveFloorAssessment const& Floor,
        HistoricalMGResidualCorrectionDiagnostics& Diagnostics)
    {
        Diagnostics.positive_floor_applied = Floor.Applied;
        Diagnostics.positive_floor_collected_globally =
            Floor.CollectedGlobally;
        Diagnostics.positive_floor_local_cells = Floor.LocalFlooredCells;
        Diagnostics.positive_floor_local_groups = Floor.LocalFlooredGroups;
        Diagnostics.positive_floor_global_cells = Floor.FlooredCells;
        Diagnostics.positive_floor_global_groups = Floor.FlooredGroups;
        Diagnostics.positive_floor_local_injected_energy =
            Floor.LocalInjectedEnergy;
        Diagnostics.positive_floor_global_injected_energy =
            Floor.GlobalInjectedEnergy;
        Diagnostics.positive_floor_global_positive_energy =
            Floor.GlobalPositiveEnergy;
        Diagnostics.positive_floor_global_maximum_cell_energy =
            Floor.GlobalMaximumCellEnergy;
        Diagnostics.positive_floor_maximum_cell_injection_ratio =
            Floor.MaximumCellInjectionRatio;
        Diagnostics.positive_floor_global_injection_ratio =
            Floor.GlobalInjectionRatio;
        Diagnostics.positive_floor_energy_discrepancy_ratio =
            Floor.EnergyDiscrepancyRatio;
        Diagnostics.positive_floor_representative_rank =
            Floor.RepresentativeRank;
        Diagnostics.positive_floor_representative_cell_id =
            Floor.RepresentativeCellId;
        Diagnostics.positive_floor_representative_group =
            Floor.RepresentativeGroup;
        Diagnostics.positive_floor_representative_pre_correction_Eg =
            Floor.RepresentativePreCorrectionEg;
        Diagnostics.positive_floor_representative_post_correction_Eg =
            Floor.RepresentativePostCorrectionEg;
        Diagnostics.positive_floor_representative_floor_Eg =
            Floor.RepresentativeFloorEg;
        Diagnostics.positive_floor_representative_group_injected_energy =
            Floor.RepresentativeGroupInjectedEnergy;
        Diagnostics.positive_floor_representative_cell_injected_energy =
            Floor.RepresentativeCellInjectedEnergy;
        Diagnostics.positive_floor_representative_cell_volume =
            Floor.RepresentativeCellVolume;
        Diagnostics.positive_floor_representative_correction_scale =
            Floor.RepresentativeCorrectionScale;
        Diagnostics.positive_floor_representative_causality =
            Floor.RepresentativeCausality;
    }

    inline void RecordHistoricalMGPositivityContinuation(
        HistoricalMGPositivityContinuation const& Continuation,
        HistoricalMGResidualCorrectionDiagnostics& Diagnostics)
    {
        if(!Continuation.Active)
            return;
        Diagnostics.positivity_initial_iteration =
            Continuation.InitialIteration;
        Diagnostics.positivity_additional_iterations =
            Continuation.AdditionalIterationsUsed;
        Diagnostics.positivity_blocks_started = Continuation.BlocksStarted;
        Diagnostics.positivity_blocks_completed =
            Continuation.BlocksCompleted;
    }

    inline void RecordHistoricalMGResidualCorrectionLimit(
        HistoricalMGResidualCorrectionLimit const& Limit,
        HistoricalMGResidualCorrectionDiagnostics& Diagnostics)
    {
        Diagnostics.available = true;
        if(!Limit.Finite) {
            Diagnostics.finite = false;
            return;
        }
        bool const Limited = HistoricalMGResidualCorrectionLimited(Limit);
        double const AppliedAfterExtent = Limited
            ? AppliedHistoricalMGResidualCorrectionExtent(Limit)
            : Limit.UnscaledAfterExtent;
        // Signed bias is accepted capped extent minus the full residual-
        // corrected extent. Positive bias means the cap retained radiation.
        double const Bias = Limited
            ? AppliedAfterExtent - Limit.UnscaledAfterExtent : 0.0;
        if(!std::isfinite(AppliedAfterExtent) || !std::isfinite(Bias) ||
           Limit.Group >= Diagnostics.signed_energy_bias_by_group.size()) {
            Diagnostics.finite = false;
            return;
        }
        Diagnostics.signed_energy_bias_by_group[Limit.Group] += Bias;
        Diagnostics.absolute_energy_bias_by_group[Limit.Group] +=
            std::abs(Bias);
        Diagnostics.signed_energy_bias += Bias;
        Diagnostics.absolute_energy_bias += std::abs(Bias);
        if(Limited) {
            ++Diagnostics.limited_group_count;
            if(Limit.Scale < Diagnostics.minimum_scale) {
                Diagnostics.minimum_scale = Limit.Scale;
                Diagnostics.limiting_cell_id = Limit.CellId;
                Diagnostics.limiting_group = Limit.Group;
                Diagnostics.limiting_before_extent = Limit.BeforeExtent;
                Diagnostics.limiting_unscaled_after_extent =
                    Limit.UnscaledAfterExtent;
                Diagnostics.limiting_applied_after_extent =
                    AppliedAfterExtent;
            }
        }
    }

#ifdef RICH_MPI
    inline void CollectHistoricalMGResidualCorrectionLimitingDiagnostic(
        HistoricalMGResidualCorrectionDiagnostics& Diagnostics)
    {
        int Rank = 0;
        RequireCGMpiSuccess(
            MPI_Comm_rank(MPI_COMM_WORLD, &Rank),
            "MPI_Comm_rank(residual correction limiter)");
        struct DoubleRank
        {
            double Value;
            int Rank;
        } LocalPick{
            Diagnostics.limited_group_count > 0
                ? Diagnostics.minimum_scale : 1.0,
            Rank}, GlobalPick{1.0, 0};
        RequireCGMpiSuccess(
            MPI_Allreduce(&LocalPick, &GlobalPick, 1, MPI_DOUBLE_INT,
                          MPI_MINLOC, MPI_COMM_WORLD),
            "MPI_Allreduce(residual correction limiter)");
        if(GlobalPick.Value >= 1.0)
            return;
        unsigned long long Identity[2] = {0, 0};
        double Details[3] = {0, 0, 0};
        if(Rank == GlobalPick.Rank) {
            Identity[0] = static_cast<unsigned long long>(
                Diagnostics.limiting_cell_id);
            Identity[1] = static_cast<unsigned long long>(
                Diagnostics.limiting_group);
            Details[0] = Diagnostics.limiting_before_extent;
            Details[1] = Diagnostics.limiting_unscaled_after_extent;
            Details[2] = Diagnostics.limiting_applied_after_extent;
        }
        RequireCGMpiSuccess(
            MPI_Bcast(Identity, 2, MPI_UNSIGNED_LONG_LONG, GlobalPick.Rank,
                      MPI_COMM_WORLD),
            "MPI_Bcast(residual correction limiter identity)");
        RequireCGMpiSuccess(
            MPI_Bcast(Details, 3, MPI_DOUBLE, GlobalPick.Rank,
                      MPI_COMM_WORLD),
            "MPI_Bcast(residual correction limiter details)");
        Diagnostics.minimum_scale = GlobalPick.Value;
        Diagnostics.limiting_cell_id =
            static_cast<std::size_t>(Identity[0]);
        Diagnostics.limiting_group =
            static_cast<std::size_t>(Identity[1]);
        Diagnostics.limiting_before_extent = Details[0];
        Diagnostics.limiting_unscaled_after_extent = Details[1];
        Diagnostics.limiting_applied_after_extent = Details[2];
    }
#endif

    struct HistoricalMGCorrectionSpectralFailure
    {
        bool CausedRejection = false;
        RadiationPositivity::SpectralRepairFailure Failure =
            RadiationPositivity::SpectralRepairFailure::None;
        std::size_t CellId = max_size_t;
        std::size_t Group = max_size_t;
        double SignedGroupExtent = 0;
        double NegativeExtent = 0;
        double PositiveExtent = 0;
        double RelativeDeficit = 0;
        double GlobalMaximumCellExtent = 0;
        double NegativeExtentToGlobalMaximumRatio =
            std::numeric_limits<double>::quiet_NaN();
        int Rank = -1;
        double PreCorrectionEg = 0;
        double ResidualCorrection = 0;
        double PostCorrectionEg = 0;
        double GlobalMaximumAbsoluteEg = 0;
        double NegativeToGlobalMaximumAbsoluteRatio = 0;
        double GlobalNegativeExtent = 0;
        double GlobalPositiveExtent = 0;
        double GlobalNegativeToPositiveRatio = 0;
        double CorrectionScale = 1;
        HistoricalMGResidualCorrectionCausality Causality =
            HistoricalMGResidualCorrectionCausality::Neither;
    };

    inline HistoricalMGCorrectionSpectralFailure
    HistoricalMGPositiveFloorFailure(
        HistoricalMGPositiveFloorAssessment const& Floor,
        HistoricalMGCorrectedNegativity const& Negativity)
    {
        HistoricalMGCorrectionSpectralFailure Result;
        if(!Floor.HasNegative || (Floor.Finite && Floor.Eligible))
            return Result;
        Result.CausedRejection = true;
        Result.Failure = Floor.Failure;
        Result.CellId = Floor.RepresentativeCellId;
        Result.Group = Floor.RepresentativeGroup;
        Result.SignedGroupExtent = Floor.RepresentativePostCorrectionEg *
            Floor.RepresentativeCellVolume;
        Result.NegativeExtent = std::max(
            0.0, -Result.SignedGroupExtent);
        Result.PositiveExtent = Floor.GlobalPositiveEnergy;
        Result.RelativeDeficit = Floor.MaximumCellInjectionRatio;
        Result.GlobalMaximumCellExtent = Floor.GlobalMaximumCellEnergy;
        Result.NegativeExtentToGlobalMaximumRatio =
            Floor.MaximumCellInjectionRatio;
        Result.Rank = Floor.RepresentativeRank;
        Result.PreCorrectionEg = Floor.RepresentativePreCorrectionEg;
        Result.ResidualCorrection = Floor.RepresentativePostCorrectionEg -
            Floor.RepresentativePreCorrectionEg;
        Result.PostCorrectionEg = Floor.RepresentativePostCorrectionEg;
        Result.GlobalMaximumAbsoluteEg =
            Negativity.GlobalMaximumAbsoluteEg;
        if(Negativity.GlobalMaximumAbsoluteEg > 0)
            Result.NegativeToGlobalMaximumAbsoluteRatio =
                std::abs(Result.PostCorrectionEg) /
                Negativity.GlobalMaximumAbsoluteEg;
        Result.GlobalNegativeExtent = Negativity.GlobalNegativeExtent;
        Result.GlobalPositiveExtent = Negativity.GlobalPositiveExtent;
        Result.GlobalNegativeToPositiveRatio =
            Negativity.GlobalNegativeToPositiveRatio;
        Result.CorrectionScale = Floor.RepresentativeCorrectionScale;
        Result.Causality = Floor.RepresentativeCausality;
        return Result;
    }

    inline void RecordHistoricalMGResidualCorrectionFailure(
        HistoricalMGCorrectionSpectralFailure const& Failure,
        HistoricalMGResidualCorrectionDiagnostics& Diagnostics,
        std::size_t const SolverIterations = 0,
        HistoricalMGPositivityContinuation const* const Continuation = nullptr)
    {
        switch(Failure.Failure) {
        case RadiationPositivity::SpectralRepairFailure::
                InvalidGlobalComparisonScale:
            Diagnostics.failure_reason =
                "historical_positive_floor_invalid_comparison_scale";
            break;
        case RadiationPositivity::SpectralRepairFailure::
                SingleCellInjectedEnergyLimit:
            Diagnostics.failure_reason =
                "historical_positive_floor_single_cell_limit";
            break;
        case RadiationPositivity::SpectralRepairFailure::
                GlobalInjectedEnergyLimit:
            Diagnostics.failure_reason =
                "historical_positive_floor_global_limit";
            break;
        case RadiationPositivity::SpectralRepairFailure::
                FloorEnergyDiscrepancyLimit:
            Diagnostics.failure_reason =
                "historical_positive_floor_energy_discrepancy";
            break;
        default:
            Diagnostics.failure_reason =
                "historical_residual_correction_post_cap_nonphysical";
            break;
        }
        Diagnostics.failure_class = Failure.Failure;
        Diagnostics.failure_cell_id = Failure.CellId;
        Diagnostics.failure_group = Failure.Group;
        Diagnostics.failure_signed_group_extent = Failure.SignedGroupExtent;
        Diagnostics.failure_negative_extent = Failure.NegativeExtent;
        Diagnostics.failure_positive_extent = Failure.PositiveExtent;
        Diagnostics.failure_relative_deficit = Failure.RelativeDeficit;
        Diagnostics.failure_global_maximum_cell_extent =
            Failure.GlobalMaximumCellExtent;
        Diagnostics.failure_rank = Failure.Rank;
        Diagnostics.failure_pre_correction_Eg = Failure.PreCorrectionEg;
        Diagnostics.failure_residual_correction =
            Failure.ResidualCorrection;
        Diagnostics.failure_post_correction_Eg = Failure.PostCorrectionEg;
        Diagnostics.failure_global_maximum_absolute_Eg =
            Failure.GlobalMaximumAbsoluteEg;
        Diagnostics.failure_negative_to_global_maximum_ratio =
            Failure.NegativeToGlobalMaximumAbsoluteRatio;
        Diagnostics.failure_global_negative_extent =
            Failure.GlobalNegativeExtent;
        Diagnostics.failure_global_positive_extent =
            Failure.GlobalPositiveExtent;
        Diagnostics.failure_global_negative_to_positive_ratio =
            Failure.GlobalNegativeToPositiveRatio;
        Diagnostics.failure_correction_scale = Failure.CorrectionScale;
        Diagnostics.failure_causality = Failure.Causality;
        Diagnostics.failure_solver_iterations = SolverIterations;
        if(Continuation != nullptr)
            RecordHistoricalMGPositivityContinuation(
                *Continuation, Diagnostics);
    }

    inline void RecordHistoricalMGPositivityRescueFailure(
        HistoricalMGPositivityContinuation const& Continuation,
        HistoricalMGResidualCorrectionDiagnostics& Diagnostics,
        std::size_t const SolverIterations,
        char const* const Reason,
        RadiationPositivity::SpectralRepairFailure const FallbackClass =
            RadiationPositivity::SpectralRepairFailure::
                PositivityRescueNotConverged)
    {
        HistoricalMGPositiveFloorAssessment const& Floor =
            Continuation.LastFloor;
        HistoricalMGCorrectedNegativity const& Negativity =
            Continuation.LastNegativity;
        HistoricalMGCorrectionSpectralFailure const FloorFailure =
            HistoricalMGPositiveFloorFailure(Floor, Negativity);
        if(FloorFailure.CausedRejection)
            RecordHistoricalMGResidualCorrectionFailure(
                FloorFailure, Diagnostics, SolverIterations, &Continuation);
        else {
            Diagnostics.failure_class =
                FallbackClass;
            Diagnostics.failure_rank = Floor.HasNegative ?
                Floor.RepresentativeRank : Negativity.Rank;
            Diagnostics.failure_cell_id = Floor.HasNegative ?
                Floor.RepresentativeCellId : Negativity.CellId;
            Diagnostics.failure_group = Floor.HasNegative ?
                Floor.RepresentativeGroup : Negativity.Group;
            Diagnostics.failure_pre_correction_Eg = Floor.HasNegative ?
                Floor.RepresentativePreCorrectionEg :
                Negativity.PreCorrectionEg;
            Diagnostics.failure_post_correction_Eg = Floor.HasNegative ?
                Floor.RepresentativePostCorrectionEg :
                Negativity.PostCorrectionEg;
            Diagnostics.failure_residual_correction =
                Diagnostics.failure_post_correction_Eg -
                Diagnostics.failure_pre_correction_Eg;
            Diagnostics.failure_correction_scale = Floor.HasNegative ?
                Floor.RepresentativeCorrectionScale :
                Negativity.CorrectionScale;
            Diagnostics.failure_causality = Floor.HasNegative ?
                Floor.RepresentativeCausality : Negativity.Causality;
            Diagnostics.failure_signed_group_extent = Floor.HasNegative ?
                Floor.RepresentativePostCorrectionEg *
                    Floor.RepresentativeCellVolume : 0;
            Diagnostics.failure_negative_extent = std::max(
                0.0, -Diagnostics.failure_signed_group_extent);
            Diagnostics.failure_positive_extent = Floor.GlobalPositiveEnergy;
            Diagnostics.failure_relative_deficit = Floor.HasNegative ?
                Floor.MaximumCellInjectionRatio :
                Negativity.NegativeToGlobalMaximumRatio;
            Diagnostics.failure_global_maximum_cell_extent =
                Floor.GlobalMaximumCellEnergy;
            Diagnostics.failure_global_maximum_absolute_Eg =
                Negativity.GlobalMaximumAbsoluteEg;
            Diagnostics.failure_negative_to_global_maximum_ratio =
                Negativity.NegativeToGlobalMaximumRatio;
            Diagnostics.failure_global_negative_extent =
                Negativity.GlobalNegativeExtent;
            Diagnostics.failure_global_positive_extent =
                Negativity.GlobalPositiveExtent;
            Diagnostics.failure_global_negative_to_positive_ratio =
                Negativity.GlobalNegativeToPositiveRatio;
            Diagnostics.failure_solver_iterations = SolverIterations;
            RecordHistoricalMGPositivityContinuation(
                Continuation, Diagnostics);
        }
        Diagnostics.failure_reason = Reason;
    }

    inline void AppendHistoricalMGResidualCorrectionFailureDiagnostics(
        std::ostream& Stream,
        HistoricalMGResidualCorrectionDiagnostics const& Diagnostics,
        std::size_t const TotalSolverIterations)
    {
        Stream << Diagnostics.failure_reason
               << " timestep_halving_reason=" << Diagnostics.failure_reason
               << " failure_class="
               << RadiationPositivity::SpectralRepairFailureLabel(
                      Diagnostics.failure_class)
               << " positivity_failure_class="
               << RadiationPositivity::SpectralRepairFailureLabel(
                      Diagnostics.failure_class)
               << " failure_rank=" << Diagnostics.failure_rank
               << " failure_cell_id=" << Diagnostics.failure_cell_id
               << " group=" << Diagnostics.failure_group
               << " Eg_before_residual_correction="
               << Diagnostics.failure_pre_correction_Eg
               << " residual_correction_added="
               << Diagnostics.failure_residual_correction
               << " Eg_after_residual_correction="
               << Diagnostics.failure_post_correction_Eg
               << " global_maximum_absolute_Eg="
               << Diagnostics.failure_global_maximum_absolute_Eg
               << " negative_Eg_to_global_maximum_ratio="
               << Diagnostics.failure_negative_to_global_maximum_ratio
               << " positivity_restart_trigger_fraction="
               << historical_mg_positivity_continuation_trigger_fraction
               << " global_negative_extent="
               << Diagnostics.failure_global_negative_extent
               << " global_positive_extent="
               << Diagnostics.failure_global_positive_extent
               << " global_negative_to_positive_ratio="
               << Diagnostics.failure_global_negative_to_positive_ratio
               << " global_negative_fraction_threshold="
               << historical_mg_positivity_continuation_global_negative_fraction
               << " correction_lambda="
               << Diagnostics.failure_correction_scale
               << " correction_cap_minimum_retained_fraction="
               << historical_mg_residual_correction_minimum_retained_fraction
               << " residual_correction_causality="
               << HistoricalMGResidualCorrectionCausalityLabel(
                      Diagnostics.failure_causality)
               << " solver_iterations=" << TotalSolverIterations
               << " failing_solve_iterations="
               << Diagnostics.failure_solver_iterations
               << " positivity_initial_iteration="
               << Diagnostics.positivity_initial_iteration
               << " positivity_additional_iterations="
               << Diagnostics.positivity_additional_iterations
               << " positivity_blocks_started="
               << Diagnostics.positivity_blocks_started
               << " positivity_blocks_completed="
               << Diagnostics.positivity_blocks_completed
               << " positive_floor_applied="
               << Diagnostics.positive_floor_applied
               << " positive_floor_global_cells="
               << Diagnostics.positive_floor_global_cells
               << " positive_floor_global_groups="
               << Diagnostics.positive_floor_global_groups
               << " positive_floor_global_injected_energy="
               << Diagnostics.positive_floor_global_injected_energy
               << " positive_floor_global_positive_energy="
               << Diagnostics.positive_floor_global_positive_energy
               << " positive_floor_global_maximum_cell_energy="
               << Diagnostics.positive_floor_global_maximum_cell_energy
               << " positive_floor_maximum_cell_injection_ratio="
               << Diagnostics.positive_floor_maximum_cell_injection_ratio
               << " positive_floor_single_cell_limit="
               << historical_mg_positive_floor_single_cell_fraction
               << " positive_floor_global_injection_ratio="
               << Diagnostics.positive_floor_global_injection_ratio
               << " positive_floor_global_limit="
               << historical_mg_positive_floor_global_fraction
               << " positive_floor_energy_discrepancy_ratio="
               << Diagnostics.positive_floor_energy_discrepancy_ratio
               << " positive_floor_energy_discrepancy_limit="
               << historical_mg_positive_floor_energy_discrepancy_fraction
               << " positive_floor_representative_rank="
               << Diagnostics.positive_floor_representative_rank
               << " positive_floor_representative_cell_id="
               << Diagnostics.positive_floor_representative_cell_id
               << " positive_floor_representative_group="
               << Diagnostics.positive_floor_representative_group
               << " positive_floor_Eg_before_residual_correction="
               << Diagnostics.positive_floor_representative_pre_correction_Eg
               << " positive_floor_Eg_after_residual_correction="
               << Diagnostics.positive_floor_representative_post_correction_Eg
               << " positive_floor_Eg="
               << Diagnostics.positive_floor_representative_floor_Eg
               << " positive_floor_group_injected_energy="
               << Diagnostics.positive_floor_representative_group_injected_energy
               << " positive_floor_cell_injected_energy="
               << Diagnostics.positive_floor_representative_cell_injected_energy
               << " positive_floor_correction_lambda="
               << Diagnostics.positive_floor_representative_correction_scale
               << " positive_floor_causality="
               << HistoricalMGResidualCorrectionCausalityLabel(
                      Diagnostics.positive_floor_representative_causality)
               << " post_floor_true_residual_evaluated="
               << Diagnostics.post_floor_true_residual_evaluated
               << " post_floor_true_residual_finite="
               << Diagnostics.post_floor_true_residual_finite
               << " post_floor_true_residual_error="
               << Diagnostics.post_floor_true_residual_error
               << " signed_group_extent="
               << Diagnostics.failure_signed_group_extent
               << " negative_extent=" << Diagnostics.failure_negative_extent
               << " positive_extent=" << Diagnostics.failure_positive_extent
               << " relative_deficit=" << Diagnostics.failure_relative_deficit
               << " global_E_max="
               << Diagnostics.failure_global_maximum_cell_extent
               << " minimum_correction_scale=" << Diagnostics.minimum_scale
               << " limiting_cell_id=" << Diagnostics.limiting_cell_id
               << " limiting_group=" << Diagnostics.limiting_group
               << " limiting_before_extent="
               << Diagnostics.limiting_before_extent
               << " limiting_unscaled_after_extent="
               << Diagnostics.limiting_unscaled_after_extent
               << " limiting_applied_after_extent="
               << Diagnostics.limiting_applied_after_extent;
    }

    inline bool ShouldDeferComptonForResidualCorrectionCausality(
        bool const PreCorrectionNonphysical,
        bool const PostCorrectionNonphysical)
    {
        static_cast<void>(PreCorrectionNonphysical);
        return PostCorrectionNonphysical;
    }

    inline bool HistoricalMGNegativeValueRequiresComptonFallback(
        double const Value, double const GlobalMaximumAbsoluteEg)
    {
        if(!std::isfinite(GlobalMaximumAbsoluteEg) ||
           GlobalMaximumAbsoluteEg < 0)
            return false;
        double const Threshold =
            historical_mg_positivity_continuation_trigger_fraction *
            GlobalMaximumAbsoluteEg;
        return std::isfinite(Value) && Value < 0 && -Value > Threshold;
    }

    template<class GroupContainer>
    inline bool HistoricalMGCellRequiresComptonFallback(
        GroupContainer const& Groups,
        double const GlobalMaximumAbsoluteEg)
    {
        for(double const Value : Groups)
            if(HistoricalMGNegativeValueRequiresComptonFallback(
                   Value, GlobalMaximumAbsoluteEg))
                return true;
        return false;
    }

    template<class GroupContainer, class ScaleContainer>
    inline HistoricalMGCorrectionSpectralFailure
    ClassifyHistoricalMGResidualCorrection(
        GroupContainer const& PreCorrectionGroups,
        GroupContainer const& PostCorrectionGroups,
        ScaleContainer const& CorrectionScales,
        double const CellVolume,
        double const GlobalMaximumAbsoluteEg,
        double const PreCorrectionGlobalMaximumCellExtent,
        double const PostCorrectionGlobalMaximumCellExtent,
        std::size_t const CellId)
    {
        HistoricalMGCorrectionSpectralFailure Result;
        if(PreCorrectionGroups.size() != PostCorrectionGroups.size() ||
           PostCorrectionGroups.size() != CorrectionScales.size() ||
           PostCorrectionGroups.empty() || !std::isfinite(CellVolume) ||
           CellVolume <= 0 || !std::isfinite(GlobalMaximumAbsoluteEg) ||
           GlobalMaximumAbsoluteEg < 0)
            return Result;

        GroupContainer PreRepairGroups = PreCorrectionGroups;
        GroupContainer PostRepairGroups = PostCorrectionGroups;
        double PreCorrectionTotalExtent = 0;
        double PostCorrectionTotalExtent = 0;
        long double PositiveExtent = 0;
        long double NegativeExtent = 0;
        double MostNegativeExtent = 0;
        std::size_t MostNegativeGroup = max_size_t;
        for(std::size_t Group = 0; Group < PostCorrectionGroups.size(); ++Group) {
            PreCorrectionTotalExtent += PreCorrectionGroups[Group];
            PostCorrectionTotalExtent += PostCorrectionGroups[Group];
            if(PostCorrectionGroups[Group] < 0) {
                NegativeExtent -= static_cast<long double>(
                    PostCorrectionGroups[Group]);
                if(MostNegativeGroup == max_size_t ||
                   PostCorrectionGroups[Group] < MostNegativeExtent) {
                    MostNegativeGroup = Group;
                    MostNegativeExtent = PostCorrectionGroups[Group];
                }
            }
            else
                PositiveExtent += static_cast<long double>(
                    PostCorrectionGroups[Group]);
        }
        auto const PreRepair =
            RadiationPositivity::RepairControlledNegativeGroupExtents(
                PreRepairGroups, PreCorrectionTotalExtent,
                RadiationPositivity::spectral_repair_relative_limit,
                PreCorrectionGlobalMaximumCellExtent);
        auto const PostRepair =
            RadiationPositivity::RepairControlledNegativeGroupExtents(
                PostRepairGroups, PostCorrectionTotalExtent,
                RadiationPositivity::spectral_repair_relative_limit,
                PostCorrectionGlobalMaximumCellExtent);
        Result.CausedRejection = PreRepair.repair.valid &&
            !PostRepair.repair.valid && NegativeExtent > 0;
        if(!Result.CausedRejection)
            return Result;

        Result.Failure = PostRepair.repair.failure;
        Result.CellId = CellId;
        Result.Group = MostNegativeGroup;
        Result.SignedGroupExtent = MostNegativeExtent;
        Result.NegativeExtent = static_cast<double>(NegativeExtent);
        Result.PositiveExtent = static_cast<double>(PositiveExtent);
        Result.RelativeDeficit = PositiveExtent > 0 ?
            static_cast<double>(NegativeExtent / PositiveExtent) :
            std::numeric_limits<double>::max();
        Result.GlobalMaximumCellExtent =
            PostCorrectionGlobalMaximumCellExtent;
        if(std::isfinite(PostCorrectionGlobalMaximumCellExtent) &&
           PostCorrectionGlobalMaximumCellExtent > 0)
            Result.NegativeExtentToGlobalMaximumRatio =
                Result.NegativeExtent /
                PostCorrectionGlobalMaximumCellExtent;
        Result.PreCorrectionEg =
            PreCorrectionGroups[MostNegativeGroup] / CellVolume;
        Result.PostCorrectionEg =
            PostCorrectionGroups[MostNegativeGroup] / CellVolume;
        Result.ResidualCorrection =
            Result.PostCorrectionEg - Result.PreCorrectionEg;
        Result.GlobalMaximumAbsoluteEg = GlobalMaximumAbsoluteEg;
        if(GlobalMaximumAbsoluteEg > 0)
            Result.NegativeToGlobalMaximumAbsoluteRatio =
                std::abs(Result.PostCorrectionEg) /
                GlobalMaximumAbsoluteEg;
        Result.CorrectionScale = CorrectionScales[MostNegativeGroup];
        Result.Causality =
            ClassifyHistoricalMGResidualCorrectionCausality(
                Result.PreCorrectionEg, Result.PostCorrectionEg);
#ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Comm_rank(MPI_COMM_WORLD, &Result.Rank),
            "MPI_Comm_rank(residual correction classification)");
#else
        Result.Rank = 0;
#endif
        return Result;
    }

    template<class GroupContainer>
    inline HistoricalMGCorrectionSpectralFailure
    ClassifyHistoricalMGResidualCorrection(
        GroupContainer const& PreCorrectionGroups,
        GroupContainer const& PostCorrectionGroups,
        double const PreCorrectionGlobalMaximumCellExtent,
        double const PostCorrectionGlobalMaximumCellExtent,
        std::size_t const CellId)
    {
        std::vector<double> CorrectionScales(PostCorrectionGroups.size(), 1);
        double GlobalMaximumAbsoluteEg = 0;
        for(auto const Value : PostCorrectionGroups)
            GlobalMaximumAbsoluteEg = std::max(
                GlobalMaximumAbsoluteEg, std::abs(Value));
        return ClassifyHistoricalMGResidualCorrection(
            PreCorrectionGroups, PostCorrectionGroups, CorrectionScales, 1,
            GlobalMaximumAbsoluteEg,
            PreCorrectionGlobalMaximumCellExtent,
            PostCorrectionGlobalMaximumCellExtent, CellId);
    }

    inline double HistoricalMGCorrectionSpectralFailureSeverity(
        HistoricalMGCorrectionSpectralFailure const& Failure)
    {
        if(!Failure.CausedRejection)
            return -1;
        return std::isfinite(Failure.RelativeDeficit) ?
            Failure.RelativeDeficit : std::numeric_limits<double>::max();
    }

    inline void KeepMoreSevereHistoricalMGCorrectionSpectralFailure(
        HistoricalMGCorrectionSpectralFailure const& Candidate,
        HistoricalMGCorrectionSpectralFailure& Representative)
    {
        if(HistoricalMGCorrectionSpectralFailureSeverity(Candidate) >
           HistoricalMGCorrectionSpectralFailureSeverity(Representative))
            Representative = Candidate;
    }

#ifdef RICH_MPI
    inline void CollectHistoricalMGCorrectionSpectralFailure(
        HistoricalMGCorrectionSpectralFailure& Failure)
    {
        int Rank = 0;
        RequireCGMpiSuccess(
            MPI_Comm_rank(MPI_COMM_WORLD, &Rank),
            "MPI_Comm_rank(residual correction spectral failure)");
        struct DoubleRank
        {
            double Value;
            int Rank;
        } LocalPick{HistoricalMGCorrectionSpectralFailureSeverity(Failure),
                    Rank}, GlobalPick{-1, 0};
        RequireCGMpiSuccess(
            MPI_Allreduce(&LocalPick, &GlobalPick, 1, MPI_DOUBLE_INT,
                          MPI_MAXLOC, MPI_COMM_WORLD),
            "MPI_Allreduce(residual correction spectral failure)");
        if(GlobalPick.Value < 0) {
            Failure = HistoricalMGCorrectionSpectralFailure{};
            return;
        }

        unsigned long long Identity[2] = {0, 0};
        int Classes[2] = {0, 0};
        double Details[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
        if(Rank == GlobalPick.Rank) {
            Identity[0] = static_cast<unsigned long long>(Failure.CellId);
            Identity[1] = static_cast<unsigned long long>(Failure.Group);
            Classes[0] = static_cast<int>(Failure.Failure);
            Classes[1] = static_cast<int>(Failure.Causality);
            Details[0] = Failure.SignedGroupExtent;
            Details[1] = Failure.NegativeExtent;
            Details[2] = Failure.PositiveExtent;
            Details[3] = Failure.RelativeDeficit;
            Details[4] = Failure.GlobalMaximumCellExtent;
            Details[5] = Failure.NegativeExtentToGlobalMaximumRatio;
            Details[6] = Failure.PreCorrectionEg;
            Details[7] = Failure.ResidualCorrection;
            Details[8] = Failure.PostCorrectionEg;
            Details[9] = Failure.GlobalMaximumAbsoluteEg;
            Details[10] = Failure.NegativeToGlobalMaximumAbsoluteRatio;
            Details[11] = Failure.CorrectionScale;
        }
        RequireCGMpiSuccess(
            MPI_Bcast(Identity, 2, MPI_UNSIGNED_LONG_LONG, GlobalPick.Rank,
                      MPI_COMM_WORLD),
            "MPI_Bcast(residual correction failure identity)");
        RequireCGMpiSuccess(
            MPI_Bcast(Classes, 2, MPI_INT, GlobalPick.Rank, MPI_COMM_WORLD),
            "MPI_Bcast(residual correction failure classes)");
        RequireCGMpiSuccess(
            MPI_Bcast(Details, 12, MPI_DOUBLE, GlobalPick.Rank,
                      MPI_COMM_WORLD),
            "MPI_Bcast(residual correction failure details)");
        Failure.CausedRejection = true;
        Failure.Rank = GlobalPick.Rank;
        Failure.CellId = static_cast<std::size_t>(Identity[0]);
        Failure.Group = static_cast<std::size_t>(Identity[1]);
        Failure.Failure = static_cast<
            RadiationPositivity::SpectralRepairFailure>(Classes[0]);
        Failure.Causality = static_cast<
            HistoricalMGResidualCorrectionCausality>(Classes[1]);
        Failure.SignedGroupExtent = Details[0];
        Failure.NegativeExtent = Details[1];
        Failure.PositiveExtent = Details[2];
        Failure.RelativeDeficit = Details[3];
        Failure.GlobalMaximumCellExtent = Details[4];
        Failure.NegativeExtentToGlobalMaximumRatio = Details[5];
        Failure.PreCorrectionEg = Details[6];
        Failure.ResidualCorrection = Details[7];
        Failure.PostCorrectionEg = Details[8];
        Failure.GlobalMaximumAbsoluteEg = Details[9];
        Failure.NegativeToGlobalMaximumAbsoluteRatio = Details[10];
        Failure.CorrectionScale = Details[11];
    }
#endif

    inline HistoricalMGCorrectionDisposition
    ClassifyHistoricalMGCorrection(
        HistoricalMGCorrectionAssessment const& assessment)
    {
        if(!assessment.finite)
            return HistoricalMGCorrectionDisposition::RejectNonFinite;
        return HistoricalMGCorrectionDisposition::Commit;
    }

    char const* HistoricalMGBranchLabel(HistoricalMGBranch branch);

    HistoricalMGMetrics MeasureHistoricalMG(
        std::vector<double> const& physical_solution,
        std::vector<double> const& previous_physical_solution,
        std::vector<double> const& physical_residual,
        std::vector<double> const& physical_rhs,
        std::vector<double> const& diagonal,
        std::size_t runtime_group_count,
        double* reduction_seconds = nullptr);

    HistoricalMGDecision ClassifyHistoricalMG(
        HistoricalMGMetrics const& metrics,
        std::size_t zero_based_iteration,
        double squared_tolerance,
        HistoricalMGBreakdown breakdown = HistoricalMGBreakdown::None,
        bool coefficients_finite = true);

    // Use only when an accepted iterate is immediately followed by the true
    // residual correction and ClassifyHistoricalMGCorrection.  That corrected
    // candidate, rather than the provisional iterate, owns the positivity
    // decision.
    HistoricalMGDecision ClassifyHistoricalMGCorrectionCandidate(
        HistoricalMGMetrics const& metrics,
        std::size_t zero_based_iteration,
        double squared_tolerance);

    /*! \brief Exact-order fixed-16 cell-block/spatial-stencil matrix.
     *
     * LocalBlockValues stores each 16-entry same-cell row in the matrix
     * builder's encounter order: the diagonal first, followed by the other
     * groups in ascending order.  NeighborCells and NeighborValues preserve
     * the spatial-face order, with 16 group coefficients per neighbor.
     */
    struct Fixed16BlockStencilMatrix
    {
        static std::size_t constexpr BlockSize = 16;

        std::size_t LocalCellCount = 0;
        std::size_t VectorCellCount = 0;
        std::vector<double> LocalBlockValues;
        std::vector<std::size_t> NeighborOffsets;
        std::vector<matrix_index_t> NeighborCells;
        std::vector<double> NeighborValues;

        bool HasAllocatedStorage() const
        {
            return LocalBlockValues.capacity() != 0 ||
                NeighborOffsets.capacity() != 0 ||
                NeighborCells.capacity() != 0 ||
                NeighborValues.capacity() != 0;
        }

        /*! \brief Mark storage pending while preserving sizes for overwrite.
         *
         * The native builder resizes and overwrites every entry before this
         * object may be validated.  Keeping the old vector sizes deliberately
         * avoids value-initializing the full candidate on steady topology.
         */
        void PrepareForOverwrite() noexcept
        {
            LocalCellCount = 0;
            VectorCellCount = 0;
        }

        void Release()
        {
            *this = Fixed16BlockStencilMatrix{};
        }
    };

    enum class Fixed16BlockStencilFallback : unsigned int
    {
        None,
        BuilderUnsupported,
        BuilderIneligible,
        GroupCount,
        DirectCSRUnavailable,
        LocalCellCount,
        VectorCellCount,
        LocalBlockValues,
        NeighborOffsets,
        NeighborCells,
        NeighborValues,
        NonFiniteValue,
        InputSize,
        CSRRowOffsets,
        CSRBlockColumnOrder,
        CSRNeighborOrder,
        CSRValueMismatch
    };

    char const* Fixed16BlockStencilFallbackLabel(
        Fixed16BlockStencilFallback fallback);

    Fixed16BlockStencilFallback ValidateFixed16BlockStencil(
        Fixed16BlockStencilMatrix const& matrix);

    Fixed16BlockStencilFallback BuildFixed16BlockStencilFromCSR(
        std::vector<std::size_t> const& row_offsets,
        std::vector<matrix_index_t> const& columns,
        std::vector<double> const& values,
        std::size_t vector_row_count,
        Fixed16BlockStencilMatrix& matrix);

    Fixed16BlockStencilFallback ValidateFixed16BlockStencilCSRShadow(
        Fixed16BlockStencilMatrix const& matrix,
        std::vector<std::size_t> const& row_offsets,
        std::vector<matrix_index_t> const& columns,
        std::vector<double> const& values);

    void mat_times_vec_fixed16_block_stencil(
        Fixed16BlockStencilMatrix const& matrix,
        std::vector<double> const& input,
        std::vector<double>& output,
        bool vectorize_neighbors = false);

    /*! \brief Runtime-sized, rank-local preconditioner for cell-major systems.
     *
     * Each consecutive block_size rows is one cell.  CellBlockJacobi extracts
     * and factors the complete same-cell block, including duplicate sparse
     * entries.  A cell block that cannot be factored safely uses its scalar
     * Jacobi entries instead.
     * The object owns only one candidate's factors; callers should call
     * Release after the solve.
     */
    class CellBlockJacobiPreconditioner
    {
    public:
        bool Setup(mat const& matrix,
                   size_t_mat const& columns,
                   size_t block_size,
                   PreconditionerKind kind,
                   std::vector<double> const& scalar_inverse_diagonal);

        bool SetupCSR(std::vector<size_t> const& row_offsets,
                      std::vector<matrix_index_t> const& columns,
                      std::vector<double> const& values,
                      size_t block_size,
                      PreconditionerKind kind,
                      std::vector<double> const& scalar_inverse_diagonal);

        bool SetupCSR(std::vector<size_t> const& row_offsets,
                      std::vector<std::uint32_t> const& columns,
                      std::vector<double> const& values,
                      size_t block_size,
                      PreconditionerKind kind,
                      std::vector<double> const& scalar_inverse_diagonal);

        bool SetupFixed16BlockStencil(
            Fixed16BlockStencilMatrix const& matrix,
            PreconditionerKind kind,
            std::vector<double> const& scalar_inverse_diagonal);

        void Apply(std::vector<double> const& input,
                   std::vector<double>& output);

        void ApplyCSRForwardGaussSeidel(
            std::vector<double> const& input,
            std::vector<double>& output,
            std::vector<size_t> const& row_offsets,
            std::vector<matrix_index_t> const& columns,
            std::vector<double> const& values);

        void ApplyCSRRankLocalILU0(
            std::vector<double> const& input,
            std::vector<double>& output,
            std::vector<size_t> const& row_offsets,
            std::vector<matrix_index_t> const& columns);

        void Release();

        PreconditionerKind Kind() const { return kind_; }
        PreconditionerKind RequestedKind() const { return requested_kind_; }
        bool RequestedKindSupported() const
        {
            return requested_kind_supported_;
        }
        void DowngradeToCellBlockJacobi();
        size_t BlockSize() const { return block_size_; }
        size_t BlockCount() const { return block_count_; }
        size_t FactorizedBlockCount() const { return factorized_block_count_; }
        size_t FallbackBlockCount() const { return fallback_block_count_; }
        size_t FirstFallbackBlock() const { return first_fallback_block_; }
        size_t FirstFallbackGroup() const { return first_fallback_group_; }
        CellBlockFallbackReason FirstFallbackReason() const
        {
            return first_fallback_reason_;
        }
        double MinimumNormalizedPivot() const
        {
            return minimum_normalized_pivot_;
        }
        double SetupSeconds() const { return setup_seconds_; }
        size_t ApplyCalls() const { return apply_calls_; }
        double ApplySeconds() const { return apply_seconds_; }
        size_t LocalLowerCouplingCount() const
        {
            return local_lower_coupling_count_;
        }
        size_t IgnoredLocalUpperCouplingCount() const
        {
            return ignored_local_upper_coupling_count_;
        }
        size_t IgnoredRemoteCouplingCount() const
        {
            return ignored_remote_coupling_count_;
        }
        size_t StorageBytes() const;

    private:
        bool SetupImpl(
            mat const* matrix,
            size_t_mat const* columns,
            std::vector<size_t> const* row_offsets,
            std::vector<matrix_index_t> const* csr_columns,
            std::vector<std::uint32_t> const* csr_columns32,
            std::vector<double> const* csr_values,
            Fixed16BlockStencilMatrix const* fixed16_block_stencil,
            size_t row_count,
            size_t block_size,
            PreconditionerKind kind,
            std::vector<double> const& scalar_inverse_diagonal);
        bool SetupRankLocalILU0(
            std::vector<size_t> const& row_offsets,
            std::vector<matrix_index_t> const& columns,
            std::vector<double> const& values);
        void RecordFallback(size_t block, size_t group,
                            CellBlockFallbackReason reason);
        void SolveBlock(size_t block, std::vector<double>& work) const;
        PreconditionerKind kind_ = PreconditionerKind::ScalarJacobi;
        PreconditionerKind requested_kind_ = PreconditionerKind::ScalarJacobi;
        bool requested_kind_supported_ = true;
        size_t unknown_count_ = 0;
        size_t block_size_ = 0;
        size_t block_count_ = 0;
        size_t factorized_block_count_ = 0;
        size_t fallback_block_count_ = 0;
        size_t first_fallback_block_ = max_size_t;
        size_t first_fallback_group_ = max_size_t;
        CellBlockFallbackReason first_fallback_reason_ =
            CellBlockFallbackReason::None;
        double minimum_normalized_pivot_ =
            std::numeric_limits<double>::quiet_NaN();
        double setup_seconds_ = 0;
        size_t apply_calls_ = 0;
        double apply_seconds_ = 0;
        size_t local_lower_coupling_count_ = 0;
        size_t ignored_local_upper_coupling_count_ = 0;
        size_t ignored_remote_coupling_count_ = 0;
        std::vector<double> scalar_inverse_diagonal_;
        std::vector<double> lu_factors_;
        std::vector<double> inverse_row_scales_;
        std::vector<size_t> pivots_;
        std::vector<std::uint8_t> factorized_blocks_;
        std::vector<double> apply_scratch_;
        std::vector<double> ilu0_factors_;
        std::vector<size_t> ilu0_diagonal_positions_;
    };

    /*! \brief Add one fixed neighbor-aware block-Jacobi correction sweep.
     *
     * Callers provide preconditioned and matrix_product = A preconditioned.
     * This routine adds 0.5 B^-1(input - matrix_product).  Repeating it a
     * fixed number of times remains a linear right preconditioner and includes
     * spatial-neighbor terms through A without storing overlapping blocks.
     */
    void ApplyCellBlockJacobiCorrectionSweep(
        CellBlockJacobiPreconditioner& preconditioner,
        std::vector<double> const& input,
        std::vector<double> const& matrix_product,
        std::vector<double>& preconditioned,
        std::vector<double>& residual_work);

    //! Compatibility wrapper for the original two-sweep focused test/API.
    void ApplyCellBlockJacobiSecondSweep(
        CellBlockJacobiPreconditioner& preconditioner,
        std::vector<double> const& input,
        std::vector<double> const& first_matrix_product,
        std::vector<double>& first_preconditioned,
        std::vector<double>& residual_work);

    struct BiCGSTABWorkspace
    {
        mat A;
        size_t_mat A_indeces;
        std::vector<double> A_diag;
        std::vector<double> b;
        std::vector<double> sub_x;
        std::vector<size_t> A_row_ptr;
        std::vector<size_t> A_col_idx;
        std::vector<double> A_values;
        Fixed16BlockStencilMatrix fixed16_block_stencil;
        std::vector<double> M;
        std::vector<double> r_old;
        std::vector<double> sub_a_times_p;
        std::vector<double> sub_r;
        std::vector<double> sub_p;
        std::vector<double> sub_r0;
        std::vector<double> y;
        std::vector<double> z;
        std::vector<double> v;
        std::vector<double> h;
        std::vector<double> s;
        std::vector<double> t;
        std::vector<double> scratch_rescale1;
        std::vector<double> scratch_rescale2;
        std::vector<double> old_x;
        HistoricalMGResidualCorrectionDiagnostics historical_correction;

        bool HasAllocatedStorage() const
        {
            return A.capacity() != 0 || A_indeces.capacity() != 0 ||
                A_diag.capacity() != 0 || b.capacity() != 0 ||
                sub_x.capacity() != 0 || A_row_ptr.capacity() != 0 ||
                A_col_idx.capacity() != 0 || A_values.capacity() != 0 ||
                fixed16_block_stencil.HasAllocatedStorage() ||
                M.capacity() != 0 || r_old.capacity() != 0 ||
                sub_a_times_p.capacity() != 0 || sub_r.capacity() != 0 ||
                sub_p.capacity() != 0 || sub_r0.capacity() != 0 ||
                y.capacity() != 0 || z.capacity() != 0 ||
                v.capacity() != 0 || h.capacity() != 0 ||
                s.capacity() != 0 || t.capacity() != 0 ||
                scratch_rescale1.capacity() != 0 ||
                scratch_rescale2.capacity() != 0 ||
                old_x.capacity() != 0;
        }

        void Release()
        {
            *this = BiCGSTABWorkspace{};
        }
    };

    //! \brief Class that build the data for the solution of the linear system A*x=b
    class MatrixBuilder
    {
         public:
            MatrixBuilder(std::vector<std::string> const zero_cells = std::vector<std::string> ()) : zero_cells_(zero_cells){}
        /*!
            \brief Builds the initial conditions for the CG method to solve A*x=b
            \param tess The tessellation
            \param A The A matrix to build
            \param A_indeces The indeces of the values in A, this is needed since A is sparse
            \param cells The computational cells
            \param dt The time step
            \param b The b vector to calculate
            \param x0 The initial solution guess
            \param current_time The time
        */

        virtual ~MatrixBuilder() = default;

        /**
         * @brief Virtual function to get the length scale used in the CG method.
         * 
         * This function returns the length scale used in the CG method. The default implementation returns 1.0.
         * Derived classes may override this function to provide their own length scale.
         * 
         * @return A double representing the length scale.
         */
        virtual double GetLengthScale() const {return 1.0;}

        /*! \brief Number of implicit unknowns contributed by each mesh cell.
         *
         * Grey diffusion contributes one unknown even when RICH was compiled
         * with multiple radiation energy groups.  Multigroup builders
         * override this with their runtime group count.
         */
        virtual size_t GetUnknownsPerCell() const {return 1;}

        /*! \brief Preconditioner used for Krylov direction solves.
         *
         * Scalar Jacobi remains the default so existing matrix builders retain
         * their current numerical path unless they explicitly opt in.
         */
        virtual PreconditionerKind GetPreconditionerKind() const
        {
            return PreconditionerKind::ScalarJacobi;
        }

        virtual bool HistoricalMGComptonFallbackAvailable(
            std::size_t const) const
        {
            return false;
        }

        virtual void BuildMatrix(Tessellation3D const& tess, mat& A, size_t_mat& A_indeces, std::vector<ComputationalCell3D> const& cells,
            double const dt, std::vector<double>& b, std::vector<double>& x0, double const current_time) const = 0;

        /*! \brief Whether this builder can assemble directly into flat CSR. */
        virtual bool SupportsDirectCSR() const noexcept {return false;}

        /*! \brief Build the matrix directly in flat CSR form.
         *
         * Builders must place the diagonal first in every row.  The default
         * implementation is deliberately unavailable so existing builders
         * continue through BuildMatrix unless they explicitly opt in.
         */
        virtual void BuildMatrixCSR(
            Tessellation3D const&,
            std::vector<size_t>&,
            std::vector<size_t>&,
            std::vector<double>&,
            std::vector<ComputationalCell3D> const&,
            double const,
            std::vector<double>&,
            std::vector<double>&,
            double const) const
        {
            throw UniversalError(
                "Direct CSR matrix assembly requested from an unsupported builder");
        }

        /*! \brief Whether this builder can assemble the fixed-16 stencil. */
        virtual bool SupportsFixed16BlockStencil() const noexcept
        {
            return false;
        }

        /*! \brief Whether this mesh state may use native stencil assembly. */
        virtual bool Fixed16BlockStencilEligible(
            Tessellation3D const&) const noexcept
        {
            return true;
        }

        /*! \brief Build the fixed-16 stencil and, when requested, CSR shadow.
         *
         * A selected build may leave row_offsets, columns, and values empty.
         * A shadow build must populate both representations.  The default
         * keeps existing builders source compatible and produces CSR only.
         */
        virtual void BuildMatrixCSRFixed16BlockStencil(
            Tessellation3D const& tess,
            std::vector<size_t>& row_offsets,
            std::vector<size_t>& columns,
            std::vector<double>& values,
            std::vector<ComputationalCell3D> const& cells,
            double const dt,
            std::vector<double>& b,
            std::vector<double>& x0,
            double const current_time,
            bool const build_csr_shadow,
            Fixed16BlockStencilMatrix& fixed16_block_stencil) const
        {
            (void)build_csr_shadow;
            fixed16_block_stencil.Release();
            BuildMatrixCSR(
                tess, row_offsets, columns, values, cells, dt, b, x0,
                current_time);
        }
        /*!
        \brief This method does post processing after the CG has finished (e.g. update the thermal energy)
        \param tess The tesselation
        \param extensives The extensives
        \param dt The time step
        \param cells The primitive variables
        \param CG_result The result from the CG
        */
        virtual void PostCG(Tessellation3D const& tess, std::vector<Conserved3D>& extensives, double const dt, std::vector<ComputationalCell3D>& cells,
            std::vector<double>const& CG_result, std::vector<double> const&  full_CG_result)const = 0;

        virtual void PrintDebugData(size_t const index) const {;}
        
        std::vector<std::string> const zero_cells_;
    };

    //! The fastest implementation of conjugate gradient algorithm, using data-based parallelism only
    std::vector<double> conj_grad_solver(const double tolerance, int &total_iters,
        Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells,
        double const dt, MatrixBuilder const& matrix_builder, double const time, std::vector<double> &sub_x_solution);
/**
 * @brief Performs the BiCGSTAB (Biconjugate Gradient Stabilized) method to solve the linear system A*x=b.
 * 
 * @param tolerance The tolerance for the solution. The method stops when the relative residual norm is less than or equal to this value.
 * @param total_iters Reference to an integer that will store the total number of iterations performed by the method.
 * @param tess The 3D tessellation used to discretize the problem.
 * @param cells The computational cells associated with the tessellation.
 * @param dt The time step for the problem.
 * @param matrix_builder A reference to an object that implements the MatrixBuilder interface, used to build the A matrix and b vector.
 * @param time The current time of the simulation.
 * @param sub_x_solution Reference to a vector that will store the solution x.
 * @param good_end Reference to a boolean that will be set to true if the method successfully converged to the desired tolerance.
 * @note RICH_MG_FUSED_REDUCTIONS enables the default-off MPI reduction
 * batching path. Its setting is validated collectively before ranks enter
 * different collective schedules; solver algebra and diagnostic criteria are
 * unchanged.
 * 
 * @return A vector containing the solution x.
 */
    std::vector<double> BiCGSTAB(const double tolerance, int &total_iters,
        Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells,
        double const dt, MatrixBuilder const& matrix_builder, double const time, std::vector<double> &sub_x_solution,
        bool &good_end);

    std::vector<double> &BiCGSTAB(const double tolerance, int &total_iters,
        Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells,
        double const dt, MatrixBuilder const& matrix_builder, double const time, std::vector<double> &sub_x_solution,
        bool &good_end, BiCGSTABWorkspace &workspace);

    double mpi_dot_product(const std::vector<double> &sub_u, const std::vector<double> &sub_v);

    double mpi_dot_product2(const std::vector<double> &sub_u, const std::vector<double> &sub_v);
}

#endif
