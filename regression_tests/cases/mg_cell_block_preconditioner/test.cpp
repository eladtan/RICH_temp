#include "source/Radiation/conj_grad_solve.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace
{

void Require(bool const condition, char const* message)
{
    if(!condition)
        throw std::runtime_error(message);
}

void CheckRuntimeBlockSize(std::size_t const groups)
{
    std::size_t constexpr cells = 4;
    std::size_t const rows = cells * groups;
    CG::mat matrix(rows);
    CG::size_t_mat columns(rows);
    std::vector<std::vector<double> > blocks(
        cells, std::vector<double>(groups * groups, 0));
    std::vector<double> scalar_inverse(rows, 0);

    for(std::size_t cell = 0; cell < cells; ++cell)
        for(std::size_t row_group = 0; row_group < groups; ++row_group) {
            std::size_t const row = cell * groups + row_group;
            for(std::size_t column_group = 0;
                column_group < groups; ++column_group) {
                double const value = row_group == column_group ?
                    3.0 + 0.1 * static_cast<double>(row_group + cell) :
                    -0.12 / (1.0 + std::abs(
                        static_cast<double>(row_group) -
                        static_cast<double>(column_group)));
                blocks[cell][row_group * groups + column_group] = value;
                std::size_t const column = cell * groups + column_group;
                matrix[row].push_back(0.4 * value);
                columns[row].push_back(column);
                matrix[row].push_back(0.6 * value);
                columns[row].push_back(column);
            }
            scalar_inverse[row] = 1.0 /
                blocks[cell][row_group * groups + row_group];
            if(cell + 1 < cells) {
                matrix[row].push_back(-0.03);
                columns[row].push_back((cell + 1) * groups + row_group);
            }
        }

    std::vector<double> rhs(rows, 0);
    for(std::size_t row = 0; row < rows; ++row)
        rhs[row] = 0.25 + static_cast<double>((row * 17 + 3) % 23);

    CG::CellBlockJacobiPreconditioner block;
    Require(block.Setup(matrix, columns, groups,
                        CG::PreconditionerKind::CellBlockJacobi,
                        scalar_inverse),
            "cell-block setup failed");
    Require(block.BlockSize() == groups, "runtime block size changed");
    Require(block.BlockCount() == cells, "wrong block count");
    Require(block.FactorizedBlockCount() == cells,
            "well-conditioned block used scalar fallback");
    Require(block.FallbackBlockCount() == 0,
            "unexpected scalar fallback");

    std::vector<double> solution;
    block.Apply(rhs, solution);
    double maximum_relative_residual = 0;
    for(std::size_t cell = 0; cell < cells; ++cell)
        for(std::size_t row_group = 0; row_group < groups; ++row_group) {
            double product = 0;
            for(std::size_t column_group = 0;
                column_group < groups; ++column_group)
                product += blocks[cell][row_group * groups + column_group] *
                    solution[cell * groups + column_group];
            std::size_t const row = cell * groups + row_group;
            maximum_relative_residual = std::max(maximum_relative_residual,
                std::abs(product - rhs[row]) /
                std::max(1.0, std::abs(rhs[row])));
        }
    Require(maximum_relative_residual < 5e-13,
            "cell-block application does not solve the extracted block");

    std::vector<double> alias = rhs;
    block.Apply(alias, alias);
    for(std::size_t row = 0; row < rows; ++row)
        Require(std::abs(alias[row] - solution[row]) <
                    5e-13 * std::max(1.0, std::abs(solution[row])),
                "aliased block application changed the answer");

    CG::CellBlockJacobiPreconditioner scalar;
    Require(scalar.Setup(matrix, columns, groups,
                         CG::PreconditionerKind::ScalarJacobi,
                         scalar_inverse),
            "scalar setup failed");
    std::vector<double> scalar_solution;
    scalar.Apply(rhs, scalar_solution);
    for(std::size_t row = 0; row < rows; ++row)
        Require(scalar_solution[row] == rhs[row] * scalar_inverse[row],
                "scalar Jacobi path changed");

    block.Release();
    Require(block.StorageBytes() == 0,
            "candidate block storage survived Release");
}

void CheckNeighborCorrectedSecondSweep()
{
    CG::mat matrix(2);
    CG::size_t_mat columns(2);
    matrix[0] = {4.0, -1.0};
    columns[0] = {0, 1};
    matrix[1] = {-1.0, 3.0};
    columns[1] = {0, 1};
    std::vector<double> const inverse_diagonal{0.25, 1.0 / 3.0};

    CG::CellBlockJacobiPreconditioner preconditioner;
    Require(preconditioner.Setup(
                matrix, columns, 1,
                CG::PreconditionerKind::CellBlockJacobiTwoSweep,
                inverse_diagonal),
            "two-sweep cell-block setup failed");
    Require(std::string(CG::PreconditionerKindLabel(
                preconditioner.Kind())) ==
                "cell_block_jacobi_two_sweep",
            "two-sweep preconditioner label changed");

    std::vector<double> const rhs{1.0, 2.0};
    std::vector<double> preconditioned;
    preconditioner.Apply(rhs, preconditioned);
    Require(std::abs(preconditioned[0] - 0.25) < 1e-14 &&
            std::abs(preconditioned[1] - 2.0 / 3.0) < 1e-14,
            "first block-Jacobi sweep changed");

    std::vector<double> matrix_product(2, 0);
    for(std::size_t row = 0; row < matrix.size(); ++row)
        for(std::size_t entry = 0; entry < matrix[row].size(); ++entry)
            matrix_product[row] += matrix[row][entry] *
                preconditioned[columns[row][entry]];
    std::vector<double> residual_work;
    CG::ApplyCellBlockJacobiSecondSweep(
        preconditioner, rhs, matrix_product, preconditioned, residual_work);

    Require(std::abs(preconditioned[0] - 1.0 / 3.0) < 1e-14 &&
            std::abs(preconditioned[1] - 17.0 / 24.0) < 1e-14,
            "neighbor-aware second sweep produced the wrong correction");
    Require(preconditioner.ApplyCalls() == 2,
            "two-sweep apply did not perform exactly two block solves");
}

void CheckNeighborCorrectedFourSweep()
{
    CG::mat matrix(2);
    CG::size_t_mat columns(2);
    matrix[0] = {4.0, -1.0};
    columns[0] = {0, 1};
    matrix[1] = {-1.0, 3.0};
    columns[1] = {0, 1};
    std::vector<double> const inverse_diagonal{0.25, 1.0 / 3.0};

    CG::CellBlockJacobiPreconditioner preconditioner;
    Require(preconditioner.Setup(
                matrix, columns, 1,
                CG::PreconditionerKind::CellBlockJacobiFourSweep,
                inverse_diagonal),
            "four-sweep cell-block setup failed");
    Require(std::string(CG::PreconditionerKindLabel(
                preconditioner.Kind())) ==
                "cell_block_jacobi_four_sweep",
            "four-sweep preconditioner label changed");
    Require(CG::CellBlockJacobiSweepCount(preconditioner.Kind()) == 4,
            "four-sweep preconditioner reported the wrong sweep count");

    std::vector<double> const rhs{1.0, 2.0};
    std::vector<double> preconditioned;
    preconditioner.Apply(rhs, preconditioned);
    std::vector<double> matrix_product(2, 0);
    std::vector<double> residual_work;
    for(std::size_t sweep = 1; sweep < 4; ++sweep) {
        std::fill(matrix_product.begin(), matrix_product.end(), 0);
        for(std::size_t row = 0; row < matrix.size(); ++row)
            for(std::size_t entry = 0; entry < matrix[row].size(); ++entry)
                matrix_product[row] += matrix[row][entry] *
                    preconditioned[columns[row][entry]];
        CG::ApplyCellBlockJacobiCorrectionSweep(
            preconditioner, rhs, matrix_product, preconditioned,
            residual_work);
    }

    Require(std::abs(preconditioned[0] - 235.0 / 576.0) < 1e-14 &&
            std::abs(preconditioned[1] - 295.0 / 384.0) < 1e-14,
            "neighbor-aware four-sweep correction produced the wrong result");
    Require(preconditioner.ApplyCalls() == 4,
            "four-sweep apply did not perform exactly four block solves");
}

void CheckNeighborCorrectedEightSweep()
{
    CG::mat matrix(2);
    CG::size_t_mat columns(2);
    matrix[0] = {4.0, -1.0};
    columns[0] = {0, 1};
    matrix[1] = {-1.0, 3.0};
    columns[1] = {0, 1};
    std::vector<double> const inverse_diagonal{0.25, 1.0 / 3.0};

    CG::CellBlockJacobiPreconditioner preconditioner;
    Require(preconditioner.Setup(
                matrix, columns, 1,
                CG::PreconditionerKind::CellBlockJacobiEightSweep,
                inverse_diagonal),
            "eight-sweep cell-block setup failed");
    Require(std::string(CG::PreconditionerKindLabel(
                preconditioner.Kind())) ==
                "cell_block_jacobi_eight_sweep",
            "eight-sweep preconditioner label changed");
    Require(CG::CellBlockJacobiSweepCount(preconditioner.Kind()) == 8,
            "eight-sweep preconditioner reported the wrong sweep count");

    std::vector<double> const rhs{1.0, 2.0};
    std::vector<double> preconditioned;
    preconditioner.Apply(rhs, preconditioned);
    std::vector<double> matrix_product(2, 0);
    std::vector<double> residual_work;
    for(std::size_t sweep = 1; sweep < 8; ++sweep) {
        std::fill(matrix_product.begin(), matrix_product.end(), 0);
        for(std::size_t row = 0; row < matrix.size(); ++row)
            for(std::size_t entry = 0; entry < matrix[row].size(); ++entry)
                matrix_product[row] += matrix[row][entry] *
                    preconditioned[columns[row][entry]];
        CG::ApplyCellBlockJacobiCorrectionSweep(
            preconditioner, rhs, matrix_product, preconditioned,
            residual_work);
    }

    Require(std::abs(preconditioned[0] - 592921.0 / 1327104.0) < 1e-14 &&
            std::abs(preconditioned[1] - 2147981.0 / 2654208.0) < 1e-14,
            "neighbor-aware eight-sweep correction produced the wrong result");
    Require(preconditioner.ApplyCalls() == 8,
            "eight-sweep apply did not perform exactly eight block solves");
}

void CheckSingularFallback()
{
    CG::mat matrix(2);
    CG::size_t_mat columns(2);
    matrix[0].push_back(1);
    columns[0].push_back(0);
    matrix[0].push_back(1);
    columns[0].push_back(1);
    matrix[1].push_back(1);
    columns[1].push_back(0);
    matrix[1].push_back(1);
    columns[1].push_back(1);
    std::vector<double> const scalar_inverse(2, 1);

    CG::CellBlockJacobiPreconditioner preconditioner;
    Require(preconditioner.Setup(matrix, columns, 2,
                                 CG::PreconditionerKind::CellBlockJacobi,
                                 scalar_inverse),
            "singular block should use scalar fallback");
    Require(preconditioner.FallbackBlockCount() == 1,
            "singular block fallback was not counted");
    Require(preconditioner.FactorizedBlockCount() == 0,
            "singular block was marked factorized");
    std::vector<double> const rhs{2, 4};
    std::vector<double> solution;
    preconditioner.Apply(rhs, solution);
    Require(solution == rhs, "singular block did not use scalar Jacobi");
}

CG::HistoricalMGMetrics BaseHistoricalMetrics()
{
    CG::HistoricalMGMetrics metrics;
    metrics.weighted_residual_squared = 1;
    metrics.weighted_rhs_squared = 1;
    metrics.historical_error = 1;
    metrics.max0 = 0;
    metrics.max1 = 0;
    metrics.negative = 0;
    metrics.finite = true;
    return metrics;
}

void RequireBranch(CG::HistoricalMGDecision const& decision,
                   CG::HistoricalMGBranch const branch,
                   bool const accepted,
                   bool const rejected,
                   char const* const message)
{
    Require(decision.branch == branch && decision.accept == accepted &&
            decision.reject == rejected, message);
}

void CheckHistoricalAcceptanceBranches()
{
    double constexpr tolerance = 1e-11;
    Require(CG::historical_mg_maximum_iterations == 10000,
            "historical fixed iteration budget changed");

    CG::HistoricalMGMetrics metrics = BaseHistoricalMetrics();
    metrics.historical_error = 1e-3;
    metrics.max0 = 1e-7;
    metrics.max1 = 1e-7;
    RequireBranch(CG::ClassifyHistoricalMG(metrics, 9, tolerance),
                  CG::HistoricalMGBranch::Continue, false, false,
                  "loose branch used the wrong zero-based iteration boundary");
    RequireBranch(CG::ClassifyHistoricalMG(metrics, 10, tolerance),
                  CG::HistoricalMGBranch::LooseMaxima, true, false,
                  "loose historical branch was not retained");

    metrics = BaseHistoricalMetrics();
    metrics.historical_error = 1e-12;
    metrics.max0 = 2e-6;
    metrics.max1 = 3e-6;
    metrics.negative = 0;
    RequireBranch(CG::ClassifyHistoricalMG(metrics, 0, tolerance),
                  CG::HistoricalMGBranch::NormalTolerance, true, false,
                  "normal historical tolerance branch was not retained");
    metrics.negative = 1;
    RequireBranch(CG::ClassifyHistoricalMG(metrics, 25, tolerance),
                  CG::HistoricalMGBranch::Continue, false, false,
                  "normal branch ignored its negativity/iteration predicate");
    metrics.max1 = 1e-11;
    RequireBranch(CG::ClassifyHistoricalMG(metrics, 26, tolerance),
                  CG::HistoricalMGBranch::NormalTolerance, true, false,
                  "late normal branch did not use its strict max1 predicate");

    metrics = BaseHistoricalMetrics();
    metrics.historical_error = 1e-101;
    metrics.max0 = 1;
    metrics.max1 = 1;
    metrics.negative = 1;
    RequireBranch(CG::ClassifyHistoricalMG(metrics, 0, tolerance),
                  CG::HistoricalMGBranch::ExtremelySmallError, true, false,
                  "extremely-small-error branch was not retained");

    metrics = BaseHistoricalMetrics();
    metrics.historical_error = 1e-3;
    metrics.max0 = 1e-7;
    metrics.max1 = 1e-7;
    RequireBranch(CG::ClassifyHistoricalMG(
                      metrics, 8, tolerance,
                      CG::HistoricalMGBreakdown::TinyRho),
                  CG::HistoricalMGBranch::RejectEarlyBreakdown, false, true,
                  "breakdown was accepted before ten completed iterations");
    RequireBranch(CG::ClassifyHistoricalMG(
                      metrics, 9, tolerance,
                      CG::HistoricalMGBreakdown::TinyRho),
                  CG::HistoricalMGBranch::TinyRho, true, false,
                  "tiny-rho historical branch was not retained");
    RequireBranch(CG::ClassifyHistoricalMG(
                      metrics, 9, tolerance,
                      CG::HistoricalMGBreakdown::TinyAlphaOmega),
                  CG::HistoricalMGBranch::TinyAlphaOmega, true, false,
                  "tiny-alpha/omega historical branch was not retained");
    metrics.max0 = CG::historical_mg_loose_max0;
    RequireBranch(CG::ClassifyHistoricalMG(
                      metrics, 9, tolerance,
                      CG::HistoricalMGBreakdown::TinyRho),
                  CG::HistoricalMGBranch::RejectBreakdownMetrics,
                  false, true,
                  "breakdown accepted metrics at the strict loose boundary");

    metrics = BaseHistoricalMetrics();
    metrics.finite = false;
    RequireBranch(CG::ClassifyHistoricalMG(metrics, 100, tolerance),
                  CG::HistoricalMGBranch::RejectNonFinite, false, true,
                  "non-finite historical metrics were not rejected");
    metrics = BaseHistoricalMetrics();
    RequireBranch(CG::ClassifyHistoricalMG(
                      metrics, 100, tolerance,
                      CG::HistoricalMGBreakdown::None, false),
                  CG::HistoricalMGBranch::RejectNonFinite, false, true,
                  "non-finite solver coefficients were not rejected");
}

void CheckPhysicalRhsNormalization()
{
    std::vector<double> const physical_solution{10.000001};
    std::vector<double> const previous_solution{10};
    std::vector<double> const physical_residual{1e-3};
    std::vector<double> const physical_rhs{1e3};
    std::vector<double> const diagonal{2};
    CG::HistoricalMGMetrics const metrics = CG::MeasureHistoricalMG(
        physical_solution, previous_solution, physical_residual,
        physical_rhs, diagonal, 1);
    Require(metrics.finite,
            "physical-RHS historical metric was not finite");
    Require(std::abs(metrics.historical_error - 1e-12) < 1e-24,
            "historical metric was not normalized by the full physical RHS");

    std::vector<double> const zero_rhs{0};
    CG::HistoricalMGMetrics const zero_denominator =
        CG::MeasureHistoricalMG(
            physical_solution, previous_solution, physical_residual,
            zero_rhs, diagonal, 1);
    Require(!zero_denominator.finite,
            "zero historical RHS denominator was treated as convergence");
}

void CheckPositivityContinuationEnergyGuards()
{
    std::vector<std::size_t> const cell_ids{73};
    std::vector<double> const scales{1, 1};
    auto const cell_boundary = CG::AssessHistoricalMGCorrectedNegativity(
        std::vector<double>{2, 0}, std::vector<double>{2, -1}, scales,
        2, cell_ids, 1e7, false);
    CG::HistoricalMGPositivityContinuation cell_boundary_continuation;
    Require(!CG::HistoricalMGCorrectedNegativityExceedsSingleCellContinuationThreshold(
                cell_boundary) &&
            CG::EvaluateHistoricalMGPositivityContinuation(
                cell_boundary, 5, cell_boundary_continuation) ==
                CG::HistoricalMGPositivityContinuationDecision::NoExtension,
            "the inclusive 1e-7 single-cell boundary opened a restart");

    auto const cell_above = CG::AssessHistoricalMGCorrectedNegativity(
        std::vector<double>{2, 0}, std::vector<double>{2, -1.000001},
        scales, 2, cell_ids, 1e7, false);
    CG::HistoricalMGPositivityContinuation cell_continuation;
    Require(CG::HistoricalMGCorrectedNegativityExceedsSingleCellContinuationThreshold(
                cell_above) &&
            CG::EvaluateHistoricalMGPositivityContinuation(
                cell_above, 5, cell_continuation) ==
                CG::HistoricalMGPositivityContinuationDecision::Restart,
            "a single-cell negative above 1e-7 did not restart");

    auto global_boundary = CG::AssessHistoricalMGCorrectedNegativity(
        std::vector<double>{2, 0}, std::vector<double>{2, -0.5}, scales,
        2, cell_ids, 1e7, false);
    Require(CG::CollectHistoricalMGCorrectedNegativityGlobalExtents(
                global_boundary, 1, 1e8, false) &&
            !CG::HistoricalMGCorrectedNegativityExceedsGlobalContinuationThreshold(
                global_boundary),
            "the inclusive 1e-8 global boundary opened a restart");

    auto global_above = CG::AssessHistoricalMGCorrectedNegativity(
        std::vector<double>{2, 0}, std::vector<double>{2, -0.5}, scales,
        2, cell_ids, 1e7, false);
    Require(CG::CollectHistoricalMGCorrectedNegativityGlobalExtents(
                global_above, 2, 1e8,
#ifdef RICH_MPI
                true
#else
                false
#endif
                ) &&
            !CG::HistoricalMGCorrectedNegativityExceedsSingleCellContinuationThreshold(
                global_above) &&
            CG::HistoricalMGCorrectedNegativityExceedsGlobalContinuationThreshold(
                global_above),
            "MPI-global negative energy above 1e-8 did not restart");
}

#ifdef RICH_MPI
void CheckRankWithNoActiveRows(bool const zero_owned_case)
{
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if(ranks < 2)
        return;

    bool const empty_active_rank = rank == 0;
    std::size_t const canonical_owned =
        zero_owned_case && empty_active_rank ? 0 : 1;
    if(!zero_owned_case && empty_active_rank)
        Require(canonical_owned == 1,
                "zero-active test rank did not retain canonical ownership");

    std::vector<double> const solution = empty_active_rank
        ? std::vector<double>() : std::vector<double>{10.000001};
    std::vector<double> const previous = empty_active_rank
        ? std::vector<double>() : std::vector<double>{10};
    std::vector<double> const residual = empty_active_rank
        ? std::vector<double>() : std::vector<double>{1e-3};
    std::vector<double> const rhs = empty_active_rank
        ? std::vector<double>() : std::vector<double>{1e3};
    std::vector<double> const diagonal = empty_active_rank
        ? std::vector<double>() : std::vector<double>{2};
    CG::HistoricalMGMetrics const metrics = CG::MeasureHistoricalMG(
        solution, previous, residual, rhs, diagonal, 1);
    Require(metrics.finite &&
            std::abs(metrics.historical_error - 1e-12) < 1e-24,
            zero_owned_case
                ? "zero-owned rank corrupted the collective historical metric"
                : "nonempty zero-active rank corrupted the collective historical metric");
}
#endif

} // namespace

int main(int argc, char** argv)
{
#ifdef RICH_MPI
    MPI_Init(&argc, &argv);
#else
    (void)argc;
    (void)argv;
#endif
    int status = 0;
    try {
        for(std::size_t const groups : {1u, 2u, 3u, 7u, 16u})
            CheckRuntimeBlockSize(groups);
        CheckNeighborCorrectedSecondSweep();
        CheckNeighborCorrectedFourSweep();
        CheckNeighborCorrectedEightSweep();
        CheckSingularFallback();
        CheckHistoricalAcceptanceBranches();
        CheckPhysicalRhsNormalization();
        CheckPositivityContinuationEnergyGuards();
#ifdef RICH_MPI
        CheckRankWithNoActiveRows(true);
        CheckRankWithNoActiveRows(false);
#endif
    }
    catch(std::exception const& error) {
        std::cerr << "MG_CELL_BLOCK_PRECONDITIONER_FAIL "
                  << error.what() << std::endl;
        status = 1;
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(status == 0 && rank == 0)
#else
    if(status == 0)
#endif
        std::cout << "MG_CELL_BLOCK_PRECONDITIONER_PASS groups=1,2,3,7,16"
                  << std::endl;
#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return status;
}
