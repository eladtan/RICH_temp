#include "conj_grad_solve.hpp"
#include "RadiationMpiFailure.hpp"
#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <utility>
#include <vectorclass.h>
#include "boost/math/special_functions/pow.hpp"
#include "misc/memory_profile.hpp"

using boost::math::pow;

namespace CG
{
    static constexpr bool use_crs_matvec = true;

#ifdef RICH_MPI
    void RequireCGMpiSuccess(
        int const MpiError, char const* const Operation) noexcept
    {
        RadiationMpi::RequireSuccess(
            MpiError, "MG_CG_MPI_FATAL", Operation);
    }
#endif

    namespace
    {

        double BoundedHistoricalRatio(
            long double const numerator,
            long double const denominator)
        {
            if(numerator == 0)
                return 0;
            if(!(denominator > 0))
                return std::numeric_limits<double>::max();
            long double const ratio = numerator / denominator;
            if(!std::isfinite(ratio) ||
               ratio > std::numeric_limits<double>::max())
                return std::numeric_limits<double>::max();
            return static_cast<double>(ratio);
        }

#ifdef RICH_MPI
        void RecordReductionBatch(
            unsigned long long* const collective_calls,
            unsigned long long* const causal_rounds,
            unsigned long long const calls,
            unsigned long long const rounds)
        {
            if(collective_calls != nullptr)
                *collective_calls += calls;
            if(causal_rounds != nullptr)
                *causal_rounds += rounds;
        }
#endif

        std::size_t constexpr Fixed16BlockSize = 16;

        std::size_t Fixed16OrderedBlockColumn(
            std::size_t const row_group,
            std::size_t const slot)
        {
            return slot == 0 ? row_group :
                (slot <= row_group ? slot - 1 : slot);
        }

        enum class Fixed16BlockMatvecFallback : unsigned int
        {
            None = 0,
            GroupCount,
            RowCount,
            RowOffsets,
            ShortRow,
            BlockColumnOrder,
            NeighborCoupling
        };

        char const* Fixed16BlockMatvecFallbackLabel(
            Fixed16BlockMatvecFallback const fallback)
        {
            switch(fallback)
            {
            case Fixed16BlockMatvecFallback::None:
                return "none";
            case Fixed16BlockMatvecFallback::GroupCount:
                return "group_count";
            case Fixed16BlockMatvecFallback::RowCount:
                return "row_count";
            case Fixed16BlockMatvecFallback::RowOffsets:
                return "row_offsets";
            case Fixed16BlockMatvecFallback::ShortRow:
                return "short_row";
            case Fixed16BlockMatvecFallback::BlockColumnOrder:
                return "block_column_order";
            case Fixed16BlockMatvecFallback::NeighborCoupling:
                return "neighbor_coupling";
            }
            return "unknown";
        }

        struct Fixed16BlockMatvecSchedule
        {
            std::vector<std::size_t> const* RowOffsets = nullptr;
            std::vector<std::size_t> const* ColumnIndices = nullptr;
            std::size_t RowCount = 0;
            std::size_t CellCount = 0;
            std::size_t NeighborCouplings = 0;
            bool Ready = false;
        };

        Fixed16BlockMatvecFallback BuildFixed16BlockMatvecSchedule(
            std::vector<std::size_t> const& row_offsets,
            std::vector<std::size_t> const& column_indices,
            std::size_t const row_count,
            std::size_t const group_count,
            Fixed16BlockMatvecSchedule& schedule)
        {
            schedule = Fixed16BlockMatvecSchedule{};
            if(group_count != Fixed16BlockSize)
                return Fixed16BlockMatvecFallback::GroupCount;
            if(row_count % Fixed16BlockSize != 0)
                return Fixed16BlockMatvecFallback::RowCount;
            if(row_offsets.size() != row_count + 1 || row_offsets.empty() ||
               row_offsets.front() != 0 ||
               row_offsets.back() != column_indices.size())
                return Fixed16BlockMatvecFallback::RowOffsets;

            std::size_t neighbor_couplings = 0;
            for(std::size_t row = 0; row < row_count; ++row)
            {
                std::size_t const begin = row_offsets[row];
                std::size_t const end = row_offsets[row + 1];
                if(begin > end || end > column_indices.size())
                    return Fixed16BlockMatvecFallback::RowOffsets;
                if(end - begin < Fixed16BlockSize)
                    return Fixed16BlockMatvecFallback::ShortRow;

                std::size_t const cell = row / Fixed16BlockSize;
                std::size_t const row_group = row % Fixed16BlockSize;
                std::size_t const cell_base = cell * Fixed16BlockSize;
                for(std::size_t slot = 0; slot < Fixed16BlockSize; ++slot)
                {
                    std::size_t const expected_group =
                        Fixed16OrderedBlockColumn(row_group, slot);
                    if(column_indices[begin + slot] !=
                       cell_base + expected_group)
                        return Fixed16BlockMatvecFallback::BlockColumnOrder;
                }
                for(std::size_t entry = begin + Fixed16BlockSize;
                    entry < end; ++entry)
                {
                    std::size_t const column = column_indices[entry];
                    if(column / Fixed16BlockSize == cell ||
                       column % Fixed16BlockSize != row_group)
                        return Fixed16BlockMatvecFallback::NeighborCoupling;
                    ++neighbor_couplings;
                }
            }

            schedule.RowOffsets = &row_offsets;
            schedule.ColumnIndices = &column_indices;
            schedule.RowCount = row_count;
            schedule.CellCount = row_count / Fixed16BlockSize;
            schedule.NeighborCouplings = neighbor_couplings;
            schedule.Ready = true;
            return Fixed16BlockMatvecFallback::None;
        }

        unsigned int Fixed16BlockMatvecFallbackBit(
            Fixed16BlockMatvecFallback const fallback)
        {
            if(fallback == Fixed16BlockMatvecFallback::None)
                return 0u;
            unsigned int const offset =
                static_cast<unsigned int>(fallback) - 1u;
            return 1u << offset;
        }

        char const* Fixed16BlockMatvecFallbackMaskLabel(
            unsigned int const fallback_mask)
        {
            if(fallback_mask == 0u)
                return "none";
            if((fallback_mask & (fallback_mask - 1u)) != 0u)
                return "multiple";
            for(unsigned int value =
                    static_cast<unsigned int>(
                        Fixed16BlockMatvecFallback::GroupCount);
                value <= static_cast<unsigned int>(
                    Fixed16BlockMatvecFallback::NeighborCoupling);
                ++value)
            {
                Fixed16BlockMatvecFallback const fallback =
                    static_cast<Fixed16BlockMatvecFallback>(value);
                if((fallback_mask &
                    Fixed16BlockMatvecFallbackBit(fallback)) != 0u)
                    return Fixed16BlockMatvecFallbackLabel(fallback);
            }
            return "unknown";
        }

        unsigned int Fixed16BlockStencilFallbackBit(
            Fixed16BlockStencilFallback const fallback)
        {
            if(fallback == Fixed16BlockStencilFallback::None)
                return 0u;
            unsigned int const offset =
                static_cast<unsigned int>(fallback) - 1u;
            return 1u << offset;
        }

        char const* Fixed16BlockStencilFallbackMaskLabel(
            unsigned int const fallback_mask)
        {
            if(fallback_mask == 0u)
                return "none";
            if((fallback_mask & (fallback_mask - 1u)) != 0u)
                return "multiple";
            for(unsigned int value = static_cast<unsigned int>(
                    Fixed16BlockStencilFallback::BuilderUnsupported);
                value <= static_cast<unsigned int>(
                    Fixed16BlockStencilFallback::CSRValueMismatch); ++value)
            {
                Fixed16BlockStencilFallback const fallback =
                    static_cast<Fixed16BlockStencilFallback>(value);
                if((fallback_mask &
                    Fixed16BlockStencilFallbackBit(fallback)) != 0u)
                    return Fixed16BlockStencilFallbackLabel(fallback);
            }
            return "unknown";
        }
    }

    char const* Fixed16BlockStencilFallbackLabel(
        Fixed16BlockStencilFallback const fallback)
    {
        switch(fallback)
        {
        case Fixed16BlockStencilFallback::None:
            return "none";
        case Fixed16BlockStencilFallback::BuilderUnsupported:
            return "builder_unsupported";
        case Fixed16BlockStencilFallback::BuilderIneligible:
            return "builder_ineligible";
        case Fixed16BlockStencilFallback::GroupCount:
            return "group_count";
        case Fixed16BlockStencilFallback::DirectCSRUnavailable:
            return "direct_csr_unavailable";
        case Fixed16BlockStencilFallback::LocalCellCount:
            return "local_cell_count";
        case Fixed16BlockStencilFallback::VectorCellCount:
            return "vector_cell_count";
        case Fixed16BlockStencilFallback::LocalBlockValues:
            return "local_block_values";
        case Fixed16BlockStencilFallback::NeighborOffsets:
            return "neighbor_offsets";
        case Fixed16BlockStencilFallback::NeighborCells:
            return "neighbor_cells";
        case Fixed16BlockStencilFallback::NeighborValues:
            return "neighbor_values";
        case Fixed16BlockStencilFallback::NonFiniteValue:
            return "non_finite_value";
        case Fixed16BlockStencilFallback::InputSize:
            return "input_size";
        case Fixed16BlockStencilFallback::CSRRowOffsets:
            return "csr_row_offsets";
        case Fixed16BlockStencilFallback::CSRBlockColumnOrder:
            return "csr_block_column_order";
        case Fixed16BlockStencilFallback::CSRNeighborOrder:
            return "csr_neighbor_order";
        case Fixed16BlockStencilFallback::CSRValueMismatch:
            return "csr_value_mismatch";
        }
        return "unknown";
    }

    Fixed16BlockStencilFallback ValidateFixed16BlockStencil(
        Fixed16BlockStencilMatrix const& matrix)
    {
        std::size_t constexpr block_size =
            Fixed16BlockStencilMatrix::BlockSize;
        if(matrix.LocalCellCount > max_size_t / block_size ||
           matrix.LocalCellCount * block_size > max_size_t / block_size)
            return Fixed16BlockStencilFallback::LocalCellCount;
        if(matrix.VectorCellCount < matrix.LocalCellCount ||
           matrix.VectorCellCount > max_size_t / block_size)
            return Fixed16BlockStencilFallback::VectorCellCount;
        if(matrix.LocalBlockValues.size() !=
           matrix.LocalCellCount * block_size * block_size)
            return Fixed16BlockStencilFallback::LocalBlockValues;
        if(matrix.LocalCellCount == max_size_t ||
           matrix.NeighborOffsets.size() != matrix.LocalCellCount + 1 ||
           matrix.NeighborOffsets.empty() ||
           matrix.NeighborOffsets.front() != 0 ||
           matrix.NeighborOffsets.back() != matrix.NeighborCells.size())
            return Fixed16BlockStencilFallback::NeighborOffsets;
        for(std::size_t cell = 0; cell < matrix.LocalCellCount; ++cell)
        {
            if(matrix.NeighborOffsets[cell] >
               matrix.NeighborOffsets[cell + 1])
                return Fixed16BlockStencilFallback::NeighborOffsets;
            for(std::size_t entry = matrix.NeighborOffsets[cell];
                entry < matrix.NeighborOffsets[cell + 1]; ++entry)
                if(matrix.NeighborCells[entry] >= matrix.VectorCellCount ||
                   matrix.NeighborCells[entry] == cell)
                    return Fixed16BlockStencilFallback::NeighborCells;
        }
        if(matrix.NeighborCells.size() > max_size_t / block_size ||
           matrix.NeighborValues.size() !=
               matrix.NeighborCells.size() * block_size)
            return Fixed16BlockStencilFallback::NeighborValues;
        for(double const value : matrix.LocalBlockValues)
            if(!std::isfinite(value))
                return Fixed16BlockStencilFallback::NonFiniteValue;
        for(double const value : matrix.NeighborValues)
            if(!std::isfinite(value))
                return Fixed16BlockStencilFallback::NonFiniteValue;
        return Fixed16BlockStencilFallback::None;
    }

    Fixed16BlockStencilFallback BuildFixed16BlockStencilFromCSR(
        std::vector<std::size_t> const& row_offsets,
        std::vector<matrix_index_t> const& columns,
        std::vector<double> const& values,
        std::size_t const vector_row_count,
        Fixed16BlockStencilMatrix& matrix)
    {
        matrix.Release();
        std::size_t constexpr block_size =
            Fixed16BlockStencilMatrix::BlockSize;
        if(vector_row_count % block_size != 0)
            return Fixed16BlockStencilFallback::VectorCellCount;
        if(row_offsets.empty() || row_offsets.front() != 0 ||
           row_offsets.back() != columns.size() ||
           columns.size() != values.size())
            return Fixed16BlockStencilFallback::CSRRowOffsets;
        std::size_t const row_count = row_offsets.size() - 1;
        if(row_count % block_size != 0)
            return Fixed16BlockStencilFallback::LocalCellCount;
        if(row_count > max_size_t / block_size)
            return Fixed16BlockStencilFallback::LocalBlockValues;
        if(vector_row_count < row_count)
            return Fixed16BlockStencilFallback::VectorCellCount;

        Fixed16BlockStencilMatrix candidate;
        candidate.LocalCellCount = row_count / block_size;
        candidate.VectorCellCount = vector_row_count / block_size;
        candidate.LocalBlockValues.resize(row_count * block_size);
        candidate.NeighborOffsets.reserve(candidate.LocalCellCount + 1);
        candidate.NeighborOffsets.push_back(0);

        for(std::size_t cell = 0; cell < candidate.LocalCellCount; ++cell)
        {
            std::size_t const first_row = cell * block_size;
            std::size_t const first_begin = row_offsets[first_row];
            std::size_t const first_end = row_offsets[first_row + 1];
            if(first_begin > first_end || first_end > columns.size() ||
               first_end - first_begin < block_size)
                return Fixed16BlockStencilFallback::CSRRowOffsets;
            std::size_t const neighbor_count =
                first_end - first_begin - block_size;
            if(candidate.NeighborCells.size() >
               max_size_t - neighbor_count)
                return Fixed16BlockStencilFallback::NeighborCells;

            for(std::size_t neighbor = 0;
                neighbor < neighbor_count; ++neighbor)
            {
                std::size_t const column =
                    columns[first_begin + block_size + neighbor];
                if(column >= vector_row_count || column % block_size != 0 ||
                   column / block_size == cell)
                    return Fixed16BlockStencilFallback::CSRNeighborOrder;
                candidate.NeighborCells.push_back(column / block_size);
            }
            candidate.NeighborOffsets.push_back(
                candidate.NeighborCells.size());

            for(std::size_t row_group = 0;
                row_group < block_size; ++row_group)
            {
                std::size_t const row = first_row + row_group;
                std::size_t const begin = row_offsets[row];
                std::size_t const end = row_offsets[row + 1];
                if(begin > end || end > columns.size() ||
                   end - begin != block_size + neighbor_count)
                    return Fixed16BlockStencilFallback::CSRNeighborOrder;
                for(std::size_t slot = 0; slot < block_size; ++slot)
                {
                    std::size_t const expected_column = first_row +
                        Fixed16OrderedBlockColumn(row_group, slot);
                    if(columns[begin + slot] != expected_column)
                        return Fixed16BlockStencilFallback::
                            CSRBlockColumnOrder;
                    candidate.LocalBlockValues[
                        row * block_size + slot] = values[begin + slot];
                }
                for(std::size_t neighbor = 0;
                    neighbor < neighbor_count; ++neighbor)
                {
                    std::size_t const entry =
                        begin + block_size + neighbor;
                    std::size_t const expected_column =
                        candidate.NeighborCells[
                            candidate.NeighborOffsets[cell] + neighbor] *
                            block_size + row_group;
                    if(columns[entry] != expected_column)
                        return Fixed16BlockStencilFallback::CSRNeighborOrder;
                    candidate.NeighborValues.push_back(values[entry]);
                }
            }
        }

        // Values were appended group-major above.  Transpose each cell's
        // spatial tail to the neighbor-major layout used by the hot kernel.
        if(candidate.NeighborCells.size() > max_size_t / block_size)
            return Fixed16BlockStencilFallback::NeighborValues;
        std::vector<double> neighbor_major(candidate.NeighborCells.size() *
                                           block_size);
        std::size_t group_major_offset = 0;
        for(std::size_t cell = 0; cell < candidate.LocalCellCount; ++cell)
        {
            std::size_t const begin = candidate.NeighborOffsets[cell];
            std::size_t const count =
                candidate.NeighborOffsets[cell + 1] - begin;
            for(std::size_t group = 0; group < block_size; ++group)
                for(std::size_t neighbor = 0; neighbor < count; ++neighbor)
                    neighbor_major[(begin + neighbor) * block_size + group] =
                        candidate.NeighborValues[
                            group_major_offset + group * count + neighbor];
            group_major_offset += count * block_size;
        }
        candidate.NeighborValues.swap(neighbor_major);

        Fixed16BlockStencilFallback const fallback =
            ValidateFixed16BlockStencil(candidate);
        if(fallback != Fixed16BlockStencilFallback::None)
            return fallback;
        matrix = std::move(candidate);
        return Fixed16BlockStencilFallback::None;
    }

    Fixed16BlockStencilFallback ValidateFixed16BlockStencilCSRShadow(
        Fixed16BlockStencilMatrix const& matrix,
        std::vector<std::size_t> const& row_offsets,
        std::vector<matrix_index_t> const& columns,
        std::vector<double> const& values)
    {
        Fixed16BlockStencilFallback const structural_fallback =
            ValidateFixed16BlockStencil(matrix);
        if(structural_fallback != Fixed16BlockStencilFallback::None)
            return structural_fallback;
        std::size_t constexpr block_size =
            Fixed16BlockStencilMatrix::BlockSize;
        std::size_t const row_count = matrix.LocalCellCount * block_size;
        if(row_offsets.size() != row_count + 1 || row_offsets.empty() ||
           row_offsets.front() != 0 || row_offsets.back() != columns.size() ||
           columns.size() != values.size())
            return Fixed16BlockStencilFallback::CSRRowOffsets;

        auto const same_value = [](double const left, double const right)
        {
            return left == right &&
                (left != 0.0 || std::signbit(left) == std::signbit(right));
        };
        for(std::size_t cell = 0; cell < matrix.LocalCellCount; ++cell)
        {
            std::size_t const neighbor_begin = matrix.NeighborOffsets[cell];
            std::size_t const neighbor_count =
                matrix.NeighborOffsets[cell + 1] - neighbor_begin;
            for(std::size_t row_group = 0;
                row_group < block_size; ++row_group)
            {
                std::size_t const row = cell * block_size + row_group;
                std::size_t const begin = row_offsets[row];
                std::size_t const end = row_offsets[row + 1];
                if(begin > end || end > columns.size() ||
                   end - begin != block_size + neighbor_count)
                    return Fixed16BlockStencilFallback::CSRNeighborOrder;
                for(std::size_t slot = 0; slot < block_size; ++slot)
                {
                    std::size_t const expected_column = cell * block_size +
                        Fixed16OrderedBlockColumn(row_group, slot);
                    if(columns[begin + slot] != expected_column)
                        return Fixed16BlockStencilFallback::
                            CSRBlockColumnOrder;
                    if(!same_value(values[begin + slot],
                                   matrix.LocalBlockValues[
                                       row * block_size + slot]))
                        return Fixed16BlockStencilFallback::CSRValueMismatch;
                }
                for(std::size_t neighbor = 0;
                    neighbor < neighbor_count; ++neighbor)
                {
                    std::size_t const stencil_neighbor =
                        neighbor_begin + neighbor;
                    std::size_t const entry = begin + block_size + neighbor;
                    std::size_t const expected_column =
                        matrix.NeighborCells[stencil_neighbor] * block_size +
                        row_group;
                    if(columns[entry] != expected_column)
                        return Fixed16BlockStencilFallback::CSRNeighborOrder;
                    if(!same_value(values[entry], matrix.NeighborValues[
                            stencil_neighbor * block_size + row_group]))
                        return Fixed16BlockStencilFallback::CSRValueMismatch;
                }
            }
        }
        return Fixed16BlockStencilFallback::None;
    }

    void mat_times_vec_fixed16_block_stencil(
        Fixed16BlockStencilMatrix const& matrix,
        std::vector<double> const& input,
        std::vector<double>& output,
        bool const vectorize_neighbors)
    {
        std::size_t constexpr block_size =
            Fixed16BlockStencilMatrix::BlockSize;
        if(matrix.VectorCellCount > max_size_t / block_size ||
           input.size() < matrix.VectorCellCount * block_size)
        {
            UniversalError eo(
                "Fixed-16 block-stencil input vector is too short");
            eo.addEntry("Input rows", static_cast<double>(input.size()));
            eo.addEntry("Vector cells",
                        static_cast<double>(matrix.VectorCellCount));
            throw eo;
        }
        if(matrix.LocalCellCount == 0)
        {
            output.clear();
            return;
        }

        std::vector<double> alias_copy;
        std::vector<double> const* source = &input;
        if(&input == &output)
        {
            alias_copy = input;
            source = &alias_copy;
        }
        output.resize(matrix.LocalCellCount * block_size);
        double const* const vector_values = source->data();
        double* const result = output.data();
        for(std::size_t cell = 0; cell < matrix.LocalCellCount; ++cell)
        {
            std::size_t const cell_base = cell * block_size;
            if(vectorize_neighbors)
            {
                for(std::size_t row_group = 0;
                    row_group < block_size; row_group += 4)
                {
                    Vec4d dot_products(0.0);
                    for(std::size_t slot = 0; slot < block_size; ++slot)
                    {
                        Vec4d const coefficients(
                            matrix.LocalBlockValues[
                                (cell_base + row_group) * block_size + slot],
                            matrix.LocalBlockValues[
                                (cell_base + row_group + 1) * block_size + slot],
                            matrix.LocalBlockValues[
                                (cell_base + row_group + 2) * block_size + slot],
                            matrix.LocalBlockValues[
                                (cell_base + row_group + 3) * block_size + slot]);
                        Vec4d const local_values(
                            vector_values[cell_base + Fixed16OrderedBlockColumn(
                                row_group, slot)],
                            vector_values[cell_base + Fixed16OrderedBlockColumn(
                                row_group + 1, slot)],
                            vector_values[cell_base + Fixed16OrderedBlockColumn(
                                row_group + 2, slot)],
                            vector_values[cell_base + Fixed16OrderedBlockColumn(
                                row_group + 3, slot)]);
                        dot_products = mul_add(
                            coefficients, local_values, dot_products);
                    }
                    dot_products.store(result + cell_base + row_group);
                }
            }
            else
                for(std::size_t row_group = 0;
                    row_group < block_size; ++row_group)
                {
                    std::size_t const row = cell_base + row_group;
                    std::size_t const block_begin = row * block_size;
                    double dot_product = 0.0;
                    for(std::size_t slot = 0; slot < block_size; ++slot)
                        dot_product += matrix.LocalBlockValues[
                            block_begin + slot] * vector_values[cell_base +
                                Fixed16OrderedBlockColumn(row_group, slot)];
                    for(std::size_t neighbor = matrix.NeighborOffsets[cell];
                        neighbor < matrix.NeighborOffsets[cell + 1]; ++neighbor)
                        dot_product += matrix.NeighborValues[
                            neighbor * block_size + row_group] * vector_values[
                                matrix.NeighborCells[neighbor] * block_size +
                                row_group];
                    result[row] = dot_product;
                }
            if(vectorize_neighbors)
                for(std::size_t neighbor = matrix.NeighborOffsets[cell];
                    neighbor < matrix.NeighborOffsets[cell + 1]; ++neighbor)
                {
                    double const* const neighbor_coefficients =
                        matrix.NeighborValues.data() + neighbor * block_size;
                    double const* const neighbor_input = vector_values +
                        matrix.NeighborCells[neighbor] * block_size;
                    for(std::size_t group = 0; group < block_size; group += 4)
                    {
                        Vec4d accumulated;
                        Vec4d coefficients;
                        Vec4d neighbor_values;
                        accumulated.load(result + cell_base + group);
                        coefficients.load(neighbor_coefficients + group);
                        neighbor_values.load(neighbor_input + group);
                        accumulated =
                            mul_add(coefficients, neighbor_values, accumulated);
                        accumulated.store(result + cell_base + group);
                    }
                }
        }
    }

    char const* PreconditionerKindLabel(PreconditionerKind const kind)
    {
        switch(kind)
        {
        case PreconditionerKind::ScalarJacobi:
            return "scalar_jacobi";
        case PreconditionerKind::CellBlockJacobi:
            return "cell_block_jacobi";
        case PreconditionerKind::CellBlockGaussSeidel:
            return "cell_block_gauss_seidel";
        case PreconditionerKind::RankLocalILU0:
            return "rank_local_ilu0";
        case PreconditionerKind::CellBlockJacobiTwoSweep:
            return "cell_block_jacobi_two_sweep";
        case PreconditionerKind::CellBlockJacobiFourSweep:
            return "cell_block_jacobi_four_sweep";
        case PreconditionerKind::CellBlockJacobiEightSweep:
            return "cell_block_jacobi_eight_sweep";
        }
        return "unknown";
    }

    char const* CellBlockFallbackReasonLabel(
        CellBlockFallbackReason const reason)
    {
        switch(reason)
        {
        case CellBlockFallbackReason::None:
            return "none";
        case CellBlockFallbackReason::NonFiniteEntry:
            return "non_finite_entry";
        case CellBlockFallbackReason::ZeroRowScale:
            return "zero_row_scale";
        case CellBlockFallbackReason::UnsafePivot:
            return "unsafe_pivot";
        case CellBlockFallbackReason::NonFiniteFactor:
            return "non_finite_factor";
        }
        return "unknown";
    }

    char const* HistoricalMGBranchLabel(HistoricalMGBranch const branch)
    {
        switch(branch)
        {
        case HistoricalMGBranch::Continue:
            return "continue";
        case HistoricalMGBranch::LooseMaxima:
            return "historical_loose_maxima";
        case HistoricalMGBranch::NormalTolerance:
            return "historical_normal_tolerance";
        case HistoricalMGBranch::ExtremelySmallError:
            return "historical_extremely_small_error";
        case HistoricalMGBranch::TinyRho:
            return "historical_tiny_rho";
        case HistoricalMGBranch::TinyAlphaOmega:
            return "historical_tiny_alpha_omega";
        case HistoricalMGBranch::RejectNonFinite:
            return "historical_nonfinite";
        case HistoricalMGBranch::RejectEarlyBreakdown:
            return "historical_breakdown_before_iteration_10";
        case HistoricalMGBranch::RejectBreakdownMetrics:
            return "historical_breakdown_maxima";
        }
        return "unknown";
    }

#ifdef RICH_MPI
    struct HistoricalMGPipelinePhaseA
    {
        HistoricalMGMetrics metrics;
        int valid = 0;
        double maximum_solution = 0;
        double weighted[2] = {0, 0};
    };

    struct HistoricalMGPipelineDoubleRank
    {
        double value = 0;
        int rank = 0;
    };

    struct HistoricalMGPipelinePhaseB
    {
        HistoricalMGPipelineDoubleRank maxima[2];
        std::size_t local_max0 = max_size_t;
        std::size_t local_max1 = max_size_t;
        std::size_t local_negative = max_size_t;
        int negative_rank = -1;
    };

    static HistoricalMGPipelinePhaseA BuildHistoricalMGPipelinePhaseA(
        std::vector<double> const& physical_solution,
        std::vector<double> const& previous_physical_solution,
        std::vector<double> const& physical_residual,
        std::vector<double> const& physical_rhs,
        std::vector<double> const& diagonal,
        std::size_t const runtime_group_count)
    {
        HistoricalMGPipelinePhaseA phase;
        std::size_t const size = physical_solution.size();
        phase.valid = runtime_group_count > 0 &&
            previous_physical_solution.size() == size &&
            physical_residual.size() == size && physical_rhs.size() == size &&
            diagonal.size() == size ? 1 : 0;
        if(phase.valid != 0)
            for(std::size_t row = 0; row < size; ++row)
            {
                double const x = physical_solution[row];
                double const old_x = previous_physical_solution[row];
                double const residual = physical_residual[row];
                double const rhs = physical_rhs[row];
                double const row_diagonal = diagonal[row];
                if(!std::isfinite(x) || !std::isfinite(old_x) ||
                   !std::isfinite(residual) || !std::isfinite(rhs) ||
                   !std::isfinite(row_diagonal) || row_diagonal <= 0)
                {
                    phase.valid = 0;
                    continue;
                }
                phase.maximum_solution = std::max(
                    phase.maximum_solution, std::abs(x));
                phase.metrics.weighted_residual_squared +=
                    residual * residual / row_diagonal;
                phase.metrics.weighted_rhs_squared +=
                    rhs * rhs / row_diagonal;
                if(!std::isfinite(phase.metrics.weighted_residual_squared) ||
                   !std::isfinite(phase.metrics.weighted_rhs_squared))
                    phase.valid = 0;
            }
        phase.weighted[0] = phase.metrics.weighted_residual_squared;
        phase.weighted[1] = phase.metrics.weighted_rhs_squared;
        return phase;
    }

    static void CompleteHistoricalMGPipelinePhaseA(
        HistoricalMGPipelinePhaseA& phase)
    {
        phase.metrics.weighted_residual_squared = phase.weighted[0];
        phase.metrics.weighted_rhs_squared = phase.weighted[1];
    }

    static HistoricalMGPipelinePhaseB BuildHistoricalMGPipelinePhaseB(
        std::vector<double> const& physical_solution,
        std::vector<double> const& previous_physical_solution,
        std::vector<double> const& physical_residual,
        std::vector<double> const& diagonal,
        std::size_t const runtime_group_count,
        HistoricalMGPipelinePhaseA const& phase_a,
        int const rank)
    {
        HistoricalMGPipelinePhaseB phase_b;
        std::size_t const size = physical_solution.size();
        phase_b.maxima[0].value = size > 0 ? 0.0 : -1.0;
        phase_b.maxima[0].rank = rank;
        phase_b.maxima[1].value = size > 0 ? 0.0 : -1.0;
        phase_b.maxima[1].rank = rank;
        double const max0_factor = historical_mg_max0_factor *
            (runtime_group_count > 1 ? 3.0 : 1.0);
        double const tiny = std::numeric_limits<double>::min() * 100;
        if(phase_a.valid != 0)
            for(std::size_t row = 0; row < size; ++row)
            {
                double const max0_denominator =
                    std::abs(physical_solution[row]) + tiny +
                    phase_a.maximum_solution * max0_factor;
                double const max1_denominator = std::abs(diagonal[row] *
                    (std::abs(physical_solution[row]) + tiny +
                     phase_a.maximum_solution * max0_factor * 0.5));
                double max0 = std::abs(physical_solution[row] -
                    previous_physical_solution[row]) / max0_denominator;
                double max1 = std::abs(physical_residual[row]) /
                    max1_denominator;
                if(!std::isfinite(max0))
                    max0 = BoundedHistoricalRatio(
                        std::abs(static_cast<long double>(
                            physical_solution[row]) -
                            static_cast<long double>(
                                previous_physical_solution[row])),
                        static_cast<long double>(max0_denominator));
                if(!std::isfinite(max1))
                    max1 = BoundedHistoricalRatio(
                        std::abs(static_cast<long double>(
                            physical_residual[row])),
                        std::abs(static_cast<long double>(diagonal[row])) *
                            (std::abs(static_cast<long double>(
                                physical_solution[row])) +
                             static_cast<long double>(tiny) +
                             static_cast<long double>(
                                 phase_a.maximum_solution) *
                                 static_cast<long double>(max0_factor) * 0.5L));
                if(max0 > phase_b.maxima[0].value ||
                   phase_b.local_max0 == max_size_t)
                {
                    phase_b.maxima[0].value = max0;
                    phase_b.local_max0 = row;
                }
                if(max1 > phase_b.maxima[1].value ||
                   phase_b.local_max1 == max_size_t)
                {
                    phase_b.maxima[1].value = max1;
                    phase_b.local_max1 = row;
                }
                if(phase_b.local_negative == max_size_t &&
                   physical_solution[row] <
                       -phase_a.maximum_solution * 1e-10)
                    phase_b.local_negative = row;
            }
        phase_b.negative_rank = phase_b.local_negative == max_size_t ?
            std::numeric_limits<int>::max() : rank;
        return phase_b;
    }

    static HistoricalMGMetrics CompleteHistoricalMGPipelinePhaseB(
        HistoricalMGPipelinePhaseA const& phase_a,
        HistoricalMGPipelinePhaseB const& phase_b,
        int const rank)
    {
        HistoricalMGMetrics result = phase_a.metrics;
        result.max0 = std::max(0.0, phase_b.maxima[0].value);
        result.max1 = std::max(0.0, phase_b.maxima[1].value);
        result.max0_rank = phase_b.maxima[0].rank;
        result.max1_rank = phase_b.maxima[1].rank;
        result.negative = phase_b.negative_rank ==
            std::numeric_limits<int>::max() ? 0 : 1;
        result.negative_rank = result.negative ?
            phase_b.negative_rank : -1;
        if(rank == result.max0_rank)
            result.max0_unknown = phase_b.local_max0;
        if(rank == result.max1_rank)
            result.max1_unknown = phase_b.local_max1;
        if(result.negative && rank == phase_b.negative_rank)
            result.negative_unknown = phase_b.local_negative;
        result.finite = phase_a.valid != 0 &&
            std::isfinite(result.weighted_residual_squared) &&
            std::isfinite(result.weighted_rhs_squared) &&
            result.weighted_rhs_squared > 0;
        if(result.finite)
        {
            result.historical_error = result.weighted_residual_squared /
                result.weighted_rhs_squared;
            result.finite = std::isfinite(result.historical_error) &&
                std::isfinite(result.max0) && std::isfinite(result.max1);
        }
        return result;
    }

    static bool HistoricalMGPipelineMetricsBitwiseEqual(
        HistoricalMGMetrics const& left,
        HistoricalMGMetrics const& right)
    {
        return std::memcmp(&left.weighted_residual_squared,
                           &right.weighted_residual_squared,
                           sizeof(double)) == 0 &&
            std::memcmp(&left.weighted_rhs_squared,
                        &right.weighted_rhs_squared,
                        sizeof(double)) == 0 &&
            std::memcmp(&left.historical_error, &right.historical_error,
                        sizeof(double)) == 0 &&
            std::memcmp(&left.max0, &right.max0, sizeof(double)) == 0 &&
            std::memcmp(&left.max1, &right.max1, sizeof(double)) == 0 &&
            left.negative == right.negative &&
            left.max0_unknown == right.max0_unknown &&
            left.max1_unknown == right.max1_unknown &&
            left.negative_unknown == right.negative_unknown &&
            left.max0_rank == right.max0_rank &&
            left.max1_rank == right.max1_rank &&
            left.negative_rank == right.negative_rank &&
            left.finite == right.finite;
    }
#endif

    static HistoricalMGMetrics MeasureHistoricalMGImpl(
        std::vector<double> const& physical_solution,
        std::vector<double> const& previous_physical_solution,
        std::vector<double> const& physical_residual,
        std::vector<double> const& physical_rhs,
        std::vector<double> const& diagonal,
        std::size_t const runtime_group_count,
        double* const reduction_seconds,
        bool const concurrent_reductions,
        double* const concurrent_sum_values,
        std::size_t const concurrent_sum_count,
        unsigned long long* const collective_calls,
        unsigned long long* const causal_rounds)
    {
#ifndef RICH_MPI
        (void)concurrent_reductions;
        (void)concurrent_sum_values;
        (void)concurrent_sum_count;
        (void)collective_calls;
        (void)causal_rounds;
#endif
        HistoricalMGMetrics result;
        std::size_t const size = physical_solution.size();
        int rank = 0;
#ifdef RICH_MPI
        RequireCGMpiSuccess(MPI_Comm_rank(MPI_COMM_WORLD, &rank),
                            "MPI_Comm_rank(historical MG metrics)");
#endif
        int valid = runtime_group_count > 0 &&
            previous_physical_solution.size() == size &&
            physical_residual.size() == size && physical_rhs.size() == size &&
            diagonal.size() == size ? 1 : 0;
        double maximum_solution = 0;
        if(valid != 0)
            for(std::size_t row = 0; row < size; ++row)
            {
                double const x = physical_solution[row];
                double const old_x = previous_physical_solution[row];
                double const residual = physical_residual[row];
                double const rhs = physical_rhs[row];
                double const row_diagonal = diagonal[row];
                if(!std::isfinite(x) || !std::isfinite(old_x) ||
                   !std::isfinite(residual) || !std::isfinite(rhs) ||
                   !std::isfinite(row_diagonal) || row_diagonal <= 0)
                {
                    valid = 0;
                    continue;
                }
                maximum_solution = std::max(maximum_solution, std::abs(x));
                result.weighted_residual_squared +=
                    residual * residual / row_diagonal;
                result.weighted_rhs_squared += rhs * rhs / row_diagonal;
                if(!std::isfinite(result.weighted_residual_squared) ||
                   !std::isfinite(result.weighted_rhs_squared))
                    valid = 0;
            }

        auto const reduction_start = std::chrono::steady_clock::now();
#ifdef RICH_MPI
        double weighted[2] = {result.weighted_residual_squared,
                              result.weighted_rhs_squared};
        if(concurrent_reductions)
        {
            MPI_Request requests[4];
            int request_count = 0;
            RequireCGMpiSuccess(
                MPI_Iallreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN,
                               MPI_COMM_WORLD, &requests[request_count++]),
                "MPI_Iallreduce(historical MG initial validity)");
            RequireCGMpiSuccess(
                MPI_Iallreduce(MPI_IN_PLACE, &maximum_solution, 1,
                               MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD,
                               &requests[request_count++]),
                "MPI_Iallreduce(historical MG maximum solution)");
            RequireCGMpiSuccess(
                MPI_Iallreduce(MPI_IN_PLACE, weighted, 2, MPI_DOUBLE,
                               MPI_SUM, MPI_COMM_WORLD,
                               &requests[request_count++]),
                "MPI_Iallreduce(historical MG weighted norms)");
            if(concurrent_sum_count > 0)
            {
                RequireCGMpiSuccess(
                    MPI_Iallreduce(MPI_IN_PLACE, concurrent_sum_values,
                                   static_cast<int>(concurrent_sum_count),
                                   MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD,
                                   &requests[request_count++]),
                    "MPI_Iallreduce(historical MG companion sums)");
            }
            RequireCGMpiSuccess(
                MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE),
                "MPI_Waitall(historical MG initial reductions)");
            RecordReductionBatch(collective_calls, causal_rounds,
                                 static_cast<unsigned long long>(request_count),
                                 1);
        }
        else
        {
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(historical MG initial validity)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &maximum_solution, 1, MPI_DOUBLE,
                              MPI_MAX, MPI_COMM_WORLD),
                "MPI_Allreduce(historical MG maximum solution)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, weighted, 2, MPI_DOUBLE, MPI_SUM,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(historical MG weighted norms)");
            RecordReductionBatch(collective_calls, causal_rounds, 3, 3);
        }
        result.weighted_residual_squared = weighted[0];
        result.weighted_rhs_squared = weighted[1];
#endif
        if(reduction_seconds != nullptr)
            *reduction_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - reduction_start).count();

        double const max0_factor = historical_mg_max0_factor *
            (runtime_group_count > 1 ? 3.0 : 1.0);
        double const tiny = std::numeric_limits<double>::min() * 100;
        struct DoubleRank
        {
            double value;
            int rank;
        } maxima[2] = {{size > 0 ? 0.0 : -1.0, rank},
                       {size > 0 ? 0.0 : -1.0, rank}};
        std::size_t local_max0 = max_size_t;
        std::size_t local_max1 = max_size_t;
        std::size_t local_negative = max_size_t;
        if(valid != 0)
            for(std::size_t row = 0; row < size; ++row)
            {
                double const max0_denominator =
                    std::abs(physical_solution[row]) + tiny +
                    maximum_solution * max0_factor;
                double const max1_denominator = std::abs(diagonal[row] *
                    (std::abs(physical_solution[row]) + tiny +
                     maximum_solution * max0_factor * 0.5));
                double max0 = std::abs(physical_solution[row] -
                    previous_physical_solution[row]) / max0_denominator;
                double max1 = std::abs(physical_residual[row]) /
                    max1_denominator;
                // Both maxima are auxiliary historical acceptance guards.  A
                // finite state can still form 0/0 here when the double
                // product in max1_denominator underflows.  Re-evaluate only
                // that exceptional path in the wider type and saturate a
                // genuinely huge ratio so it cannot cause acceptance.
                if(!std::isfinite(max0))
                    max0 = BoundedHistoricalRatio(
                        std::abs(static_cast<long double>(
                            physical_solution[row]) -
                            static_cast<long double>(
                                previous_physical_solution[row])),
                        static_cast<long double>(max0_denominator));
                if(!std::isfinite(max1))
                    max1 = BoundedHistoricalRatio(
                        std::abs(static_cast<long double>(
                            physical_residual[row])),
                        std::abs(static_cast<long double>(diagonal[row])) *
                            (std::abs(static_cast<long double>(
                                physical_solution[row])) +
                             static_cast<long double>(tiny) +
                             static_cast<long double>(maximum_solution) *
                                 static_cast<long double>(max0_factor) * 0.5L));
                if(max0 > maxima[0].value || local_max0 == max_size_t)
                {
                    maxima[0].value = max0;
                    local_max0 = row;
                }
                if(max1 > maxima[1].value || local_max1 == max_size_t)
                {
                    maxima[1].value = max1;
                    local_max1 = row;
                }
                if(local_negative == max_size_t &&
                   physical_solution[row] < -maximum_solution * 1e-10)
                    local_negative = row;
            }
        int negative_rank = local_negative == max_size_t ?
            std::numeric_limits<int>::max() : rank;

        auto const maxima_reduction_start = std::chrono::steady_clock::now();
#ifdef RICH_MPI
        if(concurrent_reductions)
        {
            // valid was made global in phase A and is not modified while the
            // per-row maxima are formed, so its historical second reduction
            // is redundant on this opt-in path.
            MPI_Request requests[2];
            RequireCGMpiSuccess(
                MPI_Iallreduce(MPI_IN_PLACE, maxima, 2, MPI_DOUBLE_INT,
                               MPI_MAXLOC, MPI_COMM_WORLD, &requests[0]),
                "MPI_Iallreduce(historical MG maxima)");
            RequireCGMpiSuccess(
                MPI_Iallreduce(MPI_IN_PLACE, &negative_rank, 1, MPI_INT,
                               MPI_MIN, MPI_COMM_WORLD, &requests[1]),
                "MPI_Iallreduce(historical MG negative rank)");
            RequireCGMpiSuccess(
                MPI_Waitall(2, requests, MPI_STATUSES_IGNORE),
                "MPI_Waitall(historical MG final reductions)");
            RecordReductionBatch(collective_calls, causal_rounds, 2, 1);
        }
        else
        {
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(historical MG final validity)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, maxima, 2, MPI_DOUBLE_INT,
                              MPI_MAXLOC, MPI_COMM_WORLD),
                "MPI_Allreduce(historical MG maxima)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &negative_rank, 1, MPI_INT,
                              MPI_MIN, MPI_COMM_WORLD),
                "MPI_Allreduce(historical MG negative rank)");
            RecordReductionBatch(collective_calls, causal_rounds, 3, 3);
        }
#endif
        if(reduction_seconds != nullptr)
            *reduction_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() -
                maxima_reduction_start).count();

        result.max0 = std::max(0.0, maxima[0].value);
        result.max1 = std::max(0.0, maxima[1].value);
        result.max0_rank = maxima[0].rank;
        result.max1_rank = maxima[1].rank;
        result.negative = negative_rank == std::numeric_limits<int>::max() ?
            0 : 1;
        result.negative_rank = result.negative ? negative_rank : -1;
        if(rank == result.max0_rank)
            result.max0_unknown = local_max0;
        if(rank == result.max1_rank)
            result.max1_unknown = local_max1;
        if(result.negative && rank == negative_rank)
            result.negative_unknown = local_negative;
        result.finite = valid != 0 &&
            std::isfinite(result.weighted_residual_squared) &&
            std::isfinite(result.weighted_rhs_squared) &&
            result.weighted_rhs_squared > 0;
        if(result.finite)
        {
            result.historical_error = result.weighted_residual_squared /
                result.weighted_rhs_squared;
            result.finite = std::isfinite(result.historical_error) &&
                std::isfinite(result.max0) && std::isfinite(result.max1);
        }
        return result;
    }

    HistoricalMGMetrics MeasureHistoricalMG(
        std::vector<double> const& physical_solution,
        std::vector<double> const& previous_physical_solution,
        std::vector<double> const& physical_residual,
        std::vector<double> const& physical_rhs,
        std::vector<double> const& diagonal,
        std::size_t const runtime_group_count,
        double* const reduction_seconds)
    {
        return MeasureHistoricalMGImpl(
            physical_solution, previous_physical_solution,
            physical_residual, physical_rhs, diagonal, runtime_group_count,
            reduction_seconds, false, nullptr, 0, nullptr, nullptr);
    }

    HistoricalMGDecision ClassifyHistoricalMG(
        HistoricalMGMetrics const& metrics,
        std::size_t const zero_based_iteration,
        double const squared_tolerance,
        HistoricalMGBreakdown const breakdown,
        bool const coefficients_finite)
    {
        HistoricalMGDecision result;
        if(!metrics.finite || !coefficients_finite ||
           !std::isfinite(squared_tolerance) || squared_tolerance <= 0 ||
           breakdown == HistoricalMGBreakdown::NonFinite)
        {
            result.reject = true;
            result.branch = HistoricalMGBranch::RejectNonFinite;
            return result;
        }
        if(breakdown != HistoricalMGBreakdown::None)
        {
            std::size_t const completed_iterations = zero_based_iteration + 1;
            if(completed_iterations <
               historical_mg_minimum_breakdown_iteration)
            {
                result.reject = true;
                result.branch = HistoricalMGBranch::RejectEarlyBreakdown;
                return result;
            }
            if(metrics.max0 >= historical_mg_loose_max0 ||
               metrics.max1 >= historical_mg_loose_max1)
            {
                result.reject = true;
                result.branch = HistoricalMGBranch::RejectBreakdownMetrics;
                return result;
            }
            result.accept = true;
            result.branch = breakdown == HistoricalMGBreakdown::TinyRho ?
                HistoricalMGBranch::TinyRho :
                HistoricalMGBranch::TinyAlphaOmega;
            return result;
        }
        if(zero_based_iteration >=
               historical_mg_minimum_breakdown_iteration &&
           metrics.historical_error < historical_mg_loose_error &&
           metrics.max0 < historical_mg_loose_max0 &&
           metrics.max1 < historical_mg_loose_max1)
        {
            result.accept = true;
            result.branch = HistoricalMGBranch::LooseMaxima;
            return result;
        }
        if(metrics.historical_error < squared_tolerance)
        {
            if(metrics.historical_error < 1e-100)
            {
                result.accept = true;
                result.branch = HistoricalMGBranch::ExtremelySmallError;
                return result;
            }
            if(metrics.max1 < historical_mg_normal_max1 &&
               metrics.max0 < historical_mg_normal_max0 &&
               ((zero_based_iteration > 25 && metrics.max1 < 1e-10) ||
                metrics.negative < 0.5))
            {
                result.accept = true;
                result.branch = HistoricalMGBranch::NormalTolerance;
            }
        }
        return result;
    }

    HistoricalMGDecision ClassifyHistoricalMGCorrectionCandidate(
        HistoricalMGMetrics const& metrics,
        std::size_t const zero_based_iteration,
        double const squared_tolerance)
    {
        HistoricalMGDecision const direct = ClassifyHistoricalMG(
            metrics, zero_based_iteration, squared_tolerance);
        if(direct.accept || direct.reject || metrics.negative == 0)
            return direct;
        HistoricalMGMetrics corrected_candidate_metrics = metrics;
        corrected_candidate_metrics.negative = 0;
        return ClassifyHistoricalMG(
            corrected_candidate_metrics, zero_based_iteration,
            squared_tolerance);
    }

    void CellBlockJacobiPreconditioner::RecordFallback(
        size_t const block,
        size_t const group,
        CellBlockFallbackReason const reason)
    {
        ++fallback_block_count_;
        if(first_fallback_block_ == max_size_t)
        {
            first_fallback_block_ = block;
            first_fallback_group_ = group;
            first_fallback_reason_ = reason;
        }
    }

    bool CellBlockJacobiPreconditioner::Setup(
        mat const& matrix,
        size_t_mat const& columns,
        size_t const block_size,
        PreconditionerKind const kind,
        std::vector<double> const& scalar_inverse_diagonal)
    {
        return SetupImpl(
            &matrix, &columns, nullptr, nullptr, nullptr, nullptr, nullptr,
            matrix.size(),
            block_size, kind, scalar_inverse_diagonal);
    }

    bool CellBlockJacobiPreconditioner::SetupCSR(
        std::vector<size_t> const& row_offsets,
        std::vector<matrix_index_t> const& columns,
        std::vector<double> const& values,
        size_t const block_size,
        PreconditionerKind const kind,
        std::vector<double> const& scalar_inverse_diagonal)
    {
        size_t const row_count = row_offsets.empty() ? 0 :
            row_offsets.size() - 1;
        return SetupImpl(
            nullptr, nullptr, &row_offsets, &columns, nullptr, &values,
            nullptr,
            row_count,
            block_size, kind, scalar_inverse_diagonal);
    }

    bool CellBlockJacobiPreconditioner::SetupCSR(
        std::vector<size_t> const& row_offsets,
        std::vector<std::uint32_t> const& columns,
        std::vector<double> const& values,
        size_t const block_size,
        PreconditionerKind const kind,
        std::vector<double> const& scalar_inverse_diagonal)
    {
        size_t const row_count = row_offsets.empty() ? 0 :
            row_offsets.size() - 1;
        return SetupImpl(
            nullptr, nullptr, &row_offsets, nullptr, &columns, &values,
            nullptr,
            row_count,
            block_size, kind, scalar_inverse_diagonal);
    }

    bool CellBlockJacobiPreconditioner::SetupFixed16BlockStencil(
        Fixed16BlockStencilMatrix const& matrix,
        PreconditionerKind const kind,
        std::vector<double> const& scalar_inverse_diagonal)
    {
        if(matrix.LocalCellCount >
           max_size_t / Fixed16BlockStencilMatrix::BlockSize)
            return false;
        return SetupImpl(
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &matrix,
            matrix.LocalCellCount * Fixed16BlockStencilMatrix::BlockSize,
            Fixed16BlockStencilMatrix::BlockSize, kind,
            scalar_inverse_diagonal);
    }

    bool CellBlockJacobiPreconditioner::SetupRankLocalILU0(
        std::vector<size_t> const& row_offsets,
        std::vector<matrix_index_t> const& columns,
        std::vector<double> const& values)
    {
        auto const reject = [&]()
        {
            std::vector<double>().swap(ilu0_factors_);
            std::vector<size_t>().swap(ilu0_diagonal_positions_);
            minimum_normalized_pivot_ =
                std::numeric_limits<double>::quiet_NaN();
            return false;
        };
        if(row_offsets.size() != unknown_count_ + 1 || row_offsets.empty() ||
           row_offsets.front() != 0 || row_offsets.back() != columns.size() ||
           columns.size() != values.size())
            return reject();

        ilu0_factors_ = values;
        ilu0_diagonal_positions_.assign(unknown_count_, max_size_t);
        std::vector<size_t> positions(unknown_count_, max_size_t);
        std::vector<size_t> lower_entries;
        double minimum_pivot = std::numeric_limits<double>::infinity();
        double constexpr relative_pivot_threshold =
            64 * std::numeric_limits<double>::epsilon();

        for(size_t row = 0; row < unknown_count_; ++row)
        {
            lower_entries.clear();
            double row_scale = 0;
            for(size_t entry = row_offsets[row];
                entry < row_offsets[row + 1]; ++entry)
            {
                size_t const column = columns[entry];
                if(column >= unknown_count_)
                {
                    ++ignored_remote_coupling_count_;
                    continue;
                }
                if(positions[column] != max_size_t)
                    return reject();
                positions[column] = entry;
                row_scale = std::max(row_scale, std::abs(values[entry]));
                if(column == row)
                    ilu0_diagonal_positions_[row] = entry;
                else if(column < row)
                {
                    lower_entries.push_back(entry);
                    ++local_lower_coupling_count_;
                }
                else
                    ++ignored_local_upper_coupling_count_;
            }
            if(ilu0_diagonal_positions_[row] == max_size_t ||
               !std::isfinite(row_scale) || row_scale == 0)
                return reject();

            std::sort(lower_entries.begin(), lower_entries.end(),
                [&](size_t const left, size_t const right)
                {
                    return columns[left] < columns[right];
                });
            for(size_t const lower_entry : lower_entries)
            {
                size_t const lower_column = columns[lower_entry];
                size_t const lower_diagonal =
                    ilu0_diagonal_positions_[lower_column];
                if(lower_diagonal == max_size_t)
                    return reject();
                double const diagonal = ilu0_factors_[lower_diagonal];
                if(!std::isfinite(diagonal) || diagonal == 0)
                    return reject();
                double const multiplier =
                    ilu0_factors_[lower_entry] / diagonal;
                if(!std::isfinite(multiplier))
                    return reject();
                ilu0_factors_[lower_entry] = multiplier;

                for(size_t upper_entry = row_offsets[lower_column];
                    upper_entry < row_offsets[lower_column + 1];
                    ++upper_entry)
                {
                    size_t const upper_column = columns[upper_entry];
                    if(upper_column <= lower_column ||
                       upper_column >= unknown_count_)
                        continue;
                    size_t const target = positions[upper_column];
                    if(target == max_size_t)
                        continue;
                    double const updated = ilu0_factors_[target] -
                        multiplier * ilu0_factors_[upper_entry];
                    if(!std::isfinite(updated))
                        return reject();
                    ilu0_factors_[target] = updated;
                }
            }

            double const pivot =
                ilu0_factors_[ilu0_diagonal_positions_[row]];
            double const normalized_pivot = std::abs(pivot) / row_scale;
            if(!std::isfinite(normalized_pivot) ||
               normalized_pivot <= relative_pivot_threshold)
                return reject();
            minimum_pivot = std::min(minimum_pivot, normalized_pivot);

            for(size_t entry = row_offsets[row];
                entry < row_offsets[row + 1]; ++entry)
                if(columns[entry] < unknown_count_)
                    positions[columns[entry]] = max_size_t;
        }

        minimum_normalized_pivot_ = unknown_count_ == 0 ?
            std::numeric_limits<double>::quiet_NaN() : minimum_pivot;
        return true;
    }

    bool CellBlockJacobiPreconditioner::SetupImpl(
        mat const* const matrix,
        size_t_mat const* const columns,
        std::vector<size_t> const* const row_offsets,
        std::vector<matrix_index_t> const* const csr_columns,
        std::vector<std::uint32_t> const* const csr_columns32,
        std::vector<double> const* const csr_values,
        Fixed16BlockStencilMatrix const* const fixed16_block_stencil,
        size_t const row_count,
        size_t const block_size,
        PreconditionerKind const kind,
        std::vector<double> const& scalar_inverse_diagonal)
    {
        auto const setup_start = std::chrono::steady_clock::now();
        Release();
        kind_ = kind;
        requested_kind_ = kind;
        requested_kind_supported_ = true;
        unknown_count_ = row_count;
        block_size_ = block_size;
        block_count_ = 0;
        factorized_block_count_ = 0;
        fallback_block_count_ = 0;
        first_fallback_block_ = max_size_t;
        first_fallback_group_ = max_size_t;
        first_fallback_reason_ = CellBlockFallbackReason::None;
        minimum_normalized_pivot_ =
            std::numeric_limits<double>::quiet_NaN();
        setup_seconds_ = 0;
        apply_calls_ = 0;
        apply_seconds_ = 0;
        local_lower_coupling_count_ = 0;
        ignored_local_upper_coupling_count_ = 0;
        ignored_remote_coupling_count_ = 0;

        auto const finish = [&](bool const result)
        {
            setup_seconds_ = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - setup_start).count();
            return result;
        };

        bool const row_storage = matrix != nullptr && columns != nullptr &&
            row_offsets == nullptr && csr_columns == nullptr &&
            csr_columns32 == nullptr && csr_values == nullptr &&
            fixed16_block_stencil == nullptr;
        bool const csr_storage = matrix == nullptr && columns == nullptr &&
            row_offsets != nullptr &&
            ((csr_columns != nullptr) != (csr_columns32 != nullptr)) &&
            csr_values != nullptr && fixed16_block_stencil == nullptr;
        bool const fixed16_storage = matrix == nullptr && columns == nullptr &&
            row_offsets == nullptr && csr_columns == nullptr &&
            csr_columns32 == nullptr && csr_values == nullptr &&
            fixed16_block_stencil != nullptr;
        bool storage_valid = row_storage || csr_storage || fixed16_storage;
        if(row_storage)
            storage_valid = matrix->size() == row_count &&
                columns->size() == row_count;
        if(csr_storage) {
            storage_valid = row_offsets->size() == row_count + 1 &&
                !row_offsets->empty() && row_offsets->front() == 0 &&
                row_offsets->back() == csr_values->size() &&
                (csr_columns != nullptr ? csr_columns->size() :
                 csr_columns32->size()) == csr_values->size();
            for(size_t row = 0; storage_valid && row < row_count; ++row)
                storage_valid = (*row_offsets)[row] <=
                    (*row_offsets)[row + 1];
        }
        if(fixed16_storage)
            storage_valid = block_size ==
                Fixed16BlockStencilMatrix::BlockSize &&
                row_count == fixed16_block_stencil->LocalCellCount *
                    Fixed16BlockStencilMatrix::BlockSize &&
                ValidateFixed16BlockStencil(*fixed16_block_stencil) ==
                    Fixed16BlockStencilFallback::None;

        if((kind != PreconditionerKind::ScalarJacobi &&
            kind != PreconditionerKind::CellBlockJacobi &&
            kind != PreconditionerKind::CellBlockGaussSeidel &&
            kind != PreconditionerKind::RankLocalILU0 &&
            kind != PreconditionerKind::CellBlockJacobiTwoSweep &&
            kind != PreconditionerKind::CellBlockJacobiFourSweep &&
            kind != PreconditionerKind::CellBlockJacobiEightSweep) ||
           !storage_valid || block_size_ == 0 ||
           row_count != scalar_inverse_diagonal.size() ||
           row_count % block_size_ != 0)
            return finish(false);

        auto const rowEntryCount = [&](size_t const row)
        {
            return row_storage ? (*matrix)[row].size() :
                (csr_storage ?
                 (*row_offsets)[row + 1] - (*row_offsets)[row] :
                 Fixed16BlockStencilMatrix::BlockSize);
        };
        auto const columnAt = [&](size_t const row, size_t const entry)
        {
            return row_storage ? (*columns)[row][entry] :
                (fixed16_storage ?
                 (row / Fixed16BlockStencilMatrix::BlockSize) *
                    Fixed16BlockStencilMatrix::BlockSize +
                    Fixed16OrderedBlockColumn(
                        row % Fixed16BlockStencilMatrix::BlockSize, entry) :
                 (csr_columns != nullptr ?
                 (*csr_columns)[(*row_offsets)[row] + entry] :
                 static_cast<size_t>(
                     (*csr_columns32)[(*row_offsets)[row] + entry])));
        };
        auto const valueAt = [&](size_t const row, size_t const entry)
        {
            return row_storage ? (*matrix)[row][entry] :
                (fixed16_storage ? fixed16_block_stencil->LocalBlockValues[
                    row * Fixed16BlockStencilMatrix::BlockSize + entry] :
                 (*csr_values)[(*row_offsets)[row] + entry]);
        };

        scalar_inverse_diagonal_ = scalar_inverse_diagonal;
        for(double const inverse_diagonal : scalar_inverse_diagonal_)
            if(!std::isfinite(inverse_diagonal) || inverse_diagonal <= 0)
                return finish(false);
        for(size_t row = 0; row < row_count; ++row)
        {
            if(row_storage &&
               (*matrix)[row].size() != (*columns)[row].size())
                return finish(false);
            for(size_t entry = 0; entry < rowEntryCount(row); ++entry)
            {
                double const value = valueAt(row, entry);
                if(!std::isfinite(value))
                    return finish(false);
            }
        }

        block_count_ = unknown_count_ / block_size_;
        apply_scratch_.assign(block_size_, 0);

        if(kind_ == PreconditionerKind::CellBlockGaussSeidel)
        {
            requested_kind_supported_ = !row_storage && !fixed16_storage &&
                csr_columns != nullptr;
            for(size_t row = 0;
                requested_kind_supported_ && row < row_count; ++row)
            {
                size_t const row_block = row / block_size_;
                size_t const row_group = row % block_size_;
                for(size_t entry = 0; entry < rowEntryCount(row); ++entry)
                {
                    size_t const column = columnAt(row, entry);
                    if(column == max_size_t)
                    {
                        requested_kind_supported_ = false;
                        break;
                    }
                    size_t const column_block = column / block_size_;
                    if(column_block == row_block)
                        continue;
                    if(column % block_size_ != row_group)
                    {
                        requested_kind_supported_ = false;
                        break;
                    }
                    if(column >= unknown_count_)
                        ++ignored_remote_coupling_count_;
                    else if(column_block < row_block)
                        ++local_lower_coupling_count_;
                    else
                        ++ignored_local_upper_coupling_count_;
                }
            }
            if(!requested_kind_supported_)
                kind_ = PreconditionerKind::CellBlockJacobi;
        }
        if(kind_ == PreconditionerKind::RankLocalILU0)
        {
            requested_kind_supported_ = !row_storage && !fixed16_storage &&
                csr_columns != nullptr &&
                SetupRankLocalILU0(
                    *row_offsets, *csr_columns, *csr_values);
            if(requested_kind_supported_)
                return finish(true);
            kind_ = PreconditionerKind::CellBlockJacobi;
            local_lower_coupling_count_ = 0;
            ignored_local_upper_coupling_count_ = 0;
            ignored_remote_coupling_count_ = 0;
        }
        if(kind_ == PreconditionerKind::ScalarJacobi)
            return finish(true);

        if(block_count_ > max_size_t / block_size_ ||
           block_count_ * block_size_ > max_size_t / block_size_)
            return finish(false);

        size_t const factor_count =
            block_count_ * block_size_ * block_size_;
        lu_factors_.assign(factor_count, 0);
        inverse_row_scales_.assign(unknown_count_, 0);
        pivots_.assign(unknown_count_, 0);
        factorized_blocks_.assign(block_count_, 0);

        double minimum_pivot = std::numeric_limits<double>::infinity();
        double constexpr relative_pivot_threshold =
            64 * std::numeric_limits<double>::epsilon();

        for(size_t block = 0; block < block_count_; ++block)
        {
            size_t const row_begin = block * block_size_;
            size_t const factor_begin =
                block * block_size_ * block_size_;
            bool valid_block = true;
            CellBlockFallbackReason fallback_reason =
                CellBlockFallbackReason::None;
            size_t fallback_group = max_size_t;

            // Accumulate every same-cell sparse entry.  This intentionally
            // handles duplicate columns rather than relying on slot order.
            for(size_t block_row = 0;
                block_row < block_size_ && valid_block; ++block_row)
            {
                size_t const row = row_begin + block_row;
                for(size_t entry = 0; entry < rowEntryCount(row); ++entry)
                {
                    size_t const column = columnAt(row, entry);
                    if(column == max_size_t || column < row_begin ||
                       column >= row_begin + block_size_)
                        continue;
                    double const value = valueAt(row, entry);
                    size_t const offset = factor_begin +
                        block_row * block_size_ + (column - row_begin);
                    if(!std::isfinite(value) ||
                       !std::isfinite(lu_factors_[offset] + value))
                    {
                        valid_block = false;
                        fallback_reason =
                            CellBlockFallbackReason::NonFiniteEntry;
                        fallback_group = block_row;
                        break;
                    }
                    lu_factors_[offset] += value;
                }
            }

            // Row equilibration is part of the preconditioner only.  Solving
            // (D B) x = D r is algebraically identical to B x = r.
            for(size_t block_row = 0;
                block_row < block_size_ && valid_block; ++block_row)
            {
                double row_scale = 0;
                for(size_t block_column = 0;
                    block_column < block_size_; ++block_column)
                    row_scale = std::max(row_scale, std::abs(
                        lu_factors_[factor_begin +
                            block_row * block_size_ + block_column]));
                if(!std::isfinite(row_scale) || row_scale <= 0)
                {
                    valid_block = false;
                    fallback_reason =
                        CellBlockFallbackReason::ZeroRowScale;
                    fallback_group = block_row;
                    break;
                }
                double const inverse_scale = 1.0 / row_scale;
                inverse_row_scales_[row_begin + block_row] = inverse_scale;
                for(size_t block_column = 0;
                    block_column < block_size_; ++block_column)
                    lu_factors_[factor_begin +
                        block_row * block_size_ + block_column] *=
                            inverse_scale;
            }

            double block_minimum_pivot =
                std::numeric_limits<double>::infinity();
            for(size_t k = 0; k < block_size_ && valid_block; ++k)
            {
                size_t pivot = k;
                double pivot_abs = 0;
                double active_scale = 0;
                for(size_t row = k; row < block_size_; ++row)
                {
                    double const candidate = std::abs(
                        lu_factors_[factor_begin + row * block_size_ + k]);
                    if(candidate > pivot_abs)
                    {
                        pivot = row;
                        pivot_abs = candidate;
                    }
                    for(size_t column = k;
                        column < block_size_; ++column)
                        active_scale = std::max(active_scale, std::abs(
                            lu_factors_[factor_begin +
                                row * block_size_ + column]));
                }
                double const normalized_pivot =
                    active_scale > 0 ? pivot_abs / active_scale : 0;
                if(!std::isfinite(normalized_pivot) ||
                   normalized_pivot <= relative_pivot_threshold)
                {
                    valid_block = false;
                    fallback_reason = CellBlockFallbackReason::UnsafePivot;
                    fallback_group = k;
                    break;
                }
                block_minimum_pivot =
                    std::min(block_minimum_pivot, normalized_pivot);
                pivots_[row_begin + k] = pivot;
                if(pivot != k)
                    for(size_t column = 0;
                        column < block_size_; ++column)
                        std::swap(
                            lu_factors_[factor_begin +
                                k * block_size_ + column],
                            lu_factors_[factor_begin +
                                pivot * block_size_ + column]);

                double const diagonal =
                    lu_factors_[factor_begin + k * block_size_ + k];
                if(!std::isfinite(diagonal))
                {
                    valid_block = false;
                    fallback_reason =
                        CellBlockFallbackReason::NonFiniteFactor;
                    fallback_group = k;
                    break;
                }
                for(size_t row = k + 1;
                    row < block_size_ && valid_block; ++row)
                {
                    size_t const lower_offset =
                        factor_begin + row * block_size_ + k;
                    double const factor =
                        lu_factors_[lower_offset] / diagonal;
                    if(!std::isfinite(factor))
                    {
                        valid_block = false;
                        fallback_reason =
                            CellBlockFallbackReason::NonFiniteFactor;
                        fallback_group = k;
                        break;
                    }
                    lu_factors_[lower_offset] = factor;
                    for(size_t column = k + 1;
                        column < block_size_; ++column)
                    {
                        size_t const offset =
                            factor_begin + row * block_size_ + column;
                        lu_factors_[offset] -= factor *
                            lu_factors_[factor_begin +
                                k * block_size_ + column];
                        if(!std::isfinite(lu_factors_[offset]))
                        {
                            valid_block = false;
                            fallback_reason =
                                CellBlockFallbackReason::NonFiniteFactor;
                            fallback_group = k;
                            break;
                        }
                    }
                }
            }

            if(valid_block)
            {
                factorized_blocks_[block] = 1;
                ++factorized_block_count_;
                minimum_pivot =
                    std::min(minimum_pivot, block_minimum_pivot);
            }
            else
                RecordFallback(block, fallback_group, fallback_reason);
        }

        if(std::isfinite(minimum_pivot))
            minimum_normalized_pivot_ = minimum_pivot;
        return finish(true);
    }

    void CellBlockJacobiPreconditioner::SolveBlock(
        size_t const block,
        std::vector<double>& work) const
    {
        size_t const row_begin = block * block_size_;
        bool const use_factor =
            kind_ != PreconditionerKind::ScalarJacobi &&
            factorized_blocks_[block] != 0;
        if(!use_factor)
        {
            for(size_t row = 0; row < block_size_; ++row)
                work[row] *= scalar_inverse_diagonal_[row_begin + row];
            return;
        }

        size_t const factor_begin = block * block_size_ * block_size_;
        for(size_t row = 0; row < block_size_; ++row)
            work[row] *= inverse_row_scales_[row_begin + row];
        for(size_t k = 0; k < block_size_; ++k)
            if(pivots_[row_begin + k] != k)
                std::swap(work[k], work[pivots_[row_begin + k]]);
        for(size_t row = 0; row < block_size_; ++row)
            for(size_t column = 0; column < row; ++column)
                work[row] -= lu_factors_[factor_begin +
                    row * block_size_ + column] * work[column];
        for(size_t row = block_size_; row-- > 0;)
        {
            for(size_t column = row + 1; column < block_size_; ++column)
                work[row] -= lu_factors_[factor_begin +
                    row * block_size_ + column] * work[column];
            work[row] /= lu_factors_[factor_begin +
                row * block_size_ + row];
        }
    }

    void CellBlockJacobiPreconditioner::Apply(
        std::vector<double> const& input,
        std::vector<double>& output)
    {
        auto const apply_start = std::chrono::steady_clock::now();
        if(input.size() != unknown_count_ ||
           scalar_inverse_diagonal_.size() != unknown_count_ ||
           block_size_ == 0)
            throw UniversalError("Invalid cell-block preconditioner application");

        bool const aliases = &input == &output;
        if(!aliases)
            output.resize(unknown_count_);
        if(apply_scratch_.size() != block_size_)
            throw UniversalError("Missing cell-block preconditioner scratch storage");
        std::vector<double>& work = apply_scratch_;

        for(size_t block = 0; block < block_count_; ++block)
        {
            size_t const row_begin = block * block_size_;
            for(size_t row = 0; row < block_size_; ++row)
                work[row] = input[row_begin + row];
            SolveBlock(block, work);
            for(size_t row = 0; row < block_size_; ++row)
                output[row_begin + row] = work[row];
        }

        ++apply_calls_;
        apply_seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - apply_start).count();
    }

    void CellBlockJacobiPreconditioner::ApplyCSRForwardGaussSeidel(
        std::vector<double> const& input,
        std::vector<double>& output,
        std::vector<size_t> const& row_offsets,
        std::vector<matrix_index_t> const& columns,
        std::vector<double> const& values)
    {
        auto const apply_start = std::chrono::steady_clock::now();
        if(kind_ != PreconditionerKind::CellBlockGaussSeidel ||
           !requested_kind_supported_ || input.size() != unknown_count_ ||
           scalar_inverse_diagonal_.size() != unknown_count_ ||
           block_size_ == 0 || row_offsets.size() != unknown_count_ + 1 ||
           row_offsets.empty() || row_offsets.front() != 0 ||
           row_offsets.back() != columns.size() ||
           columns.size() != values.size())
            throw UniversalError(
                "Invalid forward block-Gauss-Seidel application");

        bool const aliases = &input == &output;
        if(!aliases)
            output.resize(unknown_count_);
        if(apply_scratch_.size() != block_size_)
            throw UniversalError(
                "Missing block-Gauss-Seidel scratch storage");
        std::vector<double>& work = apply_scratch_;

        for(size_t block = 0; block < block_count_; ++block)
        {
            size_t const row_begin = block * block_size_;
            for(size_t block_row = 0; block_row < block_size_; ++block_row)
            {
                size_t const row = row_begin + block_row;
                work[block_row] = input[row];
                for(size_t entry = row_offsets[row];
                    entry < row_offsets[row + 1]; ++entry)
                {
                    size_t const column = columns[entry];
                    if(column >= unknown_count_ ||
                       column / block_size_ >= block)
                        continue;
                    work[block_row] -= values[entry] * output[column];
                }
            }
            SolveBlock(block, work);
            for(size_t block_row = 0; block_row < block_size_; ++block_row)
                output[row_begin + block_row] = work[block_row];
        }

        ++apply_calls_;
        apply_seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - apply_start).count();
    }

    void CellBlockJacobiPreconditioner::ApplyCSRRankLocalILU0(
        std::vector<double> const& input,
        std::vector<double>& output,
        std::vector<size_t> const& row_offsets,
        std::vector<matrix_index_t> const& columns)
    {
        auto const apply_start = std::chrono::steady_clock::now();
        if(kind_ != PreconditionerKind::RankLocalILU0 ||
           !requested_kind_supported_ || input.size() != unknown_count_ ||
           row_offsets.size() != unknown_count_ + 1 || row_offsets.empty() ||
           row_offsets.front() != 0 || row_offsets.back() != columns.size() ||
           ilu0_factors_.size() != columns.size() ||
           ilu0_diagonal_positions_.size() != unknown_count_)
            throw UniversalError("Invalid rank-local ILU(0) application");

        bool const aliases = &input == &output;
        if(!aliases)
            output.resize(unknown_count_);
        for(size_t row = 0; row < unknown_count_; ++row)
        {
            double value = input[row];
            for(size_t entry = row_offsets[row];
                entry < row_offsets[row + 1]; ++entry)
            {
                size_t const column = columns[entry];
                if(column < row)
                    value -= ilu0_factors_[entry] * output[column];
            }
            output[row] = value;
        }
        for(size_t row = unknown_count_; row-- > 0;)
        {
            double value = output[row];
            for(size_t entry = row_offsets[row];
                entry < row_offsets[row + 1]; ++entry)
            {
                size_t const column = columns[entry];
                if(column > row && column < unknown_count_)
                    value -= ilu0_factors_[entry] * output[column];
            }
            output[row] = value /
                ilu0_factors_[ilu0_diagonal_positions_[row]];
        }

        ++apply_calls_;
        apply_seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - apply_start).count();
    }

    void CellBlockJacobiPreconditioner::DowngradeToCellBlockJacobi()
    {
        if(requested_kind_ == PreconditionerKind::CellBlockGaussSeidel)
            kind_ = PreconditionerKind::CellBlockJacobi;
    }

    void ApplyCellBlockJacobiCorrectionSweep(
        CellBlockJacobiPreconditioner& preconditioner,
        std::vector<double> const& input,
        std::vector<double> const& matrix_product,
        std::vector<double>& preconditioned,
        std::vector<double>& residual_work)
    {
        if(!UsesCellBlockNeighborCorrection(preconditioner.Kind()) ||
           input.size() != matrix_product.size() ||
           input.size() != preconditioned.size())
            throw UniversalError(
                "Invalid neighbor-corrected cell-block preconditioner application");

        residual_work.resize(input.size());
        for(size_t i = 0; i < input.size(); ++i)
            residual_work[i] = input[i] - matrix_product[i];
        preconditioner.Apply(residual_work, residual_work);
        for(size_t i = 0; i < input.size(); ++i)
            preconditioned[i] +=
                cell_block_neighbor_correction_damping * residual_work[i];
    }

    void ApplyCellBlockJacobiSecondSweep(
        CellBlockJacobiPreconditioner& preconditioner,
        std::vector<double> const& input,
        std::vector<double> const& first_matrix_product,
        std::vector<double>& first_preconditioned,
        std::vector<double>& residual_work)
    {
        ApplyCellBlockJacobiCorrectionSweep(
            preconditioner, input, first_matrix_product,
            first_preconditioned, residual_work);
    }

    size_t CellBlockJacobiPreconditioner::StorageBytes() const
    {
        return scalar_inverse_diagonal_.capacity() * sizeof(double) +
            lu_factors_.capacity() * sizeof(double) +
            inverse_row_scales_.capacity() * sizeof(double) +
            pivots_.capacity() * sizeof(size_t) +
            factorized_blocks_.capacity() * sizeof(std::uint8_t) +
            apply_scratch_.capacity() * sizeof(double) +
            ilu0_factors_.capacity() * sizeof(double) +
            ilu0_diagonal_positions_.capacity() * sizeof(size_t);
    }

    void CellBlockJacobiPreconditioner::Release()
    {
        std::vector<double>().swap(scalar_inverse_diagonal_);
        std::vector<double>().swap(lu_factors_);
        std::vector<double>().swap(inverse_row_scales_);
        std::vector<size_t>().swap(pivots_);
        std::vector<std::uint8_t>().swap(factorized_blocks_);
        std::vector<double>().swap(apply_scratch_);
        std::vector<double>().swap(ilu0_factors_);
        std::vector<size_t>().swap(ilu0_diagonal_positions_);
        kind_ = PreconditionerKind::ScalarJacobi;
        requested_kind_ = PreconditionerKind::ScalarJacobi;
        requested_kind_supported_ = true;
        local_lower_coupling_count_ = 0;
        ignored_local_upper_coupling_count_ = 0;
        ignored_remote_coupling_count_ = 0;
    }

    static unsigned long long bicgstab_diagnostic_cell_id(
        size_t const unknown_location,
        size_t const slice,
        int const owner_rank,
        int const rank,
        std::vector<ComputationalCell3D> const& cells)
    {
        unsigned long long cell_id = std::numeric_limits<unsigned long long>::max();
        size_t const cell_index = slice > 0 ? unknown_location / slice : max_size_t;
        if(rank == owner_rank && cell_index < cells.size())
            cell_id = static_cast<unsigned long long>(cells[cell_index].ID);
#ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Bcast(&cell_id, 1, MPI_UNSIGNED_LONG_LONG, owner_rank,
                      MPI_COMM_WORLD),
            "MPI_Bcast(BiCGSTAB diagnostic cell)");
#endif
        return cell_id;
    }

    struct GlobalTrueResidualAssessment
    {
        double eta_inf = std::numeric_limits<double>::quiet_NaN();
        size_t representative_unknown = max_size_t;
        int representative_rank = -1;
        double representative_residual =
            std::numeric_limits<double>::quiet_NaN();
        double representative_scale =
            std::numeric_limits<double>::quiet_NaN();
        double maximum_scale = 0;
        double safe_minimum_scale = std::numeric_limits<double>::min();
        size_t maximum_row_nonzeros = 0;
        bool finite = false;
    };

    class LongDoubleAccumulator
    {
    public:
        void Add(long double const value)
        {
            long double const next = sum_ + value;
            if(std::abs(sum_) >= std::abs(value))
                correction_ += (sum_ - next) + value;
            else
                correction_ += (value - next) + sum_;
            sum_ = next;
        }

        long double Value() const { return sum_ + correction_; }

    private:
        long double sum_ = 0;
        long double correction_ = 0;
    };

    static bool compute_global_true_residual(
        std::vector<size_t> const& row_ptr,
        std::vector<size_t> const& column_indices,
        std::vector<double> const& matrix_values,
        Fixed16BlockStencilMatrix const* const fixed16_block_stencil,
        std::vector<double> const& solution,
        std::vector<double> const& rhs,
        size_t const local_rows,
        size_t const runtime_group_count,
        std::vector<double>& residual,
        GlobalTrueResidualAssessment& assessment,
        double* const reduction_seconds,
        bool const concurrent_reductions,
        unsigned long long* const collective_calls,
        unsigned long long* const causal_rounds)
    {
#ifndef RICH_MPI
        (void)concurrent_reductions;
        (void)collective_calls;
        (void)causal_rounds;
#endif
        bool const use_fixed16_block_stencil =
            fixed16_block_stencil != nullptr;
        bool valid = runtime_group_count > 0 && rhs.size() >= local_rows;
        if(use_fixed16_block_stencil)
            valid = valid && runtime_group_count ==
                    Fixed16BlockStencilMatrix::BlockSize &&
                fixed16_block_stencil->LocalCellCount <=
                    max_size_t / Fixed16BlockStencilMatrix::BlockSize &&
                fixed16_block_stencil->LocalCellCount *
                    Fixed16BlockStencilMatrix::BlockSize == local_rows;
        else
            valid = valid && row_ptr.size() == local_rows + 1 &&
                column_indices.size() == matrix_values.size();
        residual.assign(local_rows, 0);
        std::vector<double> row_scales(local_rows, 0);
        std::vector<double> maximum_group_scales(runtime_group_count, 0);
        assessment = GlobalTrueResidualAssessment();
        assessment.finite = true;

        for(size_t row = 0; row < local_rows; ++row)
        {
            if(!use_fixed16_block_stencil &&
               (row + 1 >= row_ptr.size() ||
                row_ptr[row] > row_ptr[row + 1] ||
                row_ptr[row + 1] > matrix_values.size()))
            {
                valid = false;
                assessment.finite = false;
                continue;
            }
            LongDoubleAccumulator row_residual;
            LongDoubleAccumulator row_scale;
            row_residual.Add(static_cast<long double>(rhs[row]));
            row_scale.Add(std::abs(static_cast<long double>(rhs[row])));
            size_t row_nonzeros = 0;
            auto const accumulate_entry = [&](double const value,
                                               size_t const column)
            {
                if(column >= solution.size())
                {
                    valid = false;
                    assessment.finite = false;
                    return;
                }
                double const x = solution[column];
                if(!std::isfinite(value) || !std::isfinite(x))
                    assessment.finite = false;
                row_residual.Add(-static_cast<long double>(value) *
                                 static_cast<long double>(x));
                row_scale.Add(std::abs(static_cast<long double>(value)) *
                              std::abs(static_cast<long double>(x)));
                ++row_nonzeros;
            };
            if(use_fixed16_block_stencil)
            {
                std::size_t constexpr block_size =
                    Fixed16BlockStencilMatrix::BlockSize;
                std::size_t const cell = row / block_size;
                std::size_t const row_group = row % block_size;
                std::size_t const cell_base = cell * block_size;
                std::size_t const block_begin = row * block_size;
                for(std::size_t slot = 0; slot < block_size; ++slot)
                    accumulate_entry(
                        fixed16_block_stencil->LocalBlockValues[
                            block_begin + slot],
                        cell_base +
                            Fixed16OrderedBlockColumn(row_group, slot));
                for(std::size_t neighbor =
                        fixed16_block_stencil->NeighborOffsets[cell];
                    neighbor <
                        fixed16_block_stencil->NeighborOffsets[cell + 1];
                    ++neighbor)
                    accumulate_entry(
                        fixed16_block_stencil->NeighborValues[
                            neighbor * block_size + row_group],
                        fixed16_block_stencil->NeighborCells[neighbor] *
                            block_size + row_group);
            }
            else
                for(size_t entry = row_ptr[row];
                    entry < row_ptr[row + 1]; ++entry)
                    accumulate_entry(
                        matrix_values[entry], column_indices[entry]);
            long double const residual_value = row_residual.Value();
            long double const scale_value = row_scale.Value();
            residual[row] = static_cast<double>(residual_value);
            row_scales[row] = static_cast<double>(scale_value);
            bool const row_finite = std::isfinite(residual[row]) &&
                std::isfinite(row_scales[row]) && row_scales[row] >= 0;
            assessment.finite = assessment.finite && row_finite;
            if(row_finite)
                maximum_group_scales[row % runtime_group_count] = std::max(
                    maximum_group_scales[row % runtime_group_count],
                    row_scales[row]);
            assessment.maximum_row_nonzeros = std::max(
                assessment.maximum_row_nonzeros, row_nonzeros);
        }

        int globally_valid = valid ? 1 : 0;
        int globally_finite = assessment.finite ? 1 : 0;
        unsigned long long maximum_row_nonzeros =
            static_cast<unsigned long long>(assessment.maximum_row_nonzeros);
        unsigned long long global_rows =
            static_cast<unsigned long long>(local_rows);
        auto const reduction_start = std::chrono::steady_clock::now();
#ifdef RICH_MPI
        if(concurrent_reductions)
        {
            int validity[2] = {globally_valid, globally_finite};
            MPI_Request requests[4];
            RequireCGMpiSuccess(
                MPI_Iallreduce(MPI_IN_PLACE, validity, 2, MPI_INT, MPI_MIN,
                               MPI_COMM_WORLD, &requests[0]),
                "MPI_Iallreduce(true residual validity)");
            RequireCGMpiSuccess(
                MPI_Iallreduce(MPI_IN_PLACE, &maximum_row_nonzeros, 1,
                               MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                               MPI_COMM_WORLD, &requests[1]),
                "MPI_Iallreduce(true residual row nonzeros)");
            RequireCGMpiSuccess(
                MPI_Iallreduce(MPI_IN_PLACE, &global_rows, 1,
                               MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                               MPI_COMM_WORLD, &requests[2]),
                "MPI_Iallreduce(true residual rows)");
            RequireCGMpiSuccess(
                MPI_Iallreduce(MPI_IN_PLACE, maximum_group_scales.data(),
                               static_cast<int>(
                                   maximum_group_scales.size()),
                               MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD,
                               &requests[3]),
                "MPI_Iallreduce(true residual group scales)");
            RequireCGMpiSuccess(
                MPI_Waitall(4, requests, MPI_STATUSES_IGNORE),
                "MPI_Waitall(true residual initial reductions)");
            globally_valid = validity[0];
            globally_finite = validity[1];
            RecordReductionBatch(
                collective_calls, causal_rounds, 4, 1);
        }
        else
        {
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &globally_valid, 1, MPI_INT,
                              MPI_MIN, MPI_COMM_WORLD),
                "MPI_Allreduce(true residual validity)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &globally_finite, 1, MPI_INT,
                              MPI_MIN, MPI_COMM_WORLD),
                "MPI_Allreduce(true residual finiteness)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &maximum_row_nonzeros, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(true residual row nonzeros)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &global_rows, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(true residual rows)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, maximum_group_scales.data(),
                              static_cast<int>(
                                  maximum_group_scales.size()),
                              MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
                "MPI_Allreduce(true residual group scales)");
            RecordReductionBatch(
                collective_calls, causal_rounds, 5, 5);
        }
#endif
        assessment.maximum_row_nonzeros =
            static_cast<size_t>(maximum_row_nonzeros);

        int rank = 0;
#ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Comm_rank(MPI_COMM_WORLD, &rank),
            "MPI_Comm_rank(true residual assessment)");
#endif
        struct DoubleRank
        {
            double value;
            int rank;
        } maximum_eta = {local_rows > 0 ? 0.0 : -1.0, rank};
        assessment.representative_unknown = max_size_t;
        for(size_t row = 0; row < local_rows; ++row)
        {
            if(!std::isfinite(residual[row]) ||
               !std::isfinite(row_scales[row]))
                continue;
            double const maximum_scale =
                maximum_group_scales[row % runtime_group_count];
            double const safe_minimum_scale = std::max(
                std::numeric_limits<double>::min(),
                32 * std::numeric_limits<double>::epsilon() *
                    maximum_scale);
            double const row_eta = std::abs(residual[row]) /
                std::max(row_scales[row], safe_minimum_scale);
            if(row_eta > maximum_eta.value ||
               assessment.representative_unknown == max_size_t)
            {
                maximum_eta.value = row_eta;
                assessment.representative_unknown = row;
                assessment.representative_residual = residual[row];
                assessment.representative_scale = row_scales[row];
                assessment.maximum_scale = maximum_scale;
                assessment.safe_minimum_scale = safe_minimum_scale;
            }
        }
#ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &maximum_eta, 1, MPI_DOUBLE_INT,
                          MPI_MAXLOC, MPI_COMM_WORLD),
            "MPI_Allreduce(true residual representative)");
        RecordReductionBatch(collective_calls, causal_rounds, 1, 1);
#endif
        assessment.eta_inf = global_rows == 0 ? 0.0 :
            std::max(0.0, maximum_eta.value);
        assessment.representative_rank = global_rows == 0 ? -1 :
            maximum_eta.rank;
        if(global_rows > 0)
        {
            unsigned long long representative_unknown =
                rank == maximum_eta.rank ?
                static_cast<unsigned long long>(
                    assessment.representative_unknown) :
                static_cast<unsigned long long>(max_size_t);
            double representative_values[4] = {
                assessment.representative_residual,
                assessment.representative_scale,
                assessment.maximum_scale,
                assessment.safe_minimum_scale};
#ifdef RICH_MPI
            if(concurrent_reductions)
            {
                MPI_Request requests[2];
                RequireCGMpiSuccess(
                    MPI_Ibcast(&representative_unknown, 1,
                               MPI_UNSIGNED_LONG_LONG, maximum_eta.rank,
                               MPI_COMM_WORLD, &requests[0]),
                    "MPI_Ibcast(true residual representative unknown)");
                RequireCGMpiSuccess(
                    MPI_Ibcast(representative_values, 4, MPI_DOUBLE,
                               maximum_eta.rank, MPI_COMM_WORLD,
                               &requests[1]),
                    "MPI_Ibcast(true residual representative values)");
                RequireCGMpiSuccess(
                    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE),
                    "MPI_Waitall(true residual representative)");
                RecordReductionBatch(
                    collective_calls, causal_rounds, 2, 1);
            }
            else
            {
                RequireCGMpiSuccess(
                    MPI_Bcast(&representative_unknown, 1,
                              MPI_UNSIGNED_LONG_LONG, maximum_eta.rank,
                              MPI_COMM_WORLD),
                    "MPI_Bcast(true residual representative unknown)");
                RequireCGMpiSuccess(
                    MPI_Bcast(representative_values, 4, MPI_DOUBLE,
                              maximum_eta.rank, MPI_COMM_WORLD),
                    "MPI_Bcast(true residual representative values)");
                RecordReductionBatch(
                    collective_calls, causal_rounds, 2, 2);
            }
#endif
            assessment.representative_unknown =
                static_cast<size_t>(representative_unknown);
            assessment.representative_residual = representative_values[0];
            assessment.representative_scale = representative_values[1];
            assessment.maximum_scale = representative_values[2];
            assessment.safe_minimum_scale = representative_values[3];
        }
        assessment.finite = globally_finite != 0 &&
            std::isfinite(assessment.eta_inf);
        if(reduction_seconds != nullptr)
            *reduction_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - reduction_start).count();
        return globally_valid != 0;
    }

    static void print_bicgstab_convergence(
        char const* const outcome,
        size_t const slice,
        size_t const max_loc0,
        size_t const max_loc1,
        int const rank,
        size_t const iterations,
        double const error,
        double const max0,
        double const max1,
        double const negative_value,
        int const max0_rank,
        int const max1_rank,
        std::vector<ComputationalCell3D> const& cells)
    {
        unsigned long long const missing_id =
            std::numeric_limits<unsigned long long>::max();
        unsigned long long const max0_cell_id = bicgstab_diagnostic_cell_id(
            max_loc0, slice, max0_rank, rank, cells);
        unsigned long long const max1_cell_id = bicgstab_diagnostic_cell_id(
            max_loc1, slice, max1_rank, rank, cells);
        if(rank != 0)
            return;

        std::clog << "MG_BICGSTAB_CONVERGENCE outcome=" << outcome
                  << " iterations=" << iterations
                  << " error=" << error
                  << " max0=" << max0
                  << " max1=" << max1
                  << " negative=" << negative_value
                  << " max0_rank=" << max0_rank
                  << " max1_rank=" << max1_rank
                  << " max0_cell_id=";
        if(max0_cell_id == missing_id)
            std::clog << "none";
        else
            std::clog << max0_cell_id;
        std::clog << " max1_cell_id=";
        if(max1_cell_id == missing_id)
            std::clog << "none";
        else
            std::clog << max1_cell_id;
        std::clog << std::endl;
    }

    static unsigned long long bicgstab_diagnostic_group(
        size_t const unknown_location,
        size_t const slice,
        int const owner_rank,
        int const rank)
    {
        unsigned long long group = std::numeric_limits<unsigned long long>::max();
        if(owner_rank < 0)
            return group;
        if(rank == owner_rank && unknown_location != max_size_t && slice > 0)
            group = static_cast<unsigned long long>(unknown_location % slice);
#ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Bcast(&group, 1, MPI_UNSIGNED_LONG_LONG, owner_rank,
                      MPI_COMM_WORLD),
            "MPI_Bcast(BiCGSTAB diagnostic group)");
#endif
        return group;
    }

    static void print_historical_bicgstab_convergence(
        char const* const outcome,
        char const* const reason,
        size_t const slice,
        int const rank,
        size_t const iterations,
        HistoricalMGMetrics const& metrics,
        GlobalTrueResidualAssessment const& diagnostic,
        size_t const last_true_eta_iteration,
        double const pre_correction_eta_inf,
        std::vector<ComputationalCell3D> const& cells)
    {
        unsigned long long const missing =
            std::numeric_limits<unsigned long long>::max();
        unsigned long long const max0_cell_id = bicgstab_diagnostic_cell_id(
            metrics.max0_unknown, slice, metrics.max0_rank, rank, cells);
        unsigned long long const max1_cell_id = bicgstab_diagnostic_cell_id(
            metrics.max1_unknown, slice, metrics.max1_rank, rank, cells);
        unsigned long long const negative_cell_id = metrics.negative ?
            bicgstab_diagnostic_cell_id(metrics.negative_unknown, slice,
                metrics.negative_rank, rank, cells) : missing;
        unsigned long long const eta_cell_id =
            diagnostic.representative_rank >= 0 ?
            bicgstab_diagnostic_cell_id(diagnostic.representative_unknown,
                slice, diagnostic.representative_rank, rank, cells) :
            missing;
        unsigned long long const max0_group = bicgstab_diagnostic_group(
            metrics.max0_unknown, slice, metrics.max0_rank, rank);
        unsigned long long const max1_group = bicgstab_diagnostic_group(
            metrics.max1_unknown, slice, metrics.max1_rank, rank);
        unsigned long long const negative_group = metrics.negative ?
            bicgstab_diagnostic_group(metrics.negative_unknown, slice,
                metrics.negative_rank, rank) : missing;
        unsigned long long const eta_group =
            diagnostic.representative_rank >= 0 ?
            bicgstab_diagnostic_group(diagnostic.representative_unknown,
                slice, diagnostic.representative_rank, rank) : missing;
        if(rank != 0)
            return;
        size_t const eta_age = iterations >= last_true_eta_iteration ?
            iterations - last_true_eta_iteration : 0;
        std::clog << "MG_BICGSTAB_CONVERGENCE scope=global"
                  << " outcome=" << outcome
                  << " reason=" << reason
                  << " iterations=" << iterations
                  << " error=" << metrics.historical_error
                  << " weighted_residual_squared="
                  << metrics.weighted_residual_squared
                  << " weighted_rhs_squared="
                  << metrics.weighted_rhs_squared
                  << " max0=" << metrics.max0
                  << " max1=" << metrics.max1
                  << " negative=" << metrics.negative
                  << " max0_rank=" << metrics.max0_rank
                  << " max1_rank=" << metrics.max1_rank
                  << " negative_rank=" << metrics.negative_rank
                  << " max0_cell_id=";
        if(max0_cell_id == missing)
            std::clog << "none";
        else
            std::clog << max0_cell_id;
        std::clog << " max0_group=";
        if(max0_group == missing)
            std::clog << "none";
        else
            std::clog << max0_group;
        std::clog << " max1_cell_id=";
        if(max1_cell_id == missing)
            std::clog << "none";
        else
            std::clog << max1_cell_id;
        std::clog << " max1_group=";
        if(max1_group == missing)
            std::clog << "none";
        else
            std::clog << max1_group;
        std::clog << " negative_cell_id=";
        if(negative_cell_id == missing)
            std::clog << "none";
        else
            std::clog << negative_cell_id;
        std::clog << " negative_group=";
        if(negative_group == missing)
            std::clog << "none";
        else
            std::clog << negative_group;
        std::clog << " last_true_eta_inf=" << diagnostic.eta_inf
                  << " last_true_eta_iteration=" << last_true_eta_iteration
                  << " last_true_eta_age=" << eta_age
                  << " true_residual_rank="
                  << diagnostic.representative_rank
                  << " true_residual_cell_id=";
        if(eta_cell_id == missing)
            std::clog << "none";
        else
            std::clog << eta_cell_id;
        std::clog << " true_residual_group=";
        if(eta_group == missing)
            std::clog << "none";
        else
            std::clog << eta_group;
        std::clog << " true_residual="
                  << diagnostic.representative_residual
                  << " true_scale=" << diagnostic.representative_scale
                  << " maximum_scale=" << diagnostic.maximum_scale
                  << " safe_minimum_scale="
                  << diagnostic.safe_minimum_scale
                  << " max_global_row_nnz="
                  << diagnostic.maximum_row_nonzeros
                  << " pre_correction_eta_inf="
                  << pre_correction_eta_inf
                  << " final_eta_inf=not_evaluated"
                  << " eta_inf_role=diagnostic_only" << std::endl;
    }

    struct GlobalHistoricalCorrectionCandidate
    {
        HistoricalMGCorrectionAssessment assessment;
        std::vector<double> corrected_solution;
        std::vector<double> capped_corrected_solution;
        double pre_negative_extent = 0;
        double pre_positive_extent = 0;
        double post_positive_extent = 0;
        double absolute_residual_extent = 0;
        std::uint64_t pre_correction_negative_group_count = 0;
        std::uint64_t introduced_negative_group_count = 0;
        double introduced_negative_extent = 0;
        std::uint64_t amplified_negative_group_count = 0;
        double amplified_negative_extent = 0;
        HistoricalMGResidualCorrectionDiagnostics correction_diagnostics;
        HistoricalMGCorrectedNegativity corrected_negativity;
        HistoricalMGPositiveFloorAssessment positive_floor;
        HistoricalMGCorrectionSpectralFailure spectral_failure;
    };

    enum class HistoricalMGBreakdownResolution
    {
        Converged,
        Restart,
        Rejected
    };

    static bool build_historical_final_correction(
        size_t const slice,
        size_t const iterations,
        GlobalTrueResidualAssessment const& diagnostic,
        size_t const last_true_eta_iteration,
        std::vector<double> const& true_residual,
        std::vector<double> const& sub_x,
        Tessellation3D const& tess,
        size_t const local_rows,
        double const lengthscale,
        std::vector<ComputationalCell3D> const& cells,
        GlobalHistoricalCorrectionCandidate& candidate)
    {
        bool locally_valid = diagnostic.finite &&
            last_true_eta_iteration == iterations &&
            slice > 0 && local_rows % slice == 0 &&
            cells.size() >= local_rows / slice &&
            true_residual.size() == local_rows && sub_x.size() == local_rows;
        for(size_t row = 0; locally_valid && row < local_rows; ++row)
        {
            double const volume = tess.GetVolume(row / slice) *
                pow<3>(lengthscale);
            locally_valid = std::isfinite(volume) && volume > 0 &&
                std::isfinite(sub_x[row]) &&
                std::isfinite(true_residual[row]) &&
                std::isfinite(sub_x[row] + true_residual[row] / volume);
        }
        int globally_valid = locally_valid ? 1 : 0;
#ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &globally_valid, 1, MPI_INT,
                          MPI_MIN, MPI_COMM_WORLD),
            "MPI_Allreduce(final correction validity)");
#endif
        if(globally_valid == 0)
            return false;

        candidate = GlobalHistoricalCorrectionCandidate{};
        ResetHistoricalMGResidualCorrectionDiagnostics(
            candidate.correction_diagnostics, slice, local_rows);
        candidate.corrected_solution.resize(local_rows);
        std::vector<double> correction_volume(local_rows, 0);
        for(size_t cell = 0; cell < local_rows / slice; ++cell) {
            double const volume = tess.GetVolume(cell) * pow<3>(lengthscale);
            for(size_t group = 0; group < slice; ++group) {
                size_t const row = cell * slice + group;
                correction_volume[row] = volume;
                double const before_extent = sub_x[row] * volume;
                double const unscaled_after_extent =
                    before_extent + true_residual[row];
                HistoricalMGResidualCorrectionLimit const limit =
                    DetermineHistoricalMGResidualCorrectionLimit(
                        before_extent, unscaled_after_extent,
                        cells[cell].ID, group);
                RecordHistoricalMGResidualCorrectionLimit(
                    limit, candidate.correction_diagnostics);
                candidate.correction_diagnostics.correction_scale[row] =
                    limit.Scale;
                candidate.correction_diagnostics.pre_correction_solution[row] =
                    sub_x[row];
                candidate.corrected_solution[row] = sub_x[row] +
                    limit.Scale * true_residual[row] / volume;
            }
        }
#ifdef RICH_MPI
        CollectHistoricalMGResidualCorrectionLimitingDiagnostic(
            candidate.correction_diagnostics);
#endif
        std::vector<std::size_t> cell_ids(local_rows / slice, max_size_t);
        for(size_t cell = 0; cell < cell_ids.size(); ++cell)
            cell_ids[cell] = cells[cell].ID;
        candidate.corrected_negativity =
            AssessHistoricalMGCorrectedNegativity(
                sub_x, candidate.corrected_solution,
                candidate.correction_diagnostics.correction_scale, slice,
                cell_ids, 0, true);
        if(!candidate.corrected_negativity.Finite)
            return false;
        if(!candidate.correction_diagnostics.finite)
            return false;

        double correction_diagnostics[7] = {0, 0, 0, 0, 0, 0, 0};
        std::uint64_t correction_counts[4] = {0, 0, 0, 0};
        for(size_t row = 0; row < local_rows; ++row)
        {
            double const volume = tess.GetVolume(row / slice) *
                pow<3>(lengthscale);
            if(sub_x[row] < 0) {
                ++correction_counts[1];
                correction_diagnostics[0] -= volume * sub_x[row];
            }
            else
                correction_diagnostics[1] += volume * sub_x[row];
            if(candidate.corrected_solution[row] < 0) {
                ++correction_counts[0];
                correction_diagnostics[2] -=
                    volume * candidate.corrected_solution[row];
                if(sub_x[row] >= 0) {
                    ++correction_counts[2];
                    correction_diagnostics[5] -=
                        volume * candidate.corrected_solution[row];
                }
                else if(candidate.corrected_solution[row] < sub_x[row]) {
                    ++correction_counts[3];
                    correction_diagnostics[6] += volume *
                        (sub_x[row] - candidate.corrected_solution[row]);
                }
            }
            else
                correction_diagnostics[3] +=
                    volume * candidate.corrected_solution[row];
            correction_diagnostics[4] += std::abs(true_residual[row]);
        }
#ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, correction_diagnostics, 7,
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
            "MPI_Allreduce(final correction diagnostics)");
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, correction_counts, 4, MPI_UINT64_T,
                          MPI_SUM, MPI_COMM_WORLD),
            "MPI_Allreduce(final correction counts)");
#endif
        if(!CollectHistoricalMGCorrectedNegativityGlobalExtents(
               candidate.corrected_negativity, correction_diagnostics[2],
               correction_diagnostics[3], false))
            return false;
        candidate.capped_corrected_solution = candidate.corrected_solution;
        candidate.positive_floor = AssessAndApplyHistoricalMGPositiveFloor(
            sub_x, candidate.corrected_solution,
            candidate.correction_diagnostics.correction_scale,
            correction_volume, slice, cell_ids, 0, 0, true);
        RecordHistoricalMGPositiveFloor(
            candidate.positive_floor, candidate.correction_diagnostics);
        candidate.spectral_failure = HistoricalMGPositiveFloorFailure(
            candidate.positive_floor, candidate.corrected_negativity);
        candidate.assessment.finite = true;
        candidate.assessment.negative_group_count = correction_counts[0];
        candidate.assessment.negative_extent = correction_diagnostics[2];
        candidate.pre_negative_extent = correction_diagnostics[0];
        candidate.pre_positive_extent = correction_diagnostics[1];
        candidate.post_positive_extent = correction_diagnostics[3];
        candidate.absolute_residual_extent = correction_diagnostics[4];
        candidate.pre_correction_negative_group_count = correction_counts[1];
        candidate.introduced_negative_group_count = correction_counts[2];
        candidate.introduced_negative_extent = correction_diagnostics[5];
        candidate.amplified_negative_group_count = correction_counts[3];
        candidate.amplified_negative_extent = correction_diagnostics[6];
        return true;
    }
    static void commit_historical_final_correction(
        size_t const slice,
        int const rank,
        size_t const iterations,
        HistoricalMGBranch const branch,
        HistoricalMGMetrics const& metrics,
        GlobalTrueResidualAssessment const& diagnostic,
        size_t const last_true_eta_iteration,
        GlobalHistoricalCorrectionCandidate&& candidate,
        std::vector<double>& sub_x_solution,
        std::vector<ComputationalCell3D> const& cells,
        int& total_iters)
    {
        sub_x_solution = std::move(candidate.corrected_solution);
        total_iters = static_cast<int>(std::min<size_t>(
            iterations, static_cast<size_t>(std::numeric_limits<int>::max())));
        print_historical_bicgstab_convergence(
            "converged", HistoricalMGBranchLabel(branch), slice, rank,
            iterations, metrics, diagnostic, last_true_eta_iteration,
            diagnostic.eta_inf, cells);
        if(rank == 0 && candidate.assessment.negative_extent > 0 &&
           !candidate.positive_floor.Applied) {
            std::clog << "MG_HISTORICAL_FINAL_CORRECTION"
                      << " pre_negative_extent="
                      << candidate.pre_negative_extent
                      << " pre_positive_extent="
                      << candidate.pre_positive_extent
                      << " post_negative_extent="
                      << candidate.assessment.negative_extent
                      << " post_positive_extent="
                      << candidate.post_positive_extent
                      << " absolute_residual_extent="
                      << candidate.absolute_residual_extent
                      << " pre_correction_negative_groups="
                      << candidate.pre_correction_negative_group_count
                      << " residual_correction_introduced_negative="
                      << (candidate.introduced_negative_group_count > 0 ? 1 : 0)
                      << " introduced_negative_groups="
                      << candidate.introduced_negative_group_count
                      << " introduced_negative_extent="
                      << candidate.introduced_negative_extent
                      << " residual_correction_amplified_negative="
                      << (candidate.amplified_negative_group_count > 0 ? 1 : 0)
                      << " amplified_negative_groups="
                      << candidate.amplified_negative_group_count
                      << " amplified_negative_extent="
                      << candidate.amplified_negative_extent << std::endl;
            std::clog << "Negative raw radiation candidate energy "
                      << candidate.assessment.negative_extent << std::endl;
        }
    }

    // Matrix times vector
    void mat_times_vec(const mat &sub_A_values, const size_t_mat &sub_A_indices, const std::vector<double> &v, 
        std::vector<double> &result)
    {

        // NOTE: when using MPI with > 1 proc, A will be only a sub-matrix (a subset of rows) of the full matrix
        // since we are 1D decomposing the matrix by rows

        size_t const sub_num_rows = sub_A_values.size();
        if(sub_num_rows == 0) return;

        size_t const sub_num_cols = sub_A_values[0].size();
        result.resize(sub_num_rows, 0);
        double dot_prod;
        for (size_t i = 0; i < sub_num_rows; i++) {
            dot_prod = 0;  // rezero the dot_prod buffer. we need this buffer so we can make it private to the thread to avoid race conditions.
            for (size_t j = 0; j < sub_num_cols; j++) {
                if(sub_A_indices[i][j] == max_size_t)
                    break;
                dot_prod += sub_A_values[i][j] * v[sub_A_indices[i][j]]; 
            }
            result[i] = dot_prod;
        }
    }

    void build_crs(const mat &sub_A_values, const size_t_mat &sub_A_indices,
        std::vector<size_t> &row_ptr, std::vector<size_t> &col_idx, std::vector<double> &values)
    {
        size_t const sub_num_rows = sub_A_values.size();
        row_ptr.assign(sub_num_rows + 1, 0);
        col_idx.clear();
        values.clear();
        col_idx.reserve(sub_num_rows);
        values.reserve(sub_num_rows);
        for (size_t i = 0; i < sub_num_rows; ++i) {
            for (size_t j = 0; j < sub_A_values[i].size(); ++j) {
                if (sub_A_indices[i][j] == max_size_t)
                    break;
                col_idx.push_back(sub_A_indices[i][j]);
                values.push_back(sub_A_values[i][j]);
            }
            row_ptr[i + 1] = col_idx.size();
        }
    }

    void mat_times_vec_crs(const std::vector<size_t> &row_ptr, const std::vector<size_t> &col_idx,
        const std::vector<double> &values, const std::vector<double> &v, std::vector<double> &result)
    {
        size_t const sub_num_rows = row_ptr.size() > 0 ? row_ptr.size() - 1 : 0;
        if(sub_num_rows == 0) return;
        result.resize(sub_num_rows, 0);
        const double* __restrict__ val_ptr = values.data();
        const size_t* __restrict__ col_ptr = col_idx.data();
        const double* __restrict__ v_ptr = v.data();
        double* __restrict__ res_ptr = result.data();
        const size_t* __restrict__ rp = row_ptr.data();
        for (size_t i = 0; i < sub_num_rows; ++i) {
            double dot_prod = 0.0;
            for (size_t k = rp[i]; k < rp[i + 1]; ++k)
                dot_prod += val_ptr[k] * v_ptr[col_ptr[k]];
            res_ptr[i] = dot_prod;
        }
    }

    void mat_times_vec_fixed16_block(
        Fixed16BlockMatvecSchedule const& schedule,
        std::vector<double> const& values,
        std::vector<double> const& v,
        std::vector<double>& result)
    {
        if(schedule.RowCount == 0)
            return;
        result.resize(schedule.RowCount, 0);
        std::size_t const* __restrict__ const row_offsets =
            schedule.RowOffsets->data();
        std::size_t const* __restrict__ const column_indices =
            schedule.ColumnIndices->data();
        double const* __restrict__ const matrix_values = values.data();
        double const* __restrict__ const input = v.data();
        double* __restrict__ const output = result.data();

        for(std::size_t cell = 0; cell < schedule.CellCount; ++cell)
        {
            std::size_t const cell_base = cell * Fixed16BlockSize;
            for(std::size_t row_group = 0;
                row_group < Fixed16BlockSize; ++row_group)
            {
                std::size_t const row = cell_base + row_group;
                std::size_t entry = row_offsets[row];
                double dot_product = 0.0;

                // The schedule validator proved the exact assembly order:
                // diagonal first, then the other cell-local groups in ascending
                // order.  Keep every multiply-add in that original CSR order.
                dot_product += matrix_values[entry++] *
                    input[cell_base + row_group];
                for(std::size_t block_group = 0;
                    block_group < row_group; ++block_group)
                    dot_product += matrix_values[entry++] *
                        input[cell_base + block_group];
                for(std::size_t block_group = row_group + 1;
                    block_group < Fixed16BlockSize; ++block_group)
                    dot_product += matrix_values[entry++] *
                        input[cell_base + block_group];
                for(; entry < row_offsets[row + 1]; ++entry)
                    dot_product += matrix_values[entry] *
                        input[column_indices[entry]];
                output[row] = dot_product;
            }
        }
    }

    std::size_t CountFixed16BlockMatvecShadowMismatches(
        std::vector<double> const& candidate,
        std::vector<double> const& generic)
    {
        if(candidate.size() != generic.size())
            return std::max(candidate.size(), generic.size());
        std::size_t mismatches = 0;
        for(std::size_t row = 0; row < candidate.size(); ++row)
            if(std::memcmp(&candidate[row], &generic[row],
                           sizeof(candidate[row])) != 0)
                ++mismatches;
        return mismatches;
    }

    std::vector<double> vector_rescale(std::vector<double> const& a, std::vector<double> const& b)
    {
        size_t const N = a.size(); 
        if(a.size() != b.size())
            throw UniversalError("Sizes do not match in vector_rescale");
        std::vector<double> res(N);
        for(size_t i = 0; i < N; ++i)
            res[i] = a[i] * b[i];
        return res;
    }

    void vector_rescale(std::vector<double> const& a, std::vector<double> const& b,
        std::vector<double> &result)
    {
        size_t const N = a.size(); 
        if(a.size() != b.size())
            throw UniversalError("Sizes do not match in vector_rescale");
        result.resize(N);
        const double* a_ptr = a.data();
        const double* __restrict__ b_ptr = b.data();
        double* res_ptr = result.data();
        for(size_t i = 0; i < N; ++i)
            res_ptr[i] = a_ptr[i] * b_ptr[i];
    }

    // Linear combination of vectors; safe when result aliases u or v
    void vec_lin_combo(double a, const std::vector<double> &u, double b, const std::vector<double> &v, 
        std::vector<double> &result)
    {
        if(u.size() != v.size())
            throw UniversalError("Unequal vector sizes in vec_lin_combo");
        size_t n = u.size();
        result.resize(n);
        for (size_t j = 0; j < n; j++)
            result[j] = a * u[j] + b * v[j];
    }

    static double local_dot_product(
        const std::vector<double> &sub_u,
        const std::vector<double> &sub_v)
    {
        if(sub_u.size() != sub_v.size())
            throw UniversalError("Unequal vector sizes in vec_lin_combo");
        size_t length = sub_u.size();

        double sub_prod = 0;
        size_t length4 = length / 4;
        size_t loop_number = length4 * 4;
        for(size_t i = 0; i < loop_number; i += 4)
        {
            Vec4d _u(sub_u[i], sub_u[i+1], sub_u[i+2], sub_u[i+3]);
            Vec4d _v(sub_v[i], sub_v[i+1], sub_v[i+2], sub_v[i+3]);
            Vec4d _uv = _u * _v;
            sub_prod += ((_uv[0] + _uv[1]) + (_uv[2] + _uv[3]));
        }
        for(size_t i = loop_number; i < length; i++)
            sub_prod += sub_u[i] * sub_v[i];
        return sub_prod;
    }

    double mpi_dot_product(const std::vector<double> &sub_u, const std::vector<double> &sub_v) // need to pass it the buffer where to keep the result
    {
        double sub_prod = local_dot_product(sub_u, sub_v);
#ifdef RICH_MPI
        // do a reduction over sub_prod to get the total dot product
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &sub_prod, 1, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(CG dot product)");
#endif
        return sub_prod;
    }


    // performs a reduction over the sub-vectors which are passed to it... All_Reduce broadcasts the value to all procs
    double mpi_dot_product2(const std::vector<double> &sub_u, const std::vector<double> &sub_v) // need to pass it the buffer where to keep the result
    {
        if(sub_u.size() != sub_v.size())
            throw UniversalError("Unequal vector sizes in vec_lin_combo");
        size_t length = sub_u.size();
        const double* __restrict__ u_ptr = sub_u.data();
        const double* __restrict__ v_ptr = sub_v.data();

        double sub_prod = 0.0;
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER) || defined(__GNUC__)
#pragma omp simd reduction(+:sub_prod)
#endif
        for (size_t i = 0; i < length; i++) {
            sub_prod += u_ptr[i] * v_ptr[i];
        }
#ifdef RICH_MPI
        // do a reduction over sub_prod to get the total dot product
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &sub_prod, 1, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(CG dot product2)");
#endif
        return sub_prod;
    }
    
    void finalize_conjugate_gradient(size_t slice, size_t max_loc0, size_t max_loc1, int rank, size_t i,
        double error, double max_data_0_val, double max_data_1_val, double max_data_2_val,
        int max_data_0_id, int max_data_1_id, std::vector<double>& sub_x, std::vector<double>& sub_x_solution,
        std::vector<ComputationalCell3D> const& cells, Tessellation3D const& tess, int &total_iters,
        const std::vector<size_t>& A_row_ptr, const std::vector<size_t>& A_col_idx, const std::vector<double>& A_values,
        std::vector<double> const& b, size_t Nlocal, double lengthscale,
        std::vector<double>& sub_a_times_p, std::vector<double>& sub_r)
    {
        print_bicgstab_convergence(
            "finalized", slice, max_loc0, max_loc1, rank, i, error,
            max_data_0_val, max_data_1_val, max_data_2_val,
            max_data_0_id, max_data_1_id, cells);
        max_loc0 /= slice;
        max_loc1 /= slice;
        if (cells.size() > max_loc0 and cells.size() > max_loc1){
            if(rank == 0)
                std::clog << "Converged at iter = " << i <<" error "<<error<<" negative value "<<max_data_2_val<<std::endl;
        }
        total_iters = i;
#ifdef RICH_MPI
        MPI_exchange_data(tess, sub_x, true, slice);
#endif
        mat_times_vec_crs(A_row_ptr, A_col_idx, A_values, sub_x, sub_a_times_p);
        sub_x.resize(Nlocal);
        vec_lin_combo(1.0, b, -1.0, sub_a_times_p, sub_r);
        sub_x_solution.resize(Nlocal);
        double negative_x = 0;
        for(size_t k = 0; k < Nlocal; ++k)
        {
            // std::cout<<"subx["<<k<<"] "<<sub_x[k]<<std::endl;
            // sub_x_solution[k] = sub_x[k];
            double const cell_volume = tess.GetVolume(k / slice) * pow<3>(lengthscale);
            sub_x_solution[k] = sub_x[k] + sub_r[k] / cell_volume;
            if(sub_x_solution[k] < 0)
            {
                negative_x -= cell_volume * sub_x_solution[k];
            }
        }
        if (rank == 0 && negative_x > 0.0)
            std::clog << "Negative raw radiation candidate energy " << negative_x << std::endl;
    }
    
    void build_M(const mat &sub_A_values, const size_t_mat &sub_A_indices, std::vector<double> &M)
    {
        size_t const sub_num_rows = sub_A_values.size();
        if(sub_num_rows == 0) return;
        int rank = 0;

        size_t const sub_num_cols = sub_A_values[0].size();
        M.resize(sub_num_rows);
        for (size_t i = 0; i < sub_num_rows; i++) {
            for (size_t j = 0; j < sub_num_cols; j++) {
                if(sub_A_indices[i][j] == max_size_t)
                    break;
                if(sub_A_indices[i][j] == i)
                {
                    M[i] = 1.0 / sub_A_values[i][j];
                    // M[i] = 1.0;
                    break;
                }
            }
        }
    }

    void build_M_crs(const std::vector<size_t> &row_ptr, const std::vector<size_t> &col_idx,
        const std::vector<double> &values, std::vector<double> &M)
    {
        size_t const sub_num_rows = row_ptr.size() > 0 ? row_ptr.size() - 1 : 0;
        if(sub_num_rows == 0) return;
        M.resize(sub_num_rows);
        for (size_t i = 0; i < sub_num_rows; ++i) {
            for (size_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                if (col_idx[k] == i) {
                    M[i] = 1.0 / values[k];
                    break;
                }
            }
        }
    }

    static bool abs_compare(double a, double b)
    {
        return (std::abs(a) < std::abs(b));
    }

std::vector<double> conj_grad_solver(const double tolerance, int &total_iters,
        Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells,
        double const dt, MatrixBuilder const& matrix_builder, double const time, std::vector<double> &sub_x_solution)  //total_iters is to store # of iters in it
    {
        size_t Nlocal = tess.GetPointNo();
        
        // NOTE: when using MPI with > 1 proc, A will be only a sub-matrix (a subset of rows) of the full matrix
        // since we are 1D decomposing the matrix by rows
        // b will be the full vector

        int nprocs = 1, rank = 0;
    #ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Comm_size(MPI_COMM_WORLD, &nprocs),
            "MPI_Comm_size(CG)");
        RequireCGMpiSuccess(
            MPI_Comm_rank(MPI_COMM_WORLD, &rank),
            "MPI_Comm_rank(CG)");
    #endif
        int constexpr max_iter = static_cast<int>(
            historical_mg_maximum_iterations);

        mat A;
        size_t_mat A_indeces;
        std::vector<double> b;
        std::vector<double> sub_x; // this is for the initial guess
        matrix_builder.BuildMatrix(tess, A, A_indeces, cells, dt, b, sub_x, time);
        std::vector<size_t> A_row_ptr, A_col_idx;
        std::vector<double> A_values;
        build_crs(A, A_indeces, A_row_ptr, A_col_idx, A_values);
        std::vector<double> M; // The preconditioner
        if(use_crs_matvec)
            build_M_crs(A_row_ptr, A_col_idx, A_values, M);
        else
            build_M(A, A_indeces, M);
        std::vector<double> r_old, sub_a_times_p;
        std::vector<double> sub_r;
#ifdef RICH_MPI
        MPI_exchange_data(tess, sub_x, true);
#endif
        if(use_crs_matvec)
            mat_times_vec_crs(A_row_ptr, A_col_idx, A_values, sub_x, sub_a_times_p);
        else
            mat_times_vec(A, A_indeces, sub_x, sub_a_times_p);
        // Find maximum value of A, this is used for normalization of the error
        double maxA[2] = {0, 0};
        size_t const Na = A.size();
        for(size_t i = 0; i < Na ; ++i)
        {
            maxA[0] = std::max(maxA[0], std::abs(sub_x[i]));
            maxA[1] = std::max(maxA[1], std::abs(b[i]));
        }       
#ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &maxA, 2, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(CG normalization)");
#endif
        vec_lin_combo(1.0, b, -1.0, sub_a_times_p, sub_r);    
        std::vector<double> sub_p(sub_r);
        sub_p.resize(Nlocal);
        sub_x.resize(Nlocal);
        vector_rescale(sub_p, M, sub_p);
        std::vector<double> result2(Nlocal, 0), result3, p(sub_p), old_result2(Nlocal, 0);
        std::vector<double> old_x = sub_x;
        size_t Ntotal = Nlocal;
#ifdef RICH_MPI
        MPI_exchange_data(tess, p, true);
        RequireCGMpiSuccess(
            MPI_Allreduce(&Nlocal, &Ntotal, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(CG row count)");
#endif
        double sub_r_sqrd = mpi_dot_product(sub_r, sub_p);
        double const delta_init = sub_r_sqrd;
        // double sub_r_sqrd_convergence = mpi_dot_product(sub_r, sub_r);
        // if(rank == 0)
        //     std::cout<<"CG init delta "<<delta_init<<std::endl;
        double sub_r_sqrd_old = 0, sub_p_by_ap = 0, alpha = 0, beta = 0;
        bool good_end = false;
        struct
        {
            double val;
            int mpi_id;
        }max_data[3];
        max_data[0].mpi_id = rank;
        max_data[1].mpi_id = rank;
        max_data[2].mpi_id = rank;
        max_data[0].val = 0;
        max_data[1].val = 0;
        max_data[2].val = 0;
        
        size_t max_loc0 = 0, max_loc1 = 0, max_loc2 = 0;
        // Main Conjugate Gradient loop
        // this loop must be serial b/c CG is an iterative method
        for (int i = 0; i < max_iter; i++) {
            // note: make sure matrix is big enough for the number of processors you are using!
            max_data[2].mpi_id = rank;
            max_data[0].mpi_id = rank;
            r_old = sub_r;                 // Store previous residual
            sub_r_sqrd_old = sub_r_sqrd;  // save a recalculation of r_old^2 later

            if(use_crs_matvec)
                mat_times_vec_crs(A_row_ptr, A_col_idx, A_values, p, sub_a_times_p);  //split up with MPI and then finer parallelize with openmp
            else
                mat_times_vec(A, A_indeces, p, sub_a_times_p);

            sub_p_by_ap = mpi_dot_product(sub_p, sub_a_times_p);

            alpha = sub_r_sqrd / (sub_p_by_ap + std::numeric_limits<double>::min() * 100);         

            // Next estimate of solution
            vec_lin_combo(1.0, sub_x, alpha, sub_p, sub_x);
            if(i > 1 && i % 50 == 0)
            {
#ifdef RICH_MPI
                MPI_exchange_data(tess, sub_x, true);
#endif
                if(use_crs_matvec)
                    mat_times_vec_crs(A_row_ptr, A_col_idx, A_values, sub_x, sub_a_times_p);
                else
                    mat_times_vec(A, A_indeces, sub_x, sub_a_times_p);
                sub_x.resize(Nlocal);
                vec_lin_combo(1.0, b, -1.0, sub_a_times_p, sub_r);    
            }
            else
                vec_lin_combo(1.0, sub_r, -alpha, sub_a_times_p, sub_r);

            max_data[0].val = 0;
            max_data[1].val = 0;
            max_data[2].val = 0;
            for(size_t j = 0; j < Nlocal; ++j)
            {
                double const local_scale = std::abs(b[j]);
                if(std::abs(sub_r[j]) > max_data[1].val * (std::abs(A[j][0] * (std::abs(sub_x[j]) + std::numeric_limits<double>::min() * 100 + maxA[0] * 1e-9))))
                {
                    max_data[1].val = std::abs(sub_r[j]) / (std::abs(A[j][0] * (std::abs(sub_x[j]) + std::numeric_limits<double>::min() * 100 + maxA[0] * 1e-9)));
                    max_loc1 = j;
                }
                if(std::abs(sub_x[j] - old_x[j]) > max_data[0].val * (std::abs(sub_x[j]) + std::numeric_limits<double>::min() * 100 + maxA[0] * 1e-9))
                {
                     max_data[0].val = std::abs(sub_x[j] - old_x[j]) / (std::abs(sub_x[j]) + std::numeric_limits<double>::min() * 100 + maxA[0] * 1e-9);
                     max_loc0 = j;
                }
                if(sub_x[j] < 0)
                {
                    max_loc2 = j;
                    max_data[2].val = 1;
                }
            }

#ifdef RICH_MPI
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, max_data, 3, MPI_DOUBLE_INT,
                              MPI_MAXLOC, MPI_COMM_WORLD),
                "MPI_Allreduce(CG convergence maxima)");
#endif
            old_result2 = result2;
            vector_rescale(sub_r, M, result2);
            sub_r_sqrd = mpi_dot_product(sub_r, result2);
            // recall that we can't have a 'break' within an openmp parallel region, so end it here then all threads are merged, and the convergence is checked
            // Convergence test
            if (sub_r_sqrd < delta_init * tolerance//std::sqrt(sub_r_sqrd_convergence / Ntotal) < tolerance * maxA [1]
                && max_data[1].val < 1e-5 && max_data[0].val < 1e-5 && (i > 250 || max_data[2].val < 0.5)) { // norm is just sqrt(dot product so don't need to use a separate norm fnc) // vector norm needs to use a all reduce!
                if(rank == 0)
                    std::clog << "Converged at iter = " << i <<" delta "<<sub_r_sqrd<<" negative value "<<max_data[2].val<<std::endl;
                total_iters = i;
                good_end = true;
#ifdef RICH_MPI
                MPI_exchange_data(tess, sub_x, true);
#endif
                sub_x_solution = sub_x;
                if(use_crs_matvec)
                    mat_times_vec_crs(A_row_ptr, A_col_idx, A_values, sub_x, sub_a_times_p);
                else
                    mat_times_vec(A, A_indeces, sub_x, sub_a_times_p);
                sub_x.resize(Nlocal);
                vec_lin_combo(1.0, b, -1.0, sub_a_times_p, sub_r);
                break;
            }
            old_x = sub_x;
            vec_lin_combo(1.0, result2, -1.0, old_result2, result3);   
            double const  Polak_Ribiere = mpi_dot_product(sub_r, result3);
            beta = std::max(0.0, Polak_Ribiere / sub_r_sqrd_old);       
            
            vec_lin_combo(1.0, result2, beta, sub_p, sub_p);
            p = sub_p;
    #ifdef RICH_MPI
            MPI_exchange_data(tess, p, true);
    #endif
        }
        if(not good_end)
        {
            total_iters = max_iter;
            std::vector<double> volumes = tess.GetAllVolumes();
#ifdef RICH_MPI
			MPI_exchange_data(tess, volumes, true);
#endif
            if(rank == 0)
	      std::clog <<"not good end, delta "<<sub_r_sqrd<<" maxdata2 "<<max_data[2].val<<std::endl;
            if(rank == max_data[0].mpi_id)
            {
                std::vector<size_t> neigh = tess.GetNeighbors(max_loc0);
                double min_volume = std::numeric_limits<double>::max(), max_volume = 0;
                for(size_t k = 0; k < neigh.size(); ++k)
                {
                    if(not tess.IsPointOutsideBox(neigh[k]))
                    {
                        min_volume = std::min(min_volume, volumes[neigh[k]]);
                        max_volume = std::max(max_volume, volumes[neigh[k]]);
                    }
                }
            }
            if(rank == max_data[1].mpi_id)
            {
                std::vector<size_t> neigh = tess.GetNeighbors(max_loc1);
                double min_volume = std::numeric_limits<double>::max(), max_volume = 0;
                for(size_t k = 0; k < neigh.size(); ++k)
                {
                    if(not tess.IsPointOutsideBox(neigh[k]))
                    {
                        min_volume = std::min(min_volume, volumes[neigh[k]]);
                        max_volume = std::max(max_volume, volumes[neigh[k]]);
                    }
                }
            }
	    if(rank == max_data[2].mpi_id)
        std::fill_n(sub_x.begin(), sub_x.size(), -1.0);
	    // throw UniversalError("CG did not converge");
        }
#ifdef RICH_MPI
        MPI_exchange_data(tess, sub_x, true);
#endif
        return sub_x;
    }
    
    std::vector<double> BiCGSTAB(const double tolerance, int &total_iters,
        Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells,
        double const dt, MatrixBuilder const& matrix_builder, double const time, std::vector<double> &sub_x_solution, bool &good_end)  //total_iters is to store # of iters in it
    {
        BiCGSTABWorkspace workspace;
        return BiCGSTAB(tolerance, total_iters, tess, cells, dt, matrix_builder, time, sub_x_solution, good_end, workspace);
    }

    std::vector<double> &BiCGSTAB(const double tolerance, int &total_iters,
        Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells,
        double const dt, MatrixBuilder const& matrix_builder, double const time, std::vector<double> &sub_x_solution, bool &good_end, BiCGSTABWorkspace &workspace)  //total_iters is to store # of iters in it
    {
        MEMORY_PROFILE_SCOPE("diffusion BiCGSTAB");
        good_end = false;
        workspace.historical_correction =
            HistoricalMGResidualCorrectionDiagnostics{};
        size_t const slice = matrix_builder.GetUnknownsPerCell();
        if(slice == 0)
            throw UniversalError("Diffusion matrix builder returned zero unknowns per cell");
        size_t const Nlocal = tess.GetPointNo() * slice;
        
        // NOTE: when using MPI with > 1 proc, A will be only a sub-matrix (a subset of rows) of the full matrix
        // since we are 1D decomposing the matrix by rows
        // b will be the full vector

        int nprocs = 1, rank = 0;
    #ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Comm_size(MPI_COMM_WORLD, &nprocs),
            "MPI_Comm_size(global BiCGSTAB)");
        RequireCGMpiSuccess(
            MPI_Comm_rank(MPI_COMM_WORLD, &rank),
            "MPI_Comm_rank(global BiCGSTAB)");
    #endif
        int const max_iter = 10000;

        mat &A = workspace.A;
        size_t_mat &A_indeces = workspace.A_indeces;
        std::vector<double> &b = workspace.b;
        std::vector<double> &sub_x = workspace.sub_x; // this is for the initial guess
        std::vector<size_t> &A_row_ptr = workspace.A_row_ptr;
        std::vector<size_t> &A_col_idx = workspace.A_col_idx;
        std::vector<double> &A_values = workspace.A_values;
        Fixed16BlockStencilMatrix &fixed16_block_stencil =
            workspace.fixed16_block_stencil;
        char const* const direct_csr_value =
            std::getenv("RICH_MG_DIRECT_GLOBAL_CSR");
        bool const direct_csr_requested = direct_csr_value != nullptr &&
            direct_csr_value[0] != '\0' &&
            std::strcmp(direct_csr_value, "0") != 0 &&
            std::strcmp(direct_csr_value, "false") != 0 &&
            std::strcmp(direct_csr_value, "off") != 0 &&
            std::strcmp(direct_csr_value, "no") != 0;
        bool const direct_csr_supported =
            matrix_builder.SupportsDirectCSR();
        char const* const fused_reductions_value =
            std::getenv("RICH_MG_FUSED_REDUCTIONS");
        bool const fused_reductions_requested =
            fused_reductions_value != nullptr &&
            fused_reductions_value[0] != '\0' &&
            std::strcmp(fused_reductions_value, "0") != 0 &&
            std::strcmp(fused_reductions_value, "false") != 0 &&
            std::strcmp(fused_reductions_value, "off") != 0 &&
            std::strcmp(fused_reductions_value, "no") != 0;
        char const* const three_round_pipeline_value =
            std::getenv("RICH_MG_THREE_ROUND_DIAGNOSTIC_PIPELINE");
        bool const three_round_pipeline_requested =
            three_round_pipeline_value != nullptr &&
            three_round_pipeline_value[0] != '\0' &&
            std::strcmp(three_round_pipeline_value, "0") != 0 &&
            std::strcmp(three_round_pipeline_value, "false") != 0 &&
            std::strcmp(three_round_pipeline_value, "off") != 0 &&
            std::strcmp(three_round_pipeline_value, "no") != 0;
        char const* const three_round_pipeline_shadow_value =
            std::getenv("RICH_MG_THREE_ROUND_DIAGNOSTIC_PIPELINE_SHADOW");
        bool const three_round_pipeline_shadow_requested =
            three_round_pipeline_shadow_value != nullptr &&
            three_round_pipeline_shadow_value[0] != '\0' &&
            std::strcmp(three_round_pipeline_shadow_value, "0") != 0 &&
            std::strcmp(three_round_pipeline_shadow_value, "false") != 0 &&
            std::strcmp(three_round_pipeline_shadow_value, "off") != 0 &&
            std::strcmp(three_round_pipeline_shadow_value, "no") != 0;
        char const* const fixed16_block_matvec_value =
            std::getenv("RICH_MG_FIXED16_BLOCK_MATVEC");
        bool const fixed16_block_matvec_requested =
            fixed16_block_matvec_value != nullptr &&
            fixed16_block_matvec_value[0] != '\0' &&
            std::strcmp(fixed16_block_matvec_value, "0") != 0 &&
            std::strcmp(fixed16_block_matvec_value, "false") != 0 &&
            std::strcmp(fixed16_block_matvec_value, "off") != 0 &&
            std::strcmp(fixed16_block_matvec_value, "no") != 0;
        char const* const fixed16_block_matvec_shadow_value =
            std::getenv("RICH_MG_FIXED16_BLOCK_MATVEC_SHADOW");
        bool const fixed16_block_matvec_shadow_requested =
            fixed16_block_matvec_shadow_value != nullptr &&
            fixed16_block_matvec_shadow_value[0] != '\0' &&
            std::strcmp(fixed16_block_matvec_shadow_value, "0") != 0 &&
            std::strcmp(fixed16_block_matvec_shadow_value, "false") != 0 &&
            std::strcmp(fixed16_block_matvec_shadow_value, "off") != 0 &&
            std::strcmp(fixed16_block_matvec_shadow_value, "no") != 0;
        char const* const fixed16_block_stencil_value =
            std::getenv("RICH_MG_FIXED16_BLOCK_STENCIL");
        bool const fixed16_block_stencil_requested =
            fixed16_block_stencil_value != nullptr &&
            fixed16_block_stencil_value[0] != '\0' &&
            std::strcmp(fixed16_block_stencil_value, "0") != 0 &&
            std::strcmp(fixed16_block_stencil_value, "false") != 0 &&
            std::strcmp(fixed16_block_stencil_value, "off") != 0 &&
            std::strcmp(fixed16_block_stencil_value, "no") != 0;
        char const* const fixed16_block_stencil_shadow_value =
            std::getenv("RICH_MG_FIXED16_BLOCK_STENCIL_SHADOW");
        bool const fixed16_block_stencil_shadow_requested =
            fixed16_block_stencil_shadow_value != nullptr &&
            fixed16_block_stencil_shadow_value[0] != '\0' &&
            std::strcmp(fixed16_block_stencil_shadow_value, "0") != 0 &&
            std::strcmp(fixed16_block_stencil_shadow_value, "false") != 0 &&
            std::strcmp(fixed16_block_stencil_shadow_value, "off") != 0 &&
            std::strcmp(fixed16_block_stencil_shadow_value, "no") != 0;
        char const* const fixed16_avx2_neighbors_value =
            std::getenv("RICH_MG_FIXED16_AVX2_NEIGHBORS");
        bool const fixed16_avx2_neighbors_requested =
            fixed16_avx2_neighbors_value != nullptr &&
            fixed16_avx2_neighbors_value[0] != '\0' &&
            std::strcmp(fixed16_avx2_neighbors_value, "0") != 0 &&
            std::strcmp(fixed16_avx2_neighbors_value, "false") != 0 &&
            std::strcmp(fixed16_avx2_neighbors_value, "off") != 0 &&
            std::strcmp(fixed16_avx2_neighbors_value, "no") != 0;
        bool const fixed16_block_stencil_builder_declared =
            matrix_builder.SupportsFixed16BlockStencil();
        bool const local_fixed16_block_stencil_eligible =
            !(fixed16_block_stencil_requested ||
              fixed16_block_stencil_shadow_requested) ||
            (fixed16_block_stencil_builder_declared &&
             matrix_builder.Fixed16BlockStencilEligible(tess));
        int const local_direct_csr_state =
            (direct_csr_requested ? 1 : 0) |
            (direct_csr_supported ? 2 : 0);
        int const local_pipeline_route_state =
            (three_round_pipeline_requested ? 1 : 0) |
            (three_round_pipeline_shadow_requested ? 2 : 0);
        int const local_runtime_route_state = local_direct_csr_state |
            (fused_reductions_requested ? 4 : 0) |
            (fixed16_block_matvec_requested ? 8 : 0) |
            (fixed16_block_matvec_shadow_requested ? 16 : 0) |
            (three_round_pipeline_requested ? 32 : 0) |
            (three_round_pipeline_shadow_requested ? 64 : 0) |
            (fixed16_block_stencil_requested ? 128 : 0) |
            (fixed16_block_stencil_shadow_requested ? 256 : 0) |
            (fixed16_block_stencil_builder_declared ? 512 : 0);
        unsigned long long runtime_route_state_mask[16] = {};
        runtime_route_state_mask[
            static_cast<unsigned int>(local_runtime_route_state) / 64] =
            1ull <<
            (static_cast<unsigned int>(local_runtime_route_state) % 64);
#ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, runtime_route_state_mask, 16,
                          MPI_UNSIGNED_LONG_LONG, MPI_BOR, MPI_COMM_WORLD),
            "MPI_Allreduce(global BiCGSTAB runtime route state)");
#endif
        bool route_state_is_one_hot = false;
        for(unsigned int word = 0; word < 16; ++word)
            if(runtime_route_state_mask[word] != 0)
            {
                bool const this_word_is_one_hot =
                    (runtime_route_state_mask[word] &
                     (runtime_route_state_mask[word] - 1ull)) == 0;
                route_state_is_one_hot = this_word_is_one_hot &&
                    !route_state_is_one_hot;
                if(!this_word_is_one_hot)
                    break;
                for(unsigned int other = word + 1; other < 16; ++other)
                    if(runtime_route_state_mask[other] != 0)
                        route_state_is_one_hot = false;
                break;
            }
        if(!route_state_is_one_hot) {
            UniversalError error(
                "Inconsistent BiCGSTAB runtime route across MPI ranks");
            error.addEntry(
                "Route state mask lower",
                static_cast<double>(runtime_route_state_mask[0]));
            error.addEntry(
                "Route state mask upper",
                static_cast<double>(runtime_route_state_mask[1]));
            throw error;
        }
        int fixed16_block_stencil_eligible_state =
            local_fixed16_block_stencil_eligible ? 1 : 0;
#ifdef RICH_MPI
        if(fixed16_block_stencil_requested ||
           fixed16_block_stencil_shadow_requested)
            RequireCGMpiSuccess(
                MPI_Allreduce(
                    MPI_IN_PLACE, &fixed16_block_stencil_eligible_state, 1,
                    MPI_INT, MPI_MIN, MPI_COMM_WORLD),
                "MPI_Allreduce(global fixed16 block-stencil eligibility)");
#endif
        bool const fixed16_block_stencil_globally_eligible =
            fixed16_block_stencil_eligible_state != 0;
#ifdef RICH_MPI
        bool const three_round_pipeline_supported =
            fused_reductions_requested;
#else
        bool const three_round_pipeline_supported = false;
#endif
        if(three_round_pipeline_requested &&
           !three_round_pipeline_supported)
            throw UniversalError(
                "RICH_MG_THREE_ROUND_DIAGNOSTIC_PIPELINE requires MPI and RICH_MG_FUSED_REDUCTIONS");
        if(three_round_pipeline_shadow_requested &&
           !three_round_pipeline_requested)
            throw UniversalError(
                "RICH_MG_THREE_ROUND_DIAGNOSTIC_PIPELINE_SHADOW requires RICH_MG_THREE_ROUND_DIAGNOSTIC_PIPELINE");
        bool direct_global_csr = direct_csr_requested &&
            direct_csr_supported;
        bool const fused_reductions = fused_reductions_requested;
        bool const three_round_pipeline =
            three_round_pipeline_requested &&
            three_round_pipeline_supported;
        bool const three_round_pipeline_shadow =
            three_round_pipeline && three_round_pipeline_shadow_requested;
        PreconditionerKind const requested_preconditioner_kind =
            matrix_builder.GetPreconditionerKind();
        bool const fixed16_block_stencil_preconditioner_supported =
            requested_preconditioner_kind ==
                PreconditionerKind::ScalarJacobi ||
            requested_preconditioner_kind ==
                PreconditionerKind::CellBlockJacobi ||
            UsesCellBlockNeighborCorrection(requested_preconditioner_kind);
        bool const fixed16_block_stencil_route_requested =
            fixed16_block_stencil_requested ||
            fixed16_block_stencil_shadow_requested;
        bool const fixed16_block_stencil_builder_supported =
            fixed16_block_stencil_builder_declared &&
            direct_csr_supported;
        bool const fixed16_block_stencil_build_attempted =
            fixed16_block_stencil_route_requested &&
            fixed16_block_stencil_builder_supported &&
            fixed16_block_stencil_globally_eligible &&
            slice == Fixed16BlockStencilMatrix::BlockSize;
        bool const fixed16_block_stencil_build_csr_shadow =
            fixed16_block_stencil_shadow_requested ||
            direct_csr_requested ||
            !fixed16_block_stencil_preconditioner_supported;
        if(fixed16_block_stencil_route_requested &&
           fixed16_block_stencil_builder_supported &&
           !fixed16_block_stencil_globally_eligible)
            direct_global_csr = true;
        char const* const reuse_rows_value =
            std::getenv("RICH_MG_REUSE_MATRIX_ROW_CAPACITY");
        bool const reuse_matrix_row_capacity = reuse_rows_value != nullptr &&
            reuse_rows_value[0] != '\0' &&
            std::strcmp(reuse_rows_value, "0") != 0 &&
            std::strcmp(reuse_rows_value, "false") != 0 &&
            std::strcmp(reuse_rows_value, "off") != 0 &&
            std::strcmp(reuse_rows_value, "no") != 0;
        if(direct_global_csr || fixed16_block_stencil_build_attempted) {
            release_container_memory(A);
            release_container_memory(A_indeces);
        }
        else if(!reuse_matrix_row_capacity) {
            A.clear();
            A_indeces.clear();
        }
        if(fixed16_block_stencil_build_attempted)
        {
            fixed16_block_stencil.PrepareForOverwrite();
            if(fixed16_block_stencil_build_csr_shadow)
            {
                A_row_ptr.clear();
                A_col_idx.clear();
                A_values.clear();
            }
            else
            {
                release_container_memory(A_row_ptr);
                release_container_memory(A_col_idx);
                release_container_memory(A_values);
            }
        }
        else
            fixed16_block_stencil.Release();
        b.clear();
        sub_x.clear();
        char const* const trace_value =
            std::getenv("RICH_INDIVIDUAL_PERF_TRACE");
        bool const trace_matrix_build = trace_value != nullptr &&
            trace_value[0] != '\0' &&
            std::strcmp(trace_value, "0") != 0 &&
            std::strcmp(trace_value, "false") != 0 &&
            std::strcmp(trace_value, "off") != 0 &&
            std::strcmp(trace_value, "no") != 0;
        auto const matrix_build_start = std::chrono::steady_clock::now();
        {
            MEMORY_PROFILE_SCOPE("diffusion matrix build");
            if(fixed16_block_stencil_build_attempted)
                matrix_builder.BuildMatrixCSRFixed16BlockStencil(
                    tess, A_row_ptr, A_col_idx, A_values,
                    cells, dt, b, sub_x, time,
                    fixed16_block_stencil_build_csr_shadow,
                    fixed16_block_stencil);
            else if(direct_global_csr)
                matrix_builder.BuildMatrixCSR(
                    tess, A_row_ptr, A_col_idx, A_values,
                    cells, dt, b, sub_x, time);
            else
                matrix_builder.BuildMatrix(
                    tess, A, A_indeces, cells, dt, b, sub_x, time);
        }
        unsigned int fixed16_block_stencil_fallback_mask = 0u;
        double fixed16_block_stencil_setup_seconds = 0.0;
        bool fixed16_block_stencil_fallback_rebuilt_csr = false;
        if(fixed16_block_stencil_route_requested)
        {
            auto const setup_start = std::chrono::steady_clock::now();
            Fixed16BlockStencilFallback local_fallback =
                Fixed16BlockStencilFallback::None;
            if(!fixed16_block_stencil_builder_declared)
                local_fallback =
                    Fixed16BlockStencilFallback::BuilderUnsupported;
            else if(!fixed16_block_stencil_globally_eligible)
                local_fallback =
                    Fixed16BlockStencilFallback::BuilderIneligible;
            else if(slice != Fixed16BlockStencilMatrix::BlockSize)
                local_fallback = Fixed16BlockStencilFallback::GroupCount;
            else if(!direct_csr_supported)
                local_fallback =
                    Fixed16BlockStencilFallback::DirectCSRUnavailable;
            else
            {
                local_fallback =
                    ValidateFixed16BlockStencil(fixed16_block_stencil);
                if(local_fallback == Fixed16BlockStencilFallback::None &&
                   fixed16_block_stencil.LocalCellCount !=
                       tess.GetPointNo())
                    local_fallback =
                        Fixed16BlockStencilFallback::LocalCellCount;
                if(local_fallback == Fixed16BlockStencilFallback::None &&
                   fixed16_block_stencil_build_csr_shadow)
                    local_fallback =
                        ValidateFixed16BlockStencilCSRShadow(
                            fixed16_block_stencil, A_row_ptr, A_col_idx,
                            A_values);
            }
            fixed16_block_stencil_fallback_mask =
                Fixed16BlockStencilFallbackBit(local_fallback);
#ifdef RICH_MPI
            RequireCGMpiSuccess(
                MPI_Allreduce(
                    MPI_IN_PLACE, &fixed16_block_stencil_fallback_mask, 1,
                    MPI_UNSIGNED, MPI_BOR, MPI_COMM_WORLD),
                "MPI_Allreduce(global fixed16 block-stencil structure)");
#endif
            fixed16_block_stencil_setup_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - setup_start).count();

            if(fixed16_block_stencil_build_attempted &&
               fixed16_block_stencil_fallback_mask != 0u)
            {
                fixed16_block_stencil.Release();
                direct_global_csr = true;
                if(!fixed16_block_stencil_build_csr_shadow)
                {
                    A_row_ptr.clear();
                    A_col_idx.clear();
                    A_values.clear();
                    b.clear();
                    sub_x.clear();
                    matrix_builder.BuildMatrixCSR(
                        tess, A_row_ptr, A_col_idx, A_values,
                        cells, dt, b, sub_x, time);
                    fixed16_block_stencil_fallback_rebuilt_csr = true;
                }
            }
            else if(fixed16_block_stencil_build_attempted)
                direct_global_csr =
                    fixed16_block_stencil_build_csr_shadow;
        }
        bool const fixed16_block_stencil_structure_supported =
            fixed16_block_stencil_build_attempted &&
            fixed16_block_stencil_fallback_mask == 0u;
        bool const fixed16_block_stencil_enabled =
            fixed16_block_stencil_requested &&
            fixed16_block_stencil_structure_supported;
        bool const fixed16_block_stencil_shadow_enabled =
            fixed16_block_stencil_shadow_requested &&
            fixed16_block_stencil_structure_supported &&
            fixed16_block_stencil_build_csr_shadow;
        double matrix_build_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - matrix_build_start).count();
        static unsigned int reported_direct_csr_states = 0;
        unsigned int const direct_csr_state_bit =
            1u << static_cast<unsigned int>(local_direct_csr_state);
        if(trace_matrix_build && rank == 0 &&
           (reported_direct_csr_states & direct_csr_state_bit) == 0u) {
            std::clog << "MG_DIRECT_GLOBAL_CSR requested="
                      << (direct_csr_requested ? 1 : 0)
                      << " supported=" << (direct_csr_supported ? 1 : 0)
                      << " enabled="
                      << (direct_csr_requested && direct_csr_supported ? 1 : 0)
                      << std::endl;
            reported_direct_csr_states |= direct_csr_state_bit;
        }
        static unsigned int reported_fused_reduction_states = 0;
        unsigned int const fused_reduction_state_bit =
            1u << static_cast<unsigned int>(fused_reductions ? 1 : 0);
        if(trace_matrix_build && rank == 0 &&
           (reported_fused_reduction_states &
            fused_reduction_state_bit) == 0u) {
            std::clog << "MG_FUSED_REDUCTIONS requested="
                      << (fused_reductions_requested ? 1 : 0)
                      << " enabled=" << (fused_reductions ? 1 : 0)
                      << std::endl;
            reported_fused_reduction_states |= fused_reduction_state_bit;
        }
        static unsigned int reported_three_round_pipeline_states = 0;
        unsigned int const three_round_pipeline_state_bit =
            1u << static_cast<unsigned int>(local_pipeline_route_state);
        if(trace_matrix_build && rank == 0 &&
           (reported_three_round_pipeline_states &
            three_round_pipeline_state_bit) == 0u) {
            std::clog
                << "MG_THREE_ROUND_DIAGNOSTIC_PIPELINE requested="
                << (three_round_pipeline_requested ? 1 : 0)
                << " supported="
                << (three_round_pipeline_supported ? 1 : 0)
                << " enabled=" << (three_round_pipeline ? 1 : 0)
                << " shadow=" << (three_round_pipeline_shadow ? 1 : 0)
                << std::endl;
            reported_three_round_pipeline_states |=
                three_round_pipeline_state_bit;
        }
        static bool reported_matrix_row_reuse = false;
        if(trace_matrix_build && rank == 0 && !reported_matrix_row_reuse) {
            std::clog << "MG_MATRIX_ROW_CAPACITY_REUSE enabled="
                      << (reuse_matrix_row_capacity ? 1 : 0) << std::endl;
            reported_matrix_row_reuse = true;
        }
        if(trace_matrix_build) {
            unsigned long long local_rows = static_cast<unsigned long long>(
                fixed16_block_stencil_enabled
                    ? Nlocal
                    : (direct_global_csr
                    ? (A_row_ptr.empty() ? 0 : A_row_ptr.size() - 1)
                    : A.size()));
            unsigned long long local_nonzeros =
                fixed16_block_stencil_enabled
                    ? static_cast<unsigned long long>(
                        fixed16_block_stencil.LocalBlockValues.size() +
                        fixed16_block_stencil.NeighborValues.size())
                    : (direct_global_csr
                       ? static_cast<unsigned long long>(A_values.size()) : 0);
            if(!fixed16_block_stencil_enabled && !direct_global_csr)
                for(auto const& row : A)
                    local_nonzeros +=
                        static_cast<unsigned long long>(row.size());
            unsigned long long total_rows = local_rows;
            unsigned long long maximum_rows = local_rows;
            unsigned long long total_nonzeros = local_nonzeros;
            unsigned long long maximum_nonzeros = local_nonzeros;
#ifdef RICH_MPI
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &matrix_build_seconds, 1,
                              MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
                "MPI_Allreduce(global matrix build time)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &total_rows, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(global matrix row total)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &maximum_rows, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(global matrix row maximum)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &total_nonzeros, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(global matrix nonzero total)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, &maximum_nonzeros, 1,
                              MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(global matrix nonzero maximum)");
#endif
            if(rank == 0)
                std::clog << std::setprecision(17)
                          << "MG_MATRIX_BUILD_TIMING scope=global"
                          << " seconds_max=" << matrix_build_seconds
                          << " rows_total=" << total_rows
                          << " rows_max=" << maximum_rows
                          << " nonzeros_total=" << total_nonzeros
                          << " nonzeros_max=" << maximum_nonzeros
                          << std::endl;
        }
        bool const common_dimensions_valid =
            b.size() >= Nlocal && sub_x.size() >= Nlocal;
        bool const matrix_dimensions_valid = fixed16_block_stencil_enabled
            ? (fixed16_block_stencil.LocalCellCount == tess.GetPointNo() &&
               fixed16_block_stencil.LocalBlockValues.size() ==
                   Nlocal * Fixed16BlockStencilMatrix::BlockSize)
            : (direct_global_csr
               ? (A_row_ptr.size() == Nlocal + 1 &&
               A_col_idx.size() == A_values.size() &&
               !A_row_ptr.empty() && A_row_ptr.front() == 0 &&
               A_row_ptr.back() == A_values.size())
               : (A.size() == Nlocal && A_indeces.size() == Nlocal));
        if(!common_dimensions_valid || !matrix_dimensions_valid) {
            UniversalError eo("Diffusion matrix dimensions do not match the declared unknown count");
            eo.addEntry("Mesh cells", static_cast<double>(tess.GetPointNo()));
            eo.addEntry("Unknowns per cell", static_cast<double>(slice));
            eo.addEntry("Expected local rows", static_cast<double>(Nlocal));
            eo.addEntry("Matrix rows", static_cast<double>(
                fixed16_block_stencil_enabled
                    ? fixed16_block_stencil.LocalCellCount *
                        Fixed16BlockStencilMatrix::BlockSize
                    : (direct_global_csr
                       ? (A_row_ptr.empty() ? 0 : A_row_ptr.size() - 1)
                       : A.size())));
            eo.addEntry("Index rows", static_cast<double>(
                fixed16_block_stencil_enabled
                    ? fixed16_block_stencil.LocalCellCount *
                        Fixed16BlockStencilMatrix::BlockSize
                    : (direct_global_csr
                       ? (A_row_ptr.empty() ? 0 : A_row_ptr.size() - 1)
                       : A_indeces.size())));
            eo.addEntry("RHS size", static_cast<double>(b.size()));
            eo.addEntry("Initial guess size", static_cast<double>(sub_x.size()));
            throw eo;
        }
        if(direct_global_csr) {
            for(size_t row = 0; row < Nlocal; ++row) {
                std::size_t const begin = A_row_ptr[row];
                std::size_t const end = A_row_ptr[row + 1];
                if(begin >= end || end > A_values.size() ||
                   A_col_idx[begin] != row) {
                    UniversalError eo("Invalid direct CSR diffusion matrix row");
                    eo.addEntry("Row", static_cast<double>(row));
                    eo.addEntry("Row begin", static_cast<double>(begin));
                    eo.addEntry("Row end", static_cast<double>(end));
                    throw eo;
                }
                for(std::size_t entry = begin; entry < end; ++entry)
                    if(A_col_idx[entry] == max_size_t) {
                        UniversalError eo(
                            "Invalid direct CSR diffusion matrix column");
                        eo.addEntry("Row", static_cast<double>(row));
                        eo.addEntry("Column", static_cast<double>(
                            A_col_idx[entry]));
                        throw eo;
                    }
            }
        }
        else if(!fixed16_block_stencil_enabled) {
            for(size_t row = 0; row < Nlocal; ++row) {
                if(A[row].empty() ||
                   A[row].size() != A_indeces[row].size()) {
                    UniversalError eo("Invalid diffusion matrix row");
                    eo.addEntry("Row", static_cast<double>(row));
                    eo.addEntry("Value count", static_cast<double>(
                        A[row].size()));
                    eo.addEntry("Index count", static_cast<double>(
                        A_indeces[row].size()));
                    throw eo;
                }
            }
        }
        auto const csr_setup_start = std::chrono::steady_clock::now();
        if(!direct_global_csr && !fixed16_block_stencil_enabled) {
            A_row_ptr.clear();
            A_col_idx.clear();
            A_values.clear();
            build_crs(A, A_indeces, A_row_ptr, A_col_idx, A_values);
        }
        std::vector<double> &M = workspace.M; // The preconditioner
        M.clear();
        if(fixed16_block_stencil_enabled)
        {
            M.resize(Nlocal);
            for(std::size_t row = 0; row < Nlocal; ++row)
                M[row] = 1.0 / fixed16_block_stencil.LocalBlockValues[
                    row * Fixed16BlockStencilMatrix::BlockSize];
        }
        else if(use_crs_matvec)
            build_M_crs(A_row_ptr, A_col_idx, A_values, M);
        else if(direct_global_csr)
            throw UniversalError(
                "Direct CSR assembly requires the CSR matrix-vector path");
        else
            build_M(A, A_indeces, M);
        Fixed16BlockMatvecSchedule fixed16_block_matvec_schedule;
        unsigned int fixed16_block_matvec_fallback_mask = 0u;
        double fixed16_block_matvec_setup_seconds = 0.0;
        if(fixed16_block_matvec_requested ||
           fixed16_block_matvec_shadow_requested)
        {
            auto const fixed16_setup_start =
                std::chrono::steady_clock::now();
            Fixed16BlockMatvecFallback const local_fallback =
                BuildFixed16BlockMatvecSchedule(
                    A_row_ptr, A_col_idx, Nlocal, slice,
                    fixed16_block_matvec_schedule);
            fixed16_block_matvec_fallback_mask =
                Fixed16BlockMatvecFallbackBit(local_fallback);
#ifdef RICH_MPI
            RequireCGMpiSuccess(
                MPI_Allreduce(
                    MPI_IN_PLACE, &fixed16_block_matvec_fallback_mask, 1,
                    MPI_UNSIGNED, MPI_BOR, MPI_COMM_WORLD),
                "MPI_Allreduce(global fixed16 block matvec structure)");
#endif
            fixed16_block_matvec_setup_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    fixed16_setup_start).count();
        }
        bool const fixed16_block_matvec_structure_supported =
            fixed16_block_matvec_fallback_mask == 0u &&
            (fixed16_block_matvec_requested ||
             fixed16_block_matvec_shadow_requested);
        bool const fixed16_block_matvec_enabled =
            fixed16_block_matvec_requested &&
            fixed16_block_matvec_structure_supported;
        bool const fixed16_block_matvec_shadow_enabled =
            fixed16_block_matvec_shadow_requested &&
            fixed16_block_matvec_structure_supported;
        double csr_setup_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - csr_setup_start).count();


        // Keep M as the scalar convergence/error scaling used historically.
        // The selected preconditioner changes only the two BiCGSTAB direction
        // solves below, so stopping criteria and max0/max1 retain their exact
        // scalar-diagonal semantics.
        auto const bicgstab_total_start = std::chrono::steady_clock::now();
        double matvec_seconds = 0;
        double fixed16_block_matvec_kernel_seconds = 0;
        double fixed16_block_matvec_shadow_seconds = 0;
        unsigned long long fixed16_block_matvec_kernel_calls = 0;
        unsigned long long fixed16_block_matvec_selected_calls = 0;
        unsigned long long fixed16_block_matvec_generic_calls = 0;
        unsigned long long fixed16_block_matvec_shadow_checks = 0;
        unsigned long long fixed16_block_matvec_shadow_mismatch_calls = 0;
        unsigned long long fixed16_block_matvec_shadow_mismatch_values = 0;
        double fixed16_block_stencil_kernel_seconds = 0;
        double fixed16_block_stencil_shadow_seconds = 0;
        unsigned long long fixed16_block_stencil_kernel_calls = 0;
        unsigned long long fixed16_block_stencil_selected_calls = 0;
        unsigned long long fixed16_block_stencil_generic_calls = 0;
        unsigned long long fixed16_block_stencil_shadow_checks = 0;
        unsigned long long fixed16_block_stencil_shadow_mismatch_calls = 0;
        unsigned long long fixed16_block_stencil_shadow_mismatch_values = 0;
        double exchange_seconds = 0;
        double reduction_seconds = 0;
        unsigned long long reduction_collective_calls = 0;
        unsigned long long reduction_causal_rounds = 0;
        double three_round_pipeline_wait_seconds[3] = {0, 0, 0};
        unsigned long long three_round_pipeline_round_counts[3] = {0, 0, 0};
        unsigned long long three_round_pipeline_collective_calls = 0;
        unsigned long long three_round_pipeline_drain_rounds = 0;
        unsigned long long three_round_pipeline_speculative_matvec_calls = 0;
        unsigned long long
            three_round_pipeline_discarded_speculative_matvec_calls = 0;
        unsigned long long three_round_pipeline_fail_closed_restarts = 0;
        unsigned long long three_round_pipeline_shadow_checks = 0;
        unsigned long long three_round_pipeline_shadow_mismatches = 0;
        auto timed_exchange = [&](std::vector<double>& values)
        {
#ifdef RICH_MPI
            auto const start = std::chrono::steady_clock::now();
            MPI_exchange_data(tess, values, true, slice);
            exchange_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
#else
            (void)values;
#endif
        };
        auto timed_dot_product = [&](std::vector<double> const& left,
                                     std::vector<double> const& right)
        {
            auto const start = std::chrono::steady_clock::now();
            double const result = mpi_dot_product(left, right);
#ifdef RICH_MPI
            RecordReductionBatch(&reduction_collective_calls,
                                 &reduction_causal_rounds, 1, 1);
#endif
            reduction_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            return result;
        };
        auto timed_dot_product_pair = [&](
                std::vector<double> const& left0,
                std::vector<double> const& right0,
                std::vector<double> const& left1,
                std::vector<double> const& right1)
        {
            auto const start = std::chrono::steady_clock::now();
            std::array<double, 2> result = {{
                local_dot_product(left0, right0),
                local_dot_product(left1, right1)}};
#ifdef RICH_MPI
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, result.data(), 2, MPI_DOUBLE,
                              MPI_SUM, MPI_COMM_WORLD),
                "MPI_Allreduce(CG paired dot products)");
            RecordReductionBatch(&reduction_collective_calls,
                                 &reduction_causal_rounds, 1, 1);
#endif
            reduction_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            return result;
        };
        auto timed_dot_product_triple = [&](
                std::vector<double> const& left0,
                std::vector<double> const& right0,
                std::vector<double> const& left1,
                std::vector<double> const& right1,
                std::vector<double> const& left2,
                std::vector<double> const& right2)
        {
            auto const start = std::chrono::steady_clock::now();
            std::array<double, 3> result = {{
                local_dot_product(left0, right0),
                local_dot_product(left1, right1),
                local_dot_product(left2, right2)}};
#ifdef RICH_MPI
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, result.data(), 3, MPI_DOUBLE,
                              MPI_SUM, MPI_COMM_WORLD),
                "MPI_Allreduce(CG initialization dot products)");
            RecordReductionBatch(&reduction_collective_calls,
                                 &reduction_causal_rounds, 1, 1);
#endif
            reduction_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            return result;
        };
        auto timed_local_dot_product_pair = [&reduction_seconds](
                std::vector<double> const& left0,
                std::vector<double> const& right0,
                std::vector<double> const& left1,
                std::vector<double> const& right1)
        {
            auto const start = std::chrono::steady_clock::now();
            std::array<double, 2> result = {{
                local_dot_product(left0, right0),
                local_dot_product(left1, right1)}};
            reduction_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            return result;
        };
        CellBlockJacobiPreconditioner direction_preconditioner;
        bool const local_preconditioner_setup_valid =
            fixed16_block_stencil_enabled &&
            fixed16_block_stencil_preconditioner_supported
                ? direction_preconditioner.SetupFixed16BlockStencil(
                    fixed16_block_stencil,
                    requested_preconditioner_kind, M)
                : (direct_global_csr
                ? direction_preconditioner.SetupCSR(
                    A_row_ptr, A_col_idx, A_values, slice,
                    requested_preconditioner_kind, M)
                : direction_preconditioner.Setup(
                    A, A_indeces, slice,
                    requested_preconditioner_kind, M));
        double const requested_preconditioner_setup_seconds =
            direction_preconditioner.SetupSeconds();
        int preconditioner_setup_state =
            (local_preconditioner_setup_valid ? 0 : 1) |
            (local_preconditioner_setup_valid &&
             direction_preconditioner.FallbackBlockCount() > 0 ? 2 : 0) |
            (local_preconditioner_setup_valid &&
             !direction_preconditioner.RequestedKindSupported() ? 4 : 0);
#ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &preconditioner_setup_state, 1,
                          MPI_INT, MPI_BOR, MPI_COMM_WORLD),
            "MPI_Allreduce(global preconditioner setup state)");
#endif
        if((preconditioner_setup_state & 1) != 0)
            throw UniversalError(
                "Failed to construct the diffusion preconditioner");
        if((preconditioner_setup_state & 4) != 0)
        {
            if(requested_preconditioner_kind ==
               PreconditionerKind::RankLocalILU0)
            {
                bool const fallback_setup_valid = direct_global_csr
                    ? direction_preconditioner.SetupCSR(
                        A_row_ptr, A_col_idx, A_values, slice,
                        PreconditionerKind::CellBlockJacobi, M)
                    : direction_preconditioner.Setup(
                        A, A_indeces, slice,
                        PreconditionerKind::CellBlockJacobi, M);
                if(!fallback_setup_valid)
                    throw UniversalError(
                        "Failed to construct the ILU(0) fallback preconditioner");
            }
            else
                direction_preconditioner.DowngradeToCellBlockJacobi();
        }
        double preconditioner_setup_max =
            requested_preconditioner_setup_seconds +
            ((preconditioner_setup_state & 4) != 0 &&
             requested_preconditioner_kind ==
                 PreconditionerKind::RankLocalILU0 ?
             direction_preconditioner.SetupSeconds() : 0);
        if((preconditioner_setup_state & 2) != 0)
        {
        unsigned long long preconditioner_counts[7] = {
            static_cast<unsigned long long>(
                direction_preconditioner.BlockCount()),
            static_cast<unsigned long long>(
                direction_preconditioner.FactorizedBlockCount()),
            static_cast<unsigned long long>(
                direction_preconditioner.FallbackBlockCount()),
            static_cast<unsigned long long>(
                direction_preconditioner.StorageBytes()),
            static_cast<unsigned long long>(
                direction_preconditioner.LocalLowerCouplingCount()),
            static_cast<unsigned long long>(
                direction_preconditioner.IgnoredLocalUpperCouplingCount()),
            static_cast<unsigned long long>(
                direction_preconditioner.IgnoredRemoteCouplingCount())};
        unsigned long long preconditioner_storage_max =
            preconditioner_counts[3];
        double preconditioner_minimum_pivot =
            direction_preconditioner.MinimumNormalizedPivot();
        if(!std::isfinite(preconditioner_minimum_pivot))
            preconditioner_minimum_pivot =
                std::numeric_limits<double>::infinity();
        int representative_rank =
            direction_preconditioner.FallbackBlockCount() > 0 ? rank :
            std::numeric_limits<int>::max();
#ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, preconditioner_counts, 7,
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
            "MPI_Allreduce(global preconditioner counts)");
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &preconditioner_storage_max, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD),
            "MPI_Allreduce(global preconditioner storage)");
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &preconditioner_setup_max, 1,
                          MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
            "MPI_Allreduce(global preconditioner setup time)");
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &preconditioner_minimum_pivot, 1,
                          MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD),
            "MPI_Allreduce(global preconditioner pivot)");
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &representative_rank, 1,
                          MPI_INT, MPI_MIN, MPI_COMM_WORLD),
            "MPI_Allreduce(global preconditioner representative rank)");
#endif
        if(!std::isfinite(preconditioner_minimum_pivot))
            preconditioner_minimum_pivot =
                std::numeric_limits<double>::quiet_NaN();
        if(rank == 0)
            std::clog << "MG_PRECONDITIONER_SETUP kind="
                      << PreconditionerKindLabel(
                             direction_preconditioner.Kind())
                      << " requested_kind="
                      << PreconditionerKindLabel(
                             requested_preconditioner_kind)
                      << " requested_kind_supported="
                      << ((preconditioner_setup_state & 4) == 0 ? 1 : 0)
                      << " scope=global"
                      << " block_size="
                      << direction_preconditioner.BlockSize()
                      << " blocks=" << preconditioner_counts[0]
                      << " factorized_blocks=" << preconditioner_counts[1]
                      << " fallback_blocks=" << preconditioner_counts[2]
                      << " local_lower_couplings="
                      << preconditioner_counts[4]
                      << " ignored_local_upper_couplings="
                      << preconditioner_counts[5]
                      << " ignored_remote_couplings="
                      << preconditioner_counts[6]
                      << " min_normalized_pivot="
                      << preconditioner_minimum_pivot
                      << " row_equilibrated="
                      << (direction_preconditioner.Kind() !=
                              PreconditionerKind::ScalarJacobi ? 1 : 0)
                      << " block_sweeps="
                      << CellBlockJacobiSweepCount(
                             direction_preconditioner.Kind())
                      << " neighbor_correction_damping="
                      << (UsesCellBlockNeighborCorrection(
                              direction_preconditioner.Kind()) ?
                          cell_block_neighbor_correction_damping : 0)
                      << " relative_pivot_threshold="
                      << 64 * std::numeric_limits<double>::epsilon()
                      << " scalar_fallback_enabled=1 regularization=0"
                      << " setup_seconds_max=" << preconditioner_setup_max
                      << " storage_bytes_total=" << preconditioner_counts[3]
                      << " storage_bytes_max="
                      << preconditioner_storage_max << std::endl;
        if(representative_rank != std::numeric_limits<int>::max())
        {
            unsigned long long representative[3] = {
                std::numeric_limits<unsigned long long>::max(),
                std::numeric_limits<unsigned long long>::max(),
                std::numeric_limits<unsigned long long>::max()};
            int representative_reason =
                static_cast<int>(CellBlockFallbackReason::None);
            if(rank == representative_rank)
            {
                size_t const block =
                    direction_preconditioner.FirstFallbackBlock();
                representative[0] = static_cast<unsigned long long>(block);
                representative[1] = static_cast<unsigned long long>(
                    direction_preconditioner.FirstFallbackGroup());
                if(block < cells.size())
                    representative[2] = static_cast<unsigned long long>(
                        cells[block].ID);
                representative_reason = static_cast<int>(
                    direction_preconditioner.FirstFallbackReason());
            }
#ifdef RICH_MPI
            RequireCGMpiSuccess(
                MPI_Bcast(representative, 3, MPI_UNSIGNED_LONG_LONG,
                          representative_rank, MPI_COMM_WORLD),
                "MPI_Bcast(global preconditioner representative)");
            RequireCGMpiSuccess(
                MPI_Bcast(&representative_reason, 1, MPI_INT,
                          representative_rank, MPI_COMM_WORLD),
                "MPI_Bcast(global preconditioner fallback reason)");
#endif
            if(rank == 0)
                std::clog << "MG_PRECONDITIONER_FALLBACK count="
                          << preconditioner_counts[2]
                          << " representative_rank=" << representative_rank
                          << " representative_block=" << representative[0]
                          << " representative_cell_id=" << representative[2]
                          << " representative_group=" << representative[1]
                          << " reason=" << CellBlockFallbackReasonLabel(
                              static_cast<CellBlockFallbackReason>(
                                  representative_reason))
                          << std::endl;
        }
        }

        std::vector<double> &A_diag = workspace.A_diag;
        A_diag.resize(Nlocal);
        for(size_t j = 0; j < Nlocal; ++j)
            A_diag[j] = fixed16_block_stencil_enabled
                ? fixed16_block_stencil.LocalBlockValues[
                    j * Fixed16BlockStencilMatrix::BlockSize]
                : (direct_global_csr
                   ? A_values[A_row_ptr[j]] : A[j][0]);
        if(!direct_global_csr) {
            release_container_memory(A);
            release_container_memory(A_indeces);
        }

        // After releasing A/A_indeces, iterative products use either CSR or
        // the validated exact-order fixed-16 stencil.  True-residual sampling
        // traverses the selected representation in that same matrix order.
        std::vector<double> fixed16_block_matvec_shadow_output;
        bool fixed16_block_matvec_shadow_failed = false;
        std::vector<double> fixed16_block_stencil_shadow_output;
        bool fixed16_block_stencil_shadow_failed = false;
        bool const fixed16_block_matvec_route_requested =
            fixed16_block_matvec_requested ||
            fixed16_block_matvec_shadow_requested;
        auto matvec = [&](const std::vector<double> &in, std::vector<double> &out)
        {
            auto const start = std::chrono::steady_clock::now();
            if(fixed16_block_stencil_structure_supported)
            {
                bool const candidate_available =
                    !fixed16_block_stencil_shadow_failed;
                bool const select_candidate =
                    fixed16_block_stencil_enabled && candidate_available;
                bool const verify_candidate =
                    fixed16_block_stencil_shadow_enabled &&
                    candidate_available;
                if(select_candidate)
                {
                    auto const kernel_start =
                        std::chrono::steady_clock::now();
                    mat_times_vec_fixed16_block_stencil(
                        fixed16_block_stencil, in, out,
                        fixed16_avx2_neighbors_requested);
                    fixed16_block_stencil_kernel_seconds +=
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() -
                            kernel_start).count();
                    ++fixed16_block_stencil_kernel_calls;
                    if(verify_candidate)
                    {
                        auto const shadow_start =
                            std::chrono::steady_clock::now();
                        mat_times_vec_crs(
                            A_row_ptr, A_col_idx, A_values, in,
                            fixed16_block_stencil_shadow_output);
                        std::size_t const mismatches =
                            CountFixed16BlockMatvecShadowMismatches(
                                out, fixed16_block_stencil_shadow_output);
                        fixed16_block_stencil_shadow_seconds +=
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                shadow_start).count();
                        ++fixed16_block_stencil_shadow_checks;
                        if(mismatches != 0)
                        {
                            out = fixed16_block_stencil_shadow_output;
                            fixed16_block_stencil_shadow_failed = true;
                            ++fixed16_block_stencil_shadow_mismatch_calls;
                            fixed16_block_stencil_shadow_mismatch_values +=
                                static_cast<unsigned long long>(mismatches);
                            ++fixed16_block_stencil_generic_calls;
                        }
                        else
                            ++fixed16_block_stencil_selected_calls;
                    }
                    else
                        ++fixed16_block_stencil_selected_calls;
                }
                else
                {
                    mat_times_vec_crs(
                        A_row_ptr, A_col_idx, A_values, in, out);
                    ++fixed16_block_stencil_generic_calls;
                    if(verify_candidate)
                    {
                        auto const shadow_start =
                            std::chrono::steady_clock::now();
                        auto const kernel_start = shadow_start;
                        mat_times_vec_fixed16_block_stencil(
                            fixed16_block_stencil, in,
                            fixed16_block_stencil_shadow_output,
                            fixed16_avx2_neighbors_requested);
                        fixed16_block_stencil_kernel_seconds +=
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                kernel_start).count();
                        ++fixed16_block_stencil_kernel_calls;
                        std::size_t const mismatches =
                            CountFixed16BlockMatvecShadowMismatches(
                                fixed16_block_stencil_shadow_output, out);
                        fixed16_block_stencil_shadow_seconds +=
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                shadow_start).count();
                        ++fixed16_block_stencil_shadow_checks;
                        if(mismatches != 0)
                        {
                            fixed16_block_stencil_shadow_failed = true;
                            ++fixed16_block_stencil_shadow_mismatch_calls;
                            fixed16_block_stencil_shadow_mismatch_values +=
                                static_cast<unsigned long long>(mismatches);
                        }
                    }
                }
            }
            else if(!fixed16_block_matvec_route_requested)
                mat_times_vec_crs(
                    A_row_ptr, A_col_idx, A_values, in, out);
            else
            {
                bool const candidate_available =
                    fixed16_block_matvec_structure_supported &&
                    !fixed16_block_matvec_shadow_failed;
                bool const select_candidate =
                    fixed16_block_matvec_enabled && candidate_available;
                bool const verify_candidate =
                    fixed16_block_matvec_shadow_enabled &&
                    candidate_available;

                if(select_candidate)
                {
                    auto const kernel_start =
                        std::chrono::steady_clock::now();
                    mat_times_vec_fixed16_block(
                        fixed16_block_matvec_schedule, A_values, in, out);
                    fixed16_block_matvec_kernel_seconds +=
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() -
                            kernel_start).count();
                    ++fixed16_block_matvec_kernel_calls;

                    if(verify_candidate)
                    {
                        auto const shadow_start =
                            std::chrono::steady_clock::now();
                        mat_times_vec_crs(
                            A_row_ptr, A_col_idx, A_values, in,
                            fixed16_block_matvec_shadow_output);
                        std::size_t const mismatches =
                            fixed16_block_matvec_schedule.RowCount == 0 ? 0 :
                            CountFixed16BlockMatvecShadowMismatches(
                                out, fixed16_block_matvec_shadow_output);
                        fixed16_block_matvec_shadow_seconds +=
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                shadow_start).count();
                        ++fixed16_block_matvec_shadow_checks;
                        if(mismatches != 0)
                        {
                            out = fixed16_block_matvec_shadow_output;
                            fixed16_block_matvec_shadow_failed = true;
                            ++fixed16_block_matvec_shadow_mismatch_calls;
                            fixed16_block_matvec_shadow_mismatch_values +=
                                static_cast<unsigned long long>(mismatches);
                            ++fixed16_block_matvec_generic_calls;
                        }
                        else
                            ++fixed16_block_matvec_selected_calls;
                    }
                    else
                        ++fixed16_block_matvec_selected_calls;
                }
                else
                {
                    mat_times_vec_crs(
                        A_row_ptr, A_col_idx, A_values, in, out);
                    ++fixed16_block_matvec_generic_calls;
                    if(verify_candidate)
                    {
                        auto const shadow_start =
                            std::chrono::steady_clock::now();
                        auto const kernel_start = shadow_start;
                        mat_times_vec_fixed16_block(
                            fixed16_block_matvec_schedule, A_values, in,
                            fixed16_block_matvec_shadow_output);
                        fixed16_block_matvec_kernel_seconds +=
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                kernel_start).count();
                        ++fixed16_block_matvec_kernel_calls;
                        std::size_t const mismatches =
                            fixed16_block_matvec_schedule.RowCount == 0 ? 0 :
                            CountFixed16BlockMatvecShadowMismatches(
                                fixed16_block_matvec_shadow_output, out);
                        fixed16_block_matvec_shadow_seconds +=
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                shadow_start).count();
                        ++fixed16_block_matvec_shadow_checks;
                        if(mismatches != 0)
                        {
                            fixed16_block_matvec_shadow_failed = true;
                            ++fixed16_block_matvec_shadow_mismatch_calls;
                            fixed16_block_matvec_shadow_mismatch_values +=
                                static_cast<unsigned long long>(mismatches);
                        }
                    }
                }
            }
            matvec_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
        };
        std::vector<double> &r_old = workspace.r_old;
        std::vector<double> &sub_a_times_p = workspace.sub_a_times_p;
        std::vector<double> &sub_r = workspace.sub_r;
        r_old.clear();
        sub_a_times_p.clear();
        sub_r.clear();
        timed_exchange(sub_x);
        matvec(sub_x, sub_a_times_p);
        double maxA[2] = {0, 0};
        for(size_t i = 0; i < Nlocal; ++i)
        {
            maxA[0] = std::max(maxA[0], std::abs(sub_x[i]));
            maxA[1] = std::max(maxA[1], std::abs(b[i]));
        }       
#ifdef RICH_MPI
        auto const max_a_reduction_start = std::chrono::steady_clock::now();
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &maxA, 2, MPI_DOUBLE, MPI_MAX,
                          MPI_COMM_WORLD),
            "MPI_Allreduce(global BiCGSTAB normalization)");
        RecordReductionBatch(&reduction_collective_calls,
                             &reduction_causal_rounds, 1, 1);
        reduction_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - max_a_reduction_start).count();
#endif
        vec_lin_combo(1.0, b, -1.0, sub_a_times_p, sub_r);  
        double rho0 = 0;
        if(!fused_reductions)
            rho0 = timed_dot_product(sub_r, sub_r);
        std::vector<double> &sub_p = workspace.sub_p;
        std::vector<double> &sub_r0 = workspace.sub_r0;
        sub_p = sub_r;
        sub_r0 = sub_r;
        sub_p.resize(Nlocal);
        sub_x.resize(Nlocal);
        sub_r0.resize(Nlocal);
        vector_rescale(sub_p, M, sub_p);
        std::vector<double> &y = workspace.y;
        std::vector<double> &z = workspace.z;
        std::vector<double> &v = workspace.v;
        std::vector<double> &h = workspace.h;
        std::vector<double> &s = workspace.s;
        std::vector<double> &t = workspace.t;
        std::vector<double> &scratch_rescale1 = workspace.scratch_rescale1;
        std::vector<double> &scratch_rescale2 = workspace.scratch_rescale2;
        std::vector<double> &old_x = workspace.old_x;
        y.assign(Nlocal, 0);
        z.assign(Nlocal, 0);
        v.assign(Nlocal, 0);
        h.assign(Nlocal, 0);
        s.assign(Nlocal, 0);
        t.assign(Nlocal, 0);
        scratch_rescale1.assign(Nlocal, 0);
        scratch_rescale2.assign(Nlocal, 0);
        old_x = sub_x; 
        size_t Ntotal = Nlocal;
        double sub_r_sqrd = 0;
        if(!fused_reductions)
            sub_r_sqrd = timed_dot_product(sub_r, sub_p);
        vector_rescale(b, M, scratch_rescale1);
        double scale_b = 0;
        if(fused_reductions)
        {
            std::array<double, 3> const initial_dots =
                timed_dot_product_triple(
                    sub_r0, sub_r, sub_r, sub_p, scratch_rescale1, b);
            rho0 = initial_dots[0];
            sub_r_sqrd = initial_dots[1];
            scale_b = initial_dots[2];
        }
        else
            scale_b = timed_dot_product(scratch_rescale1, b);
        double const delta_init = sub_r_sqrd;
        unsigned long long preconditioner_applications = 0;
        unsigned long long neighbor_correction_matvec_calls = 0;
        auto apply_direction_preconditioner =
            [&](std::vector<double> const& input,
                std::vector<double>& output,
                std::vector<double>& matrix_output)
        {
            if(direction_preconditioner.Kind() ==
               PreconditionerKind::CellBlockGaussSeidel)
                direction_preconditioner.ApplyCSRForwardGaussSeidel(
                    input, output, A_row_ptr, A_col_idx, A_values);
            else if(direction_preconditioner.Kind() ==
                    PreconditionerKind::RankLocalILU0)
                direction_preconditioner.ApplyCSRRankLocalILU0(
                    input, output, A_row_ptr, A_col_idx);
            else
                direction_preconditioner.Apply(input, output);
            ++preconditioner_applications;
            std::size_t const block_sweeps = CellBlockJacobiSweepCount(
                direction_preconditioner.Kind());
            for(std::size_t sweep = 1; sweep < block_sweeps; ++sweep) {
                timed_exchange(output);
                matvec(output, matrix_output);
                output.resize(Nlocal);
                ApplyCellBlockJacobiCorrectionSweep(
                    direction_preconditioner, input, matrix_output, output,
                    scratch_rescale1);
                ++neighbor_correction_matvec_calls;
            }
        };
        double sub_r_sqrd_old = 0, sub_p_by_ap = 0, alpha = 0, beta = 0;
        struct
        {
            double val;
            int mpi_id;
        } max_data[3];
        max_data[0].mpi_id = rank;
        max_data[1].mpi_id = rank;
        max_data[2].mpi_id = rank;
        
        size_t max_loc0 = 0, max_loc1 = 0, max_loc2 = 0;
        // Main Conjugate Gradient loop
        // this loop must be serial b/c CG is an iterative method

        sub_p = sub_r;
        double error = std::numeric_limits<double>::quiet_NaN();
        bool print = false;//rank == 0 && time >  136.97915 && time <  136.98;
        std::vector<double> sampled_true_residual;
        GlobalTrueResidualAssessment true_residual_assessment;
        size_t last_true_eta_iteration = 0;
        auto const sample_true_residual = [&](size_t const iterations)
        {
            timed_exchange(sub_x);
            double const reduction_before = reduction_seconds;
            auto const start = std::chrono::steady_clock::now();
            bool const valid = compute_global_true_residual(
                A_row_ptr, A_col_idx, A_values,
                fixed16_block_stencil_enabled &&
                !fixed16_block_stencil_shadow_failed ?
                    &fixed16_block_stencil : nullptr,
                sub_x, b, Nlocal, slice,
                sampled_true_residual, true_residual_assessment,
                &reduction_seconds, fused_reductions,
                &reduction_collective_calls, &reduction_causal_rounds);
            double const elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            matvec_seconds += std::max(
                0.0, elapsed - (reduction_seconds - reduction_before));
            sub_x.resize(Nlocal);
            last_true_eta_iteration = iterations;
            return valid && true_residual_assessment.finite;
        };
        HistoricalMGMetrics last_metrics;
        auto const update_legacy_metrics = [&]()
        {
            error = last_metrics.historical_error;
            max_data[0].val = last_metrics.max0;
            max_data[1].val = last_metrics.max1;
            max_data[2].val = last_metrics.negative;
            max_data[0].mpi_id = last_metrics.max0_rank;
            max_data[1].mpi_id = last_metrics.max1_rank;
            max_data[2].mpi_id = last_metrics.negative_rank;
            if(rank == last_metrics.max0_rank &&
               last_metrics.max0_unknown != max_size_t)
                max_loc0 = last_metrics.max0_unknown;
            if(rank == last_metrics.max1_rank &&
               last_metrics.max1_unknown != max_size_t)
                max_loc1 = last_metrics.max1_unknown;
            if(rank == last_metrics.negative_rank &&
               last_metrics.negative_unknown != max_size_t)
                max_loc2 = last_metrics.negative_unknown;
        };
        bool initial_diagnostic_valid = sample_true_residual(0);
        if(initial_diagnostic_valid)
        {
            sub_r = sampled_true_residual;
            sub_r0 = sub_r;
            sub_p = sub_r;
            if(!fused_reductions)
                rho0 = timed_dot_product(sub_r0, sub_r);
            vector_rescale(sub_r, M, scratch_rescale1);
            if(fused_reductions)
            {
                std::array<double, 2> const restart_dots =
                    timed_dot_product_pair(
                        sub_r0, sub_r, scratch_rescale1, sub_r);
                rho0 = restart_dots[0];
                sub_r_sqrd = restart_dots[1];
            }
            else
                sub_r_sqrd = timed_dot_product(scratch_rescale1, sub_r);
            last_metrics = MeasureHistoricalMGImpl(
                sub_x, old_x, sub_r, b, A_diag, slice,
                &reduction_seconds, fused_reductions, nullptr, 0,
                &reduction_collective_calls, &reduction_causal_rounds);
            update_legacy_metrics();
            initial_diagnostic_valid = last_metrics.finite;
        }
        char const* failure_reason = initial_diagnostic_valid ?
            "maximum_iterations" : "diagnostic_eta_nonfinite";
        size_t completed_iterations = 0;
        HistoricalMGPositivityContinuation positivity_continuation;
        HistoricalMGBranch AcceptedConvergenceBranch =
            HistoricalMGBranch::Continue;
        auto const attempt_finalize_success =
            [&](HistoricalMGDecision const& decision,
                size_t const iterations)
        {
            if(decision.accept && !positivity_continuation.Active)
                AcceptedConvergenceBranch = decision.branch;
            GlobalHistoricalCorrectionCandidate candidate;
            if(!build_historical_final_correction(
                   slice, iterations, true_residual_assessment,
                   last_true_eta_iteration, sampled_true_residual, sub_x,
                   tess, Nlocal, matrix_builder.GetLengthScale(),
                   cells, candidate)) {
                workspace.historical_correction =
                    HistoricalMGResidualCorrectionDiagnostics{};
                failure_reason = "historical_final_correction_nonfinite";
                return HistoricalMGCorrectionDisposition::RejectNonFinite;
            }
            workspace.historical_correction =
                candidate.correction_diagnostics;
            HistoricalMGPositivityContinuationDecision const
                continuation_decision =
                    EvaluateHistoricalMGPositivityContinuation(
                        candidate.corrected_negativity,
                        candidate.positive_floor, iterations,
                        positivity_continuation);
            if(continuation_decision ==
               HistoricalMGPositivityContinuationDecision::Restart)
                ReportHistoricalMGPositivityContinuationOpen(
                    "global", positivity_continuation, rank == 0);
            else if(continuation_decision ==
                    HistoricalMGPositivityContinuationDecision::Cleared)
                ReportHistoricalMGPositivityContinuationClose(
                    "global", positivity_continuation,
                    candidate.corrected_negativity, "cleared", rank == 0);
            else if(continuation_decision ==
                    HistoricalMGPositivityContinuationDecision::Exhausted)
                ReportHistoricalMGPositivityContinuationClose(
                    "global", positivity_continuation,
                    candidate.corrected_negativity, "budget_exhausted",
                    rank == 0);
            RecordHistoricalMGPositivityContinuation(
                positivity_continuation, workspace.historical_correction);
            if(continuation_decision ==
               HistoricalMGPositivityContinuationDecision::Restart)
                return HistoricalMGCorrectionDisposition::DeferPositivity;
            if(continuation_decision ==
               HistoricalMGPositivityContinuationDecision::Continue)
                return HistoricalMGCorrectionDisposition::ContinuePositivity;
            if(ShouldCommitHistoricalMGComptonFallback(
                   candidate.corrected_negativity, continuation_decision,
                   matrix_builder.HistoricalMGComptonFallbackAvailable(
                       candidate.corrected_negativity.CellId))) {
                workspace.historical_correction.compton_fallback_candidate =
                    true;
                RecordHistoricalMGPositiveFloor(
                    HistoricalMGPositiveFloorAssessment{},
                    workspace.historical_correction);
                candidate.corrected_solution =
                    std::move(candidate.capped_corrected_solution);
                candidate.positive_floor =
                    HistoricalMGPositiveFloorAssessment{};
                candidate.spectral_failure =
                    HistoricalMGCorrectionSpectralFailure{};
            }
            if(candidate.spectral_failure.CausedRejection) {
                RecordHistoricalMGResidualCorrectionFailure(
                    candidate.spectral_failure,
                    workspace.historical_correction, iterations,
                    &positivity_continuation);
                failure_reason =
                    "historical_residual_correction_post_cap_nonphysical";
                return HistoricalMGCorrectionDisposition::
                    RejectPostCapNonphysical;
            }
            HistoricalMGCorrectionDisposition const disposition =
                ClassifyHistoricalMGCorrection(candidate.assessment);
            if(disposition != HistoricalMGCorrectionDisposition::Commit) {
                failure_reason = "historical_final_correction_nonfinite";
                return disposition;
            }
            if(candidate.positive_floor.Applied) {
                sub_x = candidate.corrected_solution;
                workspace.historical_correction.
                    post_floor_true_residual_evaluated = true;
                if(!sample_true_residual(iterations)) {
                    workspace.historical_correction.
                        post_floor_true_residual_finite = false;
                    workspace.historical_correction.failure_reason =
                        "post_floor_true_residual_nonfinite";
                    workspace.historical_correction.failure_class =
                        RadiationPositivity::SpectralRepairFailure::
                            NonfiniteRepairedExtent;
                    failure_reason = "post_floor_true_residual_nonfinite";
                    return HistoricalMGCorrectionDisposition::RejectNonFinite;
                }
                HistoricalMGMetrics const post_floor_metrics =
                    MeasureHistoricalMGImpl(
                        sub_x, old_x, sampled_true_residual, b, A_diag, slice,
                        &reduction_seconds, fused_reductions, nullptr, 0,
                        &reduction_collective_calls,
                        &reduction_causal_rounds);
                workspace.historical_correction.
                    post_floor_true_residual_finite = post_floor_metrics.finite;
                workspace.historical_correction.
                    post_floor_true_residual_error =
                        post_floor_metrics.historical_error;
                if(!post_floor_metrics.finite) {
                    workspace.historical_correction.failure_reason =
                        "post_floor_true_residual_nonfinite";
                    workspace.historical_correction.failure_class =
                        RadiationPositivity::SpectralRepairFailure::
                            NonfiniteRepairedExtent;
                    failure_reason = "post_floor_true_residual_nonfinite";
                    return HistoricalMGCorrectionDisposition::RejectNonFinite;
                }
            }
            if(positivity_continuation.Active) {
                ReportHistoricalMGPositivityContinuationClose(
                    "global", positivity_continuation,
                    candidate.corrected_negativity, "accepted", rank == 0);
                RecordHistoricalMGPositivityContinuation(
                    positivity_continuation,
                    workspace.historical_correction);
            }
            HistoricalMGCorrectionAssessment const final_assessment =
                candidate.assessment;
            HistoricalMGPositiveFloorAssessment const final_positive_floor =
                candidate.positive_floor;
            HistoricalMGBranch const CompletionBranch = decision.accept ?
                decision.branch : AcceptedConvergenceBranch;
            commit_historical_final_correction(
                slice, rank, iterations, CompletionBranch, last_metrics,
                true_residual_assessment, last_true_eta_iteration,
                std::move(candidate), sub_x_solution, cells, total_iters);
            if(rank == 0)
                std::clog << std::setprecision(17)
                          << "MG_BICGSTAB_FINAL_CORRECTION"
                          << " scope=global phase=final"
                          << " final_correction_iteration=" << iterations
                          << " pre_floor_negative_groups="
                          << final_assessment.negative_group_count
                          << " pre_floor_negative_extent="
                          << final_assessment.negative_extent
                          << " positive_floor_applied="
                          << (final_positive_floor.Applied ? 1 : 0)
                          << " positive_floor_collected_globally="
                          << (final_positive_floor.CollectedGlobally ? 1 : 0)
                          << " positive_floor_global_cells="
                          << final_positive_floor.FlooredCells
                          << " positive_floor_global_groups="
                          << final_positive_floor.FlooredGroups
                          << " positive_floor_global_injected_energy="
                          << final_positive_floor.GlobalInjectedEnergy
                          << " positive_floor_global_positive_energy="
                          << final_positive_floor.GlobalPositiveEnergy
                          << " positive_floor_global_injection_ratio="
                          << final_positive_floor.GlobalInjectionRatio
                          << " positive_floor_maximum_cell_injection_ratio="
                          << final_positive_floor.MaximumCellInjectionRatio
                          << " positive_floor_energy_discrepancy_ratio="
                          << final_positive_floor.EnergyDiscrepancyRatio
                          << std::endl;
            return disposition;
        };
        auto const restart_from_true_residual = [&]()
        {
            sub_r = sampled_true_residual;
            sub_r0 = sub_r;
            sub_p = sub_r;
            if(!fused_reductions)
                rho0 = timed_dot_product(sub_r0, sub_r);
            vector_rescale(sub_r, M, scratch_rescale1);
            if(fused_reductions)
            {
                std::array<double, 2> const restart_dots =
                    timed_dot_product_pair(
                        sub_r0, sub_r, scratch_rescale1, sub_r);
                rho0 = restart_dots[0];
                sub_r_sqrd = restart_dots[1];
            }
            else
                sub_r_sqrd = timed_dot_product(scratch_rescale1, sub_r);
        };
        auto const finish_breakdown = [&](HistoricalMGBreakdown const breakdown,
                                          size_t const zero_based_iteration)
        {
            size_t const iterations = zero_based_iteration + 1;
            completed_iterations = iterations;
            if(!sample_true_residual(iterations))
            {
                failure_reason = "diagnostic_eta_nonfinite";
                return HistoricalMGBreakdownResolution::Rejected;
            }
            last_metrics = MeasureHistoricalMGImpl(
                sub_x, old_x, sampled_true_residual, b, A_diag, slice,
                &reduction_seconds, fused_reductions, nullptr, 0,
                &reduction_collective_calls, &reduction_causal_rounds);
            update_legacy_metrics();
            if(ShouldRestartHistoricalMGFiniteBreakdown(
                   breakdown, last_metrics, iterations)) {
                restart_from_true_residual();
                return HistoricalMGBreakdownResolution::Restart;
            }
            HistoricalMGDecision const decision = ClassifyHistoricalMG(
                last_metrics, zero_based_iteration, tolerance, breakdown,
                breakdown != HistoricalMGBreakdown::NonFinite);
            if(decision.accept) {
                HistoricalMGCorrectionDisposition const disposition =
                    attempt_finalize_success(decision, iterations);
                if(disposition ==
                       HistoricalMGCorrectionDisposition::DeferPositivity ||
                   disposition == HistoricalMGCorrectionDisposition::
                       ContinuePositivity) {
                    restart_from_true_residual();
                    return HistoricalMGBreakdownResolution::Restart;
                }
                if(disposition == HistoricalMGCorrectionDisposition::Commit)
                    return HistoricalMGBreakdownResolution::Converged;
                return HistoricalMGBreakdownResolution::Rejected;
            }
            failure_reason = HistoricalMGBranchLabel(decision.branch);
            print_historical_bicgstab_convergence(
                "rejected", failure_reason, slice, rank, iterations,
                last_metrics, true_residual_assessment,
                last_true_eta_iteration,
                std::numeric_limits<double>::quiet_NaN(), cells);
            return HistoricalMGBreakdownResolution::Rejected;
        };

        auto const run_legacy_iterations = [&](int const first_iteration)
        {
        for (int i = first_iteration; initial_diagnostic_valid &&
             (i < max_iter ||
              (positivity_continuation.Active &&
               static_cast<size_t>(i) <
                   positivity_continuation.InitialIteration +
                       historical_mg_positivity_continuation_iteration_budget));
             i++) {
            // note: make sure matrix is big enough for the number of processors you are using!
            if(i > 1 && i % 50 == 0)
            {
                if(!sample_true_residual(static_cast<size_t>(i)))
                {
                    failure_reason = "diagnostic_eta_nonfinite";
                    break;
                }
                last_metrics = MeasureHistoricalMGImpl(
                    sub_x, old_x, sampled_true_residual, b, A_diag, slice,
                    &reduction_seconds, fused_reductions, nullptr, 0,
                    &reduction_collective_calls, &reduction_causal_rounds);
                update_legacy_metrics();
                HistoricalMGDecision const sampled_decision =
                    ClassifyHistoricalMGCorrectionCandidate(last_metrics,
                        static_cast<size_t>(i - 1), tolerance);
                if(sampled_decision.reject)
                {
                    failure_reason = HistoricalMGBranchLabel(
                        sampled_decision.branch);
                    break;
                }
                if(sampled_decision.accept)
                {
                    HistoricalMGCorrectionDisposition const disposition =
                        attempt_finalize_success(
                            sampled_decision, static_cast<size_t>(i));
                    if(disposition ==
                       HistoricalMGCorrectionDisposition::DeferPositivity) {
                        restart_from_true_residual();
                        continue;
                    }
                    if(disposition != HistoricalMGCorrectionDisposition::
                           ContinuePositivity) {
                        good_end = disposition ==
                            HistoricalMGCorrectionDisposition::Commit;
                        break;
                    }
                }
                sub_r = sampled_true_residual;
                sub_r0 = sub_r;
                sub_p = sub_r;
                if(!fused_reductions)
                    rho0 = timed_dot_product(sub_r0, sub_r);
                vector_rescale(sub_r, M, scratch_rescale1);
                if(fused_reductions)
                {
                    std::array<double, 2> const restart_dots =
                        timed_dot_product_pair(
                            sub_r0, sub_r, scratch_rescale1, sub_r);
                    rho0 = restart_dots[0];
                    sub_r_sqrd = restart_dots[1];
                }
                else
                    sub_r_sqrd = timed_dot_product(
                        scratch_rescale1, sub_r);
            }
            if(!std::isfinite(rho0)) {
                HistoricalMGBreakdownResolution const resolution =
                    finish_breakdown(
                    HistoricalMGBreakdown::NonFinite,
                    static_cast<size_t>(i));
                if(resolution == HistoricalMGBreakdownResolution::Restart)
                    continue;
                good_end = resolution ==
                    HistoricalMGBreakdownResolution::Converged;
                break;
            }
            if(std::abs(rho0) < std::numeric_limits<double>::min()*1e100){
                HistoricalMGBreakdownResolution const resolution =
                    finish_breakdown(
                    HistoricalMGBreakdown::TinyRho,
                    static_cast<size_t>(i));
                if(resolution == HistoricalMGBreakdownResolution::Restart)
                    continue;
                good_end = resolution ==
                    HistoricalMGBreakdownResolution::Converged;
                break;
            }
            if(print)
                std::clog<<"iter "<<i<<" rho0 "<<rho0<<" sub_r_sqrd "<<sub_r_sqrd<<std::endl;
            // r_old = sub_r;                 // Store previous residual
            sub_r_sqrd_old = sub_r_sqrd;  // save a recalculation of r_old^2 later
            if(!std::isfinite(sub_r_sqrd_old)) {
                HistoricalMGBreakdownResolution const resolution =
                    finish_breakdown(
                    HistoricalMGBreakdown::NonFinite,
                    static_cast<size_t>(i));
                if(resolution == HistoricalMGBreakdownResolution::Restart)
                    continue;
                good_end = resolution ==
                    HistoricalMGBreakdownResolution::Converged;
                break;
            }
            if(std::min(sub_r_sqrd_old, std::abs(rho0)) <
               std::numeric_limits<double>::min()*1e100){
                HistoricalMGBreakdownResolution const resolution =
                    finish_breakdown(
                    HistoricalMGBreakdown::TinyRho,
                    static_cast<size_t>(i));
                if(resolution == HistoricalMGBreakdownResolution::Restart)
                    continue;
                good_end = resolution ==
                    HistoricalMGBreakdownResolution::Converged;
                break;
            }
            max_data[2].mpi_id = rank;
            max_data[0].mpi_id = rank;
            apply_direction_preconditioner(sub_p, y, v);
            timed_exchange(y);
            matvec(y, v);
            y.resize(Nlocal);
            double const sub_r0_v = timed_dot_product(sub_r0, v);
            if(!std::isfinite(sub_r0_v))
            {
                HistoricalMGBreakdownResolution const resolution =
                    finish_breakdown(
                    HistoricalMGBreakdown::NonFinite,
                    static_cast<size_t>(i));
                if(resolution == HistoricalMGBreakdownResolution::Restart)
                    continue;
                good_end = resolution ==
                    HistoricalMGBreakdownResolution::Converged;
                break;
            }
            double const alpha = std::abs(sub_r0_v) < std::numeric_limits<double>::min()*1e100 ? 0.0 : rho0 / sub_r0_v;
            if(!std::isfinite(alpha))
            {
                HistoricalMGBreakdownResolution const resolution =
                    finish_breakdown(
                    HistoricalMGBreakdown::NonFinite,
                    static_cast<size_t>(i));
                if(resolution == HistoricalMGBreakdownResolution::Restart)
                    continue;
                good_end = resolution ==
                    HistoricalMGBreakdownResolution::Converged;
                break;
            }
            if(print)
                std::clog<<"alpha "<<alpha<<" sub_r0_v "<<sub_r0_v<<std::endl;
            vec_lin_combo(1.0, sub_x, alpha, y, h);
            vec_lin_combo(1.0, sub_r, -alpha, v, s);
            old_x = sub_x;
            sub_x = h;
            completed_iterations = static_cast<size_t>(i + 1);
            last_metrics = MeasureHistoricalMGImpl(
                sub_x, old_x, s, b, A_diag, slice, &reduction_seconds,
                fused_reductions, nullptr, 0, &reduction_collective_calls,
                &reduction_causal_rounds);
            update_legacy_metrics();
            HistoricalMGDecision intermediate_decision =
                ClassifyHistoricalMGCorrectionCandidate(
                    last_metrics, static_cast<size_t>(i), tolerance);
            if(intermediate_decision.reject)
            {
                failure_reason = HistoricalMGBranchLabel(
                    intermediate_decision.branch);
                break;
            }
            if(intermediate_decision.accept)
            {
                size_t const iterations = static_cast<size_t>(i + 1);
                if(!sample_true_residual(iterations))
                {
                    failure_reason = "diagnostic_eta_nonfinite";
                    break;
                }
                last_metrics = MeasureHistoricalMGImpl(
                    sub_x, old_x, sampled_true_residual, b, A_diag, slice,
                    &reduction_seconds, fused_reductions, nullptr, 0,
                    &reduction_collective_calls, &reduction_causal_rounds);
                update_legacy_metrics();
                intermediate_decision =
                    ClassifyHistoricalMGCorrectionCandidate(
                    last_metrics, static_cast<size_t>(i), tolerance);
                if(intermediate_decision.reject)
                {
                    failure_reason = HistoricalMGBranchLabel(
                        intermediate_decision.branch);
                    break;
                }
                if(intermediate_decision.accept)
                {
                    HistoricalMGCorrectionDisposition const disposition =
                        attempt_finalize_success(
                            intermediate_decision, iterations);
                    if(disposition ==
                       HistoricalMGCorrectionDisposition::DeferPositivity) {
                        restart_from_true_residual();
                        continue;
                    }
                    if(disposition != HistoricalMGCorrectionDisposition::
                           ContinuePositivity) {
                        good_end = disposition ==
                            HistoricalMGCorrectionDisposition::Commit;
                        break;
                    }
                }
            }
            apply_direction_preconditioner(s, z, t);
            timed_exchange(z);
            matvec(z, t);
            z.resize(Nlocal);
            vector_rescale(t, M, scratch_rescale1);
            vector_rescale(s, M, scratch_rescale2);
            double up = 0;
            double down = 0;
            if(fused_reductions)
            {
                std::array<double, 2> const omega_dots =
                    timed_dot_product_pair(
                        scratch_rescale1, scratch_rescale2,
                        scratch_rescale1, scratch_rescale1);
                up = omega_dots[0];
                down = omega_dots[1];
            }
            else
            {
                up = timed_dot_product(
                    scratch_rescale1, scratch_rescale2);
                down = timed_dot_product(
                    scratch_rescale1, scratch_rescale1);
            }
            if(!std::isfinite(up) || !std::isfinite(down))
            {
                HistoricalMGBreakdownResolution const resolution =
                    finish_breakdown(
                    HistoricalMGBreakdown::NonFinite,
                    static_cast<size_t>(i));
                if(resolution == HistoricalMGBreakdownResolution::Restart)
                    continue;
                good_end = resolution ==
                    HistoricalMGBreakdownResolution::Converged;
                break;
            }
            double w = std::abs(down) < std::numeric_limits<double>::min()*1e100 ? 0.0 : up / down;
            if(!std::isfinite(w))
            {
                HistoricalMGBreakdownResolution const resolution =
                    finish_breakdown(
                    HistoricalMGBreakdown::NonFinite,
                    static_cast<size_t>(i));
                if(resolution == HistoricalMGBreakdownResolution::Restart)
                    continue;
                good_end = resolution ==
                    HistoricalMGBreakdownResolution::Converged;
                break;
            }
            if(print)
                std::clog<<"w "<<w<<" up "<<up<<" down "<<down<<std::endl;
            if(std::abs(alpha) < std::numeric_limits<double>::min()*1e100 && std::abs(w) < std::numeric_limits<double>::min()*1e100)
            {
                HistoricalMGBreakdownResolution const resolution =
                    finish_breakdown(
                    HistoricalMGBreakdown::TinyAlphaOmega,
                    static_cast<size_t>(i));
                if(resolution == HistoricalMGBreakdownResolution::Restart)
                    continue;
                good_end = resolution ==
                    HistoricalMGBreakdownResolution::Converged;
                break;
            }
            if(std::abs(w) <= std::numeric_limits<double>::min())
            {
                HistoricalMGBreakdownResolution const resolution =
                    finish_breakdown(
                    HistoricalMGBreakdown::TinyAlphaOmega,
                    static_cast<size_t>(i));
                if(resolution == HistoricalMGBreakdownResolution::Restart)
                    continue;
                good_end = resolution ==
                    HistoricalMGBreakdownResolution::Converged;
                break;
            }
            vec_lin_combo(1.0, h, w, z, sub_x);
            vec_lin_combo(1.0, s, -w, t, sub_r);
            completed_iterations = static_cast<size_t>(i + 1);
            vector_rescale(sub_r, M, scratch_rescale1);
            std::array<double, 2> residual_dots = {{0, 0}};
            double rho0_new = 0;
            if(fused_reductions)
                residual_dots = timed_local_dot_product_pair(
                    scratch_rescale1, sub_r, sub_r0, sub_r);
            else
                sub_r_sqrd = timed_dot_product(scratch_rescale1, sub_r);
            last_metrics = MeasureHistoricalMGImpl(
                sub_x, old_x, sub_r, b, A_diag, slice,
                &reduction_seconds, fused_reductions,
                // Keep the two solver sums as an independent MPI_SUM request
                // in the first diagnostic phase; no recurrence is changed.
                fused_reductions ? residual_dots.data() : nullptr,
                fused_reductions ? residual_dots.size() : 0,
                &reduction_collective_calls, &reduction_causal_rounds);
            if(fused_reductions)
            {
                sub_r_sqrd = residual_dots[0];
                rho0_new = residual_dots[1];
            }
            update_legacy_metrics();
            if(print)
                std::clog<<"iter "<<i<<" error "<<error<<std::endl;
            HistoricalMGDecision decision =
                ClassifyHistoricalMGCorrectionCandidate(
                    last_metrics, static_cast<size_t>(i), tolerance);
            if(decision.reject)
            {
                failure_reason = HistoricalMGBranchLabel(decision.branch);
                break;
            }
            bool const positivity_block_boundary =
                HistoricalMGPositivityContinuationBlockBoundaryReached(
                    positivity_continuation,
                    static_cast<size_t>(i + 1));
            bool const positivity_budget_reached =
                HistoricalMGPositivityContinuationBudgetReached(
                    positivity_continuation,
                    static_cast<size_t>(i + 1));
            if(decision.accept || positivity_block_boundary)
            {
                size_t const iterations = static_cast<size_t>(i + 1);
                if(!sample_true_residual(iterations))
                {
                    failure_reason = "diagnostic_eta_nonfinite";
                    break;
                }
                last_metrics = MeasureHistoricalMGImpl(
                    sub_x, old_x, sampled_true_residual, b, A_diag, slice,
                    &reduction_seconds, fused_reductions, nullptr, 0,
                    &reduction_collective_calls, &reduction_causal_rounds);
                update_legacy_metrics();
                decision = ClassifyHistoricalMGCorrectionCandidate(
                    last_metrics, static_cast<size_t>(i), tolerance);
                if(decision.reject)
                {
                    failure_reason = HistoricalMGBranchLabel(
                        decision.branch);
                    break;
                }
                if(ShouldRestartHistoricalMGPositivityContinuation(
                       decision.accept, positivity_block_boundary,
                       positivity_budget_reached)) {
                    positivity_continuation.AdditionalIterationsUsed =
                        iterations - positivity_continuation.InitialIteration;
                    positivity_continuation.BlocksCompleted = std::min(
                        historical_mg_positivity_continuation_maximum_blocks,
                        positivity_continuation.AdditionalIterationsUsed /
                            historical_mg_positivity_continuation_block_iterations);
                    positivity_continuation.BlocksStarted = std::min(
                        historical_mg_positivity_continuation_maximum_blocks,
                        positivity_continuation.BlocksCompleted + 1);
                    restart_from_true_residual();
                    continue;
                }
                if(ShouldAttemptHistoricalMGPositivityFinalization(
                       decision.accept, positivity_block_boundary,
                       positivity_budget_reached))
                {
                    HistoricalMGCorrectionDisposition const disposition =
                        attempt_finalize_success(decision, iterations);
                    if(disposition ==
                       HistoricalMGCorrectionDisposition::DeferPositivity) {
                        restart_from_true_residual();
                        continue;
                    }
                    if(disposition != HistoricalMGCorrectionDisposition::
                           ContinuePositivity) {
                        good_end = disposition ==
                            HistoricalMGCorrectionDisposition::Commit;
                        break;
                    }
                }
            }
            if(!fused_reductions)
                rho0_new = timed_dot_product(sub_r0, sub_r);
            double const beta = rho0_new * alpha / (w * rho0);
            if(!std::isfinite(rho0_new) || !std::isfinite(beta))
            {
                HistoricalMGBreakdownResolution const resolution =
                    finish_breakdown(
                    HistoricalMGBreakdown::NonFinite,
                    static_cast<size_t>(i));
                if(resolution == HistoricalMGBreakdownResolution::Restart)
                    continue;
                good_end = resolution ==
                    HistoricalMGBreakdownResolution::Converged;
                break;
            }
            if(print)
                std::clog<<"beta "<<beta<<" rho0_new "<<rho0_new<<" rho0 "<<rho0<<" w "<<w<<" alpha "<<alpha<<std::endl;
            vec_lin_combo(1.0, sub_p, -w, v, t);
            vec_lin_combo(1.0, sub_r, beta, t, sub_p); 
            rho0 = rho0_new;
        }
        };

        if(!three_round_pipeline)
            run_legacy_iterations(0);
#ifdef RICH_MPI
        else
        {
            // Steady-state dependency schedule:
            //   round 1: alpha and the previous full-step phase B;
            //   round 2: omega and the current half-step phase A;
            //   round 3: exact rho/r^T M r, current half-step phase B, and
            //            current full-step phase A.
            // The previous full-step decision is resolved before h/old_x are
            // reused. Candidate, drain, and breakdown paths intentionally
            // retain the legacy extra reductions.
            enum class PipelineResolution
            {
                Continue,
                Restart,
                Stop,
                FailClosed
            };

            auto const complete_pipeline_round = [&]
                (MPI_Request* const requests, int const request_count,
                 std::size_t const round_index, bool const drain,
                 std::chrono::steady_clock::time_point const round_start)
            {
                auto const wait_start = std::chrono::steady_clock::now();
                RequireCGMpiSuccess(
                    MPI_Waitall(request_count, requests,
                                MPI_STATUSES_IGNORE),
                    "MPI_Waitall(three-round BiCGSTAB pipeline)");
                double const wait_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - wait_start).count();
                if(round_index < 3)
                {
                    three_round_pipeline_wait_seconds[round_index] +=
                        wait_seconds;
                    ++three_round_pipeline_round_counts[round_index];
                }
                if(drain)
                    ++three_round_pipeline_drain_rounds;
                three_round_pipeline_collective_calls +=
                    static_cast<unsigned long long>(request_count);
                RecordReductionBatch(
                    &reduction_collective_calls, &reduction_causal_rounds,
                    static_cast<unsigned long long>(request_count), 1);
                reduction_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - round_start).count();
            };

            auto const reduce_pipeline_phase_b = [&]
                (HistoricalMGPipelinePhaseB& phase_b)
            {
                MPI_Request requests[2];
                auto const round_start = std::chrono::steady_clock::now();
                RequireCGMpiSuccess(
                    MPI_Iallreduce(
                        MPI_IN_PLACE, phase_b.maxima, 2, MPI_DOUBLE_INT,
                        MPI_MAXLOC, MPI_COMM_WORLD, &requests[0]),
                    "MPI_Iallreduce(three-round pipeline drain maxima)");
                RequireCGMpiSuccess(
                    MPI_Iallreduce(
                        MPI_IN_PLACE, &phase_b.negative_rank, 1, MPI_INT,
                        MPI_MIN, MPI_COMM_WORLD, &requests[1]),
                    "MPI_Iallreduce(three-round pipeline drain negative rank)");
                complete_pipeline_round(
                    requests, 2, 3, true, round_start);
            };

            auto const validate_pipeline_metrics = [&]
                (char const* const phase_label,
                 std::size_t const iterations,
                 std::vector<double> const& solution,
                 std::vector<double> const& previous_solution,
                 std::vector<double> const& residual,
                 HistoricalMGMetrics& metrics)
            {
                if(!three_round_pipeline_shadow)
                    return true;
                ++three_round_pipeline_shadow_checks;
                HistoricalMGMetrics const legacy_metrics =
                    MeasureHistoricalMGImpl(
                        solution, previous_solution, residual, b, A_diag,
                        slice, &reduction_seconds, true, nullptr, 0,
                        &reduction_collective_calls,
                        &reduction_causal_rounds);
                int mismatch =
                    HistoricalMGPipelineMetricsBitwiseEqual(
                        metrics, legacy_metrics) ? 0 : 1;
                auto const reduction_start =
                    std::chrono::steady_clock::now();
                RequireCGMpiSuccess(
                    MPI_Allreduce(MPI_IN_PLACE, &mismatch, 1, MPI_INT,
                                  MPI_MAX, MPI_COMM_WORLD),
                    "MPI_Allreduce(three-round pipeline shadow mismatch)");
                RecordReductionBatch(
                    &reduction_collective_calls, &reduction_causal_rounds,
                    1, 1);
                reduction_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    reduction_start).count();
                metrics = legacy_metrics;
                if(mismatch != 0)
                {
                    ++three_round_pipeline_shadow_mismatches;
                    if(rank == 0)
                        std::clog
                            << "MG_THREE_ROUND_DIAGNOSTIC_PIPELINE_SHADOW"
                            << " outcome=mismatch phase=" << phase_label
                            << " iterations=" << iterations << std::endl;
                }
                return mismatch == 0;
            };

            auto const resolve_pipeline_full_candidate = [&]
                (HistoricalMGMetrics const& pipeline_metrics,
                 std::size_t const zero_based_iteration,
                 bool const positivity_block_boundary,
                 bool const positivity_budget_reached)
            {
                std::size_t const iterations = zero_based_iteration + 1;
                completed_iterations = iterations;
                last_metrics = pipeline_metrics;
                update_legacy_metrics();
                HistoricalMGDecision decision =
                    ClassifyHistoricalMGCorrectionCandidate(
                        last_metrics, zero_based_iteration, tolerance);
                if(decision.reject)
                {
                    failure_reason = HistoricalMGBranchLabel(
                        decision.branch);
                    return PipelineResolution::Stop;
                }
                if(decision.accept || positivity_block_boundary)
                {
                    if(!sample_true_residual(iterations))
                    {
                        failure_reason = "diagnostic_eta_nonfinite";
                        return PipelineResolution::Stop;
                    }
                    last_metrics = MeasureHistoricalMGImpl(
                        sub_x, old_x, sampled_true_residual, b, A_diag,
                        slice, &reduction_seconds, true, nullptr, 0,
                        &reduction_collective_calls,
                        &reduction_causal_rounds);
                    update_legacy_metrics();
                    decision = ClassifyHistoricalMGCorrectionCandidate(
                        last_metrics, zero_based_iteration, tolerance);
                    if(decision.reject)
                    {
                        failure_reason = HistoricalMGBranchLabel(
                            decision.branch);
                        return PipelineResolution::Stop;
                    }
                    if(ShouldRestartHistoricalMGPositivityContinuation(
                           decision.accept, positivity_block_boundary,
                           positivity_budget_reached))
                    {
                        positivity_continuation.AdditionalIterationsUsed =
                            iterations -
                            positivity_continuation.InitialIteration;
                        positivity_continuation.BlocksCompleted = std::min(
                            historical_mg_positivity_continuation_maximum_blocks,
                            positivity_continuation.AdditionalIterationsUsed /
                                historical_mg_positivity_continuation_block_iterations);
                        positivity_continuation.BlocksStarted = std::min(
                            historical_mg_positivity_continuation_maximum_blocks,
                            positivity_continuation.BlocksCompleted + 1);
                        restart_from_true_residual();
                        return PipelineResolution::Restart;
                    }
                    if(ShouldAttemptHistoricalMGPositivityFinalization(
                           decision.accept, positivity_block_boundary,
                           positivity_budget_reached))
                    {
                        HistoricalMGCorrectionDisposition const disposition =
                            attempt_finalize_success(decision, iterations);
                        if(disposition ==
                           HistoricalMGCorrectionDisposition::DeferPositivity)
                        {
                            restart_from_true_residual();
                            return PipelineResolution::Restart;
                        }
                        if(disposition !=
                           HistoricalMGCorrectionDisposition::
                               ContinuePositivity)
                        {
                            good_end = disposition ==
                                HistoricalMGCorrectionDisposition::Commit;
                            return PipelineResolution::Stop;
                        }
                    }
                }
                return PipelineResolution::Continue;
            };

            auto const resolve_pipeline_half_candidate = [&]
                (HistoricalMGMetrics const& pipeline_metrics,
                 std::size_t const zero_based_iteration)
            {
                std::size_t const iterations = zero_based_iteration + 1;
                completed_iterations = iterations;
                last_metrics = pipeline_metrics;
                update_legacy_metrics();
                HistoricalMGDecision intermediate_decision =
                    ClassifyHistoricalMGCorrectionCandidate(
                        last_metrics, zero_based_iteration, tolerance);
                if(intermediate_decision.reject)
                {
                    failure_reason = HistoricalMGBranchLabel(
                        intermediate_decision.branch);
                    sub_x = h;
                    return PipelineResolution::Stop;
                }
                if(!intermediate_decision.accept)
                    return PipelineResolution::Continue;

                scratch_rescale2.swap(sub_x);
                sub_x = h;
                if(!sample_true_residual(iterations))
                {
                    failure_reason = "diagnostic_eta_nonfinite";
                    return PipelineResolution::Stop;
                }
                last_metrics = MeasureHistoricalMGImpl(
                    sub_x, old_x, sampled_true_residual, b, A_diag, slice,
                    &reduction_seconds, true, nullptr, 0,
                    &reduction_collective_calls,
                    &reduction_causal_rounds);
                update_legacy_metrics();
                intermediate_decision =
                    ClassifyHistoricalMGCorrectionCandidate(
                        last_metrics, zero_based_iteration, tolerance);
                if(intermediate_decision.reject)
                {
                    failure_reason = HistoricalMGBranchLabel(
                        intermediate_decision.branch);
                    return PipelineResolution::Stop;
                }
                if(intermediate_decision.accept)
                {
                    HistoricalMGCorrectionDisposition const disposition =
                        attempt_finalize_success(
                            intermediate_decision, iterations);
                    if(disposition ==
                       HistoricalMGCorrectionDisposition::DeferPositivity)
                    {
                        restart_from_true_residual();
                        return PipelineResolution::Restart;
                    }
                    if(disposition !=
                       HistoricalMGCorrectionDisposition::ContinuePositivity)
                    {
                        good_end = disposition ==
                            HistoricalMGCorrectionDisposition::Commit;
                        return PipelineResolution::Stop;
                    }
                }
                sub_x.swap(scratch_rescale2);
                return PipelineResolution::Continue;
            };

            bool pending_full = false;
            HistoricalMGPipelinePhaseA pending_full_phase_a;
            HistoricalMGPipelinePhaseB pending_full_phase_b;
            std::size_t pending_full_zero_based_iteration = 0;
            bool pending_full_positivity_block_boundary = false;
            bool pending_full_positivity_budget_reached = false;
            bool pipeline_finished = !initial_diagnostic_valid;
            bool pipeline_fail_closed = false;
            int pipeline_legacy_start = 0;
            int i = 0;

            auto const fail_closed_to_legacy = [&]
                (int const restart_iteration,
                 std::size_t const residual_iteration)
            {
                ++three_round_pipeline_fail_closed_restarts;
                if(!sample_true_residual(residual_iteration))
                {
                    failure_reason =
                        "three_round_pipeline_fail_closed_residual_nonfinite";
                    pipeline_finished = true;
                    return;
                }
                restart_from_true_residual();
                pending_full = false;
                pipeline_fail_closed = true;
                pipeline_legacy_start = restart_iteration;
                pipeline_finished = true;
                if(rank == 0)
                    std::clog
                        << "MG_THREE_ROUND_DIAGNOSTIC_PIPELINE"
                        << " outcome=fail_closed_restart"
                        << " restart_iteration=" << restart_iteration
                        << " residual_iteration=" << residual_iteration
                        << std::endl;
            };

            auto const resolve_pending_full = [&]
                (int const fail_closed_restart_iteration)
            {
                HistoricalMGMetrics metrics =
                    CompleteHistoricalMGPipelinePhaseB(
                        pending_full_phase_a, pending_full_phase_b, rank);
                if(!validate_pipeline_metrics(
                       "full", pending_full_zero_based_iteration + 1,
                       sub_x, old_x, sub_r, metrics))
                {
                    fail_closed_to_legacy(
                        fail_closed_restart_iteration,
                        pending_full_zero_based_iteration + 1);
                    return PipelineResolution::FailClosed;
                }
                PipelineResolution const resolution =
                    resolve_pipeline_full_candidate(
                        metrics, pending_full_zero_based_iteration,
                        pending_full_positivity_block_boundary,
                        pending_full_positivity_budget_reached);
                pending_full = false;
                return resolution;
            };

            auto const resolve_pipeline_start_breakdown = [&]
                (int const iteration)
            {
                HistoricalMGBreakdown breakdown =
                    HistoricalMGBreakdown::None;
                if(!std::isfinite(rho0) ||
                   !std::isfinite(sub_r_sqrd))
                    breakdown = HistoricalMGBreakdown::NonFinite;
                else if(std::abs(rho0) <
                            std::numeric_limits<double>::min() * 1e100 ||
                        std::min(sub_r_sqrd, std::abs(rho0)) <
                            std::numeric_limits<double>::min() * 1e100)
                    breakdown = HistoricalMGBreakdown::TinyRho;
                if(breakdown == HistoricalMGBreakdown::None)
                    return PipelineResolution::Continue;
                HistoricalMGBreakdownResolution const resolution =
                    finish_breakdown(
                        breakdown, static_cast<size_t>(iteration));
                if(resolution == HistoricalMGBreakdownResolution::Restart)
                    return PipelineResolution::Restart;
                good_end = resolution ==
                    HistoricalMGBreakdownResolution::Converged;
                return PipelineResolution::Stop;
            };

            while(!pipeline_finished)
            {
                bool iteration_allowed =
                    i < max_iter ||
                    (positivity_continuation.Active &&
                     static_cast<size_t>(i) <
                         positivity_continuation.InitialIteration +
                             historical_mg_positivity_continuation_iteration_budget);
                bool const periodic_true_residual = i > 1 && i % 50 == 0;
                if(pending_full &&
                   (!iteration_allowed || periodic_true_residual))
                {
                    reduce_pipeline_phase_b(pending_full_phase_b);
                    PipelineResolution const resolution =
                        resolve_pending_full(i);
                    if(resolution == PipelineResolution::Stop ||
                       resolution == PipelineResolution::FailClosed)
                    {
                        pipeline_finished = true;
                        continue;
                    }
                    if(resolution == PipelineResolution::Restart)
                        continue;
                    iteration_allowed =
                        i < max_iter ||
                        (positivity_continuation.Active &&
                         static_cast<size_t>(i) <
                             positivity_continuation.InitialIteration +
                                 historical_mg_positivity_continuation_iteration_budget);
                }
                if(!iteration_allowed)
                    break;

                if(periodic_true_residual)
                {
                    if(!sample_true_residual(static_cast<size_t>(i)))
                    {
                        failure_reason = "diagnostic_eta_nonfinite";
                        break;
                    }
                    last_metrics = MeasureHistoricalMGImpl(
                        sub_x, old_x, sampled_true_residual, b, A_diag,
                        slice, &reduction_seconds, true, nullptr, 0,
                        &reduction_collective_calls,
                        &reduction_causal_rounds);
                    update_legacy_metrics();
                    HistoricalMGDecision const sampled_decision =
                        ClassifyHistoricalMGCorrectionCandidate(
                            last_metrics, static_cast<size_t>(i - 1),
                            tolerance);
                    if(sampled_decision.reject)
                    {
                        failure_reason = HistoricalMGBranchLabel(
                            sampled_decision.branch);
                        break;
                    }
                    if(sampled_decision.accept)
                    {
                        HistoricalMGCorrectionDisposition const disposition =
                            attempt_finalize_success(
                                sampled_decision, static_cast<size_t>(i));
                        if(disposition ==
                           HistoricalMGCorrectionDisposition::DeferPositivity)
                        {
                            restart_from_true_residual();
                            ++i;
                            continue;
                        }
                        if(disposition !=
                           HistoricalMGCorrectionDisposition::
                               ContinuePositivity)
                        {
                            good_end = disposition ==
                                HistoricalMGCorrectionDisposition::Commit;
                            break;
                        }
                    }
                    restart_from_true_residual();
                }

                if(!pending_full)
                {
                    PipelineResolution const start_resolution =
                        resolve_pipeline_start_breakdown(i);
                    if(start_resolution == PipelineResolution::Restart)
                    {
                        ++i;
                        continue;
                    }
                    if(start_resolution == PipelineResolution::Stop)
                        break;
                }

                apply_direction_preconditioner(sub_p, y, v);
                timed_exchange(y);
                matvec(y, v);
                y.resize(Nlocal);
                if(pending_full)
                    ++three_round_pipeline_speculative_matvec_calls;
                double sub_r0_v = local_dot_product(sub_r0, v);
                MPI_Request round_one_requests[3];
                int round_one_request_count = 0;
                auto const round_one_start =
                    std::chrono::steady_clock::now();
                RequireCGMpiSuccess(
                    MPI_Iallreduce(
                        MPI_IN_PLACE, &sub_r0_v, 1, MPI_DOUBLE, MPI_SUM,
                        MPI_COMM_WORLD,
                        &round_one_requests[round_one_request_count++]),
                    "MPI_Iallreduce(three-round pipeline alpha)");
                if(pending_full)
                {
                    RequireCGMpiSuccess(
                        MPI_Iallreduce(
                            MPI_IN_PLACE, pending_full_phase_b.maxima, 2,
                            MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD,
                            &round_one_requests[
                                round_one_request_count++]),
                        "MPI_Iallreduce(three-round pipeline prior full maxima)");
                    RequireCGMpiSuccess(
                        MPI_Iallreduce(
                            MPI_IN_PLACE,
                            &pending_full_phase_b.negative_rank, 1,
                            MPI_INT, MPI_MIN, MPI_COMM_WORLD,
                            &round_one_requests[
                                round_one_request_count++]),
                        "MPI_Iallreduce(three-round pipeline prior full negative rank)");
                }
                complete_pipeline_round(
                    round_one_requests, round_one_request_count, 0, false,
                    round_one_start);

                if(pending_full)
                {
                    PipelineResolution const pending_resolution =
                        resolve_pending_full(i);
                    if(pending_resolution == PipelineResolution::Stop ||
                       pending_resolution ==
                           PipelineResolution::FailClosed)
                    {
                        ++three_round_pipeline_discarded_speculative_matvec_calls;
                        pipeline_finished = true;
                        continue;
                    }
                    if(pending_resolution == PipelineResolution::Restart)
                    {
                        ++three_round_pipeline_discarded_speculative_matvec_calls;
                        continue;
                    }
                }

                PipelineResolution const start_resolution =
                    resolve_pipeline_start_breakdown(i);
                if(start_resolution == PipelineResolution::Restart)
                {
                    ++three_round_pipeline_discarded_speculative_matvec_calls;
                    ++i;
                    continue;
                }
                if(start_resolution == PipelineResolution::Stop)
                {
                    ++three_round_pipeline_discarded_speculative_matvec_calls;
                    break;
                }

                if(!std::isfinite(sub_r0_v))
                {
                    HistoricalMGBreakdownResolution const resolution =
                        finish_breakdown(
                            HistoricalMGBreakdown::NonFinite,
                            static_cast<size_t>(i));
                    if(resolution ==
                       HistoricalMGBreakdownResolution::Restart)
                    {
                        ++i;
                        continue;
                    }
                    good_end = resolution ==
                        HistoricalMGBreakdownResolution::Converged;
                    break;
                }
                double const alpha = std::abs(sub_r0_v) <
                    std::numeric_limits<double>::min() * 1e100 ?
                    0.0 : rho0 / sub_r0_v;
                if(!std::isfinite(alpha))
                {
                    HistoricalMGBreakdownResolution const resolution =
                        finish_breakdown(
                            HistoricalMGBreakdown::NonFinite,
                            static_cast<size_t>(i));
                    if(resolution ==
                       HistoricalMGBreakdownResolution::Restart)
                    {
                        ++i;
                        continue;
                    }
                    good_end = resolution ==
                        HistoricalMGBreakdownResolution::Converged;
                    break;
                }

                old_x = sub_x;
                vec_lin_combo(1.0, sub_x, alpha, y, h);
                vec_lin_combo(1.0, sub_r, -alpha, v, s);
                completed_iterations = static_cast<size_t>(i + 1);
                HistoricalMGPipelinePhaseA half_phase_a =
                    BuildHistoricalMGPipelinePhaseA(
                        h, old_x, s, b, A_diag, slice);

                apply_direction_preconditioner(s, z, t);
                timed_exchange(z);
                matvec(z, t);
                z.resize(Nlocal);
                ++three_round_pipeline_speculative_matvec_calls;
                vector_rescale(t, M, scratch_rescale1);
                vector_rescale(s, M, scratch_rescale2);
                double omega_dots[2] = {
                    local_dot_product(
                        scratch_rescale1, scratch_rescale2),
                    local_dot_product(
                        scratch_rescale1, scratch_rescale1)};
                MPI_Request round_two_requests[4];
                auto const round_two_start =
                    std::chrono::steady_clock::now();
                RequireCGMpiSuccess(
                    MPI_Iallreduce(
                        MPI_IN_PLACE, omega_dots, 2, MPI_DOUBLE, MPI_SUM,
                        MPI_COMM_WORLD, &round_two_requests[0]),
                    "MPI_Iallreduce(three-round pipeline omega)");
                RequireCGMpiSuccess(
                    MPI_Iallreduce(
                        MPI_IN_PLACE, &half_phase_a.valid, 1, MPI_INT,
                        MPI_MIN, MPI_COMM_WORLD, &round_two_requests[1]),
                    "MPI_Iallreduce(three-round pipeline half validity)");
                RequireCGMpiSuccess(
                    MPI_Iallreduce(
                        MPI_IN_PLACE, &half_phase_a.maximum_solution, 1,
                        MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD,
                        &round_two_requests[2]),
                    "MPI_Iallreduce(three-round pipeline half maximum)");
                RequireCGMpiSuccess(
                    MPI_Iallreduce(
                        MPI_IN_PLACE, half_phase_a.weighted, 2,
                        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD,
                        &round_two_requests[3]),
                    "MPI_Iallreduce(three-round pipeline half weighted norms)");
                complete_pipeline_round(
                    round_two_requests, 4, 1, false, round_two_start);
                CompleteHistoricalMGPipelinePhaseA(half_phase_a);

                double const up = omega_dots[0];
                double const down = omega_dots[1];
                double const w =
                    !std::isfinite(up) || !std::isfinite(down) ?
                    std::numeric_limits<double>::quiet_NaN() :
                    (std::abs(down) <
                         std::numeric_limits<double>::min() * 1e100 ?
                     0.0 : up / down);
                HistoricalMGPipelinePhaseB half_phase_b =
                    BuildHistoricalMGPipelinePhaseB(
                        h, old_x, s, A_diag, slice, half_phase_a, rank);
                HistoricalMGBreakdown omega_breakdown =
                    HistoricalMGBreakdown::None;
                if(!std::isfinite(up) || !std::isfinite(down) ||
                   !std::isfinite(w))
                    omega_breakdown = HistoricalMGBreakdown::NonFinite;
                else if((std::abs(alpha) <
                             std::numeric_limits<double>::min() * 1e100 &&
                         std::abs(w) <
                             std::numeric_limits<double>::min() * 1e100) ||
                        std::abs(w) <=
                            std::numeric_limits<double>::min())
                    omega_breakdown =
                        HistoricalMGBreakdown::TinyAlphaOmega;
                if(omega_breakdown != HistoricalMGBreakdown::None)
                {
                    reduce_pipeline_phase_b(half_phase_b);
                    HistoricalMGMetrics half_metrics =
                        CompleteHistoricalMGPipelinePhaseB(
                            half_phase_a, half_phase_b, rank);
                    if(!validate_pipeline_metrics(
                           "half_breakdown", static_cast<size_t>(i + 1),
                           h, old_x, s, half_metrics))
                    {
                        ++three_round_pipeline_discarded_speculative_matvec_calls;
                        sub_x = old_x;
                        fail_closed_to_legacy(i, static_cast<size_t>(i));
                        continue;
                    }
                    PipelineResolution const half_resolution =
                        resolve_pipeline_half_candidate(
                            half_metrics, static_cast<size_t>(i));
                    if(half_resolution == PipelineResolution::Stop)
                    {
                        ++three_round_pipeline_discarded_speculative_matvec_calls;
                        break;
                    }
                    if(half_resolution == PipelineResolution::Restart)
                    {
                        ++three_round_pipeline_discarded_speculative_matvec_calls;
                        ++i;
                        continue;
                    }
                    sub_x = h;
                    HistoricalMGBreakdownResolution const resolution =
                        finish_breakdown(
                            omega_breakdown,
                            static_cast<size_t>(i));
                    if(resolution ==
                       HistoricalMGBreakdownResolution::Restart)
                    {
                        ++i;
                        continue;
                    }
                    good_end = resolution ==
                        HistoricalMGBreakdownResolution::Converged;
                    break;
                }
                vec_lin_combo(1.0, h, w, z, sub_x);
                vec_lin_combo(1.0, s, -w, t, sub_r);
                vector_rescale(sub_r, M, scratch_rescale1);
                std::array<double, 2> residual_dots = {{
                    local_dot_product(scratch_rescale1, sub_r),
                    local_dot_product(sub_r0, sub_r)}};
                HistoricalMGPipelinePhaseA full_phase_a =
                    BuildHistoricalMGPipelinePhaseA(
                        sub_x, old_x, sub_r, b, A_diag, slice);

                MPI_Request round_three_requests[6];
                auto const round_three_start =
                    std::chrono::steady_clock::now();
                RequireCGMpiSuccess(
                    MPI_Iallreduce(
                        MPI_IN_PLACE, residual_dots.data(), 2,
                        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD,
                        &round_three_requests[0]),
                    "MPI_Iallreduce(three-round pipeline residual dots)");
                RequireCGMpiSuccess(
                    MPI_Iallreduce(
                        MPI_IN_PLACE, half_phase_b.maxima, 2,
                        MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD,
                        &round_three_requests[1]),
                    "MPI_Iallreduce(three-round pipeline half maxima)");
                RequireCGMpiSuccess(
                    MPI_Iallreduce(
                        MPI_IN_PLACE, &half_phase_b.negative_rank, 1,
                        MPI_INT, MPI_MIN, MPI_COMM_WORLD,
                        &round_three_requests[2]),
                    "MPI_Iallreduce(three-round pipeline half negative rank)");
                RequireCGMpiSuccess(
                    MPI_Iallreduce(
                        MPI_IN_PLACE, &full_phase_a.valid, 1, MPI_INT,
                        MPI_MIN, MPI_COMM_WORLD,
                        &round_three_requests[3]),
                    "MPI_Iallreduce(three-round pipeline full validity)");
                RequireCGMpiSuccess(
                    MPI_Iallreduce(
                        MPI_IN_PLACE, &full_phase_a.maximum_solution, 1,
                        MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD,
                        &round_three_requests[4]),
                    "MPI_Iallreduce(three-round pipeline full maximum)");
                RequireCGMpiSuccess(
                    MPI_Iallreduce(
                        MPI_IN_PLACE, full_phase_a.weighted, 2,
                        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD,
                        &round_three_requests[5]),
                    "MPI_Iallreduce(three-round pipeline full weighted norms)");
                complete_pipeline_round(
                    round_three_requests, 6, 2, false,
                    round_three_start);
                CompleteHistoricalMGPipelinePhaseA(full_phase_a);

                double const rho0_new = residual_dots[1];
                sub_r_sqrd = residual_dots[0];
                HistoricalMGMetrics half_metrics =
                    CompleteHistoricalMGPipelinePhaseB(
                        half_phase_a, half_phase_b, rank);
                if(!validate_pipeline_metrics(
                       "half", static_cast<size_t>(i + 1), h, old_x, s,
                       half_metrics))
                {
                    ++three_round_pipeline_discarded_speculative_matvec_calls;
                    sub_x = old_x;
                    fail_closed_to_legacy(i, static_cast<size_t>(i));
                    continue;
                }
                PipelineResolution const half_resolution =
                    resolve_pipeline_half_candidate(
                        half_metrics, static_cast<size_t>(i));
                if(half_resolution == PipelineResolution::Stop)
                {
                    ++three_round_pipeline_discarded_speculative_matvec_calls;
                    break;
                }
                if(half_resolution == PipelineResolution::Restart)
                {
                    ++three_round_pipeline_discarded_speculative_matvec_calls;
                    ++i;
                    continue;
                }

                pending_full_phase_a = full_phase_a;
                pending_full_phase_b = BuildHistoricalMGPipelinePhaseB(
                    sub_x, old_x, sub_r, A_diag, slice,
                    pending_full_phase_a, rank);
                pending_full_zero_based_iteration =
                    static_cast<size_t>(i);
                pending_full_positivity_block_boundary =
                    HistoricalMGPositivityContinuationBlockBoundaryReached(
                        positivity_continuation,
                        static_cast<size_t>(i + 1));
                pending_full_positivity_budget_reached =
                    HistoricalMGPositivityContinuationBudgetReached(
                        positivity_continuation,
                        static_cast<size_t>(i + 1));
                pending_full = true;

                double const beta = rho0_new * alpha / (w * rho0);
                if(!std::isfinite(rho0_new) || !std::isfinite(beta))
                {
                    reduce_pipeline_phase_b(pending_full_phase_b);
                    PipelineResolution const pending_resolution =
                        resolve_pending_full(i + 1);
                    if(pending_resolution == PipelineResolution::Stop ||
                       pending_resolution ==
                           PipelineResolution::FailClosed)
                    {
                        pipeline_finished = true;
                        continue;
                    }
                    if(pending_resolution == PipelineResolution::Restart)
                    {
                        ++i;
                        continue;
                    }
                    HistoricalMGBreakdownResolution const resolution =
                        finish_breakdown(
                            HistoricalMGBreakdown::NonFinite,
                            static_cast<size_t>(i));
                    if(resolution ==
                       HistoricalMGBreakdownResolution::Restart)
                    {
                        ++i;
                        continue;
                    }
                    good_end = resolution ==
                        HistoricalMGBreakdownResolution::Converged;
                    break;
                }
                vec_lin_combo(1.0, sub_p, -w, v, t);
                vec_lin_combo(1.0, sub_r, beta, t, sub_p);
                rho0 = rho0_new;
                ++i;
            }

            if(pipeline_fail_closed && !good_end)
                run_legacy_iterations(pipeline_legacy_start);
        }
#endif
        if(!good_end && positivity_continuation.Active &&
           workspace.historical_correction.failure_reason.empty()) {
            std::string const rescue_reason =
                std::string("positivity_rescue_solver_") + failure_reason;
            RecordHistoricalMGPositivityRescueFailure(
                positivity_continuation, workspace.historical_correction,
                completed_iterations, rescue_reason.c_str(),
                RadiationPositivity::SpectralRepairFailure::
                    PositivityRescueSolverFailure);
        }
        if(positivity_continuation.Active &&
           !positivity_continuation.Closed) {
            positivity_continuation.AdditionalIterationsUsed =
                completed_iterations >=
                    positivity_continuation.InitialIteration ?
                completed_iterations -
                    positivity_continuation.InitialIteration : 0;
            positivity_continuation.BlocksCompleted = std::min(
                historical_mg_positivity_continuation_maximum_blocks,
                (positivity_continuation.AdditionalIterationsUsed +
                 historical_mg_positivity_continuation_block_iterations - 1) /
                    historical_mg_positivity_continuation_block_iterations);
            ReportHistoricalMGPositivityContinuationClose(
                "global", positivity_continuation,
                positivity_continuation.LastNegativity, "solver_failure",
                rank == 0);
            RecordHistoricalMGPositivityContinuation(
                positivity_continuation, workspace.historical_correction);
        }
        if(not good_end)
        {
            total_iters = static_cast<int>(std::min<size_t>(
                completed_iterations,
                static_cast<size_t>(std::numeric_limits<int>::max())));
            print_historical_bicgstab_convergence(
                "rejected", failure_reason, slice, rank,
                completed_iterations, last_metrics,
                true_residual_assessment, last_true_eta_iteration,
                std::numeric_limits<double>::quiet_NaN(), cells);
            if(rank == 0)
                std::clog << "not good end, delta " << sub_r_sqrd
                          << " maxdata2 " << max_data[2].val
                          << " iterations " << total_iters
                          << " scale_b " << scale_b
                          << " reason " << failure_reason << std::endl;
            std::fill_n(sub_x.begin(), sub_x.size(), -1.0);
            // throw UniversalError("CG did not converge");
        }
        if(!good_end)
        {
        if(rank == 0)
            std::clog << "MG_BICGSTAB_HISTORICAL_POLICY scope=global"
                      << " squared_scaled_tolerance=" << tolerance
                      << " effective_norm_tolerance="
                      << std::sqrt(tolerance)
                      << " last_true_eta_inf="
                      << true_residual_assessment.eta_inf
                      << " last_true_eta_iteration="
                      << last_true_eta_iteration
                      << " last_true_eta_age="
                      << (completed_iterations >= last_true_eta_iteration ?
                          completed_iterations - last_true_eta_iteration : 0)
                      << " pre_correction_eta_inf="
                      << (good_end ? true_residual_assessment.eta_inf :
                          std::numeric_limits<double>::quiet_NaN())
                      << " final_eta_inf=not_evaluated"
                      << " eta_inf_role=diagnostic_only" << std::endl;
        }
        if(trace_matrix_build)
        {
        unsigned long long three_round_pipeline_stats[10] = {
            three_round_pipeline_round_counts[0],
            three_round_pipeline_round_counts[1],
            three_round_pipeline_round_counts[2],
            three_round_pipeline_collective_calls,
            three_round_pipeline_drain_rounds,
            three_round_pipeline_speculative_matvec_calls,
            three_round_pipeline_discarded_speculative_matvec_calls,
            three_round_pipeline_fail_closed_restarts,
            three_round_pipeline_shadow_checks,
            three_round_pipeline_shadow_mismatches};
        double three_round_pipeline_wait_max[3] = {
            three_round_pipeline_wait_seconds[0],
            three_round_pipeline_wait_seconds[1],
            three_round_pipeline_wait_seconds[2]};
#ifdef RICH_MPI
        if(three_round_pipeline)
        {
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, three_round_pipeline_stats, 10,
                              MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                              MPI_COMM_WORLD),
                "MPI_Allreduce(global three-round pipeline counters)");
            RequireCGMpiSuccess(
                MPI_Allreduce(MPI_IN_PLACE, three_round_pipeline_wait_max, 3,
                              MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
                "MPI_Allreduce(global three-round pipeline waits)");
        }
#endif
        if(rank == 0)
            std::clog << "MG_BICGSTAB_REDUCTIONS scope=global"
                      << " enabled=" << (fused_reductions ? 1 : 0)
                      << " route=" << (three_round_pipeline ?
                          "three_round_diagnostic_pipeline" : "legacy")
                      << " shadow="
                      << (three_round_pipeline_shadow ? 1 : 0)
                      << " tracked_scope=krylov_and_historical_diagnostics"
                      << " tracked_collective_calls="
                      << reduction_collective_calls
                      << " tracked_causal_rounds="
                      << reduction_causal_rounds
                      << " normal_full_iteration_collective_calls="
                      << (fused_reductions ? 13 : 17)
                      << " normal_full_iteration_causal_rounds="
                      << (three_round_pipeline ? 3 :
                          (fused_reductions ? 6 : 17))
                      << std::endl;
        if(rank == 0 && three_round_pipeline)
            std::clog
                << "MG_THREE_ROUND_DIAGNOSTIC_PIPELINE_RESULT"
                << " round1_count_max=" << three_round_pipeline_stats[0]
                << " round2_count_max=" << three_round_pipeline_stats[1]
                << " round3_count_max=" << three_round_pipeline_stats[2]
                << " pipeline_collective_calls_max="
                << three_round_pipeline_stats[3]
                << " drain_rounds_max=" << three_round_pipeline_stats[4]
                << " speculative_matvec_calls_max="
                << three_round_pipeline_stats[5]
                << " discarded_speculative_matvec_calls_max="
                << three_round_pipeline_stats[6]
                << " fail_closed_restarts_max="
                << three_round_pipeline_stats[7]
                << " shadow_checks_max="
                << three_round_pipeline_stats[8]
                << " shadow_mismatches_max="
                << three_round_pipeline_stats[9]
                << " round1_wait_seconds_max="
                << three_round_pipeline_wait_max[0]
                << " round2_wait_seconds_max="
                << three_round_pipeline_wait_max[1]
                << " round3_wait_seconds_max="
                << three_round_pipeline_wait_max[2]
                << std::endl;
        unsigned long long preconditioner_apply_calls =
            static_cast<unsigned long long>(
                direction_preconditioner.ApplyCalls());
        double preconditioner_apply_seconds =
            direction_preconditioner.ApplySeconds();
#ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &preconditioner_apply_calls, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD),
            "MPI_Allreduce(global preconditioner apply calls)");
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &preconditioner_apply_seconds, 1,
                          MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
            "MPI_Allreduce(global preconditioner apply time)");
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &preconditioner_applications, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD),
            "MPI_Allreduce(global preconditioner applications)");
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, &neighbor_correction_matvec_calls, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD),
            "MPI_Allreduce(global neighbor correction matvec calls)");
#endif
        if(rank == 0)
            std::clog << "MG_PRECONDITIONER_APPLY kind="
                      << PreconditionerKindLabel(
                             direction_preconditioner.Kind())
                      << " requested_kind="
                      << PreconditionerKindLabel(
                             requested_preconditioner_kind)
                      << " scope=global"
                      << " calls_max=" << preconditioner_apply_calls
                      << " applications_max=" << preconditioner_applications
                      << " neighbor_correction_matvec_calls_max="
                      << neighbor_correction_matvec_calls
                      << " seconds_max=" << preconditioner_apply_seconds
                      << std::endl;
        double bicgstab_timing[6] = {
            matvec_seconds,
            exchange_seconds,
            reduction_seconds,
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() -
                bicgstab_total_start).count(),
            csr_setup_seconds,
            preconditioner_setup_max};
#ifdef RICH_MPI
        RequireCGMpiSuccess(
            MPI_Allreduce(MPI_IN_PLACE, bicgstab_timing, 6, MPI_DOUBLE,
                          MPI_MAX, MPI_COMM_WORLD),
            "MPI_Allreduce(global BiCGSTAB timing)");
#endif
        if(rank == 0)
            std::clog << "MG_BICGSTAB_TIMING scope=global"
                      << " outcome=" << (good_end ? "converged" : "failed")
                      << " iterations=" << total_iters
                      << " matvec_seconds_max=" << bicgstab_timing[0]
                      << " exchange_seconds_max=" << bicgstab_timing[1]
                      << " reduction_seconds_max=" << bicgstab_timing[2]
                      << " csr_setup_seconds_max=" << bicgstab_timing[4]
                      << " preconditioner_setup_seconds_max="
                      << bicgstab_timing[5]
                      << " preconditioner_apply_seconds_max="
                      << preconditioner_apply_seconds
                      << " total_seconds_max=" << bicgstab_timing[3]
                      << std::endl;
        }
        if(fixed16_block_matvec_route_requested)
        {
            unsigned long long fixed16_totals[3] = {
                static_cast<unsigned long long>(
                    fixed16_block_matvec_schedule.Ready ?
                        fixed16_block_matvec_schedule.RowCount : 0),
                static_cast<unsigned long long>(
                    fixed16_block_matvec_schedule.Ready ?
                        fixed16_block_matvec_schedule.NeighborCouplings : 0),
                fixed16_block_matvec_shadow_mismatch_values};
            unsigned long long fixed16_call_maxima[5] = {
                fixed16_block_matvec_kernel_calls,
                fixed16_block_matvec_selected_calls,
                fixed16_block_matvec_generic_calls,
                fixed16_block_matvec_shadow_checks,
                fixed16_block_matvec_shadow_mismatch_calls};
            double fixed16_timing_maxima[3] = {
                fixed16_block_matvec_setup_seconds,
                fixed16_block_matvec_kernel_seconds,
                fixed16_block_matvec_shadow_seconds};
#ifdef RICH_MPI
            RequireCGMpiSuccess(
                MPI_Allreduce(
                    MPI_IN_PLACE, fixed16_totals, 3,
                    MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
                "MPI_Allreduce(global fixed16 block matvec totals)");
            RequireCGMpiSuccess(
                MPI_Allreduce(
                    MPI_IN_PLACE, fixed16_call_maxima, 5,
                    MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD),
                "MPI_Allreduce(global fixed16 block matvec calls)");
            RequireCGMpiSuccess(
                MPI_Allreduce(
                    MPI_IN_PLACE, fixed16_timing_maxima, 3,
                    MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
                "MPI_Allreduce(global fixed16 block matvec timing)");
#endif
            char const* const fixed16_outcome =
                fixed16_block_matvec_enabled ? "selected" :
                (fixed16_block_matvec_shadow_enabled ?
                    "shadow_only" : "generic_fallback");
            char const* const shadow_outcome =
                !fixed16_block_matvec_shadow_requested ? "not_requested" :
                (!fixed16_block_matvec_shadow_enabled ? "not_run" :
                    (fixed16_call_maxima[4] == 0 ? "match" : "mismatch"));
            if(rank == 0)
                std::clog << std::setprecision(17)
                          << "MG_FIXED16_BLOCK_MATVEC scope=global"
                          << " requested="
                          << (fixed16_block_matvec_requested ? 1 : 0)
                          << " shadow_requested="
                          << (fixed16_block_matvec_shadow_requested ? 1 : 0)
                          << " structure_supported="
                          << (fixed16_block_matvec_structure_supported ? 1 : 0)
                          << " enabled="
                          << (fixed16_block_matvec_enabled ? 1 : 0)
                          << " shadow_enabled="
                          << (fixed16_block_matvec_shadow_enabled ? 1 : 0)
                          << " outcome=" << fixed16_outcome
                          << " fallback_reason="
                          << Fixed16BlockMatvecFallbackMaskLabel(
                                 fixed16_block_matvec_fallback_mask)
                          << " validated_rows_total=" << fixed16_totals[0]
                          << " neighbor_couplings_total=" << fixed16_totals[1]
                          << " candidate_calls_max=" << fixed16_call_maxima[0]
                          << " selected_calls_max=" << fixed16_call_maxima[1]
                          << " generic_calls_max=" << fixed16_call_maxima[2]
                          << " shadow_checks_max=" << fixed16_call_maxima[3]
                          << " shadow_mismatch_calls_max="
                          << fixed16_call_maxima[4]
                          << " shadow_mismatch_values_total="
                          << fixed16_totals[2]
                          << " shadow_outcome=" << shadow_outcome
                          << " setup_seconds_max=" << fixed16_timing_maxima[0]
                          << " candidate_seconds_max="
                          << fixed16_timing_maxima[1]
                          << " shadow_seconds_max="
                          << fixed16_timing_maxima[2]
                          << std::endl;
        }
        if(fixed16_block_stencil_route_requested)
        {
            unsigned long long fixed16_stencil_totals[4] = {
                static_cast<unsigned long long>(
                    fixed16_block_stencil_structure_supported ?
                        fixed16_block_stencil.LocalCellCount : 0),
                static_cast<unsigned long long>(
                    fixed16_block_stencil_structure_supported ?
                        fixed16_block_stencil.NeighborCells.size() : 0),
                static_cast<unsigned long long>(
                    fixed16_block_stencil_structure_supported ?
                        fixed16_block_stencil.LocalBlockValues.size() +
                        fixed16_block_stencil.NeighborValues.size() : 0),
                fixed16_block_stencil_shadow_mismatch_values};
            unsigned long long fixed16_stencil_call_maxima[5] = {
                fixed16_block_stencil_kernel_calls,
                fixed16_block_stencil_selected_calls,
                fixed16_block_stencil_generic_calls,
                fixed16_block_stencil_shadow_checks,
                fixed16_block_stencil_shadow_mismatch_calls};
            double fixed16_stencil_timing_maxima[3] = {
                fixed16_block_stencil_setup_seconds,
                fixed16_block_stencil_kernel_seconds,
                fixed16_block_stencil_shadow_seconds};
#ifdef RICH_MPI
            RequireCGMpiSuccess(
                MPI_Allreduce(
                    MPI_IN_PLACE, fixed16_stencil_totals, 4,
                    MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD),
                "MPI_Allreduce(global fixed16 block-stencil totals)");
            RequireCGMpiSuccess(
                MPI_Allreduce(
                    MPI_IN_PLACE, fixed16_stencil_call_maxima, 5,
                    MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD),
                "MPI_Allreduce(global fixed16 block-stencil calls)");
            RequireCGMpiSuccess(
                MPI_Allreduce(
                    MPI_IN_PLACE, fixed16_stencil_timing_maxima, 3,
                    MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD),
                "MPI_Allreduce(global fixed16 block-stencil timing)");
#endif
            char const* const fixed16_stencil_outcome =
                fixed16_stencil_call_maxima[4] != 0 ?
                    "shadow_mismatch_fallback" :
                (fixed16_block_stencil_enabled ? "selected" :
                 (fixed16_block_stencil_shadow_enabled ?
                    "shadow_only" : "generic_fallback"));
            char const* const fixed16_stencil_shadow_outcome =
                !fixed16_block_stencil_shadow_requested ? "not_requested" :
                (!fixed16_block_stencil_shadow_enabled ? "not_run" :
                 (fixed16_stencil_call_maxima[4] == 0 ?
                    "match" : "mismatch"));
            if(rank == 0)
                std::clog << std::setprecision(17)
                          << "MG_FIXED16_BLOCK_STENCIL scope=global"
                          << " requested="
                          << (fixed16_block_stencil_requested ? 1 : 0)
                          << " shadow_requested="
                          << (fixed16_block_stencil_shadow_requested ? 1 : 0)
                          << " avx2_neighbors_requested="
                          << (fixed16_avx2_neighbors_requested ? 1 : 0)
                          << " builder_declared="
                          << (fixed16_block_stencil_builder_declared ? 1 : 0)
                          << " builder_supported="
                          << (fixed16_block_stencil_builder_supported ? 1 : 0)
                          << " local_eligible="
                          << (local_fixed16_block_stencil_eligible ? 1 : 0)
                          << " globally_eligible="
                          << (fixed16_block_stencil_globally_eligible ? 1 : 0)
                          << " build_attempted="
                          << (fixed16_block_stencil_build_attempted ? 1 : 0)
                          << " csr_shadow_built="
                          << (fixed16_block_stencil_build_attempted &&
                              fixed16_block_stencil_build_csr_shadow ? 1 : 0)
                          << " structure_supported="
                          << (fixed16_block_stencil_structure_supported ? 1 : 0)
                          << " enabled="
                          << (fixed16_block_stencil_enabled ? 1 : 0)
                          << " shadow_enabled="
                          << (fixed16_block_stencil_shadow_enabled ? 1 : 0)
                          << " fallback_rebuilt_csr="
                          << (fixed16_block_stencil_fallback_rebuilt_csr ? 1 : 0)
                          << " outcome=" << fixed16_stencil_outcome
                          << " fallback_reason="
                          << Fixed16BlockStencilFallbackMaskLabel(
                                 fixed16_block_stencil_fallback_mask)
                          << " validated_cells_total="
                          << fixed16_stencil_totals[0]
                          << " neighbor_cells_total="
                          << fixed16_stencil_totals[1]
                          << " stored_coefficients_total="
                          << fixed16_stencil_totals[2]
                          << " candidate_calls_max="
                          << fixed16_stencil_call_maxima[0]
                          << " selected_calls_max="
                          << fixed16_stencil_call_maxima[1]
                          << " generic_calls_max="
                          << fixed16_stencil_call_maxima[2]
                          << " shadow_checks_max="
                          << fixed16_stencil_call_maxima[3]
                          << " shadow_mismatch_calls_max="
                          << fixed16_stencil_call_maxima[4]
                          << " shadow_mismatch_values_total="
                          << fixed16_stencil_totals[3]
                          << " shadow_outcome="
                          << fixed16_stencil_shadow_outcome
                          << " exact_order=diagonal_other_groups_faces"
                          << " true_residual="
                          << (fixed16_block_stencil_enabled ?
                              "stencil" : "csr")
                          << " setup_seconds_max="
                          << fixed16_stencil_timing_maxima[0]
                          << " candidate_seconds_max="
                          << fixed16_stencil_timing_maxima[1]
                          << " shadow_seconds_max="
                          << fixed16_stencil_timing_maxima[2]
                          << std::endl;
        }
        direction_preconditioner.Release();
#ifdef RICH_MPI
        MPI_exchange_data(tess, sub_x, true, slice);
        MPI_exchange_data(tess, sub_x_solution, true, slice);
#endif
        return sub_x;
    }
}
