#ifndef RADIATION_DRIVER_TEST_HOOKS_HPP
#define RADIATION_DRIVER_TEST_HOOKS_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "newtonian/three_dimensional/computational_cell.hpp"

struct IndividualStepContext;

#ifdef RICH_MPI
namespace RadiationDriverTestHooks
{
struct OwnedCanonicalMappingProbeResult
{
    bool valid = false;
    bool identity = false;
};

OwnedCanonicalMappingProbeResult ProbeOwnedCanonicalMapping(
    std::vector<ComputationalCell3D> const& owned_cells,
    std::vector<ComputationalCell3D> const& canonical_cells,
    std::vector<std::size_t> const& owned_to_canonical);

bool ProbeCompleteOwnedActivity(IndividualStepContext const& context,
                                std::size_t owned_count);

struct SchedulerEventIntervalProbeResult
{
    bool valid = false;
    bool matches_cell_interval = false;
    double interval = 0;
};

SchedulerEventIntervalProbeResult ProbeSchedulerEventInterval(
    IndividualStepContext const& context, double cell_interval);

struct DistributedActiveMatVecProbeResult
{
    bool exchange_initialized = false;
    bool multiplied = false;
    bool second_multiply_succeeded = false;
    bool exchange_storage_reused = false;
    bool csr_uses_narrow_columns = false;
    std::size_t global_size = 0;
    std::size_t csr_nonzeros = 0;
    std::size_t local_row_count = 0;
    std::size_t remote_row_count = 0;
    std::size_t request_send_chunks = 0;
    std::size_t request_receive_chunks = 0;
    unsigned long long exchange_validity_reductions = 0;
    std::vector<double> output;
    std::vector<double> second_output;
};

// Focused regression hook for the production distributed-active request,
// exchange, and matvec path.  Global cell IDs remain size_t end to end.
DistributedActiveMatVecProbeResult ProbeDistributedActiveMatVec(
    std::vector<ComputationalCell3D> const& owned_active_cells,
    std::vector<double> const& local_values,
    std::vector<int> const& remote_owners,
    std::vector<std::size_t> const& remote_cell_ids,
    std::vector<std::vector<double> > const& matrix,
    std::vector<std::vector<std::size_t> > const& columns,
    std::size_t request_chunk_limit = 0,
    int request_tag = 21059);

struct DistributedCSRColumnProbeResult
{
    bool uses_narrow_columns = false;
    unsigned long long storage_bytes = 0;
    std::vector<std::size_t> columns;
};

DistributedCSRColumnProbeResult ProbeDistributedCSRColumns(
    std::vector<std::size_t> const& columns);

struct DistributedActiveTransferChunkProbeResult
{
    bool valid = false;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> counts;
};

// Exercises the production transfer segmentation without allocating the
// represented value buffer.  This keeps totals above 2^32 testable.
DistributedActiveTransferChunkProbeResult
ProbeDistributedActiveTransferChunks(std::size_t total_count);

struct DistributedActiveSizeArithmeticProbeResult
{
    bool add_valid = false;
    std::size_t sum = 0;
    bool multiply_valid = false;
    std::size_t product = 0;
};

DistributedActiveSizeArithmeticProbeResult
ProbeDistributedActiveSizeArithmetic(std::size_t left,
                                     std::size_t right);

struct DistributedActiveToggleProbeResult
{
    bool valid = false;
    bool value = false;
};

DistributedActiveToggleProbeResult ProbeDistributedActiveToggle(
    std::string const& setting, bool fallback);
bool ProbeDistributedActiveOverlapDefault();

// Used only by the focused MPI subprocess test.  The selected rank enters the
// same fail-stop path used after an unrecoverable communication-call error.
[[noreturn]] void TriggerDistributedActiveFailStopForTest();

// Installs MPI_ERRORS_RETURN, leaves one real receive outstanding, then passes
// a real MPI_ERR_COUNT return through the production fail-stop checker.
[[noreturn]] void TriggerDistributedActiveReturnedMpiFailureForTest();

// Makes a branch-defining collective return MPI_ERR_COUNT under
// MPI_ERRORS_RETURN, then routes that real error through the same checker.
[[noreturn]] void TriggerDistributedActiveReturnedCollectiveFailureForTest();
}
#endif

#endif
