#include "IndividualTimeStep.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <unordered_set>

#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

#ifdef RICH_MPI
#include "mpi/mpi_commands.hpp"
#endif

namespace
{
    // Cells named in RICH_INDIVIDUAL_TRACE_CELL_IDS (comma-separated stable
    // IDs, as for the hydro cell trace), sorted; empty when unset.
    std::vector<std::size_t> const& SchedulerTraceCellIds()
    {
        static std::vector<std::size_t> const ids = []()
        {
            std::vector<std::size_t> parsed;
            char const* const value = std::getenv("RICH_INDIVIDUAL_TRACE_CELL_IDS");
            if(value == nullptr || value[0] == '\0')
                return parsed;
            std::string const text(value);
            std::size_t begin = 0;
            while(begin <= text.size())
            {
                std::size_t const end = std::min(text.find(',', begin), text.size());
                std::string const token = text.substr(begin, end - begin);
                if(token.empty() || token.find_first_not_of("0123456789") != std::string::npos)
                    throw std::invalid_argument(
                        "RICH_INDIVIDUAL_TRACE_CELL_IDS must be a comma-separated list of cell IDs");
                parsed.push_back(static_cast<std::size_t>(std::stoull(token)));
                begin = end + 1;
            }
            std::sort(parsed.begin(), parsed.end());
            return parsed;
        }();
        return ids;
    }

    // RICH_INDIVIDUAL_MAX_BIN_SPREAD = K >= 0: no cell's bin may exceed the
    // anchor bin (initial_bin) by more than K, so no interval is longer than 2^K
    // anchor-bin intervals; -1 is off (the behaviour before 2026-09-27).  It was
    // first finest-bin + K, which let one pinned cell drag every cell down.  Default 2: on the TDE (jobs 10222210, 10222217) it cut the
    // overrun cell-events by 70 % and the worst interval/allowance to 2.0, and
    // brought the terminal state closer to the global run's, at about the same
    // speed as K = 4.  Collective on first use (commitEvent or restore, on
    // every rank); must agree.
    int MaximumBinSpread()
    {
        static int const spread = []()
        {
            char const* const value = std::getenv("RICH_INDIVIDUAL_MAX_BIN_SPREAD");
            int parsed = 2;
            if(value != nullptr && value[0] != '\0')
            {
                char* end = nullptr;
                long const number = std::strtol(value, &end, 10);
                parsed = end != value && *end == '\0' && number >= -1 && number <= 64 ?
                    static_cast<int>(number) : -100;
            }
            int code[2] = {parsed, -parsed};
#ifdef RICH_MPI
            MPI_Allreduce(MPI_IN_PLACE, code, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
            if(code[0] < -1 || code[0] != -code[1] || parsed < -1)
                throw std::invalid_argument(
                    "RICH_INDIVIDUAL_MAX_BIN_SPREAD must be one integer in [-1, 64] on every rank");
            return parsed;
        }();
        return spread;
    }

    // A strict integer switch in [0, MaximumValue]: unset or empty is 0, anything but one digit in range is
    // rejected, and every rank must hold the same value (all ranks throw together otherwise).  Collective: each
    // switch below is read once, at its first use, from a function that every rank enters.
    int AgreedIntegerSwitch(char const* Name, int MaximumValue)
    {
        char const* const Value = std::getenv(Name);
        std::string const Text = Value == nullptr ? std::string() : std::string(Value);
        int Local = -1;
        if(Text.empty())
            Local = 0;
        else if(Text.size() == 1 && Text[0] >= '0' && Text[0] <= static_cast<char>('0' + MaximumValue))
            Local = Text[0] - '0';
        int Extrema[2] = {Local, -Local};
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, Extrema, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
        if(Extrema[0] < 0 || Extrema[0] != -Extrema[1])
        {
            std::string Allowed = "0";
            for(int AllowedValue = 1; AllowedValue <= MaximumValue; ++AllowedValue)
                Allowed += (AllowedValue == MaximumValue ? " or " : ", ") + std::to_string(AllowedValue);
            throw std::invalid_argument(std::string(Name) + " must be " + Allowed + " and the same on every MPI rank");
        }
        return Extrema[0];
    }

    // RICH_D2_TIMERS (0/1, shared by the D2 diagnostics): here, the RICH_D2_CLOSURE split of the neighbour-bin
    // closure.  First used in enforceNeighborBinClosure or applyAMRChangeSet.
    bool D2TimersEnabled()
    {
        static bool const Enabled = AgreedIntegerSwitch("RICH_D2_TIMERS", 1) != 0;
        return Enabled;
    }

    // RICH_INDIVIDUAL_CLOSURE_SCAN (0 off, 1 skip the full closure when a scan finds nothing for it to do on any
    // rank, 2 shadow: scan, then run the full closure anyway and count disagreements).  First used in
    // enforceNeighborBinClosure.
    int ClosureScanMode()
    {
        static int const Mode = AgreedIntegerSwitch("RICH_INDIVIDUAL_CLOSURE_SCAN", 2);
        return Mode;
    }

    // RICH_INDIVIDUAL_SCHEDULER_COUNTS (0 off, 1 tracked counts replace the per-event scans, 2 both, compared).
    // First used in commitEvent.
    int SchedulerCountsSwitch()
    {
        static int const Mode = AgreedIntegerSwitch("RICH_INDIVIDUAL_SCHEDULER_COUNTS", 2);
        return Mode;
    }

    // RICH_INDIVIDUAL_CLOSURE_FAST (0 reference full closure, 1 fast full closure, 2 both on copies of the state,
    // compared bin for bin and request for request, abort on the first difference).  Full closures only; the
    // single-pass propagation closure of commitEvent always takes the reference path.  First used in
    // limitNeighborBins.  MPI builds only: a serial build always runs the reference.
    int ClosureFastMode()
    {
        static int const Mode = AgreedIntegerSwitch("RICH_INDIVIDUAL_CLOSURE_FAST", 2);
        return Mode;
    }

    double D2Now()
    {
#ifdef RICH_MPI
        return MPI_Wtime();
#else
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
    }

    int WorldRank()
    {
        int Rank = 0;
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
#endif
        return Rank;
    }

    // Process totals of the closure scan, printed in RICH_D2_CLOSURE.  Every rank holds the same values: they count
    // collective outcomes only.
    struct ClosureScanTotals
    {
        std::uint64_t Scans = 0;
        std::uint64_t Clean = 0;
        std::uint64_t Skipped = 0;
        std::uint64_t ShadowCleanChanged = 0;
        std::uint64_t ShadowDirtyUnchanged = 0;
    };
    ClosureScanTotals& ClosureTotals()
    {
        static ClosureScanTotals Totals;
        return Totals;
    }

    // RICH_INDIVIDUAL_SCHEDULER_COUNTS mismatch or corrupt count: this rank prints one example and aborts the job,
    // since the other ranks cannot know.
    [[noreturn]] void AbortScheduleCounts(std::string const& Message)
    {
        std::cerr << "RICH_SCHED_COUNTS_MISMATCH rank=" << WorldRank() << " " << Message << std::endl;
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 97);
#endif
        throw std::logic_error("RICH_INDIVIDUAL_SCHEDULER_COUNTS mismatch: " + Message);
    }

    // One line per scheduler mutation of a traced cell, printed by its owner:
    // what changed its bin or endpoint, from what to what, at which event.
    void TraceSchedulerMutation(char const* kind, std::size_t cell_id, std::uint64_t event_tick,
        std::uint64_t begin_tick, unsigned old_bin, unsigned new_bin, std::uint64_t old_end, std::uint64_t new_end)
    {
        std::vector<std::size_t> const& ids = SchedulerTraceCellIds();
        if(ids.empty() || !std::binary_search(ids.begin(), ids.end(), cell_id))
            return;
        std::cout << "INDIVIDUAL_SCHED_TRACE kind=" << kind << " cell_id=" << cell_id
                  << " event_tick=" << event_tick << " begin_tick=" << begin_tick
                  << " old_bin=" << old_bin << " new_bin=" << new_bin
                  << " old_end=" << old_end << " new_end=" << new_end
                  << " elapsed_over_new_allowance=" << static_cast<double>(event_tick > begin_tick ?
                         event_tick - begin_tick : 0) / std::ldexp(1.0, static_cast<int>(new_bin))
                  << std::endl;
    }

    // Rank-0 aggregate per event, one representative cell.  An interval that
    // outlasts its own bin's allowance is a scheduling defect, not a physical
    // estimate: no wake, clamp or overlay can lengthen an interval, only cut
    // it short.  A nonzero count means some cell was told to run finer and
    // kept its old, longer interval anyway.
    void ReportIndividualBinOverrun(unsigned long long local_count,
                                    double local_worst, std::size_t local_id,
                                    std::uint64_t event_tick)
    {
        unsigned long long count = local_count;
        int rank = 0;
        struct { double value; int rank; } worst{local_worst, 0};
        std::size_t example = local_id;
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        worst.rank = rank;
        MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                      MPI_COMM_WORLD);
        // The worst offender is rarely on rank 0, so carry it with its owner.
        MPI_Allreduce(MPI_IN_PLACE, &worst, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                      MPI_COMM_WORLD);
        unsigned long long example_value = static_cast<unsigned long long>(
            rank == worst.rank ? local_id : 0);
        MPI_Bcast(&example_value, 1, MPI_UNSIGNED_LONG_LONG, worst.rank,
                  MPI_COMM_WORLD);
        example = static_cast<std::size_t>(example_value);
#endif
        if(count == 0 || rank != 0)
            return;
        std::cout << std::setprecision(6)
                  << "INDIVIDUAL_BIN_OVERRUN event_tick=" << event_tick
                  << " cells=" << count
                  << " worst_interval_over_allowance=" << worst.value
                  << " worst_cell=" << example
                  << " worst_rank=" << worst.rank << std::endl;
    }
}


namespace
{
std::uint64_t checked_add(std::uint64_t left, std::uint64_t right)
{
    if(right > std::numeric_limits<std::uint64_t>::max() - left)
        throw std::overflow_error("Individual timestep timeline overflow");
    return left + right;
}

#ifdef RICH_MPI
struct NeighborBinSnapshot : public Serializable
{
    std::size_t cell_id = 0;
    std::uint8_t time_bin = std::numeric_limits<std::uint8_t>::max();
    std::uint8_t source_bin = std::numeric_limits<std::uint8_t>::max();
    unsigned char valid = 0;

    force_inline std::size_t dump(Serializer* serializer) const override
    {
        std::size_t bytes = 0;
        bytes += serializer->insert(cell_id);
        bytes += serializer->insert(time_bin);
        bytes += serializer->insert(source_bin);
        bytes += serializer->insert(valid);
        return bytes;
    }

    force_inline std::size_t load(const Serializer* serializer,
                                  std::size_t byte_offset) override
    {
        std::size_t bytes = 0;
        bytes += serializer->extract(cell_id, byte_offset + bytes);
        bytes += serializer->extract(time_bin, byte_offset + bytes);
        bytes += serializer->extract(source_bin, byte_offset + bytes);
        bytes += serializer->extract(valid, byte_offset + bytes);
        return bytes;
    }
};

struct NeighborBinRequest : public Serializable
{
    std::size_t cell_id = 0;
    std::uint8_t maximum_bin = 0;

    force_inline std::size_t dump(Serializer* serializer) const override
    {
        std::size_t bytes = 0;
        bytes += serializer->insert(cell_id);
        bytes += serializer->insert(maximum_bin);
        return bytes;
    }

    force_inline std::size_t load(const Serializer* serializer,
                                  std::size_t byte_offset) override
    {
        std::size_t bytes = 0;
        bytes += serializer->extract(cell_id, byte_offset + bytes);
        bytes += serializer->extract(maximum_bin, byte_offset + bytes);
        return bytes;
    }
};

struct AMRPendingBinRequest : public Serializable
{
    std::size_t source_cell_id = 0;
    std::size_t recipient_cell_id = 0;

    force_inline std::size_t dump(Serializer* serializer) const override
    {
        std::size_t bytes = 0;
        bytes += serializer->insert(source_cell_id);
        bytes += serializer->insert(recipient_cell_id);
        return bytes;
    }

    force_inline std::size_t load(const Serializer* serializer,
                                  std::size_t byte_offset) override
    {
        std::size_t bytes = 0;
        bytes += serializer->extract(source_cell_id, byte_offset + bytes);
        bytes += serializer->extract(recipient_cell_id, byte_offset + bytes);
        return bytes;
    }
};

struct AMRPendingBinResponse : public Serializable
{
    std::size_t recipient_cell_id = 0;
    std::uint8_t time_bin = 0;
    std::uint8_t pending_bin = std::numeric_limits<std::uint8_t>::max();
    double radiation_accuracy_limit = 0;

    force_inline std::size_t dump(Serializer* serializer) const override
    {
        std::size_t bytes = 0;
        bytes += serializer->insert(recipient_cell_id);
        bytes += serializer->insert(time_bin);
        bytes += serializer->insert(pending_bin);
        bytes += serializer->insert(radiation_accuracy_limit);
        return bytes;
    }

    force_inline std::size_t load(const Serializer* serializer,
                                  std::size_t byte_offset) override
    {
        std::size_t bytes = 0;
        bytes += serializer->extract(recipient_cell_id, byte_offset + bytes);
        bytes += serializer->extract(time_bin, byte_offset + bytes);
        bytes += serializer->extract(pending_bin, byte_offset + bytes);
        bytes += serializer->extract(radiation_accuracy_limit, byte_offset + bytes);
        return bytes;
    }
};

template<typename T>
std::vector<T> SyncCanonicalDataToMesh(
    const Tessellation3D& tess,
    const Tessellation3D::AllPointsMap& local_to_global,
    const std::vector<T>& canonical)
{
    std::vector<T> local(tess.getMeshPoints().size());
    bool mapping_valid = true;
    for(auto const& mapping : local_to_global)
    {
        if(mapping.first >= local.size() || mapping.second >= canonical.size())
        {
            mapping_valid = false;
            continue;
        }
        local[mapping.first] = canonical[mapping.second];
    }
    int mapping_valid_int = mapping_valid ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &mapping_valid_int, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if(mapping_valid_int == 0)
        throw std::logic_error(
            "Individual timestep canonical-to-mesh map is invalid on at least one rank");
    std::vector<T> canonical_copy = canonical;
    tess.SyncPartialBuildData(local, canonical_copy);
    return local;
}

// Appends one closure iteration's outgoing requests to a log, per destination rank in send order
// (RICH_INDIVIDUAL_CLOSURE_FAST=2 compares the two implementations' logs).
void RecordClosureRequests(std::vector<std::vector<std::vector<std::pair<std::size_t, std::uint8_t> > > >& Log,
                           std::vector<std::vector<NeighborBinRequest> > const& Outgoing)
{
    std::vector<std::vector<std::pair<std::size_t, std::uint8_t> > >& Iteration = Log.emplace_back(Outgoing.size());
    for(std::size_t Destination = 0; Destination < Outgoing.size(); ++Destination)
        for(NeighborBinRequest const& Request : Outgoing[Destination])
            Iteration[Destination].emplace_back(Request.cell_id, Request.maximum_bin);
}

// One canonical cell in the first mesh sync of the fast closure: its stable ID, and its bin plus one (zero marks a
// mesh entry the sync did not fill, the reference's valid == 0).  Later iterations sync the bin byte alone.
struct FastClosureEntry
{
    std::size_t CellId;
    std::uint8_t Bin;
};
#endif
}

// Filled by limitNeighborBins when a caller passes it.  Iterations and Outgoing are the same length on every rank
// (the loop ends on a collective flag); Outgoing[k] is 1 when this rank had a request to send in iteration k.
struct IndividualTimeStepScheduler::ClosureSplit
{
    unsigned Iterations = 0;
    std::vector<double> Outgoing;
    double TotalSeconds = 0;
    double ExchangeSeconds = 0;
    // First cell this rank lowered, for the shadow's mismatch example.
    std::size_t ExampleCellId = std::numeric_limits<std::size_t>::max();
    // RICH_INDIVIDUAL_CLOSURE_FAST=2: every request this rank sent, per iteration and destination rank, as
    // (cell ID, maximum bin) in send order.
    bool RecordRequests = false;
    std::vector<std::vector<std::vector<std::pair<std::size_t, std::uint8_t> > > > Requests;
    // The fast run of mode 2 prints no scheduler trace lines; the reference run that follows prints them.
    bool SuppressTrace = false;
};

#ifdef RICH_MPI
std::size_t CellTimeState::dump(Serializer* serializer) const
{
    std::size_t bytes = 0;
    bytes += serializer->insert(cell_id);
    bytes += serializer->insert(begin_tick);
    bytes += serializer->insert(end_tick);
    bytes += serializer->insert(last_primitive_tick);
    bytes += serializer->insert(time_bin);
    bytes += serializer->insert(pending_neighbor_bin);
    bytes += serializer->insert(point_velocity);
    bytes += serializer->insert(cached_acceleration);
    const std::uint8_t gravity_phase = gravity_half_kick_pending ? 1 : 0;
    bytes += serializer->insert(gravity_phase);
    bytes += serializer->insert(change_wake_pending);
    bytes += serializer->insert(change_wake_ratio);
    bytes += serializer->insert(radiation_accuracy_limit);
    return bytes;
}

std::size_t CellTimeState::load(const Serializer* serializer,
                                std::size_t byte_offset)
{
    std::size_t bytes = 0;
    bytes += serializer->extract(cell_id, byte_offset + bytes);
    bytes += serializer->extract(begin_tick, byte_offset + bytes);
    bytes += serializer->extract(end_tick, byte_offset + bytes);
    bytes += serializer->extract(last_primitive_tick, byte_offset + bytes);
    bytes += serializer->extract(time_bin, byte_offset + bytes);
    bytes += serializer->extract(pending_neighbor_bin, byte_offset + bytes);
    bytes += serializer->extract(point_velocity, byte_offset + bytes);
    bytes += serializer->extract(cached_acceleration, byte_offset + bytes);
    std::uint8_t gravity_phase = 0;
    bytes += serializer->extract(gravity_phase, byte_offset + bytes);
    gravity_half_kick_pending = gravity_phase != 0;
    bytes += serializer->extract(change_wake_pending, byte_offset + bytes);
    bytes += serializer->extract(change_wake_ratio, byte_offset + bytes);
    bytes += serializer->extract(radiation_accuracy_limit, byte_offset + bytes);
    return bytes;
}
#endif

bool AdvanceRadiationRepairWarning(
    RadiationRepairAccounting& accounting, double const injection_fraction)
{
    bool const warn = std::isfinite(injection_fraction) &&
        injection_fraction >= accounting.next_warning_fraction;
    if(!warn)
        return false;
    while(accounting.next_warning_fraction <= injection_fraction) {
        double const next = 2 * accounting.next_warning_fraction;
        if(!std::isfinite(next) || next <= accounting.next_warning_fraction) {
            accounting.next_warning_fraction =
                std::numeric_limits<double>::max();
            break;
        }
        accounting.next_warning_fraction = next;
    }
    return true;
}

bool IndividualStepContext::isActive(std::size_t index) const
{
    return index < active_mask.size() && active_mask[index] != 0;
}

double IndividualStepContext::cellTimeStep(std::size_t index) const
{
    if(index >= cell_time_steps.size())
        throw std::out_of_range("IndividualStepContext cell index is out of range");
    return cell_time_steps[index];
}

double IndividualStepContext::nominalCellTimeStep(std::size_t index) const
{
    if(index < cell_time_bins.size() &&
       time_quantum > 0 && std::isfinite(time_quantum))
        return std::ldexp(
            time_quantum, static_cast<int>(cell_time_bins[index]));
    return cellTimeStep(index);
}

double IndividualStepContext::faceTimeStep(std::size_t left, std::size_t right) const
{
    return std::min(cellTimeStep(left), cellTimeStep(right));
}

double IndividualStepContext::hydroFaceTimeStep(
    std::size_t left, std::size_t right) const
{
    if(!(time_quantum > 0 && std::isfinite(time_quantum)) ||
       primitive_ticks.size() != cell_time_steps.size() ||
       left >= primitive_ticks.size() || right >= primitive_ticks.size())
        throw std::invalid_argument(
            "Individual hydro face interval requires endpoint activation ticks");

    // Wakes can end an interval between the neighbor's normal bin boundaries.
    // min(cell_dt) would then repeat flux already applied at its last activation.
    std::uint64_t const last_face_tick =
        std::max(primitive_ticks[left], primitive_ticks[right]);
    if(last_face_tick >= event_tick)
        throw std::invalid_argument(
            "Individual hydro face interval must advance both endpoint ticks");
    return time_quantum * static_cast<double>(event_tick - last_face_tick);
}

std::uint64_t IndividualActiveTimeBinMask(
    IndividualStepContext const& context,
    std::vector<CellTimeState> const& states)
{
    if(context.active_mask.size() != states.size())
        throw std::invalid_argument(
            "Active-bin mask requires scheduler-aligned activity state");

    std::uint64_t result = 0;
    for(std::size_t const active : context.active_indices)
    {
        if(active >= states.size() || !context.isActive(active))
            throw std::out_of_range(
                "Active-bin mask contains an invalid active-cell index");
        std::uint8_t const time_bin = states[active].time_bin;
        if(time_bin > 62)
            throw std::out_of_range(
                "Active-bin mask cannot represent scheduler bins above 62");
        result |= std::uint64_t(1) << time_bin;
    }
    return result;
}

IndividualTimeStepScheduler::IndividualTimeStepScheduler(IndividualTimeStepOptions options)
    : options_(options)
{
    if(options_.initial_bin > options_.maximum_bin)
        throw std::invalid_argument("Individual timestep initial bin exceeds maximum bin");
    if(options_.maximum_bin > 62)
        throw std::invalid_argument("Individual timestep maximum bin must not exceed 62");
    if(options_.full_source_sweep_interval_minimum_steps == 0)
        throw std::invalid_argument(
            "Individual full-source sweep interval must be positive");
    if(!(options_.partial_build_fraction > 0 && options_.partial_build_fraction <= 1))
        throw std::invalid_argument("Partial build fraction must be in (0, 1]");
}

void IndividualTimeStepScheduler::initialize(const std::vector<ComputationalCell3D> &cells,
                                             double current_time,
                                             double initial_time_step,
                                             double anchor_time_step)
{
    if(initialized_)
        throw std::logic_error("Individual timestep scheduler is already initialized");
    if(!(std::isfinite(initial_time_step) && initial_time_step > 0))
        throw std::invalid_argument("Individual timestep initialization requires a positive finite timestep");
    if(!(std::isfinite(anchor_time_step) && anchor_time_step >= 0))
        throw std::invalid_argument("Individual timestep anchor must be finite and nonnegative");
    if(anchor_time_step > 0 && options_.time_quantum > 0)
        throw std::invalid_argument("Individual timestep anchor conflicts with an explicit time quantum");
    double const anchor = anchor_time_step > 0 ? anchor_time_step : initial_time_step;

    time_quantum_ = options_.time_quantum;
    if(!(time_quantum_ > 0))
        time_quantum_ = std::ldexp(anchor, -static_cast<int>(options_.initial_bin));
    if(!(std::isfinite(time_quantum_) && time_quantum_ > 0))
        throw std::invalid_argument("Individual timestep quantum must be positive and finite");

    time_origin_ = current_time;
    current_tick_ = 0;
    last_full_source_sweep_tick_ = 0;
    states_.resize(cells.size());
    // The first interval may not exceed initial_time_step, the step every
    // cell's limits allowed: on its own grid that is initial_bin; on an anchor
    // grid, the coarsest bin within it (quantizeTimeStep throws below one
    // quantum).
    // On an anchor grid the first bin is also held under the bin-spread
    // ceiling (initial_bin + K); an unanchored start is initial_bin itself.
    // The anchor is collective, so MaximumBinSpread's first use is too.
    std::uint8_t start_bin = anchor_time_step > 0 ?
        quantizeTimeStep(initial_time_step) : options_.initial_bin;
    // An explicit quantum keeps initial_bin, but never a first interval
    // longer than initial_time_step (the entry generator velocities are fixed
    // for that step); a step below one quantum is rejected (quantizeTimeStep).
    if(!(anchor_time_step > 0) && options_.time_quantum > 0)
        start_bin = std::min(start_bin, quantizeTimeStep(initial_time_step));
    if(anchor_time_step > 0 && MaximumBinSpread() >= 0)
        start_bin = static_cast<std::uint8_t>(std::min<unsigned>(start_bin,
            std::min<unsigned>(options_.maximum_bin,
                static_cast<unsigned>(options_.initial_bin) + static_cast<unsigned>(MaximumBinSpread()))));
    const std::uint64_t initial_ticks = ticksForBin(start_bin);
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        CellTimeState &state = states_[i];
        state.cell_id = cells[i].ID;
        state.begin_tick = 0;
        state.end_tick = initial_ticks;
        state.last_primitive_tick = 0;
        state.time_bin = start_bin;
        // Before the first commit refreshes it, the finest bin in use is the
        // one every cell starts in.  Leaving it at zero would send an overdue
        // cell to the tick after this one.
        minimum_occupied_bin_ = start_bin;
        state.pending_neighbor_bin =
            std::numeric_limits<std::uint8_t>::max();
        state.point_velocity = Vector3D();
        state.cached_acceleration = Vector3D();
        state.gravity_half_kick_pending = false;
        state.change_wake_pending = 0;
        state.change_wake_ratio = 0;
        state.radiation_accuracy_limit = 0;
    }
    rebuildIndex(cells);
    initialized_ = true;
}

IndividualStepContext IndividualTimeStepScheduler::prepareEvent(
    const std::vector<ComputationalCell3D> &cells) const
{
    if(!initialized_)
        throw std::logic_error("Individual timestep scheduler is not initialized");
    validateCells(cells);
    if(states_.empty())
        throw std::logic_error("Individual timestep scheduler has no cells");

    return prepareEvent(cells, nextEventTick());
}

IndividualStepContext IndividualTimeStepScheduler::prepareEvent(
    const std::vector<ComputationalCell3D> &cells,
    std::uint64_t event_tick) const
{
    return prepareEvent(cells, event_tick, false);
}

IndividualStepContext IndividualTimeStepScheduler::prepareEvent(
    const std::vector<ComputationalCell3D> &cells,
    std::uint64_t event_tick,
    bool force_all_active) const
{
    if(!initialized_)
        throw std::logic_error("Individual timestep scheduler is not initialized");
    validateCells(cells);
    if(event_tick <= current_tick_)
        throw std::logic_error("Individual timestep scheduler produced a non-advancing event");
    // Catches writes made between events outside the scheduler that skipped rebuildIndex.
    if(CountsMode == 2)
        VerifyScheduleCounts("prepareEvent");
    const std::uint64_t local_next = nextEventTick();
    if(event_tick > local_next)
        throw std::logic_error("Collective individual event skipped a local cell end tick");

    IndividualStepContext result;
    result.previous_event_tick = current_tick_;
    result.event_tick = event_tick;
    result.time_origin = time_origin_;
    result.previous_event_time = time_origin_ + time_quantum_ * static_cast<double>(current_tick_);
    result.event_time = time_origin_ + time_quantum_ * static_cast<double>(event_tick);
    result.time_quantum = time_quantum_;
    result.anchor_time_step = std::ldexp(time_quantum_, static_cast<int>(options_.initial_bin));
    result.mesh_build_policy = options_.mesh_build_policy;
    result.partial_build_fraction = options_.partial_build_fraction;
    result.verify_partial_build = options_.verify_partial_build;
    result.radiation_repair_accounting = &radiation_repair_accounting_;
    result.radiation_defect_accounting = &radiation_defect_accounting_;
    result.active_mask.assign(states_.size(), 0);
    result.cell_time_steps.resize(states_.size());
    result.cell_time_bins.resize(states_.size());
    result.primitive_ticks.resize(states_.size());
    result.point_velocities.resize(states_.size());
    result.radiation_accuracy_limits.resize(states_.size());
    result.cached_accelerations.resize(states_.size());
    result.gravity_half_kick_pending.resize(states_.size());

    for(std::size_t i = 0; i < states_.size(); ++i)
    {
        const CellTimeState &state = states_[i];
        result.point_velocities[i] = state.point_velocity;
        result.radiation_accuracy_limits[i] = state.radiation_accuracy_limit;
        result.cached_accelerations[i] = state.cached_acceleration;
        result.gravity_half_kick_pending[i] = state.gravity_half_kick_pending ? 1 : 0;
        if(force_all_active &&
           !(state.begin_tick < event_tick && event_tick <= state.end_tick))
            throw std::logic_error(
                "Forced individual event lies outside a cell's scheduled interval");
        const std::uint64_t step_end_tick = force_all_active ?
            event_tick : state.end_tick;
        result.cell_time_steps[i] = time_quantum_ *
            static_cast<double>(step_end_tick - state.begin_tick);
        result.cell_time_bins[i] = state.time_bin;
        result.primitive_ticks[i] = state.last_primitive_tick;
        if(force_all_active || state.end_tick == event_tick)
        {
            result.active_indices.push_back(i);
            result.active_mask[i] = 1;
        }
    }
    if(CountsMode == 2 && !force_all_active)
    {
        auto const Due = EndTickOccupancy.find(event_tick);
        std::uint64_t const Tracked = Due == EndTickOccupancy.end() ? 0 : Due->second;
        if(Tracked != static_cast<std::uint64_t>(result.active_indices.size()))
            AbortScheduleCounts("where=prepareEvent kind=due_count event_tick=" + std::to_string(event_tick) +
                " tracked=" + std::to_string(Tracked) + " scanned=" + std::to_string(result.active_indices.size()));
        ++CountsChecks;
    }
    return result;
}

bool IndividualOverdueActivationNominal()
{
    static bool const Enabled = []()
    {
        char const* const Value = std::getenv("RICH_INDIVIDUAL_OVERDUE_ACTIVATION_NOMINAL");
        return !(Value != nullptr && (std::string(Value) == "0" || std::string(Value) == "false" ||
                                      std::string(Value) == "off"));
    }();
    return Enabled;
}

void IndividualTimeStepScheduler::commitEvent(
    const IndividualStepContext &context,
    const Tessellation3D &tess,
    const std::vector<ComputationalCell3D> &cells,
    const std::vector<double> &time_step_limits,
    const std::vector<double> &signal_wake_deadlines,
    const std::vector<double> &change_wake_ratios)
{
    bool const OverdueNominal = IndividualOverdueActivationNominal();
    validateCells(cells);
    if(context.previous_event_tick != current_tick_)
        throw std::logic_error("Individual timestep event was prepared from stale scheduler state");
    if(time_step_limits.size() != states_.size())
        throw std::invalid_argument("Individual timestep limit vector has the wrong size");
    if(!signal_wake_deadlines.empty() &&
       signal_wake_deadlines.size() != states_.size())
        throw std::invalid_argument(
            "Individual signal-wake deadline vector has the wrong size");
    if(!change_wake_ratios.empty() && change_wake_ratios.size() != states_.size())
        throw std::invalid_argument("Individual change-wake vector has the wrong size");
    // Collective on its first call (the switch agreement); a no-op once tracking runs or when the switch is 0.
    ActivateScheduleCounts();
    committed_event_spacing_ = context.event_tick - context.previous_event_tick;
    // Stored before the synchronized branch returns; finalizeChangeWakes
    // applies them once AMR and the load balance have set every endpoint.
    if(!change_wake_ratios.empty())
        for(std::size_t index = 0; index < states_.size(); ++index)
            if(!context.isActive(index) && change_wake_ratios[index] > 0)
            {
                if(CountsMode != 0 && states_[index].change_wake_pending == 0)
                    ChangeWakeIndices.push_back(index);
                states_[index].change_wake_pending = 1;
                states_[index].change_wake_ratio = change_wake_ratios[index];
            }
    if(context.point_velocities.size() == states_.size())
        for(std::size_t i = 0; i < states_.size(); ++i)
            states_[i].point_velocity = context.point_velocities[i];
    if(context.radiation_accuracy_limits.size() == states_.size())
        for(std::size_t i = 0; i < states_.size(); ++i)
            states_[i].radiation_accuracy_limit = context.radiation_accuracy_limits[i];
    if(context.cached_accelerations.size() == states_.size())
        for(std::size_t i = 0; i < states_.size(); ++i)
            states_[i].cached_acceleration = context.cached_accelerations[i];
    if(context.gravity_half_kick_pending.size() == states_.size())
        for(std::size_t i = 0; i < states_.size(); ++i)
            states_[i].gravity_half_kick_pending =
                context.gravity_half_kick_pending[i] != 0;

    std::uint8_t const no_pending_bin =
        std::numeric_limits<std::uint8_t>::max();
    std::vector<std::uint8_t> propagation_bins(
        states_.size(), no_pending_bin);

    if(options_.force_synchronized)
    {
        if(context.active_indices.size() != states_.size())
            throw std::logic_error(
                "Synchronized individual event did not activate every cell");
        double shared_limit = std::numeric_limits<double>::infinity();
        for(double limit : time_step_limits)
            shared_limit = std::min(shared_limit, limit);
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &shared_limit, 1, MPI_DOUBLE, MPI_MIN,
            MPI_COMM_WORLD);
#endif
        std::uint8_t shared_bin = options_.maximum_bin;
        if(!states_.empty())
            shared_bin = chooseNextBin(states_.front(), context.event_tick,
                shared_limit);
#ifdef RICH_MPI
        unsigned int shared_bin_value = shared_bin;
        MPI_Allreduce(MPI_IN_PLACE, &shared_bin_value, 1, MPI_UNSIGNED,
            MPI_MIN, MPI_COMM_WORLD);
        shared_bin = static_cast<std::uint8_t>(shared_bin_value);
#endif
        // The shared bin also stays under the bin-spread ceiling.
        if(MaximumBinSpread() >= 0)
            shared_bin = static_cast<std::uint8_t>(std::min<unsigned>(shared_bin,
                std::min<unsigned>(options_.maximum_bin,
                    static_cast<unsigned>(options_.initial_bin) + static_cast<unsigned>(MaximumBinSpread()))));
        for(CellTimeState& state : states_)
        {
            state.begin_tick = context.event_tick;
            state.last_primitive_tick = context.event_tick;
            state.time_bin = shared_bin;
            state.pending_neighbor_bin = no_pending_bin;
            state.end_tick = checked_add(context.event_tick,
                ticksForBin(shared_bin));
        }
        // Every state changed, so the tracked counts are rebuilt.
        if(CountsMode != 0)
            RebuildScheduleCounts();
        // The shared bin is collective, so it is the finest bin in use.
        minimum_occupied_bin_ = shared_bin;
        needs_full_neighbor_closure_ = false;
        current_tick_ = context.event_tick;
        return;
    }

    unsigned long long bin_overruns = 0;
    double worst_bin_overrun = 0;
    std::size_t worst_bin_overrun_id = 0;
    for(std::size_t index : context.active_indices)
    {
        if(index >= states_.size())
            throw std::out_of_range("Active individual timestep cell is out of range");
        CellTimeState &state = states_[index];
        std::uint8_t const completed_bin = state.time_bin;
        std::uint64_t const completed_ticks =
            context.event_tick - state.begin_tick;
        std::uint64_t const allowance = ticksForBin(completed_bin);
        if(completed_ticks > allowance && allowance > 0)
        {
            ++bin_overruns;
            double const ratio = static_cast<double>(completed_ticks) /
                static_cast<double>(allowance);
            if(ratio > worst_bin_overrun)
            {
                worst_bin_overrun = ratio;
                worst_bin_overrun_id = state.cell_id;
            }
        }
        bool const completed_nominal_interval =
            state.end_tick == context.event_tick &&
            (OverdueNominal ?
                 context.event_tick - state.begin_tick >= ticksForBin(completed_bin) :
                 context.event_tick - state.begin_tick == ticksForBin(completed_bin));
        if(state.end_tick != context.event_tick)
        {
            if(state.begin_tick >= context.event_tick ||
               context.event_tick > state.end_tick)
                throw std::logic_error(
                    "Forced individual cell committed outside its scheduled interval");
        }
        // Wakes, aligned catch-up intervals, terminal clamps, and forced
        // all-active overlays may complete less than the retained physical
        // bin interval.  That scheduler interruption is not a timestep
        // estimate.  Preserve the bin here; chooseNextBin still applies any
        // genuinely smaller physical limit computed for this event.
        std::uint8_t const next_bin = chooseNextBin(
            state, context.event_tick, time_step_limits[index]);
        if(next_bin < completed_bin)
            propagation_bins[index] = next_bin;
        if(completed_nominal_interval &&
           state.pending_neighbor_bin != no_pending_bin)
        {
            propagation_bins[index] = std::min(
                propagation_bins[index], state.pending_neighbor_bin);
            state.pending_neighbor_bin = no_pending_bin;
        }
        std::uint64_t const traced_begin = state.begin_tick;
        std::uint64_t const traced_end = state.end_tick;
        state.begin_tick = context.event_tick;
        state.last_primitive_tick = context.event_tick;
        state.time_bin = next_bin;
        state.end_tick = nextAlignedTick(context.event_tick, state.time_bin);
        if(CountsMode != 0)
            NoteScheduleChange(completed_bin, traced_end, state);
        TraceSchedulerMutation("activate", state.cell_id, context.event_tick, traced_begin, completed_bin,
            next_bin, traced_end, state.end_tick);
    }

    for(std::size_t index = 0; index < states_.size(); ++index)
    {
        if(context.isActive(index))
            continue;
        double const limit = time_step_limits[index];
        if(std::isinf(limit) && limit > 0)
            continue;
        std::uint8_t const limited_bin = quantizeTimeStep(limit);
        std::uint8_t const target_bin = std::min(
            states_[index].time_bin, limited_bin);
        std::uint64_t const limited_end = binnedEndTick(
            states_[index].begin_tick, context.event_tick, target_bin);
        if(limited_end < states_[index].end_tick)
        {
            std::uint8_t const old_bin = states_[index].time_bin;
            std::uint64_t const old_end = states_[index].end_tick;
            if(target_bin < old_bin)
                noteLowering(0, states_[index].begin_tick, context.event_tick, target_bin);
            states_[index].time_bin = target_bin;
            states_[index].end_tick = limited_end;
            if(CountsMode != 0)
                NoteScheduleChange(old_bin, old_end, states_[index]);
            TraceSchedulerMutation("passive_limit", states_[index].cell_id, context.event_tick,
                states_[index].begin_tick, old_bin, target_bin, old_end, limited_end);
            if(states_[index].time_bin < old_bin)
                propagation_bins[index] = states_[index].time_bin;
        }
    }

    // A signal arrival changes when a passive cell next becomes active, not
    // the physical timestep it should select after that interrupted update.
    // Keep time_bin unchanged so chooseNextBin starts from the pre-wake bin.
    if(!signal_wake_deadlines.empty())
        for(std::size_t index = 0; index < states_.size(); ++index)
        {
            if(context.isActive(index))
                continue;
            double const wake_interval = signal_wake_deadlines[index];
            if(std::isinf(wake_interval) && wake_interval > 0)
                continue;
            std::uint8_t const wake_bin = quantizeTimeStep(wake_interval);
            // Wake intervals are measured from this event. Aligning them to
            // the global bin grid could turn an interval of many ticks into a
            // one-tick wake when event_tick is just below a bin boundary.
            std::uint64_t const wake_ticks = ticksForBin(wake_bin);
            std::uint64_t const remaining_ticks =
                states_[index].end_tick - context.event_tick;
            if(wake_ticks < remaining_ticks)
            {
                std::uint64_t const old_end = states_[index].end_tick;
                states_[index].end_tick = checked_add(
                    context.event_tick, wake_ticks);
                if(CountsMode != 0)
                    NoteScheduleChange(states_[index].time_bin, old_end, states_[index]);
                TraceSchedulerMutation("signal_wake", states_[index].cell_id, context.event_tick,
                    states_[index].begin_tick, states_[index].time_bin, states_[index].time_bin, old_end,
                    states_[index].end_tick);
            }
        }

    ReportIndividualBinOverrun(bin_overruns, worst_bin_overrun,
                               worst_bin_overrun_id, context.event_tick);

    refreshMinimumOccupiedBin();

    if(needs_full_neighbor_closure_)
    {
        std::vector<std::uint8_t> const no_sources(
            states_.size(), no_pending_bin);
        limitNeighborBins(
            tess, cells, context.event_tick, no_sources, true);
        needs_full_neighbor_closure_ = false;
    }

    int has_propagation_source = std::any_of(
        propagation_bins.begin(), propagation_bins.end(),
        [no_pending_bin](std::uint8_t const bin)
        { return bin != no_pending_bin; }) ? 1 : 0;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &has_propagation_source, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if(has_propagation_source != 0)
        limitNeighborBins(
            tess, cells, context.event_tick, propagation_bins, false);
    // Every bin of this event is final (the propagation above can still lower
    // the minimum), so the spread cap goes last.
    applyBinSpreadCap(context.event_tick);
    current_tick_ = context.event_tick;
    if(CountsMode == 2)
    {
        VerifyScheduleCounts("commitEvent");
        // Rank 0's own checks, every 256 commits: evidence that the verify mode ran (no reduction).
        if(++CountsCommits % 256 == 0 && WorldRank() == 0)
            std::cout << "RICH_SCHED_COUNTS_VERIFY tick=" << current_tick_ << " commits=" << CountsCommits
                      << " checks_rank0=" << CountsChecks << " mismatches=0" << std::endl;
    }
}

void IndividualTimeStepScheduler::applyBinSpreadCap(std::uint64_t const event_tick)
{
    if(MaximumBinSpread() < 0)
        return;
    // A fixed ceiling, the anchor bin (initial_bin, m x the global CFL
    // suggestion) + K: every cell above it takes it with the begin-aware end
    // (binnedEndTick: an overdue cell ends at the next finest-bin tick).  A
    // common ceiling cannot widen any neighbour difference, so no closure
    // follows.  It is not tied to the finest bin: on the TDE continuation one
    // bin-26 cell (radiation retry limiter) held 8.9M cells at bin 28 under a
    // finest-bin + K cap (job 10222326, t ~ 30.35).  Cells may still take any
    // finer bin their limits or the closure require.  Collective (the minimum
    // is refreshed for the begin-aware ends).
    refreshMinimumOccupiedBin();
    unsigned const cap = std::min<unsigned>(options_.maximum_bin,
        static_cast<unsigned>(options_.initial_bin) + static_cast<unsigned>(MaximumBinSpread()));
    // With tracked counts the loop below runs only when some state lies above the cap; for every other state it
    // does nothing.
    std::uint64_t AboveCap = 0;
    if(CountsMode != 0)
        for(unsigned Bin = cap + 1; Bin < BinOccupancy.size(); ++Bin)
            AboveCap += BinOccupancy[Bin];
    std::uint64_t Capped = 0;
    if(CountsMode != 1 || AboveCap > 0)
        for(CellTimeState& state : states_)
        {
            if(state.time_bin <= cap)
                continue;
            std::uint8_t const capped = static_cast<std::uint8_t>(cap);
            noteLowering(3, state.begin_tick, event_tick, capped);
            unsigned const old_bin = state.time_bin;
            std::uint64_t const old_end = state.end_tick;
            state.time_bin = capped;
            state.end_tick = std::min(state.end_tick, binnedEndTick(state.begin_tick, event_tick, capped));
            if(CountsMode != 0)
            {
                NoteScheduleChange(static_cast<std::uint8_t>(old_bin), old_end, state);
                ++Capped;
            }
            TraceSchedulerMutation("spread_cap", state.cell_id, event_tick, state.begin_tick, old_bin, capped,
                old_end, state.end_tick);
        }
    if(CountsMode == 2)
    {
        if(Capped != AboveCap)
            AbortScheduleCounts("where=applyBinSpreadCap kind=above_cap cap=" + std::to_string(cap) + " tracked=" +
                std::to_string(AboveCap) + " scanned=" + std::to_string(Capped));
        ++CountsChecks;
    }
    // The ceiling can lower the minimum itself (every cell above it): if the
    // global minimum was above the cap, every cell was capped to it.
    minimum_occupied_bin_ = static_cast<std::uint8_t>(std::min<unsigned>(minimum_occupied_bin_, cap));
}

std::size_t IndividualTimeStepScheduler::clampToTerminalTick(
    std::uint64_t terminal_tick)
{
    if(!initialized_)
        throw std::logic_error(
            "Cannot clamp an uninitialized individual timestep scheduler");
    if(terminal_tick <= current_tick_)
        throw std::invalid_argument(
            "Individual timestep terminal tick must follow the current tick");

    std::size_t clamped = 0;
    for(CellTimeState& state : states_)
    {
        if(state.end_tick > terminal_tick)
        {
            std::uint64_t const OldEnd = state.end_tick;
            state.end_tick = terminal_tick;
            if(CountsMode != 0)
                NoteScheduleChange(state.time_bin, OldEnd, state);
            ++clamped;
        }
    }
    if(CountsMode == 2)
        VerifyScheduleCounts("clampToTerminalTick");
    return clamped;
}

double IndividualTimeStepScheduler::nextEventTimeStep(void) const
{
    if(!initialized_)
        return 0;
    const std::uint64_t next_tick = nextEventTick();
    if(next_tick == std::numeric_limits<std::uint64_t>::max())
        return std::numeric_limits<double>::infinity();
    if(next_tick <= current_tick_)
        return 0;
    return time_quantum_ * static_cast<double>(next_tick - current_tick_);
}

std::uint64_t IndividualTimeStepScheduler::nextEventTick(void) const
{
    if(!initialized_)
        return std::numeric_limits<std::uint64_t>::max();
    if(CountsMode == 0)
        return ScannedNextEventTick();
    std::uint64_t const Tracked = EndTickOccupancy.empty() ? std::numeric_limits<std::uint64_t>::max() :
        EndTickOccupancy.begin()->first;
    if(CountsMode == 1)
        return Tracked;
    // Mode 2: the scan's value is returned and the tracked one must equal it.
    std::uint64_t const next_tick = ScannedNextEventTick();
    if(Tracked != next_tick)
    {
        std::size_t ExampleId = 0;
        for(CellTimeState const& State : states_)
            if(State.end_tick == next_tick)
            {
                ExampleId = State.cell_id;
                break;
            }
        AbortScheduleCounts("where=nextEventTick kind=minimum_end_tick tracked=" + std::to_string(Tracked) +
            " scanned=" + std::to_string(next_tick) + " cell_id=" + std::to_string(ExampleId));
    }
    ++CountsChecks;
    return next_tick;
}

std::uint64_t IndividualTimeStepScheduler::ScannedNextEventTick(void) const
{
    std::uint64_t next_tick = std::numeric_limits<std::uint64_t>::max();
    for(const CellTimeState &state : states_)
        next_tick = std::min(next_tick, state.end_tick);
    return next_tick;
}

void IndividualTimeStepScheduler::setInitialPointVelocities(
    std::vector<std::pair<std::size_t, Vector3D> > const& velocities)
{
    if(!initialized_ || current_tick_ != 0)
        throw std::logic_error("Initial point velocities are set only on a fresh timeline");
    for(auto const& entry : velocities)
    {
        auto const found = id_to_index_.find(entry.first);
        if(found != id_to_index_.end())
            states_[found->second].point_velocity = entry.second;
    }
}

void IndividualTimeStepScheduler::resetTimeOrigin(double current_time)
{
    if(initialized_ && current_tick_ != 0)
        throw std::logic_error("Cannot reset an active individual timestep timeline");
    time_origin_ = current_time;
}

void IndividualTimeStepScheduler::rebuildIndex(const std::vector<ComputationalCell3D> &cells)
{
    if(states_.size() != cells.size())
        throw std::invalid_argument("Individual timestep state and cell counts differ");
    id_to_index_.clear();
    id_to_index_.reserve(cells.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        if(!id_to_index_.emplace(cells[i].ID, i).second)
            throw std::invalid_argument("Duplicate cell ID in individual timestep scheduler");
        states_[i].cell_id = cells[i].ID;
    }
    // Every reorder of states_ (AMR remap, migration, domain growth, restore) ends here, so the tracked counts and
    // the change-wake indices are rebuilt for the new order.
    if(CountsMode != 0)
        RebuildScheduleCounts();
}

void IndividualTimeStepScheduler::restore(
    const std::vector<ComputationalCell3D>& cells,
    double time_origin,
    double time_quantum,
    std::uint64_t current_tick,
    std::vector<CellTimeState> states,
    RadiationRepairAccounting repair_accounting,
    IndividualRadiationDefectAccounting defect_accounting,
    bool force_all_active_latched,
    std::uint64_t last_full_source_sweep_tick)
{
    if(initialized_)
        throw std::logic_error("Cannot restore over an initialized individual timestep scheduler");
    if(!(std::isfinite(time_origin) && std::isfinite(time_quantum) &&
         time_quantum > 0))
        throw std::invalid_argument("Invalid individual timestep restart clock");
    if(last_full_source_sweep_tick > current_tick)
        throw std::invalid_argument(
            "Individual full-source sweep tick exceeds restart time");
    if(states.size() != cells.size())
        throw std::invalid_argument("Individual timestep restart state count differs from cells");
    bool const residual_bias_vectors_valid =
        repair_accounting.residual_correction_signed_bias_by_group.size() ==
            repair_accounting.
                residual_correction_absolute_bias_by_group.size() &&
        std::all_of(
            repair_accounting.residual_correction_signed_bias_by_group.begin(),
            repair_accounting.residual_correction_signed_bias_by_group.end(),
            [](double const value) { return std::isfinite(value); }) &&
        std::all_of(
            repair_accounting.
                residual_correction_absolute_bias_by_group.begin(),
            repair_accounting.
                residual_correction_absolute_bias_by_group.end(),
            [](double const value)
            { return std::isfinite(value) && value >= 0; });
    if(!std::isfinite(repair_accounting.cumulative_injected_energy) ||
       repair_accounting.cumulative_injected_energy < 0 ||
       !std::isfinite(repair_accounting.maximum_relative_deficit) ||
       repair_accounting.maximum_relative_deficit < 0 ||
       !std::isfinite(repair_accounting.maximum_global_radiation_energy) ||
       repair_accounting.maximum_global_radiation_energy < 0 ||
       !std::isfinite(repair_accounting.next_warning_fraction) ||
       repair_accounting.next_warning_fraction < 1e-4 ||
       !std::isfinite(
           repair_accounting.residual_correction_signed_energy_bias) ||
       !std::isfinite(
           repair_accounting.residual_correction_absolute_energy_bias) ||
       repair_accounting.residual_correction_absolute_energy_bias < 0 ||
       !std::isfinite(
           repair_accounting.residual_correction_minimum_scale) ||
       repair_accounting.residual_correction_minimum_scale < 0 ||
       repair_accounting.residual_correction_minimum_scale > 1 ||
       !residual_bias_vectors_valid ||
       !std::isfinite(repair_accounting.
           residual_positive_floor_cumulative_injected_energy) ||
       repair_accounting.
           residual_positive_floor_cumulative_injected_energy < 0 ||
       !std::isfinite(repair_accounting.
           residual_positive_floor_maximum_cell_injection_ratio) ||
       repair_accounting.
           residual_positive_floor_maximum_cell_injection_ratio < 0 ||
       !std::isfinite(repair_accounting.
           residual_positive_floor_maximum_global_injection_ratio) ||
       repair_accounting.
           residual_positive_floor_maximum_global_injection_ratio < 0 ||
       !std::isfinite(repair_accounting.
           residual_positive_floor_maximum_post_true_residual_error) ||
       repair_accounting.
           residual_positive_floor_maximum_post_true_residual_error < 0 ||
       !std::isfinite(repair_accounting.
           residual_positive_floor_initial_global_radiation_energy) ||
       repair_accounting.
           residual_positive_floor_initial_global_radiation_energy < 0)
        throw std::invalid_argument(
            "Invalid individual timestep radiation repair accounting");

    long double const defect_balance_scale = std::max(
        std::abs(defect_accounting.cumulative_signed_extent),
        defect_accounting.cumulative_absolute_extent);
    long double const defect_balance_slack =
        64 * std::numeric_limits<long double>::epsilon() *
        defect_balance_scale;
    bool const defect_accounting_valid =
        defect_accounting.config_version == 3 &&
        std::isfinite(
            static_cast<double>(defect_accounting.cumulative_signed_extent)) &&
        std::isfinite(
            static_cast<double>(defect_accounting.cumulative_absolute_extent)) &&
        defect_accounting.cumulative_absolute_extent >= 0 &&
        std::abs(defect_accounting.cumulative_signed_extent) <=
            defect_accounting.cumulative_absolute_extent +
                defect_balance_slack &&
        std::isfinite(defect_accounting.initial_positive_global_extent) &&
        defect_accounting.initial_positive_global_extent >= 0 &&
        std::isfinite(defect_accounting.last_normalization_scale) &&
        defect_accounting.last_normalization_scale >= 0 &&
        std::isfinite(defect_accounting.maximum_event_absolute_fraction) &&
        defect_accounting.maximum_event_absolute_fraction >= 0 &&
        std::isfinite(defect_accounting.maximum_local_fraction) &&
        defect_accounting.maximum_local_fraction >= 0 &&
        std::isfinite(
            defect_accounting.maximum_local_tolerance_ratio) &&
        defect_accounting.maximum_local_tolerance_ratio >= 0 &&
        std::isfinite(defect_accounting.local_withdrawal_limit) &&
        defect_accounting.local_withdrawal_limit > 0 &&
        std::isfinite(defect_accounting.local_absolute_limit) &&
        defect_accounting.local_absolute_limit > 0 &&
        std::isfinite(defect_accounting.event_absolute_target) &&
        defect_accounting.event_absolute_target > 0 &&
        std::isfinite(defect_accounting.cumulative_signed_limit) &&
        defect_accounting.cumulative_signed_limit > 0 &&
        std::isfinite(defect_accounting.cumulative_absolute_limit) &&
        defect_accounting.cumulative_absolute_limit > 0 &&
        defect_accounting.cooldown_required_candidates > 0 &&
        std::isfinite(defect_accounting.cooldown_fraction_ceiling) &&
        defect_accounting.cooldown_fraction_ceiling > 0 &&
        defect_accounting.cooldown_fraction_ceiling <= 1;
    if(!defect_accounting_valid)
        throw std::invalid_argument(
            "Invalid individual timestep radiation defect accounting");
    for(std::size_t i = 0; i < states.size(); ++i)
    {
        if(states[i].cell_id != cells[i].ID)
            throw std::invalid_argument("Individual timestep restart IDs are not in canonical cell order");
        if(states[i].time_bin > options_.maximum_bin ||
           (states[i].pending_neighbor_bin !=
                std::numeric_limits<std::uint8_t>::max() &&
            states[i].pending_neighbor_bin > options_.maximum_bin) ||
           states[i].begin_tick > current_tick ||
           states[i].end_tick <= current_tick ||
           states[i].last_primitive_tick != states[i].begin_tick)
            throw std::invalid_argument("Invalid individual timestep restart tick/bin state");
    }
    time_origin_ = time_origin;
    time_quantum_ = time_quantum;
    current_tick_ = current_tick;
    last_full_source_sweep_tick_ = last_full_source_sweep_tick;
    states_ = std::move(states);
    radiation_repair_accounting_ = repair_accounting;
    radiation_defect_accounting_ = defect_accounting;
    setForceAllActiveLatched(force_all_active_latched);
    rebuildIndex(cells);
    // The restart does not store the finest occupied bin.  Left at its
    // default of 0, binnedEndTick would send every overdue cell to the tick
    // after this one during the closure that follows the restore.
    refreshMinimumOccupiedBin();
    // A checkpoint written without the cap (or with a larger one) is capped
    // before its first event.
    applyBinSpreadCap(current_tick_);
    needs_full_neighbor_closure_ = true;
    initialized_ = true;
}

int IndividualTimeStepScheduler::maximumBinSpread(void)
{
    return MaximumBinSpread();
}

bool IndividualTimeStepScheduler::changeWakesJoinNextEvent(void)
{
    // First called from finalizeChangeWakes, on every rank; selects a
    // collective, so the rule must agree.
    static bool const join = []()
    {
        char const* const value = std::getenv("RICH_INDIVIDUAL_CHANGE_WAKE_RULE");
        std::string const rule = value == nullptr ? std::string() : std::string(value);
        int code[2] = {rule.empty() || rule == "next_event" ? 1 : (rule == "spacing_ticks" ? 0 : -1), 0};
        code[1] = -code[0];
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, code, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
        if(code[0] < 0 || code[0] != -code[1])
            throw std::invalid_argument(
                "RICH_INDIVIDUAL_CHANGE_WAKE_RULE must be next_event or spacing_ticks on every rank");
        return code[0] == 1;
    }();
    return join;
}

void IndividualTimeStepScheduler::noteLowering(int const cause, std::uint64_t const begin_tick,
    std::uint64_t const event_tick, std::uint8_t const new_bin)
{
    ++lowered_[cause];
    std::uint64_t const allowance = ticksForBin(new_bin);
    std::uint64_t const elapsed = begin_tick < event_tick ? event_tick - begin_tick : 0;
    lowered_worst_ratio_[cause] = std::max(lowered_worst_ratio_[cause],
        static_cast<double>(elapsed) / static_cast<double>(allowance));
    if(elapsed >= allowance)
        ++lowered_overdue_[cause];
}

IndividualTimeStepScheduler::ChangeWakeFinalization
IndividualTimeStepScheduler::finalizeChangeWakes(bool const collect_statistics)
{
    ChangeWakeFinalization result;
    result.next_tick_before = nextEventTick();
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &result.next_tick_before, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
#endif
    std::uint64_t next_tick = result.next_tick_before;
    if(!changeWakesJoinNextEvent())
    {
        // Reference rule: the wake ends one power-of-two below the spacing of
        // the event just committed, measured from it, in exact ticks.
        std::uint64_t spacing = 1;
        while(committed_event_spacing_ > 0 && spacing <= committed_event_spacing_ / 2)
            spacing *= 2;
        next_tick = checked_add(current_tick_, spacing);
    }
    auto ApplyChangeWake = [&](CellTimeState& state)
    {
        ++result.applied;
        result.largest_ratio = std::max(result.largest_ratio, state.change_wake_ratio);
        if(next_tick < state.end_tick)
        {
            TraceSchedulerMutation("change_wake", state.cell_id, current_tick_, state.begin_tick, state.time_bin,
                state.time_bin, state.end_tick, next_tick);
            std::uint64_t const OldEnd = state.end_tick;
            state.end_tick = next_tick;
            if(CountsMode != 0)
                NoteScheduleChange(state.time_bin, OldEnd, state);
            ++result.shortened;
        }
        state.change_wake_pending = 0;
        state.change_wake_ratio = 0;
    };
    if(CountsMode == 1)
    {
        // The tracked indices in ascending order, the order of the scan below.
        std::sort(ChangeWakeIndices.begin(), ChangeWakeIndices.end());
        ChangeWakeIndices.erase(std::unique(ChangeWakeIndices.begin(), ChangeWakeIndices.end()),
                                ChangeWakeIndices.end());
        for(std::size_t const Index : ChangeWakeIndices)
        {
            if(Index >= states_.size())
                AbortScheduleCounts("where=finalizeChangeWakes kind=wake_index index=" + std::to_string(Index) +
                    " states=" + std::to_string(states_.size()));
            if(states_[Index].change_wake_pending != 0)
                ApplyChangeWake(states_[Index]);
        }
    }
    else
    {
        std::vector<std::size_t> ScannedWakes;
        for(CellTimeState& state : states_)
        {
            if(state.change_wake_pending == 0)
                continue;
            if(CountsMode == 2)
                ScannedWakes.push_back(static_cast<std::size_t>(&state - states_.data()));
            ApplyChangeWake(state);
        }
        if(CountsMode == 2)
        {
            std::vector<std::size_t> TrackedWakes = ChangeWakeIndices;
            std::sort(TrackedWakes.begin(), TrackedWakes.end());
            TrackedWakes.erase(std::unique(TrackedWakes.begin(), TrackedWakes.end()), TrackedWakes.end());
            if(TrackedWakes != ScannedWakes)
            {
                std::size_t Example = 0;
                for(std::size_t k = 0; k < std::max(TrackedWakes.size(), ScannedWakes.size()); ++k)
                    if(k >= TrackedWakes.size() || k >= ScannedWakes.size() || TrackedWakes[k] != ScannedWakes[k])
                    {
                        Example = k < ScannedWakes.size() ? ScannedWakes[k] : TrackedWakes[k];
                        break;
                    }
                AbortScheduleCounts("where=finalizeChangeWakes kind=wake_set tracked=" +
                    std::to_string(TrackedWakes.size()) + " scanned=" + std::to_string(ScannedWakes.size()) +
                    " first_difference_index=" + std::to_string(Example));
            }
            ++CountsChecks;
        }
    }
    if(CountsMode != 0)
        ChangeWakeIndices.clear();
    // Every end tick is still at least next_tick and the old minimum is
    // untouched, so the next event is unchanged; the caller's own reduction
    // checks it.
    result.next_tick_after = result.next_tick_before;
    if(!changeWakesJoinNextEvent())
    {
        result.next_tick_after = nextEventTick();
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &result.next_tick_after, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
#endif
    }
    result.merge_wakes = merge_change_wakes_;
    merge_change_wakes_ = 0;
    for(int cause = 0; cause < 4; ++cause)
    {
        result.lowered[cause] = lowered_[cause];
        result.lowered_overdue[cause] = lowered_overdue_[cause];
        result.lowered_worst_ratio[cause] = lowered_worst_ratio_[cause];
        lowered_[cause] = 0;
        lowered_overdue_[cause] = 0;
        lowered_worst_ratio_[cause] = 0;
    }
#ifdef RICH_MPI
    if(collect_statistics)
    {
        std::uint64_t counts[3] = {result.applied, result.shortened, result.merge_wakes};
        MPI_Allreduce(MPI_IN_PLACE, counts, 3, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        result.applied = counts[0];
        result.shortened = counts[1];
        result.merge_wakes = counts[2];
        MPI_Allreduce(MPI_IN_PLACE, &result.largest_ratio, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        std::uint64_t lowered[8];
        std::copy(result.lowered, result.lowered + 4, lowered);
        std::copy(result.lowered_overdue, result.lowered_overdue + 4, lowered + 4);
        MPI_Allreduce(MPI_IN_PLACE, lowered, 8, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        std::copy(lowered, lowered + 4, result.lowered);
        std::copy(lowered + 4, lowered + 8, result.lowered_overdue);
        MPI_Allreduce(MPI_IN_PLACE, result.lowered_worst_ratio, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
#else
    (void)collect_statistics;
#endif
    if(CountsMode == 2)
        VerifyScheduleCounts("finalizeChangeWakes");
    return result;
}

void IndividualTimeStepScheduler::refreshMinimumOccupiedBin(void)
{
    unsigned int smallest = options_.maximum_bin;
    if(CountsMode == 0)
        for(CellTimeState const& state : states_)
            smallest = std::min(smallest, static_cast<unsigned int>(state.time_bin));
    else
    {
        // The lowest occupied bin, capped at maximum_bin exactly as the scan caps it.
        unsigned int Tracked = options_.maximum_bin;
        for(unsigned int Bin = 0; Bin < options_.maximum_bin; ++Bin)
            if(BinOccupancy[Bin] != 0)
            {
                Tracked = Bin;
                break;
            }
        smallest = Tracked;
        if(CountsMode == 2)
        {
            unsigned int Scanned = options_.maximum_bin;
            for(CellTimeState const& state : states_)
                Scanned = std::min(Scanned, static_cast<unsigned int>(state.time_bin));
            if(Scanned != Tracked)
                AbortScheduleCounts("where=refreshMinimumOccupiedBin kind=minimum_bin tracked=" +
                    std::to_string(Tracked) + " scanned=" + std::to_string(Scanned));
            ++CountsChecks;
        }
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &smallest, 1, MPI_UNSIGNED, MPI_MIN, MPI_COMM_WORLD);
#endif
    minimum_occupied_bin_ = static_cast<std::uint8_t>(smallest);
}

void IndividualTimeStepScheduler::recordFullSourceSweep(
    std::uint64_t const event_tick)
{
    if(!initialized_ || event_tick != current_tick_ ||
       event_tick <= last_full_source_sweep_tick_)
        throw std::logic_error(
            "Invalid individual full-source sweep completion tick");
    last_full_source_sweep_tick_ = event_tick;
}

void IndividualTimeStepScheduler::ActivateScheduleCounts(void)
{
    if(CountsMode != 0)
        return;
    int const Mode = SchedulerCountsSwitch();
    if(Mode == 0)
        return;
    CountsMode = Mode;
    RebuildScheduleCounts();
    if(WorldRank() == 0)
        std::cout << "RICH_SCHED_COUNTS mode=" << CountsMode << " tick=" << current_tick_
                  << " states_rank0=" << states_.size() << std::endl;
}

void IndividualTimeStepScheduler::RebuildScheduleCounts(void)
{
    BinOccupancy.fill(0);
    EndTickOccupancy.clear();
    ChangeWakeIndices.clear();
    auto Hint = EndTickOccupancy.end();
    for(std::size_t Index = 0; Index < states_.size(); ++Index)
    {
        CellTimeState const& State = states_[Index];
        ++BinOccupancy[State.time_bin];
        // Neighbouring states usually share an end tick, so the last entry is the insertion hint.
        if(Hint != EndTickOccupancy.end() && Hint->first == State.end_tick)
            ++Hint->second;
        else
        {
            Hint = EndTickOccupancy.emplace(State.end_tick, 0).first;
            ++Hint->second;
        }
        if(State.change_wake_pending != 0)
            ChangeWakeIndices.push_back(Index);
    }
}

void IndividualTimeStepScheduler::NoteScheduleChange(std::uint8_t const OldBin, std::uint64_t const OldEnd,
    CellTimeState const& State)
{
    if(OldBin != State.time_bin)
    {
        if(BinOccupancy[OldBin] == 0)
            AbortScheduleCounts("where=NoteScheduleChange kind=bin_underflow bin=" + std::to_string(OldBin) +
                " cell_id=" + std::to_string(State.cell_id));
        --BinOccupancy[OldBin];
        ++BinOccupancy[State.time_bin];
    }
    if(OldEnd != State.end_tick)
    {
        auto const Old = EndTickOccupancy.find(OldEnd);
        if(Old == EndTickOccupancy.end() || Old->second == 0)
            AbortScheduleCounts("where=NoteScheduleChange kind=end_tick_underflow end_tick=" +
                std::to_string(OldEnd) + " cell_id=" + std::to_string(State.cell_id));
        if(--Old->second == 0)
            EndTickOccupancy.erase(Old);
        ++EndTickOccupancy[State.end_tick];
    }
}

void IndividualTimeStepScheduler::VerifyScheduleCounts(char const* const Where) const
{
    std::array<std::uint64_t, 256> Bins{};
    std::map<std::uint64_t, std::uint64_t> Ends;
    std::vector<std::size_t> Wakes;
    for(std::size_t Index = 0; Index < states_.size(); ++Index)
    {
        ++Bins[states_[Index].time_bin];
        ++Ends[states_[Index].end_tick];
        if(states_[Index].change_wake_pending != 0)
            Wakes.push_back(Index);
    }
    auto const FirstCell = [this](auto const& Matches)
    {
        for(CellTimeState const& State : states_)
            if(Matches(State))
                return std::to_string(State.cell_id);
        return std::string("none");
    };
    for(std::size_t Bin = 0; Bin < Bins.size(); ++Bin)
        if(Bins[Bin] != BinOccupancy[Bin])
            AbortScheduleCounts(std::string("where=") + Where + " kind=bin bin=" + std::to_string(Bin) +
                " tracked=" + std::to_string(BinOccupancy[Bin]) + " scanned=" + std::to_string(Bins[Bin]) +
                " cell_id=" + FirstCell([Bin](CellTimeState const& State) { return State.time_bin == Bin; }));
    if(Ends != EndTickOccupancy)
    {
        // The first end tick whose counts differ, from either side.
        auto Scanned = Ends.begin();
        auto Tracked = EndTickOccupancy.begin();
        while(Scanned != Ends.end() && Tracked != EndTickOccupancy.end() && *Scanned == *Tracked)
        {
            ++Scanned;
            ++Tracked;
        }
        std::uint64_t Tick = 0;
        if(Scanned == Ends.end())
            Tick = Tracked->first;
        else if(Tracked == EndTickOccupancy.end())
            Tick = Scanned->first;
        else
            Tick = std::min(Scanned->first, Tracked->first);
        auto const ScannedAt = Ends.find(Tick);
        auto const TrackedAt = EndTickOccupancy.find(Tick);
        AbortScheduleCounts(std::string("where=") + Where + " kind=end_tick end_tick=" + std::to_string(Tick) +
            " tracked=" + std::to_string(TrackedAt == EndTickOccupancy.end() ? 0 : TrackedAt->second) +
            " scanned=" + std::to_string(ScannedAt == Ends.end() ? 0 : ScannedAt->second) + " cell_id=" +
            FirstCell([Tick](CellTimeState const& State) { return State.end_tick == Tick; }));
    }
    std::vector<std::size_t> TrackedWakes = ChangeWakeIndices;
    std::sort(TrackedWakes.begin(), TrackedWakes.end());
    TrackedWakes.erase(std::unique(TrackedWakes.begin(), TrackedWakes.end()), TrackedWakes.end());
    if(TrackedWakes != Wakes)
        AbortScheduleCounts(std::string("where=") + Where + " kind=wake_set tracked=" +
            std::to_string(TrackedWakes.size()) + " scanned=" + std::to_string(Wakes.size()));
    ++CountsChecks;
}

void IndividualTimeStepScheduler::applyAMRChangeSet(
    const std::vector<ComputationalCell3D>& cells,
    const IndividualAMRChangeSet& changes,
    const Tessellation3D* tess)
{
    if(!initialized_)
        throw std::logic_error("Cannot remap AMR before individual timestep initialization");

    std::unordered_map<std::size_t, CellTimeState> old_states;
    old_states.reserve(states_.size());
    for(CellTimeState const& state : states_)
        old_states.emplace(state.cell_id, state);
    std::unordered_map<std::size_t, std::size_t> child_to_parent;
    child_to_parent.reserve(changes.child_parent_ids.size());
    for(auto const& entry : changes.child_parent_ids)
        if(!child_to_parent.emplace(entry.first, entry.second).second)
            throw std::invalid_argument("AMR change set contains a duplicate child ID");
    std::unordered_set<std::size_t> removed(
        changes.removed_cell_ids.begin(), changes.removed_cell_ids.end());

    std::vector<CellTimeState> remapped(cells.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        std::size_t const id = cells[i].ID;
        auto existing = old_states.find(id);
        if(existing != old_states.end())
            remapped[i] = existing->second;
        else
        {
            auto parent_id = child_to_parent.find(id);
            if(parent_id == child_to_parent.end())
                throw std::invalid_argument("New AMR cell has no parent timestep state");
            auto parent = old_states.find(parent_id->second);
            if(parent == old_states.end())
                throw std::invalid_argument("AMR child references an unknown parent ID");
            remapped[i] = parent->second;
            remapped[i].cell_id = id;
        }
        if(removed.count(id) != 0)
            throw std::invalid_argument("AMR change set keeps an ID marked as removed");
    }

    std::unordered_map<std::size_t, std::size_t> remapped_index;
    remapped_index.reserve(remapped.size());
    for(std::size_t i = 0; i < remapped.size(); ++i)
        if(!remapped_index.emplace(remapped[i].cell_id, i).second)
            throw std::invalid_argument(
                "AMR result contains a duplicate stable cell ID");

    auto apply_merged_state = [&](std::size_t recipient_id,
                                  std::uint8_t source_bin,
                                  std::uint8_t pending_bin,
                                  double source_accuracy_limit)
    {
        auto const recipient = remapped_index.find(recipient_id);
        if(recipient == remapped_index.end())
            return false;
        CellTimeState& recipient_state = remapped[recipient->second];
        if(recipient_state.time_bin > source_bin)
        {
            noteLowering(2, recipient_state.begin_tick, current_tick_, source_bin);
            unsigned const old_bin = recipient_state.time_bin;
            std::uint64_t const old_end = recipient_state.end_tick;
            recipient_state.time_bin = source_bin;
            recipient_state.end_tick = std::min(
                recipient_state.end_tick,
                binnedEndTick(recipient_state.begin_tick, current_tick_,
                              source_bin));
            TraceSchedulerMutation("merge", recipient_state.cell_id, current_tick_, recipient_state.begin_tick,
                old_bin, source_bin, old_end, recipient_state.end_tick);
        }
        recipient_state.pending_neighbor_bin = std::min(
            recipient_state.pending_neighbor_bin,
            pending_bin);
        // The recipient's interval now carries the donor's material: keep the
        // tighter pending radiation accuracy limit.
        if(source_accuracy_limit > 0 && (recipient_state.radiation_accuracy_limit <= 0 ||
           source_accuracy_limit < recipient_state.radiation_accuracy_limit))
            recipient_state.radiation_accuracy_limit = source_accuracy_limit;
        // A passive recipient has absorbed material its interval never saw;
        // like a conserved-change wake it ends at the next event.  An active
        // recipient has just opened its interval with the merged state.
        if(recipient_state.begin_tick < current_tick_)
        {
            merge_change_wakes_ += recipient_state.change_wake_pending == 0 ? 1 : 0;
            recipient_state.change_wake_pending = 1;
        }
        return true;
    };

#ifdef RICH_MPI
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    bool merge_mapping_error = false;
    std::vector<std::vector<AMRPendingBinRequest> > outgoing_requests(
        static_cast<std::size_t>(rank_count));
    for(IndividualAMRChangeSet::MergeTarget const& target :
        changes.merge_targets)
    {
        if(target.source_rank < 0 || target.source_rank >= rank_count ||
           remapped_index.count(target.recipient_cell_id) == 0)
        {
            merge_mapping_error = true;
            continue;
        }
        if(target.source_rank == rank)
        {
            auto const source = old_states.find(target.removed_cell_id);
            if(source == old_states.end() ||
               removed.count(target.removed_cell_id) == 0 ||
               !apply_merged_state(
                   target.recipient_cell_id,
                   source->second.time_bin,
                   source->second.pending_neighbor_bin,
                   source->second.radiation_accuracy_limit))
                merge_mapping_error = true;
            continue;
        }
        AMRPendingBinRequest& request =
            outgoing_requests[static_cast<std::size_t>(
                target.source_rank)].emplace_back();
        request.source_cell_id = target.removed_cell_id;
        request.recipient_cell_id = target.recipient_cell_id;
    }

    const std::vector<std::vector<AMRPendingBinRequest> > incoming_requests =
        MPI_Exchange_all_to_all(outgoing_requests, MPI_COMM_WORLD);
    std::vector<std::vector<AMRPendingBinResponse> > outgoing_responses(
        static_cast<std::size_t>(rank_count));
    for(int requester = 0; requester < rank_count; ++requester)
        for(AMRPendingBinRequest const& request :
            incoming_requests[static_cast<std::size_t>(requester)])
        {
            auto const source = old_states.find(request.source_cell_id);
            if(source == old_states.end() ||
               removed.count(request.source_cell_id) == 0)
            {
                merge_mapping_error = true;
                continue;
            }
            AMRPendingBinResponse& response =
                outgoing_responses[static_cast<std::size_t>(
                    requester)].emplace_back();
            response.recipient_cell_id = request.recipient_cell_id;
            response.time_bin = source->second.time_bin;
            response.pending_bin =
                source->second.pending_neighbor_bin;
            response.radiation_accuracy_limit =
                source->second.radiation_accuracy_limit;
        }

    const std::vector<std::vector<AMRPendingBinResponse> >
        incoming_responses = MPI_Exchange_all_to_all(
            outgoing_responses, MPI_COMM_WORLD);
    for(auto const& rank_responses : incoming_responses)
        for(AMRPendingBinResponse const& response : rank_responses)
            if(!apply_merged_state(
                   response.recipient_cell_id, response.time_bin,
                   response.pending_bin, response.radiation_accuracy_limit))
                merge_mapping_error = true;

    int merge_mapping_valid = merge_mapping_error ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &merge_mapping_valid, 1, MPI_INT,
                  MPI_MIN, MPI_COMM_WORLD);
    if(merge_mapping_valid == 0)
        throw std::logic_error(
            "AMR pending-bin merge mapping is inconsistent");
#else
    for(IndividualAMRChangeSet::MergeTarget const& target :
        changes.merge_targets)
    {
        auto const source = old_states.find(target.removed_cell_id);
        if(target.source_rank != 0 || source == old_states.end() ||
           removed.count(target.removed_cell_id) == 0 ||
           !apply_merged_state(
               target.recipient_cell_id,
               source->second.time_bin,
               source->second.pending_neighbor_bin,
               source->second.radiation_accuracy_limit))
            throw std::logic_error(
                "AMR pending-bin merge mapping is inconsistent");
    }
#endif

    states_ = std::move(remapped);
    rebuildIndex(cells);
    if(tess != nullptr)
    {
        std::vector<std::uint8_t> const no_sources(
            states_.size(), std::numeric_limits<std::uint8_t>::max());
        if(D2TimersEnabled())
        {
            // The AMR path always runs the full closure; only its split is reported.
            ClosureSplit Split;
            double const FullStart = D2Now();
            limitNeighborBins(*tess, cells, current_tick_, no_sources, true, &Split);
            double const FullSeconds = D2Now() - FullStart;
            ReportClosure("amr", 0, -1, true, Split, 0, 0, FullSeconds);
        }
        else
            limitNeighborBins(
                *tess, cells, current_tick_, no_sources, true);
        needs_full_neighbor_closure_ = false;
    }
    else
        needs_full_neighbor_closure_ = true;
    if(CountsMode == 2)
        VerifyScheduleCounts("applyAMRChangeSet");
}

void IndividualTimeStepScheduler::enforceNeighborBinClosure(
    const Tessellation3D& tess,
    const std::vector<ComputationalCell3D>& cells)
{
    if(!initialized_)
        throw std::logic_error(
            "Cannot close neighbors for an uninitialized scheduler");
    // Both switches are collective on their first call, which is here on every rank.
    int const ScanMode = ClosureScanMode();
    bool const Timers = D2TimersEnabled();
    if(ScanMode == 0 && !Timers)
    {
        std::vector<std::uint8_t> const no_sources(
            states_.size(), std::numeric_limits<std::uint8_t>::max());
        limitNeighborBins(tess, cells, current_tick_, no_sources, true);
        needs_full_neighbor_closure_ = false;
        if(CountsMode == 2)
            VerifyScheduleCounts("enforceNeighborBinClosure");
        return;
    }

    // The scan finds every pair the full closure's first iteration would act on, with the same arithmetic, and
    // every entry it would report as an error.  A clean scan on every rank therefore means the full closure would
    // run one iteration and change nothing, so mode 1 skips it.
    int ScanFlag = -1;
    double ScanSeconds = 0;
    double ScanExchangeSeconds = 0;
    if(ScanMode != 0)
    {
        double const ScanStart = D2Now();
        ScanFlag = ScanNeighborBinClosure(tess, ScanExchangeSeconds);
        ScanSeconds = D2Now() - ScanStart;
    }
    bool const RunFull = ScanMode != 1 || ScanFlag != 0;
    bool const Report = Timers || ScanMode == 2;
    ClosureSplit Split;
    double FullSeconds = 0;
    if(RunFull)
    {
        std::vector<std::uint8_t> const no_sources(
            states_.size(), std::numeric_limits<std::uint8_t>::max());
        double const FullStart = D2Now();
        limitNeighborBins(tess, cells, current_tick_, no_sources, true, Report ? &Split : nullptr);
        FullSeconds = D2Now() - FullStart;
    }
    needs_full_neighbor_closure_ = false;

    // Collective outcomes only, so every rank keeps the same totals.  A full closure changed a bin exactly when it
    // ran a second iteration.
    ClosureScanTotals& Totals = ClosureTotals();
    if(ScanMode != 0)
    {
        ++Totals.Scans;
        Totals.Clean += ScanFlag == 0 ? 1 : 0;
        Totals.Skipped += RunFull ? 0 : 1;
    }
    if(ScanMode == 2)
    {
        bool const Changed = Split.Iterations > 1;
        Totals.ShadowCleanChanged += ScanFlag == 0 && Changed ? 1 : 0;
        Totals.ShadowDirtyUnchanged += ScanFlag != 0 && !Changed ? 1 : 0;
    }
    if(Report)
        ReportClosure("event", ScanMode, ScanFlag, RunFull, Split, ScanSeconds, ScanExchangeSeconds, FullSeconds);
    if(CountsMode == 2)
        VerifyScheduleCounts("enforceNeighborBinClosure");
}

int IndividualTimeStepScheduler::ScanNeighborBinClosure(const Tessellation3D& Tess, double& ExchangeSeconds) const
{
    unsigned const MaximumBin = options_.maximum_bin;
    unsigned const Difference = options_.maximum_neighbor_bin_difference;
    Tessellation3D::AllPointsMap const& LocalToGlobal = Tess.GetIndicesInAllPoints();
    std::vector<std::size_t> Neighbors;
    int Violation = 0;
#ifdef RICH_MPI
    int RankCount = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &RankCount);
    std::size_t const MeshSize = Tess.getMeshPoints().size();
    // Each owned bin plus one, so that zero marks a mesh entry the sync did not fill (the full closure's
    // valid == 0).  Bins never exceed 62; a bin of 255 would encode as zero and send the event to the full closure.
    std::vector<std::uint8_t> Canonical(states_.size());
    for(std::size_t i = 0; i < states_.size(); ++i)
        Canonical[i] = static_cast<std::uint8_t>(states_[i].time_bin + 1);
    double const SyncStart = D2Now();
    std::vector<std::uint8_t> const Mesh = SyncCanonicalDataToMesh(Tess, LocalToGlobal, Canonical);
    ExchangeSeconds += D2Now() - SyncStart;

    // The full closure's ownership checks (its issue bits 1, 4, 8 and 16; bit 2 throws inside the sync, bits 32
    // and 64 show up as unfilled entries below): any failure sends the event to the full closure, which reports it.
    if(LocalToGlobal.size() < Tess.GetPointNo())
        Violation = 1;
    std::vector<int> const DuplicatedProcs = Tess.GetDuplicatedProcs();
    std::vector<std::vector<std::size_t> > const& GhostIndices = Tess.GetGhostIndeces();
    if(DuplicatedProcs.size() != GhostIndices.size())
        Violation = 1;
    for(std::size_t Peer = 0; Violation == 0 && Peer < DuplicatedProcs.size(); ++Peer)
    {
        if(DuplicatedProcs[Peer] < 0 || DuplicatedProcs[Peer] >= RankCount)
            Violation = 1;
        for(std::size_t const Ghost : GhostIndices[Peer])
            if(Ghost >= MeshSize)
                Violation = 1;
    }

    for(std::size_t LocalI = 0; Violation == 0 && LocalI < Tess.GetPointNo(); ++LocalI)
    {
        auto const MappedI = LocalToGlobal.find(LocalI);
        if(MappedI == LocalToGlobal.end() || MappedI->second >= states_.size())
        {
            Violation = 1;
            break;
        }
        unsigned const BinI = states_[MappedI->second].time_bin;
        unsigned const LimitFromI = std::min<unsigned>(MaximumBin, BinI + Difference);
        Neighbors.clear();
        Tess.GetNeighbors(LocalI, Neighbors);
        for(std::size_t const LocalNeighbor : Neighbors)
        {
            if(LocalNeighbor >= MeshSize || Tess.IsPointOutsideBox(LocalNeighbor))
                continue;
            std::uint8_t const Encoded = Mesh[LocalNeighbor];
            if(Encoded == 0)
            {
                Violation = 1;
                break;
            }
            unsigned const BinNeighbor = static_cast<unsigned>(Encoded) - 1;
            // The full closure's two requests: this cell limited by the neighbour, the neighbour limited by it.
            if(BinI > std::min<unsigned>(MaximumBin, BinNeighbor + Difference) || BinNeighbor > LimitFromI)
            {
                Violation = 1;
                break;
            }
        }
    }
    double const ReduceStart = D2Now();
    MPI_Allreduce(MPI_IN_PLACE, &Violation, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    ExchangeSeconds += D2Now() - ReduceStart;
#else
    // The serial closure's rule: both endpoints mapped, each limited by the other.
    for(std::size_t LocalI = 0; Violation == 0 && LocalI < Tess.GetPointNo(); ++LocalI)
    {
        auto const MappedI = LocalToGlobal.find(LocalI);
        if(MappedI == LocalToGlobal.end() || MappedI->second >= states_.size())
            continue;
        unsigned const BinI = states_[MappedI->second].time_bin;
        Neighbors.clear();
        Tess.GetNeighbors(LocalI, Neighbors);
        for(std::size_t const LocalNeighbor : Neighbors)
        {
            auto const MappedNeighbor = LocalToGlobal.find(LocalNeighbor);
            if(MappedNeighbor == LocalToGlobal.end() || MappedNeighbor->second >= states_.size())
                continue;
            unsigned const BinNeighbor = states_[MappedNeighbor->second].time_bin;
            if(BinI > std::min<unsigned>(MaximumBin, BinNeighbor + Difference) ||
               BinNeighbor > std::min<unsigned>(MaximumBin, BinI + Difference))
            {
                Violation = 1;
                break;
            }
        }
    }
    (void)ExchangeSeconds;
#endif
    return Violation;
}

void IndividualTimeStepScheduler::ReportClosure(char const* Path, int ScanMode, int ScanFlag, bool RanFull,
    ClosureSplit const& Split, double ScanSeconds, double ScanExchangeSeconds, double FullSeconds) const
{
    // Per-rank seconds: total, full closure, its local work and exchanges, scan, its local work and exchanges.
    constexpr int TimerCount = 7;
    char const* const Names[TimerCount] = {"total", "full", "full_local", "full_exchange", "scan", "scan_local",
                                           "scan_exchange"};
    double const Timers[TimerCount] = {ScanSeconds + FullSeconds, FullSeconds,
                                       std::max(0.0, Split.TotalSeconds - Split.ExchangeSeconds),
                                       Split.ExchangeSeconds, ScanSeconds,
                                       std::max(0.0, ScanSeconds - ScanExchangeSeconds), ScanExchangeSeconds};
    int const Rank = WorldRank();
    int RankCount = 1;
    struct
    {
        double Value;
        int Rank;
    } Maxima[TimerCount];
    for(int k = 0; k < TimerCount; ++k)
    {
        Maxima[k].Value = Timers[k];
        Maxima[k].Rank = Rank;
    }
    // Timer sums, then each iteration's count of ranks with a request to send.
    std::vector<double> Sums(Timers, Timers + TimerCount);
    Sums.insert(Sums.end(), Split.Outgoing.begin(), Split.Outgoing.end());
    // A clean scan followed by a changed closure is the one disagreement that would make mode 1 unsafe.
    bool const ShadowMismatch = ScanMode == 2 && ScanFlag == 0 && Split.Iterations > 1;
    unsigned long long Example = static_cast<unsigned long long>(Split.ExampleCellId);
#ifdef RICH_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &RankCount);
    MPI_Allreduce(MPI_IN_PLACE, Maxima, TimerCount, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, Sums.data(), static_cast<int>(Sums.size()), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(ShadowMismatch)
        MPI_Allreduce(MPI_IN_PLACE, &Example, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
#endif
    if(Rank != 0)
        return;
    unsigned NoRequestIterations = 0;
    for(std::size_t k = TimerCount; k < Sums.size(); ++k)
        NoRequestIterations += Sums[k] == 0 ? 1 : 0;
    ClosureScanTotals const& Totals = ClosureTotals();
    std::ostringstream Line;
    Line << std::setprecision(6) << "RICH_D2_CLOSURE tick=" << current_tick_ << " path=" << Path
         << " scan_mode=" << ScanMode << " scan=" << (ScanFlag < 0 ? "off" : (ScanFlag == 0 ? "clean" : "dirty"))
         << " full=" << (RanFull ? "ran" : "skipped") << " iterations=" << Split.Iterations
         << " no_request_iterations=" << NoRequestIterations << " changed=" << (Split.Iterations > 1 ? 1 : 0)
         << " scans=" << Totals.Scans << " scans_clean=" << Totals.Clean << " scans_skipped=" << Totals.Skipped
         << " shadow_clean_changed=" << Totals.ShadowCleanChanged
         << " shadow_dirty_unchanged=" << Totals.ShadowDirtyUnchanged;
    for(int k = 0; k < TimerCount; ++k)
        Line << " " << Names[k] << "_s_max=" << Maxima[k].Value << " " << Names[k] << "_s_mean="
             << Sums[static_cast<std::size_t>(k)] / RankCount << " " << Names[k] << "_s_rank=" << Maxima[k].Rank;
    std::cout << Line.str() << std::endl;
    if(ShadowMismatch)
        std::cout << "RICH_D2_CLOSURE_SHADOW_MISMATCH tick=" << current_tick_ << " lowered_cell_id=" << Example
                  << std::endl;
}

CellTimeState IndividualTimeStepScheduler::synchronizedCellState(
    std::size_t const cell_id, std::uint8_t const bin) const
{
    if(!initialized_)
        throw std::logic_error(
            "Cannot create a cell state before individual timestep initialization");
    CellTimeState state;
    state.cell_id = cell_id;
    state.begin_tick = current_tick_;
    state.last_primitive_tick = current_tick_;
    state.time_bin = bin;
    state.end_tick = binnedEndTick(current_tick_, current_tick_, bin);
    state.pending_neighbor_bin = std::numeric_limits<std::uint8_t>::max();
    state.point_velocity = Vector3D();
    state.cached_acceleration = Vector3D();
    state.gravity_half_kick_pending = false;
    return state;
}

std::size_t IndividualTimeStepScheduler::limitSynchronizedBins(
    const std::vector<std::uint8_t>& maximum_bins)
{
    if(!initialized_)
        throw std::logic_error(
            "Cannot limit bins before individual timestep initialization");
    if(maximum_bins.size() != states_.size())
        throw std::invalid_argument(
            "Synchronized bin limits do not match the scheduler states");
    for(CellTimeState const& state : states_)
        if(state.begin_tick != current_tick_ ||
           state.last_primitive_tick != current_tick_)
            throw std::logic_error(
                "Synchronized bin limit requires every interval to open at the current tick");
    for(std::uint8_t const bin : maximum_bins)
        if(bin > options_.maximum_bin)
            throw std::out_of_range("Synchronized bin limit is out of range");
    std::size_t shortened = 0;
    if(options_.force_synchronized)
    {
        // One shared bin, as the synchronized commit keeps it: the tightest
        // of every current bin and every limit, on every rank.
        unsigned int shared = options_.maximum_bin;
        for(std::size_t i = 0; i < states_.size(); ++i)
            shared = std::min({shared,
                static_cast<unsigned int>(states_[i].time_bin),
                static_cast<unsigned int>(maximum_bins[i])});
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &shared, 1, MPI_UNSIGNED, MPI_MIN,
                      MPI_COMM_WORLD);
#endif
        std::uint8_t const shared_bin = static_cast<std::uint8_t>(shared);
        for(CellTimeState& state : states_)
        {
            if(state.time_bin != shared_bin)
                ++shortened;
            state.time_bin = shared_bin;
            state.pending_neighbor_bin =
                std::numeric_limits<std::uint8_t>::max();
            state.end_tick = checked_add(current_tick_, ticksForBin(shared_bin));
        }
        // Every state changed, so the tracked counts are rebuilt.
        if(CountsMode != 0)
            RebuildScheduleCounts();
    }
    else
        for(std::size_t i = 0; i < states_.size(); ++i)
        {
            std::uint8_t const bin = maximum_bins[i];
            if(states_[i].time_bin <= bin)
                continue;
            std::uint8_t const OldBin = states_[i].time_bin;
            std::uint64_t const OldEnd = states_[i].end_tick;
            // The interval opened at this tick, so the new bin's aligned end
            // lies within its allowance (binnedEndTick's first branch).
            states_[i].time_bin = bin;
            states_[i].end_tick = binnedEndTick(current_tick_, current_tick_, bin);
            if(CountsMode != 0)
                NoteScheduleChange(OldBin, OldEnd, states_[i]);
            ++shortened;
        }
    // As commitEvent: the finest bin any rank holds, and the spread cap.
    refreshMinimumOccupiedBin();
    applyBinSpreadCap(current_tick_);
    if(CountsMode == 2)
        VerifyScheduleCounts("limitSynchronizedBins");
    return shortened;
}

void IndividualTimeStepScheduler::invalidateCachedAccelerations(void)
{
    for(CellTimeState& state : states_)
    {
        state.cached_acceleration = Vector3D();
        state.gravity_half_kick_pending = false;
    }
}

std::uint64_t IndividualTimeStepScheduler::ticksForBin(std::uint8_t bin) const
{
    if(bin > options_.maximum_bin || bin > 62)
        throw std::out_of_range("Individual timestep bin is out of range");
    return std::uint64_t(1) << bin;
}

std::uint64_t IndividualTimeStepScheduler::nextAlignedTick(
    std::uint64_t event_tick, std::uint8_t bin) const
{
    const std::uint64_t interval = ticksForBin(bin);
    const std::uint64_t remainder = event_tick % interval;
    const std::uint64_t increment = remainder == 0 ? interval : interval - remainder;
    return checked_add(event_tick, increment);
}

std::uint64_t IndividualTimeStepScheduler::binnedEndTick(
    std::uint64_t begin_tick, std::uint64_t event_tick, std::uint8_t bin) const
{
    std::uint64_t end_tick = nextAlignedTick(event_tick, bin);
    std::uint64_t const allowance_end = checked_add(begin_tick, ticksForBin(bin));
    if(allowance_end > event_tick)
        return std::min(end_tick, allowance_end);
    // Already past the allowance when the request arrived: half of all
    // requests land here, because a cell asked to halve its bin is equally
    // likely to be in either half of its interval, and this is the half that
    // killed cell 715888 of job 10196266 - it carried bin 36 and ran a bin-37
    // interval to the end.  The cell cannot be pulled back before this event,
    // so send it to the next tick of the finest bin in use: those ticks
    // already carry events, so the cell joins one instead of forcing a new
    // one, and it stops sleeping as soon as the schedule allows.
    if(minimum_occupied_bin_ < bin)
        end_tick = std::min(end_tick,
                            nextAlignedTick(event_tick, minimum_occupied_bin_));
    return end_tick;
}

std::uint8_t IndividualTimeStepScheduler::quantizeTimeStep(double time_step) const
{
    if(!(std::isfinite(time_step) && time_step > 0))
    {
        if(std::isinf(time_step) && time_step > 0)
            return options_.maximum_bin;
        throw std::invalid_argument("Individual timestep limit must be positive and finite");
    }

    // Saturate before converting to ticks. Compare exponents and significands
    // so neither the maximum-bin interval nor time_step / time_quantum_ can
    // overflow when floating-point exceptions are enabled.
    int time_step_exponent = 0;
    int quantum_exponent = 0;
    const double time_step_significand =
        std::frexp(time_step, &time_step_exponent);
    const double quantum_significand =
        std::frexp(time_quantum_, &quantum_exponent);
    if(time_step_exponent < quantum_exponent ||
       (time_step_exponent == quantum_exponent &&
        time_step_significand < quantum_significand))
    {
        std::ostringstream message;
        message << std::setprecision(17)
                << "Individual timestep limit " << time_step
                << " is below the configured time quantum " << time_quantum_;
        throw std::runtime_error(message.str());
    }

    const int maximum_bin_exponent =
        quantum_exponent + static_cast<int>(options_.maximum_bin);
    if(time_step_exponent > maximum_bin_exponent ||
       (time_step_exponent == maximum_bin_exponent &&
        time_step_significand >= quantum_significand))
        return options_.maximum_bin;

    const double tick_limit = std::floor(time_step / time_quantum_);
    const std::uint64_t ticks = static_cast<std::uint64_t>(tick_limit);
    std::uint8_t bin = 0;
    while(bin < options_.maximum_bin && ticksForBin(static_cast<std::uint8_t>(bin + 1)) <= ticks)
        ++bin;
    return bin;
}

std::uint8_t IndividualTimeStepScheduler::chooseNextBin(const CellTimeState &state,
                                                        std::uint64_t event_tick,
                                                        double time_step_limit) const
{
    const std::uint8_t desired = quantizeTimeStep(time_step_limit);
    if(desired <= state.time_bin)
        return desired;
    if(state.time_bin >= options_.maximum_bin)
        return state.time_bin;

    const std::uint8_t candidate = static_cast<std::uint8_t>(state.time_bin + 1);
    const std::uint64_t candidate_ticks = ticksForBin(candidate);
    if(event_tick % candidate_ticks == 0)
        return candidate;
    return state.time_bin;
}

void IndividualTimeStepScheduler::validateCells(const std::vector<ComputationalCell3D> &cells) const
{
    if(cells.size() != states_.size())
        throw std::logic_error("Cell count changed without remapping individual timestep state");
    for(std::size_t i = 0; i < cells.size(); ++i)
        if(cells[i].ID != states_[i].cell_id)
            throw std::logic_error("Cell order changed without remapping individual timestep state");
}

void IndividualTimeStepScheduler::limitNeighborBins(
    const Tessellation3D &tess,
    const std::vector<ComputationalCell3D> &cells,
    std::uint64_t event_tick,
    const std::vector<std::uint8_t>& propagation_bins,
    bool full_closure,
    ClosureSplit* Split)
{
    // Collective on its first call, which every rank makes here.
    int const FastMode = ClosureFastMode();
#ifdef RICH_MPI
    if(full_closure && FastMode == 1)
    {
        LimitNeighborBinsFast(tess, cells, event_tick, propagation_bins, Split);
        return;
    }
    if(full_closure && FastMode == 2)
    {
        VerifyFastNeighborClosure(tess, cells, event_tick, propagation_bins, Split);
        return;
    }
#else
    (void)FastMode;
#endif
    LimitNeighborBinsReference(tess, cells, event_tick, propagation_bins, full_closure, Split);
}

void IndividualTimeStepScheduler::LimitNeighborBinsReference(
    const Tessellation3D &tess,
    const std::vector<ComputationalCell3D> &cells,
    std::uint64_t event_tick,
    const std::vector<std::uint8_t>& propagation_bins,
    bool full_closure,
    ClosureSplit* Split)
{
    // Diagnostics only (Split is null on the production path): wall time of the call, and of its collectives.
    struct SplitClock
    {
        ClosureSplit* Target;
        double Start;
        ~SplitClock()
        {
            if(Target != nullptr)
                Target->TotalSeconds += D2Now() - Start;
        }
    } const Clock{Split, Split != nullptr ? D2Now() : 0.0};
    auto const ExchangeStart = [Split]() { return Split != nullptr ? D2Now() : 0.0; };
    auto const ExchangeStop = [Split](double const Start)
    {
        if(Split != nullptr)
            Split->ExchangeSeconds += D2Now() - Start;
    };
    std::uint8_t const no_source =
        std::numeric_limits<std::uint8_t>::max();
    if(cells.size() != states_.size() ||
       propagation_bins.size() != states_.size())
        throw std::logic_error(
            "Individual timestep neighbor limiter state size mismatch");
    for(std::uint8_t const source_bin : propagation_bins)
        if(source_bin != no_source && source_bin > options_.maximum_bin)
            throw std::logic_error(
                "Individual timestep neighbor propagation bin is invalid");

#ifdef RICH_MPI
    int any_rank_has_cells = states_.empty() ? 0 : 1;
    double const CellsStart = ExchangeStart();
    MPI_Allreduce(MPI_IN_PLACE, &any_rank_has_cells, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    ExchangeStop(CellsStart);
    if(any_rank_has_cells == 0)
        return;
#else
    if(states_.empty())
        return;
#endif

    std::vector<std::size_t> neighbors;
    Tessellation3D::AllPointsMap const& local_to_global =
        tess.GetIndicesInAllPoints();
#ifdef RICH_MPI
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

    const std::size_t mesh_size = tess.getMeshPoints().size();
    std::vector<unsigned char> required_mesh_point(mesh_size, 0);
    for(std::size_t local_i = 0; local_i < tess.GetPointNo(); ++local_i)
    {
        if(local_i < mesh_size)
            required_mesh_point[local_i] = 1;
        neighbors.clear();
        tess.GetNeighbors(local_i, neighbors);
        for(std::size_t neighbor : neighbors)
            if(neighbor < mesh_size && !tess.IsPointOutsideBox(neighbor))
                required_mesh_point[neighbor] = 1;
    }

    std::vector<int> point_owner(mesh_size, -1);
    unsigned local_mapping_issues =
        local_to_global.size() < tess.GetPointNo() ? 1u : 0u;
    for(auto const& mapping : local_to_global)
    {
        if(mapping.first >= mesh_size || mapping.second >= states_.size())
        {
            local_mapping_issues |= 2u;
            continue;
        }
        point_owner[mapping.first] = rank;
    }
    const std::vector<int> duplicated_procs = tess.GetDuplicatedProcs();
    const std::vector<std::vector<std::size_t> >& ghost_indices =
        tess.GetGhostIndeces();
    if(duplicated_procs.size() != ghost_indices.size())
        local_mapping_issues |= 4u;
    for(std::size_t peer = 0;
        peer < duplicated_procs.size() && peer < ghost_indices.size(); ++peer)
    {
        if(duplicated_procs[peer] < 0 || duplicated_procs[peer] >= rank_count)
            local_mapping_issues |= 8u;
        for(std::size_t ghost : ghost_indices[peer])
        {
            if(ghost >= point_owner.size())
            {
                local_mapping_issues |= 16u;
                continue;
            }
            // SyncPartialBuildData applies duplicate lists in this order.
            point_owner[ghost] = duplicated_procs[peer];
        }
    }
    for(std::size_t local = 0; local < mesh_size; ++local)
        if(required_mesh_point[local] != 0 && point_owner[local] < 0)
            local_mapping_issues |= 32u;

    bool keep_iterating = true;
    bool mapping_checked = false;
    unsigned iteration = 0;
    while(keep_iterating)
    {
        if(full_closure &&
           iteration++ > static_cast<unsigned>(options_.maximum_bin) + 2)
            throw std::logic_error(
                "Individual timestep MPI neighbor-bin closure failed to converge");

        std::vector<NeighborBinSnapshot> canonical(states_.size());
        for(std::size_t i = 0; i < states_.size(); ++i)
        {
            canonical[i].cell_id = states_[i].cell_id;
            canonical[i].time_bin = states_[i].time_bin;
            canonical[i].source_bin =
                full_closure ? states_[i].time_bin : propagation_bins[i];
            canonical[i].valid = 1;
        }
        double const SyncStart = ExchangeStart();
        const std::vector<NeighborBinSnapshot> mesh =
            SyncCanonicalDataToMesh(tess, local_to_global, canonical);
        ExchangeStop(SyncStart);

        if(!mapping_checked)
        {
            for(std::size_t local = 0; local < mesh_size; ++local)
                if(required_mesh_point[local] != 0 && mesh[local].valid == 0)
                    local_mapping_issues |= 64u;
            unsigned mapping_issues = local_mapping_issues;
            double const MappingStart = ExchangeStart();
            MPI_Allreduce(MPI_IN_PLACE, &mapping_issues, 1, MPI_UNSIGNED,
                          MPI_BOR, MPI_COMM_WORLD);
            ExchangeStop(MappingStart);
            if(mapping_issues != 0)
                throw std::logic_error(
                    "Individual timestep MPI neighbor ownership mapping is "
                    "inconsistent (issue mask " +
                    std::to_string(mapping_issues) + ")");
            mapping_checked = true;
        }

        bool local_request_error = false;
        std::vector<std::uint8_t> requested_bins(states_.size(), no_source);
        std::vector<std::vector<NeighborBinRequest> > outgoing(rank_count);
        for(std::size_t local_i = 0; local_i < tess.GetPointNo(); ++local_i)
        {
            auto const mapped_i = local_to_global.find(local_i);
            if(mapped_i == local_to_global.end() ||
               mapped_i->second >= states_.size())
            {
                local_request_error = true;
                continue;
            }
            std::size_t const i = mapped_i->second;
            std::uint8_t const source_i =
                full_closure ? states_[i].time_bin : propagation_bins[i];

            neighbors.clear();
            tess.GetNeighbors(local_i, neighbors);
            for(std::size_t local_neighbor : neighbors)
            {
                if(local_neighbor >= mesh_size ||
                   tess.IsPointOutsideBox(local_neighbor))
                    continue;
                NeighborBinSnapshot const& neighbor = mesh[local_neighbor];
                if(neighbor.valid == 0)
                {
                    local_request_error = true;
                    continue;
                }

                if(neighbor.source_bin != no_source)
                {
                    unsigned const limited = std::min<unsigned>(
                        options_.maximum_bin,
                        static_cast<unsigned>(neighbor.source_bin) +
                            options_.maximum_neighbor_bin_difference);
                    std::uint8_t const limited_bin =
                        static_cast<std::uint8_t>(limited);
                    if(states_[i].time_bin > limited_bin)
                        requested_bins[i] =
                            std::min(requested_bins[i], limited_bin);
                }

                if(source_i == no_source)
                    continue;
                unsigned const limited = std::min<unsigned>(
                    options_.maximum_bin,
                    static_cast<unsigned>(source_i) +
                        options_.maximum_neighbor_bin_difference);
                std::uint8_t const limited_bin =
                    static_cast<std::uint8_t>(limited);
                if(neighbor.time_bin <= limited_bin)
                    continue;

                auto const mapped_neighbor =
                    local_to_global.find(local_neighbor);
                if(mapped_neighbor != local_to_global.end())
                {
                    if(mapped_neighbor->second >= states_.size())
                    {
                        local_request_error = true;
                        continue;
                    }
                    requested_bins[mapped_neighbor->second] = std::min(
                        requested_bins[mapped_neighbor->second], limited_bin);
                    continue;
                }

                int const owner = point_owner[local_neighbor];
                if(owner < 0 || owner >= rank_count)
                {
                    local_request_error = true;
                    continue;
                }
                NeighborBinRequest& request =
                    outgoing[owner].emplace_back();
                request.cell_id = neighbor.cell_id;
                request.maximum_bin = limited_bin;
            }
        }

        auto apply_request = [&](CellTimeState& state,
                                 std::uint8_t requested_bin)
        {
            if(requested_bin == no_source ||
               state.time_bin <= requested_bin)
                return false;
            noteLowering(1, state.begin_tick, event_tick, requested_bin);
            unsigned const old_bin = state.time_bin;
            std::uint64_t const old_end = state.end_tick;
            state.time_bin = requested_bin;
            state.end_tick = std::min(
                state.end_tick,
                binnedEndTick(state.begin_tick, event_tick, requested_bin));
            TraceSchedulerMutation("closure", state.cell_id, event_tick, state.begin_tick, old_bin,
                requested_bin, old_end, state.end_tick);
            if(CountsMode != 0)
                NoteScheduleChange(static_cast<std::uint8_t>(old_bin), old_end, state);
            if(Split != nullptr && Split->ExampleCellId == std::numeric_limits<std::size_t>::max())
                Split->ExampleCellId = state.cell_id;
            if(!full_closure)
                state.pending_neighbor_bin = std::min(
                    state.pending_neighbor_bin, requested_bin);
            return true;
        };

        if(Split != nullptr)
        {
            ++Split->Iterations;
            bool const HasOutgoing = std::any_of(outgoing.begin(), outgoing.end(),
                [](std::vector<NeighborBinRequest> const& Requests) { return !Requests.empty(); });
            Split->Outgoing.push_back(HasOutgoing ? 1.0 : 0.0);
            if(Split->RecordRequests)
                RecordClosureRequests(Split->Requests, outgoing);
        }
        // Remote requests join the local ones before anything is applied:
        // only the owner knows the cell's begin tick, so only it can honour
        // the allowance and the finest-tick overrun rule in binnedEndTick, and
        // binnedEndTick is not monotonic in the bin (an overdue finer bin can
        // end later than a coarser one within its allowance), so each cell
        // applies its minimum requested bin once, whatever the arrival order.
        double const AllToAllStart = ExchangeStart();
        const std::vector<std::vector<NeighborBinRequest> > incoming =
            MPI_Exchange_all_to_all(outgoing, MPI_COMM_WORLD);
        ExchangeStop(AllToAllStart);
        for(auto const& rank_requests : incoming)
            for(NeighborBinRequest const& request : rank_requests)
            {
                auto const found = id_to_index_.find(request.cell_id);
                if(found == id_to_index_.end())
                {
                    local_request_error = true;
                    continue;
                }
                requested_bins[found->second] =
                    std::min(requested_bins[found->second], request.maximum_bin);
            }

        bool locally_changed = false;
        for(std::size_t i = 0; i < states_.size(); ++i)
            locally_changed =
                apply_request(states_[i], requested_bins[i]) ||
                locally_changed;

        int request_error = local_request_error ? 1 : 0;
        double const ErrorStart = ExchangeStart();
        MPI_Allreduce(MPI_IN_PLACE, &request_error, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        ExchangeStop(ErrorStart);
        if(request_error != 0)
            throw std::logic_error(
                "Individual timestep MPI neighbor-bin request mapping failed");

        if(full_closure)
        {
            int changed = locally_changed ? 1 : 0;
            double const ChangedStart = ExchangeStart();
            MPI_Allreduce(MPI_IN_PLACE, &changed, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD);
            ExchangeStop(ChangedStart);
            keep_iterating = changed != 0;
        }
        else
            keep_iterating = false;
    }
#else
    (void)ExchangeStart;
    (void)ExchangeStop;
    bool keep_iterating = true;
    unsigned iteration = 0;
    while(keep_iterating)
    {
        if(full_closure &&
           iteration++ > static_cast<unsigned>(options_.maximum_bin) + 2)
            throw std::logic_error(
                "Individual timestep neighbor-bin closure failed to converge");

        std::vector<std::uint8_t> snapshot_bins(states_.size());
        std::vector<std::uint8_t> snapshot_sources(states_.size(), no_source);
        for(std::size_t i = 0; i < states_.size(); ++i)
        {
            snapshot_bins[i] = states_[i].time_bin;
            snapshot_sources[i] =
                full_closure ? states_[i].time_bin : propagation_bins[i];
        }

        std::vector<std::uint8_t> requested_bins(states_.size(), no_source);
        for(std::size_t local_i = 0; local_i < tess.GetPointNo(); ++local_i)
        {
            auto const mapped_i = local_to_global.find(local_i);
            if(mapped_i == local_to_global.end() ||
               mapped_i->second >= states_.size())
                continue;
            std::size_t const i = mapped_i->second;

            neighbors.clear();
            tess.GetNeighbors(local_i, neighbors);
            for(std::size_t local_neighbor : neighbors)
            {
                auto const mapped_neighbor =
                    local_to_global.find(local_neighbor);
                if(mapped_neighbor == local_to_global.end() ||
                   mapped_neighbor->second >= states_.size())
                    continue;
                std::size_t const neighbor = mapped_neighbor->second;

                if(snapshot_sources[neighbor] != no_source)
                {
                    unsigned const limited = std::min<unsigned>(
                        options_.maximum_bin,
                        static_cast<unsigned>(snapshot_sources[neighbor]) +
                            options_.maximum_neighbor_bin_difference);
                    std::uint8_t const limited_bin =
                        static_cast<std::uint8_t>(limited);
                    if(snapshot_bins[i] > limited_bin)
                        requested_bins[i] =
                            std::min(requested_bins[i], limited_bin);
                }
                if(snapshot_sources[i] != no_source)
                {
                    unsigned const limited = std::min<unsigned>(
                        options_.maximum_bin,
                        static_cast<unsigned>(snapshot_sources[i]) +
                            options_.maximum_neighbor_bin_difference);
                    std::uint8_t const limited_bin =
                        static_cast<std::uint8_t>(limited);
                    if(snapshot_bins[neighbor] > limited_bin)
                        requested_bins[neighbor] = std::min(
                            requested_bins[neighbor], limited_bin);
                }
            }
        }

        bool changed = false;
        for(std::size_t i = 0; i < states_.size(); ++i)
        {
            std::uint8_t const requested_bin = requested_bins[i];
            if(requested_bin == no_source ||
               states_[i].time_bin <= requested_bin)
                continue;
            std::uint8_t const OldBin = states_[i].time_bin;
            std::uint64_t const OldEnd = states_[i].end_tick;
            states_[i].time_bin = requested_bin;
            states_[i].end_tick = std::min(
                states_[i].end_tick,
                binnedEndTick(states_[i].begin_tick, event_tick,
                              requested_bin));
            if(CountsMode != 0)
                NoteScheduleChange(OldBin, OldEnd, states_[i]);
            if(Split != nullptr && Split->ExampleCellId == std::numeric_limits<std::size_t>::max())
                Split->ExampleCellId = states_[i].cell_id;
            if(!full_closure)
                states_[i].pending_neighbor_bin = std::min(
                    states_[i].pending_neighbor_bin, requested_bin);
            changed = true;
        }
        if(Split != nullptr)
        {
            ++Split->Iterations;
            Split->Outgoing.push_back(0.0);
        }
        keep_iterating = full_closure && changed;
    }
#endif
}

#ifdef RICH_MPI
void IndividualTimeStepScheduler::LimitNeighborBinsFast(const Tessellation3D& Tess,
    const std::vector<ComputationalCell3D>& Cells, std::uint64_t const EventTick,
    const std::vector<std::uint8_t>& PropagationBins, ClosureSplit* Split)
{
    // Why it equals the reference.  The mesh is fixed during the call, so the neighbour lists, owners and mapping
    // checks are built once, with the reference's filters and in its order.  In an iteration a target t (canonical
    // cell i) can make a request only from a pair (i, n) where i's bin exceeds the limit set by n's mesh bin or n's
    // exceeds the one set by i's.  Such a request always lowers its cell: the requester's snapshot equals the
    // owner's state until the apply, and the apply lowers whenever the request is below the current bin.  So a
    // pair whose two bins were both unchanged since the previous iteration made no request then and makes none
    // now.  From the second iteration on, only targets whose own bin, own mesh entry or some neighbour's mesh entry
    // changed are rescanned, in ascending order, which leaves every request, its order and the apply order as in
    // the reference.
    struct SplitClock
    {
        ClosureSplit* Target;
        double Start;
        ~SplitClock()
        {
            if(Target != nullptr)
                Target->TotalSeconds += D2Now() - Start;
        }
    } const Clock{Split, Split != nullptr ? D2Now() : 0.0};
    auto const ExchangeStart = [Split]() { return Split != nullptr ? D2Now() : 0.0; };
    auto const ExchangeStop = [Split](double const Start)
    {
        if(Split != nullptr)
            Split->ExchangeSeconds += D2Now() - Start;
    };
    std::uint8_t const NoSource = std::numeric_limits<std::uint8_t>::max();
    std::size_t const NoIndex = std::numeric_limits<std::size_t>::max();
    if(Cells.size() != states_.size() || PropagationBins.size() != states_.size())
        throw std::logic_error("Individual timestep neighbor limiter state size mismatch");
    for(std::uint8_t const SourceBin : PropagationBins)
        if(SourceBin != NoSource && SourceBin > options_.maximum_bin)
            throw std::logic_error("Individual timestep neighbor propagation bin is invalid");
    int AnyRankHasCells = states_.empty() ? 0 : 1;
    double const CellsStart = ExchangeStart();
    MPI_Allreduce(MPI_IN_PLACE, &AnyRankHasCells, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    ExchangeStop(CellsStart);
    if(AnyRankHasCells == 0)
        return;

    int Rank = 0;
    int RankCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &RankCount);
    Tessellation3D::AllPointsMap const& LocalToGlobal = Tess.GetIndicesInAllPoints();
    std::size_t const MeshSize = Tess.getMeshPoints().size();
    std::size_t const TargetCount = Tess.GetPointNo();
    std::size_t const StateCount = states_.size();
    unsigned const MaximumBin = options_.maximum_bin;
    unsigned const Difference = options_.maximum_neighbor_bin_difference;

    // Every target's neighbours, filtered as the reference filters them (inside the mesh, inside the box), in
    // GetNeighbors order; the reference's required points come from the same lists.
    std::vector<std::size_t> NeighborStart(TargetCount + 1, 0);
    std::vector<std::size_t> NeighborList;
    NeighborList.reserve(16 * TargetCount);
    std::vector<unsigned char> Required(MeshSize, 0);
    {
        std::vector<std::size_t> Neighbors;
        for(std::size_t LocalI = 0; LocalI < TargetCount; ++LocalI)
        {
            if(LocalI < MeshSize)
                Required[LocalI] = 1;
            Neighbors.clear();
            Tess.GetNeighbors(LocalI, Neighbors);
            for(std::size_t const Neighbor : Neighbors)
                if(Neighbor < MeshSize && !Tess.IsPointOutsideBox(Neighbor))
                {
                    Required[Neighbor] = 1;
                    NeighborList.push_back(Neighbor);
                }
            NeighborStart[LocalI + 1] = NeighborList.size();
        }
    }

    // Owners and the reference's ownership checks; MeshCanonical replaces its local_to_global.find on a neighbour.
    std::vector<int> PointOwner(MeshSize, -1);
    std::vector<std::size_t> MeshCanonical(MeshSize, NoIndex);
    unsigned LocalMappingIssues = LocalToGlobal.size() < TargetCount ? 1u : 0u;
    for(auto const& Mapping : LocalToGlobal)
    {
        if(Mapping.first < MeshSize)
            MeshCanonical[Mapping.first] = Mapping.second;
        if(Mapping.first >= MeshSize || Mapping.second >= StateCount)
        {
            LocalMappingIssues |= 2u;
            continue;
        }
        PointOwner[Mapping.first] = Rank;
    }
    std::vector<int> const DuplicatedProcs = Tess.GetDuplicatedProcs();
    std::vector<std::vector<std::size_t> > const& GhostIndices = Tess.GetGhostIndeces();
    if(DuplicatedProcs.size() != GhostIndices.size())
        LocalMappingIssues |= 4u;
    for(std::size_t Peer = 0; Peer < DuplicatedProcs.size() && Peer < GhostIndices.size(); ++Peer)
    {
        if(DuplicatedProcs[Peer] < 0 || DuplicatedProcs[Peer] >= RankCount)
            LocalMappingIssues |= 8u;
        for(std::size_t const Ghost : GhostIndices[Peer])
        {
            if(Ghost >= PointOwner.size())
            {
                LocalMappingIssues |= 16u;
                continue;
            }
            // SyncPartialBuildData applies duplicate lists in this order.
            PointOwner[Ghost] = DuplicatedProcs[Peer];
        }
    }
    for(std::size_t Point = 0; Point < MeshSize; ++Point)
        if(Required[Point] != 0 && PointOwner[Point] < 0)
            LocalMappingIssues |= 32u;

    // Canonical cell of every target (the reference looks it up on every iteration).  A target without one is an
    // error on every iteration, as in the reference, and is never scanned.
    std::vector<std::size_t> TargetCanonical(TargetCount, NoIndex);
    bool InvalidTarget = false;
    for(std::size_t LocalI = 0; LocalI < TargetCount; ++LocalI)
    {
        auto const Mapped = LocalToGlobal.find(LocalI);
        if(Mapped == LocalToGlobal.end() || Mapped->second >= StateCount)
            InvalidTarget = true;
        else
            TargetCanonical[LocalI] = Mapped->second;
    }

    // For every mesh point, the scanned targets that list it as a neighbour.
    std::vector<std::size_t> ReverseStart(MeshSize + 1, 0);
    for(std::size_t Target = 0; Target < TargetCount; ++Target)
        if(TargetCanonical[Target] != NoIndex)
            for(std::size_t k = NeighborStart[Target]; k < NeighborStart[Target + 1]; ++k)
                ++ReverseStart[NeighborList[k] + 1];
    for(std::size_t Point = 0; Point < MeshSize; ++Point)
        ReverseStart[Point + 1] += ReverseStart[Point];
    std::vector<std::size_t> ReverseList(ReverseStart[MeshSize]);
    {
        std::vector<std::size_t> Cursor(ReverseStart.begin(), ReverseStart.end() - 1);
        for(std::size_t Target = 0; Target < TargetCount; ++Target)
            if(TargetCanonical[Target] != NoIndex)
                for(std::size_t k = NeighborStart[Target]; k < NeighborStart[Target + 1]; ++k)
                    ReverseList[Cursor[NeighborList[k]]++] = Target;
    }

    auto const ApplyRequest = [&](CellTimeState& State, std::uint8_t const RequestedBin)
    {
        if(RequestedBin == NoSource || State.time_bin <= RequestedBin)
            return false;
        noteLowering(1, State.begin_tick, EventTick, RequestedBin);
        unsigned const OldBin = State.time_bin;
        std::uint64_t const OldEnd = State.end_tick;
        State.time_bin = RequestedBin;
        State.end_tick = std::min(State.end_tick, binnedEndTick(State.begin_tick, EventTick, RequestedBin));
        if(Split == nullptr || !Split->SuppressTrace)
            TraceSchedulerMutation("closure", State.cell_id, EventTick, State.begin_tick, OldBin, RequestedBin,
                OldEnd, State.end_tick);
        if(CountsMode != 0)
            NoteScheduleChange(static_cast<std::uint8_t>(OldBin), OldEnd, State);
        if(Split != nullptr && Split->ExampleCellId == NoIndex)
            Split->ExampleCellId = State.cell_id;
        return true;
    };

    std::vector<std::uint8_t> CanonicalBins(StateCount);
    std::vector<std::uint8_t> MeshBins;
    std::vector<std::uint8_t> PreviousMeshBins;
    std::vector<std::size_t> MeshIds;
    std::vector<std::vector<std::size_t> > DuplicatedCanonical;
    std::vector<std::uint8_t> TargetBins(TargetCount, 0);
    std::vector<unsigned char> RescanMark(TargetCount, 0);
    std::vector<std::size_t> Rescan;
    std::vector<std::uint8_t> RequestedBins(StateCount, NoSource);
    std::vector<std::size_t> Requested;
    auto const Request = [&](std::size_t const Index, std::uint8_t const Bin)
    {
        if(RequestedBins[Index] == NoSource)
            Requested.push_back(Index);
        RequestedBins[Index] = std::min(RequestedBins[Index], Bin);
    };
    bool KeepIterating = true;
    unsigned Iteration = 0;
    while(KeepIterating)
    {
        if(Iteration++ > MaximumBin + 2)
            throw std::logic_error("Individual timestep MPI neighbor-bin closure failed to converge");
        // Each bin plus one: bins never exceed 62, so zero still marks an unfilled mesh entry.
        for(std::size_t i = 0; i < StateCount; ++i)
            CanonicalBins[i] = static_cast<std::uint8_t>(states_[i].time_bin + 1);
        Rescan.clear();
        if(Iteration == 1)
        {
            // The reference's sync, with its checks, carrying the stable IDs that remote requests name.
            std::vector<FastClosureEntry> Canonical(StateCount);
            for(std::size_t i = 0; i < StateCount; ++i)
            {
                Canonical[i].CellId = states_[i].cell_id;
                Canonical[i].Bin = CanonicalBins[i];
            }
            double const SyncStart = ExchangeStart();
            std::vector<FastClosureEntry> const Mesh = SyncCanonicalDataToMesh(Tess, LocalToGlobal, Canonical);
            ExchangeStop(SyncStart);
            MeshBins.resize(Mesh.size());
            MeshIds.resize(Mesh.size());
            for(std::size_t Point = 0; Point < Mesh.size(); ++Point)
            {
                MeshBins[Point] = Mesh[Point].Bin;
                MeshIds[Point] = Mesh[Point].CellId;
            }
            for(std::size_t Point = 0; Point < MeshSize; ++Point)
                if(Required[Point] != 0 && (Point >= MeshBins.size() || MeshBins[Point] == 0))
                    LocalMappingIssues |= 64u;
            unsigned MappingIssues = LocalMappingIssues;
            double const MappingStart = ExchangeStart();
            MPI_Allreduce(MPI_IN_PLACE, &MappingIssues, 1, MPI_UNSIGNED, MPI_BOR, MPI_COMM_WORLD);
            ExchangeStop(MappingStart);
            if(MappingIssues != 0)
                throw std::logic_error(
                    "Individual timestep MPI neighbor ownership mapping is inconsistent (issue mask " +
                    std::to_string(MappingIssues) + ")");
            // The canonical entries the sync sent to each peer, for the bin exchanges of later iterations.  The
            // sync has checked every index on every rank.
            std::vector<std::vector<std::size_t> > const& DuplicatedAllPoints = Tess.GetDuplicatedAllPointIndices();
            DuplicatedCanonical.resize(DuplicatedAllPoints.size());
            for(std::size_t Peer = 0; Peer < DuplicatedAllPoints.size(); ++Peer)
            {
                DuplicatedCanonical[Peer].reserve(DuplicatedAllPoints[Peer].size());
                for(std::size_t const AllPoint : DuplicatedAllPoints[Peer])
                    DuplicatedCanonical[Peer].push_back(Tess.GetInputIndexForAllPoint(AllPoint));
            }
            for(std::size_t Target = 0; Target < TargetCount; ++Target)
                if(TargetCanonical[Target] != NoIndex)
                {
                    TargetBins[Target] = states_[TargetCanonical[Target]].time_bin;
                    Rescan.push_back(Target);
                }
        }
        else
        {
            // What the sync would produce: owned entries from the canonical bins, then ghosts from their owners.
            PreviousMeshBins = MeshBins;
            for(auto const& Mapping : LocalToGlobal)
                MeshBins[Mapping.first] = CanonicalBins[Mapping.second];
            double const SyncStart = ExchangeStart();
            std::vector<std::vector<std::uint8_t> > const Incoming =
                MPI_exchange_data_indexed(DuplicatedProcs, CanonicalBins, DuplicatedCanonical);
            ExchangeStop(SyncStart);
            for(std::size_t Peer = 0; Peer < Incoming.size(); ++Peer)
                for(std::size_t j = 0; j < Incoming[Peer].size(); ++j)
                    MeshBins[GhostIndices.at(Peer).at(j)] = Incoming[Peer][j];
            std::fill(RescanMark.begin(), RescanMark.end(), static_cast<unsigned char>(0));
            for(std::size_t Point = 0; Point < MeshSize; ++Point)
            {
                if(MeshBins[Point] == PreviousMeshBins[Point])
                    continue;
                if(Point < TargetCount)
                    RescanMark[Point] = 1;
                for(std::size_t k = ReverseStart[Point]; k < ReverseStart[Point + 1]; ++k)
                    RescanMark[ReverseList[k]] = 1;
            }
            for(std::size_t Target = 0; Target < TargetCount; ++Target)
            {
                if(TargetCanonical[Target] == NoIndex)
                    continue;
                std::uint8_t const Bin = states_[TargetCanonical[Target]].time_bin;
                if(Bin != TargetBins[Target])
                {
                    TargetBins[Target] = Bin;
                    RescanMark[Target] = 1;
                }
                if(RescanMark[Target] != 0)
                    Rescan.push_back(Target);
            }
        }

        bool LocalRequestError = InvalidTarget;
        std::vector<std::vector<NeighborBinRequest> > Outgoing(static_cast<std::size_t>(RankCount));
        for(std::size_t const Target : Rescan)
        {
            std::size_t const i = TargetCanonical[Target];
            unsigned const BinI = states_[i].time_bin;
            std::uint8_t const LimitFromI =
                static_cast<std::uint8_t>(std::min<unsigned>(MaximumBin, BinI + Difference));
            for(std::size_t k = NeighborStart[Target]; k < NeighborStart[Target + 1]; ++k)
            {
                std::size_t const Neighbor = NeighborList[k];
                std::uint8_t const Encoded = MeshBins[Neighbor];
                if(Encoded == 0)
                {
                    LocalRequestError = true;
                    continue;
                }
                unsigned const BinNeighbor = static_cast<unsigned>(Encoded) - 1;
                std::uint8_t const LimitFromNeighbor =
                    static_cast<std::uint8_t>(std::min<unsigned>(MaximumBin, BinNeighbor + Difference));
                if(BinI > LimitFromNeighbor)
                    Request(i, LimitFromNeighbor);
                if(BinNeighbor <= LimitFromI)
                    continue;
                std::size_t const Mapped = MeshCanonical[Neighbor];
                if(Mapped != NoIndex)
                {
                    if(Mapped >= StateCount)
                    {
                        LocalRequestError = true;
                        continue;
                    }
                    Request(Mapped, LimitFromI);
                    continue;
                }
                int const Owner = PointOwner[Neighbor];
                if(Owner < 0 || Owner >= RankCount)
                {
                    LocalRequestError = true;
                    continue;
                }
                NeighborBinRequest& Sent = Outgoing[static_cast<std::size_t>(Owner)].emplace_back();
                Sent.cell_id = MeshIds[Neighbor];
                Sent.maximum_bin = LimitFromI;
            }
        }

        if(Split != nullptr)
        {
            ++Split->Iterations;
            bool const HasOutgoing = std::any_of(Outgoing.begin(), Outgoing.end(),
                [](std::vector<NeighborBinRequest> const& Requests) { return !Requests.empty(); });
            Split->Outgoing.push_back(HasOutgoing ? 1.0 : 0.0);
            if(Split->RecordRequests)
                RecordClosureRequests(Split->Requests, Outgoing);
        }
        double const AllToAllStart = ExchangeStart();
        std::vector<std::vector<NeighborBinRequest> > const Incoming =
            MPI_Exchange_all_to_all(Outgoing, MPI_COMM_WORLD);
        ExchangeStop(AllToAllStart);
        for(auto const& RankRequests : Incoming)
            for(NeighborBinRequest const& Received : RankRequests)
            {
                auto const Found = id_to_index_.find(Received.cell_id);
                if(Found == id_to_index_.end())
                {
                    LocalRequestError = true;
                    continue;
                }
                Request(Found->second, Received.maximum_bin);
            }

        // The reference applies in ascending canonical order; only requested cells can change.
        std::sort(Requested.begin(), Requested.end());
        bool LocallyChanged = false;
        for(std::size_t const Index : Requested)
        {
            LocallyChanged = ApplyRequest(states_[Index], RequestedBins[Index]) || LocallyChanged;
            RequestedBins[Index] = NoSource;
        }
        Requested.clear();

        // The reference's error and changed reductions, in one.
        int Flags[2] = {LocalRequestError ? 1 : 0, LocallyChanged ? 1 : 0};
        double const FlagsStart = ExchangeStart();
        MPI_Allreduce(MPI_IN_PLACE, Flags, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        ExchangeStop(FlagsStart);
        if(Flags[0] != 0)
            throw std::logic_error("Individual timestep MPI neighbor-bin request mapping failed");
        KeepIterating = Flags[1] != 0;
    }
}

void IndividualTimeStepScheduler::VerifyFastNeighborClosure(const Tessellation3D& Tess,
    const std::vector<ComputationalCell3D>& Cells, std::uint64_t const EventTick,
    const std::vector<std::uint8_t>& PropagationBins, ClosureSplit* Split)
{
    // Everything a full closure can change: states, lowering diagnostics, tracked counts.
    std::vector<CellTimeState> const SavedStates = states_;
    std::uint64_t SavedLowered[4];
    std::uint64_t SavedOverdue[4];
    double SavedRatio[4];
    std::copy(lowered_, lowered_ + 4, SavedLowered);
    std::copy(lowered_overdue_, lowered_overdue_ + 4, SavedOverdue);
    std::copy(lowered_worst_ratio_, lowered_worst_ratio_ + 4, SavedRatio);
    std::array<std::uint64_t, 256> const SavedBins = BinOccupancy;
    std::map<std::uint64_t, std::uint64_t> const SavedEnds = EndTickOccupancy;

    ClosureSplit FastSplit;
    FastSplit.RecordRequests = true;
    FastSplit.SuppressTrace = true;
    double const FastStart = D2Now();
    LimitNeighborBinsFast(Tess, Cells, EventTick, PropagationBins, &FastSplit);
    double const FastSeconds = D2Now() - FastStart;
    std::vector<CellTimeState> FastStates;
    FastStates.swap(states_);
    states_ = SavedStates;
    std::uint64_t FastLowered[4];
    std::uint64_t FastOverdue[4];
    double FastRatio[4];
    std::copy(lowered_, lowered_ + 4, FastLowered);
    std::copy(lowered_overdue_, lowered_overdue_ + 4, FastOverdue);
    std::copy(lowered_worst_ratio_, lowered_worst_ratio_ + 4, FastRatio);
    std::copy(SavedLowered, SavedLowered + 4, lowered_);
    std::copy(SavedOverdue, SavedOverdue + 4, lowered_overdue_);
    std::copy(SavedRatio, SavedRatio + 4, lowered_worst_ratio_);
    std::array<std::uint64_t, 256> const FastBins = BinOccupancy;
    std::map<std::uint64_t, std::uint64_t> const FastEnds = EndTickOccupancy;
    BinOccupancy = SavedBins;
    EndTickOccupancy = SavedEnds;

    // The reference result is the one kept.
    ClosureSplit LocalSplit;
    ClosureSplit& Reference = Split != nullptr ? *Split : LocalSplit;
    Reference.RecordRequests = true;
    double const ReferenceStart = D2Now();
    LimitNeighborBinsReference(Tess, Cells, EventTick, PropagationBins, true, &Reference);
    double const ReferenceSeconds = D2Now() - ReferenceStart;

    std::ostringstream Mismatch;
    std::uint64_t ComparedRequests = 0;
    if(FastSplit.Iterations != Reference.Iterations || FastSplit.Requests.size() != Reference.Requests.size())
        Mismatch << "kind=iterations fast=" << FastSplit.Iterations << " reference=" << Reference.Iterations;
    for(std::size_t k = 0; Mismatch.tellp() == 0 && k < Reference.Requests.size(); ++k)
    {
        auto const& FastIteration = FastSplit.Requests[k];
        auto const& ReferenceIteration = Reference.Requests[k];
        for(std::size_t Destination = 0; Mismatch.tellp() == 0 && Destination < ReferenceIteration.size();
            ++Destination)
        {
            auto const& FastList = FastIteration.at(Destination);
            auto const& ReferenceList = ReferenceIteration[Destination];
            ComparedRequests += ReferenceList.size();
            for(std::size_t r = 0; r < std::max(FastList.size(), ReferenceList.size()); ++r)
                if(r >= FastList.size() || r >= ReferenceList.size() || FastList[r] != ReferenceList[r])
                {
                    Mismatch << "kind=request iteration=" << k + 1 << " destination=" << Destination
                             << " position=" << r << " fast_count=" << FastList.size()
                             << " reference_count=" << ReferenceList.size();
                    if(r < FastList.size())
                        Mismatch << " fast_cell_id=" << FastList[r].first
                                 << " fast_bin=" << unsigned(FastList[r].second);
                    if(r < ReferenceList.size())
                        Mismatch << " reference_cell_id=" << ReferenceList[r].first
                                 << " reference_bin=" << unsigned(ReferenceList[r].second);
                    break;
                }
        }
    }
    if(Mismatch.tellp() == 0 && FastStates.size() != states_.size())
        Mismatch << "kind=state_count fast=" << FastStates.size() << " reference=" << states_.size();
    for(std::size_t i = 0; Mismatch.tellp() == 0 && i < states_.size(); ++i)
    {
        CellTimeState const& Fast = FastStates[i];
        CellTimeState const& Kept = states_[i];
        if(Fast.cell_id != Kept.cell_id || Fast.time_bin != Kept.time_bin || Fast.end_tick != Kept.end_tick ||
           Fast.begin_tick != Kept.begin_tick || Fast.pending_neighbor_bin != Kept.pending_neighbor_bin)
            Mismatch << "kind=state cell_id=" << Kept.cell_id << " fast_bin=" << unsigned(Fast.time_bin)
                     << " reference_bin=" << unsigned(Kept.time_bin) << " fast_end=" << Fast.end_tick
                     << " reference_end=" << Kept.end_tick << " fast_pending=" << unsigned(Fast.pending_neighbor_bin)
                     << " reference_pending=" << unsigned(Kept.pending_neighbor_bin);
    }
    for(int Cause = 0; Mismatch.tellp() == 0 && Cause < 4; ++Cause)
        if(FastLowered[Cause] != lowered_[Cause] || FastOverdue[Cause] != lowered_overdue_[Cause] ||
           std::memcmp(&FastRatio[Cause], &lowered_worst_ratio_[Cause], sizeof(double)) != 0)
            Mismatch << "kind=lowering cause=" << Cause << " fast_lowered=" << FastLowered[Cause]
                     << " reference_lowered=" << lowered_[Cause];
    if(Mismatch.tellp() == 0 && CountsMode != 0 && (FastBins != BinOccupancy || FastEnds != EndTickOccupancy))
        Mismatch << "kind=schedule_counts";
    if(Mismatch.tellp() != 0)
    {
        std::cerr << "RICH_CLOSURE_FAST_MISMATCH rank=" << WorldRank() << " event_tick=" << EventTick << " "
                  << Mismatch.str() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 98);
    }

    // Rank 0: one line per verified closure, with both timings (rank maximum, mean, rank of the maximum).
    constexpr int TimerCount = 4;
    char const* const Names[TimerCount] = {"reference", "fast", "fast_local", "fast_exchange"};
    double const Timers[TimerCount] = {ReferenceSeconds, FastSeconds,
                                       std::max(0.0, FastSplit.TotalSeconds - FastSplit.ExchangeSeconds),
                                       FastSplit.ExchangeSeconds};
    int const Rank = WorldRank();
    int RankCount = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &RankCount);
    struct
    {
        double Value;
        int Rank;
    } Maxima[TimerCount];
    for(int k = 0; k < TimerCount; ++k)
    {
        Maxima[k].Value = Timers[k];
        Maxima[k].Rank = Rank;
    }
    double Sums[TimerCount + 2] = {Timers[0], Timers[1], Timers[2], Timers[3], static_cast<double>(ComparedRequests),
                                   static_cast<double>(states_.size())};
    MPI_Allreduce(MPI_IN_PLACE, Maxima, TimerCount, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, Sums, TimerCount + 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if(Rank != 0)
        return;
    std::ostringstream Line;
    Line << std::setprecision(6) << "RICH_D2_CLOSURE_FAST event_tick=" << EventTick << " iterations="
         << Reference.Iterations << " requests_compared=" << static_cast<std::uint64_t>(Sums[TimerCount])
         << " states_compared=" << static_cast<std::uint64_t>(Sums[TimerCount + 1]) << " mismatches=0";
    for(int k = 0; k < TimerCount; ++k)
        Line << " " << Names[k] << "_s_max=" << Maxima[k].Value << " " << Names[k] << "_s_mean=" << Sums[k] / RankCount
             << " " << Names[k] << "_s_rank=" << Maxima[k].Rank;
    std::cout << Line.str() << std::endl;
}
#endif
