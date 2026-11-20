#include "RadiationStep.hpp"
#include "Radiation/Diffusion.hpp"
#include "misc/memory_debug.hpp"
#include "misc/mpi_wait_profiler.hpp"
#include "misc/universal_error.hpp"
#include "newtonian/three_dimensional/simulation/ActiveMeshView.hpp"
#include "newtonian/three_dimensional/simulation/RuntimeLog.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace {

bool individualPerformanceTraceEnabled()
{
	if(RuntimeLogDetailed())
		return true;
    static bool const enabled = []()
    {
        char const* const value = std::getenv("RICH_INDIVIDUAL_PERF_TRACE");
        return value != nullptr && value[0] != '\0' &&
            std::strcmp(value, "0") != 0 &&
            std::strcmp(value, "false") != 0 &&
            std::strcmp(value, "off") != 0 &&
            std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

double elapsedSeconds(std::chrono::steady_clock::time_point const start)
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
}

#ifdef RICH_MPI
void broadcast_step_failure(RadiationDriver const& matrix_builder,
                            std::string& reason,
                            size_t& cell_id,
                            std::string& diagnostics)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int const has_local = matrix_builder.getLastStepFailureReason().empty() ? 0 : 1;
    int const has_cell_id =
        (has_local && matrix_builder.getLastStepFailureCellId() !=
            std::numeric_limits<size_t>::max()) ? 1 : 0;
    int const score = has_cell_id ? 2 : has_local;
    struct {
        int score;
        int rank;
    } local_pick{score, rank};
    struct {
        int score;
        int rank;
    } global_pick{0, std::numeric_limits<int>::max()};
    MPI_Allreduce(&local_pick, &global_pick, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
    int const source_rank = global_pick.score > 0
        ? global_pick.rank
        : std::numeric_limits<int>::max();
    if (source_rank == std::numeric_limits<int>::max()) {
        reason.clear();
        diagnostics.clear();
        cell_id = std::numeric_limits<size_t>::max();
        return;
    }

    char reason_buf[2048] = {};
    char diagnostics_buf[4096] = {};
    std::uint64_t local_cell_id = static_cast<std::uint64_t>(matrix_builder.getLastStepFailureCellId());
    if (rank == source_rank) {
        std::string const& local_reason = matrix_builder.getLastStepFailureReason();
        std::strncpy(reason_buf, local_reason.c_str(), sizeof(reason_buf) - 1);
        std::string const& local_diagnostics =
            matrix_builder.getLastStepFailureDiagnostics();
        std::strncpy(diagnostics_buf, local_diagnostics.c_str(),
                     sizeof(diagnostics_buf) - 1);
    }
    MPI_Bcast(reason_buf, static_cast<int>(sizeof(reason_buf)), MPI_CHAR, source_rank, MPI_COMM_WORLD);
    MPI_Bcast(diagnostics_buf, static_cast<int>(sizeof(diagnostics_buf)),
              MPI_CHAR, source_rank, MPI_COMM_WORLD);
    MPI_Bcast(&local_cell_id, 1, MPI_UINT64_T, source_rank, MPI_COMM_WORLD);

    reason = reason_buf;
    diagnostics = diagnostics_buf;
    cell_id = static_cast<size_t>(local_cell_id);
}

std::vector<size_t> collect_step_failure_cells(
    RadiationDriver const& matrix_builder)
{
    int rank_count = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    std::uint64_t local_cell_id = std::numeric_limits<std::uint64_t>::max();
    if(!matrix_builder.getLastStepFailureReason().empty() &&
       matrix_builder.getLastStepFailureIsCellLocal() &&
       matrix_builder.getLastStepFailureCellId() !=
           std::numeric_limits<size_t>::max())
        local_cell_id = static_cast<std::uint64_t>(
            matrix_builder.getLastStepFailureCellId());

    std::vector<std::uint64_t> gathered(
        static_cast<std::size_t>(rank_count),
        std::numeric_limits<std::uint64_t>::max());
    MPI_Allgather(&local_cell_id, 1, MPI_UINT64_T, gathered.data(), 1,
                  MPI_UINT64_T, MPI_COMM_WORLD);

    std::vector<size_t> result;
    result.reserve(gathered.size());
    for(std::uint64_t const cell_id : gathered)
        if(cell_id != std::numeric_limits<std::uint64_t>::max())
            result.push_back(static_cast<size_t>(cell_id));
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

bool has_collective_step_failure(RadiationDriver const& matrix_builder)
{
    int local_collective =
        !matrix_builder.getLastStepFailureReason().empty() &&
        !matrix_builder.getLastStepFailureIsCellLocal() &&
        !matrix_builder.getLastStepFailureIsRemote() ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &local_collective, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    return local_collective != 0;
}
#else
void broadcast_step_failure(RadiationDriver const& matrix_builder,
                            std::string& reason,
                            size_t& cell_id,
                            std::string& diagnostics)
{
    reason = matrix_builder.getLastStepFailureReason();
    diagnostics = matrix_builder.getLastStepFailureDiagnostics();
    cell_id = matrix_builder.getLastStepFailureCellId();
}

std::vector<size_t> collect_step_failure_cells(
    RadiationDriver const& matrix_builder)
{
    if(matrix_builder.getLastStepFailureReason().empty() ||
       !matrix_builder.getLastStepFailureIsCellLocal() ||
       matrix_builder.getLastStepFailureCellId() ==
           std::numeric_limits<size_t>::max())
        return std::vector<size_t>();
    return std::vector<size_t>{matrix_builder.getLastStepFailureCellId()};
}

bool has_collective_step_failure(RadiationDriver const& matrix_builder)
{
    return !matrix_builder.getLastStepFailureReason().empty() &&
        !matrix_builder.getLastStepFailureIsCellLocal() &&
        !matrix_builder.getLastStepFailureIsRemote();
}
#endif

bool passive_radiation_change_fraction(
    ComputationalCell3D const& primitive,
    Conserved3D const& before,
    Conserved3D const& after,
    double maximum_radiation_energy_density,
    double& difference)
{
    if(!(primitive.density > 0) || !std::isfinite(primitive.density) ||
       !(before.mass > 0) || !std::isfinite(before.mass))
        return false;

    double const inverse_volume = primitive.density / before.mass;
    double const before_Erad = before.Erad * inverse_volume;
    double const after_Erad = after.Erad * inverse_volume;
    double const epsilon = 64 * std::numeric_limits<double>::epsilon();
    double const total_reference = std::max(
        std::abs(before_Erad), std::abs(after_Erad));
    bool changed = std::abs(after_Erad - before_Erad) >
        epsilon * std::max(1.0, total_reference);
    difference = std::abs(after_Erad - before_Erad) /
        (std::abs(after_Erad) +
         0.02 * maximum_radiation_energy_density +
         std::numeric_limits<double>::min());

    std::size_t const group_count = std::min(
        before.Eg.size(), after.Eg.size());
    double const group_floor = group_count == 0 ? 0 :
        (0.01 * maximum_radiation_energy_density +
         std::abs(after_Erad)) /
            static_cast<double>(group_count);
    for(std::size_t group = 0; group < group_count; ++group) {
        double const before_group = before.Eg[group] * inverse_volume;
        double const after_group = after.Eg[group] * inverse_volume;
        double const group_reference = std::max(
            std::abs(before_group), std::abs(after_group));
        double const delta = std::abs(after_group - before_group);
        changed = changed || delta >
            epsilon * std::max(1.0, group_reference);
        difference = std::max(difference,
            0.2 * delta /
                (std::abs(after_group) + group_floor +
                 std::numeric_limits<double>::min()));
    }
    return changed && std::isfinite(difference) && difference > 0;
}

double passive_radiation_wake_interval(
    double difference,
    double transfer_dt,
    double time_quantum)
{
    if(!(difference > 0) || !std::isfinite(difference) ||
       !(transfer_dt > 0) || !std::isfinite(transfer_dt) ||
       !(time_quantum > 0) || !std::isfinite(time_quantum))
        return std::numeric_limits<double>::infinity();

    // The accepted conservative transfer was integrated over the
    // active-passive face timestep, not the scheduler's event spacing.
    // Extrapolate that measured rate to the same 15% target used by the
    // active radiation limiter.  If the accepted change already exceeded
    // the target, the earliest representable wake is one scheduler quantum.
    double const safe_dt = transfer_dt * 0.15 /
        std::max(difference, std::numeric_limits<double>::min());
    if(!std::isfinite(safe_dt))
        return std::numeric_limits<double>::infinity();
    return std::max(time_quantum, safe_dt);
}

class RetryProbeBackoff
{
public:
    void restore(std::uint64_t const successful_candidates,
                 std::uint64_t const cooldown)
    {
        successful_candidates_ = static_cast<std::size_t>(std::min(
            successful_candidates,
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())));
        cooldown_ = static_cast<std::size_t>(std::max<std::uint64_t>(
            initial_cooldown,
            std::min<std::uint64_t>(maximum_cooldown, cooldown)));
    }

    void recordAcceptance(bool const was_probe)
    {
        if(was_probe)
            cooldown_ = initial_cooldown;
        ++successful_candidates_;
    }

    bool readyToProbe() const
    {return successful_candidates_ >= cooldown_;}

    void recordProbeScheduled()
    {successful_candidates_ = 0;}

    void recordFailure(bool const was_probe)
    {
        cooldown_ = was_probe ?
            std::min(maximum_cooldown, 2 * cooldown_) :
            initial_cooldown;
        successful_candidates_ = 0;
    }

    std::size_t cooldown() const
    {return cooldown_;}

    std::size_t successfulCandidates() const
    {return successful_candidates_;}

private:
    static constexpr std::size_t initial_cooldown = 8;
    static constexpr std::size_t maximum_cooldown = 256;
    std::size_t successful_candidates_ = 0;
    std::size_t cooldown_ = initial_cooldown;
};

bool identityOwnedMoveRequested()
{
    static bool const enabled = []()
    {
        char const* const value =
            std::getenv("RICH_INDIVIDUAL_IDENTITY_OWNED_MOVE");
        bool local_enabled = false;
        bool local_valid = true;
        if(value != nullptr && value[0] != '\0') {
            if(std::strcmp(value, "1") == 0 ||
               std::strcmp(value, "true") == 0 ||
               std::strcmp(value, "on") == 0 ||
               std::strcmp(value, "yes") == 0)
                local_enabled = true;
            else if(std::strcmp(value, "0") != 0 &&
                    std::strcmp(value, "false") != 0 &&
                    std::strcmp(value, "off") != 0 &&
                    std::strcmp(value, "no") != 0)
                local_valid = false;
        }
        unsigned int option_state = !local_valid ? 4u :
            (local_enabled ? 2u : 1u);
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &option_state, 1, MPI_UNSIGNED, MPI_BOR,
                      MPI_COMM_WORLD);
#endif
        if((option_state & 4u) != 0u)
            throw UniversalError(
                "RICH_INDIVIDUAL_IDENTITY_OWNED_MOVE must be unset or one "
                "of 0, 1, false, true, off, on, no, yes");
        if((option_state & 3u) == 3u)
            throw UniversalError(
                "RICH_INDIVIDUAL_IDENTITY_OWNED_MOVE differs between MPI "
                "ranks");
        return (option_state & 2u) != 0u;
    }();
    return enabled;
}

// Whether a rejected radiation candidate also lowers the hydro bins of the
// cells it touched at the next commit (the historical behaviour), or only
// sub-cycles the radiation inside the event the way the global scheme
// sub-steps a rejected candidate inside its step.  On the TDE the bin
// lowering, cascaded by neighbour closure, set an event cadence 20-60x finer
// than the global dt that the same physics ran at (jobs 10199442 vs
// 10199059).  Default keeps the historical behaviour; values must agree on
// every MPI rank.
// RICH_INDIVIDUAL_RADIATION_ANCHOR_SUBCYCLE (default on): a gray relative-
// change limit in [0.5, 1) x the anchor interval lets the next hydro interval
// reach the anchor bin and caps the radiation candidate fraction at the cell's
// next activation, instead of halving the bin (TDE, job 10232945: a limit 4 %
// below the anchor put one cell, and through closure its neighbours, a whole
// bin finer).  Agreed across ranks.
bool radiationAnchorSubcycle()
{
    static bool const enabled = []()
    {
        char const* const value = std::getenv("RICH_INDIVIDUAL_RADIATION_ANCHOR_SUBCYCLE");
        int local = 1;
        if(value != nullptr && value[0] != '\0')
            local = std::strcmp(value, "0") == 0 ? 0 : (std::strcmp(value, "1") == 0 ? 1 : -1);
        int extrema[2] = {local, -local};
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, extrema, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
        if(extrema[0] < 0 || extrema[0] != -extrema[1])
            throw std::invalid_argument(
                "RICH_INDIVIDUAL_RADIATION_ANCHOR_SUBCYCLE must be 0 or 1 on every rank");
        return extrema[0] != 0;
    }();
    return enabled;
}

// Lower edge of the anchor-crossing subcycling band as a fraction of the
// anchor interval (RICH_INDIVIDUAL_RADIATION_ANCHOR_BAND, default 0.5; e.g.
// 0.25 allows up to four radiation pieces per anchor interval).  Agreed.
double anchorSubcycleBandLow()
{
    static double const low = []()
    {
        char const* const text = std::getenv("RICH_INDIVIDUAL_RADIATION_ANCHOR_BAND");
        double value = 0.5;
        if(text != nullptr && text[0] != '\0')
        {
            char* end = nullptr;
            value = std::strtod(text, &end);
            if(end == text || *end != '\0' || !(value > 0 && value < 1))
                value = -1;
        }
        double extrema[2] = {value, -value};
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, extrema, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
        if(extrema[0] < 0 || extrema[0] != -extrema[1])
            throw std::invalid_argument(
                "RICH_INDIVIDUAL_RADIATION_ANCHOR_BAND must be one number in (0, 1) on every rank");
        return extrema[0];
    }();
    return low;
}

// RICH_INDIVIDUAL_RADIATION_ENTRY_PROBE (default off): allow an earned recovery
// probe as an event's *first* radiation candidate.  The in-event probe can only
// fire when the doubled fraction still fits in the remaining event fraction, so
// a persisted ceiling of 1/2 traps the controller: the two accepted halves fill
// the event exactly, no probe is ever scheduled, and every later event keeps
// paying two solves even once the measured success threshold is met.  The global
// path escapes this only because it re-initializes the ceiling every step.
// Agreed across ranks.
//
// Enabling it also makes the persisted retry controller collective and strictly
// earned: the restored ceiling and backoff counters are reduced to one common
// triple at event entry (so the attempted fraction and the probe classification
// are identical on every rank, probe or not, and every rank persists that same
// triple), and a probe raises only the attempted interval -- the committed
// ceiling is promoted when the probe candidate is accepted, never before.  The
// second part covers the in-event probe as well, so a rejected probe can no
// longer keep a promotion that no candidate earned.
bool radiationEntryProbe()
{
    static bool const enabled = []()
    {
        char const* const value =
            std::getenv("RICH_INDIVIDUAL_RADIATION_ENTRY_PROBE");
        int local = 0;
        if(value != nullptr && value[0] != '\0')
            local = std::strcmp(value, "0") == 0 ? 0 : (std::strcmp(value, "1") == 0 ? 1 : -1);
        int extrema[2] = {local, -local};
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, extrema, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
        if(extrema[0] < 0 || extrema[0] != -extrema[1])
            throw std::invalid_argument(
                "RICH_INDIVIDUAL_RADIATION_ENTRY_PROBE must be 0 or 1 on every rank");
        return extrema[0] != 0;
    }();
    return enabled;
}

bool radiationRetryLimitsBins()
{
    static bool const enabled = []()
    {
        char const* const value =
            std::getenv("RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS");
        bool local_enabled = true;
        bool local_valid = true;
        if(value != nullptr && value[0] != '\0') {
            if(std::strcmp(value, "0") == 0 ||
               std::strcmp(value, "false") == 0 ||
               std::strcmp(value, "off") == 0 ||
               std::strcmp(value, "no") == 0)
                local_enabled = false;
            else if(std::strcmp(value, "1") != 0 &&
                    std::strcmp(value, "true") != 0 &&
                    std::strcmp(value, "on") != 0 &&
                    std::strcmp(value, "yes") != 0)
                local_valid = false;
        }
        unsigned int option_state = !local_valid ? 4u :
            (local_enabled ? 2u : 1u);
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &option_state, 1, MPI_UNSIGNED, MPI_BOR,
                      MPI_COMM_WORLD);
#endif
        if((option_state & 4u) != 0u)
            throw UniversalError(
                "RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS must be unset or "
                "one of 0, 1, false, true, off, on, no, yes");
        if((option_state & 3u) == 3u)
            throw UniversalError(
                "RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS differs between "
                "MPI ranks");
        return (option_state & 2u) != 0u;
    }();
    return enabled;
}

// RICH_TEST_FORCED_RADIATION_REJECTION (test seam, strict 0/1, default 0): every individual radiation candidate is
// first run in full and rejected by one rank (rotating over the ranks) at its last collective acceptance check, then
// replayed at the same interval, without touching the retry controller, limiter or retry reports.  With bitwise
// rollback, the run's INDIVIDUAL_STATE_HASH stream equals the same run with the switch off.  Collective on first use;
// must agree on every rank.
bool testForcedRadiationRejection()
{
    static bool const enabled = []()
    {
        char const* const value = std::getenv("RICH_TEST_FORCED_RADIATION_REJECTION");
        int local = -1;
        if(value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0)
            local = 0;
        else if(std::strcmp(value, "1") == 0)
            local = 1;
        int extrema[2] = {local, -local};
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, extrema, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
        if(extrema[0] != -extrema[1])
            throw std::invalid_argument("RICH_TEST_FORCED_RADIATION_REJECTION differs between MPI ranks");
        if(extrema[0] < 0)
            throw std::invalid_argument("RICH_TEST_FORCED_RADIATION_REJECTION must be 0 or 1");
        return extrema[0] != 0;
    }();
    return enabled;
}

template<typename T>
class MovedOwnedStorageGuard
{
public:
    MovedOwnedStorageGuard(std::vector<T>& canonical,
                           std::vector<T>& local,
                           std::size_t const owned_size,
                           bool const active) :
        canonical_(canonical), local_(local), owned_size_(owned_size),
        active_(active)
    {}

    ~MovedOwnedStorageGuard()
    {
        if(active_)
            restore();
    }

    void commit()
    {
        if(!active_)
            return;
        restore();
        active_ = false;
    }

private:
    void restore()
    {
        local_.resize(owned_size_);
        canonical_ = std::move(local_);
    }

    std::vector<T>& canonical_;
    std::vector<T>& local_;
    std::size_t owned_size_;
    bool active_;
};

// Ghost sync of the sparse canonical path after an accepted non-final piece.  The full-copy path synchronized from a
// fresh copy of the canonical array whose target slots the sync overwrote with the local targets.  Syncing through
// the canonical array itself reads the same values; restoring the target slots afterwards leaves the canonical array
// exactly as the full-copy path left it.  TargetsAtSync receives the target values the sync wrote.
template<typename T>
void SyncThroughCanonical(Tessellation3D const& Tess, std::vector<std::size_t> const& TargetToCanonical,
                          std::vector<T>& Local, std::vector<T>& Canonical, std::vector<T>& TargetsAtSync)
{
    std::size_t const Targets = TargetToCanonical.size();
    std::vector<T> Saved;
    Saved.reserve(Targets);
    for(std::size_t Target = 0; Target < Targets; ++Target)
        Saved.push_back(Canonical[TargetToCanonical[Target]]);
    // Every copy that can allocate precedes the first write into Canonical
    // (the sync does not change Local's owned targets), so a throw can never
    // leave temporary target values in it.
    TargetsAtSync.assign(Local.begin(), Local.begin() + static_cast<std::ptrdiff_t>(Targets));
    try {
        Tess.SyncPartialBuildData(Local, Canonical);
    }
    catch(...) {
        for(std::size_t Target = 0; Target < Targets; ++Target)
            Canonical[TargetToCanonical[Target]] = Saved[Target];
        throw;
    }
    for(std::size_t Target = 0; Target < Targets; ++Target)
        Canonical[TargetToCanonical[Target]] = Saved[Target];
}

// RICH_INDIVIDUAL_RADIATION_SPARSE_CHECK=1 (default off, rank-local, no collectives): the sparse canonical path also
// keeps the full copies it replaces and aborts if its rebuilt end-of-event state differs from them in any value the
// wake estimate reads.  Validation only; it never changes the step.
bool SparseCanonicalCheckRequested()
{
    static bool const Enabled = []()
    {
        char const* const Value = std::getenv("RICH_INDIVIDUAL_RADIATION_SPARSE_CHECK");
        return Value != nullptr && std::strcmp(Value, "1") == 0;
    }();
    return Enabled;
}

bool SameBits(double const Left, double const Right)
{
    return std::memcmp(&Left, &Right, sizeof(double)) == 0;
}

// Equal in every value passive_radiation_change_fraction() and the energy-density maximum read.
bool SameWakeInputs(ComputationalCell3D const& LeftCell, Conserved3D const& LeftExtensive,
                    ComputationalCell3D const& RightCell, Conserved3D const& RightExtensive)
{
    bool Same = SameBits(LeftCell.density, RightCell.density) && SameBits(LeftExtensive.mass, RightExtensive.mass) &&
        SameBits(LeftExtensive.Erad, RightExtensive.Erad) && LeftExtensive.Eg.size() == RightExtensive.Eg.size();
    for(std::size_t Group = 0; Same && Group < LeftExtensive.Eg.size(); ++Group)
        Same = SameBits(LeftExtensive.Eg[Group], RightExtensive.Eg[Group]);
    return Same;
}

double D2Seconds()
{
#ifdef RICH_MPI
    return MPI_Wtime();
#else
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

// RICH_RADIATION_REMAP_MINIMAL.  The radiation step reads only the scalars, the activity (mask and indices), the
// applied intervals and the bins of its remapped context: RadiationStep, RadiationDriver, Diffusion and
// MultigroupDiffusion never read primitive_ticks, point_velocities, cached_accelerations or
// gravity_half_kick_pending, and hydroFaceTimeStep (the one accessor of primitive_ticks) is not called there.  This
// builds what ActiveMeshView::remapContext(Global, false) builds for the fields read, with its collectives in the
// same order, and leaves the four unread fields empty, which saves their resizes, copies and four ghost syncs.
IndividualStepContext MinimalRadiationContext(ActiveMeshView const& View, Tessellation3D const& Tess,
    IndividualStepContext const& Global)
{
    IndividualStepContext Local;
    Local.previous_event_tick = Global.previous_event_tick;
    Local.event_tick = Global.event_tick;
    Local.time_origin = Global.time_origin;
    Local.previous_event_time = Global.previous_event_time;
    Local.event_time = Global.event_time;
    Local.time_quantum = Global.time_quantum;
    Local.mesh_build_policy = Global.mesh_build_policy;
    Local.partial_build_fraction = Global.partial_build_fraction;
    Local.verify_partial_build = Global.verify_partial_build;
    Local.radiation_repair_accounting = Global.radiation_repair_accounting;
    Local.radiation_defect_accounting = Global.radiation_defect_accounting;
    std::size_t const MeshSize = View.meshLocalSize();
    Local.active_mask.assign(MeshSize, 0);
    Local.cell_time_steps.assign(MeshSize, std::numeric_limits<double>::infinity());
    // The same collective bin decision as remapContext: an owner without time steps votes neutral.
    int BinsVote = Global.cell_time_steps.empty() ? 1 :
        (Global.cell_time_bins.size() == Global.cell_time_steps.size() ? 1 : 0);
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &BinsVote, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
    bool const HaveBins = BinsVote != 0;
    if(HaveBins)
        Local.cell_time_bins.assign(MeshSize, 0);
    for(std::size_t LocalIndex = 0; LocalIndex < MeshSize; ++LocalIndex) {
        std::size_t const GlobalIndex = View.meshLocalToGlobal(LocalIndex);
        if(GlobalIndex == ActiveMeshView::invalidIndex())
            continue;
        Local.cell_time_steps[LocalIndex] = Global.cellTimeStep(GlobalIndex);
        if(HaveBins)
            Local.cell_time_bins[LocalIndex] = Global.cell_time_bins[GlobalIndex];
        if(LocalIndex < View.localSize() && Global.isActive(GlobalIndex)) {
            Local.active_mask[LocalIndex] = 1;
            Local.active_indices.push_back(LocalIndex);
        }
    }
#ifdef RICH_MPI
    {
        std::vector<unsigned char> AllActive = Global.active_mask;
        Tess.SyncPartialBuildData(Local.active_mask, AllActive);
    }
    Tess.SyncPartialBuildDataReadOnly(Local.cell_time_steps, Global.cell_time_steps);
    if(HaveBins)
        Tess.SyncPartialBuildDataReadOnly(Local.cell_time_bins, Global.cell_time_bins);
#else
    (void)Tess;
#endif
    return Local;
}

// RICH_RADIATION_REMAP_MINIMAL=2: every field the radiation step reads must match remapContext bit for bit.
void VerifyMinimalRadiationContext(IndividualStepContext const& Old, IndividualStepContext const& New, int const Rank)
{
    std::string Field;
    std::size_t Index = 0;
    auto const SameBitsOf = [](double const Left, double const Right)
    {
        return std::memcmp(&Left, &Right, sizeof(double)) == 0;
    };
    if(Old.previous_event_tick != New.previous_event_tick || Old.event_tick != New.event_tick ||
       !SameBitsOf(Old.time_origin, New.time_origin) ||
       !SameBitsOf(Old.previous_event_time, New.previous_event_time) ||
       !SameBitsOf(Old.event_time, New.event_time) || !SameBitsOf(Old.time_quantum, New.time_quantum) ||
       !SameBitsOf(Old.anchor_time_step, New.anchor_time_step) || Old.mesh_build_policy != New.mesh_build_policy ||
       !SameBitsOf(Old.partial_build_fraction, New.partial_build_fraction) ||
       Old.verify_partial_build != New.verify_partial_build ||
       Old.radiation_repair_accounting != New.radiation_repair_accounting ||
       Old.radiation_defect_accounting != New.radiation_defect_accounting ||
       !Old.radiation_accuracy_limits.empty() || !New.radiation_accuracy_limits.empty())
        Field = "scalars";
    else if(Old.active_indices != New.active_indices) {
        Field = "active_indices";
        while(Index < Old.active_indices.size() && Index < New.active_indices.size() &&
              Old.active_indices[Index] == New.active_indices[Index])
            ++Index;
    }
    else if(Old.active_mask != New.active_mask) {
        Field = "active_mask";
        while(Index < Old.active_mask.size() && Index < New.active_mask.size() &&
              Old.active_mask[Index] == New.active_mask[Index])
            ++Index;
    }
    else if(Old.cell_time_bins != New.cell_time_bins) {
        Field = "cell_time_bins";
        while(Index < Old.cell_time_bins.size() && Index < New.cell_time_bins.size() &&
              Old.cell_time_bins[Index] == New.cell_time_bins[Index])
            ++Index;
    }
    else if(Old.cell_time_steps.size() != New.cell_time_steps.size())
        Field = "cell_time_steps_size";
    else
        for(Index = 0; Index < Old.cell_time_steps.size(); ++Index)
            if(!SameBitsOf(Old.cell_time_steps[Index], New.cell_time_steps[Index])) {
                Field = "cell_time_steps";
                break;
            }
    if(Field.empty()) {
        RecordRadiationRemapVerification();
        return;
    }
    std::cerr << "RICH_RADIATION_REMAP_MINIMAL verify mismatch: rank=" << Rank << " field=" << Field
              << " index=" << Index << std::endl;
#ifdef RICH_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    std::abort();
}

// Event scope of the driver's canonical stable-ID index and write journal; also ends both on an exception.
class IndividualCanonicalScopeGuard
{
public:
    explicit IndividualCanonicalScopeGuard(RadiationDriver const& Driver) : Driver_(Driver)
    {
        Driver_.BeginIndividualCanonicalScope();
    }

    ~IndividualCanonicalScopeGuard()
    {
        Driver_.EndIndividualCanonicalWriteJournal();
        Driver_.EndIndividualCanonicalScope();
    }

    IndividualCanonicalScopeGuard(IndividualCanonicalScopeGuard const&) = delete;
    IndividualCanonicalScopeGuard& operator=(IndividualCanonicalScopeGuard const&) = delete;

private:
    RadiationDriver const& Driver_;
};

} // namespace
RadiationStep::RadiationStep(Tessellation3D &tess, std::vector<ComputationalCell3D> &cells,
                    std::vector<Conserved3D> &extensives,
                    ProgressTracker &pt,
                    #ifdef RICH_MPI
                        std::shared_ptr<CostCalculator3D> cost,
                    #endif // RICH_MPI
                    const RadiationDriver &matrix_builder, bool /*no_hydro*/) :
                    tess(tess), cells(cells), extensives(extensives), pt(pt), matrix_builder(matrix_builder)
                        , suggested_dt(std::numeric_limits<double>::max())
                    #ifdef RICH_MPI
                        , cost(cost)
                    #endif // RICH_MPI
{}

#ifdef RICH_MPI
    bool RadiationStep::allowRebalance(void)
    {
        return this->cost != nullptr;
    }

    std::string RadiationStep::getRequiredLB(void) const
    {
        if (!this->cost)
            return "";
        return "hydro";
    }

    std::vector<double> RadiationStep::getLoadBalanceWeights(void)
    {
        if (this->cost)
            return this->cost->CalculateCost(this->tess, this->cells);
        return std::vector<double>(this->tess.GetPointNo(), 1.0);
    }
#endif // RICH_MPI

void RadiationStep::step(double dt)
{
	int total_iters = 0;
	double const CG_eps = 1e-11;
	size_t const N = this->tess.GetPointNo();

#ifdef DEBUG
	if(N == 0) std::clog<<"Zero cells in RadiationTimeStep"<<std::endl;
#endif

	int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

	double total_elapsed_time = 0;
	double dt_try = dt;
	double dt_try_ceiling = dt;
	RetryProbeBackoff retry_probe_backoff;
	bool candidate_is_retry_probe = false;
	size_t reduce_counter = 0;
	int max_iter_done = 0;


	this->matrix_builder.prestep(this->tess, this->cells);
	MEMORY_DEBUG_PRINT("radiation: after prestep");
	while(total_elapsed_time < dt * 0.9999999)
	{
		dt_try = std::min(dt_try, dt - total_elapsed_time);

		double const candidate_time = this->pt.getTime() + total_elapsed_time;
		double const attempted_dt = dt_try;
		auto const candidate_start = std::chrono::steady_clock::now();
		bool step_success = this->matrix_builder.step(CG_eps, total_iters,
			this->tess, this->cells, this->extensives, dt_try,
			candidate_time);
		double const candidate_seconds = std::chrono::duration<double>(
			std::chrono::steady_clock::now() - candidate_start).count();
		MEMORY_DEBUG_PRINT("radiation: after solver step");

		max_iter_done = std::max(max_iter_done, total_iters);
		
		if(not step_success)
		{
			bool const rejected_retry_probe = candidate_is_retry_probe;
			candidate_is_retry_probe = false;
			retry_probe_backoff.recordFailure(rejected_retry_probe);
			reduce_counter++;
			dt_try *= 0.5;
			dt_try_ceiling = std::min(dt_try_ceiling, dt_try);
			std::string reason;
			std::string diagnostics;
				size_t cell_id = std::numeric_limits<size_t>::max();
			broadcast_step_failure(
				this->matrix_builder, reason, cell_id, diagnostics);
			StepRetryRecord retry;
			retry.attempted_dt_min = attempted_dt;
			retry.attempted_dt_max = attempted_dt;
			retry.elapsed_seconds = candidate_seconds;
			retry.reason = reason.empty() ? "solver_rejected" : reason;
			retry.diagnostics = diagnostics;
			retry.representative_cell = cell_id;
			reportStepRetry(retry);
			
			double const next_elapsed_time = total_elapsed_time + dt_try;
			double const accepted_time = this->pt.getTime() + total_elapsed_time;
			double const next_candidate_end_time =
				this->pt.getTime() + next_elapsed_time;
			if(!(dt_try > 0) || !std::isfinite(dt_try) ||
			   !(next_elapsed_time > total_elapsed_time) ||
			   !(next_candidate_end_time > accepted_time)) {
				if (rank == 0) {
					std::ostringstream msg;
					msg << "Radiation step failed: candidate cannot advance after halving"
					    << " (target dt " << dt
					    << ", completed " << total_elapsed_time
					    << ", candidate dt " << dt_try << ")";
					if (!reason.empty()) {
						msg << ", last failure: " << reason;
						if (cell_id != std::numeric_limits<size_t>::max())
							msg << ", example cell ID " << cell_id;
					}
					std::clog << msg.str() << std::endl;
				}
				UniversalError eo(
					"radiation candidate cannot advance after halving");
				eo.addEntry("target dt", dt);
				eo.addEntry("completed elapsed time", total_elapsed_time);
				eo.addEntry("candidate dt", dt_try);
				eo.addEntry("accepted time", accepted_time);
				eo.addEntry("next candidate end time", next_candidate_end_time);
				if (!reason.empty()) {
					eo.addEntry("last radiation failure", reason);
					if (cell_id != std::numeric_limits<size_t>::max())
						eo.addEntry("example cell ID", cell_id);
				}
				throw eo;
			}
		}
		else {
			total_elapsed_time += dt_try;
			retry_probe_backoff.recordAcceptance(candidate_is_retry_probe);
			candidate_is_retry_probe = false;
			// A rejected candidate can force a very small fractional step.  Hold
			// it before probing a factor-two larger interval.  Failed recovery
			// probes exponentially extend the cooldown; a successful probe resets
			// it so transient restrictions still recover quickly.
			double const old_ceiling = dt_try_ceiling;
			double next_ceiling = old_ceiling;
			if(retry_probe_backoff.readyToProbe() && old_ceiling < dt)
				next_ceiling = std::min(dt, 2.0 * old_ceiling);
			double const next_dt = std::min(
				dt - total_elapsed_time,
				std::min(2.0 * dt_try, next_ceiling));
			if(next_dt > old_ceiling) {
				dt_try_ceiling = next_ceiling;
				retry_probe_backoff.recordProbeScheduled();
				candidate_is_retry_probe = true;
			}
			dt_try = next_dt;
		}
	}

	this->suggested_dt = this->matrix_builder.calculate_dt(dt, this->tess, this->cells);
	cell_limit_ids.clear();
	cell_limit_values.clear();
	cell_limit_minimum = std::numeric_limits<double>::infinity();
	if(std::vector<double> const* per_cell = this->matrix_builder.lastCellTimeStepLimits())
	{
		std::size_t const points = this->tess.GetPointNo();
		if(per_cell->size() == points && this->cells.size() >= points)
		{
			cell_limit_values = *per_cell;
			cell_limit_ids.resize(points);
			for(std::size_t i = 0; i < points; ++i)
			{
				cell_limit_ids[i] = this->cells[i].ID;
				if(cell_limit_values[i] > 0)
					cell_limit_minimum = std::min(cell_limit_minimum,
						cell_limit_values[i]);
			}
		}
	}

	this->matrix_builder.poststep();
	MEMORY_DEBUG_PRINT("radiation: after poststep");

#ifdef RICH_MPI
	MPI_exchange_data(this->tess, this->cells, true);
#endif
	
    
	// double grow_factor = 1.25;
	// if(max_iter_done > 200)
	// 	grow_factor = 1.02;
	// else
	// 	if(max_iter_done > 125)
	// 		grow_factor = 1.05;

	// new_dt = std::min(new_dt, dt*grow_factor) * std::pow(0.5, std::max(static_cast<double>(reduce_counter), 0.0));
	// if(max_iter_done > 300)
	// 	new_dt = dt * 0.9;
	// return this->radiation_dt_;
}

bool RadiationStep::supportsIndividualTimeSteps(void) const
{
    return matrix_builder.supportsIndividualTimeSteps();
}

void RadiationStep::afterIndividualAMR(void) noexcept
{
    matrix_builder.releaseIndividualTopologyStorage();
}

void RadiationStep::onIndividualForceAllActiveLatch(void) noexcept
{
    matrix_builder.releaseIndividualTopologyStorage();
}

void RadiationStep::beforeIndividualRebalance(void) noexcept
{
    matrix_builder.releaseIndividualTopologyStorage();
}

void RadiationStep::stepIndividual(IndividualStepContext const& context)
{
    if(!matrix_builder.supportsIndividualTimeSteps())
        throw std::runtime_error("radiation driver does not support individual timesteps");
#ifndef RICH_MPI
    if(context.active_indices.empty())
        return;
#endif

    // First use is collective: every rank reads the s49 switches here.
    RadiationS49Switches const& S49 = RadiationS49RuntimeSwitches();
    bool const D2Timers = S49.Timers != 0;
    double const D2EventStart = D2Timers ? D2Seconds() : 0;
    double const D2EventMpiStart = D2Timers ? mpi_wait_profiler::Seconds() : 0;
    std::size_t D2Candidates = 0;
    if(D2Timers)
        BeginRadiationD2Event();

    // Global-step limits describe a state individual events have left.
    cell_limit_ids.clear();
    cell_limit_values.clear();
    cell_limit_minimum = std::numeric_limits<double>::infinity();
    bool const trace_performance = individualPerformanceTraceEnabled();
    last_individual_performance.clear();
    auto phase_start = std::chrono::steady_clock::now();
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    std::size_t const canonical_owned_size = cells.size();
    ActiveMeshView const mesh_view(tess, canonical_owned_size);
    IndividualStepContext const local_context = S49.RemapMinimal == 1 ?
        MinimalRadiationContext(mesh_view, tess, context) :
        mesh_view.remapContext(context, false);
    if(S49.RemapMinimal == 2)
        VerifyMinimalRadiationContext(local_context, MinimalRadiationContext(mesh_view, tess, context), rank);
    bool owned_mapping_identity =
        mesh_view.localSize() == canonical_owned_size;
    for(std::size_t local = 0;
        owned_mapping_identity && local < mesh_view.localSize(); ++local)
        owned_mapping_identity = mesh_view.localToGlobal(local) == local;
    bool local_all_active =
        context.active_mask.size() == canonical_owned_size &&
        context.active_indices.size() == canonical_owned_size;
    for(std::size_t owned = 0;
        local_all_active && owned < canonical_owned_size; ++owned)
        local_all_active = context.active_mask[owned] != 0;
    bool use_identity_owned_move = false;
    bool const identity_owned_move_requested = identityOwnedMoveRequested();
    if(identity_owned_move_requested) {
        int collective_eligible =
            owned_mapping_identity && local_all_active ? 1 : 0;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &collective_eligible, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
#endif
        use_identity_owned_move = collective_eligible != 0;
    }
    if((trace_performance || identity_owned_move_requested) && rank == 0)
        RuntimeTraceStream()
            << "RADIATION_IDENTITY_OWNED_MOVE requested="
            << (identity_owned_move_requested ? 1 : 0)
            << " mapping_identity=" << (owned_mapping_identity ? 1 : 0)
            << " all_active=" << (local_all_active ? 1 : 0)
            << " selected=" << (use_identity_owned_move ? 1 : 0)
            << " owned_cells=" << canonical_owned_size << std::endl;

    std::vector<ComputationalCell3D> local_cells;
    std::vector<Conserved3D> local_extensives;
    std::vector<ComputationalCell3D> all_cells;
    std::vector<Conserved3D> all_extensives;
    // Sparse canonical path: when the targets are a small part of the owned cells, the ghost syncs run through the
    // canonical arrays themselves and the full copies all_cells/all_extensives are never made.  Outside a sync the
    // canonical arrays hold what they hold in the full-copy path; the end-of-event wake estimate rebuilds the values
    // all_cells/all_extensives would hold from the last sync's targets, the driver's write journal and the canonical
    // arrays.  Identity moves and dense target sets keep the full copies.
    std::size_t const TargetCount = mesh_view.localSize();
    // RICH_RADIATION_PARTIAL_SETUP_FAST: the sparse path is exact for any target count (the rank-local choice was a
    // cost heuristic), so a rank whose targets exceed a quarter of its owned cells, typically the busiest rank of a
    // partial event, stops copying its owned cells and extensives and evaluating every inactive owned cell.  Mode 2
    // runs such a rank sparse with the full-copy check on.
    bool const DenseTargets = 4 * TargetCount > canonical_owned_size;
    bool const SparseCanonical = !use_identity_owned_move && canonical_owned_size > 0 &&
        extensives.size() == canonical_owned_size && (!DenseTargets || S49.PartialSetupFast != 0) &&
        matrix_builder.JournalsAllCanonicalWrites();
    bool TargetsSynced = false;
    std::vector<ComputationalCell3D> TargetCellsAtSync;
    std::vector<Conserved3D> TargetExtensivesAtSync;
    bool const SparseCanonicalCheck = SparseCanonical &&
        (SparseCanonicalCheckRequested() || (DenseTargets && S49.PartialSetupFast == 2));
    std::vector<ComputationalCell3D> CheckCells;
    std::vector<Conserved3D> CheckExtensives;
    if(use_identity_owned_move) {
        all_cells = cells;
        all_extensives = extensives;
        local_cells = std::move(cells);
        local_extensives = std::move(extensives);
    }
    else {
        local_cells = mesh_view.gatherOwned(cells);
        local_extensives = mesh_view.gatherOwned(extensives);
        if(!SparseCanonical) {
            all_cells = cells;
            all_extensives = extensives;
        }
        else if(SparseCanonicalCheck) {
            CheckCells = cells;
            CheckExtensives = extensives;
        }
    }
    MovedOwnedStorageGuard<ComputationalCell3D> cells_move_guard(
        cells, local_cells, canonical_owned_size, use_identity_owned_move);
    MovedOwnedStorageGuard<Conserved3D> extensives_move_guard(
        extensives, local_extensives, canonical_owned_size,
        use_identity_owned_move);
    if(trace_performance)
        last_individual_performance["radiation-gather-remap"] =
            elapsedSeconds(phase_start);
    phase_start = std::chrono::steady_clock::now();
    if(SparseCanonical) {
        // The targets were just gathered from their canonical slots, so the sync writes back identical values.
        tess.SyncPartialBuildData(local_cells, cells);
        tess.SyncPartialBuildData(local_extensives, extensives);
    }
    else {
        tess.SyncPartialBuildData(local_cells, all_cells);
        tess.SyncPartialBuildData(local_extensives, all_extensives);
    }
    if(trace_performance)
        last_individual_performance["radiation-ghost-sync"] =
            elapsedSeconds(phase_start);

    double const tolerance = 1e-11;
    double completed_fraction = 0;
    IndividualRadiationDefectAccounting* const defect_accounting =
        context.radiation_defect_accounting;
    double candidate_fraction_ceiling = 1;
    RetryProbeBackoff retry_probe_backoff;
    // First use is collective: every rank reads the flag here, before any
    // state-dependent branch.
    bool const entry_probe = radiationEntryProbe();
    // With the entry probe the controller state is a collective object: every
    // rank restores and persists it, so the unified state below stays common
    // instead of a rank without retry history silently vetoing recovery.
    if(defect_accounting != nullptr &&
       (entry_probe ||
        defect_accounting->accepted_dirichlet_candidates > 0 ||
        defect_accounting->defect_rejections > 0)) {
        candidate_fraction_ceiling =
            defect_accounting->cooldown_fraction_ceiling;
        retry_probe_backoff.restore(
            defect_accounting->cooldown_accepted_candidates,
            defect_accounting->cooldown_required_candidates);
    }
    // Anchor-crossing accuracy ceiling (RICH_INDIVIDUAL_RADIATION_ANCHOR_SUBCYCLE):
    // the tightest pending limit / elapsed interval over the active cells,
    // collectively (the limits live in the scheduler state, so they follow
    // migration, AMR and restarts).  Kept apart from the retry ceiling the
    // defect accounting persists.
    double accuracy_ceiling = 1;
    bool const anchor_subcycle = radiationAnchorSubcycle() &&
        context.radiation_accuracy_limits.size() == canonical_owned_size;
    if(radiationAnchorSubcycle())
    {
        // First use is collective: every rank reads the band here.
        static_cast<void>(anchorSubcycleBandLow());
        if(anchor_subcycle)
            for(std::size_t local : local_context.active_indices)
            {
                double const limit = context.radiation_accuracy_limits.at(mesh_view.localToGlobal(local));
                double const interval = local_context.cellTimeStep(local);
                if(limit > 0 && interval > 0)
                    accuracy_ceiling = std::min(accuracy_ceiling, std::max(1e-6, limit / interval));
            }
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &accuracy_ceiling, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
    }
    double candidate_fraction = std::min(candidate_fraction_ceiling, accuracy_ceiling);
    std::uint64_t observed_defect_rejections =
        defect_accounting == nullptr ? 0 :
        defect_accounting->defect_rejections;
    std::uint64_t pending_defect_retry_substeps = 0;
    bool defect_retry_active = false;
    bool candidate_is_retry_probe = false;
    double event_minimum_candidate_fraction = 1;
    std::size_t rejected_candidates = 0;
    std::size_t rejected_retry_probes = 0;
    std::size_t maximum_retry_probe_cooldown = retry_probe_backoff.cooldown();
    std::string representative_reason;
    std::string representative_diagnostics;
    std::size_t representative_cell = std::numeric_limits<std::size_t>::max();
    std::map<std::size_t, double> cell_retry_fractions;
    bool retry_limiter_requires_all_active = false;
    int maximum_iterations = 0;
    std::size_t entry_retry_probes = 0;
    if(entry_probe) {
        // Earned recovery probe at event entry.  The in-event probe below is
        // bounded by the remaining event fraction, so a persisted ceiling of
        // 1/2 can never be tested again (see radiationEntryProbe()).  The first
        // candidate is the one candidate that is not bounded by an
        // already-completed fraction, so schedule the probe here instead --
        // keeping the measured-success backoff, the accuracy ceiling and the
        // existing acceptance gates exactly as they are.
        //
        // Step 1: make the persisted controller state common.  The restored
        // ceiling and backoff counters are per-rank data, so reduce them to the
        // most restrictive consistent triple (smallest ceiling, fewest measured
        // successes, longest cooldown).  Every rank then evaluates the same
        // predicate below, so the attempted fraction and the probe
        // classification are identical on every rank -- for the probe and for
        // the no-probe case alike -- and the unified triple is what gets
        // persisted, so ranks cannot drift apart across events.
        double unified[3] = {
            candidate_fraction_ceiling,
            static_cast<double>(retry_probe_backoff.successfulCandidates()),
            -static_cast<double>(retry_probe_backoff.cooldown())};
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, unified, 3, MPI_DOUBLE, MPI_MIN,
                      MPI_COMM_WORLD);
#endif
        candidate_fraction_ceiling = unified[0];
        retry_probe_backoff.restore(
            static_cast<std::uint64_t>(unified[1]),
            static_cast<std::uint64_t>(-unified[2]));
        maximum_retry_probe_cooldown = retry_probe_backoff.cooldown();
        candidate_fraction = std::min(candidate_fraction_ceiling,
                                      accuracy_ceiling);
        // Step 2: the probe itself raises only the attempted interval.  The
        // committed ceiling stays where it is until this candidate is actually
        // accepted (see the acceptance path), so a solver rejection cannot persist a
        // promotion that no candidate earned.  A rejection additionally halves the fraction and
        // doubles the probe cooldown through the ordinary failure path.
        if(candidate_fraction_ceiling < 1 &&
           retry_probe_backoff.readyToProbe()) {
            double const probe_fraction = std::min(
                std::min(1.0, 2 * candidate_fraction_ceiling), accuracy_ceiling);
            if(probe_fraction > candidate_fraction) {
                candidate_fraction = probe_fraction;
                retry_probe_backoff.recordProbeScheduled();
                candidate_is_retry_probe = true;
                ++entry_retry_probes;
                maximum_retry_probe_cooldown = std::max(
                    maximum_retry_probe_cooldown,
                    retry_probe_backoff.cooldown());
            }
        }
    }
    phase_start = std::chrono::steady_clock::now();
    matrix_builder.beginIndividualPassiveWakeTracking(canonical_owned_size);
    matrix_builder.prestepIndividual(tess, local_cells, local_context);
    if(trace_performance)
        last_individual_performance["radiation-prestep"] =
            elapsedSeconds(phase_start);
    phase_start = std::chrono::steady_clock::now();
    IndividualCanonicalScopeGuard canonical_scope(matrix_builder);
    bool const forced_rejection_test = testForcedRadiationRejection();
    int forced_rejection_rank_count = 1;
#ifdef RICH_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &forced_rejection_rank_count);
#endif
    bool replaying_forced_rejection = false;
    std::size_t forced_test_rejections = 0;
    while(completed_fraction < 1 - 1e-13) {
        candidate_fraction = std::min(candidate_fraction, 1 - completed_fraction);
        event_minimum_candidate_fraction = std::min(
            event_minimum_candidate_fraction, candidate_fraction);
        smallest_individual_candidate_fraction = std::min(
            smallest_individual_candidate_fraction, candidate_fraction);
        int iterations = 0;
        double attempted_dt_min = std::numeric_limits<double>::infinity();
        double attempted_dt_max = 0;
        for(std::size_t const active_index : context.active_indices) {
            double const active_dt =
                candidate_fraction * context.cellTimeStep(active_index);
            attempted_dt_min = std::min(attempted_dt_min, active_dt);
            attempted_dt_max = std::max(attempted_dt_max, active_dt);
        }
        double const candidate_end_time = context.previous_event_time +
            (completed_fraction + candidate_fraction) *
            (context.event_time - context.previous_event_time);
        auto const candidate_start = std::chrono::steady_clock::now();
        // Only an accepted candidate writes canonical extensives, so after the loop the journal holds exactly the
        // final candidate's writes.
        if(SparseCanonical)
            matrix_builder.BeginIndividualCanonicalWriteJournal();
        if(forced_rejection_test && !replaying_forced_rejection)
            matrix_builder.ArmTestForcedRejection(static_cast<int>(
                forced_test_rejections % static_cast<std::size_t>(forced_rejection_rank_count)));
        ++D2Candidates;
        bool const accepted = matrix_builder.stepIndividual(
            tolerance, iterations, tess, local_cells, local_extensives,
            local_context, candidate_fraction, candidate_end_time,
            use_identity_owned_move ? &all_cells : &cells,
            use_identity_owned_move ? &all_extensives : &extensives,
            &mesh_view.localToGlobalMapping());
        double const candidate_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - candidate_start).count();
        bool const forced_rejection = matrix_builder.TakeTestForcedRejection();
        if(!accepted && forced_rejection) {
            // Test seam: replay the same candidate; nothing else of the rejection is recorded.
            ++forced_test_rejections;
            replaying_forced_rejection = true;
            continue;
        }
        replaying_forced_rejection = false;
        if(defect_accounting != nullptr &&
           defect_accounting->defect_rejections >
               observed_defect_rejections) {
            observed_defect_rejections =
                defect_accounting->defect_rejections;
            defect_retry_active = true;
        }
        maximum_iterations = std::max(maximum_iterations, iterations);
        if(accepted) {
            if(defect_retry_active)
                ++pending_defect_retry_substeps;
            completed_fraction += candidate_fraction;
            if(completed_fraction < 1 - 1e-13) {
                // An accepted fractional solve changes owned primitives and
                // extensives.  Refresh the compact ghosts before the next matrix
                // candidate in canonical all-point index space.
                if(SparseCanonical) {
                    SyncThroughCanonical(tess, mesh_view.localToGlobalMapping(), local_cells, cells,
                                         TargetCellsAtSync);
                    SyncThroughCanonical(tess, mesh_view.localToGlobalMapping(), local_extensives, extensives,
                                         TargetExtensivesAtSync);
                    TargetsSynced = true;
                    if(SparseCanonicalCheck) {
                        // What the full-copy path's all_cells/all_extensives hold after this sync.
                        CheckCells = cells;
                        CheckExtensives = extensives;
                        for(std::size_t Target = 0; Target < TargetCount; ++Target) {
                            CheckCells[mesh_view.localToGlobal(Target)] = TargetCellsAtSync[Target];
                            CheckExtensives[mesh_view.localToGlobal(Target)] = TargetExtensivesAtSync[Target];
                        }
                    }
                }
                else {
                    if(!use_identity_owned_move) {
                        all_cells = cells;
                        all_extensives = extensives;
                    }
                    tess.SyncPartialBuildData(local_cells, all_cells);
                    tess.SyncPartialBuildData(local_extensives, all_extensives);
                }
            }
            retry_probe_backoff.recordAcceptance(candidate_is_retry_probe);
            if(entry_probe && candidate_is_retry_probe)
                // Deferred promotion: the probe interval just proved itself, so
                // commit the ceiling now.  Nothing was promoted speculatively,
                // so a rejected probe cannot leave an unaccepted
                // promotion behind.
                candidate_fraction_ceiling = std::max(
                    candidate_fraction_ceiling, candidate_fraction);
            candidate_is_retry_probe = false;
            double const old_ceiling = candidate_fraction_ceiling;
            double next_ceiling = old_ceiling;
            if(retry_probe_backoff.readyToProbe() && old_ceiling < 1)
                next_ceiling = std::min(1.0, 2 * old_ceiling);
            double const next_fraction = std::min(std::min(
                1 - completed_fraction,
                std::min(2 * candidate_fraction,
                         next_ceiling)), accuracy_ceiling);
            if(next_fraction > old_ceiling) {
                if(!entry_probe)
                    // Legacy behaviour: promote to what the next candidate
                    // tests.  With the entry probe the promotion is deferred
                    // to that candidate's acceptance instead.
                    candidate_fraction_ceiling = next_fraction;
                retry_probe_backoff.recordProbeScheduled();
                candidate_is_retry_probe = true;
            }
            candidate_fraction = next_fraction;
            continue;
        }

        bool const rejected_retry_probe = candidate_is_retry_probe;
        candidate_is_retry_probe = false;
        retry_probe_backoff.recordFailure(rejected_retry_probe);
        if(rejected_retry_probe) {
            ++rejected_retry_probes;
            maximum_retry_probe_cooldown = std::max(
                maximum_retry_probe_cooldown,
                retry_probe_backoff.cooldown());
        }
        ++rejected_candidates;
        broadcast_step_failure(
            matrix_builder, representative_reason, representative_cell,
            representative_diagnostics);
        StepRetryRecord retry;
        retry.attempted_dt_min = attempted_dt_min;
        retry.attempted_dt_max = attempted_dt_max;
        retry.elapsed_seconds = candidate_seconds;
        retry.reason = representative_reason.empty() ?
            "solver_rejected" : representative_reason;
        retry.diagnostics = representative_diagnostics;
        retry.representative_cell = representative_cell;
		reportStepRetry(retry);
        std::vector<std::size_t> const failed_cells =
            collect_step_failure_cells(matrix_builder);
        bool const collective_failure =
            has_collective_step_failure(matrix_builder);
        candidate_fraction *= 0.5;
        // A rejected candidate is transactionally rolled back.  Hold the
        // accepted substeps at or below its halved interval for a short
        // cooldown instead of probing the same known-bad larger interval
        // after every success.  Sustained acceptance then permits one
        // conservative doubling probe so a transient restriction does not
        // force the entire event to use its smallest interval.
        candidate_fraction_ceiling = std::min(
            candidate_fraction_ceiling, candidate_fraction);
        if(collective_failure || failed_cells.empty())
            retry_limiter_requires_all_active = true;
        else
            for(std::size_t const failed_cell : failed_cells) {
                auto const inserted = cell_retry_fractions.emplace(
                    failed_cell, candidate_fraction);
                if(!inserted.second)
                    inserted.first->second = std::min(
                        inserted.first->second, candidate_fraction);
            }
        double const next_completed_fraction =
            completed_fraction + candidate_fraction;
        double const accepted_time = context.previous_event_time +
            completed_fraction *
            (context.event_time - context.previous_event_time);
        double const next_candidate_end_time = context.previous_event_time +
            next_completed_fraction *
            (context.event_time - context.previous_event_time);
        if(!(candidate_fraction > 0) ||
           !(next_completed_fraction > completed_fraction) ||
           !(next_candidate_end_time > accepted_time)) {
            UniversalError error(
                "individual radiation candidate cannot advance after halving");
            error.addEntry("rejected candidates", rejected_candidates);
            error.addEntry("completed fraction", completed_fraction);
            error.addEntry("candidate fraction", candidate_fraction);
            error.addEntry("previous event time", context.previous_event_time);
            error.addEntry("event time", context.event_time);
            error.addEntry("accepted time", accepted_time);
            error.addEntry("next candidate end time", next_candidate_end_time);
            if(!representative_reason.empty())
                error.addEntry("representative failure", representative_reason);
            if(representative_cell != std::numeric_limits<std::size_t>::max())
                error.addEntry("representative cell ID", representative_cell);
            throw error;
        }
    }

    matrix_builder.EndIndividualCanonicalScope();
    if(forced_rejection_test && rank == 0)
        std::cout << "INDIVIDUAL_TEST_FORCED_RADIATION_REJECTIONS event_tick=" << context.event_tick
                  << " forced=" << forced_test_rejections << " genuine_rejections=" << rejected_candidates
                  << std::endl;

    std::vector<double> local_suggested_dt(
        tess.GetPointNo(), std::numeric_limits<double>::max());
    matrix_builder.calculateIndividualTimeSteps(
        local_context, tess, local_cells, local_suggested_dt,
        use_identity_owned_move ? &all_cells : &cells,
        &mesh_view.localToGlobalMapping());
    // Gray relative-change limits just below the anchor interval: the next
    // hydro interval may reach the anchor bin and the limit is kept in the
    // cell's state for its next activation (above); every other active cell's
    // pending limit is consumed.  Only the gray estimator (MG adds its force
    // limit to the same vector) and without the optional increment limiter,
    // which the same vector also carries; below half the anchor unchanged.
    if(anchor_subcycle)
    {
        double const anchor = context.anchor_time_step;
        char const* const increment = std::getenv("RICH_INDIVIDUAL_RADIATION_INCREMENT_LIMIT");
        bool const increment_limiter = increment != nullptr && increment[0] != '\0' &&
            std::strtod(increment, nullptr) > 0;
        bool const relax = anchor > 0 && !increment_limiter &&
            dynamic_cast<Diffusion const*>(&matrix_builder) != nullptr;
        for(std::size_t local : local_context.active_indices)
        {
            double& pending = context.radiation_accuracy_limits.at(mesh_view.localToGlobal(local));
            pending = 0;
            double& limit = local_suggested_dt.at(local);
            if(relax && limit >= anchorSubcycleBandLow() * anchor && limit < anchor)
            {
                pending = limit;
                limit = anchor;
            }
        }
    }
    // With the switch off, a rejected candidate has already been absorbed by
    // sub-cycling above; the physical limiter (calculateIndividualTimeSteps)
    // alone shapes the next bin, and the retry cooldown remembered in the
    // defect accounting keeps the next event's radiation from re-probing the
    // rejected interval at once.
    if(radiationRetryLimitsBins()) {
        if(retry_limiter_requires_all_active)
            for(std::size_t local : local_context.active_indices)
                local_suggested_dt.at(local) = std::min(
                    local_suggested_dt.at(local),
                    event_minimum_candidate_fraction *
                        local_context.cellTimeStep(local));
        else if(!cell_retry_fractions.empty())
            for(std::size_t local : local_context.active_indices) {
                auto const retry = cell_retry_fractions.find(
                    local_cells.at(local).ID);
                if(retry != cell_retry_fractions.end())
                    local_suggested_dt.at(local) = std::min(
                        local_suggested_dt.at(local),
                        retry->second * local_context.cellTimeStep(local));
            }
    }
    if(trace_performance)
        last_individual_performance["radiation-driver"] =
            elapsedSeconds(phase_start);
    phase_start = std::chrono::steady_clock::now();
    matrix_builder.poststepIndividual();
    cumulative_individual_rejected_candidates += rejected_candidates;
    // Sparse path: the values all_cells/all_extensives would hold here are the canonical state at the last ghost
    // sync, with the targets as synchronized there.  Since that sync only the final accepted candidate (its write
    // journal) and the scatter below change canonical slots, so record those slots before the scatter; every other
    // slot keeps that state.
    std::size_t const NoSyncViewSlot = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> SyncViewSlot;
    std::vector<ComputationalCell3D> SyncViewCells;
    std::vector<Conserved3D> SyncViewExtensives;
    if(SparseCanonical) {
        SyncViewSlot.assign(canonical_owned_size, NoSyncViewSlot);
        auto const AddSyncViewSlot = [&](std::size_t const Global, ComputationalCell3D const& Cell,
                                         Conserved3D const& Extensive) {
            SyncViewSlot[Global] = SyncViewCells.size();
            SyncViewCells.push_back(Cell);
            SyncViewExtensives.push_back(Extensive);
        };
        std::vector<std::size_t> const& Written = matrix_builder.GetIndividualCanonicalWriteIndices();
        std::vector<Conserved3D> const& WrittenBefore = matrix_builder.GetIndividualCanonicalWriteBefore();
        for(std::size_t Entry = 0; Entry < Written.size() && Entry < WrittenBefore.size(); ++Entry)
            if(Written[Entry] < canonical_owned_size && SyncViewSlot[Written[Entry]] == NoSyncViewSlot)
                AddSyncViewSlot(Written[Entry], cells[Written[Entry]], WrittenBefore[Entry]);
        for(std::size_t Target = 0; Target < TargetCount; ++Target) {
            std::size_t const Global = mesh_view.localToGlobal(Target);
            std::size_t const Slot = SyncViewSlot[Global];
            if(TargetsSynced && Slot != NoSyncViewSlot) {
                SyncViewCells[Slot] = TargetCellsAtSync[Target];
                SyncViewExtensives[Slot] = TargetExtensivesAtSync[Target];
            }
            else if(TargetsSynced)
                AddSyncViewSlot(Global, TargetCellsAtSync[Target], TargetExtensivesAtSync[Target]);
            else if(Slot == NoSyncViewSlot)
                AddSyncViewSlot(Global, cells[Global], extensives[Global]);
        }
        for(std::size_t Local = TargetCount; Local < mesh_view.meshLocalSize(); ++Local) {
            std::size_t const Global = mesh_view.meshLocalToGlobal(Local);
            if(Global != ActiveMeshView::invalidIndex() && !mesh_view.containsGlobal(Global) &&
               SyncViewSlot[Global] == NoSyncViewSlot)
                AddSyncViewSlot(Global, cells[Global], extensives[Global]);
        }
        matrix_builder.EndIndividualCanonicalWriteJournal();
    }
    if(use_identity_owned_move) {
        cells_move_guard.commit();
        extensives_move_guard.commit();
    }
    else {
        mesh_view.scatterMeshOwned(local_cells, cells);
        mesh_view.scatterMeshOwned(local_extensives, extensives);
    }
    suggested_individual_dt.assign(cells.size(), std::numeric_limits<double>::max());
    for(std::size_t local = 0; local < mesh_view.localSize(); ++local)
        suggested_individual_dt[mesh_view.localToGlobal(local)] =
            local_suggested_dt[local];
    if(trace_performance)
        last_individual_performance["radiation-commit-scatter"] =
            elapsedSeconds(phase_start);
    phase_start = std::chrono::steady_clock::now();

    // Diffusion may conservatively update an inactive recipient without
    // refreshing its primitive state.  Frozen Dirichlet may instead request
    // synchronization while leaving that recipient unchanged.  In both cases
    // use the shortest accepted active-passive face timestep; the unrelated
    // global event spacing is not the transfer integration interval.
    // The pre-transfer state: all_cells/all_extensives, or their sparse rebuild.
    std::size_t const SyncViewCellCount = SparseCanonical ? canonical_owned_size : all_cells.size();
    std::size_t const SyncViewExtensiveCount = SparseCanonical ? canonical_owned_size : all_extensives.size();
    auto const SyncViewCell = [&](std::size_t const Global) -> ComputationalCell3D const& {
        if(!SparseCanonical)
            return all_cells[Global];
        std::size_t const Slot = SyncViewSlot[Global];
        return Slot == NoSyncViewSlot ? cells[Global] : SyncViewCells[Slot];
    };
    auto const SyncViewExtensive = [&](std::size_t const Global) -> Conserved3D const& {
        if(!SparseCanonical)
            return all_extensives[Global];
        std::size_t const Slot = SyncViewSlot[Global];
        return Slot == NoSyncViewSlot ? extensives[Global] : SyncViewExtensives[Slot];
    };
    if(SparseCanonicalCheck) {
        std::size_t Mismatches = 0;
        std::size_t FirstMismatch = NoSyncViewSlot;
        bool const SizesMatch = CheckCells.size() == SyncViewCellCount &&
            CheckExtensives.size() == SyncViewExtensiveCount;
        for(std::size_t Global = 0; SizesMatch && Global < SyncViewCellCount; ++Global)
            if(!SameWakeInputs(SyncViewCell(Global), SyncViewExtensive(Global), CheckCells[Global],
                               CheckExtensives[Global])) {
                if(Mismatches == 0)
                    FirstMismatch = Global;
                ++Mismatches;
            }
        if(!SizesMatch || Mismatches > 0) {
            std::cerr << "RICH_INDIVIDUAL_RADIATION_SPARSE_CHECK failed: rank=" << rank
                      << " sizes_match=" << (SizesMatch ? 1 : 0) << " mismatches=" << Mismatches
                      << " first_canonical_index=" << FirstMismatch
                      << " first_cell_id=" << (FirstMismatch < cells.size() ? cells[FirstMismatch].ID : 0)
                      << " in_view=" << (FirstMismatch < SyncViewSlot.size() &&
                                         SyncViewSlot[FirstMismatch] != NoSyncViewSlot ? 1 : 0)
                      << " targets_synced=" << (TargetsSynced ? 1 : 0) << std::endl;
            std::abort();
        }
        std::vector<ComputationalCell3D>().swap(CheckCells);
        std::vector<Conserved3D>().swap(CheckExtensives);
    }
    double maximum_radiation_energy_density =
        std::numeric_limits<double>::min();
    if(SyncViewCellCount == SyncViewExtensiveCount)
        for(std::size_t global = 0; global < SyncViewExtensiveCount; ++global) {
            ComputationalCell3D const& BeforeCell = SyncViewCell(global);
            Conserved3D const& BeforeExtensive = SyncViewExtensive(global);
            if(BeforeExtensive.mass > 0 &&
               std::isfinite(BeforeExtensive.mass) &&
               BeforeCell.density > 0 &&
               std::isfinite(BeforeCell.density))
                maximum_radiation_energy_density = std::max(
                    maximum_radiation_energy_density,
                    std::abs(BeforeExtensive.Erad) *
                        BeforeCell.density /
                        BeforeExtensive.mass);
        }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &maximum_radiation_energy_density, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    suggested_individual_wake_deadline.assign(
        cells.size(), std::numeric_limits<double>::infinity());
    std::vector<double> const& passive_reference_time_steps =
        matrix_builder.getIndividualPassiveReferenceTimeSteps();
    if(SyncViewCellCount == extensives.size() &&
       SyncViewExtensiveCount == extensives.size())
        for(std::size_t global = 0; global < extensives.size(); ++global) {
            if(context.isActive(global))
                continue;
            bool const have_transfer_dt =
                global < passive_reference_time_steps.size() &&
                passive_reference_time_steps[global] > 0 &&
                std::isfinite(passive_reference_time_steps[global]) &&
                passive_reference_time_steps[global] <
                    std::numeric_limits<double>::max();
            double difference = 0;
            // A sparse slot outside the view compares an extensive with itself, which the fraction reports as
            // unchanged (every difference is 0 or NaN), so skip the evaluation.
            bool const SameExtensive = SparseCanonical && SyncViewSlot[global] == NoSyncViewSlot;
            bool const passive_changed = !SameExtensive && passive_radiation_change_fraction(
                SyncViewCell(global), SyncViewExtensive(global),
                extensives[global], maximum_radiation_energy_density,
                difference);
            if(!passive_changed && !have_transfer_dt)
                continue;

            if(!have_transfer_dt) {
                // Missing accepted-face metadata must not turn the unrelated
                // scheduler event spacing into a physical transfer rate.
                // Wake at the earliest representable event instead.
                suggested_individual_wake_deadline[global] =
                    context.time_quantum;
                continue;
            }
            if(passive_changed)
                suggested_individual_wake_deadline[global] =
                    passive_radiation_wake_interval(
                        difference, passive_reference_time_steps[global],
                        context.time_quantum);
            else
                suggested_individual_wake_deadline[global] = std::max(
                    context.time_quantum,
                    passive_reference_time_steps[global]);
        }

    // Entry probes are rare (one per cooldown worth of accepted candidates) but
    // are the only record that the persisted ceiling recovered, so they get a
    // summary even when the event had no rejection at all.
    if((rejected_candidates > 0 || entry_retry_probes > 0) &&
       trace_performance && rank == 0) {
        std::ostream& trace_stream = RuntimeTraceStream();
        trace_stream << "Individual radiation retries: "
                     << rejected_candidates;
        if(!representative_reason.empty()) {
            trace_stream << ", representative reason: "
                         << representative_reason;
            if(representative_cell != std::numeric_limits<std::size_t>::max())
                trace_stream << ", cell ID " << representative_cell;
        }
        trace_stream << ", maximum solver iterations " << maximum_iterations
                     << ", next-step limiter scope "
                     << (retry_limiter_requires_all_active ?
                         "all_active" : "failed_cells")
                     << ", limited cells " << cell_retry_fractions.size()
                     << ", minimum fraction "
                     << event_minimum_candidate_fraction
                     << ", failed recovery probes " << rejected_retry_probes
                     << ", entry recovery probes " << entry_retry_probes
                     << ", persisted ceiling " << candidate_fraction_ceiling
                     << ", maximum probe cooldown "
                     << maximum_retry_probe_cooldown
                     << std::endl;
    }
    if(trace_performance)
        last_individual_performance["radiation-limit-update"] =
            elapsedSeconds(phase_start);
    if(defect_accounting != nullptr &&
       (entry_probe ||
        defect_accounting->accepted_dirichlet_candidates > 0 ||
        defect_accounting->defect_rejections > 0)) {
        defect_accounting->cooldown_accepted_candidates =
            retry_probe_backoff.successfulCandidates();
        defect_accounting->cooldown_required_candidates =
            retry_probe_backoff.cooldown();
        defect_accounting->cooldown_fraction_ceiling =
            candidate_fraction_ceiling;
        defect_accounting->defect_retry_substeps +=
            pending_defect_retry_substeps;
    }
    if(D2Timers)
        EndRadiationD2Event(context.event_tick, D2Seconds() - D2EventStart,
                            mpi_wait_profiler::Seconds() - D2EventMpiStart, D2Candidates, rejected_candidates,
                            local_context.active_indices.size());
}

void RadiationStep::suggestIndividualTimeSteps(
    IndividualStepContext const& context,
    std::vector<double>& time_step_limits) const
{
    for(std::size_t const cell : context.active_indices)
        if(cell < suggested_individual_dt.size() &&
           cell < time_step_limits.size() &&
           suggested_individual_dt[cell] <
               std::numeric_limits<double>::max())
            time_step_limits[cell] = std::min(
                time_step_limits[cell], suggested_individual_dt[cell]);
    for(std::size_t cell : context.active_indices)
        if(cell >= suggested_individual_dt.size())
            time_step_limits.at(cell) = std::min(
                time_step_limits.at(cell), suggested_dt);
}

void RadiationStep::suggestIndividualWakeDeadlines(
    IndividualStepContext const& context,
    std::vector<double>& wake_deadlines) const
{
    std::size_t const common_size = std::min(
        suggested_individual_wake_deadline.size(), wake_deadlines.size());
    for(std::size_t cell = 0; cell < common_size; ++cell)
        if(!context.isActive(cell) &&
           suggested_individual_wake_deadline[cell] <
               std::numeric_limits<double>::max())
            wake_deadlines[cell] = std::min(
                wake_deadlines[cell],
                suggested_individual_wake_deadline[cell]);
}


double RadiationStep::suggestTimeStep(void) const
{
    return this->suggested_dt;
}

bool RadiationStep::collectCellTimeStepLimits(std::vector<double>& limits) const
{
    cell_limit_fallbacks = 0;
    if(cell_limit_ids.empty())
        return false;
    std::size_t const N = this->tess.GetPointNo();
    if(this->cells.size() < N)
        return false;
    bool aligned = cell_limit_ids.size() == N;
    for(std::size_t i = 0; aligned && i < N; ++i)
        aligned = this->cells[i].ID == cell_limit_ids[i];
    if(aligned)
    {
        limits = cell_limit_values;
        return true;
    }
    std::unordered_map<std::size_t, double> by_id;
    by_id.reserve(cell_limit_ids.size());
    for(std::size_t i = 0; i < cell_limit_ids.size(); ++i)
        by_id.emplace(cell_limit_ids[i], cell_limit_values[i]);
    // A cell without a cached limit (migrated from another rank, or new)
    // sets none: the bound may overstate the gain, never understate it.
    limits.assign(N, std::numeric_limits<double>::infinity());
    for(std::size_t i = 0; i < N; ++i)
    {
        auto const found = by_id.find(this->cells[i].ID);
        if(found != by_id.end())
            limits[i] = found->second;
        else
            ++cell_limit_fallbacks;
    }
    return true;
}

std::string RadiationStep::getName(void) const
{
    return "radiation";
}
