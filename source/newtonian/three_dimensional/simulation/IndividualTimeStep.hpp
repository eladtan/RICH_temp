#ifndef INDIVIDUAL_TIME_STEP_HPP
#define INDIVIDUAL_TIME_STEP_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <unordered_map>
#include <vector>

#include "3D/elementary/Vector3D.hpp"
#include "newtonian/three_dimensional/simulation/StepDiagnostics.hpp"
#ifdef RICH_MPI
#include <mpi_utils/serialize/Serializer.hpp>
#endif

class ComputationalCell3D;
class Tessellation3D;

enum class TimeIntegrationMode
{
    Global,
    Individual
};

enum class IndividualMeshBuildPolicy
{
    FullReference,
    AutoPartial
};

struct IndividualTimeStepOptions
{
    // A non-positive value derives the quantum from the initial global step.
    double time_quantum = 0;
    std::uint8_t initial_bin = 30;
    std::uint8_t maximum_bin = 40;
    std::uint8_t maximum_neighbor_bin_difference = 1;
    // Period between global all-source wake checks, measured in widths of the
    // smallest currently occupied time bin.
    std::uint64_t full_source_sweep_interval_minimum_steps = 128;
    IndividualMeshBuildPolicy mesh_build_policy = IndividualMeshBuildPolicy::AutoPartial;
    double partial_build_fraction = 0.5;
    bool verify_partial_build = false;
    // Keep every owned cell on one adaptively selected shared bin.
    bool force_synchronized = false;
};

struct CellTimeState
#ifdef RICH_MPI
    : public Serializable
#endif
{
    std::size_t cell_id = 0;
    std::uint64_t begin_tick = 0;
    std::uint64_t end_tick = 0;
    std::uint64_t last_primitive_tick = 0;
    std::uint8_t time_bin = 0;
    std::uint8_t pending_neighbor_bin =
        std::numeric_limits<std::uint8_t>::max();
    Vector3D point_velocity;
    Vector3D cached_acceleration;
    bool gravity_half_kick_pending = false;
    // Conserved-change wake requested at commitEvent and applied by
    // finalizeChangeWakes at the end of the same event.  Both fields are
    // clear between events; they migrate with the state because AMR and the
    // load balance run in between.
    std::uint8_t change_wake_pending = 0;
    double change_wake_ratio = 0;
    // Gray radiation accuracy limit (seconds of code time, 0 = none) that the
    // cell's current interval exceeds by less than 2x: the radiation step
    // subcycles to it at the interval's end (RICH_INDIVIDUAL_RADIATION_ANCHOR_SUBCYCLE).
    double radiation_accuracy_limit = 0;

#ifdef RICH_MPI
    std::size_t dump(Serializer* serializer) const override;
    std::size_t load(const Serializer* serializer,
                     std::size_t byte_offset) override;
#endif
};

struct IndividualAMRChangeSet
{
    struct MergeTarget
    {
        std::size_t removed_cell_id = 0;
        std::size_t recipient_cell_id = 0;
        int source_rank = 0;
    };

    // Each entry is (new child stable ID, parent stable ID).
    std::vector<std::pair<std::size_t, std::size_t>> child_parent_ids;
    std::vector<std::size_t> removed_cell_ids;
    // Actual conservative-remap recipients for each removed cell.
    std::vector<MergeTarget> merge_targets;
    // Rebuild work performed while producing this change set.
    MeshBuildTiming mesh_build_timing;

    bool empty() const
    {
        return child_parent_ids.empty() && removed_cell_ids.empty() &&
            merge_targets.empty();
    }
};

struct RadiationRepairAccounting
{
    std::uint64_t repaired_cells = 0;
    std::uint64_t repaired_groups = 0;
    double cumulative_injected_energy = 0;
    double maximum_relative_deficit = 0;
    std::uint64_t representative_cell_id =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t representative_group =
        std::numeric_limits<std::uint64_t>::max();
    int representative_rank = -1;
    double representative_original_extent = 0;
    double representative_floor_extent = 0;
    double representative_injected_extent = 0;
    double maximum_global_radiation_energy = 0;
    double next_warning_fraction = 1e-4;
    std::uint64_t residual_correction_limited_groups = 0;
    double residual_correction_signed_energy_bias = 0;
    double residual_correction_absolute_energy_bias = 0;
    std::vector<double> residual_correction_signed_bias_by_group;
    std::vector<double> residual_correction_absolute_bias_by_group;
    double residual_correction_minimum_scale = 1;
    std::uint64_t positivity_rescue_events = 0;
    std::uint64_t positivity_rescue_blocks = 0;
    std::uint64_t positivity_rescue_additional_iterations = 0;
    std::uint64_t residual_positive_floor_events = 0;
    std::uint64_t residual_positive_floor_cells = 0;
    std::uint64_t residual_positive_floor_groups = 0;
    double residual_positive_floor_cumulative_injected_energy = 0;
    double residual_positive_floor_maximum_cell_injection_ratio = 0;
    double residual_positive_floor_maximum_global_injection_ratio = 0;
    double residual_positive_floor_maximum_post_true_residual_error = 0;
    // First accepted radiation solve observed by this accounting epoch.
    double residual_positive_floor_initial_global_radiation_energy = 0;
};

// Advance the persistent 1e-4, 2e-4, 4e-4, ... warning schedule after an
// accepted transaction. Returns true exactly when this event crosses one or
// more thresholds.
bool AdvanceRadiationRepairWarning(
    RadiationRepairAccounting& accounting, double injection_fraction);

// Persistent accounting for the intentionally omitted passive-side transfer
// in frozen-Dirichlet individual radiation solves. Candidate code may update
// this record only after every collective acceptance check has passed; the
// rejection counters are the sole exception and describe rolled-back work.
struct IndividualRadiationDefectAccounting
{
    long double cumulative_signed_extent = 0;
    long double cumulative_absolute_extent = 0;
    double initial_positive_global_extent = 0;
    double last_normalization_scale = 0;
    double maximum_event_absolute_fraction = 0;
    double maximum_local_fraction = 0;
    double maximum_local_tolerance_ratio = 0;
    std::uint64_t accepted_dirichlet_candidates = 0;
    std::uint64_t defect_rejections = 0;
    std::uint64_t defect_retry_substeps = 0;

    // Versioned policy snapshot. Restart retains the limits under which the
    // cumulative history was accepted.
    std::uint64_t config_version = 3;
    double local_withdrawal_limit = 1e-2;
    double local_absolute_limit = 1e-9;
    double event_absolute_target = 1e-6;
    double cumulative_signed_limit = 1e-4;
    double cumulative_absolute_limit = 1e-3;

    // Persist retry/cooldown state even though normal checkpoints are emitted
    // only at accepted event boundaries.
    std::uint64_t cooldown_accepted_candidates = 0;
    std::uint64_t cooldown_required_candidates = 8;
    double cooldown_fraction_ceiling = 1;
    bool history_complete = true;
};

struct IndividualStepContext
{
    std::uint64_t previous_event_tick = 0;
    std::uint64_t event_tick = 0;
    double time_origin = 0;
    double previous_event_time = 0;
    double event_time = 0;
    double time_quantum = 0;
    // Interval of the anchor bin (initial_bin) on this timeline.
    double anchor_time_step = 0;
    IndividualMeshBuildPolicy mesh_build_policy = IndividualMeshBuildPolicy::AutoPartial;
    double partial_build_fraction = 0.5;
    bool verify_partial_build = false;
    std::vector<std::size_t> active_indices;
    std::vector<unsigned char> active_mask;
    // Actual interval applied by this event; it may be a short aligned
    // catch-up interval after a wake.
    std::vector<double> cell_time_steps;
    // Retained physical timestep bin; the actual event interval may be a
    // shorter aligned catch-up after a wake.
    std::vector<std::uint8_t> cell_time_bins;
    // Last completed hydro activation; scheduler begin_tick has the same value.
    // A face was last integrated when either endpoint last became active.
    std::vector<std::uint64_t> primitive_ticks;
    // Per canonical cell, CellTimeState::radiation_accuracy_limit: read by the
    // radiation step for its active cells and replaced for them (committed
    // back like the point velocities).
    mutable std::vector<double> radiation_accuracy_limits;
    mutable std::vector<Vector3D> point_velocities;
    mutable std::vector<Vector3D> cached_accelerations;
    mutable std::vector<unsigned char> gravity_half_kick_pending;
    // Canonical source arrays stay global even when this context is remapped
    // onto a partial mesh. Gravity providers may walk only the local targets.
    mutable std::vector<Vector3D> gravity_source_points;
    mutable std::vector<double> gravity_source_masses;
    mutable std::vector<std::uint64_t> gravity_source_ids;
    // Committed generator positions of every canonical cell at this event,
    // set by Simulation just before the individual AMR callback (empty
    // elsewhere).  AMR builds its mesh from these; gravity_source_points are
    // the cell centroids and must never serve as generators.
    std::vector<Vector3D> generator_points;
    // Shared scheduler-owned accounting. Candidate code updates it only after
    // every transactional radiation check has passed.
    RadiationRepairAccounting* radiation_repair_accounting = nullptr;
    IndividualRadiationDefectAccounting* radiation_defect_accounting = nullptr;

    bool isActive(std::size_t index) const;
    double cellTimeStep(std::size_t index) const;
    double nominalCellTimeStep(std::size_t index) const;
    double faceTimeStep(std::size_t left, std::size_t right) const;
    double hydroFaceTimeStep(std::size_t left, std::size_t right) const;
};

// Bit i is set when at least one active cell belongs to scheduler bin i.
// The scheduler limits bins to [0, 62], leaving the result portable in a
// uint64_t and suitable as a recurring-event cache key.
std::uint64_t IndividualActiveTimeBinMask(
    IndividualStepContext const& context,
    std::vector<CellTimeState> const& states);

class IndividualTimeStepScheduler
{
public:
    explicit IndividualTimeStepScheduler(IndividualTimeStepOptions options = IndividualTimeStepOptions());

    // anchor_time_step > 0 sets the bin grid (bin initial_bin lasts
    // anchor_time_step) instead of initial_time_step; every cell's first
    // interval is then the coarsest bin not longer than initial_time_step.
    void initialize(const std::vector<ComputationalCell3D> &cells,
                    double current_time,
                    double initial_time_step,
                    double anchor_time_step = 0);

    bool initialized(void) const { return initialized_; }
    const IndividualTimeStepOptions &options(void) const { return options_; }
    double timeQuantum(void) const { return time_quantum_; }
    double timeOrigin(void) const { return time_origin_; }
    std::uint64_t currentTick(void) const { return current_tick_; }
    // Diagnostics that must agree with the scheduler exactly: its own
    // saturating quantization of a limit into a bin, and a bin's length.
    std::uint8_t binForTimeStep(double time_step) const
    {
        return quantizeTimeStep(time_step);
    }
    std::uint64_t binTicks(std::uint8_t bin) const { return ticksForBin(bin); }
    std::uint64_t lastFullSourceSweepTick(void) const
    {return last_full_source_sweep_tick_;}
    const std::vector<CellTimeState> &states(void) const { return states_; }

    // First-interval generator velocities of a freshly initialized timeline,
    // by stable cell ID (cells not listed keep zero).  Only before the first
    // event (tick 0).
    void setInitialPointVelocities(std::vector<std::pair<std::size_t, Vector3D> > const& velocities);
    std::vector<CellTimeState> &states(void) { return states_; }
    bool forceAllActiveLatched(void) const
    {
        return force_all_active_latched_;
    }
    void setForceAllActiveLatched(bool const value)
    {
        force_all_active_latched_ = force_all_active_latched_ || value;
    }
    RadiationRepairAccounting const& radiationRepairAccounting(void) const
    {
        return radiation_repair_accounting_;
    }
    RadiationRepairAccounting& radiationRepairAccounting(void)
    {
        return radiation_repair_accounting_;
    }
    IndividualRadiationDefectAccounting const&
    radiationDefectAccounting(void) const
    {
        return radiation_defect_accounting_;
    }
    IndividualRadiationDefectAccounting& radiationDefectAccounting(void)
    {
        return radiation_defect_accounting_;
    }

    IndividualStepContext prepareEvent(const std::vector<ComputationalCell3D> &cells) const;

    /** Prepare a collective event selected outside this rank.
     *
     * `event_tick` may precede this rank's next local cell end tick.  In that
     * case the returned context has no active cells, but still advances the
     * rank's event clock so every MPI rank enters the same physics
     * collectives.
     */
    IndividualStepContext prepareEvent(
        const std::vector<ComputationalCell3D> &cells,
        std::uint64_t event_tick) const;

    /** Prepare a collective event, optionally activating every local cell.
     *
     * Forced activation is a transient context overlay: it does not modify
     * scheduler state.  A forced cell advances from its own `begin_tick` to
     * `event_tick`, which must lie strictly after that begin tick and no later
     * than its scheduled end tick.
     */
    IndividualStepContext prepareEvent(
        const std::vector<ComputationalCell3D> &cells,
        std::uint64_t event_tick,
        bool force_all_active) const;

    /** Commit physical timestep suggestions and optional passive wake intervals.
     *
     * Each signal wake entry is a duration measured from `context.event_tick`,
     * not an absolute or bin-aligned time. It can shorten `end_tick`, but it
     * does not replace the cell's physical `time_bin`.
     */
    void commitEvent(const IndividualStepContext &context,
                     const Tessellation3D &tess,
                     const std::vector<ComputationalCell3D> &cells,
                     const std::vector<double> &time_step_limits,
                     const std::vector<double> &signal_wake_deadlines =
                         std::vector<double>(),
                     const std::vector<double> &change_wake_ratios =
                         std::vector<double>());

    struct ChangeWakeFinalization
    {
        // Next scheduled event over all ranks, which the wakes leave unchanged.
        std::uint64_t next_tick_before = 0;
        std::uint64_t next_tick_after = 0;
        // Rank-local unless collect_statistics was set.
        std::uint64_t applied = 0;
        std::uint64_t shortened = 0;
        // Of applied, passive AMR merge recipients.  Their change is never
        // sampled: the AMR pass resets the hydro accumulators.
        std::uint64_t merge_wakes = 0;
        double largest_ratio = 0;
        // Bin lowerings since the last finalization, per cause (0 passive
        // limit at commit, 1 neighbour closure, 2 AMR merge, 3 bin-spread
        // cap): lowered cells, cells already past the new bin's allowance,
        // and the largest (event - begin) / new allowance at lowering.
        // Rank-local unless collect_statistics was set.
        std::uint64_t lowered[4] = {0, 0, 0, 0};
        std::uint64_t lowered_overdue[4] = {0, 0, 0, 0};
        double lowered_worst_ratio[4] = {0, 0, 0, 0};
    };
    // False under RICH_INDIVIDUAL_CHANGE_WAKE_RULE=spacing_ticks, the
    // reference rule (the old previous-spacing wake in exact ticks), which can
    // create events.
    static bool changeWakesJoinNextEvent(void);
    // RICH_INDIVIDUAL_MAX_BIN_SPREAD (default 2, -1 off).  Collective on first use.
    static int maximumBinSpread(void);
    /** Collective.  Ends every cell with a pending conserved-change wake (or
     * a passive AMR merge recipient) at the next scheduled event, the minimum
     * end tick over all cells and ranks, after every other endpoint of the
     * event has been set.  The deadline is an existing event, so the wakes
     * never create one and never depend on the spacing of past events.
     * Clears the requests.  collect_statistics adds the reductions for the
     * applied/shortened/largest_ratio totals. */
    ChangeWakeFinalization finalizeChangeWakes(bool collect_statistics = false);

    /** Shorten any interval crossing a terminal output tick.
     *
     * This is a one-event synchronization operation.  The resulting final
     * interval may be shorter than the cell's power-of-two bin, just like an
     * inactive-cell wake-up, and its exact duration is carried by the event
     * context.
     */
    std::size_t clampToTerminalTick(std::uint64_t terminal_tick);

    double nextEventTimeStep(void) const;
    std::uint64_t nextEventTick(void) const;
    void resetTimeOrigin(double current_time);
    void rebuildIndex(const std::vector<ComputationalCell3D> &cells);

    // Collective under MPI (rebuilds the finest occupied bin): every rank
    // calls it, including ranks with no cells.
    void restore(const std::vector<ComputationalCell3D>& cells,
                 double time_origin,
                 double time_quantum,
                 std::uint64_t current_tick,
                 std::vector<CellTimeState> states,
                 RadiationRepairAccounting repair_accounting =
                     RadiationRepairAccounting(),
                 IndividualRadiationDefectAccounting defect_accounting =
                     IndividualRadiationDefectAccounting(),
                 bool force_all_active_latched = false,
                 std::uint64_t last_full_source_sweep_tick = 0);

    void recordFullSourceSweep(std::uint64_t event_tick);

    void applyAMRChangeSet(const std::vector<ComputationalCell3D>& cells,
                           const IndividualAMRChangeSet& changes,
                           const Tessellation3D* tess = nullptr);

    void enforceNeighborBinClosure(
        const Tessellation3D& tess,
        const std::vector<ComputationalCell3D>& cells);

    // Smallest bin any rank holds (collective value, see minimum_occupied_bin_).
    std::uint8_t minimumOccupiedBin(void) const
    {return minimum_occupied_bin_;}

    /** State for a cell created at a synchronized state: its interval opens
     * at the current tick and carries `bin`, it has no cached acceleration
     * and no committed point velocity.  The caller places it beside the cell
     * and rebuilds the index.
     */
    CellTimeState synchronizedCellState(std::size_t cell_id,
                                        std::uint8_t bin) const;

    /** At a synchronized state (every interval opens at the current tick),
     * shorten each cell to at most its entry of `maximum_bins`; with
     * force_synchronized every cell takes the collective minimum of its
     * current bin and its entry.  Refreshes the finest occupied bin.  Returns
     * the number of cells shortened.  Collective under MPI; throws if the state
     * is not synchronized.
     */
    std::size_t limitSynchronizedBins(
        const std::vector<std::uint8_t>& maximum_bins);

    // Every cell evaluates a fresh acceleration before its next half kick.
    void invalidateCachedAccelerations(void);

private:
    std::uint64_t ticksForBin(std::uint8_t bin) const;
    std::uint64_t nextAlignedTick(std::uint64_t event_tick,
                                  std::uint8_t bin) const;
    // End tick for a cell whose bin is being lowered to `bin` part way through
    // an interval that opened at `begin_tick`.  Aligning to the bin grid alone
    // is not enough: when the next aligned tick is the end the cell already
    // had, the bin is relabelled while the interval keeps its old, longer
    // length.  An interval must never outlast the allowance of the bin it
    // carries, so the allowance bounds it too, whenever that bound is still
    // ahead of this event.
    std::uint64_t binnedEndTick(std::uint64_t begin_tick,
                                std::uint64_t event_tick,
                                std::uint8_t bin) const;
    // Smallest bin any rank currently holds, refreshed collectively once per
    // committed event.  Ticks aligned to it already carry events, so a cell
    // that must end sooner than its own grid allows can be sent there without
    // creating a new event.
    std::uint8_t minimum_occupied_bin_ = 0;
    // RICH_INDIVIDUAL_MAX_BIN_SPREAD (collective; no-op when off).
    void applyBinSpreadCap(std::uint64_t event_tick);
    // Passive merge recipients woken since the last finalizeChangeWakes.
    std::uint64_t merge_change_wakes_ = 0;
    // Spacing of the event being committed, for the reference wake rule.
    std::uint64_t committed_event_spacing_ = 0;
    // Bin-lowering diagnostics since the last finalizeChangeWakes (see ChangeWakeFinalization).
    std::uint64_t lowered_[4] = {0, 0, 0, 0};
    std::uint64_t lowered_overdue_[4] = {0, 0, 0, 0};
    double lowered_worst_ratio_[4] = {0, 0, 0, 0};
    void noteLowering(int cause, std::uint64_t begin_tick, std::uint64_t event_tick, std::uint8_t new_bin);
    // Collective: recompute minimum_occupied_bin_ from states_ on every rank.
    void refreshMinimumOccupiedBin(void);
    std::uint8_t quantizeTimeStep(double time_step) const;
    std::uint8_t chooseNextBin(const CellTimeState &state,
                               std::uint64_t event_tick,
                               double time_step_limit) const;
    void validateCells(const std::vector<ComputationalCell3D> &cells) const;
    void limitNeighborBins(const Tessellation3D &tess,
                           const std::vector<ComputationalCell3D> &cells,
                           std::uint64_t event_tick,
                           const std::vector<std::uint8_t>& propagation_bins,
                           bool full_closure);

    IndividualTimeStepOptions options_;
    bool initialized_ = false;
    double time_quantum_ = 0;
    double time_origin_ = 0;
    std::uint64_t current_tick_ = 0;
    std::uint64_t last_full_source_sweep_tick_ = 0;
    bool force_all_active_latched_ = false;
    bool needs_full_neighbor_closure_ = true;
    std::vector<CellTimeState> states_;
    std::unordered_map<std::size_t, std::size_t> id_to_index_;
    mutable RadiationRepairAccounting radiation_repair_accounting_;
    mutable IndividualRadiationDefectAccounting radiation_defect_accounting_;
};

#endif // INDIVIDUAL_TIME_STEP_HPP
