#ifndef INDIVIDUAL_TIME_STEP_HPP
#define INDIVIDUAL_TIME_STEP_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <unordered_map>
#include <vector>

#include "3D/elementary/Vector3D.hpp"
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
    std::uint8_t maximum_neighbor_bin_difference = 2;
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
    Vector3D point_velocity;
    Vector3D cached_acceleration;
    bool gravity_half_kick_pending = false;

#ifdef RICH_MPI
    std::size_t dump(Serializer* serializer) const override;
    std::size_t load(const Serializer* serializer,
                     std::size_t byte_offset) override;
#endif
};

struct IndividualAMRChangeSet
{
    // Each entry is (new child stable ID, parent stable ID).
    std::vector<std::pair<std::size_t, std::size_t>> child_parent_ids;
    std::vector<std::size_t> removed_cell_ids;

    bool empty() const
    {
        return child_parent_ids.empty() && removed_cell_ids.empty();
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
    std::uint64_t accepted_dirichlet_candidates = 0;
    std::uint64_t defect_rejections = 0;
    std::uint64_t defect_retry_substeps = 0;

    // Versioned policy snapshot. Restart retains the limits under which the
    // cumulative history was accepted.
    std::uint64_t config_version = 1;
    double local_withdrawal_limit = 1e-2;
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
    IndividualMeshBuildPolicy mesh_build_policy = IndividualMeshBuildPolicy::AutoPartial;
    double partial_build_fraction = 0.5;
    bool verify_partial_build = false;
    std::vector<std::size_t> active_indices;
    std::vector<unsigned char> active_mask;
    std::vector<double> cell_time_steps;
    std::vector<std::uint64_t> primitive_ticks;
    mutable std::vector<Vector3D> point_velocities;
    mutable std::vector<Vector3D> cached_accelerations;
    mutable std::vector<unsigned char> gravity_half_kick_pending;
    // Canonical source arrays stay global even when this context is remapped
    // onto a partial mesh. Gravity providers may walk only the local targets.
    mutable std::vector<Vector3D> gravity_source_points;
    mutable std::vector<double> gravity_source_masses;
    mutable std::vector<std::uint64_t> gravity_source_ids;
    // Shared scheduler-owned accounting. Candidate code updates it only after
    // every transactional radiation check has passed.
    RadiationRepairAccounting* radiation_repair_accounting = nullptr;
    IndividualRadiationDefectAccounting* radiation_defect_accounting = nullptr;

    bool isActive(std::size_t index) const;
    double cellTimeStep(std::size_t index) const;
    double faceTimeStep(std::size_t left, std::size_t right) const;
};

class IndividualTimeStepScheduler
{
public:
    explicit IndividualTimeStepScheduler(IndividualTimeStepOptions options = IndividualTimeStepOptions());

    void initialize(const std::vector<ComputationalCell3D> &cells,
                    double current_time,
                    double initial_time_step);

    bool initialized(void) const { return initialized_; }
    const IndividualTimeStepOptions &options(void) const { return options_; }
    double timeQuantum(void) const { return time_quantum_; }
    double timeOrigin(void) const { return time_origin_; }
    std::uint64_t currentTick(void) const { return current_tick_; }
    const std::vector<CellTimeState> &states(void) const { return states_; }
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

    void commitEvent(const IndividualStepContext &context,
                     const Tessellation3D &tess,
                     const std::vector<ComputationalCell3D> &cells,
                     const std::vector<double> &time_step_limits);

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

    void restore(const std::vector<ComputationalCell3D>& cells,
                 double time_origin,
                 double time_quantum,
                 std::uint64_t current_tick,
                 std::vector<CellTimeState> states,
                 RadiationRepairAccounting repair_accounting =
                     RadiationRepairAccounting(),
                 IndividualRadiationDefectAccounting defect_accounting =
                     IndividualRadiationDefectAccounting(),
                 bool force_all_active_latched = false);

    void applyAMRChangeSet(const std::vector<ComputationalCell3D>& cells,
                           const IndividualAMRChangeSet& changes,
                           const Tessellation3D* tess = nullptr);

private:
    std::uint64_t ticksForBin(std::uint8_t bin) const;
    std::uint64_t nextAlignedTick(std::uint64_t event_tick,
                                  std::uint8_t bin) const;
    std::uint8_t quantizeTimeStep(double time_step) const;
    std::uint8_t chooseNextBin(const CellTimeState &state,
                               std::uint64_t event_tick,
                               double time_step_limit) const;
    void validateCells(const std::vector<ComputationalCell3D> &cells) const;
    void limitNeighborBins(const Tessellation3D &tess,
                           const std::vector<ComputationalCell3D> &cells,
                           std::uint64_t event_tick);

    IndividualTimeStepOptions options_;
    bool initialized_ = false;
    double time_quantum_ = 0;
    double time_origin_ = 0;
    std::uint64_t current_tick_ = 0;
    bool force_all_active_latched_ = false;
    std::vector<CellTimeState> states_;
    std::unordered_map<std::size_t, std::size_t> id_to_index_;
    mutable RadiationRepairAccounting radiation_repair_accounting_;
    mutable IndividualRadiationDefectAccounting radiation_defect_accounting_;
};

#endif // INDIVIDUAL_TIME_STEP_HPP
