#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <functional>
#include <cstdint>
#include <limits>
#include <array>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include "ProgressTracker.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include <MeshDecomposer3D/load_balancing/LoadBalancer.hpp>
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/simulation/steps/PhysicsStep.hpp"
#include "newtonian/three_dimensional/simulation/IndividualTimeStep.hpp"
#include "newtonian/three_dimensional/time_step_function3D.hpp"
#include "utils/debug/vtune.h"

#ifdef RICH_MPI
    #include <mpi.h>
    #include "mpi/mpi_commands.hpp"
    #include "mpi/mpi_commands.hpp"
    #include "mpi/ExchangeChain.hpp"
#endif // RICH_MPI

class Simulation
{
public:
    using IndividualAMRCallback =
        std::function<IndividualAMRChangeSet(IndividualStepContext const&)>;
    using IndividualPostPhysicsCallback =
        std::function<void(IndividualStepContext const&)>;

    Simulation(Tessellation3D &tess, const std::vector<ComputationalCell3D> &cells, EquationOfState &eos, bool new_start = true);

    inline ProgressTracker &getTracker(void){return this->tracker;};

    size_t& GetMaxID(void);
    const size_t& GetMaxID(void) const;

    void initializeCellIDs(void);
    void recomputeMaxID(void);

    inline void SetTimeStepFunction(std::shared_ptr<TimeStepFunction3D> tsc){this->tsc = tsc;};

    inline Tessellation3D &getTessellation(void){return this->tess;};

    inline const Tessellation3D &getTessellation(void) const{return this->tess;};

    inline std::vector<ComputationalCell3D> &getCells(void){return this->cells;};

    inline std::vector<Conserved3D> &getExtensives(void){return this->extensives;};

    inline const std::vector<ComputationalCell3D> &getCells(void) const{return this->cells;};

    inline const std::vector<Conserved3D> &getExtensives(void) const{return this->extensives;};

    inline const std::vector<std::shared_ptr<PhysicsStep>> &getPhysicsSteps(void) const{return this->physics;};
    
    inline std::vector<std::shared_ptr<PhysicsStep>> &getPhysicsSteps(void){return this->physics;};

    inline const std::map<std::string, double> &getLastPhysicsTimes(void) const{return this->lastPhysicsTimes;};
    inline const std::map<std::string, double> &getLastLocalPhysicsTimes(void) const{return this->lastLocalPhysicsTimes;};

    void SetTimeStep(double dt);

    double GetTimeStep(void) const;

    void step(void);

    void EnableIndividualTimeSteps(
        IndividualTimeStepOptions options = IndividualTimeStepOptions());

    /** Request one all-active individual event.
     *
     * The request is process-local and clears only after the event commits.
     * Callers must not checkpoint while waiting for the requested event.
     */
    void RequestSynchronizedIndividualEvent(void);

    /** True when every owned primitive is committed at the current event tick
     * and the tessellation contains every owned cell. Collective under MPI.
     */
    bool IndividualStateSynchronized(void) const;

    /** True when the committed state can be checkpointed or switched: every
     * global step boundary, or a synchronized individual event. Collective
     * under MPI.
     */
    bool StateSynchronized(void) const;

    /** Committed generator position of every owned cell, in the order of
     * getCells().  Individual mode: the hydro step's canonical positions
     * (the tessellation's all-points array if no step keeps them); global
     * mode: the mesh points.  Collective under MPI: throws on every rank when
     * any rank cannot supply them.
     */
    std::vector<Vector3D> CommittedGeneratorPoints(void) const;

    /** Collective totals of one domain growth, see
     * GrowDomainAtSynchronizedIndividualState. */
    struct DomainGrowthReport
    {
        unsigned long long cells_before = 0;
        unsigned long long cells_after = 0;
        unsigned long long added_cells = 0;
        // New cells and cells whose volume changed (capped at seed_bin).
        unsigned long long reseeded_cells = 0;
        // Cells whose interval the rebuilt-mesh limits or seed_bin shortened.
        unsigned long long shortened_cells = 0;
        unsigned seed_bin = 0;
        double seconds = 0;  // MPI maximum
        MeshBuildTiming mesh_build_timing;  // rank 0
        double mass_before = 0;
        double mass_after = 0;
        double inserted_mass = 0;
        double energy_before = 0;
        double energy_after = 0;
        double inserted_energy = 0;
        // Smallest per-cell limit on the rebuilt mesh (collective minimum).
        double smallest_limit = 0;
        // Whether the acceleration cache was refreshed on the rebuilt state
        // (otherwise invalidated).
        bool accelerations_refreshed = false;
    };

    /** Grow the computational box at a synchronized individual state.
     *
     * The individual counterpart of the legacy global UpdateBox resize, with
     * the same semantics: the box becomes [ll, ur], `added_points` with the
     * primitive states `added_cells` (fresh IDs, possibly on one rank only)
     * join the owned cells, the mesh is rebuilt with rebalancing, and every
     * extensive is recomputed from its primitive on the new mesh.  The
     * individual state follows by stable ID: primitives, extensives and
     * scheduler states migrate together.  Before anything advances, every
     * cell's interval is capped by its limits evaluated on the rebuilt mesh
     * (PhysicsStep::synchronizedCellTimeStepLimits); new cells, and old cells
     * whose volume changed, also start no coarser than the finest bin in use.
     * The acceleration cache is refreshed on the rebuilt state (invalidated
     * when no step can refresh it), neighbour bins are closed,
     * topology caches are released, and the adaptive controller is told the
     * domain changed.  Collective; requires IndividualStateSynchronized().
     */
    DomainGrowthReport GrowDomainAtSynchronizedIndividualState(
        Vector3D const& ll, Vector3D const& ur,
        std::vector<Vector3D> const& added_points,
        std::vector<ComputationalCell3D> const& added_cells);

    /** Tell the adaptive controller that the domain changed (a box growth in
     * either mode): throughput measured before the change no longer describes
     * the workload.  The current window restarts, and a probe that ends
     * against a baseline measured before the change returns to the baseline
     * mode to re-measure it instead of comparing.  Collective in effect: call
     * it on every rank.
     */
    void NotifyDomainChanged(void);

    /** Let the simulation choose between global and individual integration.
     *
     * The controller measures the throughput (simulated time per wall
     * second) of the current mode, probes the other mode with a bounded
     * wall-time budget, and keeps the faster one with a margin; while
     * stepping globally it also bounds the possible gain of individual
     * timesteps from the per-cell CFL distribution and skips probes when
     * that bound is small.  Switches happen only at synchronized states.
     * `options` seeds the scheduler whenever individual mode is entered;
     * its time quantum is derived from the global step at that moment.
     * `RICH_INDIVIDUAL_ADAPTIVE_MODE` overrides `enabled` when set.
     */
    void SetAdaptiveIntegrationMode(bool enabled,
                                    IndividualTimeStepOptions options);

    bool AdaptiveIntegrationModeEnabled(void) const
    {return this->adaptiveMode.enabled;};

    /** Whether SetAdaptiveIntegrationMode(requested, ...) will enable the
     * controller: `RICH_INDIVIDUAL_ADAPTIVE_MODE` overrides `requested` when
     * set.  Collective on its first use (it parses the controller options).
     */
    static bool AdaptiveIntegrationModeWillEnable(bool requested);

    /** While set, the adaptive controller postpones every decision it would
     * take on a global step, so it cannot enter individual mode while the
     * driver has a target the global step sequence must land on: a fresh
     * scheduler's quantum and first event know nothing of it.  Set it to the
     * same value on every rank before each step.
     */
    void SetAdaptiveDecisionsDeferred(bool deferred)
    {this->adaptiveDecisionsDeferred = deferred;};

    /** Run a collective, run-specific update after every global step, once
     * the step is committed and before the adaptive controller acts: the
     * global-path counterpart of SetIndividualAMR/SetIndividualPostPhysics.
     * Its time counts toward the step.  Register it on every rank or none.
     */
    void SetGlobalPostStep(std::function<void()> callback)
    {this->globalPostStep = std::move(callback);};

    void SetIndividualAMR(IndividualAMRCallback callback)
    {this->individualAMR = std::move(callback);};

    /** Run a collective, run-specific state update after all individual
     * physics steps and before the next timestep limits are evaluated.
    */
    void SetIndividualPostPhysics(IndividualPostPhysicsCallback callback)
    {this->individualPostPhysics = std::move(callback);};

    TimeIntegrationMode GetTimeIntegrationMode(void) const{return this->timeIntegrationMode;};

    const IndividualTimeStepScheduler *GetIndividualTimeStepScheduler(void) const
    {return this->individualScheduler.get();};

    IndividualTimeStepScheduler *GetIndividualTimeStepScheduler(void)
    {return this->individualScheduler.get();};

    bool IndividualEventInProgress(void) const
    {return this->individualEventInProgress;};

    void addPhysics(std::shared_ptr<PhysicsStep> physics);

    double GetTime(void) const;

    size_t GetCycle(void) const;

    void SetCycle(size_t cycle);

    void SetTime(double time);

    double GetWallclockTime(void) const;

    void SetWallclockTime(double t);

    /** Collect and print one actual AMR topology change across all MPI ranks. */
    void ReportRuntimeAMREvent(
        std::string const& mode, std::size_t cycle, double time,
        std::uint64_t local_cells_before, std::uint64_t local_added_cells,
        std::uint64_t local_removed_cells,
        std::uint64_t local_cells_after) const;

    #ifdef RICH_MPI
        void buildDataTransfer(void);

        void buildDataTransfer(const ExchangeChain &chain);

        /**
         * Adds a buffer that should be transferred between mesh movements
         */
        template<typename T>
        void addMigrationBuffer(std::vector<T> &buffer);

        void storeLoadBalance(const std::string &name, std::shared_ptr<LoadBalancer<Vector3D>> lb);

        void setCurrentLoadBalance(const std::string &name);

        void PresetLoadBalance(const std::string &name);

        std::vector<std::pair<std::string, std::shared_ptr<LoadBalancer<Vector3D>>>> GetLoads(void) const;

        inline const std::string &getCurrentLB() const{return this->currentLB;};

        inline void setForceRebalanceSteps(size_t n){this->forceRebalanceSteps = n;};
    #endif // RICH_MPI

private:
    // The scheduler unit test drives the adaptive controller with synthetic
    // step walls and advances.
    friend struct AdaptiveControllerTestAccess;
    void stepIndividual(void);

    // Adaptive integration mode: measured throughput per mode, probe and
    // dwell bookkeeping, and the potential-gain bound.  Every value is
    // derived from collectively reduced quantities so all ranks decide alike.
    double individualEntryReference(void) const;
    double individualAnchorReference(void) const;

    struct AdaptiveModeState
    {
        bool enabled = false;
        IndividualTimeStepOptions individualOptions;
        bool probing = false;
        bool switchToGlobalRequested = false;
        std::string pendingReason;
        std::size_t stepsInMode = 0;
        double wallTotalInMode = 0;
        double wallMeasured = 0;
        double simMeasured = 0;
        double tauIndividual = std::numeric_limits<double>::quiet_NaN();
        double tauGlobal = std::numeric_limits<double>::quiet_NaN();
        // NotifyDomainChanged count, and the count each tau was measured at.
        std::uint64_t domainEpoch = 0;
        std::uint64_t tauIndividualEpoch = 0;
        std::uint64_t tauGlobalEpoch = 0;
        std::size_t dwellMultiplier = 1;
        double probeWallBudget = 0;
        double gainBound = std::numeric_limits<double>::quiet_NaN();
        // The bound with every step but the first per-cell one as a uniform
        // cap (radiation's grid-wide limit), logged beside gainBound.
        double gainBoundUniformCaps = std::numeric_limits<double>::quiet_NaN();
        // The latest global step's CFL/source suggestion (MPI minimum), taken
        // before its post-step callback, the reference of
        // RICH_INDIVIDUAL_BIN_ANCHOR_MARGIN; 0 before any or when unavailable.
        // The same on every rank.
        double anchorReferenceStep = 0;
        // An individual probe was judged faster and adopted this run
        // (RICH_ADAPTIVE_STAY_INDIVIDUAL keeps individual mode from then on).
        bool individualAdopted = false;
        // A due gain bound was skipped (stale per-cell limits): the next
        // global step retries it.  Replicated, as is the count of
        // consecutive skips.
        bool gainRetry = false;
        std::size_t gainSkipStreak = 0;
        std::size_t decisions = 0;
        std::size_t switches = 0;
    };
    AdaptiveModeState adaptiveMode;
    bool adaptiveDecisionsDeferred = false;
    std::function<void()> globalPostStep;
    double lastStepAdvance = 0;
    double lastStepSecondsMax = 0;
    void adaptiveAfterStep(TimeIntegrationMode mode);
    // After an individual AMR pass: refresh (or drop) the acceleration cache
    // of every cell whose interval opens at the current tick.  Collective.
    void refreshCurrentAccelerationCaches(void);
    void adaptiveResetWindow(void);
    void adaptiveRequestSwitch(TimeIntegrationMode target,
                               std::string const& reason);
    void adaptiveEnterGlobal(std::string const& reason);
    // S7 continuity record: owned-cell conserved sums around a mode switch.
    void adaptiveLogSwitchState(char const* phase, char const* direction,
                                std::size_t owned, double next_dt,
                                double scheduler_next_dt) const;
    void adaptiveEnterIndividual(std::string const& reason);
    double adaptiveGainBound(void);
    void adaptiveLogDecision(TimeIntegrationMode mode, char const* action,
                             double tau) const;

#ifdef RICH_MPI
    struct IndividualRebalanceResult
    {
        bool applied = false;
        double weightSkew = 1;
        unsigned long long migratedCells = 0;
        unsigned long long totalOwnedCells = 0;
        double localSeconds = 0;
        double maximumSeconds = 0;
        double currentRssBeforeKiB = -1;
        double currentRssAfterKiB = -1;
    };

    std::shared_ptr<PhysicsStep> findIndividualBalanceStep(void) const;

    void setCurrentLoadBalance(const std::string& name,
                               MeshBuildTiming& meshBuildTiming);

    IndividualRebalanceResult rebalanceCommittedIndividualState(
        std::shared_ptr<PhysicsStep> const& balanceStep,
        bool forceRebalance, double threshold,
        std::vector<double> explicitWeights = std::vector<double>(),
        std::shared_ptr<LoadBalancer<Vector3D>> exactLoadBalance = nullptr);

    // Collective: migrates segmented Hilbert ownership back to one
    // equal-count curve range per rank (an exact partition; a weighted
    // Rebalance would keep a segmented partition whose weights balance).
    IndividualRebalanceResult rebalanceToPositionalOwnership(
        std::shared_ptr<PhysicsStep> const& balanceStep);

    bool individualActiveHilbertBalanceEnabled(void) const;

    void maybeRebalanceBeforeFirstIndividualEvent(
        bool activeHilbertBalance);

    void reportPhaseBusyTimes(std::string const& mode, std::size_t cycle,
        double step_seconds, double step_mpi_seconds);
    void maybeBalanceIndividualEventByActiveBins(
        IndividualStepContext& context, std::uint64_t eventTick);
#endif

    int rank, size;
    Tessellation3D &tess;
    std::vector<std::shared_ptr<PhysicsStep>> physics;
    std::vector<ComputationalCell3D> cells;
    std::vector<Conserved3D> extensives;
    ProgressTracker tracker;
    EquationOfState &eos;
    size_t Max_ID;
    double wallclockTime;
    bool initializedFromRestart;
    std::shared_ptr<TimeStepFunction3D> tsc; // todo: why?
    TimeIntegrationMode timeIntegrationMode = TimeIntegrationMode::Global;
    std::unique_ptr<IndividualTimeStepScheduler> individualScheduler;
    IndividualAMRCallback individualAMR;
    IndividualPostPhysicsCallback individualPostPhysics;
    bool individualEventInProgress = false;
    bool individualSynchronizedEventRequested = false;
    std::map<std::string, double> lastPhysicsTimes;
    std::map<std::string, double> lastLocalPhysicsTimes;
    // Per-rank seconds inside MPI calls for the phases in lastLocalPhysicsTimes
    // that are timed for busy-time reports (RICH_MPI_WAIT_PROFILE).
    std::map<std::string, double> lastLocalPhysicsMpiTimes;

#ifdef RICH_MPI
    std::shared_ptr<LoadBalancer<Vector3D>> currentLoad;

    struct MigrationBuffer
    {
        std::any ref;
        std::function<void(void)> transfer;
        std::function<void(const ExchangeChain &chain)> transferChain;
    };

    std::vector<MigrationBuffer> migrationBuffers; // buffers that need to be moved after each call to 'BuildParallel'

    std::string currentLB;
    std::map<std::string, std::shared_ptr<LoadBalancer<Vector3D>>> loads;
    size_t forceRebalanceSteps = 0;
    std::pair<Vector3D, Vector3D> currentBox;
    size_t lastRebalanceCycle = std::numeric_limits<size_t>::max();
    size_t lastIndividualBalanceCheckCycle =
        std::numeric_limits<size_t>::max();
    std::uint64_t individualOwnershipEpoch = 0;
    double lastIndividualRebalanceSeconds = 0;
    MeshBuildTiming individualMeshBuildTiming;
    bool preFirstIndividualRebalanceChecked = false;
    // Cache only Hilbert cut coordinates.  Current geometry supplies the
    // convertor and indexing whenever a cached plan is validated or applied.
    std::map<std::uint64_t, std::vector<std::size_t>>
        individualActiveHilbertBoundaries;
    // Active-only rebuilds skipped because the passive-run bound proved their
    // rejection, and rebuilds actually built (diagnostics, process lifetime).
    std::uint64_t individualActiveHilbertSkippedRebuilds = 0;
    std::uint64_t individualActiveHilbertBuiltRebuilds = 0;
    // Wall seconds (max over ranks) spent planning and migrating segmented
    // Hilbert ownership (reported in RICH_ACTIVE_HILBERT_SEGMENT_DECISION).
    double individualSegmentMigrationSeconds = 0;
    // This rank's last individual event wall seconds less its AMR pass and
    // active-Hilbert decision (the segmented-ownership ledger's measure).
    double lastIndividualEventLedgerLocalSeconds = 0;
    // Segmented-ownership decision state (maybeBalanceIndividualEventByActiveBins,
    // RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENTS > 1); every field is derived
    // from reduced quantities, so it is identical on every rank.  Events are
    // compared within classes of their active fraction (see
    // IndividualSegmentEventClass), so a partition is judged against
    // positional events of the same kind.
    struct IndividualSegmentDecisionState
    {
        static constexpr int classes = 5;
        // Positional reference: event-mesh seconds per class.
        std::array<double, classes> positional_sum{};
        std::array<std::uint64_t, classes> positional_count{};
        // Class of the event whose mesh the next decision reads (-1: none).
        int pending_class = -1;
        // Events of the current segmented probe (since leaving positional
        // ownership; internal re-plans keep it), and since its last migration.
        std::uint64_t probe_events = 0;
        std::uint64_t events_since_migration = 0;
        // Probe events in the ledger classes, and those credited against a
        // same-class positional reference.
        std::uint64_t measured_events = 0;
        std::uint64_t credited_events = 0;
        // Wall seconds of those events: the evidence must cover most of the
        // probe's cost, not only its count of (possibly cheap) events.
        double measured_seconds = 0;
        double credited_seconds = 0;
        // Probe seconds per ledger class.
        std::array<double, classes> class_seconds{};
        double baseline_mesh = 0;
        bool baseline_pending = true;
        double debt = 0;
        double last_migration_seconds = 0;
        // Savings against the positional reference minus every migration and
        // planning cost of the period, charged once.
        double net_benefit = 0;
        int negative_windows = 0;
        std::uint64_t window_events = 0;
        std::uint64_t window_credited = 0;
        double window_measured_seconds = 0;
        double window_credited_seconds = 0;
        double window_saving = 0;
        std::uint64_t cooldown_until_event = 0;
        std::uint64_t event_counter = 0;
        // Reverts so far: each doubles the next cooldown.
        int reverts = 0;
        // Classes a measuring revert returned to positional ownership for
        // (bit per class; the positional stretch collects their references),
        // and the event from which re-adoption no longer waits for them.
        unsigned measure_mask = 0;
        // Aging of the current segmented plan: per class, the mean ledger
        // seconds of its first events after a migration, and the excess of
        // later same-class events over it (re-plans when it has paid for a
        // migration).
        std::array<double, classes> age_base_sum{};
        std::array<std::uint64_t, classes> age_base_count{};
        double age_debt = 0;
        std::uint64_t measure_deadline_event = 0;
        // Planning seconds of a proposal awaiting its migration.
        double pending_planning_seconds = 0;
    } individualSegmentDecision;
    // Cleared (collectively) if a built rebuild ever contradicts the bound;
    // no rebuild is skipped after that.
    bool individualActiveHilbertBoundTrusted = true;
#endif // RICH_MPI
};

#ifdef RICH_MPI
    template<typename T>
    void Simulation::addMigrationBuffer(std::vector<T> &buffer)
    {
        MigrationBuffer buff;
        buff.ref = std::ref(buffer);
        buff.transfer = [&buffer, this](void){MPI_exchange_data<T>(tess, buffer, false);};
        buff.transferChain = [&buffer, this](const ExchangeChain &chain){MPI_exchange_data<T>(chain, buffer);};
        this->migrationBuffers.push_back(buff);
    }
#endif // RICH_MPI

#endif // SIMULATION_HPP
