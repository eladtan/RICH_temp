#ifndef RADIATION_DRIVER_HPP
#define RADIATION_DRIVER_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "conj_grad_solve.hpp"
#include "newtonian/common/equation_of_state.hpp"
#include "newtonian/three_dimensional/simulation/IndividualTimeStep.hpp"
#include "boost/math/special_functions/pow.hpp"

class RadiationDriver : public CG::MatrixBuilder {
public:
    /**
 * @brief A base class for radiation drivers.
 * 
 * This class provides an interface for radiation drivers, which are used to handle radiation effects in simulations.
 * It inherits from CG::MatrixBuilder and contains common parameters and methods for radiation drivers.
 * 
 * @param eos The equation of state used to calculate thermodynamic properties.
 * @param zero_cells_ A vector of strings representing cells that should be treated as zero-radiation cells.
 * @param flux_limiter A boolean indicating whether to use a flux limiter in the radiation transport.
 * @param hydro_on A boolean indicating whether hydrodynamic effects are turned on.
 * @param compton_on A boolean indicating whether Compton scattering is turned on.
 */
    RadiationDriver(EquationOfState const& eos,
                    std::vector<std::string> const zero_cells_ = std::vector<std::string>(),
                    bool const flux_limiter = true,
                    bool const hydro_on = true,
                    bool const compton_on = false) : 
                                                        eos_(eos),
                                                        flux_limiter_(flux_limiter),
                                                        hydro_on_(hydro_on),
                                                        compton_on_(compton_on),
                                                        mass_scale_(1.0),
                                                        length_scale_(1.0),
                                                        time_scale_(1.0),
                                                        CG::MatrixBuilder(zero_cells_)
                                                        {}

/**
 * @brief A virtual destructor for the RadiationDriver class.
 * 
 * This destructor is declared as virtual to ensure proper cleanup of derived classes.
 */
        virtual ~RadiationDriver() = default;

/**
 * @brief A pure virtual function for performing pre-step operations.
 * 
 * This function should be implemented in derived classes to perform any necessary operations before each time step.
 * 
 * @param tess The current tessellation of the simulation domain.
 * @param cells The current state of computational cells.
 * 
 * @return A boolean indicating whether the pre-step operation was successful.
 */
        virtual bool prestep(Tessellation3D const& tess,
                             std::vector<ComputationalCell3D> const& cells) const = 0;

/**
 * @brief A pure virtual function for performing a time step.
 * 
 * This function should be implemented in derived classes to perform the main radiation transport and update the state of computational cells.
 * 
 * @param tolerance The tolerance for convergence in the iterative solver.
 * @param total_iters A reference to an integer that will be updated with the total number of iterations performed.
 * @param tess The current tessellation of the simulation domain.
 * @param cells The current state of computational cells.
 * @param extensives The current state of extensive conserved quantities.
 * @param dt The time step size.
 * @param time The current simulation time.
 * 
 * @return A boolean indicating whether the time step was successful.
 */
        virtual bool step(double const tolerance, 
                          int& total_iters, 
                          Tessellation3D const& tess, 
                          std::vector<ComputationalCell3D>& cells,
                          std::vector<Conserved3D>& extensives, 
                          double const dt,
                          double const time) const = 0;
        
/**
 * @brief A pure virtual function for performing post-step operations.
 * 
 * This function should be implemented in derived classes to perform any necessary operations after each time step.
 * 
 * @return A boolean indicating whether the post-step operation was successful.
 */
        virtual bool poststep() const = 0; 
/**
 * @brief A pure virtual function for calculating the maximum allowable time step size.
 * 
 * This function should be implemented in derived classes to determine the maximum allowable time step size based on the current state of the simulation.
 * 
 * @param dt The current time step size.
 * @param tess The current tessellation of the simulation domain.
 * @param cells The current state of computational cells.
 * 
 * @return The maximum allowable time step size.
 */
        virtual double calculate_dt(double const dt,
                                    Tessellation3D& tess,
                                    std::vector<ComputationalCell3D>& cells) const = 0;

        // The last calculate_dt limit cell by cell, indexed like the owned
        // cells it was given (infinity where a cell sets no limit, no growth
        // cap), or nullptr for a driver with only the global limit.  Read by
        // the adaptive gain bound; it never shapes a step.
        virtual std::vector<double> const* lastCellTimeStepLimits() const
        {
            return nullptr;
        }

        /** Serial active-row implicit solve used by individual timesteps. */
        virtual bool supportsIndividualTimeSteps() const { return false; }
        // True only for drivers whose every canonical-extensive write in an individual step goes through the base
        // RadiationDriver::stepIndividual journalled write (the sparse canonical path of RadiationStep needs it).
        virtual bool JournalsAllCanonicalWrites() const { return false; }

        // True only for drivers whose PostCG writes the cells and extensives of active rows alone, so that in an
        // individual step no inactive cell changes before the frozen-Dirichlet restore (PARTIAL_SETUP_FAST).
        virtual bool PostCGWritesActiveRowsOnly() const { return false; }

        virtual std::size_t individualUnknownsPerCell() const { return 1; }

        virtual bool prestepIndividual(
            Tessellation3D const& tess,
            std::vector<ComputationalCell3D> const& cells,
            IndividualStepContext const& context) const;

        virtual bool stepIndividual(
            double tolerance,
            int& total_iters,
            Tessellation3D const& tess,
            std::vector<ComputationalCell3D>& cells,
            std::vector<Conserved3D>& extensives,
            IndividualStepContext const& context,
            double interval_fraction,
            double time,
            std::vector<ComputationalCell3D> const* canonical_cells = nullptr,
            std::vector<Conserved3D>* canonical_extensives = nullptr,
            std::vector<std::size_t> const* owned_to_canonical = nullptr) const;

        virtual bool poststepIndividual() const { return poststep(); }

        void beginIndividualPassiveWakeTracking(
            std::size_t canonical_cell_count) const
        {
            individual_passive_reference_time_steps_.assign(
                canonical_cell_count, std::numeric_limits<double>::max());
        }

        std::vector<double> const&
        getIndividualPassiveReferenceTimeSteps() const
        {
            return individual_passive_reference_time_steps_;
        }

        // One individual radiation event: between these calls the canonical cell IDs passed to stepIndividual()
        // do not change, so the stable-ID index of the canonical cells is built once and reused by every candidate.
        void BeginIndividualCanonicalScope() const;
        void EndIndividualCanonicalScope() const noexcept;

        // Optional journal of the canonical extensives an accepted distributed-active candidate overwrote: each
        // canonical index with its value before the write, in write order.  BeginIndividualCanonicalWriteJournal()
        // enables and clears it; the driver only appends to it and never reads it.  RadiationStep's sparse path
        // rebuilds its wake inputs from it, so every canonical_extensives write made by stepIndividual() (or an
        // override) must be journalled here.
        void BeginIndividualCanonicalWriteJournal() const;
        void EndIndividualCanonicalWriteJournal() const noexcept;
        std::vector<std::size_t> const& GetIndividualCanonicalWriteIndices() const {return CanonicalWriteIndices;}
        std::vector<Conserved3D> const& GetIndividualCanonicalWriteBefore() const {return CanonicalWriteBefore;}

        // Release state whose shape follows the committed cell ownership or
        // tessellation.  Called only after an event commits, before AMR/LB
        // state is reused.
        virtual void releaseIndividualTopologyStorage() const noexcept;

        virtual void calculateIndividualTimeSteps(
            IndividualStepContext const& context,
            Tessellation3D& tess,
            std::vector<ComputationalCell3D>& cells,
            std::vector<double>& time_step_limits,
            std::vector<ComputationalCell3D> const* canonical_owned_cells = nullptr,
            std::vector<std::size_t> const* local_to_global = nullptr) const;


        // Test seam (RICH_TEST_FORCED_RADIATION_REJECTION): arm before an individual candidate, on every rank with
        // the same rank; the candidate then runs in full and `failing_rank` alone marks it invalid at its last
        // collective acceptance check, so every rank rejects and rolls it back.  TakeTestForcedRejection() disarms
        // and returns whether the last stepIndividual() call consumed the seam (identical on every rank).
        void ArmTestForcedRejection(int failing_rank) const
        {
            test_forced_rejection_rank_ = failing_rank;
            test_forced_rejection_consumed_ = false;
        }

        bool TakeTestForcedRejection() const
        {
            bool const consumed = test_forced_rejection_consumed_;
            test_forced_rejection_rank_ = -1;
            test_forced_rejection_consumed_ = false;
            return consumed;
        }

        void clearStepFailure() const
        {
            last_step_failure_reason_.clear();
            last_step_failure_diagnostics_.clear();
            last_step_failure_cell_id_ = std::numeric_limits<size_t>::max();
            last_step_failure_cell_local_ = false;
            last_step_failure_remote_ = false;
        }

        void setStepFailure(
            std::string const& reason,
            size_t cell_id = std::numeric_limits<size_t>::max(),
            std::string const& diagnostics = std::string()) const
        {
            if (!reason.empty() && last_step_failure_reason_.empty()) {
                last_step_failure_reason_ = reason;
                last_step_failure_diagnostics_ = diagnostics;
                last_step_failure_cell_id_ = cell_id;
                last_step_failure_cell_local_ = false;
                last_step_failure_remote_ = false;
            }
        }

        /** Record a failure caused by one owned active cell. */
        void setCellLocalStepFailure(
            std::string const& reason,
            size_t cell_id,
            std::string const& diagnostics = std::string()) const
        {
            if (!reason.empty() && last_step_failure_reason_.empty()) {
                last_step_failure_reason_ = reason;
                last_step_failure_diagnostics_ = diagnostics;
                last_step_failure_cell_id_ = cell_id;
                last_step_failure_cell_local_ = true;
                last_step_failure_remote_ = false;
            }
        }

        void markStepFailureCellLocal(size_t cell_id) const
        {
            if(!last_step_failure_reason_.empty()) {
                last_step_failure_cell_id_ = cell_id;
                last_step_failure_cell_local_ = true;
                last_step_failure_remote_ = false;
            }
        }

        void markStepFailureRemote() const
        {
            if(!last_step_failure_reason_.empty()) {
                last_step_failure_cell_local_ = false;
                last_step_failure_remote_ = true;
            }
        }

        std::string const& getLastStepFailureReason() const { return last_step_failure_reason_; }
        std::string const& getLastStepFailureDiagnostics() const
        {return last_step_failure_diagnostics_;}
        size_t getLastStepFailureCellId() const { return last_step_failure_cell_id_; }
        bool getLastStepFailureIsCellLocal() const
        {return last_step_failure_cell_local_;}
        bool getLastStepFailureIsRemote() const
        {return last_step_failure_remote_;}

    bool const flux_limiter_;
    bool const hydro_on_;
    bool const compton_on_;

    double mass_scale_;
    double length_scale_;
    double time_scale_;

protected:
    struct IndividualFaceCoefficient
    {
        std::size_t left = 0;
        std::size_t right = 0;
        std::size_t group = 0;
        double coefficient = 0;
        // Full scheduler face interval; coefficient includes candidate fraction.
        double time_step = 0;
    };

    struct SpectralRepairEvent
    {
        std::uint64_t repaired_cells = 0;
        std::uint64_t repaired_groups = 0;
        double injected_energy = 0;
        double maximum_relative_deficit = 0;
        std::uint64_t representative_cell_id =
            std::numeric_limits<std::uint64_t>::max();
        std::uint64_t representative_group =
            std::numeric_limits<std::uint64_t>::max();
        double representative_original_extent = 0;
        double representative_floor_extent = 0;
        double representative_injected_extent = 0;
        double owned_radiation_energy = 0;
    };

    struct IndividualRadiationLocalDefectMeasure
    {
        double relative_fraction = 0;
        double allowed_withdrawal = 0;
        double tolerance_ratio = 0;
    };

    // Candidate-local omitted active/passive interface transfer.  This record
    // remains pending until every solver, residual, mapping, positivity, and
    // post-solve check has accepted the surrounding transaction.
    struct IndividualRadiationDefectEvent
    {
        long double signed_extent = 0;
        long double absolute_extent = 0;
        long double passive_withdrawal_extent = 0;
        long double passive_deposit_extent = 0;
        double maximum_local_fraction = 0;
        double maximum_local_tolerance_ratio = 0;
        double candidate_start_positive_global_extent = 0;
        double rhs_derived_global_floor = 0;
        double normalization_scale = 0;
        double event_absolute_fraction = 0;
        double projected_cumulative_signed_fraction = 0;
        double projected_cumulative_absolute_fraction = 0;
        std::uint64_t face_group_terms = 0;
        std::uint64_t duplicate_face_group_terms = 0;
        std::uint64_t representative_active_id =
            std::numeric_limits<std::uint64_t>::max();
        std::uint64_t representative_passive_id =
            std::numeric_limits<std::uint64_t>::max();
        std::uint64_t representative_group =
            std::numeric_limits<std::uint64_t>::max();
        std::uint64_t representative_active_rank =
            std::numeric_limits<std::uint64_t>::max();
        std::uint64_t representative_rank =
            std::numeric_limits<std::uint64_t>::max();
        bool valid = true;
    };

    double individualCellTimeStep(std::size_t index, double fallback) const;
    double individualScheduledTimeStep(std::size_t index, double fallback) const;
    double individualFaceTimeStep(std::size_t left,
                                  std::size_t right,
                                  double fallback) const;
    bool individualCellActive(std::size_t index) const;
    void commitSpectralRepairAccounting(
        SpectralRepairEvent const& local_event,
        char const* scope) const;
    void commitResidualCorrectionAccounting(
        CG::HistoricalMGResidualCorrectionDiagnostics const& diagnostics) const;
    void recordIndividualFaceCoefficient(std::size_t left,
                                         std::size_t right,
                                         std::size_t group,
                                         double coefficient) const;
    double collectiveIndividualRadiationDefectScale(
        double local_candidate_start_positive_extent,
        double local_rhs_floor,
        double& candidate_start_positive_global_extent,
        double& rhs_derived_global_floor) const;
    IndividualRadiationDefectAccounting&
        individualRadiationDefectAccounting() const;
    static IndividualRadiationLocalDefectMeasure
        measureIndividualRadiationLocalDefect(
            double withdrawal,
            double passive_extent,
            double roundoff_floor,
            double normalization_scale);
    bool validateIndividualRadiationDefect(
        IndividualRadiationDefectEvent& local_event) const;
    void commitIndividualRadiationDefect(
        IndividualRadiationDefectEvent const& event) const;
    virtual void prepareIndividualCandidate(
        Tessellation3D const&,
        std::vector<ComputationalCell3D> const&) const
    {}

    // True only while stepIndividual() owns a rollback snapshot for these
    // exact state vectors and has explicitly enabled inner-snapshot elision.
    // Pointer identity prevents direct global calls, reduced active solves,
    // or nested work on different state vectors from borrowing the marker.
    bool outerAllActiveTransactionCovers(
        std::vector<ComputationalCell3D> const& cells,
        std::vector<Conserved3D> const& extensives) const
    {
        return outer_transaction_cells_ == &cells &&
            outer_transaction_extensives_ == &extensives;
    }

    // The all-active fast path may leave a large reusable global-solver
    // workspace in a derived driver.  Release it before constructing the
    // independent distributed-active system.
    virtual void ReleaseDormantGlobalSolverStorage() const
    {}

    virtual bool validateIndividualCoefficients(
        IndividualStepContext const&,
        std::vector<ComputationalCell3D> const&) const
    {
        return true;
    }

    // Local check of the reduced system's physical right-hand side (after
    // frozen-column elimination), one entry per active unknown; the caller
    // agrees the result across ranks.  Sets the step failure when false.
    virtual bool validateIndividualVerificationRhs(
        std::vector<std::size_t> const&,
        std::size_t,
        std::vector<double> const&,
        std::vector<ComputationalCell3D> const&) const
    {
        return true;
    }

    // A derived solver may compare the states immediately before and after the
    // final residual correction, replace a nonphysical local source block, and
    // ask for a rebuild before any candidate state is committed.
    virtual bool requestIndividualSolutionRetry(
        std::vector<double> const&,
        std::vector<double> const&,
        std::vector<std::size_t> const&,
        std::vector<ComputationalCell3D> const&,
        char const*) const
    {
        return false;
    }

    // Apply cell-local physics that was deliberately removed from the reduced
    // matrix.  stepIndividual() calls this only after diffusion face transfers,
    // controlled spectral repair, and Erad/group synchronization.  The final
    // scalar is the canonical owned-cell Erad maximum reduced with MPI_MAX.
    // Whether applyIndividualPostSolvePhysics reads its global maximum cell radiation extent.  Without it, and with
    // one unknown per cell, that maximum feeds only failure messages (RICH_RADIATION_PARTIAL_SETUP_FAST).
    virtual bool UsesPostSolveRadiationMaximum() const
    {
        return false;
    }

    virtual bool applyIndividualPostSolvePhysics(
        Tessellation3D const&,
        std::vector<ComputationalCell3D>&,
        std::vector<Conserved3D>&,
        double,
        double) const
    {
        return true;
    }

    // Cell-local post-solve operators can perform the same controlled
    // spectral repair as the main candidate finalizer.  Append their pending
    // accounting only after the surrounding transaction has passed every
    // validation check.
    virtual void appendPendingSpectralRepairEvent(
        SpectralRepairEvent&) const
    {}

    // The global implementation must be safe on every MPI rank, including a
    // rank with zero owned rows, so the capability is opt-in.
    virtual bool supportsAllActiveIndividualGlobalStep() const
    {
        return false;
    }

    EquationOfState const& eos_;
    mutable std::string last_step_failure_reason_;
    mutable std::string last_step_failure_diagnostics_;
    mutable size_t last_step_failure_cell_id_ =
        std::numeric_limits<size_t>::max();
    mutable bool last_step_failure_cell_local_ = false;
    mutable bool last_step_failure_remote_ = false;
    // ArmTestForcedRejection: the failing rank (-1 disarmed) and whether a stepIndividual() call consumed it.
    mutable int test_forced_rejection_rank_ = -1;
    mutable bool test_forced_rejection_consumed_ = false;
    // At the last collective acceptance check of stepIndividual(): consumes an armed seam on every rank and returns
    // false on its failing rank (with a step failure), `valid` otherwise.
    bool applyTestForcedRejection(bool valid) const;
    mutable std::vector<ComputationalCell3D> const*
        outer_transaction_cells_ = nullptr;
    mutable std::vector<Conserved3D> const*
        outer_transaction_extensives_ = nullptr;
    mutable IndividualStepContext const* individual_context_ = nullptr;
    mutable double individual_interval_fraction_ = 1.0;
    mutable std::vector<IndividualFaceCoefficient> individual_face_coefficients_;
    mutable std::vector<double> individual_passive_reference_time_steps_;
    mutable RadiationRepairAccounting standalone_repair_accounting_;
    mutable IndividualRadiationDefectAccounting
        standalone_defect_accounting_;

    // Stable-ID index of the canonical cells.  PrepareCanonicalIndex() (re)builds it unless the current scope
    // already holds it for the same vector and returns false when an ID repeats.  FindCanonicalIndex() then
    // answers like std::map::find on a map filled by emplace in canonical order (the lowest index of an ID).
    bool PrepareCanonicalIndex(std::vector<ComputationalCell3D> const& CanonicalCells) const;
    bool FindCanonicalIndex(std::size_t Id, std::size_t& Canonical) const;

    mutable bool CanonicalScopeActive = false;
    mutable std::vector<ComputationalCell3D> const* CanonicalIndexSource = nullptr;
    mutable std::size_t CanonicalIndexSize = 0;
    mutable bool CanonicalIndexUnique = false;
    // RICH_RADIATION_PARTIAL_SETUP_FAST: the index is kept across events with the IDs it was built from, in
    // canonical order, and reused while they are unchanged.
    mutable bool CanonicalIndexRetained = false;
    mutable std::vector<std::size_t> CanonicalIndexIds;
    // (stable ID, canonical index), sorted.
    mutable std::vector<std::pair<std::size_t, std::size_t> > CanonicalIndexById;
    mutable bool CanonicalWriteJournalEnabled = false;
    mutable std::vector<std::size_t> CanonicalWriteIndices;
    mutable std::vector<Conserved3D> CanonicalWriteBefore;
};

// s49 radiation switches.  Read from the environment on the first call, which every rank must make, and agreed across
// ranks with one reduction; an invalid or differing value aborts.  Unset means 0.
//   RICH_D2_TIMERS                          0/1    per-event radiation timers, one RICH_D2_RAD line per event
//   RICH_MG_DISTRIBUTED_ACTIVE_TWO_STAGE    0/1/2  two-stage Krylov reductions (2 = shadow: solve both ways)
//   RICH_RADIATION_ALL_ACTIVE_SKIP_PASSIVE  0/1/2  skip empty passive exchanges in all-active events (2 = verify)
//   RICH_RADIATION_REMAP_MINIMAL            0/1/2  remap only the context fields radiation reads (2 = verify)
//   RICH_MG_CAPTURE_SYSTEMS                 N>=0   write the first N distributed systems of each kind
//   RICH_MG_CAPTURE_DIR                     path   capture directory, required when N > 0
//   RICH_RADIATION_PARTIAL_SETUP_FAST       0/1/2  skip per-cell work of partial radiation events that nothing
//                                                  reads (2 = verify)
struct RadiationS49Switches
{
    int Timers = 0;
    int TwoStage = 0;
    int SkipPassive = 0;
    int RemapMinimal = 0;
    int PartialSetupFast = 0;
    long long CaptureSystems = 0;
    std::string CaptureDirectory;
};

RadiationS49Switches const& RadiationS49RuntimeSwitches();

// D2 radiation timers (RICH_D2_TIMERS=1).  BeginRadiationD2Event clears this rank's sums; EndRadiationD2Event reduces
// them to rank 0 (two MPI_Reduce calls) and prints one RICH_D2_RAD line.  Both are collective.
void BeginRadiationD2Event();
void EndRadiationD2Event(std::uint64_t EventTick, double EventSeconds, double EventMpiSeconds, std::size_t Candidates,
    std::size_t RejectedCandidates, std::size_t ActiveTargets);

// Counts one clean RICH_RADIATION_REMAP_MINIMAL=2 check; rank 0 reports the first and every 100th.
void RecordRadiationRemapVerification();

// Counts one matrix build whose inactive rows passed the RICH_RADIATION_PARTIAL_SETUP_FAST=2 check.
void RecordRadiationPartialSetupRowCheck();

using boost::math::pow;
static inline double get_radiation_energy_density(double const T) { return CG::radiation_constant*pow<4>(T); }
static inline double get_temperature(double const radiation_energy_density) { return std::sqrt(std::sqrt(radiation_energy_density/CG::radiation_constant)); }
static inline double get_radiation_cv(double const T) { return 4.0*CG::radiation_constant*pow<3>(T); }

#endif
