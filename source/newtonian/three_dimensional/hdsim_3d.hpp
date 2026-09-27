#ifndef HDSIM_3D_HPP
#define HDSIM_3D_HPP 1

#include <cassert>
#include <chrono>
#include <limits>
#include <memory>
#include <unordered_map>
#include <MeshDecomposer3D/hilbert/HilbertOrder3D.hpp>
#include "misc/utils.hpp"
#include "computational_cell.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "conserved_3d.hpp"
#include "IndividualChangeWakeAccounting.hpp"
#include "../common/equation_of_state.hpp"
#include "point_motion_3d.hpp"
#include "time_step_function3D.hpp"
#include "flux_calculator_3d.hpp"
#include "cell_updater_3d.hpp"
#include "extensive_updater3d.hpp"
#include "SourceTerm3D.hpp"
#include "newtonian/three_dimensional/simulation/ProgressTracker.hpp"
#include "newtonian/three_dimensional/simulation/IndividualTimeStep.hpp"
#include "newtonian/three_dimensional/simulation/StepDiagnostics.hpp"
#include "CostCalculator3D.hpp"
#include "Hllc3D.hpp"

#ifdef RICH_MPI
  #include <mpi.h>
  #include "mpi/mpi_commands.hpp"
  #include "mpi/ExchangeChain.hpp"
#endif

class SphericalShellProjector3D;
class SphericalShellGeometry3D;

//! \brief Three dimensional simulation
class HDSim3D
{
public:
  /*! \brief Class constructor
    \param tess Tessellation
    \param cells Initial computational cells
    \param eos Equation of state
    \param pm Point motion scheme
    \param tsc Time step calculator
    \param fc Flux calculator
    \param cu Cell updater
    \param eu Extensive updater
    \param source Source term
    \param tsn The names of the tracers and stickers, first is the tracers and second is stickers
    \param SR Special relativity flag
  */
  HDSim3D(Tessellation3D& tess,
	  vector<ComputationalCell3D>& cells,
    vector<Conserved3D>& extensives,
	  const EquationOfState& eos,
    ProgressTracker &pt,
	  const PointMotion3D& pm,
	  TimeStepFunction3D& tsc,
	  const FluxCalculator3D& fc,
	  const CellUpdater3D& cu,
	  const ExtensiveUpdater3D& eu,
	  const	SourceTerm3D& source,
	  const pair<vector<string>, vector<string> >& tsn,
	  bool SR=false
    #ifdef RICH_MPI
      , std::shared_ptr<CostCalculator3D> cost_calc = std::make_shared<CostCalculator3D>()
    #endif // RICH_MPI  
  );

  //! \brief Advances the simulation in time (first order)
  void timeAdvance();
  //! \brief Advances the simulation in time (second order)
  void timeAdvance2();

  void timeAdvanceIndividual(const IndividualStepContext& context);

  bool supportsIndividualTimeSteps(void) const
  {return !special_relativity_ && spherical_shell_projector_ == nullptr &&
    eu_.SupportsIndividualTimeSteps() &&
    source_.SupportsIndividualTimeSteps() &&
    GetFullStateFluxCalculator().SupportsIndividualTimeSteps();}

  void suggestIndividualTimeSteps(const IndividualStepContext& context,
    vector<double>& time_step_limits) const;

  /*! \brief Wakes passive cells whose conserved state changed too much
    since their activation (see individual_conserved_change_).
    \param context Event context
    \param wake_deadlines Per canonical cell wake interval, reduced in place
  */
  /*! \brief Per-cell CFL limits of the current full mesh
    Valid after a global step (face velocities of this mesh are cached) until
    the mesh is rebuilt (AMR, rebalance, box growth).  Returns false when the
    timestep function cannot supply them or the cached face velocities are
    not those of the current mesh (CellTimeStepLimitsStale).
    \param limits Output, one entry per owned cell
    \return Whether limits were filled
  */
  bool CollectCellTimeStepLimits(vector<double>& limits) const;

  /*! \brief True when CollectCellTimeStepLimits could supply limits but the
    cached face velocities do not belong to the current mesh: it was rebuilt
    after the last global step set them, or they were overwritten since.
  */
  bool CellTimeStepLimitsStale(void) const;

  /*! \brief Per-cell limits of every owned cell on the current full mesh,
    before anything advances on it (after a box growth): the individual
    event's rule -- wave-speed CFL, the source term's per-cell limits times the
    source factor, and the mesh-drift guard -- with face velocities and
    closing speeds from `point_velocities` (owned cells, mesh order: the
    velocities each generator moves with through its next interval).
    `accelerations` receives the refreshed individual acceleration cache when
    the source keeps one (else stays empty).  Collective under MPI; false on
    every rank when the timestep function has no per-cell rule or the mesh
    does not hold exactly the owned cells.
  */
  bool SynchronizedTimeStepLimits(vector<Vector3D> const& point_velocities,
    vector<double>& limits, vector<Vector3D>& accelerations) const;

  /*! \brief The source's individual acceleration of every owned cell on the
    current full mesh (SourceTerm3D::RefreshIndividualAccelerations), for the
    acceleration cache.  Collective under MPI; false on every rank when the
    source keeps no acceleration cache, cannot refresh it (an acceleration
    without target evaluation), or the mesh does not hold exactly the owned
    cells.
  */
  bool RefreshIndividualAccelerations(vector<Vector3D>& accelerations) const;

  void suggestIndividualChangeWakes(const IndividualStepContext& context,
    vector<double>& change_ratios) const;

  SourceStepTiming GetLastSourceStepTiming(void) const
  {return last_source_step_timing_;}

  MeshBuildTiming GetLastMeshBuildTiming(void) const
  {return last_mesh_build_timing_;}

  /*! \brief Second order time advance with Lagrangian x-boundaries
    \param left_external Exterior state at left x-boundary (nullptr = vacuum)
    \param right_external Exterior state at right x-boundary (nullptr = vacuum)
   */
  void timeAdvanceLagrangian1D(
    const ComputationalCell3D* left_external = nullptr,
    const ComputationalCell3D* right_external = nullptr);

  /*! \brief Third order time advance
   */
  void timeAdvance3();

  /*! \brief Third order time advance
   */
  void timeAdvance32();

  /*! \brief Third order time advance
   */
  void timeAdvance33();

  /*! \brief Fourth order time advance
   */
  void timeAdvance4();

  /*! \brief Access to tessellation
    \return Tessellation
   */
  const Tessellation3D& getTessellation(void) const;

  /*! \brief Access to computational cells
    \return Computational cells
   */
  const vector<ComputationalCell3D>& getCells(void) const;
  /*! \brief Access to extensive cells
  \return Extensive cells
  */
  const vector<Conserved3D>& getExtensives(void) const;
  /*! \brief Access to tessellation
  \return Tessellation
  */
  Tessellation3D& getTessellation(void);

  /*! \brief Access to computational cells
  \return Computational cells
  */
  vector<ComputationalCell3D>& getCells(void);
  /*! \brief Access to extensive cells
  \return Extensive cells
  */
  vector<Conserved3D>& getExtensives(void);

  double getTime(void) const;

  size_t getCycle(void) const;

  double getTimeStep(void) const {return this->tsc_.GetTimeStep();}

  double suggestTimeStep(void) const {return this->tsc_.SuggestTimeStep();}

  void SetTimeStep(double dt) {this->tsc_.SetTimeStep(dt);}

  void SetSphericalShellProjector(
    std::shared_ptr<SphericalShellProjector3D> projector,
    FluxCalculator3D const& perturbation_flux_calculator);

  /*! \brief The hydro part of the step a global step would take, as of the
    latest individual time-step suggestion: the guard floor's cached smallest
    CFL/source limit over all cells (see RICH_INDIVIDUAL_GUARD_FLOOR).  Zero
    before the first suggestion or with the floor off.  Identical on every rank.
  */
  double GetIndividualGlobalStepReference(void) const
  {return individual_global_step_reference_;}

  //! \brief Forget the reference (a global phase makes it stale).
  void ResetIndividualGlobalStepReference(void)
  {individual_global_step_reference_ = 0;}

  /*! \brief Seed the reference when an individual phase starts, with the step
    the last global step took, so the first event (before the first suggestion
    refreshes it) scales per-step drivers by its own interval.
  */
  void SetIndividualGlobalStepReference(double reference)
  {individual_global_step_reference_ = std::isfinite(reference) && reference > 0 ? reference : 0;}

  size_t GetSphericalPerturbationEvaluationCount(void) const
  {return spherical_perturbation_evaluation_count_;}

  void ResetIndividualMeshState(void)
  {
    individual_points_.clear();
    individual_centroids_.clear();
    DrainIndividualChangeWakeSamples();
    individual_conserved_change_.clear();
    individual_limit_at_activation_.clear();
    individual_mesh_target_ids_.clear();
    individual_adjacency_.clear();
    individual_mesh_restore_pending_ = false;
    individual_event_mesh_reusable_ = false;
  }

  void ReleaseIndividualPartialMeshScratch(void) noexcept
  {
    vector<size_t>().swap(individual_mesh_target_ids_);
    individual_mesh_restore_pending_ = false;
  }

  void ReleaseIndividualRebalanceScratch(void) noexcept
  {
    vector<Vector3D>().swap(point_vel_scratch_);
    vector<Vector3D>().swap(face_vel_scratch_);
    face_vel_build_generation_ = kNoFaceVelocityGeneration;
    vector<Vector3D>().swap(individual_points_);
    vector<Vector3D>().swap(individual_centroids_);
    vector<Vector3D>().swap(oldpoints_scratch_);
    vector<Vector3D>().swap(tessellation_points_scratch_);
    vector<Conserved3D>().swap(fluxes_scratch_);
    vector<Conserved3D>().swap(mid_extensives_scratch_);
    vector<Conserved3D>().swap(u1_scratch_);
    vector<Conserved3D>().swap(u2_scratch_);
    vector<Conserved3D>().swap(u3_scratch_);
    vector<size_t>().swap(hilbert_order_scratch_);
    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >().swap(
      face_values_scratch_);
    vector<size_t>().swap(individual_mesh_target_ids_);
    vector<IndividualAdjacencyRecord>().swap(individual_adjacency_);
    // Indexed by owned cell: an ownership change misaligns it even when the
    // count survives.
    std::vector<unsigned char>().swap(individual_limit_reason_);
    individual_mesh_restore_pending_ = false;
    individual_event_mesh_reusable_ = false;
  }

  const vector<Vector3D>& GetIndividualGeneratorPoints(void) const
  {return individual_points_;}

  // Cadence diagnostic (see individual_limit_reason_).
  std::vector<unsigned char> const& GetIndividualLimitReasons(void) const
  {
    return individual_limit_reason_;
  }

  const vector<Vector3D>& GetIndividualCellCentroids(void) const
  {return individual_centroids_;}

  const vector<size_t>& GetIndividualMeshTargetIDs(void) const
  {return individual_mesh_target_ids_;}

  void RestoreIndividualMeshTargetIDs(vector<size_t> const& target_ids)
  {
    individual_mesh_target_ids_ = target_ids;
    individual_mesh_restore_pending_ = true;
  }

  #ifdef RICH_MPI
    const ExchangeChain &GetExchangeChain(void) const {return this->exchange_chain_;}
  #endif // RICH_MPI

  #ifdef RICH_MPI
    std::shared_ptr<CostCalculator3D> cost_calc_;
  #endif // RICH_MPI

private:
  void RefreshSphericalShellGeometry(
    char const* update_name,
    bool always_report = false);

  void ApplySphericalBackgroundCorrection(
    vector<ComputationalCell3D> const& stage_input_cells,
    vector<Conserved3D> const& stage_input_extensives,
    vector<Vector3D> const& face_velocities,
    vector<Vector3D> const& point_velocities,
    double time,
    double dt,
    bool source_before_extensive_update,
    vector<Conserved3D>& full_candidate);

  FluxCalculator3D const& GetSphericalPerturbationFluxCalculator(void) const;

  FluxCalculator3D const& GetFullStateFluxCalculator(void) const;

  Tessellation3D& tess_;
  const EquationOfState& eos_;
  vector<ComputationalCell3D> &cells_;
  vector<Conserved3D> &extensive_;
  const PointMotion3D& pm_;
  TimeStepFunction3D& tsc_;
  const FluxCalculator3D& fc_;
  const CellUpdater3D& cu_;
  const ExtensiveUpdater3D& eu_;
  const	SourceTerm3D &source_;
  const ProgressTracker &pt_;
  const bool special_relativity_;
  SourceStepTiming last_source_step_timing_;
  MeshBuildTiming last_mesh_build_timing_;
	  vector<Vector3D> point_vel_scratch_;
	  vector<Vector3D> face_vel_scratch_;
	  // Build generation of tess_ that face_vel_scratch_ belongs to: set where
	  // timeAdvance2 leaves face velocities of its final mesh, cleared
	  // wherever the scratch is reused (CollectCellTimeStepLimits).
	  static constexpr size_t kNoFaceVelocityGeneration =
		  std::numeric_limits<size_t>::max();
	  size_t face_vel_build_generation_ = kNoFaceVelocityGeneration;
	  vector<Vector3D> individual_points_;
	  vector<Vector3D> individual_centroids_;
	  // Per canonical cell since its last activation (IndividualChangeWakeAccounting.hpp).
	  mutable vector<IndividualConservedChange> individual_conserved_change_;
	  // Conserved-change wake accuracy since the last report.
	  mutable IndividualChangeWakeAccounting individual_change_wake_accounting_;
	  // Censors every outstanding wake sample before the accumulators are discarded.
	  void DrainIndividualChangeWakeSamples(void) const;
	  // Ratio above which a woken cell's change at activation is a violation (twice the wake fraction).
	  double IndividualChangeWakeRatioLimit(void) const;
	  /*! \brief Timestep limit in force when each canonical cell's current
	    interval opened, kept so that the next activation can report an
	    interval that ran longer than its own limit allowed.
	  */
	  mutable vector<double> individual_limit_at_activation_;
	  // Which hydro limit set each active cell's suggestion at its latest
	  // activation: 1 CFL/source, 2 mesh drift, 3 mass loss, 4 thermal loss.
	  mutable std::vector<unsigned char> individual_limit_reason_;
	  // Hydro/source limit of each cell at its latest activation or
	  // synchronized evaluation on this rank, by cell ID, for the guard floor
	  // (RICH_INDIVIDUAL_GUARD_FLOOR).  Keyed by ID so it survives AMR, which
	  // runs ResetIndividualMeshState (that keeps it).  Entries do not follow a
	  // cell that migrates to another rank; the cell counts again from its next
	  // activation there (the floor's report gives the coverage, not the age).
	  mutable std::unordered_map<size_t, double> individual_hydro_limit_by_id_;
	  mutable double individual_global_step_reference_ = 0;
	  /*! \brief Measured costs of individual event-mesh builds, for the
	    adaptive per-rank closure threshold (RICH_INDIVIDUAL_PARTIAL_THRESHOLD_ADAPTIVE):
	    exponentially weighted sums of the per-attempt partial-build time
	    (max over ranks) against the largest per-rank target fraction, of the
	    attempts per partial build, and of the full-build time.  Updated from
	    reduced values after every build, so identical on every rank.
	  */
	  struct IndividualPartialBuildCostModel
	  {
	    double w = 0, x = 0, xx = 0, y = 0, xy = 0;  // per-attempt fit
	    double attempts = 0, attempts_w = 0;
	    double full = 0, full_w = 0;
	    std::size_t partial_samples = 0, full_samples = 0;
	    double fraction = 0;  // 0 until enough samples
	  };
	  mutable IndividualPartialBuildCostModel individual_partial_cost_;
	  vector<Conserved3D> individual_pre_flux_extensives_scratch_;
	  bool individual_event_mesh_reusable_ = false;
	  vector<size_t> individual_mesh_target_ids_;
	  bool individual_mesh_restore_pending_ = false;
	  /*! \brief Face neighbours of a canonical cell as of the last event mesh
	    that contained it, by stable ID with the owning rank.  Seeds the
	    partial-build target with the two-cell reconstruction shell before the
	    build, so the closure check after it rarely adds cells and rebuilds.
	    A record whose ID no longer matches its slot is ignored.
	  */
	  struct IndividualAdjacencyRecord
	  {
	    size_t cell_id = std::numeric_limits<size_t>::max();
	    vector<size_t> neighbor_ids;
	    vector<int> neighbor_owners;
	  };
	  vector<IndividualAdjacencyRecord> individual_adjacency_;
	  vector<Vector3D> oldpoints_scratch_;
  vector<Vector3D> tessellation_points_scratch_;
  vector<Conserved3D> fluxes_scratch_;
  vector<Conserved3D> mid_extensives_scratch_;
  vector<Conserved3D> u1_scratch_;
  vector<Conserved3D> u2_scratch_;
  vector<Conserved3D> u3_scratch_;
  vector<size_t> hilbert_order_scratch_;
  std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > face_values_scratch_;
  std::shared_ptr<SphericalShellGeometry3D> spherical_shell_geometry_;
  std::shared_ptr<SphericalShellProjector3D> spherical_shell_projector_;
  FluxCalculator3D const* spherical_perturbation_flux_calculator_ = nullptr;
  size_t spherical_perturbation_evaluation_count_ = 0;
  #ifdef RICH_MPI
    ExchangeChain exchange_chain_;
  #endif // RICH_MPI
};

#endif // HDSIM_3D_HPP
