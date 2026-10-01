# Changelog

All notable changes to RICH are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Grey diffusion momentum positivity (`RICH_RADIATION_MOMENTUM_POSITIVITY=1`, default off; `Diffusion::BuildMatrix`,
  `Diffusion::PostCG`, `MatrixBuilder::MatrixBuildRejected`, `RadiationDriver::validateIndividualVerificationRhs`):
  - The positive excess of the central velocity (pressure-work/relativistic) coupling is lumped onto the diagonal per
    interior column, and uncertified matrices are rejected collectively before preconditioning.
  - PostCG uses the matrix's lumped face values, face timesteps and alpha.
  - Two energy-conserving caps (`RICH_RADIATION_MOMENTUM_KINETIC_LOSS_FRACTION`, default 0.5) bound the kinetic gain
    from the radiation force and the relativistic thermal debit.
  - Rank-0 matrix/exchange ledgers include coefficient and reservoir closures.
  - It removes the late-time radiation retry cascades of the TDE run: snapshot 70 -> t = 50 at 1.77x the global run
    with K = 4 and the settings in docs/user-guide/individual-timesteps.md.
- Individual timesteps: `RICH_INDIVIDUAL_MAX_BIN_SPREAD`, `RICH_INDIVIDUAL_BIN_ANCHOR_MARGIN`,
  `RICH_ADAPTIVE_STAY_INDIVIDUAL`, `RICH_INDIVIDUAL_RADIATION_ANCHOR_SUBCYCLE`/`_BAND`,
  `RICH_INDIVIDUAL_RADIATION_ENTRY_PROBE`, `RICH_INDIVIDUAL_CLOSURE_REEXPAND`; recommended values in the user guide.
- Box growth during individual time stepping: `UpdateBoxSynchronized`, `BoxGrowthDue`
  (`3D/GeometryCommon/UpdateBox`) and `Simulation::GrowDomainAtSynchronizedIndividualState` grow the box at a
  synchronized individual event with the legacy `UpdateBox` semantics, migrating primitives, extensives and scheduler
  states by stable ID and capping every cell's interval by the individual event's limits on the rebuilt mesh
  (`PhysicsStep::synchronizedCellTimeStepLimits`: CFL, source per-cell limits via
  `SourceTerm3D::SynchronizedIndividualLimits`, mesh-drift guard); the gravity acceleration cache is refreshed on the
  rebuilt state; `DiffusionForce` refuses the growth. Before, the TDE driver grew the box only on global steps, so a run
  that stayed individual never grew it and mode A/B comparisons saw different boxes (the first growth multiplied the
  FMM solve time by ~10). The global path is unchanged (bitwise-identical steps, box and FMM counters from the TDE
  snapshot). The adaptive controller now expires throughput measured before a domain change (`NotifyDomainChanged`).
- Individual event-mesh build record (`HDSim3D::timeAdvanceIndividual`, under `RICH_INDIVIDUAL_PERF_TRACE`): one
  rank-0 `INDIVIDUAL_MESH_BUILD` line per event-mesh build with the result and its reason, the closure threshold,
  the per-rank target before and after closure growth (maximum, largest fraction of a rank's owned cells and that
  rank), the partial attempts, and the maximum over ranks of the whole call, the partial attempts and the full build.
  `RICH_INDIVIDUAL_PARTIAL_BUILD_FRACTION` (a number in (0, 1], agreed across ranks) overrides the per-rank closure
  threshold for measurements.
- Comprehensive documentation under `docs/`
- GitLab wiki pages under `wiki/`
- Regression test framework (`regression_tests/run_all.sh`)
- 14 regression tests: sod_1d, sedov_3d_mpi, till_compton, amr_random, voronoi_volume, lane_self_gravity, mach2_diffusion, mach2_multigroup, marshak_wave_1_diffusion-4, gresho_euler, gresho_lagrangian
- Regression result plotting (`regression_tests/plot_results.py`)
- LaTeX test report generation (`regression_tests/generate_test_report.py`)
- Marshak wave benchmarks (Problems 1-4) with self-similar analytical validation
- Gresho vortex tests (Eulerian and Lagrangian mesh)
- Multigroup diffusion radiation transport
- Compton Matrix Monte Carlo (CMMC) module
- `build_rich.sh` canonical build script with change detection
- `--build-subdir` option for parallel test builds

### Fixed
- Adaptive integration-mode gain bound (`Simulation::adaptiveGainBound`): the per-cell limits were collected after the
  global post-step callback, whose AMR pass (the TDE driver, every ten cycles) rebuilds the mesh without refreshing the
  hydro face velocities (`HDSim3D::CollectCellTimeStepLimits` reads the retained `face_vel_scratch_`). When the face
  count grew the hydro limits were dropped (hydro entered as a cap); otherwise old face velocities were indexed by new
  faces. The face velocities now carry the tessellation build generation they belong to; collecting on any other mesh
  fails as stale (`PhysicsStep::cellTimeStepLimitsStale`), the evaluation is skipped with `RICH_MODE_GAIN_SKIPPED`
  (unknown bound, no veto) and retried on the next global step. The bound is still evaluated after the callback, so it
  sees the post-callback state (the TDE sink update). Found by review, not observed in a run.
- Individual AMR (`AMR3D::Apply` via `ApplyIndividual`): the decision to skip the full rebuild when the current mesh
  already is the full owned mesh (`identity_mesh`) was taken per rank, before a collective build. A rank whose partial
  event-mesh target covered all its owned cells skipped `BuildParallel` while the others entered it, and the run hung
  (TDE, `RICH_INDIVIDUAL_PARTIAL_BUILD_FRACTION=1.0`, job 10208404, cycle 3189). The decision is now agreed across
  ranks; at the default threshold of 0.5 it could not be reached on a non-empty rank. The same per-rank gate in
  `Simulation::rebalanceCommittedIndividualState` (exact Hilbert rebalance) is now agreed across ranks as well, and
  also requires the owned index order, not only the point count.
- `RICH_INDIVIDUAL_PERF_TRACE` is agreed across ranks once before any rank uses it, so ranks with different
  environments can no longer disagree on the phase-timing `MPI_Gather`.
- Polyhedron clipping (`3D/tessellation/utils/PolyClip.cpp`, `clipFace`/`clipPolyhedron`): vertices on a clip
  plane were classified with a tolerance of 1e-12 of the face's edge length, below the rounding of the absolute
  coordinates far from the origin. After the TDE reference-frame change (cells at |x| ~ 170, sizes ~ 0.02) copies of
  one vertex held by different faces fell on opposite sides, a face vanished or a cap was wrong, and AMR removal
  remaps lost up to 0.9% of a removed cell ("Incomplete MPI neighboring-cell removal remap"); two representations of
  the same plane clipped the same cell to volumes 33% apart. Distances are now n.v - n.p with n.p formed once
  (independent of the point representing the plane); one on-plane band per clip, from the rounding of that
  expression and 1e-12 of the edge length; in/on/out classification with cuts only across strict sign changes,
  clamped to the edge; faces with no vertex strictly inside contribute only cap points; raw signs when the whole
  polyhedron lies within the band.
- AMR removal pairing (`AMR3D.cpp`, `AddNeighborRemovalPartners`): rigid-wall mirror points took the anchor flag
  of local cell 0 through the ghost exchange's default fill, so a candidate next to a wall could pair with a mirror
  and `Voronoi3D::GetOwner` threw on a point outside the box ("overflow (in 3D xyz->d)"). Mirrors now get an explicit
  zero and are skipped.
- TDE individual/adaptive driver (`runs/BaseTDEComptonIndividual`): the gravity reference-frame change aborted on
  the global path ("Reference-frame change requires one full synchronized mesh"); on the global path the state is now
  shifted in place exactly as `runs/BaseTDEComptonGlobal` does (bitwise-identical post-change state).

### Changed
- Individual timesteps: the mesh-drift, mass-loss and thermal-loss guards stop at a floor, the smallest cached hydro
  CFL/source limit over all cells (the step a global step would take); `RICH_INDIVIDUAL_GUARD_FLOOR` (default
  `apply`; `report`, `off`). On the TDE the thermal-loss guard held near-vacuum cells at the central sink's edge at
  0.02-0.05 of the global step; with the floor a trial from t = 20.82 ran 4.4x faster (0.62x global, was 0.14x) and
  ended closer to the global run's state. The crash-50107 restart (t = 0.141-0.20) passed with it, the drift and mass
  guards never below the floor there. `RICH_TDE_TERMINAL_OUTPUT_TIME` (TDE individual driver) writes a synchronized
  snapshot at a given time in either stepping mode and stops, for A/B comparisons from one restart.
- TDE individual driver (`runs/BaseTDEComptonIndividual`, `RemoveCenter`): the central sink acts per unit time in
  individual mode. Its factors (density x0.8, temperature x0.8 within [1e4, 1e7], velocity damping, the T > 1e9
  cooling outside the sink) are raised to the cell's closed interval over the step a global step would take
  (`HDSim3D::GetIndividualGlobalStepReference`, the guard floor's cached CFL/source minimum; reset on every global
  step). Applied once per event, the removal rate grew with a cell's activation rate: sink cells drained faster, their
  CFL step shrank, they took finer bins and drained faster still (a forced-individual TDE trial fell from 3.8e-4 to
  5.8e-5 sim/s). The inner temperature map max(1e4, min(1e7 x 0.8^(x-1), T x 0.8^x)) composes exactly under any split
  of the interval; the outer cooling is exact for whole numbers of steps. Global steps are unchanged bit for bit
  (checked on the TDE from snap_full_54).
- Adaptive integration controller: an individual probe decides only once its measured window covers the interval of
  the coarsest occupied bin (at most 4x its wall budget; `RICH_MODE_PROBE_EXTENDED`). A 20-event probe measured in a
  burst of closely spaced events had read 0.04x global where the previous probe read 0.78x.
- Individual timesteps: the mesh-deformation limit uses the closing speed of neighbouring generators,
  -(w_i - w_j).(x_i - x_j)/d, instead of their full relative speed, so pairs moving apart or sliding past each other
  no longer shorten a cell's interval; the default `RICH_INDIVIDUAL_MESH_DRIFT_FRACTION` is 0.25 (was 0.2).
  Tangential sliding is not bounded by this guard (user's decision; the mass-loss and conserved-change guards act on
  its consequences).  From the TDE snapshot at t = 15.03 (full gravity) a forced individual trial reached 0.97x the
  global throughput, against 0.074x with the previous rule (jobs 10205128 / 10205127).
- TDE individual/adaptive driver: after t > 5 the reference-frame check no longer aborts when the adaptive controller
  has just switched into individual mode (scheduler not yet initialized, job 10204954 at t = 6.05); it scans the
  completed global state, latches a triggered change and runs it after the new scheduler's first event.
- Adaptive integration mode: the potential-gain bound takes grey radiation's per-cell limits
  (`dt * 0.15 / diff_i` from the last global step, matched to the current cells by ID; unmatched cells set no
  radiation limit) instead of radiation's grid-wide cap (`PhysicsStep::collectCellTimeStepLimits` on
  `RadiationStep`, `RadiationDriver::lastCellTimeStepLimits` on `Diffusion`). With the cap every cell landed in
  bin 0 while radiation bound the step, so the bound was exactly 1 and the controller never probed individual
  mode (TDE, 2026-09-23/24). Global steps are unchanged. Rank 0 prints `RICH_MODE_GAIN` at every evaluation
  (the bound, the old uniform-cap form, per-step minima, bound cells, bin counts); `RICH_MODE_DECISION` gains
  `gain_bound_uniform_caps`.
- Adaptive integration mode: new driver hooks `Simulation::AdaptiveIntegrationModeWillEnable`,
  `SetGlobalPostStep` (collective update after every global step, before the controller acts) and
  `SetAdaptiveDecisionsDeferred` (no controller decision on a global step while set). The TDE driver
  (runs/BaseTDEComptonIndividual) uses them: a fresh run starts on the global path when the controller is on
  (`RICH_TDE_START_MODE=individual|global` overrides; restarts keep the checkpoint's mode), writes no initial file
  on a global start, applies its center sink and full-gravity AMR after global steps as runs/BaseTDEComptonGlobal
  does, and defers controller decisions until the early snapshot is written. A 77-event individual start had cost
  2.07x the global steps to t=-1.20 (a faster, hotter surface layer); started global, the controller took the
  global driver's exact logged step sequence (S-B discriminator, job 10204569). The global fork gains
  `RICH_TDE_EARLY_OUTPUT_CLAMP=1` (default 0) to land a step on its early snapshot like the individual driver.
- Individual timesteps: the mesh-deformation limit `RICH_INDIVIDUAL_MESH_DRIFT_FRACTION` defaults to `0.2`
  (was `0`), and rank 0 prints the effective guard settings once (`INDIVIDUAL_GUARDS`). A RoundCells kick held
  over a long interval overshoots the centroid and the following kicks can drive neighbouring generators through
  each other within one legal interval (negative-mass abort, cell 50107, 2026-09-22); with the limit the same
  restart passed the abort point (early robustness gate; the run to t = 0.46 and the regression suite are pending)
  at 0.61x the events and 0.94x the step wall over the matched window t = 0.1412-0.1747
  (docs/fixes/individual-timesteps-crash-50107-2026-09-23.md).
- TDE drivers: `RICH_TDE_UPDATE_BOX=1` grows the box with the legacy `UpdateBox` (every 7 cycles, after global
  steps only; rank 0 prints `RICH_UPDATE_BOX` with its seconds and cell counts); the fixed +-5 rigid box let the debris
  pile into the corners from t ~ 0.6 and pin the step near 3e-5. `RICH_TDE_RESTART_FROM_SNAPSHOT=1` (individual driver)
  restarts from the snapshot alone, globally from `init_dt`, as the global fork does. Both default off.
- FMM (`DistributedFmmGravityCalculator::prepareLocalTree`): a retained local root must lie on the current domain's
  lattice; after a domain change a rank could keep its old root and publish a stale lattice id ("invalid or stale LET
  source root"). New `fmm_gravity_mpi` scenario `domain_growth` (same bodies, larger domain) requires every occupied
  rank to rebuild; it reports 1 with the fix and 0 without.
- TDE driver (runs/BaseTDEComptonIndividual): `RICH_TDE_WRITE_VTU=0` skips the per-rank ParaView files written
  with every snapshot (default: written, as before; about 55% of the snapshot write time and 2.5x the HDF5 size), and
  rank 0 prints `RICH_OUTPUT` (snapshot and checkpoint seconds) for every numbered output.
- Diagnostics (default off): `RICH_CFL_DECISION_TRACE` prints which criterion (raw CFL, source limit or the
  previous step's cap) sets each global-step CFL evaluation, with the winning cell; the TDE driver's
  `RICH_TDE_CFL_DEBUG=1` enables `CourantFriedrichsLewy`'s verbose winner dump (now guarded against ranks with no
  eligible cell).
- Individual timesteps (diagnostics, default off): per-cell trace `RICH_INDIVIDUAL_TRACE_CELL_IDS`, wake-limiter
  sub-phase timers under `RICH_INDIVIDUAL_PERF_TRACE`, and the grey radiation relative-increment limit
  `RICH_INDIVIDUAL_RADIATION_INCREMENT_LIMIT`.

### Added
- Individual timesteps: mass-loss timestep limit and conserved-change wake
  (`RICH_INDIVIDUAL_MASS_LOSS_FRACTION`, `RICH_INDIVIDUAL_WAKE_CHANGE_FRACTION`),
  and a consistent explicit face step (interval-start primitives and mesh
  velocities for the fluxes, velocity choice after the update).
- Individual timesteps: mesh-deformation timestep limit
  (`RICH_INDIVIDUAL_MESH_DRIFT_FRACTION`), bounding how far a generator travels
  relative to a neighbouring generator within one interval, and a per-event
  step audit reporting intervals that outlasted the limit in force when they
  opened (`step_overruns`).

### Added
- Adaptive integration mode (`Simulation::SetAdaptiveIntegrationMode`,
  `RICH_INDIVIDUAL_ADAPTIVE_MODE`): the run switches between the global path
  and individual timesteps from the measured throughput of each mode and an
  upper bound on the individual gain computed from the per-cell CFL
  distribution (`CourantFriedrichsLewy::CellTimeSteps`,
  `PhysicsStep::collectCellTimeStepLimits`). Switches happen at synchronized
  states only; rank 0 reports `RICH_MODE_DECISION` and `RICH_MODE_SWITCH`.
  `Simulation::StateSynchronized()` covers both modes and
  `RequestSynchronizedIndividualEvent()` is a no-op in global mode.

### Changed
- Individual timesteps: the interval-start mesh is no longer built when every
  rank's source term can apply its first half from cached per-cell state
  (`SourceTerm3D::IndividualFirstHalfNeedsGeometry`,
  `ApplyIndividualFirstHalfFromCache`; `RICH_INDIVIDUAL_FIRST_HALF_FROM_CACHE=0`
  restores the build). The fluxes use the event mesh only, and a conservative
  force's first half is a kick from the cached acceleration, so on the TDE
  this build was about half of every event's mesh work.
- Individual timesteps: the partial Voronoi target is seeded with the two-cell
  reconstruction shell from a per-cell adjacency cache before the build
  (`RICH_INDIVIDUAL_ADJACENCY_SEED=0` disables it). The closure check used to
  discover that shell after the build and rebuild, two to three builds per
  mesh on the TDE.
- Individual timesteps: with the adjacency seed on, the previous accepted
  target is no longer added to the partial target as a warm start (it could
  only grow between full builds). A whole-mesh closure threshold is available
  behind `RICH_INDIVIDUAL_PARTIAL_THRESHOLD_GLOBAL=1` but measured worse and
  stays off. A rejected radiation candidate may be kept from lowering hydro
  bins (`RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS=0`, default unchanged),
  the radiation then sub-cycling inside the event as the global scheme does
  inside its step.
- `RICH_VERIFY_PARTIAL_BUILD` is honoured by the hydro step at every partial
  build, so a restart whose checkpoint carries `verify_partial_build = false`
  can still be run in parity mode.

### Fixed
- Individual timesteps: an interval no longer outlasts the allowance of the bin
  it carries. Lowering a cell's bin part way through an interval clipped the
  end only to the next tick aligned to the new bin, which is frequently the end
  the cell already had, so the bin was relabelled while the interval kept its
  old length. `INDIVIDUAL_BIN_OVERRUN` reports any remaining overrun.
- Distributed Voronoi ghost search after builds with the point exchange
  suppressed (the individual-timestep mode). `HilbertPointsManager` now routes
  sphere–rank queries through a distributed oct tree of actual point positions
  once an exchange has been suppressed, instead of the nominal Hilbert ranges
  that no longer describe ownership after the mesh drifts. Regression case
  `suppressed_exchange_ghosts`.

### Changed
- Build system migrated from Make to CMake
- MPI tests submitted via SLURM instead of direct `mpirun`

---

## Version History

### Original Publications

- **Serial RICH**: Yalinewich, Steinberg & Sari (2015), [ApJS 216, 35](http://iopscience.iop.org/0067-0049/216/2/35/)
- **Parallel RICH**: Steinberg, Yalinewich & Sari (2015), [ApJS 216, 14](http://adsabs.harvard.edu/abs/2015ApJS..216...14S)

---

*To add entries: describe the change under the appropriate section (Added, Changed, Fixed, Removed) in `[Unreleased]`. When a version is released, rename `[Unreleased]` to the version number with the date.*
