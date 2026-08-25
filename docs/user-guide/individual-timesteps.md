# Individual timesteps

RICH supports power-of-two individual timesteps for Newtonian, Cartesian,
three-dimensional hydrodynamics with supported gravity sources, grey radiation
diffusion, and multigroup radiation diffusion. The implementation works in
serial and MPI builds, supports local AMR, and stores the scheduler state in
HDF5 restarts.

Monte Carlo radiation transport is deliberately excluded. Adding a
`RadiationMCStep` to an individual-timestep simulation, or enabling individual
timesteps after adding one, throws during setup. This includes IMC and DDMC.
Multigroup diffusion with Compton and Doppler terms remains supported; it is
not Monte Carlo transport.

## Quick start

Configure the normal physics steps first, set the initial global timestep, and
then enable individual timesteps:

```cpp
auto hydro_step = std::make_shared<HydroStep>(hydro, HydroStep::TIMEADVANCE_2);
auto radiation_step = std::make_shared<RadiationStep>(
    tess, simulation.getCells(), simulation.getExtensives(),
    simulation.getTracker(),
#ifdef RICH_MPI
    nullptr,
#endif
    diffusion, true);

simulation.addPhysics(hydro_step);
simulation.addPhysics(radiation_step);

double const initial_dt = 1e-6;
simulation.SetTimeStep(initial_dt);

IndividualTimeStepOptions options;
options.initial_bin = 30;
options.maximum_bin = 40;
options.maximum_neighbor_bin_difference = 2;
options.mesh_build_policy = IndividualMeshBuildPolicy::AutoPartial;
simulation.EnableIndividualTimeSteps(options);

while(simulation.GetTime() < final_time)
    simulation.step();
```

`HydroStep::TIMEADVANCE_2` is required. The first call to `simulation.step()`
initializes the scheduler from the current owned cells and the timestep set on
the simulation.

Registering physics before `EnableIndividualTimeSteps` is recommended because
unsupported steps fail together during setup. RICH also checks steps added
after individual mode has been enabled.

## Options and timeline

`IndividualTimeStepOptions` has these defaults:

| Option | Default | Meaning |
|---|---:|---|
| `time_quantum` | `0` | Exact timeline quantum. A non-positive value derives `initial_dt / 2^initial_bin`. |
| `initial_bin` | `30` | Initial bin assigned to every cell. |
| `maximum_bin` | `40` | Largest permitted bin; it must not exceed 62. |
| `maximum_neighbor_bin_difference` | `2` | Largest bin difference allowed across a face. |
| `mesh_build_policy` | `AutoPartial` | Use partial Voronoi construction when its closure is small enough. |
| `partial_build_fraction` | `0.5` | Fall back to a full build when the partial target exceeds this fraction of owned cells. |
| `verify_partial_build` | `false` | Build a full reference mesh and compare active geometry before continuing. Intended for tests and debugging. |
| `force_synchronized` | `false` | Keep all owned cells on one shared, adaptively selected bin. This is the full-mesh synchronized oracle used by comparison tests. |

For quantum `q` and bin `b`, the represented timestep is

```text
dt(b) = q * 2^b.
```

Cell times are stored as 64-bit integer ticks. `CellTimeState` records the
stable cell ID, begin and end ticks, last primitive-state tick, bin, predicted
generator velocity, cached gravitational acceleration, and pending gravity
half-kick state. Floating-point time is reconstructed only at the physics
interface.

Choose `time_quantum` small enough to represent every physical limit expected
in the run. If a hydro, gravity, or radiation limit is below one quantum, RICH
throws instead of silently taking an unsafe step. For example:

```text
Individual timestep limit ... is below the configured time quantum ...
```

Reduce `time_quantum`, or increase `initial_bin` while preserving the desired
initial timestep.

## Event scheduler

One call to `Simulation::step()` advances one synchronization event:

1. Find the minimum cell end tick. MPI uses a global minimum reduction, so a
   rank with no locally active cell still enters the same event.
2. Mark cells ending at that tick active and construct an
   `IndividualStepContext` containing exact ticks, per-cell intervals,
   primitive timestamps, predictors, and the active mask.
3. Run each registered physics step for the event. Conserved updates may be
   deposited into passive cells, but passive primitives remain at their last
   activation time.
4. Evaluate the optional AMR callback for active cells and remap scheduler
   state using stable IDs.
5. Ask every physics step for per-active-cell timestep limits and commit the
   event to the scheduler.
6. Apply the neighbor-bin limiter, including MPI neighbors. A newly shorter
   signal can wake an inactive cell before its old end tick.
7. Advance the simulation clock to the event time and report the next event
   interval.

A bin decrease takes effect immediately. A bin can increase by at most one
level per activation, and only when the event tick is aligned with the larger
power-of-two interval.

For an exact requested output endpoint, call the scheduler's terminal clamp
before the next event whenever a cell interval crosses that endpoint.  The
benchmark does this automatically: every crossing interval ends at one common
terminal tick, the event context carries the exact shortened cell interval, and
all owned cells refresh their primitive state before the snapshot.  The log
record is `INDIVIDUAL_TERMINAL_EVENT_CLAMP`.  This one terminal synchronization
does not change bin selection during ordinary evolution.

## Partial Voronoi construction

`AutoPartial` is the production default. A full tessellation is still the
correctness oracle and fallback.

At an event, the hydro step:

1. Predicts every mesh-generating point to the event time without committing
   passive primitive states.
2. Seeds the target with active cells and the previous reconstruction halo.
3. Calls `BuildPartially` in serial or `BuildPartiallyParallel` in MPI. All
   predicted generators remain eligible geometric neighbors even though only
   target cells are requested.
4. Uses `ActiveMeshView` to maintain immutable mesh-local-to-owned and
   owned-to-mesh-local mappings.
5. Expands the target to reconstruction depth two until active faces, adjacent
   generators, cell geometry, and reconstruction stencils are closed. In MPI,
   a newly encountered remote endpoint is requested by its owner's canonical
   cell index and becomes a target on that owner before the event mesh is
   accepted. This prevents a new cross-rank face from using a stale passive
   centroid.
6. Falls back collectively to a full build for an oversized closure, mapping
   failure, geometric failure, or debug parity mismatch.

The default closure threshold is 50% of owned cells. Large active fractions
therefore avoid paying partial-build overhead when a full build is more
appropriate.

Useful timing and fallback messages include:

```text
Individual generator prediction ...
Individual partial Voronoi build ...
Individual full Voronoi build (closure threshold) ...
Individual partial Voronoi fallback: debug parity mismatch
```

Set `options.verify_partial_build = true` while developing mesh or
reconstruction changes. It compares active neighbor IDs, cell volume and
centroid, face geometry, reconstruction data, and resulting flux inputs with a
full reference build.

## Hydrodynamics and gravity

Hydrodynamics remains second order in space and time:

- A face is processed when either adjacent cell is active.
- The face interval is `min(dt_left, dt_right)`.
- Reconstructed states are temporally predicted in the moving-face frame.
- Each face is evaluated once. Equal-and-opposite conserved fluxes are applied
  to both cells, including passive recipients.
- Advected radiation groups use a donor-based positivity limiter. For each
  owned cell and group, total outgoing hydro transport is bounded by the
  conserved energy present before the event; same-event inflow is not treated
  as spendable energy. Any limited transfer applies the exact opposite
  correction to the face neighbor, including across MPI ranks by stable ID.
- Primitive recovery is performed only for active cells after incoming
  conserved updates have been assembled.

In MPI, an inter-rank face has one deterministic owner. Conserved deltas are
sent by stable cell ID, which prevents duplicate face work and avoids relying
on transient local indices.

Supported conservative gravity sources use active-target acceleration
evaluation and kick-drift-kick integration. The acceleration and half-kick
phase are cached per cell. Gravity limits participate in bin selection and can
wake a passive cell.

Gravity monopoles remain at cell centroids, matching the global integrator.
At an event, cached passive centroids are predicted with the generator velocity
and active centroids are replaced by the newly constructed exact geometry. This
same policy is used by `FullReference` and `AutoPartial`; a full build must not
silently refresh passive centroids that a partial build cannot see. The passive
centroid drift is an explicit predictor approximation until that cell becomes
active. After hydro communication has committed equal-and-opposite face deltas,
the source-mass array is rebuilt from every canonical owned conserved mass
before the second gravity kick. This includes passive recipients outside the
partial reconstruction closure.

## Grey and multigroup diffusion

Radiation diffusion is local to the event rather than a disguised global
solve.

Both grey and multigroup timestep limiters form their normalization scales from
passive canonical owned state plus the current active event state, followed by
an `MPI_MAX`. The partial reconstruction closure is never used as the reference
population, and ranks with no owned or active cells contribute a neutral zero.
This keeps `FullReference`, `AutoPartial`, and the synchronized global oracle on
one normalization definition. The legacy global grey limiter retains its
`1.25*dt` growth cap, the global multigroup limiter retains `1.4*dt`, and the
individual scheduler retains its one-bin `2.0*dt` cap.

For grey diffusion, the implicit matrix has one row per active cell. For
multigroup diffusion it has `Nactive * Ngroups` rows:

- active-active faces couple two unknown rows;
- active-passive faces place the passive primitive radiation state on the
  right-hand side;
- passive-passive faces are skipped;
- face diffusion uses the face interval;
- absorption, emission, scattering, material coupling, radiation force,
  Compton, and Doppler terms use the active cell interval.

Grey diffusion retains the immediate conservative behavior: the solve commits
active primitive states and applies the opposite face-energy transfer to
passive conserved radiation extents without refreshing passive primitives.

Multigroup individual diffusion selects one MPI-consistent passive-boundary
policy with `RICH_MG_INDIVIDUAL_PASSIVE_POLICY`:

- `legacy` is the library default. It retains the immediate equal-and-opposite
  passive conserved commit.
- `shadow` enables the owner-held conservative shadow-reservoir experiment.
  It is not the production AutoPartial policy.
- `dirichlet` freezes passive primitive group energies in the active-row
  right-hand side and commits only active state. Passive primitive and
  conserved radiation state remain unchanged during that candidate.

The deprecated `RICH_MG_INDIVIDUAL_SHADOW_RESERVOIRS` flag remains an alias
for `shadow`; conflicting selectors are rejected collectively. None of these
settings changes the synchronized global radiation path. Flux limiters,
boundary conditions, cooling limits, hydro feedback, all energy groups,
Compton safeguards, and Doppler terms remain active.

Dirichlet mode deliberately omits the equal-and-opposite passive interface
transfer. Before commit, it measures that omitted transfer from the same face
coefficients and final active unknowns used by the candidate. Accepted events
append signed, absolute, normalized, and worst-local contributions to a
rollback-safe defect ledger. Collective defect limits can reject the candidate;
a rejected candidate changes neither passive state nor committed defect
accounting. Therefore Dirichlet mode is measured-defect conservative, not
exactly conservative across active--passive interfaces.

Each decision emits `INDIVIDUAL_RADIATION_DEFECT status=accepted|rejected`.
The scheduler-owned ledger, its versioned limits, retry/cooldown state, and
`history_complete` flag are checkpointed. Benchmark restart fingerprints and
`counters.txt` print those fields so a restarted energy audit can distinguish
a complete accumulated history from a reconstructed default.

Radiation uses backward Euler and is first order in time. Each candidate is
transactional for active cells, any passive recipients used by its policy, and
pending repair/defect accounting. A failed
positivity, Fleck-factor, diagonal, or solver check restores the candidate and
halves its fraction of the scheduled interval and lowers an event-local safe
ceiling to that fraction. Accepted candidates remain at or below this ceiling.
After eight consecutive accepted candidates, the ceiling can double once;
another rejection lowers it again and restarts the cooldown. The next candidate
is the minimum of twice the accepted fraction, the safe ceiling, and the
uncovered remainder. This avoids re-solving the same known-bad larger Compton
candidate after every accepted fragment while still recovering from a transient
restriction.
Accepted fractions accumulate until the interval is covered. There is no
configurable retry or fraction cap; halving stops with a controlled error only
when floating-point resolution can no longer advance either the completed
fraction or the candidate end time. When a rejection is explicitly classified
as cell-local and reports a stable cell ID, only that failed cell receives the successful retry fraction as its next
radiation limit. Multiple failed IDs in one event retain their own smallest
fractions. A failure without a cell attribution still caps the complete active
set at the event's minimum fraction; named matrix, face, and transactional
failures remain collective unless their solver explicitly declares otherwise. This keeps collective failures safe without
letting a cell-local Compton failure resynchronize every active cell.

The legacy global radiation wrapper uses the same eight-acceptance cooldown in
collective form. It no longer aborts when a candidate falls below the old
hard-coded `0.1%` fraction. A focused unit forces 11 rejections, then verifies
  that the eight-acceptance cooldown recovers geometrically and 65 accepted
  substeps cover the original interval exactly. A second
focused unit verifies both the stable-ID cell-local limiter and the unattributed
active-set fallback.

A Compton-induced multigroup failure takes one controlled split retry before
that interval is halved. The first solve repeats diffusion, free-free
absorption/emission and scattering, and Doppler coupling over the scheduled
interval with Compton excluded from the Fleck factor and matrix. The second
solve applies Compton cell-locally to that accepted state. Its Fleck factor
contains only the Compton derivative term,
`f_C = 1 / (1 + c * dt * beta * Upsilon)`; it contains no absorption `Gamma`
term. If all occupation-number choices give an invalid `Upsilon`, RICH first
keeps the finite Compton source `S` but sets `dS/dUm = 0`, so `Upsilon = 0` and
`f_C = 1`. Only failure of that frozen-Jacobian solve halves the local Compton
substep. Accepted local substeps accumulate until they cover the original
scheduled interval; failure never commits a partial candidate.

`MG_COMPTON_ABSORPTION_ONLY_RETRY` and `MG_COMPTON_ONLY_FLECK` record the split,
the two Fleck compositions, selected occupation mode, `Upsilon`, and whether
the Jacobian was frozen.

Grey diffusion refreshes candidate-local radiation and material baselines from
the latest accepted state, then rebuilds its CGS cells, diffusion and
Planck/scattering coefficients, Fleck factors, limiter data, matrix, and
right-hand side for every fractional candidate. Its event-start radiation and
temperature arrays remain separate and fixed for timestep feedback. A focused
test forces one rejection and compares the two accepted fractional intervals
against two explicitly sequenced half steps. Multigroup diffusion performs the
equivalent refresh in its candidate-preparation hook.

In legacy immediate-conservative mode, after the equal-and-opposite
active--passive face correction, a group extent
can be negative only below the resolution of the cell's much larger total
radiation extent. Such deficits are repaired only when their sum is no larger
than `512 * solver_tolerance` times the larger of the total extent and the
sum of absolute group extents. The repair sets the affected groups to zero and
withdraws the same extent from a representable positive group in that cell;
`Erad` and material energy are unchanged. A larger deficit remains a rejected
transaction and is subcycled. Every accepted repair emits
`MG_SPECTRAL_POSITIVITY_REPAIR` with cell/group counts, redistributed extent,
maximum relative deficit, representative rank/cell/negative group/donor group,
and the floating-point conservation residual.

On that legacy path, a passive group may also enter radiation with a tiny
negative residue left by
opposing conservative hydro deltas. The face limiter accepts it only when its
magnitude is at most `512 * epsilon` times the current positive-plus-negative
face-transfer scale. Capacity is then computed from zero, and the existing
equal-and-opposite residual correction restores the passive group exactly.
Larger pre-existing negatives still reject the transaction. Accepted repairs
emit `MG_PASSIVE_ROUNDOFF_REPAIR` with the global value count and the
representative rank, cell, group, extent, transfer scale, and relative size.

When retries occur, RICH reports one aggregate count plus one representative
reason and stable cell ID:

```text
Individual radiation retries: N, representative reason: ..., cell ID ...
```

After an accepted solve, grey and multigroup drivers return per-active-cell
limits to the shared scheduler.

### Multigroup cell-block preconditioner

Multigroup diffusion uses cell-block Jacobi by default. For each active or
owned cell, RICH extracts the complete same-cell `Ngroups x Ngroups` block from
the assembled matrix. The block includes the diffusion self diagonal and all
assembled absorption/emission, material/Fleck, scattering, Compton, and Doppler
group couplings. Entries coupling different cells remain in the Krylov matrix
but are deliberately excluded from the preconditioner.

Each block is row-equilibrated and factored in row-major storage by LU with
partial pivoting. The relative pivot threshold is `64 * epsilon`. RICH never
forms an inverse and never regularizes a failed block. A finite singular or
unsafe block falls back to the historical scalar Jacobi operation for that cell;
an invalid scalar diagonal rejects the radiation candidate. Factor storage is
released after the candidate and is not retained between radiation cycles.

The historical scalar diagonal defines the convergence metric and `max0` and
`max1`. The block factorization is used only for the two BiCGSTAB direction
solves. MPI ranks communicate while applying the matrix and updating ghost
vectors exactly as before; applying a block factor requires no new MPI
communication.

Every solve reports preconditioner setup, representative fallback, apply cost,
iteration progress, convergence, and timing split into matrix-vector,
exchange, reduction, setup, apply, and total time. Progress and convergence
records include `max0`, `max1`, negativity, rank, stable cell ID, and runtime
group index. The iteration loop remains fixed at 10,000 iterations.

The active solver solves for a correction about the event-start state,
`x = x0 + delta`, with `delta = 0` initially. It forms
`A_active,active*delta = b_reduced - A_active,active*x0`; passive-neighbor terms
and row sums use compensated accumulation. This avoids subtracting two large
physical states inside every Krylov iteration when the required update is tiny.

Global and distributed-active MG now use the same historical final acceptance.
From a recomputed residual, they evaluate

```
error = sum_i(r_i^2 / A_ii) / sum_i(b_i^2 / A_ii)
```

with `r = b - A*x` in the coordinates of the current solve. The configured
`1e-11` is the tolerance on this squared, diagonal-scaled ratio; its displayed
effective norm tolerance is therefore `sqrt(1e-11)`. The established normal,
loose-maxima, tiny-breakdown, and extremely-small-error branches retain their
historical `max0`, `max1`, negativity, and minimum-iteration conditions. The
distributed path uses MPI reductions for every scalar entering that decision,
so rank count and active-row distribution cannot change the criterion.

RICH also recomputes and reports the componentwise backward error

```
eta_inf = max_i |r_i| / max(|b_i| + sum_j |A_ij|*|x_j|, safe_min(group_i))
```

using an `MPI_MAX`. This is a diagnostic, not an additional acceptance gate.
Nonfinite solutions or residuals and all physical checks still reject the
candidate. The fixed 10,000-iteration safety limit is unchanged.
`MG_BICGSTAB_TOLERANCE`, `MG_BICGSTAB_HISTORICAL_POLICY`, progress,
convergence, and worst-row records retain the historical error, `max0`, `max1`,
negativity, componentwise diagnostic, representative rank/cell/group, row
scale, maximum row nonzeros, and timing fields.

A synchronized event does not need the reduced active-row system. If every
globally owned cell is active, the active mesh has a validated canonical
mapping, and all cell and event intervals agree, grey and multigroup diffusion
solve the candidate with their established global matrix paths. The check is
collective, including ranks with zero owned cells. Candidate preparation,
transactional rollback, fractional retry, and post-event timestep selection
remain unchanged. The
`MG_INDIVIDUAL_ALL_ACTIVE_FAST_PATH` diagnostic records the global active count,
scheduled interval, fraction, and actual candidate timestep. Any sparse event
or partial mapping stays on the active-row solver. This shortcut is an explicit
driver capability. The MPI normalization gates include synchronized grey and
multigroup cases with at least one zero-owned rank.

Callers can request the old method explicitly with
`CG::PreconditionerKind::ScalarJacobi`. The coupled benchmark also accepts
`RICH_TEST_MG_PRECONDITIONER=scalar`; this environment control is test-only and
does not replace the constructor interface.

The unit gate covers runtime group counts 1, 2, 3, 7, and 16, duplicate sparse
entries, neighbor exclusion, exact block residuals, aliasing, singular fallback,
and factor release. Physical A/B gates cover global, serial-active, and
distributed-active solvers at 1, 8, and 128 MPI ranks. The retained results and
exact acceptance metrics are in
`regression_tests/cases/mg_cell_block_preconditioner/BENCHMARK_RESULTS.md`.

## Monte Carlo is not supported

`RadiationMCStep` explicitly declares individual timesteps unsupported. Both
of these orderings throw `std::invalid_argument` during setup:

```cpp
simulation.addPhysics(mc_step);
simulation.EnableIndividualTimeSteps();
```

```cpp
simulation.EnableIndividualTimeSteps();
simulation.addPhysics(mc_step);
```

The diagnostic is:

```text
Physics step 'radiation-mc' does not support individual timesteps:
Monte Carlo radiation transport, including IMC and DDMC, requires global timesteps
```

To run IMC or DDMC, keep the global integrator and continue setting the shared
timestep with `Simulation::SetTimeStep` or the existing global timestep
function. Do not call `EnableIndividualTimeSteps`.

Compton coupling inside `MultigroupDiffusion` is supported. The prohibition
applies to packet-based `RadiationMCStep`, not to the Monte Carlo tables used
internally to precompute multigroup Compton coefficients.

## Local AMR

Register event-local AMR with `Simulation::SetIndividualAMR`. The callback
receives the current `IndividualStepContext` and should act only on active
cells. It returns an `IndividualAMRChangeSet` keyed by stable IDs.

Refinement and derefinement conservatively remap mass, momentum, material
energy, grey or multigroup radiation energy, and registered extra conserved
extents. New children receive new stable IDs and initially inherit the parent
tick and bin. Derefinement may deposit conserved extents into passive
neighbors. The scheduler, predictors, acceleration cache, radiation metadata,
and active-mesh maps are remapped after the topology change.

Distributed second-order AMR neighbor requests use three distinct index
spaces. MadVoro duplicate-point entries are compact all-point indices; they are
converted first to the original build-input index and then through the inverse
owned mapping to a local mesh index. Ghost ownership comes from the paired
ghost/duplicated-rank metadata, not a spatial owner query. Mapping failures are
validated collectively before recursive neighbor traversal.

## Restarts

Write snapshots only between completed calls to `Simulation::step()`. The
snapshot stores canonical conserved extents plus a versioned
`/individual_time_steps` group containing:

- time origin, quantum, current tick, and bin-policy options;
- stable cell IDs, begin and end ticks, primitive timestamps, and bins;
- point velocities, acceleration cache, and gravity half-kick phase.

Restart block version 2 also stores `force_synchronized`. Version-1 snapshots
remain readable and default that option to `false`, preserving their original
variable-bin behavior. Parallel snapshots store the scheduler block beside
each rank's canonical owned cells; every rank, including an empty one, restores
its own stable-ID keyed state.

The hydro step also stores the stable IDs of the last partial-mesh target so
the reconstruction halo can be restored exactly before the first resumed
event. Old snapshots without individual-timestep metadata resume in global
mode unless they are explicitly converted and synchronized.

Bitwise uninterrupted-versus-restarted equality remains the serial acceptance
criterion. The focused 128-rank HDF5 gate separately checks that a written
snapshot reloads the canonical conserved state, complete scheduler state, and
`force_synchronized` bit exactly. It then checks exact next-event ticks, bins,
active sets, and scheduler hashes after continuing both runs. Floating MPI
state after the continued event is compared to solver tolerance rather than
bitwise because a fresh process rebuild can change tessellation and reduction
ordering.

## MPI behavior

MPI individual mode adds these collective rules:

- the next event tick is the global minimum;
- ranks without active cells still enter mesh, physics, retry, and acceptance
  collectives;
- ranks with zero owned cells also enter grey radiation-force exchanges and
  reductions with neutral local maxima; no empty vector is dereferenced;
- partial parallel construction suppresses routine load balancing but retains
  required ghost exchange;
- distributed partial closure exchanges owner-canonical target requests to a
  fixed point; ranks with no local additions still enter every rebuild and
  closure reduction;
- inter-rank hydro and radiation face deltas use deterministic face ownership
  and stable destination IDs;
- grey and multigroup solvers build distributed active-only row maps and
  exchange only required remote active values;
- convergence, positivity, retry, and event acceptance are collective;
- the forced-active threshold and persistent-latch selector are validated
  collectively, and a restored latch must agree on every rank before physics;
- explicit load-balancing events migrate scheduler, predictor, acceleration,
  conserved, and radiation state.

## Regression controls

Production setups should use the C++ API. Several regression cases also accept
environment variables for focused testing:

| Variable | Values or effect |
|---|---|
| `RICH_INDIVIDUAL_MODE` | `full` for one adaptive synchronized bin, `full-variable` for variable bins with full meshes, or `partial` for variable bins with `AutoPartial`. |
| `RICH_TEST_SPARSE_INITIAL_BIN` | Makes one initial cell faster, producing active-passive faces immediately. |
| `RICH_TEST_SPARSE_MAX_ER_CELL` | Test-only: makes the cell owning the global radiation reference maximum the sparse active cell. |
| `RICH_TEST_INITIAL_BIN`, `RICH_TEST_MAXIMUM_BIN` | Override the power-of-two scheduler bounds in focused and calibration runs. |
| `RICH_TEST_TIME_QUANTUM` | Overrides the exact integer-timeline quantum. |
| `RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_MIN_BIN` | Default off. Before physics, compare this threshold with the largest actual active-cell interval, measured in scheduler ticks. A qualifying partial event promotes every owned cell without changing `maximum_bin`. |
| `RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_LATCH` | `0` by default. With value `1` and a configured minimum bin, reaching the threshold sets a monotonic collective latch. The threshold event and every later partial event use the all-active overlay; naturally all-active events need no promotion. |
| `RICH_VERIFY_PARTIAL_BUILD` | Enables full-versus-partial geometry parity. |
| `RICH_TEST_POINT_COUNT` | Overrides the case size. |
| `RICH_TEST_MAX_CYCLES` | Bounds focused runs by accepted events. |
| `RICH_TEST_COMPTON_MATRIX_SAMPLES` | Reduces Compton precomputation only in focused tests. |
| `THUNDER_ARTIFACT_DIR` | Sends profiles and other test artifacts to a chosen directory. |

Forced activation is a transient event overlay. Each promoted cell advances
only from its own accepted `begin_tick` to the common event tick. Cell schedule
state changes only if the complete event succeeds and is committed. Its next
bin grows from that completed shortened interval, not from the cell's previous
longer schedule. The monotonic safety latch is the exception: it is set before
physics and is not cleared by a retry or rollback. The latch does not lower
`maximum_bin`; an uncapped bin-40 run can therefore continue to grow on
all-active events.

Individual-timestep checkpoint format version 7 stores the monotonic latch in
every rank piece. Versions 1--6 reconstruct it as false. A restored true latch
remains authoritative even when the runtime selector is absent. A pre-latch
checkpoint contains false, so its minimum-bin and latch policy must still be
re-exported when the run continues. The production runner records both values
per segment, recovers them from the preceding segment when a restart environment
omits them, and rejects an explicit policy change within one campaign root.

### Scheduler, geometry, and MC guard

The dedicated unit target covers integer scheduling, immediate bin decreases,
aligned one-level increases, neighbor wake-up, AMR state inheritance, restart
ticks, the MC setup error in both configuration orders, and partial geometry.
The geometry test uses unmoved and perturbed generators with active fractions
of 1%, 5%, 10%, 25%, 50%, and 100%.

Build and run it with GCC 12:

```bash
ml gcc
ml gcc/12.3.0
ml openmpi/4.1.6/gcc/12.3.0
ml hdf5/1.14.2/gcc/12.3.0_cxx

./build_rich.sh gnuDebug \
  --test_name=tests/newtonian/three_dimensional/individual_time_steps \
  --build-subdir=individual_time_steps --jobs=8

cd tests/newtonian/three_dimensional/individual_time_steps
../../../../build/gnuDebug/individual_time_steps/rich_gnuDebug
test -f test_passed.res
```

### Physics and integration coverage

The implementation was exercised with these existing cases:

| Area | Cases and checks |
|---|---|
| Hydro and grey diffusion | `mach2_diffusion`; synchronized-bin oracle, variable bins, active-passive faces, finite profile, conservation. |
| Hydro-carried radiation positivity | `individual_time_steps`; requests twice a donor's available group energy, then checks every extent is nonnegative and both group and total radiation are conserved. |
| Multigroup diffusion | `mach2_multigroup`; all groups, distributed active rows, finite profile, conservation. |
| Free-free coupling | `eulerian_diffusion_freefree_1d` and `eulerian_diffusion_freefree_multigroup_1d`; sparse partial events and finite 64-cell profiles. |
| Marshak diffusion | `marshak_wave_1_diffusion`; variable-bin radiation events and finite profile. |
| Gravity | `lane_self_gravity`; active-target gravity, kick phases, MPI empty-active ranks. |
| AMR | `amr_random`; local refine/derefine, stable scheduler remap, conserved totals. |
| Compton and Doppler | Multigroup Compton Marshak focused runs; sparse active rows, collective retry, positivity, finite output. |
| Restart and load balance | Uninterrupted versus split/resumed runs and scheduler migration on 2, 4, and 8 ranks. |

The current focused review gates use the production 16-group configuration.
They cover synchronized grey and MG through the all-active global-solver
capability, sparse FullReference and AutoPartial MG, globally reduced grey
normalization, grey DiffusionForce with zero-owned ranks, grey fractional-candidate refresh,
moving-generator gravity parity on a one-active-rank cross-rank face,
distributed partial AMR with nonidentity mappings, and a real HDF5 restart. The
gravity comparison checks both acceleration and conserved energy after proving
that a passive-rank mass changed.
The retry-scope gate forces an attributed active-cell failure and verifies
`failed_cells`, then forces unattributed and explicitly collective failures and
verifies `all_active`. A coupled sparse Compton run also exercises real
positivity rejections on nonzero MPI ranks and requires every attributed
rejection to remain `failed_cells` while the event reaches its requested final
time.
These are correctness gates; their short wall times are not performance data.
The normalization comparator allows `1e-6` relative discrepancy between global
and synchronized-individual reference scales, and the stricter sparse
FullReference-versus-AutoPartial check allows `1e-7` for both `max_Er` and
`max_rhoT`.

Run the three review gates with:

```bash
cd /home/elads/RICH-ablation-integration
gate_root="$(pwd)/regression_tests/results/lane_mg_timestep_gate_$(date -u +%Y%m%dT%H%M%SZ)"
sbatch --export=ALL,RICH_CASE_DIR="$(pwd)/regression_tests/cases/lane_radiation_shock_individual",RICH_MG_TIMESTEP_GATE_ROOT="${gate_root}" \
  regression_tests/cases/lane_radiation_shock_individual/submit_mg_timestep_normalization_gate.sbatch

ml openmpi/4.1.6/Intel/OneApi/2024.2.1
./build_rich.sh intelReleaseMPI/grey_timestep_review \
  --test_name=regression_tests/cases/mach2_diffusion
grey_root="$(pwd)/regression_tests/results/grey_timestep_gate_$(date -u +%Y%m%dT%H%M%SZ)"
sbatch --export=ALL,RICH_CASE_DIR="$(pwd)/regression_tests/cases/mach2_diffusion",RICH_GREY_TIMESTEP_GATE_ROOT="${grey_root}",RICH_GREY_TIMESTEP_GATE_BINARY="$(pwd)/build/intelReleaseMPI/grey_timestep_review/rich_intelReleaseMPI" \
  regression_tests/cases/mach2_diffusion/submit_grey_timestep_normalization_gate.sbatch

amr_root="$(pwd)/regression_tests/results/partial_amr_index_gate_$(date -u +%Y%m%dT%H%M%SZ)"
sbatch --export=ALL,RICH_CASE_DIR="$(pwd)/regression_tests/cases/lane_radiation_shock_individual",RICH_PARTIAL_AMR_GATE_ROOT="${amr_root}" \
  regression_tests/cases/lane_radiation_shock_individual/submit_partial_amr_index_gate.sbatch
```

The review run retained in
`regression_tests/results/review_gate_638e819b_20260811T191448Z` passed all
eight 128-rank MG checks with binary SHA-256
`638e819bec573892ad2b16b0f7dd7b7348979cde1d82935fe7092bb964a50e74`.
The companion grey and partial-AMR gates are retained in
`regression_tests/results/grey_timestep_gate_retry1_e136b35b_20260811T190500Z`
and
`regression_tests/results/partial_amr_index_gate_e00e4ec3_20260811T190300Z`.

## Partial-build benchmark

The following measurement is an illustrative development benchmark, not a
production performance guarantee. It used a GCC 12 `gnuDebug` build of
`mach2_diffusion`, 512 cells, one active cell, and one event on the development
host. Method order and machine load can affect absolute timings.

| Measurement | `AutoPartial` | `FullReference` |
|---|---:|---:|
| Active fraction | 1/512 = 0.195% | 1/512 = 0.195% |
| Mesh builds within the event | 3 + 4 + 3 + 3 ms | 1483 + 1757 ms |
| Reported hydro phase | 0.0171 s | 3.2677 s |
| Total event wall time | 0.022 s | 3.283 s |

For this event, partial construction reduced total event time by about 149x.
The repeated small partial builds are closure expansion passes. At large active
fractions, `AutoPartial` selects the full path at the configured threshold.

Reproduce the comparison after building `mach2_diffusion`:

```bash
export RICH_TEST_POINT_COUNT=512
export RICH_TEST_MAX_CYCLES=1
export RICH_TEST_SPARSE_INITIAL_BIN=1

RICH_INDIVIDUAL_MODE=partial \
  THUNDER_ARTIFACT_DIR="$(mktemp -d -p /tmp rich-partial.XXXXXX)" \
  ./build/gnuDebug/mach2_diffusion_serial/rich_gnuDebug

RICH_INDIVIDUAL_MODE=full-variable \
  THUNDER_ARTIFACT_DIR="$(mktemp -d -p /tmp rich-full.XXXXXX)" \
  ./build/gnuDebug/mach2_diffusion_serial/rich_gnuDebug
```

Use Release builds, repeated alternating method order, representative physics,
and production process counts before drawing production scaling conclusions.

## Coupled radiation-shock campaign

`regression_tests/cases/lane_radiation_shock_individual` is the coupled
performance and correctness case. It evolves a controlled `n=1.5` Lane-Emden
sphere with `M=2e33 g`, `R=7e10 cm`, `gamma=5/3`, and `mu=0.61`. A
material-temperature floor of `1e4 K` is imposed consistently on the
initial pressure and internal energy. A compact Planck-distributed radiation
pulse with peak radiation temperature `3e7 K` is deposited inside `0.20 R`.
The run includes:

- second-order moving-mesh hydro;
- distributed Barnes-Hut gravity with quadrupole moments and `theta=0.7`;
- 16-group free-free diffusion from 1 eV to 2 MeV;
- Compton redistribution and its transport scattering contribution exactly
  once, Doppler terms, flux limiting, hydro feedback, and protections;
- controlled adaptive refinement to `R/256`, hysteretic derefinement, a
  neighbor-volume limiter, and a 3,000,000-cell cap.

The primary interval is `0.20 t_dyn`. Three independent lanes use the same
immutable initial-state checksum: legacy global stepping, variable individual
steps with full meshes, and variable individual steps with automatic partial
meshes. Each executable measures its own evolution interval with an MPI
barrier, `MPI_Wtime`, and an `MPI_MAX` reduction. Setup, hydro, radiation, AMR,
and postprocessing times are separate.

For a fair time-to-solution comparison, the production default uses quantum
`2.8475356730643257e-13 s`, initial bin 30, and maximum bin 40. Thus the largest
individual interval is `0.31308985830411845 s`; the legacy global lane is
explicitly capped at that same interval. The manifest records the quantum,
both requested bins, and both maximum intervals. The comparator rejects a
campaign whose global and individual caps or initial intervals differ.

Build the 16-group Intel/OpenMPI binary once:

```bash
cd /home/elads/RICH-ablation-integration
bash regression_tests/cases/lane_radiation_shock_individual/build_production.sh
```

Submit the synchronized `dt/dt/2` calibration, the three production lanes,
and the dependent comparison job:

```bash
cd /home/elads/RICH-ablation-integration
bash regression_tests/cases/lane_radiation_shock_individual/submit_campaign.sh
```

After calibration passes, the three production jobs run concurrently subject
to the account CPU limit. Each uses 128 MPI ranks on 8 exclusive `bigrun`
nodes, 16 ranks per node, and the partition's 21-day limit. The comparison job
uses one `core` task for one hour and an
`afterany` dependency, so failed or cancelled lanes still produce a failure
verdict. Pass an explicit shared output directory as the first argument to
`submit_campaign.sh` when desired.

Production lanes use event-safe segmented execution. Every 6 hours, after a
completed event, the test writes one of two alternating distributed restart
snapshots and 128 per-rank benchmark-state sidecars, then atomically publishes
`restart_checkpoint_latest.txt`. At 20 days the same completed-event path is
used before the lane requeues its own Slurm job, one day before the hard limit.
This does not cap BiCGSTAB iterations and does not abandon or force-accept a
radiation candidate. On restart, the driver accepts only one of the two expected
snapshot names and requires the master snapshot plus every rank sidecar.

The sidecars preserve quantities that are deliberately outside the generic
RICH restart schema: original initial checksum and conservation baseline,
history samples, event/work counters, controlled-AMR next time and cumulative
changes, phase timings, and radiation retry statistics. `run.log` and
`progress.tsv` append across segments, and reported evolution time is the sum
of per-segment MPI-maximum evolution times with checkpoint I/O excluded.
`RICH_TEST_CHECKPOINT_WALL_SECONDS` and `RICH_TEST_SEGMENT_WALL_SECONDS` override
the 21,600-second and 1,728,000-second production defaults; zero disables the
corresponding behavior.

Each production segment also logs
`RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_MIN_BIN` and
`RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_LATCH`. The latter defaults to `0`; setting it
to `1` requires a minimum bin. This recorded policy is reused across pre-latch
checkpoint restarts, while checkpoint version 7 itself preserves a latch that
has already fired.

The reduced tier uses 8 ranks, 32,768 initial cells, 16 groups over the same
energy range, and `0.02 t_dyn`. It runs global, synchronized-full,
full-variable, and partial modes. Deterministic operations request 512
refinements at `0.005 t_dyn` and 256 derefinements at `0.01 t_dyn` so AMR is
tested even though the physical interval is short:

```bash
cd /home/elads/RICH-ablation-integration
bash regression_tests/cases/lane_radiation_shock_individual/build_reduced.sh
sbatch regression_tests/cases/lane_radiation_shock_individual/submit_reduced.sbatch
```

Rebuild with `build_production.sh` before submitting the 16-group campaign.

Every lane writes `timing.txt`, `counters.txt`, `metrics.txt`, `history.txt`,
`radial_profile.txt`, `spectrum.txt`, `manifest.txt`, `exit_code.txt`, initial
and final distributed HDF5 restart snapshots (`initial_state.h5` and
`final_state.h5`), and initial and final distributed VTK XML snapshots
(`initial_state.pvtu` and `final_state.pvtu`, with per-rank `.vtu` pieces).
The VTK output carries the hydrodynamic primitives, velocity, geometry, IDs,
total radiation energy, every configured group, tracers, and stickers.
Counters include active cells and faces,
gravity targets, MG rows, mesh-closure work, partial/full events, timestep
bins, AMR changes, and radiation retries.

The comparison checks immutable initial-state identity, final time,
conservation, positivity, Fleck factors, Compton scattering, AMR activity,
active-bin activity, profiles, shock properties, and significant group
spectra. Every pair must agree within 1% for normalized radial profiles, shock
location, and scalar shock properties. The full 16-group spectrum uses an
energy-weighted symmetric L1 error with a 0.1% limit, so numerically empty tail
groups cannot dominate the comparison. Runtime speedups are informational
only; thresholds are never loosened automatically.

The production comparator also scans the retained MG solver diagnostics. It
rejects any fixed 10,000-iteration safety-limit hit. For the full-variable and
partial lanes it requires the shared historical diagonal-scaled acceptance
policy and checks that the recomputed componentwise backward error is present
as a finite diagnostic. A rejected fractional candidate is allowed only as a
recorded transactional retry; it is not counted as a converged solve.

### Validation completed while adding the campaign

The 16-group case compiles with Intel OneAPI 2024.2.1 and OpenMPI 4.1.6. Small
two-rank, 512-cell MPI runs established the following implementation gates:

- legacy global and individual MG/Compton/Doppler events both completed from
  the same safe calibrated first interval;
- a sparse partial run completed three exact-tick events with three bins,
  timestep ratio 8, two real partial-mesh rank-events, reduced distributed
  gravity and MG work, no radiation retries, and nonnegative final groups;
- a controlled AMR run accepted 10 distributed refinements with mass drift
  `2.73e-16` and normalized momentum drift `8.13e-11`;
- full-variable and partial lanes both wrote the complete artifact set and a
  distributed HDF5 snapshot;
- a 512-cell MPI-build endpoint-I/O smoke wrote valid initial and final HDF5
  masters with external rank links and valid PVTU masters with referenced VTU
  pieces. Both VTK endpoints contained all 16 radiation groups and the expected
  hydrodynamic and geometry fields.

These are correctness smoke runs, not a production speedup claim. A subsequent
8-rank, 32,768-cell timestep-convergence study compared a capped global oracle
with maximum-bin 42, 43, and 44 individual runs. Bin 43 was the coarsest tested
choice that passed every agreement gate: global versus FullReference profile
L1 was `7.279e-3`, while FullReference versus AutoPartial profile, shock, and
spectrum errors were `8.953e-6`, `7.005e-9`, and `5.843e-7`. Evolution times
were 701.79 s (global), 1429.07 s (FullReference), and 1343.82 s
(AutoPartial), so partial construction was 1.063x faster than full construction
at this reduced size, while both individual lanes remained slower than global.

A controlled retry-policy A/B used 8 MPI ranks, 32,768 cells, 16 groups, no AMR,
and an identical 48-cycle endpoint. The old immediate-reprobe binary required
754 MG solves, 231 rejected probes, and 642.079 s of evolution. The
eight-acceptance cooldown required 470 solves, 71 rejected probes, and 464.771 s:
1.382x faster with 37.7% fewer solves and 69.3% fewer rejected probes. Both runs
ended at `t=28.378597646428485` with the same outer timestep, no BiCGSTAB
failure, and no fatal diagnostic. Using the production comparison formulas, the
maximum radial-profile L1 error was `2.442e-5`, the spectrum error was
`2.697e-6`, and the largest shock-property error was `4.558e-4`; all are well
inside their production limits. The retained result root is
`regression_tests/results/global_retry_cooldown_smoke_48_20260813T115000Z`.

The cooldown binary also completed one-cycle FullReference and AutoPartial MPI
smokes. Exact HDF5 comparison found no final-state difference other than
`WallclockTime` and the intentionally different `mesh_build_policy` metadata.

The current authoritative 2,000,000-cell campaign uses immutable binary SHA-256
`a1ed065d2dde080bf01a42450267dc934837dc9dc0a268b50bd857efe76e8e4f`:
calibration job `10120289`, global job `10120290`, FullReference job `10120291`,
AutoPartial job `10120292`, comparator job `10120293`, normalization/empty-rank/
restart gate `10120294`, and distributed-partial-AMR gate `10120295`.
Calibration passed with maximum profile L1 `2.490e-4` and significant-spectrum
error `1.428e-7`. The superseded campaign (`10120275`--`10120280`) and its
outputs were retained but its jobs were cancelled after the cooldown A/B. No
production speedup is claimed until the new dependent comparator accepts the
completed artifacts.

## Limitations

- Only Newtonian Cartesian 3D hydro is supported.
- `HydroStep::TIMEADVANCE_2` is required.
- Monte Carlo, IMC, and DDMC transport require global timesteps.
- Relativistic, 1D, and 2D solvers are not supported by this path.
- Radiation diffusion is backward Euler and first order in time; hydro and
  gravity remain second order.
- Physics operators share one cell bin. Independent hydro, gravity, and
  radiation clocks are not implemented.
- Global geometry outputs may deliberately request a full mesh.

## Implementation references

- Scheduler: `source/newtonian/three_dimensional/simulation/IndividualTimeStep.*`
- Mesh mappings: `source/newtonian/three_dimensional/simulation/ActiveMeshView.hpp`
- Event loop: `source/newtonian/three_dimensional/simulation/Simulation.cpp`
- Hydro and partial construction: `source/newtonian/three_dimensional/hdsim_3d.cpp`
- Grey diffusion: `source/Radiation/Diffusion.cpp`
- Multigroup diffusion: `source/Radiation/MultigroupDiffusion.cpp`
- Radiation candidate control: `source/newtonian/three_dimensional/simulation/steps/RadiationStep.cpp`
- MC prohibition: `source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp`
- Restart I/O: `source/3D/output/sim/read_simulation.cpp` and `write_simulation.cpp`
- Method background: [`arepo_paper.pdf`](../../arepo_paper.pdf) and
  [`arepo_sn.pdf`](../../arepo_sn.pdf)
