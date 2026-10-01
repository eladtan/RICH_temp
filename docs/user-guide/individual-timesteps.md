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
options.maximum_neighbor_bin_difference = 1;
options.full_source_sweep_interval_minimum_steps = 128;
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

## Recommended runtime settings

The settings below were validated on the tidal-disruption run
(`runs/BaseTDEComptonIndividual`, grey diffusion with Compton coupling, 256 MPI
ranks, 2026-10-01). From snapshot 70 (t = 41.86) to t = 50 the individual run
took 16531 s against 29173 s for the global-timestep control (1.77x). Its
run-minimum finest timestep was 0.97 of the global run's, and it had 69
radiation retries against 701. Every variable is agreed across MPI ranks at
first use; a mismatch throws.

| Variable | Default | Validated | Effect |
|---|---|---|---|
| `RICH_INDIVIDUAL_MAX_BIN_SPREAD` | 2 | 4 | No bin exceeds `initial_bin + K`, so no interval is longer than 2^K anchor intervals; -1 disables the cap. |
| `RICH_INDIVIDUAL_BIN_ANCHOR_MARGIN` | 0.8 | 0.95 | Anchors bin `initial_bin` at m times the uncapped CFL/source suggestion of the last global step instead of the (possibly ramped) step itself. |
| `RICH_ADAPTIVE_STAY_INDIVIDUAL` | off | 1 | Once the adaptive controller has adopted individual mode, it no longer spends wall time probing the global mode. |
| `RICH_INDIVIDUAL_RADIATION_ANCHOR_SUBCYCLE` | on | 1 | A grey relative-change limit in [band, 1) x the anchor interval subcycles radiation inside the hydro interval instead of moving the cell to a finer bin. |
| `RICH_INDIVIDUAL_RADIATION_ANCHOR_BAND` | 0.5 | 0.125 | Lower edge of that band; 0.125 allows up to eight radiation pieces per anchor interval. |
| `RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS` | on | 0 | Off: a rejected radiation candidate is absorbed by subcycling and does not lower the next hydro bin. |
| `RICH_INDIVIDUAL_RADIATION_ENTRY_PROBE` | off | 1 | Lets an earned recovery probe be an event's first radiation candidate, so a persisted retry ceiling of 1/2 can recover. |
| `RICH_INDIVIDUAL_CLOSURE_REEXPAND` | off | 1 | Repeats the partial-mesh closure expansion until no rank adds cells (hydro results unchanged to round-off). |
| `RICH_RADIATION_MOMENTUM_POSITIVITY` | off | 1 | Positivity-preserving, energy-conserving treatment of the grey velocity term (below). |
| `RICH_RADIATION_MOMENTUM_KINETIC_LOSS_FRACTION` | 0.5 | 0.5 | Fraction L used by the two caps of the momentum-positivity treatment. |

A complete launch, run from the problem directory, with the restart and final
time chosen by the driver's own variables:

```bash
export RICH_INDIVIDUAL_MAX_BIN_SPREAD=4 RICH_INDIVIDUAL_BIN_ANCHOR_MARGIN=0.95 \
       RICH_ADAPTIVE_STAY_INDIVIDUAL=1 RICH_INDIVIDUAL_RADIATION_ANCHOR_SUBCYCLE=1 \
       RICH_INDIVIDUAL_RADIATION_ANCHOR_BAND=0.125 RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS=0 \
       RICH_INDIVIDUAL_RADIATION_ENTRY_PROBE=1 RICH_INDIVIDUAL_CLOSURE_REEXPAND=1 \
       RICH_RADIATION_MOMENTUM_POSITIVITY=1
mpirun -np 256 ../../build/intelReleaseMPI/rich
```

### Momentum positivity (grey diffusion)

The grey diffusion matrix contains the radiation pressure-work and O(v/c)
relativistic exchange term `T_ij = ½(1 − α_i) k_ij`, central-differenced on each
face. Where it is positive and larger than the diffusion coupling, the matrix
loses its M-matrix property. Dim cells next to bright ones can then solve to
negative radiation energy at any timestep, which caused deep retry cascades in
late TDE phases.

With `RICH_RADIATION_MOMENTUM_POSITIVITY=1`:

- **Lumping.** Per interior column, only the positive excess of the assembled
  coupling is moved to the diagonal: a minimal upwind shift of that face's
  velocity term. The row action on a uniform field is unchanged.
- **Rejection.** A row is rejected collectively before preconditioning if:
  - it has a positive coupling the velocity term does not explain;
  - it has a positive diffusion coupling;
  - its diagonal or row sum is non-positive (interior rows only);
  - a physical right-hand side is negative.
- **Energy exchange.** PostCG uses the same lumped face values, face timesteps
  and α as the matrix. The radiation-gas exchange is therefore identical to what
  the matrix removed.
- **Kinetic cap.** If the radiation-force impulse would make the gas gain more
  than L of the cell's available radiation energy, the impulse is scaled down to
  exactly that amount.
- **Thermal cap.** If the relativistic exchange would take more than L of the
  gas internal energy after absorption and emission, radiation pays the excess.
- **Conservation.** Both caps conserve total energy exactly; the kinetic cap
  only withholds momentum, which flux-limited diffusion does not track.

Rank 0 prints one `RICH_RADIATION_MOMENTUM_POSITIVITY stage=matrix` line per
matrix build and one `stage=exchange` line per candidate:

- the `matrix` line counts changed rows, lumped faces and certificate
  violations;
- the `exchange` line reports the lumping energy, pressure work, cap counts and
  energies;
- a coefficient closure and a reservoir closure check that the exchange matches
  the assembled matrix. Both stay at round-off.

With `RICH_INDIVIDUAL_D5_TRACE=1`, a failing cell also prints a rank-local
`RICH_RADIATION_MOMENTUM_FAILURE` line with the terms of its final energies.

## Options and timeline

`IndividualTimeStepOptions` has these defaults:

| Option | Default | Meaning |
|---|---:|---|
| `time_quantum` | `0` | Exact timeline quantum. A non-positive value derives `initial_dt / 2^initial_bin`. |
| `initial_bin` | `30` | Initial bin assigned to every cell. |
| `maximum_bin` | `40` | Largest permitted bin; it must not exceed 62. |
| `maximum_neighbor_bin_difference` | `1` | Largest bin difference allowed across a face. |
| `full_source_sweep_interval_minimum_steps` | `128` | Run a true all-source hydro wake sweep after this many widths of the globally smallest occupied bin have elapsed. |
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
half-kick state. It also stores the finest pending neighbor-induced bin
reduction that must propagate after the cell completes that physical bin.
Floating-point time is reconstructed only at the physics interface.

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
5. Ask every physics step for per-active-cell timestep limits and passive
   wake deadlines, then commit the event to the scheduler.
6. Apply wake deadlines, then propagate physical bin reductions across one
   Voronoi-face layer, including MPI neighbors.
7. Advance the simulation clock to the event time and report the next event
   interval.

A bin decrease takes effect immediately. A bin can increase by at most one
level per activation, and only when the event tick is aligned with the larger
power-of-two interval.

The default face constraint is one bin. If a source at bin `b` reaches a
neighbor above `b + 1`, that neighbor changes directly to `b + 1` and wakes
at its earliest aligned tick. Sources are snapshotted, so a reduction cannot
cascade through multiple face layers in the same event. A neighbor-induced
reduction is retained as pending state and becomes a source only after that cell
completes its full nominal physical-bin interval. Initialization, restart, and
AMR remapping instead perform a one-time full closure without creating pending
propagation.

An interrupted cell keeps its physical bin and schedules its next activation
at the next tick aligned with that bin. This also applies to terminal clamps
and forced all-active overlays. The first catch-up interval can be shorter than
the bin interval; subsequent activations return to the shared power-of-two
phase instead of creating a permanent off-grid event sequence.
Physics integrates that first event over the actual catch-up interval, while
radiation timestep growth caps use the retained bin's nominal interval. A
short wake therefore cannot become a real small physical bin merely because
the limiter allows at most a factor-of-two increase.

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
2. Seeds the target with active cells, the previous reconstruction halo, and
   the two-cell reconstruction shell around the active cells taken from a
   per-cell adjacency cache: the face neighbours recorded the last time each
   cell was in an event mesh, by stable ID with the owning rank. Remote
   neighbours are requested from their owner in one sparse exchange per
   shell. The closure check in step 5 then rarely has to add cells and
   rebuild; without the seed a mesh typically took two or three builds.
   With the seed on, the previous accepted target is no longer added as a
   warm start: that set contains its own warm start and could only grow
   between full builds, so after a large active set events with a few active
   cells rebuilt nearly the whole mesh as a "partial" target.
   `RICH_INDIVIDUAL_ADJACENCY_SEED=0` disables the cache and the seed and
   restores the warm start.
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
  conserved updates have been assembled. The fluxes of an interval use the
  primitives stamped at the interval start and the generator velocities that
  moved the mesh during the interval; the velocity for the coming interval is
  chosen after the update, from the recovered primitives on the event mesh.
- Two conserved-state guards complement the wave-speed criteria. The mass-loss
  limit bounds the next step so that no interval removes more than a fraction
  of a cell's mass at the loss rate just measured. The conserved-change wake
  ends a passive cell's interval when its accumulated mass or energy change
  exceeds a fraction of its activation value, which signal-speed wakes cannot
  detect. Both fractions default to 0.25 (see the runtime controls).
- A mesh-deformation limit bounds how far a generator may travel relative to a
  neighbouring generator within one interval. The CFL is evaluated on the faces
  that exist when an interval opens; it bounds the gas crossing those faces and
  says nothing about the mesh redrawing itself under its own point velocities.
  Two neighbours whose generator velocities differ by `dv` replace the face
  between them on the timescale `d/dv`, and a face that grows late in a long
  interval is still charged the whole of it. The limit is floored at one
  sixteenth of the cell's own CFL limit, so a degenerate generator pair cannot
  drive it to zero; a mesh that needs more than that is the regulariser's
  problem, not the scheduler's.
- The step audit records the limit in force when each interval opens and
  reports, per event, any interval that outlasted it (`step_overruns`). Limits
  applied after the hydro step can only shorten an interval further, so a
  reported overrun is a real one, and it means a computed limit was not
  honoured rather than that a limit was missing.
- An interval may never outlast the allowance of the bin it carries. Lowering a
  cell's bin part way through an interval - by neighbour closure, by its own
  limit while passive, or by an AMR merge - used to clip the end only to the
  next tick aligned to the new bin. When that aligned tick is the end the cell
  already had, the bin was relabelled and the interval kept its old length, so
  a cell could carry bin 36 and run a bin-37 interval. The end is now bounded
  by the bin's allowance as well, and `INDIVIDUAL_BIN_OVERRUN` reports any cell
  that still completes more than its allowance, which can only happen when the
  request arrives after the allowance has already elapsed.

In MPI, an inter-rank face has one deterministic owner. Conserved deltas are
sent by stable cell ID, which prevents duplicate face work and avoids relying
on transient local indices.

Supported conservative gravity sources use active-target acceleration
evaluation and kick-drift-kick integration. The acceleration and half-kick
phase are cached per cell. Gravity limits participate in bin selection and can
wake a passive cell.

The first-half source phase runs for the cells that close their interval at
the event. For a conservative force whose active cells all carry a pending
half kick it needs no geometry: it is a momentum kick from the cached
acceleration. The mesh at the interval-start generator positions existed only
to serve that phase (the fluxes use the event mesh), so when every rank's
source reports that its first half can run from cache, that build is skipped
and the phase is applied on the canonical owned-cell arrays. A cell without a
pending kick (the first event, a cleared cache, a centre-sink reset) makes
every rank build the interval-start mesh as before.
`RICH_INDIVIDUAL_FIRST_HALF_FROM_CACHE=0` restores the unconditional build;
any event-mesh reuse experiment (`RICH_INDIVIDUAL_REUSE_EVENT_MESH*`) also
keeps it.

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

When the conservative `legacy` active--passive diffusion transfer changes an
inactive recipient, radiation supplies a wake deadline rather than changing
that cell's physical bin. The wake rate uses the shortest active--passive face
timestep that contributed nonzero accepted energy, never the global scheduler
event spacing. The face timestep is carried with cross-rank transfer metadata,
so serial, MPI, grey, and multigroup paths use the same definition.
If a changed passive recipient unexpectedly lacks that metadata, it wakes after
one scheduler quantum instead of estimating a rate from unrelated event timing.

Individual grey and multigroup diffusion select one MPI-consistent
passive-boundary policy with `RICH_MG_INDIVIDUAL_PASSIVE_POLICY`:

- `dirichlet` is the library default. It freezes passive radiation energy in
  the active-row right-hand side and commits only active state. Passive
  primitive and conserved radiation state remain unchanged during that
  candidate.
- `legacy` explicitly restores the immediate equal-and-opposite passive
  conserved commit.
- `shadow` enables the owner-held conservative shadow-reservoir experiment.
  It is not the production AutoPartial policy.

The deprecated `RICH_MG_INDIVIDUAL_SHADOW_RESERVOIRS` flag remains a
compatibility selector: `1` selects `shadow` and `0` selects `legacy`.
Conflicting selectors are rejected collectively. None of these settings
changes the synchronized global radiation path. Flux limiters, boundary
conditions, cooling limits, hydro feedback, all energy groups, Compton
safeguards, and Doppler terms remain active.

Dirichlet mode deliberately omits the equal-and-opposite passive interface
transfer. Before commit, it measures that omitted transfer from the same face
coefficients and final active unknowns used by the candidate. Accepted events
append signed, absolute, normalized, and worst-local contributions to a
rollback-safe defect ledger. Finite conservation thresholds are synchronization
targets, not candidate-acceptance limits. Crossing the local or event target
keeps the candidate, leaves passive radiation unchanged, and requests every
passive endpoint of that accepted radiation event to wake after the shortest
contributing face interval. Solver, mapping, nonfinite, and positivity failures
remain hard rejections. Dirichlet mode therefore has measured, but not exact,
conservation across active--passive interfaces.

The version-3 local synchronization target uses the mixed extensive tolerance

`withdrawal <= 1e-2 * (passive_extent + roundoff_floor) + 1e-9 * global_scale`.

The relative term remains strict for resolved passive radiation reservoirs,
while the absolute term prevents a floor-dominated cell from requesting
synchronization for globally negligible roundoff. An event absolute defect
above `1e-6` of `global_scale` also requests passive synchronization.
The cumulative signed and absolute reference levels remain `1e-4` and
`1e-3`, but are diagnostic because the scalar ledger cannot repay a
cell/group transfer. They never reduce every active cell's future timestep.

Normal solver convergence and accepted defect decisions are silent. Detailed
distributed-solver reports remain available only when the existing profiling
selector is enabled. Invalid defect bookkeeping uses the ordinary event-level
failure summary; finite target crossings do not enter the retry loop.
The scheduler-owned ledger, its versioned limits, retry/cooldown state, and
`history_complete` flag are checkpointed. Benchmark restart fingerprints and
`counters.txt` print those fields so a restarted energy audit can distinguish
a complete accumulated history from a reconstructed default. Loading a
version-1 or version-2 defect ledger preserves its cumulative extents, resets
its obsolete retry ceiling, adopts the version-3 semantics, and marks
`history_complete=false`.

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
tick, bin, and pending neighbor source. Derefinement may deposit conserved
extents into passive neighbors. Every actual overlap recipient retains the
minimum of its own and the removed source's physical and pending bins; MPI
recipients obtain that state from the source owner by stable ID. The scheduler,
predictors, acceleration cache, radiation metadata, and active-mesh maps are
remapped after the topology change, followed by full neighbor closure.

Distributed second-order AMR neighbor requests use three distinct index
spaces. MadVoro duplicate-point entries are compact all-point indices; they are
converted first to the original build-input index and then through the inverse
owned mapping to a local mesh index. Ghost ownership comes from the paired
ghost/duplicated-rank metadata, not a spatial owner query. Mapping failures are
validated collectively before recursive neighbor traversal.

## Box growth

`UpdateBox` (`source/3D/GeometryCommon/UpdateBox.hpp`) grows the box when cells
faster than `min_velocity` come within five times the largest such width of a
wall: the wall moves out by that distance, random cells with a reference state
fill the new region, the mesh is rebuilt with rebalancing, and every extensive
is recomputed from its primitive (the legacy semantics, kept in both modes by
decision of 2026-09-24; the reset can change total energy by the dual-energy
difference). It needs a synchronized state, so in individual mode:

- `BoxGrowthDue(sim, min_velocity)` evaluates the same criterion after every
  event on the committed state (generator positions from
  `Simulation::CommittedGeneratorPoints`, widths from mass / density). It is a
  trigger: when it holds, call `RequestSynchronizedIndividualEvent()`.
- `UpdateBoxSynchronized(...)` decides exactly on a synchronized state (call it
  at every synchronized individual event and on global boundaries) and grows
  through `Simulation::GrowDomainAtSynchronizedIndividualState`: primitives,
  extensives and scheduler states migrate together by stable ID; every cell's
  interval is then capped by the individual event's rule evaluated on the
  rebuilt mesh with the point velocities committed for its next interval
  (`PhysicsStep::synchronizedCellTimeStepLimits`: wave-speed CFL, source
  per-cell limits through `SourceTerm3D::SynchronizedIndividualLimits`, and
  the mesh-drift guard), new and volume-changed cells also by the finest bin
  in use (with `force_synchronized`, one shared bin). A conservative force
  with target evaluation refreshes the acceleration cache on the rebuilt
  state (one full solve), so the next first half kicks run from it;
  `DiffusionForce` cannot re-evaluate its limits and refuses the growth. The
  neighbour closure runs, topology caches are released, and the adaptive
  controller is told (`NotifyDomainChanged`).
- A request the controller answers by switching to global is served on that
  global boundary. The returned `DomainGrowthReport` carries the counts, the
  timing and the conserved totals before and after, with the inserted amounts.

The adaptive controller tags every measured throughput with a domain epoch.
`NotifyDomainChanged` (also called by drivers after a global growth) restarts
the current window, and a probe that ends against a baseline from an older
epoch returns to the baseline mode to re-measure it: box growth can change the
gravity cost ten-fold, so the old comparison would be against another workload.
`runs/BaseTDEComptonIndividual` shows the driver loop
(`RICH_TDE_UPDATE_BOX=1`; records `RICH_UPDATE_BOX_REQUEST`,
`RICH_UPDATE_BOX ... mode=individual`, `RICH_MODE_DOMAIN_CHANGE`). Growth
times differ between modes (global checks every 7 steps, individual after
every event), so mode A/B comparisons see different domain histories.

## Adaptive integration mode

A run may let the simulation decide whether individual timesteps pay at all.
`Simulation::SetAdaptiveIntegrationMode(true, options)` (or
`RICH_INDIVIDUAL_ADAPTIVE_MODE=1`, which overrides the call) installs a
controller that switches between the real global path (`timeAdvance2`, point
exchange and load balancing included) and individual events, in both
directions, from two measurements:

- **Potential gain.** While stepping globally, every `gate_interval` steps the
  per-cell limits of the full mesh are quantized to power-of-two bins above
  the smallest of them. Per-cell limits come from every physics step that
  has them (`PhysicsStep::collectCellTimeStepLimits`): hydro wave speeds and
  the source cap from `CourantFriedrichsLewy::CellTimeSteps`, and grey
  `Diffusion`'s radiation limit `dt * 0.15 / diff_i` from the last global
  step, the rule individual mode applies cell by cell (without the growth
  caps). Other steps act as a uniform cap through `suggestTimeStep()`. The
  ratio of global cell-updates to the ideal individual cell-updates,
  `N / sum_i 2^-bin_i`, bounds the speedup any individual scheme could reach
  from that distribution. Below `gain_min` (default 1.5) there is little to
  gain by construction and the controller stays global without spending a
  probe. The bound ignores neighbour-bin closure and per-event fixed cost, so
  it is optimistic; it only ever prevents probes. The radiation limits are
  matched to the current cells by ID; a cell without one (migrated from
  another rank or new from AMR) sets none, so missing coverage can only
  overstate the gain. Until 2026-09-24 radiation entered only as its grid-wide
  cap (the largest relative radiation-energy change anywhere): while that was
  below twice the smallest hydro limit every cell landed in bin 0, the bound
  was exactly 1 and the controller never probed (the TDE: gain_bound=1 at
  every decision). Multigroup diffusion still enters as its cap.
  Rank 0 prints `RICH_MODE_GAIN` whenever the bound is evaluated:
  `gain_bound`, `gain_bound_uniform_caps` (the old form, radiation as its
  cap), `dt_cell_min`, per step its smallest limit (`limits=name:cell|cap:dt`)
  and the cells whose limit it sets (`bound_cells`), the cells per bin
  (`bins`) and `fallback_cells` (cells without a radiation limit).
  The bound is evaluated after the global post-step callback, on the state the
  next step starts from. Hydro limits need the face velocities of the mesh
  they are evaluated on, known only for the mesh the hydro step leaves
  (`HDSim3D` records its build generation). If that mesh was rebuilt since on
  any rank (the callback's AMR pass or box growth, a rebalance after the hydro
  step), the evaluation is skipped rather than formed with hydro as a cap:
  rank 0 prints `RICH_MODE_GAIN_SKIPPED reason=stale_cell_limits` (the first
  skip of a streak, then at `consecutive` = powers of two), the bound is
  unknown (no veto) and the next global step retries it. In the TDE (AMR every
  ten cycles) the retry lands on the following step; a driver that rebuilds
  the mesh after every hydro step never gets a bound, and the controller then
  decides by probes alone.
- **Realised throughput.** Simulated time per wall second is accumulated in
  the current mode over a dwell window (`dwell_min_steps`, default 64, times a
  multiplier), excluding a ramp after each switch (12 individual events or 2
  global steps by default; bins grow one level per activation after a switch
  into individual mode, and the first global step rebuilds the mesh with
  exchange). At the end of a dwell the other mode is probed with a wall-time
  budget of `probe_fraction` (default 0.1) of the dwell's wall time and at
  least `minimum_samples` measured steps; the probed mode is adopted when its
  throughput exceeds the current one by `margin` (default 1.15), otherwise
  the run reverts and doubles the next dwell (up to `dwell_backoff_cap`).

Switches happen only at synchronized states. Leaving individual mode requests
one all-active event and switches after it; entering it creates a fresh
scheduler from the stored options with its time quantum derived from the
global step of that moment, so cells start at `initial_bin` with the global
step and grow from there. Physics steps see `beforeIndividualRebalance()` on
every switch and `afterIndividualAMR()` when entering individual mode, which
drops the topology-sized individual caches.

Driver hooks for runs under the controller:
`Simulation::AdaptiveIntegrationModeWillEnable(requested)` says, before
anything is enabled, whether `SetAdaptiveIntegrationMode(requested, ...)` will
turn the controller on (the environment wins when set), e.g. to start a fresh
run on the global path. `SetGlobalPostStep(callback)` runs a collective update
after every global step, once the cycle counter has advanced and before the
controller acts, so a switch at the end of the step neither skips nor
follows it: the global-path counterpart of `SetIndividualAMR` and
`SetIndividualPostPhysics`. `SetAdaptiveDecisionsDeferred(true)` postpones
every decision the controller would take on a global step, for as long as the
driver has a target the global step sequence must land on (a snapshot time a
fresh scheduler would not honour).

Rank 0 reports `RICH_MODE_CONTROLLER` once, `RICH_MODE_DECISION` at every
decision (`phase=dwell|probe`, `tau` in simulated time per wall second for
both modes, `gain_bound`, `gain_bound_uniform_caps`, `action=probe_*|adopt_*|revert_to_*|
stay_global_little_to_gain`) and `RICH_MODE_SWITCH` at every switch. All
inputs are collectively reduced values, so every rank takes the same
decision. A checkpoint written while stepping globally carries no scheduler
group; a restart resumes globally under the controller when the driver calls
`SetAdaptiveIntegrationMode` again. `Simulation::StateSynchronized()` is true
at every global step boundary and at synchronized individual events, and
`RequestSynchronizedIndividualEvent()` is a no-op in global mode, so output
logic written for individual mode keeps working.

Environment controls: `RICH_INDIVIDUAL_ADAPTIVE_MODE`,
`RICH_ADAPTIVE_DWELL_MIN_STEPS`, `RICH_ADAPTIVE_MIN_SAMPLES`,
`RICH_ADAPTIVE_RAMP_EVENTS`, `RICH_ADAPTIVE_RAMP_STEPS`,
`RICH_ADAPTIVE_GATE_INTERVAL`, `RICH_ADAPTIVE_DWELL_BACKOFF_CAP`,
`RICH_ADAPTIVE_PROBE_FRACTION`, `RICH_ADAPTIVE_MARGIN`,
`RICH_ADAPTIVE_GAIN_MIN`. Values must agree on every MPI rank.

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

## Runtime stdout

Every accepted global step and individual event writes one rank-0, flushed
six-line block followed by one blank line:

```text
RICH_STEP mode=individual cycle=42
  time   | t_start=1 | t_end=1.1 | event_dt=0.1 | applied_dt_min=0.1 | applied_dt_max=0.1 | next_event_dt=0.2
  work   | active_cells=160 | total_cells=1000 | active_bins=[bin=0,count=128,dt=0.1; bin=1,count=32,dt=0.2]
  phases | step_s=2.100000 | hydro_s=1.000000 | gravity_s=0.000000 | radiation_s=0.900000 | amr_s=0.200000
  mesh   | mesh_s=0.300000 | mesh_builds=2
  source | source_s=0.400000 | source_pct=19.047619 | source_calls=2
```

The header identifies the mode and cycle. The subject lines have stable field
order and do not repeat that identity. Simulation times and timesteps are in
code units. Fields ending in `_s` are wall seconds with microsecond print
precision; simulation times and timesteps use 12 significant digits to avoid
printing binary roundoff artifacts in human-facing logs.

For individual events, `active_cells` is the sum across all MPI ranks and
`total_cells` is the MPI-summed owned-cell population at the start of the
event, including both active and inactive cells. Thus `active_cells` is always
less than or equal to `total_cells`. For global stepping the two counts are
equal. `active_bins` lists every active scheduler bin as named fields inside
brackets.
Each count is summed across all MPI ranks. Individual `event_dt` is computed
from the scheduler tick difference and time quantum, not by subtracting two
floating-point event times. It is the scheduler event gap;
`applied_dt_min` and `applied_dt_max` are the actual active-cell intervals, and
`next_event_dt` is the next scheduler gap. Global stepping uses
`active_bins=global`.

`hydro_s`, `gravity_s`, `radiation_s`, `amr_s`, `mesh_s`, and `step_s` are MPI
maxima.
The named phase fields time distinct top-level `PhysicsStep` objects. When a
gravity calculation is installed as a hydrodynamic source term, its time is
part of `hydro_s` and the exact callback contribution is part of `source_s`;
`gravity_s` remains zero because there is no separate gravity step.
`source_s` is the MPI maximum of each rank's exact hydrodynamic source-callback
wall time across both half updates; it excludes scatter, cache maintenance,
logging, and other phase work. `source_calls` is the maximum callback count and
`source_pct=100*source_s/step_s`. `mesh_s` measures wall time inside attempted
tessellation-build operations during the step, including hydro, standalone
remeshing, individual AMR, load balancing, failed attempts, and builds used for
fallback, restart restoration, or partial-mesh verification. `mesh_builds` is
the maximum number of those attempts on any MPI rank, rather than a sum across
ranks. The mesh time is already contained in its owning phase and `step_s`; it
is not an additional phase to add to the step total. A rejected candidate
immediately writes a
rank-0, flushed three-line block followed by one blank line:

```text
RICH_RETRY mode=individual cycle=42 physics=radiation attempt=1
  attempt | active_cells=160 | active_bins=[bin=0,count=160,dt=0.1] | attempted_dt_min=0.1 | attempted_dt_max=0.1
  failure | retry_s=0.300000 | reason=solver_rejected | cell=17
```

Text values are normalized to underscore-separated tokens. A missing
representative cell prints as `cell=none`.

An accepted individual AMR callback that actually adds or removes cells writes
one separate rank-0 line after the accepted step block:

```text
RICH_AMR mode=individual cycle=42 time=1.1 cells_before=1000 added_cells=12 removed_cells=3 cells_after=1009
```

All four cell counts are totals across MPI ranks. The line is omitted when the
AMR callback only checks its criteria and returns an empty change set. Therefore
`amr_s` can be nonzero without a `RICH_AMR` line: `amr_s` includes time spent
checking AMR eligibility, while `RICH_AMR` denotes an accepted topology change.
The global `AMR3D::operator()(Simulation&)` path writes the same record with
`mode=global` at the point where the operator is called. Its `cycle` and `time`
are the simulation tracker values at that call site. A global AMR record is a
standalone event and is not paired with the preceding `RICH_STEP` block; other
application output may appear before it. Every `RICH_AMR` record is followed by
one blank line.

`RICH_RUNTIME_LOG` accepts `summary` (the default) or `detailed`. Detailed mode
inserts `source_first_s` and `source_second_s` after `source_s`, writes
rank-distribution diagnostics under the `RICH_STEP_DETAIL` prefix, and restores
existing routine diagnostics with their established prefixes, including
one representative failing rank's labeled `RICH_RADIATION_DETAIL` block. These
selector-controlled records remain stdout-only and rank-0-only. Summary mode
hides routine diagnostics and keeps stdout limited to the structured step and
retry blocks plus topology-changing `RICH_AMR` lines. New runs and campaign
parsers use only those formats, not the historical `Individual cycle`,
`INDIVIDUAL_PERF`, or single-line `RICH_STEP` formats.

`RICH_RUNTIME_COLOR` accepts `auto` (the default), `always`, or `never`.
`auto` colors labels only when stdout is a terminal and `NO_COLOR` is unset.
Use `never` for saved logs; use `always` only for an ANSI-aware consumer.
Parsers strip ANSI escapes. Accepted labels use cyan, blue, magenta, and green
by subject; retry and AMR labels use yellow, with retry failures in red. Invalid
or MPI-inconsistent runtime log or color settings fail collectively.

## MPI behavior

MPI individual mode adds these collective rules:

- the next event tick is the global minimum;
- ranks without active cells still enter mesh, physics, retry, and acceptance
  collectives;
- ranks with zero owned cells also enter grey radiation-force exchanges and
  reductions with neutral local maxima; no empty vector is dereferenced;
- partial parallel construction suppresses routine load balancing and the
  point exchange but retains required ghost exchange. Because owned points are
  never migrated to their nominal Hilbert owner on this path, the mesh library
  switches its ghost range-query routing to a distributed oct tree built from
  the ranks' actual point positions at the first suppressed-exchange build
  and refreshes it on every build. Routing by nominal Hilbert ranges is only
  valid immediately after a real exchange; with a drifting mesh it sends
  queries to the wrong ranks, leaves boundary cells with an incomplete
  neighbour set, and produces phantom cross-rank faces (the
  `INDIVIDUAL_HYDRO_INVALID_MASS` failures of September 2026). The switch is
  reported once on rank 0 as
  `MeshDecomposer: routing sphere-rank queries by actual point positions`;
- distributed partial closure exchanges owner-canonical target requests to a
  fixed point; ranks with no local additions still enter every rebuild and
  closure reduction;
- inter-rank hydro and radiation face deltas use deterministic face ownership
  and stable destination IDs;
- grey and multigroup solvers build distributed active-only row maps and
  exchange only required remote active values;
- wake propagation leaves active physical timestep suggestions unchanged and
  writes signal arrivals to a separate passive wake-deadline vector.
  Shortening a passive deadline does not change its stored timestep bin. After
  the interrupted update, its next bin is chosen from the pre-wake bin and the
  newly computed physical limit, preventing a one-tick wake from becoming a
  persistent one-tick timestep. Only active cells that completed their retained
  physical bin seed a new wake solve. A cell active because of an earlier
  signal is a consumer of that already scheduled propagation, not a new source;
  this prevents adjacent interrupted cells from waking each other every base
  tick. Secondary propagation from that cell begins at its next physical-bin
  completion. Hydrodynamic signals use the sparse distributed wake tree. Their
  arrival path is the full separation between the source and target cell
  centroids; no effective cell radii are subtracted. Active centroids are
  refreshed from the current event mesh, while inactive centroids use the
  canonical drift prediction already maintained by individual hydro and
  gravity, so the wake solve does not force a full mesh build. Tree pruning uses
  the minimum point-to-node distance, and the MPI rank-query radius is only the
  distance the maximum signal speed can cover before the latest passive
  deadline. The pair signal speed is the AREPO hydrodynamic value: both sound
  speeds plus the positive projected closing velocity. It has no speed-of-light
  cap and uses no opacity or radiation transport state. Each rank updates only
  its owned passive targets. Grey and multigroup radiation
  may still supply their own passive wake deadlines through the physics-step
  interface, but do not enter the tree signal speed;
- as a periodic safety backstop, elapsed integer ticks are compared with the
  width of the globally smallest occupied bin in the committed scheduler state
  entering the event. At the first such event where the elapsed interval reaches
  `full_source_sweep_interval_minimum_steps` bin widths, every owned cell,
  active or passive, is queried as a hydrodynamic signal source against all
  passive targets. Stable cell IDs exclude self-signals. The target tree is
  built once, and MPI source replication is split into synchronized rounds
  whose worst-case payload is bounded per rank. Rank zero prints one
  `RICH_FULL_SOURCE_SWEEP` record after the sweep and scheduler commit. The
  record includes the tick, physical time, minimum bin and timestep, interval,
  source and passive-target counts, MPI rounds and source records, peak MPI
  payload bytes, and maximum rank time;
- normal neighbor-bin propagation exchanges one packed snapshot containing
  stable ID, current bin, and source bin to MPI ghosts, then performs one
  stable-ID owner-request exchange. Requests are applied only after the face
  scan, so traversal order cannot create a same-event cascade. A reduced target
  records its finest pending source bin. Full fixed-point closure is reserved
  for initialization, restart, and AMR remapping;
- convergence, positivity, retry, and event acceptance are collective;
- the forced-active threshold and persistent-latch selector are validated
  collectively, and a restored latch must agree on every rank before physics;
- explicit load-balancing events migrate scheduler, predictor, acceleration,
  conserved, and radiation state.

## Runtime and regression controls

Production setups may set the full-source sweep interval through the C++ API or
its runtime environment override. Other environment controls include production
logging and focused-run or regression selectors:

| Variable | Values or effect |
|---|---|
| `RICH_INDIVIDUAL_MODE` | `full` for one adaptive synchronized bin, `full-variable` for variable bins with full meshes, or `partial` for variable bins with `AutoPartial`. |
| `RICH_INDIVIDUAL_FULL_SOURCE_SWEEP_INTERVAL` | Positive integer overriding `full_source_sweep_interval_minimum_steps`; default `128`. Values must agree on every MPI rank. |
| `RICH_RUNTIME_LOG` | `summary` (default) for core blocks only, or `detailed` for source halves and routine diagnostic records. |
| `RICH_RUNTIME_COLOR` | `auto` (default), `always`, or `never`; invalid or MPI-inconsistent values fail collectively. |
| `NO_COLOR` | Any value disables label colors when `RICH_RUNTIME_COLOR=auto`. |
| `RICH_TEST_SPARSE_INITIAL_BIN` | Makes one initial cell faster, producing active-passive faces immediately. |
| `RICH_TEST_SPARSE_MAX_ER_CELL` | Test-only: makes the cell owning the global radiation reference maximum the sparse active cell. |
| `RICH_TEST_INITIAL_BIN`, `RICH_TEST_MAXIMUM_BIN` | Override the power-of-two scheduler bounds in focused and calibration runs. |
| `RICH_TEST_TIME_QUANTUM` | Overrides the exact integer-timeline quantum. |
| `RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_MIN_BIN` | Default off. Before physics, compare this threshold with the largest actual active-cell interval, measured in scheduler ticks. A qualifying partial event promotes every owned cell without changing `maximum_bin`. |
| `RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_LATCH` | `0` by default. With value `1` and a configured minimum bin, reaching the threshold sets a monotonic collective latch. The threshold event and every later partial event use the all-active overlay; naturally all-active events need no promotion. |
| `RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE` | Unset selects active Hilbert balancing automatically for MPI individual-timestep runs with a Hilbert load balancer and a rebalance-capable physics step. `0` disables it. `1` requires it and rejects an incompatible setup. |
| `RICH_INDIVIDUAL_ACTIVE_HILBERT_THRESHOLD` | Maximum active-cell max/mean ratio accepted without rebuilding; default `1.25`. The mean is floored at one active cell per rank so sparse events remain attainable. |
| `RICH_INDIVIDUAL_ACTIVE_HILBERT_MAX_OWNED_SKEW` | Reject an active-only partition when its predicted total owned-cell max/mean exceeds this memory-safety cap; default `2.0`. Candidate cuts that leave any MPI rank empty are also rejected. |
| `RICH_INDIVIDUAL_MASS_LOSS_FRACTION` | Default `0.25`. Mass-loss timestep limit: at activation a cell's next step is bounded so that, at the mass-loss rate measured over the interval that just closed, no more than this fraction of its mass can leave in one step. Complements the CFL, which bounds wave speeds but not the fraction of content a face flux may carry. |
| `RICH_INDIVIDUAL_THERMAL_LOSS_FRACTION` | Default `0.5`. Same rule as the mass-loss limit applied to the internal energy: bounds the next step so that, at the thermal-loss rate measured over the closed interval, no more than this fraction of the cell's thermal energy can leave in one step. |
| `RICH_INDIVIDUAL_WAKE_CHANGE_FRACTION` | Default `0.25`. Conserved-change wake: a passive cell whose accumulated absolute mass or energy change since activation exceeds this fraction of its activation value ends its interval at the next event. Rank 0 reports both guards per event as `INDIVIDUAL_CONSERVED_GUARD`. |
| `RICH_INDIVIDUAL_MESH_DRIFT_FRACTION` | Default `0.25` (`0` = off; `0` until 2026-09-23, `0.2` until 2026-09-24). Rank 0 prints the effective value once as `INDIVIDUAL_GUARDS mesh_drift_fraction=... mass_loss_fraction=... thermal_loss_fraction=... wake_change_fraction=...`. Mesh-deformation limit: at activation a cell's next step is bounded so that its generator closes at most this fraction of the distance to any neighbouring generator: `f * d / closing` with the closing speed `-(w_i - w_j).(x_i - x_j)/d`, only for approaching pairs (since 2026-09-24; before, the full relative speed `|w_i - w_j|` limited, so separating and sliding pairs did too). Tangential sliding is not bounded by this guard. The CFL bounds gas motion across the faces that exist when an interval opens and says nothing about the mesh redrawing itself, so a face that grows late in a long interval is still charged the whole of it. The limit is floored at one sixteenth of the cell's own CFL limit so that a degenerate generator pair cannot drive it to zero. Rank 0 reports `mesh_drift_limited` per event. Needed even with bin enforcement: a RoundCells kick held over a long interval overshoots the centroid (gain chi c dt / R) and the next kicks can drive two generators through their separation within one legal interval; on the TDE restart that aborted without it (cell 50107), 0.2 passed the abort point (early robustness gate, t >= 0.180; the run to t = 0.46 is pending) at 0.61x the events and 0.94x the step wall over the matched window t = 0.1412-0.1747 (not whole-run ratios). |
| `RICH_INDIVIDUAL_GUARD_FLOOR` | Default `apply` (unset, `1`, `on`, `true`, `yes`, `apply`; since 2026-09-25); `report`; `off` (`0`, `off`, `false`, `no`). Must agree on every rank. The mesh-drift, mass-loss and thermal-loss limits may shorten a cell's next step no further than the floor: the smallest hydro CFL/source limit over all owned cells, each cached from the cell's latest activation (or synchronized evaluation at a box growth), i.e. the hydro part of the step a global step would take. `report` computes and counts without changing any limit. Rank 0 prints `INDIVIDUAL_GUARD_FLOOR` for every event in which some guard limit was below the floor (floor, applied, owned cells with a cached value, guard limits below the floor per guard, the deepest example with position, density and speed) and `INDIVIDUAL_GUARD_FLOOR_SYNCHRONIZED` at box growths. Why: on the TDE (restart from t = 20.82) the thermal-loss guard held near-vacuum cells just inside the central sink radius at 0.02-0.05 of the global step and the neighbour closure spread that bin, so events came every 0.06 of the global step; with the floor the individual trial ran 4.4x faster (0.62x global against 0.14x) and its state at t = 20.9015 was closer to the global run's (mass-weighted density difference 1.2e-2 against 2.7e-2). On the crash-50107 restart (t = 0.141-0.20) the drift and mass guards never fell below the floor, so the mesh-drift protection is unchanged there. Not a bound on a fine cell next to a coarse neighbour: a global step would refresh both. |
| `RICH_INDIVIDUAL_TRACE_CELL_IDS` | Default unset (off). Comma-separated stable cell IDs, identical on every rank. At every event whose mesh holds a listed cell as owned, rank 0 prints `INDIVIDUAL_CELL_TRACE` (interval, generator, centroid, width, gas velocity, the point velocity of the interval just closed and the one installed now), one `INDIVIDUAL_CELL_TRACE_FACE` per face (neighbour ID and owner, its generator and installed velocity, separation, relative and approach speed, face area), and `INDIVIDUAL_CELL_TRACE_LIMITS` for an activating traced cell (hydro limit, drift timescale `min |r_i - r_j| / |w_i - w_j|`, mass- and thermal-loss limits, the hydro step's final limit). Diagnostic only; set it in the submitting shell, not through `sbatch --export=VAR=...` (which splits on commas). |
| `RICH_CFL_DECISION_TRACE` | Default unset (off). Global steps only: for every `CourantFriedrichsLewy` evaluation (two per global step, before and after the point-velocity fix; the second returns the accepted step) rank 0 prints `CFL_DECISION` with the time, the returned step, the binding criterion (`raw` CFL, `force` source limit, or `cap` from the previous `SetTimeStep`), the raw winner's stable ID, rank, width, effective radius, sound speed, speed, density and largest fluid-face normal speed, the cap's value and lifecycle (`cap_armed`, `cap_reduced`, `cap_cleared`), and whether every rank returned the same step and cap (`result_agree`, `cap_agree`). Collective; must agree on every rank. Neutral on the TDE runs checked (bitwise-identical snapshots). |
| `RICH_INDIVIDUAL_RADIATION_INCREMENT_LIMIT` | Default `0` (off; `0.15` is the value of decision D5). Grey radiation only. Bounds a cell's next step to `applied_dt x f / r`, where `r` is the larger relative change of its internal and radiation energy density over the event's radiation update (both event-start values positive and finite, else the cell is skipped). Shortens only. Rank 0 reports `INDIVIDUAL_RADIATION_INCREMENT_LIMITED` (count, tightest ratio, one example cell) for every event in which it shortened at least one cell's limit. |
| `RICH_RUNTIME_LOG=detailed` with the guards | Adds the tightest cell's ID and ratio to every `INDIVIDUAL_CONSERVED_GUARD` line, including `step_overruns`: the count of intervals that outlasted the timestep limit in force when they opened. A nonzero `step_overruns` means a computed limit was not honoured, which no additional limit can compensate for. |
| `RICH_INDIVIDUAL_FIRST_HALF_FROM_CACHE` | Default on. Skip the interval-start mesh build when every rank's source term can apply its first half from cached per-cell state (a conservative force with every active cell's half kick pending). `0` builds it at every event as before. |
| `RICH_INDIVIDUAL_ADAPTIVE_MODE` and `RICH_ADAPTIVE_*` | Adaptive integration mode: let the run switch between global and individual stepping from measured throughput and the potential-gain bound of the CFL distribution. See the section above for the parameters and their defaults. |
| `RICH_INDIVIDUAL_PARTIAL_THRESHOLD_GLOBAL` | Default off. `1` judges the partial-build closure threshold (`partial_build_fraction`) on the whole mesh, summing the target and the owned cells over all ranks, instead of per rank. Measured worse on the TDE (job 10199567 against 10199440: 366 s against 146 s over the same 24 events): a partial build whose target covers most of one rank costs that rank about as much as a full build, and every rank waits for it. Kept for comparison only. |
| `RICH_INDIVIDUAL_PARTIAL_BUILD_FRACTION` | Unset keeps the scheduler option `partial_build_fraction` (0.5). A number in (0, 1] overrides the per-rank closure threshold, so the cost of partial builds above the default can be measured; it must be the same on every rank. With `RICH_INDIVIDUAL_PERF_TRACE=1`, rank 0 prints one `INDIVIDUAL_MESH_BUILD` record per event-mesh build: mesh (`first_half` or `event`), result (`full` or `partial`) and reason, partial attempts, the threshold, the largest per-rank target before and after closure growth, the largest fraction of a rank's owned cells with that rank, and the maximum over ranks of the wall time (whole call, partial attempts, full build). |
| `RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS` | Default on: a rejected radiation candidate lowers the next hydro bin of the cells it touched (all active cells for a collective failure), which the neighbour closure then cascades. `0` leaves the bins to the physical radiation limiter and absorbs the rejection by sub-cycling the radiation inside the event, as the global scheme sub-steps a rejected candidate inside its step; the retry cooldown in the defect accounting still carries the accepted fraction to the next event. Values must agree on every MPI rank. |
| `RICH_INDIVIDUAL_ADJACENCY_SEED` | Default on. Keep a per-cell face-adjacency cache from each event mesh and seed the partial target with the two-cell reconstruction shell before building, so the closure check rarely rebuilds. `0` disables both; values must agree on every MPI rank. |
| `RICH_VERIFY_PARTIAL_BUILD` | Enables full-versus-partial geometry parity. The regression drivers read it at setup; the hydro step also honours it at every partial build, so a restart whose checkpoint carries `verify_partial_build = false` can still be verified. |
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

The active Hilbert cache uses the global set of nonempty active time bins as
its key because one event can contain cells from several bins. Passive cells
have zero balance weight. Only Hilbert cut coordinates are cached; every reuse
is remeasured against current active cells and current geometry. A failed
active-balance, owned-cell skew, or nonempty-rank safety check leaves ownership
unchanged. An
accepted change migrates primitives, conserved state, scheduler state, and
registered radiation and gravity buffers through the normal stable-ID path,
then rebuilds the event context before hydro, radiation, or gravity runs. The
cache is transient and is reconstructed after restart. The production runner
recovers all three active-Hilbert controls when a continuation environment
omits them and rejects any mid-campaign change.

Automatic selection leaves non-Hilbert or otherwise unsupported individual
setups on their existing load-balancing path. Explicitly setting the cache
selector to `1` instead reports the incompatibility before the first event.

Individual-timestep checkpoint format version 10 stores the effective
all-source sweep interval and the last completed sweep tick. Older checkpoints
start a new interval at restart. Version 9 stores each cell's pending
neighbor-propagation bin and restores older checkpoints with no pending source
before applying full neighbor closure. Version 8 stores the mixed defect policy
and the monotonic latch in every rank piece. Version 7 stores the latch with the
legacy defect policy; versions 1--6 reconstruct the latch as false. A restored
true latch remains authoritative even when the runtime selector is absent. A pre-latch
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
