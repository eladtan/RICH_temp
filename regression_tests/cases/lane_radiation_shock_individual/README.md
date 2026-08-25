# Lane-Emden multigroup radiation-shock benchmark

This case compares legacy global stepping, variable individual timesteps with
full meshes, and variable individual timesteps with automatic partial Voronoi
construction. Physics: moving-mesh hydro, distributed Barnes-Hut quadrupole
gravity, controlled AMR, and free-free multigroup diffusion with Compton,
Doppler, flux limiting, and hydro feedback.

The production configuration uses 2,000,000 initial cells, 16 logarithmic
groups from 1 eV to 2 MeV, 128 MPI ranks, and `0.20 t_dyn`. Runtime is measured
inside `test.cpp`; the comparison script never treats speedup as pass/fail.
Each production lane requests the `bigrun` partition's 21-day maximum because
the full-mesh reference cannot complete this workload within 48 hours.
The lane writes an atomic rolling checkpoint every 6 hours. After a completed
event at 20 days it checkpoints and requeues the same Slurm job, leaving one
day of margin before the scheduler limit. This is a planned segment boundary,
not a radiation-iteration cap or a forced solver quit. Two alternating restart
snapshots bound disk use. Per-rank sidecars preserve the original benchmark
baseline, controlled-AMR cadence and counters, history, work counters, phase
timing, and radiation-retry statistics, so final timing and comparisons cover
all segments. The driver validates the pointer, snapshot, and all 128 sidecars
before resuming.
The default timeline uses quantum `2.8475356730643257e-13 s`, initial bin 30,
and maximum bin 40. The legacy global lane is capped at the identical
`quantum * 2^40 = 0.31308985830411845 s` interval. These values may be
overridden, but `manifest.txt` records both caps and the comparator rejects a
campaign if the global and individual maximum or initial intervals differ.

`submit_campaign.sh` copies the built executable to a hash-named, read-only
file inside the campaign directory before submitting any job. Calibration and
all three production lanes use that immutable snapshot. Rebuilding the normal
`build/intelReleaseMPI` target therefore cannot invalidate demand-paged code in
an already running MPI job. Set `RICH_SOURCE_BINARY` to snapshot a binary from
an alternate build subdirectory.

The campaign explicitly unsets `RICH_QUIET`. Each lane retains complete mesh,
radiation-step, BiCGSTAB convergence, limiting-cell, and thermodynamic output
in `run.log`; selected MG diagnostics are also copied to
`mg_solver_diagnostics.log` when the run ends. This includes
`MG_SPECTRAL_POSITIVITY_REPAIR` and `MG_PASSIVE_ROUNDOFF_REPAIR`, which record
conservative repairs of roundoff-scale group extents at active--passive faces.

The production runner pins
`RICH_MG_INDIVIDUAL_PASSIVE_POLICY=dirichlet` only for the AutoPartial
`partial` lane. It unsets the deprecated shadow alias first. The
`full-variable` and `global` lanes leave the selector unset and therefore
retain the library default; the global radiation path is unchanged. Both
`run_info.txt` and `run.log` record the effective campaign selection.

In Dirichlet mode, passive primitive and conserved radiation states are frozen
for the candidate. The omitted equal-and-opposite active--passive transfer is
recorded as a signed, absolute, normalized, and worst-local conservation-defect
ledger. A candidate over the collective defect limits rolls back and retries;
only accepted candidates append their pending defect. This lane must therefore
report both ordinary total-energy drift and the defect ledger. It must not be
described as exactly conservative at active--passive interfaces.

`mg_solver_diagnostics.log` retains both
`MG_INDIVIDUAL_PASSIVE_POLICY` and every
`INDIVIDUAL_RADIATION_DEFECT status=accepted|rejected` record. The final
`counters.txt` records cumulative signed and absolute defect extents,
normalization scale, maximum event/local fractions, accepted/rejected/retry
counts, versioned limits, cooldown state, and `history_complete`.

MG timestep normalization is MPI-global in every mode. Each rank contributes
reference maxima from all canonical owned cells, never just the active partial
reconstruction closure. For an individual event, an updated active cell
replaces its canonical pre-radiation value when forming both `max_Er` and
`max_rhoT`; inactive owned cells retain their canonical primitive state. Ranks
with no active cells still enter the MPI reductions. `MG_TIMESTEP_LIMIT`
records `max_Er`, `max_rhoT`, `reference_scope`, and `growth_cap` so this
invariant is auditable.

Distributed-active BiCGSTAB also reports `MG_BICGSTAB_TOLERANCE` and
`MG_BICGSTAB_HISTORICAL_POLICY`. Global and active solves share the historical
diagonal-scaled squared-residual acceptance, including the established `max0`,
`max1`, negativity, and breakdown branches. The independently recomputed
componentwise backward error is retained as an MPI-global diagnostic, not as a
second acceptance gate. The normalization gate verifies the shared policy,
including the sparse case in which 127 ranks own no active cells.

When every globally owned cell is active, the mesh view supplies a validated
owned-to-canonical ID bijection, and all scheduled intervals agree, individual
MG uses the established global matrix and solver for that candidate. The
bijection may be identity or a post-AMR permutation. Fractional rejection and
rollback remain in the individual event loop.
`MG_INDIVIDUAL_ALL_ACTIVE_FAST_PATH` records the owned-cell count, mapping kind,
and candidate interval. Sparse events continue to use distributed-active
BiCGSTAB; the gate requires both paths explicitly. Set
`RICH_INDIVIDUAL_MAPPED_ALL_ACTIVE=0` to retain the legacy identity-only guard.
Grey and multigroup diffusion both use the shortcut after validating that their
global solvers are safe on MPI ranks with zero owned rows. Sparse events remain
on the active-row path.

Retry feedback preserves locality only for failures explicitly classified as
cell-local. Such a candidate with a representative stable cell ID limits only that cell's next interval; different failed cells
retain their own smallest successful fractions. Only an unattributed collective
failure—or a named matrix, face, or transactional failure without an explicit
cell-local classification—limits the complete active set. The retry summary records
`next-step limiter scope`, `limited cells`, and `minimum fraction`, making an
accidental all-cell resynchronization visible in production logs.

Both global and individual fractional retry loops hold a rejected candidate's
halved interval as an event-local safe ceiling. The ceiling doubles once only
after eight consecutive accepted candidates; another rejection lowers it and
restarts the cooldown. A matched 8-rank, 32,768-cell, 48-cycle A/B reduced MG
solves from 754 to 470, rejected probes from 231 to 71, and evolution time from
642.079 s to 464.771 s. Maximum profile, spectrum, and shock-property errors
were `2.442e-5`, `2.697e-6`, and `4.558e-4`, respectively, all below production
limits. Results are retained in
`regression_tests/results/global_retry_cooldown_smoke_48_20260813T115000Z`.

The different growth caps are intentional. The legacy global timestep is a
continuous value and retains its historical `1.4` damping cap. Individual
timesteps use power-of-two bins, so `2.0` permits exactly one aligned bin
increase; the scheduler independently forbids larger or unaligned increases.
The 128-rank normalization gate checks both formulas, compares global against
synchronized individual stepping, and compares sparse FullReference against
AutoPartial so a partial closure cannot silently set either reference scale.
The global-versus-synchronized reference-scale relative tolerance is `1e-6`;
the stricter sparse FullReference-versus-AutoPartial tolerance is `1e-7` for
both `max_Er` and `max_rhoT`.
For the sparse pair, only rank 0 owns the short-bin cell; the other 127 ranks
have no active cells and must still complete every normalization collective.
An additional synchronized run uses fewer generators than ranks, requires at
least one genuinely zero-owned rank, and must collectively enter and complete
the MG all-active global path.

### AutoPartial performance controls

The post-event load balancer is off by default until its focused MPI and
checkpoint gates pass. Enable it only in a disjoint validation root:

```bash
export RICH_INDIVIDUAL_AUTO_REBALANCE=1
export RICH_INDIVIDUAL_REBALANCE_THRESHOLD=1.25
export RICH_INDIVIDUAL_REBALANCE_AMR_THRESHOLD=1.5
export RICH_INDIVIDUAL_REBALANCE_COOLDOWN=8
export RICH_INDIVIDUAL_REBALANCE_AMORTIZATION=2
```

The decision is collective and occurs only after a committed event and AMR
update. It uses the existing physics weights, migrates scheduler state through
the standard transfer path, and records ownership epochs and migrated-cell
counts. `RICH_INDIVIDUAL_PERF_TRACE=1` additionally reports rank
min/median/mean/p95/max for physics, radiation gather/sync/driver/scatter,
event wall time, peak RSS, and global matrix construction. The trace is off by
default and must be A/B checked for less than 2% overhead.

Use `analyze_resource_balance.py` on matched trace logs to gate both rank RAM
equality and aggregate live memory.  The default late-window gates require
rank RSS max/mean <= 1.25, rank RSS p95/median <= 1.20, no increase in mean or
peak total live RSS, no increase in the maximum rank RSS, and no more than a 2%
matched event-wall regression.  Total live RSS is the reported rank-mean RSS
times the fixed 128-rank job size.

```bash
python3 analyze_resource_balance.py \
  --baseline /path/to/baseline/run.log \
  --optimized /path/to/optimized/run.log \
  --json-out /path/to/resource_balance.json
```

Use `capture_performance_provenance.py` before each isolated lane to hash the
source tree, executable, restart, all 128 sidecars, configuration, toolchain,
affinity, and runtime environment. Feed final paired elapsed times to
`analyze_performance_pairs.py`; it requires five pairs, every pair at least
10x, median at least 10.5x, and a one-sided 95% log-speedup lower bound of 10x.

```bash
gate_root="$(pwd)/regression_tests/results/lane_mg_timestep_gate_$(date -u +%Y%m%dT%H%M%SZ)"
sbatch --export=ALL,RICH_CASE_DIR="$(pwd)/regression_tests/cases/lane_radiation_shock_individual",RICH_MG_TIMESTEP_GATE_ROOT="${gate_root}" \
  regression_tests/cases/lane_radiation_shock_individual/submit_mg_timestep_normalization_gate.sbatch
```

```bash
cd /home/elads/RICH-ablation-integration
bash regression_tests/cases/lane_radiation_shock_individual/build_production.sh
bash regression_tests/cases/lane_radiation_shock_individual/submit_campaign.sh
```

For the 8-rank reduced correctness tier:

```bash
bash regression_tests/cases/lane_radiation_shock_individual/build_reduced.sh
sbatch regression_tests/cases/lane_radiation_shock_individual/submit_reduced.sbatch
```

Every lane writes live progress to `run.log` and `progress.tsv`.  Records include
the phase, mode, cycle, physical time, timestep, completion fraction, rank-zero
active/owned cells, and wall time.  `RICH_TEST_PROGRESS_WALL_SECONDS` controls
the wall-clock interval (default 60 seconds) and
`RICH_TEST_PROGRESS_CYCLES` provides a cycle-count fallback (default 100).
Across a planned restart, both files append and their evolution wall time stays
cumulative. `RICH_TEST_CHECKPOINT_WALL_SECONDS` controls rolling-checkpoint
frequency (production default 21,600 seconds), while
`RICH_TEST_SEGMENT_WALL_SECONDS` controls the event-boundary requeue threshold
(production default 1,728,000 seconds). Set either to zero to disable it.

Every lane also writes both endpoints. `initial_state.h5` and `final_state.h5`
are distributed restart snapshots. `initial_state.pvtu` and `final_state.pvtu`
are the MPI VTK XML master files; their per-rank `.vtu` pieces share the
corresponding `initial_state/` and `final_state/` directories with the HDF5 rank
files. The VTK fields include density, pressure, internal energy, temperature,
velocity, generator position, ID, volume, total and per-group radiation energy,
tracers, and stickers.

Before an individual lane writes its final endpoint, intervals that would cross
the requested time are shortened into one collective terminal event.  The event
context carries each cell's exact shortened interval, all owned cells are
activated at the endpoint, and the run records
`INDIVIDUAL_TERMINAL_EVENT_CLAMP`.  Ordinary evolution events remain on the
power-of-two timeline.

See `docs/user-guide/individual-timesteps.md` for the algorithm, validation
limits, artifacts, completed smoke tests, and operational details.

The current authoritative campaign is
`regression_tests/results/lane_radiation_shock_20260813T151423Z`, using immutable
binary SHA-256
`2cc601748f9598bfb67cf8d64546c8375b083204ec6d8481c3467935bdb40e0f`.
Calibration job `10120311` passed; production jobs are global `10120312`,
FullReference `10120313`, and AutoPartial `10120314`; comparator `10120315`, MG
normalization/empty-rank/restart gate `10120316`, and partial-AMR gate
`10120317` are dependency-queued.
