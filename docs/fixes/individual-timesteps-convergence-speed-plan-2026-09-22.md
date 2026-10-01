# RICH individual timesteps: convergence failures and speed — agreed plan

> **Superseded 2026-09-22 by [`individual-timesteps-convergence-speed-plan-2026-09-22-r2.md`](individual-timesteps-convergence-speed-plan-2026-09-22-r2.md)** (second convergence run, Codex gpt-6-astra at high effort, approved). Key change: R2's robustness gate is OPEN, not met (matched-window global retries 1913/unit t vs arm D 2108).

Status: approved by Codex (read-only review, 2026-09-22 13:3x, `VERDICT: APPROVE`, one non-gating
note applied below) after the four-round Claude/Codex convergence run
`~/.codex-converge/runs/20260922-123830-2516885` ended at its round cap with only speed items
open. Authored by Claude; reconciled with measurements from this session's runs.

Reconciles the Claude/Codex convergence run `20260922-123830-2516885` (4 rounds; Codex approved
the convergence diagnosis and remedies R1-R3 by round 2 and last objected only to speed items
that stopped at instrumentation) with measurements the read-only agents could not make. Every
number below is from the named log; code cited as path:line. Worktree
`/home/elads/RICH-ablation-integration`, branch `codex/individual-timesteps`, uncommitted.

## 0. Measured inputs the draft flagged as missing ("step 0"), now done

Exact simulated-time spans from `RICH_STEP` `t_start`/`t_end`; retries = `RICH_RETRY` lines, one
per rejected candidate attempt (`attempt=N`), so 79 (earlier quote) was the count in the common
window t <= 0.4191 and 137 is the whole file; both definitions are consistent.

| run | span | events or steps per unit t | retries per unit t | negative/invalid energy | wall s per unit t |
|---|---|---|---|---|---|
| A `meshab_on_10199440.txt` (retries lower bins, default) | 0.375268-0.419568 = 0.0443 | 9458 | 10451 | 0 | 53032 |
| D `meshab_noretrybin_10199568.txt` (`RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS=0`) | 0.375268-0.440268 = 0.0650 | 1938 | 2108 | 0 | 36205 |
| global `output_10199442.txt` | 0.000886-0.460654 = 0.4598 | 557 | 2229 | 4 | 3561 |

Consequences for the draft: the estimate that arm D's normalized retry rate (5000/unit t) exceeds
global's was wrong; measured, D is at 0.95x global with zero invalid-energy failures against
global's four. R2's robustness gate (a)-(c) is met in this window without R1. The cadence gain
of R2 is 4.9x (9458 -> 1938 events per unit t), throughput 1.47x; the remaining gap to global
is 10x in wall per unit t, 3.5x in cadence.

Full-file MadVoro phase distribution (`meshab_detail_10199448.txt`, 264 partial + 16 full builds):

| phase | median | p90 | note |
|---|---|---|---|
| bringing ghosts | 0.47 s | 1.29 s | max 40 s inside full builds |
| preparing (incl. points-manager "exchange" 0.305 s) | 0.31 s | 0.33 s | fixed, independent of target |
| build Voronoi from Delaunay | 0.0085 s | 0.021 s | |
| initial build (Delaunay of targets) | ~0 | 0.0005 s | |
| partial build total | 0.754 s | 1.15 s | 0.54 s median after the warm-target fix (arm E `meshab_nowarm_10199573.txt`) |
| full build total | 25.9 s | | 16 in 221 events |

Full-build events: A 30 of 419 (676 s = 61% of its 1105 s mesh); D 36 of 126 (862 s = 37% of
wall). Whole-mesh closure threshold (arm C `meshab_thr_10199567.txt`) was worse: 366 s vs 146 s
over the same 24 events, because a partial build whose target covers most of one rank costs
8-28 s (events 10564-10572); raising the per-rank fraction moves in the same direction.
Periodic full-source sweep: 223 in 11319 events at seconds_max 0.06 s (`RICH_FULL_SOURCE_SWEEP`
records, `output_10199059.txt`): not the scheduler residual. FMM (throttled probe
`fmm_10199396.txt`, restart): `total_mean` median 0.171 s per solve, 51 rebuilds in 2117 solves
(2.4%); the rebuild-conditioned topology phase reaches 0.87 s (`topology_max`), while the
overall `total_mean` maximum is 1.06 s -> rebuild share ~12% of FMM time; the evolved long run pays 1.2 s per
event (handoff: 0.079 s from a snapshot vs 2.085 s evolved to the same state).

## 1. Convergence: ranked root causes (as agreed in the loop, unchanged)

1A PROVEN. `Diffusion::calculateIndividualTimeSteps` (`source/Radiation/Diffusion.cpp:474-489`)
limits dt by a relative Er change whose denominator carries the global floor `0.02*max_Er`; a
near-vacuum cell has `difference -> 0`, `suggested_dt` saturates at `nominal_dt*2` every event, and
`chooseNextBin` (`IndividualTimeStep.cpp`) grows it one bin per activation to the cap. Global
uses the same formula but reduces to one scalar dt set by well-coupled cells
(`Diffusion.cpp:265-292`), so vacuum cells never run at their own dt. The rejection then comes
from `AssessAndApplyHistoricalMGPositiveFloor` (`conj_grad_solve.hpp:336-411, 565-581`): the
single-cell limit compares injected energy EXTENT to the peak cell's extent (1e-7), so a cell
55x the peak cell's volume fails at a 2.3e-9 density undershoot (record for cell 32711 in
`output_10199059.txt`; emitted key `historical_positive_floor_single_cell_limit`,
`conj_grad_solve.hpp:1436-1439`).

1B PROVEN (mechanism and now the normalized rate). A rejected candidate is sub-cycled inside the
event (`RadiationStep::stepIndividual`, `RadiationStep.cpp` ~740-905) and, with the default
switch, its fraction is folded into the next hydro bin of the affected cells (~880-905);
`limitNeighborBins` cascades the lowered bin. Global halves `dt_try` inside the step and leaves
the hydro dt alone (`RadiationStep::step` ~400-520). Measured effect of the fold: table above.

1C mechanism PROVEN, current rate NOT measured on this worktree: pre-pericentre
`negative or invalid energy {during,after} radiation update` (`Diffusion.cpp:1148, 1288`), 214
vs 33 in the 2026-09-19 runs; the `RICH_INDIVIDUAL_THERMAL_LOSS_FRACTION` guard postdates part of
that evidence. Diagnostic Run #1 below.

## 2. Remedies, in order (R1 and R3 as agreed; R2's gating corrected by the measurement)

R2 — stop folding rejected radiation fractions into hydro bins: flip the default of
`RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS` to off (`RadiationStep.cpp` ~333-370, ~880-905).
Numerics: the solve, its transactional rollback and its sub-cycling are untouched; only the next
bin choice changes; positivity and conservation machinery unchanged. Failure mode: a cell that
re-rejects every event at an unchanged bin pays repeated sub-cycles; the retry cooldown in
`IndividualRadiationDefectAccounting` already carries the accepted fraction across events, and R1
removes the cause. Gate, already met post-pericentre by arm D (retries per unit t 2108 <= 2229
global; 0 invalid-energy failures; same endpoint reached, exit 0); still required: one fresh
pre-pericentre run (t -1.52 to -1.02) with the switch off, compared with `output_10199025.txt`
on the same window: retries per unit t <= global's, 0 invalid-energy failures, exit 0. Decision:
default off now; the switch stays for bisection.

R1 — bound bin growth for cells whose volume is anomalously large relative to the active set
(`Diffusion::calculateIndividualTimeSteps`, `Diffusion.cpp:474-489`): keep the existing
`difference` untouched and replace the fixed growth cap `nominal_dt*2.0` by
`nominal_dt*growth_cap` with `growth_cap = clamp(2/volume_ratio, 1, 2)`,
`volume_ratio = V_i / mean active volume` (one extra SUM/SUM reduction). Fail-closed by
construction: `growth_cap <= 2` always, so `suggested_dt` can only shrink. Known limitation: an
active set of uniformly large vacuum cells is not throttled (needs a domain-wide volume
reference; deferred). Switch `RICH_INDIVIDUAL_RADIATION_DT_VOLUME_CAP`, default off until the
gate passes. Gate (all four, probe from snapshot 23, on vs off, exact `t_start`/`t_end`):
(a) `historical_positive_floor_single_cell_limit` retries per unit t <= 0.5x arm D's 2108;
(b) events per unit t <= 1.1x arm D's 1938 (no cadence regression from over-throttling);
(c) fraction of events with fewer than 3 active cells or active-volume spread < 2 stays <= 20%,
else the domain-wide reference becomes required; (d) rank-0 record per event with count of
capped cells and one representative cell (`volume_ratio`, `growth_cap`, `suggested_dt`, the
uncapped value). R1 is the cadence lever that remains after R2 (1938 -> toward 557 events per
unit t); it is no longer a prerequisite for R2.

R3 — Diagnostic Run #1 before touching the pre-pericentre thermal path: fresh start to
t = -1.02 on the current worktree with R2 default, histogram of `RICH_RETRY` reasons vs
`output_10199025.txt`; decide from that whether `negative_or_invalid_energy_*` still needs a
remedy. No code change until then.

Not changed: the positivity floor's extent-vs-density metric in `conj_grad_solve.hpp` (48% of
failures would still fail a density metric at 1e-7; a solver policy change shared with global;
user decision, documented).

## 3. Speed plan, ranked by measured seconds

Per-event floor post-pericentre (arm D, 126 events, 2353 s): mesh 1114 s (36 full builds =
862 s; ~90 partial builds ~0.6 s), untraced scheduler-side 699 s, radiation 462 s, gravity
(restart-fresh) 14 s, hydro proper 65 s. In an evolved long run gravity is ~1.2 s per event.

S1 (done, measured) — first-half mesh skip, adjacency-seeded target, no warm target:
builds per event 4.9 -> 1.3, partial build 0.89 -> 0.54 s, small event 3.36 -> 1.43 s, parity
0 mismatches over 408 events. Switches `RICH_INDIVIDUAL_FIRST_HALF_FROM_CACHE`,
`RICH_INDIVIDUAL_ADJACENCY_SEED`, both default on.

S2 (cadence) — R2 now, R1 next: 4.9x fewer events measured; R1's gain bounded by 3.5x.

S3 — full builds under the drifted decomposition (largest remaining floor item: 862 s of 2353 s
in arm D; 26-30 s each vs 0.25 s pre-pericentre; ghost bringing up to 40 s). Concrete change:
add a repartition trigger to `Simulation::stepIndividual` that calls the existing
`rebalanceCommittedIndividualState` (`Simulation.cpp` ~2411) when the last full build's
`mesh_s` exceeded `RICH_INDIVIDUAL_REBALANCE_FULL_BUILD_SECONDS` (default off; candidate 5 s)
and at least `cooldown_events` since the last repartition; it runs at the synchronized event
that follows (the existing machinery migrates cells, scheduler states and caches). Numerics:
ownership only; the same committed state on a different partition; every physics step already
implements `beforeIndividualRebalance`. Risk: the repartition itself costs two full builds
(~60 s) and the routing oct-tree must be refreshed (`RICH_OCT_ROUTING_ALWAYS_REBUILD=1` forces
it); if the 26 s is stream geometry rather than incoherent domains the gain is nil.
Expected: full build 26-30 s -> 1-3 s if domains are the cause (pre-pericentre value 0.25 s);
arm-D-equivalent saving 700-800 s of 2353 s. Gate: `INDIVIDUAL_LOAD_BALANCE` record with
migrated cells; next full build `mesh_s` <= 3 s; parity 0 mismatches over the following 100
events; zero errors; `RICH_STEP` wall per unit t down >= 25%. Rejected alternative: raising
`partial_build_fraction` above 0.5 (draft item 3), because arm C measured larger per-rank
partial targets at 8-28 s each; kept: the rank-0 record when `exceeds_threshold`
(`hdsim_3d.cpp` ~1533-1551) fires, with count, representative rank, local target and threshold.

S4 — partial-build fixed cost (0.54 s per event, ~90% of events): candidate A, skip the
points-manager exchange bookkeeping in `Voronoi3D::PrepareToBuildParallel`
(`Voronoi3D.hpp:1818-1892`, `pointsManager->update` at :1851) when
`suppressRebalancing && suppressExchange`, keeping the oct-tree routing refresh that the
suppressed-exchange path installs (`DistributedOctEnvAgent.hpp`; the routing must still follow
actual positions, throttled by the existing rebuild interval). Switch
`RICH_INDIVIDUAL_MESH_SKIP_MANAGER_UPDATE`, default off. Expected: -0.3 s of 0.54 s per partial
build (the measured "exchange" 0.305 s). Gate: parity 0 mismatches over >= 400 events with the
switch on (the `meshab_verify_10199446.txt` method, `RICH_VERIFY_PARTIAL_BUILD=1`); "Time for
preparing" median <= 0.05 s; `mesh_builds` and full-build count unchanged. Ghost bringing
(0.47 s median): instrument `totalBigQueries`/`totalSmallQueries` (`BringGhostPointsToBuild`,
`Voronoi3D.hpp:3105-3364`) as rank-0 aggregates first; S3 is expected to shrink it too.

S5 — untraced scheduler-side cost (699 s of 2353 s in arm D, 0.39 s per small event, up to
5.5 s per large event): first add `wake_s`, `commit_s`, `suggest_s`, `sync_s` fields to
`RICH_STEP` (`Simulation::stepIndividual`, around `limitIndividualTreeWakeTimeSteps`,
`commitEvent`, the suggest loop), rank-max like the other fields. Then the two concrete
candidates, chosen by those fields: (i) `limitNeighborBins` (`IndividualTimeStep.cpp` ~1188+)
exchanges the whole mesh's bin snapshot (`SyncCanonicalDataToMesh`) plus an all-to-all whenever
any bin was lowered, i.e. nearly every event; restrict the snapshot to cells adjacent to a
propagation source (sparse exchange keyed by owner), switch
`RICH_INDIVIDUAL_SPARSE_BIN_CLOSURE`; (ii) the wake oct-tree is rebuilt over all local cells every
event (`limitIndividualTreeWakeTimeSteps`, `local_tree.insert` for every source); cache it and
rebuild only when generators moved more than a fraction of a cell, switch
`RICH_INDIVIDUAL_WAKE_TREE_CACHE`. Gates: the new fields account for >= 90% of the residual;
each change cuts its field by >= 50% with zero `INDIVIDUAL_BIN_OVERRUN` increase and identical
`active_bins` histograms on the probe window (bit-identical scheduler state is expected for (i)).
Dropped from the draft: widening `RICH_INDIVIDUAL_FULL_SOURCE_SWEEP_INTERVAL` (measured 0.06 s
per sweep, 223 sweeps in 11319 events).

S6 — FMM (1.2 s per event in evolved runs; 0.17 s per solve fresh). Measured: structural
rebuilds are 2.4% of solves and ~12% of FMM time under the current interval 32, so
`RICH_FMM_STRUCTURAL_INTERVAL=256` is bounded to ~0.02 s per event and is not the lever. The
lever is the 7-25x growth of `total_mean` between a fresh and an evolved state (accumulated
topology). Concrete change: a forced full topology rebuild plus redistribution when the running
median of `fmm_solve_trace total_mean` exceeds 3x its value after the last rebuild (switch
`RICH_FMM_RESET_FACTOR`, default off), and, as the structural item, targets-only evaluation in
`EvaluateIndividualTargets` (`FastMultipoleAcceleration3D.cpp:568`: build and LET once per event
but traverse only active targets). Gate for the reset: `total_mean` <= 0.3 s per solve over a
0.1-unit window from an evolved state (currently 1.2 s), `sampleDirectAccelerationError` within
its existing tolerance, zero errors.

S7 — adaptive integration controller (`Simulation::SetAdaptiveIntegrationMode`, built as
`rich_adaptive_20260922`, untested): validate functionally now on the snapshot-23 probe (expect
`RICH_MODE_DECISION action=probe_global` after 76 events, `RICH_MODE_SWITCH to=global`,
`adopt_global`, `gain_bound` 2-3, `stay_global_little_to_gain`), and re-validate its thresholds
after S2-S6 change the individual baseline.

## 4. Execution order

Session N: (1) R2 default flip; Diagnostic Run #1 (fresh pre-pericentre to t = -1.02, R2 off,
retry histogram vs global) — decides R3. (2) R1 implementation, default off, gate on the
snapshot-23 probe. (3) Instrumentation: closure-threshold trigger record, `RICH_STEP` scheduler
fields, ghost-query counters, `RICH_FMM_TRACE=1` on the probe. (4) S4 candidate A behind its
switch with the parity gate. (5) S7 functional run.

Session N+1: (6) S3 repartition trigger and its gate. (7) S5 candidates chosen by the new
fields. (8) S6 reset policy, then targets-only evaluation. (9) Re-validate S7 against the new
baseline; consider flipping R1 default.

Open measurements: pre-pericentre retry histogram on the current worktree (R3); whether the
26 s full build is decomposition or geometry (S3 gate answers it); the scheduler residual
split (S5 fields); FMM evolved-state trace (S6); R1's coverage criterion (c).
