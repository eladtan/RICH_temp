# Individual time steps: cadence work, results log (2026-09-27)

Plan: [individual-timesteps-cadence-plan-2026-09-27.md](individual-timesteps-cadence-plan-2026-09-27.md).
TDE, 256 ranks, restart from `snap_full_54`. Global references: 83-event window job 10208630 (122.565 s);
adaptive span to t = 21 job 10213458 (307.3 s, global only).

## Summary (final state, read this first)

**Individual stepping now beats global on both TDE spans with the new defaults** (binary rich_def2_20260927,
defaults only: `RICH_INDIVIDUAL_BIN_ANCHOR_MARGIN=0.8`, `RICH_INDIVIDUAL_MAX_BIN_SPREAD=2`, both unset in the
runs; every change astra-approved; production symlink not moved; nothing committed).

| Span | Before today | Now (defaults) | Global |
|---|---|---|---|
| 83-event window, individual `step_s` | 127.4 s (seg24) | **101.2 s (1.21×)**, job 10222320 | 122.565 s (target 1.15× = 106.6 s) |
| Adaptive to t = 21, `step_s` | 343.8 s (seg24) | **283.5 s (1.08×)**, job 10222321 | 307.3 s |
| Whole job, window / adaptive | 189 s / 384 s | 163 s / 322 s | 191 s / 341 s |

- Overruns against the m = 0 uncapped baseline at the same state and endpoint: window 3,430 / worst 2.0 against
  17,986 / 4.5; adaptive 5,943 / 2.0 against 19,954 / 5.5. Worst excess present at a closure lowering: 1.5.
- Terminal state against global (mass-weighted relative L1, density): window 4.79e-3 against 6.73e-3; t = 21
  2.12e-2 against 3.68e-2. Mass and momentum are identical before and after each mode switch.
- Tests: scheduler unit suite (MPI 4 ranks: program default, K = 2, K = 3, spacing rule; serial: default, K = 2),
  7/7 regressions with the new defaults, crash-50107 gates with K = 4 (job 10222218) and K = 2 (job 10222281) reach
  t = 0.20; the defaults-only crash gate (job 10222322) reaches t = 0.20 with no abort in 21 min (uncapped 33 min).
- Legacy behaviour: `RICH_INDIVIDUAL_BIN_ANCHOR_MARGIN=0 RICH_INDIVIDUAL_MAX_BIN_SPREAD=-1`. An explicitly
  configured time quantum is never re-anchored.
- The adaptive run includes the teammate session's gain-gate fix (stale face velocities after global AMR); adaptive
  runs before rich_cap2 (19:31) predate it.

## Summary of the first stages (kept for the record)

- **The cadence cascade is gone.** Adaptive to t = 21: 343.8 s (seg24, job 10213794) → **311.1 s / 309.1 s**
  (rich_wake3 job 10213824, rich_wake4 job 10222049), against 307.3 s global only. Event spacing stays at 512 or
  1024 × 2^20 ticks; 5 of 156 events are off-grid (signal or radiation wakes). Individual mode now covers 80 % of
  the simulated time (was 40 %) and advances 6.3e-4 per wall second, 1.08× a pure global run.
- **The 83-event window is unchanged** (129.6 s against 129.8 s for its paired default arm), as predicted: its
  anchor is dyadic and it never had rounded-down wakes. Its terminal state against global is identical to 5
  digits (mass-weighted L1 density 6.732e-3 in both).
- **Terminal state at t = 21 against global** (mass-weighted relative L1, cells that existed in snap_full_54):
  density 3.68e-2 (new) against 3.60e-2 (old), pressure 4.01e-2 against 3.92e-2, velocity 3.05e-4 against
  3.00e-4, with twice the individual-mode exposure.
- **Crash-50107 gate** (job 10213825, rich_wake3): reaches t = 0.20, no abort. It restarts from an individual
  checkpoint, so it also exercises the restore() fix.
- **7-case regression suite passes**; scheduler unit suite (serial and 4-rank MPI) passes, and each new test fails
  on the code it guards.

## What changed (binary rich_wake4_20260927, uncommitted)

| Stage | Change | Review |
|---|---|---|
| 2a | A remote neighbour-bin request is applied by the owner through `binnedEndTick`, and each cell applies the minimum of its local and remote requests once (binnedEndTick is not monotonic in the bin, so one-at-a-time application depended on arrival order) | astra r1 CHANGES_REQUIRED → r2 APPROVE |
| 2b | `restore()` rebuilds `minimum_occupied_bin_` (collective); it was 0 after a restart | same |
| 1 | Conserved-change wakes end at the next scheduled event, found by `finalizeChangeWakes` after commit, AMR and the load balance; the float spacing `event_time - previous_event_time` is gone; the next event tick is checked unchanged every event; passive AMR merge recipients are woken too | astra r1, r2 CHANGES_REQUIRED → r3 APPROVE |
| 1 (diag.) | `IndividualChangeWakeAccounting`: every woken cell is sampled once before its own update at activation, or censored with the ratio it had reached; `INDIVIDUAL_CHANGE_WAKE_ACTIVATION`, `INDIVIDUAL_EVENT_DEADLINE` (cadence trace) | astra APPROVE |
| 3 | `RICH_INDIVIDUAL_CLOSURE_REEXPAND=1`: −8.8 s on the window (129.8 → 121.0 s), 0 parity mismatches over 73 verified partial builds (job 10213789); terminal state differs at round-off (density L1 6e-12) | astra **KEEP_OFF** until a same-target rebuild comparison before physics advances |

## Open findings

- **Overruns.** Overrun cells per individual event 128 against 95, worst interval/allowance 5.5 against 4.0.
  Per unit individual-mode simulated time the incidence is unchanged (≈139.7k against 139.9k). Likely cause: the
  cascade's spurious early wakes kept restarting passive intervals. astra: PROCEED only after a matched A/B;
  the test design is being settled (interior unsynchronized restarts do not exist, adaptive schedules differ
  between runs).
- **Wake accuracy.** 157k of 280k sampled woken cells exceed 2× the wake fraction at activation (max 12). The
  ratio is already above 1 at issue in most events, above 5 in 37 of 156: one event's passive flux changes those
  cells by several times their content. No deadline rule bounds that; it is a detection limit of the guard, not
  scheduling lateness. It is measured for the first time here, so there is no before/after comparison.

## Where the remaining individual time goes (job 10222049, 156 events, 224 s)

- **All-active events: 25 events, 82.9 s (37 %).** They fall at event 1, 2 (or 3), 4 (or 5), 8, 16, 32 of each
  individual period, plus the synchronizing event that ends it. At entry every cell starts in bin 30 and
  `chooseNextBin` raises a bin by at most one per completed interval, so the bulk of the 4.66M cells climbs
  31 → 32 → … and moves together at power-of-two event counts; every re-entry after a global probe starts the climb
  again.
- **AMR passes: 15 events, 47.6 s (amr_s 26.3 s).** An individual pass costs about what a global one does
  (1.6-1.9 s; global steps that include AMR take 3.6-4.1 s against 2.17 s). The difference is cadence: every 10
  events against every 10 global steps, and events come more often than global steps.
- **Small events (< 3 % active): 119 events, 111 s, 0.94 s each.**
- **Why events come often.** In the window the bin-30 anchor is the ramp-limited global dt (0.00093 against a
  CFL limit of 0.0015), so events are 1.6× more frequent than needed. In later periods the anchor sits at the CFL
  limit and a handful of cells fall into bin 29: 93 of 156 events have bin-29 spacing. Both point at Stage 4's
  anchor margin, which the cascade had confounded overnight.
- **Global probes.** The controller measures individual at 7.4-7.5e-4 simulated time per wall second against
  5.3-5.9e-4 for global, yet re-probes global after each dwell (8 steps); the backoff doubles the dwell as designed.
  The probes themselves cost ~4-5 s each net, but each re-entry also repeats the bin climb above.
- **Whole-job wall.** The window job: 189 s (individual, job 10213543) against 191 s (global, job 10208630);
  adaptive to t = 21: 349 s (job 10222049) against 341 s (job 10213458). The `step_s` sums behind the gates exclude
  startup and output, which are common to both.

## In progress

- Stage 4 anchor margin (`RICH_INDIVIDUAL_BIN_ANCHOR_MARGIN`, binary rich_anc2) with the overnight review's three
  findings fixed and the anchor taken from the gain estimate's reference step; under astra review.
- (done, below) Overrun fork A/B.

## Overrun fork A/B (matched state, deterministic)

Parent: job 10222054 (rich_wake4, adaptive) writes a synchronized individual checkpoint at t = 20.88, inside the
period entered at 20.8717512386 with the non-dyadic step 0.00154784854196. Arms restart from it with the controller
off (individual throughout) to t = 20.93, binary rich_anc3 (m = 0): Stage 1 (`next_event`) against the reference
(`RICH_INDIVIDUAL_CHANGE_WAKE_RULE=spacing_ticks`, the old previous-spacing wake in exact ticks).

| | Stage 1 (job 10222168) | Reference (job 10222169) |
|---|---|---|
| Events to t = 20.93 | 56 | 77 |
| Individual wall | 88.8 s | 106.1 s |
| Overrun cells, total | 6,491 | 7,397 |
| Worst interval / allowance | 5.0 | 5.0 |
| Closure lowerings / already overdue | 69,164 / 6,302 | 71,321 / 7,071 |
| Worst excess already present at a lowering | 4.5 | 4.5 |

- The reference still ratchets in exact ticks (nine events one 2^20 unit apart, many non-power-of-two spacings).
- Traced worst cells (jobs 10222170/71, `INDIVIDUAL_SCHED_TRACE`): in both arms the worst overrun is a cell in a
  bin-33 interval that the neighbour-bin closure drops three bins at once (33 → 30) when a neighbour falls to bin 29,
  4.5 bin-30 lengths into its interval; it activates at the next finest-bin tick with 5.0. Stage 1: cell 3683754;
  reference: cell 4499109 (in the Stage 1 arm that cell has no such drop, and vice versa).
- So overruns are closure-driven: 4.5 of the 5.0 is present at the lowering and the wait to the next finest-bin
  tick adds 0.5. The rule only changes the trajectory. Stage 1 has 12 % fewer overrun cell-events and the same
  worst case. **astra: PROCEED** on the overrun gate, restated as "from the same initial state through the same
  physical endpoint, no increase in summed overrun cell-events or worst interval/allowance; report the ratio at
  lowering and the growth after it separately". The closure-driven excess predates Stage 1 and is tracked as a
  separate accuracy concern (preventing it needs earlier detection or a validated catch-up scheme; its physical
  error should be measured before adding that).


## Stage 4: anchor margin (binary rich_anc5, astra APPROVE)

`RICH_INDIVIDUAL_BIN_ANCHOR_MARGIN = m` anchors bin 30 at m × the last global step's CFL/source suggestion
(taken inside the global step, before its post-step AMR) instead of the ramp-limited global dt; the first
interval is the coarsest bin within the global dt; the gain estimate uses the same grid; the TDE sink's first
individual event is scaled by its interval. Default 0 (unchanged).

| Run | Individual wall | Events | AMR passes | Overrun cell-events / worst | L1 vs global (density) |
|---|---|---|---|---|---|
| Window m = 0 (job 10222172) | 126.2 s | 83 | 9 | 17,986 / 4.5 | 6.73e-3 |
| Window m = 0.8 (job 10222173) | **98.7 s** | 62 | 7 | 10,436 / **14.5** | 4.85e-3 |
| Window m = 0.9 (job 10222174) | 105.0 s | 67 | 7 | 14,355 / 8.5 | 5.34e-3 |
| Adaptive m = 0 (job 10222049) | 224.0 s (309.1 total) | 156 | 14 | 19,954 / 5.5 | 3.68e-2 |
| Adaptive m = 0.8 (job 10222175) | 194.4 s (**280.0** total) | 119 | 10 | 15,217 / 4.5 | 2.81e-2 |
| Adaptive m = 0.9 (job 10222176) | 203.8 s (288.9 total) | 124 | 12 | 16,435 / 7.0 | 3.14e-2 |

- m = 0.8: window 98.7 s against the 122.565 s global reference (**1.24×**, the 106.6 s target is met); adaptive
  280.0 s against 307.3 s global only (1.10×). Both terminal states are closer to global than at m = 0.
- m = 0.9 brings back the bin-29 cliff (23 bin-29 spacings in the window).
- The window's 14.5 is one cell (2053215, traced in job 10222181): it climbed to bin 35, and one event before the
  terminal output the closure dropped it 35 → 31 when it was 14 bin-31 lengths into its interval; it ended at the
  off-grid terminal clamp. The closure-driven mechanism again, enlarged by the bulk sitting at bin 35.

## Bin-spread cap (binary rich_cap2; production code astra-reviewed, tests extended per review)

`RICH_INDIVIDUAL_MAX_BIN_SPREAD = K`: after every commit (last, after the propagation closure) and after
synchronized limits, no cell's bin exceeds the global finest bin + K; a capped cell takes the begin-aware end
(an overdue one ends at the next finest-bin tick). Off by default. astra's recommendation for the window overrun.

| Window run | Individual wall | Overrun cell-events / worst | L1 vs global (density) |
|---|---|---|---|
| m = 0, uncapped (job 10222172) | 126.2 s | 17,986 / 4.5 | 6.73e-3 |
| m = 0.8, uncapped (job 10222173) | 98.7 s | 10,436 / 14.5 | 4.85e-3 |
| m = 0.8, K = 4 (job 10222212) | **97.8 s** | 9,404 / 4.0 | 4.80e-3 |
| m = 0.8, K = 2 (job 10222210) | **102.8 s** | **3,430 / 2.0** | 4.79e-3 |
| m = 0, K = 4 (job 10222209) | 119.9 s | 13,327 / 4.5 | 6.67e-3 |

| Adaptive to t = 21 | Total | Whole job | Overrun / worst | L1 vs global (density) |
|---|---|---|---|---|
| m = 0 (job 10222049) | 309.1 s | 349 s | 19,954 / 5.5 | 3.68e-2 |
| m = 0.8, K = 4 (job 10222216) | **277.1 s** | 315 s | 15,217 / 4.5 | 2.81e-2 |
| m = 0.8, K = 2 (job 10222217) | **280.4 s** | 317 s | **5,943 / 2.0** | **2.12e-2** |
| Global only (job 10213458) | 307.3 s | 341 s | | |

- m = 0.8 with K = 2: window 1.19× the 122.565 s global reference (target 1.15×), adaptive 1.10× global only;
  overruns and worst case well below the m = 0 baseline; terminal state the closest to global of all arms.

## Open items

- `RICH_INDIVIDUAL_CLOSURE_REEXPAND=1` (-8.8 s on the window): astra KEEP_OFF. The same-target rebuild comparison it
  asked for is implemented (`RICH_INDIVIDUAL_REEXPAND_VERIFY`, debug) but its review is CHANGES_REQUIRED (face
  matching for repeated identities, finite-value checks, parity-check interaction, re-expansion counter) and it
  has not been run.
- Stage 5 (AMR no-op path, AMR/mesh coordination) not started; AMR passes are ~1.7 s each and now fewer
  (7 in the window at m = 0.8 against 9).
- Closure-driven overruns (a neighbour's sudden bin drop lowering an aged coarse cell) are a pre-existing accuracy
  concern; the cap bounds them (worst 2.0 in both spans) but does not remove them.
- The wake accuracy report shows ratios above 1 already at issue: a guard detection limit, not scheduling.
- Everything is uncommitted; a teammate session's gain-gate fix shares files (hdsim_3d, Simulation, PhysicsStep,
  HydroStep): stage exact hunks when committing.

## 2026-09-28: adaptive t=50 slowdown fixed (fixed bin-spread ceiling)

Cause (job 10222326, after t~30.2, ~0.33x global): the bin-spread cap was finest-bin + K, so one cell pinned at
bin 26 (likely the radiation retry limiter) dragged 8.9M cells to bin 28; the stale-baseline revert kept
dwellMultiplier 16 (1036-event dwell).

Fix (astra APPROVE, round 3): cap = fixed ceiling min(maximum_bin, initial_bin + K), also on anchored initialize
and forced-synchronized commits, cached minimum lowered with it; gain model on the same ceiling; dwellMultiplier
reset on domain change and stale-baseline revert; RICH_ADAPTIVE_DWELL_WALL_MAX (default 1800 s, 0 off) ends a
dwell after that wall time once past the ramp and sample gates. Binary rich_fx2_20260928. New unit tests (cap
scenarios, restore->AMR, controller via friend AdaptiveControllerTestAccess); each fix's mutation fails them.

A/B from snap_full_61 (t=29.4973, RICH_TDE_RESTART_FROM_SNAPSHOT=1, 16 bigrun nodes), step_s summed to t=31.10:

| run | step_s 29.497 -> 31.10 | vs global |
|---|---|---|
| global 10222325 (rich_def2) | 5535 | 1.00 |
| TDE_LTF 10222675 (fix) | 4694 | 1.18x faster |
| TDE_LTR 10222676 (fix + RADIATION_RETRY_LIMITS_BINS=0) | 4644 | 1.19x faster |

No collapse: bulk stays at bin 32 (dt ~4e-3) through t=30.2-30.5; retry-limit feedback off makes no material
difference. Controller probes still measure global from a ramping dt (tau_global 2.3-3.0e-4 vs steady ~3.4e-4),
open item. Conservation/physics comparison vs global still pending. Adaptive t=50 continuation: job 10222679
(TDE_L50F_ADAPT, rich_fx2, from snap_full_61).

## 2026-09-28 (later): late-phase parity found and removed

Late phase (t>33) had individual ~= global. Matched tests restart from the global run's own snap_full_65
(t=34.052, same state as the global reference exp_LONGGLOBAL_10222325.txt), with RICH_ADAPTIVE_PROBE_FRACTION=10 to
force a long individual stretch; throughput = sum of step_s over individual events, t=34.16-34.88.

| binary / arm | individual tau | vs global log (3.63e-4) | partial event | mesh attempts |
|---|---|---|---|---|
| rich_bs1 (fix A + batched supports), job 10222834 | 3.99e-4 | 1.10x | 2.39 s | 2.65 |
| rich_bs1 + RICH_INDIVIDUAL_MAX_BIN_SPREAD=3, job 10224349 | 3.93e-4 | 1.08x | 2.71 s | 2.65 (27 full fallbacks) |
| rich_ag2, job 10225875 | 4.33e-4 | 1.19x | 2.08 s | 2.40 |

Findings and fixes (each astra APPROVE, high+fast):
- Fix A (driver): full AMR pass at the first all-active event after RICH_TDE_FULL_AMR_INTERVAL_BINS (10) finest
  intervals; the every-10th-event pass aliased with all-active events (68 of 68 missed). Removals now match global,
  probe AMR bursts (up to 290k cells) gone.
- Batched closure supports (RICH_INDIVIDUAL_CLOSURE_BATCH_SUPPORTS): no effect by itself.
- Per-attempt closure counters (closure_a<k> in INDIVIDUAL_MESH_BUILD) showed ~4900 closure additions per partial
  event without a valid adjacency record: ResetIndividualMeshState cleared the cache after every changing AMR pass
  (117 of ~740 events). Now realigned by stable ID (RealignRecordsById).
- Routine active-only AMR on partial events dropped when full passes are on (fallback at 2x interval); scheduler
  remap skipped when no rank's AMR changed anything (closure only). Partial-event AMR 0.32 -> 0.02 s.
- K=3 rejected (all-active events dearer, more full-mesh fallbacks).
Accuracy counters unchanged (overrun 0.50/event, conserved guard 3.5/event, sink removal within 0.5%).
Remaining: depth-only closure rebuilds (55% of partial builds) need RICH_INDIVIDUAL_CLOSURE_REEXPAND, still KEEP_OFF
until its verifier is fixed; all-active event 6.9 s vs global step 5.1 s.

Production: job 10225877 (TDE_P50_ADAPT, rich_ag2_20260928, controller defaults) from global snap_full_65 to t=50;
t=34-41.1 overlaps the global reference for validation.
