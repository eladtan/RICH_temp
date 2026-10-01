# Individual time steps: partial-event cost and cadence plan (2026-09-28)

Joint plan (Opus + gpt-6-astra: independent idea lists, cross reviews, diagnostics, review rounds). Inputs in the
session scratchpad `partial/`: briefing.md, opus_ideas.md, astra_ideas.md, opus_review_of_astra.md,
astra_review_of_opus.md, diagnostics.md, astra_diag.md, astra_plan_r*.md.

## Problem

Late-phase TDE (t>35, ~10M cells, 256 ranks), binary `rich_ag2_20260928`:

- **Occupancy.** 99.7-99.8% of cells sit in the ceiling bin (4x the finest dt), 15-35k in the middle bin, 100-1500
  in the finest bin. Every finest tick is an event: ~700 events per unit time, 75% partial (<1% active, mean ~10k
  cells), 25% all-active.
- **Cost per unit time.** From the late-window means: all-active 178 x 7.1 s = 1267 s plus partial 527 x 2.4-2.7 s
  = 1265-1423 s, total 2529-2687 s, against global 475 x 5.3 s = 2517 s: 0.94-1.00x. (The wider 0.83-1.19x quoted
  in the session comes from different windows of the production run and the matched tests.)
- **K.** K=3 halved the all-active events, but the removed ones reappeared as dearer mid/partial events (measured
  throughput unchanged). Under unchanged cadence and costs it would give ~2295 s, so K is re-tested once partial
  events are cheaper.

Two levers: the cost of a partial event (track A) and the number of events (track B).

### Measured partial event, mean of 2249 events, rank-max seconds

| Phase | Seconds |
|---|---|
| **Total** | **2.73** |
| Event mesh | 1.54 |
| Wake | 0.27 (tree 0.23) |
| Gravity source | 0.25 (1 FMM solve) |
| Active-Hilbert balance | 0.13 |
| Radiation | 0.13 |
| Extensive update | 0.11 |
| Rest | ~0.3 |

Median over ranks equals the maximum: every phase ends in a collective.

### Mesh details

- **Attempts.** Mean 2.24 per partial build: 66% take a second build only for depth promotions (no new cells); 26%
  take three builds (~1700 new depth-two supports after the first).
- **Concentration.** Worst rank's target is 20x the mean rank's (median).
- **Attempt time.** MadVoro rank-0 stage timers: ~0.34 s per partial attempt, of which ghost exchange 0.22 s
  (mostly waiting on busy ranks), prepare/exchange 0.05, Voronoi 0.04, tree 0.02. Timer scopes still differ
  between RICH and MadVoro; no RICH-side share is claimed before step 0.

### Cadence

- Finest-bin cells are there because of CFL 71%, scheduler (closure/alignment) 19%, thermal-loss guard 10%.
- The CFL limit is set by floor-density vacuum cells (rho~1e-20, c~1.08) near r~5. Their effective radius
  min(width, V/Amax) is 0.026 in the adaptive state against 0.041 in the global one; the minimum CFL is ~25% lower
  in the adaptive state.
- Suspected cause (confirmed in code, not yet as the cause): individual mode installs new point velocities only
  on active cells (hdsim_3d.cpp ~3249) while every generator drifts every event (~3085), so passive cells coast
  unregularised between their own events. The drift guard bounds approach, not tangential deformation (~3801).

## Principles

- No change to fluxes or conservation in track A. Keep the pairing of a generator's motion with the flux
  integration over its interval (hdsim_3d.cpp ~3049): never overwrite a passive velocity mid-interval.
- Every decision collective-consistent, with a rank-agreement check or collective rejection.
- Runtime criteria that switch both ways, not configuration advice.
- Judge each step by complete-event wall time per unit of simulated time, not by its own timer.

## Standard test

- **Restarts.** Matched restarts from one fixed snapshot, 16 bigrun nodes (prescan), `RICH_ADAPTIVE_PROBE_FRACTION=10`
  for a long individual stretch. Track A uses the global run's snap_full_65 (t=34.052) and the late adaptive
  snap_full_66 (t=36.156); track B uses one snapshot for both arms.
- **Provenance per arm.** Source revision and dirty patch (nested repos included), executable SHA-256, snapshot
  hash, compiler/MPI stack, full environment, compared simulated-time window.
- **Controls.** Speedup claims use a same-snapshot global control run with the same binary, not a historical global
  log (its mesh history differs, as the CFL diagnostics showed). The previous binary from the same restart is the
  second reference.
- **Noise.** Repeat one control to measure run-to-run noise; do not retain a gain below twice the noise. Include
  retries, fallbacks and amortized rebalance cost; report verification overhead separately.

## Acceptance rules

### Comparison definitions (used by every gate)

- **Scalar fields** (density, internal energy, pressure, radiation energy per group): pass if
  |a-b| <= atol + rtol*max(|a|,|b|), with rtol=1e-10 and atol = 1e-10 x (field floor or the 1st-percentile positive
  value of that field in the snapshot, whichever is larger). Any non-finite value in either run fails.
- **Vector fields**, per cell, never component by component: pass if ||a-b|| <= 1e-10 x S (a norm-based rtol with
  the scale as the absolute allowance, so zero vectors are covered). S is c+max(||a||,||b||) for velocity (c the
  cell's sound speed); rho x (c+max(||v_a||,||v_b||)) x V for momentum; the cell width w for centroids and
  generators.
- **Faces** are matched by stable identity (sorted pair of cell IDs). For every matched face, the flux of each
  conserved quantity passes the scalar test with the scale set by the larger of the two cells' conserved content
  per event dt. Unmatched faces fail. Aggregate sums are reported but never used as the test.
- **Conservation.** For each total Q (mass, energy, radiation energy, and each momentum component), take the
  budget residual R(t) = Q(t) - Q(t0) - (sources - sink removal - boundary outflow over [t0,t]) in each run. Pass if
  |R_test(t) - R_ref(t)| <= 1e-12 x Q_scale x (t-t0)/dt_ref + 1e-14 x Q_scale at every compared time, where dt_ref
  is the reference run's mean event dt. Q_scale is |Q(t0)| for mass and energies; for momentum it is the sum over
  cells of m|v| at t0, so zero net momentum is covered. If a scale vanishes (a motionless start, or no radiation
  energy at t0) it is replaced by the same quantity at the first compared time, or by the total energy for
  radiation, so no gate ever demands exact agreement.
- **Event sequence:** behaviour-preserving gates compare runs whose event sequences (event ticks and active sets)
  are identical. A divergent event sequence fails a behaviour-preserving gate.

### Radiation safety (every step, including A2 and B2)

- **Assertions, not counters.** On every accepted individual radiation candidate, all group energies and the gas
  energy are finite and non-negative. The passive-boundary defect stays within the configured limits of
  IndividualRadiationDefectConfiguration (RadiationDriver.cpp ~1762, version 3):
  - local withdrawal: relative 1e-2, absolute 1e-9;
  - event absolute target: 1e-6;
  - cumulative: signed 1e-4, absolute 1e-3;

  These are test validation gates, stricter than what the driver enforces. The driver checks accounting validity
  (RadiationDriver.cpp ~10073), and exceeding the local ratio or event target only requests passive-neighbour
  synchronization (~9320); it does not reject on the cumulative limits. The gate applies the combined local
  tolerance (relative term plus absolute term) and each cumulative limit separately, with the driver's
  normalization, and the report lists the maximum ratio to each.
- **Forced rejection test.** One rank's candidate is rejected by a test hook (the existing distributed fail-stop
  test hooks in the unit test show the pattern). It must be collectively rejected, the candidate state and
  accounting restored bitwise on every rank, and the retry must succeed. This test runs for A1, A4, A6, A7c and B2.

### Categories

- **Behaviour-preserving (A1, A3, A4, A5, A6, A7):** identical event sequence, and every comparison above passes
  against the unchanged binary. Also required:
  - focused MPI cases (ranks with no active cells, remote-only recipients and promotions, additions on a single
    rank, migration/AMR in the event, full-build fallbacks);
  - the MPI 4-rank unit suite (K=2, K=3, spacing rule) and the seven regression cases.
- **Decomposition-changing (A2):** cells compared by stable ID across the two decompositions, with tolerances
  widened for reduction-order differences only (rtol=1e-8, same atol scales). The event sequence must be identical
  or the difference explained by a documented reduction-order effect. Otherwise the same MPI, regression and
  radiation-safety gates as above.
- **Behaviour-changing (B2):** judged against the same-snapshot global control, plus radiation safety. Sink-adjusted
  conservation budgets, box and sink evolution, and density/energy distributions at matched times must be no worse
  than the existing individual-vs-global differences, and throughput must improve net.
- **Every step:** independent astra review of the complete diff and of the A/B results. A production restart
  remains subject to the user's build/run authorization.

## Steps

### Step 0: measurement, no behaviour change

a. **Per-rank, per-stage partial-build timers.** Stages: prepare/exchange, point tree, target Delaunay, each ghost
   round, Voronoi, volume exchange; ActiveMeshView construction and the closure loop (hdsim_3d.cpp ~2148-2500);
   event_gather_sync, SyncPartialBuildData and adjacency recording (~3106). Exclusive timers are reconciled per
   rank (stages sum to that rank's event time) before reduction. Report max/mean/argmax per stage and the critical
   rank's own split: one rank-0 line per build behind `RICH_INDIVIDUAL_PERF_TRACE`.

b. **Wake split:** local signal tree, distributed tree build (count the MPI_Allgather calls), routing/exchange.

c. **Floor-cell mesh-quality metric,** stratified by radius, at snapshot times: q = V/(Amax x width),
   generator-centroid offset / width, face-relative CFL.

**Gate:** per-rank stage sums agree with the event time within 10%; cost otherwise unchanged.

### Step B1: same-state cadence test (runs together with step 0)

From one snapshot, run 0.3 time units individual-only and 0.3 global-only. Compare floor-cell quality tails,
post-run minimum CFL, event counts and conservation.

**Gate:** in the individual run floor-cell quality worsens relative to global and the minimum CFL falls. If not,
drop track B.

### Step A1: closure re-expansion on the existing mesh

Largest measured mechanism: ~0.36-0.43 s gross per partial event, mean attempts 2.24 -> ~1.58.

- The continuation logic exists (`RICH_INDIVIDUAL_CLOSURE_REEXPAND`, hdsim_3d.cpp ~2510).
- First fix the verifier:
  - match faces by stable identity (then area);
  - reject non-finite values explicitly;
  - state the tolerances as executable inequalities, with w the cell width: |dV| <= 1e-12 x max(V, w^3);
    |dA| <= 1e-12 x max(A, w^2); ||dx|| <= 1e-12 x w + 64 x DBL_EPSILON x max(||x_a||, ||x_b||) for centroids and face centres (coordinate round-off; amended 2026-09-28 after the TDE verifier showed ~1.5e-12 w shifts at |x|~1e3); normal angle <= 1e-10 rad;
  - reject any mismatch collectively (full-build fallback);
  - no early return that bypasses the ordinary partial/full parity path (~2626);
  - count re-expansions per attempt, not cumulatively.
- Validate in two separate kinds of run:
  - verifier on, checking the geometry;
  - verifier off, the path that actually runs, against re-expansion off over the same events (states, face
    fluxes, conservation, cost).
- Focused MPI cases: remote-only promotions, additions on only one rank, empty-target ranks, full fallback.

### Step A2: balance A/B, configuration only

- Arms: `RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENTS` 4 -> 8 and 16; `_SEGMENT_WORK` > 0.
- Judge by complete-event wall time and all-active cost: more segments add ghosts, wake overlap and FMM traffic.
  The source notes that bin-balanced plans went stale within ~10 events (Simulation.cpp ~1385).
- Estimate 0.05-0.3 s per partial event; the two timing models disagree.
- If a configuration wins, adopt it as the default. If none wins but the per-rank timers show concentration,
  design weights from measured closure-plus-ghost work as a separate reviewed step.

### Step B2: earlier normal activation of mesh-sensitive floor cells (only if B1 confirms)

- Diagnostic: cap floor-density cells' intervals to the finest interval, so they close their interval and receive
  regularised velocities with the motion/flux pairing kept.
- If effective, narrow it to cells whose q lies below a runtime threshold relative to the global distribution.
- Keep only with net fewer finest events (a higher minimum CFL raises the anchor), lower wall time per unit time,
  and B2 acceptance. Runs as its own experiment; does not block the A steps.

### Step A3: gate the expensive Hilbert recount

- In segmented mode, all-point copies and `countIndividualHilbertSegments` getOwner calls run every event
  (Simulation.cpp ~4869, ~1722).
- Keep the cheap per-event accounting. Recount only on ownership, bin or motion invalidation, or on a bounded probe
  interval; collective decision; reversal preserved.
- ~0.08-0.11 s.

### Step A4a: preparation reuse across closure attempts

- **Reused within one event:** coordinates, weights, ownership, and the all-point search structures (all-point
  octree, range-finder input).
- **Rebuilt every attempt:** payloads (search radii and centroids, updated during ghost processing,
  Voronoi3D.hpp ~1847, ~3063), and target-dependent participation and mappings.
- All owned generators stay in the range searches.
- Suppressed exchange gets a direct local path instead of pushing all points through pointsExchange
  (HilbertPointsManager.hpp ~116), agreed collectively. MadVoro diagnostic reductions in the same code are folded
  in here.
- Gate: geometry and closure identical to the unchanged binary on the same events. ~0.05-0.15 s.

### Step A4b: sparse RICH mesh bookkeeping

Canonical-size masks, the ActiveMeshView mapping and owner/target exchanges proportional to the target where the
step-0 timers show cost. Same gate. ~0.05-0.1 s.

### Step A5: selective support seeding for the three-build events

- First confirm which seed shell mode rich_ag2 used, and split misses into absent edge, invalid record/owner, and
  insufficient expansion.
- Then add remote cached supports to the in-closure batch, or seed selectively.
- Averaged over all partial events: 0.26 x (0.55-0.65) = 0.14-0.17 s for one avoided rebuild, up to ~0.3 s for
  both; smaller after A1-A2.

### Step A6a: batched distributed-tree construction, reuse policy unchanged

- Replace the per-node MPI_Allgather recursion (DistributedOctTree.hpp ~132) by a level-batched occupancy exchange:
  common ordered frontier on every rank, empty ranks participate, bounded payloads.
- Gate: routing results and wake deadlines identical (bitwise) to the existing construction on the same events;
  MPI cases with empty ranks.

### Step A6b: rebuild vs widened reuse by measured cost

Separately from A6a, with drifting sources, ownership changes and conservative escape coverage checked. A6a+A6b
0.10-0.18 s.

### Step A7a: hydro touched-cell bookkeeping

Extensive update (default_extensive_updater.cpp ~412, ~427, ~791-925) and snapshot/change scans (hdsim_3d.cpp
~3197, ~3222) proportional to touched cells, including donors, remote passive recipients and activation-time
accounting. ~0.07-0.1 s.

### Step A7b: FMM adapter mapping

FastMultipoleAcceleration3D.cpp ~881: pass canonical indices/IDs directly; every gravitational source keeps its
current position and mass. ~0.02-0.05 s.

### Step A7c: radiation copies and scans

RadiationStep.cpp ~700, ~960:

- keep before-states for every cell a candidate can mutate, and for every unchanged passive cell that needs
  synchronization (accepted-face metadata, global radiation scale);
- keep retry rollback, candidate ghost refresh and defect accounting.

~0.02-0.05 s.

### Deferred or rejected

- **Persistent mesh updates with geometric certificates:** only if a single accepted partial build still costs more
  than ~0.4 s on the critical rank after A1-A4.
- **K=3:** re-test once the partial event is below ~1.5 s.
- **Complete-event full/partial selector** (extending the adaptive threshold model, hdsim_3d.cpp ~575): deferred
  until A1-A4 change the partial cost.
- **Rejected:** frozen passive gravity moments and local subcycling of finest cells (all generators drift; the event
  dependencies remain).

## Expected outcome (scenarios, not predictions)

- **Partial event.** Treating all step estimates as additive (they overlap, and B2 adds active work), A1-A7 save
  0.94-1.77 s, so 2.7 s would become ~0.93-1.76 s before overlap. Reaching ~1.2 s needs the favourable end of A1, A2 and A4 and is to
  be confirmed step by step.
- **Throughput.** With a 1.2 s partial event and all-active events unchanged: 178 x 7.1 + 527 x 1.2 = 1896 s per
  unit time against global 2517 s, 1.33x.
- **With B2.** If B2 then removes 20% of partial events only: 1.42x; 20% of both event classes: 1.66x. Both assume
  unchanged per-event costs despite B2's larger active population.

## Order and gates

**0 + B1 (together, both measurements) -> A1 -> A2 -> A3 -> A4a -> A4b -> A5 -> A6a -> A6b -> A7a -> A7b -> A7c**

- B2 is a separate gated experiment that runs when B1 confirms; it does not block the A steps.
- Each step: astra review of the complete diff and the results (high+fast), the acceptance rules above, and kept only
  if complete-event wall time per unit time drops beyond noise.
- The production t=50 run restarts once the cumulative gain over a same-snapshot global control is >= 1.3x in the
  late window and the acceptance rules pass, subject to the user's authorization.

## Results log

### Step 0 (job 10232728, rich_s1, from the global snap_full_65, 336 partial builds)

Per partial event: 0.97 s of mesh time over 2.38 attempts, 13.6 ghost rounds.

| Stage | Max over ranks | Mean over ranks |
|---|---|---|
| Ghost-point exchange | 0.68 s | 0.67 s |
| Prepare (suppressed exchange of all points) | 0.14 s | 0.14 s |
| Voronoi + volume exchange | 0.09 s | 0.09 s |
| All-point octree | 0.06 s | 0.06 s |
| Delaunay | 0.09 s | 0.004 s |
| Mapping, closure, other | ~0.02 s | ~0.02 s |

- Ghost time is the same on every rank. Round 1 dominates (0.08-0.24 s: the initial range queries for all target
  points, shipping 12k-56k ghost points); later rounds cost 0.002-0.05 s.
- So ghost cost scales with the target size (set by the worst rank and by the attempt count), and ~0.3 s per event
  is O(all points) work on every rank.
- Consequences: A1 (fewer attempts) and A4a (prepare, octree and volume exchange) are confirmed; A2/A5 act through
  the round-1 target size.

### B1 (jobs 10232722/10232723, 0.3 time units from one snapshot; follow-ups 10232729/10232730)

- **Floor-cell quality at r=2-5:** individual is equal or better (q p5 1.68 vs 1.50, effective radius p1 0.0089 vs
  0.0059, centroid offset p50 0.030 vs 0.045).
- **At r=5-20:** the q and effective-radius p1 tails are worse (eff p1 0.071 vs 0.107).
- **CFL:** raw limit median 9% lower in the individual state (1.674e-3 vs 1.841e-3), minimum 4% lower.
- **Verdict (astra): B1 gate FAIL, not demonstrated. B2 dropped.** Optional later diagnosis: follow the CFL-limiting
  cells by raw_id at matched times (geometry, sound/face-relative speed, time since activation).

### Same-window arms, t=34.16-34.352 from snap_full_65 (individual-event tau, sim time per wall second)

| Arm | Job | Binary | tau | Partial event | Builds per partial |
|---|---|---|---|---|---|
| Controls (five: B1I, S0M, A1C, A1C2, A3) | 10232722/28/33/49/44 | ag2, s1, s2, s2, s3 | 4.46-4.66e-4 (mean 4.57e-4) | 1.86-2.02 s | 2.1-2.4 |
| A1 re-expansion | 10232732 | s2 | 4.89e-4 | 1.67 s | - |
| A1 with verifier | 10232731 | s2 | 4.31e-4 | 2.19 s | 2.25 |
| A2: 8 segments | 10232724 | ag2 | 4.30e-4 | 2.07 s (wake 0.36 s) | 2.30 |
| Combined: s4 (A3 + A4a part 1) + A1 | 10232755 | s4 | 5.08e-4 | 1.53 s | 1.31 |

- **A3 (skip the segmented recount): APPROVED (astra).** Balance phase 0.154 -> 0.059 s; the net change is within
  the control spread.
- **A2, 8 segments: rejected** (wake routing grows faster than the balance improves). 16 segments held, not run.
- **A1: not accepted (astra), not even as opt-in.** The verifier found no topology mismatch in 166 checks. One event
  (tick 453119049728) exceeded round-off: volume ratio 372, area ratio 3.2e3 against the 1e-12 tolerances. It needs
  a focused MPI test, predeclared gates on a deterministic reference, flux/conservation/radiation gates, and
  diagnosis of the outlier; the verifier now reports the worst face (area / w^2, smallest face in the cell).
- **A4a part 1 (identity result for the suppressed exchange, diagnostic reductions gated): APPROVED after fixes**
  (rich_s5). The fixes make the identity setting agreed per manager on its own communicator, and stop a local
  MPI_COMM_SELF build from making the first world collective.

### Determinism (jobs 10232763-65, rich_s4, to t=34.5)

- **Default settings are not reproducible:** two identical control runs first differed at event 216 (density median
  6e-9, p99 0.2% at t=34.5). Wall-time-driven decisions cause it:
  - the segmented-ownership revert (2-6 per window);
  - the adaptive controller's wall-time dwell and probe budget;
  - FMM gravity resplitting on measured imbalance debt.
- **With RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE=0 and RICH_INDIVIDUAL_AUTO_REBALANCE=0,** FRZ1 and FRZ2 were bitwise
  identical: all 311 events, every field of all 9322198 cells.
- **But CACHE=0 turns active-Hilbert balancing off.** Builds per partial rose to 2.84 and re-expansion never fired, so
  the FRZA arm was a third control, not an A/B.
- **Replay settings** (rich_s5, jobs 10232784-86): RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENT_REVERT=0 (new),
  RICH_INDIVIDUAL_AUTO_REBALANCE=0, RICH_ADAPTIVE_DWELL_WALL_MAX=0, RICH_ADAPTIVE_PROBE_FRACTION=1000 and
  RICH_FMM_GRAVITY_RESPLIT=0, with active-Hilbert balancing on. FRZB1 and FRZB2 are controls; FRZBA adds
  re-expansion.

### Combined arm vs same-binary global (jobs 10232755 / 10232756, rich_s4, t=34.16-34.8)

- **Individual advantage decays over the window:** 1.37x to t=34.352, 1.30x to t=34.5, 1.26x to t=34.8 (1370 s vs
  1736 s wall). Global tau stays flat at 3.70e-4. Individual event rate is constant at ~700 events per time unit;
  the per-event cost grows.
- **What grows:**
  - Gravity per partial event 0.16 -> 0.32 s, per all-active event 0.47 -> 1.01 s.
  - Mesh attempts per partial event 1.26 -> 1.71 (re-expansion covers fewer promotions later).
- **Gravity cause: individual mode never re-splits FMM gravity ownership.** Every individual solve passes a target
  mask, all-active events included, and masked solves neither set the straggler baseline nor add debt. Debt froze
  at 2.64 s under the 2.67 s threshold: one resample in the whole run, against four in the global control.
  - Gravity owner: 36k -> 103k particles (mean 36k).
  - Full-solve local traversal max: 0.12 -> 0.58 s.
  - Global control: stays at owned_max 40-55k.
  - This is an unbounded late-time cost and the likely cause of the P50 decay to 0.91x.
- **Fix (rich_s7):** a mask that selects every particle on every rank counts like an unmasked solve. It uses the
  existing ski-rental rule, which is wall-time based, so replays keep RICH_FMM_GRAVITY_RESPLIT=0. A count-based
  trigger is a later option.
- **Wake routing:** the tree is reused on only 107 of 340 events. Each rebuild costs ~0.135 s: one MPI_Allgather
  per internal node, depth-first.
  - **A6a (rich_s6):** a level-synchronous build, one Allgather per level. Astra checked by static proof that the
    tree is identical, and the 4-rank suite passes with RICH_DISTRIBUTED_OCTTREE_BATCHED=2 (both builds compared on
    every construction).
  - **Unexplained ~0.11 s** of the wake-tree time is now timed separately (individual-wake-tree-reuse-check,
    -reuse-reduce).

### Replay reference with active-Hilbert on (jobs 10232784/85, rich_s5, replay settings)

- **FRZB1 == FRZB2 bitwise.** The 311 event sequences (mode, t_end, active, total) are identical, and every field and
  position of all 9322154 cells matches at t=34.5; totals agree to 16 digits.
- **Control throughput:** tau 4.70e-4 to t=34.352, 2.11 builds per partial event.
- **Gate 1 (reference reproducibility) passes on terminal state and event counts.** Per-event active-ID and state
  hashes (RICH_INDIVIDUAL_STATE_HASH) need rich_s7 or later.

### A1 under replay (job 10232786, FRZBA vs FRZB1)

- **Re-expansion fired in 149 events.** Throughput is +8% to t=34.352 (tau 5.07e-4 vs 4.70e-4) and +11% to t=34.5;
  partial events 1.59 vs 1.88 s; 1.31 vs 2.11 builds per partial event.
- **The trajectory diverges.** Active counts first differ at event 159 (20232 vs 20233), and AMR cell sets differ
  by ~730 cells at t=34.5. The fixed-limit trajectory gate (gate 2) fails, as expected from chaotic growth.
- **Compared on cells whose ID existed at the restart:** AMR assigns new IDs in an order-dependent way, so newer IDs
  do not denote the same cell in both runs.

  | Pair | density p50 | p99 | p99.9 | generator shift/w p99.9 |
  |---|---|---|---|---|
  | A1 vs control, replay | 1.8e-11 | 1.1e-4 | 2.3e-2 | 0.17 |
  | Two identical unfrozen controls (first difference at event 216) | 5.8e-9 | 3.6e-4 | 7.7e-2 | 0.36 |
  | A1 vs control, unfrozen | 1.4e-8 | 5.1e-4 | 0.11 | 0.51 |

- **The A1 perturbation is smaller than production's own run-to-run divergence.** An envelope rule to replace gate 2,
  with a fresh confirmation window, was proposed to astra (astra_gate2.md).

### A4a part 1 runtime identity (job 10232787, FRZBI0: FRZB settings + RICH_MADVORO_IDENTITY_EXCHANGE=0)

- **Bitwise identical to FRZB1** (identity exchange on, the default): all 311 events and every field and position
  of all 9322154 cells. A4a part 1 changes no result.
- **Throughput:** tau 4.64e-4 off vs 4.70e-4 on (~1%, noise level).

### A1 gate 7: MPI test seam (unit suite, 4 ranks, 2026-09-29)

- **Seam:** IndividualClosureTestProbe / IndividualClosureTestAccess (hdsim_3d.hpp). With a probe attached,
  timeAdvanceIndividual builds only the event mesh from the probe's seeds through the production closure loop, and
  records each expansion, the remote promotions, the promotions at each re-expansion continuation, the additions at
  each rebuild, and the final depths and target set. It also overrides CLOSURE_REEXPAND and REEXPAND_VERIFY, so one
  process runs both modes. Every hook is a null-pointer test and no collective is added.
- **testClosureReexpandPromotionMPI:** only A is active; B is A's remote neighbour, and every other cell is prebuilt
  at depth two.
  - B's owner sees a remote promotion.
  - Off: 2 attempts, zero additions at the rebuild, B expanded at depth one on attempt 2.
  - On: 1 attempt, 1 re-expansion, B among the continuation's promotions, no rebuild, B expanded at depth one on
    the first mesh.
  - Identical final depths and target sets.
  - The verifier run closes "after re-expansion check".
  - The retained mesh matches a full build within the verifier tolerances (worst centroid ratio 1.3e-4).
- **Result: PASSED**, with the whole suite (unit_run_s9t). Still required by gate 7: the other focused MPI cases and
  the seven regressions.

### rich_s7 throughput A/B, same binary (jobs 10232792 individual / 10232793 global, t=34.16-34.8)

- **Individual 1344.5 s vs global 1769.0 s: 1.32x.**
  - Per window: 1.31x (tau 4.76e-4 vs 3.64e-4), 1.29x (4.71 vs 3.64), 1.33x (4.79 vs 3.59).
  - No decay; the s4 run (with A1) decayed 5.08 -> 4.45.
- **FMM:** three resamples. Gravity owner <= 62k (mean 36k). All-active source 0.43-0.48 s, flat (was 0.47 -> 1.01).
  Wake 0.22 -> 0.15 s.
- **Global path:** s7 global vs s4 global 1769 vs 1736 s (+1.9%, 345 vs 341 steps); no global change is expected,
  so within noise.
- **Logs:** the same guard, fallback and closure counts as rich_s4.
- **Production (user's standing authorization, 2026-09-28):** TDE_P50S7, job 10232900, rich_s7, A1 off (astra: the
  production trajectory stays REEXPAND=0 until the opt-in gates pass). Restart from the P50 snap_full_68
  (t=38.436), adaptive controller, to t=50.

### Segmented-ownership aging and ledger repair (production 10232900; A/B 10232905/10232906, 2026-09-29)

- **Production slowdown:** from t=39.2 to 41.4 individual ran at 0.85x global. Cell counts stayed balanced
  (max/mean <= 1.09), but the cost of every build grew with the age of the segmented plan, in the ghost-exchange
  stage:
  - partial events 1.95 -> 5.53 s;
  - full builds 2.1 -> 5.1 s.
- **Why the ledger never reverted:** the dominant class (0.1-1% active) had no positional reference, which vetoed
  every revert (net_benefit -57 s, 8 losing windows). The debt re-plan is gated on SEGMENT_BINS.
- **Fix (rich_s10, astra APPROVE):**
  - all classes are in the ledger;
  - adoption requires references for the dominant positional classes;
  - a bounded revert-measure / revert-unjudgeable ends a probe that cannot be judged;
  - replay (SEGMENT_REVERT=0) is unchanged.
- **A/B from snap_full_69 (t=40.715), both with PROBE_FRACTION=10 and the cadence trace:** rich_s7 1908 s vs
  rich_s10 1443 s to t=41.219 (**1.32x**). Individual tau 2.18e-4 vs 2.75e-4. rich_s10 reverted 4 times and no plan
  aged past 100 events.
- **Separate late collapse (t>42.55):** a single cell in bin 26-27 set the cadence, ~0.5x global. Cadence trace
  attribution of the finest-bin cells in the A/B: CFL, scheduler and thermal guard, with radiation minor. Under
  investigation.
- **Production resumed:** TDE_P50S10 (job 10232907), rich_s10, from snap_full_70 (t=41.856) to t=50, with the
  cadence trace on.
