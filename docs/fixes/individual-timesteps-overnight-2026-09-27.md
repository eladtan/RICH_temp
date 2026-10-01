# Overnight work log, 2026-09-26/27: load balance and gravity for individual time steps

## Summary (read this first)

**Individual-mode wall time on the TDE's 83-event window went from 183.9 s to 127.4 s (−31 %).** Global stepping
over the same span takes about 124 s, so individual mode is now at **parity**.

**Final binary: `rich_seg24_20260927`.**

- Every change is astra-approved: FMM pruning in round 2; segmented ownership, ledger, wake reuse and the
  controller fix in round 9.
- Regression run: 7/7 pass (`amr_random`, `amr_random_individual`, `fmm_gravity_mpi`, `fmm_gravity_serial`,
  `individual_box_growth`, the new `segmented_hilbert_ownership`, `suppressed_exchange_ghosts`).
- Final confirmation, defaults only: job 10213543, 127.4 s.
- **The production symlink was not moved.**

**The plan's final gate (≥ 1.15x global) is not met.**

- Adaptive runs to t = 21 still lose (about 366-380 s against 307 s).
- The blocker is not balance or gravity; it is the **scheduler's cadence cascade** (section below). Mid-interval bin
  lowering by the accuracy guards sends overruns to the finest-bin tick, and events multiply.
- The fix is an accuracy-semantics choice, left for you.
- The second lever is individual AMR cadence (19 s of 126 s), also your open decision.

**New defaults** (individual mode only; global unchanged):

- `RICH_FMM_TARGET_PRUNE=1`
- `RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENTS=4`
- `RICH_INDIVIDUAL_ADJACENCY_SEED_SHELLS=auto`
- `RICH_INDIVIDUAL_WAKE_ROUTING_REUSE=1`

The adaptive controller's first global step after a switch is now at least the finest occupied bin.

**Nothing is committed.** Full patch: `docs/fixes/individual-timesteps-overnight-2026-09-27.patch`, plus the check
function in `regression_tests/lib/regression_checks.sh` and the new case directory.

**Run directories created tonight** (seeded copies of snap_full_54 plus terminal snapshots, 167 GB in total;
cleanup is your call): TDE_ADP11_B TDE_ADP14_B TDE_ADP17_B TDE_ADP20_B TDE_ADP20P_B TDE_ADP4_B TDE_ADP8_B TDE_ADP9_B TDE_ADPG_G TDE_BN16_B TDE_BN4_B TDE_BN8_B TDE_BN8V_B TDE_C0_B TDE_C95_B TDE_C9_B TDE_D23_B TDE_D24_B TDE_E4_B TDE_F4_B TDE_G10_G TDE_H4_B TDE_I4_B TDE_IL3_B TDE_IL4_B TDE_IL4S3_B TDE_IL6_B TDE_IL6S3_B TDE_K4_B TDE_K4W1_B TDE_PR0_B TDE_PR1_B TDE_PR1b_B TDE_PRO_B TDE_R4_B TDE_R4o_B TDE_R4W1_B TDE_S11_B TDE_SG1_B TDE_SG2_B TDE_SG4_B TDE_SG4V_B TDE_SG8_B TDE_W1_B TDE_W3_B TDE_WR0_B TDE_WR1_B 

Plan: [individual-timesteps-balance-gravity-plan-2026-09-26.md](individual-timesteps-balance-gravity-plan-2026-09-26.md)
(astra-approved). Status before tonight: [individual-timesteps-tde-status-2026-09-25.md](individual-timesteps-tde-status-2026-09-25.md).

All runs: TDE, 256 ranks (16 bigrun nodes), restart `snap_full_54` (t = 20.8214549, 4.66 M cells), terminal
snapshot at t = 20.9015, `RICH_FMM_GRAVITY_RESPLIT=0` (deterministic comparisons), `RICH_MPI_WAIT_PROFILE=1`,
`RICH_FMM_TRACE=1`, `RICH_INDIVIDUAL_PERF_TRACE=1`, script `runs/M05R05MBH1e4MGComptonIndividual/submit_gfloor.sh`
ARM=B (10 global warm-up steps, then one individual probe that stays for the window: 83 events). Every node set
prescanned clean.

## Stage 0 + B2: FMM target pruning (binary `rich_prune1_20260926`)

**What.** `DistributedFmmGravityCalculator::solve` takes an optional per-particle target mask.
`FastMultipoleAcceleration3D::EvaluateIndividualTargets` (the individual-event caller) passes the active cells.
With `RICH_FMM_TARGET_PRUNE=1` (rank 0 reads, broadcast):

- the mask travels with each particle through the gravity-owner redistribution (`GravityOwnerParticle::reserved`);
- `FmmPasses::markTargetNodes` flags tree nodes with a target in their subtree;
- the local plan (`FmmDualTreeTraversal::runLocalPlan`), the LET execution (`FmmLetPlan::beginExecute`) and the
  downward pass (`FmmPasses::downward`) skip every interaction and leaf evaluation whose target subtree has no
  target.

The LET already sent a per-solve list of the retained subscription slots it needs before the payload exchange. So
the target test there also shrinks the **traffic**: owners pack only the requested slots. The remote plan needs no
rebuild and cannot go stale; the retained plan is a superset, filtered each solve.

Global stepping never passes a mask. Direct-sum error-sample solves are never pruned. Mask agreement is folded into
the solve's validation Allreduce.

Also in this stage:

- `INDIVIDUAL_HYDRO_PHASE_TIMING` carries per-phase busy time (wall minus MPI, from `mpi_wait_profiler`):
  `<phase>_busy_mean`, `_busy_max` and `_mpi_min`.
- A new `point_motion` phase is split out of `source`.
- `fmm_solve_trace` reports the pruning counters and the caller's prepare/finish time.

**Result, A/B (jobs 10213427 off / 10213428 on, identical nodes class, same binary):**

| | Prune off | Prune on |
|---|---|---|
| Terminal snapshot | reference | **byte-identical** (8195/8195 HDF5 datasets) |
| Individual wall, 83 events | 183.9 s | **169.8 s (−7.7 %)** |
| Median small-event solve (max over ranks) | 0.322 s | **0.099 s** |
| — local traversal | 0.194 | 0.021 |
| — LET execute | 0.222 | 0.022 |
| — redistribution (unchanged, all sources) | 0.069 | 0.057 |
| LET bytes per solve | 248 MB | 4.4 MB |
| Hydro `source` phase, small events (wall max) | 0.344 s | **0.141 s** |

**Findings:**

- The "~0.15 s source assembly" in the status doc is not assembly. `EvaluateIndividualTargets` spends 5 ms
  before the solve and 1.3 ms after it. The rest of "source" was FMM straggler wait: busy max 0.25 s against busy
  mean 0.09 s.
- So **B1 (source-assembly dirty bits) has nothing to save** and is dropped, per its own gate.
- `point_motion` (the point-velocity update, `pm_` + `ApplyFix`) costs 0.10 s per small event. It was
  previously counted inside "source".

**B3 probe** (`RICH_FMM_TARGET_OWNED=1`, masked solves on mesh ownership, job 10213429): 312 s, much worse. As the
plan predicted, it needs compact per-rank domains (A2) first. It stays an experiment knob, default off.

**astra xhigh r1: CHANGES_REQUIRED.** Two majors, both fixed in `rich_seg1_20260927`:

- An ownership-mode change between solves now forces a fresh tree and a rebuild of every plan.
- Pruned solves no longer feed the resplit straggler baseline or debt.

The NIT (bit-identity not guaranteed under operator-cache pressure) is noted; parity is by tolerance, and the
measured run was bit-identical. r2 is pending.

## A1: multi-segment Hilbert ownership (binary `rich_seg1_20260927`, in test)

- **`CurveLoadBalancer`.** New `segmentOwner` field (empty = positional = the old behaviour) and
  `getOwnerOfIndex`. `CurveEnvAgent::getCellOwner` also routes through it; the plan did not list that site.
- **`HilbertLoadBalancer`.** New `setSegments`, `sameAssignment`, `getSegmentOwner`, and
  `rebalanceInterleaved(points, weights, S)`: S·P equal-weight pieces, piece i to rank i mod P.
  - `rebalance()` returns to positional.
  - `rescale`/`changeBox` sort (boundary, owner) pairs together when segmented; this is the plan's two-mode
    sort, with positional = the old bare sort.
  - `clone` copies the owners.
- **`weightedBalance3`.** New `pieces` parameter.
- **`HilbertRectangularTree3D`.** Segment-aware leaves: a leaf spans fewer than 4 segments and lists its distinct
  owners.
- **`HilbertTreeEnvAgent`.** Passes the owners through.
- **Restart IO.** Optional `segment_owner` dataset; absent means positional.
- **`Simulation`.**
  - The active-Hilbert cache and the passive-run bound apply only to positional partitions.
  - Partition change is tested with `sameAssignment`.
  - `RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENTS=S` (default 1, agreed) replaces the rejected single-range active cut
    with an interleaved segmented partition. That partition is kept while it covers every rank within the
    owned-cell skew.
- **A4 part (not yet built).** `adaptiveEnterGlobal` collapses a segmented partition to a weighted positional one
  before resetting the scheduler, and logs `RICH_SEGMENT_COLLAPSE`.

**Imposed-partition experiment and sweeps** (all with pruning; wall = individual part, 83 events; global-only
reference for the same span is about 124 s, 51 of the 61 steps of job 10208630, 148.7 s in total):

| Arm (job) | Partition | Wall | Notes |
|---|---|---|---|
| 10213428 / 10213430 | positional (S=1) | 169.8 / 167.7 | S=1 terminal snapshot byte-identical to the pre-A1 binary |
| 10213431 | interleaved S=2 | 147.7 | |
| 10213442 | interleaved S=3 | 148.5 | |
| 10213432 / 10213443 | interleaved S=4 | 132.3 / 132.5 | reproduces within 0.2 s |
| 10213444 | interleaved S=6 | 134.8 | |
| 10213433 | interleaved S=8 | 138.9 | over-fragmented |
| **10213445** | **interleaved S=4 + `RICH_INDIVIDUAL_ADJACENCY_SEED_SHELLS=3`** | **126.8** | best |
| 10213446 | interleaved S=6 + 3 seed shells | 130.4 | |
| 10213436 / 10213437 | bin-balanced (A2) S=4 / S=8 | 138.1 / 136.0 | skew decays in about 10 events; re-plans cost ~1.4 s each |

**What the partitions change** (small events at S=4):

- All small events become **partial** builds. Before, 26 of 38 were full.
- The busiest rank's partial fraction drops from 0.52 to 0.21.
- Flux, extensive and point-motion busy time on the mean rank fall about 30-fold, because the partial mesh holds
  only the region near the active cells.
- The event mesh stays at about 0.46 s wall: the hot rank is still 3x the mean (busy max 0.35 s against a mean of
  0.12 s), and a partial build takes 3 attempts.

**Parity gates:**

- 74 checks at S=4 and 35 at S=8, with **0 mismatches**.
- `regression_tests/cases/segmented_hilbert_ownership` (new, 64 ranks) **passes**: interleaved and scrambled
  owners, box change (every (boundary, owner) pair follows `changeBox`'s own remap) and HDF5 round-trips, all
  exact.

**A2/A3/A4 as built:**

- **Default plan.** The default is the interleaved plan: S·P equal-count pieces, piece i to rank i mod P. The
  bin-balanced plan is an option, `RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENT_BINS=1`.
- **Adoption as a probe.** Adoption is a measured probe, like the mode controller's:
  - it needs at least `cooldown_events` positional samples first;
  - event-mesh time is compared per active-fraction class against positional events; events above 50 % active
    are excluded, since they are full builds by the threshold; a class with fewer than 2 samples falls back to the
    mean over classes;
  - planning and migration costs are charged once;
  - the partition reverts after two losing windows, or when the net benefit is still negative after 3 windows;
  - each revert doubles the cooldown;
  - internal re-plans keep the probe's age and windows.
- **Collapse and revert.** Both use `rebalanceToPositionalOwnership`, an exact equal-count positional cut.
  `tess.Rebalance(weights)` cannot be used: PointsManager applies its own 1.15 imbalance gate and would leave a
  balanced segmented partition in place.
- **Adaptive run (job 10213447):** 8 mode switches, 4 collapses of about 0.58 s each, no errors.

**Reviews.** astra A1 r1 CHANGES_REQUIRED (fixed); A1-A4 r2 CHANGES_REQUIRED (fixed); r3 CHANGES_REQUIRED (fixed:
re-plans kept from postponing a revert, positional planning time charged, wake counters logged); r4 pending.

## Other per-event costs found

**Wake routing tree.** `individual-wake-tree` costs 0.11 s per event: every event rebuilds a `DistributedOctTree`
over all cells, one `MPI_Allgather` per node.

- A reuse test was added, the same argument as `DistributedOctEnvironmentAgent::routingStillCovers`; astra: safe.
  Terminal snapshots are byte-identical with reuse on and off (jobs 10213450/51).
- It never hit (0 reuses in 20): the root box was the tight bounding box of all sources, so the extreme sources
  left it every event.
- `rich_seg11` pads the tree root by 2 % of the extent. The index changes; the query radii and diagonal do not.
- The static tree's destructor segfaulted after `MPI_Finalize` (rc 139 after the snapshot was written). The tree is
  now a never-freed heap pointer.

**Individual AMR pass.** A pass costs 1.3-1.9 s, even with nothing to refine (cycle 5129: 0 added, 25 removed,
1.64 s). A global AMR step costs the same as an ordinary one (2.2 s).

- `AMR3D::ApplyIndividual` builds a full mesh whenever the event mesh is partial, then evaluates the criteria.
- The pass runs every 10 events, which is 1.6x more often per simulated time than global's every 10 steps.
- It totals 19 s of the 127 s.
- Cadence in simulated time, or running the pass on full-build events, is **your open decision**; not changed.

**Cadence quantization.** Bin 30 is the global dt at the switch, which is growth-ramped after a restart or switch
(0.00093 against a CFL limit of 0.0015). So every small event advances only 0.00093, which is the 1.6 events per
global step.

- `RICH_INDIVIDUAL_BIN_ANCHOR_MARGIN=m` (default 0 = unchanged) anchors the grid at m × `SuggestTimeStep()` instead.
- It is **harmful**. With m = 0.9 / 0.95, the finest cells step close to their limits, and wake deadlines then keep
  shortening intervals to unaligned ticks: about 5 events per bin-30 interval against 1, 271+ events for the same
  span (jobs 10213454/55, cancelled).
- The slack under the ramped anchor was absorbing wake signals. A fix would have to make wake shortening
  bin-aligned or batched; that is a scheduler design question, not done tonight.

## The cadence cascade: why late individual periods lose (found tonight; not changed)

**Adaptive runs to t = 21** (same window, same nodes class):

| Run | Wall |
|---|---|
| Global only (job 10213458) | 307.3 s |
| Adaptive, `rich_seg11` (job 10213459) | 342.0 s |
| Adaptive, `rich_seg14` (job 10213465) | 381.6 s |

**Per individual period** of job 10213459 (simulated time per event, wall per event):

| Period | Simulated time per event | Wall per event | Rate |
|---|---|---|---|
| First (t 20.825-20.861) | 0.00093, perfectly regular (one event per bin-30 interval) | 1.62 s | 0.000575, **parity** with global's 0.000592 |
| Later | 0.0005 | 1.3 s | 0.0004 (0.65x global) |
| Global steps | 0.0014-0.0015 | 2.4 s | reference |

An individual event now costs **0.55-0.68 of a global step**. Individual therefore wins whenever an event
advances more than about 0.65 of a global step; it loses on cadence alone.

**The later periods cascade.** Event tick spacing halves again and again: 1024 → 512 → 256 → … → 2 → 1 → 0, in
units of 2^20 ticks, with a handful of active cells per event.

- Mechanism: `IndividualTimeStepScheduler::binnedEndTick`.
  - Accuracy guards (mostly the thermal-loss guard, `INDIVIDUAL_CADENCE` category `thermal`) lower cells in bins
    31-32 to bins 29-30 in the middle of an interval.
  - `all_shortened_interval` grows from 260 to 4357; `INDIVIDUAL_BIN_OVERRUN` reports intervals 1.4-2.5x over
    their new allowance.
  - An overrun cell is sent to the next tick of the **finest bin in use**. Successive lowerings lower that bin, so
    the ticks get finer.
- This rule is a safety fix (its comment: the overrun it prevents killed a cell in job 10196266), so I did not
  change it.
- Options for your decision:
  1. end an overrun on the requested bin's grid rather than the finest;
  2. rate-limit bin drops per cell;
  3. apply accuracy-guard limits (thermal, mass) from the next interval rather than mid-interval, since they are
     accuracy guards, not stability limits.

**The first period avoids the cascade only thanks to slack.** Its anchor was the ramped global dt (0.00093 against
limits of 0.0015). Tightening the grid (`RICH_INDIVIDUAL_BIN_ANCHOR_MARGIN`, jobs 10213454/55) triggered the
cascade from the start (271+ events for the same span), so that experiment was **removed** from the code (saved as
scratchpad anchor_experiment_*.patch).

## Ledger evolution after review rounds 4-5 (A3)

- **Round 4 rules.** A class-matched reference with a pooled fallback, and a revert when the probe had not repaid
  its cost after 3 windows. This reverted winning partitions (jobs 10213460 and 10213463).
  - The pooled fallback mixed cheap below-0.1 % events, which are partial under either partition, into the
    reference for 0.1-10 % events, which are full builds under positional.
  - The ledger counted mesh seconds only, missing the flux, extensive and point-motion savings of partial meshes
    (flux 0.14 → 0.035 s).
- **Now (`rich_seg17`/`seg18`):**
  - The ledger measures whole events, less the AMR pass and the balance decision.
  - Only same-class references are credited (at least 2 positional samples).
  - A revert needs representative evidence: credited events ≥ cooldown, lifetime coverage ≥ 1/2, and each losing
    window itself at least half credited.
  - A failed repair of an unsafe segmented partition falls back to positional ownership.
- **Result: job 10213480 = 125.9 s**, the best of the night. It adopted at event 13 and never reverted.

## Plan status

| Stage | Status |
|---|---|
| 0 Measurement | done: sub-phase busy time, point_motion split, FMM prune counters, caller prepare/finish timing |
| A1 multi-segment plumbing | done; S=1 byte-identical; regression case added and passing; 0 parity mismatches |
| A2 bin-balanced plan | built; measured worse than interleaved (bins decay); option `..._SEGMENT_BINS=1`. A work-weighted interleaved variant (`..._SEGMENT_WORK`) measured neutral (142.2 vs 142.6 s) |
| A3 reversible decision | done (probe + ledger + reverts); reviewed over 6 rounds |
| A4 collapse on global entry | done; exercised (4 collapses in job 10213447, 2 in 10213459) |
| B1 source-assembly dirty bits | **dropped by its own gate**: the assembly is 6 ms, not 0.15 s |
| B2 target-pruned FMM | done; byte-identical; **default on**; astra APPROVE |
| B3 solveOwned | measured much worse (312 s); stays an experiment knob, default off |
| B4 (i-b) push vs pull | no-go: after pruning the pull exchange is 0.016 s per solve, about 1 % of an event |
| B4 (iii) incremental sources | not pursued: redistribution is 0.057 s per solve (about 4 %), and shipping only owner-changed sources was rejected in the plan (positions change every event) |
| B4 (iv) hierarchical gravity | not pursued: gated on the near-sink physics check and the half-kick cache redesign |
| Final gate (≥ 1.15x global) | **not met**: parity in the first individual period, 0.65x later, because of the cadence cascade above |

## Adaptive controller: first global step after a switch (fixed)

**The bias.** `adaptiveEnterGlobal` started the first global step at the individual schedule's next event step. After
a cascade that was 4.6e-5 to 8.5e-5, against bins of about 1e-3. With the 1.25x growth per global step, the whole
8-step global probe was spent ramping up. The controller therefore judged global slower and returned to individual:
5 wrong returns in job 10213481, which took 502 s against 307 s for global only.

**The fix.** The first global step is now at least the finest occupied bin's length. That is legal for every cell,
because at the switch the state is synchronized and every bin is within all its cell's limits, radiation included.
The global CFL still caps the step. astra r7 confirmed this argument.

- **Positional adaptive run after the fix** (job 10213497): 366 s. The global probes now start at 0.0015, and the
  controller adopts global.
- **What is left is cascades.** Each later individual probe cascades immediately (rate 5e-5 to 1.3e-4, against
  6e-4 for global) and costs time.

## Final defaults in the working tree (binary `rich_seg23_20260927`, candidate; production symlink NOT moved)

| Setting | Default | Notes |
|---|---|---|
| `RICH_FMM_TARGET_PRUNE` | **1** | exact |
| `RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENTS` | **4** | `1` restores one range per rank |
| `RICH_INDIVIDUAL_ADJACENCY_SEED_SHELLS` | **auto** | 3 when segmented, else 2 |
| `RICH_INDIVIDUAL_WAKE_ROUTING_REUSE` | **1** | exact; about 3 % |
| `RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENT_BINS` | 0 | options, default off |
| `RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENT_WORK` | 0 | |
| `RICH_FMM_TARGET_OWNED` | 0 | |

The global path is unchanged. Individual-mode throughput for the 83-event window: 169.8 s → **126 s** (S=4
interleaved, jobs 10213480 and 10213499), about parity with global over the same span (about 124 s). With the
cascades, adaptive runs to t = 21 still lose: 366-380 s against 307 s.

## What would reach the 1.15x gate

1. **Fix the cadence cascade** (the options above; your decision: accuracy-guard semantics).
   - The first individual period already runs at parity with the slack anchor.
   - An event costs 0.55-0.68 of a global step, so cascade-free events at one bin-30 step each would be 1.3-1.5x
     global.
2. **Individual AMR cadence** (your decision): each pass costs 1.3-1.9 s, 19 s of 126 s, while global's AMR is free.
3. **The hot rank's mesh.** Busy max is 0.36 s against a mean of 0.12 s at S=4: the partial build still takes 3
   attempts on the busiest rank.
