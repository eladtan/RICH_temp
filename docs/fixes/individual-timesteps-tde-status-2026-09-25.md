# Individual time stepping on the TDE — status, findings, open issues and handoff (2026-09-25)

Session: Claude `8b3f6501` ("Rich TDE individual handoff"). It continued from the handoff
`/tmp/rich-tde-individual-handoff-2026-09-25.md` (session `da9e65a8`) and absorbed the work of session `dbacb723`
("Investigate AMR over-refinement after individual stretches"), which handed over and stopped.

Everything below is **uncommitted** in the worktree `/home/elads/RICH-ablation-integration`, branch
`codex/individual-timesteps`, HEAD `e8c6ad9ec`. The tree is dirty by design (memory `preserve-dirty-worktrees`).

---

## 0. Goal (from the user)

The TDE (`runs/M05R05MBH1e4MGComptonIndividual`, driver `runs/BaseTDEComptonIndividual/test.cpp`) must run **correctly**.
With the adaptive controller it must be **at least as fast as global stepping**, using individual time steps
wherever they pay.

The user's standing design preferences (memory):

- **No hard-coded regime rules.** Decisions must be runtime criteria that switch both ways
  (`adaptive-not-hardcoded-policy`).
- **Logging.** Rank-0 aggregate records plus one example, with detail behind a flag.
- **Guards.** Never cap the RoundCells correction velocity.
- **Reviews.** Every code change gets an astra (`gpt-6-astra`, xhigh) review with a `VERDICT:` line.
- **Before every submission.** Pre-scan the nodes.
- **Production.** Never move the production symlink (`runs/M05R05MBH1e4MGComptonIndividual/rich ->
  rich_fmm_throttle_20260920`, still unchanged) without the user.

## 1. The bottom line (read this first)

**Where individual mode stands, measured on the full-gravity TDE after pericentre** (restart
`TDE_LONG_GLOBAL/snap_full_54`, t = 20.82, 4.66 M cells, 256 ranks):

| Configuration | Individual throughput vs global |
|---|---|
| Start of the day (thermal guard at the sink, per-event sink drain) | 0.12–0.27× in probes; sustained runs degrade to ~0.07× |
| + guard floor (`RICH_INDIVIDUAL_GUARD_FLOOR`, now default `apply`) | 0.62× in an 83-event window, 0.78× in a probe |
| + rate-based sink (sustained run, t 20.82 → 21.25) | 2.55× faster than without it; ~0.4–0.5× global, no degradation of cadence |
| Short window, current best (`rich_busytime2_20260925`) | median event 2.03 s vs global step 2.21 s, at 1.7 events per global step |

The time-step collapse is fixed. Two mechanisms were responsible, and both are fixed:

- the thermal-loss guard on near-vacuum cells at the sink edge;
- the sink draining cells once per event instead of per unit time.

The remaining obstacle is the **fixed per-event cost**. An event in which 0.1–2 % of the cells are active costs
about as much as a global step, because most of its work does not scale with the active count. The two dominant
parts of that fixed cost are the subject of §2 (load balance / mesh) and §3 (FMM gravity). Radiation, AMR and
synchronisation make up the rest (§4).

**Rough budget of a typical small event.** Profiled run 10208629, 0.1–2 % active, median of 7,490 active cells:

| Part | Seconds | Scales with the active count? |
|---|---|---|
| Mesh build (mostly full) | 0.64 | No (see §2) |
| Gravity source (FMM + source assembly) | 0.30 | No (see §3) |
| Radiation | 0.33 | Weakly (see §4) |
| Fluxes + cell updates | 0.21 | **Yes**, the only part that does |
| AMR (averaged over the 1-in-10 passes) | 0.15 | No |
| Sync, wake, commit, suggest, other | ~0.3 | No |
| **Total** | **2.03** | Global step: 2.21 |

---

## 2. Issue A — load balance, concentration of active cells, and the mesh

### 2.1 What is measured

- **Active cells are concentrated on a few ranks.** The fastest cells (smallest CFL steps) sit near the black
  hole and the central sink, which a handful of Hilbert ranges own.
  - `INDIVIDUAL_ACTIVE_HILBERT_DECISION` reports `current_active_max_mean` of 40–130: the busiest rank holds up to
    130× the mean active count.
  - `INDIVIDUAL_MESH_BUILD` reports a busiest-rank partial target of 30–58× the mean target.
    Example: cycle 5159, 36,634 active cells globally; rank 223 has a target of 12,630 cells against a mean of 413.
- **Concentration forces full mesh builds on every rank.** The per-rank closure threshold makes an event mesh full
  if *any* rank's partial target exceeds 0.5 of that rank's owned cells (`hdsim_3d.cpp`,
  `build_event_mesh_impl`, `exceeds_threshold`).
  - In the profiled run, 26 of the 38 events with 0.1–2 % active cells had a full build, costing 0.62–0.64 s.
    That is the same as a global step's mesh.
  - Events below 0.1 % stay partial, but still cost 0.55 s: the hot rank's large target takes 2–3 attempts.
- **The existing active-cell rebalancer cannot fix it.** `maybeBalanceIndividualEventByActiveBins` proposes a
  Hilbert cut that balances active cells only. It is single-constraint, one contiguous range per rank. The proposal
  gives `proposed_owned_max_mean` of about 200–214 against the 2.0 memory-safety cap, so it is rejected, in 69 of
  83 events in one trial and 75 of 83 in another.
  - This is not a tuning problem. Session `dbacb723` proved a lower bound: with one contiguous range per rank, all
    cells between two consecutive active keys stay on one rank. The largest such run of passive cells bounds the
    rebuild's owned skew from below, and that bound was 211–214.
- **Busy-time profiling.** `RICH_MPI_WAIT_PROFILE=1` measures time outside MPI calls per rank (§6.6). Medians from
  jobs 10208629 (individual) and 10208630 (global):

| Event type | Wall | Busy, mean rank | Busy, busiest | Busy, least busy | Busiest − mean | Idle share of the mean rank |
|---|---|---|---|---|---|---|
| Individual, <0.1 % active (31) | 1.19 | 0.37 | 0.94 | 0.32 | 0.57 (48 %) | 69 % |
| Individual, 0.1–2 % active (38) | 2.03 | 1.26 | 1.49 | 1.16 | 0.22 (11 %) | 38 % |
| Individual, all active (8) | 3.44 | 2.23 | 2.62 | 2.02 | 0.39 | 35 % |
| Global step (51) | 2.21 | 1.53 | 1.69 | 1.35 | 0.17 (8 %) | 31 % |

  - In a small event even the least-busy rank computes for 1.16 s, about 80 % of a global step's per-rank work.
    So the cost is mostly work on every rank, not a few overloaded ranks.
  - The "busiest − mean" gap is a heuristic. It misses a bottleneck that alternates between ranks within a phase
    (review of task 2), and it counts unmovable work as balanced.

### 2.2 Why balancing would still help (and how much)

The main benefit is not the 11 % busy-time imbalance. It is that spreading the active cells removes the trigger
for **full** mesh builds.

- **Mesh.** With each rank's target at about 1–3 % of its cells, builds stay partial. The measured cost of a
  partial build at a busiest-rank fraction of about 0.02 is 0.1–0.18 s (1–3 attempts × 0.06–0.08 s), against 0.64 s
  now. This saves about 0.45–0.5 s per small event.
- **Other phases.** Up to the 0.22 s busy imbalance.
- **Radiation.** Probably some saving: it grows from 0.08 s to 0.33 s to 0.49 s with the active count. Not
  quantified.
- **Unchanged:** FMM gravity (0.30 s), AMR (0.15 s), and synchronisation and collective latency (~0.3 s).

**Estimate** (an extrapolation from partial builds that happened to have small targets, not a measurement of a
balanced run):

- a small event goes from 2.0 s to about 1.3 s (−35 %);
- at 1.2–1.7 events per global step, individual mode would reach about parity to 1.3× faster than global.

Fragmented domains could make partial builds dearer than those samples, which is why the design gates on a
measurement.

### 2.3 The per-time-bin partition design (codex-converge loop, not approved)

- **Run and artifacts:** `20260925-134154-4060211`, in `~/.codex-converge/runs/20260925-134154-4060211/`
  (`solution.md`, `review-1..4-R1.md`, `dissent.md`).
- **Setup:** Claude author, astra high/fast reviewer, 1 subagent. It stopped at the 4-round cap, not approved.
- **What the author and reviewer agreed:**
  - **Chosen design, option (a), GADGET-4 style.**
    - Cut the Hilbert curve into S·P pieces, with S = 16–64 per rank.
    - Repair pieces that violate a per-bin cap, using recursive bisection batched by depth.
    - Assign pieces to ranks with a greedy multi-constraint packer. Each rank gets about 1/P of each constrained
      bin's cells and of the total cells.
    - At S = 1 it is exactly today's scheme. The decision is a runtime criterion over measured windows, invalidated
      on domain changes. The partition collapses to S = 1 before entering global mode.
  - **Rejected, option (b):** a single contiguous range with a combined weight. It cannot meet two targets (total
    and active) with P − 1 cuts when the active cells cluster.
  - **Rejected, option (c):** helper ranks or a separate compute decomposition. It would duplicate the ghost
    machinery, which has already had inconsistency bugs.
  - **Gate:** build only if the measured, recoverable saving is at least 38 % of wall time (what is needed to go
    from 0.62× to parity), with headroom.
- **Where it stopped:**
  - **BLOCKER:** the final `solution.md` says "unchanged from round 3" for its migration and cost sections instead
    of restating them. That is a paperwork failure.
  - **MAJOR:** it treats busiest − mean busy time as a lower bound on savings. Two objections, both conceded:
    - an alternating bottleneck across barriers is invisible to phase-level numbers;
    - unmovable per-rank work is counted as recoverable.
  - **The design's fix:** regress per-rank busy time on per-rank active count, credit only the explained part, and
    validate with a static single-hot-rank experiment and an injected alternating-bottleneck experiment.
- **What must change in the code for multi-segment ownership** (survey by session `dbacb723`):
  - These assume one contiguous range per rank and must change: `CurveLoadBalancer::getOwner`,
    `weightedBalance3` (size − 1 borders), the `HilbertRectangularTree3D` leaf owners, restart I/O (boundaries
    only), and the active-Hilbert cache and skip bound.
  - Only "owner of a point" is asked of the HilbertPointsManager exchange, MadVoro `GetOwner`/`Neighbors` and
    `DistributedOctEnvAgent`.
  - MadVoro ghosts use `getIntersectingRanks` on unions of leaf boxes, but fragmented ranks weaken the pruning of
    large queries.
  - The ParMETIS balancer is an empty stub, so no multi-constraint balancing exists.

### 2.4 Mesh-closure work done today (item 1), and why it did not pay

Measured with the parity check on: `RICH_VERIFY_PARTIAL_BUILD=1`, 0 mismatches in every arm (42, 20 and 16
checked builds).

- **Threshold sweep, morning.** Jobs 10208401 (0.5), 10208402 (0.35), 10208403 (0.7), 10208408 (1.0).
  - A full build costs 0.37–0.40 s at 3.1 M cells, and 0.62 s at 4.7 M.
  - Above 10 % of a rank's cells a partial build takes 3 attempts, 0.11–0.14 s each; at 100 % it takes ~0.34 s
    per attempt.
  - A partial mesh saves about 0.2 s of downstream work per event (hydro, radiation, commit), which puts the
    whole-event break-even near a fraction of 0.5–0.6.
  - The threshold made at most ±5 % difference in throughput.
- **Why partial builds take 3 attempts.** Counters were added to `INDIVIDUAL_MESH_BUILD` (`rebuilds_additions`,
  `rebuilds_depth_only`, `reexpansions`).
  - All 86 of 86 rebuilds were caused by **added** target cells: new neighbours missing from the cached adjacency,
    about 1 % of the target (for example 2,909 → 2,931 on the busiest rank).
  - No rebuild was caused by depth changes alone.
- **`RICH_INDIVIDUAL_CLOSURE_REEXPAND`** (default off; astra APPROVE). Remote depth decreases are re-expanded on
  the same mesh instead of rebuilding. Correct, but it never triggered.
- **`RICH_INDIVIDUAL_ADJACENCY_SEED_SHELLS=3`** (default 2; astra APPROVE). Seeds one extra shell.
  - Rebuilds fell from 82 to 14, and the median partial cost from 0.745 s to 0.397 s.
  - But the larger target pushed more events over the 0.5 threshold (full builds 40 → 64). Mesh time fell 8 %;
    wall time did not change (171.7 s vs 178.7 s, within node noise).
- **`RICH_INDIVIDUAL_PARTIAL_THRESHOLD_ADAPTIVE`** (default off).
  - A cost-model threshold: an exponentially weighted linear fit of time per partial attempt against the
    busiest-rank fraction, the mean attempts per build, and the mean full-build cost. It is the fraction where
    predicted partial cost equals full cost.
  - It settled at f ≈ 0.5–0.65, where partial ≈ full. No gain.
  - **astra CHANGES_REQUIRED, not fixed:** once the threshold is below every candidate's fraction, sampling can
    lock into full builds; the fit mixes attempt sizes and aborted builds; the slope is not identified with few
    samples. **Do not enable.**
- **Conclusion.** At the fractions this problem produces, partial and full builds cost about the same. The mesh
  lever is a partition that removes the concentration (§2.3), not a better choice between the two.

---

## 3. Issue B — FMM gravity per event

### 3.1 What is measured

Solve phases, medians of the max over ranks. Job 10208401, `RICH_FMM_TRACE=1`, solves without a topology rebuild:

| | Solves | Median active | Total | Redistribution | Local traversal | LET execute | of which M2L | P2P | Exchange |
|---|---|---|---|---|---|---|---|---|---|
| Global step | 51 | 3,134,737 | 0.134 s | 0.048 | 0.034 | 0.060 | 0.023 | 0.013 | 0.026 |
| Individual, all active | 10 | 3,161,063 | 0.147 s | 0.053 | 0.044 | 0.078 | 0.027 | 0.021 | 0.041 |
| Individual, ≥1 % active | 47 | 120,623 | 0.183 s | 0.056 | 0.046 | 0.092 | 0.027 | 0.027 | 0.046 |
| Individual, <1 % active | 75 | **846** | **0.151 s** | 0.036 | 0.044 | 0.078 | 0.027 | 0.023 | 0.039 |

An event with 846 active cells pays more for its solve than a global step pays for all 3.1 M.

The hydro sub-phase "source" is 0.30 s per small event. It includes about 0.15 s outside the FMM solve: assembling
the source arrays of all cells (`context.gravity_source_points/masses/ids`) and applying the kicks.

### 3.2 Why it does not shrink with the active count

- **Evaluation is already target-only.** `ConservativeForce3D::ApplyIndividual` (ConservativeForce3D.cpp:180-212)
  calls `acc_.EvaluateIndividualTargets(...)` with only the active cells as targets. The **sources**, however, are
  all cells. Gravity is long-range: every active cell's acceleration depends on the mass of all 4.7 M cells.
- **The source side is O(N) every event.** Inactive cells move along their predicted positions and masses change,
  so each solve:
  - re-sends every source particle to its Hilbert gravity owner (redistribution, 0.04–0.05 s);
  - refreshes the tree moments;
  - exchanges the locally essential tree (LET) with the other ranks (~0.04 s).
- **The interaction side runs the full cached plan.** The M2L/P2P interaction lists and LET plan are built once
  for the whole target tree and reused (`let_plan` is 0 on reuse). Only the final evaluation at the leaves is
  restricted to the active targets. So local traversal plus LET execute (~0.12 s) is the same as a full global
  solve: far-field expansions are computed for every target node, then read off at a few hundred leaves.

### 3.3 What would make it scale (not started)

1. **Prune the interaction plan to the leaves holding active targets, and their ancestors.** P2P and M2L/L2L/L2P
   work then becomes proportional to the active region. This is the most contained change: about 0.1 s per event,
   entirely inside the FMM (`source/3D/gravity/fmm/mpi/`, `DistributedFmmGravityCalculator`).
2. **Update sources incrementally.** Re-send and re-accumulate only the tree nodes whose cells moved or changed
   noticeably, and keep far, slowly moving nodes between events. This would cut redistribution, the LET exchange
   and the source assembly. It is an algorithmic change with an accuracy trade-off, and needs direct-sum error
   checks (`RICH_FMM_DIRECT_ERROR_SAMPLES`; the FMM's own error is ~11 % RMS at order 2, theta 1.0, a pre-existing
   setting).
3. **Assemble the source arrays incrementally**, instead of from all cells every event (~0.15 s).

**Earlier history** (memory `rich-fmm-gravity-imbalance-box-growth`): the gravity splitter re-sampling (fix4) and
the sink cache fix (only one first-half solve in 121 events) already cut gravity from 38 % to about 12 % of
individual wall time. What is left is the per-solve fixed cost above.

---

## 4. The other fixed per-event costs

- **Radiation, 0.33 s per small event** (0.36 s in a global step). It is an implicit iterative solve: tens of
  BiCGSTAB iterations, each ending in a global reduction, plus setup. Iteration count and latency dominate. It grows
  with the active count (0.08 → 0.33 → 0.49 s), so part of it may be concentration, not yet measured.
- **AMR.** Every 10th event, over all cells (`MassRefine::ToRefine` scans every owned cell), about 1.5 s per pass,
  0.15 s averaged. A cadence in simulated time instead of events is a possible lever; memory says the user's AMR
  cadence decision is open.
- **Synchronisation and bookkeeping, about 0.3 s:** gathering event state, wake signals, scheduler commit, suggest,
  and the collectives at the end of every phase. Even the least-waiting rank spends 0.57 s inside MPI in a small
  event.
- **Cadence quantization.** Bins are powers of two anchored at the step at the moment of the switch.
  - After a restart the step is still ramping (9.3e-4 against a CFL step of 1.5e-3), so the finest bin is 0.6× the
    global step. That gives 1.64–1.72 events per global step.
  - After a normal switch at the settled global step, expect about 1.0–1.2.

---

## 5. What was found and fixed today (chronological, with evidence)

### 5.1 Two MPI deadlocks: rank-local "mesh already canonical" tests before a collective build

- **`AMR3D::Apply`** (via `ApplyIndividual`) decided `identity_mesh` per rank and skipped the collective
  `BuildParallel`.
  - A rank whose partial target covered 100 % of its cells skipped it. Job 10208404 deadlocked at cycle 3189, and
    gdb on 32 live ranks showed the split.
  - **Fix:** agree it with MPI_LAND. astra APPROVE. Runtime-validated: job 10208408 passed the same event.
- **`Simulation::rebalanceCommittedIndividualState`** had the same point-count-only test.
  - **Fix:** point count plus identity map, MPI_MIN. astra APPROVE. Static only, since the TDE never reaches that
    path.
- Neither is reachable at the default threshold of 0.5 on non-empty ranks.
- Memory: `rich-individual-collective-identity-gates`.

### 5.2 The time-step collapse, part 1: the thermal-loss guard at the sink edge

- **Trial:** A/B/G from `snap_full_54` (jobs 10208412 A, 10208413 B, 10208414 G). It needed two driver and code
  additions:
  - `RICH_TDE_TERMINAL_OUTPUT_TIME`: a synchronized snapshot at a given time in either mode, then stop;
  - `RICH_INDIVIDUAL_GUARD_FLOOR` in report mode.
- **Result:** every guard limit below the floor was the **thermal-loss** guard. The cells were near-vacuum
  (density 1e-18 to 1e-20) at r = 5.4–5.7, just inside the sink radius `rsmooth = 5.71`, plus a few fast
  near-vacuum cells at r ≈ 10–11. Their own CFL step is 3–25× the floor, but the guard cut them to 0.02–0.05 of it,
  and neighbour closure spread the fine bin.
- **With the floor applied:**
  - 4.4× faster (0.62× global against 0.14×);
  - closer to global at t = 20.9015: mass-weighted density L1 1.2e-2 against 2.7e-2; mass to 9e-16; Erad to 1e-4
    against 1.6 %.
- **Safety gate:** the crash-50107 restart (t 0.141 → 0.20, jobs 10208525 and 10208526) passed. There, drift and
  mass never fell below the floor.
- **Now the default:** `apply` (astra APPROVE). Memory: `rich-individual-thermal-guard-sink-cadence`.

### 5.3 The time-step collapse, part 2: the sink drained cells per event

- **Symptom:** a sustained forced-individual trial (job 10208596) degraded from 3.8e-4 to 5.8e-5 sim/s, and the
  finest bin fell from 30 to 28. The finest cells were CFL-limited sink cells at the 1e-20 density floor.
- **Cause:** `RemoveCenter` applied its ×0.8 factors to every active cell at every event, so the drain rate grew
  with the activation rate, a positive feedback loop.
- **Fix: a rate-based sink** (astra APPROVE after 4 rounds).
  - The factors are raised to the cell's interval over `HDSim3D::GetIndividualGlobalStepReference()`, the guard
    floor's cached CFL minimum, which is reset on every global step.
  - Temperature map: `max(1e4, min(1e7·0.8^(x−1), T·0.8^x))`. It composes exactly under any split of the interval
    (6e-16).
  - The outer T > 1e9 cooling runs as literal legacy steps for the whole part of the exponent.
  - The global path is byte-identical: jobs 10208625 and 10208626 with `RICH_FMM_GRAVITY_RESPLIT=0`, 8,195 of
    8,195 datasets identical.
- **Result** (job 10208627): the finest bin stays at 30, a constant 1.72 events per global step, 2.55× the old
  throughput over t 20.82–21.25.

### 5.4 Adaptive controller: robust probe window

- One probe measured 0.04× global from 7 events inside a burst, where the previous probe had measured 0.78×.
- **Fix:** an individual probe decides only once its measured window covers the coarsest occupied bin's interval,
  capped at 4× its wall budget, logged as `RICH_MODE_PROBE_EXTENDED`. astra APPROVE.
- The burst itself (halving event intervals at t ≈ 21.13) did not recur in sustained individual runs. Its cause is
  unknown.

### 5.5 From session `dbacb723` (merged)

- **AMR generator fix.** Individual AMR rebuilt its mesh from cell centroids, a flux-free Lloyd step, which caused
  bursts of about 1 % extra cells per probe. `amr_generators_fix_r2.patch`, astra APPROVE; regressions
  `amr_random` and `amr_random_individual` pass.
- **Task 1: skip rebuilds that are provably rejected** (`task1_hilbert_skip_r2.patch`, astra APPROVE).
  - Decision time 24.2 s → 3.6 s; individual wall −10%; identical migration outcomes.
  - A check mode `RICH_INDIVIDUAL_ACTIVE_HILBERT_BOUND_CHECK=1`.
- **Task 2: busy-time instrumentation**, fixed here: 5 findings, astra round-3 APPROVE.
  - `source/misc/mpi_wait_profiler.{hpp,cpp}`: 57 PMPI wrappers.
  - `Simulation::reportPhaseBusyTimes`: `<MODE>_PHASE_BUSY` records, enabled by `RICH_MPI_WAIT_PROFILE=1`.
- **Task 3:** see §2.3.

### 5.6 Other changes

- **FMM backlog patches applied:** `geomlog.patch` (the `RICH_FMM_GEOM_LOG` collective hazard) and
  `fmmtest_nits.patch`. `fmm_gravity_mpi` passes (campaign `regression_results/20260925_080232`).
- **Mesh-build instrumentation:** `INDIVIDUAL_MESH_BUILD`, and `RICH_INDIVIDUAL_PARTIAL_BUILD_FRACTION` as an
  override.
- **Changelog and user guide** (`docs/changelog.md`, `docs/user-guide/individual-timesteps.md`) updated for the
  floor, the sink, the probe window, the deadlock fixes and the instrumentation.

### 5.7 Long validation runs, morning (`rich_final1_20260924`, jobs 10208399 ADAPT and 10208400 GLOBAL; cancelled at ~4 h 50 min)

- No aborts to t ≈ 28.
- The controller switches both ways.
- Mass agrees with global to 1e-15.
- Data: `/data/users/elads/TDE_LONG_{ADAPT,GLOBAL}`, snapshots 50–54+. The `snap_full_54` of the GLOBAL run is
  the restart point for all of today's A/Bs.

---

## 6. Code state

### 6.1 Changes in the main tree, with their switches and defaults

| Change | Files | Switch (default) | Review |
|---|---|---|---|
| Collective identity gates | AMR3D.cpp, Simulation.cpp | — | APPROVE ×2 |
| Guard floor | hdsim_3d.cpp/.hpp | `RICH_INDIVIDUAL_GUARD_FLOOR` (**apply**; `report`, `off`) | APPROVE (4 rounds + default) |
| Terminal output | test.cpp | `RICH_TDE_TERMINAL_OUTPUT_TIME` (unset) | APPROVE (4 rounds) |
| Rate-based sink | test.cpp, hdsim_3d.hpp | — (individual mode only; global byte-identical) | APPROVE (4 rounds) |
| Probe window | Simulation.cpp | — | APPROVE (2 rounds) |
| AMR generator fix (dbacb723) | AMR3D.cpp, Simulation.cpp, IndividualTimeStep.hpp, regression | — | APPROVE (by dbacb723) |
| Active-Hilbert skip (dbacb723) | Simulation.cpp/.hpp | `RICH_INDIVIDUAL_ACTIVE_HILBERT_BOUND_CHECK` (off) | APPROVE (by dbacb723) |
| Busy-time profiler | mpi_wait_profiler.*, Simulation.cpp/.hpp | `RICH_MPI_WAIT_PROFILE` (off) | APPROVE (3 rounds) |
| Closure re-expansion + rebuild counters | hdsim_3d.cpp | `RICH_INDIVIDUAL_CLOSURE_REEXPAND` (off) | APPROVE |
| Extra seed shell | hdsim_3d.cpp | `RICH_INDIVIDUAL_ADJACENCY_SEED_SHELLS` (2) | APPROVE |
| Adaptive closure threshold | hdsim_3d.cpp/.hpp | `RICH_INDIVIDUAL_PARTIAL_THRESHOLD_ADAPTIVE` (off) | **CHANGES_REQUIRED, do not enable** |
| FMM backlog | DistributedFmmGravityCalculator.*, FmmLetPlan.*, fmm test | — | APPROVE (by da9e65a8) |

**Patches and reviews:** `/home/elads/mpi_tmp/claude-5014/-home-elads-RICH-ablation-integration/8b3f6501-5576-40d5-ae43-ef71993c29ce/scratchpad/`.
The patches are `*.patch`, the reviews `astra_*.md`, and the backups `*.before_*`.

### 6.2 Binaries (in `runs/M05R05MBH1e4MGComptonIndividual/`, each with a `.provenance.txt`)

| Binary | Contents | Status |
|---|---|---|
| `rich_final1_20260924` | Yesterday's candidate | superseded |
| `rich_amrfix2_20260925` | + deadlock fixes + FMM backlog | superseded |
| `rich_floordefault_20260925` | + guard floor default + AMR generator fix | superseded |
| `rich_probewindow_20260925` | + probe window | superseded |
| `rich_ratesink2_20260925` | + rate-based sink | superseded |
| `rich_combined_20260925` | + active-Hilbert skip | superseded |
| **`rich_busytime2_20260925`** | + busy-time profiler (off by default) | **current production candidate** (all reviewed defaults) |
| `rich_reexpand_`, `rich_seedshell_`, `rich_adaptthr_20260925` | mesh-closure experiments | experiments only |

The production symlink `rich -> rich_fmm_throttle_20260920` has **not been moved**. That is the user's decision.

### 6.3 Scripts

All in `runs/M05R05MBH1e4MGComptonIndividual/`.

- **`submit_gfloor.sh`:** restart from `snap_full_54` with ARM=A|B|G.
  - It refuses to start without a seeded run directory.
  - Pass-through variables: `GF_BIN`, `GF_TAG`, `GF_T_OUT`, `GF_T_END`, `GF_RESPLIT`, `GF_PROFILE`, and `GF_ENV`
    (space-separated `RICH_X=value` assignments).
- **`submit_floorgate.sh`:** the crash-50107 gate. Stage first with the `stage_restart_copy.sh` of session
  `e01101c6`.
- **`submit_long.sh`:** long ADAPT and GLOBAL runs (`LONG_BIN`, `LONG_TAG`, `LONG_FINAL_TIME`). It does **not**
  seed its run directory.
- **Seeding a run directory for a `snap_full_54` restart:** copy
  `/data/users/elads/TDE_LONG_GLOBAL/R0.47M0.5BH10000beta1S50n1.5Compton/snap_full_54.h5` (md5
  `d37acabde6f3243a55af7fb5e79f32d1`) into `/data/users/elads/<TAG>_<ARM>/R0.47M0.5BH10000beta1S50n1.5Compton/`,
  with `counter.txt` = 54 and `gravity.txt` = 1. Job 10208405 started from scratch without this.
- **Readout scripts** (in the scratchpad above):
  - `meshsweep.py` (mesh-build records);
  - `gfloor_readout.py` (A/B/G throughput, cadence, floor records);
  - `gfloor_snapcmp.py` and `gfloor_snapcmp2.py` (snapshot comparison by cell ID, robust mass-weighted norms by
    region).

### 6.4 Run directories created today (seeded copies or outputs; delete when no longer needed)

All under `/data/users/elads/`:

- `TDE_GFLOOR_{A,B,G}`, `TDE_ZENO_B`, `TDE_RSINK_B`, `TDE_GBIT{0,1}_G`, `TDE_BUSY_{B,G}`, `TDE_RX{0,1,1V}_B`,
  `TDE_SS{2,3,3V}_B`, `TDE_AD{2,3,3V}_B`;
- `TDE_crash_FLOOR{C,F}`, `TDE_FD_ADAPT`, `TDE_AMRFIX100_IND`, `TDE_AMRFIX100B_IND`, `TDE_MESHF{035,050,070,100}_IND`.

The `TDE_AMRDIAG_*` and `TDE_HSKIP*` directories of session `dbacb723` were deleted with the user's approval.
Never touch `TDE_individual_dt`, `TDE_global_dt`, `TDE_step1_r2off` or `TDE_LONG_*`.

### 6.5 Key record types to grep in the logs

`INDIVIDUAL_MESH_BUILD`, `INDIVIDUAL_GUARD_FLOOR[_SYNCHRONIZED]`, `INDIVIDUAL_CADENCE` (with
`RICH_INDIVIDUAL_CADENCE_TRACE=1`), `INDIVIDUAL_PHASE_BUSY` / `GLOBAL_PHASE_BUSY`, `INDIVIDUAL_ACTIVE_HILBERT_DECISION`
(in the stderr log), `INDIVIDUAL_HYDRO_PHASE_TIMING`, `RICH_MODE_SWITCH` / `RICH_MODE_DECISION` /
`RICH_MODE_PROBE_EXTENDED`, `TDE_SINK` (`reference_dt`, `exponent_min` / `exponent_max`), `RICH_TDE_TERMINAL_OUTPUT`,
`fmm_solve_trace` (with `RICH_FMM_TRACE=1`).

### 6.6 Busy-time profiler semantics

- **"Busy"** = wall time − time inside counted, blocking or polling MPI calls. It includes the initiation of
  non-blocking operations, local queries, bookkeeping and polling loops.
- **`heuristic_balanced` and `heuristic_imbalance`** are not bounds: they miss bottlenecks that alternate between
  collectives, and they count unmovable work as balanced.

---

## 7. Open decisions for the user, and recommended next steps

1. **Production candidate.** `rich_busytime2_20260925` has all reviewed defaults. It needs a long validation
   (ADAPT vs GLOBAL, with conservation) before the symlink moves. Not yet run with this binary.
2. **Per-bin partition (issue A).** Before building it, do stage 1 of the design:
   - plumbing that changes nothing at S = 1;
   - a short TDE run from a hand-built multi-segment plan loaded at restart, measuring the real small-event cost
     and the fragmentation overhead.

   Build the full scheme only if it clears the gate. The design document also needs its migration and cost
   sections restored.
3. **FMM (issue B).** Target-pruned interaction plans are the most contained lever, about 0.1 s per event. After
   that, incremental source updates, which need accuracy checks.
4. **Radiation and AMR cadence.** Measure whether radiation's growth with the active count is concentration. Decide
   on AMR cadence in simulated time.
5. **Commit scope.** Unanswered from the previous handoff: (a) a snapshot commit of the sources plus the needed
   untracked files, (b) that plus submodules inside-out, or (c) leave it uncommitted. The tree now also contains
   today's changes.
6. **Cleanup.** Decide whether to delete today's seeded run directories (§6.4). The experiment switches `CLOSURE_REEXPAND`,
   `ADJACENCY_SEED_SHELLS` and `PARTIAL_THRESHOLD_ADAPTIVE` could be removed if not wanted; the last has unresolved
   review findings.
7. **Still open from earlier:** whether to keep the timing-based FMM re-split trigger (recommended on in
   production, `RICH_FMM_GRAVITY_RESPLIT=0` for bitwise comparisons); the unused `CellRecord` fields in
   `regression_tests/cases/individual_box_growth/test.cpp` (remove at that case's next rerun, since its md5 is
   pinned).

## 8. Standing rules and gotchas (from memory, all still in force)

- **Build:** `./build_rich.sh intelReleaseMPI --test_name=BaseTDEComptonIndividual`, with OpenMPI 4.1.6 Intel on
  PATH. Copy the binary to a new dated name with a `.provenance.txt`.
- **Before every sbatch:**
  `bash /home/elads/mpi_tmp/claude-5014/-home-elads-RICH-ablation-integration/e01101c6-e79a-43bd-948e-806c2eefa64f/scratchpad/prescan_nodes.sh bigrun d25g d25g2,d25g10,d25g58`,
  then pass the printed `--exclude`.
- **Group CPU limit:** about 64 bigrun nodes; a 5th 16-node job pends on `AssocGrpCpuLimit`. Other users' STAR
  jobs share it.
- **Cancelling:** the permission classifier blocked `scancel` of our own jobs twice. The user cancels when asked.
- **Codex review command:**
  `/home/elads/.local/bin/codex -m gpt-6-astra -c model_reasoning_effort=xhigh -c service_tier="fast" exec -s read-only --skip-git-repo-check -C <repo> -o out.md - < prompt.md`.
  Its sandbox sometimes fails to read files (`bwrap` namespace error), so quote the code in the prompt.
- **Intel's `std::pow(x, 1.0)` is not always x** (~0.6 % of doubles). Keep legacy arithmetic verbatim where bitwise
  equality with global runs matters.
- **Deterministic comparisons:** use `RICH_FMM_GRAVITY_RESPLIT=0`. The timing-based re-split makes identical runs
  diverge at round-off.
- **Restarts from a snapshot** step globally from `init_dt`, so the first individual bins are anchored at a
  ramping step. Keep this in mind when reading events per global step after a restart.
