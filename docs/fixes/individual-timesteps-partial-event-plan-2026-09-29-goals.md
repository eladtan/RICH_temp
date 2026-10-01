# Individual time steps: user goals of 2026-09-29 (K=4, min dt, 1.5x) - work log

User goals (2026-09-29, AFK):
1. All runs with K=4 (RICH_INDIVIDUAL_MAX_BIN_SPREAD=4).
2. The individual run's minimum dt must not fall more than 30% below the global one.
3. The individual run must be at least 1.5x faster than the global run.

Metrics: scratchpad g2.py.
- G2 "run minima": the individual finest dt minimum over the global dt minimum in the same window.
- G2 "time-matched": each event's finest active-bin dt over the global step at the same time.
- G3: wall time from t_lo to the common end time.

## Root causes found and fixed (all reviewed by gpt-6-astra)

1. **Frozen mesh in the first individual interval after every global -> individual switch** (rich_s14/s15).
   - `IndividualTimeStepScheduler::initialize` zeroed every point velocity, and generators move by
     `point_velocity x dt`.
   - Traced cell 13873022: density 5.7e-13 -> 2.1e-12 and Tgas 2.5e4 -> 1.2e7 K in one interval. Its radiation
     limit 3e-4 put the cadence 3 bins below the anchor, and the scheduler held its neighbours there.
   - Fix: `HDSim3D::PrepareIndividualEntryPointVelocities` (point motion + ghost exchange + ApplyFix, as a global
     step) at `adaptiveEnterIndividual`, handed to `IndividualTimeStepScheduler::setInitialPointVelocities` by
     stable ID before the first-event rebalance.
   - Verified: the cell keeps density 5.6e-13 / Tgas 2.5e4 K, and the finest bin stays at the anchor.
2. **Gray individual radiation limit: radiation temperature from code-unit energy density** (rich_s11).
   - The energy density was not converted to cgs for the equilibrium factor.
   - Units fix approved; no effect on the observed collapse.
3. **Dyadic quantization at the anchor** (rich_s17 -> s20): a gray relative-change limit 4% below the anchor cost a
   whole bin. Anchor-crossing radiation subcycling, RICH_INDIVIDUAL_RADIATION_ANCHOR_SUBCYCLE, default on:
   - a limit in [0.5, 1) x anchor lets the hydro interval reach the anchor bin;
   - the limit is kept in CellTimeState (migrates, is checkpointed, inherited by AMR children and merged) and caps
     the radiation candidate fraction at the next activation.
4. **Segmented-ownership aging** (rich_s10 ledger coverage; rich_s20 per-class aging re-plan): a plan re-plans from
   current positions once the same-class excess over its first events has paid for a migration. The measuring
   revert is gated on aging.
5. **Explicit-quantum first-interval bound** (review item) and unit tests.

## Results, restart snap_full_69 (t=40.715) -> t=41.3, global control 10232917

| Run | Configuration | G2 run minima | G2 time-matched (events < 0.7) | G3 |
|---|---|---|---|---|
| SK4 | rich_s10, K=4, margin 0.8 | 0.28 | 40% | 1.26x |
| SK4V | rich_s14, K=4 | 0.56 | 37% | 1.18x |
| M9 | rich_s17, K=4, margin 0.9 | 1.27 | 1.7% | 1.42x |
| **N9** | **rich_s20, K=4, margin 0.9** | **1.27** | **1.7% (min 0.676)** | **1.62x** |
| N95 | rich_s20, K=4, margin 0.95 | 0.002 (collapse at t=40.90) | 46% | 0.90x |

- **N9:** no finest bin below the anchor in 358 events. The 6 marginal events are global dt spikes against a fixed
  anchor.
- **N95 collapse:** diagnostic arms D95ON/D95OFF (10232966/7) with the limit trace. Subcycling mitigates it (finest
  bin 26 with it on vs 20 with it off). The trigger is a strongly coupled cell (Fleck 0.41, density 1.9e-10) whose
  Er dropped 95% in one 1.48e-3 interval and then rebounded, consistent with the active cell solving against frozen
  (Dirichlet) passive neighbours. It is an open limitation of the individual radiation coupling; margin 0.9
  avoided it in this window.

## Long validation (submitted 2026-09-29)

- **TDE_P50K4** (10233030): rich_s20, adaptive controller, K=4, margin 0.9, cadence trace, from P50S7 snap_full_70
  (t=41.856) to t=50.
- **TDE_G50** (10233031): same binary, global, same restart, to t=50.
- **P50K4 early result (cancelled to free a slot):**
  - Bin-29 dips came from thermal-guard cells at 1.82e-3, just under the anchor of 1.88e-3.
  - A bin-28 dip came from a gray radiation limit of 7.46e-4, which is 0.40 x the anchor.
  - The controller's global phases held G3 at 1.03x.

## rich_s22 / rich_s23 (astra plan2, then reviews s22, s23 and s23b)

- **Entry reference (P1):** the anchor is 0.9 x min(global suggestion, entry-state CFL minimum with the entry
  velocities).
- **First interval (s23):** bounded by all synchronized per-cell limits of the entry state: CFL, sources after the
  post-step sink/AMR callback, and the drift guard. The bound is computed after the owned-only resize. A stationary
  mesh uses zero velocities (`HDSim3D::ComputeIndividualEntryIntervalBound`).
- **Gain grid (s23):** uses the current hydro minimum. The last switch's stale entry reference could otherwise veto
  the probe that would refresh it.
- **Stay policy (P3):** `RICH_ADAPTIVE_STAY_INDIVIDUAL=1`. Once an individual probe has been adopted, completed
  individual dwells start no global performance probe.
- **Radiation anchor band:** `RICH_INDIVIDUAL_RADIATION_ANCHOR_BAND` (default 0.5).
- **Candidate accuracy check:** `RICH_INDIVIDUAL_RADIATION_CANDIDATE_ACCURACY`, default off. Its latch is reset per
  candidate, and its normalization includes transaction-appended passive cells.
  - **It is unusable as designed.** In V22A (10233093, threshold 0.5) it rejected 147 consecutive candidates in the
    first (all-active) event, down to dt 8.6e-7, and never passed. A relative-change metric flags stiff cells that
    equilibrate faster than any dt; the job was cancelled.
  - Astra: the exchange accuracy needs a nonlinear-consistency or converged-reference test. Transport across
    active/passive faces is the right diagnostic for the Dirichlet mechanism, and one-ring radiation promotion is
    the structural fix if a boundary-driven cascade reproduces.
- **V23A** (10233165, run to t=50, resubmitted on rich_s24 = s23 + `RICH_INDIVIDUAL_PASSIVE_TRANSPORT_TRACE`, astra-approved): rich_s23 config from snap_full_70 with K=4, margin 0.9, band 0.25, subcycling on, the check
  off, stay-individual, and the limit and cadence traces. It is compared against TDE_G50.
- **V23A first attempt (10233165) was invalid:** d25g[73-88] ran ~60x slower in every phase. d25g73 hosts `opensm` and was sluggish. The run was resubmitted as 10233168, excluding d25g73.
- **V23A (10233168) to t=42.51:**
  - **Speed-up:** G3 is 1.19x overall; windows in adopted individual mode reach 1.54-1.87x.
  - **Controller:** there was one stale-baseline detour, when box growth at t=41.99 fell during the first probe. It was adopted at t=42.21 with a throughput ratio of 2.14. Under STAY a later growth only resets the measurement.
  - **G2 run minimum: 0.223.** The cause is one gray-radiation-limited cell, id 14676195 at t=42.3252:
    - L = 3.6e-4, which is 0.217 x the anchor, so below band 0.25; its bin fell to 27;
    - Er dropped 96% and rebounded the next event (Tgas 1.5e5 K, Trad 2e4 K, Fleck 0.404).
    - 5 events fell below 0.7.
  - **Astra:** relative-change limits satisfy L >= 0.15 x anchor for events at the anchor, so band 1/8 absorbs them. Bin lowering and subcycling both keep the frozen Dirichlet boundary, but they are not strictly equivalent in accuracy.
- **V23B** (10233171): as V23A with band 1/8, from snap70 to t=42.65, tracing cell 14676195.
- **V23B result (band 1/8), to t=42.43:**
  - No dip: in the t=42.26-42.36 window the speed-up is 1.52x with min ratio 0.93, against 0.86x / 0.116 in V23A.
  - Time-matched G2 minimum 0.727 and run minimum 1.535; no event below 0.7.
  - Radiation cost is unchanged, with no retries.
  - Band 1/8 is the final configuration.
- **Individual cost profile (V23A, t=42.46-42.86; 922 s individual vs ~1420 s global):**
  - 15 all-active events cost 263 s (radiation 9.5 s each over 16 anchor intervals).
  - 179 events with <1% of cells active cost 469 s, 2.6 s each: partial mesh 1.18 s, source 0.37 s, radiation 0.26 s, wakes 0.2 s.
  - Partial mesh builds are ghost-search dominated: about 7 collective rounds per attempt at ~55 ms each. 436 of 540 builds took 2 attempts, mostly depth-only rebuilds.
  - Astra: the cheapest cut is A1 (`RICH_INDIVIDUAL_CLOSURE_REEXPAND`), which is still gated.
- **V23C** (10233172): the final configuration from snap_full_70 to t=50, against TDE_G50: rich_s24, K=4, margin 0.9, `RICH_ADAPTIVE_STAY_INDIVIDUAL=1`, band 0.125, subcycling on, and the traces.

## 2026-09-30 - F2 finished, D1 rules out accumulated controller state

- **F2 (10233252, rich_s27, K=4, band 0.125, A1 on) finished at t=50, rc=0.**
  - Goal 1 (K=4) met. Goal 2 met: run-minimum finest-dt ratio **0.915**.
  - Goal 3 missed: **G3 = 1.282x** (need >= 1.50x). The 41.86->47.86 segment alone is 1.541x;
    the 0.5-windows after t=47.86 are 1.102, 0.846, 0.698, 0.794, 1.001x.
- **D1 (10233470, rich_s30, entry probe OFF)** restarted from F2's own `snap_full_75` (t=48.839)
  and re-ran 48.839->49.5 with a completely fresh radiation controller: **0.811x**, against F2's
  **0.766x** over the identical physical window. A fresh controller buys ~6%, so the persisted
  `candidate_fraction_ceiling` trap is real but minor, and the entry-probe fix
  (`RICH_INDIVIDUAL_RADIATION_ENTRY_PROBE`, astra-APPROVED, binary rich_s30_20260929) cannot on
  its own close goal 3. D1 also confirms rich_s30 reproduces rich_s27 on the default path.
- **The late-time cost is the radiation solve.** Per 0.5 window, F2 goes from
  (t=45.0-45.5) 292 events / 1064 s / hydro 570 / radiation 262
  to (t=49.0-49.5) 292 events / 2769 s / hydro 908 / radiation 1562.
  The event cadence is flat; per-event radiation goes 0.91 s -> 5.4 s. `RICH_STEP_DETAIL` puts all
  of the growth inside `phase=radiation-driver`; every fixed per-event overhead (gather-remap,
  ghost-sync, limit-update, commit-scatter) stays at ~0.05 s combined. A global step at t=48.85
  costs 5.8 s (hydro 4.35 + radiation 1.40).
- **Positivity-retry cascades carry the whole gap.** Splitting radiation_s by retry count per event
  over the run gives **3833 s of radiation inside events that retried at least once**, against
  ~1 s for an event that does not retry, plus 948 s of failed-attempt wall. Removing that excess
  takes F2 from 22749 s to ~19169 s, i.e. **1.52x** - so eliminating the cascades is, on its own,
  sufficient for goal 3.
- **Individual retries far deeper than global for the same physics.** Over t=48.86-49.86: G50 has
  54 steps with retries and never exceeds attempt depth 4; F2 has 65 events with retries and
  reaches depth 10, D1 depth 11. Every failure is
  `historical_positive_floor_single_cell_limit` / `single_cell_injected_energy_limit`
  (`conj_grad_solve.hpp:571`), one cell out of 12.7M injecting ~1.3e-4 of `global_E_max` against
  the 1e-7 cap, while the *global* injection ratio stays under its 1e-8 cap. With band 0.125 a
  coarse event's first candidate dt is only ~2x the global dt, which cannot explain 6-11 extra
  halvings; the frozen Dirichlet halo of inactive neighbours (stale by up to 16 finest ticks for
  coarse cells) is the prime suspect, since halving the active dt never refreshes it.
- **W2 (10233473)**: D1's window and configuration with
  `RICH_INDIVIDUAL_RADIATION_ANCHOR_BAND=0.5` (the code default; the band caps solves per event at
  2 instead of 8) to separate "many accuracy-driven pieces" from "positivity cascades". Note the
  V23A precedent above: band 0.25 let one cell's relative-change limit fall below the band and its
  bin dropped to 27, costing goal 2 (run minimum 0.223) - so W2 is read on the finest-dt ratio as
  well as on wall time.

## 2026-10-01: goals met (TDE_P1, job 10234880, rich_s39_20261001)

Late-time slowdown root cause and fix: see docs/fixes/radiation-momentum-positivity-design-2026-09-30.md.
- Cause: the gray matrix velocity term made the matrix non-M, so near-vacuum cells were driven negative.
- Fix: `RICH_RADIATION_MOMENTUM_POSITIVITY=1`, which lumps the excess onto the diagonal and adds two energy-conserving caps (kinetic and thermal). Reviewed by astra in astra_s34 to astra_s39; approved at s39.

Final run, snap_full_70 -> t=50.0015 (same settings as F2, plus the flag):
- G3: 16531 s vs G50 29173 s, a **1.765x** speed-up; every 0.5 window is >= 1.23x.
- G2 run minima: 9.07e-4 vs 9.35e-4, ratio **0.970**.
- K = 4.
- Radiation retries: 69 (row-sum certificate) vs F2 675 and G50 701.
- Closure maxima per row: reservoir 5.4e-16, coefficient 4.1e-13.
- Caps: thermal cap in 255 cells (9.5e-15 energy); kinetic cap in 1051 cells (impulse dropped 3.6e-13 of 6.8e-7).

Evidence: runs/M05R05MBH1e4MGComptonIndividual/notes/TDE_P1_10234880/.

Open items:
- A global flag-on control, for the fair comparison the design requires.
- A focused unit test of the analytic matrix and exchange.
- Separating the radiation-energy difference (about 2% at t=49.2) between the lumping and the old undo.
- Comment-only fix at the kinetic-cap underflow bound.
