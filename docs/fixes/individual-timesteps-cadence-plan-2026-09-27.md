<!-- Converged with astra (gpt-6-astra xhigh) over two codex-converge runs: 20260927-082820-3496099 and
20260927-094506-3904357, 4 rounds each, both ended at the round cap. This is the author's final revision
after the last review (unreviewed). Open [MAJOR] objections from the last review (review-4), all in Stage 4 and
the Stage 0 evidence rule:
  1. [MAJOR] §2.4.1 / Stage 4: the “no-lengthening invariant” is false. [Line 303] confuses a physics limit with the interval assigned after quantization. With old anchor `1` and limit `0.75`, the old interval is `0.5`; re-anchoring
  2. [MAJOR] §1.3.1: merged recipients can incorrectly remain marked “directly measured.” [Line 170] ORs the donor’s inherited flag into the recipient’s. When both inputs have freshly measured bounds, both flags are zero, so the mer
  3. [MAJOR] Stage 0: the window arm cannot replace missing adaptive accuracy evidence. [Line 437], repeated at line 751, lets the window “carry” the accuracy judgment when adaptive dropped samples exceed 20%. But the window lacks t
Stages 1-3 carried no open objections. -->

# Individual time steps: cadence and per-event cost plan (2026-09-27)

*Intended path: `docs/fixes/individual-timesteps-cadence-plan-2026-09-27.md` in `/home/elads/RICH-ablation-integration` (branch `codex/individual-timesteps`).*
*Plan only — no code was written, nothing was built, run, submitted or committed while producing it.*
*Predecessor: [individual-timesteps-balance-gravity-plan-2026-09-26.md](individual-timesteps-balance-gravity-plan-2026-09-26.md) (executed; log: [individual-timesteps-overnight-2026-09-27.md](individual-timesteps-overnight-2026-09-27.md)).*

**File paths.** Unqualified `IndividualTimeStep.{hpp,cpp}` and `Simulation.cpp` mean `source/newtonian/three_dimensional/simulation/`; `steps/*` means `source/newtonian/three_dimensional/simulation/steps/`; `hdsim_3d.{hpp,cpp}` and `ConservativeForce3D.cpp` mean `source/newtonian/three_dimensional/`. Every line number below was re-checked against the worktree on 2026-09-27.

---

## 0. Baseline, target, and what changed in the diagnosis

TDE radiation-hydro restart from `snap_full_54` (t = 20.8214549, 4.66 M cells), 256 ranks on 16 `bigrun` nodes, FMM self-gravity, multigroup Compton radiation. Binary `rich_seg24_20260927`; defaults `RICH_FMM_TARGET_PRUNE=1`, `RICH_INDIVIDUAL_ACTIVE_HILBERT_SEGMENTS=4`, `RICH_INDIVIDUAL_ADJACENCY_SEED_SHELLS=auto`, `RICH_INDIVIDUAL_WAKE_ROUTING_REUSE=1`.

| Quantity | Value | Source |
|---|---|---|
| Benchmark window, individual | **127.431 s over 83 events** | job 10213543 |
| Same window, global reference | **122.565 s over 51 steps** | job 10208630 |
| **1.15× gate on the window** | **≤ 106.578 s** | — |
| **Saving required** | **20.85 s** | — |
| Adaptive to t = 21, individual | 342.0 s (best), 366–381 s | 10213459 / 10213497 / 10213465 |
| Adaptive to t = 21, global | 307.3 s | 10213458 |
| Global step | 2.20 s median | 10213543 |
| Individual event, ~1 % active | 1.20 s median (0.97–1.48) | 10213543 |
| Smallest individual events | 0.54–0.90 s at 23–5 000 active cells | 10213543:885 |
| Scheduled AMR passes | **9 passes, 16.418 s** | 10213543 |
| AMR bucket on the other 74 events | **4.986 s** (3.963 s excluding one 1.023 s outlier) | 10213543 |
| Partial event-mesh builds | **73 builds, 33.545 s**; 45 two-attempt (36 depth-only), 28 three-attempt | 10213543 |
| `individual-wake-tree` | 6.648 s (mean ≈ max: fixed work, not imbalance) | 10213543 |
| Change wakes per event, this window | 1 … 1429 | 10213543 |

**One reference only.** Job 10208630 (51 steps, 122.565 s) is the global comparand, because its recorded endpoints match the window exactly. The "same-binary" figures from the round-2 discussion are withdrawn: job 10213458 ran `rich_seg10_20260927` and its quoted span does not end at 20.9015. The gate is **106.578 s**.

### 0.1 The cascade diagnosis

Last night's log blamed accuracy guards lowering passive cells' bins mid-interval, with `binnedEndTick` sending overruns to the finest-bin tick. Two independent replays (Opus's `scratchpad/review/cascade.py`; astra's 17-digit arithmetic on job 10213459) agree that is wrong **for the halvings**:

1. `prepareEvent` builds both event times as `time_origin_ + time_quantum_ * double(tick)` — `IndividualTimeStep.cpp:421-422`.
2. `HDSim3D::suggestIndividualWakeDeadlines` subtracts them: `double const next_interval = context.event_time - context.previous_event_time;` — `hdsim_3d.cpp:3859`, min-merged at `:3878`.
3. `commitEvent`'s wake loop quantizes that duration — `:616` → `quantizeTimeStep`, whose core is a **floor** (`:1216-1255`).
4. The wake end is placed at `checked_add(context.event_tick, wake_ticks)` — `:626-628` — **unaligned** to the bin grid (comment at `:621-623`). If it lands below every other cell's end it *becomes* the next event tick.

With an origin near 20.9 the difference of two such times is off by up to ~1 ulp(21) ≈ 4e-15 ≈ 0.004 ticks. A 2^k spacing that comes out a hair short quantizes to bin k−1; at the next event the "last spacing" is 2^(k−1) and rounds down again about half the time. The spacing can only halve or hold.

Evidence:

- Jobs 10213459 + 10213497, 175 events over 7 periods: the replay predicts a short wake at **36** change-wake events and at all 36 the next gap is exactly 2^fb. At the 112 change-wake events where it predicts nothing, the spacing halves 3 times — two genuine `finest_new_bin=29` events and one non-power-of-two spacing (10213497 cycle 5182, 96 → 64).
- Job 10213454 (285 events): scanning all 46 115 doubles consistent with the 12-digit printed anchor, the best candidate `0.0014157657437685114` explains **239 of 253** change-wake events; a median candidate gets 100 wrong. The 14 residuals: ~10 non-power-of-two spacings (864, 432, 416, 384, 192 …), 2 genuine bin-29 events, 2 aligned-grid cases.
- The **first** period after each restart is clean because its anchor is dyadic: `initial_dt = 0.000931322574615478515625 = 10^6/2^30`. Job 10213543 is 82 regular bin-30 advances plus a terminal clamp of ≈ 0.000351381297323.
- The finest-tick overrun rule adds **no** sub-grid events in these jobs (e.g. 10213459 cycle 5244, 1040 overruns, all joining a tick that already carried an event).

Three consequences:

- **The three candidate fixes in last night's log target the wrong mechanism.** Relaxing the finest-tick rule would weaken the job-10196266 safety fix (`:1200-1211`) and would not remove the cascade. Closed.
- **The benchmark window is not affected.** Zero float-shortened wakes. Stage 1 buys it nothing; it is a prerequisite for interpreting every other measurement. **This is also why the window arm can never stand in for accuracy evidence about the cascade regime** — it does not exercise the code path Stage 1 changes (Stage 0 item 3, and the **C arm** in §5.2).
- **CFL and thermal bin-29 drops are a separate ~2× effect in later periods.** In 10213497, 30 events have `finest_new_bin=29`; 24 have thermal-limited finest cells (floored at `INDIVIDUAL_GUARD_FLOOR floor=0.00141`) and 20 CFL-limited, against period anchors of 0.00147–0.00158. Counterexamples confirm independence: 10213459 cycle 5237 halves with **zero** preceding change wakes; 5243–5244 halve with 1040 overruns against 3420 change wakes; 10213497 cycles 5157–5158 halve with 15 overruns. That is Stage 4's problem.

---

## 1. Open item (a): the form of the wake fix

### 1.1 Why none of the three tabled options is adopted as stated

**(i) Integer ticks**, then quantize as today. Removes the float error, leaves two ratchet inputs:
- *The floor.* `quantizeTimeStep` is `floor(x/q)` then the power of two below. Whenever the last spacing Δ is not a power of two the wake lands at `e + 2^floor(log2 Δ)`, a new tick, and the new power-of-two spacing persists. Dominant residual in job 10213454 (≈10 of 14).
- *The lock.* "Wake at the last spacing" is self-referential: after a genuine bin-29 excursion the spacing stays 2^29 until an event with no change-woken passive cells.

**(ii) Join the next tick of the finest occupied bin.** Rejected: it *asserts* that the finest-bin grid tick carries an event. `minimum_occupied_bin_` is a minimum over `time_bin` labels (`:638-646`); an end tick is on the bin grid only when `begin_tick` was aligned, and wakes (`:626-628`) and the overrun branch (`:1208-1212`) both produce unaligned begins.

**(iii) Snap down to the latest already-scheduled tick at or before the requested deadline.** Never creates a tick, never delays past today's horizon — but it is **undefined** when the next scheduled event `T` lies beyond the requested deadline `D`: there is no scheduled tick in `(event_tick, D]`. It also retains the `Δ_last` dependency and hence the lock.

### 1.2 Adopted: level 1 ships in Stage 1; the level-2 valve is a separate conditional stage

**Level 1 — the wake ends at the next scheduled event. This is the shipped default and the only behaviour the standing gates measure.**

`end_tick = min(end_tick, T)`, where `T` = the minimum `end_tick` over all cells and all ranks immediately before the wake is applied, after every other endpoint-setting operation of the event (§1.3).

- **It cannot create an event, and that is checkable for free.** `T` is already the global minimum, so applying `min(end, T)` leaves the minimum at `T`: `T1 == T0` by construction. The verification costs nothing extra, because `Simulation.cpp:6260-6264` **already** performs `nextEventTick()` followed by `MPI_Allreduce(MPI_IN_PLACE, &next_tick, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD)` immediately after the insertion point. Finalization computes `T0` with one new Allreduce of the same shape; the existing reduction then yields `T1`; the invariant is `T1 == T0`. Net collective cost: **one** extra `uint64_t` Allreduce per event (≈0.1 ms on 256 ranks against a 0.54–2.8 s event).
- **It is exactly the documented contract.** `hdsim_3d.cpp:3850-3854`: *"a passive cell whose mass or energy has moved by more than a fraction of its activation value ends its interval at the next event"*; `docs/user-guide/individual-timesteps.md:962` repeats it.
- **No floating point.** Ratchet, non-dyadic floor and lock all disappear by construction. The 416 → 256 class is gone because nothing quantizes a spacing to a power of two any more; the post-bin-29 lock is gone because no deadline references `Δ_last`.
- **Quantitative bound.** For every cell, `end_tick ≤ begin_tick + ticksForBin(time_bin) ≤ event_tick + 2^time_bin` — the allowance invariant that `binnedEndTick` enforces (`:1196-1199`), that the active path satisfies by construction (`:585`), that the wake path preserves (it only shortens), that the AMR merge applies (`:953-956`), and that **Stage 2 restores on the remote closure path** (`:1526-1535`, today's hole). Applied to a cell in the finest occupied bin:

  > **`T − event_tick ≤ 2^minimum_occupied_bin_`** — the delay is at most one interval of the finest bin present anywhere in the simulation, the strictest timescale the scheduler resolves at all.

  The exception is cells already in overrun when a lowering request arrives; those are exactly what `INDIVIDUAL_BIN_OVERRUN` counts, and its count and worst ratio gate every stage. **Stage 1 therefore ships in the same binary as Stage 2**, and the bound is stated conditionally on it.
- **It reproduces today's behaviour in every non-cascaded period.** In a clean period `Δ_last = 2^b_min = T − event_tick`, so today's `event_tick + 2^quantize(Δ_last)` *equals* `T`. Level 1 is later than today only in (i) cascaded periods, where today's behaviour is the bug being fixed, and (ii) the event after an off-grid signal or radiation wake, where `Δ_last < 2^b_min`. Case (ii)'s frequency is measured in Stage 0.

**Level 2 — a deadline valve, specified here, built only in conditional Stage 1b.**

Given the guard fraction `f` (`RICH_INDIVIDUAL_WAKE_CHANGE_FRACTION`, default 0.25), an overshoot factor `c > 1` and `g = c·f`, for a passive cell with measured ratio `r > f` and elapsed ticks `E = event_tick − begin_tick ≥ 1`:

```
project = E · (g/r − 1)                      // signed double; NEGATIVE when r > g
D       = event_tick + clamp( floor(project), 1, T − event_tick )
end_tick = min( end_tick , D , T )
```

**Implementation trap, stated because it is easy to get wrong:** `project` is negative whenever `r > g`. It must be computed and floored in **signed** arithmetic and only then clamped into `[1, T − event_tick]`; computing `E·(g/r − 1)` in `std::uint64_t` wraps the negative case to ≈2^64, which would either overflow `checked_add` or silently impose no constraint at all.

**There is no `Δ_floor`.** The round-3 version clamped `D` at half a finest-bin interval, and that clamp *defeated the bound it was attached to*: at `r = 0.49`, `g = 0.5`, `E` = one finest-bin interval, the projection is `0.0204·E` but the clamp pushed it to `0.5·E`, giving a ratio at activation of `0.49 × 1.5 = 0.735 > g` even at a constant change rate. Removing the clamp restores the bound — **for the cells the bound can apply to**, and only for those:

- **Case A, `f < r ≤ g` — the proven case.** The cell accumulated `r` over `E` ticks, so the rate is `r/E`; after a further `Δ` the ratio is `r·(E+Δ)/E`. `project ≥ 0` here, so either `floor(project) ≥ 1` and `Δ ≤ E·(g/r − 1)`, giving **ratio ≤ g**; or `floor(project) = 0`, the clamp gives `Δ = 1`, and the ratio is `r·(1 + 1/E) ≤ **g·(1 + 1/E)**`. The overshoot factor is exactly `1 + 1/E` with `E ≥ 1` — i.e. **up to 2× at `E = 1`**, tending to `g` only for large `E`. The earlier "one part in 2^20" figure assumed `E ≈ 2^20` and is withdrawn as an unconditional claim; the diagnostic reports the realized minimum `E` over level-2 deadlines (`level2_min_E`).
- **Case B, `r > g` — an unavoidable violation, not a bound failure.** With `r = 0.75`, `g = 0.5`, `E = 2^30`, `project = −0.333·E`, the clamp gives `Δ = 1` and the ratio at activation is `≈ 0.7500000007 > g`. **No deadline can repair this**: the cell was already past `g` when the request was formed, in an interval that has already executed. The valve then does the only useful thing — the earliest legal deadline, one tick — and the event is counted in `change_wake_overdue` and reported as an *unavoidable* violation attributable to the guard fraction `f`, not to the scheduling policy. The remedy for case B is a **smaller `f`**; that is a separate accuracy decision and is not made here.
- **The event-count claim is withdrawn.** "At most double the event rate" was false: distinct cells receive distinct deadlines, each of which can become an event. The number of created events is **not bounded a priori**. It is screened by Stage 0 item 1 and **measured exactly only by the valve-enabled arm W1-V**.
- **Direction, stated once:** larger `c` ⇒ later deadlines ⇒ fewer creations and a looser bound; smaller `c` ⇒ earlier deadlines ⇒ more creations and a tighter bound.
- **`c` is an accuracy target, not an optimisation target** (default 2, giving `g = 0.5`).
- **Out of Stage 1's scope entirely.** Stage 1b exists only if Stage 1's accuracy gate fails or is inconclusive; even then the valve ships default-off behind `RICH_INDIVIDUAL_WAKE_CHANGE_DEADLINE`, because enabling it creates off-grid events from a change wake, which the standing cadence gate forbids (§5.1). It appears nowhere in §6's budget.

### 1.3 Where the finalization happens, and how the metadata travels

`commitEvent` is not the last endpoint-mutating operation of an event. Verified in `Simulation::stepIndividual`:

- `applyAMRChangeSet` at `:6089-6090` lowers ends through `apply_merged_state` (`IndividualTimeStep.cpp:953-956`, lambda at `:1000`) and a full `limitNeighborBins(..., no_sources, true)` (`:1056-1067`), even for an empty change set;
- `refreshCurrentAccelerationCaches()` at `:6125` writes `cached_acceleration`/`gravity_half_kick_pending` (not `end_tick`);
- `rebalanceCommittedIndividualState` at `:6236-6238` reaches `buildDataTransfer()` (`:3759-3782`), which migrates `CellTimeState` via `MPI_exchange_data(this->tess, this->individualScheduler->states(), false)` (`:3767`) and re-indexes at `:3782` — per-cell `end_tick` values survive, but **which rank holds which state changes**;
- the rebalance block ends at `:6259` (`#endif`);
- `:6260` `std::uint64_t next_tick = this->individualScheduler->nextEventTick();` is the **first read** of the next event tick, immediately followed by the `MPI_MIN` Allreduce at `:6261-6264`; `:6269` publishes `SetTimeStep`, `:6392-6393` write output, `:6400-6402` bump the cycle and latch τ.

> **Insertion point: after line 6259, immediately before line 6260.** Earliest legal (every `end_tick` mutator is above) and latest legal (6260 reads the next tick). It sits outside `#ifdef RICH_MPI`, so any collective inside must be self-guarded. The existing reduction at `:6261-6264` is reused as the post-finalization `T1`.

The deadline-attribution report moves with it; the existing `INDIVIDUAL_CADENCE` emission at `:6002-6022` precedes the AMR block (`:6072-6111`) and never sees `applyAMRChangeSet`'s edits or the rebalance.

**Transport: the metadata lives in `CellTimeState`.** Requests originate per event but must survive AMR remapping *and* the rebalance migration before finalization, and neither `IndividualAMRChangeSet` (`IndividualTimeStep.hpp:78-99`, IDs and `MergeTarget{removed, recipient, source_rank}` only) nor `AMRPendingBinResponse` (`IndividualTimeStep.cpp:146-173`) transports arbitrary per-cell data **today** — §1.3.1 extends the latter with the one numeric field that genuinely has to cross.

**The alternative transport is unavailable.** `Simulation::addMigrationBuffer` (`Simulation.hpp:252-253`, `:479-486`; registry at `:389-396`) would carry a parallel array, but `Simulation.cpp:2674` hard-refuses domain growth whenever any buffer is registered: `RequireOnAllRanks(this->migrationBuffers.empty(), "Domain growth cannot extend registered migration buffers")`. Box growth is in scope (`individual_box_growth` is one of the seven regressions). The `ExchangeChain` overload (`:3786-3794`) also moves only migration buffers, not scheduler states.

New fields on `CellTimeState` (`IndividualTimeStep.hpp:49-70`, which has nine today: `cell_id`, `begin_tick`, `end_tick`, `last_primitive_tick`, `time_bin`, `pending_neighbor_bin`, `point_velocity`, `cached_acceleration`, `gravity_half_kick_pending`):

| Field | Stage | Kind | Lifetime | Purpose |
|---|---|---|---|---|
| `std::uint8_t change_wake_flags` | 1 | operational + diagnostic | per event / per activation | **bit 0 = `pending`**: a change-wake request is outstanding for this event; set by the producer, cleared at finalization. **bit 1 = `sampled`**: this cell has been change-woken since its last activation and is *eligible* for an activation-ratio sample; set with bit 0, cleared when the sample is taken at activation **or when drained at a state-destroying boundary** (Stage 0 item 3). **bit 2 = `limit_not_directly_measured`** (Stage 4) |
| `std::uint64_t change_wake_epoch_token` | 1 | diagnostic | until activation | **Globally unique** validity witness: `(uint64(world_rank) << 40) \| local_accumulation_epoch`. A rank-local counter alone is unsafe — a stamp transported from rank A can numerically equal rank B's counter after B reset its accumulator |
| `std::uint8_t end_source` | 1 | diagnostic | **as long as the endpoint survives** | provenance tag for the current `end_tick` (§1.5); written at every `end_tick` assignment, **never blanket-cleared** |
| `std::uint64_t end_source_tick` | 1 | diagnostic | with `end_source` | the tick **of the operation that wrote the tag** (§1.5's per-site table) |
| `double change_wake_ratio` | 1b | operational | per event | the measured `r` consumed by the valve |
| `double next_interval_limit` | 4 | operational | until the next commit | the **complete** physics limit the closing event computed for this cell's next interval (§2.4.1) |

Payload cost: 18 bytes per migrated cell after Stage 1, 26 after Stage 4.

**The precedent to follow is `cached_acceleration` / `gravity_half_kick_pending`**: field on `CellTimeState` → `dump`/`load` → per-event vector → `commitEvent` write-back (`IndividualTimeStep.cpp:481-487`) → explicit handling at every bulk-reset site. Every site in that chain, corrected against the source:

| Site | File:line | Enclosing function | New fields |
|---|---|---|---|
| Producer → scheduler | `Simulation.cpp:5803-5804` → `:5874-5876` | `suggestIndividualChangeWakes` → `commitEvent` | set bits 0 and 1 and stamp the token **before** the `force_synchronized` early-exit branch (`IndividualTimeStep.cpp:493-530`, `return` at `:530`) |
| Migration serializer | `IndividualTimeStep.cpp:206-220` (`dump`) and `:222-238` (`load`), both `#ifdef RICH_MPI` | `CellTimeState::dump`/`load` | **extend both, in matching order** |
| Initialization | `:360`, `:362-379` | `initialize` | **drain then clear** (Stage 0 item 3); `next_interval_limit = +inf`, bit 2 set |
| New-cell factory | `:1083-1099` | `synchronizedCellState` (sole caller `Simulation.cpp:2742`) | **clear** explicitly; `end_source = unset`; `next_interval_limit = +inf`, bit 2 set |
| Gravity-cache invalidation | `:1168-1175` | `invalidateCachedAccelerations` (sole caller `Simulation.cpp:2911`) | **preserve** — writes only the two gravity fields |
| The real AMR path | `:913`, `:919`, `:928-929`, `:942-962`, `:1000-1108`, `:1056` | `applyAMRChangeSet` | whole-struct copies carry everything: `remapped[i] = existing->second` / `= parent->second`, so a refined child inherits endpoint, provenance and limit, and **bit 2 is set unconditionally on the child**. `apply_merged_state` sets bits 0 and 1 on a merge recipient, stamps a fresh token, writes `end_source = amr_merge` when it lowers `end_tick`, takes `next_interval_limit = min(recipient, donor)` — §1.3.1 for a cross-rank donor — and **sets bit 2 unconditionally on the recipient** |
| Forced-synchronized commit | `:516-528` | `commitEvent` (`force_synchronized` branch) | preserve the flags; store `next_interval_limit` and clear bit 2; `end_source = bin_aligned` |
| Terminal clamp | `:658-678` | `clampToTerminalTick` (**no callers today**; kept consistent) | writes `end_tick` ⇒ writes `end_source = terminal_clamp` |
| Synchronized bin limiting | `:1134-1142`, `:1145-1155` | `limitSynchronizedBins` | writes `end_tick` ⇒ writes `end_source = synchronized_limit` |
| Restore | `:874`, validation `:857-869` | `restore` | fields return at their defaults (bit 2 set); `end_source = unset_after_restart`; validation asserts bit 0 is clear |

#### 1.3.1 Cross-rank merges: the donor's numeric limit needs its own wire field

A boolean derived from *the fact* of a merge needs no transport — that is why bits 0, 1 and 2 can all be set on a recipient without touching any MPI struct. **A numeric bound is different**, and the round-2 statement that "no MPI response struct needs extending" does not extend to `next_interval_limit`.

The cross-rank merge path, read end to end:

- `IndividualAMRChangeSet::MergeTarget` carries `{removed_cell_id, recipient_cell_id, source_rank}` (`IndividualTimeStep.hpp:80-85`).
- Local donor (`source_rank == rank`, `:1039-1049`): both states are in `states_`, so `min(recipient, donor)` is available directly.
- Remote donor: the recipient's owner packs `AMRPendingBinRequest{source_cell_id, recipient_cell_id}` (`:1028-1056`, struct `:123-145`); the exchange is at `:1058`; **the donor's owner** fills `AMRPendingBinResponse{recipient_cell_id, time_bin, pending_bin}` (`:1060-1080`, struct `:146-173`); the exchange is at `:1082`; the recipient's owner applies it at `:1086-1090`.
- Serial mirror at `:1100-1108`.

**Change (Stage 4):**
1. `AMRPendingBinResponse` (`:146-173`) gains **one** field: `double next_interval_limit = std::numeric_limits<double>::infinity();`.
2. Its inline `dump` (`:152-160`) and `load` (`:162-172`) gain matching `insert`/`extract` calls **in the same order** — the struct is `Serializable` and both bodies are three lines, so the change is mechanical and reviewable.
3. The response-packing loop at `:1063-1080` fills it from the donor's `CellTimeState`.
4. `apply_merged_state` (`:1000`) takes the donor's limit as a parameter; the local branch passes the donor state's value, the remote branch the response's, the serial mirror the `old_states` entry's. The recipient takes the `min`.

**No `limit_inherited` byte is transported.** An earlier draft ORed the donor's bit 2 into the recipient's, which is wrong in the most common case: when both donor and recipient carry freshly *measured* bounds, both bits are zero and the OR would leave the merged recipient marked "directly measured" — yet the merged cell is a *different* cell, whose accumulated mass- and thermal-loss bounds were measured for neither of its inputs. **Bit 2 is therefore set unconditionally on every merge recipient** (and on every refined child), which makes the donor's flag irrelevant and removes it from the wire. Only the `double` crosses.

**Fallback if the wire change is refused at review:** mark any recipient of a **remote** merge (`target.source_rank != rank`) as `next_interval_limit = +inf` with bit 2 set. The §2.4.1 availability rule then **defers** the next re-anchor instead of proceeding on an unbounded recipient. Safe, but it costs a re-anchor opportunity after every AMR pass with cross-rank merges, so the wire change is preferred and the fallback is recorded as the safe degradation.

**Restart I/O — a decision for the reviewer.** The restart path is a *separate* serializer: flat parallel HDF5 arrays at `write_simulation.cpp:258-287`, read at `read_simulation.cpp:120-167`, restored at `:402-405`, explicitly versioned (`write_simulation.cpp:94` writes 10; `read_simulation.cpp:88-91` whitelists 1–10).

- **(A) Do not persist the new fields — no version bump. Recommended.** The operational bit is clear at write time (finalization at `:6259` precedes output at `:6392`), **enforced by a throw** in `writeIndividualTimeSteps`. The diagnostic fields do not survive, which Stage 0 item 3 handles as the single remaining measurement boundary — a **process restart** — and nothing else. `next_interval_limit` does not survive either, so a restart defers the first re-anchor through §2.4.1's ordinary availability rule, with no special case.
- **(B) Persist the new fields and bump to version 11.** Cost: new datasets, `version = 11`, a `version >= 11` back-compat branch near `read_simulation.cpp:138-142`, defaults for older files. The only route to cross-restart continuity and to a re-anchor on the first post-restart event.

`T0` cannot be folded into the reduction at `:638-646`, which precedes closure, AMR and the rebalance.

### 1.4 Separating change wakes from physical wakes

Today the conserved-change wake and the tree/radiation wakes share one `std::vector<double> signalWakeDeadlines` (allocated `Simulation.cpp:5650-5652`; merged at `:504-516`, `:651-668`, `:5803-5804`).

- `steps/PhysicsStep.hpp:48-51` — add `virtual void suggestIndividualChangeWakes(IndividualStepContext const&, IndividualChangeWakeRequests&) const {}`, the request carrying `{global_index, ratio, epoch_token}`.
- `steps/HydroStep.hpp:34-35` — override.
- `hdsim_3d.cpp:3847-3888` — `HDSim3D::suggestIndividualWakeDeadlines` becomes `suggestIndividualChangeWakes`. Delete `:3859-3861` (the float difference and its `> 0` guard). Line `:3878` emits a request instead of a duration. Keep verbatim: the size guard (`:3855-3857`), the passive-only filter (`:3867-3868`), the ratio (`:3871-3875`), the `ratio > change_fraction` test (`:3876`), and the `INDIVIDUAL_CONSERVED_GUARD change_wakes=` report (`:3886-3887`). Update `hdsim_3d.hpp:87-91`.
- `Simulation.cpp:5803-5804` calls the new virtual; `commitEvent` stores into `CellTimeState`.
- `IndividualTimeStep.cpp:610-631` — the change-wake handling is removed from `commitEvent`; the signal/radiation wake loop stays with its existing quantization.
- `RadiationStep::suggestIndividualWakeDeadlines` (`steps/RadiationStep.cpp:1072-1086`) keeps the double vector and today's conservative downward quantization; its `context.time_quantum` fallback wake is an event-creating physical deadline and stays.

**Do not add an epsilon to `quantizeTimeStep`** — it would round genuine physical limits upward. Keep the throw at `:1236-1240`.

### 1.5 The deadline-attribution diagnostic

1. **Creation, defined without reference to provenance.** `T0` = the minimum `end_tick` over **all** cells and ranks immediately before the finalization; `T1` = the minimum immediately after, which the existing reduction at `Simulation.cpp:6261-6264` already produces. **A creation occurred iff `T1 < T0`.** With level 1 only, `T1 == T0` always — the shipped invariant, asserted in the regression.

2. **Provenance, a property of the endpoint, not of the event.** `end_source` values `{unset, unset_after_restart, bin_aligned, passive_limit, change_wake_level1, change_wake_level2, signal_wake, radiation_wake, closure_local, closure_remote, overrun_finest_tick, amr_merge, terminal_clamp, synchronized_limit}`, written wherever `end_tick` is assigned, migrated, not persisted, **never blanket-cleared**. The round-4 per-commit reset was wrong: the passive hard-limit loop at `:588-608` writes only when the limited end is *earlier*, so a cell keeping its endpoint receives no write and would report `unset` at the event where it finally wins.

   **The companion tick is the tick of the operation, not `current_tick_`.** `commitEvent` advances `current_tick_` only at its **end** (`:668`; `:528` in the synchronized branch), so stamping `current_tick_` at a commit-time assignment records the *previous* event and makes a brand-new endpoint report as inherited:

   | Assignment site | `end_source_tick` source |
   |---|---|
   | `:522`, `:585`, `:600-602`, `:626-628`, finalization | **`context.event_tick`** |
   | `:1499-1502`, `:1529-1531`, serial mirror `:1630-1637` | **the `event_tick` parameter** of `limitNeighborBins` |
   | `:953-956` (`apply_merged_state` inside `applyAMRChangeSet`) | **`current_tick_`** — correct here, because `applyAMRChangeSet` runs at `Simulation.cpp:6089`, after `commitEvent` set `current_tick_ = context.event_tick` |
   | `:1134-1155`, `:658-678`, `reanchor()` | **`current_tick_`** — these run outside an event |

   `winner_inherited` is `(end_source_tick != event_tick)`.

Emitted rank 0, one line per event from the finalization site, behind `RICH_INDIVIDUAL_CADENCE_TRACE` (`Simulation.cpp:5780`):

```
INDIVIDUAL_EVENT_DEADLINE event_tick= next_event_tick= spacing_ticks= created=
  T0= T1= level2_creations= level2_min_E= change_wake_overdue= carriers_at_min=
  change_wakes_applied= winner=<source> winner_set_tick= winner_inherited=<0|1>
  winner_id= winner_begin_tick= winner_time_bin=
  on_grid_bin=<largest b with next_event_tick % 2^b == 0>
  sources=bin_aligned:N,passive_limit:N,change_wake_level1:N,...
```

---

## 2. Open item (b): the anchor, and the bin-29 cliff

Opus ranked the anchor #2; astra ranked it #7. The agreed stage order resolves it: **first in importance for the gate, fourth in sequence.** It is the only item that changes the **number** of events; everything else reduces cost per event, and §6 shows per-event items alone cannot reach 20.85 s. It waits because its only measurement (job 10213454, 285 events, 263.837 s) is confounded — 239 of 253 change-wake events are the float ratchet — and because of a second defect:

**The old experiment pushed cells off the anchor bin.** `exp_GFLOORB_10213454.txt:709` shows `anchor_dt ≈ 0.00141576574377` against `initial_bin_dt ≈ 0.000707882871884`: the anchor exceeded what the finest cells could legally take, so they started one bin finer, in bin 29, halving the cadence immediately. Cycle 5112 creates 50 thermal-limited and 2 CFL-limited bin-29 cells; 5113 halves the spacing with zero preceding change wakes.

### 2.1 The control variable is `w_min`, not the bin label

The event rate is set by `w_min = time_quantum_ · 2^minimum_occupied_bin_`. Halving the quantum while doubling every tick preserves every physical event time: representation maintenance, **not** a cadence mechanism.

Bins are powers of two of the anchor, so the achievable `w_min` is the largest `anchor·2^k ≤ L_min`. Setting `anchor ≈ L_min` makes the quantization loss ≈0 at that instant; as `L_min` drifts down by any factor `< 2`, the finest cells drop a bin and `w_min` halves. **The only cure is a genuine re-anchor**, in either direction. Recovering that quantization loss — i.e. **lengthening** intervals that were short only because of where the bin grid fell — is the intended benefit, not a side effect (§2.4.1).

### 2.2 What the limit sets are, which provider knows what, and the entry anchor

Three limit sets exist and they are not interchangeable:

| Set | Provider | Contains | Missing |
|---|---|---|---|
| `dt_cell_min` (`adaptiveGainBound`, `Simulation.cpp:3359-3482`, printed `:3541`) | `PhysicsStep::collectCellTimeStepLimits` (`:3371-3373`) | wave-speed CFL (`hdsim_3d.cpp:3702-3710`, source only as one global scalar) + radiation from an ID-keyed cache | mesh drift, mass-loss, thermal-loss, per-cell source. Radiation fills **infinity** for unmatched IDs (`steps/RadiationStep.cpp:1112-1121`), counted in `cell_limit_fallbacks` |
| `synchronizedCellTimeStepLimits` (`Simulation.cpp:2850-2868`) | **`HydroStep` only** — `steps/HydroStep.hpp:53`; `RadiationStep` does **not** override `steps/PhysicsStep.hpp:94` | CFL without the global source cap, per-cell synchronized source limits, mesh drift | mass-loss, thermal-loss, **and all radiation** |
| `timeStepLimits` in `stepIndividual` (allocated `:5647-5649`, written `:5791-5810`) | `HydroStep` → `HDSim3D::suggestIndividualTimeSteps` (CFL + per-cell source + drift `:3455-3513` + mass `:3627-3644` + thermal `:3646-3664`) **and** `RadiationStep::suggestIndividualTimeSteps` (`steps/RadiationStep.cpp:1055-1070`) | **everything**, for every cell in `context.active_indices` | nothing, at an all-active event |

`anchor ≤ dt_cell_min` therefore establishes nothing, and Stage 4 does not use it.

**The infinity fallback is correct for its own consumer and wrong for an anchor.** `collectCellTimeStepLimits` comments that a missing cached limit "may overstate the gain, never understate it" — right for `adaptiveGainBound`, where overstating is conservative in the safe direction. For an anchor the sense is reversed: an overstated limit produces an oversized anchor. The two consumers need opposite conservatism, so the entry anchor must never consume that infinity.

**The mass- and thermal-loss limits are live at the closing event, not stale.** `hdsim_3d.cpp:3619-3663` writes `limit = loss_fraction · mass · interval / change.mass_loss` and the thermal analogue into `time_step_limits.at(global)` **for the next interval**, and only afterwards, at `:3664-3670`, resets the accumulators. An earlier draft called those bounds "meaningless at a re-anchor because the accumulators were just zeroed"; that was **wrong**.

The genuinely limit-free case is a **global→individual entry**, where no individual interval has executed. There:

- **Entry anchor:** `anchor = dyadic_floor(safety_entry · L_entry)` where `L_entry = collective_min( synchronizedCellTimeStepLimits , R )` and `R` is the **radiation bound**, obtained by this three-tier rule — never by an arbitrary safety factor:
  1. **Per-cell cache complete.** If `RadiationStep::collectCellTimeStepLimits` returns true with `cell_limit_fallbacks == 0` on every rank (`MPI_MAX` of the local count is zero), use its per-cell values.
  2. **Per-cell cache incomplete → the authoritative scalar.** Use `R = RadiationStep::suggestTimeStep()` (`steps/RadiationStep.cpp:1088-1091`), the global radiation timestep, applied uniformly. This is not an invention: it is exactly the fallback the production individual path already applies to any cell without a per-cell entry (`steps/RadiationStep.cpp:1067-1070`), and at a global→individual entry it was produced by the global step that just ran. It is a genuine bound, not a factor.
  3. **No authoritative bound → reject the anchor.** If no `RadiationStep` is present, or `suggested_dt` is not finite and positive, the new entry anchor is **refused** and the existing ramped entry path is kept unchanged, with `INDIVIDUAL_ENTRY_ANCHOR refused reason=radiation_unavailable` logged. The earlier "halve `safety_entry`" rule is **withdrawn**: an arbitrary factor cannot bound a limit whose magnitude is unknown, and `anchor ≤ L_entry` proves nothing when `L_entry` omits a term.

  Whichever tier supplies `R`, the tier is named in the log line together with the two guards that genuinely do not exist at entry (mass-loss and thermal-loss, which require an executed interval).
- **Hard invariant:** collectively assert `anchor ≤ L_entry`, so every cell can legally sit at `options_.initial_bin` (`IndividualTimeStep.cpp:361-379`). Otherwise refuse the anchor and fall back to the ramped dt.
- **`safety_entry` is measured across switches.** After the first all-active event the complete limit set exists; `L_min / L_entry` drives `safety_entry` for the next entry through a bounded trailing mean, in both directions. It compensates for the two accumulated guards, never for a missing radiation term — that is what tier 3 is for.

### 2.3 The in-period re-anchor criterion

Measured over a trailing window of `cooldown_events` (default 8): `w_min`; `L_min` refreshed at each all-active event; `w_cand = dyadic_floor(L_min)`; `mean_event_seconds`; the wall cost of the last all-active synchronized event (`lastStepSecondsMax`, `:6401-6402`); and the measured cost of a full source sweep (`IndividualFullSourceSweepReport::seconds_max`, `Simulation.cpp:564-800`), which a re-anchor forces (§2.4.4).

```
predicted_events_saved = remaining_window_sim_time · ( 1/w_min − 1/w_cand )
predicted_events_saved · mean_event_seconds
      >  amortization_factor · ( synchronized_event_seconds + full_source_sweep_seconds )
```

The subtraction is `1/w_min − 1/w_cand`, positive exactly when `w_cand > w_min`. One formula covers both directions: when `L_min` has **risen**, `w_cand > w_min` because the quantization loss shrinks; when `L_min` has **fallen** and the finest cells have dropped a bin, `w_min = anchor/2` while `w_cand ∈ [anchor/2, anchor)`, so `w_cand ≥ w_min`, up to 2× better. `amortization_factor` is `RICH_INDIVIDUAL_REBALANCE_AMORTIZATION` (2.0, `:1443`). A re-anchor that does not deliver doubles the cooldown.

**Only scalars cross the event boundary as *decision* inputs.** Per-cell physics bounds cross as a `CellTimeState` field (§2.4.1) — never as a loose vector indexed by position.

### 2.4 A dedicated, state-preserving re-anchor — not `adaptiveEnterIndividual`

`adaptiveEnterIndividual` **replaces** the scheduler: `Simulation.cpp:3339-3340` `this->individualScheduler = std::make_unique<IndividualTimeStepScheduler>(this->adaptiveMode.individualOptions);`. The new object is `initialized_ == false`, so the next `stepIndividual` re-initializes from nothing (`:5407-5409` → `IndividualTimeStep.cpp:342-382`), resetting every cell to `begin_tick = 0`, `time_bin = initial_bin`, `point_velocity = {}`, `cached_acceleration = {}`, `gravity_half_kick_pending = false` (`:366-378`). Also lost: `time_quantum_`, `time_origin_`, `current_tick_`, `last_full_source_sweep_tick_`, `force_all_active_latched_`, `minimum_occupied_bin_`, `id_to_index_`, and both radiation accounting structs (`IndividualTimeStep.hpp:416-417`).

**New method `IndividualTimeStepScheduler::reanchor(double new_anchor, double current_time, std::vector<double> const& fresh_limits)`**, operating in place.

#### 2.4.1 Limits: fresh geometry combined with carried physics bounds

Two independent defects had to be avoided, and both are real:

- *A vector measured in one event cannot be indexed in another.* `timeStepLimits` is local to `stepIndividual` (`:5647-5649`) and filled during the event (`:5791-5810`); between then and the re-anchor, `applyAMRChangeSet` remaps cells (`:6079-6090`) and `rebalanceCommittedIndividualState` changes ownership and order (`:6236-6238`).
- *Recomputing from synchronized providers loses real bounds.* `synchronizedCellTimeStepLimits` has only a `HydroStep` implementation, so a re-anchor built on it alone would drop mass-loss, thermal-loss **and every radiation bound**.

**Resolution: carry the complete bound per cell, in `CellTimeState`, and take the minimum with fresh geometry.**

- `commitEvent` already receives the complete composed vector as its `time_step_limits` argument. It stores `state.next_interval_limit = time_step_limits[i]` for every cell it commits and clears bit 2 — a *directly measured* bound.
- The field travels exactly like `point_velocity`: whole-struct copies in `applyAMRChangeSet` (a refined child inherits and has bit 2 **set unconditionally**), `min(recipient, donor)` at merges with bit 2 **set unconditionally** on the recipient and §1.3.1's wire field for remote donors, and the `dump`/`load` migration serializer.
- At the re-anchor, `fresh_limits` is built **at the invocation point, on the current mesh**, by a new `Simulation` helper running the loop at `Simulation.cpp:2850-2868` with `RequireOnAllRanks(step_limits.size() >= cells.size())` as `:2859-2860` does. The per-cell bound is

  ```
  limit[i] = min( fresh_limits[i] , states_[i].next_interval_limit )
  time_bin = quantizeTimeStep(limit[i])
  ```

- **The soundness invariant — and what it does *not* say.**

  > **Invariant (bound soundness):** for every cell, `new_interval = ticksForBin(time_bin) · time_quantum_ ≤ min(fresh_limits[i], next_interval_limit[i])`.

  It holds because `quantizeTimeStep(t)` returns the largest bin `b` with `2^b ≤ floor(t / time_quantum_)`, so `2^b · time_quantum_ ≤ t` by construction, and `t` here is the min of both bounds.

  An earlier draft stated this as a "no-lengthening invariant" — that the re-anchored interval is never physically longer than the one the closing event assigned. **That was false, and the error matters because the opposite is the point of the stage.** The limit is not the interval: quantization against a different anchor can yield a longer interval while respecting the same bound. astra's counterexample: with the old quantum giving bin widths `…, 0.5, 1, …` and a carried limit of `0.75`, the old interval is `0.5`; re-anchoring so that `0.75` is itself a bin width gives an interval of `0.75` — longer, and entirely legal, because `0.75 ≤ 0.75`. **Recovering exactly that quantization loss is the intended benefit** (§2.1): `w_min` rises, the event count falls, and §6's budget depends on it. What must never happen is an interval exceeding a physics bound, which is what the invariant above forbids and what Stage 4's tests assert.
- **Why radiation is not a deferral trigger here, unlike at entry (§2.2).** `next_interval_limit` **is the very vector `chooseNextBin` consumed at the closing event** (`IndividualTimeStep.cpp:571-572`), including whatever radiation bound `RadiationStep::suggestIndividualTimeSteps` applied — per-cell where cached, `suggested_dt` where not. The carried value therefore always contains a radiation term, so the re-anchor never needs to recompute one, and `RadiationStep::cell_limit_fallbacks` is retained in the re-anchor log line as a **cross-check diagnostic only**, not as a defer trigger. (At entry there is no predecessor interval, which is exactly why §2.2 needs its own three-tier rule.)
- **Availability, and the defer rule.** The re-anchor is **deferred**, not approximated, whenever a required bound is missing. Collectively agreed with one `MPI_LAND` before any rank branches:
  - any cell has `next_interval_limit == +inf` — a `synchronizedCellState` cell from box growth, a cell from `initialize`, any cell after a restart under option (A), or a remote-merge recipient under §1.3.1's fallback → defer, reason `missing_bounds`;
  - `fresh_limits.size() != states_.size()` on any rank → defer, reason `size_mismatch`.

  Deferral costs nothing: the re-anchor is a performance optimisation with a cooldown, so it waits for the next all-active event at which the cell set is fully bounded. `INDIVIDUAL_REANCHOR deferred reason=...` is logged; persistent deferral is itself a Stage 4 kill signal.
- **Strictness is selectable.** Default: cells carrying bit 2 (inherited or merged bounds) are accepted, because the child's or recipient's *geometric* limits are refreshed by `fresh_limits` and only the accumulated mass/thermal component is second-hand — the same treatment AMR gives every other conserved quantity. `RICH_INDIVIDUAL_REANCHOR_STRICT_BOUNDS=1` defers whenever any cell carries bit 2; since every AMR pass sets it on children and merge recipients, strict mode effectively defers re-anchoring across AMR passes, which is the intended conservative behaviour. The strict arm is run once in Stage 4 to measure what the relaxation is worth.
- **Cross-check, not a substitute.** Each re-anchor logs, for the ten tightest cells, the carried `next_interval_limit`, the `fresh_limits` value, the old and new intervals, the resulting bin, `individual_limit_reason_` (`hdsim_3d.cpp:3652`, `:3655`: 3 = mass, 4 = thermal) from the closing event, and `cell_limit_fallbacks`.
- **The decision is re-made on fresh data.** §2.3's criterion is re-evaluated against `L_fresh = collective_min(limit[])`. If `new_anchor > L_fresh` the anchor is lowered to `dyadic_floor(L_fresh)`; if the criterion no longer holds at the corrected anchor, the re-anchor is abandoned and the cooldown advances.

#### 2.4.2 Preconditions, checked collectively, throwing otherwise

1. `initialized_`.
2. Every state has `begin_tick == current_tick_ && last_primitive_tick == current_tick_` — `Simulation::IndividualStateSynchronized()` (`:2497-2525`), which `MPI_MIN`s the per-rank verdict at `:2519-2524`.
3. **The gravity half-kick phase is *uniform*, not a fixed value.** Let `v_min`/`v_max` be the collective min/max of `gravity_half_kick_pending`; require `v_min == v_max = v`.
   - **`v = 1`** — the ordinary post-event state: `ConservativeForce3D.cpp:246-248` sets `pending = (phase == FirstHalf ? 0 : 1)`, so after the `SecondHalf` phase at `hdsim_3d.cpp:3069-3072` it is 1 for every active cell, written back at `:3076-3082` and into `states_` at `IndividualTimeStep.cpp:484-487`. `IndividualSourcePhase::Full` is never used on the individual path.
   - **`v = 0`** — reachable and legitimate: `Simulation.cpp:2911` calls `invalidateCachedAccelerations()` on the box-growth fallback, clearing the flag for **every** cell; the success branch at `:2900-2906` sets it true for all `count_after` cells. Both are uniform, and box growth is in scope. In this branch additionally **assert `cached_acceleration == Vector3D()` for every cell**, the signature every zeroing path leaves (`:366-378`, `:1097`, `:1172`, `Simulation.cpp:2647-2650`). That distinguishes a cleared cache from the mid-event post-`FirstHalf` state, which must never be visible here and therefore throws.
4. No state has bit 0 set; `pending_neighbor_bin` unset for every state.
5. §2.4.1's availability rule passes, and `new_anchor ≤ collective_min(limit[])`.
6. `pending_full_source_sweep_` is about to be armed (§2.4.4).

#### 2.4.3 Why this is safe for kick-drift-kick — two separate arguments, both required

*(a) The interval length may change after the flag is set.* `gravity_half_kick_pending == true` means "the acceleration at this cell's interval-start time is cached; the first half-kick of the now-open interval has not yet been applied". The flag carries an acceleration, never a `dt`. Both halves are applied *at the event that closes the interval* — the deferred first half at `hdsim_3d.cpp:2753-2766` or `:2599-2605` (`ConservativeForce3D.cpp:290-322`), the second at `:3069-3072` — and both multiply `0.5 · context.cellTimeStep(index)` (`ConservativeForce3D.cpp:242`, `:251`, `:315`), which `prepareEvent` sets to `time_quantum_ · (step_end_tick − begin_tick)`, the *actual elapsed* interval (`IndividualTimeStep.cpp:449-450`). The scheduler already relies on this: `:588-608`, `:613-631`, `:665-667`, `:682-689`, `:1119-1155` all shorten a mid-flight `end_tick` with no fix-up, and none touches `begin_tick`.

*(b) The re-anchor does move `begin_tick`, and that is the invariant that matters.* Moving `begin_tick` while a half-kick is owed would leave the cached acceleration belonging to an earlier time than the interval it kicks over — the hazard guarded at `Simulation.cpp:2639` with `invalidateCachedAccelerations()` (`:2911`) as the blunt fallback. `reanchor()` sets `begin_tick = 0` **under a simultaneously reset origin** `time_origin_ = current_time`, and only from a state where every cell has `begin_tick == current_tick_`. Since `commitEvent` ends with `current_tick_ = context.event_tick` (`:668`, `:528`), that instant is `time_origin_ + time_quantum_ · current_tick_ = context.event_time` — exactly where `SecondHalf` evaluated the cached acceleration (`hdsim_3d.cpp:3070`). The physical interval-start time is identical before and after: a pure re-labelling of one instant. In the `v = 0` branch nothing is cached.

#### 2.4.4 The full-source-sweep clock

`Simulation.cpp:746-754` computes `elapsed_ticks = context.event_tick − last_full_source_sweep_tick`, divides by the finest-bin tick count, and sweeps when the quotient reaches `full_source_sweep_interval_minimum_steps` (default 128, `IndividualTimeStep.hpp:41`; runtime-adjustable at `:1374`). Setting `current_tick_ = 0` **and** `last_full_source_sweep_tick_ = 0` discards the elapsed age: with 64 of 128 intervals spent, the next sweep moves from 64 intervals away to 128. The earlier claim that the reset makes the sweep *"sooner, never later"* was **wrong** and is withdrawn.

The age cannot be carried in ticks: it would need a negative `last_full_source_sweep_tick_` relative to `current_tick_ = 0`, and any positive tick base would reintroduce a floating-point product into `time_origin_ + q·begin_tick`, destroying §2.4.3(b)'s exactness.

**Resolution: the re-anchor forces the sweep.** `reanchor()` sets `last_full_source_sweep_tick_ = 0` together with a new flag `pending_full_source_sweep_ = true`. The predicate at `Simulation.cpp:752` becomes `full_source_sweep = pending_full_source_sweep_ || elapsed_minimum_steps >= interval`, and the existing `recordFullSourceSweep` call at `:6027` (validated at `IndividualTimeStep.cpp:885-892`) clears the flag and sets the clock truthfully at the first post-re-anchor event. Worst case the sweep is **one event late**, never 128 intervals. Its measured cost is charged to §2.3.

#### 2.4.5 Operation and preservation

**Operation:** `time_quantum_ = ldexp(new_anchor, -options_.initial_bin)`; `time_origin_ = current_time`; `current_tick_ = 0`; `last_full_source_sweep_tick_ = 0` with `pending_full_source_sweep_ = true`; per state `begin_tick = 0`, `last_primitive_tick = 0`, `time_bin = quantizeTimeStep(limit[i])`, `end_tick = ticksForBin(time_bin)`, `end_source = bin_aligned`, `end_source_tick = 0`.

**Preserved verbatim:** `cell_id`, `point_velocity`, `cached_acceleration`, `gravity_half_kick_pending`, `change_wake_flags`, `change_wake_epoch_token`, `next_interval_limit`, `id_to_index_`, the force-all-active latch, both radiation accounting structs, and the hdsim-side accumulators.

**Invocation point:** top of `stepIndividual`, before `nextEventTick()` at `:5459`, when synchronized and collectively agreed on fresh data. Never mid-event.

---

## 3. Open item (c): is the adaptive-controller loss budget needed?

**It overlaps the cascade fix substantially but not completely. Conditional Stage 6.**

The two losing records are real: `10213497:6300` reports τ_individual = 1.049e-4 against τ_global = 6.141e-4 after 23.176 s; `:7735` shows 17.557 s. Both report `phase=dwell`, both sit in cascade periods, and τ excludes the ramp (`Simulation.cpp:3578-3583`). After Stage 1 most of astra's estimated 10–25 s is recovered by Stage 1 and must not be counted twice.

What Stage 1 does not touch: **switch entry/exit cost is never charged.** Each entry pays two all-active events at 2.3–4.9 s apiece plus a collapse of ≈0.58 s on the way out (`adaptiveEnterGlobal`, `:3203-3310`, collapse `:3230-3278`). Over the 8 switches of job 10213447 that is 40–80 s the decision at `:3660-3694` never sees.

```
ledger = Σ over EVERY event i in the excursion (entry all-active, ramp, dwell, exit all-active)
             ( advance_i / τ_global  −  wall_i )
         −  pure_switch_overhead_seconds
```

`pure_switch_overhead_seconds` covers only work that advances **no** simulation time: scheduler construction/reset, the segmented-ownership collapse, the ghost re-exchange at `:3288-3294`, and the partition migration. Ramp exclusion stays confined to τ estimation.

- **Entry gate for building Stage 6:** after Stages 1–5, in the adaptive-to-t = 21 run, **any individual excursion has `ledger < 0`**, including excursions whose post-ramp τ passed the `margin` test at `:3677`.
- **Abort rule:** abort a dwell whose running ledger falls below `−amortization_factor × pure_switch_overhead_seconds`. The abort calls `adaptiveRequestSwitch`; the **existing** gate at `:3588-3593` delays `adaptiveEnterGlobal` until `IndividualStateSynchronized()`, so synchronized-only exit is inherited.
- **Backoff is a required controller change, not existing behaviour** (§Stage 6).
- Keep the representative-cycle coverage requirement for **adoption** (`:3624-3658`).

---

## 4. The stages

Every stage: astra xhigh code review **before** any build or run; `build_rich.sh intelReleaseMPI --test_name=<dir>`; binary `rich_<tag>_20260927`. Stages 1 and 2 ship in one binary.

---

### Stage 0 — Instrumentation-only baseline (no behaviour change)

`INDIVIDUAL_CONSERVED_GUARD` prints only a count; the extreme ratio is rank-0-local and only at detailed level (`hdsim_3d.cpp:3215-3220`).

1. **Counterfactual valve screening — three quantities, each with its logical status stated.** For a grid `C = {1.2, 1.5, 2, 3, 5}`, per event:
   - **`requests[c]`** — cells with `D(c) < T`, `MPI_SUM`. A **request count**; it bounds nothing.
   - **`distinct_upper[c]`** — per-rank distinct deadline values below `T`, `MPI_SUM`med; an **upper bound** on new event times.
   - **`bites[c]`** — 1 if `min_cells D(c) < T`. Summed over the window it is a **lower bound** on added events, and the headline screening number.
   - **`min_gap[c]`**, a 16-bucket log-spaced histogram of `D(c) − event_tick`, and `overdue[c]` (§1.2 case B).

   The *cumulative* added-event count cannot come from an online counterfactual; only W1-V measures it.

2. **Paired representative observations.** Per event, the max `r`, min and max `E`, `T − event_tick`, and the top-N `(r, E, T − event_tick)` triples by `r` (`MPI_MAXLOC`).

3. **Ratio at activation: eligibility, validity, and accounting that survives mode switches.** Sampled immediately before the reset at `hdsim_3d.cpp:3664-3670`. Four hazards, all closed:

   - *Conflating eligibility with validity.* **Eligibility** is bit 1, set when a change wake is applied. **Validity** is `change_wake_epoch_token = (uint64(world_rank) << 40) | local_accumulation_epoch`, a globally unique witness; `local_accumulation_epoch` is a rank-local `uint32` in `HDSim3D`, incremented inside `ResetIndividualMeshState` itself (`hdsim_3d.hpp:211-221`, so no caller can forget) and at the re-seed branch (`hdsim_3d.cpp:2885`). A sample is **uncensored iff the stored token equals the sampling rank's current token**; a migrated stamp fails on the rank field, a reset on the counter field. Migration therefore always censors — correct, because `steps/HydroStep.cpp:126` clears `individual_conserved_change_` on every load balance.
   - *Samples vanishing inside a run.* Rank-level counters, `MPI_SUM`med at report time: `wake_requests_issued` (0→1 transitions of bit 1), `wake_samples_uncensored`, `wake_samples_censored`.
   - *State destroyed at a mode switch.* Every global→individual transition constructs a new scheduler (`Simulation.cpp:3339-3340`) initialized at `:5407-5409`, and `adaptiveEnterIndividual` also calls `physicsStep->afterIndividualAMR()` at `:3322-3323`, which for `HydroStep` clears `individual_conserved_change_` (`steps/HydroStep.cpp:89`). Both the eligibility bits and the accumulators die at **every** excursion boundary, so the adaptive arm contains several of them.

     Instead of declaring boundaries, **the loss is counted**. A new scheduler method

     ```
     std::uint64_t IndividualTimeStepScheduler::drainPendingWakeSamples();
     ```

     counts the states with bit 1 set, **clears the bit**, and returns the count; counting by clearing makes it idempotent. It is called, and its return added to a run-cumulative `wake_samples_dropped_at_switch` in `HDSim3D` (via a new `PhysicsStep` virtual overridden by `HydroStep`, mirroring `afterIndividualAMR`), at every point where the states are about to be destroyed or reset: `adaptiveEnterGlobal` (`:3203-3310`), `adaptiveEnterIndividual` (`:3312-3357`, **before** `:3339`), and `IndividualTimeStepScheduler::initialize` (`:342-382`).

     The counters live in `HDSim3D`, are **not** cleared by `ResetIndividualMeshState`, and are continuous across mode switches. The identity and the gate are **whole-arm**:

     ```
     issued   == uncensored + censored + dropped_at_switch + lost
     coverage  = uncensored / issued
     ```

     with `lost` meaning only destruction by AMR merge/derefinement or the sink. Coverage is additionally **reported per excursion**.
   - *A process restart.* The **only** remaining boundary: counters are process-local and eligibility is not persisted under option (A). Rule: a gate spanning a **process restart** is inconclusive for the pre-restart portion; `restore` clears bit 1 and emits `INDIVIDUAL_WAKE_SAMPLE restart_boundary=1 pre_restart_outstanding=unknown`. Every arm in §5.2 *begins* at its restart, so no arm contains an interior restart.
   - **The gate is one-sided.** Censoring and loss can only hide a violation, never invent one. An uncensored sample above `g` is a failure; **a censored sample above `g` is also a failure**; `dropped_at_switch` and `lost` have no measured ratio and count only against coverage. Reported: `coverage`, `censored_fraction`, `dropped_fraction`, `lost_fraction`, uncensored max and p99, censored samples above `g`, and the per-excursion breakdown.
   - **Each arm's accuracy verdict stands or falls on its own coverage.** An arm with inadequate coverage is **inconclusive**, and **no other arm's pass substitutes for it** — see the Stage 0 kill criteria and §5.1.

4. **Level-1 exposure.** Count events where `Δ_last < 2^minimum_occupied_bin_` with change wakes present (§1.2, case ii).

5. **Guard coverage gaps.** `suggestIndividualWakeDeadlines` early-returns when `individual_conserved_change_.size() != extensive_.size()` (`hdsim_3d.cpp:3855-3857`), so on the event following every clear the guard does not fire at all. Count those events.

6. **`individual-amr` sub-split.** Timers for (a) `CommittedGeneratorPoints()` (`Simulation.cpp:6078`), (b) the map/remap in `applyAMRChangeSet` (`IndividualTimeStep.cpp:901-940`), (c) the full `limitNeighborBins` (`:1056-1067`), plus a counter of bins actually changed by that closure on an empty change set, and a count of **cross-rank** merge targets (`MergeTarget::source_rank != rank`) so §1.3.1's frequency is known.

7. **Remote closure allowance.** Count remote requests arriving with `begin_tick + 2^bin ≤ event_tick`, and those where `binnedEndTick` would give an earlier end than the transmitted `maximum_end_tick`.

8. **Sweep cadence, limit provenance and radiation coverage.** Log `last_full_source_sweep_tick_`, `elapsed_minimum_steps` and the sweep's `seconds_max` per event; the distribution of `individual_limit_reason_` at each all-active event; `RadiationStep::cell_limit_fallbacks`; and `RadiationStep::suggestTimeStep()` at every global→individual entry.

9. **Cascade-regime snapshot selection.** From S0-A's `INDIVIDUAL_EVENT_DEADLINE` / `INDIVIDUAL_CADENCE` records, identify a period exhibiting the halving cascade and record the nearest preceding `snap_full_*` index. This index seeds the **C arm** (§5.2), the arm that supplies accuracy evidence *in the regime Stage 1 changes* without any mode switch to dilute coverage.

10. **Offline, no run required:** regress `step_s` on `active_cells` over the 83 `RICH_STEP mode=individual` records of job 10213543 to extract `f` and the active-count scaling.

**Runs.** Arms **S0-W** (window) and **S0-A** (adaptive to t = 21), binary `rich_instr0_20260927`, production defaults otherwise.

**Gate.** Wall time within 2 s of job 10213543 and a **byte-identical** terminal snapshot.

**Kill criteria.**
- If whole-arm `coverage < 0.80` on **S0-A**, the adaptive accuracy measurement is **inconclusive** and stays inconclusive until adequate coverage is obtained. **The window arm does not substitute**: §0.1 establishes that the window contains zero float-shortened wakes, so it never exercises the path Stage 1 changes, and a pass there says nothing about the cascade regime. The route to conclusive evidence is the **C arm** (item 9 and §5.2): a pinned-individual run seeded inside a cascaded period, which reproduces the affected regime with **no mode switch**, so `dropped_at_switch` is zero by construction and coverage is limited only by AMR/sink loss. If the C arm's own coverage is also below 0.80, the accuracy gate cannot be met with this instrumentation and the ID-keyed fallback below is mandatory before Stage 1 is judged.
- If whole-arm `coverage < 0.80` on any arm for reasons other than switches, fall back to a diagnostic accumulator keyed by stable cell ID, on the model of `individual_hydro_limit_by_id_` (`hdsim_3d.hpp:350`), which survives `ResetIndividualMeshState` and is pruned at `hdsim_3d.cpp:3373-3383`.
- If `bites[c]` at every `c ≤ 2` exceeds 0.5 per finest-bin interval, the valve is expensive; report it and leave the choice to the user.
- If `suggested_dt` is routinely absent or non-finite at entries, §2.2 tier 3 refuses every new entry anchor and Stage 4a cannot be evaluated — fix that first (Stage 7).

---

### Stage 1 — Tick-exact conserved-change wakes + deadline attribution

**Change.** §1.2 **level 1 only**; §1.3's four `CellTimeState` fields, transport table and finalization at 6259/6260; §1.4's plumbing; §1.5's diagnostic with the per-site tick rule; Stage 0 item 3's `drainPendingWakeSamples` and counters.

**Exact locations.** `hdsim_3d.cpp:3847-3888` (delete `:3859-3861`; `:3878` emits a request), `:3664-3670` (activation sampling), `hdsim_3d.hpp:87-91`, `:211-221` (epoch bump inside `ResetIndividualMeshState`), `hdsim_3d.cpp:2885-2895` (epoch bump); `steps/PhysicsStep.hpp:48-51` and the new drain-notification virtual; `steps/HydroStep.hpp:34-35`; `IndividualTimeStep.hpp:49-70` (four new fields) and the `drainPendingWakeSamples` declaration; `IndividualTimeStep.cpp:206-220`/`:222-238` (serializer), `:342-382` (drain in `initialize`), `:360-379`, `:481-487`, `:493-530`, `:516-528`, `:582-608`, `:610-631`, `:658-678`, `:857-869`, `:874`, `:913-962`, `:1083-1099`, `:1134-1155`, `:1168-1175` (preserve), `:1499-1502`, `:1529-1531`, `:1630-1637`, new `finalizeEventDeadlines`; `Simulation.cpp:3203-3310` and `:3312-3357` (drain at both switch points), `:5650-5652`, `:5803-5804`, `:5874-5876`, `:504-516`, `:651-668`, and the new call **after `:6259`, before `:6260`**, reusing the existing `MPI_MIN` at `:6261-6264` as `T1`; restart I/O per §1.3 option (A) with the enforced throw at `write_simulation.cpp:258-287`, or option (B) with `version = 11` at `:94`.

**Must stay.** Wakes never change `time_bin` (`:610-612`) and never seed closure. Wakes only shorten. Wakes apply to inactive cells only (`:616-617`). `binnedEndTick` untouched (`:1193-1213`). The passive hard-limit loop keeps `binnedEndTick` and `propagation_bins` (`:588-608`). `quantizeTimeStep` unchanged, no epsilon. Radiation and tree wakes keep their quantization and their right to create events. `invalidateCachedAccelerations` keeps writing only its two gravity fields. The global path is untouched.

**Tests.** New case `individual_change_wake_grid` (serial + 64-rank MPI; `REGRESSION_INFO` + `test.cpp` + `check_individual_change_wake_grid_case` in `regression_tests/lib/regression_checks.sh`, modelled on `individual_box_growth`). Every assertion is **per finalization**:

1. **The level-1 invariant.** After every finalization, `T1 == T0`, with `T0` the minimum over **all** endpoints beforehand.
2. **Provenance survives an unchanged endpoint.** Event A sets a change-wake endpoint at X; events B and C pass with that cell receiving no limit and no wake; at the event where X finally wins, assert `winner = change_wake_level1`, `winner_inherited = 1`, `winner_set_tick` = event A's tick.
3. **A newly assigned endpoint is not reported as inherited.** For `:522`, `:585`, `:600-602`, `:626-628` and the finalization, assert `winner_set_tick == event_tick` and `winner_inherited = 0` at the assigning event. Fails against `end_source_tick = current_tick_`.
4. **Closure and AMR tick sources.** A closure endpoint reports the `limitNeighborBins` `event_tick`; an `apply_merged_state` endpoint reports `current_tick_`.
5. **Tie handling.** A change-woken cell tying an existing carrier at `T` records `created=0` even when it wins `MPI_MINLOC` by rank.
6. **Time-origin invariance.** Identical tick sequence at origin 0 and origin 20.9, non-dyadic quantum.
7. **Non-power-of-two spacing.** A signal wake giving a 416-tick spacing: no collapse to 256, no creation.
8. **No post-excursion lock.** Force a genuine bin-29 event, remove the bin-29 cells; the spacing returns to the bin-30 grid within one interval.
9. **Post-commit AMR lowering.** A change set applied after commit lowering the minimum from `T` to `U < T` leaves change-woken cells ending at `U`, with `end_source = amr_merge` on recipients.
10. **Forced migration.** Force a rebalance between commit and finalization; all four fields arrive intact and the deadline is applied on the new owner.
11. **Remote merge flags.** A cross-rank AMR merge leaves the recipient with bits 0, 1 and 2 set and **no** extension of `AMRPendingBinResponse` — this is the test that pins the distinction §1.3.1 draws: booleans derived from the fact of a merge need no transport, the numeric limit does.
12. **Lifecycle coverage, site by site.** After `initialize` and `synchronizedCellState` the fields hold their defaults; after `invalidateCachedAccelerations` a pending request, an eligibility bit and a token are **unchanged**; after `clampToTerminalTick` and `limitSynchronizedBins` the flags are unchanged and `end_source` reflects the new endpoint; after a `force_synchronized` commit the eligibility bit survives; a `dump`/`load` round trip preserves all four in order.
13. **Censoring and accounting, five cases.**
    (a) *Reset:* a scripted `ResetIndividualMeshState` between a change wake and the activation censors that sample; without one it is uncensored.
    (b) *Cross-rank token collision:* drive two ranks to the **same** `local_accumulation_epoch`, migrate a stamped cell, assert **censored**. Fails against a rank-local epoch.
    (c) *Mode-switch drain — the `global → individual → global → individual` regression.* Two full round trips with outstanding eligible cells at each exit. Assert: `drainPendingWakeSamples` returns exactly the number of cells carrying bit 1 at each destruction point; the entry-side drain returns 0 because the exit already cleared them; `wake_samples_dropped_at_switch` accumulates across both excursions; the counters are **never reset** by the switch; the whole-run identity holds exactly; and a whole-arm coverage number is produced.
    (d) *Fresh-process restart:* run to a snapshot with outstanding eligible cells, terminate the process, relaunch, and assert no eligible bit is set, counters start at zero, `restart_boundary=1` is logged, the identity holds within the new process, and only a *restart*-spanning span is marked inconclusive.
    (e) *Idempotence:* calling `drainPendingWakeSamples` twice returns `n` then `0`.
14. **Lost accounting.** A cell change-woken then destroyed by an AMR merge appears in `lost`, not in `uncensored`, `censored` or `dropped_at_switch`.
15. **Retained hardening.** `quantizeTimeStep(q·2^k) == k` for random `q`.

Plus the 7-case suite green.

**Runtime validation.** Arms **W1** (window), **W1-C** (cascade-regime pinned, §5.2) and **W1-ADP** (t = 21), each against its Stage 0 counterpart. Crash gate `submit_floorgate.sh ARM=F FG_BIN=rich_wake1_20260927`.

**Pass/fail gates.**
- *Correctness:* no negative-mass/energy abort; crash-50107 gate completes to `FG_T_END=0.20`.
- *Cadence:* **no off-grid events except signal or radiation wakes**, on every arm. Equivalently `created=1` never appears with `winner=change_wake_level1`, and `T1 == T0` at every finalization.
- *Cascade removal (the C arm's primary purpose):* on W1-C, the halving cascade is gone — no event whose spacing is half its predecessor's carries a change-wake winner, and the replayed float-shortening prediction fires zero times.
- *Safety:* `INDIVIDUAL_BIN_OVERRUN` count and worst ratio, and `step_overruns`, no worse than Stage 0.
- *Accuracy:* Stage 0 item 3's one-sided gate, **evaluated independently on each arm, whole-arm**: uncensored max and p99 ≤ `g` = 0.5, `coverage ≥ 0.80`, zero censored samples above `g`, identity satisfied exactly. An arm below 0.80 coverage, or spanning a process restart, is **inconclusive for that arm**, and **no other arm's pass substitutes for it**. The accuracy claim for the cascade regime rests on **W1-C**; W1's pass covers only the clean regime; W1-ADP's verdict is its own.
- *Physics:* terminal snapshot at t = 20.9015 **and** t = 21.0 against the corresponding global arms, mass-weighted L1 by region, no worse than the Stage 0 individual arm.
- *Conservation:* `RICH_MODE_SWITCH_STATE phase=before/after` sums unchanged to printed precision on every span.
- *Timing:* window ≤ 128.0 s; adaptive ≤ 342 s, expected 300–325 s. (Timing verdicts are unaffected by sample coverage.)

**Kill criteria.** Halvings persist on W1-C with `T1 == T0` everywhere → the ratchet has a second input; read the attribution histogram (prime suspect: the radiation fallback wake). The accuracy gate fails, or is inconclusive on **W1-C**, → **Stage 1b's entry condition** and a user scope decision; an inconclusive W1-ADP alone does not block Stage 1 but is reported as inconclusive and never as a pass. Window regresses by more than 1 s → profile `individual-commit` and the new reduction.

---

### Stage 1b (conditional) — the level-2 deadline valve

**Built only if Stage 1's accuracy gate fails or is inconclusive on W1-C**, and only after the user has seen Stage 0's screening. Never a production default in this plan.

**Change.** §1.2 level 2 behind `RICH_INDIVIDUAL_WAKE_CHANGE_DEADLINE` (default 0) and `RICH_INDIVIDUAL_WAKE_CHANGE_OVERSHOOT` (default 2), plus `change_wake_ratio`. Only `finalizeEventDeadlines` changes behaviourally.

**Tests**, with the accuracy assertions narrowed to the case where they hold:

1. **Case A, `f < r ≤ g`:** ratio at activation `≤ g` whenever `floor(E(g/r−1)) ≥ 1` — including `r = 0.49, g = 0.5, E = 2^30` giving `Δ ≈ 2.19 × 10^7` and exactly 0.5.
2. **Case A boundary, `r = g`:** `project = 0`, `Δ = 1`, asserted bound `g·(1 + 1/E)`, **not** `g`.
3. **Case A worst tolerance, `E = 1`:** asserted bound `2g`, `level2_min_E = 1` reported.
4. **Case B, `r > g`:** `r = 0.75, g = 0.5, E = 2^30` gives `Δ = 1`, ratio ≈0.7500000007, `change_wake_overdue` incremented, and **no** claim of conformance.
5. **Signed arithmetic.** With `r > g` and large `E`, `D − event_tick == 1`; a `uint64_t` computation wraps and fails.
6. **Monotonicity:** `D` non-increasing in `r` over `(f, 4g]`, continuous at `r = g`.
7. **Clamp upper bound:** `D ≤ T` always.
8. **Super-linear rate:** an accelerating change may exceed `g`; the overshoot must be recorded.
9. `level2_creations` equals the hand-counted number of finalizations where `D < T`.

**Runtime validation.** Arms **W1-V** and **W1-C-V** (valve on); the only measurement of cumulative event cost, and the only one in the affected regime.

**Pass/fail.** Accuracy gate passes on case-A samples; case-B counts reported separately; event-count increase reported, not bounded in advance. Labelled an experiment.

**Kill criteria.** If `overdue` dominates, the valve is the wrong instrument — the remedy is a smaller `f`, outside this plan's scope, and Stage 1b stops.

---

### Stage 2 — Remote neighbour-bin parity, and the `restore()` finest-bin cache

**2a. Remote closure bypasses `binnedEndTick`.** The requester packs `request.maximum_end_tick = nextAlignedTick(event_tick, limited_bin)` (`:1486-1489`), computed without the owner's `begin_tick`. The local owner applies `binnedEndTick(state.begin_tick, event_tick, requested_bin)` (`:1499-1502`). The remote owner applies the transmitted tick verbatim (`:1529-1531`). Counterexample: `B = 2^30`, `begin_tick = 0`, `event_tick = 2B`, old end `4B`, requested bin 31, finest occupied bin 30 — local gives `3B`, remote permits `4B`. Segmented ownership (S = 4) has made cross-rank faces the common case.

*Change:* `state.end_tick = std::min(state.end_tick, binnedEndTick(state.begin_tick, event_tick, request.maximum_bin));` with `end_source = closure_remote`, `end_source_tick = event_tick`, and drop `maximum_end_tick` from the wire struct. *Must stay:* the `state.time_bin > request.maximum_bin` predicate, `pending_neighbor_bin` bookkeeping under `!full_closure` (`:1532-1535`), `locally_changed` driving the collective fixed point (`:1547-1553`), the `maximum_bin + 2` cap (`:1377-1380`). The serial mirror (`:1630-1637`) already uses `binnedEndTick`.

**This stage is what makes §1.2's bound true.**

**2b. `restore()` never reconstructs `minimum_occupied_bin_`.** `restore()` (`:736-881`) assigns clocks, states, accounting and `needs_full_neighbor_closure_ = true` (`:870-880`) but not the cache, declared `std::uint8_t minimum_occupied_bin_ = 0;` (`IndividualTimeStep.hpp:394`). `read_simulation.cpp:402-407` then invokes closure, reaching `binnedEndTick` at `:1502`; with the cache at 0 the `minimum_occupied_bin_ < bin` branch (`:1208-1211`) fires for every bin and sends any overdue cell to `event_tick + 1`. Frequency UNVERIFIED (these runs enter through `initialize`), but live for any individual-mode restart — and the **C arm restarts into a live individual state**, so it exercises this path for the first time.

*Change:* recompute collectively via the existing `refreshMinimumOccupiedBin()` helper (`:930-939`) immediately before `initialized_ = true` at `:880`. Related: hoist `minimum_occupied_bin_ = options_.initial_bin;` out of the per-cell loop at `:373`.

**Tests.** New MPI case `individual_remote_bin_closure` (64 ranks): the counterexample; variants with aligned begin, allowance still ahead, both rank placements, unaligned begins, already-expired allowances, and a full-closure fixed-point iteration. Restart: serialize with `minimum_occupied_bin_ = 32`, `restore()`, assert `minimumOccupiedBin() == 32`; a round trip asserting the first post-restore closure produces no `event_tick + 1` end; a rank-with-no-cells case. Suite 7/7.

**Pass/fail.** `INDIVIDUAL_BIN_OVERRUN` worst ratio must not grow on any arm. Window within 1 s of the Stage 1 expectation.

**Kill criteria.** If parity costs more than 2 s, bring the measured frequency and added ticks to review — a safety fix is not reverted for speed without one.

---

### Stage 3 — `RICH_INDIVIDUAL_CLOSURE_REEXPAND=1` A/B (no code; parallel with 1–2)

The option re-expands remote depth decreases on the existing mesh and rebuilds only if some rank added a target — `hdsim_3d.cpp:2249-2261`, collective continuation `closure_reexpand && !any_rank(!additions.empty()) && any_rank(!remote_depth_frontier.empty())`. Read at `:350-384` (rank-agreed strict boolean, default off). On 2026-09-25 the run recorded **zero** depth-only rebuilds (status doc :174). Job 10213543 records **36** of 45 two-attempt builds (the `(2,0,1)` pattern, e.g. `:1901`). Per-attempt proxy ≈ **7.120 s**. Each attempt rebuilds from scratch (`:2017`, re-entered from `:2440-2453`).

**Arms**, all `submit_gfloor.sh ARM=B GF_BIN=rich_seg24_20260927` on the window: **R0** defaults; **R1** `GF_ENV="RICH_INDIVIDUAL_CLOSURE_REEXPAND=1"`; **R1V** = R1 + `RICH_VERIFY_PARTIAL_BUILD=1`, short.

**Pass/fail.** Terminal snapshot **byte-identical** between R0 and R1. R1V: zero `debug parity mismatch`. `reexpansions > 0`, `rebuilds_depth_only` reduced. Wall saving ≥ 3 s. Then the default flips on for individual mode.

**Kill criteria.** Any parity mismatch, any snapshot difference, or a saving below 3 s → leave the default off, record the measurement, re-budget §6.

---

### Stage 4 — Anchor re-test after Stages 1–2, then the runtime criterion

**4a — Re-test.** Re-apply `scratchpad/anchor_experiment_*.patch` on the Stage 1+2+3 binary with §2.2's corrections, including the three-tier radiation rule. Sweep `safety_entry ∈ {1.0, 0.9, 0.8}` plus baseline on the window, with one arm at t = 21.

**4b — The runtime criterion and the re-anchor operation.** §2.3's two-way criterion on §2.4's `reanchor()`.

**Exact locations.** `Simulation.cpp:3312-3357` (entry anchor, three-tier radiation), the fresh-limits helper modelled on `:2850-2868`, the sweep predicate at `:752`, new invocation before `:5459`; `IndividualTimeStep.hpp:49-70` (`next_interval_limit` + bit 2), `IndividualTimeStep.cpp:146-173` and its `dump`/`load` (**§1.3.1's one-field `AMRPendingBinResponse` extension**), `:206-238` (serializer, second extension), `:342-346`, `:361-379`, `:481-530` (store the limit at commit, clear bit 2), `:913-933` (refine: set bit 2 on children), `:1000` (`apply_merged_state` signature; set bit 2 unconditionally on recipients), `:1039-1049` (local donor), `:1063-1080` (response packing), `:1086-1090` (response application), `:1100-1108` (serial mirror), `:1083-1099` (`+inf`, bit 2), new `reanchor()` and `pending_full_source_sweep_`, `recordFullSourceSweep` at `:885-892` clearing it.

**Must stay.** Re-anchoring only at a fully synchronized all-active event with a **uniform** gravity phase, no pending change-wake request and no pending neighbour bin. **No per-cell vector is indexed across an event boundary.** The re-anchor **defers** rather than approximates when any bound is unavailable. The entry anchor **refuses** rather than approximates when no authoritative radiation bound exists. Every decision Allreduced before any rank branches. Radiation accounting, cached accelerations and half-kick phases preserved verbatim.

**Tests.** Unit:
- the anchor is dyadic; `anchor ≤ L_entry`; no cell starts below `initial_bin`;
- **Entry with incomplete radiation coverage — three tiers.** (i) `cell_limit_fallbacks == 0` → per-cell values used, tier 1 named in the log. (ii) A forced non-zero `cell_limit_fallbacks` on one rank → the anchor uses `R = suggestTimeStep()`, verified by making `suggested_dt` the binding term and checking the anchor equals `dyadic_floor(safety_entry · suggested_dt)`, and no infinity from `collectCellTimeStepLimits` reaches the anchor. (iii) `suggested_dt` non-finite or no radiation step → the anchor is **refused**, the ramped path is taken unchanged, `refused reason=radiation_unavailable` logged;
- **a re-anchor after an actual all-active gravity event succeeds** with the gravity fields unchanged (`v = 1`);
- **a re-anchor immediately after a box growth** succeeds on both `v` branches, **unless** the growth added cells, in which case it must **defer** with `missing_bounds`;
- **a mixed phase throws**; **a `v = 0` state with a non-zero cache throws**;
- **the cache-validity invariant** — after the re-anchor, `time_origin_ + time_quantum_ · begin_tick` equals each cell's acceleration evaluation time (§2.4.3(b));
- **Physics bounds are not lost — the central test.** Construct a state where the closing event's mass-loss and thermal-loss limits (`hdsim_3d.cpp:3619-3663`, `individual_limit_reason_` 3 and 4) and a radiation limit (`steps/RadiationStep.cpp:1055-1070`) are each strictly tighter than anything `synchronizedCellTimeStepLimits` can produce. After the re-anchor, assert **by stable cell ID** that each such cell satisfies the §2.4.1 soundness invariant against its own **independently recomputed** mass-, thermal- and radiation-limited deadline — computed in the test from the recorded accumulators, *not* read back from `fresh_limits`. Fails against a re-anchor built on `fresh_limits` alone;
- **Bound soundness, stated as the interval, not the limit.** For every cell, `ticksForBin(time_bin) · time_quantum_ ≤ min(fresh_limits[i], next_interval_limit[i])`;
- **Permitted lengthening — the benefit, asserted explicitly.** Construct a cell whose carried limit sits just above an old bin boundary (astra's case: old bin widths `…, 0.5, 1, …`, carried limit `0.75`, old interval `0.5`). After re-anchoring so `0.75` is a bin width, assert the new interval is `0.75` — **longer than before and still ≤ the bound** — and that `w_min` rose and the predicted event count fell. A test asserting "the interval never lengthens" would be wrong and must not be written;
- **Cross-rank donor limit — §1.3.1.** A merge whose **donor lives on another rank** and whose `next_interval_limit` is strictly tighter than the recipient's: assert the recipient ends with the donor's value, that it arrived through `AMRPendingBinResponse` (checked by the recipient's rank never reading the donor's `states_`), and that bit 2 is set. Repeat with a local donor and the serial mirror. **Also assert bit 2 is set when *both* inputs carried freshly measured bounds** — the case an OR of the donor's flag would have left unmarked. Under the §1.3.1 fallback build, assert the recipient is `+inf` and the next re-anchor defers;
- **Limit transport across transitions.** A re-anchor requested at event N and executed after (a) an AMR pass that refines and merges, (b) a rebalance, (c) a box growth. In (a) and (b) assert by cell ID that `next_interval_limit` arrived with its cell and that every bin respects the soundness invariant; in (c) assert deferral. With `RICH_INDIVIDUAL_REANCHOR_STRICT_BOUNDS=1`, assert (a) also defers, since every AMR pass now sets bit 2 on children and recipients;
- **partially elapsed sweep cadence.** Set `last_full_source_sweep_tick_` so half the interval has elapsed, re-anchor, and assert the **very next** event performs the sweep and calls `recordFullSourceSweep` with that event's tick;
- a **null re-anchor** reproduces the next event's state within the existing parity tolerance;
- **a re-anchor with non-empty radiation defect and retry accounting leaves that accounting byte-identical**;
- conservation across a genuine re-anchor, checked after the *next* event;
- a restart taken across a re-anchor reloads consistently, and the **first** post-restart re-anchor defers under option (A);
- the criterion fires in **both** directions on scripted measurement sequences and does not fire when the synchronized-event plus sweep cost dominates.

`individual_box_growth` and `amr_random_individual` unchanged. Suite 7/7.

**Pass/fail.**
- Event count reduced by ≥ 25 % on the window, with the fraction of events at `finest_new_bin < initial_bin` no higher than baseline.
- Wall saving ≥ 6 s **net of the forced sweeps**.
- **Terminal L1 against global no worse than the Stage 0 individual arm, at every span.** Not negotiable against cadence.
- **Bound integrity.** No cell's assigned interval exceeds `min(fresh, carried)` at a re-anchor, nor its own mass-, thermal- or radiation-limited deadline anywhere; `INDIVIDUAL_GUARD_FLOOR` activations no more frequent than Stage 0.
- **AMR coverage.** The driver's cadence is `(cycle+1) % 10` (`runs/BaseTDEComptonIndividual/test.cpp:2064`), so a lower event count lowers the pass count — 9 toward 6 over the window, against global's ~5. Compare per-pass `Removing N cells and refining M cells` records and total cell counts pass by pass.
- Conservation unchanged; crash-50107 gate green.

**Kill criteria.** Fine-bin fraction rises at every `safety_entry` → **stop 4b, keep the ramped anchor**. Terminal L1 or bound integrity degrades → revert. Entry anchors refused at tier 3 in most entries → Stage 4a is unevaluable; fix the radiation provider first (Stage 7). Deferrals prevent the criterion from ever firing → fix the cause (most likely §1.3.1 shipped as the fallback, or strict mode left on) before re-attempting; do not relax the availability rule. Forced sweeps cost more than the cadence saving → the criterion must stop firing. AMR coverage diverges → decouple the AMR cadence from the event counter first; new scope, stops this stage.

---

### Stage 5 — AMR: skip provably-identity work, and stop paying for two meshes

**5a — The no-op path, with the full closure retained.** On every event: `Simulation.cpp:6078` gathers generators (full array copy plus `RequireOnAllRanks`, `:2547-2577`); `:6079` calls the callback, empty whenever `(cycle + 1) % 10 != 0` (`test.cpp:2064-2065`); `:6089` calls `applyAMRChangeSet` unconditionally — the `old_states` map (`:901-911`), the `remapped` vector (`:913-933`), and a **full** `limitNeighborBins(..., no_sources, true)` (`:1056-1067`). Budget: 74 non-cadence events, **4.986 s** (3.963 s excluding the outlier).

**The full closure stays, on every event, unconditionally.** Full and incremental closure use different source sets — `canonical[i].source_bin = full_closure ? states_[i].time_bin : propagation_bins[i]` (`:1387-1388`, `:1423-1424`) — and an unchanged fine-bin cell that acquires a coarse neighbour needs full closure without being an incremental propagation source. A zero-change benchmark cannot authorise removing it, and a later periodic check cannot repair fluxes already executed.

*What is removed:* only work provably identity on a **collectively** empty change set — the map build and remap, via an early branch straight to the closure call; and the generator gather when the collective AMR-due predicate is false. The predicate is a new `Simulation::SetIndividualAMRDue(...)` (default: always true), agreed with `MPI_Allreduce(MPI_LAND)` as `:6106-6109` already does.

*Entry gate:* build 5a only if **Stage 0 item 6** shows the non-closure part is ≥ 1.5 s of the 3.963 s.

**5b — AMR and the event mesh.** `AMR3D::ApplyIndividual` tests mesh identity collectively (`AMR3D.cpp:2688-2703`, `MPI_LAND`) and, when the event mesh is partial, does a **full** `BuildParallel` at `:2717-2724` *before* knowing whether anything will change; it filters candidates to active cells (`:2744-2757`, `:2786-2799`), then either returns at `:2824-2835` having already paid the full build, or pays a second at `:2857-2860`. Six of nine passes start from partial event meshes.

*Change:* with 5a's collective flag known before the mesh is chosen, set `context.mesh_build_policy = IndividualMeshBuildPolicy::FullReference` (`IndividualTimeStep.hpp:26-30`, `:180`) for that event. `verify_partial_build` is a separate field (`:182`) — confirm at review that `FullReference` does not enable parity checking.

*Criterion that can switch back.* `median(partial) + median(full)` can never favour the partial host for non-negative costs, and mesh timings exclude the extra hydro work. The comparison is on **whole-event wall seconds for AMR-due events**, using the A3 ledger pattern: class-matched running samples per host; ≥ 2 credited samples of the alternative before switching; revert after two consecutive losing windows; each revert doubles the cooldown. With ~6–9 passes the window cannot converge the criterion, so the **window measurement is a forced-host A/B** and the criterion is validated on the t = 21 run.

**Must stay.** The collective `MPI_LAND` identity test (`:2695-2703`). The ghost exchange at `:2726-2732`. AMR's rebuild from `canonical_points` (`:2707-2715`). Active-only candidate filtering. Conservative remapping. `RequireOnAllRanks` whenever the gather runs.

**Tests.** `amr_random_individual` green. New `individual_amr_not_due`: with a false predicate, the committed scheduler state after N events is bit-identical to running with the predicate true against an empty change set — closure included. New `individual_amr_full_mesh_host`: a pass on a forced-full event mesh produces the identical candidate set and geometry. Suite 7/7.

**Pass/fail.** Per-pass AMR counts and total cell count match `TDE_AM0` pass by pass. Terminal L1 no worse at every span. Combined 5a+5b saving ≥ 2 s on the post-anchor event count.

**Kill criteria.** Divergence in per-pass AMR counts → revert 5b, keep 5a. Stage 0's split below 1.5 s → drop 5a. The forced-full host raises whole-event cost more than it saves → the criterion must revert; if it does not, the criterion is wrong and the stage stops.

---

### Stage 6 (conditional) — Adaptive-controller loss budget

Built only if §3's entry gate fires after Stages 1–5.

**The controller changes, stated explicitly as changes.**

- `dwellMultiplier` is written at exactly three places: `:3679` (reset to 1 on adopt), `:3686-3687` and `:3705` (doubling). Both doubling sites sit in probe/dwell branches a ledger abort **bypasses**. **Change 1:** the abort path applies `a.dwellMultiplier = std::min(2 * a.dwellMultiplier, o.dwell_backoff_cap);` before calling `adaptiveRequestSwitch`.
- `NotifyDomainChanged` (`:2579-2599`) bumps `domainEpoch` and resets `stepsInMode`, `wallMeasured`, `simMeasured`, `wallTotalInMode`, then logs. It **never** touches `dwellMultiplier`; the round-4 citation of `:2593` pointed inside the log statement. **Change 2:** add `a.dwellMultiplier = std::max<unsigned>(1, a.dwellMultiplier / 2);` there, with the rationale the controller already applies to stale baselines at `:3666-3676`. New behaviour, gated by its own test.

**The testable seam.** `adaptiveAfterStep` (`:3567`) splits into a thin member that gathers inputs and

```
void adaptiveDecide(AdaptiveModeState& a,
                    AdaptiveIntegrationRuntimeOptions const& o,
                    AdaptiveStepInputs const& in,
                    AdaptiveActions& act);   // act: is_synchronized(), request_switch(), enter_global()
```

`AdaptiveActions` holds callables so a test observes every decision without a mesh, MPI or physics, while the branch logic under test is the **real** one. The ledger arithmetic lives in a pure `AdaptiveLedger` struct (`simulation/AdaptiveLedger.hpp`).

**Tests — new case `individual_controller_ledger`** (serial for the pure tests; 4-rank MPI for test 4's integration variant):

1. **Every excursion event credited.** A scripted excursion (entry all-active event, 12 ramp events, N dwell events, exit all-active event): `value()` equals the hand-computed `Σ(advance_i/τ_global − wall_i) − pure_switch_overhead`.
2. **Overhead charged once** across two cooldown windows inside one excursion.
3. **Units.** Scaling all wall times by 2 changes the ledger by exactly `−Σwall_i`.
4. **Synchronized-only exit.** `is_synchronized()` false for k calls then true: `enter_global()` on the first true, never before. The MPI variant repeats against the real `IndividualStateSynchronized()` and the gate at `:3588-3593`.
5. **Profitable-but-slow-starting excursion is not aborted.**
6. **Backoff and re-entry — through the controller.** Drive to a ledger abort; assert `dwellMultiplier` doubled (Change 1) and capped; invoke the domain-change handler and assert it halved (Change 2), never below 1; assert re-entry after the shortened dwell. Fails against today's code on both halves.

**Gate.** The t = 21 run improves by ≥ 10 s against the Stage 5 arm, switch count not increased, and the controller demonstrably re-enters individual mode after a domain-epoch change.

**Kill criteria.** If the ledger aborts an excursion the τ comparison would have kept and the t = 21 total worsens, revert to reporting-only. If Change 2 produces thrashing around box growths, drop it and keep Change 1.

### Stage 7 (contingent) — the make-up lever

**Trigger: after Stage 5, the window is above the gate, 106.578 s.**

- **Addition-driven 3-attempt rebuilds.** 28 of 73 builds take 3 attempts at 0.563 s median against 0.394 s for 2; proxy **5.422 s**. Candidate fix: seed the adjacency-cache neighbours of every depth-1 addition as depth-2 targets, via the `expand` helper at `hdsim_3d.cpp:1767-1798`. **Attribution first.** Larger seeds can cross the full-build threshold (`:1986-2003`).
- **Wake-tree fixed cost.** 6.648 s, mean ≈ max. Profile the sub-components (`Simulation.cpp:5847-5862`) before designing persistent tree state.
- **`RadiationStep::synchronizedCellTimeStepLimits`.** Today only `HydroStep` overrides `steps/PhysicsStep.hpp:94`. Implementing it would let the entry anchor use per-cell radiation limits at tier 1 always, remove tier 2's uniform scalar and tier 3's refusal. Promoted out of "contingent" if Stage 0 item 8 shows entries are routinely refused.

---

## 5. Standing gates, constraints, and exact run recipes

### 5.1 Applied to every stage that produces a binary

| Gate | Definition |
|---|---|
| Review | astra xhigh code review, CHANGES_REQUIRED resolved, **before** any build or run |
| Build | `build_rich.sh intelReleaseMPI --test_name=<dir>` |
| Regression | 7/7: `amr_random`, `amr_random_individual`, `fmm_gravity_mpi`, `fmm_gravity_serial`, `individual_box_growth`, `segmented_hilbert_ownership`, `suppressed_exchange_ghosts`, plus that stage's new cases |
| Crash gate | `submit_floorgate.sh ARM=F` completes to `FG_T_END=0.20` with no negative-mass abort |
| Correctness | no negative mass/energy aborts; terminal snapshot vs the global arm, mass-weighted L1 by region, **no worse than the Stage 0 individual arm**, at every span the stage runs |
| Conservation | `RICH_MODE_SWITCH_STATE phase=before/after` sums unchanged to printed precision on every span; interval-integrated face transfers, interrupted half-kicks, AMR remapping, sink removal and radiation defect accounting all exercised |
| **Cadence** | **no off-grid events except signal or radiation wakes** — equivalently `T1 == T0` at every finalization. A level-2 arm (Stage 1b only) states its own expectation and is not a production candidate absent a user scope decision |
| Safety counters | `INDIVIDUAL_BIN_OVERRUN` count and worst ratio, `step_overruns`, no worse than Stage 0 |
| Accuracy | one-sided, whole-arm, **evaluated independently per arm**: uncensored ratio at activation, max and p99, ≤ `g` = 0.5; `coverage ≥ 0.80` with `issued == uncensored + censored + dropped_at_switch + lost`; **zero** censored samples above `g`; case-B (`overdue`) violations reported separately. An arm below coverage, or spanning a process restart, is **inconclusive for that arm**, and **no other arm's pass substitutes for it** — in particular the window arm cannot supply cascade-regime evidence (§0.1), which is what the **C arm** is for |
| **Bound integrity** | at a re-anchor, `new_interval ≤ min(fresh_limit, carried_limit)` for every cell (lengthening within that bound is permitted and expected); elsewhere no cell's interval exceeds its own mass-, thermal- or radiation-limited deadline; every entry anchor names its radiation tier and refuses at tier 3; `INDIVIDUAL_GUARD_FLOOR` activations no more frequent than Stage 0 |
| Timing | window against job 10208630 (122.565 s); t = 21 against job 10213458 (307.3 s); the C arm against its own Stage 0 counterpart |

### 5.2 Exact run recipes

`submit_gfloor.sh` defaults are `GF_T_OUT=20.9015`, `GF_T_END=20.9016` (`:38`); `ARM=B` sets `RICH_ADAPTIVE_PROBE_FRACTION=1000` (`:41`), which pins individual mode; `GF_ENV` is exported **after** the case (`:47`) and therefore overrides. `GF_ENV` syntax is space-separated `RICH_NAME=value` assignments.

- **Window arm (W):** `ARM=B GF_BIN=<binary> GF_TAG=<tag> GF_RESPLIT=0 GF_PROFILE=1 sbatch submit_gfloor.sh` — seeded from `snap_full_54`, the clean regime.
- **Cascade-regime pinned arm (C) — new.** Same recipe with `ARM=B` (so individual mode is pinned and **no mode switch occurs**, making `dropped_at_switch` zero by construction), seeded from the `snap_full_<k>` chosen by **Stage 0 item 9** as the last snapshot before a period exhibiting the halving cascade, with `counter.txt = k`, `gravity.txt = 1`, and `GF_T_OUT`/`GF_T_END` set to span ~80–100 events past that point. This is the arm that carries the accuracy and cascade-removal evidence for the regime Stage 1 actually changes. It is a *diagnostic* arm: it is not in §6's timing budget and has no 1.15× gate of its own.
- **Adaptive-to-t = 21 arm (ADP):** `ARM=B GF_BIN=<binary> GF_TAG=<tag> GF_T_OUT=21.0 GF_T_END=21.0001 GF_ENV="RICH_ADAPTIVE_PROBE_FRACTION=0.1" GF_RESPLIT=0 sbatch submit_gfloor.sh` — matching jobs 10213459/10213497. This arm contains **several mode switches**, which the Stage 0 item 3 drain accounts for; its accuracy verdict is its own and may legitimately come back inconclusive.
- **Global reference arms:** the window and t = 21 recipes with `ARM=G`.
- **Valve experiment arms (Stage 1b only):** add `RICH_INDIVIDUAL_WAKE_CHANGE_DEADLINE=1` to `GF_ENV` on W and C.
- **Strict-bounds arm (Stage 4):** add `RICH_INDIVIDUAL_REANCHOR_STRICT_BOUNDS=1` to `GF_ENV`.
- Every run directory must be seeded (snapshot, `counter.txt`, `gravity.txt=1`; checked at `:27`), in a new directory under `/data/users/elads/${GF_TAG}_${ARM}`. Each arm therefore begins at its own process restart, which is why §5.1's accuracy gate is conclusive within each arm's span.

### 5.3 Hard constraints

- **Never move the production symlink.**
- **Never touch other users' or long-run data directories.** New directories only, under `/data/users/elads/${GF_TAG}_${ARM}`.
- **`runs/` outputs are never deleted.**
- **Every code change gets an astra xhigh code review before runs** — including the Stage 2 one-liner and §1.3.1's wire-struct extension.
- **Node prescan before every submit** — already in the scripts (`:51-53`). Orphan processes have produced uniform 10–60× slowdowns; a timing result from an unscanned node set is not evidence.
- **Performance policies are measured runtime criteria that switch both ways** (Stage 4's anchor, Stage 5b's host choice, Stage 6's ledger). Accuracy thresholds (`f`, `c`) are fixed configuration with measured defaults.
- **Every collective gate uses `MPI_Allreduce`-based agreement** before any rank branches. The new finalization call sits outside `#ifdef RICH_MPI`; its collectives must be self-guarded.
- **No per-cell vector measured in one event is indexed in another.** Physics bounds that must survive travel inside `CellTimeState`, and where a numeric bound must cross ranks it gets an explicit wire field (§1.3.1); everything else crosses only as collective scalars.
- **Availability is a precondition, never an approximation.** Where a required physics bound is missing, the optimisation **defers** (re-anchor) or **refuses** (entry anchor) and says why. No arbitrary safety factor stands in for an unknown bound.
- **Measurement loss is counted, never declared away**, and **evidence is not transferable between regimes**: an arm's verdict applies to that arm's regime only.
- **A clean diff, a passing review, or exit 0 is never runtime validation.**

---

## 6. Open item (d): the budget, from one formula

Savings are composed sequentially, not added: Stage 4 removes whole events **at their full cost**, including the mesh and AMR work Stages 3 and 5 reduce. Stage 1b is default-off and contributes nothing; the C arm is diagnostic and is not in the budget.

**Decomposition.** `f` = per-event cost that does not scale with the active count; `V = 127.431 − 83·f`. `f` is **measured offline in Stage 0** by regressing `step_s` on `active_cells` over the 83 `RICH_STEP mode=individual` records of job 10213543. Provisional bracket `f ∈ [0.54, 0.90] s`, central 0.70.

**`V` is approximately conserved under a re-anchor.** Raising the anchor by `a` raises every bin width by `a`, but cells whose limit sits just above a bin boundary drop a bin and are updated more often; over a broad limit distribution the realized bin width is log-uniform in `[L/2, L]` and its expectation is unaffected by shifting the anchor. What the anchor changes is `w_min`, hence the event count, which multiplies `f`. The mechanism by which individual intervals lengthen — recovering quantization loss while respecting every physics bound — is §2.4.1's soundness invariant, not a violation of it.

**The single formula.** With `N' = 83/a`, `δ3 = 5/83 = 0.060` s/event, `δ5a ∈ [0, 0.027]`, central 0.014, `δ5b = 0.35` s/AMR pass, AMR passes `= N'/10`:

```
tot(a, f) = N'·(f − δ3 − δ5a) + V − (N'/10)·δ5b
          = 127.431 − 83·f + (83/a)·(f − δ),        δ = 0.109
```

| realized anchor `a` | `f = 0.54` | `f = 0.70` (central) | `f = 0.90` |
|---|---|---|---|
| 1.00 (Stage 4 delivers nothing) | 118.38 s | 118.38 s | 118.38 s |
| 1.20 | 112.42 s | 110.21 s | 107.44 s |
| 1.32 | 109.71 s | **106.49 s** ✓ | 102.47 s |
| 1.43 | 107.63 s | 103.63 s | 98.64 s |
| 1.61 | 104.83 s ✓ | 99.80 s ✓ | 93.51 s ✓ |

At `a = 1` the result is **independent of `f`**: `f` enters only through the event-count change, so with the count fixed the saving is `83·(δ3 + δ5a) + 8.3·δ5b = 9.05 s`. A check on the model, not an artefact.

**Not modelled, charged at measurement time:** the forced full source sweeps a re-anchor triggers (§2.4.4), re-anchor opportunities lost to §2.4.1's deferrals, and entry anchors refused at §2.2 tier 3. Stage 0 item 8 measures the sweep cost, the deferral frequency and the radiation-tier distribution; Stage 4's ≥ 6 s saving is required **net** of all three. A high deferral or refusal rate shows up as a realized `a` below the table's assumption, not as a hidden cost.

**Pass thresholds**, `a* = 83(f − 0.109)/(83f − 20.853)`: **1.493** (`f = 0.54`), **1.317** (`f = 0.70`), **1.219** (`f = 0.90`).

**Conclusion.** Stage 4 carries the gate. At the central `f` the gate is met from `a ≳ 1.32`; at the pessimistic `f` it needs `a ≳ 1.49`; if Stage 4 delivers nothing, Stages 1–3 and 5 reach **118.4 s (1.035×)** and **this plan does not reach 1.15×** — that outcome ends the plan and returns a decision to the user. Stage 7's trigger is the gate itself.

**Adaptive-to-t = 21, stated separately and never added.** Current best 342.0 s against 307.3 s global; astra's ideal model gives 306.4 s as an upper bound. Warm-up plus the first individual period is 88.6 s and unchanged; Stage 1 takes the later periods from 60.6 s toward 35–45 s; Stages 3–5 take ~8–12 % off individual wall time; Stage 4 removes the bin-29 cliff.

- **Stage gate: ≤ 300 s.** **Campaign goal: ≤ 267 s (1.15×)**, reachable only if the window gate passes first. Stage 6 bounds the loss if it does not.

---

## 7. Risks and open questions

- **RISK.** Nothing here is runtime-validated. Every saving is a projection from logged buckets.
- **RISK.** Level 1 is later than today's behaviour in exactly one non-bug case: the event after an off-grid signal or radiation wake. Stage 0 item 4 measures its frequency; the ratio-at-activation gate bounds its consequence.
- **RISK.** The level-2 valve's bound holds only for `f < r ≤ g`, and then only to within `1 + 1/E`; for `r > g` the violation is unavoidable and is reported, not prevented. Its event cost is not bounded a priori.
- **RISK / measurement design.** The benchmark window does **not** exercise the cascade regime (§0.1), so it can never certify the wake fix's accuracy there. The **C arm** exists solely to supply that evidence, and if it cannot be seeded (no suitable snapshot, or its own coverage below 0.80), Stage 1's accuracy claim for the cascade regime is unproven and the ID-keyed fallback accumulator becomes mandatory. Nothing in this plan lets one arm's pass stand in for another's.
- **OPEN / possible defect.** `suggestIndividualWakeDeadlines` early-returns when `individual_conserved_change_.size() != extensive_.size()` (`hdsim_3d.cpp:3855-3857`), and `ResetIndividualMeshState` clears that array after **every** individual AMR pass and **every** load balance (`hdsim_3d.hpp:215`; `steps/HydroStep.cpp:89`, `:126`). The conserved-change guard therefore does not fire at all on the event following each. Stage 0 item 5 counts it; the fix is out of scope and filed separately.
- **RISK.** The new `CellTimeState` fields touch ten sites plus two serializer functions, and Stage 4 extends the serializer a second time **and** a wire struct (`AMRPendingBinResponse`). A missed site silently drops a request, un-sticks the eligibility bit, loses provenance, or — worst — loses a physics bound.
- **RISK.** `next_interval_limit` is the only per-cell *physics* datum this plan moves through AMR, migration and an MPI response. A refined child or merge recipient carries a second-hand accumulated bound, which is why bit 2 is set unconditionally on both and why strict mode exists; the strict arm measures what the relaxation is worth. If it shows a material difference, strict becomes the default.
- **RISK.** §1.3.1's wire extension changes a struct exchanged with `MPI_Exchange_all_to_all` inside the AMR closure. A mismatched `dump`/`load` order corrupts every merge silently; the cross-rank donor test is the guard, and the fallback (mark remote recipients `+inf` and defer) is the safe degradation if review rejects the wire change.
- **RISK.** A re-anchor legitimately **lengthens** intervals — that is its purpose — so it changes physics trajectories more than a pure re-labelling would. Terminal L1 at every span and the bound-integrity gate are the checks; the soundness invariant guarantees only that no interval exceeds a physics bound, not that intervals are unchanged.
- **RISK.** The entry anchor's tier 2 applies one global radiation scalar to every cell. That is the production fallback semantics, but it is uniform where the truth is per-cell, so it can be much tighter than necessary for most cells. Safe in the direction that matters; Stage 7 removes it.
- **RISK.** The coverage denominator counts 0→1 transitions of the eligibility bit. If a future change sets that bit elsewhere without incrementing the counter, `lost` goes negative — asserted in test 14 and as a debug assertion in code.
- **RISK.** `dropped_at_switch` makes mode-switch loss visible but does not make it small. An adaptive arm with many short excursions can be coverage-limited and stay inconclusive; that is an accepted, reported outcome, not a failure to be papered over.
- **RISK.** A **process restart** remains the one measurement boundary under option (A). Every arm begins at one; a future measurement resuming mid-campaign needs option (B).
- **RISK.** `reanchor()` is the only operation that moves `begin_tick` while a half-kick may be owed. Its safety rests on the origin being reset in the same operation (§2.4.3(b)).
- **RISK.** The re-anchor's `v = 0` branch trusts `cached_acceleration == Vector3D()` as the signature of a cleared cache — a heuristic, documented as one, that throws rather than proceeding when it does not hold.
- **RISK.** The re-anchor forces a full source sweep on the first following event, charged to §2.3 and required to be recovered inside Stage 4's gate. Fallback if sweeps dominate: a `sweep_credit_ticks_` field carrying the elapsed age in new tick units.
- **RISK.** Stage 6 requires two genuine controller changes. Change 2 is new policy and could thrash around frequent box growths; test 6 covers the mechanism, the t = 21 switch count the behaviour, and the kill criterion drops Change 2 alone.
- **RISK.** The finalization call sits between the rebalance and the next-tick read, compiled in both serial and MPI builds; its collectives must be self-guarded, and the end-tick checksum must be debug-only.
- **RISK.** Stage 2 changes activation schedules, and therefore physics, on cross-rank faces. Frequency and effect UNVERIFIED until Stage 0's counters run. Stage 2b's restore path is exercised for the first time by the C arm, which restarts into a live individual state.
- **RISK.** Stage 5b assumes a full event mesh at the committed generators is what AMR needs; identity indexing does not by itself prove geometric freshness.
- **RISK.** Box growth changes the gravity and mesh cost profile ~10× and invalidates every measured criterion here; all must reset on `NotifyDomainChanged()` (`Simulation.cpp:2579-2599`).
- **OPEN.** Whether the residual non-power-of-two spacings in job 10213454 are fully explained once §1.2 lands is UNVERIFIED; `INDIVIDUAL_EVENT_DEADLINE` on the C arm answers it in one run.
- **OPEN.** Controller behaviour after Stages 1–4 is not predictable from these logs and determines the t = 21 total.
- **OPEN.** Whether the 36 depth-only rebuilds are caused by segmented ownership specifically or by the changed workload generally is UNVERIFIED.
- **OPEN.** `clampToTerminalTick` (`IndividualTimeStep.cpp:658`, declared `:322`) has no callers. It is kept consistent with the provenance rules but should be wired up or removed; filed separately.

---

## Assumptions

1. **The shipped wake fix is level 1 only** — the wake ends at the next scheduled event, bound `T − event_tick ≤ 2^minimum_occupied_bin_`. The level-2 valve is conditional Stage 1b, default-off.
2. `c` (default 2, `g = 0.5`) is an accuracy target. The valve's bound is `ratio ≤ g·(1 + 1/E)` and applies **only** to `f < r ≤ g`; `r > g` is an unavoidable violation reported as `change_wake_overdue`; the projection is computed in signed arithmetic and then clamped.
3. The bound in (1) holds only after Stage 2 restores the allowance invariant on the remote closure path, and only up to cells in recorded overrun. Stages 1 and 2 ship in one binary.
4. The new fields live in `CellTimeState` because `addMigrationBuffer` is unavailable (`Simulation.cpp:2674`) and box growth is in scope.
5. **A boolean derived from the fact of a merge needs no transport; a numeric bound does.** Bits 0, 1 and 2 are set on a recipient with no wire change — **bit 2 unconditionally**, because a merged cell's accumulated bound was measured for neither input — so the donor's flag is never consulted and only the `double next_interval_limit` crosses in `AMRPendingBinResponse`. If that wire change is refused, remote-merge recipients are marked `+inf` and the next re-anchor defers.
6. Restart I/O follows option (A) — the new fields are not persisted, no version bump, clear-at-write enforced by a throw. A **process restart** is the only measurement boundary; option (B) is the route to cross-restart continuity.
7. Finalization is inserted after `Simulation.cpp:6259` and before `:6260`, reusing the existing `MPI_MIN` at `:6261-6264` as `T1`; it adds one Allreduce for `T0`.
8. Creation is `T1 < T0` with `T0` the minimum over **all** endpoints before finalization, independent of provenance.
9. **Measurement loss at a mode switch is counted, not declared away.** `drainPendingWakeSamples` counts-and-clears outstanding eligibility at `adaptiveEnterGlobal`, `adaptiveEnterIndividual` (before `:3339`) and `initialize`; the counters live in `HDSim3D`, survive `ResetIndividualMeshState`, and the identity `issued == uncensored + censored + dropped_at_switch + lost` and the coverage gate are **whole-arm**, with a per-excursion breakdown.
10. Eligibility (a sticky bit) and history validity (a globally unique `(rank, epoch)` token) are separate, so a migrated stamp can never match its destination's.
11. The accuracy gate is **one-sided and per-arm**: censored samples above `g` count as failures, never as passes; `dropped_at_switch` and `lost` count only against coverage; and **no arm's pass substitutes for another's inconclusive verdict**. The window arm cannot certify the cascade regime (§0.1); the **C arm** — pinned individual, seeded inside a cascaded period, therefore switch-free — is what does.
12. **Endpoint provenance is a property of the endpoint**, written at every `end_tick` assignment and never blanket-cleared, and its companion tick is the **tick of the writing operation**, because `commitEvent` advances `current_tick_` only at `:668` / `:528`.
13. The re-anchor is a dedicated in-place `reanchor()` requiring a **uniform** gravity half-kick phase; `v = 1` is the ordinary post-event state, `v = 0` the legitimate post-box-growth state, which additionally requires every `cached_acceleration == Vector3D()`. It resets `time_origin_` in the same operation.
14. **The re-anchor's guarantee is bound soundness, not interval preservation.** For every cell, `new_interval = ticksForBin(time_bin) · time_quantum_ ≤ min(fresh_limits[i], next_interval_limit[i])`, which holds because `quantizeTimeStep` returns the largest bin whose width does not exceed its argument. An interval may legitimately become **longer** than the one the closing event assigned — recovering quantization loss is the stage's purpose and §6's mechanism — and a test asserting non-lengthening would be wrong. `next_interval_limit` is the very vector the closing event's `chooseNextBin` consumed, so it always carries a radiation term and `cell_limit_fallbacks` is a cross-check at re-anchor rather than a defer trigger. Where any bound is missing the re-anchor **defers** with a logged reason.
15. **The entry anchor has no predecessor interval**, so it obtains its radiation bound by the three-tier rule: per-cell cache when complete, `RadiationStep::suggestTimeStep()` when not, and **refusal of the new anchor in favour of the ramped path** when neither exists. No arbitrary safety factor substitutes for a missing bound.
16. **The re-anchor forces the next full source sweep** rather than pretending one occurred, and the sweep's measured cost is charged to §2.3's criterion.
17. Stage 6 requires **two explicit controller changes** — backoff on the abort path, and halving `dwellMultiplier` in `NotifyDomainChanged` — neither of which exists today; both are tested through `adaptiveDecide`.
18. `safety_entry`, the trailing-window length, the 0.80 coverage floor and Stage 5a's 1.5 s build gate are HYPOTHESIS/tunable. `margin = 1.15`, `cooldown_events = 8`, `amortization_factor = 2.0`, `dwell_backoff_cap = 16` are reused.
19. Stage 5a retains the full neighbour-bin closure on every event unconditionally.
20. `IndividualMeshBuildPolicy::FullReference` forces a full event-mesh build and does not enable partial-build parity verification; confirmed at review.
21. Job 10208630 (122.565 s) is the sole global comparand; the gate is 106.578 s. The C arm is diagnostic and carries no timing gate.
22. `f` and the active-count scaling are measured offline from job 10213543's `RICH_STEP` records before §6's budget is relied on.
23. All runs use §5.2's recipes on `bigrun` with `ib&d25g`, 16 nodes × 16 ranks, `RICH_FMM_GRAVITY_RESPLIT=0`, with the node prescan the scripts perform.
24. Scheduler sources live under `source/newtonian/three_dimensional/simulation/`; every line number in this document was re-verified against the worktree on 2026-09-27.

