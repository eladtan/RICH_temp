# RICH individual timesteps: convergence failures and speed — agreed plan (revision 2026-09-22, hardened, round 4 + post-loop reconciliation)

**Post-loop reconciliation (author, 2026-09-22 14:45, outside the read-only loop):** the loop reached its
round cap after the round-4 revision below was written but before it was reviewed. Three things were
changed after the cap, each because a measurement or a source check the read-only agents could not make
was made: (1) §0's global wall column was re-summed from the raw log with a script (1626.997066 s over
256 records, 3538.74 s/unit t, reviewer NIT confirmed) and the origin of the stale 3561 identified;
(2) S7 gate (b) was rewritten because the round-4 revision anchored it on the `Einit`/`Efinal` print in
`MultigroupDiffusion.cpp`, but the TDE runs use the grey `Diffusion` class
(`runs/BaseTDEComptonIndividual/test.cpp:26,1542`), which has no such print: zero `Einit` hits in
`source/Radiation/Diffusion.cpp` and in both the stdout and the stderr logs of arm D and the global
run (the gray-vs-MG naming trap of the run directory names); the gate now rests on a new rank-0
conserved-sum record at the switch; (3) S3's active-Hilbert-integrated variant (Correction 2(ii)) is
now designed as S3b with its own switch and gate, because production keeps active-Hilbert balancing
on and a remedy that only works with it off would not reach production. After the first direct review
(codex_review_2.md, one MAJOR, one NIT): (4) S3b now forces fresh boundary construction when the trigger
fires, bypassing the current mask's cached cut, and treats an unchanged fresh cut as a gate failure;
(5) S7 gate (c) compares with a rounding slack matched to the six-digit decision records and raises
their precision. Nothing else was changed.

Status: revises `docs/fixes/individual-timesteps-convergence-speed-plan-2026-09-22.md` (Codex APPROVE,
2026-09-22 13:3x) without changing its structure or removing any agreed item. **This round (4)**
fixes the two MAJORs and one NIT from `review-4.md`:

1. **[MAJOR] S3's `forceRebalance`-threading fix (added last round) is unreachable under the
   configuration the plan's own data comes from.** `request_balance`/`automatic_balance` are also
   gated on `!active_hilbert_balance` (`Simulation.cpp:3931,3935`), and arm D's own log
   (`meshaberr_noretrybin_10199568.txt:5`) shows `active_hilbert_cache=1 ... requested=0` — active
   Hilbert balancing (on by default, `Simulation.cpp:1288-1290`) was live and silenced the fixed
   trigger regardless. Fixed below by adding a second standing precondition (disable active-Hilbert
   caching, enable auto-rebalance, for S3's probe/rollout only) and by explicitly flagging the
   active-Hilbert-integrated variant as unimplemented follow-on work rather than solving it by
   assertion.
2. **[MAJOR] S7 gates (b)-(c) asserted continuity/threshold-compliance from code reading instead of
   a measured run.** Fixed below: gate (b) is rewritten to use a new rank-0 `RICH_MODE_SWITCH_STATE`
   record (owned-cell count and conserved sums emitted before and after each switch, a
   no-behaviour-change instrumentation prerequisite) with fixed numeric tolerances, a zero-invalid-state
   requirement, and an adaptive-off twin comparison (post-loop: the round-4 text had anchored this on
   the `MultigroupDiffusion.cpp` `Einit`/`Efinal` print, which the grey TDE path never emits); a new gate (c) checks
   every `RICH_MODE_DECISION` record's own printed `tau_*`/`gain_bound` fields against the configured
   1.15 margin / 1.5 gain threshold directly, rather than trusting a single carried-forward example.
3. **[NIT] §0's global wall-time normalization was arithmetically off.** Reviewer's recomputation
   (1626.997066 s summed over 256 `RICH_STEP.step_s` values / 0.459767687645476 span = 3538.74 s/unit
   t, not 3561) is adopted and was re-summed independently after the loop (see the note under the table).

Worktree `/home/elads/RICH-ablation-integration`, branch `codex/individual-timesteps`, uncommitted.
All of R1, R2, R3's env var, S3-S6's proposed switches were re-confirmed absent from the current
source (`RICH_INDIVIDUAL_RADIATION_DT_VOLUME_CAP`, `RICH_INDIVIDUAL_REBALANCE_FULL_BUILD_SECONDS`,
`RICH_INDIVIDUAL_MESH_SKIP_MANAGER_UPDATE`, `RICH_INDIVIDUAL_SPARSE_BIN_CLOSURE`,
`RICH_INDIVIDUAL_WAKE_TREE_CACHE`, `RICH_FMM_RESET_FACTOR`, `RICH_FMM_INDIVIDUAL_TARGETS_ONLY` — zero
grep hits, the last confirmed again this round against the full `RICH_FMM_*` name list in source:
`RICH_FMM_GEOM_LOG`, `RICH_FMM_NONUNIFORM_DIAGNOSTICS`, `RICH_FMM_PROGRESS_INTERVAL`,
`RICH_FMM_STRUCTURAL_INTERVAL`, `RICH_FMM_TRACE`), so all remedies below are still genuinely
unimplemented proposals, and `RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS` still defaults to **on**
(`RadiationStep.cpp:339`, `local_enabled = true`). `RICH_INDIVIDUAL_ADAPTIVE_MODE`, by contrast, is
**not** a new proposed switch — it already exists in source (`Simulation.cpp:2047-2051`) and already
governs a shipped, if untested, controller (S7 below); this round's fix uses it as an override, not
as a new implementation item. `RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE` and `RICH_INDIVIDUAL_AUTO_REBALANCE`
are likewise pre-existing switches (`Simulation.cpp:1288-1290,1219-1220`), not new proposals; this
round's fix to S3 uses them as a required probe configuration, not as new implementation items.

## 0. Measured inputs the draft flagged as missing ("step 0"), now done

Exact simulated-time spans from `RICH_STEP` `t_start`/`t_end`; retries = `RICH_RETRY` lines, one
per rejected candidate attempt (`attempt=N`), so 79 (earlier quote) was the count in the common
window t <= 0.4191 and 137 is the whole file; both definitions are consistent.

| run | span | events or steps per unit t | retries per unit t | negative/invalid energy | wall s per unit t |
|---|---|---|---|---|---|
| A `meshab_on_10199440.txt` (retries lower bins, default) | 0.375268-0.419568 = 0.0443 | 9458 | 10451 | 0 | 53032 |
| D `meshab_noretrybin_10199568.txt` (`RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS=0`) | 0.375268-0.440268 = 0.0650 | 1938 | 2108 | 0 | 36205 |
| global `output_10199442.txt` | 0.000886-0.460654 = 0.4598 | 557 | 2229 | 4 | ~3539 (corrected this round, see note below) |

**Wall-time figure corrected this round (reviewer NIT, review-4) and re-summed after the loop:** summing
the `step_s` field of all 256 `RICH_STEP` blocks in `output_10199442.txt` gives 1626.997066 s (the
reviewer's figure exactly); over the file's span 0.459767687645476 that is **3538.74 s/unit t**. The
stale 3561 came from an earlier read of the same log while the job was still running: 1606 s over the
first 252 steps to t ~ 0.452 (1606/0.451 = 3561). The other two rows were re-summed the same way and
are unchanged: arm A 2349.314 s / 0.0443 = 53032, arm D 2353.296 s / 0.0650 = 36205. All three
columns are therefore the same quantity, summed `RICH_STEP.step_s` per unit simulated time; Slurm
elapsed for the global job was 1901 s (00:31:41), the difference being startup and restart reading.

**Re-verified this round, directly from the raw logs, not carried forward on trust:**
`meshab_on_10199440.txt` has exactly 419 `RICH_STEP mode=individual cycle=` records (not
`RICH_STEP_DETAIL`, which shares the `RICH_STEP` prefix and would inflate a naive grep), first
`t_start=0.375267984618`, last `t_end=0.419567984618` — span 0.044300, 463 `RICH_RETRY` lines
(463/0.0443 = 10451.5, matches the table). `meshab_noretrybin_10199568.txt` has exactly 126 such
records, first `t_start=0.375267984618`, last `t_end=0.440267984618` — span 0.065000, 137
`RICH_RETRY` lines (137/0.0650 = 2107.7, matches). `output_10199442.txt`: 256 `RICH_STEP
mode=global` records, 1025 `RICH_RETRY` lines (256/0.4598 = 556.8, 1025/0.4598 = 2229.2, both
match). The log record format is multi-line per event (`RICH_STEP mode=... cycle=N` header
followed by `time|work|phases|mesh|source` continuation lines carrying `t_start=`/`t_end=`), which
a single-line grep for `t_start=` on the header line alone will miss — worth knowing before
re-deriving these numbers.

**Re-confirmed this round (new): arms A and D are unaffected by S7's adaptive-mode controller.**
`submit_mesh_ab.sh:65` sets `RICH_BINARY="${RICH_BINARY:-./rich_mesh_seed_20260921}"`; that binary's
own provenance record (`rich_mesh_seed_20260921.provenance.txt`) shows it was built 2026-09-21 with
`--test_name=BaseTDEComptonIndividual`, one day before `SetAdaptiveIntegrationMode` was added to
that test file — confirmed by comparison against `rich_adaptive_20260922.provenance.txt`, the first
binary in this run directory whose description records "adaptive integration mode controller". Arms
A/D's `RICH_STEP mode=individual` records and retry counts above therefore come from a build with no
adaptive-mode call in it; no number in this table needed revision.

**Corrected this round (reviewer MAJOR from review-1 — matched-window retry rate and terminal-record
evidence; count refined this round, reviewer NIT from review-2 — see below):** the "D at 0.95x
global" comparison in the original draft used global's *whole-file* average (1025 retries / 0.4598 =
2229/unit t), a different physical window (t=0.0009 to 0.4607) than arm D's (t=0.3753 to 0.4403,
which is post-pericentre only). Re-extracted directly from `output_10199442.txt`, using the exact
line range bounding the steps whose span overlaps D's window: `t_start=0.376255357089` at cycle
1535 (line 6015 header, `t_start` on the line-6016 continuation) through the last step ending
inside D's window, cycle 1563's `t_end=0.438462600858` (line 6728) — 29 complete global steps
(cycle 1535 through 1563 inclusive), span 0.438462600858 - 0.376255357089 = 0.062207243769.

**Retry count, corrected this round (reviewer NIT — 119, not 118):** a naive `sed -n
'6016,6749p' | grep -c RICH_RETRY` line-range count returns 118, but that range starts one line
into cycle 1535's own `RICH_STEP` record (line 6016 is the `time |` continuation line, not the
`RICH_STEP mode=global cycle=1535` header at line 6015) and so excludes cycle 1535's own 4
`RICH_RETRY` lines, which sit at lines <6016 before the header — those retries were part of
reaching cycle 1535, the window's first included step, and belong in the matched-window count.
Symmetrically, the line range runs through line 6749 (cycle 1564's `time |` continuation, at the
exact `t_start=0.438462600858` boundary), which is past cycle 1563's own record (lines 6727-6728,
confirmed the boundary step) and so *includes* cycle 1564's 3 `RICH_RETRY` lines (lines 6736,
6740, 6744 — all read directly this round, all `RICH_RETRY mode=global cycle=1564`), even though
cycle 1564 is not one of the 29 matched-window steps (its own `t_end=0.440664010397` is past D's
window). Net: -4 (wrongly excluded) +3 (wrongly included) = -1, i.e. the line-range method
undercounts by exactly one, 118 instead of 119. Matching retries to cycle number against the 29
completed steps (cycles 1535-1563 inclusive) gives **119** `RICH_RETRY` lines, i.e.
119/0.062207243769 = **1912.96 ≈ 1913/unit t** — re-derived this round by reading
`output_10199442.txt` at offsets 6000-6039 and 6690-6760 directly (file-reading tool, not a shell
command), which confirms both halves of the boundary mechanism above against the actual log
content, not just the reviewer's arithmetic claim. This is still *lower* than D's 2108/unit t, not
higher: in the matched window, global's retry rate is below D's, the opposite of what the
mismatched whole-file comparison (2108 <= 2229) implied. The whole-file global average (2229)
overstates the matched-window rate (1913) by ~17%, so neither the original comparison nor a
same-magnitude assumption is safe — only the matched-window number is valid, and by that number
**D's retry rate exceeds global's**. The pending-gate conclusion (§2 R2) is unchanged by this
correction.

**Also corrected (reviewer MAJOR from review-1 — no terminal completion record, unchanged this
round):** arm D's job did not complete. `meshaberr_noretrybin_10199568.txt` ends with
`[2026-09-22T01:12:34.005] error: *** JOB 10199568 ON d26g1 CANCELLED AT 2026-09-22T01:12:34 DUE TO
TIME LIMIT ***`, immediately after an `INDIVIDUAL_ACTIVE_HILBERT_DECISION cycle=10674` line — i.e.
the job was killed by the scheduler mid-step, not stopped at a clean checkpoint. `grep -c "Done
sim"` on `meshab_noretrybin_10199568.txt` returns **0**. By contrast `output_10199442.txt` (global)
ends `Starting writing final file [...] snap_24.h5 at time 0.460654` / `Done sim`, and `grep -c
"Done sim"` on it returns **1**. The plan's "same endpoint reached, exit 0" claim for arm D is
false and is withdrawn; see §2 R2 for the consequence.

Full-file MadVoro phase distribution (`meshab_detail_10199448.txt`, 264 partial + 16 full builds —
re-verified: the file contains exactly 280 `Individual full Voronoi build`/`Individual partial
Voronoi build` `DisplayTime` records, 16 + 264 = 280, exact):

| phase | median | p90 | note |
|---|---|---|---|
| bringing ghosts | 0.47 s | 1.29 s | max 40 s inside full builds |
| preparing (incl. points-manager "exchange" 0.305 s) | 0.31 s | 0.33 s | fixed, independent of target |
| build Voronoi from Delaunay | 0.0085 s | 0.021 s | |
| initial build (Delaunay of targets) | ~0 | 0.0005 s | |
| partial build total | 0.754 s | 1.15 s | 0.54 s median after the warm-target fix (arm E `meshab_nowarm_10199573.txt`) |
| full build total | 25.9 s | | 16 in 221 events |

Full-build events: A 30 of 419 (676 s = 61% of its 1105 s mesh); D 36 of 126 (862 s = 37% of
wall). **Open measurement, not re-derivable this round:** unlike the 16/264 split above (which
comes from the explicit `Individual full/partial Voronoi build` `DisplayTime` records that only
exist in `meshab_detail_10199448.txt`, gated by `RuntimeLogDetailed()`), arms A and D
(`meshab_on_10199440.txt`, `meshab_noretrybin_10199568.txt`) carry no per-build full/partial
marker at all — grepping those two files for `Individual full Voronoi build` returns zero hits.
Thresholding on `mesh_s` in those files (mesh_s > 10 s) finds 31 events for arm D, close to but
not exactly the stated 36, because `mesh_s` sums every build in an event and an event can mix a
full build with partial builds, so a magnitude threshold both over- and under-counts. The 30/419
and 36/126 figures are plausible (they are bracketed by the mesh_s-threshold estimate and the
per-event `mesh_builds` field) but not independently reproducible from the cited files with a
single grep; treat them as approximate until S3's instrumentation (below) adds an explicit
`full_builds=` counter to `RICH_STEP` so this stops requiring inference. This does not change any
gate or remedy — S3's own gate is a direct `mesh_s` measurement, not this count — but the count
should not be quoted elsewhere as exact.

Whole-mesh closure threshold (arm C `meshab_thr_10199567.txt`) was worse: 366 s vs 146 s over the
same 24 events, because a partial build whose target covers most of one rank costs 8-28 s (events
10564-10572); raising the per-rank fraction moves in the same direction. **Re-verified**: this
366 s/146 s/24-event comparison is not just a log-derived number — it is transcribed nearly
verbatim in a source comment at `hdsim_3d.cpp:1520-1524` ("Judging the fraction on the whole mesh
instead was measured worse (job 10199567 against 10199440: 366 s against 146 s over the same 24
events)"), so the reasoning behind `RICH_INDIVIDUAL_PARTIAL_THRESHOLD_GLOBAL` defaulting to the
per-rank variant is already documented at the call site, not just in this plan.

Periodic full-source sweep: 223 in 11319 events at seconds_max 0.06 s (`RICH_FULL_SOURCE_SWEEP`
records, `output_10199059.txt`): not the scheduler residual. FMM (throttled probe
`fmm_10199396.txt`, restart): `total_mean` median 0.171 s per solve, 51 rebuilds in 2117 solves
(2.4%); the rebuild-conditioned topology phase reaches 0.87 s (`topology_max`), while the
overall `total_mean` maximum is 1.06 s -> rebuild share ~12% of FMM time; the evolved long run pays 1.2 s per
event (handoff: 0.079 s from a snapshot vs 2.085 s evolved to the same state). **Re-verified with
a correction to the derivation, not the number:** the file has exactly 2117 `fmm_solve_trace`
records (confirmed). "51 rebuilds" does **not** mean 51 full process/LET topology rebuilds — only
call=1 (the initial restart solve) has `process_rebuilt=1`/`root_change_ranks=256`; every other
solve in the file has `process_rebuilt=0`. The 51 figure is `leaf_change_ranks != 0`: exactly 51
of the 2117 records have a nonzero leaf-structural-change count (2117 - 2066 = 51, matching the
2.4% claim exactly), i.e. 51 windowed **leaf-capacity** changes (splits/merges batched by
`minSolvesBetweenStructuralChanges = 32`, `FmmDistributedOptions.hpp:50`), not 51 full topology
rebuilds. This matters for S6: the proposed forced-rebuild policy (`RICH_FMM_RESET_FACTOR`) would
trigger something closer to the single `process_rebuilt=1` event's cost (topology_max 0.778 s at
call=1) rather than the cheaper per-solve leaf-change cost, so S6's risk section below is updated
to say this explicitly. The `topology_max ≈ 0.87 s` / `total_mean max ≈ 1.06 s` pair itself is
unaffected by this correction and matches Codex's round-1 NIT (0.87 s is the rebuild-conditioned
value, not the file's overall maximum) — kept as a standing clarification, still non-gating. Also
re-confirmed this round: `fmm_solve_trace` (full record read at `fmm_10199396.txt:284`, call=1) has
no field recording active-target count relative to source count (no `targets=`/`active_targets=`;
only `*_sum`/`*_max` interaction-count aggregates such as `let_active_m2p_sum`) — relevant to S6b's
targets-only item below.

## 1. Convergence: ranked root causes (as agreed in the loop; unchanged this round — not flagged by review-1, review-2, review-3, or review-4)

1A PROVEN. `Diffusion::calculateIndividualTimeSteps` (`source/Radiation/Diffusion.cpp:474-489`,
re-verified: `difference` at line 479, `suggested_dt` with the fixed `nominal_dt * 2.0` cap at
line 487-490) limits dt by a relative Er change whose denominator carries the global floor
`0.02*max_Er`; a near-vacuum cell has `difference -> 0`, `suggested_dt` saturates at
`nominal_dt*2` every event, and `chooseNextBin` (`IndividualTimeStep.cpp:1162`) grows it one bin
per activation to the cap. Global uses the same formula but reduces to one scalar dt set by
well-coupled cells (`Diffusion.cpp:265-292`, re-verified: single `max_diff`/`max_loc` reduced by
one `MPI_MAXLOC`, so exactly one cell's `diff` sets the global dt), so vacuum cells never run at
their own dt. The rejection then comes from `AssessAndApplyHistoricalMGPositiveFloor`
(`conj_grad_solve.hpp:336-411, 565-581`, re-verified: function starts at line 336 exactly): the
single-cell limit compares injected energy EXTENT to the peak cell's extent (1e-7), so a cell
55x the peak cell's volume fails at a 2.3e-9 density undershoot (record for cell 32711 in
`output_10199059.txt`; emitted key `historical_positive_floor_single_cell_limit`, pinned at
`conj_grad_solve.hpp:1436-1439` — the switch-case that assigns this string is at lines 1435-1438
(`SingleCellInjectedEnergyLimit` -> `"historical_positive_floor_single_cell_limit"` at 1436-1438)).

1B PROVEN (mechanism and the normalized rate). A rejected candidate is sub-cycled inside the
event (`RadiationStep::stepIndividual`, `RadiationStep.cpp:613`, sub-cycle loop at lines 735-875)
and, with the default switch, its fraction is folded into the next hydro bin of the affected
cells (lines 888-904 — re-verified again this round, see §2 R2 below for the precise semantics of
this fold); `limitNeighborBins` (`IndividualTimeStep.cpp:1188`) cascades the lowered bin. Global
halves `dt_try` inside the step and leaves the hydro dt alone (`RadiationStep::step`,
`RadiationStep.cpp:447-591`). Measured effect of the fold: table above (now with the matched-window
correction in §0, retry count corrected to 119/1913 per unit t this round). The switch itself is
read at `RadiationStep.cpp:333-370` (function `radiationRetryLimitsBins()`, re-verified exactly),
and the "single call site, one place folds bins" claim holds: `radiationRetryLimitsBins()` is
called exactly once in the file, at line 888.

1C mechanism PROVEN, current rate NOT measured on this worktree: pre-pericentre
`negative or invalid energy {during,after} radiation update` (`Diffusion.cpp:1155, 1289`,
re-verified against the current source, the two `throw`/error-emission sites for these exact
strings), 214 vs 33 in the 2026-09-19 runs; the `RICH_INDIVIDUAL_THERMAL_LOSS_FRACTION` guard
postdates part of that evidence. Diagnostic Run #1 below.

## 2. Remedies, in order (R1 and R3 unchanged this round; R2's numerics claim and gate status corrected in round 3, unchanged this round)

R2 — stop folding rejected radiation fractions into hydro bins: flip the default of
`RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS` to off (`RadiationStep.cpp` — switch at 333-370,
fold at 888-904, both re-verified this round; **default is currently `true`/on** per line 339,
confirming this flip is still live work, not already done).

**Numerics (unchanged this round — reviewer BLOCKER from review-1, resolving the dt-growth conflict
explicitly instead of asserting two incompatible properties):** the fold at 888-904 is a `std::min`
that only ever *shrinks* `local_suggested_dt` below what the physical limiter
(`calculateIndividualTimeSteps`, called just above at 879-882) already assigned; it applies only
to cells that failed a candidate this event (`retry_limiter_requires_all_active` branch) or carry
a remembered `cell_retry_fractions` entry from a prior failed candidate (lines 888-904). With the
switch off, exactly those cells' next suggested dt becomes the physical limiter's value instead of
`min(physical limiter, retry_fraction * cellTimeStep)` — i.e., for exactly the cells this
mechanism targets, R2 **by construction produces a larger next dt than today's default-on code**,
for as long as they keep re-triggering it. This is the intended fix for 1B, not a side effect, so
R2 does **not** satisfy the standing "no dt can become larger than today's code produces"
constraint in the literal per-cell sense — stated plainly here rather than glossed over. What R2
does preserve: the relaxed dt is still bounded by the same physical-limiter formula and the same
`nominal_dt*2.0` growth cap (`Diffusion.cpp:474-489`, untouched) that already governs every other
cell's dt every event; R2 removes a same-cell, history-dependent extra penalty on top of that cap,
it does not raise the cap itself or grant any cell a dt the formula would not already permit an
unretried cell to have. The property actually established is therefore narrower than the general
constraint: *R2 never produces a dt above what `calculateIndividualTimeSteps` already allows for a
cell in that state*, not *R2 never increases any cell's dt relative to today's code*. Safety for
the removed per-cell penalty is an empirical claim resting on the retry-rate/error gate below, not
a by-construction one — and per the correction just below, that gate is not yet met.

**Gate (numbers refined this round — reviewer NIT, conclusion unchanged):** arm D's 2108
retries/unit t (its own window, span 0.0650) was previously compared against global's whole-file
average of 2229/unit t (span 0.4598) — a different, and on this run's evidence more retry-heavy,
physical window, not a like-for-like comparison. Directly re-extracted: global's 29 complete steps
whose span overlaps D's window (t=0.376255357089 to 0.438462600858, span 0.062207243769, cycles
1535-1563 inclusive) carry **119** `RICH_RETRY` lines (matched by cycle number, not line range —
see §0 for the boundary mechanism and the direct re-read that confirms it), i.e. 119/0.062207243769
= **1912.96 ≈ 1913/unit t** — **lower** than D's 2108/unit t. Gate (a) ("retries per unit t <=
global's", matched window) is **not met** by the existing arm-D log and must be marked open, not
passed. Separately, arm D's job (10199568) did not complete: its stderr ends `error: *** JOB
10199568 ON d26g1 CANCELLED AT 2026-09-22T01:12:34 DUE TO TIME LIMIT ***`, and `grep -c "Done sim"`
on its stdout returns 0, against global's `output_10199442.txt`, which ends `Starting writing
final file [...] snap_24.h5 [...]` / `Done sim` (`grep -c "Done sim"` = 1). The plan's "same
endpoint reached, exit 0" claim for arm D is false and withdrawn. Required before the default
flip: a fresh, matched-window run with the switch off that runs to a completion marker or an
explicit checkpoint (not a time-limit cancellation), with retries per unit t <= global's
matched-window rate (computed the same way, not a whole-file average) and 0 invalid-energy
failures, **and run with `RICH_INDIVIDUAL_ADAPTIVE_MODE=0` per §4's standing precondition** (this
rerun uses a fresh build from current source, which — unlike arms A/D's older binary — already
contains the unconditional adaptive-mode call; see the header note above). **Decision:** NOT
default-off yet — the switch **remains on by default in source** (`RadiationStep.cpp:339`,
`local_enabled = true`; the default flip to off has not been implemented) and R2 remains the
agreed target for 1B (it still addresses the mechanism and still cuts cadence 4.9x on the data
available), but its safety gate is open, not satisfied; the prior "default off now" instruction is
withdrawn pending the rerun above. This is a correction driven by measurement, not a removal of R2
as an agreed item.

Failure mode (unchanged): a cell that re-rejects every event at an unchanged bin pays repeated
sub-cycles; the retry cooldown in `IndividualRadiationDefectAccounting` (re-verified present:
`cooldown_fraction_ceiling`/`candidate_fraction_ceiling` restore logic at `RadiationStep.cpp:700-710`)
already carries the accepted fraction across events, and R1 removes the cause. The switch stays
for bisection regardless of the gate outcome.

R1 — bound bin growth for cells whose volume is anomalously large relative to the active set
(`Diffusion::calculateIndividualTimeSteps`, `Diffusion.cpp:474-489`, re-verified exact): keep the
existing `difference` untouched and replace the fixed growth cap `nominal_dt*2.0` by
`nominal_dt*growth_cap` with `growth_cap = clamp(2/volume_ratio, 1, 2)`,
`volume_ratio = V_i / mean active volume` (one extra SUM/SUM reduction). Fail-closed by
construction (re-derived, holds): for `volume_ratio >= 1` (an at-or-above-mean cell),
`2/volume_ratio <= 2`, clamped into `[1,2]`, so `growth_cap <= 2` always and `suggested_dt` can
only shrink relative to today's fixed `nominal_dt*2.0`; for `volume_ratio < 1` (a below-mean
cell), `2/volume_ratio > 2` clamps to exactly `2`, i.e. unchanged from today. Known limitation
(re-derived, holds): an active set of uniformly large vacuum cells has `volume_ratio -> 1` for
every member (each close to the active-set mean), so `growth_cap -> 2` and none of them is
throttled — needs a domain-wide volume reference; deferred. Switch
`RICH_INDIVIDUAL_RADIATION_DT_VOLUME_CAP` (confirmed absent from source — genuinely new), default
off until the gate passes. Gate (all four, probe from snapshot 23, on vs off, exact
`t_start`/`t_end`): (a) `historical_positive_floor_single_cell_limit` retries per unit t <= 0.5x
arm D's 2108; (b) events per unit t <= 1.1x arm D's 1938 (no cadence regression from
over-throttling); (c) fraction of events with fewer than 3 active cells or active-volume spread
< 2 stays <= 20%, else the domain-wide reference becomes required; (d) rank-0 record per event
with count of capped cells and one representative cell (`volume_ratio`, `growth_cap`,
`suggested_dt`, the uncapped value). R1 is the cadence lever that remains after R2 (1938 -> toward
557 events per unit t); it is no longer a prerequisite for R2. **This probe is also an
individual-mode-only A/B run and falls under §4's standing precondition: run with
`RICH_INDIVIDUAL_ADAPTIVE_MODE=0`.**

R3 — Diagnostic Run #1 before touching the pre-pericentre thermal path: fresh start to
t = -1.02 on the current worktree with R2 default, histogram of `RICH_RETRY` reasons vs
`output_10199025.txt`; decide from that whether `negative_or_invalid_energy_*` still needs a
remedy. No code change until then. **Also falls under §4's standing precondition** (this run
shares session N's step 1 with R2's gate run, below).

Not changed: the positivity floor's extent-vs-density metric in `conj_grad_solve.hpp` (48% of
failures would still fail a density metric at 1e-7; a solver policy change shared with global;
user decision, documented).

## 3. Speed plan, ranked by measured seconds

Per-event floor post-pericentre (arm D, 126 events, 2353 s): mesh 1114 s (36 full builds =
862 s; ~90 partial builds ~0.6 s), untraced scheduler-side 699 s, radiation 462 s, gravity
(restart-fresh) 14 s, hydro proper 65 s. In an evolved long run gravity is ~1.2 s per event.
**Not independently re-summed this round**: this event-level totals breakdown requires summing
`step_s`/`mesh_s`/`radiation_s`/`gravity_s`/`hydro_s` across all 126 arm-D events, which needs a
scripting tool (`awk`/`python`/`bc`) that this environment's permission policy blocks for
autonomous execution; the per-field per-event numbers were spot-checked individually above (§0)
and are internally consistent (e.g. a single arm-D event at cycle=10673 shows `step_s=7.491993,
mesh_s=4.679683`, both plausible components of the stated per-event floor, and matches the final
`RICH_STEP` record read directly this round while confirming the job's time-limit cancellation),
but the 2353 s / 1114 s / 862 s / 699 s / 462 s / 14 s / 65 s aggregate totals themselves should be
treated as carried over from the prior round's spot-checked figures (Codex's round-1 review states
these specific `RICH_STEP` fields were spot-checked) rather than re-derived here. Recommend the
session that implements S5 (which already adds new `RICH_STEP` fields) also add a small
log-post-processing script under version control that sums these fields per run, so this
breakdown becomes a one-command reproduction instead of a manual grep-and-add.

S1 (done, measured) — first-half mesh skip, adjacency-seeded target, no warm target:
builds per event 4.9 -> 1.3, partial build 0.89 -> 0.54 s, small event 3.36 -> 1.43 s, parity
0 mismatches over 408 events. Switches `RICH_INDIVIDUAL_FIRST_HALF_FROM_CACHE`,
`RICH_INDIVIDUAL_ADJACENCY_SEED`, both default on.

S2 (cadence) — R2 now, R1 next: 4.9x fewer events measured; R1's gain bounded by 3.5x. (R2's "now"
is qualified by the corrected gate status in §2: the default flip waits on a passing matched-window,
completed rerun.)

S3 — full builds under the drifted decomposition (largest remaining floor item: 862 s of 2353 s
in arm D; 26-30 s each vs 0.25 s pre-pericentre; ghost bringing up to 40 s). Citation and framing
(from round 1): `Simulation::stepIndividual` is defined at `Simulation.cpp:3439`, and it already
contains a repartition trigger: an `owned_cell_skew` computation (`Simulation.cpp:3894-3905`), a
`forced_balance` flag (`this->forceRebalanceSteps`, line 3891 — re-verified this round: `bool const
forced_balance = this->forceRebalanceSteps > 0 && this->tracker.getCycle() <
this->forceRebalanceSteps && this->lastRebalanceCycle != this->tracker.getCycle();`), an
`automatic_balance` flag gated on `owned_cell_skew > balance_options.threshold` plus a
cooldown/amortization check (lines 3920-3936), an `INDIVIDUAL_LOAD_BALANCE_DECISION` rank-0 log
record (lines 3938-3956), and the actual call
`this->rebalanceCommittedIndividualState(balance_step, forced_balance,
balance_options.threshold)` at lines 3965-3967 when `request_balance` is true. Concrete change:
add a new boolean condition — "last full build's `mesh_s` exceeded
`RICH_INDIVIDUAL_REBALANCE_FULL_BUILD_SECONDS` (default off; candidate 5 s) and at least
`cooldown_events` since the last repartition" — call it `mesh_rebuild_trigger`, OR'd into the
existing `automatic_balance` expression at `Simulation.cpp:3931-3934`.

**Correction 1 (unchanged this round — reviewer MAJOR from review-1, the outer OR-condition alone
does not cause a repartition):** `rebalanceCommittedIndividualState` (`Simulation.cpp:2849`
onward): `mesh_rebuild_trigger` OR'd only into `automatic_balance` changes whether
`rebalanceCommittedIndividualState` gets *called* (via `request_balance` at line 3935-3936); it
does not change whether a rebalance is *applied*. Inside the function, lines 2957-2960
independently compute `local_should_rebalance = (forceRebalance || (weightSkew > threshold &&
tess.ShouldRebalance(weights)))` — and the call site (line 3966) passes only `forced_balance` (the
unrelated `forceRebalanceSteps` start-of-run counter, lines 3891-3893) as that `forceRebalance`
argument, not the new trigger. So a domain that is stretched/incoherent (S3's actual target —
full-build cost from decomposition geometry) but balanced in owned-cell count would call the
function on every event once `mesh_s` exceeds the threshold, and every call would fall through
with `result.applied = false`: no repartition, full build cost unchanged, only extra per-event
overhead paid. Fix: thread `mesh_rebuild_trigger` through the `forceRebalance` argument too, not
only into `automatic_balance` — change line 3966's call from
`rebalanceCommittedIndividualState(balance_step, forced_balance, balance_options.threshold)` to
`rebalanceCommittedIndividualState(balance_step, forced_balance || mesh_rebuild_trigger,
balance_options.threshold)`. This reuses `forceRebalance`'s existing role as the skew-gate bypass
(the same mechanism `forced_balance` already uses to skip the weight-skew check) rather than
inventing a second path, and `mesh_rebuild_trigger` itself still respects its own cooldown (it is
OR'd with the existing `cooldown_complete`/`amortized` logic at 3931-3934 before it ever reaches
the call site, so it is not unconditionally forced every event once `mesh_s` is high).

**Correction 2 (this round — reviewer MAJOR, review-4: Correction 1's fix is necessary but not
sufficient under the exact configuration this plan's measured data comes from).** Both
`automatic_balance` and `request_balance` are additionally gated on `!active_hilbert_balance`
(`Simulation.cpp:3931,3935`, re-verified exact this round), where `active_hilbert_balance =
this->individualActiveHilbertBalanceEnabled()` (`Simulation.cpp:3486-3487`).
`individualActiveHilbertRuntimeOptions()` defaults `RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE` to
**true** (`Simulation.cpp:1288-1290`, re-verified exact), while `individualRebalanceRuntimeOptions()`
defaults `RICH_INDIVIDUAL_AUTO_REBALANCE` to **false** (`Simulation.cpp:1219-1220`, re-verified
exact); `individualActiveHilbertBalanceEnabled()` returns true whenever the mesh's load balancer is
a `HilbertLoadBalancer` and an individual balance step exists (`Simulation.cpp:3044-3060`), which is
the case for arm D's own probe run — read directly this round, the literal line, not inferred:
`meshaberr_noretrybin_10199568.txt:5` records `INDIVIDUAL_LOAD_BALANCE_DECISION ... enabled=0
active_hilbert_cache=1 ... requested=0`, confirming active-Hilbert balancing was live and
`request_balance` was consequently forced to 0 regardless of skew. So on the exact configuration
the plan's own measured data comes from, `mesh_rebuild_trigger` — even correctly threaded through
`forceRebalance` per Correction 1 — would never reach `rebalanceCommittedIndividualState` at all:
both the `automatic_balance` OR-term and the `forceRebalance` argument route through gates that are
unconditionally false while `active_hilbert_balance` is true. S3 as designed is a fix for the
non-active-Hilbert code path only.

Two consequences, resolved explicitly rather than picked silently: (i) **S3's probe and any A/B
rollout that reuses arm D's configuration must explicitly set
`RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE=0` and `RICH_INDIVIDUAL_AUTO_REBALANCE=1`** for
`mesh_rebuild_trigger` to be reachable at all — added to §4's standing precondition below,
alongside the adaptive-mode override. This trades away active-Hilbert's own active-cell balancing
for the duration of the probe, which is itself a load-balancing mechanism
(`maybeBalanceIndividualEventByActiveBins`, `Simulation.cpp:3200-3328` — re-verified this round:
decides per active-bin-mask cached partitions keyed on `activeBalanced`/`ownedMaxMean`, not on
`mesh_s`/full-build cost) and is not a no-op substitution; the probe's full-build cost reduction,
if any, is then attributable to S3's trigger and not confounded with active-Hilbert's own
rebalancing, but the probe no longer represents the exact production configuration arm D actually
ran (active Hilbert on). S3's result generalizes to a production run that keeps active Hilbert on
only once a follow-on run confirms the same effect under (ii). (ii) **S3b, the active-Hilbert-integrated variant (designed post-loop, unimplemented, unmeasured)** — the
production path, since production keeps `RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE` on.
`maybeBalanceIndividualEventByActiveBins` (`Simulation.cpp:3200-3440`) decides per event: when
`current_active_balanced && current_owned_safe` (`Simulation.cpp:3283-3287`) it only stores or
refreshes the cached cut (`action=store-balanced|refresh-balanced`) and never migrates; otherwise it
proposes a fresh Hilbert cut from the current generator positions (`this->tess.getAllPoints()`), applies
the reject rules (`reject-rebuild-empty-rank`, `reject-rebuild-owned-skew`, `reject-rebuild-active-skew`)
and migrates through `rebalanceCommittedIndividualState(balance_step, true, options.active_threshold, ...)`
(`Simulation.cpp:3387-3390`), logging `INDIVIDUAL_ACTIVE_HILBERT_DECISION ... action= migrated=
migrated_cells= migration_seconds_max=`. Arm D's log shows exactly the balanced-count, stretched-domain
case: `action=store-balanced` with `proposed_owned_max_mean=1.0000091` while full builds cost 26-30 s.
Design: compute the same `mesh_rebuild_trigger` as S3 (last full-build `mesh_s` on this rank-max exceeds
`RICH_INDIVIDUAL_REBALANCE_FULL_BUILD_SECONDS`, at least `cooldown_events` since the last migration,
all-reduced so every rank takes the same branch) and, when it fires, take the proposal branch even
though `current_active_balanced && current_owned_safe` holds, with the existing reject rules and the
existing migration call unchanged, and a new constructed action string `rebuild-full-build-cost`
(with the same `-already-current` suffix rule when the proposed cut equals the current one). When the trigger fires, `rebuild_boundaries` is forced true regardless of `cache_hit`
(`Simulation.cpp:3307-3333`: the proposal branch otherwise tries the current mask's cached cut first and
reuses it when it is count-balanced, and an unchanged cut ends as `-already-current` with no migration
at 3391-3395), so a fresh cut is constructed from the current generator positions on every firing; the
reject rules (3345-3356) and the migration call (3387-3390) are unchanged. A fresh cut that still equals
the current one is logged `rebuild-full-build-cost-already-current` and counts in the gate as a firing
without migration: if every firing ends that way the decomposition is already Hilbert-compact, the
26-30 s full builds are not domain drift, S3b fails its gate, and the cost goes back to the MadVoro
per-build items (S4). Design
check, answered post-loop: `individualActiveHilbertBoundaries` holds only Hilbert cut coordinates
(`Simulation.hpp:292-295`); its only `.clear()` calls are in the mode switches (`Simulation.cpp:2225,2258`),
the skew path overwrites just the current mask's entry (3360-3361), and a cache hit is re-validated
against the current load before reuse (`reuse-cache` requires `activeBalanced`, 3326-3328), so a cut
computed before an S3b migration is re-checked, not trusted; S3b needs no extra invalidation. Switch: the same
`RICH_INDIVIDUAL_REBALANCE_FULL_BUILD_SECONDS` as S3 (unset or 0 = off, so production is unchanged
until the gate passes). Expected saving: the same 700-800 s of 2353 s as S3 if drifted domains are the
cause, unmeasured. Gate (both arms with active Hilbert on, the production default; arm with the switch
unset vs set): at least one `INDIVIDUAL_ACTIVE_HILBERT_DECISION action=rebuild-full-build-cost
migrated=1` with `migrated_cells` > 0; the next full build's `mesh_s` <= 3 s; parity 0 mismatches over
the following 100 events; zero errors; `RICH_STEP` wall per unit t down >= 25%. Order: run S3's probe
under standing precondition 2 first, because it answers cheaply whether domain drift is the cause at
all; implement S3b only if that probe passes its `mesh_s` <= 3 s gate.

Numerics: ownership only; the same committed state on a different partition; every physics step
already implements `beforeIndividualRebalance` (re-verified present, e.g. `RadiationStep.cpp:608-611`).
Risk: the repartition itself costs two full builds (~60 s) and the routing oct-tree must be
refreshed (`RICH_OCT_ROUTING_ALWAYS_REBUILD=1` forces it — file `DistributedOctEnvAgent.hpp`
confirmed to exist under `source/3D/tessellation/MeshDecomposer3D/environment/hilbert/`); if the
26 s is stream geometry rather than incoherent domains the gain is nil. Expected: full build
26-30 s -> 1-3 s if domains are the cause (pre-pericentre value 0.25 s); arm-D-equivalent saving
700-800 s of 2353 s. Gate: `INDIVIDUAL_LOAD_BALANCE_DECISION`/"Individual load balance time" pair
with `applied=1` (not merely `requested=1` — Correction 1 is what makes `applied=1` reachable at
all for this trigger, and Correction 2's configuration is what makes `requested=1` reachable at all
under arm D's own settings) and `migrated cells=` > 0; next full build `mesh_s`
<= 3 s; parity 0 mismatches over the following 100 events; zero errors; `RICH_STEP` wall per unit t
down >= 25%. Rejected alternative: raising `partial_build_fraction` above 0.5 (draft item 3),
because arm C measured larger per-rank partial targets at 8-28 s each (documented in a source
comment at `hdsim_3d.cpp:1520-1524`); kept: the rank-0 record when `exceeds_threshold`
(`hdsim_3d.cpp:1536-1551`, lambda defined at 1536, called at 1550) fires, with count, representative
rank, local target and threshold.

S4 — partial-build fixed cost (0.54 s per event, ~90% of events): candidate A, skip the
points-manager exchange bookkeeping in `Voronoi3D::PrepareToBuildParallel`
(`Voronoi3D.hpp:1818-1892`, `pointsManager->update` at :1851 — both re-verified exact) when
`suppressRebalancing && suppressExchange`, keeping the oct-tree routing refresh that the
suppressed-exchange path installs (`DistributedOctEnvAgent.hpp`; the routing must still follow
actual positions, throttled by the existing rebuild interval). Switch
`RICH_INDIVIDUAL_MESH_SKIP_MANAGER_UPDATE` (confirmed absent from source), default off. Expected:
-0.3 s of 0.54 s per partial build (the measured "exchange" 0.305 s). Gate: parity 0 mismatches
over >= 400 events with the switch on (the `meshab_verify_10199446.txt` method,
`RICH_VERIFY_PARTIAL_BUILD=1`); "Time for preparing" median <= 0.05 s; `mesh_builds` and
full-build count unchanged. Ghost bringing (0.47 s median): instrument `totalBigQueries`/
`totalSmallQueries` (`BringGhostPointsToBuild`, `Voronoi3D.hpp:3105-3364`, re-verified: function
at 3105/3108, both counters declared at 3134-3135, accumulated at 3247-3248, printed at 3369) as
rank-0 aggregates first; S3 is expected to shrink it too.

S5 — untraced scheduler-side cost (699 s of 2353 s in arm D, 0.39 s per small event, up to
5.5 s per large event): first add `wake_s`, `commit_s`, `suggest_s`, `sync_s` fields to
`RICH_STEP` (`Simulation::stepIndividual`, around `limitIndividualTreeWakeTimeSteps`,
`commitEvent`, the suggest loop), rank-max like the other fields — and while touching this block,
also add the `full_builds=` counter recommended in §0 above, since it sits in the same code path.
Then the two concrete candidates, chosen by those fields:

(i) `limitNeighborBins` (`IndividualTimeStep.cpp:1188`, re-verified exact) exchanges the whole
mesh's bin snapshot (`SyncCanonicalDataToMesh`) plus an all-to-all whenever any bin was lowered,
i.e. nearly every event; restrict the snapshot to cells adjacent to a propagation source (sparse
exchange keyed by owner), switch `RICH_INDIVIDUAL_SPARSE_BIN_CLOSURE` (confirmed absent from
source).

(ii) [design corrected in round 3, reviewer BLOCKER from review-2 — the previous position/velocity
inflation did not bound `has_source`, remaining-sleep, or sound-speed staleness] the wake oct-tree
is rebuilt over all local cells every event (`limitIndividualTreeWakeTimeSteps`, `local_tree.insert`
for every source); the pruning test that makes the tree useful,
`individualWakeNodeCanBeReached` (`Simulation.cpp:423-443`, re-verified exact this round against the
current source), reads three things from the per-node summary built by
`buildIndividualWakeNodeSummaries` (`Simulation.cpp:215-247`, re-verified exact): `has_source`
(true only if some leaf in the subtree has `remaining_sleep[target_index] > 0` at the time the
summary was built, set at line 231), `maximum_remaining_sleep` (the max such value in the
subtree, line 234-235), and `target_bounds`'s velocity/sound-speed range (merged bottom-up via
`mergeIndividualSignalNodeSummary`, feeding `maximumIndividualWakeSignalSpeedInNode`,
`Simulation.cpp:400-421`). The reviewer's finding, re-verified directly against these lines: a
target whose `remaining_sleep` crosses from <=0 to >0 between the tree's build time and the query
(i.e. a previously-non-sleeping cell that starts sleeping) has a cached summary with
`has_source=false` propagated up from its stale leaf value, so `individualWakeNodeCanBeReached`
returns false at line 428-430 regardless of position or velocity — no bounding-box or signal-speed
inflation touches that branch at all, since it returns before either is evaluated. Position and
velocity inflation (the original design) cannot fix this because `has_source`/
`maximum_remaining_sleep`/velocity-and-sound-speed bounds are *summary* state, not *geometric*
state — they are not bounded by a distance or speed margin, they are simply wrong once stale.

**Design fix:** split what is cached from what is recomputed. Cache only the wake tree's
*structure* — the node partition and each node's target-position bounding box, i.e. the expensive
geometric build (`local_tree.insert` for every source, called once per cache lifetime instead of
once per event). Do **not** cache `IndividualWakeNodeSummaries`: rebuild it fresh every event by
calling `buildIndividualWakeNodeSummaries` (`Simulation.cpp:215-247`) against the *current*
`sources`/`remaining_sleep` arrays over the *cached* tree structure. This function is a bottom-up
merge over the existing partition — no geometric computation, no re-insertion, O(N) in the number
of local cells — so it is cheap relative to the geometric rebuild being avoided, while making
`has_source`, `maximum_remaining_sleep`, and the velocity/sound-speed bounds exactly current on
every query, every event: the staleness the reviewer identified is eliminated, not bounded. The
only state that remains genuinely stale between rebuilds is each node's `boundingBox` (built from
target *positions* at tree-construction time; positions drift as generators move, but a cell's
`remaining_sleep`/velocity/sound-speed do not live in the box, only its coordinates do). For that
position drift, track the maximum per-rank generator displacement `d_max` since the cached tree
structure was last built (a rank-max reduction in the same style already used at
`Simulation.cpp:3907-3919`), and at query time inflate every node's bounding box by `d_max` in
each dimension before calling `minimumDistanceToIndividualSignalNode` (a Minkowski-sum expansion:
subtract `d_max` from the lower-bound comparison and add it to the upper). This makes the box
comparison a superset of the true current box, so it can only make `individualWakeNodeCanBeReached`
*more* permissive on the geometry term, the direction the fail-closed constraint requires. The
separate `v_max` inflation term proposed in the prior draft is dropped as redundant: it existed to
compensate for stale velocity bounds inside the summary, and those bounds are now rebuilt fresh
every event from current velocities, so `maximumIndividualWakeSignalSpeedInNode` already receives
current data with no inflation needed.

**Invalidation (widened in round 3 per the reviewer's note on membership/indexing):** rebuild the
cached tree *structure* itself (not just the summaries) whenever (a) `d_max` exceeds a configurable
fraction of the local leaf's characteristic size (candidate 0.5), (b) a fixed event-count cooldown
elapses, whichever first — the same two-condition style as R1's and S3's triggers — or (c) the
local `sources`/`remaining_sleep` array's membership or ordering changes at all (an AMR
refine/derefine, a rebalance, or any other event that adds, removes, or reorders local cells):
each tree leaf stores a fixed `source_index` into those arrays (`Simulation.cpp:226`), so if the
array is reordered or resized without a matching tree rebuild, a leaf's index silently refers to a
different physical cell — a correctness bug independent of dt-conservativeness, not something
`d_max` inflation can address, so it is invalidation condition (c) rather than a numeric margin.
Switch `RICH_INDIVIDUAL_WAKE_TREE_CACHE` (confirmed absent from source), default off. **Gate
(unchanged from the prior draft — still the right empirical backstop regardless of the design
argument above, per the standing "never claim a change works without a run" constraint)**: zero
per-cell wake-deadline increases against fresh-tree evaluation — a paired A/B run on the probe
window that evaluates `signal_wake_deadlines` with both the cached-structure/fresh-summary tree and
a fully fresh tree every event, and requires cached deadline <= fresh deadline for every cell,
every event (not aggregate `active_bins` histogram matching, which cannot establish this per-cell
property); plus the performance target, wake-tree phase wall time down >= 50%. Because summaries
are now rebuilt every event regardless of cache state, the performance saving comes entirely from
skipping the geometric partition rebuild — re-run the wake-tree phase wall-time measurement once
implemented to confirm the >= 50% target still holds with only the structural part cached (it was
sized against caching everything, so this should be treated as re-opened by the design change, not
carried over).

Gates (both candidates): the new `RICH_STEP` fields account for >= 90% of the residual; each
change cuts its own field by >= 50% with zero `INDIVIDUAL_BIN_OVERRUN` increase; (i) additionally
requires identical `active_bins` histograms on the probe window (bit-identical scheduler state is
expected for a sparse-vs-dense exchange of the same data); (ii)'s safety gate is the per-cell
deadline check above, not a histogram match. Dropped from the draft: widening
`RICH_INDIVIDUAL_FULL_SOURCE_SWEEP_INTERVAL` (measured 0.06 s per sweep, 223 sweeps in 11319
events).

S6 — FMM (1.2 s per event in evolved runs; 0.17 s per solve fresh). Measured: structural
rebuilds are 2.4% of solves and ~12% of FMM time under the current interval 32
(`FmmDistributedOptions.hpp:50`, `minSolvesBetweenStructuralChanges = 32`, overridden by
`RICH_FMM_STRUCTURAL_INTERVAL`, re-verified both), so `RICH_FMM_STRUCTURAL_INTERVAL=256` is
bounded to ~0.02 s per event and is not the lever. That "2.4%" figure is `leaf_change_ranks != 0`
(51/2117 solves) — a windowed leaf-capacity change, not a full topology rebuild; only 1 of the 2117
traced solves (`call=1`, the initial restart) shows `process_rebuilt=1`/`root_change_ranks=256`.
S6's proposed forced-reset policy is therefore a different and heavier event than the "51
rebuilds" baseline it is compared against — its cost should be estimated from the single observed
full-rebuild solve (`topology_max=0.778 s` at that call), not from the cheaper leaf-change solves.
The lever is the 7-25x growth of `total_mean` between a fresh and an evolved state (accumulated
topology). Concrete change: a forced full topology rebuild plus redistribution when the running
median of `fmm_solve_trace total_mean` exceeds 3x its value after the last rebuild (switch
`RICH_FMM_RESET_FACTOR`, default off, confirmed absent from source). Performance gate for the
reset: `total_mean` <= 0.3 s per solve over a 0.1-unit window from an evolved state (currently
1.2 s), zero errors, and the reset's own cost (one full-rebuild-equivalent solve) amortized over
the window must not exceed the savings it produces — i.e. `(reset count in window) * (observed
full-rebuild solve cost) < (baseline total_mean - 0.3 s) * (solves in window)`. **Accuracy gate,
added in round 3 (reviewer MAJOR — "existing tolerance" cited by the prior draft does not exist as
an enforced check):** re-checked this round, `sampleDirectAccelerationError`
(`DistributedFmmGravityCalculator.cpp:1576-1750`) only computes and `printf`s
`fmm_tde_sample_error`/`max_relative_acceleration_error` (and mean/rms/percentile variants); it
never compares against a stored threshold or throws — there is no "existing tolerance" for S6 to
inherit. Gate, using the same relative-comparison style S6b already establishes below: run the
evolved-state probe with `RICH_FMM_TRACE=1` and `directErrorSampleCount>0` enabled, both with the
reset policy on and with it off (baseline), sampled at matched solve indices (before/after a reset
event on the "on" run, at the corresponding solve count on the "off" run); require
`max_relative_acceleration_error` with the policy on <= 1.1x its value with the policy off at the
matched sample. This is not measured yet — it is a new gate for S6, not a claim that S6 already
passes it.

**S6b — targets-only FMM evaluation, given its own contract (reviewer MAJOR from review-1 — this
was previously bundled into S6 with no separate switch, saving estimate, or gate; split in round 3
into an instrumentation prerequisite and a gated implementation per review-2's MAJOR that
the performance gate had no numeric target and no explicit ordering).** Mechanism, re-verified this
round: `EvaluateIndividualTargets` (`FastMultipoleAcceleration3D.cpp:568-629`) calls
`calculator_.solve(points_, masses_, ...)` with `points_ = source_points` (the full source set the
caller passes in) and requires `acc.size() == points_.size()` (lines 632/635) — i.e. today the
source set passed in is already whatever the caller chose, but every source in that set gets an
evaluated acceleration, including inactive cells that are only present as gravitating sources and
not as timestep targets this event. The proposed change adds a target-subset argument so the
upward/downward LET build stays over the full source set (needed for correctness — inactive cells
still act as sources for active ones) but the M2P/P2P evaluation (the `let_execute_mean`/
`local_traversal_mean`/`let_m2p_max` fields already in `fmm_solve_trace`) only touches active
targets. Switch `RICH_FMM_INDIVIDUAL_TARGETS_ONLY` (confirmed absent from source — checked against
the full `RICH_FMM_*` name list, distinct from `RICH_FMM_RESET_FACTOR`), default off.

**S6b-0 (instrumentation, prerequisite — must land and produce a number before S6b-1 below is
implemented):** add an `active_targets=`/`targets=` field to `fmm_solve_trace`
(`DistributedFmmGravityCalculator.cpp`, same rank-0 `printf` block as the existing `*_sum`/`*_max`
fields) recording the count of timestep-active targets passed to a solve relative to the total
source count — confirmed this round, re-checking `fmm_solve_trace`'s full record
(`fmm_10199396.txt:284`, call=1) and the printf format string, that no such field exists today
(only interaction-count aggregates like `let_active_m2p_sum`). Run this instrumentation on the
probe window to measure the typical active-target:source ratio during an individual event. This
measurement is what sizes both the expected saving and the performance gate below — it is a
one-field logging addition with no behavior change, so it carries no correctness risk of its own
and can land ahead of, and independently of, S6b-1.

**S6b-1 (implementation, gated on S6b-0's output):** do not start this until S6b-0 has produced the
active-target:source ratio. **Correctness gate:** run the same probe with
`directErrorSampleCount>0` enabled both with and without the switch and require
`max_relative_acceleration_error` with the switch on <= 1.1x its value with the switch off — a
relative comparison, since `sampleDirectAccelerationError` only logs a value and never enforces a
threshold itself (re-checked this round, same file as S6's accuracy gate above). **Performance
gate:** once S6b-0's measured ratio is available, set a numeric target for the traversal-phase
fields (`let_execute_mean` + `local_traversal_mean` + `process_upward_mean` +
`process_downward_mean`) consistent with that ratio (e.g., a reduction proportional to the
inactive-target fraction measured, discounted for the unchanged LET-build cost) and require the
implementation to meet it; no implementation work proceeds against an unset numeric target.

S7 — adaptive integration controller (restored in round 3 — reviewer BLOCKER, review-3: this
item was present in the originally agreed plan at its lines 193-197 and was dropped between round 2
and round 3 with no changelog or dissent entry recording why; restored unchanged in substance, with
every citation re-verified below against the current source). `Simulation::SetAdaptiveIntegrationMode`
(declared `Simulation.hpp:107`, defined `Simulation.cpp:2125-2158`), built as `rich_adaptive_20260922`
per that binary's own provenance record, untested in production. Mechanism, re-verified this round
against the current source: the controller measures throughput (`tau = simMeasured / wallMeasured`)
in the current mode and, once dwelled for `dwell_min_steps * dwellMultiplier` events
(`Simulation.cpp:2410`, default `dwell_min_steps=64` at `Simulation.cpp:2030`) with at least
`minimum_samples` (default 6, `Simulation.cpp:2031`), probes the other mode for a bounded wall-time
budget (`probe_fraction`, default 0.1, `Simulation.cpp:2036,2064-2065`) and adopts whichever is
faster by a `margin` (default 1.15, `Simulation.cpp:2037,2066-2067`); while stepping globally it
additionally bounds individual mode's possible gain from the per-cell CFL distribution and skips
probing when `gainBound < gain_minimum` (default 1.5, `Simulation.cpp:2038,2068-2069,2414-2422`,
action `stay_global_little_to_gain` at line 2419, re-verified exact). Switches happen only at
synchronized individual events (`Simulation.cpp:2196-2204`) and are logged rank-0 as
`RICH_MODE_DECISION cycle=... mode=... phase=... action=...` (`Simulation.cpp:2174-2186`, `action`
built as `"adopt_"+mode`/`"probe_"+mode`/`"revert_to_"+mode` — re-verified this round these are
*constructed* strings, not literal constants, at `Simulation.cpp:2393-2394,2401-2402,2425-2426`) and
`RICH_MODE_SWITCH cycle=... from=individual to=global reason=...` (`Simulation.cpp:2233-2237`,
re-verified exact for the individual-to-global direction). Every threshold is independently
overridable (`RICH_ADAPTIVE_DWELL_MIN_STEPS`, `RICH_ADAPTIVE_MIN_SAMPLES`, `RICH_ADAPTIVE_RAMP_EVENTS`,
`RICH_ADAPTIVE_RAMP_STEPS`, `RICH_ADAPTIVE_GATE_INTERVAL`, `RICH_ADAPTIVE_DWELL_BACKOFF_CAP`,
`RICH_ADAPTIVE_PROBE_FRACTION`, `RICH_ADAPTIVE_MARGIN`, `RICH_ADAPTIVE_GAIN_MIN`,
`Simulation.cpp:2052-2069`, re-verified exact), and the controller itself is switched by
`RICH_INDIVIDUAL_ADAPTIVE_MODE` (`Simulation.cpp:2047-2051,2130-2131`, re-verified exact this round —
the same override this round's MAJOR fix in §4 relies on): unset, the compiled-in argument from the
call site applies (`true` at `runs/BaseTDEComptonIndividual/test.cpp:1626`, so this run's default is
adaptive-on); set to 0/1 it overrides that argument outright; the parsed options are
MPI-consistency-checked across ranks when `RICH_MPI` is defined (`Simulation.cpp:2078-2113`, throws
`std::invalid_argument` on a rank mismatch, re-verified exact).

Validate functionally on the snapshot-23 probe (expect `RICH_MODE_DECISION action=probe_global`
after 76 events, `RICH_MODE_SWITCH ... to=global`, `action=adopt_global`, `gain_bound` 2-3,
`stay_global_little_to_gain`) — **carried forward from the originally agreed plan, not re-derived
this round**: the 76-events/gain_bound-2-3 figures come from the snapshot-23 adaptive-mode probe
log, which this round's scope did not require re-opening; carried here rather than re-asserted on
trust, and listed again in §4's open measurements below. Run this validation with
`RICH_INDIVIDUAL_ADAPTIVE_MODE` either unset or explicitly `=1` (never `=0` — that would disable the
exact mechanism under test) and expect `RICH_MODE_SWITCH` records to appear; this is the one
individual-mode run in the whole plan that is *supposed* to exhibit mode switches, as distinct from
every other individual-only A/B gate in this plan (§4's standing precondition below), which must
show zero. Re-validate its thresholds after S2-S6 change the individual baseline, since
`tau_individual` shifts with every cadence and mesh-cost change those items make.

Gates: (a) functional — the probe run produces at least one `RICH_MODE_DECISION`/`RICH_MODE_SWITCH`
pair with no thrown `std::invalid_argument` and no crash; (b) continuity across the switch, **measured, rewritten post-loop** — the round-4 text anchored this
gate on the `Einit`/`Efinal` print at `MultigroupDiffusion.cpp:6252-6253`; that print exists only in
the multigroup solver, and the TDE driver builds the grey `Diffusion` class
(`runs/BaseTDEComptonIndividual/test.cpp:26,1542`), which has no such print (zero `Einit` hits in
`source/Radiation/Diffusion.cpp` and in `meshab_noretrybin_10199568.txt`,
`meshaberr_noretrybin_10199568.txt`, `output_10199442.txt`, `error_10199442.txt`). The mechanism
argument stands (`beforeIndividualRebalance` on every physics step, `Simulation.cpp:2218-2219`; the
post-switch global dt is the scheduler's own next event, `Simulation.cpp:2211-2217,2231`) but the gate
is a measurement. **Instrumentation prerequisite (no behaviour change, same category as S6b-0):** a
rank-0 record `RICH_MODE_SWITCH_STATE cycle= time= phase=before|after direction=individual_to_global|
global_to_individual owned_cells= mass= momentum_x= momentum_y= momentum_z= energy= erad= next_dt=
scheduler_next_dt=`, the sums being `MPI_SUM` reductions over owned cells of the committed extensives,
emitted at the top and the bottom of `adaptiveEnterGlobal` (`Simulation.cpp:2206-2244`, after the ghost
exchange at 2229 for the `after` record) and of `adaptiveEnterIndividual` (`Simulation.cpp:2246-`).
Gate, fixed tolerances (the switch performs no physics, only ownership and ghost bookkeeping, so the
targets are round-off): for every before/after pair, `owned_cells` identical; `mass`, `energy`, `erad`
relative difference <= 1e-12; each momentum component's difference <= 1e-12 x the sum of |momentum|
over owned cells; `next_dt` equal to `scheduler_next_dt` whenever the latter is finite. In the 64 steps
after each switch: zero `RICH_RETRY` records with `reason=negative_or_invalid_energy_*` (the class the
global run hit 4 times, `output_10199442.txt`), zero aborts, and the rate of the ordinary
`historical_positive_floor_single_cell_limit` retries per unit simulated time no more than 2x the rate
over the 64 steps before the switch. Physics continuity against a twin: run the same snapshot to the
same final time with `RICH_INDIVIDUAL_ADAPTIVE_MODE=0`; at the common final output, total mass agrees
to relative 1e-12 and total energy (gas plus radiation) to relative 1e-6, the latter being the level at
which the two schemes' different step sequences already differ in the arm A/E comparison. **None of
this has been measured yet** — it is the concrete gate for S7, not a claim that S7 passes it; (c) controller-decision
arithmetic, **added this round (reviewer MAJOR, review-4 — the 1.15 margin and 1.5 gain threshold
were cited as configured defaults but never checked against actual logged decisions)**: for every
`RICH_MODE_DECISION` record with `action=adopt_<mode>` in the probe log, its own printed
`tau_<mode>` divided by `tau_<other>` must be >= the run's effective margin (default 1.15,
`Simulation.cpp:2037,2066-2067`, or the run's `RICH_ADAPTIVE_MARGIN` value if set — read from that
run's own environment, not assumed); for every record with `action=stay_global_little_to_gain`, its
own printed `gain_bound` must be < the run's effective gain minimum (default 1.5,
`Simulation.cpp:2038,2068-2069`, or `RICH_ADAPTIVE_GAIN_MIN` if set). Both checks use fields the
controller already logs (`Simulation.cpp:2174-2186`); those records print with `std::setprecision(6)`
(2174), so the replay is rounding-aware: adopt requires `tau_<mode>/tau_<other> >= margin x (1 - 1e-5)`
and stay-global requires `gain_bound < gain_minimum x (1 + 1e-5)` (six significant digits round at
5e-6 relative), and the instrumentation batch raises `adaptiveLogDecision`'s precision to 12 significant
digits (no behaviour change) so later logs replay exactly; no other instrumentation is needed for (c); every record in the probe log must satisfy its own decision arithmetic, not just the
carried-forward `gain_bound 2-3` example; **this has not been measured yet either**; (d)
re-validation after S2-S6 (renumbered from the prior draft's (c)): `tau_individual`/`tau_global` and
`gain_bound` recomputed against the post-remedy baseline, and decisions still land on the faster
mode within the measured margin (no independent numeric target beyond "consistent with the measured
tau ratio" — this is a controller-correctness check on a mechanism that reacts to whatever the
baseline is, not a speed item with its own saving to gate on).

## 4. Execution order

**Standing precondition 1, for every individual-mode-only A/B run below (reviewer MAJOR, review-3;
confirmed this round against build provenance, not just source reading):**
`runs/BaseTDEComptonIndividual/test.cpp:1626` calls `simulation.SetAdaptiveIntegrationMode(true,
individual_options)` unconditionally, and `Simulation.cpp:2048-2051,2130-2131` lets
`RICH_INDIVIDUAL_ADAPTIVE_MODE` override that compiled-in `true` when set. The already-measured
arms in §0 are unaffected — `rich_mesh_seed_20260921.provenance.txt` shows arm A/D's binary was
built 2026-09-21, one day before the adaptive controller was added to that test file (confirmed by
`rich_adaptive_20260922.provenance.txt`, dated 2026-09-22, the first binary here whose description
names it) — but every run this plan still has to make is a *fresh* build from the current worktree,
whose `test.cpp:1626` already contains the unconditional call today. Left unset, such a run has the
adaptive controller live and can switch itself to global stepping mid-measurement (S7 above), which
would silently validate global-mode behaviour under an individual-mode label and invalidate
whichever gate that run was measuring. Every individual-mode-only measurement/gate run below — R1's
probe, R2's Session-N run, and each of S1-S6b's A/B gates — must therefore set
`RICH_INDIVIDUAL_ADAPTIVE_MODE=0` and have its log checked for zero `RICH_MODE_SWITCH` records and
exclusively `mode=individual` `RICH_STEP` records over the measurement window before that run's gate
is considered passed; a run with even one `RICH_MODE_SWITCH` record does not satisfy the gate
regardless of the measured numbers, and must be redone with the override set. S7's own validation
(step 5 below) is the sole exception: it exists to test the switching mechanism itself, so it must
run with `RICH_INDIVIDUAL_ADAPTIVE_MODE` unset or explicitly `=1`, and is expected to produce
`RICH_MODE_SWITCH` records.

**Standing precondition 2, S3 only (added this round, reviewer MAJOR, review-4):**
`request_balance` and `automatic_balance` are both gated on `!active_hilbert_balance`
(`Simulation.cpp:3931,3935`), and `RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE` defaults to on
(`Simulation.cpp:1288-1290`) while `RICH_INDIVIDUAL_AUTO_REBALANCE` defaults to off
(`Simulation.cpp:1219-1220`) — confirmed live in arm D's own log
(`meshaberr_noretrybin_10199568.txt:5`, `active_hilbert_cache=1 ... requested=0`). S3's probe and
rollout (step 6 below) must therefore set `RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE=0
RICH_INDIVIDUAL_AUTO_REBALANCE=1`, or `mesh_rebuild_trigger` never reaches
`rebalanceCommittedIndividualState` regardless of Correction 1's threading fix; see §3 S3
Correction 2 for the tradeoff this implies (active-Hilbert's own balancing is disabled for the
probe) and for the still-unimplemented active-Hilbert-integrated alternative that would avoid it.

Session N: (1) a single fresh run with R2's switch off and **`RICH_INDIVIDUAL_ADAPTIVE_MODE=0`
(standing precondition 1 above)**, scoped to serve two purposes at once: Diagnostic Run #1's retry
histogram (fresh pre-pericentre start to t = -1.02, vs `output_10199025.txt`, decides R3) *and*
R2's matched-window/completion gate (continue the run through the post-pericentre window comparable
to arm D's, to a completion marker or an explicit checkpoint rather than a job time-limit
cancellation — allocate walltime accordingly, since arm D's prior attempt ran out of time before
finishing). Before evaluating either purpose's result, confirm this run's own log shows zero
`RICH_MODE_SWITCH` records and every `RICH_STEP` record carries `mode=individual`; a run that
switched modes mid-window answers neither Diagnostic Run #1 nor R2's gate and must be redone. Flip
R2's source default to off only if that run's matched-window retry rate (compared against the
corrected 1913/unit t global baseline, §0/§2) and completion conditions both pass; otherwise R2
stays an agreed target with an open gate. (2) R1 implementation, default off, gate on the
snapshot-23 probe, **also run with `RICH_INDIVIDUAL_ADAPTIVE_MODE=0`**. (3) Instrumentation:
closure-threshold trigger record, `RICH_STEP` scheduler fields (including the `full_builds=`
counter — see §0/§5-S5), ghost-query counters, `RICH_FMM_TRACE=1` on the probe, S6b-0's
`active_targets=` field, and S7 gate (b)'s new `RICH_MODE_SWITCH_STATE` record in
`adaptiveEnterGlobal`/`adaptiveEnterIndividual` (all cheap, no behavior change — can land
alongside the other instrumentation in this session rather than waiting for S6's or S7's session).
(4) S4 candidate A behind its switch with the parity gate, **also under standing precondition 1**
(its A/B parity comparison is individual-mode-only). (5) S7 functional run — **the sole exception
to standing precondition 1**: run with `RICH_INDIVIDUAL_ADAPTIVE_MODE` unset or `=1`, expect
`RICH_MODE_SWITCH` records, and evaluate gates (a)-(c) as specified in §3 S7 (gate (b) requires
step 3's instrumentation to have landed first).

Session N+1: (6) S3 — extend the existing `automatic_balance` condition in
`Simulation::stepIndividual` (`Simulation.cpp:3931-3934`) with the `mesh_s`-threshold trigger, *and*
thread that same trigger through the `forceRebalance` argument at the call site
(`Simulation.cpp:3966`, Correction 1), gate on `applied=1` and `migrated cells=` > 0 as corrected,
**and run with `RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE=0 RICH_INDIVIDUAL_AUTO_REBALANCE=1` per
standing precondition 2 above (§3 S3 Correction 2)** — without both env vars, `applied=1` is
unreachable regardless of the forceRebalance threading fix. (7) S5 candidates chosen by the new
fields: (i) gated by identical `active_bins` histograms; (ii) implemented per the corrected
cache-structure-only/fresh-summary design above, gated by the per-cell deadline check, not a
histogram match, and its performance target re-measured against the narrower (structure-only)
caching scope. (8) S6 reset policy, sized against the single observed full-rebuild-solve cost (not
the 51 leaf-change solves) and gated on both the performance target and the new relative accuracy
gate above; S6b-1 only after S6b-0's instrumentation (landed in session N, step 3) has produced the
active-target:source ratio and that ratio has been turned into S6b-1's numeric performance target —
S6b-1 does not start on an unset target. Steps 7-8 are individual-mode-only gates and fall under
standing precondition 1: `RICH_INDIVIDUAL_ADAPTIVE_MODE=0`, zero `RICH_MODE_SWITCH` records
required; step 6 falls under both standing preconditions. (9) Re-validate S7 against the new
baseline (`RICH_INDIVIDUAL_ADAPTIVE_MODE` unset or `=1`, the exception case again), including gates
(b) and (c) re-evaluated against the post-remedy baseline; consider flipping R1 default.

Open measurements: pre-pericentre retry histogram on the current worktree (R3); whether the
26 s full build is decomposition or geometry (S3 gate answers it, once run under standing
precondition 2's configuration); the scheduler residual split (S5 fields); FMM evolved-state trace
(S6, now sized off the corrected rebuild-cost basis, and gated on the new relative accuracy
comparison, not yet measured); R1's coverage criterion (c); the exact full-vs-partial build split
for arms A and D (30/419, 36/126 — currently inferred, not directly greppable in those log files;
resolved once the `full_builds=` field lands); the event-level cost floor (2353 s / 1114 s / 862 s /
699 s / 462 s / 14 s / 65 s) has not been independently re-summed this round and should be
reproduced with a committed script once S5's fields land; R2's matched-window retry-rate and
completion gate (no passing run exists yet, only the mismatched-window comparison that is now
withdrawn — retry count corrected to 119/1913 per unit t this round, conclusion unchanged); S6b's
active-target:source-count ratio (no existing log field carries it — S6b-0 above is the dedicated
prerequisite step to measure it); S5(ii)'s wake-tree phase performance target under the narrower
structure-only cache scope (re-opened in round 3 by the design correction); S7's 76-events/
gain_bound-2-3 figures (restored in round 3 from the originally agreed plan, not independently
re-derived); whether any run already executed under this study line other than arms A/D used a
build with the adaptive controller live and no `RICH_INDIVIDUAL_ADAPTIVE_MODE=0` override (checked
only for arms A/D via their provenance files); **S3b (§3 S3 Correction 2(ii)), the active-Hilbert-integrated trigger, is designed post-loop but
unimplemented and unmeasured, and carries one open design check (whether a migration invalidates the
other cached `active_bin_mask` cuts); it is the path to a production configuration that keeps
active-Hilbert balancing on, S3's standing-precondition-2 probe being the cheap cause test**; **S7 gate
(b)'s conserved-sum check and gate (c)'s decision-arithmetic check have not been run — they require the
S7 probe (step 5) plus, for gate (b), the `RICH_MODE_SWITCH_STATE` record landing first**; **§0's
global wall-time normalization was re-summed after the loop from the raw 256 `RICH_STEP.step_s` values
(1626.997066 s, 3538.74 s/unit t) and the stale 3561 traced to a partial read of the running job; no
longer open**.

## Assumptions

- The prior convergence run's log-derived numbers that I could not independently recompute this
  round (the arm-D per-event cost floor totals in §3, and the exact 30/419 and 36/126 full-build
  event counts in §0) are assumed correct as measured by the prior session, since Codex's round-1
  review states it spot-checked the underlying `RICH_STEP` fields directly; I flag them as open
  rather than either re-asserting or rejecting them.
- I used the `Read` tool's offset/limit parameters (not a shell command) to pull exact line ranges
  from `output_10199442.txt` in round 3 (offsets 6000 and 6690, 40 and 70 lines respectively) to
  directly verify the reviewer's NIT about the retry-count boundary mechanism, and this round to
  pull `meshaberr_noretrybin_10199568.txt`'s first 10 lines and `Simulation.cpp` line ranges around
  1109-1310, 3037-3320, 3880-3980, and 2160-2250 to verify the S3 and S7 fixes below directly
  against source — file-reading-tool re-derivations of specific, reproducible claims, not
  re-assertions on trust and not shell commands.
- "Corrected" in this plan (this round and carried forward) means either a citation/derivation
  error, a design gap, a missing gate, or a measurement compared across mismatched conditions and
  now compared correctly; in every case the fix is to state the corrected fact or design plainly,
  not to remove the underlying agreed item.
- S5(ii)'s redesign (cache structure only, rebuild summaries every event, round 3) and S7's
  restoration (round 3) are scope narrowing/restoration, not new mechanisms — see their sections
  above for the specific citations re-verified.
- **This round's two MAJOR fixes (S3's active-Hilbert gating, S7's gates (b)/(c)) were verified by
  direct source reading against the lines cited above and, for S3, against the literal log line
  `meshaberr_noretrybin_10199568.txt:5`; neither fix has been exercised by an actual run yet — both
  remain open per Open measurements above, consistent with "never claim a change works without a
  run."** S3's fix changes what configuration is required for the existing (unimplemented)
  `mesh_rebuild_trigger` design to be reachable; it does not change the trigger's own logic. S7's
  fix changes what gates (b) and (c) measure (a log-derived conservation check and a decision
  log-arithmetic check, respectively) rather than what the controller does.
- **§0's global wall-time normalization (3561 s/unit t) is corrected to 3538.74 s/unit t (reviewer
  NIT, review-4) and was re-summed after the loop from the raw log** (1626.997066 s over 256
  `RICH_STEP.step_s` values, span 0.459767687645476); arms A and D re-summed the same way and unchanged.
  The 3561 was 1606 s over the first 252 steps of the same log read while the job was running.
- The standing preconditions added to §4 (`RICH_INDIVIDUAL_ADAPTIVE_MODE=0` in round 3;
  `RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE=0 RICH_INDIVIDUAL_AUTO_REBALANCE=1` this round, S3 only) are
  each confirmed necessary for *future* runs by direct source reading and confirmed *not* to
  retroactively invalidate any number already in this plan (checked against arm A/D's own
  provenance and log records respectively). I did not check every other `rich_*` binary/`output_*.txt`
  pair in the run directory against its own provenance file, since no other gate in this plan
  currently depends on a production run's individual-mode data beyond what §0 already uses; listed
  as an open measurement above rather than silently assumed.
- I did not re-verify the regression-test or docs files modified in the git status (`docs/`,
  `regression_tests/`) since the task scope is the convergence/speed plan's code and log
  citations, not those files' content.

