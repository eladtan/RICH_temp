# Negative-mass abort, cell 50107 (2026-09-22): root cause, experiments, fix, order

Status: plan, nothing run. Sources: Claude's investigation (evidence file `crash50107_evidence.md` in the
session-e01101c6 scratchpad), the codex-converge run `~/.codex-converge/runs/20260923-070820-1141002`
(author claude-fable-5-1, reviewer gpt-6-astra xhigh; two full reviews, then the author hit its quota), and this
reconciliation by Claude, which applies the round-2 review and adds facts the loop did not have (marked **new**).
Numbers are copies or arithmetic on them; anything else says "estimate".

## 0. Verified facts used throughout

- `submit_step1.sh:35/:37` hard-set `RICH_INDIVIDUAL_MESH_DRIFT_FRACTION=0` and
  `RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS=0`; production `submit.sh:53` hard-sets the drift fraction to 0. The code
  default is 0.0 (`hdsim_3d.cpp` `IndividualMeshDriftFraction`); nothing prints the effective value, and
  `mesh_drift_limited=N` is an MPI sum printed only when > 0. A log without that line cannot distinguish "off" from
  "on, never binding".
- Drift limit: per active cell, `f * min_j |r_i - r_j| / |w_i - w_j|` over non-boundary face neighbours using
  `point_vel_scratch_`, which `SyncPartialBuildData` extends to every mesh point including ghosts
  (`Tessellation3D.hpp` ~570-600), floored at `0.0625 x` the cell's hydro limit. Evaluated in
  `suggestIndividualTimeSteps` after `stepIndividual` chose the next velocities (`Simulation.cpp:3822, 3859`).
- `RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS`: unset = on; accepts 0/1/false/true/off/on/no/yes; rank-consistent.
- Event times: `t = -1.51893201538 + 0.0064 x tick / 2^36` reproduces the record's `event_time` to 1e-11. 50107's
  killing interval: `primitive_tick 18141941858304` (opened t = 0.170668), closed at tick 18210661335040 (t = 0.177068).
  The two jobs entered the abort event with different event structure (event_dt 0.0004 vs 0.0008) but the same
  interval, cell and dominant face.
- **new** Restart 10200535 ran on `d25g[2,...]`; d25g2 carries the orphan `rich_intelReleaseMPI` pid 981123 (memory
  `rich-orphan-processes-slow-nodes`). Over the 20 events that the fresh and the restart run share with identical active
  counts, the restart's `step_s` is a median 4.53x (3.39-16.3x) the fresh run's, uniformly across hydro, radiation, mesh.
  The co-location and the uniformity support the orphan as the cause; they do not prove that all of the slowdown is
  the orphan's (E0 on clean nodes measures it).
  The fresh run 10200287 (nodes d25g[6,13-14,16-21,23,25-30], no orphan node) covered the same window
  0.141162-0.176668 in 84 events and 914.3 s of summed `step_s`. So the "73 min to the abort" is an orphan artefact;
  estimate for a clean-node reproduction: ~15 min of steps plus restart I/O and the first full build.
- **new** No capacity now: bigrun's association limit for elads is cpu=1024 and the user's tops20 arrays use 1024
  (one element pending on AssocGrpCpuLimit); core and socket have 0 idle d26g nodes. Any E-run pends until the user
  frees quota or chooses a partition.

## 1. Root cause

### Proven

- Mass bookkeeping: `applied_mass_sum = -8.561e-13 = -117.5%` of `pre_mass 7.284e-13`; post -1.277e-13. Face 11
  (to 717470, owner rank 104): `7.64e-8 x 1.120e-3 x 0.0064 = 5.48e-13 = 75.2%` of pre_mass. 15 corrections, all local.
- The killing face separates nearly identical gas (velocities within 0.1 per component, densities 2.93e-8 / 3.06e-8,
  sound speed ~0.1), so its mass flux is set by the face motion relative to the gas. This is a statement about face 11
  and the two large local faces (1, 2; 102% of pre_mass together), not a claim that the hydrodynamic flux vanishes on
  all 15 faces.
- Relative generator displacement across face 11 over the interval: `|w_L - w_R| x face_dt = 5.491 x 0.0064 = 0.0351`.
  Both factors are in the record; a point velocity is installed at activation and held for the interval, and
  face_dt = 0.0064 means the face was not re-fluxed mid-interval.
- The interval was legal for every guard that was on: cell_dt = 2^36 ticks exactly (no bin overrun); the mass-loss limit
  is reactive; the CFL uses `cs + |n.(v - w_f)|`, in which the antiparallel kicks (cosine -0.84) largely cancel; the only
  guard keyed on `|w_i - w_j|` was hard-set off.
- The kicks are RoundCells as coded (measured/model median 1.032, p10 0.955, p90 1.184 over 1,726,401 cells at
  t = 0.141162); at the killing interval 2.96 (50107) and 2.72 (717470).
- The crash event used a full BuildParallel (1,290,771 of 1,750,000 active; mean 0.738 > the per-rank 0.5 threshold, so
  at least one rank crossed it). Face areas sum to 4.284e-3 vs 4.1e-3 for the volume-equivalent sphere: evidence against
  inflated or phantom area, not proof of closure.
- Physics-level determinism: same cell, tick, face and 75% share in both jobs, fields within 1e-5..1e-3.

### Estimates

- Separation at the interval opening is unmeasured; the last value is 0.0298 at t = 0.141162. With it: displacement /
  separation ~1.2, and f = 0.2 would have limited the interval to ~0.0011 (bin 33) instead of 0.0064. The CFL/16 floor
  cannot be evaluated (the record has no CFL value).
- Uniform-flow GCL reading: `-8.56e-13 / 2.93e-8 = -2.92e-5 = -117%` of V = 2.49e-5; exact only if the flow is uniform
  across all 15 faces, which it approximately is.
- **new** Static cost of F1 at t = 0.141162 (`drift_cost_estimate.py`, 16 nearest generators as a proxy for face
  neighbours, floor dt_bin/16, no closure or event-count effects): f = 0.2 limits 4.0% of cells (0.49% by >= 2 bins),
  +1.5% cell activations per unit t; f = 0.5 limits 0.23%. Cell 50107 was not limited then (0.0077 vs 0.0064). The
  historical "~20% events" is from a pre-bin-enforcement regime. Only E1 measures the real cost.

### Hypotheses

- H1 (primary): a coarse interval with large antiparallel kicks lets two generators move ~their separation relative
  to each other; the face between them is charged the whole interval at end geometry. E1 tests the remedy.
- H1b (R2 exposure): 0.10 <= t < 0.1767: R2-off 212 events, 47.8% of active cells in bins >= 36, 0% in 26-33; fold-on
  10199059 1833 events, 22.5%, 37%. Confounded (other binary `rich_bin_enforce_late_20260919`, other partition). E2 tests
  the fold from this checkpoint only.
- H3 (offset growth): kicks grew 0.41/0.43 -> 2.96/2.72 (~7x) in <= 4.6 bin-36 intervals; estimated gain
  `G = chi c dt / R = 1.75 x 5.6 x 0.0064 / 0.0181 = 3.5`; the population median d/R is flat (0.016-0.028) across G, so
  not mesh-wide. Overshoot against a static centroid, a co-moving centroid, and neighbour-driven growth are
  indistinguishable from two snapshots; the trace in every arm is designed to separate them (see Open question for
  its limits).
- H-cost (new framing, round-2 review): whether F1 carries a recurring cost is itself a hypothesis. Shortening dt lowers
  G, which can suppress the growth and let the kicks, and the limiting, fade while RoundCells is unchanged; or the pair
  keeps large kicks and stays limited. E0 gives the unbounded history, E1 the bounded one; nothing here says F3 is
  necessary.
- H4 (bulk-speed c): a RoundCells design property; the user's decision.

### Excluded

- Bin overrun: cell_dt = 2^36 ticks exactly.
- The 2026-09-17 partial-build ghost-omission mechanism: the event used a full build. NOT excluded: other geometry
  defects of a full BuildParallel; the comparable total face area and the all-local corrections are consistent with a
  sound cell but do not prove geometric closure (low prior, no evidence, untestable with `RICH_VERIFY_PARTIAL_BUILD`).
- Gas dynamics as the cause of the dominant loss: face 11 separates near-identical states, so its 75% is mesh motion.
  Hydrodynamic contributions on the other 14 faces are not excluded; uniform flow across all 15 is approximate.
- Received MPI corrections: 0 in both records.
- Radiation as the direct killer: the abort is the hydro update (mask 16); radiation shaped the bins.

### Open question

Why the pair's offsets grew ~7x between t = 0.141162 and 0.170668. The fix does not depend on it (F1 bounds the
consequence whatever sets `|dw|`); the cost may. The trace can separate generator overshoot from centroid motion over
the intervals it records; it prints only at events whose mesh holds the traced cell as owned, so the history has gaps
while the cell sits outside a partial target. Completeness is checked first (every activation of 50107 and 717470
between 0.141162 and the abort/0.180 present, via consecutive `primitive_tick` values), and neighbour-driven growth
stays "inconclusive" unless the neighbours' recorded positions cover the same intervals.

## 2. Experiments (to authorize)

### Binary: `rich_trace_20260923`

**new** Implemented in source, not built: `RICH_INDIVIDUAL_TRACE_CELL_IDS=<id,id,...>` (default unset = off, value
must agree on all ranks; `hdsim_3d.cpp`, 236 added lines, 0 removed, syntax-only compile with the production flags
clean). At every event whose mesh holds a listed cell as owned, rank 0 prints (MPI_Gather/Gatherv, collective on
every rank while the list is non-empty):

- `INDIVIDUAL_CELL_TRACE`: event and previous tick, event_time, cell_id, rank, active, bin, cell_dt, primitive_tick,
  generator r, centroid, width, volume, d/width, gas v, `w_closed` (the velocity that moved the generator over the
  interval closing here), `w_next` (the velocity installed now);
- `INDIVIDUAL_CELL_TRACE_FACE` per face: area, neighbour id and owner rank, neighbour r and installed w, separation,
  relative speed, approach speed `(w_i - w_j).(r_j - r_i)/|r_j - r_i|`, or `boundary=1`;
- `INDIVIDUAL_CELL_TRACE_LIMITS` (from `suggestIndividualTimeSteps`, active traced cells): closed interval, hydro
  limit, drift timescale `min_j |r_i - r_j| / |w_i - w_j|` whatever the fraction (so E0 shows where f would have bound),
  the drift fraction, mass- and thermal-loss limits, and the hydro step's final limit. The assigned interval shows up
  as `bin`/`cell_dt` in the cell's next `INDIVIDUAL_CELL_TRACE`.

No physics change; the cost when unset is one empty-vector test per event and no trace collective. Round-1 astra
xhigh review of the trace: collective-safe (both hooks reached on every rank at every completed event, including both
first-half paths; the aborting event itself prints neither half), fields correct, no state mutation; one MAJOR fixed
(neighbour owner/ID for same-rank halo cells outside a partial target now via `meshLocalToGlobal`), finite-loss
checks mirrored from the limiter.

**new** The same build carries the S5 wake-limiter profile (`Simulation.cpp`, 60 added lines, 0 removed; syntax-only
compile clean): `limitIndividualTreeWakeTimeSteps` records, at every event, this rank's seconds in deadline
preparation, source states (centroids and EOS), tree build (local tree, node summaries, distributed tree), local
evaluation with destination routing, the sparse exchange (including waiting for the slowest rank) and evaluation of
received sources, plus counts of signal sources and outgoing/incoming records. Published as `RICH_STEP_DETAIL
phase=individual-wake-{prepare,sources,tree,local-route,exchange,remote}` (seconds) and
`individual-wake-{signal-sources,outgoing-records,incoming-records}` (count) under `RICH_INDIVIDUAL_PERF_TRACE=1`;
nested inside wake_s + sweep_s, not subtracted from other_s. Source reading behind it: the non-sweep path uses the
same query radius as the sweep, `max_remaining_sleep x (cs + global max cs + |v| + global max |v|)` clamped to the
domain diagonal, so pre-pericentre every signal source may be sent to every rank; the profile tests that, and S6s's
per-destination radius would then address S5 too.

Build command (to authorize):
`build_rich.sh intelReleaseMPI --test_name=BaseTDEComptonIndividual`, then copy to the run directory as
`rich_trace_20260923`, record `sha256sum`, `ldd` (Intel + OpenMPI 4.1.6) and a provenance file. It contains the
instrumentation of `rich_instr_b_20260923` (exclusive scheduler timers) because it is built from the same tree.
Fallback if the build is refused: E0/E1/E2 on `rich_instr_b_20260923` without the trace; the H3 question then stays open.

### Submission

- Script: session scratchpad `submit_crash.sh`, copied into `runs/M05R05MBH1e4MGComptonIndividual/` (to authorize).
  Same guards as `submit_step1.sh` (0.25 / 1.0 / 0.5, adaptive 0, PERF_TRACE 1); the drift fraction, the fold and the
  trace list come from the environment with the crashing values as defaults; `RICH_BINARY`, `RICH_TDE_FINAL_TIME`
  and `RICH_TDE_RUN_DIRECTORY` are required; refuses production and the source checkpoint directory; echoes every
  effective setting and the binary's sha256; scans every allocated node's top processes (`NODE_SCAN`) before `mpirun`;
  `#SBATCH --exclude=d25g2,d25g10`.
- Every arm sets its values in the **submitting process** and uses plain `--export=ALL` (sbatch splits `--export` on
  commas, which breaks the trace list, and the round-2 review flagged inherited-value ambiguity):
  `env RICH_BINARY=./rich_trace_20260923 RICH_TDE_FINAL_TIME=... RICH_TDE_RUN_DIRECTORY=... RICH_INDIVIDUAL_MESH_DRIFT_FRACTION=... RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS=... RICH_INDIVIDUAL_TRACE_CELL_IDS=50107,717470 sbatch --export=ALL --partition=<P> --constraint=<C> --nodes=16 --ntasks=256 --ntasks-per-node=16 --exclusive --time=<T> --job-name=TDE_<ARM> --output=exp_<ARM>_%j.txt --error=exp_<ARM>err_%j.txt submit_crash.sh`
  (the job-name/output flags override the script header). The startup echo must show every arm value, and the
  `cell_id=` fields of the first `INDIVIDUAL_CELL_TRACE` lines must be 50107 and 717470, before any result is read.
- Private restart copies (session scratchpad `stage_restart_copy.sh <ARM>`, to authorize): counter.txt (19),
  gravity.txt (0), snap_19.h5, individual_restart.h5, individual_restart/ (256 rank files; ~1.05 GB) into
  `/data/users/elads/TDE_crash_<ARM>/R0.47M0.5BH10000beta1S50n1.5Compton`, with a sha256 manifest of all 260 files in
  the parent, verified after the copy. **`RICH_TDE_RUN_DIRECTORY` is the parent** `/data/users/elads/TDE_crash_<ARM>`
  (the driver appends the run-name leaf, `test.cpp:1296`). Before `mpirun` the submit script resolves that path with
  `realpath -m`, refuses production and the source directory, requires counter 19 / gravity 0 in the leaf and a clean
  manifest check (else exit 3: without counter.txt the driver would silently start fresh), and aborts on a failed or
  incomplete node scan (exit 4) or any process above 50% CPU on an allocated node (exit 5). Tested on a fake tree:
  correct parent passes; passing the leaf, a corrupted rank file, a `..` alias of the source, or the production path
  all refuse.
- Verdicts (session scratchpad `crash_gate.py`, 14 tests in `test_crash_gate.py` pass, including the real logs):
  `crash_gate.py robust LOG --err ERR --stage early|final --t T --expect KEY=VALUE ... --trace-ids 50107,717470
  [--require-guard mesh_drift_limited]` -> PASS / FAIL / INCONCLUSIVE (exit 0/1/2): FAIL on any abort, non-individual
  mode, mode switch, effective-setting mismatch against the startup echo, missing or foreign traced IDs, or a required
  guard never > 0; `final` needs `Done sim`; not reaching T without a FAIL is INCONCLUSIVE (running or time limit).
  `crash_gate.py cost E1LOG E0LOG --lo 0.141161734618 --hi 0.1750` -> ACCEPT / REJECT / INVALID: each window must be
  ordered, contiguous, start at lo and cover >= 90%; rates divide by summed event durations; ACCEPT iff both ratios are
  within 1.25 (step_s) and 1.5 (events). Check on existing logs: restart 10200535 vs fresh 10200287 over that window,
  85 vs 78 events (1.09x) and 4167 vs 890 s of steps (4.68x) -> REJECT, the orphan-node run.

### Arms

| Arm | drift | fold | trace | stop | Slurm time | Separates |
|---|---|---|---|---|---|---|
| E0 control | 0 | 0 | 50107,717470 | 0.180 | 1 h | reproduction on this binary/partition; cost baseline; unbounded H3 history |
| E1 = G1 | 0.2 | 0 | 50107,717470 | 0.46 | 14 h | H1/F1 (early verdict at t >= 0.180); robustness gate to 0.46; bounded history |
| E2 | 0 | 1 | 50107,717470 | 0.180 | 3 h | H1b from this checkpoint |

E1 and G1 are one run (round-1 review: account for runs): the E1 verdict is read from its running log once
`t_end >= 0.180`, and the same job continues to the G1 endpoint. E0/E1/E2 are independent and can run concurrently
(3 x 16 nodes); E0' (the old `rich_step1_20260922` in E0's settings) only if E0 does not reproduce. Wall estimates:
E0 ~15-25 min on clean nodes (from 914 s of steps in 10200287); E2 unknown, fold-on had 967 events and 3989 s of steps
in the same window on another trajectory and partition (10199059); E1 to 0.46: estimate ~2 h at step 1's
post-pericentre pace (21219 s of steps per unit t on d25g x 0.319) times F1's unknown cost factor, hence 14 h. A time-limit stop before the endpoint is INCONCLUSIVE, not a fail.

### Outcomes

Gate invocations (G = session scratchpad `crash_gate.py`; `X=exp_<ARM>_<job>.txt`, `XE=exp_<ARM>err_<job>.txt`):

| Arm | Command | Expected |
|---|---|---|
| E0 | `G robust X --err XE --stage early --t 0.180 --expect RICH_INDIVIDUAL_MESH_DRIFT_FRACTION=0 --expect RICH_INDIVIDUAL_RADIATION_RETRY_LIMITS_BINS=0 --trace-ids 50107,717470` | FAIL with first_abort naming cell 50107, plus the `individual_hydro_invalid_mass.*:18210661335040:50107.*` record |
| E1 early | same with `--expect ...DRIFT_FRACTION=0.2 --require-guard mesh_drift_limited` | PASS |
| G1 | E1's command with `--stage final --t 0.46` | PASS |
| E2 | `--expect ...DRIFT_FRACTION=0 --expect ...RETRY_LIMITS_BINS=1`, `--stage early --t 0.180` | PASS or FAIL, both informative |
| cost | `G cost exp_E1_*.txt exp_E0_*.txt --lo 0.141161734618 --hi 0.1750` | ACCEPT required to ship F1 as default |


- **E0.** Reproduces (`produced an invalid state` for cell 50107 at event_time 0.177068 in `exp_E0err_*`, record
  `individual_hydro_invalid_mass.*:18210661335040:50107.*`): the trace binary stands in for `rich_step1_20260922`;
  its trace between 0.141162 and 0.177068 is the H3 evidence (d/width per activation, `w_closed` vs `w_next`, centroid
  vs generator displacement between activations, neighbour approach speeds, drift timescale vs hydro limit).
  Does not reproduce, or dies elsewhere: inconclusive between binary and hardware (determinism is physics-level, not
  bitwise); attribution for E1/E2 is void until E0' separates the two.
- **E1 robustness (decides correctness, independent of mechanism):** PASS = zero `produced an invalid state` in
  `exp_E1err_*`; last `t_end >= 0.180` (early verdict) and later `Done sim` at 0.46 (G1); startup echo shows drift 0.2,
  fold 0; `mesh_drift_limited` lines present (aggregate evidence the guard is live). FAIL at any tick: the new record
  and the trace are the next evidence; if the pair was never limited despite a small drift timescale, the velocity
  visibility reading above is wrong; if it was limited, f = 0.2 is insufficient or another mechanism acts.
- **E1 attribution (separate; does not gate robustness):** F1 may act earlier than 0.170668 and change the pair's
  trajectory, so no specific tick is required. From the trace across 0.141162..0.180: when the pair's drift limit first
  binds (`0.2 x drift_time < hydro_limit`), whether their d/width and kicks still grow, and whether the pair (or the new
  worst neighbour) ever reaches a relative displacement > 0.2 of its separation in one interval. Reported as support /
  contradiction / inconclusive for H1 and H-cost.
- **E2.** PASS (no abort, `t_end >= 0.180`, echo shows fold 1, drift 0): from this checkpoint the fold alone prevents
  the abort (H1b supported as masking). FAIL at the same interval: the fold does not rescue the state reached by
  t = 0.141162 with R2 off; this does not exclude H1b for a fresh run. Either way R2's robustness gate below stands.

## 3. Fix

**F1 now (in force, not just defaulted), F2 as measured follow-ups, F3 a question, F4 deferred.**

- **F1: the existing drift limit at f = 0.2.** Per-cell runtime criterion that switches both ways; inert unless
  `|w_i - w_j| dt > f |r_i - r_j|` for some face neighbour. Deployment (to authorize, touches the production script):
  (1) code default 0.0 -> 0.2 plus the doc row; (2) delete the hard-set in `submit.sh:53` and `submit_step1.sh:35`;
  (3) a rank-0 startup record of the effective guards, e.g. `INDIVIDUAL_GUARDS mesh_drift_fraction=0.2
  retry_limits_bins=... mass_loss_fraction=... thermal_loss_fraction=... wake_change_fraction=... adaptive_mode=...`,
  required in every gate log. (1)-(3) are one reviewed change after E1's early verdict.
- **F2 (designs to validate, F1 kept meanwhile).** F2a, a GCL-rate bound `dt <= f_V V / |sum_f A_f (w_f . n_f)|`:
  for 50107 `0.25 x 0.0064 / 1.17 ~ 0.0014`, looser than F1's ~0.0011; the net rate cancels under shear and an
  instantaneous rate is not a finite-interval bound, so only as an A/B with its own robustness gate. F2b, the
  passive-neighbour hole (a fine neighbour re-picking its velocity against a coarse passive cell): must be prospective,
  a deadline written into the passive neighbour at the fine cell's activation (the tick at which accumulated relative
  displacement would reach f x separation), in the style of the wake-tree deadlines; not needed for this crash.
- **F3** (kick cap `|dw| dt_cell <= k d`, or c without the bulk speed): the user's optional decision; it contradicts the
  2026-09-18 ruling, and nothing here shows it is necessary (see H-cost).
- **F4** (invalid-state rejection and event retry): there is no hydro rejection path today (the abort is a collective
  `MPI_Allreduce` MIN then throw in `default_extensive_updater.cpp` ~1150-1210); new machinery, off the critical path.

### Gate for F1 (and later F2 variants), same partition for both sides

- Robustness (hard): G1 = E1 continued to 0.46: zero invalid-state aborts, `Done sim`, effective drift 0.2 in the log.
- Cost (hard): E1 vs E0 over the completed `RICH_STEP` blocks with `t_start >= 0.141161734618` and `t_end <= 0.1750`
  (before either arm's abort; the aborted event has no `RICH_STEP` block), each normalised by its own included span:
  events per unit t, summed `step_s` per unit t, `mesh_drift_limited` totals as the explanatory column.
- Bound: step_s per unit t <= 1.25 x E0 and events per unit t <= 1.5 x E0. Basis: the earlier ~20% event cost; the
  individual scheme is ~4.6x slower than global pre-pericentre and S5 targets 55% of that wall, so > 25% of wall would
  consume about half of S5's headroom. Enforcement: over either bound, F1 is recorded as robustness-validated and **not
  shipped as default** until F2a/F3 bring it under the bound or the user sets a revised budget knowing the alternative
  default aborts.
- Scope note: this cost window is post-pericentre only; F1's pre-pericentre cost is measured by G2 below.

## 4. Revised order

1. **Crash, cluster (a):** build the trace binary -> three copies -> E0, E1(=G1), E2 concurrently; E0' only if needed.
2. **Crash, code (a), parallel:** F1 deployment change (default, hard-set removal, `INDIVIDUAL_GUARDS` record) written
   and reviewed while E1 runs; F2a/F2b designs behind default-off switches later.
3. **S5 (b):** instrumentation written (see §2, in the same build); the 55% was measured pre-pericentre (probe-c,
   -1.519..-1.02), so the profile that decides the optimisation comes from G2's first segment (shipped configuration);
   E0/E1/G1 add post-pericentre profiles only.
4. **R1' (c):** written in the same build, default off: `RICH_INDIVIDUAL_RADIATION_INCREMENT_LIMIT=f` (grey
   `Diffusion` only, the solver the TDE uses; `MultigroupDiffusion` untouched). `prestepIndividual` records the
   event-start internal energy per volume beside the existing `individual_event_old_Er`; in
   `calculateIndividualTimeSteps`, for each active cell with both event-start budgets positive and finite,
   `r = max(|e_int - e_int0| / e_int0, |E_r - E_r0| / E_r0)` (net change over the event, the baselines stay fixed
   across fractional retry candidates) and the limit is `applied_dt x f / r`, shortening only; rank-0
   `INDIVIDUAL_RADIATION_INCREMENT_LIMITED` (count, tightest ratio, one example cell) for every event in which it shortened
   at least one limit. 91 added lines, 0
   removed; syntax-only compile clean. D5's value is f = 0.15. Its runtime gate (the D5 probe, -1.519..-1.02 with the
   switch on: invalid-energy retries per unit t <= 64 at events per unit t <= 3214) follows G2's configuration.
5. **S6s (d), code now;** its verify/performance probes (-1.519..-1.02) follow the S5 profile.
6. **R2 (e), last.** The approved D4 gates (decision document 2026-09-22) are retained unchanged and robustness is
   added:
   - The acceptance record is **G2**: a fresh run with drift 0.2, fold 0, adaptive 0, from t = -1.519 to 0.46
     (splice-aware continuation as in D1), which must pass **all of D4(a)-(c) on its own spliced record**, with the
     approved additions and synthetic tests of D4(b): `step1_analyze.py --gate` (Done sim, zero `RICH_MODE_SWITCH`, all
     `mode=individual`, splice joins, contained span >= 90%, matched window 0.375268-0.440268 retries per unit t <=
     119/0.062207243769 = the **global** run's rate, zero invalid-energy retries in the window); D4(c)(i) zero errors
     or aborts over the whole run; D4(c)(ii) retry wall <= 1% of step wall over -1.519..0 (`--retry-wall`). The step-1
     run's 0.20% (47.4 s of 24120 s) was drift-off and does not count for the shipped configuration. The comparator
     was never arm D (itself R2-off), so no new comparator arm is needed. This is additional analysis of G2, not an
     additional simulation.
   - G1 (E1 continued) is supporting evidence only (robustness from the known-bad state; it inherits the drift-off
     prehistory before 0.141162), not the D4 record.
   - G2 also yields the S5 pre-pericentre profile and F1's pre-pericentre cost. It needs its own authorization
     (~14 h per segment, two segments; wall unknown).
   - R2 ships only after G2 passes D4(a)-(c) and G1 passes robustness; E2's outcome changes the interpretation,
     not the requirement.

Independent code items (2-5) proceed now; cluster items wait on capacity (§0).

## 5. Decisions for the user

1. **Capacity:** the E-runs need 3 x 16 nodes on one partition; bigrun is at your cpu=1024 limit with the tops20
   arrays and core/socket have no idle d26g nodes. Wait for tops20, or pick a partition and accept pending.
2. **Authorizations:** build `rich_trace_20260923`; copy `submit_crash.sh` into the run directory; three restart copies
   (~3.2 GB under /data/users/elads/TDE_crash_*); submit E0 (1 h), E1=G1 (14 h), E2 (3 h); later G2.
3. **Cost budget:** 1.25x wall / 1.5x events (E1 vs E0, 0.141162-0.175), or another number.
4. **RoundCells (F3):** keep the 2026-09-18 ruling, or allow a kick-side cap prototype behind a default-off switch.
5. **Production script:** permission to remove `export RICH_INDIVIDUAL_MESH_DRIFT_FRACTION=0` from `submit.sh:53` when F1
   ships (changes the production run's configuration).
6. **Orphans:** pid 981123 (`rich_intelReleaseMPI`, yours, 33 days, d25g2) slowed 10200535 4.5x and sits on your
   tops20 node 10200615_0 now; maorm's processes on d25g10 are the admins' matter.

- **Fixed-box slowdown and box growth (2026-09-24):** both TDE drivers keep a fixed +-5 rigid box; gas above 1e-10
  first reaches the walls between snap_24 (t=0.5242) and snap_25 (t=0.6565); by t~1.9 corner slivers of width ~1e-4
  (r = 5 sqrt 3) pin dt ~3e-5 identically in the global and adaptive runs (FG2 jobs 10204588/10204589, cancelled at
  t~1.97). The user chose box growth in global mode from snap_24: `RICH_TDE_UPDATE_BOX=1` in both drivers,
  `RICH_TDE_RESTART_FROM_SNAPSHOT=1` in the individual driver, and an FMM fix (a retained root must match the domain's
  lattice id; regression scenario `domain_growth`, 1 with the fix, 0 without). Jobs 10204809 (GLOBAL,
  rich_global_ub_20260924) and 10204876 (ADAPT, rich_ub2_20260924) restarted from snap_24 (identical in all 8195
  datasets): the box grew at cycles 1603 and 1610 identically (x to [-8.14, 6.5], 1.75M -> 1.78M cells), dt stays
  ~2e-3, logged steps identical over the common prefix. The first ADAPT submission (10204810) aborted on the driver's
  checkpoint-existence check, which preceded the snapshot-restart switch; fixed.

## 6. Decisions for the user (2026-09-24, after the overnight runs)

1. **Production binary:** move the production TDE from `rich_fmm_throttle_20260920` (no adaptive controller) to
   `rich_p1_20260923` or later once the full-gravity gate (jobs 10204588/10204589 and the follow-ups) passes; with
   `RICH_INDIVIDUAL_ADAPTIVE_MODE` unset the controller is on and a fresh run starts global (P2: parity, 0.99x wall).
   Existing production restarts keep their checkpoint's integration mode.
2. **submit.sh:53** `export RICH_INDIVIDUAL_MESH_DRIFT_FRACTION=0` disables the drift limit that fixed the crash
   (code default 0.2, G2 accepted with it); remove the line.
3. **submit.sh:45** `export RICH_INDIVIDUAL_WAKE_CHANGE_FRACTION=1.0` is permissive: passive exterior cells change
   velocity without being woken; the code default is 0.25 (P3: 705 -> 381 global steps after an individual start).
   Only matters for runs that step individually.
4. **Snapshot VTU files:** `RICH_TDE_WRITE_VTU=0` saves about 13.5 s per output (about 7% of a run to t=0.46) and
   2.1 GB of 3.5 GB per two outputs; keep the default if ParaView reads them.
5. **Snapshot writing:** the single-file HDF5 snapshot path takes ~11 s per output; parallel or distributed HDF5
   output is a candidate with unmeasured savings (per-rank files would change what readers expect).
6. **Individual-mode exterior perturbation:** variable-bin evolution perturbs the near-vacuum shell after an
   individual start (later runaways to ~c); the recorded lever (bounding RoundCells kicks) is excluded by the
   2026-09-18 RoundCells ruling. Revisit the ruling, or keep individual stepping out of this regime (the adaptive
   default does so here).
7. **Nodes:** d25g10 still runs maorm's two orphaned processes; d25g58 fails srun authentication; d25g65's ssh host
   key changed. Report to the admins; the submit scripts exclude d25g2, d25g10, d25g58.

## Assumptions

The trace build is physics-identical to `rich_step1_20260922` (E0 checks); separation at the interval opening is
unmeasured; sbatch command-line flags override `#SBATCH` headers; wall estimates are estimates.

## Results (2026-09-23, runs from private copies of snap_19; logs in runs/M05R05MBH1e4MGComptonIndividual)

| Arm | Job, nodes | Verdict (`crash_gate.py`) | Key numbers |
|---|---|---|---|
| E0 control (drift 0, fold 0) | 10200855, socket d26g | FAIL (reproduced) | 88 events, abort cell 50107 at tick 18210661335040; crash record byte-identical to 10200535's; guard sums identical (mass_limited 46256, change_wakes 685, thermal_limited 1592); wall 31m42s |
| E1 (drift 0.2, fold 0) | 10200856, socket d26g | early robustness PASS (t >= 0.180); cost ACCEPT | vs E0 over 0.141161735..0.174667985: events 52 vs 85 (0.612x), step_s 1575.8 vs 1681.6 (0.937x), retries 59 vs 63; continues to 0.46 (G1) |
| E2 (drift 0, fold 1) | 10200869, bigrun d25g | FAIL | 548 events, 642 retries, same abort (cell 50107, same tick, same opening tick 18141941858304) |

- **H3 answered (E0 trace):** at t=0.1579 the pair activated with d/width 0.05/0.09 and kicks 0.50/0.87, drift
  timescale 0.0204, and was given bin 37 (0.0128). The held kick overshot the centroid linearly (d/width 0.012 ->
  0.308; a RoundCells gain chi c dt / R ~ 7 predicts (G - 1) x 0.05 = 0.31) while the generators separated
  0.025 -> 0.041; at t=0.1707 the new kicks were 3.08/2.67 toward each other (approach +5.43), drift timescale 0.0041
  against the 0.0064 interval; separation 0.041 -> 0.015, abort at the next event.
- **H1 supported (E1 trace):** at t=0.1579 the drift limit (0.2 x 0.0194 = 0.00388) beat the hydro limit (0.0158) and
  the pair got bin 35; max d/width 0.055/0.092, kicks <= 0.91, relative speed <= 1.30, separation 0.025-0.029; the
  limit relaxed and the pair returned to bin 36 by t=0.1643 (H-cost: no persistent cost for this pair).
- **H1b contradicted from this checkpoint (E2):** folding radiation retries into hydro bins did not lower this pair's
  bins (same bin 37 at 0.1579, same kicks, same abort) although it produced 6.2x E0's events. The fold-on fresh run
  10199059's pass came from its different trajectory. The mesh-drift limit is needed with R2 on or off; R2's own
  gate (G2) is unaffected by this result.
- **S5 first profile (E1, post-pericentre, 93 events):** wake_s 891 s of 2035 s step_s (44%); summed over events,
  exchange 751 s on the median rank (waiting) vs local-route 855 s on the slowest rank (130 s median); signal sources
  per rank median 71,778 max 233,770; outgoing records median 8.8M max 42.6M; tree 7 s, remote 1 s.
- **G1 (E1 continued, job 10200856 COMPLETED 2h22m):** `crash_gate.py robust --stage final --t 0.46` PASS (542
  events, `Done sim`, no fatal line, mesh_drift_limited 15,939,366 cell-limitations). Supporting evidence only (drift-off
  prehistory before 0.141162): `step1_analyze.py --gate` PASS on its log, matched window 0.375268-0.440268 retries per
  unit t 1389 <= 1913 (global), 0 invalid-energy, wall 17,373 s per unit t (d26g); arm D (R2 off, drift 0, d26g,
  different history) had 2108 retries and 36,205 s per unit t there.
- **Pre-pericentre cost of F1 (G2 in progress vs step 1, both fresh from t=-1.519 on bigrun d25g):** over
  t=-1.518932..-0.272532, events 3758 vs 6004 (0.626x), summed step_s 12,424 vs 16,737 s (0.742x), retries 324 vs 312.
- **Regression gate (THUNDER on the F1 source):** every individual-timestep case that ran passes with drift 0.2
  (amr_random, marshak_wave_1..4_diffusion serial gnu; lane_self_gravity, mach2_diffusion, mach2_multigroup MPI intel).
  Remaining failures are outside the code paths this work changes: multi-node ORTE start-up failures on bigrun
  (rayleigh_taylor_mpi, spherical_collapse, moving_slab), eulerian_diffusion_freefree_* UniversalError in global mode,
  fmm_gravity_mpi self-check exit 1, desmore2012_mc segfault (campaigns 20260923_115005, _132011, _143705).
- **G2 (R2 acceptance record, job 10202681 COMPLETED 0:0, 8h00m, 5628 events, `Done sim` at t=0.460268):**
  `step1_analyze.py --gate` PASS on every predicate (matched window 0.375268-0.440268: retries per unit t 1389 <= 1913,
  0 invalid-energy; contained span 99.7%) -> R2 default flip allowed; `--retry-wall -1.519 0` 48.9 s / 18,470 s = 0.26%
  PASS; `crash_gate.py robust --stage final --t 0.46` PASS; `INDIVIDUAL_GUARDS mesh_drift_fraction=0.2`. Summed step_s
  28,102 s against the global driver's 3,780 s over the same interval (different d25g nodes). G2, `TDE_s5probe` and
  `TDE_cadence_probe` (three binaries) have identical conserved totals at t=-1.468932015 and -1.252532015.
- **S7 (adaptive controller, job 10204483, 1999 steps, 4351 s summed step_s vs the global driver 10204361's 1564 steps,
  3780 s):** one switch individual->global at cycle 77 with bit-identical before/after state; every later decision
  `stay_global_little_to_gain` with gain_bound=1. The excess is workload, not hardware or per-step cost: S7 needed 1760
  steps to t=0 against 1310 (10204361) and 1318 (older global run 10199025, which matches 10204361 for 200 steps and
  first differs by 2.04e-11 absolute in event_dt), with 69 hydro-internal dt cuts against 20/17 and 7178 radiation
  solves against 5891. Candidate causes (the 77-event individual start, the driver's early-output clamp, build
  differences, controller timing) are separated by the S-B discriminator below. Twin clause of gate (b): UNEVALUATED
  (no common final output; only t=-1.468932015 is common: mass 2.2e-16, gas+radiation energy 6.9e-6 against an
  unmeasured 1e-6 criterion). Tracing overhead unmeasured.
- **Controller finding:** in global mode `adaptiveGainBound` caps every cell's hydro limit with radiation's single global
  dt (`Diffusion::calculate_dt`, MAXLOC of the relative radiation-energy change), so while that dt binds the bound is
  exactly 1 and the controller never probes individual stepping; the bound cannot see per-cell radiation limits.
- **S-B discriminator (job 10204569, 2026-09-23):** with the early-output policy matched, the individual driver
  started global (controller off, B2; on, B3) logged the global driver's (B1) (t_start, event_dt) sequence exactly
  over all 377 steps to t=-1.20; the S7 configuration (B4) reproduced S7 exactly (782 steps). The whole S7 excess
  therefore comes from the 77-event individual start-up path. At the common snap_0 the individual-start state
  differs mainly in the near-vacuum shell outside the star (faster, hotter, more extended), the bulk agrees.
- **P1, production-safe global start (astra rounds 1-2 APPROVE; binary rich_p1_20260923, verification job 10204574):**
  a fresh run with the controller on starts global (RICH_TDE_START_MODE overrides; restarts keep the checkpoint
  mode), controller decisions on global steps wait for the early snapshot (SetAdaptiveDecisionsDeferred), the
  centre sink and full-gravity AMR run after global steps inside the timed step (SetGlobalPostStep). V1 (default)
  logged B1's sequence over 377 steps with its first decision at cycle 87 (after snap_0); V2 (forced probe) entered
  individual after the snapshot and reverted, switch states bit-identical; V3 (controller off) = G2's sequence;
  V3R/V4 restarts kept the checkpoint mode against an opposite START_MODE; V5 (individual start) = S7's sequence with
  its global-mode decision deferred to cycle 145; V6 (global, controller off) = B1. Open production gate: the
  full-gravity sink/AMR path (t > 5), jobs 10204588/10204589.
- **P3 (job 10204575):** in the S7 configuration the conserved-change wake at the code default 0.25 cut the
  global steps after the 77-event start from 705 to 381 (to t=-1.20; global start: 377) but left the snap_0 surface
  discrepancy nearly unchanged. Production submit.sh:45 exports RICH_INDIVIDUAL_WAKE_CHANGE_FRACTION=1.0.
- **dt-collapse trace (RICH_CFL_DECISION_TRACE, astra rounds 1-4; job 10204578):** neutral (logged sequences and
  bitwise snap_0/snap_1 fields identical to the references). Most global steps are bound by the cap from the
  previous suggestion. The collapse after the individual start (onset cycle 257, t=-1.38049, dt 9.1e-4 -> 1.5e-4,
  then 8.3e-6 at cycle 281 and 5.6e-6 at 282) is set by the raw CFL of cells in the near-vacuum shell (density at
  or below 1e-14, r 0.52-0.55 at snap_0): relative fluid-face normal speeds up to 370 and a cell reaching |v| ~ 693
  (about the speed of light in code units, outside the model's validity). At snap_0 these cells are 4.5-9x faster
  in both individual starts than after a global start; with wake 1.0 two of them also hold 3-4x (T1/T2 4.3 and 3.3)
  more radiation energy. The responsible update is not identified (hypothesis: individual-start perturbation of
  the exterior plus radiation build-up in passive cells under the permissive wake).
- **P2 full-length paired timing (jobs 10204576 GA, 10204577 AG; astra: gate passes):** global driver
  (rich_global_p1_20260923, early-output clamp) against the production-default adaptive run (rich_p1_20260923), fresh
  t=-1.519 -> 0.46, both orders on scanned nodes. Enclosing wall ADAPT/GLOBAL = 4377.8/4413.1 s = 0.992 (GA) and
  4384.0/4427.0 s = 0.990 (AG); summed step_s 0.5-0.8% lower. All four arms log the same 1567-step (t_start, event_dt)
  sequence and 1083 retries; the 12 checked fields are bitwise equal at snap_0, snap_12 and snap_23 (t=0.412) for
  ADAPT vs GLOBAL in each job and ADAPT vs ADAPT across jobs. The controller made 4 stay-global decisions per run
  (gain_bound=1) and never switched. Parity, marginally faster in these measurements; this validates the controller's
  global operating path, not individual-mode entry. Non-step time (initialisation plus 24 numbered outputs) is 605-621 s
  (13.8-14.1% of wall), a substantial shared target with snapshot output the leading suspected contributor.
- **Individual-scheme screen (job 10204662, rich_scheme_20260924; driver switch `RICH_TDE_INDIVIDUAL_SCHEME`):** pure
  individual start (adaptive off), wake 1.0 and the G2 guards, to t=-1.46. The partial control reproduced V3's event
  schedule (269 events); the synchronized arm (`full`, one shared bin) was all-active in all 151 events. At snap_0
  (t=-1.468932015) the four exterior cells moved at 0.03-0.06 after a global start, 0.04-0.14 with one shared bin,
  and 0.22-0.38 with variable bins on either full (`full-variable`) or AutoPartial meshes; atmosphere extent 0.538 /
  0.519 / ~0.56 and speed p99 0.139 / 0.136 / ~0.27; cells differing from the global start by >10% in density
  0 / 88k / ~259k. For these diagnostics and settings, variable-bin evolution contributes most of the exterior
  discrepancy; the mesh policy has a much smaller effect on velocity, extent and density counts, while the atmosphere
  temperature tail remains sensitive to it (T p99 1.53e5 full-variable vs 8.0e4 partial). One shared bin is still
  not the global integrator (two cells 2-3x faster, 88k cells >10%). Consistent with, not proof of, the recorded
  mechanism class (coarse-interval fluxes paired with end-of-interval point velocities; passive accumulation). The
  adaptive default does not take the individual start (V1, P2); pure individual TDE runs in this configuration show
  the perturbation over the tested early evolution; the recorded lever (bounding RoundCells kicks) is excluded by the
  user's RoundCells ruling, so no code change.
- **Output-cost pilot (job 10204637, rich_vtu_20260924; astra APPROVE with wording):** the TDE driver gains an opt-in
  `RICH_TDE_WRITE_VTU=0` (default: VTU on, unchanged) and a rank-0 `RICH_OUTPUT` line per numbered output. Arms VTU
  on / off / on (V1 configuration to -1.24): snapshot write 24.3-25.4 s with VTU against 11.2-11.3 s without (mean
  13.5 s, 54.6% saved per output), enclosing wall 28-30 s lower for two outputs, logged step sequence and all 8195
  HDF5 datasets unchanged, run directory 1.4 GB instead of 3.5 GB. For 24 comparable outputs to t=0.46 this
  extrapolates to about 324 s (7.4% of the ~4400 s wall). The remaining VTU-off snapshot path takes ~11.2 s per
  output; distributed or parallel HDF5 output is a candidate to investigate, with unmeasured savings (per-rank files
  would change what readers expect; parallel writing to one file might keep the layout).
- **Exterior-cell histories (RICH_INDIVIDUAL_TRACE_CELL_IDS, job 10204590):** the four cells behind T1's collapse are
  exterior cells whose generators move at 0.24-0.46 while their gas is at rest. Under wake 1.0 two passive ones change
  velocity (~0.0007 -> 0.02) without being woken; under wake 0.25 they do not. The active cell 676592 accelerates
  alike in both (0.13 at t=-1.514132). Two of the four are nearly at rest through the whole individual phase yet
  reach 0.22 and 0.41 by snap_0 in both individual-start configurations (0.03-0.05 after a global start), so most of
  the exterior perturbation develops after the switch to global, seeded by the state the individual phase leaves.
  Attribution needs per-operator budgets (hydro faces, gravity, radiation, floors) around t=-1.5157..-1.5141 and
  across the switch; open.
- **S-B discriminator design (astra rounds 1-4, 2026-09-23):** `RICH_TDE_START_MODE=individual|global`
  (runs/BaseTDEComptonIndividual, fresh runs only; a global start writes no initial file) and
  `RICH_TDE_EARLY_OUTPUT_CLAMP=0|1` (runs/BaseTDEComptonGlobal, default 0 = historical). One job, four arms to t=-1.20
  on the same 16 nodes with tracing off: B1 global driver with the clamp, B2 individual driver started global with the
  controller off, B3 started global with the controller on (valid only if its first decision stays global before the
  early output), B4 the S7 configuration. Readouts: first differing (t_start, event_dt), decision records, per-window
  steps/retries/cuts, enclosing wall per arm.
