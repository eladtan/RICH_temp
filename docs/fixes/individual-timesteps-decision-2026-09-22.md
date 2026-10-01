# Decision 2026-09-22

Written 2026-09-22 evening from the step-1 state at 20:36 (t=-0.197, 6568 events, 323 retries, 0 mode switches, 0 errors). All run numbers are the human session's script results; line numbers cited below were read directly this session (`step1_analyze.py` and `Simulation.cpp:3525-3619` re-read this round).

**D1 — Continuation: submit a dependent continuation tonight with `afternotok`, as unattended recovery capacity, not as a guarantee.**
Basis: pericentre ~22:05; the only post-pericentre individual pace measured (arm D, 36205 s/unit t over 0.375-0.440, d26g) gives 0.46 x 36205 = 4.6 h, finish ~02:40, 2.5 h before the 05:16 limit. That margin vanishes if the never-timed 0-0.375 stretch runs 55% slower than arm D (arm A ran 53032 s/unit t), and d25g is unverified against the d26g pace. The human is asleep 23:00-08:00, so "wait and decide near the limit" is not available. A time-limit kill without a continuation would repeat arm D's failure (no completion marker) and waste the run's only R2-gate window.
Why `afternotok`: the driver writes no final checkpoint at tf (`test.cpp:1733-1884`, the loop exits straight to "Done sim"), so an `afterany` continuation after a normal completion would restart from the last ~45-min checkpoint, re-run the tail and overwrite later `snap_N`/`counter.txt`. `afternotok` fires only on a failed or timed-out end; after a clean completion it stays pending as DependencyNeverSatisfied and is cancelled in the morning at zero cost.
Restart correctness: `restart = !start_fresh && counter.txt exists`, counter >= 0 required (`test.cpp:1338-1348`), `individual_restart.h5` read at 1611, cycle counter restored (`read_simulation.cpp:61-63`), so continuation logs have monotone cycles. The span between checkpoint and kill (<= ~45 min) is replayed with the same cycle numbers; the existing splice in the gate script handles that (D4).
Expected effect: if 10200287 completes, nothing happens. If it times out, the continuation resumes from the last checkpoint without human presence; estimated completion is queue wait + startup + (remaining span at the then-measured pace), plausibly before 08:00 but dependent on queue state, the unmeasured 0-0.375 pace, and no further failure. It converts "certain loss of the gate window" into "probable completion", nothing stronger.
Gate for the step-1 result itself: D4.
Authorize: command 1 below, tonight, before 23:00.

**D2 — Plan order after this run: S5 instrumentation first, in its own behaviour-neutral build with exclusive accounting; R1 moves to after D5's measurement, not into the same build.**
Basis: untimed remainder is ~8700 s of 18751 s (~47%) pre-pericentre, concentrated in events with 1e5-1e6 active cells (617 events, 6174 s, median 7.5 s, p90 28 s) and >1e6 (266 events, 3095 s); mesh, source terms and sweeps are excluded by measurement. The untimed code is the event collective + prepareEvent + closure (`Simulation.cpp:3532-3619`: prepareEvent at 3533/3537/3612, the local active-state traversal at 3546-3561, closure all-reduces at 3562-3565 and 3583), the wake limiter outside its sweep (`3777-3784`), and commitEvent (`3786-3788`); the suggest loop (`3764-3775`) is added into the physics names at `3773-3774`, so it is timed but hidden inside hydro_s/radiation_s. The plan ranked S5 third on arm D's 699 s of 2353 s; pre-pericentre it is the largest cost, and the rule is measure before fixing.
Change (all per-rank elapsed, then MPI_MAX like the existing fields; emitted on the RICH_STEP `phases`/`mesh` lines by `WriteRuntimeStep`, `1733-1746`), every section exclusive of the others: `sync_s` (event collective all-reduce + the closure all-reduces at 3562-3565/3583), `prepare_s` (all prepareEvent calls, 3533/3537/3612), `closure_s` (local closure work: 3541-3561 and 3566-3613 minus prepare_s, minus sync_s, minus the latch-release loop 3578-3579 which is already reported as `topology_release_seconds_max`), `suggest_s` (the 3764-3775 loop, moved OUT of the physics names so hydro_s/radiation_s become exclusive), `wake_s` (limitIndividualTreeWakeTimeSteps minus the sweep, which needs a `seconds_local` next to `seconds_max` in `IndividualFullSourceSweepReport`, `269-280`), `sweep_s` (that local sweep time), `commit_s`, `full_builds=`. Plus `other_s` = per-rank (step wall minus the sum of ALL per-rank exclusive sections incl. hydro/gravity/radiation/amr/active-hilbert/load-balance), reduced rank-max. No control-flow change.
Why not R1 in the same build: D5 shows R1's cap as written cannot lower a bin a cell already holds; whether it reaches the failing population is unmeasured, and building it now risks a second build. D5's failing-cell record does go into this build.
Probe window: fresh run -1.519 to -1.02 (RICH_TDE_FINAL_TIME=-1.02), new directory, same partition/node type as step-1 (d25g). This is the diagnostic's own window, so every baseline below is already measured on it and needs no extraction: individual step1_10200287.txt pre_early = 1603 events, 184 retries, 3173 s step wall (6362 s/unit t, d25g); global output_10199025.txt pre_early = 64 retries/unit t. At 6362 s/unit t the probe is ~53 min plus startup (2 h limit). It also reproduces the exact population the diagnostic characterized (159 failing cells), which D5's record needs. Run with `RICH_RUNTIME_LOG=detailed` so the failure record at `Diffusion.cpp:1236-1285` is emitted. A shorter probe is not acceptable without first extracting the matching shorter window from both baseline logs (command 2c, `--lo`/`--hi`).
Gate (S5), all on -1.519..-1.02: rank-max `other_s` summed over the probe <= 10% of summed `step_s` (exclusive residual, not a sum of maxima); cadence parity with step-1's pre_early: event count within 1%, retries within Poisson noise of 184, active_bins histograms compared and any difference reported (bitwise identity is expected only if the run is deterministic, which is not yet established, see Open); `step_s`/unit t within run-to-run noise of 6362. hydro_s/radiation_s are NOT compared with step-1 (they lose the suggest share).
Authorize: commands 4-6 tomorrow morning; no code lands tonight.

**D3 — S6 sweep item returns for the pre-pericentre phase as a code change to the exchange, not as an interval widening; its correctness gate is a same-state comparison inside the run, independent of run-to-run determinism.**
Basis: 46 sweeps x median 19 s (max 52 s) = 1040 s = 5.5% of wall, up to 446M MPI source records per sweep; arm D's 0.06 s/sweep post-pericentre is what the plan rejected the item on. Mechanism from source: per-source query radius = `maximum_remaining_sleep x (source c_s + global max c_s + source |v| + global max |v|)`, clamped to the domain diagonal (`Simulation.cpp:517-535`), with sleep and speed maxima as global all-reduces (`746-752`, `866-891`); at the diagonal every source goes to every rank (`985-997`), and 1.75M cells x 255 ranks = 446M records, exactly the measured maximum. Pre-pericentre sleeps reach bin 40 (0.1 units) and speeds near the hole are high, so the sweep saturates at the all-to-all bound; post-pericentre sleeps are short, so it is cheap. The interval (default 128 minimum steps, `IndividualTimeStep.hpp:41`) is a correctness cadence and per the standing rule is not the lever.
Change (S6s, switch `RICH_INDIVIDUAL_WAKE_RANK_LOCAL_RADIUS` = off | on | verify, default off): all-gather per-rank maxima of remaining sleep, sound speed and |v| once per sweep (3 x 256 doubles) and compute the radius per (source, destination rank). A strict superset of each destination's true reach, so fail-closed by construction, and a both-ways runtime criterion: it shrinks the exchange only where the destination has no long sleepers, and does nothing post-pericentre.
Measure first, read-only: from the step-1 log, `mpi_source_records / (sources x 255)` per RICH_FULL_SOURCE_SWEEP record (fields at `3813-3825`); S6s proceeds only if the median is >= 0.5 (saturation confirmed). The probe's sweep record additionally prints per-rank sleep maxima so the per-destination cut can be estimated before coding.
Correctness gate (`verify` mode): on every sweep, both exchanges run on the SAME frozen sweep input (same sources, sleeps, speeds), the full-radius result is the one applied, and the per-cell wake deadline (limited ticks/bin) produced by the rank-local exchange is compared with it for every cell on every rank; mismatches are counted (rank-0 aggregate, one representative), and the gate is zero mismatches over all sweeps of a -1.519..-1.02 probe with verify on. This gate does not depend on the run being deterministic and is not replaced by any statistical comparison.
Performance gate (verify off, same window, separate run): sweep `seconds_max` median down >= 50% and `mpi_source_records` down >= 5x vs the D2 probe; zero INDIVIDUAL_BIN_OVERRUN increase; active_bins histograms reported (identity expected only if deterministic). Ceiling 5.5% of pre-pericentre wall, so it ranks after S5 and R1', before S3.
Authorize: nothing tonight; command 2b (read-only script) tomorrow, code after D2's build.

**D4 — R2 gate on this run: cadence and retry criteria only, on the whole run as one spliced record, using the gate script as it stands, with the listed additions before the final verdict.**
(a) Hardware: the checks in `step1_analyze.py:75-103` (Done sim in the last segment, zero RICH_MODE_SWITCH, all `mode=individual`, splice joins, window coverage, matched-window retries/unit t <= 119/0.062207243769 unrounded, zero invalid-energy retries in the window) are hardware-independent; report this run's wall/unit t but compare no wall column with arm D or d26g global runs.
(b) Restart and window: the splice exists (`gate()` takes a list of logs, `59-73`: cycles `< c1` retained from the earlier segment, its retries matched to retained cycles by (segment, cycle); joins checked for equal t_end/t_start and consecutive cycles, `79-85`; `--gate log1 log2` accepted at `112-113`, `--segments` accepted as a no-op at 114). The window check (`86-97`) now selects all retained steps overlapping [lo,hi], requires first t_start <= lo, last t_end >= hi and contiguous joins, and computes the rate from wholly contained steps over their actual summed span, which is how the global baseline (119 in 29 contained steps, 0.0622 of the 0.065 window) was formed. So a clean single-log completion and a two-segment splice are both evaluated correctly today; the current `--gate` may be run as soon as the job ends and its verdict read as the plan's gate. Outstanding additions (command 2a, each can only tighten, none changes the plan's criteria): require and report contained span >= 90% of hi-lo; require cycles strictly increasing within each segment and across the retained sequence (today only the join is checked); report retries attributed to straddling steps separately (they are correctly excluded from the rate but currently invisible); print "window served by N segments, checkpoint at t=..."; fix the stale docstring at 57 ("tiled by complete retained steps"). Synthetic tests before the re-run: one contiguous sequence straddling both endpoints in one file (PASS), the same split across two files with the replayed cycle duplicated (PASS, identical numbers), a gap inside the window (FAIL), a second file whose first t_start does not meet the retained tail (FAIL), a non-monotone cycle inside a segment (FAIL, new check).
(c) Class: pre-pericentre invalid-energy dominance (369 vs 64, then 168 vs 3 retries/unit t) is R3's finding about cause 1A/1C, not evidence about the fold; R2's flip decision stays on the matched post-pericentre window as agreed. Two no-harm conditions the flip must also meet from this run: (i) zero errors/aborts over the whole run, satisfied at 20:36; (ii) retry wall <= 1% of step wall over the whole pre-pericentre stretch -1.519..0. Condition (ii) is measured only on -1.52..-1.02 so far (28 s of 3173 s, 0.9%) and is UNEVALUATED for the whole stretch until command 3c extracts it with the same method as that 28 s figure. No fold-on pre-pericentre baseline exists; none is invented.
Authorize: command 3a or 3b as soon as the job ends (provisional verdict, current script), command 2a additions with tests, then re-run 3a/3b for the final verdict, then 3c.

**D5 — R1's design: keep the switch and gates, choose the criterion after one measurement; the coupling alternative is specified on the actual net increments.**
Measured: 176 of 184 diagnostic retries are invalid-energy (165 after, 11 during), 7 positivity-floor, 1 Bad_interpolation. Failing cells are radiation-dominated near-vacuum (density 5e-15..2e-6, percentile 0-27, Erad/(rho e) ~1e15) just outside the star; the rejections are the `Erad <= 0 or internal_energy < 0` tests (`Diffusion.cpp:1229-1234` after, `1149-1152` during). What is measured is the EVENT's coarsest active bin (36-40); the failing cell's own bin, applied dt and the active-set volume distribution are not, so nothing here predicts R1 gate (c) either way.
Design fact, independent of that: R1 as planned (`growth_cap = clamp(2/volume_ratio,1,2)` on the `nominal_dt*2.0` term of `487-490`) can only slow growth, never lower a cell below its nominal dt; if the failing cells already hold their nominal bin, R1 cannot act on them. The existing `difference` (`479-482`) divides by `0.02*max_Er`, so these cells' Er changes (~2e-13 per volume) are invisible to the limiter; that is a code fact, the saturation is hypothesis until measured.
Measure (in D2's build, no behaviour change, on the -1.519..-1.02 probe, i.e. the diagnostic's population): for each failing cell, rank-0 aggregate + one representative per event: own bin, `applied_dt`, `V_i`, mean active volume (volume_ratio), pre-update `E_int0=internal_energy`, `Erad0`, net `dE_int` (1147 + relativity 1211), net `dErad` incl. the kinetic term `-new_Ek+old_Ek` (`1218-1226`), `Erad_dE` separately, `e_absorb`, `e_emitt`, `fleck_factor`.
Decision rule: median volume_ratio of failing cells >= 2 and they are below nominal -> R1 as designed. Otherwise R1' = per-cell relative-increment limiter, same switch: during the update store `r_i = max(|dE_int|/E_int0, |dErad|/Erad0)` (extensive code units, both budgets > 0 and finite, else skip the cell; r_i=0 -> no extra limit); in the suggestion, `suggested_dt = min(existing, applied_dt*0.15/r_i)`. It relaxes when coupling weakens (both ways), and can only shorten dt relative to today's suggestion; it does not itself make any state valid, so whether it prevents the failures is what the gate measures, not a property claimed here.
Gates: R1 keeps (a)-(d) plus a pre-pericentre gate on a second -1.519..-1.02 probe with the switch on: invalid-energy retries/unit t <= 64 (global's pre_early total rate, same window) with events/unit t <= 3214 (step-1 pre_early, same window); plus the snapshot-23 post-pericentre probe as agreed. If the R1 probe ever has to be shorter than -1.02, both baselines are re-extracted on the shorter window with command 2c before the gate is applied.
Authorize: the record and its fields in D2's build; the R1/R1' choice only after the probe numbers are in.

## Commands to authorize

Run from `/home/elads/RICH-ablation-integration/runs/M05R05MBH1e4MGComptonIndividual` unless stated. `PY` below is the cluster python with its library path, per the recorded gotcha; substitute whichever `python3` ran `step1_analyze.py` this afternoon if that one works plain:
```
export LD_LIBRARY_PATH=/software/x86_64/5.14.0/python/3.12.1/lib:$LD_LIBRARY_PATH
PY=/software/x86_64/5.14.0/python/3.12.1/bin/python3
SP=/home/elads/mpi_tmp/claude-5014/-home-elads-RICH-ablation-integration/468f34ec-925a-457b-a196-97f747e524a4/scratchpad
```
Nothing here builds or changes source tonight.

1. **Tonight, before 23:00 (D1):**
   ```
   sbatch --dependency=afternotok:10200287 --export=ALL,START_FRESH=0 submit_step1.sh
   ```
   Record the new job id in the handoff. Tomorrow: if 10200287 shows COMPLETED in `sacct`, `scancel <newid>` (it sits as DependencyNeverSatisfied); if TIMEOUT, its log is `step1_<newid>.txt`.

2. **Tomorrow, script work (authorize me to write into `$SP`, then astra reviews before use):**
   - 2a `step1_analyze.py` gate additions per D4(b): contained span >= 90% of hi-lo (required, reported), strictly increasing cycles within and across segments, straddling-step retries reported separately, segment/checkpoint summary line, docstring fix; plus the five synthetic tests listed in D4(b), run from `$SP` (they read no cluster logs).
   - 2b `sweep_saturation.py`: over RICH_FULL_SOURCE_SWEEP records, `mpi_source_records/(sources*255)` median/max and `seconds_max` vs `time` (D3 measurement).
   - 2c already exists: the script's `--lo` and `--hi` options set the matched window used by the summaries and the gate, and `--global-rate` its threshold, for any future probe that cannot cover -1.519..-1.02; no script work needed. Example on the step-1 log for the -1.519..-1.32 window:
     ```
     $PY $SP/step1_analyze.py --lo -1.519 --hi -1.32 step1_10200287.txt
     ```

3. **Tomorrow, read-only, R2 status (D4).** The current script may be run first for a provisional verdict; the final verdict is the re-run after 2a passes its tests.
   - 3a, job completed (one log):
     ```
     $PY $SP/step1_analyze.py --gate step1_10200287.txt
     ```
   - 3b, job timed out (spliced; the script accepts several logs after `--gate`, no extra flag needed):
     ```
     $PY $SP/step1_analyze.py --gate step1_10200287.txt step1_<newid>.txt
     ```
   - 3c, no-harm condition (ii), retry wall / step wall over -1.519..0 (`--retry-wall LO HI` exists in the script: `retry_s` of RICH_RETRY records matched to retained, wholly contained steps, divided by `step_s` of those steps; it reproduces the diagnostic's 28.3 s of 3173.3 s on -1.52..-1.02, and applies the splice rule to several logs). Job completed:
     ```
     $PY $SP/step1_analyze.py --retry-wall -1.519 0 step1_10200287.txt
     ```
     Job timed out (spliced):
     ```
     $PY $SP/step1_analyze.py --retry-wall -1.519 0 step1_10200287.txt step1_<newid>.txt
     ```
     Pass is <= 1%. Until this runs, D4(c)(ii) is unevaluated.

4. **Code change (D2, D5), one commit-sized diff, behaviour-neutral:** RICH_STEP fields `sync_s prepare_s closure_s suggest_s wake_s sweep_s commit_s other_s full_builds` in `Simulation::stepIndividual` around lines 3532-3619 and 3764-3788 (suggest time moved out of the physics names at 3773-3774), `seconds_local` in the sweep report struct at 269-280, emission in `WriteRuntimeStep` (1733-1746), rank-max; failing-cell record with D5's fields in the radiation retry paths at `Diffusion.cpp:1149-1178` and `1236-1290`.

5. **Build and install (repo root):**
   ```
   ./build_rich.sh intelReleaseMPI --test_name=BaseTDEComptonIndividual
   cp -L build/intelReleaseMPI/rich runs/M05R05MBH1e4MGComptonIndividual/rich_instr_20260923
   sha256sum runs/M05R05MBH1e4MGComptonIndividual/rich_instr_20260923
   ```
   Then write `runs/M05R05MBH1e4MGComptonIndividual/rich_instr_20260923.provenance.txt` in the existing five-line form (`rich_step1_20260922.provenance.txt`): binary + build time + command; HEAD e8c6ad9ec plus uncommitted; sha256 (first 16 hex); "= rich_step1_20260922 + S5 timing fields + failing-cell record, no behaviour change"; purpose "plan r2 step 3 instrumentation probe, D2/D5 of decision 2026-09-22". Production symlink `rich` untouched.

6. **Probe (D2, D5), new output directory, -1.519..-1.02, ~1 h:**
   ```
   RICH_BINARY=./rich_instr_20260923 RICH_TDE_FINAL_TIME=-1.02 RICH_TDE_RUN_DIRECTORY=/data/users/elads/TDE_instr_probe RICH_RUNTIME_LOG=detailed sbatch --job-name=TDEinstr --time=02:00:00 submit_step1.sh
   ```
   (fresh start is the script default; `RICH_INDIVIDUAL_ADAPTIVE_MODE=0` and the fold-off switch come from the script, `submit_step1.sh:37-38,49-50`; same partition and node type as 10200287 so the 6362 s/unit t baseline is hardware-matched.)

7. **After the probe:** S5 attribution from the new fields (D2 gate, pre_early window of both logs); D5 decision rule on the failing-cell records; then R1 or R1' implementation and, if 2b confirms saturation, S6s with its verify mode, each behind its switch with the gates above.

## Open

- Post-pericentre pace on d25g for 0-0.375 has never been measured; the 02:40 estimate rests on arm D's 0.375-0.440 window.
- Whether `afternotok` fires on TIMEOUT on this cluster (Slurm documents it does; unverified here) and whether `kill_invalid_depend` is set; if not, `scancel` the continuation after a clean completion.
- Run-to-run determinism of the individual run is not established. The D2 probe vs step-1 pre_early comparison with a timing-only build is the first test; it affects D2's histogram parity and D3's performance-gate histograms only, not D3's correctness gate (verify mode is same-state by construction).
- D4(c)(ii) (retry wall over the whole pre-pericentre stretch) is unevaluated until command 3c runs after the job ends; its method is the one behind the diagnostic's 28 s figure (verified to reproduce it).
- Build path verified by the human session: `build/intelReleaseMPI/rich` is a symlink to `rich_intelReleaseMPI`, so `cp -L` in command 5 copies the real binary.
- Volume ratio, own bin and applied dt of the failing cells are not yet measured; D5's rule waits on them.
- The gate's zero-invalid requirement is stricter than global's 2 invalid retries in the same window; kept as agreed, flag if it is the only failing check.
- Whether `RICH_RUNTIME_LOG=detailed` prints perturb the probe's timing fields (rank-0 prints only; check `step_s`/unit t against step-1's 6362 over the same window).
- S6s assumes `getIntersectingRanks` culling is the only place the radius drives the MPI exchange (read at 985-997; the non-MPI branch is unaffected). Verify mode roughly doubles sweep cost (both exchanges per sweep); acceptable on a probe, never on production.
- Whether the closure traversal at 3546-3561 executes in the step-1 configuration depends on `individualForceAllActiveRuntimeOptions().enabled` or a latch/once request (3543-3544); `closure_s` is emitted regardless and reads 0 if not.
- Moving suggest time out of the physics names changes hydro_s/radiation_s semantics from this build on; earlier logs must be read with that in mind.
- The 90% contained-span requirement in the gate additions is a choice, not a plan item: global's own contained fraction was 95.7%; individual post-pericentre events are far shorter, so it should be near 100%. Flag if it is the only failing check.

## Assumptions

- Current time ~21:30 and the human's availability window (asleep 23:00-08:00) as stated; bigrun has idle d25g nodes for the continuation and the probe (117 idle at 15:17, quota 352 of 1024 cpu used).
- The 446M-record sweeps are the all-to-all saturation case; inferred from 1.75M cells x 255 ranks matching the measured maximum, to be confirmed by 2b before S6s work starts.
- All wall, cadence and retry numbers are the human session's script results, not re-derived; no shell commands were run this round.
- The continuation's first RICH_STEP cycle is the restored checkpoint cycle or the one after it; the strict `cycle < c1` splice rule at `step1_analyze.py:66-68` is correct in both cases, and the join check at `83-84` fails closed if it is not.
- The global baseline 119/0.062207243769 was computed from wholly contained steps over their actual summed span (29 steps, 0.0622 of the 0.065 window); the script's current rate (93-100) uses the identical rule.

