# Reviews of individual-timesteps-crash-50107-2026-09-23.md (gpt-6-astra, reasoning effort xhigh)

Converge loop /home/elads/.codex-converge/runs/20260923-070820-1141002 (author claude-fable-5-1; stopped at round 2 when the author hit its quota), then three direct
read-only reviews of Claude's reconciliation, the trace/S5/R1' source changes and the experiment scripts.
Final verdict: APPROVE (review 3); its three NITs were applied afterwards (integer ID broadcast, finite-duration
checks in crash_gate.py, reporting-frequency wording).

## Loop review-1

Changes required. Changelog contains only the initial draft; dissent is empty.

1. **[BLOCKER] §3, F1 deployment leaves the guard disabled.** Changing the code default to `0.2` cannot override `export RICH_INDIVIDUAL_MESH_DRIFT_FRACTION=0` in either `submit_step1.sh:35` or `submit.sh:53`. The proposed “code default … not editing submit scripts” deployment therefore retains the failing configuration. Specify how authorized deployment removes those explicit overrides, and require the effective runtime setting in the gate logs.

2. **[BLOCKER] §2, E3 does not discriminate H3.** E3 is skipped whenever E1 passes cheaply, leaving the requested H3 investigation unperformed. Its proposed scalar trace—`d/R`, kick magnitude, gain, nearest-neighbour distance—cannot distinguish overshoot from moving-centroid or neighbour-driven growth. Include generator and centroid vectors, installed velocities, actual intervals, and the relevant neighbours’ motion across intervening events. Make that diagnostic part of the experiment set, potentially combined with the control. Keep unresolved causality distinct from F1’s independent robustness test.

3. **[MAJOR] §1, several estimates are labelled proven.** The separation `0.0298` belongs to `t=0.1412`, not the failing interval’s opening. Thus `0.0351/0.0298=1.18` and the predicted bin-33 drift limit remain estimates. The raw CFL value is also needed to evaluate the `CFL/16` floor. Similarly, `−117%` is the implied **uniform-flow** GCL diagnostic; approximate agreement across one face does not establish exact uniform flow across all 15 faces or exclude every hydrodynamic contribution. Comparable total area supports plausibility, not mathematical closure. Preserve the measured negative mass and flux contributions; qualify these stronger conclusions.

4. **[MAJOR] §2, E0/E1/E2 outcome interpretations overclaim causality.**
   - E0 failing to reproduce after changing hardware does not uniquely prove different binary physics.
   - `mesh_drift_limited` is an MPI-summed count (`hdsim_3d.cpp:2557–2575`), so a positive count does not prove this pair was limited; zero does not uniquely establish missing ghost velocities.
   - E2 failing cannot exclude H1b across the earlier fresh-run history: its checkpoint already contains the R2-off trajectory.
   
   State these outcomes as support, contradiction, or inconclusive results with the necessary cell-specific checks. R2’s robustness requirement must remain mandatory regardless of E2’s outcome.

5. **[MAJOR] §3, F2’s safety and cost claims are unsupported.** The quoted GCL estimate, `0.0014`, is **larger** than F1’s `0.00109`, contradicting “tighter.” A net volume-rate sum can cancel while individual faces deform substantially; an instantaneous rate does not establish a finite-interval deformation bound. F2b’s proposed wake *after* accumulated displacement exceeds a threshold may arrive after the invalid closing update. Treat both as designs needing validation: retain F1 while evaluating F2a, and require prospective deadlines or another demonstrated safe mechanism for F2b.

6. **[BLOCKER] §3, the maximum-cost gate is explicitly unenforced.** “Anything over it ships too” contradicts the proposed `1.25×` step-time and `1.5×` event ceilings. Exceeding either must leave acceptance pending optimization or the user’s explicit revised budget. Also exclude the aborted, incompletely logged event from the cost baseline: select a common completed window using actual `RICH_STEP` times and normalize both arms by its exact duration. Require successful robustness validation before shipping R2; “the same day” is not a validation dependency.

7. **[MAJOR] §2, the executable protocol and pass conditions disagree.** The submission writes `exp_E1_%j.txt`/`exp_E1err_%j.txt`, while the checks inspect `step1_*`. The `0.180` alternative retains E1’s later tick threshold, and E2 has no explicit completion/failure check. With `--export=ALL` and environment-reading script defaults, pin **both** drift and fold values for every arm; otherwise inherited variables can change the experiment. Use actual logged end times rather than deriving an exact epoch from rounded `t=0.1767`.

8. **[MAJOR] §§2–4, the resource plan does not meet the requested envelope.** The chosen runs are estimated at `1.5–3.5 h`, versus the requested approximately `75–90 min`; shortening to `0.180` does not establish that fold-on fits. Request explicit authorization for any longer allocation and classify a time-limit stop before the endpoint as inconclusive. Account explicitly for E0–E3 plus the separate `t=0.46` gate: that is potentially five runs before the later R2 comparison. Justify the extra validation run or consolidate it, and explicitly request 16 nodes/256 ranks.

VERDICT: CHANGES_REQUIRED
## Loop review-2

Four unresolved issues:

1. **[BLOCKER] §2: the submission command does not reliably select the intended experiment.** In [solution.md:104](/home/elads/.codex-converge/runs/20260923-070820-1141002/solution.md:104), `--export=ALL,...` permits inherited environment values to override the explicit assignments, per the installed `sbatch(1)` manual. Also, `TRACE_CELL_IDS=50107,717470` exports `50107` as the value and treats `717470` as another environment-variable name. Set all arm-specific values in the submitting process’s environment and use `--export=ALL`, or use a NUL-delimited export file. Verify both trace IDs and the effective arm settings in the startup log.

2. **[MAJOR] §2, E1: the PASS condition assumes the treated trajectory preserves the control’s dangerous interval.** [Line 85](/home/elads/.codex-converge/runs/20260923-070820-1141002/solution.md:85) requires cell 50107 to be drift-limited at tick `18141941858304`. F1 can act earlier, change neighbouring trajectories, and prevent the large offset from developing; that would invalidate this condition despite successful prevention. Separate robustness PASS from mechanistic attribution. Inspect E1’s actual activation history and limiting events across the growth window, including earlier or neighbour-mediated intervention.

3. **[MAJOR] §§1 and 3: recurring cost and F3’s necessity are asserted without evidence.** The claims that H3 necessarily leaves F1 receiving large kicks, and that only F3 removes their recurring cost, are unsupported. Under the proposed over-relaxation mechanism, shortening `dt` reduces `G = chi*c*dt/R`; this could suppress offset amplification and subsequently reduce the kicks while preserving RoundCells. Make persistent cost a hypothesis tested by E0/E1 histories. Keep F3 as the user’s optional decision.

4. **[MAJOR] §4: the revised R2 gate omits existing acceptance requirements that G1 cannot evaluate.** The approved [decision document:30](/home/elads/RICH-ablation-integration/docs/fixes/individual-timesteps-decision-2026-09-22.md:30) specifies quantitative retry and coverage checks; [line 32](/home/elads/RICH-ablation-integration/docs/fixes/individual-timesteps-decision-2026-09-22.md:32) additionally requires retry wall ≤1% of step wall over `−1.519..0`. G1 starts at `0.14116`, so it cannot close that prerequisite. Explicitly retain the existing gates, add robustness, and identify the pre-pericentre requirement as unevaluated unless suitable evidence is supplied. Likewise, G1 can yield a post-pericentre S5 profile, but cannot replace profiling the pre-pericentre regime where the 55% cost was measured.

VERDICT: CHANGES_REQUIRED
## Direct review 1

Most R1/R2 corrections are sound: environment assignments plus plain `--export=ALL`, separate robustness/attribution, optional F3, enforced cost ceilings, explicit allocations, and E1=G1 consolidation.

The comparator correction is right: arm D was R2-off; D4 uses the **global** retry rate. No D′ comparator is required. Fresh G2 can close D4(c), with the qualification below.

1. **[MAJOR] Apply the complete D4 gate to G2’s spliced record.** [Plan §4](/home/elads/RICH-ablation-integration/docs/fixes/individual-timesteps-crash-50107-2026-09-23.md:225) currently assigns D4(a)/(b) to G1 and D4(c) to G2. D4 requires acceptance on one whole-run record. G1 inherits the drift-off prehistory; G2 can reach the matched window differently. Require G2 to pass **D4(a)–(c)**, retaining the approved additions and tests. This requires another analysis of G2, not another simulation.

2. **[MAJOR] `crash_gate.py` does not implement the stated robustness verdicts.** Its final `ok` expression ignores `done`, effective settings, trace IDs and E1’s guard evidence. In-memory tests confirmed:
   - `t_last=0.461`, no `Done sim`: **PASS** with `--pass-t 0.46`.
   - Wrong E1 settings: **PASS**.
   - Time-limit termination below the endpoint: **FAIL**, contrary to the plan’s **INCONCLUSIVE**.
   
   Distinguish early progress from final completion and incomplete runs. Either enforce the remaining arm checks or explicitly make this a partial report whose PASS cannot constitute experiment acceptance.

3. **[MAJOR] The cost calculation accepts missing event history.** `span = win[-1][3] - win[0][2]` includes gaps, without checking continuity, ordering or window coverage. A synthetic gapped log returned PASS and normalized two events spanning **0.0018** by **0.0338**, substantially understating cost. Validate the sequence and window, then sum included event durations. Also make the E1/E0 ceiling comparison an explicit acceptance step; this script currently only prints rates.

4. **[MAJOR] Submission does not reliably enforce the private-restart prerequisite.** The directory refusal compares an unresolved string, so aliases, relative paths and `..` can bypass it. More immediately, the driver [appends the physical run-name directory](/home/elads/RICH-ablation-integration/runs/BaseTDEComptonIndividual/test.cpp:1296), whereas staging prints that leaf directory. Supplying the printed leaf as `RICH_TDE_RUN_DIRECTORY` creates another nested leaf; missing `counter.txt` then selects a fresh run despite the script’s `START_FRESH=0` echo. Document the required **parent** path, resolve and check the actual destination, and verify the staged restart before launching. Failed checksum or node-scan commands also currently allow execution to continue.

5. **[MAJOR] Staging’s checksum claim is incomplete.** The copied file set matches the supplied restart requirements, but `stage_restart_copy.sh` hashes only `snap_19.h5` and `individual_restart.h5`. It does not verify the approximately 621 MB of per-rank scheduler state, or `gravity.txt`; checking counter text alone is insufficient. Verify a manifest covering the complete copied restart set.

6. **[MAJOR] Trace neighbour ownership is wrong for locally owned passive halo cells.** At [hdsim_3d.cpp:2580](/home/elads/RICH-ablation-integration/source/newtonian/three_dimensional/hdsim_3d.cpp:2580), `other >= event_view.localSize()` is treated as requiring remote ownership metadata. Partial meshes also contain same-rank canonical cells outside that target range; they receive `neighbor_owner=-1`. Use the mesh-to-canonical mapping to recognize those local halo cells, with remote ownership from the exchange metadata.

7. **[MAJOR] Some evidence qualifications remain contradicted by “Excluded.”** [Plan §1](/home/elads/RICH-ablation-integration/docs/fixes/individual-timesteps-crash-50107-2026-09-23.md:87) still excludes phantom faces generally and gas dynamics as the killer. Full build excludes the identified **partial-build omission mechanism**; comparable total area and local corrections do not prove geometric closure. Similar states on the dominant face do not establish uniform flow across all 15 faces. Narrow these exclusions consistently with the preceding qualifications. Likewise, orphan co-location and slowdown support the orphan explanation but do not prove the entire slowdown was caused by it.

The remaining trace assessment is favorable **under the stated identical, valid environment-list precondition**. I found no rank-selective normal-path skip: both first-half paths rejoin before the geometry hook, and empty-record ranks still gather. The collective hydro abort occurs before that hook, so the aborting event itself produces neither trace half. `w_closed` captures the velocity used for generator drift; `w_next` captures the installed velocity. `drift_time` matches the limiter, and loss limits are sampled before accumulator reset. The added `.at()` accesses respect the inspected size/mapping invariants. No physical-state mutation was introduced; unset tracing performs no trace collectives.

**[NIT] Qualify H3’s promised outcome.** The vectors can distinguish generator overshoot from centroid motion over recorded intervals. However, target-only emission can leave gaps while a traced cell is absent from the partial target mesh. Check history completeness and retain “inconclusive” for neighbour-driven attribution; “the trace decides it” is too unconditional. Also mirror the limiter’s finite-loss checks in the diagnostic calculations.

Read-only review and in-memory parser checks only. No files changed; no build or simulation run.

VERDICT: CHANGES_REQUIRED
## Direct review 2

Most prior findings are resolved, including G2’s complete D4 gate, restart-manifest coverage, trace ownership, and the narrowed geometry exclusions. Four gating issues remain.

1. **[MAJOR] Explicitly set the node-scan task count.** In `submit_crash.sh`, `srun` inherits `SLURM_NTASKS=256`; `--ntasks-per-node=1` does not replace that total. The installed Slurm manuals confirm this precedence. The intended 16-node scan therefore has conflicting task requirements and can fail before launch. Add `--ntasks="${SLURM_JOB_NUM_NODES}"`. The login-node snippet did not exercise this allocation context.

2. **[MAJOR] Node-scan success still masks command failures.** The remote shell unconditionally prints `NODE_SCAN_OK` after `ps | awk`. Injecting a failing `ps` into the otherwise unchanged body produced **exit 0 and `NODE_SCAN_OK`**. Check process enumeration and filtering separately; emit the success marker only after both succeed. Merely adding `set -e` does not reliably catch failure of the pipeline’s first command.

3. **[MAJOR] Robustness still misses fatal terminations.** `aborts_in()` recognizes only `invalid state` and `what():`. With all E1 checks satisfied, `t_last=0.181`, and an MPI diagnostic reporting **signal 11 (Segmentation fault)**, the copied decision logic returns **early PASS / final INCONCLUSIVE**. That violates “FAIL at any tick.” Handle fatal termination evidence, while preserving INCONCLUSIVE for a genuine time-limit stop. Add this regression case.

4. **[MAJOR] Cost validation does not establish forward time progression.** `window()` accepts increasing-cycle intervals `0→0.7`, `0.7→0.6`, `0.6→1`: no problems, coverage 1, and rates identical to a valid three-event control. The comparison consequently accepts a backward interval. Require finite, positive event durations before calculating rates. A zero-duration window also currently raises `ZeroDivisionError` instead of returning INVALID; validate before dividing.

**[NIT] S5 preparation timing is incomplete on early returns.** The no-source and no-remaining-sleep exits precede the second `prepare_seconds` update in [Simulation.cpp](/home/elads/RICH-ablation-integration/source/newtonian/three_dimensional/simulation/Simulation.cpp:754). Their cadence/count/sleep preparation is omitted, although the normal path includes it. Finalize preparation timing before those returns.

Otherwise, S5 is direct and readable: no physical-state or scheduling change found; added reporting collectives stay inside the existing trace block; nested timings do not alter `other_s` or the named-phase consumers. Normal MPI timer boundaries match their descriptions.

Read-only inspection, in-memory gate checks, and shell failure injection only. No files changed; no build or simulation run.

VERDICT: CHANGES_REQUIRED
## Direct review 3

No [BLOCKER] or [MAJOR] findings. The four review-2 regression cases and S5 early-return timing issue are addressed.

Three non-gating nits:

- **[NIT] Preserve integer diagnostic IDs.** The new [broadcast](/home/elads/RICH-ablation-integration/source/Radiation/Diffusion.cpp:590) converts the cell ID through `double`, losing precision above `2^53`. Broadcast the ID with `MPI_UNSIGNED_LONG_LONG`, as the existing representative diagnostic does.
- **[NIT] Complete the finite-duration check.** `window()` checks finite endpoints, but their subtraction—and the accumulated duration—can overflow. Explicitly check `math.isfinite(d)` and `math.isfinite(dur)`. This extreme-range case does not affect the stated campaign window.
- **[NIT] Clarify reporting frequency.** Guide row 892 and plan item 4 say “per event”; the record prints only when at least one cell’s limit is shortened.

R1′ matches D5’s criterion: fixed event-start baselines survive retry candidates; positive, finite starting budgets are required; zero change imposes no additional limit; and `applied_dt*f/r` only shortens the existing suggestion. Absolute net changes cover heating and cooling, and recomputation permits relaxation when coupling weakens.

The inspected call sites use the same `local_cells` and `local_context` for prestep and timestep calculation. With fixed volume, `|Δ(E/V)|/(E₀/V) = |ΔE|/E₀`, so the density-based calculation matches the extensive criterion and includes the final radiation corrections.

For rank-consistent `f>0`, both reductions and the broadcast run on every rank, including ranks with no limited cells. With `f=0`, timestep decisions and the collective sequence remain unchanged; the new local baseline snapshot still incurs allocation/copy work.

The implementation is direct and auditable; no unnecessary abstraction or dead code found in the supplied changes. Read-only inspection and scalar checks only. The supplied 21/21 tests and syntax checks were not rerun; D5’s runtime gates remain pending.

VERDICT: APPROVE
## F1 deployment change review

No **[BLOCKER]** or **[MAJOR]** findings in the pasted F1 diff.

- **(a) Default:** Evidence supports `0.2` as the candidate default: E0 reproduces deterministically, and E1 directly suppresses the traced overshoot mechanism while passing the measured cost gate. Preparing the source flip now is consistent with the plan. **Shipping must wait** for G1 at `0.46`: zero invalid-state aborts, `Done sim`, and effective drift `0.2`. The proposed new-build regression gate must also pass. Holding the launcher edits until G1 is appropriate.

- **(b) `INDIVIDUAL_GUARDS`:** Correct for once-per-process reporting. Each rank reaching this block has its own static flag; only `MPI_COMM_WORLD` rank 0 prints. `MPI_Comm_rank` is noncollective, so the logger adds no collective ordering requirement. The excerpt alone cannot establish that every rank calls the enclosing method, but this logging block does not require that. Absence in global-only mode is acceptable. Multiple simulation objects in one process still produce only one record, matching the stated intent.

- **(c) Nine regression cases:** Running THUNDER against the new build is the right gate; leave these cases exercising the new default. Rerunning a failing case with drift `0` is a useful attribution check. Restoring the old result establishes attribution, **not correctness of a replacement baseline**: any changed baseline must still meet the case’s physical and numerical acceptance criteria.

- **[NIT] Evidence wording:** In the source comment, guide, and changelog, qualify “passed” as the **early robustness gate** while G1 remains pending. Describe `0.61×` events and `0.94×` step wall as measurements over the matched `0.141161735–0.174667985` window; they are not whole-run cost ratios.

This is approval of the pasted source change. Build, regression, and final robustness validation remain outstanding.

VERDICT: APPROVE
## Serial build fix review

No [BLOCKER] or [MAJOR] findings in the supplied change.

- **(a) MPI behavior:** Neutral. Both definitions retain their bodies, namespace scope, and linkage. Moving them earlier creates no duplicate definitions or ODR issue. Every listed call follows the new definitions; the intervening structs introduce no ordering dependency.
- **(b) Serial scope:** This fixes both reported undeclared-helper errors across the 18 serial builds. No additional serial defect is evident from the supplied material. The reported syntax checks support compilation of those translation units; full linking and regression execution remain unverified. This fix does not address scheduler timeouts.
- **(c) Rerun:** Sound, given the stated incremental reuse of preserved objects. `--rerun-failed 20260923_115005` can complete the unfinished builds and exercise the fix. **[NIT]** The unchanged eight-minute limit and build concurrency can still cause timeouts; incremental progress improves the prospects but does not guarantee completion. If timeouts recur, reduce concurrent builds or increase their time limit.

Approval covers this build fix. The F1 default-flip gate still requires successful regression results.

VERDICT: APPROVE
## S5 split-timer review

No [BLOCKER] or [MAJOR] findings in the requested instrumentation. Static review of the pasted code only; no commands run.

- **Timer boundaries are sound.** `local_eval_seconds` brackets `evaluateIndividualWakeSource`; `route_seconds` covers radius calculation, destination lookup, and buffer pushes. Route timing also includes the preceding evaluation-counter update. Source construction, round setup, and some bookkeeping remain only in `local_route_seconds`, so the two sub-timers need not sum exactly to their parent. Rank medians/maxima also do not add linearly.

- **Saturation test is correct.** With the validated finite, positive diagonal, the helper returns a radius in `[0, domain_diagonal]`; `!(query_radius < domain_diagonal)` therefore identifies equality with the cap. This includes nonfinite intermediate radii deliberately clamped to that cap.

- **[NIT] There are three clock reads per source, not two:** `eval_start`, `route_start`, and the final `now()`. Clock overhead is approximately `3 × source_count × clock_read_cost`, plus bookkeeping. For illustration, a 50 ns read would contribute 0.15–1.5 ms per rank/event at the stated counts; actual cost is unmeasured. These reads execute even when tracing is disabled. Acceptable for this diagnostic addition.

- **Collective placement is sound.** No new collective appears in the source loop. The two added phase entries and saturation count introduce three additional reporting-helper invocations inside the existing reporting block.

- **[NIT] Correct the saturation comment.** “Sent to every rank whatever the tree says” overstates the implementation: routing still uses `getIntersectingRanks` and excludes the local rank. Prefer “Own sources whose query radius reached the domain diagonal.”

- **[NIT] Watch the denominator during full sweeps.** Saturated sources then count all evaluated local cells, while `local_signal_sources` still counts `signal_source_indices`. A saturation fraction using those two counters can exceed one on sweep events; use the actual evaluated-source count.

The additions are simple and direct; they introduce no apparent change to wake deadlines, routing decisions, or physical calculations.

VERDICT: APPROVE
## Cadence diagnostic reviews

- **[MAJOR] Trace quantization can crash a valid run or misclassify its limiter.** [Simulation.cpp:4039](/home/elads/RICH-ablation-integration/source/newtonian/three_dimensional/simulation/Simulation.cpp:4039) computes `limit / quantum` and converts `floor(log2(...))` to `int` before clamping. The scheduler deliberately avoids overflow in [quantizeTimeStep](/home/elads/RICH-ablation-integration/source/newtonian/three_dimensional/simulation/IndividualTimeStep.cpp:1118). For finite `limit=1e308`, `quantum=1e-12`, the scheduler safely saturates; tracing overflows. The [TDE driver enables FE_OVERFLOW](/home/elads/RICH-ablation-integration/runs/BaseTDEComptonIndividual/test.cpp:1284), so tracing can terminate an otherwise valid run. Without traps, converting infinity to `int` is undefined behavior. Separately, `quantum=1`, `limit=7.999999999999999` yields scheduler bin **2**, but the trace computes **3**, falsely attributing a physics-limited bin to the scheduler. Read-only numerical checks confirmed both cases. **Reuse the scheduler’s exact, protected quantization.**

- **[NIT] Retained bins do not identify every cause of event cadence.** [Simulation.cpp:4036](/home/elads/RICH-ablation-integration/source/newtonian/three_dimensional/simulation/Simulation.cpp:4036) measures newly assigned bins among this event’s active cells. [Signal wakes shorten `end_tick` without changing `time_bin`](/home/elads/RICH-ablation-integration/source/newtonian/three_dimensional/simulation/IndividualTimeStep.cpp:608). A wake-driven event can therefore be classified entirely as physics-limited. Describe this as next-bin attribution; attributing extra events to wakes requires deadline/interruption information. The single example ID also identifies only one representative finest-bin cell.

- **[NIT] An unconstrained example can disappear under MPI.** At [Simulation.cpp:4088](/home/elads/RICH-ablation-integration/source/newtonian/three_dimensional/simulation/Simulation.cpp:4088), absent candidates and real candidates with infinite limits both enter `MINLOC` as infinity. If all candidates are unconstrained, a lower-ranked process without a candidate can win and broadcast the sentinel record. Include candidate presence in selection.

The remaining requested checks look sound:

- **Switch off:** no diagnostic mutation of limits, bins, or deadlines found. Reason storage allocates initially/on size changes, then updates bytes. No new `.at()` failure identified for valid canonical indices and array sizes.
- **Collectives:** with the stated rank-consistent switch, every normally completing rank reaches `commitEvent` and then the same five diagnostic collectives, including ranks without active cells. Rank agreement is assumed, not checked.
- **Growth classification:** [chooseNextBin](/home/elads/RICH-ablation-integration/source/newtonian/three_dimensional/simulation/IndividualTimeStep.cpp:1162) permits growth by one bin only at an aligned tick. With exact physics quantization, the proposed test correctly identifies that cap; alignment holds fall into `scheduler`.
- **Hydro attribution:** correct for strict winners. If an earlier step remains tighter, hydro records no winning reason and the outer tracker preserves the earlier step. Equal limits retain the earlier attribution.

Read-only review; no edits, builds, or RICH runs.

VERDICT: CHANGES_REQUIRED
The prior **[MAJOR] and both [NIT] findings are resolved**. Quantization now matches the scheduler exactly; shortened intervals are counted with clear active-cell scope; unconstrained candidates beat absent ranks in `MINLOC`. Collective order remains consistent under the stated rank-consistent switch.

- **[NIT] `example_limit` prints the selection key rather than the original limit.** An unconstrained `+inf` now appears as `DBL_MAX`. Keep the capped value for `MINLOC`, but broadcast the original limit in `example_record` for accurate reporting.

No blocking findings. Reviewed the pasted code; syntax-check results are user-reported. No commands, edits, builds, or runs performed.

VERDICT: APPROVE
## S7 instrumentation review

No [BLOCKER] or [MAJOR] findings. The instrumentation is simple, direct, and suitable for the prerequisite.

- **(a) Collective safety:** Both switches follow the common `adaptiveAfterStep` path. Timing inputs use MPI maxima; the gain bound uses collective reductions. Individual→global additionally requires the collectively reduced `IndividualStateSynchronized()` result. The new reductions precede the rank-0 print guard, so every participating rank executes them.
- **(b) Ownership:** Correct at all four call points. The [synchronization guard](/home/elads/RICH-ablation-integration/source/newtonian/three_dimensional/simulation/Simulation.cpp:2057) requires `GetPointNo() == cells.size() == extensives.size()` and committed scheduler ticks before leaving individual mode. The intervening hooks release scratch storage; the ghost exchange modifies `cells`, preserving the mesh and extensives. Global mode’s first `GetPointNo()` extensives are owned; entering individual mode removes trailing storage.
- **(c) Behaviour neutrality:** No changes to committed state, timestep selection, scheduler logic, or controller thresholds. Added summation, reductions, and output impose overhead outside the sampled step interval. Capturing `scheduler_next_dt` before fallback preserves useful diagnostic evidence.
- **(d) S7 coverage:** Sufficient for the switch-pair conservation and timestep checks. Seventeen-digit state output and twelve-digit decisions provide adequate precision. This approves the instrumentation prerequisite; the retry windows, adaptive-off twin, and performance gates still require measurements.

[NIT] `momentum_abs` is explicitly **Σ(|px| + |py| + |pz|)**. Keep that definition explicit in the validation checker; it differs from the sum of Euclidean momentum magnitudes.

Read-only source review completed. No files changed; no build or simulation run.

VERDICT: APPROVE
