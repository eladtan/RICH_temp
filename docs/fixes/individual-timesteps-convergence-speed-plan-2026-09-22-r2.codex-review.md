# Codex reviews of the r2 plan (gpt-6-astra, reasoning effort high)

Loop run: ~/.codex-converge/runs/20260922-133407-2608805 (4 rounds, max-rounds, dissent empty).

## Loop review 1

Verified the A/D event counts, spans, retry counts, and principal cost totals. Five issues remain in the revised plan:

- **[BLOCKER] §2 R2 conflicts with the explicit timestep bound.** Removing the `min` at `RadiationStep.cpp:888–902` permits larger next timesteps than today’s default produces. Preserving radiation rollback does not establish the requested “no dt can become larger” property. Retain the agreed item, but explicitly resolve this constraint conflict; do not claim both properties simultaneously.

- **[BLOCKER] §3 S5’s wake-tree cache lacks a conservative reuse rule.** `Simulation.cpp:382–442` prunes using tree bounding boxes. Even displacement smaller than a fraction of a cell can move a target outside its cached box, causing a missed wake and a longer timestep. Specify conservative bounds/refitting, current signal-speed and sleep summaries, and invalidation conditions. Require **zero per-cell deadline increases against fresh-tree evaluation**; matching aggregate `active_bins` histograms cannot establish safety.

- **[MAJOR] §3 S3’s new trigger can still produce no repartition.** The unchanged call passes `forced_balance` at `Simulation.cpp:3966–3967`. Inside `rebalanceCommittedIndividualState`, lines 2957–2960 still require weight skew and `ShouldRebalance` unless forced. Thus the proposed outer OR-condition does not address balanced-count, stretched domains. Propagate the collectively agreed geometric trigger through that inner decision, retaining cooldown and an appropriate cost gate.

- **[MAJOR] §0/§2 overstate R2’s completed validation.** D’s 2108 retries/unit-time covers `0.375268–0.440268`; global’s 2229 covers `0.000886–0.460654`. These are different physical windows. For comparison, global’s 29 complete steps within D’s interval have 119 retries over `0.376255357089–0.438462600858`, approximately **1913/unit-time**. This does not prove failure, but invalidates the claimed matched-window pass. D’s cited log also lacks a terminal completion record supporting “same endpoint reached, exit 0.” Mark this gate pending matched-window comparison and terminal evidence.

- **[MAJOR] §3 S6 leaves targets-only evaluation without its required contract.** It has no separate switch, expected saving with measured basis, or numeric correctness/performance gate. The reset gate does not cover this second change, and “existing tolerance” remains unquantified. Specify these independently; retain unmeasured savings as uncertainty.

VERDICT: CHANGES_REQUIRED
## Loop review 2

S3’s call-site correction and withdrawal of R2’s claimed completed validation are sound. Four issues remain:

- **[BLOCKER] S5(ii), lines 387–405: the new cache rule still permits missed wakes.** Displacement and velocity inflation do not bound changes in sound speed, `has_source`, or remaining sleep. `Simulation.cpp:231–235,415,428–442` uses all three when pruning. A previously nonsleeping target can become sleeping while its cached summary still rejects traversal. Require current summaries each event, conservative position bounds, and invalidation when cell membership/indexing changes. Define velocity change as the norm of the vector difference. Retain the per-cell deadline comparison.

- **[MAJOR] S6/S6b, lines 431–463: acceptance gates remain incomplete.** S6 still requires an unspecified “existing tolerance,” although S6b correctly establishes that `sampleDirectAccelerationError` only reports errors. Give the reset its own numeric accuracy gate. S6b’s performance gate explicitly has no numeric target; require the measurement stage to establish that target before implementation proceeds.

- **[NIT] §0/R2: the corrected global retry count is 119, not 118.** The range `6016–6749` omits cycle 1535’s four retries and includes cycle 1564’s three. Matching retries by cycle to the 29 completed steps gives **119 / 0.062207243769 = 1912.96** retries/unit-time. The pending-gate conclusion remains unchanged.

- **[NIT] R2, line 222: the default instruction contradicts itself.** “NOT default-off yet” is followed by “stays off by default.” Change the latter to **“remains on by default”**, consistent with `RadiationStep.cpp:339`.

VERDICT: CHANGES_REQUIRED

## Loop review 3

The wake-summary fix addresses the previous blocker. Independently confirmed **119 retries across 29 global steps**, or **1912.96/unit t**. Two issues remain:

- **[BLOCKER] §3: S7 is missing.** The revision jumps from S6b to §4, which still schedules S7. The agreed document’s lines 193–197 contain the adaptive-controller validation item. Restore it with verified citations, its environment switch, expected benefit, and numeric gates. Neither changelog nor dissent records a reason for removing it.

- **[MAJOR] §4, Session N: the R2 validation run can switch to global mode.** Current `runs/BaseTDEComptonIndividual/test.cpp:1626` calls `SetAdaptiveIntegrationMode(true, ...)`. `Simulation.cpp:2048–2051,2130–2131` provides the override `RICH_INDIVIDUAL_ADAPTIVE_MODE=0`, but the plan never specifies it. Consequently, the proposed run could reach the post-pericentre gate using global stepping, invalidating its claimed validation of individual-mode R2. Disable adaptive switching for the individual A/B tests; require **zero mode switches and exclusively `mode=individual` records** throughout their measurement windows. Enable it explicitly for the separate S7 validation.

VERDICT: CHANGES_REQUIRED
## Loop review 4

Confirmed **119 retries across 29 global steps**, arm-D’s principal timing totals, and the adaptive-mode override. Three findings remain:

- **[MAJOR] §3 S3 / §4 step 6: the rebalance trigger remains unreachable under the probe configuration.** `Simulation.cpp:3931-3936` gates both `automatic_balance` and `request_balance` on `!active_hilbert_balance`. Active Hilbert defaults on (`1288-1290`); automatic rebalance defaults off (`1219-1220`). Actual evidence: `meshaberr_noretrybin_10199568.txt:5` records `enabled=0 active_hilbert_cache=1 requested=0`. The corrected `forceRebalance` argument addresses the inner gate, but cannot overcome these outer gates. Specify compatible settings for both A/B arms—such as active Hilbert off and automatic rebalance on—or define how the trigger integrates with active Hilbert. Keep the applied/migration gate.

- **[MAJOR] §3 S7, gates (b)–(c): continuity remains asserted rather than tested.** Calling `beforeIndividualRebalance` and transferring the next timestep does not establish conservation, positivity, or continuity at runtime. Specify measured before/after quantities with numeric tolerances, zero invalid states, and explicit controller-decision checks against the configured **1.15** throughput margin and **1.5** gain threshold. S7’s restoration resolves its omission; these acceptance criteria remain incomplete.

- **[NIT] §0 global wall-time normalization:** summing all 256 `RICH_STEP.step_s` values gives **1626.997066 s** over **0.459767687645476** simulated-time units: **3538.74 s/unit t**, rather than **3561**. Correct the table, or identify a separate wall-clock measurement that explains the difference.

VERDICT: CHANGES_REQUIRED
## Direct review 1 of the post-loop reconciliation (codex_review_2.md)

One MAJOR remains.

- **[MAJOR] S3b still needs to bypass cached-cut reuse.** [S3b’s design]( /home/elads/mpi_tmp/claude-5014/-home-elads-RICH-ablation-integration/468f34ec-925a-457b-a196-97f747e524a4/scratchpad/agreed_plan_2.md:450) forces entry into the proposal branch, but that branch first tries the cached partition at [Simulation.cpp:3307](/home/elads/RICH-ablation-integration/source/newtonian/three_dimensional/simulation/Simulation.cpp:3307). A count-balanced cached cut can pass validation; if it equals the current cut, line 3395 reports `already-current` without migration. Thus the proposed trigger can still leave the expensive decomposition unchanged. **To approve:** explicitly require `mesh_rebuild_trigger` to bypass current-mask cache reuse and force fresh boundary construction, retaining the existing rejection rules and migration gate. Other cached masks need no blanket invalidation.

- **[NIT] S7 decision replay needs rounding handling.** Existing decision records use six significant digits ([Simulation.cpp:2174](/home/elads/RICH-ablation-integration/source/newtonian/three_dimensional/simulation/Simulation.cpp:2174)). Exact comparisons against 1.15 or 1.5 can therefore falsely reject decisions near a threshold. Specify rounding-aware comparisons or higher-precision logging.

The requested round-4 checks otherwise resolve:

- **S3 probe reachability:** standing precondition 2 and the `forceRebalance` threading address both gates. S3b supplies a switch, conditional saving estimate, and numeric migration/performance gates; the cache bypass above remains necessary.
- **S7:** the grey `Diffusion` correction is accurate. Gates now specify conserved quantities, numeric tolerances, zero invalid-energy retries/aborts, a twin comparison, and explicit adoption/gain arithmetic. These are planned measurements, correctly marked unrun.
- **Wall-time NIT:** independently confirmed **1626.997066 / 0.459767687645476 = 3538.7373 s/unit t**. The first 252 records sum to 1605.651678 s, consistent with the stale partial-run figure. Also confirmed 119 matched-window retries across 29 global steps.

VERDICT: CHANGES_REQUIRED


## Direct review 2, final (codex_review_3.md)

No unresolved [BLOCKER], [MAJOR], or [NIT] findings in the [revised plan](/home/elads/mpi_tmp/claude-5014/-home-elads-RICH-ablation-integration/468f34ec-925a-457b-a196-97f747e524a4/scratchpad/agreed_plan_2.md).

- **S3/S3b:** Both reachability gaps are answered. Standing precondition 2 opens the probe path; the proposed `forceRebalance` argument bypasses the inner skew gate. S3b explicitly forces fresh boundary construction despite a cache hit, preserves rejection checks, and requires actual migration, next-build `mesh_s ≤ 3 s`, zero parity mismatches over 100 events, and ≥25% wall-time improvement. The saving estimate remains conditional.
- **S7:** Confirmed the driver constructs grey `Diffusion`. Gate (b) specifies conserved-state measurements, numeric tolerances, zero invalid-energy retries/aborts, and a twin comparison. Gate (c) checks recorded decisions against the effective 1.15 margin and 1.5 gain threshold. Its rounding slack resolves the direct-review NIT.
- **Wall-time NIT:** Independently reproduced **1626.997066 / 0.459767687645476 = 3538.7373 s/unit t**. Also confirmed **119 retries across 29 matched global steps**.

Approval covers the written plan. The proposed changes and runtime acceptance gates remain unimplemented or unrun, as stated.

VERDICT: APPROVE