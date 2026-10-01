# Robust AutoPartial radiation plan

## Handoff directive

Implement this plan in the main RICH development thread. Robustness has priority over a larger nominal timestep or a small benchmark gain. Do not weaken residual, positivity, mapping, or rollback checks. Conservation-defect thresholds are synchronization targets rather than retry conditions. Preserve unrelated work, live jobs, campaign roots, and binaries. Build and run only under the authority active in the main thread and the repository `AGENTS.md` rules.

The recommended production method is an active-only implicit solve with frozen passive Dirichlet data. Passive radiation state is not immediately reconciled after the solve. The omitted equal-and-opposite interface transfer is measured as a signed, absolute, and local conservation defect. A finite over-target defect is committed and requests targeted passive synchronization; it does not lower every active timestep. Invalid accounting still fails closed. The conservative shadow-reservoir implementation remains available only as an experimental comparison path until its row/commit inconsistency is understood.

## Priority order

1. Never commit a negative, nonfinite, partially updated, or mapping-inconsistent candidate.
2. Never accept a Krylov result solely because the recursive or preconditioned residual says it converged.
3. Never update a passive radiation cell outside a transaction that can restore it exactly.
4. Measure every unit of energy intentionally omitted at an active-passive boundary.
5. Keep restart behavior and MPI decisions deterministic.
6. Preserve the synchronized all-active result as the oracle.
7. Optimize only after the safety gates pass.

## Current evidence and why the shadow path must not be promoted

The current worktree is heavily modified and contains multiple radiation, scheduler, mesh, MPI, test, and documentation changes. Before editing, record the exact tracked/untracked/submodule digest and relevant binary hashes. Do not assume `HEAD` identifies the executable.

The active-passive multigroup assembly already has the desired Dirichlet structure in `MultigroupDiffusion::BuildMatrix`: active-active faces couple unknowns, while an inactive neighbor supplies a fixed value to the active row. `RadiationDriver::stepIndividual` records active-passive face coefficients and currently contains the passive commit/reconciliation logic. The key implementation anchors are:

- `source/Radiation/RadiationDriver.cpp`: `RadiationDriver::stepIndividual` begins near line 5589; individual face processing and passive commits occur in more than one serial/distributed branch; `individualCellActive` and `recordIndividualFaceCoefficient` are near lines 8914 and 8919.
- `source/Radiation/RadiationDriver.hpp`: individual candidate hooks, failure attribution, all-active capability, transaction markers, `individual_context_`, and `individual_face_coefficients_`.
- `source/Radiation/MultigroupDiffusion.cpp`: active-passive matrix assembly near lines 4438-4531 and the multigroup solver/finalizer.
- `source/Radiation/Diffusion.cpp`: the analogous grey active-passive assembly near lines 820-838.
- `source/Radiation/conj_grad_solve.cpp` and `.hpp`: true-residual and componentwise/backward-error assessment.
- `source/newtonian/three_dimensional/simulation/IndividualTimeStep.cpp` and `.hpp`: event membership, retry feedback, bin limits, commit, and checkpoint state.
- `source/newtonian/three_dimensional/simulation/ActiveMeshView.hpp`: ID-safe active/canonical mapping.
- `regression_tests/cases/lane_radiation_shock_individual/`: campaign runner, comparator, focused gates, and provenance tools.
- `docs/user-guide/individual-timesteps.md`: current documented active-passive semantics and retry behavior.

`RICH_MG_INDIVIDUAL_SHADOW_RESERVOIRS` is currently default-off and MPI-consistency checked. Keep it default-off. Two stop-at-first-rejection experiments show that the shadow implementation is not ready:

- Job `10126983`: group 13, cell ID 127622, endpoint 300.38575199185647, paired transfer 300.38573624857816, tolerance 9.6123440637394067e-08, two Krylov iterations.
- Corrected job `10126984`: first rejected event time 0.03145728, event `dt=0.01048576` (bin 20), group 12, cell ID 125897, endpoint 402600.52764139185, paired transfer 402272.63686875789, tolerance 1.2731346545323754, two Krylov iterations. The proposed retry was `dt=0.00524288` (bin 19).

These are row/commit consistency rejections, not evidence that a harmless negative roundoff value should be floored. Do not loosen the row-residual tolerance. First distinguish:

\[
r_i^{\mathrm{CSR}} = b_i - (Ax)_i
\]

from

\[
r_i^{\mathrm{face}} = U_i^{n+1} - U_i^n - \sum_f q_{if}.
\]

If the CSR residual fails, investigate Krylov stopping, scaling, the final halo, or stale coefficients. If CSR passes but face balance fails, investigate assembly/commit mismatch, duplicate ownership, sign/orientation, ID mapping, stale epochs, or using a different solution value during commit. Removing passive shadow rows avoids making this experimental reconciliation a production dependency, but it does not excuse an active-row residual failure.

## Chosen numerical semantics

### Execution modes

Replace interacting Boolean behavior with one internal policy enum:

```cpp
enum class IndividualPassiveRadiationPolicy {
    ImmediateConservativeLegacy,
    ShadowReservoirExperimental,
    FrozenDirichletMeasuredDefect
};
```

Add one MPI-consistent runtime selector, for example `RICH_MG_INDIVIDUAL_PASSIVE_POLICY=legacy|shadow|dirichlet`. Preserve the old shadow flag only as a deprecated alias; reject conflicting settings collectively. During development, keep the library default unchanged. Promote `dirichlet` first in the AutoPartial campaign runner after all gates pass. Do not change the synchronized global path.

### Frozen Dirichlet candidate

For an active cell `a`, passive neighbor `p`, energy group `g`, and nonnegative assembled face coefficient `w`:

\[
(\ldots + w) E_{a,g}^{n+1} - w E_{p,g}^{*} = b_{a,g},
\]

where `E_p*` is the passive primitive value frozen at candidate start. The active solution is committed normally. In Dirichlet mode:

- do not add the opposite face delta to passive conserved `Eg` or `Erad`;
- do not refresh passive primitives;
- do not create passive unknown rows;
- do not run the shadow endpoint-versus-transfer reconciliation;
- do not alter material, Compton, Doppler, radiation-force, flux-limiter, boundary-condition, or active source terms;
- keep the globally all-active fast path exactly equivalent to the established global solve.

The energy received by the active cell but not removed from the passive cell is

\[
q_{p\rightarrow a,g} = w\left(E_{p,g}^{*} - E_{a,g}^{n+1}\right).
\]

Positive `q` is artificial total-energy creation because the passive donor was not debited. Negative `q` is artificial loss because the passive recipient was not credited. Compute this value from the exact coefficient, frozen passive value, and final active unknown used by the assembled row. Do not reconstruct it from rounded post-commit primitives.

### Defect accounting

Add internal records along these lines:

```cpp
struct IndividualRadiationDefectEvent {
    long double signed_extent = 0;
    long double absolute_extent = 0;
    long double passive_withdrawal_extent = 0;
    long double passive_deposit_extent = 0;
    double maximum_local_fraction = 0;
    std::uint64_t face_group_terms = 0;
    std::uint64_t representative_active_id = invalid_id;
    std::uint64_t representative_passive_id = invalid_id;
    std::size_t representative_group = invalid_group;
};

struct IndividualRadiationDefectAccounting {
    long double cumulative_signed_extent = 0;
    long double cumulative_absolute_extent = 0;
    double maximum_event_absolute_fraction = 0;
    double maximum_local_fraction = 0;
    std::uint64_t accepted_dirichlet_candidates = 0;
    std::uint64_t defect_rejections = 0;
    std::uint64_t defect_retry_substeps = 0;
    bool history_complete = true;
};
```

For every unique active-passive face/group term, accumulate:

\[
D_{\mathrm{signed}} = \sum q,
\qquad
D_{\mathrm{abs}} = \sum |q|.
\]

Also aggregate by passive stable ID and group:

\[
W_{p,g} = \sum \max(q,0),
\qquad
P_{p,g} = \sum \max(-q,0).
\]

`W` measures energy implicitly borrowed from a passive group; `P` measures omitted deposits. Signed cancellation must never hide a large absolute or local defect. Use compensated local accumulation. Reduce one packed diagnostic record collectively after candidate solution and before any commit. Ranks with zero owned rows must participate. Select a representative offender deterministically by fraction, then stable IDs, group, and rank. Finite over-target values request passive synchronization; only invalid accounting rejects.

Normalize the local metric with the passive positive group extent at candidate start plus a documented scale-aware roundoff floor. A zero-energy passive group may not be treated as a finite reservoir: withdrawal above the roundoff envelope requests synchronization of the passive endpoint.

Normalize global event and cumulative metrics with a declared physical scale, not an arbitrary `1.0`. Use the maximum of the initial positive global radiation extent, the candidate-start positive global radiation extent, and a finite RHS-derived floor. Record the denominator in diagnostics so ratios are reproducible.

### Initial defect synchronization targets

Treat these as engineering synchronization defaults to validate, not constants claimed by the literature:

- local passive-withdrawal synchronization target: `1e-2` of the passive cell-group positive extent plus the scale-aware absolute floor;
- event absolute-defect synchronization target: `1e-6` of the global radiation scale;
- cumulative signed diagnostic reference: `1e-4` of the declared global scale;
- cumulative absolute diagnostic reference: `1e-3` of the declared global scale;
- nonfinite metrics or a nonpositive normalization scale: unconditional rejection.

Keep the targets in one versioned internal configuration structure. Do not make synchronization depend only on an environment variable silently present on one rank. Retain the cumulative references for energy auditing; they do not reject a finite candidate or poison the complete active set's future timestep.

## Candidate state machine

Use one collective transaction state machine for serial and distributed-active branches:

1. Snapshot all state that the candidate can mutate. This includes active primitive and conserved state, any touched passive state in legacy modes, pending repair accounting, pending defect accounting, face-coefficient buffers, solver failure metadata, and candidate-local caches.
2. Freeze passive boundary primitives and their stable IDs.
3. Assemble the active-only matrix and RHS.
4. Solve.
5. Recompute the true `b-Ax` residual using the final solution and final halo values. Require the existing componentwise/backward-error gate as well as finiteness.
6. Validate coefficient epochs and active/canonical ID bijections.
7. Validate raw group extents, aggregate `Erad`, material energy, temperature, density, Fleck factors, diagonals, Compton state, and all existing positivity rules.
8. Compute active-passive defect metrics from the same face coefficients and final active unknowns.
9. Pack all local validity bits and defect metrics and reduce once. Invalid accounting rejects collectively; finite target crossings request passive synchronization.
10. On rejection, restore the snapshot without allocating, discard pending repair/defect accounting, clear candidate-local buffers, and enter the retry policy.
11. On acceptance, commit active state first, commit no passive radiation state in Dirichlet mode, then atomically append repair and defect accounting. Update `Erad` from the committed group sum. Only then expose the accepted candidate to timestep feedback.

No exception, early return, or remote failure may bypass rollback. Add a scoped transaction guard so future returns cannot accidentally commit half a candidate.

### Retry and fallback order

For a solver, positivity, mapping, or invalid-accounting rejection:

1. Retry the candidate with half the interval fraction using the existing exact-fraction coverage logic.
2. Attribute failures to stable IDs where possible and cap only responsible cells when the failure is cell-local.
3. After eight accepted candidates, allow the existing one-bin cooldown recovery. A new rejection resets the cooldown.
4. If halving cannot advance the candidate end time, fail with a controlled, fully attributed error. Finite conservation-target crossings do not enter this retry path.

Do not force an arbitrary all-active radiation solve in the middle of a partially advanced hydro event. That would advance passive radiation with incompatible hydro/primitive clocks unless the whole event is rolled back and those cells are genuinely promoted to the event. The existing mapped all-active fast path remains the fallback only when global activity and its current eligibility checks are actually satisfied.

An optional later exact-conservation experiment is deterministic event-level
neighbor promotion:

- take a whole-event snapshot before hydro/geometry work;
- on a finite boundary-defect target crossing, restore the entire event instead
  of committing the soft-error candidate;
- shorten selected passive neighbor end ticks to the current event tick;
- rebuild `IndividualStepContext`, `ActiveMeshView`, geometry, ownership maps, and all physics for the enlarged active set;
- repeat monotonically until the target is met or activity becomes global.

Do not implement a radiation-only one-cell halo that commits halo radiation ahead of passive hydro. Implicit diffusion can cross the entire domain in one solve; a fixed halo is not a generally valid causal boundary.

## Residual protection

The observed “two iterations” is not sufficient proof of a correct solution. Preserve and strengthen the existing true-residual path:

- evaluate `b-Ax` after the final Krylov update, residual replacement, halo exchange, and any positivity continuation;
- use componentwise backward error or a row-scaled criterion, not only a global L2 norm dominated by bright cells;
- report the recursive/preconditioned norm separately from the true norm;
- include the representative row's stable cell ID, group, `b_i`, `(Ax)_i`, diagonal, row absolute sum, and nonzero count;
- verify that the solution vector inspected by the residual gate is exactly the vector used by the face-defect calculation and commit;
- do not add 30 arbitrary iterations after nominal convergence; continue only through a declared reliable-update/true-residual policy and reject if the true gate still fails.

Add a debug-only dual identity check for active rows: independently reconstruct face/source balance and compare it with the CSR row. This diagnostic is default-off and must not be used to patch the solution.

## MPI, mapping, AMR, and cache rules

- Identify faces and recipients with stable wide cell IDs. Never persist transient mesh indices across redistribution or AMR.
- Use one deterministic owner for cross-rank face/group defect terms. In debug builds, detect duplicate `(min_id,max_id,group)` terms collectively.
- Validate runtime policy and thresholds are identical across ranks.
- Zero-owned ranks enter every required collective with neutral values.
- Invalidate face coefficients, passive snapshots, exchange plans, active layouts, and defect recipient maps on topology, ghost, ownership, group-layout, solver-configuration, rollback, and AMR epochs.
- Do not carry pending defect data across a rollback.
- All-active mapped execution remains permutation-aware: complete activity plus a valid ID bijection is sufficient; canonical index identity is not required.

## Checkpoint and restart behavior

Add optional versioned metadata for cumulative defect accounting, thresholds/version, cooldown state, and whether the history is complete. Checkpoint only accepted accounting. A checkpoint during a rejected candidate must reproduce the last accepted state.

Old checkpoints load deterministic zero accounting with `history_complete=false`. They may be used for continuation tests, but they cannot prove a full-run cumulative defect budget. A fresh-from-initial run is required for final acceptance. Preserve the runner contract: the restart pointer and every rank sidecar must be staged inside the new disjoint run root before submission.

## Implementation phases and gates

### Phase 0: immutable evidence

- Capture source content digest including dirty/untracked files and submodules.
- Capture compiler, MPI, linked libraries, affinity, node class, executable SHA-256, configuration, initial checkpoint identity, and 128 sidecars.
- Preserve jobs `10126983` and `10126984` artifacts as failure evidence.
- Create a new content-addressed build/run root. Never overwrite a binary used by a live or historical job.

Gate: provenance is sufficient to reproduce every comparison and no unrelated state was changed.

### Phase 1: isolate residual versus commit mismatch

- Add default-off CSR-versus-face diagnostics around the failing shadow rows.
- Reproduce the two-cell analytic identity and, if authorized, the short first-rejection replay.
- Determine whether the shadow mismatch is in the Krylov solution, assembly, exchange, or commit.

Gate: the code can classify a failure as CSR, face reconstruction, or physical positivity. This phase does not need to repair shadow mode before Dirichlet work proceeds.

### Phase 2: Dirichlet policy and defect ledger

- Add the policy enum and one MPI-consistent selector.
- Reuse existing active-passive RHS assembly.
- Bypass all passive conserved commits only in Dirichlet mode.
- Calculate pending defect metrics before commit.
- Add collective validity gates, deterministic representative diagnostics, rollback-safe accounting, targeted passive synchronization, and checkpoint fields.
- Update documentation without claiming exact conservation.

Gate: focused serial tests pass; passive radiation state is unchanged; active solution and defect match analytic results; over-target finite defects request passive synchronization; rejected invalid candidates restore bitwise-equivalent state and accounting.

### Phase 3: distributed correctness

- Test identity and permuted mappings, wide IDs, remote active-passive faces, duplicate protection, zero-owned ranks, AMR/redistribution epoch invalidation, and restart immediately before/after an accepted Dirichlet candidate.
- Verify one packed defect reduction, collective agreement on validity, and owner-local passive synchronization requests.

Gate: 2-rank and small multi-rank tests pass under the same MPI/compiler family used by the benchmark. No deadlock, unmatched message, rank-local mode difference, or RSS regression.

### Phase 4: controlled benchmark replay

Use `initial_dt=1e-4` and `time_quantum=1e-8`. Start from the known-safe capped ceiling, bin 25 (`dt=0.33554432`), rather than immediately restoring bin 40. Historical validation at bin 25 reached cycle 109/time 22.48146944 with zero retries/repairs and was about 20 times faster to time 11 than the earlier uncapped failure.

Run in disjoint roots:

1. startup replay through the previous early failure window;
2. stop-at-first-rejection lane with full diagnostic capture;
3. cycle-90 transition window;
4. late high-active window;
5. restart replay from a readable 128-rank checkpoint;
6. only after clean results, isolated bin-26 and bin-27 stress lanes.

Do not jump to bin 40. With the `1e-8` quantum, bin 30 is `10.73741824`; the prior uncapped run accumulated 274 radiation rejections and 501129 positivity-repair records. A larger outer timestep is not a speed or safety result.

Gate: no unclassified rejection; every retry restores cleanly; cumulative defect is finite and reported against its diagnostic references; comparator metrics remain acceptable; throughput includes accepted substeps, retries, repairs, I/O, and synchronization.

### Phase 5: production proof

Run paired Global and AutoPartial lanes with identical physics, placement, output cadence, endpoint, restart input, and provenance. Use production settings and a separate instrumented pair. Require terminal exit codes and artifact counts, not merely scheduler completion.

Gate: all correctness and artifact requirements below pass. Only then promote Dirichlet policy in the AutoPartial runner.

## Focused test matrix

Add tests for all of the following:

- one active and one passive cell with an analytic backward-Euler solution;
- flux in both directions and exact defect sign;
- several active faces sharing one passive cell/group;
- large signed cancellation with a deliberately large absolute defect;
- zero/tiny passive group energy and attempted withdrawal;
- threshold equality, just-below, and just-above behavior;
- sparse, high, and complete activity;
- inactive-cell radiation state unchanged after acceptance;
- rollback after solver, positivity, defect, and remote-rank failures;
- retry fractions exactly cover the scheduled interval;
- identity and nonidentity active/canonical mappings;
- stable IDs above `2^32`;
- cross-rank faces and zero-owned ranks;
- AMR and redistribution invalidation;
- restart before and after accepted candidates;
- checkpoint persistence of signed/absolute accounting and cooldown;
- old checkpoint with incomplete-history marker;
- all-active route gives zero boundary defect and matches the global solver;
- grey and all multigroup energy groups;
- `scale_base`, Compton split retry, Doppler, radiation force, flux limiting, free-free coupling, and existing spectral roundoff repair;
- positivity and conservation of material-plus-radiation for all-active tests;
- no accounting commit from a rejected candidate;
- exception paths restore snapshots without allocation.

## Acceptance requirements

### Run acceptance gates

- Exit code zero.
- No nonfinite primitive, conserved, matrix, RHS, solution, residual, or defect value.
- No raw negative accepted group/material extent beyond the already documented scale-aware roundoff repair policy.
- True residual and componentwise/backward-error gates pass for every accepted solve.
- Passive radiation state is unchanged in Dirichlet mode.
- Cumulative signed and absolute defects are finite, checkpointed, reported
  against their diagnostic references, and have complete history for the final
  run.
- Positivity repair accounting is finite and below its declared hard limits.
- Rollback/halving, Compton retry, restart, AMR, redistribution, mapping, and zero-owned-rank tests pass.
- Initial and final HDF5/PVTU outputs are nonempty and contain exactly 128 H5 and 128 VTU pieces per endpoint.
- Configuration, output count, stopping rule, tolerances, physics, and endpoint work are identical within each Global/AutoPartial pair.
- Existing comparator, conservation, positivity, restart, and AMR gates pass.

### Performance guardrails

Correctness comes first, but the safe path must still be useful:

- production defect accounting adds at most one packed collective per candidate and no per-face collective;
- Dirichlet unknown count remains `Nactive * Ngroups`; no passive shadow rows;
- default-off detailed diagnostics add less than 2% overhead when enabled for attribution;
- production mode has no more than 10% end-to-end regression against the clean capped AutoPartial replay at equivalent accepted work;
- report accepted event throughput, retry count, retry work, positivity repairs, defect rejections, cumulative defect, I/O, and final synchronization together;
- never report nominal outer `dt` alone as performance.

## Explicit non-goals and forbidden shortcuts

- Do not relax the shadow row-residual gate.
- Do not floor a materially negative passive value after commit.
- Do not hide nonconservation by reporting only signed defect.
- Do not apply a global post-hoc renormalization without a separately reviewed physical policy.
- Do not commit a candidate on some MPI ranks before the collective decision.
- Do not reuse face maps across ownership/topology epochs.
- Do not force a radiation-only all-active solve across cells at incompatible time states.
- Do not assume a one-cell implicit diffusion halo is causal or sufficient.
- Do not enable fast-math.
- Do not change timestep quantum without remapping every bin and absolute ceiling.
- Do not overwrite live binaries or reuse mutable campaign roots.
- Do not claim full-run conservation quality from a restart whose defect history is incomplete.

## Literature basis

- Commerçon, Debout, and Teyssier, “A fast, robust, and simple implicit method for adaptive timestepping with radiative diffusion” (2014): conservative Neumann-style synchronization can remove more energy than a coarse/passive cell contains and create negative energy; fixed Dirichlet inactive values are robust but nonconservative, with Robin as a compromise. [arXiv:1401.1112](https://arxiv.org/abs/1401.1112)
- Zier et al., implicit FLD in AREPO (2025): an active-plus-boundary-layer conservative approach trapped radiation and produced artifacts at time-bin boundaries because implicit diffusion is not limited to one cell per step; fixed inactive Dirichlet values were chosen, with small first-order energy error measured in their tests. [MNRAS article](https://academic.oup.com/mnras/article/545/3/staf2199/8377271)
- Pakmor et al. (2016) and AREPO-RT (2019) use active boundary layers/minimum adjacent timesteps successfully for explicit or differently structured transport, but those results do not establish safety for a strongly implicit whole-domain diffusion solve. [Pakmor et al.](https://academic.oup.com/mnras/article/462/3/2603/2589407) [AREPO-RT](https://academic.oup.com/mnras/article/485/1/117/5303742)
- CASTRO AMR radiation stores mismatched flux in registers and introduces it during a later implicit solve. This is a possible later conservative extension, but it adds checkpointed deferred state and must still handle an unpayable outgoing debt without creating negativity. [Zhang et al. 2011](https://arxiv.org/abs/1105.2466)
- PETSc distinguishes reported/preconditioned residuals from the true `b-Ax` residual; residual replacement literature explains why short Krylov convergence does not prove the final physical state is correct. [PETSc true-residual monitor](https://petsc.org/main/manualpages/KSP/KSPMonitorTrueResidual/) [Cools 2018](https://arxiv.org/abs/1809.01948)
- Componentwise backward error provides a scale-aware row test: `|r_i| / (|b_i| + sum_j |A_ij||x_j|)`. [LAPACK error bounds](https://www.netlib.org/lapack/explore-html/d1/db8/group__ptrfs_gaab83217859506b9826360272aa08a10f.html)

## Completion checklist for the main thread

- [ ] Record immutable source, executable, restart, environment, and placement provenance.
- [ ] Preserve unrelated work and old run artifacts.
- [ ] Classify CSR versus face residuals; do not loosen gates.
- [ ] Add policy enum and MPI-consistent runtime selection.
- [ ] Implement frozen-passive Dirichlet commit semantics.
- [ ] Implement signed, absolute, local, and cumulative defect accounting.
- [ ] Make accounting transactional and restart-safe.
- [ ] Add collective rollback and deterministic failure attribution.
- [ ] Pass focused serial tests.
- [ ] Pass focused MPI, mapping, zero-owned, AMR, and restart tests.
- [ ] Run the controlled stop-at-first-rejection replay.
- [ ] Validate safe capped benchmark before larger-bin stress tests.
- [ ] Complete paired Global/AutoPartial endpoint and artifact proof.
- [ ] Promote only after all hard gates pass.
