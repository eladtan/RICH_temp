# Pause point, 2026-10-01: goals met, cleanup done, s40 equivalence check pending

Resume phrase: **"resume the s40 equivalence check"**.

## State at pause

The working tree is uncommitted, on branch `auto/20260929-2340-RICH-ablation-integr-0daf` (user's branch:
`codex/individual-timesteps`). Nothing was committed.

Abbreviations:
- RUN = `runs/M05R05MBH1e4MGComptonIndividual`
- SP = `/home/elads/mpi_tmp/claude-5014/-home-elads-RICH-ablation-integration/8b3f6501-5576-40d5-ae43-ef71993c29ce/scratchpad`

### 1. Individual-timestep goals: met (final run TDE_P1, job 10234880, rich_s39_20261001)
- **Speed:** 1.765x vs global G50 (16531 s vs 29173 s, snap70 -> t=50).
- **Minimum dt:** run-minimum dt ratio 0.970.
- **Settings:** K=4, `RICH_RADIATION_MOMENTUM_POSITIVITY=1`.
- **Retries:** 69, vs 675 (F2) and 701 (G50).
- **Energy:** closure at roundoff.
- **Evidence:** `RUN/notes/TDE_P1_10234880/`.
- **Design and results:** `docs/fixes/radiation-momentum-positivity-design-2026-09-30.md` and the goals log `docs/fixes/individual-timesteps-partial-event-plan-2026-09-29-goals.md`.

### 2. Cleanup: deletions done (user-approved list, astra reviews 3 and 4 approved)
- **`/data/users/elads`:** 264 `TDE_*` directories deleted (about 1.36 TB); the 35 kept directories are intact.
- **RUN:** 510 files deleted (about 22 GB): superseded executables (provenance text kept), unreferenced logs, old submit scripts.
- **SP:** 324 items deleted (about 17.5 GB).
- **Lists and logs:**
  - `SP/cleanup_{rundir,data,scratch}.tsv`;
  - `SP/cleanup_*_deleted.log` (every deleted path);
  - `SP/cleanup_final_list.md`, copied to `RUN/notes/cleanup_final_list.md`;
  - generator `SP/cleanup_inventory.py`;
  - executor `SP/cleanup_execute.sh`.
- **Crash-50107 gate:** its staged copies (`TDE_crash_FLOOR*`) were deleted. Re-stage them from `TDE_step1_r2off` with `stage_restart_copy.sh` (session e01101c6 scratchpad) before using `submit_floorgate.sh`.

### 3. Dead-code removal: s40 built, reviewed, equivalence pending
- **Binary:** `RUN/rich_s40_20261001` (provenance file next to it). It is s39 minus the 9 default-off investigation diagnostics:
  - `RICH_RADIATION_SOLVER_CONTROL_TIME/_DIR`
  - `RICH_RADIATION_ROW_DIAGNOSTIC`
  - `RICH_RADIATION_POSITIVITY_DIAGNOSTIC`
  - `RICH_RADIATION_ACTIVE_SOLVER_CONTROL_TICK/_DIR`
  - `RICH_INDIVIDUAL_PASSIVE_TRANSPORT_TRACE`
  - `RICH_INDIVIDUAL_EVENT_DUMP_TICK/_DIR`
  - `RICH_INDIVIDUAL_CLOSURE_REEXPAND_TICK`
  - `RICH_RADIATION_TRACE_CELL_IDS`
  - `RICH_INDIVIDUAL_RADIATION_LIMIT_TRACE`
- **Diff:** `SP/s40_vs_s39.diff`, pure deletions (951 lines removed, 1 line simplified). Files:
  - `Diffusion.cpp`
  - `RadiationDriver.cpp`
  - `hdsim_3d.cpp`
  - `default_extensive_updater.cpp`
  - `Simulation.cpp`
- **Done:**
  - clean build;
  - MPI unit suite with the flag off and on: both `test_passed.res` (`SP/unit_run_s40_{off,on}`);
  - astra review: APPROVE (`SP/astra_s40.md`).
- **Pending: bitwise equivalence s39 vs s40.**
  - Jobs **10235273** (s39) and **10235274** (s40), data dirs `/data/users/elads/TDE_EQ39_ADAPT` and `TDE_EQ40_ADAPT`.
  - Both restart from F2's `snap_full_75` to t=49.0 with the production env plus frozen wall-time decisions (`ACTIVE_HILBERT_SEGMENT_REVERT=0`, `AUTO_REBALANCE=0`, `DWELL_WALL_MAX=0`, `PROBE_FRACTION=1000`, `FMM_GRAVITY_RESPLIT=0`).
  - Earlier pair, jobs 10235271/10235272 to t=48.92:
    - It ended at an identical final time but had 0 individual events: 58 global steps, while entry happens after about 61.
    - Hence the rerun to t=49.0.
  - Pair 10235269/10235270 aborted: invalid `RICH_TDE_TERMINAL_OUTPUT_TIME` (setup error).
- **Not done:** item 10, removal of `RICH_INDIVIDUAL_RADIATION_CANDIDATE_ACCURACY`. It touches the production retry controller (`RadiationStep.cpp`) and the unit test; it is planned as a separate, separately reviewed change (see `SP/cleanup_plan.md` section D).

## Resume steps
1. **Compare the pair.** Run from RUN:

   ```bash
   diff <(grep -a INDIVIDUAL_STATE_HASH exp_L50TADAPT_10235273.txt) <(grep -a INDIVIDUAL_STATE_HASH exp_L50TADAPT_10235274.txt)
   ```

   Also check `grep -ac 'RICH_STEP mode=individual'` (expect > 0) and that the `t_end=` sequences match.
2. **If the hashes are identical:** s40 is the new reference binary.
   - Mark its provenance "equivalence verified (10235273/10235274)".
   - Optionally delete the `TDE_EQ39/40_ADAPT` seeds (3 GB each); they were created by this session.
3. **If they differ:** find the first differing event and check whether a removed diagnostic changed any arithmetic or collective order.
4. **Item 10:** do it only after the user agrees.
5. **Still open from the goals work:**
   - a global flag-on control;
   - focused unit tests for the momentum-positivity matrix/exchange;
   - separating the about-2% radiation-energy difference (lumping vs the old undo);
   - a comment-only fix at the kinetic-cap underflow bound.
