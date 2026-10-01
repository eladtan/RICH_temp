# individual_box_growth

This regression case covers box growth (`UpdateBoxSynchronized` and `BoxGrowthDue`) during
individual time stepping, where each cell advances in its own power-of-two time bin.

A Sedov blast (energy 1, ambient density 1, pressure 1e-5) starts in the rigid
box [-1,1]^3 and uses individual time steps from the first event (bins 30-40,
AutoPartial meshes). After every event the test runs the growth idiom of
`runs/BaseTDEComptonIndividual/test.cpp` unchanged:

- at a synchronized state, it calls `UpdateBoxSynchronized`;
- otherwise, if `pending || BoxGrowthDue`, it requests a synchronized event.

The blast reaches the walls well before the end time. The box then grows two
to five times in individual mode, and the run continues after each growth.
`test.cpp` (its header comment) lists every check. Rank 0 writes
`individual_box_growth_metrics.txt` and one `RICH_TEST_BOX_GROWTH` line per growth.
`check_individual_box_growth_case` in `regression_tests/lib/regression_checks.sh`
checks both.

## Running

The case is registered for serial and MPI (16 ranks, bigrun). The runner runs
the default variant; the other variants are environment switches:

```bash
./regression_tests/run_all.sh --test individual_box_growth --mode serial --compiler intel --partition bigrun --exclude <prescan list>
./regression_tests/run_all.sh --test individual_box_growth --mode mpi    --compiler intel --partition bigrun --exclude <prescan list>
RICH_TEST_BOX_GROWTH_GRAVITY=1 ./regression_tests/run_all.sh ...          # FMM self-gravity
RICH_TEST_FORCE_SYNCHRONIZED=1 ./regression_tests/run_all.sh ...          # one shared bin
RICH_TEST_BOX_GROWTH_GRAVITY=1 RICH_TEST_GRAVITY_STALE_CACHE=1 ./regression_tests/run_all.sh ...   # negative control: must FAIL
```

## What is checked

### At every growth (collective)

- **Cell counts and IDs.** Counts agree between the report and direct counts.
  IDs are globally unique (gathered by stable ID), new IDs are fresh, and the
  largest ID equals `MaxID`.
- **Scheduler states.** `IndividualStateSynchronized()` holds afterwards. Every
  scheduler state has begin tick = last primitive tick = current tick, its own
  cell's ID, and a positive interval no longer than its bin.
- **Box geometry and generators.**
  - The new box contains the old one and is larger.
  - Every wall that moved moved by the same distance, at least the legacy
    floor 5 x 0.03 V^(1/3).
  - Every generator lies strictly inside the new box, and its mesh and
    committed positions agree.
  - Old generators did not move; new ones lie outside the old box.
- **Old cells.** Each keeps every serialized primitive field bit for bit
  (`CellDigest`: density, pressure, velocity, internal energy, temperature, Erad
  and its two rate fields, cs, tracers, stickers, Eg). `dt` is per-step scratch,
  neither serialized nor compared. Each also keeps its committed point velocity,
  and no bin gets coarser.
- **New cells.** Each has every primitive field of the ambient reference state
  and no point velocity. New and volume-changed cells are no coarser than
  `seed_bin`, and `reseeded_cells` = new + volume-changed cells.
- **Extensives.** Every extensive field equals `PrimitiveToConserved` of its
  primitive on the rebuilt mesh, bit for bit.
- **Acceleration cache.** Without gravity the cache is invalidated (zero, no
  pending kick). With gravity it is refreshed (next section).
- **Time-step limits.** Every interval and bin fits a limit the test computes
  itself on the rebuilt mesh, without the library's synchronized-limit functions:
  - the wave-speed CFL (`CourantFriedrichsLewy::CellTimeSteps`, face velocities
    from the states' point velocities, ghost-exchanged under MPI);
  - the source limit `source_cfl * sqrt(width / |a|)`;
  - its own mesh-drift loop: closing speed over non-boundary faces, fraction
    0.25, floored at 1/16 of the cell's own limit.

  The drift term must tighten some cell. It tightened 458-605 cells per growth
  serial and 1003-2865 in MPI. The report's `smallest_limit` must equal the
  smallest expected limit.
- **Conservation bookkeeping.** The report totals equal the gathered per-cell
  sums, and the new cell volumes sum to the new box volume. The mass balance is

  `mass_after - mass_before - inserted_mass - sum_old rho (V_new - V_committed)`

  and it must be at most 1e-10 of `mass_before`; it measured at most 7e-15.

  - The legacy semantics recompute every extensive from its primitive on the
    rebuilt mesh. Old cells next to a moved wall therefore change volume and
    mass, so `mass_after - mass_before - inserted_mass` alone is not small (the
    "naive" residual, 0.003-0.057). The test reports it together with its
    box-volume form but does not fail on it.
  - The energy residual of the same decomposition is bounded loosely at 1e-2.
    The recompute really does change the total energy, by up to 8.7e-4 per
    growth (the pre-growth mismatch between the dual-energy primitive and the
    total-energy extensive).

### Gravity variant (`RICH_TEST_BOX_GROWTH_GRAVITY=1`)

The force is `ConservativeForce3D` over `FastMultipoleAcceleration3D` (order 2,
theta 1, leaf 64, as the TDE driver), with G = 0.03. At every growth:

- **Direct-sum comparison.** About 8000 cells are sampled by stable ID. For
  each, the cached acceleration is compared with a direct sum
  `G sum_j m_j (x_j - x_i)/|x_j - x_i|^3` over every cell's centroid and mass on
  the rebuilt mesh. The maximum error over the force scale `G sum_j m_j / r^2`
  must be at most 0.1. The rms error relative to the rms direct acceleration is
  also reported.
- **Unrelaxed source limit.** Every sampled interval and bin must fit
  `source_cfl * sqrt(width / |a_direct|)` of the direct acceleration, with no
  allowance. The measured margin is at least 5.1 serial and 2.1 MPI.
- **Pending kicks.** Every half kick must be pending, and the report must say
  the cache was refreshed.
- **Built-in sensitivity.** The same comparison is repeated for the pre-growth
  cache by ID (new cells zero). Its error must exceed the tolerance; it measured
  0.76-0.85. The test also reports how large the new cells' pull on old cells
  is: 0.20-0.24, so a refresh that left out the new mass would also fail.

**The event after each growth.** A recording subclass of the force counts, on
every rank, how each first-half kick was applied. That event must:

- kick every first half from the cache on every rank (no first-half kick on
  the geometry path);
- have a pending kick for every kicked cell;
- use exactly the refreshed cached acceleration of each kicked cell, matched by
  stable ID on rank 0 across ranks, bit for bit;
- build no first-half mesh (a numeric `first_mesh_max` below 1e-4 s).

### Every run

- **The zero-growth path.** At every synchronized state that does not grow the
  box, nothing may change: box, `MaxID`, time and tick, mesh and committed
  generators, and every field of every cell, extensive and scheduler state
  (including cached accelerations, pending kicks, point velocities and pending
  neighbour bins). At least one such state must occur where `BoxGrowthDue` is
  false.
- **The event after every growth.** Each growth must be followed by exactly one
  verified event (the loop runs past the end time for it), with finite totals.
- **Mass drift.** Mass drift must stay at most 1e-9 between growths; it measured
  at most 5e-15.
- **End of run.** The run must reach its end time with finite, positive totals.

## Calibration of the gravity tolerance (serial, same binary, 3 growths)

| FMM order, theta | max error / force scale | RMS(error / force scale) | rms error / rms \|a_direct\| |
|---|---|---|---|
| 2, 1.0 (default) | 0.040-0.042 | 0.010-0.012 | 0.032-0.037 |
| 4, 0.5 | 6.0e-4 - 8.0e-4 | 7.9e-5 - 1.0e-4 | 2.8e-4 - 2.9e-4 |
| 6, 0.3 | 2.1e-6 - 3.6e-6 | 3.0e-7 - 3.8e-7 | 1.0e-6 - 1.1e-6 |

The MPI default measured a maximum of 0.054-0.067. The error falls by four
orders of magnitude as the expansion tightens, so the refreshed cache converges
to the direct sum. The 0.1 tolerance sits above the default FMM error (at most
0.067 MPI). It sits below the smallest failure signal: a refresh without the new
cells' mass is off by at least 0.20, and the stale cache by at least 0.76.

## Negative control (`RICH_TEST_GRAVITY_STALE_CACHE=1`)

After each growth the test puts the pre-growth cache back (new cells zero,
pending flags untouched). In the final campaign, all three growths failed on the
gravity check alone (`stale_control_growths_caught 3`,
`stale_control_other_failures 0`, maximum error 0.80-0.84). The run reported
`pass 0` and exited 1, and the checker also rejects any metrics file with
`stale_cache_control 1`.

## Final campaign (2026-09-24, one build of the current tree)

Nodes were scanned before every submission: all idle bigrun nodes were clean,
and d25g2, d25g10 and d25g58 were always excluded. All directories are under
`regression_results/`.

| Variant | Directory | Result | Growths | Events after 1st growth | Wall s |
|---|---|---|---|---|---|
| serial default | 20260924_224357 | pass | 2 | 57 | 174 |
| serial gravity | 20260924_224746 | pass | 3 | 57 | 76 |
| serial gravity, stale-cache control | 20260924_224930 | fails as required | 3 caught | 57 | 75 |
| serial force_synchronized | 20260924_225107 | pass | 5 | 38 | 869 |
| MPI default, 16 ranks | 20260924_230604 | pass | 2 | 72 | 299 |
| MPI gravity, 16 ranks | 20260924_231200 | pass | 2 | 65 | 254 |
| MPI force_synchronized, reduced (5e4 cells, volume fraction 4e-5, end 0.32) | 20260924_231641 | pass | 3 | 26 | 48 |
| MPI force_synchronized, full size | 20260924_231819 | pass | 2 | 43 | 980 |
| calibration, order 4, theta 0.5 (sbatch, scratch) | - | pass | 3 | 57 | 117 |
| calibration, order 6, theta 0.3 (sbatch, scratch) | - | pass | 3 | 57 | 290 |

### Provenance

- git HEAD `e8c6ad9ecf6b78bf574e5816f65684473ca39801`. The tree is dirty by
  design (164 entries); the md5 sums below pin the relevant sources.
- Test binaries (sha256):
  - serial `669966767cdd6324c3e187e1dcf11ad3cfd09b798fe3dd9e28db6091729db838`
    (`build/thunder_jobs/intelRelease/individual_box_growth/rich_intelRelease`,
    built 22:44:22);
  - MPI `f965eb9c742ec2018255e352b4244d41e86febb6dabe78d135a4630ebc17363a`
    (`build/thunder_jobs/intelReleaseMPI/individual_box_growth/rich_intelReleaseMPI`,
    built 23:06:41).

  Each runner invocation rebuilds incrementally. The hashes were identical
  after every run of the campaign, and no source was newer than the binaries.
- md5 of the key sources, identical at the start and end of the campaign:

  | md5 | file |
  |---|---|
  | de5bf91ea0ac44fea7854bfa00347c5b | source/newtonian/three_dimensional/simulation/Simulation.cpp |
  | 04fa78ae1812cc66b7c123ebdc0eed28 | source/newtonian/three_dimensional/hdsim_3d.cpp |
  | 8068abed10ec7ac09a1ee080d03d605c | source/newtonian/three_dimensional/ConservativeForce3D.cpp |
  | b8abd944565dba5854c482df0188bdcc | source/newtonian/three_dimensional/simulation/IndividualTimeStep.cpp |
  | 3bade3f3d9463f84590e75642b8fe817 | source/3D/GeometryCommon/UpdateBox.cpp |
  | 19123562f7d87b521ea79f2c0747ed55 | source/3D/gravity/fmm/mpi/DistributedFmmGravityCalculator.cpp |
  | 1acdf77303647a8348b7f7e52753f525 | source/newtonian/three_dimensional/RoundCells3D.cpp |
  | 44e71eda5b09d2df0617218cc44857ac | source/newtonian/three_dimensional/FastMultipoleAcceleration3D.cpp |
  | 8b359a5703c62da053245501371174d4 | regression_tests/cases/individual_box_growth/test.cpp |
  | d441581e57fc306c13dffbac94112413 | regression_tests/cases/individual_box_growth/REGRESSION_INFO |
  | 6f721906cbbfa26d4432fea1133314ca | regression_tests/lib/regression_checks.sh |

## Not covered

- **Gravity together with force_synchronized:** not run.
- **An MPI rank that owns no cells:** not arranged. MeshDecomposer's load
  balancing assigns every rank cells, and growth rebuilds and rebalances the
  mesh.
- **Rigid walls only:** the case uses no other boundaries and no radiation.
- **No AMR:** the case does not combine growth with individual AMR.
- **The global-mode `UpdateBox` path:** not exercised here.

## Notes from building this case

- The serial build failed while the `migrationBuffers` check in
  `GrowDomainAtSynchronizedIndividualState` sat outside `#ifdef RICH_MPI`
  (fixed).
- Serial runs crashed after a growth in `RoundCells3D` SlowDown
  (`vector::at` on the unbuilt support points of a serial partial mesh); fixed
  in `RoundCells3D.cpp`. A control without growth completed before the fix.
- The force_synchronized variant is slow (about 7.7 s per all-active event at
  2.9e4 cells). The default `SourceTerm3D::ApplyIndividual`, used by
  `ZeroForce3D`, copies all extensives once per active cell, which is quadratic
  in the cell count.
