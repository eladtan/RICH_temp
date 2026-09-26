# Crooked Pipe discrepancy investigation

The large downstream delay in the 65-second performance case predates this
session's transport optimizations. The earlier reference comparison used a
numerically different case from IMC_paper. It does not invalidate the paper's
successful result.

## Evidence from archived runs

All arrival times below are the first 0.08-keV crossing, interpolated in log time.
The script `analyze.py` reads the original numeric probe files and records hashes.

| Configuration | P3 arrival (ns) | P4 arrival (ns) |
|---|---:|---:|
| Gentile digitized reference | 11.085 | 27.880 |
| Paper, job 10176427, IMC + DDMC | 12.880 | 30.765 |
| Same paper mesh, job 10176424, DDMC off | 13.470 | 33.896 |
| Performance case, frozen original, trial 1 | 38.458 | Not reached by 81.57 ns |
| Performance case, frozen original, trial 2 | 38.745 | Not reached by 81.57 ns |
| Performance case, optimized, three trials | 38.902–39.152 | Not reached by 81.57 ns |

At 81.5719 ns, P3 is 0.123876–0.124274 keV in the original performance
runs, 0.124471–0.124897 keV in the optimized runs, and 0.212007 keV in the
paper DDMC run (interpolated). The reference is approximately 0.218670 keV.
Thus the large deficit exists before the optimization; these data do not exclude
small changes hidden by Monte Carlo variation.

## Configuration mismatch

| Setting | Paper job 10176427 | Timed performance case |
|---|---:|---:|
| Cells | 771,345 | 280,606 |
| Background points | 20,000 | 20,000 |
| Additional channel points | 400,000 | 0 |
| First wall layer width | 0.0001 cm | 0.01 cm |
| Requested thick/thin layers | 188 / 8 | 4 / 3 |
| Tangential spacing | 0.06 cm | 0.04 cm |
| Layer growth | 2 | 2 |
| DDMC | On, tau threshold 3 | Off |
| Maximum timestep | 10 ns | 1 ns |
| Endpoint | 1000 ns | 81.5719 ns (120-cycle cap) |
| New/min particle knobs | 6 / 6 | 2 / 10 |
| Census floor / cap | 8 / 2000 | 10 / 200 (derived defaults) |
| Emission floor | 4 | 2 (derived default) |
| Rebalance interval | 5 | 10 |
| Transport work scale | 0.01 | 1 |
| MPI ranks | 480 | 128 |

The wall mean free path is 1/2000 = 0.0005 cm. The performance case's
first wall layer spans 20 mean free paths, versus 0.2 in the paper case.
The two runs share analytic geometry and physical coefficients, but neither
equal background-point counts nor approximate total cell counts imply equal
interface resolution or accuracy. The archived DDMC-off control supports a
large mesh-related contribution; it is not an isolated sweep of each mesh knob.
Requested layers outside the domain are skipped by the generator.

## Source comparison

`IMC_paper/runs/crooked_pipe/source_driver.cpp` contains the same shared-site
structured mesh generation, material labeling, probe calculation, source
configuration, and timestep setup as the frozen pre-task driver. Its difference
from the pre-task driver is an optional `CP_BRANSON_WEIGHTS` rebalancing branch;
that environment variable is explicitly unset in the performance and replay jobs.
The difference from the frozen pre-task driver to the current driver adds only
the transport experiment CLI/configuration controls. In particular, the previous
fix that shares tangential lattice sites across interface slabs is still present.

No transport code is rolled back based on the mismatched graph. No evidence here
establishes that the delicate reallocation protocol is universally bug-free.

## Fresh paired check

`replay.sbatch`, job 10176491, runs the frozen original and accepted optimized
binaries on the paper configuration through 40 ns, on the same 128 ranks and
nodes. Both use the full paper mesh and population knobs. Snapshots are omitted;
rank count, endpoint, and explicitly selected OFI backend differ from the original
480-rank paper run. This is a regression check, not a full 1-microsecond validation
or a runtime comparison to the published 49-minute run.

### Completed result

Job 10176491 completed with Slurm state COMPLETED / exit 0:0. Both binaries
completed 63 cycles to exactly 40 ns and performed 14 mesh redistributions.
The archived probe time/cycle columns are exactly equal across the paired runs.
DDMC activity is confirmed by nonzero diffusion and leakage counters.

| Probe | Original arrival (ns) | Optimized arrival (ns) | Maximum temperature difference (keV) |
|---|---:|---:|---:|
| P1 | 0.034146 | 0.034146 | 0.004298 |
| P2 | 0.897104 | 0.877044 | 0.005247 |
| P3 | 12.917168 | 12.923620 | 0.001010 |
| P4 | 31.286820 | 30.640001 | 0.002905 |
| P5 | Not reached by 40 ns | Not reached by 40 ns | 0.0001123 |

P3 arrival changes by 0.050%; P4 by -2.067%. The largest difference over all
five probe histories is 0.005247 keV; maximum per-probe RMS difference is
0.002014 keV. This one paired replay does not estimate ensemble uncertainty or
prove bitwise/statistical equivalence. It does exclude a regression comparable
to the previously observed factor-of-three P3 delay in this tested interval.
The later P5 arrival and full 1-microsecond evolution were not rerun.

Full MPI invocation time was 906.095 s original and 794.388 s optimized
(12.33% lower), one trial each. Reported simulation times were 902.148 and
790.568 s. This is a different workload from the 65-second performance case;
neither time is a full paper run or a repeated performance estimate.
Both binaries use strict O2; the optimized binary's SHA256 remains
`2ce9fcb458acf9cdbb2cd8994649f37e49ada1c6ceb5e516dba9fe7c1bb34b3c`.

## Disposition

No new transport or mesh source change was needed. The demonstrable error was
using the small performance case to assess agreement of the paper's validated
case. The comparison reports now explicitly identify this mismatch. The existing
performance improvements remain in place; accuracy-matched STORM-versus-Branson
performance has not been established by the earlier as-run comparison.

Artifacts: `comparison.png` / `comparison.pdf`, `results.json`, `mesh_hashes.json`,
`analyze.py`, `replay.sbatch`, and raw logs/probes in
`/home/maorm/RICH/build/cp_paper_regression/run_10176491/`.
