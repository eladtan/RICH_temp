# Crooked Pipe backend timing — 16 September 2026

**Follow-up:** The large Open MPI gap below was traced to its installed AVX reduction component. See [OFI_DIAGNOSIS.md](OFI_DIAGNOSIS.md) for the cause, MPICH + OFI measurements, and the corrected six-way comparison. These original results are retained as the baseline.

Medians of five measured runs after one full warm-up per configuration. All runs used the same 3,600-cell Crooked Pipe case and completed 1,039 cycles to 1000 ns. Runs were sequential, in a balanced rotating order.

| Configuration | Simulation median (s) | Simulation min–max (s) | Launch-to-exit median (s) |
| --- | ---: | ---: | ---: |
| Open MPI + OFI | 12.02 | 11.80–12.23 | 13.22 |
| Open MPI + P2P | 12.26 | 12.04–12.31 | 13.26 |
| MPICH + P2P | 7.18 | 6.72–7.23 | 7.85 |
| Open MPI + MPI RMA | 12.01 | 11.92–12.08 | 13.12 |
| MPICH + MPI RMA | 7.05 | 6.67–7.23 | 7.69 |

MPICH was 1.71× faster for P2P and 1.70× faster for MPI RMA in the simulation phase. The differences among backends within each MPI implementation are small compared with their observed timing ranges. Open MPI + OFI and Open MPI + MPI RMA had essentially equal medians. These results describe this small four-rank case; they do not establish the ranking at production scale.

## What was held constant

- Two exclusively allocated nodes, `d25g133` and `d25g135`, Intel Xeon Gold 6534. Four MPI ranks total, two per node, one thread per rank. Both MPI implementations were explicitly pinned to OS CPUs 0 and 2 on each node; affinity was checked before timing.
- GCC 15.1.0, `-O2 -g -DNDEBUG -std=gnu++17 -fopenmp`, with identical source files and compile definitions. Both binaries include OFI support; the three Open MPI configurations use exactly the same executable. No STORM or EasyRMA production source was changed during this comparison.
- Open MPI 4.1.6 with UCX P2P and UCX RMA components; MPICH 4.2.0 `ch4:ucx`. Both use UCX 1.15.0 and `UCX_TLS=all`, allowing native network and shared-memory transports. OFI uses libfabric 2.6.0 native `verbs/ibv`, provider `verbs`, mode `MSG/RC`; its startup log confirms this selection.
- Default dynamic receive queues with initial capacity 5,000. No `--optimized-rdma`, tiny-ring stress settings, ISA-specific tuning, or MPI-call-counting wrappers.
- Benchmark arguments: `400 2 10 --wall-layers 1 --wall-points 400 --random-walls --boundary-photons 20 --final-time 1e-6 --energy-ledger`. Standard per-cycle logging, the energy ledger, and probe output are enabled for every run.

## Timing and validation

“Simulation” is the benchmark’s own `Finished at ... in ... s` timer for the cycle loop. “Launch-to-exit” is monotonic elapsed time around the complete launcher invocation, including initialization, mesh construction, transport setup, shutdown, and launcher overhead. File parsing is outside that timer.

The site requires different launchers: Open MPI is launched through `mpirun`/SSH because that installation lacks Slurm PMI support; MPICH is launched through `srun --mpi=pmi2` because its installed Hydra launcher requires an unavailable `libslurm.so.39`. Use the simulation column for the comparison without launcher overhead. The raw total times include those launcher differences.

All 30 runs (five warm-ups plus 25 measurements) passed the endpoint, energy, and temperature checks. The largest relative energy residual was `6.43324e-9`; all saved probe temperatures matched the Open MPI P2P reference at the output precision. Explicit manager/backend selection prevents silent P2P fallback. Known datatype-handle warnings from the MPICH benchmark are unchanged; no unfinished-request warning returned.

## Evidence and reproduction

- `summary.json`: all five timing samples for each configuration, medians, ranges, means, and standard deviations.
- `metadata.json`: hardware, compiler, MPI versions, binary hashes, source integrity, and numerical checks.
- `config.json`: exact commands and environment used, including allocated node names and job ID.
- `results/comparison_20260916/`: complete logs, probe histories, and per-run command/timing records (ignored build artifacts). `results/builds.json` records compiler flags and linked libraries; `results/*_affinity.log` records CPU placement.

Build the regression harness separately with each MPI toolchain, using `-DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_OFI_BACKEND=ON -DAUDIT_MPI_RMA=OFF`. Select matching C/C++ MPI wrappers and the installed libfabric include/library paths. Allocate two exclusive nodes, then update node names, job ID, binary paths, and `BENCHMARK_CPUS` in a copy of `config.json` for that allocation. Run:

```bash
python3 regression_tests/mpi_rma/timing/compare.py \
  --config regression_tests/mpi_rma/timing/config.json \
  --output regression_tests/mpi_rma/timing/results/new_comparison
```
