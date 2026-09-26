# Marshak backend comparison

This benchmark driver lives outside STORM and uses its existing Marshak problem 2 physics: a one-dimensional radiation wave, 0.2 cm domain, final time 1 ns, power-law opacity and EOS, a time-dependent thermal source at the left boundary, and reflecting remaining boundaries.

The final workload uses 128 cells, 2 new packets/cell, 128 boundary packets/step, and comb population control with target 4 packets/cell and ratio 6. The timestep schedule is the existing Marshak schedule (`dt_factor=1`). Random walk and the existing slab transport option are enabled; DDMC is disabled. The external boundary adapter advertises the reflecting faces to the slab implementation and counts injected/escaped energy without changing STORM's boundary physics.

Compare Open MPI 4.1.6, MPICH 4.2.0, and Intel MPI 2021.11, each with P2P, native EasyRMA OFI, and MPI RMA. The main comparison uses 8 ranks on four exclusive nodes. Separate 32-rank pilots are retained as diagnostic data, not a completed scaling comparison. Each configuration gets one full warm-up and three measured runs; measured configurations are interleaved in a reproducible shuffled order.

## Results — 16 September 2026

Eight ranks on four exclusive nodes; one warm-up and three measured repetitions per configuration. Seconds below are simulation-loop times. The full measured series contains 36 runs, including warm-ups.

| MPI library | STORM backend | Median (s) | Min–max (s) | Launch-to-exit median (s) |
| --- | --- | ---: | ---: | ---: |
| Open MPI 4.1.6 | P2P | 19.704 | 19.660–20.459 | 20.595 |
| Open MPI 4.1.6 | Native OFI | 15.983 | 15.941–15.996 | 17.038 |
| Open MPI 4.1.6 | MPI RMA | 15.849 | 15.821–15.869 | 16.834 |
| MPICH 4.2.0 | P2P | 20.265 | 20.104–20.271 | 20.752 |
| MPICH 4.2.0 | Native OFI | 16.055 | 16.054–16.091 | 16.732 |
| MPICH 4.2.0 | MPI RMA | 15.672 | 15.666–15.701 | 16.184 |
| Intel MPI 2021.11 | P2P | 35.888 | 35.732–36.013 | 37.720 |
| Intel MPI 2021.11 | Native OFI | 21.005 | 21.003–21.017 | 22.859 |
| Intel MPI 2021.11 | MPI RMA | 20.474 | 20.427–20.500 | 22.352 |

Both RMA backends take about 19–23% less time than P2P under Open MPI/MPICH. OFI and MPI RMA remain close to one another. Intel MPI is slower in the tested generic `verbs;ofi_rxm` configuration; this does not establish its best achievable performance on this fabric.

All 36 runs reached 1 ns in 7,912 cycles and passed the mandatory physical/backend-selection checks. Maximum relative energy imbalance was `1.77e-13`; maximum integral-relative temperature error against the diffusion reference was 5.30%. Profiles varied by up to 3.79% from the first P2P warm-up, including repeated P2P executions. These are physics-checked Monte Carlo timings, not evidence of identical histories or a completed reproducibility diagnosis. No samples in this fixed timing series were discarded based on profile agreement.

The optional Intel `mlx` attempt failed before MPI initialization: Intel's bundled libfabric does not export the `FABRIC_1.9` symbol version used by the executable linked against libfabric 2.6.0. Matching bundled development headers were not found locally. A separate compatible build is needed before comparing that stack; the failed attempt is not a timed sample. The runner's `--intel-provider mlx` option is diagnostic and must not be used with the present executable as if it were a validated configuration. Intel documents its [OFI providers, including mlx](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-guide-linux/2021-10/ofi-providers-support.html).

A separate PMPI audit counted 423,668 puts / 306,360,896 bytes at 8 ranks and 2,109,327 puts / 1,512,587,056 bytes at 32 ranks. These are aggregate MPI_Put payloads, including same-node transfers and control writes, not measured wire traffic. Average put size at 8 ranks was approximately 723 bytes. The case exercises many small transfers; it is not a bandwidth-saturation benchmark. Audited runs are excluded from the timing table.

Intel MPI also passed the focused two-rank RMA regression (transfers, resize, borrowed storage, deferred atomics) and the receive-cleanup regression (zero unretired receives). There were no additional STORM or EasyRMA source changes for this task: hashes match for all 258 tracked regular STORM files and 17 EasyRMA files present in the pre-task snapshot. The earlier authorized fixes remain in place.

Raw primary results: `results/measured_20260916/`. `summary.json` retains the compact timing table and sample values. The interrupted strict-profile runs and the separate 32-rank diagnostics are preserved but are not part of the table. Per-step maximum-rank phase counters overlap and must not be summed into an exclusive time budget.

## Fairness and timing

All three binaries use GCC 15.1.0, `-O2 -g -DNDEBUG -fopenmp`, and the same driver and STORM/EasyRMA sources. OFI support is enabled in all binaries. Open MPI uses UCX with `--mca op '^avx'` to avoid the already diagnosed installed-component transition penalty. MPICH uses UCX. Intel MPI uses `shm:ofi` with `verbs;ofi_rxm`; its startup diagnostics confirm libfabric 2.6.0. EasyRMA's native OFI backend uses `verbs/ibv`, `MSG/RC`, under every MPI implementation. Intel MPI using OFI internally does not make its MPI RMA backend the same implementation as EasyRMA's native OFI backend.

Ranks are pinned to OS CPUs 0,2 for two ranks/node, or 0,2,4,6,8,10,12,14 for eight ranks/node. All nodes are Xeon Gold 6534 with mlx5_0 InfiniBand. Open MPI and Intel MPI use their own SSH launchers; MPICH uses Slurm PMI2. There is one application thread/rank, and runs do not overlap.

The primary timer is the maximum rank's complete physical cycle-loop time, including a final MPI barrier. It excludes MPI startup, mesh creation, backend initialization, final profile output, validation, and destruction. Standard STORM per-step diagnostics remain enabled equally for every configuration. Launch-to-exit time is also retained in raw results. Backend selection is explicit, with fallback and provider checks.

## Numerical checks

Every accepted run must reach 1 ns with a complete phase history, conserve global material-plus-packet energy to relative 1e-6, and produce finite positive material temperatures. Gas and radiation profiles must have integral-relative L1 error below 10% against the existing 512-cell diffusion reference. Differences above 0.5% from the first P2P warm-up are separately flagged. The default runner stops on that flag; the reported Monte Carlo timing series explicitly uses `--record-profile-variation` to retain all physically accepted samples, without selecting runs by their agreement with one random history. This option does not weaken the energy, endpoint, finite-temperature, diffusion-reference, or backend-selection checks.

The driver's printed `diffusion_l1` retains the stock example's mean pointwise relative error for traceability. That quantity is sensitive to the different cold-temperature floors. The acceptance check instead computes `sum(abs(T-T_reference))/sum(abs(T_reference))` on the uniform simulation grid for both temperature fields. Both metrics are preserved in `runs.json` with distinct names.

The preliminary strict runs found 1.5–2.8% differences in OFI profiles, late in the simulation. Repeating the identical Open MPI P2P configuration also changed its material/radiation profiles by 2.24%/3.57%. Thus profile variability is not unique to OFI. Incoming particle order affects floating-point sums, and comb population targets use integer truncation of weight ratios; these are plausible amplification paths, but the exact first numerical divergence was not traced. Conservation and diffusion agreement alone do not prove that every stochastic trajectory is equivalent. The two initial flagged runs differed in their summed active-packet counts by only 0.007% and 0.011%, respectively. All logs, including interrupted strict series, are preserved.

Preliminary coarse-timestep IMC and DDMC pilots were excluded. The coarse timestep delayed the wave, so those runs are not valid performance comparisons. The DDMC pilot also exposed a limitation of the external boundary ledger: it counted source packets before DDMC's admission filtering. No production source fix was attempted for that diagnostic limitation. An accepted standard-timestep pilot with census target 15 preceded selection of the final target 4; its timings are separate calibration data.

## Files and reproduction

- `main.cpp`: external MPI driver; the original Marshak physics headers are included directly.
- `launch.py`: exact site launchers, node list, allocation ID and MPI environment settings. Update them for a new allocation.
- `compare.py`: sequential, repeated comparisons and checks.
- `metadata.json`: versions, flags, binary hashes, linked libraries, workload and integrity results.
- `source_before.json`: pre-task hashes of tracked STORM and EasyRMA files.
- `results/`: ignored raw logs, profiles, commands, builds, pilots and measurements.

Enable `BUILD_MARSHAK=ON`, `ENABLE_OFI_BACKEND=ON`, and `AUDIT_MPI_RMA=OFF` in the external parent CMake project. Build `marshak_mpi_rma` separately with each MPI wrapper; Intel's wrapper needs `I_MPI_ROOT` set to its installation. Each wrapper must use the same underlying C++ compiler.

Run `python3 regression_tests/mpi_rma/marshak/compare.py --output NEW_DIRECTORY --ranks 8 --repetitions 3 --record-profile-variation` after updating `launch.py` for an exclusive four-node allocation. Intel MPI's launch and external-libfabric settings follow its [scheduler documentation](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-guide-linux/2021-10/job-schedulers-support.html); actual provider selection is verified in the warm-up logs.
