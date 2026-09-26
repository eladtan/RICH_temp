# OFI follow-up and Open MPI slowdown — 16 September 2026

Most of the original Open MPI slowdown came from the installed Open MPI 4.1.6 AVX reduction component, `mca_op_avx.so`. Selecting OFI in EasyRMA does not remove the benchmark's MPI reductions. Those reductions left the CPU in a state that made subsequent transitions between STORM's SSE instructions and the math library's AVX instructions expensive.

The runtime workaround tested here is to add this option to **Open MPI's** launcher:

```bash
mpirun --mca op '^avx' ...
```

It disables that MPI reduction component. It does not disable the OFI transport or change STORM, EasyRMA, compiler flags, or floating-point accuracy settings. It is an installation-specific workaround; these measurements do not establish that every Open MPI installation needs it. No new production source changes were made during this investigation.

## Evidence for the cause

- In the original measurements, all three Open MPI backends took about 12 seconds. The extra time appeared in particle handling, approximately 7.8 seconds versus 2.8 seconds with MPICH, while source generation was similar.
- CPU profiles localized the additional work to random-walk sampling and its `log`/`exp` calls. The sampled STORM function has the same instructions in both builds apart from addresses.
- Hardware counters recorded **88,526,561 SSE/AVX transition assists on rank 0** for Open MPI + OFI. The other ranks recorded 70–82 million each. MPICH + OFI recorded **zero on every rank**. Floating-point assist counts were identical (3,407 per rank), and floating-point control settings and CPU placement also matched.
- With only `--mca op '^avx'` added, Open MPI's transition assists became **zero on every rank**. The counter-instrumented simulation fell from 11.9078 to 7.2543 seconds. Use the repeated, uninstrumented measurements below for the performance comparison.
- Disassembly of the installed AVX component contains no `vzeroupper` or `vzeroall` instructions. This, together with the controlled runtime test, points to missing AVX-state cleanup in that component. Intel documents the [AVX/SSE transition penalty and the cleanup instruction](https://www.intel.com/content/dam/develop/external/us/en/documents/11mc12-avoiding-2bavx-sse-2btransition-2bpenalties-2brh-2bfinal-809104.pdf).

## MPICH + OFI with the original runtime settings

Six measured runs per case, after one warm-up, in balanced rotating order:

| Configuration | Simulation median (s) | Min–max (s) |
| --- | ---: | ---: |
| Open MPI + OFI, original settings | 11.552 | 11.398–11.636 |
| MPICH + OFI | 6.456 | 6.325–6.568 |
| MPICH + P2P | 6.471 | 6.360–6.508 |

MPICH + OFI worked correctly and was effectively tied with MPICH + P2P. Both OFI configurations selected libfabric's native `verbs/ibv` provider, mode `MSG/RC`. These fresh measurements used the same nodes and case as the original comparison; absolute timings shifted slightly between allocations, so compare configurations within each series.

## Corrected six-way comparison

All Open MPI rows use `--mca op '^avx'`. MPICH settings are unchanged. These are six-run medians after one warm-up per configuration, in balanced rotating order; profiles/counters are disabled.

| Configuration | Simulation median (s) | Min–max (s) | Launch-to-exit median (s) |
| --- | ---: | ---: | ---: |
| Open MPI + OFI | 6.511 | 6.326–6.569 | 7.622 |
| MPICH + OFI | 6.502 | 6.314–6.575 | 7.320 |
| Open MPI + P2P | 6.553 | 6.413–6.655 | 7.599 |
| MPICH + P2P | 6.567 | 6.461–6.716 | 7.248 |
| Open MPI + MPI RMA | 6.885 | 6.438–7.136 | 7.950 |
| MPICH + MPI RMA | 6.480 | 6.420–6.550 | 7.151 |

All 42 corrected-series runs and all 21 original-settings follow-up runs passed the endpoint, energy, and temperature checks. The largest relative energy residual was `6.43324e-09`. Probe temperatures matched their common reference at output precision (maximum recorded difference `0.0` keV).

The major Open MPI slowdown is removed. Open MPI + OFI and MPICH + OFI now have essentially equal medians. Open MPI + MPI RMA has a somewhat higher median in this series, with overlapping observed ranges. These small four-rank measurements do not establish performance at production scale.

## Reproduction and raw evidence

The full original environment and build details are in [README.md](README.md). This follow-up used allocation 10185744 on exclusive nodes d25g133 and d25g135, four ranks total, pinned to OS CPUs 0 and 2 on each node. All runs used the same 3,600-cell, 1,039-cycle Crooked Pipe input. Each timing series was sequential; profiles and hardware-counter runs were separate from timing runs.

- `ofi_config.json`, `ofi_summary.json`: original runtime settings, now including MPICH + OFI.
- `corrected_config.json`, `corrected_summary.json`: six-way comparison with Open MPI's AVX reduction component disabled.
- `ofi_diagnosis.json`: exact component path/hash, hardware counters, and source-integrity check.
- `phase_totals.py`: summarizes existing phase logs. Per-step maximum-rank counters overlap and must not be added as an exclusive time budget.
- `results/diagnosis/`: all four rank profiles, counter CSVs, annotated/disassembled code, runtime probes, and phase totals. `profile_rank.sh` and `stat_rank.sh` are optional Linux/perf diagnostic launch wrappers; they are outside STORM.
- `results/ofi_20260916/` and `results/corrected_20260916/`: complete timing logs, probe histories, and exact per-run commands.

For a new allocation, update job ID and node names in the chosen config, then run `compare.py --config CONFIG --output NEW_DIRECTORY --repetitions 6`. See the original report for the installation's different Open MPI and MPICH launcher requirements. Simulation timings exclude launcher startup; launch-to-exit timings include it.
