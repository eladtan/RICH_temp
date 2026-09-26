# Crooked Pipe: STORM Voronoi versus Branson Cartesian

**Accuracy scope:** the 65-second STORM configuration is not the validated
`IMC_paper` configuration. Its delayed downstream heating predates the recent
optimizations. These runtimes do not establish performance at matched reference
accuracy; see [the investigation](../cp_paper_regression/findings.md).

**Follow-up correction:** the original review missed Branson's enabled synthetic scattering workload, which STORM does not execute. These remain valid as-run timings, but the kernel workloads were not matched. See [discrepancy.md](discrepancy.md) for diagnosis and controlled follow-up runs.

Completed jobs 10176483 and 10176484, three runs per code on d25g133–140.
Both use 128 ranks, eight exclusive nodes, 16 ranks/node, one core/rank,
strict CPU O2/x86-64-v3/no contraction, and 120 cycles to 81.5718957163 ns.
Times include the full MPI invocation. Binaries and output were node-local;
staging and archival were outside the timer.

| Code | Mesh | Whole-run trials (s) | Median (s) | Median transport loop (s) |
|---|---|---|---:|---:|
| STORM RDMA/OFI, accepted optimized binary | 280606 refined Voronoi cells | 65.690, 65.939, 65.225 | 65.690 | 50.190 |
| Branson 0.83, history/SoA, MPI particle passing | 280000 Cartesian cells | 239.850, 240.292, 240.023 | 240.023 | 237.555 |

The Branson/STORM wall-time ratio is 3.654; STORM's
runtime was 72.63% lower for these configurations.
Its new measurements reproduce the earlier 65.153-second result to about one percent.

## Actual histories, median across repeats

| Code | Generated, including initial radiation where present | Transported, including carried census on every step |
|---|---:|---:|
| STORM | 223,024,401 | 383,002,812 |
| Branson | 213,339,196 | 215,478,325 |

Branson generated 4.34% fewer histories.
Its nominal budget is the rounded mean of the earlier measured STORM generation
count, but native source allocation and census policies produce different actual
counts. Histories do not specify equal event work or equal statistical error.

## Accuracy and interpretation

The user chose each code's own mesh. Total cell counts are close, but STORM has
114284 cells in the thin channel and 0.01-cm first wall layers; Branson has 37568
thin-channel cells and spacings 0.0625 x 0.08 x 0.08 cm. The resulting probe
histories differ materially. This experiment does not establish a speedup at
equal spatial accuracy or equal statistical error, nor a speedup from RDMA alone.
STORM retains random walk, census population control, and dynamic balancing.
Branson retains its native history transport and static METIS partitioning.
The initialization and staircase-boundary differences are detailed in review.md.

Mean final material temperatures, keV, after 120 cycles:

| Probe (r,z), cm | STORM | Branson |
|---|---:|---:|
| (0,0.25) | 0.455939 | 0.449872 |
| (0,2.75) | 0.283745 | 0.224107 |
| (1.25,3.5) | 0.124691 | 0.055338 |
| (0,4.25) | 0.056345 | 0.048856 |
| (0,6.75) | 0.046467 | 0.047687 |

## Validation and artifacts

All six runs exited successfully and completed 120 cycles. Branson's conservation
residuals are included in results.json. The paired STORM binary is byte-identical
to the accepted optimized O2 executable. MPI diagnostics on separate compute
nodes selected UCX and rc_mlx5 inter-node transport with the same environment.
No transport tuning, MPI protocol changes, or RDMA reallocation changes were
made for this comparison.

- `prepare.py`, `build.sh`: reproduce the isolated Branson case and strict O2 build.
- `run_128.sbatch`: stage and run pairs; pass 3 for three trials each.
- `branson_case.patch`, `review.md`: precise changes and review.
- `inputs_sha256.json`, `upstream_commit.txt`: provenance.
- `results.json`: exact timings, counts, and probe values.
- Raw logs: `/home/maorm/RICH/build/branson_cp128/run_10176483/` and
  `/home/maorm/RICH/build/branson_cp128/run_10176484/`.
