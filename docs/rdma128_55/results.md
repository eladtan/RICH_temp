# RDMA toward 55 seconds at O2

**New median: 65.153 s, versus 68.901 s for the preceding optimized O2 build: 3.748 s / 5.44% less total runtime. The sub-55-second target was not achieved. It requires another 10.153 s / 15.58% reduction.**

## Controlled comparison

Slurm **10176482**, eight exclusive nodes `d25g[133-140]`, 16 ranks/node, 128 MPI ranks, one core/rank. GCC 15.1.0, OpenMPI 4.1.6, native OFI verbs/MSG. All 42 configured compile commands retain O2; no O3, Ofast, or fast-math options. Main compilation uses `-O2 -march=x86-64-v3 -ffp-contract=off`.

Same deck for both variants: `20000 2 10 --boundary-photons 100 --manager rdma --rdma-engine ofi --max-steps 120 --optimized-rdma`. The 20,000 argument controls background points; the graded wall layers produce 280,606 mesh cells. Every timed run completed 120 cycles, reaching **81.5719 ns**, not the full 1000 ns endpoint. Photon controls and physics parameters were unchanged.

| Variant | Total elapsed trials (s) | Median (s) |
|---|---|---:|
| Previous optimized O2 RDMA | 69.006, 68.568, 68.901 | 68.901 |
| Fixed-epoch progress shortcut + direct-log CPU sampling | 65.334, 64.648, 65.153 | 65.153 |

The median transport-loop sum decreased from **53.721 s to 50.237 s**. Each loop sum is the sum of reported loop times over the 120 cycles. The total figures above time the complete `mpirun`, including initialization, mesh setup, output, and shutdown.

**Storage control:** executables were staged to each node's local `/tmp`, and probe/stdout/stderr files were written locally. Staging and archival back to the shared filesystem were outside the timers, identically for both variants. This was necessary after shared-filesystem stalls added 12.48 s and 34.35 s to probe output in job 10176481; those timings were excluded from the final speedup. Unstalled previous shared-filesystem measurements were also approximately 68–69 s. The reported gain is not a promise about an overloaded shared filesystem. No compilation or profiling ran during this final comparison.

Exact script: [local_compare.sbatch](local_compare.sbatch). Per-run values: [final_results.json](final_results.json). Raw outputs: `/home/maorm/RICH/build/rdma128_55/local_10176482/`.

## Accepted changes and current build

Finalized after the user requested stopping speculative pursuit of 55 s. The selected configuration is the fastest repeatably measured one, not a proven global minimum. No additional tuning jobs were submitted for finalization.

A single-run launcher containing the selected binding, backend, scheduling and local-output settings is [run_128.sbatch](run_128.sbatch): `sbatch /home/maorm/RICH/docs/rdma128_55/run_128.sbatch`. It runs the same 120-cycle comparison case and archives its output after timing. It uses the current canonical executable and records its hash.

1. While fixed transport is active, RDMA skips handler scans and resize-request bookkeeping. Provider progress continues. The phase-entry/exit barriers, allocation guards, payload completion, tail publication and source-registration lifetimes retain their existing ordering.
2. New opt-in **`STORM_CPU_DIRECT_LOG=ON`** uses `-log(u)` instead of `-log1p(u-1)` in the shared CPU IMC event kernel. STORM's uniform lattice makes both expressions mathematically identical, but the math-library entry points may differ in the last rounded bit. RNG consumption is unchanged. This option is off by default; it is ON in this workspace's rebuilt canonical executable. It does not enable compiler fast math and is currently restricted to CPU builds.

The canonical executable is `/home/maorm/RICH/source/monte/examples/crooked_pipe/crooked_pipe`. It is **byte-for-byte identical** to the benchmarked `/home/maorm/RICH/build/rdma128_55/direct_log`: SHA-256 `2ce9fcb458acf9cdbb2cd8994649f37e49ada1c6ceb5e516dba9fe7c1bb34b3c`.

Relevant configuration:

```sh
cmake -S source/monte -B build/rdma128_20260912/candidate_build \
  -DSTORM_OPTIMIZE_CPU_TRANSPORT=ON -DSTORM_CPU_DIRECT_LOG=ON
cmake --build build/rdma128_20260912/candidate_build --target crooked_pipe -j4
```

These commands reuse this workspace's existing MPI/OFI/dependency configuration. Set `STORM_CPU_DIRECT_LOG=OFF` and rebuild to retain the former exponential-sampling rounding. That retains the fixed-epoch progress shortcut, whose isolated single-pair effect was only 0.64%; it is not the source of most of the new gain.

## Validation and review

- New release and STORM_DEBUG builds passed four-rank, 24-cycle conservation runs over **both OFI and MPI RMA**, with 7-entry rings, a 64-entry growth ceiling, four-event slices and 1-microsecond send age. All four maximum normalized energy residuals were **6.00716e-11**.
- The rebuilt O2 queue unit test passed saturation, wrap, 12 resize epochs, forbidden-operation guards, source registration and legacy reuse over MPI RMA. Native OFI full-application runs also exercised the new progress path. The underlying queue and backend code did not change in this round.
- The rebuilt scalar/vector geometry test passed **800,000 exact comparisons**.
- Native compute-node math checks passed **1,000,008** uniform-lattice values, including endpoints: maximum direct-log/log1p difference **1 ULP**, relative **2.22044e-16**.
- **300,000 complete IMC events**, under ASan/UBSan, preserved discrete event choices and RNG counters; maximum normalized floating state/tally difference **4.44089e-16**, including moving and stationary frames. Test and native output are archived here.
- Across the three full runs per variant, maximum absolute differences between mean probe histories were `[0.009618, 0.006632, 0.001021, 0.001105, 0.000867]` keV. These were smaller than the maximum repeat-to-repeat ranges for either variant at each probe. This is a consistency check with a small sample, not a formal statistical equivalence test. Full-history bitwise equality is not claimed.
- Full reallocation/lifetime reasoning: [review.md](review.md). Source change: [changes.patch](changes.patch); source and binary hashes are recorded alongside it.

## Why the target remains difficult

The all-rank timing build in job **10176479** separated active transport work from RDMA Progress/Poll/Flush calls. Over 120 cycles, it measured **53.911 s** as the sum of maximum loop times, **51.219 s** as the sum of maximum active-work times excluding those engine calls, and **39.344 s** as the rank-average active-work total. On the rank with the most active work in each cycle, prefix transfers totaled **1.856 s**, and engine calls inside active work totaled **2.063 s**. The prefix figure is included in engine time; do not add them. These are pre-direct-log measurements and per-step rank selections, not a single-rank wall-time breakdown.

That points to CPU transport work and its distribution as the dominant remaining opportunities. Removing the synchronous transfer protocol cannot by itself remove the roughly ten further seconds needed now. The gap between average and maximum work indicates an opportunity, but the load-balancing experiments below did not realize it. A larger CPU transport implementation change needs another measured trial; this report does not claim a demonstrated path below 55 s.

## Rejected experiments

| Experiment | Observation | Disposition |
|---|---|---|
| Extra SoA geometry/RW table caches | 0.74% median reduction with overlapping ranges; added 24 bytes per directed face | Reverted; see `docs/rdma128_next` |
| Fixed-epoch progress only | 67.944 s versus 68.382 s in one pair | Retained as small bookkeeping simplification; no large independent gain claimed |
| Early random-walk rejection | 69.829 s versus 67.944 s for progress-only | Reverted despite exact differential tests |
| Event weight scale 100 | 80.952 s versus 68.901 s | Rejected |
| Rebalance interval 20 | 74.327 s versus 68.901 s | Rejected |
| Sampled per-cell cost, interval 10 | Shared-I/O-confounded total; loop 54.492 s versus 53.457 s | Rejected; trial also included early-RW code |
| Sampled cost, intervals 5 / 2 | 71.444 / 77.541 s; repartitioning 8.008 / 13.547 s | Rejected; trial also included early-RW code |
| Moderate weight scale 2 / 0.5, node-local progress-only build | 70.787 / 67.958 s, single trials | Default retained; does not establish repeatable gain |

Sampling machinery and the rejected CPU changes are absent from the working source. Experimental snapshots, scripts and binaries remain under `/home/maorm/RICH/build/rdma128_55/`. Jobs 10176479–10176482 all completed with exit code 0. The preceding matched-CPU RDMA/P2P comparison is in `docs/rdma128_next/results.md`; this investigation focused on lowering RDMA's total time.
