# Returning optimized STORM to -O2

The optimized `-O2` executable takes **68.596 s** median, versus **66.127 s** at `-O3`: a **+3.73% runtime change**. It retains a **14.69% reduction** against the original `80.405 s` baseline. The canonical executable and CMake configuration now use `-O2`, as requested.

## Controlled measurements

Slurm 10176474: eight exclusive nodes `d25g[133-140]`, 128 ranks, 16 ranks/node, one core/rank. Three trials each of the optimized O2 and O3 binaries, interleaved; two original baseline runs bracket them. All timings include the complete `mpirun`, startup, output and shutdown. No profiler.

| Build | Full runtime trials, s | Median, s |
|---|---|---:|
| Original baseline, O2 | 80.743, 80.068 | 80.405 |
| Optimized, O3 | 65.763, 66.127, 66.362 | 66.127 |
| Optimized, O2 | 68.596, 68.487, 69.039 | 68.596 |

Same 120-cycle Crooked Pipe deck: `20000 2 10 --boundary-photons 100 --manager rdma --rdma-engine ofi --max-steps 120`. Both optimized builds use `--optimized-rdma`. All finish at 81.5719 ns. This remains a segment benchmark, not a full 1000 ns endpoint measurement.

## What changed

Only the CPU build option's forced `-O3` was removed. Standalone release compilation returns to its existing `-O2` flags. `-march=x86-64-v3`, `-ffp-contract=off`, four-face SIMD intersection, attenuation reuse, and the RDMA changes are retained. Runtime source hashes match the measured O3 version. No communication/reallocation code was changed. The actual CMake-generated compile flags are saved in `flags.make`; executable and source hashes are recorded beside this report.

GCC's `-O3` does not itself enable `-ffast-math` or `-funsafe-math-optimizations`. Floating-point contraction is controlled separately, so returning to O2 is not a substitute for `-ffp-contract=off`. The earlier architecture-tuned/default-contraction build failed mesh construction; the strict configurations were tested successfully. This does not establish that O3 alone caused the earlier failure. [GCC 15.1 optimization documentation](https://gcc.gnu.org/onlinedocs/gcc-15.1.0/gcc/Optimize-Options.html).

## Validation and review

- CMake rebuilt the application, queue protocol test, and intersection differential test at O2; compiler output has no remaining O3 override.
- The O2 CMake-built 800,000-case scalar/vector intersection test passed exact result comparisons.
- Native OFI and MPI RMA protocol tests passed saturation, partial sends, wrap, delayed consumption, 12 resize epochs, forbidden-operation guards, registration and legacy reuse.
- The rebuilt O2 application completed the native OFI 24-cycle conservation run with 7-entry initial rings, a 64-entry growth limit, 4-event slices and 1 μs cooperative flush age. Maximum absolute normalized residual: 6.00716e-11.
- All eight production trials completed successfully. This validates the measured configuration; it does not prove correctness for every backend or input.
- The preceding ASan/UBSan differential tests were already compiled at O2 with SIMD enabled. They were not rerun because no runtime source changed in this step.

## Current configuration

Keep `STORM_OPTIMIZE_CPU_TRANSPORT=ON` to retain the explicit SIMD/ISA work. It now preserves the configured optimization level instead of overriding it to O3. The current standalone release build uses:

```text
-O2 -march=x86-64-v3 -ffp-contract=off
```

The canonical binary is `/home/maorm/RICH/source/monte/examples/crooked_pipe/crooked_pipe`; its frozen O2 copy is `/home/maorm/RICH/build/rdma128_o2/candidate_o2`. The preceding O3 binary remains at `/home/maorm/RICH/build/rdma128_followup/candidate_final`. Execution nodes must support x86-64-v3. The precise allocation, flags and commands are in [compare.sbatch](compare.sbatch). Source/build change: [changes.patch](changes.patch).
