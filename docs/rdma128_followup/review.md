# CPU transport follow-up: review and validation

This follow-up targets total Crooked Pipe runtime after fixed-step RDMA queues and fair receive scheduling delivered only a small gain. It does not attribute CPU savings to RDMA. The original working-source baseline and all previous RDMA code are preserved.

## Evidence and implementation

All 128 ranks were sampled at a 10 ms process-CPU interval in Slurm 10176469. Face intersection accounted for 23.55% of baseline samples and 23.61% of the earlier RDMA candidate's samples. These percentages include polling and startup CPU time; they are not exclusive wall-time fractions or an Amdahl speedup prediction. The profiler's initial library symbolization used login-node libc/libm files; compute-node copies were subsequently used. Stripped/internal library symbols remain unsuitable for precise function attribution. Application intersection addresses resolve to the frozen binaries' debug information.

The large starting-particle imbalance is not itself a timing weight. Release builds do count physics events per cell, and later-step event imbalance is much smaller than starting-particle imbalance. This follow-up therefore optimizes measured geometry and event arithmetic rather than changing partition weights on that evidence alone.

Changes:

- `STORM_OPTIMIZE_CPU_TRANSPORT` enables release `-O3`, `-march=x86-64-v3`, and `-ffp-contract=off` for STORM consumers. It is off by default and requires CPU-only GNU/Clang configuration and compatible execution CPUs. Only GCC 15.1 was tested here. Dependencies retain their existing compilation flags.
- Four double-precision faces are intersected at once using AVX2. Normals are read from the existing arrays; no cached pointer, geometry allocation, particle representation, or RDMA descriptor is introduced.
- When Doppler shift is exactly one, the event kernel reuses its already-computed attenuation exponential. Nonunit shifts retain the original second evaluation. RNG draws, physical models, cutoffs and time-step controls are unchanged.

## Code review

The vector dot products retain the scalar multiplication/addition order under contraction-off compilation. Outgoing and positive-time tests remain strict. Each four-face group resolves candidate hits in original index order, preserving the first face at exact edge/corner ties. Remaining faces use the original scalar loop. Parallel/non-outgoing lanes use a nonzero divisor and cannot win selection. Loads occur only for complete four-face groups, so no padding or overread is required. Non-double point/particle types and GPU builds retain the scalar implementation.

The first `-O3 -march=x86-64-v3` build with default contraction failed during MadVoro mesh construction (Slurm 10176470, abort 134); it was rejected. The strict variant completed the same case. This establishes that the tested strict configuration works; it does not prove the exact cause of the rejected build's mesh failure. The public build option explicitly disables contraction. No fast-math or approximate reciprocal option is enabled.

All manager/communication/reallocation source hashes match the previous reviewed version. The barriers around fixed-step descriptor capture and retirement, source-registration lifetime, payload completion before tail publication, cached-credit accounting, and between-step pair resize ordering remain intact. Compiler changes still require rerunning the protocol tests; source identity alone is not treated as validation.

## Tests

- 800,000 independent scalar/vector intersection comparisons passed with exact agreement in time, face, next cell, boundary flag and validity. Cases include 1–40 faces, unaligned starts, tails, random planes, exact corner ties, repeated faces, parallel and grazing velocities, slab mode, zero velocity, and starting on a face. No invalid/divide-by-zero floating-point exception was raised for these inputs.
- 300,000 complete IMC events matched the frozen pre-change attenuation kernel exactly, including particle position/velocity, remaining time, weight, frequency, event, RNG counter, and material/radiation/momentum tallies. Tests cover stationary, zero-velocity comoving, and moving frames and wide opacity/time/weight ranges.
- Both differential tests passed AddressSanitizer and UndefinedBehaviorSanitizer with the vector path enabled. This instruments the standalone tests, not the full MPI application.
- The CMake-built `storm_cpu_intersection_test` passed with the actual public optimization option enabled.
- Slurm 10176473 completed with exit 0. Both final native OFI/MPI RMA protocol tests and both release/debug 24-cycle conservation runs passed; maximum absolute normalized energy residual was 6.00716e-11. All six production runs completed. Final timing disposition is recorded in `results.md`.

## Scope

The benchmark is the same 120-cycle, 128-rank CPU/OFI Crooked Pipe segment, ending at 81.5719 ns. It is not a full 1000 ns end-to-end physics validation. Scheduling and population-control ordering remain stochastic; exact local kernel equivalence and conservation do not prove ensemble/time-to-accuracy equivalence. The AVX2 path is disabled in GPU builds. The shared attenuation reuse has not been tested on GPU. No claim of perfect correctness across all configurations is made.
