# MG cell-block Jacobi benchmark results

## Provenance

- Date: 2026-08-10
- Branch: `codex/individual-timesteps`
- Source commit before working-tree changes:
  `9dba0e18e842ac897a4622a01873dcbedc0209cb`
- Compiler/MPI: Intel OneAPI 2024.2.1 with OpenMPI 4.1.6
- Calibration binary SHA-256 (temporary replay enabled):
  `030a8889ee396a13bfc2136df95d00e04f45573726cfd4e6f57c4ea4ced2ca87`
- Replay-free production-candidate SHA-256:
  `52eeaf5f1312dd5d206a6343a7facef8b086b0797615743a00564c63230b53be`
- Configuration: one MPI rank, 4,096 cells, 16 energy groups, cooling
  limiter enabled, one global step with `dt = 0.0015`, AMR disabled for this
  solver-isolation test

The initial-state checksums matched exactly:

- XOR: `6b02fbd19b5e4c03`
- sum: `dc88eb80d92339e1`

## Same-matrix Krylov replay

The temporary replay captured the assembled distributed CRS matrix, right-hand
side, and initial vector from the scalar solve and replayed that exact matrix
with the cell-block preconditioner at the same rank count.

| Preconditioner | Iterations | Solver wall time (s) | Factor fallbacks |
|---|---:|---:|---:|
| scalar Jacobi | 8,893 | 60.5125 | 0 |
| cell-block Jacobi | 10 | 0.217037 | 0 |

This is an 889.3-fold iteration reduction and a 278.8-fold reduction in the
isolated solve wall time. The scalar and block solves both satisfied the
existing scalar-diagonal stopping checks. Their raw Krylov vectors differed by
5.90% in the worst component because the block solve drove the residual many
orders of magnitude lower before the next stopping check. Therefore final
physical artifacts, rather than the raw internal vector, are the acceptance
comparison.

## End-to-end physical A/B comparison

The scalar and block cases used identical mesh, initial state, physics, rank
count, timestep, and executable. Only `RICH_TEST_MG_PRECONDITIONER` changed.

| Metric | Scalar Jacobi | Cell-block Jacobi | Ratio or difference |
|---|---:|---:|---:|
| BiCGSTAB iterations | 8,893 | 10 | 889.3x fewer |
| evolution wall time (s) | 63.1860 | 1.30347 | 48.48x faster |
| radiation phase (s) | 62.8303 | 0.954331 | 65.84x faster |
| maximum radial-profile symmetric L1 | - | - | 7.70567e-5 |
| 16-group spectrum error | - | - | 9.49287e-3 |
| shock-radius error | - | - | 0 |
| material-energy relative difference | - | - | 3.74194e-8 |
| radiation-energy relative difference | - | - | 1.88031e-5 |
| total material+radiation energy difference | - | - | 4.34235e-9 |
| mass relative difference | - | - | 0 |
| maximum momentum-component relative difference | - | - | 8.19918e-10 |

Both runs retained positive material energy, nonnegative group energy, valid
Fleck factors, and the same final cell count and shock radius. The physical
comparison passes the repository's strict synchronized-lane limits of 0.5% for
radial profiles and 1% for the spectrum. These end-to-end numbers come from the
replay-free production-candidate binary and the permanent
`compare_physical_ab.py` verdict.

## Unit and solver-path coverage

- Runtime group counts: 1, 2, 3, 7, and 16
- Duplicate same-cell sparse entries
- Exclusion of neighboring-cell entries from each block
- Exact block residual and scalar-Jacobi parity
- In-place/alias-safe application
- Singular-block scalar fallback
- Candidate-local factor-storage release
- Global, serial-active, and distributed-active BiCGSTAB paths
- Full MG progress, convergence, `max0`, `max1`, cell/group identity, and
  matvec/exchange/reduction/setup/apply/total timing diagnostics

## MPI physical A/B gates

Both production-style MPI gates ran scalar and block Jacobi sequentially on the
same `bigrun` allocation and passed `compare_physical_ab.py`.

| Ranks | Cells | Scalar iterations | Block iterations | Solver speedup | Evolution speedup | Profile L1 | Spectrum error | Total-energy error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 32,768 | 1,982 | 10 | 109.90x | 23.46x | 3.61843e-5 | 2.07509e-4 | 1.02976e-9 |
| 128 | 524,288 | 1,229 | 10 | 70.59x | 14.65x | 2.74638e-6 | 5.47931e-5 | 1.29518e-10 |

- 8-rank Slurm job: `10111638`, completed `0:0` in 53 seconds.
- 128-rank Slurm job: `10111640`, completed `0:0` in 45 seconds.
- Result root:
  `regression_tests/results/mg_preconditioner_ab_20260810T201107Z`

The initial 128-rank conditioning-similarity calibration, job `10111639`, also
passed all physics comparisons and reduced 594 scalar iterations to 10, but its
batch verdict correctly failed because scalar Jacobi did not cross the required
1,000-iteration stress threshold. The retained retry doubled only that gate's
timestep; scalar then required 1,229 iterations and all acceptance checks
passed.
