# MG scalar-Jacobi preconditioner baseline

This record freezes the calibration evidence used to evaluate the cell-block
multigroup preconditioner.  The original logs remain authoritative and are
linked below; this file is the compact comparison record retained after any
temporary matrix replay tooling is removed.

## Provenance

- Branch: `codex/individual-timesteps`
- Source commit at cancellation: `9dba0e18e842ac897a4622a01873dcbedc0209cb`
- Compiler/MPI: Intel OneAPI 2024.2.1 with OpenMPI 4.1.6
- Energy groups: 16
- Cooling-limiter binary SHA-256:
  `20e2cf4965380145a247b1954381d0186ee1c167638ec252ebbb71d1934b36dc`
- No-limiter binary SHA-256:
  `6bd6ce3361ec25433e8fc9e0e2c77322db99450ad53d7e18ea4a1730bcec5a2e`
- Calibration size: 32,768 cells on 8 MPI ranks
- Initial temperature floor: `1e4 K`
- Pulse temperature: `3e7 K`
- Pulse radius: `0.2 R`
- Radiation physics: 16-group free-free absorption and scattering, Compton,
  Doppler, flux limiting, hydro feedback, and positivity protections
- Krylov solver: BiCGSTAB with scalar Jacobi and the fixed 10,000-iteration
  loop bound

## Campaigns

### Without cooling limiter

- Root:
  `regression_tests/results/lane_radiation_shock_20260810T174238Z`
- Calibration job: `10111611`
- Production/comparison jobs `10111612` through `10111615` were cancelled
  before running.

| Cycle | Simulation time | Proposed dt | Completion | Wall time (s) |
|---:|---:|---:|---:|---:|
| 42 | 1.1866682512403512e-05 | 4.745367730987227e-06 | 3.7013519832576986e-06 | 60.149915309 |
| 61 | 0.007089943252526776 | 0.0028359759957367362 | 0.0022114331862755224 | 127.408681763 |
| 65 | 0.027236716726240548 | 0.010894685385222242 | 0.0084954388361469443 | 188.061339783 |
| 68 | 0.074737545005809522 | 0.029895016697049828 | 0.023311482391301895 | 309.045734448 |
| 69 | 0.10463256170285935 | 0.041853023375869755 | 0.03263607494069299 | 376.369379334 |
| 70 | 0.14648558507872911 | 0.058594232726217652 | 0.045690504509840524 | 455.484062649 |

### With cooling limiter

- Root:
  `regression_tests/results/lane_radiation_shock_20260810T175550Z`
- Calibration job: `10111616`, cancelled after `00:41:19`
- Production jobs: `10111617`, `10111618`, and `10111619`, cancelled before
  running
- Comparison job: `10111620`, cancelled before running

| Cycle | Simulation time | Proposed dt | Completion | Wall time (s) |
|---:|---:|---:|---:|---:|
| 42 | 1.1866682512403512e-05 | 4.745367730987227e-06 | 3.7013519832576986e-06 | 60.219287978 |
| 61 | 0.007089943252526776 | 0.0028359759957367362 | 0.0022114331862755224 | 127.801189515 |
| 65 | 0.027236716726240548 | 0.010894685385222242 | 0.0084954388361469443 | 184.799144866 |
| 67 | 0.053383961650773928 | 0.021353583355035594 | 0.016651059141736826 | 251.978085150 |
| 68 | 0.074737545005809522 | 0.029895016697049828 | 0.023311482391301895 | 307.806045455 |
| 69 | 0.10463256170285935 | 0.041853023375869755 | 0.03263607494069299 | 405.366007719 |
| 70 | 0.14648558507872911 | 0.058594232726217652 | 0.045690504509840524 | 504.248726710 |
| 71 | 0.20507981780494677 | 0.082031925816704709 | 0.063966705906647076 | 589.958953515 |
| 72 | 0.28711174362165148 | 0.11484469614338659 | 0.089553387862176231 | 1079.303395968 |
| 73 | 0.40195643976503809 | 0.16078257460074122 | 0.12537474259991707 | 1174.037297610 |
| 74 | 0.56273901436577933 | 0.22509560444103768 | 0.17552463923275424 | 1719.467368294 |
| 75 | 0.78783461880681704 | 0.31513384621745272 | 0.24573449451872625 | 2081.025987172 |

The cooling-limiter run recorded seven nonconverged BiCGSTAB candidates, all
at 9,999 iterations.  The successive retry timesteps were:

1. `0.041016`, error `1.087835500087016e-09`
2. `0.020508`, error `1.085794716186191e-10`
3. `0.0803913`, error `1.536587797571839e-08`
4. `0.0401956`, error `3.670354606934033e-09`
5. `0.112548`, error `6.898363656551596e-06`
6. `0.157567`, error `9.770824652200795e-08`
7. `0.0787835`, error `6.926253052463791e-09`

At cancellation the current candidate had reached iteration 9,000 with error
`9.993679822376036e-07`; it had not been committed.

## Acceptance gate for cell-block Jacobi

For every scalar-Jacobi baseline solve above 1,000 iterations, cell-block
Jacobi must reduce the iteration count by at least a factor of three and must
also reduce total solver wall time.  It must preserve the existing convergence
metric, positivity and conservation checks, and converged physical result.
