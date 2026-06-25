# 07 Signed Tallies MPI Boundaries

## Purpose

Make signed Compton packets safe across post-step tallies, MPI movement, boundary generation, and random-walk interactions.

## Depends On

- `01_signed_packet_foundation.md`
- `02_compton_parameters_and_precompute_skeleton.md`
- `03_compton_matrix_port_and_fleck.md`
- `04_signed_group_source_generation.md`
- `05_residual_branch_transport.md`
- `06_deterministic_parity_harness.md`

## Code Areas

- `source/3D/radiation/RadiationIMC.cpp`
- `source/3D/radiation/RadiationIMC_RandomWalk.cpp`
- `source/monte/MonteCarloParticle.hpp`
- MPI particle communication code
- Boundary source generation paths

## Implementation Tasks

- Audit `postStep` for assumptions that radiation packet weights are positive.
- Ensure radiation energy, group energy, and time-average tallies use signed packet weights.
- Keep packet count or sampling diagnostics based on absolute magnitude where appropriate.
- Confirm signed packets crossing MPI boundaries preserve `weight`, `initialWeight`, frequency, location, and time-left fields.
- Confirm boundary-generated particles remain positive and unchanged by Compton residual support.
- Add census cancellation only if it is needed for correctness of signed tallies; otherwise leave cancellation for Project 09.
- Enforce `withCompton=true` plus `withRandomWalk=true` behavior:
  - either random walk is disabled by parameter guard
  - or random walk emits mathematically equivalent residual children

## Endpoint

Signed packets move through the full IMC bookkeeping path without sign loss, incorrect deletion, or unsigned tally errors.

## Correctness Tests

- MPI and boundary packet check:
  - signed residual packets crossing MPI boundaries arrive with unchanged sign and magnitude
  - boundary source packets remain positive and unchanged
  - MPI reductions for source allocation use absolute source energy where allocation needs magnitude
- Signed post-step tally check:
  - positive and negative packets in the same group tally to their signed sum
  - group radiation energy can decrease because of anti-packets without unsigned clipping
- Random-walk-disabled test:
  - `withCompton=true` and random walk requested does not silently use old random walk
  - `withCompton=false` random walk behavior is unchanged
- No-Compton regression remains unchanged.

## Stop / Do Not Include Yet

- Do not add analog direct Compton collisions.
- Do not add roulette or aggressive packet cancellation unless required to pass signed bookkeeping tests.
