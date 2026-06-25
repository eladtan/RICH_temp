# 09 Variance Controls And Diagnostics

## Purpose

Reduce signed-packet variance and add diagnostics after residual-only correctness is proven.

## Depends On

- `01_signed_packet_foundation.md`
- `02_compton_parameters_and_precompute_skeleton.md`
- `03_compton_matrix_port_and_fleck.md`
- `04_signed_group_source_generation.md`
- `05_residual_branch_transport.md`
- `06_deterministic_parity_harness.md`
- `07_signed_tallies_mpi_boundaries.md`
- Optional: `08_optional_analog_direct_compton.md`

## Code Areas

- `source/3D/radiation/RadiationIMC.cpp`
- `source/3D/radiation/RadiationIMC.hpp`
- Diagnostics and regression output plumbing
- Any packet census or cancellation utilities

## Implementation Tasks

- Add residual child roulette with unbiased weight adjustment.
- Add census cancellation for nearby positive and negative packets only after signed tally correctness is stable.
- Add diagnostics for:
  - total positive residual source
  - total negative residual source
  - residual child count
  - anti-packet count
  - negative group radiation from Monte Carlo noise
  - cells using n=0 fallback
  - cells with Compton disabled or limited by safeguards
- Add thresholds or warnings for negative group radiation caused by sampling noise, without clipping production physics by default.
- Keep diagnostics off or low-volume by default.

## Endpoint

Variance controls reduce packet count or signed variance without biasing means, and diagnostics identify unstable Compton regimes without changing the verified core algorithm.

## Correctness Tests

- Roulette unbiasedness:
  - residual child expectation is unchanged within Monte Carlo error
  - energy conservation holds in a closed cell
- Cancellation sanity:
  - cancelling positive and negative packets preserves signed group energy
  - no cancellation occurs across incompatible groups, cells, times, or frequencies
- Diagnostics sanity:
  - negative group radiation warnings trigger in constructed noisy cases
  - ordinary no-Compton and stable Compton runs remain quiet
- Regression comparison:
  - variance controls disabled reproduces Project 06/07 baselines
  - variance controls enabled matches means within Monte Carlo error

## Stop / Do Not Include Yet

- Do not use variance controls to hide failed conservation or parity tests.
- Do not clip negative group radiation by default.
- Do not change deterministic Compton formulas.
