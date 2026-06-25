# 04 Signed Group Source Generation

## Purpose

Emit material source packets by group and sign using the total linearized Compton source `Btotal_g = Bbase_g + Bcorr_g`.

## Depends On

- `01_signed_packet_foundation.md`
- `02_compton_parameters_and_precompute_skeleton.md`
- `03_compton_matrix_port_and_fleck.md`

## Code Areas

- `source/3D/radiation/RadiationIMC.cpp`
- `source/3D/radiation/RadiationIMC.hpp`
- Existing `generateParticles(fullDt)`
- Existing `generateSingleParticle(cellIndex, cell)`

## Implementation Tasks

- Add `BuildComptonSources` after coefficient and Fleck construction.
- Compute and store per-cell, per-group:
  - `Bbase_g`
  - `Bcorr_g`
  - `Btotal_g`
  - positive and negative source CDFs
  - absolute source energy used for packet allocation
- Add `generateComptonParticles(fullDt)` and call it only when `withCompton=true`.
- Allocate source particle counts from absolute signed source energy so negative groups receive packets.
- For each created packet:
  - choose cell and group from absolute source CDFs
  - set frequency inside the chosen group
  - assign signed weight
  - set `initialWeight = abs(weight)`
  - debit or credit material energy by `-weight`
- Preserve ordinary `generateParticles(fullDt)` behavior when `withCompton=false`.

## Endpoint

With Compton enabled and transport suppressed in a closed single-cell setup, generated signed packets represent `Btotal_g` by group and sign, and material plus radiation energy is conserved immediately after source creation.

## Correctness Tests

- Single-cell source-only Compton correction:
  - no initial particles
  - no leakage
  - no physical scattering
  - call source generation
  - signed group sums equal `Btotal_g` within sampling tolerance
  - material internal energy changes by `-sum_g Btotal_g`
  - total material plus radiation energy is unchanged
- Zero-Compton source check:
  - `Bcorr = 0`
  - `Btotal = Bbase`
  - source packets reduce to ordinary positive thermal emission
- No-Compton regression remains unchanged.

## Stop / Do Not Include Yet

- Do not emit path-segment residual children.
- Do not alter event selection.
- Do not add analog direct Compton events.
