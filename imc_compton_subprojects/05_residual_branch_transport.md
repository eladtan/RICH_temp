# 05 Residual Branch Transport

## Purpose

Represent the remaining linearized Compton operator during packet transport by signed residual child packets emitted along path segments.

## Depends On

- `01_signed_packet_foundation.md`
- `02_compton_parameters_and_precompute_skeleton.md`
- `03_compton_matrix_port_and_fleck.md`
- `04_signed_group_source_generation.md`

## Code Areas

- `source/3D/radiation/RadiationIMC.cpp`
- `source/3D/radiation/RadiationIMC.hpp`
- `source/3D/radiation/RadiationIMC_RandomWalk.cpp`
- Existing `RadiationIMC::step(Particle&, std::vector<Particle>&)`

## Implementation Tasks

- Remove the unused marker for `particlesToAdd` in `RadiationIMC::step`.
- Add `BuildComptonResidualKernels` for Mode 1 with `comptonUseAnalogDirect=false`.
- Construct:
  - base effective scattering matrix `Hbase`
  - material implicit Compton matrix `Kmat`
  - residual matrix `R = Kmat - Hbase + S`
  - positive residual rows `Rpos`
  - negative residual rows `Rneg`
  - row CDFs for child group selection
- Add `EmitComptonResidualChildren` on every path segment using the segment-start packet state.
- Add `BankSignedChildPacket` to create child packets:
  - same cell path context
  - chosen target group
  - signed child weight
  - `initialWeight = abs(weight)`
  - birth time on segment
  - material energy update conservative with signed child creation
- Keep continuous absorption and time-average tallies signed.
- Disable random walk when `withCompton=true` unless random walk is extended to emit equivalent residual children.

## Endpoint

Residual branch packets are emitted through `particlesToAdd`, and their expectation matches the residual matrix for both positive and negative parent packets.

## Correctness Tests

- Residual branch expectation test:
  - one packet in source group `h`
  - known signed weight `W`
  - no absorption
  - no boundary crossing
  - fixed segment `Delta t`
  - over many histories, child mean satisfies `mean(W_child,g) = W c Delta t (Rpos[h,g] - Rneg[h,g])`
  - repeat for positive and negative parents
- Verify `Rpos` and `Rneg` are never sampled as negative probabilities.
- Verify material energy changes conservatively when residual children are created.
- Verify `withCompton=true` disables random walk or raises a clear guarded error.
- No-Compton regression remains unchanged.

## Stop / Do Not Include Yet

- Do not add deterministic parity harness beyond local diagnostics.
- Do not add physical analog Compton collisions.
- Do not add variance roulette or cancellation.
