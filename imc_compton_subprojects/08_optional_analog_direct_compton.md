# 08 Optional Analog Direct Compton

## Purpose

Add physical analog Compton group-changing events as a variance reduction and physical event channel, while preserving exact linearized parity with a residual correction.

## Depends On

- `01_signed_packet_foundation.md`
- `02_compton_parameters_and_precompute_skeleton.md`
- `03_compton_matrix_port_and_fleck.md`
- `04_signed_group_source_generation.md`
- `05_residual_branch_transport.md`
- `06_deterministic_parity_harness.md`
- `07_signed_tallies_mpi_boundaries.md`

## Code Areas

- `source/3D/radiation/RadiationIMC.cpp`
- `source/3D/radiation/RadiationIMC.hpp`
- Event selection logic in `RadiationIMC::step`
- Compton matrix and CDF data from earlier projects

## Implementation Tasks

- Add `comptonUseAnalogDirect`, defaulting to `false`.
- Add `BuildAnalogDirectComptonMatrixAndCdfs`.
- Construct a nonnegative analog direct Compton transfer matrix `Sanalog`.
- Set residual correction to `Sres = S - Sanalog`.
- Update residual identity to `R = Kmat - Hbase + Sres`.
- Add `directComptonOpacity` as an event channel in transport.
- Add `ApplyAnalogComptonEvent`:
  - sample target group
  - update packet frequency into the target group
  - exchange material energy according to signed packet energy change
  - preserve packet sign and set no negative opacity
- Avoid double-counting any existing elastic/electron scattering opacity.

## Endpoint

With `comptonUseAnalogDirect=true`, direct Compton group changes occur as physical events, and the residual still guarantees exact parity with the deterministic linearized operator.

## Correctness Tests

- Analog direct Compton event test:
  - sampled group-transfer frequencies match `Sanalog` row CDFs
  - material energy exchange equals signed packet energy change
  - packet sign is preserved
- Residual correction test:
  - `Sres = S - Sanalog`
  - `R = Kmat - Hbase + Sres`
  - `Hbase + R + Sanalog = Kmat + S`
- Compare `Sanalog` to `S`; residual should be small except known boundary or last-group terms.
- High-particle static single-cell run agrees with deterministic update within Monte Carlo error.

## Stop / Do Not Include Yet

- Do not make analog direct Compton the default until residual-only Mode 1 has passed all acceptance tests.
- Do not remove residual correction; it is required for exactness.
- Do not add variance controls unrelated to analog event correctness.
