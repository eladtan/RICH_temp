# 06 Deterministic Parity Harness

## Purpose

Add a debug harness that proves the IMC Compton kernels reconstruct the deterministic linearized Compton update from `MultigroupDiffusion`.

## Depends On

- `01_signed_packet_foundation.md`
- `02_compton_parameters_and_precompute_skeleton.md`
- `03_compton_matrix_port_and_fleck.md`
- `04_signed_group_source_generation.md`
- `05_residual_branch_transport.md`

## Code Areas

- `source/3D/radiation/RadiationIMC.cpp`
- `source/3D/radiation/RadiationIMC.hpp`
- `source/Radiation/MultigroupDiffusion.cpp`
- Regression or diagnostic test harness files

## Implementation Tasks

- Add a debug-only parity function for one cell and fixed timestep.
- Export or print enough frozen IMC data to compare:
  - `kappa_g`
  - `b_g`
  - `kappaP`
  - `S`
  - `dSdUm`
  - `Upsilon`
  - `Gamma`
  - `Kmat`
  - `Hbase`
  - `R`
  - `Bbase`
  - `Bcorr`
  - `Btotal`
- Compare IMC and deterministic formulas using the same units and old-state data.
- Assert Mode 1 identity:
  - `Sanalog = 0`
  - `Sres = S`
  - `R = Kmat - Hbase + S`
  - `Hbase + R = Kmat + S`
- Add a high-particle statistical parity test for a static single-cell problem.

## Endpoint

A deterministic parity diagnostic can be run on demand and confirms that the IMC coefficients, sources, and residual kernels reproduce the deterministic linearized operator.

## Correctness Tests

- Deterministic matrix parity:
  - `Hbase + R` equals `Kmat + S` within roundoff
  - source equals deterministic `Bbase + Bcorr`
  - zero-Compton case gives `R=0` and `Bcorr=0`
- Static single-cell Monte Carlo parity:
  - high-particle IMC group means agree with deterministic one-step update within Monte Carlo error
  - material energy exchange agrees within Monte Carlo error
- Negative effective scattering case:
  - at least one `RnegRow[h] > 0`
  - anti-packets or signed residual packets appear
  - signed mean matches deterministic update
  - closed-cell total energy is conserved

## Stop / Do Not Include Yet

- Do not make parity diagnostics mandatory in production runs.
- Do not add analog direct Compton event sampling.
- Do not add variance controls.
