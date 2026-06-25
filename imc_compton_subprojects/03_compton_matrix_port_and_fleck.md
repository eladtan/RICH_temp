# 03 Compton Matrix Port And Fleck

## Purpose

Port the deterministic Compton matrix construction from `MultigroupDiffusion` into `RadiationIMC` pre-step data and use it to compute the modified Fleck factor.

## Depends On

- `01_signed_packet_foundation.md`
- `02_compton_parameters_and_precompute_skeleton.md`

## Code Areas

- `source/3D/radiation/RadiationIMC.hpp`
- `source/3D/radiation/RadiationIMC.cpp`
- `source/Radiation/MultigroupDiffusion.cpp`
- `source/Radiation/MultigroupDiffusion.hpp`
- `source/Radiation/CMMC/src/compton_matrix_mc.hpp`

## Implementation Tasks

- Add a `ComptonMatrixMC` member and initialize its tables when `withCompton=true`, mirroring `MultigroupDiffusion`.
- Port matrix construction into an IMC helper equivalent to `generate_S_and_dSdUm_matrices`.
- Store frozen per-cell:
  - occupancy `n_g`
  - `tau`
  - `dtau_dUm`
  - `S`
  - `dSdUm`
  - `Upsilon`
  - `Gamma = kappaP + Upsilon`
- Implement the same n=0 fallback rule used by deterministic Compton when the induced-scattering occupancy gives an unsafe Fleck factor.
- Replace the ordinary IMC Fleck factor with the Compton-modified value only when `withCompton=true`.
- Keep old Fleck logic exactly when `withCompton=false`.

## Endpoint

`RadiationIMC::preStep` computes frozen Compton matrices and modified Fleck factors that match deterministic formulas, but source generation and transport still do not apply Compton residual terms.

## Correctness Tests

- Force `S=0` and `dSdUm=0` after construction:
  - `Upsilon = 0`
  - `Gamma = kappaP`
  - modified Fleck equals old Fleck
- Print or expose one-cell matrix diagnostics and compare to `MultigroupDiffusion` for the same cell, groups, density, temperature, and radiation group energies.
- Confirm n=0 fallback sets the same occupancy and yields a valid `0 <= f <= 1`.
- Run no-Compton regression and confirm old behavior is unchanged.

## Stop / Do Not Include Yet

- Do not build `Bcorr` source packets.
- Do not emit residual children.
- Do not add analog direct Compton events.
