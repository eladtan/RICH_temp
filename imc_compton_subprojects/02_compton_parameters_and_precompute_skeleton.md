# 02 Compton Parameters And Precompute Skeleton

## Purpose

Add the explicit Compton feature gate and frozen per-step data containers while keeping runtime behavior unchanged.

## Depends On

- `01_signed_packet_foundation.md`

## Code Areas

- `source/3D/radiation/RadiationIMC.hpp`
- `source/3D/radiation/RadiationIMC.cpp`
- `source/3D/radiation/MultigroupOpacity.hpp`
- `source/Radiation/OpacityCalculator.hpp`
- Existing multigroup opacity code and `ComputationalCell3D::energyBoundaries`

## Implementation Tasks

- Add `withCompton` to `RadiationIMCParameters`, constructor storage, and `operator<<`.
- Add a disabled-by-default Compton data skeleton, for example `ComptonCellData`, with zero initialization for all scalars, vectors, and matrices.
- Add group-center and group-width storage compatible with the existing `ENERGY_GROUPS_NUM` and energy boundary conventions.
- Add `precomputeComptonData(fullDt)` but make it inert unless `withCompton=true`.
- In `precomputeComptonData`, compute and store only:
  - group absorption opacity `kappa_g`
  - Planck group fractions `b_g`
  - Planck mean opacity `kappaP`
  - base source fraction `qBase_g`
- Add safe CDF helpers for future signed source generation, but do not use them in production flow yet.
- If `withCompton=true`, disable random walk or fail early with a clear error until Project 05 defines residual-safe handling.

## Endpoint

The code accepts `withCompton` and precomputes frozen group/opacity/Planck data, but generated particles and transport are still identical when Compton source and residual terms are disabled.

## Correctness Tests

- `withCompton=false`: no-Compton regression matches Project 01.
- `withCompton=true` with Compton source/residual disabled: build succeeds and no production behavior changes except optional diagnostics.
- Compare stored `kappaP` against `opacity->CalcPlanckOpacity(cell)` for representative cells.
- Verify `sum_g b_g` is 1 within tolerance for normal temperatures and groups.
- Verify `qBase_g = kappa_g * b_g / kappaP` when `kappaP > 0`, and a safe zero fallback when `kappaP <= 0`.

## Stop / Do Not Include Yet

- Do not port `S` or `dSdUm`.
- Do not modify Fleck factor.
- Do not emit signed Compton source packets.
- Do not create residual child packets.
