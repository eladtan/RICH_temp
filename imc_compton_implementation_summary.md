# IMC Compton Implementation Summary

This document summarizes the current Compton implementation in
`source/3D/radiation/RadiationIMC.hpp` and `source/3D/radiation/RadiationIMC.cpp`.

## High-Level Design

The active IMC Compton path is an exact implicit scheme in Monte Carlo
expectation without Compton child packets.

When `withCompton=true`, IMC requires multigroup opacity and random walk is
disabled. The implementation keeps the Compton-modified Fleck factor, the signed
implicit source `Btotal = Bbase + Bcorr`, and the implicit matrix built from
the same local implicit coefficients used by the diffusion Compton solve.

Signed packet weights are allowed. Negative weights represent signed implicit
terms, but they are carried by ordinary packets. No residual child packets,
roulette, or census cancellation are used.

## Main Step Flow

1. `preStep(fullDt)` calls `precomputeComptonData(fullDt)`.
2. Per cell, `precomputeComptonData` builds group opacities, Planck fractions,
   old group radiation energy, `S`, `dSdUm`, `Upsilon`, `Gamma`, and the
   Compton-modified Fleck factor.
   The group absorption opacities use the same coupling limiter as
   `MultigroupDiffusion`, so an overheated radiation group cannot exchange more
   than the reference diffusion solve allows in one step.
3. `buildComptonSources` forms `Btotal_g = Bbase_g + Bcorr_g`.
4. `buildComptonInPlaceKernels` builds the Compton base effective scattering
   matrix `Hbase` and the signed residual sampler
   `R = Kmat - Hbase + S`.
5. `generateComptonParticles` samples beginning-of-step source packets from
   `abs(Btotal_g)` and assigns the packet-weight sign from `Btotal_g`.
6. During transport, Compton events are in-place: the packet changes direction,
   group frequency, and signed weight. Material receives `W_old - W_new`.
7. The diagonal part of the implicit matrix is represented by a continuous
   packet-weight correction in the same segment update that handles Fleck
   absorption.

## In-Place Matrix Sampling

For source group `h` and target group `g != h`:

```text
Hbase[h][g] = (1-f) * kappa_h * qBase_g
implicitKernel[h][g] = Kmat[h][g] - Hbase[h][g] + S[h][g]
ratio_hg = E_g / E_h
eventRate[h][g] = abs(implicitKernel[h][g]) / ratio_hg
```

At a Compton event:

```text
W_new = sign(implicitKernel[h][g]) * W_old * ratio_hg
material += W_old - W_new
```

For the diagonal:

```text
implicitDiagonalCorrection[h] =
    implicitKernel[h][h] + sum_{g != h} eventRate[h][g]
```

The packet segment evolves with effective opacity:

```text
f * kappa_h - implicitDiagonalCorrection[h]
```

The positive `Hbase` part is sampled as ordinary in-place effective scattering.
The signed residual reconstructs the remaining implicit Compton operator in
expectation, without creating child packets.

## RadiationIMCParameters

Current Compton-related parameters:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `withCompton` | `false` | Enables implicit in-place Compton. Requires `withMultigroupOpacity=true`; incompatible with random walk. |
| `comptonUseInduced` | `true` | Includes induced terms through occupation numbers in `S` and `dSdUm`. |
| `comptonAllowNZeroFallback` | `true` | Allows fallback to the `n=0` matrix if induced terms make the Fleck denominator invalid. |
| `comptonDebugParityCheck` | `false` | Checks that the in-place event rates plus diagonal correction reconstruct the deterministic implicit local matrix. |
| `comptonCheckSignedTallies` | `false` | Optional signed tally consistency check. |
| `comptonDiagnostics` | `false` | Prints implicit event count, source/continuous/event/removal material exchange, raw group-energy extrema, projected negative-census energy/count, Fleck/Gamma/Upsilon ranges, fallback count, and opacity-limiter count. |
| `comptonSignedTallyTolerance` | `1e-10` | Relative tolerance for signed tally checks. |
| `comptonMatrixSamples` | `200000` | Monte Carlo samples used by `ComptonMatrixMC` table generation. |

The source packet budget remains `newPhotonsPerCell` per active cell.

Initial Compton census packets are sampled from the existing multigroup
radiation field `Eg`, not from the material Planck spectrum. The material
Planck spectrum only enters thermal/source packet generation.

At census, signed packet weights are tallied normally, then negative Compton
group energies are conservatively projected to zero before they become the next
step's physical radiation field. The added radiation energy is debited from the
material so total cell energy is conserved. Diagnostics report the raw minimum
group energy and the projected amount.

## Regression Coverage

- Deterministic reference: `regression_tests/cases/till_compton/test.cpp`
- MC Compton case: `regression_tests/cases/till_compton_mc/test.cpp`

The MC case uses a one-cell closed-boundary Till setup, multigroup free-free
opacity, Compton enabled, random walk disabled, and Planck initial census photons
at the material temperature.
