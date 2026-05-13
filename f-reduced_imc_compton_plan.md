---
name: f-reduced IMC Compton
overview: Replace the unstable positive-kernel/residual-solve Compton IMC scheme with an f-reduced Compton scatter approach where the material energy exchange at each Compton event is reduced by factor f, combined with a bounded post-step redistribution correction that never touches material energy.
todos:
  - id: simplify-struct
    content: Remove old kernel fields from ComptonCellData, add comptonOutRate, comptonTargetCdf, comptonResidual arrays
    status: pending
  - id: remove-transport-mode
    content: Remove ComptonTransportMode enum and comptonTransportMode parameter field
    status: pending
  - id: remove-linear-solve
    content: Delete SolveComptonGroupSystem from anonymous namespace
    status: pending
  - id: rewrite-kernel-builder
    content: "Replace buildComptonInPlaceKernels with buildComptonEventData: compute physical Compton out-rate and target CDF per group"
    status: pending
  - id: rewrite-compton-event
    content: "Replace applyImplicitComptonEvent with applyComptonScatterEvent: f-reduced weight change w*(f*ratio + 1-f), material deposit f*w*(1-ratio)"
    status: pending
  - id: update-step-opacities
    content: Change step() to use comptonOutRate[group] instead of implicitEventRate[group] for Compton event opacity
    status: pending
  - id: update-step-call
    content: Change step() to call applyComptonScatterEvent instead of applyImplicitComptonEvent
    status: pending
  - id: rewrite-residual
    content: "Replace applyComptonResidualCorrection: no linear solve, no material deposit, compute delta_E_g from S/tau/Eg_time_avg, clamp negatives, redistribute"
    status: pending
  - id: apply-bres-direct
    content: In postStep, apply Bres[g] directly to conserved.Eg[g] before residual correction
    status: pending
  - id: remove-reconcile
    content: Remove reconcileComptonParticles function and its call from adjustExistingParticles
    status: pending
  - id: remove-mean-correction
    content: Remove comptonMeanRadiationCorrection member and all its usage
    status: pending
  - id: update-precompute
    content: Update precomputeComptonData to call buildComptonEventData, remove transport mode branches
    status: pending
  - id: cleanup-header
    content: "Update RadiationIMC.hpp: rename methods, remove dead declarations, remove unused fields from parameters"
    status: pending
  - id: cleanup-diagnostics
    content: Update printComptonDiagnostics to reflect new scheme (residual exchange always 0)
    status: pending
  - id: remove-parity-check
    content: Remove or simplify validateComptonParity (old kernel identities no longer apply)
    status: pending
  - id: test-stability
    content: Build and run till_compton_mc regression test, verify no negative energy errors and correct equilibration
    status: pending
isProject: false
---

# Stable f-Reduced Compton IMC Scheme

## Mathematical Foundation

The implicit material equation (Eq. 18 of `transport_fleck_derivation.pdf`) yields:

```
U_m^{n+1} = f * U_m^n + f*beta*dt*(c*sum_g kappa_g*E_g + M_C - chi_C*U_m^n)
```

The Compton contribution to material energy is `f * beta * dt * M_C` -- reduced by factor f relative to explicit. The proposed scheme reproduces this by depositing only `f * w * (1 - ratio)` at each Compton event instead of `w * (1 - ratio)`.

**Compton event mechanics (new):**
- Rate: `|S[h][h]| * c` (physical Compton out-scatter, unchanged)
- Target group sampling: `P(g|h) = tau[h][g]*(1+n[g]) / |S[h][h]|` (physical kernel, unchanged)
- New weight: `w_new = w * (f * ratio + (1-f))` where `ratio = center[g]/center[h]`
- Material deposit: `f * w * (1 - ratio)`
- Interpretation: fraction f gets full physical Compton, fraction (1-f) is "effective Compton" (weight-preserving)

**Effective scatter (unchanged):**
- Rate: `(1-f) * kappa_h * c` (via `baseEffectiveOpacity` summed over destination groups)
- Spectrum: `M_g / Gamma` normalized to 1 (via `baseSourceCdf`)
- Weight: unchanged
- Material deposit: 0

**Residual correction (post-step):**
- Per-group correction: `delta_E_g = c*dt*(1-f) * [C_g^energy - C_g^number - M_g*C_scalar/Gamma]`
  - `C_g^energy = sum_{h!=g} S[h][g] * <E_h>` (Compton in-scattering in energy units)
  - `C_g^number = sum_{h!=g} tau[h][g]*(1+n[g]) * <E_h>` (in-scattering in photon-number units)
  - `C_scalar = sum_h rowS[h] * <E_h>` (net Compton exchange scalar)
- Key property: `sum_g delta_E_g = 0` (pure frequency redistribution, zero net energy change)
- Material is NEVER touched by the residual -- eliminates the root cause of instability

---

## Files to Modify

1. **[`source/3D/radiation/RadiationIMC.hpp`](source/3D/radiation/RadiationIMC.hpp)** -- Simplify `ComptonCellData`, remove unused fields, add new fields
2. **[`source/3D/radiation/RadiationIMC.cpp`](source/3D/radiation/RadiationIMC.cpp)** -- Rewrite kernel building, event handling, and post-step correction

---

## Detailed Changes

### Phase 1: Simplify `ComptonCellData` struct (RadiationIMC.hpp, lines 55-101)

**Remove these fields** (no longer needed):
- `GroupArray implicitEventRate` -- replaced by physical Compton rate
- `GroupArray implicitDiagonalCorrection` -- artifact of old scheme
- `GroupArray positiveMeanRatio` -- artifact of PositiveKernelMeanEvent mode
- `GroupArray fullMeanRatio` -- artifact of PositiveKernelMeanEvent mode
- `GroupCdfMatrix implicitEventCdf` -- replaced by physical Compton CDF
- `GroupCdfMatrix positiveKernelCdf` -- no longer needed
- `GroupMatrix Kmat` -- intermediate, not stored
- `GroupMatrix Hbase` -- intermediate, not stored
- `GroupMatrix implicitKernel` -- replaced by simple physical kernel
- `GroupMatrix positiveKernel` -- eliminated
- `GroupMatrix residualKernel` -- eliminated
- `GroupMatrix implicitEventRateMatrix` -- eliminated

**Add these fields:**
- `GroupArray comptonOutRate` -- per source group h: `|S[h][h]|` (total Compton out-scatter rate)
- `GroupCdfMatrix comptonTargetCdf` -- per source group h: CDF over target groups from physical kernel `tau[h][g]*(1+n[g])`
- `GroupArray comptonResidual` -- per group g: the post-step redistribution correction (computed in postStep using time-averaged energies)

**Keep unchanged:**
- `GroupArray absorptionOpacity`, `planckFraction`, `baseSourceFraction`
- `GroupCdf planckCdf`, `baseSourceCdf`
- `bool active`, all scalar fields (`planckOpacity`, `volume`, `temperature`, `Um`, `beta`, `cv`, `fleck`, `Upsilon`, `Gamma`, `betaCdtF`, `useNZero`)
- `GroupArray oldRadiationEnergy`, `occupation`, `D`, `M`, `rowS`, `Lambda`
- `GroupArray Bbase`, `Bcorr`, `Btotal`, `Bpos`, `Bres`
- `GroupArray baseEffectiveOpacity` -- still used for `(1-f)*kappa_h` effective scatter rate
- `GroupMatrix tau`, `dtau_dUm`, `S`, `dSdUm`

### Phase 2: Remove `ComptonTransportMode` enum (RadiationIMC.hpp, lines 15-18)

Remove the enum:
```cpp
enum class ComptonTransportMode
{
    PositiveKernelResidualSolve,
    PositiveKernelMeanEvent
};
```

Remove the field `ComptonTransportMode comptonTransportMode` (line 40).

### Phase 3: Remove `SolveComptonGroupSystem` (RadiationIMC.cpp, lines 79-128)

Delete the entire anonymous-namespace function `SolveComptonGroupSystem`. It is no longer needed because there is no linear system to solve.

### Phase 4: Rewrite `buildComptonInPlaceKernels` -> `buildComptonEventData` (RadiationIMC.cpp, lines 1188-1247)

Replace the function entirely. New logic:

```cpp
void RadiationIMC::buildComptonEventData(size_t cellIndex, ComptonCellData &cd)
{
    (void)cellIndex;

    for (size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
    {
        // 1. Compute physical Compton out-scatter rate = -S[h][h] = |S[h][h]|
        cd.comptonOutRate[h] = std::max(0.0, -cd.S[h][h]);

        // 2. Build target group CDF from physical kernel: tau[h][g]*(1+n[g]) for g!=h
        GroupArray weights{};
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if (g == h) continue;
            weights[g] = std::max(0.0, cd.tau[h][g] * (1.0 + cd.occupation[g]));
        }
        cd.comptonTargetCdf[h] = RadiationIMC::buildSafeComptonCdf(weights);

        // 3. Effective absorption opacity (base effective scatter rate per source group)
        //    = (1-f) * kappa_h, summed over destination M_g/Gamma contributions
        //    This is the SAME as current: sum_g betaCdtF * kappa_h * (kappa_g * b_g)
        //    Keep existing baseEffectiveOpacity computation
        cd.baseEffectiveOpacity[h] = 0.0;
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double kgbg = cd.absorptionOpacity[g] * cd.planckFraction[g];
            cd.baseEffectiveOpacity[h] += cd.betaCdtF * cd.absorptionOpacity[h] * kgbg;
        }
    }
}
```

Note: `baseEffectiveOpacity[h]` formula is `sum_g betaCdtF * kappa_h * kappa_g * b_g`. Since `betaCdtF = (1-f)/Gamma` (when `|Gamma|>1e-200`), this equals `(1-f)/Gamma * kappa_h * sum_g kappa_g*b_g = (1-f)/Gamma * kappa_h * kappa_P`. This is the effective scatter rate from group h (weight-preserving events, re-emitted with spectrum `baseSourceCdf` = `M_g/Gamma`).

### Phase 5: Rewrite `applyImplicitComptonEvent` -> `applyComptonScatterEvent` (RadiationIMC.cpp, lines 1325-1407)

Replace entirely with the f-reduced Compton scatter:

```cpp
void RadiationIMC::applyComptonScatterEvent(
    size_t cellIndex, const ComputationalCell3D &cell, size_t sourceGroup,
    const Vector3D &oldVelocity, double oldWeight, double dopplerShift,
    Particle &particle)
{
    if (sourceGroup >= ENERGY_GROUPS_NUM) return;
    ComptonCellData &cd = this->comptonData[cellIndex];
    if (cd.comptonOutRate[sourceGroup] <= 0.0) return;

    // 1. Sample target group from physical Compton CDF
    size_t targetGroup = this->sampleComptonCdf(cd.comptonTargetCdf[sourceGroup], this->dist(this->re));
    if (targetGroup == sourceGroup || targetGroup >= ENERGY_GROUPS_NUM)
        return; // safety fallback

    // 2. New velocity (isotropic scatter)
    particle.velocity = this->opacity->getNewScatterVelocity(cell, particle);

    // 3. f-reduced weight change
    double ratio = this->comptonGroupCenters[targetGroup] / this->comptonGroupCenters[sourceGroup];
    double f = cd.fleck;
    double newWeight = oldWeight * (f * ratio + (1.0 - f));
    double materialDeposit = f * oldWeight * (1.0 - ratio);

    particle.weight = newWeight;
    particle.frequency = this->frequencyForComptonGroup(targetGroup);

    // 4. Deposit to material (only fraction f of the energy change)
    if (!this->noHydroFeedback)
    {
        this->conserved[cellIndex].internal_energy += materialDeposit;
        this->conserved[cellIndex].energy += materialDeposit;
        if (this->withHydro && !this->diffusionPressureGradient)
        {
            this->conserved[cellIndex].momentum +=
                (oldWeight * oldVelocity - newWeight * particle.velocity) * units::inv_clight2;
        }
    }
    this->comptonImplicitMaterialExchange += materialDeposit;
    ++this->comptonImplicitEventCount;
}
```

### Phase 6: Modify `step()` event opacity computation (RadiationIMC.cpp, lines 210-216)

Change `implicitComptonOpacity` from the old `implicitEventRate[group]` to the physical Compton out-scatter rate:

**Before:**
```cpp
double implicitComptonOpacity = (this->withCompton && group < ENERGY_GROUPS_NUM)
    ? this->comptonData[cellIndex].implicitEventRate[group]
    : 0.0;
```

**After:**
```cpp
double implicitComptonOpacity = (this->withCompton && group < ENERGY_GROUPS_NUM)
    ? this->comptonData[cellIndex].comptonOutRate[group]
    : 0.0;
```

### Phase 7: Modify `step()` scattering event call (RadiationIMC.cpp, line 330)

Change the function call from `applyImplicitComptonEvent` to `applyComptonScatterEvent`:

**Before:**
```cpp
this->applyImplicitComptonEvent(cellIndex, cell, group, oldVelocity, oldWeight, dopplerShift, particle);
```

**After:**
```cpp
this->applyComptonScatterEvent(cellIndex, cell, group, oldVelocity, oldWeight, dopplerShift, particle);
```

### Phase 8: Rewrite `applyComptonResidualCorrection` (RadiationIMC.cpp, lines 586-761)

Replace the 175-line function with a simple bounded redistribution. The new function:

1. Does NOT solve a linear system
2. Does NOT touch material energy
3. Only redistributes radiation between groups (sum = 0)

```cpp
void RadiationIMC::applyComptonResidualCorrection(double fullDt)
{
    double const cdt = units::clight * fullDt;
    size_t const Ncells = this->tess->GetPointNo();

    for (size_t i = 0; i < Ncells; i++)
    {
        ComptonCellData &cd = this->comptonData[i];
        if (!cd.active) continue;

        double const oneMinusF = 1.0 - cd.fleck;
        if (oneMinusF < 1e-15) continue; // f~1, no correction needed

        // Use time-averaged group energies (already computed in Eg_time_avg)
        // These are E_h in [energy/volume] averaged over the time step

        // Compute C_scalar = sum_h rowS[h] * <E_h>
        double C_scalar = 0.0;
        for (size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
            C_scalar += cd.rowS[h] * this->Eg_time_avg[i][h];

        // Compute per-group residual correction
        GroupArray delta{};
        double deltaSum = 0.0;
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double C_energy_g = 0.0; // sum_{h!=g} S[h][g] * <E_h>
            double C_number_g = 0.0; // sum_{h!=g} tau[h][g]*(1+n[g]) * <E_h>
            for (size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
            {
                if (h == g) continue;
                C_energy_g += cd.S[h][g] * this->Eg_time_avg[i][h];
                C_number_g += cd.tau[h][g] * (1.0 + cd.occupation[g]) * this->Eg_time_avg[i][h];
            }
            double Mg_over_Gamma = (std::abs(cd.Gamma) > 1e-200)
                ? cd.M[g] / cd.Gamma : 0.0;

            delta[g] = cdt * oneMinusF * (C_energy_g - C_number_g - Mg_over_Gamma * C_scalar);
            deltaSum += delta[g];
        }

        // Enforce sum=0 (numerical cleanup of roundoff)
        if (std::abs(deltaSum) > 0.0)
        {
            double correction = deltaSum / ENERGY_GROUPS_NUM;
            for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
                delta[g] -= correction;
        }

        // Apply with clamping: if E_g + delta[g] < 0, clamp and redistribute deficit
        double deficit = 0.0;
        double positiveTotal = 0.0;
        GroupArray finalEnergy{};
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double Eg = this->conserved[i].Eg[g];
            double proposed = Eg + delta[g];
            if (proposed < 0.0)
            {
                deficit += -proposed;
                finalEnergy[g] = 0.0;
            }
            else
            {
                finalEnergy[g] = proposed;
                if (delta[g] > 0.0) positiveTotal += delta[g];
            }
        }

        // Redistribute deficit: reduce positive corrections proportionally
        if (deficit > 0.0 && positiveTotal > 0.0)
        {
            double scale = std::max(0.0, 1.0 - deficit / positiveTotal);
            for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            {
                if (delta[g] > 0.0 && finalEnergy[g] > 0.0)
                {
                    double originalEg = this->conserved[i].Eg[g];
                    finalEnergy[g] = originalEg + delta[g] * scale;
                }
            }
        }

        // Write back
        double totalDelta = 0.0;
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double diff = finalEnergy[g] - this->conserved[i].Eg[g];
            this->conserved[i].Eg[g] = finalEnergy[g];
            totalDelta += diff;
        }
        this->conserved[i].Erad += totalDelta;
        // NO material energy change -- this is the key stability improvement
        this->comptonResidualMaterialExchange += 0.0; // explicit zero for diagnostics
    }
}
```

### Phase 9: Remove `reconcileComptonParticles` (RadiationIMC.cpp, lines 1851-1926)

This function was part of the PositiveKernelMeanEvent scheme (scaling particle weights to match solved group energies). It is no longer needed. Remove the function body and its call site in `adjustExistingParticles`.

### Phase 10: Remove `comptonMeanRadiationCorrection` member and usage

In `RadiationIMC.hpp`: remove the member `std::vector<GroupArray> comptonMeanRadiationCorrection`.

In `RadiationIMC.cpp`: remove all references to `comptonMeanRadiationCorrection` (allocated in `precomputeComptonData`, used in `applyImplicitComptonEvent` under `PositiveKernelMeanEvent` branch, and in `applyComptonResidualCorrection`).

### Phase 11: Update `precomputeComptonData` (RadiationIMC.cpp, lines 1525-1674)

Specific changes within the per-cell loop:

1. **Line ~1640**: Replace `buildComptonInPlaceKernels(i, data)` with `buildComptonEventData(i, data)`.
2. Remove allocation of `comptonMeanRadiationCorrection`.
3. Remove references to `comptonTransportMode`.
4. All other logic (Fleck factor computation, matrix building, source building) remains unchanged.

### Phase 12: Update `postStep` residual call context (RadiationIMC.cpp, lines 521-522)

The call `applyComptonResidualCorrection(fullDt)` remains in the same location. No change needed to the call site itself, only the function body (Phase 8).

However, remove the `comptonCheckSignedTallies` block (lines 523-554) that enforces non-negative Eg after the residual -- the new clamping logic in Phase 8 already ensures this.

### Phase 13: Update `RadiationIMCParameters` (RadiationIMC.hpp)

Remove:
- `ComptonTransportMode comptonTransportMode` field
- `bool comptonDebugParityCheck` (optional, keep if parity check is still useful)
- `size_t comptonMatrixSamples` -- only if unused elsewhere

### Phase 14: Clean up diagnostics

Keep the existing diagnostic accumulators:
- `comptonContinuousMaterialExchange` (from continuous weight decay -- still present)
- `comptonImplicitMaterialExchange` (now tracks f-reduced event deposits)
- `comptonResidualMaterialExchange` (now always 0 -- can log confirmation)
- `comptonSourceMaterialExchange` (from Bpos particle creation -- unchanged)
- `comptonRemovalMaterialExchange` (from low-weight particle removal -- unchanged)

In `printComptonDiagnostics`: update labels to reflect that residual exchange is always zero.

### Phase 15: Update `generateComptonParticles` and `buildComptonSources`

These functions are UNCHANGED. The Bpos/Bres source splitting works the same way:
- `Bpos[g]` particles are created (drawing material energy)
- `Bres[g]` is applied directly to `conserved.Eg[g]` in postStep

The source correction `Bcorr` accounts for the Fleck linearization of the material temperature. This is independent of the transport event handling.

However, in the old scheme `Bres[g]` was folded into the residual RHS. In the new scheme, `Bres[g]` should be applied directly:

In `postStep`, after tallying particles into `conserved.Eg`, add `Bres[g]` to each group:
```cpp
for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    this->conserved[i].Eg[g] += cd.Bres[g];
```

This is a direct additive correction (negative values reduce group energy). If this makes a group negative, the redistribution clamping in Phase 8 handles it.

### Phase 16: Remove `validateComptonParity` if based on old kernel identities

The parity check validated kernel sum identities (`implicitKernel` row sums etc.) that no longer exist. Either remove it or rewrite to validate the simpler identity: `sum_g comptonTargetCdf weights = comptonOutRate[h]`.

### Phase 17: Header declaration updates

In `RadiationIMC.hpp` private methods section, rename/replace:
- `buildComptonInPlaceKernels` -> `buildComptonEventData`
- `applyImplicitComptonEvent` -> `applyComptonScatterEvent`
- Remove `reconcileComptonParticles`
- Remove `SolveComptonGroupSystem` (it was in anonymous namespace, not declared here)

---

## Verification Plan

After implementation, run:
```bash
./build_rich.sh gnuReleaseMPI --test_name=till_compton_mc
sbatch --wait --exclusive --partition=bigrun --ntasks=4 --wrap "mpirun -np 4 ./build/gnuReleaseMPI/rich"
```

Check:
1. No "Negative internal energy" errors
2. `event_material_exchange` and `residual_material_exchange` are well-behaved (residual should be 0)
3. Temperature/radiation equilibrium matches expected Compton equilibration
4. Compare with multigroup diffusion result for same problem

---

## Summary of Stability Improvement

| Property | Old Scheme | New Scheme |
|----------|-----------|------------|
| Material deposit per event | `w*(1-ratio)` (full) | `f*w*(1-ratio)` (reduced by f) |
| Residual touches material | YES (linear solve) | NO (pure radiation redistribution) |
| Residual can make groups negative | YES (unclamped solve) | Clamped with energy conservation |
| Cancellation of large terms | event ~7e16 vs residual ~-6.6e16 | event ~f*7e16 (smaller), residual is redistribution only |
| Linear system solve | YES (ill-conditioned) | NO (direct formula) |
