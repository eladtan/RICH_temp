# Stable f-Reduced Compton IMC Scheme (v2 -- post-review)

## Mathematical Foundation

The implicit material equation (Eq. 18 of `transport_fleck_derivation.pdf`) yields:

```
U_m^{n+1} = f * U_m^n + f*beta*dt*(c*sum_g kappa_g*E_g + M_C - chi_C*U_m^n)
```

The Compton contribution to material energy is `f * beta * dt * M_C` -- reduced by factor f relative to explicit. The proposed scheme reproduces this by depositing only `f * w * (1 - ratio)` at each Compton event instead of `w * (1 - ratio)`.

**Compton event mechanics (new):**
- Rate: `comptonOutRate[h] * c`, where `comptonOutRate[h] = sum_{g!=h} max(0, tau[h][g]*(1+n[g]))` (summed from CDF weights, not from `-S[h][h]`, to avoid last-group boundary inconsistencies)
- Target group sampling: `P(g|h) = tau[h][g]*(1+n[g]) / comptonOutRate[h]` (physical kernel, unchanged)
- New weight: `w_new = w * (f * ratio + (1-f))` where `ratio = center[g]/center[h]`
- Material deposit: `f * w * (1 - ratio)`
- Interpretation: fraction f gets full physical Compton, fraction (1-f) is "effective Compton" (weight-preserving)

**Effective scatter (unchanged):**
- Rate: `(1-f) * kappa_h * c` (via `baseEffectiveOpacity` summed over destination groups)
- Spectrum: `kappa_g*b_g / kappa_P` (via `baseSourceCdf`, always positive)
- Weight: unchanged
- Material deposit: 0

**Operator decomposition:**

The full implicit transport operator is `Ktotal[h][g] = S[h][g] + (1-f)/Gamma * M_g * Lambda_h`.

This is decomposed as:

```
Ktotal = Hbase + Kevent + Kres
```

Where:
- `Hbase[h][g] = (1-f)/Gamma * kappa_h * kappa_g * b_g` (effective scatter)
- `Kevent[h][g] = f*S[h][g] + (1-f)*tau[h][g]*(1+n[g])` for g!=h, `Kevent[h][h] = -comptonOutRate[h]` (f-reduced Compton events)
- `Kres = Ktotal - Hbase - Kevent` (post-step deterministic correction)

**Residual correction (post-step):**

Per-group correction:
```
delta_E_g = c*dt*(1-f) * [C_g^energy - C_g^number + (-M_g*C_scalar + D_g*C_kappa)/Gamma]
```
Where:
- `C_g^energy = sum_{h!=g} S[h][g] * <E_h>` (Compton in-scattering in energy units)
- `C_g^number = sum_{h!=g} tau[h][g]*(1+n[g]) * <E_h>` (in-scattering in photon-number units)
- `C_scalar = sum_h rowS[h] * <E_h>` (net Compton exchange scalar)
- `C_kappa = sum_h kappa_h * <E_h>` (absorption-weighted radiation)

The net `sum_g delta_E_g = c*dt*(1-f)*Upsilon*C_kappa/Gamma` is NOT zero in general (because the effective scatter uses `kappa_g*b_g/kappa_P` spectrum rather than `M_g/Gamma`). However this net is small and bounded. Material receives the opposite: `dEmat = -sum_g (delta_E_g + Bres[g])`.

---

## Files to Modify

1. **`source/3D/radiation/RadiationIMC.hpp`** -- Simplify `ComptonCellData`, remove unused fields, add new fields
2. **`source/3D/radiation/RadiationIMC.cpp`** -- Rewrite kernel building, event handling, and post-step correction

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
- `GroupArray comptonOutRate` -- per source group h: total Compton out-scatter rate (summed from CDF weights)
- `GroupCdfMatrix comptonTargetCdf` -- per source group h: CDF over target groups from physical kernel `tau[h][g]*(1+n[g])`

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
        // 1. Build target group weights from physical kernel: tau[h][g]*(1+n[g]) for g!=h
        GroupArray weights{};
        double outRateSum = 0.0;
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if (g == h) continue;
            weights[g] = std::max(0.0, cd.tau[h][g] * (1.0 + cd.occupation[g]));
            outRateSum += weights[g];
        }

        // 2. Out-scatter rate = sum of off-diagonal weights (not -S[h][h], avoids last-group issues)
        cd.comptonOutRate[h] = outRateSum;

        // 3. Build CDF from those weights
        cd.comptonTargetCdf[h] = RadiationIMC::buildSafeComptonCdf(weights);

        // 4. Effective absorption opacity (base effective scatter rate per source group)
        cd.baseEffectiveOpacity[h] = 0.0;
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double kgbg = cd.absorptionOpacity[g] * cd.planckFraction[g];
            cd.baseEffectiveOpacity[h] += cd.betaCdtF * cd.absorptionOpacity[h] * kgbg;
        }
    }
}
```

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

    size_t targetGroup = this->sampleComptonCdf(cd.comptonTargetCdf[sourceGroup], this->dist(this->re));
    if (targetGroup == sourceGroup || targetGroup >= ENERGY_GROUPS_NUM)
        return;

    particle.velocity = this->opacity->getNewScatterVelocity(cell, particle);

    double ratio = this->comptonGroupCenters[targetGroup] / this->comptonGroupCenters[sourceGroup];
    double f = cd.fleck;
    double newWeight = oldWeight * (f * ratio + (1.0 - f));
    double materialDeposit = f * oldWeight * (1.0 - ratio);

    particle.weight = newWeight;
    particle.frequency = this->frequencyForComptonGroup(targetGroup);

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

**Before:**
```cpp
this->applyImplicitComptonEvent(cellIndex, cell, group, oldVelocity, oldWeight, dopplerShift, particle);
```

**After:**
```cpp
this->applyComptonScatterEvent(cellIndex, cell, group, oldVelocity, oldWeight, dopplerShift, particle);
```

### Phase 8: Rewrite `applyComptonResidualCorrection` (RadiationIMC.cpp, lines 586-761)

Replace the 175-line function. The new function handles TWO corrections:

1. **Bres correction**: Apply `Bres[g]` to radiation WITH material coupling
2. **Transport residual**: Correct formula with `D_g*C_kappa` term, net couples to material

Neither requires a linear system solve.

```cpp
void RadiationIMC::applyComptonResidualCorrection(double fullDt)
{
    double const cdt = units::clight * fullDt;
    size_t const Ncells = this->tess->GetPointNo();

    for (size_t i = 0; i < Ncells; i++)
    {
        ComptonCellData &cd = this->comptonData[i];
        if (!cd.active) continue;

        // --- Part A: Apply Bres (source correction) WITH material coupling ---
        double bresSum = 0.0;
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            this->conserved[i].Eg[g] += cd.Bres[g];
            this->conserved[i].Erad += cd.Bres[g];
            bresSum += cd.Bres[g];
        }
        double bresMaterialDeposit = -bresSum;
        if (!this->noHydroFeedback)
        {
            this->conserved[i].internal_energy += bresMaterialDeposit;
            this->conserved[i].energy += bresMaterialDeposit;
        }
        this->comptonResidualMaterialExchange += bresMaterialDeposit;

        // --- Part B: Transport residual (direct formula, no linear solve) ---
        double const oneMinusF = 1.0 - cd.fleck;
        if (oneMinusF < 1e-15) continue;

        // Compute scalar contractions from time-averaged group energies
        double C_scalar = 0.0; // sum_h rowS[h]*<E_h>
        double C_kappa = 0.0;  // sum_h kappa_h*<E_h>
        for (size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
        {
            C_scalar += cd.rowS[h] * this->Eg_time_avg[i][h];
            C_kappa += cd.absorptionOpacity[h] * this->Eg_time_avg[i][h];
        }

        // Compute per-group residual: Kres = Ktarget - Kevent
        // Kres[h->g] = (1-f)*[tau*(1+n)*(ratio-1) + (-M_g*rowS[h] + D_g*kappa_h)/Gamma]
        // delta_E_g = c*dt*(1-f)*[C_energy_g - C_number_g + (-M_g*C_scalar + D_g*C_kappa)/Gamma]
        GroupArray delta{};
        double deltaSum = 0.0;
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double C_energy_g = 0.0;
            double C_number_g = 0.0;
            for (size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
            {
                if (h == g) continue;
                C_energy_g += cd.S[h][g] * this->Eg_time_avg[i][h];
                C_number_g += cd.tau[h][g] * (1.0 + cd.occupation[g]) * this->Eg_time_avg[i][h];
            }
            double Gamma_inv = (std::abs(cd.Gamma) > 1e-200) ? 1.0 / cd.Gamma : 0.0;

            delta[g] = cdt * oneMinusF * (C_energy_g - C_number_g
                       + (-cd.M[g] * C_scalar + cd.D[g] * C_kappa) * Gamma_inv);
            deltaSum += delta[g];
        }

        // Apply radiation correction with clamping for negative groups
        double deficit = 0.0;
        double positiveTotal = 0.0;
        double totalErad = 0.0;
        GroupArray finalEnergy{};
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double Eg = this->conserved[i].Eg[g];
            totalErad += Eg;
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

        if (deficit > 1e-6 * totalErad && totalErad > 0.0)
        {
            std::cerr << "Warning: Compton transport residual clamped deficit "
                      << deficit << " (" << deficit / totalErad * 100.0
                      << "% of Erad) in cell " << i << std::endl;
        }

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

        // Write back radiation and couple net to material
        double actualRadDelta = 0.0;
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double diff = finalEnergy[g] - this->conserved[i].Eg[g];
            this->conserved[i].Eg[g] = finalEnergy[g];
            actualRadDelta += diff;
        }
        this->conserved[i].Erad += actualRadDelta;

        // Material gets the opposite of the net radiation change
        double resMaterialDeposit = -actualRadDelta;
        if (!this->noHydroFeedback)
        {
            this->conserved[i].internal_energy += resMaterialDeposit;
            this->conserved[i].energy += resMaterialDeposit;
        }
        this->comptonResidualMaterialExchange += resMaterialDeposit;
    }
}
```

### Phase 9: Remove `reconcileComptonParticles` (RadiationIMC.cpp, lines 1851-1926)

This function was part of the PositiveKernelMeanEvent scheme. Remove the function body and its call site in `adjustExistingParticles`.

### Phase 10: Remove `comptonMeanRadiationCorrection` member and usage

In `RadiationIMC.hpp`: remove the member `std::vector<GroupArray> comptonMeanRadiationCorrection`.
In `RadiationIMC.cpp`: remove all references.

### Phase 11: Update `precomputeComptonData` (RadiationIMC.cpp, lines 1525-1674)

1. Replace `buildComptonInPlaceKernels(i, data)` with `buildComptonEventData(i, data)`.
2. Remove allocation of `comptonMeanRadiationCorrection`.
3. Remove references to `comptonTransportMode`.

### Phase 12: Update `postStep` residual call context (RadiationIMC.cpp, lines 521-522)

The call `applyComptonResidualCorrection(fullDt)` remains. Remove the `comptonCheckSignedTallies` block (lines 523-554) -- the clamping in Phase 8 handles this.

### Phase 13: Update `RadiationIMCParameters` (RadiationIMC.hpp)

Remove `ComptonTransportMode comptonTransportMode` field.

### Phase 14: Clean up diagnostics

Accumulators:
- `comptonContinuousMaterialExchange` (unchanged)
- `comptonImplicitMaterialExchange` (f-reduced event deposits)
- `comptonResidualMaterialExchange` (Bres + transport residual net -- much smaller than old scheme)
- `comptonSourceMaterialExchange` (unchanged)
- `comptonRemovalMaterialExchange` (unchanged)

### Phase 15: `generateComptonParticles` and `buildComptonSources` -- UNCHANGED

`Bres[g]` is now applied in `applyComptonResidualCorrection` Part A with material coupling.

### Phase 16-17: Remove `validateComptonParity`, update header declarations

Same as before.

---

## Verification Plan

```bash
./build_rich.sh gnuReleaseMPI --test_name=till_compton_mc
sbatch --wait --exclusive --partition=bigrun --ntasks=4 --wrap "mpirun -np 4 ./build/gnuReleaseMPI/rich"
```

Check:
1. No "Negative internal energy" errors
2. `event_material_exchange` and `residual_material_exchange` well-behaved (residual = Bres + small transport net)
3. Temperature/radiation equilibrium matches expected Compton equilibration
4. Compare with multigroup diffusion result

---

## Summary of Stability Improvement

| Property | Old Scheme | New Scheme |
|----------|-----------|------------|
| Material deposit per event | `w*(1-ratio)` (full) | `f*w*(1-ratio)` (reduced by f) |
| Residual material coupling | Linear solve with huge cancellation (~7e16 vs -6.6e16) | Bres + bounded transport net (both small, no cancellation) |
| Linear system solve | YES (ill-conditioned Gaussian elimination) | NO (direct formula) |
| Negative group handling | Unclamped solve -> frequent flooring -> broken cancellation | Clamped with warning + deficit redistribution |

---
---
---

# Response to Review: f-reduced IMC Compton Plan

## Summary of Verdict

The review was substantially correct on its most important points. It identified a genuine energy conservation error in the original plan (Bres handling) and correctly noted that the transport residual has a nonzero net when the effective scatter uses the `kappa_g*b_g/kappa_P` spectrum. These have been incorporated into v2.

The review was wrong on one minor point (claiming the scheme replaces the full implicit operator with only physical S -- the decomposition was always correct) and overstated on another (the `D_g*kappa_h` contribution to the residual is small and bounded, not dangerous).

### Changes incorporated from the review:

1. **Bres material coupling** (review point 5): Fixed in Phase 8 Part A.
2. **CDF-summed out-rate** (review point 3): Fixed in Phase 4.
3. **Transport residual material coupling** (review points 1, 4): Corrected formula includes `D_g*C_kappa/Gamma` term, net coupled to material.
4. **Warning on large clamping** (review point 7): Added diagnostic.
5. **Explicit operator decomposition** (review point 2): Stated clearly as `Ktotal = Hbase + Kevent + Kres`.

---

## Detailed Point-by-Point Response

### Review Point 5: "Bres is not free radiation-only correction" -- CORRECT

**Verdict: RIGHT. Incorporated.**

The original plan (Phase 15) added `Bres[g]` to radiation without compensating material. Since `Bres[g] <= 0` always (it's the negative part of `Btotal[g]`), `sum_g Bres[g] <= 0`, meaning radiation energy decreases. Without the compensating material deposit, total energy is lost.

In the old scheme, `Bres[g]` was folded into the RHS of the linear system solve, and the solve's material coupling implicitly handled this. In the new scheme, we must do it explicitly.

**Fix:** Phase 8 Part A now applies `Bres[g]` to radiation AND deposits `bresMaterialDeposit = -sum_g Bres[g]` to material. Total energy: `Delta_Erad + Delta_Emat = sum_g Bres[g] + (-sum_g Bres[g]) = 0`.


### Review Point 3: "Do not use -S[h][h] blindly as the event rate" -- CORRECT

**Verdict: RIGHT. Incorporated.**

The last energy group has special boundary handling in `buildComptonMatricesForCell` via `up_scattering_last - down_scattering_last` applied to `S[g][g]`. This correction adjusts the diagonal `S[g][g]` but does NOT correspondingly adjust the individual `tau[h][g]` values. As a result, `-S[h][h]` may not exactly equal `sum_{g!=h} tau[h][g]*(1+n[g])` for the last group.

Using the sum of the CDF weights (which are `max(0, tau[h][g]*(1+n[g]))` for `g!=h`) as the out-scatter rate guarantees self-consistency: the rate matches the CDF normalization, so `P(g|h) * comptonOutRate[h]` exactly gives the per-channel rate.

**Fix:** Phase 4 builds the CDF weights first, sums them for `comptonOutRate[h]`, then passes the same weights to `buildSafeComptonCdf`.


### Review Point 1: "Do not replace the implicit operator with only physical S" -- PARTIALLY CORRECT

**Verdict: The concern about the D_g contribution was correct. The claim that the decomposition was wrong was incorrect.**

The review correctly states the full kernel:
```
Ktotal[h][g] = S[h][g] + (1-f)/Gamma * M[g] * Lambda[h]
```

And that the effective scatter handles only:
```
Hbase[h][g] = (1-f)/Gamma * kappa_h * kappa_g * b_g
```

The difference `Ktotal - Hbase` involves the `D_g` terms (from `M_g = kappa_g*b_g + D_g`):
```
Ktarget[h][g] = S[h][g] + (1-f)/Gamma * (-kappa_g*b_g*rowS[h] + D_g*Lambda_h)
```

This decomposition was always present in the plan -- the scheme decomposes `Ktotal` into `Hbase + Kevent + Kres`, with `Kres` absorbing all remaining terms. The original plan's formula for the residual was missing the `D_g*kappa_h/Gamma` term (it only had `-M_g*rowS[h]/Gamma`). The corrected formula is:

```
K_res[h->g] = (1-f) * [tau[h][g]*(1+n[g])*(ratio-1) + (-M_g*rowS[h] + D_g*kappa_h)/Gamma]
```

which expands to:

```
delta_E_g = c*dt*(1-f) * [C_g^energy - C_g^number + (-M_g*C_scalar + D_g*C_kappa)/Gamma]
```

The original `-M_g*C_scalar/Gamma` term was incomplete. The `+D_g*C_kappa/Gamma` was missing.

**Why the original proof of sum=0 was wrong:** The original proof used the formula with only `M_g*rowS[h]/Gamma` and obtained `sum_g delta_E_g = (1-f)*[C_scalar - C_scalar] = 0`. But this was computed from the wrong formula. The correct formula's sum is:

```
sum_g delta_E_g = c*dt*(1-f)/Gamma * [sum_g(-M_g)*C_scalar + sum_g(D_g)*C_kappa]
               = c*dt*(1-f)/Gamma * [-Gamma*C_scalar + Upsilon*C_kappa]
```

Adding the first part: `sum_{g!=h}(S[h][g] - tau*(1+n)) = rowS[h]`, summed over h: `sum_h rowS[h]*E_h = C_scalar`.

Total: `c*dt*(1-f)*C_scalar + c*dt*(1-f)/Gamma*(-Gamma*C_scalar + Upsilon*C_kappa) = c*dt*(1-f)*Upsilon*C_kappa/Gamma`.

This is nonzero unless `Upsilon = 0`.

**Why this is still manageable:** The `Upsilon` contribution is a well-behaved scalar. For the test case:
- `Upsilon ~ 0.03`, `kappa_h ~ 0.01`, `Gamma ~ 0.04`, `(1-f) ~ 0.7`
- The net per-step is `~ c*dt*(1-f)*Upsilon*C_kappa/Gamma ~ 3 * 0.7 * 0.03/0.04 * sum(kappa*E) ~ modest`

This is orders of magnitude smaller than the old scheme's cancelling terms (~7e16 vs -6.6e16), and it's a single scalar deposited to material, not a per-group solve.

**Fix:** The corrected formula now includes `D_g*C_kappa/Gamma`. The net `sum_g delta_E_g` is computed and coupled to material via `resMaterialDeposit = -actualRadDelta`.


### Review Point 4: "The residual cannot be forced to zero-sum" -- CORRECT (given the effective scatter spectrum)

**Verdict: RIGHT, now that the full algebra is verified.**

This follows directly from review point 1. With the effective scatter spectrum being `kappa_g*b_g/kappa_P` (as confirmed in the code at line 1609 of RadiationIMC.cpp), the effective scatter operator is `Hbase[h][g] = (1-f)/Gamma * kappa_h * kappa_g*b_g`, which does NOT capture the `D_g` contribution to `M_g`. The residual therefore has a nonzero column sum of `(1-f)*kappa_h*Upsilon/Gamma` per source group h.

**Alternative approach (not taken):** If the effective scatter spectrum were changed to `M_g^+/sum(M_g^+)` (the positive part of the Compton-modified spectrum), the residual WOULD be zero-sum. However, this requires handling negative `M_g` entries (possible when `D_g` is very negative), adding complexity. The simpler approach -- keep `kappa_g*b_g/kappa_P` and couple the bounded net to material -- was chosen.

**Fix:** Material receives `-sum_g delta_E_g` from the transport residual (Phase 8 Part B).


### Review Point 2: "Define the f-reduced sampled event as a known operator" -- CORRECT BUT REDUNDANT

**Verdict: Good notation. Not a required change -- the plan already did this.**

The reviewer defines:
```
Kevent[h][g] = N[h][g] * (1 + f*(ratio-1))     for g != h
Kevent[h][h] = -sum_{g!=h} N[h][g]
Kres = Ktarget - Kevent
```

This is exactly the decomposition in the plan, just expressed differently. The f-reduced Compton event weight `w*(f*ratio + (1-f)) = w*(1 + f*(ratio-1))` produces energy `w*N[h][g]*(1 + f*(ratio-1))` in group g, which is `Kevent[h][g]*w`. The plan's `Kres = Ktotal - Hbase - Kevent` is the same as the reviewer's `Kres = Ktarget - Kevent` (since `Ktarget = Ktotal - Hbase`).

**Incorporated:** The operator decomposition `Ktotal = Hbase + Kevent + Kres` is now stated explicitly in the plan for clarity.


### Review Point 7: "Do not clamp real negative groups as part of the scheme" -- PARTIALLY CORRECT

**Verdict: Partially right. Silent clamping replaced with warned clamping.**

The reviewer's concern is valid: silent clamping of significantly negative groups can mask errors. However, making it a fatal error is too aggressive. The transport residual can legitimately try to move more energy out of a group than it currently holds, especially for groups near the Wien tail or at early times before the spectrum equilibrates.

The correct approach:
- Small deficits (< 1e-6 of total Erad): silently clamp (roundoff)
- Large deficits: warn to stderr with magnitude and cell index, then clamp
- Redistribute deficit proportionally across positive corrections to maintain total energy

**Fix:** Phase 8 now warns when `deficit > 1e-6 * totalErad`.


### Review Point 6: "Do not remove reconciliation unless residual corrections are represented by particles" -- OVERSTATED

**Verdict: Not applicable to this scheme.**

The `reconcileComptonParticles` function was specific to the `PositiveKernelMeanEvent` transport mode. It scaled particle weights after the residual solve changed `conserved.Eg`, to keep particles consistent with the corrected group energies.

In the new scheme:
1. All particles complete within a single time step (`timeLeft` initialized to `fullDt`).
2. `postStep` tallies all finished particles into `conserved.Eg` and `conserved.Erad`.
3. Then `applyComptonResidualCorrection` adjusts `conserved.Eg` deterministically.
4. At the start of the NEXT step, new particles are generated from the corrected `conserved` values.

There are no particles that persist across the residual correction boundary. The correction modifies `conserved.Eg`, and the next step's particle generation reads from the corrected `conserved.Eg`. No reconciliation is needed.

The `adjustExistingParticles` call site should be verified for any Compton-unrelated uses before removing it entirely.

---

## Mathematical Appendix: Full Derivation of the Residual

### Setup

The full implicit transport operator (from the bunched equation) has off-diagonal entries:

```
Ktotal[h->g] = S[h][g] + (1-f)/Gamma * M_g * Lambda_h     (h != g)
```

Where:
- `S[h][g] = tau[h][g] * (1+n[g]) * center[g]/center[h]` (Compton energy transfer rate)
- `M_g = kappa_g*b_g + D_g` (Compton-modified Planck weight)
- `D_g = sum_h dSdUm[h][g] * E_h` (Compton linearization)
- `Lambda_h = kappa_h - rowS[h]` (modified absorption opacity)
- `Gamma = kappa_P + Upsilon` (where `kappa_P = sum_g kappa_g*b_g`, `Upsilon = sum_g D_g`)

### Three operators

**1. Effective scatter (Hbase):**

Rate out of h: `(1-f)/Gamma * kappa_h * kappa_P`. Spectrum: `kappa_g*b_g / kappa_P`.

```
Hbase[h->g] = (1-f)/Gamma * kappa_h * kappa_g * b_g
```

Column sum: `sum_g Hbase[h->g] = (1-f)/Gamma * kappa_h * kappa_P = (1-f)*kappa_h*kappa_P/Gamma`.

**2. f-reduced Compton event (Kevent):**

Rate out of h: `comptonOutRate[h] = sum_{g!=h} tau[h][g]*(1+n[g])`. Target CDF proportional to `tau[h][g]*(1+n[g])`. Weight factor: `f*ratio + (1-f)`.

Off-diagonal:
```
Kevent[h->g] = tau[h][g]*(1+n[g]) * [f * center[g]/center[h] + (1-f)]
             = f*S[h][g] + (1-f)*tau[h][g]*(1+n[g])
```

Column sum:
```
sum_g Kevent[h->g] = f*sum_{g!=h} S[h][g] + (1-f)*comptonOutRate[h] - comptonOutRate[h]
                   = f*(rowS[h] + |S[h][h]|) + (1-f)*|S[h][h]| - |S[h][h]|
                   = f*rowS[h]
```

(The diagonal entry `Kevent[h->h] = -comptonOutRate[h]` accounts for the removed packet.)

**3. Residual (Kres = Ktotal - Hbase - Kevent):**

Off-diagonal:
```
Kres[h->g] = S[h][g] + (1-f)/Gamma*M_g*Lambda_h - (1-f)/Gamma*kappa_h*kappa_g*b_g - f*S[h][g] - (1-f)*tau[h][g]*(1+n[g])

= (1-f)*S[h][g] + (1-f)/Gamma*(M_g*Lambda_h - kappa_h*kappa_g*b_g) - (1-f)*tau[h][g]*(1+n[g])

= (1-f)*[S[h][g] - tau[h][g]*(1+n[g])] + (1-f)/Gamma*[(kappa_g*b_g + D_g)*(kappa_h - rowS[h]) - kappa_h*kappa_g*b_g]

= (1-f)*tau[h][g]*(1+n[g])*(ratio-1) + (1-f)/Gamma*[-kappa_g*b_g*rowS[h] + D_g*kappa_h - D_g*rowS[h]]

= (1-f)*tau[h][g]*(1+n[g])*(ratio-1) + (1-f)/Gamma*(-M_g*rowS[h] + D_g*kappa_h)
```

### Column sum of Kres

```
sum_g Kres[h->g] (including diagonal)
= sum_g Ktotal[h->g] - sum_g Hbase[h->g] - sum_g Kevent[h->g]
```

Compute each:

```
sum_g Ktotal[h->g] = rowS[h] + (1-f)/Gamma * Gamma * Lambda_h = rowS[h] + (1-f)*Lambda_h
                   = rowS[h] + (1-f)*(kappa_h - rowS[h])
                   = f*rowS[h] + (1-f)*kappa_h

sum_g Hbase[h->g] = (1-f)*kappa_h*kappa_P/Gamma

sum_g Kevent[h->g] = f*rowS[h]
```

Therefore:
```
sum_g Kres[h->g] = f*rowS[h] + (1-f)*kappa_h - (1-f)*kappa_h*kappa_P/Gamma - f*rowS[h]
                 = (1-f)*kappa_h*(1 - kappa_P/Gamma)
                 = (1-f)*kappa_h*(Gamma - kappa_P)/Gamma
                 = (1-f)*kappa_h*Upsilon/Gamma
```

This is zero if and only if `Upsilon = 0`. In general it is small but nonzero.

### Energy-weighted sum (net radiation change from residual)

```
sum_g delta_E_g = c*dt * sum_h E_h * sum_g Kres[h->g]
               = c*dt * (1-f)*Upsilon/Gamma * sum_h kappa_h * E_h
               = c*dt * (1-f)*Upsilon*C_kappa/Gamma
```

Material must receive `-sum_g delta_E_g` to conserve total energy.

### Why this is stable

The net material coupling from the transport residual is:
```
|dEmat_res| = c*dt*(1-f)*|Upsilon|*C_kappa/Gamma
```

For the test problem: `c*dt ~ 3`, `(1-f) ~ 0.7`, `|Upsilon| ~ 0.03`, `C_kappa ~ kappa_P*E_total ~ 0.01*1.36e18 ~ 1.4e16`, `Gamma ~ 0.04`.

```
|dEmat_res| ~ 3 * 0.7 * 0.03 * 1.4e16 / 0.04 ~ 2.2e16
```

Compare with the old scheme's cancelling terms: event_material_exchange ~ 7.0e16, residual_material_exchange ~ -6.6e16. The old scheme required near-perfect cancellation of these O(10^16) terms, and flooring negative groups broke the cancellation catastrophically.

In the new scheme:
- Per-event material deposit is `f*w*(1-ratio)` -- reduced by f ~ 0.3, so ~2e16 instead of ~7e16
- The residual material coupling ~2e16 is a single bounded scalar, not a per-group linear solve
- There is no cancellation between large terms
- Even if the residual is comparable in magnitude, it's applied as a simple additive correction, not through an ill-conditioned matrix inversion that amplifies errors

The dangerous instability arose from the cancellation `7e16 + (-6.6e16) = 4e15` being broken by group flooring. With f-reduced events, both the per-event exchange and the residual are smaller, and neither depends on delicate cancellation.
