# Stable f-Reduced Compton IMC Scheme (v4)

## Review Assessment (v3 -> v4)

The reviewer raised four points, all correct:

1. **Implicit solve needed**: The explicit Picard correction `delta = V*cdt*Kres*<E>` does not preserve the fully implicit equation. Since `cdt*Kres` entries can be O(0.1-1), the difference matters. Solution: keep `SolveComptonGroupSystem` and solve `(I - cdt*Kres) E_final = E_raw + Bres`.
2. **Keep reconciliation**: Particles persist across steps. After the residual solve changes `conserved.Eg`, surviving particles are inconsistent. `reconcileComptonParticles` scales them and creates deficit particles. Must keep.
3. **Use `cd.betaCdtF`**: The stored value handles tiny Gamma via fallback (`beta*cdtEff*fleck`). My `Gamma_inv = 0` fallback would zero out correction terms entirely. Use `cd.betaCdtF` in the Kres formula.
4. **Material positivity check**: Already exists at line 744 of the current `applyComptonResidualCorrection` (throws `UniversalError`). Just keep it.

## Key Insight

The old instability came from the **positive/negative kernel splitting** (catastrophic cancellation of O(10^16) terms), not from the linear solve itself. With the new Kres = Ktarget - Kevent (an organic correction, not an artificial splitting), the matrix `(I - cdt*Kres)` is well-conditioned. We keep the solve infrastructure and change only what Kres is.

## Mathematical Foundation

**Compton event mechanics (new):**
- Rate: `comptonOutRate[h] * c`, where `comptonOutRate[h] = sum_{g!=h} max(0, tau[h][g]*(1+n[g]))`
- Target group sampling: `P(g|h) = max(0, tau[h][g]*(1+n[g])) / comptonOutRate[h]`
- New weight: `w_new = w * (f * ratio + (1-f))` where `ratio = center[g]/center[h]`
- Material deposit: `f * w * (1 - ratio)`

**Effective scatter (unchanged):**
- Opacity: `baseEffectiveOpacity[h] = (1-f)/Gamma * kappa_h * kappa_P` (computed via `betaCdtF`)
- Spectrum: `kappa_g*b_g / kappa_P` (via `baseSourceCdf`)
- Weight: unchanged; Material deposit: 0

**Operator decomposition:** `Ktotal = Hbase + Kevent + Kres`
- `Hbase[h][g] = betaCdtF * kappa_h * kappa_g * b_g`
- `Kevent[h][g]`: for g!=h: `N_hg * (f*ratio + (1-f))`; for g==h: `-comptonOutRate[h]`
- `Kres[h][g] = Ktarget[h][g] - Kevent[h][g]` where `Ktarget = S + betaCdtF*(M*Lambda - kappa*kgbg)`

**Residual solve (implicit):**
```
(I - cdt * Kres) E_final = E_raw + Bres
material_deposit = -(sum_g (E_final[g] - E_raw[g]))
```

This uses `SolveComptonGroupSystem` (Gaussian elimination with partial pivoting). Well-conditioned because Kres is a bounded correction, not the negative part of a large cancelling expression.

---

## Files to Modify

1. `source/3D/radiation/RadiationIMC.hpp` -- Simplify `ComptonCellData`, remove unused fields/enum, add new fields, rename methods
2. `source/3D/radiation/RadiationIMC.cpp` -- Rewrite kernel building and event handling; simplify residual correction (keep solve)

---

## Detailed Changes

### Phase 1: Simplify `ComptonCellData` struct (RadiationIMC.hpp, lines 55-102)

**Remove these fields** (artifacts of positive/negative splitting):
- `GroupArray implicitEventRate` -- replaced by `comptonOutRate`
- `GroupArray implicitDiagonalCorrection`
- `GroupArray positiveMeanRatio`
- `GroupArray fullMeanRatio`
- `GroupCdfMatrix implicitEventCdf` -- replaced by `comptonTargetCdf`
- `GroupCdfMatrix positiveKernelCdf`
- `GroupMatrix Kmat` -- intermediate, not stored
- `GroupMatrix Hbase` -- intermediate, not stored
- `GroupMatrix implicitKernel` -- replaced by direct Kres computation
- `GroupMatrix positiveKernel` -- eliminated (no more positive/negative splitting)
- `GroupMatrix implicitEventRateMatrix`

**Keep:**
- `GroupMatrix residualKernel` -- now stores `Ktarget - Kevent` (recomputed differently)
- All other existing fields unchanged

**Add:**
- `GroupArray comptonOutRate` -- per source group h: total Compton out-scatter rate
- `GroupCdfMatrix comptonTargetCdf` -- per source group h: CDF for target group sampling

### Phase 2: Remove `ComptonTransportMode` (RadiationIMC.hpp)

Remove the enum at lines 15-19 and the field at line 39. Remove the member `comptonTransportMode` at line 174. Remove `comptonMeanRadiationCorrection` at line 175.

### Phase 3: Rename method declarations (RadiationIMC.hpp, lines 136-138)

- Line 136: `buildComptonInPlaceKernels` -> `buildComptonEventData`
- Line 138: `applyImplicitComptonEvent` -> `applyComptonScatterEvent`
- Keep `reconcileComptonParticles` (line 140)

### Phase 4: Rewrite `buildComptonInPlaceKernels` -> `buildComptonEventData` (RadiationIMC.cpp, lines 1188-1247)

New logic computes three things per source group h:
1. `comptonOutRate[h]` and `comptonTargetCdf[h]` from physical kernel weights
2. `baseEffectiveOpacity[h]` (unchanged formula)
3. `residualKernel[h][g]` = `Ktarget[h][g] - Kevent[h][g]` using `cd.betaCdtF`

```cpp
void RadiationIMC::buildComptonEventData(size_t cellIndex, ComptonCellData &cd)
{
    (void)cellIndex;
    for (size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
    {
        GroupArray weights{};
        double outRateSum = 0.0;
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if (g == h) continue;
            weights[g] = std::max(0.0, cd.tau[h][g] * (1.0 + cd.occupation[g]));
            outRateSum += weights[g];
        }
        cd.comptonOutRate[h] = outRateSum;
        cd.comptonTargetCdf[h] = RadiationIMC::buildSafeComptonCdf(weights);

        cd.baseEffectiveOpacity[h] = 0.0;
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double kgbg = cd.absorptionOpacity[g] * cd.planckFraction[g];
            cd.baseEffectiveOpacity[h] += cd.betaCdtF * cd.absorptionOpacity[h] * kgbg;
        }

        // Compute residualKernel[h][g] = Ktarget[h][g] - Kevent[h][g]
        double f = cd.fleck;
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double kgbg = cd.absorptionOpacity[g] * cd.planckFraction[g];
            double Ktarget_hg = cd.S[h][g]
                + cd.betaCdtF * (cd.M[g] * cd.Lambda[h]
                - cd.absorptionOpacity[h] * kgbg);
            double Kevent_hg;
            if (h == g)
            {
                Kevent_hg = -cd.comptonOutRate[h];
            }
            else
            {
                double N_hg = std::max(0.0,
                    cd.tau[h][g] * (1.0 + cd.occupation[g]));
                double ratio = this->comptonGroupCenters[g]
                             / this->comptonGroupCenters[h];
                Kevent_hg = N_hg * (f * ratio + (1.0 - f));
            }
            cd.residualKernel[h][g] = Ktarget_hg - Kevent_hg;
        }
    }
}
```

### Phase 5: Rewrite `applyImplicitComptonEvent` -> `applyComptonScatterEvent` (RadiationIMC.cpp, lines 1325-1407)

Replace with f-reduced Compton scatter:

```cpp
void RadiationIMC::applyComptonScatterEvent(
    size_t cellIndex, const ComputationalCell3D &cell, size_t sourceGroup,
    const Vector3D &oldVelocity, double oldWeight, double dopplerShift,
    Particle &particle)
{
    if (sourceGroup >= ENERGY_GROUPS_NUM) return;
    ComptonCellData &cd = this->comptonData[cellIndex];
    if (cd.comptonOutRate[sourceGroup] <= 0.0) return;

    size_t targetGroup = this->sampleComptonCdf(
        cd.comptonTargetCdf[sourceGroup], this->dist(this->re));
    if (targetGroup == sourceGroup || targetGroup >= ENERGY_GROUPS_NUM)
        return;

    particle.velocity = this->opacity->getNewScatterVelocity(cell, particle);
    double ratio = this->comptonGroupCenters[targetGroup]
                 / this->comptonGroupCenters[sourceGroup];
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
                (oldWeight * oldVelocity - newWeight * particle.velocity)
                * units::inv_clight2;
        }
    }
    this->comptonImplicitMaterialExchange += materialDeposit;
    ++this->comptonImplicitEventCount;
}
```

### Phase 6: Modify `step()` event opacity (RadiationIMC.cpp, ~line 214)

Change `implicitEventRate[group]` to `comptonOutRate[group]`.

### Phase 7: Modify `step()` event call (RadiationIMC.cpp, ~line 330)

Change `applyImplicitComptonEvent(...)` to `applyComptonScatterEvent(...)`.

### Phase 8: Simplify `applyComptonResidualCorrection` (RadiationIMC.cpp, lines 586-761)

**Keep the overall structure**, including `SolveComptonGroupSystem`. The changes are:

1. **Remove `PositiveKernelMeanEvent` branches** (lines 610-615, 623-628): `alreadyMaterialCoupledCorrection` is no longer needed.
2. **`residualKernel` now has new values** (Ktarget - Kevent, computed in Phase 4). The matrix construction code at lines 618-632 stays structurally identical.
3. **Change negative-energy clamping from warning to fatal** for large negatives (lines 648-662): throw `UniversalError` if `|finalGroupEnergy| > 1e-10 * totalErad`. Small roundoff negatives still get floored.
4. **Keep material limiting** (lines 670-728): the existing material withdrawal limiter is good safety.
5. **Keep material positivity check** (lines 744-753): already throws `UniversalError`.

The simplified function removes ~30 lines of mode-specific branching while preserving the implicit solve, material coupling, material limiting, and material positivity check.

### Phase 9: Remove `comptonMeanRadiationCorrection` usage

In `RadiationIMC.cpp`: remove all references to `comptonMeanRadiationCorrection`. This was allocated in `precomputeComptonData` and used only in `PositiveKernelMeanEvent` branches (which are being removed in Phase 8).

### Phase 10: Update `precomputeComptonData` (RadiationIMC.cpp, lines 1525-1674)

1. Replace `buildComptonInPlaceKernels(i, data)` with `buildComptonEventData(i, data)` (~line 1668)
2. Remove `comptonMeanRadiationCorrection` allocation
3. Remove `comptonTransportMode` references

### Phase 11: Update `postStep` (RadiationIMC.cpp)

Remove `comptonCheckSignedTallies` block (lines 523-554) -- the implicit solve and clamping in Phase 8 already ensure non-negative Eg.

### Phase 12: Update `validateComptonParity`

Rewrite to validate new identities:
- `sum_g comptonTargetCdf_weights[g] == comptonOutRate[h]`
- `residualKernel[h][g] == Ktarget[h][g] - Kevent[h][g]` (spot check)
- Remove references to `implicitKernel`, `positiveKernel`, etc.

### Phase 13: Clean up diagnostics

Update `printComptonDiagnostics` labels:
- `comptonImplicitMaterialExchange` now tracks f-reduced event deposits (smaller than old)
- `comptonResidualMaterialExchange` tracks the implicit solve's material coupling (from both Bres and Kres acting on radiation)

---

## Verification

```bash
./build_rich.sh gnuReleaseMPI --test_name=till_compton_mc
sbatch --wait --exclusive --partition=bigrun --ntasks=4 --wrap "mpirun -np 4 ./build/gnuReleaseMPI/rich"
```

Check:
1. No "Negative internal energy" errors
2. No "Failed to solve Compton deterministic residual group system" errors (solve should be well-conditioned now)
3. `event_material_exchange` is reduced by ~f compared to old scheme
4. `residual_material_exchange` is small and stable (no O(10^16) cancellations)
5. Temperature/radiation equilibrium matches expected Compton equilibration

---

## Summary: What Changed From v3

- v3 deleted `SolveComptonGroupSystem` and wrote an explicit Picard residual. v4 **keeps** the implicit solve with the new well-conditioned Kres.
- v3 deleted `reconcileComptonParticles`. v4 **keeps** it for particle consistency.
- v3 used `(1-f)/Gamma` with `Gamma_inv=0` fallback. v4 uses `cd.betaCdtF` throughout.
- v3 proposed a 200-line rewrite of `applyComptonResidualCorrection`. v4 simplifies the existing function by removing ~30 lines of mode-specific branching, keeping the proven solve/material-limit/positivity infrastructure.

## Stability Argument

- **Old scheme**: `Kres` = negative part of `implicitKernel` (huge, cancelling with positive part). Solve ill-conditioned.
- **New scheme**: `Kres` = `Ktarget - Kevent` (organic correction, no positive/negative splitting). Solve well-conditioned. Events deposit only `f*w*(1-ratio)` to material. No catastrophic cancellation.

---

## Response to v3 Review

### Point 1: Direct post-step residual is not the same implicit equation

**Verdict: Correct and important.**

The v3 plan proposed:
```cpp
delta_E[g] = V * cdt * sum_h Kres[h][g] * <E_h>
```
This is an explicit/Picard correction on the MC time-averaged field `<E_h>` (which is energy density after normalization by `fullDt * volume`). The reviewer correctly identifies that this does not preserve the fully implicit equation.

The correct implicit form is:
```
(I - cdt * Kres) E_final = E_raw + Bres
```

**Why the difference matters:** For the f-reduced scheme, `Kres` entries include terms like `N_hg*(1-f)*(ratio-1)` and `betaCdtF*(M_g*Lambda_h - kappa_h*kgbg)`. With `cdt ~ 3`, `(1-f) ~ 0.7`, and `N_hg*(ratio-1) ~ O(1)`, the product `cdt*Kres` can reach O(0.1-1). At these magnitudes, the implicit vs. explicit difference is NOT negligible -- the implicit solve couples the group corrections to each other self-consistently.

**Why the solve is now stable:** The old instability came from the positive/negative kernel splitting, where `positiveKernel` and `residualKernel` were individually much larger than `implicitKernel`, leading to catastrophic cancellation. The new `Kres = Ktarget - Kevent` is an organic correction (no splitting) and the matrix `(I - cdt*Kres)` is well-conditioned.

**Bonus:** The implicit solve uses extensive energies directly (`conserved.Eg`), avoiding the volume-conversion issue that complicated the v3 explicit formula (where `Eg_time_avg` is energy density).

### Point 2: Response is wrong about reconciliation

**Verdict: Correct.**

Particles persist across steps via `MonteCarloManagerSerial::step()` returning particles. After `applyComptonResidualCorrection` changes `conserved.Eg[g]`, surviving particles still represent the pre-correction radiation field. Without reconciliation, the next step's particle population would be inconsistent with the corrected group energies.

The existing `reconcileComptonParticles` (lines 1851-1926) handles this by:
1. Tallying particle energy per cell per group
2. Scaling down particles where particle energy exceeds `conserved.Eg[g]`
3. Creating deficit particles where `conserved.Eg[g]` exceeds particle representation

This function is retained in v4 without modification.

### Point 3: Use stored `cd.betaCdtF`, not recomputed `(1-f)/Gamma`

**Verdict: Correct.**

The existing code at lines 1664-1666 computes:
```cpp
data.betaCdtF = data.beta * cdtEff * data.fleck;
if(std::abs(data.Gamma) > 1e-200)
    data.betaCdtF = (1.0 - data.fleck) / data.Gamma;
```

When `|Gamma|` is tiny, `betaCdtF` uses the safe fallback `beta * cdtEff * fleck`. My v3 code's `Gamma_inv = 0` fallback would zero out the `M_g*Lambda_h - kappa_h*kgbg` correction terms entirely, breaking the operator identity. Using `cd.betaCdtF` directly is both more robust and consistent with how `Hbase` and `Kmat` are already computed (lines 1218-1219).

In the Kres formula, this gives:
```cpp
Ktarget_hg = cd.S[h][g] + cd.betaCdtF * (cd.M[g] * cd.Lambda[h] - cd.absorptionOpacity[h] * kgbg);
```

### Point 4: Add material positivity check after residual material deposit

**Verdict: Correct, and already implemented.**

The existing `applyComptonResidualCorrection` already has this check at lines 744-753:
```cpp
if(this->conserved[i].internal_energy < 0.0)
{
    UniversalError eo("Negative internal energy after Compton residual correction");
    eo.addEntry("Cell index", i);
    eo.addEntry("Material deposit", materialDeposit);
    eo.addEntry("Internal energy", this->conserved[i].internal_energy);
    // ... more diagnostics ...
    throw eo;
}
```

This is retained in v4 without modification. The reviewer's suggested diagnostics (f, Gamma, Upsilon) can be added to this error for enhanced debugging.
