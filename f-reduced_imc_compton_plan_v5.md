# Stable f-Reduced Compton IMC Scheme (v5)

## Review Assessment (v3 -> v4)

The reviewer raised four points, all correct:

1. **Implicit solve needed**: The explicit Picard correction `delta = V*cdt*Kres*<E>` does not preserve the fully implicit equation. Since `cdt*Kres` entries can be O(0.1-1), the difference matters. Solution: keep `SolveComptonGroupSystem` and solve `(I - cdt*Kres) E_final = E_raw + Bres`.
2. **Keep reconciliation**: Particles persist across steps. After the residual solve changes `conserved.Eg`, surviving particles are inconsistent. `reconcileComptonParticles` scales them and creates deficit particles. Must keep.
3. **Use `cd.betaCdtF`**: The stored value handles tiny Gamma via fallback (`beta*cdtEff*fleck`). My `Gamma_inv = 0` fallback would zero out correction terms entirely. Use `cd.betaCdtF` in the Kres formula.
4. **Material positivity check**: Already exists at line 744 of the current `applyComptonResidualCorrection` (throws `UniversalError`). Just keep it.

## Review Assessment (v4 -> v5)

The reviewer raised five points, all correct:

1. **Remove material withdrawal limiter**: The limiter (lines 670-728) is an ad-hoc equation change -- exactly the kind of silent fix we objected to. With well-conditioned Kres, if material goes negative it should be fatal, not limited.
2. **CDF fallback must not silently drop collisions**: `targetGroup == sourceGroup` after CDF sampling means the CDF is broken (weights[sourceGroup] = 0 by construction). Throw instead of silent return.
3. **Operator parity check**: Verify `Kevent + Kres + Hbase == Ktotal` in `validateComptonParity`. Catches inconsistencies in operator construction.
4. **Solver diagnostics**: Track min pivot / max coefficient in `SolveComptonGroupSystem`. Include cell, f, Gamma, Upsilon on failure. "Well-conditioned" is plausible, not guaranteed.
5. **No redistribution for negative groups**: Large negative = fatal. Tiny roundoff = floor only. No clamping or redistribution.

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

Replace with f-reduced Compton scatter. CDF returning source group is fatal (not a silent return):

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
    {
        UniversalError eo("Compton CDF sampling returned invalid target group");
        eo.addEntry("Cell index", static_cast<double>(cellIndex));
        eo.addEntry("Source group", static_cast<double>(sourceGroup));
        eo.addEntry("Target group", static_cast<double>(targetGroup));
        eo.addEntry("comptonOutRate", cd.comptonOutRate[sourceGroup]);
        throw eo;
    }

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

1. **Remove `PositiveKernelMeanEvent` branches** (lines 610-615, 623-628): `alreadyMaterialCoupledCorrection` is no longer needed. Simplified RHS: `rhs[g] = rawGroupEnergy[g] + cd.Bres[g]`.
2. **`residualKernel` now has new values** (Ktarget - Kevent, computed in Phase 4). The matrix construction code at lines 618-632 stays structurally identical but without the mode2 override.
3. **Large negative group energy is fatal** (lines 648-662): throw `UniversalError` if `finalGroupEnergy < 0` and `|finalGroupEnergy| > 1e-10 * totalErad`. Include cell, group, raw energy, solved energy, f, Gamma, Upsilon. **Tiny roundoff negatives only**: floor to 0.0. No redistribution.
4. **Remove material withdrawal limiter** (lines 670-728): Delete the entire block. If the solve is correct, the material deposit should be physical. If it drives material negative, the existing fatal check at line 744 catches it. The limiter is an ad-hoc equation change that masks bugs.
5. **Keep material positivity check** (lines 744-753): already throws `UniversalError`. Add `f`, `Gamma`, `Upsilon` to the error diagnostics.

The simplified function removes ~80 lines (mode branches + material limiter) while making the error handling strict: any unphysical result is a diagnostic, not silently fixed.

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
- **Operator decomposition parity**: `Kevent[h][g] + Kres[h][g] + Hbase[h][g] == Ktotal[h][g]` for all (h,g). Compute `Ktotal[h][g] = S[h][g] + betaCdtF * M[g] * Lambda[h]`, `Hbase[h][g] = betaCdtF * kappa_h * kgbg`, and `Kevent[h][g]` on the fly. Compare sum against `Ktotal[h][g]`. Fail if max absolute diff exceeds `1e-12 * max(|Ktotal|)`.
- **CDF consistency**: `sum_g weights[g] == comptonOutRate[h]`
- Remove references to `implicitKernel`, `positiveKernel`, etc.

### Phase 12b: Add residual matrix diagnostics to `SolveComptonGroupSystem`

Extend the solver to track and return conditioning info:
- Track min absolute pivot value across all columns
- Track max absolute coefficient in the original matrix
- On failure (`pivotAbs <= 1e-200` or `!isfinite`), include in the `UniversalError` thrown by the caller: cell index, min pivot, max coefficient, `f`, `Gamma`, `Upsilon`
- Approach: add output parameters `double &minPivot, double &maxCoeff` to the function signature, or return a small diagnostics struct alongside the bool success.

### Phase 13: Clean up diagnostics

Update `printComptonDiagnostics` labels:
- `comptonImplicitMaterialExchange` now tracks f-reduced event deposits (smaller than old)
- `comptonResidualMaterialExchange` tracks the implicit solve's material coupling (from both Bres and Kres acting on radiation)
- Remove `comptonMaterialLimitedRadiationEnergy` and `comptonMaterialLimitedCellCount` tracking (material limiter removed)

---

## Verification

```bash
./build_rich.sh gnuReleaseMPI --test_name=till_compton_mc
sbatch --wait --exclusive --partition=bigrun --ntasks=4 --wrap "mpirun -np 4 ./build/gnuReleaseMPI/rich"
```

Check:
1. No "Negative internal energy" errors
2. No "Failed to solve Compton deterministic residual group system" errors (solve should be well-conditioned now)
3. No "Compton CDF sampling returned invalid target group" errors
4. Parity check passes: `Kevent + Kres + Hbase == Ktotal` within tolerance
5. `event_material_exchange` is reduced by ~f compared to old scheme
6. `residual_material_exchange` is small and stable (no O(10^16) cancellations)
7. Temperature/radiation equilibrium matches expected Compton equilibration

---

## Summary: What Changed From v3

- v3 deleted `SolveComptonGroupSystem` and wrote an explicit Picard residual. v4 **keeps** the implicit solve with the new well-conditioned Kres.
- v3 deleted `reconcileComptonParticles`. v4 **keeps** it for particle consistency.
- v3 used `(1-f)/Gamma` with `Gamma_inv=0` fallback. v4 uses `cd.betaCdtF` throughout.
- v3 proposed a 200-line rewrite of `applyComptonResidualCorrection`. v4 simplifies the existing function by removing mode-specific branching.

## What Changed From v4 (v5 review)

- **Removed material withdrawal limiter** (lines 670-728): ad-hoc equation change that masks bugs. Material negativity is now fatal.
- **Compton CDF fallback is now fatal**: `targetGroup == sourceGroup` after CDF sampling throws instead of silently returning.
- **Added operator parity check**: `Kevent + Kres + Hbase == Ktotal` verified in `validateComptonParity`.
- **Added solver diagnostics**: min pivot / max coefficient tracked in `SolveComptonGroupSystem`, included in failure errors.
- **No redistribution for negative groups**: large negative = fatal, tiny roundoff = floor only.

## Stability Argument

- **Old scheme**: `Kres` = negative part of `implicitKernel` (huge, cancelling with positive part). Solve ill-conditioned.
- **New scheme**: `Kres` = `Ktarget - Kevent` (organic correction, no positive/negative splitting). Solve well-conditioned. Events deposit only `f*w*(1-ratio)` to material. No catastrophic cancellation.

## Error Philosophy (v5)

No silent fixes. Every unphysical result is a fatal diagnostic:
- Negative group energy (large): fatal with cell/group/f/Gamma/Upsilon
- Negative material energy: fatal with same diagnostics
- CDF returning source group: fatal
- Solve failure: fatal with min pivot / max coefficient / conditioning info
- Tiny roundoff negatives only: floored to zero (not redistributed)
