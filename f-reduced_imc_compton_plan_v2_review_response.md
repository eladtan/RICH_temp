# Response to Review of f-reduced IMC Compton Plan v2

## Summary

The reviewer is correct on all six points. Five are genuine bugs or dangerous imprecisions, and one is a philosophical choice I now agree with. All are incorporated into v3.

| Point | Verdict | Fix |
|-------|---------|-----|
| 1. Missing volume | **Correct, real bug** | Add `cd.volume *` to delta computation |
| 2. Effective scatter prose wrong | **Correct** | Fix prose: rate is `(1-f)/Gamma * kappa_h * kappa_P`, not `(1-f)*kappa_h` |
| 3. Diagonal boundary mismatch | **Correct** | Include `S[h][h] + comptonOutRate[h]` diagonal term in residual |
| 4. Must use stored/consistent N | **Correct** | Use `max(0, tau[h][g]*(1+n[g]))` in both event CDF and residual |
| 5. Large clamping should be fatal | **Correct for development** | Make it fatal (throw) not warn+clamp |
| 6. `tess` vs `grid` | **Correct** | All existing code uses `this->grid.GetPointNo()` |

---

## Detailed Responses

### Point 1: Residual is missing volume -- CORRECT

`Eg_time_avg[i][g]` is normalized in `postStep` (line 395-397):
```cpp
double norm = fullDt * this->grid.GetVolume(i);
this->Eg_time_avg[i][g] /= norm;
```

So `Eg_time_avg[i][g]` has units of energy/volume (energy density, time-averaged). The residual formula `delta[g] = cdt * ... * S * Eg_time_avg` produces energy/volume. But `conserved[i].Eg[g]` is extensive energy (energy in the cell). The volume multiplier is mandatory.

Note that `Bres[g]` already includes `cd.volume` (from `buildComptonSources`), so Part A of the correction is dimensionally correct as-is.

**Fix:** Multiply the transport residual by `cd.volume`:
```cpp
delta[g] = cd.volume * cdt * sum_h Kres[h][g] * Eg_time_avg[i][h]
```

### Point 2: Effective scatter description is wrong -- CORRECT

The prose said "Rate: `(1-f) * kappa_h * c`". The actual opacity is:
```cpp
baseEffectiveOpacity[h] = (1-f)/Gamma * kappa_h * kappa_P
```

These differ by a factor of `kappa_P / Gamma`. The Phase 4 code was always correct; only the prose description in the Mathematical Foundation section was wrong.

**Fix:** Correct the prose to: "Rate: `(1-f)/Gamma * kappa_h * kappa_P * c`".

### Point 3: Residual must include diagonal boundary mismatch -- CORRECT

Looking at `buildComptonMatricesForCell` (line 1112-1116), when both `g` and `gt` are the last group:
```cpp
cd.S[g][g] += (upScatteringLast - downScatteringLast) * (1.0 + cd.occupation[g]);
continue; // skips normal in/out-scattering for this (g,gt) pair
```

This boundary correction adds `(up-down)*(1+n[last])` to `S[last][last]` without a corresponding off-diagonal `tau` entry. So for the last group:
```
-S[h][h] = sum_{gt!=h} tau[h][gt]*(1+n[gt]) - (up-down)*(1+n[h])
comptonOutRate[h] = sum_{gt!=h} tau[h][gt]*(1+n[gt])

diagResidual = S[h][h] + comptonOutRate[h] = (up-down)*(1+n[h])
```

For non-last groups: `diagResidual = 0`.

Without this correction, the operator identity `Kevent + Kres + Hbase = Ktotal` is violated for the last group.

**Fix:** The cleanest approach is to compute the residual as literal `Ktarget - Kevent` for ALL (h,g) pairs, including diagonal. This automatically captures the boundary mismatch without any special-case logic.

### Point 4: Residual must use stored event kernel -- CORRECT

If the event CDF uses `N[h][g] = max(0, tau[h][g]*(1+n[g]))` and the residual uses raw `tau[h][g]*(1+n[g])` without the `max(0,...)`, there could be a mismatch when `tau[h][g]` is numerically negative. While `tau >= 0` should hold physically, numerical precision could violate this.

Storing a full `comptonNumberKernel[h][g]` matrix adds ENERGY_GROUPS_NUM^2 doubles per cell. Instead, use the same `max(0, ...)` clamp in both places:

```cpp
double N_hg = std::max(0.0, cd.tau[h][g] * (1.0 + cd.occupation[g]));
```

This is used identically in:
1. `buildComptonEventData` when building CDF weights
2. `applyComptonResidualCorrection` when computing Kevent in the residual

No extra storage needed -- the clamped value is computed on the fly in both places from the same inputs (`cd.tau`, `cd.occupation`), guaranteeing identical results.

### Point 5: Large clamping should be fatal -- AGREED

The reviewer's philosophical point is sound: if the clamping is significant, the scheme is performing a positivity projection that changes the equations being solved. This should be visible, not hidden.

During development, fatal errors for large negative groups will immediately surface any bugs in the residual formula. Once the scheme is verified, this can be relaxed to a warning if needed for production robustness.

**Fix:** Throw if any `proposed = Eg + delta[g] < 0` and `|delta[g]| > 1e-10 * totalErad`. Allow tiny roundoff floors only.

### Point 6: `this->tess->GetPointNo()` is wrong -- CORRECT

Confirmed by grep: every function in `RadiationIMC.cpp` uses `this->grid.GetPointNo()`:
- `postStep` (line 389)
- `applyComptonResidualCorrection` (line 588)
- `generateParticles` (line 769)
- `generateComptonParticles` (line 868)
- `precomputeComptonData` (line 1536)

My plan incorrectly used `this->tess->GetPointNo()` in the Phase 8 code.

**Fix:** Use `this->grid.GetPointNo()` everywhere.

---

## Updated Phase 8 Code

The corrected `applyComptonResidualCorrection` computes the residual as literal `Ktarget - Kevent` for all (h,g) pairs:

```cpp
void RadiationIMC::applyComptonResidualCorrection(double fullDt)
{
    double const cdt = units::clight * fullDt;
    size_t const Ncells = this->grid.GetPointNo();

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

        // --- Part B: Transport residual = Ktarget - Kevent, applied deterministically ---
        double const f = cd.fleck;
        double const oneMinusF = 1.0 - f;
        if (oneMinusF < 1e-15) continue;

        double const Gamma_inv = (std::abs(cd.Gamma) > 1e-200) ? 1.0 / cd.Gamma : 0.0;

        // Compute delta_E[g] = volume * cdt * sum_h Kres[h][g] * <E_h>
        GroupArray delta{};
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double sum = 0.0;
            for (size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
            {
                double Eh = this->Eg_time_avg[i][h];

                // Ktarget[h][g] = Ktotal[h][g] - Hbase[h][g]
                double kgbg = cd.absorptionOpacity[g] * cd.planckFraction[g];
                double Ktarget_hg = cd.S[h][g]
                    + oneMinusF * Gamma_inv * (cd.M[g] * cd.Lambda[h] - cd.absorptionOpacity[h] * kgbg);

                // Kevent[h][g]
                double Kevent_hg;
                if (h == g)
                {
                    Kevent_hg = -cd.comptonOutRate[h];
                }
                else
                {
                    double N_hg = std::max(0.0, cd.tau[h][g] * (1.0 + cd.occupation[g]));
                    double ratio = this->comptonGroupCenters[g] / this->comptonGroupCenters[h];
                    Kevent_hg = N_hg * (f * ratio + oneMinusF);
                }

                sum += (Ktarget_hg - Kevent_hg) * Eh;
            }
            delta[g] = cd.volume * cdt * sum;
        }

        // Apply to radiation; fatal if significant negative group
        double totalErad = 0.0;
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            totalErad += this->conserved[i].Eg[g];

        double actualRadDelta = 0.0;
        for (size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double proposed = this->conserved[i].Eg[g] + delta[g];
            if (proposed < 0.0 && std::abs(delta[g]) > 1e-10 * totalErad)
            {
                UniversalError eo("Compton transport residual produced significant negative group energy");
                eo.addEntry("Cell", static_cast<double>(i));
                eo.addEntry("Group", static_cast<double>(g));
                eo.addEntry("Eg before", this->conserved[i].Eg[g]);
                eo.addEntry("delta", delta[g]);
                eo.addEntry("totalErad", totalErad);
                throw eo;
            }
            double final_Eg = std::max(0.0, proposed); // roundoff floor only
            double diff = final_Eg - this->conserved[i].Eg[g];
            this->conserved[i].Eg[g] = final_Eg;
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

Key properties of this implementation:
1. Computes `Kres = Ktarget - Kevent` for every (h,g) pair, including diagonal
2. Automatically includes the last-group boundary correction via `S[h][h] + comptonOutRate[h]` on diagonal
3. Uses `max(0, tau*...)` consistently with the event CDF
4. Multiplies by `cd.volume` for correct extensive energy units
5. Uses `this->grid.GetPointNo()` for cell count
6. Fatal error for significant negative groups (> 1e-10 * totalErad threshold)
7. Tiny roundoff floors are allowed via `max(0, proposed)`

---

## Updated Mathematical Foundation (corrected prose)

**Effective scatter:**
- Opacity: `baseEffectiveOpacity[h] = (1-f)/Gamma * kappa_h * kappa_P` (not `(1-f)*kappa_h`)
- Spectrum: `kappa_g*b_g / kappa_P` (via `baseSourceCdf`, always positive)
- Weight: unchanged
- Material deposit: 0

---

## Summary of All Changes from v2 to v3

1. **Phase 8 delta computation**: `delta[g] = cd.volume * cdt * sum_h Kres[h][g] * <E_h>` (added `cd.volume`)
2. **Phase 8 residual formula**: Compute literal `Ktarget[h][g] - Kevent[h][g]` for ALL (h,g), including h==g diagonal
3. **Phase 8 N consistency**: Use `max(0, tau[h][g]*(1+n[g]))` in residual, matching event CDF
4. **Phase 8 negative handling**: Fatal error for significant negatives, roundoff floor for tiny ones
5. **Phase 8 cell count**: `this->grid.GetPointNo()` not `this->tess->GetPointNo()`
6. **Prose**: Effective scatter rate corrected to `(1-f)/Gamma * kappa_h * kappa_P * c`
