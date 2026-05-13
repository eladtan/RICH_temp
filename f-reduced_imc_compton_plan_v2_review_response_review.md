# Review: f-reduced IMC Compton Plan v2 Review Response

## Verdict

The response fixes the algebra of `Kres`, but the proposed direct residual application still does not preserve the fully implicit equation.

Perfect version: f-reduced events plus exact `Kres = Ktarget - Kevent`, then an implicit residual solve, plus particle reconciliation.

## Good

- It correctly accepts the six v2 review points.
- Updated Phase 8 fixes volume, `grid`, diagonal mismatch, `Bres` material coupling, and fatal large negatives.
- Literal `Ktarget - Kevent` is the right algebraic target.

## Still Not Perfect

### 1. Direct post-step residual is not the same implicit equation

This:

```cpp
delta_E[g] = V * cdt * sum_h Kres[h][g] * <E_h>
```

is an explicit/Picard residual correction on the MC time-averaged field.

The diffusion equation has the residual coupled implicitly. To preserve the same equation set, use:

```cpp
(I - cdt * Kres) E_final = E_raw + Bres
```

with the now-smaller f-reduced `Kres`.

So do not delete `SolveComptonGroupSystem` yet. Use it for the reduced residual solve.

### 2. Response is wrong about reconciliation

Particles persist across steps: `MonteCarloManagerSerial::step()` returns `populationControlParticles`.

`postStep()` changes `conserved.Eg` after tally.

Those returned particles still represent the pre-residual radiation field.

Keep `reconcileComptonParticles()` or create correction particles immediately.

### 3. Use stored `cd.betaCdtF`, not recomputed `(1-f)/Gamma`

Current code handles tiny `Gamma` with fallback:

```cpp
cd.betaCdtF = beta * cdtEff * f
```

The response code uses:

```cpp
Gamma_inv = 0
```

when `Gamma` is tiny, which can break parity.

Use:

```cpp
Ktarget_hg = cd.S[h][g] + cd.betaCdtF * (cd.M[g] * cd.Lambda[h] - kappa_h * kgbg)
```

### 4. Add material positivity check after residual material deposit

After the residual material deposit, significant negative material should be fatal with diagnostics:

- cell index
- material delta
- internal energy
- `f`
- `Gamma`
- `Upsilon`

## Bottom Line

The response is much closer, but exactness requires:

```cpp
Kres = Ktarget - Kevent
```

followed by an implicit residual solve:

```cpp
(I - cdt * Kres) E_final = E_raw + Bres
```

and then particle reconciliation.
