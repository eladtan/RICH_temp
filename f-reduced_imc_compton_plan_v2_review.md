# Review: f-reduced IMC Compton Plan v2

## Main Verdict

Do not implement v2 unchanged. It is close, but still has equation-preserving bugs.

The architecture is right: f-reduced events plus deterministic residual. But to be exact, the residual must literally be `Ktarget - Kevent`, with the same stored event kernel used by transport.

## Must Fix

### 1. Residual is missing volume

`Eg_time_avg[i][g]` is energy density. `conserved[i].Eg[g]` is extensive energy. Phase 8 must use:

```cpp
delta[g] = cd.volume * cdt * ...
```

not just:

```cpp
delta[g] = cdt * ...
```

This is a real scaling bug.

### 2. Effective scatter description is wrong

The text says rate is:

```cpp
(1-f) * kappa_h * c
```

But the code/formula is:

```cpp
baseEffectiveOpacity[h] = (1-f)/Gamma * kappa_h * kappaP
```

Phase 4 code is right. The prose is wrong and dangerous.

### 3. Residual formula must include diagonal boundary mismatch

v2 says not to use `-S[h][h]` as out-rate. Good. But then the residual must preserve the difference:

```cpp
diagResidual[h] = S[h][h] + comptonOutRate[h]
```

Add to group `h`:

```cpp
delta[h] += cd.volume * cdt * diagResidual[h] * Eg_time_avg[i][h]
```

Without this, `Kevent + Kres != Ktarget` for the last-group boundary correction.

### 4. Residual must use stored event kernel, not raw tau

If event weights use:

```cpp
N[h][g] = max(0, tau[h][g] * (1+n[g]))
```

then residual must subtract that exact same `N[h][g]`.

Do not recompute with raw:

```cpp
tau[h][g] * (1+n[g])
```

unless you guarantee it is identical. Store `comptonNumberKernel[h][g]`.

### 5. Large clamping is still ad hoc

This is still not equation-preserving:

```text
large deficit -> warn -> clamp -> redistribute
```

For the exact scheme:

- tiny roundoff floor is okay;
- significant negative group should be fatal diagnostic, or explicitly called a positivity projection that changes the equation.

Given the objection to ad hoc fixes, make large clamping fatal.

### 6. `this->tess->GetPointNo()` is wrong for this class

Use:

```cpp
this->grid.GetPointNo()
```

## Correct Exact Split

Define and validate:

```cpp
Ktarget = Ktotal - Hbase
Kevent[h][g] = N[h][g] * (1 + f * (ratio - 1))  // offdiag
Kevent[h][h] = -sum_g N[h][g]
Kres = Ktarget - Kevent
```

Then apply deterministic residual from stored `Kres`:

```cpp
delta_E[g] = volume * c * dt * sum_h Kres[h][g] * <E_h>
```

plus:

```cpp
delta_E[g] += Bres[g]
dEmat = -sum_g delta_E[g]
```

## Bottom Line

v2 has the right architecture: f-reduced events plus deterministic residual. But to be perfect, make the residual literally `Ktarget - Kevent`, include volume, include diagonal boundary correction, use stored `N`, and remove large clamping.
