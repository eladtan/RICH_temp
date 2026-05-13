# Review: f-reduced IMC Compton Plan

The f-reduced event idea is good. The plan as written is not equation-preserving yet.

The main problem is this sentence: "residual correction never touches material energy." That cannot be true for the full implicit equations. Some residual pieces are pure redistribution, but `Bres` and the implicit `D/M/Gamma` terms can have nonzero net radiation energy change, so the material must receive the opposite change.

## What Works

The event update is physically consistent because the packet energy change and material deposit are opposite:

```cpp
q = 1 + f * (ratio - 1)
W_new = q * W_old
dEmat = W_old - W_new
```

Sampling a positive Compton-number kernel and reducing the energy jump by `f` should reduce the huge stochastic material kicks.

This can be made exact as a split operator.

## Required Changes

### 1. Do not replace the implicit operator with only physical `S`

The diffusion equations include:

```cpp
Ktotal[h][g] = S[h][g] + (1-f)/Gamma * M[g] * (kappa_h - rowS[h])
```

With the existing base effective absorption handled separately:

```cpp
Hbase[h][g] = (1-f)/Gamma * kappa_h * kappa_g * b_g
```

The Compton residual target should be:

```cpp
Ktarget[h][g] = Ktotal[h][g] - Hbase[h][g]
```

### 2. Define the f-reduced sampled event as a known operator

For `g != h`:

```cpp
Kevent[h][g] = N[h][g] * (1 + f * (ratio - 1))
Kevent[h][h] = -sum_g N[h][g]
Kres[h][g]   = Ktarget[h][g] - Kevent[h][g]
```

Here `N[h][g]` is the positive Compton number transition rate.

This guarantees:

```cpp
Kevent + Kres + Hbase == Ktotal
```

### 3. Do not use `-S[h][h]` blindly as the event rate

Build and store the positive number kernel `N[h][g]`, then set:

```cpp
comptonOutRate[h] = sum_{g != h} N[h][g]
```

This avoids last-group and boundary inconsistencies.

### 4. The residual cannot be forced to zero-sum

Apply:

```cpp
dE_res[g] = V * c * dt * sum_h Kres[h][g] * <E_h>
```

Then include the signed source residual:

```cpp
dE_rad_res[g] = dE_res[g] + Bres[g]
```

The material must get the opposite net energy:

```cpp
dEmat_res = -sum_g dE_rad_res[g]
```

### 5. `Bres` is not free radiation-only correction

If you add `Bres[g]` directly to radiation, you must also subtract its sum from material:

```cpp
conserved.Eg[g] += Bres[g]
conserved.internal_energy -= sum_g Bres[g]
```

Otherwise energy conservation is broken.

### 6. Do not remove reconciliation unless residual corrections are represented by particles immediately

If post-step changes `conserved.Eg`, the particle population must be reconciled to those corrected group energies before the next transport step. Otherwise particles and conserved radiation drift apart.

### 7. Do not clamp real negative groups as part of the scheme

Allow tiny roundoff floors only. If a group becomes significantly negative, that should be a fatal diagnostic showing the split is still wrong.

## Bottom Line

The perfect version is: keep the f-reduced event, but define it as one piece of the exact diffusion operator and set the deterministic residual to:

```cpp
full implicit operator - sampled event operator - base effective absorption operator
```

Then couple the residual net energy to material. That preserves the same equations while reducing the dangerous per-event material kicks.
