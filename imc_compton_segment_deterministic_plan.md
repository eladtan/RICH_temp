# Segment-Local Deterministic IMC Compton Plan

## Problem

The f-reduced event implementation conserves total energy, but it still samples a stiff Compton redistribution operator with a small number of packets. The late-time oscillations and incomplete temperature equilibration are consistent with that split: a noisy event update is followed by a stiff deterministic residual that acts on the final packet census. A packet that crosses multiple cells also keeps the wrong weight until a sampled event occurs or until the end-of-step residual is applied.

## Core Idea

Apply the implicit Compton redistribution operator locally on every particle path segment. For a segment in cell `i` with duration `dt_seg`, source group `h`, and incoming packet weight `W_in`, solve the frozen local implicit group system

```text
(I - c dt_seg K_target^T) W_out = W_in e_h
```

where `K_target` is the same non-base Compton/material coupling kernel already used in the f-reduced decomposition:

```text
K_target[h][g] = S[h][g]
               + beta c dt f (M[g] Lambda[h] - kappa[h] kappa[g] b[g])
```

The matrix convention is unchanged:

```text
h = source group
g = target group
K[h][g] contributes from source h to target g
```

## Particle Update

After solving for `W_out[g]`:

1. Sum the positive outgoing energy.
2. Sample one target group from the positive `W_out[g]` distribution.
3. Set the packet weight to the positive outgoing sum.
4. Set the packet frequency to the sampled target group center.
5. Deposit `W_in - W_packet_new` into the material in the current cell.

This preserves one packet per incoming packet, keeps packet weights positive, and changes the packet before it crosses into the next cell.

## Negative Components

The implicit segment solve should normally produce nonnegative outgoing components for the regimes we want. If small negative components appear, they are floored to zero and the material receives the compensating energy through `W_in - W_packet_new`, preserving total energy. Diagnostics record the amount of negative segment correction. Large negative corrections mean the local segment operator is not positivity preserving and the diagnostic should point to the cell, group, timestep segment, and Fleck/operator scale.

## Interaction With Existing IMC Pieces

- The physical/flecked absorption weight decay remains unchanged.
- The base effective absorption/reemission channel remains sampled as before.
- The beginning-of-step Compton source still creates only positive `Bpos` packets.
- `Bres` remains deterministic in the end-of-step residual correction.
- In the new segment mode, the end-of-step residual matrix for `K_target` is disabled, because that operator has already been applied along path segments.

## Expected Benefit

The packet carries the correct updated weight and group into every subsequent cell. The stiff Compton exchange is no longer delayed to the final census and is no longer represented by rare high-impact stochastic events. Total energy is conserved segment by segment through the material deposit.

## Validation

- `till_compton_mc` should show bounded particle counts and no late-time stochastic residual oscillation.
- Compton diagnostics should show segment updates replacing implicit sampled events.
- `residual_material_exchange` should be dominated by `Bres`, not by a large stiff kernel solve.
- Total material plus radiation energy should remain conserved to roundoff/regression tolerance.
- The final gas/radiation temperature should approach the diffusion-run equilibrium scale.
