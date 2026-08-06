# Serial FMM Gravity Stage 1

Stage 1 adds a serial self-gravity backend beside the existing Barnes-Hut path. The public entry points are:

- `SerialFmmGravityCalculator` in `source/3D/gravity/fmm/`
- `FastMultipoleAcceleration3D` in `source/newtonian/three_dimensional/`

The numerical path is a genuine adaptive dual-tree FMM. Leaf P2M and internal M2M form source expansions, accepted source/target node pairs use M2L, parent local fields propagate through L2L, and L2P evaluates the potential and analytic gradient. Non-admissible leaf pairs use the exact P2P kernel. The traversal accepts separate target and source trees so later distributed orchestration does not need to reinterpret a remote multipole as a particle.

Key conventions:

- Kernel sum: `u(x) = sum_j m_j / |x - x_j|`.
- Unscaled acceleration: `grad u = -sum_j m_j (x - x_j) / |x - x_j|^3`.
- `FastMultipoleAcceleration3D` multiplies by `G` exactly once.
- Distinct particles at identical coordinates throw `UniversalError`; no softening is introduced in Stage 1.
- Only owned physical cells are extracted by the hydro adapter through `tess.GetPointNo()`.
- The adapter constructor and call operator throw in `RICH_MPI` builds until remote source exchange is implemented; it never runs local-only gravity silently.
- `computePotential` must match whether a potential output is supplied.

The production expansion basis is a complete symmetric Cartesian Taylor basis through total degree `p`. Multipoles store `sum(m s^alpha / alpha!)`; local coefficients multiply ordinary target monomials. M2L uses Cartesian derivatives of `1/r`, truncated at total degree `p`. During each solve, M2L translation operators are cached by source-to-target center displacement so repeated octree offsets reuse the same derivatives and coefficient layout. This convention makes P2M, M2M, M2L, L2L, and the analytic L2P gradient independently testable. The compact real solid-harmonic utility uses `C_n^m = r^n P_n^m(cos(theta)) exp(i m phi)` without the Condon-Shortley phase and stores `m=0`, then real/imaginary parts for positive `m`.

The flat tree uses RICH's existing octant bit order (`x`, `y`, `z` as bits 2, 1, 0), deterministic child masks, checked particle ranges, spatial keys, and explicit pre/post orders. It continues same-octant subdivision until capacity, depth, or floating-point resolution stops it. Domain containment and all narrowing/storage bounds are checked.

Admissibility requires non-overlapping cubes and `(target.radius + source.radius) / separation <= thetaCritical`. Otherwise the larger-radius nonterminal node is split, with target-side tie breaking. Diagnostics report real M2L/P2P counts, upward/interaction/downward timings from a monotonic clock, root mass from the root monopole, retained solver memory, leaf occupancy, and rejection reasons.

`fmm_quadrupole_benchmark` provides the Stage 1 performance and accuracy A/B test. It compares complete serial FMM and production quadrupole-tree build+walk time at 256, 512, 1024, 2048, and 16384 particles. Both use `theta=0.5`; FMM uses `p=4` and leaf capacity 32. Timings are medians of five post-warm-up measurements with alternating method order. Accuracy is measured against the same long-double direct sum with the cancellation-safe force-scale denominator. At 16384 particles, the FMM runtime must be within 25% of the quadrupole tree, its growth from 2048 particles must be smaller, and its scaled error must not exceed 1.25 times the quadrupole-tree error.
