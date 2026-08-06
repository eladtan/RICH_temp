# Gravity

RICH includes tree-based gravity solvers for self-gravitating systems, as well as specialized gravity models for tidal disruption events (TDE).

## Self-Gravity

### GravityTree

The serial gravity tree computes gravitational accelerations using a Barnes-Hut octree algorithm:

```cpp
#include "source/3D/gravity/GravityTree.hpp"

GravityTree gravity(/* opening angle theta */);
```

### DistributedGravityTree

The MPI-parallel version distributes the tree across processes:

```cpp
#include "source/3D/gravity/DistributedGravityTree.hpp"

DistributedGravityTree gravity(/* opening angle theta */);
```

### QuadrupoleGravity3D

Higher-order multipole expansion for improved accuracy:

```cpp
#include "source/newtonian/three_dimensional/QuadrupoleGravity3D.hpp"

QuadrupoleGravity3D gravity(/* ... */);
```

## Gravity as a Source Term

Gravity is applied as an acceleration source term in the hydro equations:

```cpp
#include "source/newtonian/three_dimensional/GravityAcc3D.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"

// Create gravity acceleration (opening_angle, symmetric_flag, softening)
GravityAcceleration3D grav_acc(1.05, true, 1.0);

// Wrap as a conservative force (preserves energy)
auto gravity_force = std::make_shared<ConservativeForce3D>(grav_acc, false);
```

The `ConservativeForce3D` wrapper ensures that gravitational work is properly accounted for in the energy equation. The second argument (`false`) disables external potential energy tracking when not needed.

## TDE Gravity

For tidal disruption event simulations, a composite gravity model combines self-gravity with the tidal field of a central black hole.

### Paczynski-Wiita Potential

The black hole gravity uses the Paczynski-Wiita pseudo-Newtonian potential, which mimics the innermost stable circular orbit of a Schwarzschild black hole:

```
Phi(r) = -M_BH / (r - Rg),    Rg = 4.21 * M_BH / 1e6
```

The stellar orbit is integrated using a Runge-Kutta-Cash-Karp 5(4) ODE integrator (from Boost).

### TDEGravity Class

The `TDEGravity` class (typically defined in the run's `test.cpp`) combines:

1. **Self-gravity**: Tree-based N-body gravity (`GravityAcceleration3D`)
2. **Tidal field**: BH gravity at each cell position minus the center-of-mass acceleration (tidal approximation)
3. **Smoothing**: Near the tidal radius, gravity is softened to prevent singularities
4. **Density cutoff**: Very low-density cells and non-stellar material (`Star` tracer < 0.1) feel no gravity

```cpp
TDEGravity acc(Mbh, M, R, beta, sg, /* tide_on */ true);
auto gravity_force = std::make_shared<ConservativeForce3D>(acc, false);
```

### Two-Phase Gravity

TDE simulations operate in two gravity phases:

| Phase | `tide_on_` | Description |
|-------|-----------|-------------|
| Tidal | `true` | Star follows a Keplerian orbit; gravity includes the tidal field. Simulation is in the co-moving frame of the star's center of mass. |
| Full | `false` | After disruption, the simulation switches to the lab frame with full self-gravity only. |

The transition is triggered automatically by `CheckIfFullGravityIsNeeded()`, which monitors whether debris has moved far enough from the BH. When triggered, `UpdateReferenceFrame()` shifts all positions and velocities to the lab frame, and the simulation restarts from that point.

See `runs/BaseTDECompton/test.cpp` for the full implementation.

## Combining Sources

Multiple source terms can be combined using `SeveralSources3D`:

```cpp
#include "source/newtonian/three_dimensional/SeveralSources3D.hpp"

std::vector<std::shared_ptr<SourceTerm3D>> forces;
forces.push_back(gravity_force);
// forces.push_back(radiation_force);
SeveralSources3D combined(forces);
```

## No Gravity

For simulations without gravity, use `ZeroForce3D`:

```cpp
ZeroForce3D force;
```

## Example: Lane-Emden Star

The Lane-Emden regression test (`regression_tests/cases/lane_self_gravity/test.cpp`) demonstrates self-gravity setup for a polytropic star:

1. Initial conditions from Lane-Emden tables (`data/xsi32.txt`, `data/theta32.txt`)
2. `DistributedGravityTree` for parallel gravity computation
3. `ConservativeForce3D` wrapping `GravityAcceleration3D`
4. The star should remain in hydrostatic equilibrium

## Performance Notes

- The opening angle parameter controls the accuracy/speed trade-off in the tree. Smaller angles are more accurate but slower.
- `DistributedGravityTree` scales well with MPI processes for large particle counts.
- Gravity computation is typically the most expensive part of self-gravitating simulations.

## Serial FMM Stage 1

The serial Fast Multipole Method backend is available as a separate acceleration object:

```cpp
#include "source/newtonian/three_dimensional/FastMultipoleAcceleration3D.hpp"

FmmGravityOptions options;
options.expansionOrder = 4;
options.thetaCritical = 0.5;
options.leafCapacity = 32;
options.computePotential = false;

FastMultipoleAcceleration3D acceleration(options, G);
ConservativeForce3D force(acceleration, false);
```

Stage 1 runs a serial adaptive dual-tree FMM with exact P2P near-field interactions and Taylor P2M/M2M/M2L/L2L/L2P far-field interactions. `expansionOrder` controls truncation accuracy, while smaller `thetaCritical` values accept fewer far-field node pairs and approach the direct-sum limit. The solver rebuilds the tree each call, uses the same unsoftened Newtonian sign convention as `GravityAcceleration3D`, multiplies by `G` only in the adapter, and exposes phase timings and interaction counters through `getLastStats()`.

This backend is serial-only. In an MPI build, `FastMultipoleAcceleration3D` throws rather than computing gravity from rank-local masses. Distributed FMM source exchange is a later stage.
