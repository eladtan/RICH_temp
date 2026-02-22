# Boundary Conditions

RICH supports several types of boundary conditions for the 3D Voronoi mesh. Boundary conditions are applied through ghost cell generators and the condition-action flux pattern.

## Overview

In RICH, boundary conditions are implemented through two mechanisms:

1. **Ghost generators**: Create ghost cells outside the domain that mirror or extend the interior state
2. **Condition-action flux**: Apply different flux calculations to boundary faces vs interior faces

## Available Boundary Conditions

### Rigid Wall (Reflective)

Reflects the flow at the boundary. The normal velocity component is reversed; tangential components are preserved. No mass flows through the boundary.

```cpp
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"

RigidWallGenerator3D ghost;
LinearGauss3D interp(eos, ghost);

// In the flux calculator, apply rigid wall flux to boundary faces
sequence.push_back({new IsBoundaryFace3D(), new RigidWallFlux3D(rs)});
sequence.push_back({new IsBulkFace3D(), new RegularFlux3D(rs)});
ConditionActionFlux1 flux(sequence, interp);
```

Use cases: enclosed domains, blast wave containment, Sedov tests.

### Periodic

Wraps the domain so that flow exiting one side enters from the opposite side. Requires the domain to be a box.

Use cases: turbulence simulations, infinite-medium problems, Gresho vortex.

### Inflow / Outflow

Fixed inflow conditions on one boundary with outflow (zero-gradient) on another.

```cpp
// Define inflow state
ComputationalCell3D inflow_cell;
inflow_cell.density = rho_upstream;
inflow_cell.pressure = P_upstream;
inflow_cell.velocity = Vector3D(v_upstream, 0, 0);
```

Use cases: wind tunnels, radiative shock problems, Mach 2 tests.

### Free Boundary

Extrapolates the interior state to ghost cells without reflection. Material can flow freely out of the domain.

Use cases: expanding flows, outflows.

## Condition-Action Pattern

RICH uses a flexible condition-action pattern for flux computation. Each face is tested against conditions in order; the first matching condition determines the flux calculation.

```cpp
std::vector<pair<const ConditionActionFlux1::Condition3D*,
    const ConditionActionFlux1::Action3D*>> sequence;

// Check boundary faces first
sequence.push_back({new IsBoundaryFace3D(), new RigidWallFlux3D(rs)});

// Then handle bulk (interior) faces
sequence.push_back({new IsBulkFace3D(), new RegularFlux3D(rs)});

ConditionActionFlux1 flux(sequence, interp);
```

### Available Conditions

| Condition | Matches |
|-----------|---------|
| `IsBoundaryFace3D` | Faces on the domain boundary |
| `IsBulkFace3D` | Interior faces between two real cells |

### Available Actions

| Action | Behavior |
|--------|----------|
| `RegularFlux3D` | Standard Riemann-based flux |
| `RigidWallFlux3D` | Reflective wall flux |

## Ghost Cell Generator

The ghost cell generator creates virtual cells outside the domain boundary to enable reconstruction and flux computation:

```cpp
RigidWallGenerator3D ghost;
LinearGauss3D interp(eos, ghost);
```

The ghost generator is passed to the spatial interpolation scheme so that gradients near boundaries are computed correctly.

## Extensive Updater Conditions

Similar to flux conditions, the extensive updater uses a condition-action pattern for updating conserved variables:

```cpp
std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*,
    const ConditionExtensiveUpdater3D::Action3D*>> eu_sequence;
ConditionExtensiveUpdater3D eu(eu_sequence);
```

Custom boundary treatments for conserved variables can be added here.

## MPI Domain Boundaries

In MPI simulations, inter-process boundaries are handled automatically by the Voronoi tessellation's parallel build. Ghost cells from neighboring MPI ranks are exchanged during the tessellation and flux computation steps. These are distinct from physical boundary conditions.
