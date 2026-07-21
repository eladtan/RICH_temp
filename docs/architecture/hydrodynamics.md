# Hydrodynamics

RICH solves the compressible Euler equations on a moving Voronoi mesh using a Godunov-type finite volume scheme.

## Governing Equations

The Euler equations in conservative form:

```
∂ρ/∂t + ∇·(ρv) = 0                          (mass)
∂(ρv)/∂t + ∇·(ρv⊗v + PI) = f               (momentum)
∂E/∂t + ∇·((E + P)v) = f·v + S              (energy)
```

where ρ is density, v is velocity, P is pressure, E = ρe + ρv²/2 is total energy density, f is external force, and S includes radiation source terms.

## Finite Volume Formulation

On a moving Voronoi mesh, the integral form of the conservation laws is:

```
d/dt ∫_V U dV = -∮_∂V F·n dA + ∫_V S dV
```

where U is the vector of conserved quantities, F is the flux tensor, and the surface integral is over all faces of the cell. The mesh motion is accounted for by subtracting the face velocity from the fluid velocity in the flux computation.

## Riemann Solver: HLLC

The HLLC (Harten-Lax-van Leer-Contact) approximate Riemann solver computes the inter-cell flux at each face:

```cpp
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
Hllc3D rs;
```

HLLC captures three waves: left and right acoustic waves and the contact discontinuity. This makes it suitable for both shocks and contact surfaces.

## Spatial Reconstruction: LinearGauss3D

Second-order accuracy is achieved via piecewise-linear reconstruction with Gauss gradient estimation:

```cpp
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
LinearGauss3D interp(eos, ghost);
```

The reconstruction:
1. Estimates gradients using a least-squares fit over neighboring cells
2. Extrapolates primitive variables from cell centers to face centers
3. Applies slope limiting to prevent spurious oscillations near discontinuities

The reconstructed left and right states are then fed to the Riemann solver.

## Flux Calculator: ConditionActionFlux1

The condition-action pattern dispatches different flux calculations based on face type:

```cpp
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"

std::vector<pair<const ConditionActionFlux1::Condition3D*,
    const ConditionActionFlux1::Action3D*>> sequence;

sequence.push_back({new IsBoundaryFace3D(), new RigidWallFlux3D(rs)});
sequence.push_back({new IsBulkFace3D(), new RegularFlux3D(rs)});

ConditionActionFlux1 flux(sequence, interp);
```

Faces are tested against conditions in order; the first match determines the flux calculation.

## Time Integration

RICH supports 1st through 4th order time integration:

| Method | Order | Steps | Description |
|--------|-------|-------|-------------|
| `timeAdvance()` | 1st | 1 | Forward Euler |
| `timeAdvance2()` | 2nd | 2 | Predictor-corrector |
| `timeAdvance3()` | 3rd | 3 | Multi-stage |
| `timeAdvance4()` | 4th | 4 | Multi-stage (most accurate) |

Each stage involves:
1. Mesh motion and tessellation rebuild
2. Gradient estimation and reconstruction
3. Riemann solve at all faces
4. Conserved variable update
5. Source term application

Higher-order methods are more accurate per time step but proportionally more expensive.

## CFL Time Step

The Courant-Friedrichs-Lewy condition limits the time step for stability:

```cpp
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"

double hydro_cfl = 0.3;   // CFL number for hydrodynamics
double force_cfl = 1.0;   // CFL number for force terms
CourantFriedrichsLewy tsf(hydro_cfl, force_cfl, force);
```

The time step is:

```
dt = min(hydro_cfl * min_i(R_i / (c_s,i + |v_i|)),
         force_cfl * min_i(R_i / |a_i|)^(1/2))
```

where R_i is the effective cell radius, c_s is sound speed, v is velocity, and a is acceleration.

## Cell Update

After fluxes are computed, conserved variables are updated and converted back to primitives:

```cpp
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
DefaultCellUpdater cu;
```

The `DefaultCellUpdater` uses the EOS to compute pressure from the updated density and internal energy. The `EOSConsistent` variant enforces strict thermodynamic consistency.

## Extensive Updater

The extensive updater integrates fluxes over faces to update conserved quantities:

```cpp
#include "source/newtonian/three_dimensional/default_extensive_updater.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"

std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*,
    const ConditionExtensiveUpdater3D::Action3D*>> eu_sequence;
ConditionExtensiveUpdater3D eu(eu_sequence);
```

## Source Terms

External forces (gravity, radiation) are applied as source terms:

| Class | Description |
|-------|-------------|
| `ZeroForce3D` | No external force |
| `ConservativeForce3D` | Wraps an `Acceleration3D` into a conservative force |
| `DiffusionForce` | Radiation pressure gradient |
| `SeveralSources3D` | Combines multiple source terms |

## 1D and 2D Support

RICH also includes 1D and 2D solvers:

- **1D**: `source/newtonian/one_dimensional/hdsim.hpp` -- used for the Sod shock tube
- **2D**: `source/newtonian/two_dimensional/hdsim2d.hpp` -- includes 2D AMR

These share the same solver architecture (Riemann solver, reconstruction, CFL) but operate on simpler mesh types.
