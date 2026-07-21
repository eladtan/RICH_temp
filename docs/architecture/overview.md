# Code Architecture Overview

RICH is a modular astrophysical hydrodynamics code built around pluggable components. This document describes the high-level architecture, directory layout, and key abstractions.

## Directory Layout

```
source/
├── 3D/                           # 3D geometry, mesh, and output
│   ├── elementary/               # Vector3D, Face, Tetrahedron
│   ├── tesselation/              # Voronoi, Delaunay, load balancing
│   │   ├── voronoi/              # Voronoi3D
│   │   ├── delaunay/             # Delaunay3D
│   │   ├── loadBalancing/        # Hilbert curve, ParMETIS
│   │   └── utils/                # Geometric predicates
│   ├── hilbert/                  # Hilbert space-filling curve
│   ├── gravity/                  # GravityTree, DistributedGravityTree
│   ├── range/                    # Spatial queries (OctTree, KDTree)
│   ├── output/                   # WriteSnapshot3D, Snapshot3D
│   ├── radiation/                # MonteCarloPhysics3D, RadiationIMC
│   ├── monte/                    # MonteCarloManager3D
│   ├── environment/              # Environment agents and kernels
│   └── GeometryCommon/           # RoundGrid3D, intersections
├── newtonian/                    # Hydrodynamics
│   ├── one_dimensional/          # 1D hydro sim and main loop
│   ├── two_dimensional/          # 2D hydro sim and AMR
│   ├── three_dimensional/        # Main 3D hydro logic
│   │   ├── hdsim_3d.hpp/.cpp     # HDSim3D: main simulation class
│   │   ├── Simulation.hpp        # Alternative pluggable orchestration
│   │   ├── computational_cell.hpp # ComputationalCell3D (primitives)
│   │   ├── conserved_3d.hpp      # Conserved3D (conserved variables)
│   │   ├── Hllc3D.hpp            # HLLC Riemann solver
│   │   ├── LinearGauss3D.hpp     # Linear Gauss reconstruction
│   │   ├── Lagrangian3D.hpp      # Lagrangian mesh motion
│   │   ├── RoundCells3D.hpp      # RoundCells mesh motion
│   │   ├── eulerian_3d.hpp       # Eulerian (static) mesh
│   │   ├── AMR3D.hpp             # Adaptive mesh refinement
│   │   ├── CourantFriedrichsLewy.hpp # CFL time step
│   │   └── timing/               # Distributed timestep, advance
│   └── common/                   # Equations of state
├── Radiation/                    # Radiation transport
│   ├── Diffusion.hpp/.cpp        # Grey diffusion
│   ├── MultigroupDiffusion.hpp   # Multigroup diffusion
│   ├── RadiationDriver.hpp       # Abstract radiation interface
│   ├── conj_grad_solve.hpp       # Conjugate gradient solver
│   └── CMMC/                     # Compton Matrix Monte Carlo (includes nested submodule src/planck_integral)
├── mpi/                          # MPI utilities
│   ├── mpi_commands.hpp          # MPI wrappers
│   ├── ExchangeChain.hpp         # Ring exchange pattern
│   └── serialize/                # Serialization for MPI
├── ds/                           # Data structures (OctTree)
├── misc/                         # Utilities (mesh generators, I/O)
├── tessellation/                 # 2D tessellation
├── relativistic/                 # Special relativity
└── opt/                          # Third-party (r3d, VCL, Clipper)
```

## Key Abstractions

RICH is designed around abstract interfaces that can be combined to build different simulations:

```
HDSim3D (main simulation object)
  ├── Tessellation3D          (mesh)
  ├── EquationOfState         (thermodynamics)
  ├── PointMotion3D           (mesh motion strategy)
  ├── TimeStepFunction3D      (CFL time step)
  ├── FluxCalculator3D        (face fluxes)
  ├── CellUpdater3D           (conserved -> primitive)
  ├── ExtensiveUpdater3D      (flux -> conserved update)
  └── SourceTerm3D            (external forces)
```

### Core Classes

| Class | Responsibility |
|-------|---------------|
| `HDSim3D` | Owns all simulation state; drives time advance |
| `Tessellation3D` | Abstract 3D mesh interface |
| `Voronoi3D` | Concrete Voronoi tessellation |
| `ComputationalCell3D` | Primitive variables (rho, P, e, v, T, Erad, tracers) |
| `Conserved3D` | Conserved variables (mass, momentum, energy) |

### Solver Components

| Interface | Implementations | Purpose |
|-----------|----------------|---------|
| `FluxCalculator3D` | `ConditionActionFlux1` | Face flux computation |
| `RiemannSolver3D` | `Hllc3D` | Riemann problem at cell interfaces |
| `SpatialReconstruction3D` | `LinearGauss3D` | Second-order reconstruction |
| `CellUpdater3D` | `DefaultCellUpdater`, `EOSConsistent` | Conserved to primitive |
| `ExtensiveUpdater3D` | `ConditionExtensiveUpdater3D` | Flux integration |
| `PointMotion3D` | `Lagrangian3D`, `Eulerian3D`, `RoundCells3D` | Mesh point velocities |
| `TimeStepFunction3D` | `CourantFriedrichsLewy` | CFL-based dt |
| `SourceTerm3D` | `ZeroForce3D`, `ConservativeForce3D`, `DiffusionForce`, `SeveralSources3D` | External forces |
| `EquationOfState` | `IdealGas`, `Tillotson`, `OndrejEOS`, `MixedEOS` | Thermodynamics |

### Radiation

| Interface | Implementations |
|-----------|----------------|
| `RadiationDriver` | `Diffusion`, `MultigroupDiffusion` |
| `MonteCarloPhysics<T,Grid>` | `MonteCarloPhysics3D`, `RadiationIMC` |

## Data Flow

A single time step in RICH follows this sequence:

```
1. Compute time step (CFL condition)
        │
2. Move mesh points (PointMotion3D)
        │
3. Rebuild tessellation (Voronoi3D)
        │
4. Spatial reconstruction (LinearGauss3D)
        │
5. Compute fluxes at faces (FluxCalculator3D → RiemannSolver3D)
        │
6. Update conserved quantities (ExtensiveUpdater3D)
        │
7. Apply source terms (SourceTerm3D: gravity, radiation forces)
        │
8. Convert conserved → primitive (CellUpdater3D)
        │
9. Radiation step (if enabled: Diffusion / MultigroupDiffusion)
        │
10. AMR (if enabled: refine / remove cells)
```

For higher-order time integration (`timeAdvance2`, `timeAdvance3`, `timeAdvance4`), steps 1-8 are repeated with intermediate states.

## The Simulation Class

An alternative to `HDSim3D` is the `Simulation` class, which uses a more modular "physics step" pattern:

```cpp
Simulation sim(tess, cells, eos, ...);
sim.addPhysics(new HydroStep(...));
sim.addPhysics(new RadiationStep(...));
sim.addPhysics(new GravityStep(...));
sim.advance();  // runs all physics steps in order
```

## Compile-Time Configuration

Some behaviors are controlled at compile time:

| Define | Effect |
|--------|--------|
| `RICH_MPI` | Enable MPI parallelism |
| `ENERGY_GROUPS_NUM` | Number of radiation energy groups |
| `DEBUG` | Enable debug assertions |
| `MC_DEBUG` | Enable Monte Carlo debug output |
| `ASAN` | Enable AddressSanitizer |

## Error Handling

RICH uses `UniversalError` for error reporting:

```cpp
try {
    sim.timeAdvance2();
} catch (UniversalError const& e) {
    reportError(e);  // prints diagnostic info
    throw;
}
```

`UniversalError` carries context about what went wrong (cell index, variable values, etc.), making debugging easier.
