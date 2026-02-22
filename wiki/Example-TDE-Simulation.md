# Example: Tidal Disruption Event with Multigroup Radiation and Compton

This walkthrough documents the `BaseTDECompton` simulation -- a production-grade tidal disruption event (TDE) with multigroup radiation transport, Compton scattering, self-gravity, and adaptive mesh refinement. It is one of the most complex RICH simulations and showcases nearly every feature of the code.

## Physics

A star on a parabolic orbit passes within the tidal radius of a supermassive black hole and is torn apart. The simulation models:

- **Hydrodynamics** on a Lagrangian moving Voronoi mesh
- **Self-gravity** via a distributed gravity tree
- **Tidal gravity** from the black hole using a Paczynski-Wiita pseudo-Newtonian potential
- **Multigroup radiation diffusion** with flux limiting
- **Compton scattering** (energy exchange between photons and electrons)
- **Doppler corrections** for radiation in the moving frame
- **Tabular equation of state** ([OndrejEOS](Equations-of-State))
- **Tabular multigroup opacities** (STA tables)
- **Adaptive mesh refinement** with physics-aware criteria
- **Dynamic domain box** that expands as the debris spreads
- **Center draining** to handle material falling into the black hole
- **Reference frame switching** between co-moving and lab frames

## Unit System

The simulation uses a CGS-based unit system:

| Quantity | Scale | Value | Description |
|----------|-------|-------|-------------|
| Length | `lscale` | 7e10 cm | Approximately one solar radius |
| Mass | `mscale` | 2e33 g | One solar mass |
| Time | `tscale` | 1603 s | Derived dynamical time |

These scales are passed to `OndrejEOS` and `MultigroupDiffusion` for consistent unit conversion.

## Configuration via Text Files

The simulation reads physical parameters from text files at runtime:

| File | Parameter | Typical Value |
|------|-----------|---------------|
| `Rstar.txt` | Stellar radius (code units) | 1.0 (= 7e10 cm) |
| `Mstar.txt` | Stellar mass (code units) | 1.0 (= 2e33 g) |
| `Mbh.txt` | Black hole mass (code units) | 1e5 (= 2e38 g) |
| `beta.txt` | Penetration factor (Rt/Rp) | 1.0 |
| `n.txt` | Polytropic index | 1.5 or 3.0 |

This approach allows the same binary to run different TDE configurations without recompilation.

## Paczynski-Wiita Gravity

The black hole potential uses the Paczynski-Wiita pseudo-Newtonian form, which mimics the innermost stable circular orbit of a Schwarzschild black hole:

```
Phi(r) = -M_BH / (r - Rg),    Rg = 4.21 * M_BH / 1e6
```

where `Rg` is the gravitational radius. The `PaczynskiOrbit` class integrates the stellar orbit using a Runge-Kutta-Cash-Karp 5(4) ODE integrator from Boost.

See [Gravity](Gravity) for more on the gravity system.

## TDE Gravity Model

The `TDEGravity` class combines:

1. **Self-gravity** (`GravityAcceleration3D`) -- tree-based N-body gravity
2. **Tidal field** -- BH gravity minus the center-of-mass acceleration (tidal approximation)
3. **Smoothing** -- near the tidal radius, gravity is smoothed to prevent singularities
4. **Density cutoff** -- very low-density cells and non-stellar material (`Star` tracer < 0.1) feel no gravity

The simulation operates in two phases:
- **Tidal phase** (`tide_on_ = true`): Star follows a Keplerian orbit; gravity includes the tidal field
- **Full gravity phase** (`tide_on_ = false`): After disruption, when debris spreads significantly, the simulation switches to the lab frame with full self-gravity only

The transition is triggered automatically by `CheckIfFullGravityIsNeeded()`, which monitors whether debris has moved far enough from the BH.

## Multigroup Radiation with Compton Scattering

The `MultigroupDiffusion` solver is configured with all physics enabled:

```cpp
bool const hydro_on = true;      // radiation-matter coupling
bool const compton_on = true;     // Compton energy exchange
bool const flux_limit = true;     // flux limiter
bool const doppler_on = true;     // Doppler corrections
bool const protection_on = true;  // numerical protections

MultigroupDiffusion matrix_builder(
    opacity.energy_groups_center,
    opacity.energy_groups_boundary,
    opacity,
    D_boundary,
    eos,
    std::vector<std::string>(),
    flux_limit, hydro_on, compton_on, doppler_on,
    2000,           // minimum temperature floor
    protection_on
);

// Set unit scales for radiation
matrix_builder.length_scale_ = lscale;  // 7e10 cm
matrix_builder.time_scale_ = tscale;    // 1603 s
matrix_builder.mass_scale_ = mscale;    // 2e33 g
```

See [Radiation Transport](Radiation-Transport) for the full radiation API.

### STAMGopacity -- Multigroup Opacity Tables

The `STAMGopacity` class loads multigroup opacity data from STA tables (`data/STA/MG/`). It implements `MultigroupDiffusionCoefficientCalculator` and provides per-group Rosseland, absorption (Planck), and scattering opacities. All are tabulated in log-density / log-temperature with bilinear interpolation.

### Radiation Time Stepping

The radiation step runs separately from the hydro step:

```cpp
double new_dt = sim->RadiationTimeStep(old_dt, matrix_builder);
new_dt = std::max(2.01e-4, new_dt);
if (new_dt < 0.5 * old_dt)
    new_dt = 0.5 * old_dt;
new_dt = std::min(new_dt, 0.03);
```

## Tabular Equation of State (OndrejEOS)

`OndrejEOS` is constructed with 7 table files and 3 unit scales. Beyond the standard `EquationOfState` interface, it provides `dT2e`, `dT2p`, `dp2s`, `dp2T`, `de2T`, and more. See [Equations of State](Equations-of-State) for the full API.

## Lane-Emden Initial Conditions with Radiation

The stellar initial conditions include radiation pressure. Temperature is found by solving:

```
P_gas(rho, T) + P_rad(T) = P_polytrope(rho)
```

where P_rad = a * T^4 / 3. A root-finding algorithm (Boost `bracket_and_solve_root`) solves for T at each cell. Per-group radiation energies are initialized from Planck integrals at the found temperature.

## Tracers

The simulation uses 5 tracers:

| Index | Name | Purpose |
|-------|------|---------|
| 0 | `Entropy` | Thermodynamic entropy per cell |
| 1 | `Star` | 1 inside the star, 0 outside (material tagging) |
| 2 | `WasRemoved` | Tracks mass removed by center draining |
| 3 | (unnamed) | Additional tracer |
| 4 | (unnamed) | Additional tracer |

## Adaptive Mesh Refinement

The AMR strategy is highly physics-aware. See [AMR](AMR) for the full documentation.

### Refinement

Cells are refined when cell mass exceeds a distance-dependent threshold, when volume exceeds a target in the disruption midplane, or when a cell is much larger than its neighbors.

### Removal

Cells are removed when they are too small (causing tiny timesteps), when their mass is below threshold, or when they are far from the action region. Neighbor quality checks prevent creating large volume ratios.

### Dynamic Domain Box

Every 7 cycles, `UpdateBox()` checks if mesh points approach the domain boundary and expands the box as needed. New cells are initialized with a low-density reference state.

## Center Draining

The `RemoveCenter()` function handles material falling toward the BH:
- Within the smoothing radius, density is reduced by 20% each call
- Temperature is clamped between 1e4 K and 1e7 K
- High velocities are damped
- Removed mass is tracked via the `WasRemoved` tracer
- Radiation energy is scaled with the density change

## Diagnostic Output

The simulation writes HDF5 snapshots with additional diagnostic fields:

| Diagnostic | Name | Description |
|------------|------|-------------|
| Gradient | `DsieDx/y/z` | Internal energy gradient components |
| Gradient | `DpDx/y/z` | Pressure gradient components |
| Gradient | `DrhoDx/y/z` | Density gradient components |
| Divergence | `divV` | Velocity divergence |
| Dissipation | `Dissipation` | Numerical dissipation per cell |

See [Output and Visualization](Output-and-Visualization) for more on snapshots and diagnostics.

## Restart System

The restart system handles long-running simulations:

1. **Counter file** (`counter.txt`): Tracks the current snapshot index
2. **Periodic snapshots** (`snap_N.h5`): Written when simulation time exceeds `nextT`
3. **Checkpoint** (`restart.h5`): Written when wall time exceeds ~4 hours
4. **Smart restart**: On restart, compares timestamps and loads whichever file is newer
5. **MPI rank handling**: If the restart file has more MPI ranks than the current run, extra rank data is merged into rank 0

See [Simulation Setup](Simulation-Setup) for more on restarts and diagnostics.

## Build and Run

```bash
# Build with multigroup radiation (e.g., 32 groups)
./build_rich.sh intelReleaseMPI --test_name=BaseTDECompton --energy_groups_num=32

# Create parameter files
echo "1.0" > Rstar.txt
echo "1.0" > Mstar.txt
echo "1e5" > Mbh.txt
echo "1.0" > beta.txt
echo "1.5" > n.txt

# Run via SLURM
sbatch --exclusive --partition=bigrun --ntasks=512 \
  --output=output_%j --error=error_%j \
  --wrap "mpirun -x UCX_TLS=ib -mca btl ^openib -np 512 ./build/intelReleaseMPI/rich"
```

## Key Concepts Demonstrated

- Production-grade astrophysical simulation workflow
- Runtime configuration via parameter files
- Tabular EOS and multigroup opacity tables with unit scaling
- Paczynski-Wiita pseudo-Newtonian gravity
- Combined self-gravity and tidal gravity
- Reference frame switching during simulation
- Multigroup radiation diffusion with Compton scattering, flux limiting, and Doppler corrections
- Physics-aware AMR with distance-dependent criteria
- Dynamic domain expansion
- Center draining for BH accretion
- Custom diagnostic appendices for gradients and dissipation
- Robust restart with timestamp-based snapshot selection
- Wall-time checkpointing for HPC job schedulers
