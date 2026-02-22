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
- **Tabular equation of state** (OndrejEOS)
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
Phi(r) = -M_BH / (r - Rg)
```

where `Rg = 4.21 * M_BH / 1e6` is the gravitational radius. The `PaczynskiOrbit` class integrates the stellar orbit using a Runge-Kutta-Cash-Karp 5(4) ODE integrator from Boost.

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

### STAMGopacity -- Multigroup Opacity Tables

The `STAMGopacity` class loads multigroup opacity data from STA tables (`data/STA/MG/`). It implements the `MultigroupDiffusionCoefficientCalculator` interface and provides per-group:

- **Rosseland opacity** (for diffusion coefficient): `CalcDiffusionCoefficientGroup()`
- **Absorption (Planck) opacity**: `CalcAbsorptionCoefficientGroup()`
- **Scattering opacity**: `CalcScatteringCoefficientGroup()`

All opacities are tabulated in log-density and log-temperature, with bilinear interpolation and extrapolation handling at table boundaries. The frequency group boundaries are read from `frequency_edges.txt` and converted from eV to CGS energy units.

### Radiation Time Stepping

The radiation step runs separately from the hydro step:

```cpp
double new_dt = sim->RadiationTimeStep(old_dt, matrix_builder);
// Floor on timestep to prevent stalling
new_dt = std::max(2.01e-4, new_dt);
// Prevent abrupt timestep changes
if (new_dt < 0.5 * old_dt)
    new_dt = 0.5 * old_dt;
new_dt = std::min(new_dt, 0.03);
```

## Tabular Equation of State (OndrejEOS)

`OndrejEOS` is constructed with 7 table files and 3 unit scales:

```cpp
OndrejEOS eos(
    eos_location + "density.txt",
    eos_location + "Pfile.txt",
    eos_location + "csfile.txt",
    eos_location + "Sfile.txt",
    eos_location + "Ufile.txt",
    eos_location + "Tfile.txt",
    eos_location + "CVfile.txt",
    lscale, mscale, tscale
);
```

Beyond the standard `EquationOfState` interface, it provides:

| Method | Conversion |
|--------|-----------|
| `dT2e(rho, T, tracers)` | Density + temperature to internal energy |
| `dT2p(rho, T, tracers)` | Density + temperature to pressure |
| `dp2s(rho, P, tracers)` | Density + pressure to entropy |
| `dp2T(rho, P, tracers)` | Density + pressure to temperature |
| `de2T(rho, e, tracers)` | Density + energy to temperature |

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

The AMR strategy is highly physics-aware:

### Refinement (`MassRefine`)

Cells are refined when:
- Cell mass exceeds `MaxMass` (scaled by distance from tidal radius)
- Cell volume exceeds target volume in the disruption midplane
- Cell is much larger than its smallest neighbor (smoothing)
- Cell is within the apocenter region with significant density

Cells are NOT refined when:
- Cell width is below the minimum cell size (`Rt * 1e-2`)
- Cell center-of-mass is offset from the mesh point (distorted cell)
- Cell is far from the action region

### Removal (`RemoveBig`)

Cells are removed (coarsened) when:
- Cell is too small (below minimum width, or causing tiny timesteps)
- Cell mass is below the threshold
- Cell is far from the tidal radius

Neighbor quality checks prevent removal if it would create large volume ratios between adjacent cells.

### Dynamic Domain Box

Every 7 cycles, `UpdateBox()` checks if mesh points are approaching the domain boundary and expands the box as needed. New cells in the expanded region are initialized with a low-density reference state.

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

These are produced by `DiagnosticAppendix3D` implementations (`GradDiag`, `DissipationDiag`).

## Restart System

The restart system handles long-running simulations:

1. **Counter file** (`counter.txt`): Tracks the current snapshot index
2. **Periodic snapshots** (`snap_N.h5`): Written when simulation time exceeds `nextT`
3. **Checkpoint** (`restart.h5`): Written when wall time exceeds 15000 seconds (about 4 hours)
4. **Smart restart**: On restart, compares timestamps of `snap_N.h5` and `restart.h5` and loads whichever is newer
5. **MPI rank handling**: If the restart file has more MPI ranks than the current run, extra rank data is merged into rank 0

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
