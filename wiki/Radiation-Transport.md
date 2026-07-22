# Radiation Transport

RICH includes multiple radiation transport methods: grey diffusion, multigroup diffusion, and Monte Carlo transport. These are coupled to the hydrodynamics as physics steps.

## Grey Diffusion

Single-group flux-limited diffusion for radiation transport. Solves the radiation energy equation coupled to the material energy equation.

```cpp
#include "source/Radiation/Diffusion.hpp"

// Create the diffusion solver
Diffusion diffusion(/* coefficient calculator, boundary calculator, flux limiter */);
```

### Key Components

| Class | Purpose |
|-------|---------|
| `Diffusion` | Grey radiation diffusion driver |
| `DiffusionCoefficientCalculator` | Computes opacity/diffusion coefficients per cell |
| `DiffusionBoundaryCalculator` | Sets radiation boundary conditions |
| `RadiationDriver` | Abstract base class for radiation modules |

### Implicit Solve

The diffusion equation is solved implicitly using a conjugate gradient solver:

```cpp
#include "source/Radiation/conj_grad_solve.hpp"
```

This avoids the severe timestep restriction that explicit diffusion would impose.

## Multigroup Diffusion

Extends grey diffusion to `ENERGY_GROUPS_NUM` frequency groups, capturing spectral effects.

```cpp
#include "source/Radiation/MultigroupDiffusion.hpp"

MultigroupDiffusion mg_diffusion(
    energy_groups_center,       // center frequency per group
    energy_groups_boundary,     // boundary frequencies (N+1 values for N groups)
    opacity,                    // MultigroupDiffusionCoefficientCalculator
    boundary,                   // MultigroupDiffusionBoundaryCalculator
    eos,                        // EquationOfState
    zero_cells,                 // list of tracer names for cells to skip
    flux_limit,                 // enable flux limiter (bool)
    hydro_on,                   // enable radiation-matter coupling (bool)
    compton_on,                 // enable Compton scattering (bool)
    doppler_on,                 // enable Doppler corrections (bool)
    minimum_temperature,        // temperature floor (default: -1 = off)
    protections_on              // enable numerical protections (bool)
);
```

### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `energy_groups_center` | `vector<double>` | Center frequency of each energy group |
| `energy_groups_boundary` | `vector<double>` | Group boundary frequencies (N+1 values for N groups) |
| `opacity` | `MultigroupDiffusionCoefficientCalculator` | Per-group opacity calculator |
| `boundary` | `MultigroupDiffusionBoundaryCalculator` | Radiation boundary conditions |
| `eos` | `EquationOfState` | Equation of state for temperature/energy conversion |
| `flux_limit` | `bool` | Enable flux limiter to prevent superluminal diffusion |
| `hydro_on` | `bool` | Enable radiation-matter energy exchange |
| `compton_on` | `bool` | Enable Compton scattering between groups |
| `doppler_on` | `bool` | Enable Doppler corrections for moving-frame radiation |
| `minimum_temperature` | `double` | Temperature floor (default -1 disables) |
| `protections_on` | `bool` | Enable numerical protections for stability |

### Unit Scaling

When using physical units (e.g., CGS), set the unit scales on the solver:

```cpp
matrix_builder.length_scale_ = 7e10;   // cm (solar radius)
matrix_builder.time_scale_ = 1603;     // s
matrix_builder.mass_scale_ = 2e33;     // g (solar mass)
```

### Radiation Time Stepping

The radiation solver computes its own timestep, which is coordinated with the hydro step:

```cpp
double new_dt = sim->RadiationTimeStep(old_dt, matrix_builder);
```

### Configuration

The number of energy groups is set at compile time:

```bash
./build_rich.sh gnuReleaseMPI --test_name=my_run --energy_groups_num=32
```

The radiation energy per group is stored in `ComputationalCell3D::Eg[i]` for each cell.

### Grey Opacity Data

For grey diffusion, opacity tables are in `data/STA/`:

| File | Content |
|------|---------|
| `planck.txt` | Planck mean opacity |
| `ross.txt` | Rosseland mean opacity |
| `scatter.txt` | Scattering opacity |

## Compton Scattering (CMMC)

The Compton Matrix Monte Carlo (CMMC) module handles energy exchange between photons and electrons via Compton scattering in the multigroup framework.

```cpp
#include "source/Radiation/CMMC/ComptonKernel.hpp"
```

The Compton kernel redistributes photon energy between frequency groups, driving gas and radiation temperatures toward equilibrium.

### Usage

Enable Compton in the `MultigroupDiffusion` constructor:

```cpp
MultigroupDiffusion mg(
    groups_center, groups_boundary, opacity, boundary, eos,
    {}, /* flux_limit */ true, /* hydro_on */ true,
    /* compton_on */ true,  // enable Compton
    /* doppler_on */ true,
    /* min_temp */ 2000, /* protections */ true
);
```

See `regression_tests/cases/till_compton/test.cpp` for a focused equilibration test, and [Example: TDE Simulation](Example-TDE-Simulation) for a full production TDE with Compton.

## Multigroup Opacity Tables

To use multigroup diffusion with tabulated opacities, implement `MultigroupDiffusionCoefficientCalculator`. The `BaseTDECompton` simulation provides an example (`STAMGopacity`) that loads STA opacity tables:

```cpp
class STAMGopacity : public MultigroupDiffusionCoefficientCalculator
{
public:
    STAMGopacity(std::string file_directory);

    double CalcDiffusionCoefficientGroup(ComputationalCell3D const& cell, size_t group) const override;
    double CalcAbsorptionCoefficientGroup(ComputationalCell3D const& cell, size_t group) const override;
    double CalcScatteringCoefficientGroup(ComputationalCell3D const& cell, size_t group) const override;
};
```

The opacity table files (in `data/STA/MG/`) contain:

| File | Content |
|------|---------|
| `frequency_edges.txt` | Group boundary frequencies in eV |
| `T.txt` | Temperature grid in eV |
| `rho.txt` | Density grid in CGS |
| `sigma_rossland_N.txt` | Rosseland mean opacity per group |
| `sigma_absorption_rossland_N.txt` | Absorption opacity per group |
| `sigma_scattering_planck_N.txt` | Scattering opacity per group |

All look-ups use bilinear interpolation in log-density / log-temperature space with extrapolation handling at table boundaries.

### Initializing Per-Group Radiation Energy

For initial conditions, use `planck_integral` to compute the Planck spectrum in each group:

```cpp
#include "source/Radiation/CMMC/src/planck_integral/planck_integral.hpp"

for (size_t g = 0; g < Ng; ++g)
    cell.Eg[g] = planck_integral::planck_energy_density_group_integral(
        energy_groups_boundary[g], energy_groups_boundary[g+1], T);
```

## Monte Carlo Transport

RICH also includes a Monte Carlo radiation transport module for more general transport problems:

| Class | Purpose |
|-------|---------|
| `MonteCarloRadiationPhysics3D` | 3D Monte Carlo photon transport |
| `RadiationIMC` | Implicit Monte Carlo radiation |
| `MonteCarloManager` | Serial MC manager |
| `TwoSidedMonteCarloManager` | MPI-aware MC manager |

### Boundary Conditions

Monte Carlo boundary conditions are set via `BoundaryCondition` implementations.

## Planck Integral

The Planck integral library (`source/Radiation/CMMC/src/planck_integral/`) provides numerical evaluation of Planck function integrals needed for opacity weighting and energy group calculations.

## Coupling to Hydrodynamics

Radiation is coupled to hydrodynamics through source terms. The radiation pressure gradient acts as a force on the gas, and radiation-matter energy exchange affects the gas internal energy.

### DiffusionForce

```cpp
#include "source/Radiation/DiffusionForce.hpp"

DiffusionForce rad_force(/* ... */);
```

This source term adds the radiation pressure gradient to the momentum equation.

## Example: Marshak Wave

The Marshak wave tests in the regression suite provide clean examples of grey diffusion setup. See `regression_tests/cases/marshak_wave_1_diffusion/test.cpp` for a complete configuration with:

- Temperature-dependent opacities
- Time-dependent boundary temperature
- No flux limiter

## Example: Mach 2 Radiative Shock

The Mach 2 radiative shock tests demonstrate coupled radiation-hydrodynamics. See `regression_tests/cases/mach2_diffusion/test.cpp` for grey diffusion and `regression_tests/cases/mach2_multigroup/test.cpp` for multigroup.
