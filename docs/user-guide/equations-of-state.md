# Equations of State

RICH supports multiple equations of state (EOS) through an abstract `EquationOfState` interface. The EOS converts between thermodynamic variables (density, pressure, internal energy, temperature) and computes the sound speed.

## Available Equations of State

### IdealGas

The simplest EOS, suitable for most hydrodynamic test problems.

```
P = (gamma - 1) * rho * e
```

```cpp
#include "source/newtonian/common/ideal_gas.hpp"

IdealGas eos(5.0/3.0);  // gamma = 5/3 (monatomic gas)
IdealGas eos14(1.4);     // gamma = 1.4 (diatomic gas)
```

| Parameter | Description |
|-----------|-------------|
| `gamma` | Adiabatic index (ratio of specific heats) |

Common values: 5/3 (monatomic), 7/5 = 1.4 (diatomic), 4/3 (relativistic gas / radiation-dominated).

### Tillotson EOS

A high-pressure equation of state used for impact and planetary science simulations. Models materials under extreme compression and expansion.

```cpp
#include "source/newtonian/common/Tillotson.hpp"

Tillotson eos(/* material parameters */);
```

The original Tillotson formulation (`TillotsonOrg`) is also available.

### OndrejEOS

A tabular equation of state loaded from data files. Used in production runs such as tidal disruption events (TDE). Based on the OPAL/SCVH tables, it covers a wide range of densities and temperatures relevant to stellar interiors.

```cpp
#include "source/newtonian/three_dimensional/OndrejEOS.hpp"

std::string eos_dir = "data/EOS/";
OndrejEOS eos(
    eos_dir + "density.txt",
    eos_dir + "Pfile.txt",
    eos_dir + "csfile.txt",
    eos_dir + "Sfile.txt",
    eos_dir + "Ufile.txt",
    eos_dir + "Tfile.txt",
    eos_dir + "CVfile.txt",
    lscale,   // length unit in cm (e.g. 7e10 for solar radius)
    mscale,   // mass unit in g (e.g. 2e33 for solar mass)
    tscale    // time unit in s (e.g. 1603)
);
```

The three unit-scale arguments convert the internal code units to CGS for the table lookups.

The EOS data directory (`data/EOS/`) contains:

| File | Content |
|------|---------|
| `density.txt` | Log-density grid |
| `Pfile.txt` | Pressure table |
| `csfile.txt` | Sound speed table |
| `Sfile.txt` | Entropy table |
| `Ufile.txt` | Internal energy table |
| `Tfile.txt` | Temperature table |
| `CVfile.txt` | Heat capacity table |

#### Extended Methods

Beyond the standard `EquationOfState` interface, `OndrejEOS` provides additional thermodynamic conversions:

| Method | Conversion |
|--------|-----------|
| `dT2e(rho, T, tracers)` | Density + temperature to internal energy |
| `dT2p(rho, T, tracers)` | Density + temperature to pressure |
| `dp2s(rho, P, tracers)` | Density + pressure to entropy |
| `dp2T(rho, P, tracers)` | Density + pressure to temperature |
| `de2T(rho, e, tracers)` | Density + energy to temperature |
| `dT2cv(rho, T, tracers)` | Density + temperature to heat capacity |
| `sd2p(s, rho, tracers)` | Entropy + density to pressure |

These are essential for setting up initial conditions from stellar profiles (e.g., Lane-Emden) where temperature is known.

### MixedEOS

Combines multiple equations of state, applying different EOS to different materials tracked by sticker fields.

```cpp
#include "source/newtonian/common/MixedEos.hpp"

// Define EOS for each material
IdealGas gas_eos(5.0/3.0);
Tillotson solid_eos(/* params */);

MixedEos eos(gas_eos, solid_eos, "MaterialSticker");
```

## EOS Interface

All equations of state implement the `EquationOfState` interface:

| Method | Description |
|--------|-------------|
| `de2p(rho, e, tracers, tracer_names)` | Density + energy to pressure |
| `de2c(rho, e, tracers, tracer_names)` | Density + energy to sound speed |
| `dp2e(rho, p, tracers, tracer_names)` | Density + pressure to energy |
| `dp2c(rho, p, tracers, tracer_names)` | Density + pressure to sound speed |

The `tracers` and `tracer_names` arguments allow the EOS to use composition-dependent behavior (e.g., `MixedEos`).

## Choosing an EOS

| Use Case | Recommended EOS |
|----------|----------------|
| Test problems (Sod, Sedov, Gresho) | `IdealGas` |
| Astrophysical simulations (TDE, stars) | `OndrejEOS` or tabular |
| Planetary impacts | `Tillotson` |
| Multi-material flows | `MixedEOS` |

## Adding a Custom EOS

Implement the `EquationOfState` interface:

```cpp
class MyEOS : public EquationOfState
{
public:
    double de2p(double density, double energy,
                tvector const& tracers,
                std::vector<std::string> const& tracer_names) const override
    {
        // your pressure calculation
    }

    double de2c(double density, double energy,
                tvector const& tracers,
                std::vector<std::string> const& tracer_names) const override
    {
        // your sound speed calculation
    }

    // ... implement dp2e, dp2c similarly
};
```
