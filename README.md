# RICH - Radiation Hydrodynamics Code

RICH is a compressible hydrodynamic simulation code on a moving mesh written in C++.

## Table of Contents

- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Framework Overview](#framework-overview)
  - [Quick Example](#quick-example)
  - [Complete Configuration Reference](#complete-configuration-reference)
  - [Advanced Features](#advanced-features)
- [Creating New Problems](#creating-new-problems)
- [Profiling](#profiling)

---

## Requirements

**Compilers:**
- GNU: gcc/g++/gfortran (7+)
- Intel: icx/icx-cc/ifx (OneAPI 2024+)

**Libraries:**
- CMake 3.20+
- HDF5 (with C++ bindings)
- VTK 9.x
- Boost
- pybind11
- OpenMP (optional, for performance)
- MPI (optional, for parallel runs)

### Installation

**macOS (Homebrew):**
```bash
brew install cmake gcc boost hdf5 vtk open-mpi
```

**Ubuntu/Debian:**
```bash
sudo apt install cmake gcc g++ gfortran libhdf5-dev libvtk9-dev libboost-all-dev libopenmpi-dev
```

**HPC/Cluster:**
```bash
# Load modules (names vary by system)
module load cmake gcc hdf5 vtk boost openmpi

# Save configuration (if using Lmod)
module save rich_build
```

---

## Quick Start

### Configuration

```bash
./config.py --problem=<name> [options]
```

**Required:**

- `--problem=<name>` - Problem directory in `runs/`

**Optional:**

- `--compiler=gnu|intel` (default: gnu)
- `--build=release|debug` (default: release)
- `--mpi` - Enable MPI for parallel runs
- `--profile` - Enable profiling with gprof
- `--energy-groups=N` - Multigroup radiation (default: 1)

### Compilation and Execution

```bash
# Clone with submodules
git clone --recursive https://gitlab.com/eladtan/RICH.git
cd RICH

# Configure for a specific problem
./config.py --problem=sedov3d_framework

# Build
make -j

# Run
./build/rich
```

**Examples:**

```bash
# Debug build
./config.py --problem=sedov3d_framework --build=debug
make -j

# Parallel MPI run
./config.py --problem=TDE_framework --mpi
make -j

# With profiling
./config.py --problem=sedov3d_framework --profile
make -j

# Intel compiler with MPI
./config.py --problem=TDE_framework --compiler=intel --mpi
make -j
```

---

## Framework Overview

The **RICH framework** provides a **declarative, configuration-based** interface for 3D hydrodynamic simulations.

### Quick Example

```cpp
#include "source/framework/problem_config_3d.hpp"
#include "source/framework/simulation_builder_3d.hpp"

using namespace rich3d;

ComputationalCell3D my_initial_condition(const Vector3D& position, const EquationOfState& eos) {
    ComputationalCell3D cell;
    cell.density = 1.0;
    cell.pressure = 1.0;
    cell.velocity = Vector3D(0, 0, 0);
    cell.internal_energy = eos.dp2e(cell.density, cell.pressure, cell.tracers);
    return cell;
}

int main() {
    Problem3DConfig config;
    config.name = "my_simulation";

    // Domain
    config.domain.lower_bound = Vector3D(-1, -1, -1);
    config.domain.upper_bound = Vector3D(1, 1, 1);

    // Mesh (defaults: RANDOM, 100k points, 10 round iterations)
    config.mesh.num_points = 10000;

    // Physics (required)
    config.physics.eos = std::make_shared<IdealGas>(5.0/3.0);

    // Initial condition (required)
    config.initial_condition = my_initial_condition;

    // Output
    config.output.output_directory = "./";
    config.output.output_prefix = "snap";
    config.output.mode = OutputConfig::Mode::CYCLE;
    config.output.cycle_interval = 100;
    config.output.final_time = 1.0;

    // Build and run!
    Simulation3DBuilder::build_and_run(config);
    return 0;
}
```

That's it! The framework automatically:

- Validates required configuration (EOS, initial condition)
- Creates the tessellation
- Sets up boundary conditions
- Configures numerical methods
- Runs the time evolution loop
- Writes output snapshots

---

## Complete Configuration Reference

### 1. Domain Configuration

```cpp
config.domain.lower_bound = Vector3D(-10, -10, -10);
config.domain.upper_bound = Vector3D(10, 10, 10);
config.domain.periodic_x = false;  // Default
config.domain.periodic_y = false;  // Default
config.domain.periodic_z = false;  // Default
```

### 2. Mesh Configuration

```cpp
// Default: RANDOM mesh with 100k points, 10 round iterations
config.mesh.type = MeshConfig::Type::RANDOM;       // Default
config.mesh.num_points = 100000;                   // For RANDOM
config.mesh.round_iterations = 10;                 // Default

// Alternative: Cartesian grid
config.mesh.type = MeshConfig::Type::CARTESIAN;
config.mesh.nx = 50;
config.mesh.ny = 50;
config.mesh.nz = 50;

// Custom mesh points (pre-processed)
config.mesh.type = MeshConfig::Type::CUSTOM;
config.mesh.custom_points = my_points;  // vector<Vector3D>
```

**Mesh types:**
- `RANDOM` - Random points with Lloyd relaxation (default)
- `CARTESIAN` - Regular grid
- `SPHERICAL` - Spherical coordinates (r, θ, φ) [not yet implemented]
- `CYLINDRICAL` - Cylindrical coordinates (r, φ, z) [not yet implemented]
- `CUSTOM` - User-provided points

### 3. Physics Configuration

```cpp
// Equation of state (required)
config.physics.eos = std::make_shared<IdealGas>(5.0/3.0);

// The EOS encapsulates the adiabatic index internally
```

### 4. Numerical Methods

```cpp
// Defaults (all optional - these are the default values)
config.numerical.cfl = 0.3;                                            // Default
config.numerical.max_time_step = 1e100;                                // Default
config.numerical.reconstruction = NumericalConfig::Reconstruction::LINEAR_GAUSS;  // Default
config.numerical.riemann_solver = NumericalConfig::RiemannSolver::HLLC;           // Default
config.numerical.grid_motion = NumericalConfig::GridMotion::ROUND_CELLS;          // Default
```

**Options:**
- **Reconstruction:** `PCM` (1st order), `LINEAR_GAUSS` (2nd order)
- **Riemann solver:** `HLLC`, `EXACT` [not yet implemented]
- **Grid motion:** `EULERIAN` (fixed), `LAGRANGIAN` (moves with fluid), `ROUND_CELLS` (regularized)

### 5. Boundary Conditions

```cpp
// Defaults: FREE_FLOW on all sides (all optional)
config.boundary.x_lower = BoundaryConfig::Type::FREE_FLOW;     // Default
config.boundary.x_upper = BoundaryConfig::Type::FREE_FLOW;     // Default
config.boundary.y_lower = BoundaryConfig::Type::FREE_FLOW;     // Default
config.boundary.y_upper = BoundaryConfig::Type::FREE_FLOW;     // Default
config.boundary.z_lower = BoundaryConfig::Type::FREE_FLOW;     // Default
config.boundary.z_upper = BoundaryConfig::Type::FREE_FLOW;     // Default

// Other options:
config.boundary.x_lower = BoundaryConfig::Type::RIGID_WALL;    // Reflective
config.boundary.x_upper = BoundaryConfig::Type::PERIODIC;      // Periodic (set in domain config too)
```

**Custom Boundary Conditions:**

For specialized boundary behavior, implement a custom `Ghost3D` class:

```cpp
#include "newtonian/three_dimensional/Ghost3D.hpp"

class MyInflowBoundary : public Ghost3D {
public:
    void operator()(const Tessellation3D& tess,
                   const vector<ComputationalCell3D>& cells,
                   double time,
                   boost::container::flat_map<size_t, ComputationalCell3D>& res) const override {
        // Implement custom ghost cell generation
        for(auto& pair : res) {
            ComputationalCell3D& ghost_cell = pair.second;
            // Set inflow values
            ghost_cell.density = 1.0;
            ghost_cell.pressure = 1.0;
            ghost_cell.velocity = Vector3D(1.0, 0, 0);  // Inflow in x-direction
        }
    }

    Slope3D GetGhostGradient(const Tessellation3D& tess,
                            const vector<ComputationalCell3D>& cells,
                            const vector<Slope3D>& gradients,
                            size_t ghost_index, double time,
                            size_t face_index) const override {
        return Slope3D();  // Zero gradient
    }
};

// Per-face custom boundaries
auto inflow = std::make_shared<MyInflowBoundary>();
auto outflow = std::make_shared<MyOutflowBoundary>();

config.boundary.x_lower = BoundaryConfig::Type::CUSTOM;
config.boundary.custom_x_lower = inflow;   // Custom on x_lower

config.boundary.x_upper = BoundaryConfig::Type::CUSTOM;
config.boundary.custom_x_upper = outflow;  // Different custom on x_upper

config.boundary.y_lower = BoundaryConfig::Type::FREE_FLOW;  // Standard on other faces
config.boundary.y_upper = BoundaryConfig::Type::FREE_FLOW;
config.boundary.z_lower = BoundaryConfig::Type::RIGID_WALL;
config.boundary.z_upper = BoundaryConfig::Type::RIGID_WALL;
```

### 6. Initial Conditions

```cpp
// Function signature: ComputationalCell3D(const Vector3D&, const EquationOfState&)
config.initial_condition = [](const Vector3D& pos, const EquationOfState& eos) {
    ComputationalCell3D cell;
    double r = abs(pos);
    cell.density = (r < 1.0) ? 1.0 : 0.1;
    cell.pressure = (r < 1.0) ? 1.0 : 0.1;
    cell.velocity = Vector3D(0, 0, 0);
    cell.internal_energy = eos.dp2e(cell.density, cell.pressure, cell.tracers);
    return cell;
};

// Or use a named function
ComputationalCell3D sedov_ic(const Vector3D& pos, const EquationOfState& eos) {
    ComputationalCell3D cell;
    double r = abs(pos);
    cell.density = 1.0;
    cell.velocity = Vector3D(0, 0, 0);
    cell.internal_energy = (r < 0.2) ? 1e5 : 0.1;  // Energy spike at center
    cell.pressure = eos.de2p(cell.density, cell.internal_energy);
    return cell;
}

config.initial_condition = sedov_ic;
```

### 7. Source Terms & Forces

```cpp
// Add gravity
auto gravity = std::make_shared<GravityAcceleration3D>(
    1.05,    // theta (opening angle for tree code)
    true,    // quadrupole correction
    1.0      // smoothing length factor
);
config.sources.accelerations.push_back(gravity);

// TDE tidal forces example
auto self_gravity = std::make_shared<GravityAcceleration3D>(1.05, true, 1.0);
auto tde_gravity = std::make_shared<TDEGravity>(
    M_bh, M_star, R_star, beta,
    *self_gravity,
    true  // enable tidal forces
);
config.sources.accelerations.push_back(tde_gravity);
config.sources.conservative_mass_flux = false;  // Default

// Custom source terms (implement SourceTerm3D interface)
config.sources.source_terms.push_back(my_custom_source);
```

### 8. Output Configuration

```cpp
config.output.output_directory = "./";
config.output.output_prefix = "snapshot";
config.output.format = OutputConfig::Format::HDF5;  // or VTK

// Output timing
config.output.mode = OutputConfig::Mode::CYCLE;     // or TIME
config.output.cycle_interval = 100;                 // Every N cycles
config.output.snapshot_interval = 0.1;              // Every Δt (if MODE::TIME)

// Termination
config.output.final_time = 1.0;
config.output.max_cycles = 1000000;

// Custom diagnostics
config.output.diagnostics.push_back(my_diagnostic);
```

### 9. Tracers and Stickers

```cpp
// Optional: Override default tracer/sticker names
config.tracer_names = {"metallicity", "chemical_id"};
config.sticker_names = {"boundary_flag"};

// If empty, uses ComputationalCell3D defaults
```

---

## Advanced Features

### Radiation Diffusion

```cpp
#include "Radiation/STAgreyOpacity.hpp"

auto opacity = std::make_shared<STAgreyOpacity>("path/to/opacity/STA/");

config.radiation.enabled = true;
config.radiation.type = RadiationConfig::Type::GRAY_DIFFUSION;
config.radiation.opacity = opacity;
config.radiation.boundary_type = RadiationConfig::BoundaryType::OPEN;

// Physics flags
config.radiation.compton_cooling = true;          // Default
config.radiation.flux_limiter_on = true;          // Default
config.radiation.flux_limiter = 1.0/3.0;          // Default

// Radiation time stepping (prevents instability)
config.radiation.use_radiation_timestep = true;   // Default
config.radiation.min_timestep = 2.01e-4;          // Default
config.radiation.max_timestep = 0.01;             // Default
config.radiation.timestep_smoothing = 0.5;        // Default (prevents rapid drops)
```

The framework automatically:
- Creates diffusion solver
- Adds radiation force to source terms
- Uses radiation-limited time step before each hydro advance
- Applies Compton cooling/heating

### Automatic Domain Expansion

For simulations with expanding material (TDEs, supernovae, blast waves), the domain can automatically grow to accommodate outflow:

```cpp
// Start with small domain
config.domain.lower_bound = Vector3D(-1, -1, -1);
config.domain.upper_bound = Vector3D(1, 1, 1);

// Enable automatic expansion
config.domain.dynamic.enabled = true;
config.domain.dynamic.update_frequency = 7;        // Check every 7 cycles
config.domain.dynamic.min_velocity = 0.5;          // Track material moving > 0.5
config.domain.dynamic.volume_fraction = 1e-5;      // New point density

// Define vacuum cell for expanded regions
// Signature: ComputationalCell3D(const EquationOfState&, double time)
config.domain.dynamic.new_cell_state = [](const EquationOfState& eos, double time) {
    ComputationalCell3D cell;
    cell.density = 1e-20;
    cell.pressure = eos.dT2p(cell.density, 1e7, cell.tracers);
    cell.velocity = Vector3D(0, 0, 0);
    cell.internal_energy = eos.dp2e(cell.density, cell.pressure, cell.tracers);
    cell.Erad = 1e-10;  // For radiation simulations
    return cell;
};
```

The framework automatically calls `UpdateBox` at the specified frequency:
- Identifies cells with velocity > threshold
- Calculates bounds of moving material
- Expands domain (never shrinks)
- Generates new points in expanded region
- Initializes new cells with reference state

**See:** [runs/sedov3d_framework/main.cpp](runs/sedov3d_framework/main.cpp) for a complete example with commented expansion code.

### Lifecycle Hooks

Inject custom behavior at specific points in the simulation:

```cpp
// Pre-step: Before each time step
config.hooks.pre_step = [](HDSim3D& sim, double time) {
    // Custom logic here
};

// Post-step: After each time step
config.hooks.post_step = [](HDSim3D& sim, double time) {
    // Remove center mass (accretion disk modeling)
    if(sim.getCycle() % 10 == 0) {
        RemoveCenter(sim, M_bh, M_star, R_star, eos, beta);
    }
};

// Pre-output: Before writing snapshots
config.hooks.pre_output = [](HDSim3D& sim, double time) {
    // Compute diagnostics before output
    interpolate_face_values(sim);
};

// Post-output: After writing snapshots
config.hooks.post_output = [](HDSim3D& sim, double time) {
    // Cleanup or logging
};

// Post-AMR: After mesh refinement
config.hooks.post_amr = [](HDSim3D& sim, double time) {
    // Update parameters after AMR
};
```

**Available hooks:**
- `pre_step` - Before each `timeAdvance2()`
- `post_step` - After each `timeAdvance2()`
- `pre_output` - Before `WriteSnapshot3D()`
- `post_output` - After `WriteSnapshot3D()`
- `post_amr` - After AMR refinement

### Adaptive Mesh Refinement (AMR)

AMR requires custom refinement/removal criteria:

**1. Define criteria classes:**

```cpp
class MyRefine : public CellsToRefine3D {
    std::pair<vector<size_t>, vector<Vector3D>> ToRefine(
        Tessellation3D const& tess,
        vector<ComputationalCell3D> const& cells,
        double time) const override {

        vector<size_t> cells_to_refine;
        vector<Vector3D> split_directions;  // Can be empty for default

        // Example: Refine high-mass cells
        double max_mass = 1e-5;
        for(size_t i = 0; i < cells.size(); ++i) {
            double volume = tess.GetVolume(i);
            double mass = cells[i].density * volume;
            if(mass > max_mass) {
                cells_to_refine.push_back(i);
            }
        }

        return {cells_to_refine, split_directions};
    }
};

class MyRemove : public CellsToRemove3D {
    std::pair<vector<size_t>, vector<double>> ToRemove(
        Tessellation3D const& tess,
        vector<ComputationalCell3D> const& cells,
        double time) const override {

        vector<size_t> cells_to_remove;
        vector<double> merit;  // Priority for removal

        // Example: Remove low-mass cells
        double min_mass = 1e-8;
        for(size_t i = 0; i < cells.size(); ++i) {
            double volume = tess.GetVolume(i);
            double mass = cells[i].density * volume;
            if(mass < min_mass) {
                cells_to_remove.push_back(i);
                merit.push_back(1.0 / mass);  // Lower mass = higher priority
            }
        }

        return {cells_to_remove, merit};
    }
};
```

**2. Configure AMR:**

```cpp
auto refine = std::make_shared<MyRefine>();
auto remove = std::make_shared<MyRemove>();

config.amr.enabled = true;
config.amr.frequency = 10;               // Run AMR every 10 cycles
config.amr.custom_refine = refine;
config.amr.custom_remove = remove;
```

The framework automatically:
- Calls AMR at the specified frequency
- Triggers `post_amr` hook after refinement
- Updates mesh and cells

**Note:** Both `custom_refine` and `custom_remove` are required when AMR is enabled.

---

## Creating New Problems

**Step 1: Create directory**

```bash
mkdir runs/my_simulation
```

**Step 2: Create `main.cpp`**

```cpp
#include "source/framework/problem_config_3d.hpp"
#include "source/framework/simulation_builder_3d.hpp"

using namespace rich3d;

ComputationalCell3D my_ic(const Vector3D& pos, const EquationOfState& eos) {
    // Define initial conditions
    ComputationalCell3D cell;
    cell.density = 1.0;
    cell.pressure = 1.0;
    cell.velocity = Vector3D(0, 0, 0);
    cell.internal_energy = eos.dp2e(cell.density, cell.pressure, cell.tracers);
    return cell;
}

int main() {
    Problem3DConfig config;
    config.name = "my_simulation";

    // Domain
    config.domain.lower_bound = Vector3D(-1, -1, -1);
    config.domain.upper_bound = Vector3D(1, 1, 1);

    // Mesh
    config.mesh.num_points = 10000;

    // Physics (required)
    config.physics.eos = std::make_shared<IdealGas>(5.0/3.0);

    // Initial condition (required)
    config.initial_condition = my_ic;

    // Output
    config.output.output_directory = "./";
    config.output.output_prefix = "sim";
    config.output.mode = OutputConfig::Mode::CYCLE;
    config.output.cycle_interval = 100;
    config.output.final_time = 1.0;

    // Build and run
    try {
        Simulation3DBuilder::build_and_run(config);
    }
    catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
```

**Step 3: Build and run**

```bash
./config.py --problem=my_simulation
make -j
./build/rich
```

**Example Simulations:**

- [runs/sedov3d_framework/](runs/sedov3d_framework/) - Sedov blast wave (basic hydro)
- [runs/TDE_framework/](runs/TDE_framework/) - TDE with gravity + radiation (advanced)

---

## Profiling

When built with `--profile`, gprof profiling is enabled:

```bash
# Build with profiling
./config.py --problem=sedov3d_framework --profile
make -j

# Run simulation (generates gmon.out)
./build/rich

# Visualize profile (requires gprof2dot and graphviz)
gprof ./build/rich gmon.out | gprof2dot -s -w --show-samples | dot -Tpdf -o gprof.pdf
```

---

## Framework Features

The framework provides all production-quality physics features:

- **Automatic validation:** Checks required configuration (EOS, initial condition)
- **Source terms:** Gravity, tidal forces, custom accelerations
- **Radiation diffusion:** Gray/multigroup with automatic time stepping
- **Lifecycle hooks:** Custom behavior at pre/post step, output, AMR points
- **AMR:** Adaptive mesh refinement with custom criteria
- **Domain expansion:** Automatic domain growth for expanding flows
- **Out-of-domain removal:** Automatic for FREE_FLOW boundaries
- **Automatic wiring:** No manual component setup needed
- **Sensible defaults:** Most options preconfigured
- **Clean error messages:** Early validation with helpful error reporting

---

## License

See LICENSE file in the repository.
