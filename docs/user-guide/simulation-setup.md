# Simulation Setup

This guide explains how to create a `main.cpp` file for a new RICH simulation, using the 3D Sedov blast wave (`runs/sedov_3d/main.cpp`) as a walkthrough.

## Anatomy of a RICH Simulation

Every RICH simulation follows the same pattern:

1. **Initialize MPI** (if enabled)
2. **Generate mesh points** and build the Voronoi tessellation
3. **Set up the equation of state**
4. **Define initial conditions** (fill cells with primitive variables)
5. **Configure the solver** (Riemann solver, reconstruction, flux calculator)
6. **Set boundary conditions**
7. **Configure mesh motion** (Lagrangian, Eulerian, or RoundCells)
8. **Create the simulation object** (`HDSim3D`)
9. **Run the time loop** with output
10. **Finalize MPI** (if enabled)

## Step-by-Step Walkthrough

### 1. Includes

```cpp
#include "source/3D/tesselation/voronoi/Voronoi3D.hpp"
#include "source/3D/output/write3D.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/default_extensive_updater.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
```

### 2. Domain and MPI Initialization

```cpp
int main(void)
{
    size_t const Np = 1e5;  // number of mesh points
    Vector3D ll(-1, -1, -1), ur(1, 1, 1);  // domain bounds
    int rank = 0;

#ifdef RICH_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
```

The `RICH_MPI` macro is defined automatically when building with an MPI configuration. Wrap all MPI calls in `#ifdef RICH_MPI` blocks so the same code compiles in serial.

### 3. Mesh Generation

```cpp
    // Generate random points on rank 0, then distribute
    std::vector<Vector3D> points;
    if (rank == 0)
        points = RandRectangular(Np, ll, ur);

#ifdef RICH_MPI
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif

    // Smooth the mesh (Lloyd iteration)
    points = RoundGrid3D(points, ll, ur, 10);
```

Available mesh generators:

| Function | Description |
|----------|-------------|
| `RandRectangular(N, ll, ur)` | Random points in a box |
| `RandSphereR(center, Rmin, Rmax, N)` | Random points in a spherical shell |
| `RandSphereR2(center, Rmin, Rmax, N)` | Random points biased toward center |
| `linspace(xmin, xmax, N)` | Uniform 1D points |

`RoundGrid3D` smooths the point distribution using Lloyd iteration (repeated Voronoi-centroid relaxation) to produce more uniform cells. The last argument is the number of iterations.

### 4. Build the Tessellation

```cpp
    Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
    tess.BuildParallel(points);
#else
    tess.Build(points);
#endif
```

### 5. Equation of State

```cpp
    IdealGas eos(5./3.);  // gamma = 5/3
```

See [Equations of State](equations-of-state.md) for other options.

### 6. Initial Conditions

```cpp
    size_t const Nlocal = tess.GetPointNo();
    std::vector<ComputationalCell3D> cells(Nlocal);

    ComputationalCell3D inner_cell, outer_cell;
    inner_cell.velocity = Vector3D(0, 0, 0);
    inner_cell.density = 1;
    inner_cell.internal_energy = 1e5;
    inner_cell.pressure = eos.de2p(inner_cell.density, inner_cell.internal_energy,
                                    inner_cell.tracers, ComputationalCell3D::tracerNames);

    outer_cell.velocity = Vector3D(0, 0, 0);
    outer_cell.density = 1;
    outer_cell.internal_energy = 0.1;
    outer_cell.pressure = eos.de2p(outer_cell.density, outer_cell.internal_energy,
                                    outer_cell.tracers, ComputationalCell3D::tracerNames);

    for (size_t i = 0; i < Nlocal; ++i)
    {
        if (abs(tess.GetMeshPoint(i)) < 0.2)
            cells[i] = inner_cell;
        else
            cells[i] = outer_cell;
    }
```

The `ComputationalCell3D` struct holds all primitive variables:

| Field | Type | Description |
|-------|------|-------------|
| `density` | `double` | Mass density |
| `pressure` | `double` | Pressure |
| `internal_energy` | `double` | Specific internal energy |
| `velocity` | `Vector3D` | Velocity vector |
| `Temperature` | `double` | Temperature |
| `Erad` | `double` | Radiation energy density per mass |
| `Eg` | `std::array<double, N>` | Per-group radiation energy |
| `tracers` | `flat_map<string, double>` | Passive tracers |
| `stickers` | `flat_map<string, bool>` | Boolean markers |

### 7. Riemann Solver and Flux Calculator

```cpp
    Hllc3D rs;
    RigidWallGenerator3D ghost;
    LinearGauss3D interp(eos, ghost);

    // Condition-action flux: boundary faces use rigid wall, bulk faces use regular flux
    std::vector<pair<const ConditionActionFlux1::Condition3D*,
        const ConditionActionFlux1::Action3D*>> sequence;

    sequence.push_back({new IsBoundaryFace3D(), new RigidWallFlux3D(rs)});
    sequence.push_back({new IsBulkFace3D(), new RegularFlux3D(rs)});

    ConditionActionFlux1 flux(sequence, interp);
```

The condition-action pattern lets you apply different flux calculations to different faces (boundaries vs interior).

### 8. Cell and Extensive Updaters

```cpp
    // Extensive updater (maps fluxes to conserved variable changes)
    std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*,
        const ConditionExtensiveUpdater3D::Action3D*>> eu_sequence;
    ConditionExtensiveUpdater3D eu(eu_sequence);

    // Cell updater (converts conserved back to primitive)
    DefaultCellUpdater cu;
```

### 9. External Forces and Time Step

```cpp
    ZeroForce3D force;

    double const hydro_cfl = 0.3;
    double const force_cfl = 1;
    CourantFriedrichsLewy tsf(hydro_cfl, force_cfl, force);
```

### 10. Mesh Motion

```cpp
    Lagrangian3D bpm;                  // base: move with fluid
    RoundCells3D pm(bpm, eos);         // wrapper: smoothing for rounder cells
```

Options:

| Class | Motion |
|-------|--------|
| `Lagrangian3D` | Points move with the local fluid velocity |
| `Eulerian3D` | Points stay fixed (static mesh) |
| `RoundCells3D` | Lagrangian + smoothing toward cell centroids |

### 11. Create Simulation Object

```cpp
    HDSim3D sim(tess, cells, eos, pm, tsf, flux, cu, eu, force,
                std::make_pair(ComputationalCell3D::tracerNames,
                               ComputationalCell3D::stickerNames));
```

### 12. Time Loop and Output

```cpp
    double old_time = sim.getTime();
    while (sim.getTime() < 0.0075)
    {
        if (rank == 0)
            std::cout << "Iteration " << sim.getCycle()
                      << " dt " << sim.getTime() - old_time
                      << " time " << sim.getTime() << std::endl;
        old_time = sim.getTime();

        if (sim.getCycle() % 100 == 0)
            WriteSnapshot3D(sim, "sedov_" + std::to_string(sim.getCycle()) + ".h5");

        sim.timeAdvance2();  // 2nd-order time advance
    }
    WriteSnapshot3D(sim, "sedov_final.h5");

#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}
```

Time advance methods:

| Method | Order | Description |
|--------|-------|-------------|
| `timeAdvance()` | 1st | Single-step Euler |
| `timeAdvance2()` | 2nd | Two-step predictor-corrector |
| `timeAdvance3()` | 3rd | Three-step |
| `timeAdvance4()` | 4th | Four-step (most accurate, most expensive) |

## Creating a New Run

1. Create a directory under `runs/`:
   ```bash
   mkdir runs/my_simulation
   ```

2. Create `main.cpp` in the new directory, following the pattern above.

3. Build:
   ```bash
   ./build_rich.sh gnuReleaseMPI --test_name=my_simulation
   ```

4. Run:
   ```bash
   cd runs/my_simulation
   sbatch --wait --exclusive --partition=bigrun --ntasks=4 \
     --wrap "mpirun -np 4 ../../build/gnuReleaseMPI/rich"
   ```

## Restart Support

For long-running simulations, implement restart by saving and loading state:

```cpp
// Check for restart
std::string counter_name = "counter.txt";
bool const restart = std::filesystem::exists(counter_name);
int counter = 0;
if (restart)
    counter = read_int(counter_name);

// Periodic checkpoint
if (wall_time > checkpoint_interval)
{
    WriteSnapshot3D(sim, "restart.h5");
    write_int(counter_name, counter);
}
```

On restart, load the snapshot with `ReadSnapshot3D()` and reconstruct the simulation state. See `runs/BaseTDECompton/test.cpp` for a complete restart implementation.

### Wall-Time Checkpointing (HPC)

For SLURM jobs with time limits, use MPI wall time to trigger checkpoints:

```cpp
double const restart_wtime = 15000; // seconds (~4 hours)
double last_start = MPI_Wtime();

// Inside the time loop:
int restart_dump = 0;
if (rank == 0 && MPI_Wtime() - last_start > restart_wtime)
    restart_dump = 1;
MPI_Bcast(&restart_dump, 1, MPI_INT, 0, MPI_COMM_WORLD);
if (restart_dump) {
    WriteSnapshot3D(sim, "restart.h5", appendices, true);
    last_start = MPI_Wtime();
}
```

### Smart Restart Selection

Compare timestamps of the latest snapshot and restart file to always load the most recent state:

```cpp
auto last_time_restart = std::filesystem::last_write_time(restart_name);
auto last_time_snap = std::filesystem::last_write_time(snap_name);
if (last_time_snap < last_time_restart)
    snap = ReadSnapshot3D(restart_name, ...);
else
    snap = ReadSnapshot3D(snap_name, ...);
```

## Diagnostic Appendices

Custom per-cell diagnostic fields can be written alongside each HDF5 snapshot by implementing `DiagnosticAppendix3D`:

```cpp
class DissipationDiag : public DiagnosticAppendix3D {
    Dissipation const& dissipation_;
public:
    DissipationDiag(Dissipation const& d) : dissipation_(d) {}

    std::vector<double> operator()(const HDSim3D& sim) const {
        return dissipation_.CalcDissipation(sim);
    }

    std::string getName() const { return "Dissipation"; }
};
```

Register appendices and pass them to `WriteSnapshot3D`:

```cpp
vector<DiagnosticAppendix3D*> appendices;
appendices.push_back(&my_diagnostic);

WriteSnapshot3D(sim, "snap_0.h5", appendices, true);
```

### Built-in Diagnostics

The `Dissipation` class computes per-cell numerical dissipation from the Riemann solver:

```cpp
#include "source/newtonian/three_dimensional/Dissipation.hpp"

Dissipation dissipation(rs, eos);

// Before writing, populate face values:
interp(tess, sim.getCells(), 0, dissipation.face_values);
```

### Gradient Diagnostics

You can write spatial gradients of any cell variable by extracting slopes from `LinearGauss3D`:

```cpp
std::vector<Slope3D> slopes = interp.GetSlopesUnlimited();
// Access: slopes[i].xderivative.density, .pressure, .internal_energy, .velocity
```

See `runs/BaseTDECompton/test.cpp` for `GradDiag` which writes all 9 gradient components plus velocity divergence.

## Runtime Configuration via Text Files

Instead of hardcoding parameters, read them from text files at runtime. This allows the same binary to run different configurations without recompilation:

```cpp
#include "source/misc/simple_io.hpp"

double R    = read_number("Rstar.txt");
double M    = read_number("Mstar.txt");
double Mbh  = read_number("Mbh.txt");
double beta = read_number("beta.txt");
int    n    = read_int("n.txt");
```

This pattern is used by all TDE simulations and is recommended for any simulation with tunable parameters.

## Further Examples

- **Sedov blast wave**: `runs/sedov_3d/main.cpp` -- simple setup with ideal gas
- **TDE with multigroup radiation**: `runs/BaseTDECompton/test.cpp` -- production TDE with Compton, AMR, gravity, and restarts (see [TDE example](../examples/tde-simulation.md))
