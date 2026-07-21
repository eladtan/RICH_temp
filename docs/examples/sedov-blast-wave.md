# Example: 3D Sedov Blast Wave

This walkthrough demonstrates a 3D Sedov-Taylor blast wave on a moving Voronoi mesh with MPI parallelism -- a representative RICH production simulation.

## Physics

The Sedov-Taylor blast wave is a self-similar solution for a point explosion in a uniform medium. A concentrated energy deposit drives a spherical shock wave outward. The solution is characterized by a strong density jump at the shock front and a rarefaction behind it.

## Source Code

The run source is at `runs/sedov_3d/main.cpp`. Key setup steps:

### Domain and Mesh

```cpp
size_t const Np = 1e5;
Vector3D ll(-1, -1, -1), ur(1, 1, 1);

// Generate random points on rank 0, distribute to all ranks
std::vector<Vector3D> points;
if (rank == 0)
    points = RandRectangular(Np, ll, ur);
#ifdef RICH_MPI
points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif

// Smooth with 10 Lloyd iterations
points = RoundGrid3D(points, ll, ur, 10);

// Build Voronoi tessellation
Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
tess.BuildParallel(points);
#else
tess.Build(points);
#endif
```

### Initial Conditions

```cpp
IdealGas eos(5./3.);

ComputationalCell3D inner_cell, outer_cell;
inner_cell.density = 1;
inner_cell.internal_energy = 1e5;    // high energy in center
inner_cell.pressure = eos.de2p(inner_cell.density, inner_cell.internal_energy, ...);

outer_cell.density = 1;
outer_cell.internal_energy = 0.1;    // low energy outside
outer_cell.pressure = eos.de2p(outer_cell.density, outer_cell.internal_energy, ...);

for (size_t i = 0; i < Nlocal; ++i) {
    if (abs(tess.GetMeshPoint(i)) < 0.2)  // inner sphere
        cells[i] = inner_cell;
    else
        cells[i] = outer_cell;
}
```

### Solver Configuration

```cpp
Hllc3D rs;                                    // HLLC Riemann solver
RigidWallGenerator3D ghost;                   // Rigid wall BCs
LinearGauss3D interp(eos, ghost);             // Second-order reconstruction
ConditionActionFlux1 flux(sequence, interp);  // Condition-action flux

Lagrangian3D bpm;                             // Move with fluid
RoundCells3D pm(bpm, eos);                    // + cell rounding

CourantFriedrichsLewy tsf(0.3, 1, force);    // CFL = 0.3
```

### Time Loop

```cpp
while (sim.getTime() < 0.0075) {
    if (sim.getCycle() % 100 == 0)
        WriteSnapshot3D(sim, "sedov_" + std::to_string(sim.getCycle()) + ".h5");
    sim.timeAdvance2();  // 2nd-order time advance
}
WriteSnapshot3D(sim, "sedov_final.h5");
```

## Build

### Serial

```bash
./build_rich.sh gnuRelease --test_name=sedov_3d
```

### MPI

```bash
ml openmpi/4.1.6/gcc/12.3.0  # load MPI module
./build_rich.sh gnuReleaseMPI --test_name=sedov_3d
```

## Run

### Serial

```bash
cd runs/sedov_3d
../../build/gnuRelease/rich
```

### MPI (via SLURM)

```bash
cd runs/sedov_3d
sbatch --wait --exclusive --partition=bigrun --ntasks=128 \
  --output=output_%j --error=error_%j \
  --wrap "mpirun -np 128 ../../build/gnuReleaseMPI/rich"
```

### Via Regression Framework (smaller version)

```bash
./regression_tests/run_all.sh --test sedov_3d_mpi --verbose --keep-artifacts
```

## Output Files

The simulation produces HDF5 snapshots:

```
runs/sedov_3d/
├── sedov_0.h5
├── sedov_100.h5
├── sedov_200.h5
├── ...
└── sedov_final.h5
```

## Analyze Results

### Read HDF5 in Python

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("sedov_final.h5", "r") as f:
    x = f["X"][:]
    y = f["Y"][:]
    z = f["Z"][:]
    density = f["Density"][:]

r = np.sqrt(x**2 + y**2 + z**2)

plt.scatter(r, density, s=0.1, alpha=0.3)
plt.xlabel("Radius")
plt.ylabel("Density")
plt.title("Sedov Blast Wave - Density vs Radius")
plt.savefig("sedov_profile.png")
plt.show()
```

### Compare with Analytical Solution

```python
import sys
sys.path.insert(0, "analytic")
from sedov_taylor import SedovTaylor

# Compute exact Sedov-Taylor profile
sedov = SedovTaylor(gamma=5./3., geometry=3)  # spherical
r_exact = np.linspace(0.01, 0.8, 500)
# ... compute density, pressure, velocity at r_exact
```

### Use the Regression Plotter

```bash
python3 regression_tests/plot_results.py --all
```

## Expected Results

The blast wave should show:
- A sharp density jump at the shock radius (r ~ 0.5 at t = 0.0075)
- Post-shock density about 4x the pre-shock value (for gamma = 5/3)
- Self-similar radial profiles matching the Sedov-Taylor ODE solution

## Key Concepts Demonstrated

- 3D Voronoi moving mesh
- MPI parallelism (domain decomposition, parallel tessellation build)
- RoundGrid3D mesh smoothing
- Lagrangian + RoundCells mesh motion
- HLLC Riemann solver with LinearGauss reconstruction
- HDF5 snapshot output
- Rigid wall boundary conditions
