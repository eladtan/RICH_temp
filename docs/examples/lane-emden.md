# Example: Lane-Emden Star with Self-Gravity

This walkthrough demonstrates a 3D self-gravitating polytropic star in hydrostatic equilibrium, testing the coupling between hydrodynamics and the gravity tree solver.

## Physics

The Lane-Emden equation describes the hydrostatic equilibrium of a self-gravitating polytropic gas sphere. For polytropic index n = 3/2 (appropriate for a fully convective star), the density profile is:

```
rho(r) = rho_c * theta(xi)^n
```

where theta(xi) is the solution to the Lane-Emden equation:

```
(1/xi^2) d/dxi (xi^2 d(theta)/dxi) = -theta^n
```

with boundary conditions theta(0) = 1, theta'(0) = 0.

The star should remain in equilibrium: if the gravity solver and pressure gradients are balanced correctly, the density profile stays constant over time.

## Configuration

From `regression_tests/cases/lane_self_gravity/test.cpp` (conceptual overview):

| Parameter | Value |
|-----------|-------|
| Polytrope index | n = 3/2 |
| EOS | Ideal gas, gamma = 5/3 |
| Stellar radius | R = 7e10 cm |
| Stellar mass | M = 2e33 g |
| Lane-Emden tables | `data/xsi32.txt`, `data/theta32.txt` |
| Gravity | DistributedGravityTree |
| Mesh motion | Lagrangian + RoundCells |
| MPI | 64 tasks |

### Initial Conditions

The initial density and pressure are set from the Lane-Emden tables:

```cpp
// Read Lane-Emden solution
auto xsi = read_vector("data/xsi32.txt");
auto theta = read_vector("data/theta32.txt");

// Map to physical coordinates
for (size_t i = 0; i < Nlocal; ++i) {
    double r = abs(tess.GetMeshPoint(i));
    double xi = r / R * xsi.back();
    double th = interpolate(xsi, theta, xi);
    cells[i].density = rho_c * pow(th, 1.5);
    cells[i].pressure = K * pow(cells[i].density, 5./3.);
}
```

## Build

```bash
ml openmpi/4.1.6/gcc/12.3.0
./build_rich.sh gnuReleaseMPI --test_name=regression_tests/cases/lane_self_gravity
```

## Run

### Via Regression Framework (recommended)

```bash
./regression_tests/run_all.sh --test lane_self_gravity --verbose --keep-artifacts
```

### Manual SLURM Submission

```bash
cd regression_tests/cases/lane_self_gravity
sbatch --wait --exclusive --partition=bigrun --ntasks=64 \
  --wrap "mpirun -np 64 ../../../build/gnuReleaseMPI/rich"
```

## Output

- `lane_gravity_metrics.txt` -- fields: `final_metric` (mean density deviation), `pass` (0 or 1)
- `lane_profile.txt` -- radial density profile for plotting

## Analyze Results

### Plot Density Profile

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("regression_tests/cases/lane_self_gravity/lane_profile.txt")
r, density = data[:, 0], data[:, 1]

# Load initial Lane-Emden profile for comparison
xsi = np.loadtxt("data/xsi32.txt")
theta = np.loadtxt("data/theta32.txt")

plt.figure(figsize=(8, 5))
plt.plot(r, density, 'b.', markersize=1, label="Simulation")
plt.plot(xsi * R / xsi[-1], rho_c * theta**1.5, 'r-', label="Lane-Emden")
plt.xlabel("Radius (cm)")
plt.ylabel("Density (g/cm³)")
plt.title("Lane-Emden Star - Density Profile")
plt.legend()
plt.savefig("lane_profile.png")
plt.show()
```

### Use the Regression Plotter

```bash
python3 regression_tests/plot_results.py --all
```

## Expected Results

- The density profile should remain very close to the initial Lane-Emden solution
- The `final_metric` (mean relative density deviation) should be less than 4%
- Any significant deviation indicates an imbalance between gravity and pressure

## Pass Criteria

| Metric | Threshold |
|--------|-----------|
| \|final_metric\| | < 0.04 |

## Key Concepts Demonstrated

- Self-gravity via `DistributedGravityTree`
- `ConservativeForce3D` for energy-consistent gravitational force
- Hydrostatic equilibrium verification
- Spherical mesh generation (`RandSphereR`)
- MPI parallelism for gravity + hydro
- Tabular initial conditions from data files

## References

- Lane, J. H. (1870). "On the theoretical temperature of the Sun." *Am. J. Sci.* 50, 57-74.
- Emden, R. (1907). *Gaskugeln*. Teubner.
- Chandrasekhar, S. (1939). *An Introduction to the Study of Stellar Structure*. University of Chicago Press.
