# Example: Sod Shock Tube

This walkthrough demonstrates how to build, run, and analyze a 1D Sod shock tube problem -- the simplest RICH simulation.

## Physics

The Sod shock tube is a 1D Riemann problem with a discontinuity at x = 0.5:

| Region | Density | Pressure | Velocity |
|--------|---------|----------|----------|
| Left (x < 0.5) | 1.0 | 1.0 | 0 |
| Right (x > 0.5) | 0.125 | 0.1 | 0 |

The solution develops a left-going rarefaction, a contact discontinuity, and a right-going shock.

## Source Code

The test source is at `regression_tests/cases/sod_1d/test.cpp`:

```cpp
#include <cmath>
#include <fstream>
#include "../../../source/newtonian/common/hllc.hpp"
#include "../../../source/newtonian/common/ideal_gas.hpp"
#include "../../../source/newtonian/one_dimensional/eos_consistent1d.hpp"
#include "../../../source/newtonian/one_dimensional/eulerian1d.hpp"
#include "../../../source/newtonian/one_dimensional/hdsim.hpp"
#include "../../../source/newtonian/one_dimensional/plm1d.hpp"
#include "../../../source/newtonian/one_dimensional/rigid_wall_1d.hpp"
#include "../../../source/newtonian/one_dimensional/spatial_distribution1d.hpp"
#include "../../../source/newtonian/one_dimensional/zero_force_1d.hpp"

class SimData {
public:
  SimData():
    eos_(1.4),               // Ideal gas, gamma = 1.4
    plm_(),                  // PLM reconstruction
    interp_(plm_, eos_),     // EOS-consistent interpolation
    rs_(),                   // HLLC Riemann solver
    vm_(),                   // Eulerian (static) mesh
    bc_(),                   // Rigid wall boundaries
    force_(),                // No external forces
    sim_(pg_,
         linspace(0.0, 1.0, 400),    // 400 uniform cells on [0,1]
         interp_,
         Step(1.0, 0.125, 0.5),      // density: step at x=0.5
         Step(1.0, 0.1, 0.5),        // pressure: step at x=0.5
         Uniform(0.0),               // x-velocity = 0
         Uniform(0.0),               // y-velocity = 0
         eos_, rs_, vm_, bc_, force_) {}

  hdsim1D& getSim() { return sim_; }
private:
  const SlabSymmetry1D pg_;
  const IdealGas eos_;
  PLM1D plm_;
  EOSConsistent interp_;
  const Hllc rs_;
  const Eulerian1D vm_;
  const RigidWall1D bc_;
  const ZeroForce1D force_;
  hdsim1D sim_;
};

int main()
{
  SimData sim_data;
  hdsim1D& sim = sim_data.getSim();

  // Run to t = 0.2
  while (sim.GetTime() < 0.2)
    sim.TimeAdvance2();

  // Write profile
  std::ofstream out("sod_profile.txt");
  for (int i = 0; i < sim.GetCellNo(); ++i) {
    auto cell = sim.GetCell(static_cast<size_t>(i));
    out << sim.GetCellCenter(static_cast<size_t>(i)) << " "
        << cell.Density << " " << cell.Pressure << "\n";
  }
  return 0;
}
```

## Build

```bash
./build_rich.sh gnuRelease --test_name=regression_tests/cases/sod_1d
```

## Run

```bash
cd regression_tests/cases/sod_1d
../../../build/gnuRelease/rich
```

This produces `sod_profile.txt` with columns: x, density, pressure.

## Run via Regression Framework

Alternatively, use the regression runner which also validates results:

```bash
./regression_tests/run_all.sh --test sod_1d --config gnuRelease --verbose
```

## Analyze Results

### Quick Plot with Python

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("regression_tests/cases/sod_1d/sod_profile.txt")
x, density, pressure = data[:, 0], data[:, 1], data[:, 2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(x, density, 'b-')
ax1.set_xlabel("x")
ax1.set_ylabel("Density")
ax1.set_title("Sod Shock Tube - Density")

ax2.plot(x, pressure, 'r-')
ax2.set_xlabel("x")
ax2.set_ylabel("Pressure")
ax2.set_title("Sod Shock Tube - Pressure")

plt.tight_layout()
plt.savefig("sod_plot.png")
plt.show()
```

### Compare with Exact Solution

The exact Riemann solution is available in `analytic/enrs.py`:

```python
import sys
sys.path.insert(0, "analytic")
from enrs import riemann_solve, RiemannProfile, Primitive

left = Primitive(1.0, 0.0, 1.0)
right = Primitive(0.125, 0.0, 0.1)
gamma = 1.4
prof = riemann_solve(left, right, gamma)

x_exact = np.linspace(0, 1, 1000)
t = 0.2
rho_exact = [prof.CalcPrim(xi / t).Density for xi in (x_exact - 0.5)]
```

### Use the Regression Plotter

```bash
python3 regression_tests/plot_results.py --all
```

This generates `regression_tests/plots/sod_1d.png` with the simulation overlaid on the exact solution.

## Expected Results

At t = 0.2, you should see:
- A rarefaction fan from x ~ 0.26 to x ~ 0.49
- A contact discontinuity at x ~ 0.68
- A shock at x ~ 0.85

The density jumps from 1.0 to about 0.426 at the contact, then drops to 0.125 at the shock. Pressure is continuous across the contact but jumps at the shock.

## Key Concepts Demonstrated

- 1D hydro simulation (`hdsim1D`)
- HLLC Riemann solver
- PLM reconstruction (second order)
- Eulerian (fixed) mesh
- Ideal gas EOS
- Text profile output
