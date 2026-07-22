# Example: Marshak Wave

This walkthrough demonstrates a radiation transport problem: the non-equilibrium Marshak wave with grey diffusion.

## Physics

A Marshak wave is driven by a time-dependent temperature source (bath) at the left boundary of a cold medium. Radiation diffuses into the medium, heating the gas. The problem tests the coupling between radiation transport and material energy.

The benchmark problems use self-similar analytical solutions for validation.

### Problem Variants

| Problem | Description |
|---------|-------------|
| 1 | Non-equilibrium diffusion, uniform density, T^{-3} opacity |
| 2 | Equilibrium limit (kappa_P = kappa_R), uniform density |
| 3 | Non-uniform density rho(x) = x^{20/19}, density-dependent opacity |
| 4 | Divergent density rho(x) = x^{-40/139}, stretched grid |

## Configuration

From `regression_tests/cases/marshak_wave_1_diffusion/test.cpp` (conceptual overview):

- **Domain:** 1D slab, 512 cells
- **Radiation:** Grey diffusion, no flux limiter
- **Opacity:** Temperature-dependent Rosseland and Planck opacities
- **Boundary:** Time-dependent temperature bath at x = 0
- **End condition:** Run to a specified final time

The bath temperature follows a self-similar scaling:

```
T_bath(t) = C * (t / ns)^{1/3} keV
```

where C depends on the problem variant.

## Build and Run

### Build

```bash
./build_rich.sh gnuRelease --test_name=regression_tests/cases/marshak_wave_1_diffusion
```

### Run

```bash
cd regression_tests/cases/marshak_wave_1_diffusion
../../../build/gnuRelease/rich
```

### Via Regression Framework (recommended)

```bash
# Run one Marshak test
./regression_tests/run_all.sh --test marshak_wave_1_diffusion --config gnuRelease --verbose

# Run all four variants
./regression_tests/run_all.sh --mode serial --verbose
```

## Output

The simulation produces:

- `marshak_profile.txt` -- columns: x position, gas temperature (Tgas), radiation temperature (Trad)
- `problem_number.txt` -- the problem variant number (1-4)

## Analyze Results

### Quick Plot

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("regression_tests/cases/marshak_wave_1_diffusion/marshak_profile.txt")
x, Tgas, Trad = data[:, 0], data[:, 1], data[:, 2]

plt.figure(figsize=(8, 5))
plt.plot(x, Tgas, 'b-', label="T_gas")
plt.plot(x, Trad, 'r--', label="T_rad")
plt.xlabel("x")
plt.ylabel("Temperature (keV)")
plt.title("Marshak Wave Problem 1")
plt.legend()
plt.savefig("marshak_wave.png")
plt.show()
```

### Compare with Analytical Solution

The regression checker `regression_tests/lib/check_marshak_wave.py` computes the self-similar analytical solution:

- **Problems 1-2:** ODE shooting method from Krief & McClarren (2024)
- **Problems 3-4:** Fitted profiles from Derei et al. (2024), Table III

### Use the Regression Plotter

```bash
python3 regression_tests/plot_results.py --all
```

This generates plots in `regression_tests/plots/` showing Tgas and Trad versus the analytical solution.

## Expected Results

### Problem 1 (non-equilibrium)

- Trad propagates ahead of Tgas (radiation front leads the material heating front)
- Both profiles follow a self-similar shape
- The gap between Tgas and Trad demonstrates non-equilibrium effects

### Problem 2 (equilibrium limit)

- Tgas and Trad are nearly identical (kappa_P = kappa_R forces equilibrium)
- Single wave front

### Problems 3-4 (non-uniform density)

- The density profile affects wave propagation speed
- Problem 4 with divergent density (rho -> infinity as x -> 0) uses a stretched grid

## Pass Criteria

All four problems require relative L1 error below 1% for both Tgas and Trad.

## Key Concepts Demonstrated

- Radiation diffusion (grey, no flux limiter)
- Temperature-dependent opacities
- Time-dependent boundary conditions
- Radiation-matter energy coupling
- Self-similar analytical validation

## References

- Marshak, R. E. (1958). "Effect of radiation on shock wave behavior." *Phys. Fluids* 1, 24.
- Krief, M. & McClarren, R. G. (2024). Self-similar Marshak wave solutions.
- Derei, A. et al. (2024). Non-uniform density Marshak wave benchmarks.
- Giron, J. F. et al. (2026). arXiv:2601.05120.
