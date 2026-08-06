# Regression Test Catalog

This document describes all 29 regression tests in the RICH suite. Each entry covers the physics being tested, the simulation configuration, validation methodology, pass/fail criteria, and references.

---

## 1. sod_1d -- 1D Sod Shock Tube

**Tags:** `serial`

### Physics

The Sod shock tube is a classical 1D Riemann problem that produces three distinct wave structures: a left-going rarefaction fan, a contact discontinuity, and a right-going shock wave. It tests the hydrodynamic solver's ability to capture shocks and contact discontinuities.

**Governing equations:** 1D Euler equations for an ideal gas.

### Configuration

| Parameter | Value |
|-----------|-------|
| Domain | [0, 1] |
| Cells | 400 (uniform) |
| EOS | Ideal gas, gamma = 1.4 |
| Left state | rho = 1.0, P = 1.0, v = 0 |
| Right state | rho = 0.125, P = 0.1, v = 0 |
| Discontinuity | x = 0.5 |
| End time | t = 0.2 |
| Solver | HLLC, PLM reconstruction |
| Cell updater | EOSConsistent |

**Source:** `regression_tests/cases/sod_1d/test.cpp`

### Output

`sod_profile.txt` -- columns: x position, density, pressure

### Validation

Compared against the exact Riemann solution computed by `analytic/enrs.py` (exact non-linear Riemann solver). The Python checker `regression_tests/lib/check_sod_profile.py` computes a goodness-of-fit (GOF) metric for density and pressure profiles.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| Density GOF | <= 0.02 | `SOD_MAX_DENSITY_GOF` |
| Pressure GOF | <= 0.02 | `SOD_MAX_PRESSURE_GOF` |

### References

- Sod, G. A. (1978). "A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws." *J. Comput. Phys.* 27, 1-31.
- Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed. Springer.

---

## 2. sedov_3d_mpi -- 3D Sedov-Taylor Blast Wave

**Tags:** `mpi`

### Physics

The Sedov-Taylor blast wave is a self-similar solution for a strong point explosion in a uniform medium. It tests the 3D hydrodynamic solver on a moving Voronoi mesh with MPI parallelism, validating shock propagation, density jump, and post-shock profiles.

**Governing equations:** 3D Euler equations for an ideal gas with strong shock.

### Configuration

| Parameter | Value |
|-----------|-------|
| Domain | [-1, 1]^3 |
| Points | ~5 million (MPI), rounded via `RoundGrid3D` |
| EOS | Ideal gas, gamma = 5/3 |
| Inner region | r < 0.2: rho = 1, high internal energy |
| Outer region | r >= 0.2: rho = 1, low internal energy |
| Solver | HLLC, LinearGauss3D |
| Mesh motion | Lagrangian + RoundCells |
| Boundary | Rigid wall |
| SLURM | 128 tasks, `bigrun` partition, exclusive |

**Source:** `regression_tests/cases/sedov_3d_mpi/test.cpp`

### Output

`sedov_profile.txt` -- columns: radius, density, pressure, velocity

### Validation

Compared against the Sedov-Taylor self-similar ODE solution computed by `analytic/sedov_taylor.py`. The Python checker `regression_tests/lib/check_sedov_exact.py` computes relative L1 errors for density, pressure, and velocity.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| Density relative L1 | <= 0.30 | `SEDOV_MAX_DENSITY_REL_L1` |
| Pressure relative L1 | <= 0.30 | `SEDOV_MAX_PRESSURE_REL_L1` |
| Velocity relative L1 | <= 0.30 | `SEDOV_MAX_VELOCITY_REL_L1` |

### References

- Sedov, L. I. (1959). *Similarity and Dimensional Methods in Mechanics*. Academic Press.
- Taylor, G. I. (1950). "The formation of a blast wave by a very intense explosion." *Proc. R. Soc. Lond. A* 201, 159-174.

---

## 3. till_compton -- Compton Equilibration

**Tags:** `serial`

### Physics

Tests the multigroup radiation diffusion solver with Compton scattering. The initial state has the gas and radiation at different temperatures (Tgas = 1 keV, Trad = 10 keV), and the system evolves toward thermal equilibrium via Compton energy exchange. This is "Case 3" from Till et al.

**Governing equations:** Radiation-matter energy coupling with Compton scattering in the multigroup diffusion framework.

### Configuration

| Parameter | Value |
|-----------|-------|
| Energy groups | 32 |
| Initial Tgas | 1 keV |
| Initial Trad | 10 keV |
| Diffusion | Multigroup with Compton |
| Build args | `--energy_groups_num=32` |

**Source:** `regression_tests/cases/till_compton/test.cpp`

**Reference data:** `regression_tests/cases/till_compton/data/in_fbc_reference.txt` (McGraw et al. 2023)

### Output

- `time.txt` -- simulation time values
- `Tgas.txt` -- gas temperature history
- `Trad.txt` -- radiation temperature history
- `Etotal.txt` -- total energy history

### Validation

The bash checker `check_till_case` in `regression_checks.sh` verifies:

1. No NaN or Inf values in output files
2. Final gas and radiation temperatures have equilibrated
3. Total energy is conserved

### Pass Criteria

| Metric | Threshold |
|--------|-----------|
| Final \|Tgas - Trad\| / max(Tgas, Trad) | < 1% |
| Energy conservation \|E_final - E_initial\| / E_initial | < 1e-8 |
| No NaN/Inf in outputs | Required |

### References

- Till, C. et al. (2019). "Compton scattering in particle transport." *JQSRT* 235, 106594.
- McGraw, K. et al. (2023). Compton equilibration reference data.

---

## 4. amr_random -- AMR Refine/Remove Conservation

**Tags:** `serial`, `mpi`

### Physics

Tests adaptive mesh refinement (AMR) on a uniform gas to verify that refinement (cell splitting) and coarsening (cell removal) conserve extensive quantities (mass, momentum, energy) to machine precision.

### Configuration

| Parameter | Value |
|-----------|-------|
| Initial state | Uniform gas: rho = 1, internal energy = 2.5 |
| Operations | Random refine and remove cycles |
| SLURM (MPI) | 64 tasks, `bigrun` partition |

**Source:** `regression_tests/cases/amr_random/test.cpp`

### Output

`amr_random_metrics.txt` -- fields: `mode` (serial/mpi), `max_drift`, `threshold`, `pass`

### Validation

The bash checker `check_amr_random_case` verifies that the maximum drift (change in conserved quantities) stays below the mode-dependent threshold.

### Pass Criteria

| Mode | max_drift Threshold | Environment Variable |
|------|---------------------|---------------------|
| Serial | <= 1e-8 | `AMR_RANDOM_MAX_DRIFT_SERIAL` |
| MPI | <= 1e-6 | `AMR_RANDOM_MAX_DRIFT_MPI` |
| `pass` field | Must be `1` | -- |

---

## 5. amr_distributed_clip -- Distributed AMR clipCells Conservation

**Tags:** `mpi`

### Physics

Tests the distributed `clipCells` load-balancing feature, which offloads AMR clip work from busy ranks to idle ranks. A highly imbalanced scenario is constructed: only 4 ranks refine cells and 5 ranks remove cells, while the remaining 55 ranks are idle. The test verifies that total mass and total energy are conserved after the AMR step.

### Configuration

| Parameter | Value |
|-----------|-------|
| Initial state | Uniform gas: rho = 1, internal energy = 2.5 |
| Total points | 2 x 10^5 |
| Refine ranks | 4 ranks, ~500 cells each |
| Remove ranks | 5 ranks, ~500 cells each |
| SLURM (MPI) | 64 tasks, `bigrun` partition |

**Source:** `regression_tests/cases/amr_distributed_clip/test.cpp`

### Output

`amr_distributed_clip_metrics.txt` -- fields: `mass_before`, `mass_after`, `energy_before`, `energy_after`, `mass_reldiff`, `energy_reldiff`, `threshold`, `pass`

### Validation

The bash checker `check_amr_distributed_clip_case` verifies that the relative change in total mass and total energy stays below the threshold.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| `mass_reldiff` | <= 1e-6 | `AMR_DISTRIBUTED_CLIP_THRESHOLD` |
| `energy_reldiff` | <= 1e-6 | `AMR_DISTRIBUTED_CLIP_THRESHOLD` |
| `pass` field | Must be `1` | -- |

---

## 6. voronoi_volume -- Voronoi Volume Sum Accuracy

**Tags:** `serial`, `mpi`

### Physics

Tests the accuracy of the Voronoi tessellation by verifying that the sum of all cell volumes equals the total domain volume. This is a fundamental geometric consistency check.

### Configuration

| Parameter | Value |
|-----------|-------|
| Domain | Unit cube [0, 1]^3 |
| Points | Random distribution |
| SLURM (MPI) | 64 tasks, `bigrun` partition |

**Source:** `regression_tests/cases/voronoi_volume/test.cpp`

### Output

`voronoi_volume_metrics.txt` -- fields: `rel_error`, `pass`

### Validation

The bash checker `check_voronoi_volume_case` verifies the relative error between the sum of cell volumes and the exact box volume.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| Relative error | < 1e-10 | `VORONOI_VOLUME_MAX_REL_ERROR` |
| `pass` field | Must be `1` | -- |

---

## 7. lane_self_gravity -- Lane-Emden with Self-Gravity

**Tags:** `mpi`

### Physics

Tests hydrostatic equilibrium of a polytropic star (Lane-Emden solution with index n = 3/2) under self-gravity computed by the distributed gravity tree. The star should remain in equilibrium; significant deviations indicate errors in the gravity solver or pressure gradient balance.

**Governing equations:** Euler equations with self-gravity source term.

### Configuration

| Parameter | Value |
|-----------|-------|
| Polytrope index | n = 3/2 |
| Lane-Emden tables | `data/xsi32.txt`, `data/theta32.txt` |
| Domain | Sphere of radius R = 7e10 cm |
| EOS | Ideal gas, gamma = 5/3 |
| Solver | HLLC, LinearGauss3D |
| Mesh motion | Lagrangian + RoundCells |
| Gravity | DistributedGravityTree |
| SLURM | 512 tasks, `bigrun` partition |

**Source:** `regression_tests/cases/lane_self_gravity/test.cpp`

### Output

- `lane_gravity_metrics.txt` -- fields: `final_metric`, `pass`
- `lane_profile.txt` -- radial density profile

### Validation

The bash checker `check_lane_self_gravity_case` verifies that the mean density deviation from the initial profile stays small.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| \|final_metric\| | < 4e-2 | `LANE_GRAVITY_MAX_METRIC` |
| `pass` field | Must be `1` | -- |

### References

- Lane, J. H. (1870). "On the theoretical temperature of the Sun." *Am. J. Sci.* 50, 57-74.
- Emden, R. (1907). *Gaskugeln*. Teubner.

---

## 7b. lane_self_gravity_fmm -- Lane-Emden with FMM Self-Gravity

**Tags:** `mpi`, `manual`, `benchmark`

### Physics

Same Lane-Emden hydrostatic equilibrium benchmark as `lane_self_gravity`, but gravity is computed with the distributed FMM solver (`FastMultipoleAcceleration3D`) at expansion order $P = 3$ and $\theta = 0.9$ instead of the quadrupole Barnes-Hut tree.

### Configuration

| Parameter | Value |
|-----------|-------|
| Polytrope index | n = 3/2 |
| Lane-Emden tables | `data/xsi32.txt`, `data/theta32.txt` |
| Domain | Sphere of radius R = 7e10 cm |
| EOS | Ideal gas, gamma = 5/3 |
| Solver | HLLC, LinearGauss3D |
| Mesh motion | Lagrangian + RoundCells |
| Gravity | Distributed FMM, P=3, theta=0.9 |
| SLURM | 512 tasks, `bigrun` partition |

**Source:** `regression_tests/cases/lane_self_gravity_fmm/test.cpp`

### Output

- `lane_gravity_metrics.txt` -- fields: `final_metric`, `pass`
- `lane_profile.txt` -- radial density profile

### Validation

The bash checker `check_lane_self_gravity_fmm_case` verifies that the mean density deviation from the initial profile stays small.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| \|final_metric\| | < 4e-2 | `LANE_GRAVITY_FMM_MAX_METRIC` (falls back to `LANE_GRAVITY_MAX_METRIC`) |
| `pass` field | Must be `1` | -- |

---

## 8. mach2_diffusion -- Mach 2 Radiative Shock (Grey Diffusion)

**Tags:** `mpi`

### Physics

A Mach 2 radiative shock with grey (single-group) flux-limited diffusion. The shock develops a precursor region heated by radiation ahead of the shock front. This tests the coupling between hydrodynamics and radiation transport.

**Governing equations:** Euler equations coupled with grey radiation diffusion.

### Configuration

| Parameter | Value |
|-----------|-------|
| Domain | 1D slab geometry |
| Cells | 1024 |
| Mach number | 2.0 |
| Diffusion | Grey, flux-limited |
| Mesh motion | Eulerian (fixed) |
| Boundary | Inflow |
| End time | t = 0.01 |
| SLURM | 8 tasks, `bigrun` partition |

**Source:** `regression_tests/cases/mach2_diffusion/test.cpp`

### Output

`mach2_profile.txt` -- columns: x, density, Tgas, Trad

### Validation

Compared against the NLTE analytical radiative shock solution from `analysis_files/radiative_shock/nlte_radiative_shock.py`. The Python checker `regression_tests/lib/check_mach2_profile.py` computes relative L1 errors.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| Density relative L1 | <= 0.025 | `MACH2_MAX_DENSITY_REL_L1` |
| Temperature relative L1 | <= 0.025 | `MACH2_MAX_TEMPERATURE_REL_L1` |

### References

- Lowrie, R. B. & Rauenzahn, R. M. (2007). "Radiative shock solutions in the equilibrium diffusion limit." *Shock Waves* 16, 445-453.
- Lowrie, R. B. & Edwards, J. D. (2008). "Radiative shock solutions with grey nonequilibrium diffusion." *Shock Waves* 18, 129-143.

---

## 9. mach2_multigroup -- Mach 2 Radiative Shock (32-Group Diffusion)

**Tags:** `mpi`

### Physics

Same Mach 2 radiative shock as `mach2_diffusion`, but with 32-group multigroup radiation diffusion instead of grey. This validates the multigroup solver produces consistent results with the grey solution.

### Configuration

Same as `mach2_diffusion` plus:

| Parameter | Value |
|-----------|-------|
| Energy groups | 32 |
| Build args | `--energy_groups_num=32` |

**Source:** `regression_tests/cases/mach2_multigroup/test.cpp`

### Output

- `mach2_profile.txt` -- columns: x, density, Tgas, Trad
- `mach2_spectrum.txt` -- spectral data

### Validation

Same as `mach2_diffusion`.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| Density relative L1 | <= 0.025 | `MACH2_MAX_DENSITY_REL_L1` |
| Temperature relative L1 | <= 0.025 | `MACH2_MAX_TEMPERATURE_REL_L1` |

---

## 10-13. marshak_wave_1_diffusion through marshak_wave_4_diffusion -- Marshak Wave Benchmarks

**Tags:** `serial`

### Physics

Non-equilibrium nonlinear Marshak waves driven by a time-dependent temperature bath at the left boundary. These test radiation diffusion with temperature-dependent and density-dependent opacities in various regimes.

**Governing equations:** Radiation diffusion (P1 or grey) with material energy coupling.

### Problem Variants

| Problem | Opacity Model | Density | Bath Temperature | Reference |
|---------|--------------|---------|------------------|-----------|
| 1 | kappa_R = 100 * (T/keV)^{-3}, kappa_P = 0.001 * kappa_R | Uniform | T(t) = 1.008038 * (t/ns)^{1/3} keV | Krief & McClarren (2024) |
| 2 | kappa_P = kappa_R (equilibrium limit) | Uniform | T(t) = 1.014565 * (t/ns)^{1/3} keV | Krief & McClarren (2024) |
| 3 | Density-dependent opacities | rho(x) = x^{20/19} | Fitted | Derei et al. (2024), Table III |
| 4 | Density-dependent opacities | rho(x) = x^{-40/139}, stretched grid | Fitted | Derei et al. (2024), Table III |

### Configuration (common)

| Parameter | Value |
|-----------|-------|
| Domain | 1D, [0, L] |
| Cells | 512 |
| Diffusion | Grey, no flux limiter |

**Source:** `regression_tests/cases/marshak_wave_N/test.cpp` (N = 1, 2, 3, 4)

### Output

- `marshak_profile.txt` -- columns: x, Tgas, Trad
- `problem_number.txt` -- problem variant (1-4)

### Validation

The Python checker `regression_tests/lib/check_marshak_wave.py` computes:
- For Problems 1-2: self-similar ODE shooting solution (Krief & McClarren)
- For Problems 3-4: fitted profiles from Derei et al. Table III

Relative L1 errors are computed for both Tgas and Trad.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| Tgas relative L1 | <= 1e-2 | `MARSHAK_MAX_TGAS_REL_L1` |
| Trad relative L1 | <= 1e-2 | `MARSHAK_MAX_TRAD_REL_L1` |

### References

- Marshak, R. E. (1958). "Effect of radiation on shock wave behavior." *Phys. Fluids* 1, 24.
- Krief, M. & McClarren, R. G. (2024). Self-similar Marshak wave solutions.
- Derei, A. et al. (2024). Non-uniform density Marshak wave benchmarks.
- Giron, J. F. et al. (2026). arXiv:2601.05120.

---

## 14. gresho_euler -- Gresho Vortex (Eulerian Mesh)

**Tags:** `serial`

### Physics

The Gresho vortex is an exact stationary solution to the Euler equations in 2D: an azimuthal velocity profile in pressure equilibrium that should remain unchanged over time. Running on a fixed (Eulerian) mesh tests the solver's ability to maintain this equilibrium against numerical dissipation.

**Governing equations:** 2D Euler equations (simulated in 3D with 1 cell in z).

### Configuration

| Parameter | Value |
|-----------|-------|
| Domain | 3D slab with 1 cell in z |
| Grid | 50 x 50 x 1 Cartesian |
| Mesh motion | Eulerian (fixed) |
| End time | t = 5 |

**Source:** `regression_tests/cases/gresho_euler/test.cpp`

### Output

- `gresho_profile.txt` -- columns: x, y, volume, pressure, vx, vy
- `test_type.txt` -- contains "euler"

### Validation

The Python checker `regression_tests/lib/check_gresho_profile.py` computes the volume-weighted relative L1 error of the azimuthal velocity profile compared to the initial condition.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| v_theta relative L1 | <= 0.1 | `GRESHO_EULER_MAX_L1` |

### References

- Gresho, P. M. & Chan, S. T. (1990). "On the theory of semi-implicit projection methods..." *Int. J. Numer. Methods Fluids* 11, 621-659.
- Liska, R. & Wendroff, B. (2003). "Comparison of several difference schemes on 1D and 2D test problems for the Euler equations." *SIAM J. Sci. Comput.* 25, 995-1017.

---

## 15. gresho_lagrangian -- Gresho Vortex (Lagrangian Mesh)

**Tags:** `mpi`

### Physics

Same Gresho vortex as `gresho_euler`, but run on a moving (Lagrangian + RoundCells) mesh. The Lagrangian approach should better preserve the vortex structure, hence the tighter tolerance.

### Configuration

Same as `gresho_euler` except:

| Parameter | Value |
|-----------|-------|
| Mesh motion | Lagrangian + RoundCells |
| Run mode | SLURM, 8 tasks |

**Source:** `regression_tests/cases/gresho_lagrangian/test.cpp`

### Output

- `gresho_profile.txt` -- same format as euler
- `test_type.txt` -- contains "lagrangian"

### Validation

Same as `gresho_euler`.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| v_theta relative L1 | <= 0.05 | `GRESHO_LAGRANGIAN_MAX_L1` |

---

## 16. desmore2012_mc -- Densmore 2012 Heterogeneous Step-Opacity (MC IMC, MPI, no RW)

**Tags:** `mpi`

### Physics
Densmore et al. (2012) heterogeneous step-opacity slab problem: optically thin (sigma_0=10) for x<2 cm, optically thick (sigma_0=1000) for x>=2 cm. Monte Carlo Implicit Monte Carlo (IMC) transport with 30-group frequency-dependent opacities. Random walk **enabled**. No hydrodynamics.

### Configuration
- **Mesh:** 256 Eulerian cells, x in [0, 3] cm
- **EOS:** Ideal gas, gamma=1.4, Cv=1e15/T_keV
- **Radiation:** 30-group IMC, Planck boundary at 1 keV (left), reflective (right)
- **Runtime:** t_final=1 ns, dt=5e-12 s (200 steps)
- **Execution:** MPI, 32 ranks, SLURM (bigrun, exclusive)
- **Build flags:** `--energy_groups_num=30`

### Output
- `desmore2012_mc_profile.txt` -- (x, T_K) profile

### Pass Criteria
| Metric | Threshold | Override variable |
|--------|-----------|-------------------|
| Tgas L1 error (keV) | <= 0.05 | `DESMORE2012_MC_MAX_TGAS_L1` |

---

## 17. desmore2012_mc_serial -- Densmore 2012 Heterogeneous Step-Opacity (Serial MC, RW)

**Tags:** `serial`

### Physics
Same problem as `desmore2012_mc` but run serially with random walk enabled. Validates the serial (non-MPI) execution path and random walk acceleration of the MC IMC solver.

### Configuration
- **Mesh:** 256 Eulerian cells, x in [0, 3] cm
- **Radiation:** 30-group IMC with random walk, same opacities and boundary conditions
- **Execution:** Serial (single core)
- **Build flags:** `--energy_groups_num=30`

### Output
- `desmore2012_mc_serial_profile.txt` -- (x, T_K) profile

### Pass Criteria
| Metric | Threshold | Override variable |
|--------|-----------|-------------------|
| Tgas L1 error (keV) | <= 0.05 | `DESMORE2012_MC_SERIAL_MAX_TGAS_L1` |

---

## 19. moving_slab_mc_32 -- Moving Slab MC Benchmark (32-Group Collapsed, Original Vacuum)

**Tags:** `serial`

### Physics

32-group variant of the frequency-dependent moving slab benchmark from McClarren & Gentile (2021), original vacuum variant. The 124-group aluminum opacity table is collapsed to 32 log-spaced groups over [1 eV, 30 keV] using Planck weighting at T=1 keV. A slab of aluminum (rho=0.1 g/cm^3, L=0.4 cm, T=1 keV) moves at v=0.5994 cm/ns (~2% of c) toward a stationary observer at z_O=12 cm. At t_O=10 ns the per-group radiation energy density spectrum at the observer is compared to the semi-analytic solution computed with the same collapsed opacity.

This test verifies Planck-weighted multigroup collapse (124->32), material motion corrections, manual Lagrangian mesh rebuild, and transparent boundary tally with coarser energy resolution.

**Governing equations:** Radiative transfer (IMC Monte Carlo) with frequency-dependent opacity and material motion.

### Configuration

| Parameter | Value |
|-----------|-------|
| Domain | [0, 12.1] cm in x, thin y,z |
| Mesh points | 80 (20 slab + 60 vacuum) |
| EOS | Ideal gas (irrelevant, noHydroFeedback) |
| Slab properties | rho=0.1 g/cm^3, T=1 keV, v=0.5994 cm/ns |
| Vacuum properties | rho~0, T~0, v=0 |
| Energy groups | 32 (collapsed from 124-group aluminum table, Planck-weighted at 1 keV) |
| newPhotonsPerCell | 10000 |
| Time stepping | Adaptive, dt from 1e-3 ns to 0.1 ns, ramp 1.1x |
| End time | t_O = 10 ns |
| Mesh motion | Manual Lagrangian rebuild each step |
| Radiation | IMC, withHydro=true, MMC=false, noHydroFeedback=true |

**Source:** `regression_tests/cases/moving_slab_mc_32/test.cpp`

### Output

`moving_slab_mc_32_spectrum.txt` -- per-group collapsed opacities and time-averaged radiation energy density from transparent boundary tally

### Validation

Compared against the semi-analytic solution computed by `regression_tests/moving_slab_benchmark.py` (original_vacuum variant) using the same 32-group collapsed opacity. The Python checker `regression_tests/lib/check_moving_slab_mc_32.py` performs the collapse, converts raw tally data to E_rad (GJ/cm^3/keV) and computes the energy-weighted fractional error (Eq. 20 of the 2026 paper).

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| Energy-weighted f-error | <= 0.30 | `MOVING_SLAB_MC_32_MAX_FERROR` |

### References

- McClarren, R. G. & Gentile, N. A. (2021). "Frequency-Dependent Material Motion Benchmarks for Radiative Transfer."
- Gentile, N. A. & McClarren, R. G. (2026). "A Modified Frequency-Dependent Material Motion Benchmark for Thermal Radiative Transfer."

---

## 21. yee_vortex_64 -- Yee Isentropic Vortex (64×64, Lagrangian)

**Tags:** `mpi`

### Physics

Stationary isentropic vortex (Yee et al. 1999) -- an exact steady-state solution of the compressible Euler equations. A smooth rotational velocity field is in exact pressure-gradient balance with isentropic density and pressure perturbations. The analytical solution at any time equals the initial condition.

This is the lower-resolution run of a convergence pair (see also `yee_vortex_128`).

**Governing equations:** 2D Euler equations (simulated in 3D with 1 cell in z).

### Configuration

| Parameter | Value |
|-----------|-------|
| Domain | [-5, 5]^2 x [0, dz] |
| Grid | 64 x 64 x 1 Cartesian |
| EOS | Ideal gas, gamma = 1.4 |
| Vortex strength | beta = 5 |
| Vortex center | (0, 0) |
| Mesh motion | Lagrangian + RoundCells (xy-plane) |
| Solver | HLLC, LinearGauss3D |
| Boundary | Rigid wall |
| End time | t = 10 |
| SLURM | 8 tasks, `bigrun` partition |

**Source:** `regression_tests/cases/yee_vortex_64/test.cpp`

### Output

`vortex_profile.txt` -- columns: x, y, volume, density, pressure, vx, vy

### Validation

The Python checker `regression_tests/lib/check_yee_vortex.py` computes the volume-weighted L1 density error against the analytical initial condition.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| Density L1 | <= 0.05 | `YEE_VORTEX_MAX_DENSITY_L1` |

### References

- Yee, H-C., Sandham, N. & Djomehri, M. (1999). "Low dissipative high order shock-capturing methods using characteristic-based filters." *JCP* 150, 199-238.

---

## 22. yee_vortex_128 -- Yee Isentropic Vortex (128x128, Lagrangian)

**Tags:** `mpi`

### Physics

Higher-resolution companion to `yee_vortex_64`. Same isentropic vortex problem on a 128x128x1 mesh. Together with the 64x64 run, this establishes the spatial convergence rate of the Lagrangian scheme for smooth flows.

### Configuration

Same as `yee_vortex_64` except:

| Parameter | Value |
|-----------|-------|
| Grid | 128 x 128 x 1 Cartesian |
| SLURM | 16 tasks, `bigrun` partition |

**Source:** `regression_tests/cases/yee_vortex_128/test.cpp`

### Output

`vortex_profile.txt` -- same format as 64x64 case

### Validation

Same as `yee_vortex_64`.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| Density L1 | <= 0.05 | `YEE_VORTEX_MAX_DENSITY_L1` |

---

## 23. cartesian_gauss_linear -- Cartesian Gauss-Linear Interpolation

**Tags:** `serial`

### Physics

Tests the `LinearGauss3D` spatial reconstruction scheme in Cartesian mode. A 3D Voronoi mesh is initialized with fields that are linear in Cartesian coordinates (density, pressure, internal energy, velocity). The test compares face-interpolated values from Cartesian vs spherical `LinearGauss3D` modes against exact values in an annular region (1.1 < r < 1.9).

**Source:** `regression_tests/cases/cartesian_gauss_linear/test.cpp`

### Output

`cart_gauss_linear_metrics.txt` -- scalar/velocity max relative errors for Cartesian and spherical modes, and number of faces checked.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| Cartesian scalar max rel error | < 1e-6 | `CART_GAUSS_LINEAR_MAX_SCALAR_REL` |
| Cartesian velocity max rel error | < 0.1 | `CART_GAUSS_LINEAR_MAX_VEL_REL` |
| Cartesian scalar error < spherical scalar error | Required | -- |
| faces_checked | > 0 | -- |

---

## 24. spherical_gauss_linear -- Spherical Gauss-Linear Interpolation

**Tags:** `serial`

### Physics

Complementary to `cartesian_gauss_linear`: tests `LinearGauss3D` in spherical mode. Fields are linear in spherical coordinates (r, theta, phi). Spherical interpolation should outperform Cartesian for these fields.

**Source:** `regression_tests/cases/spherical_gauss_linear/test.cpp`

### Output

`gauss_linear_metrics.txt` -- scalar/velocity max relative errors for spherical and Cartesian modes.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| Spherical scalar max rel error | < 1e-8 | `GAUSS_LINEAR_MAX_SCALAR_REL` |
| Spherical velocity max rel error | < 0.1 | `GAUSS_LINEAR_MAX_VEL_REL` |
| Spherical scalar error < Cartesian scalar error | Required | -- |

---

## 25. spherical_collapse -- Spherical Collapse Symmetry

**Tags:** `mpi`

### Physics

A dense shell collapses inward under its own pressure in a cubed-sphere mesh. Tests that the collapse remains spherically symmetric by measuring density and radial-velocity scatter in radial bins. Uses Eulerian hydro with HLLC, `SphericalLinearGauss`, and rigid walls.

### Configuration

| Parameter | Value |
|-----------|-------|
| Mesh motion | Eulerian (fixed) |
| Solver | HLLC, SphericalLinearGauss |
| SLURM | 64 tasks, `bigrun` partition |

**Source:** `regression_tests/cases/spherical_collapse/test.cpp`

### Output

`collapse_metrics.txt` -- fields: `max_density_scatter`, `max_velocity_scatter`, `pass`

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| max_density_scatter | < 0.1 | `COLLAPSE_MAX_DENSITY_SCATTER` |
| max_velocity_scatter | < 0.1 | `COLLAPSE_MAX_VELOCITY_SCATTER` |
| `pass` field | Must be `1` | -- |

---

## 27. rayleigh_taylor_mpi -- Rayleigh-Taylor Instability

**Tags:** `mpi`

### Physics

The Rayleigh-Taylor instability: a heavy fluid (rho=2) sits above a light fluid (rho=1) in a gravitational field with a perturbed interface. The instability growth rate is compared against reference values. Uses Lagrangian hydro with HLLC, LinearGauss3D, and RoundCells.

### Configuration

| Parameter | Value |
|-----------|-------|
| Mesh motion | Lagrangian + RoundCells |
| Solver | HLLC, LinearGauss3D |
| End time | t = 3 |
| SLURM | 128 tasks, `bigrun` partition |

**Source:** `regression_tests/cases/rayleigh_taylor_mpi/test.cpp`

### Output

- `rt_kinetic_energy.txt` -- time vs z-kinetic energy
- `rt_density_slice.txt` -- x, z, rho for cells near y=0.5
- `rt_final.h5` -- final HDF5 snapshot

### Validation

The Python checker `regression_tests/lib/check_rayleigh_taylor.py` validates the growth rate against a reference value.

### Pass Criteria

| Metric | Threshold | Environment Variable |
|--------|-----------|---------------------|
| Growth rate relative error | <= 0.25 | `RT_MAX_GROWTH_RATE_REL_ERROR` |

---

## 28. eulerian_diffusion_freefree_suite -- Grey Free-Free Radiation Diffusion Suite

**Tags:** `mpi`

### Physics

A suite of four grey radiation-diffusion tests with free-free opacity and Thomson scattering. Two colliding flows create a radiative shock. Tests are run at different resolutions (512 and 32 cells) with and without flux limiting, and the results are compared across configurations.

### Variants

| Case | Cells | Flux Limiter |
|------|-------|-------------|
| `eulerian_diffusion_freefree_1d` | 512 | No |
| `eulerian_diffusion_freefree_1d_512_limited` | 512 | Yes |
| `eulerian_diffusion_freefree_1d_32` | 32 | No |
| `eulerian_diffusion_freefree_1d_32_limited` | 32 | Yes |

**Source:** `regression_tests/cases/eulerian_diffusion_freefree_1d/test.cpp`

### Output

Per case: `temperature_profile.txt` (x, rho, T, Trad, vx), `shock_position.txt`

### Validation

Checks that all four temperature profiles and comparison plots are generated without fatal errors.

---

## 29. eulerian_diffusion_freefree_multigroup_suite -- Multigroup Free-Free Radiation Diffusion Suite

**Tags:** `mpi`

### Physics

Same setup as `eulerian_diffusion_freefree_suite` but with 32-group multigroup radiation diffusion instead of grey. Validates that the multigroup solver produces consistent results across resolutions and flux-limiter settings.

### Configuration

Same as grey suite plus:

| Parameter | Value |
|-----------|-------|
| Energy groups | 32 |
| Build args | `--energy_groups_num=32` |

**Source:** `regression_tests/cases/eulerian_diffusion_freefree_multigroup_1d/test.cpp`

### Output

Per case: `temperature_profile.txt` (x, rho, T, Trad, vx), `shock_position.txt`

### Validation

Checks that all four temperature profiles and comparison plots are generated without fatal errors.

---

## Summary Table

| Test | Tags | Physics | Validation | Key Threshold |
|------|------|---------|------------|---------------|
| `sod_1d` | serial | 1D Riemann problem | Exact Riemann solver | GOF <= 0.02 |
| `sedov_3d_mpi` | mpi | 3D blast wave | Sedov-Taylor ODE | rel L1 <= 0.30 |
| `till_compton` | serial | Compton equilibration | Temperature convergence | |Tgas-Trad| < 1% |
| `amr_random` | serial, mpi | AMR conservation | Extensive drift | drift <= 1e-8 (serial) |
| `amr_distributed_clip` | mpi | Distributed AMR clip conservation | Mass/energy sum | rel diff <= 1e-6 |
| `voronoi_volume` | serial, mpi | Geometric accuracy | Volume sum | rel error < 1e-10 |
| `lane_self_gravity` | mpi | Hydrostatic equilibrium | Density stability | metric < 4e-2 |
| `lane_self_gravity_fmm` | mpi, manual, benchmark | Hydrostatic equilibrium (FMM) | Density stability | metric < 4e-2 |
| `mach2_diffusion` | mpi | Radiative shock (grey) | NLTE solution | rel L1 <= 0.025 |
| `mach2_multigroup` | mpi | Radiative shock (MG) | NLTE solution | rel L1 <= 0.025 |
| `marshak_wave_1_diffusion` | serial | Marshak wave (non-eq) | Self-similar ODE | rel L1 <= 1e-2 |
| `marshak_wave_2_diffusion` | serial | Marshak wave (eq limit) | Self-similar ODE | rel L1 <= 1e-2 |
| `marshak_wave_3_diffusion` | serial | Marshak wave (non-uniform) | Fitted profiles | rel L1 <= 1e-2 |
| `marshak_wave_4_diffusion` | serial | Marshak wave (divergent) | Fitted profiles | rel L1 <= 1e-2 |
| `gresho_euler` | serial | Gresho vortex (fixed) | IC comparison | rel L1 <= 0.1 |
| `gresho_lagrangian` | mpi | Gresho vortex (moving) | IC comparison | rel L1 <= 0.05 |
| `desmore2012_mc` | mpi | MC IMC (no RW, 30 groups) | Densmore 2012 Fig. 4 | Tgas L1 <= 0.05 keV |
| `desmore2012_mc_serial` | serial | MC IMC (RW, 30 groups) | Densmore 2012 Fig. 4 | Tgas L1 <= 0.05 keV |
| `moving_slab_mc_32` | serial | Freq-dependent moving slab (original vacuum, 32-group collapsed) | Semi-analytic solution (collapsed) | f-error <= 0.30 |
| `yee_vortex_64` | mpi | Isentropic vortex (64x64) | IC density comparison | L1 <= 0.05 |
| `yee_vortex_128` | mpi | Isentropic vortex (128x128) | IC density comparison | L1 <= 0.05 |
| `cartesian_gauss_linear` | serial | Cartesian interpolation | Exact face values | scalar error < 1e-6 |
| `spherical_gauss_linear` | serial | Spherical interpolation | Exact face values | scalar error < 1e-8 |
| `spherical_collapse` | mpi | Spherical symmetry | Scatter in radial bins | scatter < 0.1 |
| `rayleigh_taylor_mpi` | mpi | RT instability | Growth rate | rel error <= 0.25 |
| `eulerian_diffusion_freefree_suite` | mpi | Grey free-free diffusion | Profile comparison | All outputs valid |
| `eulerian_diffusion_freefree_multigroup_suite` | mpi | MG free-free diffusion | Profile comparison | All outputs valid |
