# Regression Test Catalog

This document describes the regression tests in the RICH suite. Each entry covers the physics being tested, the simulation configuration, validation methodology, pass/fail criteria, and references.

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
| SLURM | 64 tasks, `bigrun` partition |

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

## 10-13. marshak_wave_1 through marshak_wave_4 -- Marshak Wave Benchmarks

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

## fmm_gravity_serial -- Serial Fast Multipole Gravity

**Tags:** `serial`

Validates analytic empty/one/two-body cases, complete solid-harmonic storage through the supported order, clustered-tree subdivision, explicit invalid-input failures, nonzero M2L use, reduced P2P work, potential/acceleration error against a long-double direct sum, and improvement with higher expansion order and tighter acceptance.

**Source:** `regression_tests/cases/fmm_gravity_serial/test.cpp`

### Pass Criteria

| Metric | Threshold |
|--------|-----------|
| Scaled acceleration error | `< 2e-5` |
| Relative positive-kernel potential error | `< 5e-5` |
| M2L interactions | `> 0` |
| Ordered P2P pairs | `< N(N-1)` |
| Order convergence | order 6 error `<` order 2 error |

---

## fmm_gravity_mpi_guard -- Distributed FMM Adapter Guard

**Tags:** `mpi`

Builds `FastMultipoleAcceleration3D` after `MPI_Init` and verifies that the distributed backend is accepted. Because the acceleration adapter does not expose potential values, the same test also requires `computePotential=true` to be rejected collectively.

**Source:** `regression_tests/cases/fmm_gravity_mpi_guard/test.cpp`

### Pass Criteria

| Metric | Required value |
|--------|----------------|
| Constructor accepted | `1` |
| Unsupported potential option rejected | `1` |
| `pass` | `1` |

---

## fmm_gravity_mpi -- Distributed FMM Numerical and Reuse Regression

**Tags:** `mpi`

Compares distributed FMM acceleration and positive-kernel potential with a direct reference, exercises duplicate application cell IDs and an empty rank, verifies that a mass-only update reuses the topology plan, forces a rank-local root breach and rebuild, and checks collective rejection of inconsistent domain bounds.

**Source:** `regression_tests/cases/fmm_gravity_mpi/test.cpp`

### Pass Criteria

| Metric | Threshold / required value |
|--------|----------------------------|
| Maximum scaled acceleration or potential error | `< 2e-4` |
| Mass-only topology epoch and rebuild count | unchanged |
| Root-breach topology epoch | increases |
| Finite timing, mass, and memory statistics | `1` |
| Inconsistent domains rejected collectively | `1` |

---

## fmm_process_pair_coverage -- Distributed FMM Interaction Coverage

**Tags:** `mpi`

Runs several process-tree geometries on a non-power-of-two rank count and verifies that every ordered pair of active ranks is classified exactly once as process-level M2L, same-rank local work, or cross-rank LET work.

**Source:** `regression_tests/cases/fmm_process_pair_coverage/test.cpp`

### Pass Criteria

| Metric | Required value |
|--------|----------------|
| MPI ranks | `> 1` |
| Coverage cases | `3` |
| Duplicate or missing ordered rank pairs | none |
| `pass` | `1` |

---

## fmm_quadrupole_benchmark -- FMM/Quadrupole Accuracy and Runtime Sweep

**Tags:** `serial`, `benchmark`

Compares the serial FMM with RICH's production `GravityTree<Vector3D>` using quadrupole moments. Nested deterministic distributions at 256, 512, 1024, 2048, and 16384 particles are evaluated against the same long-double direct sum. The output records complete build+walk runtime, scaled acceleration error, FMM M2L count, and exact near-field pair count for every resolution.

**Source:** `regression_tests/cases/fmm_quadrupole_benchmark/test.cpp`

### Pass Criteria

| Metric | Threshold |
|--------|-----------|
| Resolution rows | `5` |
| Maximum FMM scaled error | `< 5e-3` |
| Maximum quadrupole scaled error | `< 5e-2` |
| Largest-case FMM M2L interactions | `> 0` |
| Largest-case FMM P2P pairs | `< N(N-1)` |
| Runtime at `N=16384` | FMM `<` quadrupole tree |
| Scaled error at `N=16384` | FMM `<= 1.25 *` quadrupole tree |

Runtime and relative accuracy wins are reported for every row. The largest row is long enough to assert the FMM crossover while keeping the smaller rows diagnostic only.

---

## fmm_mpi_scaling_benchmark -- Distributed FMM/Quadrupole Scaling

**Tags:** `mpi`, `manual`, `benchmark`

Uses the same deterministic global particle set for both distributed solvers and for both process counts. Morton-ordered virtual bins give each rank a compact 3D subdomain. A fixed MPI rank density is used on every node (4 ranks per node by default). A single exclusive 16-node allocation runs global particle counts of one million and ten million on both 8 and 16 nodes, corresponding by default to 32 and 64 ranks. Every reported runtime covers construction, communication, and force evaluation from scratch, and stage logs report maximum-rank RSS/high-water memory.

**Source:** `regression_tests/cases/fmm_mpi_scaling_benchmark/test.cpp`

### Output and Validation

The benchmark records maximum-rank wall time, throughput, FMM communication volume and peak temporary storage, quadrupole walk time, quadrupole/FMM runtime ratio, and 8-to-16-node speedup and efficiency. Eight deterministic target particles are compared with a distributed direct sum for each run.

### Pass Criteria

| Metric | Threshold / required value |
|--------|----------------------------|
| Benchmark matrix | `1e6` and `1e7` particles on 8 and 16 nodes |
| Placement | fixed positive ranks per node; all requested nodes used |
| FMM probe scaled error | `< 5e-3` |
| Quadrupole probe scaled error | `< 5e-2` |
| Timings, throughput, checksums, and scaling metrics | finite; timings and rates positive |

Performance superiority is reported but is not itself a pass requirement. This test is manual-only and runs only when selected explicitly.

---

## DDMC Moving-Interface G_U A/B Validation

**Tags:** `mpi`

This deterministic two-cell interface microbenchmark runs the same incident
packet ensemble twice, first with the moving-interface correction disabled and
then with it enabled.  The source cell is IMC-only, the target cell is
optically thick and DDMC-eligible, and both cells move normally to the shared
face with `U_n/c = 0.04`.

The static admission draws must be identical in both runs.  For admitted
packets, the corrected-to-static total weight ratio must equal

```text
G_U(mu) = 1 + 2 (U_n/c) K(mu)
```

as evaluated by the production `DDMCWollaegerInterface.hpp` kernel.  The test
also requires exact agreement between event classifications and the runtime
interface counters, with no fallback, bypass, or packet splitting.

**Source:** `regression_tests/cases/ddmc_moving_interface_ab/test.cpp`

---

## DDMC Zero-Cell and Cross-Rank MPI Validation

**Tags:** `mpi`

This test distributes two optically thick cells over eight MPI ranks, forcing
most ranks to own zero cells while the only DDMC-DDMC face crosses a rank
boundary.  It exercises resident-packet serialization, symmetric DDMC
correspondents, ghost-cell opacity exchange, and face-flux reduction.

The checker requires sampled remote DDMC leakage, at least one zero-cell rank,
and conserved total packet weight.  It also verifies the cross-rank finite-
volume identity

```text
V_i lambda_i_to_j = V_j lambda_j_to_i
```

from per-cell internal-rate and conductance diagnostics.

**Source:** `regression_tests/cases/ddmc_mpi_zero_cell/test.cpp`

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
| `mach2_diffusion` | mpi | Radiative shock (grey) | NLTE solution | rel L1 <= 0.025 |
| `mach2_multigroup` | mpi | Radiative shock (MG) | NLTE solution | rel L1 <= 0.025 |
| `marshak_wave_1` | serial | Marshak wave (non-eq) | Self-similar ODE | rel L1 <= 1e-2 |
| `marshak_wave_2` | serial | Marshak wave (eq limit) | Self-similar ODE | rel L1 <= 1e-2 |
| `marshak_wave_3` | serial | Marshak wave (non-uniform) | Fitted profiles | rel L1 <= 1e-2 |
| `marshak_wave_4` | serial | Marshak wave (divergent) | Fitted profiles | rel L1 <= 1e-2 |
| `gresho_euler` | serial | Gresho vortex (fixed) | IC comparison | rel L1 <= 0.1 |
| `gresho_lagrangian` | mpi | Gresho vortex (moving) | IC comparison | rel L1 <= 0.05 |
| `ddmc_static_invariants` | static | DDMC/hydro implementation invariants | Source-code guard script | script exits 0 |
| `ddmc_moving_interface_ab` | mpi | Moving IMC-DDMC interface | Corrected/static admitted-weight ratio | relative error <= 1e-9 |
| `ddmc_mpi_zero_cell` | mpi | Zero-cell ranks and cross-rank DDMC | Remote leaks, reciprocity, weight conservation | errors <= 1e-10 |
| `fmm_gravity_serial` | serial | Fast multipole self-gravity | Long-double direct sum and convergence | scaled error < 2e-5 |
| `fmm_gravity_mpi_guard` | mpi | Distributed FMM adapter | MPI construction and option guard | guard exits 0 |
| `fmm_gravity_mpi` | mpi | Distributed fast multipole self-gravity | Direct reference, empty rank, topology reuse/rebuild | scaled error < 2e-4 |
| `fmm_process_pair_coverage` | mpi | Distributed FMM traversal partition | Ordered active-rank pair coverage | all pairs classified once |
| `fmm_quadrupole_benchmark` | serial, benchmark | FMM versus quadrupole tree | Direct-sum accuracy and runtime sweep | finite timings; bounded errors |
| `fmm_mpi_scaling_benchmark` | mpi, manual, benchmark | Distributed FMM versus quadrupole tree | 1e6/1e7 particles on 8/16 nodes | finite metrics; bounded probe errors |

## Static Guards

`ddmc_static_invariants` runs `regression_tests/lib/check_ddmc_static_invariants.sh`.
It checks DDMC opacity-role bookkeeping, MPI face-flux reduction ordering,
DDMC resident weight-frame lifecycle, remote-leak target tallying, and the
`w/w0 > 8` diagnostic threshold before expensive transport regressions run.
