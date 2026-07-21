# RICH -- Compressible Hydrodynamics on a Moving Mesh

RICH is a compressible hydrodynamic simulation code on a 3D moving Voronoi mesh, written in C++17 with optional MPI parallelism. It supports radiation transport (grey and multigroup diffusion, Monte Carlo), self-gravity, adaptive mesh refinement, and multiple equations of state.

**Publications:**
- Serial version: Yalinewich, Steinberg & Sari (2015), [ApJS 216, 35](http://iopscience.iop.org/0067-0049/216/2/35/)
- Parallel version: Steinberg, Yalinewich & Sari (2015), [ApJS 216, 14](http://adsabs.harvard.edu/abs/2015ApJS..216...14S)

## Documentation

Comprehensive documentation is available in two forms:

- **In-repo docs:** See the [`docs/`](docs/README.md) directory for the full user guide, architecture reference, regression test documentation, and examples.
- **GitLab Wiki:** The [`wiki/`](wiki/) directory contains GitLab-wiki-ready Markdown pages. To set up the wiki, clone your project's wiki repository and copy these files into it:
  ```bash
  git clone https://gitlab.com/eladtan/RICH.wiki.git
  cp wiki/*.md RICH.wiki/
  cd RICH.wiki && git add . && git commit -m "Initialize wiki" && git push
  ```

### Quick Links

| Topic | docs/ | Wiki |
|-------|-------|------|
| Getting Started | [docs/getting-started.md](docs/getting-started.md) | [Getting-Started](wiki/Getting-Started.md) |
| Build System | [docs/build-system.md](docs/build-system.md) | [Build-System](wiki/Build-System.md) |
| Running Simulations | [docs/running-simulations.md](docs/running-simulations.md) | [Running-Simulations](wiki/Running-Simulations.md) |
| Simulation Setup | [docs/user-guide/simulation-setup.md](docs/user-guide/simulation-setup.md) | [Simulation-Setup](wiki/Simulation-Setup.md) |
| Regression Tests | [docs/regression-tests/overview.md](docs/regression-tests/overview.md) | [Regression-Tests-Overview](wiki/Regression-Tests-Overview.md) |
| Test Catalog | [docs/regression-tests/test-catalog.md](docs/regression-tests/test-catalog.md) | [Regression-Test-Catalog](wiki/Regression-Test-Catalog.md) |
| Code Architecture | [docs/architecture/overview.md](docs/architecture/overview.md) | [Code-Architecture](wiki/Code-Architecture.md) |
| FAQ | [docs/faq.md](docs/faq.md) | [FAQ](wiki/FAQ.md) |
| Troubleshooting | [docs/troubleshooting.md](docs/troubleshooting.md) | [Troubleshooting](wiki/Troubleshooting.md) |

# Prerequisites
- A C++ compiler - GNU or Intel (Supporting C++ 17)
- Boost >= 1.74.0
- HDF5
- VTK
- Recommended: MPI - OpenMPI and IntelMPI were checked. \
Parallel HDF5 and VTK (built with the matching MPI library) are necessary.
- Recommended: UCX and UCC for performance boost on parallel applications.

# Installation

## Clone
Clone the latest version of RICH:
```shell
git clone --recursive https://gitlab.com/eladtan/RICH.git
```

Make sure the submodules are installed:
```shell
cd RICH
git submodule
```

# Compilation

RICH now uses the top-level build helper script `build_rich.sh`.  
This script wraps CMake + Make, keeps per-configuration build directories under `build/`,
and automatically re-runs CMake when relevant files or build arguments change.

## Setting up GNU compiler environment

### Modules
The list of recommended modules (in HUJI ICPL's cluster) to run RICH:
```
boost/1.78.0
hdf5/1.14.2/gcc/12.3.0_cxx
vtk/9.3.0/gcc/12.3.0/with_mesa
openmpi/4.1.6/gcc/12.3.0
gcc/15.1.0
```

### Save Configuration
It is advised to save these module configuration via 

```shell
ml save rich
```

and then reload it on a new shell via:

```shell
ml restore rich
```

Saved `module` configurations can be found in:

```shell
ls ~/.lmod.d
```

## Setting up Intel compiler environment
Similarly, for `intel` compilation, use `ml purge` and then load these modules (in HUJI ICPL's cluster):
```
Intel/OneApi/2024.2.1
boost/1.78.0
hdf5/1.14.2/Intel/OneApi-2023.2.0_cxx
gcc/15.1.0
vtk/9.3.0/Intel/OneApi/2024.2.1/with_X
```

It is advised to save these module configuration via 

```shell
ml save rich_intel
```

and then reload it on a new shell via:

```shell
ml restore rich_intel
```

## Compiling a specific run (`main.cpp`) file

Once the required modules are loaded, compile with:

```shell
./build_rich.sh gnuReleaseMPI --test_name=sedov2d_test
```

where `sedov2d_test` represents the subdirectory `runs/sedov2d_test` which contains a `main.cpp` file which defines a specific simulation. Other runs can be compiled by adding another directory with a `main.cpp` file to `runs/`.

### Build command format

```shell
./build_rich.sh <config> --test_name=<run_dir> [options]
```

- `<config>` is one of the build configurations listed below.
- `<run_dir>` is a directory under `runs/` that contains `main.cpp`.
- Optional flags:
  - `--with_asan` - Enable AddressSanitizer.
  - `--energy_groups_num=<N>` - Override `ENERGY_GROUPS_NUM`.
  - `--mc_debug` - Enable Monte-Carlo debug build flag.
  - `--debug_files=<path>` - Provide a mixed-debug file list for `DEBUG_FILES`.
  - `--build-subdir=<name>` - Build into `build/<config>/<name>/` instead of `build/<config>/`. Useful for keeping multiple test executables side by side without overwriting.
  - `--jobs=<N>` - Number of parallel make jobs (default: auto-detected via `nproc`).

### What the script does

- Builds into `build/<config>/` (or `build/<config>/<subdir>/` when `--build-subdir` is used).
- Stores command/config tracking files in that directory and rebuilds cleanly when command arguments change.
- Re-runs CMake when source files are added/removed or when `CMakeLists.txt` / `.cmake` files change.
- Runs `make -j<N>` where `<N>` defaults to `$(nproc)` (all available cores) or the value from `--jobs`.
- Writes logs to:
  - `build/<config>/<config>_cmake.out`
  - `build/<config>/<config>_cmake.err`
  - `build/<config>/<config>_build.out`
  - `build/<config>/<config>_build.err`
- Creates/updates the symlink `build/<config>/rich -> rich_<config>` after a successful build.

For other compilation configurations, replace `gnuReleaseMPI` with one of:
```shell
gnuReleaseMPI
gnuReleaseMPIProf
gnuReleaseProf
gnuRelease
gnuDebugMPI
gnuDebugMPIProf
gnuDebugProf
gnuDebug

intelReleaseMPI
intelReleaseMPIProf
intelRelease
intelReleaseProf
intelDebugMPI
intelDebugMPIProf
intelDebugProf
intelDebug
```

## Local regression tests

You can run a local full-benchmark regression suite after code changes:

```shell
./regression_tests/run_all.sh
```

The regression system is organized in its own directory:
- `regression_tests/run_all.sh` - main runner
- `regression_tests/tests/` - one test definition per benchmark (each tagged `serial`, `mpi`, or both)
- `regression_tests/lib/regression_checks.sh` - shared validation logic
- `regression_tests/cases/*/test.cpp` - dedicated benchmark entry files used by the runner

The suite builds and validates these regression cases:

| Test | Tags | Description |
|------|------|-------------|
| `sod_1d` | serial | 1D Sod benchmark |
| `till_compton` | serial | Till Compton test (case 3) |
| `sedov_3d_mpi` | mpi | 3D MPI Sedov explosion (Slurm, 128 tasks) |
| `amr_random` | serial, mpi | AMR random refine/remove |
| `voronoi_volume` | serial, mpi | Voronoi volume accuracy |
| `lane_self_gravity` | mpi | Lane-Emden self-gravity equilibrium (Slurm, 64 tasks) |
| `mach2_diffusion` | mpi | Mach 2 radiative shock, single-group diffusion (Slurm, 8 tasks) |
| `mach2_multigroup` | mpi | Mach 2 radiative shock, 32-group diffusion (Slurm, 8 tasks) |
| `marshak_wave_1` | serial | Marshak wave Problem 1 — non-equilibrium, uniform density, T^{-3} opacity |
| `marshak_wave_2` | serial | Marshak wave Problem 2 — equilibrium limit (kappa_P = kappa_R) |
| `marshak_wave_3` | serial | Marshak wave Problem 3 — non-uniform density rho=x^{20/19}, rho-dependent opacity |
| `marshak_wave_4` | serial | Marshak wave Problem 4 — divergent density rho=x^{-40/139}, stretched grid |
| `gresho_euler` | serial | Gresho vortex with Eulerian (fixed) mesh, t_end=5 |
| `gresho_lagrangian` | mpi | Gresho vortex with Lagrangian + RoundCells mesh, t_end=5 (Slurm, 8 tasks) |
| `yee_vortex_64` | mpi | Yee isentropic vortex 64x64, Lagrangian + RoundCells, t_end=10 (Slurm, 8 tasks) |
| `yee_vortex_128` | mpi | Yee isentropic vortex 128x128, Lagrangian + RoundCells, t_end=10 (Slurm, 16 tasks) |
| `spherical_collapse` | mpi | Spherical shell collapse symmetry test, Eulerian mesh (Slurm, 64 tasks) |
| `spherical_gauss_linear` | serial | SphericalLinearGauss3D LSQ gradient verification (linear field, machine precision) |
| `rayleigh_taylor_mpi` | mpi | 3D Rayleigh-Taylor instability, Lagrangian+RoundCells mesh (Slurm, 128 tasks) |
| `eulerian_diffusion_freefree_suite` | mpi | 1D Eulerian gray free-free diffusion suite: 512, 512-limited, 32, and 32-limited runs with 4-way comparison figures |
| `eulerian_diffusion_freefree_multigroup_suite` | mpi | 1D Eulerian multigroup (32-bin) free-free diffusion suite with Compton: 512, 512-limited, 32, and 32-limited runs with 4-way comparison figures |

Acceptance checks are physics-based:
- **Sod**: compare simulated density/pressure profiles to the exact Riemann solution (`analytic/enrs.py`).
- **Sedov**: compare radial density/pressure/velocity profiles to the exact Sedov-Taylor ODE profile (`analytic/sedov_taylor.py`).
- **Till**: require final gas and radiation temperatures to agree within **1%**.
- **AMR random**: enforce `max_drift` below threshold (serial: 1e-8, MPI: 1e-6).
- **Voronoi volume**: enforce `rel_error < 1e-10`.
- **Lane self-gravity**: evolve a Lane-Emden n=3/2 star with tree self-gravity to t=5; require `|mean(density - density_initial)| < 4e-2`.
- **Mach2 diffusion / multigroup**: run a Mach 2 radiative shock to t=0.01, gather MPI-distributed profiles, and compare density, gas temperature, and radiation temperature against the analytical NLTE radiative shock solution (`analysis_files/radiative_shock/nlte_radiative_shock.py`). Require relative L1 error below 2.5% for density, gas temperature, and radiation temperature.
- **Marshak wave 1-4**: non-equilibrium nonlinear Marshak wave benchmarks from Giron et al. (2026, arXiv:2601.05120). Grey diffusion (no flux limiter), 512-cell 1D, compared to self-similar analytical solutions from Krief & McClarren (2024) and Derei et al. (2024). Require relative L1 error below 1e-2 for both Tgas and Trad.
- **Gresho vortex (Euler / Lagrangian)**: Gresho vortex in 3D with one cell in z. Azimuthal velocity profile at t=5 compared to initial condition (exact stationary solution). Require relative L1 error below 0.1 (Euler) / 0.05 (Lagrangian).
- **Yee isentropic vortex (64 / 128)**: stationary isentropic vortex (Yee et al. 1999) with beta=5, gamma=1.4, domain [-5,5]^2. Lagrangian + RoundCells mesh, run to t=10. Volume-weighted L1 density error vs analytical IC must be <= 0.05. Run at 64x64 and 128x128 to verify second-order convergence.
- **Spherical collapse**: collapse a dense shell (0.9 < r < 1.0) on an Eulerian mesh built from replicated rounded sphere templates. Run until inward velocity at r=0.05 reaches 1. Require max angular scatter (std-dev/mean) of density and velocity across radial bins below 0.1.
- **Spherical Gauss linear**: fill cells with fields linear in spherical coordinates (r, theta) and verify that the LSQ gradient in `SphericalLinearGauss3D` recovers them to machine precision. Scalar max relative error < 1e-8, velocity max relative error < 0.1.
- **Rayleigh-Taylor**: 3D RT instability with ~1e6 cells, heavy-over-light density stratification with constant gravity. Flat interface with velocity perturbation in vz (amplitude 0.03, Gaussian-localised). Fit the exponential growth rate of vertical kinetic energy in the t=2 to t=3 window and require it to be within 25% of the analytical value sigma = sqrt(A*g*k).
- **Eulerian free-free diffusion suite**: run all four configured variants (`512`, `512-limited`, `32`, `32-limited`) and require fresh `temperature_profile.txt` outputs for each plus 4-way comparison figures (`Tgas`, `Trad`, `density`, `vx`) in `regression_tests/cases/eulerian_diffusion_freefree_compare/`.
- **Eulerian multigroup free-free diffusion suite**: run all four configured variants (`512`, `512-limited`, `32`, `32-limited`) with `ENERGY_GROUPS_NUM=32`, free-free multigroup opacity, and Compton enabled; require fresh `temperature_profile.txt` outputs for each plus 4-way comparison figures (`Tgas`, `Trad`, `density`, `vx`) in `regression_tests/cases/eulerian_diffusion_freefree_multigroup_compare/`.

The regression cases write lightweight profile/text outputs (for example `sod_profile.txt` and `sedov_profile.txt`) and avoid snapshot dumps from the test cases.

You can tune tolerances with environment variables:
- `SOD_MAX_DENSITY_GOF`, `SOD_MAX_PRESSURE_GOF`
- `SEDOV_MAX_DENSITY_REL_L1`, `SEDOV_MAX_PRESSURE_REL_L1`, `SEDOV_MAX_VELOCITY_REL_L1`
- `LANE_GRAVITY_MAX_METRIC`
- `MACH2_MAX_DENSITY_REL_L1`, `MACH2_MAX_TEMPERATURE_REL_L1`
- `MARSHAK_MAX_TGAS_REL_L1`, `MARSHAK_MAX_TRAD_REL_L1`
- `GRESHO_EULER_MAX_L1`, `GRESHO_LAGRANGIAN_MAX_L1`
- `YEE_VORTEX_MAX_DENSITY_L1`
- `COLLAPSE_MAX_DENSITY_SCATTER`, `COLLAPSE_MAX_VELOCITY_SCATTER`
- `RT_MAX_GROWTH_RATE_REL_ERROR`

### Parallel execution

Tests are built and run in a **pipelined** fashion:

1. Up to **4 tests build concurrently**, each in its own build subdirectory (`build/<config>/<test_id>/`) so executables don't overwrite each other. Available CPU cores are split evenly across concurrent builds (e.g. on a 64-core machine, each build gets `make -j16`).
2. As soon as a test finishes building, it **immediately starts running** while remaining tests continue to build. Serial tests run directly; MPI tests are submitted via Slurm (`sbatch --wait`) by default, or run locally via `mpirun` when `--local` is passed.
3. After all tests finish, results are checked and a summary table is printed.

Use `--nproc <N>` to override the auto-detected core count (e.g. to limit resource usage on a shared machine).
Use `--partition <name>` to override the SLURM partition for all MPI tests, or `--local` to bypass SLURM entirely and run MPI tests via direct `mpirun`.

Progress is printed in real time, showing which test is compiling, running, and its final pass/fail status.

### Running by mode (serial / MPI)

Use `--mode` to run only serial or only MPI tests:

```shell
# Run all serial tests (auto-selects gnuRelease config)
./regression_tests/run_all.sh --mode serial

# Run all MPI tests (auto-selects gnuReleaseMPI config)
./regression_tests/run_all.sh --mode mpi

# Run all MPI tests with a specific config
./regression_tests/run_all.sh --mode mpi --config intelReleaseMPI

# Run MPI tests on a different SLURM partition
./regression_tests/run_all.sh --mode mpi --partition short

# Run MPI tests locally (no SLURM)
./regression_tests/run_all.sh --mode mpi --local

# Run all tests (default, equivalent to --mode all)
./regression_tests/run_all.sh
```

Tests tagged with both `serial` and `mpi` (e.g. `amr_random`, `voronoi_volume`) appear in both modes.

### Running serial then MPI (`serial_then_mpi`)

Use `--mode serial_then_mpi` to run all serial-tagged tests first (with `gnuRelease`),
then all MPI-tagged tests (with `gnuReleaseMPI`) in a single invocation:

```shell
./regression_tests/run_all.sh --mode serial_then_mpi
```

This ensures serial tests are built and checked with a non-MPI config while MPI tests
use an MPI config. Each pass gets its own artifact directory. The overall exit code is
`0` only if both passes succeed. You can override the config with `--config`, in which
case the same config is used for both passes.

### Options

```shell
./regression_tests/run_all.sh \
  --mode serial \
  --config gnuRelease \
  --keep-artifacts \
  --verbose
```

- `--mode <serial|mpi|all|serial_then_mpi>`: filter tests by tag (default: `all`).
  - `serial`: default config `gnuRelease`.
  - `mpi`: default config `gnuReleaseMPI`.
  - `all`: default config `gnuReleaseMPI`.
  - `serial_then_mpi`: runs serial tests first (`gnuRelease`), then MPI tests (`gnuReleaseMPI`).
- `--config <name>`: build configuration (overrides the mode default).
- `--mpi-np <N>`: MPI ranks for MPI tests (default: `4`; individual tests may override).
- `--partition <name>`: override the SLURM partition for all MPI tests (default per-test, usually `bigrun`).
- `--local`: run MPI tests locally via `mpirun` instead of submitting through SLURM. Useful for machines without a SLURM scheduler or for quick local debugging.
- `--nproc <N>`: override auto-detected core count for parallel builds (default: `$(nproc)`).
- `--clean-results`: remove `regression_results/` and generated figure files under `regression_tests/`, then exit.
- `--keep-artifacts`: keep per-test logs even when all tests pass.
- `--verbose`: stream run output to terminal while also writing logs.

### Run a single regression test

Run one benchmark with `--test`:

```shell
# Sod only
./regression_tests/run_all.sh --test sod_1d --config gnuRelease

# Sedov only (set MPI ranks with --mpi-np)
./regression_tests/run_all.sh --test sedov_3d_mpi --mpi-np 8

# Till only
./regression_tests/run_all.sh --test till_compton --config gnuRelease
```

You can combine with `--mode`, `--config`, `--verbose`, and `--keep-artifacts`. For example:

```shell
./regression_tests/run_all.sh --test sod_1d --config gnuDebugMPI --verbose
```

Clean all saved regression logs and generated figures:

```shell
./regression_tests/run_all.sh --clean-results
```

### Artifacts and pass/fail behavior

- Logs are written under `regression_results/<timestamp>/`.
- The script returns exit code `0` only if all tests pass.
- Any failure returns non-zero exit code, prints a compact summary table, and keeps logs for inspection.
- If all tests pass and `--keep-artifacts` is not provided, the run's artifact directory is removed.

### Troubleshooting

- Ensure required modules and dependencies are loaded (Boost/HDF5/VTK/OpenMPI as listed above).
- Confirm `mpirun` is available in your environment for the Sedov MPI test.
- Ensure Python packages `numpy` and `h5py` are available (used by exact-solution checkers).
- If a run fails, inspect:
  - `regression_results/<timestamp>/<case>/build.stderr.log`
  - `regression_results/<timestamp>/<case>/run.stderr.log`
  - `regression_results/<timestamp>/<case>/run.stdout.log`
  - `regression_tests/cases/sod_1d/sod_check.stderr.log` (Sod exact-profile check details)
  - `regression_tests/cases/sedov_3d_mpi/sedov_check.stderr.log` (Sedov exact-profile check details)
  - `regression_tests/cases/mach2_diffusion/mach2_check.stderr.log` (Mach2 diffusion check details)
  - `regression_tests/cases/mach2_multigroup/mach2_check.stderr.log` (Mach2 multigroup check details)
  - `regression_tests/cases/marshak_wave_*/marshak_check.stderr.log` (Marshak wave check details)
  - `regression_tests/cases/gresho_euler/gresho_check.stderr.log` (Gresho Euler check details)
  - `regression_tests/cases/gresho_lagrangian/gresho_check.stderr.log` (Gresho Lagrangian check details)
  - `regression_tests/cases/yee_vortex_64/vortex_check.stderr.log` (Yee vortex 64x64 check details)
  - `regression_tests/cases/yee_vortex_128/vortex_check.stderr.log` (Yee vortex 128x128 check details)
  - `regression_tests/cases/spherical_collapse/collapse_metrics.txt` (Spherical collapse symmetry metrics)
  - `regression_tests/cases/rayleigh_taylor_mpi/rt_check.stderr.log` (Rayleigh-Taylor growth rate check details)
  - `regression_tests/cases/eulerian_diffusion_freefree_compare/run.stderr.log` (free-free suite runner errors)
  - `regression_tests/cases/eulerian_diffusion_freefree_multigroup_compare/run.stderr.log` (multigroup free-free suite runner errors)


### Plotting regression results

After running regression tests, generate comparison plots of numeric results against
analytical solutions:

```shell
python3 regression_tests/plot_results.py
```

The script finds the latest `regression_results/<timestamp>` directory to determine which
tests were run, reads profile data from the case directories, and saves PNG plots to
`regression_tests/plots/`.

Available plots:

| Test | Plot contents |
|------|---------------|
| `sod_1d` | Density and pressure vs x, compared to exact Riemann solution |
| `sedov_3d_mpi` | Density vs r (binned), compared to Sedov-Taylor ODE |
| `lane_self_gravity` | Density vs r (binned), compared to initial Lane-Emden profile |
| `till_compton` | Gas and radiation temperature vs time |
| `mach2_diffusion` | Density, Tgas, Trad vs x, compared to NLTE analytical solution |
| `mach2_multigroup` | Density, Tgas, Trad vs x, compared to NLTE analytical solution |
| `marshak_wave_1` | Tgas and Trad vs x, compared to self-similar analytical solution |
| `marshak_wave_2` | Tgas and Trad vs x (equilibrium limit) |
| `marshak_wave_3` | Tgas and Trad vs x (non-uniform density) |
| `marshak_wave_4` | Tgas and Trad vs x (divergent density, stretched grid) |
| `gresho_euler` | Pressure field, azimuthal velocity field, v_theta(r) vs IC |
| `gresho_lagrangian` | Pressure field, azimuthal velocity field, v_theta(r) vs IC |
| `yee_vortex_64` / `yee_vortex_128` | Density field, pressure field, density vs r (both resolutions), L1 convergence log-log |
| `spherical_collapse` | Radial density profile and angular scatter vs r |
| `rayleigh_taylor_mpi` | Vertical kinetic energy vs time (log scale) with fitted growth rate; density slice in xz plane |
| `eulerian_diffusion_freefree_suite` | 4-way overlays for `Tgas`, `Trad`, density, and `vx` (512 / 512-limited / 32 / 32-limited) |
| `eulerian_diffusion_freefree_multigroup_suite` | 4-way overlays for `Tgas`, `Trad`, density, and `vx` (512 / 512-limited / 32 / 32-limited, 32 energy bins with Compton) |

Options:

```shell
# Plot all tests with available data (regardless of regression_results)
python3 regression_tests/plot_results.py --all

# Save plots to a custom directory
python3 regression_tests/plot_results.py --output-dir /tmp/my_plots

# Use a specific results directory
python3 regression_tests/plot_results.py --results-dir regression_results/20260216_142301
```

Requires `numpy`, `matplotlib`, and `scipy`.

### Generating the test report (LaTeX/PDF)

Generate a standalone PDF document that describes every regression test (physics,
initial/boundary conditions, mesh movement, pass criteria, achieved metrics, and plots):

```shell
python3 regression_tests/generate_test_report.py
```

This will:
1. Run `plot_results.py --all` to produce comparison plots (PNG + PDF).
2. Write `regression_tests/test_report.tex`.
3. Compile it to `regression_tests/test_report.pdf` via `pdflatex`.

Options:

```shell
# Skip plot generation (use pre-existing plots)
python3 regression_tests/generate_test_report.py --no-plots

# Generate only the .tex file without compiling to PDF
python3 regression_tests/generate_test_report.py --no-compile

# Write output to a custom directory
python3 regression_tests/generate_test_report.py --output-dir /tmp/report

# Point to a custom plots directory
python3 regression_tests/generate_test_report.py --plots-dir /tmp/my_plots
```

The "Achieved Results" tables in the report are populated from the metric output files
produced by the most recent test run (e.g. `sod_check.stdout.log`,
`lane_gravity_metrics.txt`).  If no test outputs exist yet, those sections display a
placeholder message.

Requires `pdflatex` (any TeX Live install) for PDF compilation.  The `.tex` file is
always generated even if `pdflatex` is not available.

## Profiling

To run the `gprof` profiler (for compilation configs with `Prof`), after a simulation run is finished, a `gmon.out` file will be generated in the run directory. This file contains profiling information and can be processed into a nice PDF (`gprof.pdf`) via:

```shell
gprof RICH_EXE_PATH gmon.out | gprof2dot -s -w --show-samples | dot -Tpdf -o gprof.pdf
```

where `RICH_EXE_PATH` is a path to the `rich` binary that was used in this simulation.