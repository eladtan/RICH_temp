# huji-rich
RICH is an compressible hydrodynamic simulation on a moving mesh written in c++.
We've recently published papers explaining the [serial](http://iopscience.iop.org/0067-0049/216/2/35/) and 
[parallel](http://adsabs.harvard.edu/abs/2015ApJS..216...14S) versions of the code.

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

Saved `module` configurations can be find in:

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

### What the script does

- Builds into `build/<config>/`.
- Stores command/config tracking files in that directory and rebuilds cleanly when command arguments change.
- Re-runs CMake when source files are added/removed or when `CMakeLists.txt` / `.cmake` files change.
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

Acceptance checks are physics-based:
- **Sod**: compare simulated density/pressure profiles to the exact Riemann solution (`analytic/enrs.py`).
- **Sedov**: compare radial density/pressure/velocity profiles to the exact Sedov-Taylor ODE profile (`analytic/sedov_taylor.py`).
- **Till**: require final gas and radiation temperatures to agree within **1%**.
- **AMR random**: enforce `max_drift` below threshold (serial: 1e-8, MPI: 1e-6).
- **Voronoi volume**: enforce `rel_error < 1e-10`.
- **Lane self-gravity**: evolve a Lane-Emden n=3/2 star with tree self-gravity to t=5; require `|mean(density - density_initial)| < 1e-2`.

The regression cases write lightweight profile/text outputs (for example `sod_profile.txt` and `sedov_profile.txt`) and avoid snapshot dumps from the test cases.

You can tune tolerances with environment variables:
- `SOD_MAX_DENSITY_GOF`, `SOD_MAX_PRESSURE_GOF`
- `SEDOV_MAX_DENSITY_REL_L1`, `SEDOV_MAX_PRESSURE_REL_L1`, `SEDOV_MAX_VELOCITY_REL_L1`
- `LANE_GRAVITY_MAX_METRIC`

### Parallel execution

Tests are executed in two phases:
1. **Build phase (sequential)**: Each test is compiled one at a time. After each successful build, the binary is copied to a test-specific artifact directory so subsequent builds don't overwrite it.
2. **Run phase (parallel)**: All successfully built tests are launched simultaneously. Serial tests run directly; MPI tests are submitted via Slurm (`sbatch --wait`).

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

# Run all tests (default, equivalent to --mode all)
./regression_tests/run_all.sh
```

Tests tagged with both `serial` and `mpi` (e.g. `amr_random`, `voronoi_volume`) appear in both modes.

### Options

```shell
./regression_tests/run_all.sh \
  --mode serial \
  --config gnuRelease \
  --keep-artifacts \
  --verbose
```

- `--mode <serial|mpi|all>`: filter tests by tag (default: `all`).
  - `serial`: default config `gnuRelease`.
  - `mpi`: default config `gnuReleaseMPI`.
  - `all`: default config `gnuReleaseMPI`.
- `--config <name>`: build configuration (overrides the mode default).
- `--mpi-np <N>`: MPI ranks for the Sedov run (default: `4`).
- `--clean-results`: remove `regression_results/` and exit.
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

Clean all saved regression logs:

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


## Profiling

To run the `gprof` profiler (for compilation configs with `Prof`), after a simulation run is finished, a `gmon.out` file will be generated in the run directory. This file contains profiling information and can be processed into a nice PDF (`gprof.pdf`) via:

```shell
gprof RICH_EXE_PATH gmon.out | gprof2dot -s -w --show-samples | dot -Tpdf -o gprof.pdf
```

where `RICH_EXE_PATH` is a path to the `rich` binary that was used in this simulation.