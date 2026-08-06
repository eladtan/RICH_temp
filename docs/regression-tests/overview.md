# Regression Tests: Framework Overview

RICH includes a comprehensive regression testing framework that builds, runs,
and validates both physics benchmarks and code-correctness checks. The
framework lives under `regression_tests/` and is driven by
`regression_tests/run_all.sh`.

The unified RICH and STORM inventory contains 39 `physics` tests and 17
`code_correctness` tests. Every discovered `REGRESSION_INFO` declares exactly
one category. `physics` covers physical scenarios and benchmarks;
`code_correctness` covers algorithm, mesh, geometry, MPI, sampling, cache,
guard, and invariant checks.

## Architecture

The regression system operates in three phases:

```
Phase 1: BUILD & RUN        Phase 2: WAIT & CHECK       Phase 3: SUMMARY
┌─────────────────────┐     ┌──────────────────────┐    ┌─────────────────┐
│ Discover tests      │     │ Wait for all jobs    │    │ Print table     │
│ Load definitions    │     │ Check build status   │    │ Report pass/fail│
│ Build (up to 4      │────>│ Check run exit code  │───>│ Clean artifacts │
│   concurrent)       │     │ Run check function   │    │ Exit 0 or 1     │
│ Run immediately     │     │ Record results       │    └─────────────────┘
│   after build       │     └──────────────────────┘
└─────────────────────┘
```

### Phase 1: Build and Run

Tests are built and launched in a **pipelined** fashion with up to 4 concurrent builds. Each test gets its own build subdirectory (`build/<config>/<test_id>/`) so executables do not overwrite each other. Available CPU cores are split evenly across concurrent builds (e.g., on a 64-core machine each build gets `make -j16`).

As soon as a build completes, the test **immediately starts running** while remaining tests continue to build. Serial tests run directly; MPI tests are submitted via SLURM (`sbatch --wait`).

### Phase 2: Wait and Check

After all tests finish, the framework:

1. Verifies the build succeeded (checks `build_status.txt`).
2. Verifies the run exited with code 0 (checks `run_exit_code.txt`).
3. Scans stdout/stderr for fatal markers (segfaults, `UniversalError`, aborts, etc.).
4. Calls the test-specific check function which validates physics results against analytical solutions.

### Phase 3: Summary

A summary table is printed. The script exits 0 only if all tests pass. By default, artifact directories are cleaned on success (use `--keep-artifacts` to retain them).

## Directory Structure

```
regression_tests/
├── run_all.sh                    # Main entry point
├── lib/
│   ├── regression_checks.sh      # Bash validation helpers and check functions
│   ├── check_sod_profile.py      # Sod vs exact Riemann solver
│   ├── check_sedov_exact.py      # Sedov vs Sedov-Taylor ODE
│   ├── check_mach2_profile.py    # Mach 2 shock vs NLTE analytical solution
│   ├── check_marshak_wave.py     # Marshak wave vs self-similar solutions
│   └── check_gresho_profile.py   # Gresho vortex vs azimuthal velocity IC
├── tests/                        # Legacy .sh metadata for pre-migration tests
│   ├── sod_1d.sh
│   ├── sedov_3d_mpi.sh
│   ├── till_compton.sh
│   ├── amr_random.sh
│   ├── amr_distributed_clip.sh
│   ├── voronoi_volume.sh
│   ├── lane_self_gravity.sh
│   ├── mach2_diffusion.sh
│   ├── mach2_multigroup.sh
│   ├── marshak_wave_1_diffusion.sh .. marshak_wave_4_diffusion.sh
│   ├── gresho_euler.sh
│   └── gresho_lagrangian.sh
├── cases/                        # Per-test source and config
│   ├── sod_1d/test.cpp
│   ├── sedov_3d_mpi/test.cpp
│   ├── till_compton/test.cpp
│   │   └── data/                 # Reference data (McGraw et al. 2023)
│   ├── lane_self_gravity_fmm/
│   │   ├── test.cpp
│   │   └── REGRESSION_INFO       # THUNDER case-local metadata
│   └── ... (one directory per test)
├── plot_results.py               # Plot profiles vs analytical solutions
└── generate_test_report.py       # Generate LaTeX/PDF test report
```

## Test Definitions

THUNDER discovers tests from case-local `REGRESSION_INFO` files. They define
the following variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `TEST_ID` | Yes | Unique test identifier |
| `TAGS` | No | Space-separated tags: `serial`, `mpi`, or both (default: `serial`) |
| `CATEGORY` | Yes | Exactly one of `physics` or `code_correctness` |
| `BUILD_TEST_NAME` | Yes | Path to the test directory (passed to `--test_name=`) |
| `RUN_DIR_REL` | Yes | Relative path to the run directory |
| `RUN_COMMAND` | Yes | Shell command to execute the test binary |
| `CHECK_FUNCTION` | Yes | Name of the bash function that validates results |
| `BUILD_ARGS` | No | Extra build arguments (e.g., `--energy_groups_num=32`) |
| `RUN_MODE` | No | `direct` (default) or `slurm` |
| `SLURM_NTASKS` | No | Number of SLURM tasks (default: `32`) |
| `SLURM_PARTITION` | No | SLURM partition (default: `bigrun`) |
| `SLURM_EXCLUSIVE` | No | Whether to use `--exclusive` (default: `1`) |

### Example: Sod 1D test definition

```bash
#!/usr/bin/env bash
TEST_ID="sod_1d"
TAGS="serial"
CATEGORY="physics"
BUILD_TEST_NAME="regression_tests/cases/sod_1d"
RUN_DIR_REL="regression_tests/cases/sod_1d"
RUN_COMMAND='"${RICH_BIN}"'
CHECK_FUNCTION="check_sod_case"
```

### Example: Sedov 3D MPI test definition

```bash
#!/usr/bin/env bash
TEST_ID="sedov_3d_mpi"
TAGS="mpi"
CATEGORY="physics"
BUILD_TEST_NAME="regression_tests/cases/sedov_3d_mpi"
RUN_DIR_REL="regression_tests/cases/sedov_3d_mpi"
CHECK_FUNCTION="check_sedov_case"
RUN_MODE="slurm"
SLURM_NTASKS="128"
SLURM_PARTITION="bigrun"
SLURM_EXCLUSIVE="1"
RUN_COMMAND='mpirun -np ${SLURM_NTASKS} "${RICH_BIN}"'
```

## Helper Library: `regression_checks.sh`

The shared validation library provides these utilities:

| Function | Purpose |
|----------|---------|
| `set_check_msg` | Sets the human-readable check message for the summary table |
| `has_fatal_markers` | Scans a file for crash indicators (segfault, `UniversalError`, core dump, etc.) |
| `check_no_fatal_markers` | Ensures neither stdout nor stderr contains fatal markers |
| `is_nonempty_and_newer` | Verifies a file exists, is non-empty, and was modified after the suite start (with NFS retry logic) |
| `last_numeric_token` | Extracts the last numeric value from a text file |
| `is_finite_number` | Checks that a value is a finite number (not NaN/Inf) |

Each test type has a dedicated `check_*_case` function (e.g., `check_sod_case`, `check_sedov_case`) that orchestrates file checks, calls Python validators, and sets the pass/fail message.

## Artifact Layout

When tests run, artifacts are stored under `regression_results/<timestamp>/<test_id>/`:

```
regression_results/20260219_143000/
  sod_1d/
    build.stdout.log       # CMake + make stdout
    build.stderr.log       # CMake + make stderr
    build_status.txt       # "0" on success
    run.stdout.log         # Simulation stdout
    run.stderr.log         # Simulation stderr
    run_exit_code.txt      # Exit code
    run_start_epoch.txt    # Unix timestamp of run start
    run_end_epoch.txt      # Unix timestamp of run end
    rich                   # Copy of the built binary
```

## Adding a New Regression Test

1. **Create a case directory** under `regression_tests/cases/<your_test>/` with a `test.cpp` that produces a text-based output file (avoid HDF5 snapshots in regression tests for speed).

2. **Create `REGRESSION_INFO`** in the case directory:
   ```bash
   TEST_ID="your_test"
   TAGS="serial"  # or "mpi" or "serial mpi"
   CATEGORY="code_correctness"  # or "physics"
   BUILD_TEST_NAME="regression_tests/cases/your_test"
   RUN_DIR_REL="regression_tests/cases/your_test"
   RUN_COMMAND='"${RICH_BIN}"'
   CHECK_FUNCTION="check_your_test_case"
   ```

3. **Add a check function** in `regression_tests/lib/regression_checks.sh`:
   ```bash
   check_your_test_case() {
       local run_dir="$1"
       local run_start_epoch="$2"
       local stdout_log="$3"
       local stderr_log="$4"

       if ! check_no_fatal_markers "$stdout_log" "$stderr_log"; then
           return 1
       fi

       # Validate your output file
       if ! is_nonempty_and_newer "${run_dir}/your_output.txt" "$run_start_epoch"; then
           set_check_msg "missing or stale your_output.txt"
           return 1
       fi

       # Run your validation (inline or via Python script)
       set_check_msg "your_test passed"
       return 0
   }
   ```

4. **Run the test** to verify:
   ```bash
   ./regression_tests/run_all.sh --test your_test --config gnuRelease --verbose
   ```

## Plotting and Reporting

After running tests, you can generate comparison plots and a PDF report:

```bash
# Plot results vs analytical solutions
python3 regression_tests/plot_results.py

# Generate LaTeX/PDF test report
python3 regression_tests/generate_test_report.py
```

See [Running Tests](running-tests.md) for details on options and output.
