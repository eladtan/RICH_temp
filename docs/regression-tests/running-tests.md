# Running Regression Tests

## Quick Start

Run the full regression suite:

```bash
./regression_tests/run_all.sh --mode all
```

This selects all 56 discovered tests: 39 physics tests and 17 code-correctness
tests. Normal execution submits work through SLURM.

## Command Reference

```
Usage: run_all.sh [options]

Options:
  --mode <mode>            Run mode (project default: serial)
  --category <category>    Run physics or code_correctness tests
  --config <name>          Build configuration (auto-derived from --mode if omitted)
  --mpi-np <N>             MPI ranks for MPI tests (default: 4)
  --test <id>              Run only one test
  --clean-results          Delete regression_results directory and exit
  --nproc <N>              Override detected core count (default: $(nproc))
  --keep-artifacts         Keep all logs even if all tests pass
  --verbose                Stream run output to terminal as well
  -h, --help               Show this help
```

## Run Modes

The `--mode` flag controls which tests are executed based on their tags:

| Mode | Tests Run | Default Config | Description |
|------|-----------|----------------|-------------|
| `serial` | Tests tagged `serial` | `gnuRelease` | Serial tests only, run directly |
| `mpi` | Tests tagged `mpi` | `gnuReleaseMPI` | MPI tests only, submitted via SLURM |
| `all` | All tests | `gnuReleaseMPI` | Every test regardless of tag |
| `serial_then_mpi` | All tests | `gnuRelease` then `gnuReleaseMPI` | Two-pass: serial first, then MPI |

`--category physics` and `--category code_correctness` select one of the two
disjoint primary categories. The category filter intersects with `--mode` and
`--test`. Dependencies are still expanded, so a selected test may bring in a
prerequisite from the other category.

### Examples

```bash
# Run all serial tests with auto-selected gnuRelease config
./regression_tests/run_all.sh --mode serial

# Run all MPI tests with auto-selected gnuReleaseMPI config
./regression_tests/run_all.sh --mode mpi

# Run MPI tests with Intel compiler
./regression_tests/run_all.sh --mode mpi --config intelReleaseMPI

# Two-pass mode: serial (gnuRelease) then MPI (gnuReleaseMPI)
./regression_tests/run_all.sh --mode serial_then_mpi

# Run all tests across both execution modes
./regression_tests/run_all.sh --mode all

# Run all physics regressions
./regression_tests/run_all.sh --category physics

# Run MPI code-correctness regressions
./regression_tests/run_all.sh --category code_correctness --mode mpi

# List physics tests without submitting work
./regression_tests/run_all.sh --list-tests --category physics
```

## Running a Single Test

Use `--test <id>` to run just one benchmark:

```bash
# Sod shock tube (serial)
./regression_tests/run_all.sh --test sod_1d --config gnuRelease

# Sedov blast wave (MPI, 128 tasks)
./regression_tests/run_all.sh --test sedov_3d_mpi

# Till Compton (serial, 32 energy groups)
./regression_tests/run_all.sh --test till_compton --config gnuRelease

# Run with verbose output to see simulation progress
./regression_tests/run_all.sh --test sod_1d --config gnuRelease --verbose
```

Use `./regression_tests/run_all.sh --list-tests` for the authoritative test
inventory. Each tab-separated row contains test ID, tags, case directory,
metadata path, and category. The grouped category inventory is also maintained
in the [test catalog](test-catalog.md).

## Interpreting Output

### Real-Time Progress

During execution, you see status updates for each phase:

```
Running regression suite
  Mode:      all
  Config:    gnuReleaseMPI
  Tests:     sod_1d sedov_3d_mpi till_compton ...
  MPI ranks: 4
  Cores:     64 (override with --nproc)
  Artifacts: regression_results/20260219_143000

=== BUILD & RUN PHASE (max 4 concurrent builds, 16 make-jobs each) ===
[BUILD] sod_1d               compiling...
[BUILD] sod_1d               OK
[RUN  ] sod_1d               started
[BUILD] sedov_3d_mpi         compiling...
...

=== RESULTS ===
[RUN  ] sod_1d               finished (3s)
[CHECK] sod_1d               PASS: Sod exact-profile comparison passed
[RUN  ] sedov_3d_mpi         finished (120s)
[CHECK] sedov_3d_mpi         PASS: Sedov exact-ODE comparison passed
...
```

### Summary Table

After all tests complete, a summary table is printed:

```
=== SUMMARY ===
Test                  Status  Details                                              Logs
--------------------  ------  ---------------------------------------------------- ----
sod_1d                PASS    Sod exact-profile comparison passed                  regression_results/20260219_143000/sod_1d
sedov_3d_mpi          PASS    Sedov exact-ODE comparison passed                    regression_results/20260219_143000/sedov_3d_mpi
till_compton          PASS    Till passed: Tgas/Trad agree within 1%               regression_results/20260219_143000/till_compton
...
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All tests passed |
| 1 | One or more tests failed |
| 2 | Invalid arguments or missing prerequisites |

## Artifact Management

### Artifact Layout

Logs for each run are stored under `regression_results/<timestamp>/<test_id>/`:

| File | Content |
|------|---------|
| `build.stdout.log` | CMake and make stdout |
| `build.stderr.log` | CMake and make stderr |
| `build_status.txt` | `0` if build succeeded |
| `run.stdout.log` | Simulation stdout |
| `run.stderr.log` | Simulation stderr |
| `run_exit_code.txt` | Simulation exit code |
| `run_start_epoch.txt` | Unix timestamp of run start |
| `run_end_epoch.txt` | Unix timestamp of run end |
| `rich` | Copy of the built binary |

### Cleanup

By default, artifact directories are removed when all tests pass. To change this:

```bash
# Keep artifacts even on success
./regression_tests/run_all.sh --keep-artifacts

# Clean all saved regression results
./regression_tests/run_all.sh --clean-results
```

## Environment Variable Overrides

You can tune pass/fail thresholds via environment variables:

| Variable | Default | Test |
|----------|---------|------|
| `SOD_MAX_DENSITY_GOF` | `2e-2` | sod_1d |
| `SOD_MAX_PRESSURE_GOF` | `2e-2` | sod_1d |
| `SEDOV_MAX_DENSITY_REL_L1` | `0.30` | sedov_3d_mpi |
| `SEDOV_MAX_PRESSURE_REL_L1` | `0.30` | sedov_3d_mpi |
| `SEDOV_MAX_VELOCITY_REL_L1` | `0.30` | sedov_3d_mpi |
| `AMR_RANDOM_MAX_DRIFT_SERIAL` | `1e-8` | amr_random (serial) |
| `AMR_RANDOM_MAX_DRIFT_MPI` | `1e-6` | amr_random (MPI) |
| `AMR_DISTRIBUTED_CLIP_THRESHOLD` | `1e-6` | amr_distributed_clip |
| `VORONOI_VOLUME_MAX_REL_ERROR` | `1e-10` | voronoi_volume |
| `LANE_GRAVITY_MAX_METRIC` | `4e-2` | lane_self_gravity |
| `LANE_GRAVITY_FMM_MAX_METRIC` | `4e-2` | lane_self_gravity_fmm |
| `MACH2_MAX_DENSITY_REL_L1` | `0.025` | mach2_diffusion, mach2_multigroup |
| `MACH2_MAX_TEMPERATURE_REL_L1` | `0.025` | mach2_diffusion, mach2_multigroup |
| `MARSHAK_MAX_TGAS_REL_L1` | `1e-2` | marshak_wave_1_diffusion through _4 |
| `MARSHAK_MAX_TRAD_REL_L1` | `1e-2` | marshak_wave_1_diffusion through _4 |
| `GRESHO_EULER_MAX_L1` | `0.1` | gresho_euler |
| `GRESHO_LAGRANGIAN_MAX_L1` | `0.05` | gresho_lagrangian |

Example usage:

```bash
SEDOV_MAX_DENSITY_REL_L1=0.20 ./regression_tests/run_all.sh --test sedov_3d_mpi
```

## Plotting Results

After running tests, generate comparison plots:

```bash
# Plot all tests with data from the latest run
python3 regression_tests/plot_results.py

# Plot all tests regardless of regression_results
python3 regression_tests/plot_results.py --all

# Save to custom directory
python3 regression_tests/plot_results.py --output-dir /tmp/my_plots

# Use a specific results directory
python3 regression_tests/plot_results.py --results-dir regression_results/20260219_143000
```

Requires `numpy`, `matplotlib`, and `scipy`.

## Generating a PDF Report

```bash
# Full report with plots
python3 regression_tests/generate_test_report.py

# Skip plot generation (use existing plots)
python3 regression_tests/generate_test_report.py --no-plots

# Generate .tex only, skip PDF compilation
python3 regression_tests/generate_test_report.py --no-compile
```

The report includes test descriptions, pass criteria, achieved metrics, and comparison plots. Requires `pdflatex` for PDF output.

## Troubleshooting

### Build Failures

Inspect the build logs:

```bash
cat regression_results/<timestamp>/<test_id>/build.stderr.log
```

Common causes: missing modules (Boost, HDF5, VTK, MPI), wrong compiler version.

### Run Failures

Inspect the run logs:

```bash
cat regression_results/<timestamp>/<test_id>/run.stderr.log
cat regression_results/<timestamp>/<test_id>/run.stdout.log
```

### Check Failures

Each Python checker writes detailed output:

```bash
cat regression_tests/cases/<test_id>/<test>_check.stderr.log
```

### SLURM Issues

For MPI tests submitted via SLURM, check that `--partition=bigrun` is available:

```bash
sinfo -p bigrun
```

Ensure OpenMPI is loaded:

```bash
ml openmpi/4.1.6/Intel/OneApi/2024.2.1
```
