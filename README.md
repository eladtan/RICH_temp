# RICH — Compressible Hydrodynamics on a Moving Mesh

RICH is a C++17 astrophysical simulation code for compressible hydrodynamics on
moving Voronoi meshes. It supports serial and MPI execution, adaptive mesh
refinement, self-gravity, radiation transport, and multiple equations of state.

This tree combines the current RICH feature set with the ablation branch's
radiation, transport, mesh, gravity, and regression-test work. The integration
keeps the current submodule interfaces while porting the ablation capabilities
that were still missing.

## Capabilities

- Moving-mesh finite-volume hydrodynamics in one, two, and three dimensions.
- Voronoi tessellation, MPI domain decomposition, load balancing, and
  distributed adaptive mesh refinement.
- Ideal-gas, Tillotson, tabulated, and mixed equations of state.
- Tree and fast-multipole self-gravity, including distributed MPI FMM and
  quadrupole support.
- Grey and multigroup radiation diffusion.
- Implicit Monte Carlo (IMC), discrete diffusion Monte Carlo (DDMC), Compton
  coupling, polarization, moving-interface transport, and spherical observers.
- HDF5 snapshots, VTK output, Python analysis tools, and a metadata-driven
  serial/MPI regression suite.

## Documentation

Start with the [documentation index](docs/README.md). The most useful guides are:

| Topic | Guide |
|---|---|
| Installation and first build | [Getting started](docs/getting-started.md) |
| Build configurations and flags | [Build system](docs/build-system.md) |
| Serial, MPI, and SLURM execution | [Running simulations](docs/running-simulations.md) |
| Creating a problem setup | [Simulation setup](docs/user-guide/simulation-setup.md) |
| Radiation and Monte Carlo transport | [Radiation transport](docs/user-guide/radiation.md) |
| Gravity and FMM | [Gravity](docs/user-guide/gravity.md) and [FMM architecture](docs/architecture/fmm-gravity.md) |
| AMR | [Adaptive mesh refinement](docs/user-guide/amr.md) |
| Regression framework | [Running tests](docs/regression-tests/running-tests.md) and [test catalog](docs/regression-tests/test-catalog.md) |
| Output and post-processing | [Output and visualization](docs/user-guide/output-and-visualization.md) |
| Common failures | [Troubleshooting](docs/troubleshooting.md) |

The repository also contains GitLab-wiki-ready pages under [`wiki/`](wiki/).

### Integration notes

For the history and technical decisions behind the ablation integration, see:

- [Ablation branch changes](docs/ablation_branch_changes.md)
- [Ablation merge hazards and resolutions](docs/ablation_merge_hazards.md)
- [DDMC moving-interface design](docs/radiation/ddmc_wollaeger_interface.md)
- [Distributed MPI FMM](docs/architecture/mpi-fmm-gravity.md)

## Requirements

### Required

| Dependency | Requirement |
|---|---|
| C++ compiler | C++17; GNU or Intel |
| CMake | 3.20.2 or newer |
| Boost | 1.74.0 or newer |
| HDF5 | Recent release; parallel build for MPI runs |
| VTK | 9.x recommended; MPI-compatible build for MPI runs |
| Make | Any recent release |

### Recommended or optional

- OpenMPI or Intel MPI for parallel runs.
- Python 3 with NumPy, SciPy, and Matplotlib for checks and analysis.
- UCX/UCC for MPI performance.
- ParMETIS as an alternative load-balancing backend.
- `pdflatex` for generated regression reports.

Site-specific module examples and dependency-discovery variables are documented
in [Getting started](docs/getting-started.md) and the
[Build system guide](docs/build-system.md).

## Clone and initialize

```bash
git clone --recursive https://gitlab.com/eladtan/RICH.git
cd RICH
git submodule status
```

For an existing clone or an incomplete checkout:

```bash
git submodule update --init --recursive
```

Several submodules use SSH GitHub URLs. Configure GitHub SSH access before
initializing them.

## Build

Each simulation is selected by the directory containing its `main.cpp` or
`test.cpp`. The top-level helper wraps CMake and keeps separate build trees for
each configuration.

```text
./build_rich.sh <config> --test_name=<path> [options]
```

### Build configurations

| Configuration | Compiler | Build type | MPI |
|---|---|---|---|
| `gnuRelease` | GNU | Release | No |
| `gnuDebug` | GNU | Debug | No |
| `gnuReleaseMPI` | GNU | Release | Yes |
| `gnuDebugMPI` | GNU | Debug | Yes |
| `gnuMixedMPI` | GNU | Release with selected debug files | Yes |
| `intelRelease` | Intel | Release | No |
| `intelDebug` | Intel | Debug | No |
| `intelReleaseMPI` | Intel | Release | Yes |
| `intelDebugMPI` | Intel | Debug | Yes |
| `intelMixedMPI` | Intel | Release with selected debug files | Yes |

These are the configurations advertised by the script's completion support.
The parser derives compiler, build type, and MPI support from the configuration
name; CMake rejects unknown compiler or build-type components.

### Complete build option reference

| Argument | Value/default | Effect |
|---|---|---|
| `<config>` | Required positional value | Selects GNU or Intel, Release/Debug/Mixed, and MPI as shown above. |
| `--test_name=<path>` | Required | Selects the run or regression directory containing `main.cpp` or `test.cpp`; passed to CMake as `TEST_DIR`. |
| `--with_asan` | Off | Sets `ASAN=1` for AddressSanitizer instrumentation. |
| `--energy_groups_num=<N>` | Compile-time default | Sets `ENERGY_GROUPS_NUM=N`. |
| `--debug_files=<file>` | None | Resolves a source-list file and sets `DEBUG_FILES`; intended for Mixed builds. |
| `--mc_debug` | Off | Sets `MC_DEBUG=1`. |
| `--mc_trace_debug=<N>` | Off | Sets `MC_TRACE_DEBUG=N`. |
| `--shared` | Off | Sets `DYNAMIC_LIBS=1` to use the shared-library build path. |
| `--high-res` | Off | Sets `HIGH_RES=1`. |
| `--memory_debug` | Off | Sets `MEMORY_DEBUG=1`. |
| `--memory_profile` | Off | Sets `MEMORY_PROFILE=1`. |
| `--assert` | Off | Sets `FORCE_ASSERT=1`. |
| `--timing` | Off | Sets `TIMING=1`. |
| `--montecarlo-polarization` | Off | Sets `RICH_MONTECARLO_POLARIZATION=ON`. |
| `--build-subdir=<name>` | None | Uses `build/<config>/<name>/` so multiple executables can coexist. It does not itself invalidate an existing configuration. |
| `--jobs=<N>` | `nproc` | Sets parallel build jobs. It does not itself invalidate an existing configuration. |
| `--completions` | Special first argument | Defines Bash completion when sourced: `source ./build_rich.sh --completions`. |

The script has no separate `--help` mode. Missing required arguments print the
usage line; unknown options fail immediately.

### Examples

```bash
# Serial GNU release build
./build_rich.sh gnuRelease --test_name=sedov_3d

# MPI GNU release build
./build_rich.sh gnuReleaseMPI --test_name=sedov_3d

# MPI build with a compile-time multigroup override
./build_rich.sh gnuReleaseMPI --test_name=BaseTDE --energy_groups_num=32

# Debug build with AddressSanitizer
./build_rich.sh gnuDebug --test_name=sod --with_asan
```

The default executable and build logs are written under `build/<config>/`:

```text
build/<config>/rich_<config>
build/<config>/rich
build/<config>/<config>_cmake.out
build/<config>/<config>_cmake.err
build/<config>/<config>_build.out
build/<config>/<config>_build.err
```

See [Build system](docs/build-system.md) for all configurations, flags, and
dependency lookup rules.

## Run

Run from the selected problem directory so snapshots and text output are kept
with that run.

```bash
# Serial example
cd runs/sedov_3d
../../build/gnuRelease/rich
```

For MPI, build with an MPI configuration and launch through the scheduler or
MPI runtime required by your site. A typical SLURM launch is:

```bash
cd runs/sedov_3d
sbatch --wait --exclusive --partition=<partition> --ntasks=4 \
  --wrap "mpirun -np 4 ../../build/gnuReleaseMPI/rich"
```

Do not assume the example partition or module stack matches another cluster;
follow [Running simulations](docs/running-simulations.md) for site-specific
setup and output handling.

## Regression tests

RICH uses the embedded THUNDER framework. Test metadata lives beside each case,
and artifacts are collected under `regression_results/`. The unified RICH and
STORM inventory contains 56 tests split into 39 `physics` tests and 17
`code_correctness` tests; every discovered case declares exactly one category.

The root wrapper clears `CPATH`, then invokes THUNDER with
`regression_tests/config.json`:

```text
./regression_tests/run_all.sh [options]
```

With no arguments it runs the `serial` mode using GNU Release builds. It does
not accept the old `--config` or `--mpi-np` options.

```bash
# List the tests discovered from current metadata
./regression_tests/run_all.sh --list-tests

# List only physics tests
./regression_tests/run_all.sh --list-tests --category physics

# Resolve builds and launches without executing them
./regression_tests/run_all.sh --dry-run --verbose

# Run serial tests
./regression_tests/run_all.sh --mode serial

# Run MPI tests through the configured scheduler
./regression_tests/run_all.sh --mode mpi

# Run code-correctness tests through serial and MPI phases
./regression_tests/run_all.sh --category code_correctness --mode serial_then_mpi

# Run MPI tests directly on the local host
./regression_tests/run_all.sh --mode mpi --local

# Run one test
./regression_tests/run_all.sh --test sod_1d --compiler gnu --build-type Release
```

### Complete regression-runner option reference

| Option | Value/default | Effect |
|---|---|---|
| `-h`, `--help` | — | Prints the generated THUNDER help and exits. |
| `--thunder-config <path>` | Wrapper supplies `regression_tests/config.json` | Selects another THUNDER project configuration; normally internal to the wrapper. |
| `--list-tests` | Off | Lists all discovered tests and exits. |
| `--test <id>` | None; repeatable | Runs only the named test IDs. Repeat to select multiple tests. |
| `--mode <mode>` | `serial` | Filters by tags. Choices: `serial`, `mpi`, `all`, `serial_then_mpi`, `mpi_then_serial`. Ordered modes perform separate serial and MPI passes. |
| `--category <category>` | All | Filters to `physics` or `code_correctness`. |
| `--nproc <N>` | Host CPU count | Sets build parallelism in local mode. |
| `--compile-jobs <N>` | `8` | Sets CPUs for SLURM build jobs. |
| `--artifact-dir <path>` | Configured `regression_results/` tree | Overrides the artifact directory. |
| `--keep-artifacts` | Off | Compatibility flag; THUNDER always retains test metadata. |
| `--verbose` | Off | Prints commands and subprocess output. |
| `--dry-run` | Off | Resolves and prints work without building or running. |
| `--recheck` | Off | Re-runs checks against the latest artifacts. |
| `--clean-results` | Off | Deletes the configured regression artifact directory. |
| `--local` | Off | Runs SLURM-tagged tests directly on the host; rank counts are not reduced. |
| `--partition <name>` | Metadata/default | Overrides the SLURM partition. |
| `--exclude <nodes>` | None | Excludes comma-separated SLURM nodes or ranges. |
| `--project <account>` | None | Sets the SLURM project/account. |
| `--sequential` | Off | Alias for `--no-concurrent`. |
| `--no-concurrent` | Off | Runs tests sequentially. |
| `--concurrent <N>` | `0` | Caps concurrent tests; `0` means unlimited. |
| `--nohup` | Off | Leaves submitted SLURM jobs running if a non-interactive runner is interrupted. |
| `--compiler <family>` | `gnu` | Selects `gnu` or `intel`. |
| `--build-type <type>` | `Release` | Selects `Release`, `Debug`, or `Mixed`. |

`--category` intersects with `--mode` and `--test`; dependency expansion may
include a prerequisite from the other category. `--mode mpi` and `--mode all`
include large cases such as the manual 128-rank `lane_self_gravity_fmm` test and
the 512-rank STORM `hohlraum_parallel` test. Inspect `--list-tests` or select
explicit tests before submitting on a shared cluster.

### Complete regression-test catalog

Modes, rank counts, build arguments, checks, and thresholds below come from the
current metadata and checker libraries. Unless stated otherwise, checks also
reject fatal markers and missing or stale artifacts.

#### RICH tests

| Test ID | Category; mode/resources | Coverage and pass criteria |
|---|---|---|
| `amr_distributed_clip` | Correctness; MPI 64 | Distributed AMR clipping; executable pass flag plus relative mass and energy drift at most `1e-6`. |
| `amr_random` | Correctness; serial and MPI 64 | Random AMR stress; maximum drift at most `1e-8` serial or `1e-6` MPI. |
| `cartesian_gauss_linear` | Correctness; serial | Cartesian/spherical Gauss reconstruction of linear fields. Limits: scalar relative error `1e-6`, Cartesian velocity `0.1`, spherical velocity `0.5`. |
| `ddmc_mpi_zero_cell` | Correctness; MPI 8 | Zero-cell ranks and cross-rank DDMC faces; reciprocity, rate/conductance, and packet-weight errors within `1e-10` to `1e-12` limits. |
| `desmore2012_mc` | Physics; MPI 32; 30 groups | Densmore 2012 heterogeneous-opacity MC benchmark; gas-temperature L1 at most `0.05`. |
| `desmore2012_mc_ddmc` | Physics; MPI 32; 30 groups | Coupled MC+DDMC Densmore benchmark; gas-temperature L1 at most `0.06`. |
| `desmore2012_mc_serial` | Physics; serial; 30 groups | Serial MC plus random-walk Densmore benchmark; gas-temperature L1 at most `0.05`. |
| `eulerian_diffusion_freefree_1d` | Physics; MPI 16 | 512-cell grey free-free profile; requires fresh temperature/shock data and gas/radiation-temperature and velocity plots. |
| `eulerian_diffusion_freefree_1d_32` | Physics; MPI 4 | 32-cell grey counterpart with the same data and plot checks. |
| `eulerian_diffusion_freefree_1d_32_limited` | Physics; MPI 4 | 32-cell limited grey case with fresh profiles and plots. |
| `eulerian_diffusion_freefree_1d_512_limited` | Physics; MPI 16 | 512-cell limited grey case with fresh profiles and plots. |
| `eulerian_diffusion_freefree_multigroup_1d` | Physics; MPI 8; 32 groups | 512-cell multigroup free-free profile with fresh-data and plot checks. |
| `eulerian_diffusion_freefree_multigroup_1d_32` | Physics; MPI 4; 32 groups | 32-cell multigroup counterpart with fresh-data and plot checks. |
| `eulerian_diffusion_freefree_multigroup_1d_32_limited` | Physics; MPI 4; 32 groups | 32-cell limited multigroup case with fresh-data and plot checks. |
| `eulerian_diffusion_freefree_multigroup_1d_512_limited` | Physics; MPI 8; 32 groups | 512-cell limited multigroup case with fresh-data and plot checks. |
| `eulerian_diffusion_freefree_suite` | Physics; MPI-tagged aggregate, 1 task | No simulation build; depends on four grey cases and requires fresh four-way temperature, radiation-temperature, density, and velocity figures. |
| `eulerian_diffusion_freefree_multigroup_suite` | Physics; MPI-tagged aggregate, 1 task | No simulation build; depends on four multigroup cases and requires analogous four-way figures. |
| `fmm_gravity_mpi` | Correctness; MPI 7 | Distributed FMM versus direct acceleration/potential; covers duplicate IDs, an empty rank, reuse, root rebuild, and inconsistent-domain rejection. |
| `fmm_gravity_mpi_guard` | Correctness; MPI 4 | Collective API guards: supported construction succeeds and the invalid potential option is rejected. |
| `fmm_gravity_serial` | Correctness; serial | Serial FMM acceleration/potential against a direct reference; requires executable pass. |
| `fmm_operator_cache` | Correctness; serial | Bounded operator-cache memory, warm hits, and bypass behavior. |
| `fmm_peer_exchange_rebuild` | Correctness; MPI 7 | Persistent peer exchanger while ranks change isolated/connected roles; every rebuild round must pass. |
| `fmm_process_pair_coverage` | Correctness; MPI 7 | Non-power-of-two process-pair interaction coverage; all enumerated cases must pass. |
| `fmm_quadrupole_benchmark` | Correctness/benchmark; serial | Monopole/quadrupole FMM versus direct gravity; requires benchmark accuracy pass. |
| `gresho_euler` | Physics; serial | Eulerian Gresho vortex profile; L1 at most `0.1`. |
| `gresho_lagrangian` | Physics; MPI 8 | Lagrangian Gresho vortex profile; L1 at most `0.05`. |
| `lane_self_gravity` | Physics; MPI 128 on 8 nodes | Lane-Emden equilibrium with tree gravity; final metric at most `4e-2`. |
| `lane_self_gravity_fmm` | Physics/manual benchmark; MPI 128 on 8 nodes | Lane-Emden equilibrium with distributed FMM; final metric at most `4e-2`. |
| `mach2_diffusion` | Physics; MPI 8 | Grey Mach-2 radiative shock; density and temperature relative L1 each at most `0.025`. |
| `mach2_multigroup` | Physics; MPI 8; 32 groups | Multigroup Mach-2 counterpart with the same `0.025` limits. |
| `marshak_wave_1_diffusion` | Physics; serial | Diffusion Marshak problem 1; gas/radiation temperature relative L1 each at most `1e-2`. |
| `marshak_wave_2_diffusion` | Physics; serial | Diffusion Marshak problem 2 with the same `1e-2` limits. |
| `marshak_wave_3_diffusion` | Physics; serial | Diffusion Marshak problem 3 with the same `1e-2` limits. |
| `marshak_wave_4_diffusion` | Physics; serial | Diffusion Marshak problem 4 with the same `1e-2` limits. |
| `moving_slab_mc_32` | Physics; MPI 32 on 8 nodes; 32 groups | Moving-slab MC spectrum; F-error at most `0.30`. |
| `radiation_direction_sampling` | Correctness; serial | Isotropic direction moments, angular chi-square, and observer Voronoi-area closure; requires executable pass. |
| `rayleigh_taylor_mpi` | Physics; MPI 128 | Rayleigh-Taylor growth from kinetic-energy/density data; relative growth-rate error at most `0.25`. |
| `sedov_3d_mpi` | Physics; MPI 128 | Sedov exact-ODE profile; relative L1 limits: density `0.50`, pressure `0.30`, velocity `0.60`. |
| `sod_1d` | Physics; serial | Sod exact Riemann profile; density and pressure goodness-of-fit at most `2e-2`. |
| `spherical_collapse` | Physics; MPI 64 | Collapse symmetry; density scatter, velocity scatter, and tangential RMS each at most `1e-4`. |
| `spherical_gauss_linear` | Correctness; serial | Spherical Gauss linear reconstruction; scalar relative error `1e-8`, velocity relative error `0.1`. |
| `spherical_gauss_tangential` | Correctness; serial | Spherical tangential face basis; maximum absolute error at most `1e-8`. |
| `till_compton` | Physics; serial; 32 groups | Till Compton equilibration; temperature relative difference at most `1e-2`, total-energy relative error at most `1e-8`. |
| `voronoi_volume` | Correctness; serial and MPI 64 | Summed Voronoi volume versus exact box volume within `1e-10`. |
| `yee_vortex_128` | Physics; MPI 16 | 128-resolution Yee vortex; density L1 at most `0.05`. |
| `yee_vortex_64` | Physics; MPI 8 | 64-resolution Yee vortex; density L1 at most `0.05`. |

#### Embedded STORM tests

These are discovered from `source/monte/examples/`; the superproject points
STORM at the checked-out RICH dependency submodules.

| Test ID | Category; mode/resources | Coverage and pass criteria |
|---|---|---|
| `cartesian_parallel_check` | Correctness; MPI, metadata default 1 task | Parallel Cartesian transport self-check; requires the explicit `cartesian_parallel_check PASS` marker. |
| `densmore2012` | Physics; MPI 4 | Native STORM Densmore benchmark; requires `PASS` (`DENSMORE2012_TGAS_L1`, documented threshold `0.10 keV`). |
| `hohlraum_parallel` | Physics; MPI 512 | Large hohlraum run; clean completion without fatal markers. No analytic threshold. |
| `marshak_wave_1` | Physics; serial | Native Marshak problem 1. `PASS` succeeds; reported `WARN` above nominal `0.10` L1 is also accepted as MC noise. |
| `marshak_wave_2` | Physics; serial | Native Marshak problem 2 with the same PASS-or-WARN policy. |
| `marshak_wave_3` | Physics; serial | Native Marshak problem 3 with the same PASS-or-WARN policy. |
| `marshak_wave_4` | Physics; MPI 16 | Parallel native Marshak problem 4 with the same PASS-or-WARN policy. |
| `moving_slab` | Physics; MPI 48 on 16 nodes | Native moving-slab spectrum, cyclic node distribution; requires checker `PASS` with reported F-error/L1. |
| `serial_cartesian` | Correctness; serial | Serial Cartesian smoke test; clean completion without fatal markers. |
| `till_compton_mc` | Physics; serial | Fresh four-column Till-Compton profile with at least two rows plus comparison/energy diagnostics; intentionally no curve threshold. |

The live metadata under [`regression_tests/cases/`](regression_tests/cases/) and
[`source/monte/examples/`](source/monte/examples/) is authoritative if a test
changes. Additional background is in the
[regression catalog](docs/regression-tests/test-catalog.md).

## Repository layout

```text
source/                    RICH source and bundled submodules
runs/                      Production and example problem definitions
regression_tests/cases/    RICH regression cases and metadata
docs/                      User, architecture, and regression documentation
wiki/                      GitLab wiki mirror
examples/                  Additional examples
analysis_files/            Analysis and diagnostic utilities
scripts/                   Development and operational helpers
```

Important embedded projects include STORM (`source/monte`), THUNDER
(`regression_tests/THUNDER`), MadVoro, MadCart, MeshDecomposer3D, EasyHDF5,
EasyRMA, `mpi_utils`, and `spatial_ds`. Always initialize submodules recursively
and keep their recorded commits synchronized with the superproject.

## Publications

- Serial RICH: Yalinewich, Steinberg & Sari (2015),
  [ApJS 216, 35](https://doi.org/10.1088/0067-0049/216/2/35).
- Parallel RICH: Steinberg, Yalinewich & Sari (2015),
  [ApJS 216, 14](https://ui.adsabs.harvard.edu/abs/2015ApJS..216...14S/abstract).
