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

Common configurations are `gnuRelease`, `gnuDebug`, `gnuReleaseMPI`,
`gnuDebugMPI`, `intelRelease`, `intelDebug`, `intelReleaseMPI`, and
`intelDebugMPI`. Useful optional flags include `--build-subdir=<name>`,
`--jobs=<N>`, `--memory_debug`, `--memory_profile`, `--assert`, `--timing`, and
`--montecarlo-polarization`.

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
and artifacts are collected under `regression_results/`.

```bash
# List the tests discovered from current metadata
./regression_tests/run_all.sh --list-tests

# Resolve builds and launches without executing them
./regression_tests/run_all.sh --dry-run --verbose

# Run serial tests
./regression_tests/run_all.sh --mode serial

# Run MPI tests through the configured scheduler
./regression_tests/run_all.sh --mode mpi

# Run MPI tests directly on the local host
./regression_tests/run_all.sh --mode mpi --local

# Run one test
./regression_tests/run_all.sh --test sod_1d --config gnuRelease
```

Use `--partition`, `--project`, and `--exclude` for scheduler settings;
`--concurrent` or `--no-concurrent` for orchestration; and `--artifact-dir` or
`--keep-artifacts` for result retention. The authoritative case list is the
[regression catalog](docs/regression-tests/test-catalog.md) and the live
metadata under [`regression_tests/cases/`](regression_tests/cases/).

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
