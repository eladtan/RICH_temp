# Getting Started

This guide walks you through installing RICH, setting up your environment, and running your first simulation.

## Prerequisites

### Required

| Dependency | Version | Notes |
|------------|---------|-------|
| C++ compiler | C++17 support | GNU (gcc) or Intel (icx/icpx) |
| Boost | >= 1.74.0 | Headers and libraries |
| HDF5 | Any recent | For snapshot I/O |
| VTK | 9.x recommended | For visualization output |
| CMake | >= 3.20.2 | Build system |
| Make | Any | Build tool |

### Recommended

| Dependency | Notes |
|------------|-------|
| MPI | OpenMPI or IntelMPI for parallel runs |
| Parallel HDF5 | Built with MPI support |
| Parallel VTK | Built with MPI support |
| UCX / UCC | Performance boost for MPI |
| Python 3 | With `numpy`, `matplotlib`, `scipy` for analysis |
| ParMETIS | Optional load balancing (alternative to Hilbert) |
| pdflatex | For generating test reports |

## Installation

### Clone the Repository

```bash
git clone --recursive https://gitlab.com/eladtan/RICH.git
cd RICH
```

Verify submodules are initialized:

```bash
git submodule status
```

If submodules are missing, initialize them:

```bash
git submodule update --init --recursive
```

## Setting Up the Compiler Environment

RICH supports both GNU and Intel compiler toolchains. On HPC clusters, use environment modules to load the required dependencies.

### GNU Compiler Environment (HUJI ICPL cluster)

```bash
ml purge
ml boost/1.78.0
ml hdf5/1.14.2/gcc/12.3.0_cxx
ml vtk/9.3.0/gcc/12.3.0/with_mesa
ml openmpi/4.1.6/gcc/12.3.0
ml gcc/15.1.0
```

Save the configuration for future sessions:

```bash
ml save rich
```

Restore it later:

```bash
ml restore rich
```

### Intel Compiler Environment (HUJI ICPL cluster)

```bash
ml purge
ml Intel/OneApi/2024.2.1
ml boost/1.78.0
ml hdf5/1.14.2/Intel/OneApi-2023.2.0_cxx
ml gcc/15.1.0
ml vtk/9.3.0/Intel/OneApi/2024.2.1/with_X
ml openmpi/4.1.6/Intel/OneApi/2024.2.1  # required for MPI configs
```

Save and restore:

```bash
ml save rich_intel
ml restore rich_intel
```

### Verifying the Environment

Check that key tools are available:

```bash
# Compiler
g++ --version    # or icpx --version for Intel

# MPI (if using MPI configs)
mpirun --version

# CMake
cmake --version

# Python (for regression tests and analysis)
python3 --version
```

## Your First Build

Build the Sedov blast wave example (serial):

```bash
./build_rich.sh gnuRelease --test_name=sedov_3d
```

This will:
1. Create a build directory at `build/gnuRelease/`
2. Run CMake to configure the project
3. Compile the code with `make -j$(nproc)`
4. Produce the binary `build/gnuRelease/rich`

### Build with MPI

For MPI builds, first ensure MPI is loaded, then use an MPI config:

```bash
ml openmpi/4.1.6/gcc/12.3.0  # if not already loaded
./build_rich.sh gnuReleaseMPI --test_name=sedov_3d
```

The binary is at `build/gnuReleaseMPI/rich`.

## Your First Run

### Serial

```bash
cd runs/sedov_3d
../../build/gnuRelease/rich
```

### MPI (via SLURM)

```bash
cd runs/sedov_3d
sbatch --wait --exclusive --partition=bigrun --ntasks=4 \
  --wrap "mpirun -np 4 ../../build/gnuReleaseMPI/rich"
```

The simulation produces HDF5 snapshot files (e.g., `sedov_0.h5`, `sedov_100.h5`, `sedov_final.h5`).

## Running Regression Tests

Verify your installation by running the regression suite:

```bash
# Serial tests only (faster)
./regression_tests/run_all.sh --mode serial

# All tests
./regression_tests/run_all.sh
```

See [Running Tests](Running-Regression-Tests) for full options.

## Next Steps

- [Build System](Build-System) -- detailed build options and configurations
- [Running Simulations](Running-Simulations) -- serial, MPI, and SLURM execution
- [Simulation Setup](Simulation-Setup) -- how to write your own `main.cpp`
- [Examples](Example-Sod-Shock-Tube) -- step-by-step walkthroughs
