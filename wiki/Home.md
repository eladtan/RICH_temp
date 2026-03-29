# RICH -- Compressible Hydrodynamics on a Moving Mesh

RICH is a 3D compressible hydrodynamic simulation code on a moving Voronoi mesh, written in C++17 with optional MPI parallelism. It supports radiation transport (grey and multigroup diffusion, Monte Carlo), self-gravity, adaptive mesh refinement, and multiple equations of state.

## Quick Start

```bash
# Clone
git clone --recursive https://gitlab.com/eladtan/RICH.git
cd RICH

# Load modules (HUJI ICPL)
ml restore rich

# Build (serial)
./build_rich.sh gnuRelease --test_name=sedov_3d

# Run
cd runs/sedov_3d && ../../build/gnuRelease/rich

# Run regression tests
./regression_tests/run_all.sh --mode serial
```

## Documentation

### Getting Started
- [Getting Started](Getting-Started) -- Prerequisites, installation, first build
- [Build System](Build-System) -- `build_rich.sh`, configurations, CMake
- [Running Simulations](Running-Simulations) -- Serial, MPI, SLURM execution

### User Guide
- [Simulation Setup](Simulation-Setup) -- How to write `main.cpp`
- [Output and Visualization](Output-and-Visualization) -- HDF5, VTK, Python
- [Equations of State](Equations-of-State) -- IdealGas, Tillotson, OndrejEOS
- [Radiation Transport](Radiation-Transport) -- Diffusion, multigroup, CMMC
- [Gravity](Gravity) -- Self-gravity, TDE gravity
- [AMR](AMR) -- Adaptive mesh refinement
- [Boundary Conditions](Boundary-Conditions) -- Rigid wall, periodic, inflow

### Architecture
- [Code Architecture](Code-Architecture) -- Directory layout, key abstractions
- [Mesh and Tessellation](Mesh-and-Tessellation) -- Voronoi, Delaunay, point motion
- [Hydrodynamics](Hydrodynamics) -- Godunov scheme, HLLC, reconstruction
- [MPI Parallelism](MPI-Parallelism) -- Domain decomposition, load balancing

### Regression Tests
- [Regression Tests Overview](Regression-Tests-Overview) -- Framework architecture
- [Running Regression Tests](Running-Regression-Tests) -- CLI, modes, results
- [Regression Test Catalog](Regression-Test-Catalog) -- All 20 tests in detail

### Examples
- [Sod Shock Tube](Example-Sod-Shock-Tube) -- 1D serial example
- [Sedov Blast Wave](Example-Sedov-Blast-Wave) -- 3D MPI example
- [Marshak Wave](Example-Marshak-Wave) -- Radiation example
- [Lane-Emden Star](Example-Lane-Emden) -- Self-gravity example
- [TDE with Multigroup Radiation](Example-TDE-Simulation) -- Production TDE with Compton, AMR, gravity

### Reference
- [FAQ](FAQ) -- Frequently asked questions
- [Troubleshooting](Troubleshooting) -- Common errors and solutions
- [Contributing](Contributing) -- Code style, adding tests, workflow

## Publications

- Serial: Yalinewich, Steinberg & Sari (2015), [ApJS 216, 35](http://iopscience.iop.org/0067-0049/216/2/35/)
- Parallel: Steinberg, Yalinewich & Sari (2015), [ApJS 216, 14](http://adsabs.harvard.edu/abs/2015ApJS..216...14S)
