# RICH Documentation

RICH is a compressible hydrodynamic simulation code on a 3D moving Voronoi mesh, written in C++17 with optional MPI parallelism. It supports radiation transport (grey and multigroup diffusion, Monte Carlo), self-gravity, adaptive mesh refinement, and multiple equations of state.

## Quick Links

| Section | Description |
|---------|-------------|
| [Getting Started](getting-started.md) | Prerequisites, installation, and your first build |
| [Build System](build-system.md) | `build_rich.sh`, configurations, CMake internals |
| [Running Simulations](running-simulations.md) | Serial, MPI, and SLURM execution |
| **User Guide** | |
| [Simulation Setup](user-guide/simulation-setup.md) | How to write `main.cpp` for a new problem |
| [Output and Visualization](user-guide/output-and-visualization.md) | HDF5 snapshots, VTK, Python post-processing |
| [Equations of State](user-guide/equations-of-state.md) | IdealGas, Tillotson, OndrejEOS, MixedEOS |
| [Radiation Transport](user-guide/radiation.md) | Grey diffusion, multigroup, Compton/CMMC |
| [Gravity](user-guide/gravity.md) | Self-gravity trees, TDE gravity |
| [AMR](user-guide/amr.md) | Adaptive mesh refinement |
| [Boundary Conditions](user-guide/boundary-conditions.md) | Rigid wall, periodic, inflow/outflow |
| **Architecture** | |
| [Code Overview](architecture/overview.md) | Directory layout, key abstractions |
| [Mesh and Tessellation](architecture/mesh-and-tessellation.md) | Voronoi, Delaunay, point motion |
| [Hydrodynamics](architecture/hydrodynamics.md) | Godunov scheme, HLLC, reconstruction |
| [MPI Parallelism](architecture/mpi-parallelism.md) | Domain decomposition, load balancing |
| **Regression Tests** | |
| [Overview](regression-tests/overview.md) | How the regression framework works |
| [Running Tests](regression-tests/running-tests.md) | CLI usage, modes, interpreting results |
| [Test Catalog](regression-tests/test-catalog.md) | All 14 tests with full details |
| **Examples** | |
| [Sod Shock Tube](examples/sod-shock-tube.md) | 1D serial example |
| [Sedov Blast Wave](examples/sedov-blast-wave.md) | 3D MPI example |
| [Marshak Wave](examples/marshak-wave.md) | Radiation example |
| [Lane-Emden](examples/lane-emden.md) | Self-gravity example |
| [TDE with Multigroup Radiation](examples/tde-simulation.md) | Full production TDE with Compton, AMR, gravity |
| **Reference** | |
| [FAQ](faq.md) | Frequently asked questions |
| [Troubleshooting](troubleshooting.md) | Common errors and solutions |
| [Contributing](contributing.md) | Code style, adding tests, workflow |
| [Changelog](changelog.md) | Version history |

## Publications

- Serial version: Yalinewich, Steinberg & Sari (2015), [ApJS 216, 35](http://iopscience.iop.org/0067-0049/216/2/35/)
- Parallel version: Steinberg, Yalinewich & Sari (2015), [ApJS 216, 14](http://adsabs.harvard.edu/abs/2015ApJS..216...14S)
