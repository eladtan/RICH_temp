# Frequently Asked Questions

## General

### What is RICH?

RICH is a compressible hydrodynamic simulation code that solves the Euler equations on a 3D moving Voronoi mesh. It supports radiation transport, self-gravity, adaptive mesh refinement, and MPI parallelism. It is written in C++17.

### What problems can RICH solve?

RICH is designed for astrophysical and high-energy-density physics problems, including:

- Blast waves and shock propagation
- Tidal disruption events (TDE)
- Stellar structure and evolution
- Radiative shocks and Marshak waves
- Multi-material flows with different equations of state

### Where can I find the publications about RICH?

- Serial version: Yalinewich, Steinberg & Sari (2015), [ApJS 216, 35](http://iopscience.iop.org/0067-0049/216/2/35/)
- Parallel version: Steinberg, Yalinewich & Sari (2015), [ApJS 216, 14](http://adsabs.harvard.edu/abs/2015ApJS..216...14S)

## Building

### Which compiler should I use?

Both GNU (gcc) and Intel (icx/icpx) compilers are supported. Intel compilers often produce faster code for numerical workloads. For development, GNU with Debug mode is recommended. For production, try both `gnuReleaseMPI` and `intelReleaseMPI` and benchmark.

### What is the difference between Release and Debug configurations?

| Feature | Release | Debug |
|---------|---------|-------|
| Optimization | `-O2` | `-O0` |
| Assertions | Disabled | Enabled (`_GLIBCXX_ASSERTIONS`, `DEBUG`) |
| Debug symbols | Minimal | Full (`-gdwarf-3`) |
| Speed | Fast | Slow (5-50x slower) |

Use Debug for development and debugging; Release for production runs.

### How do I add a new dependency?

Dependencies are discovered via `LD_LIBRARY_PATH` or explicit directory variables. Set the appropriate environment variable or module before running `build_rich.sh`. See [Build System](build-system.md) for details.

### Why does CMake re-run every time I build?

`build_rich.sh` re-runs CMake when:
- Source files are added or removed
- `CMakeLists.txt` or `*.cmake` files change
- Build arguments change

This is by design to ensure build consistency.

## Running

### Do I need MPI for every simulation?

No. You can build and run in serial mode using any non-MPI configuration (e.g., `gnuRelease`). MPI is only needed for large-scale parallel simulations.

### Why must I use SLURM for MPI jobs?

On shared HPC clusters, `mpirun` should not be called directly on the login node. SLURM manages resource allocation and ensures your job gets exclusive access to compute nodes.

### How do I choose the number of MPI ranks?

A good rule of thumb is to have at least 10,000-100,000 cells per rank for efficiency. Too few cells per rank means communication overhead dominates; too many means you are not utilizing available parallelism.

### How do I restart a simulation?

RICH supports restart via HDF5 snapshots. The pattern is:
1. Periodically write `restart.h5` and `counter.txt` during the simulation
2. On restart, check for `counter.txt` and load the state from `restart.h5`

See [Simulation Setup](user-guide/simulation-setup.md) for details.

## Physics

### What equations of state are available?

IdealGas, Tillotson, OndrejEOS (tabular), and MixedEOS (composite). See [Equations of State](user-guide/equations-of-state.md).

### How does the moving mesh work?

RICH constructs a Voronoi tessellation from mesh-generating points. These points can move with the fluid (Lagrangian), stay fixed (Eulerian), or use a hybrid approach (RoundCells). The mesh is rebuilt each time step. See [Mesh and Tessellation](architecture/mesh-and-tessellation.md).

### What radiation methods are available?

- Grey (single-group) flux-limited diffusion
- Multigroup diffusion (N energy groups)
- Compton scattering (CMMC)
- Monte Carlo transport (IMC)

See [Radiation Transport](user-guide/radiation.md).

### How do I add a custom physics module?

Implement the appropriate abstract interface (`SourceTerm3D`, `PointMotion3D`, `CellUpdater3D`, etc.) and pass your implementation to `HDSim3D`. See [Code Architecture](architecture/overview.md) for the list of interfaces.

## Regression Tests

### How do I run the regression tests?

```bash
./regression_tests/run_all.sh
```

See [Running Tests](regression-tests/running-tests.md) for full options.

### A regression test failed. How do I debug it?

1. Check the build logs: `regression_results/<timestamp>/<test_id>/build.stderr.log`
2. Check the run logs: `regression_results/<timestamp>/<test_id>/run.stderr.log`
3. Check the checker output: `regression_tests/cases/<test_id>/<test>_check.stderr.log`
4. Run with `--verbose` to see real-time output
5. Run with `--keep-artifacts` to preserve logs on success

### Can I add my own regression test?

Yes. See [Regression Tests Overview](regression-tests/overview.md) for step-by-step instructions on adding a new test.

## Output and Visualization

### What format are the output files?

HDF5 (`.h5`) for full snapshots. Some regression tests write text profiles (`.txt`) for speed.

### How do I visualize 3D results?

Options:
1. Write VTU output and open in ParaView
2. Read HDF5 in Python with h5py and plot with matplotlib
3. Use the MATLAB readers under `visualisation/three_dimensional/matlab/`

See [Output and Visualization](user-guide/output-and-visualization.md).
