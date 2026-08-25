# Example: 3D Sedov Blast Wave

`runs/sedov_3d/main.cpp` is the only run definition tracked in the public
repository. It is a byte-for-byte copy of
`regression_tests/cases/sedov_3d_mpi/test.cpp`, so the example and regression
case use the same RICH API and physics setup.

## Setup

- Cubic domain: `[-1, 1]^3`
- Mesh points: `100,000` in serial; `5,000,000` for a multi-rank MPI run
- Ideal-gas equation of state with `gamma = 5/3`
- Density `1` everywhere
- Specific internal energy `8e5` for `r < 0.1`, and `0.1` outside
- Final simulation time `0.0075`

The calculation uses a moving 3D Voronoi mesh, HLLC fluxes, linear
reconstruction, and CFL time steps.

## Compile

From the repository root:

```bash
./build_rich.sh gnuRelease --test_name=sedov_3d
./build_rich.sh gnuReleaseMPI --test_name=sedov_3d
```

These commands compile only. A multi-rank run uses five million mesh points;
launch it only through an allocation sized for that workload.

## Output

When executed from `runs/sedov_3d/`, the program writes
`sedov_profile.txt`. It contains up to 500 volume-weighted radial bins:
radius, density, pressure, and radial velocity. It writes no HDF5 snapshots,
and generated text output is ignored by Git.

## Regression validation

```bash
./regression_tests/run_all.sh --test sedov_3d_mpi --verbose --keep-artifacts
```

This full regression is substantially more expensive than a compile check.
The source and checker in `regression_tests/cases/sedov_3d_mpi/` are
authoritative.
