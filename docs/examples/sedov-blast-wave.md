# Example: 3D Sedov Blast Wave

`runs/sedov_3d/main.cpp` is the only run definition tracked in the public
repository. It is a byte-for-byte copy of
`regression_tests/cases/sedov_3d_mpi/test.cpp`, so the public example and the
validated regression case use the same RICH API and physics setup.

## Setup

- Cubic domain: `[-1, 1]^3`
- Mesh points: `100,000` in serial; `5,000,000` for a multi-rank MPI run
- Equation of state: ideal gas with `gamma = 5/3`
- Initial density: `1` everywhere
- Specific internal energy: `8e5` for `r < 0.1`, `0.1` outside
- Final simulation time: `0.0075`

The calculation uses a moving three-dimensional Voronoi mesh, HLLC fluxes,
linear reconstruction, and CFL time steps.

## Compile

From the repository root:

```bash
# Serial
./build_rich.sh gnuRelease --test_name=sedov_3d

# MPI
./build_rich.sh gnuReleaseMPI --test_name=sedov_3d
```

These commands compile the example; they do not run it. The multi-rank setup
uses five million mesh points, so launch it only through an allocation sized
for that workload.

## Output

When executed from `runs/sedov_3d/`, the program writes
`sedov_profile.txt`. It contains up to 500 volume-weighted radial bins with
four whitespace-separated columns:

1. radius
2. density
3. pressure
4. radial velocity

The example does not write HDF5 snapshots. Generated text output is ignored by
Git.

## Regression validation

The regression harness compares the generated profile against the
Sedov--Taylor self-similar solution:

```bash
./regression_tests/run_all.sh --test sedov_3d_mpi --verbose --keep-artifacts
```

That regression is substantially more expensive than a compile check. Its
source and checker under `regression_tests/cases/sedov_3d_mpi/` are
authoritative.
