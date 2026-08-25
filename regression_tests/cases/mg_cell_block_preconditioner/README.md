# MG cell-block preconditioner unit test

This test exercises runtime block sizes 1, 2, 3, 7, and 16. It checks exact
same-cell extraction (including duplicate sparse entries), exclusion of
neighbor-cell entries, scalar-Jacobi parity, alias-safe application, singular
block fallback, and release of candidate-local factor storage.

Build with the repository-native test builder and run the resulting executable.

Measured solver and physical A/B results are retained in
`BENCHMARK_RESULTS.md`.

The MPI physical A/B gates run scalar and cell-block Jacobi sequentially on
the same allocation. The 8-rank case uses 32,768 cells and `dt=3.75e-4`; the
128-rank case uses 524,288 cells and `dt=1.1811759842764439e-4`. The base
timestep scales with the cell spacing squared; the 128-rank value is doubled
from that similarity value so scalar Jacobi crosses the required 1,000-iteration
stress threshold while still converging within the fixed 10,000-iteration loop.

Each gate requires more than 1,000 scalar iterations, at least a threefold
iteration reduction, lower block-solver wall time, no fallback in the
representative matrix, positive/finite physical state, profile error below
0.5%, spectrum error below 1%, and integrated-energy error below 0.5%.

Each physical lane writes initial and final distributed restart snapshots in
HDF5 plus MPI VTK XML (`.pvtu` master and per-rank `.vtu` pieces). The physical
A/B comparator requires all four endpoint master artifacts.
