# Distributed FMM versus quadrupole-tree scaling benchmark

This manual benchmark compares the distributed FMM backend with the existing
MPI distributed quadrupole gravity tree using the same deterministic particle
set and the same opening parameter (`theta = 0.5`).

The registered regression allocates 16 exclusive nodes and runs four subcases,
with a fixed MPI rank density on every node:

| Global particles | Nodes | Default MPI ranks |
|---:|---:|---:|
| 1,000,000 | 8 | 128 |
| 1,000,000 | 16 | 256 |
| 10,000,000 | 8 | 128 |
| 10,000,000 | 16 | 256 |

Each solver is rebuilt from scratch for every repetition. Reported solve time
is the minimum, over repetitions, of the maximum rank wall time. The output
also records mean maximum-rank time, throughput, FMM communication volume,
FMM peak remote/process storage, quadrupole walk time, 8-to-16-node speedup and
efficiency, and quadrupole/FMM runtime ratio.

Eight deterministic target particles are checked against a distributed direct
sum. This accuracy sample is intentionally small so that the direct reference
is affordable at ten million particles.

The particle set is independent of MPI size. A fixed set of 4096 virtual
spatial bins is assigned to ranks, so the 8-node and 16-node runs use identical
positions and masses while retaining a spatial decomposition.

Run it explicitly because it is tagged `manual benchmark`:

```bash
./regression_tests/run_all.sh \
  --mode mpi \
  --config intelReleaseMPI \
  --test fmm_mpi_scaling_benchmark \
  --partition bigrun \
  --sequential \
  --keep-artifacts \
  --verbose
```

The default is 16 MPI ranks per node and two complete rebuild-and-solve
repetitions per solver and configuration. Set `FMM_MPI_BENCH_RANKS_PER_NODE`
to match the compute-node core layout, and `FMM_MPI_BENCH_REPEATS` for
exploratory timing runs. The formal particle counts remain one million and ten
million. OpenMP and MKL thread counts default to one so the MPI rank density is
the controlled scaling variable.
