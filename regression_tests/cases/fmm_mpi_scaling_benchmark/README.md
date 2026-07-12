# Distributed FMM versus quadrupole-tree scaling benchmark

This manual benchmark compares the distributed FMM backend with the existing
MPI distributed quadrupole gravity tree using the same deterministic particle
set and the same opening parameter (`theta = 0.5`).

The registered regression allocates 16 exclusive nodes and runs four subcases,
with a fixed MPI rank density on every node:

| Global particles | Nodes | Default MPI ranks |
|---:|---:|---:|
| 1,000,000 | 8 | 32 |
| 1,000,000 | 16 | 64 |
| 10,000,000 | 8 | 32 |
| 10,000,000 | 16 | 64 |

Each solver is rebuilt from scratch for every repetition. Reported solve time
is the minimum, over repetitions, of the maximum rank wall time. The output
also records mean maximum-rank time, throughput, FMM communication volume,
FMM peak remote/process storage, quadrupole walk time, 8-to-16-node speedup and
efficiency, and quadrupole/FMM runtime ratio.

Eight deterministic target particles are checked against a distributed direct
sum. This accuracy sample is intentionally small so that the direct reference
is affordable at ten million particles.

The particle set is independent of MPI size. A fixed set of 4096 virtual
spatial bins is ordered by a 3D Morton key and contiguous key ranges are assigned
to ranks. The 8-node and 16-node runs therefore use identical positions and
masses while each rank owns a compact spatial subdomain. Compact ownership is
important here: slab ownership with full transverse extent makes all rank roots
overlap and can turn both LET and quadrupole exchange into an O(NP)-like memory
transient.

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

The memory-safe default is 4 MPI ranks per node and two complete
rebuild-and-solve repetitions per solver and configuration. Set
`FMM_MPI_BENCH_RANKS_PER_NODE` to increase the fixed rank density only after
checking the stage `max_rss_kib` output. `FMM_MPI_BENCH_REPEATS` controls the
repetition count and `FMM_MPI_BENCH_MAX_REMOTE_MIB` controls the per-rank FMM
LET memory budget (512 MiB by default). The formal particle counts remain one
million and ten million. OpenMP and MKL thread counts default to one.
