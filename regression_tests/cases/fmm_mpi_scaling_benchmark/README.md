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

The benchmark reports two FMM timings.  The cold timing reconstructs the solver,
local tree, process tree, LET plan, and force solution for every repetition.
The warm timing keeps one solver alive after a setup solve and measures repeated
solutions with unchanged positions, verifying that the topology epoch and
rebuild count remain fixed.  The quadrupole timing remains a complete rebuild
and solve.  All timings are the minimum, over repetitions, of the maximum rank
wall time.

The output also records mean timings, throughput, communication volume, peak
remote/process storage, persistent-memory attribution, bounded M2L operator-cache
entries/bytes/hits/misses/bypasses, quadrupole walk time, 8-to-16-node scaling,
and cold-to-warm FMM speedup.

One hundred deterministic target particles are checked against a distributed
direct sum. The benchmark reports both maximum and arithmetic-mean scaled force
errors.

The executable accepts runtime FMM tuning options without changing the stable
result-row schema:

```text
--fmm-order <1..FMM_MAX_ORDER>
--fmm-theta <0..1>
--fmm-leaf-capacity <positive integer>
--fmm-mean-error-target <positive finite value>
--fmm-max-error-target <positive finite value>
--warm-only
```

The defaults remain order 4, theta 0.5, leaf capacity 32, mean-error target
`1e-3`, and maximum-error target `5e-3`. A separate `fmm_config` line records
the selected values plus the Taylor coefficient and M2L-term counts.

`--warm-only` skips the redundant independent cold-repeat loop. The setup solve
is still timed and recorded as the single cold sample, supplies the accuracy
and checksum values, populates the persistent topology/operator cache, and is
followed by the requested number of measured warm solves.

For parameter screening inside an existing allocation, use
`run_accuracy_sweep.sh <rich-binary> <result-directory>`. It runs the initial
order/theta matrix and calls `summarize_accuracy_sweep.py`, which maps the
`columns` line to the `row` by field name and ranks candidates by warm mean time
subject to the configured error targets and the `8e-4` mean-error guardrail.

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
repetition count, `FMM_MPI_BENCH_MAX_REMOTE_MIB` controls the per-rank FMM
LET wire/decoded-payload budget (512 MiB by default), and
`FMM_MPI_BENCH_OPERATOR_CACHE_MIB` controls the total persistent
exact-displacement M2L cache budget (64 MiB per rank by default; zero disables
caching).  The distributed solver assigns two thirds to the local dual-tree
cache and one third to the LET cache.  Process-tree M2L operators are computed
in reusable scratch because their count is small.  Both persistent caches are
strictly bounded: after a cache reaches its share, additional operators are
computed in reusable scratch rather than retained.  The formal particle counts remain
one million and ten million. OpenMP and MKL thread counts default to one.
