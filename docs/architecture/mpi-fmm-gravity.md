# MPI Fast Multipole Gravity

This implementation adds an MPI backend for `FastMultipoleAcceleration3D`
without changing the Cartesian Taylor expansion convention used by the serial
solver.  The implementation is C++17 and requires MPI-3 distributed-graph and
neighborhood-collective support.

## Interaction decomposition

Each active MPI rank builds one adaptive local FMM tree over its owned cells.
The local root is an exact, retained cube; it expands only when a local body
leaves it.  All ranks then hold a compact deterministic binary **process tree**
over the active local-root cubes.

The process dual-tree traversal assigns every ordered target/source process
pair to the owner of the target process node.  Every ordered pair of active
ranks must terminate in exactly one class:

1. a well-separated process-tree M2L interaction;
2. a same-rank local dual-tree traversal; or
3. a cross-rank Local Essential Tree (LET) traversal.

The regression case `fmm_process_pair_coverage` expands accepted process-node
pairs back to rank pairs and checks this exact-one classification explicitly.

## Distributed passes

The solve is separated into these phases:

1. rebuild the local tree and compute local P2M/M2M;
2. aggregate local-root multipoles through the distributed process tree;
3. exchange only process multipoles required by target-owned process M2L pairs;
4. propagate process local expansions downward to rank leaves;
5. exchange LET payloads for unresolved cross-rank interactions;
6. perform the unchanged local self traversal and local L2L/L2P pass.

Process-pair tasks, dependencies, process coefficients, descriptors,
subscriptions, and LET payloads carry protocol magic, version, packet kind, and
`topologyEpoch`.  Stale or malformed packets are rejected.

## Communication and scaling properties

Per-solve payload communication uses cached directed graph communicators and
`MPI_Neighbor_alltoall[v]`.  There is no per-solve `MPI_Alltoall`, no rank-sized
send-buffer table in the LET or coefficient exchanges, and no globally
replicated particle tree.

Topology construction still performs:

- one compact `FmmRankRootDescriptor` `MPI_Allgather`, so process-tree geometry
  storage is `O(P)` per rank;
- bounded-wave `MPI_Allreduce` operations while distributed process-pair and
  descriptor traversal completes; and
- collective validation of numerical options and common domain bounds.

The process tree is intentionally replicated because it contains at most
`2 * active_ranks - 1` nodes.  Interaction lists and LET descriptors remain
rank-local.

## Topology reuse

The local tree is rebuilt every solve.  Distributed communication plans are
split into two levels:

1. the rank-root process tree, process interaction plan, and process
   communication graphs;
2. the local-leaf-dependent LET plan.

A changed exact local-root cube or rank activation change rebuilds both
levels.  `rebuildTopologyEverySolve` also forces both levels to rebuild, but is
reported separately from actual root/leaf changes.  The forced mode also
disables local interaction-plan reuse, preserving its previous semantics.  A
changed exact local leaf-key/particle-count signature with unchanged rank roots rebuilds only the
LET.  The process tree and its three communication graphs remain valid because
their geometry and routing depend only on rank-root cubes.  The LET descriptor
exchange still receives current root topology metadata before rebuilding.

Mass-only changes therefore recompute expansions while reusing the process and
LET plans.  Reuse is not decided by a probabilistic hash: a compact exact list
of local leaf spatial keys and particle counts is compared.  Moving bodies that
stay within the same spatial-node topology and preserve those leaf counts reuse
the plan safely; payloads always contain current masses and positions.  The
64-bit topology hash retained in diagnostics is not a correctness decision.

MPI distributed-graph communicators are also retained when every rank reports
the same normalized peer set.  `FmmSolveStats` reports root/leaf change counts,
separate process/LET rebuild counters, communicator reuse, and topology phase
timings.  Set `RICH_FMM_TRACE=1` to print one maximum-rank timing/rebuild line
per gravity solve; this is intended for short profiling runs rather than normal
production output.

## Identity

Application `ComputationalCell3D::ID` values are transported only for
troubleshooting.  They are not assumed globally unique.  Exact self identity is
the solver-owned pair `(ownerRank, ownerLocalIndex)`, so duplicate application
IDs do not remove physical interactions.

## Memory controls

`FmmDistributedOptions::maxRemoteBytes` limits the dominant simultaneous LET
working storage:

- per-peer encoded send buffers;
- the contiguous MPI send scratch;
- received wire bytes;
- flat decoded payload-record tables; and
- decoded multipole and particle arrays.

Payload sizes are planned before packing.  The decoder performs a wire preflight and then uses pre-sized flat record,
coefficient, and particle arrays rather than one allocation per remote node, so
the dominant requested storage is measurable before decoding.
An individual MPI neighborhood exchange is rejected if a count or displacement
cannot be represented by MPI's `int` interface.  Payload chunking beyond
`INT_MAX` is not implemented in this patch.

Protocol corruption, impossible peer routing, or a remote-memory-budget breach
inside a sparse collective calls `MPI_Abort`: allowing only one rank to throw
would otherwise strand its peers in a later collective.

`FmmSolveStats::bytesOwned` estimates persistent solver-owned storage after the
solve.  `peakRemoteBytes` records dominant LET working storage and
`peakProcessBytes` records process-expansion working storage.  MPI library
communicator internals and allocator metadata are not measurable by this code.

## Construction and API requirements

Construction and `solve` are collective on the supplied communicator.  All
ranks must use identical numerical and distributed options and identical domain
bounds.  MPI must be initialized before construction, and the solver must be
destroyed before `MPI_Finalize` for communicators to be released normally.

```cpp
FmmGravityOptions numerical;
numerical.expansionOrder = 4;
numerical.thetaCritical = 0.5;
numerical.leafCapacity = 32;

FmmDistributedOptions distributed;
distributed.rootSlackFactor = 1.25;
distributed.maxRemoteBytes = 2ull * 1024ull * 1024ull * 1024ull;

FastMultipoleAcceleration3D gravity(numerical, distributed, G);
```

`FastMultipoleAcceleration3D` is an acceleration-only adapter and rejects
`computePotential=true`.  Code that needs the positive `1/r` kernel potential
must call `DistributedFmmGravityCalculator::solve` directly and provide the
potential output vector.

The wire protocol sends trivially-copyable native records.  All ranks must use
the same executable ABI, endianness, integer widths, and floating-point format.
The code enforces IEEE-754 binary64 `double`, 64-bit `uint64_t`, 32-bit `int`,
`size_t` no wider than 64 bits, and exact no-padding sizes for every transmitted
record at compile time.  It still does not provide byte-order conversion or a
heterogeneous-cluster serialization format.

## Regression coverage

- `fmm_gravity_mpi_guard`: verifies construction after `MPI_Init` and verifies
  that the acceleration adapter rejects the unsupported potential option.
- `fmm_gravity_mpi`: compares acceleration and potential with direct summation,
  deliberately duplicates application cell IDs, includes an empty rank for
  three or more ranks, checks mass-only plan reuse, forces a topology rebuild,
  and verifies mismatched domain bounds are rejected collectively.
- `fmm_process_pair_coverage`: checks exact ordered-rank-pair coverage for
  separated, overlapping/empty-rank, and irregular process geometries.

## Remaining performance work

This is a correctness-first MPI backend.  The following are deliberately left
for later patches and benchmark-driven review:

- real MPI strong/weak scaling on hundreds to thousands of ranks;
- MPI-4 persistent neighborhood collectives;
- payload chunking beyond `INT_MAX`;
- OpenMP/SIMD tuning of P2P, M2L, and translation kernels;
- gravity-aware repartitioning and combined hydro/gravity work weights;
- reducing the replicated `O(P)` process-tree geometry at extreme rank counts;
- heterogeneous-ABI wire serialization; and
- GPU/offload kernels.
