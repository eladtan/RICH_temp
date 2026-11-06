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

The local particle ordering, ranges, radii, and expansions are rebuilt every
solve.  By default, the distributed solver uses the previous local tree as a
hysteresis template: an existing leaf splits only above
`persistentLeafSplitFactor * leafCapacity`, while an existing internal subtree
merges only at or below `persistentLeafMergeFactor * leafCapacity`.  The
defaults are 4.0 and 0, so a subtree merges only once it is empty. A newly
initialized tree still refines at `leafCapacity`; the wider thresholds apply
only to later refits, and a rank whose root cube changes starts from scratch.

Every split or merge changes the topology and rebuilds the LET plan.  On the
moving-mesh TDE workload a merge factor of 0.2 merged subtrees that hovered
near the threshold on a third of the solves, and those rebuilds took 80% of the
FMM time; with 0 the LET plan was rebuilt 7 times instead of 212 over the same
322 cycles, with unchanged force error, warm-solve cost and memory.  Over
about 1,000 solves without a root change the number of empty leaves grew by
0.6%.

Every persistent internal node materializes all eight spatial children,
including empty leaf children.  A particle crossing into or out of an octant
therefore changes occupancy without creating or deleting a child.  Empty source
subscriptions are omitted from each per-solve LET payload and contribute no
P2P pairs.

The full tree is only a stable address space and interaction superset. Hot
execution is occupancy-masked on every solve: empty nodes skip P2M/M2M/L2L,
local and LET interactions with an empty source or target skip operator
resolution and kernel execution, and active LET interactions use compact source
and geometry indices. This remains valid when occupancy changes because every
retained interaction is still present in the plan and automatically becomes
active when both endpoint subtrees become non-empty.

Automatic merging removes an under-filled retained subtree when the low
threshold is crossed. Setting `persistentLocalTreeTopology=false` restores the
fresh sparse tree built independently on every solve.

Distributed communication plans are split into two levels:

1. the rank-root process tree, process interaction plan, and process
   communication graphs;
2. the local-leaf-dependent LET plan.

A changed exact local-root cube or rank activation change rebuilds both
levels.  `rebuildTopologyEverySolve` also forces both levels to rebuild, but is
reported separately from actual root/leaf changes.  The forced mode also
disables local interaction-plan reuse, preserving its previous semantics.  A
changed exact local node structure with unchanged rank roots rebuilds only the
LET.  The process tree and its three communication graphs remain valid because
their geometry and routing depend only on rank-root cubes.  The LET descriptor
exchange still receives current root topology metadata before rebuilding.

Mass-only changes therefore recompute expansions while reusing the process and
LET plans.  Reuse is not decided by a probabilistic hash: a compact exact list
of node spatial keys, child masks, leaf status, and depths is compared.  A
separate exact leaf-occupancy signature records particle counts for diagnostics.
Particle-count changes no longer invalidate interaction plans when the full
node structure is unchanged: M2L geometry depends on fixed node cubes, while
P2P subscriptions are keyed by leaf spatial key and payloads carry their current
particle counts, masses, positions, and identities.

`FmmDistributedOptions::reuseInteractionPlansAcrossLeafCountChanges` controls
this behavior and defaults to true.  Setting it to false restores the previous
conservative invalidation rule for A/B testing.  Structural changes still
rebuild the LET, and root changes still rebuild both the process and LET
topologies.  The 64-bit topology hash retained in diagnostics is not a
correctness decision.

Persistent-tree diagnostics report the number of refit ranks, leaf splits,
subtree merges, and currently empty leaves.  A split or merge is a real
structural change and rebuilds the LET; ordinary octant crossings and occupancy
changes reuse it.

MPI distributed-graph communicators are also retained when every rank reports
the same normalized peer set.  `FmmSolveStats` reports root/leaf change counts,
separate process/LET rebuild counters, communicator reuse, and topology phase
timings.  Set `RICH_FMM_TRACE=1` to print one maximum-rank timing/rebuild line
per gravity solve; this is intended for short profiling runs rather than normal
production output.

The trace splits the two most expensive parts of a LET rebuild.
`let_subscription` is split into
- `let_sub_build`: local record and wave planning;
- `let_sub_wave_sync`: the elapsed time of the wave-count `MPI_Allreduce`,
  which includes waiting for the slowest rank as well as the collective
  itself;
- `let_sub_pack`: packing and interaction reordering;
- `let_sub_exchange`: the subscription exchange;
- `let_sub_receive`: decoding.

`let_finalize` is split into `let_fin_sort`, `let_fin_sources`,
`let_fin_geometry` (M2L operator grouping) and `let_fin_p2p`.

Every solve also reports `let_plan_age_solves` and `process_plan_age_solves`
(solves since each plan was rebuilt) and the remote source counts.  With
`FmmDistributedOptions::letStalenessDiagnostics`, which is off by default and
changes no result, it also reports how far a retained compat LET plan has
drifted from the current geometry:
- `g_growth_beyond_<eps>_ranks`: ranks where some node's planning radius `g`
  (the retained radius, or the cube radius when that is zero) grew by more than
  `1+eps` since the LET rebuild;
- `root_radius_growth_ratio_max`: rank-root growth since the process rebuild;
- a bracket on executed M2L/M2P interactions that would fail their build-time
  admissibility test today. The optimistic count takes a source's radius at
  build as its current radius, a lower bound that is zero on a rebuild solve.
  The bound count takes its cube radius, an upper bound.

The optimistic bound relies on the monotone retained envelope, so the option
requires `persistentLocalTreeTopology`.  The one-tree-per-rank solver alone
fills these fields.

To measure the force error of a run, set `directErrorSampleCount`.  On solve
`directErrorSampleSolve` (one-based), a deterministic sample of target
accelerations is compared with a distributed direct sum over all particles.
The comparison repeats every `directErrorSamplePeriod` solves after the
selected solve; zero disables repetition.  Rank 0 prints one
`fmm_tde_sample_error` line per sampled solve with the mean, RMS, median, p90,
p99 and maximum relative errors.  Each sampled solve costs every rank about
samples times its local particle count in pair evaluations; the accelerations
are unchanged.

For comparing two builds of the compat LET planner, set `RICH_FMM_PLAN_DUMP=1`.
Every LET rebuild then writes `fmm_plan_dump_r<rank>_<n>.bin` to the working
directory: the source tables, interactions, prepared M2L geometries and their
use counts, wave and target ranges, the subscription bytes sent, the
subscriptions received and the remote descriptors kept after pruning.  `<n>`
counts rebuilds in the process.  Two runs of the same input are compared file
by file with `cmp`.  This is a debugging aid: it writes one file per rank and
rebuild.

## Gravity-owner redistribution

With `spatiallyRedistributeForGravity` (the default), every solve first moves
the particles to temporary gravity owners: each particle gets a level-20 Hilbert
key of its position normalised by the solve domain, and `size - 1` splitters
cut the key range into one interval per rank.  The splitters are sampled once
and then frozen, because resampling every step moves interval boundaries and
invalidates the LET plan.

Frozen splitters are only valid while the keys mean the same thing.  They are
resampled when:

- the solve domain changes (for example `UpdateBox`), since that moves every
  key.  The same change also forces every rank to rebuild its local root,
  because a retained root keeps the old domain's dyadic lattice while a rebuilt
  one snaps to the new lattice, and the LET plan rejects that mix;
- the worst owner would receive more than `gravityOwnerImbalanceTrigger`
  (default 8) times the owner count the last sampling produced (at least the
  mean), which catches drift of the particles relative to the splitters;
- the worst owner would exceed `maxGravityOwnerBytesPerRank` (default and
  maximum `INT_MAX`, the MPI count limit of the exchange).

The load is measured with a reduce-scatter of the per-destination counts before
the exchange, so a stale partition is visible before it is replaced.  Rank 0
prints one `FMM gravity splitters sampled` line on the first solve and one
`FMM gravity splitters resampled: reason=...` line per resample, with the worst
owner before and after.  An owner still over its byte limit after resampling is
a collective `UniversalError` naming the worst rank, its count and the mean.
The record exchange also checks each rank's received count against the
prediction, which separates corrupted counts from a genuine imbalance.

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
- `fmm_gravity_mpi_redistribute`: gravity-owner redistribution lifecycle on a
  clustered body set: a domain change and a drift each collapse the stale
  partition and trigger exactly one resample, the warm result matches a fresh
  calculator, the owner byte limit fails collectively with diagnostics but does
  not limit the result return, the record exchange detects count mismatches and
  over-limit receives, and a domain change with a partial root rebuild (no
  redistribution) moves every rank to the new lattice.
- `fmm_trace_smoke`: three hydro steps of a small self-gravitating sphere
  through `FastMultipoleAcceleration3D` with the trace and the LET reuse
  diagnostics on.  Rank 0's output goes to a log of its own.  The check
  requires exactly one complete `fmm_solve_trace` record per gravity solve,
  with the phase and diagnostic fields, at least one LET rebuild, zero
  optimistic stale counts on rebuild solves, and forward simulation time.
- `fmm_let_staleness`: the staleness diagnostics on a two-rank geometry whose
  interaction classes and stale verdicts follow from distances alone (the
  derivation is in the test source).  Four solves of one retained plan
  (rebuild, unchanged, three particles moved, one leaf emptied) are checked rank
  by rank against the expected active, inactive and stale counts, growth flags
  and root growth ratio, once with one LET wave and once with eight.
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
