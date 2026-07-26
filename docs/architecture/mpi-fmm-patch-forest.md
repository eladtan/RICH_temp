# MPI FMM Patch Forest: Persistent Reuse and Production Defaults

This document describes the Phase 6 and Phase 7 patch-forest behavior layered on
top of the patch-aware process tree and LET implementation. The one-tree-per-rank
solver remains available as a compatibility fallback.

## Production defaults

Patch-forest mode is the MPI default:

```cpp
enablePatchForest = true;
minimumPatchLevel = 7;
maximumPatchLevel = FMM_MAX_TREE_DEPTH;
targetParticlesPerPatch = 0;
maxLocalPatchCount = 65536;
maxTargetPatchesPerWave = 64;
maxReplicatedDescriptorBytes = 256ull * 1024ull * 1024ull;
```

The fixed level-7 partition is deliberately conservative. Count-based adaptive
patch splitting remains opt-in through `targetParticlesPerPatch`. Set
`enablePatchForest=false` for one-tree-per-rank A/B comparisons, emergency
fallback, and historical benchmark reproduction.

The replicated descriptor guard is intentionally lower than the LET working
budget. Crossing it means the replicated patch directory is no longer a viable
architecture and must be replaced by a distributed directory rather than by
silently allocating more memory.

## Interaction decomposition

Each rank partitions its particles on one global dyadic lattice and builds an
adaptive FMM tree for every occupied patch. The replicated process tree has one
leaf per `(ownerRank, patchId)`. Every ordered patch pair terminates in exactly
one class:

1. process-tree M2L;
2. same-patch local self traversal;
3. same-rank cross-patch dual-tree traversal;
4. cross-rank patch-aware LET traversal.

`fmm_patch_process_pair_coverage` audits this exact-one classification and also
checks deterministic process-node ownership.

## Persistent patch objects

Patch objects persist by stable patch ID. An existing patch whose root lattice
cube is unchanged is refit with the same split/merge hysteresis used by the
compatibility tree. Its particle buffers, tree storage, coefficient arrays, and
local interaction plan are retained where valid. Removed patches release their
owned storage.

Each persistent patch also carries a monotonic topology generation. The
generation advances on a root/structure change or conservative-radius growth
and is unchanged by occupancy changes or radius contraction. Remote target
subplans compare this generation rather than relying on a probabilistic geometry
hash, so a retained plan cannot survive a source-topology change through a hash
collision.

Local self-interaction plans are reused only when node identity, leaf status,
centers, half sizes, and conservative admissibility bounds remain valid.
Particle-count changes alone do not invalidate a local plan.

## Three-level invalidation model

The solver tracks three independent change classes:

1. **Patch geometry**: patch set, owner, root center/half size, or tight-root
   radius growth beyond the retained conservative bound.
2. **Patch structure**: node spatial keys, child masks, leaf status, or node
   radius growth that could invalidate LET admissibility.
3. **Patch occupancy**: current particle membership and counts in retained
   leaves.

The collective invalidation rules are:

- patch-set or root-geometry expansion: rebuild process topology and LET;
- structure-only change: retain process topology and rebuild LET;
- occupancy-only change: retain process topology and LET while every particle
  payload fits its reserved capacity;
- retained particle leaf exceeds its reserved capacity: retain process topology
  but rebuild the LET wave plan before communication begins.

Radius contractions retain the previous larger geometry in cached topology and
are therefore conservative. Expansions trigger invalidation.

## Incremental target-patch LET rebuilds

A structure-only LET rebuild does not discard every target plan. A cached target
subplan is retained when:

- the target patch identity and topology are unchanged;
- its set of remote source patches is unchanged;
- every source patch has the same conservative topology signature.

Only invalidated targets repeat descriptor traversal. The combined retained and
rebuilt terminal interactions are then compacted, and the global wave assignment
and subscription exchange are regenerated. Source payloads remain deduplicated
within each wave.

Particle occupancy is payload state rather than topology state. Payload headers
carry the current count. The planner reserves at least the maximum occupancy of
a stable hysteretic leaf; the source performs a collective preflight before each
wave, so no rank can enter communication with stale payload sizing.

## Memory accounting

In patch mode, `maxRemoteBytes` bounds:

- persistent LET plan and target-subplan cache storage; plus
- peak transient send, receive, wire-validation, and decoded storage for one
  wave.

A solve fails collectively if the persistent plan leaves less than the minimum
transient exchange budget. `peakRemoteBytes` reports persistent LET bytes plus
the measured transient peak.

`FmmSolveStats` additionally reports:

- local/global patch counts;
- reused patches and local plans;
- retained and released patch bytes;
- replicated descriptor, process-tree, and process-plan bytes;
- process-node and process-M2L ownership maxima and node imbalance;
- target LET subplans reused/rebuilt;
- source-triggered invalidations;
- wave-plan rebuilds and skipped descriptor traversals;
- payload-shape-triggered rebuilds.

`RICH_FMM_TRACE=1` includes these counters in `fmm_solve_trace` output. Use
`analysis_files/analyze_fmm_patch_trace.py` to summarize them.

## Deterministic process ownership

Patch leaves remain on their owning ranks. Internal process nodes are assigned
bottom-up to one of their child owners. The owner with less accumulated
process-node work is chosen; rank is the deterministic tie-break. This is a
low-cost proxy intended to remove severe node ownership imbalance. The exposed
M2L ownership statistics must be used to decide whether a later weighted owner
policy is justified.

## Regression and promotion gates

`fmm_patch_moving_mesh` covers:

- warm occupancy-only motion with process and LET reuse;
- stable particle-count changes without payload-shape rebuild;
- persistent patch-tree split and merge events;
- empty full-octant leaves transported through patch-aware descriptor
  traversal;
- incremental target-subplan reuse and source invalidation;
- patch appearance/disappearance and full process rebuild;
- bounded multi-wave LET execution;
- an empty rank;
- acceleration and potential against direct summation;
- replicated descriptor bounds and process-owner imbalance.

The default remains provisional until the target-cluster gates pass: clean MPI
regressions, the 30M warm-solve benchmark, a gravity-only replay of the TDE
snapshot, and a near-production TDE memory/performance run. These cluster gates
are not replaced by the small regression.
