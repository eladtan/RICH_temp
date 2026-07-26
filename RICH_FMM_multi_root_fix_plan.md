# RICH MPI FMM: Detailed Plan to Eliminate the Sparse-Rank LET Blowup

**Repository:** `eladtan/RICH_temp`  
**Target branch:** `ablation`  
**Repository snapshot inspected:** commit `0a22afcc6e551d855ea60e3097aa6080355a12a4`  
**Primary failure case:** `runs/M05R05MBH1e5ComptonGrayFrom148`  
**Status of this document:** implementation plan, not a code patch

---

## 1. Executive decision

The production fix should replace the current assumption

> one MPI rank = one FMM root = one process-tree leaf

with

> one MPI rank owns zero or more compact, globally aligned FMM patches, and the process tree is built over patches rather than ranks.

The new patch-based solver must also execute the Local Essential Tree (LET) in bounded waves. Multi-root geometry removes the false global near field caused by huge rank bounding boxes. Bounded LET waves prevent the union of many real local neighborhoods, owned by one spatially scattered hydro rank, from being materialized in one multi-gigabyte receive buffer.

This is the minimum redesign that addresses both parts of the observed failure:

1. **Geometry failure:** a rank with approximately 33,000 cells spread through the whole TDE domain is represented by one cube with half-size approximately 12,621, so almost every rank pair is unresolved at the rank level.
2. **Memory failure:** the same rank requests approximately 37.16 million of 37.20 million particles as P2P sources in one LET execution, producing approximately 2.17 GB of particle payload before decoder and temporary-storage overhead.

The plan intentionally does **not** rely on:

- increasing `maxRemoteBytes`;
- lowering `thetaCritical`;
- MPI-4 large-count collectives;
- message chunking alone;
- forcing every local tree leaf to a tiny physical size;
- per-particle M2P as the primary cure;
- changing expansion order or `leafCapacity`.

Those can be useful supporting changes, but none removes the rank-root pathology.

---

## 2. Current-state summary

### 2.1 Current distributed architecture

On the inspected `ablation` branch, `DistributedFmmGravityCalculator` owns exactly one:

- `FmmRootGeometry localRoot_`;
- `FmmTree localTree_`;
- `FmmLocalInteractionPlan localInteractionPlan_`;
- local multipole array;
- local local-expansion array;
- `FmmLetPlan letPlan_`.

`prepareLocalTree()` wraps all rank-local points in one dyadic root and builds one tree. `rebuildTopology()` gathers one `FmmRankRootDescriptor` per rank with `MPI_Allgather`. `FmmProcessTree` therefore has one leaf per active rank. An unresolved leaf-leaf process pair becomes a rank-level LET pair.

The current identity model is also rank-root specific:

- process leaf identity: `leafRank`;
- process lookup: `leafForRank(rank)`;
- remote local-tree node identity: `(sourceRank, spatialKey)`;
- root descriptor count: exactly one per MPI rank;
- `FmmRankRootDescriptor.rank` is expected to equal the descriptor-array index.

### 2.2 Current branch versus the experimental working tree

The GitHub `ablation` snapshot inspected for this plan still has:

- FMM packet version 3;
- one descriptor per rank;
- box-derived remote radii in the LET path;
- no patch identifier;
- no multi-root forest;
- no bounded LET-wave planner.

The investigation handoff describes later local experimental changes that may not yet be committed to `ablation`, including:

- tight node radii on the LET wire;
- detailed P2P payload accounting;
- optional physical leaf-size limits;
- optional leaf M2P;
- better pre-abort diagnostics.

Before implementing the redesign, reconcile those changes into clean, reviewable commits. Do not begin the multi-root work on top of an uncommitted experimental tree.

### 2.3 Confirmed facts that constrain the design

The new implementation must be designed around the following measured facts:

- The cell-count balance is good, but spatial compactness is extremely bad for a few ranks.
- A small number of ranks have local-root dimensions orders of magnitude larger than the median.
- Rank 0 requests approximately 99.9% of all particles as direct P2P sources.
- Tight radii and smaller physical target leaves reduce interaction-record count but do not materially reduce particle payload.
- The pathological rank has target cells in or near nearly every source-rank domain.
- A single large-count exchange would only move the failure from MPI representability to memory pressure.
- `thetaCritical = 1.0` gives the lowest measured communication among the tested values. Lowering theta is counterproductive for this run.

---

## 3. Target architecture

The target architecture has five layers.

### 3.1 Local patch partition

Each MPI rank partitions its owned particles/cells into compact **FMM patches**. Every nonempty patch has:

- a globally meaningful dyadic root cube;
- a stable patch key;
- one local adaptive `FmmTree` below that root;
- a list or contiguous range of original rank-local particle indices;
- local multipole and local-expansion storage;
- a local self-interaction plan;
- a topology signature and occupancy signature;
- an owner rank.

A rank may own zero, one, or many patches.

### 3.2 Global process tree over patches

All patch-root descriptors are gathered and a replicated process tree is built over patch roots. A process-tree leaf represents a patch, not a rank.

Leaf identity becomes:

```text
FmmPatchKey = (ownerRank, patchId)
```

where `patchId` is stable and globally unique within the owner rank. Prefer a globally derived dyadic key so that the same spatial patch naturally keeps the same ID as particles move.

### 3.3 Patch-pair classification

Every ordered target/source patch pair terminates in exactly one of:

1. process-tree M2L;
2. same-patch local self traversal;
3. same-rank cross-patch local traversal;
4. cross-rank patch LET traversal.

This replaces the current three-way rank-pair classification. The exact-one classification remains mandatory and must be tested.

### 3.4 Patch-aware LET

The LET operates on a target patch and a source patch. Remote node identity becomes:

```text
(sourceRank, sourcePatchId, sourceSpatialKey)
```

The target side also records `targetPatchId` or a compact local patch index.

The LET must never infer geometry from a rank-wide bounding box. Its starting source descriptor is the compact source-patch root.

### 3.5 Bounded-wave execution

LET topology can be built globally, but payload exchange and execution occur in one or more waves. Each wave has a strict planned upper bound for:

- outgoing encoded buffers;
- contiguous send scratch;
- received wire bytes;
- decoded multipole storage;
- decoded particle storage;
- temporary record tables;
- operator-preparation scratch.

A wave may contain one or many target patches. The planner must use descriptor particle counts to estimate payload before packing.

A single wave must fit both:

```text
configured memory budget
and
MPI int count/displacement limits
```

No wave may depend on a partial/truncated `INT_MAX` accumulator.

---

## 4. Required invariants

The implementation is not complete until all of these are enforced.

### 4.1 Particle ownership and coverage

For every solve and every local input index `i`:

- exactly one local patch contains `i`;
- no patch contains `i` twice;
- the union of patch particle ranges is the full local input set;
- acceleration and optional potential are scattered back to original input order exactly once.

### 4.2 Patch geometry

Every patch root:

- is active only if it contains at least one particle;
- is aligned to the common global dyadic lattice;
- contains all assigned particles;
- has a stable `patchId` derived from its level and lattice coordinates;
- is no larger than the configured patch-size bound unless an explicit emergency fallback is enabled;
- has a valid tight radius in addition to its cube half-size.

### 4.3 Global patch identity

The tuple `(ownerRank, patchId)` is globally unique for an epoch.

The tuple `(ownerRank, patchId, spatialKey)` uniquely identifies a local-tree node for an epoch.

No packet, cache, map, or diagnostic may continue to assume that `(sourceRank, spatialKey)` is sufficient.

### 4.4 Interaction coverage

Every ordered pair of physical source and target particles is represented exactly once by one of:

- process M2L;
- patch-level LET M2L;
- same-patch local P2P/M2L;
- same-rank cross-patch P2P/M2L;
- cross-rank patch P2P.

No patch-pair may be dropped because target and source have the same owner rank.

### 4.5 Memory safety

For every LET wave:

```text
planned receive bytes <= waveReceiveBudget
planned send bytes    <= waveSendBudget
all MPI counts        <= INT_MAX
all MPI displacements <= INT_MAX
```

If one indivisible payload record is itself too large, abort collectively with a diagnostic that names:

- source rank;
- source patch;
- source node key;
- target patch or batch;
- particle count;
- required bytes;
- configured limit.

### 4.6 Determinism

Given identical inputs and options:

- patch IDs and patch order are deterministic;
- descriptor order is deterministic;
- process-tree construction is deterministic;
- wave composition is deterministic;
- topology hashes are deterministic;
- reductions preserve the current reproducibility level.

### 4.7 Backward-compatible mode

`maxPatchLevel = 0` or `enablePatchForest = false` must reproduce the current one-root-per-rank topology for A/B testing during development.

The one-patch mode should pass all existing MPI FMM regression tests before multi-patch mode is enabled by default.

---

## 5. Proposed public options

Add the following to `FmmDistributedOptions`.

```cpp
struct FmmDistributedOptions
{
    // Existing fields...

    bool enablePatchForest = false; // initially false, later default true

    // A patch root may not be coarser than this global dyadic level.
    // Zero means one patch per rank for compatibility.
    int minimumPatchLevel = 0;

    // Hard cap on adaptive patch refinement. Must be <= FMM_MAX_TREE_DEPTH.
    int maximumPatchLevel = FMM_MAX_TREE_DEPTH;

    // Optional count target used only to split very dense coarse patches.
    // Zero disables count-driven patch splitting.
    std::size_t targetParticlesPerPatch = 0;

    // Safety cap for total local patches. Exceeding it is a collective error
    // during development; later it may trigger a documented fallback.
    std::size_t maxLocalPatchCount = 65536;

    // Maximum planned wire+decoded storage per LET wave. This should be lower
    // than maxRemoteBytes because maxRemoteBytes includes non-wave storage.
    std::size_t maxLetWaveBytes = 256ull * 1024ull * 1024ull;

    // Optional target-patch cap per wave for latency and progress control.
    std::size_t maxTargetPatchesPerWave = 64;

    // Allow same-rank cross-patch pairs to use the generic LET machinery
    // without MPI during the first implementation.
    bool useLocalPatchLet = true;
};
```

Do not add all options in one unreviewed patch. Introduce them in phases, and include them in the cross-rank consistency reductions.

### 5.1 Recommended first TDE settings

For the first production experiment:

```cpp
FmmDistributedOptions distributed;
distributed.enablePatchForest = true;
distributed.minimumPatchLevel = 7;
distributed.targetParticlesPerPatch = 0;
distributed.maxLocalPatchCount = 65536;
distributed.maxLetWaveBytes = 256ull * 1024ull * 1024ull;
distributed.maxTargetPatchesPerWave = 32;
```

Keep:

```cpp
numerical.expansionOrder = 3;
numerical.thetaCritical = 1.0;
numerical.leafCapacity = 64;
```

These are starting values for measurement, not final defaults.

---

## 6. New core data structures

### 6.1 Patch identity

Create `source/3D/gravity/fmm/mpi/FmmPatchKey.hpp`.

```cpp
struct FmmPatchKey
{
    int ownerRank = -1;
    std::uint64_t patchId = 0;

    bool operator==(const FmmPatchKey&) const;
    bool operator<(const FmmPatchKey&) const;
};
```

Provide a hash functor for unordered containers.

`patchId = 0` should be invalid. Reserve it so malformed packets are easy to reject.

### 6.2 Local patch

Create `FmmLocalPatch.hpp`.

```cpp
struct FmmLocalPatch
{
    FmmPatchKey key;
    FmmRootGeometry root;

    // Original solve-input indices assigned to this patch.
    std::vector<std::size_t> inputIndices;

    // Compact patch-local arrays.
    std::vector<Vector3D> positions;
    std::vector<double> masses;
    std::vector<std::uint64_t> cellIds;

    FmmTree tree;
    FmmLocalInteractionPlan localPlan;
    std::vector<double> multipoles;
    std::vector<double> locals;

    std::uint64_t topologyHash = 0;
    std::vector<std::uint64_t> structuralSignature;
    std::vector<std::uint64_t> occupancySignature;

    bool rootGeometryChanged = false;
    bool leafTopologyChanged = false;
    bool leafOccupancyChanged = false;
};
```

An optimized version can later avoid copying patch-local positions by using a rank-local permutation and index spans. The first correctness implementation may use copies if memory remains acceptable. For 37 million global cells, copies must be measured carefully before production.

### 6.3 Local patch forest

Create `FmmPatchForest.hpp/.cpp`.

Responsibilities:

- partition rank-local inputs into patches;
- maintain stable patch order;
- build or refit one `FmmTree` per patch;
- hold input-to-patch and patch-to-input maps;
- gather/scatter acceleration and potential;
- compute aggregate signatures;
- expose patch-root descriptors;
- report diagnostics.

Suggested API:

```cpp
class FmmPatchForest
{
public:
    FmmPatchForestChange prepare(
        const std::vector<Vector3D>& positions,
        const std::vector<double>& masses,
        const std::vector<std::uint64_t>& cellIds,
        const Vector3D& domainLower,
        const Vector3D& domainUpper,
        const FmmGravityOptions& numerical,
        const FmmDistributedOptions& distributed,
        int ownerRank);

    std::vector<FmmPatchRootDescriptor> rootDescriptors(
        std::uint64_t epoch) const;

    const std::vector<FmmLocalPatch>& patches() const;
    std::vector<FmmLocalPatch>& patches();

    void scatterAcceleration(
        const std::vector<std::vector<Vector3D>>& patchAcceleration,
        std::vector<Vector3D>& output) const;
};
```

### 6.4 Patch root descriptor

Replace or supersede `FmmRankRootDescriptor` with `FmmPatchRootDescriptor`.

Required fields:

```cpp
struct FmmPatchRootDescriptor
{
    double center[3];
    double halfSize;
    double radius;

    std::uint64_t particleCount;
    std::uint64_t topologyHash;
    std::uint64_t epoch;
    std::uint64_t patchId;

    std::uint64_t latticeId;
    std::int64_t latticeCenter[3];
    std::uint64_t latticeHalfUnits;

    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t reserved16;

    int ownerRank;
    int active;
    int rootLeaf;
    int childMask;
};
```

Keep a compatibility conversion from the current rank descriptor while one-patch mode exists.

### 6.5 Remote node identity

Extend `FmmRemoteNodeDescriptor`:

```cpp
std::uint64_t patchId;
double radius;
```

Every descriptor lookup becomes keyed by:

```cpp
FmmRemoteNodeKey {
    int sourceRank;
    std::uint64_t sourcePatchId;
    std::uint64_t spatialKey;
};
```

### 6.6 Process leaf identity

Change `FmmProcessNode` from:

```cpp
int leafRank;
```

to:

```cpp
int leafOwnerRank;
std::uint64_t leafPatchId;
std::size_t descriptorIndex;
```

`isLeaf()` should test a dedicated invalid descriptor index or invalid patch ID, not rank sign alone.

Replace:

```cpp
leafForRank(int rank)
```

with:

```cpp
leafForPatch(const FmmPatchKey& key)
```

Do not retain a unique rank-to-leaf map because multiple leaves can share one rank.

---

## 7. Patch construction algorithm

### 7.1 Initial implementation: globally fixed patch level

The safest first implementation uses one global dyadic level `Lpatch`.

For each local position:

1. Map it to integer lattice coordinates at level `Lpatch` relative to the global domain root.
2. Compute a stable Morton-style or octree path key.
3. Group particles by that key.
4. Create one patch per nonempty key.
5. Use the exact level-`Lpatch` dyadic cube as the patch root.

This guarantees compact roots and stable IDs.

Pseudocode:

```cpp
std::map<std::uint64_t, std::vector<std::size_t>> groups;
for(std::size_t i = 0; i < positions.size(); ++i)
{
    const DyadicCell cell = locateGlobalDyadicCell(
        positions[i], globalRoot, minimumPatchLevel);
    groups[cell.patchId].push_back(i);
}

for(const auto& [patchId, indices] : groups)
{
    FmmLocalPatch patch;
    patch.key = {rank, patchId};
    patch.root = geometryForGlobalDyadicCell(patchId);
    patch.inputIndices = indices;
    buildPatchTree(patch);
}
```

Use `std::map` or sort-by-key for deterministic ordering. Do not depend on unordered-map iteration order.

### 7.2 Later optimization: adaptive patch level

After fixed-level correctness is established, permit adaptive patches:

- Start at `minimumPatchLevel`.
- Split an occupied patch if:
  - `particleCount > targetParticlesPerPatch`, and
  - level < `maximumPatchLevel`.
- Never merge children belonging to different global dyadic parent cells.
- Patch IDs must encode level and path without collisions.

Adaptive splitting reduces dense-patch work without forcing sparse outer regions to remain geometrically large.

### 7.3 Patch-level hysteresis

Moving Voronoi points can cross patch boundaries every hydro step. Avoid oscillatory topology rebuilds by adding patch-level hysteresis only after the fixed-level version works.

Possible scheme:

- Patch ownership is determined by exact global dyadic cells; no geometric hysteresis is possible without overlap.
- Reuse patch objects by stable `patchId`.
- A crossing only changes the occupancy of two existing patch objects.
- Remove an empty patch only after the topology rebuild phase has collectively established that it is absent in the new epoch.
- Keep vector capacity for recently empty patches if beneficial, but do not advertise empty patch descriptors.

This is simpler and safer than allowing fuzzy patch boundaries.

---

## 8. Descriptor gathering

The current one-record `MPI_Allgather` must become a variable-count gather.

### 8.1 Correctness-first implementation

1. Each rank builds `localDescriptors`.
2. Gather descriptor counts with `MPI_Allgather` using 64-bit local counts, validating they fit the selected MPI call.
3. Compute total descriptor count and byte displacements with overflow checks.
4. Gather descriptor bytes with `MPI_Allgatherv` if total bytes fit `INT_MAX`.
5. Sort the resulting descriptor array by `(ownerRank, patchId)` and validate uniqueness.

Because process topology is replicated, this creates `O(Npatch)` storage per rank.

### 8.2 Scale guard

Add:

```cpp
std::size_t maxReplicatedPatchDescriptors;
```

and abort collectively with a useful message if exceeded.

The first target is to make the TDE run work, not to support billions of process patches. The replicated tree is acceptable only while `Npatch` remains moderate.

### 8.3 Future scalable replacement

If `Npatch` becomes too large, replace replicated patch descriptors with a distributed top-level tree. That is explicitly outside the first fix. Do not mix that larger redesign into the first patch series.

---

## 9. Process-tree changes

### 9.1 Build over descriptors, not active ranks

Change `FmmProcessTree` storage from:

```cpp
std::vector<FmmRankRootDescriptor> descriptorsByRank_;
std::vector<int> activeRanks_;
std::unordered_map<int, std::size_t> leafByRank_;
```

to:

```cpp
std::vector<FmmPatchRootDescriptor> descriptors_;
std::vector<std::size_t> activeDescriptorIndices_;
std::unordered_map<FmmPatchKey, std::size_t, FmmPatchKeyHash> leafByPatch_;
```

`buildRange(begin, end, depth)` sorts descriptor indices by geometry. Internal-node ownership may remain the owner rank of the median descriptor, but verify load concentration. A later improvement may choose owners by subtree work.

### 9.2 Tight process-node radii

Process nodes must carry a tight radius computed from child spheres, not only `sqrt(3) * halfSize`.

For a binary parent with center `c`, child centers `c1,c2`, and child radii `r1,r2`:

```text
parent.radius = max(|c1-c| + r1, |c2-c| + r2)
```

The parent cube can remain for lattice and overlap checks, but MAC decisions should use tight radii.

### 9.3 Process-pair terminal logic

Replace current rank-leaf logic with:

```cpp
if(target.isLeaf() && source.isLeaf())
{
    const FmmPatchKey targetKey = target.patchKey();
    const FmmPatchKey sourceKey = source.patchKey();

    if(targetKey == sourceKey)
        plan.localSelfPatches.push_back(targetKey);
    else if(targetKey.ownerRank == sourceKey.ownerRank)
        plan.localPatchPairs.push_back({targetKey, sourceKey});
    else
        plan.letPatchPairs.push_back({targetKey, sourceKey});
}
```

Do not collapse all same-rank leaves into one self case.

### 9.4 Routing dependencies

Dependencies are sent to owner ranks, but packet payloads must include patch identity.

For process M2L, `sourceNode` remains sufficient because the process tree is replicated and node indices are epoch-specific. For LET dependencies, include the source patch ID and target patch ID or a compact patch-pair index.

### 9.5 Exact coverage regression

Generalize `fmm_process_pair_coverage` so it expands process-node pairs to ordered patch pairs and verifies:

```text
accepted process M2L
xor same-patch local
xor same-rank cross-patch
xor cross-rank patch LET
```

for every ordered active patch pair.

---

## 10. Distributed process passes

The existing upward, process-M2L, and downward phases assume one local root per rank. Generalize them carefully.

### 10.1 Upward initialization

For every local process-tree leaf patch:

- copy the patch root multipole into the corresponding process-tree leaf coefficient slot;
- do not sum all local patch roots into one rank leaf;
- preserve patch-specific centers.

### 10.2 Internal reduction

The current process tree can still aggregate internal multipoles by owner-routed reduction. Multiple leaves with the same owner rank are valid.

Audit all code for assumptions like:

```cpp
leafForRank(rank_)
rootDescriptors_[rank_]
```

and replace them with descriptor or patch lookup.

### 10.3 Downward pass

Each local patch leaf receives its own process local expansion. Apply that expansion to the patch root local array before patch-level L2L/L2P.

The local expansion for patch A must never be applied to patch B merely because both share an MPI owner.

### 10.4 Storage layout

Use a patch-indexed vector:

```cpp
std::vector<std::vector<double>> patchMultipoles;
std::vector<std::vector<double>> patchLocals;
```

for correctness first. Later replace many allocations with flat arenas:

```text
allPatchMultipoles + offsets
allPatchLocals + offsets
```

if profiling shows allocator overhead.

---

## 11. Same-rank cross-patch interactions

This case is new and must be explicit.

### 11.1 First implementation

Reuse the LET dual-tree classification code without MPI:

- target patch tree is local;
- source patch descriptors and coefficients are local;
- M2L reads source multipoles directly;
- P2P reads source particles directly;
- no subscription exchange;
- no wire format.

Create a `FmmPatchPairPlan` that shares the same classification kernel as `FmmLetPlan`.

Avoid maintaining two subtly different MAC implementations. Refactor common logic into a traversal helper that accepts a descriptor provider.

### 11.2 Optimization

Later, fuse all same-rank patch trees into a local forest traversal to reduce repeated source walks and improve cache locality. This is optional and should not block correctness.

---

## 12. Patch-aware LET topology

### 12.1 API redesign

Replace the single-tree build API with either:

```cpp
void build(
    const FmmPatchForest& localForest,
    const std::vector<FmmPatchRootDescriptor>& globalDescriptors,
    const FmmProcessPairPlan& processPlan,
    ...);
```

or a per-target-patch plan collection:

```cpp
std::vector<FmmPatchLetPlan> patchPlans;
```

The per-patch plan is easier to batch and debug. The global wrapper can deduplicate remote source descriptors across patches.

### 12.2 Descriptor maps

Replace:

```cpp
remoteDescriptors_[sourceRank][spatialKey]
```

with:

```cpp
remoteDescriptors_[FmmPatchKey][spatialKey]
```

or one flat map keyed by `FmmRemoteNodeKey`.

The root descriptor for a source patch has `spatialKey = 1` within that patch, but patch ID disambiguates identical local keys.

### 12.3 Interaction records

Add target patch index and source patch identity.

```cpp
struct FmmLetM2LInteraction
{
    std::uint32_t targetPatchIndex;
    std::uint32_t targetNode;
    std::uint32_t sourceIndex;
    std::uint32_t geometryIndex;
};

struct FmmLetP2PInteraction
{
    std::uint32_t targetPatchIndex;
    std::uint32_t targetNode;
    std::uint32_t sourceIndex;
};
```

If the compact sizes become too large, group interactions by target patch so the patch index is implicit in an outer range table.

### 12.4 Source records

```cpp
struct M2LSource
{
    FmmPatchKey patch;
    std::uint64_t spatialKey;
    FmmNode node;
};

struct P2PSource
{
    FmmPatchKey patch;
    std::uint64_t spatialKey;
    std::uint64_t particleCount;
};
```

### 12.5 Tight radii

Carry `radius` in every patch root and remote node descriptor. Use it in:

- process MAC;
- LET MAC;
- overlap tests where appropriate;
- diagnostics.

Retain cube geometry for exact containment, lattice reconstruction, and conservative overlap checks.

---

## 13. Bounded LET-wave planner

This is required even after patch geometry is fixed.

### 13.1 Why waves are required

A spatially scattered hydro rank can own many compact patches. Each patch may have a small near field, but the union of all source particles required by all patches can still be very large. If the solver deduplicates and receives all sources for the rank in one exchange, it can recreate the current multi-gigabyte peak.

### 13.2 Planner input

For each target patch plan, compute:

- unique remote multipole source records;
- unique remote particle source records;
- estimated wire bytes by source rank;
- estimated decoded bytes;
- estimated temporary record bytes;
- estimated operator geometry bytes;
- interaction-record bytes.

Particle bytes are known from descriptor `particleCount`:

```text
sizeof(FmmPayloadRecordHeader)
+ particleCount * sizeof(FmmWireParticle)
```

Multipole bytes are known from expansion order.

### 13.3 Greedy deterministic batching

Sort target patches by stable patch key. Add patches to the current wave while:

- deduplicating sources already present in the wave;
- recomputing incremental wire and decode cost;
- respecting `maxLetWaveBytes`;
- respecting `maxTargetPatchesPerWave`;
- respecting per-peer and total `INT_MAX` limits.

If adding a patch exceeds the budget:

- close the current wave;
- start a new wave;
- retry the patch.

If the patch alone exceeds the budget, split that target patch more finely or abort with a diagnostic. Do not silently exceed the limit.

### 13.4 Wave execution

For each wave:

1. Send subscriptions for sources used in that wave.
2. Owners pack only the requested records.
3. Launch neighborhood exchange.
4. Overlap local same-patch or same-rank work where possible.
5. Validate and decode.
6. Execute M2L and P2P interactions for wave target patches.
7. Accumulate into persistent patch local arrays and acceleration arrays.
8. Release receive buffers and decoded source arrays.
9. Continue to the next wave.

### 13.5 Avoiding double counting

Each interaction belongs to exactly one target-patch plan and one wave. Source records may be resent in later waves, but interactions must not be repeated.

Maintain wave ranges over immutable interaction arrays rather than rebuilding topology every wave.

### 13.6 Generic exchange chunking

After the wave planner works, optionally add record-aware chunking inside `FmmPeerExchange`. This protects other FMM exchanges and handles a wave with many small records. It is secondary; the wave planner is still needed to bound decoded storage.

---

## 14. Packet protocol changes

Bump `FMM_MPI_PACKET_VERSION` exactly once for the multi-root patch series after the packet layout is stable.

### 14.1 Required packet fields

Add `patchId` to:

- root descriptors;
- remote node descriptors;
- descriptor requests;
- descriptor replies, directly or through the child descriptor;
- subscriptions;
- LET payload record headers;
- dependency records for patch LET.

A descriptor request becomes:

```cpp
struct FmmDescriptorRequest
{
    FmmPacketStamp stamp;
    std::uint64_t patchId;
    std::uint64_t spatialKey;
};
```

A subscription becomes:

```cpp
struct FmmSubscription
{
    FmmPacketStamp stamp;
    std::uint64_t patchId;
    std::uint64_t spatialKey;
    int kind;
    int reserved;
};
```

A payload record header also includes `patchId`.

### 14.2 Validation

Every packet validator must check:

- magic;
- protocol version;
- packet kind;
- topology epoch;
- valid nonzero patch ID;
- source owner matches actual MPI sender;
- requested node belongs to the named local patch;
- child spatial key is a child of the requested key;
- descriptor geometry matches the patch lattice metadata.

### 14.3 ABI assertions

Update all static size assertions and retain trivially-copyable checks.

Rebuild all ranks from the same binary. Old and new binaries must fail fast rather than interoperate.

---

## 15. Persistent topology and moving-mesh behavior

### 15.1 Patch-set changes

Define three independent topology-change levels:

1. **Patch set changed:** patch added or removed, or patch root geometry changed.
2. **Patch local tree changed:** node structure changed within an existing patch.
3. **Patch occupancy only changed:** counts changed but structure remained stable.

### 15.2 Rebuild policy

- Patch-set change rebuilds the process tree and all affected LET routing.
- Local-tree structural change with unchanged patch roots rebuilds only LET plans involving that patch.
- Occupancy-only change can reuse interaction plans if current invariants permit it.
- Mass-only change recomputes multipoles but reuses all topology.

### 15.3 Initial correctness policy

For the first implementation, rebuild all process and LET topology whenever the patch set changes anywhere. This is simpler and correct.

Only add incremental patch-level rebuilds after regression coverage exists.

### 15.4 Persistent local trees

Maintain one persistent topology per stable patch ID. A patch that survives to the next solve can refit using current split/merge hysteresis. A new patch initializes from scratch. A removed patch releases or parks its storage.

Do not materialize full-octant empty children solely to stabilize patch boundaries; patch boundaries are already globally fixed.

---

## 16. Diagnostics to add before enabling the new solver

### 16.1 Patch geometry

Per rank:

```text
patchCount
patchHalfSizeMin/Median/Max
patchParticlesMin/Median/Max
patchRootInflationMin/Median/Max
patchesAtEachLevel
patchSetChanged
```

Global reductions:

```text
totalPatchCount
maxLocalPatchCount
median/max patch half-size
max patches per owner rank
```

### 16.2 Process tree

```text
processPatchLeafCount
processNodeCount
processAcceptedM2LCount
samePatchPairCount
sameRankCrossPatchPairCount
crossRankLetPatchPairCount
```

### 16.3 LET waves

For each rank and global maxima:

```text
letWaveCount
maxTargetPatchesInWave
maxUniqueMultipoleSourcesInWave
maxUniqueParticleSourcesInWave
maxPlannedWireBytesInWave
maxPlannedDecodedBytesInWave
maxActualReceiveBytesInWave
maxActualSendBytesInWave
maxParticlesReceivedInWave
duplicateSourceBytesAcrossWaves
```

### 16.4 End-to-end communication

Keep existing totals, but distinguish:

```text
process bytes
LET descriptor bytes
LET subscription bytes
LET multipole payload bytes
LET particle payload bytes
same-rank patch traversal work
```

### 16.5 Pathology detector

Add a warning or optional collective abort when:

```text
rankRootHalfSize / medianPatchHalfSize > threshold
```

in compatibility mode, or when:

```text
planned rank-wide P2P payload > maxLetWaveBytes
```

before wave partitioning. This prevents future silent regressions to one giant exchange.

---

## 17. File-by-file implementation map

### 17.1 New files

```text
source/3D/gravity/fmm/mpi/FmmPatchKey.hpp
source/3D/gravity/fmm/mpi/FmmPatchForest.hpp
source/3D/gravity/fmm/mpi/FmmPatchForest.cpp
source/3D/gravity/fmm/mpi/FmmPatchPairTraversal.hpp
source/3D/gravity/fmm/mpi/FmmPatchPairTraversal.cpp
source/3D/gravity/fmm/mpi/FmmLetWavePlanner.hpp
source/3D/gravity/fmm/mpi/FmmLetWavePlanner.cpp
```

### 17.2 `FmmConfig.hpp`

- Keep numerical tree options separate from distributed patch options.
- If retaining `maxLeafHalfSize`, document it as a local-tree control, not the distributed fix.

### 17.3 `DistributedFmmGravityCalculator.hpp/.cpp`

Replace single-tree members with:

```cpp
FmmPatchForest localForest_;
std::vector<FmmPatchRootDescriptor> patchDescriptors_;
```

Generalize:

- input preparation;
- topology-change tracking;
- descriptor gathering;
- process root initialization;
- process downward application;
- LET build and wave execution;
- local traversal loops;
- output scatter;
- bytes-owned accounting;
- trace output.

Keep the current one-patch path behind an option until equivalence is demonstrated.

### 17.4 `FmmProcessTree.hpp/.cpp`

- build over patch descriptors;
- replace rank-indexed arrays and maps;
- add patch leaf identity;
- compute tight radii;
- update topology hash;
- update memory accounting.

### 17.5 `FmmProcessTraversal.hpp/.cpp`

- terminal classification by patch identity;
- same-rank cross-patch pair list;
- cross-rank patch LET dependencies;
- patch-aware diagnostics;
- exact coverage assertions.

### 17.6 `FmmLetPlan.hpp/.cpp`

- add patch identity to source and target records;
- accept a forest or per-patch plan inputs;
- refactor descriptor traversal to share logic with local cross-patch traversal;
- use tight radii;
- produce immutable topology plus wave metadata;
- separate topology build from payload-wave execution;
- preserve current progress API where practical.

### 17.7 `FmmPackets.hpp/.cpp`

- patch-aware descriptors and record headers;
- radius fields;
- version bump;
- size assertions;
- validation helpers.

### 17.8 `FmmPeerExchange.hpp/.cpp`

First patch:

- improve exact preflight diagnostics;
- expose planned per-peer send and receive sizes;
- never print a truncated total as if it were the requirement.

Later patch:

- optional record-aware multi-round exchange helper;
- retain graph communicator reuse;
- enforce per-round `INT_MAX` safety.

### 17.9 `FmmDiagnostics.hpp`

Add patch and wave metrics without removing existing fields until downstream scripts are updated.

### 17.10 Documentation

Update:

```text
docs/architecture/mpi-fmm-gravity.md
docs/regression-tests/test-catalog.md
```

The architecture documentation must explicitly state:

- process leaves are patches, not MPI ranks;
- hydro ownership and gravity geometry are decoupled;
- LET payloads are bounded by waves;
- replicated process storage scales with patch count, not rank count.

---

## 18. Regression-test plan

### 18.1 Unit-level patch tests

#### `fmm_patch_partition`

Create points in known dyadic cells and verify:

- exact patch IDs;
- exact root geometry;
- deterministic order;
- full input coverage;
- no duplicate indices;
- correct behavior on empty ranks and boundary points.

#### `fmm_patch_stability`

Move points within a patch and across one patch boundary. Verify:

- stable IDs for unchanged patches;
- one removal and one addition for crossing;
- correct topology-change classification.

### 18.2 One-patch compatibility

#### `fmm_mpi_one_patch_equivalence`

With patch forest enabled but `minimumPatchLevel = 0`:

- compare accelerations and potentials with the current solver;
- compare process-pair counts;
- compare LET M2L/P2P classifications;
- compare communication within expected packet-header differences;
- require bitwise equality where possible, otherwise a very tight tolerance.

This test must pass before true multi-patch work proceeds.

### 18.3 Multi-patch correctness

#### `fmm_mpi_multi_patch_direct`

Use a small irregular particle set distributed so each rank owns several separated clusters. Compare against direct summation.

Exercise:

- same-patch local interactions;
- same-rank cross-patch interactions;
- cross-rank patch LET;
- process M2L;
- duplicate application cell IDs;
- an empty rank.

### 18.4 Process-pair coverage

Generalize the existing coverage test to enumerate ordered patch pairs. Include:

- multiple patches on one rank;
- overlapping patch root spheres;
- empty ranks;
- unequal patch sizes;
- repeated owner ranks in different process-tree branches.

### 18.5 LET-wave tests

#### `fmm_let_wave_budget`

Set a tiny wave budget that forces many waves. Verify:

- results match a one-wave reference;
- no wave exceeds the budget;
- all MPI counts and displacements fit `int`;
- source records resent across waves do not double-count interactions.

#### `fmm_let_single_patch_over_budget`

Construct one target patch whose required payload exceeds the artificial budget. Verify a collective, informative failure.

### 18.6 Synthetic sparse-rank pathology

Create a regression case that reproduces the actual geometric failure in miniature:

- most particles concentrated in a dense core;
- one rank owns a sparse set of target particles scattered through the full domain;
- cell counts are balanced;
- the one-root compatibility mode produces a huge rank root and high P2P fraction;
- patch mode keeps process leaves compact;
- wave peak stays below the configured budget;
- acceleration matches direct summation.

This is the most important new regression. It prevents future refactors from reintroducing rank-root coupling.

### 18.7 Moving-mesh sequence

Run several solves with particles crossing:

- local tree leaves;
- patch boundaries;
- MPI ownership boundaries.

Verify topology epochs, plan reuse, and correct identity handling.

### 18.8 Existing suite

Run all current FMM regression cases, including:

- serial gravity;
- MPI guard;
- MPI direct comparison;
- process pair coverage;
- operator cache;
- scaling benchmark smoke configuration.

---

## 19. Accuracy validation

### 19.1 Direct-sum comparisons

For small and medium cases, compare:

```text
max scaled acceleration error
mean scaled acceleration error
max potential error
momentum-symmetry diagnostics where applicable
```

Run both one-patch and multi-patch modes with identical numerical parameters.

### 19.2 Patch-boundary sensitivity

Translate the same physical distribution slightly so particles cross patch boundaries while the physical problem is unchanged. Error statistics should remain within normal FMM variation and must not show discontinuous spikes at patch boundaries.

### 19.3 Theta and order sweeps

Repeat existing order/theta sweeps in multi-patch mode. Confirm:

- expected monotonic communication trend with theta;
- no unexpected error degradation from process-patch decomposition;
- no dependence on number of patches beyond ordinary roundoff and approximation-order effects.

### 19.4 Conservation and self-interaction

Verify exact self exclusion still uses solver-owned identity:

```text
(ownerRank, ownerLocalIndex)
```

Patch reassignment must not alter identity or accidentally remove interactions between different particles with duplicate application IDs.

---

## 20. Performance and memory acceptance criteria

The first production-ready patch is accepted only if all of the following hold.

### 20.1 TDE restart

On the original 37.2 million-cell, 1152-rank restart:

- no abort 91 or 92;
- first gravity solve completes;
- maximum actual LET wave receive storage remains below the configured wave budget;
- no MPI count/displacement exceeds `INT_MAX`;
- no rank requests one monolithic approximately 2 GB payload;
- node memory remains below node allocation limits with 192 ranks per node.

### 20.2 Communication

Targets for the first pass:

- maximum per-wave particle payload <= 256 MB;
- median per-wave payload substantially below 64 MB;
- no more than a factor of 2 increase in total LET bytes relative to the theoretical unbounded patch-plan payload;
- repeated source transmission across waves is reported explicitly.

Total bytes may initially increase because bounded waves resend sources. Peak memory reduction is the first priority.

### 20.3 Runtime

On the existing 30 million-particle benchmark:

- one-patch compatibility mode: within 5% of current warm runtime;
- multi-patch mode on a uniform distribution: within 15% initially;
- no 49% regression from M2P in default mode;
- topology build cost and wave count reported separately.

### 20.4 Patch count

For the TDE case, record:

- global patch count;
- max local patch count;
- process-tree node count;
- replicated process-tree memory.

If the process tree becomes the dominant memory consumer, reduce patch level or proceed to a distributed top-level tree. Do not hide the cost.

---

## 21. Phased implementation sequence

### Phase 0: clean baseline and preserve diagnostics

**Goal:** establish a reproducible starting point.

Tasks:

1. Reconcile experimental working-tree changes against `ablation`.
2. Commit tight wire radii and payload accounting separately.
3. Keep leaf M2P disabled by default.
4. Keep the physical leaf-size control optional and clearly documented as non-fundamental.
5. Add exact untruncated planned-byte diagnostics in `FmmPeerExchange`.
6. Reproduce the TDE failure and the 30M control benchmark.

Exit criterion:

- clean Git history;
- reproducible baseline numbers;
- no stale run-directory binary.

### Phase 1: introduce patch identity with one patch per rank

**Goal:** change identities without changing geometry.

Tasks:

1. Add `FmmPatchKey`.
2. Add patch ID fields to internal structs, initially fixed to one ID per active rank.
3. Generalize maps from rank key to patch key.
4. Keep one root and one tree per rank.
5. Update packet protocol and regressions.

Exit criterion:

- one-patch mode matches current results and performance closely.

### Phase 2: add local patch forest

**Goal:** build multiple fixed-level patch trees locally, but do not yet execute distributed interactions.

Tasks:

1. Implement deterministic fixed-level partition.
2. Build/refit one tree per patch.
3. Add coverage and geometry tests.
4. Add aggregate diagnostics.
5. Verify gather/scatter to original input order.

Exit criterion:

- local forest results match a serial reference when all patch-pair interactions are evaluated locally.

### Phase 3: process tree over patches

**Goal:** replace rank leaves with patch leaves.

Tasks:

1. Variable-count descriptor gather.
2. Generalize `FmmProcessTree`.
3. Generalize pair coverage.
4. Generalize process upward/downward passes.
5. Add same-rank cross-patch terminal class.

Exit criterion:

- all ordered patch pairs classified exactly once;
- small MPI cases match direct summation.

### Phase 4: patch-aware LET without waves

**Goal:** correctness of distributed patch traversal.

Tasks:

1. Patch-aware node descriptors and subscriptions.
2. Cross-rank patch LET.
3. Local cross-patch traversal.
4. Packet validation and protocol bump.
5. Direct-sum regression.

Exit criterion:

- sparse-rank synthetic test is accurate;
- monolithic payload may still be large, but geometry metrics improve.

### Phase 5: bounded LET waves

**Goal:** guarantee bounded peak memory and MPI representability.

Tasks:

1. Implement preflight byte estimates.
2. Implement deterministic wave planner.
3. Execute and release one wave at a time.
4. Add wave budget tests.
5. Add per-wave diagnostics.

Exit criterion:

- synthetic pathology completes under a tiny enforced budget;
- TDE first gravity solve completes under the production budget.

### Phase 6: topology reuse and optimization

**Goal:** recover performance.

Tasks:

1. Reuse patch objects by stable ID.
2. Rebuild only affected patch LET plans.
3. Arena-allocate per-patch coefficients.
4. Deduplicate common sources across adjacent waves where budget permits.
5. Tune patch level and batch size.
6. Profile same-rank cross-patch work.

Exit criterion:

- uniform benchmark overhead is acceptable;
- moving-mesh warm solves reuse most topology.

### Phase 7: default enablement

**Goal:** make patch mode the normal MPI implementation.

Tasks:

1. Enable patch forest by default after production validation.
2. Retain one-root mode as a diagnostic fallback for one release cycle.
3. Update architecture documentation and test catalog.
4. Remove obsolete rank-root-only code after no downstream user depends on it.

---

## 22. Recommended commit series

Keep the review surface small. Suggested commits:

1. `fmm: preserve exact LET payload diagnostics`
2. `fmm: carry tight radii in distributed descriptors`
3. `fmm: add patch identity in one-root compatibility mode`
4. `fmm: add deterministic local patch forest`
5. `fmm: gather variable patch-root descriptors`
6. `fmm: build process tree over patch leaves`
7. `fmm: classify same-rank cross-patch pairs`
8. `fmm: generalize process upward and downward passes`
9. `fmm: make LET descriptors patch-aware`
10. `fmm: execute local cross-patch traversal`
11. `fmm: add deterministic LET wave planner`
12. `fmm: execute bounded LET payload waves`
13. `test: add sparse-rank FMM pathology regression`
14. `docs: document patch-based MPI FMM architecture`
15. `perf: reuse patch topology across moving-mesh solves`

Do not combine identity changes, process-tree changes, packet changes, and wave execution in one commit.

---

## 23. Risk register

### Risk 1: patch count becomes too large

**Symptom:** replicated process-tree storage and topology time dominate.

**Mitigation:**

- start with a modest fixed patch level;
- report patch-count distributions;
- cap local/global descriptors;
- add adaptive coarse patches in dense regions;
- later distribute the process tree if necessary.

### Risk 2: total communication remains high

Even compact patches cannot eliminate true near-field work for targets scattered throughout the domain.

**Mitigation:**

- waves bound memory even if total bytes remain high;
- tune gravity-aware hydro repartitioning later;
- consider temporary gravity-target redistribution as a future optimization;
- cache recently used remote source patches across adjacent waves if memory permits.

### Risk 3: same-rank cross-patch interactions are missed or duplicated

**Mitigation:** exact ordered patch-pair coverage regression and direct-sum tests.

### Risk 4: moving points cause excessive patch churn

**Mitigation:** stable fixed dyadic patch IDs, object reuse, occupancy-only detection, and incremental rebuilds after correctness.

### Risk 5: packet protocol complexity

**Mitigation:** one protocol bump, strict size assertions, centralized key serialization, collective validation tests.

### Risk 6: performance regression on uniform cases

**Mitigation:** compatibility mode, patch-level auto-disable when one compact root is already sufficient, and delayed optimization after correctness.

A useful auto-disable criterion is:

```text
local one-root half-size <= compactnessThreshold * expected local spatial scale
```

but do not add this heuristic until multi-patch correctness is stable.

### Risk 7: patch boundaries alter numerical error

**Mitigation:** fixed MAC, tight radii, translation tests across patch boundaries, and order/theta sweeps.

---

## 24. Alternative long-term architecture

A separate globally aligned gravity mesh, closer to the RAMSES design, remains the cleanest long-term solution:

1. deposit Voronoi-cell masses onto a sparse global dyadic gravity hierarchy;
2. solve gravity on that hierarchy;
3. interpolate acceleration back to Voronoi generators.

This fully decouples gravity geometry from hydro ownership and naturally supports bounded neighbor stencils. It is, however, a larger physics and discretization change because deposition/interpolation and force accuracy must be validated.

The patch-forest plan in this document is recommended first because it:

- preserves the existing particle/cell FMM kernels;
- preserves direct force evaluation between actual Voronoi cell masses;
- reuses most local-tree and Taylor-expansion code;
- directly addresses the observed failure;
- can be introduced incrementally with a one-patch compatibility mode.

If patch count or total communication remains unacceptable after Phase 6, the global gravity mesh should become the next project rather than further complicating rank-owned particle LETs.

---

## 25. Immediate next actions

1. Create a clean development branch from `ablation`.
2. Port and commit the proven tight-radius and true-payload diagnostics.
3. Add the synthetic sparse-rank regression before changing the architecture.
4. Implement `FmmPatchKey` and one-patch compatibility mode.
5. Implement a fixed-level local patch forest with `minimumPatchLevel = 7`.
6. Measure patch count on the TDE snapshot before modifying the process tree.
7. If global patch count is manageable, proceed with the patch process tree.
8. Implement bounded waves before attempting a full TDE production run.
9. Run the 30M uniform control after every major phase.
10. Do not enable M2P or lower theta as a workaround for failed intermediate versions.

---

## 26. Definition of done

The RICH MPI FMM sparse-rank problem is fixed when:

- the TDE restart completes its first and subsequent gravity solves at 1152 ranks;
- no rank-wide bounding box is used as a process-tree leaf when that rank owns multiple distant patches;
- peak LET memory is bounded by configuration rather than by the union of all rank-local target neighborhoods;
- all MPI exchanges are representable by the available MPI-3 `int` interface;
- results agree with direct summation and existing FMM accuracy baselines;
- uniform benchmarks retain acceptable performance;
- moving-mesh topology reuse works at patch granularity;
- the synthetic sparse-rank regression remains permanently in the test suite.

