# MadVoro MPI and ghost construction audit

Audit date: 2026-09-07. Read-only inspection of the current, already dirty workspace. Paths below are relative to `/home/maorm/RICH`; line numbers describe this inspected working tree. No controlled performance benchmark was run. All proposed runtime contributions are conditional planning estimates, not measured speedups. Do not add them together: most affect the same ghost-construction time.

## Executive findings

1. The distributed ghost loop currently invokes **19 MPI collectives per loop iteration**, before counting any additional collectives in called application code. Four only produce diagnostic output. Fourteen belong to two query-batch setup/teardown paths. One decides whether everybody has no queries. The first iteration always has an empty large-query batch, but pays its full protocol cost. The final globally empty iteration executes both protocols before stopping.
2. This is not merely a "use nonblocking MPI" opportunity. Queries already use `MPI_Issend`, `MPI_Irecv`, and cooperative polling. The expensive protocol repeatedly initializes and drains globally synchronized batch machinery, even with no payload.
3. The radius growth factor is 1.1. If the required coverage radius is a factor F above the current radius and no topology changes alter it, growth alone costs roughly `ceil(log(F)/log(1.1))` iterations. F=10 means 25 growth steps; F=100 means 49. This can multiply all other MPI overhead. The radius is a correctness certificate: do not remove the final geometric completion check.
4. Point payloads are copied through several representations although the caller only needs a count per original query and one ordered point vector per sender. Remove these copies before complex communication overlap.
5. There are correctness prerequisites for aggressive changes: partial-build tree indexing, ordering after request completion, explicit protocol epochs/tags, and preservation of `(physical point, periodic image)` deduplication. A faster invalid mesh is not a performance improvement.

## Observed call path and exact collective count

`Voronoi3D::BuildPartiallyParallel` prepares/migrates points, builds local Delaunay, builds local range tree, initializes radii, then calls `BringGhostPointsToBuild`; afterward it constructs Voronoi cells, synchronizes centroids, and exchanges volumes. Sources: `source/3D/tessellation/voronoi/Voronoi3D.hpp:2176`, `:2198`, `:2233`, `:2263`, `:2267`, `:2288`, `:2299`, `:2310`, `:2315`.

Within one distributed ghost iteration:

| Operation | Calls | Evidence |
|---|---:|---|
| Sum small-point count to rank 0 for printing | 1 Reduce | Voronoi3D.hpp:3710 |
| Sum large-point count to rank 0 for printing | 1 Reduce | Voronoi3D.hpp:3711 |
| Sum current stored-point count for printing | 1 Reduce | Voronoi3D.hpp:3718 |
| Global no-query test | 1 Iallreduce | Voronoi3D.hpp:3768 |
| Large query batch | 7 collectives | See below |
| Small query batch | 7 collectives | See below |
| Sum newly received ghosts for printing | 1 Allreduce | Voronoi3D.hpp:3835 |
| **Total** | **19** | Excludes point-exchange/setup outside loop |

Each `BuffersManagerQueryAgent::runBatch` does:

* `AmountManager` construction: Barrier, `source/utils/mpi_utils/AmountManager.cpp:69`.
* `AmountManager::Initialize(1)`: Reduce, `AmountManager.cpp:74`, called at `queryAgent/BuffersManagerQueryAgent.hpp:235`.
* Request-buffer Destroy: Reduce_scatter + Barrier, `BuffersManager.hpp:160`, `:210`.
* Answer-buffer Destroy: Reduce_scatter + Barrier, same code.
* Explicit final Barrier: `queryAgent/BuffersManagerQueryAgent.hpp:328`.

The two buffers are explicitly destroyed at `BuffersManagerQueryAgent.hpp:324`, `:325`; this is not merely an inference about destructor timing. In addition there is a tree of done, verification, and verification-response messages (`AmountManager.cpp:143`, `:150`, `:167`, `:225`). All batches take this path even when a rank has no queries, and even when every rank has no queries.

Do not infer "19 collectives means 19 equal latencies." Algorithms, input size, arrival skew, MPI implementation, and overlap differ. Count is evidence for an optimization target, not a speedup measurement.

The printed "average ghost points" includes `del_.points_.size()`, thus owned points and the four bounding points as well as ghosts (`Voronoi3D.hpp:3715`), and the printed "total big queries" accumulates large *points*, not actual tetrahedron/periodic query records (`:3722`, `:3299`–`:3330`). These labels should not be used directly as communication-volume metrics.

## Performance-contribution estimates and how to use them

Let `T` be max-rank elapsed MadVoro build time, not rank-0 time and not sum of rank times. Let `f_X` be the measured fraction of `T` spent in the affected phase on its critical path; if that phase is reduced by fraction `r`, first-order build reduction is `f_X*r`. There is no universal percentage independent of workload.

The following deliberately broad positive ranges apply only when the trigger in the third column is verified. A result can be 0% or a regression when the trigger is absent. The actual baseline may already contain other proposals, so remeasure after each accepted change. "MPI dominated" means the ghost protocol's synchronization, polling, and data movement are a substantial fraction of the build. "Compute dominated" means local geometry/tree work dominates.

| ID / change | Estimated build-time reduction: MPI dominated / compute dominated | Trigger, dependency, overlap |
|---|---|---|
| M0: counters and trustworthy timing | 0% / 0%; instrumentation target under 1% overhead when enabled | Prerequisite for all estimates, not claimed as a speed optimization. |
| M1: disable hot diagnostic reductions and output | 3–20% / 0–3% | Repeated short MPI rounds. Overlaps M2/M3/M8 because their round count/protocol reductions remove some of same cost. |
| M2: global empty-type/final-round skipping | 3–18% for 3–12-round builds / 0–3% | Savings are proportional to skipped batches, not a universal constant. In 80+ rounds, often below 2% unless many rounds have only one query type. Depends on collective branch agreement. |
| M3: replace or amortize query batch protocol | 15–55% / 0–10% | Setup/teardown/global waiting shown to dominate. Largest MPI-specific candidate, highest protocol risk. Evaluate on baseline after M1/M2; does not add to their original-baseline savings. |
| M4: tune buffering, receive slots and tail flush | 2–15% / 0–3% | Poll/spin time or receive serialization visible; parameter search may regress. Alternative or refinement to M3, not an independent guaranteed gain. |
| M5: single payload storage plus counts | 1–10% / 0–4% | Ghost-heavy builds or many allocated answer vectors. May matter more than these ranges for memory-bandwidth-limited cases. Overlaps M3 packing. |
| M6: O(1) rank-to-slot lookup, sparse metadata | 0–8% / 0–2% | Hundreds/thousands of ranks or many correspondent lookups; usually below 1% at small P. Overlaps routing/backend rewrites. |
| M7: avoid all-rank distance cache per big point | 0–15% / 0–3% | `askOnlyClose` routing and `B*P` cache consume time/memory. Needs routing counters; overlaps load-balancer/environment work covered by other audit. |
| M8: fewer radius-growth rounds | 10–60% / 0–15% | Long low-yield 1.1-growth sequences. Can be 0% or negative when warm radii are already accurate or larger spheres add too many ghosts. Overlaps every per-round optimization. |
| M9: previous-neighbor ghost warm start | 5–30% on repeated mild-motion builds / 0–8% | No benefit on first build; invalidation/migration tracking required. Can regress on large motion/repartitioning. Existing radius history already supplies some warm-start benefit. |
| M10: overlap local self-query work with remote service | 2–15% / 0–5% | Both local work and remote waiting substantial; requires M3's explicit phase/epoch design. Estimate bounded by `min(T_local,T_remote)/T`. |
| M11: known-size centroid/volume exchange | 1–8% / 0–2% | Many tiny repeated builds/large rank counts. Profile separately from ghost query path. Can reuse a common implementation developed for M3. |
| M12: remove supplied-tree point copy | 0–3% / 0–3% | UpdateRangeFinder's one full local-point copy is measurable; primarily memory benefit. Depends on immutable lifetime ownership. |
| C0: partial-build indexing fix | 0% speedup claimed | Correctness prerequisite, may change runtime either way. |
| C1: request-completion compaction fix | 0% speedup claimed | Portability/correctness prerequisite for buffer experiments. |
| C2: protocol state/tag/lifetime hardening | 0% speedup claimed | Enables safe removal of collectives and concurrency. |

For a concrete conditional example, if instrumentation shows 50% of build time in batch setup/teardown and M3 removes 70% of that cost, estimate a 35% build-time reduction and a `1/(1-.35)=1.54x` build speedup. If MadVoro is 20% of whole RICH runtime, the whole run improves by only 7% before secondary effects. Report both denominators explicitly.

## M0: add measurements that answer which proposal matters

Add per-build/per-round local counters and timers, accumulated in memory and reduced only at the end of a build or benchmark. Do not add another collective around every timer. Required counters:

* Owned active points, all locally owned points, remote ghost images, local periodic images, mirror points, live/dead/new/changed tetrahedra.
* Base small queries, base big queries, periodic-expanded records, rank fanout histogram, empty/nonempty responses, unique returned points/images, bytes/messages per peer and response-size histogram.
* Query creation/routing, local octree answering, packing, request/response progress, time from local submission complete to local responses complete, time from local completion to global completion, each setup/teardown component, and Delaunay insertion time.
* Peak actual buffer capacity, pending-buffer count, outstanding send count, poll cycles with no work, getTalkList time, and closest/furthest cache entries/bytes.
* Count global rounds, globally empty type calls, final globally empty rounds, and reason a point stays active (radius too small, initial near-only big query, nonempty big reply).

Use max/mean/p95 rank elapsed times, but do not sum independently maximized phase times as if they identify one real rank's critical path. Log timing records only after the timed region. Use `MPI_Wtime` or monotonic local clocks for elapsed local durations, not comparisons between unsynchronized absolute clocks. Fix labels described above. Tie every artifact to superproject/submodule SHAs, dirty patch hash, compiler flags, MPI version, P, nodes, affinity, point counts/seed, boundary mode, and build number.

## M1: remove diagnostic collectives from the default fast path

In `BringGhostPointsToBuild`, gate the three Reduce calls at `:3710`, `:3711`, `:3718` and the Allreduce at `:3835`, together with corresponding output, behind a configuration flag that is identical across ranks. Default quiet mode must execute none of them. Keep rank-local counters available independently.

At `BuildPartiallyParallel:2205`–`:2211`, max/min point-count Allreduces also only feed diagnostics and should use the same policy. Coalesce the two build-policy Allreduces at `PrepareToBuildParallel:2015`–`:2020` into a length-two logical-AND collective if profiling shows any material contribution; preserve each policy flag exactly. This subchange's contribution is ordinarily below 1% build time, included within M1's range rather than additional savings.

Never condition an MPI collective on a rank-local verbosity setting. Validate flag uniformity outside the timed loop (for example, broadcast the chosen rank-0 configuration once). Tests must execute both diagnostic modes at P=1,2,3,4 with empty ranks and compare geometry/state, not printed formatting.

## M2: exact safe pseudocode for empty work

The present code starts the global empty test at `:3768` but waits at `:3854`, after all work, and only then breaks. A straightforward implementation:

```text
while true:
    small, big = CreateBatches(...)  // includes current checkBig handling
    mirrors = make_mirrors(small,big)
    small,big = expand_periodic_if_needed(small,big)
    local_mask = (!small.empty ? 1 : 0) | (!big.empty ? 2 : 0)
    global_mask = Allreduce(local_mask, MPI_BOR, build_comm)
    if global_mask == 0:
        break

    initialize_result_counts_for_LOCAL_queries()
    perform_local_self_queries_if_required()
    if global_mask & 2:
        large_batch = big_agent.runBatch(big)  // called even if LOCAL big is empty
        append_ghost_mapping_then_insert_points(large_batch)
    if global_mask & 1:
        small_batch = small_agent.runBatch(small)
        append_ghost_mapping_then_insert_points(small_batch)
    insert_mirrors()
    update_changed_tetrahedron_state()
    determine_next_iteration_points()
```

This collective replaces the existing Iallreduce; do not leave both. The simpler blocking form can lose some useful overlap and must be timed. A follow-up form can start `MPI_Iallreduce` on the mask, run local self-query work, wait, then skip remote batch types. In that form, globally empty iterations still have no self query work, and nothing needs insertion. All ranks must take the same global-type branches.

Why this is safe: zero globally expanded queries means no rank has an outstanding new ghost-search requirement for that round. Every previous batch has completed and drained under the existing protocol. Mirrors are derived exclusively from these base queries, so the zero-query final round creates no new mirrors. `CreateBatches` can yield no queries even when a large-point set has entries whose incident tetrahedra have `checkBig=false`; therefore test actual query vectors, not only point-set emptiness.

Do not stop after a round with zero *received* ghosts. A small point may still need a larger radius (`DetermineNextIterationPoints:3536`–`:3547`). Do not skip a local empty batch when another rank has queries: it must answer incoming requests. Preserve near-only first large iteration behavior (`:3304`, `:3559`): it deliberately forces a later unrestricted round.

Validation: all empty, one nonempty rank, globally only small, globally only big, ranks disagreeing on their local type, many final radius expansions returning zero points, mixed periodicity, and multiple consecutive builds. Count expected removed batch calls: first round always has no big queries (`CreateBatches:3274`–`:3295`), final all-empty round has none of either type.

## M3: redesign batch boundaries before adding more overlap

Observed: request/answer buffers are constructed anew each batch (`BuffersManagerQueryAgent.hpp:210`, `:219`), the finish manager is recreated (`:233`), and completion requires both counter-tree work and collective drain verification. This is the principal MPI-specific structural target.

Implement one conservative experimental backend first: **known-message two-phase batches**, preserving big-before-small order and the existing `TalkAgent` and `AnswerAgent` behavior. Keep the old backend selectable for differential testing.

```text
build per-destination request records (parent_query_id, serialized query)
exchange request byte counts; allocate exact incoming buffers
post ALL known request receives, then bounded request sends; complete requests
for each received sender segment in documented order:
    for each request record:
        answer = existing AnswerAgent::answer(query, sender)
        append response record (parent_query_id, answer_count, answer_points)
        // include a zero-count response for every empty answer
exchange response byte counts; post receives and bounded sends; complete them
validate every issued (destination,parent_query_id) has exactly one response
construct per-query counts and stable per-sender received point order
return only after all buffers referenced by MPI requests are complete
```

Initially this needs two count collectives per query type, instead of seven setup/teardown collectives plus the done tree, but may sacrifice query/answer streaming and increase peak memory. That tradeoff is measurable. It is a valid intermediate baseline, not automatically the final preferred backend. At large P, dense count Alltoall overhead can become dominant; then introduce a sparse unknown-source protocol or cached directed graph with an escape route for newly discovered destinations. Do not jump straight to neighbor-only communication: ghost-search spheres can discover nonprevious neighbors.

**Do not blindly implement this using the current `MPI_Exchange_sparse_by_rank` and assume it is fully concurrent.** It exchanges counts with MPI_Alltoall, then iterates P-1 ring steps, posting and waiting inside each step (`source/utils/mpi_utils/mpi_alltoall.hpp:204`, `:240`–`:289`, and serialized branch `:321`, `:348`–`:400`). A local sparse pattern does not make this an O(neighbor-degree) protocol. Add a dedicated helper that posts all known receives before bounded sends or uses a bounded peer window without matching deadlocks.

A second path, only if measurements favor streaming, retains persistent request/answer buffers across epochs. It needs explicit epoch state and proof of no messages in flight. Do not remove `Destroy` collectives and cancellation checks without replacing the proof they supply. Create the build communicator and receive pool once, not per query. Maintain a small bounded high-watermark buffer pool and reclaim only after requests complete.

Combining small and large query messages in one epoch may remove another synchronization boundary but is a later experiment. Both share `SentPointsContainer`; changing service order can change which query receives credit for a point, whether a small query reaches its cap, and how later large rounds proceed. Require a correctness argument and differential geometry tests before fusing, and include per-kind counts in the wire record. Shared tags 200/201 and finish tags 9918–9921 make naively running the existing agents concurrently invalid: the record layouts differ.

## Protocol invariants required for M3/M4/M10

1. **One response per subquery**, including zero-length answers. A rank can stop submitting while it continues answering others. Local completion does not mean global completion.
2. **Exactly one local finish contribution** per epoch. Do not double-decrement `AmountManager` when progress executes repeatedly after local completion.
3. **No borrowed send memory changes** before its request completes; vector reallocation is a mutation. Stable receive buffers remain valid through Irecv completion and callback extraction.
4. **Pending serialized buffers count as pending work.** Existing `CountOutcoming()` reports `sendRequests.size()` only (`BuffersManager.hpp:32`); `GetPendingNumber()` reports `ranksSendBuffers.size()` (`:34`). A new termination test must account for both, inbound work queues, answer queues, and sent-versus-received logical responses. The current call site only checks outgoing request count at `BuffersManagerQueryAgent.hpp:287`; its broader protocol invariant is stronger than that predicate alone. Do not copy that predicate as a complete quiescence proof.
5. **No old-epoch receive consumes a next-epoch message.** Use communicator contexts or explicit epoch identifiers with buffering, and do not reuse an epoch tag until outstanding sends/receives have been settled. MPI_Barrier alone is not a message flush.
6. **Preserve ghost mapping order.** For each peer `j`, `duplicated_points_[j][k]` and the peer's corresponding `Nghost_[...][k]` must refer to the same physical point/image. `SetGhostArray` computes indices from current `del_.points_.size()` (`Voronoi3D.hpp:1987`); call before insertion and account for any earlier concatenated segment offset.
7. **Periodic identity is `(source point, image code)`**, not coordinates alone and not source point alone. `SentPointsContainer` has 27 sets per peer (`range/SentPointsContainer.hpp:17` and `RangeQueryData.h:11`); zero-image index is 13. A physical point may legitimately occur multiple times in the exported point-index array for different periodic images.
8. **No global completion on idle ranks only.** Empty ranks still initialize bounding tetrahedra (`Voronoi3D.hpp:2218`) and serve remote requests. Test a rank owning no points but participating in every required epoch.
9. **Serialization bounds use actual bytes**, not `sizeof` of C++ objects with vtables/vectors. Check MPI int limits or chunk messages. Current BuffersManager appends a value before checking capacity (`:287`–`:297`); generic larger answers need preflight/fragmentation or a sized-message route.
10. **Threading is a separate project.** `BuffersManager::Receive` and CleanSendRequests use static scratch vectors (`:216`–`:218`, `:349`–`:350`); environment trees can use mutable traversal scratch. Adding OpenMP tasks or an MPI progress thread requires eliminating shared mutable scratch and checking MPI thread support. Nonblocking calls alone do not provide thread safety or guaranteed asynchronous progress.

## M4: buffering and cooperative progress

The current answer receive buffer assumes 1024 points per average answer, multiplied by 64 (`BuffersManagerQueryAgent.hpp:218`–`:219`). For an ordinary 24-byte double-coordinate point this is about 1.5 MiB of receive storage, while small queries request at most 16 points and large queries request at most one per destination. Each manager starts with just one wildcard receive (`:210`, `:219`). This is an observed heuristic mismatch, not proof that 1.5 MiB itself dominates.

Measure real serialized size histograms and then parameterize byte targets, receive slots, maximum inflight buffers, and tail flush. Test candidate byte targets 4/16/64 KiB and receive slots 1/2/4/8 on the target cluster; treat these as a search grid, not prescribed defaults. One value must never exceed configured message capacity silently. Repost completed receive buffers promptly, then process callbacks with a bounded work quota so one large receive cannot starve answers or outgoing progress.

Current dispatch waits on loop counts: queries use 1024 and answers 50 (`:210`, `:219`; `BuffersManager::ShouldSend:320`). CPU/compiler/network speed therefore changes real delay. Add explicit FlushPending when local submission ends, and a bounded wall-time deadline for small tail buffers while remote work continues. Preserve batching in the steady state. Do not replace `MPI_Issend` with `MPI_Isend` and credit its completion as proof of remote receipt: standard-send completion can mean only a copied eager buffer.

## M5/M6/M12: payload and metadata overhead

At receive time, each answer's points are appended both to `queries[id].finalResults` and `batch.dataByRanks[rank]` (`BuffersManagerQueryAgent.hpp:120`–`:121`). Rearrangement copies points to a third `queriesBatch.result` (`:181`–`:188`). BringRemoteGhostPoints copies those into `newPoints` (`Voronoi3D.hpp:3472`, `:3487`), then Delaunay copies them. The caller reads per-query `finalResults` only to get `.size()` (`:3470`, `:3483`).

Add a MadVoro-specific batch output policy or specialized result type that stores `answer_counts[parent_id]` plus one point segment per sender. Do not globally break QueryAgent consumers that need the full per-query result. Record counts after actual sent-point deduplication, not before; they drive small/large decisions. Reserve the aggregate output once using the sum of peer segment sizes. Existing `rearrangeResult` reserves `result.size()+rankRecvData.size()` although rankRecvData was cleared at batch start (`:184`, `:198`–`:200`); use actual newDataFromRank size or the full sum.

Use a rank-to-slot vector initialized to a sentinel once per build if P is moderate, or a reserved hash map if metadata must be sparse. SentPointsContainer currently linearly searches peer ranks both when retrieving the ignore set and adding returned points (`range/SentPointsContainer.hpp:72`, `:109`, `:145`); receive finalization scans all ranks (`BuffersManagerQueryAgent.hpp:305`) and then linearly searches existing peers (`:315`). Preserve append order of peer arrays while changing lookup. Preserve 27 image sets; do not deduplicate the flattened `sentData` across images.

OctTreeFinder's external-tree constructor copies `myPoints` (`range/finders/OctTreeFinder.hpp:20`, private vector `:57`) although Voronoi's `allMyPoints` is immutable throughout BringGhostPointsToBuild and already owns the indexed coordinates. Introduce an explicit borrowed/shared immutable storage variant, while keeping ownership for iterator constructors. Inspect Voronoi copy-constructor semantics before using a raw span/reference; a copy must not retain a view into another object's later-mutated storage (`Voronoi3D.hpp:4996`). Rebind or share an immutable owning snapshot.

## M7: large-query rank-distance cache

`BigRangeTalkAgent::getTalkList` caches a full `vector<pair<coord_type,coord_type>>` of length P for every large point that uses near-rank routing (`range/BigRangeAgent.hpp:192`–`:198`, `:250`). With doubles this is at least `16*B*P` payload bytes, before map/vector overhead. At B=100,000 and P=1,024, the payload alone is about 1.6 GB per rank. This is a conditional size calculation, not a claim that production B has that value.

HilbertRectangularTree3D's implementation allocates P entries and traverses all leaves to compute bounds (`MeshDecomposer3D/hilbert/rectangular/HilbertRectangularTree3D.hpp:429`, `:437`–`:475`). Other environment types may implement different costs. Profile current selected environment and cache bytes before changing.

Conservative staged alternative: compute conservative lower/upper distance bounds once per rank from its aggregate AABB, with geometry stored O(P), then evaluate only intersecting ranks for a query. A looser aggregate bound is allowed to overselect ranks but must never exclude a rank the exact geometry could require. Use the maximum corner distance to guarantee an upper bound; do not substitute centroid distance. If boxes are too loose and communication increases, use a compact per-rank box hierarchy or candidate-only exact distance API. A sparse cache still must include every intersecting candidate and be invalidated when the decomposition changes. The existing cache is scoped to a BigRangeAgent recreated per build; do not accidentally make it persistent without a decomposition version.

## M8: radius policy with explicit correctness constraints

Default settings: `RANGE_MAX_POINTS_TO_GET=15`, returned cap16, `RADIUSES_GROWING_FACTOR=1.1`, large-radius history shrink0.95 (`Voronoi3D.hpp:99`–`:101`, `:3285`, `:3532`). Initial uninitialized radii use local nearest-neighbor distance (`:3214`–`:3218`); this can be a poor cold-start predictor for anisotropy, density jumps, or voids. Existing radii are retained across builds; do not describe every build as cold.

Measure per-round radius/current-maximum-circumradius ratio, zero-result streak, query fanout, and ghost overfetch. Experiment with growth factors 1.25, 1.5, 2.0, independently from the result cap. F=100 requires about 21,12,7 steps respectively instead of49 for1.1. This is a mathematical growth comparison only: topology changes, result saturation, and overfetch can invalidate any naive runtime prediction.

A bounded adaptive policy can grow faster only after repeated low-yield rounds, then return to conservative growth when result caps are reached. Suggested first experiment: retain factor1.1 default; selectable factor1.5 after two consecutive rounds with no new result for a point and an insufficient completion radius; reset the streak on nonzero results, small-to-large transition, new build/migration, and point identity changes. Do not jump directly to `2*GetMaxRadius` when an incident tetrahedron includes artificial bounding vertices: its radius can be huge and cause global flooding. Always retain the exact existing test `currentRadius >= 2*GetMaxRadius` for small-point completion and the unrestricted zero-result requirement for large points.

Do not increase cap15 and growth simultaneously in one experiment: both change overfetch and query decisions, obscuring cause. Tuning a cap alone might plausibly reduce 0–10% build time when small/big transition overhead matters, or regress substantially; this subordinate experiment is included in M8's range, not an additive estimate.

## M9: warm ghost cache is an unimplemented opportunity with index hazards

`InitialGhostPointsExchange` exists (`Voronoi3D.hpp:1625`), but its supplying `FilterRealGhostPoints` call is commented out at `:2334`; a symbol search found no other active call. The current private arrays are only populated inside that function. Consequently the normal nonperiodic build path still allocates rank arrays, does a has-data Allreduce (`:1661`), and usually returns no points. Periodic mode skips warm exchange altogether (`:3627`).

Do not just uncomment FilterRealGhostPoints. The filter currently stores `indicesInAllMyPoints[pointIdxInBuild]` (`:1737`), while InitialGhostPointsExchange checks this stored index against the *next* build's Norg and indexes *next* `allMyPoints` with it (`:1646`–`:1649`). Migration/reordering/partial builds can invalidate those indices. The filter also uses repeated linear scans of Nghost (`:1728`).

First implement a cache guarded by an ownership/point-order generation token, use only after builds with unchanged point identity/order, and invalidate for migration, rebalancing, insertion/deletion, boundary-mode changes, and unsupported partial builds. Cache stable point IDs plus image translations for later migration-aware support; IDs need a separate mapping when local indices change. Reuse cached ghosts as hints only, followed by the full search/completeness algorithm. Accelerate actual-neighbor filtering by creating a ghost-index-to-owner array and walking each active cell's adjacency once. Warm starts should reduce discovery rounds, but stale yet valid extra ghosts can increase Delaunay work, so impose a measured cache budget.

There is also a small safe current-path cleanup: if the cache is known disabled by configuration, omit its no-op communication altogether consistently on every rank. Its expected standalone contribution is below 1–3% of tiny MPI builds and near0 for large builds; this is included in M9 rather than a separate gain.

## M10/M11: overlap and final field exchange

BringSelfGhostPoints currently completes self-small answers, self-big answers, and their Delaunay insertion before remote querying begins (`Voronoi3D.hpp:3403`, `:3420`, `:3437`, outer `:3780`, `:3785`). With periodic boundaries or partial builds, local work can be significant. After M3 introduces explicit start/progress/finish, submit remote requests then alternate bounded local query chunks with network progress. Accumulate points and insert only at a defined geometry-safe phase. MPI requests need progress while local Delaunay work runs; do not claim automatic overlap from starting an Irecv. A dedicated MPI thread requires separate thread-safety work and is not the first step.

The final indexed field helper posts sends then uses blocking Probe/Recv one message at a time, and finishes with a global Barrier (`source/utils/mpi_utils/mpi_exchange.hpp:21`, `:28`, `:41`, `:52`). The ghost correspondence arrays already give expected element counts at established call sites. Add a helper accepting receive counts, post all receives first, send packed data, then Waitall. For fixed centroid/volume records this avoids wildcard probing and rank lookup. Consider a combined centroid+volume record only after checking `SyncPartialBuildData`'s dependencies and periodic centroid translation order (`Voronoi3D.hpp:3159`–`:3198`). It is not safe to fuse exchanges simply because they both send to ghosts. Replace the global barrier only when each caller's epoch order and buffer completion are proved.

## Correctness prerequisites and criticism traps

### C0: partial-build full-tree indexing

`UpdatePointsTree` sets `allPointsNum=allMyPoints.size()` then indexes `activePoints[pointIdx]` in that full-length loop (`Voronoi3D.hpp:2137`–`:2141`). BuildPartiallyParallel passes its subset `activePoints` at `:2263`. When the subset is shorter, this reads beyond its bounds. The full allMyPointsTree should use `allMyPoints`, while the active myPointsTree should use active/build indices explicitly. Require a meaningful partial subset with noncontiguous original indices and `active_count < local_owned_count`, checked under ASan/UBSan and compared against full serial geometry. This is a correctness fix; do not claim it as measured performance work.

### C1: MPI_Testsome completion order and compaction

`BuffersManager::CleanSendRequests` iterates returned completion indices in reverse returned order, and swap-pops the request vector (`BuffersManager.hpp:365`–`:383`). Reverse *returned* order is only equivalent to descending *index* order if the MPI implementation returns an ascending list. The API does not provide that ordering contract. A mocked completion list `[2,0]` for three requests makes the loop erase0 first, shrink the vector to2, then access index2 on its next iteration. Fix by sorting completed indices descending before any compaction, or use stable slots/free-list bookkeeping. Keep the associated request-to-buffer mapping correct under the same permutation. Test all permutations of completion subsets and actual multiple MPI implementations. This is a portability risk demonstrated by the bookkeeping algorithm, not an observed production crash.

Primary API reference checked: [MPI Forum, MPI 4.1 §4.7.5 Multiple Completions](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node76.htm). Its Waitsome/Testsome descriptions specify the completed indices and their associated statuses, but no ascending-index guarantee. The portability concern is an inference from that contract and the local swap-pop algorithm.

### C2: do not remove safety without an explicit replacement

The current distributed completion tree has verification and explicit draining. Removing "extra" barriers independently can expose old posted receives to a different query type using the same tags. Retaining MPI_Issend alone does not establish the full final state of every peer. Test tiny/eager and larger/rendezvous messages, intentionally delayed ranks, asymmetric query counts, multiple back-to-back builds, and empty ranks. A request-counter unit test is not sufficient: validate end-to-end sent/received image identity and mesh correctness.

Other traps:

* `SetPointTetras` is **already incremental**: it processes empty, changed and new tetrahedra (`Voronoi3D.hpp:1268`, `:1287`, `:1324`), not every tetrahedron. Do not propose "make it incremental" as if absent. It still has empty-list and container work, and its cost can be measured.
* `Delaunay3D::BuildExtra` already returns for empty input (`delaunay/Delaunay3D.hpp:703`). Do not attribute a full rebuild to those calls. Nonempty calls do copy existing points and classify lattice predicate state (`:706`–`:713`), which is a different legitimate geometry optimization.
* Big queries for one tetrahedron can be associated with different original points and ask for the nearest point to different `originalPoint` values (`CreateBatches:3327`). Deduplicating solely on center/radius/tetra ID changes semantics. Sharing candidate enumeration is possible; merging the queries without preserving each nearest-point question is not a safe local edit. Expected contribution of this speculative optimization is 0–10% only if repeated tetrahedron answering dominates; otherwise0. This is lower priority than the measured M1–M8 work and overlaps M7/M8.
* Periodic expansion intentionally sets shifted big queries `askOnlyClose=false` (`Voronoi3D.hpp:209`–`:213`). Do not reenable cached nearest-rank filtering for translations without including the translated original point and proving validity.
* Bounding-box routing may overselect; it must never underselect. Approximate geometric pruning requires conservative bounds.
* Hash-table substitutions must preserve deterministic exported ghost ordering. Faster membership does not authorize random order in subsequent physical field exchanges.
* The range-answer ignore set already excludes points sent previously to the same peer/image. A response count is the number newly contributed to the receiver, not the raw number geometrically inside a sphere. Keep this distinction in adaptive rules.
* Removing MPI_Alltoall count metadata can improve scaling but requires a sparse arrival-discovery/termination protocol. The fact that "most peers are neighbors" does not tell a rank which nonneighbors will send it a query.

## Acceptance and sequencing

Order: record baseline and C0/C1 correctness gates; M1; M2; M5/M6/M12 low-risk data/layout improvements; then M8 and M3 as separate competing experiments driven by the baseline; M4 refines the chosen backend; M7 if routing/cache profiles justify it; M9 for repeated builds; M10 last. M11 can be independent if field exchange is measurable.

For each patch, retain a reference backend/configuration, run same points/seeds and ownership policy, validate global owned-point count/IDs, finite positive active volumes, total volume conservation against the domain, face/neighbor consistency, periodic-image displacement correctness, and centroid/volume ghost payload agreement. Compare geometry keyed by physical global IDs and image shifts, not local array order. Exact face topology equality can be unsuitable for truly degenerate cospherical sets; use the repository's documented degeneracy policy and geometric invariants as appropriate. Require serial, P=2/3/4, more ranks than active points, nonperiodic, each single periodic axis, full periodic, cold/repeated builds, mild motion, large motion, migration, and noncontiguous partial subsets.

For performance, report medians and distribution over repeated independent runs, max-rank build time, strong and weak scaling, enough iterations to dominate launch/jitter, and per-phase counters. Do not accept a change based only on lower rank-0 logs or fewer MPI calls. Suggested performance gate: confidence interval shows improvement on its target workload and no unexplained material regression (for example >3%) on designated protected workloads. The threshold should reflect actual cluster jitter; it is an engineering acceptance criterion, not a promised speedup.
