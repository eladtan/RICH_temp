# MPI / ghost-query correctness audit

Read-only source audit on 2026-09-08. No production file changed. Evidence programs and logs are under `docs/madvoro_bugs/evidence/mpi_*`; executables are under `/tmp`. Exact source paths below are relative to `/home/maorm/RICH`. Root owns the general partial-build index defects and the serial API in an MPI-enabled build; this note reports their communication consequences separately.

## Severity and evidence summary

No P0 finding was established. Priorities below distinguish current MadVoro reachability from bugs in a configurable shared dependency.

| ID | Priority | Finding | Evidence / applicability |
| --- | --- | --- | --- |
| MPI-01 | P1 | Request completion compaction assumes sorted MPI completion indices | Actual production method throws under an injected API-permitted `[2,0]` completion list. Active in ordinary MadVoro query transport; the inspected MPI implementation's natural return order was not shown to trigger it. |
| MPI-02 | P1 for supported parallel partial builds | `EnsureSymmetry` deletes required one-way ghost state | Exact-production-body fixture proves receive-only peer and its required Nghost map are removed. Reachable intended partial-build state; full end-to-end symptom is currently masked by independently identified partial tree-index defects. |
| MPI-03 | P2 | Mixed periodicity generates each shifted query repeatedly | Exact production helper emits nine copies of a required image with one periodic axis, three with two. Active overfetch/performance defect, not established wrong final tessellation. |
| DEP-01 | P2 | Buffer overflow/truncation when appending records that individually fit | Real two-rank run of unchanged header, reasonable constructor settings, two individually fitting records. With MPI_ERRORS_RETURN it silently delivers 48/80 application bytes. Standard MadVoro's current capacity headroom and capped answers do not trigger this tested case. |
| DEP-02 | P2 | Eight-byte payload is mistaken for an empty buffer and never sent | Real two-rank test, 10,000 progress calls, sent=0 and pending=1. Generic `BuffersManager<uint64_t>` defect; current MadVoro subquery/answer records are larger. |
| DEP-03 | P2 | `sendToSelf=true` drops answers from flattened results | Real one-rank query batch gives per-query=1/by-rank=1/flattened=0. Current MadVoro explicitly uses sendToSelf=false, so dependency/API finding only. |

P1 means fix before relying on the affected supported configuration or broadening transport optimizations; it does not claim a crash on every run. The generic defects must not be counted as three independently observed failures in ordinary MadVoro builds.

## MPI-01: invalid request compaction for legal completion-index order

**Source:** `source/utils/mpi_utils/BuffersManager.hpp:365`–`:385`; specifically reverse iteration at `:368`, vector access at `:371`, swap/pop at `:382`–`:383` and map updates at `:372`–`:380`. This method is called on the active ghost-query path through `BuffersManagerQueryAgent.hpp:282`–`:283`.

**Trigger and failure:** at least two sends complete in one MPI_Testsome call, and the returned indices are not ascending. For a request vector with three entries and completion indices `[2,0]`, the current reverse loop processes0 first. It swap-pops slot2 into slot0 and shrinks to2 entries. Processing the next index2 then calls `.at(2)` on a2-entry vector and throws. Other permutations can also misassociate buffers before an exception.

**Proof:** `mpi_dependency_repro.cpp compaction` includes the actual unchanged production header and calls its `CleanSendRequests`. A shim injects exactly one post-completion list `[2,0]`; this is a bookkeeping fixture, not a natural MPI-network reproduction. The output in `mpi_compaction.log` is:

```text
COMP_ACTION_DEFECT completion_indices=[2,0] exception=vector::_M_range_check:
__n (which is 2) >= this->size() (which is 2) remaining_slots=2
```

The [MPI Forum multiple-completion contract](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node76.htm) specifies completed operation indices and associated statuses but no ascending-order guarantee. The implementation must handle every allowed index ordering. This is stronger than merely asserting the code looks suspicious, but weaker than claiming this Open MPI build naturally returns that order.

**Fix:** collect completion indices and sort them descending numerically before mutating the request array; if statuses are used, sort paired index/status records. Alternatively retain stable request slots and a free list. For each removal, release exactly that request's buffer, update the moved last-slot map exactly once, then pop. Check `outcount` before iteration, including0 and MPI_UNDEFINED, and choose an explicit policy for MPI_ERR_IN_STATUS. Do not reverse a returned array and assume that sorts it.

**Acceptance:** exhaustive subsets/permutations for request arrays0–8, including cases containing the last slot and nonlast slots; verify remaining request identity, buffer identity, active count and free-buffer uniqueness. Then run actual multi-rank eager/rendezvous traffic, delayed peers, and repeated query batches. Keep synthetic index-order perturbation in the test harness because a single MPI implementation may always return ascending indices and conceal the defect.

## MPI-02: deleting receive-only and send-only ghost peers breaks partial builds

**Sources:** `source/3D/tessellation/voronoi/Voronoi3D.hpp:1776`–`:1790`, active call at `:3885`; `SetGhostArray` at `:1969`–`:1987`; centroid synchronization at `:5600`–`:5611`; indexed helper `source/utils/mpi_utils/mpi_exchange.hpp:18`–`:41`.

**Trigger:** rank A owns generators but builds no active cells this partial step; rank B builds an active cell adjacent to one of A's generators. A must answer B's range queries despite having no local queries. B therefore receives a required ghost from A but sends no points to A. This is a valid directed data dependency. The current `EnsureSymmetry` removes a peer if it is absent from either sent peers or all received-peer lists, erasing Nghost mapping as well as export mapping.

**Proof:** `mpi_extracted_helpers.cpp` copies the production EnsureSymmetry body verbatim, changing only the class qualification to a fixture. Starting from a receive-only peer1 with ghost index5, invoking `EnsureSymmetry({},{{1}})` produces:

```text
ONE_WAY_GHOST peer_count=0 ghost_lists=0 expected_peer_count=1
```

The ghost point remains in Delaunay storage; only its owner/field mapping is removed. Later SyncPartialBuildData has no mapping through which to populate its remote centroid. Subsequent physical field exchange through exposed ghost mappings can similarly miss it. This note does not claim an end-to-end partial-build output reproduction: the root audit's earlier tree/subset defects currently obstruct a trustworthy full partial build.

**Fix:** distinguish symmetric *communication peer lists* from symmetric *nonempty data*. Preserve the union of send/receive peers and keep an empty segment for a missing direction, or use separate directed send/receive peer arrays. Existing indexed helper sends an empty message for an empty segment, so a union peer list can be compatible with its matching model. Filtering actual unneeded ghosts should be based on active-cell adjacency/ownership requirements, not reciprocity of point requests.

**Do not apply a one-character OR-to-AND fix alone.** `duplicated_points_` contains all-owned indices. The final volume exchange at `Voronoi3D.hpp:2315` sends from `volume_`, which was sized to active `Norg_` at `:2296`; an inactive donor's all-owned index is not a valid active volume index. Centroid SyncPartialBuildData instead sends from the all-owned `allBuildData` array at `:5600`. The partial-build contract must define valid current/cached fields for inactive donors and supply a correctly mapped source array. Otherwise preserving the previously deleted peer can reveal an out-of-bounds volume access. Coordinate with the root partial-build fix rather than hiding the dependency by dropping the peer.

**Acceptance:** two ranks, no active cells on one rank but required generators on that rank; one populated rank plus an empty rank; noncontiguous active subsets; active masks differing per rank; fields initialized with distinctive nonzero sentinels. Verify every required ghost's source identity and centroid/volume/state rather than only owned volume sum. All ranks must finish the same protocol. For both directions, assert that expected receive field counts match Nghost lengths, including zero counts. Include full builds as a protected case.

## MPI-03: repeated zero entries multiply mixed-periodic image queries

**Source:** `source/3D/tessellation/voronoi/Voronoi3D.hpp:176` initializes `std::array<int,3> shiftRanges[3]={{0},{0},{0}}`; `:189`–`:193` range-iterates all three elements of every axis. Only periodic axes are overwritten with `{-1,0,1}`. Active call sites are `:3746`–`:3747`.

**Trigger:** one or two periodic axes and a query touching at least one periodic image. A nonperiodic axis has `{0,0,0}`, so each logical zero shift is traversed three times. A shifted query is emitted `3^(3-d)` times when d axes are periodic: nine copies for d=1 and three for d=2. Full three-axis periodicity does not have this defect; the unshifted base query is pushed once before the loop and the repeated all-zero combinations are skipped.

**Proof:** verbatim production SphereIntersectsAABB and ExpandPeriodicQueries bodies, unit box, center(.05,.5,.5), radius.1, exactly one required negative-x image:

```text
PERIODIC_EXPANSION axes=1 records=10 unique_images=2 negative_x_image_multiplicity=9
PERIODIC_EXPANSION axes=2 records=4 unique_images=2 negative_x_image_multiplicity=3
PERIODIC_EXPANSION axes=3 records=2 unique_images=2 negative_x_image_multiplicity=1
```

See `mpi_extracted_helpers.log` and the source extraction provenance. This is an active performance/overfetch bug. Per-peer/image ignore sets prevent sending the *same* point repeatedly, but do not make repeated queries harmless: repeated capped small queries may each obtain more previously unsent points (up to nine caps), and repeated big nearest-point queries may obtain successive distinct points instead of one. No wrong final mesh was demonstrated; do not relabel extra query/ghost work as proven numerical corruption.

**Fix:** iterate a nonperiodic axis exactly once. An allocation-free choice is bounds `lo = periodic[d] ? -1 : 0`, `hi = periodic[d] ? 1 : 0` and integer loops `for(s=lo;s<=hi;++s)`. Preserve current shift order, all-zero skip, sphere/AABB test, center translation, originalPoint translation and askOnlyClose=false on shifted big queries. Alternatively use arrays with explicit active lengths. Do not deduplicate the returned point set across distinct image translations.

**Acceptance:** for all eight periodic-axis masks and face/edge/corner spheres, enumerate unique shift tuples from a simple reference and require exactly one query per intersecting tuple. Verify no duplicated parent/image pair and no missing shift. Include both small/big record types and the shifted originalPoint. Run mesh regressions for one, two and three periodic axes; measure ghost/query count changes but compare physical cell geometry, since redundant query elimination can legitimately alter insertion order in degenerate inputs.

## DEP-01: serialized appends exceed receive capacity; MPI errors are ignored

**Sources:** `source/utils/mpi_utils/BuffersManager.hpp:287`–`:297` appends the entire new value before considering dispatch, checking only an assert; `:334`–`:338` asserts packet size and sends it; receive buffer is sized once at `:122`–`:125`; `:230` ignores MPI_Testsome's return code and `:243`–`:247` deserializes based on the packet header without checking MPI_Get_count or status error. Serializer extraction loops at `source/utils/mpi_utils/serialize/Serializer.hpp:278`–`:287` stop at storage end and can return a partial vector without reporting the requested length was unmet.

**Valid-setting trigger:** `buffersSize=64`, `minSizeToDispatch=64`, two Blob values with40 payload chars each. Each value's serialized size is48 bytes and individually fits the configured64-byte payload capacity. After the first append the payload counter48 is below the internally adjusted threshold72. The second append makes payload96 and actual packet104, larger than the posted receive72. No preflight dispatch occurs. A release build with NDEBUG sends it anyway.

**Actual real-MPI reproduction:** `mpi_dependency_repro.cpp overflowfit`,2 ranks, unchanged production header, MPI_ERRORS_RETURN, shows in `mpi_overflowfit.log`:

```text
BUFFER_BOUNDS mode=overflowfit rank=0 callbacks=0 received_payload_bytes=0 sent_payload_bytes=80
BUFFER_BOUNDS mode=overflowfit rank=1 callbacks=2 received_payload_bytes=48 sent_payload_bytes=0
```

Both callbacks are accepted, but the second value has only8 of40 payload bytes. A separate single-large-value test (`oversize`) delivered56/200 payload bytes. The ordinary MPI error handler typically aborts instead; this fixture deliberately uses MPI_ERRORS_RETURN to reveal that the library continues with corrupted data rather than checking the error. Assertions enabled can abort on the sending side earlier. None of these behaviors is correct supported dispatch handling for individually fitting records.

**Applicability:** the exact dependency used by MadVoro is defective under these constructor settings. Current MadVoro config has much larger capacity than its dispatch threshold, and small/big answer counts are capped16/1. Therefore no ordinary current MadVoro overflow was established. This becomes relevant to generic shared users, custom serialization, or proposed buffer-size tuning; do not publish it as an observed large Voronoi workload failure.

**Fix:** use payload-byte units consistently for capacity, threshold and header. Determine serialized size before appending (or serialize a single record to a temporary bounded buffer). If current payload + record > capacity, dispatch current nonempty packet before append. If a single record cannot fit, use a documented sized-message/chunk route or return a controlled error; never rely on a release-disabled assert. Check every MPI return code/status; do not invoke callbacks for truncated/failed messages. Validate declared payload bytes against actual received MPI byte count and require exact deserialization consumption. Retain send memory until completion. Tests must distinguish application record bytes from packet header bytes.

**Acceptance:** payload sizes0,1,7,8,9, capacity-1, capacity, capacity+1; two individually fitting records that overflow together; min threshold equal to capacity; variable Serializable records; byte-count overflow; default fatal and MPI_ERRORS_RETURN modes. All successful callbacks must contain exact sent records. A controlled rejection must be explicit and must not leave other ranks waiting forever.

## DEP-02: eight-byte pending payload cannot dispatch

**Source:** `source/utils/mpi_utils/BuffersManager.hpp:287`–`:291` writes a header counting payload bytes only. `ShouldSend` at `:312` compares that count to `sizeof(size_t)` and returns false, intending to recognize empty storage. An empty payload counter is0, not8 on this host. The check bypasses both byte threshold and cycle-age dispatch checks.

**Actual reproduction:**2 ranks, `BuffersManager<uint64_t>` with buffer1024, threshold64, cycle limit2. Rank0 adds one uint64_t to rank1; both call HandleIncomingOutcoming10,000 times. `mpi_scalar8.log` shows sender sent_messages0/pending_buffers1 and receiver callbacks0. Destroy does not flush the remaining rank buffer; its send counters are0, so it cancels receives and silently drops the buffered value.

**Applicability:** generic shared dependency defect. Current MadVoro SubQuery and AnswerInfo records have payloads larger than8; this exact stall is not established in their ordinary path. Any configuration/type yielding an accumulated serialized payload of8 bytes can trigger it, including two4-byte records.

**Fix:** use `payloadBytes==0` to test emptiness, keeping all other thresholds in the same byte convention. Add an explicit flush-complete contract so Destroy cannot silently abandon nonempty pending buffers; be careful that destructors cannot safely invent new collective traffic after peers leave or MPI finalizes. Test finalization through a collective finish API and assert no pending payload after successful completion.

**Acceptance:** one and multiple uint32/uint64 payloads, idle tails, zero-length Serializable, delay-based flush below threshold, and normal query records. Require exact delivery and pending/outgoing counts0 after finish.

## DEP-03: self-directed query answers disappear from flattened batch results

**Source:** `source/utils/mpi_utils/queryAgent/BuffersManagerQueryAgent.hpp:145`–`:158` supports sendToSelf=true; `:120`–`:121` stores received self answers in both result representations. Finalization at `:305`–`:309` unconditionally skips the local rank when creating recvProcessorsRanks. Rearrangement at `:174`–`:189` only emits ranks in that list.

**Actual reproduction:** one rank, trivial serializable query123, TalkAgent returning rank0, AnswerAgent returning `{123}`, sendToSelf=true. `mpi_self.log` reports:

```text
SELF per_query=1 by_rank=1 flattened=0 recv_peers=0
```

The answer was transmitted and accounted for; flattening omits it. **Current MadVoro explicitly selects false** (`range/BigRangeAgent.hpp:272`, `range/SmallRangeAgent.hpp:184`), so it is not a present MadVoro self-ghost loss; the normal self-ghost path calls answer methods directly.

**Fix:** when sendToSelf is enabled, include the local rank in receive peer bookkeeping and flattening under a documented order, or explicitly include the self segment separately while preserving the index contract. Keep disabled behavior unchanged. If self mode is intentionally unsupported, reject the argument at construction rather than returning inconsistent result views.

**Acceptance:**1 rank/self enabled and disabled;2 ranks with self plus remote answers; empty answers; multiple batches where only later batches have self results; compare per-query counts, by-rank segments, flattened payloads and recv-index mappings.

## Reviewed concerns not promoted to demonstrated ordinary-path defects

* `CountOutcoming` excludes serialized pending buffers. A replacement termination predicate must include them. However the current MadVoro completion tree also waits for each issued query's reply; with exactly-once replies and a once-only finish contribution, outstanding unsent queries/answers prevent global logical completion. No present premature termination was reproduced. Do not claim the incomplete local predicate alone proves a live current ghost race.
* Big/small agents reuse tags200/201 and finish tags9918–9921 on the same communicator. Their current batches run serially with drain/cancel/barrier handling. Concurrent instances/types would require context/epoch isolation, but concurrency was not established as a currently supported executed scenario. This is a change prerequisite, not evidence of routine cross-talk today.
* Generic integer MPI counts, size_t/MPI_UNSIGNED_LONG_LONG pairing, and actual-received-length validation need portability/large-message hardening. A2GB+ ordinary MadVoro message or32-bit-platform corruption was not reproduced; do not inflate severity using hypothetical sizes.
* The previous-ghost filter is currently not called. Its stale index/migration concerns matter before reenabling caching, but were not counted as an active populated-cache failure.
* Periodic identity by image code is valid for the current one-box shifts generated here. Collapsing arbitrary multiple-box translations by sign would need a broader supported API trigger before being called an active bug.

## Reproduction commands and interpretation

From `/home/maorm/RICH` with the available compiler/MPI modules:

```bash
mpicxx -std=c++17 -O1 -DNDEBUG -g -D__WITH_MPI \
  -Isource/utils -I/software/x86_64/5.14.0/boost/1.78.0/include \
  docs/madvoro_bugs/evidence/mpi_dependency_repro.cpp \
  source/utils/mpi_utils/AmountManager.cpp -o /tmp/madvoro_mpi_dependency_repro
UCX_TLS=tcp,self,sm timeout 20s mpiexec --bind-to none -np 1 /tmp/madvoro_mpi_dependency_repro compaction
UCX_TLS=tcp,self,sm timeout 20s mpiexec --bind-to none -np 2 /tmp/madvoro_mpi_dependency_repro scalar8
UCX_TLS=tcp,self,sm timeout 20s mpiexec --bind-to none -np 1 /tmp/madvoro_mpi_dependency_repro self
UCX_TLS=tcp,self,sm timeout 20s mpiexec --bind-to none -np 2 /tmp/madvoro_mpi_dependency_repro overflowfit
python3 docs/madvoro_bugs/evidence/mpi_extract_repros.py
c++ -std=c++17 -O1 -Isource/3D/tessellation/voronoi \
  docs/madvoro_bugs/evidence/mpi_extracted_helpers.cpp -o /tmp/madvoro_mpi_extracted_helpers
/tmp/madvoro_mpi_extracted_helpers
```

Dependency harness exit1 means the programmed defect condition was observed; MPI launcher nonzero-exit boilerplate is expected. These bounded cases did not time out. Initial compaction log includes local UCX transport warnings from a preexisting `ib` selection; subsequent runs selected tcp/self/shared memory explicitly. They are not evidence for the code defect. The extracted helper fixture uses minimal data types but verbatim algorithm bodies; it is not a patched production mesh executable.
