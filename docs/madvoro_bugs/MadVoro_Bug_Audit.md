# RICH / MadVoro correctness audit and repair plan

8 September 2026. Current source review, bounded reproductions, and implementation contracts.

## 1. Findings and how to read this report

**The most consequential confirmed MadVoro failure is a partial build that silently changes the geometry.** A request for eight active cells supported by 32 retained generators deletes the 24 inactive generators. Every requested cell's volume changes; the eight volumes sum to the entire unit box instead of the reference sum 0.246862. Other reproduced failures include invalid copied meshes, incorrect index maps, a mixed-rank suppression assertion, and a wrong periodic identity result. RICH's separate migration helper also misorders physical fields and can hang when an originally empty rank receives a cell.

This document contains **24 findings: 10 high, 13 medium and 1 low priority; no P0 finding was established.** These counts include MadVoro, its shared dependencies and two explicitly separate RICH integration defects. They are not 24 independent failures of an ordinary full MadVoro build. Twenty findings have a direct production-API/component reproduction; two use verbatim extracted helper fixtures, one injects permitted MPI completion metadata into the production method, and one is a source-level recovery defect without a demonstrated valid-mesh trigger. Scope and evidence matter as much as severity.

**Severity measures the consequence when the stated trigger occurs.** P1 / high means a crash, hang, loss of required mesh generators, or silent incorrect geometry/data association in an affected supported workflow. Address these before relying on that workflow or changing the relevant communication protocol. P2 / medium means a narrower lifecycle/API failure, portability/configuration defect, or systematic avoidable work without demonstrated wrong numerical output. P3 / low means an isolated helper defect with no current internal caller found. P0 would require evidence of an unconditional or very broad catastrophic failure; this audit does not provide it. Priorities are engineering judgments, not measured occurrence rates.

**Evidence labels:** R = actual current production API or component executed; F = verbatim production algorithm exercised in a minimal helper fixture; I = unchanged production method with explicitly injected MPI completion metadata; S = concrete source defect without a successful end-to-end trigger. A recorded nonzero exit often means the harness detected a defect deliberately. A timeout is attributed to a hang only when progress markers and the source show incompatible collective participation. No claim of a fix passing is made: production code was not changed.

| ID | Priority | Evidence | Scope | Located error |
| --- | --- | --- | --- | --- |
| B01 | P1 | R | Partial MPI / manager | Inactive generators are dropped; all eight reference cell volumes change. |
| B02 | P1 | R + masked MPI source | Partial core | All-point tree indexes shorter active storage. |
| B03 | P1 | R | Partial serial core | Active-to-all map is written in reverse. |
| B04 | P1 conditional | F + source | Partial MPI core | One-way ghost dependencies are removed by a reciprocity filter. |
| B05 | P1 | R | Core copy / clone | Built copy has null point-location trees. |
| B06 | P2 | R | Core copy / manager | Unbuilt copy dereferences absent load balancer. |
| B07 | P1 | R | Core MPI preparation | Mixed suppression flags select undersized cached weights. |
| B08 | P2 | R | Serial API in MPI binary | Build selects absent distributed environment. |
| B09 | P2 | R | First suppressed MPI build | Missing initial balancer reaches environment creation. |
| B10 | P2 | R | Custom-domain lifecycle | ReleaseMemory changes the domain predicate. |
| B11 | P2 | R | Empty MPI mesh query | Continuity query writes element zero of an empty vector. |
| B12 | P2 | R, dependency method | SetKernel / manager | Replacing a kernel leaves a null converter that is not rebuilt. |
| B13 | P2 | R | Periodic core API | Remote image is mapped to an unrelated owned point. |
| B14 | P2 | F | Mixed-periodic core | Duplicate zero shifts multiply image queries. |
| B15 | P2, occurrence unmeasured | S | Face cleanup | Recovery uses stale polygon size and wrong second candidate. |
| B16 | P3 | R | Unused public helper | Face centroid omits normalization by area. |
| D01 | P1 conditional | I | Active MPI dependency | Request compaction assumes sorted completion indices. |
| D02 | P1 | R | Optional manager communicator | Subgroup manager routes in WORLD rank space. |
| D03 | P2 | R | Optional 1D balancer | One-rank rebalance contradicts intersection bin invariant. |
| D04 | P2 | R | Generic buffer settings | Appends exceed packet capacity; truncated data is accepted with returning MPI errors. |
| D05 | P2 | R | Generic eight-byte payload | Pending data is mistaken for an empty buffer. |
| D06 | P2 | R | Generic self-query option | Flattened results omit self answers. |
| R01 | P1 | R | RICH integration | ExchangeChain ignores destination indices. |
| R02 | P1 | R | RICH integration | Empty-origin rank skips required collective transfer. |

**Recommended order.** Repair R01/R02 together where chain-based physical-field migration is in use. Repair the linked partial-build contract B01-B04 as one validated series before enabling true subsets. Repair B05/B06 copy semantics, B07 global suppression state and D01 request bookkeeping before broad MPI optimization. Then address lifecycle and periodic identity, followed by scoped generic dependency defects. A low-effort helper fix can proceed independently, but it does not reduce the high-impact risks above.

## 2. Exact source snapshot and evidence limits

The audited workspace is `/home/maorm/RICH`. Source locations below are one-based lines in this checkout, not necessarily the latest upstream repository. V denotes `source/3D/tessellation/voronoi/Voronoi3D.hpp`. The source hashes and revision files accompany the report under `docs/madvoro_bugs/evidence/`; when a line shifts, use the named function and the recorded source text.

| Component | Audited revision |
| --- | --- |
| RICH | `2c71b28f49cd40ec46e031e4e896849e40e363c8` |
| MadVoro | `ea2f32a2f1cef7816a1db80d118eb30fbae9f0da` |
| MeshDecomposer3D | `d65b8b3886bb0d129dc8a88862d42f18eaacd1ba` |
| mpi_utils | `84212d05d2de5c44f5cd4254c8afa67bb65f4c04` |
| spatial_ds | `29bb801065a01907f041be7450e3d6d4efebc16f` |

**Workspace condition.** The checkout had unrelated pre-existing modifications, including build configuration and generated regression binaries. The audit did not reset those changes and did not modify production source. `rich_build_config.patch`, `source_sha256.txt` and `submodule_versions.txt` preserve the relevant context. Findings target inspected source, not an old installed executable. Additional RICH integration files are fingerprinted in `integration_source_sha256.txt`.

**Execution environment.** Local Linux x86-64, GCC 15.1.0 and Open MPI 4.1.6. Small executables were compiled directly from current headers/sources, primarily C++17 at O1, with MPI/OpenMP defines and one OpenMP thread. Root API and selected integration tests used `_GLIBCXX_ASSERTIONS`; this turns invalid subscripts into useful assertions instead of letting undefined behavior continue. The partial-volume differential used O0. The packet-overflow test deliberately used NDEBUG and MPI_ERRORS_RETURN to observe unchecked error handling. No cluster-scale performance measurement, full RICH physical simulation, full sanitizer sweep, alternate MPI implementation or exhaustive degeneracy proof was performed.

**What the evidence can establish.** A production helper returning a wrong result establishes its API defect; it does not prove an unguarded downstream application call. A legal injected completion order establishes a portability defect; it does not show that this Open MPI version naturally produced that order. A verbatim extracted cleanup or expansion algorithm establishes local behavior; its fixture does not reproduce a whole distributed build. The report explicitly distinguishes these cases. Subsequent implementations must turn the failing witnesses into positive acceptance tests and add the stated integration coverage.

The main geometry/MPI findings were independently reviewed in `audit_notes/root_api_review.md` and `audit_notes/mpi_final_review.md`. Raw compiler logs, diagnostic outputs and small source fixtures are retained. Early `root_mpi_identity.log` and `root_mpi_subset.log` exercise fresh suppressed initialization; use the v2 logs and the independent volume test for the initialized partial path. `geometry_cleanup.log` is a negative search result, not evidence that the full-mesh cleanup failure was reproduced.

**Relation to the performance plan.** Correctness repair must precede cached ghost reuse, communication overlap and broader partial-build optimization. This audit refines the earlier partial-build concern: the default MPI path currently discards inactive points, which masks a second out-of-bounds tree loop. The duplicate periodic-image factors in B14 are query multiplicities, not predicted whole-program speedups. This report makes no new measured speedup claims.

## 3. Contracts an implementation must preserve

### Three index spaces and physical identity

Treat all-owned index k, current active/mesh index j and remote owner-local index as different values even when they happen to be numerically equal in a full build. Before ghost insertion, `allMyPoints[k]` contains every retained local generator, while `del_.points_[j]` for j < Norg_ contains active generators. The active prefix requires `indicesInAllMyPoints.at(j) = k` and matching coordinates. After insertion, some j >= Norg_ can represent locally owned inactive support and need additional valid mappings; other entries are remote ghosts or boundary/periodic images.

| Object | Required index meaning |
| --- | --- |
| All-point tree | Coordinates of all retained local generators, payload index k. |
| Active-point tree | Coordinates of active generators, payload index j. |
| Active-to-all map | Current local mesh index j -> corresponding all-owned index k where one exists. |
| Export list `duplicated_points_` | The all-owned source indices expected by each field exchange; verify the actual source array. |
| Receive list `Nghost_` | Destination mesh indices corresponding to received records in exact message order. |
| Periodic image identity | Physical generator identity plus image translation; nearest local point is insufficient. |
| ExchangeChain target | Final communicator rank and final local index; both components determine field placement. |

Do not prove identity from matching vector sizes, total volume, a sum of payload values or membership in the same rank. Use stable fixture IDs and compare each coordinate, field and mapping. Tree sharing requires identical indexed contents, not just identical cardinality. Every export array must cover the index space used to subscript it. Every required receive index needs a live ownership/mapping route even if there is no nonempty reverse message.

### Lifecycle and collective participation

Model at least 'configured but unbuilt', 'built', 'released/unbuilt' and 'reconfiguration pending'. Initialization of a manager pointer, its converter and its environment are distinct state transitions. A copy must be a coherent snapshot with independent valid owners, or the API must state a different explicit contract; a mixture of copied geometry and missing query state is unacceptable. Release mesh memory without silently replacing persistent physical-domain configuration.

For a distributed operation, every rank follows the same agreed phase sequence. A rank with zero local work may still supply generators, answer queries, receive cells or participate in collective completion. Decisions reduced globally must control state preparation globally too. Validation that can fail on only one rank needs an agreed error policy before peers start blocking operations. A serial entry point in an MPI-enabled binary must not infer its communication mode from a configured object's incidental pointer state.

Partial fields need an explicit age/validity contract. Retaining an inactive generator does not imply its cached volume or centroid is current after points move. A repair must either compute required donor geometry, preserve a demonstrably valid cache, or declare that the affected consumer cannot use that field during a partial step. Resizing storage with zero values is not an acceptable substitute for valid scientific data.


## 4. Partial builds: linked defects that require a coherent repair


### B01 / P1: inactive generators are deleted

**Criticality: P1 / high. Evidence: R, directly reproduced with the production mesh API. Scope: the default MeshDecomposer3D manager and MadVoro's partial-build preparation.** This is silent wrong geometry for the intended partial-build contract, not merely a return-vector size discrepancy.

**Location.** `source/3D/tessellation/MeshDecomposer3D/points_manager/PointsManager.hpp:288-297` constructs exchange records only for `indicesToWorkWith`, setting every record's `participating` flag to true. At `:304-319`, source-index reconstruction also assumes this filtered record order. `source/3D/tessellation/voronoi/Voronoi3D.hpp:2032` replaces `allMyPoints` with the returned records, and `:2053-2059` constructs active points from their participation flags. Thus inactive generators disappear instead of remaining available as geometric support.

**Trigger and observed result.** Build a valid 32-point unit-box mesh, retain all 32 input generators, and ask to rebuild only indices 0 through 7 with exchange and rebalancing suppressed. The full-build reference sum of those eight cell volumes is **0.24686200340708858**. After the partial call, `GetAllPointsNo()` and the returned list have only 8 points, and the eight active volumes sum to **0.99999999999999989**. All eight individual volumes differ. The routine has tessellated eight generators over the entire domain, removing the other 24 constraints. This experiment used one MPI rank, so network ordering and redistribution cannot explain the difference.

**Evidence.** `evidence/geometry_partial_volume_repro.cpp` and `geometry_partial_volume.log`, deterministic seed 1729, current source without a patch. `root_v2_subset.log` independently shows owned=8/all=8. A full identity-mask build succeeds. The partial contract is supported by the distinct `allPoints` and `indicesToBuild` API, `participatingIndices`, all-point tree, inactive support logic and the serial implementation retaining all input points. If maintainers intended destructive filtering instead, the API and all supporting machinery need an explicit redesign; silently presenting this as partial rebuilding is unsafe.

**Required fix.** Construct exchange records for every original all-point index k, carrying `participating = activeMask[k]`. Build and validate the mask from the selected indices first. Preserve point coordinates, weight, payload and original index for inactive records as well. When exchange is suppressed, retain all generators on their original rank. When exchange is enabled, route every generator and its participation flag consistently with the chosen ownership policy. Reconstruct `indicesToSelf` and per-peer original indices from `entry.originalIndex`; the current `indicesToWorkWith[localIdx]` translation becomes invalid when records include all points. Keep all-point output arrays the same length and build the active prefix only from participating records after ownership changes.

**Fix dependency.** Ship this with B02, B03 and B04 validation. Retaining all 32 immediately exposes the separate MPI all-tree subscript bug at V:2140. Inactive donor volumes also cannot be read from an array sized only to active cells. A patch that changes only the exchange loop can replace silent wrong geometry with a crash. Define whether inactive geometric fields are valid cached values or must be refreshed, and enforce that definition; adding zero entries does not compute them.

**Acceptance.** On identical generator coordinates, each requested active cell must match the same physical generator's full-build reference volume, centroid, physical neighbors and face geometry within the numerical tolerance defined in the validation section. Do not require the partial active-volume sum to equal the whole domain volume. Test sorted, noncontiguous and permuted subsets, zero active points with nonzero owned support, 1/2/4 ranks, different per-rank masks, periodic boundaries and active-set changes between steps. Check that the global set of generator IDs is conserved and that retained inactive weights/payloads remain attached to their IDs.


### B02 / P1: all-point tree reads active-point storage

**Criticality: P1 / high. Evidence: R for the non-MPI API; source-proven but currently masked in the default MPI partial path.** Valid true subsets cause an out-of-bounds read before geometry can be completed.

**Location.** `source/3D/tessellation/voronoi/Voronoi3D.hpp:4176-4180` loops over `allMyPoints.size()` but reads `activePoints[pointIdx]`. `UpdatePointsTree` repeats the same defect at `:2137-2141`. A 32-generator, 8-active non-MPI call reaches a libstdc++ vector assertion. An empty active subset of a nonempty all-point set also fails. Logs: `root_serial_subset.log`, `root_serial_empty.log`; fixture: `root_api_repro.cpp` compiled without MPI macros. An identity mask with 32 active points succeeds.

**Precise applicability.** The default MPI manager currently drops inactive points first (B01), making both lengths equal. The earlier `root_mpi_subset.log` is a different first-build suppression crash, not evidence that the MPI tree overrun was reached. The initialized v2 subset run returns successfully with wrong geometry. A custom manager preserving inactive points, or the B01 fix, exposes the MPI loop defect. These must not be counted as two independently reproduced MPI crashes.

**Required fix.** Build the all-point tree from `allMyPoints[k]` with index k. Build the active tree from `activePoints[j]` with active index j. Use a shared internal helper for the serial and MPI paths to prevent the two implementations diverging again. Merely changing the loop bound to `activePoints.size()` prevents the read but removes required inactive geometric support, reproducing B01's mathematical error. Sharing the two trees is valid only when their coordinate/index associations are identical; equal lengths alone are insufficient for a permuted full mask. Empty trees need an explicit no-result query contract where queries are possible.

**Acceptance.** Run assertions/ASan on active counts 0, 1, 8 and 32 out of 32, including noncontiguous and permuted masks. Inspect nearest-neighbor results against brute-force searches separately in all and active index spaces. Then compare real active-cell geometry to a full build; a clean tree test alone does not establish a correct partial mesh. Include the repaired MPI retention path before declaring this fixed.


### B03 / P1: reversed active-to-all map

**Criticality: P1 / high. Evidence: R, incorrect public index mapping directly reproduced.** Physical-field corruption in a complete simulation was not separately reproduced; the incorrect mapping itself is the confirmed output.

**Location.** `source/3D/tessellation/voronoi/Voronoi3D.hpp:4103-4108` stores `indicesInAllMyPoints[pointIdx] = pointsCounter`, mapping original all-point index to active index. Consumers require the reverse: the MPI preparation at `:2057`, radius gathering at `:3608-3621`, and `SyncPartialBuildData` at `:5573-5581` treat a key as a current mesh index and its value as an all-point index.

**Trigger and evidence.** Rotate the full mask for 32 points to `{1,2,...,31,0}` and call the non-MPI API. Because both lengths are 32, this bypasses B02 and completes. `GetIndicesInAllPoints()` then reports active 0 -> all 31, while the requested generator is all 1. `root_serial_permutation.log` prints `MAP_MISMATCH active=0 expected_all=1 got=31`, exit 4. An identity mask hides the inversion. Fixture: `root_api_repro.cpp`, mode `permutation`, non-MPI build.

**Required fix.** Initialize the active prefix with `indicesInAllMyPoints[j] = indicesToBuild[j]`. Validate that each input index is in range and define whether duplicate selections are rejected; rejecting them before mutation is the simpler contract. Use checked lookup for required mappings rather than `operator[]`, which can silently insert a default zero value. Preserve additional mappings for locally owned inactive support points inserted later into the Delaunay list; the map is not restricted to the active prefix after ghost discovery. Audit all readers and writers using the shared index contract in this report.

**Acceptance.** Use a 3-cycle or the 32-cycle above, not only a swap: an involution is its own inverse and can conceal this bug. Assert `meshPoint[j] == allPoints[map.at(j)]`, then scatter a distinctive per-generator scalar and centroid through `SyncPartialBuildData` and require exact ID association. Check inactive local support indices too. Protect full identity builds, arbitrary subsets and MPI-generated mappings. Invalid masks should produce a defined error before any rank enters an incompatible collective; the existing invalid-mask assertion is input-hardening evidence, not a separate valid-input bug.


### B04 / P1 conditional: one-way ghost dependencies are erased

**Sources:** `source/3D/tessellation/voronoi/Voronoi3D.hpp:1776`–`:1790`, active call at `:3885`; `SetGhostArray` at `:1969`–`:1987`; centroid synchronization at `:5600`–`:5611`; indexed helper `source/utils/mpi_utils/mpi_exchange.hpp:18`–`:41`.

**Trigger:** rank A owns generators but builds no active cells this partial step; rank B builds an active cell adjacent to one of A's generators. A must answer B's range queries despite having no local queries. B therefore receives a required ghost from A but sends no points to A. This is a valid directed data dependency. The current `EnsureSymmetry` removes a peer if it is absent from either sent peers or all received-peer lists, erasing Nghost mapping as well as export mapping.

**Proof:** `mpi_extracted_helpers.cpp` copies the production EnsureSymmetry body verbatim, changing only the class qualification to a fixture. Starting from a receive-only peer1 with ghost index5, invoking `EnsureSymmetry({},{{1}})` produces:

```text
ONE_WAY_GHOST peer_count=0 ghost_lists=0 expected_peer_count=1
```

The ghost point remains in Delaunay storage; only its owner/field mapping is removed. Later SyncPartialBuildData has no mapping through which to populate its remote centroid. Subsequent physical field exchange through exposed ghost mappings can similarly miss it. This report does not claim an end-to-end partial-build output reproduction: B01-B03 currently obstruct a trustworthy full partial build.

**Fix:** distinguish symmetric *communication peer lists* from symmetric *nonempty data*. Preserve the union of send/receive peers and keep an empty segment for a missing direction, or use separate directed send/receive peer arrays. Existing indexed helper sends an empty message for an empty segment, so a union peer list can be compatible with its matching model. Filtering actual unneeded ghosts should be based on active-cell adjacency/ownership requirements, not reciprocity of point requests.

**Do not apply a one-character OR-to-AND fix alone.** `duplicated_points_` contains all-owned indices. The final volume exchange at `Voronoi3D.hpp:2315` sends from `volume_`, which was sized to active `Norg_` at `:2296`; an inactive donor's all-owned index is not a valid active volume index. Centroid SyncPartialBuildData instead sends from the all-owned `allBuildData` array at `:5600`. The partial-build contract must define valid current/cached fields for inactive donors and supply a correctly mapped source array. Otherwise preserving the previously deleted peer can reveal an out-of-bounds volume access. Coordinate with B01-B03 rather than hiding the dependency by dropping the peer.

**Acceptance:** two ranks, no active cells on one rank but required generators on that rank; one populated rank plus an empty rank; noncontiguous active subsets; active masks differing per rank; fields initialized with distinctive nonzero sentinels. Verify every required ghost's source identity and centroid/volume/state rather than only owned volume sum. All ranks must finish the same protocol. For both directions, assert that expected receive field counts match Nghost lengths, including zero counts. Include full builds as a protected case.


## 5. Copy, build mode and lifecycle errors


### B05 / P1: built copies have null query trees

**Status:** directly reproduced segmentation fault on the current source using a 32-point valid, nonperiodic mesh built with `BuildParallel`, one local MPI process. The original grid returns containing cell 0; the copy still reports 32 points and volume 1; querying the same point on the copy segfaults.

**Locations:**

- `source/3D/tessellation/voronoi/Voronoi3D.hpp:378` and `:379`: `myPointsTree` and `allMyPointsTree` members.
- `source/3D/tessellation/voronoi/Voronoi3D.hpp:4984`: copy constructor; its initializer list ends at :4998 without initializing either tree.
- `source/3D/tessellation/voronoi/Voronoi3D.hpp:4827`: `GetContainingCell` dereferences `myPointsTree` in the ordinary nonperiodic path.
- `source/3D/tessellation/voronoi/Voronoi3D.hpp:3054`: periodic image resolution also dereferences the missing active tree.
- `source/3D/tessellation/Voronoi3D.hpp:223`: RICH adapter `clone()` constructs `new Voronoi3D(*this)`.
- `source/newtonian/three_dimensional/AMR3D.cpp:3094`: RICH AMR actually creates tessellation clones. This establishes that cloning is supported/used, but does not establish that this exact AMR call subsequently invokes point location; avoid claiming an AMR crash was reproduced.

**Trigger:** copy or clone an already built mesh, then call `GetContainingCell` or a periodic operation that needs `ResolvePeriodicImageIndex` before rebuilding the copy. Destroying the source is not necessary to trigger the null dereference.

**Cause:** `shared_ptr` members omitted from the explicit copy constructor default-initialize to null. The copied geometric arrays make the object appear like a valid built snapshot, while its query support is absent. Merely sharing `rangeFinder` in the MPI copy constructor does not populate these separate members. That shared finder itself contains a raw tree pointer, so using it after the source drops its trees would also need an ownership audit.

**Evidence:** `docs/madvoro_bugs/evidence/geometry_repro.cpp`, mode `copy`; `geometry_copy.log`. Output immediately before failure:

```text
source containing cell=0
copy points=32 volume=1
copy containing cell=... SIGSEGV ... address (nil)
```

**Required fix:** specify a complete snapshot-copy contract, then copy/rebuild all query state consistently. The safest independent-copy implementation rebuilds the all-point tree from copied `allMyPoints[k]` indexed by k and the active tree from copied active mesh points indexed by j, respecting the active-to-all map; then builds a finder whose tree and point storage are owned by the copy. Sharing immutable tree ownership is possible for a smaller patch, but the finder must retain the same owned tree lifetime and the current mutable tree traversal stack makes concurrent queries unsafe. Do not create a tree copy using OctTree's implicit shallow pointer copy. Audit `all_CM`, `periodicImagePhysical_`, `buildGeneration_`, real ghost maps and MPI/non-MPI member parity when repairing the explicit constructor. Non-MPI copies currently also omit allMyPoints/radiuses/rangeFinder because those initializers sit inside an MPI preprocessor block.

Explicitly define or delete copy assignment alongside the constructor: implicit assignment presently shares mutable pointsManager by shared_ptr assignment while the copy constructor attempts a clone. Leaving the two operations with inconsistent ownership semantics is unsafe. This assignment issue is source-backed and should be tested independently; it was not reproduced as wrong mesh output here.

**Regression:** fresh built copy and RICH clone; query every original generator; random interior points; periodic image resolution; true active subsets; source destroyed; source rebuilt while copy remains queryable; copy rebuilt independently; serial and MPI configurations. Assert source/copy geometry and physical mappings agree before independent mutation. Use ASan for source-destruction tests and a small clone API integration test.

**Why P1:** an exposed, supported copy/clone operation produces an invalid object that crashes on ordinary queries. The impact is conditional on querying the copy; the source mesh itself remains usable.


### B06 / P2: unbuilt copies dereference a null balancer

**Status:** directly reproduced segmentation fault before any tessellation work. It is a distinct precondition failure from B05; fixing missing trees alone does not repair this case.

**Locations:**

- `source/3D/tessellation/voronoi/Voronoi3D.hpp:1420`: constructor creates a HilbertPointsManager.
- `source/3D/tessellation/voronoi/Voronoi3D.hpp:4995`: copy constructor unconditionally calls manager clone.
- `source/3D/tessellation/MeshDecomposer3D/points_manager/HilbertPointsManager.hpp:80`: loadBalancer defaults to nullptr.
- `source/3D/tessellation/MeshDecomposer3D/points_manager/HilbertPointsManager.hpp:115` through `:120`: clone constructs a new manager and unconditionally evaluates `this->loadBalancer->clone()`.

**Trigger:** after MPI_Init, construct a grid with valid box coordinates and immediately copy it, before the first BuildParallel. The operation is legal according to the public copy constructor signature and has no documented built-state precondition.

**Evidence:** `geometry_repro.cpp`, mode `unbuilt_copy`; `geometry_unbuilt_copy.log`, exit by SIGSEGV. `addr2line` identifies the crashing frame as HilbertPointsManager::clone at line 120. The source line proves the null precondition; no geometric degeneracy or communication is involved.

**Required fix:** make manager cloning preserve both initialized and uninitialized state. Guard the nullable loadBalancer, environment agent and pending indexing state, and copy the base PointsManager lifecycle/cost fields rather than reconstructing some fields as default zero. Preserve the original communicator. Do not manufacture a fake load balancer simply to make the copy nonnull. Also guard a nullable pointsManager in the Voronoi copy constructor if other API paths intentionally support serial mode with a null manager. Public uninitialized copies should remain uninitialized and able to build later.

**Regression:** unbuilt grid copy under MPI; copy after SetKernel but before build; standalone manager clone with null loadBalancer; copied object then first BuildParallel; initialized manager clone and subsequent rebalance decisions. Test 1 and 2 ranks consistently. If copying unbuilt grids is deliberately unsupported, enforce that through a clear exception/documented API rather than a null dereference, but allowing ordinary value copying is preferable.

**Why P2:** reliable crash, but requires copying an unbuilt object rather than the usual built-state snapshot path.


### B07 / P1: mixed suppression selects stale weight storage

**Location and source chain**

- `source/3D/tessellation/voronoi/Voronoi3D.hpp:2019-2021`: combines every rank's exchange permission with MPI_LAND, producing a global `allowExchange`.
- Same file `2023-2024`: resizes `allPointsWeights` only if this rank's original `suppressExchange` argument is true.
- Same file `2030`: when global `allowExchange` is false, passes `this->allPointsWeights` to PointsManager on every rank, even ranks whose local flag was false.
- `source/3D/tessellation/MeshDecomposer3D/points_manager/PointsManager.hpp:290-296`: selected original indices index `allWeights[pointIdx]` without a bounds check.

**Trigger.** Start with an initialized MPI mesh. Increase local input length on a rank whose local suppressExchange=false, while another rank requests true. Pass a correctly sized weights vector on every rank. Global suppression takes effect but the nonsuppressing rank retains the previous shorter cached weights.

**Actual reproducer.** `docs/madvoro_bugs/evidence/integration_mixed_suppression.cpp` builds 128 deterministic random points with two ranks, yielding 64 points each. Each rank then adds one point and supplies 65 valid weights. Rank 0 requests suppressExchange=true and rank 1 requests false. Rebalance is explicitly suppressed on both ranks to isolate the exchange-flag path. The second call fails on rank 1:

```text
rank=0 count_before=64 input_count=65 weight_count=65 suppressExchange=1 entering_second_build
rank=1 count_before=64 input_count=65 weight_count=65 suppressExchange=0 entering_second_build
std::vector<double>::operator[] ... Assertion '__n < this->size()' failed.
```

Raw log: `evidence/integration_mixed_suppression.log`. Exit: 134. `_GLIBCXX_ASSERTIONS` catches the invalid access; unchecked production builds have undefined behavior and can silently import an invalid point weight. The optimized debug stack symbolizer points around the adjacent payload access at PM:296, but the assertion message explicitly names vector<double>, and the source-selected stale vector explains that exact access. Symbolized frames are in `integration_mixed_suppression_frames.txt`.

The exact same executable with argument `all_suppress` requests suppression on both ranks and succeeds with 65 returned points each: `integration_all_suppression_control.log`, exit 0. This excludes invalid point input, insufficient caller weights, and lack of ordinary suppressed-exchange support as the explanation.

**Fix contract.** Base all state preparation decisions on the globally agreed `allowExchange`, not the local argument. If retaining cached weights is intentional when exchange is suppressed, initialize/resize/remap them on **every** globally suppressed rank before use. At minimum the existing resize condition must follow `!allowExchange`; more broadly define how caller-supplied weights relate to cached weights after insertion/removal/permutation. Do not silently reinterpret those semantics as part of the safety patch. Collectively validate the actual selected weight vector and selected indices before starting communication; one rank throwing before other ranks enter collectives requires a consistent error policy.

**Regression tests.** Mixed local flags with input length growth on the nonsuppressing rank; shrink; unchanged length; full permutation after preparing identity metadata; all flags true/false; empty rank; different ranks choosing the suppressor; caller weights valid but distinctive. Require no out-of-bounds access under libstdc++ assertions/ASan, globally matching control flow, correct stored weight values under the documented policy, and unchanged point/payload mappings. Do not fix only the helper's subscript with `.at()`; that merely converts corruption to rank-local exception/deadlock.


### B08 / P2: serial Build selects distributed state

**Criticality: P2 / medium. Evidence: R, public API failure on a fresh MPI-enabled object.** This does not mean `BuildParallel` fails with one rank; that path is a successful control.

**Location.** `source/3D/tessellation/voronoi/Voronoi3D.hpp:439-443` exposes `Build` in both compilation modes and forwards to `BuildPartially`. At `:4208` the serial path deliberately supplies `MPI_COMM_SELF`. However `BringGhostPointsToBuild` at `:3582` determines serial mode from `pointsManager == nullptr`. The MPI constructor at `:1420` creates a nonnull manager even before any distributed setup. At `:3662-3665`, the fresh manager's absent environment agent triggers an exception.

**Reproduction.** After MPI_Init on one rank, construct a unit-box grid and call `Build` on 32 valid points. `root_v2_mpi_serial.log` reports `BringGhostPointsToBuild: environment agent is null`, caught by the v2 fixture, exit 3. The test did not rebuild distributed state first. If the manager was initialized earlier, the guard can pass while referring to stale distributed routing; that broader outcome is a source concern, not a reproduced failure here.

**Required fix.** Pass an explicit build execution mode through preparation, ghost discovery, centroid synchronization and final exchange. A serial build must use only its local generator set, local range answers and no cross-rank field protocol. Audit `SyncPartialBuildData` at `:5600`, whose MPI-compiled branch currently exchanges without checking a build mode. Select MPI_COMM_SELF only where MPI operations are actually required by reused components. Do not temporarily null the manager and leave later unconditional dereferences or old ghost maps alive. Preserve configured manager state for a later distributed build while invalidating mesh-specific routing from the previous build.

**Acceptance.** Compile with MPI and call serial Build on one rank; then test independent serial builds with different data on every rank of a multi-rank world, and only one world rank building while others perform unrelated work. No WORLD collective may be required by the local build. Test transitions serial -> parallel -> serial and exception cleanup. If the intended public API disallows serial builds in MPI binaries, explicitly reject/document that at the entry point instead; current source comments and MPI_COMM_SELF routing strongly suggest local mode was intended.


### B09 / P2: first suppressed build skips initialization

**Criticality: P2 / medium. Evidence: R, null-pointer crash.** The supported lifecycle is incompletely documented, so this is at minimum a missing precondition check rather than a claim that meaningful initial distributed cuts can be inferred without configuration.

**Location.** `source/3D/tessellation/MeshDecomposer3D/points_manager/HilbertPointsManager.hpp:283-291` initializes a missing balancer inside the exchange-enabled branch. The no-exchange branch at `:295-303` bypasses that work, then `:315` still calls `CreateEnvironmentAgent`. The latter at `:95-105` selects a Hilbert environment even when `loadBalancer` is null. The fresh MadVoro constructor has not supplied a usable load balancer.

**Reproduction.** On a new MPI grid, `BuildPartiallyParallel(points, weights, identityMask, true, true)` with 32 valid points exits 139. Source: `root_api_repro.cpp first_suppressed`; log: `root_v2_first_suppressed.log`. An ordinary first `BuildParallel`, followed by the same suppression flags, succeeds for the full mask (`root_v2_identity.log`). Older root MPI identity/subset logs exercise this same fresh-state bug and are superseded by the v2 controls.

**Required fix.** Check manager readiness before any no-exchange build. Either initialize a valid environment without migrating the supplied ownership, or reject the first suppressed build collectively with a precise message requiring an initial normal build or a preset compatible balancer. For one rank, a whole-domain environment is sufficient. For multiple ranks, do not fabricate cuts that contradict caller ownership; use a supplied routing environment or require setup. Separate 'balancer configured', 'environment initialized' and 'mesh built' states. Coordinate with B06 and B12 lifecycle repairs so a nonnull pointer is never the only readiness test.

**Acceptance.** Fresh full/partial suppressed calls on 1 and 2 ranks must succeed under a defined ownership contract or terminate with a consistent documented error, never dereference null or hang peers. Test preset balancer, empty input, warm suppressed rebuild and mixed suppression flags (B07). A fix that always enables migration despite `suppressExchange=true` violates the request.


### B10 / P2: releasing memory changes a custom domain

**Status:** directly reproduced change in public geometric classification, without rebuilding or changing the box explicitly. Full rectangular release/rebuild also tested and still gives volume 1; do not claim ReleaseMemory breaks every rebuild.

**Locations:**

- `source/3D/tessellation/voronoi/Voronoi3D.hpp:1375`: constructor accepting custom boundary faces; :1377 stores the faces and :1381-1393 computes their enclosing bounds.
- `source/3D/tessellation/voronoi/Voronoi3D.hpp:3984`: ReleaseMemory implementation.
- `source/3D/tessellation/voronoi/Voronoi3D.hpp:3997`: releases `box_faces_`.
- `source/3D/tessellation/voronoi/Voronoi3D.hpp:5163` through `:5168`: `IsPointOutsideBox` falls back to the enclosing ll/ur rectangular box when the face list is empty.
- `source/3D/tessellation/voronoi/Voronoi3D.hpp:1530`: later builds similarly replace the empty face list with BuildBox(ll,ur).

**Trigger:** construct a nonrectangular convex domain, call ReleaseMemory to release mesh storage, then classify points or rebuild without resupplying the original custom faces. A tetrahedron with vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1) encloses volume 1/6 while its ll/ur box encloses volume 1.

**Evidence:** `geometry_repro.cpp`, mode `custom_release`; `geometry_custom_release.log`:

```text
point outside tetra before release=1
point outside tetra after release=0
```

The test point is (0.8,0.8,0.8), strictly outside the tetrahedron and strictly inside its enclosing box. No invalid boundary coordinates are involved. The minimal test uses domain queries and does not claim a rebuilt tetrahedral volume was measured. The next build's substitution is source-proven by :1530.

**Related state evidence:** after building 32 points then releasing storage, `GetPointNo()` still reports 32 while `GetTotalFacesNumber()` is zero (`geometry_release.log`). ReleaseMemory does not reset Norg_, bigtet_, trees, finder, or the cached periodic physical-index vector. This makes the lifecycle state internally misleading and leaves some storage alive. A subsequent standard rectangular full build was successful in the smoke run; post-release mesh queries should either be documented invalid and rejected, or observe a well-defined empty state.

**Required fix:** separate persistent domain/configuration data from rebuildable mesh storage. Preserve custom boundary faces, periodic flags and box/kernel configuration when releasing mesh memory. Reset logical mesh counts, invalidate generation and caches, and release/rebind tree/finder owners so an unbuilt state is explicit. If the intended operation is a full destructive reset, expose/document a separate ResetDomain operation and require new domain configuration; silently substituting a different physical boundary is not acceptable.

**Regression:** custom tetrahedron and slanted/hexahedral box classification before/after release; rebuild with identical points and compare volume, face planes and boundary flags; periodic rectangular configuration retained; empty/post-release API behavior; repeated ReleaseMemory; then a normal rebuild. Preserve custom geometry while releasing all genuinely temporary buffers.

**Why P2:** silently changes domain semantics on a less frequent lifecycle path. Severity becomes higher if callers use this operation during long simulations with custom domains, but no such RICH production call was found in this audit.


### B11 / P2: continuity query indexes an empty mesh

**Criticality: P2 / medium. Evidence: R, bounds assertion after a successful empty rebuild.** Empty ownership is a normal distributed state; this finding concerns the public continuity query, not proof of an automatic failure in every empty-rank build.

**Location.** `source/3D/tessellation/voronoi/Voronoi3D.hpp:5202-5207` allocates `reached(Norg_, false)`, immediately sets `reached[0] = true` and pushes cell 0. With Norg_=0, the access is invalid.

**Reproduction.** Initialize an MPI1 grid normally, rebuild with no points and both suppression flags true, then call `CheckContinuityOfZone`. The rebuild returns and reports owned=0; the query then fails in `std::vector<bool>::operator[]`, exit 134. `root_v2_empty_continuity.log` preserves the marker and assertion. This avoids conflating the query bug with first-build initialization or partial active-tree failures.

**Required fix.** Handle Norg_=0 before creating a traversal seed. Returning true is consistent with the current final `all_of` expression's vacuous truth; if the application needs 'no zone' as a distinct outcome, document that result instead. Do not call GetNeighbors(0) or inspect old geometry on an empty mesh. Check B10's post-release state: a stale nonzero Norg_ cannot be repaired by an empty check alone.

**Acceptance.** Empty initialized grid, valid empty rebuild, one cell, connected region and a known disconnected region. Include empty ranks in a larger job and verify the local query itself introduces no collective. Test after release according to the chosen lifecycle contract.


### B12 / P2: kernel replacement cannot rebuild its converter

**Location and source chain**

- `source/3D/tessellation/voronoi/Voronoi3D.hpp:5510-5520`: public `SetKernel` delegates to `HilbertPointsManager::setIndexing`.
- `source/3D/tessellation/MeshDecomposer3D/points_manager/HilbertPointsManager.hpp:260-270`: an existing Hilbert balancer receives setIndexing, and the environment agent is cleared.
- `source/3D/tessellation/MeshDecomposer3D/load_balancing/HilbertLoadBalancer.hpp:290-294`: stores new indexing and sets `convertor=nullptr`.
- `HilbertPointsManager.hpp:283-291`: initialization constructs a new balancer only if the balancer pointer is null. Here the pointer remains nonnull, so initialization immediately calls rebalance on the converter-less balancer.
- `HilbertLoadBalancer.hpp:160-163`: rebalance throws if converter is null.

**Trigger.** Successfully initialize the normal Hilbert manager, then set a valid kernel and update again. Even replacing the kernel with a fresh Identity object fails; this is not a malformed transform.

**Actual reproducer.** `evidence/integration_load_balancer_lifecycle.cpp`, mode `kernel`, one MPI rank:

```text
first_update_succeeded changing_to_identity_kernel
caught_exception=HilbertLoadBalancer::rebalance: convertor was not initialized yet
```

Log: `integration_kernel_change.log`. Exit: 3. The reproducer calls the dependency methods directly to isolate the lifecycle, not a complete RICH simulation. The MadVoro SetKernel delegation is source-verified; no separate core SetKernel runtime was needed for this claim. Setting the kernel before initial construction follows the pendingIndexing path and is a distinct case that may work.

**Impact.** Valid dynamic kernel replacement cannot be followed by a normal rebuild. An implementation might work around it by re-creating the entire grid, but the public setter is currently an invalidating operation without a viable reconstruction path. This is P2; no physical corruption was demonstrated for this case because the next rebalance throws explicitly.

**Fix contract.** Preserve the new kernel as pending state and rebuild the converter from current bounds/points before using it. Either recreate the Hilbert balancer on the next initialization, or explicitly reconstruct its converter and recompute cuts. Old cuts in an old key space must not be reused blindly. Preserve the communicator and documented imbalance settings when recreating state; invalidate/rebuild the environment agent and routing caches consistently. Handle identical kernel objects as valid too. Do not simply remove the null-converter check.

**Regression tests.** Set Identity before initial build; replace with Identity after first build; replace with a valid nonidentity transform; repeat twice; clone after replacement; SetBox plus kernel transition; multiple ranks with synchronized setter/update; confirm final owner assignment and geometry against a fresh grid configured with the target kernel.


## 6. Periodic identity and geometric helper errors


### B13 / P2: remote image resolves to an unrelated local point

**Criticality: P2 / medium. Evidence: R, wrong physical identity from the public API.** Severity rises to P1 for a caller that trusts the returned local index to fetch physical state without checking ownership. Such downstream corruption was not demonstrated in current RICH callers.

**Location.** `source/3D/tessellation/voronoi/Voronoi3D.hpp:3028-3061`, especially `:3054`, wraps the image's physical coordinate then returns `myPointsTree->closestPoint(wrapped).getIndex()`. That tree contains only owned active points. For an image of a remote generator, a nearest owned point always exists on a nonempty rank but is not the generator's physical preimage. The result is cached without verifying identity or distance.

**Reproduction.** Build 32 deterministic points on 2 MPI ranks with all axes periodic, inspect each periodic image and compare the returned owned point to the wrapped physical coordinate. Both ranks report a mismatch in `root_v2_periodic_resolution.log`, exit 5. One example on rank 1: physical `(0.446993,0.835512,0.0893802)` resolves to local index 4 at `(0.650812,0.618903,0.0565723)`. This is a large coordinate difference, not a rounding tie. The fixture only reports a failure when the method returns an owned index inconsistent with the image coordinate.

**Existing caller protection.** `MockMesh` at V:2539-2556 verifies coordinate distance before accepting its local preimage. `source/newtonian/three_dimensional/LinearGauss3D.cpp:49-51` returns recognized MPI ghosts before local resolution. `source/monte/utils/GhostMap.hpp:47-50` accepts an owned result from this helper, but its transport caller at `source/monte/manager/MonteCarloTransport.hpp:258-261` first checks the MPI ghost map. No complete Monte Carlo transport failure was reproduced. These guards prevent inferring that every periodic RICH simulation currently misroutes; they do not repair the public helper's identity contract.

**Required fix.** Carry physical ownership/identity and translation with each periodic image at insertion, and resolve that identity to the correct owned or existing ghost representation. Preserve an unresolved image index, or introduce an explicit optional result, when no physical representation exists locally. A conservative first patch may accept a nearest candidate only after a scale-aware coordinate match and ownership validation; it must not silently alias distinct coincident IDs. Define cache invalidation for rebuild, migration, translation change, box change and copy. Do not deduplicate distinct periodic translations just because they have the same physical generator.

**Acceptance.** One/two/four ranks, local and remote source images, all periodic masks, face/edge/corner images, moved generators, repeated builds and source clones. Assert physical source identity, wrapped coordinate and translation, not merely that a returned index is in range. Inspect guarded RICH consumers before changing unresolved-return semantics.


### B14 / P2: mixed-periodic image queries are duplicated

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


### B15 / P2, source-only: face cleanup retries with stale state

**Status:** source-proven control-flow defect in active face construction. No full valid-mesh input triggering this path was reproduced. A bounded synthetic six-vertex polygon search did not find a case where the one-line candidate fix changed throw/success; that negative experiment is recorded rather than presented as proof of a production failure.

**Locations:**

- `source/3D/tessellation/voronoi/Voronoi3D.hpp:992`: CleanSameLine.
- `:1034-1047`: Nindeces tracks the number of vertices remaining after the first cleanup.
- `:1050-1057`: on failure Nindeces<3, the function restores `indeces = old` but does not restore Nindeces to the restored vector size before its alternate-normal loop.
- `:1068`: the loop can update Nindeces when it erases an element.
- `:1073-1096`: if Nindeces remains below 3 it throws Bad CleanSameLine.
- `:4357-4365`: BuildVoronoi catches the exception and skips this face candidate.

**Precise defect:** the retry uses the stale length of the failed polygon rather than the restored polygon. If the alternate loop erases no point, the stale Nindeces<3 makes the retry unconditionally fail despite the restored vector having N>=3 vertices. If it erases a point, Nindeces is refreshed to the vector's size at :1068, which can be >=3, so it is too strong to say every alternate-normal attempt necessarily throws. The guaranteed bug is incorrect loop extent and neighbor indexing before the first erase, and unconditional false failure when no erase occurs. Any report must preserve this distinction.

There is an additional source-proven candidate-selection bug at :1008-1025: `second_max_value` starts equal to the first maximum and `second_max_index` starts 0. If area_vec_temp[0] is the strict largest value, no later lower value can beat second_max_value; the supposed second-best normal remains the largest normal at index 0. If the first two values tie and no larger arrives, the second distinct index can also be lost. This defeats the intended alternate candidate even after fixing the restored length.

**Required fix:** after restoring `indeces = old`, set `Nindeces = indeces.size()` before computing retry neighbors or loop extent. Select two distinct normal candidates using an explicit best/second-best initialization and tie rule; do not initialize the second-best value to the current best without a distinct eligible index. Validate N>=3 and zero-area normal handling at the helper boundary. Preserve candidate face rejection semantics unless reference geometry proves a face is valid; do not simply suppress exceptions to hide the issue.

**Regression:** unit-test best/second-best selection when maximum occurs at index 0, at a later index, and with ties. Construct direct helper cases that enter recovery, verify the second pass visits all restored vertices and uses the restored cyclic length, and compare against a simple reference cleanup. Then stress near-collinear circumcenter rings, high-valence coarse/fine seams, ring rotations and near-planar input from actual tetrahedra. Add a diagnostic counter for recovery entry and skipped face so production occurrence can be quantified. A valid full-mesh reproduction is still required before claiming this defect causes a particular observed volume error.

**Evidence artifact:** `geometry_cleanup_repro.cpp` contains a test-only verbatim extraction of the production helper plus a renamed one-line candidate fix; production code is unchanged. `geometry_cleanup.log` says no distinguishing polygon found in the bounded search. The extraction documents the exact investigated branch; it is not a successful regression test or complete fix.

**Why P2:** wrong branch bookkeeping in active geometry cleanup can reject recoverable face candidates or use incorrect cyclic neighbors, but its production triggering frequency and output impact remain unmeasured.


### B16 / P3: standalone face centroid lacks area normalization

**Status:** directly reproduced wrong numerical result from the public free helper. No internal caller was found in the current MadVoro/RICH tree, so do not claim this currently corrupts BuildVoronoi's face centers; those use a different routine.

**Location:** `source/3D/tessellation/voronoi/elementary/Face3D.hpp:55` through `:67`.

**Trigger:** any nonzero polygon area other than 1, e.g. triangle (0,0,0),(2,0,0),(0,2,0), area 2.

**Evidence:** `geometry_face_centroid.cpp` includes and calls the production helper; `geometry_face_centroid.log`:

```text
area=2 computed=1.3333333333333333,1.3333333333333333,0 expected=0.66666666666666663,0.66666666666666663,0
```

**Cause:** the function sums `triangle_area * triangle_centroid` but never divides by the sum of triangle areas. It therefore returns a first area moment, inconsistent with the function name. Uniform scaling changes the result cubically rather than linearly.

**Required fix:** accumulate area and area-weighted centroid in one consistent triangulation; divide the moment by total area; reject/document zero-area input. If the API intentionally wants an area moment, rename it to an explicit moment helper and provide a correctly named centroid function, taking care with external users.

**Regression:** triangles and convex quadrilaterals with area 0.5,1,2; translated/scaled copies; centroid affine behavior; degenerate-area behavior. This fix should not alter main Voronoi face-center code unless a deliberate refactor is separately validated.


## 7. Shared dependencies: active and configuration-specific defects

D01 is used by the active ghost-query transport. D02-D06 have concrete component failures under the stated communicator, backend, payload or option; ordinary current MadVoro use of those exact configurations was not established. Each repair belongs in its owning dependency, followed by a reviewed submodule-pointer update.


### D01 / P1 conditional: request completion compaction

**Source:** `source/utils/mpi_utils/BuffersManager.hpp:365`–`:385`; specifically reverse iteration at `:368`, vector access at `:371`, swap/pop at `:382`–`:383` and map updates at `:372`–`:380`. This method is called on the active ghost-query path through `BuffersManagerQueryAgent.hpp:282`–`:283`.

**Trigger and failure:** at least two sends complete in one MPI_Testsome call, and the returned indices are not ascending. For a request vector with three entries and completion indices `[2,0]`, the current reverse loop processes 0 first. It swap-pops slot 2 into slot 0 and shrinks to 2 entries. Processing the next index 2 then calls `.at(2)` on a 2-entry vector and throws. Other permutations can also misassociate buffers before an exception.

**Proof:** `mpi_dependency_repro.cpp compaction` includes the actual unchanged production header and calls its `CleanSendRequests`. A shim injects exactly one post-completion list `[2,0]`; this is a bookkeeping fixture, not a natural MPI-network reproduction. The output in `mpi_compaction.log` is:

```text
COMP_ACTION_DEFECT completion_indices=[2,0] exception=vector::_M_range_check:
__n (which is 2) >= this->size() (which is 2) remaining_slots=2
```

The [MPI Forum multiple-completion contract](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf) specifies completed operation indices and associated statuses but no ascending-order guarantee. The implementation must handle every allowed index ordering. This is stronger than merely asserting the code looks suspicious, but weaker than claiming this Open MPI build naturally returns that order.

**Fix:** collect completion indices and sort them descending numerically before mutating the request array; if statuses are used, sort paired index/status records. Alternatively retain stable request slots and a free list. For each removal, release exactly that request's buffer, update the moved last-slot map exactly once, then pop. Check `outcount` before iteration, including 0 and MPI_UNDEFINED, and choose an explicit policy for MPI_ERR_IN_STATUS. Do not reverse a returned array and assume that sorts it.

**Acceptance:** exhaustive subsets/permutations for request arrays 0–8, including cases containing the last slot and nonlast slots; verify remaining request identity, buffer identity, active count and free-buffer uniqueness. Then run actual multi-rank eager/rendezvous traffic, delayed peers, and repeated query batches. Keep synthetic index-order perturbation in the test harness because a single MPI implementation may always return ascending indices and conceal the defect.


### D02 / P1: manager and balancer use different communicators

**Location and source chain**

- `source/3D/tessellation/MeshDecomposer3D/points_manager/HilbertPointsManager.hpp:88-90` accepts and forwards a caller MPI communicator.
- `source/3D/tessellation/MeshDecomposer3D/points_manager/PointsManager.hpp:88-93` sets local rank/size from that communicator.
- `HilbertPointsManager.hpp:289` constructs `HilbertLoadBalancer` without this communicator.
- `source/3D/tessellation/MeshDecomposer3D/load_balancing/HilbertLoadBalancer.hpp:47-60` has no communicator parameter in the used constructors and forwards only boundaries.
- `source/3D/tessellation/MeshDecomposer3D/load_balancing/CurveLoadBalancer.hpp:15-16` calls `LoadBalancer<PointT>()`, selecting its default.
- `source/3D/tessellation/MeshDecomposer3D/load_balancing/LoadBalancer.hpp:19-23` defaults to MPI_COMM_WORLD and records world rank/size.
- Owner lookup clamps against the balancer's world size (`CurveLoadBalancer.hpp:38`), while `source/utils/mpi_utils/exchange.hpp:30,39` allocates destination arrays for the manager's communicator and indexes by that owner.

**Trigger.** In a two-rank world, create an independent HilbertPointsManager on `MPI_COMM_SELF` on each rank, then update valid points. Each manager's communicator has one rank, but the shared-world balancer can return rank 1. If only one subgroup enters the operation, unintended WORLD collectives can hang even before routing; the reproduced case has both ranks entering to reach the invalid route deterministically.

**Actual reproducer.** `evidence/integration_points_manager_comm.cpp`, two ranks. World rank 0 returns its two points; rank 1 aborts in a vector-of-destination-payload-vectors subscript:

```text
world_rank=1 entering MPI_COMM_SELF manager update
world_rank=0 entering MPI_COMM_SELF manager update
Weighted borders determined in 1 pass.
std::vector<std::vector<ExchangePoint<...>>>::operator[] ... Assertion '__n < this->size()' failed.
world_rank=0 result_count=2
```

Log: `integration_points_manager_comm.log`. Exit: 134. This reproduces the dependency's explicit communicator contract. **Do not describe it as proof that arbitrary subcommunicators are supported by the entire current MadVoro engine**: the core has many explicit WORLD callsites and no general communicator constructor. The error is valid in the independently usable dependency and can also affect custom manager injection if a caller reasonably expects its supplied communicator to be honored.

**Fix contract.** Thread the communicator through CurveLoadBalancer and both HilbertLoadBalancer constructors; pass it from HPM initialization; preserve it through HilbertLoadBalancer::clone and any factory/restoration path. Audit every collective and returned owner for the same local rank namespace. Add a cheap assertion/checked error that destination owner lies in `[0,manager_size)` before indexing, but communicator correctness is the real fix. Never fix by clamping world owner indices into subgroup size: that discards decomposition semantics and can hide world-collective hangs.

**Regression tests.** MPI_COMM_SELF on every world rank, disjoint split communicators of different sizes, reordered communicator rank order, a subgroup performing updates while other world ranks do unrelated work, clone inside subgroup, empty local ranks and supplied/custom load balancers. Check that no WORLD collective is invoked by the dependency in subgroup tests and that all returned ownership values belong to the subgroup.


### D03 / P2: one-rank 1D bins violate the query invariant

**Location and source chain**

- `source/3D/tessellation/MeshDecomposer3D/load_balancing/OneDimensionalLoadBalancer.hpp:58-67`: sets `bins_` to the result of `getWeightedBorders3<double>`.
- `source/3D/tessellation/MeshDecomposer3D/balance/weightedBalance3.hpp:218-220`: direct route returns an empty list for communicator size <=1. Root fallback has the same return at 121-123.
- `OneDimensionalLoadBalancer.hpp:87-94`: `getIntersectingRanks` throws unless `bins_.size()` equals the communicator size.
- `getOwner` at 72-75 already accepts empty bins and returns owner 0, so the two query APIs disagree on the state produced by rebalance.

**Trigger and reproducer.** On one rank, construct a valid unit-box OneDimensionalLoadBalancer, call rebalance on two ordinary points with unit weights, then query the box's center with radius 0.1. `integration_load_balancer_lifecycle.cpp onedim` produces:

```text
bins_after_rebalance=0
caught_exception=OneDimensionalLoadBalancer: bin count does not match MPI size
  Bin count: 0
  MPI size: 1
```

Log: `integration_onedim_single_rank.log`. Exit: 3. This is a dependency API failure after a successful normal setup; a full MadVoro geometry build was not needed. The generic environment adapter forwards intersection queries to this API when providesIntersectingRanks() is true (`environment/PlainDistributedOctEnvAgent.hpp:86-90`). The actual single-rank core search may bypass remote routing, so do not claim every MPI1 MadVoro build crashes; the intersection API does.

**Fix contract.** Define the bin/sentinel convention for one rank and keep constructors, rebalance, getOwner, getIntersectingRanks and restart IO consistent. Options: store one valid upper-domain sentinel during single-rank rebalance, or explicitly handle size==1 in the intersection query using the real domain box. Preserve conservative box/sphere intersection and empty-domain semantics. Do not invent P-1 or P requirements inconsistently: the existing multi-rank helper appends an extra sentinel.

**Regression tests.** One rank, each axis, inner/intersecting/outside spheres, explicit and generated bins, clone and IO roundtrip; two/three ranks with repeated cuts; all-empty or all-zero-weight policy as separately specified. Query owners and candidate-rank sets should remain consistent after rebalance and changeBox.


### D04 / P2: buffer assembly exceeds receive capacity

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


### D05 / P2: eight-byte payload never dispatches

**Source:** `source/utils/mpi_utils/BuffersManager.hpp:287`–`:291` writes a header counting payload bytes only. `ShouldSend` at `:312` compares that count to `sizeof(size_t)` and returns false, intending to recognize empty storage. An empty payload counter is0, not8 on this host. The check bypasses both byte threshold and cycle-age dispatch checks.

**Actual reproduction:**2 ranks, `BuffersManager<uint64_t>` with buffer1024, threshold64, cycle limit2. Rank0 adds one uint64_t to rank1; both call HandleIncomingOutcoming10,000 times. `mpi_scalar8.log` shows sender sent_messages0/pending_buffers1 and receiver callbacks0. Destroy does not flush the remaining rank buffer; its send counters are0, so it cancels receives and silently drops the buffered value.

**Applicability:** generic shared dependency defect. Current MadVoro SubQuery and AnswerInfo records have payloads larger than8; this exact stall is not established in their ordinary path. Any configuration/type yielding an accumulated serialized payload of8 bytes can trigger it, including two4-byte records.

**Fix:** use `payloadBytes==0` to test emptiness, keeping all other thresholds in the same byte convention. Add an explicit flush-complete contract so Destroy cannot silently abandon nonempty pending buffers; be careful that destructors cannot safely invent new collective traffic after peers leave or MPI finalizes. Test finalization through a collective finish API and assert no pending payload after successful completion.

**Acceptance:** one and multiple uint32/uint64 payloads, idle tails, zero-length Serializable, delay-based flush below threshold, and normal query records. Require exact delivery and pending/outgoing counts0 after finish.


### D06 / P2: flattened query results lose self answers

**Source:** `source/utils/mpi_utils/queryAgent/BuffersManagerQueryAgent.hpp:145`–`:158` supports sendToSelf=true; `:120`–`:121` stores received self answers in both result representations. Finalization at `:305`–`:309` unconditionally skips the local rank when creating recvProcessorsRanks. Rearrangement at `:174`–`:189` only emits ranks in that list.

**Actual reproduction:** one rank, trivial serializable query123, TalkAgent returning rank0, AnswerAgent returning `{123}`, sendToSelf=true. `mpi_self.log` reports:

```text
SELF per_query=1 by_rank=1 flattened=0 recv_peers=0
```

The answer was transmitted and accounted for; flattening omits it. **Current MadVoro explicitly selects false** (`range/BigRangeAgent.hpp:272`, `range/SmallRangeAgent.hpp:184`), so it is not a present MadVoro self-ghost loss; the normal self-ghost path calls answer methods directly.

**Fix:** when sendToSelf is enabled, include the local rank in receive peer bookkeeping and flattening under a documented order, or explicitly include the self segment separately while preserving the index contract. Keep disabled behavior unchanged. If self mode is intentionally unsupported, reject the argument at construction rather than returning inconsistent result views.

**Acceptance:**1 rank/self enabled and disabled;2 ranks with self plus remote answers; empty answers; multiple batches where only later batches have self results; compare per-query counts, by-rank segments, flattened payloads and recv-index mappings.


## 8. RICH integration: physical-field transfer errors outside MadVoro

These two errors are physically in RICH, not the MadVoro submodule. They are included because a correct mesh and correct ownership map are insufficient if application fields are transferred to the wrong slots or ranks fail to participate.


### R01 / P1: chain transfer ignores destination indices

**Scope.** RICH integration code outside MadVoro. This can corrupt application buffers that are transferred through a chain even if MadVoro geometry and each ownership mapping are correct.

**Location and source chain**

- `source/mpi/ExchangeChain.hpp:34-35`: explicitly documents the mapping as original index -> (current rank,current index).
- `source/mpi/ExchangeChain.cpp:52-55` correctly assigns a retained original point the current position given by localIndices order; chain composition retains target indices.
- `source/mpi/ExchangeChain.hpp:56-60` groups values by target.first but never sends target.second.
- Same file `64-70`: reconstructs output by self-origin values then other source ranks; this is not the chain's specified final index order.
- `source/newtonian/three_dimensional/simulation/Simulation.hpp:174` registers this helper as `MigrationBuffer::transferChain`.
- `source/newtonian/three_dimensional/simulation/Simulation.cpp:298` applies physical steps' chains to registered buffers.

**Trigger.** Any valid chain in which final local order differs from the accidental self-first/original-index/source-rank concatenation. A self-only permutation is the smallest example. Multi-stage migrations can also reorder origins within a final rank; original indices alone cannot reconstruct that order.

**Actual reproducer.** `evidence/integration_exchange_chain.cpp`, mode `permutation`, one rank:

```text
rank=0 targets=1,0
rank=0 expected=101,100 actual=100,101 pass=0
```

It calls Reset(2), Exchange with retained local indices `{1,0}`, then transfers `{100,101}`. No geometry, remote messaging or floating point is involved. Exit: 3. Raw log `integration_exchange_chain_permutation.log`; clean-transport rerun `integration_exchange_chain_permutation_clean.log`. This proves silent field misassociation, not just a bookkeeping disagreement.

**Fix contract.** Send/receive enough metadata to scatter each value into the chain's declared target index, e.g. `{newIndex,value}` records with explicit serialization, or a communication plan derived from the reverse map. Allocate output to the final local map size; enforce one assignment per final slot; reject duplicate/missing/out-of-range target slots. Preserve variable-size serializable fields and all origin/rank namespaces. Do not sort values by source rank as a substitute for the declared target index. Old input storage must remain valid until all sends complete.

**Regression tests.** Single-rank swaps and random permutations; multiple ranks and nonmonotone receive-peer orders; two/three stage chains; reverse chain; distinct heterogeneous per-ID sentinel fields; registered RICH buffers after moving-mesh and remesh stages. Compare every final slot against `GetReversedTranslationMap()` and ensure preservation of all point-associated state. The ordinary tessellation-based MPI_exchange_data helper already compacts retained data in supplied exact order; do not regress that correct behavior while fixing the chain helper.


### R02 / P1: empty-origin rank skips field transfer

**Scope.** RICH integration code outside MadVoro.

**Location and source chain**

- `source/mpi/ExchangeChain.hpp:23` defines GetNorg() as the size of the original->final map. It is an original local population count, not a global indication that the step did not migrate.
- `source/mpi/ExchangeChain.hpp:49-52` returns locally when that count is zero, before the required all-to-all at 62.
- `source/newtonian/three_dimensional/simulation/Simulation.cpp:133-134` duplicates the same local skip before calling any registered transferChain function. Fixing only the lower helper leaves the upper deadlock path.
- `ExchangeChain.cpp:11-24` correctly supports Reset(0); subsequent Exchange can populate globalTransferOrigins with received points while globalTransfer remains empty.

**Trigger.** An originally empty rank participates in a mesh migration and receives points. Its original map remains empty, but it must receive their physical fields. A nonempty peer enters the collective field transfer while it skips directly out.

**Actual reproducer.** `evidence/integration_exchange_chain.cpp empty`, exactly two ranks. Rank 0 initially owns a single value 1234 and the chain sends its point to rank 1, which initially owns nothing. Chain construction itself completes. Immediately before field transfer:

```text
rank=1 original_count=0 final_count=1 entering_field_transfer
rank=0 original_count=1 final_count=0 entering_field_transfer
rank=1 returned_field_transfer size=0
```

Rank 0 never prints returned_field_transfer. The bounded MPI job exits via timeout code 124 after 8 seconds. Log: `integration_exchange_chain_empty.log`. The logged completed chain and rank 1's early-return marker isolate the mismatch from earlier MPI setup. The logged rank-local stages and source branch, rather than transport warnings, establish the mismatch.

**Fix contract.** Distinguish an absent/uninitialized chain from a valid initialized chain whose local original population happens to be zero. Add explicit chain state or equivalent globally agreed operation state. All ranks enter a valid transfer even when local send count is zero; the reverse map determines how much arrives. Preserve a true no-movement/default-chain operation as a no-op on every rank. **Do not simply remove both early returns and clear every data vector unconditionally:** `RadiationMCStep.cpp:325-328` returns a default empty ExchangeChain to mean no point movement; a default inactive chain must not erase valid physical buffers. Make Reset(0) a valid initialized empty-source chain, and carry the state through copies/Reverse. Audit the helper's hardcoded WORLD communicator separately from operation participation.

**Regression tests.** Initially empty rank receiving first cell; initially nonempty rank becoming empty; several empty ranks; valid globally empty transfer; default inactive chain with nonempty data preserved; multiple registered buffer types; exchange chains across repeated builds; two communicators. Use a watchdog and explicit completion markers rather than treating timeout alone as evidence. Verify final sentinel 1234 reaches rank 1 and rank 0 returns normally with zero entries.

## 9. Repair sequence and review gates

### Work package A: preserve generator identity through partial builds

Own B01, B02, B03 and B04 together across MadVoro and MeshDecomposer3D. First add the full32-to-active8 reference fixture and the full cyclic-mask mapping fixture. Then repair record retention, original-index reconstruction, both tree builders and active/all map direction. Keep these commits in one tested series: shipping retention alone exposes the masked tree error. Finally preserve directed ghost dependencies and supply a correctly indexed, valid donor-field representation. Review non-MPI `BoundaryFace` at V:5179-5194: its nonactive-inside-domain exclusion is compiled only with MPI, so retained inactive local neighbors can otherwise be classified as reflective boundaries. This is a source-backed follow-on hazard; no completed serial partial mesh demonstrating its numerical effect was obtained because B02 fails earlier.

**Completion gate.** Every selected physical cell agrees with a full-build reference from the same retained generators, on 1/2/4 ranks and all periodic masks. Verify active cell IDs and individual volumes/centroids/neighbors, not just the domain total. A rank with no active cells must still donate a needed generator and its required fields. Use nonzero distinct sentinels to detect missing receives. No undefined array access and no dropped global generator ID is acceptable. Include serial builds once B08's compiled-MPI mode contract is repaired.

### Work package B: make copy, release and initialization states coherent

Own B05/B06/B08/B09/B10/B11/B12. Inventory every state member of the explicit copy constructor and every reference held by range finders, trees, managers and caches. Implement nullable manager cloning and independent query ownership. Define copy assignment consistently; avoid the tree class's implicit shallow raw-node copy. Preserve domain faces across ReleaseMemory, reset build state and invalidate caches explicitly. Track converter readiness after SetKernel and require a defined environment before a suppressed first build. Thread explicit serial/distributed execution mode through all relevant geometry and field exchange operations.

**Completion gate.** Construct, configure, copy, build, copy, destroy original, query copy, rebuild original/copy independently, release, and rebuild. Repeat in non-MPI and MPI builds. Test a tetrahedral custom domain, periodic box and kernel replacement. Require expected exceptions for explicitly unsupported state transitions before communication begins; an exception arising from an incidental null dependency is not a satisfactory contract. Post-release getters must follow the chosen empty/unbuilt policy consistently.

### Work package C: repair MPI state and transport contracts

Own B07 and D01 first, followed by D02-D06 for the shared dependency configurations. B07 must use the globally agreed exchange decision for cache preparation and validation. D01 must support arbitrary completed-index permutations without losing request/buffer associations. D02 must propagate a communicator through every constructor, factory and clone; clamping an invalid WORLD rank into subgroup bounds is not a fix. D04 requires both safe packet assembly and error-aware receive validation. D05 requires correct payload byte accounting and an explicit finish contract. D06 must keep all answer views consistent.

**Completion gate.** No pending serialized records or outstanding requests after a successful finish; each record is delivered exactly once with the expected bytes and target index. Delayed peers and rendezvous-size messages must not change the result. Completion-order fixture permutations pass independently of the MPI library's preferred ordering. One subgroup can operate while other world ranks perform unrelated work. Error paths must terminate consistently without continuing with partial records or stranding peers.

### Work package D: preserve periodic and face geometry contracts

Own B13-B16. Repair physical preimage identity first; audit callers' unresolved-result behavior and existing MPI-ghost guards. Correct mixed-periodic shift enumeration using singleton nonperiodic axes. For CleanSameLine, fix restored loop length and distinct alternate-normal selection, then create a deterministic branch fixture before claiming production geometry impact. Normalize the standalone Face3D centroid and document degenerate input behavior without unnecessarily replacing the main mesh's separate center computation.

**Completion gate.** Every periodic image has a verified physical source or an explicit unresolved representation. Each permitted shift appears once; tests check originalPoint translation for big queries as well as centers. CleanSameLine recovery visits the restored ring and never relies on stale cardinality; full geometry stress tests report closure and per-cell volume/centroid errors. The centroid helper obeys translation and linear scaling. Preserve the evidence distinction if no valid-mesh cleanup trigger has yet been found.

### Work package E: repair RICH's chain-based field migration

Own R01/R02 together outside MadVoro. Introduce an explicit initialized/active chain state, distinguishing a default 'no motion' chain from Reset(0) on a valid empty-origin rank. Move each registered field to its declared final rank and local index. All participating ranks enter the transfer even with zero original cells. Update the early exit in Simulation.cpp as well as the helper. Preserve default no-motion semantics used by RadiationMCStep.

**Completion gate.** Single-rank cyclic permutations, initially empty receivers, ranks becoming empty, multistage migrations, reverse maps and multiple registered field types all retain exact per-ID field associations. A default inactive chain leaves nonempty buffers intact. A valid initialized globally empty transfer finishes on every rank. Include a small RICH mesh-update integration test once the helper tests pass; the current audit's proof is at the actual chain/helper API level.

## 10. Regression matrix and numerical acceptance

The small witnesses below are baseline failures; after implementing a fix, update each test to assert the corrected result and exit zero. Keep fixtures minimal enough to isolate the contract, then use larger integration cases only where they can reveal interaction errors. Do not remove assertions or relax expected identities merely to make a test green.

| Test family | Required cases | Acceptance rule |
| --- | --- | --- |
| Full/partial reference | N=32 seeded random; masks full, first8, noncontiguous, cyclic permutation, empty | Preserve all generator IDs; compare each active cell to the same physical cell in the full reference. |
| Local tree/index integrity | All/active counts equal and unequal; equal count with different order | Brute-force nearest results in the correct index space; exact map/coordinate association. |
| Distributed ghost fields | 1/2/4 ranks; zero-active donor; receive-only peer; changing mask | Every required ghost receives a valid field from its actual owner; no reliance on reciprocal nonempty data. |
| Lifecycle | Fresh/built copy; assignment; release; source destruction; kernel change | Defined state, independent ownership and valid queries; unchanged domain unless explicitly changed. |
| Periodic geometry | Eight axis masks; face/edge/corner images; local/remote preimages | Correct source identity and translation; one query per intersecting shift; valid mesh closure. |
| MPI scheduling | Synthetic completion permutations; delayed ranks; eager/rendezvous traffic | Exact request/buffer association and exactly-once complete record delivery. |
| Communicators | SELF, split groups, reordered ranks, inactive world peers | All collectives and owner indices use the configured communicator. |
| Field migration | Cyclic permutation; initially empty receiver; default inactive chain | Exact per-ID field placement, all ranks complete, inactive chain preserves existing buffers. |
| Invalid/precondition cases | Out-of-range/duplicate mask, bad lengths, unsupported fresh suppression | Defined collectively safe rejection before incompatible communication or mesh mutation. |

**Numerical oracle.** For the unit-box random fixtures, start with absolute plus relative tolerances: volume `abs(v-vref) <= 1e-12 + 1e-10*abs(vref)`; each centroid component `abs(c-cref) <= 1e-11` in box units. These are proposed regression thresholds, not proven universal error bounds. Measure the unchanged full/full control under different insertion orders first; retain its output in the test. If the control fails a threshold, investigate scale, degeneracy and rounding before choosing a justified tolerance. For general domains scale length tolerances by a characteristic domain extent and volume tolerances by its cube. Never use an absolute tolerance of one box unit or loosen tolerances enough to hide B01's order-one volume changes.

Match cells by stable fixture ID or exact input-generator coordinates when no IDs are carried by the API. Compare physical neighbors canonically. Degenerate coplanar/cospherical inputs can admit equivalent triangulations, so avoid insisting on identical tetrahedron ordering; verify the resulting geometric cells, face planes and conservation invariants. Use integer/sentinel field equality for migration and indexing tests. These are exact association problems and do not need a floating-point tolerance.

**Diagnostic builds.** Run the small valid-input cases with libstdc++ assertions and, after repairs, ASan/UBSan builds where compatible with the selected MPI environment. Also run an optimized build: disappearing debug assertions cannot be the only error policy. For D04 use both the ordinary MPI error handler and MPI_ERRORS_RETURN, checking a controlled rejection or exact delivery. Use a watchdog for every MPI test, record stage/rank markers, and collect per-rank failure state. Sanitizer or MPI launcher noise must not be mistaken for the underlying mechanism without the matching code path.

**Review questions for the implementing model.** Does every changed index retain a named index space? Is the test a genuine permutation with a cycle longer than two? Does partial geometry still include inactive support? Can an empty rank supply or receive data? Are globally reduced decisions used by every rank's state preparation? Does a copied finder own a live tree after source destruction? Can a packet record fit alone but overflow in combination? Does a periodic resolution prove identity instead of closeness? Can a default inactive chain be distinguished from an active chain with zero original cells? Each answer needs a source change and a regression witness, not a claim that ordinary smoke runs passed.

## 11. Reproduction guide and evidence inventory

Run from `/home/maorm/RICH` using the checked-out headers rather than an installed MadVoro library. `reproduce.py` in this report directory provides explicit build/run recipes with per-case labels, expected baseline behavior and bounded execution. It writes fresh results under `evidence/rerun_<timestamp>/` and executables under that rerun's build directory. It does not patch production source. List the cases before running if a different MPI/compiler environment needs adjustment:

```bash
cd /home/maorm/RICH
python3 docs/madvoro_bugs/reproduce.py --list
python3 docs/madvoro_bugs/reproduce.py --build --run
```

The script's defaults reproduce the audited host: `mpicxx`, `c++`, C++17, the installed Boost include directory, `OMP_NUM_THREADS=1`, and `UCX_TLS=self,sm,tcp`. Override `BOOST_INCLUDE` and MPI/compiler locations in the environment when moving the bundle. A recorded crash is intentional baseline evidence, not a successful regression. The script records raw exit codes and logs; its own completion is not proof that the library passed. After fixes, the harnesses that intentionally detect defects must be converted to positive assertions rather than preserving old failure expectations.

| Evidence group | Contents and interpretation |
| --- | --- |
| `geometry_partial_volume_*` | B01 full/partial differential, eight mismatched volumes and point-retention counts. |
| `root_api_repro.cpp`, `root_serial_*`, `root_v2_*` | B02/B03/B08/B09/B11/B13; identity controls; v2 explicitly initializes the manager before subset tests. |
| `geometry_repro.cpp`, `geometry_*` logs | B05/B06/B10; successful ordinary rectangular release/rebuild control. |
| `geometry_cleanup_*`, `geometry_face_centroid.*` | B15 negative bounded search and extracted helper; B16 actual public centroid call. |
| `mpi_extract_repros.py`, `mpi_extracted_helpers.*` | B04/B14 verbatim extracted bodies, minimal fixture types, provenance hashes. |
| `mpi_dependency_repro.cpp`, `mpi_*` logs | D01 injected completion order; D04/D05/D06 real MPI component cases. |
| `integration_mixed_suppression*` | B07 two-rank assertion and all-suppression successful control. |
| `integration_load_balancer_lifecycle*` | B12 kernel setter/update; D03 single-rank 1D intersection failure. |
| `integration_points_manager_comm*` | D02 actual MPI_COMM_SELF manager inside a two-rank world. |
| `integration_exchange_chain*` | R01 exact self-permutation mismatch; R02 completed chain then mismatched collective participation. |
| `audit_notes/*.md` | Separate source audits, independent reviews, command details and rejected hypotheses. |
| Revision/hash/configuration files | Exact source and compiler context; production changes were not made by this audit. |

The report's single external contract reference is [MPI Forum, MPI 4.1 standard, Multiple Completions](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf), the MPI_Waitsome/MPI_Testsome discussion on printed pages 88-89. D01's portability conclusion is an inference from the absence of an ascending-index guarantee plus the failing production bookkeeping fixture. The standard does not explicitly say that this installed implementation must exhibit the injected order.

## 12. Boundaries of the conclusions

**These are located defects and implementation plans, not a completeness certificate.** Review and tiny fixtures cannot establish that no other Voronoi/Delaunay, predicate, field or MPI errors exist. They also do not show how frequently each affected path occurs in production RICH runs. No generic allegation of wrong scientific results is warranted for workflows that do not meet the stated trigger.

Several investigated concerns were deliberately not promoted to additional findings. A custom-face MPI constructor initially gives its manager zero bounds, but a one-rank custom-box build passed; multi-rank routing consequences remain untested. SetBox replaces manager configuration, but the intended setting-retention contract needs clarification. Hilbert equality-collision remapping may affect ordering, yet lost insertions or invalid geometry were not demonstrated. No wrong lattice-predicate result was reproduced. Invalid masks, nonfinite weights, arbitrary unsupported call orders and huge MPI counts need defined policies, but hypothetical invalid input is not counted as another valid-workload failure.

Likewise, `CountOutcoming` omits packed pending buffers, but the present protocol also accounts for logical outstanding query replies; that omission alone did not establish premature completion. Shared MPI tags constrain future concurrent query instances, but current batches run serially and routine cross-talk was not reproduced. Old-ghost reuse is currently disabled, so future caching hazards are prerequisites for an optimization, not proof of a present active-cache error. Mixed-periodic duplicate queries demonstrably cause excess work; numerical corruption from that duplication was not observed.

The proposed repairs are precise contracts with targeted implementation directions. Except for the test-only renamed cleanup variant used in a negative search, no fix was implemented or experimentally validated here. Treat the report as a handoff: implement a coherent change, preserve the evidence, and require the relevant positive regression and integration gates before declaring a finding closed.
