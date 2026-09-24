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
