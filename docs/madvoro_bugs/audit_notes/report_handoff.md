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
