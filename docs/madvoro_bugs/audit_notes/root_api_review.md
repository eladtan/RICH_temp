# Independent review of root API findings

Reviewed on 2026-09-08 against current production source and `evidence/root_api_repro.cpp`, `root_v2_results.json`, and the cited raw logs. No production code changed. Additional one-rank MPI volume differential reproduction was compiled directly from current headers, O0, using the same seed 1729 as root.

## New independent reproduction: MPI partial build deletes supporting generators and changes all eight requested volumes

Artifacts: `docs/madvoro_bugs/evidence/geometry_partial_volume_repro.cpp` and `geometry_partial_volume.log`. One MPI process; full 32 random points in [0,1]^3; snapshot full volumes and coordinates; request first eight active indices while passing all 32 retained input points to `BuildPartiallyParallel(..., true, true)`. Match resulting cells to original cells by exact generating coordinates rather than assuming identical output order.

Actual output:

```text
PARTIAL_VOLUME input_all=32 active_requested=8 output_all=8 returned=8 active_built=8
CELL id=0 full_volume=0.026057755518833325 partial_volume=0.11710940435107145
CELL id=1 full_volume=0.020864708713191807 partial_volume=0.12720350729574328
CELL id=2 full_volume=0.02455102431937714 partial_volume=0.092709427151980009
CELL id=3 full_volume=0.031107118574504246 partial_volume=0.1552114594465453
CELL id=4 full_volume=0.024596174684458016 partial_volume=0.074871145301961206
CELL id=5 full_volume=0.031512601036428646 partial_volume=0.1206876120070022
CELL id=6 full_volume=0.042144561980046645 partial_volume=0.15778704578159264
CELL id=7 full_volume=0.046028058580248749 partial_volume=0.15442039866410384
SUMMARY reference_active_volume=0.24686200340708858 partial_active_volume=0.99999999999999989 mismatched_cells=8
EXIT_STATUS=5
```

The function completes successfully internally; exit5 is the harness detecting the wrong retention/geometry result. The eight active cells now tessellate essentially the entire box, showing this is a full tessellation of eight generators rather than the requested eight cells of the 32-generator mesh. A global total-volume-only test would misleadingly pass.

**Verified cause:** `source/3D/tessellation/MeshDecomposer3D/points_manager/PointsManager.hpp:289-297` serializes only indicesToWorkWith and unconditionally marks every emitted entry participating=true. No inactive entry reaches dataExchange. Lines330-334 reconstruct the complete newPoints array from this filtered stream. `Voronoi3D.hpp:2032` then replaces allMyPoints with newPoints; :2053-2058 consequently sees every retained point active. `HilbertPointsManager.hpp:153-160` uses the same filter even in noExchange mode. Suppressed communication does not preserve inactive local generators.

**Contract assessment:** the partial-build API lacks an extensive formal public contract, so the report should state its intended semantics explicitly. However, treating indicesToBuild as an implicit deletion mask conflicts with serial BuildPartially assigning allMyPoints=allPoints at V:4098, the participatingIndices field, the separate all-point/active trees, and the explicit active-to-all mapping. This is strong evidence of a defect rather than an arbitrary benchmark expectation. If maintainers intended a subset-only mesh operation, it should be a separately named operation; it must not silently replace retained allPoints in partial construction.

**Recommended severity:** P1 for silent wrong active-cell geometry and loss of retained inactive points. Actual tested scope is MPI1, nonperiodic, no exchange/rebalance, first8 active after full build. The root multi-rank broader impact remains an inference from the same filtering logic.

**Fix requirements:** iterate allPoints, set each entry.participating from an active mask, preserve inactive coordinates/weights/payloads and their ownership/mappings, then construct activePoints by the participation flags. Reconstruct self/sent index maps using actual originalIndex instead of indexing indicesToWorkWith by a now-full packed-array offset. In noExchange mode retain every original local record and mark active separately. Fix the tree loop described below in the same dependency series, because preserving inactive points exposes that overrun. Do not merely retain an extra unused allPoints copy while still searching only active generators.

## Reviewed root findings and qualifications

### 1. Serial partial all-point tree out-of-bounds: verified active defect

V:4098 retains the full allPoints array; :4100 makes activePoints the subset; :4176-4179 loops full count but indexes subset. The root `subset` case using indices0..7 isolates this independently from inverse-map failure because those map entries are identity. Non-MPI bounds-check/sanitizer failure is a valid direct reproduction. The MPI UpdatePointsTree duplicate at V:2137-2140 has the same bug but is currently **masked** in the ordinary MPI subset path by deletion of inactive points. Do not claim root MPI subset's successful execution proves it safe. The volume reproduction above explains why it survives.

Fix all-point tree input to allMyPoints[k] with all-local k; active tree input to activePoints[j] with build-local j. Alias only for identity mapping and matching coordinates, not merely equal cardinalities. Cover both duplicated implementations or centralize the helper.

### 2. Serial active-to-all map reversed: verified independently of the overrun

V:4106 writes map[allIndex]=activeIndex. Consumers V:3207-3212 and :5581-5584 require map[activeIndex]=allIndex; parallel constructor does that correctly at :2057. Root full-rotation mask has 32 active/32 all, so it avoids the size overrun while showing expected map[0]=1 but observed31. It is a good isolating test. A general sparse mask can additionally write indexes>=Norg into query arrays; fix map direction before interpreting those downstream failures as separate root causes. Root output demonstrates wrong mapping, not necessarily wrong geometry for a full permutation with identical point set.

### 3. MPI retained inactive points filtering: independently reproduced geometry error

Use the new volume evidence as above. Merge this with the tree/map finding only if the report still gives each independent fix and states the masking relationship. Fixing just PointsManager retention is not sufficient for safe partial builds.

### 4. Remote periodic preimage resolves to unrelated owned point: verified API identity error

V:3047-3054 removes translation, wraps the coordinate, then asks **myPointsTree**, which contains owned active points, for the nearest point. A remote physical generator is absent from this tree. The nearest owned point is generally a different generator, and :3056-3058 treats any in-range result as the true preimage without an equality/identity check. Root two-rank logs show large coordinate mismatches (e.g. physical (0.446993,0.835512,0.0893802), returned owned (0.650812,0.618903,0.0565723)); these are not periodic roundoff.

Qualification: the primitive hydrodynamics resolver in `source/newtonian/three_dimensional/LinearGauss3D.cpp:49-51` explicitly returns known MPI ghosts unchanged before invoking this resolver. Thus the root API reproduction alone does not prove every hydro reconstruction reads the wrong state. `source/monte/utils/GhostMap.hpp:47-50` trusts any owned result; core MockMesh V:2545 and some centroid fallback paths also call it. Report a demonstrated identity contract violation with plausible affected callers, not a measured full simulation corruption.

Correct contract: return a genuine locally owned physical index only when identity matches; otherwise preserve/return the correct remote ghost identity or an explicitly unresolved sentinel according to callers' contract. Prefer stored owner/global-ID/image translation metadata. A nearest-point match requires verified coordinate/ID equality; nearest alone is insufficient. An arbitrary loose spatial tolerance is dangerous for closely spaced generators. Add remote/local images, multiple images, source migration, rank-empty cases and field-ID regression. root wraps image coordinate directly; production subtracts translation then wraps, which is equivalent for these one-box image translations up to tiny roundoff, far below the observed mismatch.

### 5. Serial Build on an MPI-configured object: verified first-call failure, state dependent

Fresh constructor initializes a nonnull pointsManager but no environment. Serial Build skips PrepareToBuildParallel and calls BringGhostPointsToBuild(MPI_COMM_SELF). V:3582 incorrectly defines serialMode as pointsManager==nullptr, so it treats the fresh object as parallel and throws at :3665 because envAgent is null. Root catches that exact exception, and geometry audit independently observed the same failure before switching its copy harness to BuildParallel.

Do not claim every serial Build after a prior BuildParallel must throw: an initialized environment changes the path. The contract defect is that the public serial API's behavior unexpectedly depends on MPI configuration and previous parallel initialization. Fix through explicit build/protocol mode and communicator ownership, not by declaring every size1 communicator serial or simply removing the null guard; the latter permits downstream dereferences. Tests should include fresh MPI object serial Build, repeated serial Build, serial Build after parallel Build, and explicit communicator-self operations on multiple world ranks.

### 6. Empty continuity query: verified public-API out-of-bounds

V:5204 makes reached.size()==Norg; :5205 unconditionally writes reached[0] and :5207 pushes cell0. Root log shows owned0 before the vector<bool> bounds assertion. This is a clear valid-empty-state bug; empty ranks can occur after supported distributed builds. Return a documented vacuous true (or an explicit empty status) before indexing for Norg==0. A false return is also defensible only if documented, but an out-of-bounds access is not. Test empty, one cell, disconnected owned zones, and empty ranks inside collective build sequences.

## Reproduction command

```bash
mpicxx -std=c++17 -O0 -g1 -fno-omit-frame-pointer -fopenmp \
 -DMADVORO_WITH_MPI -DSPATIAL_DS_WITH_MPI -DRICH_MPI -D__WITH_MPI \
 -Isource/3D/tessellation/voronoi -Isource/3D/tessellation -Isource/utils \
 -I/software/x86_64/5.14.0/boost/1.78.0/include \
 docs/madvoro_bugs/evidence/geometry_partial_volume_repro.cpp \
 source/3D/tessellation/voronoi/exception/MadVoroException.cpp \
 source/3D/tessellation/voronoi/exception/InvalidArgumentException.cpp \
 source/3D/tessellation/voronoi/exception/SizeException.cpp \
 source/utils/mpi_utils/AmountManager.cpp -o /tmp/madvoro_partial_volume_repro
UCX_TLS=self,sm,tcp OMP_NUM_THREADS=1 /tmp/madvoro_partial_volume_repro
```

Direct execution initializes exactly one MPI rank; the harness rejects P!=1. It runs in well below one second here. The output is diagnostic evidence, not a performance benchmark or passing regression. Production code remains unchanged.
