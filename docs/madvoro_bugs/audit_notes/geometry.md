# MadVoro geometry and lifecycle bug audit

Audit date: 2026-09-08. Read-only audit of production source in `/home/maorm/RICH`. No production source was edited. Small diagnostic sources/logs are in `docs/madvoro_bugs/evidence/geometry_*`; executables were built under `/tmp`. Findings distinguish directly reproduced public-API failures from source-proven control-flow defects without a complete mesh reproducer. No P0 defect was established by this work.

## Finding G-B01 — P1 — Copying a built mesh leaves point-location trees null

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

## Finding G-B02 — P2 — Copying a never-built MPI mesh dereferences a null load balancer

**Status:** directly reproduced segmentation fault before any tessellation work. It is a distinct precondition failure from G-B01; fixing missing trees alone does not repair this case.

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

## Finding G-B03 — P2 — ReleaseMemory changes a custom physical domain into its enclosing rectangular box

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

## Finding G-B04 — P2 — CleanSameLine retries with stale polygon length and wrong second-normal selection

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

## Finding G-B05 — P3 — Face3D calc_centroid returns an area moment rather than a centroid

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

## Investigated issues not promoted to demonstrated mesh bugs

1. **Custom-face MPI constructor retains zero initial manager bounds:** V:1375 delegates to the default point-box constructor, creates a manager using zero ll/ur at :1420, then only replaces Voronoi ll_/ur_. Source shows inconsistent manager bounds, but a one-rank custom rectangular-face BuildParallel still produced volume 1 (`geometry_custom_parallel.log`). This needs multi-rank owner/environment queries before elevating it from a configuration-consistency risk to a concrete mesh defect. The root/integration team was notified.
2. **Hilbert equality collision remapping:** local helper H:403-420 remaps via original indices rather than collision positions. This can change intended spatial ordering, but this audit did not show missing/duplicated insertion IDs or invalid Delaunay geometry; do not equate poor ordering with a mesh correctness failure.
3. **Lattice scope caching/performance proposals:** no caching patch was implemented. The current scope exclusions and dyadic bounds were reviewed, but no wrong predicate result was reproduced. Bounds-only certification must not be propagated to derived points if future threading/caching is implemented; this is a future-change constraint, not an established current bug.
4. **Delaunay public Build/Clean exception lifecycle:** externally supplied invalid order vectors or rebuilding without the caller's expected Clean can leave questionable state, but supported API preconditions and a production trigger were not established. No reportable finding from this work.
5. **Periodic query duplicate expansion:** sent to MPI agent to own. Nonperiodic axis ranges contain three zero entries, producing repeated image queries; final wrong tessellation was not demonstrated locally.
6. **Empty continuity query and non-MPI partial BoundaryFace handling:** sent to root who owns active/empty API testing. Avoid duplicate IDs in the consolidated report.

## Reproduction commands and limitations

Run from `/home/maorm/RICH` with the installed compiler/MPI environment. The benchmark remains one process; UCX_TLS avoids requesting unavailable IB transport on this login node. This is not a cluster performance run.

```bash
mpicxx -std=c++17 -O1 -g1 -fno-omit-frame-pointer -fopenmp \
 -DMADVORO_WITH_MPI -DSPATIAL_DS_WITH_MPI -DRICH_MPI -D__WITH_MPI \
 -Isource/3D/tessellation/voronoi -Isource/3D/tessellation -Isource/utils \
 -I/software/x86_64/5.14.0/boost/1.78.0/include \
 docs/madvoro_bugs/evidence/geometry_repro.cpp \
 source/3D/tessellation/voronoi/exception/MadVoroException.cpp \
 source/3D/tessellation/voronoi/exception/InvalidArgumentException.cpp \
 source/3D/tessellation/voronoi/exception/SizeException.cpp \
 source/utils/mpi_utils/AmountManager.cpp -o /tmp/madvoro_geometry_repro
UCX_TLS=self,sm,tcp OMP_NUM_THREADS=1 /tmp/madvoro_geometry_repro copy
UCX_TLS=self,sm,tcp OMP_NUM_THREADS=1 /tmp/madvoro_geometry_repro unbuilt_copy
UCX_TLS=self,sm,tcp OMP_NUM_THREADS=1 /tmp/madvoro_geometry_repro custom_release
UCX_TLS=self,sm,tcp OMP_NUM_THREADS=1 /tmp/madvoro_geometry_repro release
```

Expected failures: copy and unbuilt_copy exit by SIGSEGV; custom_release exits normally but flips the domain predicate incorrectly; ordinary release/rebuild returns volume 1. These logs document baseline failures, not passing fixes. The core copy log was generated before extra modes were added to the same harness; source line numbers/backtrace addresses can differ after recompilation, but its minimal copy sequence is unchanged.

```bash
g++ -std=c++17 -O2 -Isource/3D/tessellation/voronoi -Isource/utils \
 -I/software/x86_64/5.14.0/boost/1.78.0/include \
 docs/madvoro_bugs/evidence/geometry_face_centroid.cpp -o /tmp/madvoro_face_centroid
/tmp/madvoro_face_centroid
```

The centroid helper test needs no MPI or external exception translation units. The cleanup investigation separately links MadVoroException.cpp and records a negative search result. No production bug fix is included in these artifacts.
