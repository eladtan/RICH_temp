---
name: rich-self-gravity
description: >-
  Documents RICH's self-gravity subsystem: Barnes-Hut octree, serial and MPI
  parallel gravity, multipole expansions, data structures, and hydro
  integration. Use when writing, debugging, profiling, or extending self-gravity
  code, or when the user mentions gravity tree, octree, opening angle, theta,
  quadrupole, monopole, GravityTree, DistributedGravityCalculator,
  ConservativeForce3D, or self-gravity acceleration.
---

# RICH Self-Gravity

## 1. File Map

### Core tree (`source/3D/gravity/`)

| File | Role |
|------|------|
| `GravityTree.hpp` | Barnes-Hut octree: build, mass/CM/quadrupole aggregation, tree walk, force |
| `MassedValue.hpp` | `MassedValue<T>`: node payload (value, CM, mass, Q[6]), MPI `Serializable` |
| `GravityTypes.h` | `gravity_result_t` (double), `MassedPoint<T>` (leaf input) |
| `SerialGravityCalculator.hpp` | Thin wrapper: build tree from tessellation, `getAcceleration`. **Exists but unused** -- `GravityAcc3D` inlines tree build for serial builds instead. |
| `DistributedGravityCalculator.hpp` | **Production MPI path**: local tree + exchange of node data + local walk |
| `DistributedGravityTree.hpp` | `GravityNodeData` struct (MPI payload); also contains `DistributedGravityTree` class (not used in production) |

### Hydro wrappers (`source/newtonian/three_dimensional/`)

| File | Role |
|------|------|
| `ConservativeForce3D.hpp/.cpp` | `Acceleration3D` (abstract base), `ConservativeForce3D` (wraps accel into `SourceTerm3D`) |
| `GravityAcc3D.hpp` | `GravityAcceleration3D : Acceleration3D` -- the main entry point; dispatches serial vs MPI |
| `MonopoleSelfGravity3D.hpp/.cpp` | Radial-bin monopole gravity with `MPI_Allreduce` |
| `QuadrupoleGravity3D.hpp/.cpp` | Spherical-harmonic quadrupole expansion with adaptive radial bins (**different** from the octree `Q[6]` quadrupole in `MassedValue`) |

### Infrastructure

| File | Role |
|------|------|
| `source/ds/OctTree/OctTree.hpp` | Generic `OctTree<T>`: 8-child tree, insert, find, range queries |
| `source/mpi/serialize/mpi_commands.hpp` | `MPI_Exchange_all_to_all`, `MPI_All_cast_by_ranks` |

---

## 2. Architecture

### Class hierarchy

```
Acceleration3D                    (abstract, ConservativeForce3D.hpp:12)
 +-- GravityAcceleration3D        (GravityAcc3D.hpp) -- tree-based
 +-- MonopoleSelfGravity3D        (MonopoleSelfGravity3D.hpp) -- radial bins
 +-- QuadrupoleGravity3D          (QuadrupoleGravity3D.hpp) -- spherical harmonics
 +-- ConstantAcceleration3D       (ConservativeForce3D.hpp:31) -- uniform g

SourceTerm3D                      (abstract, SourceTerm3D.hpp)
 +-- ConservativeForce3D          (wraps any Acceleration3D)

SeveralSources3D                  (aggregates multiple SourceTerm3D)
```

### Wiring in a simulation

Gravity is wired through `HDSim3D`'s source-term mechanism regardless of
whether the legacy `HDSim3D` loop or the newer `Simulation` class is used.
`HydroStep` (a `PhysicsStep` subclass used by `Simulation`) wraps `HDSim3D`
internally:

```cpp
GravityAcceleration3D acc(theta, use_quadrupole, G);
ConservativeForce3D gravity(acc, mass_flux);
std::vector<std::shared_ptr<SourceTerm3D>> forces = {std::make_shared<ConservativeForce3D>(acc)};
SeveralSources3D combined(forces);
// pass `combined` as the source term to HDSim3D
// HDSim3D is then wrapped in HydroStep for use with Simulation
```

### Where gravity runs in the timestep

In `HDSim3D::timeAdvance` (`hdsim_3d.cpp`), gravity executes as the `source_()` call between flux computation (`fc_`) and extensive update (`eu_`):

1. Flux computation (`fc_`)
2. **Source terms** (`source_` -- gravity runs here)
3. Extensive update (`eu_`)
4. Mesh motion / tessellation update
5. Cell update (`cu_`)

When using the `Simulation` class, `Simulation::step()` calls
`HydroStep::step(dt)`, which delegates to `HDSim3D::timeAdvance2` (or
similar variant), so gravity runs in the same position within the hydro
substep.

`ConservativeForce3D::operator()` calls `acc_(tess, cells, fluxes, t, acc)` to get accelerations, then updates momentum and energy on `extensives`. It also computes an inverse-timestep suggestion via `sqrt(|acc| / cell_width)` with an `MPI_Allreduce(MPI_MAXLOC)`.

---

## 3. Serial Algorithm

### Tree build (`GravityAcceleration3D::operator()` / `GravityTree::build`)

**Note:** The serial path in `GravityAcc3D.hpp` builds `GravityTree` directly
rather than using `SerialGravityCalculator`.

1. Collect cell CMs from `tess.GetAllCM()`, masses from `density * volume`.
2. Construct `GravityTree<Vector3D>(ll, ur, theta, quadrupole)`.
3. For each cell, `OctTree::insert(MassedValue)` -- splits leaf nodes as needed.
4. `calculateMasses()` -- post-order traversal via `calculateMassHelper`:
   - Leaf: mass/CM already set from input.
   - Internal: sum child masses, compute weighted CM, accumulate quadrupole tensor Q from children using the parallel-axis theorem.

### Tree walk (`GravityTree::gravity`)

Stack-based DFS from root (or a specified node via `directions` parameter).
For each node:
- If **internal** AND (contains evaluation point OR `ShouldOpenBox`): push children.
- Otherwise: accumulate force via `CalculateLeafGravityContribution`.

When the evaluation point is inside a node, that node is always opened and its
child containing the point gets `containsPoint=true`, avoiding the
`ShouldOpenBox` distance calculation for ancestors.

**`ShouldOpenBox`** (Barnes-Hut criterion): open if `width^2 >= theta^2 * dist(CM, point)^2`. Has optional VCL vectorized distance path (`USE_VCL_VECTORIZATION`).

**`CalculateLeafGravityContribution`**: Newtonian monopole `-m * r_hat / r^2` plus optional quadrupole correction using the traceless Q tensor.

Self-interaction guard: if `|r - CM| < EPSILON` (1e-12), returns zero force.

### Key parameters

| Parameter | Constructor arg | Effect |
|-----------|----------------|--------|
| `theta` | `GravityAcceleration3D(theta, ...)` | Opening angle (0.5--1.0 typical). Smaller = more accurate, slower. |
| `quadrupole` | second arg (bool) | If true, accumulate and use Q[6] multipole corrections. |
| `G` | third arg (double, default 1) | Gravitational constant scaling applied after tree walk. |

---

## 4. MPI Algorithm (`DistributedGravityCalculator`)

**Important**: The production MPI path is `DistributedGravityCalculator`, not `DistributedGravityTree`. `GravityAcceleration3D` uses it under `#ifdef RICH_MPI`.

### Construction

```cpp
DistributedGravityCalculator(const Tessellation3D &tess_,
    const std::vector<gravity_result_t> &masses_,
    double theta_, bool quadrupole_ = false,
    const MPI_Comm &comm_ = MPI_COMM_WORLD);
```

1. Build **local** `GravityTree` from owned cells only.
2. Find `realRootOfGravityTree` -- walk down single-child octants to reach the first node with multiple children (skips empty space above local matter).
3. Initialize `relevantRanksByDepths[0]` = all other ranks. This is a
   `vector<boost::container::flat_set<int>>` indexed by tree depth.
4. `calculateBoundingBoxesOfRanks` returns `vector<vector<GravityNodeData>>`:
   broadcast each rank's root `GravityNodeData` (bounding box, CM, mass, Q)
   via `MPI_All_cast_by_ranks`.

### `getAcceleration` (per evaluation)

1. **`getSendList()`**: recursive `getSendListHelper` from `realRootOfGravityTree`.
   - For each local tree node and each relevant remote rank, decide whether to **open** (recurse to children) or **send** (pack node's `MassedValue` for that rank).
   - Decision criteria:
     - If remote bounding box **contains** local node: check if sphere of radius `width/theta` around local CM intersects with the remote rank (via `getIntersectingRanks`).
     - Otherwise: apply `ShouldOpenBox` against the remote's closest point.
2. **`MPI_Exchange_all_to_all(sendList)`**: all-to-all exchange of `MassedValue` vectors.
3. **`addExternalValues`**: insert received remote summaries as extra leaves in the local octree.
4. **`calculateMasses()`**: recompute mass/CM/Q for the augmented tree.
5. **Local tree walk**: for each evaluation point, `gravityTree->gravity(point)`.

### Communication pattern

- Construction: one `MPI_All_cast_by_ranks` (all-gather of `GravityNodeData`).
- Per evaluation: one `MPI_Exchange_all_to_all` (all-to-all of `MassedValue` vectors via `MPI_Alltoall` counts + `MPI_Alltoallv` payload).
- No ghost-cell exchange in the hydro sense -- remote geometry enters as extra tree leaves.

---

## 5. Data Structures

### `MassedValue<T>` (`MassedValue.hpp`)

```cpp
template<typename T>
struct MassedValue : Serializable {  // Serializable only under RICH_MPI
    T value;           // insertion position
    T CM;              // center of mass
    gravity_result_t mass;
    std::array<coord_type, 6> Q;  // symmetric traceless quadrupole: Qxx, Qxy, Qxz, Qyy, Qyz, Qzz
    // coord_type = typename T::coord_type
};
```
`Q[5]` (Qzz) is set to `-Q[0] - Q[3]` (traceless constraint) in
`calculateMassHelper`.

MPI serialization: `dump`/`load` serialize `value`, `CM`, `mass`, `Q` via
`Serializer`. Total: 104 bytes per node (3+3+1+6 doubles).

### `MassedPoint<T>` (`GravityTypes.h`)

Input to `GravityTree::build`: just `T point` + `gravity_result_t mass`.

### `GravityNodeData` (`DistributedGravityTree.hpp`)

MPI exchange payload for rank bounding-box broadcast:
```cpp
struct GravityNodeData : Serializable {
    BoundingBox<Vector3D> boundingBox;
    gravity_result_t mass;
    Vector3D CM;
    std::array<double, 6> Q;
};
```

### `OctTreeNode` (`OctTree.hpp`)

```cpp
class OctTreeNode {
    bool isLeaf;
    T value;                              // MassedValue<Vector3D> for gravity
    BoundingBox<Raw_type> boundingBox;
    std::array<OctTreeNode*, CHILDREN> children;  // CHILDREN = 8
    OctTreeNode *parent;
    int height, depth;
    octnode_id_t id;
};
```

Child index via `getChildNumberContaining`: a 3-bit direction number.
For each axis `i` (0=x, 1=y, 2=z), bit `(DIM-1-i)` is set to 1 if
`point[i] > center[i]`. So bit 2 = x comparison, bit 1 = y, bit 0 = z.
Example: point in (+x, -y, +z) octant = binary 101 = child 5.

Other constants: `DIM = 3`, `MAX_DEPTH = 64`, `PATH_END_DIRECTION = -1`.

### `GravityTree<T>` additional API

| Method | Description |
|--------|-------------|
| `build(points)` | Insert `MassedPoint`s, then `calculateMasses()` |
| `gravity(point, directions)` | Stack-based tree walk returning force vector |
| `addExternalValues(values)` | Insert remote `MassedValue`s as leaves |
| `findMatchingMassedValue(bbox)` | Find node matching a bounding box |
| `getOpenNodesData(point)` | Return `(node_id, MassedValue)` pairs for opened nodes |
| `calculateMasses()` | Post-order mass/CM/Q aggregation |

---

## 6. Debugging Guide

### Common issues

- **Self-interaction**: `CalculateLeafGravityContribution` returns zero if `|r| < EPSILON`. If you change cell positions or add softening, verify this guard.
- **Opening angle too aggressive**: theta > 1 can skip nearby mass. Compare with direct summation for a few cells.
- **Mass conservation in MPI**: after `addExternalValues` + `calculateMasses`, the root mass should equal the global total. Print `gravityTree->getOctTree()->getRoot()->value.mass` and compare with `MPI_Allreduce`-d sum of cell masses.
- **Send-list symmetry**: if rank A sends N blobs to rank B, rank B's tree walk should use those blobs. Mismatches indicate a bug in `getSendListHelper`'s containment/opening logic.
- **Two quadrupole concepts**: the octree `Q[6]` in `MassedValue` / `CalculateLeafGravityContribution` is distinct from the spherical-harmonic `Q20/Q21/Q22` in `QuadrupoleGravity3D`. They use completely different physics and code paths.

### Verification strategies

1. **Serial vs MPI**: build both paths for the same problem (small N), compare accelerations element-wise. Compile without `-DRICH_MPI` for the serial path.
2. **Conservation check**: sum `density * volume * acc` over all cells -- should be near zero (Newton's third law).
3. **Tree printing**: compile with `-DDEBUG_MODE` to enable `GravityTree::print()` / `OctTree::print()`.
4. **MPI debugging tools**: use `SmartCollectives` (`source/mpi/SmartCollectives.hpp`) for backtrace-matching collective assertions. Use `MPI_Timed_barrier` to detect hangs.

### Useful debug insertions

```cpp
// Print total tree mass after MPI exchange
double localTreeMass = gravityTree->getOctTree()->getRoot()->value.mass;
std::cout << "Rank " << rank << " tree mass after exchange: " << localTreeMass << std::endl;

// Verify send-list sizes
auto sendList = getSendList();
for(int r = 0; r < size; r++)
    std::cout << "Rank " << rank << " -> " << r << ": " << sendList[r].size() << " nodes" << std::endl;
```

---

## 7. Profiling Guide

### Build options

- **gprof**: set `cmake -DPROF=ON` (adds `-pg` flag, see `source/CMakeLists.txt:31-38`).
- **VTune**: set `-DVTUNE_INCLUDE=/path/to/vtune/include` (see `source/CMakeLists.txt:191-195`).
- **Manual timing**: insert `MPI_Wtime()` calls around gravity phases.

### Performance hotspots (in order of typical cost)

1. **`gravity()` tree walk** -- O(N log N) per rank. The inner loop is `CalculateLeafGravityContribution` + `ShouldOpenBox`. Profile this function first.
2. **`getSendListHelper`** recursion -- determines MPI message sizes. Cost scales with number of ranks and tree depth.
3. **`MPI_Exchange_all_to_all`** -- serialization overhead + network. Message sizes depend on theta and geometry. Print send-list sizes to assess.
4. **`calculateMasses`** after `addExternalValues` -- full post-order traversal of augmented tree.
5. **`OctTree::insert`** during `build` and `addExternalValues` -- repeated node splitting.

### Instrumentation template

```cpp
double t0 = MPI_Wtime();
DistributedGravityCalculator agent(tess, masses, theta, quadrupole);
double t1 = MPI_Wtime();
acc = agent.getAcceleration(points);
double t2 = MPI_Wtime();
std::cout << "Rank " << rank << " gravity: construct=" << (t1-t0)
          << "s eval=" << (t2-t1) << "s" << std::endl;
```

For finer granularity, add timing inside `getAcceleration` around: `getSendList`, `MPI_Exchange_all_to_all`, `addExternalValues` + `calculateMasses`, and the tree-walk loop.

### SIMD opportunities

Currently only `ShouldOpenBox` has a VCL vectorized path. `CalculateLeafGravityContribution` is scalar and is the main force-computation kernel -- a strong candidate for SIMD (structure-of-arrays layout for batched evaluation).

---

## 8. Adding New Features

### New gravity model

1. Subclass `Acceleration3D` in a new `.hpp`/`.cpp` under `source/newtonian/three_dimensional/`.
2. Implement `operator()(tess, cells, fluxes, time, acc)` -- fill `acc` with per-cell accelerations.
3. In your test's `test.cpp`, wrap it with `ConservativeForce3D` and add to `SeveralSources3D`, then pass to `HDSim3D`.
4. If using the `Simulation` class, `HydroStep` wraps `HDSim3D`, so gravity
   is still wired through `SourceTerm3D` inside `HDSim3D`.
5. No CMake changes needed -- the glob picks up all `source/**/*.cpp`.

### Modifying the tree

- All tree template code lives in `GravityTree.hpp` (header-only).
- Node data is `MassedValue.hpp`. To add fields: extend the struct, update `calculateMassHelper` aggregation, update `CalculateLeafGravityContribution`, and update MPI `dump`/`load`.
- `DistributedGravityCalculator.hpp` is also header-only.

### Adding higher multipoles

1. Extend `MassedValue::Q` from 6 to more components.
2. Update `calculateMassHelper` to aggregate the new terms (parallel-axis theorem).
3. Update `CalculateLeafGravityContribution` to apply the new multipole force corrections.
4. Update `GravityNodeData` in `DistributedGravityTree.hpp` and its `dump`/`load` to serialize the extra components.

### MPI changes

- Extend `GravityNodeData` serialization if node payload changes.
- The `getSendListHelper` decision logic (containment test, `ShouldOpenBox`, `getIntersectingRanks`) may need updating if you change how remote contributions are evaluated.
- All MPI gravity code is behind `#ifdef RICH_MPI`.
- `DistributedGravityCalculator` accepts an `MPI_Comm` parameter (defaults to `MPI_COMM_WORLD`).

---

## 9. Alternative Gravity Models

### `MonopoleSelfGravity3D`

Spherically symmetric approximation. Bins mass into radial shells around the global CM, then applies `acc = -M(r) * r_hat / r^2` with inner smoothing. Uses `MPI_Allreduce` for global CM, total mass, and radial mass profile. Cheap but assumes near-spherical geometry.

### `QuadrupoleGravity3D`

Extends the monopole with l=2 spherical harmonic corrections (Q20, Q21, Q22 real/imaginary, inner/outer). Uses adaptive radial binning (3 iterations to equalize mass per bin). Gradients are computed via finite differences. Many `MPI_Allreduce` calls (one per moment array). More accurate for mildly aspherical systems but still assumes a well-defined center.

**Important:** This is a completely different "quadrupole" from the octree
`Q[6]` in `MassedValue` / `GravityTree`. The octree quadrupole is a
traceless tensor for Barnes-Hut force corrections; `QuadrupoleGravity3D`
uses radial-profile spherical harmonics.

---

## 10. Test Reference

| Test | Path | What it tests |
|------|------|---------------|
| Lane-Emden regression | `regression_tests/cases/lane_self_gravity/test.cpp` | Polytropic star in hydrostatic equilibrium with `GravityAcceleration3D` |
| Lane-Emden FMM regression | `regression_tests/cases/lane_self_gravity_fmm/test.cpp` | Same equilibrium with the FMM solver |
| Serial FMM regression | `regression_tests/cases/fmm_gravity_serial/test.cpp` | Serial tree construction and force evaluation |
| MPI FMM regression | `regression_tests/cases/fmm_gravity_mpi/test.cpp` | Distributed topology and force evaluation |

Run the focused public regression:
```bash
./regression_tests/run_all.sh --test lane_self_gravity --config intelReleaseMPI --verbose
```

For detailed code excerpts and annotated algorithm logic, see [reference.md](reference.md).
