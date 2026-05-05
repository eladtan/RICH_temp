---
name: rich-mpi
description: >-
  Explains RICH's MPI parallel scheme: ghost/halo exchange, Hilbert domain
  decomposition, ExchangeChain repartitioning, serialization, parallel hydro
  loop, and Monte Carlo particle transport. Use when writing, debugging, or
  understanding parallel MPI code in the RICH codebase, or when the user
  mentions ghost cells, domain decomposition, MPI exchange, halo, or
  parallel tessellation.
---

# RICH MPI Parallel Architecture

All MPI code is guarded by `#ifdef RICH_MPI` (defined via `-DRICH_MPI` in MPI
build configs). The `rank_t` alias (`int`) is declared in
`source/mpi/mpi_commands.hpp`. Almost everything uses `MPI_COMM_WORLD`.

For detailed signatures and code excerpts, see [reference.md](reference.md).

---

## 1. Architecture Overview

Each MPI rank owns a subset of Voronoi mesh generators (indices `0..Norg_-1`).
The local Delaunay/Voronoi build adds **ghost points** from neighboring ranks
(indices `>= Norg_`) so that Voronoi faces at domain boundaries are consistent.

Data exchange between ranks is handled by templated free functions (not a
wrapper class). The three main paths are:

- **`Serializer`-based** (byte buffers, `MPI_BYTE`): `MPI_exchange_data_indexed`,
  `MPI_Exchange_all_to_all` (in `source/mpi/serialize/mpi_commands.hpp`).
- **Native MPI datatype fast path**: types that specialize
  `MPI_has_complex_dtype<T>` (e.g. `Particle3D` via `MPI_Particle3D_dtype.hpp`)
  bypass the `Serializer` and use `MPI_Type_create_struct` +
  `MPI_Isend`/`MPI_Recv` with the native datatype directly.
- **Double-buffer + graph communicator** (`source/mpi/mpi_exchange_commands.hpp`):
  types with `serialize() -> vector<double>`, `unserialize(vector<double>)`,
  and `getChunkSize()`, using `MPI_Dist_graph_create` +
  `MPI_Neighbor_alltoallv`.

---

## 2. Domain Decomposition (Hilbert Curve)

**Key files:**
- `source/3D/tessellation/loadBalancing/HilbertLoadBalancer.hpp` / `.cpp`
- `source/3D/tessellation/loadBalancing/CurveLoadBalancer.hpp` / `.cpp`
- `source/3D/tessellation/voronoi/pointsManager/HilbertPointsManager.hpp` / `.cpp`
- `source/utils/balance/weightedBalance2.hpp`

**How it works:**

1. `HilbertLoadBalancer::rebalance(points, weights)` maps each point to a
   Hilbert key via `convertor->xyz2d((*indexing)(point))`, then partitions with
   `getWeightedBorders2` (free function in `source/utils/balance/weightedBalance2.hpp`,
   iterates until max-rank weight is within ~5% of ideal, up to 20 iterations).
2. The constructor takes an optional `IndexingKernel3D` (defaults to
   `Kernelization3D::Identity`) which transforms points before Hilbert
   mapping. The concrete convertor is `HilbertRectangularConvertor3D`.
3. `CurveLoadBalancer` stores `vector<curve_index_t> boundaries`
   (`curve_index_t = size_t`) — the SFC cut positions separating ranks.
4. Ownership: `CurveLoadBalancer::getOwner` uses
   `upper_bound(boundaries, hilbert_key)` to find the owning rank.
   `Voronoi3D::GetOwner(point)` delegates to `EnvironmentAgent::getOwner`,
   which is built from the load balancer (concrete implementations:
   `DistributedOctEnvironmentAgent` or `HilbertTreeEnvironmentAgent`).
5. `HilbertPointsManager` moves `PointData` (a `Serializable` struct with
   position, radius, CM, weight, `participating` flag, and `indexInAllPoints`)
   between ranks during rebalancing. It delegates to `PointsManager::pointsExchange`,
   which calls `dataExchange` from `utils/exchange/exchange.hpp` (ultimately
   uses `MPI_Exchange_all_to_all`).

**Note:** `ParMETISPointsManager` exists in the tree but is entirely commented
out. Only the Hilbert path is active.

---

## 3. Ghost Cells and Halo Exchange

### Two "ghost" concepts

| Concept | Meaning | Where |
|---------|---------|-------|
| **Tessellation ghost** | Extra mesh point from a neighbor rank (`index >= Norg_`) added to the local Delaunay so Voronoi faces are correct. | `Voronoi3D::IsGhostPoint` |
| **Physical BC ghost** (`Ghost3D`) | Primitive-variable fill on boundary faces (reflecting, outflow, etc.). Operates on faces where one neighbor is `>= GetPointNo()`. | `source/newtonian/three_dimensional/Ghost3D.cpp` |

### Key data structures on `Voronoi3D` / `Tessellation3D`

| Member | Type | Meaning |
|--------|------|---------|
| `duplicatedprocs_` | `vector<rank_t>` | Neighbor ranks I communicate with |
| `duplicated_points_` | `vector<vector<size_t>>` | Per-neighbor: which of **my** owned indices they need |
| `Nghost_` | `vector<vector<size_t>>` | Per-neighbor: local indices of points received from them |
| `sentprocs_` / `sentpoints_` | same shape | Inverse: who I receive from when data flows the other way |
| `self_index_` | `vector<size_t>` | Maps old local indices to compacted post-send layout |
| `real_duplicated_proc` | `vector<rank_t>` | Previous build's duplicated procs (for seeding next ghost exchange) |
| `real_duplicated_points` | `vector<vector<size_t>>` | Previous build's duplicated points |

### `MPI_exchange_data` — the workhorse

`source/mpi/mpi_commands_3d.hpp`:

```cpp
template<class T>
void MPI_exchange_data(const Tessellation3D& tess,
                       std::vector<T>& cells,
                       bool ghost_or_sent,
                       const size_t extent = 1,
                       const T* example_cell = nullptr);
```

- **`ghost_or_sent == true`**: pack local data at `duplicated_points_` indices,
  send to `duplicatedprocs_`, write received data into `cells` at `Nghost_`
  positions. Resizes `cells` to `GetTotalPointNumber() * extent`.
- **`ghost_or_sent == false`**: pack local data at `sentpoints_` indices, send
  to `sentprocs_`, compact `cells` to `self_index_`, append received data.

Internally calls `MPI_exchange_data_indexed` which uses `Serializer` +
`MPI_Isend` / `MPI_Probe` / `MPI_Recv` / `MPI_Waitall` / `MPI_Barrier`.

A second overload exists for `vector<vector<T>>` exchange using
`GetDuplicatedProcs()` directly.

### `ExchangeGhosts` / `ExchangeFaces`

`source/3D/tessellation/ExchangeGhosts.cpp` builds a
`flat_map<size_t, pair<rank_t, size_t>>` from local ghost index to
`(owner_rank, owner_local_index)` by exchanging duplicated-point index lists
via `MPI_exchange_data(tess, tess.GetDuplicatedPoints())`.

`ExchangeFaces` maps local face indices to `(rank, remote_face_index)` for
faces that straddle domain boundaries. Internally builds the ghost map via
`ExchangeGhosts`, then uses `MPI_Exchange_all_to_all` for `NeighborsInfo`
and `FacesMatch` structs.

---

## 4. Parallel Voronoi Build

**Key file:** `source/3D/tessellation/voronoi/Voronoi3D.cpp`

**Entry point:** `Tessellation3D::BuildParallel(points, weights)` (base class)
delegates to `Voronoi3D::BuildPartiallyParallel(points, weights, indicesToBuild)`,
which accepts a subset of indices to rebuild.

1. **`PrepareToBuildParallel`**: uses `MPI_Allreduce(MPI_LAND)` to agree on
   whether to rebalance/exchange; calls `PointsManager::update` which may
   trigger `HilbertLoadBalancer::rebalance` + `pointsExchange`. Populates
   `sentprocs_`, `sentpoints_`, `self_index_`.
2. **`BuildPartiallyParallel`** calls `PrepareToBuildParallel`, builds
   Delaunay, `UpdatePointsTree`, `UpdateRadiuses`,
   **`BringGhostPointsToBuild(MPI_COMM_WORLD)`**, then `BuildVoronoi`, and
   finally `MPI_exchange_data(*this, volume_, true)` to sync ghost volumes.
3. **`BringGhostPointsToBuild`** (AREPO-style, paper ref: AREPO 2.4):
   - Seeds from `InitialGhostPointsExchange` (reuse previous build's
     `real_duplicated_*` lists for continuity).
   - Iterates: `SetGhostArray` + `del_.BuildExtra` to append received
     positions into local Delaunay, spherical range queries via
     `BigRangeAgent` / `SmallRangeAgent` + `EnvironmentAgent`
     (backed by `BuffersManagerQueryAgent` for MPI query batching),
     plus `BringSelfGhostPoints` for local boundary mirroring.
   - Convergence: `MPI_Iallreduce(MPI_LAND)` on "finished" flags across ranks.
   - Finalizes with `UpdateDuplicatedPoints` and `EnsureSymmetry`.
4. After `BuildParallel`, the caller invokes `ExchangeChain::Exchange` to
   track which indices moved where.

---

## 5. ExchangeChain (Repartitioning Bookkeeping)

**Key file:** `source/mpi/ExchangeChain.hpp` / `.cpp`

Tracks global index provenance across mesh rebuilds / load rebalancing.

```cpp
class ExchangeChain {
    using RankTransferMap = boost::container::flat_map<size_t, std::pair<rank_t, size_t>>;

    void Reset(size_t num);        // identity maps for 0..num-1
    void Exchange(ranks, indices, localIndices);
    ExchangeChain Reverse() const; // swap forward/backward maps

    // globalTransfer:        old_index -> (current_rank, current_index)
    // globalTransferOrigins: current_index -> (origin_rank, origin_index)
};
```

`Exchange` internally uses `MPI_Alltoall` on `MPI_UNSIGNED_LONG_LONG` for
counts/offsets, then two `MPI_Exchange_all_to_all` calls to ship
`ForwardInformation` and `UpdateOriginalInformation` structs.

A free-function template in `ExchangeChain.hpp` provides chain-based data
migration:

```cpp
template<typename T>
void MPI_exchange_data(const ExchangeChain& chain, std::vector<T>& data);
```

**Caveat:** this uses hardcoded `MPI_COMM_WORLD`, not the chain's stored
`comm` member.

**Usage in hydro loop:**

```cpp
exchange_chain_.Reset(tess_.GetPointNo());
// ... timestep ...
UpdateTessellation(tess_, point_vel, dt, exchange_chain_);
// chain now knows where each point migrated
MPI_exchange_data(tess_, extensive_, false);  // repack using tess info
MPI_exchange_data(tess_, cells_, false);
```

---

## 6. Serialization

**Key files:**
- `source/mpi/serialize/Serializer.hpp` / `.cpp`
- `source/mpi/serialize/Serializable.hpp`
- `source/mpi/serialize/mpi_commands.hpp`
- `source/mpi/MPI_complex_dtype.hpp`
- `source/mpi/MPI_Particle3D_dtype.hpp`

### `Serializer` (byte buffer)

Owns `vector<char> internal`. Template `insert`/`extract`:
- Trivially copyable `T`: raw `memcpy`.
- `vector<T>`: writes a `size_t` byte-length placeholder, payload, patches
  length.
- `Serializable` subclasses: `data.dump(this)` / `data.load(this, offset)`.
- `insert_all_indexed(data, indices, extent)`: strided gather for ghost
  patterns.

### `Serializable` interface

```cpp
class Serializable {
    virtual size_t dump(Serializer *) const = 0;
    virtual size_t load(const Serializer *, size_t byteOffset) = 0;
};
```

### Native MPI datatype fast path

`MPI_has_complex_dtype<T>` (default `false_type`) can be specialized to
`true_type` for types that define a custom `MPI_Datatype`. When specialized,
`MPI_Exchange_by_ranks` skips the `Serializer` and sends/receives using
the native MPI datatype directly. Example: `MPI_Particle3D_dtype.hpp`
registers `Particle3D` with `MPI_Type_create_struct`.

### Double-buffer path (`mpi_exchange_commands.hpp`)

Types expose `serialize() -> vector<double>`, `unserialize(vector<double>)`,
`getChunkSize()`. Uses `MPI_Dist_graph_create` + `MPI_Neighbor_alltoallv`.
This is a separate subsystem from the `Serializer`-based path.

---

## 7. Parallel Hydro Loop

### Legacy: `HDSim3D`

**Key file:** `source/newtonian/three_dimensional/hdsim_3d.cpp`

`HDSim3D::timeAdvance` (and variants `timeAdvance2`, `timeAdvance3`,
`timeAdvance32`, `timeAdvance33`, `timeAdvance4`):

1. `exchange_chain_.Reset(tess_.GetPointNo())`
2. Compute point velocities -> `MPI_exchange_data(tess_, point_vel, true)`
   (sync ghosts)
3. Global dt via `CourantFriedrichsLewy`: one `MPI_Allreduce(MPI_MIN)` pass
   in `source/newtonian/three_dimensional/CourantFriedrichsLewy.cpp`
4. Compute fluxes, source terms, extensive update (local)
5. If mesh moved: `UpdateTessellation` + `ExchangeChain::Exchange`
6. `MPI_exchange_data(tess_, extensive_, false)` — repack sent data
7. `MPI_exchange_data(tess_, cells_, false)` — repack sent data
8. Cell update (local)
9. `MPI_exchange_data(tess_, cells_, true)` — refresh ghost cells
10. Advance time and cycle

Higher-order variants add the same exchanges around predictor-corrector
substeps. `PreparePoints` (Hilbert reorder) runs every ~10 cycles, only
in `timeAdvance2`, guarded by `pt_.getCycle() % 10 == 0 && pm_.MovedPoints()`.

### Modern: `Simulation` class

**Key file:** `source/newtonian/three_dimensional/simulation/Simulation.hpp` / `.cpp`

`Simulation` wraps `HDSim3D` (via `HydroStep`) for newer simulations,
adding higher-level orchestration. Key MPI aspects:

- **Cell IDs:** `initializeCellIDs()` uses `MPI_Allgather` to compute global
  ID offsets; `recomputeMaxID()` uses `MPI_Allreduce(MPI_MAX)`. `Max_ID` is
  `size_t`.
- **Migration buffers:** `addMigrationBuffer<T>(vector<T>&)` registers
  buffers that are automatically transferred via `MPI_exchange_data` after
  each `BuildParallel` call. `buildDataTransfer()` triggers the transfer,
  optionally using an `ExchangeChain`.
- **Load balancing:** `storeLoadBalance` / `setCurrentLoadBalance` manage
  named load balancers; `forceRebalanceSteps` forces rebalancing for N steps.
- **Physics steps:** the simulation is composed of `PhysicsStep` objects
  (`HydroStep`, `RadiationStep`, `RadiationMCStep`, `RemeshStep`).
  Gravity remains a `SourceTerm3D` inside `HDSim3D`, wrapped by `HydroStep`.

---

## 8. Monte Carlo Particle Transport

**Key files:**
- `source/3D/monte/Voronoi3DMovement.cpp`
- `source/3D/monte/MonteCarloManager3D.hpp`
- `source/monte/manager/MonteCarloManager.hpp` (RMA-based)
- `source/monte/two_sided_manager/TwoSidedMonteCarloManager.hpp`
- `source/monte/two_sided_manager/BuffersManager.hpp`
- `source/utils/rma-helpers/ProgressCounter.hpp`
- `source/utils/rma-helpers/GlobalCounter.hpp`

### Two MPI MC manager paths

| Class | Communication | Header |
|-------|--------------|--------|
| `RDMAMonteCarloManager3D` | MPI one-sided (RMA) via `RMAFactory` / `MPIRemoteMemoryAgent` | `MonteCarloManager3D.hpp` wrapping `MonteCarloManager.hpp` |
| `TwoSidedMonteCarloManager3D` | MPI two-sided via `BuffersManager` (`MPI_Isend`/`MPI_Irecv`) | `MonteCarloManager3D.hpp` wrapping `TwoSidedMonteCarloManager.hpp` |

Both inherit from the abstract `MonteCarloManager3D` interface.

### Particle redistribution after mesh rebuild

`TransferParticlesWithTranslationMap` in `Voronoi3DMovement.cpp`:
- Each particle's `cellIndex` is looked up in `ExchangeChain`'s translation
  map to get `(new_rank, new_cell_index)`.
- Particles staying local go to `selfParticles`.
- Others are bucketed into `particlesToProcessors[rank]`.
- `MPI_Exchange_all_to_all_serializers` ships them (Serializer-based).
- `MPI_Reduce` on rank 0 for sent/received counters.

### Runtime particle exchange

When a particle crosses a face into a ghost cell, the **ghost map**
(`ExchangeGhosts` result) tells which rank owns it.

- **RMA path** (`MonteCarloManager`): uses `RankHandler` with
  `MPIRemoteMemoryAgent` (`MPI_Win_create`, `MPI_Put`, `MPI_Win_flush`)
  for particle transfer.
- **Two-sided path** (`TwoSidedMonteCarloManager`): `BuffersManager` handles
  async sends/receives with `MPI_Isend` / `MPI_Irecv` and batched dispatch.

### Progress and load tracking

- `ProgressCounter` uses **MPI one-sided (RMA) windows** (`MPI_Win_allocate`,
  `MPI_Put`, `MPI_Win_lock`/`unlock`/`flush`) for lightweight done-flag
  signaling.
- `GlobalCounter` and `ConditionVariable` (in `source/utils/rma-helpers/`)
  also use RMA windows for distributed coordination.
- `MPI_Reduce` / `MPI_Allreduce` for diagnostics.
- `RankSync` (`ForEachRankSync`, `ForEachRankSyncByList`) serializes
  execution across ranks with ordered barriers using `MPI_Allgather` /
  `MPI_Allgatherv`.

---

## 9. Common Pitfalls and Debugging

### `MPI_COMM_WORLD` everywhere

Most exchange helpers hardcode `MPI_COMM_WORLD`. Notable examples:
`MPI_exchange_data(ExchangeChain&, ...)` uses `MPI_COMM_WORLD` even though
`ExchangeChain` stores its own `comm`. If you introduce subcommunicators,
you must thread them through manually.

### `SmartCollectives` (debug builds)

`RMPI_Barrier`, `RMPI_Reduce`, `RMPI_Allreduce` in
`source/mpi/SmartCollectives.hpp` — before calling the real MPI collective,
they gather **backtraces** from every rank via `MPI_All_cast_by_ranks` and
assert all stacks match (`EnsureSameStack`). Use these in debug builds to
catch collective ordering bugs (rank A calls `Barrier` while rank B calls
`Allreduce`).

### `MPI_Timed_barrier`

`source/mpi/mpi_commands.cpp` — all-to-all with timeout. If a rank doesn't
respond within `seconds`, throws `UniversalError`. Use to debug hangs:

```cpp
MPI_Timed_barrier(MPI_COMM_WORLD, 60.0, "after flux computation");
```

### Debugging exchange mismatches

- Verify `duplicatedprocs_` / `sentprocs_` are symmetric: if rank A lists
  rank B as a correspondent, rank B must list A.
- `EnsureSymmetry` in the Voronoi build enforces this after ghost search.
- Size mismatches in `MPI_exchange_data_indexed` throw
  `UniversalError("Extent size does not match")`.

### `#ifdef RICH_MPI` discipline

Every MPI call and MPI-dependent type must be inside `#ifdef RICH_MPI`.
Serial builds compile without MPI headers entirely. `rank_t` is defined
unconditionally (as `int`) so it can appear in serial code signatures.
