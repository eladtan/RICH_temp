# RICH MPI Reference

Detailed signatures, code excerpts, and the MPI collectives inventory.
Companion to [SKILL.md](SKILL.md).

All file paths are relative to the RICH `source/` directory.

---

## Point-to-Point Indexed Exchange

**File:** `mpi/mpi_commands.hpp`

```cpp
template<typename T, typename Index_T = size_t>
std::vector<std::vector<T>> MPI_exchange_data_indexed(
    const std::vector<rank_t> &correspondents,
    const std::vector<T> &data,
    const std::vector<std::vector<Index_T>> &indices = std::vector<std::vector<Index_T>>(),
    const size_t &extent = 1);
```

- Packs `data[indices[i][j] * extent + k]` for each correspondent via
  `Serializer::insert_all_indexed`.
- `MPI_Isend` to each correspondent (tag `MPI_EXCHANGE_TAG = 5`), then
  `MPI_Probe(MPI_ANY_SOURCE)` + `MPI_Recv` for each, `MPI_Waitall`,
  `MPI_Barrier(MPI_COMM_WORLD)`.
- Returns `result[i]` = vector of `T` received from `correspondents[i]`.

---

## Tessellation-Based Exchange (3D)

**File:** `mpi/mpi_commands_3d.hpp`

```cpp
template<class T>
void MPI_exchange_data(
    const Tessellation3D& tess,
    std::vector<T>& cells,
    bool ghost_or_sent,
    const size_t extent = 1,
    const T* example_cell = nullptr);
```

### `ghost_or_sent == true` (refresh ghost copies)

1. `correspondents = tess.GetDuplicatedProcs()`
2. `indices = tess.GetDuplicatedPoints()` (my owned indices the neighbor needs)
3. Calls `MPI_exchange_data_indexed(correspondents, cells, indices, extent)`
4. `cells.resize(tess.GetTotalPointNumber() * extent)`
5. Scatters received data into `cells[Nghost_[i][j] * extent + k]`

### `ghost_or_sent == false` (repack after repartitioning)

1. `correspondents = tess.GetSentProcs()`
2. `indices = tess.GetSentPoints()`
3. Calls `MPI_exchange_data_indexed`
4. `cells = VectorValues(cells, tess.GetSelfIndex())` (compact)
5. Appends received data

### `vector<vector<T>>` overload

```cpp
template<class T>
void MPI_exchange_data(
    const Tessellation3D& tess,
    std::vector<std::vector<T>>& data);
```

Uses `GetDuplicatedProcs()` and delegates to the correspondent-based
`MPI_exchange_data(correspondents, data)` overload.

---

## Serializer-Based Collectives

**File:** `mpi/serialize/mpi_commands.hpp`

### `MPI_Exchange_all_to_all`

```cpp
template<typename T, template<typename...> class Container, typename... Ts>
std::vector<std::vector<T>> MPI_Exchange_all_to_all(
    const std::vector<Container<T, Ts...>> &data,
    const MPI_Comm &comm);
```

- Single `Serializer` for all outgoing data.
- `MPI_Alltoall` for byte counts, `MPI_Alltoallv` for payload (`MPI_BYTE`).
- More efficient than `Iexchange` for true all-to-all patterns.
- No final barrier (unlike `MPI_Exchange_by_ranks`).

### `MPI_Exchange_by_ownership_by_ranks` / `MPI_Exchange_by_ownership`

Bucket items by `ownership(value) -> rank`, then `MPI_Exchange_all_to_all`.

### `MPI_All_cast_by_ranks` / `MPI_All_cast`

Broadcast every rank's data to all ranks. `MPI_Allgather` for sizes,
`MPI_Alltoallv` for payload.

### `MPI_Bcast_serializable`

```cpp
template<typename T>
T MPI_Bcast_serializable(const T &data, rank_t owner,
                         const MPI_Comm &comm = MPI_COMM_WORLD);
```

`MPI_Bcast` size, then `MPI_Bcast` bytes.

### `MPI_Gatherv_serializable`

Gather serialized data to `root`. `MPI_Gather` for byte counts,
`MPI_Gatherv` for payload.

### `MPI_Spread`

Scatter from `root` evenly across ranks. `MPI_Scatter` for counts,
`MPI_Scatterv` for payload.

### `MPI_Ask_data`

```cpp
template<typename T, typename Index_T = size_t>
std::vector<std::vector<T>> MPI_Ask_data(
    const std::vector<rank_t> &correspondents,
    const std::vector<T> &myData,
    const std::vector<std::vector<Index_T>> &myRequestedIndices);
```

1. All-to-all exchange of requested index lists.
2. `MPI_exchange_data_indexed` over all ranks to fulfill requests.
3. Returns only data from `correspondents`.

---

## Native MPI Datatype Fast Path

**Files:** `mpi/MPI_complex_dtype.hpp`, `mpi/MPI_Particle3D_dtype.hpp`

`MPI_has_complex_dtype<T>` (default `std::false_type`) can be specialized to
`std::true_type` for types that register a custom `MPI_Datatype` via
`MPI_Type_create_struct`. When specialized, `MPI_Exchange_by_ranks` uses
the native datatype instead of `Serializer` + `MPI_BYTE`.

Currently `Particle3D` is the only type with a native dtype specialization.

---

## Double-Buffer Graph Exchange

**File:** `mpi/mpi_exchange_commands.hpp`

For types with `serialize() -> vector<double>`, `unserialize(vector<double>)`,
`getChunkSize()`.

```cpp
MPI_Comm MPI_Create_graph_comm(sendProcs, sendCounts, comm);
// -> MPI_Dist_graph_create

// Send/recv helpers use:
//   MPI_Neighbor_alltoallv on double buffers
//   AgreeChunkSize: MPI_Allreduce(MPI_MAX) on chunk size
```

`CalculateSerializableSendData` builds the flat double array + counts +
displacements. `SyncReceiveData` exchanges counts via `MPI_Neighbor_alltoall`
and returns receive layout.

This is a separate subsystem from the `Serializer`-based path above.

---

## ExchangeChain Internals

**File:** `mpi/ExchangeChain.hpp` / `.cpp`

### Data members

```cpp
MPI_Comm comm;
int rank, size;
RankTransferMap globalTransfer;        // old_index -> (current_rank, current_index)
RankTransferMap globalTransferOrigins; // current_index -> (origin_rank, origin_index)
RankTransferMap lastTransfer;          // pre-transfer index -> (now_rank, now_index)
RankTransferMap origins;               // post-transfer index -> (pre-transfer_rank, index)
```

### `Exchange` flow

1. `UpdateTransferMap`: `MPI_Alltoall` on `MPI_UNSIGNED_LONG_LONG` for
   per-destination send counts and offsets into merged receive layout.
2. First `MPI_Exchange_all_to_all`: ships `ForwardInformation` (new index,
   sender's previous index, global original rank/index). Rebuilds
   `globalTransferOrigins` and `origins`.
3. Second `MPI_Exchange_all_to_all`: ships `UpdateOriginalInformation`
   (original index -> new index). Rebuilds `globalTransfer`.

### `Reverse`

Swaps `globalTransfer <-> globalTransferOrigins` and
`lastTransfer <-> origins`.

### Chain-based data migration template

```cpp
template<typename T>
void MPI_exchange_data(const ExchangeChain& chain, std::vector<T>& data);
```

Defined in `ExchangeChain.hpp`. Builds `toSend[worldSize]` from
`GetTranslationMap()`, calls `MPI_Exchange_all_to_all(toSend, MPI_COMM_WORLD)`,
rebuilds `data`. **Caveat:** hardcodes `MPI_COMM_WORLD`, ignoring the
chain's stored `comm`.

---

## Parallel Voronoi Build: `BringGhostPointsToBuild`

**File:** `3D/tessellation/voronoi/Voronoi3D.cpp`

### Entry point

`Tessellation3D::BuildParallel(points, weights)` delegates to
`Voronoi3D::BuildPartiallyParallel(points, weights, indicesToBuild)`.
`BuildPartiallyParallel` calls `PrepareToBuildParallel`, builds the
Delaunay, then calls `BringGhostPointsToBuild(MPI_COMM_WORLD)`, then
`BuildVoronoi`, and finally `MPI_exchange_data(*this, volume_, true)`.

### Algorithm (AREPO-style iterative ghost search)

```
1. InitialGhostPointsExchange
     Re-send generators from previous build's real_duplicated_* lists.
     -> SetGhostArray + del_.BuildExtra

2. Repeat until all ranks converge:
   a. BringRemoteGhostPoints (BigRangeAgent / SmallRangeAgent)
      - Spherical range queries to EnvironmentAgent
      - MPI query batching via BuffersManagerQueryAgent
      - Sends/receives point positions across ranks
   b. BringSelfGhostPoints
      - Local boundary mirroring
   c. SetGhostArray + del_.BuildExtra for newly received points
   d. MPI_Iallreduce (MPI_LAND) on "finished" flags

3. UpdateDuplicatedPoints from range agents' sent lists

4. EnsureSymmetry
     Makes duplicatedprocs_ / sentprocs_ reciprocal
```

### Key functions called

| Function | Role |
|----------|------|
| `SetGhostArray(recvProc, recvPoints)` | Appends to `Nghost_[rankIdx]` |
| `del_.BuildExtra(points)` | Inserts ghost vertices into Delaunay |
| `UpdateDuplicatedPoints(...)` | Populates `duplicated_points_` |
| `EnsureSymmetry()` | Fixes asymmetric neighbor lists |

---

## MPI Collectives Inventory

| Area | File | Collective | Purpose |
|------|------|-----------|---------|
| Hydro dt | `CourantFriedrichsLewy.cpp` | `MPI_Allreduce MPI_MIN` (x1) | Global minimum timestep |
| Simulation init | `simulation/Simulation.cpp` | `MPI_Allgather` | Global cell ID offsets |
| Simulation init | `simulation/Simulation.cpp` | `MPI_Allreduce MPI_MAX` | Max ID on restart |
| Ghost exchange | `mpi_commands.hpp` | `MPI_Isend` / `MPI_Recv` / `MPI_Waitall` / `MPI_Barrier` | Halo data sync |
| ExchangeChain | `ExchangeChain.cpp` | `MPI_Alltoall` + `MPI_Exchange_all_to_all` (x2) | Repartition bookkeeping |
| Voronoi build | `Voronoi3D.cpp` | `MPI_Iallreduce MPI_LAND` | Ghost search convergence |
| Voronoi build | `Voronoi3D.cpp` | `MPI_Allreduce MPI_LAND` | Rebalance/exchange decisions |
| Load balancing | `HilbertLoadBalancer.cpp` | `MPI_Allreduce MPI_MAX` | Uniform-weight agreement |
| AMR | `AMR3D.cpp` | `MPI_Allreduce MPI_SUM` | Refine/remove counts |
| AMR | `AMR3D.cpp` | `MPI_Allgather` | ID offsets |
| Gravity | `QuadrupoleGravity3D.cpp` | `MPI_Allreduce MPI_SUM` / `MPI_Barrier` | Multipole moments |
| Gravity | `MonopoleSelfGravity3D.cpp` | `MPI_Allreduce MPI_SUM` / `MPI_Barrier` | Monopole sums |
| Gravity (tree) | `DistributedGravityCalculator.hpp` | `MPI_All_cast_by_ranks` + `MPI_Exchange_all_to_all` | Tree data exchange |
| MC transport | `Voronoi3DMovement.cpp` | `MPI_Exchange_all_to_all_serializers` | Particle redistribution |
| MC transport | `Voronoi3DMovement.cpp` | `MPI_Reduce MPI_SUM` / `MPI_Barrier` | Diagnostics |
| MC manager (two-sided) | `BuffersManager` | `MPI_Isend` / `MPI_Irecv` | Async particle transfer |
| MC manager (RMA) | `MonteCarloManager` / `RMAFactory` | `MPI_Win_create` / `MPI_Put` / `MPI_Win_flush` | One-sided particle transfer |
| MC counters | `ProgressCounter` / `GlobalCounter` | MPI RMA windows (`MPI_Win_allocate`) | Global counters |
| Debug | `SmartCollectives.cpp` | `RMPI_Barrier` / `RMPI_Reduce` / `RMPI_Allreduce` | Stack-checked collectives |
| Debug | `mpi_commands.cpp` | `MPI_Timed_barrier` (all-to-all + timeout) | Hang detection |
| Native dtype | `serialize/mpi_commands.hpp` | `MPI_Isend` / `MPI_Recv` with native `MPI_Datatype` | Fast path for `MPI_has_complex_dtype` types |
| Serialized all-to-all | `serialize/mpi_commands.hpp` | `MPI_Alltoall` + `MPI_Alltoallv` | Bulk byte exchange |
| Serialized gather | `serialize/mpi_commands.hpp` | `MPI_Gather` + `MPI_Gatherv` | Root collection |
| Serialized broadcast | `serialize/mpi_commands.hpp` | `MPI_Bcast` (x2) | Root broadcast |
| Serialized scatter | `serialize/mpi_commands.hpp` | `MPI_Scatter` + `MPI_Scatterv` | Root distribution |
| All-cast | `serialize/mpi_commands.hpp` | `MPI_Allgather` + `MPI_Alltoallv` | Everyone gets everything |
| Load imbalance | `ProcessorUpdate.cpp` | `MPI_Allgather` | Per-rank point counts |
| Graph exchange | `mpi_exchange_commands.hpp` | `MPI_Dist_graph_create` + `MPI_Neighbor_alltoallv` | Sparse neighbor exchange |

---

## Tags and Constants

| Constant | Value | Where |
|----------|-------|-------|
| `MPI_EXCHANGE_TAG` | 5 | `mpi/mpi_commands.hpp` |
| `MPI_EXCHANGE_ALLTOALL_TAG` | 1039 | `mpi/serialize/mpi_commands.hpp` |
| `MPI_TIMED_BARRIER_TAG` | 110503 | `mpi/mpi_commands.hpp` |
