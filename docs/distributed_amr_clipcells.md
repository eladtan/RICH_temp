# Distributed AMR clipCells Load Balancing

## Problem

During AMR (Adaptive Mesh Refinement) steps in MPI-parallel RICH simulations, the computational
load is severely imbalanced. Only a small subset of MPI ranks have cells marked for refinement or
removal, while the majority sit idle at an `MPI_Barrier`. Nearly all of the AMR wall time is spent
inside `clipCells` -- a geometric function that computes the volume and center-of-mass of a
polyhedron clipped against a set of half-planes. On the few busy ranks, hundreds or thousands of
`clipCells` calls execute sequentially while the rest of the machine does nothing.

## Solution Overview

The solution offloads excess `clipCells` work from busy ranks to idle ranks using MPI collective
communication. The feature is controlled by the `distribute_clips_` flag on the `AMR3D` class
(default: `true`). When disabled, the original sequential code paths are used unchanged.

The approach introduces four new functions that mirror the original four AMR clip routines:

| Original (sequential)  | New (distributed)             | Scope                        |
|-------------------------|-------------------------------|------------------------------|
| `LocalRefine`           | `DistributedLocalRefine`      | Same-rank refine clips       |
| `LocalRemove`           | `DistributedLocalRemove`      | Same-rank remove clips       |
| `MPIRefine`             | `DistributedMPIRefine`        | Cross-rank refine clips      |
| `MPIRemove`             | `DistributedMPIRemove`        | Cross-rank remove clips      |

Each distributed function: (1) pre-builds a flat list of clip tasks, (2) calls a shared
`DistributeAndComputeClips` function that balances work across all ranks, and (3) applies the
returned results to the extensive variables.

---

## Files Modified

### `source/newtonian/three_dimensional/AMR3D.hpp`

- Added `bool distribute_clips_` private member to `AMR3D`.
- Added optional `bool distribute_clips = true` parameter to the constructor.
- Added `void SetDistributeClips(bool v)` public setter.

### `source/newtonian/three_dimensional/AMR3D.cpp`

This file contains all of the new logic. Changes are organized into the sections below.

---

## Detailed Changes in `AMR3D.cpp`

### 1. Geometry Packing / Unpacking Helpers (anonymous namespace, `#ifdef RICH_MPI`)

Six helper functions serialize and deserialize the geometric data structures used by `clipCells`
into flat `std::vector<double>` buffers suitable for `MPI_Exchange_all_to_all`:

- **`PackPolyhedron` / `UnpackPolyhedron`** -- Serialize a `vector<Face>` (the polyhedron). Format:
  `[nfaces, nvert_0, x,y,z, ..., nvert_1, x,y,z, ...]`. Each face stores its vertex count followed
  by the 3D coordinates of each vertex.

- **`PackPlanes` / `UnpackPlanes`** -- Serialize a `vector<Plane>`. Format:
  `[nplanes, d_0, nx,ny,nz, d_1, nx,ny,nz, ...]` where `d = dot(normal, point)`. On unpack, a
  valid point on the plane is reconstructed from `d` and the normal's largest component. This
  representation is the same one already used by `SendRecvMPIFullRemove` and `MPIRefine` for
  cross-rank plane exchange.

- **`PackBounds` / `UnpackBounds`** -- Serialize a `ClipBounds` (AABB + valid flag) as 7 doubles:
  `[lower.x, lower.y, lower.z, upper.x, upper.y, upper.z, valid]`.

### 2. `ClipResultEntry` Struct

A simple POD struct holding the result of a single clip computation:

```cpp
struct ClipResultEntry {
    size_t task_id;   // index into the caller's task list
    double dv;        // clipped volume
    Vector3D clip_CM; // center of mass of the clipped region
};
```

### 3. `DistributeAndComputeClips` -- The Core Load Balancer

This function is the heart of the feature. It takes a flat buffer of packed clip tasks, the number
of local tasks, the original cell volumes, and an MPI communicator. It returns a vector of
`ClipResultEntry` results.

The function executes in five phases: census, assignment, exchange, computation, and return. Each
phase is described below.

#### Phase 1 -- Census: Who Has Work, Who Is Idle?

Every rank broadcasts its local task count via a single `MPI_Allgather`:

```
task_counts = [0, 0, 150, 0, 0, 80, 0, 0]   (8 ranks, only ranks 2 and 5 have work)
total_tasks = 230
fair_share  = ceil(230 / 8) = 29
```

`fair_share` is the target number of tasks per rank if work were evenly distributed. Any rank with
more than `fair_share` tasks is *overloaded*; any rank with fewer is *underloaded* and has capacity
to absorb work. Ranks with exactly `fair_share` tasks are neither.

In the example above:
- Overloaded: rank 2 (excess = 150 - 29 = 121), rank 5 (excess = 80 - 29 = 51)
- Underloaded: ranks 0, 1, 3, 4, 6, 7 (deficit = 29 each, total deficit = 174)

#### Phase 2 -- Assignment: Building a Globally-Deterministic Mapping

The critical challenge is that multiple overloaded ranks must agree on *who sends to whom* without
any additional communication. We achieve this by having every rank independently compute the same
assignment table from the same `task_counts` array.

The algorithm iterates through overloaded ranks in rank order. For each overloaded rank, it
greedily fills the deficit of underloaded ranks (also in rank order), consuming deficit capacity
as it goes. Because the iteration order and the data are identical on all ranks, the result is
identical everywhere.

**Worked example** (continuing from above):

```
Overloaded ranks (in order):  [rank 2 (excess 121), rank 5 (excess 51)]
Underloaded ranks (in order): [rank 0 (deficit 29), rank 1 (deficit 29),
                                rank 3 (deficit 29), rank 4 (deficit 29),
                                rank 6 (deficit 29), rank 7 (deficit 29)]
```

Processing rank 2 (121 excess tasks to shed):
- Send 29 to rank 0 (rank 0 full, remaining excess = 92)
- Send 29 to rank 1 (rank 1 full, remaining excess = 63)
- Send 29 to rank 3 (rank 3 full, remaining excess = 34)
- Send 29 to rank 4 (rank 4 full, remaining excess = 5)
- Send  5 to rank 6 (rank 6 still has 24 capacity, remaining excess = 0)

Processing rank 5 (51 excess tasks to shed):
- Send 24 to rank 6 (rank 6 now full, remaining excess = 27)
- Send 27 to rank 7 (rank 7 still has 2 capacity, remaining excess = 0)

Final distribution:

| Rank | Original | Kept | Received from rank 2 | Received from rank 5 | Total |
|------|----------|------|----------------------|----------------------|-------|
| 0    | 0        | 0    | 29                   | --                   | 29    |
| 1    | 0        | 0    | 29                   | --                   | 29    |
| 2    | 150      | 29   | --                   | --                   | 29    |
| 3    | 0        | 0    | 29                   | --                   | 29    |
| 4    | 0        | 0    | 29                   | --                   | 29    |
| 5    | 80       | 29   | --                   | --                   | 29    |
| 6    | 0        | 0    | 5                    | 24                   | 29    |
| 7    | 0        | 0    | --                   | 27                   | 27    |

Each rank computes this same table. Only the overloaded rank whose `rank == myrank` actually
records its assignments into a local `my_assignments` list. All other ranks skip the recording but
still advance the deficit pointer, so the global cursor position stays synchronized.

**Why not a simpler approach?** An earlier version had each overloaded rank independently scan the
deficit array and send to the first idle ranks it found. This was buggy: when two overloaded ranks
both tried to send to the same idle rank (e.g., both pick rank 0 first), that idle rank would
receive double the intended load. The current approach avoids this entirely because each overloaded
rank consumes deficit space left over by the previous overloaded rank -- the cursor never resets.

**Edge cases:**
- If `total_tasks == 0` (no work anywhere): `fair_share = 0`, both lists are empty, nothing
  happens.
- If all tasks are on one rank: that rank is the only overloaded one, it fills idle ranks in order.
- If total deficit < total excess (more excess than capacity): excess tasks that cannot be placed
  remain on the overloaded rank and are computed locally. This is handled because the inner while
  loop exits when `u_idx` exhausts the underloaded list.

#### Phase 3 -- Packing and Exchanging Tasks

Each overloaded rank now knows exactly which tasks to send to which ranks. It needs to extract
individual tasks from its flat `packed_tasks` buffer. Since tasks are variable-length (different
polyhedra have different numbers of faces and vertices), the rank first builds an offset table by
scanning through the packed buffer:

```
task_offsets[t] = byte position where task t starts
                  (scan: nfaces -> for each face: nverts -> skip 3*nverts doubles
                         -> nplanes -> skip 4*nplanes doubles -> skip 14 doubles for two bounds)
```

Then for each assignment `{target_rank, count}`, it copies the relevant byte ranges from
`packed_tasks` into `send_data[target_rank]`, prepending each task with its original task index
(so results can be mapped back later). The corresponding `org_volumes` are copied into a parallel
`send_volumes[target_rank]` array.

Both `send_data` and `send_volumes` are exchanged via `MPI_Exchange_all_to_all`. After the
exchange, every rank has:
- `recv_data[r]`: packed tasks received from rank `r`
- `recv_volumes[r]`: original volumes for those tasks

Ranks that were not overloaded send empty vectors; ranks that were not underloaded receive empty
vectors. The `MPI_Exchange_all_to_all` handles this sparsity efficiently.

#### Phase 4 -- Computation: Everyone Works

Each rank now has two sources of work:

1. **Received tasks** (from overloaded ranks): Unpack each task from `recv_data`, run `clipCells`,
   store results tagged with the originating rank and the original task index.

2. **Local tasks** (kept by this rank): The overloaded rank iterates through all its tasks in
   order, unpacking each one. Tasks in the range `[fair_share, fair_share + my_send_count)` were
   sent to other ranks, so they are skipped (the data is still parsed to advance the offset, but
   `clipCells` is not called). All other tasks are computed locally.

   For non-overloaded ranks, `my_send_count == 0` and `sent_start == local_task_count`, so nothing
   is skipped -- all local tasks are computed.

Both paths use the same threshold `dv > org_volume * 1e-10` to filter negligible intersections.

#### Phase 5 -- Returning Results

Remote results are packed as 5 doubles per result: `[task_id, dv, clip_CM.x, clip_CM.y, clip_CM.z]`
and sent back to the originating rank via a third `MPI_Exchange_all_to_all`. The originating rank
unpacks these and merges them with its locally-computed results into a single `results` vector.

The caller (e.g., `DistributedLocalRefine`) then iterates over `results`, uses `task_id` to look up
which cell pair the result corresponds to, and applies the conserved-variable update.

#### Communication Summary

| Step              | MPI call                   | Data volume                         |
|-------------------|----------------------------|-------------------------------------|
| Census            | `MPI_Allgather` (1 int)    | 4 bytes * world_size                |
| Task exchange     | `MPI_Exchange_all_to_all`  | ~200-400 doubles per offloaded task |
| Volume exchange   | `MPI_Exchange_all_to_all`  | 1 double per offloaded task         |
| Result return     | `MPI_Exchange_all_to_all`  | 5 doubles per non-trivial result    |

The census is negligible. The task exchange dominates bandwidth but only involves the excess tasks
(those beyond `fair_share`). The result return is very compact since most clip pairs produce zero
overlap and are filtered out before packing.

### 4. `DistributedLocalRefine`

Replaces the BFS-based `LocalRefine` for the distributed path. For each refined cell:

1. Build a clip task for the refined cell's polyhedron against every k=1 neighbor (the cell itself
   plus its immediate neighbors in the old tessellation). This is geometrically conservative: since
   the refined cell is created at 0.25 * cell_width from the parent center, it cannot extend beyond
   the parent's immediate Voronoi neighborhood.
2. Pack all tasks, call `DistributeAndComputeClips`.
3. Apply results: subtract extensives from old cells, add to the new refined cell.
4. Fallback: if a refined cell gets zero clipped volume, use the same heuristic as the original
   code (assign extensives proportionally from the parent cell).

### 5. `DistributedLocalRemove`

Replaces `LocalRemove` for the distributed path. For each removed cell:

1. Build a clip task for each local neighbor: the removed cell's polyhedron (from old tess) clipped
   against the neighbor's planes (from new tess). This exactly matches the neighbor set used by the
   original `LocalRemove`.
2. Pack all tasks, call `DistributeAndComputeClips`.
3. Apply results: add the clipped extensives to the absorbing neighbor cells.

### 6. `DistributedMPIRefine`

Replaces `MPIRefine` for the distributed path. Handles refined cells near MPI domain boundaries:

1. Identify refined cells whose immediate neighbors include a ghost cell.
2. Call `GetKOrderNeighbors(oldtess, boundary_cells, 1, ...)` to discover remote k=1 neighbors.
3. Call `SendRecvMPIRefine` to exchange the refined cell's planes and the remote neighbor indices
   with all relevant ranks.
4. On the receiving side, build clip tasks for each received neighbor cell against the received
   planes. No extra ring expansion -- the k=1 neighbor list is the complete clip set.
5. Pack all tasks, call `DistributeAndComputeClips`.
6. Accumulate results into `extensive_tosend`, exchange back to the originating rank via
   `MPI_Exchange_all_to_all`, and apply to extensives.

### 7. `DistributedMPIRemove`

Replaces `MPIRemove` for the distributed path. Uses the existing `SendRecvMPIFullRemove` to
exchange planes of removed cells with MPI neighbors. Then for each received entry, builds clip
tasks (local cell polyhedron vs. received planes), distributes via `DistributeAndComputeClips`, and
applies results.

### 8. `SendRecvMPIRefine` -- New Cross-Rank Communication for Refine

This function handles the plane and neighbor-index exchange between ranks for the MPI refine path.
Unlike the old approach that used `tess.GetDuplicatedProcs()` (communicating only with tessellation
neighbors), this function uses `MPI_Exchange_all_to_all` with world-rank-indexed arrays. This is
necessary because `GetKOrderNeighbors` can return neighbors on any rank, not just tessellation
neighbor ranks.

For each refined cell with remote neighbors, it packs:
- The refined cell's planes (as `d, nx, ny, nz` per plane).
- The remote neighbor indices (already in the remote rank's local index space).
- The refined cell's index in the new tessellation (stored in `changed_byouter` for result mapping).

### 9. `AMR3D::operator()` -- Integration

The main AMR driver conditionally calls the distributed or original functions based on
`distribute_clips_`:

```cpp
#ifdef RICH_MPI
if (distribute_clips_) {
    DistributedLocalRefine(...);
    DistributedMPIRefine(...);
} else {
    LocalRefine(...);
    MPIRefine(...);
}
#else
LocalRefine(...);
#endif
```

The same pattern is used for the remove phase. Serial builds (`#else`) always use the original
functions.

### 10. Constructor

The `AMR3D` constructor accepts the new `bool distribute_clips` parameter (default `true`) and
initializes the member in its initializer list.

---

## API

The feature is transparent to existing users. The `AMR3D` constructor's new parameter has a default
value of `true`, so all existing call sites (e.g., `runs/TDEbench/test.cpp`) automatically use the
distributed path without modification. To disable:

```cpp
AMR3D amr(eos, refine, remove, interp, nullptr, nullptr, false);
// or at runtime:
amr.SetDistributeClips(false);
```

## Risks and Assumptions

- **k=1 neighbor sufficiency**: The distributed refine path clips only against immediate neighbors
  (k=1) instead of using BFS. This is geometrically justified because refined cells are created at
  0.25 * width from the parent center, well within the parent's Voronoi neighborhood. The original
  BFS-based path (used when `distribute_clips_ = false`) remains available as a fallback if edge
  cases arise.

- **Communication overhead**: Each `DistributeAndComputeClips` call involves one `MPI_Allgather`
  and three `MPI_Exchange_all_to_all` operations. This overhead is worthwhile only when the load
  imbalance is significant (many idle ranks, few busy ranks with many clips). For uniformly
  distributed AMR, the original sequential path may be faster.

- **Serialization cost**: Packing and unpacking polyhedra and planes into flat double arrays adds
  overhead proportional to the geometric complexity. For typical Voronoi cells (10-20 faces, 5-10
  vertices each), this is ~200-400 doubles per task, which is negligible compared to the `clipCells`
  computation.
