# RICH Monte Carlo and Scaling Changes Since 21efdb53

Comparison: `21efdb53d66c0af6ff2015420aa524ee42a55baa` -> `e87ff55c1e04592859756f63ebc3785cbb20e4e8`.

This note summarizes the committed diff only. The local worktree has additional uncommitted files and edits, but they are not included here.

## 1. MonteCarloManager, Legacy RDMA Path

The old MPI/RDMA `MonteCarloManager` was moved out of the top-level manager directory and preserved as the legacy RDMA implementation:

- Removed: `source/monte/manager/MonteCarloManager.hpp`
- Added: `source/monte/manager/rdma_legacy/MonteCarloManager.hpp`
- Moved/expanded: `source/monte/manager/rdma_legacy/RankHandler.hpp`
- Integrated through: `RDMAMonteCarloManagerLegacy3D` in `source/3D/monte/MonteCarloManager3D.hpp`

Main behavioral changes:

- The manager now takes a `MonteCarloConfig` object instead of fixed compile-time constants. This centralizes initial buffer size, minimal buffer size, reallocation factor, shrink cadence, send-buffer thresholds, amount-manager progress cadence, active-rank scan chunk size, and async reallocation polling cadence in `source/monte/manager/MonteCarloConfig.hpp`.
- The manager records more state for diagnostics and later load balancing: per-cell MC step counts, per-cell beginning particle counts, starting/ending/initial particle counts, pure compute time, and handler memory footprint.
- Transfers are now buffered by destination rank. If a rank handler cannot immediately transfer because the remote buffer needs reallocation, the particles are placed in per-rank send buffers and retried later.
- Send-buffer flushing is adaptive. Large buffers flush on a threshold, small buffers can be held during idle drain, and verify/final drain paths force progress before termination.
- Active-rank scanning is chunked with a cursor and prefetching instead of scanning every neighbor on every loop iteration. This reduces overhead when a rank has many neighbors but only a few active particle queues.
- `AmountManager` progress is throttled by `amountProgressMinCycles` instead of being forced on every loop iteration.
- Completion verification now requires both empty send buffers and no pending async reallocations. This fixes the case where the global particle count reaches zero while particles are still staged in local buffers or blocked behind reallocation metadata.
- Buffer shrinking is more selective. It uses sparse pair negotiation, shrinks mostly non-neighbor handlers, honors a shrink budget via `shrinkPercent`, and avoids a global all-to-all-style synchronization.
- Handler setup creates only missing neighbor handlers and uses the optimized rank-pair synchronization in `RankSync`, avoiding the old broad barrier-heavy pattern.

Legacy `RankHandler` changes:

- `av_length` and `th_length` now share one RMA window (`lengths_storage`), reducing remote metadata windows and synchronization points.
- Transfers sort available indices into contiguous runs and use `RemoteMemoryAgent::PutBatch` where possible. This turns many one-particle writes into fewer larger RDMA writes.
- Transfer timing counters were added for lock wait, reallocation wait, available-index fetches, particle puts, TH puts, length publishing, contiguous/scatter writes, and peak buffer usage.
- Reallocation supports an asynchronous IBV path. A sender can request remote growth, return `false`, and let the manager buffer the particles while metadata is exchanged.
- RMA backend support was broadened through the updated `source/utils/rma` submodule, including signaled-operation control, external memory registration, `PutBatch`, and source `lkey` plumbing.

In short: the legacy manager kept the original `available list` plus `to-handle list` model, but made it less chatty and less blocking by adding batching, adaptive buffering, async resize, lighter progress polling, and better diagnostics.

## 2. New RDMA MonteCarloManager Approach

A separate RDMA manager was introduced:

- `source/monte/manager/rdma/RDMAMonteCarloManager.hpp`
- `source/monte/manager/rdma/RankHandler2.hpp`
- 3D wrapper: `RDMAMonteCarloManager3D`
- Selection aliases in `RadiationMCStep`: `NEW_RDMA` and `NEW_IBV_RDMA`

The new design changes the core communication structure. Instead of keeping remote `available` and `to-handle` index arrays, each ordered rank pair owns a single-producer, single-consumer ring queue:

- `RankHandler2` stores only particle storage plus monotonic `head` and `tail` counters.
- The receiver drains local work with `DetachLocalParticles`, copies the current queue into a local vector, and advances `head`.
- The sender reads the remote counters, writes particles contiguously into the peer ring buffer, handles wraparound with at most two puts, and publishes the transfer by atomically advancing `tail`.
- Normal transfers do not need the old remote TH mutex or available-index bookkeeping.
- Local self-transfer uses the same queue abstraction through local append/drain operations.

The manager keeps the same high-level MC loop shape as the legacy path, but the particle storage semantics are simpler:

- Work is represented as active rank queues plus optional detached local vectors.
- If a particle hits a remote ghost, the manager updates the cell index to the peer-local index and appends it to that peer's send buffer.
- Send buffers are `RegisteredSendBuffer` objects. They cache external source memory registration for IBV, release registration when vector capacity changes, and pass the cached source `lkey` into `RankHandler2::TransferParticles`.
- If the remote ring has no space, the sender requests async reallocation and keeps the particles staged locally until the peer publishes new remote memory metadata.
- `RankHandler2::LocalReallocate` resizes only the local side, preserves queued particles, resets the ring counters, and returns new remote buffer metadata to the requester.

The new approach is aimed at the scaling bottleneck in the old handler: many senders contended with remote locks and remote list updates. The ring queue removes the list-management path and makes the common transfer path a small sequence of remote counter read, particle put, and tail publish.

## 3. Other Optimizations in RICH

Several non-manager changes support the same scaling work:

- Sparse MPI exchange helpers were added in `source/mpi/serialize/mpi_commands.hpp`: `MPI_Exchange_sparse_by_rank`, `MPI_Exchange_by_ranks`, and `MPI_Exchange_all_to_all_sparse`. These avoid posting payload messages for ranks with no data and use direct memory copies for trivially copyable types.
- Voronoi and particle-remap code now use sparse exchange instead of the older broad all-to-all helpers in several hot paths.
- `source/monte/utils/RankSync.cpp` now builds an edge-colored pair schedule for rank-pair operations. This lets RICH create pair communicators without a full global barrier per pair.
- `source/utils/amountManager` now uses a tree-shaped completion and verification protocol. Children flush deltas to parents; rank 0 starts verification when the global count reaches zero; completion is propagated back down the tree.
- `source/monte/utils/GhostMap.hpp` caches ghost-rank maps by tessellation pointer and build generation, avoiding repeated ghost map exchange when the mesh has not changed.
- `source/3D/monte/Voronoi3DMovement.cpp` was refactored to resolve post-exchange particle cells more efficiently: first approximate ownership through the load balancer, then resolve remaining particles through sparse radius expansion and local oct-tree checks. Face cases now test two closest candidate cells.
- `source/3D/tessellation/voronoi/Voronoi3D.cpp` replaced several dense exchange phases with sparse exchanges, short-circuits empty ghost exchanges, compacts ghost requests, and adds memory release points with `shrink_to_fit`.
- `source/utils/balance/weightedBalance3.hpp` adds a one-pass weighted Hilbert boundary calculation when local curve intervals are globally ordered, with a root-gather fallback only when ordering is not sufficient.
- `source/3D/tessellation/loadBalancing/HilbertLoadBalancer.*` now owns the convertor/indexing state more cleanly and uses `getWeightedBorders3` for weighted partition boundaries.
- IMC load-balancing cost calculators were added: `IMCCostCalculator`, `IMCParticleCountCostCalculator`, and `IMCMemoryCostCalculator`. These allow balancing by MC step counters, particle counts, or handler memory pressure instead of raw cell count only.
- `RadiationMCStep` now accepts a `MonteCarloConfig`, exposes manager timing/cost data, and can select legacy RDMA, new RDMA, IBV, MPI RMA, or two-sided P2P manager backends.
- `RadiationIMC` reuses vectors in `preStep`, reserves generated particle storage, improves photon-count rounding, and optionally accumulates per-group time-averaged radiation energy.
- `CombPopulationControl` uses a static RNG and rounded counts, avoiding repeated RNG construction and reducing systematic truncation in population targets.
- New regression and benchmark material was added for Doppler scattering, moving slab MC, Densmore-style MC, Hohlraum strong/weak scaling, and the ball emission benchmark.

## 4. Scaling Issues Addressed

| Scaling issue | Main locations | Efficiency change |
| --- | --- | --- |
| Too many tiny MC transfers | `rdma_legacy/MonteCarloManager.hpp`, `rdma_legacy/RankHandler.hpp`, `MonteCarloConfig.hpp` | Per-rank send buffers batch particles, thresholds avoid small sends, `PutBatch` combines contiguous remote writes, and diagnostics feed adaptive settings. |
| Blocking remote buffer growth | `ReallocationAgent.*`, `rdma_legacy/RankHandler.hpp`, `rdma/RankHandler2.hpp` | IBV reallocation can be requested asynchronously. Senders stage particles locally while peers resize and publish fresh remote memory metadata. |
| Remote lock/list contention | `rdma/RankHandler2.hpp` | The new RDMA manager replaces remote AV/TH list mutation with an SPSC ring queue. Common transfers avoid the old remote TH mutex. |
| O(neighbors) polling in MC loop | `rdma_legacy/MonteCarloManager.hpp`, `rdma/RDMAMonteCarloManager.hpp` | Active-rank scanning is chunked, cursor-based, and prefetches future handler metadata. Active ranks are carried between iterations. |
| Expensive global completion detection | `source/utils/amountManager/*` | Global particle count and verification moved to a tree protocol with batched parent flushes, reducing allreduce pressure. |
| Pair communicator setup at high rank count | `source/monte/utils/RankSync.cpp`, manager `PrepareHandlers` | Pair creation is scheduled through edge coloring and only new neighbor handlers are constructed. |
| Sparse communication expressed as dense all-to-all | `source/mpi/serialize/mpi_commands.hpp`, `Voronoi3D.cpp`, `Voronoi3DMovement.cpp` | Sparse exchange routines send payloads only between ranks with real data, while still using a count exchange to discover peers. |
| MC load imbalance after transport | `MonteCarloManager3D.hpp`, `RadiationMCStep.*`, `IMCCostCalculator.hpp`, `IMCParticleCountCostCalculator.hpp`, `IMCMemoryCostCalculator.hpp`, `weightedBalance3.hpp` | Managers expose particle/step/memory counters, and Hilbert load balancing can use weighted boundaries instead of equal cell counts. |
| Particle remap cost after mesh changes | `source/3D/monte/Voronoi3DMovement.cpp` | Approximate owner routing plus sparse radius expansion reduces the number of ranks queried for unresolved particles. |
| Memory growth in mesh and handlers | `Voronoi3D.cpp`, `MonteCarloConfig.hpp`, manager shrink paths | Voronoi containers are trimmed after rebuilds, handlers shrink old non-neighbor buffers, and memory diagnostics expose the worst ranks. |

Overall, the changes attack scaling from both sides: the MC manager reduces communication pressure during transport, while the mesh/exchange/load-balancing changes reduce remesh overhead and keep the MC work distribution closer to the actual particle workload.
