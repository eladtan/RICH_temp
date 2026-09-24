# Fixed-step RDMA queues and fair CPU receive scheduling — review

Workload: standalone Crooked Pipe, 128 MPI ranks, 8 nodes x 16 ranks, OFI verbs/MSG-RC. Reference command: `20000 2 10 --boundary-photons 100 --manager rdma --rdma-engine ofi --max-steps 120`. Timings cover this 120-cycle segment, not the full 1000 ns physical endpoint. Full launcher wall time includes initialization and shutdown; application cycle time and summed max-rank transport-loop time are reported separately.

The starting workspace was dirty. The baseline was copied from the working source before implementation, not checked out from HEAD. `build/rdma128_20260912/implementation.patch` isolates this task's changes from that snapshot. Build and test logs and binary hashes live in the same directory. No speedup is inferred from source inspection.

## Lifetime and publication invariants

1. `Prepare` runs outside fixed transport. Handler retirement, creation, adaptive growth, reset, and any rebalance-triggered shrinking happen before `BeginTransport`.
2. The first `BeginTransport` barrier closes boundary changes on every rank. Each producer then captures its peer's memory descriptor and verifies zero counters. The second barrier prevents an early producer from writing before a late peer finishes its snapshot.
3. During fixed transport, no allocation, remote descriptor, capacity, or queue epoch can change. `Reset`, `Destroy`, `Reallocate`, `LocalReallocate`, remote metadata updates, local appends to peer rings, and the legacy all-or-nothing transfer API reject use in this phase.
4. Each ring has one remote producer and one local consumer. Only the producer advances tail. Only the consumer advances head. A cached head can underestimate available capacity but cannot authorize overwrite of unread entries. A refresh checks head <= producer tail and occupied <= capacity.
5. A transfer writes only `min(pending, available)` particles, splitting at wrap. Both payload segments complete remotely via the existing `QuiesceTarget` before the tail is published with the existing flushed Put. No visibility primitive was removed or weakened.
6. The source allocation and registration survive until the synchronous transfer returns. The outgoing buffer consumes only the completed prefix. It compacts in the existing allocation, or deregisters before growing. An incomplete prefix remains counted as pending and remains eligible for retry, even below the usual batch threshold.
7. Global completion still includes local particles, deferred histories, incoming queues, queued outgoing particles, and completion accounting. After successful completion, `EndTransport` checks local engine emptiness and executes a communicator barrier before enabling allocation changes again.
8. Next-step demand is exchanged with MPI_Alltoall. Both endpoints compute the same growth requirement. Pair reallocation calls run in increasing peer rank order, giving an acyclic pair ordering. The existing pair-synchronized resize implementation is retained. The demand cap does not shrink an existing larger allocation and the allocator's minimum capacity still applies.
9. Under pressure, sends return instead of entering a resize wait. Every active source gets a compute slice; discovery continues while other sources remain active. No peer must complete a send in order to consume its own receive queue.

## Scheduling and accounting review

- Deduplication prevents a rank discovered during an active pass from being processed twice in that pass.
- Suspended histories remain in the deferred vector with their RNG state, remaining time, and already-applied event updates. Suspension occurs between complete `physics->step` / `ApplyTransportEvent` operations.
- Only finished histories are removed and decremented. A yielded history marks the pass non-idle and remains available to the completion verification pass.
- Debug duplicate detection uses a transport-visit generation because completion verification may invoke the handler twice in one outer iteration. Resuming an unfinished history is intentional.
- The event budget bounds calls to the physics event API. It does not preempt an individual physics call.
- A send age is checked when cooperative flushing executes; it is not a hard real-time deadline.
- Receive rings are bounded, but outgoing/deferred host vectors may grow under sustained pressure. This change does not impose a total host-memory limit.
- Scheduling changes floating-point accumulation and population ordering. Bitwise physics reproducibility is not asserted.

## Validation recorded so far

- Release build of baseline and candidate succeeded with the same toolchain and OFI dependency.
- MPI RMA two-rank protocol test passed: saturation, partial sends, wrap, delayed consumer, 12 alternating resize epochs, forbidden-operation guards, registered-buffer compaction, and return to the legacy path.
- The same test passed across two compute nodes with the actual OFI verbs/MSG-RC provider (Slurm 10176461, exit 0).
- AddressSanitizer/UndefinedBehaviorSanitizer with container assertions passed on the MPI protocol test. Vptr sanitizer is excluded because native particle transport already uses raw polymorphic object storage; no claim is made that this audit fixes that pre-existing representation issue. Leak checking is disabled for the MPI run.
- Full manager smoke passed on two MPI RMA ranks and four native OFI ranks, with 7-entry initial rings and 4-event slices.
- STORM_DEBUG and container assertions passed a 12-step manager smoke with 4-event slices, and a P2P smoke with 1-event slices.
- Slurm 10176462 passed its native gates but failed before running the baseline because compute nodes lack /usr/bin/time. The resubmission uses Bash's built-in timer.
- Slurm 10176463 compares three baseline and three candidate runs in one exclusive allocation. Initial results show a regression; they are not accepted as an optimization win. A subsequent controlled ablation job (10176464) tests scheduling granularity and flush age.

## Limits

These checks provide evidence, not proof that every supported configuration works. GPU, CXI and native IBV backends have not been exercised for this change. The production benchmark uses 120 cycles at a low per-cell photon budget; it cannot establish full-endpoint time-to-accuracy or ensemble equivalence. See the final performance report for measured results and adoption decision.

## Review disposition and final configuration

The initial 8,192-event / 250 us configuration regressed consistently (three runs).
The tested 65,536-event / 5 ms configuration recovered that regression but only
showed a small gain. The final source keeps the previous default behavior; the
Crooked Pipe option `--optimized-rdma` explicitly enables both experimental changes.
All backend visibility fences and flushed tail publication remain unchanged.

The final review added a destruction guard while a fixed step is active, rejected
a growth ceiling below the allocator minimum, and restored the legacy path's
debug destination check on prefix transfers. Release-mode visit-generation work
and disabled-age clock reads were removed. Source buffer registration remains
synchronous throughout this patch.

A 24-cycle four-rank OFI run using STORM_DEBUG, container assertions, 7-entry
initial rings, 4-event slices and the energy ledger passed (Slurm 10176466).
Maximum absolute normalized energy residual was 6.01e-11 in both the baseline
and candidate. This is a conservation check for that case, not full-endpoint
ensemble validation.

Final source and external dependency hashes are recorded in the build directory.
The external dependency snapshot was unchanged during the initial and ablation
comparisons. The final source is rebuilt and tested with explicit opt-in in
Slurm 10176467, including a run with the option disabled.

Final confirmation: Slurm 10176467 completed all five production runs and its
native gates with exit status 0. Slurm 10176468 completed the final opt-in
STORM_DEBUG/native OFI conservation run with exit status 0; maximum absolute
normalized energy residual was 6.01e-11. Final runtime results are in results.md.
