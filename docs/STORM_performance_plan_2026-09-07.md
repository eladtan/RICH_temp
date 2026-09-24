# STORM performance engineering plan

CPU / InfiniBand first. Portable communication and transport throughout.

Prepared 7 September 2026 for RICH and its STORM submodule. This is an implementation specification and review guide, not an implemented change or a measured performance report.

## 1. Recommendation and scope

The strongest opportunity is to reduce the number of serialized control operations surrounding each RDMA particle batch, and to let independent peers make progress concurrently. The current data path is already batched and uses single-producer/single-consumer (SPSC) receive rings. Its present implementation nevertheless locks the remote queue, refreshes its memory descriptor, reads two counters separately, writes the payload, waits for remote visibility, publishes the tail, and unlocks. The network payload can be efficient while this surrounding transaction remains latency-limited.

Start with measurement, fair CPU scheduling, bounded packet dispatch, and backpressure. Then introduce explicit completion ownership and an epoch-based resize protocol before removing the hot-path mutex or caching descriptors. Build asynchronous publication on those foundations. Preserve the existing conservative implementation as a selectable reference until every protocol test and production comparison passes.

The primary target is CPUs on InfiniBand, as requested. OFI native verbs is the principal native-RDMA target in the current high-level factory. MPI RMA, two-sided MPI, other OFI providers, generic CPU physics, and optional GPU execution remain supported. The plan does not assume that an InfiniBand-specific ordering property applies to every provider.

**Planning target:** a combined 15–35% reduction in STORM elapsed time on a communication-heavy CPU/InfiniBand workload is plausible enough to justify the work, provided measurement finds 30–50% of elapsed time exposed to communication and its scheduling consequences. This is a hypothesis, not a promise. On compute-heavy runs the same communication changes may save only 2–10%. A 2x full-application improvement is not defensible without a profile showing a much larger removable bottleneck.

No production simulation, hardware benchmark, or physics test was run during this review. Only source analysis and document generation were authorized and performed. Existing local changes were examined as part of the current source snapshot and were not reverted or edited.

### 1.1 Source snapshot and evidence policy

Workspace root: `/home/maorm/RICH`.

| Repository | HEAD observed during review |
| --- | --- |
| RICH | `2c71b28f49cd40ec46e031e4e896849e40e363c8` |
| STORM, `source/monte` | `55ed49fa60d2c26d05eb70d1c41449364cdf78d1` |
| EasyRMA, `source/utils/rma` | `ff42f8709c1f61067b25dbc4a1e02ec850494326` |
| mpi_utils | `84212d05d2de5c44f5cd4254c8afa67bb65f4c04` |

These repositories have local modifications. HEAD alone does not reproduce the reviewed source. The companion evidence manifest records hashes of the relevant files. Before implementation, compare those hashes, reread changed functions, and revise this plan where behavior has moved. Do not blindly apply line-number-based edits.

Evidence labels used below: **Observed** means present in the inspected source; **Proposed** means a new design; **Estimated** means an unmeasured engineering forecast. Historical comments and older design documents are context, not evidence of present runtime. In particular, older notes describe a mutex-free queue transaction and an atomic tail increment; the current code contains a remote mutex and a tail Put.

### 1.2 What already exists

- Per-destination send buffering and persistent source registration where supported; do not propose adding them from scratch.
- SPSC rings, contiguous payload writes with at most two ring segments, and asynchronous native-provider reallocation.
- Chunked neighbor scanning and tree-based particle completion. Neither feature should be replaced wholesale without evidence.
- A common manager for communication backends; concrete physics templates; cached host transport views and generation-based device geometry caching.
- Counter-based particle RNG for supported radiation physics, CPU OpenMP private energy tallies in the Kokkos path, resident GPU particles/census, a 64-event GPU wave default, and optional overlap of GPU work with communication.

## 2. Runtime estimates and how to interpret them

Every idea has an estimate below. All percentages denote reduction in total STORM elapsed time for the stated regime, including setup, transport, and census. They are not reductions in the named subsystem alone. Zero is always possible if that workload does not exercise the bottleneck; a bad implementation can make runtime worse.

Let `f` be the fraction of baseline elapsed time that a change can actually shorten on the critical path, and `r` the fraction of that cost removed. The first approximation is `saving = f * r`; equivalently, a subsystem speedup `s` gives `saving = f * (1 - 1/s)`. A 20% elapsed-time reduction means a speedup of `1 / 0.8 = 1.25x`, not 1.20x.

For full RICH runtime, let `q` be the measured fraction spent inside STORM. Then the approximate RICH saving is `q * STORM saving`, assuming the other phases do not change. Example: at `q=0.8`, an estimated STORM saving of 5–15% means 4–12% for RICH. At `q=0.3`, it means 1.5–4.5%. Repartitioning changes outside-STORM costs, so measure the whole application for that idea instead of applying this approximation blindly.

Do not add rows. Caching, lock removal, batching, and asynchronous publication attack overlapping control time. Compare each incremental patch against the latest accepted baseline. Even multiplying independent savings is inappropriate when their affected work overlaps. Confidence below concerns the expected benefit before measurement, not the strength of the cited code evidence.

### 2.1 CPU and portable priorities

| ID / idea | Estimated STORM time saved when applicable | Basis and confidence | Order |
| --- | --- | --- | --- |
| M0: trustworthy measurements and backend identity | 0% directly; target measurement overhead below 2% | Enables decisions; no speedup claim | First |
| R1: fair receive scheduling and bounded dispatch age | 1–15%; 0–3% if already balanced | 5–25% exposed delay, recover 20–60%; medium-low | Early |
| R2: partial sends, credits and bounded memory growth | 1–10%; 0–2% with no pressure | 5–20% resize/stall cost, remove 20–50%; medium-low | Early |
| R3: explicit wire packet and reusable host buffers | 0–8%; 3–15% only if copies/bytes dominate | 10–30% packet movement cost, remove 20–50%; low | Early / separate |
| R4: operation tickets and target-scoped completion | 0% for foundation; 0–10% with measured cross-peer wait coupling | 5–20% coarse wait cost, remove 40–50%; low | Foundation |
| R5: epoch resize handshake; remove hot-path mutex | 3–12%; 0–2% if lock overhead is tiny | 10–20% transaction locking cost, remove 30–60%; medium-low | After R4 |
| R6: producer-owned tail and cached receiver credits | 4–15%; 0–3% compute-heavy | 10–25% metadata latency, remove 40–60%; medium-low | After R5 |
| R7: pipeline peers and batch publication | 8–25%; 1–5% compute-heavy | 20–40% exposed network waiting, hide/remove 40–65%; low | After R4–R6 |
| R8: OFI CQ moderation, submission batching, small-write injection | 2–10%; 0–2% for large batches | 5–20% posting/CQ cost, remove 30–50%; low | After R4 / R7 |
| R9: cache capabilities; sparse active-peer bookkeeping | 0–5%; 2–8% with many ranks and sparse activity | 5–10% control CPU cost, remove 40–80%; medium-low | Early, small patch |
| R10: amortize setup, MR and handler lifecycle | 0–5%; 5–15% for setup-heavy short steps | 10–25% setup, remove 40–60%; low | Profile-gated |
| R11: intra-node route and NUMA placement | 0–8%; 5–15% if much traffic is same-node/misplaced | 10–25% affected movement, remove 40–60%; low | Later |
| C1: cost-aware balancing and amortized repartition | 0–15%; 15–30% only with large removable skew | Critical-rank imbalance must be measured; low | Parallel study |
| C2: CPU locality / existing Kokkos OpenMP tuning | 0–20%; regresses if memory pressure rises | 50–80% affected work, improve 5–25%; low | Profile-gated |
| C3: DDMC static/dynamic cache separation | 0–8%; 8–15% setup-heavy | 10–25% repeated preparation, remove 30–60%; medium-low | Profile-gated |
| C4: narrowly specialized exact math | 0–4% | 5–15% affected math, remove 10–30%; low | Small isolated patch |
| C5: faster DDMC angular inversion | 0–3%; 3–10% only if sampler is dominant | Measured sampler fraction required; low | Optional |
| C6: CPU Comb scratch and independent-bin work | 0–8% | 5–20% population cost, remove 10–40%; medium-low | Small first patch |
| A1: DDMC/random-walk policy at fixed accuracy | 0–10% where acceleration already effective; 10–60% only in suitable opaque cases | Different event algorithm; require time-to-accuracy study; low | Separate science track |

The upper endpoints are screening estimates, not numerical guarantees. In particular, the large C1 and A1 ranges are conditional opportunities that cannot be assumed for the user's production cases.

### 2.2 Optional GPU and research ideas

| ID / idea | Estimated STORM time saved | Applicability and confidence |
| --- | --- | --- |
| G1: tune existing wave/overlap/hold policy | 0–15%; 0–25% with poor current tuning | GPU only; CPU benefit 0%; medium-low |
| G2: persistent Comb scratch and parallel exact host ordering | Scratch 0–5%; ordering 0–10%; combined 0–12% | GPU or Kokkos CPU population control; medium-low |
| G3: device Comb ordering without host round trip | 0–20% | Large GPU census, after G2; low |
| G4: repeated-cell tally aggregation and particle locality | Tallies 0–15%; locality 0–15%; combined target 0–20% | GPU contention/bandwidth cases; CPU locality belongs to C2; low |
| X1: dedicated progress thread | 0–10%; allow 5–15% regression in trials | Only if cooperative progress still stalls; low |
| X2: GPUDirect / provider heterogeneous memory | 0–15% in staging-heavy GPU runs | CPU-only benefit 0%; hardware-specific research; very low |
| X3: native IBV credit/batch repair | 0% for default OFI; 0–12% in explicitly exercised IBV path | Correctness prerequisites first; low |

## 3. Current execution path and concrete bottlenecks

### 3.1 Source map

All paths in this document are relative to the workspace root unless explicitly absolute. Line numbers refer to the reviewed working files.

| Evidence | File and starting lines | Meaning |
| --- | --- | --- |
| E1 | `source/monte/manager/MonteCarloManagerFactory.hpp:75` | Auto tries OFI; exceptions fall back to P2P |
| E2 | `source/monte/manager/MonteCarloLifecycle.hpp:369`, `:456` | Progress / HandleAll / flush / count; full verification sweep |
| E3 | `source/monte/manager/MonteCarloTransport.hpp:706`, `:830`, `:1063` | Receive discovery only when active list empty; unlimited history loop; batch counts histories |
| E4 | `source/monte/manager/parallel/RDMASendBufferProtocol.hpp:115` | Threshold or idle flush; entire queued vector attempted |
| E5 | `source/monte/manager/parallel/RankHandler2.hpp:758` | Current synchronous queue transfer transaction |
| E6 | `source/monte/manager/parallel/RankHandler2.hpp:654` | Local resize changes allocation and resets queue indices |
| E7 | `source/utils/rma/OFIContext.cpp:1618`, `:1925` | Per-write completion; context-global completion count |
| E8 | `source/utils/rma/OFIRemoteMemoryAgent.hpp:440` | Flush/quiesce uses fenced read plus global drain |
| E9 | `source/utils/rma/DistributedMutex.cpp:61`, `:103` | Current remote mutex protocol includes communication |
| E10 | `source/monte/manager/parallel/RegisteredSendBuffer.hpp:108`, `:136` | Registration retained until growth/handler change |
| E11 | `source/monte/manager/communication/RDMACommunicationEngine.hpp:194`, `:370` | Pending checks and repeated handler capability scan |
| E12 | `source/utils/mpi_utils/AmountManager.cpp:96`, `:245` | Tree progress and sticky negative verification votes |

### 3.2 One ordinary remote batch today

The sequence below is a source-level dependency chain, not a count of wire packets. Provider completion and mutex helpers may add operations internally.

```text
CPU history crosses rank boundary
  -> CommunicationEngine::Send copies into destination vector
  -> threshold reached, or local idle, or verification drain
  -> TransferParticles checks pending resize
  -> remoteListMutex.Lock
  -> Get descriptor {address, key, count, generation}
  -> Get tail; Get head
  -> if no space: quiesce, unlock, request resize, retry later
  -> Put payload segment 1 [and segment 2 on wrap]
  -> QuiesceTarget(payload): establish remote visibility
  -> Put new tail with flush
  -> remoteListMutex.Unlock
  -> sender clears its registered vector
receiver: observe tail -> copy ring entries -> advance head
```

The tail is already producer-owned; its increment is already an ordinary Put, not a fetch-add. Receiver copying already uses two contiguous insert ranges rather than default-constructing all received objects. Both facts matter when estimating remaining savings.

### 3.3 Measurement limitations in current diagnostics

The current `rma` loop timer does not capture every communication call: `HandleAll` invokes progress and flush inside CPU histories. Those calls fall inside `handle`. GPU callbacks also overlap communication with execution. Do not infer a clean communication fraction from that label alone. Per-component rank maxima can come from different ranks; adding them does not reconstruct the maximum full step.

The current GPU `deviceSeconds` is measured with a host clock around a wave that includes more than kernel execution. Use device event timing or tracing to isolate kernels. `runs/Elad_paper_mach45/run4_profile.sh` in the Mach45 run directory is not a trustworthy performance harness by name alone: the current `--profile` option in that application specifies an analytic initialization profile, not a profiler.

## 4. M0 — establish the evidence before optimization

**Estimate:** 0% direct benefit. Target less than 2% overhead for summary instrumentation; sampled tracing may cost more and must be reported.

Add a versioned performance record per run and summary per step. The initial implementation task should deliver a reproducible baseline and the following measurements, not tune defaults.

1. Record RICH/STORM/EasyRMA revisions plus dirty-file hashes, build flags, compiler, MPI/libfabric versions, actual engine/provider/endpoint mode, node/rank count, rank-to-node mapping, affinity, thread count, NIC choice, mesh/particles/groups, seed policy, physics acceleration switches, and output cadence. Never use `Auto` as the only backend label.
2. Time full STORM step and full application elapsed time. Store per-rank records for selected steps and maximum/minimum/median summaries. Add exclusive phase attribution or use nested trace events; do not sum nested intervals.
3. Instrument `TransferParticles`: mutex wait, descriptor read, tail read, head read, payload posting, payload remote-completion wait, tail publication/wait, resize wait. Count calls, bytes, particles, ring-wrap splits, retries, and failed capacity attempts.
4. Instrument provider operations: Get/Put/atomic count, CQ entries, signaled/unsignaled operations, drains, drained targets, inflight depth, EAGAIN returns, source-staging copies, MR registration/deregistration count/bytes/time, and actual source registration cache hits.
5. Record send occupancy, first-packet enqueue age, oldest pending age, per-peer batch-size distribution, receiver queue occupancy, credits consumed/returned, and peak staged/registered/receive bytes. Use sampled fixed buckets, not an allocation per event.
6. Record CPU physics event classes, events per history, maximum interval between receive discovery/progress calls, work per rank, and tail completion time. A useful tail metric is the interval from the 90th percentile rank's last terminal history to the last rank's last terminal history; record termination-protocol time separately.
7. Keep hot-loop counters rank-local and write outside transport. Do not add a global reduction per packet or per queue poll. Verify timers are inactive when disabled. Cache any new configuration switches at step start.

**Baseline matrix:** single rank; two ranks on one node; two ranks across nodes; 4/16/64 ranks; production scale when available. Compare explicit RDMA+OFI, MPI RMA, and P2P. Add direct IBV as a labeled diagnostic backend, not as an assumed equivalent native payload path: current resizable-agent selection can map IBV payload storage to MPI RMA.

Use at least: a transport-only deterministic hopping workload; a burst/credit-pressure workload; an optically thin cross-rank case; a thick/random-walk or DDMC case; CrookedPipe or Hohlraum; and one moving/rebalanced production case. Fix physics, particle count, seeds, mesh, output and hardware allocation for comparisons. Maintain both strong scaling (fixed problem) and weak scaling (fixed problem per rank).

For steady-state runs, separate warmup/setup from representative later steps. Start with 3 warmups and at least 5 paired measured runs, alternating baseline/candidate order in the same allocation where practical. Report median, variability and paired speedup; increase repetitions when variability is comparable to the claimed improvement. Do not call a 1% change a win with 5% run-to-run noise.

**Gate:** do not start the risky protocol rewrite until operation counts and exclusive timing establish which dependency costs matter. A two-node microbenchmark does not establish production scaling.

## 5. R1 — fair CPU scheduling and bounded packet age

**Estimate:** 1–15% in arrival-delay or straggler cases; 0–3% if queues and ranks are already well balanced. This includes scheduling and dispatch gains together, not two additive estimates.

**Observed:** new neighbors are scanned only when `currentActiveRanks.empty()` (E3). An expensive history can run until terminal despite `localTransportBatchSize`, because that parameter limits complete particles rather than physics events. Progress every 1024 events helps the network but does not schedule another received history. Small outgoing buffers flush on threshold/idle, with no elapsed-time age limit in the current RDMA send protocol (E4).

**Implementation tasks:**

1. Discover a bounded number of receive queues on every manager slice, even while other ranks remain active. Maintain an active-membership bitmap or generation stamp to prevent duplicate queue entries. Round-robin the scan cursor; guarantee eventual coverage of every live peer.
2. Add an event budget for one CPU history slice. Check it only after a complete `physics->step` and `ApplyTransportEvent` sequence. An unfinished local particle is a continuation; retain its exact object, RNG, cell, time and pending radiation state. It is not DONE, removed, a new particle, or a transfer. Do not call the new-particle insertion path to resume it.
3. Add per-peer oldest enqueue time and byte/particle threshold. Dispatch when threshold is reached, a monotonic deadline expires, local work is exhausted, memory pressure applies, or completion verification asks for drain. Clear the timestamp only when no queued suffix remains.
4. Initial sweeps, not production defaults: history event budgets `{64,256,1024}`; send thresholds `{64,256,1024,4096}` particles; maximum age `{10,50,200}` microseconds; local batch `{256,1024,4096}`. Measure clock overhead. Check time once per bounded group of events, not every event.
5. Keep P2P behavior independently measurable; it already has cycle-aged dispatch. Express policy in portable engine-level concepts where useful, but do not force identical underlying buffering implementations.

Bound total events or elapsed time for a `HandleAll` slice as well as events per history. Retain an active-rank round-robin cursor across calls. On exhausting the total budget, preserve remaining work and return non-idle; a busy early rank cannot monopolize every slice. The send-age threshold is a best-effort trigger at completed-event boundaries, not a hard delivery guarantee. Record actual enqueue-to-post delay and threshold overshoot. One expensive physics event can exceed the nominal microsecond threshold.

**Correctness and acceptance:** a hot local queue must not starve a cold remote queue; every live peer must be discovered within a documented number of slices. Verify unchanged existing-particle identities, RNG state and event accounting across yields. For a first patch, resume continuations in a later outer iteration so the debug duplicate-handling check remains meaningful. Preserve the `sent` entry-nudge state. New child identity assignment can depend on processing order: define the required child identity contract rather than promising automatic bitwise equivalence. Generic user physics may use shared RNG or mutable state; expose fair history slicing as a capability/opt-in until that physics is audited. Empty-rank and completion-scan tests must pass. Retain unconditional full verification sweeps. Reject a policy that decreases average flush size so much that posting overhead outweighs its lower latency.

## 6. R2 — partial sends and bounded backpressure

**Estimate:** 1–10% when bursts cause growth/retries; 0–2% when capacity is ample. Potentially larger memory-pressure benefits are not included in the normal range.

**Observed:** flush attempts send the whole destination vector. If the entire batch does not fit, the handler requests growth even when a useful prefix would fit. The receiver releases credits when detached entries have been copied. This supports a bounded producer/consumer policy without changing physics.

**Proposed API:** replace the all-or-nothing boolean only in a new path with `TryTransferResult { acceptedCount, state, ticket }`. States distinguish complete, inflight, need-credit, resize-pending and error. Define whether acceptance transfers ownership; this plan requires engine ownership until local completion. Existing boolean semantics remain the reference path.

1. Send `n = min(queued, knownFree, maxBatch)` when `n>0`. Retain the suffix with an offset or chunk deque; never `erase(begin, begin+n)` for every partial batch. Update pending counts by accepted particles only, and separately count inflight particles until their tickets retire.
2. Before requesting growth, allow the receiver bounded progress and credit return. A full queue from a transient burst is not by itself evidence that the allocation should expand.
3. Set explicit per-peer and per-rank staging/receive memory budgets. Choose receive capacity using measured arrival rate and high-percentile drain delay, then cap it. Retain a bounded headroom rather than unbounded growth.
4. When at budget, return backpressure and continue servicing receives, completions and other peers. Do not busy-wait inside one sender while the peer waits for this rank's progress. Do not drop packets or reduce their weights to make room.
5. If sustained traffic justifies growth, use the resize protocol in R5 once available. Until then retain the current mutex/reallocation protections and make partial sends a separate, conservative patch.

**Tests:** queue capacity 2 or 3; batch larger than capacity; wrap split; both directions full; one slow receiver; alternating hot peers; suffix retained across resize; many retries; pending count never underflows; memory remains below configured limits. Successful termination must include staged suffixes and accepted-but-inflight batches.

**Manager admission must also be specified.** Current `CommunicationEngine::Send` returns void. `ApplyTransportEvent` already prepares the handoff and changes `cellIndex` to the destination's index before calling Send (`source/monte/manager/MonteCarloTransport.hpp:311–357`). If bounded engine admission can reject a packet, retain a dedicated pending handoff `{destination, fullyPreparedParticle}`. Retry only admission: never rerun physics, replay `ApplyTransportEvent`, nudge again, or execute this remote-index packet as local physics. Handoff ownership transfers exactly once. Pending handoffs count as local work; they do not change terminal/global counts. Stop further transport/source production at high-water marks and reserve bounded capacity for a completed event's handoffs/children and for control progress. Audit the maximum output of an atomic physics event; if a generic event can emit an unbounded number of children, a strict total-memory cap requires an additional incremental source/event API and cannot be promised by this patch alone.

## 7. R3 — explicit wire state and reusable receive storage

**Estimate:** 0–8% ordinarily; 3–15% only where packet bytes/copies dominate. Packing can regress runtime if byte savings are small. Safety and ABI clarity are worthwhile even if speedup is zero.

**Observed:** RDMA stores and writes `MCParticle` objects directly. Under MPI, `Particle` derives from `Serializable`; this is not a general trivially-copyable wire format. The existing P2P fast path already packs device hot/cold state for supported release configurations (`source/monte/manager/communication/P2PCommunicationEngine.hpp:17–30`). Debug/tracing and arbitrary coordinate types need explicit handling. Do not assume the existing packed representation is much smaller: measure `sizeof` in the actual build.

1. Define a shared packet trait with `WireType`, `pack`, `unpack`, supported feature flags and schema version. A first release packet may reuse the existing lossless hot/cold representation, moved to a communication-neutral location. Do not couple CPU communication availability to a GPU runtime.
2. Include every transport-semantic field: identity, source identity, local destination cell, position, velocity, remaining time, frequency, current/initial weight, RNG key and counter, step count, radiation transport state, tracking policy, routing fields needed after arrival, and enabled polarization state. Audit `RadiationTransportState` field by field. Preserve debug/history data in a matching schema or select a safe serialized fallback.
3. Require a trivially-copyable packet for byte transport. Do not transmit vtable pointers. For arbitrary `PointT`, provide an explicit point serialization trait or use the existing generic serialization route. Reject unsupported raw layouts rather than silently assuming three doubles.
4. Handshake a packet schema/feature fingerprint before communication. For homogeneous fixed-width packets, validate endianness, type widths and build feature agreement. Cross-architecture canonical serialization is a separate supported path, not a promise made by C++ struct copying.
5. Reuse detached receive-vector capacity across slices. The current detach uses two contiguous inserts; preserve that improvement. Do not process directly in ring slots after publishing head, because the producer is then allowed to overwrite them.
6. Consider borrowed receive spans only later, if copies remain expensive. Such a span is a lease that retains credits until processing finishes and participates in resize quiescence. Estimate for this optional substep is 0–5% and it may instead increase queue pressure; it is not needed for the first packet change.

**Acceptance:** pack/unpack round trips preserve exact semantic fields for release, debug, tracing and polarization cases that are claimed supported. Deterministic rank-hop tests preserve IDs and RNG; unsupported builds take the documented safe route. Compare pack CPU time plus bytes/copies saved. No reduced-precision positions, time, weights or RNG state in this task.

For raw-struct transport, require standard layout as well as trivial copyability and fingerprint `sizeof`, `alignof`, and offsets/sizes of semantic fields and nested layouts. Matching type widths alone does not establish matching padding/offsets across compilers. Alternatively encode a defined byte layout. Do not compare uninitialized padding as semantic data. Reusable receive vectors belong to persistent per-rank manager state or a bounded scratch pool, not a temporary recreated each slice. Release ring credits only after a complete safe copy into owned storage, and respect R2 memory budgets while retaining capacity.

## 8. R4 — explicit operation ownership and completion tickets

**Estimate:** the API foundation alone saves 0%. Scoped waits save an estimated 0–10%; reaching 2–10% requires measured unrelated-peer wait coupling. Today's largely serialized schedule may expose little independently removable coupling. The larger overlap benefit belongs to R7 and must not be counted here again.

**Observed:** OFI completion tracking uses a single outstanding count; `Flush` and `QuiesceTarget` drain the shared context. The current `signaled` argument is ignored for OFI writes. Existing send vectors can be cleared immediately after `TransferParticles` returns because that transaction is synchronous. An asynchronous replacement must change ownership as well as posting.

Do not implement this by changing `flush=true` to `false`. In OFI and IBV `Get` paths, external destinations can use staging; deferred reads do not currently provide a later copy-to-caller contract. Atomic result methods may return shared scratch before an unflushed operation finishes. Neither interface is a safe general asynchronous future.

### 8.1 Proposed contract

The following is design pseudocode, not an existing API. Keep C++17 compatibility: use pointer/count views or the project's existing view type, not unconditionally `std::span`.

```cpp
struct QueueId { Rank peer; uint64_t directionId; };
struct Epoch { uint64_t value; };
struct TransferTicket { uint64_t id; Epoch epoch; };
enum class CompletionLevel { LocalReusable, RemoteVisible };
enum class SubmitState { Accepted, Backpressure, Failed };

SubmitResult SubmitWrite(QueueId, Epoch, RegisteredBufferLease,
                         RemoteRegion, Offset, Count);
ReadTicket SubmitRead(QueueId, Epoch, OwnedReadResult,
                      RemoteRegion, Offset, Count);
CompletionState Test(TransferTicket, CompletionLevel);
void Progress(ProgressBudget);  // bounded, nonblocking
FenceTicket EstablishRemoteVisibility(QueueId, Epoch);
bool HasInflight(QueueId, Epoch);
```

`LocalReusable` means the network will no longer read/write the local operation buffers. `RemoteVisible` means the operation has met the backend's documented target-visibility contract. Neither means the receiver has processed the particle. These are separate facts; don't collapse them into `done`.

Ticket IDs are unique within one owning transport context and cannot be passed between engines. Include/check context identity in real handles. Preserve communicator-relative peer IDs. For the low-level Submit API, `Accepted` transfers ownership of the entire logical request to its ticket; partial internal posting stays ticket-owned and retries internally. `Backpressure` accepts nothing. `Failed` after any accepted network post retains its leases until completion or safe endpoint teardown. R2's higher-level accepted particle prefix remains explicit; do not confuse partial particle admission with retrying an accepted low-level request.

### 8.2 Implementation sequence

1. Introduce a context-owned operation pool with stable addresses and explicit lifetime. Include peer, queue, epoch, logical batch sequence, operation type, requested completion level, registration lease and read-result ownership. Do not allocate an object per physics event.
2. Attach an operation context to submitted OFI work and inspect it on CQ completion. Honor required `FI_CONTEXT`/`FI_CONTEXT2` layouts if the negotiated provider requires them. Keep all operations signaled initially, preserving existing ordering. Do not turn on selective completion in the same patch.
3. Track per-operation completion and a contiguous completed frontier per peer/epoch. A completion for peer B cannot retire a buffer for peer A. Out-of-order completions must not advance a frontier across a missing operation.
4. Separate CQ occupancy, submitted operation count, and signaled completion count. Error CQ entries must mark the associated batch failed and retain memory until safe shutdown. Never decrement unsigned counters blindly.
5. Give read and atomic results persistent registered storage owned until completion and consumption. The result becomes accessible only when the ticket is ready. Do not return references to shared scratch that the next operation can overwrite.
6. Provide a conservative synchronous adapter for MPI RMA, IBV and unsupported OFI capabilities. It may return an already-completed ticket. This preserves portability while asynchronous fast paths are developed independently.
7. Make all engine pending checks include operation tickets, publication work, held registrations and control transactions. Add separate queued/inflight counters. Keep the external particle-count protocol unchanged.
8. Only then replace global drain waits with waiting/testing the particular required ticket. Polling may process all peers, but a wait for A must not require B to finish. Do not hold a synchronous remote mutex while invoking reentrant control callbacks that could try to acquire it again.

**Tests:** delay completions for A while B completes; complete tickets out of order; force staging wrap and send-vector growth; corrupt one completion; submit zero-length and split transfers; force EAGAIN; complete operations during reset/destruction. Assert every lease is returned exactly once and no source allocation changes while the network can still read it.

## 9. R5 — remove per-batch locking through an explicit resize epoch

**Estimate:** 3–12% when the mutex is a material part of transfer latency. Removing only the mutex without removing surrounding reads will not realize the combined RDMA target.

**Observed:** the mutex is not redundant merely because the queue is SPSC. The receiver may resize/replace its memory, change the rkey and reset head/tail. The lock prevents that from happening between the sender's descriptor read and tail publication. `LocalReallocate` takes the corresponding local lock. Deleting these locks without a replacement protocol risks writes into deregistered memory.

### 9.1 Ownership and state

For each directed receive queue, the sender alone reserves/publishes tail during a RUNNING epoch; the receiver alone advances head. The receiver is the authority for allocation/descriptor changes. A generation is a queue epoch, not just a statistic. Every data/control descriptor and asynchronous operation refers to `(queueId, epoch)`.

Sender states: `RUNNING -> PAUSING -> QUIESCED -> INSTALLING -> RUNNING`.
Receiver states: `RUNNING -> PAUSE_REQUESTED -> DRAINING -> REPLACING -> AWAIT_ACK -> RUNNING`.

There is at most one resize transaction per directed queue, identified by a monotonically increasing request ID. Simultaneous requests coalesce under the receiver's transaction. The reverse direction can be independent only for providers supporting noncollective region replacement. Current MPI payload windows, including the direct-IBV/MPI hybrid, use collective window free/create on the pair communicator (`source/utils/rma/MPIRemoteMemoryAgent.hpp:157–225`): pause both directions and replace collectively in identical pair order. Retain this conservative branch until a separately validated window design exists. Duplicate and delayed messages are validated by queue/request/epoch and handled idempotently; an old message cannot overwrite a newer descriptor.

### 9.2 Required handshake

1. Growth can be requested by the sender or initiated by the receiver. The receiver requests a pause for epoch `e`. The sender closes the gate for new reservations/batches; newly generated particles remain staged. PAUSING still permits remaining segments, markers and publication of already accepted batches to finish. After QUIESCED, no retry/callback may submit an old-epoch operation. Continue progress for other peers.
2. The sender completes EVERY already accepted operation referencing epoch `e` or its regions: payload, publication, credit/head reads, atomics, visibility fences and deferred partial-post retries. Writes reach required remote visibility; reads/results reach safe completion. Only then send `QUIESCED(e, requestId, finalPublishedTail)`. No old-epoch callback may post again after this acknowledgement. MPI control-message ordering by itself does not prove earlier RDMA work completed.
3. The receiver waits for that acknowledgement and for all local readers/borrowed leases on the old allocation to finish. Verify that its observed published tail agrees with the final sender frontier before copying. Prevent further old-epoch detach/lease creation during replacement.
4. Preserve the exact unread FIFO suffix, allocate/register the replacement, and install its initial counters, for example `head=0`, `tail=unreadCount`. Send `INSTALL(e+1, requestId, descriptor, capacity, initialHead, initialTail)`.
5. The sender validates the epoch and schema, replaces its descriptor, resets cached credit/tail state to the supplied counters, and acknowledges installation. It resumes only once its new epoch state is complete. Any queued old-epoch submission that was never posted must be retargeted explicitly, not replayed blindly.
6. The receiver retires the old allocation only after the quiescence proof and local-lease release; retaining it until installation acknowledgement is a conservative first policy. Bound the number of retained epochs. Resume ordinary receives/publication under the new epoch.
7. Use the same quiescence contract for shrink, reset, handler retirement, rebalance and destruction. Existing pair barriers may remain for lifecycle safety initially. Removing steady-state locking does not authorize removing retirement barriers.

For a first implementation, freeze receive processing during the short replacement transaction. Optimizing overlap during replacement is unnecessary and increases the proof burden. Recoverable allocation failure requires a NEW prepare/commit/abort replacement primitive: allocate/register replacement storage while the old allocation remains valid, commit after success/peer agreement, or release preparation resources and send `RESUME_OLD_EPOCH(e, requestId)` on failure. Do not use current OFI `LocalResize` for that promise: it deregisters/frees old storage before allocating replacement (`source/utils/rma/OFIRemoteMemoryAgent.hpp:586–604`). All MPI window participants must agree preparation succeeded before entering replacement collectives. If provider MR limits prevent holding both generations, retain a conservative quiesced failure policy; do not promise recovery after destroying the old allocation. Do not publish partial new metadata.

Control messages use the correct engine communicator and a collision-free tag namespace or dedicated duplicated communicator. Respect the MPI tag bound. Keep control progress independent of data credits so a full ring cannot prevent its own recovery.

MPI pair collectives also require liveness coordination across pairs: three ranks entering different pair replacements can form a wait cycle. For the initial MPI/hybrid branch, retain the existing conservative protocol or introduce a globally coordinated resize phase with the tested pair schedule; never enter arbitrary blocking pair collectives directly from independent callbacks. A new distributed pair-arbitration mechanism is a separate protocol requiring its own proof. Test simultaneous growth on a three-or-more-rank cycle as well as a two-rank bidirectional case.

**Required proof:** after `QUIESCED(e)` there can be no future network operation naming epoch `e` from that producer. A delayed request, CQ event, source retry or reverse-direction transaction cannot violate this. The implementer must write this argument before enabling lock removal.

**Tests:** simultaneous bidirectional growth; receiver-initiated shrink; ring wrap with unread entries; delayed INSTALL/ACK; duplicate control message; allocation failure; repeated resize; rebalance creating/removing peers; operation error during pause; old epoch ticket arriving late. No stale-rkey access, loss, duplication, use-after-free or indefinite wait is acceptable.

## 10. R6 — cache producer tail and receiver credits

**Estimate:** 4–15% on latency-sensitive CPU/IB traffic, 0–3% compute-heavy. This estimate overlaps with R5 and R7.

**Prerequisite:** R5 makes descriptors and counter interpretation stable for a RUNNING epoch. Before that, a supposedly cached tail can be invalid after resize. An incremental experiment that retains the lock must still refresh epoch information before trusting cached values; do not treat that as the final no-read fast path.

Maintain sender state `reservedTail`, `publicationSubmittedTail`, `publicationCompletedTail`, `cachedHead`, `capacity`, `epoch`. The submitted frontier covers markers already accepted for submission after their payload is safe; the completed frontier records origin-observed marker completion. A receiver may consume data and return credits before the origin observes that marker completion, so `cachedHead` may exceed `publicationCompletedTail`. For the first version allow one publication batch per peer at a time. Reserve space before posting any payload.

```text
assert cachedHead <= publicationSubmittedTail <= reservedTail
assert publicationCompletedTail <= publicationSubmittedTail
assert reservedTail - cachedHead <= capacity
free = capacity - (reservedTail - cachedHead)
if free == 0:
    request/progress credit refresh; return backpressure
n = min(queuedCount, free, maxBatch)
begin = reservedTail
reservedTail += n
post at begin % capacity, split at ring end if necessary
publish through R7 only after payload visibility is proved
```

The receiver can advance head without telling the sender immediately. A stale head underestimates free space and is safe; a head from a different epoch or from entries still being processed is unsafe. Head advances only through a contiguous FIFO prefix whose ownership has moved out of the ring or whose leases have all been released. Releasing a later lease cannot return credits across an earlier live lease.

1. Eliminate the remote tail read; the sender knows its own reserved/published frontier within the epoch.
2. Cache the descriptor for the epoch. Fetch it only at establishment/installation, not every batch.
3. Initially refresh head with a safe asynchronous read when cached free space is insufficient or a bounded refresh deadline is reached. Later, if measurements justify it, push coalesced head credits into a sender-owned mailbox. Credit-push substep estimate: 0–5% incremental after cached reads; do not count the full row again.
4. Initially push credits using structured engine-control MPI messages `(queueId, epoch, head)`. A later RDMA mailbox requires an explicit coherent publication/reader-validation protocol; a multiword epoch/head struct can tear, and aligned fields alone do not fix that. Publish after local copy/lease release. Validate epoch and monotonicity before increasing cached head. Threshold coalescing must have an idle/full-queue fallback so a final small credit release cannot remain hidden forever. Retain current local queue/payload synchronization hooks.
5. Use separate suitably aligned control fields or messages for independently written state. CPU acquire/release operations are not a substitute for the provider's NIC-to-CPU visibility guarantee. Do not infer payload ordering from polling a volatile tail alone.
6. Define 64-bit counter wrap handling: reject unsafe overflow and rebase only under quiescence. Tests should seed counters near the limit rather than waiting for practical overflow.

**Acceptance:** after warmup, ordinary batches with sufficient cached credits perform no descriptor read and no remote tail read. Head-read frequency scales with actual credit exhaustion/refresh policy, not automatically with every batch. All queue bounds and epoch checks remain enabled in debug testing.

## 11. R7 — asynchronous publication across peers

**Estimate:** 8–25% on communication-heavy CPU/IB cases with 20–40% exposed waiting AND independent work available to overlap; 1–5% compute-heavy. Tickets alone do not produce this benefit. The nonblocking guarantee initially applies to the new OFI path; synchronous portable adapters remain correct but may still block peers and are excluded from that performance expectation. This is the highest-upside engineering item and one of the highest correctness risks.

### 11.1 Conservative first state machine

Use two or more engine-owned registered send chunks per busy peer, with a global memory cap. Start at two chunks and one uncommitted publication batch per peer; raise depth only after measuring. Filling another chunk cannot mutate a chunk held by an inflight ticket.

```text
FILLING -> READY -> PAYLOAD_POSTED -> PAYLOAD_VISIBLE
        -> TAIL_POSTED -> PUBLISHED -> REUSABLE
```

1. Detach a stable outgoing chunk from the fill queue, reserve receive credits, and post at most two payload segments. Keep a descriptor for each accepted segment so EAGAIN resumes at the first unposted operation rather than reposting accepted data.
2. Submit a provider operation establishing payload remote visibility and return to the manager. Poll a ticket instead of blocking. While it is pending, compute a bounded slice or service another peer.
3. When all payload segments are remotely visible, post the new tail/publication marker using persistent storage. The receiver must never observe a tail covering incomplete payload. Wait/test the marker's required completion before marking the publication finished.
4. Recycle source storage once its local-reuse contract is met, while retaining any remaining publication state. A simpler first version holds the entire chunk through publication; optimize earlier reuse only if memory pressure warrants it.
5. Begin the next same-peer publication only when order is guaranteed. Later multiple batches may coexist, but publication must advance a contiguous completed prefix; a completed later payload cannot expose a hole left by an earlier batch.
6. `Pending()` includes payload tickets, markers, retries, chunks, credits/control transactions and receive work. `FlushAll()` must drive all these states; it may require repeated nonblocking calls. An empty fill vector is not an idle engine.

This first version still uses a remote visibility boundary followed by tail publication. It gains overlap without claiming that one network operation can replace both safely. Retain this as a portable reference.

### 11.2 Optional provider publication fast path

Only after the conservative state machine passes: introduce a provider method such as `PublishAfterWrites(batch, marker)` whose documented contract covers payload visibility, marker ordering and local-buffer ownership. A capability-checked OFI implementation may use a suitable fenced publication or remote CQ notification, while MPI RMA can retain separate flushes. The incremental estimate is 0–8% beyond conservative pipelining, contingent on a publication RTT remaining exposed.

Do not assume that ordinary write-after-write message order to different addresses establishes CPU-visible payload before a polled marker. Check negotiated endpoint ordering, operation limits and memory domain. If using remote CQ data, configure the receive CQ, handle its entries independently from transmit completions, bound CQ capacity and provide receive resources where the provider requires them. Notification processing must map safely to a queue/epoch; a notification is not a new particle.

### 11.3 Provider semantic constraints

Libfabric distinguishes local reuse, transport completion, target visibility and ordering. A completion at the sender is not automatically proof of target CPU visibility. Use the documented completion level and a tested publication contract; keep a conservative path when unsupported. See [libfabric completion semantics](https://ofiwg.github.io/libfabric/main/man/fi_cq.3.html).

MPI RMA `Win_flush` completes at origin and target; `Win_flush_local` only establishes origin completion. `Win_sync` addresses local memory synchronization and does not complete network operations. Separate windows/agents require the appropriate flush/sync on each; do not assume a lengths-window flush covers the particle window. See [MPI 4.1 flush and sync](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node331.htm).

For CXI, ordinary transmit acknowledgement does not by itself establish CPU-visible data. Delivery completion has a provider cost. Avoid adding it blindly to every fragment; investigate one correct publication boundary per batch. InfiniBand-specific assumptions still need verification for the negotiated verbs endpoint and memory registration settings. See [CXI provider operation flags](https://ofiwg.github.io/libfabric/main/man/fi_cxi.7.html).

**Acceptance:** independent peers overlap in a trace; one delayed destination cannot prevent ready traffic to another; no use-after-recycle under forced delayed completion; no marker before payload; no gaps on wrap; termination waits for final publication. Run real two-node IB tests, not just a mocked transport.

## 12. R8 — OFI posting and CQ efficiency

**Estimate:** 2–10% when small operations/CQ/posting are expensive; 0–2% with large batches or low packet rates. All substeps share this envelope.

**Observed:** `PostRDMAWrite` discards its `signaled` parameter and requests a completion for every posted fragment. `PutBatch` in the OFI agent loops over entries. The context contains an injection helper, but the ordinary small unregistered Put path stages data. Persistent registration already works for warmed-up native verbs source buffers; do not attribute a fictitious new registration-cache win to this task.

1. After R4, support selective completion with explicit accounting for unsignaled work covered by a later marker. The first implementation retains the provider-approved read/fence publication marker and suppresses only payload CQEs that marker demonstrably covers. A CQE on the last ordinary write alone is not a generic proof of prior completion. Persist whether selective CQ binding actually succeeded; a fallback non-selective binding requires accounting for every ordinary CQE. Sweep moderation intervals `{1,4,16,64}` operations only if the provider's completion/order contract supports the intended coverage. CQ entry count and outstanding operation count are no longer interchangeable.
2. Reserve TX/CQ capacity for a terminating signaled marker and essential control work. Never fill the queue with unsignaled operations and then wait for a completion that cannot be posted. Error handling must still identify affected leases.
3. For ring wrap, use a vector RMA submission if supported by the negotiated local and remote IOV limits. Otherwise retain two writes. Do not confuse one API call with guaranteed one wire packet or one PCIe transaction. Obey provider maximum message sizes rather than only a hard-coded chunk threshold.
4. Experiment with `FI_MORE` across already-ready operations, always ending a bounded posting group correctly. On EAGAIN, retain the first unposted descriptor, progress, and retry without resubmitting accepted operations. The target libfabric version's delayed-start/error semantics must be tested. See [libfabric RMA operations and flags](https://ofiwg.github.io/libfabric/main/man/fi_rma.3.html).
5. Use injection for suitably small control writes when supported and when its source-lifetime/ordering behavior fits the protocol. Injection returning does not retire remote-publication state. Do not require a local CQ completion for an injected operation that produces none.
6. Keep fallback methods for providers lacking vector writes, inject capacity or selective completion. Report which fast paths are enabled. Make capability checks during setup; do not query capabilities per particle.

Selective completion needs indirect proof that earlier operations are finished; the endpoint documentation describes this requirement and the relevant ordering/IOV limits. See [libfabric endpoint attributes](https://ofiwg.github.io/libfabric/main/man/fi_endpoint.3.html). The implementation must use attributes actually negotiated with the installed provider, not assumptions from a newer manual.

**Tests:** last partial group, ring wrap, tiny TX/CQ capacities, EAGAIN before/after a batched post, no further application data to trigger a marker, endpoint error, and teardown with unsignaled work. Verify local buffers are never reused based only on an unrelated later completion.

## 13. R9–R11 — reduce control CPU cost and resource overhead

### 13.1 R9: sparse bookkeeping and cached capabilities

**Estimate:** 0–5%; 2–8% only at large rank count with sparse active peers. This is a good small patch because it need not change wire semantics.

`UsesAsyncReallocation()` traverses the world-sized handler vector to find a capable handler. It is called by progress paths, including nested calls. `Progress()` pumps the RMA context and later calls `MakeRDMAProgress()` again. `Pending()` also walks all handlers. World-sized vectors are used even where actual degree is small (E11).

Maintain a compact live-handler list and a cached native-progress requirement, updated whenever a handler is created, retired or its backend changes. Preserve direct rank lookup if it remains useful; first optimize scans without forcing a hash map into the hot path. Use one clearly defined bounded provider poll per progress phase unless new work was posted and another poll is justified. Do not stop polling a manual-progress receiver just because it has no local outgoing operation.

Measure number of visited handlers, empty polls and actual delivered work. Test ranks with zero peers, one peer and thousands of rank IDs with few peers; handler retirement must remove every list reference. Do not replace full termination verification with sampling. Small safe bookkeeping changes can precede R4; asynchronous pending semantics must be updated again with R7.

### 13.2 R10: amortized setup and bounded registration resources

**Estimate:** 0–5% in long steps; 5–15% for repeated short steps with significant setup. Connection-only savings have no assumed steady-state benefit: if startup is 2% of job time and improves by 20–60%, total savings are only 0.4–1.2%.

Handler preparation, retirement negotiation, synchronization and reset run through `source/monte/manager/parallel/RDMARankHandlerLifecycle.hpp:137–308`. The manager recreates `AmountManager` each step, whose constructor includes a barrier and whose initialization reduces initial counts (`source/monte/manager/MonteCarloLifecycle.hpp:314–318`; `source/utils/mpi_utils/AmountManager.cpp:24–75`). OFI native verbs establishes connections to all other ranks in connected mode (`source/utils/rma/OFIContext.cpp:1326–1355`), even if the eventual mesh-neighbor graph is sparse.

1. Measure lifecycle cost first. Cache topology-dependent neighbor work using a communicator-consistent mesh/topology generation. A rank-local early return cannot skip a collective that other ranks enter; agree globally on whether collective lifecycle work is needed.
2. If significant, make the amount protocol reusable through an explicit reset-after-quiescence method, retaining requests/storage safely between steps. Initial particle counts still require correct global accounting. Do not simply remove its barrier without proving posted receive/reset ordering.
3. Add an age/hysteresis policy for stale handler storage where memory permits, reducing retire/recreate churn. Respect per-rank MR/bytes budgets and quiescence. Existing generation-aware ghost-map caching and missing-neighbor-only creation must remain intact.
4. Persistent registered segments can amortize allocations. Keep handle ownership tied to context/provider and MR lifetime; do not share a registration token across unrelated domains. A bounded source arena is an alternative to proliferating per-peer regions, especially for RDM providers. Measure live MR count and provider resources as well as resident bytes.
5. Sparse verbs connection setup is a separate late experiment. Pass a complete peer graph during setup, preserve bounded connection pairing, and define progress-safe connection establishment for newly appearing peers before removing all-to-all setup. Do not reconnect all peers simultaneously; current code already documents connection-manager pressure. Treat this substep as startup savings using the formula above.

**Tests:** unchanged mesh over many steps; one rank changes adjacency; repeated neighbor disappearance/reappearance; custom/reordered communicators; no-work rank; memory budget pressure; shutdown after failed setup. Log setup separately from steady-state performance.

### 13.3 R11: same-node traffic and NUMA placement

**Estimate:** 0–8% normally; 5–15% only when substantial time is spent in same-node transfers or poor placement. If current MPI/OFI placement is already effective, this idea may contribute nothing.

First benchmark rank/NIC/NUMA placement without changing the transport. Bind CPU memory first-touch and rank execution to the relevant NUMA domain, accounting for the actual NIC topology. Keep total cores and memory fixed. Do not publish a hardware-specific binding command as universal.

If same-node traffic remains expensive, identify node-local peers with an appropriate MPI shared-memory communicator. Compare the existing MPI shared-memory route with a dedicated shared-memory queue. A hybrid engine can route local-node particles through a mature MPI path and remote-node particles through RDMA, retaining one particle ownership and completion protocol. This first experiment has lower complexity than a new process-shared lock-free queue.

A new shared-memory ring needs process-shared synchronization supported by the target platform; standard C++ atomics in arbitrary cross-process storage are not by themselves a portable guarantee. NUMA traffic, false sharing and cache-line alignment must be measured. Do not introduce mandatory node leaders: leader aggregation can bottleneck CPU transport and adds another hop. Compare locality gains against that overhead before considering aggregation.

## 14. CPU transport, physics and population work

### 14.1 C1: balance measured work and amortize migration

**Estimate:** 0–15%; 15–30% only with large, reducible critical-rank skew. This overlaps scheduler-tail improvements. Evaluate full RICH runtime because migration/rebuild cost lies partly outside STORM.

`source/3D/radiation/IMCCostCalculator.hpp:13–30` defaults to `1 + 0.005*cellSteps + 10*beginningParticles`. Step counts treat unlike event costs alike. Preserve the existing weighted balancer and improve its input before adopting a new partitioner.

Record a small set of per-cell event categories: IMC, DDMC, random walk, expensive Compton/multigroup work and boundary/rank handoffs. Fit nonnegative cost coefficients from representative rank compute times; use smoothed historical counts to reduce oscillation. Estimate compute time separately from blocking communication to avoid assigning a slow network path to the wrong cells. Start with current event/particle counters if new instrumentation is too expensive.

If the decomposition interface supports edges, penalize frequently crossed boundaries using packet traffic. A scalar Hilbert cell-weight API cannot express an arbitrary edge-cut objective; do not pretend changing one weight implements a graph partitioner. First try calibrated scalar weights and measure communication after migration. Introduce a graph-aware alternative only as a separately justified design.

Repartition when predicted savings across a conservative future horizon exceed migration, mesh/ghost rebuild and RDMA reconfiguration cost by a safety factor, for example 1.5. Validate the predictor on previous steps before using it online. Preserve particle/global cell identity and RNG state through migration. Test both a moving radiation front and a stationary balanced case where the policy should avoid needless work.

On real GPUs, per-cell device step increments/exports are currently host-accessible-only (`source/monte/gpu/KokkosLocalTransportExecutor.hpp:200–210`, `:904–908`), although rank-wide GPU counts are exported. Restoring device cell counts is a GPU-specific prerequisite to this balancing idea, with 0% direct CPU benefit. Use per-cell counters, aggregate once per step, and verify the sum against device events without double counting host fallbacks. Potential GPU balancing benefit is 0–20%, conditional on skew; counter instrumentation alone may cost time.

### 14.2 C2: tune existing CPU Kokkos execution and locality

**Estimate:** 0–20% at fixed total physical cores. Extra cores are resource scaling, not an optimization speedup; report them separately.

Kokkos's host-accessible path already uses dynamic scheduling and thread-private energy tallies (`source/monte/gpu/KokkosLocalTransportExecutor.hpp:874–887`, `:1144–1164`). It is gated by the confusingly named `STORM_WITH_GPU` build option but can execute on CPU OpenMP when Kokkos is configured accordingly. Make the selected execution space explicit in logs and configuration; preserve generic physics fallback.

Compare pure MPI with ranks-per-node × threads-per-rank combinations at equal total cores. Keep RDMA/MPI calls on the controlling thread and verify the appropriate MPI thread initialization/support; OpenMP use does not automatically require `MPI_THREAD_MULTIPLE`. Tune CPU event caps separately from GPU defaults. Measure eligibility and fallback fraction before attributing speedup to threaded execution.

Private energy tallies consume roughly `threads * 2 * cells * sizeof(double)` before additional state. Record memory/merge cost and NUMA effects. Do not add an OpenMP pragma to generic `HandleAll`: shared physics, tracker, children, send buffers and completion counters are not established thread-safe.

If cache misses dominate, trial stable grouping of queued particles by cell/material or transport mode between slices. Charge grouping time to the candidate. Preserve all particle/RNG/radiation state, and audit generic physics scheduling semantics as in R1. Locality-only estimate is 0–8%, included within this row's 0–20% envelope. Avoid a full AoSoA rewrite until measured layout inefficiency justifies it.

### 14.3 C3: split stable DDMC topology from changing rates

**Estimate:** 0–8%; 8–15% in short steps dominated by preparation. Low benefit when the mesh genuinely changes every step.

`source/monte/radiation/gpu/IMCDeviceExecutor.hpp:45–79` prepares temperatures, DDMC and interface snapshots even for relevant shared CPU-kernel use. `source/monte/radiation/ddmc/AdvanceDDMC.hpp:183–225` flattens face leak information. Geometry uploads and host-view reuse already have caching; extend the missing static/dynamic separation rather than replacing those caches.

Create explicit version keys for geometry/topology, material/opacity, groups and DDMC eligibility/interface policy. Keep face routing/geometry arrays persistent; rebuild them only on their real invalidation events. Update temperatures, leak rates and eligibility whenever their inputs change. Reuse temporary vectors. Never use only mesh generation as the key for a table containing temperature-dependent rates.

Compare against an always-rebuild reference on fixed geometry/changing opacity, moving geometry, repartition, group cutoff changes, DDMC toggles and boundary updates. Measure the bytes recomputed/uploaded and end-to-end step time. Stale tables that happen to pass one static benchmark are unacceptable.

### 14.4 C4: exact math specialization

**Estimate:** 0–4% CPU, 0–6% GPU, possibly zero if the compiler already eliminates the work.

`source/monte/radiation/transport/AdvanceIMC.hpp:315–317` evaluates `Expm1(-dt*decayRate)` and `Expm1(-dt*decayRate*dopplerShift)`. Where `dopplerShift == 1`, a measured/codegen-verified specialization can reuse the first value. Retain the original expression for other cases. Compare stationary and moving-frame cases and tiny/large optical depths.

This is a narrow, reversible change. Do not enable global fast-math, approximate logarithms, reduced precision, or assume particle speed always equals the configured speed of light. The existing spectral lookup deliberately uses linear search; replace it only if the actual CPU profile and supported group sizes justify a different implementation. Such a search change would need its own measured estimate and is not approved by this plan as a presumed improvement.

### 14.5 C5: DDMC angular inversion

**Estimate:** 0–3% ordinarily; 3–10% only when the measured interface sampler consumes a substantial fraction of runtime.

`source/monte/radiation/ddmc/AdvanceDDMC.hpp:565–584` uses 56 bisection iterations to invert `CDF(mu)=mu^2*(0.455+0.545*mu)`. Keep this reference and first measure call rate/cost. A bracket-preserving Newton/bisection hybrid can reduce work without changing the intended distribution.

Define endpoint handling, monotonicity and CDF residual tolerance before implementation. Keep the same number of RNG draws. Compare residuals over a dense uniform and endpoint-concentrated input grid; test distribution moments and DDMC interface physics. Angles may differ by floating-point rounding, so do not claim bitwise trajectories are guaranteed. Avoid unbounded Newton iterations and unexplained interpolation tables.

### 14.6 C6: CPU Comb scratch and independent-bin work

**Estimate:** 0–8%; scratch reuse alone is likely near the low end. Zero if population control is unused.

`source/monte/population/CombPopulationControl.hpp:215–230` allocates weights/indices per bin and applies established identity sorting plus seeded Fisher–Yates. `source/monte/population/CombCore.hpp:330–365` also constructs group scratch.

First reuse per-activation or per-worker vectors and reserve observed maximum bin sizes, retaining exact sort/shuffle semantics. Then, if population time is significant, parallelize independent bins: compute output counts, prefix-sum output offsets, and write disjoint ranges. Never race on output `push_back`, RNG state or child identity assignment. Preserve empty-rank collective participation.

Differential tests compare selected source identities, output weights, initial weights, RNG keys, counts and conservation. Include one enormous bin, many tiny bins, zero-weight bins, spectral grouping and repeated capacity growth. A random-key sort is not a substitute for the existing Fisher–Yates sequence.

### 14.7 A1: time-to-accuracy through acceleration policy

**Estimate:** 0–10% where current random walk/DDMC already works well; 10–60% only for a suitable opaque workload currently spending most time on reducible events. This changes the transport algorithm and belongs to a separate physics validation track.

STORM already includes both random walk and DDMC. Do not implement another accelerator before measuring existing event counts, eligibility, fallback and interface traffic. Study threshold/eligibility choices on representative thick cells, mixed thin/thick interfaces and moving material. More aggressive DDMC can reduce events but alter approximation error or move cost to interfaces.

Hold the error target fixed. For each candidate, compare material/radiation temperature profiles, energy conservation, interface fluxes and uncertainty across independent seeds against the current validated method/reference. Use enough histories and seeds for confidence intervals to distinguish bias from Monte Carlo noise; 10 seeds is a starting pilot, not a universal proof. Do not claim speedup by merely reducing particle count, loosening error tolerances or changing final time.

Adopt only settings that meet predeclared accuracy thresholds and improve total runtime at that accuracy. Keep network protocol experiments separate so a change in event count cannot masquerade as faster RDMA.

## 15. Secondary GPU work and deferred experiments

These are optional after CPU/IB priorities, with independent capability gates and budgets. CPU-only runtime contribution is zero unless a host-accessible Kokkos path is explicitly mentioned.

### 15.1 G1: tune existing GPU policies

**Estimate:** 0–15%, up to 25% only with demonstrably poor initial tuning. Overlap can regress short, imbalanced runs.

The current defaults already include 64 events/wave, minimum launch 1024, a 64-skip hold limit and optional communication overlap (`source/monte/manager/MonteCarloConfig.hpp:43–76`). Begin with a staged sweep of events `{16,32,64,128,256}`, minimum launch `{0,256,1024,4096}` and overlap off/on. Measure kernel duration, active lanes, survivor fraction, remote departure age, copy time and total step.

Do not tune `gpuLaunchSize` as though it is implemented; the config documents it unused. A hard-coded wave population cap exists at `source/monte/gpu/KokkosLocalTransportExecutor.hpp:603–609`; expose a real validated limit only as a separate patch retaining the current default. Time-based urgency can share R1 policy. An adaptive controller may select among measured settings with bounded changes; it must not stall final small batches.

### 15.2 G2 and G3: population scratch and exact ordering

**Estimates:** G2 scratch 0–5%, parallel host ordering 0–10%, combined 0–12%; G3 device ordering 0–20% when host ordering/copies remain a large phase after G2. These ranges overlap.

`source/monte/gpu/CombDeviceActivate.hpp:59–87` allocates many scratch views per activation; `:185–236` copies metadata to host, sorts/shuffles per cell and copies indices back. Start with a persistent scratch owner, geometric capacity growth, reused mirrors and exact logical extents. Explicitly reset counts/weights/cursors; do not rely on fresh allocation zeroing. Preserve collectives even for empty ranks.

Parallel host bins are a lower-risk next step. Only then consider scalable stable device ordering on full `(cell,rank,id)` keys while preserving the existing permutation. Do not pack keys into insufficient bit widths or use a quadratic one-thread-per-cell sort. Compare selected identities, weights, RNG state and counts to the reference. A different random ordering is a population algorithm change requiring a different validation claim.

### 15.3 G4: tally aggregation and layout experiments

**Estimates:** repeated-cell tally aggregation 0–15%; particle locality 0–15%; combined target 0–20% on measured GPU contention/bandwidth cases, not their sum.

Per-event energy deposition appears in `source/monte/radiation/transport/AdvanceIMC.hpp:335–345`; hot/cold particle arrays and survivor compaction already exist (`source/monte/gpu/DeviceParticle.hpp:32–92`; `source/monte/gpu/KokkosLocalTransportExecutor.hpp:1008–1030`). First obtain evidence of atomic contention or excessive compaction/copy traffic.

Aggregate only a narrow set of scalar same-cell energy tallies in a lane-local accumulator initially. Flush on every cell change, census, removal, rank handoff, host fallback, exception/error path and wave cap. Confirm no physics step requires reading the pending shared tally as input. Keep momentum/group tallies unchanged in the first patch. Validate conservation and statistical agreement under changed floating-point addition order.

For locality, trial stable mode/material bins only when population and remaining work repay the permutation cost. Keep hot/cold state permutations aligned and preserve DDMC bypass/pending interface state, RNG and split children. Treat AoSoA as a measured experiment across supported architectures, not a mandatory rewrite.

### 15.4 X1: progress thread

**Estimate:** 0–10% in a verified cooperative-progress bottleneck; allow 5–15% regression during evaluation. Defer until R1/R7 have been tried.

Current contexts contain mutable static scratch, shared counters and registrations. A thread requires a complete thread-safety and MPI thread-level audit, ownership separation, synchronization, and a reserved CPU core. Request only the thread support actually needed and check what MPI provides. Compare at fixed total cores, including the progress core. Preserve the single-threaded path. A thread is not a generic remedy for slow RDMA transactions.

### 15.5 X2: GPUDirect / heterogeneous-memory path

**Estimate:** 0–15% only if host staging consumes a measured material fraction of GPU elapsed time; CPU-only benefit 0%.

Keep this out of the first implementation. It needs supported provider memory registration, GPU/NIC topology, device allocation lifetime, host/device visibility rules, stream synchronization and a fallback for ordinary host buffers. First measure D2H/H2D staging cost after existing overlap. Reuse the ticket/epoch contract; never infer GPU visibility from a host-memory completion guarantee. Validate on every claimed GPU/provider combination before enabling it.

### 15.6 X3: native IBV capacity accounting

**Estimate:** 0% for the default OFI path; 0–12% in an explicitly exercised IBV communication path. Correctness repairs themselves may contribute 0%.

The direct IBV layer counts signaled operations while SQ capacity is consumed by all work requests (`source/utils/rma/IBVContext.cpp:332–350`). Batch posting globally drains before lists (`:459–503`). Track per-QP posted and retired WR sequences, reserve a signaled marker slot, record actual returned QP capacities, and reclaim per-QP credits. Keep checked wide lengths until splitting, then narrow to provider fields. Fix these invariants before attempting a more aggressive IBV pipeline. Keep the current resizable-payload fallback until a separately validated native-region implementation exists.

## 16. Portable design requirements

| Backend / build | Initial implementation rule | Fast-path gate |
| --- | --- | --- |
| CPU generic physics | Keep current semantics; slicing optional until audited | Particle-local continuation and RNG contract |
| CPU Kokkos OpenMP | Use existing executor; MPI on controlling thread | Eligible physics, thread level, bounded private-tally memory |
| OFI verbs MSG / InfiniBand | Primary asynchronous implementation target | Negotiated ordering, CQ/context mode, MR lifetime and real two-node tests |
| OFI RDM / CXI and others | Preserve conservative publication until verified | Provider-specific target visibility/progress; no mandatory verbs assumptions |
| MPI RMA | Conservative ticket adapter is acceptable | Correct window epochs, target flush and local synchronization |
| P2P MPI | Keep working reference and fallback | Shared packet trait only when all required features are represented |
| Direct IBV | Secondary, honestly labeled hybrid if applicable | WR credit/ordering and resizable-region support independently validated |
| Serial | No communication resource overhead required | Local ownership and completion behavior unchanged |
| GPU | Optional execution path; retain host staging fallback | Exact memory-domain and stream-completion contract |

Create a small immutable `TransportCapabilities` record during setup: provider/endpoint identity, local and remote completion options, selective binding actually enabled, remote notification support, IOV/message limits, inject size, progress/thread mode, MR modes and supported resize strategy. Capabilities report guarantees the implementation can actually satisfy, not just bits advertised by a provider.

If a feature is unavailable, choose a safe adapter once and log it. All ranks must agree on protocol/schema before communicating. Do not switch individual peers or ranks to an incompatible protocol halfway through a transfer. Initialization failure must leave all ranks in a consistent fallback or failure state.

Memory registration behavior depends on the negotiated MR mode. Avoid unlimited source registration and do not assume keys are process-global. See [libfabric memory registration](https://ofiwg.github.io/libfabric/main/man/fi_mr.3.html). Providers may require application progress even on a rank without outgoing requests; respect the returned progress and threading model. See [libfabric domain models](https://ofiwg.github.io/libfabric/main/man/fi_domain.3.html).

## 17. Validation contract for the implementing model

### 17.1 Non-negotiable invariants

1. Every particle has one logical owner: queued, reserved/inflight, received/detached, executing, or terminal/census. Temporary physical copies do not create new logical particles.
2. Ownership transfers never increment or decrement the global number of physical histories. New children and true terminal outcomes change it exactly once through the established accounting path.
3. Tail publication cannot expose incomplete payload. Head/credit release cannot expose a slot still read by the consumer. Every index is interpreted in its own epoch.
4. A source buffer and its registration remain valid until the relevant local completion. A target region remains registered until old-epoch quiescence is proved and local leases are released.
5. Same-peer publication advances a contiguous prefix. A later completion cannot skip an earlier incomplete batch. Each accepted post is issued once; retry resumes at the unaccepted operation.
6. Zero local work, zero local outgoing CQEs, or an empty send vector is not proof of global completion. Full receive verification and the amount protocol remain mandatory.
7. Completion verification includes staged suffixes, inflight tickets, unpublished payload, control/resize work, detached CPU continuations, active device work and pending terminal accounting. Count updates created during verification must propagate before a positive vote.
8. Every wait that depends on another rank permits the required data and control progress. Backpressure/control recovery cannot depend on obtaining data-ring space.
9. Rebalance, reset and destruction use the same lifetime rules as resize. No source registration is released while referenced by a ticket; no retired peer remains in active lists.
10. Transport preserves full physics state and precision. Scheduling/algorithm changes declare their reproducibility level explicitly and validate accordingly.

The current verification code at `source/monte/manager/MonteCarloLifecycle.hpp:456–477` deliberately performs a full neighbor sweep and checks pending work. `AmountManager::Verify` retains a negative vote for the duration of an attempt. Do not simplify this into a periodically partial scan: the current `completion_scan_test` covers a resulting nontermination case.

### 17.2 Required focused tests

| Test family | Required scenarios and checks |
| --- | --- |
| Wire contract | Every supported field and build feature round-trips; schema mismatch rejected; no raw vtable transport |
| Ring | Capacity 2/3/5, wrap at every offset, batch larger than ring, zero count, counters near overflow |
| Ordering | Delay payload segment; deliver later completion first; marker must not expose holes; compare full packet checksum |
| Ownership | Poison recycled source immediately after permitted reuse; force vector/arena growth; no overwritten inflight bytes |
| Credits | Slow receiver, final small credit release, zero free space, alternating peers, bounded memory, no starvation |
| Resize | Simultaneous directions, delayed/duplicate control, old epoch completion, unread wrap suffix, allocation failure |
| Progress | Receiver has no outgoing work; long physics history; saturated CQ/SQ; EAGAIN; no future data to force final marker |
| Completion | Empty ranks, delayed last particle, partial scan, pending resize/publication, spawned children, yielded history |
| Lifecycle | Repeated steps, moving mesh, changing peers, shrink, teardown after error, split/reordered communicator |
| Physics | Conservation, RNG/identity where applicable, IMC/DDMC boundaries, moving slab, polarization/tracing if supported |

Use a deterministic fake backend to force interleavings and provider errors, then real MPI/OFI multi-rank tests to validate actual memory visibility. A mock cannot prove NIC ordering; a successful hardware run cannot exhaust all interleavings. Timeouts should report queue/epoch/ticket/frontier state so a hang can be diagnosed.

Existing test targets are in `source/monte/CMakeLists.txt:393–429`: `storm_counter_rng_test`, `storm_completion_scan_test`, `storm_p2p_packet_test`, `storm_host_transport_views_test`, `storm_portable_physics_test`, `storm_slab_transport_test`, `storm_ddmc_interface_test`, and optionally `storm_gpu_transport_overlap_test`. Extend relevant tests and add protocol-specific ones; none of the existing physics tests alone proves the new RDMA state machine.

### 17.3 Practical benchmark instructions

Use the existing build's dependency locations and toolchain rather than inventing a configure command that assumes standalone dependencies are installed. Build a separate candidate directory and retain the baseline executable. Do not change the user's build artifacts or submit large production jobs merely to test an unreviewed patch.

On an authorized compute allocation, a standalone test flow is:

```sh
# STORM_BUILD_DIR must name an existing correctly configured MPI build.
cmake --build "$STORM_BUILD_DIR" --target storm_completion_scan_test
ctest --test-dir "$STORM_BUILD_DIR" -N
ctest --test-dir "$STORM_BUILD_DIR" --output-on-failure \
  -R 'storm_(completion_scan|p2p_packet|counter_rng|portable_physics|ddmc_interface)_test'
```

The existing CTest completion and P2P tests already invoke MPI. Do not wrap the whole `ctest` command in another `mpirun`. Observe the site's launcher/allocation requirements. GPU tests require a suitable separately configured build.

For CrookedPipe, the reviewed CLI supports `--manager rdma --rdma-engine ofi`, `--manager p2p`, and `--manager rdma --rdma-engine mpi`, plus `--max-steps`, `--snapshot-interval`, and transport tuning flags (`source/monte/examples/crooked_pipe/main.cpp:80–99`). Use identical problem arguments and output policy for each comparison. Obtain actual executable location from the configured build; do not guess a legacy executable/manager alias. Record the startup backend log for every run.

### 17.4 Adoption gates

- Correctness failures, unexplained count changes, stale-key errors, deadlocks or unsupported-feature regressions block adoption regardless of speed.
- For a default-changing optimization, target at least 5% paired improvement on two relevant CPU/IB workloads or a clearly justified larger production win. Demonstrate no reproducible regression greater than 3% on the agreed protected suite, accounting for noise. These are proposed engineering gates, not results already achieved.
- A change that helps one regime but hurts another can remain an explicit option with a documented selection rule. Do not make it default just because a microbenchmark improved.
- Report total elapsed time, bytes/operations and memory together. Higher bandwidth with worse timestep time is not success. Report the full distribution and critical-rank behavior, not only mean rank throughput.
- Revalidate physics at the declared reproducibility level. Exact state transport should preserve field values; altered floating-point ordering may need tolerance/conservation checks; changed sampling/acceleration needs ensemble accuracy validation.

## 18. Implementation work packages and review order

Use small reviewable changes with one behavioral objective. Suggested effort ranges below are focused engineering time including tests and review, excluding cluster queue delays; they are not a delivery promise for an automated model.

| Package | Content | Dependencies | Effort | Exit deliverable |
| --- | --- | --- | --- | --- |
| W0 | M0 baseline, manifest, backend identity and exclusive metrics | None | 3–5 days | Reproducible profiles and ranked bottlenecks |
| W1 | R9 bookkeeping; R1 discovery/age, each separately | W0 | 2–5 days | Queue-latency result, correctness and no-regression data |
| W2 | R2 partial sends and bounded memory | W0/W1 | 3–6 days | Tiny-ring and pressure tests, bounded memory evidence |
| W3 | R3 shared packet trait and buffer reuse | W0 | 3–7 days | Field/schema tests and pack/bytes timings |
| W4 | R4 tickets with existing synchronous behavior first | W0 | 5–10 days | Ownership tests; all-CQE reference implementation |
| W5 | R5 epoch protocol, initially retaining conservative publication | W4 | 7–12 days | Written lifetime proof; forced resize/interleaving tests |
| W6 | R6 cached descriptor/tail/head credits | W5 | 3–6 days | Zero ordinary descriptor/tail reads after warmup |
| W7 | R7 cooperative cross-peer pipeline | W4–W6 | 5–10 days | Concurrent-peer traces and safe publication proof |
| W8 | R8 provider optimizations, one at a time | W4/W7 | 4–8 days | CQ/posting reduction and target-provider comparisons |
| W9 | Measured subset of R10/R11/C1–C6 | W0; relevant earlier package | 2–8 days each | Whole-application gain without physics regression |
| W10 | GPU or research X/G items | Relevant foundations | Separate estimate | Capability-specific evidence; optional path |

Do not treat all packages as mandatory. W0 may show that C6 or C3 beats an expensive provider rewrite for a particular case. The requested RDMA improvement remains the priority, but evidence should decide which portion of that work gets implemented first.

For each package the implementing model must return: exact changed files/functions; chosen invariants; baseline/candidate commands and source hashes; correctness results; performance results with variability; memory/resource changes; remaining unsupported cases; and whether the default changes. The reviewer must inspect those artifacts, not accept a prose assertion that tests passed.

## 19. Critique guide: where this plan can be wrong

1. **The bottleneck may be physics or imbalance.** The protocol has visible overhead, but source inspection does not establish its fraction of production time. If control/communication exposure is below 10%, expensive RDMA work cannot deliver the headline range by itself.
2. **New complexity may cost more than it hides.** Small populations may lose to extra state machines, packing, polls and partially filled sends. Keep a simple path and measured thresholds.
3. **The resize design is a protocol change.** Epoch fields alone do not make memory reclamation safe. The essential fact is that all old posts are completed and no future old-epoch post can occur. Ask the implementer to identify the exact line establishing each fact.
4. **A completion optimization may accidentally change visibility.** Ask which buffers each CQ entry retires, why the receiver sees payload first, how a wrapped two-segment payload is covered, and what changes on MPI RMA or CXI.
5. **The estimates overlap.** Lock removal, cached metadata and pipelining share one latency chain. Demand incremental measurements. Recompute the combined projection from the remaining profile after each accepted patch.
6. **Portability can silently narrow.** Check arbitrary point types, tracing, polarization, communicator rank mapping, older libfabric capability fallback, serial builds and generic physics. Passing release Vector3D on one IB cluster is insufficient for a library-wide claim.
7. **Scheduling may alter reproducibility.** Particle-local RNG does not settle generic shared physics state, child ID assignment or population ordering. Require a written reproducibility contract before asserting identical histories.
8. **Good averages can hide a worse tail.** Check p95/p99 queue age, maximum rank step time, final drain and rebalance overhead. Do not approve based only on average message latency or rank compute time.
9. **Baseline may drift.** This review includes dirty working files. Compare the manifest and rerun the baseline on the actual implementation starting point; do not compare different physics revisions as a transport experiment.

The reviewer should reject a patch that removes a fence, shortens a wait, recycles a buffer, skips a collective, or advances a counter without identifying the replacement guarantee. For every proposed optimization, the strongest criticism is a concrete counterexample execution or a measured regression, not a preference for a different style.

## 20. Recommended first decision

Start with W0 and the small scheduling/backpressure work. If the profile confirms per-batch metadata and visibility waits dominate, proceed through tickets, epoch-safe resize, cached credits and asynchronous publication in that order. On the requested CPU/InfiniBand target, that sequence has the clearest path to a substantial improvement while keeping STORM's portability and scientific behavior reviewable.

The 15–35% combined STORM target remains conditional. Replace it with a measured forecast after W0: identify the actual exposed fraction, estimate the removable part of each remaining dependency, convert it to whole-RICH savings, and preserve uncertainty. That forecast is the acceptance baseline for implementation.
