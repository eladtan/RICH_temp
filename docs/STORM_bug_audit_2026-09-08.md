# STORM correctness audit and repair specification

8 September 2026. RICH working tree, including local modifications. Read-only production-code audit. CPU / InfiniBand first, with portable MPI, OFI, verbs and optional GPU implications.

## 1. Findings and priorities

This report identifies ten concrete correctness defects or incomplete API behaviors in STORM and its EasyRMA dependency. Three have focused executable witnesses against current headers: mixed particle histories, invalid Fleck factors producing NaN energy, and unfinished diagnostic communication. The remaining findings are established by source inspection and explicit failure conditions; they are not claimed as observed production failures. Native InfiniBand and device-GPU tests were not available in this environment.

The most urgent current-path repairs are B01, B02 and B03. B01 reads an uninitialized completion-accounting field during first host source insertion. B02 can leave diagnostic messages unmatched at shutdown; depending on MPI buffering, this means stale messages or a hang. B03 accepts a NaN Fleck factor and can return success after contaminating particle and material energy with NaNs. B05 and B06 have high consequences when their affected APIs are used, but this audit did not establish that the reviewed STORM transfer path invokes those defective modes.

| Finding | Severity and exposure | Required action |
| --- | --- | --- |
| B01: Uninitialized completion delta | High; first nonempty host insertion | Initialize the field; preserve the step-accounting boundary |
| B02: Unmatched diagnostic sends | High; MPI shutdown timing and buffering | Give diagnostics an explicit receive-and-close protocol |
| B03: NaN Fleck factor accepted | High; invalid opacity input to shared IMC kernel | Reject nonfinite Fleck factors before state/tally mutation |
| B04: Deferred reads lose copy-back | Medium; native OFI/verbs external Get(false) | Track destination completion or reject this mode |
| B05: Atomic results used before completion | High when invoked; deferred atomic API | Complete value-return operations or redesign around owned tickets |
| B06: Subword native atomics touch a whole word | High when invoked; atomics narrower than 64 bits | Restrict supported types or implement correct typed operations |
| B07: Tracker merges distinct particles | Medium; tracking rank-local IDs after migration | Use origin rank plus particle ID everywhere |
| B08: MPI byte counts narrow to int | Medium; individual transfers above INT_MAX bytes | Check arithmetic and chunk large transfers |
| B09: Device work counters omitted | Medium; GPU memory inaccessible to host | Accumulate and export per-cell device work |
| B10: Compute-time API always returns zero | Low; callers of timing API | Implement a defined metric or report it as unavailable |

Severity describes the consequence under the stated trigger. It is not a probability estimate. High means potential hang, undefined behavior, memory corruption, or invalid scientific output. Medium means wrong diagnostics, missing data, or a bounded configuration-dependent correctness failure. Low means an observability defect with no demonstrated effect on physics. No critical/P0 defect was established; this is not a claim that none exists.

Do not describe all ten as existing radiation-answer errors. B07, B09 and B10 affect observability or load-balancing inputs. B04–B06 are dependency API defects with narrower exposure than the main application. B08 requires a sufficiently large individual transfer, not merely a large simulation.

## 2. Scope, evidence and implementation rules

### What was reviewed

The review follows manager construction, pre-step source insertion, local and remote particle transport, completion accounting, diagnostic communication, particle tracking, the shared IMC transport kernel, device work accounting, and the MPI/OFI/verbs RemoteMemoryAgent implementations. EasyRMA findings are included because STORM relies on that layer and because the proposed performance work would increase asynchronous API use. This is not a complete verification of all physical models, all compiler configurations, every mesh interface, or every provider.

Locations use repository-relative paths from `/home/maorm/RICH`. Line numbers identify this working-tree snapshot. Use the accompanying SHA-256 manifest to detect changes before implementing. Repository commits alone are insufficient because the reviewed files contain local modifications. Search by symbol if a line moves; recheck the complete surrounding control flow.

### Evidence levels

**Executed witness** means a small program compiled against the reviewed header demonstrated the stated behavior. It does not mean a complete RICH radiation benchmark failed. **Source-confirmed** means the control flow or arithmetic establishes the defect for the specified input without relying on a particular measured run. **Unresolved concern** means a risk needs more evidence and must not be implemented as a confirmed fix on this report's authority alone.

The diagnostic hang witness deliberately substitutes MPI_Issend for MPI_Isend in its own translation unit to force completion to require a matching receive. Production headers remain unchanged. A separate witness uses ordinary MPI_Isend and observes an unreceived message after FinishCounters. This distinction matters: the forced test proves a missing protocol dependency under a stronger send-completion requirement, not that the configured production MPI invariably hangs on six counters.

Tests ran locally with Open MPI and shared-memory/TCP transport options. The environment's initial default attempted unavailable InfiniBand transports, so the recorded clean MPI runs explicitly select ob1 and self/vader/tcp. There are no native RDMA or CUDA/HIP execution claims. MemorySanitizer was not used; B01 is a source-level undefined-behavior finding.

### Rules for the implementing model

Make one logical correction at a time, with an independent regression test. Keep this audit separate from the performance refactor. Do not remove lock, completion, remote-visibility, or resize-ordering operations merely because an API name sounds synchronous. Do not replace a reproducible failure with a different silent fallback. Tests must assert observable behavior, not the exact implementation proposed here.

For every patch, record the triggering input, expected result, backend and build flags, before/after test results, and any changed API contract. Preserve particle conservation, energy conservation where applicable, origin identity and completion accounting. An invalid-input test must assert the reported error and that energy tallies remain finite and unchanged. A liveness test must include a timeout and evidence of where the ranks stopped.

Repair priorities are separate from the number of lines changed. Start with B01–B03; correct B04–B06 before using deferred reads or atomics in the performance work. B07–B10 can be independent fixes. If an affected API is deliberately unsupported, reject it explicitly before posting network work, document the restriction consistently, and test the rejection.

## 3. B01 — Completion delta is read before initialization

**Severity:** High. **Evidence:** Source-confirmed. **Exposure:** A newly constructed manager that inserts a nonempty host particle batch before its first step-counter reset. This includes the ordinary host preStep source path; it is not an RDMA-only issue.

### Location and proof

- `source/monte/manager/MonteCarloManager.hpp:240`: `CompletionCounter localDecrementAmount;` has no member initializer. CompletionCounter is a signed long long alias.
- `source/monte/manager/MonteCarloTransport.hpp:5`: the constructor initializes other fields but not this one.
- `source/monte/manager/MonteCarloTransport.hpp:81`: AddParticles performs a read-modify-write subtraction from localDecrementAmount.
- `source/monte/manager/MonteCarloLifecycle.hpp:303`: the CPU source path calls AddParticles before the reset at line 311. The nonempty-batch guard inside AddParticles only avoids the defect for an empty batch.

The sequence is construction, source generation, AddParticles, and only then assignment of zero to the completion delta. The first subtraction therefore evaluates an indeterminate signed integer. The later reset does not undo C++ undefined behavior. On many runs the reset may mask any visible numerical symptom; this report does not claim to have measured missing particles from this defect.

### Repair specification

Give localDecrementAmount a zero member initializer so that every constructor establishes its invariant. Keep the distinction between particles present when completion accounting is initialized and particles created during an active transport step. The current startingParticleNum already includes initialParticlesNum plus preStepParticlesNum; pre-step source insertion must not also be counted as a dynamic addition after that initialization.

Document the invariant: before active transport begins, the completion baseline contains all starting particles and the local delta is zero; during transport, the delta records completed histories minus newly created histories not already included in that baseline. Initializing the member is the smallest repair. Separating initial insertion from live insertion is an optional later clarity improvement, not a prerequisite for this patch.

### Regression and review gates

Construct a fresh manager, generate one nonempty host source batch, run a step to completion, then run a second step with both census and new source particles. Assert the expected initial and completion counts and no duplicate source accounting. Include a zero-source case and a case where transport creates additional histories. Use an uninitialized-read sanitizer where the toolchain permits it; a normal successful run alone cannot demonstrate absence of undefined behavior.

The reviewer must trace the baseline and delta through the entire step and reject a patch that simply moves the reset earlier but lets pre-step subtraction survive into live accounting. Expected runtime effect is negligible; this is a correctness initialization, not a speedup opportunity.

## 4. B02 — Diagnostic communication has no complete shutdown protocol

**Severity:** High for a hang; stale diagnostics are a secondary medium consequence. **Evidence:** Source-confirmed plus two executed witnesses. **Exposure:** MPI-enabled engines publishing counters while ranks finish transport at different times.

### Location and failure schedule

`source/monte/manager/communication/CommunicationEngine.hpp:134` implements PublishCounters. Rank zero receives available tag-9941 messages only inside that method, at lines 150–156. Other ranks initiate MPI_Isend at lines 172–173 after a five-second publication interval. FinishCounters at line 195 only waits for a local outstanding send; it does not make rank zero receive remaining messages. `source/monte/manager/MonteCarloLifecycle.hpp:496` calls FinishCounters after leaving the transport loop.

A permitted ordering is: rank zero performs its final diagnostic poll; a worker then publishes; rank zero leaves the loop and never calls the diagnostic receiver again; the worker waits in FinishCounters. Send completion is not guaranteed merely because the payload is small. If completion needs a receive, the worker can wait indefinitely. If the message is buffered and the send completes, the message can remain unmatched. ResetCounterSnapshots at line 129 clears local snapshots and the timer, but neither drains prior messages nor supplies a step identifier. The next step can consume an old snapshot as if it were current.

MPI specifies that send completion makes the send buffer reusable and does not imply that the message has been received. Its progress guarantee for a pending send is conditioned on the corresponding receive being started. These semantics support the two failure branches above. [MPI communication completion](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node74.htm), [MPI nonblocking progress](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node75.htm).

### Executed evidence

The six-counter forced-synchronous witness printed that rank zero finished FinishCounters while rank one entered it and never returned. An eight-second timeout ended the job with code 124. A control with rank zero performing the missing receive allowed both ranks to finish. A second build, with ordinary MPI_Isend and no substitution, allowed both ranks to finish but found an unreceived snapshot afterward. These are small communication-engine tests, not full-manager timing measurements.

### Repair specification

Implement a finite close protocol. A concrete portable design is one ordered diagnostic stream per worker, including snapshot messages and a final close marker on the same communicator/tag. At shutdown each worker first completes its last snapshot while rank zero continues receiving, then sends its close marker. Rank zero services receives until it has seen a close marker from every worker, and workers complete their close sends before returning. Ensure no worker publishes another snapshot once closing starts. The root can then leave the receiver loop without stranding messages. A final collective may follow this close phase if exact final totals are needed.

A fixed message envelope should contain a message kind, step/epoch, sequence number and counter payload. Using the same ordered stream avoids a marker overtaking a previous snapshot through a separately matched control channel. Validate source rank and count. Keep buffer storage alive until its request completes. A dedicated duplicated diagnostic communicator is an option if other users can collide with tag 9941, but its construction and destruction must also be collective and explicit.

Do not fix this with MPI_Request_free, an extra barrier alone, or a single final Iprobe. None establishes receipt of all outstanding snapshots. Do not have workers block sending close markers while root has already entered a blocking collective that stops diagnostic receives.

### Regression and review gates

Exercise zero snapshots, a last snapshot immediately before close, root finishing first, workers finishing first, one rank, two ranks, and several consecutive steps. Repeat with forced synchronous sends. Require every rank to finish under timeout and no old diagnostic message to remain after close. Change the number of counters between separately initialized steps to expose stale payloads. Check that early-finished ranks continue the close protocol correctly.

The fix adds bounded end-of-step communication; its runtime cost must be measured rather than advertised as a speedup. Eliminating an occasional hang has a much larger operational benefit than any microsecond estimate. Preserve the five-second diagnostic cadence independently of particle completion.

## 5. B03 — NaN Fleck factors can silently contaminate energy

**Severity:** High under invalid input, because the kernel reports success after producing invalid scientific state. **Evidence:** Executed witness against the shared kernel. **Exposure:** Grey or spectral opacity evaluation returns a NaN Fleck factor. This does not establish that normal finite production inputs spontaneously generate NaNs.

### Location and proof

`source/monte/radiation/transport/AdvanceIMC.hpp:280` extracts opacityState.fleck. The validation at line 281 checks finiteness of absorption and scattering, but only compares fleck against zero and one. Both ordered comparisons are false for a NaN. The subsequent decay rate at line 315 uses fleck, and the exponential attenuation and material-energy accumulation propagate NaN values.

The shared kernel is reached by host IMCTransportProcess paths as well as GreyIMCKernel, so the missing check is not isolated to GPUs. The ordinary host call sites include `source/monte/radiation/transport/IMCTransportProcess.hpp:469` and the policy-based call at line 501.

### Executed witness

The audit copied the small valid slab fixture from slab_transport_test.cpp into a standalone test and replaced only the Fleck factor with quiet_NaN. The test calls the real AdvanceIMC header. It returned:

```
error=0 weight=nan deposited=-nan integrated=1.8 events=1
```

Here error=0 means the wrapper observed no TransportError. A valid-looking integrated-energy number coexists with invalid surviving and deposited energy, making a partial diagnostic check insufficient. This witness uses normal floating-point compilation without fast-math.

### Repair specification

Include `not IsFinite(fleck)` in the opacity rejection condition and return TransportError::InvalidOpacity before updating particle time, position, weight, RNG state for the event, or energy tallies. The existing validation point already precedes the attenuation and tally updates. Keep the permitted finite interval inclusive: zero and one are valid limiting values. Do not replace NaN by zero, one, or a clamped value; those substitutions change the model while hiding the input failure.

Trace the host and device error-handling paths to ensure InvalidOpacity is reported to the manager or caller with useful cell/particle context and cannot be treated as normal census completion. Independently inspect where the Fleck factor was produced if a real workload hits this error; a guard does not correct upstream thermodynamics.

### Regression and review gates

Parameterize valid values 0, 0.5 and 1 and invalid values NaN, positive infinity, negative infinity, a negative finite value and a value above one. For invalid values assert InvalidOpacity and unchanged particle weight, time, position and energy tallies. Keep absorption/scattering finite in these tests so the Fleck check itself is exercised. Run host and actual device versions where supported.

Inspect production floating-point flags: builds that assume all values are finite may invalidate ordinary NaN checks. If such flags are used, establish validation in a compilation context that preserves the promised checks. Do not claim the normal-build witness covers every fast-math configuration. Expected runtime cost is one extra scalar finiteness check per opacity validation; no measurable overhead has been established.

## 6. B04 — Deferred external RDMA reads never copy into the caller's buffer

**Severity:** Medium in current prioritization; wrong metadata could become high impact if callers use this mode for protocol decisions. **Evidence:** Source-confirmed. **Exposure:** A remote Get with flush=false and a destination outside the agent's registered local buffer, on native OFI or verbs.

### Location and proof

- `source/utils/rma/OFIRemoteMemoryAgent.hpp:318`: Get stages external destinations in registered memory at line 336 and posts the read at line 340. The only copy from staging to result is inside `if(flush)`, at lines 343–350.
- `source/utils/rma/IBVRemoteMemoryAgent.hpp:311`: the equivalent external staging branch has the same copy-back condition at lines 336–343.
- OFI Flush at line 440 and verbs Flush at line 417 drain communication and reset staging, but retain no association between a pending read and its original caller destination.

The sequence `Get(&value, 1, peer, offset, false); Flush(peer);` therefore does not update value from the staged read. Waiting longer cannot recover a destination pointer that was never retained. Self-target operations and reads directly into registered agent memory take different paths and do not demonstrate this particular defect.

The reviewed active ring metadata reads in RankHandler2.hpp:333 and :795–796 use the default synchronous mode. This report does not claim they currently receive stale values from B04. It does mean that changing those calls to false as a latency optimization would be incorrect.

### Repair specification

The smallest defensible repair is to reject external deferred Get before posting it, or explicitly make it complete synchronously and document that contract. A true asynchronous repair requires pending-read records owning a staging allocation, element count, target, user destination and completion identity. Once that operation has completed locally, copy exactly its returned elements into its destination before reporting the read as complete. Do not reset or reuse staging before that copy.

Define the lifetime rule: the caller's destination must remain valid until the returned ticket is completed, or until the documented Flush that completes that read returns. Decide what Flush(peer) does to operations for other peers; it must never discard their pending copy-back work. QuiesceTarget and destruction must honor the same ownership rules. Partial posting failures also need deterministic cleanup.

### Regression and review gates

Read known nonzero values into an external array prefilled with a different sentinel. Check synchronous Get and Get(false)+Flush, two pending reads to different destinations, repeated reads to the same destination with defined ordering, and reads spanning staging growth. Exercise two peers and verify flushing one cannot lose the other's result. Test self-target separately as a control; it must not be the only test.

Run native OFI and native verbs on real remote ranks. An MPI-only pass does not cover the staging branch. No native execution was possible here. Correct asynchronous ownership may add bookkeeping; forcing synchronous completion can reduce overlap, so this fix must precede performance measurement.

## 7. B05 — Deferred atomics expose results before they are ready

**Severity:** High when the affected API mode is used. **Evidence:** Source-confirmed, including a result-buffer lifetime defect in MPI FetchAndAdd. **Exposure:** Remote FetchAndAdd with flush=false; native CAS with flush=false has a related early-copy defect. No defective invocation was established in the reviewed active STORM ring path.

### Location and proof

`source/utils/rma/MPIRemoteMemoryAgent.hpp:132` declares a local `T old_value`, passes its address to MPI_Fetch_and_op, and skips MPI_Win_flush when flush is false. It then returns old_value at line 143. A pending operation may not have produced the result yet. The method can both read an indeterminate value and let MPI retain a result-buffer address whose stack lifetime has ended. Later caller-side Flush cannot repair a returned-by-value result or restore that lifetime.

OFI FetchAndAdd at line 399 and verbs FetchAndAdd at line 382 similarly copy or return their registered scratch result without draining when false is passed. Their CAS methods, OFI:354 and verbs:347, copy scratch into old_value immediately as well. OFI also reuses registered operand scratch across operations, which is unsafe if a previous operation still depends on those contents.

MPI CAS uses the caller-provided result address directly, so its result may be recoverable after proper completion if caller buffers remain valid. Do not claim it has the identical local-stack bug. Its lifetime contract still needs to be explicit and consistent with native implementations.

MPI_Win_flush establishes origin and target completion; MPI_Win_flush_local establishes origin completion sufficient for result-buffer use. Skipping both requires some other valid completion mechanism before reading a result. [MPI flush semantics](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node331.htm).

### Repair specification

Make value-returning FetchAndAdd complete sufficiently to return a valid old value regardless of the compatibility flag, or reject false explicitly before initiating it. A value-returning interface cannot truthfully deliver a future result. Prefer a synchronous atomic API and a separately named asynchronous API returning a ticket with owned result and operand storage.

For ticket-based CAS or FAA, retain registered result storage and any operands the provider requires until completion. Copy to a user destination only after that operation completes. Do not use one shared scratch slot for several in-flight operations. Define whether completion also guarantees remote visibility; the synchronization contract must match what callers need for lock and publication decisions.

### Regression and review gates

Initialize a remote 64-bit word to 41. FAA(1) must return 41 and leave 42 after its promised completion. CAS(desired=7, expected=42) must return 42 and leave 7; a failed compare must return the actual word and leave it unchanged. Repeat under forced delayed completion, multiple outstanding operations if supported, and heavy stack reuse after the call. A provider that happens to finish immediately can hide the bug, so include an instrumented delayed-completion backend or harness.

Test MPI, OFI and verbs independently. Verify invalid/deferred legacy mode either rejects before side effects or obeys the new contract. Current synchronous callers should retain their semantics. Synchronizing unsafe calls may make a microbenchmark slower; that is the cost of obtaining the result that the API already claims to return.

## 8. B06 — Native atomics narrower than 64 bits are not implemented correctly

**Severity:** High when invoked because adjacent data or the wrong field can be modified. **Evidence:** Source-confirmed by operand/address construction. **Exposure:** Accepted template types smaller than eight bytes, especially uint32_t, on native OFI/verbs. A 64-bit-only control protocol does not exercise the illustrated subword failures.

### Location and concrete counterexamples

OFIRemoteMemoryAgent.hpp:354 accepts sizeof(T) <= 8, but CAS posts FI_UINT64 at line 382. It aligns the address to an eight-byte word and constructs expected/desirable operands containing only the shifted subword. Suppose the low 32-bit field is 5 and its adjacent high field is 9. CAS of the low field from 5 to 6 compares the whole word against 5, rather than against the word containing both 9 and 5. The compare fails even though the requested field equals 5.

OFI FAA at line 422 also posts FI_UINT64. Adding one to a low uint32_t field holding 0xffffffff carries into the adjacent high field. The expected uint32_t modular update preserves its neighbor; the posted 64-bit addition does not. OFI's local helper uses masked CAS loops at lines 824 and 850, so local and remote subword behavior can disagree.

IBVRemoteMemoryAgent.hpp:360 and :394 align four-byte addresses downward without shifting the operation to the high half. FAA directed at the high uint32_t element therefore acts on the low field and returns the wrong old field. Types of one or two bytes are also accepted by the size test without a complete subword-alignment implementation. An eight-byte operation on a shorter registered extent additionally requires range proof that the current element-count check does not supply.

### Repair specification

The simplest portable contract is to restrict native atomics to appropriately aligned 64-bit integer storage and reject unsupported types before network access. Make this restriction explicit in the API and backend capability checks; size alone is not a type contract. Preserve ordinary typed Get and Put support.

If subword atomics are required, use provider-supported typed atomics with validated width and alignment, or implement a whole-word CAS loop that preserves every neighboring bit. Such a loop requires legal access to the entire aligned word and a single coherent atomic domain. It cannot be safely bolted onto arbitrary one-element allocations or mixed with incompatible CPU/NIC atomic access assumptions. Query actual OFI atomic capabilities rather than assuming every datatype/operation is available. [Libfabric atomic operations and capability queries](https://ofiwg.github.io/libfabric/main/man/fi_atomic.3.html).

### Regression and review gates

For two adjacent uint32_t fields, target each half with successful CAS, failed CAS and FAA. Include low-field overflow, a nonzero neighbor, high-field overflow, and concurrent updates to separate fields. Assert the untargeted field is unchanged and the returned old value belongs to the targeted field. Repeat at the final element of a minimally sized region. Compare local and remote semantics.

If the chosen repair restricts types, those cases must fail predictably before posting any work; 64-bit integer cases must still pass. Do not add masked operations without checking allocation and registration boundaries. No native hardware reproduction was performed here, and no corruption of current 64-bit STORM control words is claimed. Supported 64-bit paths need not incur a new runtime cost; emulated subword operations would cost extra round trips.

## 9. B07 — Particle tracking drops origin identity

**Severity:** Medium. **Evidence:** Executed witness plus manager identity assignment. **Exposure:** MPI particles with equal rank-local IDs reach the same tracker or are gathered into a global route. This affects diagnostic histories, not the physical identity used by all other algorithms.

### Location and proof

`source/monte/manager/MonteCarloTransport.hpp:19` initializes myIDCounter to zero separately in each manager. AddParticles assigns destination.rank from rankWorld at line 45 and destination.id from that local sequence at line 47. Thus different origins can legitimately create particles with the same id.

`source/monte/manager/MonteCarloTracker.hpp:61` reports particles into a map keyed only by particle.id, declared at line 74. GetLocalTrackParticleRoute at line 32 accepts only id. GetTrackParticleRoute at line 42 gathers those local vectors across ranks and sorts by steps; it never filters by origin rank. Two physically distinct histories therefore become one diagnostic route.

The actual tracker header was instantiated with a minimal packet carrying id, steps and rank. Reporting (id=0, rank=0) and (id=0, rank=1) produced `route size=2 origins=0,1`. This is a direct key-collision witness; the MPI-global version follows from the same key choice and gather path.

### Repair specification

Define a ParticleIdentity containing immutable origin rank and rank-local ID, with ordering/equality suitable for the map. Use it in ReportParticle, local lookup, global lookup and every external route request. Preserve the origin field during migration and packing. Do not substitute the rank currently executing the particle: that would split one migrating history across multiple identities.

For a serial build use the same semantic key with origin zero or a serial identity adapter. If old id-only queries must remain for compatibility, limit them to unambiguous serial use or make ambiguity explicit. Never silently pick one origin. Sorting by steps can remain for route order, but equal-step snapshots may need a documented tie rule if exact diagnostic ordering matters.

### Regression and review gates

Create two particles with equal IDs on two origins, report interleaved snapshots on a common destination, and request each route independently. Each route must contain only its own origin and retain its expected step order. Run a two-rank gather version with migration and a serial version with ordinary IDs. Confirm reset discards all routes and an unknown identity returns an empty route.

Review callers of the public tracker API as part of the patch; changing just the map key is incomplete. Expected runtime and storage changes are limited to an extra key field and comparison, but a tracing-heavy benchmark should measure that cost if tracking is used in production.

## 10. B08 — MPI RMA transfer lengths silently narrow to int

**Severity:** Medium; a very large transfer can fail or transfer the wrong amount. **Evidence:** Source-confirmed. **Exposure:** An individual Get or Put with byte length greater than INT_MAX, or overflow in element-to-byte arithmetic. This is a dependency boundary issue, not a claim that ordinary small messages are broken.

### Location and boundary

`source/utils/rma/MPIRemoteMemoryAgent.hpp:98` implements Put. It computes count*sizeof(T) as size_t and casts that result to int for both MPI counts at line 103. Get at line 110 does the same at line 114. There is no checked multiplication or chunking in these methods.

On a platform with 32-bit int, a byte count of 2,147,483,648 does not fit. For an eight-byte element type, 268,435,456 elements already reach that boundary. Even though count and the allocation can fit size_t, the MPI call does not receive a representable count. The precise conversion result is implementation-dependent; common two's-complement builds produce a negative value. Do not assume it always fails safely: other oversized values can narrow to small nonnegative counts.

This backend also matters to configurations where the factory selects MPI RMA for a buffer even if another provider serves other operations. Diagnose the concrete agent chosen for the affected buffer rather than relying only on a top-level transport label.

### Repair specification

Check count against SIZE_MAX/sizeof(T) before multiplication and validate target displacement arithmetic and MPI_Aint representability. Split ordinary MPI_Get/Put calls into byte chunks no larger than INT_MAX, advancing both the local address and remote byte displacement by exactly the same amount. Respect the allocated target extent. Post all permitted chunks and then satisfy the original flush contract; do not silently truncate or add a mandatory flush per small chunk unless needed by resource limits.

MPI large-count APIs are an alternative only with verified build/runtime support. Preserve a portable chunking fallback. Use byte-pointer arithmetic inside the chunk loop if chunks need not align to T boundaries; do not confuse byte offsets with element offsets. For count zero, return safely under the same API semantics.

### Regression and review gates

Use an MPI-call recording shim or a testable chunk planner to verify lengths just below, at and above INT_MAX without requiring multi-gigabyte allocations in every unit test. Assert each chunk fits, chunk lengths sum to the requested byte count, and local/remote offsets cover the range without holes or overlap. Include multiplication and displacement overflow rejection.

Also run one real large-buffer integration test where resources permit, with a patterned payload and checks at every chunk boundary and final element. Exercise both Get and Put and synchronous/deferred modes. The arithmetic tests alone do not validate MPI visibility. Small-message runtime should be almost unchanged; large transfers gain correctness, not a promised throughput increase.

## 11. B09 — Real device execution omits per-cell work counts

**Severity:** Medium for load-balancing and diagnostic correctness. **Evidence:** Source-confirmed; no GPU execution in this audit. **Exposure:** Kokkos execution memory not accessible from the host. Host-accessible Kokkos backends follow the implemented counting path.

### Location and effect

`source/monte/gpu/KokkosLocalTransportExecutor.hpp:201` implements AddCellSteps but its entire accumulation/export body is guarded by `if constexpr(hostAccessible)` at line 203. The kernel's increment at line 904 and allocation at line 1166 have the same restriction. A real device backend therefore performs transport work without contributing that work to the per-cell array through this API.

`source/monte/manager/MonteCarloLifecycle.hpp:235` clears cellsStepsCounters each step and calls AddCellSteps at line 569. Rank-wide gpuPhysicsStepCount is separately included in the printed work total. Consequently a plausible total-step log does not establish that per-cell counts are available.

`source/3D/radiation/IMCCostCalculator.hpp:24` reads these per-cell counters and at lines 29–30 weights cells using them alongside beginning particle counts. In a fully device-transported region the step-based part is absent; mixed runs can undercount only device work. The particle-count contribution remains present, so this is not a claim that load balancing has no useful input or that radiation physics itself is wrong.

### Repair specification

Maintain a device-resident per-cell counter array for non-host-accessible execution. Increment the cell in which each counted transport event begins, consistently with the existing host convention. Use a correct device accumulation strategy, then copy or reduce the counts to host at the end-of-step export point. Merge once with host counts and reset once at the next step boundary.

Keep rank-wide work totals independent so AddCellSteps does not double-count them. If precise counters are intentionally optional for performance, expose that availability in the API and make the cost calculator choose an explicit fallback. Returning a zero-filled array indistinguishable from measured zero work is the defect to remove.

### Regression and review gates

Run a known packet population confined to one cell on actual CUDA/HIP or another non-host-accessible backend. Require that cell's count to be nonzero and untouched cells to remain zero. Compare the sum with the rank-wide counted event total under the same event convention. Then move packets across cells and compare CPU/device work distribution. Include a hybrid host/device step and two successive steps to detect double merges and missing resets.

Do not accept a Kokkos host-only pass as device validation. Count accumulation and host export have real overhead; measure it on contention-heavy single-cell and distributed-cell cases. Improving future load balance may outweigh it, but no percentage improvement is justified without those measurements.

## 12. B10 — Public compute-time getter always reports zero

**Severity:** Low. **Evidence:** Source-confirmed. **Exposure:** Any caller of the timing API, independently of backend.

### Location and effect

`source/monte/manager/MonteCarloManager.hpp:172` defines GetPureComputeTime and returns the literal zero at line 174. `source/3D/monte/MonteCarloManager3D.hpp:130` forwards that value through its implementation wrapper, and the public wrapper at line 254 exposes it again.

This is a placeholder presented as a measured duration. It can mislead downstream reports or performance comparisons. No production division-by-zero or scheduling decision caused by this getter was established, so its severity remains low. Existing loop phase counters elsewhere do not automatically give this getter a correct meaning.

### Repair specification

First define what the method returns: for example, local-rank transport-compute elapsed seconds for the most recently completed step. Specify inclusion of packet handling, physics, source generation and population control. For GPU execution distinguish host submission time from device execution time and define how overlap affects the metric. If callers need a maximum across ranks, provide that as a separate documented reduction; a getter should not unexpectedly perform a collective.

Accumulate the agreed metric with suitable clocks or device timers and return it. If a valid portable metric is not yet available, expose unavailability explicitly through a status/optional result or a documented API replacement. Do not use zero to mean both measured no work and not implemented. Keep compatibility behavior explicit in the wrapper.

### Regression and review gates

For a controlled nonempty CPU workload, require a finite nonnegative duration with a meaningful positive value at the timer's resolution. Require step reset behavior and wrapper parity. Avoid brittle absolute time thresholds; inject a timer or use a sufficiently large controlled workload. Check the empty-step contract separately. Validate GPU timing with actual device work before claiming that backend is supported.

The expected cost is timer instrumentation and optional device timing events. This finding is about honest measurement, not a claimed application speedup. A reviewer should reject renaming a broad loop wall time to pure compute without documenting its included waiting and overlap.

## 13. Reproduction package and observed results

The companion directory `docs/STORM_bug_audit_2026-09-08_repro/` contains the small witness sources and recorded outputs. They are audit artifacts, not production patches. The witness programs intentionally expose existing defects and are not all pass/fail regression tests ready to commit unchanged.

| Witness | What it actually exercises | Observed result |
| --- | --- | --- |
| tracker.cpp | Actual tracker map with equal IDs and distinct origins | route size=2 origins=0,1 |
| nan_fleck.cpp | Actual shared AdvanceIMC; valid slab fixture except NaN Fleck factor | No reported error; NaN surviving and deposited energy |
| counters.cpp, FORCE_SYNCHRONOUS | Actual engine methods; test-local Isend-to-Issend substitution | Worker stuck in FinishCounters; timeout exit 124 |
| counters.cpp with drain argument | Same forced completion semantics; root supplies missing receive | Both ranks finish |
| counters_eager.cpp | Actual engine methods and unmodified MPI_Isend | An unmatched snapshot remains after FinishCounters |

The eager witness uses a blocking probe after successful shutdown to demonstrate the queued message. A corrected implementation should leave no such message, so that old witness would need to become a bounded no-message assertion. Similarly, the hang witness's timeout documents the bug; a regression should require normal completion. Use the supplied README for exact compile and run commands.

The slab fixture was copied only into the report's evidence directory and adjusted to supply the invalid input. STORM headers were not edited. The tracker uses a minimal packet to isolate its key semantics rather than constructing a complete radiation simulation. MPI witnesses use two ranks on one machine. These limitations should stay attached to any bug ticket derived from the report.

B01, B04–B06 and B08–B10 have source proofs and proposed regression recipes, not claimed hardware reproductions. In particular, a native OFI or verbs fix is not validated merely because the corresponding MPI implementation passes.

## 14. Repair order and acceptance matrix

### Phase A: Current execution and scientific error reporting

Repair B01 and verify initial-versus-dynamic source accounting over multiple steps. Repair B03 and establish a complete invalid-opacity error path. Repair B02 with an explicit close protocol and adversarial send timing. Keep these as separate patches so any physics, accounting or shutdown regression can be localized.

Acceptance requires valid-input physics regressions to retain their expected conservation and statistical behavior, invalid Fleck factors to be rejected before tally mutation, and MPI shutdown to work with synchronous send completion and across repeated steps. Test both a serial build and an MPI build whose serial engine uses MPI_COMM_NULL. A test confined to the default eager MPI transport is insufficient for B02.

### Phase B: Make the RMA contract safe before adding overlap

Resolve B04 and B05 together at the API-design level but preserve separate regression coverage for copy-back and atomic result lifetime. Resolve B06 by either limiting atomics or implementing full typed semantics. Address B08 while touching the transfer contract. Do not launch the performance plan's asynchronous metadata pipeline before these choices are complete.

Write a short backend contract table covering: when the origin buffer can be reused, when a read/atomic result can be consumed, what Flush guarantees, what QuiesceTarget guarantees, which types are supported for atomics, and how pending operations interact with resize/destruction. Audit self-target fast paths against the same semantics. A fast local path must not conceal a remote correctness mismatch.

### Phase C: Restore trustworthy diagnostics

Repair identity tracking (B07), work accounting (B09), and timing (B10). Confirm that load-balancing consumers know whether their counters are measured, estimated or unavailable. Compare performance only after these diagnostics have a defined contract; otherwise a better-looking graph can reflect changed measurement rather than faster transport.

| System / mode | Mandatory coverage | Important limitation |
| --- | --- | --- |
| Serial CPU | B01, B03, B07, B10; source/census accounting | Does not exercise MPI shutdown or remote API branches |
| CPU MPI over shared memory/TCP | B02, B05 MPI, B07 global, B08 | Useful portability baseline, not native RDMA validation |
| CPU InfiniBand with MPI RMA | B02, B05 MPI, B08; full transport conservation | Record the actual MPI RMA component and completion behavior |
| CPU native OFI on InfiniBand | B04, B05, B06 plus full ring traffic | Requires real remote reads, not self-target tests |
| CPU direct verbs | B04, B05, B06 on concrete native agents | Factory fallbacks can bypass the code being tested |
| Kokkos host backend | Shared physics and host counter controls | Does not cover B09's excluded device branch |
| Non-host-accessible GPU backend | B03 device error propagation, B09, B10 GPU metric | Must execute on a real device with representative memory space |

Record actual backend selection, rank count, build macros, compiler flags, message sizes and memory-space accessibility for each run. Test communicator isolation where the engine supports subcommunicators. Preserve existing resize/termination tests as integration gates after the targeted corrections; they protect invariants that this report does not propose removing.

## 15. Concerns that are not promoted to confirmed bugs

### Raw object transport and ABI portability

MPI-enabled Particle is not simply an explicitly serialized scalar wire record; it participates in serialization through a polymorphic base. Raw RDMA transfer of such objects deserves an ABI and object-lifetime audit. However, receiving code can copy packet fields into newly constructed local objects before virtual use, so the mere presence of a vtable pointer is insufficient to prove a current crash. This report does not label raw-particle transport as a confirmed production failure.

Before changing its representation, identify exactly which received objects are used in place, copied, serialized and destroyed. Test heterogeneous address layouts and tracing builds, or move to a defined wire packet as a separately justified portability project. Do not treat this concern as permission to combine a packet-layout rewrite with B01–B03.

### Ordering, resize and provider progress

The ring transfer path deliberately has synchronization around metadata, payload publication and resize. Expensive waits are not themselves bugs. An alleged ordering defect needs a complete interleaving showing which buffer generation, ownership interval or visibility guarantee fails. Preserve the existing separate head/tail reads and resize protections unless a replacement protocol supplies its own proof and tests.

This audit has not established a new current deadlock in the particle-completion algorithm, a systematic energy bias for valid finite opacity inputs, or a new rank-routing error in the OFI communicator map. Those are appropriate targets for additional stress and model-specific verification, not additional confirmed findings to inflate this report.

### Scope of confidence

A successful correction of all ten findings would remove the specific defects described here, not certify STORM as bug-free. Native-provider behavior, GPU execution, physical model validation and very large workloads require infrastructure beyond the focused local witnesses. Keep outstanding validation items in the implementation handoff instead of silently marking them passed.

## 16. Reviewer checklist and handoff contract

For each implemented finding, the reviewing model should answer the following questions with code and test evidence:

1. Does the reported triggering condition still exist in the current working tree, and do the source hashes or changed lines explain any difference from this audit?
2. Does the patch repair the stated observable defect, or merely suppress an error, force a favorable schedule, or alter the measurement?
3. Are every affected backend and self/remote branch covered, including the branches that local tests cannot run?
4. Does the fix preserve result-buffer and source-buffer lifetime through completion, and preserve remote memory through quiescence?
5. Are initial particles, dynamically created particles, completed histories and census particles still accounted for exactly once?
6. Do invalid-input tests check state and tally integrity as well as the error code?
7. Are unsupported API modes rejected before side effects, with a contract callers can actually follow?
8. Does the regression distinguish the original bug from an unrelated timeout, unavailable network device, or build failure?
9. Are runtime consequences measured where meaningful and separated from correctness? No percentage speedup is inferred from these fixes.
10. Are unexecuted hardware tests explicitly left pending rather than described as verified?

The final implementation handoff should include one issue/commit per logical finding, the reproducer converted into an assertion of correct behavior, the selected API contract where alternatives were offered, and an explicit list of remaining hardware validation. No production fix is included in this document or its witness package.
