# Monte Carlo manager refactoring plan

## Goal

Split the Monte Carlo managers into components with one responsibility, while
preserving the public manager and factory APIs. The refactoring should make
GPU transport another local-execution policy, rather than adding more GPU
branches to the communication managers.

This is a design and migration plan; the line counts below are estimates for
the resulting files, including template definitions where they must remain in
headers. They are approximate (roughly +/- 30%), not acceptance criteria.

## Implemented first slice

The first implementation slice preserves the existing public manager APIs and
protocols while introducing the seams needed for the larger extraction:

- `manager/ParticleQueue.hpp` provides one heap-backed FIFO particle array with
  logical `Front`, `At`, `Append`, `Consume`, and active-range iteration. Serial
  now uses this representation instead of separate `particles`, `th`, and
  `av` arrays.
- `manager/MonteCarloParticleInitialization.hpp` centralizes the common
  per-particle reset (`timeLeft`, `initialWeight`, `steps`, tracing history, and
  MPI debug routing fields). Serial, TwoSided, and RDMA use the helper without
  changing their surrounding lifecycle ordering.
- `manager/MonteCarloTransportCore.hpp` provides the shared batch HandleAll
  driver. Serial uses it directly, and RDMA uses it for its host batch path;
  RDMA progress, device sweeps, and transfer decisions remain in RDMA-specific
  code.
- `manager/LocalTransportExecutor.hpp` now exposes a capitalized `Execute`
  batch operation for both vector and store-like particle containers. The
  existing lowercase `execute` overloads remain as compatibility wrappers.

This is intentionally an intermediate step: the full `MonteCarloManagerCore`
and communication backends described below are not yet extracted. The manager
definitions themselves have now also been split so that transport and protocol
code can be reviewed independently. The next implementation step should move
the shared lifecycle and event kernel only after characterization tests cover
generated particles, IDs, counters, tracking, and boundary outcomes.

The first-slice files are currently small enough to review independently:

| Implemented file | Current lines | Role |
| --- | ---: | --- |
| `manager/ParticleQueue.hpp` | 217 | single-array FIFO store |
| `manager/MonteCarloParticleInitialization.hpp` | 77 | object-oriented particle initialization component |
| `manager/MonteCarloStepState.hpp` | 18 | shared per-step result state |
| `manager/MonteCarloTracker.hpp` | 75 | shared local/distributed tracking implementation |
| `manager/MonteCarloTransportCore.hpp` | 98 | composed shared batch driver |
| `manager/LocalTransportExecutor.hpp` | 71 | host executor and compatibility wrappers |

## Implemented manager-file split

The second slice removes the long out-of-class definitions from the three
active managers while preserving their public APIs. These are the current
working-tree sizes (template definitions are kept in component `.hpp` files so callers
still get the required instantiations):

| File | Current lines | Extracted responsibility |
| --- | ---: | --- |
| `manager/MonteCarloManagerSerial.hpp` | 129 | Serial public API, state, and component composition |
| `manager/SerialMonteCarloLifecycle.hpp` | 128 | Serial construction, initialization, and `step()` lifecycle |
| `manager/SerialMonteCarloTransport.hpp` | 107 | Serial `HandleAll()` and per-particle outcome handling |
| `manager/parallel/TwoSidedMonteCarloManager.hpp` | 164 | Two-sided public API, state, and component composition |
| `manager/parallel/TwoSidedMonteCarloLifecycle.hpp` | 325 | Two-sided construction, initialization, and `step()` lifecycle |
| `manager/parallel/TwoSidedMonteCarloTransport.hpp` | 336 | Two-sided local transport, receive processing, and termination loop |
| `manager/parallel/RDMAMonteCarloManager.hpp` | 429 | RDMA public API, state, and component composition |
| `manager/parallel/RDMAManagerOperations.hpp` | 553 | RDMA construction, ownership, insertion, transfers, and reset |
| `manager/parallel/RegisteredSendBuffer.hpp` | 154 | Registered send-buffer storage and registration ownership |
| `manager/parallel/RDMASendBufferProtocol.hpp` | 334 | send-buffer queues, flushing, and pending-transfer bookkeeping |
| `manager/parallel/RDMARankHandlerLifecycle.hpp` | 411 | handler shrinking, diagnostics, stale-handler retirement, and preparation |
| `manager/parallel/RDMAMonteCarloTransport.hpp` | 827 | shared RDMA transport event policy, host/GPU transport, and `HandleAll()` |
| `manager/parallel/RDMAStepLifecycle.hpp` | 581 | RDMA `step()` setup, pre-step, census, post-step, and validation |

The three manager headers now total 722 lines, versus roughly 4,596 lines when
the three active manager implementations were monolithic. The behavior is not
yet fully deduplicated: Serial and TwoSided still have
backend-specific loops, and RDMA retains its required progress/termination
logic. The split makes those differences explicit and gives the next
`MonteCarloManagerCore` extraction stable seams.

### HPP/CPP boundary and OOP composition

The former `.inl` files are now ordinary component `.hpp` files. The manager
implementations are templates over `T`, `Grid`, and sometimes `Physics`; moving
their definitions to `.cpp` would require a closed list of explicit
instantiations and would break the existing generic factory. Non-template
runtime code continues to use `.cpp` (for example, `ReallocationAgent.cpp`).
The new `MonteCarloTransportCore` and `MonteCarloParticleInitializer` are
composed classes, and Serial/RDMA own a transport-core object. This gives the
executor and particle initialization policies replaceable OOP boundaries while
keeping template-dependent code in valid headers.

### What is shared today versus next

Today, particle initialization is shared through
`MonteCarloParticleInitialization.hpp`, and the batch executor shape is shared
through `MonteCarloTransportCore.hpp` plus `LocalTransportExecutor::Execute`.
The Serial host loop uses that driver directly; RDMA uses it for its host batch
path. TwoSided and RDMA still retain policy code around receive/progress and
device ownership.

The next extraction should move the complete initialization sequence—pre-step
setup, insertion, `updateGridData()`, `preStep()`, generated-particle
initialization, counters, population control, and `postStep()`—into one
templated lifecycle. It should also make the `HandleAll()` loop a shared
driver with backend hooks for `nextLocalBatch`, `commit`, `progress`, and
`noPendingWork`. The current first slice is deliberately limited to seams and
does not claim that all three loops are already one implementation.

## Current baseline (before the first slice)


| File                                     | Current lines | Main responsibilities currently mixed together                                                                 |
| ---------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------- |
| `MonteCarloManagerSerial.hpp`            | 477           | indexed particle storage, transport loop, boundary handling, tracking, counters, lifecycle                     |
| `parallel/TwoSidedMonteCarloManager.hpp` | 872           | vector storage, local transport, MPI buffers, termination, tracking, counters, lifecycle                       |
| `parallel/RDMAMonteCarloManager.hpp`     | 3,247         | rank handlers, RDMA protocol, reallocation, send scheduling, termination, transport loop, metrics, GPU staging |
| `parallel/MonteCarloManagerLegacy.hpp`   | 2,333         | legacy RDMA implementation and diagnostics                                                                     |


The first refactor targets Serial, TwoSided, and current RDMA. Legacy should
remain buildable while the new interfaces settle; migrating it is a separate
decision because it has a substantially different protocol.

## Target architecture

The managers become thin composition roots around a shared step coordinator:

```text
manager public API
  -> MonteCarloManagerCore
       -> StepLifecycle
       -> TransportKernel
       -> ParticleStore
       -> TransportBackend
       -> LocalTransportExecutor (host or GPU)
       -> Tracker / StepMetrics
```

The core owns the physics lifecycle and event semantics. Stores own particle
layout. Backends own communication and distributed completion. Executors own
how local transport is run. No component should reach through another
component's storage or MPI data structures.

## Core and executor design

There are two independent kinds of replaceability in this design.

### The manager core is the algorithm, not the hardware executor

`MonteCarloManagerCore` should be a policy-based class, conceptually:

```cpp
template<class ParticleStore, class TransportBackend,
         class LocalTransportExecutor>
class MonteCarloManagerCore;
```

It owns the shared step state and implements the common operations:

- prepare grid and physics state;
- insert and initialize existing particles;
- call `physics->updateGridData()` and `physics->preStep()`;
- initialize generated particles and counters;
- run the common `HandleAll()` driver;
- call population control and `physics->postStep()`.

The core does not know whether a particle is held in a Serial ring buffer, a
TwoSided vector, an RDMA rank handler, or a GPU resident pool. It asks the
selected store/backend for work and sends completed outcomes back through the
same event contract.

The intended instantiations are:

| Public manager | Particle store | Communication backend | Local executor |
| --- | --- | --- | --- |
| Serial | `LocalParticleQueue` | `LocalBackend` | Host or GPU |
| TwoSided | vector/queue adapter | `TwoSidedMPIBackend` | Host or GPU |
| RDMA | `RDMAParticleStore` | `RDMABackend` | Host or GPU |
| Legacy (later) | legacy handler adapter | `LegacyBackend` | Host initially |

Thus there is one core implementation, but several compile-time compositions.
The public manager classes remain thin API-preserving facades. Since the
factory already creates concrete manager types through templates, static
polymorphism is preferred in the hot path; it allows the compiler to inline
the Serial and host-executor cases.

The core should not be duplicated for host and GPU. GPU is an executor choice,
not a second lifecycle. A runtime host/GPU choice can be represented by a
coarse-grained `std::variant`, a type-erased executor, or a manager selected at
construction time. None of these should introduce a virtual call for every
particle event.

### Executor contract

An executor owns *how local physics work is advanced*. It does not own
population control, boundary policy, MPI/RDMA transfers, or distributed
termination.

The common contract should be batch-oriented:

```cpp
struct TransportBatch {
    // host-owned mutable particles or an ownership token for detached data
};

struct TransportCompletion {
    Particle particle;
    PhysicsStepResult result;
    // optional device timing/error/counter data
};

class LocalTransportExecutor {
public:
    BatchResult execute(TransportBatch, const ExecutorContext &);
};
```

The host executor invokes the ordinary `physics->step()` path, generally
processing one particle until it leaves local ownership. The GPU executor
packs only device-safe transport state, launches one or more waves, and
returns completion packets. The core then applies boundary conditions,
tracking/history, removals, and rank transfers on the host through the common
transport kernel.

The current `KokkosLocalTransportExecutor` is specialized to Grey IMC and
already returns `CompletedTransport`/`CompletedBatch` records. It should first
be moved behind this contract without changing its kernel. A later step can
generalize the device physics capability. The current `Physics` API matters
here: Serial and TwoSided accept `shared_ptr<MonteCarloPhysics<...>>`, while
RDMA is templated on a concrete `Physics` type for
`UsesDeviceTransport()`/`GetDeviceTransportViews()`. To support GPU execution
uniformly, either expose a non-GPU-dependent device capability interface from
`MonteCarloPhysics`, template all manager facades on the concrete physics type,
or keep GPU selection type-erased at the executor boundary. This is an API
decision, not something the storage refactor should hide.

### Executor lifecycle and ownership

The core must make ownership transitions explicit:

```text
ParticleStore -> TransportBatch -> Executor
Executor -> TransportCompletion -> TransportKernel
TransportKernel -> Backend (remove, transfer, or requeue)
```

While a batch is executing, the store must not reuse or resize the backing
memory. The executor must complete or explicitly retain ownership before the
core commits the batch. This is straightforward for Serial's single-threaded
host queue, but essential for GPU waves and RDMA detached buffers.

The executor may report device timings and physics-step counts through a
metrics object, but it must not update global particle counters directly. The
core remains the single place that accounts for event outcomes, so host and
GPU runs are comparable.

### Which combinations are immediately valid?

The architecture permits all combinations, but implementation support should
be staged:

- Serial + Host: first and lowest-risk validation target.
- TwoSided + Host: next, after MPI progress is behind the backend.
- RDMA + Host: preserves the current rank-handler behavior.
- RDMA + GPU: current capability and the first GPU migration target.
- Serial/TwoSided + GPU: possible after device capability is no longer tied to
  the concrete RDMA `Physics` template and their backends can progress while
  device work is resident.

This separation keeps the core reusable without claiming that every physics
implementation can be offloaded automatically.

## How much can be shared?

More of `step()` can be shared than of the current `HandleAll()` functions,
provided that `HandleAll()` is split into a common driver and backend hooks.
The current functions cannot be copied into one common loop verbatim because
they iterate over different ownership structures:

- Serial iterates over the indexed `th[]` array and returns slots to `av[]`.
- TwoSided iterates over a compact vector and sends through `BuffersManager`.
- RDMA scans active rank handlers, merges detached arrivals, progresses RDMA
  and reallocation state, flushes send buffers, and may run a resident GPU
  pool.

The intended result is still one shared `HandleAll()` entry point in the core:

```cpp
bool HandleAll(StepState &state)
{
    while (auto batch = backend.nextLocalBatch()) {
        executor.execute(*batch, [&](Particle &particle) {
            TransportOutcome outcome = kernel.process(particle, state);
            backend.commit(outcome, particle, state);
        });
        backend.progress();
    }
    backend.flushCompletedTransfers();
    return backend.localWorkDone() && backend.noPendingWork();
}
```

`nextLocalBatch()`, `commit()`, `progress()`, and `noPendingWork()` are the
backend-specific parts. The physics call, step accounting, tracking, boundary
semantics, and conversion of a physics result into a `TransportOutcome` are
shared. GPU execution returns completed events to the same `kernel.process()`
and `backend.commit()` path; it does not add a second communication loop.

### Estimated sharing

| Behavior | Shared implementation after extraction | Backend-specific implementation | Expected sharing |
| --- | ---: | ---: | ---: |
| Step initialization and finalization | 100–150 lines | 20–70 lines per manager | 75–90% of the semantic workflow |
| Host `HandleAll()` driver | 80–120 lines | 20–60 lines per manager | 60–80% of Serial/TwoSided loop structure |
| Physics/event kernel and boundary handling | 160–240 lines | 30–90 lines of outcome adapters | 60–75% of Serial/TwoSided transport logic |
| Complete current RDMA `HandleAll()` | 180–260 lines | 180–320 lines for rank scans, merge, RDMA/GPU progress | 35–50% of the current function |
| GPU local execution | shared policy/executor | no MPI-specific GPU loop | 80–100% across future GPU-capable managers |

In practical terms, the common initialization and event-processing code should
be roughly 300–450 lines. Each public manager should retain approximately
100–250 lines of adapter/backend code. RDMA will remain larger because active
rank scheduling, one-sided transfer, reallocation, and distributed termination
are not present in Serial or TwoSided.

### Initialization must be normalized deliberately

The lifecycle operations are conceptually shared—prepare grid state, load
particles, initialize per-particle step fields, call `updateGridData()`, call
`preStep()`, add generated particles, initialize counters, and start
termination—but the existing ordering is not identical. Serial and TwoSided
initialize existing particles before `preStep()`; current RDMA calls `preStep()`
before its rank-handler initialization pass and has different insertion/ID
paths for generated particles. The refactor must therefore:

1. Add characterization tests for the current ordering and generated-particle
   fields.
2. Put the desired ordering in one lifecycle implementation.
3. Give the store/backend only narrow hooks for insertion, ID assignment, and
   distributed count reduction.
4. Keep a temporary compatibility hook if RDMA requires its existing ordering;
   remove it after equivalence tests demonstrate that centralized ordering is
   safe.

This makes initialization genuinely shared without silently changing
`timeLeft`, `initialWeight`, IDs, debug routing fields, or GPU staging state.

## Proposed files and size estimates



### Shared, communication-independent code


| Proposed file                   | Est. lines | Responsibility                                                                               |
| ------------------------------- | ---------- | -------------------------------------------------------------------------------------------- |
| `MonteCarloStepState.hpp`       | 120        | `MonteCarloStepFinalData`, transport outcome/event types, step context                       |
| `MonteCarloTracker.hpp`         | 140        | route recording, reset, local lookup, optional distributed lookup adapter                    |
| `MonteCarloMetrics.hpp`         | 220        | particle counts, cell counters, timing, progress data, handler-memory metric                 |
| `ParticleStore.hpp`             | 180        | narrow store contract used by the core                                                       |
| `IndexedParticleStore.hpp`      | 300        | Serial `particles/th/av` free-list implementation and invariants                             |
| `VectorParticleStore.hpp`       | 180        | compact vector storage and removal for TwoSided MPI                                          |
| `MonteCarloTransportKernel.hpp` | 380        | invoke physics, tracking/history, boundary processing, status-to-outcome mapping             |
| `MonteCarloStepLifecycle.hpp`   | 300        | setup, initialization, pre-step, transport-loop orchestration, population control, post-step |
| `MonteCarloManagerCore.hpp`     | 450        | composition/orchestration glue and common forwarding API                                     |


**Subtotal: approximately 2,270 lines.**

### Communication backends


| Proposed file                            | Est. lines | Responsibility                                                                                  |
| ---------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------- |
| `backend/TransportBackend.hpp`           | 180        | backend lifecycle, transfer/removal/progress/finished contract                                  |
| `backend/LocalBackend.hpp`               | 100        | Serial completion and no-op communication                                                       |
| `backend/TwoSidedMPIBackend.hpp`         | 360        | `BuffersManager`, MPI receive callback, local decrement accounting, `AmountManager` termination |
| `backend/TerminationController.hpp`      | 240        | common termination state machine; local and distributed implementations                         |
| `backend/RDMABackend.hpp`                | 850        | `RankHandler2`, handler traversal, transfers, send buffers, RDMA progress and flushes           |
| `backend/RDMAReallocationController.hpp` | 260        | `ReallocationAgent`, requests, progress, retry coordination                                     |
| `backend/RDMAParticleStore.hpp`          | 320        | adapter for local rank-handler particles and detached arrivals                                  |


**Subtotal: approximately 2,310 lines.** RDMA is expected to remain the
largest backend because its protocol and progress rules are intrinsically
complex; the improvement is that those rules no longer occupy the manager's
lifecycle and GPU code.

### Local execution and GPU


| Proposed file                     | Est. lines | Responsibility                                                   |
| --------------------------------- | ---------- | ---------------------------------------------------------------- |
| `LocalTransportExecutor.hpp`      | 180        | executor concept plus host implementation and batch contract     |
| `gpu/GpuTransportExecutor.hpp`    | 80         | GPU executor interface and completion packet types               |
| `gpu/GpuTransportPolicy.hpp`      | 360        | ingestion, launch/hold policy, copy-back, bounced particles      |
| `gpu/GpuTransportMetrics.hpp`     | 120        | GPU counters and timing aggregation                              |
| `gpu/KokkosTransportExecutor.hpp` | 430        | move/adapt current `KokkosLocalTransportExecutor` implementation |


**Subtotal: approximately 1,170 lines.** The existing Kokkos implementation
should be moved mechanically first; policy extraction should follow only after
host/GPU equivalence tests exist.

### Public adapters and integration


| Proposed file                            | Est. lines | Responsibility                                                     |
| ---------------------------------------- | ---------- | ------------------------------------------------------------------ |
| `MonteCarloManagerSerial.hpp`            | 140        | public adapter assembling core + indexed store + local backend     |
| `parallel/TwoSidedMonteCarloManager.hpp` | 150        | public adapter assembling core + vector store + MPI backend        |
| `parallel/RDMAMonteCarloManager.hpp`     | 180        | public adapter assembling core + RDMA store/backend + executor     |
| `MonteCarloManagerFactory.hpp`           | 220        | unchanged manager selection, updated construction and capabilities |
| `3D/monte/MonteCarloManager3D.hpp/.cpp`  | 300 total  | forwarding adapters; preserve 3D API and cell-view handling        |


**Subtotal: approximately 990 lines.**

The new structure is expected to contain roughly 6,700 lines, but with each
file focused and testable. The three current managers total 4,596 lines, so
this is not primarily a line-count reduction: it is a dependency and
responsibility reduction. Some temporary duplication is acceptable during the
migration and should be removed only after behavior is verified.

## Important interface decisions



### Transport event contract

Unify the result currently represented by `StepResult` and
`MonteCarloFunctionality` into an internal outcome type. It must distinguish:

- continue in the current local store;
- move to another local cell;
- boundary reflection;
- remove/census;
- finish and return to population control;
- transfer to a destination rank.

The kernel may decide the physics and boundary semantics, but the backend must
perform the actual transfer or removal. This prevents the GPU executor from
performing MPI/RDMA operations.

### Particle ownership

Define explicit ownership for every transport round. A particle is either in a
host store, in a backend send queue, or in a GPU resident pool; it must not be
simultaneously visible in two of them. GPU completions return ownership to the
host kernel before communication decisions are made.

### GPU capability

Use a capability/trait or executor selection object instead of manager-wide
`#ifdef` branches. A host-only build gets the host executor; a GPU-capable
physics implementation may select the Kokkos executor. `gpuMinLaunchSize`,
`gpuHoldMaxSkips`, and `gpuMaxInnerSteps` remain configuration inputs to the
GPU policy, not to RDMA or MPI code.

### Templates and compile boundaries

Because the managers are templated, template definitions that depend on `T`,
`Grid`, or `Physics` will generally remain in headers (or use explicit
instantiation). Keep MPI- and GPU-specific includes in backend/executor files
so Serial users do not inherit unnecessary dependencies.

## Migration phases

1. **Characterize current behavior.** Add deterministic tests for Serial,
  TwoSided, and RDMA: particle outcomes, boundary reflection/removal,
   generated particles, counters, tracker routes, and population control.
2. **Extract passive data components.** Move tracker, metrics, and step-state
  types first, retaining forwarding methods and existing names.
3. **Extract the transport kernel.** Start from Serial's `HandleAll()` and
  RDMA's `ApplyTransportEvent()`. Keep backend callbacks explicit.
4. **Extract lifecycle orchestration.** Share initialization, `preStep`, loop
  setup, population control, and `postStep`; preserve manager-specific
   diagnostics until the end of this phase.
5. **Wrap storage.** Introduce store adapters without changing allocation or
  rank-handler algorithms.
6. **Migrate Serial to the core.** This validates the design without MPI and
  provides the reference behavior.
7. **Migrate TwoSided.** Move `BuffersManager` and `AmountManager` behind the
  two-sided backend; retain MPI barriers and verification semantics.
8. **Split current RDMA.** Extract rank-handler storage, transfer scheduling,
  reallocation, termination, and diagnostics in separate steps. Do not alter
   the RDMA protocol while moving code.
9. **Move GPU execution.** Adapt the existing Kokkos executor, then extract
  the GPU policy and metrics. Device completion packets must pass through the
   same transport-event contract as host events.
10. **Update factory and 3D adapters.** Keep `ManagerType`, factory fallback,
  and `MonteCarloManager3D` source compatibility.
11. **Remove old implementations.** Delete duplicated manager-local code only
  after all backends pass the equivalence and stress suites.
12. **Reassess Legacy.** Either adapt it to the backend contracts or explicitly
  document it as a frozen compatibility implementation.



## Verification and performance gates

- Host Serial, TwoSided, and RDMA produce equivalent deterministic outcomes.
- Tracker ordering, tracing history, generated-particle IDs, and counters are
unchanged.
- MPI termination reaches completion with zero lost or duplicated particles.
- RDMA reallocation and send-buffer flush tests pass under forced small
buffers and rank imbalance.
- GPU and host executors produce equivalent event streams and final particles.
- `STORM_WITH_GPU` off does not include Kokkos headers or symbols.
- Existing factory and 3D call sites compile unchanged.
- Benchmark the inner transport loop, MPI progress, RDMA transfers, and GPU
staging separately; reject the refactor if the host inner loop regresses
materially without an identified cause.



## Review conclusions

The shared abstraction should be the transport lifecycle and event contract,
not a base class that owns all particle storage. Serial's indexed arrays,
TwoSided's vector, and RDMA's rank-handler matrix have incompatible mutation
and progress rules. A storage/backend policy boundary avoids forcing those
differences into virtual methods or a complicated inheritance tree.

The GPU path should be shared at the local-execution boundary only. It should
not be made a property of the RDMA manager: Serial and TwoSided may eventually
use the same executor, while RDMA remains responsible for communication and
termination. Finally, the plan deliberately keeps Legacy outside the first
cut so its older protocol does not constrain the new common design.
