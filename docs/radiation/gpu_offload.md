# STORM GPU offloading

This note describes the current GPU path in STORM (the Monte Carlo radiation
subsystem), what is worth offloading next, what should stay on the host, and
a review of the implementation as of August 2026.

Related plans: [DDMC GPU offload plan](ddmc_gpu_offload_plan.md),
[Monte Carlo manager refactoring](../monte_carlo_manager_refactoring_plan.md).
GPU support is compiled with `STORM_WITH_GPU` (Kokkos, typically HIP on
Frontier). It is **not** a CUDA-only fork: host and device share portable
kernels.

## Architecture in one picture

```text
preStep (CPU)
  opacities, Fleck, spectral tables, DDMC precompute, RW eligibility
  IMCDeviceExecutor::prepareStep()  -->  GreyIMCData upload
  source Plan (CPU) --> EmitSourcesOnDevice (GPU slots)
  Comb on device census (optional)

MonteCarloManager transport loop
  host arrivals  --pack-->  KokkosLocalTransportExecutor resident pool
  AdvanceWave:
    LaunchGreyIMCTransport  (AdvanceOne per packet, bounded inner steps)
    compact survivors / census / remotes / terminals / fallbacks
    copy only packets that left the device
  ApplyTransportEvent on host for terminals, remotes, HostOnly, REMOVE

postStep (CPU)
  copy tallies, DDMC flux RHS, Comb, hydro feedback
```

The design intent is correct: **the GPU owns local transport**, and the host
owns MPI, polymorphic boundaries, population-control policy, and anything that
is not a POD event kernel.

---

## What is already done

### Shared physics kernels (the important part)

| Kernel | Location | Device use |
| --- | --- | --- |
| Grey and multigroup IMC | `radiation/transport/AdvanceIMC.hpp` | Yes. Counter-based RNG, atomic tallies, comoving frame, momentum deposit. |
| Random walk (PGRW) | `radiation/random_walk/AdvanceRandomWalk.hpp` | Yes, via `GreyIMCKernel::AdvanceOne`. |
| DDMC event + IMC↔DDMC interface | `radiation/ddmc/AdvanceDDMC.hpp` | Yes when `withDDMC` and `ddmcGpuEnable`, including thermalizing external-source leaks and opacity-CDF in-group sampling. |
| Thermal source sampling | `radiation/source/SourceCore.hpp` + `gpu/SourceDeviceEmit.hpp` | Plan is CPU; packet construction writes into the resident pool. |
| Comb activation | `population/CombCore.hpp` + `gpu/CombDeviceActivate.hpp` | Operates on device census without a full D2H of the census population. |

`gpu/GreyIMCKernel.hpp::AdvanceOne` dispatches RW → DDMC → IMC, then tries
`TryIMCToDDMCInterface` on IMC `CELL_MOVE`. Local `CELL_MOVE` and reflecting
rigid walls stay on device (`TryKeepPacketOnDevice`). Census (`DONE`) stays
on device until Comb. Rank hops, `REMOVE`, `HostOnly` boundaries, kernel
errors, and explicit `HostFallback` compact to the host.

### Device data and executor

- `gpu/GreyIMCData.hpp` uploads a flattened directed-face grid, opacity/Fleck
  tables, hydro velocities, spectral tables, DDMC CSR graph + interface
  tables, random-walk tables, source tets, and atomic tally buffers. Grid
  connectivity is versioned with `GetBuildGeneration()`; physics tables
  re-upload every radiation step.
- `gpu/KokkosLocalTransportExecutor.hpp` owns a persistent packet pool
  (hot `DeviceParticle` + cold `DeviceParticleCold`), bounded waves
  (`kMaxWaveParticles = 4M`), inner-step cap (`gpuMaxInnerSteps`, default 64),
  census promotion, remote hold/flush, and split expansion for DDMC interface
  copies.
- `radiation/gpu/IMCDeviceExecutor.hpp` is the eligibility and upload bridge.
  CPU builds compile the lifecycle without Kokkos.

### Manager integration

GPU transport belongs to the shared **`MonteCarloManager`**. Serial, P2P, and
RDMA select the communication engine independently of the device executor;
P2P and RDMA stage remote packets through host memory.

Resident census, device source emission, Comb-on-census, and deferred D2H of
census identities are all implemented. Launch holding (`gpuMinLaunchSize`,
`gpuHoldMaxSkips`) batches small arrival bursts.

### Eligibility (what actually runs on GPU)

From `IMCDeviceExecutor`:

**Grey IMC** (`GreyKernelEligible`): no multigroup, no DDMC, no Compton, no
hydro+RW together, no post-process, no observer, no polarization, not
`STORM_DEBUG` / tracing history.

**Full IMC** (`SharedFullIMCKernelEligible`): multigroup, same exclusions,
plus not `imcDiffForceLegacy_`.

**DDMC** uses two predicates. `SharedDDMCKernelEligible` enables full mixed
IMC/DDMC transport when DDMC is on and Compton, post-process, observers, and
polarization are off. `SharedDDMCEventKernelEligible` also permits
external-source post-process DDMC: resident DDMC events run on the device,
while IMC packets return to the host for observer and boundary callbacks.
Both device routes require `ddmcGpuEnable`. Hydro, multigroup, RW, and
moving-interface snapshots are uploaded when those flags are set.

**Device source / census tallies**: same family, plus no Compton, no
post-process, no adaptive source-cell group scores.

When eligibility fails, the run stays on the existing CPU `stepImpl` path.
That is the right default.

### Evidence from recent GPU Densmore runs

A 32-rank grey Densmore GPU job (`densmore_none_gpu_1ns_spec_*`) shows the
intended split: **device+compact dominates** (~8 s/step), host-events are
small (~0.05 s), source generation is ~8 ms, census bookkeeping is ~1 ms,
and **fallbacks = 0**. Typical wave size is ~5.5×10⁵ launched packets.

The same logs also show where the path is not yet optimal (see review below):
~21 M rank-hop packets packed per step, ~4 GiB H2D and ~6 GiB D2H per step,
~1400 Kokkos reallocations per step, and millions of idle RDMA loop rounds.

---

## What to offload next (recommended)

Order is by expected wall-clock return, not by novelty.

### 1. Stop paying for rank hops as full host particles (highest return)

On the Densmore GPU run, **packed remotes ≈ launched / 3** every step, and
D2H is several gigabytes per radiation step even though census stays on
device. Overlapped RMA time is ~3×10⁻⁵ s — the copies are not hiding under
useful MPI.

Recommended, in increasing difficulty:

1. **Do not unpack remotes to `Particle<PointT>` until the send buffer
   needs them.** Keep a compact device-side remote SoA (or the existing
   `CompletedTransport`) and serialize directly into RDMA payload.
2. **Hold remotes longer** when the destination rank is not ready, and
   overlap the next wave with the previous D2H. The hold knobs exist;
   the overlap does not.
3. **GPU-aware / on-device pack for neighbor hops** only after (1)–(2)
   are measured. Full GPU MPI of IMC packets is not the first step.

Until rank-hop traffic shrinks, more physics on the GPU will not change
the H2D/D2H tax.

### 2. Kill mid-step `Kokkos::resize` churn

`AdvanceWave` still `ShrinkTo(nextPackets_, 0)` (and similarly for cold)
after every wave, then grows the buffers again. Combined with
`EnsureCapacity` on ingest, that matches the **~1400 reallocations/step**
in Densmore logs. `ReservePoolCapacity` / `gpuDevicePoolHeadroomFactor`
already exist to avoid this. The shrink of the *scratch* views undoes it.

Keep a high-water reservation for the whole step. Shrink only at
step end, or never below the reserved capacity.

### 3. Specialize or stage the transport kernel

`AdvanceOne` always compiles RW, DDMC, IMC, and interface logic into one
lambda. For grey Densmore that is wasted divergence, extra register
pressure, and extra cold-state traffic (`AssignCold` exists specifically
because whole-struct copies blew occupancy).

Recommended:

- Compile **grey-only**, **multigroup IMC**, **IMC+DDMC** as separate
  launch paths selected by eligibility (the specialized Densmore scripts
  are already chasing this).
- Keep one portable source of physics (`AdvanceIMC` / `AdvanceDDMC`);
  specialization is a dispatcher, not a second algorithm.

### 4. Finish DDMC on the current architecture (not a new executor)

The [DDMC plan](ddmc_gpu_offload_plan.md) is still the right sequence.
Remaining high-value device work, if profiles show fallbacks or host
time:

- Add regression coverage for thermalizing-boundary stay-DDMC and leave-band
  transitions; both now execute in the shared CPU/GPU kernel.
- Remaining unsupported post-process IMC work still returns to the host.
- Confirm moving-interface + remote pending-flux on GPU against
  `ddmc_mpi_zero_cell` and CrookedPipe CPU/GPU compare.

Do **not** start a second DDMC launch path. Reuse `AdvanceWave`.

### 5. Hydro + random walk together, if production needs it

Eligibility currently forbids `withHydro && withRandomWalk`. That is a
capability hole, not a physics reason. Port the comoving RW path only
after a differential test against CPU `stepImpl`.

### 6. Serial (and maybe TwoSided) GPU executor

Same `KokkosLocalTransportExecutor`, no MPI. This is for **correctness
and occupancy debugging**, not for Frontier throughput. It is the
cheapest way to get host/device event-level parity tests without RDMA.

### Lower priority (do later or never as “the next GPU project”)

- Device source *planning* (photon counts per cell). It is O(cells) and
  already cheap (~8 ms vs ~9 s transport).
- Device opacity / Fleck evaluation. `preStep` snapshots tables once;
  the kernel already reads SoA tables.
- Device DDMC *precompute*. Geometry, ghost eligibility MPI, and
  reciprocity should stay on the CPU until transport is no longer the
  bottleneck (same conclusion as the DDMC plan).
- Comb policy (targets, MPI reductions of cell scores). Activation of an
  already-decided Comb is on device; the collective decision is not a
  particle kernel.

---

## What not to offload (and why)

| Leave on the CPU | Why |
| --- | --- |
| Voronoi / tessellation, AMR, point motion | Host-only mesh APIs, irregular rebuilds, MPI of generators. Not an IMC event. |
| Hydro Godunov step, gravity | Different data motion and time scales. Radiation already couples through table snapshots and tallies. |
| RDMA protocol, termination, send-buffer policy | Must stay the single owner of rank transfers. The GPU executor must not MPI. |
| Virtual / polymorphic boundary conditions except rigid reflect | Device grid stores only `DeviceBoundaryFaceBehavior`. Observers, inflow, custom HostOnly faces need the host `ApplyTransportEvent` path. |
| Compton / CMMC | Group-coupled scattering, occupation numbers, induced processes, and a separate source generator. Eligibility correctly refuses GPU. A device Compton kernel would be a new physics port, not an offload of `AdvanceIMC`. |
| Polarization (Stokes / polarized Thomson) | Extra particle state, different scatter kernel, currently excluded. Cold SoA already has optional fields; do not turn them on in the hot kernel until IMC occupancy is healthy. |
| Observers, post-process tallies, tracing history, `STORM_DEBUG` | Side-channel I/O and extra branches. GPU eligibility is false; that is correct. |
| Adaptive source-cell group scores | Host planning that changes emission counts; would force extra reductions and invalidate device source emission. |
| Opacity objects / EOS | Virtual or heavy cell queries. Snapshot to tables on the host (already done). |
| DDMC precompute (`precomputeDDMCData`) | Nested `CellData` / `FaceLeak`, tessellation queries, ghost exchange, reciprocity. Upload the **result**. |
| Population-control *policy* (how many packets per cell) | Collective, integer, and rank-local bookkeeping. Device Comb only applies a decided map. |

For these, a GPU port would either duplicate a second algorithm, destroy
occupancy, or move work that is not in the hot path.

---

## Review of the current implementation

### Correctness — generally strong

**What works**

- One physics implementation for CPU and GPU. `AdvanceIMC` / `AdvanceDDMC` /
  `AdvanceRandomWalk` are the specification. The CPU transport process can
  call the same kernels (`IMCTransportProcess` routes through
  `SharedFullIMCKernelEligible` / `SharedDDMCKernelEligible`). That is the
  right way to avoid a silent second IMC.
- Counter-based RNG on the particle (`rngKey` / `rngCounter`) makes a
  host/device differential test well-defined.
- Explicit fallback instead of approximating unsupported DDMC events
  (`HostFallback`, overflow throws). Grey Densmore shows **zero fallbacks**.
- Census and Comb staying on device avoids a class of unpack bugs and
  weight double-counting, as long as `deviceCensusValid` /
  `hostParticlesValid` stay consistent (the RDMA lifecycle is careful here).
- Tally atomics on device, then `AddTallies` / `copyCensusTallies` in
  post-step, match the deferred-tally CPU model.
- Debug and tracing builds force CPU. Do not “GPU debug” by compiling
  history into the kernel.

**Risks and gaps**

- **GPU is RDMA-only.** Serial GPU parity is missing, so many correctness
  bugs will only show up on multi-rank GPU jobs.
- **Nudge-to-center** on every local `CELL_MOVE` (`NudgeTowardCellCenter`,
  ε = 1e-8) is a robustness hack for Voronoi clipping. It is also a
  systematic perturbation vs a strict CPU face-crossing. Keep it, but treat
  it as a known bias when comparing trajectories, not just cell energy.
- **Reflecting walls on device** reimplement host reflection. Any change to
  host BC reflection must be duplicated or it will pass CPU tests and fail
  GPU.
- **DDMC GPU is feature-complete enough to be dangerous.** `ddmcGpuEnable`
  defaults **on** in GPU builds. The DDMC plan said default off until
  validation gates pass. If CrookedPipe / Densmore-DDMC / `ddmc_mpi_zero_cell`
  GPU gates are not in normal CI, that default is too aggressive.
- **Inner-step bound (64)** changes event grouping vs CPU `step()` which
  runs until a terminal. Physics of a single event should match; *when*
  a packet is compacted as a survivor vs continued can change MPI timing
  and Comb epochs, not the event kernel itself. Still, do not raise the cap
  blindly: the config comment notes a 1.7× CrookedPipe win of 64 vs 4096
  from less lane imbalance.
- Split expansion assigns `id = max()` on copies and repairs identities
  later. That is fine if every host path that can observe an ID goes through
  `AssignPendingCensusIdentities`. Audit any diagnostic that reads IDs
  mid-step.

### Optimality — the physics kernel is ahead of the runtime

**Good**

- Resident pool, census-on-device, source-on-device, Comb-on-device, and
  “copy only terminals” are the right shape.
- Hot/cold particle split and `AssignCold` are evidence-based occupancy
  work, not cargo-cult SoA.
- Bounded waves, launch holds, and pool headroom show the authors already
  found empty launches and resize storms.
- Shared tables (grid uploaded on mesh generation only) avoid the worst
  H2D.

**Not good enough yet**

1. **Rank-hop D2H/H2D dominates data motion.** Several GiB per step on a
   problem whose census never leaves the GPU. This is the first
   optimization, not a new kernel.
2. **Reallocation churn (~10³/step)** from shrinking scratch views to
   zero. This is a bug relative to the stated `ReservePoolCapacity` policy.
3. **Idle RDMA rounds in the millions** while the device is the real
   worker. `loopRounds ≈ idleRounds` on Densmore. The host progress loop
   is still a CPU spin around a GPU wave. Pumping RMA from inside
   `progress()` during the kernel is the right idea; the idle-round
   counts say it is not effective enough (or the accounting includes
   empty scans). Worth measuring with rocprof + MPI traces before
   rewriting termination.
4. **One mega-kernel** for all physics combinations. Register pressure
   and divergence will cap occupancy on mixed IMC/DDMC/RW. Specialization
   is the next kernel-side win.
5. **Survivor compaction uses `atomic_fetch_add` into `nextPackets`.**
   That is simple and correctness-friendly; it is also a contended atomic
   on every unfinished packet. A two-pass prefix-sum compact (already
   used for split expansion) would scale better at 5×10⁵–2×10⁶ active
   packets.
6. **`Ingest` packs on the host in a serial loop** then `deep_copy`.
   Fine for modest arrivals; on Densmore, `packed ≈ 2×10⁷` per step
   (~0.3 s pack). If remotes keep bouncing, this stays visible.
7. **Atomic cell tallies** on a Voronoi mesh will collide in optically
   thick cells. Expected; only worth cell-wise privatization if profiles
   show it after (1)–(3).
8. **No Serial GPU** means you cannot occupancy-tune without MPI noise.

### Architecture and maintainability

**Strengths**

- Clear ownership: physics in portable headers, data in `GreyIMCData`,
  scheduling in the Kokkos executor, MPI in the RDMA manager.
- Eligibility is a feature matrix, not a comment.
- `GreyIMCData` is misnamed now (it holds spectral, DDMC, RW, source
  tets). Rename when touching it; do not split lifetimes.

**Weaknesses**

- GPU policy (hold, remote flush, census promote, ingest) lives
  inside `MonteCarloManager` + executor methods. The manager
  refactoring plan’s `GpuTransportPolicy` would make this testable.
- `LaunchGreyIMCTransport` is one 130-line lambda with many captured
  views. Hard to unit-test overflow and compact cases.
- Two particle layouts (`Particle<T>` vs `DeviceParticle` + cold) plus
  pack/unpack are a standing correctness tax. Every new field must be
  added in three places (`Particle`, pack, unpack). A single POD with
  host mirrors would be safer.
- `gpuLaunchSize` is documented as unused by the resident pool, while
  `kMaxWaveParticles` is a hardcoded 4 M in the executor. One cap,
  configured in `MonteCarloConfig`.

### Overall verdict

The GPU offload is **architecturally right and physically conservative**:
shared kernels, explicit fallbacks, resident census, and a narrow
eligibility matrix. For grey / multigroup IMC without Compton or
observers, it is in production shape on RDMA, with Densmore showing the
work on the device and zero fallbacks.

It is **not yet runtime-optimal**. The next wins are almost all
**data motion and buffer policy** (rank-hop copies, Kokkos resize, idle
host loop), then **kernel specialization**, then remaining DDMC
fallbacks. Do not offload hydro, mesh, Compton, or DDMC precompute in
order to look like more GPU work.

---

## Suggested near-term work list

1. Stop shrinking executor scratch views below the step reservation;
   re-run Densmore and confirm `reallocations` collapses.
2. Profile rank-hop pack/unpack; add a compact remote path that does not
   round-trip full `Particle` objects.
3. Split the launch lambda into grey vs DDMC-capable kernels without
   duplicating `AdvanceIMC` / `AdvanceDDMC`.
4. Keep `ddmcGpuEnable` off by default until GPU CrookedPipe, Densmore
   DDMC, and `ddmc_mpi_zero_cell` are in the GPU CI set; or document that
   those gates already pass and leave the default on.
5. Add a Serial GPU smoke test that advances a fixed packet list one
   event and compares to the CPU shared kernel.
6. Revisit idle RDMA rounds only after (1)–(2); the progress callback is
   the right hook if the kernel is actually overlapping.

Do not start device Compton, device tessellation, or a second DDMC
scheduler.
