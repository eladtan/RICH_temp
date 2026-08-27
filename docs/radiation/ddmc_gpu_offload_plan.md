# DDMC GPU offload plan

## Decision and target

Offload **DDMC packet transport**, while retaining DDMC geometry/physics
precomputation on the CPU initially.  The useful end state is a single
device-resident packet pool that can alternate between IMC and DDMC without a
host round trip for every DDMC event.  Packets return to the host only when
they require MPI delivery, a host-only boundary/observer action, or reach a
configured bounded device wave.

This is not a one-line extension of the existing IMC kernel.  DDMC packets do
not track a physical ray while resident, choose a variable-length face list,
may change representation at an IMC-DDMC interface, and add a face-momentum
tally at both sides of a local leak.  A correct offload therefore needs a
portable DDMC event kernel plus a compact device representation of the
precomputed DDMC graph.

The recommended delivery order is:

1. Static, grey DDMC with local DDMC-to-DDMC leaks and host handling for
   terminal events.
2. Mixed IMC/DDMC transport in the same device wave, including MPI handoff.
3. Multigroup PGRW-DDMC, device group-energy tallies, and hydro/moving
   interfaces.
4. Optional GPU DDMC precompute only if profiles prove that the CPU
   precompute is material after transport has moved to the GPU.

Do not make device DDMC the default until the validation gates below pass.
Keep a runtime capability check and an explicit CPU fallback for every
unsupported feature.

## Current implementation and gap

STORM already has a good foundation for resident GPU transport:

| Area | Current behavior | DDMC consequence |
| --- | --- | --- |
| Device execution | `gpu/KokkosLocalTransportExecutor.hpp` owns a persistent packet pool, advances bounded waves, compacts terminal packets, and lets the RDMA manager process only terminal events. | Reuse this scheduling and compaction model; do not create a separate per-DDMC-event launch path. |
| Device data | `gpu/GreyIMCData.hpp` uploads a flattened directed-face grid, IMC opacity tables, selected spectral tables, random-walk data, and atomic tally buffers. | Add a DDMC graph and DDMC tally buffers to this object (or rename it to a neutral transport-data object). |
| Shared IMC kernel | `radiation/transport/AdvanceIMC.hpp` is host/device portable and uses counter-based RNG. | Create an analogous, independently testable `ddmc::AdvanceDDMC` kernel and a dispatcher which selects IMC or DDMC per packet. |
| Admission | In `radiation/RadiationIMC.hpp`, `GreyKernelEligible()` and `SharedFullIMCKernelEligible()` both require `!withDDMC`. | DDMC is presently always CPU-only, even in a GPU build. The new eligibility test must be a feature matrix rather than simply removing this exclusion. |
| DDMC precompute | `precomputeDDMCData()` builds `CellData` and a `vector<FaceLeak>` per cell, exchanges ghost eligibility/coefficient data, and checks local reciprocity. | This host-oriented, nested structure and its MPI calls are not device-accessible. Flatten and upload its *result* first; preserve CPU precompute as the source of truth. |
| DDMC event | `tryDDMCStep()` implements residence, absorption, census, cutoff/upscatter, face selection, DDMC-to-DDMC leakage, DDMC-to-IMC leakage, and flux feedback. | It is the behavioral specification for the device kernel. Port it in small, parity-tested increments, not by duplicating an uncontrolled second algorithm. |
| MPI/handoff | `RDMAMonteCarloManager` keeps device survivors resident but compacts particles that leave the local grid and applies host/RDMA policy. | A DDMC leak to a ghost cell must become a terminal `CELL_MOVE`, carry the DDMC flags and `pendingFlux`, and follow this same path. |

The GPU path currently uploads only IMC and random-walk tallies.  DDMC also
needs material/radiation energy, optional group radiation energy, event
counters, and the face-flux right-hand side used by
`applyDDMCMomentumFeedback()`.

## Proposed architecture

### 1. Preserve CPU precompute, publish a device snapshot

Keep `RadiationIMC::precomputeDDMCData()` on the host for the first complete
implementation.  It depends on tessellation methods, boundary polymorphism,
opacity calls, and collective MPI ghost exchange.  It runs once per radiation
step after hydro state and opacities are current, so it is not in the packet
event hot path.

At the end of precompute, build a `DDMCDeviceSnapshot` with structure-of-
arrays layout.  It should contain only trivially-copyable fields and use
indices into the existing flattened grid.

```text
per cell (N local cells)
  eligible / feature flags       sigmaEnergyAbs, sigmaParticleGate
  totalLeakRate                  fleck, gamma, velocityDivergence
  groupCutoff                    cell velocity / thermal sampling metadata
  leakOffsets[N + 1]             # CSR offsets into directed leak arrays

per directed DDMC leak (L total)
  rate, ddmcRate                 nextCellIndex, faceIndex, faceKind
  targetGroupCutoff              targetDDMCEligible
  outwardNormal                  faceCenter

device accumulation
  materialEnergy[N]              radiationEnergy[N]
  groupRadiationEnergy[N*G]      fluxRhs[N] (plus local ghost slots if needed)
  DDMC counters                  optional compact diagnostic records
```

`FaceLeak::rate` must be stored exactly as finalized by precompute.  In
particular, it must preserve the two-sided resistance/conductance construction
in `DDMCGeometry.hpp`, the DDMC-vs-IMC split (`ddmcRate`), target eligibility,
and the final mixed-face refresh.  Do not recompute conductance in the device
kernel during the first implementation.

The existing device grid lacks a face-centroid array.  CPU DDMC uses
`grid.FaceCM()` to place an emitted IMC packet at the leakage face, so upload
one face center per directed face (or a shared-face table with an indirection).
Plane normals and offsets alone cannot reconstruct a face centroid.

Add explicit snapshot generation/version tracking alongside
`gpuGridBuildGeneration_`.  Re-upload DDMC data whenever precompute runs,
even if mesh connectivity did not change: eligibility, rates, Fleck factors,
velocities, cutoffs, and thermal data can change every radiation step.

### 2. Add a portable DDMC event kernel

Introduce a new header, for example
`source/monte/radiation/ddmc/AdvanceDDMC.hpp`, containing only:

- POD views and POD result types;
- Kokkos-compatible math/atomic wrappers from `TransportPortability.hpp`;
- `CounterRNG`, using the existing particle key/counter;
- no `std::vector`, virtual boundary calls, host opacity object, exceptions,
  MPI calls, or dynamic allocation.

The kernel should return the same event vocabulary as IMC:

| Device result | Meaning / manager action |
| --- | --- |
| `NO_CELL_MOVE` | Packet is still local and can take another bounded device step. This includes entering residence and a local DDMC-to-DDMC leak after its cell index is updated. |
| `CELL_MOVE` to local cell | Update cell index and continue on device. The next dispatch may be DDMC or IMC. |
| `CELL_MOVE` to ghost cell | Compact to host. The existing manager/RDMA path sends the fully serialized particle. |
| `DONE`, `REMOVE`, host-only boundary | Compact to host and run the existing policy. |
| capability/validity fallback | Compact with a distinct reason and execute the CPU `stepImpl` path, so no packet is silently dropped or approximated. |

The event calculation follows the current CPU ordering exactly:

1. Validate the state and DDMC eligibility; an incoming IMC packet must pass
   the particle optical-depth gate before it becomes resident.
2. Compete exponential leakage/upscatter time, census time, and the
   compression cutoff time.
3. Apply comoving weight/frequency evolution, implicit absorption, and
   time-integrated radiation/material tallies.
4. Select a face using the CSR leak rates; select the DDMC channel using
   `ddmcRate / rate`; sample the asymptotic or transport angular law.
5. Update particle state and accumulate the source/target flux terms with the
   same ownership semantics as CPU code.
6. Return a terminal result only for a remote action or an action deliberately
   left host-side.

Use a linear scan over a cell's face segment for the first version.  Voronoi
cell degree is normally small, which avoids a large CDF allocation.  Profile
before adding an alias table: a per-cell CDF or alias representation costs
memory, must be rebuilt every radiation step, and can add divergence without
helping typical cells.

Extend the current executor rather than replacing it.  Its wave loop should
call a generic `AdvanceTransportPacket()` dispatcher:

```text
if packet.isDDMC() or ddmcCellEligible[packet.cellIndex]:
    AdvanceDDMC(packet, coldState, deviceViews)
else:
    AdvanceIMC(packet, deviceViews, opacityPolicy)
```

This enables a local DDMC-to-IMC transition to continue in the same device
wave and avoids the worst possible design: copying every interface packet to
the host only to copy it back into the same device pool.

### 3. Particle state, tallies, and MPI semantics

`DeviceParticle` already carries the radiation-state flags, and
`DeviceParticleCold` already preserves `pendingFlux` and `bypassCellID`; this
is the right split for DDMC.  Add fields only if the CPU state has a piece of
DDMC state not represented there.  Keep `DeviceParticle` hot: DDMC geometry
belongs in the snapshot, not in every packet.

For a DDMC-to-DDMC leak:

- atomically add the source flux contribution on device;
- for a local target, atomically add the target contribution and retain the
  resident DDMC packet on device;
- for a remote target, put the contribution in `cold.pendingFlux`, set the
  existing pending-flux flag, and terminalize the packet.  Do **not** also add
  it to a local ghost buffer, or the receiver will add it again;
- on the receiving rank, the normal device dispatcher consumes pending flux
  before the packet's next transport event, exactly as `stepImpl()` does now.

At `postStep`, synchronize device DDMC tally buffers to host, add them to the
existing host accumulators, then call `STORM::MPI_reduce_ghost_data()` once
on the flux RHS.  That is the reverse of `MPI_exchange_data` (ghost → owner
`+=`) in `source/monte/utils/MpiExchangeGrid.hpp`.  Halo **fill** of
eligibility/`D` during precompute is `MPI_exchange_data(grid, field, true)`.
There is no DDMC-specific MPI file.  GPU-aware MPI can be investigated later,
but it must not be a prerequisite or an implicit assumption about the MPI
build.

Use device atomics for scalar energy and flux tallies initially.  Expect
contention in highly populated cells; if profiling shows it dominates, reduce
per-team/per-block partial tallies before one atomic add per cell.  Preserve
the same accumulation precision as CPU (currently `double`) for validation.

### 4. Feature ladder and capability predicate

Add a `DDMCDeviceEligible()` predicate which returns both `bool` and a
machine-readable rejection reason.  Record the reason in timing/progress
diagnostics.  It must be evaluated per radiation step, not once at program
startup.

| Phase | Device-supported configuration | Intentionally host/fallback |
| --- | --- | --- |
| A: transport MVP | Static, grey DDMC; no hydro, no polarization, no observer/post-process external sources; local and MPI packet handoff. | Moving Wollaeger correction, multigroup PGRW, detailed diagnostics. |
| B: production mixed mode | A plus IMC/DDMC switching in one wave and supported simple boundaries. | Unmodeled boundary callbacks and observers terminalize safely. |
| C: multigroup | `withMultigroupOpacity`, `ddmcUseMultigroupPGRW`, cutoff/upscatter, group time averages. | Any opacity model lacking a device sampling snapshot. |
| D: hydro | Comoving residence, divergence shift, energy/momentum feedback, static interfaces. | Moving-interface correction until independently ported and tested. |
| E: full supported DDMC | Wollaeger factor, splitting policy, external sources, optional polarization closure/diagnostics. | Compton remains unsupported by DDMC, as it is on CPU. |

The capability predicate must be conservative.  In particular, it should not
claim support merely because `withDDMC` is true: multigroup sampling,
post-processing sources, moving interfaces, polarization, and hydro have
different semantics.  A rejection must retain the current all-CPU DDMC
behavior, not mix partial state changes from device and host.

## Hard technical problems and their resolutions

### Variable face lists and divergent histories

DDMC selects from a variable number of leaks, while packet histories differ
widely between census, absorption, upscatter, local leak, remote leak, and
IMC conversion.  This creates warp divergence and reduces the value of a
one-thread-per-packet kernel in sparse/transition regions.

Start with CSR plus a bounded per-packet loop, which matches the current
executor.  Instrument face-count distribution, events/launch, terminal
fraction, and active-packet occupancy.  Only then consider bucketing packets
by mode, cell degree, or event class.  Avoid sorting by cell every event in
the first version: the extra traffic can cost more than it saves and changes
RNG/event ordering risks.

### Multigroup thermal sampling is not yet a generic device contract

CPU DDMC calls the opacity object to sample a thermal frequency within a
selected group.  The existing spectral IMC device path has group boundaries
and a cumulative emission table, but its in-group sampling is a simple table
operation; it is not automatically equivalent to every
`OpacityT::SampleThermalEnergyInGroup` implementation.

Before phase C, define a device opacity-snapshot interface.  It must expose
the absorption evaluation required by IMC and an inverse-CDF or tabulated
conditional sampler for each `(cell, group)`.  Build this data on CPU in
pre-step, upload it, and use the *same* sampling definition in the CPU
portable reference path.  Do not quietly sample uniformly within group and
call that multigroup parity.

### Moving IMC-DDMC interfaces and packet splitting

The CPU path uses the Wollaeger quadrature-table correction, rejection/
bypass rules, and bounded splitting of large corrected weights.  It also has
special remote-target behavior because a split packet cannot be inserted
directly into a ghost cell.

Port this last.  Upload the correction lookup table and all required
per-face/per-cell velocity data; use the existing first-order validity bounds.
Implement deterministic device splitting with a capacity-safe append/queue,
or terminalize this event to host until that queue exists.  A fixed-size
per-thread split array is unsafe because `ddmcMaxInterfaceSplits` is a runtime
parameter and bursts can exhaust capacity.  Count every device bypass, split,
and host fallback.

### Atomic feedback and reproducibility

The CPU execution order is not deterministic across MPI ranks; GPU atomics
introduce another valid ordering.  Expect bitwise differences, especially in
hot-cell flux and energy tallies.  Acceptance should be conservation and
physics tolerance based, with a deterministic one-packet/one-step differential
harness for exact state comparison.  Never relax the DDMC reciprocity or
weight-conservation tests merely to accommodate offload.

### MPI and device residency

Kokkos kernels cannot invoke MPI.  Remote DDMC leaks therefore end a device
wave for only the leaking packet; local DDMC histories remain resident.  The
manager already makes this distinction for IMC faces.  Preserve the current
remote pending-flux protocol and test partitions where a rank owns no cells.

Do not add GPU-aware MPI in the first implementation.  Its registration,
stream-ordering, and portability requirements are separate from DDMC physics.
Once the host staging version is correct, test a device-aware path behind a
runtime option and require bitwise-equivalent serialized packet state.

### Diagnostics, exceptions, and unsupported callbacks

CPU DDMC records a map-based diagnostic ledger and throws rich errors.  Those
operations cannot occur inside a kernel.  Device code should write compact
numeric error/status codes and atomic counters.  On return, host code converts
an error code into the existing `StormError` context.  For debug/tracing and
detailed interface diagnostics, begin with an explicit CPU fallback; later
add an optional bounded device event-record buffer with overflow detection.

### Memory footprint and mesh rebuilds

The added graph is roughly O(number of directed faces), plus O(cells × groups)
for exact multigroup sampling/tallies.  Measure this separately from the
current IMC tables.  Allocate persistent Kokkos views with geometric growth;
avoid resize/free every time step.  Rebuild the snapshot after mesh rebuild or
DDMC precompute and fence before replacing views used by an active wave.

## Implementation work plan

### Milestone 0 — baseline and interfaces

- Add named timers/counters around CPU DDMC precompute, CPU DDMC event time,
  device upload, device DDMC kernel, terminal copy-back, host fallback, and
  post-step tally synchronization.
- Capture baseline profiles for a DDMC-heavy grey diffusion case, the
  `desmore2012_mc_ddmc` multigroup benchmark, and `ddmc_mpi_zero_cell`.
  Report DDMC steps, leaks, remote resident leaks, event mix, packets/launch,
  and per-rank load balance.
- Add `ddmcGpuEnable` (default false while experimental), a minimum launch
  size inherited from the manager settings, and structured fallback reasons.
- Write POD layout static assertions and host-only unit tests that flatten a
  `CellData`/`FaceLeak` graph and verify every field/rate survives.

Exit criterion: no behavior change in CPU builds and an evidence-based choice
of the first DDMC-heavy benchmark.

### Milestone 1 — snapshot and grey kernel

- Add `DDMCDeviceData`/`DDMCDeviceViews` under `gpu/`, integrated with
  `GreyIMCData` initially to share lifetime and tally synchronization.
- Upload per-cell eligibility/rate/opacity data, CSR leaks, outward normals,
  and face centers after CPU precompute.
- Implement portable `AdvanceDDMC` for grey, static, no-hydro residence,
  absorption, census, local leak, and DDMC-to-IMC conversion.
- Add device counters and atomic material/radiation tally buffers; use the
  existing particle flags and cold state.
- Build a host/device differential test that starts from fixed particles and
  fixed RNG states, advances exactly one DDMC event, and compares status,
  fields, RNG counter, tallies, and selected face.

Exit criterion: serial grey diffusion agrees with CPU within statistical
uncertainty, preserves energy/packet weight, and demonstrates useful device
occupancy without per-event copies.

### Milestone 2 — integrate with the resident executor and MPI

- Generalize `KokkosLocalTransportExecutor::AdvanceWave()` to dispatch IMC or
  DDMC per packet, retaining local transitions on device.
- Preserve the existing terminal compaction and `ApplyTransportEvent()` path.
- Implement the exact remote DDMC pending-flux protocol described above.
- Copy DDMC flux RHS back in `postStep`, then use the unchanged host MPI
  reduction and momentum-feedback code.
- Add GPU-enabled serial and MPI regression entries; build GPU tests only in
  a GPU Kokkos configuration, leaving normal CI unaffected.

Exit criterion: `ddmc_mpi_zero_cell` passes with a forced cross-rank DDMC
leak, including reciprocity, conductance/rate consistency, flux ownership,
and weight conservation.

### Milestone 3 — multigroup DDMC

- Introduce the device opacity sampling snapshot and make the CPU portable
  path consume the same table for differential comparison.
- Port group cutoff, Planck-band selection, DDMC-to-IMC group sampling,
  census reconstruction, compression cutoff, and upscatter.
- Allocate and synchronize group-radiation tally buffers when
  `withEgTimeAvg` is enabled.
- Run the Densmore DDMC regression and interface diagnostics across several
  group cutoffs and compare CPU/GPU event distributions as well as final
  profiles.

Exit criterion: `desmore2012_mc_ddmc` remains within its existing physics
tolerance and CPU/GPU results agree within a predeclared statistical bound;
no silent fallback occurs in the advertised phase-C configuration.

### Milestone 4 — hydro and moving interfaces

- Port comoving frame changes, divergence-based frequency/weight update,
  absorption momentum deposition, and face-flux feedback.
- Port the Wollaeger table lookup/validity rules before enabling moving
  correction; terminalize or disable only this feature until then.
- Implement safe device packet splitting or explicitly route corrected split
  events through the host.
- Add moving-interface and homologous-expansion differential tests, including
  a remote corrected crossing.

Exit criterion: existing DDMC moving-interface validation and radiation
pressure-gradient checks pass in serial and MPI; all correction/bypass/split
counters reconcile between device and host.

### Milestone 5 — optimize based on profiles

- Compare CSR scan, per-cell CDF, and alias sampling only on measured
  high-degree meshes.
- Address atomic hotspots with hierarchical reduction if needed.
- Consider device-aware MPI and GPU precompute only if timings show they are
  significant after the resident packet path is complete.
- Tune maximum inner steps and minimum launch size separately for DDMC-heavy
  workloads; IMC values are not necessarily optimal.

## Verification matrix

| Layer | Required check |
| --- | --- |
| Build | CPU-only compile unchanged; Kokkos HIP/CUDA compile test instantiates DDMC views/kernel; sanitizers/unit tests on host flattening. |
| Exact differential | Fixed packet/RNG one-event tests for: entry, absorption, census, local DDMC leak, DDMC-to-IMC, remote terminalization, cutoff, and each fallback reason. |
| Grey physics | Uniform diffusion (`<r^2> = 6Dt`), constant-field preservation on regular/distorted Voronoi meshes, conservation, and CPU/GPU event-rate distribution. |
| Mesh algebra | For every internal pair, retain `V_i lambda_i->j = V_j lambda_j->i`; compare uploaded rates/targets against CPU precompute exactly. |
| MPI | `ddmc_mpi_zero_cell` on GPU, including zero-owned-cell ranks, asymmetric halos, remote resident leak, pending-flux delivery, and no double count. |
| Multigroup | `desmore2012_mc_ddmc`, the DDMC interface cases/cutoff variants, group-tally checks, and CPU/GPU profile comparison. |
| Hydro/interface | Static equilibrium current, moving Wollaeger tests, homologous redshift/cutoff, pressure-gradient momentum feedback, and remote interface crossing. |
| Performance | End-to-end step time plus CPU precompute, H2D/D2H, kernel, MPI, terminal fraction, tallies, and memory. Claim speedup only from end-to-end measurements. |

For Monte Carlo comparisons, fix rank layout and seeds where practical, use
confidence intervals for solution metrics, and separately assert exact
invariants (packet count/weight where applicable, invalid geometry count,
reciprocity residual, and serialized remote state).

## Non-goals and guardrails

- Do not offload Compton with DDMC; the CPU code already rejects that physics
  combination and this plan does not change the derivation.
- Do not move MPI collectives, boundary virtual calls, `std::map` diagnostics,
  or the opacity object itself into a Kokkos kernel.
- Do not recompute the DDMC graph independently on GPU during the initial
  transport offload; CPU precompute remains the authoritative definition.
- Do not let a device-unsupported feature run an approximate DDMC algorithm.
  Terminalize or select the all-CPU path before a packet state mutation.
- Do not use a host/device transfer after each DDMC event.  It would erase the
  acceleration expected from DDMC's short event histories and defeat the
  existing resident-packet executor.

## Success criteria

The project is ready to advertise GPU DDMC only when the supported feature
matrix has no unreported fallback, serial and MPI validation above pass, and
an end-to-end DDMC-dominated workload shows a reproducible speedup after
including precompute, copies, MPI, and post-step synchronization.  The likely
first bottleneck after transport offload is CPU DDMC precompute or tally/MPI
traffic, not the raw leakage event; profile before expanding scope.
