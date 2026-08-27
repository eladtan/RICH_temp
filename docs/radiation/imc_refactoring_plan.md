# Reducing `RadiationIMC` Before DDMC Work

> **Architecture revision:** the intermediate `.ipp`-fragment extraction is
> not the desired end state.  The implementation should follow
> [`imc_oop_architecture_plan.md`](imc_oop_architecture_plan.md), which moves
> behavior and state into composed objects and removes the fragments entirely.

## Objective

`source/monte/radiation/RadiationIMC.hpp` is about 10,069 lines.  It currently
owns the IMC life cycle, but also implements material access, packet RNG,
thermal-source sampling, deferred tallies, PGRW, DDMC, Compton scattering and
correction, observers, post-process sources, diagnostics, and GPU data setup.
Adding device DDMC to that class would make its ownership and host/device
boundaries still less clear.

Refactor it into a small `RadiationIMC` orchestrator before adding DDMC device
physics.  The class should retain the `MonteCarloPhysics` entry points,
configuration, and route selection; independent algorithms and their state
move to focused helper components.  This plan changes organization, not IMC or
DDMC physics.  Every phase preserves present CPU results before the next phase.

## Existing seams


| Current responsibility                               | Existing location                         | Extraction target                                                        |
| ---------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------ |
| Detection traits and cell/member adapters            | header detail namespace                   | `imc/IMCConcepts.hpp`, `IMCMaterialAccess.hpp`                           |
| Pending material/radiation/momentum tallies          | ~1904–1989                                | `imc/IMCTransportTallies.hpp`                                            |
| Random-walk cache and route                          | ~2266–2671                                | `random_walk/RandomWalkController.hpp`                                   |
| DDMC cache, events, interface, feedback, diagnostics | ~2672–4714                                | `ddmc/DDMCState.hpp`, `DDMCEngine.hpp`, `DDMCDiagnostics.hpp`            |
| Compton cache, events, algebra, correction           | ~4715–7088                                | `compton/ComptonEngine.hpp`, `ComptonCorrection.hpp`                     |
| Observer and post-process external sources           | ~7089–7128 and helpers                    | `observer/IMCObserverController.hpp`, `source/ExternalSourceSampler.hpp` |
| Thermal/adaptive emission and packet creation        | ~8243–9340                                | `source/IMCSourceGenerator.hpp`, `IMCSourceControl.hpp`                  |
| Shared particle-event transport                      | `transport/AdvanceIMC.hpp`                | retain as the portable event kernel                                      |
| GPU data/view ownership                              | GPU members and `GetDeviceTransportViews` | `gpu/IMCDeviceTransportBridge.hpp`                                       |


The first extraction should follow these domains rather than split the header
at arbitrary line boundaries.

## End-state ownership

```text
RadiationIMC (life cycle and route policy)
 ├─ IMCState / IMCMaterialAccess
 ├─ IMCRNG
 ├─ IMCTransportTallies
 ├─ IMCSourceGenerator + IMCSourceControl
 ├─ RandomWalkController
 ├─ DDMCEngine + DDMCState + DDMCDiagnostics
 ├─ ComptonEngine + ComptonCorrection
 ├─ IMCObserverController
 └─ IMCDeviceTransportBridge
```

`RadiationIMC` should only:

1. Validate configuration and construct helpers.
2. Keep the public API and lightweight diagnostic forwarding.
3. Define time-step ordering: state refresh, source creation, route selection,
  deferred updates, host/device synchronization, and post-processing.
4. Decide which route is eligible (device IMC, DDMC, PGRW, host IMC).
5. Own and wire helpers.

It should not contain detailed route algorithms or directly own large
route-specific arrays.  A practical target is under 1,500 lines, but clear
ownership and testability are more important than a precise count.

### Core references stay in the orchestrator

`cells_` and `extensives_` should remain direct non-owning references in
`RadiationIMC`.  They are the simulation's canonical material storage and are
needed to construct the base-class/lifecycle contexts; moving them into a
helper would make ownership less obvious and force the orchestrator to recover
the same references for every callback.  `eos_` and `opacity_` may likewise
remain the shared pointers owned by `RadiationIMC`.

Helpers should receive short-lived `IMCReadContext` or
`IMCMutationContext` views containing those references.  `IMCState` therefore
holds derived configuration and caches, not a second copy of the cell or
extensive references.  This gives one canonical owner while keeping helpers
independent of the full `RadiationIMC` type.

## Interface rules

Do not replace one monolith with helpers that take `RadiationIMC &`.  Introduce
narrow contexts instead:

```cpp
template<class GridT, class CellT, class ExtensivesT, class EOST,
         class OpacityT, class ParametersT, class BoundariesT>
struct IMCReadContext {
    const GridT& grid;
    std::vector<CellT>& cells;
    std::vector<ExtensivesT>& extensives;
    const EOST& eos;
    OpacityT& opacity;
    const ParametersT& parameters;
    const BoundariesT& energyBoundaries;
};
```

- `IMCReadContext`: immutable configuration plus cell/material reads.
- `IMCMutationContext`: `IMCMaterialAccess` and `IMCTransportTallies`; routes
record changes here rather than update cell state in multiple places.
- `IMCRNG`: the only owner of source and packet RNG policy, including rank
offsets and counter-RNG initialization.
- `IMCDeviceContext`: POD/Kokkos views only—no observers, MPI state, strings,
`std::map`, or host callbacks.

Hot algorithms remain template/policy code in headers.  Avoid virtual calls,
`std::function`, or type erasure in the per-particle path.

## Proposed layout

```text
source/monte/radiation/
  imc/
    IMCConcepts.hpp
    IMCMaterialAccess.hpp
    IMCRNG.hpp
    IMCState.hpp
    IMCTransportTallies.hpp
    IMCSourceControl.hpp
  source/
    IMCSourceGenerator.hpp
    ExternalSourceSampler.hpp
  random_walk/
    RandomWalkController.hpp
  ddmc/
    DDMCState.hpp
    DDMCEngine.hpp
    DDMCDiagnostics.hpp
    DDMCDeviceData.hpp              # added only when device work begins
  compton/
    ComptonEngine.hpp
    ComptonCorrection.hpp
  observer/
    IMCObserverController.hpp
  gpu/
    IMCDeviceTransportBridge.hpp
  transport/
    AdvanceIMC.hpp                  # existing shared kernel
```

These are ownership boundaries, not a mandate for many tiny classes.  Keep
tightly coupled scalar math together; extract an object when it owns state, an
independently testable algorithm, or a distinct host/device representation.

## Phased refactor



### 0. Build a regression baseline

Before moving code, capture a concise test matrix:

- Densmore CPU, GPU, and single-node GPU: final temperature L1, energy
closure, particle count, acceleration counters, and exit status.
- A PGRW case and a DDMC case, including existing DDMC TSV diagnostics.
- A Compton correction case.
- Debug/release builds with and without `STORM_WITH_GPU`, including
`gpu/compile_test.cpp`.

Run `STORM_IMC_DIFF` for host transport changes.  Store concise metrics only;
do not commit generated profiles, plots, binary executables, or core files.

### 1. Extract concepts, material access, and RNG

Move the detection traits and ID helpers from `radiation_imc_detail` to
`IMCConcepts`.  Move density/internal-energy/radiation-energy access,
synchronization, and negative-energy checks to `IMCMaterialAccess`.  Move
`randomUnitOpen`, packet initialization, rank-aware seeding, and velocity
sampling to `IMCRNG`.

Keep forwarding methods in `RadiationIMC` temporarily.  This is a mechanical
change that exposes dependencies without modifying the transport call graph.

### 2. Centralize deferred tallies

Extract `pendingMaterialEnergy_`, `pendingTotalEnergy_`, `pendingMomentum_`,
`pendingRadiationEnergy_`, `pendingGroupRadiationEnergy_`, and their
reset/tally/apply methods into `IMCTransportTallies`.

Rules:

- Every route records energy and momentum through the same tally interface.
- Preserve present accumulation and application order; do not introduce new
reductions, atomics, or synchronization in this phase.
- Make host/device refresh explicit. `GreyIMCData` must not rebuild as a side
effect of an unrelated tally operation.

This is the key shared accounting boundary for future GPU DDMC.

### 3. Extract source control and source generation

Move adaptive score maps, learned budgets, group scores, and allocation
diagnostics to `IMCSourceControl`.  Move thermal/adaptive allocation,
`generateParticles`, and `generateSingleParticle` to `IMCSourceGenerator`.

Inputs are `IMCReadContext`, `IMCMaterialAccess`, `IMCRNG`, and source control;
outputs are packets and a source-allocation summary.  The generator must not
depend on MPI managers, DDMC internals, or Kokkos views.

Verify exact packet allocation and random sequence with a fixed seed when
adaptive controls are disabled.

### 4. Move PGRW behind a route controller

Extract the random-walk cache and `tryRandomWalkStep` into
`RandomWalkController`.  Its core operation is equivalent to:

```cpp
bool TryAdvance(Packet&, IMCMutationContext&, IMCRNG&, StepResult&);
```

The controller owns eligibility/cache vectors and PGRW counters.  This
lower-risk route validates the controller pattern before DDMC is moved.

### 5. Split Compton into state, event, and correction layers

Create `ComptonEngine` for per-cell matrices/CDFs, source creation, packet
scattering, and reconciliation.  Create `ComptonCorrection` for matrix solves,
projection, residual checks, and end-of-step energy correction.

Expose phases explicitly:

1. `Precompute(sourceDt)`
2. `TryScatter(packet, mutationContext, rng)`
3. `GenerateSources(fullDt)`
4. `CorrectEndOfStep(fullDt, particles, mutationContext)`

This phase does not port Compton to device; it makes its later device boundary
independent of DDMC work.

### 6. Extract DDMC before implementing DDMC on GPUs

Create `DDMCState` first, moving cache arrays, local-index mappings, counters,
and compact diagnostics out of `RadiationIMC`.  Then move into `DDMCEngine`:

- cache precomputation;
- resident/remote leakage and event sampling;
- IMC-to-DDMC and DDMC-to-IMC interfaces;
- flux RHS and momentum feedback;
- MPI ghost exchange coordination;
- diagnostic aggregation and TSV rendering.

Keep `DDMCGeometry`, `DDMCSampling`, `DDMCTypes`, and the
Wollaeger interface as lower-level dependencies.  Halo fill and ghost-flux
reduce use `MPI_exchange_data` / `MPI_reduce_ghost_data` on the tessellation
maps (`MpiExchangeGrid.hpp`), not a DDMC MPI helper.  `DDMCEngine` should be the
only layer combining those pieces with material state and packets.

Use an API like:

```cpp
DDMCRouteResult TryAdvance(Packet&, IMCMutationContext&, IMCRNG&);
void BuildStepData(const IMCReadContext&, double fullDt);
void ApplyEndOfStepFeedback(IMCMutationContext&, double fullDt);
```

Store stable cell IDs at the engine boundary and local indices internally.
This makes `DDMCState` the natural source for a later compact
`DDMCDeviceData` layout.

### 7. Extract observer and post-process features

Move crossing bookkeeping and external face-source sampling to optional
controllers invoked at well-defined events.  The production packet path uses a
compile-time no-op policy or nullable lightweight sink, never virtual dispatch
or `std::function` in the inner loop.

### 8. Isolate the GPU bridge and simplify orchestration

Move `KokkosRuntime`, `GreyIMCData`, generation tracking, view construction,
and synchronization to `IMCDeviceTransportBridge`.  It receives common state
and tallies but no DDMC-specific host policy at first.

The final `stepImpl` should read as route selection, not route implementation:

```cpp
if (deviceBridge.CanAdvance(packet)) return deviceBridge.Advance(...);
if (ddmc.TryAdvance(packet, ...).handled) return result;
if (randomWalk.TryAdvance(packet, ...)) return result;
return transport::AdvanceIMC(packet, views, opacityPolicy, tallyPolicy);
```

The existing route order remains authoritative; this sketch defines ownership,
not a behavioral change.

## Field-by-field disposition

The following is the complete current private-data inventory.  The field names
are intentionally preserved during the first move so a diff can distinguish an
ownership change from a physics change.

| Current fields | New owner | Disposition |
| --- | --- | --- |
| `cells_`, `extensives_` | `RadiationIMC` (directly) | Keep as the canonical non-owning references; construct narrow read/mutation contexts for helpers. |
| `eos_`, `opacity_` | `RadiationIMC` (directly) | Keep shared ownership here; helpers receive references, not a back-pointer to IMC. |
| `parameters_`, `traits_`, `positionSampler_` | `IMCState` and `IMCSourceGenerator` | Configuration remains immutable after construction; sampler is source-generator state. |
| `energyBoundaries_` | `IMCState` | Keep as the canonical group-boundary value shared by source, Compton, DDMC, and transport. |
| `factorFleck_`, `planckOpacities_`, `scatteringOpacities_` | `IMCState` / opacity cache | Move together as per-cell step coefficients; expose read-only views. |
| `Erad_time_avg_`, `Eg_time_avg_` | `IMCTransportTallies` | Own time-integrated radiation and group-energy accumulators and compatibility accessors. |
| `lastSourcePhotonsPerCell_`, `lastSourceAllocationSummary_`, `lastGroupSamplingDiagnostics_` | `IMCSourceGenerator` | Source-generation results; return a summary instead of storing them in the orchestrator where possible. |
| `comptonData_`, `comptonGroupCenters_`, `comptonGroupWidths_` | `ComptonState` | Per-cell/group Compton cache and group geometry. |
| `comptonMatrixGen_`, `comptonGroupsInitialized_`, `comptonDataReusableInPreStep_`, `comptonRiskPrecomputeDt_` | `ComptonEngine` / `ComptonState` | Backend lifetime and cache-validity state; no Compton flags in `RadiationIMC`. |
| `observer_` | `IMCObserverController` | Controller owns observer and polarization configuration; IMC keeps an optional façade. |
| `postProcessExternalSourceMode_`, `postProcessExternalSources_` | `ExternalSourceSampler` | External-source definitions and enabled state. |
| `postProcessExternalSourceLocalCellIndices_`, `postProcessExternalSourceFaceIndex_`, `postProcessExternalSourceInteriorCellIDs_` | `ExternalSourceSampler` | Derived face/cell lookup maps; rebuild only when source definitions change. |
| `preStepInitialized_` | `IMCState` | Lifecycle flag for a step/cache generation, not a transport-route field. |
| `rng_`, `dist_` | `IMCRNG` | Own the standard source RNG and uniform distribution. |
| `particleRngSeed_`, `sourceRngStreamCounter_` | `IMCRNG` | Preserve exact seed/stream semantics and draw ordering. |
| `creationRank_`, `creationRankCached_` | `IMCRNG` | Rank identity cache used for deterministic streams. |
| `scratchDecomposition_` | `IMCSourceGenerator` | Reusable position-sampler scratch storage; keeps its current capacity-reuse behavior. |
| `pendingMaterialEnergy_`, `pendingTotalEnergy_`, `pendingMomentum_` | `IMCTransportTallies` | Deferred material, total-energy, and momentum updates. |
| `transportCellVelocities_`, `spectralAbsorptionScale_`, `thermalEmissionCdf_` | `IMCState` / transport coefficient cache | Per-step arrays consumed by transport and source policies; group them in a read-only coefficient view. |
| `pendingRadiationEnergy_`, `pendingGroupRadiationEnergy_` | `IMCTransportTallies` | Deferred radiation and spectral-group tallies; preserve current apply order. |
| `gpuRuntime_`, `gpuData_` | `IMCDeviceTransportBridge` | GPU runtime and GreyIMC data lifetime; only the bridge sees Kokkos types. |
| `gpuTransportEnabled_`, `gpuGridBuildGeneration_` | `IMCDeviceTransportBridge` | Device-route enablement and rebuild generation; expose `CanAdvance`/`Refresh` instead of flags. |
| `randomWalk_` | `RandomWalkController` | Own the existing random-walk policy object. |
| `rwCellEligible_`, `rwCellTotalOpacity_`, `rwCellData_` | `RandomWalkController` | PGRW eligibility and per-cell cache arrays. |
| `rwStepCount_` | `RandomWalkController` | Route counter, forwarded through a diagnostic accessor. |
| `ddmcCellData_` | `DDMCState` | Per-cell DDMC coefficients/data. |
| `ddmcPointEligible_`, `ddmcPointDiffusionCoefficient_`, `ddmcPointSigmaDiffusion_`, `ddmcPointSigmaParticleGate_` | `DDMCState` | Per-point eligibility and diffusion/gating arrays. |
| `ddmcPointGroupCutoff_`, `ddmcPointVelocity_`, `ddmcPointCellID_` | `DDMCState` | Point-to-cell/group/velocity maps; retain stable-ID conversion at the engine boundary. |
| `ddmcFluxRhsIntegrated_` | `DDMCState` / `IMCTransportTallies` | DDMC flux RHS is engine state; its material/momentum effect is emitted through tallies. |
| `ddmcLeakReciprocityResidualMax_`, `ddmcLeakReciprocityCheckCount_` | `DDMCDiagnostics` | Reciprocity quality metrics. |
| `ddmcMomentumFeedbackCount_`, `ddmcMomentumMatrixFallbackCount_` | `DDMCDiagnostics` | Momentum-feedback counters; the engine owns updates, diagnostics owns reporting. |
| `ddmcResidentLeakCount_`, `ddmcTransportLeakCount_`, `ddmcRemoteResidentLeakCount_`, `ddmcMPIFaceFluxReductionCount_` | `DDMCDiagnostics` | Local/remote leak and MPI-reduction counters. |
| `ddmcLeakInvalidGeometryCount_`, `ddmcUnsupportedBoundaryFaceCount_` | `DDMCDiagnostics` | Geometry and boundary fallback counters. |
| `ddmcInterfaceIncidentCount_`, `ddmcInterfaceAdmittedCount_`, `ddmcInterfaceReflectedCount_`, `ddmcInterfaceGuAppliedCount_`, `ddmcInterfaceGuFallbackCount_` | `DDMCDiagnostics` | IMC/DDMC interface event counters. |
| `ddmcInterfaceBypassCount_`, `ddmcInterfaceSplitPacketCount_`, `ddmcInterfaceFluxTallyCount_`, `ddmcInterfaceMinimumMu_` | `DDMCDiagnostics` | Interface bypass/split/tally metrics and minimum-angle diagnostic. |
| `ddmcDiagnosticEvents_` | `DDMCDiagnostics` | Host-only keyed event map and TSV formatting; never part of device state. |
| `ddmcExternalSourceCandidateFaceCount_`, `ddmcExternalSourceAcceleratedFaceCount_`, `ddmcExternalSourceExplicitFallbackFaceCount_`, `ddmcExternalSourceInteriorExcludedCellCount_` | `DDMCDiagnostics` | External-source acceleration/fallback counters. |
| `ddmcExternalSourceThermalizationCount_`, `ddmcExternalSourceStayDDMCCount_`, `ddmcExternalSourceToIMCCount_` | `DDMCDiagnostics` | External-source route counters. |
| `ddmcExternalSourceThermalizedEnergy_`, `ddmcExternalSourceToIMCEnergy_`, `ddmcExternalSourceMinimumFaceOpticalDepth_` | `DDMCDiagnostics` | External-source energy and optical-depth metrics. |
| `ddmcStepCount_`, `ddmcLeakCount_`, `ddmcCensusCount_`, `ddmcUpscatterCount_`, `ddmcFallbackCount_` | `DDMCDiagnostics` | Main DDMC route counters, forwarded by compatibility getters. |
| `ddmcMovingInterfaceBypassCount_`, `ddmcMovingInterfaceMaxFactor_` | `DDMCDiagnostics` | Moving-interface safety metrics. |
| `adaptiveSourceScores_`, `adaptiveSourceScoresEnabled_`, `adaptiveSourceStrength_`, `adaptiveSourceMaxFactor_` | `IMCSourceControl` | Cell-score map and adaptive source policy. |
| `adaptiveSourceLearnedReserveFrac_`, `adaptiveSourceLearnedMinFactor_`, `adaptiveSourceObserverBudgetMultiplier_` | `IMCSourceControl` | Learned-budget and observer-budget policy. |
| `adaptiveSourceLearnedMinPhotons_`, `adaptiveSourceLearnedMaxPhotons_`, `adaptiveSourceScorePower_` | `IMCSourceControl` | Learned allocation bounds and score transform. |
| `adaptiveSourceCellGroupScores_`, `adaptiveSourceCellGroupScoresEnabled_`, `adaptiveGroupStrength_`, `adaptiveGroupPdfFloor_`, `adaptiveGroupMaxBias_`, `adaptiveGroupMaxWeightCorrection_` | `IMCSourceControl` | Group-dependent adaptive source policy and correction limits. |
| `sourceEmissionControlEnabled_`, `sourceEmissionUseLearnedScores_`, `sourceEmissionIncludeUniformBase_` | `IMCSourceControl` | Source-emission mode flags. |
| `sourceEmissionBaseMultiplier_`, `sourceEmissionLearnedBoostFactor_`, `sourceEmissionLearnedExtraBudget_` | `IMCSourceControl` | Source-emission integer budgets and boost factors. |

The public nested value types (`SourceAllocationSummary`,
`GroupSamplingDiagnostics`, Compton result types, DDMC event key/accumulator,
and `PostProcessExternalSource`) should move beside the component that owns
them.  Keep type aliases and compatibility aliases in `RadiationIMC` until
external users have migrated.

## Estimated file sizes

These are implementation-line estimates after extraction, including comments,
templates, small forwarding APIs, and tests where noted.  They are ranges, not
requirements; a helper should not be padded to meet a number.

| File | Estimated lines | Main contents |
| --- | ---: | --- |
| `radiation/RadiationIMC.hpp` | 1,200–1,600 | Constructor, public façade, lifecycle, route selection, compatibility forwarding |
| `radiation/imc/IMCConcepts.hpp` | 250–450 | Detection traits, aliases, ID/group utility concepts |
| `radiation/imc/IMCMaterialAccess.hpp` | 300–550 | Cell/extensive reads, EOS conversions, synchronization and validation |
| `radiation/imc/IMCRNG.hpp` | 180–320 | Rank seeding, packet/source streams, velocity sampling |
| `radiation/imc/IMCState.hpp` | 350–650 | Common references, group/coefficient caches, lifecycle state |
| `radiation/imc/IMCTransportTallies.hpp` | 300–500 | Deferred arrays, tally methods, reset/apply, host/device handoff |
| `radiation/imc/IMCSourceControl.hpp` | 250–450 | Adaptive maps, emission flags/budgets, allocation diagnostics |
| `radiation/source/IMCSourceGenerator.hpp` | 1,000–1,600 | Thermal/adaptive packet generation and source summaries |
| `radiation/source/ExternalSourceSampler.hpp` | 450–750 | Face-source maps, Planck sampling, boundary injection |
| `radiation/random_walk/RandomWalkController.hpp` | 650–950 | PGRW cache, eligibility, route attempt, counters |
| `radiation/ddmc/DDMCState.hpp` | 500–800 | Compact DDMC arrays, maps, flux state, route counters |
| `radiation/ddmc/DDMCEngine.hpp` | 1,700–2,500 | Precompute, leakage, interfaces, feedback, MPI coordination |
| `radiation/ddmc/DDMCDiagnostics.hpp` | 350–600 | Event map, counters, TSV formatting and compatibility snapshots |
| `radiation/ddmc/DDMCDeviceData.hpp` | 250–500 | Future flat device mirrors; do not implement until CPU extraction is stable |
| `radiation/compton/ComptonEngine.hpp` | 1,900–2,800 | Cache build, event sampling, Compton particles and reconciliation |
| `radiation/compton/ComptonCorrection.hpp` | 700–1,100 | Matrix solve, projection, residual and end-step correction |
| `radiation/observer/IMCObserverController.hpp` | 250–450 | Observer ownership, crossings, polarization setup |
| `radiation/gpu/IMCDeviceTransportBridge.hpp` | 350–650 | Kokkos runtime/data, views, generation and synchronization |
| `tests/imc/*` (new tests) | 800–1,400 | Focused helper tests and numerical regression harnesses |

The extracted implementation totals roughly 9,000–14,000 lines because
interfaces, tests, and explicit ownership documentation add some code.  The
benefit is that the central class drops by roughly 85–90%, while each change
touches a bounded component.  The numbers should be revisited after the
mechanical moves; reducing duplication is a later cleanup, not a prerequisite.

## Review and validation rules

Make one focused extraction per commit, in this order: concepts/material/RNG,
tallies, source generation, PGRW, Compton, DDMC state, DDMC engine,
diagnostics, observer/source features, then GPU bridge.

For every commit:

- move code mechanically before renaming it;
- preserve public forwarding wrappers until callers are migrated;
- build CPU and GPU configurations;
- run the smallest relevant phase-0 numerical case;
- inspect include direction and template instantiations for new cycles;
- keep source RNG draws, per-packet RNG streams, tally timing, and MPI
ownership unchanged.



## Ready-for-DDMC-offload gate

Begin DDMC device work only when:

- `RadiationIMC` contains no DDMC cache vectors or DDMC event algebra;
- `DDMCEngine` has CPU regressions for interior/boundary leakage, interfaces,
moving interfaces, and MPI ghost faces;
- `DDMCState` has documented stable-ID and compact-array representations;
- all DDMC material/radiation/momentum effects flow through
`IMCTransportTallies`;
- a device-independent DDMC event/result representation exists; and
- CPU IMC, GPU IMC, PGRW, Compton, and DDMC still meet the phase-0 baselines.

This gives a device DDMC kernel a compact state and portable event API, rather
than coupling it directly to a 10,000-line host physics class.
