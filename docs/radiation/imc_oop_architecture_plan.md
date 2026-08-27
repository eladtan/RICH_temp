# Object-Oriented IMC Architecture

## Why the current extraction is not enough

The current reduction of `RadiationIMC.hpp` moved method bodies into `.ipp`
fragments.  That improves navigation, but it does not change ownership:
every algorithm still belongs to `RadiationIMC`, still accesses all of its
state, and still has the same implicit coupling.  Renaming those fragments
would not solve the design problem.

The next refactor should therefore be a composition refactor.  Each major
part of the IMC calculation becomes an object with a narrow input/output
contract and owns the state that it updates.  `RadiationIMC` becomes the
simulation-facing façade and timestep coordinator.

This plan intentionally preserves the existing numerical algorithms and
public `MonteCarloPhysics` API.  It changes who owns an operation, not what
the operation computes.

## Current implementation status

The first composition pass is now implemented. The old
`RadiationIMC_*ipp` fragments have been removed. The façade owns one process
object per route and forwards the compatibility API to those objects:

* `IMCLifecycleProcess` — validation, pre-step preparation, RNG/material
  access, and transport-tally lifecycle;
* `IMCSourceProcess` — thermal/adaptive packet generation and allocation;
* `IMCTransportProcess` — event routing, portable IMC transport, and
  differential checking;
* `IMCRandomWalkProcess` — PGRW eligibility and stepping;
* `DDMCEngine` — DDMC state, leakage, interface events, and diagnostics;
* `ComptonProcess` — matrix setup, Compton events, and correction;
* `IMCObserverProcess` — observer/external-source behavior; and
* `IMCDeviceExecutor` — GPU eligibility and host/device GreyIMC view
  construction.

`RadiationIMC` remains the source-compatible façade and retains the canonical
cell/extensive references, EOS/opacity handles, and passive shared state. The
processes currently use a non-owning owner handle while the numerical code is
being moved; this keeps the first migration behavior-preserving. The next
increment can replace those handles with the explicit context interfaces
below, one route at a time, after differential tests cover each route.

The CPU and GPU Densmore targets and the full RICH CPU and GPU targets have
been rebuilt successfully after this pass. The GPU build still uses the
existing `AdvanceIMC`/Kokkos kernel; moving its view/eligibility bridge into
`IMCDeviceExecutor` keeps that path explicit without changing the device
kernel or its data layout. A small one-rank Densmore CPU run completed
successfully, and its final profile matched the pre-composition executable.

## End-state object graph

```text
RadiationIMC
  |-- IMCState                         configuration and per-step caches
  |-- IMCMaterialView                  canonical cells/extensives/eos/opacity view
  |-- IMCRng                            source and packet random streams
  |-- IMCTallyLedger                    deferred energy/momentum accounting
  |-- IMCSourceProcess                  thermal/adaptive packet creation
  |     `-- IMCExternalSourceProcess    optional post-process face sources
  |-- IMCTransportProcess               host route selection and shared kernel
  |     |-- IMCRandomWalkProcess        PGRW route
  |     `-- DDMCEngine                  DDMC route and interface events
  |-- ComptonProcess                    matrices, scattering, correction
  |-- IMCObserverProcess                optional crossing/polarization sink
  `-- IMCDeviceExecutor                 optional GPU transport bridge
```

The members are values (or small owning smart pointers for optional
features), not inheritance mixins.  A component may use another component's
published interface, but it cannot reach into another component's private
arrays.

## Context objects

Components must not take `RadiationIMC&`.  That simply recreates the monolith
behind a pointer.  They receive explicit contexts instead.

```cpp
template<class Grid, class Cell, class Extensives, class EOS, class Opacity,
         class Parameters, class Boundaries>
struct IMCMaterialContext {
    const Grid& grid;
    std::vector<Cell>& cells;
    std::vector<Extensives>& extensives;
    const EOS& eos;
    Opacity& opacity;
    const Parameters& parameters;
    const Boundaries& energyBoundaries;
};

template<class MaterialContext, class State, class Rng, class Tallies>
struct IMCStepContext {
    MaterialContext material;
    State& state;
    Rng& rng;
    Tallies& tallies;
    double dt;
};
```

There are two intentionally different mutation paths:

* material state changes go through `IMCMaterialUpdater` (including EOS
  synchronization and negative-energy checks);
* radiation/material/momentum increments go through `IMCTallyLedger` and are
  applied once at the existing point in `postStep`.

The device context remains a trivially-copyable view object.  It contains no
MPI communicators, observers, STL maps, strings, or host callbacks.

## Responsibilities and APIs

### `RadiationIMC` façade

`RadiationIMC` keeps the canonical non-owning references to `cells_` and
`extensives_`, plus the shared `eos_` and `opacity_` pointers.  It also keeps
the public compatibility API required by existing managers and tests.

Its implementation should contain only:

1. construction and validation of contexts/components;
2. the `preStep`, `step`, `postStep`, and `onBoundaryResult` entry points;
3. route ordering and feature eligibility;
4. small compatibility getters and setters.

The final `step` should read as orchestration:

```cpp
if (deviceExecutor_.canAdvance(step))
    return deviceExecutor_.advance(step, particle, additions);
if (ddmc_.tryAdvance(step, particle, additions).handled)
    return result;
if (randomWalk_.tryAdvance(step, particle, result))
    return result;
return transport_.advance(step, particle, additions);
```

The order above is illustrative; the existing route order is authoritative
and must be captured by tests before changing it.

### `IMCState`

Owns immutable parameters/traits/sampler and derived group boundaries.  It
also owns per-step coefficient/cache arrays (`factorFleck_`, opacity caches,
cell velocities, spectral scales, and emission CDFs), lifecycle generation
flags, and the existing Compton/PGRW/DDMC state objects where appropriate.

It does not own a second copy of cells or extensives.

### `IMCRng`

Owns the standard RNG, distributions, rank-aware seed, packet seed/stream
counter, and source/packet velocity sampling.  Reseeding must preserve the
current rank offset and call order exactly.

### `IMCTallyLedger`

Owns all pending material/radiation/group-radiation/momentum/total-energy
arrays, plus `Erad_time_avg_` and `Eg_time_avg_`.  Its API is deliberately
small:

```cpp
void reset(std::size_t cellCount);
void addMaterialEnergy(std::size_t cell, double value, bool total);
void addRadiationEnergy(std::size_t cell, double integrated);
void addGroupRadiationEnergy(std::size_t cell, std::size_t group,
                             double integrated);
void addMomentum(std::size_t cell, const Point& value);
void apply(IMCMaterialUpdater& material, double dt);
```

Every route, including DDMC and Compton, emits effects through this ledger.
No route directly edits a pending array owned by another object.

### `IMCSourceProcess`

Owns thermal/adaptive allocation, learned score maps and source allocation
diagnostics.  It returns packets and a value-result summary rather than
writing source diagnostics into the façade.  Its packet creation dependencies
are the material context, `IMCState`, and `IMCRng`.

`IMCExternalSourceProcess` is a separate optional object for post-process face
sources, Planck sampling, face/cell lookup maps, and external-source boundary
events.  This keeps post-processing out of the ordinary source path.

### `IMCTransportProcess`

Owns the portable host event loop around `transport::AdvanceIMC`, host opacity
policy, transport coefficient views, and differential-harness comparison.
It does not own DDMC/PGRW/Compton caches.  It receives route results and
records effects through the tally ledger.

### `IMCRandomWalkProcess`

Owns the random-walk policy, eligibility/opacity cache, reusable geometry
data, and PGRW counters.  Its only particle-facing operation is:

```cpp
bool tryAdvance(IMCStepContext&, Particle&, StepResult&);
```

### `DDMCEngine`

Owns `DDMCState`, leakage/interface sampling, resident and remote events,
MPI ghost-face coordination, flux RHS, momentum feedback, and DDMC counters.
It is the only object that combines DDMC geometry/sampling/ghost-exchange
utilities with material state and particles.

The engine exposes a device-independent event result:

```cpp
struct DDMCRouteResult {
    bool handled = false;
    StepResult functionality{};
    std::vector<Particle> additions;
};
```

The event result is the future boundary between CPU DDMC and a flat GPU DDMC
kernel.  Diagnostics and TSV rendering live in `DDMCDiagnostics`, not in the
route algorithm.

### `ComptonProcess`

Owns Compton matrices, CDFs, event data, risky-particle splitting,
reconciliation, and end-of-step correction.  Matrix algebra that has no
material or particle dependency belongs in a private `ComptonCorrection`
value object.  The public phases are:

```cpp
void precompute(IMCStepContext&, double sourceDt);
std::vector<Particle> generateSources(IMCStepContext&);
bool tryScatter(IMCStepContext&, Particle&, StepResult&);
void correctEndOfStep(IMCStepContext&, std::vector<Particle>&);
```

### `IMCObserverProcess`

Owns the optional observer, polarization setup, crossing bookkeeping, and
observer diagnostics.  The hot transport path receives a compact nullable
sink/policy; it never performs a virtual call or accesses observer STL state
on a device.

### `IMCDeviceExecutor`

Owns Kokkos runtime/data, generation tracking, host/device view construction,
and synchronization for the shared GPU IMC kernel.  It depends on state and
tallies through POD views and does not know about observers or DDMC internals.

## Field ownership

| Existing data | New owner |
|---|---|
| `cells_`, `extensives_`, `eos_`, `opacity_` | `RadiationIMC` canonical references/pointers; exposed only through contexts |
| `parameters_`, `traits_`, `positionSampler_`, `energyBoundaries_` | `IMCState` |
| Fleck/opacity/velocity/spectral/emission coefficient arrays | `IMCState` coefficient cache |
| RNG, rank seed, stream counters | `IMCRng` |
| pending and time-integrated tallies | `IMCTallyLedger` |
| adaptive source maps/budgets/summaries | `IMCSourceProcess` |
| external-source maps and definitions | `IMCExternalSourceProcess` |
| random-walk cache and counter | `IMCRandomWalkProcess` |
| DDMC arrays, maps, events, flux, feedback | `DDMCEngine` + `DDMCState` |
| DDMC event counters/TSV map | `DDMCDiagnostics` |
| Compton cache, events, correction state | `ComptonProcess` + `ComptonCorrection` |
| observer/polarization/crossings | `IMCObserverProcess` |
| Kokkos/GreyIMC runtime and data | `IMCDeviceExecutor` |

The façade may retain read-only compatibility accessors, but it must not
retain duplicate route-specific arrays merely to implement those accessors.

## File layout (no `.ipp` fragments)

Each file contains a real class with its own state and methods.  There is no
file whose only purpose is to paste `RadiationIMC::method` definitions into a
large class.

```text
radiation/
  RadiationIMC.hpp                 # façade and orchestration only
  imc/
    IMCContexts.hpp
    IMCState.hpp
    IMCRng.hpp
    IMCMaterialUpdater.hpp
    IMCTallyLedger.hpp
  source/
    IMCSourceProcess.hpp
    IMCExternalSourceProcess.hpp
  transport/
    IMCTransportProcess.hpp
  random_walk/
    IMCRandomWalkProcess.hpp
  ddmc/
    DDMCState.hpp
    DDMCDiagnostics.hpp
    DDMCEngine.hpp
  compton/
    ComptonProcess.hpp
    ComptonCorrection.hpp
  observer/
    IMCObserverProcess.hpp
  gpu/
    IMCDeviceExecutor.hpp
```

The classes are templates because cell, grid, EOS, opacity, and group count
are compile-time types.  Definitions remain in the class headers where
required by template instantiation; this is different from `.ipp` fragments
because the definitions are members of the component that owns the state.

## Migration sequence

1. **Freeze a regression baseline.**  Record CPU/GPU Densmore, PGRW, DDMC,
   Compton, observer, and differential-harness results, including fixed-seed
   packet counts and tally checksums.
2. **Introduce contexts and `IMCMaterialUpdater`.**  Keep behavior identical;
   initially the façade constructs contexts and forwards existing operations.
3. **Move `IMCRng` and `IMCTallyLedger`.**  Replace direct array/RNG access in
   all routes.  Verify random draw order and tally application order.
4. **Move source and external-source processes.**  Return value-result
   summaries and remove source state from the façade.
5. **Move PGRW and transport.**  Preserve the exact route ordering and
   differential replay path.
6. **Move Compton.**  Keep matrix arithmetic and correction tolerances
   byte-for-byte equivalent before any cleanup.
7. **Move DDMC.**  First make CPU `DDMCEngine` pass interior, boundary,
   moving-interface, remote-face, and diagnostics regressions.  Only then
   define the flat device mirror for DDMC offload.
8. **Move observer and GPU executor.**  End with a façade that contains no
   route-specific implementation or caches.
9. **Delete all `RadiationIMC_*.ipp` files.**  Their code must already be
   owned by the component classes; deleting them is a check that no textual
   extraction remains.

## Acceptance criteria

The refactor is complete only when:

* `RadiationIMC.hpp` is an orchestration façade (target: under 500 lines,
  excluding public compatibility documentation);
* no component stores a `RadiationIMC&` or reaches into another component's
  private state;
* cells and extensives have exactly one canonical storage location;
* all deferred effects flow through `IMCTallyLedger`;
* CPU and GPU builds compile with and without Compton, PGRW, and DDMC;
* fixed-seed packet/tally checksums match the pre-refactor baseline;
* DDMC diagnostics TSV output is unchanged;
* no `.ipp` implementation fragments remain; and
* the device-facing contexts remain POD/Kokkos-safe and contain no host-only
  objects.

The implementation should be delivered as one focused commit per component
move, with a CPU/GPU build and the relevant numerical smoke test after every
commit.  This keeps a regression attributable and makes reverting one
component possible without reverting the entire architecture.
