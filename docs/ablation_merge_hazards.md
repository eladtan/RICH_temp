# Ablation Merge — Hazard Report

**Merge commit:** `534df787` on branch `submodules_integration`
**Date:** 2026-06-25

---

## Phase 1: RICH-proper conflict resolutions (14 files)

### Files requiring manual inspection

| File | Resolution | Inspect? |
|------|-----------|----------|
| `source/3D/monte/MonteCarloManager3D.cpp` | Kept HEAD's `MonteCarloConfig` constructor and `RDMAMonteCarloManager` base; dropped ablation's `progressOpacityPtr_`/`progressCellsPtr_` hooks (they targeted the removed `MonteCarloManager` API) | **YES** — ablation's progress hooks were dropped |
| `source/3D/monte/MonteCarloManager3D.hpp` | Kept HEAD's `RDMAMonteCarloManager` delegations; added `GetPreStepParticleCount`; removed duplicate pure virtuals | **YES** — interface may have lost ablation features |
| `source/3D/gravity/DistributedGravityCalculator.hpp` | Kept ablation's `topNodesOfRanks` member name; dropped HEAD's rename to `boundingBoxesOfRanks`/`tempRanks` | **YES** — verify member name consistency with usage |
| `source/mpi/mpi_commands.hpp` | Kept HEAD's empty stub (implementations in mpi_utils submodule); dropped ablation's inline duplicate implementations | **YES** — ablation added functions here that now live in mpi_utils; verify nothing is missing |
| `source/mpi/mpi_commands_3d.hpp` | Kept ablation's `compact_to_indices_exact_order` helper + memory_profile include; used HEAD's include paths | Probably fine |
| `source/newtonian/three_dimensional/simulation/Simulation.cpp` | Combined HEAD's `vtune_stop()` with ablation's hydro dt logging | Probably fine |
| `source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.cpp` | Kept HEAD's `dumpCost()` | Probably fine |
| `regression_tests/test_report.tex` | Used ablation's reorganized structure with HEAD's additions | Low risk |
| `build_rich.sh` | Merged both flags (`--timing`, `--montecarlo-polarization`) | Low risk |
| `config/compiler_flags.cmake` | Kept both `TIMING` and `MEMORY_PROFILE` blocks | Low risk |
| `runs/BaseTDE/test.cpp`, `runs/BaseTDECompton/test.cpp` | Style-only (`size_t` vs `std::size_t`) | No action needed |
| `regression_tests/cases/moving_slab_mc/test.cpp` | Kept ablation's load-balance setup | Low risk |

---

## Phase 2: Submodule porting hazards

### STORM (highest risk) — 13+ files ported

| Hazard | Severity | Details |
|--------|----------|---------|
| MPI serialization break | **HIGH** | `sourceCellID` added to `Particle::dump/load`. All MPI ranks must use matching code or transfers will desynchronize. |
| `STORMError` include path | MEDIUM | Files include `"monte/STORMError.hpp"` but the actual file is `StormError.hpp`. Case-sensitive systems may fail. |
| `progressCellsPtr_`/`progressOpacityPtr_` hooks incomplete | MEDIUM | Hooks are present in `RDMAMonteCarloManager` but the RICH-specific cell/opacity printing was not ported — only `cellIndex` is logged when pointer is set. |
| `MovingSideTemperature` verbose output | LOW | Contains `std::cout` diagnostics (left velocity, fluid energy, total weight) — may be noisy in production. |
| DDMC progress counters | MEDIUM | `[Progress]` DDMC/RW totals only show nonzero values if the host physics class overrides new `MonteCarloPhysics` virtuals. RICH's DDMC physics needs to override them. |
| `LorentzTransformation` energy boundaries | LOW | Uses boundary's `energyBoundaries` when multigroup; may differ slightly from RICH's `ComputationalCell3D::energyBoundaries` global. |
| `RigidBoundary` reflection semantics changed | MEDIUM | Uses single-sided reflection + small inward nudge (`1e-12`) instead of old double-distance + center nudge (`1e-6`). Intentional but may affect edge-case particle positioning. |
| `MONTECARLO_POLARIZATION` not in STORM CMake | LOW | Still defined by RICH parent build; STORM's standalone `CMakeLists.txt` has no equivalent toggle. |
| `GetNeighbors` buffer overload not available | LOW | STORM tessellation API uses range-for over `GetNeighbors(i)`, no buffer overload. Minor performance difference. |

**New files to inspect:**
- `boundary/MovingSideTemperature.hpp` — new boundary condition, STORM-adapted
- `boundary/Vacuum.hpp` — new vacuum boundary
- `utils/LorentzTransformation.hpp` — lightweight Lorentz boost utility

### MadVoro (medium risk) — 2 files ported

| Hazard | Severity | Details |
|--------|----------|---------|
| MockMesh volume exchange API | **MEDIUM** | Used indexed MPI exchange (`MPI_exchange_data_indexed`) instead of `MPI_exchange_data(Tessellation3D&)` since MadVoro's `Voronoi3D<PointT>` no longer inherits `Tessellation3D`. Equivalent if ghost/sent proc lists are consistent; validate in MPI rebalance scenario. |
| `conditional_shrink` is a no-op by default | LOW | Memory only aggressively released via explicit `ReleaseMemory()` unless `RICH_AGGRESSIVE_SHRINK` is defined. |
| `ReleaseMemory()` not auto-called | LOW | Must be called explicitly by the host; not wired into destructors or `Clean()`. |
| Intel LLVM compiler pragmas | LOW | All 15+ `#ifdef __INTEL_COMPILER` sites updated to include `__INTEL_LLVM_COMPILER`. Mechanical change. |
| No face clipping logic changes | INFO | Despite the report title suggesting face clipping fixes, the actual patch only contains memory optimizations and Intel pragma updates. If additional clipping fixes exist on ablation, they may be in a different file or commit range. |

### MeshDecomposer3D (low risk) — 6+ files ported

| Hazard | Severity | Details |
|--------|----------|---------|
| Double-sort after `getWeightedBorders3` | LOW | Sorting applied even when the function may already return sorted cuts. Harmless for lookup but worth knowing. |
| `reportImbalance` stale count | LOW | Max-weight rank broadcasts its count; if that rank did not just rebalance, the count could be stale. Same behavior as ablation. |
| `DomainDecompError` vs `UniversalError` | LOW | Adapted to submodule error type; message format differs slightly (`addEntry` uses `ostringstream`). |

### mpi_utils (medium risk) — 3 files ported

| Hazard | Severity | Details |
|--------|----------|---------|
| `MPI_Exchange_sparse` vs `MPI_Exchange_sparse_by_rank` | MEDIUM | Both exist with different semantics (graph-neighbor vs count-based). Call sites must pick the right one. |
| `FlatSparseHandle` requires `T::FLAT_BYTE_SIZE` | LOW | Types without `dumpFlat`/`loadFlat` cannot use the flat sparse API. |
| `reset()` without `shrink_to_fit()` | LOW | Serializers keep capacity across calls — lower allocation churn but higher peak memory. |
| Tag numbering (1040-1043) | LOW | Sparse iexchange uses 1040/1041, flat sparse uses 1042/1043. No overlap with existing tags. |

### spatial_ds (low risk) — 2 files ported

| Hazard | Severity | Details |
|--------|----------|---------|
| Pool memory retention | LOW | Freed OctTree nodes are recycled but blocks are only released at static destruction. Long-lived trees may retain memory. |
| Pool slot size assumption | LOW | All `OctTreeNode` instances for a given `OctTree<T>` must be the same size. Mixed sizes would corrupt the free list. Safe for current usage. |

---

## Files to inspect first (prioritized)

1. `source/3D/monte/MonteCarloManager3D.cpp` — ablation's progress hooks dropped
2. `source/3D/monte/MonteCarloManager3D.hpp` — interface changes
3. `source/monte/manager/parallel/RDMAMonteCarloManager.hpp` — largest STORM port
4. `source/monte/boundary/MovingSideTemperature.hpp` — entirely new file
5. `source/monte/boundary/Vacuum.hpp` — entirely new file
6. `source/monte/particle/Particle.hpp` — MPI serialization changed
7. `source/monte/population/CombPopulationControl.hpp` — StratifiedComb addition
8. `source/3D/tessellation/voronoi/Voronoi3D.hpp` — memory optimizations
9. `source/mpi/mpi_commands.hpp` — verify nothing dropped vs mpi_utils
10. `source/3D/gravity/DistributedGravityCalculator.hpp` — member name choice

---

## Integration execution: 2026-08-05

This section records the decisions made while merging `8031e78e` into
`78022104` from common base `c0c2aed3` on
`codex/merge-ablation-20260805`.  It supersedes the speculative file-port
notes above where the actual submodule interfaces differ from the legacy
vendored layout.

### Structural conflict decisions

- The `rich_features` repository layout remains authoritative.  All 13
  dependency paths remain mode-`160000` gitlinks; the legacy `source/monte`
  directory and `source/monte~HEAD` were removed.
- `source/monte` remains STORM.  `MeshDecomposer3D` remains a gitlink at its
  current commit because its Hilbert empty-boundary fallback covers both the
  ablation single-rank case and broader empty-boundary layouts.
- Current CMake dependency targets, include paths, STORM integration, and
  THUNDER wrapper were retained.  Only new RICH-owned FMM, spherical-shell,
  radiation, test, and documentation sources were added.
- `compile_commands.json` remains deleted.  `.gitignore` contains the union of
  the current branch additions and ablation's `**/mpi_tmp*` rule.
- The current STORM-adapted `gold_heat_wave_tau0_imc` case was retained.  It is
  still referenced by current regression configuration and documentation.
- `source/mpi/MPI_Particle3D_dtype.hpp` was restored exactly to the
  `rich_features` side.  The merge does not add the ablation branch's legacy
  flat DDMC members or a second `sourceCellID` entry to the MPI datatype, so the
  current STORM particle serialization layout remains unchanged.
- Ablation-era local includes and types in the new tests and retained runs were
  migrated to the current CMMC, STORM, `MeshDecomposer3D`, population-control,
  particle-status, and radiation-state interfaces.  A changed-source include
  resolution scan found no unresolved quoted local includes.

### Semantic ports into the current STORM adapter

The current STORM particle layout and transport interfaces were kept.  No
legacy Monte Carlo manager API or duplicate particle serialization format was
restored.  The local `codex/ablation-port-20260805` STORM branch, commit
`2b01032`, adds only the capabilities absent from the current transport
facade:

- adaptive postprocess source allocation, learned per-cell/per-group controls,
  external surface-source registration, Planck sampling, and generation
  diagnostics;
- DDMC external-source face eligibility and thermalization, detailed
  resident/transport/MPI counters, moving-interface controls, bounded `G_U`
  handling, local corrected-weight packet splitting, and full per-face and
  per-event diagnostic TSV contracts;
- LTE temperature evaluation for Compton Planck fallback and derivative
  scaling while retaining the current Compton and polarization machinery;
- RICH facade conversions for postprocess control structures, source
  diagnostics, external-source descriptors, and DDMC debug output.

The TSV methods deliberately emit data rows only.  The Densmore diagnostic
driver owns the headers and prefixes each row with step and time.  Emitting a
second header or a one-line counter summary would corrupt the tracked analysis
scripts' input schema.

### Regression metadata migration

The 14 ablation runner files were moved byte-for-byte to case-local
`REGRESSION_INFO` files.  Their build arguments, ranks, SLURM resources, run
commands, and check functions are unchanged.  THUNDER discovery found all 14,
and a combined `--dry-run --verbose` resolved all serial and MPI build/run
commands successfully.  The cluster's default `python3` initially could not
locate `libpython3.12.so.1.0`; discovery was rerun with command-local `PATH`
and `LD_LIBRARY_PATH` entries for the active Python runtime.
No repository or shell configuration was changed for that workaround.

### Both-side auto-merge audit

All 22 files changed by both parents but merged automatically were audited.
A line-retention comparison found no nontrivial addition from either parent
missing in the integration result.  Manual semantic inspection covered the
gravity tree/calculator, polarization and Lorentz transforms, spherical
observer, opacity adapter, radiation-step particle identity resynchronization,
Densmore cases, regression checks, report generators, plots, and mirrored
regression documentation.  Generated report outputs themselves remain pending
fresh run artifacts as described below.

### Post-merge review fixes

- Restored the current `runs/**/*.txt` ignore rule alongside ablation's
  `**/mpi_tmp*` rule and removed stale links to the deleted monolithic IMC
  Compton plans.
- Added direct standard-library includes to new FMM, postprocess, spherical,
  and Densmore headers so they do not depend on transitive include order.
- Made controlled multigroup-diffusion failures transactional: every
  reduced-timestep return restores the pre-attempt cells and extensives.  The
  representative failure-cell sentinel now preserves legitimate cell ID `0`
  and avoids attributing a remote MPI failure to a local cell `0`.
- Corrected the regression overview to place `lane_self_gravity_fmm` in the
  case-local THUNDER metadata layout rather than the legacy runner directory.
- Classified the 512-rank `lane_self_gravity_fmm` case as a manual benchmark,
  matching the validation policy that it receives discovery and dry-run checks
  unless a dedicated allocation is explicitly provided.
- Repaired the first GNU compile failures: the STORM namespace leak, the FMM
  Hilbert template argument, both self-shadowing points-manager aliases, and
  the obsolete Monte Carlo progress override.
- Changed the VTK/MPI compatibility probe to use CMake's imported
  `MPI::MPI_CXX` target instead of raw wrapper flags.  The direct shared-library
  dependency check remains in place.

### Remaining validation and publication risks

- No RICH build or executable has been run by the merge integrator.  GNU
  Release+MPI, Intel Release+MPI, the targeted cases, and the complete
  non-manual THUNDER suite remain user-run gates.
- The 512-rank `lane_self_gravity_fmm` case and 30-million-particle/64-node
  FMM matrix are discovery/dry-run checks only unless resources are allocated.
- Tracked generated reports must be regenerated from the returned fresh
  validation artifacts.  They must not be updated from dry-run metadata or by
  selecting either branch's pre-merge generated output.
- Excluded the unreferenced 38,033-line `luminosity_grey.vtk` snapshot from the
  merge because it is generated output, not a source or test input.  The
  tracked gold heat-wave profile is likewise pending regeneration by the
  merged regression before it can be accepted as validation evidence.
- The superproject merge commit remains intentionally pending until required
  build/run exit codes, scheduler states, logs, and generated artifacts have
  been reviewed.  No submodule or superproject branch is to be pushed before
  separate approval.
