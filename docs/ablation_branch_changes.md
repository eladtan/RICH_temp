# Ablation Branch — Changes Report

**Branch:** `origin/ablation`
**Common ancestor with `submodules_integration`:** `16c798bc`
**Commits ahead:** 122
**Total changed files:** 228

---

## Summary

The `ablation` branch diverged from the shared ancestor long before the submodule
extraction. Many files it modifies no longer exist in RICH's `submodules_integration`
branch — they now live inside submodules. Below is a breakdown of every changed file,
grouped by destination (submodule or RICH proper).

| Target | Files changed | Lines added | Lines removed |
|--------|:------------:|:-----------:|:-------------:|
| **RICH (stays in RICH)** | 183 | ~16,000+ | ~1,500+ |
| **STORM (source/monte)** | 13 | 1,252 | 80 |
| **MadVoro (source/3D/tessellation/voronoi)** | 8 | 150 | 101 |
| **MeshDecomposer3D** | 6 | 48 | 22 |
| **mpi_utils (source/mpi/serialize)** | 2 | 318 | 16 |
| **spatial_ds (source/ds)** | 2 | 87 | 3 |
| **Total** | **228** (some overlap counted once) | | |

---

## 1. STORM (Monte Carlo submodule) — 13 files, +1252 / −80

These files used to live at `source/monte/` in RICH but are now inside the
STORM submodule at `source/monte/`. The ablation branch paths use the **old**
directory layout (e.g. `serial_manager/`, `two_sided_manager/`,
`manager/MonteCarloManager.hpp`) which must be mapped to the **new** STORM paths
(e.g. `manager/MonteCarloManagerSerial.hpp`, `manager/parallel/TwoSidedMonteCarloManager.hpp`,
`manager/parallel/RDMAMonteCarloManager.hpp`).

| Ablation path | STORM equivalent | Change |
|---------------|------------------|--------|
| `source/monte/MonteCarloParticle.hpp` | `particle/Particle.hpp` | Extended particle fields (+39/−0) |
| `source/monte/boundary/BoundaryCondition.hpp` | `boundary/BoundaryCondition.hpp` | Major rework (+102/−0) |
| `source/monte/boundary/MovingSideTemperature.hpp` | **NEW** — does not exist in STORM | New file (+236) |
| `source/monte/boundary/Rigid.hpp` | `boundary/RigidBoundary.hpp` | Modified (+35/−35) |
| `source/monte/boundary/SideTemperature.hpp` | `boundary/SideTemperature.hpp` | Modified (+44/−0) |
| `source/monte/boundary/TwoSidesTemperature.hpp` | `boundary/TwoSidesTemperature.hpp` | Modified (+45/−45) |
| `source/monte/boundary/Vacuum.hpp` | **NEW** — does not exist in STORM | New file (+52) |
| `source/monte/manager/MonteCarloManager.hpp` | `manager/parallel/RDMAMonteCarloManager.hpp` | Major changes (+345/−0) |
| `source/monte/physics/MonteCarloPhysics.hpp` | `physics/MonteCarloPhysics.hpp` | Modified (+12/−0) |
| `source/monte/population/Comb.hpp` | `population/CombPopulationControl.hpp` | Major rework (+349/−0) |
| `source/monte/serial_manager/MonteCarloManagerSerial.hpp` | `manager/MonteCarloManagerSerial.hpp` | Modified (+40/−0) |
| `source/monte/two_sided_manager/TwoSidedMonteCarloManager.hpp` | `manager/parallel/TwoSidedMonteCarloManager.hpp` | Modified (+28/−0) |
| `source/monte/utils/GhostMap.hpp` | **in RICH** (`source/mpi/` or removed) | Minor (+5/−1) |

**Key new features from ablation:**
- `MovingSideTemperature` — new boundary condition for moving temperature sources
- `Vacuum` — new vacuum boundary condition
- `StratifiedCombPopulationControl` — new population control strategy
- Major extensions to `BoundaryCondition` interface and `MonteCarloManager`

---

## 2. MadVoro (Voronoi submodule) — 8 files, +150 / −101

These files now live inside the MadVoro submodule at `source/3D/tessellation/voronoi/`.
Note: in the current submodule, `Voronoi3D` is a header-only template (`Voronoi3D.hpp`),
`Delaunay3D` is embedded inside MadVoro, and `PointsManager` has been moved into
`MeshDecomposer3D`.

| Ablation path | MadVoro equivalent | Change |
|---------------|-------------------|--------|
| `source/3D/tessellation/voronoi/Voronoi3D.cpp` | `Voronoi3D.hpp` (merged into header) | Face clipping logic, edge cases (+205/−101) |
| `source/3D/tessellation/voronoi/Voronoi3D.hpp` | `Voronoi3D.hpp` | Minor additions (+2) |
| `source/3D/tessellation/delaunay/Delaunay3D.cpp` | `Delaunay3D.hpp` (header-only) | Memory fixes (+22/−0) |
| `source/3D/tessellation/delaunay/Delaunay3D.hpp` | `Delaunay3D.hpp` | Minor additions (+3) |
| `source/3D/tessellation/voronoi/pointsManager/HilbertPointsManager.cpp` | **MeshDecomposer3D** submodule | Minor fix (+2/−2) |
| `source/3D/tessellation/voronoi/pointsManager/HilbertPointsManager.hpp` | **MeshDecomposer3D** submodule | Minor fix (+2/−2) |
| `source/3D/tessellation/voronoi/pointsManager/PointsManager.cpp` | **MeshDecomposer3D** submodule | Repartition logic (+13/−0) |
| `source/3D/tessellation/voronoi/pointsManager/PointsManager.hpp` | **MeshDecomposer3D** submodule | Minor fix (+2/−2) |

**Note:** The 4 `pointsManager` files are actually in the **MeshDecomposer3D** submodule
now, not MadVoro. They are listed here because ablation's paths put them under `voronoi/`.

---

## 3. MeshDecomposer3D — 6 files, +48 / −22

Files now in the MeshDecomposer3D submodule at
`source/3D/tessellation/MeshDecomposer3D/`.

| Ablation path | MeshDecomposer3D equivalent | Change |
|---------------|---------------------------|--------|
| `source/3D/hilbert/HilbertOrder3D.cpp` | `hilbert/HilbertOrder3D.hpp` | Minor fixes (+4/−2) |
| `source/3D/hilbert/HilbertOrder3D.hpp` | `hilbert/HilbertOrder3D.hpp` | Minor fix (+2/−1) |
| `source/3D/hilbert/rectangular/HilbertRectangularTree3D.hpp` | `hilbert/rectangular/HilbertRectangularTree3D.hpp` | Fixes (+10/−3) |
| `source/3D/tessellation/loadBalancing/CurveLoadBalancer.cpp` | `load_balancing/CurveLoadBalancer.hpp` | Additions (+12) |
| `source/3D/tessellation/loadBalancing/HilbertLoadBalancer.cpp` | `load_balancing/HilbertLoadBalancer.hpp` | Rebalance logic (+40/−7) |
| `source/3D/tessellation/loadBalancing/HilbertLoadBalancer.hpp` | `load_balancing/HilbertLoadBalancer.hpp` | Minor fix (+2/−1) |

---

## 4. mpi_utils — 2 files, +318 / −16

Files now in the mpi_utils submodule at `source/utils/mpi_utils/`.

| Ablation path | mpi_utils equivalent | Change |
|---------------|---------------------|--------|
| `source/mpi/serialize/Serializer.hpp` | `serialize/Serializer.hpp` | Minor additions (+5/−1) |
| `source/mpi/serialize/mpi_commands.hpp` | `serialize/mpi_commands.hpp` | Major additions: new exchange/sync patterns (+329/−15) |

---

## 5. spatial_ds — 2 files, +87 / −3

Files now in the spatial_ds submodule at `source/utils/spatial_ds/`.

| Ablation path | spatial_ds equivalent | Change |
|---------------|----------------------|--------|
| `source/ds/OctTree/OctTree.hpp` | `OctTree/OctTree.hpp` | Major extension: range query, iterator (+86/−1) |
| `source/ds/DistributedOctTree/DistributedOctTree.hpp` | `DistributedOctTree/DistributedOctTree.hpp` | Minor fixes (+4/−2) |

---

## 6. RICH proper — 183 files

These files stay in RICH and can be merged normally. Major areas:

### Radiation / IMC (largest body of work)
- **~10,400 lines added** across `source/3D/radiation/`
- New reverse estimator framework: `ReverseAdjointTransport3D`, `ReverseDDMC`,
  `ReverseDoppler`, `ReverseObserverTallies`, `ReversePolarizationMueller`,
  `ReversePacket`, `ReverseEstimatorConfig`
- New spherical observer: `SphericalObserver` (~2,700 lines)
- IMC polarization: `IMCPolarization`
- Fleck factor helpers, post-processing helpers
- Enhanced `RadiationIMC` with DDMC support (+1,449 lines in `.cpp`, +726 in `_DDMC.cpp`)
- `IMCMeasuredLoadBalance`, `IMCStepCounterCostCalculator`

### Newtonian / hydro
- AMR3D distributed clip support
- CFL1D, CourantFriedrichsLewy enhancements
- LagrangianExtensiveUpdater3D, ConditionExtensiveUpdater3D updates
- Simulation step framework updates (HydroStep, RadiationStep, RadiationMCStep)

### Tessellation (RICH layer, not submodule)
- `ExchangeGhosts.cpp`, `ExchangeFaces.cpp`, `Neighbors.cpp` updates
- `Tessellation3D.hpp` interface changes
- `PolyClip` and `RandomInCell` utilities

### Regression tests & runs
- New tests: `doppler_scatter_mc`, `spherical_gauss_tangential`, `till_compton_mc`,
  `amr_distributed_clip`
- New runs: `compton_marshak_wave`, `gold_heat_wave_tau0_imc`, `imc_postprocess_tde`
- Test infrastructure: `generate_vv_report.py`, updated `run_all.sh`

### Other
- MPI exchange commands updates
- Config / build system updates
- Documentation (imc_compton plans, wiki pages, regression test docs)

---

## Approach for merging

1. **RICH proper (183 files):** Can be cherry-picked or merged directly — these
   paths still exist in RICH.

2. **Submodule files (31 files across 5 submodules):** Cannot be merged directly
   because the files have moved to submodules and may have been restructured
   (renamed, templated, merged into headers). Each submodule needs the ablation
   diff applied manually:
   - Extract each file's diff: `git diff 16c798bc..origin/ablation -- <old_path>`
   - Map to the new submodule path
   - Apply the patch (may need manual conflict resolution due to templating)

3. **Path mapping challenges:**
   - STORM: directory structure changed (`serial_manager/` → `manager/`,
     `two_sided_manager/` → `manager/parallel/`, files renamed)
   - MadVoro: `.cpp` files merged into `.hpp` templates, `PointsManager` moved
     to MeshDecomposer3D
   - MeshDecomposer3D: `.cpp` files merged into `.hpp` templates,
     `loadBalancing/` → `load_balancing/`
   - mpi_utils: `source/mpi/serialize/` → top-level `serialize/`
   - spatial_ds: `source/ds/` → top-level in submodule
