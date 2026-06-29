# RICH Modularization — Context Transfer

## Project Overview

RICH is a large astrophysical hydrodynamics simulation code. Over the past sessions, the codebase has been modularized into standalone Git submodules, each usable independently of RICH.

## Current State

**RICH branch:** `submodules_integration`
**Last RICH commit:** `534df787` (Merge ablation branch)
**RICH has ~113 uncommitted file changes** (across source, regression tests, runs, config). These include the CMake refactoring, submodule pointer updates, regression test fixes, and the ablation merge adaptations.

## Submodule Map

| Submodule | Location in RICH | Repo | Branch | Status |
|---|---|---|---|---|
| **MadVoro** | `source/3D/tessellation/voronoi/` | `git@github.com:maormizrachi/MadVoro.git` | `master` | Pushed (`dd97670`) |
| **MadCart** | `source/3D/tessellation/cartesian/` | `git@github.com:maormizrachi/MadCart.git` | `main` | Clean |
| **STORM** | `source/monte/` | (Monte Carlo transport) | — | Has own CMakeLists.txt |
| **MeshDecomposer3D** | `source/3D/tessellation/MeshDecomposer3D/` | `git@github.com:maormizrachi/MeshDecomposer3D.git` | `master` | Clean, header-only |
| **spatial_ds** | `source/utils/spatial_ds/` | `git@github.com:maormizrachi/spatial_ds.git` | — | Clean, header-only |
| **mpi_utils** | `source/utils/mpi_utils/` | `git@github.com:maormizrachi/mpi_utils.git` | — | Has `AmountManager.cpp` |
| **EasyRMA** | `source/utils/rma/` | — | — | Has 5 .cpp files |
| **range_finders** | `source/3D/tessellation/voronoi/range/finders/` | `git@github.com:maormizrachi/range_finders.git` | `master` | Sub-submodule of MadVoro, clean |

## Architecture Decisions

### PointT Templating
All submodules (MadVoro, STORM, range_finders) are templated on `PointT`. The contract:
- Must have `x`, `y`, `z` members
- Must have `coord_type` typedef (e.g., `double`)
- Must have a 3-arg constructor `PointT(x, y, z)` and default constructor
- Arithmetic/geometric operators are provided as **ADL fallback templates** in `elementary/PointOps.hpp` (SFINAE-guarded with `is_point3d` trait), but external submodules (`spatial_ds`, `MeshDecomposer3D`) require `operator[]`, `operator==`, and arithmetic operators as member/friend functions on the actual `PointT` type.

### MPI Guard Macros
Each submodule uses its own MPI macro:
- RICH: `RICH_MPI`, `__WITH_MPI`
- MadVoro: `MADVORO_WITH_MPI`
- STORM: `STORM_WITH_MPI`
- All defined in `config/compiler_flags.cmake` when `MPI` is set.

### CMake Build System (Refactored This Session)
`source/CMakeLists.txt` was refactored to create **separate library targets** for each submodule:

```
madvoro    — STATIC (exception/*.cpp)
madcart    — INTERFACE (header-only)
storm      — STATIC (RankSync.cpp, ReallocationAgent.cpp)
mpi_utils  — STATIC (AmountManager.cpp)
spatial_ds — INTERFACE (header-only)
mesh_decomposer — INTERFACE (header-only)
easyrma    — STATIC (5 .cpp files)
```

RICH's own sources use `GLOB_RECURSE` with all submodule directories subtracted. The executable links: `target_link_libraries(${EXE_NAME} PRIVATE madvoro madcart storm mpi_utils spatial_ds mesh_decomposer easyrma ...)`.

The submodules' own `CMakeLists.txt` files are **not** used via `add_subdirectory` (to avoid `find_package` conflicts with RICH's custom dependency scripts). They remain for standalone builds only.

**Both MPI and serial builds pass cleanly.**

### Standalone Submodule Builds
MadVoro, MadCart, and STORM each have their own `CMakeLists.txt` with:
- `find_package(Boost)`, `find_package(MPI)`, etc. for standalone use
- `install_deps.sh` to clone dependencies into `deps/`
- `MADVORO_DEPS_DIR` defaults to `${CMAKE_CURRENT_SOURCE_DIR}/deps`
- Examples in `examples/` directory with minimal standalone `Vector3D.hpp`

### Error Handling
- MadVoro: `MadVoroException` (in `exception/`)
- STORM: `STORMError` (in `StormError.hpp`)
- RICH: `UniversalError` (in `misc/`)

### Removed / Relocated
- `source/utils/queryAgent/` — removed, use `<mpi_utils/queryAgent/>`
- `source/utils/rma-helpers/` — removed (dead code, functionality in `rma/`)
- `source/3D/range/` — removed, moved into MadVoro's `range/` and `range_finders` submodule
- MadVoro's `io3D`, `simple_io` — removed (only HDF5/VTK supported)
- MadVoro's `Mat33` dependency — removed (determinant inlined)

## Key Files

- **RICH CMake:** `source/CMakeLists.txt` — the refactored build with submodule targets
- **Compiler flags:** `config/compiler_flags.cmake` — defines MPI macros for all submodules
- **RICH adapter for MadVoro:** `source/3D/tessellation/Voronoi3D.hpp` — wraps `MadVoro::Voronoi3D<Vector3D>`
- **MadVoro core:** `source/3D/tessellation/voronoi/Voronoi3D.hpp` (~4949 lines, header-only template)
- **ADL fallbacks:** `source/3D/tessellation/voronoi/elementary/PointOps.hpp` and `source/monte/elementary/PointOps.hpp`
- **MadVoro standalone CMake:** `source/3D/tessellation/voronoi/CMakeLists.txt`

## MadVoro Recent Commits (master branch)
```
dd97670 Replace RICH_MPI with MADVORO_WITH_MPI, fix standalone deps path
0859adc Update CMake for header-only core, bump range_finders submodule
7059d16 Update examples for standalone build with minimal Vector3D
bff5f0d Template MadVoro core on PointT, remove RICH dependencies
64d9a0d Add range_finders submodule and move range query agents into MadVoro
```

## Known Issues / Technical Debt
- RICH still uses global `include_directories()` — not yet migrated to per-target `target_include_directories()`
- `source/CMakeLists.txt` line 160: `include_directories("${PROJECT_SOURCE_DIR}/..")` marked `# TODO: REMOVE!`
- Pre-existing compiler warnings in `tessellation/Delaunay.cpp` (unused variable/function)
- MadVoro standalone compilation was just fixed (deps path + `MADVORO_WITH_MPI` macro) — user was testing at `/home/maorm/MadVoro/`
- RICH's uncommitted changes need to be committed and pushed to the `submodules_integration` branch
