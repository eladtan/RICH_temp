# Build System

RICH uses `build_rich.sh` as the canonical build interface, wrapping CMake and Make with per-configuration build directories, automatic change detection, and parallel compilation.

## Quick Reference

```bash
./build_rich.sh <config> --test_name=<run_dir> [options]
```

**Example:**

```bash
./build_rich.sh gnuReleaseMPI --test_name=sedov_3d
```

## Build Configurations

Configuration names are composed from tokens that are parsed by CMake:

| Token | Meaning |
|-------|---------|
| `gnu` | GNU compiler toolchain (gcc, g++, gfortran) |
| `intel` | Intel compiler toolchain (icx, icpx, ifx) |
| `Release` | Optimized build (-O2) |
| `Debug` | Debug build (-O0, assertions, DWARF symbols) |
| `Mixed` | Release build with selected files compiled in debug mode |
| `MPI` | MPI-enabled build (uses mpicc, mpicxx, mpif90) |
| `Prof` | GProf profiling enabled (-g -pg) |

### Available Configurations

**GNU:**

```
gnuRelease          gnuReleaseMPI       gnuReleaseProf      gnuReleaseMPIProf
gnuDebug            gnuDebugMPI         gnuDebugProf        gnuDebugMPIProf
```

**Intel:**

```
intelRelease        intelReleaseMPI     intelReleaseProf    intelReleaseMPIProf
intelDebug          intelDebugMPI       intelDebugProf      intelDebugMPIProf
```

**Mixed (requires `--debug_files`):**

```
gnuMixed            gnuMixedMPI
intelMixed          intelMixedMPI
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `<config>` | Build configuration (required, first positional argument) | -- |
| `--test_name=<name>` | Run directory containing `main.cpp` or `test.cpp` (required) | -- |
| `--with_asan` | Enable AddressSanitizer | Off |
| `--energy_groups_num=<N>` | Override `ENERGY_GROUPS_NUM` for multigroup radiation | Compile-time default |
| `--mc_debug` | Enable Monte Carlo debug mode | Off |
| `--memory_debug` | Print per-cycle RSS memory usage (max-rank and sum-all in GB) to stderr at key simulation, hydro, radiation, and I/O checkpoints | Off |
| `--debug_files=<path>` | File list for mixed-debug builds | -- |
| `--build-subdir=<name>` | Build into `build/<config>/<name>/` | `build/<config>/` |
| `--jobs=<N>` | Parallel make jobs | `$(nproc)` |

### Examples

```bash
# Basic release build
./build_rich.sh gnuRelease --test_name=sedov_3d

# MPI release build in a separate output directory
./build_rich.sh gnuReleaseMPI --test_name=sedov_3d --build-subdir=sedov-mpi

# Debug build with AddressSanitizer
./build_rich.sh gnuDebug --test_name=sod --with_asan

# MPI build with memory tracking (prints RSS to stderr at each simulation phase)
./build_rich.sh intelReleaseMPI --test_name=sedov_3d --memory_debug

# Build into a named subdirectory (keeps multiple executables)
./build_rich.sh gnuReleaseMPI --test_name=sedov_3d --build-subdir=sedov
```

## Build Artifacts

After a successful build:

```
build/<config>/
├── rich_<config>             # The compiled binary
├── rich -> rich_<config>     # Convenience symlink
├── <config>_cmake.out        # CMake configuration stdout
├── <config>_cmake.err        # CMake configuration stderr
├── <config>_build.out        # Make stdout
├── <config>_build.err        # Make stderr
├── .build_cmd                # Stored command for change detection
└── ... (CMake/make generated files)
```

With `--build-subdir=name`:

```
build/<config>/<name>/
├── rich_<config>
├── rich -> rich_<config>
└── ...
```

## How the Build Script Works

1. **Directory setup:** Creates `build/<config>/` (or `build/<config>/<subdir>/`).

2. **Change detection:** Stores the full build command in `.build_cmd`. If arguments change between builds, the build directory is cleaned and CMake re-runs from scratch.

3. **Source tracking:** Detects added or removed source files (`.cpp`, `.c`, `.hpp`, `.h`) and re-runs CMake when the file set changes.

4. **CMake tracking:** Watches `CMakeLists.txt` and all `*.cmake` files under `source/` and `config/`. CMake re-runs when any of these change.

5. **Compilation:** Runs `make -j<N>` where N defaults to all available cores.

6. **Symlink:** Creates `build/<config>/rich` pointing to `rich_<config>`.

## CMake Details

The CMake build system is defined in `source/CMakeLists.txt` with configuration modules in `config/`.

### Required CMake Variables

| Variable | Set By | Description |
|----------|--------|-------------|
| `CONFIG` | `build_rich.sh` | Build configuration name |
| `TEST_DIR` | `build_rich.sh` (`--test_name`) | Path to the test/run directory |

### Optional CMake Variables

| Variable | Description |
|----------|-------------|
| `ASAN` | Enable AddressSanitizer (set to 1) |
| `ENERGY_GROUPS_NUM` | Number of energy groups for multigroup radiation |
| `MC_DEBUG` | Enable Monte Carlo debug mode |
| `DEBUG_FILES` | File list for mixed-mode compilation |

### Compiler Flags

| Mode | Flags |
|------|-------|
| Release | `-O2` |
| Debug | `-O0 -gdwarf-3 -D_GLIBCXX_ASSERTIONS -DDEBUG` |
| Common | `-std=c++17 -Wextra -Wshadow -Wunused-parameter -Werror=return-type` |
| MPI | Adds `-DRICH_MPI` |
| Prof | Adds `-g -pg` |

### Dependency Discovery

| Dependency | Discovery Method |
|------------|-----------------|
| HDF5 | `LD_LIBRARY_PATH` or `HDF5_DIRECTORY` |
| VTK | `LD_LIBRARY_PATH` or `VTK_DIRECTORY` |
| Boost | `BOOST_DIR` or `BOOST_ROOT` |
| ZLIB | `find_package(ZLIB)` |
| ParMETIS | `LD_LIBRARY_PATH` |
| jsoncpp | `LD_LIBRARY_PATH` |

### Bundled Third-Party Libraries

| Library | Location | Purpose |
|---------|----------|---------|
| VCL | `source/opt/vcl` | Vector Class Library (SIMD) |
| Clipper | `source/opt/clipper` | Polygon clipping |

## Config Module Files

| File | Purpose |
|------|---------|
| `config/parse_arguments.cmake` | Parse CONFIG string into compiler/mode flags |
| `config/set_compilers.cmake` | Set C/C++/Fortran compilers |
| `config/compiler_flags.cmake` | Compiler flags, defines, optimization |
| `config/mixed.cmake` | Mixed build configuration |
| `config/find_HDF5.cmake` | HDF5 discovery |
| `config/find_VTK.cmake` | VTK discovery |
| `config/find_Boost.cmake` | Boost discovery |
| `config/find_ParMETIS.cmake` | ParMETIS/METIS/GKlib discovery |
| `config/find_JSON.cmake` | jsoncpp discovery |
| `config/find_python.cmake` | Python discovery |
| `config/find_vtune.cmake` | VTune discovery |
| `config/find_pybind11.cmake` | pybind11 discovery |
| `config/find_static_library.cmake` | Static library discovery helper |
| `config/find_cgal.cmake` | CGAL discovery |
| `config/set_placeholders.cmake` | Placeholder variable setup |
| `config/include_3rdparty.cmake` | r3d, VCL, Clipper includes |

## Profiling

For `Prof` configurations, the simulation produces a `gmon.out` file. Process it with:

```bash
gprof build/<config>/rich gmon.out | gprof2dot -s -w --show-samples | dot -Tpdf -o gprof.pdf
```
