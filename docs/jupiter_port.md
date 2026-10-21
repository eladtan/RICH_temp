# RICH on JUPITER (JSC booster): port notes and next steps

Summary of the JUPITER porting/optimization session, October 2026.
Branches: RICH `gpu_v1`, STORM (`source/monte`) `gpu_v1`.

## Machine

- JUPITER booster: GH200 nodes, aarch64 Grace, 4 superchips/node, 288 cores, 4 H100 GPUs.
- Account `e-dev-2026d09-248`, partition `booster`. Login nodes: use at most 2 cores; real runs go through SLURM.
- Scratch: `$SCRATCH_e_dev_2026d09_248` (`/e/scratch/e-dev-2026d09-248`). Project software: `$PROJECT_e_dev_2026d09_248/software`.
- On 2026-10-21 the whole booster was in a maintenance reservation (3889 nodes `maint`, the rest down/drained). Jobs pend with `ReqNodeNotAvail, Reserved for maintenance`.

## Environment

Use Stages/2025. Stages/2026 has no VTK and no `mpirun`.

```bash
module --force purge
module load Stages/2025 GCC/13.3.0 OpenMPI/5.0.5 HDF5/1.14.5 VTK/9.4.1 Boost/1.86.0 CMake/3.30.3
export Kokkos_ROOT=$PROJECT_e_dev_2026d09_248/software/kokkos/4.7.03-cuda12.6-gcc13.3
```

`~/.bashrc` defines `rich_modules()` with the module line above and calls it in interactive shells. Lmod collections (`module save/restore`) drop MPI-settings/UCX-settings, so don't use them.

Kokkos 4.7.03 builds (GCC 13.3):

| Build | Path under `$PROJECT_e_dev_2026d09_248/software/kokkos/` | Backends / arch |
|---|---|---|
| CUDA | `4.7.03-cuda12.6-gcc13.3` | CUDA 12.6 + OpenMP, HOPPER90 + ARMV9_GRACE |
| CPU | `4.7.03-openmp-gcc13.3` | OpenMP only |

To build Kokkos itself with only the OpenMP backend, pass `-DCMAKE_PROJECT_INCLUDE=<file containing include(CheckIncludeFileCXX)>`. Kokkos calls `check_include_file_cxx` without including that module.

## What was changed (committed)

### RICH

- **aarch64 build:**
  - `-mcmodel=medium` and VCL (`opt/vcl`) are used only on x86_64.
  - `conj_grad_solve.cpp` and `fastsqrt` in `misc/utils.cpp` fall back to scalar code.
- **`-ffp-contract=off`** (`config/compiler_flags.cmake`):
  - GCC fuses multiply-adds (FMA) on aarch64 by default.
  - That breaks MadVoro's exact predicates (stages 3/4 of `insphereadapt`) and crashed the tessellation.
  - Any new compiler-flag setup on ARM must keep this flag.
- **GPU builds** (`--with-gpu`):
  - The host compiler stays `g++` and MPI comes in through `MPI::MPI_CXX`. With the `mpicxx` wrapper, `mpi.h` is not found in device files.
  - STORM's `find_package(Kokkos COMPONENTS launch_compiler)` sends only Kokkos files to nvcc_wrapper.
  - roctx is searched only for HIP.
  - `RICH_WITH_KOKKOS` is defined.
- **`build_rich.sh`:** cleaning a build directory no longer deletes nested build directories that contain a `.build_cmd`.
- **RICH `RadiationIMC` wrapper:** forwards the portable opacity laws (`GetPortableAbsorptionLaw`, `GetPortableThermalFrequencyLaw`) and the device source/census paths. Without this, STORM falls back to the host path.
- **`source/utils/KokkosInitializer.hpp`:**
  - A function-local static singleton, created in the `Simulation` constructor.
  - It initializes Kokkos if nobody has yet and finalizes it at exit only if it did.
  - Drivers never call `Kokkos::initialize/finalize`.
- **`source/utils/MPIInitializer.hpp`:**
  - Never initializes MPI; at exit it calls `MPI_Finalize` only if MPI is initialized and not yet finalized.
  - Tested in isolation but **not wired in anywhere yet**.
- **Run scripts:**
  - `runs/Elad_paper_mach2/submit_{cpu,cpu_kokkos,gpu}.sh`.
  - `runs/moving_slab_mc/submit_{cpu_omp,gpu}.sh`.
  - `runs/Elad_paper_hohlraum/submit_kokkos_omp.sh`: ranks × threads are chosen at sbatch time.
  - All scripts send BEGIN/END/FAIL mail.
  - The GPU script starts MPS in the batch step. A daemon started inside an `srun` step dies with the step.
- **Hohlraum driver:**
  - Boundary photons are generated in parallel (OpenMP + counter RNG keyed on call/rank/slot).
  - New `--no-output` flag.
  - New `[startup]` stage timers.

### STORM

- **Automatic Kokkos lifetime:** `gpu/KokkosInitializer.hpp` replaces `gpu/KokkosRuntime.hpp`.
  - It uses the same ownership rule as RICH's initializer: whichever initializer runs first owns and finalizes Kokkos.
  - The examples call `KokkosInitializer::FinalizeIfInitialized()` before `MPI_Finalize`.
  - STORM must not initialize MPI.
- **CMake:**
  - `launch_compiler`, plus a clear error when `Kokkos_ROOT` is unset.
  - `Kokkos_DEVICES` is exported to the parent project.
  - `-fopenmp` is added at link time when Kokkos has OpenMP (otherwise libgomp fails with "DSO missing").
- **nvcc lambdas:** extended-lambda captures must be identical, in the same first-use order, in the host and device passes. A mismatch gives `cudaErrorInvalidDeviceFunction` (see the `(void)usePrivateEnergy; (void)threadViews;` lines in `KokkosLocalTransportExecutor.hpp`).
- **OpenMP-backend scaling:**
  - Wave counters are aligned to cache lines.
  - Ingest and the census copy run in parallel.
  - `CombDeviceActivate` uses `parallel_scan` and a parallel per-cell sort.

## Benchmark results so far

All on one booster node (288 cores).

| Case | Configuration | Result |
|---|---|---|
| Mach2 | GPU, 16 ranks × 18 threads, MPS, 4 GPUs | 409 s, then 335 s with the latest code |
| Hohlraum (892) | 1 rank × 288 threads, OpenMP Kokkos | Compute 991 s → 421 s. Serial phases per step 2.0 s → 0.2 s. 455 s end-to-end with no output. |
| Hohlraum (892) | 288 ranks × 1 thread | Compute 162 s. 420 s end-to-end, of which ~243 s is startup before main's first timer. |
| Moving slab | one node | Too slow to be practical |

## Open problems / next steps

1. **Pushed:** RICH, STORM, MadVoro, MeshDecomposer3D and spatial_ds are all on GitHub/GitLab.
2. **MadVoro work from the parallel session:**
   - RICH's submodule pointers for these still point at their old commits. On a new machine, check out branch `gpu_v1` in each of:
     - `source/3D/tessellation/voronoi` (portable Voronoi kernels, thread subdomains);
     - `source/3D/tessellation/MeshDecomposer3D`;
     - `source/utils/spatial_ds`.
   - Uncommitted edits in `voronoi` and `MeshDecomposer3D` were saved as a snapshot on branch `gpu_v1-wip` (one commit on top of `gpu_v1`; may not build).
   - Its job scripts are in `$SCRATCH/madvoro_work/sub/` (`timeD.sh`, `checkD.sh`, `profD.sh`, `recheck.sh`).
   - Jobs 2428974/6/7/8 are queued on JUPITER behind the maintenance; cancel them if they are no longer wanted.
   - `runs/mc_results_dir.hpp` has a JUPITER-only path change, not committed.
3. **288×1 startup (~243 s):**
   - Probably `MPI_Init` with UCX loading CUDA transports.
   - Try `UCX_TLS=rc_x,self,sm` (no `cuda_*` / `gdr_copy`) for CPU-only runs, and time `MPI_Init` alone.
4. **1×288 kernel is still ~2.5× slower than 288×1:**
   - Suspected contention on shared counters and census energy tallies.
   - Next: `perf c2c` on a short run; per-thread tallies with a final reduction.
5. **Wire in `MPIInitializer`:** e.g. call `MPIInitializer::Instance()` right after `MPI_Init` in the drivers, before RICH's `KokkosInitializer` is created. That way MPI is finalized after Kokkos.
6. **Moving slab:** rebalance with a cost function that accounts for particle work, then rerun on GPU.
7. **GPU benchmarks for the hohlraum and moving slab:** the CUDA builds exist (`build/gnuReleaseMPI/slab-gpu`), but these runs were not completed before the maintenance.

## Build directories (on JUPITER, under `RICH/build/gnuReleaseMPI/`)

| Directory | Contents |
|---|---|
| `.` | CUDA GPU Mach2 |
| `cpu` | CPU Mach2 |
| `gpu` | Older GPU build |
| `kokkos-omp` | Mach2 with OpenMP Kokkos |
| `slab-omp`, `slab-gpu` | Moving slab |
| `hohl-omp` | Hohlraum with OpenMP Kokkos |

Each directory has a `.build_cmd` with the command that built it.
