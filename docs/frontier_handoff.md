# Continuing on Frontier: handoff from JUPITER

This picks up the work recorded in [jupiter_port.md](jupiter_port.md) (read it first). That work was done on JUPITER (aarch64 + NVIDIA H100, CUDA Kokkos) and moved to Frontier (x86 + AMD MI250X, HIP Kokkos) on 2026-10-21.

## 1. Branches to check out

| Repository | Path in RICH | Branch | Notes |
|---|---|---|---|
| RICH | `.` | `gpu_v1` | At `65047f9a` or later |
| STORM | `source/monte` | `gpu_v1` | RICH points at `5dcc575` |
| MadVoro | `source/3D/tessellation/voronoi` | `gpu_v1` | RICH still points at the old commit `7742109`, so check out the branch by hand |
| MeshDecomposer3D | `source/3D/tessellation/MeshDecomposer3D` | `gpu_v1` | Same: RICH still points at `38044c2` |
| spatial_ds | `source/utils/spatial_ds` | `gpu_v1` | Same: RICH still points at `29bb801` |

```bash
cd RICH
git fetch origin && git checkout gpu_v1 && git pull
git submodule update --init --recursive
git -C source/monte checkout gpu_v1 && git -C source/monte pull
for s in source/3D/tessellation/voronoi source/3D/tessellation/MeshDecomposer3D source/utils/spatial_ds; do
    git -C $s fetch origin && git -C $s checkout gpu_v1 && git -C $s pull
done
```

**WIP snapshots.** MadVoro and MeshDecomposer3D also have a `gpu_v1-wip` branch: one commit on top of `gpu_v1` with work that was still uncommitted on JUPITER.
- In MadVoro it is threaded subdomain search/merge in `Voronoi3D.hpp` and `SubdomainSearch.hpp`, plus a new `voronoi_thread_subdomains_mpi` regression test.
- It was never built or tested as a whole. Use `gpu_v1` unless you are continuing that MadVoro work. To continue it, check out `gpu_v1-wip`, get it building and passing the regression tests, then squash it into proper commits on `gpu_v1`.

**Submodule pointers.** After you commit in a submodule, update RICH's pointer to it (`git add <submodule>`) in a RICH commit. Push the submodule first.

## 2. Cautions

1. **The HIP GPU build path changed and is untested on Frontier.** This is the most important caution.
   - **Before** (up to RICH `e03c50fb`): `config/set_compilers.cmake` made `hipcc` the C++ compiler for all of RICH when `STORM_WITH_GPU` was on.
   - **Now**:
     - The host compiler (`g++`, or the MPI wrapper for CPU builds) compiles everything.
     - Kokkos's `launch_compiler` reroutes only Kokkos-dependent sources to `Kokkos_CXX_COMPILER` (`hipcc` / `amdclang++`).
     - In GPU builds MPI comes in through `MPI::MPI_CXX`, not through a wrapper used as the compiler.
   - This worked with CUDA on JUPITER but has not been tried with HIP.
   - If it fails, the old behavior is in `git show e03c50fb:config/set_compilers.cmake`. Fix forward (keep launch_compiler working for both) rather than reverting, if possible.
   - Check:
     - that the installed Kokkos provides the `launch_compiler` component;
     - that HIP flags (`-fno-optimize-sibling-calls`) and `roctx` are still applied (they are keyed on `"HIP" IN_LIST Kokkos_DEVICES`);
     - that libstdc++ / ABI are compatible between g++ and hipcc objects (use the same GCC toolchain that hipcc uses: `--gcc-toolchain`).
2. **Order of Kokkos and MPI finalization.**
   - Kokkos is now finalized automatically at program exit by static singletons:
     - RICH: `source/utils/KokkosInitializer.hpp`;
     - STORM: `gpu/KokkosInitializer.hpp`;
     - MadVoro: `MadVoro::KokkosInitializer`.
   - The drivers (`runs/*/test.cpp`) still call `MPI_Finalize()` explicitly at the end of `main`. Kokkos is therefore finalized **after** MPI.
   - With CUDA on JUPITER that was fine. With HIP + Cray MPICH/OFI on Frontier, finalizing HIP after `MPI_Finalize` is known to hang or crash at exit (this is why STORM's examples call `STORM::gpu::KokkosInitializer::FinalizeIfInitialized()` first).
   - Fix (next-steps item 2): call `MPIInitializer::Instance()` right after `MPI_Init` and drop the explicit `MPI_Finalize()`. MPIInitializer is then created before any Kokkos initializer, so it is destroyed after them and MPI is finalized last.
   - Quick workaround: call `KokkosInitializer::Instance().Finalize()` (RICH) before `MPI_Finalize()`.
   - Initializer ownership: whichever runs first owns Kokkos; the others never finalize it. Don't add calls to `Kokkos::initialize/finalize` anywhere.
3. **`-ffp-contract=off`** is added only for GNU (`config/compiler_flags.cmake`).
   - Fused multiply-add broke MadVoro's exact Delaunay predicates on aarch64. x86 GCC does not fuse without `-mfma`/`-march=native`, but clang/amdclang/hipcc fuse by default (`-ffp-contract=on`).
   - If MadVoro or other predicate code is ever compiled by a clang-based compiler (Cray CCE, amdclang, hipcc), add `-ffp-contract=off` for it too.
   - Symptom: tessellation crashes or `insphereadapt` asserts on degenerate (lattice) point sets.
4. **JUPITER-specific settings in the run scripts (do not copy blindly):**
   - Module lines (`Stages/2025 ...`), account `e-dev-2026d09-248`, and `$SCRATCH_e_dev_2026d09_248` paths.
   - The NVIDIA MPS setup in `runs/Elad_paper_mach2/submit_gpu.sh`. On Frontier, give each rank one GCD with `ROCR_VISIBLE_DEVICES` (see `runs/CrookedPipe/run_gpu_profile.sh`, the reference Frontier script, account `ast246`).
   - `--ibv` in `runs/Elad_paper_hohlraum/submit_kokkos_omp.sh` selects the InfiniBand transport. Frontier is Slingshot, so use the OFI/MPI path instead (`FI_CXI_RX_MATCH_MODE=hybrid` as in the CrookedPipe script).
   - Rank/thread counts were for 288 cores/node. A Frontier node has 64 cores, 56 usable with low-noise mode, and 8 GCDs.
   - `runs/mc_results_dir.hpp` was edited on JUPITER only (not committed); the committed path is the original one.
5. **Partition: use `--partition=service` on Frontier** (`#SBATCH -p service`).
6. **Don't run on the login nodes** beyond 2 cores. Build and run through SLURM.
7. **`build_rich.sh` and nested builds.** It now deletes only the contents of a build directory, keeping nested build directories that contain a `.build_cmd`. Use `--build-subdir=<name>` to keep several variants (e.g. `hip`, `omp`, `cpu`) side by side.

## 3. Where to continue (in order)

1. **Get the HIP build working with the new launch_compiler path** (caution 1).
   - Build Kokkos for Frontier if there is no suitable one: HIP + OpenMP, `Kokkos_ARCH_AMD_GFX90A=ON` and `Kokkos_ARCH_ZEN3=ON`, built with the ROCm version that will be loaded at run time.
   - Also build an OpenMP-only Kokkos for CPU comparisons.
   - Then build Mach2 (`runs/Elad_paper_mach2`) with `--with-gpu` and run a short case.
2. **Fix the Kokkos-after-MPI finalization order** (caution 2) by wiring in `MPIInitializer` in the drivers (`runs/Elad_paper_mach2`, `runs/Elad_paper_hohlraum`, `runs/moving_slab_mc`). Confirm clean exit on HIP.
3. **Benchmarks on Frontier:**
   - Mach2 on GPU (8 ranks/node, one GCD each), compared with the CPU build.
   - Hohlraum 892 with OpenMP Kokkos: 1 rank × 56 threads against 56 ranks × 1 thread (the JUPITER comparison was 1×288 vs 288×1; see the results table in `jupiter_port.md`).
   - Moving slab on GPU.
4. **Open performance items carried over from JUPITER:**
   - **Many threads per rank is ~2.5× slower than 1 thread per rank in the transport kernel** (OpenMP backend). Suspected contention on shared counters and census energy tallies in `gpu/KokkosLocalTransportExecutor.hpp`. Profile, then use per-thread tallies with a final reduction.
   - **Startup of ~243 s before main's first timer with 288 ranks** on JUPITER, probably UCX loading CUDA transports in `MPI_Init`. Check whether Frontier shows anything similar; the hohlraum driver prints `[startup]` stage timers.
   - **Moving slab load balance:** a cost function that includes particle work.
5. **MadVoro GPU/threading work** (`gpu_v1` / `gpu_v1-wip` in `voronoi`): finish, test, commit. Then bump RICH's submodule pointers for `voronoi`, `MeshDecomposer3D` and `spatial_ds`.

## 4. Prompt for the next Claude session on Frontier

> I'm continuing RICH GPU work on Frontier (OLCF; AMD EPYC + MI250X, HIP). Read `docs/frontier_handoff.md` and `docs/jupiter_port.md` in RICH first; they describe what was done on JUPITER and what is open.
>
> Setup: RICH and STORM (`source/monte`) are on branch `gpu_v1`. Check out `gpu_v1` in the `voronoi`, `MeshDecomposer3D` and `spatial_ds` submodules as described in section 1 of the handoff. Don't use the `gpu_v1-wip` branches unless I ask.
>
> Rules: don't run anything heavier than 2 cores on the login nodes. Submit builds and runs through SLURM with `--partition=service` and account `ast246`, with `--mail-type=BEGIN,END,FAIL --mail-user=maor.mizrachi@mail.huji.ac.il`. Never add `Kokkos::initialize/finalize` calls to drivers: Kokkos is started and stopped by the `KokkosInitializer` singletons (one per library). STORM must never initialize MPI. Use `runs/CrookedPipe/run_gpu_profile.sh` as the reference for Frontier modules and GPU binding.
>
> Tasks, in order:
> 1. Find or build Kokkos (HIP + OpenMP, `AMD_GFX90A` + `ZEN3`) and an OpenMP-only Kokkos. Build RICH's Mach2 case with `build_rich.sh ... --with-gpu`. The GPU build no longer uses hipcc as the global C++ compiler; it relies on Kokkos' launch_compiler (see caution 1 in the handoff). Make it work on HIP without breaking the CUDA path. If it fails, compare with `git show e03c50fb:config/set_compilers.cmake`.
> 2. Wire `MPIInitializer` (`source/utils/MPIInitializer.hpp`) into the Mach2, hohlraum and moving-slab drivers so that MPI is finalized after Kokkos. Verify a clean exit on HIP with several ranks.
> 3. Run Mach2 on GPU (8 ranks/node, one GCD each) and on CPU. Run the hohlraum (892) with OpenMP Kokkos as 1×56 and 56×1 on one node. Report the timings against the JUPITER table in `jupiter_port.md`.
> 4. Then profile the 1-rank-many-threads transport slowdown (shared-counter / census-tally contention in `source/monte/gpu/KokkosLocalTransportExecutor.hpp`) and propose a fix.
>
> Commit each step in small conventional commits on `gpu_v1`. Push the submodule first, then RICH. Ask me before anything destructive.
