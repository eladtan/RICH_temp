# FMM individual-timestep benchmark

This MPI benchmark answers two bounded questions.

1. Correctness: the `parity` global and `individual_sync` lanes start from the
   same nonuniform moving gas, use nonzero FMM self-gravity, and advance sixteen
   `1e-4` ticks to the same physical end time (`0.0016`) used by the original
   eight-tick trial. The finer step tests convergence without weakening any
   endpoint tolerance. Before time integration, an independent probe evaluates the
   same cells through the full FMM and individual-target FMM APIs, seeds the
   retained target solver, reverses every rank's local source order while
   retaining persistent cell IDs, and requires normalized acceleration
   agreement within `5e-13`. End states are compared
   by field: position `5e-8`, thermodynamic/extensive scalars `5e-5`, volume
   `5e-7`, and velocity/momentum `3e-3`. These endpoint tolerances account for
   the global RK2 source evaluation and the individual kick-cache integrator
   being distinct second-order schemes. A second side-2 global/sparse pair runs
   eight cells for sixteen ticks on 16 ranks and requires both zero-owned and
   nonzero-owned ranks.
   An untimed one-rank, eight-cell contract probe changes active tracer
   extensives between gravity half-kicks and requires the individual-target
   source to observe tracer 0 on its first call and the recovered value 0.75 on
   its second. This covers state-dependent accelerations such as the TDE tracer
   mask without contaminating measured timing lanes.
2. Benefit: the `performance` lanes use a synthetic, prescribed local-timestep
   distribution with spatially nested bins 0, 2, and 4 on an exact Cartesian
   mesh. The central `2x2x2` cube has bin 0, the rest of the central `8x8x8`
   cube has bin 2, and the exterior has bin 4. On the measured side-48 grid
   this is 8, 504, and 110080 cells, respectively: a 15.77x reduction in active
   updates while keeping fine-event partial meshes local. With a `2e-4` base
   quantum, the largest step is `0.0032`, below the acoustic
   cell-crossing time even on the measured side-48 grid. The
   initially uniform, cold (`density=1`, `pressure=1e-8`) Lagrangian gas
   evolves under weak, nonzero self-gravity. Disabling mesh regularization in
   this lane avoids unrelated regularization transients while isolating the
   scheduler and partial hydro/mesh work in a deliberately favorable
   gravity-dominated multirate workload.
   Global and sparse-individual lanes stop at synchronized tick 64. Every
   measured lane must have nonzero motion, conserve mass, and agree by explicit
   field tolerances. The sparse lane must reduce active-cell updates by at least
   8x. After two short warmups, three interleaved global/individual pairs must
   have median wall-clock speedup at least 2x, every pair at least 1.5x, and
   timing coefficient of variation no larger than 0.20.

This is not a physical TDE speed benchmark. The prescribed bins are a synthetic
scheduler hierarchy; they are not derived from the uniform gas's local CFL
limits, although the largest prescribed step is CFL-safe.
Also, the current individual-target FMM adapter still performs one full-source
FMM solve per gravity evaluation. The benchmark records FMM calls, source work,
targets, and FMM wall time explicitly. Overall speedup can come from two bounded
mechanisms: the individual kick cache needs fewer full-source FMM evaluations
than global RK2, and inactive cells reduce hydro/mesh work. It must not be
reported as sparse-source FMM acceleration within one evaluation.

Initial mesh construction and state writing are outside the timed region. The
build wrapper records the exact source manifest beside the binary, and the
runner refuses any binary/source pair that does not match that build record.
The runner freezes the executable and all benchmark scripts into a new campaign
root; records source, Git, compiler, module, MPI, linked-library, allocation,
and CPU-topology evidence; then rehashes the frozen files and the live runner
before every lane. Every 16-rank launch uses explicit core mapping/binding and a
20-minute process-group timeout, followed by bounded TERM/KILL drainage and an
explicit empty-group check. After the campaign root and frozen analyzer exist,
lane failure or Slurm termination still runs that analyzer while preserving the
initiating nonzero status unless process-group cleanup fails, which a lane
reports as status 92. Preflight failures exit before creating a root; a
root-created bootstrap failure records exit files and `BOOTSTRAP_FAILURE`.
The frozen analyzer runs with a pinned Python 3.12.1 library path applied only
to Python, not MPI lanes. The runner preflights startup, standard-library
imports, and shared-library resolution and records `PYTHON_RUNTIME.txt`.

Build in a unique directory. The wrapper loads the pinned GCC/OpenMPI/HDF5,
FFTW, VTK, JsonCpp, and Boost modules and rejects source changes during build:

```bash
bash runs/FmmIndividualTimestepBenchmark/build_benchmark.sh \
  fmm_individual_benchmark_<timestamp> 8
```

Submit with a campaign path that does not yet exist:

```bash
repo_root="$(pwd)"
case_dir="${repo_root}/runs/FmmIndividualTimestepBenchmark"
campaign_root="${repo_root}/regression_tests/results/fmm_individual_benchmark_<timestamp>"
binary="${repo_root}/build/gnuReleaseMPI/fmm_individual_benchmark_<timestamp>/rich_gnuReleaseMPI"
build_provenance="$(dirname "${binary}")/FMM_INDIVIDUAL_BUILD_PROVENANCE.txt"
binary_sha="$(awk -F= '$1 == "binary_sha256" {print $2}' "${build_provenance}")"
source_manifest_sha="$(awk -F= '$1 == "source_manifest_sha256" {print $2}' "${build_provenance}")"
sbatch \
  --output="${repo_root}/regression_tests/results/fmm_individual_slurm_%j.out" \
  --export=ALL,RICH_FMM_ITS_CASE_DIR="${case_dir}",RICH_FMM_ITS_CAMPAIGN_ROOT="${campaign_root}",RICH_FMM_ITS_BINARY="${binary}",RICH_FMM_ITS_BINARY_SHA256="${binary_sha}",RICH_FMM_ITS_SOURCE_MANIFEST_SHA256="${source_manifest_sha}" \
  "${case_dir}/submit.sbatch"
```

The authoritative terminal artifacts are `campaign_exit_code.txt`,
`analyzer_exit_code.txt`, `verdict.txt`, `analysis.json`, and `ANALYSIS.md` in
the campaign root.
