# submit: configure and submit RICH runs

A small tool that turns "which partition, which problem, how many ranks, and how
each part of RICH, STORM and MadVoro runs (serial, OpenMP threads or GPUs)" into
a checked SLURM job. It is a web page served from the login node, a command
line, and a Python API, and it uses only the Python standard library (Python 3.7
or newer).

## Start it

```bash
cd ~/RICH
submit/rich-submit
```

It listens on port 8765 of the login node (or the next free port) and prints
what to do. Your browser reaches it through an SSH tunnel to that same login node:

```bash
# on your computer, in a new terminal; leave it open
ssh -N -L 8765:127.0.0.1:8765 <user>@login07.frontier.olcf.ornl.gov
```

Then open the printed `http://127.0.0.1:8765/?token=...` link in your browser.

- **Use the node it printed** (login07, login08, ...). `frontier.olcf.ornl.gov`
  can land on any login node.
- **"Address already in use"** means the port is busy on your computer, often
  because of a VS Code port forward (see the Ports tab) or an old tunnel. Change
  only the local, first number, e.g. `-L 9876:127.0.0.1:8765`, and open
  `http://127.0.0.1:9876/?token=...`.
- **To avoid a second login,** add the tunnel to the session you already have.
  Press Enter, type `~C`, then `-L 8765:127.0.0.1:8765`. OpenSSH 9.2+ needs
  `-o EnableEscapeCommandline=yes` when connecting for this to work.
- **To make it permanent,** put `LocalForward 8765 127.0.0.1:8765` under the
  login node's `Host` entry in `~/.ssh/config`.
- **In VS Code connected to that node,** Ctrl+clicking the link forwards the port
  by itself.

Ctrl+C in the terminal stops it. `--port N` picks another port.

The page only listens on the login node itself, and every request must carry the
random token in the link, because login nodes are shared.

## Using the page

The page walks through six steps; the left column shows them, each with what you
chose and a mark: a check when it is fine, an exclamation mark when something
there must be fixed. The right column sums up the run in plain words. Every step
lists its own problems at the top, and each field says what it is for.

1. **Partition & time.** One card per partition: its nodes, and the cores,
   memory and GPUs of one node, and how long a job may run. Pick a time limit
   from the presets or type one. A partition that requires a QOS sets it.
2. **Problem.** Search the benchmarks under `runs/`, pick one, and write its
   arguments; the options the benchmark understands are listed (hover to read,
   click to add). Add **environment variables** for the job (exported before the
   program starts; values may use `$USER`, `$SLURM_JOB_ID` and other variables).
   Variables the benchmark requires (declared with `EnvironmentParser`, below)
   appear by themselves and must be filled in, e.g. `RICH_OUTPUT_DIR`, the
   directory for the results, which has no default (the page offers an example you
   can apply); optional ones are offered as buttons. Choose how to build: in the job, in a separate build
   job that the run waits for, or not at all (an existing binary).
3. **Tasks.** The total number of MPI ranks, the cores per rank, and optionally
   ranks per node and a minimum memory per rank. On partitions with GPUs: GPUs
   per rank (at most a node's GPUs) and how many ranks share one GPU. A picture
   shows how the ranks fill each node's cores and GPUs, and what limits it.
4. **Parallel features.** For each part of RICH, STORM and MadVoro: Serial,
   OpenMP (with a thread count, at most the cores per rank) or GPU (only on
   partitions with GPUs). Parts that cannot run a way yet show it crossed out,
   with the reason. Each part shows what it actually runs on, and the bottom line
   says which build these choices need.
5. **Advanced.** Coming later.
6. **Review & submit.** The whole run in plain words, anything left to fix (with
   a link to its step), and the generated files. **Check with Slurm** asks
   Slurm whether it would accept the job (`sbatch --test-only`; nothing is
   submitted). **Submit job** submits it after a confirmation.

**Save setup** and **Load saved setup** keep setups in `submit/specs/`.
**My jobs** lists the jobs submitted from the page with their Slurm state, and
shows their output.

A submission writes `submit/jobs/<timestamp>_<benchmark>/`, containing
`spec.json`, `plan.json`, `run.sh`, optionally `build.sh`, `execution.conf`,
`job_ids` and the SLURM output. The program runs from its `runs/<benchmark>`
directory.

## Command line

```bash
submit/rich-submit partitions           # what SLURM offers: cores, memory and GPUs per node, time limits
submit/rich-submit benchmarks           # the problems under runs/
submit/rich-submit features             # the parallel features and their backends
submit/rich-submit plan examples/rmtv_gpu_subdomains.json        # errors, warnings and the job script
submit/rich-submit submit --dry-run examples/rmtv_gpu_subdomains.json   # sbatch --test-only
submit/rich-submit submit examples/rmtv_gpu_subdomains.json      # submit
```

On Frontier the system `python3` is 3.6, so `rich-submit` picks a newer one by
itself (or loads `python/3.10-miniforge3`). `RICH_SUBMIT_PYTHON` picks one
explicitly.

## How a feature choice reaches the run

Most of today's choices are fixed when RICH is compiled:

- **`--with-gpu`** (`STORM_WITH_GPU`) turns Kokkos on for STORM, MadVoro and
  RICH together.
- **The Kokkos install** that `runs/frontier_env.sh hip|omp` points at decides
  whether Kokkos runs on the GPU or on OpenMP threads.
- **`--voronoi-on-host`** keeps MadVoro's geometry on host threads.

Each backend in a manifest names the build-variable values it can run with. The
planner picks the first build that satisfies every feature and names the
features that conflict when no build does. Each build variant gets its own
build directory, `build/<config>/submit-<benchmark>-<variant>/`.

At run time a choice reaches the code in three ways:

- **`OMP_NUM_THREADS`** is the largest OpenMP thread count over all features.
  Features marked `"threads": "global"` share that one pool, and the plan warns
  when one of them asks for fewer threads.
- **Program arguments** a backend declares, such as `--voronoi-subdomains
  {threads}`, are added when the benchmark has that option. Your own arguments
  take precedence.
- **The execution config.** The job exports `RICH_EXECUTION_CONFIG`, a file with
  one line per feature (`feature backend threads gpus`). C++ code reads it with
  `source/utils/ExecutionConfig.hpp`:

  ```cpp
  #include "utils/ExecutionConfig.hpp"
  slopes.RunOnDevice(rich::execution::OnDevice("rich.hydro.slopes"));
  const int threads = rich::execution::Threads("madvoro.build.subdomains", HostThreads());
  ```

  Without the variable, or for a feature the file does not name, each query
  returns its fallback, so runs started by hand behave as before.
  `rich-submit features --check` lists the feature ids that code reads and the
  ones nothing reads yet.

GPUs are bound by `submit/bind_gpus.sh`. It runs under `mpirun`/`srun` and gives
each rank its GPUs from its local rank, in the site's GPU order. On Frontier
that order is the GCD next to each CCD, so ranks placed on consecutive cores
get their nearest GCD.

## Adding or changing a feature

Edit `submit/features/<project>.json`:

```json
{"id": "madvoro.build.subdomains", "title": "Thread subdomains", "default": "serial",
 "backends": {
   "serial": {"args": {"voronoi-subdomains": "0"}},
   "openmp": {"build": [{"kokkos": "host"}, {"kokkos": "device"}], "threads": "feature",
              "args": {"voronoi-subdomains": "{threads}"}}}}
```

A backend can set these keys:

| Key | Meaning |
| --- | --- |
| `build` | alternatives of build-variable values (variables are in `config.json`) |
| `threads` | `global` (the shared pool) or `feature` (honours its own count) |
| `max_gpus` | GPUs per rank it can use (Kokkos: 1) |
| `args`, `env` | program options and environment, with `{threads}` and `{gpus}` |
| `requires` | backends other features must have |
| `available` | `false` while the code cannot honour it; set `note` to say why |

`subfeatures` nest, and `inherit` is a backend for subfeatures. Manifest
`rules` add a warning, or a note with `"level": "note"`, when features meet
given backends.

When code starts reading a feature through `ExecutionConfig.hpp`, do two things
in its manifest entry:

- Mark its OpenMP backend `"threads": "feature"`.
- Set `"available": true` on GPU backends that now work, such as the hydro
  kernels' `RunOnDevice`.

## Python API (for THUNDER and scripts)

```python
from richsubmit import Context, RunSpec, make_plan, submit

context = Context()                                   # submit/config.json
spec = RunSpec.from_dict({
    "partition": "batch", "time_limit": "02:00:00", "benchmark": "RMTV",
    "program_args": "--octant --n 96",
    "environment": {"MY_SETTING": "on"},       # exported in the job; $VARIABLES expand
    "resources": {"ntasks": 56, "cores_per_task": 7, "min_memory_per_rank_mb": 16000},
    "features": {"storm.transport": {"backend": "gpu"},
                 "madvoro.build.subdomains": {"backend": "openmp", "threads": 7}},
})
plan = make_plan(context, spec)        # .ok, .errors, .warnings, .layout, .run_script, ...
if plan.ok:
    result = submit(context, spec)     # .job_ids, .job_dir; submit(..., dry_run=True) for --test-only
```

Add `submit/` to `sys.path`, or run Python with `PYTHONPATH=submit`. A spec is
plain JSON (`RunSpec.to_dict()`), so a THUNDER test can carry one, for example
as a `SUBMIT_SPEC` file next to its `REGRESSION_INFO`.

## Environment variables a benchmark reads

Benchmarks declare the environment variables they read with
`source/utils/arguments/EnvironmentParser.hpp`, the way they declare options
with `ArgumentParser`:

```cpp
#include "utils/arguments/EnvironmentParser.hpp"

EnvironmentParser environment("Mach2 benchmark");
environment.addRequired<std::string>("RICH_OUTPUT_DIR", "directory the benchmark writes its results to");
environment.addOptional<int>("STORM_SEED", 7, "Monte Carlo seed");
environment.parse();  // throws, naming every missing or malformed variable, then the help
const std::string dir = environment.get<std::string>("RICH_OUTPUT_DIR");
```

The page reads these declarations from the benchmark's sources and the `runs/`
headers they include, without building anything. A required variable gets its
own row that must be filled in before the job can be submitted; optional ones
are offered as buttons, with their default and description. Keep the name and
description string literals so they can be read. `runs/mc_results_dir.hpp`
declares `RICH_OUTPUT_DIR` this way, so every benchmark that uses
`McResultsDirectory()` asks for it.

A site's `env_examples` are only shown as suggestions. Variables the tool sets
itself (`OMP_NUM_THREADS`, `RICH_EXECUTION_CONFIG`, the GPU binding variables, ...)
cannot be set this way.

## Sites

`config.json` describes each cluster:

- environment setup lines, e.g. `source runs/frontier_env.sh {kokkos_variant}`
- the launcher, `mpirun` or `srun`
- the default account
- the GPU visibility variable and GPU order
- extra `sbatch` flags
- whether to request GPUs explicitly

The site is chosen by matching SLURM's `ClusterName`, with `generic` as the
fallback. The build variables and their `build_rich.sh` flags are in the same
file.

## Tests

```bash
python3 -m unittest discover -s submit/tests     # no SLURM needed
```
