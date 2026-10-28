"""From a run specification to a checked plan, its batch scripts, and a submission."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import benchmarks, features, layout, slurm

SUBMIT_DIR = Path(__file__).resolve().parent.parent
BUILD_MODES = ("in_job", "separate_job", "existing")
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Set for every rank by bind_gpus.sh; a value from the user would be overwritten
GPU_VARIABLES = ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")


@dataclass
class RunSpec:
    """Everything a run needs; JSON round-trips through to_dict/from_dict."""
    partition: str = ""
    time_limit: str = "01:00:00"
    account: str = ""
    qos: str = ""
    job_name: str = ""
    benchmark: str = ""
    program_args: str = ""
    environment: Dict[str, str] = field(default_factory=dict)  # exported in the job; $VARIABLES expand there
    build_config: str = ""
    build_mode: str = "in_job"        # in_job, separate_job (afterok dependency) or existing
    build_time_limit: str = "01:00:00"
    binary: str = ""                  # with build_mode existing: overrides the variant's binary
    resources: layout.Resources = field(default_factory=layout.Resources)
    features: Dict[str, Dict] = field(default_factory=dict)   # id -> {"backend", "threads", "gpus"}
    site: str = ""                    # "" picks the site whose match_cluster names this cluster
    sbatch_args: List[str] = field(default_factory=list)      # advanced, passed verbatim

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "RunSpec":
        data = dict(data or {})
        resources = layout.Resources.from_dict(data.pop("resources", {}))
        known = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        return cls(resources=resources, **known)


class Context:
    """The project configuration, feature catalog and (lazily) SLURM state."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = Path(config_path or SUBMIT_DIR / "config.json").resolve()
        self.config = json.loads(self.config_path.read_text())
        self.repo = (self.config_path.parent / self.config.get("repo_root", "..")).resolve()
        self.runs_dir = self.repo / self.config.get("runs_dir", "runs")
        self.jobs_dir = self.repo / self.config.get("jobs_dir", "submit/jobs")
        self.build = self.config["build"]
        self.catalog = features.load([self.repo / path for path in self.config["feature_manifests"]],
                                     self.build["variables"])
        self._partitions: Optional[List[slurm.Partition]] = None
        self._cluster: Optional[str] = None

    def partitions(self, refresh: bool = False) -> List[slurm.Partition]:
        if self._partitions is None or refresh:
            self._partitions = slurm.partitions()
        return self._partitions

    def partition(self, name: str) -> slurm.Partition:
        for candidate in self.partitions():
            if candidate.name == name:
                return candidate
        raise slurm.SlurmError(f"No SLURM partition named {name!r}")

    def cluster(self) -> str:
        if self._cluster is None:
            try:
                match = re.search(r"^ClusterName\s*=\s*(\S+)", slurm.run(["scontrol", "show", "config"]), re.M)
                self._cluster = match.group(1) if match else ""
            except slurm.SlurmError:
                self._cluster = ""
        return self._cluster

    def site(self, name: str = "") -> Dict:
        sites = self.config.get("sites", {})
        if name:
            if name not in sites:
                raise KeyError(f"No site {name!r} in {self.config_path}")
            return dict(sites[name], name=name)
        cluster = self.cluster()
        for site_name, site in sites.items():
            if site.get("match_cluster") and re.fullmatch(site["match_cluster"], cluster):
                return dict(site, name=site_name)
        return dict(sites.get("generic", {}), name="generic")

    def find_benchmark(self, name: str) -> Optional[benchmarks.Benchmark]:
        return benchmarks.find(self.runs_dir, name)

    def env_example(self, variable: str, benchmark: str, account: str = "") -> str:
        """The site's example value for a variable (only shown to the user, never applied)."""
        site = self.site()
        example = site.get("env_examples", {}).get(variable, "")
        return example.format(account=account or site.get("account", "<project>"), benchmark=benchmark or "<benchmark>")

    def benchmarks(self, with_options: bool = False) -> List[benchmarks.Benchmark]:
        return benchmarks.discover(self.runs_dir, with_options)


@dataclass
class Plan:
    spec: Dict
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    # The same messages, each tagged with the step it belongs to: partition, problem, tasks or features
    issues: List[Dict] = field(default_factory=list)
    site: str = ""
    partition: Dict = field(default_factory=dict)
    layout: Dict = field(default_factory=dict)
    resolution: Dict = field(default_factory=dict)
    gpus_per_rank: int = 0
    build_variant: str = ""
    build_command: List[str] = field(default_factory=list)
    binary: str = ""
    environment: Dict[str, str] = field(default_factory=dict)
    launch_command: str = ""
    sbatch_args: List[str] = field(default_factory=list)
    execution_config: str = ""
    run_script: str = ""
    build_script: str = ""
    user_environment: Dict[str, str] = field(default_factory=dict)  # the spec's variables, as exported

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["ok"] = self.ok
        return data


def _variant_name(benchmark: str, build: Dict[str, str], variables: Dict) -> str:
    pieces = [f"{name}.{value}" for name, value in build.items()
              if value != next(iter(variables[name]["values"]))]
    return "submit-" + "-".join([re.sub(r"[^\w.-]", "_", benchmark)] + (pieces or ["plain"]))


def _double_quoted(value: str) -> str:
    """value in double quotes for bash: $VARIABLES expand, nothing else is interpreted."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`") + '"'


def _sbatch_header(lines: List[str]) -> str:
    return "\n".join(f"#SBATCH {line}" for line in lines)


def make_plan(context: Context, spec: RunSpec, job_dir: Optional[Path] = None) -> Plan:
    """Check a spec against the partition, the features and the build, and render its scripts.

    job_dir is where the scripts, the spec and the SLURM output go; the plan
    names a placeholder when it is not known yet (previews).
    """
    plan = Plan(spec=spec.to_dict())
    errors, warnings = plan.errors, plan.warnings
    tagged = {"error": 0, "warning": 0, "note": 0}

    def tag(step: str):
        """Tag the messages added since the last call with step."""
        for level, items in (("error", plan.errors), ("warning", plan.warnings), ("note", plan.notes)):
            plan.issues += [{"level": level, "step": step, "text": text} for text in items[tagged[level]:]]
            tagged[level] = len(items)
    job_dir_text = str(job_dir) if job_dir else str(context.jobs_dir / "<timestamp>_<benchmark>")

    site = context.site(spec.site)
    plan.site = site["name"]

    # 1. Partition and time
    part = None
    if not spec.partition:
        errors.append("Choose a partition.")
    else:
        try:
            part = context.partition(spec.partition)
            plan.partition = part.to_dict()
        except slurm.SlurmError as error:
            errors.append(str(error))
    minutes = slurm.parse_time_minutes(spec.time_limit)
    if minutes is None or minutes <= 0:
        errors.append(f"Time limit {spec.time_limit!r} is not a SLURM time (e.g. 02:00:00 or 1-12:00:00).")
    elif part:
        if part.state.upper() != "UP":
            warnings.append(f"Partition {part.name} is {part.state}; the job waits until it is up.")
        limit = slurm.parse_time_minutes(part.max_time)
        if limit is not None and minutes > limit:
            errors.append(f"Time limit {spec.time_limit} exceeds the {part.max_time} of partition {part.name}.")
    if part and part.allowed_qos and spec.qos not in part.allowed_qos:
        errors.append(f"Partition {part.name} admits only QOS {', '.join(part.allowed_qos)}.")
    account = spec.account or site.get("account", "")
    tag("partition")

    # 2. Benchmark
    bench = context.find_benchmark(spec.benchmark) if spec.benchmark else None
    if not spec.benchmark:
        errors.append("Choose a benchmark.")
    elif bench is None:
        errors.append(f"No benchmark {spec.benchmark!r} in {context.runs_dir}.")
    options = [option.name for option in bench.options] if bench and bench.options else None
    user_env: Dict[str, str] = {}
    for name, value in (spec.environment or {}).items():
        name = str(name).strip()
        if not name:
            continue
        if ENV_NAME.match(name):
            user_env[name] = str(value)
        else:
            errors.append(f"{name!r} is not a valid environment variable name: use letters, digits and _, "
                          f"not starting with a digit.")
    for variable in (bench.environment if bench else []):
        if variable.required and not user_env.get(variable.name, "").strip():
            errors.append(f"{bench.name} needs {variable.name}"
                          f"{f' ({variable.help})' if variable.help else ''}. Set it under Environment variables.")
    plan.user_environment = user_env
    tag("problem")

    # 3-4. Features, build, layout
    r = spec.resources
    gpus_per_node = part.gpus_per_node if part else 0
    resolution = features.resolve(context.catalog, spec.features, context.build["variables"],
                                   cores_per_task=max(r.cores_per_task, 1), gpus_per_node=gpus_per_node,
                                   program_options=options, user_args=spec.program_args)
    errors += resolution.errors
    warnings += resolution.warnings
    plan.notes += resolution.notes
    plan.resolution = resolution.to_dict()
    tag("features")

    gpus_per_rank = 0
    if resolution.uses_gpus:
        gpus_per_rank = r.gpus_per_rank or max(resolution.max_feature_gpus, 1)
        if gpus_per_rank < resolution.max_feature_gpus:
            errors.append(f"A feature uses {resolution.max_feature_gpus} GPUs, but each rank has {gpus_per_rank}.")
        if r.ranks_per_gpu > 1 and resolution.max_feature_gpus > 1:
            errors.append("Ranks that share a GPU have one GPU each; no feature can use more.")
        if gpus_per_rank > resolution.max_feature_gpus:
            warnings.append(f"Each rank gets {gpus_per_rank} GPUs, but no feature uses more than "
                            f"{resolution.max_feature_gpus}; the rest stay idle.")
    elif r.gpus_per_rank:
        warnings.append("No feature runs on a GPU, so the GPUs per rank setting is ignored.")
    plan.gpus_per_rank = gpus_per_rank
    tag("tasks")

    mpi_build = "MPI" in (spec.build_config or context.build["default_config"])
    if part:
        placed = layout.compute(layout.Resources(**dict(asdict(r), gpus_per_rank=gpus_per_rank)),
                                part, resolution.uses_gpus)
        errors += placed.errors
        warnings += placed.warnings
        plan.layout = placed.to_dict()
        if placed.idle_cores_per_node > 0 and not placed.errors and placed.nodes > 0:
            warnings.append(f"{placed.idle_cores_per_node} of {part.cores_per_node} cores per node stay idle.")
        rule = site.get("partition_rules", {}).get(part.name, {})
        max_nodes = rule.get("max_nodes")
        if max_nodes and placed.nodes > max_nodes:
            fit = max_nodes * placed.max_ranks_per_node
            errors.append(f"Partition {part.name} runs a job on at most {max_nodes} node"
                          f"{'s' if max_nodes > 1 else ''}, but these ranks need {placed.nodes}. "
                          f"{rule.get('reason', '')} Use at most {fit} ranks with these settings, "
                          f"or choose another partition.")
    tag("tasks")
    config = spec.build_config or context.build["default_config"]
    if config not in context.build["configs"]:
        errors.append(f"Build config {config!r} is not one of {', '.join(context.build['configs'])}.")
    if r.ntasks > 1 and not mpi_build:
        errors.append(f"{r.ntasks} tasks need an MPI build config, not {config}.")

    # Build
    if spec.build_mode not in BUILD_MODES:
        errors.append(f"Build mode must be one of {', '.join(BUILD_MODES)}.")
    if resolution.build and bench:
        plan.build_variant = _variant_name(bench.name, resolution.build, context.build["variables"])
        plan.binary = str(context.repo / context.build.get("build_dir", "build") / config /
                          plan.build_variant / f"rich_{config}")
        plan.build_command = ([f"./{context.build['script']}", config, f"--test_name={bench.name}"]
                              + resolution.build_flags
                              + [f"--build-subdir={plan.build_variant}", f"--jobs={context.build.get('jobs', 16)}"])
    if spec.build_mode == "existing":
        if spec.binary:
            plan.binary = spec.binary
        if plan.binary and not Path(plan.binary).is_file():
            errors.append(f"No binary at {plan.binary}; build it first or let the job build it.")
    build_minutes = slurm.parse_time_minutes(spec.build_time_limit)
    if spec.build_mode == "separate_job" and (build_minutes is None or build_minutes <= 0):
        errors.append(f"Build time limit {spec.build_time_limit!r} is not a SLURM time.")
    tag("problem")

    # Environment
    kokkos_variant = site.get("kokkos_variants", {}).get(resolution.build.get("kokkos", ""), "")
    env_setup = [line.format(repo=context.repo, kokkos_variant=kokkos_variant)
                 for line in site.get("env_setup", [])]
    env = dict(site.get("env", {}))
    env["OMP_NUM_THREADS"] = str(resolution.pool_threads)
    env["RICH_EXECUTION_CONFIG"] = f"{job_dir_text}/execution.conf"
    env["RICH_JOB_DIR"] = job_dir_text
    if resolution.uses_gpus:
        env["RICH_SUBMIT_GPUS_PER_RANK"] = str(gpus_per_rank)
        env["RICH_SUBMIT_RANKS_PER_GPU"] = str(r.ranks_per_gpu)
        env["RICH_SUBMIT_GPU_ENV"] = site.get("gpu_visible_env", "CUDA_VISIBLE_DEVICES")
        if site.get("gpu_order"):
            env["RICH_SUBMIT_GPU_ORDER"] = " ".join(map(str, site["gpu_order"]))
        env["RICH_REPORT_BINDING"] = "1"
    env.update(resolution.env)
    for name in user_env:
        if name in env or name in GPU_VARIABLES:
            errors.append(f"{name} is set by the tool for this run (from steps 3-4), so it cannot be set under "
                          f"Environment variables; remove it there.")
    tag("problem")
    plan.environment = dict(env, **user_env)

    # Launch
    feature_args = " ".join(f"--{option} {shlex.quote(value)}" for option, value in resolution.args.items())
    program = " ".join(part_ for part_ in [shlex.quote(plan.binary or "<binary>"), spec.program_args.strip(),
                                           feature_args] if part_)
    wrapper = f"{shlex.quote(str(SUBMIT_DIR / 'bind_gpus.sh'))} " if resolution.uses_gpus else ""
    rpn = plan.layout.get("ranks_per_node") or 1
    if not mpi_build:
        plan.launch_command = f"{wrapper}{program}"
    elif site.get("launcher") == "mpirun":
        plan.launch_command = (f"mpirun -np {r.ntasks} --map-by ppr:{rpn}:node:pe={r.cores_per_task} "
                               f"--bind-to core {wrapper}{program}")
    else:
        plan.launch_command = (f"srun --ntasks={r.ntasks} --ntasks-per-node={rpn} "
                               f"--cpus-per-task={r.cores_per_task} --cpu-bind=cores {wrapper}{program}")

    # SLURM request
    job_name = spec.job_name or f"rich_{spec.benchmark or 'run'}"
    sbatch = [f"--job-name={job_name}", f"--partition={spec.partition}", f"--time={spec.time_limit}"]
    if account:
        sbatch.append(f"--account={account}")
    if spec.qos:
        sbatch.append(f"--qos={spec.qos}")
    if plan.layout.get("nodes"):
        sbatch += [f"--nodes={plan.layout['nodes']}", f"--ntasks={r.ntasks}",
                   f"--ntasks-per-node={plan.layout['ranks_per_node']}",
                   f"--cpus-per-task={r.cores_per_task}"]
        if part and part.threads_per_core > 1:
            sbatch.append("--threads-per-core=1")
        if plan.layout.get("memory_per_node_mb"):
            sbatch.append(f"--mem={plan.layout['memory_per_node_mb']}M")
        if resolution.uses_gpus and site.get("request_gpus", True):
            sbatch.append(f"--gpus-per-node={plan.layout['gpus_per_node']}")
    sbatch += list(site.get("sbatch_args", [])) + list(spec.sbatch_args)
    sbatch.append(f"--output={job_dir_text}/slurm-%j.out")
    plan.sbatch_args = sbatch

    if bench is None:
        return plan
    plan.execution_config = features.execution_config(
        context.catalog, resolution, f"{job_name}: {spec.benchmark} on {spec.partition}")

    exports = "\n".join(f"export {name}={shlex.quote(value)}" for name, value in env.items())
    if user_env:
        exports += "\n# Environment variables of the spec ($VARIABLES expand here)\n" + "\n".join(
            f"export {name}={_double_quoted(value)}" for name, value in user_env.items())
    build_lines = ""
    if spec.build_mode == "in_job" and plan.build_command:
        build_lines = (f"cd {shlex.quote(str(context.repo))}\n"
                       f"{' '.join(shlex.quote(word) for word in plan.build_command)}\n")
    plan.run_script = "\n".join([
        "#!/bin/bash",
        _sbatch_header(sbatch),
        f"# Generated by {SUBMIT_DIR}/rich-submit on {time.strftime('%Y-%m-%d %H:%M:%S')}; "
        f"spec: {job_dir_text}/spec.json",
        "",
        *env_setup,
        "set -eo pipefail",
        build_lines,
        exports,
        f"cd {shlex.quote(bench.path)}",
        f'echo "RICH run: {r.ntasks} ranks on {plan.layout.get("nodes", "?")} nodes x '
        f'{r.cores_per_task} cores, OMP_NUM_THREADS=$OMP_NUM_THREADS, build {plan.build_variant}"',
        'echo "Execution config:"; cat "$RICH_EXECUTION_CONFIG"',
        plan.launch_command,
        "",
    ])
    if spec.build_mode == "separate_job" and plan.build_command:
        build_sbatch = [f"--job-name={job_name}_build", f"--partition={spec.partition}",
                        f"--time={spec.build_time_limit}", "--nodes=1", "--ntasks=1",
                        f"--cpus-per-task={min(int(context.build.get('jobs', 16)), part.cores_per_node if part else 16)}",
                        f"--output={job_dir_text}/build-%j.out"]
        if account:
            build_sbatch.append(f"--account={account}")
        if spec.qos:
            build_sbatch.append(f"--qos={spec.qos}")
        build_sbatch += list(site.get("sbatch_args", []))
        plan.build_script = "\n".join([
            "#!/bin/bash", _sbatch_header(build_sbatch), "", *env_setup, "set -eo pipefail",
            f"cd {shlex.quote(str(context.repo))}",
            " ".join(shlex.quote(word) for word in plan.build_command), "",
        ])
    return plan


@dataclass
class Submission:
    job_dir: str
    job_ids: List[str]
    plan: Dict
    dry_run: bool
    message: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def submit(context: Context, spec: RunSpec, dry_run: bool = False) -> Submission:
    """Write the job directory and submit; dry_run checks the request with sbatch --test-only.

    Raises ValueError with the plan's errors when the spec does not check out.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w.-]", "_", spec.benchmark or "run")
    # A check writes its files to a hidden folder that is removed afterwards
    job_dir = context.jobs_dir / f"{'.check_' if dry_run else ''}{stamp}_{safe_name}"
    plan = make_plan(context, spec, job_dir)
    if not plan.ok:
        raise ValueError("\n".join(plan.errors))
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2) + "\n")
    (job_dir / "plan.json").write_text(json.dumps(plan.to_dict(), indent=2) + "\n")
    (job_dir / "execution.conf").write_text(plan.execution_config)
    run_script = job_dir / "run.sh"
    run_script.write_text(plan.run_script)
    run_script.chmod(0o755)
    build_script = None
    if plan.build_script:
        build_script = job_dir / "build.sh"
        build_script.write_text(plan.build_script)
        build_script.chmod(0o755)

    if dry_run:
        messages = []
        try:
            for script in [s for s in (build_script, run_script) if s]:
                result = subprocess.run(["sbatch", "--test-only", str(script)], stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, universal_newlines=True)
                messages.append(f"{script.name}: {result.stdout.strip()}")
                if result.returncode != 0:
                    raise ValueError("sbatch --test-only rejected the request:\n" + "\n".join(messages))
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
        return Submission(str(job_dir), [], plan.to_dict(), True, "\n".join(messages))

    job_ids = []
    if build_script:
        job_ids.append(slurm.sbatch(str(build_script), cwd=str(job_dir)))
    job_ids.append(slurm.sbatch(str(run_script), [f"--dependency=afterok:{job_ids[0]}"] if job_ids else [],
                                cwd=str(job_dir)))
    (job_dir / "job_ids").write_text(" ".join(job_ids) + "\n")
    return Submission(str(job_dir), job_ids, plan.to_dict(), False,
                      f"Submitted {' then '.join(job_ids)}; files in {job_dir}")
