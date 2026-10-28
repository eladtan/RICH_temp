"""RICH run submission: partitions, benchmarks, parallel features, plans and SLURM jobs.

The functional interface, for scripts and for THUNDER:

    from richsubmit import Context, RunSpec, make_plan, submit

    context = Context()                       # submit/config.json
    spec = RunSpec.from_dict({
        "partition": "batch", "time_limit": "02:00:00", "benchmark": "RMTV",
        "program_args": "--octant --n 96",
        "resources": {"ntasks": 56, "cores_per_task": 7, "min_memory_per_rank_mb": 8000},
        "features": {"storm.transport": {"backend": "gpu"},
                     "madvoro.build": {"backend": "openmp", "threads": 7}},
    })
    plan = make_plan(context, spec)           # errors, warnings, layout, build, scripts
    if plan.ok:
        result = submit(context, spec)        # writes submit/jobs/<stamp>_RMTV/, returns job IDs
"""

from .benchmarks import Benchmark, ProgramOption
from .features import Catalog, Feature, Resolution, execution_config, resolve, scan_sources
from .layout import Layout, Resources
from .plan import BUILD_MODES, Context, Plan, RunSpec, Submission, make_plan, submit
from .slurm import Partition, SlurmError

__all__ = [
    "BUILD_MODES", "Benchmark", "Catalog", "Context", "Feature", "Layout", "Partition", "Plan",
    "ProgramOption", "Resolution", "Resources", "RunSpec", "SlurmError", "Submission",
    "execution_config", "make_plan", "resolve", "scan_sources", "submit",
]
