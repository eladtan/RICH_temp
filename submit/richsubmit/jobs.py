"""Job directories written by submit() and the SLURM state of their jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from . import slurm


def states(job_ids: List[str]) -> Dict[str, Dict[str, str]]:
    """{job id: {"state", "elapsed"}} from sacct; empty when sacct is unavailable."""
    if not job_ids:
        return {}
    try:
        text = slurm.run(["sacct", "-n", "-P", "-X", "-o", "JobID,State,Elapsed", "-j", ",".join(job_ids)])
    except slurm.SlurmError:
        return {}
    found = {}
    for line in text.splitlines():
        fields = line.split("|")
        if len(fields) >= 3:
            found[fields[0]] = {"state": fields[1], "elapsed": fields[2]}
    return found


def recent(jobs_dir: Path, limit: int = 30) -> List[Dict]:
    """The newest job directories, each with its job IDs, their states and its SLURM output files."""
    jobs = []
    if jobs_dir.is_dir():
        visible = (d for d in jobs_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
        for directory in sorted(visible, reverse=True)[:limit]:
            ids_file = directory / "job_ids"
            ids = ids_file.read_text().split() if ids_file.is_file() else []
            outputs = sorted(str(path) for path in directory.glob("*.out"))
            jobs.append({"dir": str(directory), "name": directory.name, "job_ids": ids, "outputs": outputs})
    found = states([job_id for job in jobs for job_id in job["job_ids"]])
    for job in jobs:
        job["states"] = {job_id: found.get(job_id, {}) for job_id in job["job_ids"]}
    return jobs
