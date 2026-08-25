#!/usr/bin/env python3
"""Capture immutable inputs for a disjoint AutoPartial performance lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable


CODE_ROOTS = (
    "source",
    "tests",
)
BENCHMARK_ROOT = "regression_tests/cases/lane_radiation_shock_individual"
SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".in", ".md", ".py",
    ".sbatch", ".sh", ".yaml", ".yml",
}
SOURCE_BASENAMES = {"CMakeLists.txt", "SConscript", "SConstruct", "build_rich.sh"}
SOURCE_FILES = (
    "SConstruct",
    "build_rich.sh",
)
ENV_PREFIXES = (
    "RICH_",
    "SLURM_",
    "OMP_",
    "OMPI_",
    "I_MPI_",
    "FI_",
    "UCX_",
)
ENV_NAMES = {
    "LD_LIBRARY_PATH",
    "PATH",
    "PYTHONNOUSERSITE",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(repo: Path, *command: str) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            cwd=repo,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
        }
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.rstrip(),
        "stderr": completed.stderr.rstrip(),
    }


def source_paths(repo: Path) -> list[Path]:
    result: set[Path] = set()
    for root_name in CODE_ROOTS:
        root = repo / root_name
        if root.is_dir():
            result.update(
                path
                for path in root.rglob("*")
                if path.is_file()
                and (path.suffix in SOURCE_SUFFIXES or
                     path.name in SOURCE_BASENAMES)
            )
    benchmark_root = repo / BENCHMARK_ROOT
    if benchmark_root.is_dir():
        result.update(
            path
            for path in benchmark_root.rglob("*")
            if path.is_file()
            and (path.suffix in SOURCE_SUFFIXES or
                 path.name in SOURCE_BASENAMES)
        )
    for file_name in SOURCE_FILES:
        path = repo / file_name
        if path.is_file():
            result.add(path)
    return sorted(result, key=lambda path: path.relative_to(repo).as_posix())


def file_records(repo: Path, paths: Iterable[Path]) -> list[dict[str, object]]:
    records = []
    for path in paths:
        stat = path.stat()
        records.append(
            {
                "path": path.relative_to(repo).as_posix(),
                "size": stat.st_size,
                "mode": oct(stat.st_mode & 0o7777),
                "sha256": sha256_file(path),
            }
        )
    return records


def aggregate_digest(records: list[dict[str, object]]) -> str:
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_record(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--sidecar-glob",
        required=True,
        help="Absolute or repo-relative glob selecting checkpoint sidecars",
    )
    parser.add_argument("--expected-sidecars", type=int, default=128)
    parser.add_argument("--config", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve(strict=True)
    output = args.output.resolve()
    binary = args.binary if args.binary.is_absolute() else repo / args.binary
    checkpoint = (
        args.checkpoint
        if args.checkpoint.is_absolute()
        else repo / args.checkpoint
    )
    sidecar_pattern = Path(args.sidecar_glob)
    if sidecar_pattern.is_absolute():
        sidecar_parent = sidecar_pattern.parent
        sidecars = sorted(sidecar_parent.glob(sidecar_pattern.name))
    else:
        sidecars = sorted(repo.glob(args.sidecar_glob))
    if len(sidecars) != args.expected_sidecars:
        raise SystemExit(
            f"expected {args.expected_sidecars} sidecars, found {len(sidecars)}"
        )

    source_records = file_records(repo, source_paths(repo))
    configurations = []
    for config in args.config:
        path = config if config.is_absolute() else repo / config
        configurations.append(artifact_record(path))

    environment = {
        name: value
        for name, value in sorted(os.environ.items())
        if name in ENV_NAMES or name.startswith(ENV_PREFIXES)
    }
    manifest: dict[str, object] = {
        "schema": "rich-autopartial-performance-provenance-v1",
        "repository": str(repo),
        "git": {
            "head": run(repo, "git", "rev-parse", "HEAD"),
            "branch": run(repo, "git", "branch", "--show-current"),
            "status": run(repo, "git", "status", "--porcelain=v1", "-uall"),
            "submodules": run(repo, "git", "submodule", "status", "--recursive"),
        },
        "source_tree_sha256": aggregate_digest(source_records),
        "source_files": source_records,
        "binary": artifact_record(binary),
        "checkpoint": artifact_record(checkpoint),
        "checkpoint_sidecars": [artifact_record(path) for path in sidecars],
        "configuration": configurations,
        "environment": environment,
        "toolchain": {
            "cxx": run(repo, "mpicxx", "--version"),
            "mpi": run(repo, "mpirun", "--version"),
            "cpu": run(repo, "lscpu"),
            "affinity": run(repo, "taskset", "-pc", str(os.getpid())),
        },
    }
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest["manifest_content_sha256"] = hashlib.sha256(canonical).hexdigest()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, output)
    print(manifest["manifest_content_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
