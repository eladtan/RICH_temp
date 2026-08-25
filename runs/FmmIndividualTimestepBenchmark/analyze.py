#!/software/x86_64/5.14.0/python/3.12.1/bin/python3
"""Validate FMM individual-timestep parity, sparse work, and wall time."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


STATE_FIELDS = (
    "x", "y", "z", "density", "pressure", "specific_internal_energy",
    "vx", "vy", "vz", "mass", "momx", "momy", "momz",
    "total_energy", "extensive_internal_energy", "volume",
)
POSITIVE_STATE_FIELDS = (
    "density", "pressure", "specific_internal_energy", "mass",
    "total_energy", "extensive_internal_energy", "volume",
)
EXPECTED_RANKS = 16
MASS_DRIFT_TOLERANCE = 5.0e-11
INITIAL_STATE_TOLERANCE = 2.0e-13
FMM_PROBE_TOLERANCE = 5.0e-13
PARITY_FIELD_TOLERANCES = {
    "x": 5.0e-8,
    "y": 5.0e-8,
    "z": 5.0e-8,
    "density": 5.0e-5,
    "pressure": 5.0e-5,
    "specific_internal_energy": 5.0e-5,
    "vx": 3.0e-3,
    "vy": 3.0e-3,
    "vz": 3.0e-3,
    "mass": 5.0e-5,
    "momx": 3.0e-3,
    "momy": 3.0e-3,
    "momz": 3.0e-3,
    "total_energy": 5.0e-5,
    "extensive_internal_energy": 5.0e-5,
    "volume": 5.0e-7,
}
PERFORMANCE_FIELD_TOLERANCES = {
    "x": 5.0e-8,
    "y": 5.0e-8,
    "z": 5.0e-8,
    "density": 2.0e-5,
    "pressure": 2.0e-5,
    "specific_internal_energy": 2.0e-5,
    "vx": 2.0e-2,
    "vy": 2.0e-2,
    "vz": 2.0e-2,
    "mass": 2.0e-5,
    "momx": 2.0e-2,
    "momy": 2.0e-2,
    "momz": 2.0e-2,
    "total_energy": 2.0e-5,
    "extensive_internal_energy": 2.0e-5,
    "volume": 2.0e-5,
}
MINIMUM_DYNAMIC_SPEED = 1.0e-9
MINIMUM_DYNAMIC_DISPLACEMENT = 1.0e-12
MINIMUM_ACTIVE_WORK_REDUCTION = 8.0
MINIMUM_MEDIAN_SPEEDUP = 2.0
MINIMUM_PAIRED_SPEEDUP = 1.5
MAXIMUM_TIMING_CV = 0.20


LANES = {
    "warmup_global": ("performance", "global", 48, 4),
    "warmup_individual": ("performance", "individual_sparse", 48, 4),
    "empty_global": ("parity", "global", 2, 16),
    "empty_individual": ("parity", "individual_sparse", 2, 16),
    "parity_global": ("parity", "global", 20, 16),
    "parity_individual": ("parity", "individual_sync", 20, 16),
    "perf_global_1": ("performance", "global", 48, 64),
    "perf_individual_1": ("performance", "individual_sparse", 48, 64),
    "perf_individual_2": ("performance", "individual_sparse", 48, 64),
    "perf_global_2": ("performance", "global", 48, 64),
    "perf_global_3": ("performance", "global", 48, 64),
    "perf_individual_3": ("performance", "individual_sparse", 48, 64),
}


class Verdict:
    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []

    def require(self, condition: bool, name: str, detail: str) -> None:
        self.checks.append({"name": name, "pass": bool(condition), "detail": detail})

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(bool(check["pass"]) for check in self.checks)


def read_metrics(path: Path) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing metrics file: {path}")
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split(maxsplit=1)
        if len(fields) != 2 or fields[0] in result:
            raise RuntimeError(f"malformed or duplicate metrics line in {path}: {raw_line!r}")
        result[fields[0]] = fields[1]
    return result


def read_provenance(path: Path) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing provenance file: {path}")
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split("=", 1)
        if len(fields) != 2 or not fields[0] or fields[0] in result:
            raise RuntimeError(f"malformed or duplicate provenance line: {raw_line!r}")
        result[fields[0]] = fields[1]
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_campaign_provenance(verdict: Verdict, root: Path) -> dict[str, str]:
    verdict.require(root.is_dir(), "campaign.root", str(root))
    for marker in ("ROOT_WAS_FRESH", "campaign_started", "campaign_lanes_finished"):
        path = root / marker
        verdict.require(path.is_file() and path.stat().st_size > 0,
                        f"campaign.{marker}", str(path))
    fresh_marker = (root / "ROOT_WAS_FRESH").read_text(encoding="utf-8").strip()
    verdict.require(fresh_marker == "1", "campaign.fresh_root", fresh_marker)

    provenance = read_provenance(root / "CAMPAIGN_PROVENANCE.txt")
    expected_provenance = {
        "root_was_fresh": "1",
        "rank_count": str(EXPECTED_RANKS),
        "empty_side": "2",
        "parity_side": "20",
        "performance_side": "48",
        "lane_timeout_seconds": "1200",
        "mpi_binding": "core",
        "mpi_mapping": "core",
    }
    for key, expected in expected_provenance.items():
        verdict.require(provenance.get(key) == expected, f"campaign.provenance.{key}",
                        f"actual={provenance.get(key)!r}, expected={expected!r}")
    slurm_job_id = provenance.get("slurm_job_id", "")
    verdict.require(re.fullmatch(r"[0-9]+", slurm_job_id) is not None,
                    "campaign.slurm_job_id", slurm_job_id)

    evidence_files = (
        "SOURCE_MANIFEST.sha256",
        "BUILD_PROVENANCE.txt",
        "BUILD_SOURCE_MANIFEST.sha256",
        "BUILD_ENVIRONMENT.txt",
        "FROZEN_ARTIFACTS.sha256",
        "git_status.txt",
        "tracked_diff.patch",
        "untracked_files.txt",
        "submodule_status.txt",
        "compiler_mpi_identity.txt",
        "loaded_modules.txt",
        "binary_ldd.txt",
        "PYTHON_RUNTIME.txt",
        "cpu_topology.txt",
        "allocation_identity.txt",
    )
    for name in evidence_files:
        verdict.require((root / name).is_file(), f"campaign.evidence.{name}", str(root / name))
    for name in ("SOURCE_MANIFEST.sha256", "BUILD_PROVENANCE.txt",
                 "BUILD_SOURCE_MANIFEST.sha256", "BUILD_ENVIRONMENT.txt",
                 "FROZEN_ARTIFACTS.sha256", "compiler_mpi_identity.txt",
                  "loaded_modules.txt", "binary_ldd.txt", "cpu_topology.txt",
                  "allocation_identity.txt", "PYTHON_RUNTIME.txt"):
        verdict.require((root / name).stat().st_size > 0,
                        f"campaign.evidence_nonempty.{name}", str(root / name))

    python_runtime = (root / "PYTHON_RUNTIME.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    verdict.require(
        "python_binary=/software/x86_64/5.14.0/python/3.12.1/bin/python3" in
        python_runtime,
        "campaign.python_binary", "pinned Python executable recorded",
    )
    verdict.require(
        "python_library_directory=/software/x86_64/5.14.0/python/3.12.1/lib" in
        python_runtime,
        "campaign.python_library", "pinned Python library directory recorded",
    )
    verdict.require("python_version=Python 3.12.1" in python_runtime and
                    "python_preflight_exit=0" in python_runtime and
                    "not found" not in python_runtime,
                    "campaign.python_preflight",
                    "Python 3.12.1 started and all shared libraries resolved")

    allocation_identity = (root / "allocation_identity.txt").read_text(
        encoding="utf-8", errors="replace"
    )
    allocation_job = re.search(r"\bJobId=([0-9]+)\b", allocation_identity)
    allocation_nodes = re.search(r"\bNumNodes=([0-9]+)\b", allocation_identity)
    allocation_tasks = re.search(r"\bNumTasks=([0-9]+)\b", allocation_identity)
    allocation_sharing = re.search(r"\bOverSubscribe=([^\s]+)", allocation_identity)
    verdict.require(allocation_job is not None and allocation_job.group(1) == slurm_job_id,
                    "campaign.allocation_job_id",
                    allocation_job.group(1) if allocation_job else "missing")
    verdict.require(allocation_nodes is not None and allocation_nodes.group(1) == "1",
                    "campaign.allocation_nodes",
                    allocation_nodes.group(1) if allocation_nodes else "missing")
    verdict.require(allocation_tasks is not None and allocation_tasks.group(1) == str(EXPECTED_RANKS),
                    "campaign.allocation_tasks",
                    allocation_tasks.group(1) if allocation_tasks else "missing")
    verdict.require(allocation_sharing is not None and allocation_sharing.group(1) == "NO",
                    "campaign.allocation_exclusive",
                    allocation_sharing.group(1) if allocation_sharing else "missing")

    source_manifest_sha = file_sha256(root / "SOURCE_MANIFEST.sha256")
    verdict.require(source_manifest_sha == provenance.get("source_manifest_sha256"),
                    "campaign.source_manifest_sha256",
                    f"actual={source_manifest_sha}, recorded={provenance.get('source_manifest_sha256')}")

    build_provenance_path = root / "BUILD_PROVENANCE.txt"
    build_provenance = read_provenance(build_provenance_path)
    verdict.require(file_sha256(build_provenance_path) == provenance.get("build_provenance_sha256"),
                    "campaign.build_provenance_sha256",
                    f"actual={file_sha256(build_provenance_path)}, recorded={provenance.get('build_provenance_sha256')}")
    verdict.require(build_provenance.get("binary_sha256") == provenance.get("binary_sha256"),
                    "campaign.build_binary_binding",
                    f"build={build_provenance.get('binary_sha256')}, campaign={provenance.get('binary_sha256')}")
    verdict.require(build_provenance.get("source_manifest_sha256") == source_manifest_sha,
                    "campaign.build_source_binding",
                    f"build={build_provenance.get('source_manifest_sha256')}, campaign={source_manifest_sha}")
    build_source_path = root / "BUILD_SOURCE_MANIFEST.sha256"
    verdict.require(file_sha256(build_source_path) == source_manifest_sha and
                    build_source_path.read_bytes() == (root / "SOURCE_MANIFEST.sha256").read_bytes(),
                    "campaign.build_source_manifest_content",
                    f"build_sha={file_sha256(build_source_path)}, campaign_sha={source_manifest_sha}")
    build_environment_sha = file_sha256(root / "BUILD_ENVIRONMENT.txt")
    verdict.require(build_environment_sha == build_provenance.get("build_environment_sha256"),
                    "campaign.build_environment_binding",
                    f"actual={build_environment_sha}, recorded={build_provenance.get('build_environment_sha256')}")
    verdict.require(build_provenance.get("source_head") == provenance.get("source_head"),
                    "campaign.build_source_head",
                    f"build={build_provenance.get('source_head')}, campaign={provenance.get('source_head')}")

    manifest_path = root / "FROZEN_ARTIFACTS.sha256"
    frozen_entries = manifest_path.read_text(encoding="utf-8").splitlines()
    verdict.require(len(frozen_entries) == 11, "campaign.frozen_artifact_count",
                    f"actual={len(frozen_entries)}, expected=11")
    frozen_root = root.resolve()
    frozen_binary_sha = ""
    frozen_runner_sha = ""
    for line_number, raw_line in enumerate(frozen_entries, 1):
        fields = raw_line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise RuntimeError(f"malformed frozen manifest line {line_number}")
        expected_sha, relative_name = fields[0], fields[1].lstrip("*")
        target = (root / relative_name).resolve()
        try:
            target.relative_to(frozen_root)
        except ValueError as error:
            raise RuntimeError(f"frozen manifest path escapes campaign root: {relative_name}") from error
        actual_sha = file_sha256(target)
        verdict.require(actual_sha == expected_sha,
                        f"campaign.frozen_sha256.{relative_name}",
                        f"actual={actual_sha}, expected={expected_sha}")
        if relative_name == "frozen/rich_fmm_individual_benchmark":
            frozen_binary_sha = actual_sha
        elif relative_name == "frozen/run_campaign.sh":
            frozen_runner_sha = actual_sha
    verdict.require(frozen_binary_sha == provenance.get("binary_sha256"),
                    "campaign.binary_sha256",
                    f"actual={frozen_binary_sha}, recorded={provenance.get('binary_sha256')}")
    verdict.require(frozen_runner_sha == provenance.get("live_runner_sha256") ==
                    provenance.get("frozen_runner_sha256"),
                    "campaign.runner_sha256",
                    f"frozen={frozen_runner_sha}, live={provenance.get('live_runner_sha256')}, recorded_frozen={provenance.get('frozen_runner_sha256')}")
    return provenance


def number(metrics: dict[str, str], key: str) -> float:
    if key not in metrics:
        raise RuntimeError(f"missing metric {key!r}")
    value = float(metrics[key])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite metric {key!r}: {metrics[key]!r}")
    return value


def integer(metrics: dict[str, str], key: str) -> int:
    value = number(metrics, key)
    if value < 0 or value != math.floor(value):
        raise RuntimeError(f"metric {key!r} is not a nonnegative integer: {value}")
    return int(value)


def load_state(directory: Path, stem: str) -> tuple[dict[int, tuple[float, ...]], int]:
    files = sorted(directory.glob(f"{stem}_rank_*.tsv"))
    if not files:
        raise RuntimeError(f"no {stem} files in {directory}")
    expected_header = ("id",) + STATE_FIELDS
    result: dict[int, tuple[float, ...]] = {}
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or tuple(lines[0].split("\t")) != expected_header:
            raise RuntimeError(f"empty or malformed state header in {path}")
        for line_number, line in enumerate(lines[1:], 2):
            fields = line.split("\t")
            if len(fields) != len(expected_header):
                raise RuntimeError(f"malformed {path}:{line_number}")
            cell_id = int(fields[0])
            values = tuple(float(value) for value in fields[1:])
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"non-finite state at {path}:{line_number}")
            if cell_id in result:
                raise RuntimeError(f"duplicate cell ID {cell_id} across {directory}")
            result[cell_id] = values
    return result, len(files)


def compare_states(
    left: dict[int, tuple[float, ...]],
    right: dict[int, tuple[float, ...]],
) -> dict[str, object]:
    if left.keys() != right.keys():
        raise RuntimeError("state cell-ID sets differ")
    field_maximum_absolute_error = {field: 0.0 for field in STATE_FIELDS}
    field_maximum_magnitude = {field: 0.0 for field in STATE_FIELDS}
    field_representative_cell = {field: -1 for field in STATE_FIELDS}
    for cell_id in sorted(left):
        for index, field in enumerate(STATE_FIELDS):
            left_value = left[cell_id][index]
            right_value = right[cell_id][index]
            difference = abs(left_value - right_value)
            magnitude = max(abs(left_value), abs(right_value))
            field_maximum_magnitude[field] = max(field_maximum_magnitude[field], magnitude)
            if difference > field_maximum_absolute_error[field]:
                field_maximum_absolute_error[field] = difference
                field_representative_cell[field] = cell_id
    field_normalized_errors: dict[str, float] = {}
    for field in STATE_FIELDS:
        floor = 1.0e-14 if field in {"vx", "vy", "vz", "momx", "momy", "momz"} else 1.0e-300
        field_normalized_errors[field] = field_maximum_absolute_error[field] / max(
            field_maximum_magnitude[field], floor
        )
    return {
        "cell_count": len(left),
        "field_maximum_absolute_errors": field_maximum_absolute_error,
        "field_maximum_magnitudes": field_maximum_magnitude,
        "field_normalized_errors": field_normalized_errors,
        "field_representative_cells": field_representative_cell,
    }


def positive_state_minima(
    state: dict[int, tuple[float, ...]],
) -> dict[str, float]:
    return {
        field: min(values[STATE_FIELDS.index(field)] for values in state.values())
        for field in POSITIVE_STATE_FIELDS
    }


def gate_comparison(
    verdict: Verdict,
    label: str,
    comparison: dict[str, object],
    tolerances: dict[str, float],
) -> None:
    errors = comparison["field_normalized_errors"]
    assert isinstance(errors, dict)
    for field, tolerance in tolerances.items():
        error = float(errors[field])
        verdict.require(
            error <= tolerance,
            f"{label}.{field}",
            f"normalized_error={error:.6e}, limit={tolerance:.6e}",
        )


def expected_histogram(scenario: str, side: int) -> dict[int, int]:
    cells = side ** 3
    if scenario == "parity":
        return {0: cells, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    central_fast_cells = 2 ** 3
    central_intermediate_cells = 8 ** 3 - central_fast_cells
    return {0: central_fast_cells, 1: 0,
            2: central_intermediate_cells, 3: 0,
            4: cells - 8 ** 3, 5: 0, 6: 0}


def expected_active_updates(mode: str, histogram: dict[int, int], final_tick: int) -> int:
    if mode in {"global", "individual_sync"}:
        return sum(histogram.values()) * final_tick
    return sum(
        count * math.ceil(final_tick / (2 ** bin_number))
        for bin_number, count in histogram.items()
    )


def expected_fmm_counts(
    mode: str, histogram: dict[int, int], final_tick: int
) -> tuple[int, int]:
    if mode == "global":
        return 2 * final_tick, 0

    bins_with_cached_acceleration: set[int] = set()
    calls = 0
    targets = 0
    for tick in range(1, final_tick + 1):
        active_bins = [
            bin_number
            for bin_number, count in histogram.items()
            if count > 0 and (tick % (2 ** bin_number) == 0 or tick == final_tick)
        ]
        active_cells = sum(histogram[bin_number] for bin_number in active_bins)
        if any(bin_number not in bins_with_cached_acceleration for bin_number in active_bins):
            calls += 1
            targets += active_cells
        calls += 1
        targets += active_cells
        bins_with_cached_acceleration.update(active_bins)
    return calls, targets


def mass_drift(metrics: dict[str, str]) -> float:
    initial = number(metrics, "initial_mass")
    final = number(metrics, "final_mass")
    return abs(final - initial) / max(abs(initial), 1.0e-300)


def dynamics(
    initial: dict[int, tuple[float, ...]],
    final: dict[int, tuple[float, ...]],
) -> dict[str, float]:
    if initial.keys() != final.keys():
        raise RuntimeError("initial/final cell-ID sets differ")
    maximum_speed = 0.0
    maximum_displacement = 0.0
    maximum_density_change = 0.0
    for cell_id in initial:
        before = initial[cell_id]
        after = final[cell_id]
        maximum_speed = max(
            maximum_speed,
            math.sqrt(after[6] ** 2 + after[7] ** 2 + after[8] ** 2),
        )
        maximum_displacement = max(
            maximum_displacement,
            math.sqrt(sum((after[index] - before[index]) ** 2 for index in range(3))),
        )
        maximum_density_change = max(maximum_density_change, abs(after[3] - before[3]))
    return {
        "maximum_speed": maximum_speed,
        "maximum_displacement": maximum_displacement,
        "maximum_density_change": maximum_density_change,
    }


def validate_lane(
    verdict: Verdict,
    label: str,
    directory: Path,
    scenario: str,
    mode: str,
    side: int,
    final_tick: int,
) -> tuple[dict[str, str], dict[int, tuple[float, ...]], dict[int, tuple[float, ...]]]:
    exit_path = directory / "exit_code.txt"
    exit_code = exit_path.read_text(encoding="utf-8").strip() if exit_path.is_file() else "missing"
    verdict.require(exit_code == "0", f"{label}.exit", f"exit_code={exit_code}")
    for evidence_name in ("started_utc.txt", "finished_utc.txt",
                          "frozen_rehash.txt", "launcher_identity.txt",
                          "launcher_cleanup.log", "launcher_cleanup_exit_code.txt",
                          "stderr.log"):
        evidence_path = directory / evidence_name
        verdict.require(evidence_path.is_file() and evidence_path.stat().st_size > 0,
                        f"{label}.evidence.{evidence_name}", str(evidence_path))
    launcher_identity = (directory / "launcher_identity.txt").read_text(encoding="utf-8")
    verdict.require("binding=core" in launcher_identity and "mapping=core" in launcher_identity,
                    f"{label}.launcher_binding", launcher_identity.replace("\n", ", ").strip())
    cleanup_exit = (directory / "launcher_cleanup_exit_code.txt").read_text(
        encoding="utf-8").strip()
    verdict.require(cleanup_exit == "0", f"{label}.launcher_cleanup",
                    f"exit_code={cleanup_exit}")
    rehash = (directory / "frozen_rehash.txt").read_text(encoding="utf-8")
    verdict.require("live_runner_sha256=" in rehash and "FAILED" not in rehash,
                    f"{label}.frozen_rehash", "live runner recorded and all hashes passed")
    stderr = (directory / "stderr.log").read_text(encoding="utf-8", errors="replace")
    binding_matches = list(re.finditer(
        r"\[([^:\]]+):[0-9]+\]\s+MCW rank\s+(\d+)\s+bound to\s+([^\n]+)",
        stderr,
    ))
    binding_ranks = {int(match.group(2)) for match in binding_matches}
    binding_hosts = {match.group(1) for match in binding_matches}
    binding_descriptions = {match.group(3) for match in binding_matches}
    verdict.require(binding_ranks == set(range(EXPECTED_RANKS)),
                    f"{label}.mpi_core_binding",
                    f"reported_ranks={sorted(binding_ranks)}, expected=0..{EXPECTED_RANKS - 1}")
    verdict.require(len(binding_hosts) == 1,
                    f"{label}.mpi_single_host", f"hosts={sorted(binding_hosts)}")
    verdict.require(len(binding_descriptions) == EXPECTED_RANKS,
                    f"{label}.mpi_distinct_core_bindings",
                    f"distinct={len(binding_descriptions)}, expected={EXPECTED_RANKS}")
    metrics = read_metrics(directory / "metrics.txt")
    verdict.require(metrics.get("scenario") == scenario, f"{label}.scenario", str(metrics.get("scenario")))
    verdict.require(metrics.get("mode") == mode, f"{label}.mode", str(metrics.get("mode")))
    verdict.require(integer(metrics, "world_size") == EXPECTED_RANKS, f"{label}.ranks", metrics["world_size"])
    verdict.require(integer(metrics, "grid_side") == side, f"{label}.side", metrics["grid_side"])
    verdict.require(integer(metrics, "global_cells") == side ** 3, f"{label}.cells", metrics["global_cells"])
    verdict.require(integer(metrics, "final_tick") == final_tick, f"{label}.final_tick", metrics["final_tick"])
    expected_quantum = 1.0e-4 if scenario == "parity" else 2.0e-4
    verdict.require(
        abs(number(metrics, "time_quantum") - expected_quantum) <= 1.0e-18,
        f"{label}.quantum",
        metrics["time_quantum"],
    )
    expected_gravity = 0.1 if scenario == "parity" else 1.0e-6
    verdict.require(abs(number(metrics, "gravity_constant") - expected_gravity) <=
                    1.0e-15 * expected_gravity,
                    f"{label}.gravity_constant", metrics["gravity_constant"])
    expected_profile = "warm_nonuniform" if scenario == "parity" else "cold_uniform"
    expected_layout = "perturbed_cartesian" if scenario == "parity" else "cartesian"
    expected_motion = "round_cells" if scenario == "parity" else "lagrangian"
    expected_timestep_layout = (
        "uniform_bin_0" if scenario == "parity" else "localized_nested_cubes"
    )
    verdict.require(metrics.get("initial_state_profile") == expected_profile,
                    f"{label}.initial_state_profile", str(metrics.get("initial_state_profile")))
    verdict.require(metrics.get("point_layout") == expected_layout,
                    f"{label}.point_layout", str(metrics.get("point_layout")))
    verdict.require(metrics.get("point_motion") == expected_motion,
                    f"{label}.point_motion", str(metrics.get("point_motion")))
    verdict.require(metrics.get("timestep_layout") == expected_timestep_layout,
                    f"{label}.timestep_layout",
                    str(metrics.get("timestep_layout")))
    expected_maximum_bin = 0 if scenario == "parity" else 4
    expected_maximum_step = expected_quantum * (2 ** expected_maximum_bin)
    verdict.require(integer(metrics, "prescribed_maximum_bin") == expected_maximum_bin,
                    f"{label}.prescribed_maximum_bin",
                    metrics["prescribed_maximum_bin"])
    verdict.require(abs(number(metrics, "prescribed_maximum_time_step") -
                        expected_maximum_step) <= 1.0e-18,
                    f"{label}.prescribed_maximum_time_step",
                    metrics["prescribed_maximum_time_step"])
    if scenario == "performance":
        initial_density = number(metrics, "performance_initial_density")
        initial_pressure = number(metrics, "performance_initial_pressure")
        verdict.require(abs(initial_density - 1.0) <= 1.0e-15,
                        f"{label}.performance_initial_density", str(initial_density))
        verdict.require(abs(initial_pressure - 1.0e-8) <= 1.0e-23,
                        f"{label}.performance_initial_pressure", str(initial_pressure))
        sound_speed = math.sqrt((5.0 / 3.0) * initial_pressure / initial_density)
        acoustic_crossing_time = (1.0 / side) / sound_speed
        verdict.require(expected_maximum_step < acoustic_crossing_time,
                        f"{label}.acoustic_step_bound",
                        f"dt_max={expected_maximum_step:.6e}, crossing={acoustic_crossing_time:.6e}")
    verdict.require(integer(metrics, "events") == final_tick, f"{label}.events", metrics["events"])
    verdict.require(integer(metrics, "final_synchronized") == 1, f"{label}.synchronized", metrics["final_synchronized"])
    histogram = expected_histogram(scenario, side)
    for bin_number, expected_count in histogram.items():
        actual = integer(metrics, f"initial_bin_{bin_number}")
        verdict.require(actual == expected_count, f"{label}.bin_{bin_number}", f"actual={actual}, expected={expected_count}")
    expected_updates = expected_active_updates(mode, histogram, final_tick)
    verdict.require(
        integer(metrics, "active_updates") == expected_updates,
        f"{label}.active_updates",
        f"actual={integer(metrics, 'active_updates')}, expected={expected_updates}",
    )
    minimum_owned = integer(metrics, "minimum_owned_cells")
    maximum_owned = integer(metrics, "maximum_owned_cells")
    zero_owned = integer(metrics, "zero_owned_ranks")
    verdict.require(minimum_owned <= side ** 3 / EXPECTED_RANKS <= maximum_owned,
                    f"{label}.ownership_bounds", f"min={minimum_owned}, max={maximum_owned}")
    if label.startswith("empty_"):
        verdict.require(minimum_owned == 0 and zero_owned > 0 and maximum_owned > 0,
                        f"{label}.empty_ranks", f"min={minimum_owned}, max={maximum_owned}, zero={zero_owned}")
    drift = mass_drift(metrics)
    verdict.require(drift <= MASS_DRIFT_TOLERANCE, f"{label}.mass_conservation",
                    f"relative_drift={drift:.6e}, limit={MASS_DRIFT_TOLERANCE:.6e}")
    wall = number(metrics, "wall_seconds")
    fmm_seconds = number(metrics, "fmm_seconds")
    verdict.require(wall > 0 and 0 < fmm_seconds <= 1.01 * wall,
                    f"{label}.timing", f"wall={wall:.6e}, fmm={fmm_seconds:.6e}")
    calls = integer(metrics, "fmm_full_calls") if mode == "global" else integer(metrics, "fmm_target_calls")
    wrong_calls = integer(metrics, "fmm_target_calls") if mode == "global" else integer(metrics, "fmm_full_calls")
    expected_calls, expected_targets = expected_fmm_counts(mode, histogram, final_tick)
    verdict.require(calls == expected_calls and wrong_calls == 0, f"{label}.fmm_route",
                    f"selected_calls={calls}, expected_calls={expected_calls}, wrong_calls={wrong_calls}")
    expected_sources = integer(metrics, "global_cells") * calls
    verdict.require(integer(metrics, "fmm_source_count") == expected_sources,
                    f"{label}.fmm_source_work",
                    f"actual={integer(metrics, 'fmm_source_count')}, expected={expected_sources}")
    verdict.require(integer(metrics, "fmm_target_count") == expected_targets,
                    f"{label}.fmm_target_work",
                    f"actual={integer(metrics, 'fmm_target_count')}, expected={expected_targets}")
    probe_error = number(metrics, "fmm_probe_max_normalized_error")
    verdict.require(integer(metrics, "fmm_probe_targets") == side ** 3,
                    f"{label}.fmm_probe_targets", metrics["fmm_probe_targets"])
    verdict.require(probe_error <= FMM_PROBE_TOLERANCE,
                    f"{label}.fmm_probe_parity",
                    f"normalized_error={probe_error:.6e}, limit={FMM_PROBE_TOLERANCE:.6e}")
    initial, initial_files = load_state(directory, "initial_state")
    final, final_files = load_state(directory, "final_state")
    verdict.require(initial_files == EXPECTED_RANKS and final_files == EXPECTED_RANKS,
                    f"{label}.state_files", f"initial={initial_files}, final={final_files}")
    verdict.require(len(initial) == side ** 3 and len(final) == side ** 3,
                    f"{label}.state_count", f"initial={len(initial)}, final={len(final)}")
    for state_name, state in (("initial", initial), ("final", final)):
        minima = positive_state_minima(state)
        for field, minimum in minima.items():
            verdict.require(minimum > 0, f"{label}.{state_name}_positive.{field}",
                            f"minimum={minimum:.17e}")
    return metrics, initial, final


def validate_primitive_recovery_probe(
    verdict: Verdict, directory: Path
) -> dict[str, str]:
    label = "primitive_recovery_probe"
    exit_path = directory / "exit_code.txt"
    exit_code = exit_path.read_text(encoding="utf-8").strip() \
        if exit_path.is_file() else "missing"
    verdict.require(exit_code == "0", f"{label}.exit", f"exit_code={exit_code}")
    for evidence_name in ("started_utc.txt", "finished_utc.txt",
                          "frozen_rehash.txt", "launcher_identity.txt",
                          "launcher_cleanup.log", "launcher_cleanup_exit_code.txt",
                          "stderr.log"):
        evidence_path = directory / evidence_name
        verdict.require(evidence_path.is_file() and evidence_path.stat().st_size > 0,
                        f"{label}.evidence.{evidence_name}", str(evidence_path))
    cleanup_exit = (directory / "launcher_cleanup_exit_code.txt").read_text(
        encoding="utf-8").strip()
    verdict.require(cleanup_exit == "0", f"{label}.launcher_cleanup",
                    f"exit_code={cleanup_exit}")
    launcher_identity = (directory / "launcher_identity.txt").read_text(
        encoding="utf-8")
    verdict.require("binding=core" in launcher_identity and
                    "mapping=core" in launcher_identity and
                    "rank_count=1" in launcher_identity,
                    f"{label}.launcher", launcher_identity.replace("\n", ", ").strip())
    rehash = (directory / "frozen_rehash.txt").read_text(encoding="utf-8")
    verdict.require("live_runner_sha256=" in rehash and "FAILED" not in rehash,
                    f"{label}.frozen_rehash",
                    "live runner recorded and all hashes passed")
    stderr = (directory / "stderr.log").read_text(
        encoding="utf-8", errors="replace")
    binding_matches = list(re.finditer(
        r"\[([^:\]]+):[0-9]+\]\s+MCW rank\s+(\d+)\s+bound to\s+([^\n]+)",
        stderr,
    ))
    verdict.require({int(match.group(2)) for match in binding_matches} == {0},
                    f"{label}.mpi_core_binding",
                    f"reported={[int(match.group(2)) for match in binding_matches]}")
    receipt = read_metrics(directory / "primitive_recovery_receipt.txt")
    verdict.require(integer(receipt, "active_targets") == 8,
                    f"{label}.active_targets", receipt["active_targets"])
    verdict.require(integer(receipt, "target_calls") == 2,
                    f"{label}.target_calls", receipt["target_calls"])
    first_tracer = number(receipt, "first_target_tracer")
    second_tracer = number(receipt, "second_target_tracer")
    verdict.require(abs(first_tracer) <= 1.0e-15,
                    f"{label}.first_target_tracer", str(first_tracer))
    verdict.require(abs(second_tracer - 0.75) <= 1.0e-15,
                    f"{label}.second_target_tracer", str(second_tracer))
    verdict.require(integer(receipt, "pass") == 1,
                    f"{label}.pass", receipt["pass"])
    return receipt


def coefficient_of_variation(values: list[float]) -> float:
    return statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0.0


def write_outputs(root: Path, verdict: Verdict, analysis: dict[str, object]) -> None:
    thresholds = {
        "initial_state_tolerance": INITIAL_STATE_TOLERANCE,
        "parity_field_tolerances": PARITY_FIELD_TOLERANCES,
        "performance_field_tolerances": PERFORMANCE_FIELD_TOLERANCES,
        "mass_drift_tolerance": MASS_DRIFT_TOLERANCE,
        "fmm_probe_tolerance": FMM_PROBE_TOLERANCE,
        "minimum_dynamic_speed": MINIMUM_DYNAMIC_SPEED,
        "minimum_dynamic_displacement": MINIMUM_DYNAMIC_DISPLACEMENT,
        "minimum_active_work_reduction": MINIMUM_ACTIVE_WORK_REDUCTION,
        "minimum_median_speedup": MINIMUM_MEDIAN_SPEEDUP,
        "minimum_paired_speedup": MINIMUM_PAIRED_SPEEDUP,
        "maximum_timing_cv": MAXIMUM_TIMING_CV,
    }
    payload = {"pass": verdict.passed, "thresholds": thresholds,
               "analysis": analysis, "checks": verdict.checks}
    root.mkdir(parents=True, exist_ok=True)
    (root / "analysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["PASS" if verdict.passed else "FAIL"]
    lines.extend(
        f"{'PASS' if check['pass'] else 'FAIL'}: {check['name']}: {check['detail']}"
        for check in verdict.checks
    )
    (root / "verdict.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    performance = analysis.get("performance", {})
    parity = analysis.get("parity", {})
    fmm_probe = analysis.get("fmm_probe", {})
    assert isinstance(performance, dict) and isinstance(parity, dict) and isinstance(fmm_probe, dict)
    markdown = [
        "# FMM individual-timestep benchmark analysis", "",
        f"Overall verdict: **{'PASS' if verdict.passed else 'FAIL'}**", "",
        "| Quantity | Result | Gate |", "|---|---:|---:|",
        f"| Parity worst tolerance ratio | {parity.get('worst_tolerance_ratio', float('nan')):.6e} | <= 1 |",
        f"| Direct full/target FMM error | {fmm_probe.get('maximum_normalized_error', float('nan')):.6e} | <= {FMM_PROBE_TOLERANCE:.1e} |",
        f"| Sparse worst tolerance ratio | {performance.get('worst_tolerance_ratio', float('nan')):.6e} | <= 1 |",
        f"| Active-work reduction | {performance.get('active_work_reduction', float('nan')):.3f}x | >= {MINIMUM_ACTIVE_WORK_REDUCTION:.3f}x |",
        f"| Median paired wall speedup | {performance.get('median_paired_speedup', float('nan')):.3f}x | >= {MINIMUM_MEDIAN_SPEEDUP:.3f}x |",
        f"| Slowest paired wall speedup | {performance.get('minimum_paired_speedup', float('nan')):.3f}x | >= {MINIMUM_PAIRED_SPEEDUP:.3f}x |",
        f"| Global FMM wall fraction | {performance.get('global_fmm_wall_fraction', float('nan')):.3f} | measured |",
        f"| Individual FMM wall fraction | {performance.get('individual_fmm_wall_fraction', float('nan')):.3f} | measured |",
        "",
        "This is a synthetic prescribed-local-timestep scheduler benchmark. Overall speedup can come from fewer full-source FMM calls through the kick cache and from partial hydro/mesh work. The current target adapter still solves all sources within each FMM evaluation.",
    ]
    (root / "ANALYSIS.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def worst_tolerance_ratio(comparison: dict[str, object], tolerances: dict[str, float]) -> float:
    errors = comparison["field_normalized_errors"]
    assert isinstance(errors, dict)
    return max(float(errors[field]) / tolerance for field, tolerance in tolerances.items())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    verdict = Verdict()
    analysis: dict[str, object] = {}
    metrics: dict[str, dict[str, str]] = {}
    initial_states: dict[str, dict[int, tuple[float, ...]]] = {}
    final_states: dict[str, dict[int, tuple[float, ...]]] = {}

    try:
        analysis["provenance"] = validate_campaign_provenance(verdict, root)
    except Exception as error:
        verdict.require(False, "campaign.provenance_exception", str(error))

    try:
        analysis["primitive_recovery_probe"] = validate_primitive_recovery_probe(
            verdict, root / "primitive_recovery_probe")
    except Exception as error:
        verdict.require(False, "primitive_recovery_probe.exception", str(error))

    for label, (scenario, mode, side, final_tick) in LANES.items():
        try:
            metrics[label], initial_states[label], final_states[label] = validate_lane(
                verdict, label, root / label, scenario, mode, side, final_tick
            )
        except Exception as error:
            verdict.require(False, f"{label}.exception", str(error))

    try:
        analysis["positive_state_minima"] = {
            label: {
                "initial": positive_state_minima(initial_states[label]),
                "final": positive_state_minima(final_states[label]),
            }
            for label in LANES
        }
    except Exception as error:
        verdict.require(False, "positive_state_minima.analysis_exception", str(error))

    try:
        probe_errors = {
            label: number(metrics[label], "fmm_probe_max_normalized_error")
            for label in LANES
        }
        analysis["fmm_probe"] = {
            "lane_normalized_errors": probe_errors,
            "maximum_normalized_error": max(probe_errors.values()),
        }
    except Exception as error:
        verdict.require(False, "fmm_probe.analysis_exception", str(error))

    try:
        empty_initial = compare_states(initial_states["empty_global"], initial_states["empty_individual"])
        gate_comparison(verdict, "empty.initial", empty_initial,
                        {field: INITIAL_STATE_TOLERANCE for field in STATE_FIELDS})
        empty_final = compare_states(final_states["empty_global"], final_states["empty_individual"])
        gate_comparison(verdict, "empty.endpoint", empty_final, PARITY_FIELD_TOLERANCES)
        analysis["empty_rank"] = {"initial": empty_initial, "final": empty_final}
    except Exception as error:
        verdict.require(False, "empty.comparison_exception", str(error))

    try:
        parity_initial = compare_states(initial_states["parity_global"], initial_states["parity_individual"])
        gate_comparison(verdict, "parity.initial", parity_initial,
                        {field: INITIAL_STATE_TOLERANCE for field in STATE_FIELDS})
        parity_final = compare_states(final_states["parity_global"], final_states["parity_individual"])
        gate_comparison(verdict, "parity.endpoint", parity_final, PARITY_FIELD_TOLERANCES)
        analysis["parity"] = {
            "initial": parity_initial,
            "final": parity_final,
            "worst_tolerance_ratio": worst_tolerance_ratio(parity_final, PARITY_FIELD_TOLERANCES),
        }
    except Exception as error:
        verdict.require(False, "parity.comparison_exception", str(error))

    try:
        performance_comparisons = []
        dynamic_results = []
        for sample in (1, 2, 3):
            global_label = f"perf_global_{sample}"
            individual_label = f"perf_individual_{sample}"
            initial_comparison = compare_states(initial_states[global_label], initial_states[individual_label])
            gate_comparison(verdict, f"performance.initial_{sample}", initial_comparison,
                            {field: INITIAL_STATE_TOLERANCE for field in STATE_FIELDS})
            final_comparison = compare_states(final_states[global_label], final_states[individual_label])
            gate_comparison(verdict, f"performance.endpoint_{sample}", final_comparison,
                            PERFORMANCE_FIELD_TOLERANCES)
            performance_comparisons.append(final_comparison)
            for label in (global_label, individual_label):
                response = dynamics(initial_states[label], final_states[label])
                dynamic_results.append({"lane": label, **response})
                verdict.require(response["maximum_speed"] >= MINIMUM_DYNAMIC_SPEED,
                                f"{label}.dynamic_speed",
                                f"max_speed={response['maximum_speed']:.6e}")
                verdict.require(response["maximum_displacement"] >= MINIMUM_DYNAMIC_DISPLACEMENT,
                                f"{label}.dynamic_displacement",
                                f"max_displacement={response['maximum_displacement']:.6e}")

        global_walls = [number(metrics[f"perf_global_{sample}"], "wall_seconds") for sample in (1, 2, 3)]
        individual_walls = [number(metrics[f"perf_individual_{sample}"], "wall_seconds") for sample in (1, 2, 3)]
        paired_speedups = [global_walls[index] / individual_walls[index] for index in range(3)]
        median_speedup = statistics.median(paired_speedups)
        minimum_speedup = min(paired_speedups)
        global_cv = coefficient_of_variation(global_walls)
        individual_cv = coefficient_of_variation(individual_walls)
        verdict.require(median_speedup >= MINIMUM_MEDIAN_SPEEDUP,
                        "performance.median_speedup",
                        f"median={median_speedup:.6f}x, required={MINIMUM_MEDIAN_SPEEDUP:.6f}x")
        verdict.require(minimum_speedup >= MINIMUM_PAIRED_SPEEDUP,
                        "performance.minimum_paired_speedup",
                        f"minimum={minimum_speedup:.6f}x, required={MINIMUM_PAIRED_SPEEDUP:.6f}x")
        verdict.require(global_cv <= MAXIMUM_TIMING_CV and individual_cv <= MAXIMUM_TIMING_CV,
                        "performance.dispersion",
                        f"global_cv={global_cv:.6f}, individual_cv={individual_cv:.6f}")
        global_updates = statistics.median([number(metrics[f"perf_global_{sample}"], "active_updates") for sample in (1, 2, 3)])
        individual_updates = statistics.median([number(metrics[f"perf_individual_{sample}"], "active_updates") for sample in (1, 2, 3)])
        active_work_reduction = global_updates / individual_updates
        verdict.require(active_work_reduction >= MINIMUM_ACTIVE_WORK_REDUCTION,
                        "performance.active_work",
                        f"reduction={active_work_reduction:.6f}x")
        global_fmm = [number(metrics[f"perf_global_{sample}"], "fmm_seconds") for sample in (1, 2, 3)]
        individual_fmm = [number(metrics[f"perf_individual_{sample}"], "fmm_seconds") for sample in (1, 2, 3)]
        worst_ratio = max(
            worst_tolerance_ratio(comparison, PERFORMANCE_FIELD_TOLERANCES)
            for comparison in performance_comparisons
        )
        analysis["performance"] = {
            "comparisons": performance_comparisons,
            "dynamics": dynamic_results,
            "global_wall_seconds": global_walls,
            "individual_wall_seconds": individual_walls,
            "paired_speedups": paired_speedups,
            "median_paired_speedup": median_speedup,
            "minimum_paired_speedup": minimum_speedup,
            "global_timing_cv": global_cv,
            "individual_timing_cv": individual_cv,
            "active_work_reduction": active_work_reduction,
            "worst_tolerance_ratio": worst_ratio,
            "global_fmm_seconds": global_fmm,
            "individual_fmm_seconds": individual_fmm,
            "global_fmm_wall_fraction": statistics.median(
                [global_fmm[index] / global_walls[index] for index in range(3)]
            ),
            "individual_fmm_wall_fraction": statistics.median(
                [individual_fmm[index] / individual_walls[index] for index in range(3)]
            ),
            "global_fmm_source_counts": [integer(metrics[f"perf_global_{sample}"], "fmm_source_count") for sample in (1, 2, 3)],
            "individual_fmm_source_counts": [integer(metrics[f"perf_individual_{sample}"], "fmm_source_count") for sample in (1, 2, 3)],
        }
    except Exception as error:
        verdict.require(False, "performance.analysis_exception", str(error))

    write_outputs(root, verdict, analysis)
    print("PASS" if verdict.passed else "FAIL")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
