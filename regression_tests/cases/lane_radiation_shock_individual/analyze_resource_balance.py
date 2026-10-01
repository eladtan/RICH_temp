#!/usr/bin/env python3
"""Gate rank-RSS equality and whole-run memory/runtime without cycle pairing."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path


STEP_PREFIX = "RICH_STEP "
TIME_PREFIX = "  time   | "
WORK_PREFIX = "  work   | "
PHASES_PREFIX = "  phases | "
MESH_PREFIX = "  mesh   | "
SOURCE_PREFIX = "  source | "
AMR_PREFIX = "RICH_AMR "
DETAIL_PREFIX = "RICH_STEP_DETAIL "
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FIELD_RE = re.compile(r"([A-Za-z0-9_-]+)=(\[[^]]*\]|[^ ]+)")
DEFAULT_WORK_RELATIVE_TOLERANCE = 1.0e-4
EXPECTED_PERF_UNITS = {
    "event-wall": "seconds",
    "current-rss": "KiB",
    "peak-rss": "KiB",
}
REQUIRED_STATISTICS = ("min", "median", "mean", "p95", "max")
STEP_FIELDS = (("mode", "cycle"),)
TIME_FIELDS = ((
    "t_start", "t_end", "event_dt", "applied_dt_min", "applied_dt_max",
    "next_event_dt",
),)
WORK_FIELDS = (
    ("active_cells", "total_cells", "active_bins"),
    ("active_cells", "active_bins"),
)
PHASE_FIELDS = ((
    "step_s", "hydro_s", "gravity_s", "radiation_s", "amr_s",
),)
MESH_FIELDS = (("mesh_s", "mesh_builds"),)
SOURCE_FIELDS = (
    ("source_s", "source_pct", "source_calls"),
    (
        "source_s", "source_first_s", "source_second_s", "source_pct",
        "source_calls",
    ),
)
AMR_FIELDS = ((
    "mode", "cycle", "time", "cells_before", "added_cells",
    "removed_cells", "cells_after",
),)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sequence")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("missing values for distribution")
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def fields_after(line: str, marker: str) -> dict[str, str] | None:
    offset = line.find(marker)
    if offset < 0:
        return None
    return dict(FIELD_RE.findall(line[offset + len(marker) :]))


def ordered_fields(
    path: Path,
    line: str,
    prefix: str,
    separator: str,
    expected_orders: tuple[tuple[str, ...], ...],
    subject: str,
) -> dict[str, str]:
    payload = line[len(prefix) :].rstrip()
    pieces = payload.split(separator)
    matches = [FIELD_RE.fullmatch(piece) for piece in pieces]
    if any(match is None for match in matches):
        raise ValueError(f"malformed {subject} line in {path}: {line.rstrip()}")
    items = [match.groups() for match in matches if match is not None]
    if tuple(name for name, _ in items) not in expected_orders:
        raise ValueError(f"malformed {subject} line in {path}: {line.rstrip()}")
    return dict(items)


def valid_active_bins(value: str, mode: str, active_cells: int) -> bool:
    if mode == "global":
        return value == "global"
    if len(value) < 2 or value[0] != "[" or value[-1] != "]":
        return False
    entries = value[1:-1].split("; ")
    if not entries or any(not entry for entry in entries):
        return False
    previous_bin = -1
    count_sum = 0
    for entry in entries:
        pieces = entry.split(",")
        matches = [FIELD_RE.fullmatch(piece) for piece in pieces]
        if any(match is None for match in matches):
            return False
        items = [match.groups() for match in matches if match is not None]
        if tuple(name for name, _ in items) != ("bin", "count", "dt"):
            return False
        fields = dict(items)
        bin_number = parse_int(fields, "bin")
        count = parse_int(fields, "count")
        timestep = parse_float(fields, "dt")
        if (
            bin_number is None
            or not previous_bin < bin_number < 63
            or count is None
            or count <= 0
            or timestep is None
            or timestep <= 0
        ):
            return False
        count_sum += count
        previous_bin = bin_number
    return count_sum == active_cells


def require_finite_fields(
    path: Path,
    line: str,
    fields: dict[str, str],
    names: tuple[str, ...],
    subject: str,
) -> dict[str, float]:
    values = {name: parse_float(fields, name) for name in names}
    if any(value is None for value in values.values()):
        raise ValueError(f"malformed {subject} line in {path}: {line.rstrip()}")
    return {name: value for name, value in values.items() if value is not None}


def parse_float(fields: dict[str, str], name: str) -> float | None:
    try:
        value = float(fields[name])
    except (KeyError, ValueError):
        return None
    return value if math.isfinite(value) else None


def parse_int(fields: dict[str, str], name: str) -> int | None:
    try:
        return int(fields[name])
    except (KeyError, ValueError):
        return None


def validate_performance_statistics(
    path: Path,
    line: str,
    phase: str,
    unit: str | None,
    statistics_by_name: dict[str, float],
) -> None:
    expected_unit = EXPECTED_PERF_UNITS.get(phase)
    if expected_unit is None:
        return
    if unit != expected_unit:
        raise ValueError(
            f"expected {phase} unit={expected_unit} in {path}: {line.rstrip()}"
        )
    missing = [name for name in REQUIRED_STATISTICS if name not in statistics_by_name]
    if missing:
        raise ValueError(
            f"missing {phase} statistics {missing} in {path}: {line.rstrip()}"
        )
    values = [statistics_by_name[name] for name in REQUIRED_STATISTICS]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite {phase} statistics in {path}: {line.rstrip()}")
    minimum, median, mean, p95, maximum = values
    if minimum < 0.0 or median <= 0.0 or mean <= 0.0 or p95 <= 0.0 or maximum <= 0.0:
        raise ValueError(f"non-positive {phase} statistics in {path}: {line.rstrip()}")
    if not minimum <= median <= p95 <= maximum or not minimum <= mean <= maximum:
        raise ValueError(f"unordered {phase} statistics in {path}: {line.rstrip()}")


def parse_log(path: Path) -> dict[str, object]:
    records: dict[int, dict[str, dict[str, float]]] = {}
    events: dict[int, dict[str, float | int]] = {}
    amr_events: list[dict[str, float | int | str]] = []
    endpoint: dict[str, object] = {"complete": False, "source": None}
    benchmark: dict[str, str] = {}
    runtime_cycle: int | None = None
    runtime_mode: str | None = None
    runtime_subject = 0
    seen_event_cycles: set[int] = set()
    seen_amr_cycles: set[int] = set()
    amr_eligible_cycle: int | None = None
    amr_terminator_cycle: int | None = None
    pending_event: dict[str, float | int] = {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_ESCAPE_RE.sub("", raw_line)

            if amr_terminator_cycle is not None:
                if line.rstrip("\r\n") != "":
                    raise ValueError(
                        f"missing blank line after RICH_AMR cycle="
                        f"{amr_terminator_cycle} in {path}"
                    )
                amr_terminator_cycle = None
                continue

            if (
                amr_eligible_cycle is not None
                and not line.startswith(AMR_PREFIX)
            ):
                amr_eligible_cycle = None

            if runtime_cycle is not None:
                expected_prefixes = (
                    TIME_PREFIX, WORK_PREFIX, PHASES_PREFIX, MESH_PREFIX,
                    SOURCE_PREFIX,
                )
                if runtime_subject == len(expected_prefixes):
                    if line.rstrip("\r\n") != "":
                        raise ValueError(
                            f"missing blank line after RICH_STEP cycle="
                            f"{runtime_cycle} in {path}"
                        )
                    amr_eligible_cycle = (
                        runtime_cycle if runtime_mode == "individual" else None
                    )
                    runtime_cycle = None
                    runtime_mode = None
                    runtime_subject = 0
                    continue
                if (
                    not line.startswith(expected_prefixes[runtime_subject])
                    and not line.startswith(expected_prefixes)
                ):
                    raise ValueError(
                        f"interleaved RICH_STEP block cycle={runtime_cycle} "
                        f"in {path}: {line.rstrip()}"
                    )

            if line.startswith(STEP_PREFIX):
                if runtime_cycle is not None:
                    raise ValueError(
                        f"incomplete RICH_STEP block cycle={runtime_cycle} in {path}"
                    )
                fields = ordered_fields(
                    path, line, STEP_PREFIX, " ", STEP_FIELDS,
                    "RICH_STEP header",
                )
                mode = fields.get("mode")
                cycle = parse_int(fields, "cycle")
                if (
                    mode not in {"global", "individual"}
                    or cycle is None
                    or cycle < 0
                ):
                    raise ValueError(
                        f"malformed RICH_STEP header in {path}: {line.rstrip()}"
                    )
                runtime_mode = mode
                runtime_cycle = cycle
                runtime_subject = 0
                if mode == "individual":
                    if cycle in seen_event_cycles:
                        raise ValueError(
                            f"duplicate RICH_STEP cycle={cycle} in {path}"
                        )
                    seen_event_cycles.add(cycle)
                    event = events.setdefault(cycle, {"cycle": cycle})
                    event.update(pending_event)
                    pending_event.clear()

            if line.startswith(TIME_PREFIX):
                if runtime_cycle is None or runtime_subject != 0:
                    raise ValueError(
                        f"unexpected time line in {path}: {line.rstrip()}"
                    )
                fields = ordered_fields(
                    path, line, TIME_PREFIX, " | ", TIME_FIELDS, "time",
                )
                time_values = require_finite_fields(
                    path, line, fields, TIME_FIELDS[0], "time",
                )
                if any(
                    time_values[name] < 0
                    for name in (
                        "event_dt", "applied_dt_min", "applied_dt_max",
                        "next_event_dt",
                    )
                ):
                    raise ValueError(
                        f"malformed time line in {path}: {line.rstrip()}"
                    )
                if runtime_mode == "individual":
                    events.setdefault(
                        runtime_cycle, {"cycle": runtime_cycle}
                    )["event_time"] = time_values["t_end"]
                runtime_subject = 1

            if line.startswith(WORK_PREFIX):
                if runtime_cycle is None or runtime_subject != 1:
                    raise ValueError(
                        f"unexpected work line in {path}: {line.rstrip()}"
                    )
                fields = ordered_fields(
                    path, line, WORK_PREFIX, " | ", WORK_FIELDS, "work",
                )
                active_cells = parse_int(fields, "active_cells")
                total_cells = (
                    parse_int(fields, "total_cells")
                    if "total_cells" in fields
                    else None
                )
                if (
                    active_cells is None
                    or active_cells < 0
                    or (
                        total_cells is not None
                        and (total_cells < 0 or total_cells < active_cells)
                    )
                    or runtime_mode is None
                    or not valid_active_bins(
                        fields["active_bins"], runtime_mode, active_cells
                    )
                ):
                    raise ValueError(
                        f"malformed work line in {path}: {line.rstrip()}"
                    )
                if runtime_mode == "individual":
                    event = events.setdefault(
                        runtime_cycle, {"cycle": runtime_cycle}
                    )
                    event["active_cells"] = active_cells
                    if total_cells is not None:
                        event["total_cells"] = total_cells
                        event.setdefault("amr", 0)
                runtime_subject = 2

            if line.startswith(PHASES_PREFIX):
                if runtime_cycle is None or runtime_subject != 2:
                    raise ValueError(
                        f"unexpected phases line in {path}: {line.rstrip()}"
                    )
                fields = ordered_fields(
                    path, line, PHASES_PREFIX, " | ", PHASE_FIELDS, "phases",
                )
                phase_values = require_finite_fields(
                    path, line, fields, PHASE_FIELDS[0], "phases",
                )
                if any(value < 0 for value in phase_values.values()):
                    raise ValueError(
                        f"malformed phases line in {path}: {line.rstrip()}"
                    )
                runtime_subject = 3

            if line.startswith(MESH_PREFIX):
                if runtime_cycle is None or runtime_subject != 3:
                    raise ValueError(
                        f"unexpected mesh line in {path}: {line.rstrip()}"
                    )
                fields = ordered_fields(
                    path, line, MESH_PREFIX, " | ", MESH_FIELDS, "mesh",
                )
                mesh_s = require_finite_fields(
                    path, line, fields, ("mesh_s",), "mesh",
                )["mesh_s"]
                mesh_builds = parse_int(fields, "mesh_builds")
                if mesh_s < 0 or mesh_builds is None or mesh_builds < 0:
                    raise ValueError(
                        f"malformed mesh line in {path}: {line.rstrip()}"
                    )
                if runtime_mode == "individual":
                    event = events.setdefault(
                        runtime_cycle, {"cycle": runtime_cycle}
                    )
                    event["mesh_s"] = mesh_s
                    event["mesh_builds"] = mesh_builds
                runtime_subject = 4

            if line.startswith(SOURCE_PREFIX):
                if runtime_cycle is None or runtime_subject not in {3, 4}:
                    raise ValueError(
                        f"unexpected source line in {path}: {line.rstrip()}"
                    )
                fields = ordered_fields(
                    path, line, SOURCE_PREFIX, " | ", SOURCE_FIELDS, "source",
                )
                source_names = tuple(fields)
                source_values = require_finite_fields(
                    path, line, fields, source_names[:-1], "source",
                )
                source_calls = parse_int(fields, "source_calls")
                if (
                    any(value < 0 for value in source_values.values())
                    or source_calls is None
                    or source_calls < 0
                ):
                    raise ValueError(
                        f"malformed source line in {path}: {line.rstrip()}"
                    )
                runtime_subject = 5

            if line.startswith(AMR_PREFIX):
                fields = ordered_fields(
                    path, line, AMR_PREFIX, " ", AMR_FIELDS, "RICH_AMR",
                )
                mode = fields.get("mode")
                cycle = parse_int(fields, "cycle")
                event_time = parse_float(fields, "time")
                cells_before = parse_int(fields, "cells_before")
                added_cells = parse_int(fields, "added_cells")
                removed_cells = parse_int(fields, "removed_cells")
                cells_after = parse_int(fields, "cells_after")
                counts = (
                    cells_before, added_cells, removed_cells, cells_after,
                )
                if (
                    mode not in {"global", "individual"}
                    or cycle is None
                    or cycle < 0
                    or event_time is None
                    or any(value is None or value < 0 for value in counts)
                ):
                    raise ValueError(
                        f"malformed RICH_AMR line in {path}: {line.rstrip()}"
                    )
                if (
                    cells_before + added_cells < removed_cells
                    or cells_after != cells_before + added_cells - removed_cells
                ):
                    raise ValueError(
                        f"malformed RICH_AMR line in {path}: {line.rstrip()}"
                    )
                if mode == "individual":
                    if amr_eligible_cycle is None:
                        if cycle in seen_amr_cycles:
                            raise ValueError(
                                f"duplicate RICH_AMR cycle={cycle} in {path}"
                            )
                        raise ValueError(
                            f"orphan RICH_AMR line in {path}: {line.rstrip()}"
                        )
                    if cycle != amr_eligible_cycle:
                        raise ValueError(
                            f"malformed RICH_AMR line in {path}: {line.rstrip()}"
                        )
                    event = events[cycle]
                    if (
                        event.get("event_time") != event_time
                        or (
                            "total_cells" in event
                            and event["total_cells"] != cells_before
                        )
                    ):
                        raise ValueError(
                            f"RICH_AMR does not match RICH_STEP cycle={cycle} "
                            f"in {path}: {line.rstrip()}"
                        )
                    event["amr"] = 1
                    event["amr_added_cells"] = added_cells
                    event["amr_removed_cells"] = removed_cells
                    event["amr_cells_after"] = cells_after
                    seen_amr_cycles.add(cycle)
                amr_events.append({
                    "mode": mode,
                    "cycle": cycle,
                    "time": event_time,
                    "cells_before": cells_before,
                    "added_cells": added_cells,
                    "removed_cells": removed_cells,
                    "cells_after": cells_after,
                })
                amr_eligible_cycle = None
                amr_terminator_cycle = cycle
                continue

            fields = fields_after(line, DETAIL_PREFIX)
            if fields is not None:
                try:
                    cycle = int(fields.pop("cycle"))
                    phase = fields.pop("phase")
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        f"malformed RICH_STEP_DETAIL line in {path}: {line.rstrip()}"
                    ) from error
                fields.pop("mode", None)
                unit = fields.pop("unit", None)
                numeric: dict[str, float] = {}
                for key, value in fields.items():
                    try:
                        numeric[key] = float(value)
                    except ValueError:
                        continue
                validate_performance_statistics(path, line, phase, unit, numeric)
                cycle_records = records.setdefault(cycle, {})
                if phase in cycle_records:
                    raise ValueError(
                        f"duplicate performance cycle={cycle} phase={phase} "
                        f"in {path}"
                    )
                cycle_records[phase] = numeric

            fields = fields_after(line, "MG_TIMESTEP_LIMIT mode=individual")
            if fields is not None:
                event_time = parse_float(fields, "event_time")
                active_cells = parse_int(fields, "active_cells")
                event = (
                    events.setdefault(runtime_cycle, {"cycle": runtime_cycle})
                    if runtime_mode == "individual" and
                    runtime_cycle is not None
                    else pending_event
                )
                if event_time is not None:
                    event["event_time"] = event_time
                if active_cells is not None:
                    event["active_cells"] = active_cells

            fields = fields_after(line, "INDIVIDUAL_HYDRO_PHASE_TIMING")
            if fields is not None:
                active_cells = parse_float(fields, "active_cells_global")
                canonical_cells = parse_float(fields, "canonical_cells_global")
                event = (
                    events.setdefault(runtime_cycle, {"cycle": runtime_cycle})
                    if runtime_mode == "individual" and
                    runtime_cycle is not None
                    else pending_event
                )
                if active_cells is not None and "active_cells" not in event:
                    event["active_cells"] = active_cells
                if canonical_cells is not None:
                    event["canonical_cells"] = canonical_cells

            fields = fields_after(line, "INDIVIDUAL_LOAD_BALANCE_DECISION")
            if fields is not None:
                cycle = parse_int(fields, "cycle")
                if cycle is not None:
                    event = events.setdefault(cycle, {"cycle": cycle})
                    amr = parse_int(fields, "amr")
                    ownership_epoch = parse_int(fields, "ownership_epoch")
                    if amr is not None:
                        event["amr"] = amr
                    if ownership_epoch is not None:
                        event["ownership_epoch"] = ownership_epoch

            fields = fields_after(line, "PROGRESS phase=final")
            if fields is not None:
                final_cycle = parse_int(fields, "cycle")
                final_time = parse_float(fields, "time")
                wall_seconds = parse_float(fields, "wall_seconds")
                endpoint.update(
                    {
                        "final_cycle": final_cycle,
                        "final_time": final_time,
                        "evolution_wall_seconds": wall_seconds,
                        "source": "PROGRESS phase=final",
                    }
                )

            fields = fields_after(line, "INDIVIDUAL_RESTART_FINGERPRINT phase=final")
            if fields is not None:
                final_tick = parse_int(fields, "current_tick")
                final_cells = parse_int(fields, "cells")
                if final_tick is not None:
                    endpoint["final_tick"] = final_tick
                if final_cells is not None:
                    endpoint["final_cells"] = final_cells

            fields = fields_after(line, "BENCHMARK_RESULT")
            if fields is not None:
                benchmark = fields

    if runtime_cycle is not None:
        if runtime_subject == 4:
            raise ValueError(
                f"missing blank line after RICH_STEP cycle={runtime_cycle} "
                f"in {path}"
            )
        raise ValueError(
            f"incomplete RICH_STEP block cycle={runtime_cycle} in {path}"
        )
    if amr_terminator_cycle is not None:
        raise ValueError(
            f"missing blank line after RICH_AMR cycle={amr_terminator_cycle} "
            f"in {path}"
        )
    if not records:
        raise ValueError(f"no RICH_STEP_DETAIL records in {path}")

    if endpoint.get("final_time") is None:
        endpoint["final_time"] = parse_float(benchmark, "final_time")
    if endpoint.get("evolution_wall_seconds") is None:
        endpoint["evolution_wall_seconds"] = parse_float(
            benchmark, "evolution_wall_max"
        )
    if endpoint.get("final_cells") is None:
        endpoint["final_cells"] = parse_int(benchmark, "cells")
    if endpoint.get("source") is None and benchmark:
        endpoint["source"] = "BENCHMARK_RESULT"
    evolution_wall_seconds = endpoint.get("evolution_wall_seconds")
    endpoint["complete"] = (
        endpoint.get("final_time") is not None
        and evolution_wall_seconds is not None
        and float(evolution_wall_seconds) > 0.0
    )
    return {
        "records": records,
        "events": events,
        "amr_events": amr_events,
        "endpoint": endpoint,
    }


def usable_cycles(
    records: dict[int, dict[str, dict[str, float]]], late_cycles: int
) -> list[int]:
    cycles = sorted(
        cycle
        for cycle, phases in records.items()
        if "current-rss" in phases and "event-wall" in phases
    )
    if not cycles:
        raise ValueError("no cycles contain both current-rss and event-wall records")
    return cycles[-late_cycles:] if late_cycles > 0 else cycles


def summarize(
    path: Path,
    records: dict[int, dict[str, dict[str, float]]],
    cycles: list[int],
    ranks: int,
) -> dict[str, object]:
    max_mean: list[float] = []
    p95_median: list[float] = []
    total_gib: list[float] = []
    rank_max_gib: list[float] = []
    event_wall: list[float] = []
    lifetime_peak_rank_gib: list[float] = []
    for cycle in cycles:
        rss = records[cycle]["current-rss"]
        wall = records[cycle]["event-wall"]
        if rss.get("mean", 0.0) <= 0.0 or rss.get("median", 0.0) <= 0.0:
            raise ValueError(f"non-positive RSS statistic at cycle {cycle} in {path}")
        max_mean.append(rss["max"] / rss["mean"])
        p95_median.append(rss["p95"] / rss["median"])
        total_gib.append(rss["mean"] * ranks / (1024.0 * 1024.0))
        rank_max_gib.append(rss["max"] / (1024.0 * 1024.0))
        event_wall.append(wall["max"])
        peak = records[cycle].get("peak-rss")
        if peak is not None and "max" in peak:
            lifetime_peak_rank_gib.append(peak["max"] / (1024.0 * 1024.0))
    result: dict[str, object] = {
        "log": str(path.resolve()),
        "rank_count": ranks,
        "cycle_first": cycles[0],
        "cycle_last": cycles[-1],
        "cycle_count": len(cycles),
        "rank_rss_max_over_mean": distribution(max_mean),
        "rank_rss_p95_over_median": distribution(p95_median),
        "total_live_rss_gib": distribution(total_gib),
        "max_rank_live_rss_gib": distribution(rank_max_gib),
        "event_wall_seconds": {
            **distribution(event_wall),
            "sum": sum(event_wall),
        },
    }
    if lifetime_peak_rank_gib:
        result["max_rank_lifetime_peak_rss_gib"] = max(lifetime_peak_rank_gib)
    return result


def whole_run_summary(
    path: Path, data: dict[str, object], ranks: int
) -> dict[str, object]:
    records = data["records"]
    assert isinstance(records, dict)
    cycles = usable_cycles(records, 0)
    result = summarize(path, records, cycles, ranks)
    result.pop("max_rank_lifetime_peak_rss_gib", None)
    final_peak = records[cycles[-1]].get("peak-rss")
    if final_peak is not None and "max" in final_peak:
        result["max_rank_lifetime_peak_rss_gib"] = final_peak["max"] / (
            1024.0 * 1024.0
        )
    wall_seconds = [records[cycle]["event-wall"]["max"] for cycle in cycles]
    wall_sum = sum(wall_seconds)
    if wall_sum <= 0.0:
        raise ValueError(f"non-positive event-wall sum in {path}")
    mean_rss_kib = [records[cycle]["current-rss"]["mean"] for cycle in cycles]
    max_rss_kib = [records[cycle]["current-rss"]["max"] for cycle in cycles]
    scale = ranks / (1024.0 * 1024.0)
    result.update(
        {
            "endpoint": data["endpoint"],
            "total_live_rss_wall_weighted_mean_gib": sum(
                rss * wall for rss, wall in zip(mean_rss_kib, wall_seconds)
            )
            / wall_sum
            * scale,
            "total_live_rss_peak_gib": max(mean_rss_kib) * scale,
            "max_rank_current_rss_peak_gib": max(max_rss_kib) / (1024.0 * 1024.0),
            "endpoint_total_live_rss_gib": mean_rss_kib[-1] * scale,
            "endpoint_max_rank_current_rss_gib": max_rss_kib[-1] / (1024.0 * 1024.0),
        }
    )
    return result


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0)


def event_time_map(
    data: dict[str, object],
) -> tuple[dict[float, dict[str, float | int]], int]:
    events = data["events"]
    assert isinstance(events, dict)
    by_time: dict[float, dict[str, float | int]] = {}
    duplicate_times: set[float] = set()
    duplicate_count = 0
    for event in events.values():
        if "event_time" not in event:
            continue
        event_time = float(event["event_time"])
        if event_time in duplicate_times:
            duplicate_count += 1
            continue
        if event_time in by_time:
            duplicate_count += 1
            duplicate_times.add(event_time)
            del by_time[event_time]
            continue
        by_time[event_time] = event
    return by_time, duplicate_count


def matched_comparison(
    baseline_data: dict[str, object],
    optimized_data: dict[str, object],
    late_cycles: int,
    work_relative_tolerance: float,
) -> dict[str, object]:
    baseline_by_time, baseline_duplicates = event_time_map(baseline_data)
    optimized_by_time, optimized_duplicates = event_time_map(optimized_data)
    common_times = sorted(baseline_by_time.keys() & optimized_by_time.keys())
    excluded_reasons: Counter[str] = Counter()
    pairs: list[tuple[int, int, float]] = []

    baseline_records = baseline_data["records"]
    optimized_records = optimized_data["records"]
    assert isinstance(baseline_records, dict)
    assert isinstance(optimized_records, dict)

    for event_time in common_times:
        baseline = baseline_by_time[event_time]
        optimized = optimized_by_time[event_time]
        baseline_cycle = int(baseline["cycle"])
        optimized_cycle = int(optimized["cycle"])
        reasons: list[str] = []
        for field in ("active_cells", "canonical_cells"):
            if field not in baseline or field not in optimized:
                reasons.append(f"missing_{field}")
            elif (
                relative_difference(float(baseline[field]), float(optimized[field]))
                > work_relative_tolerance
            ):
                reasons.append(field)
        for field in ("amr", "ownership_epoch"):
            if field not in baseline or field not in optimized:
                reasons.append(f"missing_{field}")
            elif int(baseline[field]) != int(optimized[field]):
                reasons.append(field)
        if (
            baseline_cycle not in baseline_records
            or optimized_cycle not in optimized_records
            or "current-rss" not in baseline_records[baseline_cycle]
            or "event-wall" not in baseline_records[baseline_cycle]
            or "current-rss" not in optimized_records[optimized_cycle]
            or "event-wall" not in optimized_records[optimized_cycle]
        ):
            reasons.append("performance")
        if reasons:
            excluded_reasons.update(set(reasons))
            continue
        pairs.append((baseline_cycle, optimized_cycle, event_time))

    selected = pairs[-late_cycles:] if late_cycles > 0 else pairs
    result: dict[str, object] = {
        "available": bool(baseline_by_time and optimized_by_time),
        "basis": "exact physical event_time plus equivalent event work",
        "work_relative_tolerance": work_relative_tolerance,
        "baseline_event_time_count": len(baseline_by_time),
        "optimized_event_time_count": len(optimized_by_time),
        "common_event_time_count": len(common_times),
        "accepted_event_count": len(pairs),
        "selected_event_count": len(selected),
        "baseline_duplicate_event_time_count": baseline_duplicates,
        "optimized_duplicate_event_time_count": optimized_duplicates,
        "baseline_only_event_time_count": len(
            baseline_by_time.keys() - optimized_by_time.keys()
        ),
        "optimized_only_event_time_count": len(
            optimized_by_time.keys() - baseline_by_time.keys()
        ),
        "excluded_common_event_count": len(common_times) - len(pairs),
        "excluded_by_reason": dict(sorted(excluded_reasons.items())),
    }
    if not selected:
        return result

    baseline_cycles = [pair[0] for pair in selected]
    optimized_cycles = [pair[1] for pair in selected]
    baseline_mean = [
        baseline_records[cycle]["current-rss"]["mean"] for cycle in baseline_cycles
    ]
    optimized_mean = [
        optimized_records[cycle]["current-rss"]["mean"] for cycle in optimized_cycles
    ]
    baseline_max = [
        baseline_records[cycle]["current-rss"]["max"] for cycle in baseline_cycles
    ]
    optimized_max = [
        optimized_records[cycle]["current-rss"]["max"] for cycle in optimized_cycles
    ]
    baseline_wall = [
        baseline_records[cycle]["event-wall"]["max"] for cycle in baseline_cycles
    ]
    optimized_wall = [
        optimized_records[cycle]["event-wall"]["max"] for cycle in optimized_cycles
    ]
    result.update(
        {
            "event_time_first": selected[0][2],
            "event_time_last": selected[-1][2],
            "baseline_cycle_first": baseline_cycles[0],
            "baseline_cycle_last": baseline_cycles[-1],
            "optimized_cycle_first": optimized_cycles[0],
            "optimized_cycle_last": optimized_cycles[-1],
            "total_live_rss_mean_ratio": statistics.fmean(optimized_mean)
            / statistics.fmean(baseline_mean),
            "total_live_rss_peak_ratio": max(optimized_mean) / max(baseline_mean),
            "max_rank_live_rss_peak_ratio": max(optimized_max) / max(baseline_max),
            "event_wall_sum_ratio": sum(optimized_wall) / sum(baseline_wall),
            "event_wall_median_ratio": statistics.median(optimized_wall)
            / statistics.median(baseline_wall),
            # Schema-1 aliases. They now name the baseline side explicitly above.
            "cycle_first": baseline_cycles[0],
            "cycle_last": baseline_cycles[-1],
            "cycle_count": len(selected),
        }
    )
    return result


def compare_endpoints(
    baseline: dict[str, object],
    optimized: dict[str, object],
    work_relative_tolerance: float,
) -> dict[str, object]:
    reasons: list[str] = []
    if not baseline.get("complete") or not optimized.get("complete"):
        reasons.append("missing_completed_endpoint")
    baseline_tick = baseline.get("final_tick")
    optimized_tick = optimized.get("final_tick")
    if baseline_tick is None or optimized_tick is None:
        reasons.append("final_tick_missing")
    elif int(baseline_tick) != int(optimized_tick):
        reasons.append("final_tick")
    baseline_time = baseline.get("final_time")
    optimized_time = optimized.get("final_time")
    if baseline_time is None or optimized_time is None:
        reasons.append("final_time_missing")
    elif not math.isclose(
        float(baseline_time),
        float(optimized_time),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        reasons.append("final_time")
    baseline_cells = baseline.get("final_cells")
    optimized_cells = optimized.get("final_cells")
    final_cells_relative_difference: float | None = None
    if baseline_cells is None or optimized_cells is None:
        reasons.append("final_cells_missing")
    else:
        final_cells_relative_difference = relative_difference(
            float(baseline_cells), float(optimized_cells)
        )
        if final_cells_relative_difference > work_relative_tolerance:
            reasons.append("final_cells")
    return {
        "compatible": not reasons,
        "reasons": reasons,
        "final_cells_relative_difference": final_cells_relative_difference,
        "baseline": baseline,
        "optimized": optimized,
    }


def positive_ratio(numerator: object, denominator: object) -> float | None:
    if numerator is None or denominator is None:
        return None
    numerator_value = float(numerator)
    denominator_value = float(denominator)
    if (
        not math.isfinite(numerator_value)
        or not math.isfinite(denominator_value)
        or numerator_value <= 0.0
        or denominator_value <= 0.0
    ):
        return None
    return numerator_value / denominator_value


def whole_run_comparison(
    baseline: dict[str, object],
    optimized: dict[str, object],
    work_relative_tolerance: float,
) -> dict[str, object]:
    endpoint = compare_endpoints(
        baseline["endpoint"], optimized["endpoint"], work_relative_tolerance
    )
    baseline_endpoint = baseline["endpoint"]
    optimized_endpoint = optimized["endpoint"]
    comparison = {
        "endpoint": endpoint,
        "evolution_wall_ratio": positive_ratio(
            optimized_endpoint.get("evolution_wall_seconds"),
            baseline_endpoint.get("evolution_wall_seconds"),
        ),
        "event_wall_sum_ratio": positive_ratio(
            optimized["event_wall_seconds"]["sum"],
            baseline["event_wall_seconds"]["sum"],
        ),
        "total_live_rss_mean_ratio": positive_ratio(
            optimized["total_live_rss_wall_weighted_mean_gib"],
            baseline["total_live_rss_wall_weighted_mean_gib"],
        ),
        "total_live_rss_peak_ratio": positive_ratio(
            optimized["total_live_rss_peak_gib"],
            baseline["total_live_rss_peak_gib"],
        ),
        "max_rank_live_rss_peak_ratio": positive_ratio(
            optimized["max_rank_current_rss_peak_gib"],
            baseline["max_rank_current_rss_peak_gib"],
        ),
        "endpoint_total_live_rss_ratio": positive_ratio(
            optimized["endpoint_total_live_rss_gib"],
            baseline["endpoint_total_live_rss_gib"],
        ),
        "endpoint_max_rank_current_rss_ratio": positive_ratio(
            optimized["endpoint_max_rank_current_rss_gib"],
            baseline["endpoint_max_rank_current_rss_gib"],
        ),
        "max_rank_lifetime_peak_rss_ratio": positive_ratio(
            optimized.get("max_rank_lifetime_peak_rss_gib"),
            baseline.get("max_rank_lifetime_peak_rss_gib"),
        ),
    }
    return comparison


def analyze(
    optimized_path: Path,
    baseline_path: Path | None = None,
    ranks: int = 128,
    late_cycles: int = 58,
    max_over_mean: float = 1.25,
    p95_over_median: float = 1.20,
    wall_regression: float = 0.02,
    work_relative_tolerance: float = DEFAULT_WORK_RELATIVE_TOLERANCE,
) -> dict[str, object]:
    optimized_data = parse_log(optimized_path)
    optimized_records = optimized_data["records"]
    assert isinstance(optimized_records, dict)
    optimized_cycles = usable_cycles(optimized_records, late_cycles)
    optimized_summary = summarize(
        optimized_path, optimized_records, optimized_cycles, ranks
    )
    optimized_whole_run = whole_run_summary(optimized_path, optimized_data, ranks)
    gates: dict[str, bool] = {
        "rss_max_over_mean": optimized_summary["rank_rss_max_over_mean"]["max"]
        <= max_over_mean,
        "rss_p95_over_median": optimized_summary["rank_rss_p95_over_median"]["max"]
        <= p95_over_median,
    }
    result: dict[str, object] = {
        "schema_version": 2,
        "optimized": optimized_summary,
        "whole_run": {"optimized": optimized_whole_run},
        "thresholds": {
            "rank_rss_max_over_mean": max_over_mean,
            "rank_rss_p95_over_median": p95_over_median,
            "maximum_event_wall_regression": wall_regression,
            "work_relative_tolerance": work_relative_tolerance,
        },
        "gates": gates,
    }

    if baseline_path is not None:
        baseline_data = parse_log(baseline_path)
        baseline_records = baseline_data["records"]
        assert isinstance(baseline_records, dict)
        baseline_cycles = usable_cycles(baseline_records, late_cycles)
        baseline_summary = summarize(
            baseline_path, baseline_records, baseline_cycles, ranks
        )
        baseline_whole_run = whole_run_summary(baseline_path, baseline_data, ranks)
        matching = matched_comparison(
            baseline_data,
            optimized_data,
            late_cycles,
            work_relative_tolerance,
        )
        comparison = whole_run_comparison(
            baseline_whole_run, optimized_whole_run, work_relative_tolerance
        )
        result["baseline"] = baseline_summary
        result["matched"] = matching
        result["whole_run"] = {
            "baseline": baseline_whole_run,
            "optimized": optimized_whole_run,
            "comparison": comparison,
        }

        endpoint_compatible = bool(comparison["endpoint"]["compatible"])
        evolution_wall_ratio = comparison["evolution_wall_ratio"]
        gates.update(
            {
                "comparable_completed_endpoints": endpoint_compatible,
                "total_live_rss_mean_not_higher": comparison[
                    "total_live_rss_mean_ratio"
                ]
                <= 1.0,
                "total_live_rss_peak_not_higher": comparison[
                    "total_live_rss_peak_ratio"
                ]
                <= 1.0,
                "max_rank_live_rss_peak_not_higher": comparison[
                    "max_rank_live_rss_peak_ratio"
                ]
                <= 1.0,
                "event_wall_no_more_than_2pct_slower": endpoint_compatible
                and evolution_wall_ratio is not None
                and evolution_wall_ratio <= 1.0 + wall_regression,
            }
        )
        lifetime_peak_ratio = comparison["max_rank_lifetime_peak_rss_ratio"]
        gates["lifetime_peak_rss_evidence_available"] = lifetime_peak_ratio is not None
        gates["max_rank_lifetime_peak_rss_not_higher"] = (
            lifetime_peak_ratio is not None and lifetime_peak_ratio <= 1.0
        )

    result["pass"] = all(gates.values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimized", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--ranks", type=int, default=128)
    parser.add_argument(
        "--late-cycles",
        type=int,
        default=58,
        help=(
            "use the final N optimized cycles for equality and final N "
            "physically matched events for diagnostics; 0 means all"
        ),
    )
    parser.add_argument("--max-over-mean", type=float, default=1.25)
    parser.add_argument("--p95-over-median", type=float, default=1.20)
    parser.add_argument("--wall-regression", type=float, default=0.02)
    parser.add_argument(
        "--work-relative-tolerance",
        type=float,
        default=DEFAULT_WORK_RELATIVE_TOLERANCE,
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.ranks <= 0 or args.late_cycles < 0:
        parser.error("--ranks must be positive and --late-cycles non-negative")
    for name in (
        "max_over_mean",
        "p95_over_median",
        "wall_regression",
        "work_relative_tolerance",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")

    result = analyze(
        optimized_path=args.optimized,
        baseline_path=args.baseline,
        ranks=args.ranks,
        late_cycles=args.late_cycles,
        max_over_mean=args.max_over_mean,
        p95_over_median=args.p95_over_median,
        wall_regression=args.wall_regression,
        work_relative_tolerance=args.work_relative_tolerance,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
