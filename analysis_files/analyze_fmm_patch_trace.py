#!/usr/bin/env python3
"""Summarize patch-forest fields from RICH ``fmm_solve_trace`` lines."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable

INTEGER_FIELDS = (
    "call",
    "epoch",
    "process_rebuilds",
    "let_rebuilds",
    "local_patch_count_sum",
    "global_patch_count",
    "reused_patch_count_sum",
    "reused_patch_plan_count_sum",
    "rebuilt_patch_plan_count_sum",
    "patch_geometry_expansions_sum",
    "patch_retained_bytes_sum",
    "patch_released_bytes_sum",
    "replicated_descriptor_bytes_max",
    "process_tree_bytes_max",
    "process_plan_bytes_max",
    "process_owned_nodes_max",
    "process_owned_m2l_max",
    "let_target_subplans_reused_sum",
    "let_target_subplans_rebuilt_sum",
    "let_source_invalidations_sum",
    "let_wave_plan_rebuilds_sum",
    "let_descriptor_traversal_skipped_sum",
    "let_payload_shape_rebuild_ranks",
    "let_wave_count_max",
    "bytes_owned_max",
    "peak_remote_bytes_max",
)

FLOAT_FIELDS = (
    "total_max",
    "topology_max",
    "let_plan_max",
    "let_execute_max",
    "process_owner_imbalance_max",
)


def parse_line(line: str) -> dict[str, int | float] | None:
    marker = "fmm_solve_trace "
    location = line.find(marker)
    if location < 0:
        return None
    record: dict[str, int | float] = {}
    for token in line[location + len(marker) :].split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        try:
            if key in INTEGER_FIELDS:
                record[key] = int(value)
            elif key in FLOAT_FIELDS:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("non-finite value")
                record[key] = number
        except ValueError as error:
            raise ValueError(f"could not parse token {token!r}") from error
    if "call" not in record:
        return None
    return record


def read_records(paths: Iterable[Path]) -> list[dict[str, int | float]]:
    records: list[dict[str, int | float]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = parse_line(line)
                except ValueError as error:
                    raise SystemExit(f"{path}:{line_number}: {error}") from error
                if record is not None:
                    records.append(record)
    records.sort(key=lambda item: int(item["call"]))
    return records


def numeric_values(
    records: list[dict[str, int | float]], field: str
) -> list[float]:
    return [float(record[field]) for record in records if field in record]


def print_summary(records: list[dict[str, int | float]]) -> None:
    print(f"records {len(records)}")
    for field in INTEGER_FIELDS + FLOAT_FIELDS:
        values = numeric_values(records, field)
        if not values:
            continue
        print(
            f"{field} median={statistics.median(values):.8e} "
            f"mean={statistics.fmean(values):.8e} "
            f"min={min(values):.8e} max={max(values):.8e}"
        )

    rebuilds = [
        int(record.get("process_rebuilds", 0)) +
        int(record.get("let_rebuilds", 0))
        for record in records
    ]
    payload_rebuilds = [
        int(record.get("let_payload_shape_rebuild_ranks", 0)) > 0
        for record in records
    ]
    if rebuilds:
        print(f"combined_rebuild_counter_max {max(rebuilds)}")
    if payload_rebuilds:
        print(
            "payload_shape_rebuild_fraction "
            f"{statistics.fmean(payload_rebuilds):.8e}"
        )


def write_tsv(path: Path, records: list[dict[str, int | float]]) -> None:
    fields = INTEGER_FIELDS + FLOAT_FIELDS
    with path.open("w", encoding="utf-8") as stream:
        stream.write("\t".join(fields) + "\n")
        for record in records:
            stream.write(
                "\t".join(str(record.get(field, "")) for field in fields) +
                "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--tsv", type=Path)
    parser.add_argument(
        "--skip-calls",
        type=int,
        default=0,
        help="ignore this many startup gravity calls",
    )
    args = parser.parse_args()

    records = read_records(args.logs)
    if not records:
        print("No fmm_solve_trace records found.", file=sys.stderr)
        return 1
    selected = records[max(0, args.skip_calls) :]
    if args.tsv is not None:
        write_tsv(args.tsv, selected)
    print_summary(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
