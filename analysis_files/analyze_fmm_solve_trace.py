#!/usr/bin/env python3
"""Summarize opt-in ``fmm_solve_trace`` lines from RICH logs."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable

PHASE_NAMES = (
    "total",
    "build",
    "topology",
    "descriptor",
    "process_topology",
    "let_plan",
    "let_reset",
    "let_descriptor_traversal",
    "let_finalize",
    "let_subscription",
    "let_prune_compact",
    "local_traversal",
    "let_execute",
    "upward",
    "process_upward",
    "process_interaction",
    "process_downward",
    "let_exchange",
    "let_m2l",
    "let_p2p",
    "downward",
)
TIMING_KEYS = tuple(
    f"{phase}_{suffix}"
    for phase in PHASE_NAMES
    for suffix in ("min", "mean", "max")
) + ("let_residual_wait_max", "let_payload_lifetime_max")

INTEGER_KEYS = (
    "call",
    "epoch",
    "rebuilds",
    "process_rebuilds",
    "let_rebuilds",
    "root_change_ranks",
    "leaf_change_ranks",
    "occupancy_change_ranks",
    "count_only_change_ranks",
    "persistent_refit_ranks",
    "persistent_leaf_splits",
    "persistent_subtree_merges",
    "persistent_empty_leaves",
    "count_only_reused",
    "process_rebuilt",
    "let_rebuilt",
    "process_comm_reused",
    "let_comm_reused",
    "let_storage_reused",
    "let_plan_bytes_max",
    "local_inactive_m2l_sum",
    "local_inactive_p2p_blocks_sum",
    "let_inactive_m2l_sum",
    "let_inactive_p2p_blocks_sum",
    "let_zero_multipole_payloads_sum",
    "let_omitted_multipole_payloads_sum",
    "let_omitted_particle_payloads_sum",
    "bytes_sent_sum",
    "bytes_received_sum",
    "local_planned_m2l_sum",
    "local_planned_p2p_blocks_sum",
    "let_planned_m2l_sum",
    "let_planned_p2p_blocks_sum",
    "let_active_m2l_sum",
    "let_active_p2p_blocks_sum",
    "bytes_owned_max",
    "peak_remote_bytes_max",
    "forced_rebuild",
    "active_ranks",
    "local_plan_reused_ranks",
    "local_plan_reused_all",
    "local_cache_hits_sum",
    "local_cache_misses_sum",
    "local_cache_bypasses_sum",
    "let_cache_hits_sum",
    "let_cache_misses_sum",
    "let_cache_bypasses_sum",
    "process_cache_misses_sum",
    "process_cache_bypasses_sum",
)


def parse_line(line: str) -> dict[str, float | int] | None:
    marker = "fmm_solve_trace "
    location = line.find(marker)
    if location < 0:
        return None
    record: dict[str, float | int] = {}
    for token in line[location + len(marker) :].split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        try:
            if key in INTEGER_KEYS:
                record[key] = int(value)
            elif key in TIMING_KEYS:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"non-finite value for {key}")
                record[key] = number
        except ValueError as error:
            raise ValueError(f"could not parse token {token!r}") from error
    if "call" not in record or "total_max" not in record:
        return None
    return record


def read_records(paths: Iterable[Path]) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
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


def median(records: list[dict[str, float | int]], key: str) -> float:
    values = [float(record[key]) for record in records if key in record]
    return statistics.median(values) if values else float("nan")


def mean(records: list[dict[str, float | int]], key: str) -> float:
    values = [float(record[key]) for record in records if key in record]
    return statistics.fmean(values) if values else float("nan")


def fraction(records: list[dict[str, float | int]], key: str) -> float:
    values = [int(record[key]) != 0 for record in records if key in record]
    return statistics.fmean(values) if values else float("nan")


def number(record: dict[str, float | int], key: str) -> float:
    value = record.get(key)
    return float(value) if value is not None else float("nan")


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if math.isfinite(numerator) and denominator > 0.0 else float("nan")


def derived_values(record: dict[str, float | int]) -> dict[str, float]:
    local_active_m2l = number(record, "local_planned_m2l_sum") - number(
        record, "local_inactive_m2l_sum")
    local_active_p2p = number(record, "local_planned_p2p_blocks_sum") - number(
        record, "local_inactive_p2p_blocks_sum")
    let_active_m2l = number(record, "let_active_m2l_sum")
    let_active_p2p = number(record, "let_active_p2p_blocks_sum")
    total = number(record, "total_max")
    local_traversal = number(record, "local_traversal_max")
    let_execute = number(record, "let_execute_max")
    accounted = (
        number(record, "build_max")
        + number(record, "upward_max")
        + max(local_traversal, let_execute)
        + number(record, "downward_max")
    )
    local_cache_total = sum(number(record, key) for key in (
        "local_cache_hits_sum", "local_cache_misses_sum"))
    let_cache_total = sum(number(record, key) for key in (
        "let_cache_hits_sum", "let_cache_misses_sum"))
    omitted = sum(number(record, key) for key in (
        "let_omitted_multipole_payloads_sum",
        "let_omitted_particle_payloads_sum"))
    payload_records = omitted + let_active_m2l + let_active_p2p
    payload_lifetime = number(record, "let_payload_lifetime_max")
    residual_wait = number(record, "let_residual_wait_max")
    return {
        "phase_share_build": safe_ratio(number(record, "build_max"), total),
        "phase_share_local_traversal": safe_ratio(local_traversal, total),
        "phase_share_let_execute": safe_ratio(let_execute, total),
        "phase_share_let_m2l": safe_ratio(number(record, "let_m2l_max"), total),
        "phase_share_let_p2p": safe_ratio(number(record, "let_p2p_max"), total),
        "phase_share_let_exchange": safe_ratio(number(record, "let_exchange_max"), total),
        "unaccounted_seconds": total - accounted if math.isfinite(total) else float("nan"),
        "unaccounted_fraction": safe_ratio(total - accounted, total),
        "local_active_m2l": local_active_m2l,
        "local_active_p2p_blocks": local_active_p2p,
        "local_active_fraction_m2l": safe_ratio(local_active_m2l, number(record, "local_planned_m2l_sum")),
        "local_active_fraction_p2p": safe_ratio(local_active_p2p, number(record, "local_planned_p2p_blocks_sum")),
        "let_active_fraction_m2l": safe_ratio(let_active_m2l, number(record, "let_planned_m2l_sum")),
        "let_active_fraction_p2p": safe_ratio(let_active_p2p, number(record, "let_planned_p2p_blocks_sum")),
        "payload_omission_fraction": safe_ratio(omitted, payload_records),
        "bytes_per_active_let_m2l": safe_ratio(number(record, "bytes_received_sum"), let_active_m2l),
        "bytes_per_active_let_p2p_block": safe_ratio(number(record, "bytes_received_sum"), let_active_p2p),
        "local_cache_hit_rate": safe_ratio(number(record, "local_cache_hits_sum"), local_cache_total),
        "let_cache_hit_rate": safe_ratio(number(record, "let_cache_hits_sum"), let_cache_total),
        "exchange_hidden_fraction": (
            1.0 - safe_ratio(residual_wait, payload_lifetime)
            if math.isfinite(payload_lifetime) and payload_lifetime > 0.0 else float("nan")
        ),
    }


def print_group(name: str, records: list[dict[str, float | int]]) -> None:
    print(f"[{name}] calls={len(records)}")
    for phase in PHASE_NAMES:
        values = " ".join(
            f"{suffix}={median(records, f'{phase}_{suffix}'):.8e}"
            for suffix in ("min", "mean", "max")
        )
        print(f"  {phase:28s}{values}")
    for key in ("let_residual_wait_max", "let_payload_lifetime_max"):
        print(f"  {key:28s}median={median(records, key):.8e} mean={mean(records, key):.8e}")
    for key in (
        "process_rebuilt",
        "let_rebuilt",
        "process_comm_reused",
        "let_comm_reused",
        "let_storage_reused",
        "count_only_reused",
        "forced_rebuild",
        "local_plan_reused_all",
    ):
        print(f"  {key:24s} fraction={fraction(records, key):.6f}")
    root_changes = [int(record.get("root_change_ranks", 0)) for record in records]
    leaf_changes = [int(record.get("leaf_change_ranks", 0)) for record in records]
    occupancy_changes = [
        int(record.get("occupancy_change_ranks", 0)) for record in records
    ]
    count_only_changes = [
        int(record.get("count_only_change_ranks", 0)) for record in records
    ]
    persistent_refits = [
        int(record.get("persistent_refit_ranks", 0)) for record in records
    ]
    persistent_splits = [
        int(record.get("persistent_leaf_splits", 0)) for record in records
    ]
    persistent_merges = [
        int(record.get("persistent_subtree_merges", 0)) for record in records
    ]
    persistent_empty = [
        int(record.get("persistent_empty_leaves", 0)) for record in records
    ]
    print(f"  root_change_ranks_max    {max(root_changes, default=0)}")
    print(f"  leaf_change_ranks_max    {max(leaf_changes, default=0)}")
    print(f"  occupancy_change_ranks_max {max(occupancy_changes, default=0)}")
    print(f"  count_only_change_ranks_max {max(count_only_changes, default=0)}")
    print(f"  persistent_refit_ranks_max {max(persistent_refits, default=0)}")
    print(f"  persistent_leaf_splits_max {max(persistent_splits, default=0)}")
    print(f"  persistent_subtree_merges_max {max(persistent_merges, default=0)}")
    print(f"  persistent_empty_leaves_max {max(persistent_empty, default=0)}")
    for key in (
        "local_inactive_m2l_sum",
        "local_inactive_p2p_blocks_sum",
        "let_inactive_m2l_sum",
        "let_inactive_p2p_blocks_sum",
        "let_zero_multipole_payloads_sum",
    ):
        print(
            f"  {key:32s} median={median(records, key):.1f} "
            f"mean={mean(records, key):.1f}"
        )
    derived = [derived_values(record) for record in records]
    print("  derived_metrics")
    for key in sorted({key for values in derived for key in values}):
        values = [values[key] for values in derived if math.isfinite(values[key])]
        print(
            f"    {key:30s} median={statistics.median(values):.8e} "
            f"mean={statistics.fmean(values):.8e}"
            if values else f"    {key:30s} median=nan mean=nan"
        )


def write_tsv(path: Path, records: list[dict[str, float | int]]) -> None:
    fields = INTEGER_KEYS + TIMING_KEYS + ("source", "step", "second_over_first")
    by_call = {int(record["call"]): record for record in records}
    with path.open("w", encoding="utf-8") as stream:
        stream.write("\t".join(fields) + "\n")
        for record in records:
            call = int(record["call"])
            source = "first" if call % 2 == 1 else "second"
            step = (call + 1) // 2
            ratio = ""
            if source == "second" and call - 1 in by_call:
                first = float(by_call[call - 1]["total_max"])
                if first > 0.0:
                    ratio = f"{float(record['total_max']) / first:.16e}"
            values: list[str] = []
            for field in fields:
                if field == "source":
                    values.append(source)
                elif field == "step":
                    values.append(str(step))
                elif field == "second_over_first":
                    values.append(ratio)
                else:
                    values.append(str(record.get(field, "")))
            stream.write("\t".join(values) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--tsv", type=Path)
    parser.add_argument(
        "--skip-calls",
        type=int,
        default=2,
        help="ignore this many startup gravity calls in aggregate statistics",
    )
    args = parser.parse_args()

    records = read_records(args.logs)
    if not records:
        print("No fmm_solve_trace records found.", file=sys.stderr)
        return 1
    if args.tsv is not None:
        write_tsv(args.tsv, records)

    selected = records[max(0, args.skip_calls) :]
    first = [record for record in selected if int(record["call"]) % 2 == 1]
    second = [record for record in selected if int(record["call"]) % 2 == 0]
    print(f"records={len(records)} selected={len(selected)}")
    print_group("first source", first)
    print_group("second source", second)

    paired_ratios: list[float] = []
    by_call = {int(record["call"]): record for record in selected}
    for call, record in by_call.items():
        if call % 2 != 0 or call - 1 not in by_call:
            continue
        first_time = float(by_call[call - 1]["total_max"])
        if first_time > 0.0:
            paired_ratios.append(float(record["total_max"]) / first_time)
    if paired_ratios:
        print(
            "second_over_first "
            f"median={statistics.median(paired_ratios):.8e} "
            f"mean={statistics.fmean(paired_ratios):.8e} "
            f"min={min(paired_ratios):.8e} max={max(paired_ratios):.8e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
