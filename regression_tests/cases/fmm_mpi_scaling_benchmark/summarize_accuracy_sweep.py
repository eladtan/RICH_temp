#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_key_value_line(tokens: List[str]) -> Dict[str, str]:
    if (len(tokens) - 1) % 2 != 0:
        raise ValueError(f"expected key/value pairs: {' '.join(tokens)}")
    return {tokens[i]: tokens[i + 1] for i in range(1, len(tokens), 2)}


def parse_result(path: Path) -> Dict[str, object]:
    columns: Optional[List[str]] = None
    row: Optional[List[str]] = None
    config: Optional[Dict[str, str]] = None
    profiles: Dict[Tuple[str, str, str], Tuple[float, float, float, float]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            tokens = raw_line.split()
            if not tokens:
                continue
            if tokens[0] == "fmm_config":
                config = parse_key_value_line(tokens)
            elif tokens[0] == "columns":
                columns = tokens[1:]
            elif tokens[0] == "row":
                row = tokens[1:]
            elif tokens[0] == "profile" and len(tokens) == 8:
                profiles[(tokens[1], tokens[2], tokens[3])] = tuple(
                    float(value) for value in tokens[4:8]
                )

    if config is None or columns is None or row is None:
        raise ValueError(f"{path}: missing fmm_config, columns, or row line")
    if len(columns) != len(row):
        raise ValueError(
            f"{path}: columns/row width mismatch ({len(columns)} != {len(row)})"
        )

    values = dict(zip(columns, row))

    def number(name: str) -> float:
        value = float(values[name])
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite {name}")
        return value

    def profile(mode: str, category: str, metric: str, index: int) -> float:
        return profiles.get((mode, category, metric), (math.nan,) * 4)[index]

    mean_target = float(config["mean_error_target"])
    max_target = float(config["max_error_target"])
    mean_error = number("fmm_probe_mean_scaled_error")
    max_error = number("fmm_probe_scaled_error")
    run_pass = int(float(values["run_pass"])) == 1
    hard_pass = run_pass and mean_error <= mean_target and max_error <= max_target
    safe_pass = hard_pass and mean_error <= min(8e-4, mean_target)

    return {
        "file": path.name,
        "order": int(config["order"]),
        "theta": float(config["theta"]),
        "leaf_capacity": int(config["leaf_capacity"]),
        "coefficient_count": int(config["coefficient_count"]),
        "m2l_term_count": int(config["m2l_term_count"]),
        "warm_best_seconds": number("fmm_warm_best_max_seconds"),
        "warm_mean_seconds": number("fmm_warm_mean_max_seconds"),
        "mean_scaled_error": mean_error,
        "max_scaled_error": max_error,
        "hard_pass": int(hard_pass),
        "safe_pass": int(safe_pass),
        "let_m2l_rank_mean_seconds": profile("warm", "timing", "let_m2l_seconds", 1),
        "let_m2l_rank_max_seconds": profile("warm", "timing", "let_m2l_seconds", 2),
        "local_traversal_rank_mean_seconds": profile(
            "warm", "timing", "local_traversal_seconds", 1
        ),
        "local_traversal_rank_max_seconds": profile(
            "warm", "timing", "local_traversal_seconds", 2
        ),
        "let_m2l_rank_mean_count": profile("warm", "work", "let_m2l", 1),
        "let_p2p_rank_mean_pairs": profile("warm", "work", "let_p2p_pairs", 1),
        "bytes_sent": int(float(values["fmm_bytes_sent"])),
        "bytes_received": int(float(values["fmm_bytes_received"])),
        "local_cache_bypasses": int(float(values["fmm_local_operator_cache_bypasses"])),
        "let_cache_bypasses": int(float(values["fmm_let_operator_cache_bypasses"])),
    }


def sort_key(row: Dict[str, object]) -> Tuple[int, float, int, float]:
    return (
        0 if row["safe_pass"] else (1 if row["hard_pass"] else 2),
        float(row["warm_mean_seconds"]),
        int(row["order"]),
        float(row["theta"]),
    )


def write_tsv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("no result rows to write")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize parameterized distributed-FMM accuracy sweeps."
    )
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = sorted(args.result_directory.glob("p*_t*_l*.txt"))
    if not paths:
        parser.error(f"no sweep result files found in {args.result_directory}")

    rows = sorted((parse_result(path) for path in paths), key=sort_key)
    output = args.output or args.result_directory / "accuracy_sweep_summary.tsv"
    write_tsv(output, rows)

    print(
        "rank order theta leaf warm_mean_s mean_error max_error safe hard "
        "let_m2l_mean_s local_mean_s"
    )
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:4d} {row['order']:5d} {row['theta']:5.2f} "
            f"{row['leaf_capacity']:4d} {row['warm_mean_seconds']:11.6f} "
            f"{row['mean_scaled_error']:10.3e} {row['max_scaled_error']:10.3e} "
            f"{row['safe_pass']:4d} {row['hard_pass']:4d} "
            f"{row['let_m2l_rank_mean_seconds']:15.6f} "
            f"{row['local_traversal_rank_mean_seconds']:12.6f}"
        )

    safe = [row for row in rows if row["safe_pass"]]
    hard = [row for row in rows if row["hard_pass"]]
    selected = safe[0] if safe else (hard[0] if hard else None)
    if selected is None:
        print("selection none: no configuration met both error targets")
        return 1

    print(
        "selection "
        f"p={selected['order']} theta={selected['theta']:.2f} "
        f"leaf={selected['leaf_capacity']} "
        f"warm_mean={selected['warm_mean_seconds']:.6f}s "
        f"mean_error={selected['mean_scaled_error']:.3e} "
        f"max_error={selected['max_scaled_error']:.3e} "
        f"safety_guardrail={selected['safe_pass']}"
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
