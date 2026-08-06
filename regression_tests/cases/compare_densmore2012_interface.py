#!/usr/bin/env python3
"""Compare the paired exposed-interface MC and DDMC diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def read_tsv(path: Path) -> List[Dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"{path} has no header")
        rows: List[Dict[str, float]] = []
        for line_number, raw in enumerate(reader, start=2):
            try:
                rows.append({key: float(value) for key, value in raw.items()})
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Failed to parse {path}:{line_number}: {raw}"
                ) from exc
    return rows


def safe_relative(a: float, b: float) -> float:
    scale = abs(a)
    if scale == 0.0:
        return math.nan if b == 0.0 else math.copysign(math.inf, b)
    return (b - a) / scale


def region_summary(
    rows: Sequence[Dict[str, float]],
    x_min: float,
    x_max: float,
) -> Tuple[int, float, float, float]:
    selected = [row for row in rows if x_min <= row["x_cm"] < x_max]
    if not selected:
        return 0, math.nan, math.nan, math.nan

    abs_values = [abs(row["temperature_relative_difference"]) for row in selected]
    volume_sum = sum(row["volume_cm3"] for row in selected)
    volume_weighted = (
        sum(
            abs(row["temperature_relative_difference"]) * row["volume_cm3"]
            for row in selected
        )
        / volume_sum
        if volume_sum > 0.0
        else math.nan
    )
    signed_mean = sum(
        row["temperature_relative_difference"] for row in selected
    ) / len(selected)
    return len(selected), signed_mean, sum(abs_values) / len(abs_values), volume_weighted


def compare(mc_rows: Sequence[Dict[str, float]], ddmc_rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    if len(mc_rows) != len(ddmc_rows):
        raise RuntimeError(
            f"Cell counts differ: MC={len(mc_rows)}, DDMC={len(ddmc_rows)}"
        )

    result: List[Dict[str, float]] = []
    for index, (mc, ddmc) in enumerate(zip(mc_rows, ddmc_rows)):
        if not math.isclose(mc["x_cm"], ddmc["x_cm"], rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"Mesh mismatch at row {index}: MC x={mc['x_cm']}, "
                f"DDMC x={ddmc['x_cm']}"
            )
        if not math.isclose(mc["x_left_cm"], ddmc["x_left_cm"], rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Left face mismatch at x={mc['x_cm']}")
        if not math.isclose(mc["x_right_cm"], ddmc["x_right_cm"], rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Right face mismatch at x={mc['x_cm']}")

        row = {
            "x_cm": mc["x_cm"],
            "x_left_cm": mc["x_left_cm"],
            "x_right_cm": mc["x_right_cm"],
            "volume_cm3": mc["volume_cm3"],
            "mc_temperature_K": mc["temperature_K"],
            "ddmc_temperature_K": ddmc["temperature_K"],
            "temperature_difference_K": ddmc["temperature_K"] - mc["temperature_K"],
            "temperature_relative_difference": safe_relative(
                mc["temperature_K"], ddmc["temperature_K"]
            ),
            "mc_internal_energy_extensive": mc["internal_energy_extensive"],
            "ddmc_internal_energy_extensive": ddmc["internal_energy_extensive"],
            "internal_energy_relative_difference": safe_relative(
                mc["internal_energy_extensive"],
                ddmc["internal_energy_extensive"],
            ),
            "mc_Erad_specific": mc["Erad_specific"],
            "ddmc_Erad_specific": ddmc["Erad_specific"],
            "Erad_relative_difference": safe_relative(
                mc["Erad_specific"], ddmc["Erad_specific"]
            ),
            "ddmc_eligible": ddmc["ddmc_eligible"],
            "ddmc_group_cutoff": ddmc["ddmc_group_cutoff"],
            "ddmc_sigmaDiffusion": ddmc["ddmc_sigmaDiffusion"],
            "ddmc_sigmaGroupExit": ddmc["ddmc_sigmaGroupExit"],
            "ddmc_gamma": ddmc["ddmc_gamma"],
            "ddmc_D": ddmc["ddmc_D"],
            "ddmc_total_leak_rate": ddmc["ddmc_total_leak_rate"],
            "ddmc_internal_leak_rate_sum": ddmc["ddmc_internal_leak_rate_sum"],
            "ddmc_channel_rate_sum": ddmc["ddmc_channel_rate_sum"],
            "ddmc_transport_channel_rate_sum": ddmc["ddmc_transport_channel_rate_sum"],
            "ddmc_boundary_rate_sum": ddmc["ddmc_boundary_rate_sum"],
            "ddmc_mixed_face_count": ddmc["ddmc_mixed_face_count"],
        }
        result.append(row)
    return result


def write_comparison(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        raise RuntimeError("No comparison rows")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Sequence[Dict[str, float]]) -> None:
    worst = max(rows, key=lambda row: abs(row["temperature_relative_difference"]))
    eligible = [row for row in rows if row["ddmc_eligible"] > 0.5]
    first_eligible = min(eligible, key=lambda row: row["x_cm"]) if eligible else None

    print("Paired Densmore exposed-interface comparison")
    print(f"  cells: {len(rows)}")
    print(
        "  worst |DDMC-MC|/MC: "
        f"{100.0 * abs(worst['temperature_relative_difference']):.4f}% "
        f"at x={worst['x_cm']:.12g} cm"
    )
    if first_eligible is None:
        print("  no DDMC-eligible cell was reported")
    else:
        print(
            "  first DDMC cell: "
            f"x={first_eligible['x_cm']:.12g} cm, "
            f"cutoff={first_eligible['ddmc_group_cutoff']:.0f}, "
            f"relative T error="
            f"{100.0 * first_eligible['temperature_relative_difference']:.4f}%"
        )

    regions = [
        ("thin bulk", 0.0, 1.90),
        ("interface shoulder", 1.90, 2.0),
        ("first thick cell", 2.0, 2.0050000001),
        ("near thick", 2.0050000001, 2.10),
        ("thick bulk", 2.10, 2.50),
        ("full domain", 0.0, 3.0000000001),
    ]
    print("  region summaries (signed mean, mean absolute, volume-weighted absolute):")
    for name, x_min, x_max in regions:
        count, signed_mean, mean_abs, volume_weighted = region_summary(
            rows, x_min, x_max
        )
        print(
            f"    {name:20s} n={count:3d} "
            f"signed={100.0 * signed_mean:9.4f}% "
            f"mean_abs={100.0 * mean_abs:9.4f}% "
            f"vol_abs={100.0 * volume_weighted:9.4f}%"
        )


def parse_args() -> argparse.Namespace:
    cases = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mc",
        type=Path,
        default=cases
        / "desmore2012_interface_mc"
        / "desmore2012_interface_mc_cells.tsv",
    )
    parser.add_argument(
        "--ddmc",
        type=Path,
        default=cases
        / "desmore2012_interface_ddmc"
        / "desmore2012_interface_ddmc_cells.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=cases / "densmore2012_interface_comparison.tsv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mc_rows = read_tsv(args.mc)
    ddmc_rows = read_tsv(args.ddmc)
    rows = compare(mc_rows, ddmc_rows)
    write_comparison(args.output, rows)
    print_summary(rows)
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
