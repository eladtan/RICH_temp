#!/usr/bin/env python3
"""Aggregate the per-rank Densmore IMC/DDMC interface event ledger."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple


NumericRow = Dict[str, float]


def read_rank_rows(paths: Iterable[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(paths):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise RuntimeError(f"{path} has no header")
            rows.extend(reader)
    return rows


def as_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {key!r} in row {row}") from exc


def direction_sign(row: Dict[str, str]) -> float:
    source = as_float(row, "source_generator_x_cm")
    target = as_float(row, "target_generator_x_cm")
    if not (math.isfinite(source) and math.isfinite(target)):
        return 0.0
    return 1.0 if target > source else -1.0


def interface_rows(
    rows: Iterable[Dict[str, str]], face_x: float, tolerance: float
) -> List[Dict[str, str]]:
    return [
        row
        for row in rows
        if abs(as_float(row, "face_x_cm") - face_x) <= tolerance
    ]


def summarize_events(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: DefaultDict[Tuple[int, str, int], Dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "signed_energy": 0.0,
            "absolute_energy": 0.0,
            "mu_sum": 0.0,
            "mu_count": 0.0,
            "probability_sum": 0.0,
            "probability_count": 0.0,
        }
    )
    for row in rows:
        key = (
            int(as_float(row, "step")),
            row["kind"],
            int(as_float(row, "group")),
        )
        out = grouped[key]
        out["count"] += as_float(row, "count")
        out["signed_energy"] += as_float(row, "signed_energy")
        out["absolute_energy"] += as_float(row, "absolute_energy")
        out["mu_sum"] += as_float(row, "mu_sum")
        out["mu_count"] += as_float(row, "mu_count")
        out["probability_sum"] += as_float(
            row, "admission_probability_sum"
        )
        out["probability_count"] += as_float(
            row, "admission_probability_count"
        )

    result: List[Dict[str, object]] = []
    for (step, kind, group), values in sorted(grouped.items()):
        mu_count = values["mu_count"]
        probability_count = values["probability_count"]
        result.append(
            {
                "step": step,
                "kind": kind,
                "group": group,
                "count": int(values["count"]),
                "signed_energy": values["signed_energy"],
                "absolute_energy": values["absolute_energy"],
                "mean_mu": values["mu_sum"] / mu_count
                if mu_count > 0.0
                else math.nan,
                "mean_admission_probability": values["probability_sum"]
                / probability_count
                if probability_count > 0.0
                else math.nan,
            }
        )
    return result


def summarize_admission_balance(
    rows: List[Dict[str, str]],
) -> List[Dict[str, object]]:
    grouped: DefaultDict[Tuple[int, str, int], Dict[str, float]] = defaultdict(
        lambda: {
            "incident_count": 0.0,
            "admitted_count": 0.0,
            "reflected_count": 0.0,
            "incident_energy": 0.0,
            "admitted_energy": 0.0,
            "reflected_energy": 0.0,
            "probability_sum": 0.0,
            "probability_count": 0.0,
        }
    )
    for row in rows:
        kind = row["kind"]
        if kind not in {"imc_incident", "imc_admitted", "imc_reflected"}:
            continue
        source = as_float(row, "source_generator_x_cm")
        target = as_float(row, "target_generator_x_cm")
        direction = "left_to_right" if target > source else "right_to_left"
        key = (
            int(as_float(row, "step")),
            direction,
            int(as_float(row, "group")),
        )
        out = grouped[key]
        count = as_float(row, "count")
        energy = as_float(row, "absolute_energy")
        if kind == "imc_incident":
            out["incident_count"] += count
            out["incident_energy"] += energy
            out["probability_sum"] += as_float(
                row, "admission_probability_sum"
            )
            out["probability_count"] += as_float(
                row, "admission_probability_count"
            )
        elif kind == "imc_admitted":
            out["admitted_count"] += count
            out["admitted_energy"] += energy
        else:
            out["reflected_count"] += count
            out["reflected_energy"] += energy

    result: List[Dict[str, object]] = []
    for (step, direction, group), values in sorted(grouped.items()):
        incident_count = values["incident_count"]
        incident_energy = values["incident_energy"]
        probability_count = values["probability_count"]
        result.append(
            {
                "step": step,
                "direction": direction,
                "group": group,
                "incident_count": int(incident_count),
                "admitted_count": int(values["admitted_count"]),
                "reflected_count": int(values["reflected_count"]),
                "count_closure": values["admitted_count"]
                + values["reflected_count"]
                - incident_count,
                "observed_count_admission_fraction": values["admitted_count"]
                / incident_count
                if incident_count > 0.0
                else math.nan,
                "prescribed_mean_admission_probability": values[
                    "probability_sum"
                ]
                / probability_count
                if probability_count > 0.0
                else math.nan,
                "incident_absolute_energy": incident_energy,
                "admitted_absolute_energy": values["admitted_energy"],
                "reflected_absolute_energy": values["reflected_energy"],
                "energy_closure": values["admitted_energy"]
                + values["reflected_energy"]
                - incident_energy,
                "observed_energy_admission_fraction": values[
                    "admitted_energy"
                ]
                / incident_energy
                if incident_energy > 0.0
                else math.nan,
            }
        )
    return result


def summarize_net_crossing(rows: List[Dict[str, str]]) -> List[Dict[str, float]]:
    crossing_kinds = {
        "imc_frequency_reject",
        "imc_admitted",
        "imc_bypass",
        "ddmc_to_ddmc",
        "ddmc_to_imc",
    }
    by_step: DefaultDict[int, Dict[str, float]] = defaultdict(
        lambda: {
            "left_to_right_energy": 0.0,
            "right_to_left_energy": 0.0,
            "net_left_to_right_energy": 0.0,
            "imc_admitted_energy": 0.0,
            "imc_transport_energy": 0.0,
            "ddmc_common_energy": 0.0,
            "ddmc_transport_energy": 0.0,
        }
    )
    for row in rows:
        kind = row["kind"]
        if kind not in crossing_kinds:
            continue
        step = int(as_float(row, "step"))
        energy = as_float(row, "signed_energy")
        sign = direction_sign(row)
        out = by_step[step]
        if sign > 0.0:
            out["left_to_right_energy"] += energy
        elif sign < 0.0:
            out["right_to_left_energy"] += energy
        out["net_left_to_right_energy"] += sign * energy
        if kind == "imc_admitted":
            out["imc_admitted_energy"] += sign * energy
        elif kind in {"imc_frequency_reject", "imc_bypass"}:
            out["imc_transport_energy"] += sign * energy
        elif kind == "ddmc_to_ddmc":
            out["ddmc_common_energy"] += sign * energy
        elif kind == "ddmc_to_imc":
            out["ddmc_transport_energy"] += sign * energy

    return [dict(step=step, **values) for step, values in sorted(by_step.items())]


def write_tsv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def print_final_face(face_rows: List[Dict[str, str]]) -> None:
    if not face_rows:
        print("  no static face rows found at requested face")
        return
    final_step = max(int(as_float(row, "step")) for row in face_rows)
    print(f"  final-step directed coefficients at step {final_step}:")
    selected = [
        row for row in face_rows if int(as_float(row, "step")) == final_step
    ]
    for row in sorted(selected, key=lambda item: as_float(item, "source_generator_x_cm")):
        print(
            "    "
            f"{as_float(row, 'source_generator_x_cm'):.9g} -> "
            f"{as_float(row, 'target_generator_x_cm'):.9g}: "
            f"cutoff {int(as_float(row, 'source_cutoff'))} -> "
            f"{int(as_float(row, 'target_cutoff'))}, "
            f"di={as_float(row, 'source_distance_to_face'):.6g}, "
            f"internal={as_float(row, 'internal_rate'):.6e}, "
            f"boundary={as_float(row, 'boundary_rate'):.6e}, "
            f"fraction={as_float(row, 'ddmc_fraction'):.6g}, "
            f"ddmc={as_float(row, 'ddmc_rate'):.6e}, "
            f"transport={as_float(row, 'transport_rate'):.6e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--face-x", type=float, default=2.0)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    event_paths = list(
        args.case_dir.glob(f"{args.prefix}_rank*_ddmc_interface_events.tsv")
    )
    face_paths = list(
        args.case_dir.glob(f"{args.prefix}_rank*_ddmc_face_history.tsv")
    )
    if not event_paths:
        raise RuntimeError(f"No event ledgers found under {args.case_dir}")
    if not face_paths:
        raise RuntimeError(f"No face histories found under {args.case_dir}")

    events = interface_rows(
        read_rank_rows(event_paths), args.face_x, args.tolerance
    )
    faces = interface_rows(
        read_rank_rows(face_paths), args.face_x, args.tolerance
    )
    event_summary = summarize_events(events)
    admission_summary = summarize_admission_balance(events)
    net_summary = summarize_net_crossing(events)
    event_output = args.case_dir / f"{args.prefix}_interface_event_summary.tsv"
    admission_output = (
        args.case_dir / f"{args.prefix}_interface_admission_balance.tsv"
    )
    net_output = args.case_dir / f"{args.prefix}_interface_net_energy.tsv"
    write_tsv(event_output, event_summary)
    write_tsv(admission_output, admission_summary)
    write_tsv(net_output, net_summary)

    print(f"Densmore interface event analysis: {args.prefix}")
    print(f"  rank event files: {len(event_paths)}")
    print(f"  event rows at x={args.face_x}: {len(events)}")
    print_final_face(faces)
    if net_summary:
        final = net_summary[-1]
        print(
            f"  final step net L->R energy={final['net_left_to_right_energy']:.8e} "
            f"(DDMC common={final['ddmc_common_energy']:.8e}, "
            f"DDMC->IMC={final['ddmc_transport_energy']:.8e}, "
            f"IMC admitted={final['imc_admitted_energy']:.8e}, "
            f"IMC transport={final['imc_transport_energy']:.8e})"
        )
    print(f"  wrote {event_output}")
    print(f"  wrote {admission_output}")
    print(f"  wrote {net_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
