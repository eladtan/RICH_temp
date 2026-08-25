#!/usr/bin/env python3
"""Audit AutoPartial material/radiation energy and Dirichlet defect records."""

from __future__ import annotations

import argparse
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ENERGY_RE = re.compile(
    r"Einit = (?P<initial>[-+0-9.eE]+), Efinal = "
    r"(?P<final>[-+0-9.eE]+) min_T_E_added = "
    r"(?P<floor>[-+0-9.eE]+) d_Ek (?P<kinetic>[-+0-9.eE]+)"
)
RELATIVE_RE = re.compile(
    r"\|Einit-Efinal\|/Einit = (?P<relative>[-+0-9.eE]+)"
)
KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def as_float(value: str) -> int | float | str:
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def finite_number(record: dict[str, Any], key: str) -> float | None:
    value = record.get(key)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def first_finite(
    record: dict[str, Any], *keys: str
) -> float | None:
    for key in keys:
        value = finite_number(record, key)
        if value is not None:
            return value
    return None


def close(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right), np.finfo(float).tiny)
    return abs(left - right) <= 256 * np.finfo(float).eps * scale


def parse_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False, "path": str(path)}
    lines = path.read_text(errors="replace").splitlines()
    candidate_energy: list[dict[str, Any]] = []
    defects: list[dict[str, Any]] = []
    first_rejection: dict[str, Any] | None = None
    positivity_repairs = 0
    passive_repairs = 0
    pending_energy: dict[str, Any] | None = None
    for line_number, line in enumerate(lines, 1):
        match = ENERGY_RE.search(line)
        if match:
            pending_energy = {
                "line": line_number,
                **{key: float(value) for key, value in match.groupdict().items()},
            }
            pending_energy["delta"] = (
                pending_energy["final"] - pending_energy["initial"]
            )
            candidate_energy.append(pending_energy)
            continue
        match = RELATIVE_RE.search(line)
        if match and pending_energy is not None:
            pending_energy["reported_relative_absolute_delta"] = float(
                match.group("relative")
            )
            continue
        if line.startswith("INDIVIDUAL_RADIATION_DEFECT "):
            raw_fields = {
                key: value.rstrip(",")
                for key, value in KEY_VALUE_RE.findall(line)
            }
            record = {
                "line": line_number,
                "_raw_fields": raw_fields,
                **{
                    key: as_float(value)
                    for key, value in raw_fields.items()
                },
            }
            defects.append(record)
            if pending_energy is not None:
                pending_energy["defect_status"] = record.get("status")
                pending_energy["defect_line"] = line_number
                pending_energy = None
            continue
        if line.startswith("INDIVIDUAL_RADIATION_REJECTION "):
            if first_rejection is None:
                first_rejection = {
                    "line": line_number,
                    "text": line,
                    **{
                        key: as_float(value.rstrip(","))
                        for key, value in KEY_VALUE_RE.findall(line)
                    },
                }
            continue
        if line.startswith("MG_SPECTRAL_POSITIVITY_REPAIR"):
            positivity_repairs += 1
        if line.startswith("MG_PASSIVE_ROUNDOFF_REPAIR"):
            passive_repairs += 1

    violations: list[str] = []
    for index, defect in enumerate(defects):
        prefix = f"defect[{index}] line {defect['line']}"
        numeric_values = [
            value for value in defect.values()
            if isinstance(value, (int, float))
        ]
        if not all(math.isfinite(float(value)) for value in numeric_values):
            violations.append(prefix + ": nonfinite value")
        signed = first_finite(defect, "signed", "signed_extent")
        absolute = first_finite(defect, "absolute", "absolute_extent")
        withdrawal = first_finite(
            defect, "withdrawal", "passive_withdrawal_extent"
        )
        deposit = first_finite(
            defect, "deposit", "passive_deposit_extent"
        )
        scale = first_finite(defect, "scale", "normalization_scale")
        event_fraction = first_finite(
            defect, "event_abs_fraction", "event_absolute_fraction"
        )
        local_fraction = first_finite(
            defect, "max_local_fraction", "maximum_local_fraction"
        )
        projected_signed = finite_number(
            defect, "projected_cumulative_signed_fraction"
        )
        projected_absolute = finite_number(
            defect, "projected_cumulative_absolute_fraction"
        )
        if signed is not None and absolute is not None:
            if absolute + 256 * np.finfo(float).eps * max(absolute, 1.0) < abs(signed):
                violations.append(prefix + ": absolute defect below |signed|")
        if None not in (signed, withdrawal, deposit):
            if not close(signed, withdrawal - deposit):
                violations.append(prefix + ": signed != withdrawal-deposit")
        if None not in (absolute, withdrawal, deposit):
            if not close(absolute, withdrawal + deposit):
                violations.append(prefix + ": absolute != withdrawal+deposit")
        if None not in (absolute, scale, event_fraction) and scale > 0:
            if not close(event_fraction, absolute / scale):
                violations.append(prefix + ": event fraction mismatch")
        if defect.get("status") == "accepted":
            if local_fraction is not None and local_fraction > 1e-2:
                violations.append(prefix + ": accepted above local hard limit")
            if projected_signed is not None and abs(projected_signed) > 1e-4:
                violations.append(prefix + ": accepted above cumulative signed limit")
            if projected_absolute is not None and projected_absolute > 1e-3:
                violations.append(prefix + ": accepted above cumulative absolute limit")

    running_signed = Decimal(0)
    running_absolute = Decimal(0)
    running_accepted = 0
    recurrence_checked = 0
    decimal_tolerance = Decimal("5e-18")
    for index, defect in enumerate(defects):
        raw_fields = defect.get("_raw_fields", {})
        if defect.get("status") != "accepted":
            continue
        running_accepted += 1
        prefix = f"defect[{index}] line {defect['line']}"
        try:
            signed_text = raw_fields.get("signed", raw_fields.get("signed_extent"))
            absolute_text = raw_fields.get(
                "absolute", raw_fields.get("absolute_extent")
            )
            cumulative_signed_text = raw_fields["cumulative_signed_extent"]
            cumulative_absolute_text = raw_fields["cumulative_absolute_extent"]
            accepted_text = raw_fields["accepted_candidates"]
            if signed_text is None or absolute_text is None:
                raise KeyError("signed/absolute")
            running_signed += Decimal(signed_text)
            running_absolute += Decimal(absolute_text)
            reported_signed = Decimal(cumulative_signed_text)
            reported_absolute = Decimal(cumulative_absolute_text)
            reported_accepted = int(accepted_text)
        except (InvalidOperation, KeyError, TypeError, ValueError) as error:
            violations.append(prefix + f": incomplete accepted ledger ({error})")
            continue
        signed_scale = max(abs(running_signed), abs(reported_signed), Decimal(1))
        absolute_scale = max(
            abs(running_absolute), abs(reported_absolute), Decimal(1)
        )
        if abs(running_signed - reported_signed) > decimal_tolerance * signed_scale:
            violations.append(prefix + ": cumulative signed recurrence mismatch")
        if (
            abs(running_absolute - reported_absolute)
            > decimal_tolerance * absolute_scale
        ):
            violations.append(prefix + ": cumulative absolute recurrence mismatch")
        if reported_accepted != running_accepted:
            violations.append(prefix + ": accepted candidate count mismatch")
        recurrence_checked += 1

    for defect in defects:
        defect.pop("_raw_fields", None)
    relative_values = [
        item["reported_relative_absolute_delta"]
        for item in candidate_energy
        if "reported_relative_absolute_delta" in item
    ]
    return {
        "present": True,
        "path": str(path),
        "candidate_energy_records": candidate_energy,
        "candidate_count": len(candidate_energy),
        "maximum_candidate_relative_absolute_delta": (
            max(relative_values) if relative_values else None
        ),
        "defect_records": defects,
        "accepted_defect_records": sum(
            item.get("status") == "accepted" for item in defects
        ),
        "rejected_defect_records": sum(
            item.get("status") == "rejected" for item in defects
        ),
        "first_rejection": first_rejection,
        "spectral_positivity_repair_records": positivity_repairs,
        "passive_roundoff_repair_records": passive_repairs,
        "defect_identity_violations": violations,
        "accepted_ledger_recurrence": {
            "checked_records": recurrence_checked,
            "computed_cumulative_signed_extent": str(running_signed),
            "computed_cumulative_absolute_extent": str(running_absolute),
            "computed_accepted_candidates": running_accepted,
            "relative_tolerance": str(decimal_tolerance),
        },
    }


def long_sum(values: np.ndarray) -> np.longdouble:
    return np.sum(values.astype(np.longdouble), dtype=np.longdouble)


def audit_snapshot(path: Path, expected_pieces: int) -> dict[str, Any]:
    if not path.is_dir():
        return {"present": False, "path": str(path)}
    pieces = sorted(
        (item for item in path.glob("*.h5") if item.stem.isdigit()),
        key=lambda item: int(item.stem),
    )
    if len(pieces) != expected_pieces:
        raise RuntimeError(
            f"{path}: expected {expected_pieces} HDF5 pieces, found {len(pieces)}"
        )
    totals = {
        "mass": np.longdouble(0),
        "hydro_total_energy": np.longdouble(0),
        "material_internal_energy": np.longdouble(0),
        "kinetic_from_energy_difference": np.longdouble(0),
        "kinetic_from_momentum": np.longdouble(0),
        "radiation_group_energy": np.longdouble(0),
        "radiation_aggregate_Erad": np.longdouble(0),
    }
    ids: list[np.ndarray] = []
    cells = 0
    group_totals: np.ndarray | None = None
    minimum_mass = math.inf
    minimum_group = math.inf
    minimum_internal = math.inf
    ledger_reference_raw: dict[str, tuple[str, tuple[int, ...], bytes]] | None = None
    ledger_reference_report: dict[str, str] = {}
    ledger_metadata_pieces = 0
    scheduler_metadata_reference: tuple[int, int] | None = None
    scheduler_metadata_report: dict[str, int] = {}
    scheduler_metadata_pieces = 0
    for piece in pieces:
        with h5py.File(piece, "r") as handle:
            extensives = handle["extensives"][...]
            cell_data = handle["cells"][...]
            scheduler_path = "/individual_time_steps"
            if f"{scheduler_path}/version" in handle:
                checkpoint_version = int(
                    np.asarray(handle[f"{scheduler_path}/version"][()]).item()
                )
                force_all_active_latched = 0
                latch_path = f"{scheduler_path}/force_all_active_latched"
                if checkpoint_version >= 7:
                    if latch_path not in handle:
                        raise RuntimeError(
                            f"{piece}: checkpoint v{checkpoint_version} misses "
                            "force_all_active_latched"
                        )
                    force_all_active_latched = int(
                        np.asarray(handle[latch_path][()]).item()
                    )
                    if force_all_active_latched not in (0, 1):
                        raise RuntimeError(
                            f"{piece}: invalid force_all_active_latched="
                            f"{force_all_active_latched}"
                        )
                scheduler_metadata = (
                    checkpoint_version,
                    force_all_active_latched,
                )
                if scheduler_metadata_reference is None:
                    scheduler_metadata_reference = scheduler_metadata
                    scheduler_metadata_report = {
                        "version": checkpoint_version,
                        "force_all_active_latched": force_all_active_latched,
                    }
                elif scheduler_metadata != scheduler_metadata_reference:
                    raise RuntimeError(
                        f"{piece}: individual scheduler checkpoint metadata mismatch"
                    )
                scheduler_metadata_pieces += 1
            defect_path = "/individual_time_steps/radiation_defect"
            if defect_path in handle:
                defect_group = handle[defect_path]
                current_raw: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
                current_report: dict[str, str] = {}
                for name, dataset in defect_group.items():
                    value = np.asarray(dataset[()])
                    current_raw[name] = (
                        value.dtype.str,
                        value.shape,
                        value.tobytes(),
                    )
                    current_report[name] = (
                        str(value.item()) if value.shape == () else str(value.tolist())
                    )
                if ledger_reference_raw is None:
                    ledger_reference_raw = current_raw
                    ledger_reference_report = current_report
                elif current_raw != ledger_reference_raw:
                    raise RuntimeError(
                        f"{piece}: radiation defect checkpoint metadata mismatch"
                    )
                ledger_metadata_pieces += 1
        if len(extensives) != len(cell_data):
            raise RuntimeError(f"{piece}: cell/extensive count mismatch")
        for field in ("mass", "energy", "internal_energy", "Erad", "Eg"):
            if not np.all(np.isfinite(extensives[field])):
                raise RuntimeError(f"{piece}: nonfinite extensives.{field}")
        if np.any(extensives["mass"] <= 0):
            raise RuntimeError(f"{piece}: nonpositive mass")
        if np.any(extensives["internal_energy"] < 0):
            raise RuntimeError(f"{piece}: negative internal energy")
        if np.any(extensives["Eg"] < 0):
            raise RuntimeError(f"{piece}: negative group radiation energy")
        momentum = extensives["momentum"]
        momentum_x = momentum["x"].astype(np.longdouble)
        momentum_y = momentum["y"].astype(np.longdouble)
        momentum_z = momentum["z"].astype(np.longdouble)
        mass_long = extensives["mass"].astype(np.longdouble)
        kinetic_momentum = (
            momentum_x * momentum_x
            + momentum_y * momentum_y
            + momentum_z * momentum_z
        ) / (2 * mass_long)
        totals["mass"] += long_sum(extensives["mass"])
        totals["hydro_total_energy"] += long_sum(extensives["energy"])
        totals["material_internal_energy"] += long_sum(
            extensives["internal_energy"]
        )
        totals["kinetic_from_energy_difference"] += np.sum(
            extensives["energy"].astype(np.longdouble)
            - extensives["internal_energy"].astype(np.longdouble),
            dtype=np.longdouble,
        )
        totals["kinetic_from_momentum"] += np.sum(
            kinetic_momentum, dtype=np.longdouble
        )
        totals["radiation_group_energy"] += long_sum(
            extensives["Eg"].reshape(-1)
        )
        piece_group_totals = np.sum(
            extensives["Eg"].astype(np.longdouble),
            axis=0,
            dtype=np.longdouble,
        )
        if group_totals is None:
            group_totals = np.zeros_like(piece_group_totals, dtype=np.longdouble)
        if piece_group_totals.shape != group_totals.shape:
            raise RuntimeError(f"{piece}: inconsistent radiation group count")
        group_totals += piece_group_totals
        totals["radiation_aggregate_Erad"] += long_sum(extensives["Erad"])
        minimum_mass = min(minimum_mass, float(np.min(extensives["mass"])))
        minimum_group = min(minimum_group, float(np.min(extensives["Eg"])))
        minimum_internal = min(
            minimum_internal, float(np.min(extensives["internal_energy"]))
        )
        ids.append(np.asarray(cell_data["ID"], dtype=np.uint64))
        cells += len(extensives)
    all_ids = np.concatenate(ids)
    unique_ids = int(np.unique(all_ids).size)
    if unique_ids != cells:
        raise RuntimeError(f"{path}: duplicate stable IDs ({unique_ids}/{cells})")
    if ledger_metadata_pieces not in (0, len(pieces)):
        raise RuntimeError(
            f"{path}: defect metadata present in only "
            f"{ledger_metadata_pieces}/{len(pieces)} pieces"
        )
    if scheduler_metadata_pieces not in (0, len(pieces)):
        raise RuntimeError(
            f"{path}: individual scheduler metadata present in only "
            f"{scheduler_metadata_pieces}/{len(pieces)} pieces"
        )
    nongrav = totals["hydro_total_energy"] + totals["radiation_group_energy"]
    erad_delta = (
        totals["radiation_aggregate_Erad"] - totals["radiation_group_energy"]
    )
    result = {
        "present": True,
        "path": str(path),
        "pieces": len(pieces),
        "cells": cells,
        "unique_ids": unique_ids,
        **{key: str(value) for key, value in totals.items()},
        "nongrav_material_plus_kinetic_plus_radiation": str(nongrav),
        "aggregate_Erad_minus_sum_Eg": str(erad_delta),
        "aggregate_Erad_relative_mismatch": str(
            erad_delta / totals["radiation_group_energy"]
            if totals["radiation_group_energy"] != 0 else np.longdouble(0)
        ),
        "kinetic_crosscheck_difference": str(
            totals["kinetic_from_energy_difference"]
            - totals["kinetic_from_momentum"]
        ),
        "radiation_energy_by_group": (
            [str(value) for value in group_totals]
            if group_totals is not None else []
        ),
        "minimum_mass": minimum_mass,
        "minimum_group_extent": minimum_group,
        "minimum_internal_energy": minimum_internal,
        "radiation_defect_metadata_pieces": ledger_metadata_pieces,
        "radiation_defect_metadata_consistent": True,
        "radiation_defect_metadata": ledger_reference_report,
        "individual_scheduler_metadata_pieces": scheduler_metadata_pieces,
        "individual_scheduler_metadata_consistent": True,
        "individual_scheduler_metadata": scheduler_metadata_report,
    }
    return result


def read_counters(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False, "path": str(path)}
    result: dict[str, Any] = {"present": True, "path": str(path)}
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            result[fields[0]] = as_float(fields[1])
    return result


def relative_change(initial: str, final: str) -> str:
    left = np.longdouble(initial)
    right = np.longdouble(final)
    if left == 0:
        return "0"
    return str((right - left) / np.abs(left))


def render_markdown(report: dict[str, Any]) -> str:
    log = report["log"]
    initial = report["initial_snapshot"]
    final = report["final_snapshot"]
    lines = [
        "# AutoPartial energy-conservation audit",
        "",
        "This audit distinguishes material internal, kinetic, and 16-group "
        "radiation energy. Gravitational potential energy is not present in "
        "the benchmark history or HDF5 extensive records, so nongravitational "
        "energy drift is not claimed to be full total-energy drift.",
        "",
        f"- First radiation halving: {json.dumps(log.get('first_rejection'))}",
        f"- Candidate energy records: {log.get('candidate_count', 0)}",
        f"- Maximum logged candidate |dE|/E: "
        f"{log.get('maximum_candidate_relative_absolute_delta')}",
        f"- Accepted/rejected Dirichlet defect records: "
        f"{log.get('accepted_defect_records', 0)}/"
        f"{log.get('rejected_defect_records', 0)}",
        f"- Spectral positivity/passive roundoff repair records: "
        f"{log.get('spectral_positivity_repair_records', 0)}/"
        f"{log.get('passive_roundoff_repair_records', 0)}",
        f"- Defect identity/gate violations: "
        f"{len(log.get('defect_identity_violations', []))}",
        f"- Accepted-ledger recurrence: "
        f"{json.dumps(log.get('accepted_ledger_recurrence'))}",
        "",
        "## HDF5 state",
        "",
    ]
    for label, snapshot in (("Initial", initial), ("Final", final)):
        if not snapshot.get("present"):
            lines.append(f"- {label}: unavailable ({snapshot.get('path')})")
            continue
        lines.extend([
            f"- {label}: {snapshot['pieces']} pieces, {snapshot['cells']} cells",
            f"  - internal: {snapshot['material_internal_energy']}",
            f"  - kinetic (energy-internal): "
            f"{snapshot['kinetic_from_energy_difference']}",
            f"  - kinetic (p^2/2m): {snapshot['kinetic_from_momentum']}",
            f"  - radiation sum(Eg): {snapshot['radiation_group_energy']}",
            f"  - nongrav sum(energy)+sum(Eg): "
            f"{snapshot['nongrav_material_plus_kinetic_plus_radiation']}",
            f"  - sum(Erad)-sum(Eg): "
            f"{snapshot['aggregate_Erad_minus_sum_Eg']}",
            f"  - kinetic closure difference: "
            f"{snapshot['kinetic_crosscheck_difference']}",
            f"  - minimum mass/internal/group extent: "
            f"{snapshot['minimum_mass']} / "
            f"{snapshot['minimum_internal_energy']} / "
            f"{snapshot['minimum_group_extent']}",
            f"  - per-group radiation: "
            f"{json.dumps(snapshot['radiation_energy_by_group'])}",
            f"  - defect metadata pieces/consistent: "
            f"{snapshot['radiation_defect_metadata_pieces']}/"
            f"{snapshot['radiation_defect_metadata_consistent']}",
            f"  - scheduler metadata pieces/consistent: "
            f"{snapshot['individual_scheduler_metadata_pieces']}/"
            f"{snapshot['individual_scheduler_metadata_consistent']}",
            f"  - scheduler checkpoint version/latch: "
            f"{snapshot['individual_scheduler_metadata'].get('version', 'absent')} / "
            f"{snapshot['individual_scheduler_metadata'].get('force_all_active_latched', 'absent')}",
        ])
    if initial.get("present") and final.get("present"):
        lines.extend([
            "",
            "## Endpoint relative changes",
            "",
            "- internal: " + relative_change(
                initial["material_internal_energy"],
                final["material_internal_energy"],
            ),
            "- kinetic: " + relative_change(
                initial["kinetic_from_energy_difference"],
                final["kinetic_from_energy_difference"],
            ),
            "- radiation: " + relative_change(
                initial["radiation_group_energy"],
                final["radiation_group_energy"],
            ),
            "- nongrav: " + relative_change(
                initial["nongrav_material_plus_kinetic_plus_radiation"],
                final["nongrav_material_plus_kinetic_plus_radiation"],
            ),
        ])
    elif log.get("first_rejection") is not None:
        lines.extend([
            "",
            "No post-halving HDF5 state exists because the watcher stopped the "
            "job. The report therefore proves the exact initial state and the "
            "logged candidate/defect ledger only; it does not claim endpoint "
            "conservation.",
        ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--lane", default="partial")
    parser.add_argument("--expected-pieces", type=int, default=128)
    args = parser.parse_args()
    lane_root = args.run_root / args.lane
    report = {
        "run_root": str(args.run_root),
        "lane": args.lane,
        "log": parse_log(lane_root / "run.log"),
        "counters": read_counters(lane_root / "counters.txt"),
        "initial_snapshot": audit_snapshot(
            lane_root / "initial_state", args.expected_pieces
        ),
        "final_snapshot": audit_snapshot(
            lane_root / "final_state", args.expected_pieces
        ),
        "scope_warning": (
            "Gravitational potential energy is unavailable; nongrav energy "
            "is not full total energy for this self-gravitating benchmark."
        ),
    }
    json_path = args.run_root / "energy_conservation_audit.json"
    markdown_path = args.run_root / "energy_conservation_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(report))
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
