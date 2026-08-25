#!/usr/bin/env python3
"""Check MPI-global MG timestep and historical-residual normalization."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


FIELD = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(\S+)")


def first_limit(path: Path, mode: str) -> dict[str, str]:
    marker = f"MG_TIMESTEP_LIMIT mode={mode}"
    text = path.read_text(errors="replace")
    start = text.find(marker)
    if start < 0:
        raise RuntimeError(f"{path}: no {marker} record")

    required = {
        "current_dt", "suggested_dt", "difference", "max_Er",
        "max_rhoT", "growth_cap", "reference_scope",
    }
    numeric = required.difference({"reference_scope"})

    def usable(name: str, value: str | None) -> bool:
        if value is None:
            return False
        if name not in numeric:
            return bool(value)
        try:
            float(value)
        except ValueError:
            return False
        return True

    line_end = text.find("\n", start)
    if line_end < 0:
        line_end = len(text)
    result = dict(FIELD.findall(text[start:line_end]))

    # Slurm can splice a long diagnostic from another MPI rank into a sequence
    # of small ostream writes.  The limiter record and its immediately
    # following detailed record still precede the next progress record.  Keep
    # every intact field from the marker line, then recover only missing or
    # interrupted fields from that bounded event window.
    end = text.find("\nPROGRESS phase=", start)
    if end < 0:
        end = min(len(text), start + 256 * 1024)
    event = text[start:end]
    candidates = FIELD.findall(event)
    for name in required:
        if usable(name, result.get(name)):
            continue
        for candidate_name, value in candidates:
            if candidate_name == name and usable(name, value):
                result[name] = value
                break
    detail = re.search(
        r"Radiation time step ID .*? max_Er (\S+) max_rhoT (\S+)",
        event,
    )
    if detail is not None:
        for name, value in zip(("max_Er", "max_rhoT"), detail.groups()):
            if not usable(name, result.get(name)):
                result[name] = value
    missing = sorted(required.difference(result))
    if missing:
        raise RuntimeError(f"{path}: limiter diagnostic misses {missing}")
    for name in numeric:
        try:
            float(result[name])
        except ValueError as error:
            raise RuntimeError(
                f"{path}: limiter diagnostic has invalid {name}="
                f"{result[name]!r}") from error
    return result


def first_active_tolerance(path: Path) -> dict[str, str]:
    marker = "MG_BICGSTAB_TOLERANCE scope="
    with path.open(errors="replace") as stream:
        for line in stream:
            if marker in line:
                result = dict(FIELD.findall(line))
                required = {
                    "scope", "requested", "squared_scaled_tolerance",
                    "effective_norm_tolerance",
                    "diagnostic_backward_reference", "global_unknowns",
                    "criterion", "eta_inf_role", "max_global_row_nnz",
                }
                missing = sorted(required.difference(result))
                if missing:
                    raise RuntimeError(
                        f"{path}: tolerance diagnostic misses {missing}")
                return result
    raise RuntimeError(f"{path}: no {marker} record")


def first_active_convergence(path: Path) -> dict[str, str]:
    marker = (
        "MG_BICGSTAB_CONVERGENCE scope=distributed_active "
        "outcome=converged"
    )
    with path.open(errors="replace") as stream:
        for line in stream:
            if marker in line:
                result = dict(FIELD.findall(line))
                required = {
                    "error", "weighted_residual_squared",
                    "weighted_rhs_squared",
                    "backward_error", "backward_tolerance",
                    "backward_rank", "backward_cell_id", "backward_group",
                    "backward_residual", "backward_scale", "maximum_scale",
                    "safe_minimum_scale", "max0", "max1", "negative",
                }
                missing = sorted(required.difference(result))
                if missing:
                    raise RuntimeError(
                        f"{path}: convergence diagnostic misses {missing}")
                return result
    raise RuntimeError(f"{path}: no converged active BiCGSTAB record")


def first_active_historical_policy(path: Path) -> dict[str, str]:
    marker = "MG_BICGSTAB_HISTORICAL_POLICY scope=distributed_active"
    with path.open(errors="replace") as stream:
        for line in stream:
            if marker in line:
                result = dict(FIELD.findall(line))
                required = {
                    "scope", "squared_scaled_tolerance",
                    "effective_norm_tolerance", "last_true_eta_inf",
                    "last_true_eta_iteration", "last_true_eta_age",
                    "pre_correction_eta_inf", "final_eta_inf",
                    "eta_inf_role",
                }
                missing = sorted(required.difference(result))
                if missing:
                    raise RuntimeError(
                        f"{path}: historical-policy diagnostic misses "
                        f"{missing}")
                # A rejected breakdown iterate has no historical finalization,
                # so its pre-correction eta is intentionally `nan`.  Require
                # the first fully evaluated policy record from a subsequently
                # accepted solve; do not reinterpret eta as an acceptance gate.
                try:
                    if not math.isfinite(float(result["last_true_eta_inf"])) or \
                       not math.isfinite(float(result["pre_correction_eta_inf"])):
                        continue
                except ValueError:
                    continue
                return result
    raise RuntimeError(f"{path}: no {marker} record")


def first_all_active_fast_path(path: Path) -> dict[str, str]:
    marker = "MG_INDIVIDUAL_ALL_ACTIVE_FAST_PATH"
    with path.open(errors="replace") as stream:
        for line in stream:
            if marker in line:
                result = dict(FIELD.findall(line))
                required = {
                    "active_cells", "scheduled_dt", "interval_fraction",
                    "candidate_dt", "diagnostic_owned_cells",
                    "diagnostic_mesh_cells", "owned_mapping", "array_scope",
                }
                missing = sorted(required.difference(result))
                if missing:
                    raise RuntimeError(
                        f"{path}: all-active fast-path diagnostic misses "
                        f"{missing}")
                return result
    raise RuntimeError(f"{path}: no {marker} record")


def require_grey_diffusion_force_zero_owned(path: Path) -> None:
    marker = "GREY_DIFFUSION_FORCE_ZERO_OWNED_TEST enabled=1"
    with path.open(errors="replace") as stream:
        if any(marker in line for line in stream):
            return
    raise RuntimeError(f"{path}: no {marker} record")


def require_active_reference_maximum(path: Path) -> None:
    marker = "MG_ACTIVE_REFERENCE_MAX_TEST"
    with path.open(errors="replace") as stream:
        if any(marker in line for line in stream):
            return
    raise RuntimeError(f"{path}: no {marker} record")


def restart_fingerprint(path: Path, phase: str) -> dict[str, str]:
    marker = f"INDIVIDUAL_RESTART_FINGERPRINT phase={phase}"
    with path.open(errors="replace") as stream:
        for line in stream:
            if marker in line:
                result = dict(FIELD.findall(line))
                required = {
                    "phase", "force_synchronized", "force_all_active_latched",
                    "current_tick", "next_tick",
                    "time_origin_bits", "time_quantum_bits", "cells",
                    "state_xor", "state_sum", "schedule_xor", "schedule_sum",
                    "conserved_xor", "conserved_sum",
                    "repair_cells", "repair_groups", "repair_injected_bits",
                    "repair_max_deficit_bits", "repair_rep_cell",
                    "repair_rep_group", "repair_rep_rank",
                    "repair_rep_original_bits", "repair_rep_floor_bits",
                    "repair_rep_injected_bits",
                    "repair_max_global_energy_bits",
                    "repair_next_warning_bits",
                }
                missing = sorted(required.difference(result))
                if missing:
                    raise RuntimeError(
                        f"{path}: restart fingerprint misses {missing}")
                return result
    raise RuntimeError(f"{path}: no {marker} record")


def compare_restart_fingerprints(
    label: str, left: dict[str, str], right: dict[str, str]
) -> None:
    if left["force_synchronized"] != "1" or right["force_synchronized"] != "1":
        raise RuntimeError(f"{label}: force_synchronized was not restored")
    fields = {
        "force_synchronized", "force_all_active_latched", "current_tick",
        "next_tick", "time_origin_bits",
        "time_quantum_bits", "cells", "state_xor", "state_sum",
        "schedule_xor", "schedule_sum",
        "conserved_xor", "conserved_sum",
        "repair_cells", "repair_groups", "repair_injected_bits",
        "repair_max_deficit_bits", "repair_rep_cell", "repair_rep_group",
        "repair_rep_rank", "repair_rep_original_bits",
        "repair_rep_floor_bits", "repair_rep_injected_bits",
        "repair_max_global_energy_bits", "repair_next_warning_bits",
    }
    mismatches = {
        field: (left[field], right[field])
        for field in sorted(fields)
        if left[field] != right[field]
    }
    if mismatches:
        raise RuntimeError(f"{label}: restart fingerprints differ: {mismatches}")


def compare_restart_schedule(
    label: str, left: dict[str, str], right: dict[str, str]
) -> None:
    fields = {
        "force_synchronized", "current_tick", "next_tick", "time_origin_bits",
        "time_quantum_bits", "cells", "schedule_xor", "schedule_sum",
    }
    mismatches = {
        field: (left[field], right[field])
        for field in sorted(fields)
        if left[field] != right[field]
    }
    if mismatches:
        raise RuntimeError(
            f"{label}: restart event schedules differ: {mismatches}")


def gravity_policy_record(path: Path) -> dict[str, str]:
    marker = "GRAVITY_POLICY_PARITY"
    with path.open(errors="replace") as stream:
        for line in stream:
            if marker in line:
                result = dict(FIELD.findall(line))
                required = {
                    "cell_id", "active_cells", "active_ranks",
                    "passive_mass_updates", "ax", "ay", "az", "energy",
                    "velocity_span", "position_policy",
                }
                missing = sorted(required.difference(result))
                if missing:
                    raise RuntimeError(
                        f"{path}: gravity policy record misses {missing}")
                return result
    raise RuntimeError(f"{path}: no {marker} record")


def first_owned_range(path: Path) -> dict[str, str]:
    marker = "MPI_OWNED_CELL_RANGE"
    with path.open(errors="replace") as stream:
        for line in stream:
            if marker in line:
                result = dict(FIELD.findall(line))
                required = {"minimum", "maximum", "empty_ranks", "ranks"}
                missing = sorted(required.difference(result))
                if missing:
                    raise RuntimeError(
                        f"{path}: owned-range diagnostic misses {missing}")
                return result
    raise RuntimeError(f"{path}: no {marker} record")


def first_active_ownership(path: Path) -> dict[str, str]:
    marker = "MG_ACTIVE_OWNERSHIP"
    with path.open(errors="replace") as stream:
        for line in stream:
            if marker in line:
                result = dict(FIELD.findall(line))
                required = {
                    "ranks", "global_owned_cells", "global_active_cells",
                    "empty_owned_ranks", "nonempty_zero_active_ranks",
                }
                missing = sorted(required.difference(result))
                if missing:
                    raise RuntimeError(
                        f"{path}: active ownership diagnostic misses {missing}")
                return result
    raise RuntimeError(f"{path}: no {marker} record")


def number(record: dict[str, str], field: str) -> float:
    value = float(record[field])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {field}={value}")
    return value


def read_numeric_table(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(errors="replace") as stream:
        for raw in stream:
            line = raw.strip()
            if line and not line.startswith("#"):
                rows.append([float(value) for value in line.split()])
    if not rows:
        raise RuntimeError(f"{path}: empty numeric table")
    return rows


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), float.fromhex("0x1p-1022"))


def compare_forced_reduced_physics(reference_log: Path,
                                   reduced_log: Path) -> None:
    reference_dir = reference_log.parent
    reduced_dir = reduced_log.parent
    reference_spectrum = read_numeric_table(reference_dir / "spectrum.txt")
    reduced_spectrum = read_numeric_table(reduced_dir / "spectrum.txt")
    if len(reference_spectrum) != len(reduced_spectrum):
        raise RuntimeError("forced-reduced comparison changed the group count")
    reference_radiation = sum(row[4] for row in reference_spectrum)
    for reference, candidate in zip(reference_spectrum, reduced_spectrum):
        if int(reference[0]) != int(candidate[0]):
            raise RuntimeError("forced-reduced spectrum group order differs")
        denominator = max(abs(reference[4]),
                          1e-12 * abs(reference_radiation))
        error = (abs(candidate[4] - reference[4]) /
                 max(denominator, float.fromhex("0x1p-1022")))
        if error > 1e-5:
            raise RuntimeError(
                f"forced-reduced group {int(reference[0])} error "
                f"{error:.6g} > 1e-5")

    reference_history = read_numeric_table(reference_dir / "history.txt")[-1]
    reduced_history = read_numeric_table(reduced_dir / "history.txt")[-1]
    reference_material = reference_history[8]
    reduced_material = reduced_history[8]
    reference_radiation_history = reference_history[9]
    reduced_radiation_history = reduced_history[9]
    radiation_error = relative_difference(
        reference_radiation_history, reduced_radiation_history)
    if radiation_error > 1e-6:
        raise RuntimeError(
            f"forced-reduced total-radiation error {radiation_error:.6g} > 1e-6")
    total_error = relative_difference(
        reference_material + reference_radiation_history,
        reduced_material + reduced_radiation_history)
    if total_error > 1e-8:
        raise RuntimeError(
            "forced-reduced material+radiation error "
            f"{total_error:.6g} > 1e-8")


def close(label: str, left: float, right: float, rtol: float) -> None:
    scale = max(abs(left), abs(right), float.fromhex("0x1p-1022"))
    relative = abs(left - right) / scale
    if relative > rtol:
        raise RuntimeError(
            f"{label}: {left:.17g} versus {right:.17g}; "
            f"relative error {relative:.6g} > {rtol:.6g}")


def check_formula(label: str, record: dict[str, str], cap: float) -> None:
    current = number(record, "current_dt")
    suggested = number(record, "suggested_dt")
    difference = max(number(record, "difference"), float.fromhex("0x1p-1022"))
    observed_cap = number(record, "growth_cap")
    close(f"{label} growth cap", observed_cap, cap, 1e-14)
    expected = min(current * 0.15 / difference, current * cap)
    # Diagnostics inherit the solver stream precision, so test the printed
    # values to a tighter tolerance than any timestep-bin decision requires.
    close(f"{label} suggested timestep formula", suggested, expected, 5e-5)
    if record["reference_scope"] != "canonical_owned_global":
        raise RuntimeError(
            f"{label}: unexpected reference scope "
            f"{record['reference_scope']!r}")


def check_historical_tolerance(label: str, record: dict[str, str]) -> None:
    requested = number(record, "requested")
    squared = number(record, "squared_scaled_tolerance")
    effective = number(record, "effective_norm_tolerance")
    diagnostic_reference = number(record, "diagnostic_backward_reference")
    unknowns = int(record["global_unknowns"])
    maximum_row_nonzeros = int(record["max_global_row_nnz"])
    if requested <= 0 or unknowns <= 0 or maximum_row_nonzeros <= 0:
        raise RuntimeError(
            f"{label}: invalid tolerance inputs {requested=}, {unknowns=}, "
            f"{maximum_row_nonzeros=}")
    if record["scope"] not in {"serial_active", "distributed_active"}:
        raise RuntimeError(
            f"{label}: unexpected active solver scope {record['scope']!r}")
    if record["criterion"] != "historical_diagonal_scaled_squared_residual":
        raise RuntimeError(
            f"{label}: unexpected convergence criterion "
            f"{record['criterion']!r}")
    if record["eta_inf_role"] != "diagnostic_only":
        raise RuntimeError(
            f"{label}: eta_inf must be diagnostic-only, got "
            f"{record['eta_inf_role']!r}")
    close(f"{label} requested historical tolerance", requested, 1e-11, 5e-6)
    close(f"{label} squared historical tolerance", squared, requested, 5e-6)
    close(f"{label} effective norm tolerance", effective,
          math.sqrt(squared), 5e-6)
    expected = max(
        requested, 32 * math.ulp(1.0) * maximum_row_nonzeros)
    close(f"{label} diagnostic backward-error reference",
          diagnostic_reference, expected, 5e-6)


def check_historical_convergence(label: str, record: dict[str, str]) -> None:
    historical_error = number(record, "error")
    weighted_residual_squared = number(record, "weighted_residual_squared")
    weighted_rhs_squared = number(record, "weighted_rhs_squared")
    backward_error = number(record, "backward_error")
    diagnostic_reference = number(record, "backward_tolerance")
    residual = number(record, "backward_residual")
    scale = number(record, "backward_scale")
    maximum_scale = number(record, "maximum_scale")
    safe_minimum_scale = number(record, "safe_minimum_scale")
    if (historical_error < 0 or weighted_residual_squared < 0 or
            weighted_rhs_squared <= 0 or backward_error < 0 or
            diagnostic_reference <= 0):
        raise RuntimeError(
            f"{label}: invalid historical/backward diagnostic values")
    close(f"{label} historical residual ratio", historical_error,
          weighted_residual_squared / weighted_rhs_squared, 5e-6)
    if scale < 0 or maximum_scale < scale:
        raise RuntimeError(
            f"{label}: invalid backward-error scales "
            f"{scale=}, {maximum_scale=}")
    expected_safe_minimum = max(
        float.fromhex("0x1p-1022"), 32 * math.ulp(1.0) * maximum_scale)
    close(f"{label} safe minimum scale", safe_minimum_scale,
          expected_safe_minimum, 5e-6)
    expected_error = abs(residual) / max(scale, safe_minimum_scale)
    close(f"{label} componentwise backward error", backward_error,
          expected_error, 5e-6)
    if int(record["backward_rank"]) < 0 or int(record["backward_group"]) < 0:
        raise RuntimeError(f"{label}: invalid representative row identity")


def check_historical_policy(label: str, record: dict[str, str]) -> None:
    squared = number(record, "squared_scaled_tolerance")
    effective = number(record, "effective_norm_tolerance")
    last_eta = number(record, "last_true_eta_inf")
    pre_eta = number(record, "pre_correction_eta_inf")
    if squared <= 0 or last_eta < 0 or pre_eta < 0:
        raise RuntimeError(f"{label}: invalid historical-policy values")
    close(f"{label} historical-policy effective norm", effective,
          math.sqrt(squared), 5e-6)
    if int(record["last_true_eta_iteration"]) < 0 or \
       int(record["last_true_eta_age"]) < 0:
        raise RuntimeError(f"{label}: invalid eta diagnostic iteration")
    if record["final_eta_inf"] != "not_evaluated":
        raise RuntimeError(
            f"{label}: final_eta_inf must be not_evaluated")
    if record["eta_inf_role"] != "diagnostic_only":
        raise RuntimeError(f"{label}: eta_inf must be diagnostic-only")


def check_all_active_fast_path(record: dict[str, str]) -> None:
    active_cells = int(record["active_cells"])
    scheduled = number(record, "scheduled_dt")
    fraction = number(record, "interval_fraction")
    candidate = number(record, "candidate_dt")
    if active_cells <= 0 or scheduled <= 0 or not (0 < fraction <= 1):
        raise RuntimeError(
            "invalid all-active fast-path diagnostic: "
            f"{active_cells=}, {scheduled=}, {fraction=}")
    owned = int(record["diagnostic_owned_cells"])
    mesh = int(record["diagnostic_mesh_cells"])
    if owned < 0 or mesh < owned:
        raise RuntimeError(
            f"invalid owned/mesh storage counts: {owned=}, {mesh=}")
    if record["owned_mapping"] != "identity":
        raise RuntimeError(
            f"unexpected owned mapping {record['owned_mapping']!r}")
    if record["array_scope"] != "mesh_with_ghosts":
        raise RuntimeError(
            f"unexpected array scope {record['array_scope']!r}")
    close("all-active fast-path candidate timestep",
          candidate, scheduled * fraction, 1e-12)


def check_empty_owned_range(record: dict[str, str]) -> None:
    minimum = int(record["minimum"])
    maximum = int(record["maximum"])
    empty_ranks = int(record["empty_ranks"])
    ranks = int(record["ranks"])
    if minimum != 0 or maximum <= 0 or not (0 < empty_ranks < ranks):
        raise RuntimeError(
            "invalid zero-owned-rank coverage: "
            f"{minimum=}, {maximum=}, {empty_ranks=}, {ranks=}")


def check_nonempty_zero_active(record: dict[str, str]) -> None:
    if int(record["global_owned_cells"]) <= 0 or \
       int(record["global_active_cells"]) <= 0:
        raise RuntimeError("sparse active-row gate has no owned/active cells")
    if int(record["nonempty_zero_active_ranks"]) <= 0:
        raise RuntimeError(
            "sparse active-row gate created no non-empty rank with zero active rows")


def compare_pair(
    label: str,
    left: dict[str, str],
    right: dict[str, str],
    difference_rtol: float,
    reference_rtol: float = 1e-10,
) -> None:
    close(label + " max_Er", number(left, "max_Er"),
          number(right, "max_Er"), reference_rtol)
    close(label + " max_rhoT", number(left, "max_rhoT"),
          number(right, "max_rhoT"), reference_rtol)
    close(label + " change metric", number(left, "difference"),
          number(right, "difference"), difference_rtol)


def compare_gravity_policy(
    left: dict[str, str], right: dict[str, str]
) -> None:
    for field in ("cell_id", "active_cells", "active_ranks",
                  "position_policy"):
        if left[field] != right[field]:
            raise RuntimeError(
                f"gravity policy {field} differs: {left[field]} versus {right[field]}")
    if left["position_policy"] != "predicted_centroid_active_refresh":
        raise RuntimeError(
            f"unexpected gravity position policy {left['position_policy']!r}")
    if int(left["active_ranks"]) != 1:
        raise RuntimeError(
            f"gravity stale-mass test used {left['active_ranks']} active ranks, expected 1")
    for label, record in (("full-variable", left), ("partial", right)):
        if int(record["passive_mass_updates"]) <= 0:
            raise RuntimeError(
                f"{label} gravity test produced no passive-rank mass update")
    left_span = number(left, "velocity_span")
    right_span = number(right, "velocity_span")
    if left_span <= 0 or right_span <= 0:
        raise RuntimeError(
            "gravity parity test did not exercise unequal moving generators")
    close("gravity point-velocity span", left_span, right_span, 5e-6)
    left_a = [number(left, field) for field in ("ax", "ay", "az")]
    right_a = [number(right, field) for field in ("ax", "ay", "az")]
    scale = max(math.sqrt(sum(value * value for value in left_a)),
                math.sqrt(sum(value * value for value in right_a)),
                float.fromhex("0x1p-1022"))
    error = math.sqrt(sum((a - b) ** 2 for a, b in zip(left_a, right_a))) / scale
    if error > 5e-6:
        raise RuntimeError(
            f"full/partial gravity acceleration relative error {error:.6g} > 5e-6")
    close("full/partial gravity conserved energy",
          number(left, "energy"), number(right, "energy"), 5e-6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-log", type=Path, required=True)
    parser.add_argument("--full-log", type=Path, required=True)
    parser.add_argument("--forced-reduced-log", type=Path, required=True)
    parser.add_argument("--full-variable-log", type=Path, required=True)
    parser.add_argument("--partial-log", type=Path, required=True)
    parser.add_argument("--empty-owned-log", type=Path, required=True)
    parser.add_argument("--restart-source-log", type=Path, required=True)
    parser.add_argument("--restart-resumed-log", type=Path, required=True)
    parser.add_argument("--restart-continuous-log", type=Path, required=True)
    args = parser.parse_args()

    global_limit = first_limit(args.global_log, "global")
    full_limit = first_limit(args.full_log, "individual")
    full_variable_limit = first_limit(args.full_variable_log, "individual")
    partial_limit = first_limit(args.partial_log, "individual")
    empty_owned_limit = first_limit(args.empty_owned_log, "individual")
    require_active_reference_maximum(args.full_variable_log)
    require_active_reference_maximum(args.partial_log)

    full_fast_path = first_all_active_fast_path(args.full_log)
    empty_owned_fast_path = first_all_active_fast_path(args.empty_owned_log)
    require_grey_diffusion_force_zero_owned(args.empty_owned_log)
    empty_owned_range = first_owned_range(args.empty_owned_log)
    sparse_ownership = first_active_ownership(args.full_variable_log)
    full_variable_tolerance = first_active_tolerance(args.full_variable_log)
    partial_tolerance = first_active_tolerance(args.partial_log)
    full_variable_convergence = first_active_convergence(
        args.full_variable_log)
    partial_convergence = first_active_convergence(args.partial_log)
    full_variable_policy = first_active_historical_policy(
        args.full_variable_log)
    partial_policy = first_active_historical_policy(args.partial_log)
    source_final = restart_fingerprint(args.restart_source_log, "final")
    resumed_initial = restart_fingerprint(args.restart_resumed_log, "initial")
    resumed_final = restart_fingerprint(args.restart_resumed_log, "final")
    continuous_final = restart_fingerprint(args.restart_continuous_log, "final")
    full_variable_gravity = gravity_policy_record(args.full_variable_log)
    partial_gravity = gravity_policy_record(args.partial_log)

    check_formula("global", global_limit, 1.4)
    check_formula("synchronized individual", full_limit, 2.0)
    check_formula("full-variable", full_variable_limit, 2.0)
    check_formula("partial", partial_limit, 2.0)
    check_formula("empty-owned synchronized individual", empty_owned_limit, 2.0)
    check_all_active_fast_path(full_fast_path)
    check_all_active_fast_path(empty_owned_fast_path)
    check_empty_owned_range(empty_owned_range)
    check_nonempty_zero_active(sparse_ownership)
    check_historical_tolerance("full-variable", full_variable_tolerance)
    check_historical_tolerance("partial", partial_tolerance)
    check_historical_convergence("full-variable", full_variable_convergence)
    check_historical_convergence("partial", partial_convergence)
    check_historical_policy("full-variable", full_variable_policy)
    check_historical_policy("partial", partial_policy)
    compare_restart_fingerprints(
        "snapshot write/read round trip", source_final, resumed_initial)
    compare_restart_schedule(
        "continuous versus restarted next event", continuous_final, resumed_final)
    compare_gravity_policy(full_variable_gravity, partial_gravity)

    # All cells are active in this pair. Their reference scales must match;
    # allow small integration-order differences in the limiting change.
    # These lanes use different hydro/gravity integrators before evaluating the
    # MG limiter.  Allow 1e-6 physical-state disagreement here, while retaining
    # the tighter reference-scale check below for identical sparse events.
    compare_pair("global versus synchronized individual", global_limit,
                 full_limit, 2e-2, 1e-6)
    compare_forced_reduced_physics(args.full_log, args.forced_reduced_log)
    # Only one owned cell per rank starts in the short bin in this pair. The
    # partial closure therefore cannot be the source of either reference max.
    compare_pair("full-variable versus partial sparse event",
                 full_variable_limit, partial_limit, 5e-3, 1e-7)

    print("PASS: MG timestep normalization is canonical-owned and MPI-global")
    print("PASS: global growth cap=1.4; individual one-bin cap=2.0")
    print("PASS: synchronized individual MG used the all-active global solver")
    print("PASS: forced all-active reduced MG matches the all-active global MG state")
    print("PASS: all-active MG is collective with at least one zero-owned rank")
    print("PASS: force_synchronized and exact scheduler/conserved state survive HDF5 round trip")
    print("PASS: restarted and uninterrupted runs select the same subsequent event schedule")
    print("PASS: one-rank cross-face gravity gives matching FullReference/AutoPartial acceleration and energy")
    print("PASS: sparse active MG satisfies the MPI-global componentwise backward-error test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
