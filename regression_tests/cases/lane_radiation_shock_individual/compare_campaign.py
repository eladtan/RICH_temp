#!/software/x86_64/5.14.0/python/3.12.1/bin/python3
"""Compare the three Lane radiation-shock benchmark lanes.

Runtime ratios are always reported but never used as pass/fail criteria.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable


def read_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(" ")
        result[key] = value.strip()
    return result


def read_table(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            rows.append([float(value) for value in line.split()])
    return rows


def number(values: dict[str, str], key: str) -> float:
    return float(values[key])


def interpolate(rows: list[list[float]], radius: float, column: int) -> float:
    if radius <= rows[0][0]:
        return rows[0][column]
    if radius >= rows[-1][0]:
        return rows[-1][column]
    lo = 0
    hi = len(rows) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if rows[mid][0] <= radius:
            lo = mid
        else:
            hi = mid
    width = rows[hi][0] - rows[lo][0]
    fraction = 0.0 if width == 0 else (radius - rows[lo][0]) / width
    return rows[lo][column] * (1 - fraction) + rows[hi][column] * fraction


def weighted_symmetric_l1(reference: list[list[float]], candidate: list[list[float]],
                          column: int) -> float:
    numerator = 0.0
    denominator = 0.0
    for index, row in enumerate(reference):
        radius = row[0]
        left = reference[index - 1][0] if index else 0.0
        right = reference[index + 1][0] if index + 1 < len(reference) else radius
        weight = max(radius * radius * (right - left), 0.0)
        a = row[column]
        b = interpolate(candidate, radius, column)
        numerator += weight * abs(a - b)
        denominator += weight * 0.5 * (abs(a) + abs(b))
    return numerator / max(denominator, 1e-300)


def symmetric_relative(a: float, b: float) -> float:
    return abs(a - b) / max(0.5 * (abs(a) + abs(b)), 1e-300)


def spectrum_error(reference: list[list[float]], candidate: list[list[float]]) -> float:
    if len(reference) != len(candidate):
        return math.inf
    numerator = sum(abs(a[4] - b[4]) for a, b in zip(reference, candidate))
    denominator = sum(0.5 * (abs(a[4]) + abs(b[4]))
                      for a, b in zip(reference, candidate))
    return numerator / max(denominator, 1e-300)


class Verdict:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)


def check_pair(verdict: Verdict, label: str, reference_dir: Path,
               candidate_dir: Path, profile_limit: float, shock_limit: float,
               spectrum_limit: float, property_limit: float) -> None:
    reference_profile = read_table(reference_dir / "radial_profile.txt")
    candidate_profile = read_table(candidate_dir / "radial_profile.txt")
    profile_names = ["density", "pressure", "radial_velocity", "temperature",
                     "radiation_energy_density"]
    profile_errors = {
        name: weighted_symmetric_l1(reference_profile, candidate_profile, column)
        for column, name in enumerate(profile_names, start=1)
    }
    for name, error in profile_errors.items():
        verdict.require(error <= profile_limit,
                        f"{label} {name} profile L1 {error:.6g} > {profile_limit:.6g}")

    reference_metrics = read_kv(reference_dir / "metrics.txt")
    candidate_metrics = read_kv(candidate_dir / "metrics.txt")
    reference_counters = read_kv(reference_dir / "counters.txt")
    candidate_counters = read_kv(candidate_dir / "counters.txt")
    shock_error = symmetric_relative(number(reference_metrics, "shock_radius"),
                                     number(candidate_metrics, "shock_radius"))
    verdict.require(shock_error <= shock_limit,
                    f"{label} shock radius error {shock_error:.6g} > {shock_limit:.6g}")
    for key in ("peak_outward_mach", "shock_compression_ratio"):
        error = symmetric_relative(number(reference_metrics, key),
                                   number(candidate_metrics, key))
        verdict.require(error <= property_limit,
                        f"{label} {key} error {error:.6g} > {property_limit:.6g}")

    spectral_error = spectrum_error(read_table(reference_dir / "spectrum.txt"),
                                    read_table(candidate_dir / "spectrum.txt"))
    verdict.require(spectral_error <= spectrum_limit,
                    f"{label} spectrum error {spectral_error:.6g} > {spectrum_limit:.6g}")
    reference_amr = (number(reference_counters, "refined_cells") +
                     number(reference_counters, "derefined_cells"))
    candidate_amr = (number(candidate_counters, "refined_cells") +
                     number(candidate_counters, "derefined_cells"))
    amr_error = symmetric_relative(reference_amr, candidate_amr)
    verdict.require(amr_error <= 0.05,
                    f"{label} AMR operation-count error {amr_error:.6g} > 0.05")
    verdict.note(f"{label}_profile_max_l1 {max(profile_errors.values()):.17g}")
    verdict.note(f"{label}_shock_error {shock_error:.17g}")
    verdict.note(f"{label}_spectrum_error {spectral_error:.17g}")
    verdict.note(f"{label}_amr_count_error {amr_error:.17g}")


def required_artifacts(directory: Path) -> Iterable[Path]:
    for name in ("manifest.txt", "timing.txt", "counters.txt", "metrics.txt",
                 "history.txt", "radial_profile.txt", "spectrum.txt",
                 "initial_state.h5", "initial_state.pvtu",
                 "final_state.h5", "final_state.pvtu", "exit_code.txt",
                 "mg_solver_diagnostics.log"):
        yield directory / name


def require_endpoint_snapshots(verdict: Verdict, label: str, directory: Path,
                               expected_ranks: int) -> None:
    for stem in ("initial_state", "final_state"):
        for suffix in ("h5", "pvtu"):
            master = directory / f"{stem}.{suffix}"
            verdict.require(master.is_file() and master.stat().st_size > 0,
                            f"{label} has empty {master.name}")
        pieces = directory / stem
        verdict.require(pieces.is_dir(),
                        f"{label} missing {stem}/ rank-piece directory")
        if not pieces.is_dir():
            continue
        h5_count = sum(path.is_file() and path.stat().st_size > 0
                       for path in pieces.glob("*.h5"))
        vtu_count = sum(path.is_file() and path.stat().st_size > 0
                        for path in pieces.glob("*.vtu"))
        verdict.require(h5_count == expected_ranks,
                        f"{label} {stem} has {h5_count} HDF5 rank pieces, "
                        f"expected {expected_ranks}")
        verdict.require(vtu_count == expected_ranks,
                        f"{label} {stem} has {vtu_count} VTU rank pieces, "
                        f"expected {expected_ranks}")


def read_mg_diagnostic_counts(path: Path) -> dict[str, int | float]:
    counts = {
        "progress": 0,
        "convergence": 0,
        "active_tolerance": 0,
        "active_bad_tolerance": 0,
        "active_converged": 0,
        "active_stagnated": 0,
        "active_other": 0,
        "active_bad_backward": 0,
        "active_historical_policy": 0,
        "active_bad_historical_policy": 0,
        "spectral_repair_records": 0,
        "spectral_repair_bad": 0,
        "spectral_repair_warnings": 0,
        "spectral_repair_cumulative_injected": 0.0,
        "spectral_repair_cumulative_fraction": 0.0,
        "iteration_cap_hits": 0,
        "maximum_iterations": 0,
        "global_limit": 0,
        "individual_limit": 0,
        "normalized_limit": 0,
        "state_detail": 0,
        "coupling_detail": 0,
        "iteration_total": 0,
    }
    with path.open(errors="replace") as diagnostics:
        for line in diagnostics:
            fields = dict(token.split("=", 1) for token in line.split()
                          if "=" in token)
            if "MG_BICGSTAB_PROGRESS" in line and "max0=" in line and "max1=" in line:
                counts["progress"] += 1
            if ("MG_BICGSTAB_CONVERGENCE" in line and "max0=" in line and
                    "max1=" in line and "max0_cell_id=" in line and
                    "max1_cell_id=" in line):
                counts["convergence"] += 1
            if "MG_BICGSTAB_CONVERGENCE" in line:
                try:
                    iterations = int(fields["iterations"])
                except (KeyError, ValueError):
                    iterations = 10000
                counts["maximum_iterations"] = max(
                    counts["maximum_iterations"], iterations)
                if iterations >= 10000:
                    counts["iteration_cap_hits"] += 1
            if ("MG_BICGSTAB_TOLERANCE" in line and
                    fields.get("scope") == "distributed_active"):
                counts["active_tolerance"] += 1
                required = {
                    "requested", "squared_scaled_tolerance",
                    "effective_norm_tolerance",
                    "diagnostic_backward_reference", "criterion",
                    "eta_inf_role", "max_global_row_nnz",
                }
                try:
                    requested = float(fields["requested"])
                    squared = float(fields["squared_scaled_tolerance"])
                    effective = float(fields["effective_norm_tolerance"])
                    diagnostic_reference = float(
                        fields["diagnostic_backward_reference"])
                    row_nonzeros = int(fields["max_global_row_nnz"])
                    expected = max(requested,
                                   32 * math.ulp(1.0) * row_nonzeros)
                    valid = (not required.difference(fields) and
                             fields["criterion"] ==
                             "historical_diagonal_scaled_squared_residual" and
                             fields["eta_inf_role"] == "diagnostic_only" and
                             requested > 0 and
                             row_nonzeros > 0 and
                             all(math.isfinite(value) for value in
                                 (requested, squared, effective,
                                  diagnostic_reference)) and
                             math.isclose(squared, requested,
                                          rel_tol=5e-6,
                                          abs_tol=math.ulp(1.0)) and
                             math.isclose(effective, math.sqrt(squared),
                                          rel_tol=5e-6,
                                          abs_tol=math.ulp(1.0)) and
                             math.isclose(diagnostic_reference, expected,
                                          rel_tol=5e-6,
                                          abs_tol=math.ulp(1.0)))
                except (KeyError, ValueError):
                    valid = False
                if not valid:
                    counts["active_bad_tolerance"] += 1
            if ("MG_BICGSTAB_CONVERGENCE" in line and
                    fields.get("scope") == "distributed_active"):
                outcome = fields.get("outcome")
                if outcome == "converged":
                    counts["active_converged"] += 1
                    required = {
                        "error", "weighted_residual_squared",
                        "weighted_rhs_squared",
                        "backward_error", "backward_tolerance",
                        "backward_rank", "backward_cell_id",
                        "backward_group", "backward_residual",
                        "backward_scale", "maximum_scale",
                        "safe_minimum_scale", "max0", "max1", "negative",
                    }
                    try:
                        historical_error = float(fields["error"])
                        weighted_residual_squared = float(
                            fields["weighted_residual_squared"])
                        weighted_rhs_squared = float(
                            fields["weighted_rhs_squared"])
                        backward_error = float(fields["backward_error"])
                        backward_tolerance = float(
                            fields["backward_tolerance"])
                        residual = float(fields["backward_residual"])
                        scale = float(fields["backward_scale"])
                        maximum_scale = float(fields["maximum_scale"])
                        safe_scale = float(fields["safe_minimum_scale"])
                        max0 = float(fields["max0"])
                        max1 = float(fields["max1"])
                        negative = float(fields["negative"])
                        expected_safe = max(
                            float.fromhex("0x1p-1022"),
                            32 * math.ulp(1.0) * maximum_scale)
                        expected_error = abs(residual) / max(scale,
                                                             safe_scale)
                        valid = (not required.difference(fields) and
                                 all(math.isfinite(value) for value in
                                     (backward_error, backward_tolerance,
                                      residual, scale, maximum_scale,
                                      safe_scale, expected_error, max0, max1,
                                      negative, historical_error,
                                      weighted_residual_squared,
                                      weighted_rhs_squared)) and
                                 historical_error >= 0 and
                                 weighted_residual_squared >= 0 and
                                 weighted_rhs_squared > 0 and
                                 math.isclose(
                                     historical_error,
                                     weighted_residual_squared /
                                     weighted_rhs_squared,
                                     rel_tol=5e-5,
                                     abs_tol=math.ulp(1.0)) and
                                 backward_error >= 0 and
                                 backward_tolerance > 0 and
                                 int(fields["backward_rank"]) >= 0 and
                                 int(fields["backward_group"]) >= 0 and
                                 math.isclose(safe_scale, expected_safe,
                                              rel_tol=5e-5,
                                              abs_tol=float.fromhex(
                                                  "0x1p-1022")) and
                                 math.isclose(backward_error, expected_error,
                                              rel_tol=5e-5,
                                              abs_tol=math.ulp(1.0)))
                    except (KeyError, ValueError, ZeroDivisionError):
                        valid = False
                    if not valid:
                        counts["active_bad_backward"] += 1
                elif outcome == "stagnated":
                    counts["active_stagnated"] += 1
                else:
                    counts["active_other"] += 1
            if ("MG_BICGSTAB_HISTORICAL_POLICY" in line and
                    fields.get("scope") == "distributed_active"):
                counts["active_historical_policy"] += 1
                required = {
                    "squared_scaled_tolerance", "effective_norm_tolerance",
                    "last_true_eta_inf", "last_true_eta_iteration",
                    "last_true_eta_age", "pre_correction_eta_inf",
                    "final_eta_inf", "eta_inf_role",
                }
                try:
                    squared = float(fields["squared_scaled_tolerance"])
                    effective = float(fields["effective_norm_tolerance"])
                    last_eta = float(fields["last_true_eta_inf"])
                    pre_eta = float(fields["pre_correction_eta_inf"])
                    valid = (not required.difference(fields) and
                             squared > 0 and last_eta >= 0 and pre_eta >= 0 and
                             all(math.isfinite(value) for value in
                                 (squared, effective, last_eta, pre_eta)) and
                             math.isclose(effective, math.sqrt(squared),
                                          rel_tol=5e-6,
                                          abs_tol=math.ulp(1.0)) and
                             int(fields["last_true_eta_iteration"]) >= 0 and
                             int(fields["last_true_eta_age"]) >= 0 and
                             fields["final_eta_inf"] == "not_evaluated" and
                             fields["eta_inf_role"] == "diagnostic_only")
                except (KeyError, ValueError):
                    valid = False
                if not valid:
                    counts["active_bad_historical_policy"] += 1
            if "MG_SPECTRAL_POSITIVITY_REPAIR_WARNING" in line:
                counts["spectral_repair_warnings"] += 1
            elif "MG_SPECTRAL_POSITIVITY_REPAIR" in line:
                counts["spectral_repair_records"] += 1
                required = {
                    "cells", "groups", "injected_extent_sum",
                    "max_relative_deficit", "cumulative_cells",
                    "cumulative_groups", "cumulative_injected_extent",
                    "maximum_global_radiation_energy",
                    "cumulative_injection_fraction",
                }
                try:
                    values = [float(fields[key]) for key in
                              ("injected_extent_sum", "max_relative_deficit",
                               "cumulative_injected_extent",
                               "maximum_global_radiation_energy",
                               "cumulative_injection_fraction")]
                    valid = (not required.difference(fields) and
                             int(fields["cells"]) > 0 and
                             int(fields["groups"]) > 0 and
                             int(fields["cumulative_cells"]) > 0 and
                             int(fields["cumulative_groups"]) > 0 and
                             all(math.isfinite(value) and value >= 0
                                 for value in values))
                    if valid:
                        counts["spectral_repair_cumulative_injected"] = max(
                            counts["spectral_repair_cumulative_injected"],
                            values[2])
                        counts["spectral_repair_cumulative_fraction"] = max(
                            counts["spectral_repair_cumulative_fraction"],
                            values[4])
                except (KeyError, ValueError):
                    valid = False
                if not valid:
                    counts["spectral_repair_bad"] += 1
            if ("MG_TIMESTEP_LIMIT mode=global" in line and
                    "cell_id=" in line and "suggested_dt=" in line):
                counts["global_limit"] += 1
            if ("MG_TIMESTEP_LIMIT mode=individual" in line and
                    "cell_id=" in line and "suggested_dt=" in line):
                counts["individual_limit"] += 1
            if ("MG_TIMESTEP_LIMIT mode=" in line and
                    "max_Er=" in line and "max_rhoT=" in line and
                    "growth_cap=" in line and
                    "reference_scope=canonical_owned_global" in line):
                counts["normalized_limit"] += 1
            if ("Radiation time step ID" in line and "old Er " in line and
                    "new Er " in line and "Tgas " in line and
                    "Trad " in line and "density " in line):
                counts["state_detail"] += 1
            if ("kp=" in line and "fleck factor " in line and
                    "upsilon " in line and "occupation " in line):
                counts["coupling_detail"] += 1
            if "Total iterations:" in line:
                counts["iteration_total"] += 1
                try:
                    iterations = int(line.split(
                        "Total iterations:", 1)[1].strip().split()[0])
                    counts["maximum_iterations"] = max(
                        counts["maximum_iterations"], iterations)
                    if iterations >= 10000:
                        counts["iteration_cap_hits"] += 1
                except (IndexError, ValueError):
                    counts["iteration_cap_hits"] += 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-dir", type=Path, required=True)
    parser.add_argument("--full-variable-dir", type=Path, required=True)
    parser.add_argument("--partial-dir", type=Path, required=True)
    parser.add_argument("--full-dir", type=Path)
    parser.add_argument("--tier", choices=("production", "reduced"),
                        default="production")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requested_preconditioner = os.environ.get(
        "RICH_TEST_MG_PRECONDITIONER", "cell_block")
    expected_preconditioner = {
        "scalar": "scalar_jacobi",
        "cell_block": "cell_block_jacobi",
        "cell_block_two_sweep": "cell_block_jacobi_two_sweep",
        "cell_block_four_sweep": "cell_block_jacobi_four_sweep",
        "cell_block_eight_sweep": "cell_block_jacobi_eight_sweep",
    }.get(requested_preconditioner)
    if expected_preconditioner is None:
        raise ValueError(
            "unsupported RICH_TEST_MG_PRECONDITIONER: " +
            requested_preconditioner)

    verdict = Verdict()
    lanes = {
        "global": args.global_dir,
        "full-variable": args.full_variable_dir,
        "partial": args.partial_dir,
    }
    if args.full_dir is not None:
        lanes["full"] = args.full_dir
    try:
        for lane, directory in lanes.items():
            for artifact in required_artifacts(directory):
                verdict.require(artifact.exists(), f"{lane} missing {artifact.name}")
        if verdict.failures:
            raise FileNotFoundError("required campaign artifacts are missing")

        manifests = {lane: read_kv(directory / "manifest.txt")
                     for lane, directory in lanes.items()}
        metrics = {lane: read_kv(directory / "metrics.txt")
                   for lane, directory in lanes.items()}
        counters = {lane: read_kv(directory / "counters.txt")
                    for lane, directory in lanes.items()}
        timings = {lane: read_kv(directory / "timing.txt")
                   for lane, directory in lanes.items()}

        expected_ranks = 128 if args.tier == "production" else 8
        expected_cells = 2_000_000 if args.tier == "production" else 32_768
        expected_groups = 16
        for lane, directory in lanes.items():
            require_endpoint_snapshots(
                verdict, lane, directory, expected_ranks)
        checksums = set()
        maximum_intervals = {}
        initial_intervals = {}
        for lane in lanes:
            manifest = manifests[lane]
            verdict.require(manifest.get("mode") == lane,
                            f"{lane} manifest mode is {manifest.get('mode')}")
            verdict.require(int(manifest["mpi_ranks"]) == expected_ranks,
                            f"{lane} rank count is not {expected_ranks}")
            verdict.require(int(manifest["requested_initial_cells"]) == expected_cells,
                            f"{lane} initial cell request is not {expected_cells}")
            verdict.require(int(manifest["energy_groups"]) == expected_groups,
                            f"{lane} group count is not {expected_groups}")
            verdict.require(
                manifest.get("mg_preconditioner") == expected_preconditioner,
                f"{lane} MG preconditioner is "
                f"{manifest.get('mg_preconditioner')}, expected "
                f"{expected_preconditioner}")
            verdict.require(math.isclose(
                number(manifest, "energy_min_eV"), 1.60218e-12,
                rel_tol=5e-6, abs_tol=0.0),
                f"{lane} lower MG boundary is not 1 eV")
            verdict.require(math.isclose(
                number(manifest, "energy_max_eV"), 3.20435e-6,
                rel_tol=5e-6, abs_tol=0.0),
                f"{lane} upper MG boundary is not 2 MeV")
            verdict.require(
                manifest.get("gravity") ==
                "barnes_hut_quadrupole_theta_0.7",
                f"{lane} did not use the quadrupole tree gravity model")
            for feature in ("compton", "doppler", "flux_limiter",
                            "hydro_feedback", "protections",
                            "cooling_limiter"):
                verdict.require(
                    manifest.get(feature) == "1",
                    f"{lane} manifest does not enable {feature}")
            verdict.require(math.isclose(
                number(manifest, "initial_temperature_floor_K"), 1.0e4,
                rel_tol=0.0, abs_tol=1.0e-10),
                f"{lane} initial temperature floor is not 1e4 K")
            verdict.require(math.isclose(
                number(manifest, "pulse_peak_temperature_K"), 3.0e7,
                rel_tol=0.0, abs_tol=1.0e-6),
                f"{lane} pulse peak temperature is not 3e7 K")
            verdict.require(math.isclose(
                number(manifest, "pulse_radius_fraction"), 0.20,
                rel_tol=0.0, abs_tol=1.0e-14),
                f"{lane} pulse radius is not 0.2 R")
            quantum = number(manifest, "time_quantum")
            maximum_bin = int(manifest["requested_maximum_bin"])
            maximum_individual_dt = number(manifest,
                                           "maximum_individual_dt")
            maximum_global_dt = number(manifest, "maximum_global_dt")
            expected_maximum_dt = math.ldexp(quantum, maximum_bin)
            verdict.require(math.isfinite(maximum_global_dt) and
                            maximum_global_dt > 0,
                            f"{lane} has no finite positive global timestep cap")
            verdict.require(math.isclose(
                maximum_individual_dt, expected_maximum_dt,
                rel_tol=1e-14, abs_tol=0.0),
                f"{lane} individual maximum interval does not match its bin")
            verdict.require(math.isclose(
                maximum_global_dt, maximum_individual_dt,
                rel_tol=1e-14, abs_tol=0.0),
                f"{lane} global and individual maximum intervals differ")
            maximum_intervals[lane] = maximum_individual_dt
            initial_intervals[lane] = number(manifest, "initial_dt")
            checksums.add((manifest["initial_checksum_xor"],
                           manifest["initial_checksum_sum"]))
            exit_code = int((lanes[lane] / "exit_code.txt").read_text().strip())
            verdict.require(exit_code == 0, f"{lane} executable exit code {exit_code}")
            diagnostic_counts = read_mg_diagnostic_counts(
                lanes[lane] / "mg_solver_diagnostics.log")
            verdict.require(diagnostic_counts["convergence"] > 0,
                            f"{lane} has no MG max0/max1 convergence records")
            expected_limit = "global_limit" if lane == "global" else "individual_limit"
            verdict.require(diagnostic_counts[expected_limit] > 0,
                            f"{lane} has no MG limiting-cell timestep records")
            verdict.require(diagnostic_counts["normalized_limit"] > 0,
                            f"{lane} has no canonical-owned global MG "
                            "normalization record")
            verdict.require(diagnostic_counts["state_detail"] > 0,
                            f"{lane} has no detailed MG density/T/Erad records")
            verdict.require(diagnostic_counts["coupling_detail"] > 0,
                            f"{lane} has no detailed MG opacity/coupling records")
            verdict.require(diagnostic_counts["iteration_total"] > 0,
                            f"{lane} has no MG total-iteration records")
            verdict.require(diagnostic_counts["iteration_cap_hits"] == 0,
                            f"{lane} reached the fixed 10,000-iteration "
                            "BiCGSTAB safety limit")
            if lane != "global":
                verdict.require(diagnostic_counts["active_tolerance"] > 0,
                                f"{lane} has no active MG historical "
                                "tolerance records")
                verdict.require(
                    diagnostic_counts["active_bad_tolerance"] == 0,
                    f"{lane} has malformed or inconsistent active MG "
                    "historical tolerances")
                verdict.require(diagnostic_counts["active_converged"] > 0,
                                f"{lane} has no converged active MG solve")
                verdict.require(
                    diagnostic_counts["active_bad_backward"] == 0,
                    f"{lane} has an invalid active MG historical/eta "
                    "diagnostic")
                verdict.require(
                    diagnostic_counts["active_historical_policy"] > 0,
                    f"{lane} has no historical-policy diagnostic")
                verdict.require(
                    diagnostic_counts["active_bad_historical_policy"] == 0,
                    f"{lane} has a malformed historical-policy diagnostic")
            verdict.require(
                diagnostic_counts["spectral_repair_bad"] == 0,
                f"{lane} has malformed spectral-repair accounting")
            verdict.note(
                f"{lane}_mg_diagnostics progress={diagnostic_counts['progress']} "
                f"convergence={diagnostic_counts['convergence']} "
                f"timestep_limits={diagnostic_counts[expected_limit]} "
                f"normalized_limits={diagnostic_counts['normalized_limit']} "
                f"state_details={diagnostic_counts['state_detail']} "
                f"coupling_details={diagnostic_counts['coupling_detail']} "
                f"iteration_totals={diagnostic_counts['iteration_total']} "
                f"maximum_iterations={diagnostic_counts['maximum_iterations']} "
                f"active_converged={diagnostic_counts['active_converged']} "
                f"active_stagnated={diagnostic_counts['active_stagnated']} "
                f"active_other={diagnostic_counts['active_other']} "
                f"repair_records={diagnostic_counts['spectral_repair_records']} "
                f"repair_warnings={diagnostic_counts['spectral_repair_warnings']} "
                f"repair_cumulative_injected="
                f"{diagnostic_counts['spectral_repair_cumulative_injected']:.17g} "
                f"repair_cumulative_fraction="
                f"{diagnostic_counts['spectral_repair_cumulative_fraction']:.17g}")

        reference_maximum_interval = maximum_intervals["global"]
        reference_initial_interval = initial_intervals["global"]
        for lane in lanes:
            verdict.require(math.isclose(
                maximum_intervals[lane], reference_maximum_interval,
                rel_tol=1e-14, abs_tol=0.0),
                f"{lane} maximum interval differs from the global lane")
            verdict.require(math.isclose(
                initial_intervals[lane], reference_initial_interval,
                rel_tol=1e-14, abs_tol=0.0),
                f"{lane} initial interval differs from the global lane")

            values = metrics[lane]
            requested_target = number(values, "target_final_time")
            effective_target = (number(values, "effective_target_final_time")
                                if "effective_target_final_time" in values
                                else requested_target)
            endpoint_tolerance = 1e-12 * max(abs(effective_target), 1.0)
            verdict.require(abs(number(values, "final_time") -
                                effective_target) <= endpoint_tolerance,
                            f"{lane} did not reach its effective final time")
            terminal_quantum = (number(values, "terminal_time_quantum")
                                if "terminal_time_quantum" in values else 0.0)
            rounding_slack = (16.0 * 2.220446049250313e-16 *
                              max(abs(requested_target),
                                  abs(effective_target), 1.0))
            verdict.require(math.isfinite(terminal_quantum) and
                            terminal_quantum >= 0,
                            f"{lane} terminal quantum is invalid")
            if terminal_quantum > 0:
                terminal_tick = number(values, "terminal_tick")
                terminal_origin = (number(values, "terminal_time_origin")
                                   if "terminal_time_origin" in values else 0.0)
                verdict.require(math.isfinite(terminal_tick) and
                                terminal_tick >= 0 and
                                terminal_tick == math.floor(terminal_tick),
                                f"{lane} terminal tick is invalid")
                verdict.require(math.isfinite(terminal_origin),
                                f"{lane} terminal origin is not finite")
                lattice_target = (terminal_origin +
                                  terminal_quantum * terminal_tick)
                verdict.require(abs(effective_target - lattice_target) <=
                                rounding_slack,
                                f"{lane} effective final time is off the "
                                "scheduler lattice")
                verdict.require(abs(effective_target - requested_target) <=
                                0.5 * terminal_quantum + rounding_slack,
                                f"{lane} effective final time is not the "
                                "nearest scheduler tick")
            verdict.require(number(values, "mass_drift") <= 1e-10,
                            f"{lane} mass drift exceeds 1e-10")
            # This diagnostic is material kinetic/internal plus radiation
            # energy.  The benchmark has self-gravity, but the acceleration
            # adapter does not expose gravitational potential energy, so this
            # is not a conserved total.  Keep it as a finite stability guard;
            # endpoint agreement is checked separately below.
            nongrav_change = number(values, "nongrav_energy_drift")
            verdict.require(math.isfinite(nongrav_change) and
                            nongrav_change <= 1e-3,
                            f"{lane} nongrav energy change exceeds 1e-3")
            verdict.require(number(values, "normalized_momentum_drift") <= 1e-5,
                            f"{lane} normalized momentum drift exceeds 1e-5")
            verdict.require(number(values, "normalized_center_of_mass") <= 1e-3,
                            f"{lane} center-of-mass/R exceeds 1e-3")
            verdict.require(number(values, "minimum_material_energy") > 0,
                            f"{lane} has non-positive material energy")
            verdict.require(number(values, "minimum_group_energy") >= 0,
                            f"{lane} has negative group energy")
            verdict.require(0 < number(values, "minimum_fleck_factor") and
                            number(values, "maximum_fleck_factor") <= 1,
                            f"{lane} Fleck factors are outside (0,1]")
            verdict.require(number(values, "maximum_transport_scattering") > 0,
                            f"{lane} has no Compton transport scattering")
            verdict.require(number(values, "significant_groups") >= 8,
                            f"{lane} has fewer than 8 significant groups")
            verdict.require(number(values, "final_cells") <= 3_000_000,
                            f"{lane} exceeded the 3,000,000-cell cap")
            minimum_candidate_fraction = number(
                counters[lane], "minimum_radiation_candidate_fraction")
            verdict.require(math.isfinite(minimum_candidate_fraction) and
                            minimum_candidate_fraction > 0,
                            f"{lane} radiation candidate made no progress")
            verdict.require(number(counters[lane], "refined_cells") > 0,
                            f"{lane} performed no refinement")
            verdict.require(number(counters[lane], "derefined_cells") > 0,
                            f"{lane} performed no derefinement")

        verdict.require(len(checksums) == 1,
                        "the three lanes did not use the same immutable initial state")

        for lane in ("full-variable", "partial"):
            values = metrics[lane]
            lane_counters = counters[lane]
            if args.tier == "production":
                verdict.require(number(values, "peak_outward_mach") >= 3,
                                f"{lane} peak outward Mach is below 3")
                verdict.require(number(values, "shock_compression_ratio") >= 2.5,
                                f"{lane} compression ratio is below 2.5")
                radius = number(manifests[lane], "radius_cm")
                verdict.require(number(values, "maximum_shock_radius") >
                                0.10 * radius,
                                f"{lane} shock did not leave the injection core")
                verdict.require(number(values, "maximum_shock_radius") <
                                1.5 * radius,
                                f"{lane} shock reached the comparison boundary")
            verdict.require(number(lane_counters, "distinct_time_bins") >= 3,
                            f"{lane} exercised fewer than 3 timestep bins")
            verdict.require(number(values, "time_bin_ratio") >= 8,
                            f"{lane} timestep ratio is below 8")
            if args.tier == "production":
                verdict.require(number(values, "mean_event_active_fraction") <= 0.25,
                                f"{lane} mean event active fraction exceeds 25%")
            verdict.require(number(lane_counters,
                                   "reduced_gravity_rank_events") > 0,
                            f"{lane} did no reduced active-target gravity work")
            verdict.require(number(lane_counters,
                                   "reduced_multigroup_rank_events") > 0,
                            f"{lane} did no reduced active-row MG work")
        verdict.require(number(counters["partial"],
                               "partial_mesh_rank_events") > 0,
                        "partial lane never used a partial mesh")

        check_pair(verdict, "full_variable_vs_partial", args.full_variable_dir,
                   args.partial_dir, 0.01, 0.01, 0.001, 0.01)
        check_pair(verdict, "global_vs_full_variable", args.global_dir,
                   args.full_variable_dir, 0.01, 0.01, 0.001, 0.01)
        check_pair(verdict, "global_vs_partial", args.global_dir,
                   args.partial_dir, 0.01, 0.01, 0.001, 0.01)
        if args.full_dir is not None:
            check_pair(verdict, "global_vs_full", args.global_dir,
                       args.full_dir, 0.01, 0.01, 0.001, 0.01)

        global_time = number(timings["global"], "evolution_wall_max")
        full_time = number(timings["full-variable"], "evolution_wall_max")
        partial_time = number(timings["partial"], "evolution_wall_max")
        verdict.note(f"global_evolution_seconds {global_time:.17g}")
        verdict.note(f"full_variable_evolution_seconds {full_time:.17g}")
        verdict.note(f"partial_evolution_seconds {partial_time:.17g}")
        verdict.note(f"global_over_full_variable_speedup {global_time/full_time:.17g}")
        verdict.note(f"global_over_partial_speedup {global_time/partial_time:.17g}")
        verdict.note(f"full_variable_over_partial_speedup {full_time/partial_time:.17g}")
    except Exception as error:  # Always leave a verdict artifact.
        verdict.failures.append(f"comparison exception: {error}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        output.write(f"status {'PASS' if not verdict.failures else 'FAIL'}\n")
        for note in verdict.notes:
            output.write(f"metric {note}\n")
        for failure in verdict.failures:
            output.write(f"failure {failure}\n")
    return 0 if not verdict.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
