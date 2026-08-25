#!/software/x86_64/5.14.0/python/3.12.1/bin/python3
"""Check the synchronized dt/dt/2 calibration used by the production jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

from compare_campaign import (read_kv, read_mg_diagnostic_counts, read_table,
                              spectrum_error, weighted_symmetric_l1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt-dir", type=Path, required=True)
    parser.add_argument("--half-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    failures: list[str] = []
    notes: list[str] = []
    try:
        for directory in (args.dt_dir, args.half_dir):
            if int((directory / "exit_code.txt").read_text().strip()) != 0:
                failures.append(f"{directory.name} executable failed")
            diagnostics = read_mg_diagnostic_counts(
                directory / "mg_solver_diagnostics.log")
            if diagnostics["convergence"] == 0:
                failures.append(
                    f"{directory.name} has no MG max0/max1 convergence records")
            if diagnostics["global_limit"] == 0:
                failures.append(
                    f"{directory.name} has no MG limiting-cell timestep records")
            if diagnostics["state_detail"] == 0:
                failures.append(
                    f"{directory.name} has no detailed MG density/T/Erad records")
            if diagnostics["coupling_detail"] == 0:
                failures.append(
                    f"{directory.name} has no detailed MG opacity/coupling records")
            if diagnostics["iteration_total"] == 0:
                failures.append(
                    f"{directory.name} has no MG total-iteration records")
            notes.append(
                f"{directory.name}_mg_diagnostics "
                f"progress={diagnostics['progress']} "
                f"convergence={diagnostics['convergence']} "
                f"timestep_limits={diagnostics['global_limit']} "
                f"state_details={diagnostics['state_detail']} "
                f"coupling_details={diagnostics['coupling_detail']} "
                f"iteration_totals={diagnostics['iteration_total']}")
        dt_profile = read_table(args.dt_dir / "radial_profile.txt")
        half_profile = read_table(args.half_dir / "radial_profile.txt")
        errors = [weighted_symmetric_l1(dt_profile, half_profile, column)
                  for column in range(1, 6)]
        max_profile_error = max(errors)
        spectral_error = spectrum_error(read_table(args.dt_dir / "spectrum.txt"),
                                        read_table(args.half_dir / "spectrum.txt"))
        notes.append(f"maximum_profile_l1 {max_profile_error:.17g}")
        notes.append(f"maximum_significant_spectrum_error {spectral_error:.17g}")
        if max_profile_error > 0.005:
            failures.append(f"dt/dt2 profile error {max_profile_error:.6g} > 0.005")
        if spectral_error > 0.01:
            failures.append(f"dt/dt2 spectrum error {spectral_error:.6g} > 0.01")
        for label, directory in (("dt", args.dt_dir), ("dt2", args.half_dir)):
            metrics = read_kv(directory / "metrics.txt")
            if float(metrics["minimum_material_energy"]) <= 0:
                failures.append(f"{label} has non-positive material energy")
            if float(metrics["minimum_group_energy"]) < 0:
                failures.append(f"{label} has negative radiation energy")
    except Exception as error:
        failures.append(f"calibration exception: {error}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        output.write(f"status {'PASS' if not failures else 'FAIL'}\n")
        for note in notes:
            output.write(f"metric {note}\n")
        for failure in failures:
            output.write(f"failure {failure}\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
