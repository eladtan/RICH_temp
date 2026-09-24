#!/usr/bin/env python3
"""Average independent TRE-3D histories and compare the ensemble to theory.

The ensemble is formed at every common census time before fitting.  This is a
Monte Carlo variance-reduction diagnostic, not a replacement for reporting the
individual realizations: the JSON output also contains each run's fitted rates
and their sample spread.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import analyze_tre_3d as single


AVERAGED_KEYS = (
    "A_U1", "A_E1", "A_U2", "A_E2", "A_U3", "A_E3",
    "material_energy", "radiation_energy", "total_energy",
)


def read_ensemble(paths: list[Path]):
    if not paths:
        raise RuntimeError("at least one history is required")

    metadata, reference_rows = single.read_history(paths[0])
    histories = [reference_rows]
    for path in paths[1:]:
        other_metadata, other_rows = single.read_history(path)
        if len(other_rows) != len(reference_rows):
            raise RuntimeError(
                f"history length mismatch: {paths[0]} has {len(reference_rows)} "
                f"rows but {path} has {len(other_rows)}")
        for index, (reference, other) in enumerate(zip(reference_rows, other_rows)):
            for key in ("tau_end", "tau_mid"):
                if abs(reference[key] - other[key]) > 1.0e-10:
                    raise RuntimeError(
                        f"time-grid mismatch at row {index} for {path}: "
                        f"{key} differs")
        for key, value in other_metadata.items():
            if key in metadata and metadata[key] != value:
                raise RuntimeError(
                    f"metadata mismatch for {key}: {metadata[key]!r} != {value!r}")
        histories.append(other_rows)

    averaged = []
    count = float(len(histories))
    for row_index, reference in enumerate(reference_rows):
        row = dict(reference)
        for key in AVERAGED_KEYS:
            row[key] = sum(history[row_index][key] for history in histories) / count
        row["particle_count"] = int(round(sum(
            history[row_index]["particle_count"] for history in histories
        ) / count))
        averaged.append(row)

    metadata = dict(metadata)
    metadata["ensemble_size"] = str(len(histories))
    return metadata, averaged, histories


def fit_rows(rows, lo: float, hi: float):
    modes = single.exact_modes()
    fits = []
    rms_values = []
    for index, mode in enumerate(modes, start=1):
        material_fit = single.fit_rate(rows, "tau_end", f"A_U{index}", lo, hi)
        radiation_fit = single.fit_rate(rows, "tau_mid", f"A_E{index}", lo, hi)
        material_rms = single.rms_normalized(
            rows, "tau_end", f"A_U{index}", mode["amplitude"],
            mode["s"], lo, hi)
        radiation_rms = single.rms_normalized(
            rows, "tau_mid", f"A_E{index}", mode["amplitude"] * mode["G"],
            mode["s"], lo, hi)
        fits.append({"material": material_fit, "radiation": radiation_fit})
        rms_values.append({"material": material_rms, "radiation": radiation_rms})
    return modes, fits, rms_values


def relative_rate_errors(modes, fits):
    errors = []
    for mode, fit in zip(modes, fits):
        errors.extend([
            abs(fit["material"]["rate"] - mode["s"]) / abs(mode["s"]),
            abs(fit["radiation"]["rate"] - mode["s"]) / abs(mode["s"]),
        ])
    return errors


def sample_mean_and_error(values):
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--average-output", type=Path, default=None)
    parser.add_argument("--rate-window", nargs=2, type=float, default=None,
                        metavar=("TAU_MIN", "TAU_MAX"))
    parser.add_argument("--rate-relative-tolerance", type=float, default=None)
    args = parser.parse_args()

    metadata, rows, histories = read_ensemble(args.input)
    final_tau = rows[-1]["tau_end"]
    lo, hi = args.rate_window or (
        max(0.5, 0.10 * final_tau),
        min(0.90 * final_tau, 0.50 * final_tau),
    )
    modes, fits, rms_values = fit_rows(rows, lo, hi)
    relative_errors = relative_rate_errors(modes, fits)

    per_run_fits = []
    for history in histories:
        _, run_fits, _ = fit_rows(history, lo, hi)
        per_run_fits.append(run_fits)

    rate_tolerance = 0.045
    rms_tolerance = 0.10
    energy_tolerance = 0.02
    initial_energy = rows[0]["total_energy"]
    energy_drift = max(
        abs(row["total_energy"] - initial_energy) / abs(initial_energy)
        for row in rows
    )
    passed = (
        max(abs(fit["material"]["rate"] - mode["s"])
            for mode, fit in zip(modes, fits)) < rate_tolerance and
        max(abs(fit["radiation"]["rate"] - mode["s"])
            for mode, fit in zip(modes, fits)) < rate_tolerance and
        max(value for rms in rms_values for value in rms.values()) < rms_tolerance and
        energy_drift < energy_tolerance
    )
    if args.rate_relative_tolerance is not None:
        if args.rate_relative_tolerance <= 0.0:
            raise RuntimeError("--rate-relative-tolerance must be positive")
        passed = passed and max(relative_errors) < args.rate_relative_tolerance

    metrics = {
        "benchmark": "TRE-3D ensemble",
        "inputs": [str(path) for path in args.input],
        "ensemble_size": len(histories),
        "rate_window": [lo, hi],
        "modes": modes,
        "fits": fits,
        "relative_rate_errors": relative_errors,
        "normalized_rms": rms_values,
        "maximum_relative_energy_drift": energy_drift,
        "tolerances": {
            "rate_absolute": rate_tolerance,
            "rate_relative": args.rate_relative_tolerance,
            "normalized_rms": rms_tolerance,
            "energy_relative": energy_tolerance,
        },
        "per_run_fits": per_run_fits,
        "pass": passed,
        "metadata": metadata,
    }

    print("TRE-3D ensemble comparison")
    print(f"  histories = {len(histories)}")
    print(f"  fit window = [{lo:.6g}, {hi:.6g}]")
    print("  mode  exact s          ensemble material/error    ensemble radiation/error")
    for index, (mode, fit) in enumerate(zip(modes, fits), start=1):
        material = fit["material"]["rate"]
        radiation = fit["radiation"]["rate"]
        print(f"  {index:>2}  {mode['s']:+.12f}  "
              f"{material:+.9f}/{abs(material - mode['s']) / abs(mode['s']):.3e}  "
              f"{radiation:+.9f}/{abs(radiation - mode['s']) / abs(mode['s']):.3e}")

    print("  per-run relative rate errors (material/radiation):")
    for run_index, run_fits in enumerate(per_run_fits, start=1):
        values = []
        for mode, fit in zip(modes, run_fits):
            values.extend([
                abs(fit["material"]["rate"] - mode["s"]) / abs(mode["s"]),
                abs(fit["radiation"]["rate"] - mode["s"]) / abs(mode["s"]),
            ])
        print(f"    run {run_index}: " + " ".join(f"{value:.3e}" for value in values))

    print(f"  maximum relative rate error = {max(relative_errors):.3e}")
    print(f"  maximum relative energy drift = {energy_drift:.3e}")
    if args.rate_relative_tolerance is not None:
        print(f"  relative rate tolerance = {args.rate_relative_tolerance:.3e}")
    print(f"  TRE_ENSEMBLE_PASS = {'true' if passed else 'false'}")

    args.plot.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    single.make_plot(rows, modes, args.plot)
    args.json.write_text(json.dumps(metrics, indent=2) + "\n")

    if args.average_output is not None:
        args.average_output.parent.mkdir(parents=True, exist_ok=True)
        with args.average_output.open("w") as stream:
            for key, value in metadata.items():
                stream.write(f"# {key}={value}\n")
            columns = list(rows[0])
            stream.write(",".join(columns) + "\n")
            for row in rows:
                values = []
                for column in columns:
                    if column == "particle_count":
                        values.append(str(int(row[column])))
                    else:
                        values.append(f"{row[column]:.17e}")
                stream.write(",".join(values) + "\n")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
