#!/usr/bin/env python3
"""Compare a STORM TRE-3D history with the semi-analytic solution.

STORM's radiation observable is a track-length average over one timestep, so
each radiation mode is compared at ``tau_mid``.  Material modes are census
values and are compared at ``tau_end``.  The energy diagnostic uses the
post-population-control census radiation energy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CHI = 1.0
BETA = 1.0
BACKGROUND = 1.0
WAVE_VECTORS = ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0))
AMPLITUDES = (0.10, 0.07, 0.05)


def exact_decay_rate(k: float, chi: float = CHI, beta: float = BETA) -> float:
    def dispersion(s: float) -> float:
        a = chi + s
        g = (chi / k) * math.atan(k / a)
        return s - beta * chi * (g - 1.0)

    lo = -chi + 1.0e-14
    hi = -1.0e-14
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if dispersion(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def read_history(path: Path):
    metadata = {}
    data_lines = []
    with path.open() as stream:
        for line in stream:
            if line.startswith("#"):
                text = line[1:].strip()
                if "=" in text:
                    key, value = text.split("=", 1)
                    metadata[key.strip()] = value.strip()
            elif line.strip():
                data_lines.append(line)
    rows = list(csv.DictReader(data_lines))
    if not rows:
        raise RuntimeError(f"no data rows found in {path}")
    converted = []
    for row in rows:
        converted.append({
            key: float(value) if key != "particle_count" else int(float(value))
            for key, value in row.items()
        })
    return metadata, converted


def linear_fit(x, y):
    if len(x) < 2:
        raise RuntimeError("not enough points for a decay-rate fit")
    xbar = sum(x) / len(x)
    ybar = sum(y) / len(y)
    denominator = sum((value - xbar) ** 2 for value in x)
    if denominator <= 0.0:
        raise RuntimeError("zero time span in decay-rate fit")
    slope = sum((x[i] - xbar) * (y[i] - ybar)
                for i in range(len(x))) / denominator
    intercept = ybar - slope * xbar
    residual = math.sqrt(sum((y[i] - (intercept + slope * x[i])) ** 2
                             for i in range(len(x))) / len(x))
    return slope, intercept, residual


def fit_rate(rows, time_key, amplitude_key, lo, hi):
    selected = [row for row in rows
                if lo <= row[time_key] <= hi and abs(row[amplitude_key]) > 1.0e-14]
    x = [row[time_key] for row in selected]
    y = [math.log(abs(row[amplitude_key])) for row in selected]
    slope, intercept, residual = linear_fit(x, y)
    return {"rate": slope, "intercept": intercept, "log_rms": residual,
            "points": len(selected)}


def rms_normalized(rows, time_key, amplitude_key, expected_factor, rate, lo, hi):
    selected = [row for row in rows if lo <= row[time_key] <= hi]
    if not selected:
        raise RuntimeError(f"no points in comparison interval for {amplitude_key}")
    errors = []
    for row in selected:
        expected = expected_factor * math.exp(rate * row[time_key])
        errors.append((row[amplitude_key] - expected) / expected_factor)
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def nearest_row(rows, key, target):
    return min(rows, key=lambda row: abs(row[key] - target))


def exact_modes():
    modes = []
    for wave_vector, amplitude in zip(WAVE_VECTORS, AMPLITUDES):
        k = math.sqrt(sum(component * component for component in wave_vector))
        s = exact_decay_rate(k)
        g = (CHI / k) * math.atan(k / (CHI + s))
        modes.append({"k": k, "s": s, "G": g, "amplitude": amplitude})
    return modes


def make_plot(rows, modes, output):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("plot skipped: matplotlib is not installed")
        return

    figure, axes = plt.subplots(2, 2, figsize=(11, 8.0), sharex=False)
    for index, mode in enumerate(modes):
        axis = axes.flat[index]
        tau_u = [row["tau_end"] for row in rows]
        tau_e = [row["tau_mid"] for row in rows]
        au = [row[f"A_U{index + 1}"] / mode["amplitude"] for row in rows]
        ae = [row[f"A_E{index + 1}"] /
              (mode["amplitude"] * mode["G"]) for row in rows]
        exact_u = [math.exp(mode["s"] * value) for value in tau_u]
        exact_e = [math.exp(mode["s"] * value) for value in tau_e]
        axis.plot(tau_u, au, "o", ms=2.2, label="STORM material")
        axis.plot(tau_e, ae, ".", ms=2.0, label="STORM radiation")
        axis.plot(tau_u, exact_u, "k--", lw=1.3, label="semi-analytic")
        k = ",".join(str(int(value)) for value in WAVE_VECTORS[index])
        axis.set_title(f"mode {index + 1}: k=({k}), |k|={mode['k']:.3g}")
        axis.set_xlabel(r"$\tau=ct/L_0$")
        axis.set_ylabel("normalized amplitude")
        axis.grid(alpha=0.25)
        if index == 0:
            axis.legend(frameon=False, fontsize=8)

    initial_energy = rows[0]["total_energy"]
    drift = [(row["total_energy"] - initial_energy) / initial_energy
             for row in rows]
    axis = axes.flat[3]
    axis.plot([row["tau_end"] for row in rows], drift, color="tab:purple")
    axis.axhline(0.0, color="k", lw=0.8)
    axis.set_xlabel(r"$\tau=ct/L_0$")
    axis.set_ylabel("relative total-energy drift")
    axis.set_title("conservation diagnostic")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("tre_3d_history.csv"))
    parser.add_argument("--plot", type=Path, default=None,
                        help="write a PNG comparison (default: next to input)")
    parser.add_argument("--json", type=Path, default=None,
                        help="write machine-readable metrics")
    parser.add_argument("--rate-window", nargs=2, type=float, default=None,
                        metavar=("TAU_MIN", "TAU_MAX"))
    parser.add_argument("--rate-relative-tolerance", type=float, default=None,
                        help="optionally require every fitted rate to meet this relative error")
    args = parser.parse_args()

    metadata, rows = read_history(args.input)
    modes = exact_modes()
    final_tau = rows[-1]["tau_end"]
    # The third mode is intentionally the smallest and fastest-decaying one.
    # Use a common early/mid-time interval where its signal is still well
    # above the Monte Carlo floor; the interval remains long enough for a
    # stable regression and is overrideable for convergence studies.
    lo, hi = args.rate_window or (max(0.5, 0.10 * final_tau),
                                  min(0.90 * final_tau, 0.50 * final_tau))

    fits = []
    rms_values = []
    for index, mode in enumerate(modes, start=1):
        material_fit = fit_rate(rows, "tau_end", f"A_U{index}", lo, hi)
        radiation_fit = fit_rate(rows, "tau_mid", f"A_E{index}", lo, hi)
        material_rms = rms_normalized(
            rows, "tau_end", f"A_U{index}", mode["amplitude"],
            mode["s"], lo, hi)
        radiation_rms = rms_normalized(
            rows, "tau_mid", f"A_E{index}", mode["amplitude"] * mode["G"],
            mode["s"], lo, hi)
        fits.append({"material": material_fit, "radiation": radiation_fit})
        rms_values.append({"material": material_rms, "radiation": radiation_rms})

    initial_energy = rows[0]["total_energy"]
    energy_drift = max(abs(row["total_energy"] - initial_energy) /
                       abs(initial_energy) for row in rows)
    rate_tolerance = 0.045
    rms_tolerance = 0.10
    energy_tolerance = 0.02
    rate_errors = []
    relative_rate_errors = []
    for mode, fit in zip(modes, fits):
        rate_errors.extend([
            abs(fit["material"]["rate"] - mode["s"]),
            abs(fit["radiation"]["rate"] - mode["s"]),
        ])
        relative_rate_errors.extend([
            abs(fit["material"]["rate"] - mode["s"]) / abs(mode["s"]),
            abs(fit["radiation"]["rate"] - mode["s"]) / abs(mode["s"]),
        ])
    passed = (
        max(rate_errors) < rate_tolerance and
        max(value for rms in rms_values for value in rms.values()) < rms_tolerance and
        energy_drift < energy_tolerance
    )
    if args.rate_relative_tolerance is not None:
        if args.rate_relative_tolerance <= 0.0:
            raise RuntimeError("--rate-relative-tolerance must be positive")
        passed = passed and max(relative_rate_errors) < args.rate_relative_tolerance

    metrics = {
        "benchmark": "TRE-3D",
        "modes": modes,
        "rate_window": [lo, hi],
        "fits": fits,
        "relative_rate_errors": relative_rate_errors,
        "normalized_rms": rms_values,
        "maximum_relative_energy_drift": energy_drift,
        "tolerances": {
            "rate_absolute": rate_tolerance,
            "rate_relative": args.rate_relative_tolerance,
            "normalized_rms": rms_tolerance,
            "energy_relative": energy_tolerance,
        },
        "pass": passed,
        "metadata": metadata,
    }

    print("TRE-3D comparison: three-mode full-transport eigenmode")
    print(f"  fit window = [{lo:.6g}, {hi:.6g}]")
    print("  mode  k       exact s          material fit/error       radiation fit/error")
    for index, (mode, fit) in enumerate(zip(modes, fits), start=1):
        material = fit["material"]["rate"]
        radiation = fit["radiation"]["rate"]
        print(f"  {index:>2}  {mode['k']:.6g}  {mode['s']:+.12f}  "
              f"{material:+.9f}/{abs(material - mode['s']):.3e}  "
              f"{radiation:+.9f}/{abs(radiation - mode['s']):.3e}")
        print(f"      normalized RMS material/radiation = "
              f"{rms_values[index - 1]['material']:.3e}/"
              f"{rms_values[index - 1]['radiation']:.3e}")
    print(f"  maximum relative energy drift = {energy_drift:.3e}")
    print(f"  maximum relative rate error = {max(relative_rate_errors):.3e}")
    if args.rate_relative_tolerance is not None:
        print(f"  relative rate tolerance = {args.rate_relative_tolerance:.3e}")
    print(f"  TRE_PASS = {'true' if passed else 'false'}")

    for target in (0.0, 1.0, 2.0, 5.0, final_tau):
        print(f"  tau={target:g}:")
        for index, mode in enumerate(modes, start=1):
            material_row = nearest_row(rows, "tau_end", target)
            radiation_row = nearest_row(rows, "tau_mid", target)
            material_exact = mode["amplitude"] * math.exp(
                mode["s"] * material_row["tau_end"])
            radiation_exact = mode["amplitude"] * mode["G"] * math.exp(
                mode["s"] * radiation_row["tau_mid"])
            print(f"    mode {index}: A_U={material_row[f'A_U{index}']:.8e} "
                  f"(exact {material_exact:.8e}), "
                  f"A_E={radiation_row[f'A_E{index}']:.8e} "
                  f"(exact {radiation_exact:.8e})")

    plot_path = args.plot or args.input.with_name(args.input.stem + "_comparison.png")
    make_plot(rows, modes, plot_path)
    if args.json:
        args.json.write_text(json.dumps(metrics, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
