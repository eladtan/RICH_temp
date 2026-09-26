#!/usr/bin/env python3
"""Analyze a STORM TRE-1 run against the semi-analytic eigenmode.

The radiation amplitude in the CSV is a track-length average over one
timestep, hence it is compared at tau_mid.  The material amplitude is a
census/material value and is compared at tau_end.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CHI = 1.0
BETA = 1.0
AMPLITUDE = 0.10
WAVE_NUMBER = 1.0


def exact_decay_rate(k: float = WAVE_NUMBER, chi: float = CHI, beta: float = BETA) -> float:
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
        converted.append({key: float(value) if key != "particle_count" else int(value)
                          for key, value in row.items()})
    return metadata, converted


def linear_fit(x, y):
    n = len(x)
    if n < 2:
        raise RuntimeError("not enough points for a decay-rate fit")
    xbar = sum(x) / n
    ybar = sum(y) / n
    denominator = sum((value - xbar) ** 2 for value in x)
    if denominator <= 0.0:
        raise RuntimeError("zero time span in decay-rate fit")
    slope = sum((x[i] - xbar) * (y[i] - ybar) for i in range(n)) / denominator
    intercept = ybar - slope * xbar
    residual = math.sqrt(sum((y[i] - (intercept + slope * x[i])) ** 2 for i in range(n)) / n)
    return slope, intercept, residual


def fit_rate(rows, time_key, amplitude_key, lo, hi):
    selected = [row for row in rows if lo <= row[time_key] <= hi and abs(row[amplitude_key]) > 1.0e-12]
    x = [row[time_key] for row in selected]
    y = [math.log(abs(row[amplitude_key])) for row in selected]
    slope, intercept, residual = linear_fit(x, y)
    return {"rate": slope, "intercept": intercept, "log_rms": residual, "points": len(selected)}


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


def make_plot(rows, exact_s, exact_g, output):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("plot skipped: matplotlib is not installed")
        return

    tau_u = [row["tau_end"] for row in rows]
    tau_e = [row["tau_mid"] for row in rows]
    au = [row["A_U"] / AMPLITUDE for row in rows]
    ae = [row["A_E"] / (AMPLITUDE * exact_g) for row in rows]
    exact_u = [math.exp(exact_s * value) for value in tau_u]
    exact_e = [math.exp(exact_s * value) for value in tau_e]
    initial_energy = rows[0]["total_energy"]
    drift = [(row["total_energy"] - initial_energy) / initial_energy for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(tau_u, au, "o", ms=3, label="STORM material")
    axes[0].plot(tau_e, ae, ".", ms=3, label="STORM radiation")
    axes[0].plot(tau_u, exact_u, "k--", lw=1.5, label="semi-analytic")
    axes[0].set_xlabel(r"$\tau=ct/L_0$")
    axes[0].set_ylabel("normalized mode amplitude")
    axes[0].set_title("TRE-1 modal decay")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].plot(tau_u, drift, color="tab:purple")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_xlabel(r"$\tau=ct/L_0$")
    axes[1].set_ylabel("relative total-energy drift")
    axes[1].set_title("conservation diagnostic")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("tre_1d_history.csv"))
    parser.add_argument("--plot", type=Path, default=None,
                        help="write a PNG comparison (default: next to input)")
    parser.add_argument("--json", type=Path, default=None,
                        help="write machine-readable metrics")
    parser.add_argument("--rate-window", nargs=2, type=float, default=None,
                        metavar=("TAU_MIN", "TAU_MAX"))
    args = parser.parse_args()

    metadata, rows = read_history(args.input)
    exact_s = exact_decay_rate()
    exact_g = (CHI / WAVE_NUMBER) * math.atan(WAVE_NUMBER / (CHI + exact_s))
    final_tau = rows[-1]["tau_end"]
    lo, hi = args.rate_window or (max(0.25, 0.15 * final_tau), 0.90 * final_tau)
    material_fit = fit_rate(rows, "tau_end", "A_U", lo, hi)
    radiation_fit = fit_rate(rows, "tau_mid", "A_E", lo, hi)
    rms_u = rms_normalized(rows, "tau_end", "A_U", AMPLITUDE, exact_s, lo, hi)
    rms_e = rms_normalized(rows, "tau_mid", "A_E", AMPLITUDE * exact_g, exact_s, lo, hi)

    initial_energy = rows[0]["total_energy"]
    energy_drift = max(abs(row["total_energy"] - initial_energy) / abs(initial_energy)
                       for row in rows)
    # The tolerances are deliberately explicit: this is a noisy MC acceptance
    # check, while dtau/spatial/history convergence should tighten them.
    rate_tolerance = 0.020
    rms_tolerance = 0.10
    energy_tolerance = 0.02
    passed = (
        abs(material_fit["rate"] - exact_s) < rate_tolerance and
        abs(radiation_fit["rate"] - exact_s) < rate_tolerance and
        rms_u < rms_tolerance and rms_e < rms_tolerance and
        energy_drift < energy_tolerance
    )

    metrics = {
        "benchmark": "TRE-1",
        "exact_decay_rate": exact_s,
        "exact_G": exact_g,
        "rate_window": [lo, hi],
        "material_fit": material_fit,
        "radiation_fit": radiation_fit,
        "material_normalized_rms": rms_u,
        "radiation_normalized_rms": rms_e,
        "maximum_relative_energy_drift": energy_drift,
        "tolerances": {
            "rate_absolute": rate_tolerance,
            "normalized_rms": rms_tolerance,
            "energy_relative": energy_tolerance,
        },
        "pass": passed,
        "metadata": metadata,
    }

    print("TRE-1 comparison: semi-analytic full-transport eigenmode")
    print(f"  exact s = {exact_s:.15f}")
    print(f"  exact G = {exact_g:.15f}")
    print(f"  fit window = [{lo:.6g}, {hi:.6g}]")
    print("  observable          fitted s             abs error")
    print(f"  material U       {material_fit['rate']:+.9f}   {abs(material_fit['rate'] - exact_s):.3e}")
    print(f"  radiation E      {radiation_fit['rate']:+.9f}   {abs(radiation_fit['rate'] - exact_s):.3e}")
    print(f"  material normalized RMS = {rms_u:.3e}")
    print(f"  radiation normalized RMS = {rms_e:.3e}")
    print(f"  maximum relative energy drift = {energy_drift:.3e}")
    print(f"  TRE_PASS = {'true' if passed else 'false'}")

    for target in (0.0, 1.0, 2.0, 5.0, final_tau):
        material_row = nearest_row(rows, "tau_end", target)
        radiation_row = nearest_row(rows, "tau_mid", target)
        print(f"  tau={target:g}: A_U={material_row['A_U']:.8e} "
              f"(exact {AMPLITUDE * math.exp(exact_s * material_row['tau_end']):.8e}), "
              f"A_E={radiation_row['A_E']:.8e} "
              f"(exact {AMPLITUDE * exact_g * math.exp(exact_s * radiation_row['tau_mid']):.8e})")

    plot_path = args.plot or args.input.with_name(args.input.stem + "_comparison.png")
    make_plot(rows, exact_s, exact_g, plot_path)
    if args.json:
        args.json.write_text(json.dumps(metrics, indent=2) + "\n")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
