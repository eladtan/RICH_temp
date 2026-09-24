#!/usr/bin/env python3
"""Semi-analytic Jiang-Davis-Stone G3 reference and RICH history comparison.

The paper's Appendix shows that, to leading order in C^{-1} and with a diagonal
Eddington tensor, the incompressible interface mode has the classical growth
rate

    n = sqrt(g k (rho_+ - rho_-) / (rho_+ + rho_-)).

For G3 that is k = 2 pi, g = 0.1, rho_+/rho_- = 4, so n ~= 0.614 (quoted as
0.61 in section 4.1). The optically thin alpha=1 limit would instead have
g_eff = 0 and n = 0. This script reports both, plus the hydrostatic radiation
energy profile used by the G3 analogue, and compares a RICH history CSV to
those references and to the paper's G3 output times.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


G = 0.1
K = 2.0 * math.pi
RHO_MINUS = 1.0
RHO_PLUS = 4.0
KAPPA = 1.0
LIGHT_SPEED = 1.0e4
GAS_PRESSURE = 1.0
FRZ = 0.1
ER_BOTTOM = 1.72
ER_INTERFACE = 1.42
ER_TOP = 0.22
BOX_X = 1.0
BOX_Y = 1.0
BOX_Z = 2.0
PAPER_N = 0.61
PAPER_G3_TIMES = (21.6, 28.9)
PAPER_G2_TIMES = (15.2, 21.6)
PAPER_LINEAR_ZMAX = 0.02


def atwood_number(rho_plus=RHO_PLUS, rho_minus=RHO_MINUS):
    return (rho_plus - rho_minus) / (rho_plus + rho_minus)


def classical_growth_rate(g=G, k=K, rho_plus=RHO_PLUS, rho_minus=RHO_MINUS):
    return math.sqrt(g * k * atwood_number(rho_plus, rho_minus))


def radiation_energy_density(z):
    if z < 0.0:
        return ER_BOTTOM - 0.3 * (z + 1.0)
    return ER_INTERFACE - 1.2 * z


def hydrostatic_radiation_energy():
    samples = 2001
    dz = BOX_Z / (samples - 1)
    integral = 0.0
    for index in range(samples):
        z = -1.0 + index * dz
        weight = 0.5 if index == 0 or index == samples - 1 else 1.0
        integral += weight * radiation_energy_density(z)
    return BOX_X * BOX_Y * integral * dz


def radiation_drag_frequency(radiation_pressure, kappa=KAPPA, p_parameter=1.0,
                             light_speed=LIGHT_SPEED):
    return 4.0 * radiation_pressure * kappa * p_parameter / light_speed


def semi_analytic_g3():
    growth_rate = classical_growth_rate()
    interface_pr_eddington = ER_INTERFACE / 3.0
    return {
        "benchmark": "Jiang_Davis_Stone_G3",
        "citation": "Jiang, Davis & Stone 2013, ApJ 763, 102, Appendix and sec. 4.1",
        "g": G,
        "k": K,
        "rho_minus": RHO_MINUS,
        "rho_plus": RHO_PLUS,
        "atwood": atwood_number(),
        "kappa": KAPPA,
        "alpha": 1.0,
        "light_speed": LIGHT_SPEED,
        "Frz": FRZ,
        "Er_bottom": ER_BOTTOM,
        "Er_interface": ER_INTERFACE,
        "Er_top": ER_TOP,
        "dEr_dz_minus": -0.3,
        "dEr_dz_plus": -1.2,
        "total_radiation_energy": hydrostatic_radiation_energy(),
        "classical_growth_rate": growth_rate,
        "paper_quoted_growth_rate": PAPER_N,
        "optically_thin_geff_growth_rate": 0.0,
        "interface_eddington_Pr": interface_pr_eddington,
        "radiation_drag_frequency_interface": radiation_drag_frequency(
            interface_pr_eddington),
        "paper_g3_times": list(PAPER_G3_TIMES),
        "paper_g2_times": list(PAPER_G2_TIMES),
        "paper_linear_zmax_limit": PAPER_LINEAR_ZMAX,
        "paper_g3_notes": (
            "G3 (VET) grows more slowly and mixes less than G2 (Eddington). "
            "The paper reports G3 density snapshots at t=21.6 and t=28.9. "
            "No tabulated zmax/mixing numbers are published; comparison at "
            "those times is against the classical exponential and the stated "
            "qualitative G3/G2 trend."
        ),
    }


def read_history(path: Path):
    metadata = {}
    rows = []
    with path.open() as stream:
        for line in stream:
            if line.startswith("#"):
                text = line[1:].strip()
                if "=" in text:
                    key, value = text.split("=", 1)
                    metadata[key.strip()] = value.strip()
            elif line.strip():
                rows.append(line)
    parsed = list(csv.DictReader(rows))
    if not parsed:
        raise RuntimeError(f"no data rows found in {path}")
    converted = []
    for row in parsed:
        item = {}
        for key, value in row.items():
            if key == "particle_count":
                item[key] = int(float(value))
            else:
                item[key] = float(value)
        converted.append(item)
    return metadata, converted


def nearest_row(rows, time):
    return min(rows, key=lambda row: abs(row["time"] - time))


def linear_fit(x, y):
    if len(x) < 2:
        raise RuntimeError("not enough points for growth-rate fit")
    xbar = sum(x) / len(x)
    ybar = sum(y) / len(y)
    denom = sum((value - xbar) ** 2 for value in x)
    if denom <= 0.0:
        raise RuntimeError("degenerate fit window")
    slope = sum((x[index] - xbar) * (y[index] - ybar) for index in range(len(x))) / denom
    intercept = ybar - slope * xbar
    residual = math.sqrt(sum((y[index] - (intercept + slope * x[index])) ** 2
                             for index in range(len(x))) / len(x))
    return slope, intercept, residual


SEED_ENVELOPE = 0.45
FIT_AMPLITUDE_MIN = PAPER_LINEAR_ZMAX
FIT_AMPLITUDE_MAX = 0.5


def uses_seed_envelope(rows):
    return abs(rows[0]["zmax"]) >= SEED_ENVELOPE


def displacement_amplitude(row, key, z0):
    value = abs(row[key])
    if z0 >= SEED_ENVELOPE:
        return max(0.0, value - z0)
    return value


def fit_displacement(rows, key, t_min, t_max, amplitude_min, amplitude_max):
    z0 = abs(rows[0][key])
    selected = []
    for row in rows:
        amplitude = displacement_amplitude(row, key, z0)
        if t_min <= row["time"] <= t_max and amplitude_min <= amplitude <= amplitude_max:
            selected.append((row["time"], amplitude))
    if len(selected) < 4:
        raise RuntimeError(f"not enough samples to fit {key}")
    x = [item[0] for item in selected]
    y = [math.log(item[1]) for item in selected]
    slope, intercept, residual = linear_fit(x, y)
    return {
        "rate": slope,
        "intercept": intercept,
        "log_rms": residual,
        "points": len(selected),
        "t_min": selected[0][0],
        "t_max": selected[-1][0],
        "amplitude_definition": (
            "penetration past t=0 10% envelope" if z0 >= SEED_ENVELOPE
            else "|rho-rho_initial|>=0.1 rho_initial zmax"
        ),
    }


def resolve_fit_window(rows, t_min, t_max):
    times = [row["time"] for row in rows]
    t_lo = min(times)
    t_hi = max(times)
    if t_hi <= t_lo:
        raise RuntimeError("degenerate simulation time range")
    if sum(t_min <= row["time"] <= t_max for row in rows) >= 4:
        return t_min, t_max
    span = t_hi - t_lo
    return t_lo + 0.15 * span, t_hi - 0.05 * span


def paper_time_comparison(rows, reference):
    comparison = []
    for time in reference["paper_g3_times"]:
        row = nearest_row(rows, time)
        comparison.append({
            "paper_time": time,
            "simulation_time": row["time"],
            "time_offset": row["time"] - time,
            "zmax": row["zmax"],
            "zmin": row["zmin"],
            "mixing_fraction": row["mixing_fraction"],
            "Frz": row.get("Frz"),
            "reached": abs(row["time"] - time) <= 0.05 * max(time, 1.0),
        })
    return comparison


def compare_history(rows, metadata, t_min, t_max):
    reference = semi_analytic_g3()
    window = resolve_fit_window(rows, t_min, t_max)
    fits = {}
    for key in ("zmax", "zmin"):
        try:
            fits[key] = fit_displacement(
                rows, key, window[0], window[1], FIT_AMPLITUDE_MIN, FIT_AMPLITUDE_MAX)
        except RuntimeError:
            continue
    last = rows[-1]
    first = rows[0]
    metrics = {
        "benchmark": "Jiang_Davis_Stone_G3_IMC_analogue",
        "history_benchmark": metadata.get("benchmark"),
        "reference": reference,
        "fit_window": list(window),
        "fits": fits,
        "seed_envelope_history": uses_seed_envelope(rows),
        "initial": {
            "time": first["time"],
            "zmax": first["zmax"],
            "zmin": first["zmin"],
            "mixing_fraction": first["mixing_fraction"],
            "total_radiation_energy": first["total_radiation_energy"],
        },
        "final": {
            "time": last["time"],
            "zmax": last["zmax"],
            "zmin": last["zmin"],
            "mixing_fraction": last["mixing_fraction"],
            "total_radiation_energy": last["total_radiation_energy"],
            "Frz": last.get("Frz"),
            "particle_count": last.get("particle_count"),
        },
        "paper_times": paper_time_comparison(rows, reference),
        "initial_radiation_energy_relative_error":
            abs(first["total_radiation_energy"] - reference["total_radiation_energy"])
            / reference["total_radiation_energy"],
        "pass": None,
        "notes": (
            "G3 is initialized with a large random density perturbation in "
            "|z|<0.5. Histories that compare density to the two-layer base "
            "pin zmax at that seed envelope; do not fit ln(zmax) there. "
            "Newer histories compare to the Eq.15 initial density, so zmax "
            "starts near 0. Fit ln(A) only for 0.02<=A<=0.5 after the 10% "
            "contour has left the noise floor. The classical e^{nt} overlay "
            "uses A0=0.02, the paper linear-regime amplitude, not zmax(0). "
            "The paper does not tabulate G3 zmax or mixing values."
        ),
    }
    if "zmax" in fits:
        metrics["simulation_growth_rate"] = fits["zmax"]["rate"]
        metrics["relative_difference_from_classical"] = (
            abs(fits["zmax"]["rate"] - reference["classical_growth_rate"])
            / reference["classical_growth_rate"]
        )
        metrics["relative_difference_from_paper_quoted"] = (
            abs(fits["zmax"]["rate"] - PAPER_N) / PAPER_N
        )
    else:
        metrics["simulation_growth_rate"] = None
        metrics["relative_difference_from_classical"] = None
        metrics["relative_difference_from_paper_quoted"] = None
    return metrics


def write_plot(path: Path, rows, metrics):
    import matplotlib.pyplot as plt

    times = [row["time"] for row in rows]
    zmax = [row["zmax"] for row in rows]
    zmin = [abs(row["zmin"]) for row in rows]
    mixing = [row["mixing_fraction"] for row in rows]
    reference = metrics["reference"]
    n = reference["classical_growth_rate"]
    z0 = abs(zmax[0])
    linear_reference = [PAPER_LINEAR_ZMAX * math.exp(n * time) for time in times]

    figure, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    axes[0].plot(times, zmax, "o", ms=2.4, label=r"simulation $z_{\max}$")
    axes[0].plot(times, zmin, "s", ms=2.4, label=r"simulation $|z_{\min}|$")
    if z0 >= SEED_ENVELOPE:
        penetration = [max(0.0, value - z0) for value in zmax]
        axes[0].plot(times, [value if value > 0.0 else float("nan") for value in penetration],
                     "^-", ms=2.4, label=r"penetration $z_{\max}(t)-z_{\max}(0)$")
    axes[0].plot(times, linear_reference, "k--",
                 label=fr"linear reference $A_0 e^{{nt}}$, $A_0={PAPER_LINEAR_ZMAX}$, $n={n:.3f}$")
    axes[0].plot(times, [PAPER_LINEAR_ZMAX * math.exp(PAPER_N * time) for time in times],
                 color="tab:green", ls=":",
                 label=fr"paper quoted $n={PAPER_N}$, $A_0={PAPER_LINEAR_ZMAX}$")
    # Drawing the paper markers before the run reaches them stretches the axis
    # to t=28.9 and squeezes the whole history into the first few percent.
    for time in PAPER_G3_TIMES:
        if time <= times[-1]:
            axes[0].axvline(time, color="0.6", ls="--", lw=0.8)
            axes[1].axvline(time, color="0.6", ls="--", lw=0.8)
    axes[0].set_ylabel("displacement")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="best")

    axes[1].plot(times, mixing, color="tab:orange", label="mixing fraction")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("mixing fraction")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        help="jds_g3_history.csv from a RICH G3 run")
    parser.add_argument("--json", type=Path, help="write comparison metrics here")
    parser.add_argument("--plot", type=Path, help="optional comparison figure")
    parser.add_argument("--reference-only", action="store_true",
                        help="print the G3 semi-analytic reference and exit")
    parser.add_argument("--rate-window", nargs=2, type=float,
                        metavar=("T_MIN", "T_MAX"), default=(2.0, 12.0),
                        help="time window used to fit ln(zmax)")
    args = parser.parse_args()

    reference = semi_analytic_g3()
    if args.reference_only:
        print(json.dumps(reference, indent=2))
        print(f"classical n={reference['classical_growth_rate']:.9f}  "
              f"paper quoted n={PAPER_N}")
        print(f"G3 hydrostatic radiation energy={reference['total_radiation_energy']:.9f}")
        return

    if args.input is None:
        raise SystemExit("--input is required unless --reference-only is set")

    metadata, rows = read_history(args.input)
    metrics = compare_history(rows, metadata, args.rate_window[0], args.rate_window[1])

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metrics, indent=2) + "\n")

    if args.plot is not None:
        try:
            write_plot(args.plot, rows, metrics)
        except ImportError:
            print("plot skipped: matplotlib is not installed")

    print(f"classical n={reference['classical_growth_rate']:.6f}, "
          f"paper quoted n={PAPER_N}")
    if metrics["simulation_growth_rate"] is None:
        print("no zmax growth-rate fit: need >=4 samples with 0.02<=A<=0.5 "
              "(A is zmax for |rho-rho_init| histories, or zmax-zmax(0) for "
              "seed-envelope histories)")
    else:
        print(f"simulation n_zmax={metrics['simulation_growth_rate']:.6f}, "
              f"relative difference from classical="
              f"{metrics['relative_difference_from_classical']:.3%}")
    print("initial Er relative error="
          f"{metrics['initial_radiation_energy_relative_error']:.3%}")
    if rows and "Frz" in rows[0]:
        print(f"initial census Frz={rows[0]['Frz']:.6f} (paper background 0.1)")
    for item in metrics["paper_times"]:
        status = "hit" if item["reached"] else "missing"
        print(f"paper t={item['paper_time']}: {status}  "
              f"sim t={item['simulation_time']:.4f}  "
              f"zmax={item['zmax']:.4f}  zmin={item['zmin']:.4f}  "
              f"mixing={item['mixing_fraction']:.4f}"
              + (f"  Frz={item['Frz']:.4f}" if item.get("Frz") is not None else ""))
    print("No pass/fail is assigned: G3 is a nonlinear VET Athena problem; "
          "this is an IMC analogue compared to the classical linear rate.")


if __name__ == "__main__":
    main()
