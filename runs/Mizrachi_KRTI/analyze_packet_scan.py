#!/usr/bin/env python3
"""Fit the KRTI mode growth rate across runs that differ only in Monte Carlo packet count.

If the growth-rate deficit is Monte Carlo shot noise, the fitted G should climb toward the
reference root as the packet count rises. If it is a coupling or physics problem, G stays put.
"""

import argparse
import csv
import io
import math
from pathlib import Path

REFERENCE_G = 0.911691       # full radiation-hydro root, KRTI-S-X_parameters.json
CLASSICAL_EFFECTIVE_G = 0.706870  # classical with g_eff = g/2, no perturbed radiation


def read_history(path):
    lines = path.read_text().splitlines(keepends=True)
    body = "".join(line for line in lines if not line.startswith("#"))
    rows = list(csv.DictReader(io.StringIO(body)))
    times = [float(row["time_rt"]) for row in rows]
    amplitudes = [float(row["A_mode1"]) for row in rows]
    counts = [int(row["particle_count"]) for row in rows]
    return times, amplitudes, counts


def fit_growth(times, amplitudes, lower, upper):
    xs = []
    ys = []
    for time, amplitude in zip(times, amplitudes):
        if lower <= time <= upper and amplitude != 0.0:
            xs.append(time)
            ys.append(math.log(abs(amplitude)))
    if len(xs) < 3:
        return None
    count = len(xs)
    meanX = sum(xs) / count
    meanY = sum(ys) / count
    sxx = sum((value - meanX) ** 2 for value in xs)
    if sxx <= 0.0:
        return None
    slope = sum((xs[i] - meanX) * (ys[i] - meanY) for i in range(count)) / sxx
    residuals = [ys[i] - (meanY + slope * (xs[i] - meanX)) for i in range(count)]
    variance = sum(value * value for value in residuals) / max(1, count - 2)
    return slope, math.sqrt(variance / sxx), count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", metavar="LABEL=PATH",
                        help="packet count and history path, e.g. 600=results/10143432/krti_history.csv")
    parser.add_argument("--window", type=float, nargs=2, default=None,
                        help="fit window in t/t_RT; default is the common overlap")
    args = parser.parse_args()

    loaded = []
    for entry in args.runs:
        label, _, pathText = entry.partition("=")
        path = Path(pathText)
        if not path.exists():
            print(f"{label:>6}  (no history yet at {path})")
            continue
        times, amplitudes, counts = read_history(path)
        if len(times) < 3:
            print(f"{label:>6}  (only {len(times)} samples so far)")
            continue
        loaded.append((label, times, amplitudes, counts))

    if not loaded:
        return

    if args.window is not None:
        lower, upper = args.window
    else:
        lower = 0.0
        upper = min(times[-1] for _, times, _, _ in loaded)

    print(f"fit window: t/t_RT in [{lower:.4f}, {upper:.4f}]")
    print()
    print(f"{'packets':>8} {'samples':>8} {'G':>9} {'+/-':>8} {'A/A0':>9} {'frac of response':>17}")
    for label, times, amplitudes, counts in loaded:
        result = fit_growth(times, amplitudes, lower, upper)
        if result is None:
            print(f"{label:>8}  insufficient samples in window")
            continue
        slope, error, count = result
        fraction = (slope - CLASSICAL_EFFECTIVE_G) / (REFERENCE_G - CLASSICAL_EFFECTIVE_G)
        print(f"{label:>8} {count:>8} {slope:>9.4f} {error:>8.4f} "
              f"{abs(amplitudes[-1] / amplitudes[0]):>9.5f} {100.0 * fraction:>16.1f}%")

    print()
    print(f"{'reference':>8} {'':>8} {REFERENCE_G:>9.4f}  (full radiation-hydro root)")
    print(f"{'no-pert':>8} {'':>8} {CLASSICAL_EFFECTIVE_G:>9.4f}  (classical with g_eff, zero perturbed radiation)")


if __name__ == "__main__":
    main()
