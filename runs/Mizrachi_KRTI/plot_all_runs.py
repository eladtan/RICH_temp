#!/usr/bin/env python3
"""Overlay multiple KRTI histories against the frozen semi-analytic mode on one figure."""

import argparse
import csv
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATWOOD = 0.5
GRAVITY = 1.0e8
WAVE_NUMBER = 4.0
ETA0 = 2.5e-4
ROOT_S = 12893.260900062034
LINEAR_LIMIT = 0.05 / WAVE_NUMBER


def read_history(path):
    lines = path.read_text().splitlines(keepends=True)
    header = {}
    for line in lines:
        if line.startswith("#") and "=" in line:
            key, _, value = line[1:].strip().partition("=")
            header[key.strip()] = value.strip()
    body = "".join(line for line in lines if not line.startswith("#"))
    rows = list(csv.DictReader(io.StringIO(body)))
    times = [float(row["time_rt"]) for row in rows]
    amplitudes = [float(row["A_mode1"]) for row in rows]
    return times, amplitudes, header


def fit_growth(times, amplitudes, lower=0.0):
    xs = []
    ys = []
    for time, amplitude in zip(times, amplitudes):
        if time >= lower and amplitude != 0.0:
            xs.append(time)
            ys.append(math.log(abs(amplitude)))
    if len(xs) < 3:
        return float("nan"), float("nan")
    meanX = sum(xs) / len(xs)
    meanY = sum(ys) / len(ys)
    sxx = sum((value - meanX) ** 2 for value in xs)
    if sxx <= 0.0:
        return float("nan"), float("nan")
    slope = sum((xs[i] - meanX) * (ys[i] - meanY) for i in range(len(xs))) / sxx
    residuals = [ys[i] - (meanY + slope * (xs[i] - meanX)) for i in range(len(xs))]
    error = math.sqrt(sum(value * value for value in residuals) / max(1, len(xs) - 2) / sxx)
    return slope, error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", metavar="LABEL=PATH",
                        help="e.g. 600=results/10143455/krti_history.csv")
    parser.add_argument("--out", type=Path, default=Path("results/krti_all_runs.png"))
    parser.add_argument("--target", type=float, default=0.2, help="planned stop t/t_RT")
    args = parser.parse_args()

    growth = ROOT_S / math.sqrt(ATWOOD * GRAVITY * WAVE_NUMBER)
    colors = ["tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:pink"]
    series = []
    maxTime = 0.0
    nz = 42

    for index, entry in enumerate(args.runs):
        label, _, pathText = entry.partition("=")
        path = Path(pathText)
        times, amplitudes, header = read_history(path)
        nz = int(header.get("nz", nz))
        maxTime = max(maxTime, times[-1])
        fitSlope, fitError = fit_growth(times, amplitudes, lower=0.01)
        series.append({
            "label": label,
            "times": times,
            "amplitudes": amplitudes,
            "color": colors[index % len(colors)],
            "fit": fitSlope,
            "fitError": fitError,
        })

    refTimes = [maxTime * value / 200.0 for value in range(201)]
    refAbs = [ETA0 * math.exp(growth * time) for time in refTimes]
    refRel = [math.exp(growth * time) for time in refTimes]
    cellHeight = 2.0 / nz

    fig, (left, right) = plt.subplots(1, 2, figsize=(13.0, 5.2))

    for item in series:
        label = f"{item['label']} packets/cell"
        if math.isfinite(item["fit"]):
            label += fr"  ($G={item['fit']:.3f}\pm{item['fitError']:.3f}$)"
        left.plot(item["times"], [abs(value) for value in item["amplitudes"]],
                  color=item["color"], lw=1.5, label=label)
        right.plot(item["times"],
                   [abs(value / item["amplitudes"][0]) for value in item["amplitudes"]],
                   color=item["color"], lw=1.5, label=label)

    left.plot(refTimes, refAbs, color="tab:green", lw=2.0, ls="--",
              label=fr"semi-analytic $\eta_0 e^{{Gt}}$, $G={growth:.4f}$")
    right.plot(refTimes, refRel, color="tab:green", lw=2.0, ls="--",
               label=fr"semi-analytic $e^{{Gt}}$, $G={growth:.4f}$")

    left.axhline(LINEAR_LIMIT, color="tab:red", ls=":", lw=1.0, label=r"linear limit $0.05/k$")
    left.axhline(cellHeight, color="0.45", ls="--", lw=1.0, label=r"cell height $\Delta z$")
    if args.target > maxTime:
        left.axvline(args.target, color="0.55", ls=":", lw=1.0)
        right.axvline(args.target, color="0.55", ls=":", lw=1.0)

    left.set_yscale("log")
    right.set_yscale("log")
    left.set_xlabel(r"$t / t_{\mathrm{RT}}$")
    right.set_xlabel(r"$t / t_{\mathrm{RT}}$")
    left.set_ylabel(r"$|A_{\mathrm{mode1}}|$  [cm]")
    right.set_ylabel(r"$|A / A_0|$")
    left.set_title("absolute amplitude")
    right.set_title("relative growth")
    left.grid(alpha=0.25)
    right.grid(alpha=0.25)
    left.legend(frameon=False, fontsize=8, loc="upper left")
    right.legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(f"KRTI-S-X packet scan  |  max t/t_RT = {maxTime:.3f}  |  target = {args.target:g}",
                 fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    for item in series:
        rel = abs(item["amplitudes"][-1] / item["amplitudes"][0])
        pred = math.exp(growth * item["times"][-1])
        print(f"{item['label']:>6}  t/t_RT={item['times'][-1]:.4f}  A/A0={rel:.4f}  "
              f"pred={pred:.4f}  G={item['fit']:.4f}")


if __name__ == "__main__":
    main()
