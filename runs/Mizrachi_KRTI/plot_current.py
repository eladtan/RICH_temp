#!/usr/bin/env python3
"""Overlay the running KRTI history against the frozen KRTI-S-X semi-analytic mode."""

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
ROOT_S = 12893.260900062034  # KRTI-S-X_parameters.json: root_s_inv_s


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    times, amplitudes, header = read_history(args.history)
    outPath = args.out if args.out is not None else args.history.with_name("krti_current.png")

    rtTime = 1.0 / math.sqrt(ATWOOD * GRAVITY * WAVE_NUMBER)
    growth = ROOT_S * rtTime
    nz = int(header.get("nz", 42))
    slabHalfHeight = float(header.get("H_cm", 1.0))
    cellHeight = 2.0 * slabHalfHeight / nz
    linearLimit = 0.05 / WAVE_NUMBER

    reference = [ETA0 * math.exp(growth * t) for t in times]
    normalized = [abs(a / amplitudes[0]) for a in amplitudes]
    normalizedReference = [math.exp(growth * t) for t in times]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    left.plot(times, [abs(a) for a in amplitudes], color="tab:blue", lw=1.6,
              label=r"simulation $|A_{\mathrm{mode1}}|$")
    left.plot(times, reference, color="tab:green", lw=1.8,
              label=fr"semi-analytic $\eta_0 e^{{Gt}}$, $G={growth:.4f}$")
    left.axhline(linearLimit, color="tab:red", ls=":", label=r"linear limit $0.05/k$")
    left.axhline(cellHeight, color="0.4", ls="--", label=r"cell height $\Delta z$")
    left.set_yscale("log")
    left.set_xlabel(r"$t / t_{\mathrm{RT}}$")
    left.set_ylabel(r"$|A_k|$  [cm]")
    left.set_title("absolute amplitude")
    left.grid(alpha=0.25)
    left.legend(frameon=False, fontsize=8, loc="center left")

    right.plot(times, normalized, color="tab:blue", lw=1.6,
               label=r"simulation $|A/A_0|$")
    right.plot(times, normalizedReference, color="tab:green", lw=1.8,
               label=fr"semi-analytic $e^{{Gt}}$, $G={growth:.4f}$")
    right.set_yscale("log")
    right.set_xlabel(r"$t / t_{\mathrm{RT}}$")
    right.set_ylabel("amplitude / initial amplitude")
    right.set_title("relative growth (rate comparison)")
    right.grid(alpha=0.25)
    right.legend(frameon=False, fontsize=9)

    fittedGrowth = math.log(normalized[-1]) / times[-1] if times[-1] > 0.0 else float("nan")

    fig.suptitle(f"KRTI-S-X  |  nz={nz}  |  t/t_RT = {times[-1]:.3f}  |  {len(times)} samples",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(outPath, dpi=160)
    print(f"grid                : nz={nz}, dz={cellHeight:.5f} cm")
    print(f"t/t_RT reached      : {times[-1]:.4f}")
    print(f"A_mode1 initial     : {amplitudes[0]:.6e} cm  (eta0 = {ETA0:.3e}, ratio {amplitudes[0] / ETA0:.6f})")
    print(f"A_mode1 latest      : {amplitudes[-1]:.6e} cm")
    print(f"simulation growth   : {normalized[-1]:.4f}")
    print(f"semi-analytic growth: {normalizedReference[-1]:.4f}")
    print(f"fitted G (endpoints): {fittedGrowth:.4f}   target {growth:.4f}")
    print(f"wrote {outPath}")


if __name__ == "__main__":
    main()
