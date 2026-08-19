#!/usr/bin/env python3
"""Compare a RICH Marshak profile to Steinberg & Heizler 2022 Fig. 1(c).

Digitized from arXiv:2108.13453. DIMC overlaps their diffusion solver at ct=500.

Usage:
    python3 compare_dimc.py
    python3 compare_dimc.py marshak_12345_ct500.txt
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIFF = os.path.join(HERE, "reference", "fig1c_diffusion.csv")
REF_DIMC = os.path.join(HERE, "reference", "fig1c_dimc.csv")


def load_rich(path):
    ct = None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                m = re.search(r"ct=([\d.eE+\-]+)", line)
                if m:
                    ct = float(m.group(1))
            else:
                break
    data = np.loadtxt(path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1], ct


def find_profile(explicit=None):
    if explicit:
        return explicit
    candidates = sorted(glob.glob(os.path.join(HERE, "*_ct500.txt")))
    candidates += sorted(glob.glob(os.path.join(HERE, "*_final.txt")))
    preferred = [c for c in candidates if "ct500" in os.path.basename(c) and "mc" in os.path.basename(c)]
    if preferred:
        return preferred[-1]
    ct500 = [c for c in candidates if "ct500" in os.path.basename(c)]
    if ct500:
        return ct500[-1]
    return candidates[-1]


def rms_on_grid(x_ref, y_ref, x, y, mask=None):
    yi = np.interp(x_ref, x, y)
    if mask is None:
        mask = np.ones(x_ref.size, dtype=bool)
    d = yi[mask] - y_ref[mask]
    return float(np.sqrt(np.mean(d ** 2))), float(np.max(np.abs(d)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", help="RICH Marshak profile (x, T)")
    parser.add_argument("--output", default=os.path.join(HERE, "compare_dimc.png"))
    args = parser.parse_args()

    path = find_profile(args.profile)
    if path is None or not os.path.exists(path):
        print("No RICH profile found. Pass a *_ct500.txt file or run the Marshak job.", file=sys.stderr)
        sys.exit(1)

    x, T, ct = load_rich(path)
    diff = np.loadtxt(REF_DIFF, delimiter=",", comments="#")
    dimc = np.loadtxt(REF_DIMC, delimiter=",", comments="#")

    mask = diff[:, 0] <= 2.0
    rms_d, max_d = rms_on_grid(diff[:, 0], diff[:, 1], x, T, mask)
    rms_m, max_m = rms_on_grid(dimc[:, 0], dimc[:, 1], x, T, dimc[:, 0] <= 2.0)

    print(f"RICH profile: {path}  ct={ct}")
    print(f"vs diffusion (Fig. 1c black), x<=2:  RMS={rms_d:.4f}  max|dT|={max_d:.4f}")
    print(f"vs DIMC      (Fig. 1c),      x<=2:  RMS={rms_m:.4f}  max|dT|={max_m:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(diff[:, 0], diff[:, 1], "k-", lw=2.0, label="Steinberg & Heizler diffusion")
    ax.plot(dimc[:, 0], dimc[:, 1], "g--", lw=1.4, label="Steinberg & Heizler DIMC")
    label = "RICH IMC"
    if ct is not None:
        label += f" (ct={ct:g})"
    ax.plot(x, T, "o", ms=3, color="C0", alpha=0.8, label=label)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("T")
    ax.set_title("Marshak wave at ct = 500  —  Fig. 1(c), Steinberg & Heizler 2022")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
