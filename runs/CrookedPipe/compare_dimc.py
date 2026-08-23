#!/usr/bin/env python3
"""Compare crooked-pipe probe histories to Steinberg & Heizler 2022 Fig. 8(a).

Digitized DIMC (green) and Gentile 2001 (black) curves from arXiv:2108.13453.

Usage:
    python3 compare_dimc.py
    python3 compare_dimc.py crookedpipe_probes.txt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIMC = os.path.join(HERE, "reference", "fig8_dimc.csv")
REF_GENTILE = os.path.join(HERE, "reference", "fig8_gentile.csv")
PROBES = ["P1 (0, 0.25)", "P2 (0, 2.75)", "P3 (1.25, 3.5)", "P4 (0, 4.25)", "P5 (0, 6.75)"]


def find_probes(explicit=None):
    if explicit:
        return explicit
    files = glob.glob(os.path.join(HERE, "crookedpipe_probes.txt"))
    files += glob.glob(os.path.join(HERE, "output_*", "crookedpipe_probes.txt"))
    return sorted(files)[-1] if files else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", nargs="?", help="RICH probe CSV (t_ns, cycle, T1..T5)")
    parser.add_argument("--output", default=os.path.join(HERE, "compare_dimc.png"))
    args = parser.parse_args()

    path = find_probes(args.probes)
    if path is None or not os.path.exists(path):
        print("No crookedpipe_probes.txt found. Run the job first.", file=sys.stderr)
        sys.exit(1)

    rich = np.loadtxt(path, delimiter=",", comments="#")
    if rich.ndim == 1:
        rich = rich.reshape(1, -1)
    dimc = np.loadtxt(REF_DIMC, delimiter=",", comments="#")
    gent = np.loadtxt(REF_GENTILE, delimiter=",", comments="#")

    print(f"RICH probes: {path}  ({rich.shape[0]} samples, t={rich[0,0]:.3g}–{rich[-1,0]:.3g} ns)")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for i in range(5):
        ax.plot(gent[:, 0], gent[:, i + 1], color="k", lw=1.6, ls="--",
                label="Gentile 2001" if i == 0 else None)
        ax.plot(dimc[:, 0], dimc[:, i + 1], color="g", lw=1.3,
                label="DIMC Fig. 8(a)" if i == 0 else None)
        ax.plot(rich[:, 0], rich[:, i + 2], color=colors[i], lw=1.4,
                label=PROBES[i])
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1e3)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("t (ns)")
    ax.set_ylabel("T (keV)")
    ax.set_title("Crooked pipe probe temperatures  —  Fig. 8(a), Steinberg & Heizler 2022")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
