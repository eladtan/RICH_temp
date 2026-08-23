#!/usr/bin/env python3
"""Compare a RICH Mach-2 profile to Steinberg & Heizler 2022 Fig. 9(a).

Digitized DIMC (green) and analytic (magenta) curves from arXiv:2108.13453.

Usage:
    python3 compare_dimc.py
    python3 compare_dimc.py mach2_storm_12345_final.txt
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
REF_DIMC = os.path.join(HERE, "reference", "fig9a_dimc.csv")
REF_ANALYTIC = os.path.join(HERE, "paper_fig9a.csv")
if not os.path.exists(REF_ANALYTIC):
    REF_ANALYTIC = os.path.join(HERE, "reference", "fig9a_analytic.csv")


def load_rich(path):
    t_ns = None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                m = re.search(r"t_ns=([\d.eE+\-]+)", line)
                if m:
                    t_ns = float(m.group(1))
            else:
                break
    data = np.loadtxt(path, delimiter=",", comments="#")
    return dict(x=data[:, 0], rho=data[:, 1], T_gas=data[:, 2],
                T_rad=data[:, 3], vx=data[:, 4], t_ns=t_ns, path=path)


def load_ref(path):
    data = np.loadtxt(path, delimiter=",", comments="#")
    return dict(x=data[:, 0], T_gas=data[:, 1], T_rad=data[:, 2],
                rho=data[:, 3], vx=data[:, 4])


def find_profile(explicit=None):
    if explicit:
        return explicit
    files = sorted(glob.glob(os.path.join(HERE, "*_final.txt")))
    files += sorted(glob.glob(os.path.join(HERE, "mach2_storm_*_final.txt")))
    return files[-1] if files else None


def shock_x(prof):
    rho = prof["rho"]
    x = prof["x"]
    target = 0.5 * (1.0 + 2.29)
    margin = max(10, len(rho) // 50)
    valid = np.arange(margin, len(rho) - margin)
    return x[valid[np.argmin(np.abs(rho[valid] - target))]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", help="RICH Mach-2 profile")
    parser.add_argument("--output", default=os.path.join(HERE, "compare_dimc.png"))
    args = parser.parse_args()

    path = find_profile(args.profile)
    if path is None or not os.path.exists(path):
        print("No RICH Mach-2 profile found. Pass *_final.txt or run the job.", file=sys.stderr)
        sys.exit(1)

    rich = load_rich(path)
    dimc = load_ref(REF_DIMC)
    analytic = load_ref(REF_ANALYTIC)
    xs = shock_x(rich)
    paper_shock = -0.1730126582
    shift = xs - paper_shock
    print(f"RICH profile: {path}  t={rich['t_ns']} ns  shock x={xs:.5g} cm")
    print(f"Overlay shift (RICH shock - DIMC shock): {shift:.5g} cm")

    vx_rich = rich["vx"] - np.median(rich["vx"][:max(5, len(rich["vx"]) // 20)])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        (axes[0, 0], "T_gas", r"$T_{\mathrm{gas}}$ (keV)", None),
        (axes[0, 1], "T_rad", r"$T_{\mathrm{rad}}$ (keV)", None),
        (axes[1, 0], "rho", r"$\rho$ (g cm$^{-3}$)", None),
        (axes[1, 1], "vx", r"$v_x$ (cm s$^{-1}$)", vx_rich),
    ]
    for ax, key, ylab, rich_y in panels:
        y_a = analytic[key]
        y_d = dimc[key]
        y_r = rich_y if rich_y is not None else rich[key]
        ax.plot(analytic["x"] + shift, y_a, "m-", lw=2.0, label="Analytic (Fig. 9a)")
        ax.plot(dimc["x"] + shift, y_d, "g--", lw=1.4, label="DIMC (Fig. 9a)")
        ax.plot(rich["x"], y_r, "C0-", lw=1.2, alpha=0.85, label="RICH IMC")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Mach 2 radiative shock  —  Fig. 9(a), Steinberg & Heizler 2022")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
