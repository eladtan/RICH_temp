#!/usr/bin/env python3
"""Compare a RICH Mach-45 profile to Steinberg & Heizler 2022 Fig. 9(b).

Digitized DIMC (green) and SN (black) curves from arXiv:2108.13453.
The DIMC paper used x in [-879, 91] cm at t = 8e-7 s; this RICH setup is a
translated window around x = 2300 cm. The overlay is shifted so the density
jumps coincide.

Usage:
    python3 compare_dimc.py
    python3 compare_dimc.py mach45_12345_final.txt
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
REF_DIMC = os.path.join(HERE, "reference", "fig9b_dimc.csv")
REF_SN = os.path.join(HERE, "reference", "fig9b_sn.csv")


def load_rich(path):
    t_us = None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                m = re.search(r"t_us=([\d.eE+\-]+)", line)
                if m:
                    t_us = float(m.group(1))
            else:
                break
    data = np.loadtxt(path, delimiter=",", comments="#")
    return dict(x=data[:, 0], rho=data[:, 1], T_gas=data[:, 2],
                T_rad=data[:, 3], vx=data[:, 4], t_us=t_us, path=path)


def load_ref(path):
    data = np.loadtxt(path, delimiter=",", comments="#")
    return dict(x=data[:, 0], T_gas=data[:, 1], T_rad=data[:, 2],
                rho=data[:, 3], vx=data[:, 4])


def find_profile(explicit=None):
    if explicit:
        return explicit
    files = glob.glob(os.path.join(HERE, "*_final.txt"))
    files += glob.glob(os.path.join(HERE, "*", "*_final.txt"))
    files += glob.glob(os.path.join(HERE, "*_latest.txt"))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]


def shock_x(prof, rho_lo=1.0, rho_hi=6.43):
    rho = prof["rho"]
    x = prof["x"]
    target = 0.5 * (rho_lo + rho_hi)
    margin = max(10, len(rho) // 50)
    valid = np.arange(margin, len(rho) - margin)
    return x[valid[np.argmin(np.abs(rho[valid] - target))]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", help="RICH Mach-45 profile")
    parser.add_argument("--output", default=os.path.join(HERE, "compare_dimc.png"))
    args = parser.parse_args()

    path = find_profile(args.profile)
    if path is None or not os.path.exists(path):
        print("No RICH Mach-45 profile found. Pass *_final.txt or run the job.", file=sys.stderr)
        sys.exit(1)

    rich = load_rich(path)
    dimc = load_ref(REF_DIMC)
    sn = load_ref(REF_SN)
    xs_rich = shock_x(rich)
    xs_dimc = shock_x(dimc)
    shift = xs_rich - xs_dimc
    vx_rich = rich["vx"] - np.median(rich["vx"][:max(5, len(rich["vx"]) // 20)])
    print(f"RICH profile: {path}  t={rich['t_us']} us  shock x={xs_rich:.5g} cm")
    print(f"DIMC Fig. 9(b) shock x={xs_dimc:.5g} cm; overlay shift={shift:.5g} cm")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        (axes[0, 0], "T_gas", r"$T_{\mathrm{gas}}$ (keV)", None),
        (axes[0, 1], "T_rad", r"$T_{\mathrm{rad}}$ (keV)", None),
        (axes[1, 0], "rho", r"$\rho$ (g cm$^{-3}$)", None),
        (axes[1, 1], "vx", r"$v_x$ (cm s$^{-1}$)", vx_rich),
    ]
    for ax, key, ylab, rich_y in panels:
        y_r = rich_y if rich_y is not None else rich[key]
        ax.plot(sn["x"] + shift, sn[key], "k-", lw=2.0, label="SN (Fig. 9b)")
        ax.plot(dimc["x"] + shift, dimc[key], "g--", lw=1.4, label="DIMC (Fig. 9b)")
        ax.plot(rich["x"], y_r, "C0-", lw=1.2, alpha=0.85, label="RICH IMC")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Mach 45 radiative shock  —  Fig. 9(b), Steinberg & Heizler 2022")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
