#!/usr/bin/env python3
"""Compare a RICH hohlraum line-out to Steinberg & Heizler 2022 Fig. 5(e).

Digitized from arXiv:2108.13453. DIMC and McClarren & Urbatsch (2009) material
temperature along r = 0.05 cm at t = 10 ns.

Usage:
    python3 compare_dimc.py
    python3 compare_dimc.py --dir profiles
    python3 compare_dimc.py path/to/profile.txt
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
REF_MC = os.path.join(HERE, "reference", "fig5e_mcclarren.csv")
REF_DIMC = os.path.join(HERE, "reference", "fig5e_dimc.csv")
MATERIAL_REGIONS = [
    (0.10, 0.15),
    (0.55, 0.95),
    (1.35, 1.40),
]


def load_profile(path):
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
    if data.ndim == 1:
        data = data.reshape(1, -1)
    T = data[:, 2] if data.shape[1] > 2 else data[:, 1]
    return data[:, 0], T, t_ns


def bin_profile(x, T, nbins=200, x_range=(0.0, 1.45)):
    edges = np.linspace(x_range[0], x_range[1], nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(nbins, np.nan)
    for i in range(nbins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if np.any(mask):
            means[i] = np.mean(T[mask])
    valid = ~np.isnan(means)
    return centers[valid], means[valid]


def find_profile(explicit=None, directory=None):
    if explicit:
        return explicit, None
    search = []
    dirs = []
    if directory:
        dirs.append(directory)
    dirs += [os.path.join(HERE, "profiles"), HERE]
    for d in dirs:
        search += glob.glob(os.path.join(d, "*final.txt"))
        search += glob.glob(os.path.join(d, "*[0-9][0-9][0-9][0-9][0-9].txt"))
    if not search:
        return None, None
    best = None
    best_dt = 1e300
    latest = None
    latest_t = -1.0
    for path in search:
        try:
            _, _, t_ns = load_profile(path)
        except Exception:
            continue
        if t_ns is None:
            continue
        if t_ns > latest_t:
            latest_t = t_ns
            latest = path
        dt = abs(t_ns - 10.0)
        if dt < best_dt:
            best_dt = dt
            best = path
    return (best if best_dt < 3.0 else latest), (10.0 if best_dt < 3.0 else latest_t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", help="RICH line-out (x, T_K, T_keV)")
    parser.add_argument("--dir", help="directory of dump files")
    parser.add_argument("--output", default=os.path.join(HERE, "compare_dimc.png"))
    args = parser.parse_args()

    path, _ = find_profile(args.profile, args.dir)
    if path is None or not os.path.exists(path):
        print("No RICH hohlraum profile found. Run the job or pass a .txt dump.", file=sys.stderr)
        sys.exit(1)

    x, T, t_ns = load_profile(path)
    xb, Tb = bin_profile(x, T)
    mc = np.loadtxt(REF_MC, delimiter=",", comments="#")
    dimc = np.loadtxt(REF_DIMC, delimiter=",", comments="#")

    print(f"RICH profile: {path}  t={t_ns} ns")
    if t_ns is not None and abs(t_ns - 10.0) > 1.0:
        print(f"Note: DIMC Fig. 5(e) is at t=10 ns; this dump is t={t_ns:.3g} ns.")

    T_on_dimc = np.interp(dimc[:, 0], xb, Tb, left=np.nan, right=np.nan)
    valid = np.isfinite(T_on_dimc)
    if np.any(valid):
        d = T_on_dimc[valid] - dimc[valid, 1]
        print(f"vs DIMC Fig. 5(e):  RMS={np.sqrt(np.mean(d**2)):.4f} keV  max|dT|={np.max(np.abs(d)):.4f} keV")

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (xlo, xhi) in enumerate(MATERIAL_REGIONS):
        ax.axvspan(xlo, xhi, alpha=0.12, color="steelblue",
                   label="Absorbing material" if i == 0 else None)
    ax.plot(mc[:, 0], mc[:, 1], "k-", lw=2.0, label="McClarren & Urbatsch 2009")
    ax.plot(dimc[:, 0], dimc[:, 1], "g-", lw=1.4, label="Steinberg & Heizler DIMC (10 ns)")
    lbl = "RICH IMC"
    if t_ns is not None:
        lbl += f" ({t_ns:.2g} ns)"
    ax.plot(xb, Tb, "C0-", lw=1.6, label=lbl)
    ax.set_xlim(0, 1.45)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel(r"$T_{\mathrm{mat}}$ (keV)")
    ax.set_title(r"Hohlraum $T_{\mathrm{mat}}$ at $r=0.05$ cm  —  Fig. 5(e), Steinberg & Heizler 2022")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
