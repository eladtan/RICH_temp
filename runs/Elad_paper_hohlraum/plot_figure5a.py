#!/usr/bin/env python3
"""
Plot material temperature along r = 0.05 cm vs x, in the style of
Figure 5a from Steinberg & Heizler (2021), arXiv:2108.13453, Section 4.2
(McClarren & Urbatsch 2009 cylindrical hohlraum benchmark).

Usage:
    python plot_figure5a.py                           # default: latest 512-rank profile
    python plot_figure5a.py <file1.txt> [file2.txt …] # explicit profile files
"""

import sys
import os
import re
import glob
import numpy as np
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = "/home/maorm/shared/Hohlraum2"

# Material regions (absorbing, blue in paper Fig. 3), at r = 0.05 cm
MATERIAL_REGIONS = [
    (0.10, 0.15, "Left wall"),
    (0.55, 0.95, "Capsule"),
    (1.35, 1.40, "Right cap"),
]


def load_profile(filepath):
    """Load a profile txt file. Returns (x, T_K, T_keV, t_ns, r_line)."""
    t_ns, r_line = None, None
    with open(filepath) as f:
        for line in f:
            if line.startswith("#"):
                m = re.search(r"t_ns=([\d.e+-]+)", line)
                if m:
                    t_ns = float(m.group(1))
                m = re.search(r"r_line=([\d.e+-]+)", line)
                if m:
                    r_line = float(m.group(1))
            else:
                break
    data = np.loadtxt(filepath, delimiter=",", comments="#")
    return data[:, 0], data[:, 1], data[:, 2], t_ns, r_line


def bin_profile(x, T_keV, nbins=200, x_range=(0, 1.45)):
    """Bin the scattered Voronoi data into uniform x-bins (mean)."""
    edges = np.linspace(x_range[0], x_range[1], nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(nbins, np.nan)
    for i in range(nbins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if mask.sum() > 0:
            means[i] = np.mean(T_keV[mask])
    valid = ~np.isnan(means)
    return centers[valid], means[valid]


def shade_material_regions(ax, ymax):
    """Add shaded bands for the absorbing-material regions at r=0.05."""
    for i, (xlo, xhi, label) in enumerate(MATERIAL_REGIONS):
        ax.axvspan(xlo, xhi, alpha=0.12, color="steelblue", zorder=0,
                   label="Absorbing material" if i == 0 else None)


def find_dumps_at_times(prefix, target_times_ns):
    """Find dump files closest to each target time."""
    pattern = os.path.join(DATA_DIR, f"{prefix}_?????.txt")
    files = sorted(glob.glob(pattern))
    results = {}
    for f in files:
        with open(f) as fh:
            header = fh.readline()
        m = re.search(r"t_ns=([\d.e+-]+)", header)
        if m:
            t = float(m.group(1))
            for tt in target_times_ns:
                if tt not in results or abs(t - tt) < abs(results[tt][1] - tt):
                    results[tt] = (f, t)
    return results


def plot_single(ax, filepath, label=None, color=None, nbins=200):
    """Plot a single profile on the given axes."""
    x, _, T_keV, t_ns, _ = load_profile(filepath)
    xb, Tb = bin_profile(x, T_keV, nbins=nbins)
    lbl = label or f"t = {t_ns:.1f} ns"
    ax.plot(xb, Tb, label=lbl, color=color, linewidth=1.5)
    return t_ns


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = None

    if files:
        # --- Mode 1: plot user-specified files ---
        fig, ax = plt.subplots(figsize=(10, 5))
        for f in files:
            plot_single(ax, f)
        shade_material_regions(ax, 1.0)
        ax.set_xlabel("x (cm)", fontsize=13)
        ax.set_ylabel(r"$T_{\mathrm{mat}}$ (keV)", fontsize=13)
        ax.set_xlim(0, 1.45)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10)
        ax.set_title("Material temperature along r = 0.05 cm", fontsize=14)
        plt.tight_layout()
        plt.savefig("figure5a.png", dpi=150)
        print("Saved figure5a.png")
        plt.show()
        return

    # --- Mode 2: automatic multi-panel figure ---
    # Panel (a): time evolution from the 512-rank run
    # Panel (b): comparison of 512 vs 1024 at t ~ 1 ns
    prefix_512 = "Hohlraum_0.03_512"
    prefix_1024 = "Hohlraum_0.03_1024"

    target_times = [1, 2, 3, 4, 5, 6]
    dumps_512 = find_dumps_at_times(prefix_512, target_times)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel (a): time evolution ---
    ax = axes[0]
    cmap = plt.cm.plasma
    available_times = sorted(dumps_512.keys())
    for i, t in enumerate(available_times):
        fpath, actual_t = dumps_512[t]
        color = cmap(0.15 + 0.75 * i / max(len(available_times) - 1, 1))
        plot_single(ax, fpath, label=f"t = {actual_t:.1f} ns", color=color)

    shade_material_regions(ax, 1.0)
    ax.set_xlabel("x (cm)", fontsize=13)
    ax.set_ylabel(r"$T_{\mathrm{mat}}$ (keV)", fontsize=13)
    ax.set_xlim(0, 1.45)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(
        r"(a) Material temperature along $r = 0.05$ cm"
        "\n"
        r"($\Delta r \approx 0.03$ cm, $\Delta t = 10^{-11}$ s, 512 ranks)",
        fontsize=12,
    )

    # --- Panel (b): 512 vs 1024 comparison at t ~ 1 ns ---
    ax = axes[1]
    dumps_1024 = find_dumps_at_times(prefix_1024, [1.0])

    if 1.0 in dumps_512:
        fpath, actual_t = dumps_512[1.0]
        plot_single(ax, fpath, label=f"512 ranks, t = {actual_t:.1f} ns", color="C0")
    if 1.0 in dumps_1024:
        fpath, actual_t = dumps_1024[1.0]
        plot_single(ax, fpath, label=f"1024 ranks, t = {actual_t:.1f} ns",
                    color="C1")

    shade_material_regions(ax, 1.0)
    ax.set_xlabel("x (cm)", fontsize=13)
    ax.set_ylabel(r"$T_{\mathrm{mat}}$ (keV)", fontsize=13)
    ax.set_xlim(0, 1.45)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    ax.set_title(
        r"(b) Comparison at $t \approx 1$ ns"
        "\n"
        r"($\Delta r \approx 0.03$ cm, $\Delta t = 10^{-11}$ s)",
        fontsize=12,
    )

    fig.suptitle(
        "McClarren & Urbatsch (2009) Hohlraum — "
        "cf. Steinberg & Heizler (2021) Fig. 5a",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    plt.savefig("figure5a.png", dpi=150, bbox_inches="tight")
    print("Saved figure5a.png")
    plt.show()


if __name__ == "__main__":
    main()
