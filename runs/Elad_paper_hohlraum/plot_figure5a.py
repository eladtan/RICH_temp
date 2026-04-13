#!/usr/bin/env python3
"""
Plot material temperature along r = 0.05 cm vs x, in the style of
Figure 5a from Steinberg & Heizler (2021), arXiv:2108.13453, Section 4.2
(McClarren & Urbatsch 2009 cylindrical hohlraum benchmark).

Usage:
    python plot_figure5a.py                                  # default dir + 512-rank profile
    python plot_figure5a.py --dir /path/to/results           # custom results directory
    python plot_figure5a.py file1.txt [file2.txt …]          # explicit profile files
"""

import argparse
import sys
import os
import re
import glob
import numpy as np
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_DATA_DIR = "/home/maorm/shared/Hohlraum/delta_0.030000/size_512"

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


def detect_prefix(data_dir):
    """Auto-detect the dump file prefix from .txt files in data_dir.
    Expects filenames like PREFIX_00001.txt and returns PREFIX, or None."""
    prefix_re = re.compile(r"^(.+)_\d{5}\.txt$")
    for name in os.listdir(data_dir):
        m = prefix_re.match(name)
        if m:
            return m.group(1)
    return None


def find_dumps_at_times(data_dir, prefix, target_times_ns):
    """Find dump files closest to each target time."""
    pattern = os.path.join(data_dir, f"{prefix}_?????.txt")
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot material temperature along r = 0.05 cm vs x "
                    "(Figure 5a, Steinberg & Heizler 2021)."
    )
    parser.add_argument("files", nargs="*",
                        help="Explicit profile .txt files to plot (skips auto-panel mode)")
    parser.add_argument("--dir", type=str, nargs="+", default=[DEFAULT_DATA_DIR],
                        help="One or two directories containing dump files. "
                             "One dir: time-evolution plot. "
                             "Two dirs: time-evolution from first + comparison panel at t~1 ns. "
                             f"(default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--output", type=str, default="figure5a.png",
                        help="Output file path (PNG or PDF). (default: figure5a.png)")
    return parser.parse_args()


def setup_dir(data_dir):
    """Resolve directory, detect prefix, return (data_dir, prefix) or exit."""
    data_dir = os.path.expanduser(data_dir)
    if not os.path.isdir(data_dir):
        print(f"Error: directory does not exist: {data_dir}", file=sys.stderr)
        sys.exit(1)
    prefix = detect_prefix(data_dir)
    if prefix is None:
        print(f"Error: no dump files (PREFIX_00000.txt) found in {data_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"  {data_dir}  ->  prefix '{prefix}'")
    return data_dir, prefix


def plot_time_evolution(ax, data_dir, prefix, target_times):
    """Plot time-evolution panel and return the dumps dict."""
    dumps = find_dumps_at_times(data_dir, prefix, target_times)
    if not dumps:
        print(f"Warning: no dumps matched target times in {data_dir}", file=sys.stderr)
        return dumps
    cmap = plt.cm.plasma
    available_times = sorted(dumps.keys())
    for i, t in enumerate(available_times):
        fpath, actual_t = dumps[t]
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
        f"\n({prefix})",
        fontsize=12,
    )
    return dumps


def main():
    args = parse_args()

    if args.files:
        fig, ax = plt.subplots(figsize=(10, 5))
        for f in args.files:
            plot_single(ax, f)
        shade_material_regions(ax, 1.0)
        ax.set_xlabel("x (cm)", fontsize=13)
        ax.set_ylabel(r"$T_{\mathrm{mat}}$ (keV)", fontsize=13)
        ax.set_xlim(0, 1.45)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10)
        ax.set_title("Material temperature along r = 0.05 cm", fontsize=14)
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        print(f"Saved {args.output}")
        plt.show()
        return

    print("Detected directories and prefixes:")
    dirs_and_prefixes = [setup_dir(d) for d in args.dir]
    print()

    target_times = [1, 2, 3, 4, 5, 6]
    data_dir_1, prefix_1 = dirs_and_prefixes[0]

    if len(dirs_and_prefixes) == 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_time_evolution(ax, data_dir_1, prefix_1, target_times)
        fig.suptitle(
            "McClarren & Urbatsch (2009) Hohlraum — "
            "cf. Steinberg & Heizler (2021) Fig. 5a",
            fontsize=14, y=1.02,
        )
    else:
        data_dir_2, prefix_2 = dirs_and_prefixes[1]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        dumps_1 = plot_time_evolution(axes[0], data_dir_1, prefix_1, target_times)

        ax = axes[1]
        dumps_2 = find_dumps_at_times(data_dir_2, prefix_2, [1.0])
        if 1.0 in dumps_1:
            fpath, actual_t = dumps_1[1.0]
            plot_single(ax, fpath, label=f"{prefix_1}, t = {actual_t:.1f} ns", color="C0")
        if 1.0 in dumps_2:
            fpath, actual_t = dumps_2[1.0]
            plot_single(ax, fpath, label=f"{prefix_2}, t = {actual_t:.1f} ns", color="C1")

        shade_material_regions(ax, 1.0)
        ax.set_xlabel("x (cm)", fontsize=13)
        ax.set_ylabel(r"$T_{\mathrm{mat}}$ (keV)", fontsize=13)
        ax.set_xlim(0, 1.45)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10)
        ax.set_title(
            r"(b) Comparison at $t \approx 1$ ns",
            fontsize=12,
        )
        fig.suptitle(
            "McClarren & Urbatsch (2009) Hohlraum — "
            "cf. Steinberg & Heizler (2021) Fig. 5a",
            fontsize=14, y=1.02,
        )

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
