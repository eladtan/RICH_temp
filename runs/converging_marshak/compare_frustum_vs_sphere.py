#!/usr/bin/env python3
"""Overlay one snapshot from the 768-rank sphere run and the 16-rank frustum run
on the paper similarity solution.

The frustum is a debug geometry: one angular cell per radial shell inside a
1e-4 sr wedge.  Comparing it against the full sphere at the same paper time
separates angular resolution from the radial physics, so only the snapshot both
runs have reached is plotted.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_profiles import CASES, author_solution, shell_average, normalized_l1


def binned(path, case, bins):
    data = np.genfromtxt(path, delimiter=",", names=True, comments="#", skip_header=1)
    radius, average, spread = shell_average(data, case["radius"], bins)
    energy = average["umat_erg_cm3"] + average["erad_erg_cm3"]
    return radius, average, spread, energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=4)
    parser.add_argument("--snapshot", type=int, default=1, help="1-based snapshot index")
    parser.add_argument("--sphere-dir", type=Path)
    parser.add_argument("--frustum-dir", type=Path)
    parser.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "DIR"),
                        help="overlay an extra run (LABEL DIR); repeatable, replaces the sphere/frustum pair")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=96)
    parser.add_argument("--table", type=Path, default=Path(__file__).with_name("author_table_ii.csv"))
    args = parser.parse_args()

    case = CASES[args.test]
    table = np.genfromtxt(args.table, delimiter=",", names=True)
    paper_time = case["times"][args.snapshot - 1]
    name = f"test{args.test}_snapshot{args.snapshot}.csv"

    palette = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    markers = ["o", "s", "^", "D", "v"]
    if args.run:
        runs = [(label, Path(d) / name, palette[i % len(palette)], markers[i % len(markers)])
                for i, (label, d) in enumerate(args.run)]
    else:
        runs = [
            ("Full sphere, 768 ranks (32000 angular points)", args.sphere_dir / name, "tab:orange", "o"),
            ("Frustum, 16 ranks (1e-4 sr, 1 angular cell)", args.frustum_dir / name, "tab:green", "s"),
        ]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 9.5), sharex=True)
    scale = case["scale_k"]
    # Same display conventions as compare_profiles.py: the micron-scale cases
    # are plotted in microns, and their temperature scale is 0.1 keV (HeV).
    radius_scale = 1.0e4 if case["radius"] <= 1.0e-3 else 1.0
    radius_unit = "micron" if case["radius"] <= 1.0e-3 else "cm"
    temperature_unit = "keV" if abs(scale - 1.160451812e7) < 1.0 else "HeV"

    # Reference on a fine grid: it is a 1D similarity solution, not binned data.
    fine = np.linspace(0.0, case["radius"], max(args.bins, 512))
    ref_t, ref_e, front = author_solution(args.test, fine, paper_time, table)
    axes[0].plot(fine * radius_scale, ref_t / scale, color="tab:blue", lw=2.5, label="Giron (paper Table II)")
    axes[1].plot(fine * radius_scale, ref_e / 1.0e13, color="tab:blue", lw=2.5, label="Giron (paper Table II)")

    print(f"test {args.test}, snapshot {args.snapshot}, paper t = {paper_time:g} ns, front = {front:.6g} cm")
    for label, path, color, marker in runs:
        if not path.exists():
            print(f"  MISSING {path}")
            continue
        radius, average, spread, energy = binned(path, case, args.bins)
        at, ae, _ = author_solution(args.test, radius, paper_time, table)
        l1t = normalized_l1(average["Tmat_K"], at, radius)
        l1e = normalized_l1(energy, ae, radius)
        peak = np.nanmax(average["Tmat_K"])
        above = radius[average["Tmat_K"] > 0.02 * peak]
        foot = float(np.min(above)) if above.size else float("nan")
        print(f"  {label}: L1(T) = {l1t:.6f}, L1(E) = {l1e:.6f}, front foot = {foot:.6g} cm "
              f"(analytic {front:.6g}, offset {(foot - front)/front:+.3%} of front)")
        axes[0].errorbar(radius * radius_scale, average["Tmat_K"] / scale, yerr=spread["Tmat_K"] / scale,
                         color=color, marker=marker, ms=3.0, lw=0.0, elinewidth=0.9,
                         capsize=1.5, label=f"{label}  [L1={l1t:.4f}]")
        axes[1].plot(radius * radius_scale, energy / 1.0e13, color=color, marker=marker, ms=3.0, lw=0.0,
                     label=f"{label}  [L1={l1e:.4f}]")

    axes[0].set_ylabel(f"Temperature [{temperature_unit}]")
    axes[1].set_ylabel(r"Total energy density [$10^{13}$ erg cm$^{-3}$]")
    axes[1].set_xlabel(f"Generator radius [{radius_unit}]")
    axes[0].set_title(f"Converging Marshak benchmark {args.test}, t = {paper_time:g} ns: "
                      f"sphere vs frustum vs paper")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
