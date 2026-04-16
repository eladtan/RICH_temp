#!/usr/bin/env python3
"""
Validate the Moving Slab MC regression test against the semi-analytic
solution from regression_tests/moving_slab_benchmark.py (original_vacuum variant).

The simulation outputs per-group time-averaged radiation energy density
(Eg_time_avg in erg/cm^3) from the observer cell at z_O.  This script
converts to GJ/(cm^3 keV) and compares against the semi-analytic reference.

Metric: energy-weighted fractional error (Eq. 20 of the 2026 paper):
    f_error = sum_g(E_code[g] * |E_code[g] - E_ref[g]| / E_ref[g]) / sum_g(E_code[g])
restricted to groups where the reference is above a noise floor.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REGRESSION_TESTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REGRESSION_TESTS_DIR))
from moving_slab_benchmark import (
    BenchmarkParams,
    compute_groups,
    load_default_opacity_table,
)


def read_spectrum(path):
    """Parse the moving_slab_mc_spectrum.txt output."""
    meta = {}
    data_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line.lstrip("# ").split(None, 1)
                if len(parts) == 2:
                    key, val = parts
                    if key == "columns:":
                        continue
                    try:
                        meta[key] = float(val)
                    except ValueError:
                        meta[key] = val
                continue
            data_lines.append(line)

    cols = np.loadtxt(data_lines, ndmin=2)
    return meta, cols


def eg_to_erad_per_keV(eg_erg_cm3, nu_min_keV, nu_max_keV):
    """
    Convert per-group time-averaged Erad (erg/cm^3 per group)
    to GJ/cm^3/keV by dividing by group width (keV) and erg_per_GJ.
    """
    erg_per_GJ = 1e16
    dnu_keV = nu_max_keV - nu_min_keV
    return eg_erg_cm3 / (dnu_keV * erg_per_GJ)


def compute_reference():
    """Compute the semi-analytic reference spectrum."""
    params = BenchmarkParams(benchmark_type="original_vacuum")
    opacity_data = load_default_opacity_table()
    rows = compute_groups(params, opacity_data)
    return np.array([r["erad_GJ_per_cm3_per_keV"] for r in rows])


def energy_weighted_fractional_error(code, ref, floor_frac=1e-4):
    """
    Eq. 20 of the 2026 paper:
    f_error = sum_g(E_code[g] * |E_code[g] - E_ref[g]| / E_ref[g]) / sum_g(E_code[g])
    Only include groups where ref > floor_frac * max(ref).
    """
    ref_max = np.max(ref)
    if ref_max <= 0:
        return 0.0

    mask = ref > floor_frac * ref_max
    if not np.any(mask):
        return 0.0

    c = code[mask]
    r = ref[mask]

    num = np.sum(c * np.abs(c - r) / r)
    den = np.sum(c)
    if den <= 0:
        return float("inf")
    return float(num / den)


def relative_l1(code, ref, floor_frac=1e-4):
    """Relative L1 norm on significant groups."""
    ref_max = np.max(ref)
    if ref_max <= 0:
        return 0.0
    mask = ref > floor_frac * ref_max
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.abs(code[mask] - ref[mask])) / np.sum(ref[mask]))


def main():
    parser = argparse.ArgumentParser(description="Moving slab MC spectrum check")
    parser.add_argument("--spectrum", required=True,
                        help="Path to moving_slab_mc_spectrum.txt")
    parser.add_argument("--max-ferror", type=float, default=0.30,
                        help="Maximum allowed energy-weighted fractional error")
    parser.add_argument("--plot-dir", default=None,
                        help="Directory for plots (default: same as spectrum)")
    args = parser.parse_args()

    meta, cols = read_spectrum(args.spectrum)

    G = cols.shape[0]
    nu_min_keV = cols[:, 1]
    nu_max_keV = cols[:, 2]
    eg_time_avg = cols[:, 3]

    code_erad = eg_to_erad_per_keV(eg_time_avg, nu_min_keV, nu_max_keV)

    print("Computing semi-analytic reference (this may take a minute)...")
    ref_erad = compute_reference()

    ferr = energy_weighted_fractional_error(code_erad, ref_erad)
    l1 = relative_l1(code_erad, ref_erad)

    print(f"MOVING_SLAB_MC_FERROR={ferr:.8e}")
    print(f"MOVING_SLAB_MC_L1={l1:.8e}")
    print(f"MOVING_SLAB_MC_MAX_FERROR={args.max_ferror:.8e}")
    print(f"Moving slab MC check:")
    print(f"  Groups              = {G}")
    print(f"  Observer x          = {meta.get('observer_x_cm', '?')} cm")
    print(f"  F-error             = {ferr:.6f}")
    print(f"  Rel. L1             = {l1:.6f}")
    print(f"  Threshold           = {args.max_ferror}")

    plot_dir = args.plot_dir or os.path.dirname(args.spectrum)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nu_center = np.sqrt(nu_min_keV * nu_max_keV)

        pos_code = code_erad > 0
        pos_ref = ref_erad > 0

        fig, ax = plt.subplots(figsize=(8, 5))
        if np.any(pos_ref):
            ax.loglog(nu_center[pos_ref], ref_erad[pos_ref],
                      "-", lw=1.5, color="C0", label="Semi-analytic")
        if np.any(pos_code):
            ax.loglog(nu_center[pos_code], code_erad[pos_code],
                      "x", ms=4, mew=1.0, color="C1", label="MC simulation")

        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel(r"$E_{r,g}$ (GJ/cm$^3$/keV)")
        ax.set_title(f"Moving slab MC (original vacuum)\n"
                     f"F-error = {ferr:.4f}, L1 = {l1:.4f}")
        ax.legend(fontsize=9)
        ax.set_xlim(1e-2, 20)
        ax.set_ylim(1e-7, 2e-3)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()

        comp_base = os.path.join(plot_dir, "moving_slab_mc_comparison")
        fig.savefig(comp_base + ".png", dpi=200)
        fig.savefig(comp_base + ".pdf")
        plt.close(fig)
        print(f"  Plot: {comp_base}.png")
        print(f"  Plot: {comp_base}.pdf")

        opacity_data = load_default_opacity_table()
        op_lo = opacity_data["nu_min"]
        op_hi = opacity_data["nu_max"]
        op_kappa = opacity_data["kappa"]
        op_center = np.sqrt(op_lo * op_hi)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.step(op_lo, op_kappa, where="post", lw=1.2, color="C3")
        ax2.scatter(op_center, op_kappa, s=16, marker="x", linewidths=1.0, color="C3", zorder=3)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Energy (keV)")
        ax2.set_ylabel(r"$\kappa$ (cm$^2$/g)")
        ax2.set_title("124-group aluminum opacity table")
        ax2.set_xlim(op_lo[0], op_hi[-1])
        ax2.grid(True, which="both", alpha=0.3)
        fig2.tight_layout()

        op_base = os.path.join(plot_dir, "moving_slab_mc_opacity")
        fig2.savefig(op_base + ".png", dpi=200)
        fig2.savefig(op_base + ".pdf")
        plt.close(fig2)
        print(f"  Plot: {op_base}.png")
        print(f"  Plot: {op_base}.pdf")
    except ImportError:
        print("  matplotlib not available -- skipping plots")

    if ferr > args.max_ferror:
        print(f"FAIL: F-error {ferr:.6f} exceeds threshold {args.max_ferror}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
