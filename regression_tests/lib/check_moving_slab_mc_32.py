#!/usr/bin/env python3
"""
Validate the Moving Slab MC 32-group regression test against the semi-analytic
solution from regression_tests/moving_slab_benchmark.py (original_vacuum variant).

The 124-group opacity table is collapsed to 32 log-spaced groups using Planck
weighting at T=1 keV (matching the C++ collapse in test.cpp).  Both the
simulation and the reference use the same 32-group piecewise-constant opacity.

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
    planck_energy_form,
    K_BOLTZMANN_GJ_PER_KEV,
    PLANCK_GJ_NS,
    C_LIGHT,
)

N_FINE = 124
N_COARSE = 32
COARSE_EMIN_KEV = 1.0e-3
COARSE_EMAX_KEV = 3.0e+1
T_COLLAPSE_KEV = 1.0


def planck_group_integral_keV(E_lo_keV, E_hi_keV, T_keV, npts=200):
    """
    Numerically integrate the Planck spectral energy density B(E,T) over
    [E_lo, E_hi] in keV units.  Returns the integral in GJ/cm^3.
    Uses the same Planck function as regression_tests/moving_slab_benchmark.py.
    """
    if E_hi_keV <= E_lo_keV or T_keV <= 0.0:
        return 0.0
    E = np.linspace(E_lo_keV, E_hi_keV, npts)
    B = np.array([planck_energy_form(e, T_keV) for e in E])
    return float(np.trapezoid(B, E)) * (4.0 * np.pi / C_LIGHT)


def collapse_opacity_planck():
    """
    Collapse the 124-group table to 32 log-spaced groups using Planck
    weighting at T = T_COLLAPSE_KEV.
    Returns a dict suitable for moving_slab_benchmark.compute_groups() (same module).
    """
    fine = load_default_opacity_table()
    fine_lo = fine["nu_min"]
    fine_hi = fine["nu_max"]
    fine_kappa = fine["kappa"]

    ratio = (COARSE_EMAX_KEV / COARSE_EMIN_KEV) ** (1.0 / N_COARSE)
    coarse_bounds = np.empty(N_COARSE + 1)
    coarse_bounds[0] = COARSE_EMIN_KEV
    for g in range(N_COARSE):
        coarse_bounds[g + 1] = coarse_bounds[g] * ratio
    coarse_bounds[N_COARSE] = COARSE_EMAX_KEV

    coarse_kappa = np.zeros(N_COARSE)
    for gp in range(N_COARSE):
        new_lo = coarse_bounds[gp]
        new_hi = coarse_bounds[gp + 1]
        numerator = 0.0
        denominator = 0.0
        for g in range(N_FINE):
            overlap_lo = max(new_lo, fine_lo[g])
            overlap_hi = min(new_hi, fine_hi[g])
            if overlap_hi <= overlap_lo:
                continue
            B_overlap = planck_group_integral_keV(overlap_lo, overlap_hi, T_COLLAPSE_KEV)
            numerator += fine_kappa[g] * B_overlap
            denominator += B_overlap
        coarse_kappa[gp] = numerator / denominator if denominator > 0.0 else 0.0

    return {
        "group": np.arange(N_COARSE, dtype=int),
        "nu_min": coarse_bounds[:-1].copy(),
        "nu_max": coarse_bounds[1:].copy(),
        "kappa": coarse_kappa,
    }


def read_spectrum(path):
    """Parse the moving_slab_mc_32_spectrum.txt output."""
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
    """Compute the semi-analytic reference spectrum on the collapsed 32-group grid."""
    params = BenchmarkParams(benchmark_type="original_vacuum")
    collapsed = collapse_opacity_planck()
    rows = compute_groups(params, collapsed)
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
    parser = argparse.ArgumentParser(description="Moving slab MC 32-group spectrum check")
    parser.add_argument("--spectrum", required=True,
                        help="Path to moving_slab_mc_32_spectrum.txt")
    parser.add_argument("--max-ferror", type=float, default=0.30,
                        help="Maximum allowed energy-weighted fractional error")
    parser.add_argument("--plot-dir", default=None,
                        help="Directory for plots (default: same as spectrum)")
    args = parser.parse_args()

    meta, cols = read_spectrum(args.spectrum)

    G = cols.shape[0]
    nu_min_keV = cols[:, 1]
    nu_max_keV = cols[:, 2]
    # column 3 is kappa, column 4 is Eg_time_avg
    eg_time_avg = cols[:, 4]

    code_erad = eg_to_erad_per_keV(eg_time_avg, nu_min_keV, nu_max_keV)

    print("Computing semi-analytic reference (32-group collapsed, this may take a minute)...")
    ref_erad = compute_reference()

    ferr = energy_weighted_fractional_error(code_erad, ref_erad)
    l1 = relative_l1(code_erad, ref_erad)

    print(f"MOVING_SLAB_MC_32_FERROR={ferr:.8e}")
    print(f"MOVING_SLAB_MC_32_L1={l1:.8e}")
    print(f"MOVING_SLAB_MC_32_MAX_FERROR={args.max_ferror:.8e}")
    print(f"Moving slab MC 32-group check:")
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
                      "-", lw=1.5, color="C0", label="Semi-analytic (32-group)")
        if np.any(pos_code):
            ax.loglog(nu_center[pos_code], code_erad[pos_code],
                      "x", ms=5, mew=1.0, color="C1", label="MC simulation (32-group)")

        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel(r"$E_{r,g}$ (GJ/cm$^3$/keV)")
        ax.set_title(f"Moving slab MC (32-group, original vacuum)\n"
                     f"F-error = {ferr:.4f}, L1 = {l1:.4f}")
        ax.legend(fontsize=9)
        ax.set_xlim(1e-2, 20)
        ax.set_ylim(1e-7, 2e-3)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()

        comp_base = os.path.join(plot_dir, "moving_slab_mc_32_comparison")
        fig.savefig(comp_base + ".png", dpi=200)
        fig.savefig(comp_base + ".pdf")
        plt.close(fig)
        print(f"  Plot: {comp_base}.png")
        print(f"  Plot: {comp_base}.pdf")

        collapsed = collapse_opacity_planck()
        op_lo = collapsed["nu_min"]
        op_hi = collapsed["nu_max"]
        op_kappa = collapsed["kappa"]
        op_center = np.sqrt(op_lo * op_hi)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.step(op_lo, op_kappa, where="post", lw=1.2, color="C3")
        ax2.scatter(op_center, op_kappa, s=24, marker="x", linewidths=1.0, color="C3", zorder=3)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Energy (keV)")
        ax2.set_ylabel(r"$\kappa$ (cm$^2$/g)")
        ax2.set_title("32-group collapsed aluminum opacity (Planck-weighted at 1 keV)")
        ax2.set_xlim(op_lo[0], op_hi[-1])
        ax2.grid(True, which="both", alpha=0.3)
        fig2.tight_layout()

        op_base = os.path.join(plot_dir, "moving_slab_mc_32_opacity")
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
