#!/usr/bin/env python3
"""
Validate the Rayleigh-Taylor instability regression test.

Reads the z-kinetic-energy time series, fits the linear growth rate,
compares to the analytical prediction, and optionally generates diagnostic
plots (Ek_z vs time with fit, density slice).
"""

import argparse
import os
import sys

import numpy as np


def analytical_growth_rate(rho_heavy=2.0, rho_light=1.0, g=0.5, k=2.0 * np.pi):
    """sigma = sqrt(A * g * k), where A is the Atwood number."""
    atwood = (rho_heavy - rho_light) / (rho_heavy + rho_light)
    return np.sqrt(atwood * g * k)


FIT_T_MIN = 2.0
FIT_T_MAX = 3.0


def fit_growth_rate(time, ekz, t_min=FIT_T_MIN, t_max=FIT_T_MAX):
    """Fit log(Ek_z) = a + 2*sigma*t in a fixed time window.

    Returns (sigma_fit, log_C_fit, mask) where mask marks the fitting window.
    """
    mask = (time >= t_min) & (time <= t_max) & (ekz > 0)
    if np.sum(mask) < 3:
        mask = ekz > 0

    log_ek = np.log(ekz[mask])
    t_fit = time[mask]

    coeffs = np.polyfit(t_fit, log_ek, 1)
    two_sigma = coeffs[0]
    log_C = coeffs[1]
    return two_sigma / 2.0, log_C, mask


def make_plots(time, ekz, sigma_fit, log_C, mask, sigma_analytical,
               slice_file, plot_dir):
    """Generate diagnostic plots and save to plot_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)

    # --- Plot 1: Ek_z(t) with fit ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(time, ekz, "k-", linewidth=1.2, label="$E_{k,z}$ (RICH)")

    t_line = np.linspace(FIT_T_MIN, FIT_T_MAX, 200)
    ek_fit = np.exp(log_C + 2.0 * sigma_fit * t_line)
    ax.semilogy(t_line, ek_fit, "r--", linewidth=2,
                label=(f"Best fit: $\\sigma$ = {sigma_fit:.4f}"))

    ek_analytical = np.exp(log_C + 2.0 * sigma_analytical * t_line)
    ax.semilogy(t_line, ek_analytical, "b:", linewidth=1.5,
                label=(f"Analytical: $\\sigma$ = {sigma_analytical:.4f}"))

    ax.set_xlabel("Time")
    ax.set_ylabel("$E_{k,z}$")
    ax.set_title("Rayleigh-Taylor -- Vertical Kinetic Energy")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "rayleigh_taylor_mpi_ekz.png"), dpi=150)
    fig.savefig(os.path.join(plot_dir, "rayleigh_taylor_mpi_ekz.pdf"))
    plt.close(fig)

    # --- Plot 2: density slice ---
    if slice_file and os.path.isfile(slice_file):
        raw = np.loadtxt(slice_file)
        if raw.ndim == 1:
            raw = np.expand_dims(raw, axis=0)
        x_s = raw[:, 0]
        z_s = raw[:, 1]
        rho_s = raw[:, 2]

        fig2, ax2 = plt.subplots(figsize=(5, 10))
        sc = ax2.scatter(x_s, z_s, c=rho_s, s=0.5, cmap="RdBu_r",
                         vmin=0.8, vmax=2.2, edgecolors="none")
        fig2.colorbar(sc, ax=ax2, label="Density")
        ax2.set_xlabel("x")
        ax2.set_ylabel("z")
        ax2.set_title("RT Density Slice (xz plane, y=0.5)")
        ax2.set_aspect("equal")
        fig2.tight_layout()
        fig2.savefig(os.path.join(plot_dir, "rayleigh_taylor_mpi_slice.png"),
                     dpi=150)
        fig2.savefig(os.path.join(plot_dir, "rayleigh_taylor_mpi_slice.pdf"))
        plt.close(fig2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate RT instability growth rate against linear theory.")
    parser.add_argument("--profile", required=True,
                        help="Path to rt_kinetic_energy.txt")
    parser.add_argument("--slice", default=None,
                        help="Path to rt_density_slice.txt")
    parser.add_argument("--max-growth-rate-rel-error", type=float, default=0.25,
                        help="Maximum allowed relative error in fitted growth rate")
    parser.add_argument("--plot-dir", default=None,
                        help="Directory to save diagnostic plots")
    args = parser.parse_args()

    raw = np.loadtxt(args.profile)
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    time = raw[:, 0]
    ekz = raw[:, 1]

    positive = ekz > 0
    if np.sum(positive) < 5:
        print("Too few positive Ek_z entries for fitting", file=sys.stderr)
        return 1
    time = time[positive]
    ekz = ekz[positive]

    sigma_analytical = analytical_growth_rate()
    sigma_fit, log_C, mask = fit_growth_rate(time, ekz)

    rel_error = abs(sigma_fit - sigma_analytical) / sigma_analytical

    print(f"RT_SIGMA_FIT={sigma_fit:.8e}")
    print(f"RT_SIGMA_ANALYTICAL={sigma_analytical:.8e}")
    print(f"RT_GROWTH_RATE_REL_ERROR={rel_error:.8e}")
    print(f"RT_MAX_GROWTH_RATE_REL_ERROR={args.max_growth_rate_rel_error:.8e}")
    print(f"RT_FIT_WINDOW_SIZE={int(np.sum(mask))}")

    if args.plot_dir:
        try:
            make_plots(time, ekz, sigma_fit, log_C, mask,
                       sigma_analytical, args.slice, args.plot_dir)
        except Exception as exc:
            print(f"Warning: plot generation failed: {exc}", file=sys.stderr)

    if rel_error > args.max_growth_rate_rel_error:
        print(f"RT growth rate relative error {rel_error:.4f} exceeds "
              f"threshold {args.max_growth_rate_rel_error:.4f}",
              file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
