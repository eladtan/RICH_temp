#!/usr/bin/env python3
"""
Validate and plot the 1D Eulerian free-free diffusion regression output.
"""

import argparse
import os
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Tgas(x) figure for eulerian_diffusion_freefree_1d."
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Path to temperature_profile.txt (columns: x density Tgas Trad vx).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where temperature_vs_x.{png,pdf} will be written.",
    )
    parser.add_argument(
        "--figure-suffix",
        default="",
        help="Optional suffix appended to figure basenames (e.g. _32).",
    )
    args = parser.parse_args()

    raw = np.loadtxt(args.profile)
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)

    if raw.shape[1] < 5:
        print("profile must have at least 5 columns (x, density, Tgas, Trad, vx)", file=sys.stderr)
        return 1

    if raw.shape[0] < 2:
        print("profile must contain at least two rows", file=sys.stderr)
        return 1

    x = raw[:, 0]
    density = raw[:, 1]
    tgas = raw[:, 2]
    trad = raw[:, 3]
    vx = raw[:, 4]
    kev_per_kelvin = 8.617333262145e-8
    tgas_kev = tgas * kev_per_kelvin
    trad_kev = trad * kev_per_kelvin

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(density)) or not np.all(np.isfinite(tgas_kev)) or not np.all(np.isfinite(trad_kev)) or not np.all(np.isfinite(vx)):
        print("profile contains non-finite x/density/Tgas/Trad/vx values", file=sys.stderr)
        return 1

    order = np.argsort(x)
    x = x[order]
    density = density[order]
    tgas_kev = tgas_kev[order]
    trad_kev = trad_kev[order]
    vx = vx[order]

    if np.any(density <= 0) or np.any(tgas_kev <= 0) or np.any(trad_kev <= 0):
        print("density/Tgas/Trad must be positive for log-scale plotting", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = args.figure_suffix
    png_path = os.path.join(args.output_dir, f"temperature_vs_x{suffix}.png")
    pdf_path = os.path.join(args.output_dir, f"temperature_vs_x{suffix}.pdf")
    trad_png_path = os.path.join(args.output_dir, f"trad_vs_x{suffix}.png")
    trad_pdf_path = os.path.join(args.output_dir, f"trad_vs_x{suffix}.pdf")
    density_png_path = os.path.join(args.output_dir, f"density_vs_x{suffix}.png")
    density_pdf_path = os.path.join(args.output_dir, f"density_vs_x{suffix}.pdf")
    velocity_png_path = os.path.join(args.output_dir, f"velocity_vs_x{suffix}.png")
    velocity_pdf_path = os.path.join(args.output_dir, f"velocity_vs_x{suffix}.pdf")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, tgas_kev, color="tab:red", linewidth=1.2, label="Tgas")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Gas temperature [keV]")
    ax.set_title("1D Eulerian diffusion (free-free): Tgas vs x")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, trad_kev, color="tab:blue", linewidth=1.2, label="Trad")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Radiation temperature [keV]")
    ax.set_title("1D Eulerian diffusion (free-free): Trad vs x")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(trad_png_path, dpi=150)
    fig.savefig(trad_pdf_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, density, color="tab:green", linewidth=1.2, label="Density")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Density [g/cm^3]")
    ax.set_title("1D Eulerian diffusion (free-free): density vs x")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(density_png_path, dpi=150)
    fig.savefig(density_pdf_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, vx, color="tab:purple", linewidth=1.2, label="vx")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Velocity x [cm/s]")
    ax.set_title("1D Eulerian diffusion (free-free): vx vs x")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(velocity_png_path, dpi=150)
    fig.savefig(velocity_pdf_path)
    plt.close(fig)

    print(f"FREEFREE_PROFILE_POINTS={x.size}")
    print(f"FREEFREE_TGAS_MIN_KEV={np.min(tgas_kev):.8e}")
    print(f"FREEFREE_TGAS_MAX_KEV={np.max(tgas_kev):.8e}")
    print(f"FREEFREE_TRAD_MIN_KEV={np.min(trad_kev):.8e}")
    print(f"FREEFREE_TRAD_MAX_KEV={np.max(trad_kev):.8e}")
    print(f"FREEFREE_PLOT_PNG={png_path}")
    print(f"FREEFREE_PLOT_PDF={pdf_path}")
    print(f"FREEFREE_TRAD_PLOT_PNG={trad_png_path}")
    print(f"FREEFREE_TRAD_PLOT_PDF={trad_pdf_path}")
    print(f"FREEFREE_DENSITY_PLOT_PNG={density_png_path}")
    print(f"FREEFREE_DENSITY_PLOT_PDF={density_pdf_path}")
    print(f"FREEFREE_VELOCITY_PLOT_PNG={velocity_png_path}")
    print(f"FREEFREE_VELOCITY_PLOT_PDF={velocity_pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
