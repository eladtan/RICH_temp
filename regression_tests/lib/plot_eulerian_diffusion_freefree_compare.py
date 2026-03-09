#!/usr/bin/env python3
"""
Overlay 256-cell and 32-cell free-free diffusion profiles.
"""

import argparse
import os
import sys

import numpy as np


def load_profile(path: str):
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    if raw.shape[1] < 5:
        raise ValueError(f"profile must have 5 columns (x density Tgas Trad vx): {path}")

    x = raw[:, 0]
    tgas_k = raw[:, 2]
    trad_k = raw[:, 3]
    vx = raw[:, 4]

    kev_per_kelvin = 8.617333262145e-8
    tgas_kev = tgas_k * kev_per_kelvin
    trad_kev = trad_k * kev_per_kelvin

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(tgas_kev)) or not np.all(np.isfinite(trad_kev)) or not np.all(np.isfinite(vx)):
        raise ValueError(f"non-finite values in profile: {path}")
    if np.any(tgas_kev <= 0) or np.any(trad_kev <= 0):
        raise ValueError(f"Tgas/Trad must be positive for log-y plot: {path}")

    order = np.argsort(x)
    return x[order], tgas_kev[order], trad_kev[order], vx[order]


def main() -> int:
    parser = argparse.ArgumentParser(description="Overlay free-free diffusion profiles for 256 vs 32 cells.")
    parser.add_argument("--profile-256", required=True, help="Path to 256-cell temperature_profile.txt")
    parser.add_argument("--profile-32", required=True, help="Path to 32-cell temperature_profile.txt")
    parser.add_argument("--output-dir", required=True, help="Directory for output figures")
    args = parser.parse_args()

    x_256, tgas_256, trad_256, vx_256 = load_profile(args.profile_256)
    x_32, tgas_32, trad_32, vx_32 = load_profile(args.profile_32)
    raw_256 = np.loadtxt(args.profile_256)
    raw_32 = np.loadtxt(args.profile_32)
    if raw_256.ndim == 1:
        raw_256 = np.expand_dims(raw_256, axis=0)
    if raw_32.ndim == 1:
        raw_32 = np.expand_dims(raw_32, axis=0)
    density_256 = raw_256[:, 1]
    density_32 = raw_32[:, 1]
    order_256 = np.argsort(raw_256[:, 0])
    order_32 = np.argsort(raw_32[:, 0])
    density_256 = density_256[order_256]
    density_32 = density_32[order_32]
    if np.any(density_256 <= 0) or np.any(density_32 <= 0):
        raise ValueError("density must be positive for log-y plot")

    os.makedirs(args.output_dir, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_256, tgas_256, color="tab:red", linewidth=1.3, label="Tgas 256")
    ax.plot(x_32, tgas_32, color="tab:orange", linewidth=1.1, linestyle="--", label="Tgas 32")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Gas temperature [keV]")
    ax.set_title("Free-free diffusion: Tgas(x), 256 vs 32")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    tgas_png = os.path.join(args.output_dir, "temperature_vs_x_compare_256_32.png")
    tgas_pdf = os.path.join(args.output_dir, "temperature_vs_x_compare_256_32.pdf")
    fig.savefig(tgas_png, dpi=150)
    fig.savefig(tgas_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_256, trad_256, color="tab:blue", linewidth=1.3, label="Trad 256")
    ax.plot(x_32, trad_32, color="tab:cyan", linewidth=1.1, linestyle="--", label="Trad 32")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Radiation temperature [keV]")
    ax.set_title("Free-free diffusion: Trad(x), 256 vs 32")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    trad_png = os.path.join(args.output_dir, "trad_vs_x_compare_256_32.png")
    trad_pdf = os.path.join(args.output_dir, "trad_vs_x_compare_256_32.pdf")
    fig.savefig(trad_png, dpi=150)
    fig.savefig(trad_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_256, density_256, color="tab:green", linewidth=1.3, label="Density 256")
    ax.plot(x_32, density_32, color="tab:olive", linewidth=1.1, linestyle="--", label="Density 32")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Density [g/cm^3]")
    ax.set_title("Free-free diffusion: density(x), 256 vs 32")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    density_png = os.path.join(args.output_dir, "density_vs_x_compare_256_32.png")
    density_pdf = os.path.join(args.output_dir, "density_vs_x_compare_256_32.pdf")
    fig.savefig(density_png, dpi=150)
    fig.savefig(density_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_256, vx_256, color="tab:purple", linewidth=1.3, label="vx 256")
    ax.plot(x_32, vx_32, color="tab:pink", linewidth=1.1, linestyle="--", label="vx 32")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Velocity x [cm/s]")
    ax.set_title("Free-free diffusion: vx(x), 256 vs 32")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    velocity_png = os.path.join(args.output_dir, "velocity_vs_x_compare_256_32.png")
    velocity_pdf = os.path.join(args.output_dir, "velocity_vs_x_compare_256_32.pdf")
    fig.savefig(velocity_png, dpi=150)
    fig.savefig(velocity_pdf)
    plt.close(fig)

    print(f"FREEFREE_COMPARE_TGAS_PNG={tgas_png}")
    print(f"FREEFREE_COMPARE_TRAD_PNG={trad_png}")
    print(f"FREEFREE_COMPARE_DENSITY_PNG={density_png}")
    print(f"FREEFREE_COMPARE_VELOCITY_PNG={velocity_png}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
