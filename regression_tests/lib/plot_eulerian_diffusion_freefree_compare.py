#!/usr/bin/env python3
"""
Overlay 256, 256-limited, 32, and 32-limited free-free diffusion profiles.
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
    density = raw[:, 1]
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
    if np.any(density <= 0):
        raise ValueError(f"density must be positive for log-y plot: {path}")

    order = np.argsort(x)
    return x[order], density[order], tgas_kev[order], trad_kev[order], vx[order]


def main() -> int:
    parser = argparse.ArgumentParser(description="Overlay free-free diffusion profiles for 256, 256-limited, 32, and 32-limited cells.")
    parser.add_argument("--profile-256", required=True, help="Path to 256-cell temperature_profile.txt")
    parser.add_argument("--profile-256-limited", required=True, help="Path to 256-cell-limited temperature_profile.txt")
    parser.add_argument("--profile-32", required=True, help="Path to 32-cell temperature_profile.txt")
    parser.add_argument("--profile-32-limited", required=True, help="Path to 32-cell-limited temperature_profile.txt")
    parser.add_argument("--output-dir", required=True, help="Directory for output figures")
    args = parser.parse_args()

    x_256, density_256, tgas_256, trad_256, vx_256 = load_profile(args.profile_256)
    x_256_limited, density_256_limited, tgas_256_limited, trad_256_limited, vx_256_limited = load_profile(args.profile_256_limited)
    x_32, density_32, tgas_32, trad_32, vx_32 = load_profile(args.profile_32)
    x_32_limited, density_32_limited, tgas_32_limited, trad_32_limited, vx_32_limited = load_profile(args.profile_32_limited)

    os.makedirs(args.output_dir, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_256, tgas_256, color="tab:red", linewidth=1.3, label="Tgas 256")
    ax.plot(x_256_limited, tgas_256_limited, color="tab:orange", linewidth=1.2, linestyle="-.", label="Tgas 256 limited")
    ax.plot(x_32, tgas_32, color="tab:brown", linewidth=1.1, linestyle="--", label="Tgas 32")
    ax.plot(x_32_limited, tgas_32_limited, color="tab:pink", linewidth=1.1, linestyle=":", label="Tgas 32 limited")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Gas temperature [keV]")
    ax.set_title("Free-free diffusion: Tgas(x), 256/256 limited/32/32 limited")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    tgas_png = os.path.join(args.output_dir, "temperature_vs_x_compare_256_256_limited_32_32_limited.png")
    tgas_pdf = os.path.join(args.output_dir, "temperature_vs_x_compare_256_256_limited_32_32_limited.pdf")
    fig.savefig(tgas_png, dpi=150)
    fig.savefig(tgas_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_256, trad_256, color="tab:blue", linewidth=1.3, label="Trad 256")
    ax.plot(x_256_limited, trad_256_limited, color="tab:cyan", linewidth=1.2, linestyle="-.", label="Trad 256 limited")
    ax.plot(x_32, trad_32, color="tab:gray", linewidth=1.1, linestyle="--", label="Trad 32")
    ax.plot(x_32_limited, trad_32_limited, color="tab:olive", linewidth=1.1, linestyle=":", label="Trad 32 limited")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Radiation temperature [keV]")
    ax.set_title("Free-free diffusion: Trad(x), 256/256 limited/32/32 limited")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    trad_png = os.path.join(args.output_dir, "trad_vs_x_compare_256_256_limited_32_32_limited.png")
    trad_pdf = os.path.join(args.output_dir, "trad_vs_x_compare_256_256_limited_32_32_limited.pdf")
    fig.savefig(trad_png, dpi=150)
    fig.savefig(trad_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_256, density_256, color="tab:green", linewidth=1.3, label="Density 256")
    ax.plot(x_256_limited, density_256_limited, color="tab:olive", linewidth=1.2, linestyle="-.", label="Density 256 limited")
    ax.plot(x_32, density_32, color="tab:purple", linewidth=1.1, linestyle="--", label="Density 32")
    ax.plot(x_32_limited, density_32_limited, color="tab:brown", linewidth=1.1, linestyle=":", label="Density 32 limited")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Density [g/cm^3]")
    ax.set_title("Free-free diffusion: density(x), 256/256 limited/32/32 limited")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    density_png = os.path.join(args.output_dir, "density_vs_x_compare_256_256_limited_32_32_limited.png")
    density_pdf = os.path.join(args.output_dir, "density_vs_x_compare_256_256_limited_32_32_limited.pdf")
    fig.savefig(density_png, dpi=150)
    fig.savefig(density_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_256, vx_256, color="tab:purple", linewidth=1.3, label="vx 256")
    ax.plot(x_256_limited, vx_256_limited, color="tab:green", linewidth=1.2, linestyle="-.", label="vx 256 limited")
    ax.plot(x_32, vx_32, color="tab:pink", linewidth=1.1, linestyle="--", label="vx 32")
    ax.plot(x_32_limited, vx_32_limited, color="tab:orange", linewidth=1.1, linestyle=":", label="vx 32 limited")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Velocity x [cm/s]")
    ax.set_title("Free-free diffusion: vx(x), 256/256 limited/32/32 limited")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    velocity_png = os.path.join(args.output_dir, "velocity_vs_x_compare_256_256_limited_32_32_limited.png")
    velocity_pdf = os.path.join(args.output_dir, "velocity_vs_x_compare_256_256_limited_32_32_limited.pdf")
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
