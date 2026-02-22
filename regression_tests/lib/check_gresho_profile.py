#!/usr/bin/env python3
"""
Check Gresho vortex profile against the analytical (initial condition) solution.

Computes per-cell azimuthal velocity and compares to the known stationary
solution via a volume-weighted relative L1 norm over all cells.
"""
import argparse
import sys
import numpy as np


def azimuthal_velocity_analytic(r):
    """Analytical azimuthal velocity for the Gresho vortex."""
    r = np.asarray(r, dtype=float)
    vtheta = np.zeros_like(r)
    mask1 = r < 0.2
    mask2 = (r >= 0.2) & (r <= 0.4)
    vtheta[mask1] = 5.0 * r[mask1]
    vtheta[mask2] = 2.0 - 5.0 * r[mask2]
    return vtheta


def main():
    parser = argparse.ArgumentParser(description="Check Gresho vortex profile")
    parser.add_argument("--profile", type=str, required=True)
    parser.add_argument("--max-vtheta-rel-l1", type=float, default=0.1)
    args = parser.parse_args()

    data = np.loadtxt(args.profile)
    x = data[:, 0]
    y = data[:, 1]
    vol = data[:, 2]
    vx = data[:, 4]
    vy = data[:, 5]

    r = np.sqrt(x**2 + y**2)
    safe_r = np.where(r > 1e-10, r, 1e-10)
    vtheta_sim = (-vx * y + vy * x) / safe_r
    vtheta_analytic = azimuthal_velocity_analytic(r)

    valid = (r > 1e-10) & (np.abs(vtheta_analytic) > 1e-10)
    if np.sum(valid) == 0:
        print("VTHETA_REL_L1=inf")
        print("PASS=0")
        sys.exit(1)

    rel_l1 = np.sum(vol[valid] * np.abs(vtheta_sim[valid] - vtheta_analytic[valid])) / np.sum(vol[valid])

    passed = rel_l1 <= args.max_vtheta_rel_l1

    print(f"VTHETA_REL_L1={rel_l1:.6e}")
    print(f"PASS={'1' if passed else '0'}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
