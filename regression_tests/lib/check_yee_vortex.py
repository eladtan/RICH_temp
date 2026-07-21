#!/usr/bin/env python3
"""
Check Yee isentropic vortex profile against the analytical (initial condition) solution.

Computes per-cell density and compares to the known stationary solution via a
volume-weighted L1 norm.
"""
import argparse
import sys
import numpy as np


GAMMA = 1.4
BETA = 5.0


def vortex_density_analytic(x, y):
    """Analytical density for the isentropic vortex (Yee et al. 1999)."""
    r2 = x**2 + y**2
    T = 1.0 - (GAMMA - 1.0) * BETA**2 / (8.0 * GAMMA * np.pi**2) * np.exp(1.0 - r2)
    return T ** (1.0 / (GAMMA - 1.0))


def main():
    parser = argparse.ArgumentParser(description="Check Yee isentropic vortex profile")
    parser.add_argument("--profile", type=str, required=True)
    parser.add_argument("--max-density-l1", type=float, default=0.05)
    args = parser.parse_args()

    data = np.loadtxt(args.profile)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    x = data[:, 0]
    y = data[:, 1]
    vol = data[:, 2]
    rho_sim = data[:, 3]

    rho_exact = vortex_density_analytic(x, y)

    total_vol = np.sum(vol)
    if total_vol <= 0:
        print("DENSITY_L1=inf")
        print("PASS=0")
        sys.exit(1)

    l1 = np.sum(vol * np.abs(rho_sim - rho_exact)) / total_vol

    passed = l1 <= args.max_density_l1

    print(f"DENSITY_L1={l1:.6e}")
    print(f"PASS={'1' if passed else '0'}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
