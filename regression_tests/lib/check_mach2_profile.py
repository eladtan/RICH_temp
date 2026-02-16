#!/usr/bin/env python3
"""
Compare a Mach2 radiative shock numerical profile against the LTE
analytical solution from the radiative_shock submodule.

Usage:
    python3 check_mach2_profile.py \
        --profile mach2_profile.txt \
        --rich-root /path/to/RICH \
        --time 2e-2 \
        [--max-density-rel-l1 0.5] \
        [--max-temperature-rel-l1 0.5]
"""

import argparse
import os
import sys

import numpy as np


def rel_l1_error(numeric, analytic):
    """Relative L1 error: mean(|num - ana|) / mean(|ana|)."""
    denom = float(np.mean(np.abs(analytic)))
    if denom < 1e-30:
        denom = 1e-30
    return float(np.mean(np.abs(numeric - analytic)) / denom)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Mach2 radiative shock profile against LTE analytical solution."
    )
    parser.add_argument("--profile", required=True, help="Path to mach2_profile.txt")
    parser.add_argument("--rich-root", required=True, help="Repository root (for analysis_files/)")
    parser.add_argument("--time", type=float, default=0.01, help="Simulation end time [s]")
    parser.add_argument("--max-density-rel-l1", type=float, default=0.50)
    parser.add_argument("--max-temperature-rel-l1", type=float, default=0.50)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load numerical profile  (columns: x  density  temperature)
    # ------------------------------------------------------------------
    raw = np.loadtxt(args.profile)
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    x_num = raw[:, 0]
    rho_num = raw[:, 1]
    T_num = raw[:, 2]

    # ------------------------------------------------------------------
    # Import the analytical solver from the radiative_shock submodule
    # ------------------------------------------------------------------
    solver_dir = os.path.join(args.rich_root, "analysis_files", "radiative_shock")
    if not os.path.isdir(solver_dir):
        print(f"Analytical solver directory not found: {solver_dir}", file=sys.stderr)
        return 1
    sys.path.insert(0, solver_dir)
    from lte_radiative_shock import LTERadiativeShock  # noqa: E402

    # ------------------------------------------------------------------
    # Physical parameters  (must match the C++ test)
    # ------------------------------------------------------------------
    gamma = 5.0 / 3.0
    k_boltz = 1.380649e-16   # erg/K
    mu = 1.67e-24            # g  (mean particle mass)
    cv = k_boltz / (mu * (gamma - 1.0))

    rho_left = 5.45887e-13   # g/cc
    v_left = 2.3547e5        # cm/s
    T_left = 100.0           # K
    sigma_ross = 0.848902    # 1/cm  (D = c / (3*sigma_ross))

    cs_left = np.sqrt(gamma * (gamma - 1.0) * cv * T_left)
    M0 = v_left / cs_left

    # ------------------------------------------------------------------
    # Solve the analytical LTE radiative shock profile
    # ------------------------------------------------------------------
    solver = LTERadiativeShock(
        M0=M0,
        gamma=gamma,
        sigma_ross=lambda T, rho: sigma_ross,
        cv=cv,
        rho_left=rho_left,
        v_left=v_left,
        T_left=T_left,
    )

    solution = solver.solve_profiles(time=args.time, x=x_num)
    rho_ana = solution["density"]
    T_ana = solution["temperature"]

    # ------------------------------------------------------------------
    # Compute relative L1 errors
    # ------------------------------------------------------------------
    density_rel_l1 = rel_l1_error(rho_num, rho_ana)
    temperature_rel_l1 = rel_l1_error(T_num, T_ana)

    print(f"MACH2_DENSITY_REL_L1={density_rel_l1:.8e}")
    print(f"MACH2_TEMPERATURE_REL_L1={temperature_rel_l1:.8e}")
    print(f"MACH2_MAX_DENSITY_REL_L1={args.max_density_rel_l1:.8e}")
    print(f"MACH2_MAX_TEMPERATURE_REL_L1={args.max_temperature_rel_l1:.8e}")

    failed = False
    if density_rel_l1 > args.max_density_rel_l1:
        print(
            f"Mach2 density profile exceeds tolerance "
            f"({density_rel_l1:.6e} > {args.max_density_rel_l1:.6e})",
            file=sys.stderr,
        )
        failed = True
    if temperature_rel_l1 > args.max_temperature_rel_l1:
        print(
            f"Mach2 temperature profile exceeds tolerance "
            f"({temperature_rel_l1:.6e} > {args.max_temperature_rel_l1:.6e})",
            file=sys.stderr,
        )
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
