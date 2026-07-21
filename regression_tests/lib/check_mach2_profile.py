#!/usr/bin/env python3
"""
Compare a Mach2 radiative shock numerical profile against the NLTE
analytical solution from analysis_files/radiative_shock.

Usage:
    python3 check_mach2_profile.py \
        --profile mach2_profile.txt \
        --rich-root /path/to/RICH \
        --time 2e-2 \
        [--max-density-rel-l1 0.5] \
        [--max-temperature-rel-l1 0.5] \
        [--max-trad-rel-l1 0.5]
"""

import argparse
import os
import sys

import numpy as np


def rel_l1_error(numeric, analytic):
    """Per-cell relative L1 error: mean(|num - ana| / |num|)."""
    mask = np.abs(numeric) > 0.01 * np.max(np.abs(numeric))
    if np.sum(mask) < 2:
        mask = np.ones(len(numeric), dtype=bool)
    return float(np.mean(np.abs(numeric[mask] - analytic[mask]) / np.abs(numeric[mask])))


def main():
    parser = argparse.ArgumentParser(
        description="Compare Mach2 radiative shock profile against NLTE analytical solution."
    )
    parser.add_argument("--profile", required=True, help="Path to mach2_profile.txt")
    parser.add_argument("--rich-root", required=True, help="Repository root (for analysis_files/)")
    parser.add_argument("--time", type=float, default=0.01, help="Simulation end time [s]")
    parser.add_argument("--max-density-rel-l1", type=float, default=0.025)
    parser.add_argument("--max-temperature-rel-l1", type=float, default=0.025)
    parser.add_argument("--max-trad-rel-l1", type=float, default=0.025)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load numerical profile  (columns: x  density  Tgas  [Trad])
    # ------------------------------------------------------------------
    raw = np.loadtxt(args.profile)
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    x_num = raw[:, 0]
    rho_num = raw[:, 1]
    T_num = raw[:, 2]
    Trad_num = raw[:, 3] if raw.shape[1] > 3 else None

    # ------------------------------------------------------------------
    # Import the analytical solver from analysis_files/radiative_shock
    # ------------------------------------------------------------------
    solver_dir = os.path.join(args.rich_root, "analysis_files", "radiative_shock")
    if not os.path.isdir(solver_dir):
        print(f"Analytical solver directory not found: {solver_dir}", file=sys.stderr)
        return 1
    sys.path.insert(0, solver_dir)
    from nlte_radiative_shock import NLTERadiativeShock  # noqa: E402

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
    sigma_abs = 3.93e-5      # 1/cm  (Planck absorption coefficient)

    cs_left = np.sqrt(gamma * (gamma - 1.0) * cv * T_left)
    M0 = v_left / cs_left

    # ------------------------------------------------------------------
    # Solve the analytical NLTE radiative shock profile
    # ------------------------------------------------------------------
    solver = NLTERadiativeShock(
        M0=M0,
        gamma=gamma,
        sigma_ross=lambda T, rho: sigma_ross,
        sigma_abs=lambda T, rho: sigma_abs,
        cv=cv,
        rho_left=rho_left,
        v_left=v_left,
        T_left=T_left,
        eps_nlte_solver=1e-4,
    )

    solution = solver.solve_profiles(time=args.time, x=x_num)
    rho_ana = solution["density"]
    T_ana = solution["temperature"]
    Trad_ana = solution["radiation_temperature"]

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

    if Trad_num is not None:
        trad_rel_l1 = rel_l1_error(Trad_num, Trad_ana)
        print(f"MACH2_TRAD_REL_L1={trad_rel_l1:.8e}")
        print(f"MACH2_MAX_TRAD_REL_L1={args.max_trad_rel_l1:.8e}")
        if trad_rel_l1 > args.max_trad_rel_l1:
            print(
                f"Mach2 Trad profile exceeds tolerance "
                f"({trad_rel_l1:.6e} > {args.max_trad_rel_l1:.6e})",
                file=sys.stderr,
            )
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
