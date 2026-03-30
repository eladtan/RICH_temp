#!/usr/bin/env python3
"""
Compare Densmore 2012 heterogeneous MC regression profile against
digitized reference data from Figure 4 of Densmore et al. (2012).

Usage:
    python3 check_desmore2012_mc.py \
        --profile desmore2012_mc_profile.txt \
        --reference data/densmore2012_fig4_mc.csv \
        [--max-tgas-l1 0.05]
"""

import argparse
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Compare Densmore 2012 MC gas temperature profile against reference."
    )
    parser.add_argument("--profile", required=True, help="Path to desmore2012_mc_profile.txt")
    parser.add_argument("--reference", required=True, help="Path to reference CSV (x_cm, T_keV)")
    parser.add_argument("--max-tgas-l1", type=float, default=0.05,
                        help="Maximum L1 norm of Tgas error in keV (default: 0.05)")
    args = parser.parse_args()

    # Load simulation profile: columns are x(cm) T(K)
    raw = np.loadtxt(args.profile)
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    x_sim = raw[:, 0]
    T_sim_K = raw[:, 1]

    kev_kelvin = 1e3 * 1.602176634e-12 / 1.380649e-16
    T_sim_keV = T_sim_K / kev_kelvin

    # Load reference data: columns are x(cm), T(keV)
    ref = np.loadtxt(args.reference, delimiter=",", comments="#")
    x_ref = ref[:, 0]
    T_ref_keV = ref[:, 1]

    # Interpolate reference onto simulation x positions
    T_ref_interp = np.interp(x_sim, x_ref, T_ref_keV)

    # L1 norm in keV
    l1 = float(np.mean(np.abs(T_sim_keV - T_ref_interp)))

    print(f"DESMORE2012_MC_TGAS_L1={l1:.8e}")
    print(f"DESMORE2012_MC_MAX_TGAS_L1={args.max_tgas_l1:.8e}")

    if l1 > args.max_tgas_l1:
        print(
            f"Densmore 2012 MC gas temperature L1 exceeds tolerance "
            f"({l1:.6e} > {args.max_tgas_l1:.6e})",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: L1 = {l1:.6e} < {args.max_tgas_l1:.6e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
