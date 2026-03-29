#!/usr/bin/env python3

import argparse
import math
import os
import sys
from importlib.machinery import SourceFileLoader

import numpy as np


def find_shock_radius(numeric):
    idx = int(np.argmax(numeric["pressure"]))
    return float(numeric["radius"][idx])


class SedovTaylorProfiles:
    """Analytical Sedov-Taylor profiles from the self-similar ODE solution.

    The profiles are built purely from the upstream state, the shock radius,
    and the simulation time via v_s = (2/5) R_s / t — no normalisation to
    numerical peak values.
    """

    def __init__(self, upstream, shock_radius, gamma, w, n, sedov_module,
                 sim_time, nip=3000):
        ssv = np.linspace(1e-6 + 1.0 / gamma, 2.0 / (gamma + 1.0), num=nip)
        self.shock_radius = shock_radius
        self.upstream = upstream

        rho_0 = upstream["density"]
        v_s = (2.0 / 5.0) * shock_radius / sim_time

        self.radius_table = np.array(
            [shock_radius * sedov_module.vtoz(v, w, gamma, n) for v in ssv])
        self.density_table = np.array(
            [rho_0 * sedov_module.vtod(v, w, gamma, n) for v in ssv])
        self.pressure_table = np.array(
            [rho_0 * v_s ** 2 * sedov_module.vtop(v, w, gamma, n) for v in ssv])
        self.velocity_table = np.array(
            [v_s * sedov_module.vtoz(v, w, gamma, n) * v for v in ssv])

    def calc(self, field: str, r: float) -> float:
        if r > self.shock_radius:
            return self.upstream[field]
        if field == "density":
            return float(np.interp(r, self.radius_table, self.density_table))
        if field == "pressure":
            return float(np.interp(r, self.radius_table, self.pressure_table))
        if field == "velocity":
            return float(np.interp(r, self.radius_table, self.velocity_table))
        raise ValueError(f"Unknown field: {field}")


def rel_l1_error(numeric: np.ndarray, analytic: np.ndarray,
                 r: np.ndarray) -> float:
    """Volume-weighted relative L1 error (weights ∝ r² for spherical shells)."""
    mask = np.abs(numeric) > 0.01 * np.max(np.abs(numeric))
    if np.sum(mask) < 2:
        mask = np.ones(len(numeric), dtype=bool)
    weights = r[mask] ** 2
    total_w = np.sum(weights)
    if total_w == 0:
        return float("inf")
    return float(
        np.sum(weights * np.abs(numeric[mask] - analytic[mask])
               / np.abs(numeric[mask]))
        / total_w
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Sedov 3D profile against exact Sedov-Taylor ODE profile.")
    parser.add_argument("--profile", required=True, help="Path to sedov_profile.txt")
    parser.add_argument("--rich-root", required=True, help="Repository root for analytic/sedov_taylor.py")
    parser.add_argument("--max-density-rel-l1", type=float, default=0.50)
    parser.add_argument("--max-pressure-rel-l1", type=float, default=0.30)
    parser.add_argument("--max-velocity-rel-l1", type=float, default=0.60)
    parser.add_argument("--sim-time", type=float, default=0.0075,
                        help="Simulation end time (for analytical v_s = 2/5 R_s/t)")
    args = parser.parse_args()

    sedov_path = os.path.join(args.rich_root, "analytic", "sedov_taylor.py")
    sedov = SourceFileLoader("sedov_taylor", sedov_path).load_module()

    raw = np.loadtxt(args.profile)
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    numeric = {
        "radius": raw[:, 0],
        "density": raw[:, 1],
        "pressure": raw[:, 2],
        "velocity": raw[:, 3],
    }
    shock_radius = find_shock_radius(numeric)
    r = numeric["radius"]

    far_mask = r > 0.8 * float(np.max(r))
    if not np.any(far_mask):
        far_mask = r > np.median(r)

    upstream = {
        "density": float(np.median(numeric["density"][far_mask])),
        "pressure": float(np.median(numeric["pressure"][far_mask])),
        "velocity": float(np.median(numeric["velocity"][far_mask])),
    }

    st = SedovTaylorProfiles(upstream, shock_radius, 5.0 / 3.0, 0.0, 3, sedov,
                             args.sim_time)
    compare_mask = r <= st.shock_radius
    if not np.any(compare_mask):
        print("No cells found within shock radius for Sedov comparison", file=sys.stderr)
        return 1

    r_inner = r[compare_mask]
    analytic_density = np.array([st.calc("density", ri) for ri in r_inner])
    analytic_pressure = np.array([st.calc("pressure", ri) for ri in r_inner])
    analytic_velocity = np.array([st.calc("velocity", ri) for ri in r_inner])

    density_rel_l1 = rel_l1_error(numeric["density"][compare_mask],
                                  analytic_density, r_inner)
    pressure_rel_l1 = rel_l1_error(numeric["pressure"][compare_mask],
                                   analytic_pressure, r_inner)
    velocity_rel_l1 = rel_l1_error(numeric["velocity"][compare_mask],
                                   analytic_velocity, r_inner)

    print(f"SEDOV_DENSITY_REL_L1={density_rel_l1:.8e}")
    print(f"SEDOV_PRESSURE_REL_L1={pressure_rel_l1:.8e}")
    print(f"SEDOV_VELOCITY_REL_L1={velocity_rel_l1:.8e}")
    print(f"SEDOV_MAX_DENSITY_REL_L1={args.max_density_rel_l1:.8e}")
    print(f"SEDOV_MAX_PRESSURE_REL_L1={args.max_pressure_rel_l1:.8e}")
    print(f"SEDOV_MAX_VELOCITY_REL_L1={args.max_velocity_rel_l1:.8e}")

    if density_rel_l1 > args.max_density_rel_l1:
        print("Sedov density profile exceeds tolerance", file=sys.stderr)
        return 1
    if pressure_rel_l1 > args.max_pressure_rel_l1:
        print("Sedov pressure profile exceeds tolerance", file=sys.stderr)
        return 1
    if velocity_rel_l1 > args.max_velocity_rel_l1:
        print("Sedov velocity profile exceeds tolerance", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
