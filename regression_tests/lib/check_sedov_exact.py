#!/usr/bin/env python3

import argparse
import math
import os
import sys
from importlib.machinery import SourceFileLoader

import numpy as np


def find_shock_front(numeric):
    idx = int(np.argmax(numeric["pressure"]))
    return {
        "radius": float(numeric["radius"][idx]),
        "density": float(numeric["density"][idx]),
        "pressure": float(numeric["pressure"][idx]),
        "velocity": float(numeric["velocity"][idx]),
    }


class SedovTaylorProfiles:
    def __init__(self, upstream, shock_front, gamma, w, n, sedov_module, nip=3000):
        ssv = np.linspace(1e-6 + 1.0 / gamma, 2.0 / (gamma + 1.0), num=nip)
        self.shock_radius = shock_front["radius"]
        self.upstream = upstream
        self.radius_table = np.array([self.shock_radius * sedov_module.vtoz(v, w, gamma, n) for v in ssv])
        self.density_table = np.array(
            [shock_front["density"] * sedov_module.vtod(v, w, gamma, n) / ((gamma + 1.0) / (gamma - 1.0)) for v in ssv]
        )
        self.pressure_table = np.array(
            [shock_front["pressure"] * sedov_module.vtop(v, w, gamma, n) / (2.0 / (gamma + 1.0)) for v in ssv]
        )
        self.velocity_table = np.array(
            [shock_front["velocity"] * sedov_module.vtoz(v, w, gamma, n) * v / (2.0 / (gamma + 1.0)) for v in ssv]
        )

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


def rel_l1_error(numeric: np.ndarray, analytic: np.ndarray) -> float:
    denom = float(np.mean(np.abs(analytic)))
    if denom < 1e-14:
        denom = 1e-14
    return float(np.mean(np.abs(numeric - analytic)) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Sedov 3D profile against exact Sedov-Taylor ODE profile.")
    parser.add_argument("--profile", required=True, help="Path to sedov_profile.txt")
    parser.add_argument("--rich-root", required=True, help="Repository root for analytic/sedov_taylor.py")
    parser.add_argument("--max-density-rel-l1", type=float, default=0.30)
    parser.add_argument("--max-pressure-rel-l1", type=float, default=0.30)
    parser.add_argument("--max-velocity-rel-l1", type=float, default=0.30)
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
    shock_front = find_shock_front(numeric)
    r = numeric["radius"]

    far_mask = r > 0.8 * float(np.max(r))
    if not np.any(far_mask):
        far_mask = r > np.median(r)

    upstream = {
        "density": float(np.median(numeric["density"][far_mask])),
        "pressure": float(np.median(numeric["pressure"][far_mask])),
        "velocity": float(np.median(numeric["velocity"][far_mask])),
    }

    st = SedovTaylorProfiles(upstream, shock_front, 5.0 / 3.0, 0.0, 3, sedov)
    compare_mask = r <= st.shock_radius
    if not np.any(compare_mask):
        print("No cells found within shock radius for Sedov comparison", file=sys.stderr)
        return 1

    analytic_density = np.array([st.calc("density", ri) for ri in r[compare_mask]])
    analytic_pressure = np.array([st.calc("pressure", ri) for ri in r[compare_mask]])
    analytic_velocity = np.array([st.calc("velocity", ri) for ri in r[compare_mask]])

    density_rel_l1 = rel_l1_error(numeric["density"][compare_mask], analytic_density)
    pressure_rel_l1 = rel_l1_error(numeric["pressure"][compare_mask], analytic_pressure)
    velocity_rel_l1 = rel_l1_error(numeric["velocity"][compare_mask], analytic_velocity)

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
