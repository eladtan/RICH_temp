#!/usr/bin/env python3

import argparse
import math
import os
import sys
from importlib.machinery import SourceFileLoader

import numpy as np


def goodness_of_fit(numeric: np.ndarray, analytic: np.ndarray) -> float:
    denom = (float(np.max(analytic)) - float(np.min(analytic))) ** 2
    if denom <= 0.0:
        denom = 1.0
    diff2 = np.square(numeric - analytic)
    return math.sqrt(float(np.sum(diff2)) / denom / len(numeric))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Sod profile against exact Riemann solution.")
    parser.add_argument("--profile", required=True, help="Path to sod_profile.txt")
    parser.add_argument("--rich-root", required=True, help="Repository root for loading analytic/enrs.py")
    parser.add_argument("--max-density-gof", type=float, default=2e-2)
    parser.add_argument("--max-pressure-gof", type=float, default=2e-2)
    parser.add_argument("--offset", type=float, default=0.5)
    parser.add_argument("--time", type=float, default=0.2)
    args = parser.parse_args()

    enrs_path = os.path.join(args.rich_root, "analytic", "enrs.py")
    enrs = SourceFileLoader("enrs", enrs_path).load_module()

    raw = np.loadtxt(args.profile)
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    x = raw[:, 0]
    density = raw[:, 1]
    pressure = raw[:, 2]
    t = args.time

    left = enrs.Primitive(1.0, 1.0, 0.0)
    right = enrs.Primitive(0.125, 0.1, 0.0)
    profile = enrs.RiemannProfile(left, right, 1.4)

    density_exact = np.array([profile.CalcPrim((xi - args.offset) / t).Density for xi in x])
    pressure_exact = np.array([profile.CalcPrim((xi - args.offset) / t).Pressure for xi in x])

    density_gof = goodness_of_fit(density, density_exact)
    pressure_gof = goodness_of_fit(pressure, pressure_exact)

    print(f"SOD_DENSITY_GOF={density_gof:.8e}")
    print(f"SOD_PRESSURE_GOF={pressure_gof:.8e}")
    print(f"SOD_MAX_DENSITY_GOF={args.max_density_gof:.8e}")
    print(f"SOD_MAX_PRESSURE_GOF={args.max_pressure_gof:.8e}")

    if density_gof > args.max_density_gof:
        print("Sod density profile exceeds tolerance", file=sys.stderr)
        return 1
    if pressure_gof > args.max_pressure_gof:
        print("Sod pressure profile exceeds tolerance", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
