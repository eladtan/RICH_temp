#!/usr/bin/env python3
"""Fit KRTI-S growth histories and compare with the semi-analytic solver."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys


DEFAULT_REFERENCE_SOLVER = (
    Path(__file__).resolve().parent
    / "KRTI_S_X_reference_package"
    / "krti_s_reference.py"
)


def load_reference_solver(path: Path):
  if not path.is_file():
    raise RuntimeError(f"semi-analytic solver not found: {path}")
  spec = importlib.util.spec_from_file_location("krti_s_reference", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import semi-analytic solver: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def solve_reference(path: Path, case_key: str, nz: int, nmu: int, nphi: int):
  solver = load_reference_solver(path)
  if case_key not in solver.CASES:
    raise RuntimeError(f"case {case_key!r} is unavailable in {path}")
  case = solver.CASES[case_key]
  numerics = solver.Numerics(nz_per_layer=nz, nmu=nmu, nphi=nphi)
  background = solver.solve_background(case, numerics)
  growth_rate = solver.find_unstable_root(case, numerics, background)
  rt_time = 1.0 / math.sqrt(case.atwood * case.g * case.k)
  classical_rate = math.sqrt(
      case.atwood * case.g * case.k * math.tanh(case.k * case.H))
  effective_gravity_rate = case.s_classical_finite
  return {
      "case": case_key,
      "theta": case.theta,
      "growth_rate": growth_rate,
      "G": growth_rate * rt_time,
      "rt_time_s": rt_time,
      "eta0_cm": case.eta0,
      "linear_amplitude_limit_cm": 0.05 / case.k,
      "classical_growth_rate": classical_rate,
      "classical_G": classical_rate * rt_time,
      "effective_gravity_growth_rate": effective_gravity_rate,
      "effective_gravity_G": effective_gravity_rate * rt_time,
      "background_net_flux": background.F0,
      "bottom_intensity": background.Ib,
      "numerics": {"nz_per_layer": nz, "nmu": nmu, "nphi": nphi},
  }


def read_history(path: Path):
  metadata = {}
  rows = []
  with path.open() as stream:
    for line in stream:
      if line.startswith("#"):
        text = line[1:].strip()
        if "=" in text:
          key, value = text.split("=", 1)
          metadata[key.strip()] = value.strip()
      elif line.strip():
        rows.append(line)
  parsed = list(csv.DictReader(rows))
  if not parsed:
    raise RuntimeError(f"no data rows found in {path}")
  converted = []
  for row in parsed:
    converted.append({key: float(value) for key, value in row.items()
                      if key != "particle_count"})
    converted[-1]["particle_count"] = int(float(row["particle_count"]))
  return metadata, converted


def validate_frozen_history(metadata):
  expected = {
      "case": "X",
      "initialization": "full_reference_eigenmode",
  }
  for key, value in expected.items():
    if metadata.get(key) != value:
      raise RuntimeError(
          f"history is not the frozen KRTI-S-X problem: {key}={metadata.get(key)!r}, "
          f"expected {value!r}")
  expected_numbers = {
      "theta": 1.0,
      "alpha": 0.5,
      "k_eta0": 1.0e-3,
      "H_cm": 1.0,
      "k_cm_inv": 4.0,
      "rho_minus_g_cm3": 1.0,
      "rho_plus_g_cm3": 3.0,
      "gravity_cm_s2": 1.0e8,
      "gamma_gas": 5.0 / 3.0,
      "light_speed": 2.99792458e10,
      "mass_scattering_opacity": 2.0,
      "net_flux": 7.49481145e17,
      "incident_flux": math.pi * 1.6856350e18,
      "incident_intensity": 1.6856350e18,
      "reference_pressure": 1.0e12,
      "interface_delta": 0.0,
      "two_modes": 0.0,
      "static_scatterers": 1.0,
  }
  for key, expected_value in expected_numbers.items():
    try:
      value = float(metadata[key])
    except (KeyError, ValueError) as error:
      raise RuntimeError(f"history lacks valid frozen parameter {key}") from error
    scale = max(abs(expected_value), 1.0)
    if abs(value - expected_value) > 1.0e-12 * scale:
      raise RuntimeError(
          f"history is not the frozen KRTI-S-X problem: {key}={value:.17g}, "
          f"expected {expected_value:.17g}")


def linear_fit(x, y):
  if len(x) < 2:
    raise RuntimeError("not enough points for growth-rate fit")
  xbar = sum(x) / len(x)
  ybar = sum(y) / len(y)
  denom = sum((value - xbar) ** 2 for value in x)
  if denom <= 0.0:
    raise RuntimeError("degenerate fit window")
  slope = sum((x[index] - xbar) * (y[index] - ybar) for index in range(len(x))) / denom
  intercept = ybar - slope * xbar
  residual = math.sqrt(sum((y[index] - (intercept + slope * x[index])) ** 2
                           for index in range(len(x))) / len(x))
  return slope, intercept, residual


def resolve_rate_window(rows, t_min: float, t_max: float):
  times = [row["time_rt"] for row in rows]
  t_lo = min(times)
  t_hi = max(times)
  if t_hi <= t_lo:
    raise RuntimeError("degenerate simulation time range")
  if sum(t_min <= row["time_rt"] <= t_max for row in rows) >= 4:
    return t_min, t_max
  span = t_hi - t_lo
  return t_lo + 0.15 * span, t_hi - 0.05 * span


def fit_growth(rows, amplitude_key: str, t_min: float, t_max: float,
               maximum_amplitude: float):
  amplitudes = [abs(row[amplitude_key]) for row in rows]
  amplitude_floor = max(max(amplitudes) * 1.0e-8, 1.0e-30)
  selected = [row for row in rows
              if t_min <= row["time_rt"] <= t_max
              and amplitude_floor <= abs(row[amplitude_key]) <= maximum_amplitude]
  if len(selected) < 4:
    raise RuntimeError(f"not enough samples to fit {amplitude_key}")
  x = [row["time_rt"] for row in selected]
  y = [math.log(abs(row[amplitude_key])) for row in selected]
  slope, intercept, residual = linear_fit(x, y)
  return {"rate": slope, "intercept": intercept, "log_rms": residual,
          "points": len(selected), "amplitude_key": amplitude_key}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, help="KRTI history CSV from RICH")
  parser.add_argument("--json", type=Path, help="write comparison metrics JSON here")
  parser.add_argument("--plot", type=Path, help="optional comparison plot")
  parser.add_argument("--reference-only", action="store_true",
                      help="solve and print the semi-analytic reference, then exit")
  parser.add_argument("--case", choices=("D", "X", "T"), default="X",
                      help="case used with --reference-only (default: X)")
  parser.add_argument("--reference-solver", type=Path,
                      default=DEFAULT_REFERENCE_SOLVER,
                      help="path to krti_s_reference.py")
  parser.add_argument("--reference-nz", type=int, default=128,
                      help="semi-analytic radiation cells per layer")
  parser.add_argument("--reference-nmu", type=int, default=24,
                      help="semi-analytic Gauss-Legendre ordinates")
  parser.add_argument("--reference-nphi", type=int, default=16,
                      help="semi-analytic azimuthal ordinates")
  parser.add_argument("--rate-window", nargs=2, type=float, metavar=("T_MIN", "T_MAX"),
                      default=(0.5, 4.0))
  parser.add_argument(
      "--rate-relative-tolerance", type=float,
      help="optional diagnostic tolerance; does not constitute benchmark pass/fail")
  args = parser.parse_args()

  if args.reference_only:
    reference = solve_reference(
        args.reference_solver, args.case, args.reference_nz,
        args.reference_nmu, args.reference_nphi)
    print(f"case {args.case}: semi-analytic s={reference['growth_rate']:.9e} 1/s, "
          f"G={reference['G']:.9f}")
    return

  if args.input is None:
    raise SystemExit("--input is required unless --reference-only is set")

  metadata, rows = read_history(args.input)
  validate_frozen_history(metadata)
  case = metadata.get("case", "X")
  reference = solve_reference(
      args.reference_solver, case, args.reference_nz,
      args.reference_nmu, args.reference_nphi)
  t_min, t_max = resolve_rate_window(rows, *args.rate_window)
  mode_keys = [key for key in ("A_mode1", "A_mode2") if key in rows[0]]
  fits = {}
  for key in mode_keys:
    try:
      fits[key] = fit_growth(
          rows, key, t_min, t_max, reference["linear_amplitude_limit_cm"])
    except RuntimeError:
      continue
  if not fits:
    raise RuntimeError("no Fourier mode has enough samples for a growth-rate fit")
  primary_key = max(fits, key=lambda key: fits[key]["points"])
  fit1 = fits[primary_key]
  fit2 = fits.get("A_mode2") if primary_key != "A_mode2" else fits.get("A_mode1")

  metrics = {
      "benchmark": "KRTI-S",
      "case": case,
      "reference_G": reference["G"],
      "reference_growth_rate": reference["growth_rate"],
      "reference_rt_time_s": reference["rt_time_s"],
      "reference_numerics": reference["numerics"],
      "reference_bottom_intensity": reference["bottom_intensity"],
      "reference_background_net_flux": reference["background_net_flux"],
      "linear_amplitude_limit_cm": reference["linear_amplitude_limit_cm"],
      "classical_G": reference["classical_G"],
      "effective_gravity_G": reference["effective_gravity_G"],
      "simulation_G": fit1["rate"],
      "simulation_growth_rate_per_rt_time": fit1["rate"],
      "simulation_growth_rate_inv_s": fit1["rate"] / reference["rt_time_s"],
      "fit_window": [t_min, t_max],
      "primary_mode": primary_key,
      "fit_mode1": fits.get("A_mode1"),
      "fit_mode2": fits.get("A_mode2"),
      "isotropy_relative_difference": None,
      "pass": None,
  }
  if fit2 is not None:
    avg = 0.5 * (fit1["rate"] + fit2["rate"])
    metrics["isotropy_relative_difference"] = abs(fit1["rate"] - fit2["rate"]) / max(avg, 1.0e-30)
  reference_error = abs(metrics["simulation_G"] - reference["G"]) / reference["G"]
  metrics["relative_difference_from_reference"] = reference_error
  metrics["rate_relative_tolerance"] = args.rate_relative_tolerance
  metrics["within_requested_tolerance"] = (
      None if args.rate_relative_tolerance is None
      else reference_error <= args.rate_relative_tolerance
  )

  if args.json is not None:
    args.json.write_text(json.dumps(metrics, indent=2) + "\n")

  if args.plot is not None:
    try:
      import matplotlib.pyplot as plt
    except ImportError:
      print("plot skipped: matplotlib is not installed")
    else:
      figure, axis = plt.subplots(figsize=(8.0, 5.0))
      plot_key = fit1["amplitude_key"]
      axis.plot([row["time_rt"] for row in rows],
                [abs(row[plot_key]) for row in rows], "o", ms=2.2,
                label=f"simulation {plot_key}")
      t_values = [row["time_rt"] for row in rows]
      fit_amplitude = math.exp(fit1["intercept"])
      axis.plot(t_values,
                [fit_amplitude * math.exp(fit1["rate"] * value) for value in t_values],
                "k--", label="fit")
      axis.plot(t_values,
                [reference["eta0_cm"] * math.exp(reference["G"] * value)
                 for value in t_values],
                color="tab:green", label="semi-analytic eigenmode")
      axis.axhline(reference["linear_amplitude_limit_cm"], color="tab:red",
                   ls=":", label="linear-amplitude limit")
      axis.set_xlabel(r"$t / t_{\mathrm{RT}}$")
      axis.set_ylabel(r"$A_k$")
      axis.set_yscale("log")
      axis.grid(alpha=0.25)
      axis.legend(frameon=False)
      figure.tight_layout()
      figure.savefig(args.plot, dpi=160)

  print(f"case {case}: simulation G={metrics['simulation_G']:.6f}, "
        f"semi-analytic G={reference['G']:.6f}, "
        f"relative difference={reference_error:.3%}")
  print(f"semi-analytic s={reference['growth_rate']:.9e} 1/s "
        f"({args.reference_nz}x{args.reference_nmu}x{args.reference_nphi} reference discretization)")
  print("No benchmark pass/fail is assigned: one ideal-gas run is not a "
        "low-Mach convergence study against the incompressible reference.")
  if fit2 is not None:
    print(f"mode isotropy relative difference={metrics['isotropy_relative_difference']:.3%}")


if __name__ == "__main__":
  main()
