#!/usr/bin/env python3
"""
Semi-analytic moving-slab benchmark calculator.

Implements two related benchmarks:

1) original_vacuum:
   McClarren & Gentile (2021), "Frequency-Dependent Material Motion
   Benchmarks for Radiative Transfer", Eqs. (13)-(14).

2) modified_absorbing_region:
   Gentile & McClarren (2026), "A Modified Frequency-Dependent Material Motion
   Benchmark for Thermal Radiative Transfer", Eqs. (1), (10), (12), and (18).

Default opacity data are the 124-group aluminum bounds/opacities tabulated in
the 2026 paper's Tables 3-7. Opacity values are kappa_g in cm^2/g. The slab
absorption coefficient used by the equations is sigma_slab = rho_slab * kappa_g
in cm^-1.

Units follow the papers:
- length: cm
- time: ns
- frequency/energy: keV
- temperature: keV
- intensity: GJ / (cm^2 s sr keV)
- energy density: GJ / (cm^3 keV) for group-average spectra, GJ/cm^3 when integrated over energy
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


K_BOLTZMANN_GJ_PER_KEV = 1.6021917e-25
PLANCK_GJ_NS = 6.6262e-34
C_LIGHT = 29.9792458  # cm/ns

# Low-energy plateau opacity (kappa in cm^2/g) for the 124-group aluminum table (2026 paper).
low_end_opacity = 1.0e4

_OPACITY_ROWS = [["0", "1.000e-03", "1.229e-03", f"{low_end_opacity:.3e}", "1.087e-09", "1.070e-09"], ["1", "1.229e-03", "1.510e-03", f"{low_end_opacity:.3e}", "1.615e-09", "1.616e-09"], ["2", "1.510e-03", "1.856e-03", f"{low_end_opacity:.3e}", "2.438e-09", "2.440e-09"], ["3", "1.856e-03", "2.281e-03", f"{low_end_opacity:.3e}", "3.682e-09", "3.684e-09"], ["4", "2.281e-03", "2.804e-03", f"{low_end_opacity:.3e}", "5.560e-09", "5.564e-09"], ["5", "2.804e-03", "3.446e-03", f"{low_end_opacity:.3e}", "8.395e-09", "8.401e-09"], ["6", "3.446e-03", "4.234e-03", f"{low_end_opacity:.3e}", "1.268e-08", "1.268e-08"], ["7", "4.234e-03", "5.204e-03", f"{low_end_opacity:.3e}", "1.914e-08", "1.915e-08"], ["8", "5.204e-03", "6.396e-03", f"{low_end_opacity:.3e}", "2.889e-08", "2.891e-08"], ["9", "6.396e-03", "7.860e-03", f"{low_end_opacity:.3e}", "4.360e-08", "4.363e-08"], ["10", "7.860e-03", "9.660e-03", f"{low_end_opacity:.3e}", "6.580e-08", "6.584e-08"], ["11", "9.660e-03", "1.187e-02", f"{low_end_opacity:.3e}", "9.926e-08", "9.931e-08"], ["12", "1.187e-02", "1.459e-02", f"{low_end_opacity:.3e}", "1.497e-07", "1.498e-07"], ["13", "1.459e-02", "1.793e-02", f"{low_end_opacity:.3e}", "2.258e-07", "2.259e-07"], ["14", "1.793e-02", "2.204e-02", f"{low_end_opacity:.3e}", "3.404e-07", "3.406e-07"], ["15", "2.204e-02", "2.708e-02", "8.933e+03", "5.129e-07", "5.132e-07"], ["16", "2.708e-02", "3.328e-02", "8.569e+03", "7.726e-07", "7.730e-07"], ["17", "3.328e-02", "4.090e-02", "7.335e+03", "1.163e-06", "1.163e-06"], ["18", "4.090e-02", "5.027e-02", "5.656e+03", "1.749e-06", "1.750e-06"], ["19", "5.027e-02", "6.178e-02", "4.031e+03", "2.628e-06", "2.629e-06"], ["20", "6.178e-02", "7.593e-02", "2.710e+03", "3.944e-06", "3.946e-06"], ["21", "7.593e-02", "9.331e-02", "1.770e+03", "5.910e-06", "5.913e-06"], ["22", "9.331e-02", "1.147e-01", "1.184e+03", "8.841e-06", "8.844e-06"], ["23", "1.147e-01", "1.409e-01", "7.924e+02", "1.319e-05", "1.320e-05"], ["24", "1.409e-01", "1.732e-01", "5.061e+02", "1.963e-05", "1.964e-05"], ["25", "1.732e-01", "2.129e-01", "3.230e+02", "2.911e-05", "2.911e-05"], ["26", "2.129e-01", "2.616e-01", "2.062e+02", "4.294e-05", "4.297e-05"], ["27", "2.616e-01", "3.215e-01", "2.100e+02", "6.311e-05", "6.308e-05"], ["28", "3.215e-01", "3.951e-01", "1.229e+02", "9.196e-05", "9.184e-05"], ["29", "3.951e-01", "4.856e-01", "7.579e+01", "1.315e-04", "1.313e-04"], ["30", "4.856e-01", "5.968e-01", "4.905e+01", "1.820e-04", "1.813e-04"], ["31", "5.968e-01", "7.334e-01", "3.110e+01", "2.376e-04", "2.360e-04"], ["32", "7.334e-01", "9.014e-01", "1.947e+01", "2.872e-04", "2.840e-04"], ["33", "9.014e-01", "1.000e+00", "1.196e+01", "2.948e-04", "2.885e-04"], ["34", "1.000e+00", "1.014e+00", "1.187e+01", "3.027e-04", "3.120e-04"], ["35", "1.014e+00", "1.028e+00", "1.149e+01", "3.177e-04", "3.130e-04"], ["36", "1.028e+00", "1.042e+00", "1.112e+01", "3.200e-04", "3.139e-04"], ["37", "1.042e+00", "1.057e+00", "1.076e+01", "3.209e-04", "3.147e-04"], ["38", "1.057e+00", "1.072e+00", "1.041e+01", "3.217e-04", "3.153e-04"], ["39", "1.072e+00", "1.087e+00", "1.007e+01", "3.223e-04", "3.157e-04"], ["40", "1.087e+00", "1.102e+00", "9.740e+00", "3.227e-04", "3.160e-04"], ["41", "1.102e+00", "1.117e+00", "9.416e+00", "3.230e-04", "3.161e-04"], ["42", "1.117e+00", "1.133e+00", "9.098e+00", "3.231e-04", "3.159e-04"], ["43", "1.133e+00", "1.149e+00", "8.785e+00", "3.229e-04", "3.155e-04"], ["44", "1.149e+00", "1.165e+00", "8.477e+00", "3.225e-04", "3.149e-04"], ["45", "1.165e+00", "1.181e+00", "8.180e+00", "3.218e-04", "3.141e-04"], ["46", "1.181e+00", "1.198e+00", "7.900e+00", "3.211e-04", "3.134e-04"], ["47", "1.198e+00", "1.214e+00", "7.635e+00", "3.204e-04", "3.126e-04"], ["48", "1.214e+00", "1.231e+00", "7.381e+00", "3.196e-04", "3.118e-04"], ["49", "1.231e+00", "1.248e+00", "7.138e+00", "3.188e-04", "3.109e-04"], ["50", "1.248e+00", "1.266e+00", "6.902e+00", "3.179e-04", "3.099e-04"], ["51", "1.266e+00", "1.283e+00", "6.674e+00", "3.169e-04", "3.087e-04"], ["52", "1.283e+00", "1.301e+00", "6.452e+00", "3.157e-04", "3.074e-04"], ["53", "1.301e+00", "1.319e+00", "6.237e+00", "3.143e-04", "3.060e-04"], ["54", "1.319e+00", "1.338e+00", "6.029e+00", "3.129e-04", "3.044e-04"], ["55", "1.338e+00", "1.357e+00", "5.827e+00", "3.113e-04", "3.027e-04"], ["56", "1.357e+00", "1.375e+00", "5.631e+00", "3.096e-04", "3.009e-04"], ["57", "1.375e+00", "1.395e+00", "5.438e+00", "3.076e-04", "2.988e-04"], ["58", "1.395e+00", "1.414e+00", "5.250e+00", "3.055e-04", "2.966e-04"], ["59", "1.414e+00", "1.434e+00", "5.066e+00", "3.032e-04", "2.941e-04"], ["60", "1.434e+00", "1.454e+00", "4.886e+00", "3.007e-04", "2.914e-04"], ["61", "1.454e+00", "1.474e+00", "4.709e+00", "2.979e-04", "2.885e-04"], ["62", "1.474e+00", "1.495e+00", "4.542e+00", "2.950e-04", "2.857e-04"], ["63", "1.495e+00", "1.516e+00", "4.387e+00", "2.922e-04", "2.829e-04"], ["64", "1.516e+00", "1.537e+00", "4.243e+00", "2.895e-04", "2.805e-04"], ["65", "1.537e+00", "1.558e+00", "4.117e+00", "2.872e-04", "2.785e-04"], ["66", "1.558e+00", "1.580e+00", "4.310e+00", "2.906e-04", "2.919e-04"], ["67", "1.580e+00", "1.602e+00", "1.572e+01", "3.867e-04", "6.227e-04"], ["68", "1.602e+00", "1.625e+00", "4.834e+00", "5.330e-04", "3.251e-04"], ["69", "1.625e+00", "1.647e+00", "3.726e+00", "3.369e-04", "2.740e-04"], ["70", "1.647e+00", "1.670e+00", "3.758e+00", "2.861e-04", "2.794e-04"], ["71", "1.670e+00", "1.694e+00", "4.706e+00", "3.053e-04", "3.321e-04"], ["72", "1.694e+00", "1.717e+00", "3.394e+01", "4.504e-04", "8.498e-04"], ["73", "1.717e+00", "1.741e+00", "9.034e+02", "8.271e-04", "9.576e-04"], ["74", "1.741e+00", "1.765e+00", "1.615e+01", "9.285e-04", "6.903e-04"], ["75", "1.765e+00", "1.790e+00", "4.098e+00", "6.401e-04", "3.168e-04"], ["76", "1.790e+00", "1.815e+00", "3.420e+00", "3.387e-04", "2.809e-04"], ["77", "1.815e+00", "1.840e+00", "3.389e+00", "2.910e-04", "2.823e-04"], ["78", "1.840e+00", "1.866e+00", "3.986e+00", "3.045e-04", "3.215e-04"], ["79", "1.866e+00", "1.892e+00", "4.350e+00", "3.366e-04", "3.460e-04"], ["80", "1.892e+00", "1.919e+00", "3.933e+00", "3.462e-04", "3.255e-04"], ["81", "1.919e+00", "1.945e+00", "4.258e+00", "3.432e-04", "3.481e-04"], ["82", "1.945e+00", "1.972e+00", "4.861e+00", "3.688e-04", "3.857e-04"], ["83", "1.972e+00", "1.995e+00", "6.836e+00", "4.172e-04", "4.863e-04"], ["84", "1.995e+00", "2.089e+00", "4.674e+01", "8.983e-04", "1.035e-03"], ["85", "2.089e+00", "2.188e+00", "2.108e+01", "9.480e-04", "8.840e-04"], ["86", "2.188e+00", "2.291e+00", "2.281e+01", "9.343e-04", "9.304e-04"], ["87", "2.291e+00", "2.399e+00", "1.963e+01", "9.274e-04", "9.025e-04"], ["88", "2.399e+00", "2.512e+00", "1.749e+01", "9.015e-04", "8.785e-04"], ["89", "2.512e+00", "2.630e+00", "1.590e+01", "8.777e-04", "8.556e-04"], ["90", "2.630e+00", "2.754e+00", "1.442e+01", "8.496e-04", "8.265e-04"], ["91", "2.754e+00", "2.884e+00", "1.294e+01", "8.130e-04", "7.884e-04"], ["92", "2.884e+00", "3.020e+00", "1.144e+01", "7.667e-04", "7.405e-04"], ["93", "3.020e+00", "3.162e+00", "1.014e+01", "7.158e-04", "6.901e-04"], ["94", "3.162e+00", "3.311e+00", "9.047e+00", "6.642e-04", "6.395e-04"], ["95", "3.311e+00", "3.467e+00", "8.057e+00", "6.106e-04", "5.864e-04"], ["96", "3.467e+00", "3.631e+00", "7.118e+00", "5.540e-04", "5.302e-04"], ["97", "3.631e+00", "3.802e+00", "6.219e+00", "4.944e-04", "4.711e-04"], ["98", "3.802e+00", "3.981e+00", "5.474e+00", "4.372e-04", "4.160e-04"], ["99", "3.981e+00", "4.169e+00", "4.861e+00", "3.848e-04", "3.657e-04"], ["100", "4.169e+00", "4.365e+00", "4.311e+00", "3.353e-04", "3.177e-04"], ["101", "4.365e+00", "4.571e+00", "3.792e+00", "2.875e-04", "2.713e-04"], ["102", "4.571e+00", "4.786e+00", "3.296e+00", "2.417e-04", "2.270e-04"], ["103", "4.786e+00", "5.012e+00", "2.888e+00", "2.012e-04", "1.886e-04"], ["104", "5.012e+00", "5.248e+00", "2.555e+00", "1.666e-04", "1.560e-04"], ["105", "5.248e+00", "5.495e+00", "2.258e+00", "1.364e-04", "1.273e-04"], ["106", "5.495e+00", "5.754e+00", "1.978e+00", "1.096e-04", "1.017e-04"], ["107", "5.754e+00", "6.026e+00", "1.713e+00", "8.601e-05", "7.942e-05"], ["108", "6.026e+00", "6.310e+00", "1.496e+00", "6.678e-05", "6.155e-05"], ["109", "6.310e+00", "6.607e+00", "1.320e+00", "5.152e-05", "4.742e-05"], ["110", "6.607e+00", "6.918e+00", "1.163e+00", "3.920e-05", "3.595e-05"], ["111", "6.918e+00", "7.244e+00", "1.016e+00", "2.919e-05", "2.663e-05"], ["112", "7.244e+00", "7.586e+00", "8.770e-01", "2.116e-05", "1.919e-05"], ["113", "7.586e+00", "7.943e+00", "7.641e-01", "1.514e-05", "1.370e-05"], ["114", "7.943e+00", "8.318e+00", "6.729e-01", "1.075e-05", "9.713e-06"], ["115", "8.318e+00", "8.710e+00", "5.919e-01", "7.505e-06", "6.752e-06"], ["116", "8.710e+00", "9.120e+00", "5.160e-01", "5.106e-06", "4.567e-06"], ["117", "9.120e+00", "9.550e+00", "4.442e-01", "3.368e-06", "2.993e-06"], ["118", "9.550e+00", "1.070e+01", "3.862e-01", "1.706e-06", "1.552e-06"], ["119", "1.070e+01", "1.315e+01", "2.385e-01", "3.472e-07", "2.928e-07"], ["120", "1.315e+01", "1.616e+01", "1.309e-01", "3.145e-08", "2.166e-08"], ["121", "1.616e+01", "1.986e+01", "7.143e-02", "1.608e-09", "8.913e-10"], ["122", "1.986e+01", "2.441e+01", "3.867e-02", "4.710e-11", "1.802e-11"], ["123", "2.441e+01", "3.000e+01", "2.076e-02", "7.358e-13", "1.551e-13"]]


def load_default_opacity_table() -> Dict[str, np.ndarray]:
    group = np.array([int(r[0]) for r in _OPACITY_ROWS], dtype=int)
    nu_min = np.array([float(r[1]) for r in _OPACITY_ROWS], dtype=float)
    nu_max = np.array([float(r[2]) for r in _OPACITY_ROWS], dtype=float)
    kappa = np.array([float(r[3]) for r in _OPACITY_ROWS], dtype=float)
    erad_modified = np.array([float(r[4]) for r in _OPACITY_ROWS], dtype=float)
    erad_v0 = np.array([float(r[5]) for r in _OPACITY_ROWS], dtype=float)
    return {
        "group": group,
        "nu_min": nu_min,
        "nu_max": nu_max,
        "kappa": kappa,
        "erad_modified_reference": erad_modified,
        "erad_stationary_reference": erad_v0,
    }


@dataclass
class BenchmarkParams:
    benchmark_type: str = "original_vacuum"
    rho_slab: float = 0.1
    L: float = 0.4
    T: float = 1.0
    v_slab: float = 0.5994
    z_obs: float = 12.0
    t_obs: float = 10.0
    z_v0: float = 11.0
    sigma_hat_F: float = 1.0e-2
    mu_order: int = 256
    nu_order: int = 32


def gamma_rel(beta: float) -> float:
    if abs(beta) >= 1.0:
        raise ValueError("abs(beta) must be < 1")
    return 1.0 / math.sqrt(1.0 - beta * beta)


def gamma_D(mu: float, beta: float) -> float:
    return gamma_rel(beta) * (1.0 - mu * beta)


def planck_energy_form(nu_keV: float, T_keV: float) -> float:
    if nu_keV <= 0.0 or T_keV <= 0.0:
        return 0.0
    x = nu_keV / T_keV
    bose = math.exp(-x) if x > 700.0 else 1.0 / math.expm1(x)
    prefac = 2.0 * (K_BOLTZMANN_GJ_PER_KEV ** 4) / (C_LIGHT ** 2 * PLANCK_GJ_NS ** 3)
    return prefac * (nu_keV ** 3) * bose


class PiecewiseOpacity:
    def __init__(self, nu_min: Sequence[float], nu_max: Sequence[float], kappa: Sequence[float], rho_slab: float):
        self.nu_min = np.asarray(nu_min, dtype=float)
        self.nu_max = np.asarray(nu_max, dtype=float)
        self.kappa = np.asarray(kappa, dtype=float)
        self.sigma = rho_slab * self.kappa
        self.boundaries = np.concatenate([self.nu_min[:1], self.nu_max])
        if not np.allclose(self.nu_max[:-1], self.nu_min[1:], rtol=0.0, atol=5e-6):
            raise ValueError("Opacity group bounds are not contiguous within tolerance.")

    def sigma_at(self, nu_keV: float) -> float:
        if nu_keV < self.nu_min[0] or nu_keV > self.nu_max[-1]:
            return 0.0
        if nu_keV == self.nu_max[-1]:
            return float(self.sigma[-1])
        idx = np.searchsorted(self.nu_max, nu_keV, side="right")
        return float(self.sigma[idx])

    def breakpoints_in_lab_interval(self, nu_lo_lab: float, nu_hi_lab: float, gD: float) -> List[float]:
        pts = [nu_lo_lab, nu_hi_lab]
        if gD <= 0.0:
            return pts
        fluid_bounds = self.boundaries[1:-1]
        mapped = fluid_bounds / gD
        for x in mapped:
            if nu_lo_lab < x < nu_hi_lab:
                pts.append(float(x))
        pts = sorted(set(round(p, 15) for p in pts))
        return pts


def positive_part(x: float) -> float:
    return x if x > 0.0 else 0.0


def tb_tf_original(mu: float, params: BenchmarkParams) -> Tuple[float, float]:
    denom = mu * C_LIGHT - params.v_slab
    if denom <= 0.0:
        return 0.0, 0.0
    tb = positive_part((mu * C_LIGHT * params.t_obs - params.z_obs) / denom)
    tf = positive_part((params.L + mu * C_LIGHT * params.t_obs - params.z_obs) / denom)
    return tb, tf


def ray_length_in_slab_original(mu: float, params: BenchmarkParams) -> float:
    tb, tf = tb_tf_original(mu, params)
    return max(0.0, C_LIGHT * (tf - tb))


def intensity_original(mu: float, nu_lab: float, params: BenchmarkParams, opacity: PiecewiseOpacity) -> float:
    s = ray_length_in_slab_original(mu, params)
    if s <= 0.0:
        return 0.0
    beta = params.v_slab / C_LIGHT
    gD = gamma_D(mu, beta)
    nu_fluid = gD * nu_lab
    sigma = opacity.sigma_at(nu_fluid)
    if sigma <= 0.0:
        return 0.0
    source = planck_energy_form(nu_fluid, params.T) / (gD ** 3)
    return source * (1.0 - math.exp(-gD * sigma * s))


def tf_modified(mu: float, params: BenchmarkParams) -> float:
    denom = mu * C_LIGHT - params.v_slab
    if denom <= 0.0:
        return 0.0
    return positive_part((params.L + mu * C_LIGHT * params.t_obs - params.z_obs) / denom)


def intensity_modified(mu: float, nu_lab: float, params: BenchmarkParams, opacity: PiecewiseOpacity) -> float:
    beta = params.v_slab / C_LIGHT
    if mu <= beta:
        return 0.0
    gD = gamma_D(mu, beta)
    nu_fluid = gD * nu_lab
    sigma_slab = opacity.sigma_at(nu_fluid)
    if sigma_slab <= 0.0:
        return 0.0

    denom = mu * C_LIGHT - params.v_slab
    if denom <= 0.0:
        return 0.0

    source = planck_energy_form(nu_fluid, params.T) / (gD ** 3)
    I_slab = source * (1.0 - math.exp(-gD * sigma_slab * params.L / (mu - beta)))
    if I_slab == 0.0:
        return 0.0

    tf = tf_modified(mu, params)
    front_at_tf = params.L + params.v_slab * tf
    if params.z_v0 <= front_at_tf:
        return 0.0

    s_max = (params.z_v0 - front_at_tf) / mu
    if s_max <= 0.0:
        return 0.0
    atten_transition = math.exp(-s_max * params.sigma_hat_F * (1.0 - 0.5 * beta * mu))
    atten_stationary = math.exp(-params.sigma_hat_F * (params.z_obs - params.z_v0) / mu)
    return I_slab * atten_transition * atten_stationary


def intensity(mu: float, nu_lab: float, params: BenchmarkParams, opacity: PiecewiseOpacity) -> float:
    if params.benchmark_type == "original_vacuum":
        return intensity_original(mu, nu_lab, params, opacity)
    if params.benchmark_type == "modified_absorbing_region":
        return intensity_modified(mu, nu_lab, params, opacity)
    raise ValueError(f"Unknown benchmark_type={params.benchmark_type!r}")


def gauss_legendre_integrate(func, a: float, b: float, order: int) -> float:
    x, w = np.polynomial.legendre.leggauss(order)
    xm = 0.5 * (b + a)
    xr = 0.5 * (b - a)
    vals = np.array([func(xm + xr * xi) for xi in x], dtype=float)
    return xr * float(np.dot(w, vals))


def frequency_integral_for_fixed_mu(mu: float, nu_lo: float, nu_hi: float,
                                    params: BenchmarkParams, opacity: PiecewiseOpacity) -> float:
    beta = params.v_slab / C_LIGHT
    gD = gamma_D(mu, beta)

    def local_int(nu):
        return intensity(mu, nu, params, opacity)

    pts = opacity.breakpoints_in_lab_interval(nu_lo, nu_hi, gD)
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        if b <= a:
            continue
        total += gauss_legendre_integrate(local_int, a, b, params.nu_order)
    return total


def group_average_energy_density(nu_lo: float, nu_hi: float,
                                 params: BenchmarkParams, opacity: PiecewiseOpacity) -> float:
    beta = params.v_slab / C_LIGHT
    mu_lo = beta if params.benchmark_type == "modified_absorbing_region" else 0.0
    if mu_lo >= 1.0:
        return 0.0

    def mu_integrand(mu):
        return frequency_integral_for_fixed_mu(mu, nu_lo, nu_hi, params, opacity)

    mu_int = gauss_legendre_integrate(mu_integrand, mu_lo, 1.0, params.mu_order)
    return (2.0 * math.pi / C_LIGHT) * mu_int / (nu_hi - nu_lo)


def gray_intensity_vs_mu(mu: float, params: BenchmarkParams, opacity: PiecewiseOpacity) -> float:
    if params.benchmark_type == "modified_absorbing_region" and mu <= params.v_slab / C_LIGHT:
        return 0.0

    def nu_func(nu):
        return intensity(mu, nu, params, opacity)

    total = 0.0
    for a, b in zip(opacity.nu_min, opacity.nu_max):
        total += gauss_legendre_integrate(nu_func, float(a), float(b), params.nu_order)
    return total


def compute_groups(params: BenchmarkParams, opacity_data: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
    opacity = PiecewiseOpacity(opacity_data["nu_min"], opacity_data["nu_max"], opacity_data["kappa"], params.rho_slab)
    rows = []
    for g, nu_lo, nu_hi, kappa in zip(opacity_data["group"], opacity_data["nu_min"], opacity_data["nu_max"], opacity_data["kappa"]):
        erad_g = group_average_energy_density(float(nu_lo), float(nu_hi), params, opacity)
        rows.append({
            "group": int(g),
            "nu_min_keV": float(nu_lo),
            "nu_max_keV": float(nu_hi),
            "kappa_cm2_per_g": float(kappa),
            "sigma_cm^-1": float(params.rho_slab * kappa),
            "erad_GJ_per_cm3_per_keV": erad_g,
            "group_integrated_energy_density_GJ_per_cm3": erad_g * float(nu_hi - nu_lo),
        })
    return rows


def compute_total_erad(group_rows: Sequence[Dict[str, float]]) -> float:
    return float(sum(r["group_integrated_energy_density_GJ_per_cm3"] for r in group_rows))


def write_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_spectrum_plot_loglog(path_png: Path, path_pdf: Path, rows: Sequence[Dict[str, float]],
                               benchmark_type: str) -> None:
    import matplotlib.pyplot as plt

    nu_min = np.array([r["nu_min_keV"] for r in rows], dtype=float)
    nu_max = np.array([r["nu_max_keV"] for r in rows], dtype=float)
    nu_center = np.sqrt(nu_min * nu_max)
    erad = np.array([r["erad_GJ_per_cm3_per_keV"] for r in rows], dtype=float)

    positive = erad > 0.0
    if not np.any(positive):
        raise RuntimeError("Cannot create log-log spectrum plot because all group values are <= 0.")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.loglog(nu_center[positive], erad[positive], marker="o", linestyle="-", linewidth=1.0, markersize=3.0)
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel(r"Group-average $E_r$ (GJ/cm$^3$/keV)")
    ax.set_title(f"Final spectrum ({benchmark_type})")
    ax.set_xlim(1.0e-2, 20.0)
    ax.set_ylim(1.0e-7, 2.0e-3)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    fig.savefig(path_png, dpi=200)
    fig.savefig(path_pdf)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute moving-slab semi-analytic benchmark spectra.")
    p.add_argument("--benchmark-type", choices=["original_vacuum", "modified_absorbing_region"],
                   default="original_vacuum")
    p.add_argument("--rho-slab", type=float, default=0.1)
    p.add_argument("--L", type=float, default=0.4)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--v-slab", type=float, default=0.5994)
    p.add_argument("--z-obs", type=float, default=12.0)
    p.add_argument("--t-obs", type=float, default=10.0)
    p.add_argument("--z-v0", type=float, default=11.0,
                   help="Used only by modified_absorbing_region.")
    p.add_argument("--sigma-hat-F", type=float, default=1.0e-2,
                   help="Gray front-region absorption coefficient in cm^-1; used only by modified_absorbing_region.")
    p.add_argument("--mu-order", type=int, default=256)
    p.add_argument("--nu-order", type=int, default=32)
    p.add_argument("--output-csv", type=Path, default=Path("benchmark_spectrum.csv"))
    p.add_argument("--output-json", type=Path, default=Path("benchmark_summary.json"))
    p.add_argument("--output-plot-png", type=Path, default=Path("benchmark_spectrum_loglog.png"))
    p.add_argument("--output-plot-pdf", type=Path, default=Path("benchmark_spectrum_loglog.pdf"))
    p.add_argument("--emit-gray-mu-csv", type=Path, default=None,
                   help="Optional CSV for 100-bin gray intensity vs mu, similar to the 2026 paper.")
    return p


def print_run_configuration(params: BenchmarkParams, args: argparse.Namespace) -> None:
    """Echo effective CLI parameters and output paths to stdout."""
    print("Moving-slab benchmark — effective parameters:")
    print(f"  benchmark_type      = {params.benchmark_type!r}")
    print(f"  rho_slab (g/cm^3)    = {params.rho_slab}")
    print(f"  L (cm)               = {params.L}")
    print(f"  T (keV)              = {params.T}")
    print(f"  v_slab (cm/ns)       = {params.v_slab}")
    print(f"  z_obs (cm)           = {params.z_obs}")
    print(f"  t_obs (ns)           = {params.t_obs}")
    print(f"  z_v0 (cm)            = {params.z_v0}")
    print(f"  sigma_hat_F (cm^-1)  = {params.sigma_hat_F}")
    print(f"  mu_order             = {params.mu_order}")
    print(f"  nu_order             = {params.nu_order}")
    print(f"  low_end_opacity      = {low_end_opacity}  (κ plateau, embedded 124-group table)")
    print("Outputs:")
    print(f"  --output-csv         = {args.output_csv}")
    print(f"  --output-json        = {args.output_json}")
    print(f"  --output-plot-png    = {args.output_plot_png}")
    print(f"  --output-plot-pdf    = {args.output_plot_pdf}")
    print(f"  --emit-gray-mu-csv   = {args.emit_gray_mu_csv}")
    print(flush=True)


def main() -> None:
    args = build_parser().parse_args()
    params = BenchmarkParams(
        benchmark_type=args.benchmark_type,
        rho_slab=args.rho_slab,
        L=args.L,
        T=args.T,
        v_slab=args.v_slab,
        z_obs=args.z_obs,
        t_obs=args.t_obs,
        z_v0=args.z_v0,
        sigma_hat_F=args.sigma_hat_F,
        mu_order=args.mu_order,
        nu_order=args.nu_order,
    )

    print_run_configuration(params, args)

    opacity_data = load_default_opacity_table()
    group_rows = compute_groups(params, opacity_data)
    total_erad = compute_total_erad(group_rows)

    write_csv(args.output_csv, group_rows)
    write_spectrum_plot_loglog(args.output_plot_png, args.output_plot_pdf, group_rows, params.benchmark_type)

    summary = {
        "benchmark_type": params.benchmark_type,
        "parameters": {
            "rho_slab": params.rho_slab,
            "L": params.L,
            "T": params.T,
            "v_slab": params.v_slab,
            "z_obs": params.z_obs,
            "t_obs": params.t_obs,
            "z_v0": params.z_v0,
            "sigma_hat_F": params.sigma_hat_F,
            "mu_order": params.mu_order,
            "nu_order": params.nu_order,
        },
        "total_erad_GJ_per_cm3": total_erad,
        "rounded_table_total_modified_GJ_per_cm3": float(np.sum(opacity_data["erad_modified_reference"] * (opacity_data["nu_max"] - opacity_data["nu_min"]))),
        "rounded_table_total_stationary_GJ_per_cm3": float(np.sum(opacity_data["erad_stationary_reference"] * (opacity_data["nu_max"] - opacity_data["nu_min"]))),
        "notes": [
            "The embedded opacity table is the 124-group aluminum table from the 2026 modified-benchmark paper.",
            "For benchmark_type=original_vacuum, the script combines the 2021 vacuum benchmark geometry with the 2026 124-group opacity data.",
            "For benchmark_type=modified_absorbing_region, the script evaluates the second-order-in-v/c closed form from the 2026 paper for the transition region.",
            "The rounded totals reconstructed from the appendix tables will not exactly match the paper's quoted total because the printed per-group values are rounded."
        ],
    }
    args.output_json.write_text(json.dumps(summary, indent=2))

    if args.emit_gray_mu_csv is not None:
        opacity = PiecewiseOpacity(opacity_data["nu_min"], opacity_data["nu_max"], opacity_data["kappa"], params.rho_slab)
        mu_edges = np.linspace(0.0, 1.0, 101)
        gray_rows = []
        for i in range(100):
            a = float(mu_edges[i])
            b = float(mu_edges[i + 1])
            mu_mid = 0.5 * (a + b)
            gray_bin = gauss_legendre_integrate(
                lambda mu: gray_intensity_vs_mu(mu, params, opacity),
                a, b, params.mu_order
            ) / (b - a)
            gray_rows.append({
                "bin": i,
                "mu_center": mu_mid,
                "gray_intensity_GJ_per_cm2_per_s_per_sr": gray_bin,
            })
        write_csv(args.emit_gray_mu_csv, gray_rows)
        print(f"Wrote {args.emit_gray_mu_csv}")

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_plot_png}")
    print(f"Wrote {args.output_plot_pdf}")
    print(f"Total erad = {total_erad:.10e} GJ/cm^3")


if __name__ == "__main__":
    main()
