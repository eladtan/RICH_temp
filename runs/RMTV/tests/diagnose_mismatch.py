#!/usr/bin/env python3
"""Summarize matched diagnostic runs and independently check the reference PDE.

Run from any directory. This reads verification/diagnosis_* without changing
the solver or its results. SciPy is required for the continuum residual check.
"""
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]


def summarize(directory):
    config = json.loads((directory / 'config.json').read_text())
    status = json.loads((directory / 'status.json').read_text())
    data = np.genfromtxt(directory / 'final_rank0.csv', delimiter=',', names=True, skip_header=1)
    diag = np.atleast_1d(np.genfromtxt(directory / 'diagnostics.csv', delimiter=',', names=True))
    result = {key: config[key] for key in ('n', 'dt_fraction', 'photons', 'initial_photons',
                                         'max_photons', 'seed', 'opacity_cap', 'rf_end',
                                         'time_unit', 'temperature_unit', 'floor_K')}
    result.update(status)
    for field in ('rho', 'T'):
        result['L1_' + field] = float(np.sum(data['volume'] * abs(data[field] - data[field + '_ref'])) /
                                     np.sum(data['volume'] * abs(data[field + '_ref'])))
    result['max_energy_drift'] = float(np.max(abs(diag['energy_drift'])))
    return result


def continuum_residual():
    """Check dE/dt + div((E+p)v) = div(chi grad T) away from jumps.

    This is independent of the table generator's jump/unit checks. Central
    differences of the dimensional fields test the actual PDE normalization.
    Interpolation/finite differences limit the accuracy near the sharp front.
    """
    metadata = json.loads((ROOT / 'reference/metadata.json').read_text())
    table = np.loadtxt(ROOT / 'reference/shape.dat')
    split = metadata['inner_rows']
    time_unit, temperature_unit = 0.0256, 937.5
    cv = time_unit**-2 / (0.25 * temperature_unit)
    chi0 = 1 / (time_unit**3 * temperature_unit**7.5)
    age = 0.00048208462655560517
    result = {}
    for xi in (0.2, 0.6, 0.9, 1.1, 1.4, 1.7, 1.9):
        branch = table[:split] if xi < 1 else table[split:]
        spline = CubicSpline(branch[:, 0], branch[:, 1:], axis=0)

        def state(radius, time):
            shock = metadata['zeta'] * (time / time_unit)**(9 / 13)
            speed = (9 / 13) * shock / (time / time_unit)
            density, velocity, temperature = spline(radius / shock)
            return (shock**(-19 / 9) * density, speed / time_unit * velocity,
                    temperature_unit * speed**2 * temperature)

        def energy(radius, time):
            density, velocity, temperature = state(radius, time)
            return density * (cv * temperature + 0.5 * velocity**2)

        radius = xi * 0.225
        dr, dt = radius * 1e-4, age * 1e-4

        def advection(position):
            density, velocity, temperature = state(position, age)
            return position**2 * velocity * (energy(position, age) + 0.25 * density * cv * temperature)

        def conduction(position):
            density, _, temperature = state(position, age)
            gradient = (state(position + dr, age)[2] - state(position - dr, age)[2]) / (2 * dr)
            return position**2 * chi0 * density**-2 * temperature**6.5 * gradient

        lhs = ((energy(radius, age + dt) - energy(radius, age - dt)) / (2 * dt) +
               (advection(radius + dr) - advection(radius - dr)) / (2 * dr * radius**2))
        rhs = (conduction(radius + dr) - conduction(radius - dr)) / (2 * dr * radius**2)
        result[str(xi)] = float(lhs / rhs)
    return result


def shell_flux_bias(directory):
    """Compare the initial LTE DDMC stencil with integrated reference heat flux.

    Sum x-facing faces enclosing cell centers with r < 0.32 cm in the octant.
    Symmetry makes the other two directions equal. All selected quadrature
    points must be strictly between the shock and heat front. This measures
    the spatial operator, not an actual Monte Carlo energy tally.
    """
    config = json.loads((directory / 'config.json').read_text())
    data = np.genfromtxt(directory / 'initial_rank0.csv', delimiter=',', names=True, skip_header=1)
    assert config['octant'] and config['rf_start'] == 0.45
    n = config['n']
    dx = config['box'] / n
    indices = np.rint(np.array([data['x'], data['y'], data['z']]).T / dx - 0.5).astype(int)
    density, temperature = np.zeros((n, n, n)), np.zeros((n, n, n))
    density[tuple(indices.T)], temperature[tuple(indices.T)] = data['rho'], data['T']
    centers = (np.arange(n) + 0.5) * dx
    xyz = np.meshgrid(centers, centers, centers, indexing='ij')
    radius = np.sqrt(sum(position**2 for position in xyz))
    table = np.loadtxt(ROOT / 'reference/shape.dat')
    metadata = json.loads((ROOT / 'reference/metadata.json').read_text())
    branch = table[metadata['inner_rows']:]
    spline = CubicSpline(branch[:, 0], branch[:, 1:], axis=0)
    shock = 0.225
    speed = (9 / 13) * shock / (config['t_start'] / config['time_unit'])
    temperature_scale = config['temperature_unit'] * speed**2
    nodes, weights = np.polynomial.legendre.leggauss(8)
    numerical, exact = 0.0, 0.0
    for i, j, k in np.argwhere((radius[:-1] < 0.32) & (radius[1:] >= 0.32)):
        left, right = temperature[i, j, k], temperature[i + 1, j, k]
        face_temperature = (0.5 * (left**4 + right**4))**0.25
        # c/(3 Sigma(Tface)) on both sides, combined by two-sided resistance.
        # At these warm faces the opacity cap is inactive.
        sigma_coefficient = 4 * 7.5657e-15 * 2.99792458e10 / (3 * config['chi0'])
        for rho in (density[i, j, k], density[i + 1, j, k]):
            assert sigma_coefficient * rho**2 * face_temperature**-3.5 < config['opacity_cap']
        numerical += (dx * config['chi0'] / (2 * (density[i, j, k]**2 + density[i + 1, j, k]**2)) *
                      face_temperature**3.5 * (left**4 - right**4))
        x = (i + 1) * dx
        for u, wu in zip(nodes, weights):
            for v, wv in zip(nodes, weights):
                y, z = centers[j] + 0.5 * dx * u, centers[k] + 0.5 * dx * v
                r = np.sqrt(x*x + y*y + z*z)
                assert 1 < r / shock < 2
                rho, _, theta = spline(r / shock)
                gradient = temperature_scale * spline(r / shock, 1)[2] / shock
                conductivity = config['chi0'] * (shock**(-19 / 9) * rho)**-2 * (temperature_scale * theta)**6.5
                exact += wu * wv * dx**2 / 4 * (-conductivity * gradient * x / r)
    return {'stencil_luminosity': float(numerical), 'reference_luminosity': float(exact),
            'ratio': float(numerical / exact)}


if __name__ == '__main__':
    runs = {directory.name: summarize(directory)
            for directory in sorted((ROOT / 'verification').glob('diagnosis_*'))
            if (directory / 'status.json').exists()}
    flux = {name: shell_flux_bias(ROOT / 'verification' / name)
            for name in ('diagnosis_base_dt', 'diagnosis_fine_grid')
            if (ROOT / 'verification' / name / 'initial_rank0.csv').exists()}
    print(json.dumps({'runs': runs, 'continuum_energy_lhs_over_rhs': continuum_residual(),
                      'initial_shell_flux': flux}, indent=2))
