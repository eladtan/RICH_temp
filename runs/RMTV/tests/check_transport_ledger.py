#!/usr/bin/env python3
"""Compare one-step regional transport with reference flux on identical faces."""
import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]


def reference_energy(config, radius, age, dt, order):
    """Time/face quadrature of -chi grad T, through a fixed voxel surface.

    By spherical symmetry each of the three positive coordinate directions
    has the same luminosity in the octant. Symmetry-plane flux is zero.
    """
    table = np.loadtxt(ROOT / 'reference/shape.dat')
    metadata = json.loads((ROOT / 'reference/metadata.json').read_text())
    outer = table[metadata['inner_rows']:]
    spline = CubicSpline(outer[:, 0], outer[:, 1:], axis=0)
    n = config['n']
    dx = config['box'] / n
    centers = (np.arange(n) + .5) * dx
    x, y, z = np.meshgrid(centers, centers, centers, indexing='ij')
    r = np.sqrt(x*x + y*y + z*z)
    faces = np.argwhere((r[:-1] < radius) & (r[1:] >= radius))
    nodes, weights = leggauss(order)
    u, v = np.meshgrid(nodes, nodes, indexing='ij')
    weight = np.outer(weights, weights).ravel() * dx*dx / 4
    xp = (faces[:, 0] + 1)[:, None] * dx + np.zeros((1, order**2))
    yp = centers[faces[:, 1]][:, None] + .5 * dx * u.ravel()
    zp = centers[faces[:, 2]][:, None] + .5 * dx * v.ravel()
    rp = np.sqrt(xp*xp + yp*yp + zp*zp)
    time_nodes, time_weights = leggauss(3)
    integrated = 0.0
    fluxes = []
    for node, time_weight in zip(time_nodes, time_weights):
        time = age + .5 * dt * (node + 1)
        shock = metadata['zeta'] * (time / config['time_unit'])**(9 / 13)
        xi = rp / shock
        assert np.all(xi > 1), 'Selected surface intersects shock; one-sided flux integration required'
        active = xi < outer[-1, 0]
        density, _, theta = np.moveaxis(spline(xi), -1, 0)
        scale = config['temperature_unit'] * ((9 / 13) * shock / (time / config['time_unit']))**2
        temperature = np.where(active, scale * theta, 0)
        gradient = np.where(active, scale * spline(xi, 1)[..., 2] / shock, 0)
        density = np.where(active, shock**(-19 / 9) * density, 1)
        flux = -config['chi0'] * density**-2 * temperature**6.5 * gradient * xp / rp
        luminosity = 3 * np.sum(flux * weight)
        integrated += .5 * dt * time_weight * luminosity
        fluxes.append(luminosity)
    return float(integrated), float((max(fluxes) - min(fluxes)) / np.mean(fluxes))


def analyze(directory):
    config = json.loads((directory / 'config.json').read_text())
    assert config['radiation_only'] and config['octant'] and config['ranks'] == 1
    ledger = np.atleast_1d(np.genfromtxt(directory / 'transport_ledger.csv', delimiter=',', names=True))
    assert len(ledger) == 4 and np.all(ledger['cycle'] == 1), 'Expected exactly one completed radiation step'
    initial = np.genfromtxt(directory / 'initial_rank0.csv', delimiter=',', names=True, skip_header=1)
    final = np.genfromtxt(directory / 'final_rank0.csv', delimiter=',', names=True, skip_header=1)
    for field in ('x', 'y', 'z', 'volume', 'rho', 'vx', 'vy', 'vz'):
        assert np.array_equal(initial[field], final[field]), 'Hydro changed ' + field
    diag = np.atleast_1d(np.genfromtxt(directory / 'diagnostics.csv', delimiter=',', names=True))
    assert np.all(diag['hydro_energy_change'] == 0)
    assert np.max(abs(diag['energy_drift'])) < 1e-10
    output = {'configuration': config, 'regions': [], 'max_energy_drift': float(np.max(abs(diag['energy_drift']))),
              'ddmc_steps': float(np.sum(diag['ddmc_steps']))}
    assert (output['ddmc_steps'] > 0) if config['ddmc'] else (output['ddmc_steps'] == 0)
    for row in ledger:
        radius = np.sqrt(initial['x']**2 + initial['y']**2 + initial['z']**2)
        selected = radius < row['radius']
        def snapshot_energy(data):
            kinetic = 0.5 * (data['vx']**2 + data['vy']**2 + data['vz']**2)
            return np.sum((data['volume'] * (data['rho'] * (data['e'] + kinetic) + data['Er']))[selected])
        assert np.isclose(snapshot_energy(initial), row['energy_before'], rtol=1e-12, atol=1e-9)
        assert np.isclose(snapshot_energy(final), row['energy_after'], rtol=1e-12, atol=1e-9)
        if row['radius'] > config['box'] * 2:
            output['whole_domain_energy_loss'] = float(row['net_outward_energy'])
            continue
        expected, time_variation = reference_energy(config, row['radius'], row['start_time'], row['dt'], 16)
        lower_order, _ = reference_energy(config, row['radius'], row['start_time'], row['dt'], 8)
        output['regions'].append({'radius': float(row['radius']), 'measured_outward_energy': float(row['net_outward_energy']),
                                  'reference_outward_energy': expected, 'ratio': float(row['net_outward_energy'] / expected),
                                  'reference_quadrature_change': float(abs(lower_order / expected - 1)),
                                  'reference_flux_time_variation': time_variation})
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    args = parser.parse_args()
    print(json.dumps({str(directory): analyze(directory) for directory in args.directories}, indent=2))
