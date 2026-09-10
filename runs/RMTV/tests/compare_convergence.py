#!/usr/bin/env python3
"""Compare completed RMTV spatial-refinement runs and plot their radial profiles."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_run(directory):
    config = json.loads((directory / 'config.json').read_text())
    status = json.loads((directory / 'status.json').read_text())
    analysis = json.loads((directory / 'analysis_final.json').read_text())
    assert status['reached_end'] and analysis['reached_end'], directory
    files = sorted(directory.glob('final_rank*.csv'))
    assert len(files) == config['ranks']
    data = np.concatenate([np.atleast_1d(np.genfromtxt(path, delimiter=',', names=True, skip_header=1))
                           for path in files])
    assert len(data) == config['n']**3
    return config, analysis, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = sorted([(path, *load_run(path)) for path in args.directories], key=lambda item: item[1]['n'])
    baseline = runs[0][1]
    baseline_diagnostics = np.atleast_1d(np.genfromtxt(runs[0][0] / 'diagnostics.csv', delimiter=',', names=True))
    matched = ('quadrature', 'box', 'rf_start', 'rf_end', 'time_unit', 'temperature_unit', 'floor_K',
               'opacity_cap', 'cfl', 'dt_fraction', 'seed', 'photons', 'initial_photons', 'max_photons',
               'ddmc', 'hydro_only', 'radiation_only', 'octant', 'radiation_momentum', 'adaptive_radiation_dt')
    for directory, config, analysis, _ in runs:
        for field in matched:
            assert config[field] == baseline[field], 'Unmatched setting: ' + field
        assert np.isclose(analysis['physical_age_s'], runs[0][2]['physical_age_s'], rtol=1e-13, atol=0)
        diagnostics = np.atleast_1d(np.genfromtxt(directory / 'diagnostics.csv', delimiter=',', names=True))
        assert np.array_equal(diagnostics['time'], baseline_diagnostics['time']), 'Unmatched timestep sequence'
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    colors = plt.get_cmap('viridis')(np.linspace(.1, .85, len(runs)))
    report = []
    for (directory, config, analysis, data), color in zip(runs, colors):
        radius = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
        radial_velocity = (data['vx'] * data['x'] + data['vy'] * data['y'] + data['vz'] * data['z']) / radius
        reference_velocity = (data['vx_ref'] * data['x'] + data['vy_ref'] * data['y'] + data['vz_ref'] * data['z']) / radius
        dx = config['box'] / config['n']
        bins = np.arange(0, config['box'] + .5 * dx, dx)
        volume = np.histogram(radius, bins, weights=data['volume'])[0]
        centers = .5 * (bins[:-1] + bins[1:])
        def shells(values):
            return np.divide(np.histogram(radius, bins, weights=data['volume'] * values)[0], volume,
                             out=np.full_like(volume, np.nan), where=volume > 0)
        for axis, value, reference in zip(axes, (data['rho'], data['T'], radial_velocity),
                                          (data['rho_ref'], data['T_ref'], reference_velocity)):
            axis.plot(centers, shells(value), '.-', color=color, label=f"{config['n']}³ DDMC")
            if directory == runs[-1][0]:
                axis.plot(centers, shells(reference), 'k--', label='Reference (finest-grid shell averages)')
        report.append({'directory': str(directory.resolve()), 'n': config['n'], 'ranks': config['ranks'], 'dx_cm': dx, **analysis})
    for axis, label in zip(axes, ('Density [g/cm³]', 'Temperature [K]', 'Radial velocity [cm/s]')):
        axis.set_ylabel(label)
        axis.axvline(baseline['rf_end'] / 2, color='gray', linewidth=.6)
        axis.axvline(baseline['rf_end'], color='gray', linewidth=.6)
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel('Radius [cm]')
    axes[-1].set_xlim(0, baseline['box'])
    fig.suptitle(f"RMTV DDMC spatial refinement, age {runs[0][2]['physical_age_s']:.7g} s")
    fig.tight_layout()
    fig.savefig(args.output / 'convergence_profiles.png', dpi=180)
    fig.savefig(args.output / 'convergence_profiles.pdf')
    result = {'matched_settings': {field: baseline[field] for field in matched}, 'runs': report,
              'note': 'Single seed per grid; MPI rank counts are recorded separately and can change random sampling. These results do not establish a formal spatial convergence order.'}
    (args.output / 'convergence.json').write_text(json.dumps(result, indent=2) + '\n')
    for row in report:
        print(f"n={row['n']}: rho L1={row['L1_rho']:.5g}, T L1={row['L1_T']:.5g}, "
              f"v L1={row['L1_radial_velocity']:.5g}, energy drift={row['max_energy_drift']:.3g}")


if __name__ == '__main__':
    main()
