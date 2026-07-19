#!/usr/bin/env python3
"""
Checker for spherical_density_hardening_ddmc_push regression test.

Compares DDMC outputs against IMC outputs for consistency.
Checks optical-depth regime (thick low-energy, thin high-energy),
positive shell momentum, and agreement between IMC and DDMC on
radial profiles, spectra, angular leakage, and hardness ratio.
"""

import argparse
import sys
import os
import numpy as np


def load_profile(path):
    """Load radial profile, skipping comment lines."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows) if rows else np.empty((0, 0))


def load_key_value(path):
    """Load key-value text file."""
    kv = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                kv[parts[0]] = parts[1:]
    return kv


def relative_l1(a, b):
    """Compute weighted relative L1 norm: sum|a-b| / (sum|a| + sum|b| + eps)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.sum(np.abs(a)) + np.sum(np.abs(b)) + 1e-30
    return np.sum(np.abs(a - b)) / denom


def check_files_exist(run_dir, required_files):
    """Check that all required output files exist and are non-empty."""
    missing = []
    for f in required_files:
        path = os.path.join(run_dir, f)
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            missing.append(f)
    return missing


def main():
    parser = argparse.ArgumentParser(description='Spherical density-hardening DDMC push checker')
    parser.add_argument('--run-dir', required=True, help='Directory containing test outputs')
    parser.add_argument('--max-shell-vr-l1', type=float, default=0.35)
    parser.add_argument('--max-shell-momentum-rel', type=float, default=0.35)
    parser.add_argument('--max-erad-l1', type=float, default=0.35)
    parser.add_argument('--max-tgas-l1', type=float, default=0.25)
    parser.add_argument('--max-spectrum-l1', type=float, default=0.45)
    parser.add_argument('--max-hardness-rel', type=float, default=0.40)
    parser.add_argument('--max-cone-fraction-rel', type=float, default=0.45)
    parser.add_argument('--min-thick-groups', type=int, default=4)
    parser.add_argument('--min-thin-groups', type=int, default=4)
    parser.add_argument('--max-final-time-rel', type=float, default=0.02)
    args = parser.parse_args()

    run_dir = args.run_dir

    required_text = [
        'spherical_push_imc_radial_profile.txt',
        'spherical_push_ddmc_radial_profile.txt',
        'spherical_push_imc_spectrum.txt',
        'spherical_push_ddmc_spectrum.txt',
        'spherical_push_imc_angular.txt',
        'spherical_push_ddmc_angular.txt',
        'spherical_push_ddmc_diagnostics.txt',
        'spherical_push_dt_history_imc.txt',
        'spherical_push_dt_history_ddmc.txt',
        'spherical_push_summary.txt',
    ]
    required_binary = [
        'spherical_push_snapshot_init_imc.h5',
        'spherical_push_snapshot_final_imc.h5',
        'spherical_push_snapshot_init_ddmc.h5',
        'spherical_push_snapshot_final_ddmc.h5',
        'spherical_push_final_imc.h5',
        'spherical_push_final_ddmc.h5',
    ]

    all_pass = True
    results = {}

    # --- File existence ---
    missing = check_files_exist(run_dir, required_text + required_binary)
    if missing:
        for m in missing:
            print(f'FAIL: missing output file: {m}')
        all_pass = False

    missing_text = check_files_exist(run_dir, required_text)
    if missing_text:
        print('Cannot proceed with metric checks; required text files missing')
        sys.exit(1)

    # --- Load profiles ---
    prof_imc = load_profile(os.path.join(run_dir, 'spherical_push_imc_radial_profile.txt'))
    prof_ddmc = load_profile(os.path.join(run_dir, 'spherical_push_ddmc_radial_profile.txt'))

    if prof_imc.shape[0] == 0 or prof_ddmc.shape[0] == 0:
        print('FAIL: radial profiles are empty')
        sys.exit(1)

    # Join profiles on matching r_mid values
    ncols = min(prof_imc.shape[1], prof_ddmc.shape[1])
    r_imc = prof_imc[:, 0]
    r_ddmc = prof_ddmc[:, 0]

    if prof_imc.shape[0] == prof_ddmc.shape[0] and np.allclose(r_imc, r_ddmc, rtol=1e-6):
        p_imc = prof_imc[:, :ncols]
        p_ddmc = prof_ddmc[:, :ncols]
    else:
        common_r = np.intersect1d(np.round(r_imc, 8), np.round(r_ddmc, 8))
        mask_i = np.isin(np.round(r_imc, 8), common_r)
        mask_d = np.isin(np.round(r_ddmc, 8), common_r)
        p_imc = prof_imc[mask_i, :ncols]
        p_ddmc = prof_ddmc[mask_d, :ncols]
        print(f'INFO: profiles joined on {len(common_r)} common radial bins')

    if p_imc.shape[0] == 0:
        print('FAIL: no common radial bins between IMC and DDMC profiles')
        sys.exit(1)

    # Column indices (from header):
    # 0:r_mid 1:volume 2:mass 3:rho_mean 4:pressure_mean 5:vr_mass_mean
    # 6:speed_mass_mean 7:Tgas_mass_mean 8:Erad_vol_mean 9:Prad_pressure_mean
    # 10:gas_Pradial 11:tau_low 12:tau_mid 13:tau_high 14+:Eg_0..Eg_31

    # --- Shell radial velocity L1 ---
    shell_mask = (p_imc[:, 0] > 0.25) & (p_imc[:, 0] < 0.55)
    if np.any(shell_mask):
        vr_l1 = relative_l1(p_imc[shell_mask, 5], p_ddmc[shell_mask, 5])
        results['shell_vr_L1'] = vr_l1
        if vr_l1 > args.max_shell_vr_l1:
            print(f'FAIL: shell_vr_L1 = {vr_l1:.4f} > {args.max_shell_vr_l1}')
            all_pass = False
        else:
            print(f'PASS: shell_vr_L1 = {vr_l1:.4f}')
    else:
        print('FAIL: no shell bins found in radial profile (r in [0.25, 0.55])')
        all_pass = False

    # --- Erad L1 ---
    erad_l1 = relative_l1(p_imc[:, 8], p_ddmc[:, 8])
    results['Erad_L1'] = erad_l1
    if erad_l1 > args.max_erad_l1:
        print(f'FAIL: Erad_L1 = {erad_l1:.4f} > {args.max_erad_l1}')
        all_pass = False
    else:
        print(f'PASS: Erad_L1 = {erad_l1:.4f}')

    # --- Tgas L1 ---
    tgas_l1 = relative_l1(p_imc[:, 7], p_ddmc[:, 7])
    results['Tgas_L1'] = tgas_l1
    if tgas_l1 > args.max_tgas_l1:
        print(f'FAIL: Tgas_L1 = {tgas_l1:.4f} > {args.max_tgas_l1}')
        all_pass = False
    else:
        print(f'PASS: Tgas_L1 = {tgas_l1:.4f}')

    # --- Spectrum L1 ---
    spec_imc = load_profile(os.path.join(run_dir, 'spherical_push_imc_spectrum.txt'))
    spec_ddmc = load_profile(os.path.join(run_dir, 'spherical_push_ddmc_spectrum.txt'))
    if spec_imc.shape[0] == 0 or spec_ddmc.shape[0] == 0:
        print('FAIL: spectrum tables are empty')
        all_pass = False
    else:
        nspec = min(spec_imc.shape[0], spec_ddmc.shape[0])
        spec_l1 = relative_l1(spec_imc[:nspec, 4], spec_ddmc[:nspec, 4])
        results['outer_spectrum_L1'] = spec_l1
        if spec_l1 > args.max_spectrum_l1:
            print(f'FAIL: outer_spectrum_L1 = {spec_l1:.4f} > {args.max_spectrum_l1}')
            all_pass = False
        else:
            print(f'PASS: outer_spectrum_L1 = {spec_l1:.4f}')

        # Hardness = sum(high groups) / sum(low groups)
        quarter = nspec // 4
        if quarter > 0:
            h_imc = np.sum(spec_imc[nspec - quarter:nspec, 4]) / (np.sum(spec_imc[:quarter, 4]) + 1e-30)
            h_ddmc = np.sum(spec_ddmc[nspec - quarter:nspec, 4]) / (np.sum(spec_ddmc[:quarter, 4]) + 1e-30)
            h_rel = abs(h_imc - h_ddmc) / (abs(h_imc) + abs(h_ddmc) + 1e-30)
            results['hardness_rel'] = h_rel
            if h_rel > args.max_hardness_rel:
                print(f'FAIL: hardness_rel = {h_rel:.4f} > {args.max_hardness_rel}')
                all_pass = False
            else:
                print(f'PASS: hardness_rel = {h_rel:.4f}')

        # Spectrum sums must be positive
        s_imc_total = np.sum(spec_imc[:nspec, 4])
        s_ddmc_total = np.sum(spec_ddmc[:nspec, 4])
        if s_imc_total <= 0:
            print('FAIL: outer_spectrum_sum_IMC <= 0')
            all_pass = False
        if s_ddmc_total <= 0:
            print('FAIL: outer_spectrum_sum_DDMC <= 0')
            all_pass = False

    # --- Angular / cone fraction ---
    ang_imc = load_key_value(os.path.join(run_dir, 'spherical_push_imc_angular.txt'))
    ang_ddmc = load_key_value(os.path.join(run_dir, 'spherical_push_ddmc_angular.txt'))
    if 'Erad_fraction' in ang_imc and 'Erad_fraction' in ang_ddmc:
        cf_imc = float(ang_imc['Erad_fraction'][0])
        cf_ddmc = float(ang_ddmc['Erad_fraction'][0])
        cf_rel = abs(cf_imc - cf_ddmc) / (abs(cf_imc) + abs(cf_ddmc) + 1e-30)
        results['cone_fraction_rel'] = cf_rel
        if cf_rel > args.max_cone_fraction_rel:
            print(f'FAIL: cone_fraction_rel = {cf_rel:.4f} > {args.max_cone_fraction_rel}')
            all_pass = False
        else:
            print(f'PASS: cone_fraction_rel = {cf_rel:.4f}')

    # --- Shell momentum ---
    summary = load_key_value(os.path.join(run_dir, 'spherical_push_summary.txt'))
    if 'shell_momentum_imc' in summary and 'shell_momentum_ddmc' in summary:
        mom_imc = float(summary['shell_momentum_imc'][0])
        mom_ddmc = float(summary['shell_momentum_ddmc'][0])
        if mom_imc <= 0:
            print(f'FAIL: shell_radial_momentum_IMC = {mom_imc:.6e} <= 0')
            all_pass = False
        else:
            print(f'PASS: shell_radial_momentum_IMC = {mom_imc:.6e} > 0')
        if mom_ddmc <= 0:
            print(f'FAIL: shell_radial_momentum_DDMC = {mom_ddmc:.6e} <= 0')
            all_pass = False
        else:
            print(f'PASS: shell_radial_momentum_DDMC = {mom_ddmc:.6e} > 0')

        mom_rel = abs(mom_ddmc - mom_imc) / (abs(mom_imc) + 1e-30)
        results['shell_momentum_rel'] = mom_rel
        if mom_rel > args.max_shell_momentum_rel:
            print(f'FAIL: shell_momentum_rel = {mom_rel:.4f} > {args.max_shell_momentum_rel}')
            all_pass = False
        else:
            print(f'PASS: shell_momentum_rel = {mom_rel:.4f}')

    # --- DDMC regime check ---
    diag = load_profile(os.path.join(run_dir, 'spherical_push_ddmc_diagnostics.txt'))
    if diag.shape[0] == 0:
        print('FAIL: DDMC diagnostics table is empty')
        all_pass = False
    else:
        # col 2 = tau_shell_mean
        thick_groups = np.sum(diag[:, 2] > 15.0)
        thin_groups = np.sum(diag[:, 2] < 1.0)
        results['thick_low_energy_groups'] = int(thick_groups)
        results['thin_high_energy_groups'] = int(thin_groups)

        if thick_groups < args.min_thick_groups:
            print(f'FAIL: thick_low_energy_groups = {thick_groups} < {args.min_thick_groups}')
            all_pass = False
        else:
            print(f'PASS: thick_low_energy_groups = {thick_groups}')

        if thin_groups < args.min_thin_groups:
            print(f'FAIL: thin_high_energy_groups = {thin_groups} < {args.min_thin_groups}')
            all_pass = False
        else:
            print(f'PASS: thin_high_energy_groups = {thin_groups}')

    # --- Per-variant dt history validation ---
    for variant in ['imc', 'ddmc']:
        dt_path = os.path.join(run_dir, f'spherical_push_dt_history_{variant}.txt')
        dt_hist = load_profile(dt_path)
        if dt_hist.shape[0] == 0:
            print(f'FAIL: {variant} dt history is empty')
            all_pass = False
            continue
        times = dt_hist[:, 1]
        dts = dt_hist[:, 2]
        if not np.all(np.diff(times) > 0):
            print(f'FAIL: {variant} dt history time is not monotonically increasing')
            all_pass = False
        elif not np.all(dts > 0):
            print(f'FAIL: {variant} dt history has non-positive timesteps')
            all_pass = False
        else:
            print(f'PASS: {variant} dt history valid ({len(times)} cycles)')

    # --- Final time comparison from summary ---
    if 'time_imc' in summary and 'time_ddmc' in summary:
        t_imc = float(summary['time_imc'][0])
        t_ddmc = float(summary['time_ddmc'][0])
        denom = max(abs(t_imc), abs(t_ddmc), 1e-99)
        t_rel = abs(t_imc - t_ddmc) / denom
        results['final_time_rel_diff'] = t_rel
        if t_rel > args.max_final_time_rel:
            print(f'FAIL: final_time_rel_diff = {t_rel:.6f} > {args.max_final_time_rel} '
                  f'(t_imc={t_imc:.8e}, t_ddmc={t_ddmc:.8e})')
            all_pass = False
        else:
            print(f'PASS: final_time_rel_diff = {t_rel:.6f} '
                  f'(t_imc={t_imc:.8e}, t_ddmc={t_ddmc:.8e})')
    if 'cycles_imc' in summary and 'cycles_ddmc' in summary:
        print(f'INFO: IMC cycles={summary["cycles_imc"][0]}, '
              f'DDMC cycles={summary["cycles_ddmc"][0]}')

    # --- HDF5 shape checks (optional) ---
    try:
        import h5py
        for variant in ['imc', 'ddmc']:
            h5path = os.path.join(run_dir, f'spherical_push_final_{variant}.h5')
            if not os.path.isfile(h5path):
                continue
            with h5py.File(h5path, 'r') as hf:
                pts = hf['/mesh/points']
                if len(pts.shape) != 2 or pts.shape[1] != 3:
                    print(f'FAIL: {variant} H5 /mesh/points shape {pts.shape} != (N,3)')
                    all_pass = False
                N = pts.shape[0]
                rho_shape = hf['/hydro/density'].shape
                if rho_shape != (N,):
                    print(f'FAIL: {variant} H5 /hydro/density shape {rho_shape} != ({N},)')
                    all_pass = False
                eg_shape = hf['/radiation/Eg_specific'].shape
                if len(eg_shape) != 2 or eg_shape[0] != N:
                    print(f'FAIL: {variant} H5 /radiation/Eg_specific shape {eg_shape}')
                    all_pass = False
                tl_shape = hf['/opacity/tau_low'].shape
                if tl_shape != (N,):
                    print(f'FAIL: {variant} H5 /opacity/tau_low shape {tl_shape}')
                    all_pass = False
            print(f'PASS: {variant} H5 shapes verified (N={N})')
    except ImportError:
        print('INFO: h5py not available, skipping HDF5 shape checks')

    # --- Final verdict ---
    print('\n--- Results ---')
    for k, v in sorted(results.items()):
        print(f'  {k}: {v}')

    if all_pass:
        print('\nPASS: All checks passed')
        sys.exit(0)
    else:
        print('\nFAIL: Some checks failed')
        sys.exit(1)


if __name__ == '__main__':
    main()
