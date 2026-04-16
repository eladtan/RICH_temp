#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import glob, os

a_rad = 7.5657e-15
kev_kelvin = 1e3 * 1.602176634e-12 / 1.380649e-16

def load_profile(filename):
    x, T, Erad, Erad_avg = [], [], [], []
    with open(filename) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(',')
            if len(parts) >= 3:
                x.append(float(parts[0]))
                T.append(float(parts[1]))
                Erad.append(float(parts[2]))
                Erad_avg.append(float(parts[3]) if len(parts) >= 4 else float(parts[2]))
    return np.array(x), np.array(T), np.array(Erad), np.array(Erad_avg)

rw_final = 'desmore_step_mc_rw_final.txt'
norw_final = 'desmore_step_mc_norw_final.txt'

x_rw, T_rw, Erad_rw, Erad_avg_rw = load_profile(rw_final)
x_norw, T_norw, Erad_norw, Erad_avg_norw = load_profile(norw_final)

Trad_rw = (Erad_avg_rw / a_rad) ** 0.25
Trad_norw = (Erad_avg_norw / a_rad) ** 0.25

snap_times = sorted(set(
    [f for f in glob.glob('desmore_step_mc_rw_snap*.txt')]
))
norw_snaps = sorted(glob.glob('desmore_step_mc_norw_snap*.txt'))

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# --- Top row: T_gas profiles ---
ax = axes[0, 0]
for sf in snap_times:
    idx = os.path.basename(sf).replace('desmore_step_mc_rw_snap', '').replace('.txt', '')
    x_s, T_s, _, _ = load_profile(sf)
    ax.plot(x_s, T_s / kev_kelvin, alpha=0.4, linewidth=0.8, label=f'snap {idx}')
ax.plot(x_rw, T_rw / kev_kelvin, 'b-', linewidth=2, label='RW final')
ax.plot(x_norw, T_norw / kev_kelvin, 'r--', linewidth=2, label='no-RW final')
ax.set_ylabel('T_gas (keV)')
ax.set_title('Gas Temperature')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for sf in norw_snaps:
    idx = os.path.basename(sf).replace('desmore_step_mc_norw_snap', '').replace('.txt', '')
    x_s, T_s, _, _ = load_profile(sf)
    ax.plot(x_s, T_s / kev_kelvin, alpha=0.4, linewidth=0.8, label=f'snap {idx}')
ax.plot(x_norw, T_norw / kev_kelvin, 'r-', linewidth=2, label='no-RW final')
ax.plot(x_rw, T_rw / kev_kelvin, 'b--', linewidth=2, label='RW final')
ax.set_ylabel('T_gas (keV)')
ax.set_title('Gas Temperature (no-RW snapshots)')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# --- Middle row: T_rad profiles ---
ax = axes[1, 0]
ax.plot(x_rw, Trad_rw / kev_kelvin, 'b-', linewidth=2, label='RW')
ax.plot(x_norw, Trad_norw / kev_kelvin, 'r--', linewidth=2, label='no-RW')
ax.set_ylabel('T_rad (keV)')
ax.set_title('Radiation Temperature (from time-avg Erad)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(x_rw, Erad_rw, 'b-', linewidth=2, label='RW')
ax.plot(x_norw, Erad_norw, 'r--', linewidth=2, label='no-RW')
ax.set_ylabel('Erad (erg/cm³)')
ax.set_title('Instantaneous Erad')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# --- Bottom row: relative differences ---
ax = axes[2, 0]
T_ref = np.maximum(T_norw, 1.0)
rel_T = (T_rw - T_norw) / T_ref * 100
ax.plot(x_rw, rel_T, 'k-', linewidth=1.5)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('x (cm)')
ax.set_ylabel('Relative diff (%)')
ax.set_title('T_gas relative difference (RW − no-RW) / no-RW')
ax.grid(True, alpha=0.3)
ax.set_ylim(-30, 30)

ax = axes[2, 1]
Trad_ref = np.maximum(Trad_norw, 1.0)
rel_Trad = (Trad_rw - Trad_norw) / Trad_ref * 100
ax.plot(x_rw, rel_Trad, 'k-', linewidth=1.5)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('x (cm)')
ax.set_ylabel('Relative diff (%)')
ax.set_title('T_rad relative difference (RW − no-RW) / no-RW')
ax.grid(True, alpha=0.3)
ax.set_ylim(-30, 30)

fig.suptitle('Desmore 2012 Step — RW vs no-RW comparison\n'
             'C=5: RW 53.1s, no-RW 398.6s (7.5x speedup)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('rw_comparison.png', dpi=150)
print("Saved rw_comparison.png")
