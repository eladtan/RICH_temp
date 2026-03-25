#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os

a_rad = 7.5657e-15  # erg cm^-3 K^-4
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

def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'output'

    final = os.path.join(output_dir, '*_final.txt')
    final_files = sorted(glob.glob(final))

    snap = os.path.join(output_dir, '*_snap*.txt')
    snap_files = sorted(glob.glob(snap))

    files = final_files if final_files else snap_files
    if not files:
        print(f"No output files found in '{output_dir}'")
        sys.exit(1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for filepath in files:
        label = os.path.basename(filepath).replace('.txt', '')
        x, T, Erad, Erad_avg = load_profile(filepath)
        Trad = (Erad_avg / a_rad) ** 0.25

        ax1.plot(x, T / kev_kelvin, label=label)
        ax2.plot(x, Trad / kev_kelvin, label=label)

    ax1.set_ylabel('T_gas (keV)')
    ax1.set_title('Desmore 2012 Step — MC Multigroup')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('x (cm)')
    ax2.set_ylabel('T_rad (keV)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'desmore_mc_profiles.png'), dpi=150)
    print(f"Saved {os.path.join(output_dir, 'desmore_mc_profiles.png')}")
    plt.show()

if __name__ == '__main__':
    main()
