#!/usr/bin/env python3
"""Plot Marshak wave comparison: MC (angle-dependent), MC (isotropic), and multigroup diffusion."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

case_dir = os.path.dirname(os.path.abspath(__file__))

kev_K = 1e3 * 1.602176634e-12 / 1.380649e-16  # 1 keV in Kelvin

def load(name):
    path = os.path.join(case_dir, name)
    if not os.path.isfile(path):
        return None
    return np.loadtxt(path)

x_mc      = load("x_pos_mc.txt")
x_iso     = load("x_pos_mc_iso.txt")
x_diff    = load("x_pos_diffusion.txt")
Tgas_mc   = load("Tgas_mc.txt")
Tgas_iso  = load("Tgas_mc_iso.txt")
Tgas_diff = load("Tgas_diffusion.txt")
Trad_mc   = load("Trad_mc.txt")
Trad_iso  = load("Trad_mc_iso.txt")
Trad_diff = load("Trad_diffusion.txt")
min_fleck = load("min_fleck.txt")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.set_title("Gas Temperature at t = 4 ns")
if Tgas_diff is not None:
    ax.plot(x_diff, Tgas_diff / kev_K, 'b-', lw=2.0, label='Diffusion')
if Tgas_mc is not None:
    ax.plot(x_mc, Tgas_mc / kev_K, 'r-', lw=2.0, alpha=0.8, label='MC (angle-dep)')
if Tgas_iso is not None:
    ax.plot(x_iso, Tgas_iso / kev_K, 'g--', lw=2.0, alpha=0.8, label='MC (isotropic)')
ax.set_xlabel("x [cm]")
ax.set_ylabel("T_gas [keV]")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.set_title("Radiation Temperature at t = 4 ns")
if Trad_diff is not None:
    ax.plot(x_diff, Trad_diff / kev_K, 'b-', lw=1.5, label='Diffusion')
if Trad_mc is not None:
    ax.plot(x_mc, Trad_mc / kev_K, 'r-', lw=1.0, alpha=0.8, label='MC (angle-dep)')
if Trad_iso is not None:
    ax.plot(x_iso, Trad_iso / kev_K, 'g--', lw=1.0, alpha=0.8, label='MC (isotropic)')
ax.set_xlabel("x [cm]")
ax.set_ylabel("T_rad [keV]")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(case_dir, "marshak_comparison.png")
plt.savefig(out_path, dpi=150)
print(f"Saved {out_path}")

if min_fleck is not None:
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(np.arange(len(min_fleck)), min_fleck, 'k-', lw=0.8)
    ax2.set_xlabel("Time step index")
    ax2.set_ylabel("Min Fleck factor")
    ax2.set_title("Minimum Fleck Factor vs Time Step")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fleck_path = os.path.join(case_dir, "min_fleck.png")
    plt.savefig(fleck_path, dpi=150)
    print(f"Saved {fleck_path}")
